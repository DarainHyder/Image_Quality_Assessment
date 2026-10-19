"""Lumina IQA — inference API (FastAPI + onnxruntime, CPU).

Endpoints
  GET  /          service + model info
  GET  /health    liveness
  POST /analyze   full read-out: score, percentile, quality & attention maps, defects, attributes, forensics
  POST /predict   v1-compatible ({"quality_score", "status"}) plus the v2 score fields
"""
from __future__ import annotations

import io
import json
import os
import time
from bisect import bisect_left
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from lumina.analysis import analyze

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("LUMINA_MODEL_DIR", ROOT / "artifacts"))
HF_REPO = os.getenv("LUMINA_HF_REPO", "sawabedarain/lumina-iqa")
MODEL_FILE = os.getenv("LUMINA_MODEL_FILE", "lumina_v2.int8.onnx")
MAX_BYTES = int(os.getenv("LUMINA_MAX_MB", "20")) * 1024 * 1024
LONG, SHORT = 512, 384  # training canvas
Image.MAX_IMAGE_PIXELS = 90_000_000

RATINGS = [(80, "excellent"), (65, "good"), (45, "fair"), (25, "poor"), (0, "bad")]


def _ensure_model() -> tuple[Path, Path]:
    onnx_path, card_path = MODEL_DIR / MODEL_FILE, MODEL_DIR / "lumina_v2.json"
    if not onnx_path.exists() or not card_path.exists():
        from huggingface_hub import hf_hub_download
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for f in (MODEL_FILE, "lumina_v2.json"):
            hf_hub_download(HF_REPO, f, local_dir=MODEL_DIR)
    return onnx_path, card_path


class Engine:
    def __init__(self):
        onnx_path, card_path = _ensure_model()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(os.getenv("LUMINA_THREADS", "0"))
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.card = json.load(open(card_path))
        self.quantiles = self.card.get("koniq_quantiles") or []
        self.name = onnx_path.name
        self.sess.run(None, {"image": np.zeros((1, 3, SHORT, LONG), np.float32)})  # warm-up

    @staticmethod
    def prepare(img: Image.Image) -> np.ndarray:
        w, h = img.size
        portrait = h > w
        lw, sw = (h, w) if portrait else (w, h)
        s = min(1.0, LONG / lw, SHORT / sw)  # never upscale: that would invent blur
        tw, th = max(64, round(w * s / 32) * 32), max(64, round(h * s / 32) * 32)
        if (tw, th) != (w, h):
            img = img.resize((tw, th), Image.LANCZOS)
        return np.asarray(img, np.float32).transpose(2, 0, 1)[None]

    def percentile(self, mos: float) -> float | None:
        if not self.quantiles:
            return None
        i = bisect_left(self.quantiles, mos)
        return float(np.clip(i, 0, 100))

    def __call__(self, img: Image.Image) -> dict:
        x = self.prepare(img)
        t = time.perf_counter()
        mos, qmap, wmap, defects, dmap, attrs = self.sess.run(None, {"image": x})
        ms = (time.perf_counter() - t) * 1000
        score = float(np.clip(mos[0], 0, 100))
        rating = next(label for th, label in RATINGS if score >= th)
        r = lambda a, n=2: np.round(a.astype(np.float64), n).tolist()  # noqa: E731
        return {
            "score": round(score, 2),
            "rating": rating,
            "percentile": self.percentile(score),
            "defects": {k: round(float(v), 4) for k, v in zip(self.card["defects"], defects[0])},
            "attributes": {k: round(float(v), 2) for k, v in zip(self.card["attrs"], attrs[0])},
            "maps": {
                "shape": list(qmap.shape[1:]),
                "quality": r(qmap[0], 1),
                "attention": r(wmap[0] / wmap[0].max(), 3),
                "defects": {k: r(dmap[0, i], 3) for i, k in enumerate(self.card["defects"])},
            },
            "input_size": [int(x.shape[3]), int(x.shape[2])],
            "inference_ms": round(ms, 1),
        }


engine: Engine | None = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(_app):
    global engine, load_error
    try:
        engine = Engine()
    except Exception as e:  # noqa: BLE001 - surface the reason on / instead of crash-looping
        load_error = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="Lumina IQA", version="2.0.0", lifespan=lifespan,
              description="Perceptual no-reference image quality assessment with spatial explanations.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _read(file: UploadFile) -> tuple[Image.Image, Image.Image]:
    if engine is None:
        raise HTTPException(503, f"Model not loaded: {load_error}")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        original = Image.open(io.BytesIO(data))
        original.load()
        img = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"Unreadable image: {e}") from e
    return img, original


@app.get("/")
def root():
    return {"status": "online" if engine else "degraded", "model_loaded": engine is not None,
            "model": engine.name if engine else None, "version": app.version, "error": load_error,
            "metrics": engine.card.get("metrics") if engine else None,
            "info": "Lumina IQA v2 — ConvNeXt-B (CLIP) multi-scale quality/attention/defect model"}


@app.get("/health")
def health():
    return {"ok": engine is not None}


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    img, original = _read(file)
    out = engine(img)
    out["technical"] = analyze(img, original)
    out["filename"] = file.filename
    return out


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img, _ = _read(file)
    out = engine(img)
    return {"quality_score": out["score"], "rating": out["rating"], "percentile": out["percentile"],
            "defects": out["defects"], "status": "success"}
