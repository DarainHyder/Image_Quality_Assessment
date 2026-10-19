"""Publish exported weights to the HF Hub model repo and sync the inference Space.

    HF_TOKEN=... python scripts/publish_hf.py --export export/ --reports reports/

* model repo  (default sawabedarain/lumina-iqa): ONNX weights, model card, metrics
* Space       (default sawabedarain/Lumina-IQA): Dockerfile + server bundle (weights are pulled at build)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Lumina IQA
emoji: 📸
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Perceptual image quality with spatial explanations
---

# Lumina IQA — inference API

CPU inference service for [Lumina v2](https://huggingface.co/{model}), a no-reference image quality model.
Front-end: <https://lumina-iqa.vercel.app> · Source: <https://github.com/DarainHyder/Image_Quality_Assessment>

`POST /analyze` (multipart `file`) → score, percentile, quality/attention/defect maps, SPAQ attributes, EXIF & forensics.
`POST /predict` → v1-compatible `quality_score`. Interactive docs at `/docs`.
"""


def model_card(model: str, metrics: dict, meta: dict) -> str:
    def row(name, key):
        m = metrics.get(key, {})
        v2, v1 = m.get("lumina_v2_tta", m.get("lumina_v2")), m.get("lumina_v1")
        if not v2:
            return ""
        f = lambda x: f"{x:.3f}" if x is not None else "—"  # noqa: E731
        return (f"| {name} | {f(v2['srcc'])} | {f(v2['plcc'])} | "
                f"{f(v1['srcc']) if v1 else '—'} | {f(v1['plcc']) if v1 else '—'} |\n")

    table = ("| Test set | v2 SRCC | v2 PLCC | v1 SRCC | v1 PLCC |\n|---|---|---|---|---|\n"
             + row("KonIQ-10k (official test, 2,015)", "koniq_test")
             + row("LIVE Challenge (zero-shot, 1,169)", "livec")
             + row("SPAQ (held-out 20%, 2,225)", "spaq_test")
             + row("KADID-10k (held-out reference scenes)", "kadid_test"))
    return f"""---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags: [image-quality-assessment, iqa, nr-iqa, convnext, onnx, koniq-10k, spaq, kadid-10k]
datasets: [koniq-10k, spaq, kadid-10k]
---

# Lumina v2 — perceptual image quality with spatial explanations

No-reference image quality assessment (NR-IQA). Given a photograph, Lumina predicts the mean opinion score (0–100,
KonIQ-10k scale) a panel of human viewers would give it, and explains the score:

* **quality map**: local quality for every 16×16 region
* **attention map**: where viewers look; the score is the attention-weighted mean of the quality map
* **six defect maps + scores**: blur, noise, artifacts, exposure, contrast, colour
* **five perceptual attributes** (SPAQ-calibrated): sharpness, noise-free, contrast, colourfulness, brightness

## Architecture
CLIP-pretrained ConvNeXt-Base (`{meta.get('backbone')}`), stride-8/16/32 features fused on the 1/16 grid, four light
heads. 92M parameters. Fully convolutional: images are scored at native aspect (long side ≤ 512 px, short ≤ 384 px).

## Training
Joint multi-source training on an RTX 5090 (bf16, EMA, AdamW, cosine schedule, 40 epochs):
KonIQ-10k MOS (official 7,058 train split) · KonIQ++ defect votes · SPAQ MOS + attributes (8,900) ·
KADID-10k MOS + distortion families (8,190, reference-disjoint split) · on-the-fly synthetic degradations with a
self-supervised ranking constraint (degraded copy must score below its original).
Losses: smooth-L1 + PLCC + pairwise rank on MOS, masked BCE on defects, L1 on attributes; per-dataset affine calibration.

## Results
{table}
v1 = the original EfficientNet-B0 regressor. It was trained on a random split of all of KonIQ, so its KonIQ test number
is inflated by train/test overlap; LIVE-C and SPAQ are the fair comparisons. v2 numbers use horizontal-flip TTA.

## Files
* `lumina_v2.int8.onnx`: dynamic int8 (MatMul) for CPU serving
* `lumina_v2.onnx`: fp32
* `lumina_v2.json`: normalisation, KonIQ score quantiles (for percentiles), label names
* `metrics.json`, `history.json`: evaluation and per-epoch validation

Input: `image` float32 RGB in [0, 255], shape (N, 3, H, W). Outputs: `mos`, `qmap`, `wmap`, `defects`, `dmap`, `attrs`.

Source & training code: https://github.com/DarainHyder/Image_Quality_Assessment
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--model-repo", default="sawabedarain/lumina-iqa")
    ap.add_argument("--space", default="sawabedarain/Lumina-IQA")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-space", action="store_true")
    ap.add_argument("--with-fp32", action="store_true", help="also upload the 369 MB fp32 ONNX")
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])

    if not a.skip_model:
        metrics = json.load(open(a.reports / "metrics.json"))
        meta = json.load(open(a.export / "lumina_v2.json"))
        api.create_repo(a.model_repo, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for f in ("lumina_v2.int8.onnx", "lumina_v2.json") + (("lumina_v2.onnx",) if a.with_fp32 else ()):
                shutil.copy(a.export / f, td / f)
            for f in ("metrics.json", "history.json"):
                shutil.copy(a.reports / f, td / f)
            (td / "README.md").write_text(model_card(a.model_repo, metrics, meta))
            api.upload_folder(repo_id=a.model_repo, repo_type="model", folder_path=td,
                              commit_message="Lumina v2: ONNX weights, model card, metrics")
        print("model repo updated:", a.model_repo)

    if not a.skip_space:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for f in ("Dockerfile", "requirements.txt", ".dockerignore"):
                shutil.copy(ROOT / f, td / f)
            (td / "lumina").mkdir()
            for f in ("__init__.py", "constants.py", "analysis.py"):
                shutil.copy(ROOT / "lumina" / f, td / "lumina" / f)
            shutil.copytree(ROOT / "server", td / "server", ignore=shutil.ignore_patterns("__pycache__"))
            (td / "README.md").write_text(SPACE_README.format(model=a.model_repo))
            api.upload_folder(repo_id=a.space, repo_type="space", folder_path=td, delete_patterns=["*"],
                              commit_message="Lumina v2: ONNX Runtime inference service")
        print("space updated:", a.space)


if __name__ == "__main__":
    main()
