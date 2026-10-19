"""Export a trained checkpoint to ONNX for CPU serving (onnxruntime, no torch at runtime).

    python -m lumina.export --ckpt runs/lumina_v2/best.pt --out export/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .constants import ATTRS, CANVAS, DEFECTS
from .evaluate import load_lumina
from .model import ExportWrapper

OUTPUTS = ["mos", "qmap", "wmap", "defects", "dmap", "attrs"]


def _headline(metrics: dict) -> dict:
    """Compact SRCC/PLCC per test suite (TTA variant when available)."""
    out = {}
    for suite, res in metrics.items():
        m = res.get("lumina_v2_tta") or res.get("lumina_v2")
        out[suite] = {"srcc": round(m["srcc"], 4), "plcc": round(m["plcc"], 4), "n": m["n"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--quantize", action="store_true", help="also write an int8 dynamically-quantised model")
    ap.add_argument("--metrics", type=Path, default=None, help="reports/metrics.json to embed in the model card")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    model, meta = load_lumina(args.ckpt, torch.device("cpu"))
    kn, an = meta["norms"]["koniq"], meta["attr_norm"]
    wrap = ExportWrapper(model, kn["mos_mean"], kn["mos_std"], an["attr_mean"], an["attr_std"]).eval()
    x = torch.randint(0, 256, (1, 3, CANVAS[1], CANVAS[0])).float()
    onnx_path = args.out / "lumina_v2.onnx"
    torch.onnx.export(
        wrap, (x,), str(onnx_path), input_names=["image"], output_names=OUTPUTS, opset_version=18,
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}} | {o: {0: "batch"} for o in OUTPUTS},
        dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = [t.numpy() for t in wrap(x)]
    got = sess.run(None, {"image": x.numpy()})
    err = {n: float(np.abs(a - b).max()) for n, a, b in zip(OUTPUTS, ref, got)}
    print("onnx parity (max abs err):", err)
    assert err["mos"] < 1e-2, err

    for shape in [(1, 3, 384, 512), (1, 3, 512, 384), (1, 3, 448, 448)]:  # dynamic-shape sanity
        sess.run(None, {"image": np.random.randint(0, 256, shape).astype(np.float32)})

    files = [onnx_path]
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q_path = args.out / "lumina_v2.int8.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        files.append(q_path)

    for f in files:
        s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        s.run(None, {"image": x.numpy()})
        t = time.time()
        for _ in range(5):
            s.run(None, {"image": x.numpy()})
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB, {(time.time() - t) / 5 * 1000:.0f} ms / 512x384 image")

    card = {"version": "2.0.0", "backbone": meta["backbone"], "canvas": CANVAS, "defects": DEFECTS, "attrs": ATTRS,
            "koniq_norm": kn, "attr_norm": an, "koniq_quantiles": meta.get("koniq_quantiles"),
            "metrics": _headline(json.load(open(args.metrics))) if args.metrics else None}
    json.dump(card, open(args.out / "lumina_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
