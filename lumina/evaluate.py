"""Evaluation: intra-dataset (KonIQ test), cross-dataset (LIVE Challenge, SPAQ) and the v1 baseline.

    python -m lumina.evaluate --ckpt runs/lumina_v2/best.pt --data /path/to/prepared --out reports/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import IQADataset, Norm, eval_collate, load_split
from .metrics import iqa_metrics
from .model import LuminaNet


@torch.inference_mode()
def predict_dataset(model, ds: IQADataset, dev, norm: Norm, tta: bool = False, bs: int = 32,
                    workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted MOS on the KonIQ scale, ground-truth MOS) for every image in ``ds``."""
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=eval_collate)
    preds = np.zeros(len(ds), np.float64)
    for b in dl:
        groups = defaultdict(list)  # batch images by shape, native resolution is preserved
        for j, im in enumerate(b["img"]):
            groups[tuple(im.shape)].append(j)
        for _, js in groups.items():
            x = torch.stack([b["img"][j] for j in js]).to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                s = model(x)["score"]
                if tta:
                    s = 0.5 * (s + model(torch.flip(x, dims=[3]))["score"])
            for j, v in zip(js, s.float().cpu().numpy()):
                preds[int(b["idx"][j])] = v
    mos = ds.df["mos"].to_numpy(np.float64)
    return preds * norm.mos_std + norm.mos_mean, mos


class LegacyV1(torch.nn.Module):
    """The original Lumina v1 model (EfficientNet-B0 regressor, 224x224 squashed input)."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.efficientnet_b0(weights=None)
        n = base.classifier[1].in_features
        base.classifier = torch.nn.Identity()
        self.base = base
        self.regressor = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(n, 1))
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = F.interpolate(x.float(), size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        return {"score": self.regressor(self.base((x - self.m) / self.s)).squeeze(1)}


def load_lumina(ckpt: Path, dev) -> tuple[LuminaNet, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LuminaNet(ck["meta"]["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), ck["meta"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--legacy", type=Path, default=None, help="v1 EfficientNet-B0 .pth for baseline comparison")
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_lumina(args.ckpt, dev)
    kn = Norm(**meta["norms"]["koniq"])

    suites = {"koniq_test": ("koniq.csv", "test", True), "livec": ("livec.csv", None, False),
              "spaq_test": ("spaq.csv", "test", True), "kadid_test": ("kadid.csv", "test", True)}
    results = {}
    for name, (csv, split, canvas) in suites.items():
        if not (args.data / csv).exists():
            continue
        df = load_split(args.data / csv, split).dropna(subset=["mos"]).reset_index(drop=True)
        ds = IQADataset(df, args.data, kn, train=False, canvas=canvas)
        res = {}
        for tag, tta in (("lumina_v2", False), ("lumina_v2_tta", True)):
            p, y = predict_dataset(model, ds, dev, kn, tta=tta, bs=16 if not canvas else 32)
            res[tag] = iqa_metrics(p, y)
        if name == "koniq_test":  # predicted vs human MOS for the frontend scatter plot (TTA predictions)
            args.out.mkdir(parents=True, exist_ok=True)
            json.dump({"pred": np.round(p, 1).tolist(), "mos": np.round(y, 1).tolist()}, open(args.out / "scatter.json", "w"))
        results[name] = res
        print(name, json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}), flush=True)

    if args.legacy and args.legacy.exists():
        v1 = LegacyV1()
        sd = torch.load(args.legacy, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        v1.load_state_dict(sd, strict=False)
        v1 = v1.to(dev).eval()
        for name, (csv, split, canvas) in suites.items():
            if name not in results:
                continue
            df = load_split(args.data / csv, split).dropna(subset=["mos"]).reset_index(drop=True)
            ds = IQADataset(df, args.data, Norm.identity(), train=False, canvas=canvas)
            p, y = predict_dataset(v1, ds, dev, Norm.identity(), bs=16)
            results[name]["lumina_v1"] = iqa_metrics(p, y)
            print(name, "v1", {m: round(v, 4) for m, v in results[name]["lumina_v1"].items()}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
