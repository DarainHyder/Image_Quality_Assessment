"""Multi-source training for LuminaNet.

    python -m lumina.train --data /path/to/prepared --out runs/lumina_v2

Each optimisation step draws a single-source batch:
  koniq  MOS (L1 + PLCC + rank)  ·  KonIQ++ defects  ·  synthetic degradations (defects + rank constraint)
  kadid  calibrated MOS          ·  synthetic-distortion defects with known family / level
  spaq   calibrated MOS          ·  human-rated perceptual attributes
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader

from .constants import BACKBONE, DEFECTS, SOURCES
from .data import Collate, IQADataset, Norm, load_split
from .evaluate import predict_dataset
from .losses import degradation_rank, masked_bce, masked_l1, plcc_loss, rank_loss
from .metrics import iqa_metrics
from .model import LuminaNet

SRC = {s: i for i, s in enumerate(SOURCES)}
COLOR = DEFECTS.index("color")


def infinite(loader):
    while True:
        yield from loader


def mos_losses(score, z):
    m = ~torch.isnan(z)
    p, y = score[m], z[m]
    return masked_l1(p, y), plcc_loss(p, y), rank_loss(p, y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="directory with prepared *.csv files")
    ap.add_argument("--out", type=Path, default=Path("runs/lumina_v2"))
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=24, help="KonIQ originals per step (synthetic copies are added)")
    ap.add_argument("--aux-bs", type=int, default=32)
    ap.add_argument("--synth-frac", type=float, default=1 / 3)
    ap.add_argument("--kadid-ratio", type=float, default=0.35, help="KADID steps per KonIQ step")
    ap.add_argument("--spaq-ratio", type=float, default=0.5, help="SPAQ steps per KonIQ step")
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup epochs")
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap steps per epoch")
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    # ---------------- data
    root = args.data
    k_tr, k_va = load_split(root / "koniq.csv", "train"), load_split(root / "koniq.csv", "val")
    norms = {"koniq": Norm.from_frame(k_tr)}
    loaders = {}
    mk = lambda ds, bs, col: DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers,
                                        collate_fn=col, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    loaders["koniq"] = mk(IQADataset(k_tr, root, norms["koniq"], train=True), args.bs, Collate(args.synth_frac))
    aux = {}
    for name, ratio in (("kadid", args.kadid_ratio), ("spaq", args.spaq_ratio)):
        csv = root / f"{name}.csv"
        if ratio > 0 and csv.exists():
            tr = load_split(csv, "train")
            norms[name] = Norm.from_frame(tr)
            aux[name] = (infinite(mk(IQADataset(tr, root, norms[name], train=True), args.aux_bs, Collate(0.0))), ratio)
            say(f"[data] {name}: {len(tr)} train images, ratio {ratio}")
    say(f"[data] koniq: {len(k_tr)} train / {len(k_va)} val")
    val_ds = IQADataset(k_va, root, norms["koniq"], train=False)
    attr_norm = norms.get("spaq", norms["koniq"])

    # ---------------- model
    model = LuminaNet(args.backbone, pretrained=True, drop_path=args.drop_path).to(dev).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True)
    opt = torch.optim.AdamW(model.param_groups(args.lr_backbone, args.lr_head, args.wd), betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    fwd = torch.compile(model) if args.compile else model

    steps_per_epoch = len(loaders["koniq"]) if not args.limit else min(args.limit, len(loaders["koniq"]))
    total = args.epochs * steps_per_epoch
    warm = int(args.warmup * steps_per_epoch)

    def set_lr(step):
        f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * max(f, 0.01 if step >= warm else f)

    def run_step(batch, src):
        x = batch["img"].to(dev, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
        z, d, a = (batch[k].to(dev, non_blocking=True) for k in ("z", "d", "a"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = fwd(x)
        score = o["score"] if src == "koniq" else model.calibrated(o["score"], SRC[src])
        l1, lp, lr_ = mos_losses(score, z)
        parts = {"l1": l1, "plcc": lp, "rank": lr_, "def": masked_bce(o["defects"], d)}
        w_src = 1.0 if src == "koniq" else 0.5
        loss = w_src * (l1 + 0.5 * lp + 0.5 * lr_) + 0.5 * parts["def"]
        if src == "koniq" and batch["pair_o"].numel():
            keep = batch["pair_k"].to(dev) != COLOR
            parts["syn"] = degradation_rank(o["score"], batch["pair_o"].to(dev), batch["pair_d"].to(dev),
                                            batch["pair_sev"].to(dev), keep)
            loss = loss + 0.5 * parts["syn"]
        if src == "spaq":
            parts["attr"] = masked_l1(o["attrs"], a)
            loss = loss + 0.5 * parts["attr"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        ema.update(model)
        return {k: v.item() for k, v in parts.items()} | {"loss": loss.item()}

    history, best, step = [], -1.0, 0
    meta = {"backbone": args.backbone, "norms": {k: asdict(v) for k, v in norms.items()},
            "attr_norm": asdict(attr_norm),
            "koniq_quantiles": np.quantile(k_tr["mos"], np.linspace(0, 1, 101)).round(3).tolist(), "args": {k: str(v) for k, v in vars(args).items()}}
    say(f"[train] {steps_per_epoch} KonIQ steps/epoch x {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, agg, n_img = time.time(), {}, 0
        for i, batch in enumerate(loaders["koniq"]):
            if args.limit and i >= args.limit:
                break
            set_lr(step); step += 1
            jobs = [("koniq", batch)]
            for name, (it, ratio) in aux.items():
                if random.random() < ratio:
                    jobs.append((name, next(it)))
            for src, b in jobs:
                for k, v in run_step(b, src).items():
                    agg.setdefault(f"{src}/{k}", []).append(v)
                n_img += b["img"].shape[0]
        train_stats = {k: float(np.mean(v)) for k, v in agg.items()}
        ema_model = ema.module.eval()
        pred, mos = predict_dataset(ema_model, val_ds, dev, norms["koniq"])
        val = iqa_metrics(pred, mos)
        dt = time.time() - t0
        rec = {"epoch": epoch, "time_s": dt, "img_per_s": n_img / dt, "val": val, "train": train_stats,
               "lr_backbone": opt.param_groups[0]["lr"]}
        history.append(rec)
        crit = val["srcc"] + val["plcc"]
        tag = ""
        if crit > best:
            best = crit
            torch.save({"model": ema_model.state_dict(), "meta": meta, "epoch": epoch, "val": val}, args.out / "best.pt")
            tag = "  *best*"
        say(f"[ep {epoch:02d}] {dt:.0f}s {n_img / dt:.0f} img/s  loss {train_stats.get('koniq/loss', 0):.3f}  "
            f"val SRCC {val['srcc']:.4f} PLCC {val['plcc']:.4f} RMSE {val['rmse']:.2f}{tag}")
        json.dump(history, open(args.out / "history.json", "w"), indent=1)
    torch.save({"model": ema.module.state_dict(), "meta": meta, "epoch": args.epochs}, args.out / "last.pt")
    say(f"[done] best val SRCC+PLCC = {best:.4f}")


if __name__ == "__main__":
    main()
