"""Datasets and batch collation.

Every source is described by a prepared CSV (see ``scripts/prepare_data.py``) with columns

    path, mos, split, d_<defect>..., a_<attr>...

Unknown labels are NaN and are masked out of the corresponding loss.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import ATTRS, CANVAS, DEFECTS
from .synth import degrade

D_COLS = [f"d_{k}" for k in DEFECTS]
A_COLS = [f"a_{k}" for k in ATTRS]


@dataclass
class Norm:
    """z-score statistics for a source's MOS and attributes (computed on its train split)."""
    mos_mean: float
    mos_std: float
    attr_mean: list[float]
    attr_std: list[float]

    @classmethod
    def identity(cls) -> "Norm":
        return cls(0.0, 1.0, [50.0] * len(ATTRS), [20.0] * len(ATTRS))

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Norm":
        am, asd = [], []
        for c in A_COLS:
            v = df[c].dropna() if c in df else pd.Series(dtype=float)
            am.append(float(v.mean()) if len(v) else 50.0)
            asd.append(float(v.std()) if len(v) > 1 else 20.0)
        return cls(float(df["mos"].mean()), float(df["mos"].std()), am, asd)


def load_split(csv: Path, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv)
    for c in D_COLS + A_COLS:
        if c not in df:
            df[c] = np.nan
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def to_canvas(img: Image.Image, size=CANVAS) -> Image.Image:
    """Landscape-orient and cover-resize to the training canvas (no distortion of aspect)."""
    W, H = size
    if img.height > img.width:
        img = img.transpose(Image.Transpose.ROTATE_90)
    if img.size == (W, H):
        return img
    s = max(W / img.width, H / img.height)
    nw, nh = max(W, round(img.width * s)), max(H, round(img.height * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - W) // 2, (nh - H) // 2
    return img.crop((l, t, l + W, t + H))


class IQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, norm: Norm, train: bool, canvas: bool = True):
        self.df, self.root, self.norm, self.train, self.canvas = df, Path(root), norm, train, canvas
        self.z = ((df["mos"] - norm.mos_mean) / norm.mos_std).to_numpy(np.float32)
        self.d = df[D_COLS].to_numpy(np.float32)
        a = df[A_COLS].to_numpy(np.float32)
        self.a = (a - np.array(norm.attr_mean, np.float32)) / np.array(norm.attr_std, np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.root / self.df.at[i, "path"]).convert("RGB")
        if self.canvas:
            img = to_canvas(img)
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {"img": img, "z": self.z[i], "d": self.d[i], "a": self.a[i], "idx": i}


def _arr(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)


class Collate:
    """Stacks samples; optionally appends synthetically degraded copies of a fraction of the batch."""

    def __init__(self, synth_frac: float = 0.0):
        self.synth_frac = synth_frac

    def __call__(self, items: list[dict]) -> dict:
        imgs = [_arr(it["img"]) for it in items]
        z = [float(it["z"]) for it in items]
        d = [it["d"] for it in items]
        a = [it["a"] for it in items]
        pairs = []
        n_syn = int(round(len(items) * self.synth_frac))
        if n_syn:
            rng = random.Random(random.getrandbits(32))
            for j in rng.sample(range(len(items)), n_syn):
                out, k, sev = degrade(items[j]["img"], rng)
                dt = np.full(len(DEFECTS), np.nan, np.float32)
                dt[k] = sev
                pairs.append((j, len(imgs), k, sev))
                imgs.append(_arr(out))
                z.append(float("nan"))
                d.append(dt)
                a.append(np.full(len(ATTRS), np.nan, np.float32))
        batch = {
            "img": torch.stack(imgs),
            "z": torch.tensor(z, dtype=torch.float32),
            "d": torch.from_numpy(np.stack(d)),
            "a": torch.from_numpy(np.stack(a)),
            "idx": torch.tensor([it["idx"] for it in items]),
        }
        p = np.array(pairs, dtype=np.float32).reshape(-1, 4)
        batch["pair_o"] = torch.from_numpy(p[:, 0]).long()
        batch["pair_d"] = torch.from_numpy(p[:, 1]).long()
        batch["pair_k"] = torch.from_numpy(p[:, 2]).long()
        batch["pair_sev"] = torch.from_numpy(p[:, 3])
        return batch


def eval_collate(items: list[dict]) -> dict:
    return {"img": [_arr(it["img"]) for it in items], "z": torch.tensor([float(it["z"]) for it in items]),
            "idx": torch.tensor([it["idx"] for it in items])}
