"""On-the-fly synthetic degradations with known family + severity.

Used two ways during training:
  * defect supervision on diverse in-the-wild content (not only KADID's 81 scenes);
  * a self-supervised ranking constraint — a strongly degraded copy must score
    below its pristine original.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .constants import DEFECTS

LEVELS = 5


def _blur(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur([0.7, 1.3, 2.2, 3.5, 5.5][lv]))
    # directional (motion) blur: average of shifted copies along a random angle
    a = np.asarray(img, dtype=np.float32)
    n = [3, 5, 9, 13, 19][lv]
    th = rng.uniform(0, np.pi)
    acc = np.zeros_like(a)
    for t in np.linspace(-(n - 1) / 2, (n - 1) / 2, n):
        dx, dy = int(round(t * np.cos(th))), int(round(t * np.sin(th)))
        acc += np.roll(a, (dy, dx), axis=(0, 1))
    return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8))


def _noise(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    sig = [5, 9, 15, 24, 38][lv]
    g = np.random.default_rng(rng.randrange(1 << 30))
    if rng.random() < 0.5:   # luma-dominant sensor noise
        n = g.normal(0, sig, a.shape[:2])[..., None] + g.normal(0, sig * 0.35, a.shape)
    else:                    # chroma / colour noise
        n = g.normal(0, sig, a.shape)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def _artifacts(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    if rng.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=[30, 18, 11, 7, 4][lv])
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    f = [1.6, 2.2, 3.2, 4.5, 6.5][lv]
    w, h = img.size
    small = img.resize((max(8, int(w / f)), max(8, int(h / f))), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def _exposure(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    k = [1.35, 1.7, 2.1, 2.7, 3.4][lv]
    a = np.clip(a * k, 0, 1) if rng.random() < 0.5 else np.power(a, k * 0.85)
    return Image.fromarray((a * 255).astype(np.uint8))


def _contrast(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance([0.72, 0.55, 0.4, 0.28, 0.16][lv])


def _color(img: Image.Image, lv: int, rng: random.Random) -> Image.Image:
    r = rng.random()
    if r < 0.4:
        return ImageEnhance.Color(img).enhance([1.7, 2.2, 2.8, 3.5, 4.5][lv])
    if r < 0.7:
        a = np.asarray(img, dtype=np.float32)
        c = rng.randrange(3)
        gain = np.ones(3, np.float32)
        gain[c] = [1.15, 1.25, 1.38, 1.55, 1.75][lv]
        return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))
    return ImageEnhance.Color(img).enhance([0.6, 0.45, 0.3, 0.18, 0.08][lv])


OPS = {"blur": _blur, "noise": _noise, "artifacts": _artifacts,
       "exposure": _exposure, "contrast": _contrast, "color": _color}


def degrade(img: Image.Image, rng: random.Random) -> tuple[Image.Image, int, float]:
    """Returns (degraded image, defect index, severity in (0, 1])."""
    k = rng.randrange(len(DEFECTS))
    lv = rng.randrange(LEVELS)
    out = OPS[DEFECTS[k]](img, lv, rng)
    return out, k, (lv + 1) / LEVELS
