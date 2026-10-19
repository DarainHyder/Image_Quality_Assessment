"""Classical, model-free photo forensics that complement the learned score.

numpy + Pillow only, so the serving image stays small.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from PIL import ExifTags, Image

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def sharpness(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian (higher = crisper detail)."""
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkær's fast noise estimate (std of additive Gaussian noise, 0-255 units)."""
    k = (gray[:-2, :-2] - 2 * gray[:-2, 1:-1] + gray[:-2, 2:]
         - 2 * gray[1:-1, :-2] + 4 * gray[1:-1, 1:-1] - 2 * gray[1:-1, 2:]
         + gray[2:, :-2] - 2 * gray[2:, 1:-1] + gray[2:, 2:])
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(k).sum() / (6 * (w - 2) * (h - 2)))


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def jpeg_quality(img: Image.Image) -> int | None:
    """Estimate the JPEG quality factor from the luminance quantisation table."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.float64)
    tbl = np.array(q[0][:64], np.float64)
    scale = float(np.mean(tbl / std)) * 100
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(np.clip(round(quality), 1, 100))


def _ratio(v) -> float | None:
    try:
        return float(Fraction(v.numerator, v.denominator)) if hasattr(v, "numerator") else float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def exif(img: Image.Image) -> dict:
    try:
        raw = img.getexif()
        ifd = raw.get_ifd(0x8769) if raw else {}
    except Exception:  # noqa: BLE001 - malformed EXIF must never fail a request
        return {}
    get = lambda name: ifd.get(_EXIF_TAGS[name], raw.get(_EXIF_TAGS[name]))  # noqa: E731
    out = {
        "make": get("Make"), "model": get("Model"), "lens": get("LensModel"),
        "iso": get("ISOSpeedRatings"), "datetime": get("DateTimeOriginal"),
        "f_number": _ratio(get("FNumber")) if get("FNumber") is not None else None,
        "exposure_s": _ratio(get("ExposureTime")) if get("ExposureTime") is not None else None,
        "focal_mm": _ratio(get("FocalLength")) if get("FocalLength") is not None else None,
    }
    clean = {}
    for k, v in out.items():
        if isinstance(v, bytes):
            v = v.decode(errors="ignore")
        if isinstance(v, tuple):
            v = v[0] if v else None
        if isinstance(v, str):
            v = v.strip("\x00 ").strip() or None
        if v is not None:
            clean[k] = v
    return clean


def palette(img: Image.Image, n: int = 5) -> list[dict]:
    small = img.copy()
    small.thumbnail((128, 128))
    q = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[: n * 3]
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts)
    return [{"hex": "#%02x%02x%02x" % tuple(pal[i * 3: i * 3 + 3]), "share": round(c / total, 4)} for c, i in counts]


def analyze(img: Image.Image, original: Image.Image | None = None) -> dict:
    """Technical read-out for an RGB image (``original`` keeps format / EXIF / JPEG tables)."""
    src = original or img
    work = img.copy()
    work.thumbnail((1024, 1024), Image.BILINEAR)
    rgb = np.asarray(work, dtype=np.float32)
    y = _luma(rgb)
    hist, _ = np.histogram(y, bins=64, range=(0, 256))
    n = y.size
    return {
        "width": img.width, "height": img.height,
        "megapixels": round(img.width * img.height / 1e6, 2),
        "format": src.format, "mode": src.mode,
        "jpeg_quality": jpeg_quality(src),
        "luma_mean": round(float(y.mean()), 2),
        "rms_contrast": round(float(y.std()), 2),
        "clipped_highlights": round(float((y >= 250).sum() / n), 4),
        "crushed_shadows": round(float((y <= 5).sum() / n), 4),
        "sharpness": round(sharpness(y), 2),
        "noise_sigma": round(noise_sigma(y), 3),
        "colorfulness": round(colorfulness(rgb), 2),
        "histogram": (hist / hist.max()).round(4).tolist() if hist.max() else hist.tolist(),
        "palette": palette(work),
        "exif": exif(src),
    }
