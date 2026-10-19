import io

import numpy as np
from PIL import Image, ImageFilter

from lumina.analysis import analyze, jpeg_quality, noise_sigma, sharpness


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(1))


def test_blur_lowers_sharpness():
    a = np.asarray(_img().convert("L"), np.float32)
    b = np.asarray(_img().filter(ImageFilter.GaussianBlur(4)).convert("L"), np.float32)
    assert sharpness(a) > 5 * sharpness(b)


def test_noise_estimate_tracks_sigma():
    flat = np.full((200, 200), 128.0)
    rng = np.random.default_rng(0)
    for sigma in (5, 15):
        est = noise_sigma(flat + rng.normal(0, sigma, flat.shape))
        assert abs(est - sigma) / sigma < 0.15


def test_jpeg_quality_roundtrip():
    for q in (40, 75, 95):
        buf = io.BytesIO()
        _img().save(buf, "JPEG", quality=q)
        assert abs(jpeg_quality(Image.open(io.BytesIO(buf.getvalue()))) - q) <= 3


def test_analyze_keys():
    d = analyze(_img())
    assert {"sharpness", "noise_sigma", "colorfulness", "histogram", "palette", "exif"} <= set(d)
    assert abs(sum(p["share"] for p in d["palette"]) - 1) < 1e-3
