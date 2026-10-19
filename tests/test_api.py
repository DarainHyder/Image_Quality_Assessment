import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MODEL_DIR = Path(os.getenv("LUMINA_MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "lumina_v2.json").exists(), reason="exported model not available")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from server.app import app
    with TestClient(app) as c:
        yield c


def _jpeg(w=800, h=600, seed=0, blur=False) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 64) * 4], -1) + rng.normal(0, 3, (h, w, 3))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if blur:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(6))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_root(client):
    r = client.get("/").json()
    assert r["model_loaded"] is True


def test_analyze_shapes(client):
    r = client.post("/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert 0 <= d["score"] <= 100
    assert d["rating"] in {"excellent", "good", "fair", "poor", "bad"}
    h, w = d["maps"]["shape"]
    assert len(d["maps"]["quality"]) == h and len(d["maps"]["quality"][0]) == w
    assert set(d["defects"]) == {"blur", "noise", "artifacts", "exposure", "contrast", "color"}
    assert len(d["technical"]["histogram"]) == 64
    assert d["input_size"] == [512, 384]


def test_portrait_and_small(client):
    for w, h in [(600, 900), (120, 90)]:
        r = client.post("/analyze", files={"file": ("p.jpg", _jpeg(w, h), "image/jpeg")})
        assert r.status_code == 200, r.text
        iw, ih = r.json()["input_size"]
        assert iw % 32 == 0 and ih % 32 == 0 and (ih > iw) == (h > w)


def test_predict_v1_compat(client):
    d = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    assert d["status"] == "success" and isinstance(d["quality_score"], float)


def test_rejects_garbage(client):
    r = client.post("/analyze", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
