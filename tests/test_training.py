"""Training-side unit tests (skipped when PyTorch / timm are not installed, e.g. in the serving CI job)."""
import random

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from lumina.constants import ATTRS, DEFECTS  # noqa: E402
from lumina.data import Collate, to_canvas  # noqa: E402
from lumina.losses import degradation_rank, masked_bce, plcc_loss, rank_loss  # noqa: E402
from lumina.synth import degrade  # noqa: E402


def _img(w=512, h=384):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_to_canvas_landscape_and_portrait():
    assert to_canvas(_img(1024, 768)).size == (512, 384)
    assert to_canvas(_img(600, 900)).size == (512, 384)
    assert to_canvas(_img(300, 300)).size == (512, 384)


def test_degrade_every_family():
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        out, k, sev = degrade(_img(), rng)
        assert out.size == (512, 384) and out.mode == "RGB" and 0 < sev <= 1
        seen.add(DEFECTS[k])
    assert seen == set(DEFECTS)


def test_collate_appends_synthetic_pairs():
    items = [{"img": _img(), "z": 0.1 * i, "d": np.full(len(DEFECTS), np.nan, np.float32),
              "a": np.full(len(ATTRS), np.nan, np.float32), "idx": i} for i in range(6)]
    b = Collate(0.5)(items)
    assert b["img"].shape == (9, 3, 384, 512) and b["img"].dtype == torch.uint8
    assert torch.isnan(b["z"][6:]).all() and len(b["pair_o"]) == 3
    assert (b["pair_d"] >= 6).all()


def test_losses_behave():
    y = torch.linspace(-1, 1, 16)
    assert plcc_loss(y * 3 + 1, y) < 1e-5
    assert rank_loss(y, y) < rank_loss(-y, y)
    assert masked_bce(torch.zeros(2, 6), torch.full((2, 6), float("nan"))) == 0
    s = torch.tensor([1.0, 0.0])
    o, d = torch.tensor([0]), torch.tensor([1])
    keep, sev = torch.tensor([True]), torch.tensor([0.8])
    assert degradation_rank(s, o, d, sev, keep) == 0           # degraded already scores lower
    assert degradation_rank(s.flip(0), o, d, sev, keep) > 0    # degraded scores higher -> penalised


def test_model_forward_shapes():
    from lumina.model import LuminaNet
    m = LuminaNet(pretrained=False).eval()
    with torch.no_grad():
        o = m(torch.randint(0, 255, (2, 3, 384, 512)).float())
    assert o["score"].shape == (2,) and o["qmap"].shape == (2, 24, 32)
    assert o["dmap"].shape == (2, len(DEFECTS), 24, 32) and o["attrs"].shape == (2, len(ATTRS))
    assert torch.allclose(o["wmap"].sum(dim=(1, 2)), torch.ones(2), atol=1e-4)
