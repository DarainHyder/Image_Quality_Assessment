"""LuminaNet — multi-scale, spatially-explainable no-reference IQA model.

Backbone features from three ConvNeXt stages (strides 8/16/32) are projected,
aligned on the stride-16 grid and fused. Three light heads read the fused map:

* quality map   q(x, y)   — local perceptual quality (z-score units)
* attention map w(x, y)   — how much each region matters to a human rater
* defect maps   d_k(x, y) — blur / noise / artifacts / exposure / contrast / color logits
* attributes             — human-rated brightness / colorfulness / contrast / noisiness / sharpness

The image score is the attention-weighted mean of the quality map, and image
level defect logits use the same pooling — so every number the model returns can
be traced back to *where* in the frame it came from.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ATTRS, BACKBONE, DEFECTS, SOURCES


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.GroupNorm(32, cout),
            nn.GELU(),
        )


class LuminaNet(nn.Module):
    def __init__(self, backbone: str = BACKBONE, pretrained: bool = True, width: int = 384,
                 drop_path: float = 0.2, n_sources: int = len(SOURCES)):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3),
            drop_path_rate=drop_path,
        )
        chs = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([ConvNormAct(c, 256) for c in chs])
        self.fuse = nn.Sequential(ConvNormAct(256 * len(chs), width, 3), ConvNormAct(width, width, 3))

        def head(cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Dropout2d(0.1), nn.Conv2d(128, cout, 1))

        self.q_head = head(1)
        self.w_head = head(1)
        self.d_head = head(len(DEFECTS))
        self.a_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, len(ATTRS)))
        # Per-dataset affine calibration of the shared score (index 0 = KonIQ = identity).
        self.calib = nn.Parameter(torch.tensor([[1.0, 0.0]] * n_sources))

        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255.0, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: float or uint8 RGB in [0, 255], shape (B, 3, H, W)."""
        x = (x.float() - self.mean) / self.std
        feats = self.backbone(x)
        size = feats[1].shape[-2:]
        aligned = []
        for i, (f, p) in enumerate(zip(feats, self.proj)):
            f = p(f)
            if i == 0:  # stride 8 -> 16 (exact 2x; export-friendly, unlike adaptive pooling)
                f = F.avg_pool2d(f, 2)
            if i != 1:
                f = F.interpolate(f, size=size, mode="bilinear", align_corners=False)
            aligned.append(f)
        h = self.fuse(torch.cat(aligned, 1))

        q = self.q_head(h).float()                       # (B,1,h,w)
        w = F.softplus(self.w_head(h).float()) + 1e-4    # (B,1,h,w)
        d = self.d_head(h).float()                       # (B,K,h,w)
        wn = w / w.sum(dim=(2, 3), keepdim=True)
        score = (q * wn).sum(dim=(1, 2, 3))
        defects = (d * wn).sum(dim=(2, 3))
        pooled = (h.float() * wn).sum(dim=(2, 3))
        attrs = self.a_head(pooled)                       # z-scored SPAQ attributes
        return {"score": score, "qmap": q[:, 0], "wmap": wn[:, 0], "defects": defects, "dmap": d, "attrs": attrs}

    def calibrated(self, score: torch.Tensor, source: int) -> torch.Tensor:
        a, b = self.calib[source]
        return score * a + b

    def param_groups(self, lr_backbone: float, lr_head: float, wd: float):
        decay, no_decay = [], []
        for n, p in self.backbone.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        head = [p for n, p in self.named_parameters() if not n.startswith("backbone.") and n != "calib"]
        return [
            {"params": decay, "lr": lr_backbone, "weight_decay": wd},
            {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head, "lr": lr_head, "weight_decay": wd},
            {"params": [self.calib], "lr": lr_head, "weight_decay": 0.0},
        ]


class ExportWrapper(nn.Module):
    """ONNX-friendly wrapper: uint8-range RGB in, calibrated KonIQ MOS + maps out."""

    def __init__(self, model: LuminaNet, mos_mean: float, mos_std: float,
                 attr_mean: list[float], attr_std: list[float]):
        super().__init__()
        self.model = model
        self.mos_mean, self.mos_std = mos_mean, mos_std
        self.register_buffer("attr_mean", torch.tensor(attr_mean))
        self.register_buffer("attr_std", torch.tensor(attr_std))

    def forward(self, x: torch.Tensor):
        o = self.model(x)
        mos = o["score"] * self.mos_std + self.mos_mean
        qmap = o["qmap"] * self.mos_std + self.mos_mean
        attrs = (o["attrs"] * self.attr_std + self.attr_mean).clamp(0, 100)
        return mos, qmap, o["wmap"], torch.sigmoid(o["defects"]), torch.sigmoid(o["dmap"]), attrs
