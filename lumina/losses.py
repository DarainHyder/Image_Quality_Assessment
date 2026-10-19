"""Losses for perceptual quality regression."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def plcc_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if p.numel() < 3:
        return p.new_zeros(())
    p, y = p - p.mean(), y - y.mean()
    r = (p * y).sum() / (p.norm() * y.norm() + 1e-8)
    return 1.0 - r


def rank_loss(p: torch.Tensor, y: torch.Tensor, tau: float = 0.3, min_gap: float = 0.05) -> torch.Tensor:
    """Pairwise logistic ranking loss over all pairs with a meaningful MOS gap."""
    if p.numel() < 2:
        return p.new_zeros(())
    dp = p[:, None] - p[None, :]
    dy = y[:, None] - y[None, :]
    m = dy.abs() > min_gap
    if not m.any():
        return p.new_zeros(())
    w = dy.abs().clamp(max=1.5)[m]
    return (F.softplus(-dp[m] * torch.sign(dy[m]) / tau) * w).sum() / w.sum()


def masked_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(target)
    if not m.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[m], target[m])


def masked_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(target)
    if not m.any():
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[m], target[m], beta=0.5)


def degradation_rank(score: torch.Tensor, o: torch.Tensor, d: torch.Tensor, sev: torch.Tensor,
                     keep: torch.Tensor, margin: float = 0.8) -> torch.Tensor:
    """A degraded copy must score at least ``margin * severity`` (z-units) below its original."""
    sel = keep & (sev >= 0.4)
    if not sel.any():
        return score.new_zeros(())
    gap = score[d[sel]] - score[o[sel]] + margin * sev[sel]
    return F.relu(gap).mean()
