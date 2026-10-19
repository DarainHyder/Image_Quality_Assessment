"""Standard IQA correlation metrics."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def _logistic4(x, b1, b2, b3, b4):
    return (b1 - b2) / (1 + np.exp(-(x - b3) / np.abs(b4))) + b2


def iqa_metrics(pred, mos) -> dict[str, float]:
    pred, mos = np.asarray(pred, np.float64), np.asarray(mos, np.float64)
    srcc = stats.spearmanr(pred, mos).statistic
    krcc = stats.kendalltau(pred, mos).statistic
    plcc = stats.pearsonr(pred, mos).statistic
    try:  # VQEG-style PLCC after monotonic logistic mapping (for cross-scale comparisons)
        p0 = [mos.max(), mos.min(), float(np.mean(pred)), float(np.std(pred)) or 1.0]
        popt, _ = curve_fit(_logistic4, pred, mos, p0=p0, maxfev=20000)
        plcc_fit = stats.pearsonr(_logistic4(pred, *popt), mos).statistic
    except RuntimeError:
        plcc_fit = plcc
    rmse = float(np.sqrt(np.mean((pred - mos) ** 2)))
    return {"srcc": float(srcc), "plcc": float(plcc), "plcc_logistic": float(plcc_fit),
            "krcc": float(krcc), "rmse": rmse, "n": int(len(pred))}
