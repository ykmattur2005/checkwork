"""Paired statistics over questions and calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_mean_ci(diff, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of per-question differences (resampling questions)."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def wilcoxon_p(diff) -> float:
    """Two-sided Wilcoxon signed-rank p-value; zero differences are dropped ('wilcox').
    Returns 1.0 if every difference is zero."""
    diff = np.asarray(diff, float)
    if np.all(diff == 0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def calibration_bins(pred, y, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]; the last bin includes 1.0."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    b = np.clip((pred * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "lo": k / n_bins, "hi": (k + 1) / n_bins, "count": int(m.sum()),
                     "mean_pred": float(pred[m].mean()) if m.any() else np.nan,
                     "frac_correct": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(pred, y, n_bins: int = 10) -> float:
    t = calibration_bins(pred, y, n_bins)
    t = t[t["count"] > 0]
    return float((t["count"] * (t.mean_pred - t.frac_correct).abs()).sum() / t["count"].sum())


def brier(pred, y) -> float:
    return float(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2))
