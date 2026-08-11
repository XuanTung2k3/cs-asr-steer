"""Paired and grouped bootstrap confidence intervals (guide sections 31, 44)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def grouped_bootstrap(values: Sequence[float], groups: Sequence[str],
                      statistic: Callable[[np.ndarray], float] = np.mean,
                      n_resamples: int = 1000, seed: int = 342,
                      alpha: float = 0.05) -> dict:
    """Resample whole groups (speakers / conversations) with replacement."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    point = float(statistic(values)) if len(values) else float("nan")
    if len(uniq) < 2 or len(values) == 0:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples": 0, "n_groups": int(len(uniq))}

    stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        stats[b] = statistic(values[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(stats.std(ddof=1)),
        "n_resamples": int(n_resamples),
        "n_groups": int(len(uniq)),
    }


def paired_bootstrap(a: Sequence[float], b: Sequence[float], groups: Sequence[str],
                     statistic: Callable[[np.ndarray], float] = np.mean,
                     n_resamples: int = 10000, seed: int = 342,
                     alpha: float = 0.05) -> dict:
    """CI on statistic(b) - statistic(a) with paired group resampling."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal length")
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    point = float(statistic(b) - statistic(a)) if len(a) else float("nan")
    if len(uniq) < 2 or len(a) == 0:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value_two_sided": float("nan"), "n_resamples": 0}

    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        diffs[i] = statistic(b[idx]) - statistic(a[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value for H0: diff == 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(diffs.std(ddof=1)),
        "p_value_two_sided": float(min(1.0, p)),
        "n_resamples": int(n_resamples),
        "n_groups": int(len(uniq)),
    }


def ratio_statistic(numer: Sequence[float], denom: Sequence[float]) -> Callable:
    """Statistic for rates computed as sum(numer)/sum(denom) under resampling."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)

    def stat(idx_values: np.ndarray) -> float:
        idx = idx_values.astype(int)
        d = denom[idx].sum()
        return float(numer[idx].sum() / d) if d else float("nan")

    return stat
