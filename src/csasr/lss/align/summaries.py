"""Per-family, per-language validity summaries.

Two wrappers around `csasr.nat5h`, which is not edited:

* its rejection codes are hard-coded to `_gt_200ms` regardless of the configured
  tolerance, so at 100 ms the label would misstate the reason;
* its confidence bins are fixed at 100 ms independently of the criteria.

Both are relabelled here rather than patched there, so the NAT5H artifacts stay
reproducible.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def family_language_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Coverage and validity for every (family, variant, language)."""
    if not len(candidates):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["aligner_family", "aligner_variant", "reference_language"]
    for keys, group in candidates.groupby(group_cols):
        valid = group.get("is_valid", pd.Series([True] * len(group))).astype(bool)
        durations = ((group["end_sample"].astype(float)
                      - group["start_sample"].astype(float)) / 16000.0 * 1000.0)
        durations = durations[valid.to_numpy()]
        failures = group.loc[~valid.to_numpy(), "failure_code"] \
            if "failure_code" in group else pd.Series(dtype=str)
        rows.append({
            **dict(zip(group_cols, keys)),
            "candidate_units": int(len(group)),
            "valid_units": int(valid.sum()),
            "valid_unit_coverage": float(valid.mean()),
            "invalid_rate": float(1.0 - valid.mean()),
            "median_duration_ms": float(np.median(durations)) if len(durations) else float("nan"),
            "top_failure_code": (failures.value_counts().index[0]
                                 if len(failures) and failures.notna().any() else ""),
        })
    return pd.DataFrame(rows)


def stratified_validity(candidates: pd.DataFrame, *,
                        duration_terciles: bool = True) -> pd.DataFrame:
    """Validity by language and unit position/duration, to expose class asymmetry."""
    if not len(candidates):
        return pd.DataFrame()
    frame = candidates.copy()
    frame["valid"] = frame.get("is_valid", True)
    frame["duration_ms"] = ((frame["end_sample"].astype(float)
                             - frame["start_sample"].astype(float)) / 16000.0 * 1000.0)
    frame["unit_position_bin"] = pd.cut(
        frame.groupby("utterance_id")["reference_unit_index"].rank(pct=True),
        bins=[0, 0.33, 0.66, 1.0], labels=["early", "middle", "late"],
        include_lowest=True)
    if duration_terciles:
        positive = frame.loc[frame["duration_ms"] > 0, "duration_ms"]
        edges = np.unique(np.quantile(positive, [0, 1 / 3, 2 / 3, 1])) if len(positive) else []
        frame["duration_tercile"] = pd.cut(frame["duration_ms"], bins=edges,
                                           labels=False, include_lowest=True) \
            if len(edges) > 2 else 0
    else:
        frame["duration_tercile"] = 0

    grouped = frame.groupby(["aligner_family", "reference_language",
                             "unit_position_bin", "duration_tercile"], observed=True)
    return grouped.agg(n=("valid", "size"),
                       valid_rate=("valid", "mean"),
                       median_duration_ms=("duration_ms", "median")).reset_index()


def relabel_rejections(rejected: pd.DataFrame, tolerance_ms: float) -> pd.DataFrame:
    """Rewrite `*_gt_200ms` codes to the tolerance that was actually applied."""
    if not len(rejected) or "rejection_code" not in rejected:
        return rejected
    out = rejected.copy()
    out["rejection_code"] = out["rejection_code"].astype(str).str.replace(
        r"_gt_\d+ms$", f"_gt_{int(round(tolerance_ms))}ms", regex=True)
    out["tolerance_ms"] = float(tolerance_ms)
    return out


def rejection_codes(rejected: pd.DataFrame, tolerance_ms: float) -> pd.DataFrame:
    """Counts per rejection reason, by language."""
    relabelled = relabel_rejections(rejected, tolerance_ms)
    if not len(relabelled):
        return pd.DataFrame()
    language = ("reference_language" if "reference_language" in relabelled
                else "language")
    return (relabelled.groupby([language, "rejection_code"])
            .size().rename("n").reset_index()
            .sort_values("n", ascending=False))


def reproduction_check(observed: Mapping[str, int], recorded: Mapping[str, int],
                       tolerance: float = 0.02) -> dict[str, Any]:
    """Compare recomputed counts against a previous run's recorded gate evidence."""
    checks: dict[str, Any] = {}
    ok = True
    for key, expected in recorded.items():
        actual = observed.get(key)
        if actual is None:
            checks[key] = {"expected": expected, "actual": None, "ok": False}
            ok = False
            continue
        allowed = max(2.0, abs(float(expected)) * float(tolerance))
        within = abs(float(actual) - float(expected)) <= allowed
        checks[key] = {"expected": expected, "actual": actual,
                       "allowed_delta": allowed, "ok": bool(within)}
        ok = ok and within
    return {"reproduced": bool(ok), "checks": checks, "tolerance": float(tolerance)}
