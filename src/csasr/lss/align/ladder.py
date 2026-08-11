"""Attribution ladder for the CTC-vs-DTW disagreement.

Production E1 reported a CTC-vs-DTW systematic offset of +1354 ms and 7%
agreement within 100 ms, and plan v1 hypothesised a coordinate-mapping defect.
Three things differ between the E1 audit path and the NAT5H path at once --
unit indexing, time coordinates, and the statistic being compared (a switch
midpoint versus a unit edge) -- so "recompute it the other way and see if it
moves" cannot attribute anything.

Each rung changes exactly one of those. Measured on the artifacts already on
disk the medians barely move (EN start 360 ms canonical versus E1's 370 ms
median), and the regression slope is flat, so the expected verdict is
`refuted`: E1's headline number was a mean inflated by a tail, not a scale
error. The ladder is run anyway, cheaply, because a hypothesis that the plan
depended on should be falsified explicitly rather than dropped quietly.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .bias import disagreement_regression, paired_edges

LADDER_RUNGS = ("unit_edge_canonical", "switch_midpoint_legacy", "unit_edge_first_only")


def _summary(values: np.ndarray, rung: str, language: str, edge: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"rung": rung, "language": language, "edge": edge, "n": 0}
    absolute = np.abs(values)
    return {
        "rung": rung, "language": language, "edge": edge, "n": int(len(values)),
        "median_signed_ms": float(np.median(values)),
        "mean_signed_ms": float(values.mean()),
        "median_abs_ms": float(np.median(absolute)),
        "p90_abs_ms": float(np.percentile(absolute, 90)),
        "within_50ms": float((absolute <= 50).mean()),
        "within_100ms": float((absolute <= 100).mean()),
        "within_200ms": float((absolute <= 200).mean()),
    }


def switch_midpoints(candidates: pd.DataFrame, family: str, *,
                     sample_rate: int = 16000) -> pd.DataFrame:
    """Language-switch midpoints, the quantity E1's audit compared.

    Nothing downstream consumes a midpoint -- steering masks and localizer
    labels use unit edges -- so this rung exists only to reproduce the old
    statistic and show whether it is what produced the old number.
    """
    frame = candidates[candidates["aligner_family"] == family]
    if "is_valid" in frame:
        frame = frame[frame["is_valid"].astype(bool)]
    rows = []
    for utt, group in frame.groupby("utterance_id"):
        group = group.sort_values("reference_unit_index")
        previous = None
        for _, row in group.iterrows():
            if previous is not None and previous["reference_language"] != row["reference_language"]:
                midpoint = (float(previous["end_sample"]) + float(row["start_sample"])) / 2.0
                rows.append({
                    "utterance_id": utt,
                    "reference_unit_index": int(row["reference_unit_index"]),
                    "direction": f"{previous['reference_language']}->{row['reference_language']}",
                    "midpoint_sec": midpoint / sample_rate,
                })
            previous = row
    return pd.DataFrame(rows)


def run_ladder(candidates: pd.DataFrame, *, family_a: str = "whisper_dtw",
               family_b: str = "existing_ctc",
               sample_rate: int = 16000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute every rung and the regression that discriminates the hypotheses."""
    pairs = paired_edges(candidates, sample_rate=sample_rate,
                         family_a=family_a, family_b=family_b)
    rows: list[dict[str, Any]] = []

    if len(pairs):
        for language in ["ALL"] + sorted(pairs["language"].unique()):
            subset = pairs if language == "ALL" else pairs[pairs["language"] == language]
            rows.append(_summary(subset["dstart_ms"].to_numpy(),
                                 "unit_edge_canonical", language, "start"))
            rows.append(_summary(subset["dend_ms"].to_numpy(),
                                 "unit_edge_canonical", language, "end"))
        first = pairs[pairs["is_first_unit"]]
        rows.append(_summary(first["dstart_ms"].to_numpy(),
                             "unit_edge_first_only", "ALL", "start"))

    mid_a = switch_midpoints(candidates, family_a, sample_rate=sample_rate)
    mid_b = switch_midpoints(candidates, family_b, sample_rate=sample_rate)
    if len(mid_a) and len(mid_b):
        merged = mid_a.merge(mid_b, on=["utterance_id", "reference_unit_index"],
                             suffixes=("_a", "_b"))
        if len(merged):
            delta = (merged["midpoint_sec_a"] - merged["midpoint_sec_b"]) * 1000.0
            rows.append(_summary(delta.to_numpy(), "switch_midpoint_legacy",
                                 "ALL", "midpoint"))
            for direction, group in merged.groupby("direction_a"):
                delta = (group["midpoint_sec_a"] - group["midpoint_sec_b"]) * 1000.0
                rows.append(_summary(delta.to_numpy(), "switch_midpoint_legacy",
                                     str(direction), "midpoint"))

    table = pd.DataFrame(rows)
    regression = disagreement_regression(pairs) if len(pairs) else {"n": 0}
    canonical = table[(table["rung"] == "unit_edge_canonical")
                      & (table["language"] == "ALL")
                      & (table["edge"] == "start")] if len(table) else table
    legacy = table[table["rung"] == "switch_midpoint_legacy"] if len(table) else table
    canonical_median = float(canonical["median_abs_ms"].iloc[0]) if len(canonical) else float("nan")
    legacy_median = float(legacy["median_abs_ms"].max()) if len(legacy) else float("nan")
    evidence = {
        "rungs_completed": int(table["rung"].nunique()) if len(table) else 0,
        "regression": regression,
        "verdict": verdict(table, regression),
        "canonical_unit_edge_median_abs_ms": canonical_median,
        "legacy_midpoint_median_abs_ms": legacy_median,
        "statistic_effect_ms": legacy_median - canonical_median,
        "residual_disagreement_ms": canonical_median,
        "family_a": family_a,
        "family_b": family_b,
        "paired_units": int(len(pairs)),
    }
    return table, evidence


def verdict(ladder: pd.DataFrame, regression: dict[str, Any], *,
            slope_tolerance_ms_per_s: float = 5.0,
            median_shift_ms: float = 100.0,
            residual_ok_ms: float = 100.0) -> str:
    """Verdict on the coordinate hypothesis, keeping two effects separate.

    Plan v1 asked one question ("was the old number a coordinate bug?") where
    there are three: a *scale* error, a *statistic* effect (E1 compared switch
    midpoints; steering masks consume unit edges), and whatever disagreement
    survives both. Returning a single "supported" for any of them would let a
    statistic change be mistaken for a repaired aligner, so the residual has to
    clear the bar too:

    ``supported``            the canonical statistic agrees well enough to use
    ``partially_supported``  the statistic explains much of the old number, but
                             the residual disagreement is still too large
    ``refuted``              neither a scale error nor a statistic effect
    """
    if not len(ladder):
        return "inconclusive"
    canonical = ladder[(ladder["rung"] == "unit_edge_canonical")
                       & (ladder["language"] == "ALL") & (ladder["edge"] == "start")]
    legacy = ladder[ladder["rung"] == "switch_midpoint_legacy"]
    # a slope only counts as a scale error when the relationship is linear
    # enough to believe; see MIN_R_FOR_SCALE_ERROR in .bias
    slope = float(regression.get("theil_slope_ms_per_s",
                                 regression.get("slope_ms_per_s", float("nan"))))
    scale_error = bool(regression.get("scale_error_evidence")
                       and np.isfinite(slope) and abs(slope) > slope_tolerance_ms_per_s)
    if not len(canonical):
        return "supported" if scale_error else "inconclusive"

    canonical_median = float(canonical["median_abs_ms"].iloc[0])
    residual_ok = canonical_median <= residual_ok_ms
    if scale_error:
        return "supported" if residual_ok else "partially_supported"
    if not len(legacy):
        return "inconclusive"
    statistic_effect = float(legacy["median_abs_ms"].max()) - canonical_median
    if statistic_effect > median_shift_ms:
        return "supported" if residual_ok else "partially_supported"
    return "refuted"
