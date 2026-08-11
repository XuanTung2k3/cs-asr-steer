"""Boundary-jitter stability of the steering mask.

Proposal §4.1 requires that conclusions survive +/-50 and +/-100 ms boundary
error; plan v1 dropped it. It matters more here than in a typical setup: the
median embedded-English consensus span is 225 ms, so +/-100 ms of jitter moves
an edge by nearly half the span, and a mask that drifts onto the neighbouring
Mandarin word is not a local intervention at all.

Three quantities are measured: how much of the mask survives (IoU), how often
the jittered mask lands on a *different-language* span (contamination), and
whether the eroded safe interior still exists.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..seeds import seeds_for


def mask_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two half-open intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 0 else 0.0


def jitter_spans(spans: pd.DataFrame, *, offset_ms: float, seed: int,
                 independent_edges: bool = True,
                 sample_rate: int = 16000) -> pd.DataFrame:
    """Perturb both edges of every span by up to +/- offset_ms."""
    rng = np.random.default_rng(int(seed))
    out = spans.copy()
    delta = int(round(offset_ms / 1000.0 * sample_rate))
    n = len(out)
    start_shift = rng.integers(-delta, delta + 1, size=n)
    end_shift = (rng.integers(-delta, delta + 1, size=n) if independent_edges
                 else start_shift)
    start = out["consensus_start_sample"].to_numpy(dtype=float) + start_shift
    end = out["consensus_end_sample"].to_numpy(dtype=float) + end_shift
    end = np.maximum(end, start + 1)
    out["jittered_start_sample"] = np.maximum(0, start).astype(int)
    out["jittered_end_sample"] = end.astype(int)
    out["jitter_offset_ms"] = float(offset_ms)
    out["jitter_seed"] = int(seed)
    return out


def safe_interior_survives(jittered: pd.DataFrame,
                           *, sample_rate: int = 16000) -> pd.Series:
    """Does the eroded safe interior still lie inside the *shifted* span?

    The question is whether the region a steering mask would actually touch is
    still covered once the boundaries move. Comparing the jittered span's total
    duration against the original interior's duration answers a different
    question -- a span shifted bodily off its content keeps its duration and
    would "survive" while covering the wrong audio entirely.
    """
    if not len(jittered):
        return pd.Series(dtype=bool)
    start = jittered["jittered_start_sample"].astype(float)
    end = jittered["jittered_end_sample"].astype(float)
    if "safe_start_sample" in jittered and "safe_end_sample" in jittered:
        safe_start = jittered["safe_start_sample"].astype(float)
        safe_end = jittered["safe_end_sample"].astype(float)
    else:
        # fall back to a centred interior of the recorded width
        width = (jittered.get("safe_interior_ms",
                              pd.Series([0.0] * len(jittered), index=jittered.index))
                 .astype(float) / 1000.0 * sample_rate)
        centre = (jittered["consensus_start_sample"].astype(float)
                  + jittered["consensus_end_sample"].astype(float)) / 2.0
        safe_start = centre - width / 2.0
        safe_end = centre + width / 2.0
    return (start <= safe_start) & (end >= safe_end) & (safe_end > safe_start)


def cross_language_contamination(jittered: pd.DataFrame,
                                 spans: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each jittered mask that lands on a different-language span."""
    rows: list[dict[str, Any]] = []
    by_utt = {utt: group for utt, group in spans.groupby("utterance_id")}
    for _, row in jittered.iterrows():
        neighbours = by_utt.get(row["utterance_id"])
        start = float(row["jittered_start_sample"])
        end = float(row["jittered_end_sample"])
        width = max(end - start, 1.0)
        contaminated = 0.0
        if neighbours is not None:
            for _, other in neighbours.iterrows():
                if int(other["unit_id"]) == int(row["unit_id"]):
                    continue
                if other.get("language") == row.get("language"):
                    continue
                overlap = max(0.0, min(end, float(other["consensus_end_sample"]))
                              - max(start, float(other["consensus_start_sample"])))
                contaminated += overlap
        rows.append({
            "utterance_id": row["utterance_id"],
            "unit_id": int(row["unit_id"]),
            "language": row.get("language", ""),
            "contaminated_fraction": float(min(1.0, contaminated / width)),
        })
    return pd.DataFrame(rows)


def jitter_stability(spans: pd.DataFrame, cfg: dict, *,
                     offsets_ms: Sequence[float] | None = None,
                     independent_edges: bool = True
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Median IoU, contamination and safe-interior survival per jitter offset."""
    jcfg = cfg.get("jitter") or {}
    offsets = offsets_ms if offsets_ms is not None else jcfg.get("offsets_ms", [50, 100])
    seeds = seeds_for(cfg, "jitter")
    rows: list[dict[str, Any]] = []

    for offset in offsets:
        for seed in seeds:
            jittered = jitter_spans(spans, offset_ms=float(offset), seed=int(seed),
                                    independent_edges=independent_edges)
            ious = [mask_iou((float(r["consensus_start_sample"]),
                              float(r["consensus_end_sample"])),
                             (float(r["jittered_start_sample"]),
                              float(r["jittered_end_sample"])))
                    for _, r in jittered.iterrows()]
            contamination = cross_language_contamination(jittered, spans)
            survives = safe_interior_survives(jittered)
            rows.append({
                "offset_ms": float(offset),
                "seed": int(seed),
                "n": int(len(jittered)),
                "median_mask_iou": float(np.median(ious)) if ious else float("nan"),
                "mean_contaminated_fraction": float(
                    contamination["contaminated_fraction"].mean()) if len(contamination) else 0.0,
                "contaminated_rate": float(
                    (contamination["contaminated_fraction"] > 0.05).mean())
                if len(contamination) else 0.0,
                "safe_interior_survival": float(survives.mean()) if len(survives) else float("nan"),
            })

    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    if len(table):
        for offset in sorted(table["offset_ms"].unique()):
            subset = table[table["offset_ms"] == offset]
            summary[f"offset_{int(offset)}ms"] = {
                "median_mask_iou": float(subset["median_mask_iou"].mean()),
                "contaminated_rate": float(subset["contaminated_rate"].mean()),
                "safe_interior_survival": float(subset["safe_interior_survival"].mean()),
                "seeds": int(len(subset)),
            }
    return table, summary
