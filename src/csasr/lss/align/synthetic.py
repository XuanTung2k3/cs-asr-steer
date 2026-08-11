"""Synthetic splice ground truth, and the offsets it licenses.

Reuses the E1 harness (`csasr.data.alignment_checks`) with three repairs:

* source clips are drawn from `D-construct`, not from whatever subset happened
  to be at hand;
* **every** family and the consensus estimator are scored, not only DTW, so a
  bias can be attributed to a family instead of to "the alignment";
* both the legacy switch-midpoint convention and the canonical unit-edge
  convention are reported, because unit edges are what steering masks consume.

The development and gate sets are drawn under **separate registered seeds**: a
configuration chosen on one set cannot then be judged on the same items.
Calibration offsets may be fitted here (external truth) and are validated
against the human audit before anything uses them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ...data.alignment_checks import (
    build_synthetic_pairs,
    render_synthetic_pair,
    summarize_boundary_error,
    synthetic_harness_fingerprint,
)


#: purposes, and the fraction of source utterances each may draw from. The
#: partition is deterministic and content-based, so it does not depend on the
#: order the two sets are built in.
PURPOSES = ("dev", "gate")


def partition_sources(manifest: pd.DataFrame, *, seed: int,
                      purposes: Sequence[str] = PURPOSES) -> dict[str, pd.DataFrame]:
    """Split the source pool into disjoint halves, one per purpose.

    Two different seeds do **not** give disjoint samples -- they give two
    independent draws from the same pool, which overlap heavily. A configuration
    chosen on the development set would then be judged partly on the same audio
    that produced it. Partitioning first is what makes the sets independent;
    `assert_sources_disjoint` verifies it afterwards rather than trusting it.
    """
    if manifest is None or not len(manifest):
        return {p: pd.DataFrame() for p in purposes}
    key = "utterance_id" if "utterance_id" in manifest.columns else manifest.columns[0]
    order = (manifest.assign(_k=manifest[key].astype(str))
             .sort_values("_k").reset_index(drop=True))
    rng = np.random.default_rng(int(seed))
    shuffled = order.iloc[rng.permutation(len(order))].reset_index(drop=True)
    chunks = np.array_split(np.arange(len(shuffled)), len(purposes))
    return {p: shuffled.iloc[idx].drop(columns=["_k"]).reset_index(drop=True)
            for p, idx in zip(purposes, chunks)}


def source_ids(frame: pd.DataFrame) -> set[str]:
    """Every source utterance that contributed audio to a rendered set."""
    out: set[str] = set()
    for column in ("zh_utterance_id", "en_utterance_id", "source_utterance_ids"):
        if frame is None or column not in getattr(frame, "columns", []):
            continue
        for value in frame[column].dropna():
            if isinstance(value, (list, tuple, set)):
                out.update(str(v) for v in value)
            else:
                out.add(str(value))
    return out


def assert_sources_disjoint(sets: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Fail loudly if two purposes share source audio."""
    ids = {name: source_ids(frame) for name, frame in sets.items()}
    names = sorted(ids)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = ids[a] & ids[b]
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)[:10]
    report = {"source_counts": {k: len(v) for k, v in ids.items()},
              "overlaps": overlaps, "disjoint": not overlaps}
    if overlaps:
        raise AssertionError(
            "synthetic development and gate sets share source audio "
            f"({ {k: len(v) for k, v in overlaps.items()} }); a configuration "
            "chosen on one would be judged on the same recordings")
    return report


def build_set(manifest: pd.DataFrame, cfg: Mapping[str, Any], *, n_pairs: int,
              seed: int, out_dir: Path, purpose: str,
              sources: pd.DataFrame | None = None
              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render `n_pairs` ZH+EN splices with known boundaries.

    ``sources`` is the pre-partitioned pool for this purpose; pass it (via
    `partition_sources`) whenever more than one purpose is built, or the two
    sets will overlap.
    """
    scfg = dict(cfg.get("synthetic") or {})
    out_dir = Path(out_dir) / purpose
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = manifest if sources is None else sources

    pairs = build_synthetic_pairs(
        pool, n_pairs=int(n_pairs), seed=int(seed),
        gap_ms=float(scfg.get("gap_ms", 0.0)),
        max_duration_sec=float(scfg.get("max_duration_sec", 28.0)))
    if not pairs:
        return pd.DataFrame(), {"purpose": purpose, "pairs": 0,
                                "reason": "no monolingual pairs available"}

    settings = {"gap_ms": scfg.get("gap_ms", 0.0),
                "trim_silence": scfg.get("trim_silence", True),
                "trim_pad_ms": scfg.get("trim_pad_ms", 20.0),
                "purpose": purpose, "seed": int(seed)}
    fingerprint = synthetic_harness_fingerprint(pairs, settings)

    rendered = []
    for pair in pairs:
        # pair ids restart at syn_0000 for every purpose, so they are namespaced
        # here; otherwise dev and gate items collide in any joined table
        pair = {**pair, "pair_id": f"{purpose}_{pair['pair_id']}"}
        target = out_dir / f"{pair['pair_id']}.wav"
        rendered.append(render_synthetic_pair(
            pair, target, trim_silence=bool(scfg.get("trim_silence", True)),
            pad_ms=float(scfg.get("trim_pad_ms", 20.0))))
    frame = pd.DataFrame(rendered)
    frame["utterance_id"] = frame["pair_id"]
    frame["purpose"] = purpose
    return frame, {"purpose": purpose, "pairs": int(len(frame)),
                   "fingerprint": fingerprint["sha256"], "settings": settings,
                   "source_utterances": sorted(source_ids(frame))}


#: schema declared by every row `score_family` emits. Gate A refuses a table
#: that does not declare it, so an older or hand-made file cannot be read as if
#: it meant the same thing.
SCORES_SCHEMA_VERSION = "lss_synthetic_scores_v1"

#: which boundary each row describes. `combined` is the worse of the two edges
#: for a unit, and is what a steering mask actually depends on.
EDGES = ("start", "end", "combined")


def score_family(predicted_sec: Sequence[float], truth: Sequence[float], *,
                 family: str, convention: str, edge: str = "combined",
                 purpose: str = "gate") -> dict[str, Any]:
    """Boundary-error statistics for one family, convention and edge.

    Emits the exact column names Gate A reads. The underlying E1 helper reports
    `pct_within_100ms` and p95; the proposal's thresholds are stated on
    `within_100ms` and p90, so both are derived here rather than left for a
    reader to guess at.
    """
    if edge not in EDGES:
        raise ValueError(f"edge must be one of {EDGES}, got {edge!r}")
    predicted = np.asarray(predicted_sec, dtype=float)
    true = np.asarray(truth, dtype=float)
    stats = summarize_boundary_error(predicted, true)

    ok = np.isfinite(predicted) & np.isfinite(true)
    abs_ms = np.abs((predicted[ok] - true[ok]) * 1000.0) if ok.any() else np.array([])
    return {
        "schema_version": SCORES_SCHEMA_VERSION,
        "family": str(family),
        "convention": str(convention),
        "edge": str(edge),
        "purpose": str(purpose),
        "num_boundaries": int(stats.get("num_boundaries", 0)),
        # the two names the proposal's thresholds are stated on
        "within_100ms": float(stats.get("pct_within_100ms", float("nan"))),
        "p90_abs_error_ms": float(np.percentile(abs_ms, 90)) if len(abs_ms)
        else float("nan"),
        "median_abs_error_ms": float(stats.get("median_abs_error_ms", float("nan"))),
        "median_signed_error_ms": float(stats.get("median_signed_error_ms",
                                                  float("nan"))),
        "p95_abs_error_ms": float(stats.get("p95_abs_error_ms", float("nan"))),
        "mean_signed_error_ms": float(stats.get("mean_signed_error_ms", float("nan"))),
        "note": stats.get("note", ""),
    }


def scores_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble scored rows into the table Gate A reads."""
    from .autoevidence import SYNTHETIC_REQUIRED_COLUMNS

    frame = pd.DataFrame(list(rows))
    if not len(frame):
        return frame
    missing = [c for c in SYNTHETIC_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"synthetic score rows are missing {missing}; use "
                         "`score_family` to build them")
    return frame


def synthetic_table(scored: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(scored))


def estimate_edge_offsets(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (family, convention) median signed error: the calibration estimand."""
    if not len(scored):
        return pd.DataFrame()
    keep = [c for c in ("family", "convention") if c in scored.columns]
    return (scored.groupby(keep)["median_signed_error_ms"].median()
            .rename("offset_ms").reset_index())


@dataclass(frozen=True)
class EdgeOffsets:
    values: Mapping[str, float]
    source: str
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "source": self.source, "n": int(self.n)}


def fit_offsets(scored: pd.DataFrame, *, source: str = "synthetic") -> EdgeOffsets:
    """Fit correction offsets to **external** truth only."""
    table = estimate_edge_offsets(scored)
    values = {f"{row['family']}": float(row["offset_ms"]) for _, row in table.iterrows()} \
        if len(table) else {}
    return EdgeOffsets(values=values, source=source,
                       n=int(scored["num_boundaries"].sum()) if "num_boundaries" in scored else 0)


def apply_offsets(candidates: pd.DataFrame, offsets: EdgeOffsets, *,
                  sample_rate: int = 16000) -> pd.DataFrame:
    """Shift each family's spans by its fitted offset."""
    if not len(candidates) or not offsets.values:
        return candidates
    out = candidates.copy()
    for family, offset_ms in offsets.values.items():
        mask = out["aligner_family"] == family
        if not mask.any():
            continue
        shift = int(round(-offset_ms / 1000.0 * sample_rate))
        for column in ("start_sample", "end_sample"):
            out.loc[mask, column] = (out.loc[mask, column].astype(float)
                                     + shift).clip(lower=0).astype(int)
    out["calibration_applied"] = True
    return out


def validate_offsets(offsets: EdgeOffsets, human: Mapping[str, Any]) -> dict[str, Any]:
    """Check the synthetic-fitted offset against the independent human audit."""
    human_offset = float(human.get("median_signed_start_ms", float("nan")))
    checks = {}
    for family, offset in offsets.values.items():
        gap = abs(float(offset) - human_offset) if np.isfinite(human_offset) else float("nan")
        checks[family] = {"synthetic_offset_ms": float(offset),
                          "human_offset_ms": human_offset,
                          "inconsistency_ms": gap}
    worst = max((c["inconsistency_ms"] for c in checks.values()
                 if np.isfinite(c["inconsistency_ms"])), default=float("nan"))
    return {"checks": checks, "max_inconsistency_ms": worst,
            "validated_on": "human audit (independent of the synthetic fit)"}
