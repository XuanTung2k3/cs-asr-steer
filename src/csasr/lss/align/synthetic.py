"""Reference-aware synthetic splice diagnostics.

Reuses the E1 harness (`csasr.data.alignment_checks`) with three repairs:

* source clips are drawn from `D-construct`, not from whatever subset happened
  to be at hand;
* **every** family and the consensus estimator are scored, not only DTW, so a
  bias can be attributed to a family instead of to "the alignment";
* both the legacy switch-midpoint convention and the canonical unit-edge
  convention are reported, because unit edges are what steering masks consume.

RMS/VAD concatenation yields a known audio seam, not an independently known
lexical edge.  Its signed offsets are diagnostics only. Exact lexical fixtures
remain supported through an explicit ``reference_kind``.

The development and gate sets are drawn under **separate registered seeds**: a
configuration chosen on one set cannot then be judged on the same items.
Calibration offsets may be fitted only from genuine lexical references; audio
seams remain coordinate diagnostics and cannot authorize a correction.
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

REFERENCE_KINDS = (
    "manual_lexical", "existing_gold_lexical", "constructed_exact_lexical",
    "audio_splice", "cross_aligner_consensus",
)
LEXICAL_REFERENCE_KINDS = frozenset({
    "manual_lexical", "existing_gold_lexical", "constructed_exact_lexical",
})
AUDIO_SPLICE = "audio_splice"


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


def set_label(purpose: str, generation: int | None = None) -> str:
    """Storage label for one rendered set.

    A gate generation gets its own label so a fresh gate can never read the audio
    or the cached candidate tables of the generation it replaces
    (`csasr.lss.align.exposure`). The scored `purpose` stays `gate`, because that
    is what Gate A judges; only the storage identity is versioned.
    """
    return str(purpose) if generation in (None, 0) else f"{purpose}_g{int(generation)}"


def build_set(manifest: pd.DataFrame, cfg: Mapping[str, Any], *, n_pairs: int,
              seed: int, out_dir: Path, purpose: str,
              sources: pd.DataFrame | None = None,
              generation: int | None = None
              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render `n_pairs` ZH+EN splices with known audio seams.

    ``sources`` is the pre-partitioned pool for this purpose; pass it (via
    `partition_sources`) whenever more than one purpose is built, or the two
    sets will overlap.
    """
    scfg = dict(cfg.get("synthetic") or {})
    label = set_label(purpose, generation)
    out_dir = Path(out_dir) / label
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
                "purpose": purpose, "generation": generation,
                "seed": int(seed)}
    fingerprint = synthetic_harness_fingerprint(pairs, settings)

    rendered = []
    for pair in pairs:
        # pair ids restart at syn_0000 for every purpose, so they are namespaced
        # here; otherwise dev and gate items collide in any joined table, and two
        # gate generations collide with each other
        pair = {**pair, "pair_id": f"{label}_{pair['pair_id']}"}
        target = out_dir / f"{pair['pair_id']}.wav"
        rendered.append(render_synthetic_pair(
            pair, target, trim_silence=bool(scfg.get("trim_silence", True)),
            pad_ms=float(scfg.get("trim_pad_ms", 20.0))))
    frame = pd.DataFrame(rendered)
    frame["utterance_id"] = frame["pair_id"]
    frame["purpose"] = purpose
    # RMS/VAD trimming determines where the waveforms are concatenated.  That
    # sample is exactly known as an audio seam, but neither source clip carries
    # an independently annotated lexical word edge.  Calling it lexical truth
    # would turn silence trimming into an accuracy oracle.
    frame["reference_kind"] = AUDIO_SPLICE
    frame["reference_semantics"] = "known_audio_seam_not_lexical_boundary"
    frame["set_label"] = label
    frame["generation"] = 0 if generation is None else int(generation)
    return frame, {"purpose": purpose, "generation": generation,
                   "set_label": label, "pairs": int(len(frame)),
                   "out_dir": str(out_dir),
                   "fingerprint": fingerprint["sha256"], "settings": settings,
                   "source_utterances": sorted(source_ids(frame))}


#: schema declared by every row `score_family` emits. Gate A refuses a table
#: that does not declare it, so an older or hand-made file cannot be read as if
#: it meant the same thing.
SCORES_SCHEMA_VERSION = "lss_reference_scores_v2"
ITEMS_SCHEMA_VERSION = "lss_reference_boundary_items_v2"

#: which boundary each row describes. `combined` is the worse of the two edges
#: for a unit, and is what a steering mask actually depends on.
EDGES = ("start", "end", "combined")

#: The convention the pipeline actually consumes: a steering mask is built from
#: reference-unit edges, so the boundary a family predicts is the end of the last
#: Mandarin unit and the start of the first English unit.
CANONICAL_CONVENTION = "unit_edge_canonical"

#: E1's convention: one instant per switch, the midpoint between the two units.
#: Scored on the same audio so the two can be compared, but kept out of the
#: gate's score table -- see `score_rendered_set`.
LEGACY_CONVENTION = "midpoint_legacy"


def aligner_manifest(rendered: pd.DataFrame) -> pd.DataFrame:
    """A manifest the aligner families can consume, from rendered splices.

    `build_synthetic_pairs` admits only single-script sources, so the
    concatenated transcript segments into Mandarin units followed by English
    units and the constructed instant is the seam between them. That is what
    makes a *unit-edge* prediction comparable to the construction: no unit
    straddles the boundary.

    `conversation_id` and `split` are labelled synthetic rather than left empty,
    so a synthetic candidate row can never be mistaken for a corpus row if the
    two ever land in the same table.
    """
    if rendered is None or not len(rendered):
        return pd.DataFrame()
    out = pd.DataFrame({
        "utterance_id": rendered["pair_id"].astype(str),
        "audio_path": rendered["audio_path"].astype(str),
        # the aligner segments this into reference units itself
        "transcript_raw": (rendered["zh_text"].astype(str).str.strip() + " "
                           + rendered["en_text"].astype(str).str.strip()),
        "duration_sec": rendered["duration_sec"].astype(float),
        "conversation_id": "synthetic",
        "split": "synthetic_" + rendered["purpose"].astype(str)
        if "purpose" in rendered else "synthetic",
        "contains_code_switch": True,
    })
    return out.reset_index(drop=True)


#: why a rendered pair could not be scored for a family
UNSCORABLE = {
    "no_candidates": "the family produced no rows for this item",
    "no_valid_zh_unit": "no valid Mandarin unit, so no end-of-Mandarin estimate",
    "no_valid_en_unit": "no valid English unit, so no start-of-English estimate",
}


def boundary_predictions(rendered: pd.DataFrame, candidates: pd.DataFrame
                         ) -> pd.DataFrame:
    """One row per (item, family, variant): predicted and typed reference edges.

    The prediction is read off the *reference units*, which is what a steering
    mask is built from:

    ``end``    the end of the last valid Mandarin unit, against ``zh_end_sec``
    ``start``  the start of the first valid English unit, against ``en_start_sec``

    With the default zero gap those two truths are the same instant; with a gap
    they bracket it, and each edge is scored against the instant it actually
    predicts rather than against the midpoint. The legacy midpoint estimate
    ``(end + start) / 2`` is scored against ``true_boundary_sec``, which is the
    midpoint convention `switch_boundaries` uses.

    A family that produced no valid unit on one side of the seam is *not* scored
    as a large error -- it made no prediction, and charging it for one would read
    as an accuracy failure instead of missing coverage. Such items are returned
    with ``scorable=False`` and a reason.
    """
    if rendered is None or not len(rendered) or candidates is None or not len(candidates):
        return pd.DataFrame()

    truth = rendered.set_index(rendered["pair_id"].astype(str))
    valid = candidates[candidates["is_valid"].astype(bool)] \
        if "is_valid" in candidates else candidates

    rows: list[dict[str, Any]] = []
    group_cols = ["aligner_family"]
    if "aligner_variant" in candidates.columns:
        group_cols.append("aligner_variant")

    for keys, family_rows in candidates.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        family = str(keys[0])
        variant = str(keys[1]) if len(keys) > 1 else family
        family_valid = valid[valid["aligner_family"].astype(str) == family]
        if "aligner_variant" in valid.columns:
            family_valid = family_valid[
                family_valid["aligner_variant"].astype(str) == variant]
        by_item = dict(tuple(family_valid.groupby(
            family_valid["utterance_id"].astype(str), sort=False)))

        for item_id in truth.index:
            item = truth.loc[item_id]
            reference_kind = str(item.get("reference_kind", AUDIO_SPLICE))
            if reference_kind not in REFERENCE_KINDS:
                raise ValueError(f"unsupported reference_kind={reference_kind!r}")
            base = {"pair_id": str(item_id), "family": family,
                    "variant": variant,
                    "reference_kind": reference_kind,
                    "reference_semantics": str(item.get(
                        "reference_semantics", reference_kind)),
                    "true_boundary_sec": float(item["true_boundary_sec"]),
                    "true_zh_end_sec": float(item["zh_end_sec"]),
                    "true_en_start_sec": float(item["en_start_sec"])}
            got = by_item.get(str(item_id))
            if got is None or not len(got):
                rows.append({**base, "scorable": False, "reason": "no_candidates"})
                continue
            zh = got[got["reference_language"].astype(str) == "ZH"]
            en = got[got["reference_language"].astype(str) == "EN"]
            if not len(zh):
                rows.append({**base, "scorable": False,
                             "reason": "no_valid_zh_unit"})
                continue
            if not len(en):
                rows.append({**base, "scorable": False,
                             "reason": "no_valid_en_unit"})
                continue
            last_zh = zh.loc[zh["reference_unit_index"].astype(int).idxmax()]
            first_en = en.loc[en["reference_unit_index"].astype(int).idxmin()]
            predicted_end = float(last_zh["end_sec"])
            predicted_start = float(first_en["start_sec"])
            rows.append({
                **base,
                "scorable": True,
                "reason": "",
                "predicted_end_sec": predicted_end,
                "predicted_start_sec": predicted_start,
                "predicted_midpoint_sec": (predicted_end + predicted_start) / 2.0,
                "last_zh_unit_index": int(last_zh["reference_unit_index"]),
                "first_en_unit_index": int(first_en["reference_unit_index"]),
            })
    return pd.DataFrame(rows)


#: predicted/truth column pair for each (convention, edge) that gets a score row
_SCORED_EDGES: tuple[tuple[str, str, str, str], ...] = (
    (CANONICAL_CONVENTION, "end", "predicted_end_sec", "true_zh_end_sec"),
    (CANONICAL_CONVENTION, "start", "predicted_start_sec", "true_en_start_sec"),
    (LEGACY_CONVENTION, "combined", "predicted_midpoint_sec", "true_boundary_sec"),
)


def score_rendered_set(rendered: pd.DataFrame, candidates: pd.DataFrame, *,
                       purpose: str = "gate"
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every family's boundary predictions. Returns (scores, per-item).

    Two tables, because they answer different questions and only one of them is
    evidence: `scores` is what Gate A reads, `per_item` is what a reviewer reads
    to find out why.

    **Only the canonical convention goes into `scores`.** Gate A takes the worst
    case over every row in that table, so including the legacy midpoint rows
    would judge the pipeline on a convention nothing downstream consumes -- and
    E1 measured that convention roughly 490 ms out, which would fail the bias
    criterion for a reason unrelated to the spans steering uses. The legacy rows
    are still computed on the same audio and returned in `per_item`, and
    `convention_comparison` summarises them, so the comparison is available
    without being mistaken for the gate's own measurement.

    ``combined`` is the canonical convention's worse edge, per item: for each
    item the edge with the larger absolute error is taken, keeping its signed
    error, because a steering mask is wrong at whichever end is wrong.
    """
    per_item = boundary_predictions(rendered, candidates)
    if not len(per_item):
        return pd.DataFrame(), per_item
    per_item = per_item.copy()
    per_item["schema_version"] = ITEMS_SCHEMA_VERSION
    per_item["purpose"] = str(purpose)

    scorable = per_item[per_item["scorable"].astype(bool)].copy()
    if not len(scorable):
        return pd.DataFrame(), per_item
    rows: list[dict[str, Any]] = []
    scorable["zh_end_minus_splice_ms"] = (
        scorable["predicted_end_sec"] - scorable["true_zh_end_sec"]) * 1000.0
    scorable["en_start_minus_splice_ms"] = (
        scorable["predicted_start_sec"] - scorable["true_en_start_sec"]) * 1000.0
    scorable["gap_around_splice_ms"] = (
        scorable["predicted_start_sec"] - scorable["predicted_end_sec"]) * 1000.0
    scorable["absolute_splice_edge_offset_ms"] = scorable[
        ["zh_end_minus_splice_ms", "en_start_minus_splice_ms"]].abs().max(axis=1)

    exact = scorable[scorable["reference_kind"].isin(LEXICAL_REFERENCE_KINDS)].copy()
    if len(exact):
        exact["signed_end_error_ms"] = exact["zh_end_minus_splice_ms"]
        exact["signed_start_error_ms"] = exact["en_start_minus_splice_ms"]
        exact["absolute_end_error_ms"] = exact["signed_end_error_ms"].abs()
        exact["absolute_start_error_ms"] = exact["signed_start_error_ms"].abs()
        exact["absolute_boundary_error_ms"] = exact[
            ["absolute_start_error_ms", "absolute_end_error_ms"]].max(axis=1)

    for (family, reference_kind), group in scorable.groupby(
            ["family", "reference_kind"], sort=True):
        if str(reference_kind) in LEXICAL_REFERENCE_KINDS:
            exact_group = exact[exact.index.isin(group.index)]
            for edge, signed_col, absolute_col in (
                    ("start", "signed_start_error_ms", "absolute_start_error_ms"),
                    ("end", "signed_end_error_ms", "absolute_end_error_ms"),
                    ("combined", None, "absolute_boundary_error_ms")):
                absolute = exact_group[absolute_col].to_numpy(dtype=float)
                if signed_col is None:
                    take_end = (exact_group["absolute_end_error_ms"]
                                >= exact_group["absolute_start_error_ms"])
                    signed = np.where(take_end, exact_group["signed_end_error_ms"],
                                      exact_group["signed_start_error_ms"])
                else:
                    signed = exact_group[signed_col].to_numpy(dtype=float)
                rows.append(_absolute_score_row(
                    absolute, np.asarray(signed, dtype=float), family=str(family),
                    reference_kind=str(reference_kind), edge=edge, purpose=purpose))
        else:
            edge_offset = group["absolute_splice_edge_offset_ms"].to_numpy(dtype=float)
            rows.append({
                "schema_version": SCORES_SCHEMA_VERSION,
                "family": str(family), "purpose": str(purpose),
                "convention": "audio_splice", "edge": "seam",
                "reference_kind": str(reference_kind),
                "metric_semantics": "audio_seam_relative_not_lexical_accuracy",
                "num_boundaries": int(len(group)),
                "median_zh_end_minus_splice_ms": float(
                    group["zh_end_minus_splice_ms"].median()),
                "median_en_start_minus_splice_ms": float(
                    group["en_start_minus_splice_ms"].median()),
                "median_gap_around_splice_ms": float(
                    group["gap_around_splice_ms"].median()),
                "median_absolute_splice_edge_offset_ms": float(np.median(edge_offset)),
                "p90_absolute_splice_edge_offset_ms": float(np.percentile(edge_offset, 90)),
                "within_100ms_of_splice": float((edge_offset <= 100.0).mean()),
            })
    # Add derived columns only to exact-reference items.  Audio-splice outputs
    # must not serialize lexical absolute-error fields at all.
    if len(exact) and len(exact) == len(scorable):
        for column in ("signed_start_error_ms", "signed_end_error_ms",
                       "absolute_start_error_ms", "absolute_end_error_ms",
                       "absolute_boundary_error_ms"):
            per_item.loc[exact.index, column] = exact[column]
    for column in ("zh_end_minus_splice_ms", "en_start_minus_splice_ms",
                   "gap_around_splice_ms", "absolute_splice_edge_offset_ms"):
        per_item.loc[scorable.index, column] = scorable[column]
    return scores_table(rows), per_item


def convention_comparison(per_item: pd.DataFrame) -> pd.DataFrame:
    """Canonical vs legacy offsets on the same reference, per family.

    This is the diagnostic the "coordinate bug" hypothesis needs: a constant
    offset that appears under the midpoint convention and not under unit edges
    is a convention artefact, not an acoustic failure.
    """
    if per_item is None or not len(per_item):
        return pd.DataFrame()
    scorable = per_item[per_item["scorable"].astype(bool)]
    if not len(scorable):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (family, purpose, reference_kind), group in scorable.groupby(
            ["family", "purpose", "reference_kind"], sort=True):
        for convention, edge, pred_col, truth_col in _SCORED_EDGES:
            signed = (group[pred_col] - group[truth_col]) * 1000.0
            rows.append({
                "family": str(family), "purpose": str(purpose),
                "reference_kind": str(reference_kind),
                "metric_semantics": ("absolute_lexical_boundary_accuracy"
                                     if reference_kind in LEXICAL_REFERENCE_KINDS
                                     else "audio_seam_relative_not_lexical_accuracy"),
                "convention": convention, "edge": edge,
                "n": int(len(group)),
                "median_signed_offset_ms": float(signed.median()),
                "median_absolute_offset_ms": float(signed.abs().median()),
                "p90_absolute_offset_ms": float(np.percentile(signed.abs(), 90)),
            })
    return pd.DataFrame(rows)


def unscorable_summary(per_item: pd.DataFrame) -> dict[str, Any]:
    """How many items each family could not predict a seam for, and why."""
    if per_item is None or not len(per_item):
        return {}
    out: dict[str, Any] = {}
    for family, group in per_item.groupby("family", sort=True):
        missed = group[~group["scorable"].astype(bool)]
        out[str(family)] = {
            "items": int(len(group)),
            "scored": int(len(group) - len(missed)),
            "unscorable": int(len(missed)),
            "reasons": missed["reason"].value_counts().to_dict() if len(missed) else {},
        }
    return out


def _absolute_score_row(absolute_ms: np.ndarray, signed_ms: np.ndarray, *,
                        family: str, reference_kind: str, edge: str,
                        purpose: str) -> dict[str, Any]:
    if reference_kind not in LEXICAL_REFERENCE_KINDS:
        raise ValueError(f"{reference_kind!r} cannot support lexical absolute error")
    absolute_ms = np.asarray(absolute_ms, dtype=float)
    signed_ms = np.asarray(signed_ms, dtype=float)
    ok = np.isfinite(absolute_ms) & np.isfinite(signed_ms)
    absolute_ms, signed_ms = absolute_ms[ok], signed_ms[ok]
    return {
        "schema_version": SCORES_SCHEMA_VERSION,
        "family": str(family),
        "convention": CANONICAL_CONVENTION,
        "edge": str(edge),
        "purpose": str(purpose),
        "reference_kind": str(reference_kind),
        "metric_semantics": "absolute_lexical_boundary_accuracy",
        "num_boundaries": int(len(absolute_ms)),
        "within_50ms": float((absolute_ms <= 50.0).mean()) if len(absolute_ms) else float("nan"),
        "within_100ms": float((absolute_ms <= 100.0).mean()) if len(absolute_ms) else float("nan"),
        "p90_abs_error_ms": float(np.percentile(absolute_ms, 90)) if len(absolute_ms) else float("nan"),
        "median_abs_error_ms": float(np.median(absolute_ms)) if len(absolute_ms) else float("nan"),
        "median_signed_error_ms": float(np.median(signed_ms)) if len(signed_ms) else float("nan"),
    }


def score_family(predicted_sec: Sequence[float], truth: Sequence[float], *,
                 family: str, convention: str = CANONICAL_CONVENTION,
                 edge: str = "combined", purpose: str = "gate",
                 reference_kind: str = "constructed_exact_lexical") -> dict[str, Any]:
    """Score data that explicitly declares genuine lexical reference edges."""
    if convention != CANONICAL_CONVENTION:
        raise ValueError("only the canonical unit-edge convention is gate evidence")
    predicted = np.asarray(predicted_sec, dtype=float)
    truth = np.asarray(truth, dtype=float)
    signed = (predicted - truth) * 1000.0
    return _absolute_score_row(np.abs(signed), signed, family=family,
                               reference_kind=reference_kind, edge=edge,
                               purpose=purpose)


def scores_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble reference-aware score rows without coercing their semantics."""
    frame = pd.DataFrame(list(rows))
    if not len(frame):
        return frame
    required = ("schema_version", "family", "purpose", "reference_kind",
                "metric_semantics", "num_boundaries")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"synthetic score rows are missing {missing}; use "
                         "`score_family` to build them")
    return frame


def synthetic_table(scored: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(scored))


def estimate_edge_offsets(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-family lexical signed error; seam-relative rows are never fitted."""
    if not len(scored):
        return pd.DataFrame()
    if "reference_kind" not in scored or "median_signed_error_ms" not in scored:
        return pd.DataFrame()
    scored = scored[scored["reference_kind"].isin(LEXICAL_REFERENCE_KINDS)]
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
    """Fit correction offsets to genuine external lexical truth only."""
    lexical = (scored[scored["reference_kind"].isin(LEXICAL_REFERENCE_KINDS)]
               if len(scored) and "reference_kind" in scored else pd.DataFrame())
    table = estimate_edge_offsets(lexical)
    values = {f"{row['family']}": float(row["offset_ms"]) for _, row in table.iterrows()} \
        if len(table) else {}
    return EdgeOffsets(values=values, source=source,
                       n=int(lexical["num_boundaries"].sum())
                       if "num_boundaries" in lexical else 0)


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
