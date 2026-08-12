"""Adjacent-overlap inventory and repair for candidate alignments.

Whisper-DTW's candidate table showed a 16.5% invalid rate (93 of 564 units) with
`adjacent_overlap` as the top failure code. Two things have to be measured
before that is called an alignment defect:

* `csasr.nat5h.schema.validate_candidate_sequence` flags any overlap beyond
  1 ms, but `sec_to_frames` floors starts and ceils ends, so a **one-frame
  (20 ms) overlap is an expected artifact of the frame grid**, not a real
  collision. If most flagged rows are one frame, the invalid rate is a property
  of the validator;
* `csasr.data.alignment.resolve_boundary_overlaps` only touches pairs whose
  language tags differ, so it structurally cannot repair same-language
  overlaps. Those must be reported, not silently left looking repaired.

A repair that buys validity by moving a boundary 200 ms is not a repair, so the
report always states how far edges actually moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlapRepairPolicy:
    enabled: bool = True
    policy: str = "truncate_earlier"
    same_language: str = "report_only"
    benign_overlap_frames: int = 1

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "OverlapRepairPolicy":
        block = dict((cfg or {}).get("adjacent_overlap") or {})
        return cls(
            enabled=bool(block.get("enabled", cls.enabled)),
            policy=str(block.get("policy", cls.policy)),
            same_language=str(block.get("same_language", cls.same_language)),
            benign_overlap_frames=int(block.get("benign_overlap_frames",
                                                cls.benign_overlap_frames)),
        )


def overlap_inventory(candidates: pd.DataFrame, *, sample_rate: int = 16000,
                      frame_sec: float = 0.02,
                      policy: OverlapRepairPolicy | None = None) -> pd.DataFrame:
    """Every adjacent-span overlap, with its size and whether it is benign."""
    policy = policy or OverlapRepairPolicy()
    benign_ms = policy.benign_overlap_frames * frame_sec * 1000.0
    rows: list[dict[str, Any]] = []
    # Filtering on `is_valid` would drop exactly the rows of interest: a span
    # flagged `adjacent_overlap` is marked invalid *because* it overlaps. What
    # must be excluded instead are rows with no usable timestamps at all, which
    # the schema encodes as -1.
    usable = candidates
    if len(usable):
        usable = usable[(usable["start_sample"].astype(float) >= 0)
                        & (usable["end_sample"].astype(float)
                           > usable["start_sample"].astype(float))]

    for (utt, family), group in usable.groupby(["utterance_id", "aligner_family"]):
        group = group.sort_values("reference_unit_index")
        previous = None
        for _, row in group.iterrows():
            if previous is not None:
                overlap_samples = int(previous["end_sample"]) - int(row["start_sample"])
                if overlap_samples > 0:
                    overlap_ms = overlap_samples / sample_rate * 1000.0
                    rows.append({
                        "utterance_id": utt,
                        "aligner_family": family,
                        "unit_id": int(row["reference_unit_index"]),
                        "neighbour_unit_id": int(previous["reference_unit_index"]),
                        "language": row.get("reference_language", ""),
                        "neighbour_language": previous.get("reference_language", ""),
                        "overlap_ms": float(overlap_ms),
                        "overlap_frames": float(overlap_ms / (frame_sec * 1000.0)),
                        "cross_language": bool(row.get("reference_language")
                                               != previous.get("reference_language")),
                        "benign": bool(overlap_ms <= benign_ms + 1e-9),
                    })
            previous = row
    return pd.DataFrame(rows, columns=[
        "utterance_id", "aligner_family", "unit_id", "neighbour_unit_id", "language",
        "neighbour_language", "overlap_ms", "overlap_frames", "cross_language", "benign"])


def repair_adjacent_overlaps(candidates: pd.DataFrame,
                             policy: OverlapRepairPolicy | None = None, *,
                             sample_rate: int = 16000
                             ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Truncate the earlier span of a cross-language overlap; report the rest.

    Mirrors `csasr.data.alignment.resolve_boundary_overlaps` on the schema-v2
    candidate table, and records the boundary movement it cost.
    """
    policy = policy or OverlapRepairPolicy()
    out = candidates.copy()
    if not policy.enabled or not len(out):
        return out, {"enabled": policy.enabled, "repaired": 0}

    moved: list[float] = []
    repaired = skipped_same_language = 0
    usable = out[(out["start_sample"].astype(float) >= 0)
                 & (out["end_sample"].astype(float) > out["start_sample"].astype(float))]
    for (_, _), group in usable.groupby(["utterance_id", "aligner_family"]):
        index = group.sort_values("reference_unit_index").index
        for a, b in zip(index, index[1:]):
            end_a = int(out.at[a, "end_sample"])
            start_b = int(out.at[b, "start_sample"])
            if end_a <= start_b:
                continue
            same_language = out.at[a, "reference_language"] == out.at[b, "reference_language"]
            if same_language and policy.same_language != "truncate_earlier":
                # Measured on this corpus every DTW overlap is ZH->ZH with a
                # median of 320 ms, so `report_only` leaves the whole 16.5%
                # invalid rate unrepaired. Enabling same-language truncation is
                # a scientific decision for the validation stage, not a default.
                skipped_same_language += 1
                continue
            new_end = max(int(out.at[a, "start_sample"]) + 1, start_b)
            moved.append((end_a - new_end) / sample_rate * 1000.0)
            out.at[a, "end_sample"] = new_end
            if "end_sec" in out.columns:
                out.at[a, "end_sec"] = new_end / sample_rate
            repaired += 1

    movement = np.asarray(moved, dtype=float)
    report = {
        "enabled": True,
        "policy": policy.policy,
        "repaired": int(repaired),
        "skipped_same_language": int(skipped_same_language),
        "median_boundary_movement_ms": float(np.median(movement)) if len(movement) else 0.0,
        "p90_boundary_movement_ms": float(np.percentile(movement, 90)) if len(movement) else 0.0,
        "max_boundary_movement_ms": float(movement.max()) if len(movement) else 0.0,
    }
    return out, report


#: Overlap causes, most specific first. `shared_source_token` is the one that
#: matters: it is not an alignment error at all.
OVERLAP_CAUSES = ("shared_source_token", "frame_grid_artifact", "partial_overlap")


def attribute_overlaps(candidates: pd.DataFrame, *, sample_rate: int = 16000,
                       frame_sec: float = 0.02,
                       family: str = "whisper_dtw") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Why a family's adjacent spans overlap, distinguished by their geometry.

    The finding this exists to record, measured on job 38573's 88,246 Whisper-DTW
    candidates over 1,800 utterances: **15,112 rows (17.1%) are flagged
    `adjacent_overlap`, all of them Mandarin, and in 100.0% of them the two spans
    are byte-identical** -- same start, same end. The overlap equals the unit's own
    duration exactly (mean 391.747 ms on both), which is only possible if the two
    reference units were assigned the same token span.

    That is exactly what happens: Whisper's BPE emits one token for several Han
    characters ("他们"), `data.alignment.align_batch` maps a unit to every token
    whose character range it intersects, and both characters therefore inherit the
    same `tok_start`/`tok_end`. There is no sub-token resolution to recover, so
    Whisper-DTW *cannot* place an independent boundary between them.

    Which is why nothing here repairs it. Truncating the earlier span would invent
    a boundary inside one token from no acoustic evidence, and it would buy exactly
    the coverage number the 0.95 threshold exists to measure. The honest reading is
    that these units have no second opinion from this family, which is what the
    invalid flag already says.
    """
    empty = {"family": family, "overlaps": 0, "causes": {},
             "identical_span_fraction": float("nan"),
             "cross_language_fraction": float("nan"),
             "note": "no overlapping spans"}
    if candidates is None or not len(candidates):
        return pd.DataFrame(), empty
    frame = candidates[candidates["aligner_family"].astype(str) == str(family)]
    frame = frame[(frame["start_sample"].astype(float) >= 0)
                  & (frame["end_sample"].astype(float)
                     > frame["start_sample"].astype(float))]
    if not len(frame):
        return pd.DataFrame(), empty

    frame = frame.sort_values(["utterance_id", "reference_unit_index"])
    grouped = frame.groupby("utterance_id", sort=False)
    previous_end = grouped["end_sample"].shift(1)
    previous_start = grouped["start_sample"].shift(1)
    previous_index = grouped["reference_unit_index"].shift(1)
    previous_text = grouped["reference_text"].shift(1) \
        if "reference_text" in frame.columns else previous_index
    previous_language = grouped["reference_language"].shift(1) \
        if "reference_language" in frame.columns else previous_index

    overlap = previous_end.astype(float) - frame["start_sample"].astype(float)
    mask = overlap > 0
    if not bool(mask.any()):
        return pd.DataFrame(), empty

    rows = frame[mask].copy()
    rows["previous_unit_index"] = previous_index[mask]
    rows["previous_reference_text"] = previous_text[mask]
    rows["previous_language"] = previous_language[mask]
    rows["overlap_ms"] = overlap[mask] / sample_rate * 1000.0
    rows["duration_ms"] = ((rows["end_sample"].astype(float)
                            - rows["start_sample"].astype(float))
                           / sample_rate * 1000.0)
    rows["identical_span"] = (
        np.isclose(previous_end[mask].astype(float),
                   rows["end_sample"].astype(float))
        & np.isclose(previous_start[mask].astype(float),
                     rows["start_sample"].astype(float)))
    rows["cross_language"] = (rows["previous_language"].astype(str)
                              != rows["reference_language"].astype(str))
    frame_ms = frame_sec * 1000.0
    rows["cause"] = np.where(
        rows["identical_span"], "shared_source_token",
        np.where(rows["overlap_ms"] <= frame_ms + 1e-9, "frame_grid_artifact",
                 "partial_overlap"))

    keep = ["utterance_id", "reference_unit_index", "previous_unit_index",
            "reference_text", "previous_reference_text", "reference_language",
            "previous_language", "overlap_ms", "duration_ms", "identical_span",
            "cross_language", "cause"]
    table = rows[[c for c in keep if c in rows.columns]].reset_index(drop=True)
    report = {
        "family": family,
        "overlaps": int(len(table)),
        "candidates": int(len(frame)),
        "overlap_rate": float(len(table) / len(frame)),
        "causes": {str(k): int(v) for k, v in table["cause"].value_counts().items()},
        "identical_span_fraction": float(table["identical_span"].mean()),
        "cross_language_fraction": float(table["cross_language"].mean()),
        "median_overlap_ms": float(table["overlap_ms"].median()),
        "by_language": {str(k): int(v) for k, v in
                        table["reference_language"].value_counts().items()}
        if "reference_language" in table else {},
        "note": ("`shared_source_token` means two reference units were assigned "
                 "the same Whisper token span, so this aligner has no sub-token "
                 "resolution for them. It is not repairable by truncation: that "
                 "would invent a boundary inside one token to buy the coverage "
                 "number the threshold exists to measure."),
    }
    return table, report


def repair_delta_report(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    """Validity before and after repair, per family and language."""
    def summarize(frame: pd.DataFrame) -> dict[str, float]:
        if not len(frame):
            return {}
        grouped = frame.groupby(["aligner_family", "reference_language"])
        return {f"{fam}/{lang}": float(sub.get("is_valid", pd.Series([True] * len(sub))).mean())
                for (fam, lang), sub in grouped}

    return {
        "valid_rate_before": summarize(before),
        "valid_rate_after": summarize(after),
        "rows": int(len(after)),
    }
