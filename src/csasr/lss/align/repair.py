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
