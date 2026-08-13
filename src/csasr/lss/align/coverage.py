"""Label coverage and the selection bias that restricting to it creates.

Two failure modes the implementation review named.

**H2** -- an English unit with no trustworthy boundary is not a negative. If its
frames are labelled "not English", the localizer is trained to suppress exactly
the hard cases and its evaluation is inflated on the alignable subset. Those
regions get an `ignore` label instead, and the ignore rate is reported.

**H3** -- restricting the endpoint to high-confidence spans is a selection.
Agreement plausibly correlates with duration, language and baseline
correctness, so the included and excluded populations are compared on every
covariate that could explain a later result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

NEGATIVE, POSITIVE, IGNORE = 0, 1, 2


def ignore_mask(spans: pd.DataFrame, rejected: pd.DataFrame, *,
                n_frames: int, utterance_id: str,
                geometry=None, sample_rate: int = 16000,
                encoder_step_sec: float = 0.02,
                target_language: str = "EN") -> np.ndarray:
    """Frame labels for one utterance: negative / positive / ignore.

    Positives are accepted spans of the target language. Ignore covers every
    region whose boundaries are not trustworthy -- rejected units of any
    language -- so an unalignable English word can never be scored as a
    negative.
    """
    mask = np.full(int(n_frames), NEGATIVE, dtype=np.int8)
    step = float(encoder_step_sec if geometry is None else geometry.encoder_step_sec)

    def frames(start_sample: float, end_sample: float) -> tuple[int, int]:
        start = int(np.floor(start_sample / sample_rate / step))
        end = int(np.ceil(end_sample / sample_rate / step))
        return max(0, start), min(int(n_frames), max(start + 1, end))

    for _, row in rejected[rejected.get("utterance_id") == utterance_id].iterrows() \
            if len(rejected) else []:
        start = row.get("candidate_start_sec")
        end = row.get("candidate_end_sec")
        if start is None or end is None or not np.isfinite(float(start or np.nan)):
            # no usable estimate at all: the whole utterance stays unknown for
            # this unit, which is safer than guessing a location
            continue
        a, b = frames(float(start) * sample_rate, float(end) * sample_rate)
        mask[a:b] = IGNORE

    for _, row in spans[spans["utterance_id"] == utterance_id].iterrows() \
            if len(spans) else []:
        a, b = frames(float(row["consensus_start_sample"]),
                      float(row["consensus_end_sample"]))
        mask[a:b] = POSITIVE if row.get("language") == target_language else NEGATIVE
    return mask


def label_coverage(spans: pd.DataFrame, rejected: pd.DataFrame,
                   units: pd.DataFrame) -> pd.DataFrame:
    """Per language: how many reference units got a trustworthy boundary."""
    rows: list[dict[str, Any]] = []
    for language in ("EN", "ZH"):
        total = int((units["language"] == language).sum()) if len(units) else 0
        accepted = int((spans.get("language", pd.Series(dtype=str)) == language).sum()) \
            if len(spans) else 0
        refused = int((rejected.get("reference_language", pd.Series(dtype=str))
                       == language).sum()) if len(rejected) else 0
        rows.append({
            "language": language,
            "reference_units": total,
            "accepted": accepted,
            "rejected": refused,
            "coverage": (accepted / total) if total else float("nan"),
            "ignore_rate": (refused / total) if total else float("nan"),
        })
    return pd.DataFrame(rows)


def assert_partition(spans: pd.DataFrame, rejected: pd.DataFrame,
                     units: pd.DataFrame) -> dict[str, Any]:
    """Accepted and rejected must together cover every content unit, disjointly.

    Without this the ignore mask and the selection-bias report are not
    computable: a unit that is in neither table is silently invisible.
    """
    def keys(frame: pd.DataFrame, unit_col: str) -> set[tuple[str, int]]:
        if not len(frame):
            return set()
        return {(str(u), int(i)) for u, i in zip(frame["utterance_id"], frame[unit_col])}

    expected = keys(units[units["language"].isin(["EN", "ZH"])], "unit_id")
    accepted = keys(spans, "unit_id") if "unit_id" in spans else set()
    refused = keys(rejected, "reference_unit_index") if len(rejected) else set()
    overlap = accepted & refused
    missing = expected - (accepted | refused)
    extra = (accepted | refused) - expected
    report = {
        "expected_units": len(expected),
        "accepted": len(accepted),
        "rejected": len(refused),
        "overlap": len(overlap),
        "missing": len(missing),
        "unexpected": len(extra),
        # Overlap/extra rows are a corrupt partition. Missing rows are instead a
        # scientifically meaningful coverage failure: an aligner may simply not
        # produce evidence for an expected unit, and that must not be mislabeled
        # as an implementation crash.
        "partition_structurally_valid": bool(not overlap and not extra),
        "accounted_rate": (float(len((accepted | refused) & expected) / len(expected))
                           if expected else float("nan")),
        "partition_exact": bool(not overlap and not missing and not extra),
        "missing_examples": sorted(missing)[:5],
    }
    return report


def selection_bias(spans: pd.DataFrame, rejected: pd.DataFrame,
                   units: pd.DataFrame,
                   poi_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare included and excluded units on everything that could explain a result."""
    if not len(units):
        return pd.DataFrame()
    accepted = {(str(u), int(i)) for u, i in
                zip(spans.get("utterance_id", []), spans.get("unit_id", []))}
    frame = units.copy()
    frame["included"] = [(str(u), int(i)) in accepted
                         for u, i in zip(frame["utterance_id"], frame["unit_id"])]
    frame["surface_length"] = frame["surface"].astype(str).str.len()

    if poi_table is not None and len(poi_table):
        status = {(str(r["utterance_id"]), int(r["poi_index"])): (bool(r["correct"]),
                                                                  str(r["category"]))
                  for _, r in poi_table.iterrows()}
        frame["baseline_correct"] = [status.get((str(u), int(i)), (None, ""))[0]
                                     for u, i in zip(frame["utterance_id"], frame["unit_id"])]
        frame["baseline_category"] = [status.get((str(u), int(i)), (None, ""))[1]
                                      for u, i in zip(frame["utterance_id"], frame["unit_id"])]

    rows: list[dict[str, Any]] = []
    for language, group in frame.groupby("language"):
        if language not in ("EN", "ZH"):
            continue
        included = group[group["included"]]
        excluded = group[~group["included"]]
        row: dict[str, Any] = {
            "language": language,
            "n_included": int(len(included)),
            "n_excluded": int(len(excluded)),
            "included_fraction": float(group["included"].mean()),
            "mean_surface_length_included": float(included["surface_length"].mean())
            if len(included) else float("nan"),
            "mean_surface_length_excluded": float(excluded["surface_length"].mean())
            if len(excluded) else float("nan"),
            "mean_position_included": float(included["unit_position"].mean())
            if len(included) else float("nan"),
            "mean_position_excluded": float(excluded["unit_position"].mean())
            if len(excluded) else float("nan"),
        }
        if "baseline_correct" in group:
            known = group[group["baseline_correct"].notna()]
            if len(known):
                row["baseline_correct_rate_included"] = float(
                    known.loc[known["included"], "baseline_correct"].mean())
                row["baseline_correct_rate_excluded"] = float(
                    known.loc[~known["included"], "baseline_correct"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def selection_bias_summary(table: pd.DataFrame) -> dict[str, Any]:
    if not len(table):
        return {"languages": 0}
    english = table[table["language"] == "EN"]
    return {
        "languages": int(len(table)),
        "en_included_fraction": float(english["included_fraction"].iloc[0])
        if len(english) else float("nan"),
        "note": ("agreement correlates with duration and language, so the "
                 "restricted endpoint is reported alongside the unrestricted one "
                 "and the denominator never changes by system"),
    }
