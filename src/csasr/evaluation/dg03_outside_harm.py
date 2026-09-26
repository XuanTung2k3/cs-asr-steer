"""Offline DG-03 outside-harm reconstruction.

This module deliberately contains no decoding or selection logic.  It rebuilds
the frozen D-dev-select target population from the existing ``existing_ctc``
alignment rows and applies the canonical correctness-flip accounting from
``csasr.lss.outcomes`` to the transcripts already stored by the DG-03 screen.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .normalization import normalize_text, segment_units
from ..lss.outcomes import CandidateUnitSets, newly_introduced_errors, utility

MIN_OVERLAP_RATIO = 0.5


def _overlap_ratio(start: float, end: float, cand_start: float,
                   cand_end: float) -> float:
    duration = max(float(end) - float(start), 1e-9)
    return max(0.0, min(float(end), float(cand_end))
                - max(float(start), float(cand_start))) / duration


def population_unit_sets(
    reference: str,
    trusted_units: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]],
    *,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> CandidateUnitSets:
    """Return one canonical inside/outside partition for an utterance.

    ``trusted_units`` are the frozen alignment units, and ``target_candidates``
    are the frozen eligible embedded-English acoustic candidates.  The inside
    population is the union of units covered by those candidates; trusted units
    outside that union are outside.  Reference units without a trusted
    alignment remain unknown and are never silently counted as outside.
    """
    ref_ids = set(range(len(segment_units(normalize_text(reference)))))
    known: set[int] = set()
    inside: set[int] = set()
    languages: dict[int, str] = {}
    intervals: dict[int, tuple[float, float]] = {}
    for row in trusted_units:
        uid = int(row["unit_id"])
        start, end = float(row["start_sample"]), float(row["end_sample"])
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            continue
        known.add(uid)
        intervals[uid] = (start, end)
        languages[uid] = str(row.get("language", ""))

    for candidate in target_candidates:
        cstart = float(candidate["start_sample"])
        cend = float(candidate["end_sample"])
        for uid, (start, end) in intervals.items():
            if _overlap_ratio(start, end, cstart, cend) >= min_overlap_ratio:
                inside.add(uid)

    outside = known - inside
    unknown = ref_ids - known
    ratios = {
        uid: max(0.0, min(end, float(c["end_sample"]))
                 - max(start, float(c["start_sample"])))
        / max(end - start, 1e-9)
        for uid, (start, end) in intervals.items()
        for c in target_candidates
        if _overlap_ratio(start, end, float(c["start_sample"]),
                          float(c["end_sample"])) >= min_overlap_ratio
    }
    return CandidateUnitSets(
        inside=tuple(sorted(inside)),
        outside=tuple(sorted(outside)),
        unknown=tuple(sorted(unknown)),
        overlap_ratios=ratios,
    ), languages


def corpus_outside_harm(
    references: Sequence[str],
    baseline: Sequence[str],
    method: Sequence[str],
    unit_sets: Sequence[CandidateUnitSets],
    languages: Sequence[Mapping[int, str]],
) -> dict[str, Any]:
    """Aggregate canonical candidate outcomes without changing transcripts."""
    totals = {
        "n_corrected_inside": 0,
        "n_corrupted_inside": 0,
        "n_corrupted_outside": 0,
        "n_new_insertions_inside": 0,
        "n_new_insertions_outside": 0,
        "n_units_inside": 0,
        "n_units_outside": 0,
        "n_units_unknown": 0,
        "n_utterances": 0,
    }
    for reference, base, steered, sets, lang in zip(
        references, baseline, method, unit_sets, languages
    ):
        outcome = newly_introduced_errors(reference, base, steered, sets,
                                           languages=lang)
        for key in totals:
            if key == "n_utterances":
                continue
            totals[key] += int(outcome.get(key, 0))
        totals["n_utterances"] += 1
    totals["outside_harm"] = int(totals["n_corrupted_outside"])
    totals["utility"] = utility(totals)
    totals["count_insertions_as_harm"] = False
    totals["outside_harm_definition"] = (
        "baseline-correct to method-wrong trusted reference units outside the "
        "union of frozen eligible embedded-English target candidates; "
        "insertions are excluded per csasr.lss.outcomes.COUNT_INSERTIONS_AS_HARM"
    )
    return totals


def baseline_outside_harm(unit_sets: Sequence[CandidateUnitSets],
                          references: Sequence[str],
                          baseline: Sequence[str],
                          languages: Sequence[Mapping[int, str]]) -> dict[str, Any]:
    """The canonical zero-change baseline record (always zero harm)."""
    return corpus_outside_harm(references, baseline, baseline, unit_sets, languages)
