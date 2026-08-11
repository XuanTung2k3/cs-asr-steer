"""Candidate-level intervention outcomes: what was corrected, what was broken.

Three defects in the existing accounting are fixed here.

1. `evaluation.correction_harm.outside_region_edits` counts transcript
   *differences* outside a fixed +/-1 reference-token radius. A change is not an
   error: an intervention that also fixes a neighbouring word is booked as harm.
   Here an outcome is a **correctness flip** measured against the reference, so
   only correct -> incorrect counts as harm.
2. A candidate is an acoustic region, not a token. It can cover several
   reference units (or none, when the localizer fires on Mandarin), so
   `N_corrected` is not confined to {0, 1} and a candidate has no single
   `baseline_correct` value. Units are assigned to a candidate by acoustic
   overlap, and every count is over sets.
3. Candidate descriptions are independent booleans, not one categorical label:
   a candidate can be an English error *and* cause outside harm at the same
   time, so one `candidate_class` column cannot represent it.

Units whose boundaries are not trustworthy are assigned to `unknown` rather
than to `outside`, so a missing alignment can never masquerade as evidence that
the intervention was safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ..data.language_tags import EN, ZH
from ..evaluation.mer import INS, align_tokens
from ..evaluation.normalization import normalize_text, segment_units
from ..evaluation.pier import unit_status

#: how much of a reference unit must lie inside the candidate to count as inside
DEFAULT_MIN_OVERLAP_RATIO = 0.5

#: insertions are reported but do not enter the primary utility, which is a
#: unit-flip quantity; the rule is recorded in the spec freeze
COUNT_INSERTIONS_AS_HARM = False


@dataclass(frozen=True)
class CandidateUnitSets:
    """Reference units partitioned by their relation to one acoustic candidate."""

    inside: tuple[int, ...] = ()
    outside: tuple[int, ...] = ()
    unknown: tuple[int, ...] = ()
    overlap_ratios: Mapping[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inside": list(self.inside),
            "outside": list(self.outside),
            "unknown": list(self.unknown),
            "n_inside": len(self.inside),
            "n_outside": len(self.outside),
            "n_unknown": len(self.unknown),
        }


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def candidate_unit_sets(spans: pd.DataFrame, start_sample: int, end_sample: int, *,
                        min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
                        unit_col: str = "unit_id",
                        start_col: str = "consensus_start_sample",
                        end_col: str = "consensus_end_sample",
                        all_unit_ids: Sequence[int] | None = None
                        ) -> CandidateUnitSets:
    """Assign reference units to inside / outside / unknown for one candidate.

    ``spans`` holds the units of a single utterance whose boundaries are
    trusted. ``all_unit_ids`` (optional) is every reference unit of that
    utterance; any id missing from ``spans`` becomes ``unknown``.
    """
    inside: list[int] = []
    outside: list[int] = []
    ratios: dict[int, float] = {}
    known: set[int] = set()

    for _, row in spans.iterrows():
        uid = int(row[unit_col])
        known.add(uid)
        u_start, u_end = float(row[start_col]), float(row[end_col])
        duration = max(u_end - u_start, 1e-9)
        ratio = _overlap(u_start, u_end, float(start_sample), float(end_sample)) / duration
        ratios[uid] = float(ratio)
        (inside if ratio >= min_overlap_ratio else outside).append(uid)

    unknown = [int(u) for u in (all_unit_ids or []) if int(u) not in known]
    return CandidateUnitSets(tuple(sorted(inside)), tuple(sorted(outside)),
                             tuple(sorted(unknown)), ratios)


def attribute_insertions(reference: str, hypothesis: str) -> dict[int, list[str]]:
    """Inserted hypothesis tokens, attributed to the preceding reference index.

    A leading insertion is attributed to index -1. This is the rule used for
    every insertion count so that "inside the candidate" is well defined for
    tokens that have no reference position of their own.
    """
    ref = [u.surface for u in segment_units(normalize_text(reference))]
    hyp = [u.surface for u in segment_units(normalize_text(hypothesis))]
    out: dict[int, list[str]] = {}
    last_ref = -1
    for op in align_tokens(ref, hyp):
        if op.ref_idx is not None:
            last_ref = op.ref_idx
        elif op.op == INS and op.hyp_idx is not None:
            out.setdefault(last_ref, []).append(hyp[op.hyp_idx])
    return out


def newly_introduced_errors(reference: str, base_hyp: str, steered_hyp: str,
                            sets: CandidateUnitSets, *,
                            languages: Mapping[int, str] | None = None) -> dict[str, Any]:
    """Correctness flips between the baseline and steered transcripts.

    Returns counts of newly correct and newly incorrect units, split by whether
    the unit lies inside the candidate, plus the per-unit detail needed to audit
    any single decision.
    """
    base = unit_status(reference, base_hyp)
    steered = unit_status(reference, steered_hyp)

    detail: list[dict[str, Any]] = []
    counts = {
        "n_corrected_inside": 0, "n_corrupted_inside": 0,
        "n_corrected_outside": 0, "n_corrupted_outside": 0,
        "n_unchanged_inside": 0, "n_unchanged_outside": 0,
    }
    for region, unit_ids in (("inside", sets.inside), ("outside", sets.outside)):
        for uid in unit_ids:
            was_ok, base_cat = base.get(uid, (False, "other"))
            now_ok, now_cat = steered.get(uid, (False, "other"))
            if not was_ok and now_ok:
                kind = "corrected"
            elif was_ok and not now_ok:
                kind = "corrupted"
            else:
                kind = "unchanged"
            counts[f"n_{kind}_{region}"] += 1
            detail.append({
                "unit_id": uid, "region": region, "outcome": kind,
                "baseline_correct": bool(was_ok), "steered_correct": bool(now_ok),
                "baseline_category": base_cat, "steered_category": now_cat,
                "language": (languages or {}).get(uid, ""),
                "overlap_ratio": sets.overlap_ratios.get(uid, 0.0),
            })

    base_ins = attribute_insertions(reference, base_hyp)
    steer_ins = attribute_insertions(reference, steered_hyp)
    inside_set = set(sets.inside)

    def _new_insertions(region_inside: bool) -> int:
        total = 0
        for anchor, tokens in steer_ins.items():
            in_region = anchor in inside_set
            if in_region is not region_inside:
                continue
            total += max(0, len(tokens) - len(base_ins.get(anchor, [])))
        return total

    return {
        **counts,
        "n_units_inside": len(sets.inside),
        "n_units_outside": len(sets.outside),
        "n_units_unknown": len(sets.unknown),
        "n_new_insertions_inside": _new_insertions(True),
        "n_new_insertions_outside": _new_insertions(False),
        "transcript_changed": normalize_text(base_hyp) != normalize_text(steered_hyp),
        "unit_detail": detail,
    }


def utility(outcome: Mapping[str, Any], *, eta: float = 1.0, kappa: float = 1.0,
            count_insertions_as_harm: bool = COUNT_INSERTIONS_AS_HARM) -> float:
    """U_k = corrected - eta * local harm - kappa * outside harm.

    Corrections *outside* the candidate are deliberately excluded: the method
    claims to repair what it targets, and crediting incidental outside gains
    would let an indiscriminate intervention score well.
    """
    value = float(outcome["n_corrected_inside"])
    value -= float(eta) * float(outcome["n_corrupted_inside"])
    value -= float(kappa) * float(outcome["n_corrupted_outside"])
    if count_insertions_as_harm:
        value -= float(eta) * float(outcome["n_new_insertions_inside"])
        value -= float(kappa) * float(outcome["n_new_insertions_outside"])
    return float(value)


def candidate_flags(sets: CandidateUnitSets,
                    languages: Mapping[int, str],
                    baseline_correct: Mapping[int, bool],
                    embedded_english: Mapping[int, bool] | None = None,
                    outcome: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """Independent descriptors of one candidate. Several may be true at once."""
    inside = list(sets.inside)
    langs = [languages.get(u, "") for u in inside]
    embedded = embedded_english or {}
    has_embedded_en = any(embedded.get(u, langs[i] == EN) for i, u in enumerate(inside))
    return {
        "covers_embedded_english": bool(has_embedded_en),
        "covers_baseline_error": any(not baseline_correct.get(u, True) for u in inside),
        "covers_baseline_correct_english": any(
            langs[i] == EN and baseline_correct.get(u, False) for i, u in enumerate(inside)),
        "covers_matrix_language": any(lang == ZH for lang in langs),
        "is_localizer_false_positive": not has_embedded_en,
        "covers_no_content_unit": len(inside) == 0,
        "caused_outside_harm": bool((outcome or {}).get("n_corrupted_outside", 0) > 0),
    }


def outcome_row(outcome: Mapping[str, Any], flags: Mapping[str, bool], *,
                eta: float = 1.0, kappa: float = 1.0) -> dict[str, Any]:
    """Flat record for the utility-label table (no nested detail, no categorical)."""
    row = {k: v for k, v in outcome.items() if k != "unit_detail"}
    row.update(flags)
    row["utility"] = utility(outcome, eta=eta, kappa=kappa)
    row["utility_eta05"] = utility(outcome, eta=0.5, kappa=0.5)
    row["utility_eta2"] = utility(outcome, eta=2.0, kappa=2.0)
    return row
