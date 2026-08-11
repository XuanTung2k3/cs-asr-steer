"""LSS unit test: harm means a newly introduced error, not merely a change.

These cases are the ones the implementation review named: deletion recovery,
multi-unit candidates, insertions, beneficial outside changes, and false
positive Mandarin candidates.
"""
from __future__ import annotations

import pandas as pd

from csasr.evaluation.correction_harm import outside_region_edits
from csasr.lss.outcomes import (
    attribute_insertions,
    candidate_flags,
    candidate_unit_sets,
    newly_introduced_errors,
    outcome_row,
    utility,
)

SR = 16000


def _spans(entries):
    """entries: (unit_id, start_sec, end_sec)"""
    return pd.DataFrame([{
        "unit_id": uid,
        "consensus_start_sample": int(s * SR),
        "consensus_end_sample": int(e * SR),
    } for uid, s, e in entries])


def test_units_are_assigned_by_acoustic_overlap():
    spans = _spans([(0, 0.0, 0.5), (1, 0.5, 1.0), (2, 1.0, 1.5)])
    sets = candidate_unit_sets(spans, int(0.45 * SR), int(1.05 * SR))
    assert sets.inside == (1,)
    assert sets.outside == (0, 2)


def test_units_without_trustworthy_boundaries_are_unknown_not_outside():
    """A missing alignment must never look like evidence the region was safe."""
    spans = _spans([(0, 0.0, 0.5)])
    sets = candidate_unit_sets(spans, 0, int(0.5 * SR), all_unit_ids=[0, 1, 2])
    assert sets.inside == (0,)
    assert sets.unknown == (1, 2)
    assert sets.outside == ()


def test_deletion_recovery_counts_as_a_correction():
    ref = "我 用 machine 做 实验"
    base = "我 用 做 实验"                 # the English word was deleted
    steered = "我 用 machine 做 实验"
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 1.0),
                    (3, 1.0, 1.2), (4, 1.2, 1.6)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(1.0 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    assert sets.inside == (2,)
    assert out["n_corrected_inside"] == 1
    assert out["n_corrupted_inside"] == 0
    assert utility(out) == 1.0


def test_multi_unit_candidate_can_correct_and_corrupt_at_once():
    """K=1 bounds the number of regions, not the number of units in one."""
    ref = "我 用 machine learning 做"
    base = "我 用 做 learning 做"           # unit 2 wrong, unit 3 right
    steered = "我 用 machine 做 做"          # unit 2 fixed, unit 3 broken
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 0.9),
                    (3, 0.9, 1.4), (4, 1.4, 1.6)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(1.4 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    assert sets.inside == (2, 3)
    assert out["n_corrected_inside"] == 1
    assert out["n_corrupted_inside"] == 1
    assert utility(out) == 0.0


def test_a_beneficial_outside_change_is_not_harm():
    """The existing metric books any outside change as harm; this one does not."""
    # one unit per position: Han is segmented per character
    ref = "我 用 machine 做 好 的 事"          # units 0..6
    base = "我 用 做 做 好 的 情"              # unit 2 and unit 6 both wrong
    steered = "我 用 machine 做 好 的 事"      # both fixed; unit 6 is far outside
    spans = _spans([(i, i * 0.2, (i + 1) * 0.2) for i in range(7)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(0.6 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    assert sets.inside == (2,)
    assert out["n_corrected_inside"] == 1
    assert out["n_corrupted_outside"] == 0
    assert out["n_corrected_outside"] >= 1
    assert utility(out) == 1.0

    legacy = outside_region_edits(ref, base, steered, poi_index=2)
    assert legacy["outside_changed"] > 0        # the old metric calls this harm
    assert out["n_corrupted_outside"] == 0      # the new one does not


def test_outside_corruption_is_charged_to_utility():
    ref = "我 用 machine 做 实验"
    base = "我 用 做 做 实验"
    steered = "我 用 machine 做 做"           # fixed inside, broke unit 4 outside
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 1.0),
                    (3, 1.0, 1.2), (4, 1.2, 1.6)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(1.0 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    assert out["n_corrupted_outside"] == 1
    assert utility(out) == 0.0
    assert utility(out, kappa=2.0) == -1.0


def test_insertions_are_attributed_to_the_preceding_reference_position():
    ref = "我 用 machine"
    hyp = "我 用 the machine"
    attributed = attribute_insertions(ref, hyp)
    assert attributed == {1: ["the"]}
    leading = attribute_insertions(ref, "oh 我 用 machine")
    assert leading == {-1: ["oh"]}


def test_new_insertions_are_counted_but_do_not_silently_become_harm():
    ref = "我 用 machine 做"
    base = "我 用 machine 做"
    steered = "我 用 machine machine 做"
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 1.0), (3, 1.0, 1.2)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(1.0 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    # which side of the boundary a repeated token anchors to depends on the
    # aligner's tie-breaking; what must hold is that it is counted exactly once
    # and that it does not enter the primary utility
    assert out["n_new_insertions_inside"] + out["n_new_insertions_outside"] == 1
    assert out["n_corrupted_inside"] == out["n_corrupted_outside"] == 0
    assert utility(out) == 0.0
    assert utility(out, count_insertions_as_harm=True) == -1.0


def test_false_positive_mandarin_candidate_is_described_independently():
    ref = "我 用 machine 做 实验"
    base = "我 用 machine 做 实验"
    steered = "我 因 machine 做 实验"          # steering broke a Mandarin unit
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 1.0)])
    sets = candidate_unit_sets(spans, 0, int(0.4 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    flags = candidate_flags(
        sets,
        languages={0: "ZH", 1: "ZH", 2: "EN"},
        baseline_correct={0: True, 1: True, 2: True},
        embedded_english={0: False, 1: False, 2: True},
        outcome=out,
    )
    assert flags["is_localizer_false_positive"] is True
    assert flags["covers_matrix_language"] is True
    assert flags["covers_embedded_english"] is False
    assert out["n_corrupted_inside"] == 1
    assert utility(out) == -1.0


def test_flags_are_independent_booleans_not_one_category():
    """A candidate can be an English error *and* cause outside harm."""
    sets = candidate_unit_sets(_spans([(0, 0.0, 0.5), (1, 0.5, 1.0)]),
                               0, int(0.5 * SR))
    flags = candidate_flags(
        sets, languages={0: "EN", 1: "ZH"},
        baseline_correct={0: False, 1: True},
        embedded_english={0: True, 1: False},
        outcome={"n_corrupted_outside": 1},
    )
    assert flags["covers_embedded_english"] is True
    assert flags["covers_baseline_error"] is True
    assert flags["caused_outside_harm"] is True
    assert sum(bool(v) for v in flags.values()) >= 3


def test_outcome_row_is_flat_and_carries_sensitivity_values():
    ref, base, steered = "我 用 machine", "我 用 做", "我 用 machine"
    spans = _spans([(0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 1.0)])
    sets = candidate_unit_sets(spans, int(0.4 * SR), int(1.0 * SR))
    out = newly_introduced_errors(ref, base, steered, sets)
    row = outcome_row(out, candidate_flags(sets, {2: "EN"}, {2: False}))
    assert "unit_detail" not in row
    assert row["utility"] == 1.0
    assert {"utility_eta05", "utility_eta2"} <= set(row)
    assert all(not isinstance(v, (list, dict)) for v in row.values())
