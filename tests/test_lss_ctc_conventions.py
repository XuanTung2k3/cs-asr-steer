"""LSS unit test: CTC blank-run conventions reassign, report, and select nothing.

Every test here guards a way this diagnostic could produce a believable number
that is wrong: a fitted offset wearing a convention's name, a reassignment that
chains through a shared unit, an edge case silently skipped instead of counted,
held-out roles reaching a development computation, or a convention quietly
winning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csasr.experiments.ctc_convention_diagnostic import (
    convention_tables,
    evaluate_conventions,
    resolve_output,
)
from csasr.lss.align import annotation_pack as pack_mod
from csasr.lss.align import conventions as conv
from csasr.lss.align import target_objects as target_mod

SR = 16000


def _candidate(utt, unit, family, start_s, end_s, language="ZH", *,
               role="D-dev-select", valid=True, failure=""):
    return {
        "schema_version": "nat5h_alignment_v2",
        "utterance_id": utt, "reference_unit_index": unit,
        "reference_language": language, "reference_text": "x",
        "aligner_family": family, "aligner_variant": f"{family}/default",
        "start_sample": int(round(start_s * SR)), "end_sample": int(round(end_s * SR)),
        "start_sec": start_s, "end_sec": end_s, "sample_rate": SR,
        "audio_duration_sec": 30.0, "is_valid": valid, "failure_code": failure,
        "role": role,
    }


def _switch_pair(gap_start=1.0, gap_end=1.4):
    """A ZH run and an EN run bracketing one blank run, hand-checkable."""
    return pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, gap_start, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, gap_end, 2.0, "EN"),
    ])


# --------------------------------------------------------------------------
# the conventions are pure functions of the blank run
# --------------------------------------------------------------------------

@pytest.mark.parametrize("convention,expected_end,expected_start", [
    ("blank_excluded", 1.0, 1.4),
    ("blank_to_preceding", 1.4, 1.4),
    ("blank_to_following", 1.0, 1.0),
    ("blank_midpoint", 1.2, 1.2),
])
def test_convention_boundary_is_the_hand_computed_arithmetic(
        convention, expected_end, expected_start):
    """A 400 ms blank run spanning [1.0, 1.4], worked out by hand.

    This is the test that separates a genuine blank reassignment from an offset
    subtraction: the answers are stated as absolute times derived from the blank
    run's own edges, not as a shift applied to the raw boundary.
    """
    preceding_end, following_start = conv.convention_boundary(convention, 1.0, 1.4)
    assert preceding_end == pytest.approx(expected_end)
    assert following_start == pytest.approx(expected_start)


def test_convention_boundary_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        conv.convention_boundary("blank_to_whoever", 1.0, 1.4)


@pytest.mark.parametrize("convention,expected_end,expected_start", [
    ("blank_excluded", 1.0, 1.4),
    ("blank_to_preceding", 1.4, 1.4),
    ("blank_to_following", 1.0, 1.0),
    ("blank_midpoint", 1.2, 1.2),
])
def test_each_convention_lands_the_hand_computed_boundary_on_the_frame(
        convention, expected_end, expected_start):
    """The same arithmetic, applied end to end over two known language runs."""
    out = conv.apply_convention(_switch_pair(), convention)
    assert float(out.iloc[0]["end_sec"]) == pytest.approx(expected_end)
    assert float(out.iloc[1]["start_sec"]) == pytest.approx(expected_start)
    # The far edges are never touched by any convention.
    assert float(out.iloc[0]["start_sec"]) == pytest.approx(0.0)
    assert float(out.iloc[1]["end_sec"]) == pytest.approx(2.0)


def test_blank_excluded_is_exactly_the_cached_geometry():
    """The reference convention must reproduce the production path bit for bit."""
    frame = _switch_pair()
    out = conv.apply_convention(frame, "blank_excluded")
    for column in ("start_sec", "end_sec", "start_sample", "end_sample"):
        pd.testing.assert_series_equal(out[column], frame[column])
    assert not out["convention_adjusted_start"].any()
    assert not out["convention_adjusted_end"].any()


def test_a_convention_is_a_reassignment_not_a_fitted_offset():
    """Two blanks of different width must move their edges by different amounts.

    A constant offset subtraction fitted to reduce measured error would move
    both boundaries by the same number of milliseconds; a reassignment moves
    each by exactly its own blank run.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),   # 400 ms blank
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 2.1, 3.0, "ZH"),   # 100 ms blank
    ])
    out = conv.apply_convention(frame, "blank_to_preceding")
    moved = (out["end_sec"] - frame["end_sec"]) * 1000.0
    assert moved.tolist() == pytest.approx([400.0, 100.0, 0.0])


def test_reassignment_never_chains_through_a_shared_run():
    """A middle run is the right of one transition and the left of the next.

    Both of its new edges must come from the *original* spans; recomputing the
    second from an already-moved first would compound the shift.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 2.4, 3.0, "ZH"),
    ])
    out = conv.apply_convention(frame, "blank_midpoint")
    middle = out.iloc[1]
    assert float(middle["start_sec"]) == pytest.approx(1.2)   # mid(1.0, 1.4)
    assert float(middle["end_sec"]) == pytest.approx(2.2)     # mid(2.0, 2.4)


def test_conventions_do_not_mutate_the_input_frame():
    """Conventions are a derived view; the raw token spans must survive intact."""
    frame = _switch_pair()
    before = frame.copy(deep=True)
    for convention in conv.CONVENTIONS:
        conv.apply_convention(frame, convention)
    pd.testing.assert_frame_equal(frame, before)


def test_convention_edges_do_not_depend_on_the_other_aligner():
    """No convention may consult Whisper, or it is fitting to another estimator."""
    ctc = _switch_pair()
    with_reference = pd.concat([
        ctc,
        pd.DataFrame([_candidate("u1", 0, conv.REFERENCE_FAMILY, 0.0, 1.35, "ZH"),
                      _candidate("u1", 1, conv.REFERENCE_FAMILY, 1.35, 2.0, "EN")]),
    ], ignore_index=True)
    for convention in conv.CONVENTIONS:
        alone = conv.apply_convention(ctc, convention)
        together = conv.apply_convention(with_reference, convention)
        together = together[
            together["aligner_family"] == conv.CONVENTION_FAMILY].reset_index(drop=True)
        pd.testing.assert_series_equal(alone["start_sec"], together["start_sec"])
        pd.testing.assert_series_equal(alone["end_sec"], together["end_sec"])


def test_only_the_side_a_convention_moves_is_marked_adjusted():
    """An untouched edge must not be flagged, or its sample column gets rewritten."""
    out = conv.apply_convention(_switch_pair(), "blank_to_preceding")
    assert out["convention_adjusted_end"].tolist() == [True, False]
    assert out["convention_adjusted_start"].tolist() == [False, False]
    assert int(out.iloc[1]["start_sample"]) == int(round(1.4 * SR))


# --------------------------------------------------------------------------
# the switch universe is the annotation pack's, not a second one
# --------------------------------------------------------------------------

def test_the_switch_universe_comes_from_the_annotation_pack_helpers():
    """On a clean corpus the two enumerations agree exactly.

    The general relationship is asserted separately, in
    ``test_every_actionable_transition_is_a_pack_adjacent_run_pair``; here the
    fixture has no third language, no missing unit and no degenerate gap, so
    the eligible sets should coincide rather than merely be nested.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 2.4, 3.0, "ZH"),
    ])
    frame["surface"] = "x"
    frame["conversation_id"] = "c"
    frame["speaker"] = "s"
    frame["audio_path"] = "/dev/null"
    universe, _ = pack_mod.switch_universe(frame)
    transitions, _ = conv.switch_transitions(frame)
    actionable = transitions[transitions["actionable"].astype(bool)]
    assert sorted(zip(universe["left_reference_unit_index"],
                      universe["right_reference_unit_index"],
                      universe["switch_direction"])) == \
        sorted(zip(actionable["left_reference_unit_index"],
                   actionable["right_reference_unit_index"],
                   actionable["switch_direction"]))


# --------------------------------------------------------------------------
# every named edge case is classified and counted, never silently skipped
# --------------------------------------------------------------------------

def test_a_blank_spanning_a_missing_unit_is_counted_as_non_contiguous():
    """Reference indices 0 and 2 mean unit 1 is absent from the table.

    Giving that blank to a neighbour would hand it a third unit's audio, so the
    transition is not actionable -- and it is reported, not silently dropped.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
    ])
    transitions, edges = conv.switch_transitions(frame)
    assert transitions.iloc[0]["classification"] == conv.NON_CONTIGUOUS
    assert not bool(transitions.iloc[0]["actionable"])
    assert conv.edge_case_counts(transitions, edges)[conv.NON_CONTIGUOUS] == 1
    out = conv.apply_convention(frame, "blank_to_preceding")
    assert float(out.iloc[0]["end_sec"]) == pytest.approx(1.0)


def test_a_third_language_between_the_runs_is_counted_as_multiple_blank_runs():
    """ZH, OTHER, EN leaves two blank runs plus audio the OTHER unit owns.

    The aggregation drops the OTHER unit, so a ZH->EN boundary does exist; a
    convention still must not hand a neighbour the third unit's frames.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.2, 1.6, "OTHER"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 1.8, 2.0, "EN"),
    ])
    transitions, edges = conv.switch_transitions(frame)
    assert transitions.iloc[0]["classification"] == conv.MULTIPLE_BLANK_RUNS
    assert int(transitions.iloc[0]["intervening_runs"]) == 1
    assert conv.edge_case_counts(transitions, edges)[conv.MULTIPLE_BLANK_RUNS] == 1
    for convention in conv.CONVENTIONS:
        out = conv.apply_convention(frame, convention)
        assert float(out.iloc[0]["end_sec"]) == pytest.approx(1.0)
        assert float(out.iloc[2]["start_sec"]) == pytest.approx(1.8)


def test_zero_length_and_overlapping_blank_runs_are_classified_not_dropped():
    """Touching runs have no blank; overlapping runs have no blank run at all."""
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.0, 2.0, "EN"),    # zero length
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u2", 1, conv.CONVENTION_FAMILY, 0.9, 2.0, "EN"),    # overlap
    ])
    transitions, edges = conv.switch_transitions(frame)
    classified = dict(zip(transitions["utterance_id"], transitions["classification"]))
    assert classified["u1"] == conv.ZERO_LENGTH_BLANK_RUN
    assert classified["u2"] == conv.NO_BLANK_RUN
    counts = conv.edge_case_counts(transitions, edges)
    assert counts[conv.ZERO_LENGTH_BLANK_RUN] == 1
    assert counts[conv.NO_BLANK_RUN] == 1
    assert counts["transitions_total"] == 2
    assert counts["transitions_actionable"] == 0
    # A zero-length blank run is a no-op for every convention, by arithmetic.
    for convention in conv.CONVENTIONS:
        out = conv.apply_convention(frame, convention)
        assert float(out.iloc[0]["end_sec"]) == pytest.approx(1.0)
        assert float(out.iloc[1]["start_sec"]) == pytest.approx(1.0)


def test_utterance_boundary_edges_are_counted_as_edges_not_transitions():
    """The first run's start and the last run's end have no adjacent run.

    No convention can move them, so they are counted rather than mistaken for
    boundaries a convention addressed.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
    ])
    transitions, edges = conv.switch_transitions(frame)
    counts = conv.edge_case_counts(transitions, edges)
    assert counts[conv.UTTERANCE_BOUNDARY_EDGE] == 4      # two utterances
    assert counts["target_language_runs"] == 3
    # Two edges per run; two of them meet at the single transition.
    assert (2 * counts["target_language_runs"]
            - 2 * counts["transitions_total"]) == counts[conv.UTTERANCE_BOUNDARY_EDGE]


def test_same_language_runs_around_a_third_language_are_counted_as_merged():
    """ZH, OTHER, ZH is one target object, so there is no boundary between them."""
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.2, 1.6, "OTHER"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 1.8, 2.0, "ZH"),
    ])
    transitions, edges = conv.switch_transitions(frame)
    counts = conv.edge_case_counts(transitions, edges)
    assert counts["transitions_total"] == 0
    assert counts[conv.MERGED_BY_AGGREGATION] == 1
    assert counts["target_language_runs"] == 1


def test_a_unit_that_cannot_support_an_edge_never_receives_blank():
    """Reassignment must not resurrect a span the aggregation would refuse."""
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN",
                   valid=False, failure="mapping_failed"),
    ])
    transitions, edges = conv.switch_transitions(frame)
    assert transitions.iloc[0]["classification"] == conv.RIGHT_UNUSABLE
    assert conv.edge_case_counts(transitions, edges)[conv.RIGHT_UNUSABLE] == 1
    out = conv.apply_convention(frame, "blank_to_preceding")
    assert float(out.iloc[0]["end_sec"]) == pytest.approx(1.0)


def test_every_named_edge_case_is_reported_even_when_it_never_occurs():
    """An absent key and a zero count are not the same claim."""
    transitions, edges = conv.switch_transitions(_switch_pair())
    counts = conv.edge_case_counts(transitions, edges)
    for name in conv.CLASSIFICATIONS:
        assert name in counts
    assert counts[conv.MULTIPLE_BLANK_RUNS] == 0
    assert counts[conv.SINGLE_BLANK_RUN] == 1


# --------------------------------------------------------------------------
# split discipline
# --------------------------------------------------------------------------

def test_held_out_roles_cannot_reach_a_development_computation():
    """D-dev-confirm and an unlabelled frame are both refusals, not warnings."""
    held_out = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, role="D-dev-confirm"),
    ])
    with pytest.raises(conv.SplitDisciplineError):
        conv.assert_development_only(held_out)
    unlabelled = held_out.drop(columns=["role"])
    with pytest.raises(conv.SplitDisciplineError):
        conv.assert_development_only(unlabelled)


def test_the_evaluator_refuses_a_held_out_row_instead_of_filtering_it():
    """Filtering here would mean the row had already been read.

    The guarantee has to live at the read, so anything downstream must treat a
    held-out row as a defect rather than quietly drop it and carry on.
    """
    frame = pd.concat([
        _paired_fixture(),
        pd.DataFrame([_candidate("u9", 0, conv.CONVENTION_FAMILY, 0.0, 1.0,
                                 role="D-dev-confirm")]),
    ], ignore_index=True)
    with pytest.raises(conv.SplitDisciplineError):
        evaluate_conventions(frame)


def _publish_mixed_role_table(path, cfg):
    """A cached table containing both permitted and held-out roles."""
    from csasr.lss import manifest as manifest_mod

    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH",
                   role="D-construct"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN",
                   role="D-construct"),
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH",
                   role="D-dev-select"),
        _candidate("u3", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH",
                   role="D-dev-confirm"),
        _candidate("u4", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH",
                   role="loc-train"),
    ])
    manifest_mod.publish_frame(path, frame, stage="l1b_valid", cfg=cfg,
                               key_columns=("utterance_id",))
    return frame


def test_held_out_rows_are_never_materialised_by_the_read(tmp_path):
    """The role predicate must be applied by the parquet scan, not afterwards."""
    cfg = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "fake"}}
    path = tmp_path / "candidates_all.parquet"
    published = _publish_mixed_role_table(path, cfg)

    frame, verdict = conv.read_development_candidates(path)
    assert verdict["ok"]
    assert sorted(frame["role"].unique()) == ["D-construct", "D-dev-select"]
    assert verdict["rows_read"] == 3 < len(published)
    assert verdict["roles_requested"] == list(conv.DEVELOPMENT_ROLES)


def test_an_unauthenticated_table_is_refused_before_any_row_is_read(tmp_path):
    """A tampered table must not be read at all, filtered or otherwise."""
    cfg = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "fake"}}
    path = tmp_path / "candidates_all.parquet"
    _publish_mixed_role_table(path, cfg)
    path.write_bytes(path.read_bytes() + b"tamper")

    frame, verdict = conv.read_development_candidates(path)
    assert not verdict["ok"]
    assert verdict["verdict"] == "sha256_mismatch"
    assert not len(frame)


def test_withheld_roles_are_named_from_configuration_not_from_the_table():
    """Naming a held-out role must not require reading one of its rows."""
    development = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, role="D-construct"),
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, role="D-dev-select"),
    ])
    scope = conv.role_scope(development, declared_source_roles=[
        "D-construct", "loc-train", "D-dev-select", "D-dev-confirm"])
    assert scope["roles_withheld"] == ["D-dev-confirm", "loc-train"]
    assert scope["rows_read"] == 2
    assert scope["rows_read_by_role"] == {"D-construct": 1, "D-dev-select": 1}


# --------------------------------------------------------------------------
# what the conventions can and cannot move, and what is measured
# --------------------------------------------------------------------------

def test_within_run_reassignment_cannot_move_a_language_run_edge():
    """The stated prediction: only cross-language blanks move a target boundary.

    A run's outer edges are its first unit's start and its last unit's end, so
    handing blank between two same-language units cannot change them.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "ZH"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 2.5, 3.0, "EN"),
    ])
    edges = {}
    for convention in conv.CONVENTIONS:
        targets, _ = conv.convention_targets(frame, convention)
        zh = targets[targets["language"] == "ZH"].iloc[0]
        edges[convention] = (float(zh["start_sec"]), float(zh["end_sec"]))
    # The ZH run's start is a within-run edge and never moves...
    assert {start for start, _ in edges.values()} == {0.0}
    # ...while its end sits at the switch and moves with the convention.
    assert edges["blank_excluded"][1] == pytest.approx(2.0)
    assert edges["blank_to_preceding"][1] == pytest.approx(2.5)
    assert edges["blank_midpoint"][1] == pytest.approx(2.25)


def test_encoder_frames_are_flagged_stale_rather_than_quietly_rewritten():
    frame = _switch_pair()
    frame["start_encoder_frame"] = [0, 70]
    frame["end_encoder_frame"] = [50, 100]
    out = conv.apply_convention(frame, "blank_to_preceding")
    assert out["end_encoder_frame"].tolist() == [50, 100]
    assert out.loc[0, "encoder_frame_stale"]
    assert bool(out.loc[0, "convention_adjusted_end"])
    # The second-domain edge did move, and the sample domain followed it.
    assert int(out.loc[0, "end_sample"]) == int(round(1.4 * SR))


def test_a_missing_edge_is_neither_flagged_adjusted_nor_cast_to_an_integer():
    """NaN != NaN is True in pandas, which once marked untouched rows adjusted.

    The sample column then round-tripped that NaN through an int64 cast and came
    back as INT64_MIN -- a corrupted timestamp on a row nothing reported.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        # Mirrors the cached table: a finite sample column beside a missing
        # second, on a row the aggregation already refuses.
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 2.0, "ZH",
                   valid=False, failure="mapping_failed"),
        _candidate("u2", 1, conv.CONVENTION_FAMILY, 2.4, 3.0, "EN"),
    ])
    frame.loc[2, "end_sec"] = float("nan")
    out = conv.apply_convention(frame, "blank_to_preceding")
    absent = out["end_sec"].isna()
    assert int(absent.sum()) == 1
    assert not bool(out.loc[absent, "convention_adjusted_end"].any())
    assert (out["end_sample"] >= 0).all()
    assert int(out.loc[absent, "end_sample"].iloc[0]) == int(round(2.0 * SR))


def _paired_fixture():
    """Two utterances both families aligned, with a deliberate disagreement."""
    rows = []
    for utterance, ctc_gap, whisper_edge in (("u1", 1.4, 1.2), ("u2", 1.6, 1.05)):
        rows += [
            _candidate(utterance, 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
            _candidate(utterance, 1, conv.CONVENTION_FAMILY, ctc_gap, 2.0, "EN"),
            _candidate(utterance, 0, conv.REFERENCE_FAMILY, 0.0, whisper_edge, "ZH"),
            _candidate(utterance, 1, conv.REFERENCE_FAMILY, whisper_edge, 2.0, "EN"),
        ]
    return pd.DataFrame(rows)


def test_the_local_pairing_reproduces_the_production_statistic():
    """If this harness measures a different quantity, every number it prints is suspect."""
    targets, _ = conv.convention_targets(_paired_fixture(), "blank_excluded")
    pair = [conv.CONVENTION_FAMILY, conv.REFERENCE_FAMILY]
    production = target_mod.cross_aligner_target_agreement(targets, pair)
    summary = conv.disagreement_summary(targets, pair)
    assert summary["n"] == production["n"]
    assert summary["median_boundary_disagreement_ms"] == \
        production["cross_aligner_boundary_disagreement_ms"]
    assert summary["p90_boundary_disagreement_ms"] == \
        production["cross_aligner_boundary_disagreement_p90_ms"]
    assert summary["within_100ms"] == production["within_100ms"]
    assert summary["abs_en_zh_median_difference_ms"] == \
        pytest.approx(production["en_zh_disagreement_diff_ms"])


def test_per_language_medians_use_the_same_paired_rows_as_the_overall_median():
    """A per-language median over a differently-paired subset would be a fiction."""
    targets, _ = conv.convention_targets(_paired_fixture(), "blank_excluded")
    summary = conv.disagreement_summary(
        targets, [conv.CONVENTION_FAMILY, conv.REFERENCE_FAMILY])
    assert sum(block["n"] for block in summary["by_language"].values()) == summary["n"]


def test_utterance_final_split_partitions_the_same_paired_rows():
    """Position diagnostics must not obtain a more convenient paired set."""
    targets, _ = conv.convention_targets(_paired_fixture(), "blank_excluded")
    pair = [conv.CONVENTION_FAMILY, conv.REFERENCE_FAMILY]
    paired = conv.paired_boundaries(targets, pair)
    summary = conv.disagreement_summary(targets, pair)

    assert paired.groupby("utterance_id")["utterance_final"].sum().eq(1).all()
    split = summary["by_utterance_final"]
    assert split["utterance_final"]["n"] == 2
    assert split["not_utterance_final"]["n"] == 2
    assert sum(block["n"] for block in split.values()) == summary["n"]


def test_both_tolerance_fractions_are_reported_and_ordered():
    targets, _ = conv.convention_targets(_paired_fixture(), "blank_excluded")
    summary = conv.disagreement_summary(
        targets, [conv.CONVENTION_FAMILY, conv.REFERENCE_FAMILY])
    assert 0.0 <= summary["within_100ms"] <= summary["within_200ms"] <= 1.0


def test_the_diagnostic_reports_every_convention_and_selects_none():
    payload = evaluate_conventions(
        _paired_fixture(),
        declared_source_roles=["D-construct", "D-dev-select", "D-dev-confirm"])

    reported = {entry["boundary_convention"]
                for entry in payload["per_convention_own_paired_set"]}
    assert reported == set(conv.CONVENTIONS)
    assert payload["convention_selected"] is None
    # Nothing anywhere in the payload names a winner.
    assert not [key for key in payload
                if key.startswith(("best_", "selected_", "recommended_"))]
    assert payload["role_scope"]["rows_read"] == 8   # two utterances, two families
    assert payload["role_scope"]["roles_withheld"] == ["D-dev-confirm"]
    # blank_excluded moves nothing; the others move the one switch per utterance.
    moved = payload["target_edges_moved_vs_blank_excluded"]
    assert moved["blank_excluded"] == 0
    assert moved["blank_to_preceding"] == 2
    assert payload["family_variant_used"] == f"{conv.CONVENTION_FAMILY}/default"


def test_an_invalid_target_is_not_counted_as_a_moved_edge():
    """NaN != NaN is True, which once reported blank_excluded as moving edges.

    An invalid target's boundary is NaN under every convention, so a plain
    comparison calls it moved -- including under the identity convention, where
    by definition nothing moves.
    """
    frame = pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        # A ZH run whose only unit cannot support an edge: its target boundaries
        # come out NaN under every convention.
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 0.0, "ZH",
                   valid=False, failure="mapping_failed"),
        _candidate("u2", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
    ])
    built = {convention: conv.convention_targets(frame, convention)
             for convention in conv.CONVENTIONS}
    baseline = built["blank_excluded"][0]
    invalid = baseline[baseline["start_sec"].isna() | baseline["end_sec"].isna()]
    assert len(invalid), "fixture must actually produce a missing boundary"

    moved = conv.target_edges_moved(built)
    assert moved["blank_excluded"] == 0
    # The reassignments still move the one switch they can act on.
    assert moved["blank_to_preceding"] == 1


def _two_variant_fixture():
    """The same utterance aligned by two variants of the convention family."""
    rows = []
    for variant, gap in (("existing_ctc/a", 1.4), ("existing_ctc/b", 1.6)):
        for unit, (start, end, language) in enumerate(
                ((0.0, 1.0, "ZH"), (gap, 2.0, "EN"))):
            row = _candidate("u1", unit, conv.CONVENTION_FAMILY, start, end, language)
            row["aligner_variant"] = variant
            rows.append(row)
    rows += [_candidate("u1", 0, conv.REFERENCE_FAMILY, 0.0, 1.2, "ZH"),
             _candidate("u1", 1, conv.REFERENCE_FAMILY, 1.2, 2.0, "EN")]
    return pd.DataFrame(rows)


def test_two_variants_do_not_double_count_moved_edges_or_edge_cases():
    """Metrics use one representative variant, so the counts must use it too.

    Merging on target/utterance/language alone goes many-to-many across
    variants, which can overcount moves or compare one variant with another.
    """
    frame = _two_variant_fixture()
    transitions, edges = conv.switch_transitions(frame)
    assert sorted(transitions["aligner_variant"].unique()) == \
        ["existing_ctc/a", "existing_ctc/b"]

    built = {convention: conv.convention_targets(frame, convention,
                                                 transitions=transitions)
             for convention in conv.CONVENTIONS}
    variant = conv.representative_variant(built["blank_excluded"][0])
    assert variant == "existing_ctc/a"          # the production sort rule

    # One switch in one representative variant: exactly one edge can move.
    moved = conv.target_edges_moved(built)
    assert moved["blank_excluded"] == 0
    assert moved["blank_to_preceding"] == 1
    assert moved["blank_midpoint"] == 2         # both sides of the one switch

    both = conv.edge_case_counts(transitions, edges)
    one = conv.edge_case_counts(transitions, edges, variant=variant)
    assert both["transitions_total"] == 2       # summed over both variants
    assert one["transitions_total"] == 1        # the metric's denominator
    assert one[conv.UTTERANCE_BOUNDARY_EDGE] == 2
    assert both[conv.UTTERANCE_BOUNDARY_EDGE] == 4


def test_the_payload_counts_only_the_variant_the_metrics_use():
    payload = evaluate_conventions(_two_variant_fixture())
    assert payload["family_variant_used"] == "existing_ctc/a"
    assert payload["family_variants_present"] == ["existing_ctc/a", "existing_ctc/b"]
    assert payload["edge_cases"]["transitions_total"] == 1
    assert payload["blank_run_summary"]["all_actionable"]["n"] == 1


# --------------------------------------------------------------------------
# how the inventory relates to the annotation pack's universe
# --------------------------------------------------------------------------

def _pack_ready(frame):
    frame = frame.copy()
    frame["surface"] = "x"
    frame["conversation_id"] = "c"
    frame["speaker"] = "s"
    frame["audio_path"] = "/dev/null"
    return frame


def test_every_actionable_transition_is_a_pack_adjacent_run_pair():
    """A convention may only move a boundary the annotation pack recognises.

    The inventory is wider than the pack's universe -- it also enumerates run
    pairs separated by a third language -- but everything it will *act* on has
    to be one of the pack's directly adjacent EN<->ZH pairs.
    """
    frame = _pack_ready(pd.DataFrame([
        _candidate("u1", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u1", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN"),
        _candidate("u1", 2, conv.CONVENTION_FAMILY, 2.4, 3.0, "ZH"),
        # ZH, OTHER, EN: no direct adjacency, so the pack never sees it.
        _candidate("u2", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH"),
        _candidate("u2", 1, conv.CONVENTION_FAMILY, 1.2, 1.6, "OTHER"),
        _candidate("u2", 2, conv.CONVENTION_FAMILY, 1.8, 2.5, "EN"),
    ]))
    universe, _ = pack_mod.switch_universe(frame)
    transitions, _ = conv.switch_transitions(frame)

    def rows(frame, left, right, direction):
        return sorted(zip(frame[left], frame[right], frame[direction]))

    pack = rows(universe, "left_reference_unit_index",
                "right_reference_unit_index", "switch_direction")
    actionable = transitions[transitions["actionable"].astype(bool)]
    acted = rows(actionable, "left_reference_unit_index",
                 "right_reference_unit_index", "switch_direction")
    assert set(acted) <= set(pack)
    assert actionable["directly_adjacent_runs"].all()

    # And the extra rows are exactly the ones the pack drops, all inert.
    extra = transitions[~transitions["directly_adjacent_runs"].astype(bool)]
    assert set(extra["classification"]) == {conv.MULTIPLE_BLANK_RUNS}
    assert not extra["actionable"].any()


def test_a_partial_run_says_so_instead_of_looking_complete():
    """The mandatory gate runs blank_excluded alone; that must be legible."""
    from csasr.experiments.ctc_convention_diagnostic import resolve_conventions

    assert resolve_conventions(None) == conv.CONVENTIONS
    assert resolve_conventions(["blank_excluded"]) == ("blank_excluded",)
    # The identity convention is the reference every number is stated against,
    # so it is added back rather than left out.
    assert resolve_conventions(["blank_midpoint"]) == ("blank_excluded",
                                                       "blank_midpoint")
    with pytest.raises(SystemExit):
        resolve_conventions(["blank_to_nowhere"])

    payload = evaluate_conventions(_paired_fixture(),
                                   conventions=["blank_excluded"])
    assert payload["conventions_computed"] == ["blank_excluded"]
    assert payload["all_conventions_computed"] is False
    assert payload["convention_selected"] is None
    table = convention_tables(payload)
    assert set(table["boundary_convention"]) == {"blank_excluded"}
    # A one-convention run still reports its own reproduction check.
    assert table["reproduces_production_statistics"].all()


def test_the_output_table_carries_every_required_column():
    payload = evaluate_conventions(_paired_fixture())
    table = convention_tables(payload)
    required = {"boundary_convention", "n_paired",
                "median_boundary_disagreement_ms", "p90_boundary_disagreement_ms",
                "en_median_boundary_disagreement_ms",
                "zh_median_boundary_disagreement_ms", "en_minus_zh_median_ms",
                "within_100ms", "within_200ms",
                "n_utterance_final", "n_not_utterance_final",
                "utterance_final_median_boundary_disagreement_ms",
                "not_utterance_final_median_boundary_disagreement_ms",
                "target_edges_moved_vs_blank_excluded"}
    assert required <= set(table.columns)
    for name in conv.CLASSIFICATIONS:
        assert f"edge_case_{name}" in table.columns
    assert f"edge_case_{conv.UTTERANCE_BOUNDARY_EDGE}" in table.columns
    assert f"edge_case_{conv.MERGED_BY_AGGREGATION}" in table.columns
    assert set(conv.TRANSITION_TOTALS) <= set(table.columns)
    # One row per convention per paired set, and every row states the caveat.
    assert len(table) == 2 * len(conv.CONVENTIONS)
    assert set(table["paired_set"]) == {"own", "common_across_conventions"}
    assert table["reproduces_production_statistics"].all()
    assert table["note"].str.contains("not gold").all()


def test_diagnostic_output_may_not_land_inside_the_artifacts_root(tmp_path):
    root = tmp_path / "artifacts_lss"
    (root / "metrics").mkdir(parents=True)
    with pytest.raises(SystemExit):
        resolve_output(root / "metrics" / "conventions", root)
    outside = resolve_output(tmp_path / "diag", root)
    assert Path(outside).is_absolute()


def _cli_production_root(tmp_path, monkeypatch):
    """A production root holding a real, authenticated, mixed-role table."""
    from csasr.experiments import ctc_convention_diagnostic as cli
    from csasr.lss import manifest as manifest_mod

    production = tmp_path / "production"
    path = production / cli.CANDIDATE_RELPATH
    path.parent.mkdir(parents=True)
    cfg = {"experiment": {"output_root": str(production)},
           "model": {"id": "fake"},
           "roles_to_label": ["D-construct", "loc-train", "D-dev-select",
                              "D-dev-confirm"]}
    frame = pd.concat([
        _paired_fixture(),
        pd.DataFrame([
            _candidate("u9", 0, conv.CONVENTION_FAMILY, 0.0, 1.0, "ZH",
                       role="D-dev-confirm"),
            _candidate("u9", 1, conv.CONVENTION_FAMILY, 1.4, 2.0, "EN",
                       role="D-dev-confirm"),
        ]),
    ], ignore_index=True)
    manifest_mod.publish_frame(path, frame, stage="l1b_valid", cfg=cfg,
                               key_columns=("utterance_id",))
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    return cli, production, path


def test_the_cli_writes_only_tainted_artifacts_outside_the_production_root(
        tmp_path, monkeypatch):
    """Every artifact carries the taint, and none of them lands in production."""
    cli, production, candidates = _cli_production_root(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    before = sorted(p.name for p in (production / "alignments").iterdir())

    assert cli.main(["--diagnostic-output", str(output)]) == 0
    for name in (cli.REPORT_JSON, cli.REPORT_MD, cli.DISAGREEMENT_TABLE,
                 cli.TRANSITION_TABLE):
        manifest = json.loads((output / f"{name}.manifest.json").read_text())
        assert manifest["diagnostic_only"] is True
        assert manifest["taint_reasons"] == [conv.TAINT]
        assert manifest["held_out_gate_generation_consumed"] is False
        assert manifest["convention_selected"] is None
    payload = json.loads((output / cli.REPORT_JSON).read_text())
    assert payload["config_sha256"]
    assert payload["convention_selected"] is None
    assert set(payload["edge_cases"]) >= set(conv.CLASSIFICATIONS)
    # The held-out utterance never reached the computation, and the withheld
    # role was named from configuration rather than from its rows.
    assert payload["role_scope"]["rows_read"] == 8
    assert payload["role_scope"]["roles_withheld"] == ["D-dev-confirm", "loc-train"]
    # The production tree is byte-for-byte what it was.
    assert sorted(p.name for p in (production / "alignments").iterdir()) == before
    assert not (production / "status").exists()
    assert not (production / "freeze").exists()

    with pytest.raises(SystemExit, match="outside the production artifacts root"):
        cli.main(["--diagnostic-output", str(production / "reports" / "conventions")])


def test_the_cli_refuses_unauthenticated_cached_candidates(tmp_path, monkeypatch):
    """Unverified evidence is a refusal, not a silently-diagnosed table."""
    cli, _, candidates = _cli_production_root(tmp_path, monkeypatch)
    candidates.write_bytes(candidates.read_bytes() + b"tamper")
    with pytest.raises(SystemExit, match="not authentic"):
        cli.main(["--diagnostic-output", str(tmp_path / "diagnostic")])
