"""LSS unit test: absolute boundary error from constructed ground truth.

This is the evidence automatic Gate A blocks without, so these tests are about
one question: does a *known* boundary come back as a measured error, and does an
aligner that is right, wrong, or silent get described as exactly that?

The aligners themselves are not run here. `boundary_predictions` and
`score_rendered_set` take a candidate table, so a synthetic-exact case can be
constructed on the CPU: predictions placed at a known offset from a known truth
must produce that offset and no other number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.lss.align import synthetic as syn

SR = 16000


def _rendered(n=3, *, gap_sec=0.0, zh_end=2.0):
    """`n` rendered splices whose seam is at `zh_end`."""
    rows = []
    for i in range(n):
        rows.append({
            "pair_id": f"gate_syn_{i:04d}",
            "audio_path": f"/tmp/gate_syn_{i:04d}.wav",
            "zh_text": "我们 用",
            "en_text": "machine learning",
            "duration_sec": zh_end + gap_sec + 1.5,
            "purpose": "gate",
            "zh_end_sec": zh_end,
            "en_start_sec": zh_end + gap_sec,
            "true_boundary_sec": zh_end + gap_sec / 2.0,
        })
    return pd.DataFrame(rows)


def _candidate(pair_id, unit, language, start_s, end_s, *,
               family="existing_ctc", valid=True):
    return {
        "schema_version": "nat5h_alignment_v2",
        "utterance_id": pair_id,
        "reference_unit_index": unit,
        "reference_language": language,
        "reference_text": "x",
        "aligner_family": family,
        "aligner_variant": f"{family}/default",
        "start_sec": start_s, "end_sec": end_s,
        "start_sample": int(start_s * SR), "end_sample": int(end_s * SR),
        "is_valid": valid, "failure_code": "" if valid else "invalid_span",
    }


def _predictions(rendered, *, offset_sec=0.0, family="existing_ctc"):
    """Two ZH units then two EN units per item, the seam off by `offset_sec`."""
    rows = []
    for _, item in rendered.iterrows():
        zh_end = float(item["zh_end_sec"]) + offset_sec
        en_start = float(item["en_start_sec"]) + offset_sec
        pid = item["pair_id"]
        rows += [
            _candidate(pid, 0, "ZH", 0.2, 1.0, family=family),
            _candidate(pid, 1, "ZH", 1.0, zh_end, family=family),
            _candidate(pid, 2, "EN", en_start, en_start + 0.6, family=family),
            _candidate(pid, 3, "EN", en_start + 0.6, en_start + 1.2, family=family),
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the seam is read off the units that touch it
# ---------------------------------------------------------------------------
def test_a_perfect_aligner_measures_zero_error():
    rendered = _rendered()
    scores, per_item = syn.score_rendered_set(
        rendered, _predictions(rendered, offset_sec=0.0), purpose="gate")
    assert per_item["scorable"].all()
    for _, row in scores.iterrows():
        assert row["median_abs_error_ms"] == pytest.approx(0.0, abs=1e-6)
        assert row["median_signed_error_ms"] == pytest.approx(0.0, abs=1e-6)
        assert row["within_100ms"] == pytest.approx(1.0)


def test_a_known_offset_comes_back_as_that_offset_signed():
    """The whole point: a constructed boundary yields absolute error.

    An aligner placing every boundary 150 ms early must be reported as -150 ms,
    not as 150, not as a disagreement, and not as within-100ms.
    """
    rendered = _rendered(n=5)
    scores, _ = syn.score_rendered_set(
        rendered, _predictions(rendered, offset_sec=-0.150), purpose="gate")
    canonical = scores[scores["convention"] == syn.CANONICAL_CONVENTION]
    assert len(canonical) == 3                        # start, end, combined
    for _, row in canonical.iterrows():
        assert row["median_signed_error_ms"] == pytest.approx(-150.0, abs=1e-3)
        assert row["median_abs_error_ms"] == pytest.approx(150.0, abs=1e-3)
        assert row["within_100ms"] == pytest.approx(0.0)


def test_the_seam_uses_the_last_zh_unit_and_the_first_en_unit():
    """Not the first ZH unit, not the last EN unit -- the units that touch it."""
    rendered = _rendered(n=1)
    per_item = syn.boundary_predictions(rendered, _predictions(rendered))
    row = per_item.iloc[0]
    assert row["last_zh_unit_index"] == 1
    assert row["first_en_unit_index"] == 2
    assert row["predicted_end_sec"] == pytest.approx(2.0)
    assert row["predicted_start_sec"] == pytest.approx(2.0)


def test_with_a_gap_each_edge_is_scored_against_the_instant_it_predicts():
    """A gap makes the two edges predict different instants.

    Scoring both against the midpoint would charge a perfect aligner half the
    gap as error in opposite directions.
    """
    rendered = _rendered(n=4, gap_sec=0.200)
    scores, _ = syn.score_rendered_set(
        rendered, _predictions(rendered, offset_sec=0.0), purpose="gate")
    canonical = scores[scores["convention"] == syn.CANONICAL_CONVENTION]
    for _, row in canonical.iterrows():
        assert row["median_abs_error_ms"] == pytest.approx(0.0, abs=1e-6), row["edge"]
    # while the legacy midpoint estimate is exactly right at the midpoint
    per_item = syn.boundary_predictions(rendered, _predictions(rendered))
    assert per_item["predicted_midpoint_sec"].iloc[0] == pytest.approx(
        rendered["true_boundary_sec"].iloc[0])


def test_combined_is_the_worse_edge_of_each_item():
    """A mask is wrong at whichever end is wrong, so combined takes the worse."""
    rendered = _rendered(n=1)
    rows = [
        _candidate("gate_syn_0000", 0, "ZH", 0.2, 2.010),     # end +10 ms
        _candidate("gate_syn_0000", 1, "EN", 1.700, 2.4),     # start -300 ms
    ]
    scores, _ = syn.score_rendered_set(rendered, pd.DataFrame(rows), purpose="gate")
    combined = scores[(scores["convention"] == syn.CANONICAL_CONVENTION)
                      & (scores["edge"] == "combined")].iloc[0]
    assert combined["median_abs_error_ms"] == pytest.approx(300.0, abs=1e-3)
    assert combined["median_signed_error_ms"] == pytest.approx(-300.0, abs=1e-3)


# ---------------------------------------------------------------------------
# what the gate is allowed to read
# ---------------------------------------------------------------------------
def test_the_legacy_convention_is_measured_but_kept_out_of_the_score_table():
    """Gate A takes the worst row in the table, so a convention the pipeline
    does not consume must not be in it -- E1 measured the midpoint convention
    ~490 ms out, which would fail the bias criterion for an unrelated reason.
    The comparison still has to exist, on the same items."""
    rendered = _rendered(n=3, gap_sec=0.400)
    candidates = _predictions(rendered, offset_sec=0.0)
    scores, per_item = syn.score_rendered_set(rendered, candidates, purpose="gate")

    assert set(scores["convention"]) == {syn.CANONICAL_CONVENTION}
    comparison = syn.convention_comparison(per_item)
    assert syn.LEGACY_CONVENTION in set(comparison["convention"])
    assert syn.CANONICAL_CONVENTION in set(comparison["convention"])
    # same items on both sides of the comparison, or it compares nothing
    assert comparison["n"].nunique() == 1


def test_the_score_table_declares_the_schema_gate_a_requires():
    from csasr.lss.align.autoevidence import (
        SYNTHETIC_REQUIRED_COLUMNS,
        SYNTHETIC_SCORES_SCHEMA,
        synthetic_calibration_status,
    )

    rendered = _rendered(n=1)
    scores, _ = syn.score_rendered_set(
        rendered, _predictions(rendered), purpose="gate")
    assert set(SYNTHETIC_REQUIRED_COLUMNS) <= set(scores.columns)
    assert set(scores["schema_version"]) == {SYNTHETIC_SCORES_SCHEMA}
    assert SYNTHETIC_SCORES_SCHEMA == syn.SCORES_SCHEMA_VERSION
    # and the reader accepts what the writer produces
    assert callable(synthetic_calibration_status)


def test_gate_a_reads_a_written_score_table_end_to_end(tmp_path):
    """Writer and reader agreeing is the only thing that makes this evidence."""
    from csasr.lss.align import autoevidence

    rendered = _rendered(n=40)
    scores, _ = syn.score_rendered_set(
        rendered, _predictions(rendered, offset_sec=-0.020), purpose="gate")
    path = tmp_path / autoevidence.SYNTHETIC_SCORES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path, index=False)

    status = autoevidence.synthetic_calibration_status(
        tmp_path, min_boundaries=40, require_authentication=False)
    assert status["available"] is True, status.get("detail")
    assert status["num_boundaries"] == 40
    absolute = status["absolute_boundary_error"]
    assert absolute["median_abs_error_ms"] == pytest.approx(20.0, abs=1e-3)
    assert absolute["max_abs_bias_ms"] == pytest.approx(20.0, abs=1e-3)
    assert absolute["within_100ms"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# silence is not accuracy
# ---------------------------------------------------------------------------
def test_a_family_that_predicted_nothing_is_unscorable_not_accurate():
    """The trap: an aligner producing no valid unit on one side of the seam has
    made no prediction. Scoring it as a large error reads as an accuracy
    failure; dropping it silently reads as a perfect score on fewer items."""
    rendered = _rendered(n=2)
    rows = []
    for _, item in rendered.iterrows():
        # ZH only: nothing predicts the start of English
        rows.append(_candidate(item["pair_id"], 0, "ZH", 0.2, 2.0))
    per_item = syn.boundary_predictions(rendered, pd.DataFrame(rows))
    assert not per_item["scorable"].any()
    assert set(per_item["reason"]) == {"no_valid_en_unit"}

    summary = syn.unscorable_summary(per_item)
    assert summary["existing_ctc"]["scored"] == 0
    assert summary["existing_ctc"]["unscorable"] == 2
    assert summary["existing_ctc"]["reasons"] == {"no_valid_en_unit": 2}

    scores, _ = syn.score_rendered_set(rendered, pd.DataFrame(rows), purpose="gate")
    assert not len(scores), "an unscorable family must produce no score row"


def test_invalid_spans_do_not_count_as_predictions():
    rendered = _rendered(n=1)
    rows = [
        _candidate("gate_syn_0000", 0, "ZH", 0.2, 2.0, valid=False),
        _candidate("gate_syn_0000", 1, "EN", 2.0, 2.6),
    ]
    per_item = syn.boundary_predictions(rendered, pd.DataFrame(rows))
    assert bool(per_item.iloc[0]["scorable"]) is False
    assert per_item.iloc[0]["reason"] == "no_valid_zh_unit"


def test_each_family_is_scored_separately():
    rendered = _rendered(n=3)
    candidates = pd.concat([
        _predictions(rendered, offset_sec=0.0, family="existing_ctc"),
        _predictions(rendered, offset_sec=-0.400, family="whisper_dtw"),
    ], ignore_index=True)
    scores, _ = syn.score_rendered_set(rendered, candidates, purpose="gate")
    ctc = scores[scores["family"] == "existing_ctc"]
    dtw = scores[scores["family"] == "whisper_dtw"]
    assert len(ctc) == 3 and len(dtw) == 3
    assert ctc["median_abs_error_ms"].max() == pytest.approx(0.0, abs=1e-6)
    assert dtw["median_abs_error_ms"].min() == pytest.approx(400.0, abs=1e-3)


def test_empty_inputs_produce_no_scores_rather_than_an_exception():
    assert not len(syn.boundary_predictions(pd.DataFrame(), pd.DataFrame()))
    scores, per_item = syn.score_rendered_set(_rendered(), pd.DataFrame())
    assert not len(scores) and not len(per_item)
    assert syn.unscorable_summary(pd.DataFrame()) == {}
    assert not len(syn.convention_comparison(pd.DataFrame()))


# ---------------------------------------------------------------------------
# the manifest the aligners consume
# ---------------------------------------------------------------------------
def test_the_aligner_manifest_carries_the_concatenated_transcript():
    rendered = _rendered(n=2)
    manifest = syn.aligner_manifest(rendered)
    assert list(manifest["utterance_id"]) == list(rendered["pair_id"])
    assert manifest["transcript_raw"].iloc[0] == "我们 用 machine learning"
    # a synthetic row must be distinguishable from a corpus row
    assert set(manifest["conversation_id"]) == {"synthetic"}
    assert manifest["split"].iloc[0] == "synthetic_gate"


def test_the_concatenated_transcript_segments_into_zh_then_en_units():
    """The seam is only a unit edge if no unit straddles it."""
    from csasr.nat5h.units import build_reference_units

    manifest = syn.aligner_manifest(_rendered(n=1))
    units = build_reference_units("gate_syn_0000",
                                  manifest["transcript_raw"].iloc[0])
    content = [u for u in units if u.language in {"EN", "ZH"}]
    languages = [u.language for u in content]
    assert set(languages) == {"ZH", "EN"}
    # every ZH unit precedes every EN unit
    assert languages == sorted(languages, key=lambda l: 0 if l == "ZH" else 1)
