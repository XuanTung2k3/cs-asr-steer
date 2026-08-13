"""LSS unit test: consensus spans, confidence bins and safe interiors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.lss.align.consensus_prod import (
    ConsensusConfig,
    add_safe_interior,
    assign_confidence_bins,
    select_primary_tolerance,
)
from csasr.lss.align.coverage import assert_partition, ignore_mask, label_coverage
from csasr.lss.spans import high_confidence_subset, spans_to_frames

SR = 16000


def _accepted(rows):
    return pd.DataFrame([{
        "utterance_id": u, "unit_id": i, "language": lang,
        "consensus_start_sample": int(s * SR), "consensus_end_sample": int(e * SR),
        "start_disagreement_ms": ds, "end_disagreement_ms": de,
        "sample_rate": SR,
    } for u, i, lang, s, e, ds, de in rows])


def test_confidence_bins_use_the_configured_thresholds():
    """nat5h hard-codes `high` at 100 ms regardless of the criteria."""
    spans = _accepted([("u1", 0, "EN", 0.0, 0.4, 40, 45),
                       ("u1", 1, "EN", 0.5, 0.9, 40, 95),
                       ("u1", 2, "ZH", 1.0, 1.2, 150, 10)])
    binned = assign_confidence_bins(spans, {"high": 50, "medium": 100, "low": 200})
    assert list(binned["confidence_bin"]) == ["high", "medium", "low"]


def test_a_disagreement_beyond_every_bin_is_out_of_range():
    spans = _accepted([("u1", 0, "EN", 0.0, 0.4, 250, 250)])
    binned = assign_confidence_bins(spans, {"high": 50, "medium": 100, "low": 200})
    assert binned.iloc[0]["confidence_bin"] == "out_of_range"


def test_fractional_erosion_keeps_a_short_span_usable():
    """The regression that matters: nat5h's fixed 250 ms erosion with a 200 ms
    minimum interior needs 700 ms spans, and the measured medians are 225 ms
    (EN) and 140 ms (ZH), so it would reject every span in the corpus."""
    spans = _accepted([("u1", 0, "EN", 0.0, 0.37, 10, 10)])     # a 370 ms span
    config = ConsensusConfig(bins={"high": 50}, erosion_ms=None,
                             erosion_fraction=0.25, min_safe_interior_ms=60.0)
    out = add_safe_interior(spans, config)
    assert bool(out.iloc[0]["safe_interior_eroded"]) is True
    assert out.iloc[0]["safe_interior_ms"] > 0
    assert out.iloc[0]["safe_end_sample"] > out.iloc[0]["safe_start_sample"]

    inherited = ConsensusConfig(bins={"high": 50}, erosion_ms=250.0,
                                erosion_fraction=10.0, min_safe_interior_ms=200.0)
    destroyed = add_safe_interior(spans, inherited)
    assert bool(destroyed.iloc[0]["safe_interior_eroded"]) is False


def test_erosion_never_produces_an_empty_interior():
    spans = _accepted([("u1", i, "ZH", i * 0.14, i * 0.14 + 0.14, 5, 5)
                       for i in range(10)])
    config = ConsensusConfig(bins={"high": 50}, erosion_ms=100.0,
                             erosion_fraction=0.25, min_safe_interior_ms=60.0)
    out = add_safe_interior(spans, config)
    assert (out["safe_end_sample"] > out["safe_start_sample"]).all()


def test_mask_padding_widens_the_span_on_both_sides():
    spans = _accepted([("u1", 0, "EN", 1.0, 1.4, 10, 10)])
    config = ConsensusConfig(bins={"high": 50}, union_padding_ms=100.0)
    out = add_safe_interior(spans, config)
    assert out.iloc[0]["mask_start_sample"] == int(0.9 * SR)
    assert out.iloc[0]["mask_end_sample"] == int(1.5 * SR)


def test_mask_start_is_clamped_at_zero():
    spans = _accepted([("u1", 0, "EN", 0.02, 0.4, 10, 10)])
    config = ConsensusConfig(bins={"high": 50}, union_padding_ms=100.0)
    assert add_safe_interior(spans, config).iloc[0]["mask_start_sample"] == 0


def test_frames_round_trip_and_stay_ordered():
    spans = _accepted([("u1", 0, "EN", 0.31, 0.77, 10, 10)])
    config = ConsensusConfig(bins={"high": 50}, union_padding_ms=100.0)
    framed = spans_to_frames(add_safe_interior(spans, config))
    assert framed.iloc[0]["start_frame"] < framed.iloc[0]["end_frame"]
    assert framed.iloc[0]["start_frame"] == int(0.31 / 0.02)


def test_high_confidence_subset_filters_by_bin():
    spans = assign_confidence_bins(
        _accepted([("u1", 0, "EN", 0.0, 0.4, 40, 40),
                   ("u1", 1, "EN", 0.5, 0.9, 150, 150)]),
        {"high": 50, "medium": 100, "low": 200})
    assert len(high_confidence_subset(spans)) == 1


def test_tolerance_is_selected_by_accuracy_not_coverage():
    sweep = pd.DataFrame({"tolerance_ms": [50.0, 100.0, 200.0],
                          "accepted": [10, 60, 300]})
    accuracy = {50.0: {"median_abs_error_ms": 40, "p90_abs_error_ms": 90},
                100.0: {"median_abs_error_ms": 80, "p90_abs_error_ms": 150},
                200.0: {"median_abs_error_ms": 260, "p90_abs_error_ms": 400}}
    chosen = select_primary_tolerance(sweep, accuracy)
    # 200 ms accepts by far the most spans and is still rejected
    assert chosen["selected_tolerance_ms"] == 100.0
    assert chosen["selected_by"].startswith("external accuracy")


def test_no_tolerance_qualifies_when_accuracy_fails_everywhere():
    sweep = pd.DataFrame({"tolerance_ms": [50.0, 100.0]})
    accuracy = {50.0: {"median_abs_error_ms": 300, "p90_abs_error_ms": 500},
                100.0: {"median_abs_error_ms": 400, "p90_abs_error_ms": 600}}
    assert select_primary_tolerance(sweep, accuracy)["selected_tolerance_ms"] is None


def _units(rows):
    return pd.DataFrame([{"utterance_id": u, "unit_id": i, "language": lang,
                          "surface": "x", "unit_position": i}
                         for u, i, lang in rows])


def test_partition_must_cover_every_content_unit():
    units = _units([("u1", 0, "ZH"), ("u1", 1, "EN"), ("u1", 2, "ZH")])
    spans = _accepted([("u1", 1, "EN", 0.5, 0.9, 10, 10)])
    rejected = pd.DataFrame([{"utterance_id": "u1", "reference_unit_index": 0,
                              "reference_language": "ZH"},
                             {"utterance_id": "u1", "reference_unit_index": 2,
                              "reference_language": "ZH"}])
    assert assert_partition(spans, rejected, units)["partition_exact"] is True

    incomplete = rejected.head(1)
    report = assert_partition(spans, incomplete, units)
    assert report["partition_exact"] is False and report["missing"] == 1
    assert report["partition_structurally_valid"] is True
    assert report["accounted_rate"] == pytest.approx(2 / 3)


def test_unaligned_english_is_ignored_not_labelled_negative():
    """Training on it as background teaches the localizer to suppress the hard
    cases and inflates its evaluation on the alignable subset."""
    spans = _accepted([("u1", 1, "EN", 0.2, 0.4, 10, 10)])
    rejected = pd.DataFrame([{"utterance_id": "u1", "reference_unit_index": 3,
                              "reference_language": "EN",
                              "candidate_start_sec": 0.6, "candidate_end_sec": 0.8}])
    mask = ignore_mask(spans, rejected, n_frames=50, utterance_id="u1")
    assert mask[10:20].max() == 1          # the accepted English span is positive
    assert set(np.unique(mask[30:40])) == {2}   # the unalignable one is ignore
    assert mask[45] == 0                   # everything else is a real negative


def test_label_coverage_reports_the_ignore_rate():
    units = _units([("u1", 0, "EN"), ("u1", 1, "EN"), ("u1", 2, "ZH")])
    spans = _accepted([("u1", 0, "EN", 0.0, 0.3, 10, 10)])
    rejected = pd.DataFrame([{"utterance_id": "u1", "reference_unit_index": 1,
                              "reference_language": "EN"}])
    table = label_coverage(spans, rejected, units)
    english = table[table["language"] == "EN"].iloc[0]
    assert english["coverage"] == pytest.approx(0.5)
    assert english["ignore_rate"] == pytest.approx(0.5)
