from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from csasr.data.alignment_checks import synthetic_harness_fingerprint
from csasr.evaluation.correction_harm import summarize_correction_harm
from csasr.experiments.e1_alignment import (
    human_audit_gate_state,
    independent_audit_gate_state,
)
from csasr.experiments.e4_oracle import (
    correction_ratio_gate_pass,
    degradation_within_limit,
    fraction_delta,
    percentage_point_label,
    phase_metric_limits,
    randloc_candidate_starts,
    relocated_spans,
)
from csasr.experiments.e5_boundaries import retained_gain_analysis
from csasr.utils.status import STAGES, read_status, require_passed, write_status


def test_synthetic_harness_hash_changes_with_source_audio_content():
    pair = {
        "pair_id": "syn_0000",
        "zh_utterance_id": "zh",
        "en_utterance_id": "en",
        "zh_audio_path": "/x/zh.wav",
        "en_audio_path": "/x/en.wav",
        "zh_audio_content_sha256": "a",
        "en_audio_content_sha256": "b",
        "zh_text": "你好",
        "en_text": "hello",
        "gap_sec": 0.0,
        "sample_rate": 16000,
    }
    h1 = synthetic_harness_fingerprint([pair], {"trim": True})["sha256"]
    pair2 = dict(pair, zh_audio_content_sha256="changed")
    h2 = synthetic_harness_fingerprint([pair2], {"trim": True})["sha256"]
    assert h1 != h2


def test_human_audit_is_mandatory_and_diagnostic_override_never_passes():
    missing = human_audit_gate_state(
        None, required_items=100, min_usable_fraction=0.90,
        allow_unreviewed_alignment=False)
    assert not missing["complete"] and not missing["passes"]

    diagnostic = human_audit_gate_state(
        None, required_items=100, min_usable_fraction=0.90,
        allow_unreviewed_alignment=True)
    assert diagnostic["diagnostic_override_prohibits_pass"]
    assert not diagnostic["passes"]

    reviewed = human_audit_gate_state(
        {"num_verdicts": 100, "accepted_fraction": 0.91},
        required_items=100, min_usable_fraction=0.90)
    assert reviewed["complete"] and reviewed["passes"]


def test_independent_audit_gate_blocks_unavailable_or_too_small_samples():
    unavailable = independent_audit_gate_state(
        {"status": "unavailable"}, required_items=200,
        min_within_100ms=0.90, max_median_abs_ms=100)
    assert not unavailable["passes"]

    too_small = independent_audit_gate_state(
        {"status": "ok", "num_compared": 199, "pct_within_100ms": 1.0,
         "median_abs_diff_ms": 10},
        required_items=200, min_within_100ms=0.90, max_median_abs_ms=100)
    assert not too_small["passes"]

    ok = independent_audit_gate_state(
        {"status": "ok", "num_compared": 200, "pct_within_100ms": 0.91,
         "median_abs_diff_ms": 80},
        required_items=200, min_within_100ms=0.90, max_median_abs_ms=100)
    assert ok["passes"]


def test_phase_specific_status_names_and_smoke_rejection(tmp_path):
    assert "e2_pilot" in STAGES and "e4_confirm" in STAGES
    write_status(tmp_path, "e1", "passed", complete=True,
                 provenance={"production_artifact": False})
    with pytest.raises(RuntimeError, match="smoke-artifact"):
        require_passed(tmp_path, ["e1"])
    assert read_status(tmp_path, "e1")["status"] == "passed"


def test_metric_fraction_thresholds_and_percentage_point_label():
    delta = fraction_delta(0.2322, 0.2342)
    assert delta == pytest.approx(0.002)
    assert degradation_within_limit(0.2322, 0.2342, 0.002)
    assert not degradation_within_limit(0.2322, 0.4322, 0.002)
    assert percentage_point_label(delta) == "+0.20 percentage points"
    limits = phase_metric_limits({
        "pilot_max_mer_degradation_abs": 0.005,
        "confirm_max_mer_degradation_abs": 0.002,
        "pilot_max_zh_cer_degradation_abs": 0.005,
        "confirm_max_zh_cer_degradation_abs": 0.002,
        "max_outside_region_edit_rate": 0.05,
    }, "confirm")
    assert limits["max_mer_degradation_abs"] == 0.002


def test_zero_corruption_ratio_is_not_infinite_but_gate_uses_counts():
    table = pd.DataFrame([
        {"baseline_correct": False, "corrected": True,
         "corrupted": False, "outside_edit_rate": 0.0, "any_change": True},
        {"baseline_correct": True, "corrected": False,
         "corrupted": False, "outside_edit_rate": 0.0, "any_change": False},
    ])
    summary = summarize_correction_harm(table)
    assert math.isnan(summary["correction_to_corruption_ratio"])
    assert summary["zero_corruption"]
    assert correction_ratio_gate_pass(summary, 2.0)


def test_randloc_candidates_exclude_shoulders_and_prefer_zh_speech():
    speech = np.ones(80, dtype=bool)
    preferred = np.zeros(80, dtype=bool)
    preferred[40:55] = True
    candidates, preferred_candidates = randloc_candidate_starts(
        80, width=10, excluded_start=15, excluded_end=35,
        speech_mask=speech, preferred_mask=preferred)
    assert all(c + 10 <= 15 or c >= 35 for c in candidates)
    assert preferred_candidates
    assert all(c >= 40 and c + 10 <= 55 for c in preferred_candidates)


def test_relocated_spans_preserve_width_and_avoid_shoulders():
    bundle = SimpleNamespace(valid_frames=lambda d: 100)
    targets = pd.DataFrame([{
        "utterance_id": "u1",
        "start_frame": 40,
        "end_frame": 50,
        "duration_sec": 2.0,
        "zh_spans": [(0, 20), (70, 90)],
        "audio_path": "",
    }])
    spans, unrelocated = relocated_spans(bundle, targets, seed=7, shoulder_frames=5)
    assert unrelocated == 0
    s, e = spans["u1"]
    assert e - s == 10
    assert e <= 35 or s >= 55


def test_e5_refuses_retained_gain_when_exact_denominator_is_not_positive():
    rows = pd.DataFrame([
        {"mask": "EXACT_TAPER", "pier_baseline": 0.50, "pier": 0.50},
        {"mask": "JITTER_100", "pier_baseline": 0.50, "pier": 0.40},
    ])
    out, summary = retained_gain_analysis(
        rows, "EXACT_TAPER", min_exact_gain=0.005,
        min_retained_gain=0.70, bootstrap_resamples=100, seed=1)
    assert not summary["exact_gain_ok"]
    assert out["retained_gain"].isna().all()
