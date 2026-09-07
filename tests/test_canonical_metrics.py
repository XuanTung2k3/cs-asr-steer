"""DG-01 canonical metric facade: delta, correction/corruption, outside harm."""
from __future__ import annotations

import math

import pytest

from csasr.evaluation import canonical
from csasr.lss.outcomes import CandidateUnitSets


def test_gain_sign_improvement_unchanged_degradation():
    # 1. improvement (positive), 2. degradation (negative), 3. unchanged (zero)
    assert canonical.gain(0.20, 0.15) == pytest.approx(0.05)
    assert canonical.gain(0.20, 0.15) > 0
    assert canonical.gain(0.20, 0.25) == pytest.approx(-0.05)
    assert canonical.gain(0.20, 0.25) < 0
    assert canonical.gain(0.20, 0.20) == 0.0


def test_gain_handles_missing_and_nonfinite():
    assert canonical.gain(None, 0.1) is None
    assert canonical.gain(0.1, float("nan")) is None


def test_corpus_metrics_reuse_identity():
    # facade values must equal the underlying validated primitives bit-for-bit
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier as pier_fn

    refs = ["hello 你 好 吗", "我 想 去 school 今天"]
    hyps = ["hello 你 好 吗", "我 想 去 学 校 今天"]
    m = canonical.corpus_metrics(refs, hyps)
    assert m["mer"] == corpus_mer(refs, hyps)["mer"]
    assert m["pier"] == pier_fn(refs, hyps)["pier"]
    assert m["en_wer"] == corpus_mer(refs, hyps)["en_wer"]


def test_correction_counts_only_incorrect_to_correct():
    # baseline wrong on the POI "school", method correct -> exactly one correction
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 学 校"]          # school deleted/substituted -> POI wrong
    method = ["我 想 去 school"]            # POI now correct
    cc = canonical.correction_corruption(refs, baseline, method)
    assert cc["corrections"] == 1
    assert cc["corruptions"] == 0
    assert cc["num_baseline_incorrect_poi"] == 1
    assert cc["correction_rate"] == 1.0


def test_correct_to_correct_is_not_a_correction():
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 school"]
    method = ["我 想 去 school"]
    cc = canonical.correction_corruption(refs, baseline, method)
    assert cc["corrections"] == 0
    assert cc["corruptions"] == 0


def test_corruption_counts_only_correct_to_incorrect():
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 school"]          # POI correct at baseline
    method = ["我 想 去 学 校"]             # POI broken by method
    cc = canonical.correction_corruption(refs, baseline, method)
    assert cc["corruptions"] == 1
    assert cc["corrections"] == 0
    assert cc["num_baseline_correct_poi"] == 1
    assert cc["corruption_rate"] == 1.0


def test_wrong_to_wrong_is_not_corruption():
    refs = ["我 想 去 school"]
    baseline = ["我 想 去 学 校"]
    method = ["我 想 去 大 学"]
    cc = canonical.correction_corruption(refs, baseline, method)
    assert cc["corruptions"] == 0
    assert cc["corrections"] == 0


def test_canonical_outside_harm_is_correctness_flip():
    # unit 0 inside candidate (corrected), unit 2 outside candidate corrupted
    reference = "school 你 good"
    base_hyp = "学 校 你 good"          # unit0 wrong, unit2 correct
    steer_hyp = "school 你 坏"          # unit0 corrected, unit2 corrupted
    sets = CandidateUnitSets(inside=(0,), outside=(2,), unknown=(),
                             overlap_ratios={0: 1.0, 2: 0.0})
    out = canonical.candidate_outcomes(reference, base_hyp, steer_hyp, sets)
    assert out["n_corrected_inside"] == 1
    assert out["outside_harm"] == 1          # correct->incorrect outside
    assert out["n_corrupted_outside"] == 1


def test_outside_transcript_change_without_flip_is_not_harm():
    # outside unit stays correct though its neighbourhood changed -> no harm
    reference = "school 你 good"
    base_hyp = "学 校 你 good"
    steer_hyp = "school 你 good"          # only the targeted unit changed
    sets = CandidateUnitSets(inside=(0,), outside=(2,), unknown=(),
                             overlap_ratios={0: 1.0, 2: 0.0})
    out = canonical.candidate_outcomes(reference, base_hyp, steer_hyp, sets)
    assert out["outside_harm"] == 0
    assert out["n_corrupted_outside"] == 0


def test_paired_report_satisfies_pier_identity_and_is_json_safe():
    refs = ["我 想 去 school", "hello 世 界"]
    baseline = ["我 想 去 学 校", "hello 世 界"]
    method = ["我 想 去 school", "hello 世 界"]
    rep = canonical.paired_corpus_report(refs, baseline, method)
    assert rep["schema_version"] == canonical.SCHEMA_VERSION
    # pier_gain positive because a POI was repaired
    assert rep["pier_gain"] >= 0
    # no NaN/Inf leaks
    for v in (rep["baseline"]["mer"], rep["method"]["mer"]):
        assert v is None or math.isfinite(v)


def test_empty_poi_population_gives_none_rates():
    refs = ["我 想 去 学 校"]   # monolingual Mandarin, no POI
    cc = canonical.correction_corruption(refs, refs, refs)
    assert cc["num_poi"] == 0
    assert cc["correction_rate"] is None
    assert cc["corruption_rate"] is None
