"""POI identification, PIER, error taxonomy and correction/corruption accounting."""
from __future__ import annotations

import pandas as pd

from csasr.evaluation.correction_harm import (
    correction_harm_table,
    outside_region_edits,
    summarize_correction_harm,
)
from csasr.evaluation.pier import (
    LANGUAGE_CONFUSION,
    annotate_unit_status,
    evaluate_pois,
    pier,
    reference_pois,
    unit_status,
)

REF = "我 想 用 python 写 代码"


def test_pois_are_the_embedded_english_units():
    pois = reference_pois("我 想 用 python 写 代码")
    assert [s for _, s in pois] == ["python"]
    assert pois[0][0] == 3          # unit index within the reference


def test_correct_poi_is_marked_correct():
    res = evaluate_pois(REF, "我想用 python 写代码")
    assert len(res) == 1 and res[0].correct and res[0].category == "correct"


def test_han_substitution_is_language_confusion():
    res = evaluate_pois(REF, "我想用拍森写代码")
    assert not res[0].correct
    assert res[0].category in LANGUAGE_CONFUSION


def test_english_substitution_is_same_language():
    res = evaluate_pois(REF, "我想用 pytorch 写代码")
    assert not res[0].correct and res[0].category == "same_language_substitution"


def test_deletion_is_detected():
    res = evaluate_pois(REF, "我想用写代码")
    assert not res[0].correct and res[0].category in ("deletion", "boundary_error")


def test_pier_is_the_error_rate_over_pois_only():
    refs = [REF, REF]
    hyps = ["我想用 python 写代码", "我想用拍森写代码"]
    out = pier(refs, hyps)
    assert out["num_poi"] == 2 and out["num_poi_errors"] == 1
    assert out["pier"] == 0.5
    # matrix-language errors around a correct POI do not move PIER
    out2 = pier([REF], ["我想要 python 写程序"])
    assert out2["pier"] == 0.0


def test_pier_depends_on_the_alignment_surviving():
    """Documented limitation of any alignment-based POI metric.

    When the matrix language is so wrong that the cheapest edit path no longer
    matches the POI to itself, the POI is scored as an error even though the
    English word was produced. E4/E5 therefore also report outside-region edit
    rate, so this failure mode is visible rather than hidden inside PIER.
    """
    out = pier([REF], ["完全不同的句子 python"])
    assert out["pier"] == 1.0


def test_unit_status_covers_every_reference_unit():
    st = unit_status(REF, "我想用 python 写代码")
    assert len(st) == 7          # 我 想 用 python 写 代 码
    assert all(ok for ok, _ in st.values())


def test_annotate_unit_status_fills_the_table():
    units = pd.DataFrame([
        {"utterance_id": "u1", "unit_id": 0, "language_tag": "ZH",
         "baseline_status": "", "baseline_error_type": ""},
        {"utterance_id": "u1", "unit_id": 3, "language_tag": "EN",
         "baseline_status": "", "baseline_error_type": ""},
        {"utterance_id": "u2", "unit_id": 0, "language_tag": "ZH",
         "baseline_status": "", "baseline_error_type": ""},
    ])
    manifest = pd.DataFrame([{"utterance_id": "u1", "transcript_raw": REF},
                             {"utterance_id": "u2", "transcript_raw": REF}])
    preds = pd.DataFrame([{"utterance_id": "u1", "hypothesis_raw": "我想用拍森写代码"}])
    out = annotate_unit_status(units, manifest, preds)
    assert out.loc[0, "baseline_status"] == "correct"
    assert out.loc[1, "baseline_status"] == "incorrect"
    assert out.loc[2, "baseline_status"] == ""     # no prediction -> unknown, excluded


def test_outside_region_edits_ignore_the_target_and_its_neighbours():
    same_outside = outside_region_edits(REF, "我想用拍森写代码", "我想用 python 写代码", 3)
    assert same_outside["outside_changed"] == 0
    changed = outside_region_edits(REF, "我想用拍森写代码", "你想用 python 写程序", 3)
    assert changed["outside_changed"] > 0


def test_correction_and_corruption_summary():
    rows = [
        {"utterance_id": "u1", "speaker_id": "s1", "reference": REF, "poi_index": 3,
         "baseline_hypothesis": "我想用拍森写代码", "steered_hypothesis": "我想用 python 写代码",
         "baseline_correct": False, "baseline_category": "wrong_language_substitution"},
        {"utterance_id": "u2", "speaker_id": "s1", "reference": REF, "poi_index": 3,
         "baseline_hypothesis": "我想用 python 写代码", "steered_hypothesis": "我想用拍森写代码",
         "baseline_correct": True, "baseline_category": ""},
        {"utterance_id": "u3", "speaker_id": "s2", "reference": REF, "poi_index": 3,
         "baseline_hypothesis": "我想用 python 写代码", "steered_hypothesis": "我想用 python 写代码",
         "baseline_correct": True, "baseline_category": ""},
    ]
    table = correction_harm_table(rows)
    s = summarize_correction_harm(table)
    assert s["num_corrected"] == 1 and s["num_corrupted"] == 1
    assert s["correction_rate"] == 1.0
    assert s["corruption_rate"] == 0.5
    assert s["net_corrected_pois"] == 0
