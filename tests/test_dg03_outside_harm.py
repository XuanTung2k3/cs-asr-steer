import json

from csasr.evaluation.dg03_outside_harm import (
    corpus_outside_harm,
    population_unit_sets,
)
from csasr.evaluation.result_schema import from_json, validate


def _fixture_sets():
    reference = "hello中world"
    trusted = [
        {"unit_id": 0, "start_sample": 0, "end_sample": 10, "language": "EN"},
        {"unit_id": 1, "start_sample": 10, "end_sample": 20, "language": "ZH"},
        {"unit_id": 2, "start_sample": 20, "end_sample": 30, "language": "EN"},
    ]
    sets, languages = population_unit_sets(
        reference, trusted, [{"start_sample": 0, "end_sample": 10}])
    return reference, sets, languages


def test_outside_harm_is_correctness_flip_not_transcript_edit():
    reference, sets, languages = _fixture_sets()
    out = corpus_outside_harm(
        [reference], [reference], ["hello错world"], [sets], [languages])
    assert out["outside_harm"] == 1
    assert out["n_corrupted_outside"] == 1


def test_zero_outside_corruption_and_unknown_are_safe():
    reference, sets, languages = _fixture_sets()
    same = corpus_outside_harm(
        [reference], [reference], [reference], [sets], [languages])
    assert same["outside_harm"] == 0
    assert same["n_corrupted_outside"] == 0

    unknown_sets, _ = population_unit_sets(
        reference,
        [{"unit_id": 0, "start_sample": 0, "end_sample": 10,
          "language": "EN"}],
        [{"start_sample": 0, "end_sample": 10}],
    )
    assert 1 in unknown_sets.unknown
    assert 1 not in unknown_sets.outside


def test_result_v1_round_trip_preserves_outside_harm():
    payload = {
        "schema_version": "result_v1",
        "run_id": "dg03/test",
        "system_name": "dg03_test",
        "metrics": {
            "outside_harm": 3,
            "gate_coverage": {"value": None, "numerator": None,
                               "denominator": None, "denominator_id": None,
                               "deferred": True, "note": "test"},
        },
    }
    validate(payload)
    restored = from_json(from_json(json.dumps(payload)).to_json()).to_dict()
    assert restored["metrics"]["outside_harm"] == 3


def test_frozen_l24_selection_inputs_are_unchanged():
    with open("results/dg03/screen/dg03_screen_L24.json") as handle:
        record = json.load(handle)
    assert record["selection_inputs"]["U_C1"] == 33
    assert record["selection_inputs"]["U_C4"] == 10
