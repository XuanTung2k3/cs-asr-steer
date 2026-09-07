"""DG-01 ``result_v1`` schema: round-trip, determinism, deferred gate coverage."""
from __future__ import annotations

import json

import pytest

from csasr.evaluation.result_schema import (
    RESULT_SCHEMA_VERSION,
    CanonicalResult,
    MethodConfig,
    from_json,
    populate_gate_coverage,
    reserved_gate_coverage,
    validate,
)


def _sample() -> CanonicalResult:
    return CanonicalResult(
        run_id="run-001",
        system_name="F5",
        model_id="whisper-x",
        dataset="cs-asr",
        split="D-dev-select",
        data_role="D-dev-select",
        seed=42,
        decode_regime="greedy",
        beam=1,
        method=MethodConfig(layer=24, direction_type="resid", gate_type="factorized"),
        metrics={"mer": 0.18, "pier": 0.42, "mer_gain": 0.01, "pier_gain": 0.02},
    )


def test_schema_version_present_and_validates():
    payload = _sample().to_dict()
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    validate(payload)


def test_json_round_trip_preserves_content():
    original = _sample()
    text = original.to_json()
    back = from_json(text)
    assert back.to_dict() == original.to_dict()


def test_optional_method_fields_default_null():
    r = CanonicalResult(run_id="r", system_name="F0", metrics={"mer": 0.2})
    d = r.to_dict()
    assert d["method"]["layer"] is None
    assert d["method"]["direction_artifact_id"] is None
    assert d["baseline_ref_run_id"] is None


def test_gate_coverage_reserved_null_and_survives_round_trip():
    r = _sample()
    d = r.to_dict()
    gc = d["metrics"]["gate_coverage"]
    assert gc["value"] is None
    assert gc["denominator"] is None
    assert gc["deferred"] is True
    # round-trip keeps it null (not silently converted to 0)
    back = from_json(r.to_json()).to_dict()
    assert back["metrics"]["gate_coverage"]["value"] is None


def test_gate_coverage_not_defaulted_to_zero_by_validate():
    payload = _sample().to_dict()
    payload["metrics"]["gate_coverage"]["value"] = 0.0
    with pytest.raises(ValueError):
        validate(payload)


def test_populate_gate_coverage_is_guarded():
    with pytest.raises(RuntimeError):
        populate_gate_coverage(numerator=1, denominator=2)


def test_reserved_block_shape():
    gc = reserved_gate_coverage()
    for key in ("value", "numerator", "denominator", "denominator_id"):
        assert gc[key] is None


def test_serialization_is_deterministic_and_json_safe():
    r = _sample()
    r.metrics["mer"] = float("nan")   # non-finite must serialize as null
    a = r.to_json()
    b = r.to_json()
    assert a == b                     # sort_keys => deterministic
    parsed = json.loads(a)            # standard JSON, no NaN token
    assert parsed["metrics"]["mer"] is None
