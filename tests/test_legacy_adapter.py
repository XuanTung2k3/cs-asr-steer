"""DG-01 legacy adapter: re-signing, identical-metric mapping, no fabrication."""
from __future__ import annotations

import json
from pathlib import Path

from csasr.evaluation import legacy_adapter as la
from csasr.evaluation.result_schema import RESULT_SCHEMA_VERSION, validate

REPO = Path(__file__).resolve().parents[1]


def test_resign_delta_translates_method_minus_baseline():
    # legacy delta_PIER = method - baseline = -0.05 (an improvement)
    assert la.resign_delta(-0.05) == 0.05
    assert la.resign_delta(0.05) == -0.05
    assert la.resign_delta(None) is None
    assert la.resign_delta(float("nan")) is None


def test_adapt_round1_paired_report_resigns_and_marks_legacy():
    record = {"MER": 0.15, "PIER": 0.40, "delta_MER": -0.05, "delta_PIER": -0.02,
              "corrections": 3, "corruptions": 1, "net_corrections": 2,
              "outside_edit_rate": 1.2, "beam": 1, "method_protocol": "greedy"}
    res = la.adapt_round1_paired_report(record, source="unit-test")
    d = res.to_dict()
    assert d["metrics"]["pier_gain"] == 0.02      # -(method-baseline)
    assert d["metrics"]["mer_gain"] == 0.05
    assert d["legacy_source"].endswith("round1")
    assert d["provenance"]["derivation"] == "legacy-derived"
    assert d["provenance"]["production_artifact"] is False
    # canonical outside harm is not fabricated from the legacy edit rate
    assert d["metrics"]["candidate"]["outside_harm"] is None
    assert d["descriptive"]["legacy_outside_edit_rate"] == 1.2


def test_adapt_job_a_summary_roundtrips_committed_artifact():
    path = REPO / "results" / "job_a" / "summary.json"
    if not path.exists():
        import pytest
        pytest.skip("legacy Job-A summary not present")
    results = la.adapt_job_a_summary(path)
    assert results, "expected at least the baseline + systems"
    baseline = json.loads(path.read_text())["systems"]["F0: Baseline"]
    for res in results:
        d = res.to_dict()
        validate(d)                       # emits valid result_v1
        assert d["schema_version"] == RESULT_SCHEMA_VERSION
        assert d["provenance"]["production_artifact"] is False
        assert d["legacy_source"].startswith(str(path))
        # gains are computed against the F0 baseline (baseline - method)
        if d["metrics"]["pier"] is not None and d["system_name"] != "F0: Baseline":
            expected = baseline["PIER"] - d["metrics"]["pier"]
            assert abs(d["metrics"]["pier_gain"] - expected) < 1e-9
        # retention/outside-harm not decomposed from legacy => null, not fabricated
        assert d["metrics"]["retention"]["matrix_zh"] is None
        assert d["metrics"]["candidate"]["outside_harm"] is None
        # deferred gate coverage
        assert d["metrics"]["gate_coverage"]["value"] is None
        # serializes to standard JSON (no NaN token)
        json.loads(res.to_json())


def test_adapt_job_a_maps_correction_harm_identically():
    path = REPO / "results" / "job_a" / "summary.json"
    if not path.exists():
        import pytest
        pytest.skip("legacy Job-A summary not present")
    payload = json.loads(path.read_text())
    results = {r.system_name: r for r in la.adapt_job_a_summary(path)}
    for name, block in payload["systems"].items():
        ch = block.get("correction_harm")
        if not ch:
            continue
        m = results[name].to_dict()["metrics"]
        assert m["corrections"] == ch["num_corrected"]
        assert m["corruptions"] == ch["num_corrupted"]
