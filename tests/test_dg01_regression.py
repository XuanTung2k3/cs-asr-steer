"""DG-01 regression validation: legacy Job A/B round-trip + canonical emission.

Covers the DG-01 completion-pass regression contract:
  1-2. Job A and Job B baseline/method comparisons round-trip through the adapter.
  3-4. historical MER / PIER are preserved bit-for-bit (EXACT REPRODUCTION).
  5.   canonical gain sign is baseline - method (positive = improvement).
  6.   method - baseline legacy delta is re-signed to a gain.
  7.   baseline - method legacy gain is computed directly.
  8.   missing legacy provenance stays null (never fabricated).
  9.   blended legacy retention is not decomposed into MC §8 populations.
  10.  legacy transcript outside-edits are not mapped to canonical outside harm.
  11-12. a canonically emitted result_v1 pair validates and round-trips, carrying
         real repository provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csasr.evaluation import legacy_adapter as la
from csasr.evaluation.result_schema import (
    RESULT_SCHEMA_VERSION,
    MethodConfig,
    from_json,
    validate,
)

REPO = Path(__file__).resolve().parents[1]
JOB_A = REPO / "results" / "job_a" / "summary.json"
JOB_B = REPO / "results" / "job_b" / "summary.json"


# --------------------------------------------------------------------------- #
# Phase A — legacy regression                                                  #
# --------------------------------------------------------------------------- #

def _require(path: Path):
    if not path.exists():
        pytest.skip(f"legacy artifact not present: {path}")


def test_job_a_baseline_method_preserves_mer_pier_and_gain_sign():
    _require(JOB_A)
    raw = json.loads(JOB_A.read_text())["systems"]
    recs = {r.system_name: r.to_dict() for r in la.adapt_job_a_summary(JOB_A)}
    base = raw["F0: Baseline"]
    # a representative method system with correctness-transition accounting
    name = "F2: Global α=2"
    assert name in recs
    m = recs[name]["metrics"]
    # 3-4. EXACT REPRODUCTION of historical MER/PIER
    assert m["mer"] == raw[name]["MER"]
    assert m["pier"] == raw[name]["PIER"]
    # 5/7. canonical gain = baseline - method (F2 improves PIER => positive)
    assert m["pier_gain"] == pytest.approx(base["PIER"] - raw[name]["PIER"])
    assert m["pier_gain"] > 0
    # correction/corruption mapped identically from the legacy correction_harm block
    ch = raw[name]["correction_harm"]
    assert m["corrections"] == ch["num_corrected"]
    assert m["corruptions"] == ch["num_corrupted"]
    validate(recs[name])


def test_job_b_baseline_method_roundtrip_and_no_fabrication():
    _require(JOB_B)
    raw = json.loads(JOB_B.read_text())
    base = raw["baseline"]
    table = raw["results_table"]
    recs = {r.system_name: r.to_dict() for r in la.adapt_job_b_summary(JOB_B)}
    # 2. the method systems are actually adapted (not silently dropped)
    name = "T1: Ours-local (greedy)"
    assert name in recs, f"adapter dropped Job-B method systems: {list(recs)}"
    m = recs[name]["metrics"]
    d = recs[name]["descriptive"]
    # 3-4. historical MER/PIER preserved
    assert m["mer"] == table[name]["MER"]
    assert m["pier"] == table[name]["PIER"]
    # 5/7. gain sign against the top-level baseline
    assert m["pier_gain"] == pytest.approx(base["PIER"] - table[name]["PIER"])
    # 8. missing legacy provenance stays null
    prov = recs[name]["provenance"]
    assert prov["git_commit"] is None
    assert prov["config_hash"] is None
    assert prov["production_artifact"] is False
    # 9. blended legacy retention preserved as legacy, canonical retention null
    assert d["legacy_zh_retention_blended"] == table[name]["zh_retention"]
    assert m["retention"] == {"matrix_zh": None, "embedded_en": None, "monolingual": None}
    # 10. transcript outside-edits are not canonical outside harm
    assert d["legacy_outside_edits"] == table[name]["outside_edits"]
    assert m["candidate"]["outside_harm"] is None
    validate(recs[name])


def test_round1_paired_report_resigns_method_minus_baseline_delta():
    # 6. legacy delta_* (method - baseline) becomes a canonical gain (baseline - method)
    record = {"MER": 0.15, "PIER": 0.40, "delta_MER": -0.05, "delta_PIER": -0.02}
    res = la.adapt_round1_paired_report(record, source="unit-test").to_dict()
    assert res["metrics"]["mer_gain"] == pytest.approx(0.05)
    assert res["metrics"]["pier_gain"] == pytest.approx(0.02)
    assert res["provenance"]["production_artifact"] is False


# --------------------------------------------------------------------------- #
# Phase B — canonical emission with real provenance                            #
# --------------------------------------------------------------------------- #

def test_canonical_emission_pair_validates_and_carries_real_provenance():
    from csasr.evaluation import result_emit

    refs = ["我 想 去 school", "hello 世 界"]
    base = ["我 想 去 学 校", "hello 世 界"]
    meth = ["我 想 去 school", "hello 世 界"]
    cfg = {
        "model": {"id": "whisper-large-v3"},
        "data": {"name": "cs-asr", "split": "D-dev-select", "role": "D-dev-select"},
        "experiment": {"seed": 7, "decode_regime": "greedy", "beam": 1,
                       "normalization_version": "v1"},
    }
    baseline, method = result_emit.emit_baseline_and_method(
        refs, base, meth, cfg=cfg, method=MethodConfig(layer=24, direction_type="resid"))
    bd, md = baseline.to_dict(), method.to_dict()

    # 11. validates as result_v1
    validate(bd)
    validate(md)
    assert md["schema_version"] == RESULT_SCHEMA_VERSION
    # correct identity / decode regime / baseline reference / gain sign
    assert md["data_role"] == "D-dev-select"
    assert md["decode_regime"] == "greedy"
    assert md["baseline_ref_run_id"] == bd["run_id"]
    assert md["metrics"]["pier_gain"] >= 0            # method repairs a POI
    # real provenance where available; unavailable stays null (not fabricated)
    prov = md["provenance"]
    assert prov["git_commit"] and len(prov["git_commit"]) == 40
    assert prov["config_hash"] and len(prov["config_hash"]) == 64
    assert prov["direction_artifact_hash"] is None
    assert prov["metrics_schema_version"] == "metrics_v1"
    assert prov["result_schema_version"] == RESULT_SCHEMA_VERSION
    # 12. deterministic round-trip, standard JSON (no NaN/Infinity)
    text = method.to_json()
    assert method.to_json() == text
    assert from_json(text).to_dict() == md
    assert "NaN" not in text and "Infinity" not in text
