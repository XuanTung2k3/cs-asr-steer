"""LSS unit test: frozen decisions are immutable, self-verifying and auditable."""
from __future__ import annotations

import json

import pytest

from csasr.lss.features_contract import (
    FEATURE_ALLOWLIST,
    assert_inference_safe,
    is_forbidden,
)
from csasr.lss.seeds import REQUIRED_PURPOSES
from csasr.lss.specfreeze import (
    SPEC_FREEZE_SCHEMA,
    SpecFreeze,
    SpecFreezeError,
    assert_matches,
    build,
    canonical_payload,
    compare,
    load,
    seal,
    supersede,
    verify,
)

CFG = {
    "experiment": {"spec_version": "v1", "normalization_version": "v1"},
    "statistics": {"cluster_key": "dialogue_id", "bootstrap_resamples": 10000,
                   "bootstrap_seed": 342, "ci": 0.95},
    "decoding": {"num_beams": 1, "temperature": 0.0},
    "seeds": {p: (7 if p != "jitter" else [1, 2]) for p in REQUIRED_PURPOSES},
    "steering_spec": {"scale": "projection_std", "norm_preserve": True,
                      "decoder_depth_rescale": False},
    "alignment_prereg": {"consensus_estimator_primary": "both_edges"},
}


def _spec(**kwargs):
    return build(CFG, roles={"assignment_hash": "abc"},
                 sites={"decoder": {"tensor": "decoder_post_cross_attn_residual"}},
                 model_assets={"config.json": "sha"},
                 environment={"lock_sha256": "envsha"}, **kwargs)


def test_canonical_payload_blanks_the_hash_field():
    spec = _spec()
    payload = json.loads(canonical_payload(spec))
    assert payload["sha256"] == ""
    assert list(payload) == sorted(payload)


def test_seal_load_verify_round_trip(tmp_path):
    path = tmp_path / "spec_freeze_v1.json"
    spec = _spec()
    seal(spec, path)
    reloaded = load(path)
    assert reloaded.sha256 == spec.sha256
    report = verify(path, expected_sha=spec.sha256)
    assert report["ok"] and report["self_hash_ok"] and report["matches_expected"]


def test_sealing_refuses_to_overwrite(tmp_path):
    path = tmp_path / "spec_freeze_v1.json"
    seal(_spec(), path)
    with pytest.raises(SpecFreezeError, match="already sealed"):
        seal(_spec(), path)


def test_tampering_with_a_nested_value_is_detected(tmp_path):
    path = tmp_path / "spec_freeze_v1.json"
    seal(_spec(), path)
    payload = json.loads(path.read_text())
    payload["outcomes"]["eta_kappa_primary"] = [0.1, 0.1]
    path.write_text(json.dumps(payload))
    report = verify(path)
    assert report["self_hash_ok"] is False and report["ok"] is False


def test_expected_hash_mismatch_fails_verification(tmp_path):
    path = tmp_path / "spec_freeze_v1.json"
    seal(_spec(), path)
    report = verify(path, expected_sha="0" * 64)
    assert report["matches_expected"] is False and report["ok"] is False


def test_schema_version_mismatch_raises(tmp_path):
    path = tmp_path / "spec_freeze_v1.json"
    seal(_spec(), path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = "something_else"
    path.write_text(json.dumps(payload))
    with pytest.raises(SpecFreezeError, match="schema mismatch"):
        load(path)


def test_referenced_artifact_hash_mismatch_fails(tmp_path):
    target = tmp_path / "roles.parquet"
    target.write_bytes(b"original")
    from csasr.lss.artifacts import artifact_ref

    ref = artifact_ref(target)
    path = tmp_path / "spec_freeze_v1.json"
    seal(_spec(referenced=[ref]), path)
    assert verify(path)["referenced_artifacts_ok"] is True

    target.write_bytes(b"rewritten after the gate passed")
    report = verify(path)
    assert report["referenced_artifacts_ok"] is False and report["ok"] is False


def test_supersede_records_what_it_replaced(tmp_path):
    old = tmp_path / "spec_freeze_v1.json"
    seal(_spec(), old)
    new_path = supersede(old, _spec(), reason="decoder site re-decided")
    assert new_path.name == "spec_freeze_v2.json"
    payload = json.loads(new_path.read_text())
    assert payload["supersedes"]["path"] == str(old)
    assert payload["supersedes"]["reason"] == "decoder site re-decided"
    assert payload["spec_version"] == "v2"
    assert old.exists(), "the superseded freeze must remain readable"


def test_assert_matches_catches_config_drift():
    spec = _spec()
    assert_matches(spec, CFG)
    drifted = {**CFG, "statistics": {**CFG["statistics"], "cluster_key": "speaker_id"}}
    with pytest.raises(SpecFreezeError, match="cluster_key"):
        assert_matches(spec, drifted)


def test_freeze_records_the_decisions_the_review_demanded():
    spec = _spec()
    assert spec.payload["schema_version"] == SPEC_FREEZE_SCHEMA
    assert spec.get("eligibility", "eligible_unit") == "embedded_english"
    assert spec.get("sites", "decoder", "tensor") == "decoder_post_cross_attn_residual"
    assert spec.get("steering_scale", "scale") == "projection_std"
    assert "E_pre" in spec.get("steering_scale", "energy_pre")
    assert spec.get("steering_scale", "decoder_depth_rescale") is False
    assert spec.get("outcomes", "deprecates").endswith("outside_region_edits")
    assert spec.get("statistics", "cluster_key") == "dialogue_id"
    assert spec.get("selector_features", "allowlist_version")


def test_feature_guard_rejects_reference_derived_columns():
    import pandas as pd

    good = pd.DataFrame({f.name: [0.0] for f in FEATURE_ALLOWLIST})
    good["candidate_id"] = ["u#0"]
    assert_inference_safe(good, extra_allowed=("candidate_id",))

    leaky = good.copy()
    leaky["consensus_start_disagreement_ms"] = [12.0]
    with pytest.raises(ValueError, match="reference- or alignment-derived"):
        assert_inference_safe(leaky, extra_allowed=("candidate_id",))


def test_feature_guard_does_not_reject_rows_by_split():
    """A guard that refused test rows would block the locked evaluation."""
    import pandas as pd

    frame = pd.DataFrame({f.name: [0.0, 1.0] for f in FEATURE_ALLOWLIST})
    frame["split"] = ["train", "test"]
    assert_inference_safe(frame, extra_allowed=("split",))


def test_removed_plan_v1_feature_is_forbidden_by_pattern():
    assert is_forbidden("consensus_edge_disagreement_ms")
    assert is_forbidden("reference_language")
    assert not is_forbidden("localizer_score_max")


def test_every_allowlisted_feature_declares_its_three_contract_fields():
    for f in FEATURE_ALLOWLIST:
        assert f.source_artifact
        assert isinstance(f.inference_available, bool) and f.inference_available
        assert f.missing_policy in {"nan_plus_flag", "zero", "forbidden"}
        assert f.candidate_mapping


def test_a_rerun_with_unchanged_decisions_reports_no_drift(tmp_path):
    """Measured pilot numbers must not read as a changed decision.

    L0 seals the freeze with the pilot's throughput and the site-check residual
    inside it. Those are wall-clock and floating-point measurements: rerunning
    the identical configuration produces different values, and if `compare`
    counted them the stage could never pass twice -- which is exactly the
    situation after a transient failure.
    """
    import copy

    first = _spec(extra={"pilot": {"projected": {"decode_rate_utt_per_s": 5.819},
                                   "decoder_site_check": {"rel_err": 0.00167}}})
    second = _spec(extra={"pilot": {"projected": {"decode_rate_utt_per_s": 5.744},
                                    "decoder_site_check": {"rel_err": 0.00181}}})
    drift = compare(first, second)
    assert drift["identical"], drift["differing_sections"]

    # a real decision change must still be caught
    moved = copy.deepcopy(CFG)
    moved["steering_spec"] = {**moved["steering_spec"], "norm_preserve": False}
    changed = compare(first, build(
        moved, roles={"assignment_hash": "abc"},
        sites={"decoder": {"tensor": "decoder_post_cross_attn_residual"}},
        model_assets={"config.json": "sha"}, environment={"lock_sha256": "envsha"}))
    assert not changed["identical"]
    assert "steering_scale" in changed["differing_sections"]
