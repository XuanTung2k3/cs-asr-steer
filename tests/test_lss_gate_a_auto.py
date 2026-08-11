"""LSS unit tests: Gate A concludes automatically, and cannot conclude falsely.

The policy these tests pin down:

* the primary path requires no human annotation and no human-verdict file is
  ever opened in automatic mode;
* preparing evidence is a complete, successful piece of work -- exit 0 -- and
  decides nothing;
* natural-speech cross-aligner disagreement is never presented as boundary
  error, and absolute error is only claimed where a boundary is actually known;
* every way Gate A could pass without the evidence to support it is closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.lss.align import autoevidence
from csasr.lss.gates import check, classify, evaluate, exit_code, reported
from csasr.lss.prereq import (
    inherited_taint,
    require_prerequisites,
    PrerequisiteError,
    taint_of,
)
from csasr.utils.status import ALLOWED, read_status, write_status

SR = 16000


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _candidates(families=("existing_ctc", "whisper_dtw"), *, n=40,
                invalid_family=None, offset_ms=0.0, nonmonotonic_family=None):
    """A candidate table with two families placing the same units."""
    rows = []
    for family in families:
        for utt in range(n // 4):
            for unit in range(4):
                start = 1.0 * unit + (offset_ms / 1000.0 if family == "whisper_dtw" else 0.0)
                end = start + 0.4
                if family == nonmonotonic_family and unit == 2:
                    start, end = 0.0, 0.1          # jumps backwards
                rows.append({
                    "utterance_id": f"u{utt}",
                    "reference_unit_index": unit,
                    "reference_language": "EN" if unit % 2 else "ZH",
                    "aligner_family": family,
                    "aligner_variant": "default",
                    "start_sample": int(start * SR),
                    "end_sample": int(end * SR),
                    "is_valid": family != invalid_family,
                    "failure_code": "" if family != invalid_family else "no_alignment",
                    "reference_text": "hello" if unit % 2 else "你好",
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4 + 7: a missing or non-independent second aligner cannot pass Gate A
# --------------------------------------------------------------------------
def test_a_single_valid_aligner_cannot_satisfy_the_two_aligner_requirement():
    """The observed smoke state: existing_ctc valid, whisper_dtw failed."""
    candidates = _candidates(invalid_family="whisper_dtw")
    validity = autoevidence.family_validity(candidates)
    independence = autoevidence.independent_valid_families(validity)
    assert independence["n_independent_valid_families"] == 1
    assert independence["qualifying_families"] == ["existing_ctc"]
    assert "whisper_dtw" in independence["rejected_families"]


def test_two_variants_of_one_estimator_are_not_two_aligners():
    """Independence is by estimator class: a second `pred_start` offset of the
    same DTW is the same evidence, not corroboration."""
    a = _candidates(families=("whisper_dtw",))
    b = a.copy()
    b["aligner_variant"] = "pred_start_0"
    b["start_sample"] = b["start_sample"] + int(0.02 * SR)
    b["end_sample"] = b["end_sample"] + int(0.02 * SR)
    validity = autoevidence.family_validity(pd.concat([a, b], ignore_index=True))
    independence = autoevidence.independent_valid_families(validity)
    assert independence["n_independent_valid_families"] == 1
    assert independence["qualifying_families"] == ["whisper_dtw"]


def test_a_family_over_the_invalid_rate_limit_does_not_qualify():
    """The 0.95 coverage floor alone would admit 5% invalid spans against the
    proposal's 1% limit; both floors are required, independently."""
    candidates = _candidates(families=("existing_ctc", "whisper_dtw"), n=400)
    # make exactly 4% of one family's rows invalid: above 0.01, below 0.05
    rows = candidates.index[candidates["aligner_family"] == "whisper_dtw"][:8]
    candidates.loc[rows, "is_valid"] = False
    validity = autoevidence.family_validity(candidates)
    dtw = validity.set_index("aligner_family").loc["whisper_dtw"]
    assert 0.01 < dtw["invalid_rate"] <= 0.05        # would pass a 0.95 floor
    independence = autoevidence.independent_valid_families(validity)
    assert "whisper_dtw" not in independence["qualifying_families"]
    assert any("invalid_rate" in why
               for why in independence["rejection_reasons"]["whisper_dtw"])
    assert independence["n_independent_valid_families"] == 1


def test_coverage_denominator_is_the_expected_universe_not_the_candidate_rows():
    """Candidate-relative coverage reads 1.0 even when half the intended units
    were never aligned -- exactly the failure the threshold exists to catch."""
    candidates = _candidates(n=20)                    # 5 utterances x 4 units
    expected = pd.DataFrame([
        {"utterance_id": f"u{u}", "reference_unit_index": i}
        for u in range(10) for i in range(4)          # 10 utterances were intended
    ])
    naive = autoevidence.unit_coverage(candidates)
    assert naive["coverage"] == pytest.approx(1.0)
    assert naive["denominator"] == "candidate_rows_only"

    honest = autoevidence.unit_coverage(candidates, expected)
    assert honest["denominator"] == "expected_universe"
    assert honest["coverage"] == pytest.approx(0.5)
    assert honest["missing_units"] == 20


def test_missing_second_aligner_blocks_the_gate_rather_than_failing_it():
    """A second opinion that does not exist is missing infrastructure, not a
    broken stage and not a measurement."""
    candidates = _candidates(invalid_family="whisper_dtw")
    validity = autoevidence.family_validity(candidates)
    independence = autoevidence.independent_valid_families(validity)
    assert independence["n_independent_valid_families"] < 2
    payload, status = evaluate(
        [reported("independent_valid_aligner_families",
                  independence["n_independent_valid_families"])],
        name="gate_a",
        blocked_reasons=[autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS])
    assert status == "blocked"
    assert payload["passed"] is False
    assert autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS in payload["blocked_reasons"]


def test_missing_evidence_outranks_a_no_go_but_not_a_malformed_artifact():
    """Precedence: failed > blocked > completed_no_go > passed. A gate that
    never saw its evidence may not report the experiment's result -- but the
    thresholds that did fail stay visible in the payload."""
    no_go = [check("spans", 12, 500, ">=", group="coverage")]
    payload, status = evaluate(no_go, name="gate_a",
                               blocked_reasons=["blocked_missing_synthetic_calibration"])
    assert status == "blocked"
    assert payload["groups"]["coverage"] is False       # not hidden

    malformed = [check("partition_exact", 0, 1, "==", group="mechanical")]
    _, status = evaluate(malformed, name="gate_a",
                         blocked_reasons=["blocked_missing_synthetic_calibration"])
    assert status == "failed"


def test_agreement_refuses_to_pair_a_family_with_itself():
    candidates = _candidates(families=("whisper_dtw",))
    result = autoevidence.agreement_for_qualifying_pair(candidates, ["whisper_dtw"])
    assert result["n"] == 0
    assert "fewer than two independent" in result["reason"]


# --------------------------------------------------------------------------
# 5: natural-speech disagreement is not absolute error
# --------------------------------------------------------------------------
def test_natural_speech_metrics_are_named_as_disagreement_not_error():
    candidates = _candidates(offset_ms=120.0)
    result = autoevidence.cross_aligner_agreement(
        candidates, family_a="whisper_dtw", family_b="existing_ctc")
    assert result["measurement"] == "cross_aligner_disagreement"
    assert result["cross_aligner_start_disagreement_ms"] == pytest.approx(120.0, abs=1.0)
    for key in result:
        assert "absolute_boundary_error" not in key
        assert "ground_truth" not in key
    assert "not boundary error" in result["note"]


def test_the_guard_rejects_a_natural_speech_criterion_named_as_error():
    ok = [check("cross_aligner_boundary_disagreement_ms", 40.0, 100.0, "<=")]
    autoevidence.assert_no_absolute_error_claims(ok)          # does not raise
    bad = ok + [check("median_absolute_boundary_error_ms", 40.0, 100.0, "<=")]
    with pytest.raises(AssertionError, match="named as absolute error"):
        autoevidence.assert_no_absolute_error_claims(bad)


def test_two_aligners_sharing_a_bias_agree_perfectly_while_both_are_wrong():
    """The reason agreement may never stand in for accuracy."""
    candidates = _candidates(offset_ms=0.0)
    result = autoevidence.cross_aligner_agreement(
        candidates, family_a="whisper_dtw", family_b="existing_ctc")
    assert result["cross_aligner_boundary_disagreement_ms"] == 0.0
    # ... and the gate still has no accuracy evidence at all
    assert autoevidence.synthetic_calibration_status("/nonexistent")["available"] is False


# --------------------------------------------------------------------------
# 6: synthetic known boundaries produce true absolute error
# --------------------------------------------------------------------------
def test_missing_synthetic_calibration_is_explicit_and_never_substituted(tmp_path):
    status = autoevidence.synthetic_calibration_status(tmp_path)
    assert status["available"] is False
    assert status["reason"] == autoevidence.BLOCKED_MISSING_SYNTHETIC
    assert "NOT implemented" in status["missing_implementation"]
    # it names what is missing rather than gesturing at it
    assert "run_families" in status["missing_implementation"]
    assert autoevidence.SYNTHETIC_SCORES_FILE in status["missing_implementation"]


def _synthetic_row(family, *, edge="combined", purpose="gate", n=120,
                   within=0.94, bias=-12.0):
    return {"schema_version": autoevidence.SYNTHETIC_SCORES_SCHEMA,
            "family": family, "convention": "unit_edge_canonical",
            "edge": edge, "purpose": purpose, "num_boundaries": n,
            "within_100ms": within, "median_abs_error_ms": 40.0,
            "p90_abs_error_ms": 90.0, "median_signed_error_ms": bias}


def _write_scores(tmp_path, rows):
    path = tmp_path / autoevidence.SYNTHETIC_SCORES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_scored_synthetic_boundaries_yield_absolute_error(tmp_path):
    _write_scores(tmp_path, [
        _synthetic_row("existing_ctc", within=0.94, bias=-12.0),
        _synthetic_row("whisper_dtw", within=0.91, bias=-44.0),
    ])
    status = autoevidence.synthetic_calibration_status(
        tmp_path, min_boundaries=100, require_authentication=False)
    assert status["available"] is True
    assert status["sufficient_boundaries"] is True
    # only here is the phrase "absolute error" permitted
    assert status["absolute_boundary_error"]["within_100ms"] == pytest.approx(0.91)
    assert status["absolute_boundary_error"]["max_abs_bias_ms"] == pytest.approx(44.0)
    assert status["independence_classes_scored"] == [
        "ctc_forced_alignment", "whisper_cross_attention_dtw"]


def test_boundary_counts_are_not_double_counted_across_rows(tmp_path):
    """The same 120 boundaries described per family, convention and edge summed
    to 720 and sailed past a 100-boundary threshold."""
    _write_scores(tmp_path, [
        _synthetic_row("existing_ctc", edge=e, n=40) for e in ("start", "end", "combined")
    ] + [
        _synthetic_row("whisper_dtw", edge=e, n=40) for e in ("start", "end", "combined")
    ])
    status = autoevidence.synthetic_calibration_status(
        tmp_path, min_boundaries=100, require_authentication=False)
    assert status["num_boundaries"] == 40           # not 240
    assert status["sufficient_boundaries"] is False


def test_only_the_gate_set_is_scored_never_the_development_set(tmp_path):
    _write_scores(tmp_path, [_synthetic_row("existing_ctc", purpose="dev")])
    status = autoevidence.synthetic_calibration_status(
        tmp_path, require_authentication=False)
    assert status["available"] is False
    assert "purpose='gate'" in status["detail"]


@pytest.mark.parametrize("mutate,expected", [
    (lambda r: {k: v for k, v in r.items() if k != "p90_abs_error_ms"},
     autoevidence.BLOCKED_MISSING_SYNTHETIC),
    (lambda r: {**r, "schema_version": "something_else"},
     autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC),
    (lambda r: {**r, "within_100ms": 1.7},
     autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC),
    (lambda r: {**r, "median_abs_error_ms": float("nan")},
     autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC),
])
def test_a_malformed_synthetic_score_file_does_not_count_as_calibration(
        tmp_path, mutate, expected):
    _write_scores(tmp_path, [mutate(_synthetic_row("existing_ctc"))])
    status = autoevidence.synthetic_calibration_status(
        tmp_path, require_authentication=False)
    assert status["available"] is False
    assert status["reason"] == expected


def test_an_unauthenticated_synthetic_file_is_refused(tmp_path):
    """A five-column parquet dropped in by hand is not evidence."""
    _write_scores(tmp_path, [_synthetic_row("existing_ctc")])
    status = autoevidence.synthetic_calibration_status(
        tmp_path, require_authentication=True)      # no manifest was published
    assert status["available"] is False
    assert status["reason"] == autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC


def test_score_family_emits_exactly_what_gate_a_reads():
    """The helper used to emit `pct_within_100ms` and p95; the reader wanted
    `within_100ms` and p90, so a file produced as documented was rejected."""
    from csasr.lss.align.synthetic import score_family

    truth = np.arange(0.0, 1.0, 0.01)
    predicted = truth + 0.04
    row = score_family(predicted, truth, family="existing_ctc",
                       convention="unit_edge_canonical", edge="start")
    for column in autoevidence.SYNTHETIC_REQUIRED_COLUMNS:
        assert column in row, column
    assert row["schema_version"] == autoevidence.SYNTHETIC_SCORES_SCHEMA
    assert row["within_100ms"] == pytest.approx(1.0)
    assert row["p90_abs_error_ms"] == pytest.approx(40.0, abs=1.0)


def test_a_score_family_row_round_trips_through_gate_a(tmp_path):
    from csasr.lss.align.synthetic import score_family, scores_table

    truth = np.arange(0.0, 2.0, 0.01)
    rows = [score_family(truth + 0.02, truth, family=f,
                         convention="unit_edge_canonical", edge=e, purpose="gate")
            for f in ("existing_ctc", "whisper_dtw")
            for e in ("start", "end", "combined")]
    table = scores_table(rows)
    path = tmp_path / autoevidence.SYNTHETIC_SCORES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)

    status = autoevidence.synthetic_calibration_status(
        tmp_path, min_boundaries=100, require_authentication=False)
    assert status["available"] is True
    assert status["num_boundaries"] == 200
    assert status["absolute_boundary_error"]["within_100ms"] == pytest.approx(1.0)
    assert status["absolute_boundary_error"]["max_abs_bias_ms"] == pytest.approx(
        20.0, abs=1.0)


# --------------------------------------------------------------------------
# validity, coverage, monotonicity
# --------------------------------------------------------------------------
def test_nonmonotonic_spans_are_measured_per_family():
    candidates = _candidates(nonmonotonic_family="whisper_dtw")
    validity = autoevidence.family_validity(candidates)
    dtw = validity.set_index("aligner_family").loc["whisper_dtw"]
    ctc = validity.set_index("aligner_family").loc["existing_ctc"]
    assert dtw["nonmonotonic_rate"] > 0
    assert ctc["nonmonotonic_rate"] == 0.0


def test_unit_coverage_counts_units_any_family_aligned():
    candidates = _candidates(invalid_family="whisper_dtw")
    stats = autoevidence.unit_coverage(candidates)
    assert stats["reference_units"] == stats["covered_units"]
    assert stats["coverage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 10 + 11: forced-prerequisite taint propagates and blocks production
# --------------------------------------------------------------------------
def test_forced_prereq_taints_the_stage_and_every_descendant(tmp_path):
    write_status(tmp_path, "l0_freeze", "failed")
    # l1a forced past a failed l0
    with pytest.raises(PrerequisiteError):
        require_prerequisites(tmp_path, "l1a_diag")
    report = require_prerequisites(tmp_path, "l1a_diag", force=True)
    assert report["forced_run"] is True
    assert report["taint"]["diagnostic_only"] is True
    assert "forced_prerequisite" in report["taint"]["taint_reasons"]

    write_status(tmp_path, "l1a_diag", "passed", complete=True,
                 taint=report["taint"], run_dir="/runs/l1a-1")
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)

    # l1b's prerequisites are now all `passed`, but l1a's output is diagnostic
    with pytest.raises(PrerequisiteError, match="diagnostic-only"):
        require_prerequisites(tmp_path, "l1b_valid")


def test_taint_reasons_are_a_union_and_carry_parent_run_ids(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True,
                 run_dir="/runs/l0-1",
                 taint={"diagnostic_only": True, "taint_reasons": ["smoke_artifact"],
                        "parent_run_ids": []})
    write_status(tmp_path, "l1a_diag", "passed", complete=True,
                 run_dir="/runs/l1a-1",
                 taint={"diagnostic_only": True,
                        "taint_reasons": ["forced_prerequisite"],
                        "parent_run_ids": ["/runs/l0-1"]})
    taint = inherited_taint(tmp_path, "l1b_valid")
    assert taint["taint_reasons"] == ["forced_prerequisite", "smoke_artifact"]
    assert "/runs/l0-1" in taint["parent_run_ids"]
    assert "/runs/l1a-1" in taint["parent_run_ids"]


def test_a_retry_cannot_launder_a_tainted_stage(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    write_status(tmp_path, "l1a_diag", "passed", complete=True,
                 taint={"diagnostic_only": True,
                        "taint_reasons": ["forced_prerequisite"],
                        "parent_run_ids": []})
    # re-running l1a with clean prerequisites keeps the marker ...
    sticky = inherited_taint(tmp_path, "l1a_diag")
    assert sticky["diagnostic_only"] is True
    # ... unless the artifacts are actually recomputed
    fresh = inherited_taint(tmp_path, "l1a_diag", sticky=False)
    assert fresh["diagnostic_only"] is False


def test_production_gate_a_rejects_tainted_evidence():
    taint = {"diagnostic_only": True, "taint_reasons": ["forced_prerequisite"],
             "parent_run_ids": []}
    reasons = [autoevidence.BLOCKED_TAINTED_INPUTS] if taint["diagnostic_only"] else []
    payload, status = evaluate([check("everything_fine", 1, 1, "==")],
                               name="gate_a", blocked_reasons=reasons)
    assert status == "blocked"
    assert payload["passed"] is False


def test_legacy_forced_prereq_marker_is_still_recognised_as_taint():
    """Status files written before taint blocks existed must not read as clean."""
    assert taint_of({"forced_prereq": True})["diagnostic_only"] is True
    assert taint_of({"provenance": {"production_artifact": False}})["diagnostic_only"]
    assert taint_of({"status": "passed"})["diagnostic_only"] is False


# --------------------------------------------------------------------------
# 8 + 9: L0 partial and failed runs cannot unlock L1a
# --------------------------------------------------------------------------
def test_roles_only_is_terminal_but_cannot_unlock_l1a(tmp_path):
    write_status(tmp_path, "l0_freeze", "completed_roles_only", complete=True,
                 full_l0_pass=False)
    assert read_status(tmp_path, "l0_freeze")["status"] in ALLOWED
    assert exit_code("completed_roles_only") == 0        # nothing went wrong
    with pytest.raises(PrerequisiteError, match="completed_roles_only"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_a_partial_l0_that_reports_passed_still_cannot_unlock_l1a(tmp_path):
    """`--pilot-only` passes its own checks without building the roles."""
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=False)
    with pytest.raises(PrerequisiteError, match="partial-run"):
        require_prerequisites(tmp_path, "l1a_diag")


@pytest.mark.parametrize("status", ["failed", "completed_no_go", "blocked",
                                    "running", "insufficient_data"])
def test_no_l0_status_other_than_a_full_pass_unlocks_l1a(tmp_path, status):
    write_status(tmp_path, "l0_freeze", status, complete=True)
    with pytest.raises(PrerequisiteError):
        require_prerequisites(tmp_path, "l1a_diag")


def _publish_l0_artifacts(tmp_path, *, taint_reasons=()):
    """The artifacts L1a declares as prerequisites, with real manifests."""
    from csasr.lss import manifest as manifest_mod

    cfg = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "fake"}}
    (tmp_path / "freeze").mkdir(parents=True, exist_ok=True)
    freeze = tmp_path / "freeze" / "spec_freeze_v1.json"
    freeze.write_text("{}", encoding="utf-8")
    manifest_mod.publish(freeze, None, stage="l0_freeze", cfg=cfg,
                         run_dir=tmp_path / "runs" / "l0-1",
                         taint_reasons=taint_reasons)

    roles = tmp_path / "manifests" / "roles"
    roles.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"utterance_id": ["u1"]})
    manifest_mod.publish_frame(roles / "role_D-construct.parquet", frame,
                               stage="l0_freeze", cfg=cfg,
                               run_dir=tmp_path / "runs" / "l0-1",
                               taint_reasons=taint_reasons,
                               key_columns=("utterance_id",))
    return cfg


def test_a_full_l0_pass_does_unlock_l1a(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    _publish_l0_artifacts(tmp_path)
    report = require_prerequisites(tmp_path, "l1a_diag")
    assert report["satisfied"] is True
    assert report["taint"]["diagnostic_only"] is False


# --------------------------------------------------------------------------
# artifact-level authentication and taint (review P0-1, P0-2)
# --------------------------------------------------------------------------
def test_an_artifact_without_a_manifest_does_not_satisfy_a_prerequisite(tmp_path):
    """A parquet with no provenance may be a leftover from another
    configuration or a half-finished write; either way it is not evidence."""
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    (tmp_path / "freeze").mkdir(parents=True, exist_ok=True)
    (tmp_path / "freeze" / "spec_freeze_v1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifests" / "roles").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"utterance_id": ["u1"]}).to_parquet(
        tmp_path / "manifests" / "roles" / "role_D-construct.parquet", index=False)
    with pytest.raises(PrerequisiteError, match="missing_manifest"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_an_artifact_rewritten_after_its_stage_passed_is_detected(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    _publish_l0_artifacts(tmp_path)
    # somebody edits the freeze after L0 recorded `passed`
    (tmp_path / "freeze" / "spec_freeze_v1.json").write_text('{"changed": 1}',
                                                             encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="sha256_mismatch"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_overwrite_cannot_launder_taint_bound_to_reused_artifacts(tmp_path):
    """The P0 scenario: a forced run writes tainted artifacts, a later
    `--overwrite` clears the *status* taint, but the bytes were never rebuilt."""
    from csasr.lss.prereq import artifact_taint

    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    _publish_l0_artifacts(tmp_path, taint_reasons=("forced_prerequisite",))

    assert artifact_taint(tmp_path, "l1a_diag")["diagnostic_only"] is True
    # status history dropped, artifact taint retained
    laundered = inherited_taint(tmp_path, "l1a_diag", sticky=False)
    assert laundered["diagnostic_only"] is True
    assert "forced_prerequisite" in laundered["taint_reasons"]

    report = require_prerequisites(tmp_path, "l1a_diag", overwrite=True)
    assert report["taint"]["diagnostic_only"] is True


def test_rebuilding_the_artifacts_does_clear_the_taint(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    _publish_l0_artifacts(tmp_path, taint_reasons=("forced_prerequisite",))
    _publish_l0_artifacts(tmp_path)                 # recomputed cleanly
    report = require_prerequisites(tmp_path, "l1a_diag", overwrite=True)
    assert report["taint"]["diagnostic_only"] is False


def test_a_truncated_artifact_is_detected_by_its_recorded_key_set(tmp_path):
    from csasr.lss import manifest as manifest_mod

    cfg = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "fake"}}
    path = tmp_path / "candidates.parquet"
    full = pd.DataFrame({"utterance_id": [f"u{i}" for i in range(10)]})
    manifest_mod.publish_frame(path, full, stage="l1b_valid", cfg=cfg,
                               key_columns=("utterance_id",))
    assert manifest_mod.verify(path, frame=full)["ok"] is True

    # a prefix of the intended rows is still a valid, nonempty parquet
    full.head(4).to_parquet(path, index=False)
    frame, verdict = manifest_mod.read_verified(path)
    assert verdict["ok"] is False
    assert verdict["verdict"] in {manifest_mod.SHA_MISMATCH, manifest_mod.KEY_MISMATCH}
    assert not len(frame)


def test_an_artifact_from_another_configuration_is_refused(tmp_path):
    from csasr.lss import manifest as manifest_mod

    cfg = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "whisper-a"}}
    path = tmp_path / "candidates.parquet"
    manifest_mod.publish_frame(path, pd.DataFrame({"utterance_id": ["u1"]}),
                               stage="l1b_valid", cfg=cfg)
    other = {"experiment": {"output_root": str(tmp_path)}, "model": {"id": "whisper-b"}}
    frame, verdict = manifest_mod.read_verified(path, cfg=other, require_identity=True)
    assert verdict["verdict"] == manifest_mod.IDENTITY_MISMATCH
    assert not len(frame)


# --------------------------------------------------------------------------
# 14: the three terminal outcomes stay distinguishable
# --------------------------------------------------------------------------
def test_completed_no_go_blocked_and_failed_are_distinct_and_all_terminal():
    from csasr.utils.status import TERMINAL_STATUSES

    outcomes = {"completed_no_go", "blocked", "failed"}
    assert outcomes <= TERMINAL_STATUSES
    assert len({exit_code(s) for s in outcomes}) == 2      # only `failed` is nonzero
    assert exit_code("completed_no_go") == 0
    assert exit_code("blocked") == 0
    assert exit_code("failed") == 2

    coverage_fail = [check("spans", 12, 500, ">=", group="coverage")]
    mechanical_fail = [check("partition_exact", 0, 1, "==", group="mechanical")]
    assert classify(coverage_fail) == "completed_no_go"
    assert classify(mechanical_fail) == "failed"
    assert classify([check("ok", 1, 1, "==")], blocked=True) == "blocked"


def test_no_terminal_status_other_than_passed_satisfies_a_prerequisite(tmp_path):
    from csasr.lss.prereq import SATISFYING_STATUSES

    assert SATISFYING_STATUSES == ("passed",)
    for status in ("completed", "completed_no_go", "awaiting_manual_verdicts",
                   "blocked", "completed_roles_only", "failed"):
        write_status(tmp_path, "l1b_valid", status, complete=True)
        with pytest.raises(PrerequisiteError):
            require_prerequisites(tmp_path, "l1c_labels")


# --------------------------------------------------------------------------
# 1 + 2 + 3 + 13: automatic and manual workflows
# --------------------------------------------------------------------------
class _Args:
    """The subset of the parsed namespace `_resolve_parts` reads."""

    def __init__(self, **kw):
        self.mode = kw.get("mode", "automatic")
        self.only = kw.get("only")
        self.prepare_audit = kw.get("prepare_audit", False)
        self.prepare_manual_audit = kw.get("prepare_manual_audit", False)
        self.evaluate_gate = kw.get("evaluate_gate", False)


def _resolve(**kw):
    from csasr.experiments.lss_l1b_valid import _resolve_parts

    return _resolve_parts(_Args(**kw))


def test_the_default_run_prepares_and_evaluates_automatically():
    parts, mode, intent = _resolve()
    assert mode == "automatic"
    assert intent == "prepare_and_evaluate"
    assert "gate" in parts and "evidence" in parts
    assert "verdicts" not in parts
    assert "pack" not in parts        # no human pack is built unless asked


def test_automatic_preparation_decides_nothing_and_ends_successfully():
    from csasr.experiments.lss_l1b_valid import _next_action

    parts, mode, intent = _resolve(prepare_audit=True)
    assert mode == "automatic" and intent == "prepare"
    assert "gate" not in parts        # preparation emits no gate at all
    assert "verdicts" not in parts
    # a prepared automatic pack is a completed piece of work: exit 0, and the
    # next action is machine evaluation, not annotation
    assert exit_code("completed") == 0
    assert _next_action("passed", "automatic", []) == "run l1c_labels"


@pytest.mark.parametrize("kw", [
    {},
    {"prepare_audit": True},
    {"evaluate_gate": True},
    {"only": "verdicts"},
    {"only": "verdicts,gate", "evaluate_gate": True},
    {"mode": "automatic", "only": "synthetic,consensus,verdicts"},
])
def test_automatic_mode_never_reads_human_verdicts(kw):
    """Not even when explicitly asked: `--only verdicts` in automatic mode is
    dropped rather than honoured."""
    parts, mode, _ = _resolve(**kw)
    assert mode == "automatic"
    assert "verdicts" not in parts


def test_manual_preparation_is_the_only_path_that_waits_for_people():
    from csasr.experiments.lss_l1b_valid import _next_action

    parts, mode, intent = _resolve(prepare_manual_audit=True)
    assert mode == "manual" and intent == "prepare"
    assert "pack" in parts and "gate" not in parts
    assert exit_code("awaiting_manual_verdicts") == 0     # not a failed job
    assert _next_action("blocked", "manual",
                        [autoevidence.BLOCKED_MISSING_VERDICTS]) == "annotate_audit_pack"


def test_manual_evaluation_reads_verdicts_and_blocks_without_them():
    parts, mode, intent = _resolve(mode="manual", evaluate_gate=True)
    assert mode == "manual" and intent == "evaluate"
    assert "verdicts" in parts and "gate" in parts
    payload, status = evaluate(
        [check("spans", 900, 500, ">=", group="coverage")], name="gate_a",
        blocked_reasons=[autoevidence.BLOCKED_MISSING_VERDICTS])
    assert status == "blocked" and payload["passed"] is False


def test_a_preparation_only_run_cannot_report_gate_a_as_passed():
    """The regression: `--only synthetic,consensus,pack` used to emit a gate
    whose only criteria were mechanical, so Gate A recorded `passed` without a
    single piece of accuracy evidence and unlocked l1c."""
    parts, _, _ = _resolve(only="synthetic,consensus,pack")
    assert "gate" not in parts

    mechanical_only = [check("span_schema_valid", 1, 1, "=="),
                       check("audit_pack_is_blinded", 1, 1, "==")]
    # were such a gate ever evaluated, the missing evidence still blocks it
    _, status = evaluate(mechanical_only, name="gate_a",
                         blocked_reasons=[autoevidence.BLOCKED_MISSING_SYNTHETIC])
    assert status == "blocked"


# --------------------------------------------------------------------------
# 7: both jitter offsets are measured and both are gated
# --------------------------------------------------------------------------
def _spans(n=12, duration_ms=400.0):
    rows = []
    for i in range(n):
        start = int(i * 1.0 * SR)
        end = start + int(duration_ms / 1000.0 * SR)
        rows.append({
            "utterance_id": f"u{i // 3}", "unit_id": i, "language": "EN" if i % 2 else "ZH",
            "consensus_start_sample": start, "consensus_end_sample": end,
            "safe_interior_ms": duration_ms * 0.5,
        })
    return pd.DataFrame(rows)


def test_both_jitter_offsets_are_measured():
    from csasr.lss.align.jitter import jitter_stability

    cfg = {"jitter": {"offsets_ms": [50, 100]},
           "seeds": {"jitter": [1, 2]}}
    table, summary = jitter_stability(_spans(), cfg)
    assert set(summary) == {"offset_50ms", "offset_100ms"}
    for key in summary:
        assert np.isfinite(summary[key]["median_mask_iou"])
        assert np.isfinite(summary[key]["safe_interior_survival"])
    # a smaller perturbation must not damage the mask more than a larger one
    assert summary["offset_50ms"]["median_mask_iou"] >= \
        summary["offset_100ms"]["median_mask_iou"]


def test_both_jitter_offsets_are_gated_not_only_reported():
    """Proposal 4.1 requires stability at +/-50 AND +/-100 ms; only the 100 ms
    criteria were gated before."""
    gcfg = {"min_jitter_mask_iou_50ms": 0.60, "min_jitter_mask_iou_100ms": 0.60}
    criteria = []
    for offset in (50, 100):
        measured = {50: 0.80, 100: 0.42}[offset]
        criteria.append(check(f"jitter_median_mask_iou_{offset}ms", measured,
                              gcfg[f"min_jitter_mask_iou_{offset}ms"], ">=",
                              group="jitter"))
    assert classify(criteria) == "completed_no_go"
    names = {c["name"] for c in criteria}
    assert names == {"jitter_median_mask_iou_50ms", "jitter_median_mask_iou_100ms"}


def test_a_missing_jitter_measurement_fails_rather_than_passes():
    missing = [check("jitter_measured_50ms", 0, 1, "==", group="jitter")]
    assert classify(missing) == "completed_no_go"
    nan = [check("jitter_median_mask_iou_50ms", float("nan"), 0.6, ">=",
                 group="jitter")]
    assert classify(nan) == "completed_no_go"


def test_next_action_names_the_missing_evidence():
    from csasr.experiments.lss_l1b_valid import _next_action

    action = _next_action("blocked", "automatic",
                          [autoevidence.BLOCKED_MISSING_SYNTHETIC])
    assert "synthetic" in action and "--prepare-manual-audit" in action
    action = _next_action("blocked", "automatic",
                          [autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS])
    assert "second independent valid aligner" in action
