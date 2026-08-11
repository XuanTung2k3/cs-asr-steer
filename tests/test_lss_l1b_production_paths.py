"""End-to-end Gate-A runs: every way a false pass could happen, closed.

These call `lss_l1b_valid.main()` against a temporary artifacts root with mocked
aligner output, so they exercise the production assembly -- which criteria are
constructed, which blockers fire, what status and exit code come out, and
whether `l1c` is unlocked afterwards -- rather than the helpers underneath it.

The helpers are tested in `test_lss_gate_a_auto.py`; a helper that behaves
correctly while production never calls it is exactly the gap this file covers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csasr.experiments import lss_l1b_valid
from csasr.lss import manifest as manifest_mod
from csasr.lss.align import autoevidence
from csasr.lss.prereq import PrerequisiteError, require_prerequisites
from csasr.nat5h.schema import ALIGNMENT_SCHEMA_VERSION
from csasr.utils.status import read_status, write_status

SR = 16000
UNITS_PER_UTT = 4


# --------------------------------------------------------------------------
# a minimal but complete artifacts root
# --------------------------------------------------------------------------
def _manifest(n_utterances: int, role: str = "D-construct") -> pd.DataFrame:
    return pd.DataFrame([{
        "utterance_id": f"{role}_u{i}",
        "conversation_id": f"c{i // 5}",
        "speaker_id": f"c{i // 5}",
        "audio_path": f"/nonexistent/{role}_u{i}.wav",
        "duration_sec": 4.0,
        "transcript_raw": "我 用 machine learning 做 这个",
        "contains_code_switch": True,
        "role": role,
    } for i in range(n_utterances)])


def _candidate_rows(manifest: pd.DataFrame, families, *, offset_ms=0.0,
                    invalid_family=None, drop_utterances=0, languages=("ZH", "EN")):
    rows = []
    keep = manifest.iloc[drop_utterances:]
    for family in families:
        for _, utt in keep.iterrows():
            for unit in range(UNITS_PER_UTT):
                shift = offset_ms / 1000.0 if family == "whisper_dtw" else 0.0
                start = unit * 0.5 + shift
                rows.append({
                    "schema_version": ALIGNMENT_SCHEMA_VERSION,
                    "utterance_id": utt["utterance_id"],
                    "conversation_id": utt["conversation_id"],
                    "split": "train",
                    "role": utt["role"],
                    "reference_unit_index": unit,
                    "reference_language": languages[unit % len(languages)],
                    "reference_text": "machine" if unit % 2 else "我",
                    "aligner_family": family,
                    "aligner_variant": "default",
                    "start_sec": float(start),
                    "end_sec": float(start + 0.4),
                    "audio_duration_sec": 4.0,
                    "start_sample": int(start * SR),
                    "end_sample": int((start + 0.4) * SR),
                    "sample_rate": SR,
                    "waveform_num_samples": int(4.0 * SR),
                    "coordinate_policy": "short_wav_no_crop_v1",
                    "is_valid": family != invalid_family,
                    "failure_code": "" if family != invalid_family else "no_alignment",
                    "failure_detail": "",
                    "source_token_count": 1,
                    "mapping_method": "test",
                    "model_id": "fake", "model_revision": "0",
                    "code_commit": "unknown", "config_hash": "test",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An artifacts root with L0 and L1a passed and their artifacts published."""
    artifacts = tmp_path / "artifacts_lss"
    (artifacts / "status").mkdir(parents=True, exist_ok=True)
    cfg_stub = {"experiment": {"output_root": str(artifacts)}, "model": {"id": "fake"}}

    write_status(artifacts, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    write_status(artifacts, "l1a_diag", "passed", complete=True, full_stage_pass=True)

    freeze = artifacts / "freeze" / "spec_freeze_v1.json"
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps({"sha256": "x"}), encoding="utf-8")
    manifest_mod.publish(freeze, None, stage="l0_freeze", cfg=cfg_stub)

    roles_dir = artifacts / "manifests" / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    for role in ("D-construct", "loc-train", "D-dev-select", "D-dev-confirm"):
        manifest_mod.publish_frame(roles_dir / f"role_{role}.parquet",
                                   _manifest(30, role), stage="l0_freeze",
                                   cfg=cfg_stub, key_columns=("utterance_id",))

    summary = artifacts / "metrics" / "l1a_family_language_summary.parquet"
    summary.parent.mkdir(parents=True, exist_ok=True)
    manifest_mod.publish_frame(summary, pd.DataFrame([{"aligner_family": "existing_ctc",
                                                       "valid_units": 100}]),
                               stage="l1a_diag", cfg=cfg_stub)
    return artifacts


def _resolved_cfg(root: Path) -> dict:
    """The exact config L1b will load, so the identity check is a real test.

    Publishing with a hand-made stub would always mismatch, which would make
    every one of these tests pass for the wrong reason.
    """
    from csasr.utils.config import load_config

    return load_config("lss/l1b_valid.yaml", [f"experiment.output_root={root}"])


def _publish_candidates(root: Path, frame: pd.DataFrame, *, cfg=None,
                        with_manifest: bool = True, taint_reasons=()):
    path = root / "alignments" / "candidates_all.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_manifest:
        manifest_mod.publish_frame(
            path, frame, stage="l1b_valid", cfg=cfg or _resolved_cfg(root),
            taint_reasons=taint_reasons,
            key_columns=lss_l1b_valid.CANDIDATE_KEYS)
    else:
        frame.to_parquet(path, index=False)
    return path


def _run(root: Path, *extra: str) -> tuple[int, dict]:
    """Run L1b's real entry point against this root, and read what it wrote."""
    argv = ["--config", "lss/l1b_valid.yaml",
            "--set", f"experiment.output_root={root}",
            "--only", "consensus,evidence,jitter,coverage,gate", *extra]
    code = lss_l1b_valid.main(argv)
    return code, read_status(root, "l1b_valid")


def _assert_not_a_pass(status: dict, code: int):
    assert status["status"] != "passed", status.get("blocked_reasons")
    assert (status.get("gate") or {}).get("passed") is not True
    assert code in (0, 2)


def _assert_l1c_locked(root: Path):
    with pytest.raises(PrerequisiteError):
        require_prerequisites(root, "l1c_labels")


# --------------------------------------------------------------------------
# the false-pass paths
# --------------------------------------------------------------------------
def test_two_valid_aligners_but_disjoint_units_cannot_pass(root):
    """Both families qualify, candidate-relative coverage is 1.0, and not one
    unit was aligned by both -- so there is no cross-aligner evidence at all."""
    manifest = _manifest(30)
    ctc = _candidate_rows(manifest.iloc[:15], ["existing_ctc"])
    dtw = _candidate_rows(manifest.iloc[15:], ["whisper_dtw"])
    _publish_candidates(root, pd.concat([ctc, dtw], ignore_index=True))

    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_NO_PAIRED_AGREEMENT in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_empty_consensus_cannot_pass(root):
    """Nothing reaches the primary bins, so jitter, coverage and span-schema
    criteria have nothing to run on. Omitting them was a vacuous pass."""
    manifest = _manifest(30)
    # 5 s apart: no unit will ever reach consensus
    _publish_candidates(root, _candidate_rows(manifest, ["existing_ctc", "whisper_dtw"],
                                              offset_ms=5000.0))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_an_unauthenticated_candidate_table_cannot_pass(root):
    """A nonempty parquet with no manifest could be anything."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]),
                        with_manifest=False)
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_tainted_candidates_cannot_pass(root):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]),
                        taint_reasons=("forced_prerequisite",))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_TAINTED_INPUTS in status["blocked_reasons"]
    assert status["taint"]["diagnostic_only"] is True
    _assert_l1c_locked(root)


def test_four_percent_invalid_rows_cannot_pass(root):
    """Above the proposal's 1% limit, below the 0.95 coverage floor."""
    frame = _candidate_rows(_manifest(30), ["existing_ctc", "whisper_dtw"])
    rows = frame.index[frame["aligner_family"] == "whisper_dtw"][:5]
    frame.loc[rows, "is_valid"] = False
    _publish_candidates(root, frame)
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    names = {c["name"] for c in status["gate"]["criteria"]}
    assert "invalid_rate_whisper_dtw" in names
    _assert_l1c_locked(root)


def test_omitted_utterances_are_visible_in_coverage(root):
    """Half the intended universe never aligned. Candidate-relative coverage
    would read 1.0; the expected-universe denominator reads 0.5."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              drop_utterances=15))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    coverage = [c for c in status["gate"]["criteria"]
                if c["name"] == "alignment_unit_coverage"]
    assert coverage and coverage[0]["passed"] is False
    _assert_l1c_locked(root)


def test_one_language_missing_cannot_pass(root):
    """An EN-ZH comparison with only one language compares nothing."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              languages=("ZH",)))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_MISSING_LANGUAGE in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_the_previously_observed_one_valid_aligner_case_remains_a_no_pass(root):
    """existing_ctc valid, whisper_dtw not. The recorded production state."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              invalid_family="whisper_dtw"))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_perfect_agreement_still_cannot_pass_without_absolute_error(root):
    """The end-to-end version of "a shared bias agrees perfectly and is wrong":
    two families, identical spans, full coverage -- and still no accuracy
    evidence, so the gate blocks rather than passing."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    # the two families agree to the sample, every unit is covered, and there is
    # still no source of absolute error
    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    assert evidence["independence"]["n_independent_valid_families"] == 2
    assert evidence["cross_aligner_agreement"][
        "cross_aligner_boundary_disagreement_ms"] == 0.0
    assert autoevidence.BLOCKED_MISSING_SYNTHETIC in status["blocked_reasons"]
    _assert_l1c_locked(root)
    assert not (root / "freeze" / "l1b_spans_freeze.json").exists()


def test_a_blocked_gate_is_a_completed_command_not_a_failed_job(root):
    """Exit 0 for a gate that could not conclude: the job did its work."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code = lss_l1b_valid.main([
        "--config", "lss/l1b_valid.yaml", "--set", f"experiment.output_root={root}",
        "--only", "consensus,evidence,jitter,gate"])
    status = read_status(root, "l1b_valid")
    assert status["status"] == "blocked", \
        [c["name"] for c in status["gate"]["criteria"] if not c["passed"]]
    assert code == 0
    assert autoevidence.BLOCKED_MISSING_SYNTHETIC in status["blocked_reasons"]
    _assert_l1c_locked(root)


# --------------------------------------------------------------------------
# workflow, status and exit-code behaviour on the production path
# --------------------------------------------------------------------------
def test_automatic_preparation_exits_zero_and_never_opens_a_verdict_file(root, monkeypatch):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))

    def explode(*a, **k):                      # pragma: no cover - must not run
        raise AssertionError("automatic mode read a human verdict file")

    monkeypatch.setattr("csasr.lss.audit.verdicts.ingest", explode)
    code = lss_l1b_valid.main([
        "--config", "lss/l1b_valid.yaml", "--set", f"experiment.output_root={root}",
        "--prepare-audit", "--only", "consensus,evidence"])
    status = read_status(root, "l1b_valid")
    assert code == 0
    assert status["status"] == "completed"
    assert status["mode"] == "automatic"
    assert status["next_action"] == "evaluate_gate_a"
    assert status.get("gate") is None, "a preparation run decides nothing"


def test_manual_pack_in_automatic_mode_is_refused(root):
    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    with pytest.raises(SystemExit, match="manual workflow"):
        lss_l1b_valid.main(["--config", "lss/l1b_valid.yaml",
                            "--set", f"experiment.output_root={root}",
                            "--only", "pack"])


def test_an_unhandled_exception_is_recorded_as_failed_not_left_running(root, monkeypatch):
    """A crash used to leave `running`, which reads as an in-flight job forever."""
    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    monkeypatch.setattr(lss_l1b_valid.consensus_prod, "tolerance_sweep",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = lss_l1b_valid.main(["--config", "lss/l1b_valid.yaml",
                               "--set", f"experiment.output_root={root}",
                               "--only", "consensus,gate"])
    status = read_status(root, "l1b_valid")
    assert code == 2
    assert status["status"] == "failed"
    assert "boom" in status["error"]
    _assert_l1c_locked(root)


def test_the_lock_is_taken_before_the_status_is_claimed(root):
    """A second job that cannot get the lock must not overwrite the active
    run's status on its way out."""
    from csasr.utils.status import StageLock

    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    write_status(root, "l1b_valid", "passed", complete=True)
    with StageLock(root, "l1b_valid"):
        code = lss_l1b_valid.main(["--config", "lss/l1b_valid.yaml",
                                   "--set", f"experiment.output_root={root}",
                                   "--only", "consensus"])
    assert code == 2, "the refused run reports that it did not do its work"
    assert read_status(root, "l1b_valid")["status"] == "passed", \
        "the refused run must not overwrite the active run's status"
