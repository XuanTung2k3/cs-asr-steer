"""P2-TTA-FUNNEL focused tests (contract ff75e1a). Stage A (TTA0-R): original decision copied exactly and only the
numerical gradient tolerance repaired; sealed bytes unchanged; repaired integrity re-adjudicates only the live
gradient criterion (reference-free); references closed without the committed gate; R code path free of
adaptation/backward/optimizer/generation; auditor independence."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import experiments.inference_cf_p2tta0_analyze as tan
import experiments.inference_cf_p2tta_funnel_analyze as fan
import experiments.inference_cf_p2tta_funnel_audit as fau

C = json.loads(Path("configs/inference_cf/p2_tta_funnel.json").read_text())
T0 = json.loads(Path("configs/inference_cf/p2_tta0.json").read_text())


def test_original_decision_exact_and_only_tolerance_repaired():
    assert C["original_tta0_decision"] == T0["decision"]
    R = C["TTA0_R"]
    assert R["relative_gradient_tolerance"] == 0.02 and R["previous_relative_gradient_tolerance"] == 0.001
    assert R["absolute_loss_tolerance"] == 1e-5 and R["absolute_gradient_floor"] == 1e-8 and R["new_gpu_jobs"] == 0
    assert C["common"]["optimization"] == T0["optimization"] and C["common"]["trainables"] == T0["trainables"]
    assert C["common"]["teacher_forcing"] == T0["teacher_forcing"] and R["baseline20"] == T0["reuse"]["rows"]
    assert tan.TH == C["original_tta0_decision"]["thresholds"]


def test_sealed_tta0_bytes_unchanged():
    for p, h in C["TTA0_R"]["sealed_sha256"].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p


def test_repaired_integrity_readjudicates_only_live_gradient():
    data = tan.load(Path("results/inference_cf/p2tta0/run1"))
    orig, rep = tan.integrity(data), tan.integrity(data, 0.02)
    assert orig["live_audit_pass"] is False and orig["no_runtime_invalid"] is False
    assert rep["live_audit_pass"] is True and rep["no_runtime_invalid"] is True
    assert {k: v for k, v in orig.items() if k not in ("live_audit_pass", "no_runtime_invalid")} == \
        {k: v for k, v in rep.items() if k not in ("live_audit_pass", "no_runtime_invalid")}
    assert tan.integrity(data, 0.0102)["live_audit_pass"] is False          # 1.025% live A1 discrepancy is not hidden


def test_references_closed_without_committed_gate(monkeypatch):
    monkeypatch.setattr(fan, "R_GATE", "results/inference_cf/p2tta_funnel/tta0_r/__missing_gate__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        fan.r_evaluate()
    assert not called


def test_r_path_has_no_adaptation_and_auditor_is_independent():
    for rel in ("experiments/inference_cf_p2tta_funnel_analyze.py", "experiments/inference_cf_p2tta_funnel_audit.py"):
        src = Path(rel).read_text()
        assert not any(f in src for f in ("adapt(", ".backward(", "optim.", "generate(", "load_whisper", "forced_decode("))
    src = Path("experiments/inference_cf_p2tta_funnel_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta|inference_cf_p2tta_funnel\b)", src, re.M) is None
    assert fau.R_MAP == fan.R_MAP


def test_r_gate_runtime_code_scope_excludes_tests_only():
    """Gate attempt 1 (preserved) treated the post-run-extended TTA0 test file as runtime code; runtime code must still
    include the runner and episodic module and be byte-identical to the run1 manifest."""
    src = Path("experiments/inference_cf_p2tta_funnel_audit.py").read_text()
    assert 'not p.startswith("tests/")' in src and '"runtime_sources_unchanged"' in src
    man = json.loads(Path("results/inference_cf/p2tta0/run1/manifest.json").read_text())
    runtime = [p for p in man["sources"] if not p.endswith(("_analyze.py", "_audit.py")) and not p.startswith("tests/")]
    assert {"experiments/inference_cf_p2tta0.py", "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_cached.py"} <= set(runtime)
    for p in runtime:
        assert "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest() == man["sources"][p], p


def test_repaired_post_audit_source_scope_matches_gate():
    """R post-audit attempt 1 (preserved) failed sources_at_commit only on the post-run-extended TTA0 test file; in
    repaired mode current bytes may differ only for post-run analysis/audit extensions and tests, and every manifest
    source must still equal its blob at the run's manifest commit."""
    src = Path("experiments/inference_cf_p2tta0_audit.py").read_text()
    assert 'p.endswith(("_analyze.py", "_audit.py")) or p.startswith("tests/")' in src
    assert '"sha256:" + (blob_sha(m["git_commit"], p) or "") == h and' in src
