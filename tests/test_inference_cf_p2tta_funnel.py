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


# ---- Stage C: TTA1 ----------------------------------------------------------------------------------------------

def test_tta1_selection_inherits_exact_master_settings():
    sel = json.loads(Path("results/inference_cf/p2tta_funnel/tta0_r/selected_objective.json").read_text())
    assert sel["objective"]["id"] == "A2" and sel["objective"]["loss"] == T0["objectives"]["A2"]["loss"]
    assert sel["optimizer"] == {k: C["common"]["optimization"][k] for k in sel["optimizer"]}
    assert sel["optimizer"]["lr"] == 1e-3 and sel["optimizer"]["steps"] == 2 and sel["optimizer"]["weight_decay"] == 0
    assert [p["name"] for p in sel["trainables"]["parameters"]] == [p["name"] for p in C["common"]["trainables"]["parameters"]]
    assert sel["teacher_forcing"] == C["common"]["teacher_forcing"] and sel["decode"] == C["common"]["decode"]


def test_tta1_plan_reuse_all_compatible_and_panel100():
    plan = json.loads(Path("results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())
    panel = json.loads(Path(C["TTA1"]["panel"]).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    assert hashlib.sha256(Path(C["TTA1"]["panel"]).read_bytes()).hexdigest() == C["TTA1"]["panel_sha256"] and plan["ids"] == ids
    t0ids = json.loads(Path("results/inference_cf/p2tta0/run1/manifest.json").read_text())["ids"]
    assert sorted(plan["reuse_ids"]) == sorted(t0ids) and len(plan["new_ids"]) == 80 and set(plan["reuse_ids"]) | set(plan["new_ids"]) == set(ids)
    assert plan["planned"]["optimizer_steps"] == 160 <= C["compute"]["TTA1_optimizer_steps_max"]
    assert plan["objective"] == "A2" and plan["references_used"] is False and all(plan["checks"].values())


def test_tta1_decision_precedence_and_boundaries():
    base = {"mer_increase": 0.0, "zh_cer_increase": 0.0, "zh_retention": 0.99, "en_retention": 0.96, "outside_harm_rate": 0.02,
            "poi_corruption_rate": 0.02, "added_caps": 0, "new_severe_truncations": 0, "net_poi_error_reduction": 5, "pier_gain": 0.005}
    L = lambda valid=True, **k: fan.tta1_decide(valid, {**base, **k})["label"]
    assert L() == "P2_TTA1_SUPPORTED"
    assert L(net_poi_error_reduction=4) == "P2_TTA1_NO_USEFUL_GAIN" and L(pier_gain=0.0049) == "P2_TTA1_NO_USEFUL_GAIN"
    assert L(mer_increase=0.0101) == "P2_TTA1_SEQUENCE_DAMAGE" and L(new_severe_truncations=1) == "P2_TTA1_SEQUENCE_DAMAGE"
    assert L(added_caps=2) == "P2_TTA1_SEQUENCE_DAMAGE" and L(zh_retention=0.9799) == "P2_TTA1_SEQUENCE_DAMAGE"
    assert L(added_caps=1, zh_retention=None) == "P2_TTA1_SUPPORTED"
    assert L(valid=False) == "P2_TTA1_INVALID" and L(valid=False, mer_increase=1.0) == "P2_TTA1_INVALID"


def test_tta1_references_closed_without_seal(monkeypatch):
    monkeypatch.setattr(fan, "T1", "results/inference_cf/p2tta_funnel/__no_tta1__")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        fan.tta1_evaluate("results/inference_cf/p2tta_funnel/__no_tta1__/run1")
    assert not called


def test_tta1_runner_single_objective_no_steering():
    import ast
    src = Path("experiments/inference_cf_p2tta_funnel.py").read_text()
    code = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name.startswith("cmd_tta1"))
    assert '"A1"' not in code and '"A3"' not in code and "run_objective(bundle, guard, \"A2\"" in code
    assert not any(f in code for f in ("load_references", "ReadoutDirection", "native_lid", "_edit_hook", "num_beams=", "clip_grad"))
