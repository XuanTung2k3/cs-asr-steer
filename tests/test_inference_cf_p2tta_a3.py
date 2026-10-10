"""P2-TTA-A3 focused tests (freeze bd41bc5): reference-free 24-panel (primary builder and independent auditor replay);
forward-KL orientation, allowed-vocabulary masking, detached teacher, float64 identity with the independent formula,
fp32 repeat; exactly 2 updates, LN-only gradients, exact reset, fresh optimizer, identical-condition no-op; live
first-row agreement; forced-ZH final decode; references closed before the seal; frozen thresholds and precedence."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, PART, VOCAB  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2tta0 import EOS_T, TokT, _setup  # noqa: E402

import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import csasr.inference_cf.soft_auto_tta as a3  # noqa: E402
import experiments.inference_cf_p2tta_a3 as run  # noqa: E402
import experiments.inference_cf_p2tta_a3_analyze as an  # noqa: E402
import experiments.inference_cf_p2tta_a3_audit as au  # noqa: E402

C = json.loads(Path(run.CONFIG).read_text())


# ---- panel ------------------------------------------------------------------------------------------------------

def test_panel_exact_and_independently_replayed(monkeypatch):
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: (_ for _ in ()).throw(AssertionError("refs")))
    p = json.loads(Path(run.PANEL).read_text())
    assert hashlib.sha256(Path(run.PANEL).read_bytes()).hexdigest() == C["panel"]["byte_sha256"]
    rows = [(r["utterance_id"], r["dialogue_id"], r["group"]) for r in p["rows"]]
    assert rows == au.own_panel(C) and [r[0] for r in rows] == C["panel"]["ids"]
    assert sum(r[2] == "D" for r in rows) == 12 == p["parent_D"] and p["parent_A"] == 88 and len({r[1] for r in rows}) == 20
    f = json.loads(Path(run.FUNNEL).read_text())
    assert sorted(r[0] for r in rows if r[2] == "D") == sorted(b["utterance_id"] for b in f["baseline100"] if b["group"] == "D")


def test_panel_builder_rules_on_fixture():
    par = [{"utterance_id": f"u{i}", "dialogue_id": d} for i, d in enumerate("aabbccddeeffgg" * 2)]
    ft = {r["utterance_id"]: "x" for r in par}
    at = {r["utterance_id"]: ("y" if r["utterance_id"] in ("u0", "u1", "u2") else "x") for r in par}
    sel = run.build_panel(par, ft, at)
    assert [r["utterance_id"] for r in sel if r["group"] == "D"] == ["u0", "u1", "u2"]
    ctrl = [r for r in sel if r["group"] == "A"]
    assert [r["dialogue_id"] for r in ctrl[:5]] == list("cdefg")      # unrepresented dialogues first, first-appearance order
    assert ctrl[5]["dialogue_id"] == "b"                              # then D-represented dialogue with < 2 selected
    assert len(sel) == 24 and len({r["utterance_id"] for r in sel}) == 24


# ---- objective math ------------------------------------------------------------------------------------------------

def test_forward_kl_orientation_masking_and_float64_identity():
    torch.manual_seed(0)
    V, T = 12, 3
    zq, zp = torch.randn(T + 1, V, dtype=torch.float64) * 3, torch.randn(T + 1, V, dtype=torch.float64) * 3
    sup, beg = [7, 8], [5]
    lq, lp = a3.allowed_log_probs(zq, T, sup, beg), a3.allowed_log_probs(zp, T, sup, beg)
    kl = a3.kl_terms(lq, lp)
    for t in range(T):
        allowed = [v for v in range(V) if v not in sup and not (t == 0 and v in beg)]
        q, p = torch.softmax(zq[t, allowed], -1), torch.softmax(zp[t, allowed], -1)
        fwd, rev = float((q * (q.log() - p.log())).sum()), float((p * (p.log() - q.log())).sum())
        assert float(kl[t]) == pytest.approx(fwd, abs=1e-12) and abs(fwd - rev) > 1e-6
        ban = torch.zeros(V, dtype=torch.bool)
        ban[sup] = True
        if t == 0:
            ban[beg] = True
        lqm, lpm = torch.log_softmax(zq[t].masked_fill(ban, -math.inf), -1), torch.log_softmax(zp[t].masked_fill(ban, -math.inf), -1)
        ind = F.kl_div(torch.where(~ban, lpm, 0.0), torch.where(~ban, lqm, 0.0), reduction="none", log_target=True)
        assert float(torch.where(~ban, ind, 0.0).sum()) == pytest.approx(float(kl[t]), abs=1e-12)     # float64 identity


def _a3(b, guard, cA, y, mask=None, keep=False, counters=None):
    g = b.model.generation_config
    mask = mask if mask is not None else tta.valid_mask(y, g.suppress_tokens, g.begin_suppress_tokens, {EOS_T}, EOS_T)
    return a3.adapt_a3(b.model, guard, _enc(), cA, CB, y, mask, suppress=g.suppress_tokens, begin=g.begin_suppress_tokens,
                       eos=EOS_T, partition=PART, counters=counters, keep_grad0=keep)


def test_adapt_a3_two_steps_ln_only_reset_fresh_optimizer_and_detached_teacher():
    b, names, guard = _setup()
    y = [10, 11, 12, 13, 14]
    lq = a3.teacher_distribution(b.model, _enc(), CE, y, [], [])
    assert all(not x.requires_grad for x in lq)
    c = {}
    r1 = _a3(b, guard, CE, y, counters=c)
    lg = r1["log"]
    assert lg["steps"] == 2 and lg["loss_evaluations"] == 3 and lg["losses"] == lg["d_cond"] and lg["d_cond"][0] > 0
    assert c == {"teacher_forwards": 1, "student_forwards": 3, "backwards": 2, "optimizer_steps": 2}
    assert lg["master_delta_l2"] > 0 and guard.verify() and all(p.grad is None and not p.requires_grad for p in b.model.parameters())
    r2 = _a3(b, guard, CE, y)
    assert r2["log"]["d_cond"] == lg["d_cond"] and all(torch.equal(r1["final_masters"][n], r2["final_masters"][n]) for n in names)
    assert lg["relative_gap_reduction"] == pytest.approx((lg["d_cond"][0] - lg["d_cond"][2]) / lg["d_cond"][0])


def test_identical_condition_is_exact_noop():
    b, names, guard = _setup()
    r = _a3(b, guard, CB, [10, 11, 12])
    lg = r["log"]
    assert lg["noop"] and lg["identical_condition"] and lg["losses"] == [0.0, 0.0, 0.0] and lg["d_cond"][0] == 0.0
    assert lg["grad_l2"] == [0.0, 0.0] and lg["master_delta_l2"] == 0 and lg["effective_changed_scalars"] == 0


def test_live_independent_check_agrees_on_tiny_model():
    b, names, guard = _setup(suppress=[20, 21], begin=[10])
    y = [10, 12, 20, 13, 14]
    r = _a3(b, guard, CE, y, keep=True)
    g = b.model.generation_config
    la = au.live_kl_check(b.model, names, _enc(), CE, CB, y, suppress=g.suppress_tokens, begin=g.begin_suppress_tokens, tokenizer=TokT())
    assert la["valid"] == 3 and abs(la["auditor_loss"] - r["log"]["losses"][0]) <= 1e-5
    gd = float(torch.linalg.vector_norm(r["grad0"].double() - la["_grad"].double()))
    assert gd <= max(1e-8, 0.02 * float(torch.linalg.vector_norm(la["_grad"].double())))
    assert guard.verify()


# ---- runner / leakage / decision ----------------------------------------------------------------------------------

def test_runner_final_decode_forced_no_steering_and_references_closed(monkeypatch):
    src = Path("experiments/inference_cf_p2tta_a3.py").read_text()
    assert "dec = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)" in src
    assert au.cmd_prerun.__code__ is not None and re.search(r"^\s*(from|import)\s+\S*(_analyze|soft_auto_tta|episodic_tta)",
                                                             Path("experiments/inference_cf_p2tta_a3_audit.py").read_text(), re.M) is None
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2tta_a3/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.evaluate("results/inference_cf/p2tta_a3/run1")
    assert not called


def test_frozen_thresholds_and_precedence():
    gap = an.gap_closed([1.0] * 12, [0.7] * 9 + [1.0] * 3, C)
    assert gap["rows_decreased"] == 9 and gap["rows_decreased_required"] == 9 and gap["median_relative_reduction"] == pytest.approx(0.3)
    assert gap["pass"]
    assert not an.gap_closed([1.0] * 12, [0.7] * 8 + [1.0] * 4, C)["pass"]
    assert not an.gap_closed([1.0] * 12, [0.76] * 12, C)["pass"]                      # median 0.24 < 0.25
    assert an.gap_closed([0.0] * 12, [0.0] * 12, C)["pass"] is False
    assert an.movement([10] * 12, [9] * 4 + [11] + [10] * 7, C)["pass"]
    assert not an.movement([10] * 12, [9] * 4 + [11] * 2 + [10] * 6, C)["pass"]
    assert an.movement([10] * 12, [8] * 12, C)["R_dist"] == pytest.approx(0.2) and an.movement([10] * 12, [8.5] * 12, C)["pass"]
    assert not an.movement([10] * 12, [9] * 3 + [10] * 9, C)["pass"]
    safe = {"MER": True, "ZH_CER": True}
    G, NG = {"pass": True}, {"pass": False}
    assert an.decide(False, G, G, safe, {"promising_condition": True}) == "P2_TTA_A3_INVALID"
    assert an.decide(True, NG, G, safe, {"promising_condition": True}) == "P2_TTA_A3_GAP_NOT_CLOSED"
    assert an.decide(True, G, NG, {"MER": False}, None) == "P2_TTA_A3_SEQUENCE_LEVERAGE_LIMIT"
    assert an.decide(True, G, G, {"MER": False}, {"promising_condition": True}) == "P2_TTA_A3_TEACHER_UNSAFE"
    assert an.decide(True, G, G, safe, {"promising_condition": False}) == "P2_TTA_A3_CONDITIONING_ONLY"
    assert an.decide(True, G, G, safe, {"promising_condition": True}) == "P2_TTA_A3_PROMISING"
    assert an.vs_a2(10, 10, 0.30, 0.29, C)["promising_condition"] and not an.vs_a2(10, 10, 0.3001, 0.29, C)["promising_condition"]
    assert an.vs_a2(11, 10, 0.2, 0.3, C)["no_advantage"] and not an.vs_a2(9, 10, 0.32, 0.30, C)["promising_condition"]
    S = C["safety"]
    assert (S["max_MER_increase"], S["max_ZH_CER_increase"], S["min_matrix_ZH_retention"], S["min_embedded_EN_retention"],
            S["max_outside_POI_harm_rate"], S["max_POI_corruption_rate"], S["max_additional_caps"], S["max_new_severe_truncations"]) == \
        (0.01, 0.015, 0.98, 0.95, 0.03, 0.05, 1, 0)
