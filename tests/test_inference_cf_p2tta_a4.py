"""P2-TTA-A4 focused tests (freeze 37441eb): exact A3 panel reuse; canonical V_E/V_M token classes (disjoint,
target-identity only); safe teacher = q_A at EN positions, q_B elsewhere; forward-KL orientation and float64 identity;
theta0 anchor (non-EN KL exactly 0 and ~zero non-EN gradient); exactly 2 updates, LN-only, exact reset, fresh optimizer;
exact no-op (identical condition / no EN position); live independent agreement; forced-ZH final decode; references
closed before the seal; frozen thresholds, rescue, benefit and precedence."""
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
from test_inference_cf_p1 import CB, CE, PART  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2tta0 import EOS_T, TokT, _setup  # noqa: E402

import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import csasr.inference_cf.script_safe_tta as a4  # noqa: E402
import csasr.inference_cf.soft_auto_tta as a3  # noqa: E402
import experiments.inference_cf_p2tta_a4 as run  # noqa: E402
import experiments.inference_cf_p2tta_a4_analyze as an  # noqa: E402
import experiments.inference_cf_p2tta_a4_audit as au  # noqa: E402

C = json.loads(Path(run.CONFIG).read_text())
Y_MIX = [10, 45, 12, 46, 13]          # PART: matrix 8..43 (M), embedded 44..53 (E) -> M E M E M


def test_panel_is_exact_a3_panel():
    a3c = json.loads(Path("configs/inference_cf/p2_tta_a3.json").read_text())
    assert hashlib.sha256(Path(run.PANEL).read_bytes()).hexdigest() == C["panel"]["byte_sha256"] == a3c["panel"]["byte_sha256"] == \
        "6422741b91e64ff007458b7a04d2f559e8ed8ffb592a087a538adf63632b2206"
    p = json.loads(Path(run.PANEL).read_text())
    assert [r["utterance_id"] for r in p["rows"]] == C["panel"]["ids"] and sum(r["group"] == "D" for r in p["rows"]) == 12


def test_token_classes_canonical_and_disjoint():
    assert a4.token_classes(Y_MIX + [2, 60], PART) == ["M", "E", "M", "E", "M", "O", "O"] == au.own_classes(Y_MIX + [2, 60], PART)
    with pytest.raises(tta.TTAInvalid):
        a4.token_classes([1], {"embedded_ids": [1], "matrix_ids": [1]})


def test_safe_teacher_selection_and_forward_kl_float64():
    torch.manual_seed(0)
    V, T = 12, 4
    zA, zB, zP = (torch.randn(T + 1, V, dtype=torch.float64) * 3 for _ in range(3))
    sup, beg = [7], [5]
    lA, lB, lP = (a3.allowed_log_probs(z, T, sup, beg) for z in (zA, zB, zP))
    cls = ["E", "M", "O", "E"]
    S = a4.safe_teacher(lA, lB, cls)
    rows = lambda parts: [parts[0][0]] + list(parts[1]) if len(parts) > 1 else [parts[0][0]]
    for t, cl in enumerate(cls):
        exp = rows(lA)[t] if cl == "E" else rows(lB)[t]
        assert torch.equal(rows(S)[t], exp)
    kl = a3.kl_terms(S, lP)
    for t in range(T):
        q, p = rows(S)[t].exp(), rows(lP)[t].exp()
        fwd = float((q * (q.log() - p.log())).sum())
        ind = float(F.kl_div(rows(lP)[t], rows(S)[t], reduction="sum", log_target=True))
        assert float(kl[t]) == pytest.approx(fwd, abs=1e-12) == pytest.approx(ind, abs=1e-12)


def _a4(b, guard, cA, y, keep=False, counters=None):
    g = b.model.generation_config
    mask = tta.valid_mask(y, g.suppress_tokens, g.begin_suppress_tokens, {EOS_T}, EOS_T)
    return a4.adapt_a4(b.model, guard, _enc(), cA, CB, y, mask, a4.token_classes(y, PART), suppress=g.suppress_tokens,
                       begin=g.begin_suppress_tokens, eos=EOS_T, partition=PART, counters=counters, keep_grad0=keep)


def test_theta0_anchor_is_zero_and_non_en_gradient_vanishes():
    b, names, guard = _setup()
    lqB = a3.teacher_distribution(b.model, _enc(), CB, Y_MIX, [], [])
    masters = guard.fresh_masters()
    with torch.enable_grad():
        lp = a3.allowed_log_probs(tta.teacher_logits(b.model, _enc(), CB, Y_MIX, {n: m.to(torch.bfloat16) for n, m in masters.items()}),
                                  len(Y_MIX), [], [])
        klB = a3.kl_terms(lqB, lp)
        nonE = torch.tensor([c != "E" for c in a4.token_classes(Y_MIX, PART)])
        assert float(klB[nonE].abs().max()) <= 1e-6
        klB[nonE].mean().backward()
    gnon = math.sqrt(sum(float(m.grad.double().pow(2).sum()) for m in masters.values()))
    r = _a4(b, guard, CE, Y_MIX, keep=True)
    assert gnon <= 1e-5 * max(1.0, float(torch.linalg.vector_norm(r["grad0"].double())))
    assert r["log"]["d_anchor"][0] is not None and abs(r["log"]["d_anchor"][0]) <= 1e-6


def test_adapt_a4_mechanics_reset_and_determinism():
    b, names, guard = _setup()
    c = {}
    r1 = _a4(b, guard, CE, Y_MIX, counters=c)
    lg = r1["log"]
    assert lg["steps"] == 2 and lg["loss_evaluations"] == 3 and not lg["noop"] and (lg["n_E"], lg["n_M"], lg["n_O"]) == (2, 3, 0)
    assert c == {"teacher_forwards": 2, "student_forwards": 3, "backwards": 2, "optimizer_steps": 2}
    assert lg["losses"][0] == pytest.approx((2 * lg["d_E"][0] + 3 * lg["d_anchor"][0]) / 5, rel=1e-6)
    assert lg["d_E"][0] > 0 and lg["master_delta_l2"] > 0 and guard.verify()
    assert all(p.grad is None and not p.requires_grad for p in b.model.parameters())
    r2 = _a4(b, guard, CE, Y_MIX)
    assert r2["log"]["losses"] == lg["losses"] and all(torch.equal(r1["final_masters"][n], r2["final_masters"][n]) for n in names)
    assert lg["R_E"] == pytest.approx((lg["d_E"][0] - lg["d_E"][2]) / lg["d_E"][0])


def test_exact_noop_identical_condition_and_no_en_position():
    b, names, guard = _setup()
    for cA, y in ((CB, Y_MIX), (CE, [10, 12, 13])):
        lg = _a4(b, guard, cA, y)["log"]
        assert lg["noop"] and lg["losses"] == [0.0, 0.0, 0.0] and lg["grad_l2"] == [0.0, 0.0] and lg["master_delta_l2"] == 0
    assert guard.verify()


def test_live_independent_check_agrees():
    b, names, guard = _setup()
    r = _a4(b, guard, CE, Y_MIX, keep=True)
    la = au.live_safe_kl_check(b.model, names, _enc(), CE, CB, Y_MIX, PART, suppress=[], begin=[], tokenizer=TokT())
    assert abs(la["auditor_loss"] - r["log"]["losses"][0]) <= 1e-5
    assert float(torch.linalg.vector_norm(r["grad0"].double() - la["_grad"].double())) <= \
        max(1e-8, 0.02 * float(torch.linalg.vector_norm(la["_grad"].double())))


def test_runner_forced_decode_auditor_independent_and_references_closed(monkeypatch):
    src = Path("experiments/inference_cf_p2tta_a4.py").read_text()
    assert "dec = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)" in src
    asrc = Path("experiments/inference_cf_p2tta_a4_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_a4_analyze|script_safe_tta|soft_auto_tta|episodic_tta)", asrc, re.M) is None
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2tta_a4/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.evaluate("results/inference_cf/p2tta_a4/run1")
    assert not called


def test_frozen_thresholds_rescue_benefit_and_precedence():
    row = lambda e0, e2, a2, nE=10, nNE=5: {"d_E": [e0, 0, e2], "R_E": (e0 - e2) / e0, "n_E": nE, "n_nonE": nNE, "d_anchor": [0.0, 0, a2],
                                            "n_M": 2, "d_M": [0.0, 0, a2]}
    ok = [row(1.0, 0.6, 0.05)] * 9 + [row(1.0, 1.0, 0.0)] * 2
    en = an.en_transfer(ok, C)
    assert en["required"] == 9 and en["rows_decreased"] == 9 and en["pass"]
    assert not an.en_transfer([row(1.0, 0.6, 0)] * 8 + [row(1.0, 1.0, 0)] * 3, C)["pass"]
    assert not an.en_transfer([row(1.0, 0.76, 0)] * 11, C)["pass"]
    anc = an.anchor([row(1.0, 0.6, 0.1)] * 11, C)                        # G = 0.4, D_ANCHOR2 = 0.1 -> rho 0.25
    assert anc["rho_anchor"] == pytest.approx(0.25) and anc["pass"]
    assert not an.anchor([row(1.0, 0.6, 0.11)] * 11, C)["pass"]
    ref = C["partial_rescue"]["A3_reference"]
    good = {"zh_cer_increase": 0.015, "zh_retention": 0.98, "outside_harm_rate": ref["outside_harm_rate"]}
    pr = an.partial_rescue(good, C)
    assert pr["half_rescued"] == {"zh_cer": True, "retention": True, "outside": False} and pr["partial"]
    assert not an.partial_rescue({**good, "outside_harm_rate": ref["outside_harm_rate"] + 0.006}, C)["partial"]
    assert an.benefit(130, 128, 0.297, 0.292, 0, C)["pass"] and not an.benefit(131, 128, 0.29, 0.29, 0, C)["pass"]
    assert not an.benefit(128, 128, 0.2971, 0.292, 0, C)["pass"] and not an.benefit(120, 128, 0.2, 0.3, 1, C)["pass"]
    P, Fp = {"pass": True}, {"pass": False}
    assert an.decide(False, P, P, True, P) == "P2_TTA_A4_INVALID"
    assert an.decide(True, Fp, P, True, P) == "P2_TTA_A4_ENGLISH_TRANSFER_WEAK"
    assert an.decide(True, P, Fp, True, P) == "P2_TTA_A4_SHARED_PARAMETER_INTERFERENCE"
    assert an.decide(True, P, P, False, P) == "P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED"
    assert an.decide(True, P, P, True, Fp) == "P2_TTA_A4_SAFE_BUT_NO_GAIN"
    assert an.decide(True, P, P, True, P) == "P2_TTA_A4_PROMISING"
    assert C["safety"]["max_ZH_CER_increase"] == 0.015 and C["safety"]["min_matrix_ZH_retention"] == 0.98 and \
        C["safety"]["max_outside_POI_harm_rate"] == 0.03 and C["safety"]["max_MER_increase"] == 0.01
