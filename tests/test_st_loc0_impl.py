"""ST-LOC0 implementation tests: production self-site adapter, cached pulse/restore lineage, no-edit rules,
runtime firewall, arm matrix, calibration builder reproduction, label/winner precedence and auditor independence."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from csasr.inference_cf.loc0_sites import (Composite, SelfAttnResidualInterventionHook, assert_no_any_site_hooks, cache_fingerprint,
                                           cross_pulse_hook, pulse_action)
from csasr.lss.sites import DecoderPostCrossAttnRecorder

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/inference_cf/st_loc0.json").read_text())
PANEL = json.loads((ROOT / "docs/inference_cf/ST_LOC0_PANEL.json").read_text())


def _branch(bundle):
    import experiments.inference_cf_p2r as p2r
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(3)
    feats = torch.randn(1, 8, 100).to(bundle.dtype)
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=feats).last_hidden_state
    B = p2r.DiagBranch(bundle, BaseModelOutput(last_hidden_state=h), [1, 5, 7, 9], "B")
    B.step([1, 5, 7, 9], attention=False)
    B.step([11], attention=False)
    return B


def _step(B, new, layer, hook=None):
    rec = DecoderPostCrossAttnRecorder(B.bundle, [layer], keep_last_only=True)
    logits, _, _ = B.step(new, attention=False, hook=Composite(hook, rec))
    assert_no_any_site_hooks(B.bundle)
    return logits, rec.q_states[layer][0, -1].float().clone(), rec.states[layer][0, -1].float().clone()


@pytest.mark.parametrize("site", ["POST_SELF_ATTENTION", "POST_CROSS_ATTENTION_PRE_FFN"])
def test_cached_single_pulse_zero_identity_energy_and_restore(tiny_bundle, site):
    import experiments.inference_cf_p2r as p2r
    b = copy.deepcopy(tiny_bundle)
    layer = 1
    B = _branch(b)
    L, new = B.length, [13]
    query = L + len(new) - 1
    fp0 = cache_fingerprint(B.cache, L)
    base, q0, r0 = _step(B, new, layer)
    site0 = q0 if site == "POST_SELF_ATTENTION" else r0
    v = torch.zeros(16)
    v[3] = 1.0

    def make(v_fn, info, target):
        act = pulse_action(query, v_fn, target, info, 0.02, p2r.solve_scale, p2r.scaled_direction)
        return cross_pulse_hook(b, layer, act, 4) if site != "POST_SELF_ATTENTION" else \
            SelfAttnResidualInterventionHook(b, layer, act, num_forced_prefix=4)

    # zero dose: bitwise baseline, recorded as no-edit
    B.crop(L)
    info0: dict = {}
    z, _, _ = _step(B, new, layer, make(lambda r: (None, "zero_dose"), info0, 1.0))
    assert torch.equal(z, base) and info0["no_edit_reason"] == "zero_dose"
    # matched-energy pulse: only the query, consumed edit at the target, solver == hook
    target = 0.3 * float(site0.norm())
    B.crop(L)
    info: dict = {}
    hook = make(lambda r: (v.clone(), "ok"), info, target)
    la, q1, r1 = _step(B, new, layer, hook)
    assert not torch.equal(la, base)
    assert info.get("no_edit_reason") is None and info["status"] == "ok"
    rec = hook.records[-1]
    rec = rec.to_dict() if hasattr(rec, "to_dict") else rec
    consumed = (q1 if site == "POST_SELF_ATTENTION" else r1) - site0
    assert rec["steered"] and abs(rec["edit_norm"] / target - 1) <= 0.02
    assert abs(float(consumed.double().norm()) - rec["edit_norm"]) <= 5e-3 * rec["edit_norm"]
    assert abs(info["edit_norm"] - rec["edit_norm"]) <= 1e-6 * max(1.0, rec["edit_norm"])
    assert torch.allclose(info["_proposed"], site0 + consumed, atol=1e-5)
    if site == "POST_SELF_ATTENTION":      # norm-preserving repair of the consumed q
        assert abs(float(q1.norm()) - float(q0.norm())) <= 1e-4 * float(q0.norm())
    # restore: prefix cache unchanged, clean replay bitwise
    B.crop(L)
    assert cache_fingerprint(B.cache, L) == fp0 and B.positions == list(range(L))
    again, q2, r2 = _step(B, new, layer)
    assert torch.equal(again, base) and torch.equal(q2, q0) and torch.equal(r2, r0)
    assert all(p.grad is None for p in b.model.parameters())


def test_cross_pulse_reuses_p2r_hook_bitwise(tiny_bundle):
    import experiments.inference_cf_p2r as p2r
    b = copy.deepcopy(tiny_bundle)
    B = _branch(b)
    L, new = B.length, [13]
    query = L
    _, _, r0 = _step(B, new, 1)
    v = torch.linspace(-1, 1, 16)
    target = 0.25 * float(r0.norm())
    out = []
    for mk in ("p2r", "loc0"):
        B.crop(L)
        info: dict = {}
        hook = p2r.pulse_hook(b, 1, query, lambda r: (v.clone(), "ok"), target, 4, info) if mk == "p2r" else \
            cross_pulse_hook(b, 1, pulse_action(query, lambda r: (v.clone(), "ok"), target, info, 0.02, p2r.solve_scale, p2r.scaled_direction), 4)
        out.append(_step(B, new, 1, hook))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][2], out[1][2])


def test_cache_fingerprint_detects_prefix_change(tiny_bundle):
    b = copy.deepcopy(tiny_bundle)
    B = _branch(b)
    fp = cache_fingerprint(B.cache, B.length)
    leg = B.cache.self_attention_cache
    k = leg.layers[0].keys if hasattr(leg, "layers") else leg.key_cache[0]
    with torch.inference_mode():
        k[0, 0, 0, 0] += 1.0
    assert cache_fingerprint(B.cache, B.length) != fp


def test_no_edit_rules_are_predetermined():
    r = torch.ones(1, 3, 8)
    pos = torch.tensor([3, 4, 5])
    cases = [((None, "tiny_or_nonfinite_V1_delta"), None, "tiny_or_nonfinite_V1_delta"),
             ((torch.ones(8), "ok"), {"status": "energy_unreachable", "s": 0.0, "edit_norm": 0.0, "evals": 0}, "energy_unreachable"),
             ((torch.ones(8), "ok"), {"status": "ok", "s": 1.0, "edit_norm": 0.5, "evals": 8, "rel_sq_err": 0.5},
              "solver_target_unattainable_within_8_evaluations")]
    for vret, sol, reason in cases:
        info: dict = {}
        act = pulse_action(4, lambda x, vret=vret: vret, 1.0, info, 0.02, lambda *a, sol=sol: dict(sol), lambda v, s, like: (s * v).to(like.dtype))
        g, d = act(r=r, abs_pos=pos)
        assert float(g.abs().sum()) == 0 and float(d.abs().sum()) == 0 and info["no_edit_reason"] == reason
    assert set(CFG["cell_eligibility"]["reasons_for_no_edit"]) == {c[2] for c in cases} | {"invalid_new_site_D1_fold"}


def test_runtime_projection_firewall_and_arm_matrix():
    import experiments.inference_cf_st_loc0 as R
    rp = R.runtime_projection(PANEL)
    s = json.dumps(rp)
    assert set(rp) == {"runtime_queries", "utterances"}
    for f in ("stratum", "EN-confusion", "target_ids", "competitor", "calibration_membership", "construction_positions"):
        assert f not in s
    arms = R.arm_table(CFG)
    assert len(arms) == 21 and len({a["id"] for a in arms}) == 21
    assert [a["id"] for a in arms[:16]] == [a["id"] for a in CFG["primary_arms"]]
    assert {a["family"] for a in arms[16:]} == {"random", "D2"}
    assert len(R.random_vectors(CFG)) == 4
    assert tuple(R.BARRIER_ARMS[:2]) == ("v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "v_unq_L16_POST_CROSS_ATTENTION_PRE_FFN_plus")
    run_src = (ROOT / "experiments/inference_cf_st_loc0.py").read_text()
    for f in ("positions.json", "target_ids", "competitor", "p2rj", "logit_metrics", '"stratum"'):
        assert f not in run_src
    assert R.SITE_KEYS[(16, "POST_CROSS_ATTENTION_PRE_FFN")] == "L16_CROSS"


def test_calibration_builder_reproduces_historical_l16_folds():
    import experiments.inference_cf_st_loc0_calibrate as C
    H = np.load(ROOT / "results/inference_cf/p2dir/extract_run1/states.npz")["H_B"]
    folds = C.fit_site({k: PANEL[k] for k in C.ALLOWED}, H)
    hist = json.loads((ROOT / "results/inference_cf/p2dir/folds_run1/folds.json").read_text())["folds"]
    assert sorted(folds) == sorted(hist)
    for d, f in folds.items():
        assert f["summary"]["status"] == hist[d]["status"] == "ok"
        assert f["summary"]["selected_index"] == hist[d]["selected_index"]
        assert float(np.max(np.abs(f["vector"] - np.load(ROOT / hist[d]["vector_path"])))) <= 2e-6
        assert f["summary"]["moment_hashes"] == hist[d]["moment_hashes"]


def _arm(i, **kw):
    a = {"id": f"a{i:02d}", "eligible": True, "integrity_pass": True, "adj_lower": 0.1, "macro": 0.6, "corrections": 3,
         "correction_dialogues": 3, "correct_corruptions": 0}
    a.update(kw)
    return a


def test_label_precedence_and_winner_order_agree_with_independent_auditor():
    from experiments.inference_cf_st_loc0_analyze import decide
    from experiments.inference_cf_st_loc0_audit import label_logic
    rng = np.random.default_rng(0)
    for trial in range(400):
        arms = [_arm(i, eligible=bool(rng.random() < .8), adj_lower=float(rng.normal(0, .2)), macro=float(rng.normal(.4, .3)),
                     corrections=int(rng.integers(0, 6)), correction_dialogues=int(rng.integers(0, 5)), correct_corruptions=int(rng.integers(0, 4)))
                for i in range(16)]
        valid = bool(rng.random() < .9)
        dec = decide(valid, arms, CFG)
        mine = [{"id": a["id"], "eligible": a["eligible"], "lo": a["adj_lower"], "est": a["macro"], "corr": a["corrections"],
                 "corr_d": a["correction_dialogues"], "damage": a["correct_corruptions"]} for a in arms]
        assert (dec["label"], dec["winner"]) == label_logic(mine, valid, CFG)
    assert decide(True, [_arm(0, adj_lower=0.0)], CFG)["label"] == "ST_LOC0_NO_FIXED_DIRECTION_LEVER"
    assert decide(True, [_arm(0, eligible=False)], CFG)["label"] == "ST_LOC0_NO_FIXED_DIRECTION_LEVER"
    assert decide(True, [_arm(0, macro=0.49)], CFG)["label"] == "ST_LOC0_LOCAL_EFFECT_ONLY"
    assert decide(True, [_arm(0, correction_dialogues=2)], CFG)["label"] == "ST_LOC0_LOCAL_EFFECT_ONLY"
    assert decide(False, [_arm(0)], CFG)["label"] == "ST_LOC0_INVALID"
    # exhaustive winner tie-break order
    keys = list(itertools.product([3, 4], [3, 4], [0.5, 0.7], [0, 1]))
    arms = [_arm(i, corrections=c, correction_dialogues=d, macro=m, correct_corruptions=k) for i, (c, d, m, k) in enumerate(keys)]
    arms.append(_arm(99, corrections=4, correction_dialogues=4, macro=0.7, correct_corruptions=0))
    assert decide(True, arms, CFG)["winner"] == "a" + f"{[i for i, k in enumerate(keys) if k == (4, 4, 0.7, 0)][0]:02d}"


def test_eligibility_rule_and_bonferroni_quantiles():
    from experiments.inference_cf_st_loc0_analyze import eligible
    ok = {"EN-confusion": 57, "EN-correct": 60, "ZH-correct": 58}
    dl = {"EN-confusion": 12, "EN-correct": 20, "ZH-correct": 15}
    assert eligible(ok, dl, CFG)
    assert not eligible({**ok, "ZH-correct": 56}, dl, CFG)
    assert not eligible(ok, {**dl, "EN-confusion": 11}, CFG)
    a = CFG["bootstrap"]["family_alpha"] / CFG["bootstrap"]["family_size"]
    assert [a / 2, 1 - a / 2] == CFG["bootstrap"]["adjusted_quantiles"]


def test_auditor_is_independent_of_primary_code():
    src = (ROOT / "experiments/inference_cf_st_loc0_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(st_loc0_analyze|p2dir_analyze|inference_cf_st_loc0\b|unique|st_loc0_calibrate)", src, re.M) is None
    ana = (ROOT / "experiments/inference_cf_st_loc0_analyze.py").read_text()
    prim = ana[ana.index("def primary("):ana.index("def secondary(")]
    for f in ("POSITIONS", "target_ids", "competitor", "logit_metrics", "HIST_ANALYSIS"):
        assert f not in prim
    sec = ana[ana.index("def secondary("):]
    assert sec.index("committed(SEAL)") < sec.index("committed(PRIMARY_AUDIT)") < sec.index("(ROOT / POSITIONS)")
