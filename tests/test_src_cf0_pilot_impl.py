"""SRC-CF0-P implementation tests (CPU; synthetic arrays, sealed metadata and a tiny random Whisper only; no pretrained
science, no lexical references)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from csasr.inference_cf import cf_pilot_contract as C
from csasr.inference_cf import src_cf0_pilot as H

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/inference_cf/src_cf0_pilot.json").read_text())
PANEL = json.loads((ROOT / "docs/inference_cf/SRC_CF0_PANEL.json").read_text())


def _bf16(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def test_config_is_the_frozen_file():
    h = "sha256:" + hashlib.sha256((ROOT / "configs/inference_cf/src_cf0_pilot.json").read_bytes()).hexdigest()
    assert h == "sha256:f1881050b0718559d05a3ddaa0cb2832953e0fc7553be4c360d952d57f51aa87"


@pytest.mark.parametrize("seed", range(6))
def test_direction_matches_frozen_contract_bitwise(seed):
    rng = np.random.default_rng(seed)
    h = _bf16(rng.standard_normal(1280) * 0.4)
    m = _bf16(h + rng.standard_normal(1280) * [1e-3, 0.05, 0.5, 2.0, 0.01, 0.2][seed])
    a, b = H.direction(h, m), C.direction(h, m)
    assert a["status"] == b["status"] == "OK"
    np.testing.assert_array_equal(a["vector"], b["vector"])
    for k in ("raw_norm", "tangent_ratio", "raw_tangent_norm", "direction_norm"):
        assert a[k] == b[k]
    assert a["direction_norm"] < 1                                                   # historical epsilon, not a forced unit vector
    assert a["direction_norm"] == pytest.approx(a["raw_norm"] / (a["raw_norm"] + 1e-6), rel=1e-6)
    assert a["vector_sha256"] == "sha256:" + hashlib.sha256(a["vector"].tobytes()).hexdigest()


def test_direction_degenerate_and_invalid():
    h = _bf16(np.ones(1280))
    assert H.direction(h, h)["status"] == "ABSTAIN_DEGENERATE"
    m = h.copy()
    m[0] += 2 ** -9          # raw delta 2e-3 > 1e-4 -> OK
    assert H.direction(h, m)["status"] == "OK"
    for bad, msg in ((np.full(1280, np.nan), "NONFINITE"), (np.zeros(1280), "CLEAN_STATE_NORM")):
        with pytest.raises(ValueError, match=msg):
            H.direction(bad, h)
    with pytest.raises(ValueError, match="SHAPE"):
        H.direction(h[:16], h[:16])


def test_random_direction_matches_contract_and_is_sign_dose_shared():
    for uid, t, l in (("U", 3, 16), ("ZH-CN_U0011_S0_265", 18, 24)):
        np.testing.assert_array_equal(H.random_direction(uid, t, l), C.random_direction(uid, t, l))
    assert not np.array_equal(H.random_direction("U", 3, 16), H.random_direction("U", 3, 24))


def test_arm_matrix_is_complete_and_keyed():
    a = H.arms()
    assert len(a) == 24 and len({x["id"] for x in a}) == 24
    assert sum(x["family"] == "target" for x in a) == 8 == sum(x["family"] == "off" for x in a) == sum(x["family"] == "random" for x in a)
    assert H.arm_id(16, .15, -1) == "L16_eta0.15_minus" and H.arm_id(24, .30, 1, "random") == "random_L24_eta0.30_plus"
    assert {(x["layer"], x["eta"], x["sign"]) for x in a} == {(l, e, s) for l in CFG["matrix"]["layers"] for e in CFG["matrix"]["eta"] for s in CFG["matrix"]["signs"]}


def test_runtime_projection_allowlist_and_firewall():
    p = H.runtime_projection(PANEL, CFG)
    fw = CFG["firewall"]
    assert all(set(q) == set(fw["runtime_query_keys"]) for q in p["runtime_queries"])
    assert all(set(u) == set(fw["runtime_utterance_keys"]) for u in p["utterances"])
    assert all(set(r) == set(fw["region_keys"]) for r in p["regions"])
    assert len(p["runtime_queries"]) == 180 and len(p["utterances"]) == 80
    assert not H.forbidden_hits(p)
    text = json.dumps(p)
    assert "coverage_review" not in text and "EN-confusion" not in text and "stratum" not in text
    tmax = {}
    for q in p["runtime_queries"]:
        tmax[q["utterance_id"]] = max(tmax.get(q["utterance_id"], 0), q["t"])
    full = {u["utterance_id"]: u["baseline_content_tokens"] for u in PANEL["utterances"]}
    assert all(u["content_prefix_tokens"] == full[u["utterance_id"]][:tmax[u["utterance_id"]]] for u in p["utterances"])
    assert sum(r["target"]["status"] == "OK" for r in p["regions"]) == 73 and sum(r["paired_available"] for r in p["regions"]) == 59
    assert len(H.mask_groups(p)) == 124
    assert H.forbidden_hits({"x": "EN-confusion"}) and H.forbidden_hits({"stratum": 1})


def test_energy_cfg_maps_frozen_values():
    e = H.energy_cfg(CFG)
    assert e["max_relative_norm_error"] == e["max_relative_squared_error"] == e["max_pairwise_relative_squared_error"] == .02
    assert e["consumed_vs_proposed_relative_norm_tolerance"] == .005 and e["solver_matches_hook_abs_scaled_tolerance"] == 1e-6
    assert e["solver_max_evaluations"] == 8 and e["minimum_state_norm"] == 1e-8


def test_reach_ledger_reachable_and_unreachable():
    import experiments.inference_cf_p2r as p2r
    rng = np.random.default_rng(1)
    h = torch.from_numpy(rng.standard_normal(1280).astype(np.float32)).to(torch.bfloat16)
    ec = H.energy_cfg(CFG)
    v = rng.standard_normal(1280).astype(np.float32)
    for s in (1, -1):
        for e in (.15, .30):
            r = H.reach(h, v, e, s, ec, p2r.solve_scale)
            assert r["reachable"] and abs(r["emulated_edit_norm"] / r["target"] - 1) <= .02, r
    par = h.float().numpy()                 # direction parallel to the state: chord cannot rotate -> unreachable
    assert H.reach(h, par, .3, 1, ec, p2r.solve_scale)["status"] == "energy_unreachable"
    assert H.reach(h, None, .3, 1, ec, p2r.solve_scale)["status"] == "no_direction"


def test_paired_energy_guard():
    assert H.paired_energy_ok([1.0, 1.005, 0.999], .02)
    assert not H.paired_energy_ok([1.0, 1.02], .02)
    assert H.paired_energy_ok([], .02)


def test_pair_geometry_and_zero_off_tangent_abstains():
    t = {"vector": np.ones(1280, np.float32), "raw_tangent_norm": 2.0}
    o = {"vector": -np.ones(1280, np.float32), "raw_tangent_norm": 0.0}
    g = H.pair_geometry(t, o)
    assert g["cos"] == pytest.approx(-1) and g["abs_cos"] == pytest.approx(1) and g["raw_tangent_ratio"] is None
    assert g["ratio_status"] == "ABSTAIN_ZERO_OFF_TANGENT"


# ---- tiny random Whisper: DG-02 site, pre-FFN consumption, zero dose, relative pulse checks, restoration --------------

@pytest.mark.parametrize("layer", [0, 2])
def test_engine_site_consumption_zero_pulse_and_restore(tiny_bundle, layer, monkeypatch):
    import experiments.inference_cf_p2r as p2r
    import experiments.inference_cf_src_cf0_pilot as R
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.loc0_sites import cache_fingerprint, pulse_action
    from csasr.inference_cf.prompt_r1 import relative_action
    monkeypatch.setattr(R, "LAYERS", (0, 2))
    b = copy.deepcopy(tiny_bundle).__class__(**{**copy.deepcopy(tiny_bundle).__dict__})
    b.model.to(torch.bfloat16)
    b.dtype = torch.bfloat16
    b.model.generation_config.alignment_heads = [[1, 0], [2, 1]]                 # tiny model: any heads (attention readout only)
    b.model.config._attn_implementation = "eager"                                # production model is eager (attention returned)
    for mod in b.model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "eager"
    c = {"decoder_forwards": 0, "pulse_forwards": 0}
    eng = R.Engine(b, c)
    torch.manual_seed(3)
    with torch.inference_mode():
        enc = b.model.model.encoder(input_features=torch.randn(1, 8, 100).to(torch.bfloat16)).last_hidden_state
    B = p2r.DiagBranch(b, BaseModelOutput(last_hidden_state=enc), [1, 5, 7, 9], "B")
    eng.step(B, [1, 5, 7, 9])
    L, new = B.length, [11]
    query = L
    fp0 = cache_fingerprint(B.cache, L)
    lb, _, sb, ffn = eng.step(B, new)
    assert all(torch.equal(ffn[l], sb[l]) for l in (0, 2))                     # recorder site == tensor the FFN consumes
    la, sa, ffn_z, rec = eng.pulse(B, L, new, layer, pulse_action(query, lambda r: (None, "zero_dose"), 1.0, {}, .02, p2r.solve_scale, p2r.scaled_direction))
    assert torch.equal(la, lb) and all(torch.equal(sa[l], sb[l]) for l in (0, 2)) and not (rec and rec["steered"])
    ec = H.energy_cfg(CFG)
    ec["minimum_state_norm"] = 1e-8
    v = torch.randn(16)
    for eta in (.15, .30):
        info: dict = {}
        act = relative_action(query, lambda r: (v.float(), "ok"), eta, info, .02, 1e-8, p2r.solve_scale, p2r.scaled_direction)
        la, sa, ffn_p, rec = eng.pulse(B, L, new, layer, act)
        assert rec["steered"] and all(torch.equal(ffn_p[l], sa[l]) for l in (0, 2))
        ch = H.pulse_checks(eta, info, rec, sb[layer], sa[layer], ec)
        assert ch["valid"] and set(ch["checks"]) == set(H.FROZEN_CHECKS), ch
        assert set(ch["descriptive"]) == set(H.DESCRIPTIVE_CHECKS)
        if layer == 2:
            assert torch.equal(sa[0], sb[0])                                      # upstream layer untouched
    B.crop(L)
    assert cache_fingerprint(B.cache, L) == fp0
    again, _, s2, _ = eng.step(B, new)
    assert torch.equal(again, lb) and all(torch.equal(s2[l], sb[l]) for l in (0, 2))
    assert all(p.grad is None for p in b.model.parameters())


# ---- evaluator / auditor pure pieces --------------------------------------------------------------------------------

def _ledger(rng, ok=True):
    lay = {f: {"status": "OK", "tangent_ratio": float(rng.uniform(.3, 1))} for f in ("target", "off", "random")}
    lay["reach"] = {}
    for f in ("target", "off", "random"):
        for s in (1, -1):
            for e in (.15, .30):
                tgt = e * 10
                en = tgt * (1 + rng.uniform(-.004, .004))
                lay["reach"][f"{f}|{s}|{e:.2f}"] = {"reachable": True, "solver_status": "ok", "rel_sq_err": abs(en * en / tgt / tgt - 1),
                                                    "emulated_edit_norm": en, "target": tgt}
    if not ok:
        lay["off"]["tangent_ratio"] = .2
    return lay


def test_evaluator_and_auditor_joint_agree():
    import experiments.inference_cf_src_cf0_pilot_audit as AU
    import experiments.inference_cf_src_cf0_pilot_evaluate as EV
    rng = np.random.default_rng(0)
    for k in range(40):
        lay = _ledger(rng, ok=k % 5 != 0)
        if k % 7 == 0:
            lay["reach"]["random|-1|0.30"]["emulated_edit_norm"] *= 1.02
        if k % 9 == 0:
            lay["target"]["status"] = "ABSTAIN_DEGENERATE"
        assert EV.joint(lay, .02) == AU.my_joint(lay, .02)


def test_metrics_tie_rule_and_bootstrap_agree_with_auditor():
    import experiments.inference_cf_src_cf0_pilot_audit as AU
    import experiments.inference_cf_src_cf0_pilot_evaluate as EV
    z = np.zeros(60, np.float32)
    z[[7, 3]] = 5.0
    m = EV.metrics(z, 1, [0], [], [7], 3)
    assert m["top1"] == 3 and not m["in_ref"] and m["rank_ref"] == 2                 # lower token ID on exact ties
    assert AU.processed_top1(z, 1, [0], []) == 3
    rng = np.random.default_rng(1)
    D = [f"D{i % 20:02d}" for i in range(180)]
    keys, W = EV.draw_weights(D, 2000, 240924)
    idx = np.random.default_rng(240924).integers(0, 20, size=(2000, 20))
    vals = [(D[j], float(rng.integers(-1, 2))) for j in range(0, 180, 7)]
    b = EV.boot(vals, keys, W, .0015625, .9984375, 1900)
    est, lo, hi, n = AU.my_boot(vals, keys, idx, .0015625, .9984375, 1900)
    assert b["estimate"] == pytest.approx(est) and b["finite_draws"] == n
    assert b["lower"] == pytest.approx(lo, abs=1e-12) and b["upper"] == pytest.approx(hi, abs=1e-12)


def test_auditor_direction_and_random_independent_agreement():
    import experiments.inference_cf_src_cf0_pilot_audit as AU
    rng = np.random.default_rng(5)
    for k in range(10):
        h = _bf16(rng.standard_normal(1280))
        m = _bf16(h + rng.standard_normal(1280) * 0.1)
        a, b = H.direction(h, m), AU.my_direction(h, m)
        np.testing.assert_array_equal(a["vector"], b["vector"])
        assert AU.close(a["tangent_ratio"], b["tangent_ratio"]) and AU.close(a["raw_tangent_norm"], b["raw_tangent_norm"])
    np.testing.assert_array_equal(AU.my_random("Q", 4, 24), H.random_direction("Q", 4, 24))


def test_static_firewall_and_independence():
    import experiments.inference_cf_src_cf0_pilot_audit as AU
    st = AU.static_firewall()
    assert all(st.values()), st
    rs = (ROOT / "experiments/inference_cf_src_cf0_pilot.py").read_text()
    me = (ROOT / "experiments/inference_cf_src_cf0_pilot_audit.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+\S*(src_cf0_pilot\b|cf_pilot_contract|s1_evidence)", me, re.M)
    code = rs.split('"""', 2)[2]
    assert not re.search(r"^\s*(from|import)\s+\S*inference_cf_src_cf0_pilot_(evaluate|audit)", code, re.M)
    assert code.count("inference_cf_src_cf0_pilot_evaluate") == 1                  # only in the not-executed provenance list


def test_frozen_decision_predicates_via_contract():
    """The evaluator delegates decisions to the Codex-frozen predicates; the auditor's re-implementation must agree."""
    arm = {"qualified_layer": True, "energy_valid": True, "paired_rows": 15, "paired_dialogues": 6, "corrections": 3, "corrected_dialogues": 3,
           "random": {"net_corrections": 2, "positive_dialogues": 2, "macro_advantage": .1, "lodo_net": [1, 1, 2]},
           "off_target": {"net_corrections": 2, "positive_dialogues": 2, "macro_advantage": .1, "lodo_net": [1, 2, 1]},
           "EN_corruptions": 3, "ZH_corruptions": 0, "correct_EOS_promotions": 0, "margin_macro": 0., "margin_adjusted_lower": None}
    assert C.power_pass(arm, CFG["gate_b"]) and C.observed_safety_pass(arm, CFG["gate_b"])
    arms = [copy.deepcopy(arm) for _ in range(8)]
    assert C.terminal(CFG, integrity=True, construction=True, specificity=True, arms=arms) == "SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED"
    arms[3]["ZH_corruptions"] = 3
    assert C.terminal(CFG, integrity=True, construction=True, specificity=True, arms=arms) == "SRC_CF0_PILOT_OBSERVED_DAMAGE"
    arms = [copy.deepcopy(arm) for _ in range(8)]
    for a in arms:
        a["off_target"]["lodo_net"] = [1, 0, 2]
    assert C.terminal(CFG, integrity=True, construction=True, specificity=True, arms=arms) == "SRC_CF0_PILOT_CAUSAL_INSUFFICIENT"
