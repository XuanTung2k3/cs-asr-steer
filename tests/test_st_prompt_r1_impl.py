"""ST-PROMPT-R1-A implementation tests (tiny random Whisper for engineering; no pretrained science, no references)."""
from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from csasr.inference_cf import prompt_r1 as PR
from csasr.inference_cf.core_p1 import direction

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/inference_cf/st_prompt_r1.json").read_text())
PANEL = json.loads((ROOT / CFG["panel"]).read_text())


def test_runtime_projection_arm_table_and_random_controls():
    rp = PR.runtime_projection(PANEL)
    s = json.dumps(rp)
    assert set(rp) == {"runtime_queries", "utterances"} and len(rp["runtime_queries"]) == 180 and len(rp["utterances"]) == 80
    for f in ("stratum", "EN-confusion", "target_ids", "competitor", "evaluation_membership"):
        assert f not in s
    arms = PR.arm_table(CFG)
    assert len(arms) == 36 and len({a["id"] for a in arms}) == 36
    prim = [a for a in arms if a["family"] == "prompt"]
    assert {(a["layer"], a["eta"], a["sign"]) for a in prim} == {(L, e, s) for L in (3, 8, 16, 24) for e in (.15, .3, .45) for s in (1, -1)}
    rv = PR.random_vectors(CFG)
    assert len(rv) == 12 and all(abs(np.linalg.norm(v.astype(np.float64)) - 1) < 1e-6 for v in rv.values())
    from experiments.inference_cf_st_prompt_r1_analyze import random_of
    assert all(random_of(a) in rv for a in prim)


def test_same_prefix_queries_and_exact_v_prompt_formula():
    from experiments.inference_cf_st_prompt_r1_audit import v_prompt
    by = {u["utterance_id"]: u for u in PANEL["utterances"]}
    for q in PANEL["runtime_queries"]:
        assert q["absolute_query"] == 4 + q["t"] - 1 and q["t"] >= 1
        assert q["utterance_id"] in by
    with np.load(ROOT / "results/inference_cf/st_loc0/run1/calibration/states_L16_CROSS.npz") as z:
        hb, he = z["H_B"], z["H_E"]
    for j in range(0, 180, 7):
        d = direction(torch.from_numpy(he[j]), torch.from_numpy(hb[j]))
        assert np.array_equal(d["d"].numpy(), v_prompt(he[j], hb[j]))
        neg = direction(torch.from_numpy(hb[j]), torch.from_numpy(he[j]))          # sign inversion = swapped roles
        assert np.allclose(neg["d"].numpy(), -d["d"].numpy(), atol=1e-7)
    tiny = direction(torch.ones(8), torch.ones(8) + 1e-6)
    assert tiny["d"] is None and tiny["status"] == "direction_fail:tiny"


def test_barrier_cells_exist_in_sealed_st_loc0():
    from experiments.inference_cf_st_prompt_r1 import BARRIER
    for i in (0, 41, 79):
        for d, aid, _, _ in BARRIER.values():
            r = json.loads((ROOT / f"results/inference_cf/st_loc0/run1/{d}/{i:03d}.json").read_text())
            for pr in r["positions"].values():
                assert aid in pr["cells"] and "s" in pr["cells"][aid]["solver"]
    assert CFG["controls"]["D2_energy"] == json.loads((ROOT / "configs/inference_cf/st_loc0.json").read_text())["energy"]["e_star"]


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


def _pulse(bundle, B, L, new, layer, act):
    from csasr.inference_cf.loc0_sites import Composite, assert_no_any_site_hooks, cross_pulse_hook
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    B.crop(L)
    hook = cross_pulse_hook(bundle, layer, act, 4)
    rec = DecoderPostCrossAttnRecorder(bundle, [layer], keep_last_only=True)
    logits, _, _ = B.step(new, attention=False, hook=Composite(hook, rec))
    assert_no_any_site_hooks(bundle)
    return logits, rec.states[layer][0, -1].float().clone(), (hook.records[-1].to_dict() if hook.records else None)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_relative_dose_pulse_energy_sign_zero_and_restore(tiny_bundle, layer):
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.loc0_sites import cache_fingerprint, pulse_action
    b = copy.deepcopy(tiny_bundle).__class__(**{**copy.deepcopy(tiny_bundle).__dict__})
    b.model.to(torch.bfloat16)
    b.dtype = torch.bfloat16
    B = _branch(b)
    L, new = B.length, [13]
    query = L
    fp0 = cache_fingerprint(B.cache, L)
    base, r0, _ = _pulse(b, B, L, new, layer, pulse_action(query, lambda r: (None, "zero_dose"), 1.0, {}, 1.0, p2r.solve_scale, p2r.scaled_direction))
    torch.manual_seed(5)
    v = torch.randn(16)
    edits = {}
    for eta in CFG["energy"]["eta"]:
        for sign in (1, -1):
            info: dict = {}
            act = PR.relative_action(query, lambda r, s=sign: ((s * v).float(), "ok"), eta, info, .02, 1e-8, p2r.solve_scale, p2r.scaled_direction)
            la, ra, rec = _pulse(b, B, L, new, layer, act)
            assert rec["steered"] and info["target"] == pytest.approx(eta * float(r0.double().norm()))
            ch = PR.cell_checks(eta, info, rec, r0, ra, CFG["energy"])
            assert all(ch["checks"].values()), ch
            assert abs(ch["actual_eta"] / eta - 1) <= .02
            edits[(eta, sign)] = (ra - r0).double()
            assert not torch.equal(la, base)
        c = torch.nn.functional.cosine_similarity(edits[(eta, 1)], edits[(eta, -1)], dim=0)
        assert float(c) < 0                                                      # sign inversion
    assert float(edits[(.45, 1)].norm()) > float(edits[(.15, 1)].norm())         # dose ladder realized
    # unreachable: direction parallel to r => explicit no-edit, exact baseline
    info = {}
    act = PR.relative_action(query, lambda r: (r.detach().float().cpu().clone(), "ok"), .3, info, .02, 1e-8, p2r.solve_scale, p2r.scaled_direction)
    la, ra, rec = _pulse(b, B, L, new, layer, act)
    assert info["no_edit_reason"] == "energy_unreachable" and torch.equal(la, base) and not (rec and rec["steered"])
    info = {}
    la, _, _ = _pulse(b, B, L, new, layer, PR.relative_action(query, lambda r: (None, "tiny_or_nonfinite_direction"), .3, info, .02, 1e-8, p2r.solve_scale, p2r.scaled_direction))
    assert info["no_edit_reason"] == "tiny_or_nonfinite_direction" and torch.equal(la, base)
    B.crop(L)
    assert cache_fingerprint(B.cache, L) == fp0 and B.positions == list(range(L))
    again, r1, _ = _pulse(b, B, L, new, layer, pulse_action(query, lambda r: (None, "zero_dose"), 1.0, {}, 1.0, p2r.solve_scale, p2r.scaled_direction))
    assert torch.equal(again, base) and torch.equal(r1, r0)
    assert all(p.grad is None for p in b.model.parameters())


def test_geometry_cell_and_coverage_gate():
    import experiments.inference_cf_p2r as p2r
    from experiments.inference_cf_st_prompt_r1_geometry import coverage
    r = torch.tensor([1., 0, 0, 0], dtype=torch.bfloat16)
    assert PR.geometry_cell(r, np.array([0, 1, 0, 0], np.float32), .3, CFG["energy"], p2r.solve_scale)["reachable"]
    assert PR.geometry_cell(r, np.array([1, 0, 0, 0], np.float32), .3, CFG["energy"], p2r.solve_scale)["status"] == "energy_unreachable"
    assert PR.geometry_cell(r, None, .3, CFG["energy"], p2r.solve_scale)["status"] == "tiny_or_nonfinite_direction"
    mem = PANEL["evaluation_membership"]
    led = {f"q{j:03d}|A": {"reachable": True} for j in range(180)}
    assert coverage(led, mem, ["A"], CFG["eligibility"])["A"]["pass"]
    conf = [j for j, p in enumerate(mem) if p["stratum"] == "ZH-correct"][:7]
    for j in conf:
        led[f"q{j:03d}|A"]["reachable"] = False
    cv = coverage(led, mem, ["A"], CFG["eligibility"])["A"]
    assert cv["reachable"]["ZH-correct"] == 53 and not cv["pass"]                # 53 < 54 in one stratum fails every arm gate


def _arm(i, **kw):
    a = {"id": f"prompt_L16_eta0.30_{i:02d}", "eta": .3, "eligible": True, "corrections": 3, "correction_dialogues": 3, "macro": .5,
         "lower_none": .01, "random_point": .01, "lower_random": .01, "EN_corruptions": 0, "ZH_corruptions": 0}
    a.update(kw)
    return a


def test_decision_precedence_boundaries_and_selection():
    from experiments.inference_cf_st_prompt_r1_analyze import decide
    assert decide(True, [_arm(0)], CFG)["label"] == "ST_PROMPT_R1_A_CAUSAL_PROMISE"                  # equality boundaries pass
    assert decide(False, [_arm(0)], CFG)["label"] == "ST_PROMPT_R1_A_INVALID"
    assert decide(True, [_arm(0, ZH_corruptions=3)], CFG)["label"] == "ST_PROMPT_R1_A_POWER_WITH_DAMAGE"
    assert decide(True, [_arm(0, EN_corruptions=4, ZH_corruptions=2)], CFG)["label"] == "ST_PROMPT_R1_A_POWER_WITH_DAMAGE"   # 6 > 5
    assert decide(True, [_arm(0, EN_corruptions=3, ZH_corruptions=2)], CFG)["label"] == "ST_PROMPT_R1_A_CAUSAL_PROMISE"     # 5 <= 5
    assert decide(True, [_arm(0, corrections=2)], CFG)["label"] == "ST_PROMPT_R1_A_MARGIN_ONLY"
    assert decide(True, [_arm(0, macro=.49)], CFG)["label"] == "ST_PROMPT_R1_A_MARGIN_ONLY"
    assert decide(True, [_arm(0, lower_random=0.0)], CFG)["label"] == "ST_PROMPT_R1_A_NO_CORRECTION_POWER"
    assert decide(True, [_arm(0, lower_none=0.0, corrections=0)], CFG)["label"] == "ST_PROMPT_R1_A_NO_CORRECTION_POWER"
    assert decide(True, [_arm(0, eligible=False)], CFG)["label"] == "ST_PROMPT_R1_A_NO_CORRECTION_POWER"
    assert decide(True, [_arm(0, random_point=0.0)], CFG)["label"] == "ST_PROMPT_R1_A_MARGIN_ONLY"          # random point must be > 0
    arms = [_arm(0, corrections=4), _arm(1, corrections=5, ZH_corruptions=1), _arm(2, corrections=5, correction_dialogues=4),
            _arm(3, corrections=5, correction_dialogues=4, eta=.15), _arm(4, corrections=5, correction_dialogues=4, lower_none=.2)]
    d = decide(True, arms, CFG)
    assert d["selected"] == [arms[4]["id"], arms[3]["id"]] and len(d["full_pass"]) == 5


def test_bootstrap_family_48_quantiles_and_shared_draws():
    from experiments.inference_cf_p2dir_analyze import boot_stat, draw_weights
    b = CFG["bootstrap"]
    adj = b["alpha"] / b["joint_family_size"]
    assert [adj / 2, 1 - adj / 2] == b["adjusted_quantiles"]
    keys, W = draw_weights([q["dialogue_id"] for q in PANEL["runtime_queries"]], 2000, b["seed"])
    assert len(keys) == 20 and np.array_equal(W, draw_weights([q["dialogue_id"] for q in PANEL["runtime_queries"]], 2000, b["seed"])[1])
    r = boot_stat([(k, 1.0) for k in keys[:17]], keys, W, adj)
    assert r["estimate"] == 1.0 and r["ci"] == [1.0, 1.0]


def test_firewall_static_and_auditor_independence():
    from experiments.inference_cf_st_prompt_r1_audit import RUNTIME_MODULES, hits
    for rel in RUNTIME_MODULES:
        assert hits(rel) == [], rel
    au = (ROOT / "experiments/inference_cf_st_prompt_r1_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(prompt_r1\b|inference_cf_st_prompt_r1\b|st_prompt_r1_analyze|st_prompt_r1_geometry|p2dir_analyze)", au, re.M) is None
    an = (ROOT / "experiments/inference_cf_st_prompt_r1_analyze.py").read_text()
    prim = an[an.index("def primary("):an.index("def kl_edit_none(")]
    assert "POSITIONS" not in prim and "target_ids" not in prim
    sec = an[an.index("def secondary("):]
    assert sec.index("committed(SEAL)") < sec.index("committed(PRIMARY_AUDIT)") < sec.index("(ROOT / POSITIONS)")
    from experiments.inference_cf_st_prompt_r1_analyze import secondary
    if not (ROOT / "results/inference_cf/st_prompt_r1/output_seal_A.json").exists():
        with pytest.raises((PermissionError, FileNotFoundError)):
            secondary("results/inference_cf/st_prompt_r1/runA")


def test_kl_and_independent_metrics_consistency():
    from experiments.inference_cf_st_prompt_r1_analyze import kl_edit_none
    z = np.random.default_rng(0).normal(size=51866).astype(np.float32)
    assert kl_edit_none(z, z, 3, [5, 6], []) == pytest.approx(0.0, abs=1e-12)
    z2 = z.copy()
    z2[100] += 3
    assert kl_edit_none(z2, z, 3, [5, 6], []) > 0
