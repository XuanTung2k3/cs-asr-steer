"""P2-SEL focused tests (spec sections 2-5): D2/gate unchanged, exact run3 gate joins, zero gate =
exact no-edit, gated dose semantics, pooled BROAD energy, no parameter grads / autograd, reference-free
runner, deterministic mini-panel, frozen S1/S2 label precedence, auditor agreement and independence."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, encoded  # noqa: E402
from test_inference_cf_p2dir_protocol import _baseline_tokens, _bf16_bundle, _enc  # noqa: E402

import experiments.inference_cf_p2dir as p2dir  # noqa: E402
import experiments.inference_cf_p2sel as run  # noqa: E402
import experiments.inference_cf_p2sel_analyze as an  # noqa: E402
import experiments.inference_cf_p2sel_audit as au  # noqa: E402
from csasr.inference_cf.core_p1 import direction as old_direction  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())
TH = CFG["S1"]["thresholds"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- frozen components ---------------------------------------------------------------------------------

def test_d2_gate_and_actuator_sources_unchanged():
    for rel in ("src/csasr/inference_cf/readout.py", "src/csasr/inference_cf/directions.py",
                "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py", "experiments/inference_cf_cached.py",
                "experiments/inference_cf_p2r.py", "src/csasr/inference_cf/broad.py"):
        assert sha(rel) == CFG["source_sha256"][rel].replace("sha256:", "")
    assert run.ALPHA == CFG["frozen_intervention"]["alpha"] == 2 and run.LAYER == 16


def test_s1_reuses_sealed_p2dir_d2_vectors():
    from csasr.inference_cf.unique import array_hash
    con = json.loads(Path(run.CONSTRUCTION).read_text())
    rows, vecs, nones = run.load_p2dir_rows(con)          # verifies every sealed file hash
    for p in con["positions"][::17]:
        rec = rows[p["utterance_id"]]["positions"][str(p["t"])]["directions"]["D2"]
        assert rec["status"] == "ok" and array_hash(vecs[p["utterance_id"]][f"t{p['t']}_D2"]) == rec["sha256"]
        assert f"t{p['t']}_none" in nones[p["utterance_id"]]


def test_gate_join_exact_180_and_product():
    con = json.loads(Path(run.CONSTRUCTION).read_text())
    rows, _, _ = run.load_p2dir_rows(con)
    pop = json.loads(Path("results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(pop["utterances"])}
    p2r_rows = {u: json.loads(Path(f"{run.P2R_RUN}/rows/{pidx[u]:03d}.json").read_text()) for u in con["utterances"]}
    g = run.join_gates(con, p2r_rows, rows)
    assert len(g) == 180 and all(abs(v["E"] * v["R_B"] - v["g"]) <= 1e-12 for v in g.values())
    ind = au.gate_join(con)
    assert not ind["bad"] and all(ind["gates"][k]["g"] == v["g"] for k, v in g.items())
    broken = {u: {**r, "current": {"steps": [s for s in r["current"]["steps"] if s["t"] != 1]}} for u, r in p2r_rows.items()}
    with pytest.raises(ValueError):
        run.join_gates(con, broken, rows)


# ---- pulse semantics on a bf16 tiny model ------------------------------------------------------------

def _refs(bundle, monkeypatch, ts=(1, 2)):
    toks = _baseline_tokens(bundle, monkeypatch)
    ctx = p2dir.Ctx(bundle, PART, 4, 0.0, cB=CB, cE=CE)
    with torch.inference_mode():
        _, st = p2dir.extract_utterance(ctx, encoded=_enc(), tokens=toks, ts=list(ts))
        B = p2dir.p2r.DiagBranch(bundle, _enc(), CB, "B")
        new, none = list(CB), {}
        for t in range(max(ts) + 1):
            lb, _, _ = B.step(new, capture_layer=16, attention=True)
            if t in ts:
                none[f"t{t}_none"] = p2dir.pack_bf16(lb)
            new = [toks[t]]
    rng = np.random.default_rng(3)
    vecs = {}
    for t in ts:
        hb, he = st[str(t)]
        vecs[f"t{t}_hb"], vecs[f"t{t}_he"] = hb.numpy(), he.numpy()
        vecs[f"t{t}_D0"] = old_direction(he, hb)["d"].numpy()
        v = rng.standard_normal(16)
        vecs[f"t{t}_D2"] = (v / np.linalg.norm(v)).astype(np.float32)
    return ctx, toks, vecs, none


def _gate(g, q):
    return {"query": q, "E": 1.0, "R_B": g, "g": g, "fb": None, "window": None}


def test_zero_gate_is_exact_no_edit_and_positive_gate_uses_p2_dose(monkeypatch):
    bundle = _bf16_bundle()
    ctx, toks, vecs, none = _refs(bundle, monkeypatch)
    gates = {1: _gate(0.0, 4), 2: _gate(0.6, 5)}
    with torch.inference_mode():
        recs, post, logits = run.s1_pass_a(ctx, encoded=_enc(), tokens=toks, ts=[1, 2], gates=gates, ref_vecs=vecs, ref_none=none)
    r1, r2 = recs["1"], recs["2"]
    for k in ("query", "hb_bitwise_p2dir", "he_bitwise_p2dir", "none_logits_bitwise_p2dir", "d0_bitwise_p2dir"):
        assert r1["identity"][k] and r2["identity"][k]
    for arm in ("C1", "C2"):
        assert not r1["arms"][arm]["applied"] and r1["arms"][arm]["edit_norm"] == 0.0
        assert r1["arms"][arm]["zero_gain_bitwise_none"] and np.array_equal(logits[f"t1_{arm}"], none["t1_none"])
        a = r2["arms"][arm]
        assert a["applied"] and a["dose"] == pytest.approx(1.2) and a["edit_norm"] > 0
        d = vecs["t2_D0"] if arm == "C1" else vecs["t2_D2"]
        assert a["edit_norm"] == pytest.approx(au.ref_edit_norm(vecs["t2_hb"], d, 1.2), rel=2e-2)
    assert r1["restore_bitwise"] and r2["restore_bitwise"]
    assert all(p.grad is None for p in bundle.model.parameters())


def test_broad_pooled_target_and_energy(monkeypatch):
    assert run.broad_target([3.0] + [0.0] * 179) == pytest.approx(math.sqrt(9 / 180))
    assert run.broad_target([0.0] * 180) == 0.0
    with pytest.raises(ValueError):
        run.broad_target([1.0] * 179)
    bundle = _bf16_bundle(seed=2)
    ctx, toks, vecs, none = _refs(bundle, monkeypatch)
    with torch.inference_mode():
        recs, post, logits = run.s1_pass_b(ctx, encoded=_enc(), tokens=toks, ts=[1, 2], target=0.03, ref_vecs=vecs)
        zrecs, _, zlog = run.s1_pass_b(ctx, encoded=_enc(), tokens=toks, ts=[1, 2], target=0.0, ref_vecs=vecs)
    for t in ("1", "2"):
        a = recs[t]["arm"]
        assert a["steered"] and a["solver"]["status"] == "ok"
        assert abs(a["edit_norm"] ** 2 / 0.03 ** 2 - 1) <= CFG["S1"]["broad_rule"]["aggregate_relative_energy_tolerance"]
        assert recs[t]["restore_bitwise"] and recs[t]["hb_bitwise_p2dir"]
        z = zrecs[t]["arm"]
        assert not z["steered"] and z["edit_norm"] == 0.0 and z["zero_bitwise_none"]
        assert np.array_equal(zlog[f"t{t}_C3"], none[f"t{t}_none"])


def test_runner_reference_free_and_no_readout_or_autograd():
    assert au.runner_reference_free()
    src = Path("experiments/inference_cf_p2sel.py").read_text()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in ("s1_pass_a", "s1_pass_b"):
            seg = ast.get_source_segment(src, n)
            assert "readout" not in seg and "autograd" not in seg and "ReadoutDirection" not in seg


# ---- mini panel ----------------------------------------------------------------------------------------

def test_mini_panel_deterministic_and_frozen():
    panel = json.loads(Path(CFG["S2"]["panel"]).read_text())
    assert sha(CFG["S2"]["panel"]) == CFG["S2"]["panel_sha256"]
    rows = au.panel_rederive(panel)
    assert rows == [(r["dialogue_id"], r["utterance_id"]) for r in panel["rows"]]
    assert len(rows) == 100 and len({d for d, _ in rows}) == 20
    assert all(sum(d == k for d, _ in rows) == 5 for k in {d for d, _ in rows})


# ---- frozen decision rules -------------------------------------------------------------------------------

def iv(est, l, h, n=10000):
    return {"estimate": est, "ci": [l, h], "valid_draws": n}


def fam(**over):
    base = {"C2_conf": (1.0, 0.2, 2.0), "C3_conf": (1.0, 0.2, 2.0), "C2_en": (0, -0.1, 0.1), "C2_zh": (-0.1, -0.2, 0.0),
            "C3_en": (0, -0.1, 0.1), "C3_zh": (-1.0, -1.5, -0.5), "C2_corr_en": (0, 0, 0.0), "C2_corr_zh": (0, 0, 0.05),
            "C3_corr_en": (0, 0, 0), "C3_corr_zh": (0.1, 0.0, 0.2), "diff_conf": (0, -0.2, 0.2), "diff_zh": (0.9, 0.3, 1.5)}
    base.update(over)
    return {k: iv(*v) for k, v in base.items()}


EQ_NO = {"conf": iv(0, -0.1, 0.1), "zh": iv(0.9, 0.5, 1.2)}
EQ_YES = {"conf": iv(0, -0.25, 0.25), "zh": iv(0, -0.1, 0.1)}


def obs(c2z=0.05, c3z=0.1):
    return {"C2": {"EN-correct": 0.0, "ZH-correct": c2z}, "C3": {"EN-correct": 0.0, "ZH-correct": c3z}}


def L(f, eq=EQ_NO, o=None, valid=True):
    return an.decide_s1(valid, f, eq, o or obs(), TH)["label"]


def test_s1_precedence_and_equalities():
    assert L(fam()) == "P2_SEL_GATE_RESCUES_D2"
    assert L(fam(diff_zh=(0.9, 0.0999, 1))) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(diff_conf=(0, -0.2501, 1))) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(diff_conf=(0, -0.25, 1), diff_zh=(0.2, 0.1, 1))) == "P2_SEL_GATE_RESCUES_D2"
    assert L(fam(C2_conf=(0.4999, 0.1, 1))) == "P2_SEL_GATE_TOO_CONSERVATIVE"
    assert L(fam(C2_conf=(0.9, 0.0, 1))) == "P2_SEL_GATE_TOO_CONSERVATIVE"
    assert L(fam(C2_conf=(0.1, -0.1, 1), C2_zh=(-1, -1.5, -0.5))) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(C2_zh=(-0.3, -0.2501, 0))) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(C2_corr_zh=(0.02, 0, 0.0501))) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(), o=obs(c2z=0.0501)) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    # equivalence needs C3 standalone pass too
    ok3 = dict(C3_zh=(0, -0.2, 0.1), C3_corr_zh=(0, 0, 0.05))
    assert L(fam(**ok3), eq=EQ_YES, o=obs(c3z=0.0)) == "P2_SEL_GATE_ADDS_LITTLE_VS_BROAD"
    not_eq = {"conf": iv(0, -0.2501, 0.2), "zh": iv(0, -0.1, 0.1)}       # not equivalent -> falls to selective rule
    assert L(fam(**ok3), eq=not_eq, o=obs(c3z=0.0)) == "P2_SEL_GATE_RESCUES_D2"
    assert L(fam(**ok3, diff_zh=(0, -0.1, 0.1)), eq=not_eq, o=obs(c3z=0.0)) == "P2_SEL_GATE_INSUFFICIENTLY_SELECTIVE"
    assert L(fam(), eq=EQ_YES) != "P2_SEL_GATE_ADDS_LITTLE_VS_BROAD"          # C3 fails safety
    assert L(fam(), valid=False) == "P2_SEL_INVALID"


def test_auditor_label_agrees_with_analysis():
    cases = [(fam(), EQ_NO, obs()), (fam(C2_conf=(0.3, 0.1, 1)), EQ_NO, obs()),
             (fam(diff_zh=(0.1, 0.05, 0.2)), EQ_NO, obs()),
             (fam(C3_zh=(0, -0.2, 0.1), C3_corr_zh=(0, 0, 0.05)), EQ_YES, obs(c3z=0.0))]
    for f, eq, o in cases:
        tup = {k: (v["estimate"], v["ci"], v["valid_draws"]) for k, v in f.items()}
        eqt = {k: (v["estimate"], v["ci"], v["valid_draws"]) for k, v in eq.items()}
        ob = {a: {"en": o[a]["EN-correct"], "zh": o[a]["ZH-correct"]} for a in ("C2", "C3")}
        assert au.s1_label(True, tup, eqt, ob, TH) == an.decide_s1(True, f, eq, o, TH)["label"]


def test_s2_rule():
    th = CFG["S2"]["thresholds"]
    ok = {"MER": 0.005, "ZH_CER": 0.0, "EN_WER": 0.01, "EN_retention_loss": 0.01, "ZH_retention_loss": 0.0, "outside_harm": 0.005}
    g = iv(0.005, 0.0001, 0.01)
    assert an.decide_s2(True, ok, g, g, th) == "P2_SEL_MINI_PROMISING"
    assert an.decide_s2(True, {**ok, "MER": 0.0051}, g, g, th) == "P2_SEL_MINI_DAMAGE_UNRESOLVED"
    assert an.decide_s2(True, ok, g, iv(0.01, 0.0, 0.02), th) == "P2_SEL_MINI_INCONCLUSIVE"
    assert an.decide_s2(True, ok, iv(0.0049, 0.001, 0.01), g, th) == "P2_SEL_MINI_INCONCLUSIVE"
    assert an.decide_s2(False, ok, g, g, th) == "P2_SEL_MINI_INVALID"


def test_auditor_independent_of_analysis():
    src = Path("experiments/inference_cf_p2sel_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2sel|p2dir)_analyze", src, re.M) is None


def test_auditor_hook_emulation_matches_real_hook_at_tiny_and_normal_gates(monkeypatch):
    """Audit attempt-1 compared bf16 hook energy to a float64 ideal; at tiny g the bf16 rounding floor
    dominates. The exact bf16 emulation must reproduce the real hook at tiny and normal gates."""
    bundle = _bf16_bundle()
    ctx, toks, vecs, none = _refs(bundle, monkeypatch)
    for g in (1e-6, 0.7):
        gates = {1: _gate(g, 4), 2: _gate(g, 5)}
        with torch.inference_mode():
            recs, _, _ = run.s1_pass_a(ctx, encoded=_enc(), tokens=toks, ts=[1, 2], gates=gates, ref_vecs=vecs, ref_none=none)
        for t in (1, 2):
            for arm, key in (("C1", "D0"), ("C2", "D2")):
                hook = recs[str(t)]["arms"][arm]["edit_norm"]
                emu = au.hook_edit_emulation(vecs[f"t{t}_hb"], vecs[f"t{t}_{key}"], g)
                assert hook == pytest.approx(emu, rel=1e-5, abs=0)
