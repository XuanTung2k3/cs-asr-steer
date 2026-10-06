"""P2-SEL-XA focused tests (contract 595e174): S/C/F exact + degenerate/nonfinite rules, fixed 0.95 source
scaling isolated to the current query (q unchanged, lambda=1 exact identity), XA0 capture identities and
cache isolation, first-order finite-difference sanity on a smooth float32 model, frozen FD validity and
XA0/XA1 precedence, unchanged J, reference-free surface, 180 keys/groups, panel, auditor independence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, PART, encoded, tiny_bundle  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle, _enc  # noqa: E402
from test_inference_cf_p2sel import _refs  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2sel_xa as run  # noqa: E402
import experiments.inference_cf_p2sel_xa_analyze as an  # noqa: E402
import experiments.inference_cf_p2sel_xa_audit as au  # noqa: E402
from csasr.inference_cf import readout as R  # noqa: E402
from csasr.inference_cf.source_compatibility import compatibility  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())


# ---- S / C / F ------------------------------------------------------------------------------------------

def test_compatibility_exact_and_clip_bounds():
    g = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    for u, S in (([0.1, 0.0, 5.0], 0.1), ([0.35, 0.0, 0.0], 0.35), ([0.5, 0.0, 1.0], 0.5), ([1.0, 0.0, 0.0], 1.0),
                 ([3.0, 1.0, 0.0], 5.0), ([-1.0, 0.0, 0.0], -1.0), ([0.0, 0.0, 7.0], 0.0)):
        c = compatibility(g, np.array(u, dtype=np.float32))
        assert c["S"] == pytest.approx(S)
        assert c["F"] == min(1.0, max(0.0, c["S"]))
        gn, un = math.sqrt(5), float(np.linalg.norm(u))
        assert c["C"] == pytest.approx(S / (gn * un + 1e-12))
        assert -1 <= c["C"] <= 1 and (c["S"] > 0) == (c["C"] > 0)
    assert compatibility(np.zeros(3), np.ones(3)) == {"S": 0.0, "C": 0.0, "F": 0.0, "g_norm": 0.0, "u_norm": math.sqrt(3)}
    assert compatibility(np.ones(3), np.zeros(3))["F"] == 0.0
    with pytest.raises(ValueError):
        compatibility(np.array([np.nan, 0, 0]), np.ones(3))
    with pytest.raises(ValueError):
        compatibility(np.ones(3), np.ones(4))
    assert CFG["XA0"]["F_src"] == "min(1,max(0,S_src)); sole bounded parameter-free factor"


# ---- source scaling ----------------------------------------------------------------------------------------

def _prefix(bundle, enc, steps=3):
    br = cached.Branch(bundle, enc, CB, "B")
    new, toks = list(CB), [10, 47, 12, 50]
    for t in range(steps):
        br.step(new, capture_layer=16)
        new = [toks[t]]
    return br, new


def _scaled_forward(bundle, br, enc, new, lam):
    scratch, e = R.clone_scratch(br.cache, enc)
    sc = run.SourceScale(bundle, lam=lam)
    with torch.no_grad(), sc:
        out = bundle.model(encoder_outputs=e, decoder_input_ids=torch.tensor([new]), past_key_values=scratch, use_cache=True,
                           cache_position=torch.arange(br.length, br.length + len(new)))
    return out.logits[0, -1].float(), sc


def test_lambda_one_identity_and_query_isolation():
    bundle = _bf16_bundle()
    enc = _enc()
    br, new = _prefix(bundle, enc)
    fp = R.cache_fingerprint(br.cache)
    l1, sc1 = _scaled_forward(bundle, br, enc, new, 1.0)
    l95, sc = _scaled_forward(bundle, br, enc, new, 0.95)
    assert R.cache_fingerprint(br.cache) == fp
    from csasr.lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks
    rec = DecoderPostCrossAttnRecorder(bundle, [16], keep_last_only=True)
    lb, _, hb = br.step(new, capture_layer=16, hook=rec)
    assert_no_site_hooks(bundle)
    assert torch.equal(l1, lb)                                   # lambda=1: exact identity
    q, u = rec.q_states[16][0, -1], rec.u_source_states[16][0, -1]
    assert torch.equal(sc.q.float(), q)                          # q unchanged
    assert torch.equal(sc.u.float(), (0.95 * u.to(torch.bfloat16)).to(torch.bfloat16).float())
    assert torch.equal(sc.r, sc.q + sc.u) and sc.calls == 1
    assert not torch.equal(l95, lb)
    # multi-token step: only the last position's u is scaled
    br2, _ = _prefix(bundle, enc, steps=0)
    scr, e2 = R.clone_scratch(br2.cache, enc) if br2.cache is not None else (None, enc)
    seen = {}
    sc2 = run.SourceScale(bundle)
    orig = sc2._post

    def spy(m, i, o):
        from csasr.lss.sites import _first
        seen["in"] = _first(o).clone()
        res = orig(m, i, o)
        seen["out"] = _first(res).clone()
        return res
    sc2._post = spy
    with torch.no_grad(), sc2:
        bundle.model(encoder_outputs=enc, decoder_input_ids=torch.tensor([CB]), use_cache=True, cache_position=torch.arange(4))
    assert torch.equal(seen["in"][:, :-1], seen["out"][:, :-1]) and not torch.equal(seen["in"][:, -1], seen["out"][:, -1])


def test_xa0_utterance_identities(monkeypatch):
    bundle = _bf16_bundle()
    ctx, toks, vecs, none = _refs(bundle, monkeypatch)
    with torch.inference_mode():
        recs, V, L = run.xa0_utterance(bundle, encoded=_enc(), tokens=toks, ts=[1, 2], cB=CB, ref_vecs=vecs, ref_none=none)
    for t in ("1", "2"):
        r = recs[t]
        assert r["native_sum_bitwise_r"] and r["r_bitwise_capture"] and r["r_bitwise_p2dir_hb"] and r["logits_bitwise_p2dir_none"]
        assert r["q_unchanged_in_scratch"] and r["cache_object_and_length"] and r["scale_calls"] == 1
        q, u, r1 = (V[f"t{t}_{k}"].astype(np.float64) for k in ("q", "u", "r1"))
        assert np.max(np.abs(q + u - r1)) / np.max(np.abs(r1)) <= an.RECON_TOL
    assert all(p.grad is None for p in bundle.model.parameters())


def test_first_order_sanity_on_smooth_float32_model():
    """Synthetic: observed J(0.95)-J(1) ~ dot(g_J, r0.95-r1) with g_J from the unchanged readout provider."""
    bundle = tiny_bundle(seed=4)
    enc = encoded()
    agree, n = 0, 0
    for steps in (1, 2, 3):
        br, new = _prefix(bundle, enc, steps)
        rd = R.readout_direction(bundle, layer=16, cache=br.cache, encoded=enc, new_tokens=new, start=br.length, step=steps,
                                 suppress=[], begin=[], partition=PART)
        l95, sc = _scaled_forward(bundle, br, enc, new, 0.95)
        from csasr.lss.sites import DecoderPostCrossAttnRecorder
        rec = DecoderPostCrossAttnRecorder(bundle, [16], keep_last_only=True)
        lb, _, hb = br.step(new, capture_layer=16, hook=rec)
        obs = float(R.objective(l95, steps, [], [], PART)["J"] - R.objective(lb, steps, [], [], PART)["J"])
        pred = float(rd["gradient"].double() @ (sc.r.double() - hb.double()))
        nom = -0.05 * compatibility(rd["gradient"].numpy(), rec.u_source_states[16][0, -1].numpy())["S"]
        assert pred == pytest.approx(nom, rel=1e-4, abs=1e-9)          # float32: realized == nominal
        assert obs == pytest.approx(pred, rel=0.2, abs=1e-6)
        n += 1
    assert n == 3


# ---- frozen decisions ----------------------------------------------------------------------------------------

def _fd_rows(n=40, dlg=12, obs_scale=1.0, flip=0):
    rows = []
    for i in range(n):
        pr = 0.05 + 0.01 * i
        rows.append({"dialogue_id": f"D{i % dlg:02d}", "P_nominal": pr, "P_realized": pr,
                     "observed": (-pr if i < flip else pr * obs_scale)})
    return rows


def test_fd_validity_boundaries():
    assert an.fd_validity(_fd_rows(), CFG)["ok"]
    assert not an.fd_validity(_fd_rows(n=29), CFG)["ok"]                 # < 30 material rows
    assert not an.fd_validity(_fd_rows(dlg=9), CFG)["ok"]                # < 10 dialogues
    assert not an.fd_validity(_fd_rows(flip=5), CFG)["ok"]               # sign agreement 35/40 < 0.9
    assert an.fd_validity(_fd_rows(flip=4), CFG)["ok"]                   # 36/40 = 0.9
    assert not an.fd_validity(_fd_rows(obs_scale=0.4), CFG)["ok"]        # median ratio < 0.5
    assert not an.fd_validity(_fd_rows(obs_scale=1.6), CFG)["ok"]        # ratio > 1.5 (and RMS)
    rows = _fd_rows()
    rows[0]["observed"] = 0.0
    assert an.fd_validity(rows, CFG)["sign_obs_realized"] == pytest.approx(39 / 40)   # zero counts as disagreement
    rows = _fd_rows() + [{"dialogue_id": "D00", "P_nominal": 0.019, "P_realized": 5.0, "observed": -5.0}]
    assert an.fd_validity(rows, CFG)["n"] == 40                          # nominal below 0.02 excluded


def _groups(**mods):
    rows = []
    for g, n in an.EXPECTED.items():
        for i in range(n):
            rows.append({"dialogue_id": f"D{(i * 7 + len(rows)) % 20:02d}", "group": g, "F": 0.9 if g in ("EN_TP",) else 0.0})
    for g, f in mods.items():
        for i, r in enumerate([r for r in rows if r["group"] == g]):
            f(i, r)
    return {g: [r for r in rows if r["group"] == g] for g in an.GROUPS}


def _dec(G, valid=True):
    keys, W = an.draw_weights([f"D{i:02d}" for i in range(20)], 10000, 240924)
    return an.decide_xa0(valid, G, keys, W)


def test_xa0_precedence():
    assert _dec(_groups())["label"] == "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    fp_bad = lambda i, r: r.update(F=0.5) if i < 2 else None
    assert _dec(_groups(ZH_FP=fp_bad))["label"] == "P2_SEL_XA_SOURCE_COMPATIBILITY_NOT_DISCRIMINATIVE"
    assert _dec(_groups(ZH_FP=fp_bad, EN_FN=lambda i, r: r.update(F=0.7) if i < 9 else None))["label"] == \
        "P2_SEL_XA_SOURCE_COMPATIBILITY_RECALL_ONLY"
    assert _dec(_groups(EN_FN=lambda i, r: r.update(F=0.8)))["label"] == "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    assert _dec(_groups(EN_TP=lambda i, r: r.update(F=0.69) if i < 9 else None))["label"] != "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    assert _dec(_groups(ZH_FP=lambda i, r: r.update(F=0.10)))["label"] == "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    assert _dec(_groups(), valid=False)["label"] == "P2_SEL_XA_INVALID"
    with pytest.raises(ValueError):
        an.selection_artifact({"label": "P2_SEL_XA_SOURCE_COMPATIBILITY_RECALL_ONLY"}, "x", "y")


def iv(e, l, h):
    return {"estimate": e, "ci": [l, h], "valid_draws": 10000}


def test_xa1_labels():
    th = CFG["XA1"]["thresholds"]
    f = {"conf": iv(3.0, 1.0, 4.0), "en": iv(0, 0, 0), "zh": iv(-0.05, -0.2, 0), "paired_zh": iv(0.15, 0.01, 0.3),
         "corr_en": iv(0, 0, 0), "corr_zh": iv(0, 0, 0.1)}
    R_ = {"benefit_retention": 0.8, "harm_ratio": 0.3}
    O = {"en": 0.0, "zh": 0.0}
    E = {"mean_new_minus_old": 0.0, "new_corruptions": 0}
    L = lambda **k: an.decide_xa1(k.get("v", True), k.get("f", f), k.get("r", R_), k.get("o", O), k.get("e", E), th)
    assert L() == "P2_SEL_XA_GATE_SUPPORTED"
    assert L(e={"mean_new_minus_old": -1e-5, "new_corruptions": 0}) == "P2_SEL_XA_GATE_STILL_UNSAFE"
    assert L(o={"en": 0, "zh": 0.051}) == "P2_SEL_XA_GATE_STILL_UNSAFE"
    assert L(r={**R_, "benefit_retention": 0.699}) == "P2_SEL_XA_GATE_TOO_CONSERVATIVE"
    assert L(r={**R_, "harm_ratio": 0.51}) == "P2_SEL_XA_NO_MATERIAL_GAIN"
    assert L(f={**f, "paired_zh": iv(0.2, 0.0, 0.3)}) == "P2_SEL_XA_NO_MATERIAL_GAIN"
    assert L(v=False) == "P2_SEL_XA_INVALID"


# ---- lineage / surface ------------------------------------------------------------------------------------

def test_J_unchanged_and_auditor_agrees():
    part = {"embedded_ids": list(range(44, 54)), "matrix_ids": list(range(8, 44))}
    rng = np.random.default_rng(0)
    for t in (0, 3):
        z = rng.normal(size=64).astype(np.float32) * 3
        a = float(R.objective(torch.from_numpy(z), t, [0, 3], [5], part)["J"])
        assert a == pytest.approx(au.J_np(z, t, [0, 3], [5], part), abs=1e-5)
    assert an.J_of(z, 3, [0, 3], [5], part) == pytest.approx(float(R.objective(torch.from_numpy(z), 3, [0, 3], [5], part)["J"]))


def test_frozen_sources_population_panel_surface():
    for rel, h in CFG["source_sha256"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == h, rel
    rows = json.loads(Path("results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]
    assert {g: sum(r["group"] == g for r in rows) for g in an.GROUPS} == an.EXPECTED == CFG["population"]["historical_groups"]
    assert au.surface_clean()["ok"]
    src = Path("experiments/inference_cf_p2sel_xa_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_analyze|source_compatibility)", src, re.M) is None
    assert an.RECON_TOL == 0.03125
