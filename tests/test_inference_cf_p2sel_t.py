"""P2-SEL-T focused tests (contract fdace84): frozen L/C/R bounds incl. short/clipped W, heard containment,
fractional-frame mass integration and head normalization, CENTER == P2-SEL-E short_bounds reuse on all
exposed keys, exact L>C>R tie rule, E_tok/E_ctx/C exact, runtime feature surface reference-free and
contrast-free, T0 predicates/precedence (exactly one repair, no contrast fallback), T1 labels incl. EN
no-regression, auditor independence and agreement, 180 keys/groups, frozen sources, mini-panel identity."""
from __future__ import annotations

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
from test_inference_cf_p1 import CB, FRAMES  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle, _enc  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2sel_e as pe  # noqa: E402
import experiments.inference_cf_p2sel_t as run  # noqa: E402
import experiments.inference_cf_p2sel_t_analyze as an  # noqa: E402
import experiments.inference_cf_p2sel_t_audit as au  # noqa: E402
from csasr.inference_cf.core_r2 import local_support, max_attention_window  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())


# ---- bounds / masses --------------------------------------------------------------------------------

def test_bounds_exact_full_short_and_odd():
    assert run.candidate_bounds(16000, 32000) == {"L": (16000, 24000), "C": (20000, 28000), "R": (24000, 32000)}
    b = run.candidate_bounds(0, 6000)                      # W shorter than 0.5 s: all equal W
    assert b["L"] == b["C"] == b["R"] == (0, 6000)
    b = run.candidate_bounds(0, 12345)                    # 0.5 <= |W| < 1 s
    assert b["L"] == (0, 8000) and b["R"] == (4345, 12345) and b["C"] == ((12345 - 8000) // 2, (12345 - 8000) // 2 + 8000)
    for s0, s1 in ((31, 16031), (0, 7999), (100, 8100), (3, 16004)):
        bb = run.candidate_bounds(s0, s1)
        assert bb == {k: tuple(v) for k, v in au.bounds_ind(s0, s1).items()}
        assert all(s0 <= a < c <= s1 and c - a == min(8000, s1 - s0) for a, c in bb.values())


def test_center_equals_e0_short_bounds_formula():
    # W = [s0, s0+min(16000, heard)) inside [0, heard): frozen CENTER == P2-SEL-E short_bounds
    rnd = random.Random(0)
    for _ in range(2000):
        heard = rnd.randint(1, 480000)
        size = min(16000, heard)
        s0 = rnd.randint(0, heard - size)
        s1 = s0 + size
        assert run.candidate_bounds(s0, s1)["C"] == pe.short_bounds(s0, s1, heard)


def test_mass_fractional_frames_and_normalization():
    heads = np.array([[1.0, 3.0, 0.0, 4.0], [3.0, 1.0, 2.0, 0.0]])
    m = run.frame_mean(heads, 1000)                        # valid = ceil(1000/320) = 4 frames
    assert np.allclose(m, np.array([2.0, 2.0, 1.0, 2.0]) / 7.0)
    # last frame is heard only on [960,1000): 40 samples
    assert run.crop_mass(m, 0, 1000, 1000) == pytest.approx(1.0)
    assert run.crop_mass(m, 0, 320, 1000) == pytest.approx(2 / 7)
    assert run.crop_mass(m, 160, 480, 1000) == pytest.approx(0.5 * 2 / 7 + 0.5 * 2 / 7)
    assert run.crop_mass(m, 980, 1000, 1000) == pytest.approx(0.5 * 2 / 7)
    assert run.crop_mass(m, 0, 960, 1000) == pytest.approx(5 / 7)
    assert run.crop_mass(m, 160, 480, 1000) == pytest.approx(au.mass_ind(m, 160, 480, 1000))
    with pytest.raises(ValueError):
        run.frame_mean(np.zeros((2, 4)), 1000)
    with pytest.raises(ValueError):
        run.frame_mean(-heads, 1000)


def test_mass_matches_core_r2_window_mass():
    rng = np.random.default_rng(1)
    heads = rng.random((4, 1500))
    for heard in (480000, 100000, 16001):
        w = max_attention_window(heads, heard)
        m = run.frame_mean(heads, heard)
        assert float(m[w.start_frame:w.end_frame].sum()) == pytest.approx(w.attention_mass, abs=1e-12)
        if w.start_sample == w.start_frame * 320:                     # frame-aligned W
            assert run.crop_mass(m, w.start_sample, w.end_sample, w.heard_samples) == pytest.approx(w.attention_mass, abs=1e-12)


def test_tie_rule_and_features_exact():
    E = {"L": 0.1, "C": 0.7, "R": 0.4}
    assert run.select({"L": 0.5, "C": 0.5, "R": 0.5}) == "L"
    assert run.select({"L": 0.4, "C": 0.5, "R": 0.5}) == "C"
    assert run.select({"L": 0.4, "C": 0.4, "R": 0.5}) == "R"
    f = run.token_features(E, {"L": 0.2, "C": 0.6, "R": 0.3})
    assert f["j"] == "C" and f["E_tok"] == 0.7 and f["E_ctx"] == (0.1 + 0.4) / 2 and f["C_tokctx"] == 0.7 - 0.25
    assert f["a_score"] == pytest.approx(0.6 / 1.1)
    # selection independent of E values
    assert run.token_features({"L": 9, "C": 0, "R": 0}, {"L": 0.2, "C": 0.6, "R": 0.3})["j"] == "C"


def test_runtime_surface_reference_and_contrast_free():
    assert au.runtime_surface_clean()["ok"]
    src = Path("experiments/inference_cf_p2sel_t.py").read_text()
    # contrast is never a gate: selection artifact gates use only E_tok and R_B
    asrc = Path("experiments/inference_cf_p2sel_t_analyze.py").read_text()
    sel = asrc[asrc.index("def selection_artifact"):asrc.index("# ---- T1")]
    assert '"g_new": r["E_tok"] * r["R_B"]' in sel and "C_tokctx" not in sel.split("gates = ")[1]


def test_t0_utterance_on_tiny_model(monkeypatch):
    bundle = _bf16_bundle()
    calls = []

    def fake_lid(b, w, ids):
        calls.append(len(w))
        return {i: 1.0 / len(ids) for i in ids}
    monkeypatch.setattr(cached, "native_lid", fake_lid)
    monkeypatch.setattr(run, "EN_ID", 4)
    monkeypatch.setattr(run, "ZH_ID", 5)
    wav = np.ones(FRAMES * 320, dtype=np.float32)          # 32000 samples
    e0 = {}
    with torch.inference_mode():
        # E0-like center rows from the same frozen window rule
        B = __import__("experiments.inference_cf_p2r", fromlist=["DiagBranch"]).DiagBranch(bundle, _enc(), CB, "B")
        new, toks = list(CB), [10, 47, 12]
        for t in range(3):
            _, heads, _ = B.step(new, attention=True)
            if t in (1, 2):
                w = max_attention_window(heads.numpy(), len(wav))
                e0[str(t)] = {"short": {"bounds": list(pe.short_bounds(w.start_sample, w.end_sample, w.heard_samples)),
                                        "pi_E": 0.3, "pi_M": 0.2}}
            new = [toks[t]]
        out, extra = run.t0_utterance(bundle, encoded=_enc(), waveform=wav, tokens=toks, ts=[1, 2], language_ids=tuple(range(100)),
                                      cB=CB, e0_rows=e0, null={"pi_E": 0.3, "pi_M": 0.02})
    from csasr.lss.sites import assert_no_site_hooks
    assert_no_site_hooks(bundle)
    for t in ("1", "2"):
        r = out[t]
        assert r["probs"]["C"]["source"] == "e0_center" and r["bounds"]["C"] == e0[t]["short"]["bounds"]
        assert r["E"]["C"] == local_support(0.3, 0.2, 0.3, 0.02)["E"]
        assert r["E_tok"] == r["E"][r["j"]] and r["j"] == run.select(r["masses"])
        W = r["window"]
        assert all(W[2] <= a < b <= W[3] <= r["heard_samples"] for a, b in r["bounds"].values())
        assert abs(float(extra["vecs"][f"t{t}_mean"].sum()) - 1) < 1e-12
    assert extra["lid_calls"] == len(calls) <= 4 and all(n == 8000 for n in calls)     # only L/R new
    assert all(p.grad is None for p in bundle.model.parameters())


# ---- population / sources -----------------------------------------------------------------------------

def test_exposed_center_reuse_and_groups():
    con = json.loads(Path(run.CONSTRUCTION).read_text())
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    for p in con["positions"]:
        e = json.loads(Path(f"{run.E0_RUN}/rows/{cidx[p['utterance_id']]:03d}.json").read_text())["positions"][str(p["t"])]
        assert list(run.candidate_bounds(e["window"][2], e["window"][3])["C"]) == e["short"]["bounds"]
    rows = json.loads(Path("results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]
    assert {g: sum(r["group"] == g for r in rows) for g in an.GROUPS} == an.EXPECTED == CFG["population"]["historical_groups"]


def test_frozen_sources_and_panel():
    for rel, h in CFG["source_sha256"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == h, rel
    assert hashlib.sha256(Path(CFG["T2"]["panel"]).read_bytes()).hexdigest() == CFG["T2"]["panel_sha256"]


# ---- T0 decision ---------------------------------------------------------------------------------------

def _rows(**mods):
    rows = []
    for g, n in an.EXPECTED.items():
        for i in range(n):
            rows.append({"dialogue_id": f"D{(i * 7 + len(rows)) % 20:02d}", "group": g, "E_1s": 0.6 if g in ("EN_TP", "ZH_FP") else 0.0,
                         "E_tok": 0.6 if g in ("EN_TP", "ZH_FP") else 0.0, "C_tokctx": 0.0})
    for g, f in mods.items():
        for i, r in enumerate([r for r in rows if r["group"] == g]):
            f(i, r)
    return rows


def _decide(rows):
    G = {g: [r for r in rows if r["group"] == g] for g in an.GROUPS}
    keys, W = an.draw_weights([f"D{i:02d}" for i in range(20)], 10000, 240924)
    P = an.predicates(G, keys, W)
    return P, an.decide_t0(True, P)


def fp_supp(i, r):
    if i < 4:
        r["E_tok"] = 0.05


def test_t0_supported_and_boundaries():
    P, lab = _decide(_rows(ZH_FP=fp_supp))
    assert lab == "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED"
    P, lab = _decide(_rows(ZH_FP=lambda i, r: r.update(E_tok=0.05) if i < 3 else None))
    assert not P["predicates"]["fp_suppression"] and lab == "P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE"
    P, lab = _decide(_rows(ZH_FP=fp_supp, EN_TP=lambda i, r: r.update(E_tok=0.19) if i < 9 else None))
    assert not P["predicates"]["tp_evidence"] and lab != "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED"
    # E_1s - E_tok must be >= 0.25
    P, lab = _decide(_rows(ZH_FP=lambda i, r: r.update(E_tok=0.05, E_1s=0.29)))
    assert not P["predicates"]["fp_suppression"]


def test_t0_contrast_recall_precedence_no_fallback():
    def fpc(i, r):
        r["C_tokctx"] = -0.3
    P, lab = _decide(_rows(ZH_FP=fpc))
    assert lab == "P2_SEL_T_CONTEXT_CONTRAST_ONLY"
    P, lab = _decide(_rows(EN_FN=lambda i, r: r.update(E_tok=0.3) if i < 9 else None))
    assert lab == "P2_SEL_T_RECALL_ONLY"
    P, lab = _decide(_rows(ZH_FP=fp_supp, EN_FN=lambda i, r: r.update(E_tok=0.3)))
    assert lab == "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED"       # supported precedes recall
    assert an.decide_t0(False, P) == "P2_SEL_T_INVALID"
    P2 = json.loads(json.dumps(P))
    P2["draws_ok"] = False
    assert an.decide_t0(True, P2) == "P2_SEL_T_INVALID"
    with pytest.raises(ValueError):
        an.selection_artifact({"label": "P2_SEL_T_CONTEXT_CONTRAST_ONLY"}, "x", "y")


# ---- T1 decision ---------------------------------------------------------------------------------------

def iv(e, l, h, n=10000):
    return {"estimate": e, "ci": [l, h], "valid_draws": n}


def test_t1_labels_including_en_no_regression():
    th = CFG["T1"]["thresholds"]
    f = {"conf": iv(3.0, 1.0, 4.0), "en": iv(0, 0, 0), "zh": iv(-0.05, -0.2, 0), "paired_zh": iv(0.15, 0.01, 0.3),
         "corr_en": iv(0, 0, 0), "corr_zh": iv(0.0, 0, 0.1)}
    R = {"benefit_retention": 0.85, "harm_ratio": 0.25}
    O = {"en": 0.0, "zh": 0.0}
    ok = {"mean_new_minus_old": 0.0, "new_corruptions": 0}
    L = lambda f=f, r=R, o=O, e=ok, v=True: an.decide_t1(v, f, r, o, e, th)["label"]
    assert L() == "P2_SEL_T_TOKEN_GATE_SUPPORTED"
    assert L(e={"mean_new_minus_old": -2e-6, "new_corruptions": 0}) == "P2_SEL_T_TOKEN_GATE_STILL_UNSAFE"
    assert L(e={"mean_new_minus_old": 0.0, "new_corruptions": 1}) == "P2_SEL_T_TOKEN_GATE_STILL_UNSAFE"
    assert L(o={"en": 0, "zh": 0.0501}) == "P2_SEL_T_TOKEN_GATE_STILL_UNSAFE"
    assert L(r={**R, "benefit_retention": 0.6999}) == "P2_SEL_T_TOKEN_GATE_TOO_CONSERVATIVE"
    assert L(r={**R, "harm_ratio": 0.51}) == "P2_SEL_T_NO_MATERIAL_GAIN"
    assert L(f={**f, "paired_zh": iv(0.2, 0.0, 0.3)}) == "P2_SEL_T_NO_MATERIAL_GAIN"
    assert L(v=False) == "P2_SEL_T_INVALID"


def test_auditor_independent():
    src = Path("experiments/inference_cf_p2sel_t_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_analyze|inference_cf_p2sel_t\b)", src, re.M) is None
