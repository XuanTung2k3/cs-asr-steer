"""P2-SEL-E focused tests (revised contract b693b8d, section 5): old-E regression and decomposition,
100-way Q semantics, same-center short window, causal lags/no future, exact branch formulas (R3 bound),
reference-free E_new inputs, zero E_new exact no-edit, C_new == P2-SEL gated hook, 180-key/group
identity, deterministic H_E1..H_E4 precedence and exactly-one/no-fallback, E1 labels, auditor agreement."""
from __future__ import annotations

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
from test_inference_cf_p1 import CB, CE, FRAMES, PART  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle, _enc  # noqa: E402
from test_inference_cf_p2sel import _gate, _refs  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2sel as p2sel  # noqa: E402
import experiments.inference_cf_p2sel_e as run  # noqa: E402
import experiments.inference_cf_p2sel_e_analyze as an  # noqa: E402
import experiments.inference_cf_p2sel_e_audit as au  # noqa: E402
from csasr.inference_cf.core_r2 import local_support  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())


# ---- quantities ------------------------------------------------------------------------------------

def test_decomposition_matches_frozen_local_support_and_q_is_pair_mass():
    for pe, pm, ne, nm in ((0.03, 0.07, 0.2, 0.7), (0.5, 0.1, 0.01, 0.9), (1e-9, 0.4, 0.3, 0.3), (0.0, 0.0, 0.1, 0.2)):
        d = run.decompose(pe, pm, ne, nm)
        s = local_support(pe, pm, ne, nm)
        assert d["E"] == s["E"] and abs(d["A"] - s["A"]) < 1e-12
        assert d["Q"] == pe + pm and 0 <= d["Q"] <= 1
        assert d["ell_local"] == pytest.approx(math.log((pe + 1e-12) / (pm + 1e-12)))
        assert d["E"] == max(0.0, math.tanh(d["A"] / 2)) or d["E"] == pytest.approx(max(0.0, math.tanh(d["A"] / 2)), abs=1e-15)
        assert (d["null_share"] is None) == (d["A"] <= 0)


def test_short_bounds_same_center_and_clamped():
    assert run.short_bounds(16000, 32000, 160000) == (20000, 28000)              # center 24000 kept
    assert run.short_bounds(0, 16000, 160000) == (4000, 12000)
    assert run.short_bounds(1000, 17000, 20000) == (5000, 13000)
    assert run.short_bounds(144000, 160000, 160000) == (148000, 156000)
    assert run.short_bounds(0, 7000, 7000) == (0, 7000)                         # heard < 8000
    assert run.short_bounds(8001, 8003, 9000) == (1000, 9000)                   # clamped to heard-8000
    a, b = run.short_bounds(31, 16032, 50000)
    assert b - a == 8000 and a == (31 + 16032 - 8000) // 2


def test_lags_causal_missing_none_zero_no_future():
    e = {3: 0.2, 4: None, 5: 0.9, 6: 0.7, 7: 0.99}
    assert run.lag_values(e, 6) == (0.9, 0.0)        # q-1=5, q-2=4 (None -> 0)
    assert run.lag_values(e, 4) == (0.2, 0.0)        # q-2=2 missing
    e2 = dict(e)
    e2[7] = 123.0                                     # future value cannot matter
    e2[6] = -5.0                                      # current value cannot matter
    assert run.lag_values(e2, 6) == run.lag_values(e, 6)


def test_branch_formulas_exact_and_r3_bound():
    r = {"E": 0.64, "E_lag1": 0.3, "E_lag2": 0.5, "E_short": 0.25, "Q": 0.4, "ell_local": -0.2, "ell_null": -3.0}
    assert run.e_new("R1", r) == 0.64 * 0.5
    assert run.e_new("R2", r) == math.sqrt(0.64 * 0.25)
    assert run.e_new("R3", r) == 0.64 * 0.4 and 0 <= run.e_new("R3", r) <= r["E"]
    assert run.e_new("R4", r) == max(0.0, math.tanh((-0.2 + 1.5) / 2))
    with pytest.raises(ValueError):
        run.e_new("R3", {**r, "Q": 1.5})
    with pytest.raises(ValueError):
        run.e_new("R5", r)
    assert CFG["E0"]["repair_formulas"]["R2_multiscale"] == "E_new=sqrt(E_long*E_short)"


def test_e_new_inputs_reference_free():
    assert au.runtime_surface_clean()["ok"]
    for br, ins in run.E_NEW_INPUTS.items():
        assert not any(x in ("E_oracle", "timing_error_sec", "stratum", "group", "target_ids") for x in ins)


def test_provider_integrity_frozen():
    pv = run.provider_integrity(CFG["E0"]["integrity"]["generation_config_path"])
    I = CFG["E0"]["integrity"]
    assert pv == {"generation_config_sha256": I["generation_config_sha256"], "mapping_sha256": I["canonical_mapping_sha256"],
                  "n": 100, "en_zh": I["en_zh_token_ids"]}


# ---- GPU-path functions on the bf16 tiny model ---------------------------------------------------------

def test_e0_utterance_no_steering_frozen_window_and_crops(monkeypatch):
    bundle = _bf16_bundle()
    calls = []

    def fake_lid(b, w, ids):
        calls.append(len(w))
        p = np.full(len(ids), 1.0 / len(ids))
        return {i: float(x) for i, x in zip(ids, p)}
    monkeypatch.setattr(cached, "native_lid", fake_lid)
    ids = tuple(range(100))
    run_en, run_zh = run.EN_ID, run.ZH_ID
    monkeypatch.setattr(run, "EN_ID", 4)
    monkeypatch.setattr(run, "ZH_ID", 5)
    wav = np.ones(FRAMES * 320, dtype=np.float32)
    with torch.inference_mode():
        out = run.e0_utterance(bundle, encoded=_enc(), waveform=wav, tokens=[10, 47, 12], ts=[1, 2], language_ids=ids, cB=CB)
    from csasr.lss.sites import assert_no_site_hooks
    assert_no_site_hooks(bundle)
    for t in ("1", "2"):
        r = out[t]
        assert r["query"] == len(CB) + int(t) - 1
        s0, s1 = r["window"][2], r["window"][3]
        assert r["long"]["bounds"] == [s0, s1]
        assert r["short"]["bounds"] == list(run.short_bounds(s0, s1, r["heard_samples"]))
        assert r["long"]["softmax_sum"] == pytest.approx(1.0) and r["long"]["n_probs"] == 100
    assert calls == [16000, 8000, 16000, 8000]


def test_e1_zero_gate_exact_noedit_and_positive_gate_equals_p2sel_hook(monkeypatch):
    bundle = _bf16_bundle()
    ctx, toks, vecs, none = _refs(bundle, monkeypatch)
    gates = {1: {"g_new": 0.0, "query": 4}, 2: {"g_new": 0.6, "query": 5}}
    with torch.inference_mode():
        recs, post, logits = run.e1_utterance(ctx, encoded=_enc(), tokens=toks, ts=[1, 2], gates=gates, ref_vecs=vecs, ref_none=none)
        old, _, ologits = p2sel.s1_pass_a(ctx, encoded=_enc(), tokens=toks, ts=[1, 2],
                                          gates={1: _gate(0.0, 4), 2: _gate(0.6, 5)}, ref_vecs=vecs, ref_none=none)
    assert recs["1"]["arm"]["zero_gain_bitwise_none"] and np.array_equal(logits["t1_Cnew"], none["t1_none"])
    assert recs["2"]["arm"]["edit_norm"] == old["2"]["arms"]["C2"]["edit_norm"]
    assert np.array_equal(logits["t2_Cnew"], ologits["t2_C2"])
    assert all(recs[t]["restore_bitwise"] and recs[t]["hb_bitwise_p2dir"] for t in ("1", "2"))
    assert all(p.grad is None for p in bundle.model.parameters())


# ---- population / groups ---------------------------------------------------------------------------------

def test_180_keys_and_expected_groups_from_frozen_scalars():
    pos = json.loads(Path(CFG["E0"]["population"]).read_text())["positions"]
    e = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in
         json.loads(Path("results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    g = [an.group_of(p["stratum"], e[(p["utterance_id"], p["t"])]) for p in pos]
    assert {k: g.count(k) for k in an.GROUPS} == an.EXPECTED


# ---- diagnosis -------------------------------------------------------------------------------------------

def _rows(seed=0, **mods):
    """Synthetic E0 rows: 20 dialogues, groups with the frozen sizes."""
    rnd = random.Random(seed)
    rows = []
    sizes = {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60}
    for g, n in sizes.items():
        for i in range(n):
            d = f"D{(i * 7 + len(rows)) % 20:02d}"
            zero = g in ("EN_FN", "ZH_TN")
            r = {"dialogue_id": d, "group": g, "E": 0.0 if zero else 0.6, "E_short": 0.0 if zero else 0.5,
                 "Q": 0.9, "ell_local": 0.5, "ell_null": -1.0, "A": 1.5, "null_share": 1 / 1.5,
                 "E_lag1": 0.0, "E_lag2": 0.0, "E_oracle": None}
            rows.append(r)
    for g, f in mods.items():
        for i, r in enumerate([r for r in rows if r["group"] == g]):
            f(i, r)
    return rows


def _decide(rows):
    G = {g: [r for r in rows if r["group"] == g] for g in an.GROUPS}
    keys, W = an.draw_weights([f"D{i:02d}" for i in range(20)], 10000, 240924)
    H = an.hypotheses(G, keys, W, CFG)
    return H, an.decide_e0(True, H)


def _audit_decide(rows):
    rr = [{"d": r["dialogue_id"], "g": r["group"], "E": r["E"], "Es": r["E_short"], "Q": r["Q"], "ll": r["ell_local"],
           "ln": r["ell_null"], "A": r["A"], "ns": r["null_share"], "lag": [r["E_lag1"], r["E_lag2"]], "Eo": r["E_oracle"]}
          for r in rows]
    return au.e0_decide(rr)


def lowq_fp(i, r):
    r["Q"] = 0.1 if i < 4 else 0.9


def test_h3_selects_r3_and_auditor_agrees():
    rows = _rows(ZH_FP=lowq_fp)
    H, d = _decide(rows)
    assert H["H_E3"]["pass"] and d == {"label": "P2_SEL_E_REPAIR_SELECTED", "selected": "R3"}
    a = _audit_decide(rows)
    assert a["selected"] == "R3" and abs(a["H"]["h3_lower80"] - H["H_E3"]["bootstrap_FP_minus_TP_lowQ"]["lower80"]) <= 1e-12


def test_h3_fails_when_tp_also_low_q():
    rows = _rows(ZH_FP=lowq_fp, EN_TP=lambda i, r: r.update(Q=0.1) if i < 11 else None)
    H, d = _decide(rows)
    assert not H["H_E3"]["pass"] and d["label"] == "P2_SEL_E_DIAGNOSIS_AMBIGUOUS"


def test_precedence_h1_before_h2_h3_and_localizer_stop():
    oracle = lambda i, r: r.update(E_oracle=0.5)
    rows = _rows(ZH_FP=lowq_fp, EN_FN=oracle)
    H, d = _decide(rows)
    assert H["H_E1"]["oracle_FN_pattern"] and H["H_E3"]["pass"]
    assert d == {"label": "P2_SEL_E_LOCALIZER_PRIMARY", "selected": None}
    assert _audit_decide(rows)["label"] == "P2_SEL_E_LOCALIZER_PRIMARY"
    static = lambda i, r: r.update(E_short=0.05) if i < 4 else None
    H2, d2 = _decide(_rows(ZH_FP=lambda i, r: (lowq_fp(i, r), static(i, r))))
    assert H2["H_E1"]["static_scale_pattern"] and d2["selected"] == "R2"


def test_h2_selects_r1_before_h3():
    def fp(i, r):
        lowq_fp(i, r)
        r["E"] = 0.8
    rows = _rows(ZH_FP=fp, EN_TP=lambda i, r: r.update(E_lag1=0.5))
    H, d = _decide(rows)
    assert H["H_E2"]["pass"] and H["H_E3"]["pass"] and d["selected"] == "R1"
    assert _audit_decide(rows)["selected"] == "R1"


def test_h4_selects_r4_and_no_fallback_when_nothing_passes():
    def fp(i, r):
        if i < 4:
            r.update(ell_local=-0.1, ell_null=-3.0, A=2.9, null_share=3.0 / 2.9)
    H, d = _decide(_rows(ZH_FP=fp))
    assert H["H_E4"]["pass"] and d["selected"] == "R4"
    H0, d0 = _decide(_rows())
    assert d0 == {"label": "P2_SEL_E_DIAGNOSIS_AMBIGUOUS", "selected": None}
    assert an.decide_e0(False, H)["label"] == "P2_SEL_E_INVALID"


# ---- E1 decision ------------------------------------------------------------------------------------------

def iv(e, l, h, n=10000):
    return {"estimate": e, "ci": [l, h], "valid_draws": n}


TH = CFG["E1"]["thresholds"]


def fam(**o):
    b = {"conf": (3.0, 1.0, 4.0), "en": (0, -0.1, 0.1), "zh": (-0.05, -0.2, 0), "paired_zh": (0.15, 0.01, 0.3),
         "corr_en": (0, 0, 0.1), "corr_zh": (0.05, 0, 0.1)}
    b.update(o)
    return {k: iv(*v) for k, v in b.items()}


def test_e1_label_precedence():
    R = {"benefit_retention": 3.0 / 3.52, "harm_ratio": 0.05 / 0.195}
    O = {"en": 0.0, "zh": 0.05}
    L = lambda f=None, r=R, o=O, v=True: an.decide_e1(v, f or fam(), r, o, TH)["label"]
    assert L() == "P2_SEL_E_REPAIR_SUPPORTED"
    assert L(o={"en": 0, "zh": 0.0501}) == "P2_SEL_E_REPAIR_STILL_UNSAFE"
    assert L(fam(corr_zh=(0.0, 0, 0.1001))) == "P2_SEL_E_REPAIR_STILL_UNSAFE"
    assert L(fam(zh=(-0.3, -0.2501, 0))) == "P2_SEL_E_REPAIR_STILL_UNSAFE"
    assert L(r={**R, "benefit_retention": 0.6999}) == "P2_SEL_E_REPAIR_TOO_CONSERVATIVE"
    assert L(r={**R, "benefit_retention": 0.70}) == "P2_SEL_E_REPAIR_SUPPORTED"
    assert L(r={**R, "harm_ratio": 0.5001}) == "P2_SEL_E_NO_MATERIAL_GAIN"
    assert L(fam(paired_zh=(0.0999, 0.01, 0.3))) == "P2_SEL_E_NO_MATERIAL_GAIN"
    assert L(fam(paired_zh=(0.2, 0.0, 0.3))) == "P2_SEL_E_NO_MATERIAL_GAIN"
    assert L(fam(conf=(3.0, 0.0, 4.0))) == "P2_SEL_E_NO_MATERIAL_GAIN"
    assert L(v=False) == "P2_SEL_E_INVALID"


def test_auditor_independent_and_panel_frozen():
    src = Path("experiments/inference_cf_p2sel_e_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*_analyze", src, re.M) is None
    import hashlib
    assert hashlib.sha256(Path(CFG["E2"]["panel"]).read_bytes()).hexdigest() == CFG["E2"]["panel_sha256"]
