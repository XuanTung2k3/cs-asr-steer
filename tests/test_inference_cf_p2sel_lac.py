"""P2-SEL-LAC focused tests (contract be460ce): unmasked-only candidates with lowest-ID ties, hard-zero mask
bounds/outside bytes/no-op, S identity/offset invariance/F bounds, waveform feature adapter identity, fresh
masked encoder + exact-prefix branch reuse on a bf16 tiny model (no future token, query identity, no autograd),
frozen LAC0/LAC1 precedence, auditor agreement/independence, 180 keys/groups, frozen sources and panel."""
from __future__ import annotations

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
from test_inference_cf_p1 import CB, FRAMES, encoded  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle  # noqa: E402

import experiments.inference_cf_p2sel_lac as run  # noqa: E402
import experiments.inference_cf_p2sel_lac_analyze as an  # noqa: E402
import experiments.inference_cf_p2sel_lac_audit as au  # noqa: E402
from csasr.inference_cf.lexical_compatibility import hard_mask, lexical, select_candidates, waveform_model_inputs  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())


def test_candidates_unmasked_lowest_id_ties():
    z = np.zeros(20, dtype=np.float32)
    z[[7, 3, 12]] = 5.0          # tie in V_E at 3, 7, 12
    z[[15, 9]] = 4.0             # tie in V_M at 9, 15
    VE, VM = [12, 7, 3, 1], [15, 9, 2]                 # scrambled order
    assert select_candidates(z, VE, VM) == (3, 9)
    assert (au.cand(z, VE), au.cand(z, VM)) == (3, 9)
    z2 = z.copy()
    z2[1] = 9.0
    assert select_candidates(z2, VE, VM)[0] == 1
    with pytest.raises(ValueError):
        select_candidates(np.array([np.nan, 1.0]), [0], [1])


def test_hard_mask_bounds_bytes_and_noop():
    x = np.array([0.5, -0.0, 0.25, -0.75, 0.125, -0.0], dtype=np.float32)
    xm = hard_mask(x, 2, 4)
    assert xm.shape == x.shape and xm.dtype == np.float32
    assert xm[:2].tobytes() == x[:2].tobytes() and xm[4:].tobytes() == x[4:].tobytes()     # incl. negative zeros
    assert np.all(xm[2:4].view(np.uint32) == 0)                                           # positive zeros
    assert not np.array_equal(xm, x)
    z = np.zeros(6, dtype=np.float32)
    assert hard_mask(z, 1, 3).tobytes() == z.tobytes()                                     # zero-energy no-op valid
    with pytest.raises(ValueError):
        hard_mask(x, 4, 7)                                                                # beyond heard
    with pytest.raises(ValueError):
        hard_mask(x.astype(np.float64), 1, 2)
    big = np.ones(480010, dtype=np.float32)
    with pytest.raises(ValueError):
        hard_mask(big, 479990, 480005)                                                    # beyond 30 s heard cap


def test_lexical_identity_offset_and_F():
    z = np.array([1.0, 3.5, -2.0, 0.25], dtype=np.float32)
    zm = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    r = lexical(z, zm, 1, 3)
    assert r["M0"] == 3.25 and r["Mmask"] == 0.5 and r["S_lex"] == 2.75
    assert r["Delta_E"] == 2.5 and r["Delta_M"] == -0.25 and r["S_lex"] == r["Delta_E"] - r["Delta_M"]
    assert r["F_lex"] == pytest.approx(math.tanh(2.75 / 2))
    r2 = lexical(z + 7, zm - 3, 1, 3)                                                     # common offsets cancel
    assert r2["S_lex"] == pytest.approx(r["S_lex"])
    assert lexical(z, zm, 3, 1)["F_lex"] == 0.0                                           # negative S -> 0
    assert lexical(z, z, 1, 3)["S_lex"] == 0.0
    assert math.tanh(math.log(17 / 3) / 2) == pytest.approx(0.70) and math.tanh(math.log(11 / 9) / 2) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        lexical(np.array([np.inf, 0.0]), np.zeros(2), 0, 1)


def test_waveform_adapter_equals_batch_model_inputs():
    from types import SimpleNamespace
    from transformers import WhisperProcessor
    from csasr.models.whisper import batch_model_inputs, load_audio
    proc = WhisperProcessor.from_pretrained(au.MODEL, local_files_only=True)
    b = SimpleNamespace(processor=proc, sample_rate=16000, device="cpu", dtype=torch.bfloat16)
    path = json.loads(Path("results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"][0]["audio_path"]
    ref = batch_model_inputs(b, [path])
    got = waveform_model_inputs(b, load_audio(path, 16000))
    assert torch.equal(ref["input_features"], got["input_features"]) and torch.equal(ref["attention_mask"], got["attention_mask"])


def test_lac0_masked_replay_fresh_and_prefix_exact(monkeypatch):
    bundle = _bf16_bundle()
    calls = []

    def fake_inputs(b, w):                 # tiny model: features depend on the (masked) waveform
        calls.append(float(np.abs(w).sum()))
        torch.manual_seed(0)
        base = torch.randn(1, 8, 2 * FRAMES)
        return {"input_features": (base * (1 + float(np.abs(w[:16000]).sum()) / 1e4)).to(torch.bfloat16),
                "attention_mask": torch.ones(1, 2 * FRAMES, dtype=torch.long)}
    monkeypatch.setattr(run, "waveform_model_inputs", fake_inputs)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32000).astype(np.float32)
    toks = [10, 47, 12, 50]
    rows = [{"t": t, "query": len(CB) + t - 1, "W_star": [1000, 9000], "input_ids": CB + toks[:t]} for t in (1, 3)]
    rows.append({"t": 2, "query": len(CB) + 1, "W_star": [2000, 10000], "input_ids": CB + toks[:2]})
    counters = {"encoder_calls": 0, "decoder_steps": 0, "counterfactual_evaluations": 0}
    with torch.inference_mode():
        recs, logits = run.lac0_utterance(bundle, waveform=x, seal_rows=rows, counters=counters, prompt=CB)
    assert counters["counterfactual_evaluations"] == 3 and counters["encoder_calls"] == 2       # identical mask reused once
    assert recs["1"]["branch_origin"] == "cold" and recs["3"]["branch_origin"] == "reused_exact_prefix"
    assert recs["2"]["branch_origin"] == "cold"
    for t in ("1", "2", "3"):
        r = recs[t]
        assert r["fed_equals_input_ids"] and r["query_matches"] and r["outside_left_equal"] and r["outside_right_equal"]
        assert r["inside_positive_zero"] and r["changed"] and r["logits_finite"]
    # reused-branch logits equal a cold rebuild for the same mask/prefix
    counters2 = dict.fromkeys(counters, 0)
    with torch.inference_mode():
        recs2, logits2 = run.lac0_utterance(bundle, waveform=x, seal_rows=[rows[1]], counters=counters2, prompt=CB)
    assert np.array_equal(logits2["t3_masked"], logits["t3_masked"])
    # zero-energy window is a valid no-op (no encoder/decoder work)
    xz = x.copy()
    xz[1000:9000] = 0.0
    counters3 = dict.fromkeys(counters, 0)
    with torch.inference_mode():
        recs3, logits3 = run.lac0_utterance(bundle, waveform=xz, seal_rows=[rows[0]], counters=counters3, prompt=CB)
    assert recs3["1"]["noop"] and counters3["encoder_calls"] == 0 and logits3 == {}
    assert all(p.grad is None for p in bundle.model.parameters())


# ---- frozen decisions ------------------------------------------------------------------------------

def _G(**mods):
    rows = []
    for g, n in an.EXPECTED.items():
        for i in range(n):
            rows.append({"dialogue_id": f"D{(i * 7 + len(rows)) % 20:02d}", "group": g, "S_lex": 2.0 if g == "EN_TP" else 0.0})
    for g, f in mods.items():
        for i, r in enumerate([r for r in rows if r["group"] == g]):
            f(i, r)
    return {g: [r for r in rows if r["group"] == g] for g in an.GROUPS}


def _dec(G, valid=True):
    keys, W = an.draw_weights([f"D{i:02d}" for i in range(20)], 10000, 240924)
    return an.decide_lac0(valid, G, keys, W)


def test_lac0_precedence_and_thresholds():
    assert _dec(_G())["label"] == "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    assert _dec(_G(EN_TP=lambda i, r: r.update(S_lex=an.S_STRONG) if i < 34 else r.update(S_lex=0.0)))["label"] == \
        "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    assert _dec(_G(EN_TP=lambda i, r: r.update(S_lex=1.7346) if i < 9 else None))["label"] != "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    assert _dec(_G(ZH_FP=lambda i, r: r.update(S_lex=an.S_WEAK)))["label"] == "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    fp_bad = lambda i, r: r.update(S_lex=0.21) if i < 2 else None
    assert _dec(_G(ZH_FP=fp_bad))["label"] == "P2_SEL_LAC_NOT_DISCRIMINATIVE"
    assert _dec(_G(ZH_FP=fp_bad, EN_FN=lambda i, r: r.update(S_lex=2.0) if i < 9 else None))["label"] == "P2_SEL_LAC_RECALL_ONLY"
    assert _dec(_G(EN_FN=lambda i, r: r.update(S_lex=2.0)))["label"] == "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    assert _dec(_G(), valid=False)["label"] == "P2_SEL_LAC_INVALID"
    with pytest.raises(ValueError):
        an.selection_artifact({"label": "P2_SEL_LAC_RECALL_ONLY"}, "x", "y")


def iv(e, l, h):
    return {"estimate": e, "ci": [l, h], "valid_draws": 10000}


def test_lac1_labels():
    f = {"conf": iv(3.0, 1.0, 4.0), "en": iv(0, 0, 0), "zh": iv(-0.05, -0.2, 0), "paired_zh": iv(0.15, 0.01, 0.3),
         "corr_en": iv(0, 0, 0), "corr_zh": iv(0, 0, 0.1)}
    R = {"benefit_retention": 0.8, "harm_ratio": 0.3}
    O = {"en": 0.0, "zh": 0.0}
    E = {"mean_new_minus_old": 0.0, "new_corruptions": 0}
    L = lambda **k: an.decide_lac1(k.get("v", True), k.get("f", f), k.get("r", R), k.get("o", O), k.get("e", E), None)
    assert L() == "P2_SEL_LAC_GATE_SUPPORTED"
    assert L(e={"mean_new_minus_old": 0.0, "new_corruptions": 1}) == "P2_SEL_LAC_GATE_STILL_UNSAFE"
    assert L(r={**R, "benefit_retention": 0.69}) == "P2_SEL_LAC_GATE_TOO_CONSERVATIVE"
    assert L(r={**R, "harm_ratio": 0.501}) == "P2_SEL_LAC_NO_MATERIAL_GAIN"
    assert L(v=False) == "P2_SEL_LAC_INVALID"


# ---- lineage / surface ------------------------------------------------------------------------------

def test_frozen_sources_population_panel_surface():
    for rel, h in CFG["source_sha256"].items():
        assert "sha256:" + hashlib.sha256(Path(rel).read_bytes()).hexdigest() == "sha256:" + h.replace("sha256:", ""), rel
    rows = json.loads(Path("results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())["rows"]
    assert {g: sum(r["group"] == g for r in rows) for g in an.GROUPS} == an.EXPECTED == CFG["population"]["historical_groups"]
    assert hashlib.sha256(Path("docs/inference_cf/P2_SEL_MINI_PANEL.json").read_bytes()).hexdigest() == \
        "266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5"
    assert au.surface_clean()["ok"]
    src = Path("experiments/inference_cf_p2sel_lac_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_analyze|lexical_compatibility)", src, re.M) is None
    assert an.S_STRONG == CFG["LAC0"]["thresholds"]["EN_TP_S_min_nat"] and an.S_WEAK == CFG["LAC0"]["thresholds"]["ZH_FP_S_max_nat"]
