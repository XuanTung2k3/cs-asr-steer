"""P2-SEQ focused tests (contract 660a619): panel identity; matched S0 zero-dose B/S bitwise topology equal to the
historical cached path; STEER D2 coupling (pre-step snapshot, scratch identity, gate E*R_B, alpha 2, L16, query>=4,
zero-gate site identity, first divergence after an edit, no parameter grads); exact AUTO call; canonical PIER
transition identity, outside-POI projection edits vs harm; deterministic count bootstrap; frozen label precedence;
reference/TTA-free runner; auditor independence."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle, _enc  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2seq as run  # noqa: E402
import experiments.inference_cf_p2seq_analyze as an  # noqa: E402
import experiments.inference_cf_p2seq_audit as au  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())


def test_panel_exact():
    p = json.loads(Path(run.PANEL).read_text())
    ids = [r["utterance_id"] for r in p["rows"]]
    assert hashlib.sha256(Path(run.PANEL).read_bytes()).hexdigest() == CFG["panel"]["sha256"]
    assert ids == CFG["panel"]["ids"] and len(set(ids)) == 100
    per = {}
    for r in p["rows"]:
        per[r["dialogue_id"]] = per.get(r["dialogue_id"], 0) + 1
    assert len(per) == 20 and set(per.values()) == {5}


def _decode(bundle, alpha, monkeypatch, lid=(0.9, 0.1)):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: lid[0], 5: lid[1]})
    counters = {"forwards": 0, "lid_calls": 0, "lid_cache_hits": 0, "d2_calls": 0, "autograd_calls": 0}
    with torch.inference_mode():
        r, vecs = run.seq_decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=_enc(), partition=PART,
                                 null_probs={4: .5, 5: .5}, language_ids=(4, 5), alpha=alpha, lid_cache={}, lid_key="u",
                                 counters=counters, nfp=4, max_new_tokens=8, cB=CB, cE=CE, language_token_ids=(4, 5))
    return r, vecs, counters


def test_s0_matched_zero_dose_equals_historical_cached_path(monkeypatch):
    bundle = _bf16_bundle()
    r, vecs, c = _decode(bundle, 0.0, monkeypatch)
    assert all(st["S_bitwise_B"] and st["next"] == st["unsteered_next"] and not st["d2_called"] and not st["edited"]
               for st in r["steps"])
    assert all(st.get("noedit_site_identity") for st in r["steps"]) and r["lineage_ok"] and r["distinct_caches"]
    assert c["d2_calls"] == 0 and c["autograd_calls"] == 0 and vecs == {}
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    with torch.inference_mode():
        h = cached.cached_decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=_enc(),
                                 conditions={"cB": CB, "cE": CE, "language_token_ids": [4, 5]}, partition=PART,
                                 null_probs={4: .5, 5: .5}, language_ids=(4, 5), layer=16, alpha=0.0, max_new_tokens=8,
                                 num_forced_prefix=4)
    assert h["tokens"] == r["tokens"] and h["terminated"] == r["terminated"]


def test_s2_d2_coupling_gate_alpha_and_attribution(monkeypatch):
    bundle = _bf16_bundle()
    r0, _, _ = _decode(bundle, 0.0, monkeypatch)
    r2, vecs, c = _decode(bundle, 2.0, monkeypatch)
    calls = [st for st in r2["steps"] if st["d2_called"]]
    assert calls and c["d2_calls"] == len(calls) and c["autograd_calls"] == len(calls)
    assert all(st["d2_logits_bitwise_B"] and st["d2_site_bitwise_B"] for st in calls)
    assert all(st["query"] >= 4 and st["t"] >= 1 for st in calls)
    assert r2["steps"][0]["d2_called"] is False and r2["steps"][0]["edited"] is False          # t0 / query 3 never edited
    for st in r2["steps"]:
        if st["fb"] is None:
            assert st["g"] == pytest.approx(st["E"] * st["R_B"], abs=1e-12) and st["dose"] == pytest.approx(2 * st["g"])
        if not st["edited"]:
            assert st["noedit_site_identity"]
    fe = r2["first_edit_t"]
    assert fe is not None and all(st["S_bitwise_B"] for st in r2["steps"] if st["t"] < fe)
    for t in [st["t"] for st in r2["steps"] if st["edited"]]:
        d = vecs[f"t{t}_d2"]
        assert abs(float(np.linalg.norm(d.astype(np.float64))) - 1) < 2e-6 and f"t{t}_pre" in vecs and f"t{t}_post" in vecs
    if r2["tokens"] != r0["tokens"]:
        k = next((t for t, (a, b) in enumerate(zip(r2["tokens"], r0["tokens"])) if a != b), min(len(r2["tokens"]), len(r0["tokens"])))
        assert fe <= k
    assert all(p.grad is None for p in bundle.model.parameters())
    with pytest.raises(ValueError):
        _decode(bundle, 1.0, monkeypatch)


def test_zero_gate_when_no_english_evidence_is_no_edit(monkeypatch):
    bundle = _bf16_bundle()
    r2, vecs, c = _decode(bundle, 2.0, monkeypatch, lid=(0.01, 0.99))        # E = 0 -> g = 0 everywhere
    r0, _, _ = _decode(bundle, 0.0, monkeypatch, lid=(0.01, 0.99))
    assert c["d2_calls"] == 0 and not any(st["edited"] for st in r2["steps"]) and r2["tokens"] == r0["tokens"]
    assert all(st["S_bitwise_B"] for st in r2["steps"])


def test_auto_call_exact():
    src = Path("experiments/inference_cf_p2seq.py").read_text()
    assert ('bundle.model.generate(**inputs, task="transcribe", language=None, do_sample=False, num_beams=1,\n'
            '                                                 max_new_tokens=MAX_NEW, condition_on_prev_tokens=False)') in src
    assert "content = auto[:auto.index(eos)] if eos in auto else auto" in src
    assert CFG["decode"]["S1_auto"].startswith("exact historical model.generate")


def test_outside_projection_edits_vs_harm():
    ref = "我们 去 school 吃饭"
    same = an.outside(ref, ref, ref)
    assert same["outside_edits"] == 0 and same["outside_harm"] == 0
    o = an.outside(ref, ref, "我们 去 school 吃面")                 # outside Mandarin unit corrupted
    assert o["outside_edits"] >= 1 and o["outside_harm"] >= 1
    o2 = an.outside(ref, "我门 去 school 吃饭", ref)                # baseline wrong -> fixed: edit but not harm
    assert o2["outside_edits"] >= 1 and o2["outside_harm"] == 0
    o3 = an.outside(ref, ref, "嗯 我们 去 school 吃饭")
    assert o3["leading_insertion_change"] == 1


def test_counts_and_transition_identity():
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    R = ["我 喜欢 apple pie", "today 天气 很 好"]
    b = ["我 喜欢 苹果 pie", "today 天气 很 好"]
    s = ["我 喜欢 apple pie", "to day 天气 很 好"]
    cc = correction_corruption(R, b, s)
    assert cc["net_corrections"] == corpus_metrics(R, b)["num_poi_errors"] - corpus_metrics(R, s)["num_poi_errors"]
    c = an.utt_counts(R[0], b[0])
    m = corpus_metrics([R[0]], [b[0]])
    assert c[0] == m["num_poi_errors"] and c[1] == m["num_poi"] and c[3] == m["num_ref_units"]


def test_bootstrap_deterministic_and_empty_denominator():
    d = [f"D{i % 20:02d}" for i in range(100)]
    a, b = an.draws(d), an.draws(d)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and len(a) == 2000 and all(len(x) == 100 for x in a)
    ca = np.ones((100, 2))
    cb = np.zeros((100, 2))
    r = an.ratio_delta(ca, cb, 0, 1, a)
    assert r["delta"] is None and r["valid_draws"] == 0
    assert an.rate_ci(cb, 0, 1, a)["rate"] is None


BASEPT = {"mer_increase": 0.0, "zh_cer_increase": 0.0, "en_wer_increase": 0.0, "pier_gain": 0.01, "net_poi_error_reduction": 5,
          "zh_retention": 0.995, "en_retention": 0.96, "poi_corruption_rate": 0.02, "outside_harm_rate": 0.005, "cap_increase": 1}


def test_label_precedence_and_boundaries():
    L = lambda **k: an.decide(k.pop("valid", True), {**BASEPT, **k})["label"]
    assert L() == "P2_SEQ_PROMISING"
    assert L(net_poi_error_reduction=4) == "P2_SEQ_NO_USEFUL_GAIN"
    assert L(pier_gain=0.0049) == "P2_SEQ_NO_USEFUL_GAIN"
    assert L(pier_gain=0.005) == "P2_SEQ_PROMISING"
    assert L(mer_increase=0.0051) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(zh_retention=0.9899) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(en_retention=0.9499) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(poi_corruption_rate=0.0501) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(outside_harm_rate=0.0101) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(cap_increase=2) == "P2_SEQ_SEQUENCE_DAMAGE"
    assert L(en_retention=None, poi_corruption_rate=None) == "P2_SEQ_PROMISING"           # empty -> not assessable, passes
    assert L(net_poi_error_reduction=0, pier_gain=0.0) == "P2_SEQ_NO_USEFUL_GAIN"          # zero edits valid
    assert L(valid=False) == "P2_SEQ_INVALID"
    assert L(mer_increase=1.0, net_poi_error_reduction=50, pier_gain=0.2) == "P2_SEQ_SEQUENCE_DAMAGE"


def test_runner_reference_tta_free_and_auditor_independent():
    assert au.runner_clean()["ok"]
    src = Path("experiments/inference_cf_p2seq_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*_analyze", src, re.M) is None
    for rel, h in CFG["source_sha256"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == h, rel
