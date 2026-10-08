"""S1 implementation tests (CPU, synthetic values only; no pretrained forward, no S1 outcome, no reference file)."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf import s1_evidence as S  # noqa: E402

CFG = json.loads((ROOT / "configs/inference_cf/s1_acoustic_evidence.json").read_text())
PANEL = json.loads((ROOT / CFG["panel"]).read_text())


# ---- runtime projection / candidates ---------------------------------------------------------------------------------

def test_runtime_projection_is_label_free_and_prefix_bounded():
    rp = S.runtime_projection(PANEL)
    assert len(rp["runtime_queries"]) == 180 and len(rp["utterances"]) == 80
    assert rp["runtime_queries"] == PANEL["runtime_queries"]
    text = json.dumps(rp)
    for bad in ("stratum", "EN-confusion", "target_ids", "competitor", "p2rj", "evaluation"):
        assert bad not in text
    full = {u["utterance_id"]: u["baseline_content_tokens"] for u in PANEL["utterances"]}
    for u in rp["utterances"]:
        mt = max(q["t"] for q in rp["runtime_queries"] if q["utterance_id"] == u["utterance_id"])
        assert u["max_t"] == mt and u["content_prefix_tokens"] == full[u["utterance_id"]][:mt]


def test_topk_descending_value_ascending_id_ties_finite_only():
    x = np.full(50, -np.inf, dtype=np.float32)
    x[[3, 7, 9, 11, 20]] = [1.0, 2.0, 2.0, 0.5, 2.0]
    assert S.topk(x, 4) == [7, 9, 20, 3]
    with pytest.raises(ValueError):
        S.topk(x, 6)
    with pytest.raises(ValueError):
        S.topk(x.astype(np.float64), 2)


def test_union_ascending_origin_bits_ranks_and_action_types():
    tops = {"M": [10, 50257, 7], "E": [7, 300, 50364], "AUTO": [10, 50257, 7]}
    u = S.union(tops, 50257)
    assert u["ids"] == [7, 10, 300, 50257, 50364]
    assert u["origin"] == [7, 5, 2, 5, 2]
    assert u["ranks"]["M"] == [3, 1, None, 2, None] and u["ranks"]["E"] == [1, None, 2, None, 3]
    assert u["types"] == ["LEXICAL", "LEXICAL", "LEXICAL", "TERMINATE", "OTHER_ACTION"]
    assert len(u["ids"]) <= 3 * 3
    assert S.membership_digest(u["ids"]) == "sha256:" + hashlib.sha256(b"7,10,300,50257,50364").hexdigest()


def test_processing_suppression_begin_only_at_t0_and_full_logsoftmax():
    from experiments.acoustic_s1 import logsoftmax_processed
    raw = np.random.default_rng(0).standard_normal(100).astype(np.float32)
    p0, l0 = logsoftmax_processed(raw, 0, [1, 2], [5])
    p1, l1 = logsoftmax_processed(raw, 1, [1, 2], [5])
    assert np.isneginf(p0[[1, 2, 5]]).all() and np.isfinite(p1[5]) and np.isneginf(p1[[1, 2]]).all()
    for p, l in ((p0, l0), (p1, l1)):
        fin = np.isfinite(p)
        assert l.dtype == np.float32 and np.isneginf(l[~fin]).all()
        assert abs(np.log(np.exp(l[fin].astype(np.float64)).sum())) < 1e-5
        assert np.array_equal(np.argsort(-p[fin], kind="stable"), np.argsort(-l[fin], kind="stable"))


def test_bf16_pack_roundtrip_and_rejects_non_bf16():
    import torch
    from experiments.acoustic_s1 import pack, unpack
    x = torch.randn(64).to(torch.bfloat16).float()
    assert np.array_equal(unpack(pack(x)), x.numpy())
    with pytest.raises(ValueError):
        pack(torch.tensor([1.0 + 2 ** -20]))


# ---- regions --------------------------------------------------------------------------------------------------------

def att_from(frames: np.ndarray, heard: int):
    heads = np.zeros((10, 1500), dtype=np.float32)
    heads[:, :len(frames)] = frames
    return S.heard_attention(heads, heard, 0.5)


def test_heard_attention_partial_frame_and_exact_integrals():
    heard = 320 * 9 + 100                                    # 10 frames, last frame has 100 samples
    rng = np.random.default_rng(1)
    fr = (rng.random(10) / 10).astype(np.float32)
    fr = fr / fr.sum() * np.float32(0.9)
    att = att_from(fr, heard)
    assert att["frames"] == 10 and att["mapped"] and abs(att["raw_mass"] - float(fr.astype(np.float64).sum())) < 1e-6
    I = S.Integrals(att["H"], heard)
    assert I.num(0, heard) == I.den and I.value(I.num(0, heard)) == 1.0
    # half-open additivity and uniform density inside the short last frame
    assert I.num(0, 1000) + I.num(1000, heard) == I.num(0, heard)
    a = np.array([Fraction(float(v)) for v in fr]) / sum(Fraction(float(v)) for v in fr)
    assert Fraction(I.num(320 * 9, 320 * 9 + 50), I.den) == a[9] / 2
    assert Fraction(I.num(160, 480), I.den) == a[0] / 2 + a[1] / 2
    # the float64-normalized query_mapping vector agrees with the exact rational
    assert np.allclose(att["a"], [float(v) for v in a], rtol=0, atol=1e-12)


def test_low_heard_mass_and_no_en_region_abstentions():
    heard = 320 * 100
    x = (np.random.default_rng(2).standard_normal(heard) * 0.05).astype(np.float32)
    low = att_from(np.full(100, 0.004, dtype=np.float32), heard)          # raw mass 0.4 < 0.5
    ivs_en = np.array([[0, 8000, 2], [8000, 20000, 1], [20000, heard, 0]])
    assert S.select_target(low, ivs_en, x, heard, CFG)["status"] == "LOW_HEARD_MASS"
    ok = att_from(np.full(100, 0.009, dtype=np.float32), heard)
    no_en = np.array([[0, heard, 2]])
    short = np.array([[0, 8000, 2], [8000, 11999, 1], [11999, heard, 0]])  # 3999 EN samples < 4000
    assert S.select_target(ok, no_en, x, heard, CFG)["status"] == "NO_EN_REGION"
    assert S.select_target(ok, short, x, heard, CFG)["status"] == "NO_EN_REGION"
    assert S.select_target(low, no_en, x, heard, CFG)["status"] == "NO_EN_REGION"       # absent EN precedes low mass


def test_target_choice_length_cap_ties_association_and_energy():
    heard = 320 * 200
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(heard) * 0.05).astype(np.float32)
    fr = np.full(200, 0.001, dtype=np.float32)
    fr[150:160] = 0.05                                       # attention peak inside the second EN interval
    att = att_from(fr, heard)
    ivs = np.array([[0, 10000, 1], [10000, 40000, 0], [40000, 64000, 1]])
    t = S.select_target(att, ivs, x, heard, CFG)
    assert t["status"] == "OK" and t["bounds"][1] - t["bounds"][0] == 24000           # min(len, 32000)
    assert t["bounds"] == [40000, 64000]
    long = np.array([[0, heard, 1]])                         # 64000 samples -> 32000-sample crop
    t2 = S.select_target(att, long, x, heard, CFG)
    assert t2["bounds"][1] - t2["bounds"][0] == 32000 and t2["bounds"][0] <= 48000 < 51200 <= t2["bounds"][1]
    flat = att_from(np.full(200, 0.005, dtype=np.float32), heard)
    t3 = S.select_target(flat, np.array([[0, 20000, 1], [20000, heard, 0]]), x, heard, CFG)
    assert t3["bounds"] == [0, 20000]                        # single crop
    t4 = S.select_target(flat, np.array([[0, 4000, 1], [4000, 8000, 0], [8000, 12000, 1], [12000, heard, 0]]), x, heard, CFG)
    assert t4["bounds"] == [0, 4000]                         # exact tie -> earliest start
    assert t4["status"] == "LOW_ASSOCIATION"                 # 4000/64000 = .0625 < .10
    silent = x.copy()
    silent[40000:64000] = 0
    t5 = S.select_target(att, ivs, silent, heard, CFG)
    assert t5["status"] == "LOW_ENERGY" and t5["bounds"] == [40000, 64000]             # never shortened to qualify


def test_offtarget_duration_disjoint_guards_and_tie_by_energy_ratio():
    heard = 320 * 200
    rng = np.random.default_rng(4)
    x = (rng.standard_normal(heard) * 0.05).astype(np.float32)
    fr = np.full(200, 0.001, dtype=np.float32)
    fr[150:160] = 0.08
    att = att_from(fr, heard)
    ivs = np.array([[0, 6400, 1], [6400, 40000, 2], [40000, 56000, 1], [56000, heard, 0]])
    t = S.select_target(att, ivs, x, heard, CFG)
    o = S.select_offtarget(att, ivs, x, heard, t, CFG)
    L = t["bounds"][1] - t["bounds"][0]
    assert o["status"] == "OK" and o["bounds"][1] - o["bounds"][0] == L
    a, b = t["bounds"]
    s, e = o["bounds"]
    assert e <= a or s >= b
    assert o["en_fraction"] <= 0.1 and o["integral"] <= min(0.1, t["integral"] / 2) and 0.5 <= o["energy_ratio"] <= 2
    assert s % 320 == 0 or s == heard - L
    # equal attention everywhere outside the target: the energy ratio closest to 1 wins, then the earliest start
    y = x.copy()
    y[:12800] *= 3.0                                        # far from target energy -> not preferred
    o2 = S.select_offtarget(att, ivs, y, heard, t, CFG)
    assert o2["status"] == "OK" and o2["bounds"][0] >= 12800 - L
    # no eligible crop -> explicit abstention, never a fallback crop
    quiet = x.copy()
    quiet[:a] = 0
    quiet[b:] = 0
    assert S.select_offtarget(att, ivs, quiet, heard, t, CFG)["status"] == "NO_MATCHED_OFFTARGET"
    assert S.select_offtarget(att, ivs, x, heard, {"status": "LOW_ENERGY"}, CFG)["status"] == "NO_TARGET"


def test_hard_mask_outside_bytes_positive_zero_heard_horizon():
    from csasr.inference_cf.lexical_compatibility import hard_mask
    x = np.random.default_rng(5).standard_normal(500000).astype(np.float32)
    xm = hard_mask(x, 1000, 5000)
    assert xm[:1000].tobytes() == x[:1000].tobytes() and xm[5000:].tobytes() == x[5000:].tobytes()
    assert np.all(xm[1000:5000].view(np.uint32) == 0)
    with pytest.raises(ValueError):
        hard_mask(x, 470000, 490000)                        # never mask beyond the 480000-sample heard horizon


def test_zero_change_audio_has_identical_features_and_mask_changes_features():
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(CFG["model"]["dir"], local_files_only=True)
    x = (np.random.default_rng(6).standard_normal(48000) * 0.05).astype(np.float32)
    f1 = fe([x], sampling_rate=16000, return_tensors="np", return_attention_mask=True)
    f2 = fe([x.copy()], sampling_rate=16000, return_tensors="np", return_attention_mask=True)
    assert np.array_equal(f1.input_features, f2.input_features) and np.array_equal(f1.attention_mask, f2.attention_mask)
    xm = x.copy()
    xm[16000:24000] = 0
    f3 = fe([xm], sampling_rate=16000, return_tensors="np", return_attention_mask=True)
    assert not np.array_equal(f1.input_features, f3.input_features)


# ---- scoring --------------------------------------------------------------------------------------------------------

def test_fixed_score_lambda_one_and_tie_ranking():
    l0 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    lm = np.array([-1.5, -1.0, -5.0], dtype=np.float32)
    s = S.scores(l0, lm)
    assert np.array_equal(s, 2 * l0.astype(np.float64) - lm.astype(np.float64))
    assert S.ranking([5, 9, 2], np.array([1.0, 1.0, 0.0])) == [5, 9, 2]
    assert S.ranking([9, 5, 2], np.array([1.0, 1.0, 3.0])) == [2, 5, 9]
    with pytest.raises(ValueError):
        S.scores(np.array([-np.inf]), np.array([0.0]))
    # a constant log-normalizer shift of the masked distribution cannot change the within-union ranking
    assert S.ranking([1, 2, 3], S.scores(l0, lm)) == S.ranking([1, 2, 3], S.scores(l0, lm + np.float32(4.0)))


def test_shuffle_seed_and_permutation_determinism():
    seed = int(hashlib.sha256(b"s1-shuffle-v1|UID|7|20").hexdigest()[:16], 16)
    assert S.shuffle_seed("UID", 7, 20) == seed
    v = np.arange(30, dtype=np.float64)
    a, b = S.shuffled(v, "UID", 7, 20), S.shuffled(v, "UID", 7, 20)
    assert np.array_equal(a, b) and sorted(a) == list(v)
    assert np.array_equal(a, v[np.random.Generator(np.random.PCG64(seed)).permutation(30)])
    assert not np.array_equal(S.shuffled(v, "UID", 7, 5), a)


# ---- AUTO, firewall, evaluator --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [50259, 50260, 50300])
def test_native_auto_prompt_helper_is_installed_semantics(d):
    from experiments.acoustic_s1 import auto_generation_config, native_auto_prompt
    g = auto_generation_config(CFG["model"]["dir"])
    assert (g.language, g.task, g.return_timestamps, g.forced_decoder_ids) == (None, "transcribe", False, None)
    assert native_auto_prompt(g, d) == [50258, d, 50360, 50364]


def test_runner_import_does_not_load_evaluator_or_reference_modules():
    code = ("import sys; sys.path[:0]=['src','.']; import experiments.acoustic_s1 as R; "
            "bad=[m for m in sys.modules if 'acoustic_s1_evaluate' in m or 'acoustic_s1_audit' in m or 'p2rj' in m]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "[]"


def test_static_firewall_and_auditor_independence():
    from experiments.acoustic_s1_audit import static_firewall
    c = static_firewall()
    assert all(c.values()), c
    src = (ROOT / "experiments/acoustic_s1_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(s1_evidence|acoustic_s1\b|acoustic_s1_evaluate|r0_regions)", src, re.M) is None
    assert "torch" not in (ROOT / "src/csasr/inference_cf/s1_evidence.py").read_text()      # provider: no model / autograd


def test_label_precedence():
    from experiments.acoustic_s1_evaluate import decide
    assert decide(False, True, True, True) == "S1_INVALID"
    assert decide(True, False, True, True) == "S1_CANDIDATE_HEADROOM_INSUFFICIENT"
    assert decide(True, True, False, True) == "S1_ACOUSTIC_DISCRIMINATION_INSUFFICIENT"
    assert decide(True, True, True, False) == "S1_PARTIAL_FEASIBILITY"
    assert decide(True, True, True, True) == "S1_READY_FOR_S2"


def test_dialogue_bootstrap_multiplicity_and_empty_draws():
    from experiments.acoustic_s1_evaluate import boot, draw_weights, macro
    dialogues = [f"D{i:02d}" for i in range(20)] * 3
    keys, W = draw_weights(dialogues, 2000, 240924)
    assert keys == sorted(set(dialogues)) and W.shape == (2000, 20) and np.all(W.sum(axis=1) == 20)
    vals = [("D00", 1.0), ("D00", 0.0), ("D01", 1.0)]
    est, dm = macro(vals)
    assert dm == {"D00": 0.5, "D01": 1.0} and est == 0.75
    b = boot(vals, keys, W, 0.003125, 0.996875)
    empty = int(((W[:, 0] + W[:, 1]) == 0).sum())
    assert b["finite_draws"] == 2000 - empty and empty > 0
    r = 0
    while W[r, 0] + W[r, 1] == 0:
        r += 1
    assert b["lower"] <= (W[r, 0] * 0.5 + W[r, 1] * 1.0) / (W[r, 0] + W[r, 1]) <= b["upper"]
    assert boot([], keys, W, 0.025, 0.975)["finite_draws"] == 0


def test_family_and_quantiles_from_frozen_config():
    b = CFG["bootstrap"]
    assert b["family_size"] == 8 and b["lower_quantile"] == 0.05 / 16 and b["upper_quantile"] == 1 - 0.05 / 16
    assert b["minimum_finite_draws"] == 9900
