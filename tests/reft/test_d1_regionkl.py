"""CPU tests for D1 (region-weighted GoldKL): switch-back window construction and the frozen edge-case policy, token
classification (ambiguous / unmapped / partial UTF-8), the decoder shift of the weight tensor, exact equivalence of the
all-ones path with P2-1b GoldKL (loss and gradients, bitwise), weighting arithmetic, empty masks, the target-window
evaluation metric, the pre-registered decision rules and verdict precedence."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.auto_policy import build_tf_batch  # noqa: E402
from reft.d1_regionkl_evaluate import (KEYS, RECOMMEND, component_gain, decide, dialogue_table, en_guard, mer_guard,  # noqa: E402
                                       paired_tables, primary_gain, principal_cosines, strong, tw_from_events, tw_support)
from reft.d1_regionkl_train import (FREEZE_SOURCES, batch_region_counts, cd, classify_tokens, effective_factors,  # noqa: E402
                                    lang_labels, region_kl, switchback_windows, utterance_region, weight_tensor)
from reft.p2_1b_train import kl_masked, teacher_mask  # noqa: E402
from reft.text import units  # noqa: E402

FILL = {"嗯", "啊", "um", "uh", "yeah", "okay"}
EOS = 999


def fake_tb(pieces):
    """Token id -> bytes for a list of byte pieces; ids are list indices; EOS -> b''."""
    def tb(i):
        return b"" if i == EOS else pieces[i]
    return tb


def tokens(chunks):
    pieces = [c if isinstance(c, bytes) else c.encode() for c in chunks]
    return list(range(len(pieces))) + [EOS], fake_tb(pieces)


# ---------------------------------------------------------------- windows

def test_windows_simple_short_long_multiple_and_none():
    L = ["ZH", "ZH", "EN", "ZH", "ZH", "ZH", "ZH", "ZH", "ZH", "ZH", "ZH", "ZH", "EN", "ZH", "ZH"]
    assert switchback_windows(L, 7) == [[3, 4, 5, 6, 7, 8, 9], [13, 14]]  # > 7 truncated; short run kept
    assert switchback_windows(["ZH", "ZH", "EN", "EN"], 7) == []            # no switch-back
    assert switchback_windows(["ZH", "ZH"], 7) == [] and switchback_windows([], 7) == []
    assert switchback_windows(["EN", "ZH", "EN", "ZH", "ZH"], 7) == [[1], [3, 4]]


def test_windows_skip_fillers_other_and_mixed_units():
    # None = filler / OTHER / UNKNOWN: skipped, neither counted nor terminating; EN filler between EN and ZH keeps the switch
    L = ["EN", None, "ZH", None, "ZH", "ZH", None, "ZH", "ZH", "ZH", "ZH", "ZH"]
    assert switchback_windows(L, 7) == [[2, 4, 5, 7, 8, 9, 10]]
    assert switchback_windows(["ZH", None, "ZH"], 7) == []


def test_lang_labels_use_canonical_tags_and_fillers():
    assert lang_labels(["我", "use", "嗯", "yeah", "3", "啊"], FILL) == ["ZH", "EN", None, None, None, None]


# ---------------------------------------------------------------- token classification

def test_classify_tokens_policy():
    wins = [[3, 4, 5]]
    lab = ["ZH", "ZH", "EN", "ZH", "ZH", "ZH", "ZH", None]
    um = [[0, 1], [2], [3], [3], [4, 5], [5, 6], [2, 3], [7], [], [5, 7]]
    assert classify_tokens(um, wins, lab) == ["outside", "outside", "region", "region", "region", "ambiguous_window_end",
                                              "ambiguous_language_boundary", "outside", "unmapped", "ambiguous_nonlexical"]
    assert classify_tokens([[1, 4]], [[1], [4]], ["ZH"] * 5) == ["ambiguous_multi_window"]


def test_utterance_region_window_end_and_second_switch():
    text = "我们use这个方法来做一个好的test然后结束"
    U = units(text)
    assert U[2] == "use" and U[13] == "test" and len(U) == 18
    ids, tb = tokens(["我们", "use", "这个", "方法", "来做", "一个", "好的", "test", "然后", "结束"])
    r = utterance_region(text, ids, tb, FILL, 7)
    assert r["mappable"] and r["windows"] == [[3, 4, 5, 6, 7, 8, 9], [14, 15, 16, 17]]
    assert r["region_tokens"] == [2, 3, 4, 8, 9]
    assert r["ambiguous"] == [[5, "ambiguous_window_end"]]          # 一个 spans the 7th and 8th Mandarin unit
    assert r["units_without_region_token"] == 1                      # unit 9 (一) only reachable through the ambiguous token
    assert r["n_switchbacks"] == 2 and r["window_sizes"] == [7, 4]
    assert [f["t"] for f in r["first_sb_tokens"]] == [2, 8] and all(f["unambiguous"] for f in r["first_sb_tokens"])
    assert len(ids) - 1 not in r["region_tokens"]                     # EOS never in the region


def test_utterance_region_language_boundary_filler_and_partial_bytes():
    z = "这".encode()
    text = "我们use这个嗯方法"
    ids, tb = tokens(["我们", "us", b"e" + z[:1], z[1:], "个", "嗯方", "法"])
    r = utterance_region(text, ids, tb, FILL, 7)
    assert r["mappable"]
    cls = dict((t, c) for t, c in r["ambiguous"])
    assert cls == {2: "ambiguous_language_boundary", 5: "ambiguous_nonlexical"}
    assert r["region_tokens"] == [3, 4, 6]                            # partial UTF-8 bytes of 这 map to the window unit
    ids2, tb2 = tokens(["我们", "use", z[:2], z[2:], "个"])
    r2 = utterance_region("我们use这个", ids2, tb2, FILL, 7)
    assert r2["region_tokens"] == [2, 3, 4]


def test_unmappable_and_no_switch_utterances_get_no_weights():
    ids, tb = tokens(["我们", "use", "这"])                            # bytes do not reconstruct the text
    r = utterance_region("我们use这个", ids, tb, FILL, 7)
    assert not r["mappable"] and r["region_tokens"] == []
    ids, tb = tokens(["我们", "今天"])
    r = utterance_region("我们今天", ids, tb, FILL, 7)
    assert r["mappable"] and r["windows"] == [] and r["region_tokens"] == []
    ids, tb = tokens(["你好", "hello"])
    assert utterance_region("你好hello", ids, tb, FILL, 7)["region_tokens"] == []


def test_real_whisper_tokenizer_regions_are_han_only_and_match_windows():
    from reft.common import cfg
    p = Path(cfg()["model"]["path"])
    if not p.exists():
        pytest.skip("model files not available")
    from transformers import WhisperTokenizer
    from reft.text import tok_bytes_fn
    tok = WhisperTokenizer.from_pretrained(str(p), local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    for text in ("我们今天要discuss这个project的进度然后再说吧", "所以这个model其实还蛮好用的对吧", "OK那我们就这样做"):
        ids = tok.encode(text, add_special_tokens=False) + [eos]
        r = utterance_region(text, ids, tb, FILL, 7)
        assert r["mappable"]
        U = units(text)
        win_text = ["".join(U[k] for k in w) for w in r["windows"]]
        got = b"".join(tb(ids[t]) for t in r["region_tokens"]).decode("utf-8", errors="ignore")
        assert not any(c.isascii() and c.isalnum() for c in got)
        # the region bytes are a (possibly ambiguity-trimmed) subsequence of the window characters
        assert set(got) <= set("".join(win_text)), (got, win_text)
    r = utterance_region("我们今天要discuss这个project的进度然后再说吧", tok.encode("我们今天要discuss这个project的进度然后再说吧",
                                                                                add_special_tokens=False) + [eos], tb, FILL, 7)
    U = units("我们今天要discuss这个project的进度然后再说吧")
    assert ["".join(U[k] for k in w) for w in r["windows"]] == ["这个", "的进度然后再说"]


# ---------------------------------------------------------------- weight tensor / decoder shift

def test_weight_tensor_decoder_shift_and_eos():
    prompts = [[50258, 50260, 50359, 50363]] * 2
    tg = [[11, 12, 13, 14, EOS], [21, 22, EOS]]
    X, Y = build_tf_batch(prompts, tg, EOS, EOS)
    W = weight_tensor(Y, tg, [[1, 3], [0]], 2.0)
    assert W.dtype == torch.float32 and not W.requires_grad
    assert torch.nonzero(W > 1).tolist() == [[0, 4], [0, 6], [1, 3]]
    for i, ts in enumerate([[1, 3], [0]]):
        for t in ts:
            assert int(Y[i, 3 + t]) == tg[i][t]                                     # label predicted at position 3 + t
            assert int(X[i, 3 + t]) == (tg[i][t - 1] if t else prompts[i][-1])      # input at that position = previous token
    assert int(W[0, 3 + 4]) == 1 and int(W[1, 3 + 2]) == 1                          # EOS positions
    assert bool((W[:, :3] == 1).all())
    with pytest.raises(ValueError):
        weight_tensor(Y, tg, [[4], []], 2.0)                                        # the EOS index is never a region token
    with pytest.raises(ValueError):
        weight_tensor(Y, [[11, 12, 99, 14, EOS], tg[1]], [[2], []], 2.0)            # off-by-one / wrong label detected


def test_batch_region_counts():
    assert batch_region_counts(["a", "b", "c"], {"a": [1, 2], "c": [0]}, 2) == [2, 1]


# ---------------------------------------------------------------- region-weighted KL

def _logits(seed=0, B=3, T=9, V=40):
    g = torch.Generator().manual_seed(seed)
    zt = torch.randn(B, T, V, generator=g) * 2
    zs = (zt + torch.randn(B, T, V, generator=g) * 0.3).requires_grad_(True)
    Y = torch.randint(0, V, (B, T), generator=g)
    Y[:, :3] = -100
    Y[:, ::4] = zt.argmax(-1)[:, ::4]
    Y[:, 5:7] = zt.argmax(-1)[:, 5:7]
    Y[2, 7:] = -100
    Y[:, :3] = -100
    return zt, zs, Y


def test_all_ones_is_bitwise_goldkl_loss_and_gradient():
    zt, zs, Y = _logits()
    l0, n0 = kl_masked(zt, zs, Y)
    g0 = torch.autograd.grad(l0, zs)[0]
    l1, d1 = region_kl(zt, zs, Y, torch.ones(Y.shape))
    g1 = torch.autograd.grad(l1, zs)[0]
    assert torch.equal(l0, l1) and torch.equal(g0, g1) and n0 == d1["n_mask"] == d1["den"]
    assert d1["n_region_mask"] == 0 and d1["kl_unweighted"] == float(l0)


def test_weighted_arithmetic_and_factors():
    zt, zs, Y = _logits(1)
    m = teacher_mask(zt, Y)
    W = torch.ones(Y.shape)
    W[0, 4] = W[1, 5] = W[1, 6] = 2.0
    l, d = region_kl(zt, zs, Y, W)
    lp0 = torch.log_softmax(zt.double(), -1)[m]
    kl = (lp0.exp() * (lp0 - torch.log_softmax(zs.detach().double(), -1)[m])).sum(-1)
    w = W.double()[m]
    assert abs(float((kl * w).sum() / w.sum()) - float(l)) < 1e-6
    k = int((W[m] > 1).sum())
    assert d["n_region_mask"] == k and d["den"] == d["n_mask"] + k
    assert abs(d["region_part"] + d["other_part"] - float(l)) < 1e-6
    f = effective_factors(d["n_mask"], k, 2.0)
    assert abs(f["region"] - 2 * d["n_mask"] / (d["n_mask"] + k)) < 1e-12 and f["region"] > 1 > f["other"]
    g_w = torch.autograd.grad(l, zs)[0]
    g_u = torch.autograd.grad(region_kl(zt, zs, Y, torch.ones(Y.shape))[0], zs)[0]
    assert not torch.equal(g_w, g_u)
    assert d["kl_min"] >= -1e-6


def test_teacher_incorrect_and_unlabelled_weights_are_ignored_and_empty_mask():
    zt, zs, Y = _logits(2)
    m = teacher_mask(zt, Y)
    W = torch.ones(Y.shape)
    W[~m] = 7.0                                   # weights outside the mask never matter
    assert torch.equal(region_kl(zt, zs, Y, W)[0], region_kl(zt, zs, Y, torch.ones(Y.shape))[0])
    Ye = torch.full_like(Y, -100)
    le, de = region_kl(zt, zs, Ye, torch.ones(Y.shape))
    assert float(le) == 0.0 and le.requires_grad and de["n_mask"] == 0
    assert torch.equal(torch.autograd.grad(le, zs)[0], torch.zeros_like(zs))


def test_teacher_logits_detached_inside_region_kl():
    zt, zs, Y = _logits(3)
    zt = zt.clone().requires_grad_(True)
    l, _ = region_kl(zt, zs, Y, torch.ones(Y.shape))
    g = torch.autograd.grad(l, [zt, zs], allow_unused=True)
    assert g[0] is None and g[1] is not None


# ---------------------------------------------------------------- target-window evaluation metric

def _ev(spec):
    """spec: list of (label, b0_ok, sys_ok, sys_op); label in EN / ZH / F (filler)."""
    out = []
    for k, (lab, bo, so, op) in enumerate(spec):
        tr = {(False, True): "repair", (True, False): "corruption", (True, True): "unchanged_correct", (False, False): "unchanged_wrong"}[(bo, so)]
        out.append({"k": k, "substantive_en": lab == "EN", "zh_non_filler": lab == "ZH", "b0_ok": bo, "sys_ok": so, "sys_op": op, "transition": tr})
    return out


def test_tw_from_events_counts_and_window_rule():
    spec = [("ZH", True, True, "match"), ("EN", True, True, "match"), ("F", True, True, "match")] + \
           [("ZH", True, False, "del")] * 3 + [("ZH", True, True, "match")] * 3 + [("ZH", False, True, "match"), ("ZH", True, False, "sub")] + \
           [("EN", True, True, "match"), ("ZH", True, False, "sub")]
    r = tw_from_events(_ev(spec), 7)
    # window 1 = units 3..9 (7 ZH units after the filler; the 8th Mandarin unit's substitution at 10 is outside), window 2 = unit 12
    assert r["windows"] == 2 and r["eligible"] == 8 and r["b0_correct"] == 7
    assert r["new"] == 4 and r["new_del"] == 3 and r["new_sub"] == 1 and r["repairs"] == 1


# ---------------------------------------------------------------- decision rules

D = {"preservation_margin_units": 5, "seed_consistency_min": 2, "en_net_sum_ratio_min": 0.90, "strong_zh_new_max": 49,
     "component_margin_units": 5}


def arm(zh, en, mer, tw=(20, 20, 20), sb=(30, 30, 30)):
    return {s: {"zh_new": z, "en_net": e, "cs_mer": m, "tw_new": t, "sb": b} for s, z, e, m, t, b in zip((42, 43, 44), zh, en, mer, tw, sb)}


GK = arm((53, 48, 50), (34, 41, 35), (0.1124, 0.1105, 0.1103), (20, 18, 19), (30, 31, 26))


def test_primary_gain():
    a = primary_gain(arm((44, 42, 50), (34, 41, 35), (0.11,) * 3), GK, D)                     # -5.0, 2 seeds lower (tie in 44)
    assert a["A1_mean_margin"] and a["A2_seed_consistent"] and a["pass"]
    a = primary_gain(arm((30, 50, 56), (34, 41, 35), (0.11,) * 3), GK, D)                     # mean -5 driven by one seed
    assert a["A1_mean_margin"] and not a["A2_seed_consistent"] and not a["pass"]


def test_primary_gain_margin_boundary():
    a = primary_gain(arm((45, 42, 52), (34, 41, 35), (0.11,) * 3), GK, D)
    assert a["d_zh_new_mean"] == pytest.approx(-4.0) and not a["A1_mean_margin"] and not a["pass"]


def test_en_and_mer_guards():
    assert en_guard(arm((0,) * 3, (33, 33, 33), (0,) * 3), GK, D)["pass"]                     # 99 / 110 = 0.9
    assert not en_guard(arm((0,) * 3, (33, 33, 32), (0,) * 3), GK, D)["pass"]                 # 98 / 110
    assert not en_guard(GK, arm((0,) * 3, (0, -1, 0), (0,) * 3), D)["pass"]                    # undefined denominator -> fail
    assert mer_guard(arm((0,) * 3, (0,) * 3, (0.1124, 0.1105, 0.1103)), GK)["pass"]           # equal passes
    assert not mer_guard(arm((0,) * 3, (0,) * 3, (0.1124, 0.1105, 0.11031)), GK)["pass"]      # strict, no tolerance


def test_components_support_strong():
    x = arm((50,) * 3, (35,) * 3, (0.11,) * 3, tw=(12, 10, 18), sb=(30, 31, 26))
    assert component_gain(x, GK, "tw_new", D)["pass"] and not component_gain(x, GK, "sb", D)["pass"]
    assert tw_support(x, GK, D)["supported"]
    assert not tw_support(arm((50,) * 3, (35,) * 3, (0.11,) * 3, tw=(19, 25, 25)), GK, D)["supported"]
    assert strong(arm((49, 50, 10), (0,) * 3, (0,) * 3), D) == {"42": True, "43": False, "44": True}


def test_verdict_precedence():
    P, F = {"pass": True}, {"pass": False}
    nc = {"tw_new": F, "sb": F}
    assert decide(False, P, P, P, nc)["verdict"] == "D1_INVALID"
    assert decide(True, P, P, P, nc)["verdict"] == "D1_INCREMENTAL_PRESERVATION_GAIN"
    assert decide(True, P, F, P, nc)["verdict"] == "D1_MIXED_TRADEOFF"
    assert decide(True, P, P, F, nc)["verdict"] == "D1_MIXED_TRADEOFF"
    assert decide(True, F, P, P, {"tw_new": P, "sb": F})["verdict"] == "D1_MIXED_TRADEOFF"
    assert decide(True, F, P, P, nc)["verdict"] == "D1_NO_INCREMENTAL_GAIN"
    assert decide(True, F, F, F, nc)["recommendation"] == "STOP_FIXED_SWITCHBACK_WEIGHTING"
    assert set(RECOMMEND) == {"D1_INVALID", "D1_INCREMENTAL_PRESERVATION_GAIN", "D1_MIXED_TRADEOFF", "D1_NO_INCREMENTAL_GAIN"}


def test_dialogue_and_paired_tables():
    def pu(d, z, tw):
        return {"dialogue": d, "stratum": "CS", "s_err": 10, "units": 100, "zh_new": z, "sb": 1, "en_corr": 3, "en_corrupt": 1, "b_err": 11,
                "tw_new": tw, "tw_b0_correct": 5}
    tabs = {}
    for s in (42, 43, 44):
        tabs[("GoldKL", s)] = dialogue_table([pu("a", 5, 2), pu("b", 3, 1)])
        tabs[("D1", s)] = dialogue_table([pu("a", 4, 1), pu("b", 3, 0)])
    dl = ["a", "b"]
    rng = np.random.default_rng(42)
    pr = paired_tables(tabs, dl, [rng.choice(2, 2) for _ in range(50)])
    assert pr["seed_mean"]["zh_new"]["point"] == -1 and pr["seed_mean"]["tw_new"]["point"] == -2
    assert set(pr["42"]) == set(KEYS) | {"dialogue_contribution"}


def test_principal_cosines():
    I = np.eye(4)[:2]
    assert principal_cosines(I, I) == pytest.approx([1.0, 1.0])
    assert principal_cosines(I, np.eye(4)[2:]) == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------- frozen configuration and sources

def test_frozen_config_and_sources():
    C = cd()
    assert C["objective"]["region_weight"] == 2.0 and C["objective"]["lambda_kl"] == 1.0 and C["region"]["max_units_per_switch"] == 7
    assert C["seeds"] == [42, 43, 44] and C["max_new_training_runs"] == 3
    assert C["decision"] == {**D, "cs_mer_strict": True}
    assert C["coverage_gate"]["min_teacher_correct_region_coverage"] == 0.10
    for s in FREEZE_SOURCES:
        assert (ROOT / s).exists(), s
