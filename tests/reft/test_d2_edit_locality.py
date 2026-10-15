"""CPU tests for D2 (representation edit-locality): the Mandarin-positive token policy (ambiguous / unmapped / partial
UTF-8 / fillers / EOS), the decoder shift of the edit mask, the differentiable edit observer on the unchanged R2 adapter
(one call, no output change, exact energy of the applied edit, rank-2 identity, identity-init zeros, gradients to every
trainable tensor, no state leakage, mu = 0 bitwise parity), the edit-loss arithmetic, the frozen calibration rule and
stop checks, the same-prefix diagnostic helpers, and the pre-registered decision rules and verdict precedence."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.adapters import LoReFTAdapter  # noqa: E402
from reft.auto_policy import build_tf_batch  # noqa: E402
from reft.d2_edit_locality_evaluate import (RECOMMEND, components, decide, merge_stats, paired_tables, position_stats,  # noqa: E402
                                            prefix_cases, token_script, zhlike)
from reft.d2_edit_locality_train import (D_MODEL, FREEZE_SOURCES, EditObserver, calibration_checks, calibration_rows,  # noqa: E402
                                         cd, classify_zh_tokens, edit_energy, edit_loss, flat_cos, flat_norm, mu_rule,
                                         token_mask_tensor, training_order_counts, utterance_mask)
from reft.d1_regionkl_evaluate import KEYS, dialogue_table  # noqa: E402
from reft.text import units  # noqa: E402

FILL = {"嗯", "啊", "um", "uh", "yeah", "okay"}
EOS = 999


def fake_tb(pieces):
    def tb(i):
        return b"" if i == EOS else pieces[i]
    return tb


def tokens(chunks):
    pieces = [c if isinstance(c, bytes) else c.encode() for c in chunks]
    return list(range(len(pieces))) + [EOS], fake_tb(pieces)


# ---------------------------------------------------------------- Mandarin-positive token policy

def test_classify_zh_tokens_policy():
    lab = ["ZH", "ZH", "EN", "ZH", None, "EN"]
    um = [[0], [0, 1], [2], [1, 2], [3, 4], [4], [], [4, 5], [2, 5]]
    assert classify_zh_tokens(um, lab) == ["zh", "zh", "en", "ambiguous_language_boundary", "ambiguous_nonlexical", "other",
                                           "unmapped", "other", "en"]


def test_utterance_mask_all_mandarin_positions_and_switchback_subset():
    text = "我们use这个方法来做一个好的test然后结束"
    ids, tb = tokens(["我们", "use", "这个", "方法", "来做", "一个", "好的", "test", "然后", "结束"])
    r = utterance_mask(text, ids, tb, FILL, 7)
    assert r["mappable"] and r["zh_tokens"] == [0, 2, 3, 4, 5, 6, 8, 9] and r["en_tokens"] == [1, 7]
    assert r["zh_sb_tokens"] == [2, 3, 4, 8, 9] and set(r["zh_sb_tokens"]) <= set(r["zh_tokens"])
    assert r["ambiguous"] == [] and len(ids) - 1 not in r["zh_tokens"] and r["zh_units_without_token"] == 0


def test_utterance_mask_boundary_filler_partial_bytes():
    z = "这".encode()
    text = "我们use这个嗯方法"
    ids, tb = tokens(["我们", "us", b"e" + z[:1], z[1:], "个", "嗯方", "法"])
    r = utterance_mask(text, ids, tb, FILL, 7)
    assert r["mappable"] and r["zh_tokens"] == [0, 3, 4, 6] and r["en_tokens"] == [1]
    assert dict((t, c) for t, c in r["ambiguous"]) == {2: "ambiguous_language_boundary", 5: "ambiguous_nonlexical"}
    ids2, tb2 = tokens(["嗯", "我们"])
    assert utterance_mask("嗯我们", ids2, tb2, FILL, 7)["zh_tokens"] == [1]          # a filler-only token is never Mandarin-positive


def test_unmappable_and_english_only():
    ids, tb = tokens(["我们", "use", "这"])
    r = utterance_mask("我们use这个", ids, tb, FILL, 7)
    assert not r["mappable"] and r["zh_tokens"] == [] and r["zh_units_without_token"] == 4
    ids, tb = tokens(["hello", " world"])
    r = utterance_mask("hello world", ids, tb, FILL, 7)
    assert r["mappable"] and r["zh_tokens"] == [] and r["en_tokens"] == [0, 1]


def test_real_whisper_tokenizer_mandarin_tokens_are_han_only():
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
        r = utterance_mask(text, ids, tb, FILL, 7)
        assert r["mappable"] and r["zh_tokens"]
        got = b"".join(tb(ids[t]) for t in r["zh_tokens"]).decode("utf-8", errors="ignore")
        assert not any(c.isascii() and c.isalnum() for c in got)
        assert set(r["zh_sb_tokens"]) <= set(r["zh_tokens"]) and len(ids) - 1 not in r["zh_tokens"]
        en = b"".join(tb(ids[t]) for t in r["en_tokens"]).decode("utf-8", errors="ignore")
        assert not any("一" <= c <= "鿿" for c in en)


def test_token_mask_tensor_decoder_shift_and_eos():
    prompts = [[50258, 50260, 50359, 50363]] * 2
    tg = [[11, 12, 13, 14, EOS], [21, 22, EOS]]
    X, Y = build_tf_batch(prompts, tg, EOS, EOS)
    Z = token_mask_tensor(Y, tg, [[0, 3], [1]])
    assert Z.dtype == torch.bool and torch.nonzero(Z).tolist() == [[0, 3], [0, 6], [1, 4]]
    for i, ts in enumerate([[0, 3], [1]]):
        for t in ts:
            assert int(Y[i, 3 + t]) == tg[i][t] and int(X[i, 3 + t]) == (tg[i][t - 1] if t else prompts[i][-1])
    assert not Z[:, :3].any()
    with pytest.raises(ValueError):
        token_mask_tensor(Y, tg, [[4], []])                                       # EOS index
    with pytest.raises(ValueError):
        token_mask_tensor(Y, [[11, 12, 99, 14, EOS], tg[1]], [[2], []])           # wrong label / shift


# ---------------------------------------------------------------- edit observer on the unchanged R2 adapter (tiny decoder)

class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return (self.lin(x), None)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = _Attn(d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d)

    def forward(self, x, cache_position=None):
        h = x + self.encoder_attn(self.encoder_attn_layer_norm(x))[0]
        return h + self.fc(self.final_layer_norm(h))


class _Dec(nn.Module):
    def __init__(self, d, V=17):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(d), _Layer(d)])
        self.emb = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)


class _Model(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Module()
        self.model.decoder = _Dec(d)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, X):
        x = self.model.decoder.emb(X)
        cp = torch.arange(X.shape[1])
        for L in self.model.decoder.layers:
            x = L(x, cache_position=cp)
        return self.model.decoder.head(x)


def _setup(perturb=True, d=16):
    m = _Model(d)
    ad = LoReFTAdapter(d, 2, 1, seed=3, start_pos=3)
    if perturb:
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            ad.module.learned.weight.add_(torch.randn(ad.module.learned.weight.shape, generator=g) * 0.3)
            ad.module.learned.bias.add_(torch.randn(2, generator=g))
    X = torch.randint(0, 17, (3, 9), generator=torch.Generator().manual_seed(1))
    return m, ad, X


def _fwd(m, ad, X, obs=None):
    with ad.active(m):
        if obs is None:
            return m(X), None
        with obs.attached():
            c0 = ad.calls
            z = m(X)
            assert ad.calls - c0 == 1 and len(obs.records) == 1
            return z, obs.single()


def test_observer_single_call_no_output_change_and_exact_energy():
    m, ad, X = _setup()
    z0, _ = _fwd(m, ad, X)
    obs = EditObserver(ad, keep_input=True)
    z1, rec = _fwd(m, ad, X, obs)
    assert torch.equal(z0, z1)                                             # observing does not change the forward
    assert len(ad.module._forward_hooks) == 0 and obs.records == [] and obs.handle is None
    h = rec["h"]
    delta = ad.module(h) * (torch.arange(9) >= 3).float()[None, :, None]   # what the adapter adds (reft/adapters.py)
    assert torch.equal(rec["energy"], delta.pow(2).sum(-1))
    assert bool((rec["energy"][:, :3] == 0).all()) and bool((rec["energy"][:, 3:] > 0).all())
    assert rec["energy"].requires_grad and torch.equal(rec["pos"], torch.arange(9))


def test_rank2_identity_and_orthonormal_rows():
    m, ad, X = _setup()
    _, rec = _fwd(m, ad, X, EditObserver(ad, keep_input=True))
    M = ad.module
    a = nn.functional.linear(rec["h"].double(), (M.learned.weight - M.rot.weight).double()) + M.learned.bias.double()
    R = M.rot.weight.double()
    assert torch.allclose(R @ R.T, torch.eye(2, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(rec["energy"][:, 3:].double(), a.pow(2).sum(-1)[:, 3:], rtol=1e-4)
    assert torch.allclose((a @ R).pow(2).sum(-1), a.pow(2).sum(-1), rtol=1e-6)   # ||R^T a|| = ||a|| (float64)


def test_identity_init_gives_zero_energy_and_zero_gradient():
    m, ad, X = _setup(perturb=False)
    _, rec = _fwd(m, ad, X, EditObserver(ad))
    assert bool((rec["energy"] == 0).all())
    le, n = edit_loss(rec["energy"], torch.ones(3, 9, dtype=torch.bool), d=16)
    assert float(le) == 0.0 and n == 27
    gs = torch.autograd.grad(le, [p for p in ad.module.parameters() if p.requires_grad])
    assert all(bool((g == 0).all()) for g in gs)


def test_edit_gradient_reaches_every_trainable_tensor_and_stays_orthonormal():
    m, ad, X = _setup()
    _, rec = _fwd(m, ad, X, EditObserver(ad))
    Mz = torch.zeros(3, 9, dtype=torch.bool)
    Mz[:, 4:7] = True
    le, n = edit_loss(rec["energy"], Mz, d=16)
    assert float(le) > 0 and np.isfinite(float(le)) and n == 9
    le.backward()
    named = {k: p for k, p in ad.module.named_parameters() if p.requires_grad}
    assert set(named) == {"rot.parametrizations.weight.original", "learned.weight", "learned.bias"}
    assert all(p.grad is not None and float(p.grad.norm()) > 0 and torch.isfinite(p.grad).all() for p in named.values())
    assert all(p.grad is None for p in m.parameters())
    R0 = ad.module.rot.weight.detach().clone()
    opt = torch.optim.AdamW(ad.parameters(), lr=5e-4, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0)   # frozen recipe
    opt.step()
    R = ad.module.rot.weight.detach()
    assert not torch.equal(R, R0) and float((R @ R.T - torch.eye(2)).abs().max()) < 1e-5


def test_no_state_leakage_between_microbatches():
    m, ad, X = _setup()
    obs = EditObserver(ad)
    _, r1 = _fwd(m, ad, X, obs)
    _, r2 = _fwd(m, ad, X[:, :6], obs)
    _, r3 = _fwd(m, ad, X, obs)
    assert r2["energy"].shape == (3, 6) and torch.equal(r1["energy"], r3["energy"])
    with ad.active(m):
        m(X)                                                               # without the observer nothing is recorded
    assert obs.records == [] and len(ad.module._forward_hooks) == 0
    with pytest.raises(RuntimeError):
        obs.single()


def test_mu_zero_is_bitwise_the_observer_free_loss_and_gradient():
    m, ad, X = _setup()
    Y = torch.randint(0, 17, (3, 9), generator=torch.Generator().manual_seed(2))

    def base_loss(z):
        return nn.functional.cross_entropy(z.reshape(-1, 17).float(), Y.reshape(-1))
    z0, _ = _fwd(m, ad, X)
    l0 = base_loss(z0)
    g0 = torch.autograd.grad(l0, [p for p in ad.parameters()])
    z1, rec = _fwd(m, ad, X, EditObserver(ad))
    le, _ = edit_loss(rec["energy"], torch.ones(3, 9, dtype=torch.bool), d=16)
    l1 = base_loss(z1) + 0.0 * le
    g1 = torch.autograd.grad(l1, [p for p in ad.parameters()])
    assert torch.equal(l0, l1) and all(torch.equal(a, b) for a, b in zip(g0, g1))
    z2, rec2 = _fwd(m, ad, X, EditObserver(ad))
    l2 = base_loss(z2) + 0.5 * edit_loss(rec2["energy"], torch.ones(3, 9, dtype=torch.bool), d=16)[0]
    g2 = torch.autograd.grad(l2, [p for p in ad.parameters()])
    assert not all(torch.equal(a, b) for a, b in zip(g0, g2))


# ---------------------------------------------------------------- edit-loss arithmetic

def test_edit_loss_arithmetic_and_empty_mask():
    E = torch.rand(2, 7, dtype=torch.float32, generator=torch.Generator().manual_seed(4)).requires_grad_(True)
    Mz = torch.zeros(2, 7, dtype=torch.bool)
    Mz[0, 3] = Mz[1, 4] = Mz[1, 6] = True
    le, n = edit_loss(E, Mz)
    assert n == 3 and abs(float(le) - float((E[0, 3] + E[1, 4] + E[1, 6]) / D_MODEL / 3)) < 1e-9
    g = torch.autograd.grad(le, E)[0]
    assert torch.equal(g != 0, Mz)
    l0, n0 = edit_loss(E, torch.zeros_like(Mz))
    assert float(l0) == 0.0 and n0 == 0 and l0.requires_grad
    assert torch.equal(edit_energy(torch.ones(1, 5, 4), torch.arange(5)), torch.tensor([[0.0, 0, 0, 4, 4]]))


# ---------------------------------------------------------------- calibration rule and stop checks

S = {"reproducibility_rtol": 1e-5, "min_edit_grad_norm": 1e-12, "max_ratio_spread": 10.0, "max_scaled_ratio": 1.0,
     "min_mask_positions_per_batch": 1}


def test_calibration_rows_hash_order():
    uids = [f"u{i}" for i in range(100)]
    a = calibration_rows(uids, 8)
    assert a == calibration_rows(list(reversed(uids)), 8) and len(a) == 8 and all(len(x) == 8 for x in a)
    assert len({u for x in a for u in x}) == 64
    with pytest.raises(ValueError):
        calibration_rows(uids[:10], 8)


def test_mu_rule_and_checks():
    gb, ge = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0], [0.1, 0.2, 0.2, 0.2, 0.3, 0.2]
    mu = mu_rule(gb, ge, 0.2)
    assert mu == pytest.approx(0.2 * 2.0 / 0.2)
    c = calibration_checks(gb, ge, [5] * 6, mu, mu, S)
    assert c["ok"] and c["ratio_spread"] == pytest.approx(15 / 6.6666666667, rel=1e-6)
    assert max(c["scaled_ratios"]) <= 1.0
    assert not calibration_checks(gb, ge, [5] * 6, mu, mu * 1.01, S)["reproducible"]
    assert "edit_grad_nonzero" in calibration_checks(gb, [0.1, 0.0, 0.2, 0.2, 0.3, 0.2], [5] * 6, mu, mu, S)["failed"]
    assert "mask_positions_every_batch" in calibration_checks(gb, ge, [5, 0, 5, 5, 5, 5], mu, mu, S)["failed"]
    spread = calibration_checks([1.0, 1.0, 1.0, 1.0, 1.0, 50.0], [1.0] * 6, [5] * 6, 0.2, 0.2, S)
    assert not spread["ratio_spread_ok"] and not spread["ok"]
    dom = calibration_checks([1.0] * 6, [1.0] * 6, [5] * 6, 1.5, 1.5, S)
    assert not dom["no_pathological_dominance"]
    nan = calibration_checks([1.0, float("nan")] + [1.0] * 4, [1.0] * 6, [5] * 6, 0.2, 0.2, S)
    assert not nan["finite"] and not nan["ok"]


def test_training_order_counts_follow_p2_1b_orders():
    tm = [{"utterance_id": f"u{i}"} for i in range(20)]
    per = {f"u{i}": (1 if i < 10 else 0) for i in range(20)}
    res = training_order_counts(tm, per)
    assert set(res) == {f"{s}_{e}" for s in (42, 43, 44) for e in (1, 2, 3)}
    order = np.random.default_rng(42 + 1).permutation(20)
    cnt = [sum(per[tm[i]["utterance_id"]] for i in order[j:j + 8]) for j in (0, 8, 16)]
    assert res["42_1"] == {"min": min(cnt), "median": float(np.median(cnt)), "max": max(cnt), "zero_batches": cnt.count(0), "n_batches": 3}
    assert all(v["n_batches"] == 3 and v["min"] >= 0 for v in res.values())
    assert sum(1 for v in training_order_counts(tm, {}).values() if v["zero_batches"] == 3) == 9


def test_flat_norm_and_cos():
    a = [torch.tensor([3.0, 0.0]), torch.tensor([[4.0]])]
    assert flat_norm(a) == pytest.approx(5.0)
    assert flat_cos(a, a) == pytest.approx(1.0) and flat_cos(a, [-x for x in a]) == pytest.approx(-1.0)


# ---------------------------------------------------------------- same-prefix diagnostic helpers

def test_token_script():
    assert token_script("这个".encode(), False) == "han" and token_script(" model".encode(), False) == "latin"
    assert token_script("这".encode()[:2], False) == "partial" and token_script(b"", True) == "eos"
    assert token_script(b" 3", False) == "other" and zhlike("partial") and zhlike("han") and not zhlike("latin")


def test_position_stats_and_merge():
    lp_off = torch.log_softmax(torch.tensor([[0.0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]), -1)
    lp_on = torch.log_softmax(torch.tensor([[0.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 5, 0], [0, 5, 0], [0, 0, 5]]), -1)
    cls = {0: "han", 1: "latin", 2: "eos"}
    c_off = ["other"] * 3 + [cls[i] for i in lp_off.argmax(-1).tolist()[3:]]
    c_on = ["other"] * 3 + [cls[i] for i in lp_on.argmax(-1).tolist()[3:]]
    st = position_stats(lp_off, lp_on, c_off, c_on)
    assert st["all"]["n"] == 3 and st["all"]["top1_change"] == 1 and st["zhlike"]["zh_to_nonzh"] == 1
    assert st["zhlike"]["n"] == 1 and st["nonzh"]["kl"] == pytest.approx(0.0, abs=1e-6) and st["zhlike"]["kl"] > 0
    mg = merge_stats([st, st])
    assert mg["all"]["n"] == 6 and mg["zhlike"]["zh_to_nonzh_rate"] == 1.0 and mg["all"]["top1_change_rate"] == pytest.approx(1 / 3)


def test_prefix_cases_fixed_order():
    u = [f"x{i}" for i in range(50)]
    assert prefix_cases(u, 40) == prefix_cases(list(reversed(u)), 40) and len(prefix_cases(u, 40)) == 40


# ---------------------------------------------------------------- decision rules

D = {"preservation_margin_units": 5, "seed_consistency_min": 2, "en_net_sum_ratio_min": 0.90, "strong_zh_new_max": 49,
     "component_margin_units": 5}


def _arm(zh, en, mer, tw=(30, 30, 30), sb=(30, 30, 30)):
    return {s: {"zh_new": z, "en_net": e, "cs_mer": m, "tw_new": t, "sb": b} for s, z, e, m, t, b in zip((42, 43, 44), zh, en, mer, tw, sb)}


GK = _arm((53, 48, 50), (34, 41, 35), (0.1124, 0.1105, 0.1103))


def test_verdict_precedence_and_recommendations():
    good = components({"D2": _arm((45, 43, 44), (34, 40, 35), (0.1110, 0.1100, 0.1100)), "GoldKL": GK}, D)
    assert decide(True, good["A"], good["B"], good["C"], good["components"])["verdict"] == "D2_INCREMENTAL_PRESERVATION_GAIN"
    assert decide(False, good["A"], good["B"], good["C"], good["components"])["verdict"] == "D2_INVALID"
    weak_en = components({"D2": _arm((45, 43, 44), (20, 20, 20), (0.1110, 0.1100, 0.1100)), "GoldKL": GK}, D)
    assert decide(True, weak_en["A"], weak_en["B"], weak_en["C"], weak_en["components"])["verdict"] == "D2_MIXED_TRADEOFF"
    worse_mer = components({"D2": _arm((45, 43, 44), (34, 40, 35), (0.1130, 0.1110, 0.1110)), "GoldKL": GK}, D)
    assert decide(True, worse_mer["A"], worse_mer["B"], worse_mer["C"], worse_mer["components"])["verdict"] == "D2_MIXED_TRADEOFF"
    comp = components({"D2": _arm((53, 48, 50), (34, 41, 35), (0.1124, 0.1105, 0.1103), tw=(20, 20, 30)), "GoldKL": GK}, D)
    assert decide(True, comp["A"], comp["B"], comp["C"], comp["components"])["verdict"] == "D2_MIXED_TRADEOFF"
    none = components({"D2": _arm((53, 49, 39), (35, 37, 37), (0.1127, 0.1105, 0.1105)), "GoldKL": GK}, D)
    v = decide(True, none["A"], none["B"], none["C"], none["components"])
    assert v["verdict"] == "D2_NO_INCREMENTAL_GAIN" and v["recommendation"] == "STOP_GOLD_PREFIX_LOCALITY_OBJECTIVE_EXPANSION"
    border = components({"D2": _arm((48, 43, 45), (34, 41, 35), (0.1124, 0.1105, 0.1103)), "GoldKL": GK}, D)
    assert border["A"]["pass"] and border["C"]["pass"]                     # exactly -5 and equal MER pass (<=)
    assert set(RECOMMEND) == {"D2_INVALID", "D2_INCREMENTAL_PRESERVATION_GAIN", "D2_MIXED_TRADEOFF", "D2_NO_INCREMENTAL_GAIN"}


def test_paired_tables_d2():
    def pu(d, z, tw):
        return {"dialogue": d, "stratum": "CS", "s_err": 10, "units": 100, "zh_new": z, "sb": 1, "en_corr": 3, "en_corrupt": 1, "b_err": 11,
                "tw_new": tw, "tw_b0_correct": 5}
    tabs = {}
    for s in (42, 43, 44):
        tabs[("GoldKL", s)] = dialogue_table([pu("a", 5, 2), pu("b", 3, 1)])
        tabs[("D2", s)] = dialogue_table([pu("a", 4, 1), pu("b", 3, 0)])
    rng = np.random.default_rng(42)
    pr = paired_tables(tabs, ["a", "b"], [rng.choice(2, 2) for _ in range(50)])
    assert pr["seed_mean"]["zh_new"]["point"] == -1 and pr["seed_mean"]["tw_new"]["point"] == -2
    assert set(pr["42"]) == set(KEYS) | {"dialogue_contribution"}


# ---------------------------------------------------------------- frozen configuration and sources

def test_frozen_config_and_sources():
    C = cd()
    assert C["objective"]["lambda_kl"] == 1.0 and C["calibration"]["rho"] == 0.20 and C["calibration"]["n_batches"] >= 6
    assert C["seeds"] == [42, 43, 44] and C["max_new_training_runs"] == 3 and C["method"]["expected_trainable_parameters"] == 5122
    assert C["decision"] == {**D, "cs_mer_strict": True} and C["bootstrap"] == {"unit": "dialogue", "reps": 2000, "seed": 42}
    assert C["calibration"]["stop"] == S and C["evaluation"]["K"] == 7
    assert units("我们use") == ["我", "们", "use"]
    for s in FREEZE_SOURCES:
        assert (ROOT / s).exists(), s
