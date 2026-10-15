"""CPU tests for S0/S1 (R3 vs M1 under GoldKL): parameter counts and exact initialization at d = 1280; hook semantics,
zero-update identity, M1 = R2 equivalence with W_u = 0, single application at the exact site, absolute positions in
cached decoding and gradient flow on a tiny random Whisper (real transformers classes); contribution decomposition;
paired dialogue statistics; the pre-registered decision rules and verdict precedence."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.adapters import LoReFTAdapter, site_hook_count  # noqa: E402
from reft.arch_s0s1_adapters import (EXPECTED, R2_KEYS, CondLoReFTAdapter, named_trainable, orth_error, perturb,  # noqa: E402
                                     transplant_r2_into_m1)
from reft.arch_s0s1_evaluate import (RECOMMEND, contributions, corrections, decide, dialogue_table, gain, guard,  # noqa: E402
                                     improvement_over_r2, loops_guard, paired_tables, strong, tab_stat, token_lang)
from reft.site import Site  # noqa: E402

LAYER, D = 2, 16


# ---------------------------------------------------------------- full-size modules (d = 1280)

def test_parameter_counts_and_exact_init_full_size():
    for cls, r, arch in ((LoReFTAdapter, 2, "R2"), (LoReFTAdapter, 3, "R3"), (CondLoReFTAdapter, 2, "M1")):
        torch.manual_seed(42)
        ad = cls(1280, r, 31, 42, 3)
        assert ad.n_trainable() == EXPECTED[arch]
        nt = named_trainable(ad)
        exp = {"rot.parametrizations.weight.original": (r, 1280), "learned.weight": (r, 1280), "learned.bias": (r,)}
        if arch == "M1":
            exp["w_u"] = (r, 1280)
        assert {n: tuple(p.shape) for n, p in nt.items()} == exp
        m = ad.module
        assert torch.equal(m.learned.weight, m.rot.weight) and bool((m.learned.bias == 0).all())
        assert orth_error(ad) < 1e-5
        # buffers are not parameters: the orthogonal-parametrization base is excluded from the count
        assert sum(p.numel() for p in m.parameters() if p.requires_grad) == EXPECTED[arch]
        assert any(k.endswith(".base") for k in ad.state_dict())
        if arch == "M1":
            assert bool((m.w_u == 0).all())
    assert EXPECTED == {"R2": 5122, "R3": 7683, "M1": 7682}


def test_m1_init_equals_r2_init_under_same_global_seed():
    torch.manual_seed(7)
    r2 = LoReFTAdapter(64, 2, LAYER, 43, 3)
    torch.manual_seed(7)
    m1 = CondLoReFTAdapter(64, 2, LAYER, 43, 3)
    s2, s1 = r2.state_dict(), m1.state_dict()
    assert set(s1) == set(R2_KEYS) | {"w_u"} and set(s2) == set(R2_KEYS)
    assert all(torch.equal(s2[k], s1[k]) for k in R2_KEYS)


def test_transplant_requires_exactly_the_r2_fields():
    torch.manual_seed(1)
    r2 = LoReFTAdapter(32, 2, LAYER, 1, 3)
    perturb(r2, 0.1, 0.1, 3)
    m1 = CondLoReFTAdapter(32, 2, LAYER, 9, 3)
    with torch.no_grad():
        m1.module.w_u.fill_(1.0)
    transplant_r2_into_m1(m1, r2.state_dict())
    assert bool((m1.module.w_u == 0).all())
    assert all(torch.equal(r2.state_dict()[k], m1.state_dict()[k]) for k in R2_KEYS)
    with pytest.raises(AssertionError):
        transplant_r2_into_m1(m1, {**r2.state_dict(), "extra": torch.zeros(1)})


# ---------------------------------------------------------------- tiny Whisper

@pytest.fixture(scope="module")
def tiny():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    c = WhisperConfig(vocab_size=64, num_mel_bins=8, encoder_layers=1, decoder_layers=3, encoder_attention_heads=2,
                      decoder_attention_heads=2, d_model=D, encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=12,
                      max_target_positions=32, pad_token_id=1, decoder_start_token_id=2, eos_token_id=1, attn_implementation="eager")
    m = WhisperForConditionalGeneration(c).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    enc = torch.randn(2, 12, D)
    X = torch.tensor([[2, 5, 6, 7, 9, 11, 13, 4], [2, 8, 6, 3, 12, 10, 14, 15]])
    return m, enc, X


def fwd(m, enc, X, ad=None, grad=False):
    import contextlib
    ctx = ad.active(m) if ad is not None else contextlib.nullcontext()
    with torch.set_grad_enabled(grad), ctx:
        return m(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits


def record(m, enc, X):
    with torch.no_grad(), Site(m, LAYER, record=True) as s:
        z = m(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
    return s.q, s.s, z


def manual(ad, q, u, start=3):
    h = q + u
    mask = (torch.arange(h.shape[1]) >= start).float()[None, :, None]
    with torch.no_grad():
        d = ad.module(h.float(), u.float()) if ad.method == "cond_loreft" else ad.module(h.float())
    return d * mask


def site_logits(m, enc, X, delta):
    with torch.no_grad(), Site(m, LAYER, delta=delta):
        return m(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits


def mk(cls, r, seed=42):
    torch.manual_seed(seed)
    return cls(D, r, LAYER, seed, 3)


def test_zero_init_identity_disabled_and_hook_cleanup(tiny):
    m, enc, X = tiny
    plain = fwd(m, enc, X)
    for cls, r in ((LoReFTAdapter, 3), (CondLoReFTAdapter, 2)):
        ad = mk(cls, r)
        q, u, _ = record(m, enc, X)
        assert bool((manual(ad, q, u) == 0).all())
        assert torch.equal(fwd(m, enc, X, ad), plain)
        p = mk(cls, r)
        perturb(p, 0.3, 0.5, 7)
        assert not torch.equal(fwd(m, enc, X, p), plain)
        with p.disabled():
            assert torch.equal(fwd(m, enc, X, p), plain)
        assert site_hook_count(m, LAYER) == 0 and p.q is None and p.cache_pos is None


def test_m1_with_zero_wu_equals_r2_update_and_logits(tiny):
    m, enc, X = tiny
    r2 = mk(LoReFTAdapter, 2)
    perturb(r2, 0.3, 0.5, 11)
    m1 = mk(CondLoReFTAdapter, 2, seed=5)
    transplant_r2_into_m1(m1, r2.state_dict())
    q, u, _ = record(m, enc, X)
    assert torch.equal(manual(r2, q, u), manual(m1, q, u))
    assert torch.equal(fwd(m, enc, X, r2), fwd(m, enc, X, m1))
    assert set(named_trainable(m1)) - set(named_trainable(r2)) == {"w_u"}


def test_single_application_at_the_site_and_no_stale_u(tiny):
    m, enc, X = tiny
    for cls, r in ((LoReFTAdapter, 3), (CondLoReFTAdapter, 2)):
        ad = mk(cls, r)
        perturb(ad, 0.3, 0.5, 7)
        for sl in (slice(0, 1), slice(1, 2), slice(0, 2)):
            q, u, _ = record(m, enc[sl], X[sl])
            c0 = ad.calls
            z = fwd(m, enc[sl], X[sl], ad)
            assert ad.calls - c0 == 1
            assert torch.equal(z, site_logits(m, enc[sl], X[sl], manual(ad, q, u)))


def test_m1_reads_cross_attention_output_not_residual(tiny):
    """With W_h = R, b = 0 and W_u != 0, the update is exactly (W_u u) R for u = the encoder_attn output."""
    m, enc, X = tiny
    ad = mk(CondLoReFTAdapter, 2)
    with torch.no_grad():
        ad.module.w_u.copy_(torch.randn(2, D))
    q, u, _ = record(m, enc, X)
    R = ad.module.rot.weight.detach()
    expect = F.linear(u.float(), ad.module.w_u.detach()) @ R
    expect = expect * (torch.arange(X.shape[1]) >= 3).float()[None, :, None]
    assert torch.allclose(manual(ad, q, u), expect, atol=1e-6)
    assert torch.equal(fwd(m, enc, X, ad), site_logits(m, enc, X, manual(ad, q, u)))


def test_start_pos_leaves_prefix_positions_unchanged(tiny):
    m, enc, X = tiny
    plain = fwd(m, enc, X)
    for cls, r in ((LoReFTAdapter, 3), (CondLoReFTAdapter, 2)):
        ad = mk(cls, r)
        perturb(ad, 0.3, 0.5, 7)
        z = fwd(m, enc, X, ad)
        assert torch.equal(z[:, :3], plain[:, :3]) and not torch.equal(z[:, 3:], plain[:, 3:])


def test_cached_incremental_equals_full_with_absolute_positions(tiny):
    m, enc, X = tiny
    for cls, r in ((LoReFTAdapter, 3), (CondLoReFTAdapter, 2)):
        ad = mk(cls, r)
        perturb(ad, 0.3, 0.5, 7)
        seen = []
        L = m.model.decoder.layers[LAYER]
        hk = L.register_forward_pre_hook(lambda mod, a, k: seen.append(k["cache_position"].tolist()), with_kwargs=True)
        try:
            with torch.no_grad(), ad.active(m):
                full = m(encoder_outputs=(enc[:1],), decoder_input_ids=X[:1], use_cache=False).logits[0]
                o = m(encoder_outputs=(enc[:1],), decoder_input_ids=X[:1, :4], use_cache=True)
                rows, pkv = [o.logits[0]], o.past_key_values
                for t in range(4, X.shape[1]):
                    o = m(encoder_outputs=(enc[:1],), decoder_input_ids=X[:1, t:t + 1], past_key_values=pkv, use_cache=True,
                          cache_position=torch.tensor([t]))
                    pkv = o.past_key_values
                    rows.append(o.logits[0])
        finally:
            hk.remove()
        inc = torch.cat(rows, 0)
        assert torch.allclose(full, inc, atol=1e-5)
        assert seen[1] == [0, 1, 2, 3] and [s[0] for s in seen[2:]] == list(range(4, X.shape[1]))


def test_gradients_reach_intended_parameters_only(tiny):
    m, enc, X = tiny
    Y = torch.randint(3, 64, X.shape)
    for cls, r in ((LoReFTAdapter, 3), (CondLoReFTAdapter, 2)):
        ad = mk(cls, r)
        z = fwd(m, enc, X, ad, grad=True)
        F.cross_entropy(z[:, 3:].reshape(-1, 64), Y[:, 3:].reshape(-1)).backward()
        g = {n: p.grad for n, p in named_trainable(ad).items()}
        assert all(g[n] is not None and float(g[n].norm()) > 0 for n in ("learned.weight", "learned.bias"))
        if cls is CondLoReFTAdapter:
            assert g["w_u"] is not None and float(g["w_u"].norm()) > 0       # W_u learns from the first step
        assert all(p.grad is None for p in m.parameters())
        p = mk(cls, r)
        perturb(p, 0.3, 0.5, 7)
        z = fwd(m, enc, X, p, grad=True)
        F.cross_entropy(z[:, 3:].reshape(-1, 64), Y[:, 3:].reshape(-1)).backward()
        assert all(q.grad is not None and float(q.grad.norm()) > 0 and torch.isfinite(q.grad).all() for q in named_trainable(p).values())
        # frozen optimizer settings (weight decay 0): the Householder parametrization's diagonal stays exactly -1
        opt = torch.optim.AdamW(p.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0)
        before = {n: q.detach().clone() for n, q in named_trainable(p).items()}
        opt.step()
        assert all(float((q.detach() - before[n]).norm()) > 0 for n, q in named_trainable(p).items())
        assert orth_error(p) < 1e-5
        orig = p.module.rot.parametrizations.weight.original
        assert bool((orig.detach().diagonal() == -1).all())


def test_weight_decay_would_break_the_orthogonal_parametrization():
    """Documents why the frozen recipe's weight_decay = 0 matters: torch's Householder map multiplies by
    X.diagonal().int(); decay moves the diagonal off -1 and int() zeroes a row of R (both arms share this)."""
    torch.manual_seed(0)
    ad = LoReFTAdapter(16, 3, LAYER, 1, 3)
    for q in ad.parameters():
        q.grad = torch.zeros_like(q)
    torch.optim.AdamW(ad.parameters(), lr=1e-2, weight_decay=0.01).step()
    assert orth_error(ad) > 0.5


def test_contribution_decomposition_matches_forward():
    torch.manual_seed(3)
    ad = CondLoReFTAdapter(32, 2, LAYER, 3, 3)
    perturb(ad, 0.2, 0.3, 1)
    h, u = torch.randn(10, 32), torch.randn(10, 32)
    c = contributions(ad.module, h, u)
    with torch.no_grad():
        assert torch.allclose(c["update"], ad.module(h, u), atol=1e-6)
    assert torch.allclose(c["e"], c["e_h"] + c["e_u"] + ad.module.learned.bias.detach(), atol=1e-6)
    assert torch.allclose(c["update"].norm(dim=-1), c["e"].norm(dim=-1), atol=1e-5)
    r3 = LoReFTAdapter(32, 3, LAYER, 3, 3)
    perturb(r3, 0.2, 0.3, 1)
    c3 = contributions(r3.module, h, u)
    with torch.no_grad():
        assert torch.allclose(c3["update"], r3.module(h), atol=1e-6) and bool((c3["e_u"] == 0).all())


def test_token_lang():
    assert token_lang("我们".encode()) == "ZH" and token_lang(" Python".encode()) == "EN"
    assert token_lang(b",") == "OTHER" and token_lang("学".encode()[:2]) == "OTHER"


# ---------------------------------------------------------------- paired statistics

def test_dialogue_table_and_paired_tables():
    def pu(d, st, s_err, zh, sb, ec, ek):
        return {"dialogue": d, "stratum": st, "s_err": s_err, "units": 10, "b_err": 2, "zh_new": zh, "sb": sb, "en_corr": ec, "en_corrupt": ek}
    a = [pu("d1", "CS", 3, 2, 1, 2, 0), pu("d2", "ZH_ONLY", 1, 1, 0, 0, 0)]
    b = [pu("d1", "CS", 2, 1, 0, 1, 0), pu("d2", "ZH_ONLY", 1, 0, 0, 0, 0)]
    ta, tb = dialogue_table(a), dialogue_table(b)
    dl = ["d1", "d2"]
    s = tab_stat(ta, dl, [0, 1])
    assert s["zh_new"] == 3 and s["zh_new_cs"] == 2 and s["en_net"] == 2 and abs(s["cs_mer"] - 0.3) < 1e-12
    dt = {(x, k): t for k in (42, 43, 44) for x, t in (("M1", ta), ("R3", tb))}
    rng = np.random.default_rng(42)
    draws = [rng.choice(2, 2) for _ in range(50)]
    pr = paired_tables(dt, "M1", "R3", dl, draws)
    assert pr["42"]["zh_new"]["point"] == 2 and pr["seed_mean"]["zh_new"]["point"] == 2
    assert pr["42"]["dialogue_contribution"]["zh_new"]["max_abs_dialogue_delta"] == 1


# ---------------------------------------------------------------- decision rules

DR = {"preservation_margin_units": 5, "seed_consistency_min": 2, "en_net_ratio_min": 0.75, "cs_mer_tolerance": 0.005,
      "worst_dialogue_tolerance": 10, "loops_nonterm_tolerance": 1, "strong_zh_new_max": 49}


def arm(zh, sb, en=(40, 40, 40), cs=(0.11, 0.11, 0.11), wd=(10, 10, 10), loops=(0, 0, 0), nt=(1, 1, 1)):
    return {s: {"zh_new": zh[i], "sb": sb[i], "en_net": en[i], "cs_mer": cs[i], "worst_dlg": wd[i], "loops": loops[i], "nonterm": nt[i]}
            for i, s in enumerate((42, 43, 44))}


R2 = arm((53, 48, 50), (30, 31, 26), en=(34, 41, 35))


def test_gain_margin_and_seed_consistency():
    ref = arm((60, 60, 60), (40, 40, 40))
    assert gain(arm((55, 55, 55), (35, 35, 35)), ref, DR)["pass"]                     # exactly 5 lower: passes
    assert not gain(arm((56, 55, 55), (35, 35, 35)), ref, DR)["pass"]                 # 4.67 lower: fails margin
    g = gain(arm((40, 61, 61), (20, 41, 41)), ref, DR)                               # big mean gain from one seed
    assert g["zh_mean_margin"] and not g["zh_seed_consistent"] and not g["pass"]
    assert not gain(arm((50, 50, 50), (40, 40, 40)), ref, DR)["pass"]                # Mandarin only, SB unchanged


def test_guard_corrections_strong():
    m1 = arm((50, 48, 50), (28, 30, 26), en=(30, 30, 30))
    assert guard(m1, R2, DR)["pass"]
    assert not guard(arm((50, 48, 50), (28, 30, 26), loops=(1, 1, 0)), R2, DR)["B3"]
    r3 = arm((60, 60, 60), (40, 40, 40))
    c = corrections(m1, R2, r3, DR)
    assert c["C1"] and c["C2"] and c["C3"] and c["C4"] and abs(c["en_net_ratio_vs_R2"] - 30 / (110 / 3)) < 1e-12
    assert not corrections(arm((50, 48, 50), (28, 30, 26), en=(20, 20, 20)), R2, r3, DR)["C2"]       # ratio 0.545
    assert not corrections(arm((50, 48, 50), (28, 30, 26), en=(30, 0, -1)), R2, r3, DR)["C1"]
    assert not corrections(arm((50, 48, 50), (28, 30, 26), cs=(0.12, 0.12, 0.12)), R2, r3, DR)["C3"]
    assert not corrections(arm((50, 48, 50), (28, 30, 26), wd=(21, 10, 10)), R2, r3, DR)["C4"]
    assert corrections(m1, arm((1, 1, 1), (1, 1, 1), en=(0, 0, 0)), r3, DR)["en_net_ratio_vs_R2"] is None
    assert strong(arm((49, 49, 80), (0, 0, 0)), DR) and not strong(arm((49, 50, 80), (0, 0, 0)), DR)
    assert loops_guard(arm((0,) * 3, (0,) * 3, loops=(1, 0, 0)), arm((0,) * 3, (0,) * 3), DR)


def test_verdict_precedence_and_recommendations():
    good = {"pass": True}
    bad = {"pass": False}
    assert decide(False, good, good, good, good, good)["verdict"] == "ARCH_S1_ENGINEERING_BLOCKED"
    assert decide(True, good, good, good, good, good)["verdict"] == "ARCH_S1_M1_PROMISING"
    assert decide(True, good, bad, good, good, good)["verdict"] == "ARCH_S1_M1_BEATS_R3_ONLY"
    assert decide(True, good, good, bad, good, bad)["verdict"] == "ARCH_S1_R3_CAPACITY_GAIN"     # A passes, C fails
    assert decide(True, bad, good, good, good, good)["verdict"] == "ARCH_S1_R3_CAPACITY_GAIN"
    assert decide(True, bad, good, good, bad, bad)["verdict"] == "ARCH_S1_NO_ARCHITECTURAL_GAIN"
    assert decide(True, bad, good, good, bad, good)["verdict"] == "ARCH_S1_MIXED_OR_INCONCLUSIVE"
    assert set(RECOMMEND.values()) == {"HOLD_FOR_ENGINEERING_REPAIR", "GO_M1_MECHANISM_AND_INDEPENDENT_VALIDATION", "NO_NEW_BEST_METHOD",
                                       "CAPACITY_BOTTLENECK_MORE_PLAUSIBLE", "STOP_GENERIC_LOREFT_ARCHITECTURE_EXPANSION", "HOLD_ARCHITECTURE_CLAIMS"}


def test_improvement_over_r2_end_to_end():
    r3 = arm((45, 42, 44), (24, 25, 20), en=(34, 40, 36))         # >= 5 better on both, corrections kept
    assert improvement_over_r2(r3, R2, DR)["pass"]
    r3b = arm((45, 42, 44), (24, 25, 20), en=(20, 20, 20))        # preservation via suppressed corrections
    assert not improvement_over_r2(r3b, R2, DR)["pass"]
    m1 = arm((47, 44, 46), (29, 30, 25))                          # ~4 better on Mandarin, ~1 on SB
    A = gain(m1, r3, DR)
    assert not A["pass"]
    v = decide(True, A, guard(m1, R2, DR), corrections(m1, R2, r3, DR), improvement_over_r2(r3, R2, DR), improvement_over_r2(m1, R2, DR))
    assert v["verdict"] == "ARCH_S1_R3_CAPACITY_GAIN" and v["recommendation"] == "CAPACITY_BOTTLENECK_MORE_PLAUSIBLE"


def test_freeze_sources_exist():
    from reft.arch_s0s1_train import FREEZE_SOURCES, REUSED_SOURCES
    missing = [s for s in FREEZE_SOURCES + REUSED_SOURCES if not (ROOT / s).exists()]
    assert not missing, missing
    _ = copy  # keep import used
