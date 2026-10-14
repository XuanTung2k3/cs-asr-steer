"""CPU tests for P2-1b: KL objective (direction, mask, normalization), switch-back unit definition, decision rules."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.p2_1b_evaluate import correction_retained, decide, preservation_success, sb_units_from_events, seed_unstable  # noqa: E402
from reft.p2_1b_prefix_audit import classify  # noqa: E402
from reft.p2_1b_train import kl_masked, seed_config, teacher_mask  # noqa: E402


def logits(seed, shape=(2, 6, 11)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * 2


def labels(zt):
    Y = zt.argmax(-1).clone()
    Y[:, :3] = -100
    Y[0, 4] = (Y[0, 4] + 1) % zt.shape[-1]  # a teacher-wrong labelled position
    Y[1, 5] = -100                           # padding
    return Y


def test_kl_direction_mask_and_value():
    zt, zs = logits(0), logits(1)
    Y = labels(zt)
    kl, n = kl_masked(zt, zs, Y)
    m = teacher_mask(zt, Y)
    assert n == int(m.sum()) and not m[0, 4] and not m[1, 5] and not m[:, :3].any()
    lp0, lps = F.log_softmax(zt, -1)[m], F.log_softmax(zs, -1)[m]
    manual = (lp0.exp() * (lp0 - lps)).sum(-1).mean()
    reverse = (lps.exp() * (lps - lp0)).sum(-1).mean()
    assert torch.allclose(kl, manual, atol=1e-6) and not torch.allclose(kl, reverse, atol=1e-3)  # KL(p0 || p_theta)
    assert float(kl) > 0


def test_kl_zero_when_equal_and_nonnegative():
    zt = logits(2)
    Y = labels(zt)
    kl, _ = kl_masked(zt, zt.clone(), Y)
    assert abs(float(kl)) < 1e-6
    for s in range(5):
        assert float(kl_masked(zt, logits(10 + s), Y)[0]) >= -1e-7


def test_kl_padding_independent_and_empty_mask():
    zt, zs = logits(3), logits(4)
    Y = labels(zt)
    kl, n = kl_masked(zt, zs, Y)
    # append padded positions with arbitrary logits: unchanged
    zt2 = torch.cat([zt, logits(5, (2, 4, 11)) * 9], 1)
    zs2 = torch.cat([zs, logits(6, (2, 4, 11)) * 9], 1)
    Y2 = torch.cat([Y, torch.full((2, 4), -100)], 1)
    kl2, n2 = kl_masked(zt2, zs2, Y2)
    assert n2 == n and torch.allclose(kl, kl2)
    Ye = torch.full_like(Y, -100)
    zs_g = zs.clone().requires_grad_(True)
    ke, ne = kl_masked(zt, zs_g, Ye)
    assert ne == 0 and float(ke) == 0.0 and not torch.isnan(ke)
    ke.backward()
    assert zs_g.grad is not None and float(zs_g.grad.abs().sum()) == 0.0


def test_kl_gradient_reaches_student_only():
    zt = logits(7).requires_grad_(True)
    zs = logits(8).requires_grad_(True)
    kl, _ = kl_masked(zt, zs, labels(zt.detach()))
    kl.backward()
    assert zt.grad is None and zs.grad is not None and float(zs.grad.abs().sum()) > 0


def test_seed_config_changes_only_seeds():
    C = {"methods": {"R2": {"init_seed": 42, "lr": 5e-4}}, "training": {"seed": 42, "epochs": 3}}
    D = seed_config(C, 44)
    assert D["methods"]["R2"] == {"init_seed": 44, "lr": 5e-4} and D["training"] == {"seed": 44, "epochs": 3}
    assert C["methods"]["R2"]["init_seed"] == 42


def test_orthogonal_base_follows_global_rng():
    """Amendment r1: R is fixed by the init seed, but the parametrization coordinates (`base`) follow the global RNG."""
    from reft.adapters import LoReFTIntervention

    def build(global_seed):
        torch.manual_seed(global_seed)
        return LoReFTIntervention(16, 2, 42)

    def grads(m, gR):
        (m.rot.weight * gR).sum().backward()
        return m.rot.parametrizations.weight.original.grad

    gR = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    a, b, c = build(0), build(0), build(1)
    base = "parametrizations.weight.0.base"
    assert torch.equal(a.rot.weight, c.rot.weight) and not torch.equal(a.rot.get_buffer(base), c.rot.get_buffer(base))
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    ga, gb, gc = grads(a, gR), grads(b, gR), grads(c, gR)
    assert torch.equal(ga, gb) and not torch.allclose(ga, gc, atol=1e-4)
    assert torch.allclose(ga.norm(), gc.norm(), rtol=1e-5)  # same gradient, different orthonormal coordinates


def ev(k, op, en=False, zh=False, tr="unchanged_correct"):
    return {"k": k, "sys_op": op, "substantive_en": en, "zh_non_filler": zh, "transition": tr}


def test_switchback_units_definition():
    # EN at k=1, then 6 deleted units (5 Mandarin corruptions + 1 already-wrong), then a 2-unit deletion elsewhere
    e = [ev(0, "match", zh=True), ev(1, "match", en=True)] + \
        [ev(k, "del", zh=True, tr="corruption") for k in range(2, 7)] + [ev(7, "del", zh=True, tr="unchanged_wrong")] + \
        [ev(8, "match", zh=True), ev(9, "del", zh=True, tr="corruption"), ev(10, "del", zh=True, tr="corruption")]
    r = sb_units_from_events(e)
    assert r == {"damaged_runs": 2, "long_runs": 1, "sb_runs": 1, "sb_units": 5}
    # same long run not preceded by English -> not a switch-back
    e[1]["substantive_en"] = False
    assert sb_units_from_events(e)["sb_units"] == 0


def arms(zh, sb, net, cs, loops=0, nonterm=1, worst=5):
    return {s: {"zh_new": z, "sb": b, "en_net": n, "cs_mer": c, "loops": loops, "nonterm": nonterm, "worst_dlg": worst}
            for s, z, b, n, c in zip((42, 43, 44), zh, sb, net, cs)}


G = {"zh_new_max": 49}


def test_decision_rules():
    ce = arms([73, 70, 75], [55, 50, 60], [37, 35, 40], [0.110, 0.111, 0.109])
    rescued = arms([40, 45, 60], [20, 25, 50], [33, 30, 31], [0.112, 0.112, 0.111])
    P, R = preservation_success(ce, rescued, G), correction_retained(ce, rescued)
    assert P["P"] and R["R"] and decide(True, P, R, seed_unstable(ce, rescued)) == "P2_1B_PRESERVATION_RESCUED"
    suppressed = arms([10, 12, 15], [0, 0, 2], [2, 0, -1], [0.121, 0.122, 0.12])
    P, R = preservation_success(ce, suppressed, G), correction_retained(ce, suppressed)
    assert P["D"] and not R["R"] and decide(True, P, R, False) == "P2_1B_KL_SUPPRESSES_CORRECTIONS"
    same = arms([72, 71, 74], [54, 49, 61], [36, 35, 40], [0.110, 0.111, 0.109])
    P, R = preservation_success(ce, same, G), correction_retained(ce, same)
    assert not P["D"] and decide(True, P, R, seed_unstable(ce, same)) == "P2_1B_KL_NOT_EFFECTIVE"
    unstable = arms([40, 90, 70], [20, 75, 55], [36, 35, 40], [0.110, 0.111, 0.109])
    assert seed_unstable(ce, unstable)
    assert decide(False, P, R, False) == "P2_1B_ENGINEERING_BLOCKED"
    zero_ce = arms([73, 70, 75], [55, 50, 60], [0, -2, 0], [0.110, 0.111, 0.109])
    assert correction_retained(zero_ce, rescued)["en_net_ratio"] is None and not correction_retained(zero_ce, rescued)["R2"]


def test_prefix_typing():
    g0 = {"logp": -0.5, "argmax_is_gold": True}
    assert classify("damaged", g0, {"logp": -2.0, "argmax_is_gold": False}, None, 1.0) == "A"
    div = {"sys_next_script": "eos", "b0_next_script": "han"}
    assert classify("damaged", g0, {"logp": -0.6, "argmax_is_gold": True}, div, 1.0) == "B"
    assert classify("damaged", g0, {"logp": -0.6, "argmax_is_gold": True}, None, 1.0) == "C"
    assert classify("safe", g0, g0, None, 1.0) is None
