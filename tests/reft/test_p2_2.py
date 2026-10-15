"""CPU tests for P2-2: on-policy event construction (locations, teacher / reference targets, label consistency),
the OP loss (KL direction, positions, token/event normalization, padding, empty), and the decision rules."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.p2_1c_continuation_audit import boundaries  # noqa: E402
from reft.p2_2_alignment import (context_unique, locate, location_label_consistent, op_event_losses, opportunities,  # noqa: E402
                                 order_key, prefix_class, reference_target, teacher_target)
from reft.p2_2_evaluate import decide, exposure, preservation, retention, span_fates, strong  # noqa: E402
from reft.text import one, tokenize  # noqa: E402

E = {"zh_control_min_run": 6, "zh_control_offset": 2, "context_units": 2, "relevant_units": 8, "scope_units": 4,
     "max_tokens_per_event": 8}

# toy vocabulary: token id -> bytes
V = {1: "我".encode(), 2: "们".encode(), 3: " use".encode(), 4: " Python".encode(), 5: "然".encode(), 6: "后".encode(),
     7: "去".encode(), 8: "学".encode(), 9: " python".encode(), 10: ",".encode(), 11: "习".encode(), 12: b"use", 99: b""}
EOS = 99


def tb(i):
    return V[i]


def enc(s):
    """Greedy longest-match toy encoder over V (no EOS)."""
    ids, i = [], 0
    b = s.encode()
    items = sorted(((v, k) for k, v in V.items() if v), key=lambda x: -len(x[0]))
    while i < len(b):
        for v, k in items:
            if b[i:i + len(v)] == v:
                ids.append(k)
                i += len(v)
                break
        else:
            return []
    return ids


REF = "我们use Python然后去学习"   # units: 我 们 use python 然 后 去 学 习 -> SB at 4 (然), PRE at 2 (use)


def hyp(tokens):
    content = [t for t in tokens if t != EOS]
    ms, full = boundaries([tb(t) for t in content])
    text = b"".join(tb(t) for t in content).decode()
    return content, ms, full, one(REF, text)


def test_opportunities_and_context():
    o = one(REF, REF)
    tags = o["tags"]
    ops = opportunities(o["ref"], tags, set(), E)
    assert ops["SB"] == [4] and ops["PRE"] == [2] and ops["ZH"] == []
    E2 = dict(E, zh_control_min_run=5)
    assert opportunities(o["ref"], tags, set(), E2)["ZH"] == [6]   # run 然后去学习 (4..8), third unit
    assert context_unique(["a", "b", "a", "b"], 2, 2) is False and context_unique(["a", "b", "c"], 2, 2) is True
    assert context_unique(["a"], 0, 2) is True
    assert order_key("s", 42, 1, "u", "SB", 4) != order_key("s", 43, 1, "u", "SB", 4)


def test_locate_exact_and_rejections():
    content, ms, full, o = hyp([1, 2, 3, 4, 5, 6, EOS])
    loc = locate(o, ms, 4, 2)
    assert loc == {"ok": True, "h": 4, "k": 4}
    # adapter omitted the Mandarin: own prefix after its English is still locatable (premature-EOS histories kept)
    content, ms, full, o = hyp([1, 2, 3, 4, EOS])
    assert locate(o, ms, 4, 2)["ok"]
    # wrong English before the switch-back: context not matched -> no location (no approximate supervision)
    content, ms, full, o = hyp([1, 2, 3, 6, 5, 6, EOS])
    assert locate(o, ms, 4, 2) == {"ok": False, "reason": "context_not_matched"}
    assert locate(o, ms, 0, 2)["ok"] and locate(o, ms, 0, 2)["h"] == 0


def test_teacher_target_valid_and_invalid():
    content, ms, full, o = hyp([1, 2, 3, 4, EOS])          # own prefix ends after English (omission path)
    loc = locate(o, ms, 4, 2)
    pre = content[:loc["k"]]
    pb, pu = b"".join(tb(t) for t in pre), full[:loc["h"]]
    T = teacher_target(o["ref"], 4, pb, pu, [5, 6, 7, 8, 11, EOS], tb, EOS, REF, E)
    assert T["ok"] and T["targets"] == [5, 6, 7, 8] and T["scope"] == [4, 8]
    T2 = teacher_target(o["ref"], 4, pb, pu, [EOS], tb, EOS, REF, E)            # teacher ends: premature EOS
    assert not T2["ok"] and T2["reason"] == "teacher_continuation_wrong" and T2["teacher_rel_retention"] == 0.0
    T3 = teacher_target(o["ref"], 4, pb, pu, [6, 7, 8, 11, EOS], tb, EOS, REF, E)  # skips 然: omission
    assert not T3["ok"]


def test_reference_target_and_label_consistency():
    content, ms, full, o = hyp([1, 2, 3, 4, EOS])
    loc = locate(o, ms, 4, 2)
    pb, pu = b"".join(tb(t) for t in content[:loc["k"]]), full[:loc["h"]]
    R = reference_target(REF, o["ref"], 4, pb, pu, enc, tb, E)
    assert R["ok"] and R["targets"] == [5, 6, 7, 8] and R["sep"] == ""
    tbytes = b"".join(tb(t) for t in R["targets"])
    assert location_label_consistent(o["ref"], pb, pu, tbytes, 4, 4, False)
    # a mismatched reference suffix (labelled at the wrong location) is rejected by the independent check
    assert not location_label_consistent(o["ref"], pb, pu, "后去学习".encode(), 4, 4, False)
    assert not location_label_consistent(o["ref"], pb, pu, "然后".encode(), 4, 4, False)          # incomplete, not truncated
    assert location_label_consistent(o["ref"], pb, pu, "然后去".encode(), 4, 4, True)             # cut by the token cap
    # repeated characters later in the reference do not confuse the check (alignment-free)
    ref2 = ["我", "们", "use", "python", "然", "后", "然", "后"]
    assert location_label_consistent(ref2, pb, pu, "然后然后".encode(), 4, 4, False)
    # PRE location r = 2 after 我们: English continuation gets a leading-space-free standalone tokenization check
    content2, ms2, full2, o2 = hyp([1, 2, 3, 4, 5, 6, 7, 8, 11, EOS])
    l2 = locate(o2, ms2, 2, 2)
    R2 = reference_target(REF, o2["ref"], 2, b"".join(tb(t) for t in content2[:l2["k"]]), full2[:l2["h"]], enc, tb, E)
    assert R2["ok"] and tokenize(b"".join(tb(t) for t in content2[:l2["k"]] + R2["targets"]).decode()) == ["我", "们", "use", "python", "然", "后"]
    # an encoder that cannot reproduce the continuation bytes -> rejected
    R3 = reference_target(REF, o["ref"], 4, pb, pu, lambda s: [], tb, E)
    assert not R3["ok"] and R3["reason"] == "reference_tokenization_mismatch"


def test_prefix_class():
    assert prefix_class([1, 2], ["我", "们"], {"ok": True, "k": 2, "h": 2}, [1, 2, 3], ["我", "们", "use"]) == "identical_tokens"
    assert prefix_class([1, 2, 10], ["我", "们"], {"ok": True, "k": 2, "h": 2}, [1, 2, 3], ["我", "们", "use"]) == "format_only"
    assert prefix_class([1, 7], ["我", "去"], {"ok": True, "k": 2, "h": 2}, [1, 2, 3], ["我", "们", "use"]) == "lexical"
    assert prefix_class([1], ["我"], None, [], []) == "b0_unlocated"


def logits(seed, shape):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 2


def test_op_losses_positions_direction_normalization_padding():
    zs, zt = logits(0, (3, 9, 13)), logits(1, (3, 9, 13))
    plen, tg, kinds = [4, 5, 3], [[2, 7], [5], [1, 2, 3, 4]], ["T", "R", "T"]
    losses, L = op_event_losses(zs, zt, plen, tg, kinds)
    lp0, lps = F.log_softmax(zt[0, 3:5], -1), F.log_softmax(zs[0, 3:5], -1)
    assert torch.allclose(losses[0], (lp0.exp() * (lp0 - lps)).sum(-1).mean())                 # KL(p0 || p_theta)
    assert not torch.allclose(losses[0], (lps.exp() * (lps - lp0)).sum(-1).mean())
    assert torch.allclose(losses[1], -F.log_softmax(zs[1, 4], -1)[5])                           # CE at plen - 1
    assert torch.allclose(L, (losses[0] + losses[1] + losses[2]) / 3)                           # event mean, token mean within
    # appending padded positions with arbitrary logits leaves the loss unchanged
    zs2 = torch.cat([zs, logits(2, (3, 4, 13)) * 9], 1)
    zt2 = torch.cat([zt, logits(3, (3, 4, 13)) * 9], 1)
    assert torch.allclose(op_event_losses(zs2, zt2, plen, tg, kinds)[1], L)
    # identical student / teacher -> zero KL; empty event set -> zero with gradient defined
    assert float(op_event_losses(zt, zt, [4], [[1, 2]], ["T"])[1]) == 0.0
    zg = zs.clone().requires_grad_(True)
    ls, L0 = op_event_losses(zg, zt, [], [], [])
    L0.backward()
    assert ls == [] and float(L0) == 0.0 and float(zg.grad.abs().sum()) == 0.0
    zg = zs.clone().requires_grad_(True)
    ztg = zt.clone().requires_grad_(True)
    op_event_losses(zg, ztg, plen, tg, kinds)[1].backward()
    assert ztg.grad is None and float(zg.grad.abs().sum()) > 0                                  # teacher detached


def arms(zh, sb, net, cs, loops=(0, 0, 0), nonterm=(1, 1, 1), worst=(5, 5, 5)):
    return {s: {"zh_new": a, "sb": b, "en_net": c, "cs_mer": d, "loops": e, "nonterm": f, "worst_dlg": g}
            for s, a, b, c, d, e, f, g in zip((42, 43, 44), zh, sb, net, cs, loops, nonterm, worst)}


def test_decision_rules():
    kl = arms([53, 48, 50], [30, 31, 26], [34, 41, 35], [0.112, 0.110, 0.110])
    good = arms([40, 45, 47], [18, 20, 15], [33, 38, 36], [0.111, 0.111, 0.110])
    D_ok = {"D": True}
    A, B = preservation(kl, good), retention(kl, good)
    assert A["A"] and B["B"]
    d = decide(True, A, B, strong(good), D_ok)
    assert d == {"verdict": "P2_2_ONPOLICY_FEASIBLE", "substatus": "P2_2_PRESERVATION_RESCUED",
                 "recommendation": "GO_P2_3_MATCHED_SUPERVISION_CONTROL"}
    # an apparent gain never hides insufficient exposure; integrity first
    assert decide(True, A, B, True, {"D": False})["verdict"] == "P2_2_ONPOLICY_EXPOSURE_INSUFFICIENT"
    assert decide(False, A, B, True, D_ok)["verdict"] == "P2_2_ENGINEERING_BLOCKED"
    supp = arms([30, 28, 33], [10, 12, 9], [5, -2, 3], [0.121, 0.120, 0.12])
    A, B = preservation(kl, supp), retention(kl, supp)
    assert decide(True, A, B, strong(supp), D_ok)["verdict"] == "P2_2_CORRECTIONS_SUPPRESSED"
    same = arms([54, 47, 51], [29, 32, 27], [34, 40, 36], [0.112, 0.110, 0.110])
    A, B = preservation(kl, same), retention(kl, same)
    assert not A["A"] and decide(True, A, B, strong(same), D_ok)["verdict"] == "P2_2_NO_INCREMENTAL_GAIN"
    mixed = arms([40, 47, 44], [29, 32, 20], [34, 40, 36], [0.112, 0.110, 0.110])   # 13% fewer new errors, SB -8%
    A, B = preservation(kl, mixed), retention(kl, mixed)
    assert A["A1"] and not A["A2"] and A["mean_zh_new_reduction"] >= 0.10
    assert decide(True, A, B, strong(mixed), D_ok)["verdict"] == "P2_2_INCONCLUSIVE"
    G = {"min_active_non_identity_events": 30, "min_valid_share_of_located": 0.5, "min_seeds": 2}
    assert exposure({42: {"non_identity_active": 40, "valid_share": 0.6}, 43: {"non_identity_active": 31, "valid_share": 0.5},
                     44: {"non_identity_active": 5, "valid_share": 0.9}}, G)["D"]
    assert not exposure({42: {"non_identity_active": 40, "valid_share": 0.4}, 43: {"non_identity_active": 29, "valid_share": 0.9},
                         44: {"non_identity_active": 50, "valid_share": None}}, G)["D"]


def ev(k, en=False, zh=False, b0=True, tr="unchanged_correct"):
    return {"k": k, "substantive_en": en, "zh_non_filler": zh, "b0_ok": b0, "transition": tr}


def test_span_fates():
    e = [ev(0, en=True), ev(1, zh=True), ev(2, zh=True), ev(3, en=True), ev(4, zh=True, tr="corruption"),
         ev(5, zh=True, tr="corruption"), ev(6, en=True), ev(7, zh=True, tr="corruption"), ev(8, zh=True), ev(9, zh=True, b0=False)]
    assert span_fates(e) == {"intact": 1, "dropped": 1, "partial": 1}


def test_label_check_cap_inside_utf8_character():
    """Amendment r1: a capped target ending inside a Han character's UTF-8 bytes is accepted only if those bytes begin the
    next reference unit; an uncapped target with a broken character, or a wrong partial character, is rejected."""
    ref = ["那", "个", "ielts", "ielts", "考", "试"]
    pb, pu = "那个".encode(), ["那", "个"]
    shi = "试".encode()
    good = " IELTS IELTS考".encode() + shi[:2]
    assert location_label_consistent(ref, pb, pu, good, 2, 4, True)
    assert not location_label_consistent(ref, pb, pu, good, 2, 4, False)
    assert not location_label_consistent(ref, pb, pu, " IELTS IELTS考".encode() + "你".encode()[:2], 2, 4, True)
    assert not location_label_consistent(ref, pb, pu, " IELTS 考".encode() + shi[:2], 2, 4, True)
