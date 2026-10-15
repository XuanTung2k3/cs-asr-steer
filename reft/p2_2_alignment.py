"""P2-2 on-policy event construction (pure logic; training data only). docs/reft/P2_2_PROTOCOL.md section 3.

An on-policy event cuts an actual generated hypothesis (token ids) at a clean token boundary that sits at a validated
reference location r (the next reference unit to be produced), and attaches a trustworthy continuation target:

  T (teacher)   frozen B0 replayed from exactly that generated prefix reproduces the next <= relevant_units reference
                units; target = B0's continuation tokens covering the next <= scope_units units (KL(p0 || p_theta)).
  R (reference) otherwise, if the reference scope can follow the generated prefix verbatim (standalone-tokenized
                continuation whose bytes and canonical units extend the prefix exactly); target = those tokens (CE).
  U             otherwise: no loss, reason recorded, kept in every denominator.

Locations: SB = first Mandarin unit after a substantive English unit; PRE = start of that English run; ZH = third unit
of a non-filler Mandarin run of >= zh_control_min_run units. A location is valid only if the <= context_units reference
units before r are exact matches in the hypothesis alignment and that context is unique in the reference, and a clean
boundary holds exactly the hypothesis units through r - 1. No splicing of histories, no hidden-state transplantation.
"""
from __future__ import annotations

import hashlib

from reft.p2_1c_continuation_audit import anchor_at, boundaries, en_region, hyp_after
from reft.text import tokenize

CATS = ("SB", "PRE", "ZH")


def opportunities(ref: list, tags: list, fill: set, E: dict) -> dict:
    """Reference-side candidate locations per category (training reference only)."""
    lang = [k for k in range(len(ref)) if tags[k] in ("EN", "ZH") and ref[k] not in fill]
    sb = [lang[q] for q in range(1, len(lang)) if tags[lang[q]] == "ZH" and tags[lang[q - 1]] == "EN"]
    pre = sorted({en_region(tags, ref, r, fill)[0] for r in sb})
    zh, run = [], []
    for k in range(len(ref) + 1):
        if k < len(ref) and tags[k] == "ZH" and ref[k] not in fill and (not run or run[-1] == k - 1):
            run.append(k)
            continue
        if len(run) >= E["zh_control_min_run"]:
            zh.append(run[E["zh_control_offset"]])
        run = [k] if (k < len(ref) and tags[k] == "ZH" and ref[k] not in fill) else []
    return {"SB": sb, "PRE": pre, "ZH": zh}


def context_unique(ref: list, r: int, n: int) -> bool:
    ctx = ref[max(0, r - n):r]
    if not ctx:
        return True
    L = len(ctx)
    return sum(ref[i:i + L] == ctx for i in range(len(ref) - L + 1)) == 1


def order_key(salt: str, seed: int, epoch: int, uid: str, cat: str, r: int) -> str:
    return hashlib.sha256(f"{salt}:{seed}:{epoch}:{uid}:{cat}:{r}".encode()).hexdigest()


def locate(o: dict, ms: list, r: int, n_ctx: int) -> dict:
    """Generated-prefix location of reference unit r in hypothesis alignment o (text.one) with clean boundaries ms."""
    if r == 0:
        a = anchor_at(ms, 0)
        return {"ok": True, "h": 0, "k": a["k"]}
    ctx = range(max(0, r - n_ctx), r)
    if any(o["by"][i][0] != "match" for i in ctx):
        return {"ok": False, "reason": "context_not_matched"}
    if not context_unique(o["ref"], r, n_ctx):
        return {"ok": False, "reason": "context_ambiguous"}
    h = hyp_after(o["ops"], r - 1)
    a = anchor_at(ms, h)
    if not a["exact"]:
        return {"ok": False, "reason": "no_clean_boundary"}
    return {"ok": True, "h": h, "k": a["k"]}


def teacher_target(ref_units: list, r: int, prefix_bytes: bytes, prefix_units: list, cont: list, tb, eos: int,
                   ref_text: str, E: dict) -> dict:
    """T validity: B0's continuation `cont` (token ids, possibly ending in EOS) from the generated prefix reproduces the
    next <= relevant_units reference units exactly and immediately (alignment-free canonical-unit equality, clean
    junction); target = its tokens up to the clean boundary after the next <= scope_units units, capped at
    max_tokens_per_event. A descriptive retention of the relevant units (alignment against the reference truncated at
    the last relevant unit) is returned for invalid teachers."""
    from reft.text import one
    ctoks = [t for t in cont if t != eos]
    try:
        cbytes = b"".join(tb(t) for t in ctoks)
        joint_text = (prefix_bytes + cbytes).decode("utf-8")
        cunits = tokenize(cbytes.decode("utf-8"))
    except UnicodeDecodeError:
        return {"ok": False, "reason": "teacher_invalid_utf8"}
    if tokenize(joint_text) != prefix_units + cunits:
        return {"ok": False, "reason": "teacher_junction_not_clean"}
    rel = ref_units[r:r + E["relevant_units"]]
    if not rel:
        return {"ok": False, "reason": "no_reference_continuation"}
    if cunits[:len(rel)] != rel:
        sp = unit_spans(ref_text)
        ret = None
        if sp is not None:
            o = one(ref_text[:sp[r + len(rel) - 1][1]], joint_text)
            ret = sum(o["by"][x][0] == "match" for x in range(r, r + len(rel))) / len(rel)
        return {"ok": False, "reason": "teacher_continuation_wrong", "teacher_rel_retention": ret,
                "teacher_terminated_early": (eos in cont) and len(cunits) < len(rel)}
    S = min(E["scope_units"], len(rel))
    ms_c, _ = boundaries([tb(t) for t in ctoks])
    a = anchor_at(ms_c, S)
    if not a["exact"] or a["k"] == 0:
        return {"ok": False, "reason": "teacher_scope_boundary"}
    return {"ok": True, "targets": ctoks[:min(a["k"], E["max_tokens_per_event"])], "scope": [r, r + S],
            "teacher_rel_retention": 1.0}


def unit_spans(target_text: str):
    """Character spans of the canonical units of a target_text string (None if normalization changes its length)."""
    from csasr.data.normalize import normalize_text, segment_units
    norm = normalize_text(target_text)
    if len(norm) != len(target_text) or norm != target_text.lower():
        return None
    return [(u.char_start, u.char_end) for u in segment_units(norm)]


def reference_target(target_text: str, ref_units: list, r: int, prefix_bytes: bytes, prefix_units: list, encode, tb,
                     E: dict) -> dict:
    """R validity: the reference scope (next <= scope_units units, verbatim from the training target text) can follow
    the actual generated prefix: standalone tokenization whose bytes reproduce the text, and whose canonical units extend
    the prefix's units by exactly the scope. Target = those tokens, capped at max_tokens_per_event."""
    sp = unit_spans(target_text)
    if sp is None or [target_text[a:b].lower() for a, b in sp] != ref_units:
        return {"ok": False, "reason": "reference_unmappable"}
    S = min(E["scope_units"], len(ref_units) - r)
    if S <= 0:
        return {"ok": False, "reason": "no_reference_continuation"}
    cont = target_text[sp[r][0]:sp[r + S - 1][1]]
    try:
        ptext = prefix_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "reason": "prefix_invalid_utf8"}
    sep = " " if (ptext and not ptext[-1].isspace() and cont[:1].isascii() and cont[:1].isalnum()) else ""
    ids = encode(sep + cont)
    if not ids or b"".join(tb(i) for i in ids) != (sep + cont).encode("utf-8"):
        return {"ok": False, "reason": "reference_tokenization_mismatch"}
    if tokenize(ptext + sep + cont) != prefix_units + ref_units[r:r + S]:
        return {"ok": False, "reason": "reference_suffix_incompatible"}
    return {"ok": True, "targets": ids[:E["max_tokens_per_event"]], "scope": [r, r + S], "sep": sep}


def prefix_class(own_tokens: list, own_units: list, b0_loc: dict | None, b0_tokens: list, b0_units: list) -> str:
    """How the generated prefix differs from B0's prefix at the same reference location."""
    if b0_loc is None or not b0_loc.get("ok"):
        return "b0_unlocated"
    if own_tokens == b0_tokens[:b0_loc["k"]]:
        return "identical_tokens"
    return "format_only" if own_units == b0_units[:b0_loc["h"]] else "lexical"


def location_label_consistent(ref_units: list, prefix_bytes: bytes, prefix_units: list, target_bytes: bytes, r: int,
                              S: int, truncated: bool) -> bool:
    """Independent re-check from bytes: the joined prefix + supervised target decodes to the prefix's own units followed
    by reference units r, r+1, ... (all fully covered scope units; the last one may be cut by the token cap when
    `truncated`, in which case it must be a prefix of that reference unit)."""
    try:
        units = tokenize((prefix_bytes + target_bytes).decode("utf-8"))
    except UnicodeDecodeError:
        return False
    h = len(prefix_units)
    if units[:h] != prefix_units:
        return False
    cont = units[h:]
    if not cont or len(cont) > S:
        return False
    if truncated:
        return cont[:-1] == ref_units[r:r + len(cont) - 1] and ref_units[r + len(cont) - 1].startswith(cont[-1])
    return cont == ref_units[r:r + len(cont)] and len(cont) == S


def op_event_losses(zs, zt, plen: list, targets: list, kinds: list):
    """Per-event losses on padded batch logits zs (student) / zt (teacher), each [E, T, V]. Event i is scored at the
    positions plen[i] - 1 ... plen[i] - 2 + len(targets[i]) (the logit at position p predicts token p + 1): token-mean
    KL(p0 || p_theta) in float32 for kind "T" (teacher detached), token-mean CE for kind "R". Returns (list of losses,
    event-mean loss) -- each event weighs equally regardless of its token count."""
    import torch
    import torch.nn.functional as F
    losses = []
    for i, (n0, tg, k) in enumerate(zip(plen, targets, kinds)):
        pos = torch.arange(n0 - 1, n0 - 1 + len(tg), device=zs.device)
        lps = F.log_softmax(zs[i, pos].float(), -1)
        if k == "T":
            lp0 = F.log_softmax(zt[i, pos].detach().float(), -1)
            losses.append((lp0.exp() * (lp0 - lps)).sum(-1).mean())
        elif k == "R":
            t = torch.tensor(tg, device=zs.device)
            losses.append(-lps.gather(-1, t[:, None])[:, 0].mean())
        else:
            raise ValueError(k)
    if not losses:
        return [], zs.sum() * 0.0
    return losses, torch.stack(losses).sum() / len(losses)


def loss_aggregate(per_event: list[tuple[str, float]]) -> dict:
    """Event-level mean of per-event (token-mean) losses; T and R reported separately."""
    t = [v for k, v in per_event if k == "T"]
    rr = [v for k, v in per_event if k == "R"]
    n = len(t) + len(rr)
    return {"op": (sum(t) + sum(rr)) / max(1, n), "n_T": len(t), "n_R": len(rr),
            "T_mean": sum(t) / len(t) if t else None, "R_mean": sum(rr) / len(rr) if rr else None}
