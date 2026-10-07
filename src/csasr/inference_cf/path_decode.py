"""P2-PATH0 first-divergence clamp decoding (frozen contract configs/inference_cf/p2_path0.json).

``clamp_decode`` is the historical ``episodic_tta.forced_decode`` loop (fresh ``cached.Branch`` KV cache, prompt fed once,
then exactly one consumed token per step, passive L16 recorder, attention=True, ``core_p1.processed_argmax`` with
unchanged suppression / begin-suppression at content step 0, 200-decision cap) with ONE difference: at content decision
indices ``site .. site+len(forced)-1`` the sealed donor token is consumed instead of the greedy choice. A forced EOS stops
immediately. Before the site the greedy choice must equal the sealed common prefix (recorded; mismatch is INVALID).

Pure helpers: decision streams (content IDs + one logical EOS iff terminated=eos), first divergence, unit-cost
Levenshtein, suffix slicing at the release index, processed-logit branch diagnostics. No reference input here.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

EOS = 50257


def stream(tokens: Sequence[int], terminated: str) -> list[int]:
    return [int(t) for t in tokens] + ([EOS] if terminated == "eos" else [])


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def suffix_distance(x: Sequence[int], y: Sequence[int], release: int) -> int:
    """ED of streams after the forced span; every stream is sliced at the same release index."""
    return edit_distance(list(x)[release:], list(y)[release:])


def allowed_at(token: int, step: int, suppress: Sequence[int], begin: Sequence[int]) -> bool:
    return int(token) not in set(suppress) and not (step == 0 and int(token) in set(begin))


def branch_diagnostics(logits: torch.Tensor, step: int, suppress, begin, alt: int, base: int, partition: dict) -> dict:
    """Processed float32 diagnostics at the branch state: z(alt)-z(base), probabilities, P_E/P_M, entropy."""
    x = logits.detach().float().clone()
    if suppress:
        x[list(suppress)] = -float("inf")
    if step == 0 and begin:
        x[list(begin)] = -float("inf")
    if not (math.isfinite(float(x[alt])) and math.isfinite(float(x[base]))):
        raise ValueError("candidate suppressed at branch state")
    lp = torch.log_softmax(x, -1)
    p = lp.exp()
    fin = torch.isfinite(lp)
    ent = float(-(p[fin] * lp[fin]).sum())
    return {"margin_alt_minus_base": float(x[alt] - x[base]), "p_alt": float(p[alt]), "p_base": float(p[base]),
            "P_E": float(p[list(partition["embedded_ids"])].sum()), "P_M": float(p[list(partition["matrix_ids"])].sum()),
            "entropy": ent, "argmax": int(torch.argmax(x))}


def clamp_decode(bundle, encoded, prompt: Sequence[int], *, site: int | None = None, forced: Sequence[int] = (),
                 expected_prefix: Sequence[int] | None = None, max_new_tokens: int = 200, capture_layer: int | None = 16,
                 at_site: Callable[[torch.Tensor, int], None] | None = None) -> dict:
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    eos = bundle.processor.tokenizer.eos_token_id
    forced = [int(t) for t in forced]
    br = cached.Branch(bundle, encoded, list(prompt), "P")
    new, tokens, term = list(prompt), [], "cap"
    trace = {"prefix_ok": True, "forced": [], "suppression_ok": True, "release_index": None}
    with torch.inference_mode():
        for t in range(max_new_tokens):
            assert_no_site_hooks(bundle)
            logits, _, _ = br.step(new, capture_layer=capture_layer, attention=True)
            greedy = processed_argmax(logits, t, suppress, begin)
            if at_site is not None and site is not None and t == site:
                at_site(logits, t)
            chosen = greedy
            if site is not None and expected_prefix is not None and t < site and greedy != int(expected_prefix[t]):
                trace["prefix_ok"] = False
            if site is not None and site <= t < site + len(forced):
                chosen = forced[t - site]
                ok = allowed_at(chosen, t, suppress, begin)
                trace["suppression_ok"] &= ok
                trace["forced"].append({"t": t, "token": chosen, "greedy": greedy, "allowed": ok, "cache_len": br.length})
                trace["release_index"] = t + 1
            if chosen == eos:
                term = "eos"
                break
            tokens.append(chosen)
            new = [chosen]
    trace["positions_ok"] = br.positions == list(range(len(br.fed)))
    return {"tokens": tokens, "terminated": term, "length": len(tokens), "trace": trace}
