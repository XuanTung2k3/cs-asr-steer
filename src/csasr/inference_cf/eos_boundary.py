"""PATH4_R1_EOS_BOUNDARY_V1: additive H=1 EOS/content action scoring.

Historical content/content G1 and its scorer are unchanged. This helper owns fresh
single-model prefix paths; it accepts no row IDs, reference, donor or outcome inputs.
It is not an experiment runner. The caller invokes it once at the first disagreement.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from csasr.inference_cf.branch_adjudication import consensus, processed_log_probs

REVISION = "PATH4_R1_EOS_BOUNDARY_V1"


def action_mode(c0: int, c2: int, *, eos: int, vocab_size: int, special_ids: Sequence[int],
                step: int, suppress: Sequence[int], begin: Sequence[int]) -> str:
    """EOS is a legal action, not a content token. A cap is never an EOS action."""
    if not 0 <= step < 200:
        raise ValueError("no action outside the frozen output decision budget")
    forbidden = set(suppress) | (set(begin) if step == 0 else set())
    specials = set(special_ids) - {eos}
    for c in (c0, c2):
        if not isinstance(c, int) or not 0 <= c < vocab_size or c in forbidden or c in specials:
            raise ValueError("invalid/suppressed/non-content action")
    if c0 == c2:
        return "NO_TRIGGER"
    return "EOS_BOUNDARY" if (c0 == eos) != (c2 == eos) else "CONTENT_G1"


def boundary_scores(logits0: torch.Tensor, logits2: torch.Tensor, c0: int, c2: int, *, eos: int,
                    special_ids: Sequence[int], step: int, suppress: Sequence[int], begin: Sequence[int]) -> dict:
    """Four next-action float32 log probabilities, equal horizon1, inherited consensus/tie."""
    from csasr.inference_cf.core_p1 import processed_argmax
    if logits0.ndim != 1 or logits0.shape != logits2.shape or not all(
        bool(torch.isfinite(x).all()) for x in (logits0, logits2)
    ):
        raise ValueError("invalid next-action logits")
    mode = action_mode(c0, c2, eos=eos, vocab_size=logits0.numel(), special_ids=special_ids,
                       step=step, suppress=suppress, begin=begin)
    if mode != "EOS_BOUNDARY":
        raise ValueError("boundary scorer only accepts EOS/content disagreement")
    if (processed_argmax(logits0, step, suppress, begin), processed_argmax(logits2, step, suppress, begin)) != (c0, c2):
        raise ValueError("proposals are not the live processed argmax")
    lp0 = processed_log_probs(logits0, step, suppress, begin)
    lp2 = processed_log_probs(logits2, step, suppress, begin)
    s0 = {"B": float(lp0[c0]), "ALT": float(lp0[c2])}
    s2 = {"B": float(lp2[c0]), "ALT": float(lp2[c2])}
    if not all(math.isfinite(v) for s in (s0, s2) for v in s.values()):
        raise ValueError("nonfinite action score")
    result = consensus(s0, s2)
    winner = "theta0" if result["choice"] == "B" else "A2"
    return {"contract_revision": REVISION, "mode": mode, "H": 1, "c0": c0, "c2": c2,
            "S_theta0": s0, "S_A2": s2, "S_cons": result["S_cons"],
            "margin": result["margin_B_minus_ALT"], "winner": winner,
            "selected_token": c0 if winner == "theta0" else c2}


def score_boundary(b0, b2, encoded, prompt: Sequence[int], prefix: Sequence[int], c0: int, c2: int,
                   *, owners: dict, hash_fns: dict) -> dict:
    """Replay each prefix under its own immutable state; score only the current query.

    Decoder KV is fresh and never shared. The frozen detached encoder may be shared.
    No candidate rollout or additional token forward occurs in this mode.
    """
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    if not 0 <= len(prefix) < 200:
        raise ValueError("no boundary action outside output budget")
    tok = b0.processor.tokenizer
    gen = b0.model.generation_config
    sup, beg, eos = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or []), tok.eos_token_id
    other = b2.model.generation_config
    if (sup, beg, eos, list(tok.all_special_ids)) != (
        list(other.suppress_tokens or []), list(other.begin_suppress_tokens or []),
        b2.processor.tokenizer.eos_token_id, list(b2.processor.tokenizer.all_special_ids)
    ) or b0.model is b2.model:
        raise ValueError("generation/provider identity or independent model ownership mismatch")
    logs, traces = {}, {}
    for name, bundle in (("theta0", b0), ("A2", b2)):
        if hash_fns[name]() != owners[name]:
            raise ValueError("resident state != declared owner")
        assert_no_site_hooks(bundle)
        br = cached.Branch(bundle, encoded, list(prompt), "EOS_BOUNDARY")
        new = list(prompt)
        with torch.inference_mode():
            for t in range(len(prefix) + 1):
                logits, _, _ = br.step(new, capture_layer=16, attention=True)
                if t < len(prefix):
                    if processed_argmax(logits, t, sup, beg) != int(prefix[t]):
                        raise ValueError("prefix is not common greedy history")
                    new = [int(prefix[t])]
        if br.fed != list(prompt) + list(prefix) or br.positions != list(range(len(br.fed))):
            raise ValueError("cache history/position mismatch")
        if hash_fns[name]() != owners[name]:
            raise ValueError("model state changed during scoring")
        logs[name] = logits.detach().clone()
        traces[name] = {"owner": name, "state_hash": owners[name], "state_locked": True,
                        "fed": list(br.fed), "positions": list(br.positions)}
        del br
    d = boundary_scores(logs["theta0"], logs["A2"], c0, c2, eos=eos, special_ids=tok.all_special_ids,
                        step=len(prefix), suppress=sup, begin=beg)
    return {**d, "k": len(prefix), "prefix": list(prefix), "paths": traces}


def execute_boundary(b2, encoded, prompt: Sequence[int], decision: dict, *, owner_hash: str, hash_fn,
                     max_new_tokens: int = 200) -> dict:
    """One selected action under A2: EOS stops; content commits once then ordinary A2.

    Existing owned_clamp provides exact suppression, cache ownership and cap semantics.
    There is no detector, rollout or second guard in this execution path.
    """
    from csasr.inference_cf.branch_adjudication import owned_clamp
    if decision["mode"] != "EOS_BOUNDARY" or decision["H"] != 1 or max_new_tokens != 200:
        raise ValueError("boundary execution contract mismatch")
    return owned_clamp(b2, encoded, prompt, site=decision["k"], forced=[decision["selected_token"]],
                       expected_prefix=decision["prefix"], owner={"state": "A2", "state_hash": owner_hash,
                       "condition": "EOS_BOUNDARY"}, state_hash_fn=hash_fn, max_new_tokens=max_new_tokens)
