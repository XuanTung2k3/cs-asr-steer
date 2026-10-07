"""P2-PATH1 state-owned branch paths (frozen contract configs/inference_cf/p2_path1.json).

``score_branch``: short reference-free donor scoring on a FRESH ``cached.Branch`` created after the caller has
materialized/verified the model state (theta0 or reconstructed A4). Prompt fed once, the common prefix replayed token by
token under that same state (greedy must equal the sealed prefix), then each donor token is scored from canonical
generation-processed float32 log_softmax at its ABSOLUTE content index (begin-suppression only at index 0) BEFORE it is
consumed. No gradients, no greedy completion, EOS never scored.

``owned_clamp``: wraps ``path_decode.clamp_decode`` (fresh Branch per call) with a state-ownership trace: resident LN hash
before/after must equal the declared owner state; the cache is created inside the call after the state is selected and
discarded with it, so no KV ever crosses model states.

Consensus: S_cons(b) = 0.5*S_theta0(b) + 0.5*S_A4(b); |S_cons(B) - S_cons(ALT)| <= 1e-12 chooses B.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

TIE = 1e-12


def processed_log_probs(logits: torch.Tensor, step: int, suppress, begin) -> torch.Tensor:
    x = logits.detach().float().clone()
    if suppress:
        x[list(suppress)] = -float("inf")
    if step == 0 and begin:
        x[list(begin)] = -float("inf")
    return torch.log_softmax(x, -1)


def score_branch(bundle, encoded, prompt: Sequence[int], prefix: Sequence[int], donor: Sequence[int], *, owner: dict,
                 state_hash_fn, capture_layer: int | None = 16) -> dict:
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    eos = bundle.processor.tokenizer.eos_token_id
    if any(int(t) == eos for t in donor) or not donor:
        raise ValueError("donor must be 1..H content tokens without EOS")
    h0 = state_hash_fn()
    if h0 != owner["state_hash"]:
        raise ValueError("resident state != declared owner before branch creation")
    br = cached.Branch(bundle, encoded, list(prompt), "S")          # fresh cache, created AFTER state selection
    k = len(prefix)
    new, lps, prefix_ok = list(prompt), [], True
    with torch.inference_mode():
        for t in range(k + len(donor)):
            assert_no_site_hooks(bundle)
            logits, _, _ = br.step(new, capture_layer=capture_layer, attention=True)
            if t < k:
                if processed_argmax(logits, t, suppress, begin) != int(prefix[t]):
                    prefix_ok = False
                tok = int(prefix[t])
            else:
                tok = int(donor[t - k])
                lp = float(processed_log_probs(logits, t, suppress, begin)[tok])
                if not math.isfinite(lp):
                    raise ValueError(f"donor token {tok} suppressed/nonfinite at absolute index {t}")
                lps.append(lp)
            new = [tok]
    positions_ok = br.positions == list(range(len(br.fed)))
    fed_ok = br.fed == list(prompt) + list(prefix) + list(donor[:-1])
    del br
    h1 = state_hash_fn()
    return {"logprobs": lps, "H_eff": len(lps), "score": sum(lps) / len(lps), "prefix_ok": prefix_ok, "positions_ok": positions_ok,
            "fed_ok": fed_ok, "owner": owner, "state_hash_before": h0, "state_hash_after": h1, "state_locked": h0 == h1 == owner["state_hash"]}


def owned_clamp(bundle, encoded, prompt, *, site: int, forced: Sequence[int], expected_prefix: Sequence[int], owner: dict,
                state_hash_fn, max_new_tokens: int = 200) -> dict:
    from csasr.inference_cf.path_decode import clamp_decode
    h0 = state_hash_fn()
    if h0 != owner["state_hash"]:
        raise ValueError("resident state != declared owner before decode")
    d = clamp_decode(bundle, encoded, prompt, site=site, forced=list(forced), expected_prefix=list(expected_prefix), max_new_tokens=max_new_tokens)
    h1 = state_hash_fn()
    d["owner"] = owner
    d["state_hash_before"], d["state_hash_after"] = h0, h1
    d["state_locked"] = h0 == h1 == owner["state_hash"]
    return d


def consensus(s_t0: dict, s_a4: dict) -> dict:
    """s_m = {'B': score, 'ALT': score}."""
    cons = {b: 0.5 * s_t0[b] + 0.5 * s_a4[b] for b in ("B", "ALT")}
    diff = cons["B"] - cons["ALT"]
    choice = "B" if diff >= -TIE else "ALT"
    return {"S_cons": cons, "margin_B_minus_ALT": diff, "choice": choice, "strict_B_win": diff > TIE,
            "S_min": {b: min(s_t0[b], s_a4[b]) for b in ("B", "ALT")}}
