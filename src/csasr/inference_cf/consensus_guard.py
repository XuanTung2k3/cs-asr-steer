"""P2-PATH2 online consensus guard (frozen contract configs/inference_cf/p2_path2.json).

Two independent resident model instances (immutable theta0; reconstructed A4) with identical non-LN weights. Every
decoder path owns a fresh ``cached.Branch`` on ONE instance; decoder KV is never shared across instances or paths.

* ``lockstep_detect``: feed the same prompt / emitted history to a theta0 Branch and an A4 Branch, compare canonical
  processed argmax at each absolute content step; equal non-EOS -> emit and feed both; equal EOS -> NO_TRIGGER; first
  unequal -> trigger (k, live prefix, both argmax). One event only. No historical site/group input.
* ``rollout``: fresh owner Branch, replay prompt + live prefix (greedy must equal it), greedy up to H=3 CONTENT tokens,
  stopping at EOS or the global 200-decision budget (EOS never part of the scored branch).
* Scores: unchanged ``branch_adjudication.score_branch``/``consensus`` (theta0 branch == 'B', A4 branch == 'ALT';
  tie -> theta0).
* ``g1``: A4 path forcing only the winner's first token at k (fresh A4 cache), then ordinary A4 greedy.
* ``g2``: theta0 winner -> fresh A4 cache replaying prompt + prefix + the whole theta0 branch token by token (plus its
  EOS if the branch terminated), then ordinary A4 greedy; A4 winner -> ordinary A4 (the G1 output).
No reference, C/U, historical site, donor or AUTO output is accepted by any function here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import torch

H = 3
MAX_NEW = 200


def _gen(bundle):
    g = bundle.model.generation_config
    return list(g.suppress_tokens or []), list(g.begin_suppress_tokens or []), bundle.processor.tokenizer.eos_token_id


def lockstep_detect(b_t0, b_a4, enc, prompt: Sequence[int], *, owners: dict, hash_fns: dict, max_new: int = MAX_NEW) -> dict:
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    sup, beg, eos = _gen(b_t0)
    if (sup, beg, eos) != _gen(b_a4):
        raise ValueError("generation settings differ between instances")
    h0 = {k: hash_fns[k]() for k in ("theta0", "A4")}
    if any(h0[k] != owners[k] for k in h0):
        raise ValueError("resident state != declared owner before detector")
    br = {"theta0": cached.Branch(b_t0, enc, list(prompt), "D0"), "A4": cached.Branch(b_a4, enc, list(prompt), "D4")}
    new, tokens, pairs = list(prompt), [], []
    out = {"trigger": False, "terminated": "cap"}
    with torch.inference_mode():
        for t in range(max_new):
            assert_no_site_hooks(b_t0)
            assert_no_site_hooks(b_a4)
            a = {}
            for k, bundle in (("theta0", b_t0), ("A4", b_a4)):
                lg, _, _ = br[k].step(new, capture_layer=16, attention=True)
                a[k] = processed_argmax(lg, t, sup, beg)
            pairs.append([a["theta0"], a["A4"]])
            if a["theta0"] != a["A4"]:
                out.update(trigger=True, k=t, argmax_theta0=a["theta0"], argmax_A4=a["A4"], terminated=None)
                break
            if a["theta0"] == eos:
                out["terminated"] = "eos"
                break
            tokens.append(a["theta0"])
            new = [a["theta0"]]
    out.update(prefix=tokens, argmax_pairs=pairs,
               fed_identical=br["theta0"].fed == br["A4"].fed,
               positions_ok=all(b.positions == list(range(len(b.fed))) for b in br.values()))
    del br
    h1 = {k: hash_fns[k]() for k in ("theta0", "A4")}
    out["state_locked"] = all(h0[k] == h1[k] == owners[k] for k in h0)
    return out


def rollout(bundle, enc, prompt, prefix: Sequence[int], *, owner_hash: str, hash_fn, h: int = H, max_new: int = MAX_NEW) -> dict:
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    sup, beg, eos = _gen(bundle)
    if hash_fn() != owner_hash:
        raise ValueError("resident state != declared owner before rollout")
    br = cached.Branch(bundle, enc, list(prompt), "R")
    new, toks, prefix_ok, term = list(prompt), [], True, None
    k = len(prefix)
    with torch.inference_mode():
        for t in range(max_new):
            lg, _, _ = br.step(new, capture_layer=16, attention=True)
            g = processed_argmax(lg, t, sup, beg)
            if t < k:
                prefix_ok &= g == int(prefix[t])
                new = [int(prefix[t])]
                continue
            if g == eos:
                term = "eos"
                break
            toks.append(g)
            if len(toks) == h:
                break
            new = [g]
        else:
            term = "cap"
    if term is None and k + len(toks) >= max_new:
        term = "cap"
    positions_ok = br.positions == list(range(len(br.fed)))
    del br
    return {"tokens": toks, "H_eff": len(toks), "terminated": term, "prefix_ok": prefix_ok, "positions_ok": positions_ok,
            "state_locked": hash_fn() == owner_hash}


def decision_hash(d: dict) -> str:
    keys = ("k", "prefix", "b0", "b4", "scores", "winner", "margin")
    return "sha256:" + hashlib.sha256(json.dumps({k: d[k] for k in keys}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def g1_forced(decision: dict) -> list[int]:
    w = decision["b0"] if decision["winner"] == "theta0" else decision["b4"]
    return [int(w["tokens"][0])]


def g2_forced(decision: dict, eos: int) -> list[int] | None:
    """None -> A4 winner (ordinary A4). Theta0 winner -> whole selected branch (+EOS if the branch terminated)."""
    if decision["winner"] != "theta0":
        return None
    b = decision["b0"]
    return [int(x) for x in b["tokens"]] + ([eos] if b["terminated"] == "eos" else [])
