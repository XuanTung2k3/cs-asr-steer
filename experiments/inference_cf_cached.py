#!/usr/bin/env python
"""KV-cached steering execution for Pre-P2 / P2.

Three scientifically isolated cached decoder branches share one encoder output and are fed the
same emitted tokens:

* **B**: forced-ZH (cB = cM), unsteered. Gives the gate inputs (current-query alignment-head
  attention -> window -> E; next-token logits -> R_B) and the L24 site hB.
* **E**: forced-EN (cE), unsteered, own cache built from its own prompt (never a hot swap of
  B's KV). Gives hE, and for the g_cf ablation only, R_Ecf.
* **S**: forced-ZH, steered. The canonical DG-02 hook edits only the current query position
  with (alpha * phi(g), sign * d). Earlier positions' edits live in S's own KV cache, exactly
  as they were produced.

The frozen R2 providers (window, LS-B, BC-B, partition, fallbacks) and the P1 direction/edit
semantics are reused unchanged. No reference, label or evaluator timing enters this module.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from csasr.inference_cf.core_p1 import direction, processed_argmax, selected_gate
from csasr.inference_cf.core_r2 import (conflict_from_logits, local_support, max_attention_window,
                                        prefix_utf8_complete)
from csasr.lss.sites import (DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder,
                             assert_no_site_hooks)
import experiments.inference_cf_p0_r2 as r2

GATE_POLICIES = ("ER", "one", "E", "R", "cf")
DOSE_MAPS = ("id", "sqrt")


def native_lid(bundle, waveform: np.ndarray, language_ids: tuple[int, ...]) -> dict[int, float]:
    """Frozen R2 LocalSupport provider (unchanged)."""
    return r2.native_lid(bundle, waveform, language_ids)


def dose_map(g: float, name: str) -> float:
    """phi(g): 'id' -> g (reference), 'sqrt' -> sqrt(g) (the single predeclared alternative)."""
    if name == "id":
        return float(g)
    if name == "sqrt":
        return math.sqrt(max(0.0, float(g)))
    raise ValueError(f"unknown dose map {name!r}")


def gate_value(policy: str, support: dict | None, baseline: dict | None, ecf: dict | None,
               constant: float = 1.0) -> float:
    """Gate under a named policy; 'ER' is the frozen detector g = E * R_B."""
    if policy == "ER":
        return float(selected_gate(support, baseline))
    if policy == "one":
        return float(constant)
    if policy == "E":
        return float(support["E"])
    if policy == "R":
        return float(baseline["R"])
    if policy == "cf":
        return float(support["E"]) * max(0.0, float(baseline["R"]) - float(ecf["R"]))
    raise ValueError(f"unknown gate policy {policy!r}")


class Branch:
    """One isolated KV-cached decoder branch (its own prompt, cache and positions)."""

    def __init__(self, bundle, encoded, prompt: list[int], name: str):
        self.bundle, self.encoded, self.name = bundle, encoded, name
        self.prompt = list(prompt)
        self.cache = None
        self.fed: list[int] = []          # every token this branch has consumed, in order
        self.positions: list[int] = []    # cache_position of every consumed token

    @property
    def length(self) -> int:
        return len(self.fed)

    @torch.inference_mode()
    def step(self, new_tokens: list[int], *, capture_layer: int | None = None,
             attention: bool = True, hook=None):
        """Feed tokens (the prompt on the first call, then one emitted token) and return the
        last query's logits, alignment-head attention and site at ``capture_layer``."""
        b = self.bundle
        start = self.length
        ids = torch.tensor([list(new_tokens)], device=b.device, dtype=torch.long)
        positions = torch.arange(start, start + len(new_tokens), device=b.device)
        rec = DecoderPostCrossAttnRecorder(b, [capture_layer], keep_last_only=True) \
            if capture_layer is not None else None
        ctx = [c for c in (hook, rec) if c is not None]     # hook first: recorder sees r'
        for c in ctx:
            c.__enter__()
        try:
            out = b.model(encoder_outputs=self.encoded, decoder_input_ids=ids,
                          past_key_values=self.cache, use_cache=True, cache_position=positions,
                          output_attentions=attention, return_dict=True)
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
        self.cache = out.past_key_values
        self.fed.extend(int(x) for x in new_tokens)
        self.positions.extend(int(x) for x in positions.tolist())
        if self.cache.get_seq_length() != self.length:
            raise RuntimeError(f"{self.name}: cache length {self.cache.get_seq_length()} != {self.length}")
        logits = out.logits[0, -1].float().cpu()
        heads = None
        if attention:
            frozen = b.model.generation_config.alignment_heads
            heads = torch.stack([out.cross_attentions[l][0, h, -1].float().cpu() for l, h in frozen])
        h_site = rec.states[capture_layer][0, -1].float().cpu() if rec is not None else None
        return logits, heads, h_site


def _edit_hook(bundle, layer: int, query: int, alpha: float, gain: float, d: torch.Tensor | None,
               num_forced_prefix: int):
    """Canonical DG-02 hook editing only ``query`` (absolute position) of the current call."""
    def action_fn(q, u_source, r, abs_pos):
        n = r.shape[1]
        g = torch.zeros((1, n), device=r.device, dtype=r.dtype)
        dirs = torch.zeros((1, n, r.shape[-1]), device=r.device, dtype=r.dtype)
        if d is not None:
            hit = (abs_pos == query).nonzero().flatten()
            if hit.numel():
                g[0, hit] = gain
                dirs[0, hit] = d.to(r.device, r.dtype)
        return g, dirs
    return DecoderPostCrossAttnInterventionHook(
        bundle, layer, None, alpha=float(alpha), num_forced_prefix=num_forced_prefix,
        action_fn=action_fn, norm_preserve=True, record=True, record_last_only=True)


def cached_decode(bundle, *, waveform: np.ndarray, encoded, conditions: dict, partition: dict,
                  null_probs: dict, language_ids: tuple[int, ...], layer: int, alpha: float,
                  gate_policy: str = "ER", dose: str = "id", direction_sign: float = 1.0,
                  constant_gate: float = 1.0, max_new_tokens: int = 200,
                  num_forced_prefix: int = 4, lid_cache: dict | None = None,
                  lid_key: str = "", on_step=None) -> dict:
    """Reference-free cached greedy decode with the frozen gate-driven intervention."""
    if gate_policy not in GATE_POLICIES or dose not in DOSE_MAPS or direction_sign not in (1.0, -1.0):
        raise ValueError("unknown gate policy, dose map or direction sign")
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    en_id, zh_id = conditions["language_token_ids"]
    lid_cache = {} if lid_cache is None else lid_cache
    B = Branch(bundle, encoded, conditions["cB"], "B")
    E = Branch(bundle, encoded, conditions["cE"], "E")
    S = Branch(bundle, encoded, conditions["cB"], "S")
    new_b, new_e = list(conditions["cB"]), list(conditions["cE"])
    tokens: list[int] = []
    steps = []
    counters = {"forwards": 0, "lid_calls": 0, "lid_cache_hits": 0}
    terminated = "cap"
    for t in range(max_new_tokens):
        assert_no_site_hooks(bundle)
        logits_b, heads, h_b = B.step(new_b, capture_layer=layer, attention=True)
        logits_e, _, h_e = E.step(new_e, capture_layer=layer, attention=False)
        counters["forwards"] += 2
        assert_no_site_hooks(bundle)
        query = S.length + len(new_b) - 1
        step = {"t": t, "query_index": query, "fallback_reason": None, "window": None,
                "local_support": None, "baseline": None, "ecf": None, "g": 0.0, "dose": 0.0}
        reasons = []
        if not prefix_utf8_complete(tok, tokens, byte_decoder):
            reasons.append("mid_character")
        window = None
        try:
            window = max_attention_window(heads.numpy(), len(waveform))
            step["window"] = [window.start_frame, window.end_frame, window.start_sample, window.end_sample]
        except Exception as exc:
            reasons.append(r2._reason(exc, "localizer_fail"))
        try:
            step["baseline"] = conflict_from_logits(logits_b, partition)
        except Exception as exc:
            reasons.append(r2._reason(exc, "baseline_provider_fail"))
        if gate_policy == "cf":
            try:
                step["ecf"] = conflict_from_logits(logits_e, partition)
            except Exception as exc:
                reasons.append("ecf_provider_fail")
        if not reasons:
            try:
                key = (lid_key, window.start_sample, window.end_sample)
                if key in lid_cache:
                    counters["lid_cache_hits"] += 1
                else:
                    lid_cache[key] = native_lid(bundle, waveform[key[1]:key[2]], language_ids)
                    counters["lid_calls"] += 1
                probs = lid_cache[key]
                step["local_support"] = local_support(probs[en_id], probs[zh_id],
                                                       null_probs[en_id], null_probs[zh_id])
            except Exception as exc:
                reasons.append(r2._reason(exc, "local_support_fail"))
        priority = ("mid_character", "localizer_fail", "local_support_fail",
                    "baseline_provider_fail", "ecf_provider_fail", "nonfinite_signal")
        step["fallback_reason"] = next((r for r in priority if r in reasons), None)
        if step["fallback_reason"] is None:
            step["g"] = gate_value(gate_policy, step["local_support"], step["baseline"],
                                   step["ecf"], constant_gate)
            step["g_ER"] = float(selected_gate(step["local_support"], step["baseline"]))
        step["dose"] = float(alpha) * dose_map(step["g"], dose)
        dirn = direction(h_e, h_b)
        step["direction_status"], step["direction_norm"] = dirn["status"], dirn["norm"]
        d = None if dirn["d"] is None else direction_sign * dirn["d"]
        step["edit_applied"] = bool(step["dose"] > 0 and d is not None and query >= num_forced_prefix)
        hook = _edit_hook(bundle, layer, query, alpha,
                          dose_map(step["g"], dose) if step["edit_applied"] else 0.0,
                          d if step["edit_applied"] else None, num_forced_prefix)
        logits_s, _, _ = S.step(new_b, attention=True, hook=hook)
        counters["forwards"] += 1
        assert_no_site_hooks(bundle)
        audit = hook.records[-1].to_dict() if hook.records else None
        step["hook_audit"] = audit
        if step["edit_applied"]:
            step["relative_edit"] = audit["edit_norm"] / max(audit["pre_norm"], 1e-12)
        step["f3_bitwise_equals_b"] = bool(torch.equal(logits_s, logits_b))
        step["unsteered_next"] = processed_argmax(logits_b, t, suppress, begin)
        step["steered_next"] = processed_argmax(logits_s, t, suppress, begin)
        top2 = torch.topk(_processed(logits_s, t, suppress, begin), 2).values
        step["steered_top2_margin"] = float(top2[0] - top2[1])
        steps.append(step)
        if on_step is not None:      # read-only probe (equivalence harness); never alters the decode
            on_step(t=t, prefix=list(tokens), step=step, h_b=h_b, h_e=h_e, d=d,
                    logits_b=logits_b, logits_s=logits_s)
        nxt = step["steered_next"]
        if nxt == eos:
            terminated = "eos"
            break
        tokens.append(nxt)
        new_b, new_e = [nxt], [nxt]
    lineage_ok = (B.fed[len(conditions["cB"]):] == tokens[:len(B.fed) - len(conditions["cB"])]
                  and E.fed[len(conditions["cE"]):] == B.fed[len(conditions["cB"]):]
                  and S.fed == B.fed and B.positions == list(range(len(B.fed)))
                  and E.positions == list(range(len(E.fed))) and S.positions == list(range(len(S.fed))))
    distinct_caches = len({id(B.cache), id(E.cache), id(S.cache)}) == 3
    return {"tokens": tokens, "text": tok.decode(tokens, skip_special_tokens=True),
            "terminated": terminated, "steps": steps, "counters": counters,
            "lineage_ok": bool(lineage_ok), "distinct_caches": bool(distinct_caches),
            "config": {"layer": layer, "alpha": alpha, "gate_policy": gate_policy, "dose": dose,
                       "direction_sign": direction_sign, "constant_gate": constant_gate}}


def _processed(logits: torch.Tensor, step: int, suppress, begin) -> torch.Tensor:
    x = logits.detach().float().clone()
    if suppress:
        x[list(suppress)] = -float("inf")
    if step == 0 and begin:
        x[list(begin)] = -float("inf")
    return x


@torch.inference_mode()
def cached_greedy(bundle, encoded, prompt: list[int], max_new_tokens: int = 200) -> dict:
    """Plain cached greedy decode under one prompt (B0 cached / B1 English-forced)."""
    tok = bundle.processor.tokenizer
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    br = Branch(bundle, encoded, prompt, "G")
    new, tokens, margins, terminated = list(prompt), [], [], "cap"
    for t in range(max_new_tokens):
        logits, _, _ = br.step(new, attention=True)
        x = _processed(logits, t, suppress, begin)
        top2 = torch.topk(x, 2)
        margins.append(float(top2.values[0] - top2.values[1]))
        nxt = int(top2.indices[0])
        if nxt == tok.eos_token_id:
            terminated = "eos"
            break
        tokens.append(nxt)
        new = [nxt]
    return {"tokens": tokens, "text": tok.decode(tokens, skip_special_tokens=True),
            "terminated": terminated, "top2_margins": margins}
