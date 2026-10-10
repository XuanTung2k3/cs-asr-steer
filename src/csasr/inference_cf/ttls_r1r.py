"""TTLS-R1R numerical repair path (frozen contract docs/inference_cf/TTLS_R1R_SPEC.md, configs/inference_cf/ttls_r1r.json).

Versioned, additive repair of the TTLS-R1 zero-edit defect found by the independent audit
(docs/inference_cf/TTLS_R1_INDEPENDENT_AUDIT.md). The historical shared kernel ``models.hooks.apply_steering``,
``lss.sites.DecoderPostCrossAttnInterventionHook`` and ``inference_cf.ttls`` are NOT modified; TTLS-R1 remains
exactly replayable.

The only numerical change (``NUMERICS = "ratio_first_v1"``):

    historical: steered = steered / (new_norm + EPS) * orig_norm            # BF16 divide-then-multiply
    R1R:        steered = steered * (orig_norm / new_norm.clamp_min(EPS))   # ratio first, floored denominator

At an exact zero edit ``steered == hidden`` bitwise, so ``new_norm == orig_norm`` bitwise and the ratio is exactly 1:
the site, the FFN input and every downstream logit are bitwise the clean decoder's. The edit stays inside the
autograd graph (no z == 0 branch, no detach), so d(loss)/dz at z = 0 is the projected clean gradient.

Everything else (site, layer, masks, variable, optimizer, objectives, budget, decoding) is imported unchanged from
``inference_cf.ttls``. No reference, evaluator or outcome input anywhere in this module.
"""
from __future__ import annotations

from typing import Sequence

import torch

from csasr.inference_cf import ttls as L
from csasr.inference_cf.soft_auto_tta import allowed_log_probs, kl_terms
from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, _first, _rebuild
from csasr.models.hooks import EPS

NUMERICS = "ratio_first_v1"


def apply_steering_ratio_first(hidden: torch.Tensor, direction: torch.Tensor, alpha: float, scale: float,
                               gain: torch.Tensor | None, norm_preserve: bool = True) -> torch.Tensor:
    """``models.hooks.apply_steering`` with the norm repair computed ratio-first. Identical argument contract,
    shape checks and zero-gain / alpha-zero bypasses; the only difference is the final renormalisation line."""
    if alpha == 0.0:
        return hidden
    d = direction.to(hidden.device, hidden.dtype)
    if d.ndim == 1:
        if d.shape[0] != hidden.shape[-1]:
            raise ValueError(f"direction dim {d.shape[0]} != hidden dim {hidden.shape[-1]}")
        d = d.view(1, 1, -1)
    elif d.ndim == 3:
        if d.shape != hidden.shape:
            raise ValueError(f"direction shape {tuple(d.shape)} != hidden shape {tuple(hidden.shape)}")
    else:
        raise ValueError(f"direction must have shape (D,) or (B,T,D), got {tuple(d.shape)}")
    if gain is None:
        g = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
    else:
        g = gain.to(hidden.device, hidden.dtype)
        if g.shape != hidden.shape[:2]:
            raise ValueError(f"gain shape {tuple(g.shape)} != {tuple(hidden.shape[:2])}")
    active = g > 0
    if not bool(active.any()):
        return hidden

    delta = (alpha * scale) * g.unsqueeze(-1) * d
    steered = hidden + delta
    if norm_preserve:
        orig_norm = hidden.norm(dim=-1, keepdim=True)
        new_norm = steered.norm(dim=-1, keepdim=True)
        steered = steered * (orig_norm / new_norm.clamp_min(EPS))
    return torch.where(active.unsqueeze(-1), steered, hidden)


class RatioFirstInterventionHook(DecoderPostCrossAttnInterventionHook):
    """The canonical DG-02 exact-site hook (post-cross-attn, pre-FFN; same capture, positions, forced prefix,
    rebuild and lifecycle) with ``apply_steering_ratio_first`` as its repair kernel."""

    def _hook(self, _mod, _inp, output):
        if self._q is None:
            raise RuntimeError("cross-attention ran without its layer-norm pre-hook")
        u_source = _first(output)
        q = self._q
        site = q + u_source
        self.calls += 1
        T = site.shape[1]
        abs_pos = self._abs_positions(T, site.device)
        eligible = abs_pos >= self.num_forced_prefix
        if self.action_fn is not None:
            gain, direction = self.action_fn(q=q, u_source=u_source, r=site, abs_pos=abs_pos)
            if not torch.is_tensor(gain) or not torch.is_tensor(direction):
                raise TypeError("action_fn must return (gain_tensor, direction_tensor)")
            gain = self._coerce_gain(gain, site, eligible)
        else:
            gain = self._resolve_gain(q, u_source, site, abs_pos, eligible)
            direction = self._resolve_direction(q, u_source, site, abs_pos)
        steered = apply_steering_ratio_first(site, direction, self.alpha, self.scale, gain, self.norm_preserve)
        if self.record:
            self._emit_records(abs_pos, eligible, gain, site, steered)
        if steered is site:
            return output
        self.steered_calls += 1
        return _rebuild(output, u_source + (steered - site))


def ttls_hook_r1r(bundle, z: torch.Tensor, steps, *, layer: int = L.LAYER, mode: str = "train", record: bool = False):
    """``ttls.ttls_hook`` (same unit-gain 0/1 step mask, absolute query PROMPT_LEN - 1 + t, forced prefix 4) on the
    ratio-first kernel."""
    targets = None if steps == L.ALL else torch.tensor(sorted(L.PROMPT_LEN - 1 + int(t) for t in steps), dtype=torch.long)

    def action_fn(q, u_source, r, abs_pos):
        if targets is None:
            g = torch.ones((1, r.shape[1]), device=r.device, dtype=r.dtype)
        else:
            g = torch.isin(abs_pos, targets.to(abs_pos.device)).to(r.dtype).view(1, -1)
        return g, z
    return RatioFirstInterventionHook(bundle, layer, None, alpha=1.0, num_forced_prefix=L.PROMPT_LEN,
                                      action_fn=action_fn, norm_preserve=True, mode=mode, record=record,
                                      record_last_only=False)


class EpisodeR1R(L.Episode):
    """``ttls.Episode`` (TTLS variable only): identical fresh zero fp32 z, AdamW (lr E*/(2 sqrt d)), projection,
    2 steps / 3 loss evaluations; the teacher-forced forward uses the ratio-first hook."""

    def __init__(self, bundle, encoded, steps, layer: int = L.LAYER):
        super().__init__(bundle, None, encoded, "TTLS", steps=steps, layer=layer)
        if bool((self.z != 0).any()) or self.opt.state:
            raise L.TTAInvalid("TTLS episode did not start from a fresh zero vector / empty optimizer")

    def forward(self, prompt, y) -> torch.Tensor:
        with ttls_hook_r1r(self.bundle, self.z, self.steps, layer=self.layer, mode="train"):
            return L.teacher_logits(self.model, self.encoded, prompt, y, None)


@torch.no_grad()
def site_ffn_probe(bundle, encoded, y_B, *, z, steps, layer: int = L.LAYER, prompt: Sequence[int] = L.CB,
                   kernel: str = NUMERICS) -> dict:
    """Teacher-forced B0-path forward with/without the hook, recording the tensor the layer's FFN block actually
    consumes (forward pre-hook on ``final_layer_norm``) and the logits. Returns exact-identity flags, edited-position
    chord / norm-preservation error and maximum out-of-mask displacement. ``kernel='historical'`` probes the
    TTLS-R1 kernel for contrast."""
    model = bundle.model
    T = len(y_B)
    ids = torch.tensor([list(prompt) + list(y_B)], device=next(model.parameters()).device, dtype=torch.long)
    kw = dict(encoder_outputs=encoded, decoder_input_ids=ids, use_cache=False, return_dict=True)
    ffn_ln = bundle.decoder_layer(layer).final_layer_norm
    cap = {}

    def grab(_m, args):
        cap["x"] = args[0].detach().clone()
    h = ffn_ln.register_forward_pre_hook(grab)
    try:
        clean = model(**kw).logits[0].float()
        x0 = cap.pop("x")[0].float()
        hook = (ttls_hook_r1r if kernel == NUMERICS else L.ttls_hook)(bundle, z, steps, layer=layer, mode="steer")
        with hook:
            edited = model(**kw).logits[0].float()
        x1 = cap.pop("x")[0].float()
    finally:
        h.remove()
    n = ids.shape[1]
    edit_q = list(range(L.PROMPT_LEN, n)) if steps == L.ALL else [L.PROMPT_LEN - 1 + int(t) for t in steps if L.PROMPT_LEN - 1 + int(t) < n]
    outside = [q for q in range(n) if q not in set(edit_q)]
    chord = torch.linalg.vector_norm(x1 - x0, dim=-1)
    nrel = (torch.linalg.vector_norm(x1, dim=-1) - torch.linalg.vector_norm(x0, dim=-1)).abs() / torch.linalg.vector_norm(x0, dim=-1)
    return {"kernel": kernel, "logits_bitwise_equal": bool(torch.equal(clean, edited)),
            "ffn_input_bitwise_equal": bool(torch.equal(x0, x1)),
            "max_abs_logit_diff": float((clean - edited).abs().max()),
            "edited_queries": len(edit_q),
            "edited_chord_max": float(chord[edit_q].max()) if edit_q else 0.0,
            "edited_chord_mean": float(chord[edit_q].mean()) if edit_q else 0.0,
            "edited_norm_rel_err_max": float(nrel[edit_q].max()) if edit_q else 0.0,
            "outside_chord_max": float(chord[outside].max()) if outside else 0.0}


def zero_kl(bundle, encoded, y_B, base_logq, S, steps, suppress, begin, *, layer: int = L.LAYER,
            prompt: Sequence[int] = L.CB) -> float:
    """Preservation KL at an exact zero vector through the training forward (should be exactly 0)."""
    z = torch.zeros(int(bundle.d_model), device=next(bundle.model.parameters()).device)
    with torch.no_grad(), ttls_hook_r1r(bundle, z, steps, layer=layer, mode="train"):
        logits = L.teacher_logits(bundle.model, encoded, prompt, y_B, None)
    if not S:
        return 0.0
    kl = kl_terms(base_logq, allowed_log_probs(logits, len(y_B), suppress, begin))
    return float(kl[list(S)].mean())


@torch.no_grad()
def displacement_r1r(bundle, encoded, y_B, *, z, steps, base: dict, suppress=(), begin=(), S: Sequence[int] = (),
                     edit_steps: Sequence[int] = (), layer: int = L.LAYER, prompt: Sequence[int] = L.CB) -> dict:
    """``ttls.displacement`` (adapted branch) with the ratio-first hook; same outputs and definitions."""
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    model = bundle.model
    T = len(y_B)
    ids = torch.tensor([list(prompt) + list(y_B)], device=next(model.parameters()).device, dtype=torch.long)
    kw = dict(encoder_outputs=encoded, decoder_input_ids=ids, use_cache=False, return_dict=True)
    hook = ttls_hook_r1r(bundle, z, steps, layer=layer, mode="steer")
    rec = DecoderPostCrossAttnRecorder(bundle, [layer])
    hook.__enter__()
    rec.__enter__()
    try:
        out = model(**kw)
    finally:
        rec.__exit__(None, None, None)
        hook.__exit__(None, None, None)
    q0 = L.PROMPT_LEN - 1
    site = rec.states[layer][0, q0:q0 + T + 1].float()
    logits = out.logits[0, q0:q0 + T + 1].float()
    rel = (torch.linalg.vector_norm(site - base["site"], dim=-1) / torch.linalg.vector_norm(base["site"], dim=-1)).cpu()
    kl = kl_terms(base["logq"], allowed_log_probs(logits, T, suppress, begin)).cpu() if T else torch.zeros(0)
    es = [t for t in edit_steps if 0 <= t <= T]
    return {"site_rel_mean": float(rel[:T + 1].mean()), "site_rel_edited_mean": float(rel[es].mean()) if es else None,
            "site_rel_max": float(rel.max()), "kl_mean_all": float(kl.mean()) if T else 0.0,
            "kl_mean_stable": float(kl[list(S)].mean()) if S else None, "kl_max": float(kl.max()) if T else 0.0}
