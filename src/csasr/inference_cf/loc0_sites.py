"""ST-LOC0 site support (frozen spec docs/inference_cf/ST_LOC0_SPEC.md section 5/7).

Additive mechanical support only; the historical providers, solver and DG-02 hook are reused unchanged.

* ``POST_SELF_ATTENTION`` -- q = h_in + u_self, the decoder residual after self-attention and before
  cross-attention. :class:`SelfAttnResidualInterventionHook` captures h_in (self-attn-LN pre-hook) and
  u_self (self_attn output[0]), forms q in native dtype, repairs ONLY the query position with
  ``apply_steering`` (NormPreserve, no depth rescale) and returns ``u_self + (q_repaired - q)`` so the
  layer's own residual addition feeds the edited q to BOTH the cross-attention LN and its residual bypass.
  The actually consumed q is observed by :class:`DecoderPostCrossAttnRecorder` (``q_states``).
* ``POST_CROSS_ATTENTION_PRE_FFN`` -- r, exact DG-02, via the unchanged
  :class:`DecoderPostCrossAttnInterventionHook`.

:func:`pulse_action` mirrors ``experiments.inference_cf_p2r.pulse_hook``'s action exactly (same solver,
same CPU-float64 scaled direction, same per-position gain) and adds only the frozen ST-LOC0 rule that a
solver result whose emulated relative squared-energy error exceeds the frozen tolerance is a
predetermined no-edit (``solver_target_unattainable``) instead of an off-target pulse.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import torch

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, _first, assert_dropout_disabled
from csasr.models.hooks import apply_steering

SITES = ("POST_SELF_ATTENTION", "POST_CROSS_ATTENTION_PRE_FFN")
SELF, CROSS = SITES


def pulse_action(query: int, v_fn: Callable, target: float, info: dict, max_rel_sq: float, solve_scale, scaled_direction):
    """Action editing only absolute position ``query`` at realized chord ``target`` (P2-R arithmetic)."""
    def action_fn(q=None, u_source=None, r=None, abs_pos=None):
        n = r.shape[1]
        g = torch.zeros((1, n), device=r.device, dtype=r.dtype)
        dirs = torch.zeros((1, n, r.shape[-1]), device=r.device, dtype=r.dtype)
        hit = (abs_pos == query).nonzero().flatten()
        if hit.numel():
            k = int(hit[0])
            v, vstatus = v_fn(r[0, k])
            info["direction_status"] = vstatus
            if v is not None:
                sol = solve_scale(r[0, k], v, target)
                info.update(sol)
                vv = v.detach().cpu().double()
                vv = vv / torch.linalg.vector_norm(vv)
                rc = r[0, k].detach().cpu().double()
                info["cos_v_state"] = float(torch.dot(vv, rc) / torch.linalg.vector_norm(rc))
                if sol["status"] == "ok":
                    if not sol["rel_sq_err"] <= max_rel_sq:
                        info["no_edit_reason"] = "solver_target_unattainable_within_8_evaluations"
                    else:
                        g[0, k] = 1.0
                        dirs[0, k] = scaled_direction(vv, sol["s"], r)
                        info["v"] = vv
                        x = r[0, k].reshape(1, 1, -1)
                        prop = apply_steering(x, dirs[0, k].reshape(1, 1, -1), 1.0, 1.0,
                                              torch.ones((1, 1), device=x.device, dtype=x.dtype), True)
                        info["_proposed"] = prop.reshape(-1).detach().float().cpu()
                else:
                    info["no_edit_reason"] = sol["status"]
            else:
                info["no_edit_reason"] = vstatus
        return g, dirs
    return action_fn


class SelfAttnResidualInterventionHook:
    """Residual-aware POST_SELF_ATTENTION intervention (one decoder layer)."""

    def __init__(self, bundle, layer: int, action_fn, *, num_forced_prefix: int, record: bool = True):
        self.bundle, self.layer, self.action_fn = bundle, int(layer), action_fn
        self.num_forced_prefix = int(num_forced_prefix)
        self.record = record
        self.records: list[dict] = []
        self.calls = self.steered_calls = 0
        self._h_in = self._cache_position = self._past_key_values = None
        self._handles: list = []

    def _layer_pre_hook(self, _mod, args, kwargs):
        self._cache_position = kwargs.get("cache_position", None)
        self._past_key_values = kwargs.get("past_key_values", None)
        return None

    def _ln_pre_hook(self, _mod, args):
        self._h_in = args[0]
        return None

    def _abs_positions(self, T: int, device) -> torch.Tensor:
        cp = self._cache_position
        if cp is not None:
            pos = cp.detach().to(device=device).long().reshape(-1)
            if pos.numel() >= T:
                return pos[:T]
        raise RuntimeError("self-site hook requires the layer's cache_position")

    def _hook(self, _mod, _inp, output):
        if self._h_in is None:
            raise RuntimeError("self_attn ran without its layer-norm pre-hook")
        u_self = _first(output)
        q = self._h_in + u_self                                   # native dtype, == layer's residual + output
        self.calls += 1
        T = q.shape[1]
        abs_pos = self._abs_positions(T, q.device)
        eligible = abs_pos >= self.num_forced_prefix
        gain, dirs = self.action_fn(q=None, u_source=u_self, r=q, abs_pos=abs_pos)
        gain = gain.to(q.device, q.dtype) * eligible.to(q.dtype).view(1, T)
        steered = apply_steering(q, dirs, 1.0, 1.0, gain, True)
        if self.record:
            x, y = q[:, -1].detach().float(), steered[:, -1].detach().float()
            self.records.append({"layer": self.layer, "abs_pos": int(abs_pos[-1]), "gate": float(gain[0, -1]),
                                 "pre_norm": float(x.norm()), "post_norm": float(y.norm()),
                                 "edit_norm": float((steered[:, -1] - q[:, -1]).detach().float().norm()),
                                 "steered": bool(float(gain[0, -1]) != 0.0)})
        if steered is q:
            return output
        self.steered_calls += 1
        new = u_self + (steered - q)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    def __enter__(self):
        assert_dropout_disabled(self.bundle, [self.layer])
        lay = self.bundle.decoder_layer(self.layer)
        self._handles.append(lay.register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True))
        self._handles.append(lay.self_attn_layer_norm.register_forward_pre_hook(self._ln_pre_hook))
        self._handles.append(lay.self_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._h_in = self._cache_position = self._past_key_values = None
        return False


def cross_pulse_hook(bundle, layer: int, action_fn, nfp: int) -> DecoderPostCrossAttnInterventionHook:
    """Exactly the P2-R / DG-02 hook construction (alpha=1, NormPreserve, record last only)."""
    return DecoderPostCrossAttnInterventionHook(bundle, layer, None, alpha=1.0, num_forced_prefix=nfp, action_fn=action_fn,
                                                norm_preserve=True, record=True, record_last_only=True)


class Composite:
    """Enter context managers in order (pulse hook first, passive recorder after), exit in reverse."""

    def __init__(self, *items):
        self.items = [i for i in items if i is not None]

    def __enter__(self):
        for i in self.items:
            i.__enter__()
        return self

    def __exit__(self, *exc):
        for i in reversed(self.items):
            i.__exit__(*exc)
        return False


def assert_no_any_site_hooks(bundle) -> None:
    """DG-02 leak check plus the self-attention hooks this module installs."""
    leaked = []
    for i, lay in enumerate(bundle.model.model.decoder.layers):
        n = (len(lay.encoder_attn._forward_hooks) + len(lay.encoder_attn_layer_norm._forward_pre_hooks) + len(lay._forward_pre_hooks)
             + len(lay.self_attn._forward_hooks) + len(lay.self_attn_layer_norm._forward_pre_hooks))
        if n:
            leaked.append(f"decoder[{i}]:{n}")
    if leaked:
        raise RuntimeError(f"site hooks leaked on {leaked}")


def cache_fingerprint(cache: Any, length: int) -> str:
    """sha256 over every self-attention key/value prefix [:length] (bytes, the only cache content a pulse step can
    append to or crop) plus the shape/dtype of every static cross-attention key/value tensor."""
    h = hashlib.sha256()
    legacy = cache.to_legacy_cache()
    for layer in legacy:
        for j, x in enumerate(layer):
            if j < 2:
                t = x[..., :length, :].detach().contiguous().cpu()
                h.update(str(tuple(t.shape)).encode() + str(t.dtype).encode())
                h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())
            else:
                h.update(str(tuple(x.shape)).encode() + str(x.dtype).encode())
    return "sha256:" + h.hexdigest()
