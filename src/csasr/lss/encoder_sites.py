"""Exact Whisper encoder post-self-attention/pre-FFN site primitives.

The reusable ``EncoderSteeringHook`` attaches to an encoder-layer output,
which is after the FFN and is therefore not an exact analogue of the locked
decoder site.  BASIS-A3 uses this module instead: Whisper's encoder layer is
decomposed as ``r = q + u_self`` immediately after self-attention and before
the final/FFN sub-block.  The hook rewrites the self-attention output so the
layer's own residual add hands the repaired ``r`` to its FFN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..models.hooks import apply_steering

ENCODER_TENSOR = "encoder_post_self_attn_residual_pre_ffn"


def _first(output):
    return output[0] if isinstance(output, tuple) else output


def _rebuild(output, new):
    return (new,) + output[1:] if isinstance(output, tuple) else new


def _assert_encoder_eval(bundle, layers: Iterable[int]) -> None:
    if bundle.model.training:
        raise RuntimeError("encoder site reconstruction requires model.eval()")
    for layer in layers:
        mod = bundle.encoder_layer(int(layer))
        if float(getattr(mod, "dropout", 0.0)) and mod.training:
            raise RuntimeError(f"encoder layer {layer} has active dropout")
        if float(getattr(mod, "activation_dropout", 0.0)) and mod.training:
            raise RuntimeError(f"encoder layer {layer} has active activation dropout")


@dataclass
class EncoderSiteRecord:
    layer: int
    pre_norm: float
    post_norm: float
    edit_norm: float
    active_frames: int


class EncoderPostSelfAttnRecorder:
    """Capture ``q``, self-attention output, and exact encoder site ``r``."""

    def __init__(self, bundle, layers: Iterable[int], *, to_dtype=torch.float32):
        self.bundle = bundle
        self.layers = [int(x) for x in layers]
        self.to_dtype = to_dtype
        self.q_states = {}
        self.u_self_states = {}
        self.states = {}
        self._q = {}
        self._handles = []

    def _pre(self, layer):
        def hook(_module, args):
            self._q[layer] = args[0].detach()
        return hook

    def _attn(self, layer):
        def hook(_module, _inputs, output):
            q = self._q.pop(layer, None)
            if q is None:
                raise RuntimeError("encoder self-attention ran without its LN pre-hook")
            u = _first(output).detach()
            self.q_states[layer] = q.to(self.to_dtype)
            self.u_self_states[layer] = u.to(self.to_dtype)
            self.states[layer] = (q + u).to(self.to_dtype)
        return hook

    def __enter__(self):
        _assert_encoder_eval(self.bundle, self.layers)
        for layer in self.layers:
            mod = self.bundle.encoder_layer(layer)
            self._handles.append(mod.self_attn_layer_norm.register_forward_pre_hook(self._pre(layer)))
            self._handles.append(mod.self_attn.register_forward_hook(self._attn(layer)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._q.clear()
        return False

    def clear(self):
        self.q_states.clear()
        self.u_self_states.clear()
        self.states.clear()


class EncoderPostSelfAttnInterventionHook:
    """Steer one exact encoder site, with a frame-aligned gain mask."""

    def __init__(self, bundle, layer: int, direction: torch.Tensor, *,
                 alpha: float, scale: float = 1.0,
                 gain: torch.Tensor | None = None,
                 norm_preserve: bool = True, record: bool = False):
        layer = int(layer)
        if not 0 <= layer < int(bundle.num_encoder_layers):
            raise ValueError(f"encoder layer {layer} out of range")
        self.bundle = bundle
        self.layer = layer
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.gain = gain
        self.norm_preserve = bool(norm_preserve)
        self.record = bool(record)
        self.records: list[EncoderSiteRecord] = []
        self.calls = 0
        self.steered_calls = 0
        self._q = None
        self._handles = []

    def _pre(self, _module, args):
        self._q = args[0]

    def _hook(self, _module, _inputs, output):
        if self._q is None:
            raise RuntimeError("encoder self-attention ran without its LN pre-hook")
        u = _first(output)
        site = self._q + u
        gain = self.gain
        if gain is not None:
            gain = gain.to(device=site.device, dtype=site.dtype)
            gain = gain[:, :site.shape[1]]
            if tuple(gain.shape) != tuple(site.shape[:2]):
                raise ValueError(f"gain shape {tuple(gain.shape)} != {tuple(site.shape[:2])}")
        steered = apply_steering(site, self.direction, self.alpha, self.scale,
                                 gain, self.norm_preserve)
        self.calls += 1
        if self.record:
            pre = site.detach().float().norm(dim=-1)
            post = steered.detach().float().norm(dim=-1)
            edit = (steered - site).detach().float().norm(dim=-1)
            active = (torch.ones_like(pre, dtype=torch.bool) if gain is None
                      else gain.detach() > 0)
            self.records.append(EncoderSiteRecord(
                layer=self.layer,
                pre_norm=float(pre.mean()), post_norm=float(post.mean()),
                edit_norm=float(edit.sum()), active_frames=int(active.sum())))
        if steered is site:
            return output
        self.steered_calls += 1
        return _rebuild(output, u + (steered - site))

    def __enter__(self):
        _assert_encoder_eval(self.bundle, [self.layer])
        layer = self.bundle.encoder_layer(self.layer)
        self._handles.append(layer.self_attn_layer_norm.register_forward_pre_hook(self._pre))
        self._handles.append(layer.self_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._q = None
        return False


def encoder_site_report(bundle, layers: Iterable[int] | None = None) -> dict:
    """Return the frozen site description used by BASIS-A3 manifests."""
    return {
        "tensor": ENCODER_TENSOR,
        "module_path": "model.model.encoder.layers[l]",
        "pre_site": "args[0] of self_attn_layer_norm pre-hook",
        "self_attn_output": "output[0] of self_attn forward hook",
        "definition": "r_enc = q_enc + u_self",
        "intervention": "self_attn output += steer(r_enc) - r_enc; layer FFN is unchanged",
        "indexing": "zero-based encoder layer index",
        "layers": None if layers is None else [int(x) for x in layers],
        "dropout_precondition": "model.eval(); encoder dropout inactive",
    }
