"""Forward hooks for observation and rank-one activation steering.

Hook convention (guide section 18): the residual-stream *output* of transformer
block ``l``, i.e. what block ``l+1`` receives. For HuggingFace Whisper this is
element 0 of ``WhisperEncoderLayer``/``WhisperDecoderLayer``'s output tuple.

All hooks are context managers and always remove themselves.
"""
from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, Iterable, Sequence

import torch

from ..utils.logging import get_logger

log = get_logger(__name__)

EPS = 1e-6


def _split_output(output):
    """Return (hidden_states, rebuild_fn) for tuple-or-tensor layer outputs."""
    if isinstance(output, tuple):
        rest = output[1:]
        return output[0], (lambda new: (new,) + rest)
    return output, (lambda new: new)


class ActivationRecorder:
    """Observation-only hooks: capture block outputs without changing them."""

    def __init__(self, bundle, layers: Iterable[int], module: str = "encoder",
                 to_dtype: torch.dtype = torch.float32, keep_last_only: bool = False):
        self.bundle = bundle
        self.layers = list(layers)
        self.module = module
        self.to_dtype = to_dtype
        self.keep_last_only = keep_last_only
        self.states: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _make(self, layer_idx: int) -> Callable:
        def hook(_mod, _inp, output):
            hs, _ = _split_output(output)
            captured = hs.detach()
            if self.keep_last_only:
                captured = captured[:, -1:, :]
            self.states[layer_idx] = captured.to(self.to_dtype)
            return output  # unchanged

        return hook

    def __enter__(self) -> "ActivationRecorder":
        get = (self.bundle.encoder_layer if self.module == "encoder"
               else self.bundle.decoder_layer)
        for l in self.layers:
            self._handles.append(get(l).register_forward_hook(self._make(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def clear(self):
        self.states.clear()


def apply_steering(hidden: torch.Tensor, direction: torch.Tensor, alpha: float,
                   scale: float, gain: torch.Tensor | None,
                   norm_preserve: bool = True) -> torch.Tensor:
    """h_t <- h_t + alpha * s * g_t * d, optionally renormalised to ||h_t||.

    ``hidden``    (B, T, D)
    ``direction`` (D,) unit norm
    ``gain``      (B, T) in [0, 1], or None for "all positions"
    Positions with zero gain are returned bit-identical.
    """
    if alpha == 0.0:
        return hidden
    d = direction.to(hidden.device, hidden.dtype).view(1, 1, -1)
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
        steered = steered / (new_norm + EPS) * orig_norm
    return torch.where(active.unsqueeze(-1), steered, hidden)


class EncoderSteeringHook:
    """Add a direction to one encoder layer over a per-utterance frame mask."""

    def __init__(self, bundle, layer: int, direction: torch.Tensor, alpha: float,
                 scale: float, gain: torch.Tensor | None,
                 norm_preserve: bool = True):
        self.bundle = bundle
        self.layer = layer
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.gain = gain              # (B, T_frames) or None => global
        self.norm_preserve = norm_preserve
        self._handle = None
        self.calls = 0

    def _hook(self, _mod, _inp, output):
        hs, rebuild = _split_output(output)
        gain = self.gain
        if gain is not None:
            gain = gain[:, : hs.shape[1]]
            if gain.shape[0] != hs.shape[0]:
                raise ValueError("gain batch size does not match hidden states")
        self.calls += 1
        new = apply_steering(hs, self.direction, self.alpha, self.scale, gain,
                             self.norm_preserve)
        return rebuild(new)

    def __enter__(self):
        self._handle = self.bundle.encoder_layer(self.layer).register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class DecoderSteeringHook:
    """Per-layer decoder steering applied to generated (non-prefix) positions.

    The fixed special-token prefix is decoded in a single prefill call with
    ``T > 1``; every later call has ``T == 1`` and is the activation that
    produces the next content token. Prefill is therefore never steered.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, alpha: float,
                 scale: float, num_layers: int, norm_preserve: bool = True,
                 steer_prefill: bool = False):
        self.bundle = bundle
        self.layer = layer
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.num_layers = int(num_layers)
        self.norm_preserve = norm_preserve
        self.steer_prefill = steer_prefill
        self._handle = None
        self.calls = 0

    def _hook(self, _mod, _inp, output):
        hs, rebuild = _split_output(output)
        if hs.shape[1] > 1 and not self.steer_prefill:
            return output
        self.calls += 1
        eff_alpha = self.alpha / (self.num_layers ** 0.5)
        new = apply_steering(hs, self.direction, eff_alpha, self.scale, None,
                             self.norm_preserve)
        return rebuild(new)

    def __enter__(self):
        self._handle = self.bundle.decoder_layer(self.layer).register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class MultiHook(ExitStack):
    """Compose several hooks; all are removed on exit even if one raises."""

    def __init__(self, hooks: Sequence):
        super().__init__()
        self.hooks = list(hooks)

    def __enter__(self):
        super().__enter__()
        for h in self.hooks:
            self.enter_context(h)
        return self


def assert_no_hooks(bundle) -> None:
    """Fail loudly if a previous run leaked hooks (guide section 19.8)."""
    leaked = []
    for name, layers in (("encoder", bundle.model.model.encoder.layers),
                         ("decoder", bundle.model.model.decoder.layers)):
        for i, layer in enumerate(layers):
            if len(layer._forward_hooks) > 0:
                leaked.append(f"{name}[{i}]:{len(layer._forward_hooks)}")
    if leaked:
        raise RuntimeError(f"forward hooks leaked on {leaked}")
