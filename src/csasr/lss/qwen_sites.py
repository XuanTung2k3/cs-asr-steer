"""Frozen Qwen3-ASR intervention sites for BASIS-A4.

Qwen3-ASR is a decoder-only multimodal model.  The audio tower and the text
decoder each expose a residual immediately before their feed-forward module;
these hooks modify that tensor through the module's existing pre-LayerNorm
input.  No projection, whole-block output, or audio-placeholder embedding is
treated as a site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from ..models.hooks import apply_steering

AUDIO_SITE = "qwen_audio_encoder_post_self_attn_residual_pre_ffn"
TEXT_SITE = "qwen_text_decoder_post_self_attn_residual_pre_mlp"


def _cache_length(cache: Any) -> int:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    try:
        return int(cache[0][0].shape[2])
    except Exception:
        return 0


def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _as_3d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        return x.unsqueeze(0), True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"site tensor must be (T,D) or (B,T,D), got {tuple(x.shape)}")


def _restore(x: torch.Tensor, flat: bool) -> torch.Tensor:
    return x[0] if flat else x


@dataclass
class QwenSiteRecord:
    layer: int
    site: str
    n_positions: int
    active_positions: int
    pre_norm_mean: float
    post_norm_mean: float
    edit_norm_sum: float
    active_pre_norm_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer), "site": self.site,
            "n_positions": int(self.n_positions),
            "active_positions": int(self.active_positions),
            "pre_norm_mean": float(self.pre_norm_mean),
            "post_norm_mean": float(self.post_norm_mean),
            "edit_norm_sum": float(self.edit_norm_sum),
            "active_pre_norm_sum": float(self.active_pre_norm_sum),
        }


class _QwenSiteBase:
    def __init__(self, bundle, layers: Iterable[int], *, site: str):
        self.bundle = bundle
        self.layers = [int(x) for x in layers]
        self.site = site
        self.states: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _modules(self):
        raise NotImplementedError

    def _capture(self, layer):
        def hook(_module, args):
            if not args:
                raise RuntimeError(f"{self.site} L{layer}: missing residual input")
            self.states[layer] = args[0].detach().float().cpu()
            return None
        return hook

    def __enter__(self):
        if self.bundle.model.training:
            raise RuntimeError("Qwen site capture requires model.eval()")
        for layer, module in zip(self.layers, self._modules()):
            self._handles.append(module.register_forward_pre_hook(self._capture(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


class QwenAudioSiteRecorder(_QwenSiteBase):
    """Capture post-self-attention residuals before the audio FFN."""

    def __init__(self, bundle, layers: Iterable[int]):
        super().__init__(bundle, layers, site=AUDIO_SITE)

    def _modules(self):
        return [self.bundle.audio_layer(x).final_layer_norm for x in self.layers]


class QwenTextSiteRecorder(_QwenSiteBase):
    """Capture post-self-attention residuals before the text MLP."""

    def __init__(self, bundle, layers: Iterable[int]):
        super().__init__(bundle, layers, site=TEXT_SITE)

    def _modules(self):
        return [self.bundle.text_layer(x).post_attention_layernorm for x in self.layers]


class QwenSiteInterventionHook:
    """Norm-preserving steering at one exact Qwen site.

    ``gain`` is aligned to the tensor passed to the site: ``(T,)`` for the
    packed audio tower or ``(B,T)`` for the text decoder.  A callable gain may
    be supplied for generation and receives the site's absolute cache
    positions.  Inactive positions are returned bit-identically.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, *,
                 alpha: float, site: str, scale: float = 1.0,
                 gain: torch.Tensor | None = None,
                 gain_fn=None, norm_preserve: bool = True,
                 allowed_positions: set[int] | None = None,
                 excluded_token_ids: set[int] | None = None,
                 record: bool = True):
        if site not in (AUDIO_SITE, TEXT_SITE):
            raise ValueError(f"unknown Qwen site {site!r}")
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction
        self.alpha = float(alpha)
        self.site = site
        self.scale = float(scale)
        self.gain = gain
        self.gain_fn = gain_fn
        self.norm_preserve = bool(norm_preserve)
        self.allowed_positions = allowed_positions
        self.excluded_token_ids = set(excluded_token_ids or ())
        self.record = bool(record)
        self.records: list[QwenSiteRecord] = []
        self.states: list[torch.Tensor] = []
        self.calls = 0
        self.steered_calls = 0
        self._cache_position = None
        self._past_key_values = None
        self._input_ids = None
        self._handles: list[Any] = []

    @property
    def _layer(self):
        return (self.bundle.audio_layer(self.layer) if self.site == AUDIO_SITE
                else self.bundle.text_layer(self.layer))

    def _layer_pre(self, _module, args, kwargs):
        self._cache_position = kwargs.get("cache_position")
        self._past_key_values = kwargs.get("past_key_values")
        return None

    def _model_pre(self, _module, args, kwargs):
        self._input_ids = kwargs.get("input_ids")
        if self._input_ids is None and args:
            self._input_ids = args[0]
        return None

    def _positions(self, n: int, device) -> torch.Tensor:
        if self._cache_position is not None:
            p = self._cache_position.detach().to(device=device).long().reshape(-1)
            if p.numel() >= n:
                return p[:n]
        start = _cache_length(self._past_key_values)
        return torch.arange(start, start + n, device=device, dtype=torch.long)

    def _resolve_gain(self, site: torch.Tensor, pos: torch.Tensor):
        x, flat = _as_3d(site)
        b, t = x.shape[:2]
        if self.gain_fn is not None:
            g = self.gain_fn(site=site, abs_pos=pos)
        elif self.gain is not None:
            g = self.gain.to(device=site.device, dtype=site.dtype)
        else:
            g = torch.ones((b, t), device=site.device, dtype=site.dtype)
        g = g if torch.is_tensor(g) else torch.as_tensor(g, device=site.device, dtype=site.dtype)
        if g.ndim == 0:
            g = g.expand(b, t)
        elif g.ndim == 1:
            if flat and g.shape[0] == t:
                g = g.view(1, t)
            elif g.shape[0] == b:
                g = g[:, None].expand(b, t)
            else:
                raise ValueError(f"gain shape {tuple(g.shape)} incompatible with {tuple(x.shape)}")
        elif g.ndim == 2 and tuple(g.shape) == (b, t):
            pass
        else:
            raise ValueError(f"gain shape {tuple(g.shape)} incompatible with {tuple(x.shape)}")
        if self.allowed_positions is not None:
            allow = torch.tensor([int(p) in self.allowed_positions for p in pos],
                                 device=site.device, dtype=torch.bool)
            g = g * allow.view(1, t).to(g.dtype)
        if self.excluded_token_ids and self._input_ids is not None:
            ids = self._input_ids.to(device=site.device).long()
            if ids.ndim == 1: ids = ids.unsqueeze(0)
            if ids.shape[-1] >= t:
                ids = ids[:, -t:]
                keep = torch.ones_like(ids, dtype=torch.bool)
                for token_id in self.excluded_token_ids:
                    keep &= ids != int(token_id)
                g = g * keep.to(g.dtype)
        return g, flat

    def _hook(self, _module, args):
        if not args:
            raise RuntimeError(f"{self.site} L{self.layer}: missing site input")
        site = args[0]
        x, flat = _as_3d(site)
        pos = self._positions(x.shape[1], x.device)
        gain, _ = self._resolve_gain(site, pos)
        steered = apply_steering(x, self.direction, self.alpha, self.scale,
                                 gain, self.norm_preserve)
        self.calls += 1
        self.states.append(site.detach().float().cpu())
        if self.record:
            pre = x.detach().float().norm(dim=-1)
            post = steered.detach().float().norm(dim=-1)
            edit = (steered - x).detach().float().norm(dim=-1)
            active = gain > 0
            self.records.append(QwenSiteRecord(
                self.layer, self.site, int(pre.numel()), int((gain > 0).sum()),
                float(pre.mean()), float(post.mean()), float(edit.sum()),
                float(pre.masked_select(active).sum())))
        if steered is x:
            return (site,)
        self.steered_calls += 1
        return (_restore(steered, flat),)

    def __enter__(self):
        if self.bundle.model.training:
            raise RuntimeError("Qwen site intervention requires model.eval()")
        self._handles.append(self._layer.register_forward_pre_hook(
            self._layer_pre, with_kwargs=True))
        text_model = getattr(self.bundle, "text_model", None)
        if self.site == TEXT_SITE and text_model is not None:
            self._handles.append(text_model.register_forward_pre_hook(
                self._model_pre, with_kwargs=True))
        target = (self._layer.final_layer_norm if self.site == AUDIO_SITE
                  else self._layer.post_attention_layernorm)
        self._handles.append(target.register_forward_pre_hook(self._hook))
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache_position = None
        self._past_key_values = None
        self._input_ids = None
        return False


def qwen_site_report(bundle, *, encoder_layers=None, decoder_layers=None) -> dict[str, Any]:
    return {
        "encoder": {"module_path": "model.thinker.audio_tower.layers[l]",
                    "tensor": AUDIO_SITE,
                    "capture": "pre-hook on final_layer_norm input",
                    "definition": "residual + self_attn(self_attn_layer_norm(residual))",
                    "layers": None if encoder_layers is None else list(map(int, encoder_layers))},
        "text_decoder": {"module_path": "model.thinker.model.layers[l]",
                          "tensor": TEXT_SITE,
                          "capture": "pre-hook on post_attention_layernorm input",
                          "definition": "residual + self_attn(input_layernorm(residual))",
                          "layers": None if decoder_layers is None else list(map(int, decoder_layers))},
        "depth_rescale": False,
        "norm_preserve": True,
    }
