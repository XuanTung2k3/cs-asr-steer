"""The exact tensors the method reads and writes.

The proposal defines the decoder intervention on the residual stream *after
cross-attention*. The existing `csasr.models.hooks.DecoderSteeringHook` attaches
to the output of a whole `WhisperDecoderLayer`, i.e. after the feed-forward
block -- a different representation. Constructing a direction in one space and
adding it in another is not a naming difference, so this module implements the
site the proposal actually specifies.

In `transformers` the cross-attention block of `WhisperDecoderLayer.forward` is:

    residual = hidden_states
    hidden_states = self.encoder_attn_layer_norm(hidden_states)
    hidden_states, cross_attn_weights = self.encoder_attn(...)
    hidden_states = nn.functional.dropout(hidden_states, ...)
    hidden_states = residual + hidden_states          # <-- the site

so the site is reconstructed from two hooks: a forward *pre*-hook on
`encoder_attn_layer_norm` (whose first argument is exactly `residual`) and a
forward hook on `encoder_attn` (whose first output is the attention result).
Intervening from the `encoder_attn` hook -- by returning
`attn_out + (steer(site) - site)` -- makes the layer compute `steer(site)`
itself, so no assumption about what happens downstream is baked in.

Dropout is identity in `eval()`; `assert_dropout_disabled` enforces that rather
than assuming it.

A useful side effect: `encoder_attn` also returns its cross-attention weights,
so the same hook yields per-step attention at kilobytes per utterance instead of
the ~2.9 GB that `output_attentions=True` materialises for every layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from ..models.hooks import apply_steering
from ..utils.logging import get_logger

log = get_logger(__name__)

ENCODER_TENSOR = "encoder_block_output"
DECODER_TENSOR = "decoder_post_cross_attn_residual"
DECODER_REJECTED_TENSOR = "decoder_block_output"


@dataclass(frozen=True)
class SiteSpec:
    """Where an intervention happens, in a form that can be frozen and hashed."""

    module: str          # "encoder" | "decoder"
    tensor: str          # ENCODER_TENSOR | DECODER_TENSOR
    layer: int

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "tensor": self.tensor, "layer": int(self.layer)}


def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _rebuild(output: Any, new: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new,) + output[1:]
    return new


def assert_dropout_disabled(bundle, layers: Iterable[int]) -> None:
    """The site identity holds only when dropout is inactive."""
    if bundle.model.training:
        raise RuntimeError("model is in training mode; the decoder site is only "
                           "reconstructible under model.eval()")
    for k in layers:
        layer = bundle.decoder_layer(int(k))
        if float(getattr(layer, "dropout", 0.0)) and layer.training:
            raise RuntimeError(f"decoder layer {k} has active dropout")


class DecoderPostCrossAttnRecorder:
    """Capture the post-cross-attention residual (and optionally its weights)."""

    def __init__(self, bundle, layers: Iterable[int], *,
                 to_dtype: torch.dtype = torch.float32,
                 keep_attention: bool = False, keep_last_only: bool = False):
        self.bundle = bundle
        self.layers = [int(k) for k in layers]
        self.to_dtype = to_dtype
        self.keep_attention = keep_attention
        self.keep_last_only = keep_last_only
        self.states: dict[int, torch.Tensor] = {}
        self.attentions: dict[int, torch.Tensor] = {}
        self._residual: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def _pre_hook(self, layer: int):
        def hook(_mod, args):
            self._residual[layer] = args[0].detach()
            return None
        return hook

    def _post_hook(self, layer: int):
        def hook(_mod, _inp, output):
            residual = self._residual.get(layer)
            if residual is None:
                raise RuntimeError(
                    f"decoder layer {layer}: cross-attention ran without its "
                    "layer-norm pre-hook; the site cannot be reconstructed")
            attn_out = _first(output).detach()
            site = residual + attn_out
            if self.keep_last_only:
                site = site[:, -1:, :]
            self.states[layer] = site.to(self.to_dtype)
            if self.keep_attention and isinstance(output, tuple) and len(output) > 1:
                weights = output[1]
                if weights is not None:
                    self.attentions[layer] = weights.detach().to(self.to_dtype)
            return output
        return hook

    def __enter__(self) -> "DecoderPostCrossAttnRecorder":
        assert_dropout_disabled(self.bundle, self.layers)
        for k in self.layers:
            layer = self.bundle.decoder_layer(k)
            self._handles.append(
                layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook(k)))
            self._handles.append(
                layer.encoder_attn.register_forward_hook(self._post_hook(k)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._residual.clear()
        return False

    def clear(self) -> None:
        self.states.clear()
        self.attentions.clear()


class DecoderPostCrossAttnSteeringHook:
    """Add a direction to the post-cross-attention residual of one layer.

    Unlike `models.hooks.DecoderSteeringHook` this does **not** divide the
    strength by sqrt(num_layers): that implicit rescale would make rho=1 mean
    something different at the decoder than at the encoder, and the frozen
    action compares the two.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, alpha: float,
                 scale: float, gain: torch.Tensor | None = None,
                 norm_preserve: bool = True, steer_prefill: bool = False):
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.gain = gain                   # (B, T) aligned to this call, or None
        self.norm_preserve = norm_preserve
        self.steer_prefill = steer_prefill
        self.calls = 0
        self.steered_calls = 0
        self._residual: torch.Tensor | None = None
        self._handles: list = []

    def _pre_hook(self, _mod, args):
        self._residual = args[0]
        return None

    def _hook(self, _mod, _inp, output):
        if self._residual is None:
            raise RuntimeError("cross-attention ran without its layer-norm pre-hook")
        attn_out = _first(output)
        site = self._residual + attn_out
        self.calls += 1
        if site.shape[1] > 1 and not self.steer_prefill:
            return output
        gain = self.gain
        if gain is not None:
            gain = gain.to(site.device)[:, : site.shape[1]]
            if gain.shape[0] != site.shape[0]:
                raise ValueError("gain batch size does not match decoder states")
        steered = apply_steering(site, self.direction, self.alpha, self.scale,
                                 gain, self.norm_preserve)
        if steered is site:
            return output
        self.steered_calls += 1
        return _rebuild(output, attn_out + (steered - site))

    def __enter__(self) -> "DecoderPostCrossAttnSteeringHook":
        assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(
            layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._residual = None
        return False


def assert_no_site_hooks(bundle) -> None:
    """Fail loudly if a previous run leaked cross-attention hooks."""
    leaked = []
    for i, layer in enumerate(bundle.model.model.decoder.layers):
        n = len(layer.encoder_attn._forward_hooks) + \
            len(layer.encoder_attn_layer_norm._forward_pre_hooks)
        if n:
            leaked.append(f"decoder[{i}]:{n}")
    if leaked:
        raise RuntimeError(f"decoder cross-attention hooks leaked on {leaked}")


def site_report(bundle, layers: Sequence[int] | None = None) -> dict[str, Any]:
    """Machine-readable description of both sites, for the spec freeze."""
    decoder0 = bundle.decoder_layer(0)
    return {
        "encoder": {
            "module_path": "model.model.encoder.layers[l]",
            "tensor": ENCODER_TENSOR,
            "capture": "forward_hook, output[0]",
            "class": type(bundle.encoder_layer(0)).__name__,
            "axis": "encoder frames, 0-based; valid = bundle.valid_frames(duration_sec)",
        },
        "decoder": {
            "module_path": "model.model.decoder.layers[k]",
            "tensor": DECODER_TENSOR,
            "definition": "residual_in + encoder_attn(encoder_attn_layer_norm(residual_in))",
            "capture": {
                "residual_in": "forward_pre_hook on encoder_attn_layer_norm, args[0]",
                "attn_out": "forward_hook on encoder_attn, output[0]",
            },
            "intervention": ("encoder_attn hook returns "
                             "attn_out + (steer(residual_in + attn_out) - (residual_in + attn_out))"),
            "rejected_site": DECODER_REJECTED_TENSOR,
            "rejected_site_reason": "models.hooks.DecoderSteeringHook steers after the FFN block",
            "depth_rescale": False,
            "requires": ["model.eval()", "dropout inactive"],
            "cross_attn_class": type(decoder0.encoder_attn).__name__,
            "layer_norm_class": type(decoder0.encoder_attn_layer_norm).__name__,
            "layers": list(layers) if layers is not None else None,
        },
    }


@torch.inference_mode()
def assert_site_reconstruction(bundle, forward_fn, layer: int, *,
                               alpha: float = 0.5, scale: float = 1.0,
                               tol: float = 1e-3, ulp_tol: float = 4.0) -> dict[str, Any]:
    """Verify on the real model that steering the site does exactly what it says.

    ``forward_fn`` runs one teacher-forced forward pass with no hooks of its
    own. Two properties are checked:

    * the site captured under an active steering hook equals the unsteered site
      plus `alpha * scale * direction` (norm preservation off), and
    * the site is *not* the decoder block output, so the intervention is
      demonstrably happening at the tensor the proposal names.

    The identity is exact in real arithmetic, so the residual error is pure
    rounding: the hook rewrites the cross-attention output and the layer then
    recomputes `residual + attn_out` in the model dtype. Measured on a fixed
    tiny model the error tracks the dtype epsilon and nothing else -- 6.7 ULP in
    float32, 2.9 in float16, 1.5 in bfloat16 -- so an absolute `tol` alone
    states a different requirement at each precision, and at bfloat16
    (eps 7.8e-3) a 1e-3 tolerance is not satisfiable by any correct
    implementation. Two scale-free quantities are therefore reported alongside
    it:

    * ``rel_err_ulp`` -- error in units of the dtype's epsilon, and
    * ``err_vs_block_gap`` -- error relative to how far the site sits from the
      decoder block output.

    The second is the one that actually discriminates: hooking the wrong tensor
    or getting the arithmetic wrong misses by the size of that gap, so a correct
    implementation sits orders of magnitude below 1 no matter the precision.
    """
    from ..models.hooks import ActivationRecorder

    with DecoderPostCrossAttnRecorder(bundle, [layer]) as site_rec, \
            ActivationRecorder(bundle, [layer], module="decoder") as block_rec:
        forward_fn()
        base_site = site_rec.states[layer].clone()
        base_block = block_rec.states[layer].clone()

    direction = torch.zeros(bundle.d_model, dtype=torch.float32)
    direction[0] = 1.0

    hook = DecoderPostCrossAttnSteeringHook(
        bundle, layer, direction, alpha=alpha, scale=scale,
        norm_preserve=False, steer_prefill=True)
    with hook, DecoderPostCrossAttnRecorder(bundle, [layer]) as steered_rec:
        forward_fn()
        steered_site = steered_rec.states[layer].clone()

    expected = base_site.clone()
    expected[..., 0] += alpha * scale
    delta = float((steered_site - expected).abs().max())
    scale_ref = float(base_site.abs().max()) or 1.0
    site_vs_block = float((base_site - base_block).abs().max())

    assert_no_site_hooks(bundle)
    rel_err = delta / scale_ref
    # The recorder upcasts what it captures to float32, so the recorded tensor's
    # dtype says nothing about the precision the arithmetic ran at. Take it from
    # the weights of the layer that was actually hooked.
    try:
        compute_dtype = next(bundle.decoder_layer(layer).parameters()).dtype
    except (StopIteration, AttributeError):                    # pragma: no cover
        compute_dtype = getattr(bundle, "dtype", base_site.dtype)
    eps = float(torch.finfo(compute_dtype).eps)
    tol_eff = max(float(tol), float(ulp_tol) * eps)
    report = {
        "layer": int(layer),
        "abs_err": delta,
        "rel_err": rel_err,
        "dtype": str(compute_dtype),
        "dtype_eps": eps,
        "rel_err_ulp": rel_err / eps,
        "err_vs_block_gap": delta / (site_vs_block or float("inf")),
        "tol": float(tol),
        "tol_effective": tol_eff,
        "reconstruction_ok": bool(rel_err <= tol_eff),
        "site_vs_block_max_abs_diff": site_vs_block,
        "site_differs_from_block_output": bool(site_vs_block > 0.0),
        "steered_calls": int(hook.steered_calls),
        "tensor": DECODER_TENSOR,
    }
    if not report["reconstruction_ok"]:
        log.error("decoder site reconstruction failed: %s", report)
    return report
