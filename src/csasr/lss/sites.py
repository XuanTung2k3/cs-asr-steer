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

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from ..models.hooks import apply_steering
from ..utils.logging import get_logger

log = get_logger(__name__)

ENCODER_TENSOR = "encoder_block_output"
DECODER_TENSOR = "decoder_post_cross_attn_residual"
DECODER_REJECTED_TENSOR = "decoder_block_output"

# The only decoder layers this contract permits at the site (MC §3). Layer
# *selection* between the two is DG-03; DG-02 refuses anything else on the
# contract path so a stray index cannot slip through as a plumbing default.
CONTRACT_DECODER_LAYERS = (16, 24)


def _cache_length(past_key_values: Any) -> int:
    """Absolute number of decoder positions already cached (prefill => 0).

    Mirrors the reusable ``steer_sweep.hooks.cache_length`` (spec §5) without a
    cross-package import from a core library module. Only used as the secondary
    cross-check for the absolute position; the primary source is the layer's own
    ``cache_position`` kwarg, which this transformers version always supplies.
    """
    if past_key_values is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:                                        # pragma: no cover
            pass
    try:                                                         # legacy tuple cache
        return int(past_key_values[0][0].shape[2])
    except Exception:                                            # pragma: no cover
        return 0


def num_forced_prefix_from(processor, language: str = "zh",
                           task: str = "transcribe") -> int:
    """The single source of truth for the forced-prefix width (spec §7).

    Derived dynamically from ``len(build_prefix(...))`` rather than a hard-coded
    4, so a config change to the forced prompt cannot silently desynchronise the
    steering-eligibility boundary from the tokens Whisper is actually forced to
    emit.
    """
    from ..data.alignment import build_prefix
    return len(build_prefix(processor, language=language, task=task))


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
    """Capture the post-cross-attention residual (and optionally its weights).

    The forward decomposition of the site (spec §2) is exposed separately:

    * ``q_states[k]``        -- ``q``, the ``encoder_attn_layer_norm`` pre-hook
      ``args[0]``: the post-self-attention decoder-side state *before* the
      cross-attention residual add (line 517 ``residual``).
    * ``u_source_states[k]`` -- ``u_source``, the ``encoder_attn`` ``output[0]``:
      the source-conditioned cross-attention contribution (line 519).
    * ``states[k]``          -- ``r = q + u_source``, the site the FFN consumes
      (line 528). Kept under the historical name so existing callers still read
      the site here.

    This is a detached analysis mode: every stored tensor is ``.detach()``-ed and
    upcast, so no autograd graph is retained (spec §9). ``u_source`` is taken
    directly from the cross-attention module output -- never approximated from an
    unrelated hidden state -- so ``r == q + u_source`` holds to dtype rounding.
    """

    def __init__(self, bundle, layers: Iterable[int], *,
                 to_dtype: torch.dtype = torch.float32,
                 keep_attention: bool = False, keep_last_only: bool = False):
        self.bundle = bundle
        self.layers = [int(k) for k in layers]
        self.to_dtype = to_dtype
        self.keep_attention = keep_attention
        self.keep_last_only = keep_last_only
        self.states: dict[int, torch.Tensor] = {}
        self.q_states: dict[int, torch.Tensor] = {}
        self.u_source_states: dict[int, torch.Tensor] = {}
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
            q = self._residual.get(layer)
            if q is None:
                raise RuntimeError(
                    f"decoder layer {layer}: cross-attention ran without its "
                    "layer-norm pre-hook; the site cannot be reconstructed")
            u_source = _first(output).detach()
            site = q + u_source
            if self.keep_last_only:
                q = q[:, -1:, :]
                u_source = u_source[:, -1:, :]
                site = site[:, -1:, :]
            self.q_states[layer] = q.to(self.to_dtype)
            self.u_source_states[layer] = u_source.to(self.to_dtype)
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
        self.q_states.clear()
        self.u_source_states.clear()
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


@dataclass
class AuditRecord:
    """One compact, JSON-serialisable steering record (spec §10).

    Records raw eligibility quantities only. It deliberately does **not** compute
    a ``gate_coverage`` ratio: that denominator is deferred to the gate ticket
    (DG-01 spec §4 / MC §6).
    """

    layer: int
    abs_pos: int
    row: int
    is_forced_prefix: bool
    gate: float
    alpha: float
    pre_norm: float
    post_norm: float
    edit_norm: float
    steered: bool
    beam_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "abs_pos": int(self.abs_pos),
            "row": int(self.row),
            "is_forced_prefix": bool(self.is_forced_prefix),
            "gate": float(self.gate),
            "alpha": float(self.alpha),
            "pre_norm": float(self.pre_norm),
            "post_norm": float(self.post_norm),
            "edit_norm": float(self.edit_norm),
            "steered": bool(self.steered),
            "beam_index": None if self.beam_index is None else int(self.beam_index),
        }


# gate_fn signature: sees each row's OWN (q, u_source, r) and the absolute
# positions, so a beam's gate is intrinsically its own -- never fabricated from a
# beam index (spec §6). Returns a gain of shape (B,), (B, T), or a scalar.
GateFn = Callable[..., torch.Tensor]
ActionFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class DecoderPostCrossAttnInterventionHook:
    """Canonical DG-02 exact-site intervention (post-cross-attn, pre-FFN).

    Repairs the site ``r = q + u_source`` (line 528 of ``WhisperDecoderLayer``)
    and hands the repaired ``r̃`` to the FFN, by returning
    ``u_source + (r̃ - r)`` from the ``encoder_attn`` hook so the layer's own
    ``residual + attn_out`` recomputes to ``r̃`` (no downstream assumption). The
    final block output (line 537) is never the intervention target.

    This is infrastructure only. The steering *direction*/edit tensor and the
    *gate* are supplied externally -- nothing here hard-codes ``v_nat``, the
    conditioning-residualized direction, the disagreement score, the factorized
    gate, the layer choice, or ``β``. Those are DG-03+ decisions.

    Effective edit per row/position (MC §2): ``ũ^S = α · scale · g · d``, then
    ``r̃ = NormPreserve(r + ũ^S)`` via the contract-approved
    ``models.hooks.apply_steering`` (no ``sqrt(num_layers)`` rescale).

    * ``num_forced_prefix`` -- forced-prefix width (spec §7); every position with
      ``abs_pos < num_forced_prefix`` gets a zero effective edit. Pass
      ``num_forced_prefix_from(processor, ...)``; there is no hard-coded 4.
    * ``gate_fn`` / ``gain`` -- per-token, per-row gate (spec §6). ``gate_fn`` is
      called with the row's own ``(q, u_source, r, abs_pos)``.
    * ``mode`` -- ``"steer"`` (inference) or ``"train"`` (gradient-enabled: the
      steering path is not detached, so gradients reach trainable
      direction/gain/alpha while the frozen backbone stays frozen, spec §9).
    * ``record`` -- emit the §10 audit schema into ``self.records`` (detached,
      no autograd graph retained).

    Absolute decode position comes from the layer's own ``cache_position`` kwarg
    (primary), falling back to the KV-cache length (spec §5); it is never reset
    to zero on cached calls.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor | None, *,
                 alpha: float, num_forced_prefix: int, scale: float = 1.0,
                 gate_fn: GateFn | None = None, gain: torch.Tensor | None = None,
                 action_fn: ActionFn | None = None,
                 norm_preserve: bool = True, mode: str = "steer",
                 record: bool = False, record_last_only: bool = True,
                 enforce_contract_layer: bool = False):
        if mode not in ("steer", "train"):
            raise ValueError(f"mode must be 'steer' or 'train', got {mode!r}")
        layer = int(layer)
        if not (0 <= layer < int(bundle.num_decoder_layers)):
            raise ValueError(
                f"decoder layer {layer} out of range [0, {bundle.num_decoder_layers})")
        if enforce_contract_layer and layer not in CONTRACT_DECODER_LAYERS:
            raise ValueError(
                f"layer {layer} is not a contract candidate {CONTRACT_DECODER_LAYERS}; "
                "layer selection is DG-03")
        if direction is None and action_fn is None:
            raise ValueError("provide direction or action_fn")
        if direction is not None and action_fn is not None:
            raise ValueError("provide direction or action_fn, not both")
        if action_fn is not None and (gate_fn is not None or gain is not None):
            raise ValueError("action_fn cannot be combined with gate_fn or gain")
        if gate_fn is not None and gain is not None:
            raise ValueError("pass gate_fn or gain, not both")
        if int(num_forced_prefix) < 0:
            raise ValueError("num_forced_prefix must be >= 0")
        self.bundle = bundle
        self.layer = layer
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.action_fn = action_fn
        self.gate_fn = gate_fn
        self.gain = gain
        self.num_forced_prefix = int(num_forced_prefix)
        self.norm_preserve = norm_preserve
        self.mode = mode
        self.record = record
        self.record_last_only = record_last_only
        self.calls = 0
        self.steered_calls = 0
        self.records: list[AuditRecord] = []
        self._q: torch.Tensor | None = None
        self._cache_position: torch.Tensor | None = None
        self._past_key_values: Any = None
        self._handles: list = []
        self._installed = False

    # -- position bookkeeping ------------------------------------------------
    def _layer_pre_hook(self, _mod, args, kwargs):
        # Runs at the start of the target layer's forward, before q is formed.
        self._cache_position = kwargs.get("cache_position", None)
        self._past_key_values = kwargs.get("past_key_values", None)
        return None

    def _abs_positions(self, T: int, device) -> torch.Tensor:
        cp = self._cache_position
        if cp is not None:
            pos = cp.detach().to(device=device).long().reshape(-1)
            if pos.numel() >= T:
                return pos[:T]
        start = _cache_length(self._past_key_values)
        return torch.arange(start, start + T, device=device, dtype=torch.long)

    # -- q capture -----------------------------------------------------------
    def _pre_hook(self, _mod, args):
        # Not detached: in train mode the graph from q must stay intact so a
        # state-dependent gate can receive gradient (spec §9).
        self._q = args[0]
        return None

    # -- gate assembly -------------------------------------------------------
    def _coerce_gain(self, g, site, eligible):
        B, T = site.shape[0], site.shape[1]
        # Preserve autograd when the gate is a trainable tensor (spec §9);
        # only wrap plain scalars/sequences.
        g = g.to(site.device, site.dtype) if torch.is_tensor(g) \
            else torch.as_tensor(g, device=site.device, dtype=site.dtype)
        if g.ndim == 0:
            g = g.reshape(1, 1).expand(B, T)
        elif g.ndim == 1:                      # per-row (B,) -> (B, T)
            if g.shape[0] != B:
                raise ValueError(f"gain rows {g.shape[0]} != decoder rows {B}")
            g = g.unsqueeze(1).expand(B, T)
        elif g.ndim == 2:
            g = g[:, :T]
            if g.shape[0] != B:
                raise ValueError(f"gain rows {g.shape[0]} != decoder rows {B}")
        else:
            raise ValueError(f"gate must be scalar/(B,)/(B,T); got shape {tuple(g.shape)}")
        # Forced-prefix positions always receive a zero effective edit (spec §7).
        return g * eligible.to(site.dtype).view(1, T)

    def _resolve_gain(self, q, u_source, site, abs_pos, eligible):
        B, T = site.shape[0], site.shape[1]
        if self.gate_fn is not None:
            raw = self.gate_fn(q=q, u_source=u_source, r=site, abs_pos=abs_pos)
        elif self.gain is not None:
            raw = self.gain
        else:
            raw = torch.ones(B, T, device=site.device, dtype=site.dtype)
        return self._coerce_gain(raw, site, eligible)

    def _resolve_direction(self, q, u_source, site, abs_pos):
        if self.direction is None:
            raise RuntimeError("fixed direction is unavailable without action_fn output")
        return self.direction

    def _emit_records(self, abs_pos, eligible, gain, site, steered):
        B, T = site.shape[0], site.shape[1]
        cols = [T - 1] if self.record_last_only else range(T)
        pre = site.detach().float().norm(dim=-1)          # (B, T)
        post = steered.detach().float().norm(dim=-1)
        edit = (steered - site).detach().float().norm(dim=-1)
        elig = eligible.detach().cpu()
        gain_d = gain.detach().float().cpu()
        pos = abs_pos.detach().cpu()
        for t in cols:
            for b in range(B):
                g_bt = float(gain_d[b, t])
                self.records.append(AuditRecord(
                    layer=self.layer, abs_pos=int(pos[t]), row=b,
                    is_forced_prefix=not bool(elig[t]), gate=g_bt,
                    alpha=self.alpha, pre_norm=float(pre[b, t]),
                    post_norm=float(post[b, t]), edit_norm=float(edit[b, t]),
                    steered=bool(g_bt != 0.0 and self.alpha != 0.0),
                    beam_index=b))

    # -- the site intervention ----------------------------------------------
    def _hook(self, _mod, _inp, output):
        if self._q is None:
            raise RuntimeError("cross-attention ran without its layer-norm pre-hook")
        u_source = _first(output)
        q = self._q
        site = q + u_source                          # r = q + u_source (line 528)
        self.calls += 1
        T = site.shape[1]
        abs_pos = self._abs_positions(T, site.device)
        eligible = abs_pos >= self.num_forced_prefix            # (T,)
        if self.action_fn is not None:
            gain, direction = self.action_fn(
                q=q, u_source=u_source, r=site, abs_pos=abs_pos)
            if not torch.is_tensor(gain) or not torch.is_tensor(direction):
                raise TypeError("action_fn must return (gain_tensor, direction_tensor)")
            gain = self._coerce_gain(gain, site, eligible)
        else:
            gain = self._resolve_gain(q, u_source, site, abs_pos, eligible)
            direction = self._resolve_direction(q, u_source, site, abs_pos)
        steered = apply_steering(site, direction, self.alpha, self.scale,
                                 gain, self.norm_preserve)
        if self.record:
            self._emit_records(abs_pos, eligible, gain, site, steered)
        if steered is site:                          # α=0 or no eligible/positive gain
            return output
        self.steered_calls += 1
        return _rebuild(output, u_source + (steered - site))

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "DecoderPostCrossAttnInterventionHook":
        if self._installed:
            raise RuntimeError("intervention hook already installed; refusing to "
                               "double-install on the same site")
        assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(
            layer.register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True))
        self._handles.append(
            layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        self._installed = True
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q = None
        self._cache_position = None
        self._past_key_values = None
        self._installed = False
        return False


def assert_no_site_hooks(bundle) -> None:
    """Fail loudly if a previous run leaked cross-attention / site hooks."""
    leaked = []
    for i, layer in enumerate(bundle.model.model.decoder.layers):
        n = len(layer.encoder_attn._forward_hooks) + \
            len(layer.encoder_attn_layer_norm._forward_pre_hooks) + \
            len(layer._forward_pre_hooks)
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
