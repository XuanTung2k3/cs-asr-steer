"""Steering hooks satisfying requirements 1.1-1.6.

1.1  Encoder steering is restricted to ceil(duration_sec * 50) valid frames.
     Whisper pads every clip to 1500 encoder frames and applies NO encoder
     attention mask, so unrestricted global encoder steering puts roughly 80%
     of its energy into padding. `valid_frame_ratio` is logged per utterance.
1.2  Decoder absolute position is read from the KV-cache length through a
     forward_pre_hook on `model.model.decoder`, not inferred from the hidden
     width. Legacy tuple caches and `Cache` objects are both handled.
1.3  Under beam search the batch dimension is B*k. Every beam is touched and
     `steered_positions % num_beams == 0` is asserted.
1.4  Energy accounting: E_total = alpha^2 * ||v||^2 * |S| with |S| the steered
     positions summed over layers and sites. Under matched energy the applied
     coefficient is alpha_eff = alpha / sqrt(|S|), so E_total is invariant to
     coverage and dose is separable from coverage.
1.5  scale_mode "act_norm" (default): h += alpha * mean_norm(h) * v.
     scale_mode "unit": h += alpha * v.
1.6  A prompt-derived direction at an encoder site raises. Whisper's encoder
     never receives the language token; there is no such object to apply.
"""
from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from . import config as C


class PromptDirectionAtEncoderError(AssertionError):
    """Requirement 1.6. Raised, never worked around."""


class BeamCoverageError(AssertionError):
    """Requirement 1.3. Raised when a beam was not steered."""


# ---------------------------------------------------------------------------
# cache-length reading (1.2)
# ---------------------------------------------------------------------------

def cache_length(past_key_values: Any) -> int:
    """Absolute number of positions already in the decoder KV cache.

    Handles both shapes transformers has used:
      * legacy tuple cache  `pkv[0][0].shape[2]`
      * `Cache` object      `pkv.get_seq_length()`
    An absent or empty cache is length zero, which is the prefill call.
    """
    if past_key_values is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    try:
        first = past_key_values[0]
        if first is None:
            return 0
        key = first[0]
        if key is None:
            return 0
        return int(key.shape[2])
    except (IndexError, TypeError, AttributeError):
        return 0


class DecoderPositionTracker:
    """Publishes the absolute start position of the current decoder forward."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.start_position = 0
        self.forward_calls = 0
        self._handle = None

    def _pre_hook(self, _mod, args, kwargs):
        self.start_position = cache_length(kwargs.get("past_key_values"))
        self.forward_calls += 1
        return None

    def __enter__(self) -> "DecoderPositionTracker":
        self._handle = self.bundle.model.model.decoder.register_forward_pre_hook(
            self._pre_hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


# ---------------------------------------------------------------------------
# the plan: everything fixed before the forward pass
# ---------------------------------------------------------------------------

@dataclass
class DirectionSpec:
    """One unit-norm direction bound to a site."""

    kind: str
    site: str
    layer: int
    vector: torch.Tensor           # (D,) float32, unit norm
    source: str                    # "loaded" | "reconstructed" | "derived"
    norm_before_unit: float = 1.0

    def __post_init__(self):
        if self.kind in C.DECODER_ONLY_KINDS and self.site == C.SITE_ENCODER:
            raise PromptDirectionAtEncoderError(
                f"{self.kind} is decoder-only: Whisper's encoder never receives "
                f"the language token, so a prompt-derived direction has no "
                f"defined meaning at an encoder site. Refusing to produce a "
                f"number for site={self.site} layer={self.layer}.")


@dataclass
class UtterancePlan:
    """Per-utterance coverage, resolved before the run."""

    utterance_id: str
    duration_sec: float
    valid_frames: int
    encoder_frames: Sequence[tuple[int, int]] = field(default_factory=list)
    decoder_steps: Sequence[int] | None = None      # None => all generated
    planned_decoder_positions: int = 0

    @property
    def valid_frame_ratio(self) -> float:
        return float(self.valid_frames) / float(C.EXPECTED_ENCODER_FRAMES)


@dataclass
class SteeringPlan:
    """A complete steering cell."""

    site: str
    coverage: str
    alpha: float
    scale_mode: str = C.DEFAULT_SCALE_MODE
    matched_energy: bool = True
    encoder_layers: tuple[int, ...] = ()
    decoder_layers: tuple[int, ...] = ()
    encoder_direction: DirectionSpec | None = None
    decoder_direction: DirectionSpec | None = None
    num_beams: int = 1
    norm_preserve: bool = False

    def validate(self) -> None:
        if self.site in (C.SITE_ENCODER, C.SITE_BOTH):
            if self.encoder_direction is None or not self.encoder_layers:
                raise ValueError("encoder site needs a direction and layers")
            if self.encoder_direction.kind in C.DECODER_ONLY_KINDS:
                raise PromptDirectionAtEncoderError(
                    f"{self.encoder_direction.kind} requested at an encoder site")
        if self.site in (C.SITE_DECODER, C.SITE_BOTH):
            if self.decoder_direction is None or not self.decoder_layers:
                raise ValueError("decoder site needs a direction and layers")
        if self.scale_mode not in (C.SCALE_ACT_NORM, C.SCALE_UNIT):
            raise ValueError(f"unknown scale_mode {self.scale_mode!r}")
        if self.coverage not in (C.COVERAGE_GLOBAL, C.COVERAGE_LOCAL):
            raise ValueError(f"unknown coverage {self.coverage!r}")


# ---------------------------------------------------------------------------
# energy accounting (1.4)
# ---------------------------------------------------------------------------

def planned_positions(plan: SteeringPlan, utterances: Sequence[UtterancePlan]) -> int:
    """|S|: steered positions summed over layers AND sites AND utterances."""
    total = 0
    if plan.site in (C.SITE_ENCODER, C.SITE_BOTH):
        per_utt = 0
        for u in utterances:
            if plan.coverage == C.COVERAGE_GLOBAL:
                per_utt += int(u.valid_frames)
            else:
                per_utt += sum(max(0, hi - lo) for lo, hi in u.encoder_frames)
        total += per_utt * len(plan.encoder_layers)
    if plan.site in (C.SITE_DECODER, C.SITE_BOTH):
        per_utt = 0
        for u in utterances:
            if plan.coverage == C.COVERAGE_GLOBAL:
                per_utt += int(u.planned_decoder_positions)
            else:
                per_utt += len(u.decoder_steps or ())
        total += per_utt * len(plan.decoder_layers)
    return int(total)


def energy_report(plan: SteeringPlan, n_positions: int,
                  direction_norm: float = 1.0) -> dict[str, float]:
    """E_total, |S| and alpha_eff for one cell.

    Under matched energy the coefficient actually applied is
    alpha / sqrt(|S|), which makes E_total = alpha^2 ||v||^2 independent of
    coverage. Without it, a global cell and a local cell differ in dose AND in
    coverage and neither effect is identifiable.
    """
    n = max(0, int(n_positions))
    alpha_eff = (plan.alpha / math.sqrt(n)) if (plan.matched_energy and n > 0) \
        else plan.alpha
    e_total = float(alpha_eff ** 2) * float(direction_norm) ** 2 * float(n)
    return {"E_total": e_total, "S": n, "alpha_eff": float(alpha_eff),
            "alpha_nominal": float(plan.alpha),
            "direction_norm": float(direction_norm),
            "matched_energy": bool(plan.matched_energy)}


# ---------------------------------------------------------------------------
# the edit
# ---------------------------------------------------------------------------

def _split(output):
    if isinstance(output, tuple):
        rest = output[1:]
        return output[0], (lambda new: (new,) + rest)
    return output, (lambda new: new)


def apply_edit(hidden: torch.Tensor, direction: torch.Tensor, coefficient: float,
               mask: torch.Tensor, scale_mode: str,
               norm_preserve: bool = False) -> tuple[torch.Tensor, int, float]:
    """h[mask] += coefficient * scale * v.

    `mask` is (B, T) boolean. Positions outside it are returned bit-identical.
    Returns (new_hidden, positions_edited, mean_norm_used).
    """
    if coefficient == 0.0:
        return hidden, 0, 0.0
    edited = int(mask.sum().item())
    if edited == 0:
        return hidden, 0, 0.0
    v = direction.to(hidden.device, hidden.dtype).view(1, 1, -1)
    if scale_mode == C.SCALE_ACT_NORM:
        norms = hidden.norm(dim=-1)                      # (B, T)
        selected = norms[mask]
        scale = float(selected.mean().item()) if selected.numel() else 0.0
    else:
        scale = 1.0
    delta = (coefficient * scale) * v
    steered = hidden + delta
    if norm_preserve:
        # Per-token norm preservation, not unit normalization.
        steered = steered * (hidden.norm(dim=-1, keepdim=True) /
                             steered.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return torch.where(mask.unsqueeze(-1), steered, hidden), edited, float(scale)


# ---------------------------------------------------------------------------
# encoder hook (1.1)
# ---------------------------------------------------------------------------

class EncoderSteering:
    """Steer encoder layers over VALID frames only."""

    def __init__(self, bundle, plan: SteeringPlan, batch: Sequence[UtterancePlan],
                 coefficient: float):
        self.bundle = bundle
        self.plan = plan
        self.batch = list(batch)
        self.coefficient = float(coefficient)
        self.fired = 0
        self.positions = 0
        self.scale_samples: list[float] = []
        self._handles: list = []
        self._mask: torch.Tensor | None = None

    def _build_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        b, t = hidden.shape[0], hidden.shape[1]
        if b != len(self.batch):
            raise ValueError(
                f"encoder batch {b} does not match the plan's {len(self.batch)}")
        mask = torch.zeros((b, t), dtype=torch.bool, device=hidden.device)
        for i, u in enumerate(self.batch):
            valid = min(int(u.valid_frames), t)
            if self.plan.coverage == C.COVERAGE_GLOBAL:
                mask[i, :valid] = True
            else:
                for lo, hi in u.encoder_frames:
                    lo = max(0, min(int(lo), valid))
                    hi = max(lo, min(int(hi), valid))
                    if hi > lo:
                        mask[i, lo:hi] = True
        return mask

    def _hook(self, _mod, _inp, output):
        hidden, rebuild = _split(output)
        if self.coefficient == 0.0:
            return output                          # bit-identical (preflight 1)
        mask = self._build_mask(hidden)
        new, edited, scale = apply_edit(
            hidden, self.plan.encoder_direction.vector, self.coefficient, mask,
            self.plan.scale_mode, self.plan.norm_preserve)
        self.fired += 1
        self.positions += edited
        if scale:
            self.scale_samples.append(scale)
        return rebuild(new)

    def __enter__(self) -> "EncoderSteering":
        for layer in self.plan.encoder_layers:
            self._handles.append(
                self.bundle.encoder_layer(int(layer)).register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ---------------------------------------------------------------------------
# decoder hook (1.2, 1.3)
# ---------------------------------------------------------------------------

class DecoderSteering:
    """Steer decoder layers at absolute positions read from the KV cache.

    Global coverage steers every generated position, i.e. absolute position
    >= PREFIX_WIDTH; the four forced prefix tokens are never edited. Local
    coverage steers only the oracle step indices supplied by the plan, mapped
    to absolute positions as PREFIX_WIDTH + step.
    """

    def __init__(self, bundle, plan: SteeringPlan, batch: Sequence[UtterancePlan],
                 coefficient: float, tracker: DecoderPositionTracker,
                 prefix_width: int = C.PREFIX_WIDTH):
        self.bundle = bundle
        self.plan = plan
        self.batch = list(batch)
        self.coefficient = float(coefficient)
        self.tracker = tracker
        self.prefix_width = int(prefix_width)
        self.num_beams = max(1, int(plan.num_beams))
        self.fired = 0
        self.positions = 0
        self.prefill_calls = 0
        self.scale_samples: list[float] = []
        self.absolute_positions_seen: list[int] = []
        self._handles: list = []

    def _allowed(self, absolute: int, item: int) -> bool:
        if absolute < self.prefix_width:
            return False
        if self.plan.coverage == C.COVERAGE_GLOBAL:
            return True
        steps = self.batch[item].decoder_steps
        if not steps:
            return False
        return (absolute - self.prefix_width) in set(int(s) for s in steps)

    def _build_mask(self, hidden: torch.Tensor, start: int) -> torch.Tensor:
        rows, t = hidden.shape[0], hidden.shape[1]
        # under beam search the batch dimension is B * num_beams and the beams
        # of one item are contiguous; positions stay synchronised across beams
        if rows % self.num_beams != 0:
            raise BeamCoverageError(
                f"decoder batch rows {rows} is not a multiple of num_beams "
                f"{self.num_beams}: beams are not aligned")
        items = rows // self.num_beams
        if items != len(self.batch):
            raise ValueError(
                f"decoder resolves {items} items but the plan has {len(self.batch)}")
        mask = torch.zeros((rows, t), dtype=torch.bool, device=hidden.device)
        for r in range(rows):
            item = r // self.num_beams
            for j in range(t):
                if self._allowed(start + j, item):
                    mask[r, j] = True
        return mask

    def _hook(self, _mod, _inp, output):
        hidden, rebuild = _split(output)
        start = int(self.tracker.start_position)
        if hidden.shape[1] > 1 and start == 0:
            self.prefill_calls += 1
        if self.coefficient == 0.0:
            return output                          # bit-identical (preflight 1)
        mask = self._build_mask(hidden, start)
        edited = int(mask.sum().item())
        if edited == 0:
            self.fired += 1
            return output
        new, edited, scale = apply_edit(
            hidden, self.plan.decoder_direction.vector, self.coefficient, mask,
            self.plan.scale_mode, self.plan.norm_preserve)
        self.fired += 1
        self.positions += edited
        for j in range(hidden.shape[1]):
            if bool(mask[:, j].any()):
                self.absolute_positions_seen.append(start + j)
        if scale:
            self.scale_samples.append(scale)
        return rebuild(new)

    def assert_every_beam_steered(self) -> None:
        """Requirement 1.3: steered_positions % num_beams == 0.

        This is checked per BATCH (decode_population calls it after every
        batch), and under local coverage most batches legitimately steer
        zero positions: only utterances that carry an oracle target get a
        nonzero mask, and a batch of utterances sorted by duration can
        easily contain none of them (only 116 of 300 D-dev-select utterances
        carry a target at all). Zero is therefore not itself an error --
        0 % k == 0 holds trivially. What must never happen is a PARTIAL
        beam group: some beams of a targeted item steered and others not,
        which the modulus check below catches regardless of the count.

        `self.fired == 0` is a different, stronger failure: it means the
        hook was never even invoked (wrong layer, hook never attached), and
        is checked separately from position count.
        """
        if self.num_beams == 1:
            return
        if self.fired == 0:
            raise BeamCoverageError(
                "the decoder steering hook never fired at all this batch; "
                "it should fire once per non-prefill forward regardless of "
                "whether any position gets steered")
        if self.positions % self.num_beams != 0:
            raise BeamCoverageError(
                f"steered_positions ({self.positions}) is not divisible by "
                f"num_beams ({self.num_beams}): at least one beam was missed")

    def __enter__(self) -> "DecoderSteering":
        for layer in self.plan.decoder_layers:
            self._handles.append(
                self.bundle.decoder_layer(int(layer)).register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

class SteeringSession(ExitStack):
    """Install the hooks a plan needs and collect their audit fields."""

    def __init__(self, bundle, plan: SteeringPlan, batch: Sequence[UtterancePlan],
                 coefficient: float, prefix_width: int = C.PREFIX_WIDTH):
        super().__init__()
        plan.validate()
        self.bundle = bundle
        self.plan = plan
        self.batch = list(batch)
        self.coefficient = float(coefficient)
        self.prefix_width = int(prefix_width)
        self.encoder: EncoderSteering | None = None
        self.decoder: DecoderSteering | None = None
        self.tracker: DecoderPositionTracker | None = None

    def __enter__(self) -> "SteeringSession":
        super().__enter__()
        if self.plan.site in (C.SITE_ENCODER, C.SITE_BOTH):
            self.encoder = self.enter_context(
                EncoderSteering(self.bundle, self.plan, self.batch, self.coefficient))
        if self.plan.site in (C.SITE_DECODER, C.SITE_BOTH):
            self.tracker = self.enter_context(DecoderPositionTracker(self.bundle))
            self.decoder = self.enter_context(
                DecoderSteering(self.bundle, self.plan, self.batch,
                                self.coefficient, self.tracker, self.prefix_width))
        return self

    def audit(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hook_fire_count": 0,
            "steered_positions": 0,
            "expected_hook_fires_per_forward": (
                len(self.plan.encoder_layers) if self.encoder else 0)
            + (len(self.plan.decoder_layers) if self.decoder else 0),
            "valid_frame_ratio": [round(u.valid_frame_ratio, 6) for u in self.batch],
            "valid_frame_ratio_mean": float(
                np.mean([u.valid_frame_ratio for u in self.batch])) if self.batch else float("nan"),
        }
        if self.encoder is not None:
            out["encoder_hook_fires"] = self.encoder.fired
            out["encoder_steered_positions"] = self.encoder.positions
            out["encoder_act_norm_mean"] = (
                float(np.mean(self.encoder.scale_samples))
                if self.encoder.scale_samples else float("nan"))
            out["hook_fire_count"] += self.encoder.fired
            out["steered_positions"] += self.encoder.positions
        if self.decoder is not None:
            out["decoder_hook_fires"] = self.decoder.fired
            out["decoder_steered_positions"] = self.decoder.positions
            out["decoder_prefill_calls"] = self.decoder.prefill_calls
            out["decoder_forward_calls"] = self.tracker.forward_calls if self.tracker else 0
            out["decoder_act_norm_mean"] = (
                float(np.mean(self.decoder.scale_samples))
                if self.decoder.scale_samples else float("nan"))
            out["decoder_min_absolute_position"] = (
                min(self.decoder.absolute_positions_seen)
                if self.decoder.absolute_positions_seen else None)
            out["hook_fire_count"] += self.decoder.fired
            out["steered_positions"] += self.decoder.positions
            out["num_beams"] = self.decoder.num_beams
        return out
