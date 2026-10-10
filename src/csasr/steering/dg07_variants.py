"""DG-07 learned baselines and structural ablations.

The classes in this module are deliberately small adapters around the frozen
DG-02 intervention primitive.  They do not load data, choose checkpoints, or
define a scientific objective.  The DG-07 runner owns those concerns.

Variants:

* ``GlobalVector`` — LB1, one position-invariant learned vector at the exact
  post-cross-attention/pre-FFN site;
* ``GateOnlyController`` — A1/A2/A3, a shared controller trunk that learns
  only a scalar gate while the direction is frozen;
* ``ExactLayerQvLoRA`` — LB2, manual LoRA adapters on Q/V in decoder layer 24.

The frozen Whisper model is never registered as a child of the LoRA module,
which keeps its checkpoint and parameter accounting limited to the adapters.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the exact number of optimizer parameters in ``module``."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class GlobalVector(nn.Module):
    """LB1: one learned global vector, applied at every eligible position.

    The vector is intentionally not normalized: SALSA-style learning includes
    the learned intervention magnitude.  The exact-site hook applies it with
    ``alpha=scale=1`` and the canonical NormPreserve repair.
    """

    variant = "LB1_SALSA_EXACT_GLOBAL"

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Standard SALSA-style zero initialization.  A zero intervention is a
        # fair starting point and gradients through the exact-site repair train
        # the vector without importing the proposed controller checkpoint.
        self.vector = nn.Parameter(torch.zeros(int(d_model), dtype=torch.float32))
        self.d_model = int(d_model)
        self.alpha = 1.0

    def action(self, *, r: torch.Tensor, **_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an all-position gate and the same vector for every token."""
        gate = torch.ones(r.shape[:2], device=r.device, dtype=r.dtype)
        return gate, self.vector.to(device=r.device, dtype=r.dtype)


class GateOnlyController(nn.Module):
    """A1/A2/A3: DG-05-sized trunk with one learned gate output.

    ``direction`` is a detached, unit-norm frozen vector.  No mixture logits
    exist in this module, so the ablations cannot accidentally learn a hidden
    direction or receive gradients through the basis.
    """

    def __init__(self, direction: torch.Tensor, bottleneck: int = 32) -> None:
        super().__init__()
        d = int(direction.numel())
        if direction.ndim != 1 or d <= 0:
            raise ValueError("direction must be a non-empty vector")
        direction = direction.detach().float()
        norm = direction.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
            raise ValueError("direction must be finite and non-degenerate")
        self.d_model = d
        self.bottleneck = int(bottleneck)
        self.norm = nn.LayerNorm(d)
        self.hidden = nn.Linear(d, self.bottleneck)
        self.output = nn.Linear(self.bottleneck, 1)
        self.register_buffer("direction", (direction / norm).clone(), persistent=True)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, site_state: torch.Tensor) -> torch.Tensor:
        if site_state.ndim < 2 or site_state.shape[-1] != self.d_model:
            raise ValueError(
                f"site_state must end in {self.d_model}, got {tuple(site_state.shape)}")
        x = site_state.float()
        logits = self.output(torch.nn.functional.gelu(self.hidden(self.norm(x))))
        return torch.sigmoid(logits[..., 0])

    def action(self, *, r: torch.Tensor, **_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self(r)
        return gate, self.direction.to(device=r.device, dtype=r.dtype)


def lora_parameter_count(rank: int, *, in_features: int, out_features: int,
                         target_modules: int = 2) -> int:
    """Count trainable A/B entries for Q/V LoRA, excluding frozen base weights."""
    rank = int(rank)
    if rank <= 0 or in_features <= 0 or out_features <= 0 or target_modules <= 0:
        raise ValueError("LoRA rank, dimensions, and target count must be positive")
    return int(target_modules * rank * (int(in_features) + int(out_features)))


def select_lora_rank(target_count: int, *, in_features: int, out_features: int,
                     target_modules: int = 2, max_rank: int = 64) -> tuple[int, int]:
    """Select the nearest integer rank mechanically, before any outcomes.

    Ties are resolved toward the smaller rank.  This is accounting, not a
    performance sweep; the caller freezes the returned rank in the DG-07
    config and manifest.
    """
    if target_count <= 0 or max_rank <= 0:
        raise ValueError("target_count and max_rank must be positive")
    candidates = [(abs(lora_parameter_count(r, in_features=in_features,
                                            out_features=out_features,
                                            target_modules=target_modules) - target_count), r)
                  for r in range(1, int(max_rank) + 1)]
    _, rank = min(candidates, key=lambda item: (item[0], item[1]))
    return int(rank), lora_parameter_count(rank, in_features=in_features,
                                           out_features=out_features,
                                           target_modules=target_modules)


class ExactLayerQvLoRA(nn.Module):
    """LB2: LoRA adapters on Q/V of one frozen decoder layer.

    The target layer is supplied by the caller and must be the selected L24
    decoder layer.  The base module is held as a non-child reference so
    ``state_dict`` and optimizer accounting contain adapters only.
    """

    variant = "LB2_LORA_MATCHED_BUDGET"

    def __init__(self, layer_module: nn.Module, rank: int, *, alpha: float | None = None,
                 target_layer: int = 24) -> None:
        super().__init__()
        if not hasattr(layer_module, "self_attn"):
            raise ValueError("layer_module must expose self_attn Q/V projections")
        q_proj = layer_module.self_attn.q_proj
        v_proj = layer_module.self_attn.v_proj
        if q_proj.in_features != v_proj.in_features or q_proj.out_features != v_proj.out_features:
            raise ValueError("Q/V dimensions must match for the frozen DG-07 target")
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.alpha = float(self.rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.target_layer = int(target_layer)
        self.in_features = int(q_proj.in_features)
        self.out_features = int(q_proj.out_features)
        object.__setattr__(self, "layer_module", layer_module)

        self.q_A = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.q_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.v_A = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.v_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self._orig_q_forward = q_proj.forward
        self._orig_v_forward = v_proj.forward
        self._attached = False

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_count(self) -> int:
        return trainable_parameter_count(self)

    def _patch_q(self, x: torch.Tensor) -> torch.Tensor:
        base = self._orig_q_forward(x)
        delta = (x.to(self.q_A.dtype) @ self.q_A.T @ self.q_B.T) * self.scaling
        return base + delta.to(base.dtype)

    def _patch_v(self, x: torch.Tensor) -> torch.Tensor:
        base = self._orig_v_forward(x)
        delta = (x.to(self.v_A.dtype) @ self.v_A.T @ self.v_B.T) * self.scaling
        return base + delta.to(base.dtype)

    def attach(self) -> "ExactLayerQvLoRA":
        if self._attached:
            raise RuntimeError("LoRA adapter is already attached")
        self.layer_module.self_attn.q_proj.forward = self._patch_q
        self.layer_module.self_attn.v_proj.forward = self._patch_v
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        self.layer_module.self_attn.q_proj.forward = self._orig_q_forward
        self.layer_module.self_attn.v_proj.forward = self._orig_v_forward
        self._attached = False

    def __enter__(self) -> "ExactLayerQvLoRA":
        return self.attach()

    def __exit__(self, *_exc: Any) -> bool:
        self.detach()
        return False


def assert_frozen_basis(module: nn.Module) -> None:
    """Reject an ablation whose direction/basis buffer became trainable."""
    for name, parameter in module.named_parameters():
        if name in {"direction", "basis"} and parameter.requires_grad:
            raise AssertionError(f"frozen basis buffer unexpectedly trainable: {name}")
    for name, buffer in module.named_buffers():
        if name in {"direction", "basis"} and buffer.requires_grad:
            raise AssertionError(f"frozen basis buffer unexpectedly requires grad: {name}")
