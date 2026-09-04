"""Trainable modules: the shared LID head, the low-rank intervention, LoRA, AGA.

The backbone is frozen in every arm. `assert_frozen_backbone` is called after
every backward pass in preflight: if any backbone gradient norm is nonzero the
arm is not doing what its name says and the run stops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from ..hooks import cache_length


# ---------------------------------------------------------------------------
# LID head -- identical architecture and supervision in every arm (4.1)
# ---------------------------------------------------------------------------

class LIDHead(nn.Module):
    """A small linear head predicting EN vs ZH per token (or per frame)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden.to(self.proj.weight.dtype))

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hidden)
        width = min(logits.shape[1], labels.shape[1])
        return F.cross_entropy(
            logits[:, :width].reshape(-1, 2).float(),
            labels[:, :width].reshape(-1), ignore_index=-100)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# rank-r intervention (4.2)
# ---------------------------------------------------------------------------

class LoReftIntervention(nn.Module):
    """h <- h + R^T (W h + b - R h), R an orthonormal rank-r projection."""

    variant = "loreft"

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model, self.rank = int(d_model), int(rank)
        self.R = nn.Parameter(torch.empty(rank, d_model))
        self.W = nn.Parameter(torch.empty(rank, d_model))
        self.b = nn.Parameter(torch.zeros(rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.R)
        nn.init.normal_(self.W, std=1.0 / math.sqrt(self.d_model))
        nn.init.zeros_(self.b)

    def orthonormal_R(self) -> torch.Tensor:
        # re-orthonormalise on the fly; keeps R on the Stiefel manifold without
        # a parametrisation object, which keeps the state_dict a plain tensor
        q, _ = torch.linalg.qr(self.R.float().T)
        return q.T.to(self.R.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        R = self.orthonormal_R()
        proj = h @ R.T                                    # (..., r)
        target = h @ self.W.T + self.b                    # (..., r)
        return h + (target - proj) @ R

    def subspace(self) -> torch.Tensor:
        return self.orthonormal_R().detach().float()

    def set_subspace(self, basis: torch.Tensor) -> None:
        with torch.no_grad():
            self.R.copy_(basis.to(self.R.dtype))


class AdditiveIntervention(nn.Module):
    """h <- h + B A h, the cheaper additive variant."""

    variant = "additive"

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model, self.rank = int(d_model), int(rank)
        self.A = nn.Parameter(torch.empty(rank, d_model))
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.A)
        nn.init.normal_(self.B, std=1.0 / math.sqrt(self.d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (h @ self.A.T) @ self.B.T

    def subspace(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.A.float().T)
        return q.T.detach()

    def set_subspace(self, basis: torch.Tensor) -> None:
        with torch.no_grad():
            self.A.copy_(basis.to(self.A.dtype))


INTERVENTIONS = {"loreft": LoReftIntervention, "additive": AdditiveIntervention}


def informed_basis(v_nat: np.ndarray, v_prompt: np.ndarray, rank: int, *,
                   seed: int) -> torch.Tensor:
    """span{v_nat, v_prompt}, filled with orthogonalised random for r > 2.

    Near-orthogonality (cos 0.010-0.030) makes the rank-2 basis
    well-conditioned, which is the whole point of the B2 arm.
    """
    d = int(v_nat.shape[0])
    columns = [np.asarray(v_nat, dtype=np.float64).reshape(-1)]
    if rank >= 2:
        columns.append(np.asarray(v_prompt, dtype=np.float64).reshape(-1))
    rng = np.random.default_rng(seed)
    while len(columns) < rank:
        columns.append(rng.standard_normal(d))
    matrix = np.stack(columns[:rank], axis=1)
    q, _ = np.linalg.qr(matrix)
    return torch.from_numpy(q.T.astype(np.float32))       # (rank, d)


def random_basis_matched(rank: int, d_model: int, *, seed: int) -> torch.Tensor:
    """Random orthonormal basis, matched in scale to `informed_basis`.

    Both are orthonormal, so B0/B1 differ from B2 in DIRECTION only, never in
    magnitude. That is what makes the comparison about the interpretability
    work rather than about a scale accident.
    """
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(d_model, rank, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.T.float()


class InterventionModule(nn.Module):
    """The intervention at one or more sites/layers, plus its hooks."""

    def __init__(self, bundle, *, site: str, layers: Sequence[int], rank: int,
                 variant: str = "loreft"):
        super().__init__()
        self.bundle = bundle
        self.site = site
        self.layers = tuple(int(l) for l in layers)
        self.rank = int(rank)
        self.variant = variant
        cls = INTERVENTIONS[variant]
        self.blocks = nn.ModuleDict(
            {str(l): cls(bundle.d_model, rank) for l in self.layers})
        self._handles: list = []
        self.fired = 0

    def _hook(self, layer: int):
        block = self.blocks[str(layer)]

        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                hidden, rest = output[0], output[1:]
                new = block(hidden.to(block_dtype(block)))
                self.fired += 1
                return (new.to(hidden.dtype),) + rest
            new = block(output.to(block_dtype(block)))
            self.fired += 1
            return new.to(output.dtype)

        return hook

    def attach(self) -> "InterventionModule":
        get = (self.bundle.encoder_layer if self.site == C.SITE_ENCODER
               else self.bundle.decoder_layer)
        for layer in self.layers:
            self._handles.append(get(layer).register_forward_hook(self._hook(layer)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    def initialize(self, *, informed: torch.Tensor | None, seed: int) -> dict[str, Any]:
        """Set every block's subspace. `informed=None` means matched random."""
        record = {}
        for i, layer in enumerate(self.layers):
            block = self.blocks[str(layer)]
            basis = (informed if informed is not None
                     else random_basis_matched(self.rank, self.bundle.d_model,
                                               seed=seed + 97 * i))
            block.set_subspace(basis.to(next(block.parameters()).device))
            record[str(layer)] = {"init": "informed" if informed is not None else "random",
                                  "basis_shape": tuple(basis.shape)}
        return record

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def analytic_parameter_count(self) -> int:
        d, r, n = self.bundle.d_model, self.rank, len(self.layers)
        return n * ((2 * r * d + r) if self.variant == "loreft" else (2 * r * d))

    def subspace(self) -> torch.Tensor:
        """The learned rank-r subspace of the FIRST configured layer.

        B0/B1/B2 run at "whatever Track A Stage D selected" (4.2), which is
        typically one layer -- the Track B default is a single decoder layer.
        When more than one layer is configured, this diagnoses only the first
        one; the 4.3 diagnostics are defined over one subspace, not a stack.
        """
        return self.blocks[str(self.layers[0])].subspace()


def block_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


# ---------------------------------------------------------------------------
# AGA adapters (4.4)
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """Linear(d -> d/k) -> GELU -> Linear(d/k -> d), residual, then LayerNorm.

    Exactly the reference implementation's `Adapter` followed by its
    `adapter_*_ln`; the bottleneck divisor is 4 there and here.
    """

    def __init__(self, d_model: int, divisor: int = 4):
        super().__init__()
        bottleneck = max(1, int(d_model // divisor))
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.down.weight.dtype
        y = x.to(dtype)
        y = y + self.up(F.gelu(self.down(y)))
        return self.norm(y).to(x.dtype)


class AdapterStack(nn.Module):
    """One adapter per selected block, applied to the block's output.

    The reference applies two adapters inside each residual block (after
    self-attention and after the MLP). A forward hook on the block sees only
    its output, so the post-MLP position is reproduced exactly and the
    post-attention position is folded into the same output-side adapter. The
    deviation is recorded in the fidelity note; the parameter count is kept
    faithful by placing TWO adapters at each block in sequence.
    """

    def __init__(self, bundle, *, encoder: bool, decoder: bool, divisor: int = 4):
        super().__init__()
        self.bundle = bundle
        self.use_encoder, self.use_decoder = bool(encoder), bool(decoder)
        d = bundle.d_model
        self.encoder_adapters = nn.ModuleList(
            [nn.ModuleList([Adapter(d, divisor), Adapter(d, divisor)])
             for _ in range(bundle.num_encoder_layers)] if encoder else [])
        self.decoder_adapters = nn.ModuleList(
            [nn.ModuleList([Adapter(d, divisor), Adapter(d, divisor)])
             for _ in range(bundle.num_decoder_layers)] if decoder else [])
        self._handles: list = []

    def _hook(self, pair: nn.ModuleList):
        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                hidden, rest = output[0], output[1:]
                for adapter in pair:
                    hidden = adapter(hidden)
                return (hidden,) + rest
            out = output
            for adapter in pair:
                out = adapter(out)
            return out
        return hook

    def attach(self) -> "AdapterStack":
        if self.use_encoder:
            for i, pair in enumerate(self.encoder_adapters):
                self._handles.append(
                    self.bundle.encoder_layer(i).register_forward_hook(self._hook(pair)))
        if self.use_decoder:
            for i, pair in enumerate(self.decoder_adapters):
                self._handles.append(
                    self.bundle.decoder_layer(i).register_forward_hook(self._hook(pair)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# freezing and gradient audit
# ---------------------------------------------------------------------------

def freeze_backbone(bundle) -> int:
    frozen = 0
    for p in bundle.model.parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    return frozen


def backbone_grad_norm(bundle) -> float:
    """Total gradient norm over parameters marked frozen. Must be exactly zero.

    "Backbone" here means `requires_grad=False`, not "everything under
    `bundle.model`". For the hook-based arms (B0-B2, B3-reimpl) every
    parameter of `bundle.model` is frozen, so the two definitions coincide and
    this is unchanged from summing over the whole model. B4 (LoRA) is
    different: `LoraModel` injects `lora_A`/`lora_B` AS CHILDREN of
    `bundle.model` with `requires_grad=True`, so those tensors are the arm's
    own trainable parameters living inside the backbone's module tree, not
    backbone leakage -- summing them here would misreport a correctly working
    LoRA arm as a frozen-backbone violation.
    """
    total = 0.0
    for p in bundle.model.parameters():
        if p.requires_grad:
            continue
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def assert_frozen_backbone(bundle, *, arm: str) -> None:
    norm = backbone_grad_norm(bundle)
    if norm != 0.0:
        raise AssertionError(
            f"arm {arm}: frozen-backbone gradient norm is {norm!r}, expected "
            f"exactly 0.0. The arm is not doing what its name says. Stopping.")


def trainable_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def backbone_parameters(bundle) -> int:
    return sum(p.numel() for p in bundle.model.parameters())
