"""DG-05 fixed-basis adaptive steering controller.

This module contains the small trainable controller only.  The Whisper
backbone and the rank-2 basis are deliberately kept outside the optimizer:
``FixedBasisAdaptiveController.basis`` is a non-trainable buffer and the
forward interface accepts only the exact-site state ``r``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


CONTROLLER_SCHEMA_VERSION = "dg05_adaptive_controller_v1"
DEFAULT_BOTTLENECK = 32
OUTPUT_DIM = 3                         # gate logit + two mixture logits
EXPECTED_BASIS_RANK = 2
INFERENCE_INPUTS = ("site_state",)


def normalize_directions(direction: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the final feature dimension and reject degenerate vectors."""
    if direction.ndim < 1:
        raise ValueError("direction must have at least one dimension")
    norm = direction.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm < eps).any()):
        raise ValueError("cannot normalize a non-finite or degenerate direction")
    return direction / norm


class FixedBasisAdaptiveController(nn.Module):
    """``f_theta(LN(r)) -> (g, pi, d)`` over a frozen rank-2 basis.

    The module is intentionally modest and fixed for DG-05A:
    ``LayerNorm -> Linear(d, 32) -> GELU -> Linear(32, 3)``.
    """

    def __init__(self, d_model: int, basis: torch.Tensor | np.ndarray,
                 bottleneck: int = DEFAULT_BOTTLENECK):
        super().__init__()
        if int(bottleneck) <= 0:
            raise ValueError("bottleneck must be positive")
        basis_t = torch.as_tensor(basis, dtype=torch.float32)
        if basis_t.ndim != 2 or basis_t.shape[1] != EXPECTED_BASIS_RANK:
            raise ValueError(
                f"basis must have shape (d_model, 2), got {tuple(basis_t.shape)}")
        if basis_t.shape[0] != int(d_model):
            raise ValueError(f"basis dim {basis_t.shape[0]} != d_model {d_model}")
        if not bool(torch.isfinite(basis_t).all()):
            raise ValueError("basis contains non-finite values")
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.norm = nn.LayerNorm(self.d_model)
        self.hidden = nn.Linear(self.d_model, self.bottleneck)
        self.output = nn.Linear(self.bottleneck, OUTPUT_DIM)
        # A basis is an input artifact, never an optimizer parameter.
        self.register_buffer("basis", basis_t.detach().clone(), persistent=True)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, site_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gate ``(B,T)``, mixture ``(B,T,2)``, direction ``(B,T,D)``."""
        if site_state.ndim < 2 or site_state.shape[-1] != self.d_model:
            raise ValueError(
                f"site_state must end in {self.d_model}, got {tuple(site_state.shape)}")
        # Keep the controller numerically stable and independent of Whisper's
        # bf16 activation dtype.  This does not add any inference feature.
        x = site_state.float()
        logits = self.output(torch.nn.functional.gelu(self.hidden(self.norm(x))))
        gate = torch.sigmoid(logits[..., 0])
        pi = torch.softmax(logits[..., 1:], dim=-1)
        direction = torch.einsum("...k,dk->...d", pi, self.basis.to(pi.device))
        direction = normalize_directions(direction)
        return gate, pi, direction

    def action(self, *, r: torch.Tensor, **_: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapter for ``DecoderPostCrossAttnInterventionHook.action_fn``."""
        gate, _pi, direction = self(r)
        return gate, direction


def assert_inference_input_names(names: list[str] | tuple[str, ...]) -> None:
    """Guard the controller's runtime feature contract."""
    unknown = sorted(set(names) - set(INFERENCE_INPUTS))
    if unknown:
        raise ValueError(
            "DG-05 controller inputs must be only LN(r); forbidden inputs: "
            f"{unknown}")


def load_frozen_basis(path: str | Path, *, expected_layer: int,
                      expected_local_hash: str | None = None,
                      expected_cond_hash: str | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and hash-check the frozen DG-03 basis without reconstructing it."""
    import hashlib
    import json

    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record.get("scientific_layer", record.get("layer", -1))) != int(expected_layer):
        raise ValueError("DG-05 basis layer does not match the frozen selected layer")
    base = path.parent
    arrays = {}
    for name, key in (("local", "conditioning_residualized_local"),
                      ("cond", "language_conditioning")):
        arr = np.asarray(np.load(base / record["tensor_files"][key]))
        digest = "sha256:" + hashlib.sha256(
            np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()
        expected = record["tensor_hashes"][key]
        if digest != expected:
            raise ValueError(f"DG-03 basis tensor hash mismatch for {key}")
        if name == "local" and expected_local_hash and digest != expected_local_hash:
            raise ValueError("DG-05 local direction is not the frozen DG-04 input")
        if name == "cond" and expected_cond_hash and digest != expected_cond_hash:
            raise ValueError("DG-05 conditioning direction is not the frozen DG-04 input")
        arrays[name] = torch.from_numpy(arr.astype(np.float32, copy=False))
    basis = torch.stack([arrays["local"], arrays["cond"]], dim=1)
    # The columns must be individually unit norm; preserve the artifact's
    # orthogonality rather than altering the frozen vectors.
    if not torch.allclose(basis.norm(dim=0), torch.ones(2), atol=1e-5):
        raise ValueError("frozen DG-03 basis columns are not unit norm")
    return basis, record


def apply_controller_action(site: torch.Tensor, controller: FixedBasisAdaptiveController,
                             beta: float, *, norm_preserve: bool = True) -> torch.Tensor:
    """Apply one adaptive action through the canonical steering primitive."""
    from ..models.hooks import apply_steering

    gate, _pi, direction = controller(site)
    return apply_steering(site, direction, float(beta), 1.0, gate, norm_preserve)
