"""Pure P1 causal-acceptance primitives.

P1 consumes the R2-selected gate ``g_old = E * R_B`` unchanged (``core_r2``) and a dynamic
direction ``d = normalize(hE - hB)``. The actual edit is applied by the canonical DG-02 hook
(``csasr.lss.sites.DecoderPostCrossAttnInterventionHook`` -> ``models.hooks.apply_steering``);
the functions here are the frozen reference math used for logging and audit.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .core_r2 import gate_values

VERSION_P1 = "p1_causal_acceptance_v1"
LAYER = 24
DIRECTION_TOL = 1e-4        # ||hE - hB|| below this (or nonfinite) -> no edit
DIRECTION_EPS = 1e-6        # d = delta / (||delta|| + eps)
NORM_EPS = 1e-6             # matches models.hooks.EPS


def direction(h_e: torch.Tensor, h_b: torch.Tensor) -> dict:
    """d = (hE - hB) / (||hE - hB|| + eps), with a frozen tiny/nonfinite fallback."""
    delta = h_e.detach().double() - h_b.detach().double()
    norm = float(torch.linalg.vector_norm(delta))
    if not math.isfinite(norm) or not bool(torch.isfinite(delta).all()):
        return {"status": "direction_fail:nonfinite", "norm": norm, "d": None}
    if norm < DIRECTION_TOL:
        return {"status": "direction_fail:tiny", "norm": norm, "d": None}
    return {"status": "ok", "norm": norm, "d": (delta / (norm + DIRECTION_EPS)).float()}


def selected_gate(support: dict, baseline: dict) -> float:
    """The R2-selected gate, g_old = E * R_B, computed by the frozen R2 function."""
    return gate_values(support, baseline, {"R": 0.0})["g_old"]


def reference_edit(h_b: torch.Tensor, d: torch.Tensor, alpha: float, g: float) -> dict:
    """Float64 reference of the hook edit: h~ = h + a*g*d; h' = h~ * ||h|| / (||h~|| + eps).

    Zero dose returns the input unchanged (no addition, no rescaling).
    """
    h = h_b.detach().double()
    if alpha * g == 0:
        return {"h_prime": h, "edit_norm": 0.0, "cos_edit_d": None, "applied": False}
    dd = d.detach().double()
    tilde = h + alpha * g * dd
    h_prime = tilde * torch.linalg.vector_norm(h) / (torch.linalg.vector_norm(tilde) + NORM_EPS)
    edit = h_prime - h
    en = float(torch.linalg.vector_norm(edit))
    cos = float(torch.dot(edit, dd) / (en * float(torch.linalg.vector_norm(dd)) + 1e-30))
    return {"h_prime": h_prime, "edit_norm": en, "cos_edit_d": cos, "applied": True}


def processed_argmax(logits: torch.Tensor, step: int, suppress: Sequence[int],
                     begin_suppress: Sequence[int]) -> int:
    """Greedy choice with the same suppression as Whisper generate()."""
    x = logits.detach().float().clone()
    if suppress:
        x[list(suppress)] = -float("inf")
    if step == 0 and begin_suppress:
        x[list(begin_suppress)] = -float("inf")
    return int(torch.argmax(x))


def step_inputs(prompt: Sequence[int], prefix: Sequence[int]) -> list[int]:
    if any(not isinstance(t, int) for t in list(prompt) + list(prefix)):
        raise ValueError("non-integer token")
    return list(prompt) + list(prefix)
