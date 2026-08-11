"""DEC-ALL-GLOBAL: per-layer decoder steering during generation (section 41).

This is a confound control, not the principal method. It is expected to be
unselective and to damage Mandarin output; that is exactly what it tests.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import torch

from ..models.hooks import DecoderSteeringHook


class DecoderAllLayerSteering:
    """Apply d_k^dec at every decoder layer k with strength alpha/sqrt(K)."""

    def __init__(self, bundle, directions: Mapping[int, torch.Tensor],
                 scales: Mapping[int, float], alpha: float,
                 norm_preserve: bool = True, layers: Sequence[int] | None = None):
        self.bundle = bundle
        self.directions = directions
        self.scales = scales
        self.alpha = float(alpha)
        self.norm_preserve = norm_preserve
        self.layers = list(layers) if layers is not None else sorted(directions)
        if not self.layers:
            raise ValueError("no decoder directions supplied")
        missing = [k for k in self.layers if k not in directions or k not in scales]
        if missing:
            raise ValueError(f"missing decoder direction/scale for layers {missing}")

    def __call__(self, batch: pd.DataFrame) -> list:
        k = len(self.layers)
        return [
            DecoderSteeringHook(
                self.bundle, layer, self.directions[layer], self.alpha,
                self.scales[layer], num_layers=k, norm_preserve=self.norm_preserve,
                steer_prefill=False,
            )
            for layer in self.layers
        ]
