"""Hook builders that turn per-utterance target spans into encoder steering."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch

from ..models.hooks import EncoderSteeringHook
from .masks import MaskSpec, build_gain, derive_jitter_seed


class EncoderLocalSteering:
    """Build an ``EncoderSteeringHook`` per decoding batch.

    ``spans`` maps utterance_id -> (start_frame, end_frame) of the target POI.
    ``context_spans`` optionally maps utterance_id -> (start, end) for the
    WHOLE_WORD_PLUS_CONTEXT mask.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, alpha: float,
                 scale: float, spec: MaskSpec,
                 spans: Mapping[str, tuple[int, int]],
                 context_spans: Mapping[str, tuple[int, int]] | None = None,
                 norm_preserve: bool = True):
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.spec = spec
        self.spans = spans
        self.context_spans = context_spans or {}
        self.norm_preserve = norm_preserve
        self.applied_meta: dict[str, dict] = {}

    def gain_matrix(self, batch: pd.DataFrame) -> torch.Tensor:
        n_frames = self.bundle.max_encoder_frames
        rows = []
        for _, r in batch.iterrows():
            utt = r["utterance_id"]
            n_valid = self.bundle.valid_frames(r["duration_sec"])
            spec = self.spec
            if spec.kind.startswith("JITTER_") and spec.seed is not None:
                spec = MaskSpec(**{**spec.__dict__,
                                   "seed": derive_jitter_seed(spec.seed, str(utt))})
            if spec.kind == "WHOLE_WORD_PLUS_CONTEXT":
                spec = MaskSpec(**{**spec.__dict__, "context_span": self.context_spans.get(utt)})
            if spec.kind == "GLOBAL":
                start, end = 0, n_valid
            else:
                start, end = self.spans[utt]
            g, meta = build_gain(spec, start, end, n_frames, n_valid)
            meta["utterance_id"] = utt
            self.applied_meta[utt] = meta
            rows.append(g)
        return torch.from_numpy(np.stack(rows)).to(self.bundle.device)

    def __call__(self, batch: pd.DataFrame) -> list:
        gain = self.gain_matrix(batch)
        return [EncoderSteeringHook(self.bundle, self.layer, self.direction,
                                    self.alpha, self.scale, gain, self.norm_preserve)]


def global_encoder_steering(bundle, layer: int, direction: torch.Tensor, alpha: float,
                            scale: float, norm_preserve: bool = True) -> Callable:
    """ENC-GLOBAL-1L: same direction/layer over every valid encoder frame."""
    spec = MaskSpec(kind="GLOBAL")
    return EncoderLocalSteering(bundle, layer, direction, alpha, scale, spec,
                                spans={}, norm_preserve=norm_preserve)
