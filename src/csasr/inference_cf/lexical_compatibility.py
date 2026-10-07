"""P2-SEL-LAC pure helpers (frozen contract configs/inference_cf/p2_sel_lac.json).

* ``select_candidates``: from RAW UNMASKED float logits only, c_E / c_M = lowest token ID attaining the
  maximum over the canonical embedded (V_E) / matrix (V_M) partition sets (IDs sorted numerically).
* ``hard_mask``: x_mask = x.copy(); x_mask[a:b] = float32(0.0) for W* = [a, b) inside the heard waveform.
* ``lexical``: M0, Mmask, Delta_E, Delta_M, S_lex = M0 - Mmask = Delta_E - Delta_M (float64), F_lex.
* ``waveform_model_inputs``: the batch_model_inputs feature-extractor call on an in-memory waveform.

No reference, evaluator, label or future quantity enters any function here.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

HEARD_MAX = 480000
IDENTITY_TOL = 1e-10


def select_candidates(z, V_E: Sequence[int], V_M: Sequence[int]) -> tuple[int, int]:
    z = np.asarray(z)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("unmasked logits must be finite 1-D")

    def best(ids):
        ids = np.array(sorted(int(i) for i in ids), dtype=np.int64)
        vals = z[ids]
        return int(ids[int(np.argmax(vals))])          # np.argmax: first (lowest sorted ID) maximum
    return best(V_E), best(V_M)


def hard_mask(x: np.ndarray, a: int, b: int) -> np.ndarray:
    x = np.asarray(x)
    heard = min(len(x), HEARD_MAX)
    if x.dtype != np.float32 or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("waveform must be finite float32 1-D")
    if not (0 <= int(a) < int(b) <= heard):
        raise ValueError("mask bounds outside heard waveform")
    xm = x.copy()
    xm[int(a):int(b)] = np.float32(0.0)
    return xm


def lexical(z, zm, c_E: int, c_M: int) -> dict:
    z, zm = np.asarray(z, dtype=np.float64), np.asarray(zm, dtype=np.float64)
    vals = (z[c_E], z[c_M], zm[c_E], zm[c_M])
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("nonfinite candidate logit")
    M0, Mm = vals[0] - vals[1], vals[2] - vals[3]
    dE, dM = vals[0] - vals[2], vals[1] - vals[3]
    S = M0 - Mm
    if abs(S - (dE - dM)) > IDENTITY_TOL:
        raise ValueError("S_lex identity violated")
    return {"M0": M0, "Mmask": Mm, "Delta_E": dE, "Delta_M": dM, "S_lex": S, "F_lex": max(0.0, math.tanh(S / 2))}


def waveform_model_inputs(bundle, waveform: np.ndarray) -> dict:
    """Exactly batch_model_inputs' processor call, on an in-memory waveform list (no WAV export)."""
    feats = bundle.processor.feature_extractor([waveform], sampling_rate=bundle.sample_rate, return_tensors="pt",
                                               return_attention_mask=True)
    if not hasattr(feats, "attention_mask"):
        raise RuntimeError("Whisper feature extractor returned no attention_mask")
    return {"input_features": feats.input_features.to(bundle.device, bundle.dtype),
            "attention_mask": feats.attention_mask.to(bundle.device)}
