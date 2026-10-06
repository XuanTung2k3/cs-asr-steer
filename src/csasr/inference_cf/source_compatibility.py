"""P2-SEL-XA source-compatibility quantities (frozen contract configs/inference_cf/p2_sel_xa.json).

Inputs are only the raw (pre-tangent) float32 readout gradient g_J = dJ/dh of the reference-free
script-mass objective J = log(P_E+1e-12) - log(P_M+1e-12) at the current L16 DG-02 site, and the exact
current-query cross-attention source contribution u_source (encoder_attn output[0]). No label,
reference, evaluator or future quantity enters.

    S_src = dot(g_J, u_source)                          (float64)
    C_src = S_src / (norm(g_J) * norm(u_source) + 1e-12)  (descriptive only)
    F_src = min(1, max(0, S_src))                       (sole gate factor)
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def compatibility(g, u) -> dict:
    g64 = np.asarray(g, dtype=np.float64).reshape(-1)
    u64 = np.asarray(u, dtype=np.float64).reshape(-1)
    if g64.shape != u64.shape:
        raise ValueError("shape mismatch")
    if not (np.isfinite(g64).all() and np.isfinite(u64).all()):
        raise ValueError("nonfinite input")
    gn, un = float(np.linalg.norm(g64)), float(np.linalg.norm(u64))
    if gn == 0.0 or un == 0.0:
        return {"S": 0.0, "C": 0.0, "F": 0.0, "g_norm": gn, "u_norm": un}
    S = float(g64 @ u64)
    C = S / (gn * un + EPS)
    if not (math.isfinite(S) and math.isfinite(C)):
        raise ValueError("nonfinite result")
    if not (-1 - 1e-6 <= C <= 1 + 1e-6) or (S > 0) != (C > 0) or (S < 0) != (C < 0):
        raise ValueError("S/C numerical inconsistency")
    return {"S": S, "C": C, "F": min(1.0, max(0.0, S)), "g_norm": gn, "u_norm": un}
