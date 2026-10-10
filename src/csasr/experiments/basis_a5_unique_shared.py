"""CPU-side BASIS-A5 unique/shared direction construction.

The module is deliberately independent of recognition outcomes.  It consumes
uncentered first and second moments and implements the frozen principal-angle
construction from ``BASIS_A5_SPEC.md``.  GPU extraction and decoding live in
``experiments/basis_a5_unique_shared.py``; keeping this numerical core small
makes the construction and rank gates testable without model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ORTHOGONALITY_TOL = 1e-3
UNIT_TOL = 2e-5


def _unit(x: np.ndarray, *, name: str = "vector") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"{name} is degenerate: norm={n}")
    return x / n


def _top_eigenvectors(second: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return descending top eigenpairs of a symmetric second moment."""
    c = np.asarray(second, dtype=np.float64)
    c = (c + c.T) * 0.5
    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1][: int(count)]
    vals = np.maximum(vals[order], 0.0)
    return vecs[:, order], vals


@dataclass
class UniqueSharedResult:
    v_unique: np.ndarray
    v_shared: np.ndarray
    neg_shared: np.ndarray
    pos_unique: np.ndarray
    unique_minus_shared: np.ndarray
    diagnostics: dict[str, Any]


def construct_unique_shared(
    second_a: np.ndarray,
    count_a: int,
    sum_a: np.ndarray,
    second_b: np.ndarray,
    count_b: int,
    sum_b: np.ndarray,
    *,
    rank: int = 32,
    top_vectors: int | None = None,
    _basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> UniqueSharedResult:
    """Construct the frozen A5 directions from uncentered moments.

    ``second_*`` are sums of ``h h^T`` rather than centered covariance
    matrices.  The caller may provide ``top_vectors=64`` so the same retained
    eigensystem supports the r=16/32/64 stability gate.
    """
    if int(count_a) < 1 or int(count_b) < 1:
        raise ValueError("both groups must contain observations")
    sa = np.asarray(second_a, dtype=np.float64)
    sb = np.asarray(second_b, dtype=np.float64)
    if sa.shape != sb.shape or sa.ndim != 2 or sa.shape[0] != sa.shape[1]:
        raise ValueError("second moments must be matching square matrices")
    d = sa.shape[0]
    if np.asarray(sum_a).shape != (d,) or np.asarray(sum_b).shape != (d,):
        raise ValueError("moment sums must match hidden dimension")
    r_eff = min(int(rank), int(count_a), int(count_b), d)
    if r_eff < 1:
        raise ValueError("effective rank is zero")
    k = min(int(top_vectors or r_eff), d, int(count_a), int(count_b))
    k = max(k, r_eff)
    if _basis is None:
        ua, eva = _top_eigenvectors(sa / float(count_a), k)
        ub, evb = _top_eigenvectors(sb / float(count_b), k)
    else:
        ua, eva, ub, evb = _basis
        if ua.shape[1] < k or ub.shape[1] < k:
            raise ValueError("cached eigensystem is smaller than requested rank")
    p, sigma, qt = np.linalg.svd(ua.T @ ub, full_matrices=False)
    a = ua @ p
    b = ub @ qt.T
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, 1.0)
    ea = np.einsum("ij,ij->j", a, (sa / float(count_a)) @ a)
    eb = np.einsum("ij,ij->j", b, (sb / float(count_b)) @ b)
    unique_score = ea * (1.0 - sigma * sigma)
    shared_score = sigma * sigma * np.sqrt(np.maximum(ea, 0.0) * np.maximum(eb, 0.0))
    iu = int(np.argmax(unique_score[:r_eff]))
    shared_candidates = shared_score[:r_eff].copy()
    shared_candidates[iu] = -np.inf
    if not np.isfinite(shared_candidates).any():
        raise ValueError("no distinct shared principal vector")
    is_ = int(np.argmax(shared_candidates))
    vu_raw = a[:, iu]
    pair_dot = float(a[:, is_] @ b[:, is_])
    bp = b[:, is_] * (-1.0 if pair_dot < 0.0 else 1.0)
    vs_raw = _unit(a[:, is_] + bp, name="raw shared direction")
    mu_a = np.asarray(sum_a, dtype=np.float64) / float(count_a)
    mu_b = np.asarray(sum_b, dtype=np.float64) / float(count_b)
    delta = mu_a - mu_b
    delta_norm = float(np.linalg.norm(delta))
    vu = _unit(vu_raw, name="unique direction")
    sign_flip_unique = False
    if delta_norm > 0.0 and float(vu @ (delta / delta_norm)) < 0.0:
        vu = -vu
        sign_flip_unique = True
    pooled = mu_a + mu_b
    pooled_norm = float(np.linalg.norm(pooled))
    fallback = "none"
    vs = vs_raw
    if pooled_norm < 1e-8 * max(float(np.linalg.norm(mu_a)), float(np.linalg.norm(mu_b))):
        fallback = "mu_A"
        mu_a_norm = float(np.linalg.norm(mu_a))
        if mu_a_norm >= 1e-12 and float(vs @ (mu_a / mu_a_norm)) < 0.0:
            vs = -vs
            sign_flip_shared = True
        elif mu_a_norm < 1e-12:
            fallback = "indeterminate"
            sign_flip_shared = False
        else:
            sign_flip_shared = False
    else:
        sign_flip_shared = False
        if float(vs @ (pooled / pooled_norm)) < 0.0:
            vs = -vs
            sign_flip_shared = True
    composite = _unit(vu - vs, name="unique-minus-shared")
    orth_cos = float(vu @ vs)
    diag = {
        "count_a": int(count_a), "count_b": int(count_b),
        "r_requested": int(rank), "r_eff": int(r_eff), "r_eigensystem": int(k),
        "i_unique": iu, "i_shared": is_,
        "sigma_unique": float(sigma[iu]), "sigma_shared": float(sigma[is_]),
        "E_A_unique": float(ea[iu]), "E_A_shared": float(ea[is_]),
        "E_B_shared": float(eb[is_]),
        "s_unique": float(unique_score[iu]), "s_shared": float(shared_score[is_]),
        "norm_unique": float(np.linalg.norm(vu)), "norm_shared": float(np.linalg.norm(vs)),
        "norm_unique_minus_shared_raw": float(np.linalg.norm(vu - vs)),
        "cos_unique_shared": orth_cos,
        "projection_energy_A_unique": float(vu @ (sa / float(count_a)) @ vu),
        "projection_energy_B_unique": float(vu @ (sb / float(count_b)) @ vu),
        "projection_energy_A_shared": float(vs @ (sa / float(count_a)) @ vs),
        "projection_energy_B_shared": float(vs @ (sb / float(count_b)) @ vs),
        "sign_flip_unique": bool(sign_flip_unique),
        "sign_flip_shared": bool(sign_flip_shared),
        "sign_fallback": fallback,
        "finite": bool(np.isfinite(vu).all() and np.isfinite(vs).all()),
        "unit_pass": bool(abs(np.linalg.norm(vu) - 1.0) <= UNIT_TOL and
                           abs(np.linalg.norm(vs) - 1.0) <= UNIT_TOL and
                           abs(np.linalg.norm(composite) - 1.0) <= UNIT_TOL),
        "orthogonality_pass": bool(abs(orth_cos) <= ORTHOGONALITY_TOL),
        "distinct_principal_indices": bool(iu != is_),
    }
    return UniqueSharedResult(vu, vs, -vs, vu, composite, diag)


def rank_directions(second_a, count_a, sum_a, second_b, count_b, sum_b,
                    ranks=(16, 32, 64)) -> dict[str, UniqueSharedResult]:
    """Recompute directions at the frozen stability ranks."""
    top = max(int(r) for r in ranks)
    sa = np.asarray(second_a, dtype=np.float64); sb = np.asarray(second_b, dtype=np.float64)
    ua, eva = _top_eigenvectors(sa / float(count_a), top)
    ub, evb = _top_eigenvectors(sb / float(count_b), top)
    basis = (ua, eva, ub, evb)
    return {str(int(r)): construct_unique_shared(
        sa, count_a, sum_a, sb, count_b, sum_b,
        rank=int(r), top_vectors=top, _basis=basis) for r in ranks}
