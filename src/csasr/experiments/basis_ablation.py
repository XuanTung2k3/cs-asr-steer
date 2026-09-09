"""CPU-side primitives for BASIS-A's frozen direction comparison.

This module deliberately contains no model code.  It freezes the vector
construction and geometry calculations used by the one-process evaluator so
the GPU path and the focused tests share the same numerical definitions.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a finite float64 unit vector, refusing a zero/non-finite input."""
    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("direction must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(x))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must have a finite non-zero norm")
    return x / norm


def direction_hash(vector: np.ndarray) -> str:
    """Hash the canonical contiguous float64 representation used by DG-03."""
    x = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
    return "sha256:" + hashlib.sha256(x.tobytes()).hexdigest()


def construct_directions(v_raw: np.ndarray, v_local: np.ndarray,
                         v_cond: np.ndarray, *, a_local: float = 0.5,
                         a_cond: float = 0.5) -> dict[str, np.ndarray]:
    """Construct the five physical unit directions in the BASIS-A matrix."""
    r = normalize(v_raw)
    l = normalize(v_local)
    c = normalize(v_cond)
    rc = normalize(float(a_local) * r + float(a_cond) * c)
    lc = normalize(float(a_local) * l + float(a_cond) * c)
    return {"raw": r, "local": l, "conditioning": c,
            "raw_cond": rc, "local_cond": lc}


def dose(direction: np.ndarray, rho: float, scale: float) -> np.ndarray:
    """Return the physical nominal update; all input directions must be unit."""
    d = normalize(direction)
    if not np.isfinite(rho) or not np.isfinite(scale):
        raise ValueError("rho and scale must be finite")
    return float(rho) * float(scale) * d


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0))


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(_cosine(a, b))))


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode="reduced")
    return q


def principal_angles_degrees(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Principal angles between column spaces, in nondecreasing degrees."""
    qa = _orthonormal_columns(a)
    qb = _orthonormal_columns(b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    # SVD round-off can turn an exactly shared axis into 1-ε and an apparent
    # micro-degree angle.  Preserve the geometric zero at machine precision.
    singular[np.isclose(singular, 1.0, atol=1e-14, rtol=0.0)] = 1.0
    return [float(x) for x in np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))]


def projection_matrix(matrix: np.ndarray) -> np.ndarray:
    q = _orthonormal_columns(matrix)
    return q @ q.T


def _effective_rank(singular: np.ndarray, shape: tuple[int, int]) -> int:
    if not len(singular) or singular[0] == 0.0:
        return 0
    tol = np.finfo(np.float64).eps * max(shape) * float(singular[0])
    return int(np.sum(singular > tol))


def basis_geometry(directions: Mapping[str, np.ndarray], *, a_local: float,
                   a_cond: float) -> dict:
    """Compute the preregistered pairwise, Gram, SVD and subspace diagnostics."""
    names = ["raw", "local", "conditioning", "raw_cond", "local_cond"]
    matrix = np.asarray([[ _cosine(directions[x], directions[y]) for y in names]
                         for x in names], dtype=np.float64)
    r, l, c = (directions[x] for x in ("raw", "local", "conditioning"))
    rc, lc = directions["raw_cond"], directions["local_cond"]
    alpha = _cosine(r, c)
    residual = r - alpha * c
    b_raw = np.column_stack([r, c])
    b_local = np.column_stack([l, c])
    svd = {}
    for key, b in (("raw", b_raw), ("local", b_local)):
        s = np.linalg.svd(b, compute_uv=False)
        svd[key] = {
            "singular_values": [float(x) for x in s],
            "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else None,
            "effective_rank": _effective_rank(s, b.shape),
        }
    p_raw = projection_matrix(b_raw)
    p_local = projection_matrix(b_local)
    return {
        "direction_order": names,
        "cosine_matrix": matrix.tolist(),
        "pairwise": {
            "cos_raw_local": _cosine(r, l),
            "angle_raw_local_degrees": angle_degrees(r, l),
            "cos_raw_cond": _cosine(r, c),
            "angle_raw_cond_degrees": angle_degrees(r, c),
            "cos_local_cond": _cosine(l, c),
            "angle_local_cond_degrees": angle_degrees(l, c),
            "cos_raw_cond_mixture_local_cond_mixture": _cosine(rc, lc),
            "angle_raw_cond_mixture_local_cond_mixture_degrees": angle_degrees(rc, lc),
            "cos_raw_cond_mixture_raw": _cosine(rc, r),
            "angle_raw_cond_mixture_raw_degrees": angle_degrees(rc, r),
            "cos_raw_cond_mixture_cond": _cosine(rc, c),
            "angle_raw_cond_mixture_cond_degrees": angle_degrees(rc, c),
            "cos_local_cond_mixture_local": _cosine(lc, l),
            "angle_local_cond_mixture_local_degrees": angle_degrees(lc, l),
            "cos_local_cond_mixture_cond": _cosine(lc, c),
            "angle_local_cond_mixture_cond_degrees": angle_degrees(lc, c),
        },
        "gram_matrices": {"raw": (b_raw.T @ b_raw).tolist(),
                          "local": (b_local.T @ b_local).tolist()},
        "svd": svd,
        "residualization": {
            "alpha_raw_cond": alpha,
            "removed_energy_fraction": float(alpha * alpha),
            "residual_norm_before_renorm": float(np.linalg.norm(residual)),
            "raw_local_l2_distance": float(np.linalg.norm(r - l)),
            "raw_local_angle_degrees": angle_degrees(r, l),
        },
        "subspace": {
            "principal_angles_degrees": principal_angles_degrees(b_raw, b_local),
            "projection_frobenius_distance": float(np.linalg.norm(p_raw - p_local, ord="fro")),
            "raw_projection_matrix": p_raw.tolist(),
            "local_projection_matrix": p_local.tolist(),
            "equivalent_within_1e-10": bool(np.allclose(p_raw, p_local, atol=1e-10, rtol=1e-10)),
        },
        "mixture_coefficients": {"a_local": float(a_local), "a_cond": float(a_cond)},
    }


def fit_pca_rows(rows: np.ndarray, n_components: int = 3) -> dict:
    """Fit PCA to representation *rows*, never to steering vectors."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("PCA requires a 2-D representation-row matrix with >=2 rows")
    if not np.all(np.isfinite(x)):
        raise ValueError("representation rows contain non-finite values")
    mean = x.mean(axis=0)
    centered = x - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular * singular / max(1, x.shape[0] - 1)
    total = float(variance.sum())
    ratios = variance / total if total else np.zeros_like(variance)
    k = min(int(n_components), vt.shape[0])
    return {"mean": mean, "components": vt[:k],
            "explained_variance": variance[:k],
            "explained_variance_ratio": ratios[:k],
            "n_rows": int(x.shape[0]), "n_features": int(x.shape[1])}


def project_pca_directions(pca: Mapping[str, np.ndarray],
                           directions: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    components = np.asarray(pca["components"], dtype=np.float64)
    return {name: (components @ normalize(vector)).tolist()
            for name, vector in directions.items()}
