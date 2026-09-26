"""Canonical v6 steering-basis builder (DG-03).

Assembles the frozen contrastive basis ``V^0 = [v_local, v_cond]`` at a decoder
layer, from exact-site (post-cross-attention, pre-FFN) states, in the **v6
aggregate-then-residualize order** (proposal v6 §4; METHOD_CONTRACT §4):

    v_raw   = μ_E^correct − μ_M^correct         (dialogue-balanced mean of per-span E−M)
    v_cond  = normalize( E[ r(c_E) − r(c_M) ] ) (language-conditioning direction)
    v_local = normalize( v_raw − ⟨v_raw, v_cond⟩ v_cond )
    V^0     = [ v_local, v_cond ]

This module is pure NumPy assembly + artifact schema + validation, so it is
unit-testable on CPU with no model. The GPU state capture lives in the DG-03
construction runner. It deliberately does **not** reuse the legacy
``v2r3_directions.assemble`` (which projects conditioning out of each sample
*before* aggregation — the wrong order for v6), and it must **not** be fed a
legacy ``v_nat`` artifact as ``v_raw``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BASIS_SCHEMA_VERSION = "steering_basis_v1"
#: Only these decoder layers may carry a contract basis (MC §3).
CONTRACT_LAYERS = (16, 24)
#: Conservative direction names (never "pure acoustic" / "universal" / "causal").
DIRECTION_NAMES = ("raw_language_contrast", "language_conditioning",
                   "conditioning_residualized_local")
_ORTHO_TOL = 1e-5
_EPS = 1e-12


def _as2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (n, dim) array, got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError("empty population: no rows to aggregate")
    if not np.isfinite(a).all():
        raise ValueError("non-finite values in input states")
    return a


def _balanced_mean(rows: np.ndarray, groups: Sequence[Any] | None) -> np.ndarray:
    """Mean of per-group means (equal weight per dialogue); plain mean if no groups."""
    rows = _as2d(rows)
    if groups is None:
        return rows.mean(axis=0)
    groups = list(groups)
    if len(groups) != rows.shape[0]:
        raise ValueError(f"groups ({len(groups)}) != rows ({rows.shape[0]})")
    per_group = {}
    for g, r in zip(groups, rows):
        per_group.setdefault(g, []).append(r)
    means = [np.mean(np.stack(v), axis=0) for v in per_group.values()]
    return np.mean(np.stack(means), axis=0)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < _EPS:
        raise ValueError("degenerate (zero/non-finite) vector cannot be normalized")
    return v / n


def raw_language_contrast(contrasts_EM: np.ndarray,
                          groups: Sequence[Any] | None = None) -> np.ndarray:
    """v_raw = μ_E − μ_M, as the dialogue-balanced mean of per-span (r_E − r_M).

    ``contrasts_EM`` is (n_spans, dim) with each row = r_E − r_M for one span.
    Not normalized (v_raw keeps its magnitude; only v_cond/v_local are unit).
    """
    return _balanced_mean(_as2d(contrasts_EM), groups)


def raw_from_states(states_E: np.ndarray, states_M: np.ndarray,
                    groups: Sequence[Any] | None = None) -> np.ndarray:
    """Convenience: v_raw from paired E and M state rows (row i pairs E_i, M_i)."""
    e, m = _as2d(states_E), _as2d(states_M)
    if e.shape != m.shape:
        raise ValueError(f"E {e.shape} and M {m.shape} must pair 1:1")
    return raw_language_contrast(e - m, groups)


def language_conditioning(cond_contrasts: np.ndarray,
                          groups: Sequence[Any] | None = None) -> np.ndarray:
    """v_cond = normalize(balanced mean of per-utterance mean(r(c_E) − r(c_M)))."""
    return _normalize(_balanced_mean(_as2d(cond_contrasts), groups))


def conditioning_residualized_local(v_raw: np.ndarray, v_cond: np.ndarray) -> np.ndarray:
    """v_local = normalize(v_raw − ⟨v_raw, v_cond⟩ v_cond). Order: aggregate THEN residualize."""
    v_raw = np.asarray(v_raw, dtype=np.float64)
    v_cond = _normalize(v_cond)          # ensure unit before projection
    residual = v_raw - float(v_raw @ v_cond) * v_cond
    return _normalize(residual)


def _sha256_array(a: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def build_basis(*, layer: int, contrasts_EM: np.ndarray, cond_contrasts: np.ndarray,
                groups_raw: Sequence[Any] | None = None,
                groups_cond: Sequence[Any] | None = None,
                enforce_contract_layer: bool = True) -> dict[str, Any]:
    """Assemble V^0 at one layer and compute the report metrics + validation.

    Returns a dict with float64 vectors ``v_raw``/``v_cond``/``v_local``/``V0``
    and a ``metrics`` block. Raises on degeneracy or a broken invariant.
    """
    layer = int(layer)
    if enforce_contract_layer and layer not in CONTRACT_LAYERS:
        raise ValueError(f"layer {layer} not in contract layers {CONTRACT_LAYERS}")
    v_raw = raw_language_contrast(contrasts_EM, groups_raw)
    v_cond = language_conditioning(cond_contrasts, groups_cond)
    v_local = conditioning_residualized_local(v_raw, v_cond)
    V0 = np.stack([v_local, v_cond], axis=1)             # (dim, 2): columns [local, cond]

    raw_norm = float(np.linalg.norm(v_raw))
    proj = float(v_raw @ v_cond)                          # v_cond is unit
    cos_raw_cond = proj / (raw_norm + _EPS)
    removed_fraction = (proj ** 2) / (raw_norm ** 2 + _EPS)   # energy of v_raw along v_cond
    cos_local_cond = float(v_local @ v_cond)

    basis = {
        "schema_version": BASIS_SCHEMA_VERSION,
        "layer": layer,
        "dim": int(v_raw.shape[0]),
        "v_raw": v_raw, "v_cond": v_cond, "v_local": v_local, "V0": V0,
        "metrics": {
            "n_raw_contrasts": int(np.asarray(contrasts_EM).shape[0]),
            "n_cond_contrasts": int(np.asarray(cond_contrasts).shape[0]),
            "v_raw_norm": raw_norm,
            "v_cond_norm": float(np.linalg.norm(v_cond)),
            "v_local_norm": float(np.linalg.norm(v_local)),
            "cos_raw_cond": cos_raw_cond,
            "cos_local_cond": cos_local_cond,
            "conditioning_fraction_removed": removed_fraction,
        },
    }
    validate_basis(basis)
    return basis


def validate_basis(basis: dict[str, Any]) -> None:
    """Hard invariants (spec §4). Raises ValueError on any violation."""
    for name in ("v_raw", "v_cond", "v_local", "V0"):
        if not np.isfinite(basis[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    dim = int(basis["dim"])
    for name in ("v_raw", "v_cond", "v_local"):
        if basis[name].shape != (dim,):
            raise ValueError(f"{name} has shape {basis[name].shape}, expected ({dim},)")
    if basis["V0"].shape != (dim, 2):
        raise ValueError(f"V0 has shape {basis['V0'].shape}, expected ({dim}, 2)")
    if abs(basis["metrics"]["v_cond_norm"] - 1.0) > 1e-6:
        raise ValueError("v_cond is not unit-norm")
    if abs(basis["metrics"]["v_local_norm"] - 1.0) > 1e-6:
        raise ValueError("v_local is not unit-norm")
    if abs(basis["metrics"]["cos_local_cond"]) > _ORTHO_TOL:
        raise ValueError(f"v_local not orthogonal to v_cond "
                         f"(|cos|={abs(basis['metrics']['cos_local_cond']):.2e} > {_ORTHO_TOL})")


def diagnostic_dose(scale: float, rho: float = 1.0) -> dict[str, float]:
    """Pre-registered DG-03 diagnostic dose (spec §5): nominal ‖ũ‖ = ρ·s_ℓ.

    Applied through the frozen hook as ``alpha=ρ``, ``scale=s_ℓ`` on a unit
    direction; all controls reuse the same (alpha, scale) so their nominal edit
    magnitude matches. NormPreserve is applied by the hook.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale (mean site-norm) must be positive and finite")
    return {"alpha": float(rho), "scale": scale, "rho": float(rho),
            "nominal_update_norm": float(rho) * scale}


def write_basis_artifact(out_dir: str | Path, basis: dict[str, Any],
                         provenance: dict[str, Any]) -> dict[str, Any]:
    """Serialize one layer's basis to ``steering_basis_v1`` (tensors + deterministic JSON).

    ``provenance`` supplies the non-derivable metadata (model id/revision, git
    commit, dataset role/fingerprint, config hash, seeds, scale s_ℓ, reuse
    provenance, …). Missing metadata must be passed explicitly as null by the
    caller — this function never fabricates it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = int(basis["layer"])
    files, hashes = {}, {}
    for key, name in zip(("v_raw", "v_cond", "v_local"), DIRECTION_NAMES):
        arr = np.ascontiguousarray(basis[key], dtype=np.float64)
        fpath = out_dir / f"{name}_L{layer}.npy"
        np.save(fpath, arr)
        files[name] = fpath.name
        hashes[name] = _sha256_array(arr)
    record = {
        "schema_version": BASIS_SCHEMA_VERSION,
        "scientific_layer": layer,
        "python_layer_index": layer,
        "site": "decoder_post_cross_attn_residual",
        "dim": int(basis["dim"]),
        "direction_names": list(DIRECTION_NAMES),
        "tensor_files": files,
        "tensor_hashes": hashes,
        "tensor_dtype": "float64",
        "metrics": basis["metrics"],
        "provenance": provenance,
    }
    jpath = out_dir / f"{BASIS_SCHEMA_VERSION}_L{layer}.json"
    jpath.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False),
                     encoding="utf-8")
    record["_json_path"] = str(jpath)
    return record
