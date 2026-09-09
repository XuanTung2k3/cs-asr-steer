"""Configurable steering-basis assembly for the additive learned branch.

This module loads the **already-frozen** DG-03 ``steering_basis_v1`` artifact
(``v_raw``/``v_cond``/``v_local`` at one decoder layer, MC §4) and assembles the
per-mode steering input the learned controller consumes. It never rebuilds a
direction, never rescales the backbone, and never relabels a legacy ``v_nat``
vector; it only *selects and normalises* frozen columns.

The DG-05/DG-06 proposed method (``M*``) uses the ``local_cond`` mode. The
additive branch adds the symmetric ``raw_only`` and ``raw_cond`` modes requested
by the training-expansion ticket, alongside the ``local_only`` / ``cond_only`` /
``fixed_mixture`` ablation directions already instantiated in DG-07. **No mode is
given special treatment**: every rank-1 direction is unit-normalised and every
rank-2 basis has unit-norm columns assembled in a fixed ``[structural, cond]``
order, with identical validation and provenance.

The frozen artifact stores ``v_raw`` with its real magnitude (``v_raw_norm`` ≈
3.81 at L24); ``raw_only`` / ``raw_cond`` therefore normalise ``v_raw`` to a unit
column exactly as ``v_cond`` / ``v_local`` were unit at construction, so the
controller's learned ``beta``·gate carries all magnitude. This keeps the raw
modes on the same footing as the conditioning-residualised modes rather than
letting the raw contrast's larger norm act as a hidden strength advantage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

#: Keys in the frozen ``steering_basis_v1`` artifact (DG-03 direction names).
_RAW_KEY = "raw_language_contrast"
_COND_KEY = "language_conditioning"
_LOCAL_KEY = "conditioning_residualized_local"

#: Supported modes and the rank each produces. ``local_cond`` is the frozen
#: ``M*`` basis; the rest are symmetric siblings. Adding a mode here is the only
#: place the branch enumerates basis choices.
MODE_RANK: dict[str, int] = {
    "local_cond": 2,     # [v_local, v_cond]  == frozen M* (MC §4)
    "raw_cond": 2,       # [normalize(v_raw), v_cond]
    "raw_only": 1,       # normalize(v_raw)
    "local_only": 1,     # v_local
    "cond_only": 1,      # v_cond
    "fixed_mixture": 1,  # normalize(0.5 v_local + 0.5 v_cond)
}

_EPS = 1e-12
_UNIT_TOL = 1e-5


@dataclass(frozen=True)
class BasisMode:
    """A frozen, validated steering input for one mode at one layer.

    ``tensor`` is ``(d,)`` for a rank-1 mode or ``(d, 2)`` for a rank-2 mode with
    unit-norm columns ordered ``[structural, cond]``. ``provenance`` records the
    artifact path, the frozen tensor hashes consumed, and any normalisation
    applied, so a later manifest can prove which frozen columns were used.
    """

    mode: str
    rank: int
    layer: int
    tensor: torch.Tensor
    provenance: Mapping[str, Any]

    def as_controller_input(self) -> torch.Tensor:
        """Return the tensor in the shape the controller factory expects."""
        return self.tensor


def _sha256_f64(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("cannot normalise a non-finite or degenerate direction")
    return vec / norm


def _load_frozen_columns(path: str | Path, *, expected_layer: int,
                         expected_hashes: Mapping[str, str] | None = None,
                         ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and hash-check ``v_raw``/``v_cond``/``v_local`` from the artifact.

    Mirrors ``controller.load_frozen_basis`` hashing but also returns ``v_raw``.
    Every consumed tensor's recorded hash is verified against the artifact, and
    optionally against an explicit ``expected_hashes`` (the caller's frozen
    config) so a relinked artifact cannot silently change the direction.
    """
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    artifact_layer = int(record.get("scientific_layer", record.get("layer", -1)))
    if artifact_layer != int(expected_layer):
        raise ValueError(
            f"basis artifact layer {artifact_layer} != expected {expected_layer}")
    base = path.parent
    cols: dict[str, np.ndarray] = {}
    for key in (_RAW_KEY, _COND_KEY, _LOCAL_KEY):
        arr = np.asarray(np.load(base / record["tensor_files"][key]), dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"frozen direction {key} is not a vector")
        digest = _sha256_f64(arr)
        if digest != record["tensor_hashes"][key]:
            raise ValueError(f"DG-03 basis tensor hash mismatch for {key}")
        if expected_hashes and key in expected_hashes and digest != expected_hashes[key]:
            raise ValueError(f"{key} is not the frozen configured direction")
        cols[key] = arr
    return cols, record


def assemble_mode_tensor(mode: str, *, v_raw: np.ndarray, v_cond: np.ndarray,
                         v_local: np.ndarray) -> tuple[np.ndarray, int, list[str], list[str]]:
    """Assemble the mode tensor from the three frozen columns (single source).

    Returns ``(tensor, rank, columns_used, columns_normalised_here)``. No mode is
    privileged: rank-1 directions are unit-normalised; rank-2 bases have unit
    columns ordered ``[structural, cond]``. ``v_raw`` is normalised to unit length
    exactly as the other columns, so its construction magnitude is not a hidden
    strength. Shared by the DG-03-artifact and per-layer-atlas loaders.
    """
    if mode not in MODE_RANK:
        raise ValueError(f"unknown basis mode {mode!r}; known: {sorted(MODE_RANK)}")
    used: list[str] = []
    normalised: list[str] = []

    def pick(name: str, vec: np.ndarray, *, renormalise: bool) -> np.ndarray:
        used.append(name)
        if renormalise:
            normalised.append(name)
            return _unit(np.asarray(vec, dtype=np.float64))
        if abs(float(np.linalg.norm(vec)) - 1.0) > _UNIT_TOL:
            raise ValueError(f"{name} was expected unit-norm in the source columns")
        return np.asarray(vec, dtype=np.float64)

    if mode == "local_cond":
        tensor = np.stack([pick(_LOCAL_KEY, v_local, renormalise=False),
                           pick(_COND_KEY, v_cond, renormalise=False)], axis=1)
    elif mode == "raw_cond":
        tensor = np.stack([pick(_RAW_KEY, v_raw, renormalise=True),
                           pick(_COND_KEY, v_cond, renormalise=False)], axis=1)
    elif mode == "raw_only":
        tensor = pick(_RAW_KEY, v_raw, renormalise=True)
    elif mode == "local_only":
        tensor = pick(_LOCAL_KEY, v_local, renormalise=False)
    elif mode == "cond_only":
        tensor = pick(_COND_KEY, v_cond, renormalise=False)
    elif mode == "fixed_mixture":
        used.extend([_LOCAL_KEY, _COND_KEY])
        normalised.append("equal_mix")
        tensor = _unit(0.5 * np.asarray(v_local, np.float64) + 0.5 * np.asarray(v_cond, np.float64))
    else:                                                    # pragma: no cover
        raise ValueError(mode)
    rank = MODE_RANK[mode]
    validate_basis_tensor(tensor, rank)
    return tensor, rank, sorted(set(used)), sorted(set(normalised))


def load_basis_mode(path: str | Path, *, mode: str, layer: int,
                    expected_hashes: Mapping[str, str] | None = None) -> BasisMode:
    """Assemble one ``BasisMode`` from the frozen DG-03 ``steering_basis_v1`` artifact."""
    cols, record = _load_frozen_columns(path, expected_layer=layer,
                                        expected_hashes=expected_hashes)
    tensor, rank, used, normalised = assemble_mode_tensor(
        mode, v_raw=cols[_RAW_KEY], v_cond=cols[_COND_KEY], v_local=cols[_LOCAL_KEY])
    provenance = {
        "mode": mode, "rank": rank, "layer": int(layer),
        "basis_source": "dg03_steering_basis_v1",
        "artifact_path": str(path),
        "artifact_schema_version": record.get("schema_version"),
        "site": record.get("site"),
        "columns_used": used,
        "columns_normalised_here": normalised,
        "frozen_tensor_hashes": {k: record["tensor_hashes"][k] for k in used},
        "no_mode_favoured": ("raw columns unit-normalised identically to local/cond; "
                             "magnitude carried only by controller beta*gate"),
    }
    return BasisMode(mode=mode, rank=rank, layer=int(layer),
                     tensor=torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)),
                     provenance=provenance)


def load_basis_mode_from_atlas(atlas_directions_json: str | Path, *, mode: str,
                               layer: int) -> BasisMode:
    """Assemble one ``BasisMode`` from the frozen per-layer atlas ``directions.json``.

    The atlas (``results/basis_frozen_layer_atlas/``) built canonical per-layer
    ``raw``/``conditioning``/``local`` columns on D-construct for all 32 layers.
    This reuses those frozen columns (hash-checked) for candidate layers that have
    no dedicated ``steering_basis_v1`` artifact (e.g. L26/L27), with the **same**
    assembly as the DG-03 path — no mode favoured.
    """
    path = Path(atlas_directions_json)
    record = json.loads(path.read_text(encoding="utf-8"))
    layer_rec = record["layers"][str(int(layer))]
    base = path.parent
    names = {"raw": _RAW_KEY, "conditioning": _COND_KEY, "local": _LOCAL_KEY}
    cols: dict[str, np.ndarray] = {}
    for atlas_key, canonical_key in names.items():
        # directions.json stores repo-relative paths; fall back to the sibling
        # file next to the json if the relative path is not resolvable here.
        rel = Path(layer_rec["files"][atlas_key])
        fpath = rel if rel.exists() else base / rel.name
        arr = np.asarray(np.load(fpath), dtype=np.float64)
        digest = _sha256_f64(arr)
        if digest != layer_rec["hashes"][atlas_key]:
            raise ValueError(f"atlas direction hash mismatch for {atlas_key} at L{layer}")
        cols[canonical_key] = arr
    tensor, rank, used, normalised = assemble_mode_tensor(
        mode, v_raw=cols[_RAW_KEY], v_cond=cols[_COND_KEY], v_local=cols[_LOCAL_KEY])
    provenance = {
        "mode": mode, "rank": rank, "layer": int(layer),
        "basis_source": "basis_frozen_layer_atlas_directions",
        "artifact_path": str(path),
        "columns_used": used,
        "columns_normalised_here": normalised,
        "frozen_tensor_hashes": {names[k]: layer_rec["hashes"][k] for k in names},
        "no_mode_favoured": ("raw columns unit-normalised identically to local/cond; "
                             "magnitude carried only by controller beta*gate"),
    }
    return BasisMode(mode=mode, rank=rank, layer=int(layer),
                     tensor=torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)),
                     provenance=provenance)


def validate_basis_tensor(tensor: np.ndarray | torch.Tensor, rank: int) -> None:
    """Hard invariants shared by every mode (finite, unit columns, right shape)."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if not np.isfinite(arr).all():
        raise ValueError("basis tensor contains non-finite values")
    if rank == 1:
        if arr.ndim != 1:
            raise ValueError(f"rank-1 mode expects a vector, got shape {arr.shape}")
        if abs(float(np.linalg.norm(arr)) - 1.0) > _UNIT_TOL:
            raise ValueError("rank-1 direction is not unit-norm")
    elif rank == 2:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"rank-2 mode expects (d, 2), got shape {arr.shape}")
        norms = np.linalg.norm(arr, axis=0)
        if np.any(np.abs(norms - 1.0) > _UNIT_TOL):
            raise ValueError(f"rank-2 columns are not unit-norm (norms={norms})")
    else:
        raise ValueError(f"unsupported rank {rank}")
