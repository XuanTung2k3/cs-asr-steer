"""Frozen A6 fixed and oracle-per-sample direction construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..experiments.basis_a5_unique_shared import construct_unique_shared
from ..utils.hashing import sha256_bytes

EPS = 1e-12


def normalize(v: np.ndarray, *, name: str = "direction") -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError(f"{name} is non-finite or zero")
    return out / norm


def direction_hash(v: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
    return "sha256:" + sha256_bytes(arr.tobytes())


def raw_direction(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(states_a, dtype=np.float64), np.asarray(states_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError("raw direction requires non-empty A/B states with matching dimensions")
    return normalize(a.mean(axis=0) - b.mean(axis=0), name="raw")


@dataclass(frozen=True)
class UniqueShared:
    v_unique: np.ndarray
    v_shared: np.ndarray
    v_unique_minus_shared: np.ndarray
    rank: int
    diagnostics: dict[str, Any]


def dynamic_rank(n_a: int, n_b: int, hidden_dim: int) -> int:
    return min(32, int(n_a), int(n_b), int(hidden_dim))


def unique_shared_direction(states_a: np.ndarray, states_b: np.ndarray) -> UniqueShared | None:
    a, b = np.asarray(states_a, dtype=np.float64), np.asarray(states_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError("unique/shared requires non-empty A/B states with matching dimensions")
    rank = dynamic_rank(len(a), len(b), a.shape[1])
    if rank < 2:
        return None
    built = construct_unique_shared(a.T @ a, len(a), a.sum(axis=0),
                                    b.T @ b, len(b), b.sum(axis=0), rank=rank)
    if abs(float(built.v_unique @ built.v_shared)) > 1e-3:
        raise ValueError("A5 unique/shared orthogonality gate failed")
    return UniqueShared(built.v_unique, built.v_shared, built.unique_minus_shared,
                        rank, dict(built.diagnostics))


def conditioning_direction(deltas: np.ndarray, positions: np.ndarray | list[int] | None = None) -> np.ndarray:
    x = np.asarray(deltas, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError("conditioning requires at least one aligned delta")
    if positions is not None:
        idx = np.asarray(positions, dtype=int)
        if idx.ndim != 1 or not len(idx):
            raise ValueError("conditioning position selection is empty")
        x = x[idx]
    return normalize(x.mean(axis=0), name="conditioning")


def build_method_directions(states_a: np.ndarray, states_b: np.ndarray,
                            *, conditioning_deltas: np.ndarray | None = None,
                            conditioning_cs_positions: np.ndarray | list[int] | None = None) -> dict[str, np.ndarray]:
    """Build the methods available for one source/sample/layer.

    The caller controls the A/B state selection (including oracle-local
    eligibility). No direction is borrowed from another utterance.
    """
    out = {"raw": raw_direction(states_a, states_b)}
    us = unique_shared_direction(states_a, states_b)
    if us is not None:
        out.update({"add_unique": us.v_unique, "minus_shared": -us.v_shared,
                    "unique_minus_shared": us.v_unique_minus_shared})
    if conditioning_deltas is not None:
        out["conditioning_all"] = conditioning_direction(conditioning_deltas)
        if conditioning_cs_positions is not None:
            out["conditioning_cs"] = conditioning_direction(conditioning_deltas, conditioning_cs_positions)
    return {name: normalize(vec, name=name) for name, vec in out.items()}


def build_fixed_source(*, groups_a: list[np.ndarray], groups_b: list[np.ndarray],
                       conditioning_all: np.ndarray | None = None,
                       conditioning_cs: np.ndarray | None = None) -> dict[str, Any]:
    """Aggregate all construction representations, then construct one source vector."""
    if not groups_a or not groups_b:
        raise ValueError("fixed construction requires non-empty source groups")
    a = np.concatenate([np.asarray(x, dtype=np.float64) for x in groups_a], axis=0)
    b = np.concatenate([np.asarray(x, dtype=np.float64) for x in groups_b], axis=0)
    out = build_method_directions(a, b)
    if conditioning_cs is not None:
        out["conditioning_cs"] = conditioning_direction(conditioning_cs)
    if conditioning_all is not None:
        out["conditioning_all"] = conditioning_direction(conditioning_all)
    return {"directions": out, "n_A": len(a), "n_B": len(b),
            "direction_hashes": {k: direction_hash(v) for k, v in out.items()}}
