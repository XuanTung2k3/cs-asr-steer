"""Shared, deterministic specifications for the three-round programme.

This module is deliberately model independent.  The runners use it to build
cell manifests before loading a model, which makes dry-runs and resume safe.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROUND1_LAYERS = (8, 12, 16, 20, 22, 24, 26)
COARSE_MIXTURES = ((1.0, 0.0), (4.0, 1.0), (0.0, 1.0))
DETAIL_MIXTURES = ((1.0, 0.0), (4.0, 1.0), (2.0, 1.0), (1.0, 1.0),
                   (1.0, 2.0), (1.0, 4.0), (0.0, 1.0))
SCOPES = ("oracle_local", "global")
BETAS_COARSE = (1.0, 2.0)
BETAS_DETAIL = (0.5, 1.0, 2.0)
PREFIX_WIDTH = 4


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class RoundCell:
    stage: str
    layer: int
    mixture: tuple[float, float]
    scope: str
    beta: float
    beam: int = 1
    condition: str = "steering"

    @property
    def cell_id(self) -> str:
        return f"{self.stage}_L{self.layer}_N{self.mixture[0]:g}_P{self.mixture[1]:g}_{self.scope}_b{self.beta:g}_beam{self.beam}_{self.condition}"

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["mixture"] = list(self.mixture)
        out["cell_id"] = self.cell_id
        out["config_hash"] = self.config_hash
        return out


def enumerate_coarse() -> list[RoundCell]:
    cells = [RoundCell("coarse", l, m, s, b)
             for l in ROUND1_LAYERS for m in COARSE_MIXTURES
             for s in SCOPES for b in BETAS_COARSE]
    assert len(cells) == 84
    return cells


def enumerate_detail(selected_layers: Sequence[int]) -> list[RoundCell]:
    layers = tuple(dict.fromkeys(int(x) for x in selected_layers))
    if len(layers) != 3:
        raise ValueError("detailed sweep requires exactly three unique layers")
    cells = [RoundCell("detail", l, m, s, b)
             for l in layers for m in DETAIL_MIXTURES
             for s in SCOPES for b in BETAS_DETAIL]
    assert len(cells) == 126
    return cells


def enumerate_localization(best: Sequence[tuple[int, tuple[float, float], float]]) -> list[RoundCell]:
    scopes = ("oracle_local", "global", "auto_soft_local", "constant_matched",
              "shuffled_matched", "random_local")
    return [RoundCell("localization", int(layer), tuple(mix), scope, float(beta))
            for layer, mix, beta in best for scope in scopes]


def enumerate_negative(layer: int, beta: float) -> list[RoundCell]:
    return [RoundCell("negative", int(layer), m, "global", float(beta))
            for m in ((-1.0, 0.0), (0.0, -1.0), (1.0, -1.0), (-1.0, 1.0))]


def normalize_direction(v: Any, eps: float = 1e-8):
    """Return a finite unit vector, rejecting a zero vector."""
    import torch
    t = torch.as_tensor(v)
    n = t.norm()
    if not bool(torch.isfinite(n)) or float(n) < eps:
        raise ValueError("direction norm is zero, non-finite, or below tolerance")
    return t / n


def mixture_direction(v_nat: Any, v_prompt: Any, cn: float, cp: float,
                      eps: float = 1e-8):
    """Normalize directions first; coefficients define only a ratio."""
    import torch
    n = normalize_direction(v_nat, eps)
    p = normalize_direction(v_prompt, eps)
    raw = float(cn) * n + float(cp) * p
    norm = raw.norm()
    if not bool(torch.isfinite(norm)) or float(norm) < eps:
        raise ValueError("mixture is degenerate after unit-direction normalization")
    return raw / norm


def deterministic_gate_controls(values: Sequence[float], coverage: int, *, seed: int,
                                utterance_id: str) -> dict[str, list[float]]:
    """Build automatic-gate controls without using gold localization."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("gate values must be one-dimensional")
    rng = np.random.default_rng(int(seed) ^ int(stable_hash(utterance_id)[:8], 16))
    k = max(0, min(int(coverage), len(x)))
    order = np.argsort(-x, kind="stable")
    auto = np.zeros_like(x)
    auto[order[:k]] = x[order[:k]]
    shuffled = auto[rng.permutation(len(auto))]
    random_local = np.zeros_like(x)
    if k:
        random_local[rng.choice(len(x), size=k, replace=False)] = float(auto[order[:k]].mean())
    return {
        "auto_soft_local": auto.tolist(),
        "constant_matched": [float(auto.sum() / len(auto)) if len(auto) else 0.0] * len(auto),
        "shuffled_matched": shuffled.tolist(),
        "random_local": random_local.tolist(),
    }


def assert_disjoint_manifests(manifests: dict[str, Iterable[dict[str, Any]]]) -> dict[str, Any]:
    """Check utterance/dialogue/id disjointness and duplicate-free manifests."""
    report: dict[str, Any] = {"roles": {}, "overlap_errors": []}
    seen_u: dict[str, str] = {}
    seen_d: dict[str, str] = {}
    seen_i: dict[str, str] = {}
    for role, rows in manifests.items():
        rows = list(rows)
        us = [str(r.get("utterance_id", r.get("id", ""))) for r in rows]
        ds = [str(r.get("dialogue_id", "")) for r in rows]
        ids = [str(r.get("id", r.get("utterance_id", ""))) for r in rows]
        for label, vals, seen in (("utterance", us, seen_u), ("dialogue", ds, seen_d), ("id", ids, seen_i)):
            if len(vals) != len(set(vals)):
                report["overlap_errors"].append(f"duplicate {label} in {role}")
            for value in vals:
                if value and value in seen and seen[value] != role:
                    report["overlap_errors"].append(f"{label} {value} in {seen[value]} and {role}")
                seen[value] = role
        report["roles"][role] = {"utterances": len(set(us)), "dialogues": len(set(ds)), "ids": len(set(ids))}
    report["ok"] = not report["overlap_errors"]
    if not report["ok"]:
        raise AssertionError("split disjointness failed: " + "; ".join(report["overlap_errors"]))
    return report


def atomic_json(path: Path, payload: Any) -> None:
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise
