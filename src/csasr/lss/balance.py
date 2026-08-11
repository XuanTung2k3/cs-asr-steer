"""Balanced assignment of conversations to data roles.

Sorted-identifier slicing ("conversations 1-70 are D-construct") is
deterministic but not balanced: corpus identifiers usually preserve collection
order, so slicing can hand one role systematically longer recordings, a
different device mix, or a different code-switching rate. Any difference between
roles then shows up later as a method effect.

The assignment here is seeded **rerandomization** (Morgan & Rubin): draw many
size-respecting partitions, score each by how far its role means sit from the
corpus means, and keep the best. The score distribution of the rejected draws is
reported too, so the achieved balance can be read against what chance would
have produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.hashing import sha256_strings


@dataclass(frozen=True)
class BalanceSpec:
    continuous: tuple[str, ...] = ("hours", "utterances", "cs_utterances", "cs_rate",
                                   "embedded_en_units", "mean_duration_sec")
    categorical: tuple[str, ...] = ("gender", "device", "region")
    distributional: tuple[str, ...] = ("topics",)
    #: categoricals whose imbalance is a *gate*; the rest are optimized and
    #: reported but cannot be thresholded (see `gated_categorical` below)
    gated_categorical: tuple[str, ...] = ("gender", "device")
    max_abs_smd: float = 0.25
    max_categorical_tv: float = 0.15
    draws: int = 20000

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "BalanceSpec":
        block = dict(cfg or {})
        return cls(
            continuous=tuple(block.get("continuous", cls.continuous)),
            categorical=tuple(block.get("categorical", cls.categorical)),
            distributional=tuple(block.get("distributional", cls.distributional)),
            gated_categorical=tuple(block.get("gated_categorical", cls.gated_categorical)),
            max_abs_smd=float(block.get("max_abs_smd", cls.max_abs_smd)),
            max_categorical_tv=float(block.get("max_categorical_tv", cls.max_categorical_tv)),
            draws=int(block.get("draws", cls.draws)),
        )

    def is_gated(self, covariate: str) -> bool:
        return covariate in self.gated_categorical


@dataclass
class _Design:
    """Numeric view of the conversations being partitioned."""

    ids: list[str]
    continuous: np.ndarray               # (n, k) standardized
    continuous_names: list[str]
    blocks: dict[str, np.ndarray]        # name -> (n, levels) proportions
    block_levels: dict[str, list[str]] = field(default_factory=dict)


def _standardize(values: np.ndarray) -> np.ndarray:
    sd = values.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (values - values.mean(axis=0)) / sd


def _proportion_block(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """One-hot (categorical) or normalized multi-hot (list-valued) block."""
    levels: list[str] = sorted({
        str(v)
        for entry in series
        for v in (entry if isinstance(entry, (list, tuple, np.ndarray)) else [entry])
        if str(v)
    })
    if not levels:
        return np.zeros((len(series), 0)), []
    index = {lv: i for i, lv in enumerate(levels)}
    out = np.zeros((len(series), len(levels)), dtype=float)
    for row, entry in enumerate(series):
        values = entry if isinstance(entry, (list, tuple, np.ndarray)) else [entry]
        values = [str(v) for v in values if str(v)]
        if not values:
            continue
        weight = 1.0 / len(values)
        for v in values:
            out[row, index[v]] += weight
    return out, levels


def build_design(features: pd.DataFrame, spec: BalanceSpec) -> _Design:
    cont_names = [c for c in spec.continuous if c in features.columns]
    cont = _standardize(features[cont_names].astype(float).to_numpy()) if cont_names \
        else np.zeros((len(features), 0))
    blocks: dict[str, np.ndarray] = {}
    levels: dict[str, list[str]] = {}
    for name in list(spec.categorical) + list(spec.distributional):
        if name not in features.columns:
            continue
        block, lv = _proportion_block(features[name])
        if lv:
            blocks[name] = block
            levels[name] = lv
    return _Design(ids=[str(v) for v in features["conversation_id"]],
                   continuous=cont, continuous_names=cont_names,
                   blocks=blocks, block_levels=levels)


def _role_means(matrix: np.ndarray, indicator: np.ndarray) -> np.ndarray:
    """(roles, cols) means given a (roles, n) 0/1 indicator."""
    counts = indicator.sum(axis=1, keepdims=True)
    counts = np.where(counts > 0, counts, 1.0)
    return (indicator @ matrix) / counts


#: weight of ungated covariates in the objective. They are optimized but must
#: not dominate: a 28-level covariate has a TV floor around 0.21 for a
#: 20-conversation role, and inside a plain `max` that floor would swamp every
#: attainable improvement in the covariates that are actually gated. Measured on
#: this corpus, 0.3 keeps region better than a random draw (0.31 vs a ~0.38
#: chance median) while leaving the gated criteria comfortably inside their
#: thresholds; the objective is flat above 0.3.
UNGATED_WEIGHT = 0.3


def _score_from_indicator(design: _Design, indicator: np.ndarray,
                          spec: "BalanceSpec") -> tuple[float, float, float, float]:
    """(objective, max |SMD|, max gated TV, max ungated TV) for one partition."""
    max_smd = 0.0
    if design.continuous.shape[1]:
        means = _role_means(design.continuous, indicator)     # corpus mean is 0
        max_smd = float(np.abs(means).max())
    max_tv_gated = 0.0
    max_tv_ungated = 0.0
    for name, block in design.blocks.items():
        overall = block.mean(axis=0)
        means = _role_means(block, indicator)
        tv = float((0.5 * np.abs(means - overall).sum(axis=1)).max())
        if spec.is_gated(name):
            max_tv_gated = max(max_tv_gated, tv)
        else:
            max_tv_ungated = max(max_tv_ungated, tv)
    objective = max(max_smd, max_tv_gated) + UNGATED_WEIGHT * max_tv_ungated
    return objective, max_smd, max_tv_gated, max_tv_ungated


def _indicator(order: np.ndarray, sizes: Sequence[int], n: int) -> np.ndarray:
    ind = np.zeros((len(sizes), n), dtype=float)
    start = 0
    for r, size in enumerate(sizes):
        ind[r, order[start:start + size]] = 1.0
        start += size
    return ind


def imbalance(features: pd.DataFrame, assignment: Mapping[str, str],
              spec: BalanceSpec) -> pd.DataFrame:
    """Per role x covariate balance diagnostics for a given assignment."""
    design = build_design(features, spec)
    roles = sorted(set(assignment.values()))
    role_index = {r: i for i, r in enumerate(roles)}
    ind = np.zeros((len(roles), len(design.ids)), dtype=float)
    for j, conv in enumerate(design.ids):
        role = assignment.get(conv)
        if role is not None:
            ind[role_index[role], j] = 1.0

    rows: list[dict[str, Any]] = []
    raw = features.set_index("conversation_id")
    if design.continuous.shape[1]:
        means = _role_means(design.continuous, ind)
        for r, role in enumerate(roles):
            members = [c for c, v in assignment.items() if v == role]
            for k, name in enumerate(design.continuous_names):
                rows.append({
                    "role": role, "covariate": name, "kind": "continuous",
                    "role_mean": float(raw.loc[members, name].astype(float).mean()),
                    "corpus_mean": float(raw[name].astype(float).mean()),
                    "smd": float(means[r, k]), "tv": float("nan"),
                    "levels": 1, "gated": True,
                    "passed": bool(abs(float(means[r, k])) <= spec.max_abs_smd),
                })
    for name, block in design.blocks.items():
        overall = block.mean(axis=0)
        means = _role_means(block, ind)
        gated = spec.is_gated(name)
        for r, role in enumerate(roles):
            tv = float(0.5 * np.abs(means[r] - overall).sum())
            rows.append({
                "role": role, "covariate": name, "kind": "categorical",
                "role_mean": float("nan"), "corpus_mean": float("nan"),
                "smd": float("nan"), "tv": tv,
                "levels": int(block.shape[1]), "gated": bool(gated),
                # a covariate with more levels than a role has conversations
                # cannot reach a fixed TV; those are optimized and reported,
                # never thresholded
                "passed": bool(tv <= spec.max_categorical_tv) if gated else True,
            })
    return pd.DataFrame(rows)


def balanced_assignment(features: pd.DataFrame, targets: Mapping[str, int],
                        spec: BalanceSpec, *, seed: int
                        ) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign conversations to roles by seeded rerandomization."""
    ids = [str(v) for v in features["conversation_id"]]
    n = len(ids)
    roles = list(targets)
    sizes = [int(targets[r]) for r in roles]
    if sum(sizes) != n:
        raise ValueError(
            f"role targets sum to {sum(sizes)} but there are {n} conversations")

    design = build_design(features, spec)
    rng = np.random.default_rng(int(seed))
    best_order: np.ndarray | None = None
    best: tuple[float, float, float, float] = (np.inf, np.inf, np.inf, np.inf)
    scores = np.empty(int(spec.draws), dtype=float)

    for d in range(int(spec.draws)):
        order = rng.permutation(n)
        score = _score_from_indicator(design, _indicator(order, sizes, n), spec)
        scores[d] = score[0]
        if score[0] < best[0]:
            best, best_order = score, order

    assert best_order is not None
    assignment: dict[str, str] = {}
    start = 0
    for role, size in zip(roles, sizes):
        for j in best_order[start:start + size]:
            assignment[ids[j]] = role
        start += size

    report = {
        "seed": int(seed),
        "draws": int(spec.draws),
        "accepted_score": float(best[0]),
        "accepted_max_abs_smd": float(best[1]),
        "accepted_max_categorical_tv": float(best[2]),
        "accepted_max_ungated_tv": float(best[3]),
        "gated_categorical": list(spec.gated_categorical),
        "ungated_weight": UNGATED_WEIGHT,
        "null_score_median": float(np.median(scores)),
        "null_score_quantiles": {q: float(np.quantile(scores, q))
                                 for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)},
        "beats_random_median": bool(best[0] < float(np.median(scores))),
        "max_abs_smd_threshold": spec.max_abs_smd,
        "max_categorical_tv_threshold": spec.max_categorical_tv,
        "continuous_covariates": design.continuous_names,
        "categorical_covariates": sorted(design.blocks),
        "targets": {r: int(t) for r, t in targets.items()},
    }
    return assignment, report


def assignment_hash(assignment: Mapping[str, str]) -> str:
    """Stable hash of the conversation -> role map."""
    return sha256_strings(f"{k}={assignment[k]}" for k in sorted(assignment))
