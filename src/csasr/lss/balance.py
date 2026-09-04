"""Balanced assignment of conversations to data roles.

Sorted-identifier slicing ("conversations 1-70 are D-construct") is
deterministic but not balanced: corpus identifiers usually preserve collection
order, so slicing can hand one role systematically longer recordings, a
different device mix, or a different code-switching rate. Any difference between
roles then shows up later as a method effect.

The assignment here is seeded **rerandomization** (Morgan & Rubin): draw many
size-respecting partitions, reject candidates outside every predeclared gated
balance threshold, and rank only the feasible candidates by the predeclared
objective. Proposal-specific feasibility and score distributions are reported,
so the accepted draw can be read against both unconstrained randomization and a
stratified proposal without mixing the two reference distributions.
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
    ungated_weight: float = 0.3

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
            ungated_weight=float(block.get("ungated_weight", cls.ungated_weight)),
        )

    def is_gated(self, covariate: str) -> bool:
        return covariate in self.gated_categorical


@dataclass(frozen=True)
class ProposalSpec:
    """One predeclared source of random candidate partitions."""

    mechanism: str
    draws: int
    stratify_by: tuple[str, ...] = ()

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], *, default_draws: int) -> "ProposalSpec":
        block = dict(cfg or {})
        mechanism = str(block.get("mechanism", "")).strip()
        if mechanism not in {"uniform_permutation", "stratified_deal"}:
            raise ValueError(
                "balance proposal mechanism must be `uniform_permutation` or "
                f"`stratified_deal`, found {mechanism!r}")
        draws = int(block.get("draws", default_draws))
        if draws < 1:
            raise ValueError("balance proposal draws must be positive")
        stratify_by = tuple(str(v) for v in (block.get("stratify_by") or ()))
        if mechanism == "stratified_deal" and not stratify_by:
            raise ValueError("stratified_deal requires nonempty `stratify_by`")
        if mechanism == "uniform_permutation" and stratify_by:
            raise ValueError("uniform_permutation cannot declare `stratify_by`")
        return cls(mechanism=mechanism, draws=draws, stratify_by=stratify_by)


class NoFeasibleAssignment(RuntimeError):
    """The declared proposal budget contained no gate-passing partition."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(
            "no feasible balance draw: 0 of "
            f"{self.report.get('draws', 0)} candidates satisfied every gated criterion")


#: How the candidate-draw stream is derived from the recorded seed.
#:
#: The stream is part of the allocation rule, not an implementation detail: two
#: derivations produce different candidate sequences from the same seed, so an
#: identity that omitted it would call two different partitions the same. It is
#: recorded in the role-partition fingerprint and in the v2 spec freeze, and it
#: lives here, beside `balanced_assignment`, so the record cannot drift away
#: from the expression it describes.
RNG_DERIVATION: dict[str, str] = {
    "current": ("numpy.random.default_rng(numpy.random.SeedSequence("
                "[seed, proposal_index, 0xC5A5]))"),
    "previous": "numpy.random.default_rng(seed)",
    "implementation": "csasr.lss.balance.balanced_assignment",
    "changed_in": "dialogue-rerandomization-v2",
    "reason": ("independent deterministic streams per proposal mechanism, needed "
               "because uniform_permutation and stratified_deal now run within "
               "one allocation; a single shared stream would make the candidates "
               "drawn by one mechanism depend on whether another was declared"),
    "comparability": ("the v2 allocation is NOT a counterfactual reproduction of "
                      "the v1 allocation under a different threshold. It is a "
                      "fresh allocation under a corrected acceptance rule and a "
                      "rebuilt proposal mechanism, drawn from a different stream. "
                      "v1 and v2 allocations are not comparable draw-for-draw"),
}


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
    objective = max(max_smd, max_tv_gated) + spec.ungated_weight * max_tv_ungated
    return objective, max_smd, max_tv_gated, max_tv_ungated


def _indicator(order: np.ndarray, sizes: Sequence[int], n: int) -> np.ndarray:
    ind = np.zeros((len(sizes), n), dtype=float)
    start = 0
    for r, size in enumerate(sizes):
        ind[r, order[start:start + size]] = 1.0
        start += size
    return ind


def _stratified_indicator(features: pd.DataFrame, sizes: Sequence[int], *,
                          stratify_by: Sequence[str], rng: np.random.Generator
                          ) -> np.ndarray:
    """Deal each observed stratum across roles near its proportional quota.

    The integer stratum-by-role table has the exact observed stratum margins
    and exact role-size margins.  Floors of the proportional quotas are filled
    first; remaining units go to the largest current quota deficit, with seeded
    random tie-breaking.  Members are then shuffled within every stratum.  Thus
    the proposal is random while keeping the gated categorical composition near
    the attainable proportional allocation.
    """
    missing = [c for c in stratify_by if c not in features.columns]
    if missing:
        raise ValueError(f"stratified proposal columns are missing: {missing}")
    n = len(features)
    labels = features[list(stratify_by)].astype(str).agg("\x1f".join, axis=1)
    levels = sorted(labels.unique())
    members = [np.flatnonzero(labels.to_numpy() == level) for level in levels]
    counts = np.asarray([len(v) for v in members], dtype=int)
    role_sizes = np.asarray(list(sizes), dtype=int)
    expected = np.outer(counts, role_sizes) / float(n)
    allocation = np.floor(expected).astype(int)
    row_left = counts - allocation.sum(axis=1)
    col_left = role_sizes - allocation.sum(axis=0)

    # Process rows in a random order. Within a row, prefer the role with the
    # largest deficit from its proportional quota; random noise only resolves
    # exact ties and therefore cannot override the proportional dealing rule.
    for s in rng.permutation(len(levels)):
        while row_left[s] > 0:
            available = np.flatnonzero(col_left > 0)
            if not len(available):  # pragma: no cover - margin arithmetic guard
                raise RuntimeError("stratified deal exhausted role capacity")
            deficit = expected[s, available] - allocation[s, available]
            best = float(deficit.max())
            tied = available[np.isclose(deficit, best, rtol=0.0, atol=1e-12)]
            role = int(rng.choice(tied))
            allocation[s, role] += 1
            row_left[s] -= 1
            col_left[role] -= 1

    if row_left.any() or col_left.any():  # pragma: no cover - invariant guard
        raise RuntimeError("stratified deal did not satisfy both integer margins")

    indicator = np.zeros((len(sizes), n), dtype=float)
    for s, indices in enumerate(members):
        shuffled = rng.permutation(indices)
        start = 0
        for role, take in enumerate(allocation[s]):
            indicator[role, shuffled[start:start + take]] = 1.0
            start += int(take)
    return indicator


def _assignment_from_indicator(ids: Sequence[str], roles: Sequence[str],
                               indicator: np.ndarray) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for role_index, role in enumerate(roles):
        for item_index in np.flatnonzero(indicator[role_index] > 0):
            assignment[str(ids[item_index])] = str(role)
    return assignment


def _is_feasible(score: tuple[float, float, float, float],
                 spec: BalanceSpec) -> bool:
    return bool(score[1] <= spec.max_abs_smd
                and score[2] <= spec.max_categorical_tv)


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
                    "metric": "abs_smd", "observed": abs(float(means[r, k])),
                    "threshold": float(spec.max_abs_smd),
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
                "metric": "tv", "observed": tv,
                "threshold": float(spec.max_categorical_tv) if gated else float("nan"),
                # a covariate with more levels than a role has conversations
                # cannot reach a fixed TV; those are optimized and reported,
                # never thresholded
                "passed": bool(tv <= spec.max_categorical_tv) if gated else True,
            })
    return pd.DataFrame(rows, columns=(
        "role", "covariate", "kind", "role_mean", "corpus_mean", "smd", "tv",
        "levels", "gated", "metric", "observed", "threshold", "passed"))


def balanced_assignment(features: pd.DataFrame, targets: Mapping[str, int],
                        spec: BalanceSpec, *, seed: int,
                        proposals: Sequence[ProposalSpec] | None = None
                        ) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign units by seeded rerandomization over gate-feasible draws only.

    The proposal mechanisms generate candidate partitions.  Gated SMD and TV
    criteria are acceptance constraints; the unchanged objective ranks only
    candidates satisfying both.  Infeasible candidates can never be returned.

    Two things changed here relative to the single-proposal version, not one:
    the acceptance rule above, *and* the proposal machinery -- a stratified deal
    joined the uniform permutation, and each mechanism now draws from its own
    stream (see `RNG_DERIVATION`).  An allocation produced here is therefore a
    fresh allocation, never a draw-for-draw replay of an earlier one under a
    different criterion.
    """
    ids = [str(v) for v in features["conversation_id"]]
    n = len(ids)
    roles = list(targets)
    sizes = [int(targets[r]) for r in roles]
    if sum(sizes) != n:
        raise ValueError(
            f"role targets sum to {sum(sizes)} but there are {n} conversations")

    design = build_design(features, spec)
    declared = tuple(proposals or (
        ProposalSpec("uniform_permutation", int(spec.draws)),))
    if not declared:
        raise ValueError("at least one balance proposal must be declared")
    best: tuple[float, float, float, float] = (np.inf, np.inf, np.inf, np.inf)
    best_indicator: np.ndarray | None = None
    accepted_mechanism: str | None = None
    all_scores: list[float] = []
    proposal_reports: list[dict[str, Any]] = []
    total_feasible = 0

    for proposal_index, proposal in enumerate(declared):
        # Independent, stable streams keep adding a later proposal from changing
        # the candidates already drawn by an earlier one.  `RNG_DERIVATION`
        # records this expression; change one and you must change the other.
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(proposal_index), 0xC5A5]))
        scores = np.empty(int(proposal.draws), dtype=float)
        feasible = 0
        proposal_best = np.inf
        for draw in range(int(proposal.draws)):
            if proposal.mechanism == "uniform_permutation":
                indicator = _indicator(rng.permutation(n), sizes, n)
            elif proposal.mechanism == "stratified_deal":
                indicator = _stratified_indicator(
                    features, sizes, stratify_by=proposal.stratify_by, rng=rng)
            else:  # ProposalSpec validates this; retain an execution guard.
                raise ValueError(f"unknown balance proposal {proposal.mechanism!r}")
            score = _score_from_indicator(design, indicator, spec)
            scores[draw] = score[0]
            if _is_feasible(score, spec):
                feasible += 1
                total_feasible += 1
                proposal_best = min(proposal_best, score[0])
                if score[0] < best[0]:
                    best, best_indicator = score, indicator.copy()
                    accepted_mechanism = proposal.mechanism
        all_scores.extend(float(v) for v in scores)
        proposal_reports.append({
            "mechanism": proposal.mechanism,
            "stratify_by": list(proposal.stratify_by),
            "seed_stream": [int(seed), int(proposal_index), 0xC5A5],
            "draws": int(proposal.draws),
            "feasible_draws": int(feasible),
            "feasibility_rate": float(feasible / proposal.draws),
            "best_feasible_score": None if not feasible else float(proposal_best),
            "all_score_median": float(np.median(scores)),
            "all_score_quantiles": {q: float(np.quantile(scores, q))
                                    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95)},
        })

    scores_array = np.asarray(all_scores, dtype=float)
    uniform_report = next((r for r in proposal_reports
                           if r["mechanism"] == "uniform_permutation"), None)
    base_report = {
        "seed": int(seed),
        "draws": int(sum(p.draws for p in declared)),
        "feasible_draws": int(total_feasible),
        "feasibility_rate": float(total_feasible / len(scores_array)),
        "proposal_reports": proposal_reports,
        "accepted_proposal_mechanism": accepted_mechanism,
        "gated_categorical": list(spec.gated_categorical),
        "ungated_weight": float(spec.ungated_weight),
        "scoring_formula": "max(max_SMD, max_gated_TV) + ungated_weight*ungated_TV",
        # Only the uniform permutation proposal is the randomization null.
        # Mixing it with the deliberately stratified proposal would give a
        # number with no interpretable reference distribution.
        "null_score_median": None if uniform_report is None
        else uniform_report["all_score_median"],
        "null_score_quantiles": {} if uniform_report is None
        else uniform_report["all_score_quantiles"],
        "combined_candidate_score_median": float(np.median(scores_array)),
        "max_abs_smd_threshold": spec.max_abs_smd,
        "max_categorical_tv_threshold": spec.max_categorical_tv,
        "continuous_covariates": design.continuous_names,
        "categorical_covariates": sorted(design.blocks),
        "targets": {r: int(t) for r, t in targets.items()},
    }
    if best_indicator is None:
        raise NoFeasibleAssignment(base_report)

    assignment = _assignment_from_indicator(ids, roles, best_indicator)

    report = {
        **base_report,
        "accepted_score": float(best[0]),
        "accepted_max_abs_smd": float(best[1]),
        "accepted_max_categorical_tv": float(best[2]),
        "accepted_max_ungated_tv": float(best[3]),
        "beats_proposal_median": bool(best[0] < float(np.median(scores_array))),
        "beats_random_median": None if uniform_report is None else bool(
            best[0] < float(uniform_report["all_score_median"])),
    }
    return assignment, report


def assignment_hash(assignment: Mapping[str, str]) -> str:
    """Stable hash of the conversation -> role map."""
    return sha256_strings(f"{k}={assignment[k]}" for k in sorted(assignment))
