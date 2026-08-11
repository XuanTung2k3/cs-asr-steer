"""One registry for every random seed the pipeline uses.

Modules may not invent seeds. `seed_for` raises on an unregistered purpose, so
a new source of randomness cannot enter the pipeline without appearing in the
seed map that gets frozen into the spec and printed in the paper appendix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SEED_MAP_VERSION = "lss_seed_map_v1"

#: Every purpose that may draw randomness, and what it controls.
REQUIRED_PURPOSES: tuple[str, ...] = (
    "role_assignment",       # which conversation lands in which data role
    "role_draws",            # number of rerandomization draws (not a seed, a count)
    "pilot_sample",          # utterances used for the compute pilot
    "synthetic_dev",         # synthetic splices used to CHOOSE an aligner config
    "synthetic_gate",        # synthetic splices used to JUDGE it (must be disjoint)
    "audit_sample",          # which units enter the human audit pack
    "audit_blinding",        # item id salt and presentation order
    "audit_perturb",         # which audit items are deliberately mis-positioned decoys
    "jitter",                # boundary-jitter replicates
    "bootstrap",             # conversation block bootstrap
    "direction_subsample",   # direction-stability subsamples
    "permutation_null",      # within-pair EN/ZH label permutation controls
    "localizer_init",        # neural localizer initialisation (>=3 seeds)
    "minibatch_order",       # training data order
    "lora",                  # LoRA baseline training
    "selector",              # utility selector training
    "calibration_folds",     # grouped calibration folds
)


@dataclass(frozen=True)
class SeedMap:
    values: Mapping[str, Any]

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SeedMap":
        raw = dict(cfg.get("seeds") or {})
        missing = [p for p in REQUIRED_PURPOSES if p not in raw]
        if missing:
            raise KeyError(
                f"seed map is incomplete; missing purposes: {missing}. "
                "Every source of randomness must be registered in `seeds:`.")
        return cls(values=raw)

    def to_dict(self) -> dict[str, Any]:
        return {"version": SEED_MAP_VERSION,
                "values": {k: self.values[k] for k in sorted(self.values)}}


def _lookup(cfg: dict, purpose: str) -> Any:
    if purpose not in REQUIRED_PURPOSES:
        raise KeyError(
            f"unregistered seed purpose {purpose!r}; add it to "
            "csasr.lss.seeds.REQUIRED_PURPOSES and to the config's `seeds:` block")
    seeds = cfg.get("seeds") or {}
    if purpose not in seeds:
        raise KeyError(f"config has no seed for purpose {purpose!r}")
    return seeds[purpose]


def seed_for(cfg: dict, purpose: str) -> int:
    """The single seed registered for ``purpose``."""
    value = _lookup(cfg, purpose)
    if isinstance(value, (list, tuple)):
        raise TypeError(f"purpose {purpose!r} holds {len(value)} seeds; use seeds_for()")
    return int(value)


def seeds_for(cfg: dict, purpose: str) -> tuple[int, ...]:
    """Every seed registered for ``purpose`` (a scalar becomes a 1-tuple)."""
    value = _lookup(cfg, purpose)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)
