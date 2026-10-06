"""P2-DIR common direction interface (frozen spec sections 3-5): exactly D0 OLD, D1 UNIQUE, D2 READOUT.

Each provider is called with a reference-free :class:`DirectionContext` and returns a
:class:`DirectionResult` ``{id, direction, status, reason, provenance, counters}``. ``direction`` is a
detached float32 D-vector or ``None``; a failed provider never falls through to another provider.
Providers do not own dose or energy: the caller applies either the frozen deployment dose or the
diagnostic chord solver.

* D0 delegates :func:`csasr.inference_cf.core_p1.direction` unchanged (epsilon-normalized, not
  renormalized to exactly unit norm).
* D1 returns a copy of the sealed leave-one-dialogue-out fold vector (``unique.py``); an invalid
  fold yields ``status='invalid'`` (zero edit), never D0/D2/A5/A6 or another rank.
* D2 runs the reference-free readout scratch forward (``readout.py``) from the PRE-step B cache.

No reference transcript, reference token set, competitor, alignment, future token or evaluator
quantity is part of the context or of any provider (enforced by tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from . import readout as _readout
from .core_p1 import direction as _old_direction
from .unique import VERSION as UNIQUE_VERSION, array_hash

IDS = ("D0", "D1", "D2")


@dataclass
class DirectionContext:
    """Reference-free inputs available at the current B query."""
    h_b: torch.Tensor | None = None                 # unedited B site (float32 of the bf16 state)
    h_e: torch.Tensor | None = None                 # unedited E site (D0 only)
    b_cache_pre_step: Any = None                    # persistent B cache BEFORE the current token(s)
    encoded: Any = None
    new_tokens: Sequence[int] = field(default_factory=list)
    start: int = 0                                  # cache length before the current token(s)
    step: int = 0                                   # content step t


@dataclass
class DirectionResult:
    id: str
    direction: torch.Tensor | None
    status: str
    reason: str | None
    provenance: dict
    counters: dict
    extras: dict = field(default_factory=dict)

    def record(self) -> dict:
        d = self.direction
        return {"id": self.id, "status": self.status, "reason": self.reason, "provenance": self.provenance,
                "counters": self.counters,
                "norm": None if d is None else float(torch.linalg.vector_norm(d.double())),
                "sha256": None if d is None else array_hash(d.detach().cpu().numpy().astype(np.float32))}


class OldDirection:
    """D0: norm(hE - hB) exactly as core_p1.direction (frozen epsilon / tiny / nonfinite fallback)."""
    id = "D0"

    def __call__(self, ctx: DirectionContext) -> DirectionResult:
        r = _old_direction(ctx.h_e, ctx.h_b)
        ok = r["d"] is not None
        return DirectionResult("D0", r["d"], "ok" if ok else "invalid", None if ok else r["status"],
                               {"function": "csasr.inference_cf.core_p1.direction", "delta_norm": r["norm"],
                                "core_status": r["status"]}, {"autograd_calls": 0})


class UniqueDirection:
    """D1: sealed fold vector for the evaluated dialogue (read-only)."""
    id = "D1"

    def __init__(self, vector: np.ndarray | None, fold: dict):
        self._vector = None if vector is None else np.array(vector, dtype=np.float32, copy=True)
        if self._vector is not None:
            self._vector.setflags(write=False)
        self.fold = dict(fold)
        if self._vector is not None and array_hash(self._vector) != self.fold.get("vector_sha256"):
            raise ValueError("sealed D1 vector hash mismatch")
        if (self._vector is None) != (self.fold.get("status") != "ok"):
            raise ValueError("D1 fold status inconsistent with vector presence")

    def __call__(self, ctx: DirectionContext) -> DirectionResult:
        prov = {"definition": UNIQUE_VERSION, "fold_dialogue": self.fold.get("dialogue"),
                "fold_status": self.fold.get("status"), "vector_sha256": self.fold.get("vector_sha256")}
        if self._vector is None:
            return DirectionResult("D1", None, "invalid", f"fold_invalid:{self.fold.get('reason')}", prov,
                                   {"autograd_calls": 0})
        return DirectionResult("D1", torch.from_numpy(self._vector.copy()), "ok", None, prov, {"autograd_calls": 0})


class ReadoutDirection:
    """D2: reference-free readout tangent from the pre-step B cache."""
    id = "D2"

    def __init__(self, bundle, *, layer: int, suppress: Sequence[int], begin: Sequence[int], partition: dict):
        self.bundle, self.layer = bundle, int(layer)
        self.suppress, self.begin, self.partition = list(suppress), list(begin), partition

    def __call__(self, ctx: DirectionContext) -> DirectionResult:
        r = _readout.readout_direction(self.bundle, layer=self.layer, cache=ctx.b_cache_pre_step,
                                       encoded=ctx.encoded, new_tokens=list(ctx.new_tokens), start=ctx.start,
                                       step=ctx.step, suppress=self.suppress, begin=self.begin,
                                       partition=self.partition)
        prov = {"version": _readout.VERSION, "partition_hash": self.partition.get("hash"), "query": r["query"],
                **{k: r.get(k) for k in ("J", "log_PE", "log_PM", "g_norm", "h_norm", "tangent_norm",
                                         "g_radial", "unit_error", "radial_dot")}}
        extras = {k: r.get(k) for k in ("logits", "site", "gradient")}
        extras["runtime_sec"] = r["runtime_sec"]
        return DirectionResult("D2", r["direction"], r["status"], r["reason"], prov, r["counters"], extras)
