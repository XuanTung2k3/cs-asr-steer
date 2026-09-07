"""Canonical result artifact schema (``result_v1``).

A canonical result record carries **result metrics only** plus a pointer to its
reproducibility provenance. It contains no scientific verdict (GO / NO-GO lives
with the gate machinery, ``csasr.lss.gates``) and no interpretation.

Serialization is deterministic, standard JSON: ``sort_keys``, ``allow_nan=False``
(so NaN/Inf never leak); unavailable values are intentional ``null``.

Gate coverage is **reserved but deferred** (spec §4 / MC §6): its eligible-
position denominator depends on gate semantics that are still an OPEN DECISION
and the factorized gate is unimplemented. The schema keeps a ``gate_coverage``
block whose ``value``/``numerator``/``denominator``/``denominator_id`` are all
``null`` in DG-01, and ``populate_gate_coverage`` refuses to fill it.

Spec: ``docs/current/DG01_METRICS_RESULTS_SPEC.md`` §7.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

RESULT_SCHEMA_VERSION = "result_v1"


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with ``None`` for standard JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def reserved_gate_coverage() -> dict[str, Any]:
    """The deferred gate-coverage block: every field null in DG-01 (spec §4)."""
    return {
        "value": None,
        "numerator": None,
        "denominator": None,
        "denominator_id": None,
        "deferred": True,
        "note": ("gate-coverage denominator is an OPEN DECISION resolved with the "
                 "factorized-gate implementation ticket (MC §6); not defined in DG-01"),
    }


def populate_gate_coverage(*_args: Any, **_kwargs: Any) -> "dict[str, Any]":
    """Guard: populating canonical gate coverage is forbidden in DG-01."""
    raise RuntimeError(
        "canonical gate coverage is deferred (spec §4 / MC §6): its denominator "
        "is not defined until the factorized gate is implemented; do not populate it")


@dataclass
class MethodConfig:
    """Intervention configuration. Every field is optional and null when N/A."""
    layer: int | None = None
    direction_artifact_id: str | None = None
    direction_type: str | None = None
    gate_type: str | None = None
    steering_strength: float | None = None
    gate_params: dict[str, Any] | None = None


@dataclass
class CanonicalResult:
    """One ``result_v1`` record: identity, method config, metrics, provenance."""
    run_id: str
    system_name: str
    schema_version: str = RESULT_SCHEMA_VERSION
    # identity
    model_id: str | None = None
    model_revision: str | None = None
    dataset: str | None = None
    split: str | None = None
    data_role: str | None = None
    seed: int | None = None
    decode_regime: str | None = None
    beam: int | None = None
    baseline_ref_run_id: str | None = None
    # method configuration (optional)
    method: MethodConfig = field(default_factory=MethodConfig)
    # measured outcomes (canonical metrics + gains + retention + reserved coverage)
    metrics: dict[str, Any] = field(default_factory=dict)
    # descriptive, non-harm transcript-change magnitudes (never scientific claims)
    descriptive: dict[str, Any] = field(default_factory=dict)
    # reproducibility, kept logically separate from metrics
    provenance: dict[str, Any] = field(default_factory=dict)
    # set on records produced from a legacy artifact by the adapter
    legacy_source: str | None = None

    def __post_init__(self) -> None:
        self.metrics.setdefault("gate_coverage", reserved_gate_coverage())

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False,
                          ensure_ascii=False, indent=indent)


def from_dict(payload: dict[str, Any]) -> CanonicalResult:
    """Rebuild a :class:`CanonicalResult` from a plain dict (round-trip)."""
    data = dict(payload)
    method = data.pop("method", None) or {}
    result = CanonicalResult(
        run_id=data.pop("run_id"),
        system_name=data.pop("system_name"),
        method=MethodConfig(**method),
        **{k: data[k] for k in data if k in _RESULT_FIELDS},
    )
    return result


def from_json(text: str) -> CanonicalResult:
    return from_dict(json.loads(text))


def validate(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` if a record is not a well-formed ``result_v1``."""
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    for key in ("run_id", "system_name", "metrics"):
        if key not in payload or payload[key] in (None, ""):
            raise ValueError(f"missing required field: {key}")
    gc = payload["metrics"].get("gate_coverage")
    if gc is None:
        raise ValueError("metrics.gate_coverage block must be present (reserved)")
    if gc.get("value") is not None:
        raise ValueError("gate_coverage.value must be null in DG-01 (deferred)")


#: field names accepted by :func:`from_dict` beyond the two required positionals
_RESULT_FIELDS = {
    "schema_version", "model_id", "model_revision", "dataset", "split",
    "data_role", "seed", "decode_regime", "beam", "baseline_ref_run_id",
    "metrics", "descriptive", "provenance", "legacy_source",
}
