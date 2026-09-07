"""Canonical ``result_v1`` emission with real repository provenance (DG-01 §8).

This is the smallest *current* result-emission path: given a scored text triple
``(references, baseline_hyps, method_hyps)`` and a resolved config, it produces
canonical ``result_v1`` records whose metrics come **only** from the reuse facade
(:mod:`csasr.evaluation.canonical` / :mod:`csasr.evaluation.retention`) and whose
provenance comes **only** from the existing infrastructure
(:func:`csasr.utils.provenance.stage_provenance`, :func:`csasr.utils.logging.git_state`,
:func:`csasr.lss.manifest.run_id`, :mod:`csasr.utils.hashing`).

It changes no decoding, steering, training, or data behaviour: it consumes
already-produced hypotheses and only scores + serialises them. Provenance fields
that no artifact supplies are left ``null`` — never fabricated (spec §8).

Spec: ``docs/current/DG01_METRICS_RESULTS_SPEC.md`` §7-§8.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import canonical, retention
from .result_schema import (
    RESULT_SCHEMA_VERSION,
    CanonicalResult,
    MethodConfig,
    reserved_gate_coverage,
)

STAGE = "dg01_metrics"


def build_provenance(cfg: Mapping[str, Any] | None = None, *,
                     repo_root: str | None = None) -> dict[str, Any]:
    """Assemble the DG-01 manifest from existing provenance primitives.

    Reuses ``stage_provenance`` (model/dataset/config/source hashes, seed,
    normalization version, timestamp) and ``git_state`` (HEAD commit) and stamps
    the schema versions. Nothing is invented: a field the infrastructure returns
    as ``None`` stays ``None``.
    """
    from ..utils.config import REPO_ROOT
    from ..utils.logging import git_state
    from ..utils.provenance import stage_provenance

    cfg = dict(cfg or {})
    root = repo_root or str(REPO_ROOT)
    prov = stage_provenance(cfg, STAGE)
    git = git_state(root)
    prov.update({
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "config_hash": prov.get("resolved_config_hash"),
        "dataset_fingerprint": prov.get("dataset_manifest_hash"),
        "metrics_schema_version": canonical.SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        # populated only when a real artifact exists; never fabricated
        "direction_artifact_hash": None,
        "calibrator_artifact_hash": None,
    })
    return prov


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(cfg.get("model") or {})
    exp = dict(cfg.get("experiment") or {})
    data = dict(cfg.get("data") or {})
    return {
        "model_id": model.get("id") or model.get("hub_id"),
        "model_revision": model.get("revision"),
        "dataset": data.get("name") or data.get("dataset"),
        "split": data.get("split"),
        "data_role": data.get("role") or data.get("data_role"),
        "seed": exp.get("seed"),
        "decode_regime": exp.get("decode_regime") or exp.get("decoding"),
        "beam": exp.get("beam"),
    }


def emit_result(references: Sequence[str], hypotheses: Sequence[str], *,
                run_id: str, system_name: str,
                cfg: Mapping[str, Any] | None = None,
                method: MethodConfig | None = None,
                baseline_metrics: Mapping[str, Any] | None = None,
                baseline_ref_run_id: str | None = None,
                baseline_hyps: Sequence[str] | None = None,
                provenance: Mapping[str, Any] | None = None) -> CanonicalResult:
    """Score one hypothesis set into a canonical ``result_v1`` record.

    Metrics (MER/PIER/EN-WER/ZH-CER) come from ``canonical.corpus_metrics``. When
    ``baseline_metrics`` is given, ``*_gain = baseline - method`` is attached; when
    ``baseline_hyps`` is also given, POI transitions, candidate-free retention, and
    the reserved gate-coverage block are attached too. No metric is rescored.
    """
    cfg = dict(cfg or {})
    ident = _identity(cfg)
    m = canonical.corpus_metrics(references, hypotheses)
    metrics: dict[str, Any] = dict(m)
    metrics["gate_coverage"] = reserved_gate_coverage()

    if baseline_metrics is not None:
        metrics.update(canonical.error_metric_gains(baseline_metrics, m))
    if baseline_hyps is not None:
        metrics["transitions"] = canonical.correction_corruption(
            references, baseline_hyps, hypotheses)
        metrics["retention"] = retention.retention_report(
            references, baseline_hyps, hypotheses)

    return CanonicalResult(
        run_id=run_id,
        system_name=system_name,
        method=method or MethodConfig(),
        metrics=metrics,
        baseline_ref_run_id=baseline_ref_run_id,
        provenance=dict(provenance) if provenance is not None else build_provenance(cfg),
        **ident,
    )


def emit_baseline_and_method(references: Sequence[str],
                             baseline_hyps: Sequence[str],
                             method_hyps: Sequence[str], *,
                             cfg: Mapping[str, Any] | None = None,
                             baseline_run_id: str | None = None,
                             method_run_id: str | None = None,
                             method: MethodConfig | None = None,
                             ) -> tuple[CanonicalResult, CanonicalResult]:
    """Emit a paired ``(baseline, method)`` pair of ``result_v1`` records.

    The two records share one provenance manifest (same code/config/git state)
    and the method record references the baseline run id and carries canonical
    ``*_gain`` deltas plus the three retention populations.
    """
    from ..lss.manifest import run_id as make_run_id

    cfg = dict(cfg or {})
    prov = build_provenance(cfg)
    base_run = baseline_run_id or make_run_id(f"{STAGE}/baseline")
    meth_run = method_run_id or make_run_id(f"{STAGE}/method")

    baseline = emit_result(references, baseline_hyps, run_id=base_run,
                           system_name="baseline", cfg=cfg, provenance=prov)
    base_metrics = {k: baseline.metrics.get(k) for k in canonical.ERROR_METRICS}
    method_res = emit_result(references, method_hyps, run_id=meth_run,
                             system_name="method", cfg=cfg, method=method,
                             baseline_metrics=base_metrics,
                             baseline_ref_run_id=base_run,
                             baseline_hyps=baseline_hyps, provenance=prov)
    return baseline, method_res
