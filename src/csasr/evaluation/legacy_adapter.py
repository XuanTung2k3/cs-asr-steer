"""Read-only adapter: legacy result summaries -> canonical ``result_v1``.

The adapter parses representative historical Job A / Job B summaries **in
memory** and never edits them. It performs only justified transformations:
field-name normalization, explicit delta re-signing, identical-metric mapping,
and attaching legacy provenance. It never fabricates unavailable fields, never
turns a legacy transcript-edit count into canonical outside harm, and never
decomposes the blended legacy ``zh_retention`` into the three MC §8 populations.

Every record it emits is marked ``legacy_source`` + ``production_artifact=False``
and carries ``derivation = "legacy-derived"``.

Spec: ``docs/current/DG01_METRICS_RESULTS_SPEC.md`` §9.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import canonical
from .result_schema import CanonicalResult, MethodConfig, reserved_gate_coverage

DERIVATION = "legacy-derived"


def resign_delta(delta_method_minus_baseline: float | None) -> float | None:
    """Translate a legacy ``method - baseline`` delta into a canonical gain.

    Legacy ``steer_sweep.round1_metrics`` emits ``delta_PIER = method - baseline``
    (negative is better). The canonical gain is ``baseline - method`` (positive is
    better), i.e. the negation.
    """
    v = canonical._finite_or_none(delta_method_minus_baseline)
    return None if v is None else -v


def _legacy_provenance(source: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prov = {
        "derivation": DERIVATION,
        "legacy_source": source,
        "production_artifact": False,
        # nothing below is fabricated: unknown historical provenance stays null
        "git_commit": None,
        "config_hash": None,
        "dataset_fingerprint": None,
        "direction_artifact_hash": None,
        "calibrator_artifact_hash": None,
        "timestamp": None,
        "metrics_schema_version": canonical.SCHEMA_VERSION,
    }
    if extra:
        prov.update(extra)
    return prov


def _system_metrics(system: Mapping[str, Any],
                    baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Map one legacy system block onto canonical metric names + gains."""
    mer = canonical._finite_or_none(system.get("MER"))
    pier = canonical._finite_or_none(system.get("PIER"))
    en_wer = canonical._finite_or_none(system.get("en_wer"))
    zh_cer = canonical._finite_or_none(system.get("zh_cer"))

    # correction/corruption: the legacy correction_harm block already uses the
    # correctness-transition definition; map it identically (with denominators).
    ch = system.get("correction_harm") or {}
    corrections = ch.get("num_corrected", system.get("corrections"))
    corruptions = ch.get("num_corrupted", system.get("corruptions"))
    base_wrong = ch.get("num_baseline_incorrect")
    base_correct = ch.get("num_baseline_correct")

    metrics: dict[str, Any] = {
        "mer": mer, "pier": pier, "en_wer": en_wer, "zh_cer": zh_cer,
        "mer_gain": canonical.gain(baseline.get("MER"), mer),
        "pier_gain": canonical.gain(baseline.get("PIER"), pier),
        "en_wer_gain": canonical.gain(baseline.get("en_wer"), en_wer),
        "zh_cer_gain": canonical.gain(baseline.get("zh_cer"), zh_cer),
        "corrections": None if corrections is None else int(corrections),
        "corruptions": None if corruptions is None else int(corruptions),
        "num_baseline_incorrect_poi": None if base_wrong is None else int(base_wrong),
        "num_baseline_correct_poi": None if base_correct is None else int(base_correct),
        "correction_rate": canonical._finite_or_none(ch.get("correction_rate")),
        "corruption_rate": canonical._finite_or_none(ch.get("corruption_rate")),
        # canonical outside harm is NOT derivable from legacy transcript edits
        "candidate": {"outside_harm": None},
        # blended legacy retention cannot be split into MC §8 populations
        "retention": {"matrix_zh": None, "embedded_en": None, "monolingual": None},
        "gate_coverage": reserved_gate_coverage(),
    }
    return metrics


def _legacy_descriptive(system: Mapping[str, Any]) -> dict[str, Any]:
    """Historical transcript-difference / blended fields, clearly namespaced."""
    desc: dict[str, Any] = {}
    if "outside_region_edits" in system:
        desc["legacy_outside_region_edits"] = system["outside_region_edits"]
    if "outside_edits" in system:
        desc["legacy_outside_edits"] = system["outside_edits"]
    ch = system.get("correction_harm") or {}
    if "outside_edit_rate" in ch:
        desc["legacy_outside_edit_rate"] = canonical._finite_or_none(ch["outside_edit_rate"])
    if "zh_retention" in system:
        # kept only with its historical (blended matrix+monolingual) semantics
        desc["legacy_zh_retention_blended"] = canonical._finite_or_none(system["zh_retention"])
    return desc


def adapt_system(system_name: str, system: Mapping[str, Any],
                 baseline: Mapping[str, Any], *, source: str,
                 run_id: str | None = None,
                 model_id: str | None = None) -> CanonicalResult:
    """Convert one legacy system block (+ its baseline) into a canonical record."""
    return CanonicalResult(
        run_id=run_id or f"legacy::{source}::{system_name}",
        system_name=system_name,
        model_id=model_id,
        method=MethodConfig(),
        metrics=_system_metrics(system, baseline),
        descriptive=_legacy_descriptive(system),
        provenance=_legacy_provenance(source),
        legacy_source=f"{source}::{system_name}",
    )


def _pick_baseline(systems: Mapping[str, Any]) -> Mapping[str, Any]:
    for name, block in systems.items():
        if name.split(":", 1)[0].strip().upper() in {"F0", "BASELINE"} or "baseline" in name.lower():
            return block
    # fall back to a top-level "baseline" block or an empty one
    return {}


def adapt_job_a_summary(path: str | Path) -> list[CanonicalResult]:
    """Adapt a Job-A ``summary.json`` (``systems`` map, F0 baseline)."""
    source = str(path)
    payload = json.loads(Path(path).read_text())
    systems = payload.get("systems", {})
    baseline = _pick_baseline(systems)
    return [adapt_system(name, block, baseline, source=source)
            for name, block in systems.items()]


def adapt_job_b_summary(path: str | Path) -> list[CanonicalResult]:
    """Adapt a Job-B ``summary.json`` (top-level ``baseline`` + method systems).

    Committed Job-B summaries carry the flat per-system metrics under
    ``results_table`` (the ``methods`` block nests ``epoch_results`` and has no
    top-level ``MER``); older/synthetic summaries may use ``systems``. Both are
    accepted. The top-level ``baseline`` block is the gain reference; the
    baseline-named row in the table is skipped so it is emitted exactly once.
    """
    source = str(path)
    payload = json.loads(Path(path).read_text())
    systems = payload.get("results_table") or payload.get("systems", {})
    baseline = payload.get("baseline") or _pick_baseline(systems)
    out: list[CanonicalResult] = []
    if baseline:
        out.append(adapt_system("baseline", baseline, baseline, source=source))
    for name, block in systems.items():
        if not (isinstance(block, Mapping) and "MER" in block):
            continue
        first = name.split(":", 1)[0].strip().upper()
        if first in {"F0", "BASELINE"} or "baseline" in name.lower():
            continue  # already emitted from the top-level baseline block
        out.append(adapt_system(name, block, baseline, source=source))
    return out


def adapt_round1_paired_report(record: Mapping[str, Any], *, source: str,
                               system_name: str = "round1") -> CanonicalResult:
    """Adapt a ``round1_metrics.paired_metric_report`` dict, re-signing deltas.

    ``delta_PIER``/``delta_MER`` (method - baseline) become ``pier_gain``/
    ``mer_gain`` (baseline - method) via :func:`resign_delta`.
    """
    metrics = {
        "mer": canonical._finite_or_none(record.get("MER")),
        "pier": canonical._finite_or_none(record.get("PIER")),
        "en_wer": canonical._finite_or_none(record.get("EN_WER")),
        "zh_cer": canonical._finite_or_none(record.get("ZH_CER")),
        "mer_gain": resign_delta(record.get("delta_MER")),
        "pier_gain": resign_delta(record.get("delta_PIER")),
        "corrections": record.get("corrections"),
        "corruptions": record.get("corruptions"),
        "net_corrections": record.get("net_corrections"),
        "candidate": {"outside_harm": None},
        "retention": {"matrix_zh": None, "embedded_en": None, "monolingual": None},
        "gate_coverage": reserved_gate_coverage(),
    }
    descriptive = {
        "legacy_outside_edit_rate": canonical._finite_or_none(record.get("outside_edit_rate")),
        "legacy_mean_edit_distance": canonical._finite_or_none(
            record.get("mean_baseline_to_method_edit_distance")),
    }
    return CanonicalResult(
        run_id=f"legacy::{source}::{system_name}",
        system_name=system_name,
        beam=record.get("beam"),
        decode_regime=record.get("method_protocol"),
        metrics=metrics,
        descriptive=descriptive,
        provenance=_legacy_provenance(source),
        legacy_source=f"{source}::{system_name}",
    )
