"""CPU-only re-evaluation of cached Alignment Gate-A development evidence.

This command never invokes an aligner, allocates/exposes a gate generation,
writes a production stage status, builds consensus spans, freezes spans, or
unlocks L1c.  Its outputs are explicitly diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.align import autoevidence
from ..lss.align import target_objects as target_mod
from ..lss.artifacts import write_json, write_report
from ..utils.config import load_config
from ..utils.hashing import sha256_obj
from .lss_l1b_valid import (
    _expected_unit_universe,
    _full_role_data_sufficiency,
    _validate_gate_configuration,
    sweep_roles,
)


def _seam_diagnostics(root: Path) -> dict[str, Any]:
    """Read old/new per-item tables and state only seam-relative quantities."""
    paths = [root / autoevidence.SYNTHETIC_ITEMS_FILE,
             root / "metrics/l1b_synthetic_boundary_error.parquet"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return {"available": False, "reason": "no cached per-item table"}
    frame = pd.read_parquet(path)
    if "reference_kind" in frame:
        frame = frame[frame["reference_kind"].astype(str) == "audio_splice"]
        if not len(frame):
            return {"available": False, "path": str(path),
                    "reason": "cached table has no audio_splice rows"}
    required = {"family", "predicted_end_sec", "predicted_start_sec",
                "true_zh_end_sec", "true_en_start_sec"}
    if not required <= set(frame.columns):
        return {"available": False, "path": str(path),
                "reason": f"missing columns {sorted(required - set(frame.columns))}"}
    if "purpose" in frame:
        frame = frame[frame["purpose"].astype(str) == "dev"]
    if "scorable" in frame:
        frame = frame[frame["scorable"].fillna(False).astype(bool)]
    frame = frame.copy()
    frame["zh_end_minus_splice_ms"] = (
        frame["predicted_end_sec"] - frame["true_zh_end_sec"]) * 1000.0
    frame["en_start_minus_splice_ms"] = (
        frame["predicted_start_sec"] - frame["true_en_start_sec"]) * 1000.0
    frame["gap_around_splice_ms"] = (
        frame["predicted_start_sec"] - frame["predicted_end_sec"]) * 1000.0
    return {
        "available": True, "path": str(path), "reference_kind": "audio_splice",
        "metric_semantics": "seam_relative_not_lexical_absolute_error",
        "by_family": {
            str(family): {
                "n": int(len(group)),
                "median_zh_end_minus_splice_ms": float(
                    group["zh_end_minus_splice_ms"].median()),
                "median_en_start_minus_splice_ms": float(
                    group["en_start_minus_splice_ms"].median()),
                "median_gap_around_splice_ms": float(
                    group["gap_around_splice_ms"].median()),
            }
            for family, group in frame.groupby("family", sort=True)
        },
    }


def evaluate_cached(cfg: dict, candidates: pd.DataFrame,
                    candidate_manifest: dict | None = None) -> dict[str, Any]:
    """Pure CPU diagnostic payload used by both CLI and regression tests."""
    _validate_gate_configuration(cfg)
    roles = sweep_roles(cfg)
    expected_units = _expected_unit_universe(cfg, roles)
    expected_targets = target_mod.reference_target_universe(expected_units)
    raw_family = autoevidence.family_validity(candidates)
    raw_detail = target_mod.raw_unit_diagnostics(candidates)
    targets, runs, aggregation = target_mod.target_objects(candidates)
    validity = target_mod.target_family_validity(targets, expected_targets)
    gcfg = cfg.get("gate_a") or {}
    min_coverage = float(gcfg.get("min_family_valid_unit_coverage", .95))
    independence = autoevidence.independent_valid_families(
        validity, min_coverage=min_coverage,
        max_invalid_rate=float(gcfg.get("max_invalid_rate", .01)),
        max_nonmonotonic_rate=float(gcfg.get("max_nonmonotonic_rate", .01)),
        min_units=int(math.ceil(min_coverage * len(expected_targets))))
    priority = list(((cfg.get("alignment") or {}).get("consensus") or {})
                    .get("aligner_pair_priority") or [])
    pair = target_mod.select_pair(
        targets, independence["qualifying_families"], priority,
        min_count=int(gcfg.get("min_paired_units", 100)),
        min_rate=float(gcfg.get("min_paired_target_rate", .90)))
    return {
        "diagnostic_only": True,
        "production_gate_evaluated": False,
        "production_spans_frozen": False,
        "l1c_unlocked": False,
        "source_candidate_manifest": candidate_manifest or {},
        "target_alignment_objects": (
            "language-run outer boundaries, embedded-English spans, and EN-ZH switch edges"),
        "raw_family_validity": raw_family.to_dict(orient="records"),
        "raw_unit_validity_by_family_language_reason": raw_detail.to_dict(orient="records"),
        "raw_unit_rows": int(len(candidates)),
        "language_runs": int(len(runs)),
        "target_objects": int(len(targets)),
        "aggregation": aggregation,
        "target_family_validity": validity.to_dict(orient="records"),
        "internal_vs_boundary_invalid_duration": {
            str(family): {
                "internal": int(group["internal_invalid_duration_units"].sum()),
                "boundary": int(group["boundary_invalid_duration_units"].sum()),
                "switch_adjacent": int(
                    group["switch_adjacent_invalid_duration_units"].sum()),
                "utterance_outer": int(
                    group["utterance_outer_invalid_duration_units"].sum()),
            }
            for family, group in runs.groupby("aligner_family", sort=True)
        } if len(runs) else {},
        "paired_overlap_before_target_filtering": [
            {"family_a": a, "family_b": b,
             "paired_raw_valid_units": autoevidence.cross_aligner_agreement(
                 candidates, family_a=a, family_b=b).get("n", 0)}
            for a, b in priority],
        "paired_overlap_after_target_filtering": target_mod.diagnostic_pair_overlaps(targets),
        "natural_family_qualification": independence,
        "eligible_natural_family_pairs": pair,
        "selected_pair_natural_disagreement": target_mod.cross_aligner_target_agreement(
            targets, pair.get("selected_pair", [])),
        "full_role_data_sufficiency": _full_role_data_sufficiency(cfg, roles),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only cached Alignment Gate-A development diagnostic")
    parser.add_argument("--config", default="lss/l1b_valid.yaml")
    parser.add_argument("--diagnostic-output", required=True,
                        help="new diagnostic-only output directory")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments/candidates_all.parquet"
    candidates, verdict = manifest_mod.read_verified(
        candidate_path, cfg=None, require_identity=False)
    if not verdict["ok"]:
        raise SystemExit(f"cached candidates are not authentic: {verdict}")

    output = Path(args.diagnostic_output).resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        # Keep diagnostic artifacts completely outside the production root.
        # A taint sidecar is a safety net, not permission to place a file where
        # a downstream glob or older consumer might mistake it for production.
        raise SystemExit("diagnostic output must be outside the production artifacts root")
    output.mkdir(parents=True, exist_ok=True)
    payload = evaluate_cached(cfg, candidates, verdict.get("manifest"))
    payload["seam_relative_development_diagnostics"] = _seam_diagnostics(root)
    payload["config_sha256"] = sha256_obj(cfg)
    payload["source_artifacts_root"] = str(root)
    report = output / "alignment_gate_a_development_diagnostic.json"
    write_json(payload, report)
    # Human-readable summary intentionally says no decision/freeze occurred.
    markdown = output / "alignment_gate_a_development_diagnostic.md"
    write_report(
        markdown,
        "Alignment Gate-A development diagnostic",
        [("Scope", "Diagnostic-only: no production gate was evaluated, no held-out "
          "generation was consumed, and no spans were frozen."),
         ("Evidence", "```json\n" + json.dumps(payload, indent=2, default=str)
          + "\n```")])
    parent = verdict.get("manifest")
    for path, schema in (
            (report, "lss_alignment_gate_a_development_diagnostic_v1"),
            (markdown, "lss_alignment_gate_a_development_diagnostic_report_v1")):
        manifest_mod.publish(
            path, None, stage="l1b_dev_diagnostic", cfg=cfg,
            parents=[parent] if parent else (),
            taint_reasons=("development_only_diagnostic",), schema=schema,
            extra={
                "diagnostic_only": True,
                "production_gate_evaluated": False,
                "held_out_gate_generation_consumed": False,
                "production_spans_frozen": False,
                "l1c_unlocked": False,
                "source_candidate_path": str(candidate_path),
            })
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
