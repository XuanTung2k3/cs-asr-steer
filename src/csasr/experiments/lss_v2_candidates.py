"""Generate one isolated dialogue-atomic v2 natural-candidate generation.

This is deliberately narrower than L1b: it samples the immutable v2 roles and
runs the configured automatic aligners. It cannot prepare/evaluate Gate A,
touch status/freeze/exposure state, generate synthetic items, or freeze spans.
The generation root is write-once and a complete marker is published last.

Without ``--execute`` the command performs a read-only production-config and
namespace validation. Model inference is never started implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..lss import manifest as manifest_mod
from ..lss.specfreeze import verify as verify_spec
from ..lss.v2_namespace import configured_paths, validate_namespace
from ..nat5h.statusing import atomic_write_json
from ..utils.config import load_config
from ..utils.hashing import sha256_file
from ..utils.logging import setup_logging
from . import lss_l1b_valid

STAGE = "l1b_candidates_dialogue_v2"
DEFAULT_CONFIG = "lss/l1b_candidates_dialogue_v2.yaml"
GENERATION_COMPLETE = "generation_complete.manifest.json"


def _selection(cfg: dict[str, Any]) -> dict[str, Any]:
    block = dict(cfg.get("candidate_generation") or {})
    if not block.get("candidate_generation_only"):
        raise SystemExit("candidate_generation.candidate_generation_only must be true")
    if "pred_start_offset" not in block or isinstance(
            block.get("pred_start_offset"), bool):
        raise SystemExit(
            "candidate_generation.pred_start_offset must be explicitly configured")
    try:
        offset = int(block["pred_start_offset"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "candidate_generation.pred_start_offset must be an integer") from exc
    if isinstance(block["pred_start_offset"], float) and not \
            block["pred_start_offset"].is_integer():
        raise SystemExit(
            "candidate_generation.pred_start_offset must be an integer")
    source = str(block.get("decision_source") or "").strip()
    if not source:
        raise SystemExit("candidate_generation.decision_source is required")
    return {
        "available": True,
        "pred_start_offset": offset,
        "convention": str(block.get("pred_start_convention") or ""),
        "source": source,
        "frozen_in": "v2_spec_and_versioned_execution_config",
    }


def _parents(cfg: dict, paths) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    freeze_verdict = manifest_mod.verify(
        paths.freeze_path, cfg=cfg, require_identity=True)
    if not freeze_verdict["ok"]:
        raise SystemExit(
            "v2 freeze artifact manifest is not authentic for this execution "
            f"configuration: {freeze_verdict['verdict']} "
            f"{freeze_verdict.get('detail', '')}")
    parents.append(freeze_verdict["manifest"])
    for role in lss_l1b_valid.sweep_roles(cfg):
        path = lss_l1b_valid.role_path(cfg, role)
        verdict = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not verdict["ok"]:
            raise SystemExit(
                f"v2 role {role} is not authentic: {verdict['verdict']} "
                f"{verdict.get('detail', '')}")
        published = verdict["manifest"]
        if published.get("diagnostic_only") or published.get("taint_reasons"):
            raise SystemExit(f"v2 role {role} is tainted/diagnostic-only")
        parents.append(published)
    return parents


def _write_complete(root: Path, *, report: dict[str, Any], cfg: dict,
                    paths, selection: dict[str, Any]) -> Path:
    requested = list((cfg.get("alignment") or {}).get("families") or [])
    families = dict(report.get("families") or {})
    missing = [name for name in requested
               if (families.get(name) or {}).get("state") != "ok"]
    candidate_verdict = manifest_mod.verify(
        paths.candidate_path, cfg=cfg, require_identity=True)
    if missing or not candidate_verdict["ok"]:
        raise SystemExit(
            "candidate generation is incomplete and will not receive a complete "
            f"marker; unavailable families={missing}, candidate_verdict="
            f"{candidate_verdict['verdict']}")
    inventory = []
    for artifact in sorted(path for path in root.rglob("*") if path.is_file()):
        inventory.append({
            "path": str(artifact),
            "relative_path": str(artifact.relative_to(root)),
            "sha256": sha256_file(artifact),
            "bytes": int(artifact.stat().st_size),
        })
    marker = root / GENERATION_COMPLETE
    atomic_write_json(marker, {
        "schema": "lss_dialogue_candidate_generation_v1",
        "state": "completed",
        "complete": True,
        "published": True,
        "namespace_version": "lss-dialogue-v2",
        "generation_version": (cfg.get("candidate_generation") or {}).get("version"),
        "candidate_path": str(paths.candidate_path),
        "candidate_sha256": candidate_verdict["manifest"]["sha256"],
        "configured_families": requested,
        "active_families": list(report.get("active") or []),
        "selection": selection,
        "sampling": report.get("sampling"),
        "artifacts": inventory,
    })
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--execute", action="store_true",
        help="actually run aligners; without this flag validation is read-only")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    selection = _selection(cfg)
    isolation = validate_namespace(
        cfg, for_execution=args.execute, require_freeze=True)
    paths = configured_paths(cfg)
    spec_verdict = verify_spec(paths.freeze_path)
    if not spec_verdict["ok"]:
        raise SystemExit(
            "v2 spec freeze or one of its referenced artifacts failed "
            "verification: " + json.dumps(spec_verdict, default=str))

    if not args.execute:
        print(json.dumps({
            "state": "validated_no_execution",
            "models_loaded": False,
            "artifacts_written": [],
            "isolation": isolation,
            "selection": selection,
            "freeze": spec_verdict,
        }, indent=2, default=str))
        return 0

    parents = _parents(cfg, paths)
    log = setup_logging(level=str((cfg.get("runtime") or {}).get(
        "log_level", "INFO")))
    taint = {"diagnostic_only": False, "taint_reasons": []}
    qwen = lss_l1b_valid._qwen_runner(
        cfg, log, None, taint, parents, producing_stage=STAGE)
    report = lss_l1b_valid._run_aligner_sweep(
        cfg, log, overwrite=False, run_dir=None,
        roles=lss_l1b_valid.sweep_roles(cfg), qwen_runner=qwen,
        parents=parents, taint=taint, alignment_selection=selection,
        producing_stage=STAGE,
        require_all_configured_families=bool(
            (cfg.get("candidate_generation") or {}).get(
                "require_all_configured_families", True)))
    marker = _write_complete(
        paths.candidate_generation_root, report=report, cfg=cfg, paths=paths,
        selection=selection)
    print(json.dumps({
        "state": "completed",
        "candidate": str(paths.candidate_path),
        "generation_complete": str(marker),
        "rows": report.get("rows"),
        "active_families": report.get("active"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
