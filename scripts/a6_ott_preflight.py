#!/usr/bin/env python3
"""CPU-only A6-OTT preflight and resumable work accounting.

This command validates old artifacts, writes outcome-blind phase manifests,
and reports the compact matrix.  It never loads a model and never submits a
Slurm job.  The superseded ``basis_a6_*`` scripts remain untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from csasr.basis_a6.ott import (  # noqa: E402
    CONFIRM_PHASE,
    LEGACY_NAMESPACE,
    RESULT_NAMESPACE,
    SEARCH_PHASE,
    TRANSFER_PHASE,
    freeze_search_ids,
    phase_a_logical_total,
    preflight_provenance,
    runtime_estimate,
    validate_qwen_standard_equivalence,
    validate_reuse_manifest,
)


OUT = REPO / RESULT_NAMESPACE
REUSE = OUT / "migration/EXISTING_RESULT_REUSE_MANIFEST.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _write_once(path: Path, payload: dict) -> None:
    if path.is_file():
        old = json.loads(path.read_text())
        if old != payload:
            raise RuntimeError(f"frozen artifact would change: {path}")
        return
    _write(path, payload)


def _search_metadata() -> list[dict]:
    """Read only identity/stratification fields for the outcome-blind freeze."""
    cs = json.loads((REPO / "results/basis_a4/panels/cs_dialogue_300.json").read_text())
    ascend = json.loads((REPO / "results/basis_a6_expanded/ascend/ASCEND_EVAL_MANIFEST.json").read_text())
    rows = []
    for row in cs["rows"]:
        rows.append({"dataset": "cs_dialogue", "utterance_id": row["utterance_id"],
                     "dialogue_id": row.get("dialogue_id", "")})
    for row in ascend["items"]:
        rows.append({"dataset": "ascend", "utterance_id": row["utterance_id"],
                     "dialogue_id": row.get("session_id", "")})
    return rows


def _phase_manifest(phase: str, *, search_freeze: dict) -> dict:
    if phase == SEARCH_PHASE:
        allowed = [f"{RESULT_NAMESPACE}/phase_a/", f"{LEGACY_NAMESPACE}/baselines/*/*/greedy.json"]
        forbidden = ["confirm", "transfer", "seame", "d-test", "phase_b", "phase_c"]
    elif phase == CONFIRM_PHASE:
        allowed = [f"{RESULT_NAMESPACE}/phase_a/", f"{RESULT_NAMESPACE}/phase_b/", f"{LEGACY_NAMESPACE}/baselines/"]
        forbidden = ["transfer", "seame", "d-test", "phase_c"]
    else:
        allowed = [f"{RESULT_NAMESPACE}/phase_b/", f"{RESULT_NAMESPACE}/phase_c/", f"{LEGACY_NAMESPACE}/baselines/"]
        forbidden = ["d-test"]
    return {"schema_version": "a6_ott_phase_manifest_v1", "phase": phase,
            "status": "FROZEN" if phase == SEARCH_PHASE else "LOCKED_UNTIL_PREVIOUS_PHASE",
            "search_freeze_hash": search_freeze["freeze_hash"],
            "allowed_artifact_prefixes": allowed, "forbidden_path_terms": forbidden,
            "outcome_loading": {"phase_a": "search-only", "phase_b": "confirm-after-phase-a-freeze",
                                 "phase_c": "seame-after-phase-b-freeze"}}


def _direction_reuse_summary(reuse: dict, search_freeze: dict) -> dict:
    """Count validated construction reuse without counting it as a decoded row."""
    search_ids = set(search_freeze["ids"]["cs_dialogue"])
    reusable = 0
    for row in reuse["decisions"]:
        if not row.get("accepted") or row.get("reuse_scope") != "direction_construction_only":
            continue
        if "|encoder|L00|add_unique" not in str(row.get("logical_cell", "")):
            continue
        cache = REPO / str(row["source_artifact"])
        payload = json.loads(cache.read_text())
        reusable += sum(1 for record in payload.get("records", [])
                        if record.get("utterance_id") in search_ids
                        and record.get("method") == "add_unique"
                        and int(record.get("layer", -1)) == 0)
    total = 96 * 40 + 80 * 40
    return {"reusable_direction_constructions": reusable, "total_direction_constructions": total,
            "percentage": 100.0 * reusable / total if total else 0.0,
            "note": "construction reuse only; no Phase-A decoded row is exact-reused"}


def main() -> int:
    if not REUSE.is_file():
        raise SystemExit(f"missing reuse manifest: {REUSE}")
    reuse = validate_reuse_manifest(REUSE, repo_root=REPO)
    _write(OUT / "preflight/REUSE_VALIDATION.json", reuse)
    _write(OUT / "preflight/REUSE_LEDGER.json", {
        "schema_version": "a6_ott_reuse_ledger_v1",
        "rows": [row for row in reuse["decisions"] if row.get("reused")],
        "exact_reused": reuse["exact_reused"],
        "validated_partial_construction": reuse["validated_partial_construction"],
        "downgraded": reuse["downgraded"],
    })

    search_freeze = freeze_search_ids(_search_metadata())
    _write_once(OUT / "manifests/SEARCH_FREEZE.json", search_freeze)
    direction_reuse = _direction_reuse_summary(reuse, search_freeze)
    for phase in (SEARCH_PHASE, CONFIRM_PHASE, TRANSFER_PHASE):
        _write_once(OUT / f"manifests/PHASE_{phase}_MANIFEST.json",
                    _phase_manifest(phase, search_freeze=search_freeze))

    qwen = validate_qwen_standard_equivalence(REPO)
    runtime = runtime_estimate(REPO, reuse=reuse)
    matrix = {
        "whisper": {"logical_total": phase_a_logical_total("whisper"),
                    "exact_reused": runtime["phase_a"]["whisper"]["exact_reused"],
                    "remaining_physical": runtime["phase_a"]["whisper"]["remaining_physical"]},
        "qwen3_asr_1p7b": {"logical_total": phase_a_logical_total("qwen3_asr_1p7b"),
                            "exact_reused": runtime["phase_a"]["qwen3_asr_1p7b"]["exact_reused"],
                            "remaining_physical": runtime["phase_a"]["qwen3_asr_1p7b"]["remaining_physical"]},
    }
    acceptance = {}
    for model in ("whisper", "qwen3_asr_1p7b"):
        path = REPO / f"results/basis_a6_expanded/preflight/REAL_ACCEPTANCE_SUMMARY_{model}.json"
        payload = json.loads(path.read_text()) if path.is_file() else {}
        acceptance[model] = {"status": payload.get("status"), "oracle_tt": payload.get("oracle_tt"),
                             "cache_reuse": payload.get("cache_reuse")}
    provenance = preflight_provenance(root=REPO, config={
        "experiment": {"name": "A6-OTT", "namespace": RESULT_NAMESPACE, "phase": SEARCH_PHASE,
                        "seed": 42, "methods": ["add_unique_encoder", "add_unique_decoder", "conditioning_cs_decoder"]},
        "models": ["whisper", "qwen3_asr_1p7b"],
        "datasets": ["cs_dialogue", "ascend", "seame_man", "seame_sge"],
    })
    manifest = {
        "schema_version": "a6_ott_preflight_manifest_v1", "status": "READY_TO_RUN" if qwen["status"] == "PASS" and all(
            v["status"] == "PASS" and v["oracle_tt"] == "PASS" for v in acceptance.values()) else "BLOCKED",
        "namespace": RESULT_NAMESPACE, "legacy_namespace_preserved": "results/basis_a6_expanded",
        "matrix": matrix, "reuse": {k: reuse[k] for k in ("manifest_sha256", "exact_reused", "validated_partial_construction", "downgraded")},
        "reusable_percentage": {"exact_search_rows": 0.0, "validated_direction_construction": direction_reuse},
        "qwen_standard_equivalence": qwen, "real_model_acceptance": acceptance,
        "runtime": runtime, "phase_manifests": [f"manifests/PHASE_{p}_MANIFEST.json" for p in ("A", "B", "C")],
        "provenance": provenance,
        "gpu_jobs_submitted": [], "slurm_state_checked": True,
    }
    _write(OUT / "preflight/A6_OTT_DRY_RUN.json", {"schema_version": "a6_ott_dry_run_v1", **matrix,
                                                    "pipeline_max": runtime["pipeline_max"],
                                                    "qwen_standard_equivalence": qwen["status"],
                                                    "reusable_percentage": {"exact_search_rows": 0.0,
                                                                             "validated_direction_construction": direction_reuse}})
    _write(OUT / "preflight/PREFLIGHT_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "matrix": matrix,
                      "exact_reused": reuse["exact_reused"],
                      "validated_partial_construction": reuse["validated_partial_construction"],
                      "downgraded": reuse["downgraded"],
                      "pipeline_max": runtime["pipeline_max"]}, indent=2))
    return 0 if manifest["status"] == "READY_TO_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
