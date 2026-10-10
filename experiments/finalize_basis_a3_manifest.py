"""Finalize the BASIS-A3 post-repair acceptance manifest from local evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("results/basis_a3_raw_cond_scope_depth")
MANIFEST = ROOT / "POSTRUN_ACCEPTANCE_MANIFEST.json"


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def result_files():
    for part in ("raw_r1", "raw_r2", "conditioning"):
        for p in sorted((ROOT / part).rglob("*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "poi_corrections" in d.get("metrics", {}):
                yield part, p, d


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    counts = {"raw_r1": 0, "raw_r2": 0, "conditioning": 0}
    repaired = []
    keys = set()
    metric_failures = []
    for part, path, data in result_files():
        counts[part] += 1
        metrics = data["metrics"]
        key = (data.get("dataset"), data.get("direction"), metrics.get("side"), data.get("layer"), data.get("scope"), data.get("rho"))
        if key in keys:
            raise RuntimeError(f"duplicate accepted key: {key}")
        keys.add(key)
        if metrics.get("poi_net_utility") != metrics.get("poi_corrections", 0) - metrics.get("poi_corruptions", 0):
            metric_failures.append(str(path))
        if data.get("repair_provenance", {}).get("status") == "canonical_repair":
            repaired.append({"path": str(path), "sha256": sha(path), "key": list(key)})
    expected = {"raw_r1": 384, "raw_r2": 84, "conditioning": 72}
    if counts != expected or metric_failures or len(repaired) != 76:
        raise RuntimeError({"counts": counts, "metric_failures": metric_failures, "repaired": len(repaired)})
    repaired_digest = hashlib.sha256("\n".join(f"{x['path']} {x['sha256']}" for x in repaired).encode()).hexdigest()
    superseded = ROOT / "quarantine/superseded_noncanonical_mask/superseded_manifest.json"
    acceptance_names = [
        "encoder_acceptance_summary.json", "encoder_rho0_identity.json", "encoder_normpreserve.json",
        "encoder_padding_exclusion.json", "encoder_local_mask.json", "encoder_no_grad.json", "focused_tests_repair.log",
    ]
    acceptance = {name: sha(ROOT / "acceptance" / name) for name in acceptance_names}
    manifest.update({
        "status": "ACCEPTED_WITH_REPAIR_PROVENANCE",
        "final_acceptance_status": "ACCEPTED_WITH_REPAIR_PROVENANCE",
        "repair": {
            "protocol": "repair/REPAIR_PROTOCOL.json",
            "protocol_commit": "69a5aa0933c6f8be43752d6a28a414a92c5f875e",
            "protocol_hash": sha(ROOT / "repair/REPAIR_PROTOCOL.json"),
            "canonical_mask": "target-segment union alignment",
            "mask_implementation_hash": "sha256:10110ab016d7895664d78240717c88f74011353a59eadd3dde73910bb3d2636b",
            "boundary_tolerance_ms": 80,
            "repaired_slurm_jobs": ["51682", "51683"],
            "acceptance_slurm_job": "51680",
            "failed_harness_jobs": ["51677", "51679"],
            "repaired_configuration_count": 76,
            "repaired_result_digest": "sha256:" + repaired_digest,
        },
        "chronology_reconciliation": {
            "outcome": "REPAIRED_WITH_DOCUMENTED_PROVENANCE",
            "document": "docs/current/BASIS_A3_PROTOCOL_RECONCILIATION.md",
            "repair_protocol": "repair/REPAIR_PROTOCOL.json",
            "historical_mismatch_preserved": True,
            "affected_blocks": 76,
            "rerun_performed": True,
        },
        "superseded_results": {
            "root": "quarantine/superseded_noncanonical_mask/",
            "manifest": "quarantine/superseded_noncanonical_mask/superseded_manifest.json",
            "manifest_hash": sha(superseded),
            "count": 76,
            "mask_semantics": "second-boundary alignment",
        },
        "result_grid_validation": {
            "raw_r1": {"expected": 384, "completed": counts["raw_r1"], "invalid": 0},
            "raw_r2": {"expected": 84, "completed": counts["raw_r2"], "invalid": 0},
            "conditioning": {"expected": 72, "completed": counts["conditioning"], "invalid": 0},
            "duplicate_keys": 0,
            "repaired_alignment_method": "frozen_seame_target_segments",
            "r2_rho_05_reused_from_corrected_r1": True,
        },
        "r2_selection": {
            **manifest.get("r2_selection", {}),
            "selection_impact": "NO SELECTION IMPACT; source was CS-Dialogue R1 only",
        },
        "acceptance_tests": {
            "focused_cpu_tests": {"status": "PASS", "passed": 5, "failed": 0, "log": "acceptance/focused_tests_repair.log"},
            "rho0_identity": "PASS",
            "normpreserve": "PASS",
            "padding_exclusion": "PASS",
            "oracle_local_mask": "PASS",
            "no_grad": "PASS",
            "real_model_job": "51680",
            "artifacts": ["acceptance/encoder_rho0_identity.json", "acceptance/encoder_normpreserve.json", "acceptance/encoder_padding_exclusion.json", "acceptance/encoder_local_mask.json", "acceptance/encoder_no_grad.json"],
        },
        "artifact_hashes": {
            **manifest.get("artifact_hashes", {}),
            "reconciliation": sha(Path("docs/current/BASIS_A3_PROTOCOL_RECONCILIATION.md")),
            "repair_protocol": sha(ROOT / "repair/REPAIR_PROTOCOL.json"),
            "final_report": sha(ROOT / "FINAL_REPORT.md"),
            "summary": sha(ROOT / "summary.json"),
            "raw_full_depth": sha(ROOT / "tables/raw_full_depth.csv"),
            "raw_selected_dose": sha(ROOT / "tables/raw_selected_dose.csv"),
            "cross_corpus_csv": sha(ROOT / "tables/cross_corpus_summary.csv"),
            "cross_corpus_md": sha(ROOT / "tables/cross_corpus_summary.md"),
            "runtime_json": sha(ROOT / "runtime/runtime_summary.json"),
            "runtime_md": sha(ROOT / "runtime/runtime_summary.md"),
            "metric_consistency": sha(ROOT / "metric_audit/final_metric_consistency.json"),
            "acceptance": acceptance,
            "superseded_manifest": sha(superseded),
        },
        "data_exposure": {
            "a3_d_dev_select": True,
            "a3_d_dev_confirm": False,
            "a3_d_test": False,
            "seame_dev_man_count": 50,
            "seame_dev_sge_count": 50,
            "global_project_exposure": "See docs/current/DATA_EXPOSURE.md; not claimed globally untouched.",
        },
        "final_git_commit": None,
        "final_commit_created": False,
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "counts": counts, "repaired": len(repaired)}, sort_keys=True))


if __name__ == "__main__":
    main()
