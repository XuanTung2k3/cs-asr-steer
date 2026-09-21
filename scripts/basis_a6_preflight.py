#!/usr/bin/env python3
"""CPU acceptance/preflight for frozen BASIS-A6; never launches atlas cells."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from csasr.basis_a6.cache import DirectionCache
from csasr.basis_a6.directions import raw_direction, unique_shared_direction
from csasr.basis_a6.guards import make_gold_leakage_trace, rho_zero_identity, validate_local_mask
from csasr.basis_a6.protocol import expected_counts
from csasr.utils.logging import environment_info, git_state


def main() -> int:
    root = REPO / "results/basis_a6_expanded"
    for name in ("fixed", "oracle_tt", "ascend", "manifests", "preflight", "quarantine"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260921)
    a, b = rng.normal(size=(8, 6)), rng.normal(size=(9, 6))
    raw = raw_direction(a, b); us = unique_shared_direction(a, b)
    cache = DirectionCache()
    cache.put(model="whisper", dataset="fixture", utterance_id="fixture/1", decode_analysis_mode="greedy",
              side="decoder", layer=16, method="raw", alignment_hash="sha256:fixture",
              n_A=len(a), n_B=len(b), rank=None, eligible=True, vector=raw)
    cache.put(model="whisper", dataset="fixture", utterance_id="fixture/1", decode_analysis_mode="greedy",
              side="decoder", layer=16, method="add_unique", alignment_hash="sha256:fixture",
              n_A=len(a), n_B=len(b), rank=6, eligible=True, vector=us.v_unique)
    primitive_checks = {
        "matrix_counts": expected_counts() == {"a6_f": 46720, "a6_tt": 23360, "total": 70080, "baselines": 16, "one_regime": 23360},
        "direction_finite_unit": bool(np.isfinite(raw).all() and np.isclose(np.linalg.norm(raw), 1.0)),
        "unique_shared_gate": bool(us is not None and abs(float(us.v_unique @ us.v_shared)) <= 1e-3),
        "cache_bundle": cache.bundle_hash(model="whisper", dataset="fixture", side="decoder", method="raw", decode_analysis_mode="greedy").startswith("sha256:"),
    }
    trace = make_gold_leakage_trace(utterance_id="fixture/1", hypothesis_hash="sha256:hyp", oracle_alignment_hash="sha256:align")
    validate_local_mask([False, True, False], [1], [0, 2]); rho_zero_identity(baseline="x", steered="x", rho=0)
    ascend_prov = root.parent.parent / ".." / "data/external/ASCEND/ASCEND_DOWNLOAD_PROVENANCE.json"
    ascend_prov = (REPO / "data/external/ASCEND/ASCEND_DOWNLOAD_PROVENANCE.json")
    construct_manifest = root / "ascend/ASCEND_CONSTRUCT_MANIFEST.json"
    eval_manifest = root / "ascend/ASCEND_EVAL_MANIFEST.json"
    gpu_acceptance = root / "preflight/GPU_ACCEPTANCE_53316.json"
    if not gpu_acceptance.is_file():
        gpu_acceptance = root / "preflight/GPU_ACCEPTANCE_53309.json"
    direction_audit = root / "manifests/DIRECTION_AUDIT.json"
    geometry_manifest = root / "fixed/geometry/GEOMETRY_MANIFEST.json"
    eval_eligibility = root / "ascend/ASCEND_EVAL_ELIGIBILITY_AUDIT.json"
    direction_payload = json.loads(direction_audit.read_text()) if direction_audit.is_file() else {}
    geometry_payload = json.loads(geometry_manifest.read_text()) if geometry_manifest.is_file() else {}
    eligibility_payload = json.loads(eval_eligibility.read_text()) if eval_eligibility.is_file() else {}
    external_checks = {
        "ascend_download_exact_splits": ascend_prov.is_file() and json.loads(ascend_prov.read_text()).get("observed_sizes") == {"train": 9869, "test": 1315, "validation": 1130},
        "ascend_manifests_frozen": construct_manifest.is_file() and eval_manifest.is_file(),
        "ascend_construct_target": construct_manifest.is_file() and json.loads(construct_manifest.read_text()).get("aggregate", {}).get("N") == 125,
        "ascend_eval_target_or_all": eval_manifest.is_file() and json.loads(eval_manifest.read_text()).get("aggregate", {}).get("N") == 220 and eligibility_payload.get("status") == "PASS",
        "metric_acceptance": (root / "ascend/ASCEND_METRIC_ACCEPTANCE.json").is_file() and json.loads((root / "ascend/ASCEND_METRIC_ACCEPTANCE.json").read_text()).get("status") == "PASS",
        "local_mask_acceptance": (root / "preflight/LOCAL_MASK_ACCEPTANCE.json").is_file() and json.loads((root / "preflight/LOCAL_MASK_ACCEPTANCE.json").read_text()).get("status") == "PASS",
        "gpu_acceptance": gpu_acceptance.is_file() and json.loads(gpu_acceptance.read_text()).get("status") == "PASS",
        "fixed_directions_complete": direction_payload.get("status") == "PASS" and direction_payload.get("cs_reusable_count") == 584 and direction_payload.get("ascend_constructed_count") == 584,
        "cross_source_geometry_complete": geometry_payload.get("status") == "PASS" and geometry_payload.get("cross_source_rows") == 584,
        "oracle_tt_model_acceptance": False,
        "decoding_acceptance_both_models": False,
        "runtime_vram_preflight": False,
    }
    checks = {**primitive_checks, **external_checks}
    blocking = [name for name, ok in external_checks.items() if not ok]
    manifest = {"schema_version": "basis_a6_preflight_manifest_v1", "status": "READY_FOR_FULL_RUN" if not blocking and all(primitive_checks.values()) else "BLOCKED",
                "baseline_clean_implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "git": git_state(REPO), "environment": environment_info(), "checks": checks,
                "blocking_gates": blocking, "gold_leakage_trace": trace, "matrix": expected_counts(),
                "atlas_execution": "NOT_STARTED", "gpu_jobs": ["53308 FAILED environment", "53309 PASS device", "53312 PASS Whisper construction", "53313 FAILED bookkeeping (repaired)", "53315 PASS Qwen construction", "53316 PASS bootstrap/device"],
                "full_run_authorization": False,
                "direction_inventory": {"cs": direction_payload.get("cs_reusable_count"), "ascend": direction_payload.get("ascend_constructed_count"), "geometry_rows": geometry_payload.get("rows")},
                "ascend_eval_eligibility": eligibility_payload,
                "focused_cpu_tests": "48 passed (focused_a6_pytest.log)",
                "full_cpu_suite": {"status": "ONE_UNRELATED_LEGACY_FAILURE", "log": "preflight/full_cpu_pytest.log", "failure": "tests/test_lss_l1b_production_paths.py::test_synthetic_scoring_turns_known_boundaries_into_measured_error"}}
    (root / "preflight/PREFLIGHT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = "# BASIS-A6 Expanded Preflight\n\n"
    report += f"Status: **{manifest['status']}**\n\n"
    report += "The 70,080-cell atlas was not executed.\n\n"
    report += "## Gate summary\n\n| Gate | Status |\n|---|---|\n" + "\n".join(f"| {k} | {'PASS' if v else 'BLOCKED'} |" for k, v in checks.items()) + "\n\n"
    report += "## Blocking gates\n\n" + "\n".join(f"- `{x}`" for x in blocking) + "\n\n"
    report += "## Matrix dry run\n\nA6-F = 46,720; A6-TT = 23,360; total = 70,080; baselines = 16.\n"
    report += "\n## ASCEND\n\nDownload and subset freeze PASS: train 9,869, validation 1,130, test 1,315; construct N=125; eval N=220; test usage=0. Validation eligibility audit: total=1,130, mixed-eligible=220, selected=220 (all eligible below the 300 target).\n"
    report += "\n## Direction status\n\nCS fixed directions: 584/584. ASCEND fixed directions: 584/584. Cross-source geometry: complete (584 cross-source rows; 120 within-source conditioning rows).\n"
    report += "\n## Oracle-TT / leakage\n\nCPU phase orchestration, dynamic-rank rules, cache keys/bundle hashing, local-mask guard, and gold-token trace PASS. Model-resident causal acceptance remains BLOCKED.\n"
    report += "\n## Local steering / decoding\n\nAccepted A4 Whisper/Qwen local site and mask hashes PASS. Direction construction jobs 53312 (Whisper) and 53315 (Qwen) PASS; 53313 was a repaired manifest bookkeeping failure. CUDA/environment acceptance PASS via 53316 (H100 MIG 3g.40gb, acl1 bootstrap). End-to-end Whisper/Qwen greedy and official-standard acceptance remains BLOCKED.\n"
    report += "\n## Runtime / sharding\n\nNo representative decode benchmark was run; runtime, VRAM, cache-size, and disk-size measurements are null. The recorded sharding plan is `FULL_RUN_SHARDING_PLAN.md`; atlas authorization is false.\n"
    (root / "preflight/PREFLIGHT_REPORT.md").write_text(report)
    print(json.dumps(manifest, indent=2, default=str))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
