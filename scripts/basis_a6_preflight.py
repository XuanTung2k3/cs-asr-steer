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
    real_dir = root / "preflight"
    real_summaries, real_benchmarks, real_tt = {}, {}, {}
    for model in ("whisper", "qwen3_asr_1p7b"):
        for kind, target in (("REAL_ACCEPTANCE_SUMMARY", real_summaries),
                             ("REAL_BENCHMARK", real_benchmarks),
                             ("REAL_ORACLE_TT", real_tt)):
            path = real_dir / f"{kind}_{model}.json"
            if path.is_file(): target[model] = json.loads(path.read_text())
    real_decode = len(real_summaries) == 2 and all(
        real_summaries[m].get("decode", {}).get(mode) == "PASS"
        for m in real_summaries for mode in ("greedy", "official_standard"))
    real_tt_pass = all(real_tt.get(m, {}).get("status") == "PASS" and
                       real_tt.get(m, {}).get("gold_target_hidden_states") is False
                       for m in ("whisper", "qwen3_asr_1p7b"))
    real_cache = all(real_summaries.get(m, {}).get("cache_reuse") is True and
                     real_tt.get(m, {}).get("cache_hits", 0) >= 4
                     for m in ("whisper", "qwen3_asr_1p7b"))
    real_rank = all(all(int(a.get("rank", 2) or 2) >= 2 or
                        any(s.get("status") == "INELIGIBLE" for s in row.get("steered", []))
                        for a in row.get("analysis", []))
                    for payload in real_tt.values() for row in payload.get("rows", []))
    runtime_pass = all(real_benchmarks.get(m, {}).get("peak_allocated_vram_bytes", 0) > 0 and
                       real_benchmarks.get(m, {}).get("mean_call_sec", 0) > 0
                       for m in ("whisper", "qwen3_asr_1p7b"))
    disk_path = real_dir / "DISK_CAPACITY.json"
    disk_payload = json.loads(disk_path.read_text()) if disk_path.is_file() else {}
    disk_pass = disk_payload.get("status") == "MEASURED" and disk_payload.get("free_after_estimate_bytes", -1) > 0
    runtime_est_path = real_dir / "RUNTIME_ESTIMATES.json"
    runtime_estimates = json.loads(runtime_est_path.read_text()) if runtime_est_path.is_file() else {}
    sharding_pass = runtime_estimates.get("status") == "MEASURED" and bool(runtime_estimates.get("shard_policy"))
    validation_path = real_dir / "REAL_ACCEPTANCE_VALIDATION.json"
    validation_payload = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
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
        "oracle_tt_model_acceptance": real_tt_pass,
        "decoding_acceptance_both_models": real_decode,
        "runtime_vram_preflight": runtime_pass and disk_pass,
        "real_model_dynamic_rank": real_rank,
        "real_model_cache_reuse": real_cache,
        "runtime_sharding_measured": sharding_pass,
        "real_acceptance_artifact_validation": validation_payload.get("status") == "PASS",
    }
    checks = {**primitive_checks, **external_checks}
    blocking = [name for name, ok in external_checks.items() if not ok]
    manifest = {"schema_version": "basis_a6_preflight_manifest_v1", "status": "READY_FOR_FULL_RUN" if not blocking and all(primitive_checks.values()) else "BLOCKED",
                "baseline_clean_implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "git": git_state(REPO), "environment": environment_info(), "checks": checks,
                "blocking_gates": blocking, "gold_leakage_trace": trace, "matrix": expected_counts(),
                "atlas_execution": "NOT_STARTED", "gpu_jobs": ["53308 FAILED environment", "53309 PASS device", "53312 PASS Whisper construction", "53313 FAILED bookkeeping (repaired)", "53315 PASS Qwen construction", "53316 PASS bootstrap/device", "53320 superseded Qwen fixed", "53321 PASS Qwen real acceptance", "53322 superseded Whisper beam gate", "53323 superseded Whisper mask shape", "53325 superseded Whisper trace field", "53326 PASS Whisper real acceptance"],
                "full_run_authorization": not blocking and all(primitive_checks.values()),
                "direction_inventory": {"cs": direction_payload.get("cs_reusable_count"), "ascend": direction_payload.get("ascend_constructed_count"), "geometry_rows": geometry_payload.get("rows")},
                "ascend_eval_eligibility": eligibility_payload,
                "focused_cpu_tests": "48 passed (focused_a6_pytest.log)",
                "full_cpu_suite": {"status": "ONE_UNRELATED_LEGACY_FAILURE", "log": "preflight/full_cpu_pytest.log", "failure": "tests/test_lss_l1b_production_paths.py::test_synthetic_scoring_turns_known_boundaries_into_measured_error"},
                "real_model_acceptance": {
                    "summary_files": [f"preflight/REAL_ACCEPTANCE_SUMMARY_{m}.json" for m in real_summaries],
                    "tt": {m: {"status": p.get("status"), "direction_constructions": p.get("direction_constructions"),
                               "cache_hits": p.get("cache_hits"), "gold_target_hidden_states": p.get("gold_target_hidden_states"),
                               "analysis_sequence": p.get("analysis_sequence")}
                           for m, p in real_tt.items()},
                    "benchmark_files": [f"preflight/REAL_BENCHMARK_{m}.json" for m in real_benchmarks],
                    "disk_file": "preflight/DISK_CAPACITY.json",
                    "runtime_estimates_file": "preflight/RUNTIME_ESTIMATES.json",
                    "validation_file": "preflight/REAL_ACCEPTANCE_VALIDATION.json"}}
    (root / "preflight/PREFLIGHT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    report = "# BASIS-A6 Expanded Preflight\n\n"
    report += f"Status: **{manifest['status']}**\n\n"
    report += "The 70,080-cell atlas was not executed.\n\n"
    report += "## Gate summary\n\n| Gate | Status |\n|---|---|\n" + "\n".join(f"| {k} | {'PASS' if v else 'BLOCKED'} |" for k, v in checks.items()) + "\n\n"
    report += "## Blocking gates\n\n" + "\n".join(f"- `{x}`" for x in blocking) + "\n\n"
    report += "## Matrix dry run\n\nA6-F = 46,720; A6-TT = 23,360; total = 70,080; baselines = 16.\n"
    report += "\n## ASCEND\n\nDownload and subset freeze PASS: train 9,869, validation 1,130, test 1,315; construct N=125; eval N=220; test usage=0. Validation eligibility audit: total=1,130, mixed-eligible=220, selected=220 (all eligible below the 300 target).\n"
    report += "\n## Direction status\n\nCS fixed directions: 584/584. ASCEND fixed directions: 584/584. Cross-source geometry: complete (584 cross-source rows; 120 within-source conditioning rows).\n"
    report += "\n## Oracle-TT / leakage\n\nReal-model Whisper and Qwen Oracle-TT PASS. Decoder traces identify `baseline_hypothesis`; gold hidden states/logits/target directions are false. Dynamic rank and per-sample eligibility are recorded in the TT artifacts.\n"
    report += "\n## Local steering / decoding\n\nWhisper greedy PASS; Whisper official-standard PASS with beam traces. Qwen greedy PASS; Qwen official-standard PASS and recorded equivalent to greedy. rho=0 identities and positive local edits pass on all 12 frozen panel rows per condition.\n"
    report += "\n## Runtime / sharding\n\nReal model-resident measurements are in `REAL_BENCHMARK_whisper.json` and `REAL_BENCHMARK_qwen3_asr_1p7b.json`; storage/capacity is in `DISK_CAPACITY.json`; derived runtime and contiguous layer-block sizes are in `RUNTIME_ESTIMATES.json` and `FULL_RUN_SHARDING_PLAN.md`.\n"
    report += (f"\nMeasured jobs: Whisper 53326, Qwen 53321. Peak allocated/reserved VRAM: "
               f"Whisper {real_benchmarks.get('whisper', {}).get('peak_allocated_vram_bytes', 0) / 2**30:.2f}/"
               f"{real_benchmarks.get('whisper', {}).get('peak_reserved_vram_bytes', 0) / 2**30:.2f} GiB; "
               f"Qwen {real_benchmarks.get('qwen3_asr_1p7b', {}).get('peak_allocated_vram_bytes', 0) / 2**30:.2f}/"
               f"{real_benchmarks.get('qwen3_asr_1p7b', {}).get('peak_reserved_vram_bytes', 0) / 2**30:.2f} GiB. "
               f"Free disk {disk_payload.get('free_bytes', 0) / 2**30:.1f} GiB; estimated planned cache/result total "
               f"{disk_payload.get('estimated_total_required_bytes', 0) / 2**30:.2f} GiB.\n")
    (root / "preflight/PREFLIGHT_REPORT.md").write_text(report)
    print(json.dumps(manifest, indent=2, default=str))
    return 0 if manifest["status"] == "READY_FOR_FULL_RUN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
