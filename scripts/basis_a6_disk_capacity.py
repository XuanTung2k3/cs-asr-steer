#!/usr/bin/env python3
"""Measure A6 preflight storage and conservative full-run capacity."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a6_expanded/preflight/DISK_CAPACITY.json"


def size(path: Path) -> int:
    if path.is_file(): return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def main() -> int:
    usage = shutil.disk_usage(REPO)
    fixed = size(REPO / "results/basis_a6_expanded/fixed")
    preflight = size(REPO / "results/basis_a6_expanded/preflight")
    caches = {}
    for model in ("whisper", "qwen3_asr_1p7b"):
        caches[model] = size(REPO / "results/basis_a6_expanded/preflight" / f"tt_cache_{model}_greedy")
    # The observed cache contains 4 samples, one decode mode, and the two
    # representative layers.  Scale to the frozen 620-row panel, both decode
    # IDs, and the complete model method/layer inventory.  The planned cache
    # writer stores float32 vectors while retaining float64 canonical hashes.
    scale = {"whisper": (620 / 4) * 2 * (320 / 10),
             "qwen3_asr_1p7b": (620 / 4) * 2 * (264 / 10)}
    tt_raw = {m: int(caches[m] * scale[m]) for m in caches}
    tt_optimized = {m: int(tt_raw[m] / 2) for m in caches}
    direction_cache = sum(tt_optimized.values())
    # Result/diagnostic estimate is intentionally conservative and includes a
    # 2x allowance over the current 12-row detailed acceptance artifacts.
    diagnostics = int(sum(size(REPO / "results/basis_a6_expanded/preflight" / f"REAL_ACCEPTANCE_{m}_greedy.json") for m in caches) * (620 / 12) * 2)
    total = fixed + direction_cache + diagnostics
    payload = {
        "schema_version": "basis_a6_disk_capacity_v1", "status": "MEASURED",
        "filesystem": str(REPO), "free_bytes": int(usage.free), "used_bytes": int(usage.used),
        "total_bytes": int(usage.total), "current_fixed_result_bytes": fixed,
        "current_preflight_bytes": preflight, "current_tt_cache_bytes": caches,
        "estimated_tt_cache_uncompressed_float64_bytes": tt_raw,
        "estimated_tt_cache_planned_float32_bytes": tt_optimized,
        "estimated_oracle_tt_direction_cache_bytes": direction_cache,
        "estimated_transcript_logit_diagnostics_bytes": diagnostics,
        "estimated_fixed_result_storage_bytes": fixed,
        "estimated_total_required_bytes": total,
        "free_after_estimate_bytes": int(usage.free - total),
        "capacity_policy": "float32 persisted vectors with canonical float64 hashes; no hidden states persisted",
        "atlas_execution": "NOT_STARTED",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
