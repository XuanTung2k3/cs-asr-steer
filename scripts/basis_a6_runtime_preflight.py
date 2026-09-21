#!/usr/bin/env python3
"""Record model-resident A6 runtime-preflight requirements without atlas execution."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    gpu = REPO / "results/basis_a6_expanded/preflight/GPU_ACCEPTANCE_53309.json"
    gpu_info = json.loads(gpu.read_text()) if gpu.is_file() else {}
    payload = {
        "schema_version": "basis_a6_runtime_preflight_v1", "status": "BLOCKED_MODEL_BENCHMARK_NOT_RUN",
        "atlas_execution": "NOT_STARTED", "model_resident_runner": True,
        "models": {
            "whisper": {"baseline_analysis": "NOT_RUN", "a6_f": "NOT_RUN", "a6_tt": "NOT_RUN"},
            "qwen3_asr_1p7b": {"baseline_analysis": "NOT_RUN", "a6_f": "NOT_RUN", "a6_tt": "NOT_RUN"},
        },
        "gpu_acceptance": gpu_info,
        "measurements": {"runtime_seconds": None, "peak_vram_gb": None, "cache_size_bytes": None, "disk_size_bytes": None},
        "target_policy": {"estimated_hours_per_job_max": 2.5, "hard_hours_per_job_max": 3.0,
                          "max_simultaneous_gpu_jobs": 2, "prefer_mig_gb": 40},
        "reason": "A6-F direction construction and A6-TT model analysis require explicit GPU model runners; only the minimal CUDA device acceptance was authorized before the full atlas.",
    }
    path = REPO / "results/basis_a6_expanded/preflight/RUNTIME_PREFLIGHT.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
