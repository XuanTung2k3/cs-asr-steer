#!/usr/bin/env python3
"""Minimal GPU-only acceptance; it performs no scientific A6 decoding."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    import torch
    payload = {"schema_version": "basis_a6_gpu_acceptance_v1", "status": "PASS" if torch.cuda.is_available() else "FAIL",
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda": torch.cuda.is_available(),
               "device_count": torch.cuda.device_count(), "devices": []}
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        payload["devices"].append({"index": i, "name": p.name, "total_memory_gb": p.total_memory / 1e9})
    path = REPO / "results/basis_a6_expanded/preflight" / f"GPU_ACCEPTANCE_{payload['slurm_job_id'] or 'local'}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
