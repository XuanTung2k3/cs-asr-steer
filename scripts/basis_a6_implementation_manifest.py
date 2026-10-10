#!/usr/bin/env python3
"""Write the additive A6 implementation/provenance manifest."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from csasr.utils.hashing import sha256_file
from csasr.utils.logging import environment_info, git_state


def main() -> int:
    docs = [REPO / "docs/current" / name for name in (
        "BASIS_A6_EXPANDED_SPEC.md", "BASIS_A6_FIXED_SPEC.md", "BASIS_A6_ORACLE_TT_SPEC.md",
        "BASIS_A6_ASCEND_DATA_SPEC.md", "BASIS_A6_EXECUTION_PLAN.md")]
    out = REPO / "results/basis_a6_expanded/manifests/A6_IMPLEMENTATION_MANIFEST.json"
    payload = {"schema_version": "basis_a6_implementation_manifest_v1", "status": "BLOCKED",
               "baseline_clean_implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "git": git_state(REPO), "environment": environment_info(),
               "spec_hashes": {p.name: sha256_file(p) for p in docs},
               "ascend_download_provenance": "data/external/ASCEND/ASCEND_DOWNLOAD_PROVENANCE.json",
               "ascend_construct_manifest": "results/basis_a6_expanded/ascend/ASCEND_CONSTRUCT_MANIFEST.json",
               "ascend_eval_manifest": "results/basis_a6_expanded/ascend/ASCEND_EVAL_MANIFEST.json",
               "result_namespaces": ["fixed", "oracle_tt", "ascend", "manifests", "preflight", "quarantine"],
               "matrix": {"a6_f": 46720, "a6_tt": 23360, "total": 70080, "baselines": 16},
               "atlas_execution": "NOT_STARTED", "full_run_authorization": False}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
