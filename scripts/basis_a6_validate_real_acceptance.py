#!/usr/bin/env python3
"""CPU validator for completed real-model A6 acceptance artifacts."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a6_expanded/preflight"


def main() -> int:
    checks = {}
    for model in ("whisper", "qwen3_asr_1p7b"):
        for mode in ("greedy", "official_standard"):
            x = json.loads((OUT / f"REAL_ACCEPTANCE_{model}_{mode}.json").read_text())
            checks[f"{model}_{mode}"] = (x.get("status") == "PASS" and len(x.get("rows", [])) == 12 and
                all(r.get("rho0_identity") and r.get("metrics_equal") and r.get("positive_nonzero") and r.get("local_only")
                    for r in x["rows"]))
        tt = json.loads((OUT / f"REAL_ORACLE_TT_{model}.json").read_text())
        checks[f"{model}_tt"] = (tt.get("status") == "PASS" and tt.get("gold_target_hidden_states") is False and
                                  tt.get("cache_hits", 0) >= 4 and tt.get("direction_constructions", 0) > 0 and
                                  all(row.get("gold_leakage", {}).get("sequence_source") == "baseline_hypothesis"
                                      for row in tt.get("rows", [])))
        summary = json.loads((OUT / f"REAL_ACCEPTANCE_SUMMARY_{model}.json").read_text())
        checks[f"{model}_cache"] = summary.get("cache_reuse") is True
    payload = {"schema_version": "basis_a6_real_acceptance_validation_v1",
               "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "atlas_execution": "NOT_STARTED"}
    (OUT / "REAL_ACCEPTANCE_VALIDATION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
