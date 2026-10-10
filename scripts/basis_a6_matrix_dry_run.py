#!/usr/bin/env python3
"""Enumerate the complete A6 configuration matrix without model execution."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from csasr.basis_a6.protocol import enumerate_a6_f, enumerate_a6_tt, expected_counts


def main() -> int:
    out = REPO / "results/basis_a6_expanded/manifests"
    out.mkdir(parents=True, exist_ok=True)
    fixed, tt = enumerate_a6_f(), enumerate_a6_tt()
    rows = fixed + tt
    key_fields = ("branch", "construction_source", "model", "dataset", "side", "layer", "method", "rho", "decode_mode")
    keys = [tuple(row.get(k) for k in key_fields) for row in rows]
    counts = expected_counts()
    checks = {"a6_f": len(fixed) == 46720, "a6_tt": len(tt) == 23360,
              "total": len(rows) == 70080, "baselines": counts["baselines"] == 16,
              "duplicate_scientific_keys": len(keys) - len(set(keys)) == 0,
              "atlas_execution_not_started": True}
    manifest = {"schema_version": "basis_a6_matrix_dry_run_v1", "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks, "counts": counts, "scientific_execution": "NOT_STARTED"}
    (out / "A6_MATRIX_DRY_RUN.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "A6_MATRIX_ENUMERATION.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=key_fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
