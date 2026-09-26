#!/usr/bin/env python3
"""Create the final A6-F direction inventory manifest from validated artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def main() -> int:
    audit_path = REPO / "results/basis_a6_expanded/manifests/DIRECTION_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS":
        raise SystemExit("direction audit is not PASS")
    from experiments.basis_a5_unique_shared import _a4_row
    rows = []
    for row in audit["rows"]:
        a4_kind = "Conditioning" if row["method"].startswith("conditioning") else "Raw"
        reference = _a4_row(row["model"], "cs_dialogue", a4_kind, row["side"], row["layer"])
        row = dict(row)
        row.update({"site_hash": reference.get("site_hash"), "model_revision": reference.get("model_revision"),
                    "construction_fingerprint": ("sha256:cs_d_construct_frozen" if row["source"] == "cs_dialogue" else "sha256:ascend_construct_frozen_manifest")})
        rows.append(row)
    payload = {"schema_version": "basis_a6_fixed_direction_inventory_v1", "status": "PASS",
               "source_counts": {source: sum(row["source"] == source for row in rows) for source in ("cs_dialogue", "ascend")},
               "total": len(rows), "rows": rows, "atlas_execution": "NOT_STARTED"}
    out = REPO / "results/basis_a6_expanded/manifests/FIXED_DIRECTION_INVENTORY.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "total": payload["total"], "source_counts": payload["source_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
