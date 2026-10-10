#!/usr/bin/env python3
"""Audit frozen ASCEND validation eligibility without reading the test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from csasr.basis_a6.data import load_ascend_split
from csasr.basis_a6.subsets import deterministic_stratified_select, SEED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/external/ASCEND/dataset")
    ap.add_argument("--manifest", default="results/basis_a6_expanded/ascend/ASCEND_EVAL_MANIFEST.json")
    ap.add_argument("--output", default="results/basis_a6_expanded/ascend/ASCEND_EVAL_ELIGIBILITY_AUDIT.json")
    args = ap.parse_args()
    validation = load_ascend_split(args.dataset_dir, "validation")
    eligible = [row for row in validation if row.cs_eligible]
    manifest = json.loads(Path(args.manifest).read_text())
    selected_ids = set(manifest["utterance_ids"])
    selected = [row for row in eligible if row.utterance_id in selected_ids]
    target = 300
    expected_selected = min(target, len(eligible))
    payload = {
        "schema_version": "basis_a6_ascend_eval_eligibility_audit_v1",
        "status": "PASS" if len(selected) == expected_selected and len(eligible) == len(selected) else "FAIL",
        "source_split": "validation",
        "validation_total": len(validation),
        "eligible_mixed_total": len(eligible),
        "target": target,
        "selected_total": len(selected),
        "selected_ids_match_manifest": sorted(selected_ids) == sorted(row.utterance_id for row in selected),
        "all_eligible_selected_when_below_target": len(eligible) <= target and len(selected) == len(eligible),
        "selection_seed": SEED,
        "test_usage": 0,
        "manifest_fingerprint": manifest.get("fingerprint"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
