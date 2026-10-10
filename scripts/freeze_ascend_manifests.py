#!/usr/bin/env python3
"""Freeze the train-only ASCEND construct and validation-only eval manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from csasr.basis_a6.data import load_ascend_split
from csasr.basis_a6.subsets import build_subset_manifest, write_manifests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/external/ASCEND/dataset")
    ap.add_argument("--output-dir", default="results/basis_a6_expanded/ascend")
    ap.add_argument("--construct-n", type=int, default=125)
    ap.add_argument("--eval-n", type=int, default=300)
    args = ap.parse_args()
    train = load_ascend_split(args.dataset_dir, "train")
    validation = load_ascend_split(args.dataset_dir, "validation")
    provenance_path = Path(args.dataset_dir).parent / "ASCEND_DOWNLOAD_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    fingerprints = provenance.get("hf_fingerprints", {})
    construct = build_subset_manifest(train, split="train", role="ASCEND-construct", target=args.construct_n,
                                       dataset_fingerprint=fingerprints.get("train"))
    evaluation = build_subset_manifest(validation, split="validation", role="ASCEND-eval", target=args.eval_n,
                                      dataset_fingerprint=fingerprints.get("validation"))
    # The builder accepts iterables; counts above are made explicit here so
    # the manifest cannot accidentally claim a post-filter denominator.
    construct["aggregate"]["N_scanned"] = len(train)
    evaluation["aggregate"]["N_scanned"] = len(validation)
    write_manifests(args.output_dir, construct, evaluation)
    Path(args.output_dir, "ASCEND_ROLE_FREEZE.json").write_text(json.dumps({
        "status": "PASS", "test_usage": 0, "construct_source": "train",
        "eval_source": "validation", "construct_N": construct["aggregate"]["N"],
        "eval_N": evaluation["aggregate"]["N"], "construct_fingerprint": construct["fingerprint"],
        "eval_fingerprint": evaluation["fingerprint"], "hf_fingerprints": fingerprints,
        "download_provenance": str(provenance_path.relative_to(Path(args.dataset_dir).parent.parent.parent)) if provenance_path.is_file() else None,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
