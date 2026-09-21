#!/usr/bin/env python3
"""Materialize accepted CS-source A4/A5 vectors into the additive A6 namespace.

The source artifacts are read-only comparators. This script never writes under
results/basis_a4 or results/basis_a5_unique_shared; it creates provenance-bound
copies under results/basis_a6_expanded/fixed/cs_dialogue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.basis_a6.fixed import load_reusable_cs_direction
from csasr.basis_a6.protocol import CONDITIONING, MODELS, NON_CONDITIONING
from experiments.basis_a5_unique_shared import _a4_row


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/basis_a6_expanded/fixed/cs_dialogue")
    args = ap.parse_args()
    root = REPO / args.output_dir
    rows = []
    for model, spec in MODELS.items():
        for side in ("encoder", "decoder"):
            methods = list(NON_CONDITIONING) + (list(CONDITIONING) if side == "decoder" else [])
            for layer in spec[f"{side}_layers"]:
                for method in methods:
                    if method == "conditioning_cs":
                        continue
                    vector, provenance = load_reusable_cs_direction(model=model, side=side, layer=layer, method=method)
                    path = root / "directions" / model / side / f"L{layer:02d}" / f"{method}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, np.asarray(vector, dtype=np.float64))
                    rows.append({"source": "cs_dialogue", "model": model, "side": side,
                                 "layer": int(layer), "method": method,
                                 "path": str(path.relative_to(REPO)),
                                 "direction_hash": provenance["direction_hash"],
                                 "artifact_hash": _hash(path), "norm": float(np.linalg.norm(vector)),
                                 "definition": provenance["definition"],
                                 "site_hash": _a4_row(model, "cs_dialogue", "Raw", side, int(layer)).get("site_hash"),
                                 "reused_read_only": True,
                                 "source_path": provenance["path"],
                                 "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})
    payload = {"schema_version": "basis_a6_fixed_cs_materialization_v1", "status": "PASS",
               "source": "cs_dialogue", "rows": rows, "count": len(rows),
               "expected_before_conditioning_cs": 524}
    out = root / "CS_FIXED_MATERIALIZATION.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "count": len(rows), "manifest": str(out.relative_to(REPO))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
