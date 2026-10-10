#!/usr/bin/env python3
"""Audit both complete A6-F fixed-direction inventories."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from csasr.basis_a6.fixed import load_reusable_cs_direction
from csasr.basis_a6.protocol import CONDITIONING, MODELS, NON_CONDITIONING


def main() -> int:
    rows, missing = [], []
    expected = 0
    for source in ("cs_dialogue", "ascend"):
        for model, spec in MODELS.items():
            for side in ("encoder", "decoder"):
                methods = list(NON_CONDITIONING) + (list(CONDITIONING) if side == "decoder" else [])
                for layer in spec[f"{side}_layers"]:
                    for method in methods:
                        expected += 1
                        path = REPO / "results/basis_a6_expanded/fixed" / source / "directions" / model / side / f"L{layer:02d}" / f"{method}.npy"
                        if not path.is_file():
                            missing.append({"source": source, "model": model, "side": side, "layer": layer, "method": method, "reason": "artifact_missing"})
                            continue
                        try:
                            vec = __import__("numpy").load(path)
                            norm = float(__import__("numpy").linalg.norm(vec))
                            if vec.ndim != 1 or vec.shape[0] != spec[f"{side}_dim"] or not __import__("numpy").isfinite(vec).all() or abs(norm - 1.0) > 2e-5:
                                raise ValueError(f"shape={vec.shape}, norm={norm}")
                            digest = "sha256:" + hashlib.sha256(__import__("numpy").ascontiguousarray(vec, dtype="float64").tobytes()).hexdigest()
                            rows.append({"source": source, "model": model, "side": side, "layer": layer, "method": method, "path": str(path.relative_to(REPO)), "norm": norm, "direction_hash": digest})
                        except Exception as exc:
                            missing.append({"source": source, "model": model, "side": side, "layer": layer, "method": method, "reason": repr(exc)})
    cs_count = sum(x["source"] == "cs_dialogue" for x in rows)
    ascend_count = sum(x["source"] == "ascend" for x in rows)
    payload = {"schema_version": "basis_a6_direction_audit_v2", "status": "PASS" if not missing and cs_count == 584 and ascend_count == 584 else "BLOCKED",
               "cs_reusable_count": cs_count, "cs_expected_count": 584,
               "cs_missing_count": 584 - cs_count, "ascend_constructed_count": ascend_count, "ascend_expected_count": 584,
               "rows": rows, "missing": missing, "atlas_execution": "NOT_STARTED"}
    path = REPO / "results/basis_a6_expanded/manifests/DIRECTION_AUDIT.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "cs_reusable_count", "cs_expected_count", "cs_missing_count", "ascend_constructed_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
