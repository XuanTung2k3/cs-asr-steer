#!/usr/bin/env python3
"""Write complete same-model A6-F cross-source geometry after direction audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from csasr.basis_a6.protocol import CONDITIONING, MODELS, NON_CONDITIONING

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/basis_a6_expanded/fixed"


def digest(v):
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(v, dtype="float64").tobytes()).hexdigest()


def load(source, model, side, layer, method):
    return np.asarray(np.load(ROOT / source / "directions" / model / side / f"L{layer:02d}" / f"{method}.npy"), dtype="float64")


def main() -> int:
    rows = []
    for model, spec in MODELS.items():
        for side in ("encoder", "decoder"):
            methods = list(NON_CONDITIONING) + (list(CONDITIONING) if side == "decoder" else [])
            for layer in spec[f"{side}_layers"]:
                for method in methods:
                    cs, ac = load("cs_dialogue", model, side, layer, method), load("ascend", model, side, layer, method)
                    rows.append({"comparison": "cross_source", "model": model, "side": side, "layer": layer,
                                 "method": method, "cosine": float(cs @ ac), "cs_norm": float(np.linalg.norm(cs)),
                                 "ascend_norm": float(np.linalg.norm(ac)), "cs_hash": digest(cs), "ascend_hash": digest(ac)})
                if side == "decoder":
                    for source in ("cs_dialogue", "ascend"):
                        cs, ac = load(source, model, side, layer, "conditioning_cs"), load(source, model, side, layer, "conditioning_all")
                        rows.append({"comparison": "within_source_cond_cs_vs_all", "source": source, "model": model,
                                     "side": side, "layer": layer, "method": "conditioning_cs_vs_all",
                                     "cosine": float(cs @ ac), "cs_norm": float(np.linalg.norm(cs)),
                                     "ascend_norm": float(np.linalg.norm(ac)), "cs_hash": digest(cs), "ascend_hash": digest(ac)})
    expected = 584 + 2 * (32 + 28)
    if len(rows) != expected or any(abs(float(row["cs_norm"]) - 1) > 2e-5 or abs(float(row["ascend_norm"]) - 1) > 2e-5 for row in rows):
        raise SystemExit(f"geometry incomplete/invalid: rows={len(rows)} expected={expected}")
    out = ROOT / "geometry/fixed_cs_vs_ascend.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["comparison", "source", "model", "side", "layer", "method", "cosine", "cs_norm", "ascend_norm", "cs_hash", "ascend_hash"]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    payload = {"schema_version": "basis_a6_cross_source_geometry_v1", "status": "PASS", "rows": len(rows),
               "cross_source_rows": 584, "within_source_conditioning_rows": 120, "models": list(MODELS),
               "path": str(out.relative_to(REPO))}
    (out.parent / "GEOMETRY_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
