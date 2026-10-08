#!/usr/bin/env python
"""ST-LOC0 offline calibration builder (frozen spec section 4). Reads ONLY the panel's
``construction_positions`` / ``calibration_membership`` (ids, dialogue, t, stratum) and the site-specific
unedited forced-ZH B states. Fits NEW_P2_DIR_CROSSFIT_V1 leave-one-dialogue-out folds independently at every
physical site with the unchanged ``unique.fit_fold`` (rank 32, all guards, float64, float32 serialization).
No reference target, competitor, outcome, margin or evaluator quantity enters; the pulse runner never imports
this module's inputs (it only reads the sealed per-site/per-dialogue vectors)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf import unique as U
from csasr.inference_cf.core import digest

ALLOWED = ("construction_positions", "calibration_membership")


def fit_site(panel: dict, H_B: np.ndarray) -> dict:
    """All 20 folds at one site. H_B rows follow construction_positions order."""
    con = panel["construction_positions"]
    U.check_construction_fields(con)
    idx = {(p["utterance_id"], int(p["t"])): i for i, p in enumerate(con)}
    out = {}
    for d, frozen in sorted(panel["calibration_membership"]["folds"].items()):
        members = U.fold_members(con, d)
        if digest(members) != frozen["membership_sha256"]:
            raise ValueError(f"fold {d}: membership hash mismatch")
        if any(con[idx[(k[0], int(k[1]))]]["dialogue_id"] == d for k in members["A"] + members["B"]):
            raise ValueError(f"fold {d}: evaluation dialogue leaked into calibration")
        ia = [idx[(k[0], int(k[1]))] for k in members["A"]]
        ib = [idx[(k[0], int(k[1]))] for k in members["B"]]
        rec = U.fit_fold(H_B[ia], H_B[ib])
        out[d] = {"summary": {**U.summary(rec), "excluded_dialogue": d, "n_excluded_rows": len(members["excluded"]),
                              "membership_sha256": frozen["membership_sha256"]},
                  "vector": rec.get("vector"), "arrays": {k: np.asarray(rec[k]) for k in ("eig_A", "eig_B", "sigma", "sigma_raw", "E_A", "scores", "vector64")
                                                          if rec.get(k) is not None}}
    return out


def save_site(out_dir: Path, site_key: str, folds: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vec = {d: f["vector"] for d, f in folds.items() if f["vector"] is not None}
    np.savez(out_dir / f"folds_{site_key}_vectors.npz", **vec)
    arr = {f"{d}__{k}": v for d, f in folds.items() for k, v in f["arrays"].items()}
    np.savez_compressed(out_dir / f"folds_{site_key}_arrays.npz", **arr)
    summ = {d: f["summary"] for d, f in folds.items()}
    (out_dir / f"folds_{site_key}.json").write_text(json.dumps(summ, sort_keys=True, indent=1, default=float) + "\n")
    return summ
