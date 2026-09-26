#!/usr/bin/env python
"""DG-08 hard D-test lock.

Builds ``results/dg08/DG08_TEST_LOCK.json`` from the three selected checkpoints
per learned finalist (seeds 13/42/73), the deterministic F0/F1 configs, and the
frozen decode/metric/bootstrap protocol.  Run this AFTER all selected
checkpoints exist and BEFORE any D-test decode.  It performs no GPU work.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.utils.config import load_config

SEEDS = [13, 42, 73]
LOCAL_HASH = "sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93"
COND_HASH = "sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991"

# Checkpoint location per finalist per seed.  Seed 42 reuses the frozen
# DG-06/DG-07 selected checkpoints; seeds 13/73 come from the DG-08 training runs.
CHECKPOINTS = {
    "F2_SALSA": {
        42: "results/dg07/LB1_SALSA_EXACT_GLOBAL/selected_checkpoint.pt",
        13: "results/dg08/train/SALSA/seed13/selected_checkpoint.pt",
        73: "results/dg08/train/SALSA/seed73/selected_checkpoint.pt"},
    "F3_LORA": {
        42: "results/dg07/LB2_LORA_MATCHED_BUDGET/selected_checkpoint.pt",
        13: "results/dg08/train/LORA/seed13/selected_checkpoint.pt",
        73: "results/dg08/train/LORA/seed73/selected_checkpoint.pt"},
    "F4_MSTAR": {
        42: "results/dg06/d1/selected_checkpoint.pt",
        13: "results/dg08/train/MSTAR/seed13/selected_checkpoint.pt",
        73: "results/dg08/train/MSTAR/seed73/selected_checkpoint.pt"},
}
EXPECTED_SEED42_SHA = {
    "F2_SALSA": "sha256:03cc0bf68a8c241f55b515e48792624e6d348fa9cb6ae8407a42d1642be96f8e",
    "F3_LORA": "sha256:7861cdf83b364a417f1cf627f42660c150f1386dccc8898355ab79d04fea0046",
    "F4_MSTAR": "sha256:b9c45a1e279a5b5acf84c45adff6f48a2eae67f0e1062294b9211c7b77fef241"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _selected_epoch(rel: str) -> Any:
    sel = Path(REPO / rel).parent / "selection.json"
    if sel.exists():
        return json.loads(sel.read_text(encoding="utf-8")).get("selected_epoch")
    return None


def build_lock(cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for system, seed_map in CHECKPOINTS.items():
        checkpoints[system] = {}
        for seed in SEEDS:
            rel = seed_map[seed]
            abs_path = REPO / rel
            if not abs_path.exists():
                raise RuntimeError(f"missing selected checkpoint for {system} seed {seed}: {rel}")
            digest = _sha256_file(abs_path)
            if seed == 42 and digest != EXPECTED_SEED42_SHA[system]:
                raise RuntimeError(f"{system} seed 42 checkpoint hash "
                                   f"{digest} != frozen {EXPECTED_SEED42_SHA[system]}")
            checkpoints[system][str(seed)] = {
                "checkpoint": rel, "sha256": digest, "seed": seed,
                "selected_epoch": _selected_epoch(rel),
                "reused_from_frozen": seed == 42}

    dtest_path = Path(cfg["data"]["dtest_locked_parquet"])
    lock = {
        "schema_version": "dg08_test_lock_v1",
        "status": "D-TEST CONFIGURATION LOCKED - NO FURTHER SCIENTIFIC SELECTION",
        "git_commit": _git_commit(),
        "model": {"id": cfg["model"]["id"], "hub_id": cfg["model"].get("hub_id"),
                  "revision": cfg["model"].get("revision"), "dtype": cfg["model"].get("dtype"),
                  "local_files_only": cfg["model"].get("local_files_only")},
        "layer": 24,
        "basis_hashes": {"local": LOCAL_HASH, "cond": COND_HASH},
        "objective": "L_corr(C_E) + 1.0 * L_ret,M(R_M), KL(p0 || p_theta)",
        "lambda_m": 1.0, "beta": 4.465628877080159,
        "seeds": SEEDS,
        "finalists": {
            "F0_FROZEN_WHISPER": {"kind": "baseline", "deterministic": True},
            "F1_DG04_FROZEN_STEERING": {
                "kind": "fixed_direction_v_local", "rho": 0.5,
                "beta_nominal": 4.465628877080159, "scale_s_l": 8.931257754160319,
                "reference": "results/dg04/reference.json", "deterministic": True},
            "F2_SALSA": {"kind": "global_vector", "trainable_parameters": 1280},
            "F3_LORA": {"kind": "qv_lora_l24", "rank": 9, "alpha": 9.0,
                        "trainable_parameters": 46080},
            "F4_MSTAR": {"kind": "adaptive_controller_D1", "trainable_parameters": 43651},
        },
        "checkpoints": checkpoints,
        "decode": {"greedy": cfg["decode"]["greedy"], "beam5": cfg["decode"]["beam5"]},
        "metrics_schema": "metrics_v1",
        "outside_harm_on_dtest": "NOT_AVAILABLE_no_candidate_poi_alignments",
        "dtest_manifest_path": str(dtest_path),
        "dtest_manifest_fingerprint": _sha256_file(dtest_path),
        "bootstrap": cfg["bootstrap"],
        "normalization": "metrics_v1 canonical (DG-01); gains = baseline - method (positive better)",
    }
    return lock


def main() -> int:
    cfg = load_config("configs/dg08_locked_eval.yaml")
    lock = build_lock(cfg)
    out = REPO / "results/dg08/DG08_TEST_LOCK.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({s: {k: v["sha256"] for k, v in lock["checkpoints"][s].items()}
                      for s in lock["checkpoints"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
