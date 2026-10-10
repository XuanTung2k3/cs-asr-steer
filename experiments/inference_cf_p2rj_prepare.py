#!/usr/bin/env python
"""CPU-only P2-RJ preparation.

``positions``: build the frozen P2-RJ position file from the unchanged P2-R population and the
saved P2-R run3 records (fixed competitor c*, P2-R identity reference values, observed P2-R arm
changes for the first-order sanity check). No P2-RJ outcome is read or computed here.

``manifest``: frozen config, positions, source hashes, Git state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2rj_jacobian_diagnosis_v1"
CONFIG = "configs/inference_cf/p2_rj_jacobian_diagnosis.json"
POSITIONS = "results/inference_cf/p2rj/positions.json"
P2R_POPULATION = "results/inference_cf/p2r/population.json"
P2R_RUN = "results/inference_cf/p2r/run3"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
ARMS = ("plus_d", "minus_d", "random")
SOURCES = ("docs/inference_cf/P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md", "docs/inference_cf/P2_RJ_PRE_RUN_AUDIT.md",
           CONFIG, POSITIONS, P2R_POPULATION, "experiments/inference_cf_p2rj.py",
           "experiments/inference_cf_p2rj_prepare.py", "experiments/inference_cf_p2rj_analyze.py",
           "experiments/inference_cf_p2r.py", "experiments/inference_cf_cached.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "slurm/inference_cf_p2rj.sbatch")


def top_lookup(summary: dict, token: int) -> float | None:
    """log p of ``token`` from a saved P2-R top-20 list (None when absent)."""
    for i, lp in zip(summary["top_ids"], summary["top_logp"]):
        if int(i) == int(token):
            return float(lp)
    return None


def competitor(none_summary: dict, target_ids: list[int]) -> int:
    """Frozen c*: the first unedited top-20 token that is not in Y_ref."""
    ref = set(int(x) for x in target_ids)
    return next(int(i) for i in none_summary["top_ids"] if int(i) not in ref)


def build_positions(pop: dict, rows: dict) -> list[dict]:
    out = []
    for stratum in ("EN-confusion", "EN-correct", "ZH-correct"):
        for c in pop["d1"][stratum]:
            rec = rows[c["utterance_id"]]["pulses"][str(c["t"])]
            none = rec["none"]
            cstar = competitor(none, c["target_ids"])
            lc0 = top_lookup(none, cstar)
            p2r = {"argmax": none["argmax"], "logp_ref": none["logp_ref"],
                   "margin": none["margin_ref_vs_competitor"], "logp_competitor": lc0,
                   "cos_d_state": rec["cos_d_state"], "dir": rec["dir"], "dir_norm": rec["dir_norm"],
                   "query": rec["query"], "arms": {}}
            for a in ARMS:
                x = rec["arms"][a]
                s = x["summary"]
                lc = top_lookup(s, cstar)
                p2r["pre_norm"] = x["pre_norm"]
                p2r["arms"][a] = {"s": x["solver"].get("s"), "phi": x["solver"].get("phi"),
                                  "status": x["solver"].get("status"), "edit_norm": x["edit_norm"],
                                  "logp_ref": s["logp_ref"], "logp_competitor": lc,
                                  "obs_d_logp_ref": s["logp_ref"] - none["logp_ref"],
                                  "obs_d_margin": None if lc is None or lc0 is None else
                                  (s["logp_ref"] - lc) - (none["logp_ref"] - lc0)}
            out.append({"utterance_id": c["utterance_id"], "t": c["t"], "stratum": stratum,
                        "dialogue_id": c["dialogue_id"], "span_initial": c.get("span_initial"),
                        "companion": c["companion"], "target_ids": [int(x) for x in c["target_ids"]],
                        "baseline_token": int(c["baseline_token"]), "competitor": cstar, "p2r": p2r})
    return out


def cmd_positions(args) -> None:
    cfg = json.loads((ROOT / CONFIG).read_text())
    pop = json.loads((ROOT / P2R_POPULATION).read_text())
    if pop["population_hash"] != cfg["p2r_reference"]["population_hash"]:
        raise ValueError("P2-R population hash differs from the frozen P2-RJ config")
    man = json.loads((ROOT / P2R_RUN / "manifest.json").read_text())
    if man["manifest_hash"] != cfg["p2r_reference"]["run_manifest_hash"]:
        raise ValueError("P2-R run3 manifest hash differs from the frozen P2-RJ config")
    rows = {}
    for i, uid in enumerate(pop["utterances"]):
        r = json.loads((ROOT / P2R_RUN / "rows" / f"{i:03d}.json").read_text())
        if r["status"] != "ok" or r["identity"] != uid or r["manifest_hash"] != man["manifest_hash"]:
            raise ValueError(f"P2-R row {i} not ok")
        rows[uid] = r
    positions = build_positions(pop, rows)
    doc = {"schema": SCHEMA + "_positions", "config_hash": digest(cfg), "p2r_population_hash": pop["population_hash"],
           "p2r_run_manifest_hash": man["manifest_hash"], "utterances": pop["utterances"], "positions": positions}
    doc["positions_hash"] = digest(doc)
    path = ROOT / POSITIONS
    if path.exists():
        old = json.loads(path.read_text())
        if old["positions_hash"] != doc["positions_hash"]:
            raise FileExistsError("frozen P2-RJ positions differ; never overwrite")
        print("positions unchanged", doc["positions_hash"])
        return
    atomic_json(path, doc)
    print(json.dumps({"positions": len(positions), "positions_hash": doc["positions_hash"]}))


def cmd_manifest(args) -> None:
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition

    cfg = json.loads((ROOT / CONFIG).read_text())
    pos = json.loads((ROOT / POSITIONS).read_text())
    if pos["config_hash"] != digest(cfg) or pos["positions_hash"] != digest({k: v for k, v in pos.items() if k != "positions_hash"}):
        raise ValueError("positions do not match the frozen config")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCES],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("commit P2-RJ sources before preparing a manifest")
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)
    manifest = {"schema": SCHEMA, "git_commit": rev, "config": CONFIG, "config_hash": digest(cfg),
                "positions": POSITIONS, "positions_hash": pos["positions_hash"],
                "p2r_run_manifest_hash": pos["p2r_run_manifest_hash"], "partition_hash": partition["hash"],
                "sources": {p: file_hash(ROOT / p) for p in SOURCES}}
    manifest["manifest_hash"] = digest(manifest)
    if args.dry_run:
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "git_commit": rev}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2-RJ manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "git_commit": rev}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("positions")
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    m.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"positions": cmd_positions, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
