#!/usr/bin/env python
"""CPU-only P2-R manifest: frozen config, frozen population, source hashes, Git state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_r2 import tokenizer_partition
from experiments.inference_cf_p2r import SCHEMA

CONFIG = "configs/inference_cf/p2_r_mechanism_diagnosis.json"
POPULATION = "results/inference_cf/p2r/population.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
SOURCES = ("docs/inference_cf/P2_R_MECHANISM_DIAGNOSIS_SPEC.md", "docs/inference_cf/P2_R_PRE_RUN_AUDIT.md",
           CONFIG, POPULATION, "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2r_population.py",
           "experiments/inference_cf_p2r_prepare.py", "experiments/inference_cf_p2r_analyze.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_r2.py",
           "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py",
           "slurm/inference_cf_p2r.sbatch")


def main() -> None:
    from transformers import WhisperProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config = json.loads((ROOT / CONFIG).read_text())
    pop = json.loads((ROOT / POPULATION).read_text())
    if pop["config_hash"] != digest(config) or pop["population_hash"] != digest({k: v for k, v in pop.items() if k != "population_hash"}):
        raise ValueError("population does not match the frozen config")
    if not all(pop["estimable"].values()):
        raise ValueError("population below the frozen minimum; report limited estimability instead of running")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", *SOURCES],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("commit P2-R sources before preparing a manifest")
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)
    manifest = {"schema": SCHEMA, "git_commit": rev, "config": CONFIG, "config_hash": digest(config),
                "population": POPULATION, "population_hash": pop["population_hash"],
                "partition_hash": partition["hash"],
                "sources": {str(ROOT / p): file_hash(ROOT / p) for p in SOURCES}}
    manifest["manifest_hash"] = digest(manifest)
    if args.dry_run:
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "git_commit": rev}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2-R manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "git_commit": rev}))


if __name__ == "__main__":
    main()
