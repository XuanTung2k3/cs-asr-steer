#!/usr/bin/env python
"""CPU-only P2 manifest for one stage: frozen configs, the 300-utterance panel, source hashes."""
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
from experiments.inference_cf_p2 import SCHEMA, configs_for_stage

CONFIG = ROOT / "configs/inference_cf/p2_compact_development.json"
SELECTION = ROOT / "results/inference_cf/p2_selection.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")


def main() -> None:
    from transformers import WhisperProcessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "A_diag", "B", "C"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    if r2m["manifest_hash"] != config["r2_manifest_hash"]:
        raise ValueError("R2 manifest mismatch")
    panel = json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())
    if digest(panel) != r2m["panel_hash"] or len(panel["rows"]) != 300:
        raise ValueError("frozen 300-utterance R2 panel mismatch")
    selection = None
    if args.stage != "A":
        selection = json.loads(SELECTION.read_text())
        if args.stage in ("B", "C") and selection["p2_a_verdict"] != "P2_A_VALID_CONFIG_EXISTS":
            raise ValueError("P2-B/C require a valid P2-A configuration")
        if args.stage == "A_diag" and selection["p2_a_verdict"] != "P2_A_NO_VALID_CONFIG":
            raise ValueError("A_diag only runs after P2_A_NO_VALID_CONFIG")
    if args.stage == "A" and args.layer not in config["stages"]["A"]["layers"]:
        raise ValueError("P2-A layer must be one of the frozen layers")
    configs = configs_for_stage(args.stage, config, layer=args.layer, selection=selection)
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)
    sources = [ROOT / x for x in ("docs/inference_cf/P2_COMPACT_DEVELOPMENT_SPEC.md",
                                  "docs/inference_cf/PRE_P2_AUDIT.md",
                                  "configs/inference_cf/p2_compact_development.json",
                                  "experiments/inference_cf_cached.py", "experiments/inference_cf_p2.py",
                                  "experiments/inference_cf_p2_prepare.py",
                                  "experiments/inference_cf_p2_evaluate.py",
                                  "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py",
                                  "src/csasr/lss/sites.py", "src/csasr/models/hooks.py")]
    if selection is not None:
        sources.append(SELECTION)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("commit before preparing a P2 manifest")
    manifest = {"schema": SCHEMA, "stage": args.stage, "layer": args.layer, "git_commit": rev,
                "configs": configs, "baselines": args.stage == "A", "config_hash": digest(config),
                "panel_hash": digest(panel), "partition_hash": partition["hash"],
                "sources": {str(p): file_hash(p) for p in sources},
                "r2_manifest_hash": r2m["manifest_hash"]}
    manifest["manifest_hash"] = digest(manifest)
    if args.dry_run:
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "stage": args.stage,
                          "configs": [c["name"] for c in configs], "git_commit": rev}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2 manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "panel.json", panel)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "configs": [c["name"] for c in configs]}))


if __name__ == "__main__":
    main()
