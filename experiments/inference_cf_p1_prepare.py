#!/usr/bin/env python
"""CPU-only P1 manifest: frozen R2 gate, site, doses and a deterministic 10-utterance panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csasr.inference_cf.core import atomic_json, condition_tokens, digest, file_hash
from csasr.inference_cf.core_p1 import VERSION_P1
from csasr.inference_cf.core_r2 import tokenizer_partition

MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
CONFIG = ROOT / "configs/inference_cf/p1_causal_acceptance.json"
R2 = ROOT / "results/inference_cf/p0_r2"


def build_panel(config: dict) -> dict:
    r2_manifest = json.loads((R2 / "manifest.json").read_text())
    if r2_manifest["manifest_hash"] != config["r2_manifest_hash"]:
        raise ValueError("R2 manifest differs from the frozen P1 entry contract")
    summary = json.loads((R2 / "summary.json").read_text())
    if summary["status"] != config["r2_verdict"]:
        raise ValueError("R2 verdict differs from the frozen P1 entry contract")
    rows = json.loads((R2 / "inference_panel.json").read_text())["rows"]
    picked = [rows[i] for i in config["population_indices"]]
    return {"schema": VERSION_P1, "role": config["role"], "rule": config["population_rule"],
            "rows": [{k: r[k] for k in ("utterance_id", "audio_path", "audio_sha256", "duration_sec")}
                     for r in picked]}


def main() -> None:
    from transformers import WhisperProcessor, __version__ as transformers_version
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    panel = build_panel(config)
    processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
    partition = tokenizer_partition(processor.tokenizer)
    if partition["hash"] != config["tokenizer_partition_hash"]:
        raise ValueError("tokenizer partition changed")
    cond = condition_tokens(processor)
    native = json.loads((MODEL / "generation_config.json").read_text())["lang_to_id"]
    conditions = {"cB": cond["c0"], "cM": cond["cM"], "cE": cond["cE"],
                  "language_token_ids": [native["<|en|>"], native["<|zh|>"]]}
    if conditions["cB"] != conditions["cM"] or conditions["cB"] != [50258, 50260, 50360, 50364] \
            or conditions["cE"] != [50258, 50259, 50360, 50364]:
        raise ValueError("forced prompts changed")
    sources = [ROOT / x for x in (
        "docs/inference_cf/P1_CAUSAL_ACCEPTANCE_SPEC.md",
        "docs/inference_cf/P1_PRE_RUN_AUDIT.md",
        "docs/inference_cf/P0_R2_REPORT.md",
        "docs/inference_cf/P0_R2_POST_RUN_AUDIT.md",
        "INFERENCE_STEERING_IMPLEMENTATION_PLAN.md",
        "configs/inference_cf/p1_causal_acceptance.json",
        "configs/model/whisper_large_v3.yaml",
        "src/csasr/inference_cf/core.py", "src/csasr/inference_cf/core_r2.py",
        "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py",
        "src/csasr/models/hooks.py", "src/csasr/models/whisper.py",
        "experiments/inference_cf_p0_r2.py", "experiments/inference_cf_p1.py",
        "experiments/inference_cf_p1_prepare.py", "experiments/inference_cf_p1_accept.py",
        "slurm/inference_cf_p1.sbatch",
        "results/inference_cf/p0_r2/manifest.json", "results/inference_cf/p0_r2/summary.json")]
    sources.append(MODEL / "generation_config.json")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit P1 spec and implementation before preparing a run manifest")
    manifest = {"schema": VERSION_P1, "status": "prepared_unrun", "git_commit": rev,
                "git_branch": branch, "git_dirty": False,
                "sources": {str(p): file_hash(p) for p in sources},
                "config_hash": digest(config), "resolved_config": config,
                "panel_hash": digest(panel), "model_id": "openai/whisper-large-v3",
                "model_revision": file_hash(MODEL / "model.safetensors"),
                "tokenizer_revision": file_hash(MODEL / "tokenizer.json"),
                "partition_hash": partition["hash"], "conditions": conditions,
                "r2_manifest_hash": config["r2_manifest_hash"],
                "selected_gate": config["selected_gate"],
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "transformers": transformers_version, "hostname": platform.node()}}
    manifest["manifest_hash"] = digest(manifest)
    if args.dry_run:
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "panel_hash": manifest["panel_hash"],
                          "git_commit": rev, "utterances": len(panel["rows"]),
                          "alphas": config["alphas"]}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P1 manifest already exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "panel.json", panel)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "out": str(out)}))


if __name__ == "__main__":
    main()
