#!/usr/bin/env python
"""CPU-only immutable R2 manifest and dense, reference-free inference panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csasr.inference_cf.core import atomic_json, condition_tokens, digest, file_hash
from csasr.inference_cf.core_r2 import VERSION_R2, tokenizer_partition

DATA = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3")
ROLE = DATA / "manifests/roles/role_D-dev-select.parquet"
CTC = DATA / "candidate_generations/generation_001/alignments/candidates_existing_ctc.parquet"
AUTO = DATA / "baselines/generation_001/baselines/B0_AUTO/D-dev-select.parquet"
BASE = ROOT / "results/dg04/results/B0.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
CONFIG = ROOT / "configs/inference_cf/p0_r2_repairability.json"


def build_panels() -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    baseline_ids = sorted(json.loads(BASE.read_text())["texts"])
    roles = {r["utterance_id"]: r for r in pq.read_table(ROLE).to_pylist()}
    if len(baseline_ids) != 300 or any(uid not in roles for uid in baseline_ids):
        raise ValueError("frozen 300-item D-dev-select population mismatch")
    inference, evaluation = [], []
    for i, uid in enumerate(baseline_ids):
        role = roles[uid]
        if role["role"] != "D-dev-select":
            raise ValueError("role mismatch")
        inference.append({"utterance_id": uid, "audio_path": role["audio_path"],
                          "audio_sha256": role["audio_sha256"],
                          "duration_sec": role["duration_sec"],
                          "donor_utterance_id": baseline_ids[(i + 1) % len(baseline_ids)]})
        evaluation.append({"utterance_id": uid, "dialogue_id": role["dialogue_id"],
                           "reference": role["transcript_raw"],
                           "duration_sec": role["duration_sec"]})
    if len({r["dialogue_id"] for r in evaluation}) != 20:
        raise ValueError("expected 20 development dialogues")
    return ({"schema": VERSION_R2, "role": "D-dev-select", "ordering": "utterance_id",
             "rows": inference},
            {"schema": VERSION_R2, "role": "D-dev-select", "rows": evaluation})


def make_manifest(inference: dict, evaluation: dict) -> dict:
    from transformers import WhisperProcessor, __version__ as transformers_version
    import torch

    config = json.loads(CONFIG.read_text())
    if config["schema"] != VERSION_R2:
        raise ValueError("config version mismatch")
    processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
    condition = condition_tokens(processor)
    partition = tokenizer_partition(processor.tokenizer)
    if partition["hash"] != config["tokenizer_partition_hash"]:
        raise ValueError("tokenizer partition differs from frozen R2 config")
    native = json.loads((MODEL / "generation_config.json").read_text())["lang_to_id"]
    wrong = config["wrong_language_diagnostic"]
    if wrong in ("en", "zh"):
        raise ValueError("wrong-language diagnostic must differ from M and E")
    c_x = list(condition["cM"])
    c_x[1] = native[f"<|{wrong}|>"]
    prompts = {"cB": condition["c0"], "cM": condition["cM"], "cE": condition["cE"],
               "language_token_ids": [native["<|en|>"], native["<|zh|>"]],
               "cX": c_x, "wrong_language": wrong}
    if prompts["cB"] != prompts["cM"] or prompts["cB"] != [50258, 50260, 50360, 50364] or prompts["cE"] != [50258, 50259, 50360, 50364]:
        raise ValueError("forced prompt changed")
    sources = [ROOT / x for x in (
        "docs/inference_cf/P0_R2_REPAIRABILITY_SPEC.md",
        "docs/inference_cf/P0_R2_FORMULA_RECONCILIATION.md",
        "docs/inference_cf/P0_R2_DESIGN_AUDIT.md",
        "docs/inference_cf/P0_R2_REPAIRABILITY_AUDIT.md",
        "INFERENCE_STEERING_IMPLEMENTATION_PLAN.md",
        "configs/inference_cf/p0_r2_repairability.json",
        "configs/model/whisper_large_v3.yaml",
        "src/csasr/inference_cf/core.py",
        "src/csasr/inference_cf/core_r2.py",
        "src/csasr/models/whisper.py",
        "src/csasr/utils/hashing.py",
        "src/csasr/data/normalize.py",
        "src/csasr/evaluation/mer.py",
        "src/csasr/evaluation/pier.py",
        "experiments/inference_cf_p0_r2.py",
        "experiments/inference_cf_p0_r2_prepare.py",
        "experiments/inference_cf_p0_r2_evaluate.py",
        "slurm/inference_cf_p0_r2.sbatch")]
    sources += [ROLE, CTC, AUTO, BASE, MODEL / "generation_config.json"]
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("commit R2 spec and implementation before preparing a run manifest")
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": VERSION_R2, "status": "prepared_unrun", "git_commit": rev, "git_branch": branch,
                "git_dirty": False, "sources": {str(p): file_hash(p) for p in sources},
                "config_hash": digest(config), "resolved_config": config,
                "panel_hash": digest(inference), "evaluation_panel_hash": digest(evaluation),
                "model_id": "openai/whisper-large-v3",
                "local_model": str(MODEL),
                "model_revision": file_hash(MODEL / "model.safetensors"),
                "tokenizer_revision": file_hash(MODEL / "tokenizer.json"),
                "partition_hash": partition["hash"],
                "partition_counts": {k: len(partition[k]) for k in ("matrix_ids", "embedded_ids", "ambiguous_ids")},
                "conditions": prompts, "decode": config["decode"],
                "seed": config["seed"], "role": "D-dev-select",
                "environment": {"python": platform.python_version(),
                                "torch": torch.__version__,
                                "transformers": transformers_version,
                                "hostname": platform.node()},
                "comparator": {"name": "B0_AUTO", "path": str(AUTO), "hash": file_hash(AUTO)},
                "ctc_evaluator_only": str(CTC)}
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p0_r2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inference, evaluation = build_panels()
    manifest = make_manifest(inference, evaluation)
    if args.dry_run:
        print(json.dumps({"manifest_hash": manifest["manifest_hash"],
                          "panel_hash": manifest["panel_hash"],
                          "partition_counts": manifest["partition_counts"],
                          "utterances": len(inference["rows"])}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("R2 manifest already exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "inference_panel.json", inference)
    atomic_json(out / "evaluation_panel.json", evaluation)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "out": str(out)}))


if __name__ == "__main__":
    main()
