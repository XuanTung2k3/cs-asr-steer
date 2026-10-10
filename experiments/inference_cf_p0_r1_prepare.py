#!/usr/bin/env python
"""CPU-only P0-R1 manifest. Reuses the EXACT frozen P0 panel and permutation.

The P0-R1 diagnostic changes only K=1 -> K=3. The development panel, strata, audio
permutation, conditions, site, model and data roles are identical to P0. This script
therefore copies the byte-identical panel/permutation from the corrected P0 run
(`results/inference_cf/p0_retry`) and only re-versions the manifest (schema/candidate_k).
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from csasr.inference_cf.core_r1 import VERSION_R1, K, atomic_json, condition_tokens, digest, file_hash

SOURCE_P0 = ROOT / "results/inference_cf/p0_retry"
ROLE = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet")
POI = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/baselines/generation_001/poi_D-dev-select.parquet")
BASE = ROOT / "results/dg04/results/B0.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
PLAN = Path("/home/tungnx/cs-asr-steer-inf/INFERENCE_STEERING_IMPLEMENTATION_PLAN.md")


def main():
    from transformers import WhisperProcessor
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/inference_cf/p0_r1")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = ROOT / args.out

    # Reuse the EXACT frozen P0 panel and permutation (byte identical), verified against
    # the corrected P0 manifest hashes so the P0-R1 panel is provably the same panel.
    src_manifest = json.loads((SOURCE_P0 / "manifest.json").read_text())
    panel = json.loads((SOURCE_P0 / "panel.json").read_text())
    perm = json.loads((SOURCE_P0 / "audio_permutation.json").read_text())
    assert digest(panel) == src_manifest["panel_hash"], "panel drifted from P0"
    assert digest(perm) == src_manifest["permutation_hash"], "permutation drifted from P0"

    conditions = condition_tokens(WhisperProcessor.from_pretrained(MODEL, local_files_only=True))
    assert conditions == src_manifest["conditions"]["prompt_tokens"], "conditions drifted from P0"

    sources = {str(p): file_hash(p) for p in (PLAN, ROLE, POI, BASE,
                                              ROOT / "configs/model/whisper_large_v3.yaml",
                                              ROOT / "results/dg02_real_acceptance.json",
                                              ROOT / "docs/inference_cf/P0_FEASIBILITY_SPEC.md",
                                              ROOT / "docs/inference_cf/P0_R1_EVIDENCE_SPEC.md",
                                              ROOT / "experiments/inference_cf_p0_r1.py",
                                              ROOT / "experiments/inference_cf_p0_r1_prepare.py",
                                              ROOT / "slurm/inference_cf_p0_r1.sbatch",
                                              ROOT / "src/csasr/inference_cf/core.py",
                                              ROOT / "src/csasr/inference_cf/core_r1.py")}
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": VERSION_R1, "supersedes_manifest": src_manifest["manifest_hash"],
                "git_commit": rev, "sources": sources, "panel_hash": digest(panel),
                "permutation_hash": digest(perm), "model_id": "openai/whisper-large-v3",
                "local_model": str(MODEL), "model_revision": file_hash(MODEL / "model.safetensors"),
                "tokenizer_revision": file_hash(MODEL / "tokenizer.json"), "precision": "bfloat16",
                "conditions": src_manifest["conditions"],
                "layer": 24, "site": "decoder.post_cross_attention.pre_ffn",
                "decode": src_manifest["decode"], "candidate_k": K,
                "candidate_scoring": "K=3 greedy continuation, token-average log prob, EOS truncated",
                "output_path": str(out), "role": "D-dev-select"}
    manifest["manifest_hash"] = digest(manifest)
    report = {"manifest": manifest, "counts": panel["counts"],
              "reused_panel_from": str(SOURCE_P0),
              "output_files": [str(out / x) for x in ("manifest.json", "panel.json", "audio_permutation.json", "rows", "summary.json")]}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "panel.json", panel)
    atomic_json(out / "audio_permutation.json", perm)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"],
                      "supersedes": src_manifest["manifest_hash"],
                      "panel_hash": manifest["panel_hash"], "counts": panel["counts"]}))


if __name__ == "__main__":
    main()
