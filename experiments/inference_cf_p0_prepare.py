#!/usr/bin/env python
"""CPU-only frozen development panel and run manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from csasr.inference_cf.core import VERSION, atomic_json, condition_tokens, digest, file_hash, permutation

ROLE = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet")
POI = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/baselines/generation_001/poi_D-dev-select.parquet")
BASE = ROOT / "results/dg04/results/B0.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
PLAN = Path("/home/tungnx/cs-asr-steer-inf/INFERENCE_STEERING_IMPLEMENTATION_PLAN.md")
SEED = 240924


def build_panel():
    import pyarrow.parquet as pq
    from csasr.data.normalize import normalize_and_segment
    roles = {r["utterance_id"]: r for r in pq.read_table(ROLE).to_pylist()}
    baselines = json.loads(BASE.read_text())["texts"]
    pois = pq.read_table(POI).to_pylist()
    pool = {"english_wrong": [], "english_correct": [], "mandarin": []}
    for p in pois:
        uid = p["utterance_id"]
        if uid not in baselines or uid not in roles:
            continue
        r = roles[uid]
        norm, units = normalize_and_segment(r["transcript_raw"])
        idx = int(p["poi_index"])
        if idx < 0 or idx >= len(units) or units[idx].kind != "latin":
            continue
        kind = "english_correct" if p["correct"] else "english_wrong"
        pool[kind].append((uid, idx, units[idx].char_start / max(1, len(norm)), p["surface"]))
    for uid in sorted(baselines):
        if uid not in roles:
            continue
        norm, units = normalize_and_segment(roles[uid]["transcript_raw"])
        mids = [i for i, u in enumerate(units) if u.kind == "han"]
        if mids:
            idx = mids[len(mids) // 2]
            pool["mandarin"].append((uid, idx, units[idx].char_start / max(1, len(norm)), units[idx].surface))
    selected, used = [], set()
    counts = {}
    for kind in ("english_wrong", "english_correct", "mandarin"):
        choices = sorted(pool[kind], key=lambda x: (hashlib.sha256(f"{SEED}:{kind}:{x[0]}:{x[1]}".encode()).hexdigest(), x[0], x[1]))
        counts[kind] = {"available_positions": len(choices), "available_utterances": len(set(x[0] for x in choices))}
        for uid, idx, fraction, surface in choices:
            if uid in used:
                continue
            r = roles[uid]
            selected.append({"identity": f"{kind}:{uid}:{idx}", "stratum": kind,
                             "utterance_id": uid, "audio_path": r["audio_path"],
                             "audio_sha256": r["audio_sha256"], "duration_sec": r["duration_sec"],
                             "position_fraction": fraction, "selection_unit_index": idx,
                             "selection_surface": surface, "frozen_baseline_text": baselines[uid],
                             "role": "D-dev-select"})
            used.add(uid)
            if sum(x["stratum"] == kind for x in selected) == 20:
                break
        counts[kind]["selected"] = sum(x["stratum"] == kind for x in selected)
    return {"seed": SEED, "ordering": "sha256(seed:kind:utterance_id:unit_index), unique audio, stratum priority wrong/correct/mandarin",
            "counts": counts, "rows": selected}


def main():
    from transformers import WhisperProcessor
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/inference_cf/p0")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = ROOT / args.out
    panel = build_panel()
    conditions = condition_tokens(WhisperProcessor.from_pretrained(MODEL, local_files_only=True))
    perm = permutation([r["identity"] for r in panel["rows"]])
    sources = {str(p): file_hash(p) for p in (PLAN, ROLE, POI, BASE,
                                              ROOT / "configs/model/whisper_large_v3.yaml",
                                              ROOT / "results/dg02_real_acceptance.json",
                                              ROOT / "docs/inference_cf/P0_FEASIBILITY_SPEC.md",
                                              ROOT / "experiments/inference_cf_p0.py",
                                              ROOT / "experiments/inference_cf_p0_prepare.py",
                                              ROOT / "slurm/inference_cf_p0_retry.sbatch",
                                              ROOT / "src/csasr/inference_cf/core.py")}
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": VERSION, "git_commit": rev, "sources": sources, "panel_hash": digest(panel),
                "permutation_hash": digest(perm), "model_id": "openai/whisper-large-v3",
                "local_model": str(MODEL), "model_revision": file_hash(MODEL / "model.safetensors"),
                "tokenizer_revision": file_hash(MODEL / "tokenizer.json"), "precision": "bfloat16",
                "conditions": {"names": {"c0": "ordinary baseline Mandarin transcribe",
                                          "cM": "explicit Mandarin matrix transcribe",
                                          "cE": "explicit English embedded transcribe"},
                               "prompt_tokens": conditions,
                               "c0_equals_cM_by_tokens": conditions["c0"] == conditions["cM"]},
                "layer": 24, "site": "decoder.post_cross_attention.pre_ffn",
                "decode": {"task": "transcribe", "language": "zh", "do_sample": False,
                           "num_beams": 1, "temperature": 0.0, "max_new_tokens": 200,
                           "condition_on_prev_tokens": False}, "candidate_k": 1,
                "output_path": str(out), "role": "D-dev-select"}
    manifest["manifest_hash"] = digest(manifest)
    report = {"manifest": manifest, "counts": panel["counts"], "output_files": [str(out / x) for x in ("manifest.json", "panel.json", "audio_permutation.json", "rows", "summary.json")]}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "panel.json", panel)
    atomic_json(out / "audio_permutation.json", perm)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "counts": panel["counts"]}))


if __name__ == "__main__":
    main()
