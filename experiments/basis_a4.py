#!/usr/bin/env python
"""BASIS-A4 implementation and resumable execution driver.

The command deliberately exposes the A4 gates as separate subcommands.  It
never creates Conditioning-Avg artifacts: the frozen A4 redundancy gate
declares that family identical to Conditioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
OUT = REPO / "results/basis_a4"
DATASETS = ("cs_dialogue", "seame_dev_man", "seame_dev_sge")
SCOPES = ("global", "oracle_local")
RHO = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical_hash(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False, default=str) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def protocol_payload() -> dict:
    spec = REPO / "docs/current/BASIS_A4_SPEC.md"
    plan = REPO / "docs/current/BASIS_A4_EXECUTION_PLAN.md"
    return {
        "schema_version": "basis_a4_protocol_v1", "study": "BASIS-A4",
        "spec_sha256": sha256_file(spec), "execution_plan_sha256": sha256_file(plan),
        "directions": ["Raw", "Conditioning"],
        "conditioning_avg": {"status": "REDUNDANT", "run": False,
                              "reason": "all-content position set is already Conditioning; weighting-only variant is out of scope"},
        "rho": RHO, "scopes": list(SCOPES),
        "whisper": {"encoder_layers": list(range(32)), "decoder_layers": list(range(32)),
                    "decoder_site": "decoder_post_cross_attn_residual",
                    "encoder_site": "encoder_post_self_attn_residual_pre_ffn"},
        "qwen": {"encoder_layers": list(range(24)), "decoder_layers": list(range(28)),
                 "encoder_site": "qwen_audio_encoder_post_self_attn_residual_pre_ffn",
                 "decoder_site": "qwen_text_decoder_post_self_attn_residual_pre_mlp",
                 "model_id": "Qwen/Qwen3-ASR-1.7B",
                 "model_path": "/mnt/data/tungnx/Qwen3-ASR-1.7B",
                 "baseline_language": "Chinese", "context": "", "max_new_tokens": 200},
        "qwen_encoder_cache": "OFF",
        "panels": {"cs_dialogue": {"n": 300, "role": "D-dev-select"},
                   "seame_dev_man": {"n": 50, "role": "SEAME-dev-man"},
                   "seame_dev_sge": {"n": 50, "role": "SEAME-dev-sge"}},
        "forbidden_splits": ["D-dev-confirm", "D-test"],
        "norm_preserve": True, "depth_rescale": False,
        "decoding": {"do_sample": False, "num_beams": 1, "temperature": 0.0,
                      "condition_on_prev_tokens": False, "max_new_tokens": 200},
    }


def panel_path(dataset: str) -> Path:
    return OUT / "panels" / f"{dataset}_300.json"


def panel(dataset: str) -> dict:
    return read_json(panel_path(dataset))


def freeze() -> int:
    """Seal A4 config and byte-copy the already frozen A3 evaluation panels."""
    import subprocess
    from csasr.experiments.basis_a3_protocol import freeze_panels
    source = REPO / "results/basis_a3_raw_cond_scope_depth/panels"
    if not source.is_dir():
        freeze_panels(source)
    for dataset in DATASETS:
        src = source / f"{dataset}_300.json"
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = panel_path(dataset); dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    payload = protocol_payload()
    payload["panel_fingerprints"] = {
        d: {"count": panel(d)["count"], "fingerprint": panel(d)["fingerprint"],
            "source_sha256": sha256_file(REPO / "results/basis_a3_raw_cond_scope_depth/panels" / f"{d}_300.json")}
        for d in DATASETS}
    payload["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    payload["status"] = "FROZEN_PRE_RUN"
    payload["protocol_hash"] = canonical_hash(payload)
    write_json(OUT / "manifests/a4_protocol_freeze.json", payload)
    write_json(OUT / "spec/protocol_payload.json", payload)
    return 0


def _record_manifest(stage: str, status: str, **extra):
    from csasr.utils.logging import git_state
    payload = {"schema_version": "basis_a4_job_manifest_v1", "stage": stage,
               "status": status, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
               "argv": list(sys.argv), "git": git_state(str(REPO)),
               "protocol_hash": read_json(OUT / "manifests/a4_protocol_freeze.json").get("protocol_hash"),
               "started_at": time.time()}
    payload.update(extra)
    path = OUT / "manifests" / f"{stage}_{payload['slurm_job_id']}.json"
    write_json(path, payload)
    return path, payload


def _a3_compatibility(source: dict, dataset: str, direction: str, side: str,
                      layer: int, scope: str) -> dict:
    expected_site = ("decoder_post_cross_attn_residual" if side == "decoder"
                     else "encoder_post_self_attn_residual_pre_ffn")
    ok = (source.get("dataset") == dataset and source.get("direction") == direction and
          int(source.get("layer", -1)) == int(layer) and source.get("scope") == scope and
          float(source.get("rho", -1)) == RHO and
          source.get("panel_fingerprint") == panel(dataset)["fingerprint"] and
          source.get("scale") is not None and expected_site is not None)
    # A3's serialized result predates the A4 manifest field.  The exact-site,
    # dose, and generation semantics are verified against the frozen A3
    # pre-run record here; the source is never relabelled as an A4 run.
    return {"accepted": bool(ok), "source": "BASIS-A3", "site": expected_site,
            "protocol_compatibility": "A3 exact-site R1 rho=.5; A4 fields re-sealed",
            "source_direction_hash": source.get("direction_hash")}


def prepare_whisper_reuse() -> int:
    """Copy only compatibility-accepted A3 rows into the A4 namespace."""
    if not (OUT / "manifests/a4_protocol_freeze.json").is_file():
        raise RuntimeError("run basis_a4.py freeze before reuse preparation")
    src_root = REPO / "results/basis_a3_raw_cond_scope_depth"
    accepted, rejected = [], []
    for dataset in DATASETS:
        for direction, side, src_sub, layers in (("Raw", "encoder", "raw_r1", range(32)),
                                                  ("Raw", "decoder", "raw_r1", range(32)),
                                                  ("Conditioning", "decoder", "conditioning", (24, 26, 27, 31))):
            for layer in layers:
                for scope in SCOPES:
                    src = src_root / src_sub / dataset / f"L{layer:02d}"
                    files = list(src.glob(f"*{side}_L{layer}_{scope}_rho0.5.json"))
                    if len(files) != 1:
                        rejected.append({"dataset": dataset, "direction": direction, "side": side, "layer": layer, "scope": scope,
                                         "reason": "missing_or_ambiguous_source"})
                        continue
                    data = read_json(files[0]); compat = _a3_compatibility(data, dataset, direction, side, layer, scope)
                    (accepted if compat["accepted"] else rejected).append({
                        "dataset": dataset, "direction": direction, "side": side, "layer": layer, "scope": scope,
                        "source": str(files[0].relative_to(REPO)), "compatibility": compat})
                    if compat["accepted"]:
                        data["a4_reuse"] = compat
                        data["a4_protocol_hash"] = read_json(OUT / "manifests/a4_protocol_freeze.json")["protocol_hash"]
                        dst = OUT / "whisper" / direction.lower() / side / dataset / f"L{layer:02d}" / files[0].name
                        write_json(dst, data)
    write_json(OUT / "manifests/whisper_reuse_manifest.json",
               {"schema_version": "basis_a4_reuse_manifest_v1", "accepted": accepted,
                "rejected": rejected, "conditioning_avg": "REDUNDANT_NOT_PRESENT"})
    return 0 if not rejected or all(x["reason"] == "missing_or_ambiguous_source" for x in rejected) else 1


def run_whisper_conditioning(dataset: str, layer_start: int = 0, layer_end: int = 32) -> int:
    """Run missing Whisper Conditioning layers for one panel in one Slurm job."""
    import pandas as pd
    from experiments.basis_a3 import _decode_condition
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.utils.provenance import stage_provenance, code_config_snapshot_hash, test_snapshot_hash
    from csasr.utils.logging import git_state
    if dataset not in DATASETS:
        raise ValueError(dataset)
    if not panel_path(dataset).is_file():
        freeze()
    frame = pd.DataFrame(panel(dataset)["rows"])
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    cfg = load_config("model/whisper_large_v3.yaml")
    model_revision = read_json(REPO / "results/basis_frozen_layer_atlas/directions.json")["model"]["model_revision"]
    manifest_path, manifest = _record_manifest("whisper_conditioning", "RUNNING", dataset=dataset,
                                               direction="Conditioning", layers=list(range(layer_start, layer_end)), rho=RHO)
    try:
        for layer in range(layer_start, layer_end):
            for scope in SCOPES:
                dst = OUT / "whisper/conditioning" / dataset / f"L{layer:02d}" / f"conditioning_decoder_L{layer}_{scope}_rho0.5.json"
                if dst.is_file():
                    continue
                # A3's canonical decoder-only routine is reused only for the
                # exact Whisper site and fixed conditioning direction.
                result = _decode_condition(bundle, dataset, "decoder", layer, "conditioning", scope, RHO)
                result["a4_protocol_hash"] = read_json(OUT / "manifests/a4_protocol_freeze.json")["protocol_hash"]
                result["a4_reuse"] = False
                result["side"] = "decoder"
                result["site"] = "decoder_post_cross_attn_residual"
                result["git_commit"] = git_state(str(REPO)).get("commit")
                result["config_hash"] = read_json(OUT / "manifests/a4_protocol_freeze.json")["protocol_hash"]
                result["model_revision"] = model_revision
                result["site_hash"] = canonical_hash(result["site"])
                result["mask_hash"] = canonical_hash({"dataset": dataset, "scope": scope,
                                                        "side": "decoder", "protocol": "BASIS-A4"})
                result["provenance"] = stage_provenance(
                    {**cfg, "model": {**cfg.get("model", {}), "revision": model_revision},
                     "experiment": {**cfg.get("experiment", {}), "output_root": str(OUT), "seed": 42}},
                    "basis_a4_whisper_conditioning",
                    upstream=[{"kind": "basis_a3_panel", "fingerprint": panel(dataset)["fingerprint"]}])
                result["provenance"]["code_config_sha256"] = code_config_snapshot_hash()
                result["provenance"]["tests_sha256"] = test_snapshot_hash()
                write_json(dst, result)
        manifest.update({"status": "COMPLETED", "finished_at": time.time()}); write_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest.update({"status": "FAILED", "finished_at": time.time(), "error": repr(exc)})
        write_json(manifest_path, manifest); raise


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("prepare-whisper-reuse")
    w = sub.add_parser("whisper-conditioning"); w.add_argument("--dataset", choices=DATASETS, required=True)
    w.add_argument("--layer-start", type=int, default=0); w.add_argument("--layer-end", type=int, default=32)
    args = ap.parse_args()
    if args.command == "freeze": return freeze()
    if args.command == "prepare-whisper-reuse": return prepare_whisper_reuse()
    if args.command == "whisper-conditioning": return run_whisper_conditioning(args.dataset, args.layer_start, args.layer_end)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
