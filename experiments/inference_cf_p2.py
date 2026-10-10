#!/usr/bin/env python
"""P2 compact development runner (cached steering path; frozen detector g = E * R_B).

Configurations come only from the frozen config (and, for P2-B/C/diagnostics, the committed
P2-A selection). No reference, label or evaluator timing is read here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from csasr.inference_cf.core import atomic_json, digest, file_hash, validated_row
from csasr.inference_cf.core_r2 import tokenizer_partition
from csasr.lss.sites import assert_no_site_hooks, num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
from csasr.utils.config import load_config
import experiments.inference_cf_cached as cached

SCHEMA = "p2_compact_development_v1_1"
CONDITIONS = {"cB": [50258, 50260, 50360, 50364], "cE": [50258, 50259, 50360, 50364],
              "language_token_ids": [50259, 50260]}


def config_name(c: dict) -> str:
    sign = "neg" if c.get("direction_sign", 1.0) < 0 else "pos"
    return f"L{c['layer']}_a{c['alpha']:.6g}_{c['gate']}_{c['dose']}_{sign}"


def configs_for_stage(stage: str, config: dict, *, layer: int | None = None,
                      selection: dict | None = None) -> list[dict]:
    """Frozen configuration list for a stage; nothing outside the config is ever produced."""
    spec = config["stages"]
    out = []
    if stage == "A":
        for a in spec["A"]["alphas"]:
            out.append({"layer": int(layer), "alpha": float(a), "gate": "ER", "dose": "id", "direction_sign": 1.0})
    elif stage in ("B", "A_diag"):
        base = selection["diagnostic_config" if stage == "A_diag" else "selected"]
        for ctl in spec[stage]["controls"]:
            alpha = selection["alpha_c"] if ctl.get("alpha") == "alpha_c" else base["alpha"]
            out.append({"layer": int(base["layer"]), "alpha": float(alpha), "gate": ctl["gate"],
                        "dose": "id", "direction_sign": float(ctl.get("direction_sign", 1.0)),
                        "label": "energy_matched_constant" if ctl.get("alpha") == "alpha_c" else None})
    elif stage == "C":
        for a in spec["C"]["alphas"]:
            out.append({"layer": int(selection["selected"]["layer"]), "alpha": float(a), "gate": "ER",
                        "dose": "sqrt", "direction_sign": 1.0})
    else:
        raise ValueError(f"unknown stage {stage!r}")
    for c in out:
        c["name"] = config_name(c)
    return out


def compact_steps(steps: list[dict]) -> list[dict]:
    keep = []
    for s in steps:
        a = s.get("hook_audit") or {}
        keep.append({"t": s["t"], "fb": s["fallback_reason"], "g": s["g"], "dose": s["dose"],
                     "E": (s["local_support"] or {}).get("E"), "R_B": (s["baseline"] or {}).get("R"),
                     "dir": s["direction_status"], "edit": s["edit_applied"],
                     "edit_norm": a.get("edit_norm") if s["edit_applied"] else 0.0,
                     "pre_norm": a.get("pre_norm") if s["edit_applied"] else None,
                     "next": s["steered_next"], "unsteered_next": s["unsteered_next"]})
    return keep


def run_utterance(bundle, *, waveform, encoded, inputs, configs, partition, null_probs,
                  language_ids, nfp, uid, baselines: bool, matched_layers=(), max_new_tokens: int = 200) -> dict:
    lid_cache: dict = {}
    result = {"systems": {}}
    # v1.1 matched baseline: the alpha=0 run of the identical cached_decode path at each layer
    # (CE1: bitwise equal to its own B branch). All deltas are measured against it.
    for layer in matched_layers:
        r = cached.cached_decode(bundle, waveform=waveform, encoded=encoded, conditions=CONDITIONS,
                                 partition=partition, null_probs=null_probs, language_ids=language_ids,
                                 layer=int(layer), alpha=0.0, gate_policy="ER", dose="id",
                                 max_new_tokens=max_new_tokens, num_forced_prefix=nfp,
                                 lid_cache=lid_cache, lid_key=uid)
        assert_no_site_hooks(bundle)
        result["systems"][f"B0M_L{int(layer)}"] = {
            "tokens": r["tokens"], "text": r["text"], "terminated": r["terminated"],
            "zero_dose_bitwise": all(s["f3_bitwise_equals_b"] for s in r["steps"]),
            "lineage_ok": r["lineage_ok"]}
    if baselines:
        b0 = cached.cached_greedy(bundle, encoded, CONDITIONS["cB"], max_new_tokens)
        b1 = cached.cached_greedy(bundle, encoded, CONDITIONS["cE"], max_new_tokens)
        with torch.inference_mode():
            auto = bundle.model.generate(**inputs, task="transcribe", language=None, do_sample=False,
                                         num_beams=1, max_new_tokens=max_new_tokens,
                                         condition_on_prev_tokens=False)[0].tolist()
        eos = bundle.processor.tokenizer.eos_token_id
        auto_content = auto[:auto.index(eos)] if eos in auto else auto
        result["systems"]["B0"] = {k: b0[k] for k in ("tokens", "text", "terminated")}
        result["systems"]["B1"] = {k: b1[k] for k in ("tokens", "text", "terminated")}
        result["systems"]["B0_AUTO"] = {"tokens": auto_content, "terminated": "eos" if len(auto_content) < max_new_tokens else "cap",
                                        "text": bundle.processor.tokenizer.decode(auto_content, skip_special_tokens=True)}
    for c in configs:
        t0 = time.time()
        r = cached.cached_decode(bundle, waveform=waveform, encoded=encoded, conditions=CONDITIONS,
                                 partition=partition, null_probs=null_probs, language_ids=language_ids,
                                 layer=c["layer"], alpha=c["alpha"], gate_policy=c["gate"], dose=c["dose"],
                                 direction_sign=c["direction_sign"], max_new_tokens=max_new_tokens,
                                 num_forced_prefix=nfp, lid_cache=lid_cache, lid_key=uid)
        assert_no_site_hooks(bundle)
        result["systems"][c["name"]] = {"tokens": r["tokens"], "text": r["text"], "terminated": r["terminated"],
                                        "counters": r["counters"], "lineage_ok": r["lineage_ok"],
                                        "distinct_caches": r["distinct_caches"], "config": c,
                                        "elapsed_sec": time.time() - t0, "steps": compact_steps(r["steps"])}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or digest({k: v for k, v in manifest.items()
                                                 if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P2 manifest")
    for path, expected in manifest["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != manifest["git_commit"]:
        raise ValueError("Git commit changed after P2 manifest freeze")
    panel = json.loads((out / "panel.json").read_text())
    if digest(panel) != manifest["panel_hash"]:
        raise ValueError("panel hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    null_probs = cached.native_lid(bundle, np.zeros(480000, dtype=np.float32), language_ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    torch.cuda.reset_peak_memory_stats()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running",
               "audio_sec": 0.0}
    atomic_json(out / "runtime.json", runtime)
    for i, item in enumerate(panel["rows"]):
        dest = out / "rows" / f"{i:03d}.json"
        if validated_row(dest, item["utterance_id"], manifest["manifest_hash"]):
            continue
        t0 = time.time()
        try:
            waveform = load_audio(item["audio_path"], bundle.sample_rate)
            inputs = batch_model_inputs(bundle, [item["audio_path"]])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
            res = run_utterance(bundle, waveform=waveform, encoded=enc, inputs=inputs,
                                configs=manifest["configs"], partition=partition, null_probs=null_probs,
                                language_ids=language_ids, nfp=nfp, uid=item["utterance_id"],
                                baselines=manifest["baselines"],
                                matched_layers=manifest["matched_baseline_layers"])
            res["status"] = "ok"
            runtime["audio_sec"] += min(len(waveform) / 16000, 30.0)
        except Exception as exc:
            import traceback
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc(), "systems": {}}
        res.update(identity=item["utterance_id"], manifest_hash=manifest["manifest_hash"],
                   elapsed_sec=time.time() - t0)
        atomic_json(dest, res)
        print(f"P2 {i + 1}/{len(panel['rows'])} {item['utterance_id']} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), status="completed", peak_vram_bytes=torch.cuda.max_memory_allocated())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
