#!/usr/bin/env python
"""INTERNAL-0A reference-free capture, engineering smoke, selection and seal.

Prepare/evaluate live ONLY in internal0a_offline.py. Full capture requires an
explicit --approved-full-capture flag after human approval. Smoke never writes
tokens, decoded text, word records, or lexical metrics.
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
from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf import internal0a as I

CONFIG = "configs/research_review/internal0a.json"
BASE = "results/research_review/internal0a"
SOURCES = ("src/csasr/inference_cf/internal0a.py", "experiments/internal0a.py",
           "experiments/internal0a_offline.py", "tests/test_internal0a.py", "slurm/internal0a.sbatch",
           CONFIG, "src/csasr/inference_cf/abts0_k1.py", "experiments/abts0_k1.py",
           "experiments/acoustic_s1.py", "experiments/inference_cf_cached.py",
           "src/csasr/data/alignment.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/models/whisper.py",
           "src/csasr/inference_cf/lexical_compatibility.py", "src/csasr/lss/sites.py",
           "src/csasr/utils/provenance.py", "src/csasr/utils/config.py", "src/csasr/utils/hashing.py",
           "src/csasr/data/normalize.py", "src/csasr/data/language_tags.py",
           "src/csasr/evaluation/mer.py", "src/csasr/evaluation/normalization.py",
           "experiments/abts0_k1_evaluate.py", "experiments/abts0_k1_evaluate_v2.py",
           "configs/model/whisper_large_v3.yaml")
FORBIDDEN = ("role_D-", "evaluation_panel", "evaluation_units", "candidates_existing_ctc", "abts0_k1f/", "/references")
OPENED: set[str] = set()


def cfg():
    return json.loads((ROOT / CONFIG).read_text())


def audit_open(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if any(x in p for x in FORBIDDEN):
            raise RuntimeError(f"INTERNAL0A reference firewall: {p}")
        if p.startswith(("/mnt/", str(ROOT))):
            OPENED.add(p)


def load_manifest():
    m = json.loads((ROOT / BASE / "runtime_manifest.json").read_text())
    if digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("runtime manifest hash")
    if m["config_sha256"] != file_hash(ROOT / CONFIG):
        raise ValueError("config changed")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    allowed = {"index", "utterance_id", "dialogue_id", "audio_path", "audio_file_sha256", "waveform_sha256",
               "heard_samples", "n_samples", "global_capture", "global_capture_sha256", "baseline_units", "global_en_units"}
    if any(set(r) != allowed for r in m["rows"]):
        raise ValueError("unsafe runtime projection")
    return m


def load_bundle(m):
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    for p, h in m["model_files"].items():
        if file_hash(Path(m["model_dir"]) / p) != h:
            raise ValueError(f"model/tokenizer file changed: {p}")
    torch.manual_seed(cfg()["seed"])
    b = load_whisper(load_config(ROOT / cfg()["model_config"]))
    if b.device != "cuda" or b.dtype != torch.bfloat16 or b.model.config._attn_implementation != "eager":
        raise ValueError("requires CUDA bf16 eager, no CPU fallback")
    if [list(x) for x in b.model.generation_config.alignment_heads] != cfg()["alignment"]["heads"]:
        raise ValueError("alignment head identity")
    return b


def decode_view(b, enc, prompt, max_tokens):
    from csasr.inference_cf.abts0_k1 import matched_decode
    eng = cfg()["engine"]
    gen = b.model.generation_config
    return matched_decode(b, enc, prompt, eos=eng["eos"], suppress=list(gen.suppress_tokens or []),
                          begin=list(gen.begin_suppress_tokens or []), max_new_tokens=max_tokens,
                          capture_layer=eng["capture_layer"])


def runtime_header(m):
    import torch
    import transformers
    from csasr.utils.provenance import resolved_config_hash, PROVENANCE_VERSION
    return {"manifest_hash": m["manifest_hash"], "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
        "provenance_version": PROVENANCE_VERSION, "resolved_config_hash": resolved_config_hash(cfg()),
        "sources": m["sources"], "model_files": m["model_files"], "start_unix": time.time()}


def read_audio(r):
    from csasr.models.whisper import load_audio
    from experiments.acoustic_s1 import arr_sha
    if file_hash(r["audio_path"]) != r["audio_file_sha256"]:
        raise ValueError("audio bytes changed")
    x = load_audio(r["audio_path"], 16000)
    if len(x) != r["n_samples"] or arr_sha(x) != r["waveform_sha256"]:
        raise ValueError("waveform identity")
    return x[:r["heard_samples"]]


def original_audio(r):
    # Global replay must use the historical full waveform, including preprocessing
    # behavior for >30-s rows; local views are restricted to the heard horizon.
    from csasr.models.whisper import load_audio
    return load_audio(r["audio_path"], 16000)


def check_baseline(b, r, counters):
    from experiments.acoustic_s1 import encode
    if file_hash(ROOT / r["global_capture"]) != r["global_capture_sha256"]:
        raise ValueError("historical capture changed")
    old = json.loads((ROOT / r["global_capture"]).read_text())["B0"]
    enc, _ = encode(b, original_audio(r), counters)
    d = decode_view(b, enc, cfg()["engine"]["cB"], cfg()["engine"]["max_new_tokens_global"])
    ok = all(d[k] == old[k] for k in ("tokens", "terminated", "logits_sha256"))
    if not ok:
        raise ValueError("matched global B0 tokens/termination/raw logits identity")
    return {"tokens_sha256": digest(d["tokens"]), "queries": len(d["logits_sha256"]), "passed": True}


def cmd_gpu(args):
    import torch
    from experiments.acoustic_s1 import encode, auto_generation_config, detect_auto, arr_sha, weight_probe
    from experiments.abts0_k1 import save_heads
    from csasr.lss.sites import assert_no_site_hooks
    m = load_manifest()
    if args.stage == "capture":
        if not args.approved_full_capture:
            raise ValueError("full capture requires explicit human approval")
        forecast = json.loads((ROOT / BASE / "forecast.json").read_text())
        if forecast["manifest_hash"] != m["manifest_hash"] or not forecast["fits"]:
            raise ValueError("missing/stale/non-fitting engineering forecast")
    t_load = time.time()
    b = load_bundle(m)
    load_seconds = time.time() - t_load
    sys.addaudithook(audit_open)
    rt = runtime_header(m)
    w0 = weight_probe(b)
    torch.cuda.reset_peak_memory_stats()
    auto_cfg = auto_generation_config(m["model_dir"])
    counters = {"encoder_calls": 0, "auto_detection_queries": 0}
    measured = []
    rows = m["rows"] if args.stage == "capture" else m["rows"][:1]
    identity = []
    archive = {"outputs": {"archive": cfg()["archive"]}}
    for r in rows:
        x = read_audio(r)
        identity.append(check_baseline(b, r, counters))
        grid = I.window_grid(len(x))
        if args.stage == "smoke":
            grid = grid[:cfg()["compute"]["max_engineering_grid_items"]]
        views = I.physical_views(grid)
        # Encoder shared only between identical audio slices. Every decoder cache
        # and AUTO detection query is fresh. Dedup long crops is explicit in plan.
        for bounds in sorted({tuple(v["bounds"]) for v in views}):
            enc_start = time.time()
            xa = x[bounds[0]:bounds[1]].copy()
            enc, lineage = encode(b, xa, counters)
            torch.cuda.synchronize()
            enc_sec = time.time() - enc_start
            for v in [v for v in views if tuple(v["bounds"]) == bounds]:
                dest = ROOT / BASE / "run1/capture" / f"{r['index']:03d}" / f"{v['view_id']}.json"
                if args.stage == "capture" and dest.exists():
                    existing = json.loads(dest.read_text())
                    if existing.get("manifest_hash") != m["manifest_hash"]:
                        raise ValueError("stale capture; preserve attempt, use new run contract")
                    continue
                t0 = time.time()
                auto = detect_auto(b, enc, auto_cfg, counters) if v["branch"].endswith("AUTO") else None
                if auto is not None and not auto["agrees"]:
                    raise ValueError("native AUTO/lowest-ID identity")
                prompt = auto["prompt"] if auto else cfg()["engine"]["cE" if v["branch"].endswith("EN") else "cB"]
                d = decode_view(b, enc, prompt, cfg()["engine"]["max_new_tokens_local"])
                torch.cuda.synchronize()
                sec = time.time() - t0
                F = -(-len(xa) // 320)
                head = d["heads"][:, :, :F]
                measured.append({"crop_samples": len(xa), "branch": v["branch"], "encode_seconds": enc_sec,
                                 "decode_seconds": sec, "queries": len(d["logits_sha256"]), "capped": d["terminated"] == "cap"})
                if args.stage == "capture":
                    atomic_json(dest, dict(v, utterance_id=r["utterance_id"], manifest_hash=m["manifest_hash"],
                        waveform_sha256=arr_sha(xa), encoder_lineage=lineage, prompt=prompt,
                        tokens=d["tokens"], terminated=d["terminated"], logits_sha256=d["logits_sha256"],
                        heads=save_heads(archive, head), elapsed_sec=sec))
                elif len(measured) == 1:
                    # Fresh encoder and cold decoder: bitwise repetition, including
                    # attention and raw logits. No lexical surfaces are inspected.
                    enc2, _ = encode(b, xa, counters)
                    d2 = decode_view(b, enc2, prompt, cfg()["engine"]["max_new_tokens_local"])
                    if not all(d[k] == d2[k] for k in ("tokens", "terminated", "logits_sha256")) or not np.array_equal(d["heads"], d2["heads"]):
                        raise ValueError("cold-cache repeat identity")
            del enc
    assert_no_site_hooks(b)
    if weight_probe(b) != w0 or any(p.grad is not None or p.requires_grad for p in b.model.parameters()):
        raise ValueError("frozen weights/grad identity")
    rt.update(elapsed_seconds=time.time() - rt["start_unix"], model_load_seconds=load_seconds,
              measurements=measured, global_identity=identity, counters=counters,
              peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
              peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()), opened_paths=sorted(OPENED),
              cold_cache_repeat_pass=True, weights_unchanged_probe=True, no_parameter_gradients=True,
              lexical_outputs_inspected=False if args.stage == "smoke" else None)
    atomic_json(ROOT / BASE / f"{args.stage}_engineering.json", rt)
    print(json.dumps({"stage": args.stage, "crop_branch_runs": len(measured), "identity_pass": True,
                      "seconds": rt["elapsed_seconds"], "peak_bytes": rt["peak_vram_reserved_bytes"]}))


def cmd_forecast(args):
    m = load_manifest()
    s = json.loads((ROOT / BASE / "smoke_engineering.json").read_text())
    if s["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("stale smoke")
    views = [v for r in m["rows"] for v in I.physical_views(I.window_grid(r["heard_samples"]))]
    encoders = sum(len({tuple(v["bounds"]) for v in I.physical_views(I.window_grid(r["heard_samples"]))}) for r in m["rows"])
    sec_step = max(x["decode_seconds"] / max(1, x["queries"]) for x in s["measurements"])
    sec_enc = max(x["encode_seconds"] for x in s["measurements"])
    c = cfg()
    global_steps = sum(len(json.loads((ROOT / r["global_capture"]).read_text())["B0"]["logits_sha256"]) for r in m["rows"])
    bound_steps = len(views) * c["engine"]["max_new_tokens_local"] + global_steps
    expected_steps = len(views) * float(np.mean([x["queries"] for x in s["measurements"]])) + global_steps
    factor = c["compute"]["forecast_safety_factor"]
    reserve = max(c["compute"]["model_load_reserve_seconds"], s["model_load_seconds"])
    out = {"manifest_hash": m["manifest_hash"], "smoke_sha256": file_hash(ROOT / BASE / "smoke_engineering.json"),
           "utterances": len(m["rows"]), "grid_items": sum(len(I.window_grid(r["heard_samples"])) for r in m["rows"]),
           "crop_encoder_calls": encoders, "global_encoder_calls": len(m["rows"]), "branch_decodes": len(views),
           "decoder_query_bound": bound_steps, "observed_seconds_per_query_max": sec_step,
           "observed_seconds_per_encoder_max": sec_enc, "safety_factor": factor,
           "expected_seconds": factor * (expected_steps * sec_step + (encoders + len(m["rows"])) * sec_enc) + reserve,
           "bounded_seconds": factor * (bound_steps * sec_step + (encoders + len(m["rows"])) * sec_enc) + reserve,
           "wall_limit_seconds": c["compute"]["capture_limit_hours"] * 3600}
    out["fits"] = out["bounded_seconds"] <= out["wall_limit_seconds"]
    atomic_json(ROOT / BASE / "forecast.json", out)
    print(json.dumps(out))


def cmd_select(args):
    from experiments.abts0_k1 import load_heads, tokenizer_ctx, token_bytes_list
    from csasr.inference_cf.abts0_k1 import token_intervals
    sys.addaudithook(audit_open)
    m = load_manifest()
    tok, bd = tokenizer_ctx(m)
    rows = []
    for r in m["rows"]:
        words = []
        for v in I.physical_views(I.window_grid(r["heard_samples"])):
            p = ROOT / BASE / "run1/capture" / f"{r['index']:03d}" / f"{v['view_id']}.json"
            d = json.loads(p.read_text())
            if d["manifest_hash"] != m["manifest_hash"]:
                raise ValueError("capture manifest mismatch")
            timing = token_intervals(load_heads(d["heads"]), len(d["tokens"]), d["terminated"], v["bounds"][1] - v["bounds"][0])
            words.extend(I.words_from_view(d, token_bytes_list(tok, bd, d["tokens"]), timing, r["baseline_units"]))
        rows.append(dict(utterance_id=r["utterance_id"], dialogue_id=r["dialogue_id"],
                         **I.summarize_candidates(words, r["baseline_units"])))
    mapping = I.shuffle_map(m["rows"], cfg()["seed"])
    by_id = {r["utterance_id"]: r for r in rows}
    meta = {r["utterance_id"]: r for r in m["rows"]}
    for r in rows:
        uid, donor = r["utterance_id"], mapping[r["utterance_id"]]
        scale = meta[uid]["heard_samples"] / meta[donor]["heard_samples"]
        shuffled = []
        for w in by_id[donor]["words"]:
            sw = dict(w, interval=None if w["interval"] is None else [round(t * scale) for t in w["interval"]],
                      crop=[round(t * scale) for t in w["crop"]], donor_utterance=donor)
            sw["boundary_truncation_possible"] = (sw["interval"] is None or
                sw["interval"][0] < sw["crop"][0] + cfg()["alignment"]["boundary_samples"] or
                sw["interval"][1] > sw["crop"][1] - cfg()["alignment"]["boundary_samples"])
            sw.update(I.associate(sw["interval"], meta[uid]["baseline_units"]))
            shuffled.append(sw)
        r["shuffled"] = dict(I.summarize_candidates(shuffled, meta[uid]["baseline_units"]), donor=donor,
                             informative=donor != uid, duration_ratio=scale)
    atomic_json(ROOT / BASE / "run1/candidates.json", {"manifest_hash": m["manifest_hash"], "rows": rows,
                                                       "references_used": False, "opened_paths": sorted(OPENED)})
    print(json.dumps({"rows": len(rows), "selected": sum(r["selected"] is not None for r in rows), "references_used": False}))


def cmd_seal(args):
    from experiments.abts0_k1 import load_heads
    m = load_manifest()
    files = {}
    archives = {}
    for p in sorted((ROOT / BASE / "run1/capture").rglob("*.json")):
        d = json.loads(p.read_text())
        load_heads(d["heads"])
        archives[d["heads"]["path"]] = "sha256:" + d["heads"]["sha256"].removeprefix("sha256:")
        files[str(p.relative_to(ROOT))] = file_hash(p)
    expected = sum(len(I.physical_views(I.window_grid(r["heard_samples"]))) for r in m["rows"])
    if len(files) != expected:
        raise ValueError("incomplete planned crop matrix")
    for name in ("run1/candidates.json", "capture_engineering.json", "runtime_manifest.json", "forecast.json"):
        p = ROOT / BASE / name
        files[str(p.relative_to(ROOT))] = file_hash(p)
    seal = {"manifest_hash": m["manifest_hash"], "files": files, "archives": archives,
            "sealed_unix": time.time(), "references_opened": False}
    seal["seal_hash"] = digest(seal)
    atomic_json(ROOT / BASE / "output_seal.json", seal)
    print(json.dumps({"files": len(files), "seal_hash": seal["seal_hash"], "next": "commit + push + independent PRIMARY audit before evaluator"}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "capture", "forecast", "select", "seal"])
    p.add_argument("--approved-full-capture", action="store_true")
    a = p.parse_args()
    if a.stage in ("smoke", "capture"):
        cmd_gpu(a)
    else:
        {"forecast": cmd_forecast, "select": cmd_select, "seal": cmd_seal}[a.stage](a)


if __name__ == "__main__":
    main()
