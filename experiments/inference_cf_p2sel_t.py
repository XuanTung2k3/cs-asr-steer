#!/usr/bin/env python
"""P2-SEL-T runner (separate inference_cf development diagnostic). Frozen contract:
docs/inference_cf/P2_SEL_T_SPEC.md, P2_SEL_T_CODEX_DESIGN.md, configs/inference_cf/p2_sel_t.json (fdace84).

``t0``        GPU, NO steering / readout / autograd: replay the unedited forced-ZH B prefix to each of the
              180 frozen queries, take the current-query frozen alignment-head attention (existing cached
              branch), verify the 1 s window against the audited P2-SEL-E E0 rows, integrate the normalized
              frame mean over the three frozen 0.5 s candidates (LEFT/CENTER/RIGHT inside W), choose j* by
              the first exact maximum, and run ``native_lid`` only for LEFT/RIGHT crops not already covered
              by the reused E0 CENTER probabilities. Reads no reference/evaluator information.
``t1``        GPU (conditional): C_NEW = E_tok * R_B * D2 single pulses via the P2-SEL-E pulse helper
              (unchanged P2 hook, sealed D2); gates from the pushed selection artifact.
``manifest``  immutable stage manifest (CPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

SCHEMA = "p2_sel_t_v1"
CONFIG = "configs/inference_cf/p2_sel_t.json"
E0_RUN = "results/inference_cf/p2sel_e/e0_run1"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
BASE = "results/inference_cf/p2sel_t"
HALF = 8000
FRAME = 320
MAX_FRAMES = 1500
ORDER = ("L", "C", "R")
EN_ID, ZH_ID = 50259, 50260


# ---- pure frozen T0 quantities ---------------------------------------------------------------------

def candidate_bounds(s0: int, s1: int) -> dict:
    """Frozen half-open LEFT/CENTER/RIGHT crops inside W=[s0,s1), n=min(8000, s1-s0)."""
    s0, s1 = int(s0), int(s1)
    n = min(HALF, s1 - s0)
    c = (s0 + s1 - n) // 2
    return {"L": (s0, s0 + n), "C": (c, c + n), "R": (s1 - n, s1)}


def frame_mean(heads: np.ndarray, heard: int) -> np.ndarray:
    """Float64 head mean over the first valid=min(1500, ceil(heard/320)) frames, normalized by its sum
    (exactly the core_r2.max_attention_window normalization)."""
    a = np.asarray(heads, dtype=np.float64)
    valid = min(MAX_FRAMES, (int(heard) + FRAME - 1) // FRAME)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < valid or valid <= 0:
        raise ValueError("attention_shape")
    a = a[:, :valid]
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("invalid_attention")
    m = a.mean(axis=0)
    tot = float(m.sum())
    if not math.isfinite(tot) or tot <= 0:
        raise ValueError("zero_attention")
    return m / tot


def crop_mass(mean: np.ndarray, a: int, b: int, heard: int) -> float:
    """Sum over frames of normalized weight * |[a,b) ∩ heard frame| / |heard frame|."""
    heard = min(int(heard), MAX_FRAMES * FRAME)
    tot = 0.0
    f0, f1 = max(0, int(a) // FRAME), min(len(mean), (int(b) + FRAME - 1) // FRAME)
    for f in range(f0, f1):
        fs, fe = f * FRAME, min((f + 1) * FRAME, heard)
        if fe <= fs:
            continue
        ov = min(int(b), fe) - max(int(a), fs)
        if ov > 0:
            tot += float(mean[f]) * ov / (fe - fs)
    return tot


def select(masses: dict) -> str:
    """First exact maximum in order L, C, R (no tolerance)."""
    best = ORDER[0]
    for k in ORDER[1:]:
        if masses[k] > masses[best]:
            best = k
    return best


def token_features(E: dict, masses: dict) -> dict:
    j = select(masses)
    others = [k for k in ORDER if k != j]
    e_ctx = (E[others[0]] + E[others[1]]) / 2
    a_tok = masses[j]
    return {"j": j, "E_tok": E[j], "E_ctx": e_ctx, "C_tokctx": E[j] - e_ctx, "a_tok": a_tok,
            "a_score": a_tok / (masses["L"] + masses["C"] + masses["R"] + 1e-12)}


# ---- T0 (GPU, no steering) -------------------------------------------------------------------------

def t0_utterance(bundle, *, encoded, waveform, tokens, ts, language_ids, cB, e0_rows: dict, null: dict) -> tuple[dict, dict]:
    """Current-query attention, frozen W check, candidate masses, j*, L/R LID (CENTER reused)."""
    import experiments.inference_cf_cached as cached
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_r2 import local_support, max_attention_window
    B = p2r.DiagBranch(bundle, encoded, cB, "B")
    new = list(cB)
    out, vecs = {}, {}
    calls = 0
    lid_cache: dict = {}
    for t in range(max(ts) + 1):
        L = B.length
        _, heads, _ = B.step(new, attention=True)
        if t in ts:
            e0 = e0_rows[str(t)]
            hn = heads.numpy()
            w = max_attention_window(hn, len(waveform))
            W = [w.start_frame, w.end_frame, w.start_sample, w.end_sample]
            heard = w.heard_samples
            mean = frame_mean(hn, len(waveform))
            bounds = candidate_bounds(w.start_sample, w.end_sample)
            masses = {k: crop_mass(mean, a, b, heard) for k, (a, b) in bounds.items()}
            lid_cache.setdefault(tuple(e0["short"]["bounds"]), {"pi_E": e0["short"]["pi_E"], "pi_M": e0["short"]["pi_M"],
                                                               "source": "e0_center"})
            probs = {}
            for k in ORDER:
                key = tuple(bounds[k])
                if key not in lid_cache:
                    p = cached.native_lid(bundle, waveform[key[0]:key[1]], language_ids)
                    calls += 1
                    lid_cache[key] = {"pi_E": p[EN_ID], "pi_M": p[ZH_ID], "source": "new",
                                      "softmax_sum": float(sum(p.values())), "n_probs": len(p),
                                      "all_finite_nonneg": all(math.isfinite(v) and v >= 0 for v in p.values())}
                probs[k] = lid_cache[key]
            E = {k: local_support(probs[k]["pi_E"], probs[k]["pi_M"], null["pi_E"], null["pi_M"])["E"] for k in ORDER}
            feat = token_features(E, masses)
            out[str(t)] = {"t": t, "query": L + len(new) - 1, "window": W, "heard_samples": heard,
                           "attention_mass_W": w.attention_mass,
                           "bounds": {k: list(v) for k, v in bounds.items()}, "masses": masses, "probs": probs,
                           "E": E, **feat, "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest()}
            vecs[f"t{t}_mean"] = mean
        if t < len(tokens):
            new = [tokens[t]]
    return out, {"vecs": vecs, "lid_calls": calls}


# ---- manifest / job entry ----------------------------------------------------------------------------

SOURCES = ("docs/inference_cf/P2_SEL_T_SPEC.md", "docs/inference_cf/P2_SEL_T_CODEX_DESIGN.md", CONFIG,
           "docs/inference_cf/P2_SEL_T_PRE_RUN_AUDIT.md", "results/inference_cf/p2rj/positions.json", CONSTRUCTION,
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/readout.py", "src/csasr/lss/sites.py",
           "src/csasr/models/hooks.py", "experiments/inference_cf_p0_r2.py", "experiments/inference_cf_cached.py",
           "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2dir.py", "experiments/inference_cf_p2sel.py",
           "experiments/inference_cf_p2sel_e.py", "experiments/inference_cf_p2sel_t.py",
           "experiments/inference_cf_p2sel_t_analyze.py", "experiments/inference_cf_p2sel_t_audit.py",
           "slurm/inference_cf_p2sel_t.sbatch", f"{E0_RUN}/manifest.json", f"{E0_RUN}/null.json",
           "results/inference_cf/p2sel_e/e0_run1_audit.json", "results/inference_cf/p2sel_e/e0_run1_analysis.json",
           "results/inference_cf/p2sel/s1_run1/manifest.json",
           "results/inference_cf/p2dir/exp1_run1/directions_sealed.json")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2sel_e as pe
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = [f"{BASE}/prerun_audit.json"]
    if args.stage == "t1":
        extra += [f"{BASE}/t0_selection.json", f"{BASE}/t0_run1_audit.json", f"{BASE}/pre_t1_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-SEL-T sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / f"{BASE}/prerun_audit.json").read_text())["verdict"] != "PASS_TO_P2_SEL_T_T0":
        raise ValueError("pre-T0 audit did not pass")
    if args.stage == "t1" and json.loads((ROOT / f"{BASE}/pre_t1_audit.json").read_text())["verdict"] != "P2_SEL_T_PRE_T1_AUDIT: PASS":
        raise ValueError("pre-T1 audit did not pass")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config": CONFIG, "config_hash": digest(cfg), "construction": CONSTRUCTION, "construction_hash": con["construction_hash"],
           "partition_hash": partition["hash"],
           "suppression_hash": digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}),
           "alignment_heads": [list(map(int, x)) for x in gen.alignment_heads],
           "provider_integrity": pe.provider_integrity(str(prep.MODEL / "generation_config.json")),
           "role": "D-dev-select", "firewall": prep.firewall(con["utterances"]), "environment": prep.environment(),
           "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(extra)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": args.stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def verify_manifest(out: Path, stage: str) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    if m.get("schema") != SCHEMA or m.get("stage") != stage or \
            digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError(f"invalid P2-SEL-T {stage} manifest")
    for path, expected in m["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("Git commit changed after manifest freeze")
    return m


def run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2sel as p2sel
    import experiments.inference_cf_p2sel_e as pe
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
    from csasr.utils.config import load_config

    out = ROOT / args.out
    m = verify_manifest(out, args.cmd)
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.requires_grad_(False)
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    if len(language_ids) != 100 or (native["<|en|>"], native["<|zh|>"]) != (EN_ID, ZH_ID):
        raise ValueError("provider identity")
    if [list(map(int, x)) for x in bundle.model.generation_config.alignment_heads] != m["alignment_heads"]:
        raise ValueError("alignment heads changed")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != m["partition_hash"]:
        raise ValueError("partition hash")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    base_dir = ROOT / p2dir.BASELINE_RUN
    bidx = {r["utterance_id"]: i for i, r in enumerate(json.loads((base_dir / "panel.json").read_text())["rows"])}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / p2dir.AUDIO_PANEL).read_text())["rows"]}
    by_utt: dict = {}
    for p in con["positions"]:
        by_utt.setdefault(p["utterance_id"], []).append(int(p["t"]))
    torch.cuda.reset_peak_memory_stats()
    runtime = {"stage": args.cmd, "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running",
               "autograd_calls": 0, "readout_calls": 0, "steering_calls": 0, "lid_calls": 0}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    null = json.loads((ROOT / E0_RUN / "null.json").read_text())
    if args.cmd == "t1":
        sel = json.loads((ROOT / f"{BASE}/t0_selection.json").read_text())
        gates = {(g["utterance_id"], int(g["t"])): g for g in sel["gates"]}
        ctx = p2dir.Ctx(bundle, partition, nfp, 0.0)
        _, p2d_vecs, p2d_none = p2sel.load_p2dir_rows(con)
    failures = 0
    for i, uid in enumerate(con["utterances"]):
        ts = sorted(by_utt[uid])
        toks = json.loads((base_dir / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"][p2dir.BASELINE_SYSTEM]["tokens"]
        t0 = time.time()
        try:
            inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
                if args.cmd == "t0":
                    e0 = json.loads((ROOT / E0_RUN / f"rows/{i:03d}.json").read_text())
                    if e0["identity"] != uid or e0["status"] != "ok":
                        raise ValueError("E0 row mismatch")
                    wav = load_audio(audio[uid]["audio_path"], bundle.sample_rate)
                    recs, extra = t0_utterance(bundle, encoded=enc, waveform=wav, tokens=toks, ts=ts, language_ids=language_ids,
                                               cB=p2dir.CB, e0_rows=e0["positions"], null=null)
                    runtime["lid_calls"] += extra["lid_calls"]
                    np.savez_compressed(out / "rows" / f"{i:03d}.npz", **extra["vecs"])
                    res = {"status": "ok", "positions": recs, "lid_calls": extra["lid_calls"],
                           "vectors_sha256": file_hash(out / "rows" / f"{i:03d}.npz")}
                else:
                    recs, vecs, logits = pe.e1_utterance(ctx, encoded=enc, tokens=toks, ts=ts,
                                                         gates={t: gates[(uid, t)] for t in ts},
                                                         ref_vecs=p2d_vecs[uid], ref_none=p2d_none[uid])
                    runtime["steering_calls"] += len(recs)
                    np.savez_compressed(out / "rows" / f"{i:03d}.npz", **vecs)
                    np.savez_compressed(out / "rows" / f"{i:03d}_logits.npz", **logits)
                    res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}.npz"),
                           "logits_sha256": file_hash(out / "rows" / f"{i:03d}_logits.npz")}
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(out / "rows" / f"{i:03d}.json", res)
        print(f"P2-SEL-T {args.cmd} {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
    if any(p.grad is not None for p in bundle.model.parameters()):
        raise RuntimeError("parameter gradient present")
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures,
                   peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if failures:
        raise SystemExit(f"{failures} failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--stage", required=True, choices=("t0", "t1"))
    mf.add_argument("--out", required=True)
    for s in ("t0", "t1"):
        sub.add_parser(s).add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_manifest(args) if args.cmd == "manifest" else run(args)


if __name__ == "__main__":
    main()
