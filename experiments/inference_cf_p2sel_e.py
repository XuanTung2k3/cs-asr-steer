#!/usr/bin/env python
"""P2-SEL-E runner (separate inference_cf development diagnostic). Frozen contract (revised):
docs/inference_cf/P2_SEL_E_SPEC.md, P2_SEL_E_CODEX_DESIGN.md, configs/inference_cf/p2_sel_e.json (b693b8d).

``e0``        GPU, NO steering: for the 180 frozen keys, teacher-force the unedited forced-ZH B branch on
              the matched-baseline tokens, take the frozen current-query localizer window, and run the
              unchanged 100-way ``native_lid`` on (a) the selected 1 s window and (b) the single 0.5 s
              same-center diagnostic crop, plus the zero-waveform null once. Persists the four local/null
              probabilities, the full-softmax sum, attention mass, max-attention frame and window bounds.
              Reads NO reference/evaluator information.
``e1``        GPU: C_new = E_new * R_B * D2 single pulses (sealed P2-DIR D2, unchanged P2 hook, alpha 2,
              NormPreserve) from the identical unedited cache, gates read from the sealed pre-E1 artifact.
``manifest``  immutable stage manifest (CPU).

Pure helpers below are shared by the analysis (E0 quantities, branch formulas) and tested directly.
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
from csasr.inference_cf.core_r2 import local_support

SCHEMA = "p2_sel_e_v1"
CONFIG = "configs/inference_cf/p2_sel_e.json"
P2R_RUN = "results/inference_cf/p2r/run3"
P2DIR_RUN = "results/inference_cf/p2dir/exp1_run1"
P2SEL_RUN = "results/inference_cf/p2sel/s1_run1"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
EN_ID, ZH_ID = 50259, 50260
EPS = 1e-12
SHORT = 8000
NULL_SAMPLES = 480000
BRANCHES = ("R1", "R2", "R3", "R4")


# ---- pure frozen quantities ------------------------------------------------------------------------

def short_bounds(s0: int, s1: int, heard: int, length: int = SHORT) -> tuple[int, int]:
    """Same-center 0.5 s crop: a = clamp(floor((s0+s1-8000)/2), 0, heard-8000); [0, heard) if heard < 8000."""
    if heard < length:
        return 0, int(heard)
    a = (int(s0) + int(s1) - length) // 2
    a = min(max(a, 0), int(heard) - length)
    return a, a + length


def decompose(pe: float, pm: float, ne: float, nm: float) -> dict:
    """E0 decomposition; E via the unchanged core_r2.local_support."""
    s = local_support(pe, pm, ne, nm)
    ell_local = math.log((pe + EPS) / (pm + EPS))
    ell_null = math.log((ne + EPS) / (nm + EPS))
    A = ell_local - ell_null
    return {"ell_local": ell_local, "ell_null": ell_null, "A": A, "E": s["E"], "Q": pe + pm,
            "null_share": (-ell_null) / A if A > 0 else None, "local_support_A": s["A"]}


def lag_values(e_by_query: dict, query: int) -> tuple[float, float]:
    """Causal E_(t-1), E_(t-2) by absolute query; missing (or fallback E=None) history is zero.
    Never reads query or later."""
    def get(q):
        v = e_by_query.get(q)
        return 0.0 if v is None else float(v)
    return get(query - 1), get(query - 2)


def e_new(branch: str, row: dict) -> float:
    """Frozen single-branch formulas (reference-free runtime inputs only)."""
    if branch == "R1":
        return float(row["E"]) * max(float(row["E_lag1"]), float(row["E_lag2"]))
    if branch == "R2":
        return math.sqrt(float(row["E"]) * float(row["E_short"]))
    if branch == "R3":
        v = float(row["E"]) * float(row["Q"])
        if not (-1e-7 <= v <= float(row["E"]) + 1e-7 <= 1 + 2e-7):
            raise ValueError("R3 bound violated")
        return v
    if branch == "R4":
        return max(0.0, math.tanh((float(row["ell_local"]) - 0.5 * float(row["ell_null"])) / 2))
    raise ValueError(f"no formula for {branch}")


E_NEW_INPUTS = {"R1": ("E", "E_lag1", "E_lag2"), "R2": ("E", "E_short"), "R3": ("E", "Q"),
                "R4": ("ell_local", "ell_null")}


def provider_integrity(gen_path: str) -> dict:
    g = json.loads(Path(gen_path).read_text())
    lt = g["lang_to_id"]
    payload = json.dumps(sorted((str(k), int(v)) for k, v in lt.items()), separators=(",", ":"), ensure_ascii=False)
    return {"generation_config_sha256": hashlib.sha256(Path(gen_path).read_bytes()).hexdigest(),
            "mapping_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "n": len(set(lt.values())), "en_zh": [lt["<|en|>"], lt["<|zh|>"]]}


# ---- E0 (GPU, no steering) ----------------------------------------------------------------------------

def e0_utterance(bundle, *, encoded, waveform, tokens, ts, language_ids, cB, layer=16) -> dict:
    """Unedited B replay; frozen localizer window + 100-way LID on long and short crops."""
    import torch
    import experiments.inference_cf_cached as cached
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_r2 import max_attention_window
    B = p2r.DiagBranch(bundle, encoded, cB, "B")
    new = list(cB)
    out = {}
    for t in range(max(ts) + 1):
        L = B.length
        _, heads, _ = B.step(new, attention=True)
        if t in ts:
            w = max_attention_window(heads.numpy(), len(waveform))
            rec = {"t": t, "query": L + len(new) - 1,
                   "window": [w.start_frame, w.end_frame, w.start_sample, w.end_sample],
                   "attention_mass": w.attention_mass, "max_attention_frame": w.point_argmax_frame,
                   "heard_samples": w.heard_samples, "center_sample": (w.start_sample + w.end_sample) / 2}
            for name, (a, b) in (("long", (w.start_sample, w.end_sample)),
                                 ("short", short_bounds(w.start_sample, w.end_sample, w.heard_samples))):
                probs = cached.native_lid(bundle, waveform[a:b], language_ids)
                rec[name] = {"bounds": [a, b], "pi_E": probs[EN_ID], "pi_M": probs[ZH_ID],
                             "softmax_sum": float(sum(probs.values())), "n_probs": len(probs),
                             "all_finite_nonneg": all(math.isfinite(v) and v >= 0 for v in probs.values())}
            out[str(t)] = rec
        if t < len(tokens):
            new = [tokens[t]]
    return out


# ---- E1 (GPU, single C_new pulse) --------------------------------------------------------------------------

def e1_utterance(ctx, *, encoded, tokens, ts, gates, ref_vecs, ref_none) -> tuple[dict, dict, dict]:
    """C_new = E_new*R_B*D2 pulses from the identical unedited B cache (unchanged P2 hook)."""
    import torch
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2r as p2r
    import experiments.inference_cf_p2sel as p2sel
    from csasr.lss.sites import assert_no_site_hooks
    from csasr.inference_cf.core_p1 import processed_argmax
    B = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B")
    new = list(ctx.cB)
    recs, vecs, logits = {}, {}, {}
    for t in range(max(ts) + 1):
        L = B.length
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        if t in ts:
            pre = f"t{t}_"
            query = L + len(new) - 1
            g = float(gates[t]["g_new"])
            d2 = torch.from_numpy(ref_vecs[pre + "D2"].copy())
            B.crop(L)
            hook, applied = p2sel.gated_hook(ctx.bundle, query, g, d2, ctx.nfp)
            la, _, h_cons = B.step(new, capture_layer=ctx.layer, attention=True, hook=hook)
            assert_no_site_hooks(ctx.bundle)
            audit = hook.records[-1].to_dict() if hook.records else None
            steered = bool(audit and audit["steered"] and audit["edit_norm"] > 0)
            arm = {"applied": applied, "gain": g if applied else 0.0, "dose": 2.0 * g if applied else 0.0,
                   "steered": steered, "edit_norm": audit["edit_norm"] if steered else 0.0,
                   "pre_norm": audit["pre_norm"] if audit else None,
                   "argmax": processed_argmax(la, t, ctx.suppress, ctx.begin),
                   "zero_gain_bitwise_none": None if applied else bool(torch.equal(la, lb))}
            if steered:
                vecs[pre + "post_Cnew"] = h_cons.numpy()
            logits[pre + "Cnew"] = p2dir.pack_bf16(la)
            B.crop(L)
            lb2, _, hb2 = B.step(new, capture_layer=ctx.layer, attention=True)
            recs[str(t)] = {"t": t, "query": query, "query_matches_gate": query == gates[t]["query"], "arm": arm,
                            "hb_bitwise_p2dir": bool(np.array_equal(hb.numpy(), ref_vecs[pre + "hb"])),
                            "none_logits_bitwise_p2dir": bool(np.array_equal(p2dir.pack_bf16(lb), ref_none[pre + "none"])),
                            "restore_bitwise": bool(torch.equal(lb2, lb) and torch.equal(hb2, hb))}
        if t < len(tokens):
            new = [tokens[t]]
    return recs, vecs, logits


# ---- manifest / job entry ------------------------------------------------------------------------------

SOURCES = ("docs/inference_cf/P2_SEL_E_SPEC.md", "docs/inference_cf/P2_SEL_E_CODEX_DESIGN.md", CONFIG,
           "docs/inference_cf/P2_SEL_E_PRE_RUN_AUDIT_R1.md", "results/inference_cf/p2rj/positions.json", CONSTRUCTION,
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/readout.py",
           "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "experiments/inference_cf_p0_r2.py",
           "experiments/inference_cf_cached.py", "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2dir.py",
           "experiments/inference_cf_p2sel.py", "experiments/inference_cf_p2sel_e.py",
           "experiments/inference_cf_p2sel_e_analyze.py", "experiments/inference_cf_p2sel_e_audit.py",
           "slurm/inference_cf_p2sel_e.sbatch", f"{P2R_RUN}/manifest.json", f"{P2DIR_RUN}/directions_sealed.json",
           f"{P2SEL_RUN}/manifest.json")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = []
    if args.stage == "e1":
        extra = ["results/inference_cf/p2sel_e/e0_run1_selection.json", "results/inference_cf/p2sel_e/e0_run1_audit.json",
                 "results/inference_cf/p2sel_e/pre_e1_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-SEL-E sources before preparing a manifest:\n" + dirty)
    pre = json.loads((ROOT / "results/inference_cf/p2sel_e/prerun_audit_r1.json").read_text())
    if pre["verdict"] != "PASS_TO_P2_SEL_E_E0":
        raise ValueError("pre-E0 audit did not pass")
    if args.stage == "e1":
        pe1 = json.loads((ROOT / "results/inference_cf/p2sel_e/pre_e1_audit.json").read_text())
        if pe1["verdict"] != "P2_SEL_E_PRE_E1_AUDIT: PASS":
            raise ValueError("pre-E1 audit did not pass")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"),
           "git_tree": git("rev-parse", "HEAD^{tree}"), "config": CONFIG, "config_hash": digest(cfg),
           "construction": CONSTRUCTION, "construction_hash": con["construction_hash"],
           "partition_hash": partition["hash"],
           "suppression_hash": digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}),
           "provider_integrity": provider_integrity(str(prep.MODEL / "generation_config.json")),
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
        raise ValueError(f"invalid P2-SEL-E {stage} manifest")
    for path, expected in m["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("Git commit changed after manifest freeze")
    return m


def run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    import experiments.inference_cf_cached as cached
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2sel as p2sel
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
               "autograd_calls": 0, "readout_calls": 0, "lid_calls": 0}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    null = None
    if args.cmd == "e0":
        null = cached.native_lid(bundle, np.zeros(NULL_SAMPLES, dtype=np.float32), language_ids)
        runtime["lid_calls"] += 1
        atomic_json(out / "null.json", {"pi_E": null[EN_ID], "pi_M": null[ZH_ID], "softmax_sum": float(sum(null.values())),
                                        "n_probs": len(null), "samples": NULL_SAMPLES})
    gates = ref = None
    if args.cmd == "e1":
        sel = json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_selection.json").read_text())
        gates = {(g["utterance_id"], int(g["t"])): g for g in sel["gates"]}
        ctx = p2dir.Ctx(bundle, partition, nfp, 0.0)
        p2d_rows, p2d_vecs, p2d_none = p2sel.load_p2dir_rows(con)
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
                if args.cmd == "e0":
                    wav = load_audio(audio[uid]["audio_path"], bundle.sample_rate)
                    recs = e0_utterance(bundle, encoded=enc, waveform=wav, tokens=toks, ts=ts,
                                        language_ids=language_ids, cB=p2dir.CB)
                    runtime["lid_calls"] += 2 * len(recs)
                    res = {"status": "ok", "positions": recs}
                else:
                    recs, vecs, logits = e1_utterance(ctx, encoded=enc, tokens=toks, ts=ts,
                                                      gates={t: gates[(uid, t)] for t in ts},
                                                      ref_vecs=p2d_vecs[uid], ref_none=p2d_none[uid])
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
        print(f"P2-SEL-E {args.cmd} {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
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
    mf.add_argument("--stage", required=True, choices=("e0", "e1"))
    mf.add_argument("--out", required=True)
    for s in ("e0", "e1"):
        sub.add_parser(s).add_argument("--out", required=True)
    args = ap.parse_args()
    cmd_manifest(args) if args.cmd == "manifest" else run(args)


if __name__ == "__main__":
    main()
