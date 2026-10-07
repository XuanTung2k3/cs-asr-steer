#!/usr/bin/env python
"""P2-SEL-LAC runner (separate inference_cf development diagnostic). Frozen contract:
docs/inference_cf/P2_SEL_LAC_SPEC.md, P2_SEL_LAC_CODEX_DESIGN.md, configs/inference_cf/p2_sel_lac.json (be460ce).

``seal``      CPU: candidates c_E/c_M from the sealed lossless UNMASKED P2-DIR NONE logits only (canonical
              V_E/V_M, lowest-ID ties), with uid/t/query, exact input-prefix IDs, W* (P2-SEL-T bounds[j]) and
              all lineage hashes -> results/inference_cf/p2sel_lac/candidates_sealed.json (pushed before LAC0).
``manifest``  immutable stage manifest (CPU).
``lac0``      GPU, forward-only: per position, x_mask = x with W* hard-zeroed; whole masked encoder (same
              feature-extractor call); fresh unsteered forced-ZH B branch replayed from the prompt through
              baseline tokens[:t] (bounded exact-prefix LRU-2 per utterance); masked current-query raw logits.
              No backward / D2 / LID / steering; candidates are read from the seal, never re-selected.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
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
from csasr.inference_cf.lexical_compatibility import HEARD_MAX, hard_mask, select_candidates, waveform_model_inputs

SCHEMA = "p2_sel_lac_v1"
CONFIG = "configs/inference_cf/p2_sel_lac.json"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
P2DIR_RUN = "results/inference_cf/p2dir/exp1_run1"
T0_RUN = "results/inference_cf/p2sel_t/t0_run1"
BASE = "results/inference_cf/p2sel_lac"
SEAL = f"{BASE}/candidates_sealed.json"
CB = [50258, 50260, 50360, 50364]
LRU = 2


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def arr_sha(a: np.ndarray) -> str:
    return sha(np.ascontiguousarray(a).tobytes())


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


# ---- CPU seal ----------------------------------------------------------------------------------------

def cmd_seal(args) -> None:
    from transformers import WhisperProcessor
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2dir_prepare as prep
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2dir_analyze import unpack_bf16
    cfg = json.loads((ROOT / CONFIG).read_text())
    part = tokenizer_partition(WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer)
    if part["hash"] != cfg["candidates"]["partition_hash"]:
        raise ValueError("partition hash")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    sealed = json.loads((ROOT / P2DIR_RUN / "directions_sealed.json").read_text())
    base_dir = ROOT / p2dir.BASELINE_RUN
    bidx = {r["utterance_id"]: i for i, r in enumerate(json.loads((base_dir / "panel.json").read_text())["rows"])}
    rows = []
    for i, uid in enumerate(con["utterances"]):
        lg_rel = f"rows/{i:03d}_logits.npz"
        if file_hash(ROOT / P2DIR_RUN / lg_rel) != sealed["files"][lg_rel]:
            raise ValueError("unmasked logits seal")
        prow = json.loads((ROOT / P2DIR_RUN / f"rows/{i:03d}.json").read_text())["positions"]
        trow = json.loads((ROOT / T0_RUN / f"rows/{i:03d}.json").read_text())
        if trow["identity"] != uid or trow["status"] != "ok":
            raise ValueError("T0 row")
        toks = json.loads((base_dir / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"][p2dir.BASELINE_SYSTEM]["tokens"]
        with np.load(ROOT / P2DIR_RUN / lg_rel) as z:
            L = {k: unpack_bf16(z[k]) for k in z.files if k.endswith("_none")}
        for p in [p for p in con["positions"] if p["utterance_id"] == uid]:
            t = int(p["t"])
            z = L[f"t{t}_none"]
            cE, cM = select_candidates(z, part["embedded_ids"], part["matrix_ids"])
            tp = trow["positions"][str(t)]
            prefix = CB + [int(x) for x in toks[:t]]
            rows.append({"utterance_id": uid, "t": t, "dialogue_id": p["dialogue_id"], "query": prow[str(t)]["query"],
                         "c_E": cE, "c_M": cM, "z_E": float(z[cE]), "z_M": float(z[cM]),
                         "input_ids": prefix, "input_ids_sha256": sha(json.dumps(prefix).encode()),
                         "j": tp["j"], "W_star": tp["bounds"][tp["j"]], "W_1s": tp["window"], "heard_samples": tp["heard_samples"],
                         "unmasked_logits_file": f"{P2DIR_RUN}/{lg_rel}", "unmasked_logits_sha256": sealed["files"][lg_rel],
                         "unmasked_logit_vector_sha256": arr_sha(z.astype(np.float32))})
    doc = {"schema": SCHEMA + "_candidates", "partition_hash": part["hash"], "V_E_size": len(part["embedded_ids"]),
           "V_M_size": len(part["matrix_ids"]), "config_hash": digest(cfg), "t0_audit_sha256": file_hash(ROOT / "results/inference_cf/p2sel_t/t0_run1_audit.json"),
           "directions_sealed_sha256": file_hash(ROOT / P2DIR_RUN / "directions_sealed.json"), "model_files": prep.model_hashes(),
           "rows": rows}
    doc["seal_hash"] = digest(doc)
    out = ROOT / SEAL
    if out.exists():
        raise FileExistsError("candidate seal exists; never overwrite")
    atomic_json(out, doc)
    back = json.loads(out.read_text())
    if digest({k: v for k, v in back.items() if k != "seal_hash"}) != back["seal_hash"]:
        raise ValueError("seal round-trip")
    print(json.dumps({"rows": len(rows), "seal_hash": doc["seal_hash"]}))


# ---- GPU LAC0 ------------------------------------------------------------------------------------------

def lac0_utterance(bundle, *, waveform: np.ndarray, seal_rows: list[dict], counters: dict, prompt=CB) -> tuple[dict, dict]:
    """Masked counterfactual raw logits at the sealed queries. Fresh masked encoder + fresh B cache per mask;
    a cached branch is continued only when its fed IDs are an exact prefix of the target input IDs."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_p2r as p2r
    from csasr.lss.sites import assert_no_site_hooks
    x = waveform
    heard = min(len(x), HEARD_MAX)
    x_sha = arr_sha(x)
    lru: OrderedDict = OrderedDict()
    recs, logits = {}, {}
    for r in sorted(seal_rows, key=lambda r: r["t"]):
        t = r["t"]
        a, b = int(r["W_star"][0]), int(r["W_star"][1])
        xm = hard_mask(x, a, b)
        e64 = float((x[a:b].astype(np.float64) ** 2).sum())
        rec = {"t": t, "query": r["query"], "W_star": [a, b], "heard_samples": heard, "x_sha256": x_sha,
               "outside_left_equal": bool(xm[:a].tobytes() == x[:a].tobytes()),
               "outside_right_equal": bool(xm[b:].tobytes() == x[b:].tobytes()),
               "inside_positive_zero": bool(np.all(xm[a:b].view(np.uint32) == 0)),
               "local_energy": e64, "removed_energy_fraction": e64 / (float((x[:heard].astype(np.float64) ** 2).sum()) + 1e-12),
               "changed": not (xm.tobytes() == x.tobytes()), "shape_dtype_ok": xm.shape == x.shape and xm.dtype == np.float32,
               "finite": bool(np.isfinite(xm).all()), "x_mask_sha256": arr_sha(xm)}
        if not rec["changed"]:
            rec["noop"] = True                                        # reuse unmasked logits, S = 0
            recs[str(t)] = rec
            continue
        key = (x_sha, a, b)
        target = list(r["input_ids"])
        if target[:len(prompt)] != list(prompt):
            raise ValueError("input prefix does not start with the frozen prompt")
        ent = lru.get(key)
        if ent is not None and ent["branch"].fed == target[:len(ent["branch"].fed)] and len(ent["branch"].fed) <= len(target):
            lru.move_to_end(key)
            rec["branch_origin"] = "reused_exact_prefix"
        else:
            feats = waveform_model_inputs(bundle, xm)
            enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                input_features=feats["input_features"], attention_mask=feats["attention_mask"]).last_hidden_state)
            counters["encoder_calls"] += 1
            ent = {"branch": p2r.DiagBranch(bundle, enc, list(prompt), "Bmask"), "feat_sha256": arr_sha(feats["input_features"].float().cpu().numpy()),
                   "attn_sha256": arr_sha(feats["attention_mask"].cpu().numpy())}
            lru[key] = ent
            if len(lru) > LRU:
                lru.popitem(last=False)
            rec["branch_origin"] = "cold"
        br = ent["branch"]
        rec["feat_sha256"], rec["attn_mask_sha256"] = ent["feat_sha256"], ent["attn_sha256"]
        lz = None
        if br.length == 0:
            lz, _, _ = br.step(list(prompt), attention=True)
            counters["decoder_steps"] += 1
        while br.length < len(target):
            lz, _, _ = br.step([target[br.length]], attention=True)
            counters["decoder_steps"] += 1
        if lz is None:
            raise RuntimeError("cached branch already past the target query")
        assert_no_site_hooks(bundle)
        rec["fed_sha256"] = sha(json.dumps(br.fed).encode())
        rec["fed_equals_input_ids"] = br.fed == target
        rec["query_matches"] = br.length - 1 == r["query"]
        rec["logits_finite"] = bool(torch.isfinite(lz).all())
        rec["vocab"] = int(lz.numel())
        counters["counterfactual_evaluations"] += 1
        logits[f"t{t}_masked"] = p2dir.pack_bf16(lz)
        recs[str(t)] = rec
    return recs, logits


SOURCES = ("docs/inference_cf/P2_SEL_LAC_SPEC.md", "docs/inference_cf/P2_SEL_LAC_CODEX_DESIGN.md", CONFIG,
           "docs/inference_cf/P2_SEL_LAC_PRE_RUN_AUDIT.md", SEAL, "results/inference_cf/p2rj/positions.json", CONSTRUCTION,
           "src/csasr/inference_cf/lexical_compatibility.py", "src/csasr/models/whisper.py", "src/csasr/inference_cf/core_r2.py",
           "experiments/inference_cf_cached.py", "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2dir.py",
           "experiments/inference_cf_p2sel_lac.py", "experiments/inference_cf_p2sel_lac_analyze.py",
           "experiments/inference_cf_p2sel_lac_audit.py", "slurm/inference_cf_p2sel_lac.sbatch",
           f"{P2DIR_RUN}/directions_sealed.json", f"{T0_RUN}/manifest.json", "results/inference_cf/p2sel_t/t0_run1_audit.json")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2sel_e as pe
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = [f"{BASE}/prerun_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-SEL-LAC sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / extra[0]).read_text())["verdict"] != "PASS_TO_P2_SEL_LAC_LAC0":
        raise ValueError("pre-LAC0 audit did not pass")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    partition = tokenizer_partition(proc.tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config": CONFIG, "config_hash": digest(cfg), "construction": CONSTRUCTION, "construction_hash": con["construction_hash"],
           "partition_hash": partition["hash"], "suppression": {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])},
           "feature_extractor": proc.feature_extractor.to_dict(), "provider_integrity": pe.provider_integrity(str(prep.MODEL / "generation_config.json")),
           "seal_hash": json.loads((ROOT / SEAL).read_text())["seal_hash"], "role": "D-dev-select", "firewall": prep.firewall(con["utterances"]),
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(extra)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": args.stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def run(args) -> None:
    import torch
    import experiments.inference_cf_p2dir as p2dir
    from csasr.models.whisper import load_audio, load_whisper
    from csasr.utils.config import load_config
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or m["stage"] != "lac0" or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    seal = json.loads((ROOT / SEAL).read_text())
    if seal["seal_hash"] != m["seal_hash"]:
        raise ValueError("seal hash")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.requires_grad_(False)
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / p2dir.AUDIO_PANEL).read_text())["rows"]}
    torch.cuda.reset_peak_memory_stats()
    counters = {"encoder_calls": 0, "decoder_steps": 0, "counterfactual_evaluations": 0, "autograd_calls": 0,
                "readout_calls": 0, "lid_calls": 0, "steering_calls": 0}
    runtime = {"stage": "lac0", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "counters": counters}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    failures = 0
    for i, uid in enumerate(con["utterances"]):
        t0 = time.time()
        try:
            wav = load_audio(audio[uid]["audio_path"], bundle.sample_rate)
            with torch.inference_mode():
                recs, logits = lac0_utterance(bundle, waveform=wav, seal_rows=[r for r in seal["rows"] if r["utterance_id"] == uid],
                                              counters=counters)
            np.savez_compressed(out / "rows" / f"{i:03d}_logits.npz", **logits)
            res = {"status": "ok", "positions": recs, "logits_sha256": file_hash(out / "rows" / f"{i:03d}_logits.npz")}
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(out / "rows" / f"{i:03d}.json", res)
        print(f"P2-SEL-LAC lac0 {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
    if any(p.grad is not None for p in bundle.model.parameters()):
        raise RuntimeError("parameter gradient present")
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures,
                   peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(), peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if failures:
        raise SystemExit(f"{failures} failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seal")
    mf = sub.add_parser("manifest")
    mf.add_argument("--stage", required=True, choices=("lac0",))
    mf.add_argument("--out", required=True)
    sub.add_parser("lac0").add_argument("--out", required=True)
    args = ap.parse_args()
    {"seal": cmd_seal, "manifest": cmd_manifest, "lac0": run}[args.cmd](args)


if __name__ == "__main__":
    main()
