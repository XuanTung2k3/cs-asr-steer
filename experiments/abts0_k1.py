#!/usr/bin/env python
"""ABTS-0 K1 runner -- reference-free candidate feasibility (docs/research_review/ABTS0_DESIGN.md K1;
configs/inference_cf/abts0_k1.json; src/csasr/inference_cf/abts0_k1.py). K1 only: no steering, training or K2.

prepare   (CPU) pins panel / audio / R0 rows / model files / sources into plan_sealed.json (no reference access).
forecast  (CPU) bounded compute forecast from the plan (no model).
capture   (GPU) per utterance: one encoder; matched-engine B0 (cB) and EN (cE) decodes with cached alignment-head
          attention at every step; matched-engine AUTO (native detection + forced decode). Raw attention is archived
          content-addressed under /mnt/data; compact rows go to run1/capture/.
select    (CPU) DTW timing, units, association, R0 region + unchanged S1 crops, earliest choice, controls (U0/U1/U2,
          shuffled track); writes selection.json and candidates_sealed.json BEFORE any acoustic scoring.
score     (GPU) only the sealed chosen pairs: hard-zero target / off-target encoders, cold matched-engine replays of the
          B0 prefix, F / A / C and the frozen accept rule; bitwise query-logit identity against capture.
seal      (CPU) immutable output seal over every run file and archive object, before any reference evaluation.
Execution order: capture (GPU job A) -> select (CPU, pure-Python DTW parallelised over --workers) -> commit + push the
candidate seal -> score (GPU job B) -> seal. Selection never runs inside a GPU allocation.

No reference, evaluator target, stratum or outcome file is opened here; GPU stages log every opened path.
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

SCHEMA = "abts0_k1_v1"
CONFIG = "configs/inference_cf/abts0_k1.json"
BASE = "results/inference_cf/abts0_k1"
RUN = f"{BASE}/run1"
PLAN = f"{BASE}/plan_sealed.json"
SELECTION = f"{RUN}/selection.json"
CAND_SEAL = f"{RUN}/candidates_sealed.json"
OUTPUT_SEAL = f"{BASE}/output_seal.json"
SOURCES = ("src/csasr/inference_cf/abts0_k1.py", "experiments/abts0_k1.py", "experiments/inference_cf_cached.py",
           "src/csasr/inference_cf/s1_evidence.py", "src/csasr/inference_cf/r0_regions.py", "src/csasr/data/alignment.py",
           "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/inference_cf/lexical_compatibility.py", "experiments/acoustic_s1.py", "src/csasr/models/whisper.py",
           CONFIG)
FORBIDDEN = ("evaluation_units", "evaluation_panel", "candidates_existing_ctc", "role_D-dev", "transcript", "references",
             "abts0_k1/evaluation")
OPENED: set[str] = set()


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def _audit_open(event, args) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.startswith("/mnt/") or (p.startswith(str(ROOT)) and "/.git/" not in p):
            OPENED.add(p)


def firewall_check() -> list[str]:
    bad = sorted(p for p in OPENED if any(f in p for f in FORBIDDEN))
    if bad:
        raise RuntimeError(f"K1 firewall: reference-bearing paths opened: {bad[:5]}")
    return sorted(OPENED)


def archive_dir(c: dict) -> Path:
    return Path(c["outputs"]["archive"])


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


# ---- prepare (CPU) ------------------------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    from csasr.models.whisper import load_audio
    from experiments.acoustic_s1 import arr_sha
    c = cfg()
    if file_hash(ROOT / c["design"]["path"]) != c["design"]["sha256"]:
        raise ValueError("design changed")
    if file_hash(ROOT / c["panel"]["path"]) != c["panel"]["file_sha256"]:
        raise ValueError("panel file changed")
    panel = json.loads((ROOT / c["panel"]["path"]).read_text())
    if panel["identity_hash"] != c["panel"]["identity_hash"] or len(panel["rows"]) != 300:
        raise ValueError("panel identity")
    if file_hash(ROOT / c["inputs"]["r0_output_seal"]["path"]) != c["inputs"]["r0_output_seal"]["sha256"]:
        raise ValueError("R0 seal changed")
    r0seal = json.loads((ROOT / c["inputs"]["r0_output_seal"]["path"]).read_text())["files"]
    s1 = c["inputs"]["s1_config"]
    if file_hash(ROOT / s1["path"]) != s1["sha256"]:
        raise ValueError("S1 config changed")
    model = json.loads((ROOT / s1["path"]).read_text())["model"]
    model_ok = {f: file_hash(Path(model["dir"]) / f) == h for f, h in model["files"].items()} if args.hash_model else None
    if model_ok is not None and not all(model_ok.values()):
        raise ValueError(f"model files changed: {model_ok}")
    rows, b0m_equal = [], 0
    for r in panel["rows"]:
        i, uid = r["canonical_index"], r["utterance_id"]
        a = r["audio"]
        if file_hash(a["path"]) != a["full_sha256"]:
            raise ValueError(f"audio changed {uid}")
        x = load_audio(a["path"], 16000)
        rj, rn = f"results/inference_cf/r0/run1/rows/{i:03d}.json", f"results/inference_cf/r0/run1/rows/{i:03d}.npz"
        for p in (rj, rn):
            if r0seal.get(p) != file_hash(ROOT / p):
                raise ValueError(f"R0 row not sealed: {p}")
        r0 = json.loads((ROOT / rj).read_text())
        if r0["utterance_id"] != uid or r0["audio_sha256"] != a["full_sha256"] or r0["n_samples"] != len(x):
            raise ValueError(f"R0 audio identity {uid}")
        p2 = json.loads((ROOT / f"results/inference_cf/p2_A_r1_L16/rows/{i:03d}.json").read_text())
        if p2["identity"] != uid:
            raise ValueError("P2-A row order")
        b0m = p2["systems"]["B0M_L16"]
        b0m_equal += int(b0m["tokens"] == r["baseline"]["content_ids"])
        rows.append({"index": i, "utterance_id": uid, "dialogue_id": r["dialogue_id"], "audio_path": a["path"],
                     "audio_file_sha256": a["full_sha256"], "waveform_sha256": arr_sha(x), "n_samples": int(len(x)),
                     "heard_samples": int(min(len(x), 480000)), "r0_row": rj, "r0_row_sha256": r0seal[rj], "r0_arrays": rn,
                     "r0_arrays_sha256": r0seal[rn], "b0m_tokens_sha256": digest(b0m["tokens"]),
                     "b0m_terminated": b0m["terminated"], "b0m_length": len(b0m["tokens"])})
    plan = {"schema": SCHEMA, "config_sha256": file_hash(ROOT / CONFIG), "git_head_at_prepare": git_head(),
            "panel_identity_hash": c["panel"]["identity_hash"], "rows": rows, "ids": [r["utterance_id"] for r in rows],
            "sources": {s: file_hash(ROOT / s) for s in SOURCES}, "model_dir": model["dir"], "model_files": model["files"],
            "model_files_verified": model_ok, "b0m_equals_r0_baseline": b0m_equal, "references_used": False,
            "outcomes_computed": False, "created_unix": time.time()}
    plan["plan_hash"] = digest({k: v for k, v in plan.items() if k not in ("created_unix",)})
    atomic_json(ROOT / PLAN, plan)
    print(f"plan {plan['plan_hash']} rows={len(rows)} b0m==r0_baseline {b0m_equal}/300 model_verified={model_ok is not None}")


def load_plan() -> dict:
    plan = json.loads((ROOT / PLAN).read_text())
    if digest({k: v for k, v in plan.items() if k not in ("created_unix", "plan_hash")}) != plan["plan_hash"]:
        raise ValueError("plan hash")
    if plan["config_sha256"] != file_hash(ROOT / CONFIG):
        raise ValueError("config changed after plan")
    return plan


def check_sources(plan: dict) -> None:
    bad = [s for s, h in plan["sources"].items() if file_hash(ROOT / s) != h]
    if bad:
        raise ValueError(f"sources changed after plan: {bad}")


def cmd_forecast(args) -> None:
    plan = load_plan()
    L = [r["b0m_length"] for r in plan["rows"]]
    # capture: B0 (T+1) + EN (<=200 cap, forecast 1.5x B0) + AUTO (~B0) steps; score: 3 conditions x 2 words x (prefix+4+|w|)
    capture = sum((l + 1) + min(200, int(1.5 * l) + 1) + (l + 1) for l in L)
    score_max = sum(3 * 2 * (4 + l + 8) for l in L)
    out = {"decoder_steps_capture": capture, "decoder_steps_score_max": score_max,
           "encoder_calls_max": 300 * 3 + 300, "assumed_sec_per_step": [0.015, 0.03],
           "forecast_hours": [round((capture + score_max) * s / 3600, 2) for s in (0.015, 0.03)],
           "limit_hours": cfg()["compute"]["abort_if_forecast_hours_gt"]}
    out["fits"] = out["forecast_hours"][1] <= out["limit_hours"]
    atomic_json(ROOT / BASE / "forecast.json", out)
    print(json.dumps(out))


# ---- GPU helpers --------------------------------------------------------------------------------------------------

def load_model(c: dict):
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    torch.manual_seed(240924)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    if getattr(bundle.model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("attention implementation must be eager")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    if [list(x) for x in bundle.model.generation_config.alignment_heads] != c["alignment"]["heads"]:
        raise ValueError("alignment heads")
    return bundle


def save_heads(c: dict, arr: np.ndarray) -> dict:
    """Pack exactly-bf16 attention (int16 bit pattern) into the content-addressed archive."""
    import hashlib
    import io
    import torch
    from experiments.acoustic_s1 import pack
    packed = pack(torch.from_numpy(arr))
    buf = io.BytesIO()
    np.savez(buf, heads=packed)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    d = archive_dir(c)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sha}.npz"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return {"sha256": "sha256:" + sha, "path": str(p), "shape": list(arr.shape)}


def load_heads(ref: dict) -> np.ndarray:
    from experiments.acoustic_s1 import unpack
    if file_hash(ref["path"]) != ref["sha256"]:
        raise ValueError("archived attention changed")
    with np.load(ref["path"]) as z:
        return unpack(z["heads"])


def runtime_open(name: str) -> dict:
    import torch
    sys.addaudithook(_audit_open)
    torch.cuda.reset_peak_memory_stats()
    return {"stage": name, "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0), "git_head": git_head(), "start_unix": time.time()}


def runtime_close(bundle, rt: dict, name: str, w0: str) -> None:
    import torch
    from experiments.acoustic_s1 import weight_probe
    from csasr.lss.sites import assert_no_site_hooks
    assert_no_site_hooks(bundle)
    rt.update(end_unix=time.time(), peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
              peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()),
              model_grads_none=all(p.grad is None for p in bundle.model.parameters()),
              weights_unchanged_probe=weight_probe(bundle) == w0, opened_paths=firewall_check())
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    atomic_json(ROOT / RUN / f"runtime_{name}.json", rt)


# ---- capture (GPU) ------------------------------------------------------------------------------------------------

def capture_rows(bundle, plan: dict, c: dict) -> None:
    import torch
    from csasr.inference_cf.abts0_k1 import heard_frames, matched_decode
    from csasr.models.whisper import load_audio
    from experiments.acoustic_s1 import arr_sha, auto_generation_config, detect_auto, encode
    gen = bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    gauto = auto_generation_config(plan["model_dir"])
    eng = c["engine"]
    for r in plan["rows"]:
        dest = ROOT / RUN / "capture" / f"{r['index']:03d}.json"
        if dest.exists() and json.loads(dest.read_text()).get("plan_hash") == plan["plan_hash"]:
            continue
        t0 = time.time()
        x = load_audio(r["audio_path"], 16000)
        if arr_sha(x) != r["waveform_sha256"]:
            raise ValueError(f"waveform {r['utterance_id']}")
        counters = {"encoder_calls": 0, "auto_detection_queries": 0}
        enc, lin = encode(bundle, x, counters)
        F = heard_frames(r["heard_samples"])
        out = {"utterance_id": r["utterance_id"], "index": r["index"], "plan_hash": plan["plan_hash"], "encoder_lineage": lin}
        for name, prompt in (("B0", eng["cB"]), ("EN", eng["cE"])):
            d = matched_decode(bundle, enc, prompt, eos=eng["eos"], suppress=sup, begin=beg,
                               max_new_tokens=eng["max_new_tokens"], capture_layer=eng["capture_layer"])
            out[name] = {"tokens": d["tokens"], "terminated": d["terminated"], "logits_sha256": d["logits_sha256"],
                         "heads": save_heads(c, d["heads"][:, :, :F])}
        out["B0"]["equals_B0M"] = digest(out["B0"]["tokens"]) == r["b0m_tokens_sha256"] and out["B0"]["terminated"] == r["b0m_terminated"]
        auto = detect_auto(bundle, enc, gauto, counters)
        d = matched_decode(bundle, enc, auto["prompt"], eos=eng["eos"], suppress=sup, begin=beg,
                           max_new_tokens=eng["max_new_tokens"], capture_layer=eng["capture_layer"], keep_heads=False)
        out["AUTO"] = {"tokens": d["tokens"], "terminated": d["terminated"], "prompt": auto["prompt"],
                       "detection_agrees_lowest_id": auto["agrees"]}
        out["counters"], out["elapsed_sec"] = counters, time.time() - t0
        atomic_json(dest, out)
        del enc
        torch.cuda.empty_cache()
        print(f"capture {r['index'] + 1}/300 {r['utterance_id']} B0M={'=' if out['B0']['equals_B0M'] else 'DIFF'} "
              f"{out['elapsed_sec']:.1f}s", flush=True)


# ---- select (CPU) -------------------------------------------------------------------------------------------------

def tokenizer_ctx(plan: dict):
    from transformers import WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    tok = WhisperTokenizer.from_pretrained(plan["model_dir"], local_files_only=True)
    return tok, {v: k for k, v in bytes_to_unicode().items()}


def token_bytes_list(tok, byte_decoder, ids) -> list[bytes]:
    from csasr.inference_cf.core_r2 import token_bytes
    return [token_bytes(tok, i, byte_decoder) for i in ids]


def _timing_job(a: tuple) -> tuple:
    """Worker: frozen DTW timing of one branch from its archived cached attention (pure numpy / Python)."""
    from csasr.inference_cf.abts0_k1 import token_intervals
    ref, n, term, heard = a
    return ref["sha256"], token_intervals(load_heads(ref), n, term, heard)


def select_rows(plan: dict, c: dict, workers: int = 1) -> dict:
    from transformers import GenerationConfig
    from csasr.inference_cf import abts0_k1 as K
    from csasr.models.whisper import load_audio
    tok, bdec = tokenizer_ctx(plan)
    gen = GenerationConfig.from_pretrained(plan["model_dir"], local_files_only=True)
    sup = list(gen.suppress_tokens or [])
    eos = c["engine"]["eos"]
    s1cfg = json.loads((ROOT / c["inputs"]["s1_config"]["path"]).read_text())
    caps = []
    for r in plan["rows"]:
        cap = json.loads((ROOT / RUN / "capture" / f"{r['index']:03d}.json").read_text())
        if cap["plan_hash"] != plan["plan_hash"]:
            raise ValueError("stale capture row")
        caps.append(cap)
    jobs = [(cap[b]["heads"], len(cap[b]["tokens"]), cap[b]["terminated"], r["heard_samples"])
            for cap, r in zip(caps, plan["rows"]) for b in ("B0", "EN")]
    timings = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                timings = dict(ex.map(_timing_job, jobs, chunksize=4))
        except OSError as exc:                       # e.g. no POSIX semaphores (/dev/shm full): identical serial path
            print(f"select: process pool unavailable ({exc!r}); serial DTW", flush=True)
    if timings is None:
        timings = dict(map(_timing_job, jobs))
    out_rows = []
    for cap, r in zip(caps, plan["rows"]):
        heard = r["heard_samples"]
        x = load_audio(r["audio_path"], 16000)
        with np.load(ROOT / r["r0_arrays"]) as z:
            track = z["track_heard_intervals"].astype(np.int64)
        b0_heads, en_heads = load_heads(cap["B0"]["heads"]), load_heads(cap["EN"]["heads"])
        b0 = K.content_units(token_bytes_list(tok, bdec, cap["B0"]["tokens"]))
        en = K.content_units(token_bytes_list(tok, bdec, cap["EN"]["tokens"]))
        b0_t, en_t = timings[cap["B0"]["heads"]["sha256"]], timings[cap["EN"]["heads"]["sha256"]]
        row = {"index": r["index"], "utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "heard": heard,
               "b0_timing_status": b0_t["status"], "en_timing_status": en_t["status"]}
        pools = K.structural_pools(en=en, en_timing=en_t, b0=b0, b0_timing=b0_t, heard=heard, b0_heads=b0_heads,
                                   track=track, x=x, s1cfg=s1cfg)
        shuf = K.structural_pools(en=en, en_timing=en_t, b0=b0, b0_timing=b0_t, heard=heard, b0_heads=b0_heads,
                                  track=K.shift_track(track, heard), x=x, s1cfg=s1cfg)
        ch = pools["chosen"]
        if ch is not None:
            c_ids = tok.encode(" " + ch["surface"], add_special_tokens=False)
            b_ids = tok.encode(ch["b_surface"], add_special_tokens=False)
            ch = dict(ch, c_ids=c_ids, b_ids=b_ids,
                      encoding_valid=K.valid_pieces(c_ids, eos, sup) and K.valid_pieces(b_ids, eos, sup),
                      prefix=cap["B0"]["tokens"][:ch["t"]])
        row.update(U0=pools["U0"], proposals=pools["proposals"], chosen=ch,
                   shuffled_chosen=shuf["chosen"], shuffled_proposals=[p["status"] for p in shuf["proposals"]],
                   b0_units=[dict(u.to_dict(), interval=K.unit_interval(u, b0_t)) for u in b0["units"]],
                   en_units=[dict(u.to_dict(), interval=K.unit_interval(u, en_t)) for u in en["units"]])
        out_rows.append(row)
    sel = {"schema": SCHEMA + "_selection", "plan_hash": plan["plan_hash"], "rows": out_rows, "references_used": False}
    atomic_json(ROOT / SELECTION, sel)
    seal = {"selection_sha256": file_hash(ROOT / SELECTION), "plan_hash": plan["plan_hash"],
            "capture_files": {f"{RUN}/capture/{r['index']:03d}.json": file_hash(ROOT / RUN / "capture" / f"{r['index']:03d}.json")
                              for r in plan["rows"]},
            "chosen": sum(1 for r in out_rows if r["chosen"] is not None), "sealed_unix": time.time(),
            "note": "sealed before any acoustic scoring and before any reference access"}
    atomic_json(ROOT / CAND_SEAL, seal)
    print(f"selection sealed: chosen {seal['chosen']}/300 {seal['selection_sha256']}")
    return sel


# ---- score (GPU) --------------------------------------------------------------------------------------------------

def score_rows(bundle, plan: dict, c: dict) -> None:
    import torch
    from csasr.inference_cf import abts0_k1 as K
    from csasr.inference_cf.lexical_compatibility import hard_mask
    from csasr.models.whisper import load_audio
    from experiments.acoustic_s1 import arr_sha, encode
    seal = json.loads((ROOT / CAND_SEAL).read_text())
    if seal["selection_sha256"] != file_hash(ROOT / SELECTION):
        raise ValueError("selection changed after seal")
    sel = json.loads((ROOT / SELECTION).read_text())
    gen = bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    cB, layer = c["engine"]["cB"], c["engine"]["capture_layer"]
    for row, r in zip(sel["rows"], plan["rows"]):
        ch = row["chosen"]
        dest = ROOT / RUN / "scores" / f"{r['index']:03d}.json"
        if ch is None or (dest.exists() and json.loads(dest.read_text()).get("selection_sha256") == seal["selection_sha256"]):
            continue
        out = {"utterance_id": r["utterance_id"], "selection_sha256": seal["selection_sha256"], "status": None}
        if not ch["encoding_valid"]:
            out["status"] = "INVALID_ENCODING"
            atomic_json(dest, out)
            continue
        cap = json.loads((ROOT / RUN / "capture" / f"{r['index']:03d}.json").read_text())
        x = load_audio(r["audio_path"], 16000)
        counters = {"encoder_calls": 0, "masked_encoder_calls": 0}
        conds = {"x": x, "target": hard_mask(x, *ch["target"]["bounds"]), "off": hard_mask(x, *ch["off_target"]["bounds"])}
        F, detail = {}, {}
        for k, xa in conds.items():
            if k != "x":
                a, b = ch["target" if k == "target" else "off_target"]["bounds"]
                if xa[:a].tobytes() != x[:a].tobytes() or xa[b:].tobytes() != x[b:].tobytes():
                    raise RuntimeError("mask changed bytes outside crop")
            enc, lin = encode(bundle, xa, counters, "encoder_calls" if k == "x" else "masked_encoder_calls")
            F[k], detail[k] = {}, {"lineage": lin, "waveform_sha256": arr_sha(xa)}
            for w, ids in (("c", ch["c_ids"]), ("b", ch["b_ids"])):
                s = K.score_word(bundle, enc, cB, ch["prefix"], ids, suppress=sup, begin=beg, capture_layer=layer)
                F[k][w] = s["F"]
                detail[k][w] = s
            del enc
        qx = detail["x"]
        out["identity"] = {
            "x_query_logits_equal_capture": qx["c"]["query_logits_sha256"] == cap["B0"]["logits_sha256"][ch["t"]]
                                            and qx["b"]["query_logits_sha256"] == cap["B0"]["logits_sha256"][ch["t"]],
            "x_query_argmax_is_b0_token": qx["c"]["query_argmax"] == cap["B0"]["tokens"][ch["t"]],
            "x_encoder_equals_capture": detail["x"]["lineage"] == cap["encoder_lineage"]}
        out.update(F=F, detail=detail, rule=K.acoustic_rule(F), counters=counters)
        out["status"] = "SCORED" if all(out["identity"].values()) else "INVALID_ENGINE_IDENTITY"
        atomic_json(dest, out)
        torch.cuda.empty_cache()
        print(f"score {r['utterance_id']} {out['status']} {out['rule'].get('status')}", flush=True)


# ---- stage drivers ------------------------------------------------------------------------------------------------

def gpu_stage(name: str) -> None:
    from experiments.acoustic_s1 import weight_probe
    plan, c = load_plan(), cfg()
    check_sources(plan)
    rt = runtime_open(name)
    bundle = load_model(c)
    w0 = weight_probe(bundle)
    if name == "capture":
        capture_rows(bundle, plan, c)
    if name == "score":
        score_rows(bundle, plan, c)
    runtime_close(bundle, rt, name, w0)


def cmd_seal(args) -> None:
    plan = load_plan()
    files = {}
    for p in sorted((ROOT / RUN).rglob("*.json")):
        files[str(p.relative_to(ROOT))] = file_hash(p)
    objects = {}
    for p in sorted((ROOT / RUN / "capture").glob("*.json")):
        d = json.loads(p.read_text())
        for b in ("B0", "EN"):
            objects[d[b]["heads"]["path"]] = d[b]["heads"]["sha256"]
    bad = [p for p, h in objects.items() if file_hash(p) != h]
    if bad:
        raise ValueError(f"archive objects changed: {bad[:3]}")
    seal = {"schema": SCHEMA + "_output_seal", "plan_hash": plan["plan_hash"], "files": files, "archive_objects": objects,
            "references_used": False, "git_head": git_head(), "created_unix": time.time()}
    seal["seal_hash"] = digest({k: v for k, v in seal.items() if k != "created_unix"})
    atomic_json(ROOT / OUTPUT_SEAL, seal)
    print(f"sealed {len(files)} files + {len(objects)} archive objects {seal['seal_hash']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "forecast", "capture", "select", "score", "seal"))
    ap.add_argument("--workers", type=int, default=16, help="select: DTW worker processes (CPU)")
    ap.add_argument("--hash-model", action="store_true", help="prepare: hash all pinned model files (incl. weights)")
    args = ap.parse_args()
    if args.stage == "prepare":
        cmd_prepare(args)
    elif args.stage == "forecast":
        cmd_forecast(args)
    elif args.stage == "select":
        plan = load_plan()
        check_sources(plan)
        sys.addaudithook(_audit_open)
        select_rows(plan, cfg(), workers=args.workers)
        firewall_check()
    elif args.stage == "seal":
        cmd_seal(args)
    else:
        gpu_stage(args.stage)


if __name__ == "__main__":
    main()
