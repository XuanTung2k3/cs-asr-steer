#!/usr/bin/env python
"""P2-SEQ runner (separate inference_cf development screen). Frozen contract:
docs/inference_cf/P2_SEQ_SPEC.md, P2_SEQ_CODEX_DESIGN.md, configs/inference_cf/p2_seq.json (660a619).

``seq_decode`` is the one additive sequence driver: the historical cached B/E/S topology of
``inference_cf_cached.cached_decode`` (same Branch.step calls, frozen R2 gate g = E*R_B on RAW B logits,
same ``_edit_hook``, same processed argmax / suppression / EOS / 200 cap), with the direction supplied by the
unchanged reference-free D2 READOUT provider instead of hE - hB:

* at t >= 1 (alpha > 0 only) the PRE-step B cache is snapshotted with ``readout.clone_scratch`` before the clean
  B step; after the gate, ``readout_direction`` runs on that snapshot ONLY for an eligible positive dose, and its
  scratch logits/site must equal the clean B step bitwise;
* S0 (alpha = 0) executes the identical B/E/S forwards and zero-dose hook, no snapshot / D2;
* passive recorders before and after the S hook keep the pre/post edited S site for energy audit.

Modes: ``prepare`` (CPU reuse proof + resolved inputs, before outcomes), ``manifest`` (CPU), ``run`` (GPU).
No reference, evaluator, TTA or adaptation input anywhere in this module.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "p2_seq_v1"
CONFIG = "configs/inference_cf/p2_seq.json"
PANEL = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
HIST = "results/inference_cf/p2_A_r1_L16"
BASE = "results/inference_cf/p2seq"
REUSE = f"{BASE}/reuse_audit.json"
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
LAYER = 16
ALPHA_STEER = 2.0
MAX_NEW = 200


class Stack:
    """Enter context managers in order (passive pre-recorder, then the edit hook)."""

    def __init__(self, *ctx):
        self.ctx = [c for c in ctx if c is not None]

    def __enter__(self):
        for c in self.ctx:
            c.__enter__()
        return self

    def __exit__(self, *exc):
        for c in reversed(self.ctx):
            c.__exit__(None, None, None)
        return False


def seq_decode(bundle, *, waveform, encoded, partition, null_probs, language_ids, alpha: float, lid_cache: dict,
               lid_key: str, counters: dict, nfp: int = 4, max_new_tokens: int = MAX_NEW, cB=CB, cE=CE,
               language_token_ids=(50259, 50260)) -> dict:
    import torch
    import experiments.inference_cf_cached as cached
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.core_p1 import processed_argmax, selected_gate
    from csasr.inference_cf.core_r2 import (conflict_from_logits, local_support, max_attention_window,
                                            prefix_utf8_complete)
    from csasr.inference_cf.directions import DirectionContext, ReadoutDirection
    from csasr.inference_cf.readout import clone_scratch
    from csasr.lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    if alpha not in (0.0, ALPHA_STEER):
        raise ValueError("P2-SEQ supports alpha 0 or 2 only")
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = bundle.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    en_id, zh_id = language_token_ids
    d2p = ReadoutDirection(bundle, layer=LAYER, suppress=suppress, begin=begin, partition=partition)
    B, E, S = cached.Branch(bundle, encoded, cB, "B"), cached.Branch(bundle, encoded, cE, "E"), cached.Branch(bundle, encoded, cB, "S")
    new_b, new_e = list(cB), list(cE)
    tokens, steps, vecs = [], [], {}
    terminated, first_edit = "cap", None
    for t in range(max_new_tokens):
        assert_no_site_hooks(bundle)
        L = B.length
        snap = None
        if alpha > 0 and t >= 1:
            snap = clone_scratch(B.cache, encoded)                       # PRE-step snapshot (ordinary tensors)
        logits_b, heads, h_b = B.step(new_b, capture_layer=LAYER, attention=True)
        E.step(new_e, capture_layer=LAYER, attention=False)
        counters["forwards"] += 2
        query = S.length + len(new_b) - 1
        st = {"t": t, "query": query, "fb": None, "E": None, "R_B": None, "g": 0.0, "dose": 0.0, "window": None,
              "lid_hit": None, "d2_called": False, "dir": None, "dir_reason": None}
        reasons = []
        if not prefix_utf8_complete(tok, tokens, byte_decoder):
            reasons.append("mid_character")
        window = support = baseline = None
        try:
            window = max_attention_window(heads.numpy(), len(waveform))
            st["window"] = [window.start_frame, window.end_frame, window.start_sample, window.end_sample]
        except Exception as exc:
            reasons.append(r2._reason(exc, "localizer_fail"))
        try:
            baseline = conflict_from_logits(logits_b, partition)        # RAW B logits
        except Exception as exc:
            reasons.append(r2._reason(exc, "baseline_provider_fail"))
        if not reasons:
            try:
                key = (lid_key, window.start_sample, window.end_sample)
                st["lid_hit"] = key in lid_cache
                if key in lid_cache:
                    counters["lid_cache_hits"] += 1
                else:
                    lid_cache[key] = cached.native_lid(bundle, waveform[key[1]:key[2]], language_ids)
                    counters["lid_calls"] += 1
                probs = lid_cache[key]
                support = local_support(probs[en_id], probs[zh_id], null_probs[en_id], null_probs[zh_id])
            except Exception as exc:
                reasons.append(r2._reason(exc, "local_support_fail"))
        priority = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")
        st["fb"] = next((r for r in priority if r in reasons), None)
        if st["fb"] is None:
            st["g"] = float(selected_gate(support, baseline))
            st["E"], st["R_B"] = support["E"], baseline["R"]
        st["dose"] = float(alpha) * st["g"]
        d = None
        if st["dose"] > 0 and query >= nfp:
            res = d2p(DirectionContext(b_cache_pre_step=snap[0], encoded=snap[1], new_tokens=list(new_b), start=L, step=t))
            counters["d2_calls"] += 1
            counters["autograd_calls"] += res.counters.get("autograd_calls", 0)
            st["d2_called"] = True
            st["d2_logits_bitwise_B"] = bool(torch.equal(res.extras["logits"], logits_b))
            st["d2_site_bitwise_B"] = bool(torch.equal(res.extras["site"], h_b))
            st["dir"], st["dir_reason"] = res.status, res.reason
            d = res.direction
        snap = None
        applied = bool(st["dose"] > 0 and d is not None and query >= nfp)
        st["applied"] = applied
        hook = cached._edit_hook(bundle, LAYER, query, alpha, st["g"] if applied else 0.0, d if applied else None, nfp)
        pre = DecoderPostCrossAttnRecorder(bundle, [LAYER], keep_last_only=True)
        logits_s, _, h_post = S.step(new_b, capture_layer=LAYER, attention=True, hook=Stack(pre, hook))
        counters["forwards"] += 1
        assert_no_site_hooks(bundle)
        audit = hook.records[-1].to_dict() if hook.records else None
        edited = bool(applied and audit and audit["steered"] and audit["edit_norm"] > 0)
        st["edited"] = edited
        st["edit_norm"] = audit["edit_norm"] if edited else 0.0
        st["pre_norm"] = audit["pre_norm"] if audit else None
        st["post_norm"] = audit["post_norm"] if audit else None
        if edited:
            first_edit = t if first_edit is None else first_edit
            vecs[f"t{t}_pre"] = pre.states[LAYER][0, -1].float().cpu().numpy()
            vecs[f"t{t}_post"] = h_post.numpy()
            vecs[f"t{t}_d2"] = d.numpy()
        if not edited:     # zero effective gate: S site unchanged by the hook (identity to S, not to B)
            st["noedit_site_identity"] = bool(torch.equal(pre.states[LAYER][0, -1].float().cpu(), h_post))
        st["S_bitwise_B"] = bool(torch.equal(logits_s, logits_b))
        st["unsteered_next"] = processed_argmax(logits_b, t, suppress, begin)
        st["next"] = processed_argmax(logits_s, t, suppress, begin)
        steps.append(st)
        nxt = st["next"]
        if nxt == eos:
            terminated = "eos"
            break
        tokens.append(nxt)
        new_b, new_e = [nxt], [nxt]
    lineage = (B.fed[len(cB):] == tokens[:len(B.fed) - len(cB)] and E.fed[len(cE):] == B.fed[len(cB):] and S.fed == B.fed
               and B.positions == list(range(len(B.fed))) and S.positions == list(range(len(S.fed))))
    return {"tokens": tokens, "text": tok.decode(tokens, skip_special_tokens=True), "terminated": terminated, "steps": steps,
            "lineage_ok": bool(lineage), "distinct_caches": len({id(B.cache), id(E.cache), id(S.cache)}) == 3,
            "first_edit_t": first_edit, "alpha": alpha}, vecs


# ---- CPU prepare / reuse proof ---------------------------------------------------------------------------

def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_prepare(args) -> None:
    """Pre-outcome AUTO reuse proof (engineering only) + resolved panel inputs; sealed before any decode."""
    import datetime
    cfg = json.loads((ROOT / CONFIG).read_text())
    panel = json.loads((ROOT / PANEL).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    hist_panel = json.loads((ROOT / HIST / "panel.json").read_text())
    audio = {r["utterance_id"]: r for r in hist_panel["rows"]}
    hman = json.loads((ROOT / HIST / "manifest.json").read_text())
    hrt = json.loads((ROOT / HIST / "runtime.json").read_text())
    checks, rows = {}, []
    checks["panel_bytes"] = sha_file(ROOT / PANEL) == cfg["panel"]["sha256"]
    checks["panel_ids_order"] = ids == cfg["panel"]["ids"] and len(set(ids)) == 100
    checks["historical_sources_unchanged"] = all(
        "sha256:" + sha_file(p) == h for p, h in hman["sources"].items())
    sealed = {r["utterance_id"]: r for r in cfg["reuse"]["rows"]}
    ok_rows = True
    for u in ids:
        s = sealed[u]
        row = json.loads((ROOT / s["path"]).read_text())
        a = row["systems"].get("B0_AUTO")
        fine = (sha_file(ROOT / s["path"]) == s["sha256"] and row["identity"] == u and row["status"] == "ok"
                and row["manifest_hash"] == hman["manifest_hash"] and a is not None and "tokens" in a and "text" in a)
        ok_rows &= fine
        audio_ok = sha_file(audio[u]["audio_path"]) == audio[u]["audio_sha256"]
        ok_rows &= audio_ok
        rows.append({"utterance_id": u, "row": s["path"], "row_sha256": s["sha256"], "audio_path": audio[u]["audio_path"],
                     "audio_sha256": audio[u]["audio_sha256"], "audio_ok": audio_ok, "auto_terminated": a["terminated"] if a else None})
    checks["sealed_rows_audio"] = bool(ok_rows)
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    checks["environment_same"] = r2m["environment"] == {**r2m["environment"], **{"python": "3.11.9", "torch": "2.10.0+cu128",
                                                                                 "transformers": "4.57.6"}}
    import platform
    import torch
    import transformers
    checks["current_environment"] = (platform.python_version(), torch.__version__, transformers.__version__) == ("3.11.9", "2.10.0+cu128", "4.57.6")
    site = Path("/home/tungnx/miniconda3/envs/acl1/lib/python3.11/site-packages")
    libs = {}
    for pat in ("torch-*.dist-info", "transformers-*.dist-info", "tokenizers-*.dist-info", "numpy-*.dist-info",
                "safetensors-*.dist-info", "soundfile-*.dist-info", "scipy-*.dist-info"):
        for d in site.glob(pat):
            libs[d.name] = max(f.stat().st_mtime for f in d.iterdir())
    checks["libraries_predate_historical_run"] = bool(libs) and all(v < hrt["start_unix"] for v in libs.values())
    model = Path("/mnt/data/tungnx/whisper-large-v3")
    checks["model_files_predate_historical_run"] = max(f.stat().st_mtime for f in model.iterdir()) < hrt["start_unix"]
    checks["weights_sha256"] = r2m["model_revision"] == "sha256:" + cfg["frozen"]["weights_sha256"]
    p2src = (ROOT / "experiments/inference_cf_p2.py").read_text()
    checks["auto_call_exact"] = ("bundle.model.generate(**inputs, task=\"transcribe\", language=None, do_sample=False,\n"
                                 "                                         num_beams=1, max_new_tokens=max_new_tokens,\n"
                                 "                                         condition_on_prev_tokens=False)") in p2src
    checks["auto_postprocess_exact"] = "auto_content = auto[:auto.index(eos)] if eos in auto else auto" in p2src
    checks["whisper_io_unchanged_since_historical_commit"] = sha_file(ROOT / "src/csasr/models/whisper.py") == \
        hashlib.sha256(subprocess.run(["git", "show", f"{hman['git_commit']}:src/csasr/models/whisper.py"], cwd=ROOT,
                                      capture_output=True).stdout).hexdigest()
    decision = "REUSE_AUTO_ALL_100" if all(checks.values()) else "COMPUTE_AUTO_ALL_100"
    doc = {"schema": SCHEMA + "_reuse_audit", "decision_AUTO": decision, "S0": "NEW (matched zero-dose sequence driver)",
           "S2": "NEW", "checks": checks, "historical_manifest_hash": hman["manifest_hash"], "historical_job": hrt["job_id"],
           "historical_start_unix": hrt["start_unix"],
           "library_mtimes_utc": {k: datetime.datetime.fromtimestamp(v, datetime.timezone.utc).isoformat() for k, v in libs.items()},
           "rows": rows, "outcomes_computed": False, "created_unix": time.time()}
    doc["reuse_hash"] = digest(doc)
    out = ROOT / REUSE
    if out.exists():
        raise FileExistsError("reuse audit exists; never overwrite")
    atomic_json(out, doc)
    print(json.dumps({"decision_AUTO": decision, "failed": [k for k, v in checks.items() if not v]}))


SOURCES = ("docs/inference_cf/P2_SEQ_SPEC.md", "docs/inference_cf/P2_SEQ_CODEX_DESIGN.md", CONFIG, PANEL, REUSE,
           "docs/inference_cf/P2_SEQ_PRE_RUN_AUDIT.md", "experiments/inference_cf_p2seq.py", "experiments/inference_cf_p2seq_analyze.py",
           "experiments/inference_cf_p2seq_audit.py", "slurm/inference_cf_p2seq.sbatch", "experiments/inference_cf_cached.py",
           "experiments/inference_cf_p0_r2.py", "src/csasr/inference_cf/readout.py", "src/csasr/inference_cf/directions.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py",
           "src/csasr/models/hooks.py", "src/csasr/models/whisper.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = [f"{BASE}/prerun_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-SEQ sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / extra[0]).read_text())["verdict"] != "PASS_TO_P2_SEQ":
        raise ValueError("pre-run audit did not pass")
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    part = tokenizer_partition(proc.tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    reuse = json.loads((ROOT / REUSE).read_text())
    man = {"schema": SCHEMA, "stage": "run", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config": CONFIG, "config_hash": digest(cfg), "panel_sha256": sha_file(ROOT / PANEL), "ids": cfg["panel"]["ids"],
           "partition_hash": part["hash"], "suppression": {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])},
           "feature_extractor": proc.feature_extractor.to_dict(), "reuse_hash": reuse["reuse_hash"],
           "decision_AUTO": reuse["decision_AUTO"], "systems_new": ["S0_B0_FORCED", "S2_STEER"] + (
               [] if reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100" else ["S1_B0_AUTO"]),
           "audio": {r["utterance_id"]: {"path": r["audio_path"], "sha256": r["audio_sha256"]} for r in reuse["rows"]},
           "role": "D-dev-select", "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(extra)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"], "systems_new": man["systems_new"]}))


def run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_audio, load_whisper
    from csasr.utils.config import load_config
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    t_setup = time.time()
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.requires_grad_(False)
    part = tokenizer_partition(bundle.processor.tokenizer)
    if part["hash"] != m["partition_hash"]:
        raise ValueError("partition hash")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != 4:
        raise ValueError("forced prefix")
    native = bundle.model.generation_config.lang_to_id
    language_ids = tuple(sorted(set(int(x) for x in native.values())))
    null_probs = cached.native_lid(bundle, np.zeros(480000, dtype=np.float32), language_ids)
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running",
               "systems": {k: {"wall_sec": 0.0, "peak_alloc": 0, "peak_reserved": 0, "utterances": 0, "steps": 0}
                           for k in ("S0", "S2", "S1")}}
    counters = {k: {"forwards": 0, "lid_calls": 0, "lid_cache_hits": 0, "d2_calls": 0, "autograd_calls": 0} for k in ("S0", "S2")}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    compute_auto = "S1_B0_AUTO" in m["systems_new"]
    failures = 0
    for i, uid in enumerate(m["ids"]):
        t0 = time.time()
        res = {"systems": {}}
        try:
            path = m["audio"][uid]["path"]
            if sha_file(path) != m["audio"][uid]["sha256"]:
                raise ValueError("audio bytes changed")
            wav = load_audio(path, bundle.sample_rate)
            inputs = batch_model_inputs(bundle, [path])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
            lid_cache: dict = {}
            vec_all = {}
            for name, alpha in (("S0", 0.0), ("S2", ALPHA_STEER)):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                ts = time.time()
                with torch.inference_mode():
                    r, vecs = seq_decode(bundle, waveform=wav, encoded=enc, partition=part, null_probs=null_probs,
                                         language_ids=language_ids, alpha=alpha, lid_cache=lid_cache, lid_key=uid,
                                         counters=counters[name], nfp=nfp)
                torch.cuda.synchronize()
                sysrt = runtime["systems"][name]
                sysrt["wall_sec"] += time.time() - ts
                sysrt["peak_alloc"] = max(sysrt["peak_alloc"], torch.cuda.max_memory_allocated())
                sysrt["peak_reserved"] = max(sysrt["peak_reserved"], torch.cuda.max_memory_reserved())
                sysrt["utterances"] += 1
                sysrt["steps"] += len(r["steps"])
                res["systems"][name] = r
                vec_all.update({f"{name}_{k}": v for k, v in vecs.items()})
                del r
            if compute_auto:
                ts = time.time()
                with torch.inference_mode():
                    auto = bundle.model.generate(**inputs, task="transcribe", language=None, do_sample=False, num_beams=1,
                                                 max_new_tokens=MAX_NEW, condition_on_prev_tokens=False)[0].tolist()
                eos = bundle.processor.tokenizer.eos_token_id
                content = auto[:auto.index(eos)] if eos in auto else auto
                res["systems"]["S1"] = {"tokens": content, "raw": auto, "terminated": "eos" if len(content) < MAX_NEW else "cap",
                                        "text": bundle.processor.tokenizer.decode(content, skip_special_tokens=True)}
                runtime["systems"]["S1"]["wall_sec"] += time.time() - ts
                runtime["systems"]["S1"]["utterances"] += 1
            np.savez_compressed(out / "rows" / f"{i:03d}.npz", **vec_all)
            res["vectors_sha256"] = file_hash(out / "rows" / f"{i:03d}.npz")
            res["status"] = "ok"
            runtime.setdefault("audio_sec", 0.0)
            runtime["audio_sec"] += min(len(wav) / 16000, 30.0)
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc(), "systems": res.get("systems", {})}
        res.update(identity=uid, manifest_hash=m["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(out / "rows" / f"{i:03d}.json", res)
        print(f"P2-SEQ {i + 1}/100 {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
        runtime["counters"] = counters
        atomic_json(out / "runtime.json", runtime)
    if any(p.grad is not None for p in bundle.model.parameters()):
        raise RuntimeError("parameter gradient present")
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if failures:
        raise SystemExit(f"{failures} failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
