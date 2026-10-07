#!/usr/bin/env python
"""P2-TTA0 runner (separate inference_cf development screen). Frozen contract:
docs/inference_cf/P2_TTA0_SPEC.md, P2_TTA0_CODEX_DESIGN.md, configs/inference_cf/p2_tta0.json (eb0da2a).

Per utterance of the frozen panel20: one frozen encoder pass; theta0 ordinary forced-ZH decode must reproduce the
sealed P2-SEQ S0 tokens/termination; then A1 GREEDY-EM (teacher y_B) and A2 AUTO-CONSISTENCY (teacher y_A), each
from fresh theta0 fp32 masters + fresh AdamW, 2 updates / 3 loss forwards, final ordinary forced-ZH decode with the
effective bf16 LN values, then exact bf16 restore (finally) and bitwise verification. On the FIRST utterance the
independent auditor's live loss/gradient check runs at theta0 for both objectives (no optimizer update).

Modes: ``prepare`` (CPU reuse proof + pseudo-teacher seal, before outcomes), ``manifest`` (CPU), ``run`` (GPU).
No reference, evaluator, steering, D2 or LID input anywhere in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2_tta0_v1"
FREEZE = "eb0da2a06cdffba4395a58215458d682bbba94e9"
CONFIG = "configs/inference_cf/p2_tta0.json"
PANEL = "docs/inference_cf/P2_TTA0_PANEL20.json"
PARENT = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
SEQ = "results/inference_cf/p2seq"
BASE = "results/inference_cf/p2tta0"
SEAL = f"{BASE}/pseudo_sealed.json"
CB = [50258, 50260, 50360, 50364]
MAX_NEW = 200


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audio_fingerprint_64k(path) -> str:
    """Repository role-manifest audio identity: size-prefixed csasr.utils.hashing.sha256_file, first 64 KiB."""
    from csasr.utils.hashing import sha256_file
    return sha256_file(path, max_bytes=65536)


def audio_full_sha256(path) -> str:
    return sha_file(path)


# ---- CPU prepare: reuse proof + pseudo-teacher seal --------------------------------------------------------

def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.episodic_tta import clean_teacher, special_ids, valid_mask
    cfg = json.loads((ROOT / CONFIG).read_text())
    panel = json.loads((ROOT / PANEL).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    parent = json.loads((ROOT / PARENT).read_text())
    checks = {}
    checks["panel_bytes"] = sha_file(ROOT / PANEL) == cfg["panel"]["byte_sha256"]
    checks["parent_bytes"] = sha_file(ROOT / PARENT) == panel["parent_byte_sha256"] == cfg["source_sha256"][PARENT]
    first, seen = [], set()
    for r in parent["rows"]:
        if r["dialogue_id"] not in seen:
            seen.add(r["dialogue_id"])
            first.append(r["utterance_id"])
    checks["panel_first_per_dialogue"] = ids == first == cfg["panel"]["ids"] and len(set(ids)) == 20 and \
        len({r["dialogue_id"] for r in panel["rows"]}) == 20
    checks["anchors"] = all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items())
    seqm = json.loads((ROOT / SEQ / "run1/manifest.json").read_text())
    seq_audit = json.loads((ROOT / SEQ / "run1_audit.json").read_text())
    reuse = json.loads((ROOT / SEQ / "reuse_audit.json").read_text())
    checks["p2seq_audit_pass"] = seq_audit["verdict"] == "P2_SEQ_AUDIT: PASS"
    checks["p2seq_reuse_auto"] = reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100" and reuse["outcomes_computed"] is False \
        and all(reuse["checks"].values())
    reuse_rows = {r["utterance_id"]: r for r in reuse["rows"]}
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = proc.tokenizer
    eos = tok.eos_token_id
    specials = special_ids(tok)
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    anchor = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1/manifest.json").read_text())["model"]["files"]
    mh = prep.model_hashes()
    checks["model_files_match_anchor"] = mh == anchor and mh == seqm["model"]["files"]
    env = prep.environment()
    checks["environment_same_as_p2seq"] = all(env.get(k) == seqm["environment"].get(k) for k in ("python", "torch", "transformers"))
    checks["whisper_io_unchanged"] = file_hash(ROOT / "src/csasr/models/whisper.py") == seqm["sources"]["src/csasr/models/whisper.py"]
    checks["cached_branch_unchanged"] = file_hash(ROOT / "experiments/inference_cf_cached.py") == seqm["sources"]["experiments/inference_cf_cached.py"]
    checks["suppression_same_as_p2seq"] = {"suppress": suppress, "begin": begin} == seqm["suppression"]
    checks["feature_extractor_same"] = proc.feature_extractor.to_dict() == seqm["feature_extractor"]
    rows, ok = [], True
    for spec in cfg["reuse"]["rows"]:
        u = spec["utterance_id"]
        fr = json.loads((ROOT / spec["forced_row"]).read_text())
        ar = json.loads((ROOT / spec["auto_row"]).read_text())
        f, a = fr["systems"]["S0"], ar["systems"]["B0_AUTO"]
        audio = seqm["audio"][u]
        fine = (sha_file(ROOT / spec["forced_row"]) == spec["forced_row_sha256"] and sha_file(ROOT / spec["auto_row"]) == spec["auto_row_sha256"]
                and fr["identity"] == u and fr["status"] == "ok" and fr["manifest_hash"] == seqm["manifest_hash"]
                and ar["identity"] == u and ar["status"] == "ok" and reuse_rows[u]["row"] == spec["auto_row"]
                and reuse_rows[u]["row_sha256"] == spec["auto_row_sha256"] and reuse_rows[u]["audio_ok"]
                and audio_fingerprint_64k(audio["path"]) == audio["sha256"] == reuse_rows[u]["audio_sha256"])
        y_B = clean_teacher(f["tokens"], CB, eos, specials)
        y_A = clean_teacher(a["tokens"], CB, eos, specials)
        fine &= y_B == f["tokens"] and y_A == a["tokens"] and len(y_B) <= MAX_NEW and len(y_A) <= MAX_NEW
        ok &= bool(fine)
        mB, mA = valid_mask(y_B, suppress, begin, specials, eos), valid_mask(y_A, suppress, begin, specials, eos)
        rows.append({"utterance_id": u, "dialogue_id": next(r["dialogue_id"] for r in panel["rows"] if r["utterance_id"] == u),
                     "audio_path": audio["path"], "audio_fingerprint_64k_sizeprefixed": audio["sha256"],
                     "audio_full_sha256": audio_full_sha256(audio["path"]),
                     "forced_row": spec["forced_row"], "forced_row_sha256": spec["forced_row_sha256"],
                     "auto_row": spec["auto_row"], "auto_row_sha256": spec["auto_row_sha256"],
                     "y_B": y_B, "y_B_terminated": f["terminated"], "y_B_valid_mask": mB,
                     "y_A": y_A, "y_A_terminated": a["terminated"], "y_A_valid_mask": mA,
                     "y_B_text": f["text"], "y_A_text": a["text"], "row_ok": bool(fine)})
    checks["reuse_rows"] = bool(ok)
    doc = {"schema": SCHEMA + "_pseudo_seal", "decision_B0_FORCED": "REUSE_P2SEQ_S0_20", "decision_AUTO": "REUSE_HISTORICAL_AUTO_20",
           "checks": checks, "ids": ids, "p2seq_manifest_hash": seqm["manifest_hash"], "p2seq_reuse_hash": reuse["reuse_hash"],
           "model_files": mh, "environment": env, "suppression": {"suppress": suppress, "begin": begin}, "eos": eos,
           "special_ids_count": len(specials), "prompt_cB": CB, "rows": rows, "outcomes_computed": False,
           "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / SEAL
    if out.exists():
        raise FileExistsError("pseudo seal exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: reuse proof incomplete " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "checks": checks,
                      "valid_B": sum(sum(r["y_B_valid_mask"]) for r in rows), "valid_A": sum(sum(r["y_A_valid_mask"]) for r in rows)}))


SOURCES = ("docs/inference_cf/P2_TTA0_SPEC.md", "docs/inference_cf/P2_TTA0_CODEX_DESIGN.md", CONFIG, PANEL, PARENT, SEAL,
           "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2tta0.py",
           "experiments/inference_cf_p2tta0_analyze.py", "experiments/inference_cf_p2tta0_audit.py", "slurm/inference_cf_p2tta0.sbatch",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2tta0.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    cfg = json.loads((ROOT / CONFIG).read_text())
    extra = [f"{BASE}/prerun_audit.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *extra)
    if dirty:
        raise ValueError("commit P2-TTA0 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / extra[0]).read_text())["verdict"] != "PASS_TO_P2_TTA0":
        raise ValueError("pre-run audit did not pass")
    seal = json.loads((ROOT / SEAL).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "freeze_commit": FREEZE, "config": CONFIG, "config_hash": digest(cfg), "panel_sha256": sha_file(ROOT / PANEL),
           "ids": cfg["panel"]["ids"], "seal_hash": seal["seal_hash"], "trainables": [p["name"] for p in cfg["trainables"]["parameters"]],
           "optimization": cfg["optimization"], "role": "D-dev-select (already exposed panel20)",
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(extra)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---- GPU run -----------------------------------------------------------------------------------------------

def _summ_common(c):
    return None if c is None else {k: c[k] for k in ("entropy_sum", "valid", "eos_prob", "P_E", "P_M", "entropy")}


def run_objective(bundle, guard, kind, enc_train, enc_inf, y, mask, y_B, mask_B, *, suppress, begin, eos, partition,
                  counters, cfg_prompt=CB, keep_grad0=False, max_new_tokens=MAX_NEW):
    """One episodic objective: fresh masters/AdamW from theta0 -> 2 updates -> effective bf16 materialized into the
    selected LN only -> ordinary forced decode (fresh KV) -> finally exact restore + verify."""
    import torch
    from csasr.inference_cf.episodic_tta import TTAInvalid, adapt, common_diagnostic, forced_decode
    t0 = time.time()
    v_before = guard.other_versions()
    start_hash = guard.current_hash()
    res = adapt(bundle.model, guard, enc_train, cfg_prompt, y, mask, kind, suppress=suppress, begin=begin, eos=eos,
                partition=partition, counters=counters, keep_grad0=keep_grad0)
    common = res["common_y_B"]
    if kind == "A2":
        common = {"theta2": common_diagnostic(bundle.model, res["masters"], enc_train, cfg_prompt, y_B, mask_B, suppress=suppress,
                                              begin=begin, eos=eos, partition=partition, counters=counters)}
    res["masters"] = None
    t_adapt = time.time() - t0
    t1 = time.time()
    try:
        guard.materialize(res["effective"])
        dec = forced_decode(bundle, enc_inf, cfg_prompt, max_new_tokens=max_new_tokens)
        counters["final_decodes"] = counters.get("final_decodes", 0) + 1
    finally:
        guard.restore()
    reset_ok = guard.verify() and guard.current_hash() == guard.theta0_hash
    if not reset_ok:
        raise TTAInvalid("reset verification failed")
    log = res["log"]
    log.update(start_hash=start_hash, end_hash=guard.current_hash(), reset_ok=reset_ok, other_versions_unchanged=guard.other_versions() == v_before,
               adapt_sec=t_adapt, decode_sec=time.time() - t1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"log": log, "decode": dec, "common_y_B": {k: _summ_common(v) for k, v in (common or {}).items()},
            "final_masters": res["final_masters"], "grad0": res["grad0"]}


def run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import (LNGuard, TTAInvalid, compare, decoder_ln_names, forced_decode,
                                                 severe_truncation, tensor_bytes_hash)
    from csasr.lss.sites import assert_no_site_hooks, num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0_audit as live_audit
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    seal = json.loads((ROOT / SEAL).read_text())
    if seal["seal_hash"] != m["seal_hash"]:
        raise ValueError("pseudo seal")
    t_setup = time.time()
    torch.manual_seed(240924)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    model = bundle.model
    model.eval()
    model.requires_grad_(False)
    names = decoder_ln_names(model)
    if names != m["trainables"] or len(names) != 194:
        raise TTAInvalid("trainable enumeration != frozen 194 names")
    guard = LNGuard(model, names)
    if not guard.flags_ok():
        raise TTAInvalid("resident parameter requires grad")
    if num_forced_prefix_from(bundle.processor, language="zh", task="transcribe") != 4:
        raise ValueError("forced prefix")
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != seal["suppression"] or eos != seal["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    nonln_hash0 = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in set(names)])
    theta0_cpu = {n: guard.theta0[n].float().cpu().numpy() for n in names}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "theta0_ln_fp32.npz", **{f"p{i:03d}": theta0_cpu[n] for i, n in enumerate(names)})
    counters = {k: {"teacher_forwards": 0, "backwards": 0, "optimizer_steps": 0, "final_decodes": 0} for k in ("A1", "A2")}
    counters["theta0_integrity_decodes"] = 0
    counters["encoder_passes"] = 0
    counters["audit"] = {"forwards": 0, "backwards": 0}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln_hash0, "theta0_ln_fp32_npz_sha256": file_hash(out / "theta0_ln_fp32.npz"),
               "systems": {k: {"adapt_sec": 0.0, "decode_sec": 0.0, "peak_alloc": 0, "peak_reserved": 0} for k in ("A1", "A2")},
               "theta0_decode_sec": 0.0, "encoder_sec": 0.0, "B0_FORCED_timing": "reused (P2-SEQ S0), unavailable here",
               "B0_AUTO_timing": "reused (historical), unavailable here"}
    atomic_json(out / "runtime.json", runtime)
    invalid = []
    srows = {r["utterance_id"]: r for r in seal["rows"]}
    try:
        for i, uid in enumerate(m["ids"]):
            s = srows[uid]
            t_u = time.time()
            row = {"identity": uid, "manifest_hash": m["manifest_hash"], "objectives": {}}
            if audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                    audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
                raise TTAInvalid("audio bytes changed")
            ts = time.time()
            inputs = batch_model_inputs(bundle, [s["audio_path"]])
            with torch.inference_mode():
                h_inf = model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
            counters["encoder_passes"] += 1
            enc_inf = BaseModelOutput(last_hidden_state=h_inf)
            h_train = h_inf.clone()                                 # ordinary (non-inference) detached tensor
            if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
                raise TTAInvalid("encoder clone")
            enc_train = BaseModelOutput(last_hidden_state=h_train)
            runtime["encoder_sec"] += time.time() - ts
            # theta0 integrity decode must reproduce the sealed P2-SEQ S0 forced output
            ts = time.time()
            if not guard.verify():
                raise TTAInvalid("theta0 not resident before integrity decode")
            d0 = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_integrity_decodes"] += 1
            runtime["theta0_decode_sec"] += time.time() - ts
            row["theta0_decode"] = {"tokens_equal_S0": d0["tokens"] == s["y_B"], "terminated": d0["terminated"],
                                    "terminated_equal_S0": d0["terminated"] == s["y_B_terminated"], "length": d0["length"]}
            if not (row["theta0_decode"]["tokens_equal_S0"] and row["theta0_decode"]["terminated_equal_S0"]):
                invalid.append(f"{uid}: theta0 forced decode != P2-SEQ S0")
                row["theta0_decode"]["tokens"] = d0["tokens"]
            if i == 0:          # independent live loss/gradient check at theta0 (no optimizer update)
                row["live_audit"] = {}
                for kind, y, mk in (("A1", s["y_B"], s["y_B_valid_mask"]), ("A2", s["y_A"], s["y_A_valid_mask"])):
                    row["live_audit"][kind] = live_audit.live_objective_check(model, guard.names, enc_train, CB, y, kind,
                                                                              suppress=suppress, begin=begin, tokenizer=tok)
                    counters["audit"]["forwards"] += 1
                    counters["audit"]["backwards"] += 1
                if not guard.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            ckpt = {}
            for kind, y, mk in (("A1", s["y_B"], s["y_B_valid_mask"]), ("A2", s["y_A"], s["y_A_valid_mask"])):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                assert_no_site_hooks(bundle)
                r = run_objective(bundle, guard, kind, enc_train, enc_inf, y, mk, s["y_B"], s["y_B_valid_mask"], suppress=suppress,
                                  begin=begin, eos=eos, partition=partition, counters=counters[kind], keep_grad0=(i == 0))
                assert_no_site_hooks(bundle)
                sysrt = runtime["systems"][kind]
                sysrt["adapt_sec"] += r["log"]["adapt_sec"]
                sysrt["decode_sec"] += r["log"]["decode_sec"]
                sysrt["peak_alloc"] = max(sysrt["peak_alloc"], torch.cuda.max_memory_allocated())
                sysrt["peak_reserved"] = max(sysrt["peak_reserved"], torch.cuda.max_memory_reserved())
                dec = r["decode"]
                row["objectives"][kind] = {**r["log"], "common_y_B": r["common_y_B"], "tokens": dec["tokens"], "text": dec["text"],
                                           "terminated": dec["terminated"], "length": dec["length"],
                                           "vs_B0_FORCED": compare(dec["tokens"], s["y_B"]), "vs_B0_AUTO": compare(dec["tokens"], s["y_A"]),
                                           "severe_truncation": severe_truncation(len(s["y_B"]), dec["length"], dec["terminated"])}
                if i == 0:
                    la = row["live_audit"][kind]
                    g_aud = la.pop("_grad").cpu()
                    g_pri = r["grad0"]
                    la["primary_loss"] = r["log"]["losses"][0]
                    la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                    la["grad_diff_l2"] = float(torch.linalg.vector_norm((g_pri.double() - g_aud.double())))
                    la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                    la["primary_grad_l2"] = float(torch.linalg.vector_norm(g_pri.double()))
                    la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 1e-3 * la["auditor_grad_l2"])
                    if not la["pass"]:
                        invalid.append(f"{uid}: live audit {kind} disagreement")
                ckpt.update({f"{kind}_p{j:03d}": r["final_masters"][n].numpy() for j, n in enumerate(names)})
                del r
            np.savez_compressed(out / "rows" / f"{i:02d}_final_masters_fp32.npz", **ckpt)
            row["final_masters_npz_sha256"] = file_hash(out / "rows" / f"{i:02d}_final_masters_fp32.npz")
            row["status"] = "ok"
            row["elapsed_sec"] = time.time() - t_u
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-TTA0 {i + 1}/20 {uid} A1 L={row['objectives']['A1']['losses']} A2 L={row['objectives']['A2']['losses']} "
                  f"{row['elapsed_sec']:.1f}s", flush=True)
            runtime["counters"] = counters
            atomic_json(out / "runtime.json", runtime)
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        try:
            guard.restore()
        except Exception:
            pass
    runtime["reset_final_ok"] = bool(guard.verify() and guard.current_hash() == guard.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in set(names)])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln_hash0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-TTA0 run failed: " + runtime["failure"]["reason"])


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
