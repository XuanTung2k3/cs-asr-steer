#!/usr/bin/env python
"""P2-TTA-A4 runner (script-preserving AUTO distillation). Contract: docs/inference_cf/P2_TTA_A4_SPEC.md,
P2_TTA_A4_CLAUDE_DESIGN.md, configs/inference_cf/p2_tta_a4.json; panel = the A3 24-panel (unchanged).

prepare   (CPU) reuse the sealed A3 plan (y_B, y_A, masks, A2) and A3 outputs/lang ids (hash-checked against the A3
          output seal); canonical token classes of every y_A position; write plan_sealed.json. No reference.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_TTA_A4.
run       (GPU) per utterance: theta0 encoder once; theta0 forced decode = S0; historical detect_language -> cA (must equal
          the A3-sealed lang); A4 episode (safe teacher, 2 steps); final forced-ZH decode; exact reset. Row 0: independent
          live check.
seal      (CPU) output seal before any reference access.
No reference, evaluator, steering, D2 or LID input anywhere in this module.
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

SCHEMA = "p2_tta_a4_v1"
CONFIG = "configs/inference_cf/p2_tta_a4.json"
PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
A3_PLAN = "results/inference_cf/p2tta_a3/plan_sealed.json"
A3_SEAL = "results/inference_cf/p2tta_a3/output_seal.json"
A3_RUN = "results/inference_cf/p2tta_a3/run1"
BASE = "results/inference_cf/p2tta_a4"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
MAX_NEW = 200
REL_GRAD_TOL = 0.02


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.script_safe_tta import token_classes
    cfg = json.loads((ROOT / CONFIG).read_text())
    checks = {"anchors": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "panel_bytes": sha_file(ROOT / PANEL) == cfg["panel"]["byte_sha256"], "config_committed": committed(CONFIG)}
    a3plan = json.loads((ROOT / A3_PLAN).read_text())
    a3seal = json.loads((ROOT / A3_SEAL).read_text())
    a3man = json.loads((ROOT / A3_RUN / "manifest.json").read_text())
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    part = tokenizer_partition(tok)
    checks["partition"] = part["hash"] == cfg["partition"]["hash"] and part["version"] == cfg["partition"]["version"] and \
        not (set(part["embedded_ids"]) & set(part["matrix_ids"]))
    checks["a3_plan_ids"] = a3plan["ids"] == cfg["panel"]["ids"] and a3man["ids"] == cfg["panel"]["ids"]
    rows, ok = [], True
    for i, r in enumerate(a3plan["rows"]):
        rel = f"{A3_RUN}/rows/{i:02d}.json"
        a3 = json.loads((ROOT / rel).read_text())
        fine = a3seal["files"].get(rel) == sha_file(ROOT / rel) and a3["identity"] == r["utterance_id"] and a3["status"] == "ok"
        cls = token_classes(r["y_A"], part)
        ok &= bool(fine)
        rows.append({**{k: r[k] for k in ("utterance_id", "dialogue_id", "group", "audio_path", "audio_fingerprint_64k_sizeprefixed",
                                          "audio_full_sha256", "y_B", "y_B_terminated", "y_B_text", "y_A", "y_A_terminated", "y_A_text",
                                          "y_A_valid_mask", "A2", "d_BA", "d_A2A")},
                     "A3": {"row": rel, "row_sha256": sha_file(ROOT / rel), "tokens": a3["A3"]["tokens"], "text": a3["A3"]["text"],
                            "terminated": a3["A3"]["terminated"], "length": a3["A3"]["length"]},
                     "a3_lang_id": a3["auto_condition"]["lang_id"], "classes": cls,
                     "counts_valid": {x: sum(1 for c, m in zip(cls, r["y_A_valid_mask"]) if m and c == x) for x in "EMO"}})
    checks["a3_rows_sealed"] = bool(ok)
    d = [r for r in rows if r["group"] == "D"]
    checks["pre_outcome_counts_match_config"] = (
        {"EN": sum(r["counts_valid"]["E"] for r in d), "MATRIX": sum(r["counts_valid"]["M"] for r in d), "OTHER": sum(r["counts_valid"]["O"] for r in d)}
        == cfg["token_counts_pre_outcome"]["D"] and sum(r["counts_valid"]["E"] > 0 for r in d) == cfg["token_counts_pre_outcome"]["eligible_D_rows"])
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "ids": [r["utterance_id"] for r in rows], "suppression": a3plan["suppression"], "eos": a3plan["eos"], "partition_hash": part["hash"],
            "model_files": a3plan["model_files"], "environment": prep.environment(), "rows": rows, "outcomes_computed": False,
            "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = ("docs/inference_cf/P2_TTA_A4_SPEC.md", "docs/inference_cf/P2_TTA_A4_CLAUDE_DESIGN.md", CONFIG, PANEL, PLAN,
           "src/csasr/inference_cf/script_safe_tta.py", "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/episodic_tta.py",
           "src/csasr/inference_cf/core_r2.py", "experiments/inference_cf_p2tta_a4.py", "experiments/inference_cf_p2tta_a4_analyze.py",
           "experiments/inference_cf_p2tta_a4_audit.py", "slurm/inference_cf_p2tta_a4.sbatch", "experiments/inference_cf_p2tta0.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/lss/sites.py", "src/csasr/models/whisper.py",
           "tests/test_inference_cf_p2tta_a4.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit A4 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_TTA_A4":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "trainables": [p["name"] for p in cfg["trainables"]["parameters"]], "optimization": cfg["optimization"],
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, forced_decode, levenshtein, severe_truncation, tensor_bytes_hash
    from csasr.inference_cf.script_safe_tta import adapt_a4
    from csasr.inference_cf.soft_auto_tta import auto_prompt
    from csasr.lss.sites import assert_no_site_hooks
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
    import experiments.inference_cf_p2tta_a4_audit as live_audit
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != m["plan_hash"]:
        raise ValueError("plan")
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
        raise TTAInvalid("trainable enumeration")
    guard = LNGuard(model, names)
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or eos != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    if partition["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "theta0_ln_fp32.npz", **{f"p{i:03d}": guard.theta0[n].float().cpu().numpy() for i, n in enumerate(names)})
    counters = {"A4": {}, "theta0_forced_decodes": 0, "final_decodes": 0, "encoder_passes": 0, "detect_language_calls": 0,
                "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln0, "theta0_ln_fp32_npz_sha256": file_hash(out / "theta0_ln_fp32.npz"),
               "adapt_sec": 0.0, "decode_sec": 0.0, "baseline_decode_sec": 0.0, "peak_alloc": 0, "peak_reserved": 0}
    atomic_json(out / "runtime.json", runtime)
    invalid = []
    try:
        for i, s in enumerate(plan["rows"]):
            uid = s["utterance_id"]
            t_u = time.time()
            row = {"identity": uid, "group": s["group"], "manifest_hash": m["manifest_hash"]}
            if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                    t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
                raise TTAInvalid("audio bytes changed")
            inputs = batch_model_inputs(bundle, [s["audio_path"]])
            with torch.inference_mode():
                h_inf = model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
            counters["encoder_passes"] += 1
            enc_inf = BaseModelOutput(last_hidden_state=h_inf)
            h_train = h_inf.clone()
            if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
                raise TTAInvalid("encoder clone")
            enc_train = BaseModelOutput(last_hidden_state=h_train)
            if not guard.verify():
                raise TTAInvalid("theta0 not resident")
            ts = time.time()
            d0 = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_forced_decodes"] += 1
            row["theta0_forced"] = {"tokens_equal_S0": d0["tokens"] == s["y_B"], "terminated_equal_S0": d0["terminated"] == s["y_B_terminated"]}
            with torch.no_grad():
                lang = int(model.detect_language(input_features=inputs["input_features"], generation_config=model.generation_config)[0])
            counters["detect_language_calls"] += 1
            runtime["baseline_decode_sec"] += time.time() - ts
            cA = auto_prompt(lang)
            row["auto_condition"] = {"lang_id": lang, "lang_token": tok.convert_ids_to_tokens(lang), "cA": cA, "identical_to_cB": cA == CB,
                                     "lang_equals_A3_sealed": lang == s["a3_lang_id"]}
            if not (row["theta0_forced"]["tokens_equal_S0"] and row["theta0_forced"]["terminated_equal_S0"]):
                invalid.append(f"{uid}: theta0 forced decode != S0")
            if not row["auto_condition"]["lang_equals_A3_sealed"]:
                invalid.append(f"{uid}: detected lang != A3 sealed lang")
            if i == 0:
                la = live_audit.live_safe_kl_check(model, guard.names, enc_train, cA, CB, s["y_A"], partition, suppress=suppress, begin=begin,
                                                   tokenizer=tok)
                counters["audit"]["forwards"] += 3
                counters["audit"]["backwards"] += 1
                if not guard.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            assert_no_site_hooks(bundle)
            ts = time.time()
            r = adapt_a4(model, guard, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], s["classes"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A4"], keep_grad0=(i == 0))
            runtime["adapt_sec"] += time.time() - ts
            ts = time.time()
            v_before = guard.other_versions()
            try:
                guard.materialize(r["effective"])
                dec = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
                counters["final_decodes"] += 1
            finally:
                guard.restore()
            runtime["decode_sec"] += time.time() - ts
            assert_no_site_hooks(bundle)
            reset_ok = guard.verify() and guard.current_hash() == guard.theta0_hash
            if not reset_ok:
                raise TTAInvalid("reset verification failed")
            torch.cuda.synchronize()
            runtime["peak_alloc"] = max(runtime["peak_alloc"], torch.cuda.max_memory_allocated())
            runtime["peak_reserved"] = max(runtime["peak_reserved"], torch.cuda.max_memory_reserved())
            dA = levenshtein(dec["tokens"], s["y_A"])
            row["A4"] = {**r["log"], "tokens": dec["tokens"], "text": dec["text"], "terminated": dec["terminated"], "length": dec["length"],
                         "reset_ok": reset_ok, "end_hash": guard.current_hash(), "other_versions_unchanged": guard.other_versions() == v_before,
                         "equal_B0_FORCED": dec["tokens"] == s["y_B"], "equal_A2": dec["tokens"] == s["A2"]["tokens"],
                         "equal_A3": dec["tokens"] == s["A3"]["tokens"], "d_A4_AUTO": dA, "d_A4_B0": levenshtein(dec["tokens"], s["y_B"]),
                         "d_A4_A2": levenshtein(dec["tokens"], s["A2"]["tokens"]), "d_A4_A3": levenshtein(dec["tokens"], s["A3"]["tokens"]),
                         "d_BA": s["d_BA"], "movement": "CLOSER" if dA < s["d_BA"] else ("SAME" if dA == s["d_BA"] else "FARTHER"),
                         "severe_truncation": severe_truncation(len(s["y_B"]), dec["length"], dec["terminated"])}
            if row["A4"]["noop"] and not row["A4"]["equal_B0_FORCED"]:
                invalid.append(f"{uid}: no-op episode output != B0-FORCED")
            if i == 0:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = r["log"]["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["primary_grad_l2"] = float(torch.linalg.vector_norm(g_pri.double()))
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = la
                if not la["pass"]:
                    invalid.append(f"{uid}: live audit A4 disagreement")
            np.savez_compressed(out / "rows" / f"{i:02d}_final_masters_fp32.npz",
                                **{f"A4_p{j:03d}": r["final_masters"][n].numpy() for j, n in enumerate(names)})
            row["final_masters_npz_sha256"] = file_hash(out / "rows" / f"{i:02d}_final_masters_fp32.npz")
            del r
            row["status"] = "ok"
            row["elapsed_sec"] = time.time() - t_u
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-TTA-A4 {i + 1}/24 {uid} {s['group']} lang={row['auto_condition']['lang_token']} D_E={row['A4']['d_E']} "
                  f"D_ANCHOR={row['A4']['d_anchor']} {row['A4']['movement']} {row['elapsed_sec']:.1f}s", flush=True)
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
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-TTA-A4 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file())
    man = json.loads((run / "manifest.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"])}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "run": cmd_run, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
