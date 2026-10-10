#!/usr/bin/env python
"""P2-PATH0 runner (first-divergence rescue / induction causal test). Contract: docs/inference_cf/P2_PATH0_SPEC.md,
P2_PATH0_CODEX_DESIGN.md, P2_PATH0_SITES.json, configs/inference_cf/p2_path0.json (freeze fc3602d).

prepare   (CPU) verify config/sites/source hashes; recompute C/U, sites, donor spans and eligibility from raw sealed token
          arrays + termination; suppression validity of every forced token; write plan_sealed.json. No reference.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_PATH0.
run       (GPU) Phase 1, all 12 D rows (barrier): theta0 forced decode = B0 (+ theta0 branch diagnostics at k); sealed
          AUTO language; unchanged ``adapt_a4`` once (row 0 independent live check); effective bf16 LN == sealed checkpoint
          cast bf16; FREE-A4 = sealed A4 (+ adapted branch diagnostics); exact reset. Phase 2 per row: materialize the
          reconstructed state; SELF (must reproduce FREE), L=1 and L=3 donor clamps on fresh caches; exact reset.
seal      (CPU) immutable output seal (before references).
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

SCHEMA = "p2_path0_v1"
CONFIG = "configs/inference_cf/p2_path0.json"
SITES = "docs/inference_cf/P2_PATH0_SITES.json"
A4_PLAN = "results/inference_cf/p2tta_a4/plan_sealed.json"
BASE = "results/inference_cf/p2path0"
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


def tok_hash(a) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in a], separators=(",", ":")).encode("ascii")).hexdigest()


def compute_sites(y_B, tB, y_A, tA, a4, t4) -> dict:
    """Primary site construction from raw content arrays + termination (decision streams)."""
    from csasr.inference_cf.path_decode import EOS, first_divergence, stream, suffix_distance
    B, A, F = stream(y_B, tB), stream(y_A, tA), stream(a4, t4)
    if F != B:
        role, k = "RESCUE", first_divergence(F, B)
        alt, donor, target_base = F[k], B, F
    else:
        role, k = "INDUCE", first_divergence(B, A)
        alt, donor, target_base = A[k], A, B
    out = {"role": role, "site_index": k, "B0_token": B[k], "alternative_token": alt, "common_prefix": B[:k], "clamps": {}}
    for L in (1, 3):
        span = donor[k:min(k + L, MAX_NEW)]
        if EOS in span:
            span = span[:span.index(EOS) + 1]
        b = k + len(span)
        d0 = suffix_distance(F, B, b) if role == "RESCUE" else suffix_distance(B, A, b)
        out["clamps"][str(L)] = {"tokens": span, "effective_length": len(span), "release_index": b, "suffix_baseline_distance": d0,
                                 "suffix_eligible": d0 > 0}
    out["self_clamp"] = [F[k]] if role == "RESCUE" else [B[k]]
    return out


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.path_decode import allowed_at
    cfg = json.loads((ROOT / CONFIG).read_text())
    S = json.loads((ROOT / SITES).read_text())
    checks = {"sites_bytes": sha_file(ROOT / SITES) == cfg["sites_byte_sha256"], "config_committed": committed(CONFIG) and committed(SITES),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "site_top_hashes": sha_file(ROOT / S["parent_panel"]) == S["parent_panel_sha256"] and sha_file(ROOT / S["A4_plan"]) == S["A4_plan_sha256"]
              and sha_file(ROOT / S["A4_output_seal"]) == S["A4_output_seal_sha256"]}
    plan4 = json.loads((ROOT / A4_PLAN).read_text())
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["suppression_frozen"] = {"suppress": sup, "begin": beg} == cfg["suppression"] and tok.eos_token_id == cfg["eos_token_id"] == 50257
    rows, ok = [], True
    for s in S["rows"]:
        i = s["parent_A3_index"]
        p4 = plan4["rows"][i]
        a4rel = s["sources"]["A4"]["path"]
        a4 = json.loads((ROOT / a4rel).read_text())["A4"]
        mine = compute_sites(p4["y_B"], p4["y_B_terminated"], p4["y_A"], p4["y_A_terminated"], a4["tokens"], a4["terminated"])
        fine = (p4["utterance_id"] == s["utterance_id"] and p4["group"] == "D" and sha_file(ROOT / a4rel) == s["sources"]["A4"]["byte_sha256"]
                and mine["role"] == s["role"] and mine["site_index"] == s["site_index"] and mine["B0_token"] == s["B0_token"]
                and mine["alternative_token"] == s["alternative_token"] and mine["common_prefix"] == s["common_prefix"]
                and tok_hash(mine["common_prefix"]) == s["common_prefix_sha256"] and mine["self_clamp"] == s["self_clamp"]
                and all(mine["clamps"][L]["tokens"] == s["clamps"][L]["tokens"] and mine["clamps"][L]["effective_length"] == s["clamps"][L]["effective_length"]
                        and mine["clamps"][L]["release_index"] == s["clamps"][L]["release_index"]
                        and mine["clamps"][L]["suffix_baseline_distance"] == s["clamps"][L]["suffix_baseline_distance"]
                        and mine["clamps"][L]["suffix_eligible"] == s["clamps"][L]["suffix_eligible"] for L in ("1", "3"))
                and all(allowed_at(t, s["site_index"] + j, sup, beg) for L in ("1", "3") for j, t in enumerate(s["clamps"][L]["tokens"]))
                and allowed_at(s["self_clamp"][0], s["site_index"], sup, beg)
                and not any(t >= 50257 for t in p4["y_B"] + p4["y_A"] + a4["tokens"])
                and sha_file(ROOT / s["A4_final_master_checkpoint"]) == s["A4_final_master_sha256"])
        ok &= bool(fine)
        rows.append({"utterance_id": s["utterance_id"], "dialogue_id": s["dialogue_id"], "a4_index": i, "role": s["role"], "site": s["site_index"],
                     "B0_token": s["B0_token"], "alt_token": s["alternative_token"], "common_prefix": s["common_prefix"],
                     "clamps": {L: s["clamps"][L] for L in ("1", "3")}, "self_clamp": s["self_clamp"], "audio_path": p4["audio_path"],
                     "audio_fingerprint_64k_sizeprefixed": p4["audio_fingerprint_64k_sizeprefixed"], "audio_full_sha256": p4["audio_full_sha256"],
                     "y_B": p4["y_B"], "y_B_terminated": p4["y_B_terminated"], "y_A": p4["y_A"], "y_A_terminated": p4["y_A_terminated"],
                     "y_A_valid_mask": p4["y_A_valid_mask"], "classes": p4["classes"], "a3_lang_id": p4["a3_lang_id"],
                     "A4_tokens": a4["tokens"], "A4_terminated": a4["terminated"], "A4_checkpoint": s["A4_final_master_checkpoint"]})
    checks["sites_recomputed"] = bool(ok)
    n = lambda role, L: sum(r["role"] == role and r["clamps"][L]["suffix_eligible"] for r in rows)
    checks["population"] = (len(rows) == 12 and sum(r["role"] == "RESCUE" for r in rows) == 5 and sum(r["role"] == "INDUCE" for r in rows) == 7
                            and (n("RESCUE", "1"), n("RESCUE", "3"), n("INDUCE", "1"), n("INDUCE", "3")) == (4, 4, 6, 5))
    plan = {"schema": SCHEMA + "_plan", "sites_sha256": sha_file(ROOT / SITES), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "ids": [r["utterance_id"] for r in rows], "suppression": {"suppress": sup, "begin": beg}, "eos": 50257,
            "partition_hash": plan4["partition_hash"], "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = ("docs/inference_cf/P2_PATH0_SPEC.md", "docs/inference_cf/P2_PATH0_CODEX_DESIGN.md", SITES, CONFIG, PLAN, A4_PLAN,
           "src/csasr/inference_cf/path_decode.py", "src/csasr/inference_cf/script_safe_tta.py", "src/csasr/inference_cf/soft_auto_tta.py",
           "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2path0.py", "experiments/inference_cf_p2path0_analyze.py",
           "experiments/inference_cf_p2path0_audit.py", "experiments/inference_cf_p2tta_a4_audit.py", "slurm/inference_cf_p2path0.sbatch",
           "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py",
           "src/csasr/lss/sites.py", "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path0.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH0 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH0":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "sites_sha256": sha_file(ROOT / SITES), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "trainables": [p["name"] for p in cfg["A4_inherited"]["trainables"]["parameters"]], "budget": cfg["compute"],
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
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
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, tensor_bytes_hash
    from csasr.inference_cf.path_decode import branch_diagnostics, clamp_decode
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
    counters = {"A4": {}, "theta0_decodes": 0, "adapted_decodes": 0, "encoder_passes": 0, "detect_language_calls": 0,
                "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln0, "phase1_sec": 0.0, "phase2_sec": 0.0}
    atomic_json(out / "runtime.json", runtime)
    invalid, rows_out, keep = [], [], []
    try:
        # ---------------- Phase 1: all-12 reconstruction barrier ----------------
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for i, s in enumerate(plan["rows"]):
            uid = s["utterance_id"]
            row = {"identity": uid, "role": s["role"], "site": s["site"], "manifest_hash": m["manifest_hash"]}
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
            diag = {}
            base_tok, alt_tok = s["B0_token"], s["alt_token"]
            d0 = clamp_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW, site=s["site"],
                              at_site=lambda lg, t: diag.__setitem__("theta0", branch_diagnostics(lg, t, suppress, begin, alt_tok, base_tok, partition)))
            counters["theta0_decodes"] += 1
            row["theta0_equals_B0"] = d0["tokens"] == s["y_B"] and d0["terminated"] == s["y_B_terminated"]
            with torch.no_grad():
                lang = int(model.detect_language(input_features=inputs["input_features"], generation_config=model.generation_config)[0])
            counters["detect_language_calls"] += 1
            row["lang_equals_sealed"] = lang == s["a3_lang_id"]
            cA = auto_prompt(lang)
            if i == 0:
                la = live_audit.live_safe_kl_check(model, guard.names, enc_train, cA, CB, s["y_A"], partition, suppress=suppress, begin=begin, tokenizer=tok)
                counters["audit"]["forwards"] += 3
                counters["audit"]["backwards"] += 1
                if not guard.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            assert_no_site_hooks(bundle)
            r = adapt_a4(model, guard, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], s["classes"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A4"], keep_grad0=(i == 0))
            with np.load(ROOT / s["A4_checkpoint"]) as z:
                ck = {n: torch.from_numpy(z[f"A4_p{j:03d}"]).to(torch.bfloat16) for j, n in enumerate(names)}
            row["effective_equals_checkpoint_bf16"] = all(torch.equal(r["effective"][n].cpu(), ck[n]) for n in names)
            row["A4_noop"] = r["log"]["noop"]
            row["A4_losses"] = r["log"]["losses"]
            v_before = guard.other_versions()
            try:
                guard.materialize(r["effective"])
                free = clamp_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW, site=s["site"],
                                    at_site=lambda lg, t: diag.__setitem__("A4", branch_diagnostics(lg, t, suppress, begin, alt_tok, base_tok, partition)))
                counters["adapted_decodes"] += 1
            finally:
                guard.restore()
            row["reset_ok_phase1"] = guard.verify() and guard.current_hash() == guard.theta0_hash and guard.other_versions() == v_before
            row["FREE"] = {"tokens": free["tokens"], "terminated": free["terminated"]}
            row["FREE_equals_sealed_A4"] = free["tokens"] == s["A4_tokens"] and free["terminated"] == s["A4_terminated"]
            row["branch_diagnostics"] = {**diag, "margin_shift_A4_minus_theta0": diag["A4"]["margin_alt_minus_base"] - diag["theta0"]["margin_alt_minus_base"]}
            if i == 0:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = r["log"]["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = la
            for k in ("theta0_equals_B0", "lang_equals_sealed", "effective_equals_checkpoint_bf16", "reset_ok_phase1", "FREE_equals_sealed_A4"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            if i == 0 and not row["live_audit"]["pass"]:
                invalid.append(f"{uid}: live audit")
            rows_out.append(row)
            keep.append((enc_inf, {n: v.cpu() for n, v in r["effective"].items()}))
            del r
            print(f"P2-PATH0 phase1 {i + 1}/12 {uid} {s['role']} FREE==A4 {row['FREE_equals_sealed_A4']} ckpt {row['effective_equals_checkpoint_bf16']}", flush=True)
        runtime["phase1_sec"] = time.time() - t1
        runtime["counters"] = counters
        if invalid:
            raise TTAInvalid("phase-1 barrier failed: " + "; ".join(invalid))
        # ---------------- Phase 2: clamps on the reconstructed state ----------------
        t2 = time.time()
        for i, s in enumerate(plan["rows"]):
            enc_inf, eff = keep[i]
            row = rows_out[i]
            arms = {"SELF": s["self_clamp"], "L1": s["clamps"]["1"]["tokens"], "L3": s["clamps"]["3"]["tokens"]}
            v_before = guard.other_versions()
            try:
                guard.materialize({n: v.to(guard.theta0[n].device) for n, v in eff.items()})
                for arm, toks in arms.items():
                    d = clamp_decode(bundle, enc_inf, CB, site=s["site"], forced=toks, expected_prefix=s["common_prefix"], max_new_tokens=MAX_NEW)
                    counters["adapted_decodes"] += 1
                    row[arm] = {"tokens": d["tokens"], "terminated": d["terminated"], "trace": d["trace"]}
                    if not (d["trace"]["prefix_ok"] and d["trace"]["suppression_ok"] and d["trace"]["positions_ok"]
                            and len(d["trace"]["forced"]) == len(toks)):
                        invalid.append(f"{s['utterance_id']}: {arm} clamp integrity")
            finally:
                guard.restore()
            row["reset_ok_phase2"] = guard.verify() and guard.current_hash() == guard.theta0_hash and guard.other_versions() == v_before
            row["SELF_equals_FREE"] = row["SELF"]["tokens"] == row["FREE"]["tokens"] and row["SELF"]["terminated"] == row["FREE"]["terminated"]
            if not row["SELF_equals_FREE"]:
                invalid.append(f"{s['utterance_id']}: SELF != FREE")
            if not row["reset_ok_phase2"]:
                invalid.append(f"{s['utterance_id']}: phase-2 reset")
            row["status"] = "ok"
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-PATH0 phase2 {i + 1}/12 {s['utterance_id']} {s['role']} SELF==FREE {row['SELF_equals_FREE']}", flush=True)
        runtime["phase2_sec"] = time.time() - t2
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        for i, row in enumerate(rows_out):
            if not (out / "rows" / f"{i:02d}.json").exists():
                atomic_json(out / "rows" / f"{i:02d}.json", {**row, "status": "partial"})
        try:
            guard.restore()
        except Exception:
            pass
    runtime["peak_alloc"] = int(torch.cuda.max_memory_allocated())
    runtime["peak_reserved"] = int(torch.cuda.max_memory_reserved())
    runtime["reset_final_ok"] = bool(guard.verify() and guard.current_hash() == guard.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-PATH0 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file())
    files += [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "sites_sha256": sha_file(ROOT / SITES), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "primary_label": json.loads((ROOT / BASE / "primary_analysis.json").read_text())["label"], "references_used": False,
           "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"]), "primary_label": doc["primary_label"]}))


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
