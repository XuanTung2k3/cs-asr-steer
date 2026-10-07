#!/usr/bin/env python
"""P2-PATH1 runner (branch adjudication + state-vs-prefix decomposition). Contract: docs/inference_cf/P2_PATH1_SPEC.md,
P2_PATH1_CODEX_DESIGN.md, P2_PATH1_SITES.json, configs/inference_cf/p2_path1.json (freeze 5a159f2).

prepare   (CPU) verify config/sites/parent hashes and PATH0-row fingerprints; recompute U/C, k, prefixes, c_B/c_A, L1
          eligibility, H=3 content donors and historical A4A targets from raw sealed arrays; write plan_sealed.json.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_PATH1.
run       (GPU) Phase 1 all 12 (barrier): theta0 decode = B0; sealed AUTO language; unchanged adapt_a4 once (row 0 live
          check); effective bf16 == checkpoint; FREE-A4 = sealed A4; exact reset. Phase 2 per row, theta0 state first
          (U: T0B, T0A decodes; all: theta0 B/ALT scores), then materialized A4 (U: A4B, A4A; all: A4 B/ALT scores), reset.
          Every path: fresh Branch created after state selection, LN hash locked for the whole path.
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

SCHEMA = "p2_path1_v1"
CONFIG = "configs/inference_cf/p2_path1.json"
SITES = "docs/inference_cf/P2_PATH1_SITES.json"
PARENT = "docs/inference_cf/P2_PATH0_SITES.json"
A4_PLAN = "results/inference_cf/p2tta_a4/plan_sealed.json"
BASE = "results/inference_cf/p2path1"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def row_fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def tok_hash(a) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in a], separators=(",", ":")).encode("ascii")).hexdigest()


def verify_row(s: dict, p0: dict, p4: dict, a4: dict) -> bool:
    from csasr.inference_cf.path_decode import first_divergence, stream, suffix_distance
    B, A, F = stream(p4["y_B"], p4["y_B_terminated"]), stream(p4["y_A"], p4["y_A_terminated"]), stream(a4["tokens"], a4["terminated"])
    U = F == B
    k = first_divergence(B, A) if U else first_divergence(F, B)
    alt = A if U else F
    ok = (row_fp(p0) == s["PATH0_row_sha256"] and s["population"] == ("U_PRIMARY" if U else "C_SCORE_ONLY") and s["site_index"] == k == p0["site_index"]
          and s["common_prefix"] == B[:k] and tok_hash(B[:k]) == s["common_prefix_sha256"] and s["c_B"] == B[k] and s["c_ALT"] == alt[k]
          and s["L1"] == p0["clamps"]["1"])
    alt_content = p4["y_A"] if U else a4["tokens"]
    ok &= (s["score_branches"]["B"]["tokens"] == list(p4["y_B"][k:k + 3]) and s["score_branches"]["ALT"]["tokens"] == list(alt_content[k:k + 3])
           and s["score_branches"]["B"]["H_eff"] == len(p4["y_B"][k:k + 3]) and s["score_branches"]["ALT"]["H_eff"] == len(alt_content[k:k + 3])
           and len(s["score_branches"]["B"]["tokens"]) >= 1 and len(s["score_branches"]["ALT"]["tokens"]) >= 1)
    if U:
        p0o = json.loads((ROOT / s["PATH0_output_path"]).read_text())["L1"]
        d = suffix_distance(B, A, k + 1)
        ok &= (s["historical_A4A"] == {"tokens": p0o["tokens"], "terminated": p0o["terminated"], "release_index": k + 1} and s["c_A"] == A[k]
               and d == s["L1"]["suffix_baseline_distance"] and (d > 0) == s["L1"]["suffix_eligible"])
    return bool(ok)


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig
    from csasr.inference_cf.path_decode import allowed_at
    cfg = json.loads((ROOT / CONFIG).read_text())
    S = json.loads((ROOT / SITES).read_text())
    P0 = json.loads((ROOT / PARENT).read_text())
    checks = {"sites_bytes": sha_file(ROOT / SITES) == cfg["sites_byte_sha256"], "frozen_committed": committed(CONFIG) and committed(SITES),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "parent": sha_file(ROOT / S["parent"]) == S["parent_byte_sha256"] and sha_file(ROOT / S["PATH0_output_seal"]) == S["PATH0_output_seal_sha256"]}
    plan4 = json.loads((ROOT / A4_PLAN).read_text())
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["suppression_frozen"] = {"suppress": sup, "begin": beg} == cfg["suppression"]
    rows, ok = [], True
    for s, p0 in zip(S["rows"], P0["rows"]):
        p4 = plan4["rows"][p0["parent_A3_index"]]
        a4 = json.loads((ROOT / p0["sources"]["A4"]["path"]).read_text())["A4"]
        fine = (s["utterance_id"] == p0["utterance_id"] == p4["utterance_id"] and verify_row(s, p0, p4, a4)
                and sha_file(ROOT / s["PATH0_output_path"]) == s["PATH0_output_sha256"] and sha_file(ROOT / s["A4_checkpoint"]) == s["A4_checkpoint_sha256"]
                and all(allowed_at(t, s["site_index"] + j, sup, beg) for b in ("B", "ALT") for j, t in enumerate(s["score_branches"][b]["tokens"]))
                and all(allowed_at(t, s["site_index"], sup, beg) for t in (s["c_B"], s["c_ALT"])))
        ok &= fine
        rows.append({"utterance_id": s["utterance_id"], "dialogue_id": s["dialogue_id"], "population": s["population"], "a4_index": p0["parent_A3_index"],
                     "site": s["site_index"], "common_prefix": s["common_prefix"], "c_B": s["c_B"], "c_ALT": s["c_ALT"], "L1": s["L1"],
                     "score_branches": s["score_branches"], "historical_A4A": s.get("historical_A4A"),
                     "audio_path": p4["audio_path"], "audio_fingerprint_64k_sizeprefixed": p4["audio_fingerprint_64k_sizeprefixed"],
                     "audio_full_sha256": p4["audio_full_sha256"], "y_B": p4["y_B"], "y_B_terminated": p4["y_B_terminated"], "y_A": p4["y_A"],
                     "y_A_terminated": p4["y_A_terminated"], "y_A_valid_mask": p4["y_A_valid_mask"], "classes": p4["classes"],
                     "a3_lang_id": p4["a3_lang_id"], "A4_tokens": a4["tokens"], "A4_terminated": a4["terminated"], "A4_checkpoint": s["A4_checkpoint"]})
    checks["sites_recomputed"] = bool(ok)
    U = [r for r in rows if r["population"] == "U_PRIMARY"]
    checks["population"] = (len(U), sum(r["L1"]["suffix_eligible"] for r in U), len(rows) - len(U)) == (7, 6, 5) and \
        sum(r["L1"]["suffix_baseline_distance"] for r in U if r["L1"]["suffix_eligible"]) == 48
    plan = {"schema": SCHEMA + "_plan", "sites_sha256": sha_file(ROOT / SITES), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "ids": [r["utterance_id"] for r in rows], "suppression": {"suppress": sup, "begin": beg}, "eos": EOS,
            "partition_hash": plan4["partition_hash"], "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = ("docs/inference_cf/P2_PATH1_SPEC.md", "docs/inference_cf/P2_PATH1_CODEX_DESIGN.md", SITES, CONFIG, PLAN, PARENT, A4_PLAN,
           "src/csasr/inference_cf/branch_adjudication.py", "src/csasr/inference_cf/path_decode.py", "src/csasr/inference_cf/script_safe_tta.py",
           "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2path1.py",
           "experiments/inference_cf_p2path1_analyze.py", "experiments/inference_cf_p2path1_audit.py", "experiments/inference_cf_p2path0_analyze.py",
           "experiments/inference_cf_p2tta_a4_audit.py", "slurm/inference_cf_p2path1.sbatch", "experiments/inference_cf_p2tta0.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path1.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH1 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH1":
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
    from csasr.inference_cf.branch_adjudication import owned_clamp, score_branch
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, tensor_bytes_hash
    from csasr.inference_cf.path_decode import clamp_decode
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
    counters = {"A4": {}, "theta0_decodes": 0, "free_decodes": 0, "factorial_decodes": 0, "score_paths": 0, "scored_terms": 0,
                "encoder_passes": 0, "detect_language_calls": 0, "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln0, "phase1_sec": 0.0, "phase2_sec": 0.0}
    atomic_json(out / "runtime.json", runtime)
    invalid, rows_out, keep = [], [], []
    try:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        for i, s in enumerate(plan["rows"]):
            uid = s["utterance_id"]
            row = {"identity": uid, "population": s["population"], "site": s["site"], "manifest_hash": m["manifest_hash"]}
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
            d0 = clamp_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
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
            v_before = guard.other_versions()
            try:
                guard.materialize(r["effective"])
                a4_hash = guard.current_hash()
                free = clamp_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
                counters["free_decodes"] += 1
            finally:
                guard.restore()
            row["A4_state_hash"] = a4_hash
            row["reset_ok_phase1"] = guard.verify() and guard.current_hash() == guard.theta0_hash and guard.other_versions() == v_before
            row["FREE"] = {"tokens": free["tokens"], "terminated": free["terminated"]}
            row["FREE_equals_sealed_A4"] = free["tokens"] == s["A4_tokens"] and free["terminated"] == s["A4_terminated"]
            if i == 0:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = r["log"]["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = la
                if not la["pass"]:
                    invalid.append(f"{uid}: live audit")
            for k in ("theta0_equals_B0", "lang_equals_sealed", "effective_equals_checkpoint_bf16", "reset_ok_phase1", "FREE_equals_sealed_A4"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            rows_out.append(row)
            keep.append((enc_inf, {n: v.cpu() for n, v in r["effective"].items()}))
            del r
            print(f"P2-PATH1 phase1 {i + 1}/12 {uid} {s['population']} FREE==A4 {row['FREE_equals_sealed_A4']} ckpt {row['effective_equals_checkpoint_bf16']}", flush=True)
        runtime["phase1_sec"] = time.time() - t1
        if invalid:
            raise TTAInvalid("phase-1 barrier failed: " + "; ".join(invalid))
        # ---------------- Phase 2: state-owned factorial and branch scoring ----------------
        t2 = time.time()
        for i, s in enumerate(plan["rows"]):
            enc_inf, eff = keep[i]
            row = rows_out[i]
            U = s["population"] == "U_PRIMARY"
            pre = s["common_prefix"]
            row["paths"], row["scores"] = {}, {}
            for state in ("theta0", "A4"):
                v_before = guard.other_versions()
                try:
                    if state == "A4":
                        guard.materialize({n: v.to(guard.theta0[n].device) for n, v in eff.items()})
                    elif not guard.verify():
                        raise TTAInvalid("theta0 not resident for theta0 paths")
                    sh = guard.current_hash()
                    if (state == "A4") != (sh != guard.theta0_hash) and not row["A4_noop"]:
                        raise TTAInvalid("state selection")
                    if state == "A4" and sh != row["A4_state_hash"]:
                        raise TTAInvalid("A4 state hash != phase-1 reconstructed state")
                    tag = "T0" if state == "theta0" else "A4"
                    if U:
                        for br, ctok in (("B", s["c_B"]), ("A", s["c_ALT"])):
                            owner = {"state": state, "state_hash": sh, "row": s["utterance_id"], "condition": tag + br}
                            d = owned_clamp(bundle, enc_inf, CB, site=s["site"], forced=[ctok], expected_prefix=pre, owner=owner,
                                            state_hash_fn=guard.current_hash, max_new_tokens=MAX_NEW)
                            counters["factorial_decodes"] += 1
                            row["paths"][tag + br] = {"tokens": d["tokens"], "terminated": d["terminated"], "trace": d["trace"], "owner": owner,
                                                      "state_locked": d["state_locked"]}
                            tr = d["trace"]
                            if not (d["state_locked"] and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"] and len(tr["forced"]) == 1
                                    and tr["release_index"] == s["site"] + 1):
                                invalid.append(f"{s['utterance_id']}: {tag + br} path integrity")
                    for b in ("B", "ALT"):
                        owner = {"state": state, "state_hash": sh, "row": s["utterance_id"], "condition": f"score_{tag}_{b}"}
                        sc = score_branch(bundle, enc_inf, CB, pre, s["score_branches"][b]["tokens"], owner=owner, state_hash_fn=guard.current_hash)
                        counters["score_paths"] += 1
                        counters["scored_terms"] += sc["H_eff"]
                        row["scores"][f"{tag}_{b}"] = sc
                        if not (sc["state_locked"] and sc["prefix_ok"] and sc["positions_ok"] and sc["fed_ok"] and sc["H_eff"] == s["score_branches"][b]["H_eff"]):
                            invalid.append(f"{s['utterance_id']}: score {tag}_{b} integrity")
                finally:
                    guard.restore()
                if not (guard.verify() and guard.current_hash() == guard.theta0_hash and guard.other_versions() == v_before):
                    invalid.append(f"{s['utterance_id']}: reset after {state}")
            if U:
                p = row["paths"]
                row["T0B_equals_B0"] = p["T0B"]["tokens"] == s["y_B"] and p["T0B"]["terminated"] == s["y_B_terminated"]
                row["A4B_equals_B0"] = p["A4B"]["tokens"] == s["y_B"] and p["A4B"]["terminated"] == s["y_B_terminated"]
                h = s["historical_A4A"]
                row["A4A_equals_PATH0_L1"] = p["A4A"]["tokens"] == h["tokens"] and p["A4A"]["terminated"] == h["terminated"]
                for k in ("T0B_equals_B0", "A4B_equals_B0", "A4A_equals_PATH0_L1"):
                    if not row[k]:
                        invalid.append(f"{s['utterance_id']}: {k}")
            row["status"] = "ok"
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-PATH1 phase2 {i + 1}/12 {s['utterance_id']} {s['population']} "
                  + (f"T0B {row['T0B_equals_B0']} A4B {row['A4B_equals_B0']} A4A {row['A4A_equals_PATH0_L1']}" if U else "scores"), flush=True)
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
        raise SystemExit("P2-PATH1 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "sites_sha256": sha_file(ROOT / SITES), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "primary_label": prim["label"], "references_used": False, "created_unix": time.time()}
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
