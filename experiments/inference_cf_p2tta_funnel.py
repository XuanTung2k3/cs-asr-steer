#!/usr/bin/env python
"""P2-TTA-FUNNEL runner (frozen contract ff75e1a). Stage C — TTA1 for the single objective in the committed,
independently audited selection artifact (here: TTA0-R A2 AUTO-CONSISTENCY).

tta1-prepare  (CPU) verify selection seal vs master/TTA0 config, panel100 bytes, baseline100 hashes, resolve teachers
              (y_B = P2-SEQ S0, y_A = historical AUTO), fixed masks, reuse plan (all compatible audited ancestor
              episodes: TTA0 run1 A2, 20 rows; no outcome filtering) and new-episode IDs; write plan_sealed.json.
tta1-manifest (CPU) resolved manifest after committed PASS_TO_P2_TTA1.
tta1-run      (GPU) one allocation: for every new ID one detached encoder, theta0 forced integrity decode = S0, the
              exact TTA0 A2 episode (`inference_cf_p2tta0.run_objective`: fresh fp32 masters + fresh AdamW, 2 steps,
              3 loss forwards, final forced-ZH decode, bitwise reset), theta0/theta2 common-y_B diagnostics; first
              new row: independent live loss/gradient check (2e-2 relative).
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

SCHEMA = "p2_tta1_v1"
FREEZE = "ff75e1a6b524415d8071749554b66c09f8dd2ad2"
CONFIG = "configs/inference_cf/p2_tta_funnel.json"
TTA0_CONFIG = "configs/inference_cf/p2_tta0.json"
PANEL = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
SELECTION = "results/inference_cf/p2tta_funnel/tta0_r/selected_objective.json"
R_AUDIT = "results/inference_cf/p2tta_funnel/tta0_r/post_audit.json"
TTA1 = "results/inference_cf/p2tta_funnel/tta1"
PLAN = f"{TTA1}/plan_sealed.json"
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


def cmd_tta1_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2tta0 as t0run
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.episodic_tta import OPTIM, STEPS, clean_teacher, decoder_ln_names, special_ids, valid_mask
    c = json.loads((ROOT / CONFIG).read_text())
    t0 = json.loads((ROOT / TTA0_CONFIG).read_text())
    sel = json.loads((ROOT / SELECTION).read_text())
    raud = json.loads((ROOT / R_AUDIT).read_text())
    checks = {}
    checks["selection_and_audit_committed"] = committed(SELECTION) and committed(R_AUDIT)
    checks["selection_parent_audit"] = (sel["parent_stage_label"] == "P2_TTA0_R_OBJECTIVE_SELECTED" == raud["r_label"]
                                        and raud["verdict"] == "P2_TTA0_R_AUDIT: PASS" and raud["selected"] == sel["objective"]["id"] == "A2"
                                        and sel["independent_audit"]["sha256"] == sha_file(ROOT / R_AUDIT))
    checks["selection_settings_match_master"] = (
        sel["optimizer"] == {k: c["common"]["optimization"][k] for k in sel["optimizer"]}
        and [{k: p[k] for k in ("name", "shape", "numel")} for p in sel["trainables"]["parameters"]] == c["common"]["trainables"]["parameters"]
        and sel["teacher_forcing"] == c["common"]["teacher_forcing"] and sel["decode"] == c["common"]["decode"]
        and sel["objective"]["loss"] == t0["objectives"]["A2"]["loss"] and sel["objective"]["student_prompt_cB"] == CB
        and OPTIM["lr"] == c["common"]["optimization"]["lr"] and STEPS == c["common"]["optimization"]["steps"] == 2)
    checks["panel100_bytes"] = sha_file(ROOT / PANEL) == c["TTA1"]["panel_sha256"]
    panel = json.loads((ROOT / PANEL).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in panel["rows"]}
    checks["panel100_shape"] = len(ids) == 100 == len(set(ids)) and len(set(dlg.values())) == 20
    base = {b["utterance_id"]: b for b in c["baseline100"]}
    checks["baseline100_bytes"] = set(base) == set(ids) and all(
        sha_file(ROOT / b["forced_path"]) == b["forced_sha256"] and sha_file(ROOT / b["auto_path"]) == b["auto_sha256"] for b in base.values())
    seqm = json.loads((ROOT / "results/inference_cf/p2seq/run1/manifest.json").read_text())
    reuse = json.loads((ROOT / "results/inference_cf/p2seq/reuse_audit.json").read_text())
    rr = {r["utterance_id"]: r for r in reuse["rows"]}
    checks["p2seq_reuse_proof"] = reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100" and all(rr[u]["row"] == base[u]["auto_path"] and rr[u]["audio_ok"] for u in ids)
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = proc.tokenizer
    eos, specials = tok.eos_token_id, special_ids(tok)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["suppression_matches_selection"] = sel["vocabulary_mask"]["suppress_tokens"] == sup and sel["vocabulary_mask"]["begin_suppress_tokens"] == beg \
        and sel["vocabulary_mask"]["eos"] == eos
    mh = prep.model_hashes()
    checks["model_files"] = mh == sel["hashes"]["model_files"] == seqm["model"]["files"]
    env = prep.environment()
    checks["environment"] = all(env.get(k) == sel["hashes"]["environment"].get(k) for k in ("python", "torch", "transformers"))
    # ancestor episodes: TTA0-R A2 (all compatible rows; no outcome filtering)
    t0m = json.loads((ROOT / "results/inference_cf/p2tta0/run1/manifest.json").read_text())
    anc = {}
    for i, u in enumerate(t0m["ids"]):
        rel = f"results/inference_cf/p2tta0/run1/rows/{i:02d}.json"
        row = json.loads((ROOT / rel).read_text())
        assert sha_file(ROOT / rel) == c["TTA0_R"]["sealed_sha256"][rel]
        anc[u] = {"row": rel, "row_sha256": sha_file(ROOT / rel), "index": i,
                  "masters_npz": f"results/inference_cf/p2tta0/run1/rows/{i:02d}_final_masters_fp32.npz",
                  "masters_npz_sha256": sha_file(ROOT / f"results/inference_cf/p2tta0/run1/rows/{i:02d}_final_masters_fp32.npz")}
    rows, ok = [], True
    for u in ids:
        b = base[u]
        fr, ar = json.loads((ROOT / b["forced_path"]).read_text()), json.loads((ROOT / b["auto_path"]).read_text())
        f, a = fr["systems"]["S0"], ar["systems"]["B0_AUTO"]
        y_B, y_A = clean_teacher(f["tokens"], CB, eos, specials), clean_teacher(a["tokens"], CB, eos, specials)
        audio = seqm["audio"][u]
        fine = (fr["identity"] == u == ar["identity"] and fr["status"] == ar["status"] == "ok" and y_B == f["tokens"] and y_A == a["tokens"]
                and t0run.audio_fingerprint_64k(audio["path"]) == audio["sha256"] == rr[u]["audio_sha256"])
        ok &= bool(fine)
        rows.append({"utterance_id": u, "dialogue_id": dlg[u], "group": b["group"], "audio_path": audio["path"],
                     "audio_fingerprint_64k_sizeprefixed": audio["sha256"], "audio_full_sha256": t0run.audio_full_sha256(audio["path"]),
                     "forced_row": b["forced_path"], "forced_row_sha256": b["forced_sha256"], "auto_row": b["auto_path"],
                     "auto_row_sha256": b["auto_sha256"], "y_B": y_B, "y_B_terminated": f["terminated"], "y_B_text": f["text"],
                     "y_B_valid_mask": valid_mask(y_B, sup, beg, specials, eos), "y_A": y_A, "y_A_terminated": a["terminated"],
                     "y_A_text": a["text"], "y_A_valid_mask": valid_mask(y_A, sup, beg, specials, eos),
                     "origin": "reuse_TTA0_R_A2" if u in anc else "new", "ancestor": anc.get(u)})
    checks["teachers_resolved"] = bool(ok)
    t0seal = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta0/pseudo_sealed.json").read_text())["rows"]}
    checks["ancestor_teachers_identical"] = all(r["y_A"] == t0seal[r["utterance_id"]]["y_A"] and r["y_A_valid_mask"] == t0seal[r["utterance_id"]]["y_A_valid_mask"]
                                                and r["y_B"] == t0seal[r["utterance_id"]]["y_B"] for r in rows if r["origin"] != "new")
    new = [r["utterance_id"] for r in rows if r["origin"] == "new"]
    plan = {"schema": SCHEMA + "_plan", "objective": "A2", "selection_sha256": sha_file(ROOT / SELECTION), "checks": checks, "ids": ids,
            "reuse_ids": [r["utterance_id"] for r in rows if r["origin"] != "new"], "new_ids": new,
            "planned": {"optimizer_steps": 2 * len(new), "backwards": 2 * len(new) + 1, "teacher_forwards": 5 * len(new) + 1,
                        "final_decodes": len(new), "theta0_integrity_decodes": len(new), "encoder_passes": len(new)},
            "suppression": {"suppress": sup, "begin": beg}, "eos": eos, "model_files": mh, "environment": env, "rows": rows,
            "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    if plan["planned"]["optimizer_steps"] > c["compute"]["TTA1_optimizer_steps_max"]:
        raise SystemExit("BLOCK: optimizer budget")
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "reuse": len(plan["reuse_ids"]), "new": len(new), "planned": plan["planned"], "checks": checks}))


SOURCES = ("docs/inference_cf/P2_TTA_FUNNEL_SPEC.md", "docs/inference_cf/P2_TTA_FUNNEL_CODEX_DESIGN.md", CONFIG, TTA0_CONFIG, PANEL,
           "docs/inference_cf/P2_TTA1_INHERITANCE_CONTRACT.md", SELECTION, R_AUDIT, PLAN, "src/csasr/inference_cf/episodic_tta.py",
           "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_p2tta0_audit.py", "experiments/inference_cf_p2tta_funnel.py",
           "experiments/inference_cf_p2tta_funnel_analyze.py", "experiments/inference_cf_p2tta_funnel_audit.py",
           "slurm/inference_cf_p2tta_funnel.sbatch", "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/lss/sites.py", "src/csasr/models/whisper.py", "tests/test_inference_cf_p2tta_funnel.py")


def cmd_tta1_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{TTA1}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit TTA1 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_TTA1":
        raise ValueError("pre-TTA1 audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    c = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "tta1_run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "freeze_commit": FREEZE, "config_hash": digest(c), "objective": "A2", "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "new_ids": plan["new_ids"], "reuse_ids": plan["reuse_ids"], "planned": plan["planned"],
           "trainables": [p["name"] for p in c["common"]["trainables"]["parameters"]], "optimization": c["common"]["optimization"],
           "rel_grad_tol_live": REL_GRAD_TOL, "role": "D-dev-select (already exposed fixed100)", "environment": prep.environment(),
           "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()}, "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)},
           "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def cmd_tta1_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import (LNGuard, TTAInvalid, common_diagnostic, compare, decoder_ln_names, forced_decode,
                                                 severe_truncation, tensor_bytes_hash)
    from csasr.lss.sites import assert_no_site_hooks, num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
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
    if num_forced_prefix_from(bundle.processor, language="zh", task="transcribe") != 4:
        raise ValueError("forced prefix")
    tok = bundle.processor.tokenizer
    eos = tok.eos_token_id
    gen = model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or eos != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    names_set = set(names)
    nonln_hash0 = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "theta0_ln_fp32.npz", **{f"p{i:03d}": guard.theta0[n].float().cpu().numpy() for i, n in enumerate(names)})
    counters = {"A2": {"teacher_forwards": 0, "backwards": 0, "optimizer_steps": 0, "final_decodes": 0}, "common_theta0_forwards": 0,
                "theta0_integrity_decodes": 0, "encoder_passes": 0, "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": guard.theta0_hash,
               "nonln_hash_start": nonln_hash0, "theta0_ln_fp32_npz_sha256": file_hash(out / "theta0_ln_fp32.npz"),
               "A2": {"adapt_sec": 0.0, "decode_sec": 0.0, "peak_alloc": 0, "peak_reserved": 0}, "theta0_decode_sec": 0.0, "encoder_sec": 0.0}
    atomic_json(out / "runtime.json", runtime)
    invalid = []
    prow = {r["utterance_id"]: r for r in plan["rows"]}
    try:
        for i, uid in enumerate(m["new_ids"]):
            s = prow[uid]
            t_u = time.time()
            row = {"identity": uid, "manifest_hash": m["manifest_hash"], "objectives": {}}
            if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                    t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
                raise TTAInvalid("audio bytes changed")
            ts = time.time()
            inputs = batch_model_inputs(bundle, [s["audio_path"]])
            with torch.inference_mode():
                h_inf = model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
            counters["encoder_passes"] += 1
            enc_inf = BaseModelOutput(last_hidden_state=h_inf)
            h_train = h_inf.clone()
            if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
                raise TTAInvalid("encoder clone")
            enc_train = BaseModelOutput(last_hidden_state=h_train)
            runtime["encoder_sec"] += time.time() - ts
            ts = time.time()
            if not guard.verify():
                raise TTAInvalid("theta0 not resident")
            d0 = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_integrity_decodes"] += 1
            runtime["theta0_decode_sec"] += time.time() - ts
            row["theta0_decode"] = {"tokens_equal_S0": d0["tokens"] == s["y_B"], "terminated": d0["terminated"],
                                    "terminated_equal_S0": d0["terminated"] == s["y_B_terminated"], "length": d0["length"]}
            if not (row["theta0_decode"]["tokens_equal_S0"] and row["theta0_decode"]["terminated_equal_S0"]):
                invalid.append(f"{uid}: theta0 forced decode != P2-SEQ S0")
                row["theta0_decode"]["tokens"] = d0["tokens"]
            # theta0 common-y_B diagnostic (descriptive; no grad, no update)
            row["common_theta0"] = t0run._summ_common(common_diagnostic(model, guard.fresh_masters(), enc_train, CB, s["y_B"], s["y_B_valid_mask"],
                                                                        suppress=suppress, begin=begin, eos=eos, partition=partition))
            counters["common_theta0_forwards"] += 1
            if i == 0:
                la = live_audit.live_objective_check(model, guard.names, enc_train, CB, s["y_A"], "A2", suppress=suppress, begin=begin,
                                                     tokenizer=tok)
                counters["audit"]["forwards"] += 1
                counters["audit"]["backwards"] += 1
                if not guard.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            assert_no_site_hooks(bundle)
            r = t0run.run_objective(bundle, guard, "A2", enc_train, enc_inf, s["y_A"], s["y_A_valid_mask"], s["y_B"], s["y_B_valid_mask"],
                                    suppress=suppress, begin=begin, eos=eos, partition=partition, counters=counters["A2"],
                                    keep_grad0=(i == 0))
            assert_no_site_hooks(bundle)
            rt2 = runtime["A2"]
            rt2["adapt_sec"] += r["log"]["adapt_sec"]
            rt2["decode_sec"] += r["log"]["decode_sec"]
            rt2["peak_alloc"] = max(rt2["peak_alloc"], torch.cuda.max_memory_allocated())
            rt2["peak_reserved"] = max(rt2["peak_reserved"], torch.cuda.max_memory_reserved())
            dec = r["decode"]
            row["objectives"]["A2"] = {**r["log"], "common_y_B": r["common_y_B"], "tokens": dec["tokens"], "text": dec["text"],
                                       "terminated": dec["terminated"], "length": dec["length"],
                                       "vs_B0_FORCED": compare(dec["tokens"], s["y_B"]), "vs_B0_AUTO": compare(dec["tokens"], s["y_A"]),
                                       "severe_truncation": severe_truncation(len(s["y_B"]), dec["length"], dec["terminated"])}
            if i == 0:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = r["log"]["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["primary_grad_l2"] = float(torch.linalg.vector_norm(g_pri.double()))
                la["rel_tol"] = REL_GRAD_TOL
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = {"A2": la}
                if not la["pass"]:
                    invalid.append(f"{uid}: live audit A2 disagreement")
            np.savez_compressed(out / "rows" / f"{i:02d}_final_masters_fp32.npz",
                                **{f"A2_p{j:03d}": r["final_masters"][n].numpy() for j, n in enumerate(names)})
            row["final_masters_npz_sha256"] = file_hash(out / "rows" / f"{i:02d}_final_masters_fp32.npz")
            del r
            row["status"] = "ok"
            row["elapsed_sec"] = time.time() - t_u
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-TTA1 {i + 1}/{len(m['new_ids'])} {uid} A2 L={row['objectives']['A2']['losses']} {row['elapsed_sec']:.1f}s", flush=True)
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
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln_hash0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-TTA1 run failed: " + runtime["failure"]["reason"])


def cmd_tta1_seal(args) -> None:
    """Seal all TTA1 outputs (hashes) before any reference access."""
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": json.loads((run / "manifest.json").read_text())["manifest_hash"], "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / TTA1 / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"])}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tta1-prepare")
    sub.add_parser("tta1-manifest").add_argument("--out", required=True)
    sub.add_parser("tta1-run").add_argument("--out", required=True)
    sub.add_parser("tta1-seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"tta1-prepare": cmd_tta1_prepare, "tta1-manifest": cmd_tta1_manifest, "tta1-run": cmd_tta1_run, "tta1-seal": cmd_tta1_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
