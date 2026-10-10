#!/usr/bin/env python
"""P2-TTA-A3 runner (soft AUTO-KL conditioning-gap debug). Contract: docs/inference_cf/P2_TTA_A3_SPEC.md,
P2_TTA_A3_CLAUDE_DESIGN.md, configs/inference_cf/p2_tta_a3.json, docs/inference_cf/P2_TTA_A3_PANEL.json.

panel     (CPU) reference-free 24-panel from sealed theta0 FORCED/AUTO decoded text of the fixed-100 panel.
prepare   (CPU) teacher/reuse plan: y_B (P2-SEQ S0), y_A (historical AUTO), A2 outputs (TTA1/TTA0-R rows), masks.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_TTA_A3.
run       (GPU) per utterance: theta0 encoder once; theta0 forced decode = S0; AUTO prompt recovered with the historical
          `detect_language(input_features=...)` call; theta0 AUTO replay (must reproduce historical AUTO text);
          A3 episode (fresh fp32 LN masters + fresh AdamW, 2 steps, forward KL(q_AUTO || p_FORCED) on the common y_A
          path); final forced-ZH decode; exact reset. First row: independent live loss/gradient check.
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

SCHEMA = "p2_tta_a3_v1"
CONFIG = "configs/inference_cf/p2_tta_a3.json"
PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
PARENT = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
FUNNEL = "configs/inference_cf/p2_tta_funnel.json"
BASE = "results/inference_cf/p2tta_a3"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
TRANSCRIBE, NOTIMESTAMPS, SOT = 50360, 50364, 50258
MAX_NEW = 200
REL_GRAD_TOL = 0.02
TARGET, MAX_D, MAX_PER_DIALOGUE = 24, 16, 2


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


# ---- reference-free panel ---------------------------------------------------------------------------------------

def build_panel(parent_rows: list[dict], forced_text: dict, auto_text: dict) -> list[dict]:
    """D = exact stored theta0 AUTO text != FORCED text; A = equal. All D if |D| <= 16, else dialogue round-robin
    (dialogue first-appearance order, parent order within dialogue). Controls fill to 24 from A: first one per
    dialogue NOT represented by D (dialogue first-appearance order, parent order within), then round-robin over
    D-represented dialogues with < 2 selected, then any dialogue with < 2, then (only if still short) any dialogue.
    No reference/quality input."""
    order = []
    for r in parent_rows:
        if r["dialogue_id"] not in order:
            order.append(r["dialogue_id"])
    grp = {r["utterance_id"]: ("D" if auto_text[r["utterance_id"]] != forced_text[r["utterance_id"]] else "A") for r in parent_rows}
    idx = {r["utterance_id"]: i for i, r in enumerate(parent_rows)}
    by_d = {d: [r for r in parent_rows if r["dialogue_id"] == d] for d in order}
    D = [r for r in parent_rows if grp[r["utterance_id"]] == "D"]
    if len(D) > MAX_D:
        queues = {d: [r for r in by_d[d] if grp[r["utterance_id"]] == "D"] for d in order}
        D = []
        while len(D) < MAX_D:
            for d in order:
                if queues[d] and len(D) < MAX_D:
                    D.append(queues[d].pop(0))
    sel = list(D)
    count = {d: sum(r["dialogue_id"] == d for r in sel) for d in order}
    avail = {d: [r for r in by_d[d] if grp[r["utterance_id"]] == "A"] for d in order}

    def take(d):
        r = avail[d].pop(0)
        sel.append(r)
        count[d] += 1

    for d in order:                                  # 1) unrepresented dialogues, one each
        if len(sel) < TARGET and count[d] == 0 and avail[d]:
            take(d)
    represented = [d for d in order if any(r["dialogue_id"] == d for r in D)]
    for pool, cap in ((represented, MAX_PER_DIALOGUE), (order, MAX_PER_DIALOGUE), (order, None)):
        progress = True                              # 2) round-robin over D-represented dialogues (<2 selected);
        while len(sel) < TARGET and progress:        # 3) any dialogue <2; 4) only if still required, any dialogue
            progress = False
            for d in pool:
                if len(sel) < TARGET and avail[d] and (cap is None or count[d] < cap):
                    take(d)
                    progress = True
    if len(sel) != TARGET or len({r["utterance_id"] for r in sel}) != TARGET:
        raise ValueError("cannot build 24-panel")
    return [{"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "group": grp[r["utterance_id"]],
             "parent_index": idx[r["utterance_id"]]} for r in sel]


def cmd_panel(args) -> None:
    f = json.loads((ROOT / FUNNEL).read_text())
    parent = json.loads((ROOT / PARENT).read_text())
    if sha_file(ROOT / PARENT) != f["TTA1"]["panel_sha256"]:
        raise ValueError("parent panel bytes")
    base = {b["utterance_id"]: b for b in f["baseline100"]}
    ft, at, src = {}, {}, {}
    for r in parent["rows"]:
        b = base[r["utterance_id"]]
        if sha_file(ROOT / b["forced_path"]) != b["forced_sha256"] or sha_file(ROOT / b["auto_path"]) != b["auto_sha256"]:
            raise ValueError("baseline bytes")
        ft[r["utterance_id"]] = json.loads((ROOT / b["forced_path"]).read_text())["systems"]["S0"]["text"]
        at[r["utterance_id"]] = json.loads((ROOT / b["auto_path"]).read_text())["systems"]["B0_AUTO"]["text"]
        src[r["utterance_id"]] = {"forced_path": b["forced_path"], "forced_sha256": b["forced_sha256"],
                                  "auto_path": b["auto_path"], "auto_sha256": b["auto_sha256"]}
    rows = build_panel(parent["rows"], ft, at)
    nD = sum(1 for u in ft if ft[u] != at[u])
    doc = {"schema": "p2_tta_a3_panel_v1", "role": "already-exposed D-dev-select (fixed-100 subset, reference-free)",
           "parent": PARENT, "parent_byte_sha256": sha_file(ROOT / PARENT),
           "selection": "D = exact stored theta0 AUTO decoded text != FORCED decoded text (no normalization); all D if |D|<=16; "
                        "controls from A: one per D-unrepresented dialogue (dialogue first appearance, parent order), then "
                        "round-robin over D-represented dialogues with <2 selected, then any dialogue <2, then any only if required; no references/"
                        "errors/durations/A2 outcomes/quality",
           "parent_D": nD, "parent_A": len(ft) - nD, "rows": [{**r, **src[r["utterance_id"]]} for r in rows]}
    out = ROOT / PANEL
    if out.exists():
        raise FileExistsError("panel exists; never overwrite")
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"sha256": sha_file(out), "D": sum(r["group"] == "D" for r in rows), "A": sum(r["group"] == "A" for r in rows),
                      "dialogues": len({r["dialogue_id"] for r in rows}),
                      "max_per_dialogue": max(sum(x["dialogue_id"] == r["dialogue_id"] for x in rows) for r in rows)}))


# ---- prepare: teacher / reuse plan (CPU, no reference) --------------------------------------------------------------

def resolve_a2(u: str, t1plan: dict) -> dict:
    """A2 output exactly as the audited TTA1 evaluator resolved it (new TTA1 row, or reused TTA0-R ancestor)."""
    r = next(x for x in t1plan["rows"] if x["utterance_id"] == u)
    if r["origin"] == "new":
        rel = f"results/inference_cf/p2tta_funnel/tta1/run1/rows/{t1plan['new_ids'].index(u):02d}.json"
    else:
        rel = r["ancestor"]["row"]
    o = json.loads((ROOT / rel).read_text())["objectives"]["A2"]
    return {"row": rel, "row_sha256": sha_file(ROOT / rel), "origin": r["origin"], "tokens": o["tokens"], "text": o["text"],
            "terminated": o["terminated"], "length": o["length"]}


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2tta0 as t0run
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.episodic_tta import clean_teacher, levenshtein, special_ids, valid_mask
    cfg = json.loads((ROOT / CONFIG).read_text())
    checks = {"anchors": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "panel_bytes": sha_file(ROOT / PANEL) == cfg["panel"]["byte_sha256"], "config_committed": committed(CONFIG) and committed(PANEL)}
    panel = json.loads((ROOT / PANEL).read_text())
    t1plan = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())
    t1seal = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/output_seal.json").read_text())
    seqm = json.loads((ROOT / "results/inference_cf/p2seq/run1/manifest.json").read_text())
    proc = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    tok = proc.tokenizer
    eos, specials = tok.eos_token_id, special_ids(tok)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    mh = prep.model_hashes()
    checks["model_files"] = mh == seqm["model"]["files"]
    rows, ok = [], True
    for pr in panel["rows"]:
        u = pr["utterance_id"]
        fr, ar = json.loads((ROOT / pr["forced_path"]).read_text()), json.loads((ROOT / pr["auto_path"]).read_text())
        f, a = fr["systems"]["S0"], ar["systems"]["B0_AUTO"]
        y_B, y_A = clean_teacher(f["tokens"], CB, eos, specials), clean_teacher(a["tokens"], CB, eos, specials)
        a2 = resolve_a2(u, t1plan)
        in_seal = a2["row"] not in t1seal["files"] or t1seal["files"][a2["row"]] == a2["row_sha256"]
        audio = seqm["audio"][u]
        fine = (sha_file(ROOT / pr["forced_path"]) == pr["forced_sha256"] and sha_file(ROOT / pr["auto_path"]) == pr["auto_sha256"]
                and y_B == f["tokens"] and y_A == a["tokens"] and in_seal and t0run.audio_fingerprint_64k(audio["path"]) == audio["sha256"]
                and (pr["group"] == "D") == (f["text"] != a["text"]))
        ok &= bool(fine)
        rows.append({"utterance_id": u, "dialogue_id": pr["dialogue_id"], "group": pr["group"], "audio_path": audio["path"],
                     "audio_fingerprint_64k_sizeprefixed": audio["sha256"], "audio_full_sha256": t0run.audio_full_sha256(audio["path"]),
                     "y_B": y_B, "y_B_terminated": f["terminated"], "y_B_text": f["text"], "y_A": y_A, "y_A_terminated": a["terminated"],
                     "y_A_text": a["text"], "y_A_valid_mask": valid_mask(y_A, sup, beg, specials, eos), "A2": a2,
                     "d_BA": levenshtein(y_B, y_A), "d_A2A": levenshtein(a2["tokens"], y_A)})
    checks["rows"] = bool(ok)
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "ids": [r["utterance_id"] for r in rows], "suppression": {"suppress": sup, "begin": beg}, "eos": eos, "model_files": mh,
            "environment": prep.environment(), "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks, "valid_A": sum(sum(r["y_A_valid_mask"]) for r in rows),
                      "d_BA_D": sum(r["d_BA"] for r in rows if r["group"] == "D"), "d_A2A_D": sum(r["d_A2A"] for r in rows if r["group"] == "D")}))


SOURCES = ("docs/inference_cf/P2_TTA_A3_SPEC.md", "docs/inference_cf/P2_TTA_A3_CLAUDE_DESIGN.md", CONFIG, PANEL, PLAN,
           "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2tta_a3.py",
           "experiments/inference_cf_p2tta_a3_analyze.py", "experiments/inference_cf_p2tta_a3_audit.py", "slurm/inference_cf_p2tta_a3.sbatch",
           "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/lss/sites.py", "src/csasr/models/whisper.py", "tests/test_inference_cf_p2tta_a3.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit A3 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_TTA_A3":
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


# ---- GPU run -----------------------------------------------------------------------------------------------------

def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, forced_decode, levenshtein, severe_truncation, tensor_bytes_hash
    from csasr.inference_cf.soft_auto_tta import adapt_a3, auto_prompt
    from csasr.lss.sites import assert_no_site_hooks
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
    import experiments.inference_cf_p2tta_a3_audit as live_audit
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
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in model.named_parameters() if n not in names_set])
    (out / "rows").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "theta0_ln_fp32.npz", **{f"p{i:03d}": guard.theta0[n].float().cpu().numpy() for i, n in enumerate(names)})
    counters = {"A3": {}, "theta0_forced_decodes": 0, "auto_replay_decodes": 0, "final_decodes": 0, "encoder_passes": 0,
                "detect_language_calls": 0, "audit": {"forwards": 0, "backwards": 0}}
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
            with torch.no_grad():                                 # historical generate(language=None) detection call
                lang = int(model.detect_language(input_features=inputs["input_features"], generation_config=model.generation_config)[0])
            counters["detect_language_calls"] += 1
            cA = auto_prompt(lang)
            rep = forced_decode(bundle, enc_inf, cA, max_new_tokens=MAX_NEW)
            counters["auto_replay_decodes"] += 1
            runtime["baseline_decode_sec"] += time.time() - ts
            row["auto_condition"] = {"lang_id": lang, "lang_token": tok.convert_ids_to_tokens(lang), "cA": cA, "identical_to_cB": cA == CB,
                                     "replay_text_equal": rep["text"] == s["y_A_text"], "replay_tokens_equal": rep["tokens"] == s["y_A"],
                                     "replay_terminated": rep["terminated"]}
            if not (row["theta0_forced"]["tokens_equal_S0"] and row["theta0_forced"]["terminated_equal_S0"]):
                invalid.append(f"{uid}: theta0 forced decode != S0")
            if not row["auto_condition"]["replay_text_equal"]:
                invalid.append(f"{uid}: AUTO replay != historical AUTO text")
                row["auto_condition"]["replay_tokens"] = rep["tokens"]
            if i == 0:
                la = live_audit.live_kl_check(model, guard.names, enc_train, cA, CB, s["y_A"], suppress=suppress, begin=begin, tokenizer=tok)
                counters["audit"]["forwards"] += 2
                counters["audit"]["backwards"] += 1
                if not guard.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            assert_no_site_hooks(bundle)
            ts = time.time()
            r = adapt_a3(model, guard, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A3"], keep_grad0=(i == 0))
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
            d_A3A = levenshtein(dec["tokens"], s["y_A"])
            row["A3"] = {**r["log"], "tokens": dec["tokens"], "text": dec["text"], "terminated": dec["terminated"], "length": dec["length"],
                         "reset_ok": reset_ok, "end_hash": guard.current_hash(), "other_versions_unchanged": guard.other_versions() == v_before,
                         "equal_B0_FORCED": dec["tokens"] == s["y_B"], "d_A3B": levenshtein(dec["tokens"], s["y_B"]),
                         "d_BA": s["d_BA"], "d_A2A": s["d_A2A"], "d_A3A": d_A3A,
                         "movement": "CLOSER" if d_A3A < s["d_BA"] else ("SAME" if d_A3A == s["d_BA"] else "FARTHER"),
                         "severe_truncation": severe_truncation(len(s["y_B"]), dec["length"], dec["terminated"])}
            if row["A3"]["noop"] and not row["A3"]["equal_B0_FORCED"]:
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
                    invalid.append(f"{uid}: live audit A3 disagreement")
            np.savez_compressed(out / "rows" / f"{i:02d}_final_masters_fp32.npz",
                                **{f"A3_p{j:03d}": r["final_masters"][n].numpy() for j, n in enumerate(names)})
            row["final_masters_npz_sha256"] = file_hash(out / "rows" / f"{i:02d}_final_masters_fp32.npz")
            del r
            row["status"] = "ok"
            row["elapsed_sec"] = time.time() - t_u
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            print(f"P2-TTA-A3 {i + 1}/24 {uid} {s['group']} lang={row['auto_condition']['lang_token']} Dcond={row['A3']['d_cond']} "
                  f"{row['A3']['movement']} {row['elapsed_sec']:.1f}s", flush=True)
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
        raise SystemExit("P2-TTA-A3 run failed: " + runtime["failure"]["reason"])


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
    sub.add_parser("panel")
    sub.add_parser("prepare")
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"panel": cmd_panel, "prepare": cmd_prepare, "manifest": cmd_manifest, "run": cmd_run, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
