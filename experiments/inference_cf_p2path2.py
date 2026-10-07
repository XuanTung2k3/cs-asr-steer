#!/usr/bin/env python
"""P2-PATH2 runner (online consensus guard + short theta0 handoff). Contract: docs/inference_cf/P2_PATH2_SPEC.md,
P2_PATH2_CODEX_DESIGN.md, P2_PATH2_PANEL.json, configs/inference_cf/p2_path2.json (freeze e1c6ee7).

prepare   (CPU) verify config/panel/parent hashes, row fingerprints, C/U and C audit expectations (analysis-only),
          comparator provenance; write plan_sealed.json with the runtime-allowed inputs separated from audit fields.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_PATH2.
run       (GPU) two resident instances (theta0 immutable; A4 instance). Phase 1 all 12 (barrier): theta0 decode = B0,
          sealed language, unchanged adapt_a4 on the A4 instance, effective == checkpoint, FREE-A4 (G0) = sealed A4,
          exact reset. Phase 2 per row: materialize A4 on the A4 instance; lockstep detector; on trigger live H=3
          rollouts + 4 PATH1-scorer paths + consensus -> ONE decision for G1 and G2; G1/G2 fresh-A4 executions.
seal      (CPU) immutable output seal (before references).
The runtime controller receives only the waveform-derived encoder output, the two model states, the emitted history,
the generation config and model-owned caches. No reference, C/U, historical site, donor or AUTO output.
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

SCHEMA = "p2_path2_v1"
CONFIG = "configs/inference_cf/p2_path2.json"
PANEL = "docs/inference_cf/P2_PATH2_PANEL.json"
A4_PLAN = "results/inference_cf/p2tta_a4/plan_sealed.json"
BASE = "results/inference_cf/p2path2"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02
RUNTIME_KEYS = ("utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "classes",
                "a3_lang_id", "A4_checkpoint")             # A4 reconstruction inputs only (unchanged teacher); no controller input


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def row_fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig
    from csasr.inference_cf.path_decode import stream
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    P0 = json.loads((ROOT / P["parent_PATH0_sites"]).read_text())
    P1 = json.loads((ROOT / P["parent_PATH1_sites"]).read_text())
    checks = {"panel_bytes": sha_file(ROOT / PANEL) == cfg["panel_byte_sha256"], "frozen_committed": committed(CONFIG) and committed(PANEL),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "parents": sha_file(ROOT / P["parent_PATH0_sites"]) == P["parent_PATH0_sha256"] and sha_file(ROOT / P["parent_PATH1_sites"]) == P["parent_PATH1_sha256"]}
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    checks["suppression"] = {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])} == cfg["suppression"]
    plan4 = json.loads((ROOT / A4_PLAN).read_text())
    rows, ok = [], True
    for r, p0, p1 in zip(P["rows"], P0["rows"], P1["rows"]):
        so = r["source_outputs"]
        p4 = plan4["rows"][r["A4_index"]]
        B, A, F = stream(so["B0"]["tokens"], so["B0"]["terminated"]), stream(so["AUTO"]["tokens"], so["AUTO"]["terminated"]), stream(so["A4"]["tokens"], so["A4"]["terminated"])
        C = F != B
        fine = (row_fp(p0) == r["PATH0_row_sha256"] and row_fp(p1) == r["PATH1_row_sha256"] and all(sha_file(ROOT / so[k]["path"]) == so[k]["byte_sha256"] for k in so)
                and sha_file(ROOT / r["A4_checkpoint"]) == r["A4_checkpoint_sha256"] and B != A and r["analysis_group"] == ("C" if C else "U")
                and p4["utterance_id"] == r["utterance_id"] and p4["y_B"] == so["B0"]["tokens"] and p4["y_A"] == so["AUTO"]["tokens"]
                and not any(t >= EOS for t in so["B0"]["tokens"] + so["AUTO"]["tokens"] + so["A4"]["tokens"]))
        if C:
            a = r["audit_only_expected"]
            k = next(i for i, (x, y) in enumerate(zip(F, B)) if x != y)
            fine &= (a["k"] == k and a["common_prefix"] == B[:k] and a["b0"] == so["B0"]["tokens"][k:k + 3] and a["b4"] == so["A4"]["tokens"][k:k + 3]
                     and sha_file(ROOT / a["PATH1_score_path"]) == a["PATH1_score_sha256"])
        ok &= fine
        rows.append({"runtime": {k: (r["A4_checkpoint"] if k == "A4_checkpoint" else p4[k]) for k in RUNTIME_KEYS},
                     "analysis": {"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "group": r["analysis_group"],
                                  "B0": so["B0"], "AUTO": so["AUTO"], "A4": so["A4"], "A2": p4["A2"], "y_B_text": p4["y_B_text"], "y_A_text": p4["y_A_text"],
                                  "expected": r.get("audit_only_expected")}})
    checks["rows"] = bool(ok)
    checks["population"] = (len(rows), sum(x["analysis"]["group"] == "C" for x in rows), sum(x["analysis"]["group"] == "U" for x in rows),
                            len({x["analysis"]["dialogue_id"] for x in rows})) == (12, 5, 7, 9)
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "ids": [x["analysis"]["utterance_id"] for x in rows], "suppression": cfg["suppression"], "eos": EOS, "partition_hash": plan4["partition_hash"],
            "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = ("docs/inference_cf/P2_PATH2_SPEC.md", "docs/inference_cf/P2_PATH2_CODEX_DESIGN.md", PANEL, CONFIG, PLAN, A4_PLAN,
           "src/csasr/inference_cf/consensus_guard.py", "src/csasr/inference_cf/branch_adjudication.py", "src/csasr/inference_cf/path_decode.py",
           "src/csasr/inference_cf/script_safe_tta.py", "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/episodic_tta.py",
           "experiments/inference_cf_p2path2.py", "experiments/inference_cf_p2path2_analyze.py", "experiments/inference_cf_p2path2_audit.py",
           "experiments/inference_cf_p2tta_a4_audit.py", "slurm/inference_cf_p2path2.sbatch", "experiments/inference_cf_p2tta0.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path2.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH2 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH2":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
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
    from csasr.inference_cf.branch_adjudication import consensus, owned_clamp, score_branch
    from csasr.inference_cf.consensus_guard import decision_hash, g1_forced, g2_forced, lockstep_detect, rollout
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, tensor_bytes_hash
    from csasr.inference_cf.path_decode import clamp_decode
    from csasr.inference_cf.script_safe_tta import adapt_a4
    from csasr.inference_cf.soft_auto_tta import auto_prompt
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
    cfg_m = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    b_t0, b_a4 = load_whisper(cfg_m), load_whisper(cfg_m)                 # two independent resident instances
    for b in (b_t0, b_a4):
        if b.device != "cuda" or b.dtype != torch.bfloat16:
            raise ValueError("requires CUDA bf16")
        b.model.eval()
        b.model.requires_grad_(False)
    names = decoder_ln_names(b_a4.model)
    if names != m["trainables"] or len(names) != 194 or decoder_ln_names(b_t0.model) != names:
        raise TTAInvalid("trainable enumeration")
    g_t0, g_a4 = LNGuard(b_t0.model, names), LNGuard(b_a4.model, names)
    p_t0, p_a4 = dict(b_t0.model.named_parameters()), dict(b_a4.model.named_parameters())
    if any(p_t0[n].data_ptr() == p_a4[n].data_ptr() for n in p_t0) or g_t0.theta0_hash != g_a4.theta0_hash:
        raise TTAInvalid("instances alias storage or differ at theta0")
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in b_a4.model.named_parameters() if n not in names_set])
    if nonln0 != tensor_bytes_hash([p for n, p in b_t0.model.named_parameters() if n not in names_set]):
        raise TTAInvalid("non-LN weights differ between instances")
    tok = b_a4.processor.tokenizer
    eos = tok.eos_token_id
    gen = b_a4.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or eos != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    if partition["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
    (out / "rows").mkdir(parents=True, exist_ok=True)
    counters = {"A4": {}, "theta0_decodes": 0, "G0_decodes": 0, "detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0,
                "G1_decodes": 0, "G2_decodes": 0, "encoder_passes": 0, "detect_language_calls": 0, "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": g_t0.theta0_hash,
               "nonln_hash_start": nonln0, "phase1_sec": 0.0, "phase2_sec": 0.0}
    atomic_json(out / "runtime.json", runtime)
    invalid, rows_out, keep = [], [], []
    try:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        for i, x in enumerate(plan["rows"]):
            s, an_ = x["runtime"], x["analysis"]
            uid = s["utterance_id"]
            row = {"identity": uid, "manifest_hash": m["manifest_hash"]}
            if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                    t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
                raise TTAInvalid("audio bytes changed")
            inputs = batch_model_inputs(b_t0, [s["audio_path"]])
            with torch.inference_mode():
                h_inf = b_t0.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
            counters["encoder_passes"] += 1
            enc_inf = BaseModelOutput(last_hidden_state=h_inf)
            h_train = h_inf.clone()
            if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
                raise TTAInvalid("encoder clone")
            enc_train = BaseModelOutput(last_hidden_state=h_train)
            if not (g_t0.verify() and g_a4.verify()):
                raise TTAInvalid("theta0 not resident")
            d0 = clamp_decode(b_t0, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_decodes"] += 1
            row["theta0_equals_B0"] = d0["tokens"] == an_["B0"]["tokens"] and d0["terminated"] == an_["B0"]["terminated"]
            with torch.no_grad():
                lang = int(b_t0.model.detect_language(input_features=inputs["input_features"], generation_config=b_t0.model.generation_config)[0])
            counters["detect_language_calls"] += 1
            row["lang_equals_sealed"] = lang == s["a3_lang_id"]
            cA = auto_prompt(lang)
            if i == 0:
                la = live_audit.live_safe_kl_check(b_a4.model, g_a4.names, enc_train, cA, CB, s["y_A"], partition, suppress=suppress, begin=begin, tokenizer=tok)
                counters["audit"]["forwards"] += 3
                counters["audit"]["backwards"] += 1
            r = adapt_a4(b_a4.model, g_a4, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], s["classes"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A4"], keep_grad0=(i == 0))
            with np.load(ROOT / s["A4_checkpoint"]) as z:
                ck = {n: torch.from_numpy(z[f"A4_p{j:03d}"]).to(torch.bfloat16) for j, n in enumerate(names)}
            row["effective_equals_checkpoint_bf16"] = all(torch.equal(r["effective"][n].cpu(), ck[n]) for n in names)
            try:
                g_a4.materialize(r["effective"])
                row["A4_state_hash"] = g_a4.current_hash()
                free = clamp_decode(b_a4, enc_inf, CB, max_new_tokens=MAX_NEW)
                counters["G0_decodes"] += 1
            finally:
                g_a4.restore()
            row["reset_ok_phase1"] = g_a4.verify() and g_a4.current_hash() == g_a4.theta0_hash and g_t0.verify()
            row["G0"] = {"tokens": free["tokens"], "terminated": free["terminated"]}
            row["G0_equals_sealed_A4"] = free["tokens"] == an_["A4"]["tokens"] and free["terminated"] == an_["A4"]["terminated"]
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
            for k in ("theta0_equals_B0", "lang_equals_sealed", "effective_equals_checkpoint_bf16", "reset_ok_phase1", "G0_equals_sealed_A4"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            rows_out.append(row)
            keep.append((enc_inf, {n: v.cpu() for n, v in r["effective"].items()}))
            del r
            print(f"P2-PATH2 phase1 {i + 1}/12 {uid} G0==A4 {row['G0_equals_sealed_A4']} ckpt {row['effective_equals_checkpoint_bf16']}", flush=True)
        runtime["phase1_sec"] = time.time() - t1
        if invalid:
            raise TTAInvalid("phase-1 barrier failed: " + "; ".join(invalid))
        # ---------------- Phase 2: online guard ----------------
        t2 = time.time()
        for i, x in enumerate(plan["rows"]):
            enc_inf, eff = keep[i]
            row = rows_out[i]
            uid = x["runtime"]["utterance_id"]
            try:
                g_a4.materialize({n: v.to(g_a4.theta0[n].device) for n, v in eff.items()})
                h_a4, h_t0 = g_a4.current_hash(), g_t0.current_hash()
                if h_a4 != row["A4_state_hash"] or h_t0 != g_t0.theta0_hash:
                    raise TTAInvalid("instance states")
                det = lockstep_detect(b_t0, b_a4, enc_inf, CB, owners={"theta0": h_t0, "A4": h_a4},
                                      hash_fns={"theta0": g_t0.current_hash, "A4": g_a4.current_hash}, max_new=MAX_NEW)
                counters["detector_runs"] += 1
                row["detector"] = det
                if not (det["state_locked"] and det["fed_identical"] and det["positions_ok"]):
                    invalid.append(f"{uid}: detector integrity")
                if not det["trigger"]:
                    row["decision"] = None
                    row["G1"] = {"tokens": det["prefix"], "terminated": det["terminated"]}
                    row["G2"] = {"tokens": det["prefix"], "terminated": det["terminated"]}
                    row["no_trigger_identity"] = row["G1"] == row["G0"] == row["G2"]
                    if not row["no_trigger_identity"]:
                        invalid.append(f"{uid}: no-trigger output != G0")
                else:
                    counters["triggers"] += 1
                    k, pre = det["k"], det["prefix"]
                    b0 = rollout(b_t0, enc_inf, CB, pre, owner_hash=h_t0, hash_fn=g_t0.current_hash)
                    b4 = rollout(b_a4, enc_inf, CB, pre, owner_hash=h_a4, hash_fn=g_a4.current_hash)
                    counters["rollouts"] += 2
                    for nm, b, am in (("b0", b0, det["argmax_theta0"]), ("b4", b4, det["argmax_A4"])):
                        if not (b["prefix_ok"] and b["positions_ok"] and b["state_locked"]) or b["H_eff"] < 1 or b["tokens"][0] != am:
                            invalid.append(f"{uid}: rollout {nm} integrity/EOS-first")
                    if b0["H_eff"] < 1 or b4["H_eff"] < 1:
                        raise TTAInvalid(f"{uid}: EOS-first branch (frozen INVALID)")
                    sc, paths = {}, {}
                    for mdl, bundle, hh, hf in (("theta0", b_t0, h_t0, g_t0.current_hash), ("A4", b_a4, h_a4, g_a4.current_hash)):
                        for br_name, br in (("b0", b0), ("b4", b4)):
                            p = score_branch(bundle, enc_inf, CB, pre, br["tokens"], owner={"state": mdl, "state_hash": hh, "row": uid,
                                                                                               "condition": f"score_{mdl}_{br_name}"}, state_hash_fn=hf)
                            counters["score_paths"] += 1
                            paths[f"{mdl}_{br_name}"] = p
                            if not (p["state_locked"] and p["prefix_ok"] and p["positions_ok"] and p["fed_ok"]):
                                invalid.append(f"{uid}: score {mdl}_{br_name} integrity")
                    cons = consensus({"B": paths["theta0_b0"]["score"], "ALT": paths["theta0_b4"]["score"]},
                                     {"B": paths["A4_b0"]["score"], "ALT": paths["A4_b4"]["score"]})
                    dec = {"k": k, "prefix": pre, "b0": b0, "b4": b4, "scores": {kk: {"logprobs": v["logprobs"], "score": v["score"], "H_eff": v["H_eff"]}
                                                                               for kk, v in paths.items()},
                           "winner": "theta0" if cons["choice"] == "B" else "A4", "margin": cons["margin_B_minus_ALT"], "S_cons": cons["S_cons"],
                           "strict": cons["strict_B_win"]}
                    dec["decision_hash"] = decision_hash(dec)
                    row["decision"] = dec
                    row["score_paths"] = {kk: {kq: v[kq] for kq in ("owner", "state_hash_before", "state_hash_after", "state_locked", "prefix_ok",
                                                                   "positions_ok", "fed_ok")} for kk, v in paths.items()}
                    f1 = g1_forced(dec)
                    d1 = owned_clamp(b_a4, enc_inf, CB, site=k, forced=f1, expected_prefix=pre, owner={"state": "A4", "state_hash": h_a4, "row": uid,
                                                                                                    "condition": "G1"}, state_hash_fn=g_a4.current_hash)
                    counters["G1_decodes"] += 1
                    row["G1"] = {"tokens": d1["tokens"], "terminated": d1["terminated"], "trace": d1["trace"], "state_locked": d1["state_locked"],
                                 "forced": f1, "decision_hash": dec["decision_hash"]}
                    f2 = g2_forced(dec, eos)
                    if f2 is None:
                        row["G2"] = {"tokens": d1["tokens"], "terminated": d1["terminated"], "mode": "A4_winner_ordinary", "decision_hash": dec["decision_hash"]}
                    else:
                        d2 = owned_clamp(b_a4, enc_inf, CB, site=k, forced=f2, expected_prefix=pre, owner={"state": "A4", "state_hash": h_a4, "row": uid,
                                                                                                        "condition": "G2_replay"}, state_hash_fn=g_a4.current_hash)
                        counters["G2_decodes"] += 1
                        row["G2"] = {"tokens": d2["tokens"], "terminated": d2["terminated"], "trace": d2["trace"], "state_locked": d2["state_locked"],
                                     "forced": f2, "mode": "theta0_handoff_fresh_A4_replay", "decision_hash": dec["decision_hash"]}
                    for nm in ("G1", "G2"):
                        tr = row[nm].get("trace")
                        if tr is not None and not (row[nm]["state_locked"] and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"]
                                                   and [f["token"] for f in tr["forced"]] == row[nm]["forced"][:len(tr["forced"])]):
                            invalid.append(f"{uid}: {nm} execution integrity")
                    if dec["winner"] == "A4" and not (row["G1"]["tokens"] == row["G0"]["tokens"] and row["G1"]["terminated"] == row["G0"]["terminated"]):
                        invalid.append(f"{uid}: A4 winner but G1 != G0")
            finally:
                g_a4.restore()
            row["reset_ok_phase2"] = g_a4.verify() and g_a4.current_hash() == g_a4.theta0_hash and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash
            if not row["reset_ok_phase2"]:
                invalid.append(f"{uid}: reset phase 2")
            row["status"] = "ok"
            atomic_json(out / "rows" / f"{i:02d}.json", row)
            dd = row["decision"]
            print(f"P2-PATH2 phase2 {i + 1}/12 {uid} trigger {row['detector']['trigger']}"
                  + (f" k {dd['k']} winner {dd['winner']} margin {dd['margin']:.3f}" if dd else ""), flush=True)
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
            g_a4.restore()
        except Exception:
            pass
    runtime["peak_alloc"] = int(torch.cuda.max_memory_allocated())
    runtime["peak_reserved"] = int(torch.cuda.max_memory_reserved())
    runtime["reset_final_ok"] = bool(g_a4.verify() and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in b_a4.model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0 == tensor_bytes_hash([p for n, p in b_t0.model.named_parameters() if n not in names_set])
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for b in (b_t0, b_a4) for p in b.model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-PATH2 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "primary_valid": prim["valid"], "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"]), "primary_valid": doc["primary_valid"]}))


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
