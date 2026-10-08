#!/usr/bin/env python
"""P2-PATH5 runner (frozen A2 + G1A abstaining consensus guard on FULL300; primary NEW200). Contract:
docs/inference_cf/P2_PATH5_SPEC.md, P2_PATH5_CLAUDE_DESIGN.md, P2_PATH5_PANEL.json, configs/inference_cf/p2_path5.json
(freeze d1d057d).

prepare   (CPU) verify config/panel/source hashes, FULL300/FIXED100/NEW200 relation and hashes from IDs/dialogues (role manifest
          ID/dialogue/role columns only), audio and sealed AUTO/B0/A2/PATH4/OPP0 fingerprints; resolve NEW200 teachers exactly like
          TTA1 (clean_teacher + valid_mask); write plan_sealed.json (runtime inputs separated from audit/analysis). No reference.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_PATH5.
run       (GPU) ONE job, two resident instances. FIXED100 barrier: phase 1 exact A2 reconstruction 100/100 (+ first-row live
          check); phase 2 A2+G1A == OPP0 derived target, content rows == sealed PATH4 decisions, EOS/no-trigger rows == A2; any
          failure aborts before NEW200. NEW200: theta0 decode (B0), unchanged A2 episode, G1A, exact reset.
seal      (CPU) immutable output seal (before references).
The controller (``online_g1a``) receives only the encoder output, the two model states, the emitted history, the generation
config, model-owned caches and the row's own live ordinary A2 output. No reference, partition, row ID or historical target.
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

SCHEMA = "p2_path5_v1"
CONFIG = "configs/inference_cf/p2_path5.json"
PANEL = "docs/inference_cf/P2_PATH5_PANEL.json"
SPEC = "docs/inference_cf/P2_PATH5_SPEC.md"
DESIGN = "docs/inference_cf/P2_PATH5_CLAUDE_DESIGN.md"
FULL = "results/inference_cf/p2_A_r1_L16/panel.json"
FIXED = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
DERIVED = "results/inference_cf/p2opp0/derived_g1a.json"
ROLE = "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet"
BASE = "results/inference_cf/p2path5"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02
PARTS = ("FULL300", "FIXED100", "NEW200")
RUNTIME_KEYS = ("utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def row_fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def membership_hash(name: str, ids, dlg: dict) -> str:
    return row_fp({"partition": name, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in ids]})


def stream(tokens, term):
    return list(tokens) + ([EOS] if term == "eos" else [])


def partitions(full: list, fixed: list) -> dict:
    """Pure ID-only partitions: NEW200 = FULL300 - FIXED100 in FULL300 order."""
    if len(set(full)) != len(full) or len(set(fixed)) != len(fixed) or not set(fixed) < set(full):
        raise ValueError("FIXED100 must be a strict duplicate-free subset of FULL300")
    f = set(fixed)
    return {"FULL300": list(full), "FIXED100": list(fixed), "NEW200": [u for u in full if u not in f]}


def selected(path: str, selector: str) -> dict:
    a, b = selector.split(".")
    return json.loads((ROOT / path).read_text())[a][b]


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2tta0 as t0run
    import pyarrow.parquet as pq
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import clean_teacher, special_ids, valid_mask
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    checks = {"panel_bytes": sha_file(ROOT / PANEL) == cfg["panel_byte_sha256"],
              "frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN)),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items())}
    fullp, fx = json.loads((ROOT / FULL).read_text()), json.loads((ROOT / FIXED).read_text())
    checks["fixed_parent_is_full300"] = fx["parent_panel"] == FULL and fx["parent_panel_sha256"] == sha_file(ROOT / FULL) == P["parent_full300"]["byte_sha256"]
    full = [r["utterance_id"] for r in fullp["rows"]]
    fixed = [r["utterance_id"] for r in fx["rows"]]
    role = pq.read_table(ROLE, columns=["utterance_id", "dialogue_id", "role"]).to_pylist()     # ID/dialogue/role columns ONLY
    rm = {}
    for x in role:
        if x["utterance_id"] in set(full):
            rm.setdefault(x["utterance_id"], []).append((x["dialogue_id"], x["role"]))
    checks["role_manifest"] = set(rm) == set(full) and all(len(v) == 1 and v[0][1] == "D-dev-select" for v in rm.values())
    dlg = {u: rm[u][0][0] for u in full}
    parts = partitions(full, fixed)
    checks["partitions"] = (all(parts[k] == P["partitions"][k]["ids"] for k in PARTS) and {k: len(v) for k, v in parts.items()} == cfg["counts"]
                            and all(membership_hash(k, parts[k], dlg) == cfg["partition_sha256"][k] for k in PARTS)
                            and {k: len({dlg[u] for u in v}) for k, v in parts.items()} == cfg["dialogues"] == {"FULL300": 20, "FIXED100": 20, "NEW200": 20}
                            and all(dlg[r["utterance_id"]] == r["dialogue_id"] for r in fx["rows"]))
    order = parts["FIXED100"] + parts["NEW200"]
    checks["execution_order"] = order == P["execution_order"] and row_fp(order) == cfg["execution_order_sha256"]
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    sp, sup, beg = special_ids(tok), list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["suppression"] = {"suppress": sup, "begin": beg} == cfg["suppression"]
    der = json.loads((ROOT / DERIVED).read_text())
    checks["derived_committed"] = committed(DERIVED) and sha_file(ROOT / DERIVED) == P["derived_G1A"]["byte_sha256"] \
        and der["derived_hash"] == digest({k: v for k, v in der.items() if k != "derived_hash"})
    dr = {r["utterance_id"]: r for r in der["rows"]}
    rows, bad = [], []
    for i, (pr, r) in enumerate(zip(fullp["rows"], P["rows"])):
        u = pr["utterance_id"]
        fine = r["full_index"] == i and r["utterance_id"] == u and r["dialogue_id"] == dlg[u] and r["partition"] == ("FIXED100" if u in set(fixed) else "NEW200")
        ad = r["audio"]
        fine &= (ad["path"] == pr["audio_path"] and ad["fingerprint_64k_sizeprefixed"] == pr["audio_sha256"] == t0run.audio_fingerprint_64k(ad["path"])
                 and t0run.audio_full_sha256(ad["path"]) == ad["full_sha256"])
        fine &= sha_file(ROOT / r["B0_AUTO"]["path"]) == r["B0_AUTO"]["byte_sha256"]
        ar = json.loads((ROOT / r["B0_AUTO"]["path"]).read_text())
        au = ar["systems"]["B0_AUTO"]
        fine &= ar["identity"] == u and ar["status"] == "ok" and row_fp(au["tokens"]) == r["B0_AUTO"]["tokens_sha256"] and au["terminated"] == r["B0_AUTO"]["terminated"]
        y_A = clean_teacher(au["tokens"], CB, EOS, sp)
        mask = valid_mask(y_A, sup, beg, sp, EOS)
        fine &= row_fp(y_A) == r["teacher"]["y_A_token_sha256"] and row_fp(mask) == r["teacher"]["valid_mask_sha256"]
        audit = {"full_index": i, "dialogue_id": dlg[u], "partition": r["partition"]}
        if "fixed100" in r:
            f = r["fixed100"]
            b0, a2 = selected(f["B0_FORCED"]["path"], f["B0_FORCED"]["selector"]), selected(f["A2"]["path"], f["A2"]["selector"])
            fine &= (all(sha_file(ROOT / f[k]["path"]) == f[k]["byte_sha256"] for k in ("B0_FORCED", "A2", "A2_checkpoint", "PATH4_row"))
                     and row_fp(b0["tokens"]) == f["B0_sealed"]["tokens_sha256"] and row_fp(a2["tokens"]) == f["A2_sealed"]["tokens_sha256"]
                     and f["teacher"]["y_A_token_sha256"] == r["teacher"]["y_A_token_sha256"] and f["teacher"]["valid_mask_sha256"] == r["teacher"]["valid_mask_sha256"])
            d = dr[u]
            fine &= row_fp(d["G1A"]["tokens"]) == f["G1A_derived_target"]["tokens_sha256"] and d["G1A"]["terminated"] == f["G1A_derived_target"]["terminated"]
            audit.update(fixed_index=f["fixed_index"], B0={"tokens": b0["tokens"], "terminated": b0["terminated"]},
                         A2={"tokens": a2["tokens"], "terminated": a2["terminated"], "text": a2["text"]}, A2_checkpoint=f["A2_checkpoint"]["path"],
                         PATH4_row=f["PATH4_row"]["path"], PATH4_mode=f["audit_only_PATH4_mode"],
                         G1A_target={"tokens": d["G1A"]["tokens"], "terminated": d["G1A"]["terminated"]})
        if not fine:
            bad.append(u)
        rows.append({"runtime": {"utterance_id": u, "audio_path": ad["path"], "audio_fingerprint_64k_sizeprefixed": ad["fingerprint_64k_sizeprefixed"],
                                 "audio_full_sha256": ad["full_sha256"], "y_A": y_A, "y_A_valid_mask": mask},
                     "audit": audit,
                     "analysis": {"utterance_id": u, "dialogue_id": dlg[u], "partition": r["partition"],
                                  "AUTO": {"tokens": au["tokens"], "terminated": au["terminated"], "text": au["text"]}}})
    checks["rows"] = not bad
    part = tokenizer_partition(tok)
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "failures": bad, "ids": full, "partitions": parts, "execution_order": order, "suppression": cfg["suppression"], "eos": EOS,
            "partition_hash": part["hash"], "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]) + " rows " + json.dumps(bad[:10]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = (SPEC, DESIGN, PANEL, CONFIG, PLAN, FULL, FIXED, DERIVED, "experiments/inference_cf_p2opp0_abstain.py",
           "src/csasr/inference_cf/consensus_guard.py", "src/csasr/inference_cf/branch_adjudication.py", "src/csasr/inference_cf/path_decode.py",
           "src/csasr/inference_cf/eos_boundary.py", "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2path3.py",
           "experiments/inference_cf_p2path5.py", "experiments/inference_cf_p2path5_analyze.py", "experiments/inference_cf_p2path5_audit.py",
           "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_p2tta0_audit.py", "slurm/inference_cf_p2path5.sbatch",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path5.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH5 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH5":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "execution_order": plan["execution_order"], "trainables": [p["name"] for p in cfg["A2_inherited"]["trainables"]["parameters"]],
           "budget": cfg["compute"], "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def online_g1a(b_t0, b_a2, g_t0, g_a2, enc, owner_row: str, counters: dict, eos: int, a2_free: dict) -> dict:
    """G1A abstaining consensus guard. A2 must already be materialized on ``b_a2``; ``owner_row`` is a log label;
    ``a2_free`` is this row's own live ordinary A2 output. ONE logical detector. CONTENT_G1 -> original PATH3 online_g1
    (A2 in its legacy A4 slot; its replayed detection must equal this one). EOS_BOUNDARY (either orientation) -> ABSTAIN:
    the ordinary A2 output, no scoring and no new decoder path; the controller is then disabled. No further guard."""
    from csasr.inference_cf.consensus_guard import lockstep_detect
    from csasr.inference_cf.eos_boundary import action_mode
    import experiments.inference_cf_p2path3 as path3
    fails = []
    h_a2, h_t0 = g_a2.current_hash(), g_t0.current_hash()
    det = lockstep_detect(b_t0, b_a2, enc, CB, owners={"theta0": h_t0, "A4": h_a2},
                          hash_fns={"theta0": g_t0.current_hash, "A4": g_a2.current_hash}, max_new=MAX_NEW)
    counters["logical_guard_events"] += 1
    counters["detector_runs"] += 1
    if not (det["state_locked"] and det["fed_identical"] and det["positions_ok"]):
        fails.append("detector integrity")
    if det["prefix"] != a2_free["tokens"][:len(det["prefix"])]:
        fails.append("detector prefix is not the ordinary A2 history")
    if not det["trigger"]:
        G = {"tokens": det["prefix"], "terminated": det["terminated"]}
        if G["tokens"] != a2_free["tokens"] or G["terminated"] != a2_free["terminated"]:
            fails.append("no-trigger output != ordinary A2")
        return {"mode": "NO_TRIGGER", "detector": det, "decision": None, "G1": G, "fails": fails}
    counters["triggers"] += 1
    gen, tok = b_t0.model.generation_config, b_t0.processor.tokenizer
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    mode = action_mode(det["argmax_theta0"], det["argmax_A4"], eos=eos, vocab_size=int(b_t0.model.config.vocab_size),
                       special_ids=list(tok.all_special_ids), step=det["k"], suppress=sup, begin=beg)
    if mode == "CONTENT_G1":
        inner = {"detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0}
        rec = path3.online_g1(b_t0, b_a2, g_t0, g_a2, enc, owner_row, inner, eos)
        fails.extend(rec.pop("fails"))
        if json.dumps(rec["detector"], sort_keys=True) != json.dumps(det, sort_keys=True):
            fails.append("replayed detection differs from the logical detection")
        if inner["triggers"] != 1 or inner["detector_runs"] != 1:
            fails.append("content dispatch event count")
        counters["content_events"] += 1
        counters["detector_replays"] += inner["detector_runs"]
        for k in ("rollouts", "score_paths", "G1_decodes"):
            counters[k] += inner[k]
        if rec["decision"]["winner"] != "theta0" and (rec["G1"]["tokens"] != a2_free["tokens"] or rec["G1"]["terminated"] != a2_free["terminated"]):
            fails.append("A2 winner output != ordinary A2")
        return {"mode": "CONTENT_G1", **rec, "fails": fails}
    if mode != "EOS_BOUNDARY":
        raise ValueError(f"unexpected dispatch mode {mode}")
    counters["eos_abstentions"] += 1
    return {"mode": "EOS_ABSTAIN", "detector": det, "decision": None,
            "abstain": {"k": det["k"], "theta0_action": det["argmax_theta0"], "A2_action": det["argmax_A4"],
                        "orientation": "theta0_EOS" if det["argmax_theta0"] == eos else "A2_EOS"},
            "G1": {"tokens": list(a2_free["tokens"]), "terminated": a2_free["terminated"]}, "fails": fails}


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True))


def barrier_mismatches(row: dict, p4: dict, target: dict) -> list:
    """FIXED100 phase-2 barrier: G1A == OPP0 derived target; content rows == sealed PATH4 decisions exactly."""
    bad = []
    if (row["G1"]["tokens"], row["G1"]["terminated"]) != (target["tokens"], target["terminated"]):
        bad.append("G1A != derived target")
    if row["mode"] == "CONTENT_G1":
        g1k = ("tokens", "terminated", "trace", "state_locked", "forced", "decision_hash")
        for k in ("detector", "decision", "score_paths"):
            if _norm(row.get(k)) != _norm(p4.get(k)):
                bad.append(k)
        if _norm({k: row["G1"].get(k) for k in g1k}) != _norm({k: p4["G1"].get(k) for k in g1k}):
            bad.append("G1")
        if p4["mode"] != "CONTENT_G1":
            bad.append("mode")
    else:
        if (row["G1"]["tokens"], row["G1"]["terminated"]) != (row["A2_free"]["tokens"], row["A2_free"]["terminated"]):
            bad.append("abstain/no-trigger != A2")
        if _norm(row["detector"]) != _norm(p4["detector"]) or {"NO_TRIGGER": "NO_TRIGGER", "EOS_ABSTAIN": "EOS_BOUNDARY"}[row["mode"]] != p4["mode"]:
            bad.append("detector/mode")
    return bad


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, forced_decode, special_ids, tensor_bytes_hash, valid_mask
    from csasr.lss.sites import assert_no_site_hooks
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
    if plan["plan_hash"] != m["plan_hash"] or plan["execution_order"] != m["execution_order"]:
        raise ValueError("plan")
    byid = {x["runtime"]["utterance_id"]: x for x in plan["rows"]}
    order = plan["execution_order"]
    nfix = len(plan["partitions"]["FIXED100"])
    t_setup = time.time()
    torch.manual_seed(240924)
    cfg_m = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    b_t0, b_a2 = load_whisper(cfg_m), load_whisper(cfg_m)                 # two independent resident instances
    for b in (b_t0, b_a2):
        if b.device != "cuda" or b.dtype != torch.bfloat16:
            raise ValueError("requires CUDA bf16")
        b.model.eval()
        b.model.requires_grad_(False)
    names = decoder_ln_names(b_a2.model)
    if names != m["trainables"] or len(names) != 194 or decoder_ln_names(b_t0.model) != names:
        raise TTAInvalid("trainable enumeration")
    g_t0, g_a2 = LNGuard(b_t0.model, names), LNGuard(b_a2.model, names)
    p_t0, p_a2 = dict(b_t0.model.named_parameters()), dict(b_a2.model.named_parameters())
    if any(p_t0[n].data_ptr() == p_a2[n].data_ptr() for n in p_t0) or g_t0.theta0_hash != g_a2.theta0_hash:
        raise TTAInvalid("instances alias storage or differ at theta0")
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in b_a2.model.named_parameters() if n not in names_set])
    if nonln0 != tensor_bytes_hash([p for n, p in b_t0.model.named_parameters() if n not in names_set]):
        raise TTAInvalid("non-LN weights differ between instances")
    tok = b_a2.processor.tokenizer
    eos = tok.eos_token_id
    gen = b_a2.model.generation_config
    suppress, begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": suppress, "begin": begin} != plan["suppression"] or eos != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    if partition["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
    specials = special_ids(tok)
    text = lambda t: tok.decode(t, skip_special_tokens=True)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    counters = {"A2": {}, "theta0_decodes": 0, "encoder_passes": 0, "logical_guard_events": 0, "detector_runs": 0, "detector_replays": 0,
                "triggers": 0, "content_events": 0, "eos_abstentions": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0, "boundary_scores": 0,
                "G2_decodes": 0, "A4_calls": 0, "audit": {"forwards": 0, "backwards": 0}, "rows_entered_after_barrier": 0}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": g_t0.theta0_hash,
               "nonln_hash_start": nonln0, "fixed_phase1_sec": 0.0, "fixed_phase2_sec": 0.0, "new_sec": 0.0, "barrier": None}
    atomic_json(out / "runtime.json", runtime)
    invalid = []

    def write(row):
        atomic_json(out / "rows" / f"{row['full_index']:03d}.json", row)

    def encode(s):
        if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
            raise TTAInvalid("audio bytes changed")
        inputs = batch_model_inputs(b_t0, [s["audio_path"]])
        with torch.inference_mode():
            h_inf = b_t0.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
        counters["encoder_passes"] += 1
        h_train = h_inf.clone()
        if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
            raise TTAInvalid("encoder clone")
        if not (g_t0.verify() and g_a2.verify()):
            raise TTAInvalid("theta0 not resident")
        return BaseModelOutput(last_hidden_state=h_inf), BaseModelOutput(last_hidden_state=h_train)

    def a2_episode(s, enc_inf, enc_train, row, live=False):
        """theta0 decode (B0) + unchanged TTA1 A2 episode; returns (eff bf16 cpu, live-check record)."""
        d0 = forced_decode(b_t0, enc_inf, CB, max_new_tokens=MAX_NEW)
        counters["theta0_decodes"] += 1
        row["theta0"] = {"tokens": d0["tokens"], "terminated": d0["terminated"], "text": d0["text"]}
        y_B = d0["tokens"]
        la = None
        if live:
            la = live_audit.live_objective_check(b_a2.model, g_a2.names, enc_train, CB, s["y_A"], "A2", suppress=suppress, begin=begin, tokenizer=tok)
            counters["audit"]["forwards"] += 1
            counters["audit"]["backwards"] += 1
            if not g_a2.verify():
                raise TTAInvalid("live audit changed resident parameters")
        assert_no_site_hooks(b_a2)
        r = t0run.run_objective(b_a2, g_a2, "A2", enc_train, enc_inf, s["y_A"], s["y_A_valid_mask"], y_B, valid_mask(y_B, suppress, begin, specials, eos),
                                suppress=suppress, begin=begin, eos=eos, partition=partition, counters=counters["A2"], keep_grad0=live)
        assert_no_site_hooks(b_a2)
        lg = r["log"]
        row["A2_log"] = {k: lg.get(k) for k in ("losses", "grad_l2", "finite", "valid", "content", "no_valid_content", "steps", "loss_evaluations",
                                                "master_delta_l2", "master_delta_rel", "effective_delta_l2", "effective_changed_scalars",
                                                "start_hash", "end_hash", "reset_ok", "other_versions_unchanged")}
        dec = r["decode"]
        row["A2_free"] = {"tokens": dec["tokens"], "terminated": dec["terminated"], "text": dec["text"]}
        fm = r["final_masters"]
        eff = {n: fm[n].to(torch.bfloat16) for n in names}
        if live:
            g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
            la["primary_loss"] = lg["losses"][0]
            la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
            la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
            la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
            la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
            row["live_audit"] = la
            if not la["pass"]:
                invalid.append(f"{row['identity']}: live audit")
        return fm, eff

    def controller(row, enc_inf, eff, uid):
        try:
            g_a2.materialize({n: v.to(g_a2.theta0[n].device) for n, v in eff.items()})
            if g_a2.current_hash() != row["A2_state_hash"] or g_t0.current_hash() != g_t0.theta0_hash:
                raise TTAInvalid("instance states")
            rec = online_g1a(b_t0, b_a2, g_t0, g_a2, enc_inf, uid, counters, eos, row["A2_free"])
        finally:
            g_a2.restore()
        invalid.extend(f"{uid}: {f}" for f in rec.pop("fails"))
        row.update(rec)
        th = row["theta0"]
        row["trigger_relation_ok"] = row["detector"]["trigger"] == (stream(row["A2_free"]["tokens"], row["A2_free"]["terminated"]) != stream(th["tokens"], th["terminated"]))
        if not row["trigger_relation_ok"]:
            invalid.append(f"{uid}: trigger != (live A2 != live theta0)")
        row["G1A_changed_vs_A2"] = (row["G1"]["tokens"], row["G1"]["terminated"]) != (row["A2_free"]["tokens"], row["A2_free"]["terminated"])
        row["reset_ok_controller"] = g_a2.verify() and g_a2.current_hash() == g_a2.theta0_hash and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash
        row["reset_hash_after"] = g_a2.current_hash()
        if not row["reset_ok_controller"]:
            invalid.append(f"{uid}: reset after controller")
        row["G1"]["text"] = text(row["G1"]["tokens"])

    def state_hash(eff):
        try:
            g_a2.materialize({n: v.to(g_a2.theta0[n].device) for n, v in eff.items()})
            return g_a2.current_hash()
        finally:
            g_a2.restore()

    try:
        torch.cuda.reset_peak_memory_stats()
        # ================= FIXED100 phase 1: exact A2 reconstruction barrier =================
        t1 = time.time()
        fixed_rows, keep = [], []
        for i, uid in enumerate(order[:nfix]):
            x = byid[uid]
            s, au = x["runtime"], x["audit"]
            row = {"identity": uid, "full_index": au["full_index"], "dialogue_id": au["dialogue_id"], "partition": au["partition"], "execution_index": i,
                   "manifest_hash": m["manifest_hash"]}
            enc_inf, enc_train = encode(s)
            fm, eff = a2_episode(s, enc_inf, enc_train, row, live=(i == 0))
            row["theta0_equals_B0"] = row["theta0"]["tokens"] == au["B0"]["tokens"] and row["theta0"]["terminated"] == au["B0"]["terminated"]
            row["A2_equals_sealed"] = (row["A2_free"]["tokens"] == au["A2"]["tokens"] and row["A2_free"]["terminated"] == au["A2"]["terminated"]
                                       and row["A2_free"]["text"] == au["A2"]["text"])
            with np.load(ROOT / au["A2_checkpoint"]) as z:
                ck = {n: z[f"A2_p{j:03d}"] for j, n in enumerate(names)}
            row["masters_equal_checkpoint_fp32"] = all(ck[n].dtype == np.float32 and ck[n].shape == tuple(fm[n].shape) and np.array_equal(fm[n].numpy(), ck[n])
                                                       for n in names)
            row["effective_equals_checkpoint_bf16"] = all(torch.equal(eff[n], torch.from_numpy(ck[n]).to(torch.bfloat16)) for n in names)
            row["A2_state_hash"] = state_hash(eff)
            row["reset_ok_episode"] = g_a2.verify() and g_a2.current_hash() == g_a2.theta0_hash and g_t0.verify() and bool(row["A2_log"]["reset_ok"])
            for k in ("theta0_equals_B0", "A2_equals_sealed", "masters_equal_checkpoint_fp32", "effective_equals_checkpoint_bf16", "reset_ok_episode"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            fixed_rows.append(row)
            keep.append((enc_inf, {n: v.cpu() for n, v in eff.items()}))
            del fm, eff
            print(f"P2-PATH5 fixed phase1 {i + 1}/{nfix} {uid} A2==sealed {row['A2_equals_sealed']} masters {row['masters_equal_checkpoint_fp32']} "
                  f"theta0==B0 {row['theta0_equals_B0']}", flush=True)
        runtime["fixed_phase1_sec"] = time.time() - t1
        if invalid:
            runtime["barrier"] = {"phase": 1, "pass": False, "failures": list(invalid)}
            raise TTAInvalid("FIXED100 A2 reconstruction barrier failed (NEW200 not entered): " + "; ".join(invalid))
        # ================= FIXED100 phase 2: live G1A == derived target; content == sealed PATH4 =================
        t2 = time.time()
        for i, uid in enumerate(order[:nfix]):
            row = fixed_rows[i]
            enc_inf, eff = keep[i]
            controller(row, enc_inf, eff, uid)
            au = byid[uid]["audit"]
            p4 = json.loads((ROOT / au["PATH4_row"]).read_text())
            row["barrier_mismatches"] = barrier_mismatches(row, p4, au["G1A_target"])
            row["barrier_exact"] = not row["barrier_mismatches"]
            if not row["barrier_exact"]:
                invalid.append(f"{uid}: fixed100 G1A barrier {row['barrier_mismatches']}")
            row["status"] = "ok"
            write(row)
            print(f"P2-PATH5 fixed phase2 {i + 1}/{nfix} {uid} mode {row['mode']} exact {row['barrier_exact']}", flush=True)
        keep.clear()
        runtime["fixed_phase2_sec"] = time.time() - t2
        runtime["barrier"] = {"rows": nfix, "exact": sum(r["barrier_exact"] for r in fixed_rows), "pass": not invalid, "failures": list(invalid)}
        atomic_json(out / "runtime.json", runtime)
        if invalid:
            raise TTAInvalid("FIXED100 G1A barrier failed (NEW200 not entered): " + "; ".join(invalid))
        # ================= NEW200 (after the barrier) =================
        t3 = time.time()
        for i, uid in enumerate(order[nfix:], start=nfix):
            x = byid[uid]
            s, au = x["runtime"], x["audit"]
            counters["rows_entered_after_barrier"] += 1
            row = {"identity": uid, "full_index": au["full_index"], "dialogue_id": au["dialogue_id"], "partition": au["partition"], "execution_index": i,
                   "manifest_hash": m["manifest_hash"]}
            enc_inf, enc_train = encode(s)
            fm, eff = a2_episode(s, enc_inf, enc_train, row)
            row["A2_state_hash"] = state_hash(eff)
            row["A2_start_hash"] = row["A2_log"]["start_hash"]
            row["reset_ok_episode"] = g_a2.verify() and g_a2.current_hash() == g_a2.theta0_hash and g_t0.verify() and bool(row["A2_log"]["reset_ok"])
            if not row["reset_ok_episode"]:
                invalid.append(f"{uid}: reset after episode")
            controller(row, enc_inf, eff, uid)
            del fm, eff, enc_inf, enc_train
            row["status"] = "ok"
            write(row)
            dd = row["decision"]
            print(f"P2-PATH5 NEW {i + 1 - nfix}/{len(order) - nfix} {uid} mode {row['mode']}"
                  + (f" k {dd['k']} winner {dd['winner']} changed {row['G1A_changed_vs_A2']}" if dd else ""), flush=True)
        runtime["new_sec"] = time.time() - t3
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        try:
            g_a2.restore()
        except Exception:
            pass
    runtime["rows_written"] = sum(1 for _ in (out / "rows").glob("*.json"))
    runtime["peak_alloc"] = int(torch.cuda.max_memory_allocated())
    runtime["peak_reserved"] = int(torch.cuda.max_memory_reserved())
    runtime["reset_final_ok"] = bool(g_a2.verify() and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash)
    runtime["nonln_hash_end"] = tensor_bytes_hash([p for n, p in b_a2.model.named_parameters() if n not in names_set])
    runtime["nonln_unchanged"] = runtime["nonln_hash_end"] == nonln0 == tensor_bytes_hash([p for n, p in b_t0.model.named_parameters() if n not in names_set])
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for b in (b_t0, b_a2) for p in b.model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("P2-PATH5 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "primary_valid": prim["valid"], "opportunity_gate": prim.get("opportunity_gate"),
           "rows_sealed": sum(1 for p in files if p.parent.name == "rows"), "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"]), "primary_valid": doc["primary_valid"],
                      "opportunity_gate": doc["opportunity_gate"]}))


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
