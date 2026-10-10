#!/usr/bin/env python
"""P2-PATH4-R1 runner (frozen G1 transfer to A2 on fixed100, EOS_BOUNDARY_V1). Contract: docs/inference_cf/P2_PATH4_SPEC.md,
P2_PATH4_CODEX_DESIGN.md, P2_PATH4_EOS_AMENDMENT.md, P2_PATH4_PANEL.json, configs/inference_cf/p2_path4.json
(revision PATH4_R1_EOS_BOUNDARY_V1, freeze 490ec20; original BLOCKED_PRE_RUN_EOS_FIRST provenance 29661e9 preserved).

prepare   (CPU) verify config/panel/amendment/source hashes, fixed100 identity/order (parent bytes + role manifest ID/
          dialogue/role columns only), the seven token/ID-only partitions and hashes, audio/teacher/comparator/checkpoint
          fingerprints, audit-only first divergences and the four EOS/content rows; write plan_sealed.json with runtime
          inputs separated from audit/analysis fields. No reference.
manifest  (CPU) resolved manifest after committed runnable PASS_TO_P2_PATH4_R1.
run       (GPU) ONE job, two resident instances (theta0 immutable; A2 instance).
          Phase 1, all 100 rows (barrier): theta0 decode = B0; unchanged TTA1 A2 episode (`inference_cf_p2tta0.run_objective`,
          first-row independent live check); final fp32 masters == historical archive arrays, effective bf16 ==, free A2 ==
          sealed A2; exact reset. Phase 2 only after all 100 pass: materialize the saved A2 state; ONE logical detector
          (`consensus_guard.lockstep_detect`); dispatch by live action types: CONTENT_G1 -> original PATH3 `online_g1`
          (A2 in its legacy adapted A4 slot), EOS_BOUNDARY -> `eos_boundary.score_boundary` + `execute_boundary` (H=1);
          never guard again. Engineering-only direct-original comparator from the same saved A2 state.
seal      (CPU) immutable output seal (before references).
The runtime controller (``online_g1_r1``) receives only the encoder output, the two model states, the emitted history,
the generation config and model-owned caches. No reference, partition, historical site/winner/k, donor or AUTO output.
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

SCHEMA = "p2_path4_r1_v1"
REVISION = "PATH4_R1_EOS_BOUNDARY_V1"
CONFIG = "configs/inference_cf/p2_path4.json"
PANEL = "docs/inference_cf/P2_PATH4_PANEL.json"
SPEC = "docs/inference_cf/P2_PATH4_SPEC.md"
DESIGN = "docs/inference_cf/P2_PATH4_CODEX_DESIGN.md"
AMENDMENT = "docs/inference_cf/P2_PATH4_EOS_AMENDMENT.md"
FREEZE_AUDIT = "docs/inference_cf/P2_PATH4_R1_PRE_RUN_AUDIT.json"
BLOCKED_ARCHIVE = "configs/inference_cf/p2_path4_blocked_v0.json"
PARENT = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
A3_PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
P2_PANEL = "docs/inference_cf/P2_PATH2_PANEL.json"
TTA1_PLAN = "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json"
ROLE = "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet"
BASE = "results/inference_cf/p2path4"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02
PARTS = ("FULL100", "DEV24", "PATH2_12", "NOVEL76", "A2_DELTA", "A2_SAME", "A2_DELTA_OUTSIDE_DEV24")
RUNTIME_KEYS = ("utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "y_B",
                "y_B_valid_mask")


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


def partitions(full: list, dev24: list, path2: list, delta: set) -> dict:
    """Pure ID/token-only partitions (no outcome). NOVEL76 = FULL100 - DEV24; delta partitions filter canonical order."""
    if len(set(full)) != len(full) or len(set(dev24)) != len(dev24) or len(set(path2)) != len(path2):
        raise ValueError("duplicate IDs")
    if not set(path2) < set(dev24) < set(full) or not delta <= set(full):
        raise ValueError("partitions not strictly nested")
    d = set(dev24)
    return {"FULL100": list(full), "DEV24": list(dev24), "PATH2_12": list(path2), "NOVEL76": [u for u in full if u not in d],
            "A2_DELTA": [u for u in full if u in delta], "A2_SAME": [u for u in full if u not in delta],
            "A2_DELTA_OUTSIDE_DEV24": [u for u in full if u in delta and u not in d]}


def first_divergence(b_tok, b_term, a_tok, a_term):
    """Audit-only first divergence of sealed B0/A2 decision streams (logical EOS appended)."""
    bs, as_ = stream(b_tok, b_term), stream(a_tok, a_term)
    if bs == as_:
        return None
    k = next((i for i, (x, y) in enumerate(zip(bs, as_)) if x != y), min(len(bs), len(as_)))
    t0, t2 = bs[k], as_[k]
    return {"k": k, "common_prefix_sha256": row_fp(b_tok[:k]), "theta0_token": t0, "A2_token": t2,
            "theta0_content_branch": [] if t0 == EOS else [t for t in bs[k:k + 3] if t != EOS],
            "A2_content_branch": [] if t2 == EOS else [t for t in as_[k:k + 3] if t != EOS],
            "EOS_first_blocker": (t0 == EOS) != (t2 == EOS)}


def selected(path: str, selector: str) -> dict:
    a, b = selector.split(".")
    return json.loads((ROOT / path).read_text())[a][b]


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2tta0 as t0run
    import pyarrow.parquet as pq
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2tta_a3 import resolve_a2
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    fa = json.loads((ROOT / FREEZE_AUDIT).read_text())
    checks = {"panel_bytes": sha_file(ROOT / PANEL) == cfg["panel_byte_sha256"],
              "frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN, AMENDMENT, FREEZE_AUDIT, BLOCKED_ARCHIVE)),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "amendment_sources": all(sha_file(ROOT / p) == h for p, h in cfg["amendment_source_sha256"].items()),
              "revision": cfg["contract_revision"] == REVISION and cfg["audit"]["pre"] == "PASS_TO_P2_PATH4_R1",
              "freeze_audit": fa["verdict"] == "PASS_TO_P2_PATH4_R1" and fa["config_sha256"] == sha_file(ROOT / CONFIG)
              and fa["contract_revision"] == REVISION and all(fa["checks"].values()),
              "blocked_provenance": sha_file(ROOT / BLOCKED_ARCHIVE) == cfg["original_blocked_provenance"]["config_archive_sha256"]
              and P["status"] == "BLOCKED_PRE_RUN_EOS_FIRST" and cfg["original_blocked_provenance"]["label"] == "P2_PATH4_BLOCKED_EOS_FIRST_BRANCH",
              "panel_sources": all(sha_file(ROOT / p) == h for p, h in P["source_sha256"].items())}
    parent = json.loads((ROOT / PARENT).read_text())
    checks["parent_bytes"] = sha_file(ROOT / PARENT) == P["parent_byte_sha256"] == cfg["fixed100_parent_byte_sha256"] \
        and parent["panel_sha256"] == P["parent_internal_panel_sha256"]
    full = [r["utterance_id"] for r in parent["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in parent["rows"]}
    checks["fixed100_identity_order"] = (len(full) == 100 and len(set(full)) == 100 and [r["utterance_id"] for r in P["rows"]] == full == P["canonical_order"]
                                         and [r["canonical_index"] for r in P["rows"]] == list(range(100))
                                         and all(row_fp(pr) == r["parent_row_sha256"] and r["dialogue_id"] == pr["dialogue_id"]
                                                 for pr, r in zip(parent["rows"], P["rows"])))
    role = pq.read_table(ROLE, columns=["utterance_id", "dialogue_id", "role"]).to_pylist()     # ID/dialogue/role columns ONLY
    rmap = {}
    for x in role:
        if x["utterance_id"] in dlg:
            rmap.setdefault(x["utterance_id"], []).append((x["dialogue_id"], x["role"]))
    checks["role_manifest_ids_dialogues"] = (set(rmap) == set(full) and all(len(v) == 1 and v[0] == (dlg[u], "D-dev-select") for u, v in rmap.items())
                                             and len(set(dlg.values())) == 20)
    t1 = json.loads((ROOT / TTA1_PLAN).read_text())
    checks["tta1_plan"] = t1["ids"] == full and t1["suppression"] == cfg["suppression"]
    rows, bad, B, A = [], [], {}, {}
    for r, t in zip(P["rows"], t1["rows"]):
        u = r["utterance_id"]
        b0 = selected(r["B0_FORCED"]["path"], r["B0_FORCED"]["selector"])
        au = selected(r["B0_AUTO"]["path"], r["B0_AUTO"]["selector"])
        a2 = selected(r["A2"]["path"], r["A2"]["selector"])
        B[u], A[u] = b0, a2
        fine = t["utterance_id"] == u and row_fp(t) == r["TTA1_plan_row_sha256"] and t["dialogue_id"] == dlg[u]
        fine &= row_fp(t["y_A"]) == r["teacher"]["y_A_token_sha256"] and row_fp(t["y_A_valid_mask"]) == r["teacher"]["valid_mask_sha256"]
        ad = r["audio"]
        fine &= (t["audio_path"] == ad["path"] and t["audio_full_sha256"] == ad["full_sha256"] and t["audio_fingerprint_64k_sizeprefixed"] == ad["fingerprint_64k_sizeprefixed"]
                 and t0run.audio_full_sha256(ad["path"]) == ad["full_sha256"] and t0run.audio_fingerprint_64k(ad["path"]) == ad["fingerprint_64k_sizeprefixed"])
        fine &= all(sha_file(ROOT / r[k]["path"]) == r[k]["byte_sha256"] for k in ("B0_FORCED", "B0_AUTO", "A2"))
        fine &= sha_file(ROOT / r["A2_checkpoint"]["path"]) == r["A2_checkpoint"]["byte_sha256"]
        fine &= (t["forced_row"] == r["B0_FORCED"]["path"] and t["auto_row"] == r["B0_AUTO"]["path"] and b0["tokens"] == t["y_B"] and au["tokens"] == t["y_A"]
                 and b0["terminated"] == t["y_B_terminated"] and au["terminated"] == t["y_A_terminated"] and b0["text"] == t["y_B_text"] and au["text"] == t["y_A_text"])
        ra2 = resolve_a2(u, t1)
        fine &= ra2["row"] == r["A2"]["path"] and ra2["origin"] == r["A2"]["origin"] == t["origin"] and ra2["tokens"] == a2["tokens"] and ra2["text"] == a2["text"]
        fine &= (row_fp(b0["tokens"]) == r["B0_sealed"]["tokens_sha256"] and len(b0["tokens"]) == r["B0_sealed"]["length"] and b0["terminated"] == r["B0_sealed"]["terminated"]
                 and row_fp(a2["tokens"]) == r["A2_sealed"]["tokens_sha256"] and len(a2["tokens"]) == r["A2_sealed"]["length"]
                 and a2["terminated"] == r["A2_sealed"]["terminated"])
        fine &= not any(x >= EOS for x in t["y_B"] + t["y_A"] + a2["tokens"])
        fd = first_divergence(b0["tokens"], b0["terminated"], a2["tokens"], a2["terminated"])
        fine &= fd == r.get("audit_only_first_divergence") and r["A2_DELTA"] == (fd is not None)
        if fd is not None:
            fine &= stream(b0["tokens"], b0["terminated"])[:fd["k"]] == stream(a2["tokens"], a2["terminated"])[:fd["k"]]
        if not fine:
            bad.append(u)
        rows.append({"runtime": {"utterance_id": u, "audio_path": ad["path"], "audio_fingerprint_64k_sizeprefixed": ad["fingerprint_64k_sizeprefixed"],
                                 "audio_full_sha256": ad["full_sha256"], "y_A": t["y_A"], "y_A_valid_mask": t["y_A_valid_mask"], "y_B": t["y_B"],
                                 "y_B_valid_mask": t["y_B_valid_mask"]},
                     "audit": {"canonical_index": r["canonical_index"], "dialogue_id": dlg[u], "partitions": r["partitions"],
                               "B0": {"tokens": b0["tokens"], "terminated": b0["terminated"]},
                               "A2": {"tokens": a2["tokens"], "terminated": a2["terminated"], "text": a2["text"]},
                               "A2_checkpoint": r["A2_checkpoint"]["path"], "first_divergence": fd},
                     "analysis": {"utterance_id": u, "dialogue_id": dlg[u], "y_B_text": t["y_B_text"],
                                  "AUTO": {"tokens": au["tokens"], "terminated": au["terminated"], "text": au["text"]},
                                  "A2": {"tokens": a2["tokens"], "terminated": a2["terminated"], "text": a2["text"], "origin": ra2["origin"], "row": ra2["row"]}}})
    checks["rows"] = not bad
    a3p, p2p = json.loads((ROOT / A3_PANEL).read_text()), json.loads((ROOT / P2_PANEL).read_text())
    dev24, path2 = [r["utterance_id"] for r in a3p["rows"]], [r["utterance_id"] for r in p2p["rows"]]
    delta = {u for u in full if stream(B[u]["tokens"], B[u]["terminated"]) != stream(A[u]["tokens"], A[u]["terminated"])}
    parts = partitions(full, dev24, path2, delta)
    checks["partitions"] = (all(parts[k] == P["partitions"][k]["ids"] for k in PARTS)
                            and {k: len(v) for k, v in parts.items()} == cfg["counts"]
                            and all(membership_hash(k, parts[k], dlg) == cfg["partition_sha256"][k] for k in PARTS)
                            and all(sorted(r["partitions"]) == sorted(k for k in PARTS if r["utterance_id"] in parts[k]) for r in P["rows"]))
    eos_rows = [x["runtime"]["utterance_id"] for x in rows if x["audit"]["first_divergence"] and x["audit"]["first_divergence"]["EOS_first_blocker"]]
    checks["expected_triggers_and_boundary_rows"] = (parts["A2_DELTA"] == cfg["G1"]["expected_trigger_ids"] and len(parts["A2_DELTA"]) == 14
                                                     and eos_rows == [b["utterance_id"] for b in P["known_blockers"]]
                                                     == [b["utterance_id"] for b in cfg["known_pre_run_block"]["rows"]] and len(eos_rows) == 4)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    checks["suppression"] = {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])} == cfg["suppression"]
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    part = tokenizer_partition(tok)
    plan = {"schema": SCHEMA + "_plan", "contract_revision": REVISION, "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG),
            "checks": checks, "failures": bad, "ids": full, "partitions": parts, "suppression": cfg["suppression"], "eos": EOS,
            "partition_hash": part["hash"], "boundary_rows_audit_only": eos_rows, "rows": rows, "outcomes_computed": False,
            "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]) + " rows " + json.dumps(bad))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = (SPEC, DESIGN, AMENDMENT, PANEL, CONFIG, FREEZE_AUDIT, BLOCKED_ARCHIVE, PLAN, PARENT, A3_PANEL, P2_PANEL, TTA1_PLAN,
           "src/csasr/inference_cf/eos_boundary.py", "src/csasr/inference_cf/consensus_guard.py", "src/csasr/inference_cf/branch_adjudication.py",
           "src/csasr/inference_cf/path_decode.py", "src/csasr/inference_cf/episodic_tta.py", "experiments/inference_cf_p2path3.py",
           "experiments/inference_cf_p2path4.py", "experiments/inference_cf_p2path4_analyze.py", "experiments/inference_cf_p2path4_audit.py",
           "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_p2tta0_audit.py", "slurm/inference_cf_p2path4.sbatch",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path4.py", "tests/test_inference_cf_p2path4_r1.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH4 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH4_R1":
        raise ValueError("runnable pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "contract_revision": REVISION, "git_commit": git("rev-parse", "HEAD"),
           "git_tree": git("rev-parse", "HEAD^{tree}"), "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL),
           "plan_hash": plan["plan_hash"], "ids": plan["ids"], "trainables": [p["name"] for p in cfg["A2_inherited"]["trainables"]["parameters"]],
           "budget": cfg["compute"], "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def boundary_hash(d: dict) -> str:
    keys = ("contract_revision", "mode", "H", "k", "prefix", "c0", "c2", "S_theta0", "S_A2", "S_cons", "margin", "winner", "selected_token")
    return "sha256:" + hashlib.sha256(json.dumps({k: d[k] for k in keys}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def online_g1_r1(b_t0, b_a2, g_t0, g_a2, enc, owner_row: str, counters: dict, eos: int) -> dict:
    """PATH4_R1 controller. A2 must already be materialized on ``b_a2``; ``owner_row`` is a log label. ONE logical detector;
    CONTENT_G1 -> original PATH3 online_g1 (A2 in its legacy A4 slot; its replayed detection must equal this one);
    EOS_BOUNDARY -> H=1 next-action consensus + one selected action under A2. No further guard after commitment."""
    from csasr.inference_cf.consensus_guard import lockstep_detect
    from csasr.inference_cf.eos_boundary import action_mode, execute_boundary, score_boundary
    import experiments.inference_cf_p2path3 as path3
    fails = []
    h_a2, h_t0 = g_a2.current_hash(), g_t0.current_hash()
    det = lockstep_detect(b_t0, b_a2, enc, CB, owners={"theta0": h_t0, "A4": h_a2},
                          hash_fns={"theta0": g_t0.current_hash, "A4": g_a2.current_hash}, max_new=MAX_NEW)
    counters["logical_guard_events"] += 1
    counters["detector_runs"] += 1
    if not (det["state_locked"] and det["fed_identical"] and det["positions_ok"]):
        fails.append("detector integrity")
    if not det["trigger"]:
        return {"mode": "NO_TRIGGER", "detector": det, "decision": None, "G1": {"tokens": det["prefix"], "terminated": det["terminated"]},
                "fails": fails}
    counters["triggers"] += 1
    gen, tok = b_t0.model.generation_config, b_t0.processor.tokenizer
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    c0, c2 = det["argmax_theta0"], det["argmax_A4"]
    mode = action_mode(c0, c2, eos=eos, vocab_size=int(b_t0.model.config.vocab_size), special_ids=list(tok.all_special_ids), step=det["k"],
                       suppress=sup, begin=beg)
    if mode == "CONTENT_G1":
        inner = {"detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0}
        rec = path3.online_g1(b_t0, b_a2, g_t0, g_a2, enc, owner_row, inner, eos)
        fails.extend(rec.pop("fails"))
        if json.dumps(rec["detector"], sort_keys=True) != json.dumps(det, sort_keys=True):
            fails.append("replayed detection differs from the logical detection")
        if inner["triggers"] != 1 or inner["detector_runs"] != 1:
            fails.append("content dispatch event count")
        counters["detector_replays"] += inner["detector_runs"]
        for k in ("rollouts", "score_paths", "G1_decodes"):
            counters[k] += inner[k]
        return {"mode": "CONTENT_G1", **rec, "fails": fails}
    if mode != "EOS_BOUNDARY":
        raise ValueError(f"unexpected dispatch mode {mode}")
    dec = score_boundary(b_t0, b_a2, enc, CB, det["prefix"], c0, c2, owners={"theta0": h_t0, "A2": h_a2},
                         hash_fns={"theta0": g_t0.current_hash, "A2": g_a2.current_hash})
    counters["boundary_score_paths"] += 2
    if dec["k"] != det["k"] or dec["prefix"] != det["prefix"] or (dec["c0"], dec["c2"]) != (c0, c2) or dec["H"] != 1:
        fails.append("boundary decision does not match the logical detection")
    dec["decision_hash"] = boundary_hash(dec)
    d = execute_boundary(b_a2, enc, CB, dec, owner_hash=h_a2, hash_fn=g_a2.current_hash, max_new_tokens=MAX_NEW)
    counters["boundary_executions"] += 1
    tr = d["trace"]
    G = {"tokens": d["tokens"], "terminated": d["terminated"], "trace": tr, "state_locked": d["state_locked"], "forced": [dec["selected_token"]],
         "decision_hash": dec["decision_hash"]}
    if not (d["state_locked"] and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"]
            and [f["token"] for f in tr["forced"]] == [dec["selected_token"]] and d["tokens"][:dec["k"]] == dec["prefix"]):
        fails.append("boundary execution integrity")
    if dec["selected_token"] == eos and not (d["tokens"] == dec["prefix"] and d["terminated"] == "eos"):
        fails.append("EOS winner did not terminate at the common prefix")
    return {"mode": "EOS_BOUNDARY", "detector": det, "decision": dec, "G1": G, "fails": fails}


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True))


def comparator_mismatch(rec: dict, orig: dict) -> list:
    """Engineering-only: dispatched record vs direct original online_g1 from the same saved A2 state (exact)."""
    bad = [k for k in ("detector", "decision", "G1", "score_paths") if _norm(rec.get(k)) != _norm(orig.get(k))]
    return bad


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, forced_decode, tensor_bytes_hash
    from csasr.lss.sites import assert_no_site_hooks
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2path3 as path3
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
    text = lambda t: tok.decode(t, skip_special_tokens=True)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    counters = {"A2": {}, "theta0_decodes": 0, "encoder_passes": 0, "logical_guard_events": 0, "detector_runs": 0, "detector_replays": 0,
                "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0, "boundary_score_paths": 0, "boundary_executions": 0,
                "G2_decodes": 0, "A4_calls": 0, "audit": {"forwards": 0, "backwards": 0},
                "comparator": {"detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0, "refusals": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "contract_revision": REVISION, "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running",
               "theta0_ln_hash": g_t0.theta0_hash, "nonln_hash_start": nonln0, "phase1_sec": 0.0, "phase2_sec": 0.0, "barrier": None}
    atomic_json(out / "runtime.json", runtime)
    invalid, rows_out, keep = [], [], []
    try:
        torch.cuda.reset_peak_memory_stats()
        # ================= Phase 1: all-100 exact A2 reconstruction barrier =================
        t1 = time.time()
        for i, x in enumerate(plan["rows"]):
            s, au = x["runtime"], x["audit"]
            uid = s["utterance_id"]
            row = {"identity": uid, "canonical_index": au["canonical_index"], "dialogue_id": au["dialogue_id"], "manifest_hash": m["manifest_hash"]}
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
            if not (g_t0.verify() and g_a2.verify()):
                raise TTAInvalid("theta0 not resident")
            d0 = forced_decode(b_t0, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_decodes"] += 1
            row["theta0"] = {"tokens": d0["tokens"], "terminated": d0["terminated"]}
            row["theta0_equals_B0"] = d0["tokens"] == au["B0"]["tokens"] and d0["terminated"] == au["B0"]["terminated"]
            la = None
            if i == 0:
                la = live_audit.live_objective_check(b_a2.model, g_a2.names, enc_train, CB, s["y_A"], "A2", suppress=suppress, begin=begin, tokenizer=tok)
                counters["audit"]["forwards"] += 1
                counters["audit"]["backwards"] += 1
                if not g_a2.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            assert_no_site_hooks(b_a2)
            r = t0run.run_objective(b_a2, g_a2, "A2", enc_train, enc_inf, s["y_A"], s["y_A_valid_mask"], s["y_B"], s["y_B_valid_mask"],
                                    suppress=suppress, begin=begin, eos=eos, partition=partition, counters=counters["A2"], keep_grad0=(i == 0))
            assert_no_site_hooks(b_a2)
            lg = r["log"]
            row["A2_log"] = {k: lg.get(k) for k in ("losses", "grad_l2", "finite", "valid", "content", "no_valid_content", "steps", "loss_evaluations",
                                                    "master_delta_l2", "master_delta_rel", "effective_delta_l2", "effective_changed_scalars",
                                                    "start_hash", "end_hash", "reset_ok", "other_versions_unchanged")}
            dec = r["decode"]
            row["A2_free"] = {"tokens": dec["tokens"], "terminated": dec["terminated"], "text": dec["text"]}
            row["A2_equals_sealed"] = dec["tokens"] == au["A2"]["tokens"] and dec["terminated"] == au["A2"]["terminated"] and dec["text"] == au["A2"]["text"]
            fm = r["final_masters"]
            with np.load(ROOT / au["A2_checkpoint"]) as z:
                ck = {n: z[f"A2_p{j:03d}"] for j, n in enumerate(names)}
            row["masters_equal_checkpoint_fp32"] = all(ck[n].dtype == np.float32 and ck[n].shape == tuple(fm[n].shape)
                                                       and np.array_equal(fm[n].numpy(), ck[n]) for n in names)
            eff = {n: fm[n].to(torch.bfloat16) for n in names}
            row["effective_equals_checkpoint_bf16"] = all(torch.equal(eff[n], torch.from_numpy(ck[n]).to(torch.bfloat16)) for n in names)
            try:
                g_a2.materialize({n: v.to(g_a2.theta0[n].device) for n, v in eff.items()})
                row["A2_state_hash"] = g_a2.current_hash()
            finally:
                g_a2.restore()
            row["reset_ok_phase1"] = g_a2.verify() and g_a2.current_hash() == g_a2.theta0_hash and g_t0.verify() and bool(lg.get("reset_ok"))
            if la is not None:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = lg["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = la
                if not la["pass"]:
                    invalid.append(f"{uid}: live audit")
            for k in ("theta0_equals_B0", "A2_equals_sealed", "masters_equal_checkpoint_fp32", "effective_equals_checkpoint_bf16", "reset_ok_phase1"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            rows_out.append(row)
            keep.append((enc_inf, {n: v.cpu() for n, v in eff.items()}))
            del r, fm, eff
            print(f"P2-PATH4 phase1 {i + 1}/100 {uid} A2==sealed {row['A2_equals_sealed']} masters {row['masters_equal_checkpoint_fp32']} "
                  f"theta0==B0 {row['theta0_equals_B0']}", flush=True)
        runtime["phase1_sec"] = time.time() - t1
        runtime["barrier"] = {"rows": 100, "pass": not invalid, "failures": list(invalid)}
        atomic_json(out / "runtime.json", runtime)
        if invalid:
            raise TTAInvalid("A2 reconstruction barrier failed (G1 phase not entered): " + "; ".join(invalid))
        # ================= Phase 2: R1 controller on each saved A2 state =================
        t2 = time.time()
        for i, x in enumerate(plan["rows"]):
            enc_inf, eff = keep[i]
            row = rows_out[i]
            uid = x["runtime"]["utterance_id"]
            try:
                g_a2.materialize({n: v.to(g_a2.theta0[n].device) for n, v in eff.items()})
                if g_a2.current_hash() != row["A2_state_hash"] or g_t0.current_hash() != g_t0.theta0_hash:
                    raise TTAInvalid("instance states")
                rec = online_g1_r1(b_t0, b_a2, g_t0, g_a2, enc_inf, uid, counters, eos)
                # engineering-only direct-original comparator (same saved A2 state; never changes the method)
                cc = counters["comparator"]
                if rec["mode"] == "EOS_BOUNDARY":
                    try:
                        path3.online_g1(b_t0, b_a2, g_t0, g_a2, enc_inf, uid, cc, eos)
                        row["comparator"] = {"original_refused_EOS_first": False}
                    except TTAInvalid as e:
                        cc["refusals"] += 1
                        row["comparator"] = {"original_refused_EOS_first": "EOS-first" in str(e)}
                else:
                    orig = path3.online_g1(b_t0, b_a2, g_t0, g_a2, enc_inf, uid, cc, eos)
                    row["comparator"] = {"mismatches": comparator_mismatch(rec, orig), "original_fails": orig["fails"]}
            finally:
                g_a2.restore()
            invalid.extend(f"{uid}: {f}" for f in rec.pop("fails"))
            row.update(rec)
            G0 = {"tokens": row["A2_free"]["tokens"], "terminated": row["A2_free"]["terminated"]}
            Gt = {"tokens": row["G1"]["tokens"], "terminated": row["G1"]["terminated"]}
            cmp_ok = row["comparator"].get("original_refused_EOS_first", False) if rec["mode"] == "EOS_BOUNDARY" else \
                (not row["comparator"]["mismatches"] and not row["comparator"]["original_fails"])
            row["comparator"]["ok"] = bool(cmp_ok)
            if not cmp_ok:
                invalid.append(f"{uid}: direct-original comparator")
            if rec["mode"] == "NO_TRIGGER" and Gt != G0:
                invalid.append(f"{uid}: no-trigger output != live A2")
            if rec["decision"] is not None and rec["decision"]["winner"] in ("A4", "A2") and Gt != G0:
                invalid.append(f"{uid}: A2 winner but G != live A2")
            th = {"tokens": row["theta0"]["tokens"], "terminated": row["theta0"]["terminated"]}
            row["trigger_relation_ok"] = row["detector"]["trigger"] == (stream(G0["tokens"], G0["terminated"]) != stream(th["tokens"], th["terminated"]))
            if not row["trigger_relation_ok"]:
                invalid.append(f"{uid}: trigger != (live A2 != live theta0)")
            row["reset_ok_phase2"] = g_a2.verify() and g_a2.current_hash() == g_a2.theta0_hash and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash
            row["reset_hash_after"] = g_a2.current_hash()
            if not row["reset_ok_phase2"]:
                invalid.append(f"{uid}: reset phase 2")
            row["G1"]["text"] = text(row["G1"]["tokens"])
            row["status"] = "ok"
            atomic_json(out / "rows" / f"{row['canonical_index']:03d}.json", row)
            dd = row["decision"]
            print(f"P2-PATH4 phase2 {i + 1}/100 {uid} mode {row['mode']}"
                  + (f" k {dd['k']} winner {dd['winner']} margin {dd['margin']:.4f}" if dd else ""), flush=True)
        runtime["phase2_sec"] = time.time() - t2
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        for row in rows_out:
            p = out / "rows" / f"{row['canonical_index']:03d}.json"
            if not p.exists():
                atomic_json(p, {**row, "status": "partial"})
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
        raise SystemExit("P2-PATH4 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "contract_revision": REVISION, "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"],
           "primary_valid": prim["valid"], "rows_sealed": sum(1 for p in files if p.parent.name == "rows"), "references_used": False,
           "created_unix": time.time()}
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
