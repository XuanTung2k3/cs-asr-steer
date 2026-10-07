#!/usr/bin/env python
"""P2-PATH3 runner (frozen G1 fixed-100 breadth confirmation). Contract: docs/inference_cf/P2_PATH3_SPEC.md,
P2_PATH3_CODEX_DESIGN.md, P2_PATH3_PANEL.json, configs/inference_cf/p2_path3.json (freeze 5853a8d).

prepare   (CPU) verify config/panel/source hashes, fixed100 identity/order (parent bytes + role manifest ID/dialogue/role
          columns only), strict PATH2_12 < DEV24 < FULL100, NOVEL76 = FULL100 - DEV24, membership/execution-order hashes,
          audio, comparator (B0/AUTO/A2) and teacher fingerprints, DEV24 A4 teacher/mask/class/language identity, PATH2
          replay rows; write plan_sealed.json with runtime inputs separated from audit/analysis fields. No reference.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_PATH3.
run       (GPU) ONE job, two resident instances (theta0 immutable; A4 instance).
          Replay12 (PATH2 order) first, exactly as PATH2: phase 1 reconstruction (theta0 decode = B0, sealed language,
          unchanged adapt_a4, effective == checkpoint, G0 = sealed A4, exact reset; original first-row live check), then
          the unchanged G1 online controller (no G2), then an exact scientific-field comparison against the sealed PATH2
          rows. Any mismatch aborts before the remaining 88 rows touch the encoder. Remaining 88 (fixed100 order):
          fresh A4 episode, live A4 (G0) decode, the same G1 controller, exact reset.
seal      (CPU) immutable output seal (before references).
The runtime controller (``online_g1``) receives only the encoder output, the two model states, the emitted history,
the generation config and model-owned caches. No reference, partition, historical site/winner, donor or AUTO output.
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

SCHEMA = "p2_path3_v1"
CONFIG = "configs/inference_cf/p2_path3.json"
PANEL = "docs/inference_cf/P2_PATH3_PANEL.json"
SPEC = "docs/inference_cf/P2_PATH3_SPEC.md"
DESIGN = "docs/inference_cf/P2_PATH3_CODEX_DESIGN.md"
PARENT = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
A3_PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
P2_PANEL = "docs/inference_cf/P2_PATH2_PANEL.json"
TTA1_PLAN = "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json"
A4_PLAN = "results/inference_cf/p2tta_a4/plan_sealed.json"
P2_PLAN = "results/inference_cf/p2path2/plan_sealed.json"
ROLE = "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet"
BASE = "results/inference_cf/p2path3"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02
RUNTIME_KEYS = ("utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "classes")
REPLAY_FIELDS = ("A4_state_hash", "theta0_equals_B0", "lang_equals_sealed", "effective_equals_checkpoint_bf16", "reset_ok_phase1", "G0",
                 "G0_equals_sealed_A4", "detector", "decision", "reset_ok_phase2")


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


def partitions(full: list, dev24: list, path2: list) -> dict:
    """Pure partition construction from ID lists (no outcome): NOVEL76 = FULL100 - DEV24 in parent order."""
    if len(set(full)) != len(full) or len(set(dev24)) != len(dev24) or len(set(path2)) != len(path2):
        raise ValueError("duplicate IDs")
    if not set(path2) < set(dev24) < set(full):
        raise ValueError("partitions not strictly nested")
    d = set(dev24)
    return {"FULL100": list(full), "DEV24": list(dev24), "PATH2_12": list(path2), "NOVEL76": [u for u in full if u not in d]}


def execution_order(full: list, path2: list) -> list:
    """PATH2's original 12 first, then the remaining 88 in fixed100 order."""
    p = set(path2)
    return list(path2) + [u for u in full if u not in p]


def selected(path: str, selector: str) -> dict:
    a, b = selector.split(".")
    return json.loads((ROOT / path).read_text())[a][b]


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    import experiments.inference_cf_p2tta0 as t0run
    import pyarrow.parquet as pq
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.script_safe_tta import token_classes
    from experiments.inference_cf_p2tta_a3 import resolve_a2
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    checks = {"panel_bytes": sha_file(ROOT / PANEL) == cfg["panel_byte_sha256"],
              "frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN)),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "panel_parents": all(sha_file(ROOT / p) == h for p, h in P["parents"].items())}
    parent = json.loads((ROOT / PARENT).read_text())
    checks["parent_bytes"] = sha_file(ROOT / PARENT) == P["parent_byte_sha256"] == cfg["fixed100_parent_byte_sha256"] \
        and parent["panel_sha256"] == P["parent_internal_panel_sha256"]
    full = [r["utterance_id"] for r in parent["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in parent["rows"]}
    checks["fixed100_identity_order"] = (len(full) == 100 and len(set(full)) == 100 and [r["utterance_id"] for r in P["rows"]] == full
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
    a3p, p2p = json.loads((ROOT / A3_PANEL).read_text()), json.loads((ROOT / P2_PANEL).read_text())
    dev24, path2 = [r["utterance_id"] for r in a3p["rows"]], [r["utterance_id"] for r in p2p["rows"]]
    parts = partitions(full, dev24, path2)
    checks["partitions"] = (all(parts[k] == P["partitions"][k]["ids"] for k in parts)
                            and {k: len(v) for k, v in parts.items()} == cfg["counts"] == {"FULL100": 100, "DEV24": 24, "PATH2_12": 12, "NOVEL76": 76}
                            and {k: len({dlg[u] for u in v}) for k, v in parts.items()} == {"FULL100": 20, "DEV24": 20, "PATH2_12": 9, "NOVEL76": 20}
                            and all(membership_hash(k, v, dlg) == cfg["partition_sha256"][k] == P["partitions"][k]["membership_order_sha256"]
                                    for k, v in parts.items())
                            and all(dlg[r["utterance_id"]] == r["dialogue_id"] for r in a3p["rows"] + p2p["rows"]))
    tag = {u: ("PATH2_12" if u in path2 else ("DEV24_OTHER" if u in dev24 else "NOVEL76")) for u in full}
    checks["row_partition_tags"] = all(r["partition"] == tag[r["utterance_id"]] for r in P["rows"])
    order = execution_order(full, path2)
    checks["execution_order"] = order == P["execution_order"] and row_fp(order) == P["execution_order_sha256"] and order[:12] == cfg["PATH2_barrier"]["first_ids"]
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    checks["suppression"] = {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])} == cfg["suppression"]
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    part = tokenizer_partition(tok)
    checks["tokenizer_partition"] = part["hash"] == cfg["A4_inherited"]["partition"]["hash"]
    t1 = json.loads((ROOT / TTA1_PLAN).read_text())
    a4 = json.loads((ROOT / A4_PLAN).read_text())
    p2plan = json.loads((ROOT / P2_PLAN).read_text())
    checks["parent_plans"] = (t1["ids"] == full and a4["ids"] == dev24 and p2plan["ids"] == path2 and a4["partition_hash"] == part["hash"]
                              and t1["suppression"] == a4["suppression"] == cfg["suppression"])
    rows, bad = [], []
    for r, t in zip(P["rows"], t1["rows"]):
        u = r["utterance_id"]
        fine = t["utterance_id"] == u and row_fp(t) == r["TTA1_plan_row_sha256"] and t["dialogue_id"] == dlg[u]
        fine &= row_fp(t["y_A"]) == r["teacher"]["y_A_token_sha256"] and row_fp(t["y_A_valid_mask"]) == r["teacher"]["valid_mask_sha256"]
        au = r["audio"]
        fine &= (t["audio_path"] == au["path"] and t["audio_full_sha256"] == au["full_sha256"] and t["audio_fingerprint_64k_sizeprefixed"] == au["fingerprint_64k_sizeprefixed"]
                 and t0run.audio_full_sha256(au["path"]) == au["full_sha256"] and t0run.audio_fingerprint_64k(au["path"]) == au["fingerprint_64k_sizeprefixed"])
        fine &= all(sha_file(ROOT / r[k]["path"]) == r[k]["byte_sha256"] for k in ("B0_FORCED", "B0_AUTO", "A2"))
        B, A, A2 = selected(r["B0_FORCED"]["path"], r["B0_FORCED"]["selector"]), selected(r["B0_AUTO"]["path"], r["B0_AUTO"]["selector"]), \
            selected(r["A2"]["path"], r["A2"]["selector"])
        fine &= (t["forced_row"] == r["B0_FORCED"]["path"] and t["auto_row"] == r["B0_AUTO"]["path"] and B["tokens"] == t["y_B"] and A["tokens"] == t["y_A"]
                 and B["terminated"] == t["y_B_terminated"] and A["terminated"] == t["y_A_terminated"] and B["text"] == t["y_B_text"] and A["text"] == t["y_A_text"])
        ra2 = resolve_a2(u, t1)
        fine &= ra2["row"] == r["A2"]["path"] and ra2["origin"] == r["A2"]["origin"] == t["origin"] and ra2["tokens"] == A2["tokens"] and ra2["text"] == A2["text"]
        fine &= not any(x >= EOS for x in t["y_B"] + t["y_A"])
        cls = token_classes(t["y_A"], part)
        audit = {"canonical_index": r["canonical_index"], "dialogue_id": dlg[u], "partition": tag[u],
                 "B0": {"tokens": t["y_B"], "terminated": t["y_B_terminated"]}}
        if tag[u] != "NOVEL76":
            x = r["A4_DEV24_audit_only"]
            p4 = a4["rows"][x["index"]]
            a4row = json.loads((ROOT / x["row_path"]).read_text())
            fine &= (p4["utterance_id"] == u and p4["y_A"] == t["y_A"] and p4["y_A_valid_mask"] == t["y_A_valid_mask"] and p4["classes"] == cls
                     and p4["y_B"] == t["y_B"] and sha_file(ROOT / x["row_path"]) == x["row_sha256"] and sha_file(ROOT / x["checkpoint"]) == x["checkpoint_sha256"]
                     and a4row["identity"] == u and a4row["auto_condition"]["lang_id"] == p4["a3_lang_id"] == x["sealed_native_language_id"]
                     and a4row["final_masters_npz_sha256"] == "sha256:" + x["checkpoint_sha256"])
            audit["A4_sealed"] = {"tokens": a4row["A4"]["tokens"], "terminated": a4row["A4"]["terminated"], "text": a4row["A4"]["text"],
                                  "checkpoint": x["checkpoint"], "lang_id": x["sealed_native_language_id"]}
        if tag[u] == "PATH2_12":
            x = r["PATH2_replay_audit_only"]
            p2row = json.loads((ROOT / x["path"]).read_text())
            fine &= sha_file(ROOT / x["path"]) == x["byte_sha256"] and p2row["identity"] == u and p2row["status"] == "ok" and path2.index(u) == x["index"]
            audit["PATH2_row"] = x["path"]
        if not fine:
            bad.append(u)
        rows.append({"runtime": {"utterance_id": u, "audio_path": au["path"], "audio_fingerprint_64k_sizeprefixed": au["fingerprint_64k_sizeprefixed"],
                                 "audio_full_sha256": au["full_sha256"], "y_A": t["y_A"], "y_A_valid_mask": t["y_A_valid_mask"], "classes": cls},
                     "audit": audit,
                     "analysis": {"utterance_id": u, "dialogue_id": dlg[u], "partition": tag[u], "y_B_text": t["y_B_text"],
                                  "AUTO": {"tokens": A["tokens"], "terminated": A["terminated"], "text": A["text"]},
                                  "A2": {"tokens": A2["tokens"], "terminated": A2["terminated"], "text": A2["text"], "origin": ra2["origin"], "row": ra2["row"]}}})
    checks["rows"] = not bad
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "failures": bad, "ids": full, "execution_order": order, "partitions": parts, "suppression": cfg["suppression"], "eos": EOS,
            "partition_hash": part["hash"], "rows": rows, "outcomes_computed": False, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]) + " rows " + json.dumps(bad))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = (SPEC, DESIGN, PANEL, CONFIG, PLAN, PARENT, A3_PANEL, P2_PANEL, TTA1_PLAN, A4_PLAN, P2_PLAN,
           "src/csasr/inference_cf/consensus_guard.py", "src/csasr/inference_cf/branch_adjudication.py", "src/csasr/inference_cf/path_decode.py",
           "src/csasr/inference_cf/script_safe_tta.py", "src/csasr/inference_cf/soft_auto_tta.py", "src/csasr/inference_cf/episodic_tta.py",
           "experiments/inference_cf_p2path3.py", "experiments/inference_cf_p2path3_analyze.py", "experiments/inference_cf_p2path3_audit.py",
           "experiments/inference_cf_p2tta_a4_audit.py", "slurm/inference_cf_p2path3.sbatch", "experiments/inference_cf_p2tta0.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "tests/test_inference_cf_p2path3.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit PATH3 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_PATH3":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "execution_order": plan["execution_order"], "trainables": [p["name"] for p in cfg["A4_inherited"]["trainables"]["parameters"]],
           "budget": cfg["compute"], "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def online_g1(b_t0, b_a4, g_t0, g_a4, enc, owner_row: str, counters: dict, eos: int) -> dict:
    """Unchanged PATH2 CONSENSUS-1 (G1 only). A4 must already be materialized on ``b_a4``; ``owner_row`` is a log label.
    Returns the controller record and a list of integrity failures; EOS-first branch raises (frozen INVALID)."""
    from csasr.inference_cf.branch_adjudication import consensus, owned_clamp, score_branch
    from csasr.inference_cf.consensus_guard import decision_hash, g1_forced, lockstep_detect, rollout
    from csasr.inference_cf.episodic_tta import TTAInvalid
    fails = []
    h_a4, h_t0 = g_a4.current_hash(), g_t0.current_hash()
    det = lockstep_detect(b_t0, b_a4, enc, CB, owners={"theta0": h_t0, "A4": h_a4},
                          hash_fns={"theta0": g_t0.current_hash, "A4": g_a4.current_hash}, max_new=MAX_NEW)
    counters["detector_runs"] += 1
    rec = {"detector": det}
    if not (det["state_locked"] and det["fed_identical"] and det["positions_ok"]):
        fails.append("detector integrity")
    if not det["trigger"]:
        rec["decision"] = None
        rec["G1"] = {"tokens": det["prefix"], "terminated": det["terminated"]}
        return {**rec, "fails": fails}
    counters["triggers"] += 1
    k, pre = det["k"], det["prefix"]
    b0 = rollout(b_t0, enc, CB, pre, owner_hash=h_t0, hash_fn=g_t0.current_hash, max_new=MAX_NEW)
    b4 = rollout(b_a4, enc, CB, pre, owner_hash=h_a4, hash_fn=g_a4.current_hash, max_new=MAX_NEW)
    counters["rollouts"] += 2
    for nm, b, am in (("b0", b0, det["argmax_theta0"]), ("b4", b4, det["argmax_A4"])):
        if not (b["prefix_ok"] and b["positions_ok"] and b["state_locked"]) or b["H_eff"] < 1 or b["tokens"][0] != am:
            fails.append(f"rollout {nm} integrity/EOS-first")
    if b0["H_eff"] < 1 or b4["H_eff"] < 1:
        raise TTAInvalid(f"{owner_row}: EOS-first branch (frozen INVALID)")
    paths = {}
    for mdl, bundle, hh, hf in (("theta0", b_t0, h_t0, g_t0.current_hash), ("A4", b_a4, h_a4, g_a4.current_hash)):
        for br_name, br in (("b0", b0), ("b4", b4)):
            p = score_branch(bundle, enc, CB, pre, br["tokens"], owner={"state": mdl, "state_hash": hh, "row": owner_row,
                                                                         "condition": f"score_{mdl}_{br_name}"}, state_hash_fn=hf)
            counters["score_paths"] += 1
            paths[f"{mdl}_{br_name}"] = p
            if not (p["state_locked"] and p["prefix_ok"] and p["positions_ok"] and p["fed_ok"]):
                fails.append(f"score {mdl}_{br_name} integrity")
    cons = consensus({"B": paths["theta0_b0"]["score"], "ALT": paths["theta0_b4"]["score"]},
                     {"B": paths["A4_b0"]["score"], "ALT": paths["A4_b4"]["score"]})
    dec = {"k": k, "prefix": pre, "b0": b0, "b4": b4, "scores": {kk: {"logprobs": v["logprobs"], "score": v["score"], "H_eff": v["H_eff"]}
                                                               for kk, v in paths.items()},
           "winner": "theta0" if cons["choice"] == "B" else "A4", "margin": cons["margin_B_minus_ALT"], "S_cons": cons["S_cons"],
           "strict": cons["strict_B_win"]}
    dec["decision_hash"] = decision_hash(dec)
    rec["decision"] = dec
    rec["score_paths"] = {kk: {kq: v[kq] for kq in ("owner", "state_hash_before", "state_hash_after", "state_locked", "prefix_ok", "positions_ok", "fed_ok")}
                          for kk, v in paths.items()}
    f1 = g1_forced(dec)
    d1 = owned_clamp(b_a4, enc, CB, site=k, forced=f1, expected_prefix=pre, owner={"state": "A4", "state_hash": h_a4, "row": owner_row, "condition": "G1"},
                     state_hash_fn=g_a4.current_hash, max_new_tokens=MAX_NEW)
    counters["G1_decodes"] += 1
    rec["G1"] = {"tokens": d1["tokens"], "terminated": d1["terminated"], "trace": d1["trace"], "state_locked": d1["state_locked"], "forced": f1,
                 "decision_hash": dec["decision_hash"]}
    tr = d1["trace"]
    if not (d1["state_locked"] and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"] and [f["token"] for f in tr["forced"]] == f1):
        fails.append("G1 execution integrity")
    return {**rec, "fails": fails}


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True))


def replay_mismatches(row: dict, p2row: dict) -> list:
    """Exact comparison of the shared scientific fields against the sealed PATH2 row (no float tolerance). Only owner
    instance/condition labels, manifest, timing, counters and the absent G2 are ignored."""
    bad = [f for f in REPLAY_FIELDS if _norm(row.get(f)) != _norm(p2row.get(f))]
    g1k = ("tokens", "terminated", "trace", "state_locked", "forced", "decision_hash")
    if _norm({k: row["G1"].get(k) for k in g1k}) != _norm({k: p2row["G1"].get(k) for k in g1k}):
        bad.append("G1")
    sp = lambda d: None if d is None else {k: {**{q: v[q] for q in v if q != "owner"}, "owner_state_hash": v["owner"]["state_hash"]} for k, v in d.items()}
    if _norm(sp(row.get("score_paths"))) != _norm(sp(p2row.get("score_paths"))):
        bad.append("score_paths")
    return bad


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
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
    if plan["plan_hash"] != m["plan_hash"] or plan["execution_order"] != m["execution_order"]:
        raise ValueError("plan")
    byid = {x["runtime"]["utterance_id"]: x for x in plan["rows"]}
    order = plan["execution_order"]
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
    text = lambda t: tok.decode(t, skip_special_tokens=True)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    counters = {"A4": {}, "theta0_decodes": 0, "G0_decodes": 0, "detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0,
                "G1_decodes": 0, "G2_decodes": 0, "encoder_passes": 0, "detect_language_calls": 0, "audit": {"forwards": 0, "backwards": 0},
                "rows_entered_after_barrier": 0}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": time.time(), "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": g_t0.theta0_hash,
               "nonln_hash_start": nonln0, "replay_phase1_sec": 0.0, "replay_phase2_sec": 0.0, "remaining_sec": 0.0, "barrier": None}
    atomic_json(out / "runtime.json", runtime)
    invalid, written = [], {}

    def write(u, row):
        ci = byid[u]["audit"]["canonical_index"]
        atomic_json(out / "rows" / f"{ci:03d}.json", row)
        written[u] = row

    def episode(x):
        """Unchanged A4 reconstruction for one utterance; returns (enc_inf, adapt result, inputs)."""
        s = x["runtime"]
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
        return enc_inf, enc_train, inputs

    def native(inputs):
        with torch.no_grad():
            lang = int(b_t0.model.detect_language(input_features=inputs["input_features"], generation_config=b_t0.model.generation_config)[0])
        counters["detect_language_calls"] += 1
        return lang

    def live_check(enc_train, cA, s):
        la = live_audit.live_safe_kl_check(b_a4.model, g_a4.names, enc_train, cA, CB, s["y_A"], partition, suppress=suppress, begin=begin, tokenizer=tok)
        counters["audit"]["forwards"] += 3
        counters["audit"]["backwards"] += 1
        return la

    def finish_live(la, r, row, uid):
        g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
        la["primary_loss"] = r["log"]["losses"][0]
        la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
        la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
        la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
        la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
        row["live_audit"] = la
        if not la["pass"]:
            invalid.append(f"{uid}: live audit")

    def a4_log(r):
        lg = r["log"]
        return {k: lg[k] for k in ("losses", "grad_l2", "finite", "d_E", "d_M", "d_anchor", "valid", "content", "n_E", "n_M", "n_O", "n_nonE",
                                   "identical_condition", "noop", "steps", "loss_evaluations", "master_delta_l2", "master_delta_rel",
                                   "effective_delta_l2", "effective_changed_scalars", "R_E")}

    def ckpt_equal(path, eff):
        with np.load(ROOT / path) as z:
            ck = {n: torch.from_numpy(z[f"A4_p{j:03d}"]).to(torch.bfloat16) for j, n in enumerate(names)}
        return all(torch.equal(eff[n].cpu(), ck[n]) for n in names)

    try:
        torch.cuda.reset_peak_memory_stats()
        # ================= Replay12: PATH2 phase 1 (reconstruction barrier) =================
        t1 = time.time()
        rows_r, keep = {}, {}
        for i, uid in enumerate(order[:12]):
            x = byid[uid]
            s, au = x["runtime"], x["audit"]
            row = {"identity": uid, "canonical_index": au["canonical_index"], "dialogue_id": au["dialogue_id"], "partition": au["partition"],
                   "execution_index": i, "manifest_hash": m["manifest_hash"]}
            enc_inf, enc_train, inputs = episode(x)
            d0 = clamp_decode(b_t0, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_decodes"] += 1
            row["theta0_equals_B0"] = d0["tokens"] == au["B0"]["tokens"] and d0["terminated"] == au["B0"]["terminated"]
            lang = native(inputs)
            row["auto_condition"] = {"lang_id": lang, "cA": auto_prompt(lang), "identical_to_cB": auto_prompt(lang) == CB}
            row["lang_equals_sealed"] = lang == au["A4_sealed"]["lang_id"]
            cA = auto_prompt(lang)
            la = live_check(enc_train, cA, s) if i == 0 else None
            r = adapt_a4(b_a4.model, g_a4, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], s["classes"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A4"], keep_grad0=(i == 0))
            row["A4_log"] = a4_log(r)
            row["effective_equals_checkpoint_bf16"] = ckpt_equal(au["A4_sealed"]["checkpoint"], r["effective"])
            try:
                g_a4.materialize(r["effective"])
                row["A4_state_hash"] = g_a4.current_hash()
                free = clamp_decode(b_a4, enc_inf, CB, max_new_tokens=MAX_NEW)
                counters["G0_decodes"] += 1
            finally:
                g_a4.restore()
            row["reset_ok_phase1"] = g_a4.verify() and g_a4.current_hash() == g_a4.theta0_hash and g_t0.verify()
            row["G0"] = {"tokens": free["tokens"], "terminated": free["terminated"]}
            row["G0_equals_sealed_A4"] = free["tokens"] == au["A4_sealed"]["tokens"] and free["terminated"] == au["A4_sealed"]["terminated"]
            if la is not None:
                finish_live(la, r, row, uid)
            for k in ("theta0_equals_B0", "lang_equals_sealed", "effective_equals_checkpoint_bf16", "reset_ok_phase1", "G0_equals_sealed_A4"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            rows_r[uid] = row
            keep[uid] = (enc_inf, {n: v.cpu() for n, v in r["effective"].items()})
            del r
            print(f"P2-PATH3 replay phase1 {i + 1}/12 {uid} G0==A4 {row['G0_equals_sealed_A4']} ckpt {row['effective_equals_checkpoint_bf16']}", flush=True)
        runtime["replay_phase1_sec"] = time.time() - t1
        if invalid:
            raise TTAInvalid("replay phase-1 barrier failed: " + "; ".join(invalid))
        # ================= Replay12: unchanged G1 controller =================
        t2 = time.time()
        for uid in order[:12]:
            enc_inf, eff = keep.pop(uid)
            row = rows_r[uid]
            try:
                g_a4.materialize({n: v.to(g_a4.theta0[n].device) for n, v in eff.items()})
                if g_a4.current_hash() != row["A4_state_hash"] or g_t0.current_hash() != g_t0.theta0_hash:
                    raise TTAInvalid("instance states")
                rec = online_g1(b_t0, b_a4, g_t0, g_a4, enc_inf, uid, counters, eos)
            finally:
                g_a4.restore()
            invalid.extend(f"{uid}: {f}" for f in rec.pop("fails"))
            row.update(rec)
            if rec["decision"] is None and row["G1"] != row["G0"]:
                invalid.append(f"{uid}: no-trigger output != live A4")
            if rec["decision"] is not None and rec["decision"]["winner"] == "A4" and row["G1"]["tokens"] != row["G0"]["tokens"]:
                invalid.append(f"{uid}: A4 winner but G1 != live A4")
            row["reset_ok_phase2"] = g_a4.verify() and g_a4.current_hash() == g_a4.theta0_hash and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash
            row["reset_hash_after"] = g_a4.current_hash()
            p2row = json.loads((ROOT / byid[uid]["audit"]["PATH2_row"]).read_text())
            row["replay_mismatches"] = replay_mismatches(row, p2row)
            row["replay_exact"] = not row["replay_mismatches"]
            if not row["replay_exact"]:
                invalid.append(f"{uid}: replay mismatch {row['replay_mismatches']}")
            row["G0"]["text"], row["G1"]["text"] = text(row["G0"]["tokens"]), text(row["G1"]["tokens"])
            row["A4_text_equals_sealed"] = row["G0"]["text"] == byid[uid]["audit"]["A4_sealed"]["text"]
            row["status"] = "ok"
            write(uid, row)
            dd = row["decision"]
            print(f"P2-PATH3 replay G1 {uid} trigger {row['detector']['trigger']} exact {row['replay_exact']}"
                  + (f" k {dd['k']} winner {dd['winner']} margin {dd['margin']:.4f}" if dd else ""), flush=True)
        runtime["replay_phase2_sec"] = time.time() - t2
        runtime["barrier"] = {"rows": 12, "exact": sum(r["replay_exact"] for r in rows_r.values()), "pass": not invalid}
        atomic_json(out / "runtime.json", runtime)
        if invalid:
            raise TTAInvalid("PATH2 replay barrier failed (remaining 88 not entered): " + "; ".join(invalid))
        # ================= Remaining 88 (fixed100 order), after the barrier =================
        t3 = time.time()
        for i, uid in enumerate(order[12:], start=12):
            x = byid[uid]
            s, au = x["runtime"], x["audit"]
            counters["rows_entered_after_barrier"] += 1
            row = {"identity": uid, "canonical_index": au["canonical_index"], "dialogue_id": au["dialogue_id"], "partition": au["partition"],
                   "execution_index": i, "manifest_hash": m["manifest_hash"]}
            enc_inf, enc_train, inputs = episode(x)
            lang = native(inputs)
            cA = auto_prompt(lang)
            row["auto_condition"] = {"lang_id": lang, "cA": cA, "identical_to_cB": cA == CB}
            r = adapt_a4(b_a4.model, g_a4, enc_train, cA, CB, s["y_A"], s["y_A_valid_mask"], s["classes"], suppress=suppress, begin=begin, eos=eos,
                         partition=partition, counters=counters["A4"])
            row["A4_log"] = a4_log(r)
            row["A4_start_hash"] = g_a4.theta0_hash
            try:
                g_a4.materialize(r["effective"])
                row["A4_state_hash"] = g_a4.current_hash()
                if g_t0.current_hash() != g_t0.theta0_hash:
                    raise TTAInvalid("theta0 instance changed")
                free = clamp_decode(b_a4, enc_inf, CB, max_new_tokens=MAX_NEW)
                counters["G0_decodes"] += 1
                if g_a4.current_hash() != row["A4_state_hash"]:
                    raise TTAInvalid("A4 state changed during G0")
                rec = online_g1(b_t0, b_a4, g_t0, g_a4, enc_inf, uid, counters, eos)
            finally:
                g_a4.restore()
            invalid.extend(f"{uid}: {f}" for f in rec.pop("fails"))
            row["G0"] = {"tokens": free["tokens"], "terminated": free["terminated"]}
            row.update(rec)
            if rec["decision"] is None and row["G1"] != row["G0"]:
                invalid.append(f"{uid}: no-trigger output != live A4")
            if rec["decision"] is not None and rec["decision"]["winner"] == "A4" and row["G1"]["tokens"] != row["G0"]["tokens"]:
                invalid.append(f"{uid}: A4 winner but G1 != live A4")
            row["reset_ok"] = g_a4.verify() and g_a4.current_hash() == g_a4.theta0_hash and g_t0.verify() and g_t0.current_hash() == g_t0.theta0_hash
            row["reset_hash_after"] = g_a4.current_hash()
            if not row["reset_ok"]:
                invalid.append(f"{uid}: reset")
            if au["partition"] == "DEV24_OTHER":                       # integrity comparator only (after the controller)
                row["lang_equals_sealed"] = lang == au["A4_sealed"]["lang_id"]
                row["effective_equals_checkpoint_bf16"] = ckpt_equal(au["A4_sealed"]["checkpoint"], r["effective"])
                row["G0_equals_sealed_A4"] = free["tokens"] == au["A4_sealed"]["tokens"] and free["terminated"] == au["A4_sealed"]["terminated"]
                for k in ("lang_equals_sealed", "effective_equals_checkpoint_bf16", "G0_equals_sealed_A4"):
                    if not row[k]:
                        invalid.append(f"{uid}: {k}")
            del r
            row["G0"]["text"], row["G1"]["text"] = text(row["G0"]["tokens"]), text(row["G1"]["tokens"])
            if au["partition"] == "DEV24_OTHER":
                row["A4_text_equals_sealed"] = row["G0"]["text"] == au["A4_sealed"]["text"]
            row["status"] = "ok"
            write(uid, row)
            dd = row["decision"]
            print(f"P2-PATH3 {i + 1}/100 {uid} {au['partition']} noop {row['A4_log']['noop']} trigger {row['detector']['trigger']}"
                  + (f" k {dd['k']} winner {dd['winner']} margin {dd['margin']:.4f}" if dd else ""), flush=True)
        runtime["remaining_sec"] = time.time() - t3
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        try:
            g_a4.restore()
        except Exception:
            pass
    runtime["rows_written"] = len(written)
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
        raise SystemExit("P2-PATH3 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
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
