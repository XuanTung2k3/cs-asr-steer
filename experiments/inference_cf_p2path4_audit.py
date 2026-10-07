#!/usr/bin/env python
"""Independent P2-PATH4-R1 auditor. Does NOT import the PATH4 runner/analysis, or csasr.inference_cf.consensus_guard /
branch_adjudication / path_decode / eos_boundary. Own fixed100/partition/order hashing from sealed token files, own first
divergences and EOS/content classification, trigger/prefix recomputation from recorded per-step argmax pairs, content
G1 score arithmetic/consensus/tie/winner/decision hash, EOS_BOUNDARY H=1 consensus/tie/selected action/termination,
reconstruction identities, own streams (TTA0 auditor ``own_lev``), canonical counts and transitions (TTA0 auditor
``counts``/``transitions``), safety/rescue/benefit/breadth/LODO/bootstrap/label.

prerun -> PASS_TO_P2_PATH4_R1 (runnable manifest/code/environment review) | BLOCK_BEFORE_P2_PATH4_R1
post   -> P2_PATH4_AUDIT: PASS | BLOCK  (--phase primary before references; --phase full after secondary)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

import experiments.inference_cf_p2tta0_audit as tau

FREEZE = "490ec20"
ORIGINAL = "29661e972f0ebfd727d228d2f6dbd9b17dd41c5a"
CONFIG = "configs/inference_cf/p2_path4.json"
PANEL = "docs/inference_cf/P2_PATH4_PANEL.json"
FROZEN = ("docs/inference_cf/P2_PATH4_SPEC.md", "docs/inference_cf/P2_PATH4_CODEX_DESIGN.md", "docs/inference_cf/P2_PATH4_EOS_AMENDMENT.md",
          PANEL, CONFIG, "docs/inference_cf/P2_PATH4_R1_PRE_RUN_AUDIT.json", "configs/inference_cf/p2_path4_blocked_v0.json",
          "src/csasr/inference_cf/eos_boundary.py", "tests/test_inference_cf_p2path4_r1.py", "experiments/inference_cf_p2path4_r1_contract_audit.py")
BASE = ROOT / "results/inference_cf/p2path4"
PLAN_REL = "results/inference_cf/p2path4/plan_sealed.json"
RUNNER = "experiments/inference_cf_p2path4.py"
EOS = 50257
CB = [50258, 50260, 50360, 50364]
G = 1e-12
AUDIT_TOKENS = ('["audit"]', "au[", '["analysis"]', "first_divergence", "partitions", "A2_DELTA", "NOVEL76", "DEV24", "AUTO", "y_A_text",
                "load_references", "audit_only", "B0_FORCED", "known_blockers", "boundary_rows", "expected_trigger")


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def st(t, term):
    return list(t) + ([EOS] if term == "eos" else [])


def severe(base_len, new_len, term):
    return base_len >= 10 and term == "eos" and new_len <= math.floor(0.5 * base_len)


def own_population():
    """Own fixed100/partitions/first divergences from parent panels and sealed B0/A2 token files (no references)."""
    par = json.loads((ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json").read_text())
    a3 = json.loads((ROOT / "docs/inference_cf/P2_TTA_A3_PANEL.json").read_text())
    p2 = json.loads((ROOT / "docs/inference_cf/P2_PATH2_PANEL.json").read_text())
    Pn = {r["utterance_id"]: r for r in json.loads((ROOT / PANEL).read_text())["rows"]}
    full = [r["utterance_id"] for r in par["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in par["rows"]}
    dev = [r["utterance_id"] for r in a3["rows"]]
    p12 = [r["utterance_id"] for r in p2["rows"]]
    tok = {}
    for u in full:
        b = json.loads((ROOT / Pn[u]["B0_FORCED"]["path"]).read_text())["systems"]["S0"]
        a = json.loads((ROOT / Pn[u]["A2"]["path"]).read_text())["objectives"]["A2"]
        tok[u] = (b, a)
    fd = {}
    for u in full:
        bs, as_ = st(tok[u][0]["tokens"], tok[u][0]["terminated"]), st(tok[u][1]["tokens"], tok[u][1]["terminated"])
        if bs != as_:
            k = next(i for i, (x, y) in enumerate(zip(bs, as_)) if x != y)
            fd[u] = {"k": k, "t0": bs[k], "t2": as_[k], "boundary": (bs[k] == EOS) != (as_[k] == EOS)}
    sdev = set(dev)
    parts = {"FULL100": full, "DEV24": dev, "PATH2_12": p12, "NOVEL76": [u for u in full if u not in sdev],
             "A2_DELTA": [u for u in full if u in fd], "A2_SAME": [u for u in full if u not in fd],
             "A2_DELTA_OUTSIDE_DEV24": [u for u in full if u in fd and u not in sdev]}
    return parts, dlg, fd, tok, Pn


# ---- own frozen rules -------------------------------------------------------------------------------------------

def own_rules(Z, POI, PIER, MER, MIX, r, dlg, novel):
    X = max(Z[1] - Z[0], 0)
    req = max(1, math.ceil(0.5 * X)) if X > 0 else 1
    resc = Z[1] - Z[2] >= req
    IA, IG = POI[0] - POI[1], POI[0] - POI[2]
    ret = IG > 0 and IA > 0 and IG / IA >= 0.75 - G and PIER[1] - PIER[0] <= 0.01 + G and MER[1] - MER[0] <= 0.005 + G and MIX[1] <= MIX[0]
    pos = {u: max(x, 0) for u, x in r.items()}
    Rp = sum(pos.values())
    Pd = {}
    for u, x in pos.items():
        Pd[dlg[u]] = Pd.get(dlg[u], 0) + x
    cu = max(pos.values()) / Rp if Rp > 0 else 1.0
    cd = max(Pd.values()) / Rp if Rp > 0 else 1.0
    brd = (sum(x >= 1 for x in r.values()) >= 3 and sum(v > 0 for v in Pd.values()) >= 3 and cu <= 0.5 + G and cd <= 0.6 + G
           and sum(x for u, x in r.items() if u in novel) >= 1 and sum(1 for u, x in r.items() if u in novel and x >= 1) >= 1)
    return {"X": X, "required": req, "rescue": resc, "I_A2": IA, "I_G": IG, "retention_pass": ret, "C_utt": cu, "C_dlg": cd, "breadth": brd}


def own_label(valid, safe, resc, ret, brd):
    for ok, lab in ((valid, "P2_PATH4_INVALID"), (safe, "P2_PATH4_SEQUENCE_DAMAGE"), (resc, "P2_PATH4_SAFE_NO_ADDED_VALUE"),
                    (ret, "P2_PATH4_OVERCONSERVATIVE"), (brd, "P2_PATH4_SIGNAL_CONCENTRATED")):
        if not ok:
            return lab
    return "P2_PATH4_GUARD_TRANSFER_SUPPORTED"


def _fn_source(rel: str, name: str) -> str:
    src = (ROOT / rel).read_text()
    return next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in FROZEN}
    checks["panel_hash"] = sha(ROOT / PANEL) == c["panel_byte_sha256"] == "f51da5b1a5a51a9f5d7d979c09ec23a095eca8c7528dad3f1f098368c6a1a084"
    checks["anchors_and_helpers_unchanged"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items()) and \
        all(sha(ROOT / p) == h for p, h in c["amendment_source_sha256"].items())
    arch = c["original_blocked_provenance"]
    checks["blocked_provenance_preserved"] = (arch["commit"] == ORIGINAL and arch["label"] == "P2_PATH4_BLOCKED_EOS_FIRST_BRANCH"
                                              and sha(ROOT / arch["config_archive"]) == arch["config_archive_sha256"]
                                              and tau.blob_sha(ORIGINAL, "configs/inference_cf/p2_path4.json") == arch["config_archive_sha256"]
                                              and tau.blob_sha(ORIGINAL, PANEL) == sha(ROOT / PANEL)
                                              and json.loads((ROOT / PANEL).read_text())["status"] == "BLOCKED_PRE_RUN_EOS_FIRST")
    fa = json.loads((ROOT / "docs/inference_cf/P2_PATH4_R1_PRE_RUN_AUDIT.json").read_text())
    checks["freeze_contract_audit"] = fa["verdict"] == "PASS_TO_P2_PATH4_R1" and fa["config_sha256"] == sha(ROOT / CONFIG) and all(fa["checks"].values())
    checks["revision"] = c["contract_revision"] == "PATH4_R1_EOS_BOUNDARY_V1" and c["G1"]["EOS_BOUNDARY"]["H_actions"] == 1 and c["G1"]["H"] == 3 \
        and c["G1"]["weights"] == [0.5, 0.5] and c["G1"]["tie_abs"] == 1e-12
    parts, dlg, fd, tok, Pn = own_population()
    checks["partitions_independent"] = (
        set(parts["PATH2_12"]) < set(parts["DEV24"]) < set(parts["FULL100"]) and {k: len(v) for k, v in parts.items()} == c["counts"]
        and all(fp({"partition": k, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in v]}) == c["partition_sha256"][k] for k, v in parts.items())
        and parts["A2_DELTA"] == c["G1"]["expected_trigger_ids"])
    bnd = [u for u in parts["A2_DELTA"] if fd[u]["boundary"]]
    checks["four_boundary_ten_content"] = (bnd == [r["utterance_id"] for r in c["known_pre_run_block"]["rows"]] and len(bnd) == 4
                                           and all(fd[u]["t0"] == EOS and fd[u]["t2"] != EOS for u in bnd)
                                           and sum(not fd[u]["boundary"] for u in parts["A2_DELTA"]) == 10)
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and plan["ids"] == parts["FULL100"] and plan["partitions"] == parts
    t1 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())["rows"]}
    bad = []
    for x in plan["rows"]:
        s = x["runtime"]
        u = s["utterance_id"]
        okr = set(s) == {"utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "y_B", "y_B_valid_mask"}
        okr &= all(s[k] == t1[u][k] for k in ("y_A", "y_A_valid_mask", "y_B", "y_B_valid_mask", "audio_full_sha256", "audio_path"))
        okr &= x["audit"]["B0"] == {"tokens": tok[u][0]["tokens"], "terminated": tok[u][0]["terminated"]} and x["audit"]["A2"]["tokens"] == tok[u][1]["tokens"]
        okr &= sha(ROOT / x["audit"]["A2_checkpoint"]) == Pn[u]["A2_checkpoint"]["byte_sha256"]
        if not okr:
            bad.append(u)
    checks["runtime_inputs_teachers_checkpoints"] = not bad
    rsrc = (ROOT / RUNNER).read_text()
    ctrl = _fn_source(RUNNER, "online_g1_r1")
    run_code = _fn_source(RUNNER, "cmd_run")
    checks["controller_input_allowlist"] = not any(t in ctrl for t in AUDIT_TOKENS) and \
        "def online_g1_r1(b_t0, b_a2, g_t0, g_a2, enc, owner_row: str, counters: dict, eos: int)" in rsrc
    checks["controller_dispatch_semantics"] = (ctrl.count("lockstep_detect(") == 1 and ctrl.count("path3.online_g1(") == 1 and ctrl.count("score_boundary(") == 1
                                               and ctrl.count("execute_boundary(") == 1 and "action_mode(" in ctrl and "max_new_tokens=MAX_NEW" in ctrl
                                               and "replayed detection differs" in ctrl and "rollout(" not in ctrl.replace("rollouts", ""))
    checks["original_G1_unchanged"] = sha(ROOT / "experiments/inference_cf_p2path3.py") == c["source_sha256"]["experiments/inference_cf_p2path3.py"]
    checks["no_G2_no_A4"] = ("g2_forced" not in rsrc and "G2_replay" not in rsrc and "adapt_a4" not in rsrc and "script_safe_tta" not in rsrc
                             and "detect_language" not in rsrc and 'run_objective(b_a2, g_a2, "A2"' in run_code)
    checks["two_instances_no_alias"] = "b_t0, b_a2 = load_whisper(cfg_m), load_whisper(cfg_m)" in run_code and "data_ptr()" in run_code
    i_bar = run_code.index("A2 reconstruction barrier failed (G1 phase not entered)")
    checks["barrier100_before_G1"] = (run_code.index("for i, x in enumerate(plan[\"rows\"])") < i_bar < run_code.index("online_g1_r1(")
                                      and "masters_equal_checkpoint_fp32" in run_code and "np.array_equal" in run_code and run_code.count("online_g1_r1(") == 1)
    i_ctrl = run_code.index("online_g1_r1(")
    checks["comparator_engineering_only"] = all(m.start() > i_ctrl for m in re.finditer(r"path3\.online_g1\(", run_code)) and "cc = counters[\"comparator\"]" in run_code
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path4_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path4_analyze|inference_cf_p2path4\b|consensus_guard|branch_adjudication|path_decode|eos_boundary)",
                                              asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path4.py").exists()
    fsrc = (ROOT / "experiments/inference_cf_p2path4_analyze.py").read_text()
    prim_src = fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
    checks["reference_barrier"] = ("load_references" not in prim_src and "PermissionError" in fsrc and "committed(SEAL)" in fsrc
                                   and "committed(PRIMARY_AUDIT)" in fsrc and "load_references" not in rsrc)
    TH = json.loads((ROOT / "configs/inference_cf/p2_tta_funnel.json").read_text())["TTA1"]["thresholds"]
    checks["safety_thresholds_TTA1"] = all(c["safety"][k] == TH[k] for k in ("max_MER_increase", "max_ZH_CER_increase", "min_matrix_ZH_retention",
                                                                            "min_embedded_EN_retention", "max_outside_POI_harm_rate", "max_POI_corruption_rate",
                                                                            "max_additional_caps", "max_new_severe_truncations", "float_guard"))
    # own frozen-cutoff arithmetic (historical aggregates) and truth tables
    D = {u: dlg[u] for u in parts["FULL100"]}
    nov = set(parts["NOVEL76"])
    z = lambda zg: own_rules((1107, 1116, zg), (345, 316, 330), (0.4957, 0.4540, 0.45), (0.257, 0.2537, 0.25), (1483, 1464, 1460), {}, D, nov)["rescue"]
    pier = lambda poi: (poi) / 696
    rt_ = lambda poi, mix=1460, dm=0.0: own_rules((1107, 1116, 1100), (345, 316, poi), (pier(316), pier(poi)), (0.25, 0.25 + dm), (1464, mix), {}, D, nov)["retention_pass"]
    checks["frozen_cutoffs"] = (z(1111) and not z(1112) and rt_(322) and not rt_(323) and not rt_(324) and rt_(316) and not rt_(345)
                                and not rt_(320, mix=1465) and rt_(320, mix=1464) and not rt_(320, dm=0.0051) and rt_(320, dm=0.005))
    nv = sorted(nov)
    dd = lambda us: {u: D[u] for u in us}
    B = lambda rr: own_rules((1107, 1116, 1100), (345, 316, 320), (0, 0), (0, 0), (1464, 1460), rr, D, nov)["breadth"]
    novd = {}
    for u in nv:
        novd.setdefault(D[u], []).append(u)
    nd = sorted(novd)
    three = {novd[nd[0]][0]: 2, novd[nd[1]][0]: 1, novd[nd[2]][0]: 1}
    tt = [B(three), not B({novd[nd[0]][0]: 20}), not B({novd[nd[0]][0]: 2, novd[nd[0]][1]: 2, novd[nd[0]][2]: 2}),
          not B({**three, novd[nd[3]][0]: -4}), not B({}), B({**three, novd[nd[3]][0]: 1})]
    devs = [u for u in parts["DEV24"]]
    tt.append(not B({devs[0]: 2, devs[5]: 1, devs[10]: 1} if len({D[devs[0]], D[devs[5]], D[devs[10]]}) == 3 else {}))   # no NOVEL76 rescue
    labs = {(v, s, r_, t, b): own_label(v, s, r_, t, b) for v, s, r_, t, b in itertools.product((False, True), repeat=5)}
    exp = lambda v, s, r_, t, b: ("P2_PATH4_INVALID" if not v else "P2_PATH4_SEQUENCE_DAMAGE" if not s else "P2_PATH4_SAFE_NO_ADDED_VALUE" if not r_
                                  else "P2_PATH4_OVERCONSERVATIVE" if not t else "P2_PATH4_SIGNAL_CONCENTRATED" if not b else "P2_PATH4_GUARD_TRANSFER_SUPPORTED")
    checks["own_rule_truth_tables"] = all(tt) and all(labs[k] == exp(*k) for k in labs) and len(labs) == 32
    # own H=1 boundary arithmetic on synthetic log-probs (no helper import)
    s0, s2 = {"B": -1.0, "ALT": -2.0}, {"B": -2.0, "ALT": -1.0}
    cons = {k: 0.5 * s0[k] + 0.5 * s2[k] for k in s0}
    checks["own_H1_tie_rule"] = (cons["B"] - cons["ALT"]) >= -G and not ((-1e-11) >= -G) and (-1e-12 >= -G)
    Bc = c["compute"]
    checks["compute_plan"] = (Bc["jobs_max"] == 1 and Bc["optimizer_steps"] == 200 and Bc["backwards_max_including_independent_live_check"] == 201
                              and Bc["hard_minutes"] == 30 and Bc["rows"] == 100)
    v = "PASS_TO_P2_PATH4_R1" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH4_R1"
    return {"schema": "p2_path4_r1_runnable_prerun_audit_v1", "audit_scope": "runnable manifest/code/environment/reconstruction/cache review",
            "verdict": v, "checks": checks, "failures": bad, "git_commit": _git("rev-parse", "HEAD")}


def cmd_post(args) -> dict:
    c = cfg()
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    parts, dlg, fd, tok, Pn = own_population()
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]
              and rt["elapsed_sec"] <= 1800 and rt["barrier"]["pass"]}
    if args.phase == "full":
        seal = json.loads((BASE / "output_seal.json").read_text())
        checks["seal_committed_unchanged"] = committed("results/inference_cf/p2path4/output_seal.json") and \
            all(sha(ROOT / p) == h for p, h in seal["files"].items()) and seal["rows_sealed"] == 100
    files = sorted((run / "rows").glob("*.json"))
    checks["exactly_100_rows"] = [f.name for f in files] == [f"{i:03d}.json" for i in range(100)]
    rows = {}
    for f in files:
        r = json.loads(f.read_text())
        rows[r["identity"]] = r
    from transformers import WhisperProcessor
    tk = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    bad = []
    n = {"trig": 0, "content": 0, "boundary": 0, "t0w": 0, "changed": 0}
    bnd_rows = []
    for i, u in enumerate(parts["FULL100"]):
        r = rows[u]
        b0, a2 = tok[u]
        okr = r["status"] == "ok" and r["canonical_index"] == i and r["dialogue_id"] == dlg[u]
        okr &= r["theta0"]["tokens"] == b0["tokens"] and r["theta0"]["terminated"] == b0["terminated"]
        A = r["A2_free"]
        okr &= A["tokens"] == a2["tokens"] and A["terminated"] == a2["terminated"] and A["text"] == a2["text"]
        okr &= r["masters_equal_checkpoint_fp32"] and r["effective_equals_checkpoint_bf16"] and r["reset_ok_phase1"] and r["reset_ok_phase2"]
        lg = r["A2_log"]
        okr &= lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"]) and lg["start_hash"] == lg["end_hash"] == rt["theta0_ln_hash"]
        Gr = r["G1"]
        okr &= Gr["text"] == tk.decode(Gr["tokens"], skip_special_tokens=True)
        det = r["detector"]
        pairs = det["argmax_pairs"]
        first = next((j for j, (p, q) in enumerate(pairs) if p != q), None)
        okr &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        okr &= det["trigger"] == (first is not None) == (u in fd) and det["prefix"] == [p for p, q in pairs[:len(det["prefix"])]]
        okr &= det["prefix"] == A["tokens"][:len(det["prefix"])]
        if first is None:
            okr &= r["mode"] == "NO_TRIGGER" and r["decision"] is None and Gr["tokens"] == A["tokens"] and Gr["terminated"] == A["terminated"]
            okr &= r["comparator"]["ok"] and r["comparator"]["mismatches"] == []
        else:
            n["trig"] += 1
            d = r["decision"]
            c0, c2 = pairs[first]
            okr &= first == det["k"] == len(det["prefix"]) == fd[u]["k"] and (c0, c2) == (fd[u]["t0"], fd[u]["t2"])
            is_b = (c0 == EOS) != (c2 == EOS)
            okr &= r["mode"] == ("EOS_BOUNDARY" if is_b else "CONTENT_G1") and is_b == fd[u]["boundary"]
            if not is_b:
                n["content"] += 1
                S = {k_: sum(v["logprobs"]) / len(v["logprobs"]) for k_, v in d["scores"].items()}
                okr &= all(S[k_] == d["scores"][k_]["score"] for k_ in S)
                cons = {"b0": 0.5 * S["theta0_b0"] + 0.5 * S["A4_b0"], "b4": 0.5 * S["theta0_b4"] + 0.5 * S["A4_b4"]}
                win = "theta0" if cons["b0"] - cons["b4"] >= -G else "A2"
                okr &= (d["winner"] == "theta0") == (win == "theta0") and cons["b0"] - cons["b4"] == d["margin"]
                okr &= d["b0"]["tokens"][0] == c0 and d["b4"]["tokens"][0] == c2 and d["b0"]["H_eff"] >= 1 and d["b4"]["H_eff"] >= 1
                okr &= d["b4"]["tokens"] == A["tokens"][first:first + len(d["b4"]["tokens"])]
                wtok = (d["b0"] if win == "theta0" else d["b4"])["tokens"][0]
                okr &= d["decision_hash"] == Gr["decision_hash"] == "sha256:" + hashlib.sha256(json.dumps(
                    {k_: d[k_] for k_ in ("k", "prefix", "b0", "b4", "scores", "winner", "margin")}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                okr &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"] for v in r["score_paths"].values())
                okr &= r["score_paths"]["A4_b0"]["owner"]["state_hash"] == r["A2_state_hash"] and r["score_paths"]["theta0_b0"]["owner"]["state_hash"] == rt["theta0_ln_hash"]
                okr &= r["comparator"]["ok"] and r["comparator"]["mismatches"] == []
            else:
                n["boundary"] += 1
                s0, s2 = d["S_theta0"], d["S_A2"]
                okr &= all(math.isfinite(v) and v <= 0 for v in list(s0.values()) + list(s2.values())) and d["H"] == 1 and (d["c0"], d["c2"]) == (c0, c2)
                cons = {k_: 0.5 * s0[k_] + 0.5 * s2[k_] for k_ in ("B", "ALT")}
                win = "theta0" if cons["B"] - cons["ALT"] >= -G else "A2"
                wtok = c0 if win == "theta0" else c2
                okr &= d["winner"] == win and d["S_cons"] == cons and d["margin"] == cons["B"] - cons["ALT"] and d["selected_token"] == wtok
                okr &= d["k"] == first and d["prefix"] == det["prefix"] and all(p["state_locked"] and p["fed"] == CB + det["prefix"] for p in d["paths"].values())
                okr &= d["paths"]["A2"]["state_hash"] == r["A2_state_hash"] and d["paths"]["theta0"]["state_hash"] == rt["theta0_ln_hash"]
                okr &= d["decision_hash"] == Gr["decision_hash"] == "sha256:" + hashlib.sha256(json.dumps(
                    {k_: d[k_] for k_ in ("contract_revision", "mode", "H", "k", "prefix", "c0", "c2", "S_theta0", "S_A2", "S_cons", "margin", "winner",
                                          "selected_token")}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                okr &= r["comparator"]["original_refused_EOS_first"] is True
                if wtok == EOS:
                    okr &= Gr["tokens"] == det["prefix"] and Gr["terminated"] == "eos"
                bnd_rows.append({"utterance_id": u, "orientation": "theta0_EOS" if c0 == EOS else "A2_EOS", "winner": win,
                                 "action": "EOS" if wtok == EOS else "content", "G_len": len(Gr["tokens"]), "A2_len": len(A["tokens"]), "B0_len": len(b0["tokens"])})
            okr &= Gr["forced"] == [wtok] and [f_["token"] for f_ in Gr["trace"]["forced"]] == [wtok] and Gr["trace"]["prefix_ok"] and Gr["state_locked"]
            gs = st(Gr["tokens"], Gr["terminated"])
            okr &= gs[:first] == det["prefix"] and gs[first] == wtok
            if win == "A2":
                okr &= Gr["tokens"] == A["tokens"] and Gr["terminated"] == A["terminated"]
            n["t0w"] += win == "theta0"
        n["changed"] += st(Gr["tokens"], Gr["terminated"]) != st(A["tokens"], A["terminated"])
        if not okr:
            bad.append(u)
    checks["rows_reconstruction_dispatch_controller_recomputed"] = not bad
    la = rows[parts["FULL100"][0]]["live_audit"]
    checks["live_audit_2pct"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    checks["counts"] = (n["trig"] == 14 == cnt["triggers"] and n["content"] + n["boundary"] == n["trig"] and cnt["logical_guard_events"] == 100
                        and cnt["rollouts"] == 2 * n["content"] and cnt["score_paths"] == 4 * n["content"] and cnt["G1_decodes"] == n["content"]
                        and cnt["boundary_score_paths"] == 2 * n["boundary"] and cnt["boundary_executions"] == n["boundary"] and cnt["G2_decodes"] == 0
                        and cnt["A2"]["optimizer_steps"] == 200 and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= 201 and cnt["theta0_decodes"] == 100)
    eb = prim.get("EOS_boundary", {})
    sm = prim.get("summaries", {}).get("FULL100", {})
    checks["summaries_agree"] = (eb.get("count") == n["boundary"] and sm.get("triggers") == n["trig"] and sm.get("theta0_winners") == n["t0w"]
                                 and sm.get("G_changed_vs_A2") == n["changed"] and sm.get("content_G1") == n["content"]
                                 and [(b["utterance_id"], b["orientation"], b["winner"], b["winner_action"], b["G_length"]) for b in eb.get("rows", [])]
                                 == [(b["utterance_id"], b["orientation"], b["winner"], b["action"], b["G_len"]) for b in bnd_rows])
    checks["primary_valid_agrees"] = prim["valid"] == all(checks.values())
    out = {"counts": n, "boundary": bnd_rows}
    lab_out = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        refs = load_references()
        ids = parts["FULL100"]
        R = {u: refs[u]["reference"] for u in ids}
        Dd = {u: refs[u]["dialogue_id"] for u in ids}
        checks["dialogues"] = all(Dd[u] == dlg[u] for u in ids)
        t1 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())["rows"]}
        H = {u: {"B0_FORCED": t1[u]["y_B_text"], "B0_AUTO": t1[u]["y_A_text"], "A2": tok[u][1]["text"], "G": rows[u]["G1"]["text"]} for u in ids}
        SY = ("B0_FORCED", "B0_AUTO", "A2", "G")
        C = {u: {s_: tau.counts(R[u], H[u][s_]) for s_ in SY} for u in ids}
        rate = lambda us, s_, a_, b_: sum(C[u][s_][a_] for u in us) / sum(C[u][s_][b_] for u in us)
        mine = {}
        for k, us in parts.items():
            mine[k] = {s_: {"zh": sum(C[u][s_][4] for u in us), "poi": sum(C[u][s_][0] for u in us), "mixed": sum(C[u][s_][2] for u in us),
                            "pier": rate(us, s_, 0, 1), "mer": rate(us, s_, 2, 3), "zh_cer": rate(us, s_, 4, 5), "en_wer": rate(us, s_, 6, 7)} for s_ in SY}
        sm2 = sec["metrics"]
        checks["metrics_agree"] = all(mine[k][s_][q] == sm2[k][s_][q] for k in parts for s_ in SY for q in ("zh", "poi", "mixed")) and \
            max(abs(mine[k][s_][q] - sm2[k][s_][q]) for k in parts for s_ in SY for q in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12
        F = mine["FULL100"]
        checks["historical_aggregates"] = [F["B0_FORCED"][q] for q in ("zh", "poi", "mixed")] == [1107, 345, 1483] and \
            [F["A2"][q] for q in ("zh", "poi", "mixed")] == [1116, 316, 1464] and sum(C[u]["B0_FORCED"][1] for u in ids) == 696
        tr = tau.transitions([R[u] for u in ids], [H[u]["B0_FORCED"] for u in ids], [H[u]["G"] for u in ids])
        added = sum(rows[u]["G1"]["terminated"] == "cap" and t1[u]["y_B_terminated"] != "cap" for u in ids)
        sev = sum(severe(len(t1[u]["y_B"]), len(rows[u]["G1"]["tokens"]), rows[u]["G1"]["terminated"]) for u in ids)
        TH = c["safety"]
        safe = {"MER": F["G"]["mer"] - F["B0_FORCED"]["mer"] <= TH["max_MER_increase"] + G,
                "ZH_CER": F["G"]["zh_cer"] - F["B0_FORCED"]["zh_cer"] <= TH["max_ZH_CER_increase"] + G,
                "ZH_ret": tr["zr"] is None or tr["zr"] >= TH["min_matrix_ZH_retention"] - G,
                "EN_ret": tr["er"] is None or tr["er"] >= TH["min_embedded_EN_retention"] - G,
                "outside": tr["ohr"] is None or tr["ohr"] <= TH["max_outside_POI_harm_rate"] + G,
                "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= TH["max_POI_corruption_rate"] + G,
                "caps": added <= TH["max_additional_caps"], "severe": sev <= TH["max_new_severe_truncations"]}
        sp = sec["safety_point"]
        checks["safety_points_agree"] = (tr["zr"] == sp["zh_retention"] and tr["er"] == sp["en_retention"] and tr["ohr"] == sp["outside_harm_rate"]
                                         and tr["corr_rate"] == sp["poi_corruption_rate"] and added == sp["added_caps"] and sev == sp["new_severe_truncations"]
                                         and all(safe.values()) == sec["ABSOLUTE_SAFETY_PASS"])
        r = {u: C[u]["A2"][4] - C[u]["G"][4] for u in ids}
        rules = own_rules((F["B0_FORCED"]["zh"], F["A2"]["zh"], F["G"]["zh"]), (F["B0_FORCED"]["poi"], F["A2"]["poi"], F["G"]["poi"]),
                          (F["A2"]["pier"], F["G"]["pier"]), (F["A2"]["mer"], F["G"]["mer"]), (F["A2"]["mixed"], F["G"]["mixed"]), r, Dd, set(parts["NOVEL76"]))
        ar, be, br = sec["aggregate_rescue"], sec["benefit"], sec["breadth"]
        checks["rescue_benefit_breadth_agree"] = (rules["rescue"] == ar["pass"] and rules["X"] == ar["X_ZH"] and rules["required"] == ar["required_R_ZH"]
                                                  and rules["I_A2"] == be["I_A2"] and rules["I_G"] == be["I_G"]
                                                  and rules["retention_pass"] == be["BENEFIT_RETENTION_PASS"] and rules["breadth"] == br["BREADTH_PASS"]
                                                  and abs(rules["C_utt"] - br["C_utt"]) <= 1e-12 and abs(rules["C_dlg"] - br["C_dlg"]) <= 1e-12
                                                  and sum(r.values()) == br["R_net"])
        keys = sorted(set(Dd.values()))
        lv = [sum(x for u, x in r.items() if Dd[u] != k) for k in keys]
        changed = sorted({Dd[u] for u in ids if st(rows[u]["G1"]["tokens"], rows[u]["G1"]["terminated"]) != st(tok[u][1]["tokens"], tok[u][1]["terminated"])})
        lc = [sum(x for u, x in r.items() if Dd[u] != k) for k in changed]
        lo = sec["lodo"]
        checks["lodo_agrees"] = (len(keys) == 20 and min(lv) == lo["all20"]["min"] and sum(v > 0 for v in lv) == lo["all20"]["count_positive"]
                                 and changed == lo["changed_dialogues"] and sum(v > 0 for v in lc) == lo["changed"]["count_positive"]
                                 and (min(lc) if lc else None) == lo["changed"]["min"])
        grp = {k: [j for j, u in enumerate(ids) if Dd[u] == k] for k in keys}
        rng = np.random.default_rng(240924)
        drs = [sum((grp[keys[j]] for j in rng.integers(0, 20, 20)), []) for _ in range(2000)]
        CA = {s_: np.array([C[u][s_] for u in ids], dtype=float) for s_ in SY}
        bd = []
        for base in ("A2", "B0_FORCED"):
            for nme, (a_, b_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items():
                vals = [CA["G"][ix, a_].sum() / CA["G"][ix, b_].sum() - CA[base][ix, a_].sum() / CA[base][ix, b_].sum()
                        for ix in drs if CA["G"][ix, b_].sum() and CA[base][ix, b_].sum()]
                ref_ = sec["bootstrap"][f"G_minus_{base}"][nme]["ci95"]
                bd += [abs(np.quantile(vals, .025) - ref_[0]), abs(np.quantile(vals, .975) - ref_[1])]
            vals = [CA["G"][ix, 0].sum() - CA[base][ix, 0].sum() for ix in drs]
            ref_ = sec["bootstrap"][f"G_minus_{base}"]["poi_errors"]["ci95"]
            bd += [abs(np.quantile(vals, .025) - ref_[0]), abs(np.quantile(vals, .975) - ref_[1])]
        checks["bootstrap_agrees"] = max(bd) <= 1e-9
        valid = bool(prim["valid"]) and all(v for k, v in checks.items() if k != "label_agrees")
        lab_out = own_label(valid, all(safe.values()), rules["rescue"], rules["retention_pass"], rules["breadth"])
        checks["label_agrees"] = lab_out == sec["label"]
        out.update(metrics=mine, safety=safe, rules=rules, added_caps=added, severe=sev, lodo_all20=lv, lodo_changed=lc)
    v = "P2_PATH4_AUDIT: PASS" if all(checks.values()) else "P2_PATH4_AUDIT: BLOCK"
    return {"schema": "p2_path4_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab_out, "checks": checks, "failures": bad,
            "independent": out, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--phase", choices=("primary", "full"), required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
