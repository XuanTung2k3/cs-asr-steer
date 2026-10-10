#!/usr/bin/env python
"""Independent P2-PATH3 auditor. Does NOT import the PATH3 runner/analysis, or csasr.inference_cf.consensus_guard /
branch_adjudication / path_decode. Own population/partition/order hashing, own token classes and A4 no-op rule, trigger
and prefix recomputation from recorded per-step argmax pairs, score arithmetic, consensus/tie/winner, G1 forced token,
exact PATH2 replay comparison, DEV24 A4 identity, own streams (TTA0 auditor ``own_lev``), canonical counts and
transitions (TTA0 auditor ``counts``/``transitions``), safety/rescue/benefit/breadth/LODO/bootstrap/label.

prerun -> PASS_TO_P2_PATH3 | BLOCK_BEFORE_P2_PATH3
post   -> P2_PATH3_AUDIT: PASS | BLOCK  (--phase primary before references; --phase full after secondary)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
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

FREEZE = "5853a8d"
CONFIG = "configs/inference_cf/p2_path3.json"
PANEL = "docs/inference_cf/P2_PATH3_PANEL.json"
SPEC = "docs/inference_cf/P2_PATH3_SPEC.md"
DESIGN = "docs/inference_cf/P2_PATH3_CODEX_DESIGN.md"
BASE = ROOT / "results/inference_cf/p2path3"
PLAN_REL = "results/inference_cf/p2path3/plan_sealed.json"
RUNNER = "experiments/inference_cf_p2path3.py"
EOS = 50257
CB = [50258, 50260, 50360, 50364]
G = 1e-12
AUDIT_TOKENS = ('["audit"]', "au[", '["analysis"]', "A4_sealed", "PATH2_row", '["partition"]', "AUTO", "y_A_text",
                "load_references", "audit_only", "B0_FORCED", "DEV24", "NOVEL76", "PATH2_12")


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


def own_partitions():
    par = json.loads((ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json").read_text())
    a3 = json.loads((ROOT / "docs/inference_cf/P2_TTA_A3_PANEL.json").read_text())
    p2 = json.loads((ROOT / "docs/inference_cf/P2_PATH2_PANEL.json").read_text())
    full = [r["utterance_id"] for r in par["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in par["rows"]}
    dev = [r["utterance_id"] for r in a3["rows"]]
    p12 = [r["utterance_id"] for r in p2["rows"]]
    nov = [u for u in full if u not in set(dev)]
    order = p12 + [u for u in full if u not in set(p12)]
    return {"FULL100": full, "DEV24": dev, "PATH2_12": p12, "NOVEL76": nov}, dlg, order


# ---- own frozen rules -------------------------------------------------------------------------------------------

def own_rules(Z, POI, MER_G1, MER_A4, r76, d76, c):
    X = max(Z[1] - Z[0], 0)
    req = max(1, math.ceil(0.5 * X)) if X > 0 else 1
    resc = Z[1] - Z[2] >= req
    IA, IG = POI[0] - POI[1], POI[0] - POI[2]
    ret_pass = IG > 0 and (IA <= 0 or IG / IA >= 0.75 - G)
    useful = resc and IG > 0 and MER_G1 - MER_A4 <= 0.005 + G
    pos = [max(x, 0) for x in r76]
    Rp = sum(pos)
    Pd = {}
    for x, d in zip(pos, d76):
        Pd[d] = Pd.get(d, 0) + x
    cu = max(pos) / Rp if Rp > 0 else 1.0
    cd = max(Pd.values()) / Rp if Rp > 0 else 1.0
    brd = sum(r76) >= 1 and sum(x >= 1 for x in r76) >= 3 and sum(v > 0 for v in Pd.values()) >= 3 and cu <= 0.5 + G and cd <= 0.6 + G
    return {"X": X, "required": req, "rescue": resc, "I_A4": IA, "I_G1": IG, "retention_pass": ret_pass, "useful": useful, "C_utt": cu, "C_dlg": cd,
            "breadth": brd}


def own_label(valid, safe, useful, ret, brd):
    for ok, lab in ((valid, "P2_PATH3_INVALID"), (safe, "P2_PATH3_SEQUENCE_DAMAGE"), (useful, "P2_PATH3_NO_USEFUL_GAIN"),
                    (ret, "P2_PATH3_OVERCONSERVATIVE"), (brd, "P2_PATH3_SIGNAL_CONCENTRATED")):
        if not ok:
            return lab
    return "P2_PATH3_SUPPORTED"


def _fn_source(rel: str, name: str) -> str:
    src = (ROOT / rel).read_text()
    return next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in (SPEC, DESIGN, PANEL, CONFIG)}
    checks["panel_hash"] = sha(ROOT / PANEL) == c["panel_byte_sha256"] == "97de2edb21a2d8265a53ef81b74a184992116f8c727e58d88c79099464524d4c"
    checks["anchors_and_helpers_unchanged"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    parts, dlg, order = own_partitions()
    Pn = json.loads((ROOT / PANEL).read_text())
    checks["parent_bytes"] = sha(ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json") == c["fixed100_parent_byte_sha256"]
    checks["partitions_independent"] = (
        set(parts["PATH2_12"]) < set(parts["DEV24"]) < set(parts["FULL100"]) and [len(parts[k]) for k in parts] == [100, 24, 12, 76]
        and [len({dlg[u] for u in parts[k]}) for k in parts] == [20, 20, 9, 20] and len(set(parts["FULL100"])) == 100
        and all(fp({"partition": k, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in v]}) == c["partition_sha256"][k] for k, v in parts.items())
        and all(Pn["partitions"][k]["ids"] == v for k, v in parts.items()))
    checks["execution_order"] = order == Pn["execution_order"] and fp(order) == Pn["execution_order_sha256"] and order[:12] == c["PATH2_barrier"]["first_ids"]
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and plan["ids"] == parts["FULL100"] and plan["execution_order"] == order \
        and plan["partitions"] == parts
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    part = tokenizer_partition(tok)
    E, M = set(part["embedded_ids"]), set(part["matrix_ids"])
    t1 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())["rows"]}
    a4 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())["rows"]}
    bad = []
    for x in plan["rows"]:
        s = x["runtime"]
        u = s["utterance_id"]
        cls = ["E" if t in E else ("M" if t in M else "O") for t in s["y_A"]]
        okr = set(s) == {"utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "classes"}
        okr &= s["classes"] == cls and s["y_A"] == t1[u]["y_A"] and s["y_A_valid_mask"] == t1[u]["y_A_valid_mask"] and s["audio_full_sha256"] == t1[u]["audio_full_sha256"]
        okr &= x["audit"]["B0"]["tokens"] == t1[u]["y_B"] and x["analysis"]["y_B_text"] == t1[u]["y_B_text"]
        if u in set(parts["DEV24"]):
            okr &= a4[u]["classes"] == cls and a4[u]["y_A_valid_mask"] == s["y_A_valid_mask"] and x["audit"]["A4_sealed"]["lang_id"] == a4[u]["a3_lang_id"]
        if not okr:
            bad.append(u)
    checks["teachers_classes_masks"] = not bad
    rsrc = (ROOT / RUNNER).read_text()
    ctrl = _fn_source(RUNNER, "online_g1")
    run_code = _fn_source(RUNNER, "cmd_run")
    checks["controller_input_allowlist"] = not any(t in ctrl for t in AUDIT_TOKENS) and "def online_g1(b_t0, b_a4, g_t0, g_a4, enc, owner_row: str, counters: dict, eos: int)" in rsrc
    checks["controller_unchanged_semantics"] = (ctrl.count("lockstep_detect(") == 1 and ctrl.count("consensus(") == 1 and "g1_forced(dec)" in ctrl
                                                and "owned_clamp(b_a4" in ctrl and "owned_clamp(b_t0" not in ctrl and ctrl.count("rollout(") == 2
                                                and ctrl.count("score_branch(") == 1 and "EOS-first branch (frozen INVALID)" in ctrl)
    checks["no_G2"] = "g2_forced" not in rsrc and "G2_replay" not in rsrc and "handoff" not in ctrl
    checks["two_instances_no_alias"] = "b_t0, b_a4 = load_whisper(cfg_m), load_whisper(cfg_m)" in run_code and "data_ptr()" in run_code
    gsrc = (ROOT / "src/csasr/inference_cf/consensus_guard.py").read_text()
    checks["no_cross_model_cache"] = (gsrc.count("cached.Branch(") == 3 and "past_key_values" not in gsrc and ".cache =" not in gsrc
                                      and "resident state != declared owner before detector" in gsrc and "resident state != declared owner before rollout" in gsrc)
    i_bar = run_code.index("PATH2 replay barrier failed (remaining 88 not entered)")
    i_rem = run_code.index("for i, uid in enumerate(order[12:], start=12)")
    checks["barrier_before_remaining88"] = (i_bar < i_rem and run_code.index("for i, uid in enumerate(order[:12])") < i_bar
                                            and "replay_mismatches(row, p2row)" in run_code and run_code.count("online_g1(") == 2)
    rm = _fn_source(RUNNER, "replay_mismatches")
    checks["replay_exact_no_tolerance"] = "isclose" not in rm and "atol" not in rm and "rtol" not in rm and "!=" in rm
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path3_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path3_analyze|inference_cf_p2path3\b|consensus_guard|branch_adjudication|path_decode)",
                                              asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path3.py").exists()
    fsrc = (ROOT / "experiments/inference_cf_p2path3_analyze.py").read_text()
    prim_src = fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
    checks["reference_barrier"] = ("load_references" not in prim_src and "PermissionError" in fsrc and 'committed(SEAL)' in fsrc
                                   and "committed(PRIMARY_AUDIT)" in fsrc and "load_references" not in rsrc)
    TH = json.loads((ROOT / "configs/inference_cf/p2_tta_funnel.json").read_text())["TTA1"]["thresholds"]
    checks["safety_thresholds_TTA1"] = all(c["safety"][k] == TH[k] for k in ("max_MER_increase", "max_ZH_CER_increase", "min_matrix_ZH_retention",
                                                                            "min_embedded_EN_retention", "max_outside_POI_harm_rate", "max_POI_corruption_rate",
                                                                            "max_additional_caps", "max_new_severe_truncations", "float_guard"))
    # own rule truth tables (synthetic)
    r = lambda Z, POI, mg=0.3, ma=0.3, r76=(5, 4, 3, 2), d76=("a", "b", "c", "d"): own_rules(Z, POI, mg, ma, list(r76), list(d76), c)
    tt = [r((10, 20, 15), (100, 80, 85))["rescue"], not r((10, 21, 16), (100, 80, 85))["rescue"], r((10, 21, 15), (100, 80, 85))["rescue"],
          r((10, 9, 8), (100, 80, 85))["rescue"], not r((10, 9, 9), (100, 80, 85))["rescue"],
          r((10, 20, 15), (100, 80, 85))["retention_pass"], not r((10, 20, 15), (100, 80, 86))["retention_pass"],
          r((10, 20, 15), (100, 101, 99))["retention_pass"], not r((10, 20, 15), (100, 101, 100))["retention_pass"],
          not r((10, 20, 15), (100, 80, 85), mg=0.306)["useful"], r((10, 20, 15), (100, 80, 85), mg=0.305)["useful"],
          not r((10, 20, 15), (100, 80, 85), r76=(20,), d76=("a",))["breadth"], not r((10, 20, 15), (100, 80, 85), r76=(2, 2, 2), d76=("a", "a", "a"))["breadth"],
          r((10, 20, 15), (100, 80, 85), r76=(1, 1, 1, 1), d76=("a", "b", "c", "d"))["breadth"],
          not r((10, 20, 15), (100, 80, 85), r76=(7, 3, 3), d76=("a", "b", "c"))["breadth"], r((10, 20, 15), (100, 80, 85), r76=(6, 3, 3), d76=("a", "b", "c"))["breadth"],
          not r((10, 20, 15), (100, 80, 85), r76=(2, 1, 1, -4), d76=("a", "b", "c", "d"))["breadth"],
          not r((10, 20, 15), (100, 80, 85), r76=(0, 0), d76=("a", "b"))["breadth"]]
    labs = [own_label(*v) for v in ((False, True, True, True, True), (True, False, True, True, True), (True, True, False, False, True),
                                    (True, True, True, False, True), (True, True, True, True, False), (True, True, True, True, True))]
    checks["own_rule_truth_tables"] = all(tt) and labs == ["P2_PATH3_INVALID", "P2_PATH3_SEQUENCE_DAMAGE", "P2_PATH3_NO_USEFUL_GAIN",
                                                           "P2_PATH3_OVERCONSERVATIVE", "P2_PATH3_SIGNAL_CONCENTRATED", "P2_PATH3_SUPPORTED"]
    B = c["compute"]
    checks["compute_plan"] = (B["jobs_max"] == 1 and B["optimizer_steps"] == 200 and B["backwards_max_including_first_live_audit"] == 201
                              and B["hard_minutes"] == 30 and B["PATH2_first_barrier"] == 12 and B["remaining_after_barrier"] == 88 and B["no_G2"])
    v = "PASS_TO_P2_PATH3" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH3"
    return {"schema": "p2_path3_prerun_audit_v1", "verdict": v, "checks": checks, "failures": bad, "git_commit": _git("rev-parse", "HEAD")}


def cmd_post(args) -> dict:
    c = cfg()
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    parts, dlg, order = own_partitions()
    Pn = {r["utterance_id"]: r for r in json.loads((ROOT / PANEL).read_text())["rows"]}
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]
              and rt["elapsed_sec"] <= 1800}
    if args.phase == "full":
        seal = json.loads((BASE / "output_seal.json").read_text())
        checks["seal_committed_unchanged"] = committed("results/inference_cf/p2path3/output_seal.json") and \
            all(sha(ROOT / p) == h for p, h in seal["files"].items()) and seal["rows_sealed"] == 100
    files = sorted((run / "rows").glob("*.json"))
    checks["exactly_100_rows"] = [f.name for f in files] == [f"{i:03d}.json" for i in range(100)]
    rows = {}
    for f in files:
        r = json.loads(f.read_text())
        rows[r["identity"]] = r
    from transformers import WhisperProcessor
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    X = {x["runtime"]["utterance_id"]: x for x in plan["rows"]}
    bad, trig, t0w = [], {k: 0 for k in parts}, {k: 0 for k in parts}
    chg = {k: 0 for k in parts}
    for i, u in enumerate(parts["FULL100"]):
        r = rows[u]
        x = X[u]
        tag = "PATH2_12" if u in parts["PATH2_12"] else ("DEV24_OTHER" if u in parts["DEV24"] else "NOVEL76")
        okr = r["status"] == "ok" and r["canonical_index"] == i and r["partition"] == tag and order[r["execution_index"]] == u and r["dialogue_id"] == dlg[u]
        G0, G1 = r["G0"], r["G1"]
        okr &= G0["text"] == tok.decode(G0["tokens"], skip_special_tokens=True) and G1["text"] == tok.decode(G1["tokens"], skip_special_tokens=True)
        # A4 episode: own no-op rule
        s = x["runtime"]
        nE = sum(1 for cl, mk in zip(s["classes"], s["y_A_valid_mask"]) if mk and cl == "E")
        cA = r["auto_condition"]["cA"]
        okr &= cA == [50258, r["auto_condition"]["lang_id"], 50360, 50364]
        noop = cA == CB or nE == 0 or sum(s["y_A_valid_mask"]) == 0
        lg = r["A4_log"]
        okr &= lg["noop"] == noop and lg["n_E"] == nE and lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"])
        if noop:
            okr &= r["A4_state_hash"] == rt["theta0_ln_hash"] and lg["effective_changed_scalars"] == 0 and all(v == 0 for v in lg["losses"])
        if tag != "NOVEL76":
            a4row = json.loads((ROOT / Pn[u]["A4_DEV24_audit_only"]["row_path"]).read_text())
            okr &= G0["tokens"] == a4row["A4"]["tokens"] and G0["terminated"] == a4row["A4"]["terminated"] and G0["text"] == a4row["A4"]["text"]
            okr &= r["auto_condition"]["lang_id"] == a4row["auto_condition"]["lang_id"] and r["effective_equals_checkpoint_bf16"]
        else:
            okr &= r["reset_ok"] and r["A4_start_hash"] == r["reset_hash_after"] == rt["theta0_ln_hash"]
        # controller: own trigger / prefix / scores / consensus / G1
        det = r["detector"]
        pairs = det["argmax_pairs"]
        first = next((j for j, (p, q) in enumerate(pairs) if p != q), None)
        okr &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        okr &= det["trigger"] == (first is not None) and det["prefix"] == [p for p, q in pairs[:len(det["prefix"])]]
        if first is None:
            okr &= r["decision"] is None and G1["tokens"] == G0["tokens"] and G1["terminated"] == G0["terminated"] and det["prefix"] == G0["tokens"]
            okr &= pairs[-1][0] == pairs[-1][1] and (det["terminated"] != "eos" or pairs[-1][0] == EOS)
        else:
            d = r["decision"]
            okr &= first == d["k"] == len(det["prefix"]) == len(d["prefix"]) and d["prefix"] == det["prefix"] == G0["tokens"][:first]
            okr &= d["b0"]["tokens"][0] == pairs[first][0] and d["b4"]["tokens"][0] == pairs[first][1] and d["b0"]["H_eff"] >= 1 and d["b4"]["H_eff"] >= 1
            okr &= d["b4"]["tokens"] == G0["tokens"][first:first + len(d["b4"]["tokens"])] and EOS not in d["b0"]["tokens"] + d["b4"]["tokens"]
            S = {k_: sum(v["logprobs"]) / len(v["logprobs"]) for k_, v in d["scores"].items()}
            okr &= all(S[k_] == d["scores"][k_]["score"] for k_ in S) and all(len(d["scores"][f"{mm}_{bb}"]["logprobs"]) == d[bb]["H_eff"]
                                                                             for mm in ("theta0", "A4") for bb in ("b0", "b4"))
            cons = {"b0": 0.5 * S["theta0_b0"] + 0.5 * S["A4_b0"], "b4": 0.5 * S["theta0_b4"] + 0.5 * S["A4_b4"]}
            win = "theta0" if cons["b0"] - cons["b4"] >= -G else "A4"
            okr &= win == d["winner"] and cons["b0"] - cons["b4"] == d["margin"]
            wtok = (d["b0"] if win == "theta0" else d["b4"])["tokens"][0]
            okr &= G1["forced"] == [wtok] and [f["token"] for f in G1["trace"]["forced"]] == [wtok] and G1["trace"]["prefix_ok"] and G1["state_locked"]
            okr &= G1["decision_hash"] == d["decision_hash"] == "sha256:" + hashlib.sha256(json.dumps(
                {k_: d[k_] for k_ in ("k", "prefix", "b0", "b4", "scores", "winner", "margin")}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            g1s = st(G1["tokens"], G1["terminated"])
            okr &= g1s[:first] == d["prefix"] and g1s[first] == wtok
            okr &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"] for v in r["score_paths"].values())
            okr &= r["score_paths"]["A4_b0"]["owner"]["state_hash"] == r["A4_state_hash"] and r["score_paths"]["theta0_b0"]["owner"]["state_hash"] == rt["theta0_ln_hash"]
            if win == "A4":
                okr &= G1["tokens"] == G0["tokens"] and G1["terminated"] == G0["terminated"]
            for k in parts:
                if u in parts[k]:
                    trig[k] += 1
                    t0w[k] += win == "theta0"
        for k in parts:
            if u in parts[k]:
                chg[k] += st(G1["tokens"], G1["terminated"]) != st(G0["tokens"], G0["terminated"])
        if tag == "PATH2_12":
            p2 = json.loads((ROOT / Pn[u]["PATH2_replay_audit_only"]["path"]).read_text())
            okr &= r["replay_exact"] and r["replay_mismatches"] == []
            okr &= all(json.dumps(r[f], sort_keys=True) == json.dumps(p2[f], sort_keys=True) for f in ("A4_state_hash", "detector", "decision")) and \
                r["G0"]["tokens"] == p2["G0"]["tokens"] and r["G0"]["terminated"] == p2["G0"]["terminated"]
            okr &= G1["tokens"] == p2["G1"]["tokens"] and G1["terminated"] == p2["G1"]["terminated"] and r["theta0_equals_B0"]
        if not okr:
            bad.append(u)
    checks["rows_episodes_controller_replay_recomputed"] = not bad
    la = rows[order[0]]["live_audit"]
    checks["live_audit_2pct"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    n = trig["FULL100"]
    checks["counts"] = (cnt["triggers"] == n and cnt["rollouts"] == 2 * n and cnt["score_paths"] == 4 * n and cnt["G1_decodes"] == n and cnt["G2_decodes"] == 0
                        and cnt["A4"]["optimizer_steps"] == 200 and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= 201
                        and cnt["G0_decodes"] == 100 and cnt["theta0_decodes"] == 12 and cnt["rows_entered_after_barrier"] == 88
                        and rt["barrier"]["pass"] and rt["barrier"]["exact"] == 12)
    sm = prim.get("summaries", {})
    checks["summaries_agree"] = all(sm[k]["triggers"] == trig[k] and sm[k]["theta0_winners"] == t0w[k] and sm[k]["G1_changed_vs_A4"] == chg[k] for k in parts)
    checks["primary_valid_agrees"] = prim["valid"] == all(checks.values())
    out = {"triggers": trig, "theta0_winners": t0w, "changed": chg}
    lab_out = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        refs = load_references()
        ids = parts["FULL100"]
        R = {u: refs[u]["reference"] for u in ids}
        D = {u: refs[u]["dialogue_id"] for u in ids}
        checks["dialogues"] = all(D[u] == dlg[u] for u in ids)
        t1 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())["rows"]}
        H = {u: {"B0_FORCED": t1[u]["y_B_text"], "B0_AUTO": t1[u]["y_A_text"],
                 "A2": json.loads((ROOT / Pn[u]["A2"]["path"]).read_text())["objectives"]["A2"]["text"],
                 "A4": rows[u]["G0"]["text"], "G1": rows[u]["G1"]["text"]} for u in ids}
        SY = ("B0_FORCED", "B0_AUTO", "A2", "A4", "G1")
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
        tr = tau.transitions([R[u] for u in ids], [H[u]["B0_FORCED"] for u in ids], [H[u]["G1"] for u in ids])
        term = {u: (t1[u]["y_B_terminated"], rows[u]["G1"]["terminated"]) for u in ids}
        cap_delta = sum(b == "cap" for _, b in term.values()) - sum(a == "cap" for a, _ in term.values())
        sev = sum(severe(len(t1[u]["y_B"]), len(rows[u]["G1"]["tokens"]), rows[u]["G1"]["terminated"]) for u in ids)
        TH = c["safety"]
        safe = {"MER": F["G1"]["mer"] - F["B0_FORCED"]["mer"] <= TH["max_MER_increase"] + G,
                "ZH_CER": F["G1"]["zh_cer"] - F["B0_FORCED"]["zh_cer"] <= TH["max_ZH_CER_increase"] + G,
                "ZH_ret": tr["zr"] is None or tr["zr"] >= TH["min_matrix_ZH_retention"] - G,
                "EN_ret": tr["er"] is None or tr["er"] >= TH["min_embedded_EN_retention"] - G,
                "outside": tr["ohr"] is None or tr["ohr"] <= TH["max_outside_POI_harm_rate"] + G,
                "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= TH["max_POI_corruption_rate"] + G,
                "caps": cap_delta <= TH["max_additional_caps"], "severe": sev <= TH["max_new_severe_truncations"]}
        sp = sec["safety_point"]
        checks["safety_points_agree"] = (tr["zr"] == sp["zh_retention"] and tr["er"] == sp["en_retention"] and tr["ohr"] == sp["outside_harm_rate"]
                                         and tr["corr_rate"] == sp["poi_corruption_rate"] and cap_delta == sp["cap_delta"] and sev == sp["new_severe_truncations"]
                                         and all(safe.values()) == sec["ABSOLUTE_SAFETY_PASS"])
        r76 = [C[u]["A4"][4] - C[u]["G1"][4] for u in parts["NOVEL76"]]
        d76 = [D[u] for u in parts["NOVEL76"]]
        rules = own_rules((F["B0_FORCED"]["zh"], F["A4"]["zh"], F["G1"]["zh"]), (F["B0_FORCED"]["poi"], F["A4"]["poi"], F["G1"]["poi"]),
                          F["G1"]["mer"], F["A4"]["mer"], r76, d76, c)
        ar, be, br = sec["aggregate_rescue"], sec["benefit"], sec["breadth"]
        checks["rescue_benefit_breadth_agree"] = (rules["rescue"] == ar["pass"] and rules["X"] == ar["X_ZH"] and rules["required"] == ar["required_R_ZH"]
                                                  and rules["I_A4"] == be["I_A4"] and rules["I_G1"] == be["I_G1"]
                                                  and rules["retention_pass"] == be["BENEFIT_RETENTION_PASS"] and rules["useful"] == be["USEFUL_GAIN_PASS"]
                                                  and rules["breadth"] == br["BREADTH_PASS"] and abs(rules["C_utt"] - br["C_utt"]) <= 1e-12
                                                  and abs(rules["C_dlg"] - br["C_dlg"]) <= 1e-12 and sum(r76) == br["R_net_76"])
        keys = sorted(set(d76))
        lv = [sum(x for x, dd in zip(r76, d76) if dd != k) for k in keys]
        lo = sec["lodo"]
        checks["lodo_agrees"] = len(keys) == 20 and min(lv) == lo["min"] and sum(v > 0 for v in lv) == lo["count_positive"] and \
            (sum(v > 0 for v in lv) >= 18) == lo["LODO_ROBUST"]
        Dl = [D[u] for u in ids]
        kk = sorted(set(Dl))
        grp = {k: [j for j, d in enumerate(Dl) if d == k] for k in kk}
        rng = np.random.default_rng(240924)
        drs = [sum((grp[kk[j]] for j in rng.integers(0, 20, 20)), []) for _ in range(2000)]
        CA = {s_: np.array([C[u][s_] for u in ids], dtype=float) for s_ in SY}
        bd = []
        for base in ("A4", "B0_FORCED"):
            for nme, (a_, b_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items():
                vals = [CA["G1"][ix, a_].sum() / CA["G1"][ix, b_].sum() - CA[base][ix, a_].sum() / CA[base][ix, b_].sum()
                        for ix in drs if CA["G1"][ix, b_].sum() and CA[base][ix, b_].sum()]
                ref_ = sec["bootstrap"][f"G1_minus_{base}"][nme]["ci95"]
                bd += [abs(np.quantile(vals, .025) - ref_[0]), abs(np.quantile(vals, .975) - ref_[1])]
            vals = [CA["G1"][ix, 0].sum() - CA[base][ix, 0].sum() for ix in drs]
            ref_ = sec["bootstrap"][f"G1_minus_{base}"]["poi_errors"]["ci95"]
            bd += [abs(np.quantile(vals, .025) - ref_[0]), abs(np.quantile(vals, .975) - ref_[1])]
        checks["bootstrap_agrees"] = max(bd) <= 1e-9
        valid = bool(prim["valid"]) and all(v for k, v in checks.items() if k not in ("label_agrees",))
        lab_out = own_label(valid, all(safe.values()), rules["useful"], rules["retention_pass"], rules["breadth"])
        checks["label_agrees"] = lab_out == sec["label"]
        out.update(metrics=mine, safety=safe, rules=rules, cap_delta=cap_delta, severe=sev, lodo=lv)
    v = "P2_PATH3_AUDIT: PASS" if all(checks.values()) else "P2_PATH3_AUDIT: BLOCK"
    return {"schema": "p2_path3_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab_out, "checks": checks, "failures": bad,
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
