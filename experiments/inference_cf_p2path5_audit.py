#!/usr/bin/env python
"""Independent P2-PATH5 auditor. Does NOT import the PATH5 runner/analysis, or csasr.inference_cf.consensus_guard /
branch_adjudication / path_decode / eos_boundary. Own FULL300/FIXED100/NEW200 reconstruction and hashing, own teacher/mask
re-derivation (TTA0 auditor ``own_valid``), own trigger/prefix from argmax pairs, own dispatch classification, content-G1
arithmetic, abstention identity, fixed100 barrier vs sealed artifacts and the OPP0 derived target, own opportunity gate,
canonical counts/transitions (TTA0 auditor ``counts``/``transitions``), own safety/retention/breadth/LODO/bootstrap/label.

prerun -> PASS_TO_P2_PATH5 | BLOCK_BEFORE_P2_PATH5
post   -> P2_PATH5_AUDIT: PASS | BLOCK  (--phase primary before references; --phase full after secondary)
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

FREEZE = "d1d057d"
CONFIG = "configs/inference_cf/p2_path5.json"
PANEL = "docs/inference_cf/P2_PATH5_PANEL.json"
FROZEN = ("docs/inference_cf/P2_PATH5_SPEC.md", "docs/inference_cf/P2_PATH5_CLAUDE_DESIGN.md", PANEL, CONFIG,
          "results/inference_cf/p2opp0/derived_g1a.json", "experiments/inference_cf_p2opp0_abstain.py", "experiments/inference_cf_p2path5_panel.py")
BASE = ROOT / "results/inference_cf/p2path5"
PLAN_REL = "results/inference_cf/p2path5/plan_sealed.json"
RUNNER = "experiments/inference_cf_p2path5.py"
ANALYZE = "experiments/inference_cf_p2path5_analyze.py"
EOS = 50257
CB = [50258, 50260, 50360, 50364]
G = 1e-12
AUDIT_TOKENS = ('["audit"]', "au[", '["analysis"]', "partition", "FIXED100", "NEW200", "G1A_target", "PATH4", "AUTO", "load_references",
                "derived", "audit_only")


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
    full_p = json.loads((ROOT / "results/inference_cf/p2_A_r1_L16/panel.json").read_text())
    fx = json.loads((ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json").read_text())
    full = [r["utterance_id"] for r in full_p["rows"]]
    fixed = [r["utterance_id"] for r in fx["rows"]]
    Pn = json.loads((ROOT / PANEL).read_text())
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in Pn["rows"]}
    parts = {"FULL300": full, "FIXED100": fixed, "NEW200": [u for u in full if u not in set(fixed)]}
    return parts, dlg, Pn, fx


# ---- own frozen rules -------------------------------------------------------------------------------------------

def own_gate(changed_rows: list, dlg: dict) -> bool:
    return len(changed_rows) >= 3 and len({dlg[u] for u in changed_rows}) >= 3


def own_retention(POI, PIER, MER):
    """POI=(B0,A2,G), PIER/MER=(A2,G)."""
    IA, IG = POI[0] - POI[1], POI[0] - POI[2]
    strong = (IG / IA >= 0.90 - G) if IA > 0 else POI[2] <= POI[1]
    return strong and PIER[1] - PIER[0] <= 0.005 + G and MER[1] - MER[0] <= 0.005 + G


def own_breadth(r: dict, dlg: dict):
    pos = {u: max(x, 0) for u, x in r.items()}
    Rp = sum(pos.values())
    Pd = {}
    for u, x in pos.items():
        Pd[dlg[u]] = Pd.get(dlg[u], 0) + x
    cu = max(pos.values()) / Rp if Rp > 0 else 1.0
    cd = max(Pd.values()) / Rp if Rp > 0 else 1.0
    ok = sum(x >= 1 for x in r.values()) >= 3 and sum(v > 0 for v in Pd.values()) >= 3 and cu <= 0.5 + G and cd <= 0.6 + G
    return ok, cu, cd, sum(r.values())


def own_label(valid, gate, safe, r_net, ret, brd):
    for ok, lab in ((valid, "P2_PATH5_INVALID"), (gate, "P2_PATH5_OPPORTUNITY_SPARSE"), (safe, "P2_PATH5_SEQUENCE_DAMAGE"),
                    (r_net > 0, "P2_PATH5_SAFE_NO_ADDED_VALUE"), (ret, "P2_PATH5_OVERCONSERVATIVE"), (brd, "P2_PATH5_SIGNAL_CONCENTRATED")):
        if not ok:
            return lab
    return "P2_PATH5_G1A_SUPPORTED"


def _fn_source(rel: str, name: str) -> str:
    src = (ROOT / rel).read_text()
    return next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in FROZEN}
    checks["freeze_pushed"] = subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, "origin/cs-asr-steer-inf"], cwd=ROOT).returncode == 0
    checks["panel_hash"] = sha(ROOT / PANEL) == c["panel_byte_sha256"]
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    parts, dlg, Pn, fx = own_population()
    checks["population_independent"] = (
        fx["parent_panel"] == "results/inference_cf/p2_A_r1_L16/panel.json" and fx["parent_panel_sha256"] == sha(ROOT / fx["parent_panel"])
        and set(parts["FIXED100"]) < set(parts["FULL300"]) and [len(v) for v in parts.values()] == [300, 100, 200]
        and all(len({dlg[u] for u in v}) == 20 for v in parts.values())
        and all(fp({"partition": k, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in v]}) == c["partition_sha256"][k] for k, v in parts.items())
        and fp(parts["FIXED100"] + parts["NEW200"]) == c["execution_order_sha256"]
        and all(r["dialogue_id"] == dlg[r["utterance_id"]] for r in fx["rows"]))
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and plan["partitions"] == parts
    from transformers import WhisperProcessor
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    sup, beg = c["suppression"]["suppress"], c["suppression"]["begin"]
    der = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2opp0/derived_g1a.json").read_text())["rows"]}
    bad = []
    for i, x in enumerate(plan["rows"]):
        s = x["runtime"]
        u = s["utterance_id"]
        auto = json.loads((ROOT / f"results/inference_cf/p2_A_r1_L16/rows/{i:03d}.json").read_text())["systems"]["B0_AUTO"]
        okr = set(s) == {"utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask"}
        okr &= u == parts["FULL300"][i] and s["y_A"] == auto["tokens"] and all(t < EOS for t in s["y_A"])
        okr &= s["y_A_valid_mask"] == tau.own_valid(s["y_A"], sup, beg, tok)
        if u in set(parts["FIXED100"]):
            a = x["audit"]
            okr &= a["G1A_target"]["tokens"] == der[u]["G1A"]["tokens"] and a["G1A_target"]["terminated"] == der[u]["G1A"]["terminated"]
        if not okr:
            bad.append(u)
    checks["teachers_masks_targets"] = not bad
    t1 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())["rows"]}
    checks["fixed100_teachers_equal_TTA1"] = all(x["runtime"]["y_A"] == t1[x["runtime"]["utterance_id"]]["y_A"]
                                                 and x["runtime"]["y_A_valid_mask"] == t1[x["runtime"]["utterance_id"]]["y_A_valid_mask"]
                                                 for x in plan["rows"] if x["runtime"]["utterance_id"] in t1)
    rsrc = (ROOT / RUNNER).read_text()
    ctrl = _fn_source(RUNNER, "online_g1a")
    run_code = _fn_source(RUNNER, "cmd_run")
    checks["controller_input_allowlist"] = not any(t in ctrl for t in AUDIT_TOKENS) and \
        "def online_g1a(b_t0, b_a2, g_t0, g_a2, enc, owner_row: str, counters: dict, eos: int, a2_free: dict)" in rsrc
    checks["controller_semantics"] = (ctrl.count("lockstep_detect(") == 1 and ctrl.count("path3.online_g1(") == 1 and "action_mode(" in ctrl
                                      and '"EOS_ABSTAIN"' in ctrl and "score_boundary" not in ctrl and "execute_boundary" not in ctrl
                                      and "rollout(" not in ctrl.replace("rollouts", "") and "owned_clamp" not in ctrl and "clamp_decode" not in ctrl
                                      and "a2_free[\"tokens\"]" in ctrl)
    checks["original_G1_pinned"] = sha(ROOT / "experiments/inference_cf_p2path3.py") == c["source_sha256"]["experiments/inference_cf_p2path3.py"]
    checks["no_G2_A4_boundary_arbitration"] = all(t not in rsrc for t in ("g2_forced", "adapt_a4", "detect_language", "score_boundary", "execute_boundary", "script_safe_tta"))
    checks["A2_exact_path"] = 'run_objective(b_a2, g_a2, "A2"' in run_code and "np.array_equal" in run_code and "live_objective_check" in run_code
    checks["two_instances_no_alias"] = "b_t0, b_a2 = load_whisper(cfg_m), load_whisper(cfg_m)" in run_code and "data_ptr()" in run_code
    i_b1 = run_code.index("FIXED100 A2 reconstruction barrier failed (NEW200 not entered)")
    i_b2 = run_code.index("FIXED100 G1A barrier failed (NEW200 not entered)")
    i_new = run_code.index("for i, uid in enumerate(order[nfix:], start=nfix)")
    checks["fixed100_barrier_before_NEW200"] = i_b1 < i_b2 < i_new and "barrier_mismatches(row, p4" in run_code
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path5_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path5_analyze|inference_cf_p2path5\b|consensus_guard|branch_adjudication|path_decode|eos_boundary)",
                                              asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path5.py").exists()
    fsrc = (ROOT / ANALYZE).read_text()
    checks["reference_barrier"] = ("load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
                                   and "opportunity gate closed" in fsrc and "committed(SEAL)" in fsrc and "committed(PRIMARY_AUDIT)" in fsrc
                                   and "load_references" not in rsrc)
    TH = json.loads((ROOT / "configs/inference_cf/p2_tta_funnel.json").read_text())["TTA1"]["thresholds"]
    checks["safety_thresholds_TTA1"] = all(c["safety"][k] == TH[k] for k in c["safety"])
    D = {f"u{i}": f"d{i % 5}" for i in range(10)}
    tt = [not own_gate(["u0", "u5"], D), not own_gate(["u0", "u1"], D), own_gate(["u0", "u1", "u2"], D),
          not own_gate(["u0", "u5", "u1"], {**D, "u1": "d0"}),
          own_retention((100, 80, 82), (0.5, 0.5), (0.3, 0.3)), not own_retention((100, 80, 83), (0.5, 0.5), (0.3, 0.3)),
          own_retention((100, 101, 101), (0.5, 0.5), (0.3, 0.3)), not own_retention((100, 101, 102), (0.5, 0.5), (0.3, 0.3)),
          own_retention((100, 80, 80), (0.5, 0.505), (0.3, 0.3)), not own_retention((100, 80, 80), (0.5, 0.5051), (0.3, 0.3)),
          not own_retention((100, 80, 80), (0.5, 0.5), (0.3, 0.3051)),
          own_breadth({"u0": 2, "u1": 1, "u2": 1}, D)[0], not own_breadth({"u0": 20}, D)[0], not own_breadth({"u0": 2, "u5": 2, "u1": 0}, D)[0],
          not own_breadth({"u0": 3, "u1": 1, "u2": 1}, D)[0], own_breadth({"u0": 0}, D)[1:3] == (1.0, 1.0)]
    labs = {k: own_label(*k[:3], 1 if k[3] else 0, *k[4:]) for k in itertools.product((False, True), repeat=6)}
    exp = lambda v, g, s, rn, t, b: ("P2_PATH5_INVALID" if not v else "P2_PATH5_OPPORTUNITY_SPARSE" if not g else "P2_PATH5_SEQUENCE_DAMAGE" if not s
                                     else "P2_PATH5_SAFE_NO_ADDED_VALUE" if not rn else "P2_PATH5_OVERCONSERVATIVE" if not t
                                     else "P2_PATH5_SIGNAL_CONCENTRATED" if not b else "P2_PATH5_G1A_SUPPORTED")
    checks["own_rule_truth_tables"] = all(tt) and all(labs[k] == exp(*k) for k in labs) and len(labs) == 64
    B = c["compute"]
    checks["compute_plan"] = B["jobs_max"] == 1 and B["rows"] == 300 and B["optimizer_steps"] == 600 and B["backwards_max_including_live_check"] == 601 \
        and B["hard_minutes"] == 180
    sb = (ROOT / "slurm/inference_cf_p2path5.sbatch").read_text()
    checks["sbatch"] = "--time=03:00:00" in sb and "PASS_TO_P2_PATH5" in sb and "inference_cf_p2path5.py run" in sb
    v = "PASS_TO_P2_PATH5" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH5"
    return {"schema": "p2_path5_prerun_audit_v1", "verdict": v, "checks": checks, "failures": bad, "git_commit": _git("rev-parse", "HEAD")}


def cmd_post(args) -> dict:
    c = cfg()
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    parts, dlg, Pn, fx = own_population()
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]
              and rt["elapsed_sec"] <= 3 * 3600 and rt["barrier"]["pass"] and rt["barrier"]["exact"] == 100,
              "seal_committed_unchanged": committed("results/inference_cf/p2path5/output_seal.json")
              and all(sha(ROOT / p) == h for p, h in json.loads((BASE / "output_seal.json").read_text())["files"].items())}
    files = sorted((run / "rows").glob("*.json"))
    checks["exactly_300_rows"] = [f.name for f in files] == [f"{i:03d}.json" for i in range(300)]
    rows = {}
    for f in files:
        r = json.loads(f.read_text())
        rows[r["identity"]] = r
    der = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2opp0/derived_g1a.json").read_text())["rows"]}
    P4panel = {r["utterance_id"]: r for r in json.loads((ROOT / "docs/inference_cf/P2_PATH4_PANEL.json").read_text())["rows"]}
    from transformers import WhisperProcessor
    tk = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    bad = []
    nc = ne = 0
    changed = {"NEW200": [], "FIXED100": []}
    for i, u in enumerate(parts["FULL300"]):
        r = rows[u]
        part = "FIXED100" if u in set(parts["FIXED100"]) else "NEW200"
        A, Gr, th = r["A2_free"], r["G1"], r["theta0"]
        okr = r["status"] == "ok" and r["full_index"] == i and r["partition"] == part and r["dialogue_id"] == dlg[u]
        okr &= Gr["text"] == tk.decode(Gr["tokens"], skip_special_tokens=True) and A["text"] == tk.decode(A["tokens"], skip_special_tokens=True)
        lg = r["A2_log"]
        okr &= lg["steps"] == 2 and lg["loss_evaluations"] == 3 and all(lg["finite"]) and lg["start_hash"] == lg["end_hash"] == rt["theta0_ln_hash"]
        if part == "FIXED100":
            q = P4panel[u]
            b0 = json.loads((ROOT / q["B0_FORCED"]["path"]).read_text())["systems"]["S0"]
            a2 = json.loads((ROOT / q["A2"]["path"]).read_text())["objectives"]["A2"]
            okr &= th["tokens"] == b0["tokens"] and th["terminated"] == b0["terminated"] and A["tokens"] == a2["tokens"] and A["terminated"] == a2["terminated"]
            okr &= Gr["tokens"] == der[u]["G1A"]["tokens"] and Gr["terminated"] == der[u]["G1A"]["terminated"]
            okr &= r["masters_equal_checkpoint_fp32"] and r["effective_equals_checkpoint_bf16"] and r["barrier_exact"]
            p4 = json.loads((ROOT / f"results/inference_cf/p2path4/run1/rows/{parts['FIXED100'].index(u):03d}.json").read_text())
            okr &= json.dumps(r["detector"], sort_keys=True) == json.dumps(p4["detector"], sort_keys=True)
            if p4["mode"] == "CONTENT_G1":
                okr &= json.dumps(r["decision"], sort_keys=True) == json.dumps(p4["decision"], sort_keys=True) and Gr["tokens"] == p4["G1"]["tokens"]
        det = r["detector"]
        pairs = det["argmax_pairs"]
        first = next((j for j, (p, q_) in enumerate(pairs) if p != q_), None)
        okr &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        okr &= det["trigger"] == (first is not None) == (st(A["tokens"], A["terminated"]) != st(th["tokens"], th["terminated"]))
        okr &= det["prefix"] == A["tokens"][:len(det["prefix"])]
        if first is None:
            okr &= r["mode"] == "NO_TRIGGER" and r["decision"] is None and Gr["tokens"] == A["tokens"] and Gr["terminated"] == A["terminated"]
        else:
            c0, c2 = pairs[first]
            okr &= first == det["k"] == len(det["prefix"])
            if (c0 == EOS) != (c2 == EOS):
                ne += 1
                okr &= r["mode"] == "EOS_ABSTAIN" and r["decision"] is None and Gr["tokens"] == A["tokens"] and Gr["terminated"] == A["terminated"]
                okr &= r["abstain"]["orientation"] == ("theta0_EOS" if c0 == EOS else "A2_EOS") and "score_paths" not in r
            else:
                nc += 1
                d = r["decision"]
                S = {k_: sum(v["logprobs"]) / len(v["logprobs"]) for k_, v in d["scores"].items()}
                okr &= all(S[k_] == d["scores"][k_]["score"] for k_ in S)
                cons = {"b0": 0.5 * S["theta0_b0"] + 0.5 * S["A4_b0"], "b4": 0.5 * S["theta0_b4"] + 0.5 * S["A4_b4"]}
                win = "theta0" if cons["b0"] - cons["b4"] >= -G else "A2"
                okr &= r["mode"] == "CONTENT_G1" and (d["winner"] == "theta0") == (win == "theta0") and cons["b0"] - cons["b4"] == d["margin"]
                okr &= d["b0"]["tokens"][0] == c0 and d["b4"]["tokens"][0] == c2 and d["b4"]["tokens"] == A["tokens"][first:first + len(d["b4"]["tokens"])]
                wtok = (d["b0"] if win == "theta0" else d["b4"])["tokens"][0]
                okr &= Gr["forced"] == [wtok] and [f_["token"] for f_ in Gr["trace"]["forced"]] == [wtok] and Gr["state_locked"]
                okr &= st(Gr["tokens"], Gr["terminated"])[:first + 1] == det["prefix"] + [wtok]
                okr &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"] for v in r["score_paths"].values())
                okr &= d["decision_hash"] == Gr["decision_hash"] == "sha256:" + hashlib.sha256(json.dumps(
                    {k_: d[k_] for k_ in ("k", "prefix", "b0", "b4", "scores", "winner", "margin")}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if win == "A2":
                    okr &= Gr["tokens"] == A["tokens"] and Gr["terminated"] == A["terminated"]
        ch = st(Gr["tokens"], Gr["terminated"]) != st(A["tokens"], A["terminated"])
        okr &= ch == r["G1A_changed_vs_A2"]
        if ch:
            changed[part].append(u)
        if not okr:
            bad.append(u)
    checks["rows_barrier_dispatch_controller_recomputed"] = not bad
    la = rows[parts["FIXED100"][0]]["live_audit"]
    checks["live_audit_2pct"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    checks["counts"] = (cnt["content_events"] == nc and cnt["eos_abstentions"] == ne and cnt["triggers"] == nc + ne and cnt["rollouts"] == 2 * nc
                        and cnt["score_paths"] == 4 * nc and cnt["G1_decodes"] == nc and cnt["boundary_scores"] == 0 and cnt["G2_decodes"] == 0
                        and cnt["A2"]["optimizer_steps"] == 600 and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= 601 and cnt["theta0_decodes"] == 300)
    gate_open = own_gate(changed["NEW200"], dlg)
    pg = prim["opportunity_gate"]
    sm = prim["summaries"]["NEW200"]
    checks["opportunity_gate_agrees"] = (pg["open"] == gate_open and pg["N_G1A_CHANGED"] == len(changed["NEW200"])
                                         and pg["N_CHANGED_DLG"] == len({dlg[u] for u in changed["NEW200"]}) and sm["changed_rows"] == changed["NEW200"])
    checks["primary_valid_agrees"] = prim["valid"] == all(checks.values())
    out = {"content": nc, "eos_abstain": ne, "changed": changed, "gate_open": gate_open}
    lab_out = None
    if args.phase == "primary" and not gate_open:
        lab_out = own_label(all(checks.values()), False, False, 0, False, False)
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        refs = load_references()
        new = parts["NEW200"]
        R = {u: refs[u]["reference"] for u in new}
        checks["dialogues"] = all(refs[u]["dialogue_id"] == dlg[u] for u in new)
        auto = {u: json.loads((ROOT / f"results/inference_cf/p2_A_r1_L16/rows/{parts['FULL300'].index(u):03d}.json").read_text())["systems"]["B0_AUTO"]
                for u in new}
        H = {u: {"B0_FORCED": rows[u]["theta0"]["text"], "B0_AUTO": auto[u]["text"], "A2": rows[u]["A2_free"]["text"], "G1A": rows[u]["G1"]["text"]} for u in new}
        SY = ("B0_FORCED", "B0_AUTO", "A2", "G1A")
        C = {u: {s_: tau.counts(R[u], H[u][s_]) for s_ in SY} for u in new}
        rate = lambda s_, a_, b_: sum(C[u][s_][a_] for u in new) / sum(C[u][s_][b_] for u in new)
        mine = {s_: {"zh": sum(C[u][s_][4] for u in new), "poi": sum(C[u][s_][0] for u in new), "mixed": sum(C[u][s_][2] for u in new),
                     "pier": rate(s_, 0, 1), "mer": rate(s_, 2, 3), "zh_cer": rate(s_, 4, 5), "en_wer": rate(s_, 6, 7)} for s_ in SY}
        N = sec["metrics"]["NEW200"]
        checks["metrics_agree"] = all(mine[s_][q] == N[s_][q] for s_ in SY for q in ("zh", "poi", "mixed")) and \
            max(abs(mine[s_][q] - N[s_][q]) for s_ in SY for q in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12
        tr = tau.transitions([R[u] for u in new], [H[u]["B0_FORCED"] for u in new], [H[u]["G1A"] for u in new])
        added = sum(rows[u]["G1"]["terminated"] == "cap" and rows[u]["theta0"]["terminated"] != "cap" for u in new)
        sev = sum(severe(len(rows[u]["theta0"]["tokens"]), len(rows[u]["G1"]["tokens"]), rows[u]["G1"]["terminated"]) for u in new)
        TH = c["safety"]
        safe = (mine["G1A"]["mer"] - mine["B0_FORCED"]["mer"] <= TH["max_MER_increase"] + G and mine["G1A"]["zh_cer"] - mine["B0_FORCED"]["zh_cer"] <= TH["max_ZH_CER_increase"] + G
                and (tr["zr"] is None or tr["zr"] >= TH["min_matrix_ZH_retention"] - G) and (tr["er"] is None or tr["er"] >= TH["min_embedded_EN_retention"] - G)
                and (tr["ohr"] is None or tr["ohr"] <= TH["max_outside_POI_harm_rate"] + G) and (tr["corr_rate"] is None or tr["corr_rate"] <= TH["max_POI_corruption_rate"] + G)
                and added <= TH["max_additional_caps"] and sev <= TH["max_new_severe_truncations"])
        sp = sec["safety_point"]
        checks["safety_agrees"] = (tr["zr"] == sp["zh_retention"] and tr["er"] == sp["en_retention"] and tr["ohr"] == sp["outside_harm_rate"]
                                   and tr["corr_rate"] == sp["poi_corruption_rate"] and added == sp["added_caps"] and sev == sp["new_severe_truncations"]
                                   and safe == sec["ABSOLUTE_SAFETY_PASS"])
        ret = own_retention((mine["B0_FORCED"]["poi"], mine["A2"]["poi"], mine["G1A"]["poi"]), (mine["A2"]["pier"], mine["G1A"]["pier"]),
                            (mine["A2"]["mer"], mine["G1A"]["mer"]))
        r = {u: C[u]["A2"][4] - C[u]["G1A"][4] for u in new}
        brd, cu, cd, rnet = own_breadth(r, dlg)
        checks["retention_breadth_agree"] = (ret == sec["retention"]["pass"] and brd == sec["breadth"]["pass"] and rnet == sec["breadth"]["R_net"]
                                             and abs(cu - sec["breadth"]["C_utt"]) <= 1e-12 and abs(cd - sec["breadth"]["C_dlg"]) <= 1e-12)
        keys = sorted(set(dlg[u] for u in new))
        lv = [sum(x for u, x in r.items() if dlg[u] != k) for k in keys]
        checks["lodo_agrees"] = min(lv) == sec["lodo"]["min"] and sum(v > 0 for v in lv) == sec["lodo"]["count_positive"]
        grp = {k: [j for j, u in enumerate(new) if dlg[u] == k] for k in keys}
        rng = np.random.default_rng(240924)
        drs = [sum((grp[keys[j]] for j in rng.integers(0, len(keys), len(keys))), []) for _ in range(2000)]
        CA = {s_: np.array([C[u][s_] for u in new], dtype=float) for s_ in SY}
        bd = []
        for base in ("A2", "B0_FORCED"):
            for nme, (a_, b_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5)}.items():
                vals = [CA["G1A"][ix, a_].sum() / CA["G1A"][ix, b_].sum() - CA[base][ix, a_].sum() / CA[base][ix, b_].sum()
                        for ix in drs if CA["G1A"][ix, b_].sum() and CA[base][ix, b_].sum()]
                ref_ = sec["bootstrap"][f"G1A_minus_{base}"][nme]["ci95"]
                bd += [abs(np.quantile(vals, .025) - ref_[0]), abs(np.quantile(vals, .975) - ref_[1])]
        checks["bootstrap_agrees"] = max(bd) <= 1e-9
        valid = bool(prim["valid"]) and all(v for k, v in checks.items() if k != "label_agrees")
        lab_out = own_label(valid, gate_open, safe, rnet, ret, brd)
        checks["label_agrees"] = lab_out == sec["label"]
        out.update(metrics=mine, safe=safe, retention=ret, breadth=brd, R_net=rnet, added_caps=added, severe=sev)
    v = "P2_PATH5_AUDIT: PASS" if all(checks.values()) else "P2_PATH5_AUDIT: BLOCK"
    return {"schema": "p2_path5_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab_out, "checks": checks, "failures": bad,
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
