#!/usr/bin/env python
"""Independent P2-SEL-E auditor. Does NOT import inference_cf_p2sel_e_analyze (or any analysis module);
decomposition, groups, causal joins, hypotheses, precedence, E_new, statistics and labels are recomputed
here with separate code. Reuses only independent audit helpers (canonical/file hash, bf16 unpack, logit
metrics, dialogue interval, bf16 hook emulation).

prerun -> PASS_TO_P2_SEL_E_E0 | BLOCK_BEFORE_P2_SEL_E_E0
e0     -> P2_SEL_E_AUDIT: PASS (E0)
pre_e1 -> P2_SEL_E_PRE_E1_AUDIT: PASS | BLOCK
e1     -> P2_SEL_E_AUDIT: PASS (E1)
final  -> P2_SEL_E_AUDIT: PASS
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

from experiments.inference_cf_p2dir_audit import bf16, canon, ci, fhash, git_blob_hash, metrics

FREEZE = "b693b8dc97316a7a61df594f0da093f9e9185753"
CONFIG = "configs/inference_cf/p2_sel_e.json"
FROZEN = ("docs/inference_cf/P2_SEL_E_SPEC.md", "docs/inference_cf/P2_SEL_E_CODEX_DESIGN.md", CONFIG,
          "tests/test_p2_sel_e_q_contract.py", "docs/inference_cf/P2_SEL_E_PRE_RUN_AUDIT.md",
          "results/inference_cf/p2sel_e/prerun_audit.json")
BASE = ROOT / "results/inference_cf/p2sel_e"
FORBIDDEN = ("target_ids", "competitor", "E_oracle", "timing_error", "stratum", "acoustic", "positions.json", "group")


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cfg():
    return json.loads((ROOT / CONFIG).read_text())


def runtime_surface_clean() -> dict:
    """E_new formula / E0 extraction / E1 pulse code never touch evaluator labels, oracle or references."""
    src = (ROOT / "experiments/inference_cf_p2sel_e.py").read_text()
    tree = ast.parse(src)
    res = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in ("e_new", "e0_utterance", "e1_utterance", "short_bounds",
                                                         "decompose", "lag_values"):
            seg = ast.get_source_segment(src, n)
            res[n.name] = not any(f in seg for f in FORBIDDEN)
    import experiments.inference_cf_p2sel_e as run
    inputs_ok = all(set(v) <= {"E", "E_lag1", "E_lag2", "E_short", "Q", "ell_local", "ell_null"} for v in run.E_NEW_INPUTS.values())
    return {"ok": len(res) == 6 and all(res.values()) and inputs_ok, "functions": res, "inputs_ok": inputs_ok}


def provider() -> dict:
    p = Path("/mnt/data/tungnx/whisper-large-v3/generation_config.json")
    lt = json.loads(p.read_text())["lang_to_id"]
    pay = json.dumps(sorted((str(k), int(v)) for k, v in lt.items()), separators=(",", ":"), ensure_ascii=False)
    return {"gen": hashlib.sha256(p.read_bytes()).hexdigest(), "map": hashlib.sha256(pay.encode()).hexdigest(),
            "n": len(set(lt.values())), "enzh": [lt["<|en|>"], lt["<|zh|>"]]}


def p2r_steps(uid_index: dict, uid: str) -> dict:
    steps = json.loads((ROOT / f"results/inference_cf/p2r/run3/rows/{uid_index[uid]:03d}.json").read_text())["current"]["steps"]
    return {s["query"]: s for s in steps}


# ---- prerun ------------------------------------------------------------------------------------------

def cmd_prerun(args) -> dict:
    c = cfg()
    checks, notes = {}, {}
    for rel in FROZEN:
        checks[f"frozen_at_{FREEZE[:7]}:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["sources"]["code_sha256"].items():
        checks[f"code_anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    for rel, h in c["sources"]["evidence_sha256"].items():
        checks[f"evidence_anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    I = c["E0"]["integrity"]
    pv = provider()
    checks["provider_identity"] = pv["gen"] == I["generation_config_sha256"] and pv["map"] == I["canonical_mapping_sha256"] \
        and pv["n"] == 100 and pv["enzh"] == I["en_zh_token_ids"]
    src = (ROOT / "experiments/inference_cf_p0_r2.py").read_text()
    checks["native_lid_100way_source"] = "probs = torch.softmax(vals, dim=0)" in src and \
        "tuple(sorted(set(int(x) for x in native.values())))" in (ROOT / "experiments/inference_cf_p2r.py").read_text()
    checks["no_Q_near_one_rule"] = I["Q_near_one_integrity_check"] is False
    pos = json.loads((ROOT / c["E0"]["population"]).read_text())
    checks["population_hash"] = fhash(ROOT / c["E0"]["population"]) == "sha256:" + c["E0"]["population_sha256"]
    checks["population_180"] = len(pos["positions"]) == 180 and \
        {s: sum(p["stratum"] == s for p in pos["positions"]) for s in c["E0"]["counts"]} == c["E0"]["counts"]
    s1 = json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())
    a1 = json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_audit.json").read_text())
    fa = json.loads((ROOT / "results/inference_cf/p2sel/final_audit.json").read_text())
    checks["p2sel_terminal_audited"] = s1["decision"]["label"] == c["starting_evidence"]["terminal_parent_label"] \
        and a1["verdict"] == "P2_SEL_AUDIT: PASS" and fa["verdict"] == "P2_SEL_AUDIT: PASS"
    pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(pop["utterances"])}
    sel = {(r["utterance_id"], r["t"]): r for r in s1["per_position"]}
    groups, join_ok = {"EN_TP": 0, "EN_FN": 0, "ZH_TN": 0, "ZH_FP": 0}, True
    for p in pos["positions"]:
        st = p2r_steps(pidx, p["utterance_id"])
        s = st.get(p["p2r"]["query"])
        e = sel[(p["utterance_id"], p["t"])]["gate"]["E"]
        join_ok &= s is not None and s["t"] == p["t"] and s["E"] is not None and s["E"] == e
        if p["stratum"] == "EN-confusion":
            groups["EN_TP" if e > 0 else "EN_FN"] += 1
        elif p["stratum"] == "ZH-correct":
            groups["ZH_FP" if e > 0 else "ZH_TN"] += 1
    checks["p2r_join_and_E_identity"] = bool(join_ok)
    exp = c["E0"]["expected_failure_groups"]
    checks["expected_groups_from_frozen_scalars"] = groups == {"EN_TP": exp["EN_confusion_TP_E_positive"], "EN_FN": exp["EN_confusion_FN_E_zero"],
                                                               "ZH_TN": exp["ZH_correct_TN_E_zero"], "ZH_FP": exp["ZH_correct_FP_E_positive"]}
    sealed = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1/directions_sealed.json").read_text())
    checks["d2_sealed"] = sealed["status"] == "SEALED" and all(fhash(ROOT / "results/inference_cf/p2dir/exp1_run1" / r) == h
                                                               for r, h in sealed["files"].items())
    checks["mini_panel_hash"] = fhash(ROOT / c["E2"]["panel"]) == "sha256:" + c["E2"]["panel_sha256"]
    checks["no_e_stage_outcome"] = not any(x.name.startswith(("e0", "e1", "e2")) for x in BASE.iterdir())
    checks["runtime_surface_reference_free"] = runtime_surface_clean()["ok"]
    asrc = (ROOT / "experiments/inference_cf_p2sel_e_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*_analyze", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2sel_e.py").exists()
    th = c["E1"]["thresholds"]
    checks["e1_constants"] = (th["benefit_retention_min"] == 0.7 and th["harm_ratio_max"] == 0.5 and th["paired_ZH_new_minus_old_min_nat"] == 0.1
                              and th["correct_margin_simultaneous_lower_min_nat"] == -0.25 and th["correct_corruption_observed_max"] == 0.05
                              and th["correct_corruption_simultaneous_upper_max"] == 0.1 and c["E1"]["bootstrap"]["family_size"] == 6)
    notes["groups"] = groups
    notes["preregistered_conflict"] = ("E0 precedence can select R3 (H_E3), but config E1.requires and spec sections 3/6 "
                                       "authorize E1 only for R1/R2/R4. If E0 selects R3, the pre-E1 audit BLOCKs and the "
                                       "stage stops (contract conflict); no E1 is run.")
    notes["preregistered_interpretations"] = [
        "lag E=None (fallback step in the P2-R trace) is treated as missing history = 0",
        "a hypothesis bootstrap with fewer than 9900 valid draws fails that hypothesis criterion",
        "E_long is the recomputed 1 s current E (must reproduce P2-R within 1e-6)"]
    verdict = "PASS_TO_P2_SEL_E_E0" if all(checks.values()) else "BLOCK_BEFORE_P2_SEL_E_E0"
    return {"schema": "p2_sel_e_prerun_audit_r1_v1", "verdict": verdict, "checks": checks, "notes": notes,
            "git_commit": _git("rev-parse", "HEAD")}


# ---- E0 ------------------------------------------------------------------------------------------------

def _lo80(rows_a, rows_b, fa, fb, keys, idx):
    def dm(rows, f):
        acc = {}
        for r in rows:
            acc.setdefault(r["d"], []).append(float(f(r)))
        return {k: sum(v) / len(v) for k, v in acc.items()}
    A, B = dm(rows_a, fa), dm(rows_b, fb)
    diffs = []
    for draw in idx:
        ks = [keys[j] for j in draw]
        va, vb = [A[k] for k in ks if k in A], [B[k] for k in ks if k in B]
        if va and vb:
            diffs.append(sum(va) / len(va) - sum(vb) / len(vb))
    pt = (sum(A.values()) / len(A) - sum(B.values()) / len(B)) if A and B else None
    return pt, (float(np.quantile(diffs, 0.20)) if diffs else None), len(diffs)


def e0_recompute() -> dict:
    c = cfg()
    pos = json.loads((ROOT / c["E0"]["population"]).read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    run = BASE / "e0_run1"
    null = json.loads((run / "null.json").read_text())
    pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(pop["utterances"])}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in
           json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    eps = 1e-12
    lnull = math.log((null["pi_E"] + eps) / (null["pi_M"] + eps))
    rows, fails = [], []
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        e = json.loads((run / f"rows/{cidx[uid]:03d}.json").read_text())["positions"][str(t)]
        st = p2r_steps(pidx, uid)
        q = p["p2r"]["query"]
        ll = math.log((e["long"]["pi_E"] + eps) / (e["long"]["pi_M"] + eps))
        ls = math.log((e["short"]["pi_E"] + eps) / (e["short"]["pi_M"] + eps))
        E = max(0.0, math.tanh((ll - lnull) / 2))
        Es = max(0.0, math.tanh((ls - lnull) / 2))
        lag = [0.0 if st.get(q - k) is None or st[q - k]["E"] is None else float(st[q - k]["E"]) for k in (1, 2)]
        ac = st[q].get("acoustic") if p["stratum"] == "EN-confusion" else None
        g = p["stratum"]
        es = sel[(uid, t)]
        grp = {"EN-confusion": "EN_TP" if es > 0 else "EN_FN", "ZH-correct": "ZH_FP" if es > 0 else "ZH_TN"}.get(g, "EN_CORRECT")
        hs = e["heard_samples"]
        s0, s1 = e["window"][2], e["window"][3]
        a = 0 if hs < 8000 else min(max((s0 + s1 - 8000) // 2, 0), hs - 8000)
        ok = (abs(E - st[q]["E"]) <= 1e-6 and e["window"] == st[q]["window"] and e["short"]["bounds"] == ([0, hs] if hs < 8000 else [a, a + 8000])
              and all(abs(x["softmax_sum"] - 1) <= 1e-6 and x["n_probs"] == 100 for x in (e["long"], e["short"])))
        if not ok:
            fails.append((uid, t))
        A = ll - lnull
        rows.append({"uid": uid, "t": t, "d": p["dialogue_id"], "g": grp, "E": E, "Es": Es, "Q": e["long"]["pi_E"] + e["long"]["pi_M"],
                     "ll": ll, "ln": lnull, "A": A, "ns": (-lnull) / A if A > 0 else None, "lag": lag,
                     "Eo": ac.get("E_oracle") if ac and ac.get("status") == "ok" else None})
    return {"rows": rows, "fails": fails, "null_ok": abs(null["softmax_sum"] - 1) <= 1e-6 and null["n_probs"] == 100}


def e0_decide(rows) -> dict:
    c = cfg()
    G = {k: [r for r in rows if r["g"] == k] for k in ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")}
    sp = lambda rs: len({r["d"] for r in rs})
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    keys = sorted({r["d"] for r in rows})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))
    a = [r for r in fp if r["Es"] <= 0.10 and r["E"] - r["Es"] >= 0.25]
    b = [r for r in tp if r["Es"] >= 0.20]
    static = len(a) >= 4 and len(b) >= 34 and sp(a) >= 3 and sp(b) >= 3
    o = [r for r in fn if r["Eo"] is not None and r["Eo"] >= 0.2 and r["Eo"] - r["E"] >= 0.2]
    oracle = len(o) >= 9 and sp(o) >= 3
    so = [r for r in fn if r["Es"] >= 0.2 and r["E"] == 0]
    short = len(so) >= 9 and sp(so) >= 3
    P = lambda r: max(r["lag"])
    iso = [r for r in fp if r["E"] >= 0.5 and P(r) <= 0.10]
    per = [r for r in tp if P(r) >= 0.25]
    _, l2, n2 = _lo80(tp, fp, lambda r: P(r) >= 0.25, lambda r: P(r) >= 0.25, keys, idx)
    h2 = len(iso) >= 4 and sp(iso) >= 3 and len(per) >= 26 and sp(per) >= 3 and n2 >= 9900 and l2 is not None and l2 > 0
    ql = c["E0"]["hypotheses"]["H_E3"]["q_low"]
    fl = [r for r in fp if r["Q"] <= ql]
    tl = [r for r in tp if r["Q"] <= ql]
    p3, l3, n3 = _lo80(fp, tp, lambda r: r["Q"] <= ql, lambda r: r["Q"] <= ql, keys, idx)
    h3 = len(fl) >= 4 and len(tl) <= 10 and p3 is not None and p3 >= 0.5 and sp(fl) >= 3 and n3 >= 9900 and l3 is not None and l3 > 0
    nf = [r for r in fp if r["ll"] <= 0 and r["A"] > 0 and r["ns"] is not None and r["ns"] >= 0.8]
    h4 = len(nf) >= 4 and sp(nf) >= 3 and float(np.median([r["ll"] for r in tp])) > 0
    counts = {k: len(v) for k, v in G.items()}
    if oracle or short:
        lab, selr = "P2_SEL_E_LOCALIZER_PRIMARY", None
    elif static:
        lab, selr = "P2_SEL_E_REPAIR_SELECTED", "R2"
    elif h2:
        lab, selr = "P2_SEL_E_REPAIR_SELECTED", "R1"
    elif h3:
        lab, selr = "P2_SEL_E_REPAIR_SELECTED", "R3"
    elif h4:
        lab, selr = "P2_SEL_E_REPAIR_SELECTED", "R4"
    else:
        lab, selr = "P2_SEL_E_DIAGNOSIS_AMBIGUOUS", None
    return {"counts": counts, "H": {"static": static, "oracle_FN": oracle, "short_only_FN": short, "H_E2": h2, "H_E3": h3,
                                    "H_E4": h4, "h2_lower80": l2, "h3_point": p3, "h3_lower80": l3, "n2": n2, "n3": n3},
            "label": lab, "selected": selr}


def e_new_ind(br, r) -> float:
    if br == "R1":
        return r["E"] * max(r["lag"])
    if br == "R2":
        return math.sqrt(r["E"] * r["Es"])
    if br == "R3":
        return r["E"] * r["Q"]
    return max(0.0, math.tanh((r["ll"] - 0.5 * r["ln"]) / 2))


def cmd_e0(args) -> dict:
    ana = json.loads((ROOT / args.analysis).read_text())
    rc = e0_recompute()
    dec = e0_decide(rc["rows"])
    checks = {"rows_180": len(rc["rows"]) == 180, "row_integrity": not rc["fails"], "null_provider": rc["null_ok"],
              "counts": dec["counts"] == {"EN_TP": 42, "EN_FN": 18, "ZH_TN": 55, "ZH_FP": 5, "EN_CORRECT": 60},
              "counts_agree": dec["counts"] == ana["counts"]}
    valid = all(checks.values())
    label = dec["label"] if valid else "P2_SEL_E_INVALID"
    sel = dec["selected"] if valid else None
    H = ana["hypotheses"]
    checks["hypotheses_agree"] = (dec["H"]["static"] == H["H_E1"]["static_scale_pattern"] and dec["H"]["oracle_FN"] == H["H_E1"]["oracle_FN_pattern"]
                                  and dec["H"]["short_only_FN"] == H["H_E1"]["short_only_FN_pattern"] and dec["H"]["H_E2"] == H["H_E2"]["pass"]
                                  and dec["H"]["H_E3"] == H["H_E3"]["pass"] and dec["H"]["H_E4"] == H["H_E4"]["pass"])
    for k, ak in (("h2_lower80", ("H_E2", "bootstrap_TP_minus_FP_persistent", "lower80")),
                  ("h3_lower80", ("H_E3", "bootstrap_FP_minus_TP_lowQ", "lower80")),
                  ("h3_point", ("H_E3", "bootstrap_FP_minus_TP_lowQ", "point"))):
        x = H[ak[0]][ak[1]][ak[2]]
        checks[f"agree_{k}"] = (x is None and dec["H"][k] is None) or (x is not None and dec["H"][k] is not None and abs(x - dec["H"][k]) <= 1e-8)
    checks["label_agrees"] = label == ana["decision"]["label"] and sel == ana["decision"]["selected"]
    out = {"schema": "p2_sel_e_e0_audit_v1", "stage": "E0", "label": label, "selected": sel, "independent": dec["H"],
           "checks": checks}
    if sel is not None and args.selection:
        s = json.loads((ROOT / args.selection).read_text())
        by = {(r["uid"], r["t"]): r for r in rc["rows"]}
        checks["selection_branch"] = s["selected_repair"] == sel and s["diagnosis"] == label
        checks["selection_gates"] = len(s["gates"]) == 180 and all(
            abs(g["E_new"] - e_new_ind(sel, by[(g["utterance_id"], g["t"])])) <= 1e-12
            and abs(g["E_old"] - by[(g["utterance_id"], g["t"])]["E"]) <= 1e-12 for g in s["gates"])
        checks["selection_formula_frozen"] = s["formula"] == cfg()["E0"]["repair_formulas"][
            {"R1": "R1_temporal", "R2": "R2_multiscale", "R3": "R3_confidence", "R4": "R4_null_shrink"}[sel]]
        checks["selection_no_outcome"] = s["steering_outcome_exists"] is False and not (BASE / "e1_run1").exists()
    out["verdict"] = "P2_SEL_E_AUDIT: PASS" if all(checks.values()) else "P2_SEL_E_AUDIT: BLOCK"
    return out


def cmd_pre_e1(args) -> dict:
    c = cfg()
    a0 = json.loads((BASE / "e0_run1_audit.json").read_text())
    s = json.loads((BASE / "e0_run1_selection.json").read_text())
    checks = {"e0_audit_pass": a0["verdict"] == "P2_SEL_E_AUDIT: PASS",
              "exactly_one_branch": a0["selected"] in ("R1", "R2", "R3", "R4") and s["selected_repair"] == a0["selected"],
              "branch_authorized_by_E1_requires": s["selected_repair"] in ("R1", "R2", "R4")
                                                  and c["E1"]["requires"][0] == "one selected repair R1, R2, or R4",
              "gates_hash": canon(s["gates"]) == s["gates_hash"],
              "p2sel_c2_reuse": json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_audit.json").read_text())["verdict"] == "P2_SEL_AUDIT: PASS",
              "no_e1_outcome": not (BASE / "e1_run1").exists()}
    v = "P2_SEL_E_PRE_E1_AUDIT: PASS" if all(checks.values()) else "P2_SEL_E_PRE_E1_AUDIT: BLOCK"
    return {"schema": "p2_sel_e_pre_e1_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


# ---- E1 ------------------------------------------------------------------------------------------------

def cmd_e1(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2sel_audit import hook_edit_emulation
    c = cfg()
    ana = json.loads((ROOT / args.analysis).read_text())
    run = BASE / "e1_run1"
    man = json.loads((run / "manifest.json").read_text())
    pos = json.loads((ROOT / c["E0"]["population"]).read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    sel = json.loads((BASE / "e0_run1_selection.json").read_text())
    gates = {(g["utterance_id"], g["t"]): g for g in sel["gates"]}
    model = Path(man["model"]["dir"])
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    checks = {"manifest_self_hash": man["manifest_hash"] == canon({k: v for k, v in man.items() if k != "manifest_hash"}),
              "sources_at_manifest_commit": all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == man["manifest_hash"]}
    rows, efail = [], []
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        pre = f"t{t}_"
        rr = json.loads((run / f"rows/{i:03d}.json").read_text())["positions"][str(t)]
        with np.load(run / f"rows/{i:03d}_logits.npz") as z:
            ln = bf16(z[pre + "Cnew"])
        with np.load(ROOT / f"results/inference_cf/p2sel/s1_run1/rows/{i:03d}_a_logits.npz") as z:
            lo_ = bf16(z[pre + "C2"])
        with np.load(ROOT / f"results/inference_cf/p2dir/exp1_run1/rows/{i:03d}_logits.npz") as z:
            l0 = bf16(z[pre + "none"])
        with np.load(ROOT / f"results/inference_cf/p2dir/exp1_run1/rows/{i:03d}.npz") as z:
            hb, d2 = z[pre + "hb"], z[pre + "D2"]
        g = gates[(uid, t)]["g_new"]
        exp = hook_edit_emulation(hb, d2, g) if g > 0 else 0.0
        en = rr["arm"]["edit_norm"]
        if (exp == 0) != (en == 0) or (exp > 0 and abs(en / exp - 1) > 1e-5) or (g == 0 and not np.array_equal(ln, l0)):
            efail.append((uid, t, exp, en))
        Y, cc = [int(v) for v in p["target_ids"]], int(p["competitor"])
        m0 = metrics(l0, t, sup, beg, part, Y, cc)
        mo = metrics(lo_, t, sup, beg, part, Y, cc)
        mn = metrics(ln, t, sup, beg, part, Y, cc)
        rows.append({"d": p["dialogue_id"], "s": p["stratum"], "new": mn["m"] - m0["m"], "old": mo["m"] - m0["m"], "ok": mn["in_ref"]})
    checks["dose_energy_reconstruction"] = not efail
    keys = sorted({p["dialogue_id"] for p in pos["positions"]})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))
    W = np.stack([(idx == j).sum(axis=1) for j in range(len(keys))], axis=1).astype(np.float64)
    af = 0.10 / 6
    S = {s: [r for r in rows if r["s"] == s] for s in ("EN-confusion", "EN-correct", "ZH-correct")}
    f = {"conf": ci([(r["d"], r["new"]) for r in S["EN-confusion"]], keys, W, af),
         "en": ci([(r["d"], r["new"]) for r in S["EN-correct"]], keys, W, af),
         "zh": ci([(r["d"], r["new"]) for r in S["ZH-correct"]], keys, W, af),
         "paired_zh": ci([(r["d"], r["new"] - r["old"]) for r in S["ZH-correct"]], keys, W, af),
         "corr_en": ci([(r["d"], 0.0 if r["ok"] else 1.0) for r in S["EN-correct"]], keys, W, af),
         "corr_zh": ci([(r["d"], 0.0 if r["ok"] else 1.0) for r in S["ZH-correct"]], keys, W, af)}
    oc = ci([(r["d"], r["old"]) for r in S["EN-confusion"]], keys, W, .1)[0]
    oz = ci([(r["d"], r["old"]) for r in S["ZH-correct"]], keys, W, .1)[0]
    diffs = []
    for k, (est, iv, nd) in f.items():
        x = ana["family"][k]
        diffs += [abs(est - x["estimate"]), abs(iv[0] - x["ci"][0]), abs(iv[1] - x["ci"][1])]
    checks["statistics_agree_1e-8"] = max(diffs) <= 1e-8
    br, hr = f["conf"][0] / oc, abs(f["zh"][0]) / abs(oz)
    checks["ratios_agree"] = abs(br - ana["ratios"]["benefit_retention"]) <= 1e-8 and abs(hr - ana["ratios"]["harm_ratio"]) <= 1e-8
    valid = (len(rows) == 180 and all(nd >= 9900 for _, _, nd in f.values()) and abs(oc - 3.52) <= 5e-4 and abs(oz + 0.195) <= 5e-4
             and checks["dose_energy_reconstruction"] and all(ana["validity"].values()))
    safe = (f["en"][1][0] >= -0.25 and f["zh"][1][0] >= -0.25 and f["corr_en"][0] <= 0.05 and f["corr_zh"][0] <= 0.05
            and f["corr_en"][1][1] <= 0.10 and f["corr_zh"][1][1] <= 0.10)
    if not valid:
        lab = "P2_SEL_E_INVALID"
    elif not safe:
        lab = "P2_SEL_E_REPAIR_STILL_UNSAFE"
    elif br < 0.70:
        lab = "P2_SEL_E_REPAIR_TOO_CONSERVATIVE"
    elif not (f["conf"][1][0] > 0 and hr <= 0.5 and f["paired_zh"][0] >= 0.10 and f["paired_zh"][1][0] > 0):
        lab = "P2_SEL_E_NO_MATERIAL_GAIN"
    else:
        lab = "P2_SEL_E_REPAIR_SUPPORTED"
    checks["label_agrees"] = lab == ana["decision"]["label"]
    return {"schema": "p2_sel_e_e1_audit_v1", "stage": "E1", "label": lab, "verdict": "P2_SEL_E_AUDIT: PASS" if all(checks.values())
            else "P2_SEL_E_AUDIT: BLOCK", "checks": checks, "independent_family": f, "ratios": {"benefit_retention": br, "harm_ratio": hr},
            "energy_failures": efail[:10]}


def cmd_final(args) -> dict:
    c = cfg()
    checks = {}
    for rel in FROZEN:
        checks[f"unchanged:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["sources"]["code_sha256"].items():
        checks[f"code_anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    entries = sorted(x.name for x in BASE.iterdir())
    a0 = json.loads((BASE / "e0_run1_audit.json").read_text())
    checks["e0_audit_pass"] = a0["verdict"] == "P2_SEL_E_AUDIT: PASS"
    e1 = (BASE / "e1_run1").exists()
    checks["e1_only_if_selected_and_authorized"] = (not e1) or (a0["selected"] in ("R1", "R2", "R4"))
    if e1:
        checks["e1_audit_pass"] = json.loads((BASE / "e1_run1_audit.json").read_text())["verdict"] == "P2_SEL_E_AUDIT: PASS"
    e2 = any(e.startswith("e2") for e in entries)
    checks["e2_only_if_supported"] = (not e2) or (e1 and json.loads((BASE / "e1_run1_audit.json").read_text())["label"] == "P2_SEL_E_REPAIR_SUPPORTED")
    checks["no_p3_full300_extra_stage"] = not any(("p3" in e.lower()) or ("300" in e) for e in entries)
    checks["runtime_surface_reference_free"] = runtime_surface_clean()["ok"]
    for st in ("e0_run1", "e1_run1"):
        if (BASE / st).exists():
            m = json.loads((BASE / st / "manifest.json").read_text())
            checks[f"firewall_{st}"] = m["role"] == "D-dev-select" and m["firewall"]["role"] == "D-dev-select"
            checks[f"single_attempt_{st}"] = len(list((BASE / st).glob("slurm-*.out"))) == 1
    return {"schema": "p2_sel_e_final_audit_v1", "verdict": "P2_SEL_E_AUDIT: PASS" if all(checks.values()) else "P2_SEL_E_AUDIT: BLOCK",
            "e0": {"label": a0["label"], "selected": a0["selected"]}, "e1_run": e1, "e2_run": e2, "entries": entries,
            "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prerun", "pre_e1", "final"):
        sub.add_parser(name).add_argument("--out", required=True)
    e0 = sub.add_parser("e0")
    e0.add_argument("--analysis", required=True)
    e0.add_argument("--selection")
    e0.add_argument("--out", required=True)
    e1 = sub.add_parser("e1")
    e1.add_argument("--analysis", required=True)
    e1.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "e0": cmd_e0, "pre_e1": cmd_pre_e1, "e1": cmd_e1, "final": cmd_final}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "selected")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
