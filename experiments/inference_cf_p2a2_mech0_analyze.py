#!/usr/bin/env python
"""P2-A2-MECH0 analysis (frozen contract configs/inference_cf/p2_a2_mech0.json).

primary   (reference-free) integrity (FULL300 barrier, observer, archives, phase-2 completeness/ownership, budget), optimization
          dynamics, STEP1 vs STEP2, first-divergence margins (M_stop / M_branch), H3 continuation support, AUTO alignment, causal
          family-ablation action retention + sensitivity, FULL_A2 severe truncations vs B0.
secondary (references behind the committed seal + committed primary audit PASS) canonical B0/AUTO/A2 FULL300, per-row B0->A2 error
          decomposition by strata, F_ZH_EOS / F_DEL_EOS, net/positive contributions, dialogue robustness, concentration, LODO,
          paired dialogue bootstrap, MECH_PANEL condition metrics, mechanism label (frozen precedence) and confirmation readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest

CONFIG = "configs/inference_cf/p2_a2_mech0.json"
BASE = "results/inference_cf/p2a2_mech0"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
EOS = 50257
DROPS = ("DROP_SELF", "DROP_CROSS", "DROP_POST")
LABELS = ("P2_A2_MECH0_INVALID", "P2_A2_MECH0_TERMINATION_DOMINANT", "P2_A2_MECH0_NONTERMINATION_DOMINANT", "P2_A2_MECH0_MIXED",
          "P2_A2_MECH0_NO_CLEAR_MECHANISM")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def stream(t, term):
    return list(t) + ([EOS] if term == "eos" else [])


def severe(base_len, new_len, term):
    return base_len >= 10 and term == "eos" and new_len <= math.floor(0.5 * base_len)


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def rate(num, den):
    return {"num": num, "den": den, "rate": (num / den) if den else None}


# ---- frozen rules (pure) -------------------------------------------------------------------------------------------

def retention(actions: list) -> float | None:
    """Fraction of changed rows whose condition action equals the FULL_A2 action; empty group None."""
    return (sum(a == "A2" for a in actions) / len(actions)) if actions else None


def sensitive(ret_overall, ret_eos, c: dict) -> bool:
    thr = 0.25
    d = lambda r: None if r is None else 1 - r
    return any(x is not None and x >= thr - 1e-12 for x in (d(ret_overall), d(ret_eos)))


def fraction(num: float, den: float):
    return (num / den) if den else None


def group_stats(rows: list, dlg_key="dialogue_id") -> dict:
    pos = [r for r in rows if r["dZH"] > 0 or r["dMixed"] > 0]
    return {"rows": len(rows), "net_ZH": sum(r["dZH"] for r in rows), "net_mixed": sum(r["dMixed"] for r in rows),
            "net_POI": sum(r["dPOI"] for r in rows), "net_D": sum(r["dD"] for r in rows), "positive_rows": len(pos),
            "positive_dialogues": len({r[dlg_key] for r in pos}), "dialogues": len({r[dlg_key] for r in rows})}


def decide_mechanism(valid: bool, eos: dict, noneos: dict, F_ZH, F_DEL, med_dM2, frac_pos_dM2, c: dict) -> str:
    M = c["mechanism"]
    if not valid:
        return "P2_A2_MECH0_INVALID"
    both = F_ZH is not None and F_DEL is not None
    T = M["termination"]
    if (both and eos["rows"] >= T["EOS_rows_min"] and eos["dialogues"] >= T["EOS_dialogues_min"] and F_ZH >= T["F_ZH_min"] and F_DEL >= T["F_DEL_min"]
            and med_dM2 is not None and med_dM2 > 0 and frac_pos_dM2 is not None and frac_pos_dM2 >= T["fraction_deltaM2_strict_positive_min"]):
        return "P2_A2_MECH0_TERMINATION_DOMINANT"
    N = M["nontermination"]
    if (both and F_ZH < N["F_ZH_strict_below"] and F_DEL < N["F_DEL_strict_below"] and (noneos["net_ZH"] > 0 or noneos["net_mixed"] > 0)
            and noneos["positive_rows"] >= N["positive_rows_min"] and noneos["positive_dialogues"] >= N["positive_dialogues_min"]):
        return "P2_A2_MECH0_NONTERMINATION_DOMINANT"
    if both:
        mat = lambda g: (g["net_ZH"] >= 5 or g["net_mixed"] >= 5) and g["positive_rows"] >= 2 and g["positive_dialogues"] >= 2
        inter = (0.25 <= F_ZH < 0.50) or (0.25 <= F_DEL < 0.50)
        pos = lambda g: g["net_ZH"] > 0 or g["net_mixed"] > 0
        if (mat(eos) and mat(noneos)) or (inter and pos(eos) and pos(noneos)):
            return "P2_A2_MECH0_MIXED"
    return "P2_A2_MECH0_NO_CLEAR_MECHANISM"


def readiness(valid_and_audited: bool, missed_severe: int) -> str:
    return "YES" if valid_and_audited and missed_severe == 0 else "NO"


# ---- load / primary ------------------------------------------------------------------------------------------------

def load(run_dir: Path) -> dict:
    man = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = [json.loads((run_dir / f"rows/{i:03d}.json").read_text()) if (run_dir / f"rows/{i:03d}.json").exists() else {"status": "missing"}
            for i in range(len(plan["rows"]))]
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    rows, plan, rt = d["rows"], d["plan"], d["runtime"]
    run_dir = ROOT / run_rel
    integ = {"all_300_ok": len(rows) == 300 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
             "no_runtime_invalid": not rt.get("invalid"), "barrier_pass": bool((rt.get("barrier") or {}).get("pass")),
             "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged")) and bool(rt.get("model_grads_none"))}
    res = {"schema": "p2_a2_mech0_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "integrity": integ}
    if not integ["all_300_ok"]:
        res["valid"] = False
        return res
    P = plan["partitions"]
    au = {x["runtime"]["utterance_id"]: x["audit"] for x in plan["rows"]}
    R = {r["identity"]: r for r in rows}
    ok = {"reconstruction300": all(r["theta0_equals_B0"] and r["A2_equals_sealed"] and r["A2_state_hash_equals_sealed"]
                                   and r["step2_snapshot_equals_final_masters"] and r["reset_ok"] for r in rows),
          "fixed100_archives": sum(r["fp32_equals_archive"] is True for r in rows) == 100 and not any(r["fp32_equals_archive"] is False for r in rows),
          "step1_differs_step2": all(r["step1_differs_from_step2"] or r["A2_log"]["no_valid_content"] for r in rows),
          "episodes": all(r["A2_log"]["steps"] == 2 and r["A2_log"]["loss_evaluations"] == 3 and all(r["A2_log"]["finite"]) for r in rows),
          "live_audit": bool(rows[0].get("live_audit", {}).get("pass"))}
    arch_ok = (run_dir / "states" / "theta0_ln_fp32.npz").exists()
    for i, r in enumerate(rows):
        arch_ok &= sha(run_dir / "states" / f"{i:03d}_step1_masters_fp32.npz") == r["step1_archive_sha256"]
        if au[r["identity"]]["archive"] is None:
            arch_ok &= sha(run_dir / "states" / f"{i:03d}_step2_masters_fp32.npz") == r["step2_archive_sha256"]
    ok["state_archives"] = bool(arch_ok) and sum(1 for _ in (run_dir / "states").glob("*_step1_*.npz")) == 300 and \
        sum(1 for _ in (run_dir / "states").glob("*_step2_*.npz")) == 200
    mp, ch = P["MECH_PANEL"], P["MECH_CHANGED"]
    ok["phase2_complete"] = all(R[u].get("phase2") and set(R[u]["diag_decodes"]) == {"A2_STEP1", *DROPS} for u in mp) and \
        all(set(R[u]["geometry"]) == {"STEP0", "STEP1", "STEP2"} and set(R[u]["drop_query"]) == set(DROPS) for u in ch) and \
        all("geometry" not in R[u] for u in P["CONTROL40"])
    ok["ownership"] = all(R[u]["geometry"][s]["fed_ok"] and R[u]["geometry"][s]["state_locked"] for u in ch for s in ("STEP0", "STEP1", "STEP2")) and \
        all(R[u]["geometry"]["STEP0"]["state_hash"] == rt["theta0_ln_hash"] and R[u]["geometry"]["STEP2"]["state_hash"] == R[u]["step2_state_hash"]
            and R[u]["geometry"]["STEP1"]["state_hash"] == R[u]["step1_state_hash"] for u in ch) and \
        all(R[u]["diag_decodes"]["A2_STEP1"]["state_hash"] == R[u]["step1_state_hash"] for u in mp)
    cnt, B = rt["counters"], c["compute"]
    ok["budget"] = (cnt["A2"]["optimizer_steps"] == B["updates_exact"] and cnt["A2"]["backwards"] + cnt["audit"]["backwards"] <= B["backward_max_including_live_audit"]
                    and cnt["observer_snapshots"] == 600 and cnt["theta0_decodes"] + cnt["A2_final_decodes"] == B["ordinary_reconstruction_decodes"]
                    and cnt["step1_decodes"] == B["step1_decodes"] and cnt["drop_decodes"] == B["DROP_decodes"]
                    and cnt["geometry_paths"] == B["geometry_paths"] and cnt["drop_query_paths"] == B["ablation_current_query_paths"]
                    and rt["elapsed_sec"] <= 60 * B["hard_minutes"])
    integ.update(ok)
    valid = all(integ.values())
    ids = plan["ids"]
    grp = {"FULL300": ids, **{k: P[k] for k in ("A2_SAME", "A2_DELTA", "EOS_RECOVERY", "CONTENT_DIVERGENCE", "AUTO_SAME", "AUTO_DELTA", "FIXED100", "NEW200")}}
    L = lambda u, j: R[u]["A2_log"]["losses"][j]
    dyn = {g: {"n": len(us), "median_loss": [med([L(u, j) for u in us]) for j in range(3)], "mean_loss": [float(np.mean([L(u, j) for u in us])) if us else None for j in range(3)],
               "median_master_delta": {f"step{s}": med([R[u]["norms"][f"step{s}_master"]["total"] for u in us]) for s in (1, 2)},
               "median_master_rel": {f"step{s}": med([R[u]["norms"][f"step{s}_master"]["relative"] for u in us]) for s in (1, 2)},
               "median_effective_delta": {f"step{s}": med([R[u]["norms"][f"step{s}_effective"]["total"] for u in us]) for s in (1, 2)},
               "median_family_master_step2": {f: med([R[u]["norms"]["step2_master"]["family"][f] for u in us]) for f in ("SELF", "CROSS", "POST")},
               "median_family_master_step1": {f: med([R[u]["norms"]["step1_master"]["family"][f] for u in us]) for f in ("SELF", "CROSS", "POST")},
               "median_step1_to_step2": med([R[u]["norms"]["step1_to_step2_master_l2"] for u in us])} for g, us in grp.items()}
    eosR, cont = P["EOS_RECOVERY"], P["CONTENT_DIVERGENCE"]
    s1 = lambda us: rate(sum(R[u]["diag_decodes"]["A2_STEP1"]["equals_A2"] for u in us), len(us))
    step = {"STEP1_equals_A2_transcript": {"changed": s1(ch), "controls": s1(P["CONTROL40"]), "EOS_RECOVERY": s1(eosR), "CONTENT_DIVERGENCE": s1(cont),
                                           "MECH_PANEL": s1(mp)},
            "STEP1_first_action_matches_A2": {"changed": rate(sum(R[u]["geometry"]["STEP1"]["action"] == "A2" for u in ch), len(ch)),
                                              "EOS_RECOVERY": rate(sum(R[u]["geometry"]["STEP1"]["action"] == "A2" for u in eosR), len(eosR)),
                                              "CONTENT_DIVERGENCE": rate(sum(R[u]["geometry"]["STEP1"]["action"] == "A2" for u in cont), len(cont))},
            "STEP0_action_is_B0": rate(sum(R[u]["geometry"]["STEP0"]["action"] == "B0" for u in ch), len(ch)),
            "STEP2_action_is_A2": rate(sum(R[u]["geometry"]["STEP2"]["action"] == "A2" for u in ch), len(ch)),
            "STEP1_prefix_greedy_mismatch_rows": sum(bool(R[u]["geometry"]["STEP1"]["prefix_greedy_mismatch_positions"]) for u in ch)}
    lp = lambda u, s, cnd: R[u]["geometry"][s]["cands"][cnd]["logp"]

    def marg(us, a, b):
        out = []
        for u in us:
            v = {s: (lp(u, s, a) - lp(u, s, b)) if (lp(u, s, a) is not None and lp(u, s, b) is not None) else None for s in ("STEP0", "STEP1", "STEP2")}
            v["dM1"] = None if None in (v["STEP1"], v["STEP0"]) else v["STEP1"] - v["STEP0"]
            v["dM2"] = None if None in (v["STEP2"], v["STEP0"]) else v["STEP2"] - v["STEP0"]
            out.append({"utterance_id": u, "dialogue_id": au[u]["dialogue_id"], **v})
        return out

    def msum(m):
        dm2 = [x["dM2"] for x in m if x["dM2"] is not None]
        dm1 = [x["dM1"] for x in m if x["dM1"] is not None]
        return {"rows": len(m), "median": {s: med([x[s] for x in m]) for s in ("STEP0", "STEP1", "STEP2", "dM1", "dM2")},
                "dM1_strict_positive": sum(x > 0 for x in dm1), "dM2_strict_positive": sum(x > 0 for x in dm2),
                "dM2_positive_rate": (sum(x > 0 for x in dm2) / len(dm2)) if dm2 else None,
                "dM2_positive_dialogues": len({x["dialogue_id"] for x in m if x["dM2"] is not None and x["dM2"] > 0}),
                "dialogues": len({x["dialogue_id"] for x in m})}
    M_stop, M_branch = marg(eosR, "A2", "EOS"), marg(cont, "A2", "B0")
    C_H = lambda us, s: med([R[u]["geometry"][s]["continuation"]["C_H"] for u in us])
    contin = {g: {s: C_H(us, s) for s in ("STEP0", "STEP1", "STEP2")} for g, us in (("changed", ch), ("EOS_RECOVERY", eosR), ("CONTENT_DIVERGENCE", cont))}
    contin["rows_C_H_increase_step2"] = {g: sum(1 for u in us if R[u]["geometry"]["STEP2"]["continuation"]["C_H"] is not None
                                                 and R[u]["geometry"]["STEP2"]["continuation"]["C_H"] > R[u]["geometry"]["STEP0"]["continuation"]["C_H"])
                                         for g, us in (("changed", ch), ("EOS_RECOVERY", eosR), ("CONTENT_DIVERGENCE", cont))}

    def align(us):
        aa = [R[u]["auto_alignment"] for u in us]
        mt = [a for a in aa if a["AUTO_prefix_equals_common"]]
        f = lambda xs, key: rate(sum(a[key] is True for a in xs), sum(a[key] is not None for a in xs)) | {"unavailable": sum(a[key] is None for a in xs)}
        return {"all": {k: f(aa, k) for k in ("A2_action_eq_AUTO", "B0_action_eq_AUTO", "A2_H_eq_AUTO")},
                "matched_prefix": {k: f(mt, k) for k in ("A2_action_eq_AUTO", "B0_action_eq_AUTO", "A2_H_eq_AUTO")}, "matched_prefix_rows": len(mt)}
    asame, adelta = set(P["AUTO_SAME"]), set(P["AUTO_DELTA"])
    auto = {"changed": align(ch), "EOS_RECOVERY": align(eosR), "CONTENT_DIVERGENCE": align(cont), "AUTO_SAME": align([u for u in ch if u in asame]),
            "AUTO_DELTA": align([u for u in ch if u in adelta])}
    abl = {}
    for cond in ("A2_STEP1",) + DROPS:
        act = (lambda u: R[u]["geometry"]["STEP1"]["action"]) if cond == "A2_STEP1" else (lambda u, cc=cond: R[u]["drop_query"][cc]["action"])
        ro, re_, rc = retention([act(u) for u in ch]), retention([act(u) for u in eosR]), retention([act(u) for u in cont])
        dd = lambda us: {"equals_A2": rate(sum(R[u]["diag_decodes"][cond]["equals_A2"] for u in us), len(us)),
                         "equals_B0": rate(sum(R[u]["diag_decodes"][cond]["equals_B0"] for u in us), len(us)),
                         "median_ED_to_A2": med([R[u]["diag_decodes"][cond]["ED_to_A2"] for u in us]), "median_ED_to_B0": med([R[u]["diag_decodes"][cond]["ED_to_B0"] for u in us]),
                         "sum_ED_to_A2": sum(R[u]["diag_decodes"][cond]["ED_to_A2"] for u in us), "sum_ED_to_B0": sum(R[u]["diag_decodes"][cond]["ED_to_B0"] for u in us)}
        abl[cond] = {"RET_overall": ro, "RET_EOS": re_, "RET_content": rc, "dRET_overall": None if ro is None else 1 - ro, "dRET_EOS": None if re_ is None else 1 - re_,
                     "actions": {k: sum(act(u) == k for u in ch) for k in ("A2", "B0", "third")},
                     "free_decode": {"changed": dd(ch), "controls": dd(P["CONTROL40"]), "EOS_RECOVERY": dd(eosR), "CONTENT_DIVERGENCE": dd(cont)},
                     "sensitive": sensitive(ro, re_, c) if cond in DROPS else None}
    sens = [f.replace("DROP_", "") for f in DROPS if abl[f]["sensitive"]]
    missed = [u for u in ids if severe(len(R[u]["theta0"]["tokens"]), len(R[u]["A2_final"]["tokens"]), R[u]["A2_final"]["terminated"])]
    res.update(valid=valid, dynamics=dyn, step1_vs_step2=step, M_stop={"summary": msum(M_stop), "rows": M_stop},
               M_branch={"summary": msum(M_branch), "rows": M_branch}, continuation=contin, auto_alignment=auto, ablations=abl,
               MECHANISM_SENSITIVE_families=sens, FULL_A2_severe_truncations_vs_B0=missed,
               strata_counts={k: len(v) for k, v in P.items()},
               runtime={k: rt.get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "phase1_sec", "phase2_sec", "peak_alloc", "peak_reserved", "counters", "gpu")})
    return res


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from csasr.evaluation.canonical import corpus_metrics
    from experiments.inference_cf_p2seq_analyze import draws, ratio_delta, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("MECH0 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed MECH0 outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != "P2_A2_MECH0_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    refs = load_references()
    plan = d["plan"]
    P = plan["partitions"]
    ids = plan["ids"]
    R = {r["identity"]: r for r in d["rows"]}
    au = {x["runtime"]["utterance_id"]: x["audit"] for x in plan["rows"]}
    auto_txt = {}
    p5panel = {r["utterance_id"]: r for r in json.loads((ROOT / "docs/inference_cf/P2_A2_MECH0_PANEL.json").read_text())["rows"]}
    for u in ids:
        auto_txt[u] = json.loads((ROOT / p5panel[u]["B0_AUTO"]["path"]).read_text())["systems"]["B0_AUTO"]["text"]
    ref = {u: refs[u]["reference"] for u in ids}
    D = {u: refs[u]["dialogue_id"] for u in ids}
    integ = {"primary_valid": bool(prim["valid"]), "dialogues_match": all(D[u] == au[u]["dialogue_id"] for u in ids)}
    H = {u: {"B0": R[u]["theta0"]["text"], "AUTO": auto_txt[u], "A2": R[u]["A2_final"]["text"]} for u in ids}
    sdi = lambda r_, h: (lambda m: [m["substitutions"], m["deletions"], m["insertions"]])(corpus_metrics([r_], [h]))
    CNT = {u: {s: utt_counts(ref[u], H[u][s]) + sdi(ref[u], H[u][s]) for s in ("B0", "AUTO", "A2")} for u in ids}

    def corpus(us, texts):
        cm = corpus_metrics([ref[u] for u in us], [texts[u] for u in us])
        return {"pier": cm["pier"], "mer": cm["mer"], "en_wer": cm["en_wer"], "zh_cer": cm["zh_cer"], "S": cm["substitutions"], "D": cm["deletions"],
                "I": cm["insertions"], "num_poi_errors": cm["num_poi_errors"]}
    sysm = {s: {**corpus(ids, {u: H[u][s] for u in ids}), "zh": sum(CNT[u][s][4] for u in ids), "poi": sum(CNT[u][s][0] for u in ids),
                "mixed": sum(CNT[u][s][2] for u in ids)} for s in ("B0", "AUTO", "A2")}
    integ["PATH5_aggregates_reproduced"] = ([sysm["B0"][k] for k in ("zh", "poi", "mixed")] == [3309, 1074, 4456]
                                            and [sysm["A2"][k] for k in ("zh", "poi", "mixed")] == [3037, 1012, 4126])
    per = []
    for u in ids:
        b, a = CNT[u]["B0"], CNT[u]["A2"]
        per.append({"utterance_id": u, "dialogue_id": D[u], "kind": au[u]["kind"], "relation": au[u]["teacher_relation"], "PATH5_partition": au[u]["PATH5_partition"],
                    "dZH": b[4] - a[4], "dPOI": b[0] - a[0], "dMixed": b[2] - a[2], "dS": b[8] - a[8], "dD": b[9] - a[9], "dI": b[10] - a[10]})
    pe = {e["utterance_id"]: e for e in per}
    keyset = {"EOS_RECOVERY": P["EOS_RECOVERY"], "EOS_REGRESSION": P["EOS_REGRESSION"], "CONTENT_DIVERGENCE": P["CONTENT_DIVERGENCE"], "A2_SAME": P["A2_SAME"],
              "AUTO_SAME": P["AUTO_SAME"], "AUTO_DELTA": P["AUTO_DELTA"], "AUTO_SAME__A2_SAME": P["AUTO_SAME__A2_SAME"], "AUTO_SAME__A2_DELTA": P["AUTO_SAME__A2_DELTA"],
              "AUTO_DELTA__A2_SAME": P["AUTO_DELTA__A2_SAME"], "AUTO_DELTA__A2_DELTA": P["AUTO_DELTA__A2_DELTA"], "FIXED100": P["FIXED100"], "NEW200": P["NEW200"],
              "FULL300": ids}

    def gsum(us):
        es = [pe[u] for u in us]
        pos = lambda k: sum(max(e[k], 0) for e in es)
        return {"rows": len(es), "dialogues": len({e["dialogue_id"] for e in es}), **{f"net_{k}": sum(e[k] for e in es) for k in ("dZH", "dPOI", "dMixed", "dS", "dD", "dI")},
                **{f"pos_{k}": pos(k) for k in ("dZH", "dPOI", "dMixed", "dD")},
                "improved_ZH_rows": sum(e["dZH"] > 0 for e in es), "worsened_ZH_rows": sum(e["dZH"] < 0 for e in es)}
    groups = {k: gsum(v) for k, v in keyset.items()}
    integ["group_sums_consistent"] = all(sum(groups[g][f"net_{k}"] for g in ("EOS_RECOVERY", "EOS_REGRESSION", "CONTENT_DIVERGENCE", "A2_SAME")) == groups["FULL300"][f"net_{k}"]
                                         for k in ("dZH", "dPOI", "dMixed", "dD")) and \
        groups["FULL300"]["net_dZH"] == sysm["B0"]["zh"] - sysm["A2"]["zh"]
    pZ = {u: max(pe[u]["dZH"], 0) for u in ids}
    pDl = {u: max(pe[u]["dD"], 0) for u in ids}
    F_ZH = fraction(sum(pZ[u] for u in P["EOS_RECOVERY"]), sum(pZ.values()))
    F_DEL = fraction(sum(pDl[u] for u in P["EOS_RECOVERY"]), sum(pDl.values()))
    eos_g = group_stats([pe[u] for u in P["EOS_RECOVERY"]])
    non_g = group_stats([pe[u] for u in P["CONTENT_DIVERGENCE"] + P["EOS_REGRESSION"]])
    ms = prim["M_stop"]["summary"]
    valid = all(integ.values())
    label = decide_mechanism(valid, eos_g, non_g, F_ZH, F_DEL, ms["median"]["dM2"], ms["dM2_positive_rate"], c)

    def share(pos):
        tot = sum(pos.values())
        if not tot:
            return {"max_row": None, "max_row_share": None, "max_dialogue": None, "max_dialogue_share": None}
        pdlg = {}
        for u, v in pos.items():
            pdlg[D[u]] = pdlg.get(D[u], 0) + v
        mr, md = max(pos, key=pos.get), max(pdlg, key=pdlg.get)
        return {"max_row": mr, "max_row_share": pos[mr] / tot, "max_dialogue": md, "max_dialogue_share": pdlg[md] / tot, "total": tot}
    dlgs = sorted(set(D.values()))
    per_d = {dd: {k: sum(pe[u][k] for u in ids if D[u] == dd) for k in ("dZH", "dPOI", "dMixed", "dD")} for dd in dlgs}
    robust = {k: {"improved": sum(v[k] > 0 for v in per_d.values()), "worsened": sum(v[k] < 0 for v in per_d.values()), "tied": sum(v[k] == 0 for v in per_d.values())}
              for k in ("dZH", "dPOI", "dMixed", "dD")}
    lodo = {}
    for k in ("dZH", "dMixed", "dPOI", "dD"):
        v = [sum(pe[u][k] for u in ids if D[u] != dd) for dd in dlgs]
        lodo[k] = {"min": min(v), "median": float(np.median(v)), "max": max(v), "count_positive": sum(x > 0 for x in v), "proportion_positive": sum(x > 0 for x in v) / len(v)}
    idxs = draws([D[u] for u in ids], reps=c["bootstrap"]["reps"], seed=c["bootstrap"]["seed"])
    CA = {s: np.array([CNT[u][s] for u in ids], dtype=float) for s in ("B0", "A2")}
    boot = {nm: ratio_delta(CA["A2"], CA["B0"], a_, q_, idxs) for nm, (a_, q_) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}.items()}
    for nm, col in (("ZH", 4), ("POI", 0), ("mixed", 2), ("deletions", 9)):
        vals = [float(CA["B0"][ix, col].sum() - CA["A2"][ix, col].sum()) for ix in idxs]
        boot[f"B0_minus_A2_{nm}"] = {"delta": float(CA["B0"][:, col].sum() - CA["A2"][:, col].sum()), "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]}
    eos_mask = np.array([u in set(P["EOS_RECOVERY"]) for u in ids])
    pz, pd_ = np.array([pZ[u] for u in ids], dtype=float), np.array([pDl[u] for u in ids], dtype=float)
    for nm, arr in (("F_ZH_EOS", pz), ("F_DEL_EOS", pd_)):
        vals, om = [], 0
        for ix in idxs:
            den = arr[ix].sum()
            if den:
                vals.append(float((arr[ix] * eos_mask[ix]).sum() / den))
            else:
                om += 1
        boot[nm] = {"point": fraction(float((arr * eos_mask).sum()), float(arr.sum())), "ci95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else None,
                    "omitted_zero_denominator": om}
    mp = P["MECH_PANEL"]
    cond_txt = {"B0": {u: R[u]["theta0"]["text"] for u in mp}, "FULL_A2": {u: R[u]["A2_final"]["text"] for u in mp},
                **{cnd: {u: R[u]["diag_decodes"][cnd]["text"] for u in mp} for cnd in ("A2_STEP1",) + DROPS}}
    cond = {}
    for cnd, tx in cond_txt.items():
        for gname, us in (("MECH_PANEL", mp), ("MECH_CHANGED", P["MECH_CHANGED"]), ("EOS_RECOVERY", P["EOS_RECOVERY"])):
            cc = [utt_counts(ref[u], tx[u]) for u in us]
            cond.setdefault(cnd, {})[gname] = {**corpus(us, tx), "zh": sum(x[4] for x in cc), "poi": sum(x[0] for x in cc), "mixed": sum(x[2] for x in cc)}
    severe_missed = prim["FULL_A2_severe_truncations_vs_B0"]
    ready_provisional = readiness(valid, len(severe_missed))
    return {"schema": "p2_a2_mech0_secondary_v1", "output_seal_hash": seal["seal_hash"], "integrity": integ, "valid": valid, "systems_FULL300": sysm,
            "groups": groups, "F_ZH_EOS": F_ZH, "F_DEL_EOS": F_DEL, "EOS_group": eos_g, "nonEOS_group": non_g,
            "termination_consistency": {"median_dM2": ms["median"]["dM2"], "dM2_positive_rate": ms["dM2_positive_rate"], "dM2_positive_dialogues": ms["dM2_positive_dialogues"]},
            "label": label, "per_dialogue": per_d, "dialogue_robustness": robust, "concentration": {"ZH": share(pZ), "deletion": share(pDl)}, "lodo": lodo,
            "bootstrap": boot, "conditions_MECH_PANEL": cond, "A2_CONFIRMATION_READY_provisional": ready_provisional,
            "readiness_note": "final readiness additionally requires the independent full post-audit PASS", "per_row": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("primary", "secondary"):
        p = sub.add_parser(nm)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable((primary if args.cmd == "primary" else secondary)(args.run))
    atomic_json(out, res)
    keys = ("valid", "integrity", "step1_vs_step2", "MECHANISM_SENSITIVE_families", "FULL_A2_severe_truncations_vs_B0") if args.cmd == "primary" else \
        ("label", "valid", "integrity", "F_ZH_EOS", "F_DEL_EOS", "EOS_group", "nonEOS_group", "termination_consistency", "A2_CONFIRMATION_READY_provisional")
    print(json.dumps({k: res.get(k) for k in keys}, indent=1))


if __name__ == "__main__":
    main()
