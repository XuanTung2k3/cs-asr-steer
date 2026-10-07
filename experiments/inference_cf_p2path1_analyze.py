#!/usr/bin/env python
"""P2-PATH1 analysis (frozen contract configs/inference_cf/p2_path1.json).

primary   (reference-free) suffix endpoints at r = k+1 for T0A/A4A, inherited PATH0 INDUCE_1 (exact arithmetic reused from
          the PATH0 primary analysis, as the design permits), I_theta0/I_A4/Delta_state, frozen materiality, mechanism
          precedence; H=3 branch scores, 0.5/0.5 consensus, tie->B, CONSENSUS_REJECTS_AUTO. Written BEFORE the seal.
secondary (references, behind the committed seal) U canonical errors for B0/AUTO/T0A/A4A + ASR_STATE_ALIGNMENT; C score
          choices alongside sealed B0/FREE-A4 errors (descriptive). Never changes the primary label.
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

from csasr.inference_cf.branch_adjudication import consensus
from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.path_decode import edit_distance, stream
from experiments.inference_cf_p2path0_analyze import criterion

CONFIG = "configs/inference_cf/p2_path1.json"
BASE = "results/inference_cf/p2path1"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"
G = 1e-12


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def path0_cfg() -> dict:
    """Inherited PATH0 criterion block (exact copy in the PATH1 config)."""
    c = cfg()
    return {"suffix_decision": c["inherited_INDUCE_1"]}


# ---- pure decomposition / mechanism ----------------------------------------------------------------------------------

def endpoints(r: int, B0: list, AUTO: list, T0A: list, A4A: list) -> dict:
    sl = lambda x: list(x)[r:]
    d_BA = edit_distance(sl(B0), sl(AUTO))
    out = {"d_BA": d_BA, "eligible": d_BA > 0}
    for tag, X in (("theta0", T0A), ("A4", A4A)):
        dA, dB = edit_distance(sl(X), sl(AUTO)), edit_distance(sl(X), sl(B0))
        out[tag] = {"d_A": dA, "d_B": dB, "I": (1 - dA / d_BA) if d_BA else None}
    if out["eligible"]:
        out["Delta"] = out["A4"]["I"] - out["theta0"]["I"]
        out["tau"] = max(0.25, 1 / d_BA)
    return out


def mechanism(rows: list[dict], c: dict, t0a_pass: bool, a4a_pass: bool, valid: bool) -> dict:
    T = c["state_thresholds"]
    el = [r for r in rows if r["eligible"]]
    sd = sum(r["d_BA"] for r in el)
    I_t0 = 1 - sum(r["theta0"]["d_A"] for r in el) / sd if sd else None
    I_a4 = 1 - sum(r["A4"]["d_A"] for r in el) / sd if sd else None
    D = I_a4 - I_t0 if sd else None
    material = [r for r in el if r["Delta"] >= r["tau"] - G]
    approx = [r for r in el if abs(r["Delta"]) < r["tau"]]
    t0_stronger = [r for r in el if r["Delta"] <= -r["tau"] + G]
    t0_return = [r for r in el if r["theta0"]["I"] <= T["adapted_theta0_pooled_max"] + G and r["theta0"]["d_B"] <= max(1, math.floor(0.25 * r["d_BA"]))]
    t0_succ = [r for r in el if r["theta0"]["I"] >= 0.5 - G and r["theta0"]["d_B"] > 0]
    a4_succ = [r for r in el if r["A4"]["I"] >= 0.5 - G and r["A4"]["d_B"] > 0]
    dl = lambda rs: len({r["dialogue_id"] for r in rs})
    prefix = t0a_pass and D <= T["prefix_delta_pooled_max"] + G and len(material) <= T["prefix_material_A4_rows_max"]
    adapted = (not t0a_pass and I_t0 <= T["adapted_theta0_pooled_max"] + G and len(t0_succ) <= T["adapted_theta0_successes_max"]
               and D >= T["adapted_delta_pooled_min"] - G and len(material) >= T["adapted_material_A4_rows_min"]
               and dl(material) >= T["adapted_material_dialogues_min"])
    mean_rows = [r for r in el if r["theta0"]["I"] >= T["mixed_theta0_row_I_min"] - G and r["theta0"]["d_B"] > 0]
    mixed_b = I_t0 >= T["mixed_meaningful_theta0_pooled_min"] - G and len(mean_rows) >= T["mixed_theta0_rows_min"] and dl(mean_rows) >= T["mixed_theta0_dialogues_min"]
    t0_ids = {r["utterance_id"] for r in t0_succ}
    state_req = [r for r in a4_succ if r["utterance_id"] not in t0_ids and r["Delta"] >= r["tau"] - G]
    mixed_c = (len(t0_succ) >= T["mixed_heterogeneous_prefix_successes_min"] and dl(t0_succ) >= 2
               and len(state_req) >= T["mixed_heterogeneous_state_required_rows_min"] and dl(state_req) >= 2)
    if not valid or not a4a_pass:
        lab = "P2_PATH1_INVALID"
    elif prefix:
        lab = "P2_PATH1_PREFIX_DOMINANT"
    elif adapted:
        lab = "P2_PATH1_ADAPTED_STATE_DOMINANT"
    elif t0a_pass or mixed_b or mixed_c:
        lab = "P2_PATH1_MIXED"
    else:
        lab = "P2_PATH1_NO_CLEAR_MECHANISM"
    ids = lambda rs: [r["utterance_id"] for r in rs]
    return {"I_theta0_pooled": I_t0, "I_A4_pooled": I_a4, "Delta_state_pooled": D, "sum_d_BA": sd,
            "material_A4_rows": ids(material), "approx_equal_rows": ids(approx), "theta0_stronger_rows": ids(t0_stronger),
            "theta0_return_B0_rows": ids(t0_return), "T0A_row_successes": ids(t0_succ), "A4A_row_successes": ids(a4_succ),
            "state_required_rows": ids(state_req), "predicates": {"PREFIX_DOMINANT": prefix, "ADAPTED_STATE_DOMINANT": adapted,
                                                                  "MIXED_a_T0A_pass": t0a_pass, "MIXED_b_meaningful_prefix": mixed_b,
                                                                  "MIXED_c_heterogeneous": mixed_c}, "label": lab}


def consensus_rejects_auto(u_scores: list[dict], c: dict) -> dict:
    """u_scores: eligible U rows [{'dialogue_id', 'choice', 'strict_B_win'}]."""
    R = c["branch_adjudication"]["CONSENSUS_REJECTS_AUTO"]
    b = [x for x in u_scores if x["choice"] == "B"]
    sw = [x for x in u_scores if x["strict_B_win"]]
    ok = len(b) >= R["B_choices_min"] and len(sw) >= R["strict_B_wins_min"] and len({x["dialogue_id"] for x in sw}) >= R["strict_B_win_dialogues_min"]
    return {"B_choices": len(b), "strict_B_wins": len(sw), "strict_B_win_dialogues": len({x["dialogue_id"] for x in sw}), "pass": bool(ok)}


# ---- primary -------------------------------------------------------------------------------------------------------

def load(run_dir: Path) -> dict:
    man = json.loads((run_dir / "manifest.json").read_text())
    if digest({k: v for k, v in man.items() if k != "manifest_hash"}) != man["manifest_hash"]:
        raise ValueError("manifest")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != man["plan_hash"]:
        raise ValueError("plan")
    rows = [json.loads((run_dir / f"rows/{i:02d}.json").read_text()) if (run_dir / f"rows/{i:02d}.json").exists() else {"status": "missing"}
            for i in range(len(plan["rows"]))]
    return {"manifest": man, "plan": plan, "rows": rows, "runtime": json.loads((run_dir / "runtime.json").read_text())}


def integrity(d: dict) -> dict:
    rows, rt, plan = d["rows"], d["runtime"], d["plan"]
    c = {"all_12_ok": len(rows) == 12 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
         "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
         and bool(rt.get("model_grads_none"))}
    if not c["all_12_ok"]:
        return c
    U = [(r, p) for r, p in zip(rows, plan["rows"]) if p["population"] == "U_PRIMARY"]
    c["phase1_barrier"] = all(r["theta0_equals_B0"] and r["lang_equals_sealed"] and r["effective_equals_checkpoint_bf16"] and r["FREE_equals_sealed_A4"]
                              and r["reset_ok_phase1"] for r in rows)
    c["factorial_identities"] = all(r["T0B_equals_B0"] and r["A4B_equals_B0"] and r["A4A_equals_PATH0_L1"] for r, _ in U)
    c["state_ownership"] = all(v["state_locked"] and v["owner"]["state_hash"] == (rt["theta0_ln_hash"] if k.startswith("T0") else r["A4_state_hash"])
                               for r, _ in U for k, v in r["paths"].items()) and \
        all(v["state_locked"] and v["owner"]["state_hash"] == (rt["theta0_ln_hash"] if k.startswith("T0") else r["A4_state_hash"])
            for r in rows for k, v in r["scores"].items())
    c["path_traces"] = all(v["trace"]["prefix_ok"] and v["trace"]["suppression_ok"] and v["trace"]["positions_ok"] and len(v["trace"]["forced"]) == 1
                           and v["trace"]["forced"][0]["token"] == (p["c_B"] if k.endswith("B") else p["c_ALT"]) for r, p in U for k, v in r["paths"].items())
    c["score_traces"] = all(v["prefix_ok"] and v["positions_ok"] and v["fed_ok"] and v["H_eff"] == p["score_branches"][k.split("_")[1]]["H_eff"]
                            and all(math.isfinite(x) for x in v["logprobs"]) for r, p in zip(rows, plan["rows"]) for k, v in r["scores"].items())
    c["live_audit"] = bool(rows[0].get("live_audit", {}).get("pass"))
    cnt, B = rt["counters"], cfg()["compute"]
    c["compute"] = (cnt["A4"]["optimizer_steps"] == B["optimizer_steps"] and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max"]
                    and cnt["factorial_decodes"] == B["primary_factorial_U_decodes"] and cnt["free_decodes"] == B["FREE_A4_reproduction_decodes"]
                    and cnt["theta0_decodes"] == B["theta0_B0_reproduction_decodes"] and cnt["score_paths"] == B["score_paths"]
                    and cnt["scored_terms"] <= B["scored_token_logprob_terms_max"])
    return c


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    integ = integrity(d)
    res = {"schema": "p2_path1_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "integrity": integ, "references_used": False}
    if not integ["all_12_ok"]:
        res["label"] = "P2_PATH1_INVALID"
        return res
    u_rows, inh = [], {"theta0": [], "A4": []}
    score_rows = []
    for r, p in zip(d["rows"], d["plan"]["rows"]):
        sc = {m: {b: r["scores"][f"{t}_{b}"]["score"] for b in ("B", "ALT")} for m, t in (("theta0", "T0"), ("A4", "A4"))}
        cs = consensus(sc["theta0"], sc["A4"])
        score_rows.append({"utterance_id": p["utterance_id"], "dialogue_id": p["dialogue_id"], "population": p["population"],
                           "eligible": p["L1"]["suffix_eligible"], "S": sc, **cs,
                           "logprobs": {k: v["logprobs"] for k, v in r["scores"].items()}, "H_eff": {b: p["score_branches"][b]["H_eff"] for b in ("B", "ALT")}})
        if p["population"] != "U_PRIMARY":
            continue
        B0, AU = stream(p["y_B"], p["y_B_terminated"]), stream(p["y_A"], p["y_A_terminated"])
        T0A = stream(r["paths"]["T0A"]["tokens"], r["paths"]["T0A"]["terminated"])
        A4A = stream(r["paths"]["A4A"]["tokens"], r["paths"]["A4A"]["terminated"])
        e = endpoints(p["site"] + 1, B0, AU, T0A, A4A)
        e.update(utterance_id=p["utterance_id"], dialogue_id=p["dialogue_id"], sealed_d_BA_matches=e["d_BA"] == p["L1"]["suffix_baseline_distance"])
        u_rows.append(e)
        for tag in ("theta0", "A4"):
            inh[tag].append({"dialogue_id": p["dialogue_id"], "d0": e["d_BA"], "d_target": e[tag]["d_A"], "dB": e[tag]["d_B"], "eligible": e["eligible"],
                             "R": e[tag]["I"]})
    integ["sealed_denominators"] = all(e["sealed_d_BA_matches"] for e in u_rows) and sum(e["d_BA"] for e in u_rows if e["eligible"]) == 48
    crit = {f"{t}A_INDUCE_1": criterion("INDUCE", inh[t], path0_cfg()) for t in ("theta0", "A4")}
    p0 = json.loads((ROOT / "results/inference_cf/p2path0/primary_analysis.json").read_text())
    integ["A4A_matches_PATH0_INDUCE_1"] = (crit["A4A_INDUCE_1"]["pass"] is True and crit["A4A_INDUCE_1"]["successes"] == p0["criteria"]["INDUCE_1"]["successes"]
                                           and abs(crit["A4A_INDUCE_1"]["pooled_reduction"] - p0["criteria"]["INDUCE_1"]["pooled_reduction"]) <= G)
    valid = all(integ.values())
    mech = mechanism(u_rows, c, crit["theta0A_INDUCE_1"]["pass"], crit["A4A_INDUCE_1"]["pass"], valid)
    cra = consensus_rejects_auto([x for x in score_rows if x["population"] == "U_PRIMARY" and x["eligible"]], c)
    by_dlg = {}
    for e in u_rows:
        if e["eligible"]:
            g = by_dlg.setdefault(e["dialogue_id"], {"rows": [], "d_BA": 0, "d_theta0A": 0, "d_A4A": 0})
            g["rows"].append(e["utterance_id"])
            g["d_BA"] += e["d_BA"]
            g["d_theta0A"] += e["theta0"]["d_A"]
            g["d_A4A"] += e["A4"]["d_A"]
    for g in by_dlg.values():
        g["I_theta0"], g["I_A4"] = 1 - g["d_theta0A"] / g["d_BA"], 1 - g["d_A4A"] / g["d_BA"]
        g["Delta"] = g["I_A4"] - g["I_theta0"]
    res.update(valid=valid, label=mech["label"], criteria=crit, mechanism=mech, per_row=u_rows, by_dialogue=by_dlg,
               branch_scores=score_rows, CONSENSUS_REJECTS_AUTO=cra,
               C_choices={"B0": sum(x["choice"] == "B" for x in score_rows if x["population"] == "C_SCORE_ONLY"),
                          "A4_alt": sum(x["choice"] == "ALT" for x in score_rows if x["population"] == "C_SCORE_ONLY")},
               U_choices={"B0": sum(x["choice"] == "B" for x in score_rows if x["population"] == "U_PRIMARY"),
                          "AUTO": sum(x["choice"] == "ALT" for x in score_rows if x["population"] == "U_PRIMARY")},
               runtime={k: d["runtime"].get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "phase1_sec", "phase2_sec", "peak_alloc", "peak_reserved",
                                                         "counters", "gpu")})
    return res


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from transformers import WhisperProcessor
    from csasr.evaluation.canonical import corpus_metrics
    from experiments.inference_cf_p2seq_analyze import utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH1 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH1 outputs changed")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    a4plan = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())["rows"]}
    refs = load_references()
    dec = lambda t: tok.decode(t, skip_special_tokens=True)
    U, C = [], []
    for r, p in zip(d["rows"], d["plan"]["rows"]):
        u = p["utterance_id"]
        if p["population"] == "U_PRIMARY":
            H = {"B0": a4plan[u]["y_B_text"], "AUTO": a4plan[u]["y_A_text"], "T0A": dec(r["paths"]["T0A"]["tokens"]), "A4A": dec(r["paths"]["A4A"]["tokens"])}
            U.append({"utterance_id": u, "dialogue_id": p["dialogue_id"], "eligible": p["L1"]["suffix_eligible"], "ref": refs[u]["reference"], "H": H,
                      "counts": {k: utt_counts(refs[u]["reference"], h) for k, h in H.items()},
                      "length": {"B0": len(p["y_B"]), "AUTO": len(p["y_A"]), "T0A": len(r["paths"]["T0A"]["tokens"]), "A4A": len(r["paths"]["A4A"]["tokens"])},
                      "terminated": {"B0": p["y_B_terminated"], "AUTO": p["y_A_terminated"], "T0A": r["paths"]["T0A"]["terminated"],
                                     "A4A": r["paths"]["A4A"]["terminated"]}})
        else:
            sr = next(x for x in prim["branch_scores"] if x["utterance_id"] == u)
            H = {"B0": a4plan[u]["y_B_text"], "FREE_A4": dec(r["FREE"]["tokens"])}
            C.append({"utterance_id": u, "choice": sr["choice"], "margin_B_minus_ALT": sr["margin_B_minus_ALT"],
                      "counts": {k: utt_counts(refs[u]["reference"], h) for k, h in H.items()}})
    keys = ("poi_errors", "num_poi", "mixed_errors", "ref_units", "zh_errors", "zh_ref", "en_errors", "en_ref")

    def agg(rs, s_):
        tot = {k: sum(x["counts"][s_][i] for x in rs) for i, k in enumerate(keys)}
        cm = corpus_metrics([x["ref"] for x in rs], [x["H"][s_] for x in rs])
        return {**tot, "substitutions": cm["substitutions"], "deletions": cm["deletions"], "insertions": cm["insertions"], "mer": cm["mer"],
                "zh_cer": cm["zh_cer"], "pier": cm["pier"]}
    aggregate = {s_: agg(U, s_) for s_ in ("B0", "AUTO", "T0A", "A4A")}
    el = [x for x in U if x["eligible"]]
    h = {m: sum(x["counts"][m][4] - x["counts"]["B0"][4] for x in el) for m in ("T0A", "A4A")}
    rows_up = {m: sum(x["counts"][m][4] > x["counts"]["B0"][4] for x in el) for m in ("T0A", "A4A")}
    harm = {m: h[m] >= 2 and rows_up[m] >= 2 for m in ("T0A", "A4A")}
    if harm["T0A"] and harm["A4A"] and h["A4A"] - h["T0A"] <= 2:
        lab = "PREFIX"
    elif harm["A4A"] and not harm["T0A"] and h["A4A"] - h["T0A"] >= 3:
        lab = "ADAPTED_STATE"
    elif harm["T0A"] or harm["A4A"]:
        lab = "MIXED"
    else:
        lab = "NONE"
    poi = {m: sum(x["counts"][m][0] - x["counts"]["B0"][0] for x in el) for m in ("T0A", "A4A", "AUTO")}
    return {"schema": "p2_path1_secondary_v1", "output_seal_hash": seal["seal_hash"], "primary_label": prim["label"], "aggregate_U": aggregate,
            "ASR_STATE_ALIGNMENT": lab, "zh_excess_vs_B0_eligible": h, "rows_increasing_zh": rows_up, "harm": harm, "poi_change_vs_B0_eligible": poi,
            "per_row_U": [{k: v for k, v in x.items() if k != "ref"} for x in U], "C_descriptive": C}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("primary", "secondary"):
        p = sub.add_parser(n)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    res = jsonable((primary if args.cmd == "primary" else secondary)(args.run))
    atomic_json(out, res)
    if args.cmd == "primary":
        print(json.dumps({k: res.get(k) for k in ("label", "integrity", "criteria", "mechanism", "CONSENSUS_REJECTS_AUTO", "U_choices", "C_choices")}, indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("primary_label", "ASR_STATE_ALIGNMENT", "zh_excess_vs_B0_eligible", "rows_increasing_zh",
                                              "poi_change_vs_B0_eligible", "aggregate_U")}, indent=1))


if __name__ == "__main__":
    main()
