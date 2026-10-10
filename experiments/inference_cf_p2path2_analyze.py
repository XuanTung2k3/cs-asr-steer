#!/usr/bin/env python
"""P2-PATH2 analysis (frozen contract configs/inference_cf/p2_path2.json).

primary   (reference-free) validity: barrier, no-trigger identity, trigger rows/k/prefix and live branches vs audit
          expectations (applied only here, never at runtime), exact PATH1 per-token score/winner reproduction, shared
          G1/G2 decision, ownership locks, compute counts; descriptive distances, lengths, caps, severe flags.
secondary (references behind the committed seal) canonical B0/AUTO/A2/A4/G1/G2 metrics on all 12 and C/U, frozen
          29/105/143/0 cutoffs per G, raw/supported G2 material advantage, terminal label.
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

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.episodic_tta import severe_truncation
from csasr.inference_cf.path_decode import edit_distance, stream

CONFIG = "configs/inference_cf/p2_path2.json"
BASE = "results/inference_cf/p2path2"
PLAN = f"{BASE}/plan_sealed.json"
SEAL = f"{BASE}/output_seal.json"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- frozen outcome rules (pure) -------------------------------------------------------------------------------

def g_pass(counts: dict, caps: int, severe: int, c: dict) -> dict:
    S = c["controller_success"]
    crit = {"ZH_rescue": counts["zh"] <= S["ZH_errors_max"], "POI_retention": counts["poi"] <= S["POI_errors_max"],
            "mixed": counts["mixed"] <= S["mixed_errors_max"], "sequence_safety": caps <= S["caps_max"] and severe <= S["new_severe_truncations_max"]}
    return {**crit, "pass": all(crit.values())}


def raw_material(g1: dict, g2: dict, g2_new_caps_or_severe: int, c: dict) -> bool:
    M = c["G2_material_advantage"]
    return bool((g1["zh"] - g2["zh"] >= M["ZH_errors_fewer_min"] or g1["mixed"] - g2["mixed"] >= M["OR_mixed_errors_fewer_min"])
                and g2["poi"] <= g1["poi"] + M["POI_errors_extra_max"] and g2_new_caps_or_severe <= M["new_caps_or_severe_truncations"])


def decide(valid: bool, g1_pass: bool, g2_pass: bool, raw_adv: bool) -> str:
    if not valid:
        return "P2_PATH2_INVALID"
    if g1_pass and g2_pass and raw_adv:
        return "P2_PATH2_MIXED_GUARD_PROMISING"
    if g1_pass:
        return "P2_PATH2_BRANCH_CONTROL_SUFFICIENT"
    if g2_pass:
        return "P2_PATH2_ROLLBACK_NEEDED"
    return "P2_PATH2_LOCAL_GUARD_INSUFFICIENT"


# ---- load / primary ------------------------------------------------------------------------------------------------

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


def primary(run_rel: str) -> dict:
    c = cfg()
    d = load(ROOT / run_rel)
    rows, plan, rt = d["rows"], d["plan"], d["runtime"]
    integ = {"all_12_ok": len(rows) == 12 and all(r.get("status") == "ok" for r in rows), "runtime_completed": rt.get("status") == "completed",
             "no_runtime_invalid": not rt.get("invalid"), "reset_final": bool(rt.get("reset_final_ok")) and bool(rt.get("nonln_unchanged"))
             and bool(rt.get("model_grads_none"))}
    res = {"schema": "p2_path2_primary_v1", "manifest_hash": d["manifest"]["manifest_hash"], "references_used": False, "integrity": integ}
    if not integ["all_12_ok"]:
        res["valid"] = False
        return res
    per, ok = [], {"barrier": True, "trigger_rows": True, "branches": True, "path1_scores": True, "winners": True, "shared_decision": True,
                   "no_trigger_identity": True, "a4_winner_identity": True, "ownership": True}
    for r, x in zip(rows, plan["rows"]):
        a = x["analysis"]
        ok["barrier"] &= r["theta0_equals_B0"] and r["lang_equals_sealed"] and r["effective_equals_checkpoint_bf16"] and r["G0_equals_sealed_A4"] and r["reset_ok_phase1"]
        ok["ownership"] &= r["detector"]["state_locked"] and r["detector"]["fed_identical"] and r["detector"]["positions_ok"] and r["reset_ok_phase2"]
        trig = r["detector"]["trigger"]
        ok["trigger_rows"] &= trig == (a["group"] == "C")
        dec = r["decision"]
        e = {"utterance_id": a["utterance_id"], "dialogue_id": a["dialogue_id"], "group": a["group"], "trigger": trig}
        if trig:
            exp = a["expected"]
            ok["trigger_rows"] &= dec["k"] == exp["k"] and dec["prefix"] == exp["common_prefix"]
            ok["branches"] &= dec["b0"]["tokens"] == exp["b0"] and dec["b4"]["tokens"] == exp["b4"] and dec["b0"]["H_eff"] == dec["b4"]["H_eff"] == 3
            p1 = json.loads((ROOT / exp["PATH1_score_path"]).read_text())["scores"]
            m = {"theta0_b0": "T0_B", "theta0_b4": "T0_ALT", "A4_b0": "A4_B", "A4_b4": "A4_ALT"}
            ok["path1_scores"] &= all(dec["scores"][k]["logprobs"] == p1[v]["logprobs"] and dec["scores"][k]["score"] == p1[v]["score"] for k, v in m.items())
            p1row = next(z for z in json.loads((ROOT / "results/inference_cf/p2path1/primary_analysis.json").read_text())["branch_scores"]
                         if z["utterance_id"] == a["utterance_id"])
            ok["winners"] &= dec["winner"] == ("theta0" if p1row["choice"] == "B" else "A4") and dec["margin"] == p1row["margin_B_minus_ALT"]
            ok["shared_decision"] &= r["G1"]["decision_hash"] == r["G2"]["decision_hash"] == dec["decision_hash"]
            ok["ownership"] &= all(v["state_locked"] for v in r["score_paths"].values()) and r["G1"]["state_locked"] and r["G2"].get("state_locked", True)
            if dec["winner"] == "A4":
                ok["a4_winner_identity"] &= r["G1"]["tokens"] == r["G0"]["tokens"] == r["G2"]["tokens"] and r["G1"]["terminated"] == r["G0"]["terminated"]
            e.update(k=dec["k"], b0=dec["b0"]["tokens"], b4=dec["b4"]["tokens"], winner=dec["winner"], margin=dec["margin"],
                     S={kk: v["score"] for kk, v in dec["scores"].items()}, S_cons=dec["S_cons"])
        else:
            ok["no_trigger_identity"] &= r["G1"] == r["G0"] == r["G2"]
        B0, A4, AU = stream(a["B0"]["tokens"], a["B0"]["terminated"]), stream(a["A4"]["tokens"], a["A4"]["terminated"]), stream(a["AUTO"]["tokens"], a["AUTO"]["terminated"])
        for g in ("G1", "G2"):
            G = stream(r[g]["tokens"], r[g]["terminated"])
            e[g] = {"changed_vs_A4": r[g]["tokens"] != a["A4"]["tokens"] or r[g]["terminated"] != a["A4"]["terminated"], "d_B0": edit_distance(G, B0),
                    "d_A4": edit_distance(G, A4), "d_AUTO": edit_distance(G, AU), "length": len(r[g]["tokens"]), "terminated": r[g]["terminated"],
                    "severe_vs_B0": severe_truncation(len(a["B0"]["tokens"]), len(r[g]["tokens"]), r[g]["terminated"]),
                    "severe_vs_A4": severe_truncation(len(a["A4"]["tokens"]), len(r[g]["tokens"]), r[g]["terminated"])}
        per.append(e)
    cnt = rt["counters"]
    n_trig = sum(e["trigger"] for e in per)
    n_t0w = sum(e.get("winner") == "theta0" for e in per)
    B = c["compute"]
    ok["compute"] = (cnt["A4"]["optimizer_steps"] == B["A4_optimizer_steps"] and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= B["backwards_max_including_live_check"]
                     and cnt["triggers"] == n_trig <= B["guard_events_max"] and cnt["rollouts"] == 2 * n_trig and cnt["score_paths"] == 4 * n_trig
                     and cnt["G1_decodes"] == n_trig and cnt["G2_decodes"] == n_t0w and cnt["G0_decodes"] == 12 and cnt["theta0_decodes"] == 12)
    integ.update(ok)
    valid = all(integ.values())
    res.update(valid=valid, per_row=per, triggered=n_trig, no_trigger=12 - n_trig, theta0_winners=n_t0w, A4_winners=n_trig - n_t0w,
               G1_changed_vs_A4=sum(e["G1"]["changed_vs_A4"] for e in per), G2_changed_vs_A4=sum(e["G2"]["changed_vs_A4"] for e in per),
               distances={g: {t: sum(e[g][f"d_{t}"] for e in per) for t in ("B0", "A4", "AUTO")} for g in ("G1", "G2")},
               caps={g: sum(e[g]["terminated"] == "cap" for e in per) for g in ("G1", "G2")},
               severe={g: sum(e[g]["severe_vs_B0"] or e[g]["severe_vs_A4"] for e in per) for g in ("G1", "G2")},
               runtime={k: rt.get(k) for k in ("job_id", "elapsed_sec", "setup_sec", "phase1_sec", "phase2_sec", "peak_alloc", "peak_reserved", "counters", "gpu")})
    return res


# ---- secondary -----------------------------------------------------------------------------------------------------

def secondary(run_rel: str) -> dict:
    from transformers import WhisperProcessor
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import outside, utt_counts
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("PATH2 output seal missing/uncommitted: references stay closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(sha(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed PATH2 outputs changed")
    if not committed(f"{BASE}/primary_audit.json") or json.loads((ROOT / BASE / "primary_audit.json").read_text())["verdict"] != "P2_PATH2_AUDIT: PASS":
        raise PermissionError("primary-phase audit PASS required before references")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    d = load(ROOT / run_rel)
    from experiments.inference_cf_p2_evaluate import load_references
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    refs = load_references()
    dec = lambda t: tok.decode(t, skip_special_tokens=True)
    rows = []
    for r, x in zip(d["rows"], d["plan"]["rows"]):
        a = x["analysis"]
        u = a["utterance_id"]
        a4row = json.loads((ROOT / a["A4"]["path"]).read_text())["A4"]
        H = {"B0": a["y_B_text"], "AUTO": a["y_A_text"], "A2": a["A2"]["text"], "A4": a4row["text"], "G1": dec(r["G1"]["tokens"]), "G2": dec(r["G2"]["tokens"])}
        term = {"B0": a["B0"]["terminated"], "AUTO": a["AUTO"]["terminated"], "A2": a["A2"]["terminated"], "A4": a["A4"]["terminated"],
                "G1": r["G1"]["terminated"], "G2": r["G2"]["terminated"]}
        rows.append({"utterance_id": u, "group": a["group"], "ref": refs[u]["reference"], "H": H, "term": term,
                     "counts": {k: utt_counts(refs[u]["reference"], h) for k, h in H.items()}})
    systems = ("B0", "AUTO", "A2", "A4", "G1", "G2")

    def agg(rs):
        out = {}
        R = [x["ref"] for x in rs]
        for s_ in systems:
            Hs = [x["H"][s_] for x in rs]
            cm = corpus_metrics(R, Hs)
            tot = {k: sum(x["counts"][s_][i] for x in rs) for i, k in ((0, "poi"), (2, "mixed"), (4, "zh"))}
            e = {**tot, "pier": cm["pier"], "mer": cm["mer"], "en_wer": cm["en_wer"], "zh_cer": cm["zh_cer"], "substitutions": cm["substitutions"],
                 "deletions": cm["deletions"], "insertions": cm["insertions"], "caps": sum(x["term"][s_] == "cap" for x in rs)}
            if s_ != "B0":
                Bs = [x["H"]["B0"] for x in rs]
                cc = correction_corruption(R, Bs, Hs)
                outs = [outside(r_, b, h) for r_, b, h in zip(R, Bs, Hs)]
                ob = sum(o["baseline_correct_outside"] for o in outs)
                e.update(corrections_vs_B0=cc["corrections"], corruptions_vs_B0=cc["corruptions"], zh_retention_vs_B0=matrix_zh_retention(R, Bs, Hs)["rate"],
                         en_retention_vs_B0=embedded_en_retention(R, Bs, Hs)["rate"], outside_harm_vs_B0=(sum(o["outside_harm"] for o in outs) / ob) if ob else None)
            if s_ in ("G1", "G2"):
                As = [x["H"]["A4"] for x in rs]
                e.update(zh_retention_vs_A4=matrix_zh_retention(R, As, Hs)["rate"], en_retention_vs_A4=embedded_en_retention(R, As, Hs)["rate"])
            out[s_] = e
        return out
    allm = agg(rows)
    crit, newbad = {}, {}
    for g in ("G1", "G2"):
        pe = {e["utterance_id"]: e for e in prim["per_row"]}
        sev = sum(pe[x["utterance_id"]][g]["severe_vs_B0"] or pe[x["utterance_id"]][g]["severe_vs_A4"] for x in rows)
        crit[g] = g_pass(allm[g], allm[g]["caps"], sev, c)
        newbad[g] = allm[g]["caps"] + sev
    raw = raw_material(allm["G1"], allm["G2"], newbad["G2"], c)      # G2 caps + severe flags vs B0/A4 (baselines have none)
    label = decide(bool(prim["valid"]), crit["G1"]["pass"], crit["G2"]["pass"], raw)
    return {"schema": "p2_path2_secondary_v1", "output_seal_hash": seal["seal_hash"], "metrics_all12": allm,
            "metrics_C": agg([x for x in rows if x["group"] == "C"]), "metrics_U": agg([x for x in rows if x["group"] == "U"]),
            "criteria": crit, "raw_G2_material_advantage": raw, "supported_G2_extra": bool(crit["G2"]["pass"] and raw), "label": label,
            "per_row": [{"utterance_id": x["utterance_id"], "group": x["group"], "counts": x["counts"], "term": x["term"]} for x in rows]}


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
        print(json.dumps({k: res.get(k) for k in ("valid", "integrity", "triggered", "no_trigger", "theta0_winners", "A4_winners", "G1_changed_vs_A4",
                                                  "G2_changed_vs_A4", "distances", "caps", "severe")}, indent=1))
    else:
        print(json.dumps({k: res[k] for k in ("label", "criteria", "raw_G2_material_advantage", "supported_G2_extra")}, indent=1))
        print(json.dumps({s: {k: v[k] for k in ("zh", "poi", "mixed", "pier", "mer", "en_wer", "zh_cer")} for s, v in res["metrics_all12"].items()}, indent=1))


if __name__ == "__main__":
    main()
