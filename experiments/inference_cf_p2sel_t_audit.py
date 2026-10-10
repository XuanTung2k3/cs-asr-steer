#!/usr/bin/env python
"""Independent P2-SEL-T auditor. Does NOT import inference_cf_p2sel_t_analyze (or any analysis module)
nor the runner's feature helpers: crop bounds, fractional-frame attention masses, E, j*, E_tok/E_ctx/C,
historical groups, shared-draw bootstrap, predicates and label are re-implemented here.

prerun -> PASS_TO_P2_SEL_T_T0 | BLOCK_BEFORE_P2_SEL_T_T0
t0     -> P2_SEL_T_AUDIT: PASS (T0)
final  -> P2_SEL_T_AUDIT: PASS
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

from experiments.inference_cf_p2dir_audit import fhash, git_blob_hash

FREEZE = "fdace84f1ef7b241f42b5b029a7fd1201f68c118"
CONFIG = "configs/inference_cf/p2_sel_t.json"
BASE = ROOT / "results/inference_cf/p2sel_t"
E0 = ROOT / "results/inference_cf/p2sel_e/e0_run1"
FORBIDDEN = ("target_ids", "competitor", "timing_error", "E_oracle", "acoustic", "stratum", "group", "positions.json", "C_tokctx")


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cfg():
    return json.loads((ROOT / CONFIG).read_text())


def runtime_surface_clean() -> dict:
    """E_tok construction (bounds/mass/selection/T0 replay) never reads evaluator/group/reference/contrast."""
    src = (ROOT / "experiments/inference_cf_p2sel_t.py").read_text()
    res = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name in ("candidate_bounds", "frame_mean", "crop_mass", "select", "t0_utterance"):
            seg = ast.get_source_segment(src, n)
            res[n.name] = not any(f in seg for f in FORBIDDEN)
    return {"ok": len(res) == 5 and all(res.values()), "functions": res}


def bounds_ind(s0, s1):
    n = min(8000, s1 - s0)
    m = (s0 + s1 - n) // 2
    return {"L": [s0, s0 + n], "C": [m, m + n], "R": [s1 - n, s1]}


def mass_ind(mean, a, b, heard):
    heard = min(heard, 480000)
    tot = 0.0
    for f in range(len(mean)):
        lo_, hi_ = 320 * f, min(320 * (f + 1), heard)
        if hi_ <= lo_:
            continue
        ov = max(0, min(b, hi_) - max(a, lo_))
        tot += float(mean[f]) * ov / (hi_ - lo_)
    return tot


def E_ind(pe, pm, ne, nm):
    eps = 1e-12
    A = math.log((pe + eps) / (pm + eps)) - math.log((ne + eps) / (nm + eps))
    return max(0.0, math.tanh(A / 2))


def cmd_prerun(args) -> dict:
    c = cfg()
    checks, notes = {}, {}
    for rel in ("docs/inference_cf/P2_SEL_T_SPEC.md", "docs/inference_cf/P2_SEL_T_CODEX_DESIGN.md", CONFIG):
        checks[f"frozen:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    pv = c["provider"]
    gp = Path(pv["generation_config_path"])
    lt = json.loads(gp.read_text())["lang_to_id"]
    pay = json.dumps(sorted((str(k), int(v)) for k, v in lt.items()), separators=(",", ":"), ensure_ascii=False)
    checks["provider_identity"] = (hashlib.sha256(gp.read_bytes()).hexdigest() == pv["generation_config_sha256"]
                                   and hashlib.sha256(pay.encode()).hexdigest() == pv["canonical_mapping_sha256"]
                                   and len(set(lt.values())) == 100 and [lt["<|en|>"], lt["<|zh|>"]] == pv["en_zh_token_ids"])
    heads = json.loads(gp.read_text()).get("alignment_heads")
    checks["alignment_heads_present"] = bool(heads)
    notes["alignment_heads"] = heads
    e0a = json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_audit.json").read_text())
    e0an = json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_analysis.json").read_text())
    fa = json.loads((ROOT / "results/inference_cf/p2sel_e/final_audit.json").read_text())
    checks["parent_terminal_pass"] = (e0a["verdict"] == "P2_SEL_E_AUDIT: PASS" and e0a["label"] == c["parent_terminal"]
                                      and fa["verdict"] == "P2_SEL_E_AUDIT: PASS")
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    checks["population"] = len(pos["positions"]) == 180 and len(pos["utterances"]) == 80 and \
        len({p["dialogue_id"] for p in pos["positions"]}) == 20 and \
        [(p["utterance_id"], p["t"]) for p in pos["positions"]] == [(p["utterance_id"], p["t"]) for p in con["positions"]]
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in
           json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    grp = {}
    for p in pos["positions"]:
        e = sel[(p["utterance_id"], p["t"])]
        g = {"EN-confusion": "EN_TP" if e > 0 else "EN_FN", "ZH-correct": "ZH_FP" if e > 0 else "ZH_TN"}.get(p["stratum"], "EN_CORRECT")
        grp[g] = grp.get(g, 0) + 1
    checks["historical_groups"] = grp == c["population"]["historical_groups"]
    # CENTER compatibility: frozen CENTER bounds equal the stored E0 short bounds for every exposed key
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    bad, rows = [], {(r["utterance_id"], r["t"]): r for r in e0an["rows"]}
    for p in pos["positions"]:
        e = json.loads((E0 / f"rows/{cidx[p['utterance_id']]:03d}.json").read_text())["positions"][str(p["t"])]
        if bounds_ind(e["window"][2], e["window"][3])["C"] != e["short"]["bounds"]:
            bad.append((p["utterance_id"], p["t"]))
        if e["window"] != rows[(p["utterance_id"], p["t"])]["window"]:
            bad.append((p["utterance_id"], p["t"], "window"))
    checks["center_equals_e0_short_180"] = not bad
    null = json.loads((E0 / "null.json").read_text())
    checks["null_reuse"] = null["n_probs"] == 100 and abs(null["softmax_sum"] - 1) <= 1e-6 and null["samples"] == 480000
    checks["mini_panel"] = fhash(ROOT / c["T2"]["panel"]) == "sha256:" + c["T2"]["panel_sha256"]
    checks["d2_sealed"] = fhash(ROOT / "results/inference_cf/p2dir/exp1_run1/directions_sealed.json") == \
        "sha256:" + c["source_sha256"]["results/inference_cf/p2dir/exp1_run1/directions_sealed.json"]
    checks["no_T_outcome"] = not BASE.exists() or not any(x.name.startswith(("t0", "t1", "t2")) for x in BASE.iterdir())
    checks["runtime_surface_reference_free"] = runtime_surface_clean()["ok"]
    asrc = (ROOT / "experiments/inference_cf_p2sel_t_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|inference_cf_p2sel_t\b)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2sel_t.py").exists()
    checks["compute_caps"] = c["compute"]["T0_native_lid_calls_max"] == 360 and c["compute"]["T0_autograd_calls"] == 0
    verdict = "PASS_TO_P2_SEL_T_T0" if all(checks.values()) else "BLOCK_BEFORE_P2_SEL_T_T0"
    return {"schema": "p2_sel_t_prerun_audit_v1", "verdict": verdict, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


def _diff(rows_a, rows_b, fa, fb, keys, idx):
    def per(rows, f):
        acc = {}
        for r in rows:
            acc.setdefault(r["d"], []).append(float(f(r)))
        return {k: sum(v) / len(v) for k, v in acc.items()}
    A, B = per(rows_a, fa), per(rows_b, fb)
    ds = []
    for draw in idx:
        ks = [keys[j] for j in draw]
        va, vb = [A[k] for k in ks if k in A], [B[k] for k in ks if k in B]
        if va and vb:
            ds.append(sum(va) / len(va) - sum(vb) / len(vb))
    pt = sum(A.values()) / len(A) - sum(B.values()) / len(B) if A and B else None
    return pt, (float(np.quantile(ds, 0.20)) if ds else None), len(ds)


def cmd_t0(args) -> dict:
    c = cfg()
    ana = json.loads((ROOT / args.analysis).read_text())
    run = BASE / "t0_run1"
    man = json.loads((run / "manifest.json").read_text())
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    null = json.loads((E0 / "null.json").read_text())
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in
           json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    p2r_pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(p2r_pop["utterances"])}
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    checks = {"manifest_sources_at_commit": all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == man["manifest_hash"]}
    rows, fails = [], []
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        x = json.loads((run / f"rows/{i:03d}.json").read_text())["positions"][str(t)]
        e0 = json.loads((E0 / f"rows/{i:03d}.json").read_text())["positions"][str(t)]
        st = {s["query"]: s for s in json.loads((ROOT / f"results/inference_cf/p2r/run3/rows/{pidx[uid]:03d}.json").read_text())["current"]["steps"]}
        with np.load(run / f"rows/{i:03d}.npz") as z:
            mean = z[f"t{t}_mean"].astype(np.float64)
        W = x["window"]
        b = bounds_ind(W[2], W[3])
        heard = x["heard_samples"]
        a = {k: mass_ind(mean, v[0], v[1], heard) for k, v in b.items()}
        probs = dict(x["probs"])
        probs["C"] = {"pi_E": e0["short"]["pi_E"], "pi_M": e0["short"]["pi_M"]}      # reuse identity checked below
        E = {k: E_ind(probs[k]["pi_E"], probs[k]["pi_M"], null["pi_E"], null["pi_M"]) for k in "LCR"}
        j = max("LCR", key=lambda k: (a[k], -"LCR".index(k)))
        o = [k for k in "LCR" if k != j]
        e1 = E_ind(e0["long"]["pi_E"], e0["long"]["pi_M"], null["pi_E"], null["pi_M"])
        s = st[p["p2r"]["query"]]
        ok = (W == e0["window"] == s["window"] and x["query"] == p["p2r"]["query"] and b["C"] == e0["short"]["bounds"]
              and x["probs"]["C"]["pi_E"] == e0["short"]["pi_E"] and abs(e1 - s["E"]) <= 1e-6
              and abs(float(mean[W[0]:W[1]].sum()) - e0["attention_mass"]) <= 1e-6 and abs(mean.sum() - 1) <= 1e-9
              and all(W[2] <= v[0] < v[1] <= W[3] <= heard for v in b.values())
              and all(b[k] == x["bounds"][k] for k in "LCR") and j == x["j"])
        if not ok:
            fails.append((uid, t))
        es = sel[(uid, t)]
        g = {"EN-confusion": "EN_TP" if es > 0 else "EN_FN", "ZH-correct": "ZH_FP" if es > 0 else "ZH_TN"}.get(p["stratum"], "EN_CORRECT")
        rows.append({"d": p["dialogue_id"], "g": g, "E1": e1, "Et": E[j], "C": E[j] - (E[o[0]] + E[o[1]]) / 2, "j": j})
    checks["row_reconstruction"] = not fails
    G = {k: [r for r in rows if r["g"] == k] for k in ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")}
    checks["groups"] = {k: len(v) for k, v in G.items()} == c["population"]["historical_groups"]
    sp = lambda rs: len({r["d"] for r in rs})
    fp, tp, fn = G["ZH_FP"], G["EN_TP"], G["EN_FN"]
    keys = sorted({r["d"] for r in rows})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))
    s1 = [r for r in fp if r["Et"] <= 0.10 and r["E1"] - r["Et"] >= 0.25]
    s2 = [r for r in tp if r["Et"] >= 0.20]
    pt, l80, n1 = _diff(tp, fp, lambda r: r["Et"], lambda r: r["Et"], keys, idx)
    c1 = [r for r in fp if r["C"] <= -0.25]
    c2 = [r for r in tp if r["C"] >= 0]
    cpt, cl80, n2 = _diff(tp, fp, lambda r: r["C"], lambda r: r["C"], keys, idx)
    f9 = [r for r in fn if r["Et"] >= 0.20]
    fps = len(s1) >= 4 and sp(s1) >= 3
    valid = checks["row_reconstruction"] and checks["groups"] and n1 >= 9900 and n2 >= 9900 and all(ana["validity"].values())
    if not valid:
        lab = "P2_SEL_T_INVALID"
    elif fps and len(s2) >= 34 and sp(s2) >= 3 and pt >= 0.25 and l80 > 0:
        lab = "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED"
    elif len(c1) >= 4 and sp(c1) >= 3 and len(c2) >= 26 and sp(c2) >= 3 and cpt >= 0.25 and cl80 > 0:
        lab = "P2_SEL_T_CONTEXT_CONTRAST_ONLY"
    elif len(f9) >= 9 and sp(f9) >= 3 and not fps:
        lab = "P2_SEL_T_RECALL_ONLY"
    else:
        lab = "P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE"
    st = ana["statistics"]
    checks["statistics_agree"] = (abs(pt - st["E_tok_TP_minus_FP"]["point"]) <= 1e-8 and abs(l80 - st["E_tok_TP_minus_FP"]["lower80"]) <= 1e-8
                                  and abs(cpt - st["C_TP_minus_FP"]["point"]) <= 1e-8 and abs(cl80 - st["C_TP_minus_FP"]["lower80"]) <= 1e-8
                                  and n1 == st["E_tok_TP_minus_FP"]["valid_draws"])
    checks["counts_agree"] = (len(s1), len(s2), len(c1), len(c2), len(f9)) == (
        st["counts"]["fp_suppressed"], st["counts"]["tp_evidence"], st["counts"]["fp_contrast"], st["counts"]["tp_contrast"], st["counts"]["fn_recall"])
    checks["label_agrees"] = lab == ana["label"]
    return {"schema": "p2_sel_t_t0_audit_v1", "stage": "T0", "label": lab,
            "verdict": "P2_SEL_T_AUDIT: PASS" if all(checks.values()) else "P2_SEL_T_AUDIT: BLOCK", "checks": checks,
            "independent": {"fp_suppressed": len(s1), "tp_evidence": len(s2), "Etok_diff": pt, "Etok_lower80": l80,
                            "fp_contrast": len(c1), "tp_contrast": len(c2), "C_diff": cpt, "C_lower80": cl80, "fn_recall": len(f9),
                            "valid_draws": [n1, n2]}, "row_failures": fails[:10]}


def cmd_final(args) -> dict:
    c = cfg()
    checks = {}
    for rel in ("docs/inference_cf/P2_SEL_T_SPEC.md", "docs/inference_cf/P2_SEL_T_CODEX_DESIGN.md", CONFIG):
        checks[f"unchanged:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    entries = sorted(x.name for x in BASE.iterdir())
    a0 = json.loads((BASE / "t0_run1_audit.json").read_text())
    checks["t0_audit_pass"] = a0["verdict"] == "P2_SEL_T_AUDIT: PASS"
    t1 = (BASE / "t1_run1").exists()
    checks["t1_only_if_supported"] = (not t1) or (a0["label"] == "P2_SEL_T_TOKEN_LOCALIZATION_SUPPORTED" and (BASE / "t0_selection.json").exists())
    checks["t2_only_if_t1_supported"] = not any(e.startswith("t2") for e in entries) or (
        t1 and json.loads((BASE / "t1_run1_audit.json").read_text())["label"] == "P2_SEL_T_TOKEN_GATE_SUPPORTED")
    rt = json.loads((BASE / "t0_run1/runtime.json").read_text())
    checks["t0_no_steering_d2_autograd"] = rt["autograd_calls"] == 0 and rt["readout_calls"] == 0 and rt["steering_calls"] == 0
    checks["t0_lid_cap"] = rt["lid_calls"] <= c["compute"]["T0_native_lid_calls_max"]
    m = json.loads((BASE / "t0_run1/manifest.json").read_text())
    checks["firewall"] = m["role"] == "D-dev-select" and m["firewall"]["role"] == "D-dev-select"
    checks["single_attempt_t0"] = len(list((BASE / "t0_run1").glob("slurm-*.out"))) == 1
    checks["runtime_surface_reference_free"] = runtime_surface_clean()["ok"]
    checks["no_p3_full300"] = not any(("p3" in e.lower()) or ("300" in e) for e in entries)
    return {"schema": "p2_sel_t_final_audit_v1", "verdict": "P2_SEL_T_AUDIT: PASS" if all(checks.values()) else "P2_SEL_T_AUDIT: BLOCK",
            "t0_label": a0["label"], "t1_run": t1, "entries": entries, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("prerun", "final"):
        sub.add_parser(n).add_argument("--out", required=True)
    t0 = sub.add_parser("t0")
    t0.add_argument("--analysis", required=True)
    t0.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "t0": cmd_t0, "final": cmd_final}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "t0_label")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
