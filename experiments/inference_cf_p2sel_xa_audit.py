#!/usr/bin/env python
"""Independent P2-SEL-XA auditor. Does NOT import inference_cf_p2sel_xa_analyze, any other analysis module,
or source_compatibility: q+u reconstruction, J (numpy float64 from saved raw logits), S/C/F, finite-difference
predictions and validity, historical groups, shared-draw bootstrap, predicates and label are re-implemented.

prerun -> PASS_TO_P2_SEL_XA_XA0 | BLOCK_BEFORE_P2_SEL_XA_XA0
xa0    -> P2_SEL_XA_AUDIT: PASS (XA0)
final  -> P2_SEL_XA_AUDIT: PASS
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from experiments.inference_cf_p2dir_audit import bf16, fhash, git_blob_hash

FREEZE = "595e17436afa9544b487b0b956092ec80ceec734"
CONFIG = "configs/inference_cf/p2_sel_xa.json"
BASE = ROOT / "results/inference_cf/p2sel_xa"
P2DIR = ROOT / "results/inference_cf/p2dir/exp1_run1"
FORBIDDEN = ("target_ids", "competitor", "stratum", "group", "positions.json", "timing", "oracle", "evaluator_targets", "p2rj")


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cfg():
    return json.loads((ROOT / CONFIG).read_text())


def surface_clean() -> dict:
    res = {}
    for rel, names in (("experiments/inference_cf_p2sel_xa.py", ("xa0_utterance", "SourceScale")),
                       ("src/csasr/inference_cf/source_compatibility.py", ("compatibility",))):
        src = (ROOT / rel).read_text()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
                res[n.name] = not any(f in ast.get_source_segment(src, n) for f in FORBIDDEN)
    return {"ok": len(res) == 3 and all(res.values()), "parts": res}


def J_np(z, t, sup, beg, part) -> float:
    z = np.asarray(z, dtype=np.float64).copy()
    if sup:
        z[sup] = -np.inf
    if t == 0 and beg:
        z[beg] = -np.inf
    def lse(x):
        mx = np.max(x)
        return mx + math.log(np.sum(np.exp(x - mx)))
    lz = lse(z)
    pe, pm = lse(z[part["embedded_ids"]]) - lz, lse(z[part["matrix_ids"]]) - lz
    le = math.log(1e-12)
    return float(np.logaddexp(pe, le) - np.logaddexp(pm, le))


def cmd_prerun(args) -> dict:
    c = cfg()
    checks, notes = {}, {}
    for rel in ("docs/inference_cf/P2_SEL_XA_SPEC.md", "docs/inference_cf/P2_SEL_XA_CODEX_DESIGN.md", CONFIG):
        checks[f"frozen:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    e = json.loads((ROOT / "results/inference_cf/p2sel_e/e0_run1_audit.json").read_text())
    t = json.loads((ROOT / "results/inference_cf/p2sel_t/t0_run1_audit.json").read_text())
    tf = json.loads((ROOT / "results/inference_cf/p2sel_t/final_audit.json").read_text())
    checks["parents_terminal_pass"] = (e["verdict"] == "P2_SEL_E_AUDIT: PASS" and e["label"] == c["parents"][0]
                                       and t["verdict"] == "P2_SEL_T_AUDIT: PASS" and t["label"] == c["parents"][1]
                                       and tf["verdict"] == "P2_SEL_T_AUDIT: PASS")
    sealed = json.loads((P2DIR / "directions_sealed.json").read_text())
    checks["p2dir_seal"] = sealed["status"] == "SEALED" and all(fhash(P2DIR / r) == h for r, h in sealed["files"].items())
    checks["p2dir_exp1_audit_pass"] = json.loads((ROOT / "results/inference_cf/p2dir/exp1_run1_audit.json").read_text())["verdict"] == "P2_DIR_AUDIT: PASS"
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    inv = 0
    for p in con["positions"]:
        i = cidx[p["utterance_id"]]
        rec = json.loads((P2DIR / f"rows/{i:03d}.json").read_text())["positions"][str(p["t"])]
        with np.load(P2DIR / f"rows/{i:03d}.npz") as z:
            g = z.get(f"t{p['t']}_g_readout")
            hb = z.get(f"t{p['t']}_hb")
        inv += (g is not None and g.dtype == np.float32 and g.shape == (1280,) and np.isfinite(g).all() and hb is not None
                and rec["directions"]["D2"]["status"] == "ok")
    checks["raw_gradient_inventory_180"] = inv == 180
    checks["population_180"] = len(pos["positions"]) == 180 and [(p["utterance_id"], p["t"]) for p in pos["positions"]] == \
        [(p["utterance_id"], p["t"]) for p in con["positions"]]
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    grp = {}
    for p in pos["positions"]:
        es = sel[(p["utterance_id"], p["t"])]
        g = {"EN-confusion": "EN_TP" if es > 0 else "EN_FN", "ZH-correct": "ZH_FP" if es > 0 else "ZH_TN"}.get(p["stratum"], "EN_CORRECT")
        grp[g] = grp.get(g, 0) + 1
    checks["historical_groups"] = grp == c["population"]["historical_groups"]
    checks["p2sel_t_rows_180"] = len(json.loads((ROOT / "results/inference_cf/p2sel_t/t0_run1_analysis.json").read_text())["rows"]) == 180
    checks["mini_panel"] = fhash(ROOT / c["XA2"]["panel"]) == "sha256:" + c["XA2"]["panel_sha256"]
    checks["no_XA_outcome"] = not BASE.exists() or not any(x.name.startswith("xa") for x in BASE.iterdir())
    fd = c["XA0"]["finite_difference"]
    checks["constants"] = (fd["lambda_probe"] == 0.95 and fd["minimum_material_rows"] == 30 and fd["minimum_material_dialogues"] == 10
                           and c["compute"]["XA0_new_backward_calls"] == 0 and c["compute"]["XA0_new_native_lid_calls"] == 0)
    checks["construction_surface_reference_free"] = surface_clean()["ok"]
    asrc = (ROOT / "experiments/inference_cf_p2sel_xa_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|source_compatibility)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2sel_xa.py").exists()
    notes["groups"] = grp
    v = "PASS_TO_P2_SEL_XA_XA0" if all(checks.values()) else "BLOCK_BEFORE_P2_SEL_XA_XA0"
    return {"schema": "p2_sel_xa_prerun_audit_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


def _diff(A_rows, B_rows, keys, idx):
    def per(rows):
        acc = {}
        for r in rows:
            acc.setdefault(r["d"], []).append(r["F"])
        return {k: sum(v) / len(v) for k, v in acc.items()}
    A, B = per(A_rows), per(B_rows)
    ds = []
    for draw in idx:
        ks = [keys[j] for j in draw]
        a, b = [A[k] for k in ks if k in A], [B[k] for k in ks if k in B]
        if a and b:
            ds.append(sum(a) / len(a) - sum(b) / len(b))
    return sum(A.values()) / len(A) - sum(B.values()) / len(B), (float(np.quantile(ds, 0.2)) if ds else None), len(ds)


def cmd_xa0(args) -> dict:
    c = cfg()
    ana = json.loads((ROOT / args.analysis).read_text())
    run = BASE / "xa0_run1"
    man = json.loads((run / "manifest.json").read_text())
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    part, sup, beg = man["partition"], man["suppression"]["suppress"], man["suppression"]["begin"]
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    checks = {"manifest_sources_at_commit": all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == man["manifest_hash"]}
    rows, fails = [], []
    an_rows = {(r["utterance_id"], r["t"]): r for r in ana["rows"]}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        pre = f"t{t}_"
        with np.load(run / f"rows/{i:03d}.npz") as z:
            q, u, r1, r95 = (z[pre + k].astype(np.float64) for k in ("q", "u", "r1", "r095"))
        with np.load(run / f"rows/{i:03d}_logits.npz") as z:
            l1, l95 = bf16(z[pre + "l1"]), bf16(z[pre + "l095"])
        with np.load(P2DIR / f"rows/{i:03d}.npz") as z:
            g, hb = z[pre + "g_readout"].astype(np.float64), z[pre + "hb"]
        with np.load(P2DIR / f"rows/{i:03d}_logits.npz") as z:
            none = bf16(z[pre + "none"])
        gn, un = float(np.sqrt(g @ g)), float(np.sqrt(u @ u))
        S = float(g @ u) if gn > 0 and un > 0 else 0.0
        C = S / (gn * un + 1e-12) if gn > 0 and un > 0 else 0.0
        F = min(1.0, max(0.0, S))
        J1, J95 = J_np(l1, t, sup, beg, part), J_np(l95, t, sup, beg, part)
        rel = float(np.max(np.abs(q + u - r1)) / max(float(np.max(np.abs(r1))), 1e-12))
        ok = (np.array_equal(r1.astype(np.float32), hb) and np.array_equal(l1, none) and rel <= 0.03125
              and (S > 0) == (C > 0) and -1 - 1e-6 <= C <= 1 + 1e-6)
        a = an_rows[(uid, t)]
        ok = ok and abs(a["J"] - J1) <= c["XA0"]["J_audit_abs_tolerance"] and abs(a["S"] - S) <= 1e-9 * max(1, abs(S)) and abs(a["F"] - F) <= 1e-9
        if not ok:
            fails.append((uid, t))
        es = sel[(uid, t)]
        grp = {"EN-confusion": "EN_TP" if es > 0 else "EN_FN", "ZH-correct": "ZH_FP" if es > 0 else "ZH_TN"}.get(p["stratum"], "EN_CORRECT")
        rows.append({"d": p["dialogue_id"], "g": grp, "S": S, "F": F, "obs": J95 - J1, "Pn": -0.05 * S, "Pr": float(g @ (r95 - r1))})
    checks["row_reconstruction"] = not fails
    sub = [r for r in rows if abs(r["Pn"]) >= 0.02 and abs(r["Pr"]) >= 0.02]
    sg = lambda x: (x > 0) - (x < 0)
    fd = {"n": len(sub), "dialogues": len({r["d"] for r in sub})}
    if sub:
        fd.update(s1=sum(r["obs"] != 0 and sg(r["obs"]) == sg(r["Pr"]) for r in sub) / len(sub),
                  s2=sum(sg(r["Pn"]) == sg(r["Pr"]) for r in sub) / len(sub),
                  med=float(np.median([r["obs"] / r["Pr"] for r in sub])),
                  rms=math.sqrt(sum((r["obs"] - r["Pr"]) ** 2 for r in sub) / sum(r["Pr"] ** 2 for r in sub)))
        fd["ok"] = fd["n"] >= 30 and fd["dialogues"] >= 10 and fd["s1"] >= 0.9 and fd["s2"] >= 0.9 and 0.5 <= fd["med"] <= 1.5 and fd["rms"] <= 0.5
    else:
        fd["ok"] = False
    af = ana["finite_difference"]
    # attempt-1 required ratio/RMS agreement within 1e-6, but the auditor's float64 J and the primary float32
    # readout.objective J legitimately differ up to the frozen J tolerance (1e-4 each, 2e-4 on a difference);
    # agreement is therefore checked on per-row observed dJ within 2e-4 plus identical subset and validity flag.
    obs_ok = all(abs(r["obs"] - an_rows[(p["utterance_id"], int(p["t"]))]["observed"]) <= 2 * c["XA0"]["J_audit_abs_tolerance"]
                 for r, p in zip(rows, pos["positions"]))
    sub_ana = sorted((r["utterance_id"], r["t"]) for r in ana["rows"] if abs(r["P_nominal"]) >= 0.02 and abs(r["P_realized"]) >= 0.02)
    sub_aud = sorted((p["utterance_id"], int(p["t"])) for r, p in zip(rows, pos["positions"]) if abs(r["Pn"]) >= 0.02 and abs(r["Pr"]) >= 0.02)
    checks["fd_agrees"] = fd["n"] == af["n"] and fd["ok"] == af["ok"] and obs_ok and sub_ana == sub_aud
    fd["max_ratio_stat_diff_vs_analysis"] = None if not sub else max(abs(fd["med"] - af["median_ratio"]), abs(fd["rms"] - af["rel_rms"]))
    G = {k: [r for r in rows if r["g"] == k] for k in ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")}
    checks["groups"] = {k: len(v) for k, v in G.items()} == c["population"]["historical_groups"]
    sp = lambda rs: len({r["d"] for r in rs})
    keys = sorted({r["d"] for r in rows})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))
    pt, l80, nd = _diff(G["EN_TP"], G["ZH_FP"], keys, idx)
    fps = [r for r in G["ZH_FP"] if r["F"] <= 0.10]
    tps = [r for r in G["EN_TP"] if r["F"] >= 0.70]
    fns = [r for r in G["EN_FN"] if r["F"] >= 0.70]
    valid = checks["row_reconstruction"] and checks["groups"] and fd["ok"] and nd >= 9900 and all(
        v for k, v in ana["validity"].items() if k != "finite_difference")
    fp_ok = len(fps) >= 4 and sp(fps) >= 3
    if not valid:
        lab = "P2_SEL_XA_INVALID"
    elif fp_ok and len(tps) >= 34 and sp(tps) >= 3 and pt >= 0.5 and l80 > 0:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    elif not fp_ok and len(fns) >= 9 and sp(fns) >= 3:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_RECALL_ONLY"
    else:
        lab = "P2_SEL_XA_SOURCE_COMPATIBILITY_NOT_DISCRIMINATIVE"
    d = ana["decision"]["F_TP_minus_FP"]
    checks["contrast_agrees"] = abs(pt - d["point"]) <= 1e-8 and abs(l80 - d["lower80"]) <= 1e-8 and nd == d["valid_draws"]
    checks["counts_agree"] = (len(fps), len(tps), len(fns)) == (ana["decision"]["counts"]["fp_suppressed"],
                                                               ana["decision"]["counts"]["tp_retained"], ana["decision"]["counts"]["fn_available"])
    checks["label_agrees"] = lab == ana["label"]
    return {"schema": "p2_sel_xa_xa0_audit_v1", "stage": "XA0", "label": lab,
            "verdict": "P2_SEL_XA_AUDIT: PASS" if all(checks.values()) else "P2_SEL_XA_AUDIT: BLOCK", "checks": checks,
            "independent": {"fd": fd, "F_TP_minus_FP": pt, "lower80": l80, "valid_draws": nd, "fp_suppressed": len(fps),
                            "tp_retained": len(tps), "fn_available": len(fns)}, "row_failures": fails[:10]}


def cmd_final(args) -> dict:
    c = cfg()
    checks = {}
    for rel in ("docs/inference_cf/P2_SEL_XA_SPEC.md", "docs/inference_cf/P2_SEL_XA_CODEX_DESIGN.md", CONFIG):
        checks[f"unchanged:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    entries = sorted(x.name for x in BASE.iterdir())
    a0 = json.loads((BASE / "xa0_run1_audit.json").read_text())
    checks["xa0_audit_pass"] = a0["verdict"] == "P2_SEL_XA_AUDIT: PASS"
    checks["xa1_only_if_supported"] = not any(e.startswith(("xa1", "xa2")) for e in entries) or a0["label"] == "P2_SEL_XA_SOURCE_COMPATIBILITY_SUPPORTED"
    rt = json.loads((BASE / "xa0_run1/runtime.json").read_text())
    checks["xa0_no_backward_lid_steering"] = rt["autograd_calls"] == 0 and rt["readout_calls"] == 0 and rt["lid_calls"] == 0 and rt["steering_calls"] == 0
    m = json.loads((BASE / "xa0_run1/manifest.json").read_text())
    checks["firewall"] = m["role"] == "D-dev-select" and m["firewall"]["role"] == "D-dev-select"
    checks["single_attempt"] = len(list((BASE / "xa0_run1").glob("slurm-*.out"))) == 1
    checks["surface_reference_free"] = surface_clean()["ok"]
    checks["no_p3_full300"] = not any(("p3" in e.lower()) or ("300" in e) for e in entries)
    return {"schema": "p2_sel_xa_final_audit_v1", "verdict": "P2_SEL_XA_AUDIT: PASS" if all(checks.values()) else "P2_SEL_XA_AUDIT: BLOCK",
            "xa0_label": a0["label"], "entries": entries, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("prerun", "final"):
        sub.add_parser(n).add_argument("--out", required=True)
    x = sub.add_parser("xa0")
    x.add_argument("--analysis", required=True)
    x.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "xa0": cmd_xa0, "final": cmd_final}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "xa0_label")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
