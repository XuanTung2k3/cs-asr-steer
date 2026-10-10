#!/usr/bin/env python
"""Independent P2-PATH1 auditor. Does NOT import inference_cf_p2path1_analyze, inference_cf_p2path0_analyze,
csasr.inference_cf.branch_adjudication / path_decode, or the PATH1/PATH0 runners. Own streams/sites/ED (TTA0 auditor
``own_lev``), inherited INDUCE_1, I/Delta/materiality, mechanism precedence, consensus/tie/criterion, cache/state
ownership and ASR_STATE_ALIGNMENT; canonical metric primitives only.

prerun -> PASS_TO_P2_PATH1 | BLOCK_BEFORE_P2_PATH1
post   -> P2_PATH1_AUDIT: PASS | BLOCK   (--phase primary before references; --phase full after secondary)
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

import experiments.inference_cf_p2tta0_audit as tau

FREEZE = "5a159f2"
CONFIG = "configs/inference_cf/p2_path1.json"
SITES = "docs/inference_cf/P2_PATH1_SITES.json"
PARENT = "docs/inference_cf/P2_PATH0_SITES.json"
BASE = ROOT / "results/inference_cf/p2path1"
PLAN_REL = "results/inference_cf/p2path1/plan_sealed.json"
EOS = 50257
G = 1e-12


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def st(t, term):
    return list(t) + ([EOS] if term == "eos" else [])


def first_div(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return None if i == len(a) == len(b) else i


def own_sites():
    S = json.loads((ROOT / SITES).read_text())
    P0 = json.loads((ROOT / PARENT).read_text())
    plan4 = json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())
    fp = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    th = lambda a: hashlib.sha256(json.dumps(a, separators=(",", ":")).encode("ascii")).hexdigest()
    out, ok = [], True
    for s, p0 in zip(S["rows"], P0["rows"]):
        p4 = plan4["rows"][p0["parent_A3_index"]]
        a4 = json.loads((ROOT / p0["sources"]["A4"]["path"]).read_text())["A4"]
        B, A, F = st(p4["y_B"], p4["y_B_terminated"]), st(p4["y_A"], p4["y_A_terminated"]), st(a4["tokens"], a4["terminated"])
        U = F == B
        k = first_div(B, A) if U else first_div(F, B)
        alt = A if U else F
        dBA = tau.own_lev(B[k + 1:], A[k + 1:]) if U else None
        altc = p4["y_A"] if U else a4["tokens"]
        r_ok = (fp(p0) == s["PATH0_row_sha256"] and s["utterance_id"] == p4["utterance_id"] and s["population"] == ("U_PRIMARY" if U else "C_SCORE_ONLY")
                and s["site_index"] == k and s["common_prefix"] == B[:k] and th(B[:k]) == s["common_prefix_sha256"] and s["c_B"] == B[k]
                and s["c_ALT"] == alt[k] and s["score_branches"]["B"]["tokens"] == p4["y_B"][k:k + 3] and s["score_branches"]["ALT"]["tokens"] == altc[k:k + 3])
        if U:
            h = json.loads((ROOT / s["PATH0_output_path"]).read_text())["L1"]
            r_ok &= s["historical_A4A"]["tokens"] == h["tokens"] and s["historical_A4A"]["terminated"] == h["terminated"] and \
                s["L1"]["suffix_baseline_distance"] == dBA and s["L1"]["suffix_eligible"] == (dBA > 0)
        ok &= r_ok
        out.append({"u": s["utterance_id"], "dlg": s["dialogue_id"], "U": U, "k": k, "B": B, "A": A, "F": F, "cB": B[k], "cA": alt[k], "dBA": dBA,
                    "donB": p4["y_B"][k:k + 3], "donA": altc[k:k + 3], "hist": s.get("historical_A4A")})
    return out, bool(ok)


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in ("docs/inference_cf/P2_PATH1_SPEC.md", "docs/inference_cf/P2_PATH1_CODEX_DESIGN.md", SITES, CONFIG)}
    checks["sites_hash"] = sha(ROOT / SITES) == c["sites_byte_sha256"]
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    mine, ok = own_sites()
    checks["sites_recomputed_independently"] = ok
    U = [x for x in mine if x["U"]]
    checks["population"] = (len(U), sum(x["dBA"] > 0 for x in U), len(mine) - len(U)) == (7, 6, 5) and sum(x["dBA"] for x in U) == 48
    I = c["inherited_INDUCE_1"]
    p0c = json.loads((ROOT / "configs/inference_cf/p2_path0.json").read_text())["suffix_decision"]
    checks["inherited_criterion_exact"] = I["row_distance_reduction_min"] == p0c["row_distance_reduction_min"] and I["INDUCE"] == p0c["INDUCE"]
    T = c["state_thresholds"]
    checks["thresholds"] = (T["prefix_delta_pooled_max"] == 0.2 and T["prefix_material_A4_rows_max"] == 2 and T["adapted_delta_pooled_min"] == 0.3
                            and T["adapted_material_A4_rows_min"] == 3 and T["mixed_meaningful_theta0_pooled_min"] == 0.3 and T["mixed_theta0_row_I_min"] == 0.25
                            and c["branch_adjudication"]["H"] == 3 and c["branch_adjudication"]["tie_choice"] == "B"
                            and c["branch_adjudication"]["CONSENSUS_REJECTS_AUTO"]["B_choices_min"] == 5)
    a4c = json.loads((ROOT / "configs/inference_cf/p2_tta_a4.json").read_text())
    checks["a4_inherited"] = c["A4_inherited"]["trainables"] == a4c["trainables"] and c["A4_inherited"]["optimization"] == a4c["optimization"]
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False
    src = (ROOT / "experiments/inference_cf_p2path1.py").read_text()
    run_code = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "cmd_run")
    badr = ("load_references", "inference_cf_p2_evaluate", "corpus_metrics", "ReadoutDirection", "native_lid", "_edit_hook", "num_beams=", "past_key_values",
            ".cache =", "copy.deepcopy")
    checks["runner_reference_and_cache_free"] = not any(t in run_code for t in badr)
    checks["barrier_before_factorial"] = run_code.index("phase-1 barrier failed") < run_code.index("Phase 2")
    # NO STALE CROSS-MODEL CACHE: every path API creates its own Branch after the state check; no cache object is passed
    bsrc = (ROOT / "src/csasr/inference_cf/branch_adjudication.py").read_text()
    psrc = (ROOT / "src/csasr/inference_cf/path_decode.py").read_text()
    checks["no_cross_state_cache"] = (bsrc.index("resident state != declared owner before branch creation") < bsrc.index("cached.Branch(bundle")
                                      and "cache" not in [a.arg for f in ast.walk(ast.parse(bsrc)) if isinstance(f, ast.FunctionDef) for a in f.args.args + f.args.kwonlyargs]
                                      and "cached.Branch(bundle, encoded, list(prompt)" in psrc and "state_hash_after" in bsrc
                                      and run_code.count("guard.materialize") == 2 and "guard.restore()" in run_code)
    checks["forced_once_release_k1"] = "forced=[ctok]" in run_code and 'tr["release_index"] == s["site"] + 1' in run_code
    checks["path0_sources_unchanged"] = sha(ROOT / "src/csasr/inference_cf/path_decode.py") == c["source_sha256"]["src/csasr/inference_cf/path_decode.py"]
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path1_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path1_analyze|p2path0_analyze|branch_adjudication|path_decode|inference_cf_p2path[01]\b)",
                                              asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path1.py").exists()
    B = c["compute"]
    checks["compute_plan"] = B["jobs_max"] == 1 and B["primary_factorial_U_decodes"] == 28 and B["score_paths"] == 48 and B["optimizer_steps"] == 24
    v = "PASS_TO_P2_PATH1" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH1"
    return {"schema": "p2_path1_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def owner_ok(v: dict, exp: str) -> bool:
    """Factorial path records store owner + state_locked (runner checks hash before/after); score records also store
    both hashes. Attempt 1 of the primary audit crashed reading state_hash_before from path records (no output written)."""
    ok = v["owner"]["state_hash"] == exp and v["state_locked"] is True
    if "state_hash_before" in v:
        ok &= v["state_hash_before"] == exp == v["state_hash_after"]
    return bool(ok)


def own_induce(rows):
    el = [r for r in rows if r["d0"] > 0]
    n = len(el)
    succ = [r for r in el if 1 - r["dt"] / r["d0"] >= 0.5 - G and r["dB"] > 0]
    pooled = 1 - sum(r["dt"] for r in el) / sum(r["d0"] for r in el) if n else None
    return {"pass": n >= 3 and len(succ) >= max(3, -(-n // 2)) and pooled >= 0.3 - G and len({r["dlg"] for r in succ}) >= 2,
            "succ": [r["u"] for r in succ], "pooled": pooled}


def cmd_post(args) -> dict:
    c = cfg()
    T = c["state_thresholds"]
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]}
    if args.phase == "full":
        seal = json.loads((BASE / "output_seal.json").read_text())
        checks["seal_committed_unchanged"] = committed("results/inference_cf/p2path1/output_seal.json") and \
            all(sha(ROOT / p) == h for p, h in seal["files"].items()) and seal["primary_label"] == prim["label"]
    mine, ok = own_sites()
    checks["sites"] = ok
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(12)]
    bad, Urows, scores = [], [], []
    for r, x in zip(rows, mine):
        g = (r["status"] == "ok" and r["identity"] == x["u"] and r["theta0_equals_B0"] and r["effective_equals_checkpoint_bf16"]
             and r["FREE"]["tokens"] + ([EOS] if r["FREE"]["terminated"] == "eos" else []) == x["F"] and r["reset_ok_phase1"])
        own_t0 = rt["theta0_ln_hash"]
        for k_, v in list(r.get("paths", {}).items()) + list(r["scores"].items()):
            g &= owner_ok(v, own_t0 if k_.startswith("T0") else r["A4_state_hash"])
        sc = {}
        for mdl in ("T0", "A4"):
            for b, don in (("B", x["donB"]), ("ALT", x["donA"])):
                v = r["scores"][f"{mdl}_{b}"]
                g &= v["prefix_ok"] and v["positions_ok"] and v["fed_ok"] and len(v["logprobs"]) == len(don) and all(math.isfinite(z) for z in v["logprobs"])
                sc[(mdl, b)] = sum(v["logprobs"]) / len(v["logprobs"])
        cons = {b: 0.5 * sc[("T0", b)] + 0.5 * sc[("A4", b)] for b in ("B", "ALT")}
        diff = cons["B"] - cons["ALT"]
        scores.append({"u": x["u"], "dlg": x["dlg"], "U": x["U"], "elig": x["U"] and x["dBA"] > 0, "choice": "B" if diff >= -G else "ALT", "strict": diff > G})
        if x["U"]:
            P = r["paths"]
            stp = lambda k_: st(P[k_]["tokens"], P[k_]["terminated"])
            for k_, ctok in (("T0B", x["cB"]), ("T0A", x["cA"]), ("A4B", x["cB"]), ("A4A", x["cA"])):
                tr = P[k_]["trace"]
                g &= [f["token"] for f in tr["forced"]] == [ctok] and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"] and \
                    tr["release_index"] == x["k"] + 1 and stp(k_)[:x["k"]] == x["B"][:x["k"]] and stp(k_)[x["k"]] == ctok
            g &= stp("T0B") == x["B"] and stp("A4B") == x["B"] and P["A4A"]["tokens"] == x["hist"]["tokens"] and P["A4A"]["terminated"] == x["hist"]["terminated"]
            rr = x["k"] + 1
            e = {"u": x["u"], "dlg": x["dlg"], "d0": tau.own_lev(x["B"][rr:], x["A"][rr:])}
            for tag, k_ in (("t0", "T0A"), ("a4", "A4A")):
                e[tag] = {"dA": tau.own_lev(stp(k_)[rr:], x["A"][rr:]), "dB": tau.own_lev(stp(k_)[rr:], x["B"][rr:])}
            Urows.append(e)
        if not g:
            bad.append(x["u"])
    checks["rows_factorial_cache_scores"] = not bad
    ind = {tag: own_induce([{"u": e["u"], "dlg": e["dlg"], "d0": e["d0"], "dt": e[tag]["dA"], "dB": e[tag]["dB"]} for e in Urows]) for tag in ("t0", "a4")}
    el = [e for e in Urows if e["d0"] > 0]
    for e in el:
        e["It0"], e["Ia4"] = 1 - e["t0"]["dA"] / e["d0"], 1 - e["a4"]["dA"] / e["d0"]
        e["D"], e["tau"] = e["Ia4"] - e["It0"], max(0.25, 1 / e["d0"])
    sd = sum(e["d0"] for e in el)
    It0, Ia4 = 1 - sum(e["t0"]["dA"] for e in el) / sd, 1 - sum(e["a4"]["dA"] for e in el) / sd
    D = Ia4 - It0
    mat = [e for e in el if e["D"] >= e["tau"] - G]
    t0s = [e for e in el if e["It0"] >= 0.5 - G and e["t0"]["dB"] > 0]
    a4s = [e for e in el if e["Ia4"] >= 0.5 - G and e["a4"]["dB"] > 0]
    nd = lambda xs: len({e["dlg"] for e in xs})
    prefix = ind["t0"]["pass"] and D <= 0.2 + G and len(mat) <= 2
    adapted = (not ind["t0"]["pass"]) and It0 <= 0.2 + G and len(t0s) <= 1 and D >= 0.3 - G and len(mat) >= 3 and nd(mat) >= 2
    mr = [e for e in el if e["It0"] >= 0.25 - G and e["t0"]["dB"] > 0]
    mb = It0 >= 0.3 - G and len(mr) >= 2 and nd(mr) >= 2
    sr = [e for e in a4s if e["u"] not in {z["u"] for z in t0s} and e["D"] >= e["tau"] - G]
    mc = len(t0s) >= 2 and nd(t0s) >= 2 and len(sr) >= 2 and nd(sr) >= 2
    valid = all(checks.values()) and ind["a4"]["pass"]
    lab = ("P2_PATH1_INVALID" if not valid else "P2_PATH1_PREFIX_DOMINANT" if prefix else "P2_PATH1_ADAPTED_STATE_DOMINANT" if adapted
           else "P2_PATH1_MIXED" if (ind["t0"]["pass"] or mb or mc) else "P2_PATH1_NO_CLEAR_MECHANISM")
    checks["A4A_inherited_pass"] = ind["a4"]["pass"]
    checks["inherited_agree"] = ind["t0"]["pass"] == prim["criteria"]["theta0A_INDUCE_1"]["pass"] and ind["a4"]["pass"] == prim["criteria"]["A4A_INDUCE_1"]["pass"]
    checks["state_agree"] = (abs(It0 - prim["mechanism"]["I_theta0_pooled"]) <= 1e-12 and abs(Ia4 - prim["mechanism"]["I_A4_pooled"]) <= 1e-12
                             and sorted(e["u"] for e in mat) == sorted(prim["mechanism"]["material_A4_rows"]))
    ue = [s_ for s_ in scores if s_["elig"]]
    sw = [s_ for s_ in ue if s_["strict"]]
    cra = sum(s_["choice"] == "B" for s_ in ue) >= 5 and len(sw) >= 4 and nd([{"dlg": s_["dlg"]} for s_ in sw]) >= 3
    checks["scores_agree"] = all(s_["choice"] == p["choice"] for s_, p in zip(scores, prim["branch_scores"])) and cra == prim["CONSENSUS_REJECTS_AUTO"]["pass"]
    checks["label_agrees"] = lab == prim["label"]
    flag = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        from transformers import WhisperProcessor
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
        refs = load_references()
        plan4 = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())["rows"]}
        h, up = {"T0A": 0, "A4A": 0}, {"T0A": 0, "A4A": 0}
        for r, x in zip(rows, mine):
            if not (x["U"] and x["dBA"] > 0):
                continue
            zb = tau.counts(refs[x["u"]]["reference"], plan4[x["u"]]["y_B_text"])[4]
            for k_ in ("T0A", "A4A"):
                z = tau.counts(refs[x["u"]]["reference"], tok.decode(r["paths"][k_]["tokens"], skip_special_tokens=True))[4]
                h[k_] += z - zb
                up[k_] += z > zb
        harm = {k_: h[k_] >= 2 and up[k_] >= 2 for k_ in h}
        flag = ("PREFIX" if harm["T0A"] and harm["A4A"] and h["A4A"] - h["T0A"] <= 2 else "ADAPTED_STATE" if harm["A4A"] and not harm["T0A"]
                and h["A4A"] - h["T0A"] >= 3 else "MIXED" if any(harm.values()) else "NONE")
        checks["secondary_flag_agrees"] = flag == sec["ASR_STATE_ALIGNMENT"] and sec["primary_label"] == prim["label"]
    v = "P2_PATH1_AUDIT: PASS" if all(checks.values()) else "P2_PATH1_AUDIT: BLOCK"
    return {"schema": "p2_path1_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab, "ASR_STATE_ALIGNMENT": flag, "checks": checks,
            "independent": {"I_theta0": It0, "I_A4": Ia4, "Delta": D, "T0A_pass": ind["t0"]["pass"], "A4A_pass": ind["a4"]["pass"],
                            "material": [e["u"] for e in mat], "CONSENSUS_REJECTS_AUTO": cra}, "failures": bad, "git_commit": _git("rev-parse", "HEAD")}


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
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "ASR_STATE_ALIGNMENT", "independent")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
