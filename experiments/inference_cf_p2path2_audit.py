#!/usr/bin/env python
"""Independent P2-PATH2 auditor. Does NOT import inference_cf_p2path2_analyze, the PATH2 runner, or
csasr.inference_cf.consensus_guard / branch_adjudication / path_decode. Own streams/ED (TTA0 auditor ``own_lev``),
trigger/prefix recomputation from recorded per-step argmax pairs, score arithmetic, consensus/tie/winner, G1/G2 forced
traces and decision identity, cache-owner hashes plus source inspection, canonical counts (TTA0 auditor ``counts``),
29/105/143/0 cutoffs, material advantage and terminal label.

prerun -> PASS_TO_P2_PATH2 | BLOCK_BEFORE_P2_PATH2
post   -> P2_PATH2_AUDIT: PASS | BLOCK  (--phase primary before references; --phase full after secondary)
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

FREEZE = "e1c6ee7"
CONFIG = "configs/inference_cf/p2_path2.json"
PANEL = "docs/inference_cf/P2_PATH2_PANEL.json"
BASE = ROOT / "results/inference_cf/p2path2"
PLAN_REL = "results/inference_cf/p2path2/plan_sealed.json"
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


def severe(base_len, new_len, term):
    return base_len >= 10 and term == "eos" and new_len <= math.floor(0.5 * base_len)


def own_label(valid, z1, p1, m1, s1, z2, p2, m2, s2, c):
    S = c["controller_success"]
    ok = lambda z, p, m, s: z <= S["ZH_errors_max"] and p <= S["POI_errors_max"] and m <= S["mixed_errors_max"] and s == 0
    g1, g2 = ok(z1, p1, m1, s1), ok(z2, p2, m2, s2)
    raw = (z1 - z2 >= 2 or m1 - m2 >= 2) and p2 <= p1 + 1 and s2 == 0
    if not valid:
        return "P2_PATH2_INVALID", g1, g2, raw
    if g1 and g2 and raw:
        return "P2_PATH2_MIXED_GUARD_PROMISING", g1, g2, raw
    if g1:
        return "P2_PATH2_BRANCH_CONTROL_SUFFICIENT", g1, g2, raw
    if g2:
        return "P2_PATH2_ROLLBACK_NEEDED", g1, g2, raw
    return "P2_PATH2_LOCAL_GUARD_INSUFFICIENT", g1, g2, raw


def cmd_prerun(args) -> dict:
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in ("docs/inference_cf/P2_PATH2_SPEC.md", "docs/inference_cf/P2_PATH2_CODEX_DESIGN.md", PANEL, CONFIG)}
    checks["panel_hash"] = sha(ROOT / PANEL) == c["panel_byte_sha256"] == "ade7567c1e13f75e410e6a1cf27a3a613bbe9a9ac5d5898fa9dc18f8ea3d0923"
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    P = json.loads((ROOT / PANEL).read_text())
    nC = nU = 0
    ok = True
    for r in P["rows"]:
        so = r["source_outputs"]
        B, F = st(so["B0"]["tokens"], so["B0"]["terminated"]), st(so["A4"]["tokens"], so["A4"]["terminated"])
        C = F != B
        nC += C
        nU += not C
        if C:
            k = next(i for i, (x, y) in enumerate(zip(F, B)) if x != y)
            a = r["audit_only_expected"]
            ok &= a["k"] == k and a["b0"] == so["B0"]["tokens"][k:k + 3] and a["b4"] == so["A4"]["tokens"][k:k + 3]
        ok &= r["analysis_group"] == ("C" if C else "U") and all(sha(ROOT / so[x]["path"]) == so[x]["byte_sha256"] for x in so)
    checks["population_independent"] = ok and (nC, nU) == (5, 7)
    # historical counts recomputed from sealed PATH0 secondary per-row count vectors
    s0 = json.loads((ROOT / "results/inference_cf/p2path0/secondary_analysis.json").read_text())
    tot = {k: [sum(r["counts"][k][i] for r in s0["per_row"]) for i in (4, 0, 2)] for k in ("B0", "FREE")}
    H = c["historical_counts"]
    checks["historical_counts_and_cutoffs"] = (tot["B0"] == [H["B0"]["ZH_errors"], H["B0"]["POI_errors"], H["B0"]["mixed_errors"]] == [18, 119, 137]
                                               and tot["FREE"] == [H["A4"]["ZH_errors"], H["A4"]["POI_errors"], H["A4"]["mixed_errors"]] == [41, 101, 143]
                                               and c["controller_success"]["ZH_errors_max"] == 41 - math.ceil((41 - 18) / 2) == 29
                                               and c["controller_success"]["POI_errors_max"] == 119 - math.ceil(0.75 * (119 - 101)) == 105
                                               and c["controller_success"]["mixed_errors_max"] == 143)
    lab = [own_label(True, *v, c)[0] for v in ((29, 105, 143, 0, 29, 105, 141, 0), (29, 105, 143, 0, 30, 105, 143, 0), (30, 105, 143, 0, 29, 105, 143, 0),
                                               (30, 105, 143, 0, 30, 105, 143, 0), (29, 105, 143, 0, 27, 107, 143, 0))]
    checks["label_truth_table"] = lab == ["P2_PATH2_MIXED_GUARD_PROMISING", "P2_PATH2_BRANCH_CONTROL_SUFFICIENT", "P2_PATH2_ROLLBACK_NEEDED",
                                          "P2_PATH2_LOCAL_GUARD_INSUFFICIENT", "P2_PATH2_BRANCH_CONTROL_SUFFICIENT"]
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False
    src = (ROOT / "experiments/inference_cf_p2path2.py").read_text()
    run_code = next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "cmd_run")
    phase2 = run_code[run_code.index("# ---------------- Phase 2"):]
    checks["controller_input_allowlist"] = not any(t in phase2 for t in ('["analysis"]', "an_[", '["expected"]', "AUTO", "y_A_text", '["group"]',
                                                                        "load_references", "audit_only"))
    gsrc = (ROOT / "src/csasr/inference_cf/consensus_guard.py").read_text()
    checks["two_instances_no_alias"] = "b_t0, b_a4 = load_whisper(cfg_m), load_whisper(cfg_m)" in run_code and "data_ptr()" in run_code
    checks["no_cross_model_cache"] = (gsrc.count("cached.Branch(") == 3 and "past_key_values" not in gsrc and ".cache =" not in gsrc
                                      and "resident state != declared owner before detector" in gsrc and "resident state != declared owner before rollout" in gsrc
                                      and "owned_clamp(b_a4" in phase2 and "owned_clamp(b_t0" not in phase2)
    checks["one_trigger_shared_decision"] = phase2.count("lockstep_detect(") == 1 and phase2.count("consensus(") == 1 and "g1_forced(dec)" in phase2 \
        and "g2_forced(dec, eos)" in phase2
    checks["barrier_before_controller"] = run_code.index("phase-1 barrier failed") < run_code.index("# ---------------- Phase 2")
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path2_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path2_analyze|inference_cf_p2path2\b|consensus_guard|branch_adjudication|path_decode)",
                                              asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path2.py").exists()
    B = c["compute"]
    checks["compute_plan"] = B["jobs_max"] == 1 and B["A4_optimizer_steps"] == 24 and B["guard_events_max"] == 12 and B["hard_minutes"] == 30
    v = "PASS_TO_P2_PATH2" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH2"
    return {"schema": "p2_path2_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def cmd_post(args) -> dict:
    c = cfg()
    run = BASE / "run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]}
    if args.phase == "full":
        seal = json.loads((BASE / "output_seal.json").read_text())
        checks["seal_committed_unchanged"] = committed("results/inference_cf/p2path2/output_seal.json") and all(sha(ROOT / p) == h for p, h in seal["files"].items())
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(12)]
    bad, theta0_winners, trig = [], 0, 0
    p1 = {z["utterance_id"]: z for z in json.loads((ROOT / "results/inference_cf/p2path1/primary_analysis.json").read_text())["branch_scores"]}
    for r, x in zip(rows, plan["rows"]):
        a = x["analysis"]
        u = a["utterance_id"]
        okr = (r["status"] == "ok" and r["identity"] == u and r["theta0_equals_B0"] and r["effective_equals_checkpoint_bf16"] and r["G0_equals_sealed_A4"]
               and r["G0"]["tokens"] == a["A4"]["tokens"] and r["G0"]["terminated"] == a["A4"]["terminated"])
        det = r["detector"]
        pairs = det["argmax_pairs"]
        first = next((i for i, (p, q) in enumerate(pairs) if p != q), None)
        okr &= det["state_locked"] and det["fed_identical"] and det["positions_ok"]
        okr &= det["trigger"] == (first is not None) and det["prefix"] == [p for p, q in pairs[:len(det["prefix"])]] and \
            all(p == q for p, q in pairs[:-1] if first is not None) and (first is None or first == det["k"] == len(det["prefix"]))
        if first is None:
            okr &= a["group"] == "U" and r["G1"] == r["G0"] == r["G2"] and r["decision"] is None
        else:
            trig += 1
            d = r["decision"]
            exp = a["expected"]
            okr &= a["group"] == "C" and d["k"] == exp["k"] and d["prefix"] == exp["common_prefix"] and d["b0"]["tokens"] == exp["b0"] and d["b4"]["tokens"] == exp["b4"]
            okr &= d["b0"]["tokens"][0] == pairs[first][0] and d["b4"]["tokens"][0] == pairs[first][1]
            S = {k_: sum(v["logprobs"]) / len(v["logprobs"]) for k_, v in d["scores"].items()}
            okr &= all(S[k_] == d["scores"][k_]["score"] for k_ in S)
            cons = {"b0": 0.5 * S["theta0_b0"] + 0.5 * S["A4_b0"], "b4": 0.5 * S["theta0_b4"] + 0.5 * S["A4_b4"]}
            win = "theta0" if cons["b0"] - cons["b4"] >= -G else "A4"
            okr &= win == d["winner"] and cons["b0"] - cons["b4"] == d["margin"]
            p1s = json.loads((ROOT / exp["PATH1_score_path"]).read_text())["scores"]
            okr &= all(d["scores"][k_]["logprobs"] == p1s[v]["logprobs"] for k_, v in (("theta0_b0", "T0_B"), ("theta0_b4", "T0_ALT"), ("A4_b0", "A4_B"), ("A4_b4", "A4_ALT")))
            okr &= win == ("theta0" if p1[u]["choice"] == "B" else "A4")
            okr &= r["G1"]["decision_hash"] == r["G2"]["decision_hash"] == d["decision_hash"]
            wtok = (d["b0"] if win == "theta0" else d["b4"])["tokens"][0]
            okr &= [f["token"] for f in r["G1"]["trace"]["forced"]] == [wtok] and r["G1"]["trace"]["prefix_ok"] and r["G1"]["state_locked"]
            g1s = st(r["G1"]["tokens"], r["G1"]["terminated"])
            okr &= g1s[:d["k"]] == d["prefix"] and g1s[d["k"]] == wtok
            okr &= all(v["state_locked"] and v["state_hash_before"] == v["state_hash_after"] == v["owner"]["state_hash"] for v in r["score_paths"].values())
            if win == "theta0":
                theta0_winners += 1
                want = d["b0"]["tokens"] + ([EOS] if d["b0"]["terminated"] == "eos" else [])
                tr = r["G2"]["trace"]
                okr &= [f["token"] for f in tr["forced"]] == want and tr["prefix_ok"] and tr["positions_ok"] and r["G2"]["state_locked"]
                g2s = st(r["G2"]["tokens"], r["G2"]["terminated"])
                okr &= g2s[:d["k"]] == d["prefix"] and g2s[d["k"]:d["k"] + len(want)] == want
            else:
                okr &= r["G1"]["tokens"] == r["G0"]["tokens"] == r["G2"]["tokens"] and r["G2"]["mode"] == "A4_winner_ordinary"
        if not okr:
            bad.append(u)
    cnt = rt["counters"]
    checks["rows_trigger_scores_winner_G1G2_ownership"] = not bad
    checks["counts"] = (trig == 5 and cnt["triggers"] == trig and cnt["rollouts"] == 2 * trig and cnt["score_paths"] == 4 * trig and cnt["G1_decodes"] == trig
                        and cnt["G2_decodes"] == theta0_winners and cnt["A4"]["optimizer_steps"] == 24 and cnt["A4"]["backwards"] + cnt["audit"]["backwards"] <= 25)
    checks["primary_valid_agrees"] = prim["valid"] == all(checks.values())
    lab_out = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        from transformers import WhisperProcessor
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
        refs = load_references()
        tot = {g: [0, 0, 0] for g in ("G1", "G2", "B0", "A4")}
        sev = {"G1": 0, "G2": 0}
        for r, x in zip(rows, plan["rows"]):
            a = x["analysis"]
            ref = refs[a["utterance_id"]]["reference"]
            texts = {"G1": tok.decode(r["G1"]["tokens"], skip_special_tokens=True), "G2": tok.decode(r["G2"]["tokens"], skip_special_tokens=True),
                     "B0": a["y_B_text"], "A4": json.loads((ROOT / a["A4"]["path"]).read_text())["A4"]["text"]}
            for g, t in texts.items():
                cc = tau.counts(ref, t)
                tot[g][0] += cc[4]
                tot[g][1] += cc[0]
                tot[g][2] += cc[2]
            for g in ("G1", "G2"):
                sev[g] += (r[g]["terminated"] == "cap") + (severe(len(a["B0"]["tokens"]), len(r[g]["tokens"]), r[g]["terminated"])
                                                           or severe(len(a["A4"]["tokens"]), len(r[g]["tokens"]), r[g]["terminated"]))
        checks["historical_baselines_recomputed"] = tot["B0"] == [18, 119, 137] and tot["A4"] == [41, 101, 143]
        mm = sec["metrics_all12"]
        checks["counts_agree"] = all([mm[g]["zh"], mm[g]["poi"], mm[g]["mixed"]] == tot[g] for g in ("G1", "G2"))
        lab_out, g1p, g2p, raw = own_label(prim["valid"] and all(checks.values()), *tot["G1"], sev["G1"], *tot["G2"], sev["G2"], c)
        checks["criteria_agree"] = g1p == sec["criteria"]["G1"]["pass"] and g2p == sec["criteria"]["G2"]["pass"] and raw == sec["raw_G2_material_advantage"]
        checks["label_agrees"] = lab_out == sec["label"]
    v = "P2_PATH2_AUDIT: PASS" if all(checks.values()) else "P2_PATH2_AUDIT: BLOCK"
    return {"schema": "p2_path2_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab_out, "checks": checks, "failures": bad,
            "git_commit": _git("rev-parse", "HEAD")}


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
