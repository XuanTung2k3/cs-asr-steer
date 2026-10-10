#!/usr/bin/env python
"""Independent P2-PATH0 auditor. Does NOT import inference_cf_p2path0_analyze, csasr.inference_cf.path_decode, the
PATH0 runner's site construction, or any primary decision code. Own decision streams, C/U, sites, donor spans,
eligibility, suffix ED (TTA0 auditor ``own_lev``), criteria, label and ASR_HARM_ALIGNMENT; canonical metric primitives only.

prerun -> PASS_TO_P2_PATH0 | BLOCK_BEFORE_P2_PATH0
post   -> P2_PATH0_AUDIT: PASS | BLOCK   (--phase primary: before references; --phase full: after secondary)
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

FREEZE = "fc3602d"
CONFIG = "configs/inference_cf/p2_path0.json"
SITES = "docs/inference_cf/P2_PATH0_SITES.json"
BASE = ROOT / "results/inference_cf/p2path0"
PLAN_REL = "results/inference_cf/p2path0/plan_sealed.json"
EOS = 50257


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def st(tokens, term):
    return list(tokens) + ([EOS] if term == "eos" else [])


def div(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return None if (i == len(a) == len(b)) else i


def own_sites(plan4_rows, a4_rows) -> list[dict]:
    out = []
    for r, a in zip(plan4_rows, a4_rows):
        B, A, F = st(r["y_B"], r["y_B_terminated"]), st(r["y_A"], r["y_A_terminated"]), st(a["tokens"], a["terminated"])
        resc = F != B
        k = div(F, B) if resc else div(B, A)
        donor = B if resc else A
        cl = {}
        for L in (1, 3):
            span = []
            for t in donor[k:k + L]:
                span.append(t)
                if t == EOS:
                    break
            b = k + len(span)
            d0 = tau.own_lev(F[b:], B[b:]) if resc else tau.own_lev(B[b:], A[b:])
            cl[str(L)] = (span, b, d0)
        out.append({"utterance_id": r["utterance_id"], "role": "RESCUE" if resc else "INDUCE", "k": k, "B0": B[k],
                    "alt": F[k] if resc else A[k], "prefix": B[:k], "cl": cl, "self": [F[k]] if resc else [B[k]]})
    return out


def cmd_prerun(args) -> dict:
    from transformers import GenerationConfig
    c = cfg()
    checks = {f"frozen:{p}": sha(ROOT / p) == tau.blob_sha(FREEZE, p) for p in
              ("docs/inference_cf/P2_PATH0_SPEC.md", "docs/inference_cf/P2_PATH0_CODEX_DESIGN.md", SITES, CONFIG)}
    checks["sites_hash"] = sha(ROOT / SITES) == c["sites_byte_sha256"]
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    S = json.loads((ROOT / SITES).read_text())
    plan4 = json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())
    seal4 = json.loads((ROOT / "results/inference_cf/p2tta_a4/output_seal.json").read_text())
    D = [(i, r) for i, r in enumerate(plan4["rows"]) if r["group"] == "D"]
    a4rows = []
    for i, _ in D:
        rel = f"results/inference_cf/p2tta_a4/run1/rows/{i:02d}.json"
        checks.setdefault("a4_rows_sealed", True)
        checks["a4_rows_sealed"] &= seal4["files"][rel] == sha(ROOT / rel)
        a4rows.append(json.loads((ROOT / rel).read_text())["A4"])
    mine = own_sites([r for _, r in D], a4rows)
    th = lambda a: hashlib.sha256(json.dumps(a, separators=(",", ":")).encode("ascii")).hexdigest()
    ok = len(mine) == len(S["rows"]) == 12
    for m_, s in zip(mine, S["rows"]):
        ok &= (m_["utterance_id"] == s["utterance_id"] and m_["role"] == s["role"] and m_["k"] == s["site_index"] and m_["B0"] == s["B0_token"]
               and m_["alt"] == s["alternative_token"] and m_["prefix"] == s["common_prefix"] and th(m_["prefix"]) == s["common_prefix_sha256"]
               and m_["self"] == s["self_clamp"] and all(m_["cl"][L][0] == s["clamps"][L]["tokens"] and m_["cl"][L][1] == s["clamps"][L]["release_index"]
                                                         and m_["cl"][L][2] == s["clamps"][L]["suffix_baseline_distance"]
                                                         and (m_["cl"][L][2] > 0) == s["clamps"][L]["suffix_eligible"] for L in ("1", "3")))
    checks["sites_recomputed_independently"] = bool(ok)
    checks["population"] = (sum(x["role"] == "RESCUE" for x in mine), sum(x["role"] == "INDUCE" for x in mine)) == (5, 7) and \
        [sum(x["role"] == R and x["cl"][L][2] > 0 for x in mine) for R, L in (("RESCUE", "1"), ("RESCUE", "3"), ("INDUCE", "1"), ("INDUCE", "3"))] == [4, 4, 6, 5]
    gen = GenerationConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    sup, beg = set(gen.suppress_tokens or []), set(gen.begin_suppress_tokens or [])
    checks["forced_tokens_allowed"] = all(t not in sup and not (x["k"] + j == 0 and t in beg)
                                          for x in mine for L in ("1", "3") for j, t in enumerate(x["cl"][L][0]))
    checks["no_interior_controls"] = all(t < EOS for r, a in zip([r for _, r in D], a4rows) for t in r["y_B"] + r["y_A"] + a["tokens"])
    S_ = c["suffix_decision"]
    checks["thresholds"] = (S_["row_distance_reduction_min"] == 0.5 and S_["RESCUE"]["aggregate_distance_reduction_min"] == 0.5
                            and S_["INDUCE"]["aggregate_distance_reduction_min"] == 0.3 and S_["RESCUE"]["minimum_success_dialogues"] == 2
                            and S_["INDUCE"]["minimum_success_dialogues"] == 2 and S_["RESCUE"]["minimum_eligible"] == 3 == S_["INDUCE"]["minimum_eligible"])
    a4c = json.loads((ROOT / "configs/inference_cf/p2_tta_a4.json").read_text())
    checks["a4_inherited"] = c["A4_inherited"]["trainables"] == a4c["trainables"] and c["A4_inherited"]["optimization"] == a4c["optimization"]
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False
    src = (ROOT / "experiments/inference_cf_p2path0.py").read_text()
    code = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("cmd_run", "cmd_prepare"))
    checks["runner_reference_steering_free"] = not any(t in code for t in ("load_references", "inference_cf_p2_evaluate", "corpus_metrics", "ReadoutDirection",
                                                                          "native_lid", "_edit_hook", "seq_decode", "num_beams=", "do_sample", "logit_bias"))
    checks["barrier_before_clamps"] = code.index("phase-1 barrier failed") < code.index("Phase 2")
    dsrc = (ROOT / "src/csasr/inference_cf/path_decode.py").read_text()
    checks["clamp_is_exact_override"] = "chosen = forced[t - site]" in dsrc and "cached.Branch(bundle, encoded, list(prompt)" in dsrc and \
        "processed_argmax(logits, t, suppress, begin)" in dsrc and "new = [chosen]" in dsrc
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2path0_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(p2path0_analyze|path_decode|inference_cf_p2path0\b)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2path0.py").exists()
    budget = c["compute"]
    checks["compute_plan"] = budget["jobs_max"] == 1 and budget["optimizer_steps"] == 24 and budget["adapted_decodes"] == 48 and budget["hard_minutes"] == 30
    v = "PASS_TO_P2_PATH0" if all(checks.values()) else "BLOCK_BEFORE_P2_PATH0"
    return {"schema": "p2_path0_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def own_criteria(rows, c):
    S = c["suffix_decision"]
    res = {}
    for role in ("RESCUE", "INDUCE"):
        for L in ("1", "3"):
            el = [x for x in rows if x["role"] == role and x[L]["d0"] > 0]
            n = len(el)
            succ = [x for x in el if 1 - x[L]["dt"] / x[L]["d0"] >= 0.5 - 1e-12 and (role == "RESCUE" or x[L]["dB"] > 0)]
            need = n // 2 + 1 if role == "RESCUE" else max(3, -(-n // 2))
            pooled = 1 - sum(x[L]["dt"] for x in el) / sum(x[L]["d0"] for x in el) if n else None
            ok = n >= 3 and len(succ) >= need and pooled is not None and pooled >= S[role]["aggregate_distance_reduction_min"] - 1e-12 \
                and len({x["dlg"] for x in succ}) >= 2
            res[f"{role}_{L}_PASS"] = bool(ok)
            res[f"{role}_{L}_detail"] = {"n": n, "succ": len(succ), "pooled": pooled}
    return res


def cmd_post(args) -> dict:
    c = cfg()
    run = ROOT / "results/inference_cf/p2path0/run1"
    prim = json.loads((BASE / "primary_analysis.json").read_text())
    m = json.loads((run / "manifest.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    rt = json.loads((run / "runtime.json").read_text())
    S = json.loads((ROOT / SITES).read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "primary_bound": prim["manifest_hash"] == m["manifest_hash"] and prim["references_used"] is False,
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"] and rt["model_grads_none"]}
    if args.phase == "full":
        seal = json.loads((BASE / "output_seal.json").read_text())
        checks["seal_committed_unchanged"] = committed("results/inference_cf/p2path0/output_seal.json") and \
            all(sha(ROOT / p) == h for p, h in seal["files"].items()) and seal["primary_label"] == prim["label"]
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(12)]
    plan4 = json.loads((ROOT / "results/inference_cf/p2tta_a4/plan_sealed.json").read_text())
    a4rows = [json.loads((ROOT / s["sources"]["A4"]["path"]).read_text())["A4"] for s in S["rows"]]
    mine = own_sites([plan4["rows"][s["parent_A3_index"]] for s in S["rows"]], a4rows)
    own, bad = [], []
    for r, x, s, a4 in zip(rows, mine, S["rows"], a4rows):
        p4 = plan4["rows"][s["parent_A3_index"]]
        B, A = st(p4["y_B"], p4["y_B_terminated"]), st(p4["y_A"], p4["y_A_terminated"])
        F = st(r["FREE"]["tokens"], r["FREE"]["terminated"])
        okr = (r["status"] == "ok" and r["identity"] == x["utterance_id"] and r["role"] == x["role"] and r["site"] == x["k"]
               and r["FREE"]["tokens"] == a4["tokens"] and r["FREE"]["terminated"] == a4["terminated"]
               and r["SELF"]["tokens"] == r["FREE"]["tokens"] and r["SELF"]["terminated"] == r["FREE"]["terminated"]
               and r["theta0_equals_B0"] and r["effective_equals_checkpoint_bf16"] and r["lang_equals_sealed"] and r["reset_ok_phase1"] and r["reset_ok_phase2"])
        for arm, toks in (("SELF", x["self"]), ("L1", x["cl"]["1"][0]), ("L3", x["cl"]["3"][0])):
            tr = r[arm]["trace"]
            okr &= [f["token"] for f in tr["forced"]] == toks and tr["prefix_ok"] and tr["suppression_ok"] and tr["positions_ok"]
            out_st = st(r[arm]["tokens"], r[arm]["terminated"])
            okr &= out_st[:x["k"]] == x["prefix"] and out_st[x["k"]:x["k"] + len(toks)] == toks
        if not okr:
            bad.append(x["utterance_id"])
        e = {"role": x["role"], "dlg": s["dialogue_id"]}
        for L in ("1", "3"):
            b = x["cl"][L][1]
            O = st(r[f"L{L}"]["tokens"], r[f"L{L}"]["terminated"])
            if x["role"] == "RESCUE":
                e[L] = {"d0": tau.own_lev(F[b:], B[b:]), "dt": tau.own_lev(O[b:], B[b:])}
            else:
                e[L] = {"d0": tau.own_lev(B[b:], A[b:]), "dt": tau.own_lev(O[b:], A[b:]), "dB": tau.own_lev(O[b:], B[b:])}
        own.append(e)
    checks["rows_sites_clamps_reconstruction_self"] = not bad
    cr = own_criteria(own, c)
    passes = {k: v for k, v in cr.items() if k.endswith("_PASS")}
    valid = all(checks.values())
    lab = ("P2_PATH0_INVALID" if not valid else "P2_PATH0_SINGLE_TOKEN_CAUSAL" if passes["RESCUE_1_PASS"] and passes["INDUCE_1_PASS"]
           else "P2_PATH0_SHORT_PREFIX_CAUSAL" if passes["RESCUE_3_PASS"] and passes["INDUCE_3_PASS"] else "P2_PATH0_ASYMMETRIC" if any(passes.values())
           else "P2_PATH0_NO_LOCAL_BRANCH_CAUSALITY")
    checks["passes_agree"] = passes == prim["passes"]
    checks["pooled_agree"] = all(abs((cr[f"{R}_{L}_detail"]["pooled"] or 0) - (prim["criteria"][f"{R}_{L}"]["pooled_reduction"] or 0)) <= 1e-12
                                 for R in ("RESCUE", "INDUCE") for L in ("1", "3"))
    checks["label_agrees"] = lab == prim["label"]
    sec_out = None
    if args.phase == "full":
        from experiments.inference_cf_p2_evaluate import load_references
        from transformers import WhisperProcessor
        sec = json.loads((BASE / "secondary_analysis.json").read_text())
        tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
        refs = load_references()
        zh = lambda u, text: tau.counts(refs[u]["reference"], text)
        flags = {}
        for L in ("1", "3"):
            rr = [(x, r) for x, r, e in zip(mine, rows, own) if x["role"] == "RESCUE" and e[L]["d0"] > 0]
            cF = [zh(x["utterance_id"], tok.decode(r["FREE"]["tokens"], skip_special_tokens=True)) for x, r in rr]
            cL = [zh(x["utterance_id"], tok.decode(r[f"L{L}"]["tokens"], skip_special_tokens=True)) for x, r in rr]
            zr = sum(a[4] - b[4] for a, b in zip(cF, cL))
            flags[f"rescue_{L}"] = passes[f"RESCUE_{L}_PASS"] and zr >= 2 and sum(b[4] < a[4] for a, b in zip(cF, cL)) >= 2 and \
                sum(b[0] for b in cL) <= sum(a[0] for a in cF)
            uu = [(x, r) for x, r, e in zip(mine, rows, own) if x["role"] == "INDUCE" and e[L]["d0"] > 0]
            cF = [zh(x["utterance_id"], tok.decode(r["FREE"]["tokens"], skip_special_tokens=True)) for x, r in uu]
            cL = [zh(x["utterance_id"], tok.decode(r[f"L{L}"]["tokens"], skip_special_tokens=True)) for x, r in uu]
            flags[f"induce_{L}"] = passes[f"INDUCE_{L}_PASS"] and sum(b[4] - a[4] for a, b in zip(cF, cL)) >= 2 and \
                sum(b[4] > a[4] for a, b in zip(cF, cL)) >= 2
        rs, ind = flags["rescue_1"] or flags["rescue_3"], flags["induce_1"] or flags["induce_3"]
        own_flag = "BIDIRECTIONAL" if rs and ind else "RESCUE_ONLY" if rs else "INDUCTION_ONLY" if ind else "NONE"
        checks["secondary_flag_agrees"] = own_flag == sec["ASR_HARM_ALIGNMENT"] and sec["primary_label"] == prim["label"]
        sec_out = own_flag
    v = "P2_PATH0_AUDIT: PASS" if all(checks.values()) else "P2_PATH0_AUDIT: BLOCK"
    return {"schema": "p2_path0_post_audit_v1", "phase": args.phase, "verdict": v, "label": lab, "passes": passes,
            "ASR_HARM_ALIGNMENT": sec_out, "checks": checks, "criteria": cr, "failures": bad[:12], "git_commit": _git("rev-parse", "HEAD")}


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
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "passes", "ASR_HARM_ALIGNMENT")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
