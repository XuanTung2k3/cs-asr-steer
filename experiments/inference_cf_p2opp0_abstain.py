#!/usr/bin/env python
"""P2-OPP0-ABSTAIN: post-hoc DEVELOPMENT diagnostic of A2+G1A on fixed100 from already-sealed PATH4-R1 outputs (no GPU).

G1A = original content/content G1 at the first theta0/A2 disagreement; ABSTAIN TO A2 whenever that disagreement involves
EOS; no-trigger = A2. Because PATH4-R1 content/content G1 is bit-identical to the original G1 (direct-original comparator),
the derived G1A output of each row is chosen mechanically from sealed artifacts by the sealed live dispatch mode only:

    CONTENT_G1   -> sealed PATH4 G output (row G1)
    EOS_BOUNDARY -> sealed A2 output (unchanged)
    NO_TRIGGER   -> sealed A2 output (unchanged)

derive    (reference-free) verify the committed PATH4 output seal + file hashes, build derived_g1a.json (sources/hashes).
evaluate  (fixed100 references, already exposed; only after derived_g1a.json is committed) B0/AUTO/A2/PATH4-G/G1A metrics
          and the five Phase-0 questions. Descriptive only; never used to tune G1A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json, digest

P4 = "results/inference_cf/p2path4"
OUT = "results/inference_cf/p2opp0"
DERIVED = f"{OUT}/derived_g1a.json"
EOS = 50257


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return tracked and hashlib.sha256(blob).hexdigest() == sha(ROOT / rel)


def derive_row(mode: str, g: dict, a2: dict) -> tuple[dict, str]:
    """Pure G1A derivation from the sealed live dispatch mode (no reference, no row ID)."""
    if mode == "CONTENT_G1":
        return {"tokens": g["tokens"], "terminated": g["terminated"], "text": g["text"]}, "PATH4_content_G1"
    if mode in ("EOS_BOUNDARY", "NO_TRIGGER"):
        return {"tokens": a2["tokens"], "terminated": a2["terminated"], "text": a2["text"]}, ("A2_EOS_abstain" if mode == "EOS_BOUNDARY" else "A2_no_trigger")
    raise ValueError(f"unknown dispatch mode {mode}")


def cmd_derive(args) -> None:
    seal_rel = f"{P4}/output_seal.json"
    seal = json.loads((ROOT / seal_rel).read_text())
    checks = {"seal_committed": committed(seal_rel), "sealed_files_unchanged": all(sha(ROOT / p) == h for p, h in seal["files"].items()),
              "primary_audit": json.loads((ROOT / P4 / "primary_audit.json").read_text())["verdict"] == "P2_PATH4_AUDIT: PASS",
              "post_audit": json.loads((ROOT / P4 / "post_audit.json").read_text())["verdict"] == "P2_PATH4_AUDIT: PASS"}
    plan = json.loads((ROOT / P4 / "plan_sealed.json").read_text())
    rows, srcs = [], {}
    for i, x in enumerate(plan["rows"]):
        rel = f"{P4}/run1/rows/{i:03d}.json"
        r = json.loads((ROOT / rel).read_text())
        u = x["runtime"]["utterance_id"]
        a2 = x["analysis"]["A2"]
        assert r["identity"] == u and r["status"] == "ok" and seal["files"][rel] == sha(ROOT / rel)
        assert r["A2_free"]["tokens"] == a2["tokens"] and r["A2_free"]["terminated"] == a2["terminated"]
        out, src = derive_row(r["mode"], r["G1"], a2)
        srcs[src] = srcs.get(src, 0) + 1
        rows.append({"canonical_index": i, "utterance_id": u, "dialogue_id": x["analysis"]["dialogue_id"], "path4_mode": r["mode"],
                     "source": src, "source_file": rel, "source_sha256": seal["files"][rel], "G1A": out,
                     "PATH4_G": {"tokens": r["G1"]["tokens"], "terminated": r["G1"]["terminated"]},
                     "A2": {"tokens": a2["tokens"], "terminated": a2["terminated"]},
                     "changed_vs_A2": (out["tokens"], out["terminated"]) != (a2["tokens"], a2["terminated"]),
                     "differs_from_PATH4_G": (out["tokens"], out["terminated"]) != (r["G1"]["tokens"], r["G1"]["terminated"])})
    checks["all_100"] = len(rows) == 100 and len({r["utterance_id"] for r in rows}) == 100
    checks["eos_rows_equal_A2"] = all(not r["changed_vs_A2"] for r in rows if r["path4_mode"] != "CONTENT_G1")
    checks["content_rows_equal_PATH4"] = all(not r["differs_from_PATH4_G"] for r in rows if r["path4_mode"] == "CONTENT_G1")
    doc = {"schema": "p2_opp0_abstain_derived_v1", "kind": "post-hoc development diagnostic (no GPU, no new decode)",
           "rule": {"CONTENT_G1": "sealed PATH4 G (== original G1)", "EOS_BOUNDARY": "sealed A2 (abstain)", "NO_TRIGGER": "sealed A2"},
           "path4_output_seal": {"path": seal_rel, "sha256": sha(ROOT / seal_rel), "seal_hash": seal["seal_hash"]},
           "checks": checks, "source_counts": srcs, "changed_vs_A2": sum(r["changed_vs_A2"] for r in rows),
           "differs_from_PATH4_G": [r["utterance_id"] for r in rows if r["differs_from_PATH4_G"]],
           "rows": rows, "references_used": False, "created_unix": time.time()}
    doc["derived_hash"] = digest(doc)
    out = ROOT / DERIVED
    if out.exists():
        raise FileExistsError("derived output exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps(checks))
    atomic_json(out, doc)
    print(json.dumps({k: doc[k] for k in ("derived_hash", "checks", "source_counts", "changed_vs_A2", "differs_from_PATH4_G")}, indent=1))


def cmd_evaluate(args) -> None:
    from csasr.evaluation.canonical import corpus_metrics, correction_corruption
    from csasr.evaluation.retention import embedded_en_retention, matrix_zh_retention
    from experiments.inference_cf_p2seq_analyze import outside, utt_counts
    from experiments.inference_cf_p2_evaluate import load_references
    from csasr.inference_cf.episodic_tta import severe_truncation
    if not committed(DERIVED):
        raise PermissionError("derived G1A outputs must be committed before references are opened")
    d = json.loads((ROOT / DERIVED).read_text())
    if digest({k: v for k, v in d.items() if k != "derived_hash"}) != d["derived_hash"]:
        raise ValueError("derived hash")
    plan = json.loads((ROOT / P4 / "plan_sealed.json").read_text())
    X = {x["runtime"]["utterance_id"]: x for x in plan["rows"]}
    refs = load_references()
    ids = [r["utterance_id"] for r in d["rows"]]
    R = {u: refs[u]["reference"] for u in ids}
    H, T, L = {}, {}, {}
    for r in d["rows"]:
        u, x = r["utterance_id"], X[r["utterance_id"]]
        p4 = json.loads((ROOT / r["source_file"]).read_text())
        H[u] = {"B0_FORCED": x["analysis"]["y_B_text"], "B0_AUTO": x["analysis"]["AUTO"]["text"], "A2": x["analysis"]["A2"]["text"],
                "PATH4_G": p4["G1"]["text"], "A2_G1A": r["G1A"]["text"]}
        T[u] = {"B0_FORCED": x["audit"]["B0"]["terminated"], "B0_AUTO": x["analysis"]["AUTO"]["terminated"], "A2": x["analysis"]["A2"]["terminated"],
                "PATH4_G": p4["G1"]["terminated"], "A2_G1A": r["G1A"]["terminated"]}
        L[u] = {"B0_FORCED": len(x["audit"]["B0"]["tokens"]), "B0_AUTO": len(x["analysis"]["AUTO"]["tokens"]), "A2": len(x["analysis"]["A2"]["tokens"]),
                "PATH4_G": len(p4["G1"]["tokens"]), "A2_G1A": len(r["G1A"]["tokens"])}
    SY = ("B0_FORCED", "B0_AUTO", "A2", "PATH4_G", "A2_G1A")
    C = {u: {s: utt_counts(R[u], H[u][s]) for s in SY} for u in ids}
    Rs, Bs = [R[u] for u in ids], [H[u]["B0_FORCED"] for u in ids]
    M = {}
    for s in SY:
        Hs = [H[u][s] for u in ids]
        cm = corpus_metrics(Rs, Hs)
        e = {"zh": sum(C[u][s][4] for u in ids), "poi": sum(C[u][s][0] for u in ids), "mixed": sum(C[u][s][2] for u in ids),
             "pier": cm["pier"], "mer": cm["mer"], "en_wer": cm["en_wer"], "zh_cer": cm["zh_cer"], "caps": sum(T[u][s] == "cap" for u in ids),
             "severe_vs_B0": sum(severe_truncation(L[u]["B0_FORCED"], L[u][s], T[u][s]) for u in ids)}
        if s != "B0_FORCED":
            cc = correction_corruption(Rs, Bs, Hs)
            outs = [outside(a, b, h) for a, b, h in zip(Rs, Bs, Hs)]
            ob = sum(o["baseline_correct_outside"] for o in outs)
            e.update(poi_corrections=cc["corrections"], poi_corruptions=cc["corruptions"], zh_retention=matrix_zh_retention(Rs, Bs, Hs)["rate"],
                     en_retention=embedded_en_retention(Rs, Bs, Hs)["rate"], outside_harm=(sum(o["outside_harm"] for o in outs) / ob) if ob else None)
        M[s] = e
    per = []
    for r in d["rows"]:
        u = r["utterance_id"]
        per.append({"utterance_id": u, "dialogue_id": r["dialogue_id"], "path4_mode": r["path4_mode"], "source": r["source"],
                    "r_ZH_A2_minus_G1A": C[u]["A2"][4] - C[u]["A2_G1A"][4], "dPOI_G1A_minus_A2": C[u]["A2_G1A"][0] - C[u]["A2"][0],
                    "dMixed_G1A_minus_A2": C[u]["A2_G1A"][2] - C[u]["A2"][2],
                    "r_ZH_A2_minus_PATH4G": C[u]["A2"][4] - C[u]["PATH4_G"][4]})
    pe = {e["utterance_id"]: e for e in per}
    p2_12 = [r["utterance_id"] for r in json.loads((ROOT / "docs/inference_cf/P2_PATH2_PANEL.json").read_text())["rows"]]
    pos = {u: max(e["r_ZH_A2_minus_G1A"], 0) for u, e in pe.items()}
    Rp = sum(pos.values())
    q = {"Q1_EOS_harm_eliminated": all(e["r_ZH_A2_minus_G1A"] == 0 and e["dPOI_G1A_minus_A2"] == 0 and e["dMixed_G1A_minus_A2"] == 0
                                       for e in per if e["path4_mode"] == "EOS_BOUNDARY"),
         "Q2_content_G1_preserved": all(r["G1A"]["tokens"] == r["PATH4_G"]["tokens"] and r["G1A"]["terminated"] == r["PATH4_G"]["terminated"]
                                        for r in d["rows"] if r["path4_mode"] == "CONTENT_G1"),
         "Q3_U1004_S0_221_equals_A2": next(r for r in d["rows"] if r["utterance_id"] == "ZH-CN_U1004_S0_221")["changed_vs_A2"] is False,
         "Q4_any_EOS_row_differs_from_A2": any(r["changed_vs_A2"] for r in d["rows"] if r["path4_mode"] == "EOS_BOUNDARY"),
         "Q5_positive_rescue": {"R_plus": Rp, "R_minus": sum(max(-e["r_ZH_A2_minus_G1A"], 0) for e in per),
                                "rescue_rows": [u for u, v in pos.items() if v > 0], "from_PATH2_12": sum(pos[u] for u in p2_12),
                                "share_PATH2_12": (sum(pos[u] for u in p2_12) / Rp) if Rp else None}}
    res = {"schema": "p2_opp0_abstain_eval_v1", "derived_hash": d["derived_hash"], "metrics": M, "questions": q,
           "per_row_triggered": [e for e in per if e["path4_mode"] != "NO_TRIGGER"], "diagnostic_only": True}
    out = ROOT / OUT / "evaluation.json"
    if out.exists():
        raise FileExistsError("evaluation exists")
    from experiments.inference_cf_p2dir_analyze import jsonable
    atomic_json(out, jsonable(res))
    print(json.dumps({s: {k: M[s][k] for k in ("zh", "poi", "mixed", "pier", "mer", "en_wer", "zh_cer")} for s in SY}, indent=1))
    print(json.dumps(q, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("derive")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    {"derive": cmd_derive, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
