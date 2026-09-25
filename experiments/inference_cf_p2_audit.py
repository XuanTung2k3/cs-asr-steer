#!/usr/bin/env python
"""Independent P2 audit (CPU): recompute headline metrics from saved rows, re-derive validity and
selection from the frozen config, and re-check provenance and divergence attribution.

Deliberately does not import the P2 evaluator. MER totals use an independent edit-distance DP.
Per-language and PIER numbers use the project metric definitions per utterance, aggregated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.evaluation.mer import corpus_mer
from csasr.evaluation.normalization import normalize_text, segment_units
from csasr.evaluation.pier import evaluate_pois
from csasr.evaluation.retention import matrix_zh_retention

CONFIG = ROOT / "configs/inference_cf/p2_compact_development.json"


def canonical_digest(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode()).hexdigest()


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def units(text: str) -> list[str]:
    return [u.surface for u in segment_units(normalize_text(text))]


def recompute(refs: dict, ids: list[str], texts: list[str], caps: int) -> dict:
    err = n = poi = poi_err = zh_e = zh_n = en_e = en_n = 0
    for u, h in zip(ids, texts):
        r = refs[u]["reference"]
        ru = units(r)
        err += levenshtein(ru, units(h))
        n += len(ru)
        ps = evaluate_pois(r, h)
        poi += len(ps)
        poi_err += sum(not p.correct for p in ps)
        m = corpus_mer([r], [h])
        zh_n += m["num_zh_ref"]
        en_n += m["num_en_ref"]
        zh_e += round(m["zh_cer"] * m["num_zh_ref"]) if m["num_zh_ref"] else 0
        en_e += round(m["en_wer"] * m["num_en_ref"]) if m["num_en_ref"] else 0
    return {"mer": err / n, "mer_errors": err, "pier": poi_err / poi, "pier_errors": poi_err,
            "zh_cer": zh_e / zh_n, "en_wer": en_e / en_n, "caps": caps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    summ = json.loads((ROOT / args.summary).read_text())
    from experiments.inference_cf_p0_r2_prepare import build_panels
    _, ev = build_panels()
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    assert canonical_digest(ev) == r2m["evaluation_panel_hash"]
    refs = {r["utterance_id"]: r for r in ev["rows"]}
    report = {"summary": args.summary, "runs": {}, "configs": {}, "issues": []}
    matched = {}
    for d, mh in summ["manifests"].items():
        m = json.loads((ROOT / d / "manifest.json").read_text())
        ok_hash = canonical_digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"] == mh
        src_ok = {}
        for p, h in m["sources"].items():
            rel = str(Path(p).resolve().relative_to(ROOT.resolve())) if Path(p).is_absolute() else p
            blob = subprocess.run(["git", "show", f"{m['git_commit']}:{rel}"], cwd=ROOT, capture_output=True).stdout
            src_ok[rel] = "sha256:" + hashlib.sha256(blob).hexdigest() == h
        panel = json.loads((ROOT / d / "panel.json").read_text())
        ids = [r["utterance_id"] for r in panel["rows"]]
        rows = {}
        for i, u in enumerate(ids):
            rows[u] = json.loads((ROOT / d / "rows" / f"{i:03d}.json").read_text())
        statuses = {r["status"] for r in rows.values()}
        stale = sum(r["manifest_hash"] != m["manifest_hash"] or r["identity"] != u for u, r in rows.items())
        rt = json.loads((ROOT / d / "runtime.json").read_text())
        report["runs"][d] = {"manifest_hash_ok": ok_hash, "sources_match_commit": all(src_ok.values()),
                             "source_mismatches": [k for k, v in src_ok.items() if not v],
                             "statuses": sorted(statuses), "stale_rows": stale, "runtime_status": rt.get("status"),
                             "job": rt.get("job_id"), "elapsed_sec": rt.get("elapsed_sec"),
                             "peak_vram_gb": (rt.get("peak_vram_bytes") or 0) / 2**30}
        if not ok_hash or not all(src_ok.values()) or statuses != {"ok"} or stale or rt.get("status") != "completed":
            report["issues"].append(f"provenance/completeness issue in {d}")
        for L in m["matched_baseline_layers"]:
            bm = [rows[u]["systems"][f"B0M_L{L}"] for u in ids]
            matched[(d, L)] = {"texts": [x["text"] for x in bm], "tokens": [x["tokens"] for x in bm],
                               "caps": sum(x["terminated"] == "cap" for x in bm)}
            if "B0" in rows[ids[0]]["systems"]:
                report["runs"][d][f"B0M_L{L}_vs_cached_greedy_B0_token_diff_utts"] = sum(
                    rows[u]["systems"]["B0"]["tokens"] != t for u, t in zip(ids, matched[(d, L)]["tokens"]))
        for c in m["configs"]:
            name, L = c["name"], c["layer"]
            base = matched[(d, L)]
            sysr = [rows[u]["systems"][name] for u in ids]
            texts = [x["text"] for x in sysr]
            mm = recompute(refs, ids, texts, sum(x["terminated"] == "cap" for x in sysr))
            bb = recompute(refs, ids, base["texts"], base["caps"])
            rr = [refs[u]["reference"] for u in ids]
            zr = matrix_zh_retention(rr, base["texts"], texts)
            ret = zr["numerator"] / zr["denominator"]
            # divergence attribution and B-branch consistency, from raw steps
            div = unattr = bconsist = edit_at_k_changed = 0
            for x, bt in zip(sysr, base["tokens"]):
                if x["tokens"] == bt:
                    continue
                div += 1
                k = next((t for t, (p, q) in enumerate(zip(x["tokens"], bt)) if p != q), min(len(x["tokens"]), len(bt)))
                steps = {s["t"]: s for s in x["steps"]}
                if not any(s["edit"] for t, s in steps.items() if t <= k):
                    unattr += 1
                sk = steps.get(k)
                if sk is not None and k < len(bt) and sk["unsteered_next"] == bt[k]:
                    bconsist += 1
                if sk is not None and sk["edit"] and sk["next"] != sk["unsteered_next"]:
                    edit_at_k_changed += 1
            v = cfg["validity"]
            checks = {"V1": mm["caps"] <= bb["caps"] + v["max_cap_increase"],
                      "V2": mm["zh_cer"] - bb["zh_cer"] <= v["max_zh_cer_increase_abs"] + 1e-12,
                      "V3": mm["mer"] - bb["mer"] <= v["max_mer_increase_abs"] + 1e-12,
                      "V4": ret >= v["min_matrix_zh_retention"] - 1e-12,
                      "V5": bb["pier_errors"] - mm["pier_errors"] > 0}
            energy = sum(s["edit_norm"] ** 2 for x in sysr for s in x["steps"] if s["edit"])
            report["configs"][name] = {"run": d, "method": mm, "baseline": bb, "matrix_zh_retention": ret,
                                       "dpier_errors": bb["pier_errors"] - mm["pier_errors"],
                                       "dzh_cer_increase": mm["zh_cer"] - bb["zh_cer"],
                                       "dmer_increase": mm["mer"] - bb["mer"], "realized_energy": energy,
                                       "checks": checks, "valid": all(checks.values()),
                                       "diverged": div, "unattributed": unattr,
                                       "b_branch_consistent_at_divergence": bconsist,
                                       "edit_at_divergence_changed_token": edit_at_k_changed,
                                       "config": c}
    keys = list(matched)
    report["matched_baselines_identical"] = all(matched[k]["tokens"] == matched[keys[0]]["tokens"] for k in keys)
    # compare with the evaluator summary
    for e in summ["configs"]:
        a = report["configs"][e["config"]["name"]]
        em = e["metrics"]
        agree = {"pier_errors": em["pier_errors"] == a["method"]["pier_errors"],
                 "mer_errors": sum(em["mer_counts"][k] for k in ("substitutions", "deletions", "insertions"))
                 == a["method"]["mer_errors"],
                 "zh_cer": abs(em["zh_cer"] - a["method"]["zh_cer"]) < 1e-9,
                 "en_wer": abs(em["en_wer"] - a["method"]["en_wer"]) < 1e-9,
                 "caps": em["caps"] == a["method"]["caps"],
                 "retention": abs((em["matrix_zh_retention"] or 0) - a["matrix_zh_retention"]) < 1e-12,
                 "valid": e["validity"]["valid"] == a["valid"],
                 "unattributed": len(e["attribution"]["unattributed"]) == a["unattributed"]}
        a["agrees_with_evaluator"] = agree
        if not all(agree.values()):
            report["issues"].append(f"evaluator disagreement for {e['config']['name']}: {agree}")
    valid = [a for a in report["configs"].values() if a["valid"] and a["config"]["gate"] == "ER"
             and a["config"]["direction_sign"] > 0]
    order = sorted(valid, key=lambda a: (-a["dpier_errors"], a["dzh_cer_increase"], a["realized_energy"],
                                         {"id": 0, "sqrt": 1}[a["config"]["dose"]], a["config"]["alpha"]))
    report["independent_selection"] = order[0]["config"]["name"] if order else None
    evaluator_sel = (summ.get("selection") or {}).get("selected") or summ.get("final_selection")
    report["evaluator_selection"] = evaluator_sel["name"] if evaluator_sel else None
    if report["independent_selection"] != report["evaluator_selection"]:
        report["issues"].append("selection disagreement")
    report["verdict"] = "AUDIT_PASS" if not report["issues"] else "AUDIT_ISSUES"
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"verdict": report["verdict"], "issues": report["issues"],
                      "independent_selection": report["independent_selection"],
                      "evaluator_selection": report["evaluator_selection"],
                      "matched_identical": report["matched_baselines_identical"]}, indent=2))


if __name__ == "__main__":
    main()
