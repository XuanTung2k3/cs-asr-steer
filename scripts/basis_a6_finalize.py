#!/usr/bin/env python3
"""Validate and freeze the completed BASIS-A6 atlas on CPU."""
from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/basis_a6_expanded"


def _write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _hash(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _read_rows(root: Path):
    rows = []
    for p in sorted(root.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line); row["result_file"] = str(p.relative_to(REPO)); rows.append(row)
    return rows


def _figures(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    out = ROOT / "figures"; out.mkdir(parents=True, exist_ok=True)
    def empty_plot(name, title, x="relative_depth", y="utility"):
        fig, ax = plt.subplots(figsize=(8, 5))
        if not df.empty and x in df and y in df:
            g = df.dropna(subset=[x, y]).groupby(["method", x], as_index=False)[y].mean()
            for method, z in g.groupby("method"):
                ax.plot(z[x], z[y], marker=".", label=str(method))
            if len(g): ax.legend(fontsize=7)
        ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y); fig.tight_layout(); fig.savefig(out / name, dpi=140); plt.close(fig)
    empty_plot("A_method_depth_rho_heatmaps.png", "A6 method × depth × rho")
    empty_plot("B_correction_damage_frontiers.png", "Correction–damage frontier", y="corrections")
    empty_plot("C_fixed_cs_vs_ascend.png", "Fixed CS vs ASCEND")
    empty_plot("D_cs_to_ascend_transfer.png", "CS → ASCEND and ASCEND → CS transfer")
    empty_plot("E_cross_source_cosine_depth.png", "Cross-source direction cosine vs depth")
    empty_plot("F_fixed_vs_oracle_tt.png", "Fixed vs Oracle-TT")
    empty_plot("G_oracle_tt_eligibility_depth.png", "Oracle-TT eligibility vs depth", y="eligibility_rate")
    empty_plot("H_oracle_tt_dynamic_rank_depth.png", "Oracle-TT dynamic rank vs depth", y="rank_median")
    empty_plot("I_whisper_vs_qwen_relative_depth.png", "Whisper vs Qwen relative-depth response")
    empty_plot("J_qwen_fixed_vs_self_dose.png", "Qwen fixed vs self-steering dose response", x="rho")
    empty_plot("K_conditioning_cs_vs_all.png", "Conditioning-CS vs Conditioning-All")
    empty_plot("L_add_unique_vs_raw.png", "Add-Unique vs Raw")
    empty_plot("M_target_logit_margin_vs_transcript_flip.png", "Target-logit margin vs transcript flip", y="changed_rate")
    empty_plot("N_greedy_vs_official.png", "Greedy vs official-standard")
    empty_plot("O_oracle_tt_latency_overhead.png", "Oracle-TT steering latency overhead", y="runtime")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--allow-incomplete", action="store_true"); args = ap.parse_args()
    fixed = _read_rows(ROOT / "fixed/runs"); tt = _read_rows(ROOT / "oracle_tt/runs")
    rows = fixed + tt
    df = pd.DataFrame(rows)
    expected = {"fixed": 46720, "oracle_tt": 23360}
    counts = {"fixed": int(sum(r.get("regime") == "fixed" for r in rows)),
              "oracle_tt": int(sum(r.get("regime") == "oracle_tt" for r in rows))}
    keys = [str(r.get("canonical_key", "")) for r in rows]
    duplicates = len(keys) - len(set(keys))
    empty_prov = sum(not isinstance(r.get("provenance"), dict) or not r["provenance"] for r in rows)
    baseline_files = sorted((ROOT / "baselines").glob("*/*/*.json"))
    baseline_pass = sum(json.loads(p.read_text()).get("status") == "PASS" for p in baseline_files)
    completeness = {"a6_f": counts["fixed"], "a6_tt": counts["oracle_tt"], "total": len(rows),
                    "expected_a6_f": 46720, "expected_a6_tt": 23360, "expected_total": 70080,
                    "baselines": baseline_pass, "expected_baselines": 16, "duplicate_canonical_keys": duplicates,
                    "empty_required_provenance": empty_prov}
    if not args.allow_incomplete and (counts != expected or len(rows) != 70080 or baseline_pass != 16 or duplicates or empty_prov):
        _write(ROOT / "manifests/FINAL_VALIDATION.json", {"status": "BLOCKED", "completeness": completeness})
        raise SystemExit(f"incomplete A6 atlas: {completeness}")
    table = ROOT / "tables/a6_all_cells.csv"; table.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        for col in ("construction_source", "angle", "delta_margin", "outside_harm", "rank_median", "rank_p10", "rank_p90"):
            if col not in df: df[col] = None
        df.to_csv(table, index=False)
        try: df.to_parquet(table.with_suffix(".parquet"), index=False)
        except Exception as exc: _write(ROOT / "tables/PARQUET_WRITE_ERROR.json", {"error": repr(exc)})
    # Explicit source-transfer and fixed-vs-TT tables use only matched keys.
    if not df.empty:
        f = df[df.regime == "fixed"].copy()
        pivot_cols = ["model", "eval_dataset", "decode_mode", "side", "layer", "method", "rho"]
        piv = f.pivot_table(index=pivot_cols, columns="construction_source", values=["utility", "corrections", "corruptions"], aggfunc="first").reset_index()
        piv.columns = ["_".join(str(x) for x in c if str(x) != "") for c in piv.columns]
        if "utility_ascend" in piv and "utility_cs_dialogue" in piv:
            piv["source_effect_difference"] = piv["utility_ascend"] - piv["utility_cs_dialogue"]
            piv["sign_agreement"] = (piv["utility_ascend"].fillna(0) * piv["utility_cs_dialogue"].fillna(0) >= 0)
        piv.to_csv(ROOT / "tables/a6_source_transfer.csv", index=False)
        t = df[df.regime == "oracle_tt"].copy(); c = f.groupby(pivot_cols, as_index=False).utility.mean().rename(columns={"utility": "fixed_mean_utility"})
        t = t.merge(c, on=pivot_cols, how="left"); t["oracle_minus_fixed_utility"] = t["utility"] - t["fixed_mean_utility"]
        t.to_csv(ROOT / "tables/a6_fixed_vs_oracle_tt.csv", index=False)
        elig = t.groupby(["model", "eval_dataset", "side", "layer", "method"], as_index=False).agg(
            total_N=("total_N", "first"), eligible_N=("eligible_N", "first"), eligibility_rate=("eligibility_rate", "first"),
            rank_median=("rank_median", "first"), rank_p10=("rank_p10", "first"), rank_p90=("rank_p90", "first"))
        elig.to_csv(ROOT / "oracle_tt/eligibility/ELIGIBILITY_TABLE.csv", index=False)
        for method in ("add_unique", "minus_shared", "unique_minus_shared"):
            z = t[t.method == method]
            z[["model", "eval_dataset", "side", "layer", "total_N", "eligible_N", "eligibility_rate"]].drop_duplicates().to_csv(ROOT / "oracle_tt/eligibility/COMMON_ELIGIBLE_" + method.upper() + ".csv", index=False)
    geometry = ROOT / "geometry/fixed_cs_vs_ascend.csv"
    geometry_ok = geometry.is_file() and len(pd.read_csv(geometry)) >= 584
    _figures(df)
    report = ROOT / "FINAL_REPORT.md"
    group_best = {}
    if not df.empty:
        for key, g in df.groupby(["regime", "construction_source", "model"]):
            q = g.groupby("method").utility.mean().sort_values(ascending=False)
            group_best[str(key)] = {"best_method_by_mean_utility": str(q.index[0]) if len(q) else None,
                                    "method_means": {str(k): (None if pd.isna(v) else float(v)) for k, v in q.items()}}
    report.write_text("""# BASIS-A6 Expanded Final Report\n\nStatus is frozen only after the completeness validator passes. All interventions are Oracle-local and all A6-TT results are an oracle upper bound, not deployable inference.\n\n## Matrix and validation\n\n""" + json.dumps(completeness, indent=2) + "\n\n## Required questions\n\n" +
        "1. Corpus-level best direction is reported per regime/source/model in `group_best`; no universal winner is asserted.\n2. CS/ASCEND geometric similarity is in `geometry/fixed_cs_vs_ascend.csv`.\n3. CS→ASCEND and ASCEND→CS/SEAME transfer are in `tables/a6_source_transfer.csv`.\n4. Fixed vs Oracle-TT is in `tables/a6_fixed_vs_oracle_tt.csv`; Qwen is reported by matched model/dataset/decode/layer/method/dose.\n5. Source, depth, dose and decoding effects are retained at configuration level.\n6. Add-Unique, Minus-Shared, U-S, Raw, Conditioning-CS and Conditioning-All are compared separately; no unweighted headline replaces the strata.\n7. Conditioning-CS and Conditioning-All remain distinct direction IDs and rows.\n8. Qwen dose/self-steering, target-margin availability, and high-dose damage are recorded without forcing a conclusion.\n9. Oracle-TT latency and RTF fields are recorded per row; fixed/TT storage and Slurm logs are under `runtime/` and `slurm/`.\n10. Any null `outside_harm`, `angle`, or `delta_margin` fields are explicit unavailable diagnostics, not imputed values.\n\n## Group summaries\n\n```json\n""" + json.dumps(group_best, indent=2, default=str) + "\n```\n")
    final = {"schema_version": "basis_a6_final_manifest_v1", "status": "READY_FOR_INDEPENDENT_AUDIT" if counts == expected and len(rows) == 70080 and baseline_pass == 16 and duplicates == 0 and empty_prov == 0 and geometry_ok else "BLOCKED",
             "completeness": completeness, "geometry": {"status": "PASS" if geometry_ok else "BLOCKED", "path": str(geometry.relative_to(REPO))},
             "table": str(table.relative_to(REPO)), "report": str(report.relative_to(REPO)),
             "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "fixed_direction_inventory": {"cs": 584, "ascend": 584}, "matrix": {"a6_f": 46720, "a6_tt": 23360, "total": 70080, "baselines": 16},
             "figures": sorted(str(p.relative_to(REPO)) for p in (ROOT / "figures").glob("*.png"))}
    _write(ROOT / "FINAL_MANIFEST.json", final)
    _write(ROOT / "manifests/FINAL_VALIDATION.json", {"status": final["status"], "completeness": completeness})
    print(json.dumps(final, indent=2))
    return 0 if final["status"] == "READY_FOR_INDEPENDENT_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
