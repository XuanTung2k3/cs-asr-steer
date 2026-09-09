#!/usr/bin/env python
"""CPU-only BASIS-A3 aggregation, metric audit, geometry links and figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/basis_a3_raw_cond_scope_depth"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _condition_rows(root=OUT, subdir="raw_r1"):
    rows = []
    for p in sorted((root / subdir).glob("*/L*/*.json")):
        x = _read(p)
        m = x.get("metrics", {})
        t = m.get("transitions", {})
        rows.append({
            "Dataset": x["dataset"], "Side": m.get("side", x.get("side")), "Layer": x["layer"],
            "Scope": x["scope"], "rho": x["rho"],
            "MER": m.get("mer"), "PIER": m.get("pier"), "EN-WER": m.get("en_wer"),
            "Matrix-CER": m.get("zh_cer"),
            "Corr": m.get("poi_corrections", t.get("corrections")),
            "Corrupt": m.get("poi_corruptions", t.get("corruptions")),
            "Outside": m.get("outside_harm"),
            "Matrix-Ret": m.get("retention", {}).get("matrix_zh", {}).get("rate"),
            "Edited": m.get("edited_positions_or_frames"), "Energy": m.get("total_intervention_energy"),
            "FractionEdited": m.get("fraction_acoustic_frames_edited"),
            "PIER-gain": m.get("pier_gain"), "MER-gain": m.get("mer_gain"),
            "Matrix-CER-gain": m.get("zh_cer_gain"), "path": str(p.relative_to(root)),
        })
    return rows


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows({k: r.get(k) for k in fields} for r in rows)


def _num(x):
    try:
        if x in (None, "", "None"): return None
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    for rank, i in enumerate(order): out[i] = rank + 1.0
    return out


def _corr(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if _num(a) is not None and _num(b) is not None]
    if len(pairs) < 3: return None
    a, b = np.asarray(pairs).T
    if np.std(a) == 0 or np.std(b) == 0: return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if _num(a) is not None and _num(b) is not None]
    if len(pairs) < 3:
        return None
    a, b = np.asarray(pairs).T
    return _corr(_rank(a.tolist()), _rank(b.tolist()))


def _geometry_links(root, raw, cond):
    geom = _read(root / "geometry/raw_cond_decoder_geometry.json")["layers"]
    base = {int(g["layer"]): g for g in geom}
    rows = []
    for dataset in sorted({r["Dataset"] for r in raw if r["Side"] == "decoder"}):
        for layer in range(32):
            g = base[layer]
            rr = [r for r in raw if r["Dataset"] == dataset and r["Side"] == "decoder" and r["Layer"] == layer and float(r["rho"]) == .5]
            cc = [r for r in cond if r["Dataset"] == dataset and r["Side"] == "decoder" and r["Layer"] == layer and float(r["rho"]) == .5]
            for scope in ("global", "oracle_local"):
                r = next((z for z in rr if z["Scope"] == scope), None)
                c = next((z for z in cc if z["Scope"] == scope), None)
                if r is None or c is None: continue
                rows.append({"Dataset": dataset, "Layer": layer, "Scope": scope,
                             "cos_raw_cond": g["cos_raw_cond"], "angle_deg": g["angle_deg"],
                             "unit_l2": g["unit_l2"], "Raw-PIER-gain": r["PIER-gain"],
                             "Cond-PIER-gain": c["PIER-gain"], "Raw-Matrix-Ret": r["Matrix-Ret"],
                             "Cond-Matrix-Ret": c["Matrix-Ret"], "Raw-Outside": r["Outside"],
                             "Cond-Outside": c["Outside"]})
    return rows


def _write_geometry_correlations(root, links):
    metrics = ["Raw-PIER-gain", "Cond-PIER-gain", "Raw-Matrix-Ret", "Cond-Matrix-Ret",
               "Raw-Outside", "Cond-Outside"]
    out = []
    for dataset in sorted({r["Dataset"] for r in links}):
        for scope in ("global", "oracle_local"):
            z = [r for r in links if r["Dataset"] == dataset and r["Scope"] == scope]
            for g in ("cos_raw_cond", "angle_deg", "unit_l2"):
                for metric in metrics:
                    out.append({"Dataset": dataset, "Scope": scope, "geometry": g, "outcome": metric,
                                "pearson": _corr([r[g] for r in z], [r[metric] for r in z]),
                                "spearman": _spearman([r[g] for r in z], [r[metric] for r in z]),
                                "n": len([(r[g], r[metric]) for r in z if _num(r[metric]) is not None])})
    _write_csv(root / "geometry/steering_geometry_correlations.csv", out,
               ["Dataset", "Scope", "geometry", "outcome", "pearson", "spearman", "n"])
    (root / "geometry/steering_geometry_correlations.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")


def _plot(root, raw, cond, geom):
    import matplotlib.pyplot as plt
    figdir = root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    specs = [("PIER", "PIER", "Raw PIER vs depth"), ("MER", "MER", "Raw MER vs depth"),
             ("Matrix-CER", "Matrix-CER", "Raw matrix CER vs depth"),
             ("Matrix-Ret", "Matrix retention", "Raw matrix retention vs depth"),
             ("Corr", "POI corrections", "Raw POI corrections vs depth"),
             ("Outside", "Outside harm", "Raw outside harm vs depth")]
    for col, ylabel, title in specs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
        for ax, side in zip(axes, ("encoder", "decoder")):
            for ds in sorted({r["Dataset"] for r in raw}):
                for scope, ls in (("global", "-"), ("oracle_local", "--")):
                    z = [r for r in raw if r["Dataset"] == ds and r["Side"] == side and r["Scope"] == scope and _num(r[col]) is not None]
                    if z: ax.plot([r["Layer"] for r in z], [r[col] for r in z], marker="o" if side == "encoder" else "s", linestyle=ls, label=f"{ds}/{scope}")
            ax.set(xlabel=f"{side.capitalize()} layer", ylabel=ylabel, title=f"{title}: {side}"); ax.grid(alpha=.25)
        axes[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(figdir / f"{col.lower().replace('-', '_')}_vs_depth.png", dpi=160); plt.close(fig)
    # Scope rescue plots: local minus global, split by side and dataset.
    for col, ylabel in (("PIER", "Local - Global PIER"), ("Matrix-Ret", "Local - Global matrix retention")):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for ds in sorted({r["Dataset"] for r in raw}):
            for side, marker in (("encoder", "o"), ("decoder", "s")):
                vals = []
                for layer in sorted({r["Layer"] for r in raw if r["Dataset"] == ds and r["Side"] == side}):
                    g = next((r for r in raw if r["Dataset"] == ds and r["Side"] == side and r["Layer"] == layer and r["Scope"] == "global"), None)
                    l = next((r for r in raw if r["Dataset"] == ds and r["Side"] == side and r["Layer"] == layer and r["Scope"] == "oracle_local"), None)
                    if g and l and _num(g[col]) is not None and _num(l[col]) is not None: vals.append((layer, float(l[col]) - float(g[col])))
                if vals: ax.plot([x[0] for x in vals], [x[1] for x in vals], marker=marker, label=f"{ds}/{side}")
        ax.axhline(0, color="k", lw=.7); ax.set(xlabel="Layer index (sides separate)", ylabel=ylabel); ax.legend(fontsize=6); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / f"local_minus_global_{col.lower().replace('-', '_')}.png", dpi=160); plt.close(fig)
    for name, x, y, xlabel, ylabel in (("raw_correction_damage_frontier", "Corr", "Outside", "POI corrections", "Outside harm"), ("raw_intervention_efficiency", "Edited", "Corr", "Edited positions/frames", "POI corrections")):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for side, marker in (("encoder", "o"), ("decoder", "s")):
            z = [r for r in raw if r["Side"] == side and _num(r[x]) is not None and _num(r[y]) is not None]
            ax.scatter([r[x] for r in z], [r[y] for r in z], marker=marker, alpha=.65, label=side)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=name.replace("_", " ").title()); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / f"{name}.png", dpi=160); plt.close(fig)
    for key, ylabel, title in (("cos_raw_cond", "Cosine", "Raw–Conditioning cosine"), ("angle_deg", "Degrees", "Raw–Conditioning angle"), ("unit_l2", "Unit L2", "Unit-normalized Raw–Conditioning distance"), ("raw_norm", "Norm", "Raw original vector norm"), ("cond_norm", "Norm", "Conditioning original vector norm")):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot([g["layer"] for g in geom["layers"]], [g[key] for g in geom["layers"]], marker="o"); ax.set(xlabel="Decoder layer", ylabel=ylabel, title=title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir / f"geometry_{key}.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(OUT)); args = ap.parse_args(); root = Path(args.root)
    raw = _condition_rows(root, "raw_r1"); r2 = _condition_rows(root, "raw_r2"); cond = _condition_rows(root, "conditioning")
    fields_a = ["Dataset", "Side", "Layer", "Scope", "rho", "MER", "PIER", "EN-WER", "Matrix-CER", "Corr", "Corrupt", "Outside", "Matrix-Ret"]
    fields_b = fields_a + ["Edited", "Energy", "FractionEdited", "PIER-gain", "MER-gain", "Matrix-CER-gain"]
    _write_csv(root / "tables/raw_full_depth.csv", raw, fields_b)
    _write_csv(root / "tables/raw_selected_dose.csv", r2, fields_b)
    _write_csv(root / "tables/conditioning.csv", cond, ["Dataset", "Layer", "Scope", "rho", "MER", "PIER", "Matrix-CER", "Corr", "Corrupt", "Outside", "Matrix-Ret", "PIER-gain", "MER-gain"])
    delta = []
    for source, rows in (("Raw", raw + r2), ("Conditioning", cond)):
        for ds in sorted({r["Dataset"] for r in rows}):
            sides = sorted({r["Side"] for r in rows if r["Dataset"] == ds})
            for side in sides:
                layers = sorted({r["Layer"] for r in rows if r["Dataset"] == ds and r["Side"] == side})
                for layer in layers:
                    for rho in sorted({float(r["rho"]) for r in rows if r["Dataset"] == ds and r["Side"] == side and r["Layer"] == layer}):
                        g = next((r for r in rows if r["Dataset"] == ds and r["Side"] == side and r["Layer"] == layer and float(r["rho"]) == rho and r["Scope"] == "global"), None)
                        l = next((r for r in rows if r["Dataset"] == ds and r["Side"] == side and r["Layer"] == layer and float(r["rho"]) == rho and r["Scope"] == "oracle_local"), None)
                        if not g or not l: continue
                        def d(k): return None if _num(g[k]) is None or _num(l[k]) is None else float(l[k]) - float(g[k])
                        delta.append({"Dataset": ds, "Direction": source, "Side": side, "Layer": layer, "rho": rho, "Delta-MER": d("MER"), "Delta-PIER": d("PIER"), "Delta-MatrixRet": d("Matrix-Ret"), "Delta-Outside": d("Outside")})
    _write_csv(root / "tables/global_vs_local_delta.csv", delta, ["Dataset", "Direction", "Side", "Layer", "rho", "Delta-MER", "Delta-PIER", "Delta-MatrixRet", "Delta-Outside"])
    audit = {"schema_version": "basis_a3_metric_audit_v1", "poi_corrections": "canonical POI baseline-incorrect -> method-correct transitions", "poi_corruptions": "canonical POI baseline-correct -> method-incorrect transitions", "poi_net_utility": "poi_corrections - poi_corruptions", "candidate_level_utility": "not computed by the new runner; never substituted for POI net utility", "outside_harm": "not computed for SEAME; CS candidate accounting remains a post-hoc CPU step where accepted candidate sets are available", "headline_rule": "do not use ambiguous utility field"}
    (root / "metric_audit/semantics.json").parent.mkdir(parents=True, exist_ok=True); (root / "metric_audit/semantics.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    geom = _read(root / "geometry/raw_cond_decoder_geometry.json")
    links = _geometry_links(root, raw, cond) if raw and cond else []
    if links: _write_geometry_correlations(root, links)
    try: _plot(root, raw, cond, geom)
    except Exception as exc: (root / "figures/plot_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    summary = {"schema_version": "basis_a3_summary_v2", "raw_r1_rows": len(raw), "raw_r2_rows": len(r2), "conditioning_rows": len(cond), "geometry_layers": len(geom["layers"]), "status": "COMPLETE" if raw else "PENDING_GPU"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
