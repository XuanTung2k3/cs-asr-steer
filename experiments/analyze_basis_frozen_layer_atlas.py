"""CPU aggregation and figures for BASIS-A2.

This module consumes only the frozen atlas artifacts.  It never loads Whisper,
selects data, or changes a scientific condition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "results" / "basis_frozen_layer_atlas"
FIG = ATLAS / "figures"
LAYERS = tuple(range(32))
DIRECTIONS = ("Raw", "Local", "Conditioning", "Raw+Cond", "Local+Cond")
KEYS = {"Raw": "raw", "Local": "local", "Conditioning": "conditioning",
        "Raw+Cond": "raw_cond", "Local+Cond": "local_cond"}
RHO = (0.25, 0.5, 1.0, 2.0)


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row, name, default=None):
    m = row["result_v1"]["metrics"]
    if name in m:
        return m[name]
    if name == "corrections":
        return m.get("transitions", {}).get("corrections", default)
    if name == "corruptions":
        return m.get("transitions", {}).get("corruptions", default)
    if name == "utility":
        return m.get("candidate_utility", m.get("utility", default))
    if name == "pier":
        return m.get("pier", default)
    if name == "pier_gain":
        return m.get("pier_gain", default)
    if name == "mer":
        return m.get("mer", default)
    if name == "matrix_cer":
        return m.get("zh_cer", default)
    if name == "outside_harm":
        return m.get("outside_harm", default)
    if name == "energy":
        return m.get("realized_edit", {}).get("total_energy", default)
    if name == "mean_energy":
        return m.get("realized_edit", {}).get("mean_energy", default)
    if name == "edits":
        return m.get("realized_edit", {}).get("n_steered", default)
    if name == "embed_retention":
        return m.get("retention", {}).get("embedded_en", {}).get("rate", default)
    if name == "matrix_retention":
        return m.get("retention", {}).get("matrix_zh", {}).get("rate", default)
    if name == "en_wer":
        return m.get("en_wer", default)
    return default


def free_rows():
    rows = []
    for p in sorted(ATLAS.joinpath("free_decode").glob("L[0-9][0-9]/*.json")):
        if p.name.startswith("runtime"):
            continue
        x = load_json(p)
        if "result_v1" not in x:
            continue
        d = x["direction"]
        rho = float(x["rho"]); layer = int(x["layer"])
        r = {"source": "NEW", "layer": layer, "direction": d, "rho": rho,
             "direction_hash": x.get("direction_hash"), "scale_s_l": x.get("scale_s_l"),
             "wer": "n/a — no frozen canonical overall WER"}
        for k in ("mer", "pier", "en_wer", "matrix_cer", "corrections", "corruptions",
                  "utility", "outside_harm", "embed_retention", "matrix_retention",
                  "energy", "mean_energy", "edits", "pier_gain"):
            r[k] = metric(x, k)
        e = r["energy"]
        r["utility_per_energy"] = r["utility"] / e if e and e > 0 else None
        r["corrections_per_energy"] = r["corrections"] / e if e and e > 0 else None
        r["corruptions_per_energy"] = r["corruptions"] / e if e and e > 0 else None
        r["corrections_per_1000_edits"] = 1000 * r["corrections"] / r["edits"] if r["edits"] else None
        r["corruptions_per_1000_edits"] = 1000 * r["corruptions"] / r["edits"] if r["edits"] else None
        r["utility_per_1000_edits"] = 1000 * r["utility"] / r["edits"] if r["edits"] else None
        rows.append(r)
    return rows


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def angle(c):
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def projection(a):
    q, _ = np.linalg.qr(np.asarray(a, dtype=float))
    return q[:, :np.linalg.matrix_rank(a, tol=1e-10)] @ q[:, :np.linalg.matrix_rank(a, tol=1e-10)].T


def geometry():
    meta = load_json(ATLAS / "directions.json")
    vectors = {}
    for l in LAYERS:
        vectors[l] = {k: np.load(ATLAS / "directions" / f"{k}_L{l}.npy") for k in KEYS.values()}
    layer_rows = []
    for l in LAYERS:
        v = vectors[l]; r, c, loc = v["raw"], v["conditioning"], v["local"]
        rc, lc = v["raw_cond"], v["local_cond"]
        Graw = np.outer([r, c], [r, c]) if False else np.array([[r @ r, r @ c], [c @ r, c @ c]])
        Gloc = np.array([[loc @ loc, loc @ c], [c @ loc, c @ c]])
        sr = np.linalg.svd(np.column_stack([r, c]), compute_uv=False)
        sl = np.linalg.svd(np.column_stack([loc, c]), compute_uv=False)
        pr = projection(np.column_stack([r, c])); pl = projection(np.column_stack([loc, c]))
        principal = np.linalg.svd(np.linalg.qr(np.column_stack([r,c]))[0].T @
                                  np.linalg.qr(np.column_stack([loc,c]))[0], compute_uv=False)
        alpha = float(r @ c); residual = r - alpha*c
        layer_rows.append({"layer": l, "cos_raw_local": cosine(r,loc), "angle_raw_local": angle(cosine(r,loc)),
            "cos_raw_cond": cosine(r,c), "angle_raw_cond": angle(cosine(r,c)),
            "cos_local_cond": cosine(loc,c), "angle_local_cond": angle(cosine(loc,c)),
            "alpha": alpha, "removed_energy_fraction": alpha*alpha,
            "residual_norm_before_renorm": float(np.linalg.norm(residual)), "raw_local_distance": float(np.linalg.norm(r-loc)),
            "gram_raw": Graw.tolist(), "gram_local": Gloc.tolist(), "singular_values_raw": sr.tolist(),
            "singular_values_local": sl.tolist(), "condition_raw": float(sr[0]/max(sr[-1],1e-300)),
            "condition_local": float(sl[0]/max(sl[-1],1e-300)), "rank_raw": int(np.linalg.matrix_rank(np.column_stack([r,c]))),
            "rank_local": int(np.linalg.matrix_rank(np.column_stack([loc,c]))),
            "principal_angles_deg": [angle(float(x)) for x in principal],
            "projection_frobenius_distance": float(np.linalg.norm(pr-pl)),
            "cos_rc_lc": cosine(rc,lc), "angle_rc_lc": angle(cosine(rc,lc)),
            "cos_rc_raw": cosine(rc,r), "cos_rc_cond": cosine(rc,c),
            "cos_lc_local": cosine(lc,loc), "cos_lc_cond": cosine(lc,c),
            "pre_norm_raw": float(meta["layers"][str(l)]["basis_metrics"]["v_raw_norm"]),
            "pre_norm_conditioning": float(meta["layers"][str(l)]["basis_metrics"]["v_cond_norm"])})
    # The direction builder records hashes and scales; use the actual arrays as
    # the authoritative normalized vectors for all numerical diagnostics.
    drift = []
    for kind in ("raw", "local", "conditioning"):
        for l in LAYERS:
            drift.append({"layer": l, "direction": kind,
                          "cos_prev": None if l == 0 else cosine(vectors[l][kind], vectors[l-1][kind]),
                          "cos_l24": cosine(vectors[l][kind], vectors[24][kind])})
    out = {"schema_version": "basis_a2_geometry_v1", "layers": layer_rows, "cross_layer_drift": drift,
           "direction_hashes": meta.get("layers", {}),
           "subspace_equivalence": {"max_principal_angle_deg": float(max(max(x["principal_angles_deg"]) for x in layer_rows)),
                                     "max_projection_frobenius_distance": float(max(x["projection_frobenius_distance"] for x in layer_rows)),
                                     "statement": "RAW+COND and LOCAL+COND span the same 2D steering subspace within numerical precision; residualization changes coordinate geometry."}}
    write(ATLAS / "geometry" / "layer_geometry.json", out)
    write(ATLAS / "geometry" / "vector_geometry.json", {"layers": layer_rows, "schema_version": "basis_a2_vector_geometry_v1"})
    write(ATLAS / "geometry" / "subspace_geometry.json", out["subspace_equivalence"])
    return out


def tf_rows():
    out = []
    for p in sorted(ATLAS.joinpath("teacher_forced").glob("L[0-9][0-9]/*.json")):
        x = load_json(p); g = x.get("groups", {}).get("all", {})
        if g: out.append({"layer": int(x["layer"]), "direction": x["direction"], "rho": float(x["rho"]), **g})
    return out


def mean_at(rows, layer, direction, rho, field, default=np.nan):
    z = [r.get(field) for r in rows if r["layer"] == layer and r["direction"] == direction and abs(r["rho"]-rho)<1e-9]
    z = [x for x in z if x is not None]
    return float(np.mean(z)) if z else default


def probe_correlations(rows):
    ppath = ATLAS / "probe.json"
    if not ppath.exists(): return {"status": "deferred"}
    probe = load_json(ppath)["layers"]
    def corr(a,b,method):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3: return None
        if method == "pearson": return float(np.corrcoef(a[m], b[m])[0,1])
        ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
        return float(np.corrcoef(ra,rb)[0,1])
    result = {"n_layers": 32, "comparisons": {}}
    au = np.array([probe[str(l)]["auroc"] for l in LAYERS])
    for d in DIRECTIONS:
        util=np.array([mean_at(rows,l,d,.5,"utility") for l in LAYERS])
        pier=np.array([mean_at(rows,l,d,.5,"pier_gain") for l in LAYERS])
        harm=np.array([mean_at(rows,l,d,.5,"outside_harm") for l in LAYERS])
        result["comparisons"][d] = {"auroc_vs_utility": {m:corr(au,util,m) for m in ("pearson","spearman")},
            "auroc_vs_pier_gain": {m:corr(au,pier,m) for m in ("pearson","spearman")},
            "auroc_vs_outside_harm": {m:corr(au,harm,m) for m in ("pearson","spearman")}}
    write(ATLAS / "probe_correlations.json", result)
    return result


def dose_class(values):
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all(): return "UNAVAILABLE"
    dy = np.diff(y)
    if np.all(dy >= -1e-9):
        d2 = np.diff(dy)
        return "SATURATING" if d2[-1] < -1e-9 else "STABLE DOSE RESPONSE"
    if np.any(dy > 0) and np.any(dy < 0): return "NON-MONOTONIC"
    return "DAMAGE GROWS FASTER THAN CORRECTION"


def figures(rows, geom, tf):
    import matplotlib.pyplot as plt
    import seaborn as sns
    FIG.mkdir(parents=True, exist_ok=True)
    def heat(name, vals, title, cmap="viridis", center=None):
        fig, ax = plt.subplots(figsize=(8.5, 7)); sns.heatmap(vals, ax=ax, xticklabels=DIRECTIONS, yticklabels=LAYERS,
            cmap=cmap, center=center, annot=False); ax.set(xlabel="Direction", ylabel="Decoder layer", title=title)
        fig.tight_layout(); fig.savefig(FIG/name, dpi=180); plt.close(fig)
    def arr(field, rho=.5): return np.array([[mean_at(rows,l,d,rho,field) for d in DIRECTIONS] for l in LAYERS])
    heat("utility_heatmap_rho0.5.png", arr("utility"), "Free-decoding utility, rho=0.5", "RdYlGn", 0)
    heat("pier_change_heatmap_rho0.5.png", arr("pier_gain"), "Free-decoding PIER change, rho=0.5", "RdYlGn", 0)
    heat("matrix_retention_heatmap_rho0.5.png", arr("matrix_retention"), "Matrix retention, rho=0.5", "viridis")
    heat("entropy_change_heatmap_rho0.5.png", np.array([[mean_at(tf,l,d,.5,"delta_entropy") for d in DIRECTIONS] for l in LAYERS]), "Teacher-forced entropy change, rho=0.5", "RdBu_r", 0)
    heat("gold_logprob_change_heatmap_rho0.5.png", np.array([[mean_at(tf,l,d,.5,"delta_gold_logprob") for d in DIRECTIONS] for l in LAYERS]), "Teacher-forced gold log-probability change, rho=0.5", "RdYlGn", 0)
    g = geom["layers"]
    for fn, key, title, ylabel in (("raw_cond_cosine_vs_layer.png","cos_raw_cond","Raw–Conditioning cosine","cosine"),
        ("raw_local_angle_vs_layer.png","angle_raw_local","Raw–Local angle","degrees"),
        ("residualized_energy_fraction_vs_layer.png","removed_energy_fraction","Removed Raw energy fraction","fraction"),
        ("basis_condition_number_vs_layer.png","condition_local","Local basis condition number","condition number")):
        fig, ax=plt.subplots(figsize=(8,4)); ax.plot(LAYERS,[x[key] for x in g],marker="o"); ax.set(xlabel="Decoder layer",ylabel=ylabel,title=title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/fn,dpi=180); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for d in DIRECTIONS:
        y=[np.nanmean([r["utility"] for r in rows if r["layer"]==l and r["direction"]==d and r["rho"]==rho]) for rho in RHO]
        axes[0].plot(RHO,y,marker="o",label=d)
        y2=[np.nanmean([r["corruptions"] for r in rows if r["layer"]==l and r["direction"]==d and r["rho"]==rho]) for rho in RHO]
        axes[1].plot(RHO,y2,marker="o",label=d)
    axes[0].set_title("Utility dose response (mean over layers)"); axes[1].set_title("Corruptions dose response (mean over layers)")
    for ax in axes: ax.set_xlabel("rho"); ax.grid(alpha=.25)
    axes[0].set_ylabel("count"); axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"dose_response_across_layer.png",dpi=180); plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,4));
    for d in DIRECTIONS:
        ax.plot(RHO,[np.nanmean([r["utility"] for r in rows if r["direction"]==d and r["rho"]==rho]) for rho in RHO],marker="o",label=d)
    ax.set(xlabel="rho",ylabel="utility",title="Dose-response utility"); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"dose_response_utility.png",dpi=180); plt.close(fig)
    lam_files=list(ATLAS.joinpath("mixture").glob("L[0-9][0-9]/*.json"))
    if lam_files:
        lambdas=sorted({float(load_json(p)["rho"]) for p in []}) if False else sorted({float(p.stem.split("lambda")[1]) for p in lam_files})
        fig, axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
        for ax, primary in zip(axes,("raw","local")):
            vals=np.full((32,len(lambdas)),np.nan)
            for p in lam_files:
                if not p.stem.startswith(primary+"_lambda"): continue
                x=load_json(p); lam=float(x["direction"].split("lambda_cond")[1]); vals[int(x["layer"]),lambdas.index(lam)]=x["groups"]["all"]["delta_gold_nll"]
            sns.heatmap(vals,ax=ax,xticklabels=lambdas,yticklabels=LAYERS,cmap="RdBu_r",center=0); ax.set_title(primary.title()+"+lambda Cond: Δ gold NLL"); ax.set_xlabel("lambda")
        axes[0].set_ylabel("Decoder layer"); fig.tight_layout(); fig.savefig(FIG/"mixture_lambda_layer_response.png",dpi=180); plt.close(fig)
    if (ATLAS/"probe.json").exists():
        p=load_json(ATLAS/"probe.json")["layers"]; fig,ax=plt.subplots(figsize=(8,4)); ax.plot(LAYERS,[p[str(l)]["auroc"] for l in LAYERS],marker="o"); ax.set(xlabel="Decoder layer",ylabel="AUROC",title="Language probe AUROC by layer"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"probe_auroc_vs_layer.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,5)); au=np.array([p[str(l)]["auroc"] for l in LAYERS]); u=np.array([mean_at(rows,l,"Raw",.5,"utility") for l in LAYERS]); ax.scatter(au,u); ax.set(xlabel="Probe AUROC",ylabel="Raw utility",title="Probe AUROC vs causal steering utility"); fig.tight_layout(); fig.savefig(FIG/"probe_auroc_vs_steering_utility.png",dpi=180); plt.close(fig)


def qualitative(rows):
    panel=load_json(ATLAS/"panel.json")["rows"]
    by={}
    for p in sorted(ATLAS.joinpath("free_decode").glob("L[0-9][0-9]/*.json")):
        if p.name.startswith("runtime"): continue
        x=load_json(p)
        for u in x.get("per_utterance",[]): by.setdefault(u["utterance_id"],[]).append((x,u))
    lines=["# BASIS-A2 qualitative transcript summaries", "", "Descriptive panel traces; no layer is selected from these 10 utterances.", ""]
    for q in panel:
        uid=q["utterance_id"]; lines += [f"## {uid} ({q['category']}, {q['dialogue_id']})", f"Reference: {q['reference']}", f"Baseline: {q['baseline_transcript']}", "", "| Layer | Direction | rho | POI correct / total | Unit corrections | Unit corruptions | Transcript |", "|---:|---|---:|---:|---:|---:|---|"]
        for x,u in sorted(by.get(uid,[]), key=lambda z:(z[0]["layer"],z[0]["direction"],z[0]["rho"])):
            st=u.get("unit_status",{}); vals=list(st.values()); poi=[v for k,v in st.items() if q.get("poi_unit_index") is None or True]
            corr=sum(bool(v[0]) and v[1]=="correct" for v in vals); bad=sum(v[1] in ("corruption","wrong_language_substitution") for v in vals)
            lines.append(f"| {x['layer']} | {x['direction']} | {x['rho']} | {corr}/{len(vals)} | {corr} | {bad} | {u['steered']} |")
        lines.append("")
    (ATLAS/"qualitative_traces.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    rows=free_rows(); tf=tf_rows(); geom=geometry()
    write(ATLAS/"free_decode_summary.json", {"schema_version":"basis_a2_free_summary_v1","rows":rows})
    write(ATLAS/"teacher_forced_summary.json", {"schema_version":"basis_a2_tf_summary_v1","rows":tf})
    dose={}
    for d in DIRECTIONS:
        y=[float(np.nanmean([r["utility"] for r in rows if r["direction"]==d and r["rho"]==rho])) for rho in RHO]
        dose[d]={"rho":list(RHO),"mean_utility":y,"classification":dose_class(y)}
    write(ATLAS/"dose_response.json",dose)
    corr=probe_correlations(rows); figures(rows,geom,tf); qualitative(rows)
    # Compact manuscript-facing summary, preserving the full row artifacts.
    write(ATLAS/"summary.json", {"schema_version":"basis_a2_summary_v1","n_free_rows":len(rows),"n_teacher_forced_rows":len(tf),"geometry":geom["subspace_equivalence"],"dose_response":dose,"probe_correlations":corr,"pca":{"status":"deferred","reason":"no larger compatible cached representation sample was available; no extra GPU sweep launched"},"interpretation":"exploratory all-layer atlas; candidate bands are not final layer validation"})
    (ATLAS/"comparison.md").write_text("# BASIS-A2 comparison\n\nSee `summary.json`, `free_decode_summary.json`, `teacher_forced_summary.json`, and `geometry/`. All free-decoding results are panel-only exploratory evidence; WER is n/a where no frozen canonical overall WER exists.\n",encoding="utf-8")
    print(f"aggregated {len(rows)} free rows and {len(tf)} teacher-forced rows")


if __name__ == "__main__": main()
