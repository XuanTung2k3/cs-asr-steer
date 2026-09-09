#!/usr/bin/env python
"""CPU-only BASIS-A3 aggregation, geometry plots, and compact tables."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/basis_a3_raw_cond_scope_depth"

def rows(root=OUT):
    out=[]
    for p in root.glob("raw_r1/*/L*/*.json"):
        x=json.loads(p.read_text()); m=x.get("metrics",{})
        t=m.get("transitions",{})
        out.append({"dataset":x["dataset"],"side":x["side"],"layer":x["layer"],"scope":x["scope"],"rho":x["rho"],
                    "MER":m.get("mer"),"PIER":m.get("pier"),"EN-WER":m.get("en_wer"),"Matrix-CER":m.get("zh_cer"),
                    "Corr":m.get("poi_corrections",t.get("corrections")),"Corrupt":m.get("poi_corruptions",t.get("corruptions")),
                    "Outside":m.get("outside_harm"),"Matrix-Ret":m.get("retention",{}).get("matrix_zh",{}).get("rate"),
                    "Edited":m.get("edited_positions_or_frames"),"Energy":m.get("total_intervention_energy")})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=str(OUT)); args=ap.parse_args(); root=Path(args.root)
    geom=json.loads((root/"geometry/raw_cond_decoder_geometry.json").read_text())
    (root/"tables").mkdir(parents=True,exist_ok=True)
    rr=rows(root)
    fields=["dataset","side","layer","scope","rho","MER","PIER","EN-WER","Matrix-CER","Corr","Corrupt","Outside","Matrix-Ret","Edited","Energy"]
    with (root/"tables/raw_full_depth.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rr)
    (root/"figures").mkdir(parents=True,exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        for metric, name in (("PIER","pier_vs_depth"),("MER","mer_vs_depth"),("Matrix-CER","matrix_cer_vs_depth"),("Matrix-Ret","matrix_retention_vs_depth"),("Corr","corrections_vs_depth"),("Outside","outside_harm_vs_depth")):
            fig,ax=plt.subplots(figsize=(9,4.5))
            for ds in sorted({r["dataset"] for r in rr}):
                for side,marker in (("encoder","o"),("decoder","s")):
                    for scope,ls in (("global","-"),("oracle_local","--")):
                        z=[r for r in rr if r["dataset"]==ds and r["side"]==side and r["scope"]==scope and r[metric] is not None]
                        if z: ax.plot([r["layer"] for r in z],[r[metric] for r in z],marker=marker,linestyle=ls,label=f"{ds}/{side}/{scope}",alpha=.75)
            ax.set(xlabel="Layer index (encoder and decoder shown separately)",ylabel=metric,title=f"Raw {metric} vs depth"); ax.legend(fontsize=6,ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(root/"figures"/f"{name}.png",dpi=160); plt.close(fig)
        for key,ylabel,title in (("cos_raw_cond","cosine","Raw–Conditioning cosine"),("angle_deg","degrees","Raw–Conditioning angle"),("unit_l2","L2","Unit-normalized Raw–Conditioning L2"),("raw_norm","norm","Original direction norms"),("cond_norm","norm","Original Conditioning norms")):
            fig,ax=plt.subplots(figsize=(7,4)); ax.plot([x["layer"] for x in geom["layers"]],[x[key] for x in geom["layers"]],marker="o"); ax.set(xlabel="Decoder layer",ylabel=ylabel,title=title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(root/"figures"/f"geometry_{key}.png",dpi=160); plt.close(fig)
    except Exception as exc:
        (root/"figures/plot_error.txt").write_text(str(exc))
    (root/"summary.json").write_text(json.dumps({"schema_version":"basis_a3_summary_v1","raw_r1_rows":len(rr),"geometry_layers":len(geom["layers"]),"status":"COMPLETE" if rr else "PENDING_GPU"},indent=2)+"\n")

if __name__=="__main__": main()
