#!/usr/bin/env python3
"""Aggregate frozen-setting SEAME transfer rows after Phase C."""
from __future__ import annotations
import hashlib,json
from collections import defaultdict
from pathlib import Path
from csasr.evaluation import canonical,retention
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/a6_ott_upper_bound"; ROWS=OUT/"transfer/rows"; ELIG=OUT/"eligibility"
def fsha(p): return "sha256:"+hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+".tmp"); t.write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False,default=str)+"\n"); t.replace(p)
def ids(dataset,family,side,model):
 name="SEAME_MAN" if dataset=="seame_dev_man" else "SEAME_SGE"; suffix={("add_unique","encoder"):"ADD_UNIQUE_ENCODER",("add_unique","decoder"):"ADD_UNIQUE_DECODER",("conditioning_cs","decoder"):"CONDITIONING_CS_DECODER"}[(family,side)]
 prefix="WHISPER" if model=="whisper" else "QWEN3_ASR_1P7B"
 p=ELIG/f"{prefix}_{name}_{suffix}.json"
 if not p.is_file(): p=ELIG/f"{name}_{suffix}.json"
 return set(json.loads(p.read_text())["ids"])
def ineligible():
 out={}
 for p in (OUT/"transfer"/"ineligible").glob("*.json"):
  for x in json.loads(p.read_text()).get("rows",[]): out[x["canonical_key"]]=x
 return out
def metric(cell):
 ref=[x["reference"] for x in cell]; base=[x["baseline_hypothesis"] for x in cell]; st=[x["steered_hypothesis"] for x in cell]; bm=canonical.corpus_metrics(ref,base); sm=canonical.corpus_metrics(ref,st); tr=canonical.correction_corruption(ref,base,st); rt=retention.retention_report(ref,base,st)
 return {"n_total":len(cell),"n_eligible":len(cell),"eligibility_rate":1.0,"metrics":sm,"baseline_metrics":bm,"mer_gain":canonical.gain(bm.get("mer"),sm.get("mer")),"pier_gain":canonical.gain(bm.get("pier"),sm.get("pier")),"corrections":tr["corrections"],"corruptions":tr["corruptions"],"correction_minus_corruption":tr["net_corrections"],"matrix_retention":rt["matrix_zh"]["rate"],"embedded_retention":rt["embedded_en"]["rate"],"transcript_change_rate":sum(a!=b for a,b in zip(base,st))/len(st),"hidden_perturbation":sum(float((x.get("perturbation_diagnostics") or {}).get("edit_norm_sum",0.0)) for x in cell)/len(cell),"target_logit_movement":None}
def main():
 fp=json.loads((OUT/"transfer/PHASE_C_FINGERPRINT.json").read_text()); rows=[]
 for p in sorted(ROWS.glob("**/*.json")):
  if "legacy_flat" in p.parts: continue
  x=json.loads(p.read_text())
  if x.get("status")!="PASS" or x.get("phase")!="C" or x.get("regime")!="oracle_tt": raise RuntimeError(f"invalid transfer row {p}")
  rows.append(x)
 grouped=defaultdict(list)
 for x in rows: grouped[(x["model"],x["dataset"],x["method_family"],x["side"],x["layer"],x["rho"],x["decode_mode"])].append(x)
 bad=ineligible()
 cells=[]
 final=json.loads((OUT/"confirm/PHASE_B_FINAL_SETTINGS.json").read_text())["settings"]
 for s in final:
  for ds in ("seame_dev_man","seame_dev_sge"):
    ex=ids(ds,s["family"],s["side"],s["model"])
    for mode in ("greedy","official_standard"):
     cell=grouped[(s["model"],ds,s["family"],s["side"],s["layer"],s["rho"],mode)]
     excluded=set()
     for k in bad:
      parts=k.split("|")
      if len(parts)==10 and (parts[0],parts[1],parts[2],parts[4],parts[5],int(parts[6]),float(parts[7]),parts[8]) == ("C",s["model"],ds,s["family"],s["side"],int(s["layer"]),float(s["rho"]),mode):
       excluded.add(parts[3])
     expected=ex-excluded
     if {x["utterance_id"] for x in cell}!=expected: raise RuntimeError(f"incomplete transfer cell {s} {ds} {mode}: expected {len(expected)} got {len(cell)}")
     cells.append({**s,"dataset":ds,"decode_mode":mode,"eligibility_scope":"family_specific",**metric(cell),"n_panel":len(ex),"n_ineligible":len(excluded),"eligibility_rate":len(expected)/len(ex)})
 # Emit explicit common-eligible rows for cross-family comparisons.  The
 # common set is the intersection of model-specific family populations after
 # subtracting only outcome-blind transfer exclusions.
 for model in ("whisper","qwen3_asr_1p7b"):
  settings=[s for s in final if s["model"]==model]
  for ds in ("seame_dev_man","seame_dev_sge"):
   eligible_by_family={}
   for s in settings:
    key=(s["family"],s["side"])
    if key in eligible_by_family: continue
    base_ids=ids(ds,s["family"],s["side"],model)
    excluded_ids={k.split("|")[3] for k in bad if (lambda p: len(p)==10 and p[0]=="C" and p[1]==model and p[2]==ds and p[4]==s["family"] and p[5]==s["side"])(k.split("|"))}
    eligible_by_family[key]=base_ids-excluded_ids
   common=set.intersection(*eligible_by_family.values())
   for mode in ("greedy","official_standard"):
    for s in settings:
     cell=grouped[(model,ds,s["family"],s["side"],s["layer"],s["rho"],mode)]
     cc=[x for x in cell if x["utterance_id"] in common]
     cells.append({**s,"dataset":ds,"decode_mode":mode,"eligibility_scope":"common",**metric(cc),"n_panel":len(common),"n_ineligible":0,"eligibility_rate":1.0})
 atomic(OUT/"transfer/PHASE_C_TRANSFER.json",{"schema_version":"a6_ott_phase_c_transfer_v1","status":"PASS","phase":"C","manifest_fingerprint":fp["manifest_sha256"],"cells":cells,"seame_outcomes_loaded":True})
 print(json.dumps({"status":"PASS","logical_rows":len(rows),"cells":len(cells)},indent=2))
if __name__=="__main__": main()
