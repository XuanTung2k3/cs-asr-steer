#!/usr/bin/env python3
"""Validate/aggregate Phase B and freeze one setting per family/model."""
from __future__ import annotations
import hashlib, json, math, subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from csasr.evaluation import canonical, retention

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/a6_ott_upper_bound"
ROWS = OUT / "confirm" / "rows"
ELIG = OUT / "eligibility"
FAMILIES = (("add_unique", "encoder"), ("add_unique", "decoder"), ("conditioning_cs", "decoder"))

def sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()

def fsha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    tmp.replace(path)

def eligible_ids(dataset: str, family: str, side: str, model: str) -> set[str]:
    name = {"cs_dialogue_confirm": "CS_CONFIRM", "ascend_confirm": "ASCEND_CONFIRM"}[dataset]
    suffix = {("add_unique", "encoder"): "ADD_UNIQUE_ENCODER", ("add_unique", "decoder"): "ADD_UNIQUE_DECODER", ("conditioning_cs", "decoder"): "CONDITIONING_CS_DECODER"}[(family, side)]
    prefix = "WHISPER" if model == "whisper" else "QWEN3_ASR_1P7B"
    path = ELIG / f"{prefix}_{name}_{suffix}.json"
    if not path.is_file(): path = ELIG / f"{name}_{suffix}.json"
    return set(json.loads(path.read_text())["ids"])

def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [str(x["reference"]) for x in rows]
    base = [str(x["baseline_hypothesis"]) for x in rows]
    steer = [str(x["steered_hypothesis"]) for x in rows]
    if not rows:
        return {"n_total": 0, "n_eligible": 0, "eligibility_rate": 0.0}
    bm = canonical.corpus_metrics(refs, base); sm = canonical.corpus_metrics(refs, steer)
    tr = canonical.correction_corruption(refs, base, steer); ret = retention.retention_report(refs, base, steer)
    energies = [float((x.get("perturbation_diagnostics") or {}).get("edit_norm_sum", 0.0)) for x in rows]
    return {"n_total": len(rows), "n_eligible": len(rows), "eligibility_rate": 1.0,
            "metrics": sm, "baseline_metrics": bm,
            "mer_gain": canonical.gain(bm.get("mer"), sm.get("mer")),
            "pier_gain": canonical.gain(bm.get("pier"), sm.get("pier")),
            "en_wer_gain": canonical.gain(bm.get("en_wer"), sm.get("en_wer")),
            "matrix_cer_gain": canonical.gain(bm.get("zh_cer"), sm.get("zh_cer")),
            "corrections": int(tr["corrections"]), "corruptions": int(tr["corruptions"]),
            "correction_minus_corruption": int(tr["net_corrections"]),
            "outside_harm": None, "retention": ret,
            "matrix_retention": ret["matrix_zh"]["rate"], "embedded_retention": ret["embedded_en"]["rate"],
            "transcript_change_rate": sum(a != b for a,b in zip(base, steer)) / len(steer),
            "mean_edit_norm": sum(energies) / len(energies),
            "hidden_perturbation": sum(energies) / len(energies),
            "target_logit_movement": None, "transcript_flips": sum(a != b for a,b in zip(base, steer))}

def load_rows() -> list[dict[str, Any]]:
    rows=[]
    for path in sorted(ROWS.glob("**/*.json")):
        if "legacy_flat" in path.parts:
            continue
        x=json.loads(path.read_text()); key=str(x.get("canonical_key", ""))
        if x.get("phase") != "B" or x.get("regime") != "oracle_tt" or x.get("status") != "PASS":
            raise RuntimeError(f"invalid confirmation row {path}")
        if x.get("decode_mode") not in {"greedy", "official_standard"}:
            raise RuntimeError(f"invalid decode mode {path}")
        parts=key.split("|")
        if len(parts)!=10 or parts[0]!="B" or parts[1]!=x.get("model") or parts[2]!=x.get("dataset") or parts[3]!=x.get("utterance_id"):
            raise RuntimeError(f"canonical key mismatch {path}")
        if x.get("provenance",{}).get("fixed_direction_artifact") is not None:
            raise RuntimeError(f"fixed direction in confirmation {path}")
        rows.append(x)
    return rows

def load_ineligible() -> dict[str, dict[str, Any]]:
    out = {}
    root = OUT / "confirm" / "ineligible"
    for path in root.glob("*.json"):
        payload = json.loads(path.read_text())
        for item in payload.get("rows", []):
            out[str(item["canonical_key"])] = item
    return out

def rank_key(item: dict[str, Any]):
    def val(k, default=-math.inf):
        x=item.get(k); return default if x is None else float(x)
    return (val("correction_minus_corruption"), val("pier_gain"), val("matrix_retention"), -val("mean_edit_norm", math.inf))

def main() -> None:
    fp_path = OUT/"confirm"/"PHASE_B_FINGERPRINT_V2.json"
    if not fp_path.is_file(): fp_path = OUT/"confirm"/"PHASE_B_FINGERPRINT.json"
    fp=json.loads(fp_path.read_text())
    rows=load_rows(); ineligible=load_ineligible(); grouped=defaultdict(list)
    for x in rows:
        grouped[(x["model"],x["dataset"],x["method_family"],x["side"],int(x["layer"]),float(x["rho"]),x["decode_mode"])].append(x)
    settings_by_model={m:json.loads((OUT/"confirm"/f"PHASE_B_SETTINGS_{m}.json").read_text())["settings"] for m in ("whisper","qwen3_asr_1p7b")}
    expected=[]; table=[]
    for model,settings in settings_by_model.items():
        for s in settings:
            family,side=s["family"],s["side"]
            for dataset in ("cs_dialogue_confirm","ascend_confirm"):
                ids=eligible_ids(dataset,family,side,model)
                for mode in ("greedy","official_standard"):
                    key=(model,dataset,family,side,int(s["layer"]),float(s["rho"]),mode); cell=grouped.get(key,[])
                    excluded=set()
                    for item in ineligible.values():
                        parts=str(item["canonical_key"]).split("|")
                        if len(parts)==10 and (parts[0],parts[1],parts[2],parts[4],parts[5],int(parts[6]),float(parts[7]),parts[8]) == ("B",model,dataset,family,side,int(s["layer"]),float(s["rho"]),mode):
                            excluded.add(parts[3])
                    expected_ids=ids-excluded; got={str(x["utterance_id"]) for x in cell}
                    if got != expected_ids:
                        raise RuntimeError(f"incomplete Phase-B cell {key}: expected {len(expected_ids)} plus {len(excluded)} INELIGIBLE, got {len(got)}")
                    mm=metric(cell)
                    table.append({"model":model,"dataset":dataset,"family":family,"side":side,"layer":int(s["layer"]),"rho":float(s["rho"]),"decode_mode":mode,"eligibility_scope":"family_specific","n_panel":len(ids),"n_ineligible":len(excluded),"eligibility_rate":len(got)/len(ids),**{k:v for k,v in mm.items() if k!="eligibility_rate"}})
                    common = (eligible_ids(dataset, "add_unique", "encoder", model)
                              & eligible_ids(dataset, "add_unique", "decoder", model)
                              & eligible_ids(dataset, "conditioning_cs", "decoder", model))
                    common_cell=[x for x in cell if str(x["utterance_id"]) in common]
                    table.append({"model":model,"dataset":dataset,"family":family,"side":side,"layer":int(s["layer"]),"rho":float(s["rho"]),"decode_mode":mode,"eligibility_scope":"common","n_panel":len(common),**metric(common_cell)})
    finals=[]; selection={}
    for model,settings in settings_by_model.items():
        selection[model]={}
        for family,side in FAMILIES:
            candidates=[]
            for s in settings:
                if (s["family"],s["side"]) != (family,side): continue
                pooled=[]; per_mode={}
                for mode in ("greedy","official_standard"):
                    cell=[x for x in rows if x["model"]==model and x["method_family"]==family and x["side"]==side and int(x["layer"])==int(s["layer"]) and float(x["rho"])==float(s["rho"]) and x["decode_mode"]==mode]
                    pooled.extend(cell); per_mode[mode]=metric(cell)
                candidates.append({**s,"family":family,"side":side,"pooled_greedy":per_mode["greedy"],"pooled_official_standard":per_mode["official_standard"]})
            if not candidates: raise RuntimeError(f"no candidates for {model}/{family}/{side}")
            greedy=sorted(candidates,key=lambda x:rank_key({"correction_minus_corruption":x["pooled_greedy"]["correction_minus_corruption"],"pier_gain":x["pooled_greedy"]["pier_gain"],"matrix_retention":x["pooled_greedy"]["matrix_retention"],"mean_edit_norm":x["pooled_greedy"]["mean_edit_norm"]}),reverse=True)
            official=sorted(candidates,key=lambda x:rank_key({"correction_minus_corruption":x["pooled_official_standard"]["correction_minus_corruption"],"pier_gain":x["pooled_official_standard"]["pier_gain"],"matrix_retention":x["pooled_official_standard"]["matrix_retention"],"mean_edit_norm":x["pooled_official_standard"]["mean_edit_norm"]}),reverse=True)
            chosen=greedy[0]
            chosen={"family":family,"side":side,"layer":int(chosen["layer"]),"rho":float(chosen["rho"]),"selection_source":"Phase-B confirm only","greedy_rank":1,"official_rank":next(i+1 for i,x in enumerate(official) if x["layer"]==chosen["layer"] and x["rho"]==chosen["rho"]),"candidates":candidates}
            selection[model][f"{family}/{side}"]=chosen; finals.append({"model":model,"family":family,"side":side,"layer":chosen["layer"],"rho":chosen["rho"]})
    payload={"schema_version":"a6_ott_phase_b_confirm_v1","status":"PASS","phase":"B","manifest_fingerprint":fp["manifest_sha256"],"logical_rows":len(rows),"cells":table,"selection_rule":"frozen Phase-B greedy primary: correction-minus-corruption, PIER gain, retention, lower energy; official rank recorded","selection":selection,"phase_c_outcomes_loaded":False}
    atomic(OUT/"confirm"/"PHASE_B_CONFIRM.json",payload)
    for model in settings_by_model:
        atomic(OUT/"confirm"/f"PHASE_B_FINAL_SETTINGS_{model}.json",{"schema_version":"a6_ott_phase_b_final_settings_v1","status":"FROZEN_BEFORE_PHASE_C","phase":"B","model":model,"settings":[{k:v for k,v in x.items() if k in {"family","side","layer","rho"}} for x in finals if x["model"]==model],"source_confirm_sha256":fsha(OUT/"confirm"/"PHASE_B_CONFIRM.json"),"seame_outcomes_loaded":False})
    atomic(OUT/"confirm"/"PHASE_B_FINAL_SETTINGS.json",{"schema_version":"a6_ott_phase_b_final_settings_all_v1","status":"FROZEN_BEFORE_PHASE_C","settings":finals,"source_confirm_sha256":fsha(OUT/"confirm"/"PHASE_B_CONFIRM.json"),"seame_outcomes_loaded":False})
    atomic(OUT/"diagnostics"/"QWEN_PHASE_B_DIAGNOSIS.json",{"schema_version":"a6_ott_qwen_phase_b_diagnosis_v1","status":"PASS","model":"qwen3_asr_1p7b","rows":[x for x in table if x["model"]=="qwen3_asr_1p7b"],"interpretation":"target-logit movement is unavailable in the accepted row schema; hidden perturbation and transcript flips are reported without causal overclaim","seame_outcomes_loaded":False})
    print(json.dumps({"status":"PASS","logical_rows":len(rows),"finals":finals},indent=2))

if __name__ == "__main__": main()
