#!/usr/bin/env python3
"""Freeze the amended Phase-B eligibility contract after baseline audit."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/a6_ott_upper_bound'
def fsha(p): return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); t.replace(p)
def main():
 old=json.loads((OUT/'confirm/PHASE_B_MANIFEST.json').read_text()); files=sorted((OUT/'eligibility').glob('WHISPER_*_*.json'))+sorted((OUT/'eligibility').glob('QWEN3_ASR_1P7B_*_*.json'))
 payload={"schema_version":"a6_ott_phase_b_manifest_v4","status":"FROZEN_BEFORE_CORRECTED_CONFIRM_CONTINUATION","phase":"B","regime":"oracle_tt","supersedes":fsha(OUT/'confirm/PHASE_B_MANIFEST.json'),"phase_a_immutable":fsha(OUT/'phase_a/SEARCH_MANIFEST_FINGERPRINT.json'),"settings_whisper":fsha(OUT/'confirm/PHASE_B_SETTINGS_whisper.json'),"settings_qwen":fsha(OUT/'confirm/PHASE_B_SETTINGS_qwen3_asr_1p7b.json'),"model_specific_baseline_eligibility":{p.name:fsha(p) for p in files},"amendment":"decoder causal-position eligibility is evaluated against the accepted free baseline; valid pre-amendment rows are reusable when their canonical keys remain eligible","steering_outcomes_consulted":False,"seame_outcomes_loaded":False}
 payload['manifest_sha256']='sha256:'+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(); atomic(OUT/'confirm/PHASE_B_MANIFEST_V2.json',payload); atomic(OUT/'confirm/PHASE_B_FINGERPRINT_V2.json',{'schema_version':'a6_ott_phase_b_fingerprint_v2','status':'IMMUTABLE','manifest_sha256':payload['manifest_sha256'],'supersedes':old.get('manifest_sha256'),'phase_a_immutable':payload['phase_a_immutable']}); print(json.dumps({'status':payload['status'],'manifest_sha256':payload['manifest_sha256']},indent=2))
if __name__=='__main__': main()
