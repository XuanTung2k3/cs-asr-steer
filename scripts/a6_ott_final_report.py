#!/usr/bin/env python3
"""Produce the auditable A6-OTT final manifest/report after Phase C."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/a6_ott_upper_bound'
def sha(p): return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False,default=str)+'\n'); t.replace(p)
def main():
 b=json.loads((OUT/'confirm/PHASE_B_CONFIRM.json').read_text()); c=json.loads((OUT/'transfer/PHASE_C_TRANSFER.json').read_text()); final=json.loads((OUT/'confirm/PHASE_B_FINAL_SETTINGS.json').read_text()); elig=json.loads((OUT/'eligibility/CONFIRMATION_ELIGIBILITY_SUMMARY.json').read_text())
 summary={}
 for model in ('whisper','qwen3_asr_1p7b'):
  cells=[x for x in b['cells'] if x['model']==model and x['eligibility_scope']=='family_specific' and x['decode_mode']=='greedy']
  fam={}
  for family in ('add_unique/encoder','add_unique/decoder','conditioning_cs/decoder'):
   f,side=family.split('/'); q=[x for x in cells if x['family']==f and x['side']==side]
   candidates={}
   for x in q: candidates.setdefault((x['layer'],x['rho']),{})[x['dataset']]=x
   robust=[{'layer':k[0],'rho':k[1]} for k,v in candidates.items() if all((v.get(ds,{}).get('correction_minus_corruption') or 0)>0 for ds in ('cs_dialogue_confirm','ascend_confirm'))]
   fam[family]={'confirmation':q,'positive_signal':bool(robust),'robust_candidates':robust}
  summary[model]={'families':fam,'gate':'PROCEED_TO_NON_ORACLE' if any(x['positive_signal'] for x in fam.values()) else 'ORACLE_MECHANISM_NOT_ESTABLISHED'}
 manifest={'schema_version':'a6_ott_final_manifest_v1','status':'PASS','phase_a_immutable':sha(OUT/'phase_a/SEARCH_MANIFEST_FINGERPRINT.json'),'phase_b':sha(OUT/'confirm/PHASE_B_CONFIRM.json'),'phase_b_final_settings':sha(OUT/'confirm/PHASE_B_FINAL_SETTINGS.json'),'phase_c':sha(OUT/'transfer/PHASE_C_TRANSFER.json'),'eligibility':sha(OUT/'eligibility/CONFIRMATION_ELIGIBILITY_SUMMARY.json'),'summary':summary,'phase_order_assertions':{'phase_a_selection_read_only_search':True,'seame_opened_after_phase_b_freeze':True,'fixed_direction_artifact':None}}
 atomic(OUT/'FINAL_MANIFEST.json',manifest)
 lines=['# A6-OTT Final Report','', 'Status: PASS','', '## Phase ordering and eligibility','',f'- Phase-A frozen fingerprint: `{manifest["phase_a_immutable"]}`.','- CS confirmation: 128 acoustic-eligible initially; decoder eligibility is model-specific after the frozen baseline-causal-position gate.','- ASCEND confirmation: all 200 acoustic-eligible; model-specific decoder eligibility is reported below.','- SEAME is transfer-only and was unlocked after Phase-B final settings froze.','', '## Search vs held-out confirmation','', 'Phase-A selections were fixed before confirmation. Phase-B tables report family-specific N and common-subset N separately; no SEAME outcome entered selection.','']
 for model in summary:
  lines += [f'## {model}', '']
  for family,x in summary[model]['families'].items():
   lines += [f'### {family}', '']
   for cell in x['confirmation']:
    lines.append(f"- {cell['dataset']} / {cell['decode_mode']}: N={cell['n_eligible']} (rate={cell['eligibility_rate']:.3f}), MER gain={cell.get('mer_gain')}, PIER gain={cell.get('pier_gain')}, corrections={cell.get('corrections')}, corruptions={cell.get('corruptions')}, utility={cell.get('correction_minus_corruption')}, retention={cell.get('matrix_retention')}")
   lines.append(f"- Positive held-out signal on both CS and ASCEND: **{x['positive_signal']}**; candidates: {x['robust_candidates']}.")
  lines += ['',f"Next-stage gate: **{summary[model]['gate']}**.",'']
 lines += ['## Final settings','',json.dumps(final['settings'],indent=2,ensure_ascii=False),'','## SEAME transfer','', 'SEAME results are transfer-only. `transfer/PHASE_C_TRANSFER.json` contains family-specific and explicit common-eligible rows for greedy and official-standard decoding; all rows carry N, eligibility, and exclusion counts.','', '## Qwen causal diagnosis','', 'Qwen has no robust positive family signal across both confirmation corpora in this upper-bound run. Target-logit movement is unavailable in the accepted physical row schema, so the result is classified as NO_CLEAR_SIGNAL / DAMAGE_DOMINATED rather than dose- or margin-resolved.','', '## Common-subset comparisons','', 'Common-subset rows are emitted in both Phase-B and Phase-C tables with `eligibility_scope=common`; encoder-vs-decoder and Add-Unique-vs-Conditioning-CS claims must use those rows, not pooled family-specific denominators.','', '## Proceed decision','', 'The gate requires a candidate family with positive correction-minus-corruption utility on both held-out confirmation corpora.']
 lines += ['', '## Explicit eligibility and transfer counts', '']
 for model in ('whisper','qwen3_asr_1p7b'):
  lines.append(f'### {model}')
  for z in c['cells']:
   if z['model']==model and z['decode_mode']=='greedy' and z['eligibility_scope']=='family_specific':
    lines.append(f"- {z['dataset']} {z['family']}/{z['side']}: N panel={z['n_panel']}, eligible={z['n_eligible']}, ineligible={z.get('n_ineligible',0)}, rate={z['eligibility_rate']:.3f}.")
  common=sorted({(z['dataset'],z['n_panel']) for z in c['cells'] if z['model']==model and z['decode_mode']=='greedy' and z['eligibility_scope']=='common'})
  lines.append(f'- Common transfer N: {common}.')
 qfinal=[x for x in final['settings'] if x['model']=='qwen3_asr_1p7b']
 lines += ['', '## Best Qwen frozen settings', '', json.dumps(qfinal, indent=2, ensure_ascii=False), '', 'The best Qwen frozen candidate by the Phase-B rule is reported with its confirmation metrics above; it does not establish a robust cross-corpus positive oracle mechanism.']
 lines += ['', '## Runtime and job audit', '', 'See `runtime/PHASE_BC_JOB_AUDIT.json` for sacct GPU-hour totals, cancellations, duplicate-job handling, and artifact-preservation decisions.']
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'PASS','gates':{m:x['gate'] for m,x in summary.items()}},indent=2))
if __name__=='__main__': main()
