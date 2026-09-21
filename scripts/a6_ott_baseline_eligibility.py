#!/usr/bin/env python3
"""Apply the frozen baseline-hypothesis causal-position eligibility gate."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO),str(REPO/'src')]
OUT=REPO/'results/a6_ott_upper_bound'; ELIG=OUT/'eligibility'
def sha(p): return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objsha(x): return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); t.replace(p)
def main():
 from transformers import WhisperProcessor,AutoTokenizer
 from csasr.basis_a6.panels import load_panel
 from csasr.data.normalize import normalize_and_segment
 from csasr.data.alignment import token_char_offsets
 from scripts.basis_a6_real_acceptance import _reference_groups,_hypothesis_positions
 wp=WhisperProcessor.from_pretrained('/mnt/data/tungnx/whisper-large-v3',local_files_only=True); tok={'whisper':wp.tokenizer,'qwen3_asr_1p7b':AutoTokenizer.from_pretrained('/mnt/data/tungnx/Qwen3-ASR-1.7B',local_files_only=True)}
 prefix=len(wp.tokenizer.convert_tokens_to_ids(['<|startoftranscript|>','<|zh|>','<|transcribe|>','<|notimestamps|>']))
 specs={'CS_CONFIRM':('cs_dialogue_confirm','cs_dialogue_dev_select','cs_dialogue_dev_select'),'ASCEND_CONFIRM':('ascend_confirm','ascend_eval','ascend_eval'),'SEAME_MAN':('seame_dev_man','seame_dev_man','seame_dev_man'),'SEAME_SGE':('seame_dev_sge','seame_dev_sge','seame_dev_sge')}
 for name,(dataset,panel,base_ds) in specs.items():
  audit=json.loads((ELIG/f'{name}_METHODS.json').read_text()); rows,_=load_panel(panel,require_alignment=True); by={str(x['utterance_id']):x for x in rows}; model_maps={}
  for model,t in tok.items():
   modes = ['greedy', 'official_standard'] if model == 'whisper' else ['greedy']
   baselines_by_mode = {mode: json.loads((REPO/f'results/basis_a6_expanded/baselines/{model}/{base_ds}/{mode}.json').read_text())['rows'] for mode in modes}
   valid=[]; bad=[]; details={}
   for uid,entry in audit['entries'].items():
    row=dict(by[uid]); row['oracle_transcript_spans']=entry.get('oracle_transcript_spans',[]); norm,units=normalize_and_segment(row['reference']); ref_ids=list(t.encode(norm,add_special_tokens=False)[:220]); ref_idx=_reference_groups(row,t,ref_ids,norm,units); mode_details={}; ok=True
    for mode in modes:
     content_len=len(baselines_by_mode[mode][uid]['token_ids'])-(prefix if model=='whisper' else 0); hyp_idx=_hypothesis_positions(ref_idx,len(ref_ids),content_len); mode_details[mode]={'hypothesis_token_indices':hyp_idx,'valid':bool(hyp_idx and any(i>0 for i in hyp_idx))}; ok = ok and mode_details[mode]['valid']
    details[uid]={'reference_token_indices':ref_idx,'modes':mode_details,'status':'FULLY_ELIGIBLE' if ok else 'DECODER_CAUSAL_POSITION_INELIGIBLE'}; (valid if ok else bad).append(uid)
   model_maps[model]={'eligible_ids':sorted(valid),'ineligible_ids':sorted(bad),'details':details,'baseline_artifact':str(REPO/f'results/basis_a6_expanded/baselines/{model}/{base_ds}/greedy.json'),'baseline_hash':sha(REPO/f'results/basis_a6_expanded/baselines/{model}/{base_ds}/greedy.json'),'modes_checked':modes}
   prefix_name='WHISPER' if model=='whisper' else 'QWEN3_ASR_1P7B'
   for method,suffix in [('add_unique_decoder','ADD_UNIQUE_DECODER'),('conditioning_cs_decoder','CONDITIONING_CS_DECODER')]:
    atomic(ELIG/f'{prefix_name}_{name}_{suffix}.json',{'schema_version':'a6_ott_model_method_manifest_v1','status':'PASS','phase':'B/C','dataset':name,'model':model,'method':method,'ids':sorted(valid),'ineligible_ids':sorted(bad),'causal_position_rule':'baseline_hypothesis_positions must include i>0','baseline_hash':model_maps[model]['baseline_hash'],'mapping_hash':objsha(details),'steering_outcomes_consulted':False})
  audit['model_specific_decoder_eligibility']=model_maps; audit['steering_outcomes_consulted']=False; atomic(ELIG/f'{name}_METHODS.json',audit)
  enc=set(audit['methods']['add_unique_encoder']['eligible_ids'])
  for model, mapping in model_maps.items():
   common=sorted(enc & set(mapping['eligible_ids']))
   prefix_name='WHISPER' if model=='whisper' else 'QWEN3_ASR_1P7B'
   atomic(ELIG/f'{prefix_name}_{name}_COMMON.json',{'schema_version':'a6_ott_model_common_manifest_v1','status':'PASS','dataset':name,'model':model,'ids':common,'source_hash':objsha({'encoder':sorted(enc),'decoder':mapping['eligible_ids']}),'steering_outcomes_consulted':False})
  print(name,{m:{'eligible':len(v['eligible_ids']),'ineligible':len(v['ineligible_ids'])} for m,v in model_maps.items()})
if __name__=='__main__': main()
