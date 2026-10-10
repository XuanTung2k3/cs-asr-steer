#!/usr/bin/env python
"""Audit the omitted zero-vector baseline on fixed100; no adaptation or revised method."""
import json,sys,os,hashlib,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from transformers.modeling_outputs import BaseModelOutput
from csasr.models.whisper import load_whisper,batch_model_inputs
from csasr.utils.config import load_config
from csasr.inference_cf import ttls as L
from csasr.inference_cf.episodic_tta import tensor_bytes_hash,forced_decode
from csasr.lss.sites import assert_no_site_hooks
OUT=ROOT/'results/inference_cf/ttls_r1_independent_audit/zero_census'
def main():
    assert not OUT.exists(),'preserve old attempts'
    plan=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());bundle=load_whisper(load_config(ROOT/'configs/model/whisper_large_v3.yaml'));model=bundle.model;model.eval();model.requires_grad_(False);torch.manual_seed(240924);torch.set_num_threads(1)
    OUT.mkdir(parents=True);start=time.time();h0=tensor_bytes_hash(list(model.parameters()));rows=[];z=torch.zeros(bundle.d_model,device=bundle.device)
    (OUT/'manifest.json').write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_bytes()).hexdigest(),'model':bundle.metadata(),'job':os.getenv('SLURM_JOB_ID'),'purpose':'zero-vector integrity control only; no new objective or optimization','references':False},indent=2))
    for i,s in enumerate(plan['rows']):
        inp=batch_model_inputs(bundle,[s['audio_path']])
        with torch.inference_mode():enc=BaseModelOutput(last_hidden_state=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state)
        b=forced_decode(bundle,enc,L.CB,max_new_tokens=200,capture_layer=16)
        d=L.greedy_decode(bundle,enc,L.CB,hook_factory=lambda:L.ttls_hook(bundle,z,L.ALL,mode='steer'))
        r={'index':i,'id':s['utterance_id'],'dialogue':s['dialogue_id'],'baseline_match':b['tokens']==s['y_B'] and b['terminated']==s['y_B_terminated'],'zero_matches_B0':d['tokens']==b['tokens'] and d['terminated']==b['terminated'],'zero':d}
        assert r['baseline_match'];assert_no_site_hooks(bundle);rows.append(r);(OUT/f'{i:03d}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print('ZERO',i,r['zero_matches_B0'],flush=True)
    (OUT/'summary.json').write_text(json.dumps({'rows':len(rows),'mismatches':[r['id'] for r in rows if not r['zero_matches_B0']],'weights_unchanged':tensor_bytes_hash(list(model.parameters()))==h0,'elapsed_sec':time.time()-start,'job':os.getenv('SLURM_JOB_ID')},indent=2))
if __name__=='__main__':main()
