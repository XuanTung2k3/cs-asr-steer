#!/usr/bin/env python
"""Compact independent audit harness; preserves primary source and outputs.
Re-executes frozen methods, adds independent FFN-consumption and zero-vector controls.
No reference loading, candidate changes, optimizer tuning or method search.
"""
import sys,json,hashlib,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch,numpy as np
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
from csasr.utils.config import load_config
from csasr.models.whisper import load_whisper,batch_model_inputs
from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
from csasr.inference_cf.soft_auto_tta import auto_prompt
from csasr.inference_cf.episodic_tta import LNGuard,decoder_ln_names,tensor_bytes_hash,teacher_logits,adapt,forced_decode
from csasr.inference_cf.core_r2 import tokenizer_partition
from csasr.inference_cf.dir_sprint0 import legal_static
from csasr.inference_cf import ttls as L
from csasr.lss.sites import assert_no_site_hooks
from experiments.inference_cf_ttls_r1 import Ctx,process_row,ARMS
from experiments.inference_cf_p2tta0 import audio_full_sha256
OUT=ROOT/'results/inference_cf/ttls_r1_independent_audit/replay'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def th(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
    sel=json.loads((OUT.parent/'replay_selection.json').read_text());plan=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());primary=json.loads((ROOT/'results/inference_cf/ttls_r1/run1/manifest.json').read_text())
    for path,h in primary['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h.removeprefix('sha256:')
    assert not (OUT/'summary.json').exists(),'preserve earlier replay'
    torch.manual_seed(240924);torch.set_num_threads(1);bundle=load_whisper(load_config(ROOT/'configs/model/whisper_large_v3.yaml'));model=bundle.model;model.eval();model.requires_grad_(False)
    guard=LNGuard(model,decoder_ln_names(model));assert len(guard.names)==194;nonln0=tensor_bytes_hash(guard.others)
    gen=model.generation_config;tok=bundle.processor.tokenizer;sup,beg=list(gen.suppress_tokens or []),list(gen.begin_suppress_tokens or []);part=tokenizer_partition(tok)
    legal=legal_static(tok,sup,tok.eos_token_id);mask=np.zeros(model.config.vocab_size,bool);mask[legal]=True
    def encode(s):
        assert audio_full_sha256(s['audio_path'])==s['audio_full_sha256']
        inp=batch_model_inputs(bundle,[s['audio_path']]);s['_features']=inp['input_features']
        with torch.inference_mode():h=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
        return BaseModelOutput(last_hidden_state=h),BaseModelOutput(last_hidden_state=h.clone())
    def null():
        inp=waveform_model_inputs(bundle,np.zeros(30*bundle.sample_rate,np.float32))
        with torch.inference_mode():h=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
        return BaseModelOutput(last_hidden_state=h),BaseModelOutput(last_hidden_state=h.clone())
    def detect(s):
        with torch.no_grad():return int(model.detect_language(input_features=s.pop('_features'),generation_config=gen)[0])
    def a2(s):
        p=ROOT/s['A2']['masters_npz'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['A2']['masters_npz_sha256'];z=np.load(p)
        return {n:torch.from_numpy(z[f'A2_p{j:03d}']).to(device=guard.params[n].device,dtype=torch.bfloat16) for j,n in enumerate(guard.names)}
    ctx=Ctx(bundle,guard,suppress=sup,begin=beg,eos=tok.eos_token_id,embedded=set(part['embedded_ids']),legal_mask=mask,byte_decoder={v:k for k,v in bytes_to_unicode().items()},cB=L.CB,cE=L.CE_PROMPT,layer=16,encode=encode,encode_null=null,detect_lang=detect,a2_effective=a2,auto_prompt=auto_prompt)
    start=time.time();summary={'selection':sel,'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'model_revision':bundle.revision,'primary_source_commit':primary['git_commit'],'completed_revision':sel['completed_commit'],'comparisons':[],'zero_controls':[],'consumption':[],'ce_reconstruction':[]}
    save(OUT/'manifest.json',{'selection':sel,'primary_manifest':primary['manifest_hash'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'environment':{'torch':torch.__version__,'dtype':str(bundle.dtype)},'model':bundle.metadata(),'started':start,'job':os.getenv('SLURM_JOB_ID')})
    def ff_capture(enc,y,z=None,steps=L.ALL):
        states={}
        def cb(mod,args):states['x']=args[0].detach().clone()
        hook=model.model.decoder.layers[16].final_layer_norm.register_forward_pre_hook(cb)
        try:
            with torch.no_grad():
                if z is None:lg=teacher_logits(model,enc,L.CB,y,None)
                else:
                    with L.ttls_hook(bundle,z,steps,mode='steer',record=True) as rec:lg=teacher_logits(model,enc,L.CB,y,None)
            return states['x'],lg
        finally:hook.remove()
    for i in sel['indices']:
        s=dict(plan['rows'][i]);old=json.loads((ROOT/f'results/inference_cf/ttls_r1/run1/rows/{i:03d}.json').read_text())
        r=process_row(ctx,s,i);save(OUT/f'rows/{i:03d}.json',r)
        for arm in ['B2',*[a[0] for a in ARMS],'CD','ACSUB']:
            a,b=r[arm],old[arm];item={'index':i,'id':s['utterance_id'],'arm':arm,'tokens_equal':a['tokens']==b['tokens'],'termination_equal':a['terminated']==b['terminated']}
            if a.get('status')=='ok' and 'log' in a:
                item['loss_max_abs']=max(abs(x-y) for x,y in zip(a['log']['losses'],b['log']['losses']));item['gradient_max_abs']=max(abs(x-y) for x,y in zip(a['log']['grad_l2'],b['log']['grad_l2']))
                if a['variable']=='TTLS':item['z_max_abs']=float(np.max(np.abs(np.array(a['z_effective'])-b['z_effective'])))
            summary['comparisons'].append(item)
        summary['comparisons'].append({'index':i,'arm':'candidate','equal':r['candidates']==old['candidates']})
        enc_inf,enc_train=encode(dict(plan['rows'][i]));z0=torch.zeros(bundle.d_model,device=bundle.device)
        zero=L.greedy_decode(bundle,enc_inf,L.CB,hook_factory=lambda:L.ttls_hook(bundle,z0,L.ALL,mode='steer'))
        clean,lc=ff_capture(enc_train,s['y_B']);zs,lz=ff_capture(enc_train,s['y_B'],z0)
        delta=(zs-clean).float();q0=3
        zz={'index':i,'id':s['utterance_id'],'tokens_equal_B0':zero['tokens']==s['y_B'],'termination_equal_B0':zero['terminated']==s['y_B_terminated'],'zero_output':zero,'max_abs_logit':float((lc-lz).abs().max()),'ffn_bitwise_equal':torch.equal(zs,clean),'changed_ffn_entries':int(torch.count_nonzero(delta)),'ffn_max_chord':float(delta[:,q0:].norm(dim=-1).max())}
        summary['zero_controls'].append(zz)
        for arm in ['T1','T2','T4','T6']:
            if r[arm]['status']!='ok':continue
            z=torch.tensor(r[arm]['z_effective'],dtype=torch.float32,device=bundle.device);steps=L.ALL if arm in ['T1','T2'] else set(r[arm]['mask']);steered,ls=ff_capture(enc_train,s['y_B'],z,steps)
            dr=(steered-clean).float()[0];norm=clean.float()[0].norm(dim=-1);dn=dr.norm(dim=-1);positions=range(q0,len(dn))
            intended=[j for j in positions if j>=4 and (steps==L.ALL or j-q0 in steps)];outside=[j for j in positions if j not in intended]
            summary['consumption'].append({'index':i,'arm':arm,'z_norm':float(z.norm()),'native_ffn_chord_max':float(dn[intended].max()) if intended else 0,'native_rel_chord_mean':float((dn[intended]/norm[intended]).mean()) if intended else 0,'native_norm_relative_error_max':float(((steered.float()[0].norm(dim=-1)-norm).abs()/norm).max()),'outside_direct_chord_max':float(dn[outside].max()) if outside else 0,'positions':intended,'ffn_clean_hash':th(clean),'ffn_edited_hash':th(steered)})
        # Independently reconstruct old unpreserved A2 from theta0 on selected rows.
        rec=adapt(model,guard,enc_train,L.CB,s['y_A'],s['y_A_valid_mask'],'A2',suppress=sup,begin=beg,eos=tok.eos_token_id)
        eff=rec['effective'];arch=a2(s);same=all(torch.equal(eff[n],arch[n]) for n in guard.names)
        guard.materialize(eff)
        try:dd=forced_decode(bundle,enc_inf,L.CB,max_new_tokens=200,capture_layer=16)
        finally:guard.restore()
        summary['ce_reconstruction'].append({'index':i,'effective_archive_equal':same,'tokens_equal':dd['tokens']==s['A2']['tokens'],'termination_equal':dd['terminated']==s['A2']['terminated'],'losses':rec['losses'] if 'losses' in rec else rec.get('log',{}).get('losses')})
        assert guard.verify();assert_no_site_hooks(bundle)
        print('AUDIT replay',i,s['utterance_id'],'zero_same',zz['tokens_equal_B0'],'old_match',all(x.get('tokens_equal',True) for x in summary['comparisons'] if x['index']==i),flush=True)
        save(OUT/'progress.json',summary)
    summary['elapsed_seconds']=time.time()-start;summary['reset_final']=guard.verify();summary['nonln_unchanged']=tensor_bytes_hash(guard.others)==nonln0;summary['model_grads_none']=all(p.grad is None and not p.requires_grad for p in model.parameters())
    save(OUT/'summary.json',summary)
if __name__=='__main__':main()
