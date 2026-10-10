#!/usr/bin/env python
"""Independent compact audit: all100 zero-only census + selected24 frozen repaired episodes.
No references, new method, objective changes or parameter optimization. Separate FFN-input probes and gradient path.
"""
from pathlib import Path
import sys,json,hashlib,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch,numpy as np
from transformers.modeling_outputs import BaseModelOutput
from csasr.utils.config import load_config
from csasr.models.whisper import load_whisper,batch_model_inputs
from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
from csasr.inference_cf.episodic_tta import LNGuard,decoder_ln_names,tensor_bytes_hash,teacher_logits,forced_decode
from csasr.inference_cf.core_r2 import tokenizer_partition
from csasr.inference_cf import ttls as L,ttls_r1r as R
from csasr.lss.sites import assert_no_site_hooks
from experiments.inference_cf_ttls_r1r import Ctx,integrity_row,process_row
OUT=ROOT/'results/inference_cf/ttls_r1r_independent_audit/replay_r3'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_repair(h,z,g):
    # Independent direct expression; stays differentiable at zero. Uses native dtype, not a z==0 bypass.
    gain=g.to(h.dtype).unsqueeze(-1);v=h+gain*z.to(h.dtype).view(1,1,-1)
    a=torch.linalg.vector_norm(h,dim=-1,keepdim=True);b=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    out=v*(a/torch.maximum(b,torch.full_like(b,1e-6)))
    return torch.where(gain>0,out,h)
def main():
    assert not (OUT/'summary.json').exists(),'preserve prior attempt'
    sel=json.loads((OUT.parent/'replay_selection.json').read_text());plan=json.loads((ROOT/'results/inference_cf/ttls_r1/plan_sealed.json').read_text());man=json.loads((ROOT/'results/inference_cf/ttls_r1r/run1/manifest.json').read_text());auditman=json.loads((OUT.parent/'manifest_r3.json').read_text())
    for p,h in {**man['sources'],**auditman['sources']}.items():assert sha(ROOT/p)==h.removeprefix('sha256:')
    for p,h in man['model']['files'].items():assert sha(Path(man['model']['dir'])/p)==h.removeprefix('sha256:')
    torch.manual_seed(240924);torch.set_num_threads(1);bundle=load_whisper(load_config(ROOT/'configs/model/whisper_large_v3.yaml'));model=bundle.model;model.eval();model.requires_grad_(False)
    guard=LNGuard(model,decoder_ln_names(model));all0=tensor_bytes_hash(list(model.parameters()));gen=model.generation_config;tok=bundle.processor.tokenizer;sup,beg=list(gen.suppress_tokens or []),list(gen.begin_suppress_tokens or []);part=tokenizer_partition(tok)
    assert part['hash']==plan['partition_hash'];assert {'suppress':sup,'begin':beg}==plan['suppression'];assert tok.eos_token_id==plan['eos']
    def encode(s):
        assert sha(s['audio_path'])==s['audio_full_sha256']
        inp=batch_model_inputs(bundle,[s['audio_path']])
        with torch.inference_mode():h=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
        return BaseModelOutput(last_hidden_state=h),BaseModelOutput(last_hidden_state=h.clone())
    def null():
        inp=waveform_model_inputs(bundle,np.zeros(30*bundle.sample_rate,np.float32))
        with torch.inference_mode():h=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
        return BaseModelOutput(last_hidden_state=h),BaseModelOutput(last_hidden_state=h.clone())
    ctx=Ctx(bundle,guard,suppress=sup,begin=beg,eos=tok.eos_token_id,embedded=set(part['embedded_ids']),cB=L.CB,cE=L.CE_PROMPT,layer=16,encode=encode,encode_null=null)
    start=time.time();summ={'job':os.getenv('SLURM_JOB_ID'),'selection':sel,'zero100':[],'comparisons':[],'consumption':[],'gradient_crosschecks':[],'audit_manifest':auditman,'model':bundle.metadata()}
    def capture(enc,y,z=None,steps=L.ALL,historical=False):
        cap={};hk=model.model.decoder.layers[16].final_layer_norm.register_forward_pre_hook(lambda mod,args:cap.update(x=args[0].detach().clone()))
        try:
            with torch.no_grad():
                if z is None:logits=teacher_logits(model,enc,L.CB,y,None)
                else:
                    with (L.ttls_hook if historical else R.ttls_hook_r1r)(bundle,z,steps,mode='steer'):logits=teacher_logits(model,enc,L.CB,y,None)
            return cap['x'],logits
        finally:hk.remove()
    def grad_probe(enc,s):
        ep=R.EpisodeR1R(bundle,enc,L.ALL);_,loss=L.ce_terms(ep,s['y_A'],s['y_A_valid_mask'],sup,beg,tok.eos_token_id,prompt=L.CB);g=torch.autograd.grad(loss,ep.z)[0]
        z=torch.zeros(bundle.d_model,device=bundle.device,requires_grad=True)
        class Direct:
            params={'z':z}
            def forward(self,prompt,y):
                # Intervene before BOTH the FFN normalization and its residual skip, as DG-02 requires.
                # A final_layer_norm pre-hook alone misses the residual skip gradient.
                layer=model.model.decoder.layers[16];cap={}
                def remember(mod,args):cap['q']=args[0]
                def intervention(mod,args,output):
                    u=output[0] if isinstance(output,tuple) else output;h=cap['q']+u
                    q=torch.arange(h.shape[1],device=h.device);gain=(q>=4).to(h.dtype).view(1,-1)
                    edited=independent_repair(h,z,gain);out=u+(edited-h)
                    return (out,)+output[1:] if isinstance(output,tuple) else out
                hooks=[layer.encoder_attn_layer_norm.register_forward_pre_hook(remember),layer.encoder_attn.register_forward_hook(intervention)]
                try:return teacher_logits(model,enc,prompt,y,None)
                finally:
                    for hook in hooks:hook.remove()
        _,li=L.ce_terms(Direct(),s['y_A'],s['y_A_valid_mask'],sup,beg,tok.eos_token_id,prompt=L.CB);gi=torch.autograd.grad(li,z)[0]
        rel=float((g-gi).norm()/g.norm());cos=float(torch.nn.functional.cosine_similarity(g,gi,dim=0))
        assert torch.equal(loss.detach(),li.detach()) and rel<=.02 and cos>=.999 and float(g.norm())>0
        with torch.no_grad():
            clean_B=teacher_logits(model,enc,L.CB,s['y_B'],None)
            logq=L.allowed_log_probs(clean_B,len(s['y_B']),sup,beg)
        logits_B=ep.forward(L.CB,s['y_B']);S=json.loads((ROOT/f"results/inference_cf/ttls_r1r/run1/integrity/{plan['ids'].index(s['utterance_id']):03d}.json").read_text())['candidates']['S']
        P=L.preservation_term(logits_B,logq,S,len(s['y_B']),sup,beg);gp=torch.autograd.grad(P,ep.z)[0]
        assert float(P)==0.0
        return {'initial_P':float(P),'initial_P_grad_norm':float(gp.norm()),'initial_P_grad_relative_to_CE':float(gp.norm()/g.norm()),'id':s['utterance_id'],'primary_norm':float(g.norm()),'independent_norm':float(gi.norm()),'relative_difference':rel,'cosine':cos,'loss_equal':True,'tolerance_frozen':{'relative':.02,'cosine':.999}}
    for i,s in enumerate(plan['rows']):
        inf,train=encode(s);clean=forced_decode(bundle,inf,L.CB,max_new_tokens=200,capture_layer=16);zero=torch.zeros(bundle.d_model,device=bundle.device)
        zd=L.greedy_decode(bundle,inf,L.CB,hook_factory=lambda:R.ttls_hook_r1r(bundle,zero,L.ALL,mode='steer'),max_new_tokens=200,capture_layer=16)
        ff,lc=capture(train,s['y_B']);fz,lz=capture(train,s['y_B'],zero);fh,lh=capture(train,s['y_B'],zero,historical=True)
        item={'index':i,'id':s['utterance_id'],'baseline_equal':clean['tokens']==s['y_B'] and clean['terminated']==s['y_B_terminated'],'zero':zd,'zero_free_equal':all(zd[k]==clean[k] for k in ('tokens','text','terminated')),'zero_ffn_equal':torch.equal(ff,fz),'zero_logits_equal':torch.equal(lc,lz),'historical_logits_diff':float((lh-lc).abs().max()),'historical_ffn_diff':float((fh-ff).float().norm(dim=-1).max())}
        assert all(item[k] for k in ['baseline_equal','zero_free_equal','zero_ffn_equal','zero_logits_equal']);summ['zero100'].append(item);save(OUT/f'zero/{i:03d}.json',item)
        if i in sel['indices']:
            old=json.loads((ROOT/f'results/inference_cf/ttls_r1r/run1/rows/{i:03d}.json').read_text());hist=json.loads((ROOT/f'results/inference_cf/ttls_r1/run1/rows/{i:03d}.json').read_text())
            A=integrity_row(ctx,dict(s),i,hist['candidates']);assert A['pass'];r=process_row(ctx,dict(s),i,A);save(OUT/f'rows/{i:03d}.json',{'phaseA':A,'phaseB':r})
            for a in ['T1','T2','T4','T6']:
                x,y=r[a],old[a];cmp={'index':i,'arm':a,'tokens_equal':x['tokens']==y['tokens'],'text_equal':x['text']==y['text'],'termination_equal':x['terminated']==y['terminated']}
                if x['status']=='ok':
                    cmp.update(loss_diff=max(abs(u-v) for u,v in zip(x['log']['losses'],y['log']['losses'])),grad_diff=max(abs(u-v) for u,v in zip(x['log']['grad_l2'],y['log']['grad_l2'])),z_diff=float(np.max(np.abs(np.array(x['z_effective'])-y['z_effective']))))
                    z=torch.tensor(x['z_effective'],device=bundle.device);steps=L.ALL if a in ['T1','T2'] else set(x['mask']);fx,lx=capture(train,s['y_B'],z,steps);d=(fx-ff).float()[0];n=ff.float()[0].norm(dim=-1);qs=[q for q in range(len(n)) if q>=4 and (steps==L.ALL or q-3 in steps)];outside=[q for q in range(len(n)) if q not in qs]
                    probe={'index':i,'arm':a,'native_chord_max':float(d[qs].norm(dim=-1).max()),'norm_rel_max':float(((fx.float()[0].norm(dim=-1)-n).abs()/n).max()),'outside_max':float(d[outside].norm(dim=-1).max()) if outside else 0,'z_norm':float(z.norm())}
                    assert probe['native_chord_max']>0 and probe['norm_rel_max']<=.02 and probe['outside_max']==0 and probe['z_norm']<=L.E_STAR*(1+1e-6);summ['consumption'].append(probe)
                assert cmp['tokens_equal'] and cmp['termination_equal'] and cmp['text_equal'];assert cmp.get('loss_diff',0)==cmp.get('grad_diff',0)==cmp.get('z_diff',0)==0;summ['comparisons'].append(cmp)
            if i in [2,31,53]:summ['gradient_crosschecks'].append(grad_probe(train,s))
        assert guard.verify();assert_no_site_hooks(bundle)
        if i%10==0:print('audit row',i,'zero_identity',item['zero_free_equal'],flush=True)
    summ.update(elapsed_seconds=time.time()-start,weights_unchanged=tensor_bytes_hash(list(model.parameters()))==all0,reset=guard.verify(),model_grad_free=all(p.grad is None and not p.requires_grad for p in model.parameters()))
    assert summ['weights_unchanged'] and summ['reset'] and summ['model_grad_free'];save(OUT/'summary.json',summ);print('AUDIT COMPLETED',summ['elapsed_seconds'],len(summ['comparisons']),flush=True)
if __name__=='__main__':main()
