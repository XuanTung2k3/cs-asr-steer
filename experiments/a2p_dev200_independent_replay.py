#!/usr/bin/env python
"""Reference-free compact audit replay. Independent candidate/stable/KL/NLL calculations, frozen LN optimizer."""
from pathlib import Path
import sys,json,hashlib,time,os,math
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch,numpy as np
from transformers.modeling_outputs import BaseModelOutput
from csasr.utils.config import load_config
from csasr.models.whisper import load_whisper,batch_model_inputs
from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
from csasr.inference_cf.episodic_tta import LNGuard,decoder_ln_names,tensor_bytes_hash,teacher_logits,forced_decode,adapt,allowed_ids
from csasr.inference_cf.core_r2 import tokenizer_partition
from csasr.inference_cf import ttls as L
from csasr.lss.sites import assert_no_site_hooks
OUT=ROOT/'results/inference_cf/a2p_dev200_independent_audit/replay'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def allowed_lp(logits,T,sup,beg):
 out=[]
 if T:out.append(torch.log_softmax(logits[:1].index_select(1,allowed_ids(logits.shape[-1],0,sup,beg,logits.device)),dim=-1))
 if T>1:out.append(torch.log_softmax(logits[1:T].index_select(1,allowed_ids(logits.shape[-1],1,sup,beg,logits.device)),dim=-1))
 return out

def independent_terms(ep,s,logq,S,sup,beg):
 y=s['y_A'];x=ep.forward(L.CB,y);terms=[]
 for start,stop,step in [(0,1,0),(1,len(y)+1,1)]:
  z=x[start:stop];idx=allowed_ids(x.shape[-1],step,sup,beg,x.device);inv=torch.full((x.shape[-1],),-1,device=x.device,dtype=torch.long);inv[idx]=torch.arange(len(idx),device=x.device)
  lp=torch.log_softmax(z.index_select(1,idx),dim=-1);yy=torch.tensor(y[start:stop],device=x.device,dtype=torch.long)
  if len(yy):terms.append(lp[torch.arange(len(yy),device=x.device),inv[yy].clamp_min(0)])
 valid=torch.tensor(s['y_A_valid_mask'],device=x.device,dtype=torch.bool)
 ce=(-torch.cat(terms)[valid]).mean() if valid.any() else sum((p*0).sum() for p in ep.params.values())
 xb=x if s['y_A']==s['y_B'] else ep.forward(L.CB,s['y_B'])
 if S:
  logp=allowed_lp(xb,len(s['y_B']),sup,beg);kl=torch.cat([(q.exp()*(q-p)).sum(-1) for q,p in zip(logq,logp)]);P=kl[torch.tensor(S,device=x.device)].mean()
 else:P=xb.sum()*0
 return ce,P

def candidate_audit(lp,s,embedded,eos):
 M=[];acc=[];candidate=None
 for t in range(1,len(s['y_B'])):
  b=s['y_B'][t]
  if not s['utf8_ok'][t] or b==eos or b in embedded:continue
  e=int(np.argmax(lp['cE_clean'][t]))
  if e not in embedded or e==b:continue
  M.append(t);es=[];finite=True
  for c in ('cB','cE'):
   lc,ln=lp[c+'_clean'][t],lp[c+'_null'][t];vals=(lc[e],ln[e],lc[b],ln[b])
   finite=finite and math.isfinite(lc[e]) and math.isfinite(lc[b]) and not any(math.isnan(v) for v in vals)
   es.append((lc[e]-max(ln[e],math.log(1e-12)))-(lc[b]-max(ln[b],math.log(1e-12))))
  if finite and min(es)>=math.log(10):
   acc.append(t)
   if candidate is None:candidate=e
 p0=[float(np.exp(lp['cB_clean'][t,b])) for t,b in enumerate(s['y_B'])]
 S=[t for t,(p,v) in enumerate(zip(p0,s['y_B_valid_mask'])) if v and p>=.5 and t not in M]
 return {'M':M,'accepted':acc,'t_star':acc[0] if acc else None,'c_star':candidate,'S':S,'p0_top':p0,'T':len(s['y_B'])}

def main():
 assert not OUT.exists(),'never overwrite an audit attempt'
 base=OUT.parent;plan=json.loads((ROOT/'results/inference_cf/a2p_dev200/plan_sealed.json').read_text());man=json.loads((ROOT/'results/inference_cf/a2p_dev200/run1/manifest.json').read_text());am=json.loads((base/'replay_manifest.json').read_text());sel=json.loads((base/'replay_selection.json').read_text())
 for p,h in {**man['sources'],**am['sources']}.items():assert sha(ROOT/p)==h.removeprefix('sha256:')
 for p,h in man['model']['files'].items():assert sha(Path(man['model']['dir'])/p)==h.removeprefix('sha256:')
 torch.manual_seed(240924);torch.set_num_threads(1);b=load_whisper(load_config(ROOT/'configs/model/whisper_large_v3.yaml'));model=b.model;model.eval();model.requires_grad_(False)
 guard=LNGuard(model,decoder_ln_names(model));assert len(guard.names)==194 and sum(p.numel() for p in guard.theta0.values())==248320
 all0=tensor_bytes_hash(list(model.parameters()));sup,beg=list(model.generation_config.suppress_tokens or []),list(model.generation_config.begin_suppress_tokens or []);part=tokenizer_partition(b.processor.tokenizer);eos=b.processor.tokenizer.eos_token_id
 assert {'suppress':sup,'begin':beg}==plan['suppression'] and part['hash']==plan['partition_hash'] and eos==plan['eos']
 def encode(s):
  assert sha(s['audio_path'])==s['audio_full_sha256'];inp=batch_model_inputs(b,[s['audio_path']])
  with torch.inference_mode():h=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
  return BaseModelOutput(last_hidden_state=h),BaseModelOutput(last_hidden_state=h.clone())
 inp=waveform_model_inputs(b,np.zeros(30*b.sample_rate,np.float32))
 with torch.inference_mode():nh=model.model.encoder(input_features=inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
 null=BaseModelOutput(last_hidden_state=nh.clone());start=time.time();summary={'job':os.getenv('SLURM_JOB_ID'),'selection':sel,'manifest':am,'rows':[],'model':b.metadata()}
 for i in sel['indices']:
  s=plan['rows'][i];old=json.loads((ROOT/f'results/inference_cf/a2p_dev200/run1/rows/{i:03d}.json').read_text());inf,enc=encode(s);d0=forced_decode(b,inf,L.CB,max_new_tokens=200,capture_layer=16)
  assert d0['tokens']==s['y_B'] and d0['terminated']==s['y_B_terminated'] and d0['text']==s['y_B_text']
  with torch.no_grad():lg={k:teacher_logits(model,encoded,prompt,s['y_B'],None) for k,encoded,prompt in [('cB_clean',enc,L.CB),('cB_null',null,L.CB),('cE_clean',enc,L.CE_PROMPT),('cE_null',null,L.CE_PROMPT)]}
  lp={}
  for k,x in lg.items():
   z=x.detach().float().cpu().clone();z[:,sup]=-float('inf');z[0,beg]=-float('inf');lp[k]=torch.log_softmax(z,dim=-1).numpy().astype(np.float64)
  cand=candidate_audit(lp,s,set(part['embedded_ids']),eos)
  assert all(cand[k]==old['candidates'][k] for k in cand)
  logq=[x.detach() for x in allowed_lp(lg['cB_clean'],len(s['y_B']),sup,beg)];del lg,lp
  episode=L.Episode(b,guard,enc,'LN',layer=16);assert not episode.opt.state and guard.verify()
  def loss(ep):
   ce,P=independent_terms(ep,s,logq,cand['S'],sup,beg)
   return {'loss':ce+P,'parts':{'CE':float(ce.detach()),'P':float(P.detach())}}
  # Direct loss/gradient comparison before any optimization; uses original only as a check.
  own=loss(episode);og=torch.autograd.grad(own['loss'],list(episode.params.values()))
  from experiments.inference_cf_a2p_dev200 import b3_loss_fn
  from types import SimpleNamespace
  ctx=SimpleNamespace(suppress=sup,begin=beg,eos=eos,cB=L.CB)
  ref=b3_loss_fn(ctx,s,{'logq':logq},cand['S'])(episode);rg=torch.autograd.grad(ref['loss'],list(episode.params.values()))
  grad_diff=max(float((a-c).abs().max()) for a,c in zip(og,rg));assert torch.equal(own['loss'],ref['loss']) and grad_diff==0
  del own,ref,og,rg
  r=episode.run(loss,{})['log'];eff=episode.effective();guard.materialize(eff)
  try:d3=forced_decode(b,inf,L.CB,max_new_tokens=200,capture_layer=16);state=guard.current_hash()
  finally:guard.restore()
  assert all(d3[k]==old['B3'][k] for k in ('tokens','text','terminated')) and state==old['B3']['state_hash']
  lossdiff=max(abs(a-c) for a,c in zip(r['losses'],old['B3']['log']['losses']));graddiff=max(abs(a-c) for a,c in zip(r['grad_l2'],old['B3']['log']['grad_l2']))
  assert lossdiff==graddiff==0 and r['parts']==old['B3']['log']['parts'] and r['master_delta_l2']==old['B3']['log']['master_delta_l2'] and r['effective_delta_l2']==old['B3']['log']['effective_delta_l2']
  masters_hash=tensor_bytes_hash([episode.params[n] for n in guard.names]);optimizer_states_hash=tensor_bytes_hash([episode.opt.state[p][k] for p in episode.params.values() for k in ('step','exp_avg','exp_avg_sq')]);del episode,eff
  a2=adapt(model,guard,enc,L.CB,s['y_A'],s['y_A_valid_mask'],'A2',suppress=sup,begin=beg,eos=eos)
  guard.materialize(a2['effective'])
  try:d2=forced_decode(b,inf,L.CB,max_new_tokens=200,capture_layer=16);state2=guard.current_hash()
  finally:guard.restore()
  assert all(d2[k]==s['A2'][k] for k in ('tokens','text','terminated')) and state2==s['A2']['state_hash'];assert a2['log']['losses']==s['A2']['losses']
  fresh=L.Episode(b,guard,enc,'LN');assert not fresh.opt.state and all(torch.equal(p.detach(),guard.theta0[n].float()) for n,p in fresh.params.items());del fresh,a2
  assert guard.verify();assert_no_site_hooks(b)
  result={'index':i,'id':s['utterance_id'],'baseline_exact':True,'candidates':cand,'loss_gradient_crosscheck_exact':True,'B3':{**d3,'log':r,'state_hash':state,'fp32_masters_hash':masters_hash,'optimizer_states_hash':optimizer_states_hash},'B2':{**d2,'state_hash':state2},'B3_loss_diff':lossdiff,'B3_grad_diff':graddiff,'A2_state_exact':True,'reset_exact':True}
  save(OUT/f'rows/{i:03d}.json',result);summary['rows'].append({k:v for k,v in result.items() if k not in ('B2','B3','candidates')});print('REPLAY',i,s['utterance_id'],'EXACT',flush=True)
 summary.update(elapsed_sec=time.time()-start,all_parameters_unchanged=tensor_bytes_hash(list(model.parameters()))==all0,reset_exact=guard.verify(),model_grad_free=guard.flags_ok())
 assert summary['all_parameters_unchanged'] and summary['reset_exact'] and summary['model_grad_free'];save(OUT/'summary.json',summary);print('AUDIT REPLAY COMPLETE',summary['elapsed_sec'],flush=True)
if __name__=='__main__':main()
