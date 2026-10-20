"""Explicit follow-up O1/O2/O3 execution; never invoked automatically by the primary pilot.

Default is CPU dry-run. GPU execution requires --execute, a completed primary pilot, a new run ID,
an explicit time cap and a clean committed source tree. All nine cells share the same preservation regime.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import hashlib
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

from reft.adapters import LoRALinear
from reft.a5000.adapters import EncoderAdapter
from reft.a5000.data import ROOT, config, runtime, output, atomic, sha, load_rows
from reft.a5000.objectives import LIDHead, token_groups, weighted_ce


class PracticalWeightLoRA(nn.Module):
    """Broad decoder self/cross-attention q/v LoRA, r8, alpha16, no dropout/bias adaptation.

    2,621,440 learned scalars at d1280/32 layers. A practical capacity comparator, not an established
    best-tuned baseline. Encoder is frozen/unmodified, so the exact frozen encoder cache remains valid.
    """
    def __init__(self,model,rank=8,alpha=16,seed=42):
        super().__init__()
        self.wrapped=nn.ModuleDict();self.locations=[]
        for i,layer in enumerate(model.model.decoder.layers):
            for attname in ['self_attn','encoder_attn']:
                att=getattr(layer,attname)
                for projection in ['q_proj','v_proj']:
                    key=f'l{i}_{attname}_{projection}'
                    self.wrapped[key]=LoRALinear(getattr(att,projection),rank,alpha,seed+len(self.locations))
                    self.locations.append((att,projection,key))
        self._installed=False

    def forward(self,h):return h

    @contextlib.contextmanager
    def active(self,model=None):
        if self._installed:raise RuntimeError('duplicate weight-LoRA installation')
        self._installed=True
        for att,name,key in self.locations:
            assert getattr(att,name) is self.wrapped[key].base
            setattr(att,name,self.wrapped[key])
        try:yield self
        finally:
            for att,name,key in self.locations:setattr(att,name,self.wrapped[key].base)
            self._installed=False

    def adaptation_state(self):
        return {f'{key}.{name}':getattr(w,name).detach().cpu() for key,w in self.wrapped.items() for name in ['A','B']}

    def restore(self,state):
        with torch.no_grad():
            for key,w in self.wrapped.items():
                for name in ['A','B']:getattr(w,name).copy_(state[f'{key}.{name}'])

    def count(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)


def groups_for_labels(tokenizer,targets,labels):
    g=torch.full_like(labels,-100)
    g[:,3:3+len(targets)]=token_groups(tokenizer,targets).to(labels.device)
    return g


def logical_loss(logits,labels,groups,objective,token_den,weighted_den,lid_den,states=None,head=None,
                 lid_lambda=0.,english_weight=1.):
    den=weighted_den if objective=='O3' else token_den
    ce=weighted_ce(logits,labels,groups,english_weight if objective=='O3' else 1.,denominator=den)
    if objective!='O2' or lid_lambda==0:return ce
    z=head.classifier(states.float())
    if lid_den==0:return ce+z.sum()*0
    aux=F.cross_entropy(z.reshape(-1,2),groups.reshape(-1),ignore_index=-100,reduction='sum')/lid_den
    return ce+lid_lambda*aux


def label_audit():
    from transformers import WhisperTokenizer
    tok=WhisperTokenizer.from_pretrained(str(__import__('pathlib').Path(runtime()['model']['root']).expanduser()),local_files_only=True)
    from reft.text import target_text
    audit={}
    for role in ['A5000_T394','A5000_V106']:
        counts={'ZH':0,'EN':0,'UNKNOWN':0,'content_and_eos':0}
        for row in load_rows(role):
            targets=tok.encode(target_text(row['transcript_raw']),add_special_tokens=False)+[tok.eos_token_id]
            g=token_groups(tok,targets)
            for name,tag in [('ZH',0),('EN',1),('UNKNOWN',-100)]:counts[name]+=int((g==tag).sum())
            counts['content_and_eos']+=len(targets)
        counts['known_fraction']=(counts['ZH']+counts['EN'])/counts['content_and_eos']
        audit[role]=counts
    atomic(output()/'followup/label_audit.json',{'policy':'conservative script proxy; names/romanized Mandarin ambiguous; '
        'partial UTF8/mixed/numeric/special excluded from LID, default ASR weight1','roles':audit,'C150_opened':False})
    print(json.dumps(audit,indent=2))


def run(args):
    from reft.a5000.runner import Runtime,Budget,Paused,git,save_torch,evaluate,selection_key,memory,check
    if not (output()/'comparison.json').exists():raise RuntimeError('Complete primary matched O1 pilot first')
    if git('status','--porcelain'):raise RuntimeError('Commit reviewed follow-up source before GPU execution')
    if args.max_seconds<=0:raise ValueError('--execute requires an explicit positive --max-seconds')
    assert args.run_id and '/' not in args.run_id and '..' not in args.run_id
    spec=vars(args).copy();spec['source_commit']=git('rev-parse','HEAD')
    dest=output()/'followup'/args.run_id
    if dest.exists() and not args.resume:raise FileExistsError('Use a new run ID or explicit --resume')
    dest.mkdir(parents=True,exist_ok=True)
    # Resume flag itself is operational, not part of scientific configuration identity.
    spec.pop('resume')
    if (dest/'resolved.json').exists():assert json.loads((dest/'resolved.json').read_text())==spec
    else:atomic(dest/'resolved.json',spec)
    budget=Budget('followup');start=time.monotonic();status='FAILED'
    original_tick=budget.tick
    def bounded_tick():
        original_tick()
        if time.monotonic()-start>=args.max_seconds:raise Paused('Explicit follow-up time cap reached')
    budget.tick=bounded_tick
    class FollowupRuntime(Runtime):
        def __init__(self,budget):
            # Use the same verified cache/decoder without rewriting completed primary runtime evidence.
            from csasr.models.whisper import load_whisper
            from reft.provenance import weights_hash,environment
            self.budget=budget;self.C=config();self.R=runtime();self.freeze=check().copy()
            primary_source=self.freeze['source_commit'];self.freeze['source_commit']=spec['source_commit']
            torch.manual_seed(42);torch.set_num_threads(2)
            self.b=load_whisper({'model':{'id':str(Path(self.R['model']['root']).expanduser()),
                'dtype':'bfloat16','device':'cuda','local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
            self.initial_weights=weights_hash(self.b.model)
            self.T=load_rows('A5000_T394');self.V=load_rows('A5000_V106')
            self.eos=self.b.processor.tokenizer.eos_token_id
            self.fingerprint=hashlib.sha256(json.dumps({'model':self.R['model'],'data':self.freeze['role_manifest_sha256'],
                'processor':sha(Path(self.R['model']['root']).expanduser()/'tokenizer_config.json')},sort_keys=True).encode()).hexdigest()
            atomic(dest/'runtime.json',{'environment':environment(),'source_commit':spec['source_commit'],
                'primary_core_source':primary_source,'fingerprint':self.fingerprint,'initial_weights':self.initial_weights})
    with (output()/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            occupied=subprocess.check_output(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0'),
                '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if occupied:raise RuntimeError('Selected GPU is occupied: '+occupied)
            R=FollowupRuntime(budget)
            torch.manual_seed(42)
            ad=(PracticalWeightLoRA(R.b.model) if args.method=='practical_weight_lora' else EncoderAdapter(args.method)).to('cuda')
            # Same auxiliary-head initialization across parameterizations; adapter construction consumes different RNG.
            torch.manual_seed(42)
            head=LIDHead(1280).to('cuda') if args.objective=='O2' else None
            params=[p for p in ad.parameters() if p.requires_grad]+(list(head.parameters()) if head else [])
            opt=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.99),eps=1e-6,weight_decay=0.)
            atomic(dest/'parameters.json',{'adaptation':ad.count(),'auxiliary':sum(p.numel() for p in head.parameters()) if head else 0})
            last=dest/'last.pt';cursor=0;history=[]
            if last.exists():
                s=torch.load(last,map_location='cpu',weights_only=True)
                if args.method=='practical_weight_lora':ad.restore(s['adapter'])
                else:ad.load_state_dict(s['adapter'])
                if head:head.load_state_dict(s['head'])
                opt.load_state_dict(s['optimizer']);cursor=s['cursor'];history=s['history']
            # Cache before installing weight adapters: frozen language decisions must remain unadapted.
            for row in R.T+R.V:R.cache(row)
            baseline={p.stem:json.loads(p.read_text()) for p in (output()/'B0/predictions').glob('*.json')}
            global_step=0
            for epoch in range(1,args.epochs+1):
                order=np.random.default_rng(42+epoch).permutation(len(R.T))
                for lo in range(0,len(order),16):
                    global_step+=1
                    if global_step<=cursor:continue
                    budget.tick()
                    if time.monotonic()-start>=args.max_seconds:
                        atomic(dest/'status.json',{'status':'PAUSED','cursor':cursor});status='PAUSED';return
                    batch=[R.T[int(i)] for i in order[lo:lo+16]]
                    cached=[R.cache(r) for r in batch]
                    tg=[token_groups(R.b.processor.tokenizer,d['targets']) for d in cached]
                    token_den=sum(len(g) for g in tg)
                    lid_den=sum(int((g!=-100).sum()) for g in tg)
                    weighted_den=sum(float(torch.where(g==1,args.english_weight,1.).sum()) for g in tg)
                    opt.zero_grad(set_to_none=True);loss_sum=0.
                    for row in batch:
                        budget.tick()
                        d,enc,X,Y=R.inputs(row)
                        ctx=ad.active() if args.method=='practical_weight_lora' else contextlib.nullcontext()
                        with ctx:
                            h=enc if args.method=='practical_weight_lora' else ad(enc)
                            o=R.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=h),decoder_input_ids=X,
                                attention_mask=d['mask'].to('cuda'),use_cache=False,output_hidden_states=head is not None)
                        groups=groups_for_labels(R.b.processor.tokenizer,d['targets'],Y)
                        loss=logical_loss(o.logits,Y,groups,args.objective,token_den,weighted_den,lid_den,
                            states=o.decoder_hidden_states[-1] if head else None,head=head,
                            lid_lambda=args.lid_lambda,english_weight=args.english_weight)
                        assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
                        del o,loss
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    torch.nn.utils.clip_grad_norm_(params,1.);opt.step();cursor=global_step
                    state=ad.adaptation_state() if args.method=='practical_weight_lora' else ad.state_dict()
                    save_torch(last,{'adapter':state,'head':head.state_dict() if head else {},'optimizer':opt.state_dict(),
                        'cursor':cursor,'history':history})
                    with (dest/'training.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'step':cursor,'loss':loss_sum})+'\n')
                # Do not repeat completed epoch validation on resume.
                if len(history)<epoch:
                    ep=dest/f'epoch{epoch}';checkpoint=ep/'adapter.pt'
                    state=ad.adaptation_state() if args.method=='practical_weight_lora' else ad.state_dict()
                    save_torch(checkpoint,state)
                    ctx=ad.active() if args.method=='practical_weight_lora' else contextlib.nullcontext()
                    with ctx:m,_=evaluate(R,ad,ep/'validation',checkpoint,baseline)
                    history.append({'epoch':epoch,'metrics':{k:v for k,v in m.items() if k!='per_utterance'},
                        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),**memory()})
                    atomic(dest/'history.json',history)
                    save_torch(last,{'adapter':state,'head':head.state_dict() if head else {},'optimizer':opt.state_dict(),
                        'cursor':cursor,'history':history})
            selected=min(history,key=lambda h:selection_key(h['metrics'],h['epoch']))
            atomic(dest/'selection.json',selected);atomic(dest/'status.json',{'status':'COMPLETE'});status='COMPLETE'
        except Paused as e:
            status='PAUSED';atomic(dest/'status.json',{'status':status,'reason':str(e)})
        finally:budget.close(status)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['ereft','lra','practical_weight_lora'],default='ereft')
    p.add_argument('--objective',choices=['O1','O2','O3'],default='O1')
    p.add_argument('--lr',type=float,default=0.0001);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--lid-lambda',type=float,default=.1);p.add_argument('--english-weight',type=float,default=2.)
    p.add_argument('--run-id');p.add_argument('--max-seconds',type=int,default=0)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--label-audit',action='store_true');args=p.parse_args()
    if args.label_audit:label_audit()
    elif args.execute:run(args)
    else:print(json.dumps({'dry_run':True,'resolved':vars(args),'preservation':'beta0',
        'matrix':[{'method':m,'objective':o} for m in ['ereft','lra','practical_weight_lora'] for o in ['O1','O2','O3']],
        'warning':'No GPU work started. Strong comparator status requires actual preflight/tuning; labels are script proxies.'},indent=2))


if __name__=='__main__':main()
