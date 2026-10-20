"""Actual bounded scientific workers. Every arm is resumable at optimizer boundaries."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
from .state import atomic,read,sha,identity,now
from .methods import RepresentationAdapter,Allocation,EncoderWeightLoRA
from .data import rows
from .evaluation import metrics,pilot_decision
from reft.auto_policy import build_tf_batch
from reft.a5000.objectives import token_groups,weighted_ce,EN
from reft.a5000.runner import save_torch,memory
from reft.provenance import weights_hash,environment
from reft.text import target_text
from csasr.models.whisper import load_whisper,batch_model_inputs

STOP=False


class Halt(Exception): pass


def stopping(signum,frame):
    global STOP
    STOP=True


class Runtime:
    def __init__(self,root):
        self.root=Path(root); self.plan=read(self.root/'frozen_plan.json'); self.C=self.plan['config']
        self.start=time.monotonic(); self.offset=0; self.arm_usage=None; self.arm_start=None; self.arm_prior=0.
        budget=self.root/'worker_usage.json'
        if budget.exists(): self.offset=read(budget)['resident_seconds']
        self.T=rows(self.C,self.root,'A5000_T2000'); self.V=rows(self.C,self.root,'A5000_V300')
        self.usage=budget; self.tick()
        torch.set_num_threads(2); torch.manual_seed(42)
        torch.cuda.init(); self.initial_free=torch.cuda.mem_get_info()[0]
        self.b=load_whisper({'model':{'id':self.C['model_root'],'dtype':'bfloat16','device':'cuda',
            'local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
        self.b.model.eval(); self.backbone=list(self.b.model.parameters())
        if self.b.revision!='sha256:'+self.C['model_sha256']: raise ValueError('Model hash mismatch')
        if self.b.d_model!=1280 or abs(self.b.encoder_step_sec-.02)>1e-8: raise ValueError('Unexpected model geometry')
        self.initial_hash=weights_hash(self.b.model)
        self.eos=self.b.processor.tokenizer.eos_token_id
        self.fp=identity({'model':self.C['model_sha256'],'processor':sha(Path(self.C['model_root'])/'tokenizer_config.json'),
            'data':self.plan['data_sha256'],'source':self.plan['source_commit']})
        self.cache_root=Path(self.C.get('shared_cache_root',self.root/'encoder_cache')); self.cache_root.mkdir(parents=True,exist_ok=True)
        # Deterministic frozen-model mode and explicit checkpoint wrappers preserve
        # all encoder dropout/SpecAugment semantics without forcing train() mode.
        self.original_encoder_forwards=[]
        for layer in self.b.model.model.encoder.layers:
            original=layer.forward
            def checkpointed(*args,_forward=original,**kwargs):
                if torch.is_grad_enabled() and args and args[0].requires_grad:
                    from torch.utils.checkpoint import checkpoint
                    return checkpoint(_forward,*args,use_reentrant=False,**kwargs)
                return _forward(*args,**kwargs)
            layer.forward=checkpointed
            self.original_encoder_forwards.append(original)
        atomic(self.root/'environment.json',{'environment':environment(),'model':self.b.metadata(),
            'backbone_hash':self.initial_hash,'source_commit':self.plan['source_commit']})

    def tick(self,**progress):
        seconds=self.offset+time.monotonic()-self.start
        if self.arm_usage is not None:
            atomic(self.arm_usage,{'allocated_seconds':self.arm_prior+time.monotonic()-self.arm_start,'utc':now(),
                'scope':'full arm fitting+normalization+validation+checkpoint I/O, includes failed/interrupted work'})
        atomic(self.usage,{'resident_seconds':seconds,'utc':now(),'pid':os.getpid()})
        atomic(self.root/'worker_heartbeat.json',{'utc':now(),'pid':os.getpid(),**progress})
        campaign=Path(self.C.get('campaign_root',self.root))
        if STOP or (self.root/'STOP').exists() or (campaign/'STOP').exists(): raise Halt('STOPPED_BY_USER')
        if campaign!=self.root and (campaign/'status.json').exists():
            current=read(campaign/'status.json').get('gpu_seconds',0)
            if current>=(self.C['resources']['gpu_hours']-self.C['resources']['reserve_gpu_hours'])*3600:
                raise Halt('STOPPED_BUDGET')
        # Controller accounts all subprocess residency including startup failures.
        if seconds>=(self.C['resources']['gpu_hours']-self.C['resources']['reserve_gpu_hours'])*3600:
            raise Halt('STOPPED_BUDGET')
        if time.time()-self.plan['created_epoch']>=self.C['resources']['wall_hours']*3600:
            raise Halt('STOPPED_WALL_BUDGET')
        import shutil
        if shutil.disk_usage(self.root).free<self.C['resources']['free_disk_min_gib']*2**30:
            raise Halt('BLOCKED_DISK')

    def cache(self,row):
        import fcntl
        p=self.cache_root/(row['utterance_id']+'.pt')
        # Immutable completed cache files can be read concurrently. Missing
        # entries have a per-utterance process lock around atomic creation.
        if p.exists(): return self._cache(row)
        with p.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            try: return self._cache(row)
            finally: fcntl.flock(handle,fcntl.LOCK_UN)

    def _cache(self,row):
        self.tick(utterance=row['utterance_id'])
        p=self.cache_root/(row['utterance_id']+'.pt')
        if p.exists():
            d=torch.load(p,map_location='cpu',weights_only=True)
            if d['fingerprint']!=self.fp or d['audio_sha256']!=row['audio_full_sha256']: raise ValueError('Stale cache')
            return d
        if any(p.requires_grad for p in self.b.model.parameters()):
            raise ValueError('Frozen cache creation attempted with installed weight adaptation')
        inp=batch_model_inputs(self.b,[row['audio_path']])
        with torch.no_grad():
            enc=self.b.model.model.encoder(inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
            sot=self.b.model.generation_config.decoder_start_token_id
            z=self.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                decoder_input_ids=torch.tensor([[sot]],device='cuda'),use_cache=False).logits[0,-1].float()
            langs=self.b.model.generation_config.lang_to_id; ids=sorted(langs.values()); tok=ids[int(z[ids].argmax())]
        g=self.b.model.generation_config
        target=self.b.processor.tokenizer.encode(target_text(row['transcript_raw']),add_special_tokens=False)+[self.eos]
        prompt=[sot,tok,g.task_to_id['transcribe'],self.b.processor.tokenizer.convert_tokens_to_ids('<|notimestamps|>')]
        valid=self.b.valid_frames(row['duration_sec'])
        d={'enc':enc.cpu(),'features':inp['input_features'].cpu(),'mask':inp['attention_mask'].cpu(),
            'valid':valid,'prompt':prompt,'targets':target,'language':next(k.strip('<|>') for k,v in langs.items() if v==tok),
            'fingerprint':self.fp,'audio_sha256':row['audio_full_sha256']}
        save_torch(p,d); return d

    def factory(self,method,seed=42,gate=None):
        torch.manual_seed(seed)
        if method in ['LP','LS']: return EncoderWeightLoRA(self.b.model,method=='LS',seed).cuda()
        g=Allocation(gate,round(self.C['gate']['block_seconds']/self.b.encoder_step_sec),
            self.C['gate']['epsilon'],self.C['gate']['alpha']) if gate else None
        return RepresentationAdapter('ereft' if method=='R' else 'lra',seed=seed,gate=g).cuda()

    def state(self,ad):
        return ad.adaptation_state() if isinstance(ad,EncoderWeightLoRA) else {k:v.detach().cpu() for k,v in ad.state_dict().items()}

    def load(self,ad,state):
        if isinstance(ad,EncoderWeightLoRA): ad.load_adaptation(state)
        else: ad.load_state_dict(state)

    def encoder(self,ad,d,grad=True):
        if isinstance(ad,EncoderWeightLoRA):
            with (contextlib.nullcontext() if ad.installed else ad.active()):
                features=d['features'].to('cuda')
                if grad: features.requires_grad_(True)
                enc=self.b.model.model.encoder(features,attention_mask=d['mask'].to('cuda')).last_hidden_state
            return enc
        enc=d['enc'].to('cuda')
        if ad is not None:
            ad.lengths=[d['valid']]; enc=ad(enc)
        return enc

    def forward(self,ad,row,hidden=False,prefix=False):
        d=self.cache(row); X,Y=build_tf_batch([d['prompt']],[d['targets']],self.eos,self.eos)
        enc=self.encoder(ad,d)
        if prefix: enc=torch.zeros_like(enc)
        o=self.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),decoder_input_ids=X.cuda(),
            attention_mask=d['mask'].cuda(),use_cache=False,output_hidden_states=hidden)
        return o,Y.cuda(),d

    def weights(self,row,d,Y,omega):
        # Special/control prompt positions are excluded by Y. Embedded-English
        # proxy only for genuine-CS utterances; EN-only tokens have weight1.
        groups=torch.full_like(Y,-100)
        g=token_groups(self.b.processor.tokenizer,d['targets']).to(Y.device)
        if row['stratum']=='CS': groups[Y!=-100]=g
        return groups

    def decode(self,ad,row):
        d=self.cache(row)
        with torch.no_grad():
            torch.cuda.synchronize(); start=time.monotonic()
            # Timed model inference includes frozen encoder + Auto-LID + gate/
            # adapted encoder + beam search. Disk/audio preprocessing is logged
            # separately by cache construction and excluded from this model RTF.
            frozen=self.b.model.model.encoder(d['features'].cuda(),attention_mask=d['mask'].cuda()).last_hidden_state
            sot=self.b.model.generation_config.decoder_start_token_id
            z=self.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=frozen),
                decoder_input_ids=torch.tensor([[sot]],device='cuda'),use_cache=False).logits[0,-1].float()
            ids=sorted(self.b.model.generation_config.lang_to_id.values())
            if ids[int(z[ids].argmax())]!=d['prompt'][1]: raise ValueError('Frozen Auto-LID changed')
            if isinstance(ad,EncoderWeightLoRA): enc=self.encoder(ad,d,False)
            elif ad is None: enc=frozen
            else: ad.lengths=[d['valid']]; enc=ad(frozen)
            kw=dict(input_features=d['features'].cuda(),attention_mask=d['mask'].cuda(),
                encoder_outputs=BaseModelOutput(last_hidden_state=enc),language=d['language'],task='transcribe',
                return_dict_in_generate=True,**self.C['decode'])
            seq=self.b.model.generate(**kw).sequences[0].tolist(); torch.cuda.synchronize()
            if seq[:4]!=d['prompt']: raise ValueError('Decode prompt mismatch')
            p={'id':row['utterance_id'],'dialogue_id':row['dialogue_id'],'tokens':seq[4:],
                'text':self.b.processor.batch_decode([seq],skip_special_tokens=True)[0].strip(),
                'terminated':self.eos in seq[4:],'prompt':seq[:4],'audio_sha256':row['audio_full_sha256'],
                'seconds':time.monotonic()-start,'latency_scope':'encoder+frozen_AutoLID+adaptation+beam_search; audio_IO_preprocessing_excluded'}
            if isinstance(ad,RepresentationAdapter) and ad.gate is not None:
                diag=ad.gate.last
                coeff=diag['coefficients'][diag['counts']>0].cpu().tolist()
                p['allocation']={'coefficients':coeff,'energy':diag['energy'].cpu().tolist(),
                    'allocated_energy':diag['allocated_energy'].cpu().tolist(),'cast_energy':diag['cast_energy'].cpu().tolist(),
                    'zero_energy':diag['zero_energy']}
            return p

    def evaluate(self,ad,dest,checkpoint=None,population=None,baseline=None):
        dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); pop=self.V if population is None else population
        ident={'fingerprint':self.fp,'checkpoint_sha256':sha(checkpoint) if checkpoint else None,
            'population_ids':identity([r['utterance_id'] for r in pop]),'decode':self.C['decode'],
            'control':{'alpha':getattr(ad,'alpha',None),'shuffle_seed':getattr(ad,'shuffle_seed',None),
                'remove_mode':getattr(getattr(ad,'gate',None),'remove_mode',None)}}
        ip=dest/'identity.json'
        if ip.exists() and read(ip)!=ident: raise ValueError('Evaluation identity changed')
        atomic(ip,ident); predictions={}
        sealed=dest/'SEALED.json'
        if sealed.exists():
            record=read(sealed)
            if record['identity_sha256']!=sha(ip) or record['metrics_sha256']!=sha(dest/'metrics.json'):
                raise ValueError('Corrupt sealed metrics/identity')
            if set(record['predictions'])!={r['utterance_id'] for r in pop}: raise ValueError('Sealed population mismatch')
            for k,digest in record['predictions'].items():
                p=dest/'predictions'/(k+'.json')
                if not p.exists() or sha(p)!=digest: raise ValueError('Corrupt sealed prediction')
                predictions[k]=read(p)
            return read(dest/'metrics.json'),predictions
        torch.cuda.reset_peak_memory_stats()
        for row in pop:
            self.tick(evaluation=str(dest),utterance=row['utterance_id'])
            p=dest/'predictions'/(row['utterance_id']+'.json')
            if p.exists(): rec=read(p)
            else: rec=self.decode(ad,row); atomic(p,rec)
            if rec['id']!=row['utterance_id'] or rec['audio_sha256']!=row['audio_full_sha256']: raise ValueError('Corrupt prediction')
            predictions[rec['id']]=rec
        m=metrics(pop,predictions,baseline)
        coeff=[a for p in predictions.values() for a in p.get('allocation',{}).get('coefficients',[])]
        m['allocation_std']=float(np.std(coeff)) if coeff else 0.
        m.update(memory())
        atomic(dest/'metrics.json',m)
        atomic(dest/'SEALED.json',{'identity_sha256':sha(ip),'metrics_sha256':sha(dest/'metrics.json'),
            'predictions':{k:sha(dest/'predictions'/(k+'.json')) for k in predictions}})
        return m,predictions

    def train(self,method,lr,horizon,seed=42,omega=1,tag=None,warm=None,gate=None,baseline=None):
        spec={'method':method,'lr':lr,'horizon':horizon,'seed':seed,'omega':omega,'tag':tag,
            'warm_sha256':sha(warm) if warm else None,'gate':gate,'fingerprint':self.fp,
            'data':self.plan['data_sha256'],'config':self.plan['config_sha256']}
        # Horizon is part of identity: recipe trials are always trained to full
        # horizon and selected from immutable epoch outputs (equal two trials).
        dest=self.root/'runs'/identity(spec)[:20]; dest.mkdir(parents=True,exist_ok=True)
        meta=dest/'run.json'
        if meta.exists() and read(meta)!=spec: raise ValueError('Stale arm')
        atomic(meta,spec)
        self.arm_usage=dest/'usage.json'; self.arm_start=time.monotonic()
        self.arm_prior=read(self.arm_usage)['allocated_seconds'] if self.arm_usage.exists() else 0.
        ad=self.factory(method,seed,gate)
        if warm:
            state=torch.load(warm,map_location='cpu',weights_only=True)
            missing,unexpected=ad.load_state_dict(state,strict=False)
            if unexpected or any(not k.startswith('gate.') for k in missing): raise ValueError('Incompatible warm start')
        if gate:
            ad.freeze_base()
            # Normalization fit solely on T2000 valid blocks, using fixed warm
            # adapter; no held-out transcript or token labels enter the gate.
            feature_rows=[]
            with torch.no_grad():
                for row in self.T:
                    d=self.cache(row); h=d['enc'].cuda(); delta=ad.delta(h)
                    proj=torch.nn.functional.linear(h.float(),ad.reft.rot.weight if method=='R' else ad.B.T)
                    z,e,n,m=ad.gate.features(h,delta,proj,[d['valid']]); feature_rows.append(z[n>0].cpu())
            ad.gate.fit_normalization(feature_rows)
        params=[p for p in ad.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=lr,betas=self.C['training']['betas'],eps=self.C['training']['eps'],weight_decay=0)
        last=dest/'last.pt'; epoch=1; cursor=0; history=[]; steps=0; examples=0; elapsed=0.
        if last.exists():
            s=torch.load(last,map_location='cpu',weights_only=True)
            if s['identity']!=identity(spec): raise ValueError('Incompatible optimizer checkpoint')
            self.load(ad,s['adapter']); opt.load_state_dict(s['optimizer'])
            epoch,cursor,history,steps,examples,elapsed=[s[k] for k in ['epoch','cursor','history','steps','examples','elapsed']]
            torch.set_rng_state(s['rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        def save():
            save_torch(last,{'identity':identity(spec),'adapter':self.state(ad),'optimizer':opt.state_dict(),
                'epoch':epoch,'cursor':cursor,'history':history,'steps':steps,'examples':examples,'elapsed':elapsed,
                'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'scheduler':'constant','scaler':None,'sampler':{'seed':seed+epoch,'next_logical_batch':cursor}})
        train_start=time.monotonic(); elapsed_prior=elapsed
        while epoch<=horizon:
            order=np.random.default_rng(seed+epoch).permutation(len(self.T))
            batches=[order[i:i+self.C['training']['effective_batch']] for i in range(0,len(order),self.C['training']['effective_batch'])]
            torch.cuda.reset_peak_memory_stats()
            for b in range(cursor,len(batches)):
                self.tick(arm=str(dest.name),method=method,epoch=epoch,update=steps)
                idx=batches[b]; ds=[self.cache(self.T[int(i)]) for i in idx]
                den=0.
                for i,d in zip(idx,ds):
                    g=token_groups(self.b.processor.tokenizer,d['targets']) if self.T[int(i)]['stratum']=='CS' else torch.full((len(d['targets']),),-100)
                    den+=float(torch.where(g==EN,omega,1.).sum())
                opt.zero_grad(set_to_none=True); loss_value=0.
                for i in idx:
                    self.tick(arm=str(dest.name),method=method,epoch=epoch,update=steps)
                    row=self.T[int(i)]
                    # Checkpoint recomputation must see EXACTLY the installed
                    # encoder LoRA modules used by the original forward.
                    with (ad.active() if isinstance(ad,EncoderWeightLoRA) else contextlib.nullcontext()):
                        o,Y,d=self.forward(ad,row)
                        groups=self.weights(row,d,Y,omega)
                        loss=weighted_ce(o.logits,Y,groups,omega,denominator=den)
                        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite objective')
                        loss.backward(); loss_value+=float(loss.detach()); del loss,o,Y
                if any(p.grad is not None for p in self.backbone): raise ValueError('Backbone gradient leakage')
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params): raise FloatingPointError('Invalid gradients')
                grad=torch.nn.utils.clip_grad_norm_(params,self.C['training']['clip_norm'])
                if not bool(torch.isfinite(grad)): raise FloatingPointError('Nonfinite gradient norm')
                opt.step()
                if any(not bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Nonfinite parameters')
                steps+=1; examples+=len(idx); cursor=b+1; elapsed=elapsed_prior+time.monotonic()-train_start
                save()
                with (dest/'training.jsonl').open('a') as f:
                    f.write(json.dumps({'epoch':epoch,'batch':b,'step':steps,'examples':examples,'objective':loss_value,
                        'omega':omega,'gpu_seconds':elapsed,'grad_norm':float(grad),'utc':now(),**memory()})+'\n'); f.flush()
            ep=dest/f'epoch{epoch}'; ck=ep/'adapter.pt'
            if not ck.exists(): save_torch(ck,self.state(ad))
            peak=memory(); m,_=self.evaluate(ad,ep/'validation',ck,baseline=baseline)
            record={'epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'metrics':m,'steps':steps,
                'examples':examples,'seconds':elapsed,'trainable_parameters':sum(p.numel() for p in params),
                'total_adapter_gate_parameters':sum(p.numel() for p in ad.parameters() if not isinstance(ad,EncoderWeightLoRA) or p.requires_grad),
                'checkpoint_bytes':ck.stat().st_size,'training_peak':peak,'run_dir':str(dest),'spec':spec}
            if len(history)<epoch: history.append(record)
            atomic(dest/'history.json',history); epoch+=1; cursor=0; save()
        if not history: raise ValueError('No complete epoch')
        selected=min(history,key=lambda h:(h['metrics']['genuine_cs_mer'],h['metrics']['mer'],h['metrics']['zh_cer'],h['epoch']))
        atomic(dest/'selection.json',selected)
        self.tick(arm=dest.name,completed=True)
        self.arm_usage=None
        del ad,opt; torch.cuda.empty_cache()
        return selected


def baseline(R):
    return R.evaluate(None,R.root/'B0')


def n1(R):
    m,b=baseline(R); selected={}
    for method in ['R','L','LP','LS']:
        trials=[R.train(method,lr,R.C['training']['epochs'],baseline=b) for lr in R.C['training']['lr'][method]]
        selected[method]=min(trials,key=lambda h:(h['metrics']['genuine_cs_mer'],h['metrics']['mer'],h['metrics']['zh_cer'],h['spec']['lr']))
        atomic(R.root/'N1_selection.partial.json',selected)
    atomic(R.root/'N1_selection.json',selected)
    return {'selections':selected,'B0':m}


def restore(R,rec,gate=None):
    ad=R.factory(rec['spec']['method'],rec['spec']['seed'],gate if gate is not None else rec['spec'].get('gate'))
    R.load(ad,torch.load(rec['checkpoint'],map_location='cpu',weights_only=True)); return ad


def n2(R):
    sel=read(R.root/'N1_selection.json'); warm=sel['R']; b=baseline(R)[1]; ad=restore(R,warm)
    panel={'G0':warm['metrics']}; scalars=[]
    for alpha in R.C['gate']['scalar_grid']:
        ad.alpha=alpha; m,_=R.evaluate(ad,R.root/'N2'/f'scalar_{alpha}',warm['checkpoint'],baseline=b)
        scalars.append({'alpha':alpha,'metrics':m})
    winner=min(scalars,key=lambda x:(x['metrics']['genuine_cs_mer'],x['metrics']['mer'],x['alpha']))
    panel['G1']=winner['metrics']; atomic(R.root/'N2_scalar.json',winner); del ad
    recs={}
    # G8 includes the selected warm start plus exactly the same additional
    # T2000 examples/updates as frozen-adapter gates (one complete epoch).
    recs['G8']=R.train('R',warm['spec']['lr'],R.C['gate']['epochs'],warm=warm['checkpoint'],tag='G8',baseline=b)
    panel['G8']=recs['G8']['metrics']
    for kind in R.C['gate']['initial']:
        recs[kind]=R.train('R',R.C['gate']['lr'],R.C['gate']['epochs'],warm=warm['checkpoint'],gate=kind,tag=kind,baseline=b)
        panel[kind]=recs[kind]['metrics']
        atomic(R.root/'N2_panel.partial.json',panel)
    decision=pilot_decision(panel,R.C['pilot_rule'])
    atomic(R.root/'N2_panel.json',panel); atomic(R.root/'N2_selection.json',recs); atomic(R.root/'N2_decision.json',decision)
    return decision


def n3(R):
    b=baseline(R)[1]; selected={}
    for method in ['R','L','LP','LS']:
        trials=[]
        for omega in R.C['training']['weighted_omegas']:
            for lr in R.C['training']['lr'][method]:
                trials.append(R.train(method,lr,R.C['training']['epochs'],omega=omega,tag='N3_WCE',baseline=b))
        selected[method]=min(trials,key=lambda h:(h['metrics']['genuine_cs_mer'],h['metrics']['mer'],h['metrics']['zh_cer'],h['spec']['lr']))
        atomic(R.root/'N3_selection.partial.json',selected)
    # Weighted gate is diagnostic, not a pure gate-method contrast.
    if read(R.root/'N2_decision.json')['decision']=='N2_EXPLORATORY_GO':
        warm=selected['R']; selected['G5_WCE']=R.train('R',R.C['gate']['lr'],R.C['gate']['epochs'],
            omega=warm['spec']['omega'],warm=warm['checkpoint'],gate='G5',tag='N3_G5_WCE',baseline=b)
    atomic(R.root/'N3_selection.json',selected); return {'selections':selected}


def n4(R):
    b=baseline(R)[1]; N1=read(R.root/'N1_selection.json'); N2=read(R.root/'N2_selection.json')
    go=read(R.root/'N2_decision.json')['decision']=='N2_EXPLORATORY_GO'; report={}
    if go:
        warm=N1['R']
        for kind in R.C['gate']['expanded']:
            report[kind]=R.train('R',R.C['gate']['lr'],R.C['gate']['epochs'],warm=warm['checkpoint'],gate=kind,tag=kind,baseline=b)
        report['gated_LRA']=R.train('L',R.C['gate']['lr'],R.C['gate']['epochs'],warm=N1['L']['checkpoint'],gate='G5',tag='gated_LRA',baseline=b)
        ad=restore(R,N2['G5']); ad.shuffle_seed=R.C['gate']['shuffle_seed']
        report['G9']=R.evaluate(ad,R.root/'N4'/'energy_shuffle',N2['G5']['checkpoint'],baseline=b)[0]
        ad.shuffle_seed=None
        for mode in ['largest','random']:
            ad.gate.remove_mode=mode
            report['block_removal_'+mode]=R.evaluate(ad,R.root/'N4'/('block_removal_'+mode),N2['G5']['checkpoint'],baseline=b)[0]
        ad.gate=None
        report['gate_off']=R.evaluate(ad,R.root/'N4'/'gate_off',N2['G5']['checkpoint'],baseline=b)[0]
        del ad
    # Fallback D1/D2 use bounded predeclared 64 training /64 V300 utterances,
    # never C500. Readouts are proxy diagnostics, not recognition causality.
    from reft.a5000_followup.analysis import vocabulary_groups,language_lexical_report,predictive_states
    from reft.a5000.diagnostics import controlled_readout
    groups=vocabulary_groups(R.b.processor.tokenizer,R.b.model.config.vocab_size).cuda()
    ad=restore(R,N1['R']); collected=[]; d1={}
    for label,pop in [('train',R.T[:64]),('held',R.V[:64])]:
        base=[]; adapted=[]; prefix=[]; ys=[]; dialogues=[]; vals=[]
        with torch.no_grad():
            for row in pop:
                o,Y,d=R.forward(None,row,hidden=True); a,_,_=R.forward(ad,row,hidden=True); p,_,_=R.forward(None,row,hidden=True,prefix=True)
                tg=torch.full_like(Y,-100); tg[Y!=-100]=token_groups(R.b.processor.tokenizer,d['targets']).cuda()
                x,y=predictive_states(o.decoder_hidden_states[-1],Y,tg)
                base.append(x.cpu()); adapted.append(predictive_states(a.decoder_hidden_states[-1],Y,tg)[0].cpu())
                prefix.append(predictive_states(p.decoder_hidden_states[-1],Y,tg)[0].cpu()); ys.append(y.cpu())
                dialogues.append(row['dialogue_id'])
                decomp=language_lexical_report(a.logits,Y,groups)
                vals.append({k:float(decomp[k].mean()) for k in ['gold_nll','group_nll','within_group_nll']})
        collected.append((torch.cat(base),torch.cat(adapted),torch.cat(prefix),torch.cat(ys),dialogues)); d1[label]=vals
    t,h=collected
    try: probe=controlled_readout(t[0],t[1],t[3],h[0],h[1],h[3],t[2],h[2],t[4],h[4])
    except ValueError as e: probe={'status':'BLOCKED','reason':str(e)}
    report['D1']={'values':d1,'proxy':'Han/Latin/OTHER vocabulary groups'}; report['D2']=probe
    report['D3']={'status':'BLOCKED','reason':'No frozen matched-random causal intervention recipe admitted in this version'}
    atomic(R.root/'N4_analysis.json',report); return report


def n5(R):
    N1=read(R.root/'N1_selection.json'); b=baseline(R)[1]; candidates={k:v for k,v in N1.items()}
    candidates.update({k+'_WCE':v for k,v in read(R.root/'N3_selection.json').items()})
    go=read(R.root/'N2_decision.json')['decision']=='N2_EXPLORATORY_GO'
    N2=read(R.root/'N2_selection.json')
    if go: candidates.update({k:N2[k] for k in ['G3','G4','G5','G8']})
    # Fresh adapter pipelines per seed; gate replications warm-start that seed's
    # independently trained CE adapter. Horizon frozen from seed42 selection.
    for seed in [43,44]:
        independent={}
        for method in R.C['confirmation']['priority_methods']:
            old=N1[method]; independent[method]=R.train(method,old['spec']['lr'],old['epoch'],seed=seed,tag='N5',baseline=b)
            # No seed-specific epoch selection: use frozen final horizon.
            hist=read(Path(independent[method]['run_dir'])/'history.json'); independent[method]=hist[-1]
            candidates[f'{method}_seed{seed}']=independent[method]
        if go:
            for kind in ['G3','G4','G5']:
                candidates[f'{kind}_seed{seed}']=R.train('R',R.C['gate']['lr'],R.C['gate']['epochs'],seed=seed,
                    warm=independent['R']['checkpoint'],gate=kind,tag='N5_'+kind,baseline=b)
            candidates[f'G8_seed{seed}']=R.train('R',N1['R']['spec']['lr'],R.C['gate']['epochs'],seed=seed,
                warm=independent['R']['checkpoint'],tag='N5_G8',baseline=b)
        atomic(R.root/'N5_candidates.partial.json',candidates)
    freeze={'utc':now(),'data_sha256':R.plan['data_sha256'],'source_commit':R.plan['source_commit'],
        'config_sha256':R.plan['config_sha256'],'candidates':candidates,'rules':R.C['metrics'],
        'no_retuning_after_holdout':True,'official_test_authorized':False}
    p=R.root/'confirmation_freeze.json'
    if p.exists() and read(p)['candidates']!=candidates: raise ValueError('Final candidates changed after C500 freeze')
    if not p.exists(): atomic(p,freeze)
    held=rows(R.C,R.root,'A5000_C500',p)
    # Resumption continues this SAME fixed package; no extra candidate may enter.
    results={}; m,bh=R.evaluate(None,R.root/'confirmation'/'B0',population=held); results['B0']=m
    scalar=read(R.root/'N2_scalar.json')
    for name,rec in candidates.items():
        ad=restore(R,rec); results[name]=R.evaluate(ad,R.root/'confirmation'/name,rec['checkpoint'],held,bh)[0]; del ad
        atomic(R.root/'confirmation'/'results.partial.json',results)
    ad=restore(R,N1['R']); ad.alpha=scalar['alpha']
    results['best_global_scalar']=R.evaluate(ad,R.root/'confirmation'/'best_global_scalar',N1['R']['checkpoint'],held,bh)[0]
    atomic(R.root/'confirmation'/'results.json',results)
    return {'candidates':list(results),'claim':'One fixed TRAIN-derived development-holdout package; no independent final test'}


def preflight(R):
    audit={'policy':R.C['training']['weighted_policy'],'targets':0,'embedded_proxy':0,'unknown':0,'cs_utterances':0}
    for row in R.T:
        target_ids=R.b.processor.tokenizer.encode(target_text(row['transcript_raw']),add_special_tokens=False)+[R.eos]
        g=token_groups(R.b.processor.tokenizer,target_ids)
        audit['targets']+=len(g);audit['unknown']+=int((g==-100).sum())
        if row['stratum']=='CS':audit['cs_utterances']+=1;audit['embedded_proxy']+=int((g==EN).sum())
    atomic(R.root/'preflight'/'label_audit.json',audit)
    out={}; target=max(R.T,key=lambda r:r['target_tokens_incl_eos']); long=max(R.T,key=lambda r:r['duration_sec'])
    for method in ['R','L','LP','LS']:
        ad=R.factory(method); params=[p for p in ad.parameters() if p.requires_grad]
        expected={'R':5122,'L':5122,'LP':5120,'LS':1310720}[method]
        if sum(p.numel() for p in params)!=expected: raise ValueError('Parameter budget mismatch')
        opt=torch.optim.AdamW(params,lr=.0005,weight_decay=0)
        start=time.monotonic(); times=[]; torch.cuda.reset_peak_memory_stats()
        for row in [R.T[0],target,long]:
            t=time.monotonic(); opt.zero_grad(set_to_none=True)
            with (ad.active() if isinstance(ad,EncoderWeightLoRA) else contextlib.nullcontext()):
                o,Y,d=R.forward(ad,row); loss=weighted_ce(o.logits,Y); loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params): raise FloatingPointError('Preflight gradients')
            if any(p.grad is not None for p in R.backbone): raise ValueError('Backbone trainability')
            opt.step(); torch.cuda.synchronize(); times.append(time.monotonic()-t); del o,loss
        ck=R.root/'preflight'/f'{method}.pt'; save_torch(ck,R.state(ad)); copy=R.factory(method); R.load(copy,torch.load(ck,weights_only=True))
        with torch.no_grad():
            a,_,_=R.forward(ad,R.T[0]); b,_,_=R.forward(copy,R.T[0]); assert torch.equal(a.logits,b.logits)
        del copy,a,b
        decoded=R.decode(ad,R.T[0])
        out[method]={'parameters':expected,'forward_backward_seconds':times,'seconds':time.monotonic()-start,
            'decoded_terminated':decoded['terminated'],'checkpoint_sha256':sha(ck),**memory()}
        if out[method]['reserved_peak_bytes']>torch.cuda.get_device_properties(0).total_memory*R.C['resources']['peak_reserved_fraction']:
            raise RuntimeError('VRAM admission threshold exceeded')
        if method=='R':
            d=R.cache(R.T[0]); ad.lengths=[d['valid']]
            with torch.no_grad(),ad.active(R.b.model):
                native=R.b.model.model.encoder(d['features'].cuda(),attention_mask=d['mask'].cuda()).last_hidden_state
            with torch.no_grad(): cached=R.encoder(ad,d)
            if not torch.equal(native,cached): raise ValueError('Native/cached post-LN interface mismatch')
            gate=Allocation().cuda(); ad.gate=gate; ad.freeze_base(); ad.lengths=[R.cache(R.T[0])['valid']]
            o,Y,d=R.forward(ad,R.T[0]); loss=weighted_ce(o.logits,Y); loss.backward()
            if gate.net[-1].weight.grad is None or not bool(gate.net[-1].weight.grad.norm()>0): raise ValueError('No warm gate gradient')
            del o,loss
        del ad,opt; torch.cuda.empty_cache()
    if weights_hash(R.b.model)!=R.initial_hash: raise ValueError('Backbone mutated')
    # Actual beam5 Auto-LID native baseline versus cached frozen-LID decoding.
    from reft.baseline_auto import generate
    native,_=generate(R.b,R.T[0]['audio_path'],None,5)
    cached=R.decode(None,R.T[0]); assert native['tokens']==cached['tokens']
    sec=max(max(v['forward_backward_seconds']) for v in out.values())
    # Conservative bound includes all full-horizon LR trials, N2, and beam5 V300
    # evaluation every epoch; N1/N2 must fit before admitting a long campaign.
    decode_seconds=cached['seconds']; n=len(R.T); v=len(R.V)
    projected_n1=(4*2*3*n*sec + (4*2*3+1)*v*decode_seconds)/3600
    projected_n2=(4*n*sec + 10*v*decode_seconds)/3600
    report={'passed':True,'methods':out,'source_commit':R.plan['source_commit'],
        'data_sha256':R.plan['data_sha256'],'config_sha256':R.plan['config_sha256'],
        'model_sha256':R.C['model_sha256'],'projected_n1_hours':projected_n1,'projected_n2_hours':projected_n2,
        'training_sec_per_example_upper':sec,'decode_seconds_per_example':decode_seconds,
        'initial_free_bytes':R.initial_free,'longest_duration':long['duration_sec'],
        'longest_targets':target['target_tokens_incl_eos'],'backbone_hash_unchanged':True}
    if projected_n1+projected_n2>R.C['resources']['gpu_hours']-R.C['resources']['reserve_gpu_hours']:
        report['passed']=False; report['blocker']='Measured conservative N1/N2 cannot fit budget'
    atomic(R.root/'preflight'/'report.json',report); return report


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','N1','N2','N3','N4','N5']);p.add_argument('--root',required=True)
    args=p.parse_args(); signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    R=None
    try:
        R=Runtime(args.root); result=preflight(R) if args.stage=='preflight' else globals()[args.stage.lower()](R)
        if R.initial_hash!=weights_hash(R.b.model): raise ValueError('Frozen model changed')
        atomic(Path(args.root)/'stage_results'/(args.stage+'.json'),{'status':'COMPLETE','result':result,'finished':now()})
    except Halt as e:
        atomic(Path(args.root)/'stage_results'/(args.stage+'.json'),{'status':'BLOCKED','reason':str(e),'finished':now()}); sys.exit(75)
    except torch.cuda.OutOfMemoryError as e:
        atomic(Path(args.root)/'stage_results'/(args.stage+'.json'),{'status':'BLOCKED','reason':'CUDA_OOM','error':str(e)}); raise
    except Exception as e:
        atomic(Path(args.root)/'stage_results'/(args.stage+'.json'),{'status':'FAILED','reason':type(e).__name__,'error':str(e)}); raise
    finally:
        if R is not None:
            if R.arm_usage is not None:
                atomic(R.arm_usage,{'allocated_seconds':R.arm_prior+time.monotonic()-R.arm_start,'utc':now(),
                    'scope':'full arm including failed/interrupted work'})
            atomic(R.usage,{'resident_seconds':R.offset+time.monotonic()-R.start,'utc':now()})

if __name__=='__main__': main()
