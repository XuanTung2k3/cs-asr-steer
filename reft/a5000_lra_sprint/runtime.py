"""Read-only parent-artifact consumer; new checkpoints, losses and evaluations."""
import time
from pathlib import Path
import numpy as np
import torch
from reft.a5000_campaign.state import read,atomic,sha,identity,now
from reft.a5000_campaign.worker import Runtime,Halt
from reft.a5000.runner import save_torch,memory
from reft.a5000.objectives import weighted_ce
from .method import SprintLRA,LRAAllocation


def development_roles(doc):
    """Metadata-only integrity audit; never load held-out transcript/audio."""
    roles=doc['roles'];seen={key:set() for key in ['utterance_id','audio_full_sha256','waveform_sha256','speaker_id','dialogue_id']}
    for role,size in [('A5000_T2000',2000),('A5000_V300',300),('A5000_C500',500)]:
        rows=roles[role]
        if len(rows)!=size or any(r['official_split']!='train' for r in rows):raise ValueError('Invalid TRAIN membership')
        for key in seen:
            values=[r[key] for r in rows];current=set(values)
            if key in ['utterance_id','audio_full_sha256','waveform_sha256'] and len(current)!=len(values):
                raise ValueError('Within-role duplicate '+key)
            if current&seen[key]:raise ValueError('Across-role leakage '+key)
            seen[key]|=current
    # No C500 rows returned to any audio or reference loader.
    return {role:roles[role] for role in ['A5000_T2000','A5000_V300']}


class SprintRuntime(Runtime):
    def __init__(self,work):
        import hashlib
        import pyarrow.parquet as pq
        from reft.text import target_text
        from csasr.models.whisper import load_whisper
        from reft.provenance import environment,weights_hash
        self.root=Path(work);self.plan=read(self.root/'frozen_plan.json');self.C=self.plan['config']
        self.campaign=Path(self.C['campaign_root']);self.parent=Path(self.plan['spec']['parent_root']);self.spec=self.plan['spec']
        self.start=time.monotonic();self.offset=0.;self.arm_usage=None;self.usage=self.root/'worker_usage.json'
        self.arm_prior=0.;self.arm_start=None
        if self.usage.exists():self.offset=read(self.usage)['resident_seconds']
        if sha(self.C['original_manifest'])!=self.C['original_manifest_sha256']:raise ValueError('Source manifest changed')
        doc=read(self.parent/'data_manifest.json');self.T=[];self.V=[]
        wanted=development_roles(doc)
        ids=[r['utterance_id'] for rows in wanted.values() for r in rows]
        table=pq.read_table(self.C['original_manifest'],columns=['utterance_id','transcript_raw','official_split','audio_path'],
            filters=[('utterance_id','in',ids)])
        lookup={r['utterance_id']:r for r in table.to_pylist()}
        if set(lookup)!=set(ids):raise ValueError('Development membership mismatch')
        for role,dest in [('A5000_T2000',self.T),('A5000_V300',self.V)]:
            for ident in wanted[role]:
                r=lookup[ident['utterance_id']]|ident
                if r['official_split']!='train' or sha(r['audio_path'])!=r['audio_full_sha256']:raise ValueError('Audio/source identity mismatch')
                if hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest()!=r['target_text_sha256']:raise ValueError('Target mismatch')
                dest.append(r)
        self.tick();torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.init();self.initial_free=torch.cuda.mem_get_info()[0]
        self.b=load_whisper({'model':{'id':self.C['model_root'],'dtype':'bfloat16','device':'cuda',
            'local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
        self.b.model.eval();self.backbone=list(self.b.model.parameters());self.initial_hash=weights_hash(self.b.model)
        if self.b.revision!='sha256:'+self.C['model_sha256'] or self.b.d_model!=1280 or self.b.encoder_step_sec!=.02:
            raise ValueError('Model identity/geometry mismatch')
        self.eos=self.b.processor.tokenizer.eos_token_id
        self.cache_producer_fp=identity({'model':self.C['model_sha256'],'processor':sha(Path(self.C['model_root'])/'tokenizer_config.json'),
            'data':self.plan['data_sha256'],'source':self.spec['parent_source']})
        self.fp=identity({'cache_producer':self.cache_producer_fp,'sprint_source':self.plan['source_commit'],
            'sprint_config':self.plan['config_sha256']})
        atomic(self.root/'environment.json',{'environment':environment(),'backbone_hash':self.initial_hash,
            'source_commit':self.plan['source_commit'],'cache_producer_source':self.spec['parent_source'],
            'roles_opened':['A5000_T2000','A5000_V300'],'C500_opened':False})

    def tick(self,**progress):
        super().tick(**progress)
        clock=self.campaign/'run_clock.json'
        if clock.exists() and time.time()-read(clock)['started_epoch']>=self.spec['timing']['soft_stop_seconds']:
            raise Halt('SPRINT_DEADLINE_SAFE_BOUNDARY')

    def cache(self,row):
        self.tick(utterance=row['utterance_id'])
        p=self.parent/'encoder_cache'/(row['utterance_id']+'.pt')
        if not p.is_file():raise FileNotFoundError('Verified parent cache required; never regenerate in parent')
        d=torch.load(p,map_location='cpu',weights_only=True)
        if d['fingerprint']!=self.cache_producer_fp or d['audio_sha256']!=row['audio_full_sha256']:
            raise ValueError('Parent cache identity mismatch')
        if d['enc'].shape!=(1,1500,1280) or len(d['targets'])!=row['target_tokens_incl_eos']:
            raise ValueError('Parent cache dimensions/targets mismatch')
        return d

    def factory(self,method='L',seed=42,gate=None):
        if method!='L':return super().factory(method,seed,gate)
        torch.manual_seed(seed)
        g=LRAAllocation(gate,self.spec['gate']['block_frames'],self.spec['gate']['epsilon']) if gate else None
        return SprintLRA(seed=seed,gate=g).cuda()

    def warm(self,gate=None):
        rec=read(self.parent/'N1_selection.json')['L']
        if sha(rec['checkpoint'])!=self.spec['warm_checkpoint_sha256']:raise ValueError('Warm checkpoint changed')
        ad=self.factory('L',gate=gate)
        ad.load_warm(torch.load(rec['checkpoint'],map_location='cpu',weights_only=True))
        if gate:ad.freeze_base()
        return ad,rec

    def baseline(self):
        return {p.stem:read(p) for p in (self.parent/'B0/predictions').glob('*.json')}

    def fit(self,arm):
        if arm not in ['L0','L2','L3','L4']:raise ValueError('Not a training arm')
        T=self.spec['training'];dest=self.root/'training';dest.mkdir(exist_ok=True)
        gate=None if arm=='L0' else arm;ad,warm=self.warm(gate)
        normalization_start=time.monotonic()
        if gate:
            features=[]
            with torch.no_grad():
                for row in self.T:
                    d=self.cache(row);h=d['enc'].cuda();delta=ad.delta(h)
                    z,e,n,mask=ad.gate.features(h,delta,h.float()@ad.basis,[d['valid']]);features.append(z[n>0].cpu())
            ad.gate.fit_normalization(features)
            atomic(dest/'normalization.json',{'role':'A5000_T2000','mean':ad.gate.mean.cpu().tolist(),
                'std':ad.gate.std.cpu().tolist(),'n_valid_blocks':sum(len(x) for x in features),
                'seconds':time.monotonic()-normalization_start,'effective_rank':ad.basis.shape[1]})
        params=[p for p in ad.parameters() if p.requires_grad]
        expected=5122 if arm=='L0' else 289
        if sum(p.numel() for p in params)!=expected:raise ValueError('Unexpected parameter budget')
        opt=torch.optim.AdamW(params,lr=T['lr'],betas=T['betas'],eps=T['eps'],weight_decay=0)
        ident={'arm':arm,'source':self.plan['source_commit'],'config':self.plan['config_sha256'],
            'warm_sha256':self.spec['warm_checkpoint_sha256'],'data':self.plan['data_sha256'],'training':T}
        atomic(dest/'resolved.json',ident);last=dest/'last.pt';cursor=0;fit_seconds=0.;examples=0
        if last.exists():
            old=torch.load(last,map_location='cpu',weights_only=True)
            if old['identity']!=identity(ident):raise ValueError('Incompatible optimizer resume')
            ad.load_state_dict(old['adapter']);opt.load_state_dict(old['optimizer'])
            cursor,fit_seconds,examples=old['cursor'],old['fit_seconds'],old['examples']
            torch.set_rng_state(old['rng']);torch.cuda.set_rng_state_all(old['cuda_rng'])
        order=np.random.default_rng(T['seed']+1).permutation(len(self.T))
        groups=[order[i:i+T['effective_batch']] for i in range(0,len(order),T['effective_batch'])]
        if len(groups)!=T['updates']:raise ValueError('Declared step budget mismatch')
        torch.cuda.reset_peak_memory_stats()
        for batch in range(cursor,len(groups)):
            self.tick(arm=arm,update=batch,examples=examples,phase='training')
            torch.cuda.synchronize();start=time.monotonic();idx=groups[batch]
            denominator=sum(self.T[int(i)]['target_tokens_incl_eos'] for i in idx)
            opt.zero_grad(set_to_none=True);value=0.
            for i in idx:
                row=self.T[int(i)];self.tick(arm=arm,update=batch,phase='training')
                o,Y,d=self.forward(ad,row);loss=weighted_ce(o.logits,Y,denominator=denominator)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite CE')
                loss.backward();value+=float(loss.detach());del o,Y,loss
            if any(p.grad is not None for p in self.backbone):raise ValueError('Backbone gradients')
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('Invalid gradient')
            grad=torch.nn.utils.clip_grad_norm_(params,T['clip_norm'])
            if not bool(torch.isfinite(grad)):raise FloatingPointError('Invalid gradient norm')
            opt.step();torch.cuda.synchronize();fit_seconds+=time.monotonic()-start
            if any(not bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError('Invalid adaptation parameters')
            cursor=batch+1;examples+=len(idx)
            # No signal-based forced termination: parent requests STOP and tick
            # raises only at a boundary outside this atomic checkpoint write.
            atomic(self.root/'activity.json',{'phase':'checkpoint_write','cursor':cursor})
            save_torch(last,{'identity':identity(ident),'adapter':self.state(ad),'optimizer':opt.state_dict(),
                'cursor':cursor,'examples':examples,'fit_seconds':fit_seconds,'rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'scheduler':'constant','scaler':None,
                'sampler':{'seed':T['seed']+1,'next_batch':cursor}})
            atomic(self.root/'activity.json',{'phase':'safe_boundary','cursor':cursor})
            with (dest/'training.jsonl').open('a') as f:
                import json
                f.write(json.dumps({'update':cursor,'examples':examples,'token_CE':value,'gradient_norm':float(grad),
                    'fit_seconds':fit_seconds,'utc':now(),**memory()})+'\n');f.flush()
        checkpoint=dest/'adapter.pt'
        if not checkpoint.exists():save_torch(checkpoint,self.state(ad))
        train_peak=memory();m,_=self.evaluate(ad,self.root/'validation',checkpoint,baseline=self.baseline())
        result={'arm':arm,'metrics':m,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'checkpoint_bytes':checkpoint.stat().st_size,'trainable_parameters':expected,
            'total_adapter_parameters':5122+(289 if gate else 0),'fit_seconds':fit_seconds,'examples':examples,
            'updates':cursor,'training_peak':train_peak,'warm_checkpoint_sha256':self.spec['warm_checkpoint_sha256'],
            'basis_rank':ad.basis.shape[1],'basis_rank_tolerance':{'rtol':1e-6,'atol':1e-10}}
        atomic(self.root/'complete_training.json',result);return result
