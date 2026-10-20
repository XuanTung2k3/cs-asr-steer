"""Durable, bounded A5000 O1 experiment. Freeze first; all results refer to the committed source and IDs."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from reft.a5000.adapters import EncoderAdapter
from reft.a5000.data import ROOT, CONF, config, runtime, output, sha, atomic, load_rows, integrity
from reft.a5000.objectives import weighted_ce
from reft.auto_policy import build_tf_batch
from reft.common import fillers
from reft.provenance import environment, weights_hash
from reft.text import target_text, one, ratios, compare, loops, en_error_kind, utterance_stratum
from csasr.models.whisper import load_whisper, batch_model_inputs

STOP = False


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_files():
    paths = git('ls-files','reft/a5000','tests/reft/a5000','configs/reft/a5000_e1.yaml',
        'configs/reft/a5000_manifests','configs/a5000/runtime.yaml','scripts/a5000/e1.sh',
        'reft/adapters.py','reft/auto_policy.py','reft/text.py','src/csasr/models/whisper.py',
        'src/csasr/evaluation','src/csasr/data/normalize.py','src/csasr/data/language_tags.py').splitlines()
    return {p:sha(ROOT/p) for p in paths}


def freeze():
    if git('status','--porcelain'):
        raise RuntimeError('Commit reviewed source/manifests before freezing')
    p=output()/'freeze.json'
    if p.exists():
        check()
        print(p)
        return
    spec=json.loads((ROOT/config()['manifest']).read_text())
    integrity(spec['roles'])
    record={'schema':'a5000_e1_freeze_v1','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_commit':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
        'sources':source_files(),'configuration':config(),'role_manifest_sha256':sha(ROOT/config()['manifest']),
        'runtime':runtime(),'historical_replication':False,
        'authorization':'Project owner A5000 E0/E1 pilot request; two O1 arms, seed42, <=6 aggregate GPU-hours'}
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(record,f,indent=1,sort_keys=True)
    print(p)


def check():
    p=output()/'freeze.json'
    d=json.loads(p.read_text())
    assert d['sources']==source_files(),'Frozen source changed; preserve results and make explicit amendment before running'
    assert d['role_manifest_sha256']==sha(ROOT/config()['manifest'])
    return d


def save_torch(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


class Paused(Exception):pass


class Budget:
    def __init__(self,stage):
        self.path=output()/'budget.json'
        self.record=json.loads(self.path.read_text()) if self.path.exists() else {'seconds':0.,'sessions':[]}
        self.prior=self.record['seconds'];self.start=time.monotonic();self.stage=stage
        self.limit=config()['budget']['gpu_hours']*3600
    def tick(self):
        elapsed=time.monotonic()-self.start
        self.record['seconds']=self.prior+elapsed
        atomic(self.path,self.record)
        if STOP or (output()/'STOP').exists():raise Paused('Stop requested at resumable boundary')
        if self.record['seconds']>=self.limit:raise Paused('Six-hour aggregate GPU budget reached')
        if self.stage=='preflight' and elapsed>config()['budget']['preflight_seconds']:
            raise Paused('Preflight time envelope reached')
    def close(self,status):
        elapsed=time.monotonic()-self.start
        self.record['seconds']=self.prior+elapsed
        self.record['sessions'].append({'stage':self.stage,'pid':os.getpid(),'seconds':elapsed,'status':status})
        atomic(self.path,self.record)


class Runtime:
    def __init__(self,budget):
        self.budget=budget;self.C=config();self.R=runtime();self.freeze=check()
        torch.manual_seed(self.C['seed']);torch.set_num_threads(2)
        self.b=load_whisper({'model':{'id':str(Path(self.R['model']['root']).expanduser()),
            'dtype':'bfloat16','device':'cuda','local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
        assert self.b.revision=='sha256:'+self.R['model']['sha256']
        assert next(self.b.model.parameters()).dtype==torch.bfloat16
        self.initial_weights=weights_hash(self.b.model)
        self.T=load_rows('A5000_T394');self.V=load_rows('A5000_V106')
        self.eos=self.b.processor.tokenizer.eos_token_id
        self.fingerprint=hashlib.sha256(json.dumps({'model':self.R['model'],'data':self.freeze['role_manifest_sha256'],
            'processor':sha(Path(self.R['model']['root']).expanduser()/'tokenizer_config.json')},sort_keys=True).encode()).hexdigest()
        atomic(output()/'runtime.json',{'environment':environment(),'initial_weights':self.initial_weights,
            'source_commit':self.freeze['source_commit'],'freeze_sha256':sha(output()/'freeze.json'),
            'fingerprint':self.fingerprint,'pid':os.getpid()})

    def cache(self,row):
        p=output()/'encoder_cache'/f"{row['utterance_id']}.pt"
        if p.exists():
            d=torch.load(p,map_location='cpu',weights_only=True)
            assert d['fingerprint']==self.fingerprint and d['audio_sha256']==row['audio_full_sha256']
            return d
        self.budget.tick()
        inp=batch_model_inputs(self.b,[row['audio_path']])
        with torch.no_grad():
            enc=self.b.model.model.encoder(inp['input_features'],attention_mask=inp['attention_mask']).last_hidden_state
            sot=self.b.model.generation_config.decoder_start_token_id
            z=self.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),
                decoder_input_ids=torch.tensor([[sot]],device='cuda'),use_cache=False).logits[0,-1].float()
            langs=self.b.model.generation_config.lang_to_id
            ids=sorted(langs.values()); tok=ids[int(z[ids].argmax())]
        g=self.b.model.generation_config
        targets=self.b.processor.tokenizer.encode(target_text(row['transcript_raw']),add_special_tokens=False)+[self.eos]
        prompt=[sot,tok,g.task_to_id['transcribe'],self.b.processor.tokenizer.convert_tokens_to_ids('<|notimestamps|>')]
        d={'enc':enc.cpu(),'features':inp['input_features'].cpu(),'mask':inp['attention_mask'].cpu(),
           'prompt':prompt,'targets':targets,'language':next(k.strip('<|>') for k,v in langs.items() if v==tok),
           'fingerprint':self.fingerprint,'audio_sha256':row['audio_full_sha256']}
        save_torch(p,d)
        return d

    def inputs(self,row):
        d=self.cache(row)
        enc=d['enc'].to('cuda');X,Y=build_tf_batch([d['prompt']],[d['targets']],self.eos,self.eos)
        return d,enc,X.to('cuda'),Y.to('cuda')

    def forward(self,ad,row):
        d,enc,X,Y=self.inputs(row)
        enc=ad(enc) if ad is not None else enc
        z=self.b.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc),decoder_input_ids=X,
                       attention_mask=d['mask'].to('cuda'),use_cache=False).logits
        return z,Y

    def decode(self,ad,row,native=False):
        d=self.cache(row)
        kw=dict(input_features=d['features'].to('cuda'),attention_mask=d['mask'].to('cuda'),
            language=d['language'],task='transcribe',return_dict_in_generate=True,**self.C['decode'])
        if native:
            ctx=ad.active(self.b.model) if ad is not None else contextlib.nullcontext()
        else:
            enc=d['enc'].to('cuda')
            enc=ad(enc) if ad is not None else enc
            kw['encoder_outputs']=BaseModelOutput(last_hidden_state=enc)
            ctx=contextlib.nullcontext()
        torch.cuda.synchronize();t=time.monotonic()
        with torch.inference_mode(),ctx:
            seq=self.b.model.generate(**kw).sequences[0].tolist()
        torch.cuda.synchronize()
        assert seq[:4]==d['prompt'],(seq[:4],d['prompt'])
        text=self.b.processor.batch_decode([seq],skip_special_tokens=True)[0].strip()
        return {'id':row['utterance_id'],'dialogue_id':row['dialogue_id'],'tokens':seq[4:],
            'text':text,'prompt':seq[:4],'terminated':self.eos in seq[4:],
            'seconds':time.monotonic()-t,'audio_sha256':row['audio_full_sha256']}


def memory():
    torch.cuda.synchronize()
    return {'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved()}


def preflight(R):
    dest=output()/'preflight'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'report.json').exists():
        assert json.loads((dest/'report.json').read_text())['passed'];return
    short=next(r for r in R.T if r['stratum']=='CS')
    long=max(R.T,key=lambda r:r['target_tokens_incl_eos'])
    report={'source_commit':R.freeze['source_commit'],'freeze_sha256':sha(output()/'freeze.json'),'methods':{}}
    for method in ['ereft','lra']:
        R.budget.tick();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        ad=EncoderAdapter(method).to('cuda');assert ad.count()==5122
        assert not any(p.requires_grad for p in R.b.model.parameters())
        with torch.no_grad():
            plain,Y=R.forward(None,short);identity,_=R.forward(ad,short)
            assert torch.equal(plain,identity)
        del plain,identity
        opt=torch.optim.AdamW(ad.parameters(),lr=0.0005,betas=(.9,.99),eps=1e-6,weight_decay=0)
        losses=[]
        for row in [short,long]:
            opt.zero_grad(set_to_none=True)
            z,Y=R.forward(ad,row);loss=weighted_ce(z,Y);assert torch.isfinite(loss) and torch.isfinite(z).all()
            loss.backward()
            assert all(p.grad is None for p in R.b.model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ad.parameters())
            # LRA B can have zero first gradient; after the legitimate first step it must receive signal.
            if losses or method=='ereft':assert all(p.grad.norm()>0 for p in ad.parameters())
            else:assert ad.A.grad.norm()>0 and ad.c.grad.norm()>0
            opt.step();assert all(torch.isfinite(p).all() for p in ad.parameters())
            losses.append(float(loss.detach()));del z,loss
        path=dest/f'{method}.pt';save_torch(path,{k:v.cpu() for k,v in ad.state_dict().items()})
        copy=EncoderAdapter(method).to('cuda');copy.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():
            a,Y=R.forward(ad,short);b,_=R.forward(copy,short);assert torch.equal(a,b)
        del a,b,copy
        # Same adapted interface under native encoder hook and precomputed frozen encoder route.
        native=R.decode(ad,short,native=True);cached=R.decode(ad,short)
        assert native['tokens']==cached['tokens']
        with torch.no_grad():
            d,enc,X,Y=R.inputs(short)
            assert torch.equal(ad(enc,0),enc)
        report['methods'][method]={'passed':True,'losses':losses,'params':ad.count(),
            'seconds':time.monotonic()-start,'max_target_tokens':long['target_tokens_incl_eos'],
            'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':sha(path),**memory()}
        del ad,opt;torch.cuda.empty_cache()
    # Native Auto-LID agrees with frozen-baseline cached policy on the preflight audio.
    from reft.baseline_auto import generate
    native,_=generate(R.b,short['audio_path'],None,5)
    cached=R.decode(None,short)
    assert native['tokens']==cached['tokens']
    assert weights_hash(R.b.model)==R.initial_weights
    report['passed']=True;report['frozen_weights_unchanged']=True
    atomic(dest/'report.json',report)
    print(json.dumps(report),flush=True)


def score(rows,predictions,baseline=None):
    total=np.zeros(11,dtype=np.int64);cs=np.zeros(11,dtype=np.int64)
    per=[];en_kinds={};correction=corruption=new_zh=zh_repairs=0;nonterm=loopcount=0
    for row in rows:
        p=predictions[row['utterance_id']];o=one(row['transcript_raw'],p['text'])
        c=np.array(o['counts']);total+=c
        if row['stratum']=='CS':cs+=c
        for i,tag in enumerate(o['tags']):
            if tag=='EN' and o['ref'][i] not in fillers() and o['by'][i][0]!='match':
                kind=en_error_kind(o,i);en_kinds[kind]=en_kinds.get(kind,0)+1
        nonterm+=not p['terminated'];loopcount+=loops(o['hyp'])>=15
        r={'id':row['utterance_id'],'dialogue':row['dialogue_id'],'counts':c.tolist(),'stratum':row['stratum']}
        if baseline is not None:
            base=one(row['transcript_raw'],baseline[row['utterance_id']]['text'])
            delta=compare(base,o,row['utterance_id'],row['dialogue_id'],False)
            correction+=sum(e['type']=='correction' for e in delta['events'])
            corruption+=sum(e['type']=='corruption' for e in delta['events'])
            new_zh+=int(delta['new_zh']);zh_repairs+=int(delta['zh_repairs'])
            r['delta']=delta
        per.append(r)
    return {**ratios(total),'genuine_cs_mer':ratios(cs)['mer'],'counts':total.tolist(),
        'cs_counts':cs.tolist(),'n':len(rows),'dialogues':len({r['dialogue_id'] for r in rows}),
        'english_error_kinds':en_kinds,'english_corrections':correction,'english_corruptions':corruption,
        'new_mandarin_errors':new_zh,'mandarin_repairs':zh_repairs,'insertions':int(total[10]),
        'nontermination':int(nonterm),'loops':int(loopcount),'per_utterance':per,
        'seconds':sum(p['seconds'] for p in predictions.values()),
        'pier_definition':'canonical script-tagged EN reference-unit error rate; not manually annotated linguistic PIER'}


def evaluate(R,ad,dest,checkpoint=None,baseline=None):
    R.budget.tick();dest.mkdir(parents=True,exist_ok=True)
    expected={'fingerprint':R.fingerprint,'source_commit':R.freeze['source_commit'],
              'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'decode':R.C['decode'],
              'role':'A5000_V106','role_manifest_sha256':R.freeze['role_manifest_sha256']}
    manifest=dest/'decode_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==expected
    else:atomic(manifest,expected)
    pp=dest/'predictions';pp.mkdir(exist_ok=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    preds={}
    for row in R.V:
        R.budget.tick();p=pp/f"{row['utterance_id']}.json"
        if p.exists():d=json.loads(p.read_text())
        else:
            d=R.decode(ad,row);atomic(p,d)
        assert d['id']==row['utterance_id'] and d['audio_sha256']==row['audio_full_sha256']
        preds[d['id']]=d
    if not (dest/'SEALED.json').exists():
        metrics=score(R.V,preds,baseline);metrics.update(memory())
        atomic(dest/'metrics.json',metrics)
        blob=''.join(json.dumps(preds[r['utterance_id']],ensure_ascii=False,sort_keys=True)+'\n' for r in R.V)
        temp=dest/'predictions.jsonl.tmp';temp.write_text(blob);temp.replace(dest/'predictions.jsonl')
        atomic(dest/'SEALED.json',{'predictions_sha256':sha(dest/'predictions.jsonl'),
             'metrics_sha256':sha(dest/'metrics.json'),'decode_manifest_sha256':sha(manifest)})
    else:
        seal=json.loads((dest/'SEALED.json').read_text())
        assert sha(dest/'metrics.json')==seal['metrics_sha256'] and sha(dest/'predictions.jsonl')==seal['predictions_sha256']
    return json.loads((dest/'metrics.json').read_text()),preds


def selection_key(m,epoch):return (m['genuine_cs_mer'],m['mer'],m['zh_cer'],epoch)


def checkpoint_state(ad,opt,epoch,next_batch,nll,tokens,history):
    return {'adapter':{k:v.detach().cpu() for k,v in ad.state_dict().items()},'optimizer':opt.state_dict(),
        'epoch':epoch,'next_batch':next_batch,'nll':nll,'tokens':tokens,'history':history,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}


def train_trial(R,method,lr_index,horizon,baseline):
    lr=float(R.C['training']['learning_rates'][method][lr_index]);T=R.C['training']
    dest=output()/'arms'/method/f'lr{lr_index}';dest.mkdir(parents=True,exist_ok=True)
    meta={'method':method,'lr':lr,'source_commit':R.freeze['source_commit'],'freeze_sha256':sha(output()/'freeze.json')}
    if (dest/'run.json').exists():assert json.loads((dest/'run.json').read_text())==meta
    else:atomic(dest/'run.json',meta)
    torch.manual_seed(R.C['seed']);ad=EncoderAdapter(method).to('cuda');assert ad.count()==5122
    opt=torch.optim.AdamW(ad.parameters(),lr=lr,betas=T['betas'],eps=T['eps'],weight_decay=0)
    last=dest/'last.pt';epoch,next_batch,nll,tokens,history=1,0,0.,0,[]
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True)
        ad.load_state_dict(state['adapter']);opt.load_state_dict(state['optimizer'])
        epoch,next_batch,nll,tokens,history=[state[k] for k in ['epoch','next_batch','nll','tokens','history']]
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    while epoch<=horizon:
        R.budget.tick();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        order=np.random.default_rng(R.C['seed']+epoch).permutation(len(R.T))
        groups=[order[i:i+T['effective_batch']] for i in range(0,len(order),T['effective_batch'])]
        for batch in range(next_batch,len(groups)):
            R.budget.tick();idx=groups[batch]
            denominator=sum(R.T[int(i)]['target_tokens_incl_eos'] for i in idx)
            opt.zero_grad(set_to_none=True);batch_nll=0.
            for i in idx:
                R.budget.tick()
                z,Y=R.forward(ad,R.T[int(i)])
                loss=weighted_ce(z,Y,denominator=denominator)
                assert torch.isfinite(loss),'Nonfinite pure CE'
                loss.backward();batch_nll+=float(loss.detach())*denominator
                del z,loss,Y
            assert all(p.grad is None for p in R.b.model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ad.parameters())
            grad=float(torch.nn.utils.clip_grad_norm_(ad.parameters(),T['clip_norm']))
            assert np.isfinite(grad)
            opt.step();assert all(torch.isfinite(p).all() for p in ad.parameters())
            nll+=batch_nll;tokens+=denominator
            entry={'method':method,'lr':lr,'epoch':epoch,'batch':batch,'token_ce':batch_nll/denominator,
                'tokens':denominator,'grad_norm':grad,'seconds':time.monotonic()-start,**memory()}
            with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
            save_torch(last,checkpoint_state(ad,opt,epoch,batch+1,nll,tokens,history))
            print(json.dumps(entry),flush=True)
        ep=dest/f'epoch{epoch}';ep.mkdir(exist_ok=True)
        checkpoint=ep/'adapter.pt'
        if not checkpoint.exists():save_torch(checkpoint,{k:v.detach().cpu() for k,v in ad.state_dict().items()})
        train_peak=memory()
        m,_=evaluate(R,ad,ep/'validation',checkpoint,baseline)
        record={'epoch':epoch,'train_ce':nll/tokens,'checkpoint_sha256':sha(checkpoint),
            'metrics':{k:v for k,v in m.items() if k!='per_utterance'},'training_peak':train_peak,
            'checkpoint_bytes':checkpoint.stat().st_size}
        if len(history)<epoch:history.append(record)
        atomic(dest/'history.json',history)
        epoch+=1;next_batch=0;nll=0.;tokens=0
        save_torch(last,checkpoint_state(ad,opt,epoch,0,0.,0,history))
    return history


def campaign(R):
    assert json.loads((output()/'preflight/report.json').read_text())['passed']
    b0,baseline=evaluate(R,None,output()/'B0')
    atomic(output()/'status.json',{'status':'RUNNING','phase':'LR trials','pid':os.getpid()})
    # Exact equal budgets. Both trial epoch-1 results exist before either arm is continued.
    results={}
    for method in ['ereft','lra']:
        trials=[]
        for i in range(2):trials.append(train_trial(R,method,i,1,baseline))
        winner=min(range(2),key=lambda i:selection_key(trials[i][0]['metrics'],i))
        results[method]={'lr_index':winner,'lr':R.C['training']['learning_rates'][method][winner],
            'epoch1_trials':[{k:v for k,v in t[0].items() if k!='per_utterance'} for t in trials]}
    lrpath=output()/'lr_selection.json'
    if lrpath.exists():assert json.loads(lrpath.read_text())==results
    else:atomic(lrpath,results)
    selections={}
    for method in ['ereft','lra']:
        i=results[method]['lr_index']
        history=train_trial(R,method,i,R.C['training']['epochs'],baseline)
        selected=min(history,key=lambda h:selection_key(h['metrics'],h['epoch']))
        path=output()/'arms'/method/f'lr{i}'/f"epoch{selected['epoch']}"/'adapter.pt'
        selections[method]={'lr_index':i,'lr':results[method]['lr'],'epoch':selected['epoch'],
            'checkpoint':str(path),'checkpoint_sha256':sha(path),'source_commit':R.freeze['source_commit'],
            'metrics':selected['metrics'],'history':history}
    # Selection freeze is written once; confirmation remains unavailable to the initial campaign.
    if (output()/'selection.json').exists():assert json.loads((output()/'selection.json').read_text())==selections
    else:atomic(output()/'selection.json',selections)
    assert weights_hash(R.b.model)==R.initial_weights
    atomic(output()/'comparison.json',{'B0':{k:v for k,v in b0.items() if k!='per_utterance'},
        'arms':selections,'source_commit':R.freeze['source_commit'],'freeze_sha256':sha(output()/'freeze.json'),
        'claim':'Exploratory one-seed A5000-only matched O1 comparison; not historical replication or formal superiority',
        'C150_evaluated':False,'frozen_backbone_unchanged':True})
    atomic(output()/'status.json',{'status':'COMPLETE','pid':os.getpid(),'selection_sha256':sha(output()/'selection.json')})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','preflight','campaign','status'])
    args=ap.parse_args()
    if args.stage=='freeze':freeze();return
    if args.stage=='status':
        for name in ['status.json','budget.json','selection.json']:
            p=output()/name
            if p.exists():print(name,p.read_text())
        return
    output().mkdir(parents=True,exist_ok=True)
    with (output()/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output()/'STOP').exists():raise RuntimeError('STOP exists; remove explicitly when resuming')
        occupied=subprocess.check_output(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0'),
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('Selected GPU already has a compute process: '+occupied)
        def stopping(signum,frame):
            global STOP
            STOP=True
        signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
        atomic(output()/'process.json',{'pid':os.getpid(),'stage':args.stage,'started':time.time(),
            'source_commit':check()['source_commit']})
        budget=Budget(args.stage);status='FAILED'
        try:
            R=Runtime(budget)
            if args.stage=='preflight':preflight(R)
            else:campaign(R)
            status='COMPLETE'
        except Paused as e:
            status='PAUSED';atomic(output()/'status.json',{'status':status,'reason':str(e),'pid':os.getpid()})
        except Exception as e:
            atomic(output()/'status.json',{'status':'FAILED','error':repr(e),'pid':os.getpid()})
            raise
        finally:budget.close(status)


if __name__=='__main__':main()
