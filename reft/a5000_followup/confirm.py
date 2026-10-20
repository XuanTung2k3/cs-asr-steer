"""One post-selection development confirmation on original TRAIN-side C150.

Never trains, selects or tunes. Refuses access before both primary arms finish and their choices are frozen.
Resumable partial predictions do not constitute a second confirmation or a new selection opportunity.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd
import torch

from csasr.models.whisper import load_whisper
from reft.provenance import weights_hash,environment
from reft.text import target_text
from reft.a5000.adapters import EncoderAdapter
from reft.a5000.data import ROOT,config,runtime,output,atomic,sha,original_manifest
from reft.a5000.runner import Runtime,Budget,Paused,git,check,score,memory


def require_completed(root):
    status=json.loads((root/'status.json').read_text())
    if status['status']!='COMPLETE' or not (root/'comparison.json').exists():
        raise RuntimeError('Both primary arms must finish before opening C150')
    selections=json.loads((root/'selection.json').read_text())
    assert set(selections)=={'ereft','lra'}
    for d in selections.values():assert sha(d['checkpoint'])==d['checkpoint_sha256']
    return selections


def confirmation_rows(root):
    # Separate reader; the initial campaign's C150 firewall remains unchanged.
    require_completed(root)
    assert (root/'confirmation/freeze.json').exists(),'Freeze selection before opening confirmation data'
    spec=json.loads((ROOT/config()['manifest']).read_text())
    assert sha(original_manifest())==spec['source_manifest_sha256']
    df=pd.read_parquet(original_manifest()).set_index('utterance_id')
    rows=[]
    for ident in spec['roles']['A5000_C150']:
        r=df.loc[ident['utterance_id']].to_dict()|ident
        assert r['official_split']=='train' and sha(r['audio_path'])==r['audio_full_sha256']
        assert hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest()==r['target_text_sha256']
        rows.append(r)
    assert len(rows)==150
    return rows


class ConfirmationRuntime(Runtime):
    def __init__(self,budget,dest):
        self.budget=budget;self.C=config();self.R=runtime();self.freeze=check().copy()
        core=self.freeze['source_commit'];self.freeze['source_commit']=git('rev-parse','HEAD')
        torch.manual_seed(42);torch.set_num_threads(2)
        self.b=load_whisper({'model':{'id':str(Path(self.R['model']['root']).expanduser()),'dtype':'bfloat16',
            'device':'cuda','local_files_only':True,'attn_implementation':'eager','freeze_backbone':True}})
        assert self.b.revision=='sha256:'+self.R['model']['sha256']
        self.initial_weights=weights_hash(self.b.model);self.eos=self.b.processor.tokenizer.eos_token_id
        self.fingerprint=hashlib.sha256(json.dumps({'model':self.R['model'],'data':self.freeze['role_manifest_sha256'],
            'processor':sha(Path(self.R['model']['root']).expanduser()/'tokenizer_config.json')},sort_keys=True).encode()).hexdigest()
        self.T=[];self.V=confirmation_rows(output())
        atomic(dest/'runtime.json',{'source_commit':self.freeze['source_commit'],'primary_core_source':core,
            'environment':environment(),'initial_weights':self.initial_weights,'fingerprint':self.fingerprint})


def evaluate_confirmation(R,ad,dest,checkpoint,baseline):
    dest.mkdir(parents=True,exist_ok=True)
    expected={'source_commit':R.freeze['source_commit'],'checkpoint_sha256':sha(checkpoint) if checkpoint else None,
        'selection_sha256':sha(output()/'selection.json'),'role':'A5000_C150',
        'role_manifest_sha256':R.freeze['role_manifest_sha256'],'decode':R.C['decode']}
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==expected
    else:atomic(dest/'manifest.json',expected)
    preds={};pp=dest/'predictions';pp.mkdir(exist_ok=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for row in R.V:
        R.budget.tick();p=pp/f"{row['utterance_id']}.json"
        if p.exists():d=json.loads(p.read_text())
        else:d=R.decode(ad,row);atomic(p,d)
        assert d['id']==row['utterance_id'] and d['audio_sha256']==row['audio_full_sha256']
        preds[d['id']]=d
    if not (dest/'SEALED.json').exists():
        m=score(R.V,preds,baseline);m.update(memory());atomic(dest/'metrics.json',m)
        text=''.join(json.dumps(preds[r['utterance_id']],ensure_ascii=False,sort_keys=True)+'\n' for r in R.V)
        p=dest/'predictions.jsonl.tmp';p.write_text(text);p.replace(dest/'predictions.jsonl')
        atomic(dest/'SEALED.json',{'metrics_sha256':sha(dest/'metrics.json'),'predictions_sha256':sha(dest/'predictions.jsonl'),
            'manifest_sha256':sha(dest/'manifest.json')})
    seal=json.loads((dest/'SEALED.json').read_text())
    assert sha(dest/'metrics.json')==seal['metrics_sha256'] and sha(dest/'predictions.jsonl')==seal['predictions_sha256']
    return json.loads((dest/'metrics.json').read_text()),preds


def main():
    root=output();dest=root/'confirmation'
    selections=require_completed(root)
    if git('status','--porcelain'):raise RuntimeError('Commit confirmation code before execution')
    check();dest.mkdir(exist_ok=True)
    frozen={'source_commit':git('rev-parse','HEAD'),'selection_sha256':sha(root/'selection.json'),
        'checkpoints':{k:v['checkpoint_sha256'] for k,v in selections.items()},
        'role_manifest_sha256':sha(ROOT/config()['manifest']),
        'rule':'single paired post-selection confirmation; no LR/checkpoint/model selection; original TRAIN holdout only'}
    if (dest/'freeze.json').exists():
        old=json.loads((dest/'freeze.json').read_text());frozen['created_utc']=old['created_utc']
        assert old==frozen
    else:
        frozen['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        with (dest/'freeze.json').open('x') as f:json.dump(frozen,f,sort_keys=True,indent=1)
    if (dest/'comparison.json').exists():print('Confirmation already complete; no new model evaluation');return
    with (root/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','-i',os.environ.get('CUDA_VISIBLE_DEVICES','0'),
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('Selected GPU occupied: '+occupied)
        atomic(dest/'process.json',{'pid':os.getpid(),'started':time.time(),'source_commit':frozen['source_commit']})
        budget=Budget('confirmation');status='FAILED'
        try:
            R=ConfirmationRuntime(budget,dest)
            b0,baseline=evaluate_confirmation(R,None,dest/'B0',None,None)
            results={'B0':{k:v for k,v in b0.items() if k!='per_utterance'}}
            for method in ['ereft','lra']:
                d=selections[method];ad=EncoderAdapter(method).to('cuda')
                ad.load_state_dict(torch.load(d['checkpoint'],map_location='cpu',weights_only=True))
                m,_=evaluate_confirmation(R,ad,dest/method,d['checkpoint'],baseline)
                results[method]={'selection':{k:d[k] for k in ['lr','epoch','checkpoint','checkpoint_sha256']},
                    'metrics':{k:v for k,v in m.items() if k!='per_utterance'}}
                del ad
            assert weights_hash(R.b.model)==R.initial_weights
            atomic(dest/'comparison.json',{'systems':results,'freeze':frozen,
                'claim':'One paired development holdout from original TRAIN, not official/external test or formal superiority',
                'selection_changed':False,'backbone_unchanged':True})
            atomic(dest/'status.json',{'status':'COMPLETE','pid':os.getpid()});status='COMPLETE'
        except Paused as e:
            status='PAUSED';atomic(dest/'status.json',{'status':status,'reason':str(e)})
        finally:budget.close(status)


if __name__=='__main__':main()
