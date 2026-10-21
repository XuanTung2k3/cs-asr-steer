"""Six fixed exploratory jobs. Parent results are consumed read-only."""
import argparse
import time
from pathlib import Path
import signal
import sys
import torch
from reft.a5000_campaign.state import read,atomic,sha,identity,now
from reft.a5000_campaign.worker import Halt,stopping,restore
from reft.a5000_campaign.evaluation import metrics
from reft.a5000.runner import memory
from reft.provenance import weights_hash
from .runtime import SprintRuntime


def parent_records(parent):
    records={'B0':{'metrics':read(parent/'B0/metrics.json'),'checkpoint':None,'total_adapter_gate_parameters':0}}
    for name,filename in [('CE','N1_selection.json'),('WCE','N3_selection.json')]:
        for method,rec in read(parent/filename).items():records[method+'_'+name]=rec
    return records


def record_predictions(parent,rec):
    directory=parent/'B0' if rec['checkpoint'] is None else Path(rec['checkpoint']).parent/'validation'
    sealed=read(directory/'SEALED.json')
    if sha(directory/'metrics.json')!=sealed['metrics_sha256']:raise ValueError('Parent metrics changed')
    predictions={}
    for key,digest in sealed['predictions'].items():
        p=directory/'predictions'/(key+'.json')
        if sha(p)!=digest:raise ValueError('Parent prediction changed')
        predictions[key]=read(p)
    return predictions


def scalar(R):
    baseline=R.baseline();ad,warm=R.warm();grid=[]
    records=parent_records(R.parent)
    for alpha in R.spec['alpha_grid']:
        R.tick(arm='L1',alpha=alpha,phase='scalar_evaluation')
        if alpha in [0,1]:
            source=records['B0' if alpha==0 else 'L_CE']
            rec={'alpha':alpha,'metrics':source['metrics'],'reused':True,'reused_checkpoint_sha256':None if alpha==0 else source['checkpoint_sha256']}
        else:
            ad.alpha=alpha;m,_=R.evaluate(ad,R.root/f'alpha_{alpha}',warm['checkpoint'],baseline=baseline)
            rec={'alpha':alpha,'metrics':m,'reused':False}
        grid.append(rec);atomic(R.root/'scalar_grid.partial.json',grid)
    best=min(grid,key=lambda v:(v['metrics']['genuine_cs_mer'],v['metrics']['mer'],v['metrics']['zh_cer'],R.spec['alpha_grid'].index(v['alpha'])))
    atomic(R.root/'scalar_grid.json',grid)
    selected=R.root/'selected_alpha.json'
    atomic(selected,{'alpha':best['alpha'],'warm_checkpoint':warm['checkpoint'],
        'warm_checkpoint_sha256':warm['checkpoint_sha256'],'source_commit':R.plan['source_commit'],
        'configuration_sha256':R.plan['config_sha256'],'data_sha256':R.plan['data_sha256']})
    return {'arm':'L1','alpha':best['alpha'],'metrics':best['metrics'],'grid':grid,
        'trainable_parameters':0,'total_adapter_parameters':5122,'checkpoint':warm['checkpoint'],
        'checkpoint_sha256':warm['checkpoint_sha256'],'checkpoint_bytes':Path(warm['checkpoint']).stat().st_size+selected.stat().st_size,
        'fit_seconds':0.,'examples':0,'updates':0,'training_peak':None,'scope':'global scalar selection; no new adapter fit'}


def audit(R):
    records=parent_records(R.parent);result={};b=record_predictions(R.parent,records['B0'])
    for name,rec in records.items():
        R.tick(arm='AUDIT',system=name,phase='canonical_rescore')
        if rec['checkpoint'] and sha(rec['checkpoint'])!=rec['checkpoint_sha256']:raise ValueError('Baseline checkpoint mismatch')
        predictions=record_predictions(R.parent,rec);m=metrics(R.V,predictions,b if name!='B0' else None)
        for key in ['mer','genuine_cs_mer','en_wer','zh_cer','noncs_mer','loops','nontermination','insertions']:
            if m[key]!=rec['metrics'][key]:raise ValueError('Canonical scoring mismatch '+name+' '+key)
        if name=='B0':ad=None
        elif name.startswith('L_'):
            ad=R.factory('L');ad.load_warm(torch.load(rec['checkpoint'],map_location='cpu',weights_only=True))
        else:ad=restore(R,rec)
        row=R.V[0];pred=R.decode(ad,row)
        if pred['tokens']!=predictions[row['utterance_id']]['tokens']:raise ValueError('Canonical decode mismatch '+name)
        result[name]={'checkpoint_sha256':rec.get('checkpoint_sha256'),'metric_parity':True,'sample_decode_parity':True,
            'sample_inference_seconds':pred['seconds'],'sample_audio_seconds':row['duration_sec']}
        del ad;torch.cuda.empty_cache()
    # Read-only accounting reconciliation. Do not sum attributed warm-start
    # costs for L0/L1/L2/L3/L4 and call that actual campaign allocation.
    usages=[]
    for job in (R.parent/'jobs').iterdir():
        runfiles=list((job/'runs').glob('*/run.json'))
        for p in runfiles:
            u=p.parent/'usage.json'
            if u.exists():usages.append({'run':str(p.parent),'spec':read(p),'allocated_seconds':read(u)['allocated_seconds']})
    atomic(R.campaign/'parent_cost_audit.json',{'controller_recorded_gpu_seconds':read(R.parent/'status.json')['gpu_seconds'],
        'sum_arm_usage_seconds':sum(v['allocated_seconds'] for v in usages),'arms':usages,
        'difference_explanation':'Controller includes model initialization, B0/scalar, admission and failures; arm files cover their own fitting/validation. Scope differences and recovery polling prevent exact equality.'})
    return {'arm':'AUDIT','verified':True,'systems':result,'C500_opened':False,'parent_N2_decision':read(R.parent/'N2_decision.json')['decision']}


def preflight(R):
    # Selected plain LRA and frozen decode parity, real loss/backward, reload,
    # energy checks and nonzero gate opportunity on permitted training rows.
    warm_pred=record_predictions(R.parent,parent_records(R.parent)['L_CE'])
    report={'arms':{},'passed':True,'source':R.plan['source_commit'],'parent_source':R.spec['parent_source']}
    for arm in ['L0','L2','L3','L4']:
        ad,warm=R.warm(None if arm=='L0' else arm);params=[p for p in ad.parameters() if p.requires_grad]
        row=sorted(R.T,key=lambda v:v['target_tokens_incl_eos'])[len(R.T)//2]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            a,Y,d=R.forward(ad,row);plain,_=R.warm();b,_,_=R.forward(plain,row)
            if not torch.equal(a.logits,b.logits):raise ValueError('Warm gate identity mismatch')
        del a,b,plain
        opt=torch.optim.AdamW(params,lr=R.spec['training']['lr'],weight_decay=0)
        opt.zero_grad(set_to_none=True);start=time.monotonic();o,Y,d=R.forward(ad,row)
        from reft.a5000.objectives import weighted_ce
        loss=weighted_ce(o.logits,Y);loss.backward()
        if not bool(torch.isfinite(loss)):raise ValueError('Invalid real CE')
        if any(p.grad is not None for p in R.backbone):raise ValueError('Backbone unfrozen')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Gradient failure')
        if arm!='L0' and not bool(ad.gate.net[-1].weight.grad.norm()>0):raise ValueError('No warm gate opportunity')
        opt.step();torch.cuda.synchronize();seconds=time.monotonic()-start
        ck=R.root/(arm+'.pt');from reft.a5000.runner import save_torch
        save_torch(ck,R.state(ad));copy=R.factory('L',gate=None if arm=='L0' else arm)
        copy.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
        with torch.no_grad():
            a,_,_=R.forward(ad,row);b,_,_=R.forward(copy,row)
            if not torch.equal(a.logits,b.logits):raise ValueError('GPU serialization mismatch')
        report['arms'][arm]={'trainable':sum(p.numel() for p in params),'step_seconds':seconds,'finite_loss':float(loss.detach()),**memory()}
        if arm!='L0':
            p=R.decode(copy,R.V[0])
            if not p['tokens'] or not torch.isfinite(torch.tensor(p['allocation']['coefficients'])).all():
                raise ValueError('Gated free-decoding failure')
            report['arms'][arm]['free_decode_seconds']=p['seconds']
        if arm in ['L0','L4']:
            # Admit the longest target envelope as well as the median timing
            # sample; do not truncate or remove difficult training examples.
            opt.zero_grad(set_to_none=True)
            worst=max(R.T,key=lambda v:v['target_tokens_incl_eos'])
            w,wy,_=R.forward(ad,worst);wl=weighted_ce(w.logits,wy);wl.backward()
            if not bool(torch.isfinite(wl)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise ValueError('Longest training target envelope failed')
            report['arms'][arm]['longest_target_tokens']=worst['target_tokens_incl_eos']
            report['arms'][arm]['envelope_memory']=memory()
            del w,wy,wl
        del o,loss,a,b,ad,copy,opt;torch.cuda.empty_cache()
    ad,_=R.warm()
    decode_times=[]
    for row in R.V[:3]:
        p=R.decode(ad,row)
        if p['tokens']!=warm_pred[row['utterance_id']]['tokens']:raise ValueError('LRA decode parity failed')
        decode_times.append(p['seconds'])
    report['canonical_decode_parity']=True;report['canonical_score_parity']=True
    rescored=metrics(R.V,warm_pred,R.baseline());old=read(R.parent/'N1_selection.json')['L']['metrics']
    if any(rescored[k]!=old[k] for k in ['mer','genuine_cs_mer','noncs_mer','en_wer','zh_cer']):raise ValueError('Metric parity')
    if weights_hash(R.b.model)!=R.initial_hash:raise ValueError('Backbone mutated')
    archived=read(R.campaign/'runtime_estimate.json')['archived_seconds']
    # Conservative envelope combines the archived full jobs with measured real
    # decoder/backward samples. This is a projection, not a completion promise.
    decode_full=max(sum(decode_times)/len(decode_times)*len(R.V),warm['metrics']['seconds'])
    gate_estimates={arm:1.25*max(archived.get('G5') or 0,
        rec['step_seconds']*len(R.T)+decode_full)+180 for arm,rec in report['arms'].items()}
    scalar_estimate=1.25*max((archived.get('scalar') or 0)*3/5,decode_full*3)+120
    report['projected_seconds']={**gate_estimates,'L1':scalar_estimate}
    report['projected_max_seconds']=max(report['projected_seconds'].values())
    if report['projected_max_seconds']>=R.spec['timing']['soft_stop_seconds']:
        raise RuntimeError('Measured projection does not fit the safe55minute compute window')
    atomic(R.root/'admission_report.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True,
        choices=['L0','L1','L2','L3','L4','AUDIT','PREFLIGHT']);a=p.parse_args();root=Path(a.root);work=root/'jobs'/a.arm
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping);R=None;started=time.monotonic()
    try:
        R=SprintRuntime(work)
        result=preflight(R) if a.arm=='PREFLIGHT' else scalar(R) if a.arm=='L1' else audit(R) if a.arm=='AUDIT' else R.fit(a.arm)
        if weights_hash(R.b.model)!=R.initial_hash:raise ValueError('Backbone mutated')
        atomic(work/'result.json',{'status':'COMPLETE','result':result,'allocated_seconds':time.monotonic()-started,'finished':now()})
    except Halt as e:
        atomic(work/'result.json',{'status':'INCOMPLETE','reason':str(e),'allocated_seconds':time.monotonic()-started,'finished':now()})
        sys.exit(75)
    except Exception as e:
        atomic(work/'result.json',{'status':'FAILED','reason':type(e).__name__,'error':str(e),'allocated_seconds':time.monotonic()-started,'finished':now()});raise
