"""One arm on one assigned GPU; per-job state and a locked shared frozen cache."""
import argparse
from pathlib import Path
import signal
import sys
import time
import torch
from .state import read,atomic,identity,sha,now
from .worker import Runtime,Halt,stopping,restore


def execute(root,job):
    root=Path(root);work=root/'jobs'/job['id'];cfg=read(root/'frozen_plan.json')['config']
    if job['kind']=='legacy':
        # Serial post-core stages use the central artifact namespace.
        from .worker import n4,n5
        R=Runtime(root)
        try: return n4(R) if job['args']['stage']=='N4' else n5(R)
        finally: R.tick()
    R=Runtime(work);args=job['args']
    try:
        if job['kind']=='bootstrap':
            for row in R.T+R.V:R.cache(row)
            m,_=R.evaluate(None,work/'B0')
            return {'metrics':m,'directory':str(work/'B0')}
        baseline={p.stem:read(p) for p in (root/'B0'/'predictions').glob('*.json')}
        if len(baseline)!=len(R.V):raise ValueError('Incomplete bound baseline')
        if job['kind']=='train':return R.train(baseline=baseline,**args)
        if job['kind'] in ['gate','weighted_gate']:
            weighted=job['kind']=='weighted_gate'
            rec=read(root/('selection_WCE_R.json' if weighted else 'selection_CE_R.json'))
            kind='G5' if weighted else args['gate'];tag='N3_G5_WCE' if weighted else args['tag']
            lr=rec['spec']['lr'] if kind is None else cfg['gate']['lr']
            return R.train('R',lr,cfg['gate']['epochs'],warm=rec['checkpoint'],gate=kind,
                tag=tag,omega=rec['spec']['omega'] if weighted else 1,baseline=baseline)
        if job['kind']=='scalar':
            warm=read(root/'selection_CE_R.json');ad=restore(R,warm);panel=[]
            for alpha in cfg['gate']['scalar_grid']:
                ad.alpha=alpha;m,_=R.evaluate(ad,work/f'scalar_{alpha}',warm['checkpoint'],baseline=baseline)
                panel.append({'alpha':alpha,'metrics':m})
            winner=min(panel,key=lambda p:(p['metrics']['genuine_cs_mer'],p['metrics']['mer'],p['alpha']))
            return {'winner':winner,'panel':panel}
        raise ValueError('Unknown GPU job')
    finally:
        if R.initial_hash!=__import__('reft.provenance',fromlist=['weights_hash']).weights_hash(R.b.model):
            raise ValueError('Backbone mutated')
        if R.arm_usage is not None:
            atomic(R.arm_usage,{'allocated_seconds':R.arm_prior+time.monotonic()-R.arm_start,'utc':now()})
        atomic(R.usage,{'resident_seconds':R.offset+time.monotonic()-R.start,'utc':now()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job-id',required=True);a=p.parse_args()
    root=Path(a.root);job=read(root/'job_graph.json')[a.job_id];work=root/'jobs'/a.job_id
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    try:
        result=execute(root,job)
        atomic(work/'result.json',{'status':'COMPLETE','job_sha256':identity(job),'result':result,'finished':now()})
    except Halt as e:
        atomic(work/'result.json',{'status':'BLOCKED','reason':str(e)});sys.exit(75)
    except Exception as e:
        atomic(work/'result.json',{'status':'FAILED','reason':type(e).__name__,'error':str(e)});raise

if __name__=='__main__':main()
