"""Up to six independent A5000 jobs; no DDP or scientific batch changes."""
import argparse
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .state import atomic,read,sha,identity,now,event,lock,seal,verify
from .controller import check_bound,terminate,status_md
from .parallel_jobs import graph,pick_ready,selection

STOP=False
GPU_KINDS={'bootstrap','train','gate','weighted_gate','scalar','legacy'}


def outputs_valid(root,record):
    if not verify(root,record.get('outputs',{})):return False
    for name in record['outputs']:
        if name.endswith('/SEALED.json'):
            proof=read(root/name); parent=(root/name).parent
            if proof['metrics_sha256']!=sha(parent/'metrics.json'):return False
            if any(not (parent/'predictions'/(k+'.json')).exists() or sha(parent/'predictions'/(k+'.json'))!=v for k,v in proof['predictions'].items()):return False
    return True


def publish_runs(root):
    dest=root/'runs';dest.mkdir(exist_ok=True)
    for path in (root/'jobs').glob('*/runs/*'):
        link=dest/path.name
        if link.exists():
            if link.resolve()!=path.resolve():raise ValueError('Duplicate arm identity')
        else:link.symlink_to(path,target_is_directory=True)


def complete_cpu(root,job):
    from .evaluation import pilot_decision
    from .reporting import report
    jobs=read(root/'job_graph.json')
    def result(name):return read(root/'jobs'/name/'result.json')['result']
    kind=job['kind'];a=job['args']
    if kind=='select':
        rec=selection([result(d) for d in job['dependencies']]);atomic(root/f"selection_{a['objective']}_{a['method']}.json",rec)
        return rec
    if kind=='assemble':
        chosen={m:read(root/f"selection_{a['objective']}_{m}.json") for m in ['R','L','LP','LS']}
        if a['objective']=='WCE' and (root/'jobs/G5_WCE/result.json').exists() and read(root/'jobs/G5_WCE/result.json')['status']=='COMPLETE':
            chosen['G5_WCE']=result('G5_WCE')
        atomic(root/('N1_selection.json' if a['objective']=='CE' else 'N3_selection.json'),chosen);return chosen
    if kind=='decision':
        recs={k:result(k) for k in ['G3','G4','G5','G8']}
        panel={k:v['metrics'] for k,v in recs.items()};scalar=result('scalar')['winner']
        panel['G1']=scalar['metrics'];panel['G0']=read(root/'selection_CE_R.json')['metrics']
        decision=pilot_decision(panel,read(root/'frozen_plan.json')['config']['pilot_rule'])
        atomic(root/'N2_scalar.json',scalar);atomic(root/'N2_panel.json',panel);atomic(root/'N2_selection.json',recs)
        atomic(root/'N2_decision.json',decision);return decision
    if kind in ['core_report','report']:
        publish_runs(root);report(root)
        if kind=='core_report':atomic(root/'CORE_COMPLETE.json',{'utc':now(),'N1_N2_N3_complete':True,
            'hours_since_freeze':(time.time()-read(root/'frozen_plan.json')['created_epoch'])/3600})
        return {'report_generated':True,'stage':kind}
    raise ValueError('Unknown CPU job')


def run(root,synthetic_commands=None,interval=10):
    global STOP
    STOP=False;root=Path(root);plan=read(root/'frozen_plan.json') if synthetic_commands else check_bound(root)
    jobs=read(root/'job_graph.json');cfg=plan['config'];gpu_ids=cfg['gpu_indices']
    def stopped(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(root/'controller.lock'))
        # Try each GPU without waiting; other owners keep their resources.
        available=[]
        for gpu in gpu_ids:
            if synthetic_commands: available.append(gpu);continue
            try:
                stack.enter_context(lock(Path('/home/ubuntu/tung.nx')/f'a5000-campaign-gpu{gpu}.lock'))
                occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if occupied:continue
                available.append(gpu)
            except BlockingIOError:continue
        if not available:raise RuntimeError('No unoccupied authorized GPU available')
        path=root/'status.json'
        if path.exists() and 'jobs' in read(path):state=read(path)
        else:
            state={'campaign_id':root.name,'gpu_seconds':cfg.get('prior_compute_seconds',0)+(read(root.parent/'admission_cost.json')['seconds'] if (root.parent/'admission_cost.json').exists() else 0),
                'jobs':{k:{'status':'PENDING','attempts':0} for k in jobs},'stages':{},'status':'PENDING'}
        running={};handles={}
        for k,rec in state['jobs'].items():
            if rec['status']=='COMPLETE':
                if not outputs_valid(root,rec):raise ValueError('Corrupt completed job '+k)
            elif rec['status']=='RUNNING':
                pid=rec.get('pid')
                if pid:
                    try:os.kill(pid,0)
                    except ProcessLookupError:pass
                    else:raise RuntimeError('Worker survives previous controller; no duplicate: '+str(pid))
                rec['status']='PENDING'
            elif rec['status']=='BLOCKED' and rec.get('reason') in ['STOPPED_BY_USER','STOPPED_BUDGET']:
                rec['status']='PENDING'
        state.update(pid=os.getpid(),status='RUNNING',gpu_indices=available)
        cap=(cfg['resources']['gpu_hours']-cfg['resources']['reserve_gpu_hours'])*3600
        last_tick=time.monotonic();last_sample=0.;last_report=0.
        def persist():
            state['heartbeat']=now();state['current']=','.join(running) or 'scheduling'
            remaining=[k for k,v in state['jobs'].items() if v['status']=='PENDING']
            state['remaining_jobs']=remaining
            state['running_progress']={k:read(root/'jobs'/k/'worker_heartbeat.json') for k in running if (root/'jobs'/k/'worker_heartbeat.json').exists()}
            state['core_complete']=(root/'CORE_COMPLETE.json').exists()
            state['stages']={s:{'status':state['jobs'].get(s,{}).get('status','PENDING')} for s in ['N1','N2','N3','N4','N5','REPORT']}
            atomic(path,state);status_md(root,state)
        def finish(k,code):
            info=running.pop(k);handles.pop(k).close();rec=state['jobs'][k];work=root/'jobs'/k;p=work/'result.json'
            rec.update(exit_code=code,finished=now(),elapsed_seconds=time.monotonic()-info['start'])
            if code==0 and p.exists() and read(p)['status']=='COMPLETE':
                files=[p,work/'job.json'];files+=list((work/'runs').glob('*/epoch*/adapter.pt'))
                files+=list((work/'runs').glob('*/epoch*/validation/SEALED.json'));files+=list((work/'B0').glob('SEALED.json'))
                rec.update(status='COMPLETE',outputs=seal(root,files));publish_runs(root)
                if k=='B0':
                    target=Path(read(p)['result']['directory']);link=root/'B0'
                    if not link.exists():link.symlink_to(target,target_is_directory=True)
                if jobs[k]['kind']=='legacy':
                    # Legacy stages publish central N4/N5 files themselves.
                    pass
            else:
                detail=read(p) if p.exists() else {}
                rec.update(status='BLOCKED' if code==75 or detail.get('status')=='BLOCKED' else 'FAILED',
                    reason=detail.get('reason','Process failed before complete output'))
            event(root,'job_finished',job=k,status=rec['status'],exit_code=code)
            print(now(),k,rec['status'],flush=True)
        persist();event(root,'controller_started',pid=os.getpid(),gpus=available)
        print(now(),'parallel_controller_started',os.getpid(),available,flush=True)
        try:
            while True:
                tick=time.monotonic();state['gpu_seconds']+=(tick-last_tick)*len(running);last_tick=tick
                for k,info in list(running.items()):
                    proc=info['process'];limit=cfg['resources']['stage_seconds']
                    hb=root/'jobs'/k/'worker_heartbeat.json'
                    if proc.poll() is not None:finish(k,proc.returncode)
                    elif tick-info['start']>limit or (hb.exists() and time.time()-hb.stat().st_mtime>600):
                        terminate(proc);finish(k,proc.returncode)
                if not synthetic_commands:check_bound(root)
                out_of_budget=state['gpu_seconds']>=cap or time.time()-plan['created_epoch']>=cfg['resources']['wall_hours']*3600
                if STOP or (root/'STOP').exists() or out_of_budget:
                    why='STOPPED_BUDGET' if out_of_budget else 'STOPPED_BY_USER'
                    for k,info in list(running.items()):
                        start=time.monotonic();terminate(info['process'])
                        state['gpu_seconds']+=(time.monotonic()-start)*max(1,len(running))
                        finish(k,info['process'].returncode)
                        state['jobs'][k].update(status='BLOCKED',reason=why)
                    state['status']='A5000_CAMPAIGN_RESOURCE_LIMIT_REACHED' if out_of_budget else 'STOPPED_BY_USER';persist();break
                # Failed prerequisites block descendants, but independent WCE
                # and other comparisons remain eligible.
                for k,j in jobs.items():
                    if state['jobs'][k]['status']=='PENDING' and any(state['jobs'][d]['status'] in ['FAILED','BLOCKED'] for d in j['dependencies']):
                        state['jobs'][k].update(status='BLOCKED',reason='Dependency failed')
                for k in pick_ready(jobs,state['jobs']):
                    job=jobs[k];rec=state['jobs'][k];work=root/'jobs'/k;work.mkdir(parents=True,exist_ok=True)
                    atomic(work/'job.json',job)
                    for name in ['frozen_plan.json','data_manifest.json']:
                        link=work/name
                        if not link.exists():link.symlink_to(root/name)
                    if k=='G5_WCE' and read(root/'N2_decision.json')['decision']!='N2_EXPLORATORY_GO':
                        rec.update(status='SKIPPED',reason='Frozen N2 gate did not pass');event(root,'job_skipped',job=k);continue
                    if job['kind'] not in GPU_KINDS and synthetic_commands is None:
                        rec.update(status='RUNNING',started=now(),attempts=rec['attempts']+1);persist()
                        try:
                            result=complete_cpu(root,job);p=work/'result.json';atomic(p,{'status':'COMPLETE','result':result,'finished':now()})
                            extra=[]
                            if job['kind']=='select':extra=[root/f"selection_{job['args']['objective']}_{job['args']['method']}.json"]
                            elif job['kind']=='assemble':extra=[root/('N1_selection.json' if job['args']['objective']=='CE' else 'N3_selection.json')]
                            elif job['kind']=='decision':extra=[root/n for n in ['N2_scalar.json','N2_panel.json','N2_selection.json','N2_decision.json']]
                            rec.update(status='COMPLETE',outputs=seal(root,[p,work/'job.json']+extra))
                        except Exception as e:
                            rec.update(status='FAILED',reason=repr(e));event(root,'job_failed',job=k,reason=repr(e))
                        continue
                    used={v['gpu'] for v in running.values()};free=[g for g in available if g not in used]
                    if not free:continue
                    gpu=free[0]
                    # Re-check occupancy immediately before each dispatch. No
                    # unrelated processes are killed or preempted.
                    if not synthetic_commands:
                        occ=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                        if occ:continue
                    log=root/'logs'/(k+'.log');log.parent.mkdir(exist_ok=True);h=log.open('a')
                    cmd=synthetic_commands[k] if synthetic_commands else [sys.executable,'-m','reft.a5000_campaign.parallel_worker','--root',str(root),'--job-id',k]
                    proc=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
                    rec.update(status='RUNNING',attempts=rec['attempts']+1,pid=proc.pid,gpu=gpu,started=now(),command=cmd,
                        input_hashes={'config':plan['config_sha256'],'source':plan['source_commit'],'data':plan['data_sha256'],'job':identity(job),'dependencies':{d:sha(root/'jobs'/d/'result.json') for d in job['dependencies'] if (root/'jobs'/d/'result.json').exists()}})
                    running[k]={'process':proc,'gpu':gpu,'start':time.monotonic()};handles[k]=h
                    event(root,'job_started',job=k,pid=proc.pid,gpu=gpu);print(now(),k,'started GPU',gpu,flush=True)
                persist()
                if not running and not any(v['status']=='PENDING' for v in state['jobs'].values()):
                    state['status']='A5000_CAMPAIGN_COMPLETE_WITH_LIMITATIONS' if all(v['status'] in ['COMPLETE','SKIPPED'] for v in state['jobs'].values()) else 'A5000_CAMPAIGN_BLOCKED_ENGINEERING'
                    persist();break
                if tick-last_sample>=30 and not synthetic_commands:
                    sample=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    with (root/'resource_usage.csv').open('a') as f:
                        for line in sample.stdout.splitlines():f.write(now()+','+str(state['gpu_seconds'])+','+line+'\n')
                    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
                    if size>cfg['resources']['disk_gib']*2**30:atomic(root/'STOP',{'reason':'ARTIFACT_DISK_CAP','utc':now()})
                    last_sample=tick
                time.sleep(interval)
        finally:
            for k,info in list(running.items()):
                terminate(info['process']);finish(k,info['process'].returncode)
                state['jobs'][k].update(status='BLOCKED',reason='STOPPED_BY_USER')
            persist()
            if not synthetic_commands:
                from .reporting import report
                report(root)
        return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
if __name__=='__main__':main()
