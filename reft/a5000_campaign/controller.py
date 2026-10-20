"""Single remote controller: atomic stage state, lock, bounded children, sealed outputs."""
import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .state import atomic,read,sha,identity,now,event,lock,seal,verify

STAGES=['N1','N2','N3','N4','N5','REPORT']
STOP=False


def source_manifest():
    base=Path(__file__).resolve().parents[2]
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=base).decode().split('\0')
    return {p:sha(base/p) for p in paths if p}


def check_bound(root):
    plan=read(Path(root)/'frozen_plan.json')
    if identity(plan['config'])!=plan['config_sha256']: raise ValueError('Config changed')
    if source_manifest()!=plan['sources']: raise ValueError('Bound source changed')
    if sha(Path(root)/'data_manifest.json')!=plan['data_sha256']: raise ValueError('Data manifest changed')
    if not verify(root,plan['admission_seal']): raise ValueError('Admission evidence changed')
    return plan


def status_md(root,state):
    root=Path(root); lines=['# A5000 campaign status','',f"Updated: {now()}",f"Campaign: {state['campaign_id']}",
        f"Status: {state['status']}",f"Current stage: {state.get('current')}",
        f"Controller PID: {state.get('pid')}",f"GPU-hours (allocated worker residence): {state.get('gpu_seconds',0)/3600:.4f}",
        '', '| Stage | Status | Reason |','|---|---|---|']
    for k,v in state['stages'].items(): lines.append(f"| {k} | {v['status']} | {v.get('reason','')} |")
    h=root/'worker_heartbeat.json'
    if h.exists(): lines+=['','Latest worker heartbeat:', '```json',h.read_text(),'```']
    lines+=['','Final independent official-test/external evaluation remains BLOCKED: no scoped authorization.']
    tmp=root/'CAMPAIGN_STATUS.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'CAMPAIGN_STATUS.md')


def terminate(child):
    if child.poll() is not None: return
    os.killpg(child.pid,signal.SIGTERM)
    try: child.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait()


def run(root,commands=None,skip_binding=False,heartbeat=15):
    """commands/skip_binding are injected only by CPU simulation tests."""
    global STOP
    STOP=False; root=Path(root)
    plan=read(root/'frozen_plan.json') if skip_binding else check_bound(root)
    def stopped(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    with lock(root/'controller.lock'):
        sp=root/'status.json'
        state=read(sp) if sp.exists() else {'campaign_id':root.name,'status':'PENDING','gpu_seconds':0.,
            'stages':{s:{'status':'PENDING','attempts':0} for s in STAGES}}
        # If a controller died while a child survived, do not start a duplicate.
        old=state.get('child_pid')
        if old:
            try: os.kill(old,0)
            except ProcessLookupError: pass
            else: raise RuntimeError('Previous worker still alive; refuse duplicate: '+str(old))
        state.update(pid=os.getpid(),status='RUNNING'); state.pop('child_pid',None)
        def persist(): atomic(sp,state);status_md(root,state)
        persist(); event(root,'controller_started',pid=os.getpid())
        cfg=plan['config']; cap=(cfg['resources']['gpu_hours']-cfg['resources'].get('reserve_gpu_hours',0))*3600; wall=cfg['resources']['wall_hours']*3600
        remaining=[s for s in STAGES if state['stages'][s]['status']!='COMPLETE']
        for stage in STAGES:
            rec=state['stages'][stage]
            if rec['status']=='COMPLETE':
                if not verify(root,rec.get('outputs',{})): raise ValueError('Corrupt completed stage: '+stage)
                for sealfile in rec.get('outputs',{}):
                    if sealfile.endswith('/SEALED.json'):
                        proof=read(root/sealfile); parent=(root/sealfile).parent
                        if proof['metrics_sha256']!=sha(parent/'metrics.json'): raise ValueError('Corrupt sealed metrics')
                        if any(not (parent/'predictions'/(k+'.json')).exists() or sha(parent/'predictions'/(k+'.json'))!=v for k,v in proof['predictions'].items()): raise ValueError('Corrupt sealed predictions')
                continue
            state['current']=stage
            if not skip_binding: check_bound(root)
            if stage!='REPORT' and (STOP or (root/'STOP').exists()):
                state['status']='STOPPED_BY_USER'; persist();break
            if stage!='REPORT' and (state['gpu_seconds']>=cap or time.time()-plan['created_epoch']>=wall):
                state['status']='A5000_CAMPAIGN_RESOURCE_LIMIT_REACHED';persist();break
            # N2 technical failure blocks GO-dependent N4/N5, but N3 remains
            # independently eligible; legitimate NO-GO is a COMPLETE N2 stage.
            dependencies={'N1':[],'N2':['N1'],'N3':['N1'],'N4':['N1','N2','N3'],'N5':['N1','N2','N3','N4'],'REPORT':[]}[stage]
            if any(state['stages'][d]['status']!='COMPLETE' for d in dependencies):
                rec.update(status='BLOCKED',reason='Incomplete dependencies: '+','.join(dependencies));persist();continue
            cmd=commands[stage] if commands else ([sys.executable,'-m','reft.a5000_campaign.reporting','--root',str(root)] if stage=='REPORT' else
                [sys.executable,'-m','reft.a5000_campaign.worker',stage,'--root',str(root)])
            result=root/'stage_results'/(stage+'.json');result.parent.mkdir(exist_ok=True)
            started=time.monotonic(); base_seconds=state['gpu_seconds']; log=root/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True)
            rec.update(status='RUNNING',started=now(),attempts=rec['attempts']+1,command=cmd,
                inputs={'config_sha256':plan['config_sha256'],'data_sha256':plan['data_sha256'],'source_commit':plan['source_commit']})
            persist();event(root,'stage_started',stage=stage,command=cmd)
            with log.open('a') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(cfg.get('gpu_index',0)),'OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
                state['child_pid']=child.pid;persist(); reason=None
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if stage!='REPORT': state['gpu_seconds']=base_seconds+elapsed
                    state['heartbeat']=now();persist()
                    if not skip_binding:
                        try: check_bound(root)
                        except (ValueError,FileNotFoundError) as e: reason='INVALID_BOUND_IDENTITY: '+str(e)
                        sample=subprocess.run(['nvidia-smi','-i',str(cfg.get('gpu_index',0)),
                            '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,timeout=10)
                        rp=root/'resource_usage.csv'; fresh=not rp.exists()
                        with rp.open('a') as usage:
                            if fresh: usage.write('utc,stage,gpu_seconds,memory_used_mib,memory_total_mib,gpu_utilization_percent\n')
                            usage.write(now()+','+stage+','+str(state['gpu_seconds'])+','+sample.stdout.strip()+'\n')
                        if int(time.monotonic()-started)//60 != int(max(0,time.monotonic()-started-heartbeat))//60:
                            size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                            if size>cfg['resources']['disk_gib']*2**30: reason='ARTIFACT_DISK_CAP'
                    occupied_budget=state['gpu_seconds']>=cap or time.time()-plan['created_epoch']>=wall
                    if STOP or (root/'STOP').exists(): reason='STOPPED_BY_USER'
                    elif occupied_budget: reason='STOPPED_BUDGET'
                    elif reason is not None: pass
                    elif elapsed>=cfg['resources']['stage_seconds']: reason='STAGE_TIMEOUT'
                    elif (root/'worker_heartbeat.json').exists() and time.time()-(root/'worker_heartbeat.json').stat().st_mtime>600:
                        reason='STALE_HEARTBEAT'
                    if reason: terminate(child);break
                    # Short, interruptible monitoring interval.
                    time.sleep(heartbeat)
                code=child.wait()
            if stage!='REPORT': state['gpu_seconds']=base_seconds+time.monotonic()-started
            state.pop('child_pid',None)
            rec.update(finished=now(),exit_code=code,seconds=time.monotonic()-started)
            valid=result.exists() and read(result).get('status')=='COMPLETE'
            if code==0 and valid:
                # Seal all immutable scientific outputs, excluding mutable cache,
                # optimizer-latest files, heartbeat/state/logs. Later validation
                # regenerates identical metrics under bound identities.
                paths=[result]
                paths += [p for p in root.glob(stage+'*.json') if not p.name.endswith('.partial.json')]
                paths += list((root/'B0').glob('SEALED.json')) if (root/'B0').exists() else []
                paths+=list((root/'runs').glob('*/selection.json')) if (root/'runs').exists() else []
                paths+=list((root/'runs').glob('*/epoch*/adapter.pt')) if (root/'runs').exists() else []
                paths+=list((root/'runs').glob('*/epoch*/validation/SEALED.json')) if (root/'runs').exists() else []
                rec.update(status='COMPLETE',outputs=seal(root,paths))
            else:
                detail=read(result) if result.exists() else {}
                rec.update(status='BLOCKED' if reason or code==75 or detail.get('status')=='BLOCKED' else 'FAILED',
                    reason=reason or detail.get('reason','Subprocess failed without complete result'))
                event(root,'stage_failed',stage=stage,exit_code=code,reason=rec['reason'])
                # No parameter-changing retries. Resume explicitly recovers valid
                # optimizer checkpoints; infra-only retry is not currently enabled.
            persist();event(root,'stage_finished',stage=stage,status=rec['status'])
            if reason:
                state['status']=('STOPPED_BY_USER' if reason=='STOPPED_BY_USER' else 'A5000_CAMPAIGN_RESOURCE_LIMIT_REACHED' if reason in ['STOPPED_BUDGET','STAGE_TIMEOUT','ARTIFACT_DISK_CAP'] else 'A5000_CAMPAIGN_BLOCKED_ENGINEERING')
                persist();break
        else:
            done=all(v['status']=='COMPLETE' for v in state['stages'].values())
            state['status']='A5000_CAMPAIGN_COMPLETE_WITH_LIMITATIONS' if done else 'A5000_CAMPAIGN_BLOCKED_ENGINEERING'
            state['current']=None;persist()
        # Reporting is always attempted even at a resource stop. CPU-only, no
        # scientific work is added outside the frozen dependency graph.
        if not commands:
            subprocess.run([sys.executable,'-m','reft.a5000_campaign.reporting','--root',str(root)],timeout=180,check=False)
        event(root,'controller_finished',status=state['status']);persist()
        return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    run(args.root)

if __name__=='__main__': main()
