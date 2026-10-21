"""Admission, detached six-job launch, checkpoint-safe deadline and reports."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from reft.a5000_campaign.state import atomic,read,sha,identity,now,event,lock,seal,verify
from reft.a5000_campaign.controller import source_manifest

BASE=Path(__file__).resolve().parents[2]
DEFAULT=BASE/'configs/reft/a5000_lra_sprint/sprint_v1.json'
STOP=False


def bound(root):
    root=Path(root);p=read(root/'frozen_plan.json')
    admission=root/'admitted.json'
    if admission.exists() and read(admission)['frozen_plan_sha256']!=sha(root/'frozen_plan.json'):
        raise ValueError('Admitted plan changed')
    if source_manifest()!=p['sources'] or identity(p['config'])!=p['config_sha256']:raise ValueError('Sprint source/config changed')
    if sha(Path(p['spec']['parent_root'])/'data_manifest.json')!=p['data_sha256']:raise ValueError('Development manifest changed')
    for name,digest in p['parent_inputs'].items():
        if sha(name)!=digest:raise ValueError('Parent input changed: '+name)
    return p


def output_seal(root,arm):
    """Completion requires immutable weights/predictions, not only exit code0."""
    root=Path(root);job=root/'jobs'/arm;result=job/'result.json';rec=read(result)
    if rec['status']!='COMPLETE':raise ValueError('Incomplete worker result')
    files=[result];payload=rec['result']
    if arm in ['L0','L2','L3','L4']:
        ck=Path(payload['checkpoint'])
        if sha(ck)!=payload['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
        directory=job/'validation';record=read(directory/'SEALED.json')
        if sha(directory/'metrics.json')!=record['metrics_sha256'] or sha(directory/'identity.json')!=record['identity_sha256']:
            raise ValueError('Evaluation mismatch')
        if read(directory/'metrics.json')!=payload['metrics']:raise ValueError('Result metric mismatch')
        for uid,digest in record['predictions'].items():
            path=directory/'predictions'/(uid+'.json')
            if sha(path)!=digest:raise ValueError('Prediction mismatch')
            files.append(path)
        files.extend([ck,directory/'SEALED.json',directory/'metrics.json',directory/'identity.json'])
    elif arm=='L1':
        files.extend([job/'scalar_grid.json',job/'selected_alpha.json'])
        # All evaluated alpha panels have their own identity/prediction seal.
        for directory in job.glob('alpha_*'):
            record=read(directory/'SEALED.json')
            for path,digest in [(directory/'metrics.json',record['metrics_sha256']),
                                (directory/'identity.json',record['identity_sha256'])]+[
                                (directory/'predictions'/(key+'.json'),digest) for key,digest in record['predictions'].items()]:
                if sha(path)!=digest:raise ValueError('Scalar panel corrupted')
                files.append(path)
            files.append(directory/'SEALED.json')
    elif arm=='AUDIT' and not payload.get('verified'):raise ValueError('Baseline audit invalid')
    return seal(root,files)


def work(root,arm):
    dest=Path(root)/'jobs'/arm;dest.mkdir(parents=True,exist_ok=True)
    for name in ['frozen_plan.json','data_manifest.json']:
        p=dest/name
        target=Path(root)/name
        if not target.is_file():raise ValueError('Missing immutable input '+name)
        if p.is_symlink() and p.resolve()!=target.resolve():raise ValueError('Wrong input link '+name)
        if not p.exists() and not p.is_symlink():p.symlink_to(target)
    return dest


def prepare(specfile):
    spec=read(specfile);parent=Path(spec['parent_root']);root=Path(spec['output_base'])/spec['campaign_id']
    if root.exists():raise FileExistsError('Immutable sprint ID already exists')
    if subprocess.check_output(['git','status','--porcelain'],cwd=BASE,text=True).strip():raise ValueError('Commit source first')
    if sha(parent/'frozen_plan.json')!=spec['parent_plan_sha256']:raise ValueError('Parent freeze mismatch')
    pp=read(parent/'frozen_plan.json')
    if pp['source_commit']!=spec['parent_source'] or sha(parent/'data_manifest.json')!=spec['data_sha256']:raise ValueError('Parent identity mismatch')
    if read(parent/'N2_decision.json')['decision']!='N2_NO_GO':raise ValueError('Expected original NO-GO record')
    rec=read(parent/'N1_selection.json')['L']
    if sha(rec['checkpoint'])!=spec['warm_checkpoint_sha256']:raise ValueError('Warm checkpoint identity mismatch')
    import shutil
    if any(shutil.which(x) for x in ['sbatch','qsub','bsub']):raise RuntimeError('Scheduler integration required')
    root.mkdir(parents=True)
    cfg=json.loads(json.dumps(pp['config']));cfg['campaign_root']=str(root);cfg['shared_cache_root']=str(root/'unused_cache')
    cfg['resources'].update(gpu_hours=6,reserve_gpu_hours=0,wall_hours=24,max_gpus=6)
    cfg['training'].update(spec['training']);cfg['training']['weighted_omegas']=[]
    parent_inputs={str(parent/name):sha(parent/name) for name in ['frozen_plan.json','data_manifest.json','N1_selection.json','N3_selection.json','N2_decision.json','B0/metrics.json','B0/SEALED.json']}
    for filename in ['N1_selection.json','N3_selection.json']:
        for item in read(parent/filename).values():parent_inputs[item['checkpoint']]=sha(item['checkpoint'])
    plan={'schema':spec['schema'],'created_epoch':time.time(),'utc':now(),'spec':spec,'config':cfg,'config_sha256':identity(cfg),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        'source_path':str(BASE),'sources':source_manifest(),'data_sha256':spec['data_sha256'],'parent_inputs':parent_inputs,
        'parent_receipt':{'status':read(parent/'status.json'),'warm_selection':rec,'N2_decision':read(parent/'N2_decision.json')}}
    atomic(root/'frozen_plan.json',plan)
    # Copy only metadata/IDs/hashes; source reader later pushes down T/V ID
    # filters so held-out transcripts/audio are never materialized.
    (root/'data_manifest.json').write_bytes((parent/'data_manifest.json').read_bytes())
    archived={a:read(parent/'status.json')['jobs'][a].get('elapsed_seconds') for a in ['G3','G4','G5','scalar']}
    atomic(root/'runtime_estimate.json',{'archived_seconds':archived,'training_arm_minutes':[15,30],
        'scalar_minutes_estimated':archived['scalar']/5*3/60,'maximum_window_minutes':60,
        'note':'Two scalar endpoints are reused; LRA timing/features differ, so numerical preflight is mandatory'})
    atomic(root/'status.json',{'status':'PREPARED_NOT_ADMITTED','jobs':{a:{'status':'PENDING'} for a in spec['gpu_assignment']}})
    print(root);return root


def admission(root):
    root=Path(root);p=bound(root);spec=p['spec'];fleet={}
    # Read-only checks first. No runtime mutation and no old campaign resume.
    for arm,gpu in spec['gpu_assignment'].items():
        occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU occupied; admission waits for a valid six-GPU allocation: '+str(gpu))
        fleet[str(gpu)]=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
        if 'RTX A5000' not in fleet[str(gpu)]:raise ValueError('Wrong GPU class')
    test=root/'cpu-tests.xml';env={**os.environ,'PYTHONPATH':str(BASE)+':'+str(BASE/'src')}
    subprocess.run([sys.executable,'-m','pytest','tests/reft/a5000_lra_sprint','tests/reft/a5000_campaign','tests/reft/a5000',
        '-q','--junitxml='+str(test)],cwd=BASE,env=env,check=True)
    dest=work(root,'PREFLIGHT');start=time.monotonic()
    with lock(Path('/home/ubuntu/tung.nx/a5000-campaign-gpu0.lock')):
        with (root/'preflight.log').open('a') as f:
            result=subprocess.run([sys.executable,'-m','reft.a5000_lra_sprint.jobs','--root',str(root),'--arm','PREFLIGHT'],
                cwd=BASE,env={**env,'CUDA_VISIBLE_DEVICES':'0'},stdout=f,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode or not read(dest/'result.json')['result']['passed']:raise RuntimeError('GPU admission failed; see preflight.log')
    atomic(root/'admitted.json',{'passed':True,'utc':now(),'frozen_plan_sha256':sha(root/'frozen_plan.json'),
        'evidence':seal(root,[test,dest/'admission_report.json']),'fleet':fleet,'gpu0_admission_resident_seconds':time.monotonic()-start,
        'sprint_execution_window_has_not_started':True})
    print('ADMITTED',root)


def launch(root):
    import shlex
    root=Path(root);p=bound(root);a=read(root/'admitted.json')
    if a['frozen_plan_sha256']!=sha(root/'frozen_plan.json') or not verify(root,a['evidence']):raise ValueError('Admission binding invalid')
    session='a5000-'+root.name
    with lock(root/'launch.lock'):
        if subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            print('Already running',session);return
        if (root/'run_clock.json').exists():
            elapsed=time.time()-read(root/'run_clock.json')['started_epoch']
            if elapsed>=p['spec']['timing']['hard_window_seconds']:raise ValueError('Original60min window expired; no budget extension')
        if (root/'STOP').exists():raise ValueError('Stopped sprint; do not silently restart')
        cmd='cd '+shlex.quote(str(BASE))+' && PYTHONPATH='+shlex.quote(str(BASE)+':'+str(BASE/'src'))+' '+shlex.quote(sys.executable)+' -m reft.a5000_lra_sprint.runner control --root '+shlex.quote(str(root))+' >> '+shlex.quote(str(root/'sprint.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
        atomic(root/'launch.json',{'utc':now(),'session':session,'command':cmd,'backend':'tmux','reboot_recovery':'resume_within_original_window_only'})
        print(session)


def control(root,commands=None,interval=10):
    global STOP
    STOP=False;root=Path(root);p=read(root/'frozen_plan.json') if commands else bound(root);spec=p['spec'];running={};handles={}
    def stopping(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(root/'controller.lock'))
        clock=root/'run_clock.json'
        if not clock.exists():atomic(clock,{'started_epoch':time.time(),'utc':now(),'window_seconds':spec['timing']['hard_window_seconds']})
        started=read(clock)['started_epoch'];state=read(root/'status.json')
        state.update(status='RUNNING',pid=os.getpid(),gpu_seconds=state.get('gpu_seconds',0));last=time.monotonic();preliminary=False;final_report=False;lastresource=0
        event(root,'sprint_started',pid=os.getpid());print(now(),'sprint started',os.getpid(),flush=True)
        def persist():
            state['utc']=now();state['elapsed_seconds']=time.time()-started
            state['progress']={arm:read(root/'jobs'/arm/'worker_heartbeat.json') for arm in running if (root/'jobs'/arm/'worker_heartbeat.json').exists()}
            atomic(root/'status.json',state)
        def finish(arm):
            obj=running.pop(arm);code=obj['process'].wait();handles.pop(arm).close();rec=state['jobs'][arm]
            result=root/'jobs'/arm/'result.json';rec.update(exit_code=code,finished=now(),allocated_seconds=time.monotonic()-obj['started'])
            if code==0 and result.exists() and read(result)['status']=='COMPLETE':
                try:rec.update(status='COMPLETE',outputs=seal(root,[result]) if commands else output_seal(root,arm))
                except (ValueError,KeyError,FileNotFoundError) as e:rec.update(status='FAILED',reason='Output validation: '+str(e))
            else:rec.update(status='INCOMPLETE' if code==75 else 'FAILED',reason=read(result).get('reason') if result.exists() else 'No valid result')
            print(now(),arm,rec['status'],flush=True);event(root,'job_finished',arm=arm,status=rec['status'])
        try:
            for arm,gpu in spec['gpu_assignment'].items():
                work(root,arm);rec=state['jobs'][arm]
                if rec.get('status')=='COMPLETE':
                    if not verify(root,rec['outputs']):raise ValueError('Corrupt completed arm')
                    continue
                if rec.get('pid'):
                    try:os.kill(rec['pid'],0)
                    except ProcessLookupError:pass
                    else:raise RuntimeError('Existing child alive; no duplicate')
                if not commands:
                    try:stack.enter_context(lock(Path('/home/ubuntu/tung.nx')/f'a5000-campaign-gpu{gpu}.lock'))
                    except BlockingIOError:rec.update(status='BLOCKED_RESOURCE',reason='GPU lock occupied');continue
                    wait=time.monotonic()
                    while subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                        if time.monotonic()-wait>=spec['timing']['resource_wait_seconds']:break
                        time.sleep(2)
                    else:pass
                    if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                        rec.update(status='BLOCKED_RESOURCE',reason='Other GPU workload preserved');continue
                path=root/'logs'/(arm+'.log');path.parent.mkdir(exist_ok=True);h=path.open('a')
                cmd=commands[arm] if commands else [sys.executable,'-m','reft.a5000_lra_sprint.jobs','--root',str(root),'--arm',arm]
                child=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2'})
                running[arm]={'process':child,'started':time.monotonic(),'gpu':gpu};handles[arm]=h
                rec.update(status='RUNNING',pid=child.pid,gpu=gpu,started=now());persist()
                event(root,'job_started',arm=arm,gpu=gpu,pid=child.pid);print(now(),arm,'GPU',gpu,'started',flush=True)
            if not commands:
                from .report import report
                report(root,False)
            while running:
                current=time.monotonic();state['gpu_seconds']+=(current-last)*len(running);last=current
                for arm,obj in list(running.items()):
                    if obj['process'].poll() is not None:finish(arm)
                elapsed=time.time()-started
                if not commands:bound(root)
                if STOP or (root/'STOP').exists() or elapsed>=spec['timing']['soft_stop_seconds']:
                    if not (root/'STOP').exists():atomic(root/'STOP',{'utc':now(),'reason':'DEADLINE_OR_OWNER_SAFE_STOP'})
                    # Do not send SIGKILL, terminate CUDA, or interrupt a write.
                    # Worker ticks observe STOP between operations; last complete
                    # optimizer boundary remains recoverable.
                persist()
                if elapsed>=spec['timing']['preliminary_seconds'] and not preliminary:
                    if not commands:report(root,False)
                    preliminary=True;event(root,'minute30_report_ready')
                if elapsed>=spec['timing']['final_seconds'] and not final_report:
                    if not commands:report(root,True)
                    final_report=True
                if not commands and current-lastresource>=30:
                    sample=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    with (root/'resource_usage.csv').open('a') as f:
                        for line in sample.stdout.splitlines():f.write(now()+','+str(state['gpu_seconds'])+','+line+'\n')
                    lastresource=current
                time.sleep(interval)
            allcomplete=all(v['status']=='COMPLETE' for v in state['jobs'].values())
            state['status']='LRA_SPRINT_COMPLETE' if allcomplete else 'LRA_SPRINT_INCOMPLETE';persist()
            if not commands:report(root,True)
            event(root,'sprint_finished',status=state['status']);print(now(),state['status'],flush=True)
        finally:
            # Unexpected controller exceptions request safe worker exits without
            # corrupting an in-progress checkpoint; process PIDs remain recorded.
            if running:
                atomic(root/'STOP',{'utc':now(),'reason':'CONTROLLER_ERROR_SAFE_EXIT'})
                state['status']='CONTROLLER_ERROR_CHILDREN_STOP_REQUESTED';persist()
            for h in handles.values():h.close()
        return state


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','admit','launch','control','status','logs','stop','report']);p.add_argument('--spec',default=str(DEFAULT));p.add_argument('--root');a=p.parse_args()
    if a.action=='prepare':prepare(a.spec);return
    if not a.root:p.error('--root required')
    root=Path(a.root)
    if a.action=='admit':admission(root)
    elif a.action=='launch':launch(root)
    elif a.action=='control':control(root)
    elif a.action=='status':print((root/'status.json').read_text())
    elif a.action=='logs':subprocess.run(['tail','-n','80',str(root/'sprint.log')])
    elif a.action=='stop':atomic(root/'STOP',{'utc':now(),'reason':'OWNER_SAFE_STOP'})
    else:
        from .report import report
        report(root,True)
if __name__=='__main__':main()
