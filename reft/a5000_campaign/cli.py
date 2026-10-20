"""Freeze/admit/detach/status/stop/resume. Launch cannot bypass actual preflight."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .state import read,atomic,sha,identity,now,seal,lock
from .controller import source_manifest,check_bound

BASE=Path(__file__).resolve().parents[2]
DEFAULT=BASE/'configs/reft/a5000_campaign/campaign.json'


def prepare(config,run_id):
    from .data import prepare as prepare_data
    cfg=read(config)
    if '/' in run_id or not run_id or run_id in ['.','..']: raise ValueError('Invalid run identity')
    root=Path(cfg['output_base'])/run_id;root.mkdir(parents=True,exist_ok=True)
    with lock(root/'prepare.lock'):
        if (root/'frozen_plan.json').exists(): raise ValueError('Already frozen; use status/resume')
        if subprocess.check_output(['git','status','--porcelain'],cwd=BASE,text=True).strip(): raise ValueError('Commit source before prepare')
        # Honor existing scheduler policy; never bypass a detected scheduler.
        import shutil
        if any(shutil.which(x) for x in ['sbatch','qsub','bsub']): raise RuntimeError('Scheduler detected; execution integration required')
        doc=prepare_data(cfg,root)
        plan={'campaign_id':run_id,'created':now(),'created_epoch':time.time(),'config':cfg,'config_sha256':identity(cfg),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
            'source_path':str(BASE),'sources':source_manifest(),'data_sha256':sha(root/'data_manifest.json'),'admission_seal':{}}
        atomic(root/'frozen_plan.json',plan)
        atomic(root/'status.json',{'campaign_id':run_id,'status':'PREFLIGHT_PENDING','stages':{},'gpu_seconds':0})
        print(root)
    return root


def preflight(root):
    root=Path(root);plan=read(root/'frozen_plan.json')
    if source_manifest()!=plan['sources']: raise ValueError('Source changed since prepare')
    # Lock by physical GPU index across campaign IDs, and check other workloads.
    gpu=str(plan['config']['gpu_index'])
    with lock(Path('/home/ubuntu/tung.nx')/('a5000-campaign-gpu'+gpu+'.lock')):
        used=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if used: raise RuntimeError('GPU occupied; no interference: '+used)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(BASE)+':'+str(BASE/'src')}
        log=root/'preflight.log'
        with log.open('a') as f:
            result=subprocess.run([sys.executable,'-m','reft.a5000_campaign.worker','preflight','--root',str(root)],
                env=env,stdout=f,stderr=subprocess.STDOUT,timeout=plan['config']['resources']['preflight_seconds'])
        if result.returncode: raise RuntimeError('Actual GPU preflight failed; see '+str(log))
    report=read(root/'preflight/report.json')
    if not report['passed']: raise RuntimeError(report.get('blocker','Preflight rejected'))
    # CPU test evidence is generated on the actual remote environment.
    test=root/'cpu-tests.xml'
    result=subprocess.run([sys.executable,'-m','pytest','tests/reft/a5000_campaign','tests/reft/a5000','tests/reft/a5000_followup',
        '-q','--junitxml='+str(test)],cwd=BASE,env={**os.environ,'PYTHONPATH':str(BASE)+':'+str(BASE/'src')})
    if result.returncode: raise RuntimeError('Remote CPU suite failed')
    import xml.etree.ElementTree as ET
    suites=ET.parse(test).getroot()
    if any(int(s.attrib.get(k,'0')) for s in suites.iter('testsuite') for k in ['errors','failures']): raise ValueError('CPU XML rejected')
    plan['admission_seal']=seal(root,[root/'preflight/report.json',test]);atomic(root/'frozen_plan.json',plan)
    atomic(root/'admitted.json',{'source_commit':plan['source_commit'],'config_sha256':plan['config_sha256'],
        'data_sha256':plan['data_sha256'],'plan_sha256':sha(root/'frozen_plan.json'),'passed':True,'utc':now()})
    atomic(root/'status.json',{'campaign_id':root.name,'status':'ADMITTED','gpu_seconds':0,'stages':
        {s:{'status':'PENDING','attempts':0} for s in ['N1','N2','N3','N4','N5','REPORT']}})


def launch(root,resume=False):
    import shlex
    root=Path(root); plan=check_bound(root)
    admission=read(root/'admitted.json')
    if admission['plan_sha256']!=sha(root/'frozen_plan.json'): raise ValueError('Admission no longer bound')
    session='a5000-energy-'+root.name
    with lock(root/'launch.lock'):
        present=subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if present.returncode==0:
            print('Already running: '+session);return
        if (root/'status.json').exists() and read(root/'status.json')['status'].startswith('A5000_CAMPAIGN_COMPLETE'):
            print('Already finished; results preserved');return
        if resume: (root/'STOP').unlink(missing_ok=True)
        elif (root/'STOP').exists(): raise RuntimeError('Stopped; explicit resume required')
        # GPU lock spans entire detached controller lifetime.
        gpu=plan['config']['gpu_index']; wrapper=root/'supervise.py'
        wrapper.write_text('import fcntl,os,subprocess,sys\n'
            +'f=open('+repr('/home/ubuntu/tung.nx/a5000-campaign-gpu'+str(gpu)+'.lock')+',"a")\n'
            +'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
            +'u=subprocess.check_output(["nvidia-smi","-i",'+repr(str(gpu))+',"--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()\n'
            +'assert not u,"GPU occupied: "+u\n'
            +'sys.exit(subprocess.call([sys.executable,"-m","reft.a5000_campaign.controller","--root",'+repr(str(root))+']))\n')
        cmd='cd '+shlex.quote(str(BASE))+' && PYTHONPATH='+shlex.quote(str(BASE)+':'+str(BASE/'src'))+' '+shlex.quote(sys.executable)+' '+shlex.quote(str(wrapper))+' >> '+shlex.quote(str(root/'campaign.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',session,cmd],check=True)
        atomic(root/'launch.json',{'utc':now(),'session':session,'backend':'tmux','reboot_recovery':'manual_resume_only','command':cmd})
        print(session)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','launch','status','logs','summary','resume','stop']);
    p.add_argument('--config',default=str(DEFAULT));p.add_argument('--run-id');p.add_argument('--root');a=p.parse_args()
    if a.action=='prepare':
        if not a.run_id: p.error('--run-id required')
        prepare(a.config,a.run_id);return
    root=Path(a.root) if a.root else Path(read(a.config)['output_base'])/(a.run_id or 'campaign-v1')
    if a.action=='preflight':preflight(root)
    elif a.action in ['launch','resume']:launch(root,a.action=='resume')
    elif a.action=='stop':atomic(root/'STOP',{'utc':now(),'reason':'Owner stop command'});print('Stop requested; current logical batch is replayed if interrupted')
    elif a.action=='logs':subprocess.run(['tail','-n','80',str(root/'campaign.log')])
    elif a.action=='summary': print((root/'CAMPAIGN_STATUS.md').read_text())
    else:
        for name in ['status.json','worker_heartbeat.json','N2_decision.json','launch.json']:
            if (root/name).exists():print(name,(root/name).read_text())

if __name__=='__main__':main()
