"""Durable automatic completion of the authorized campaign, independent of the Mac/Codex connection.

Waits for/resumes the primary O1 worker, then bounded objective engineering checks, conditional frozen C150
confirmation, provenance checks and reports. Never launches full O2/O3 training or a hyperparameter sweep.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
# Keep the supervisor free of torch/CUDA imports: it must not occupy the worker's GPU.
import yaml
C=yaml.safe_load((ROOT/'configs/reft/a5000_e1.yaml').read_text())
OUT=Path(C['output_root']).expanduser()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def state(status,phase,**extra):
    write(OUT/'auto/status.json',{'status':status,'phase':phase,'supervisor_pid':os.getpid(),
        'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra})


def stopped():return (OUT/'AUTO_STOP').exists() or (OUT/'STOP').exists()


def run(module,phase,cpu=False):
    if stopped():raise InterruptedError('Explicit stop sentinel')
    state('RUNNING',phase)
    env=os.environ.copy()
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    log=OUT/'logs'/f'auto_{phase}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        command=[sys.executable,'-u','-m',module]
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(OUT/'auto/child.json',{'pid':p.pid,'phase':phase,'command':command,'log':str(log)})
        code=p.wait()
    write(OUT/'auto'/f'{phase}_exit.json',{'exit_code':code,'log':str(log),'pid':p.pid})
    return code


def main():
    (OUT/'auto').mkdir(parents=True,exist_ok=True)
    with (OUT/'auto/SUPERVISOR.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('RUNNING','waiting_for_primary')
        try:
            while subprocess.run(['tmux','has-session','-t','a5000-e1'],capture_output=True).returncode==0:
                if stopped():raise InterruptedError('Stop requested while primary was running')
                time.sleep(10)
            if not (OUT/'comparison.json').exists():
                if stopped():raise InterruptedError('Primary paused by stop request')
                previous=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
                if previous.get('status')=='FAILED':raise RuntimeError('Primary failed; inspect its logs before changing/retrying code')
                # Resume a cleanly paused/interrupted authorized primary within the original budget.
                spent=json.loads((OUT/'budget.json').read_text())['seconds']
                if spent>=C['budget']['gpu_hours']*3600:raise InterruptedError('Aggregate budget exhausted')
                state('RUNNING','resuming_primary')
                for stage in ['preflight','campaign']:
                    with (OUT/'logs/auto_primary_resume.log').open('a') as log:
                        code=subprocess.run([str(ROOT/'scripts/a5000/e1.sh'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                    if code:raise RuntimeError(f'Primary {stage} exited {code}')
            if not (OUT/'comparison.json').exists():raise InterruptedError('Primary remains paused/incomplete')
            if run('reft.a5000_followup.summarize','primary_summary',cpu=True):raise RuntimeError('Primary artifact aggregation failed')
            engineering_code=run('reft.a5000_followup.preflight','objective_engineering')
            spent=json.loads((OUT/'budget.json').read_text())['seconds']
            remaining=C['budget']['gpu_hours']*3600-spent
            confirmation='SKIPPED_BUDGET'
            if remaining>=1800 and not stopped():
                code=run('reft.a5000_followup.confirm','confirmation')
                confirmation='COMPLETE' if code==0 and (OUT/'confirmation/comparison.json').exists() else 'FAILED_OR_PAUSED'
            if run('reft.a5000_followup.selfcheck','contract_selfcheck',cpu=True):raise RuntimeError('Contract self-check failed')
            if run('reft.a5000_followup.final_reports','reports',cpu=True):raise RuntimeError('Final report generation failed')
            state('COMPLETE','finished',primary_complete=True,objective_engineering_exit=engineering_code,
                  confirmation=confirmation,reports=str(OUT/'reports'))
        except InterruptedError as e:
            state('PAUSED','resumable_boundary',reason=str(e))
        except Exception as e:
            state('FAILED','inspect_logs',error=repr(e))
            raise


if __name__=='__main__':main()
