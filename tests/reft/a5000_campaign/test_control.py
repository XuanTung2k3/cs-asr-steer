import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from reft.a5000_campaign.state import atomic,read,identity,sha,verify,lock
from reft.a5000_campaign.data import validate,rows,ROLE_NAMES
from reft.a5000_campaign.controller import run,STAGES


def roles():
    return {name:[{'utterance_id':str(i),'official_split':'train','dialogue_id':str(i),'speaker_id':str(i),
        'corpus_speaker_id':str(i),'audio_full_sha256':str(i),'waveform_sha256':str(i)}] for i,name in enumerate(ROLE_NAMES)}


def test_split_integrity_and_firewall(tmp_path):
    r=roles();assert validate(r)
    r['A5000_C500'][0]['speaker_id']='0'
    with pytest.raises(ValueError):validate(r)
    r=roles();r['A5000_V300'][0]['official_split']='test'
    with pytest.raises(ValueError):validate(r)
    with pytest.raises(ValueError):rows({},tmp_path,'A5000_C500')
    with pytest.raises(ValueError):rows({},tmp_path,'official_test')


def test_atomic_and_lock(tmp_path):
    p=tmp_path/'a.json';atomic(p,{'valid':1});before=sha(p)
    with pytest.raises(ValueError):atomic(p,{'bad':float('nan')})
    assert sha(p)==before
    with lock(tmp_path/'lock'):
        with pytest.raises(BlockingIOError):
            with lock(tmp_path/'lock'):pass
    assert verify(tmp_path,{'a.json':before});p.write_text('bad');assert not verify(tmp_path,{'a.json':before})


def fake(tmp_path,seconds=10):
    cfg={'resources':{'gpu_hours':1,'wall_hours':1,'stage_seconds':seconds},'gpu_index':0}
    atomic(tmp_path/'frozen_plan.json',{'config':cfg,'config_sha256':identity(cfg),'data_sha256':'fake','source_commit':'synthetic','created_epoch':time.time()})
    cmds={}
    for stage in STAGES:
        code='from pathlib import Path; import json; p=Path('+repr(str(tmp_path/'stage_results'/(stage+'.json')))+'); p.parent.mkdir(exist_ok=True); p.write_text(json.dumps({"status":"COMPLETE"}))'
        cmds[stage]=[sys.executable,'-c',code]
    return cmds


def test_controller_restart_no_duplicates_and_corruption(tmp_path):
    cmds=fake(tmp_path);a=run(tmp_path,cmds,True,.02)
    assert all(v['status']=='COMPLETE' for v in a['stages'].values())
    b=run(tmp_path,cmds,True,.02)
    assert all(v['attempts']==1 for v in b['stages'].values())
    (tmp_path/'stage_results/N1.json').write_text('{}')
    with pytest.raises(ValueError):run(tmp_path,cmds,True,.02)


def test_failed_gate_stage_does_not_prevent_objective_stage(tmp_path):
    cmds=fake(tmp_path);cmds['N2']=[sys.executable,'-c','raise RuntimeError("infrastructure failure")']
    a=run(tmp_path,cmds,True,.02)
    assert a['stages']['N3']['status']=='COMPLETE'
    assert a['stages']['N5']['status']=='BLOCKED'
    assert a['stages']['REPORT']['status']=='COMPLETE'


def test_scientific_no_go_is_complete_and_continues(tmp_path):
    cmds=fake(tmp_path)
    cmds['N2']=[sys.executable,'-c','from pathlib import Path; import json; Path('+repr(str(tmp_path/'stage_results/N2.json'))+').write_text(json.dumps({"status":"COMPLETE","decision":"N2_NO_GO"}))']
    assert run(tmp_path,cmds,True,.02)['stages']['N3']['status']=='COMPLETE'


def test_budget_and_stop(tmp_path):
    cmds=fake(tmp_path);p=read(tmp_path/'frozen_plan.json');p['config']['resources']['gpu_hours']=0;atomic(tmp_path/'frozen_plan.json',p)
    a=run(tmp_path,cmds,True,.02)
    assert a['status']=='A5000_CAMPAIGN_RESOURCE_LIMIT_REACHED'
    assert a['stages']['N1']['status']=='PENDING'


def test_actual_process_termination_restart(tmp_path):
    cmds=fake(tmp_path)
    child=tmp_path/'exercise.py'
    child.write_text('from pathlib import Path\nfrom reft.a5000_campaign.controller import run\n'+
        'run(Path('+repr(str(tmp_path))+'),'+repr(cmds)+',True,.02)\n')
    # Deliberately terminate controller while its real child process is active.
    cmds['N1']=[sys.executable,'-c','import time;time.sleep(10)']
    child.write_text('from pathlib import Path\nfrom reft.a5000_campaign.controller import run\n'+
        'run(Path('+repr(str(tmp_path))+'),'+repr(cmds)+',True,.02)\n')
    proc=subprocess.Popen([sys.executable,str(child)],env={**os.environ,'PYTHONPATH':str(Path.cwd())+':'+str(Path.cwd()/'src')})
    for _ in range(100):
        if (tmp_path/'status.json').exists() and read(tmp_path/'status.json').get('child_pid'):break
        time.sleep(.05)
    assert proc.poll() is None
    proc.terminate();proc.wait(timeout=10)
    assert read(tmp_path/'status.json')['status']=='STOPPED_BY_USER'
    done=run(tmp_path,fake(tmp_path),True,.02)
    assert done['stages']['N1']['status']=='COMPLETE'
