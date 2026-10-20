import json
import os
from pathlib import Path
import sys
import time
import pytest
from reft.a5000_campaign.state import atomic,read,identity
from reft.a5000_campaign.parallel_jobs import graph,pick_ready
from reft.a5000_campaign.parallel_controller import run


def test_graph_gates_do_not_wait_for_unrelated_baselines():
    cfg=read(Path('configs/reft/a5000_campaign/multigpu_v2.json'));j=graph(cfg)
    s={k:{'status':'PENDING'} for k in j}
    for k in ['B0','CE_R_lr0_s42','CE_R_lr1_s42','select_CE_R']:s[k]['status']='COMPLETE'
    ready=pick_ready(j,s)
    assert all(k in ready for k in ['G3','G4','G5','G8','scalar'])
    assert 'N5' not in ready
    assert len([k for k,v in j.items() if v['kind']=='train'])==16
    # No science/inference recipe is changed by scheduling.
    for k in j:
        assert k not in j[k]['dependencies']


def fake(tmp_path):
    cfg={'gpu_indices':[0,1],'resources':{'gpu_hours':1,'reserve_gpu_hours':0,'wall_hours':1,'stage_seconds':10}}
    atomic(tmp_path/'frozen_plan.json',{'config':cfg,'config_sha256':identity(cfg),'source_commit':'synthetic',
        'data_sha256':'synthetic','created_epoch':time.time()})
    atomic(tmp_path/'data_manifest.json',{'synthetic':True})
    jobs={k:{'id':k,'kind':'train','priority':1,'args':{},'dependencies':[]} for k in ['A','B']}
    jobs['C']={'id':'C','kind':'train','priority':2,'args':{},'dependencies':['A','B']}
    atomic(tmp_path/'job_graph.json',jobs);cmds={}
    for k in jobs:
        p=tmp_path/'jobs'/k/'result.json'
        code='import time,json,pathlib; time.sleep(.2); pathlib.Path('+repr(str(p))+').write_text(json.dumps({"status":"COMPLETE","result":{}}))'
        cmds[k]=[sys.executable,'-c',code]
    return cmds


def test_parallel_actual_subprocesses_dependencies_and_resume(tmp_path):
    commands=fake(tmp_path);s=run(tmp_path,commands,interval=.02)
    assert all(v['status']=='COMPLETE' for v in s['jobs'].values())
    assert s['jobs']['A']['gpu']!=s['jobs']['B']['gpu']
    assert s['jobs']['C']['started']>=s['jobs']['A']['finished']
    assert s['jobs']['C']['started']>=s['jobs']['B']['finished']
    before=s['gpu_seconds'];s=run(tmp_path,commands,interval=.02)
    assert all(v['attempts']==1 for v in s['jobs'].values())
    assert s['gpu_seconds']>=before
    (tmp_path/'jobs/A/result.json').write_text('{}')
    with pytest.raises(ValueError):run(tmp_path,commands,interval=.02)


def test_parallel_failure_does_not_cancel_independent_job(tmp_path):
    commands=fake(tmp_path);commands['A']=[sys.executable,'-c','raise RuntimeError("synthetic OOM")']
    s=run(tmp_path,commands,interval=.02)
    assert s['jobs']['A']['status']=='FAILED' and s['jobs']['B']['status']=='COMPLETE'
    assert s['jobs']['C']['status']=='BLOCKED'


def test_parallel_aggregate_budget_not_wall_clock(tmp_path):
    commands=fake(tmp_path);p=read(tmp_path/'frozen_plan.json');p['config']['resources']['gpu_hours']=.00001;atomic(tmp_path/'frozen_plan.json',p)
    s=run(tmp_path,commands,interval=.02)
    assert s['status']=='A5000_CAMPAIGN_RESOURCE_LIMIT_REACHED'
    assert not all(v['status']=='COMPLETE' for v in s['jobs'].values())


def test_multigpu_changes_execution_only_and_predicts_core_scope():
    from reft.a5000_campaign.parallel_jobs import estimate_core
    a=read(Path('configs/reft/a5000_campaign/campaign.json'))
    b=read(Path('configs/reft/a5000_campaign/multigpu_v2.json'))
    for key in ['training','gate','data','pilot_rule','decode','metrics','confirmation']:
        assert a[key]==b[key]
    est=estimate_core(graph(b),6)
    assert 0<est['core_wall_hours']<7
    assert est['aggregate_gpu_hours']<72
    assert est['core_wall_hours']<estimate_core(graph(b),4)['core_wall_hours']
