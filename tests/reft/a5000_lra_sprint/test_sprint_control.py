import copy
import json
from pathlib import Path
import sys
import pytest
from reft.a5000_campaign.state import atomic,read,lock,sha
from reft.a5000_lra_sprint.runner import control,output_seal
from reft.a5000_lra_sprint.report import decision
from reft.a5000_lra_sprint.runtime import development_roles

SPEC=Path(__file__).resolve().parents[3]/'configs/reft/a5000_lra_sprint/sprint_v1.json'


def fixture(root,soft=10):
    spec=json.loads(SPEC.read_text());spec['timing'].update(soft_stop_seconds=soft,preliminary_seconds=.01,final_seconds=9)
    atomic(root/'frozen_plan.json',{'spec':spec})
    atomic(root/'data_manifest.json',{})
    atomic(root/'status.json',{'jobs':{a:{'status':'PENDING'} for a in spec['gpu_assignment']}})
    code='''import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);arm=sys.argv[2];mode=sys.argv[3]
if mode=='stop':
 while not (root/'STOP').exists():time.sleep(.01)
 # Simulate an atomic checkpoint write which must complete before safe exit.
 (root/'jobs'/arm/'checkpoint.complete').write_text('saved')
status='INCOMPLETE' if mode=='stop' else 'COMPLETE'
(root/'jobs'/arm/'result.json').write_text(json.dumps({'status':status,'result':{},'reason':'stop'}))
if mode=='stop':sys.exit(75)
'''
    return {a:[sys.executable,'-c',code,str(root),a,'success'] for a in spec['gpu_assignment']}


def test_parallel_complete_and_idempotent_resume(tmp_path):
    commands=fixture(tmp_path);s=control(tmp_path,commands,.01)
    assert s['status']=='LRA_SPRINT_COMPLETE' and all(j['status']=='COMPLETE' for j in s['jobs'].values())
    before=(tmp_path/'events.jsonl').read_text().count('job_started')
    assert control(tmp_path,commands,.01)['status']=='LRA_SPRINT_COMPLETE'
    assert (tmp_path/'events.jsonl').read_text().count('job_started')==before==6


def test_deadline_saves_before_exit(tmp_path):
    commands=fixture(tmp_path,.02)
    for cmd in commands.values():cmd[-1]='stop'
    s=control(tmp_path,commands,.01)
    assert s['status']=='LRA_SPRINT_INCOMPLETE' and (tmp_path/'STOP').exists()
    assert all(j['status']=='INCOMPLETE' for j in s['jobs'].values())
    assert all((tmp_path/'jobs'/a/'checkpoint.complete').exists() for a in commands)


def test_missing_outputs_are_not_complete(tmp_path):
    commands=fixture(tmp_path);commands['L4']=[sys.executable,'-c','pass']
    s=control(tmp_path,commands,.01)
    assert s['jobs']['L4']['status']=='FAILED' and s['status']=='LRA_SPRINT_INCOMPLETE'


def test_exclusive_controller_and_corruption(tmp_path):
    commands=fixture(tmp_path)
    with lock(tmp_path/'controller.lock'):
        with pytest.raises(BlockingIOError):control(tmp_path,commands,.01)
    control(tmp_path,commands,.01)
    (tmp_path/'jobs/L0/result.json').write_text('{}')
    with pytest.raises(ValueError,match='Corrupt'):control(tmp_path,commands,.01)


def test_output_validation_rejects_changed_checkpoint(tmp_path):
    job=tmp_path/'jobs/L0';job.mkdir(parents=True);ck=job/'adapter.pt';ck.write_bytes(b'weights')
    atomic(job/'result.json',{'status':'COMPLETE','result':{'checkpoint':str(ck),'checkpoint_sha256':sha(ck)}})
    ck.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Checkpoint'):output_seal(tmp_path,'L0')


def panel():
    return [{'system':k,'status':'COMPLETE','genuine_cs_mer':.06 if k!='L4' else .05,
             'noncs_mer':.04,'loops':0,'nontermination':0,'allocation_std':.01} for k in ['L0','L1','L2','L3','L4']]


def test_frozen_threshold_and_nonconstant_gate():
    spec=json.loads(SPEC.read_text());rows=panel()
    assert decision(rows,spec)['outcome']=='EXPLORATORY_POSITIVE_SIGNAL'
    rows[-1]['allocation_std']=0
    assert decision(rows,spec)['outcome']=='NO_MEANINGFUL_L4_BENEFIT'
    rows[-1]['allocation_std']=.01;rows[-1]['genuine_cs_mer']=.0599
    assert decision(rows,spec)['outcome']=='NO_MEANINGFUL_L4_BENEFIT'
    rows[-1]['noncs_mer']=None
    assert decision(rows,spec)['outcome']=='INCONCLUSIVE_MISSING_PRESERVATION'
    assert decision(rows[:-1],spec)['outcome']=='INCONCLUSIVE_INCOMPLETE'


def metadata():
    doc={'roles':{}}
    for role,size in [('A5000_T2000',2000),('A5000_V300',300),('A5000_C500',500)]:
        doc['roles'][role]=[{'official_split':'train',**{k:f'{role}-{k}-{i}' for k in
            ['utterance_id','audio_full_sha256','waveform_sha256','speaker_id','dialogue_id']}} for i in range(size)]
    return doc


def test_holdout_firewall():
    doc=metadata();roles=development_roles(doc)
    assert set(roles)=={'A5000_T2000','A5000_V300'}
    assert all('C500' not in r['utterance_id'] for rows in roles.values() for r in rows)


@pytest.mark.parametrize('key',['utterance_id','audio_full_sha256','waveform_sha256','speaker_id','dialogue_id'])
def test_leakage_is_rejected(key):
    doc=metadata();doc['roles']['A5000_V300'][0][key]=doc['roles']['A5000_T2000'][0][key]
    with pytest.raises(ValueError,match='leakage'):development_roles(doc)


def test_nontrain_membership_rejected():
    doc=metadata();doc['roles']['A5000_T2000'][0]['official_split']='test'
    with pytest.raises(ValueError,match='TRAIN membership'):development_roles(doc)


def test_report_exports_real_points_and_keeps_loop(tmp_path,monkeypatch):
    from reft.a5000_lra_sprint import report as reporting
    spec=json.loads(SPEC.read_text());spec['parent_root']=str(tmp_path/'parent')
    atomic(tmp_path/'frozen_plan.json',{'spec':spec,'parent_receipt':{'status':{'gpu_seconds':3600}}})
    atomic(tmp_path/'status.json',{'gpu_seconds':1800})
    def rec(cs,loop=0):
        return {'metrics':{'genuine_cs_mer':cs,'mer':cs,'noncs_mer':.04,'en_wer':.03,'zh_cer':.04,
            'seconds':2.,'rtf':.1,'insertions':3,'loops':loop,'nontermination':loop,'allocation_std':.01},
            'trainable_parameters':5122,'total_adapter_gate_parameters':5122,'checkpoint_bytes':42}
    parent={k:rec(.06,1 if k=='R_CE' else 0) for k in ['B0','R_CE','L_CE','LP_CE','LS_CE','R_WCE','L_WCE','LP_WCE','LS_WCE']}
    monkeypatch.setattr(reporting,'parent_records',lambda p:parent)
    monkeypatch.setattr(reporting,'screening_costs',lambda p:{k:100 for k in parent})
    for a in ['L0','L1','L2','L3','L4']:
        atomic(tmp_path/'jobs'/a/'result.json',{'status':'COMPLETE','allocated_seconds':10,'result':rec(.05 if a=='L4' else .06)})
    atomic(tmp_path/'jobs/AUDIT/result.json',{'status':'COMPLETE','result':{'verified':True}})
    reporting.report(tmp_path,True)
    assert (tmp_path/'reports/combined.csv').exists() and (tmp_path/'reports/combined.tex').exists()
    assert len(list((tmp_path/'figures').glob('*.pdf')))==3
    assert read(tmp_path/'reports/decision.json')['outcome']=='EXPLORATORY_POSITIVE_SIGNAL'
    assert '|R_CE|' in (tmp_path/'reports/combined.md').read_text()
    assert '1/1' in (tmp_path/'reports/combined.md').read_text()
    assert read(tmp_path/'reports/accounting.json')['sprint_controller_allocated_gpu_hours']==.5


def test_worker_module_is_executable():
    import subprocess
    proc=subprocess.run([sys.executable,'-m','reft.a5000_lra_sprint.jobs','--help'],capture_output=True,text=True,check=True)
    assert '--arm' in proc.stdout and 'PREFLIGHT' in proc.stdout
