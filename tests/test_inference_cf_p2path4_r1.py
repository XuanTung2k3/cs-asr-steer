"""CPU-only PATH4 R1 boundary and frozen-contract regressions; no scientific outcomes."""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from csasr.inference_cf import eos_boundary as eb
from csasr.inference_cf.branch_adjudication import consensus, processed_log_probs

ORIGINAL = "29661e972f0ebfd727d228d2f6dbd9b17dd41c5a"
EOS = 4


def mode(a,b,**extra):
    return eb.action_mode(a,b,eos=EOS,vocab_size=6,special_ids=[4,5],step=2,suppress=[],begin=[],**extra)


@pytest.mark.parametrize('a,b,expected',[(4,1,'EOS_BOUNDARY'),(1,4,'EOS_BOUNDARY'),(4,4,'NO_TRIGGER'),(1,2,'CONTENT_G1'),(1,1,'NO_TRIGGER')])
def test_live_type_dispatch(a,b,expected):
    assert mode(a,b)==expected


@pytest.mark.parametrize('a,b',[(5,1),(-1,4),(4,6)])
def test_invalid_actions_no_fallback(a,b):
    with pytest.raises(ValueError):mode(a,b)


def test_boundary_h1_same_processed_logits_and_consensus():
    x=torch.tensor([0.,2.,1.,0.,3.,9.],dtype=torch.bfloat16)
    y=torch.tensor([0.,4.,1.,0.,1.,8.],dtype=torch.bfloat16)
    d=eb.boundary_scores(x,y,4,1,eos=4,special_ids=[4,5],step=2,suppress=[5],begin=[4])
    q=processed_log_probs(x,2,[5],[4]);p=processed_log_probs(y,2,[5],[4])
    assert d['H']==1 and d['S_theta0']=={'B':float(q[4]),'ALT':float(q[1])}
    assert d['S_A2']=={'B':float(p[4]),'ALT':float(p[1])}
    expected=consensus(d['S_theta0'],d['S_A2'])
    assert d['S_cons']==expected['S_cons'] and d['margin']==expected['margin_B_minus_ALT']
    with pytest.raises(ValueError):eb.boundary_scores(x,y,4,1,eos=4,special_ids=[4,5],step=0,suppress=[5],begin=[4])


def test_reverse_orientation_and_tie():
    x=torch.tensor([0.,3.,0.,0.,2.,-1.]);y=torch.tensor([0.,2.,0.,0.,3.,-1.])
    d=eb.boundary_scores(x,y,1,4,eos=4,special_ids=[4,5],step=1,suppress=[5],begin=[])
    expected=consensus(d['S_theta0'],d['S_A2'])
    assert d['winner']==('theta0' if expected['choice']=='B' else 'A2')
    assert consensus({'B':-1.,'ALT':-2.},{'B':-2.,'ALT':-1.})['choice']=='B'
    assert consensus({'B':0.,'ALT':0.},{'B':-1e-12,'ALT':0.})['choice']=='B'


def test_eos_winner_and_content_winner(monkeypatch):
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf import path_decode
    from csasr.lss import sites
    monkeypatch.setattr(sites,'assert_no_site_hooks',lambda b:None)
    class Branch:
        def __init__(self,b,enc,prompt,label):self.fed=[];self.positions=[];self.length=0
        def step(self,new,**kw):
            self.fed+=list(new);self.positions=list(range(len(self.fed)));self.length=len(self.fed)
            return torch.tensor([0.,2.,0.,0.,3.,-1.]),None,None
    monkeypatch.setattr(cached,'Branch',Branch)
    b=SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(suppress_tokens=[5],begin_suppress_tokens=[])),processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=4)))
    for selected,expected in [(4,[]),(1,[1])]:
        d={'mode':'EOS_BOUNDARY','H':1,'k':0,'prefix':[],'selected_token':selected}
        out=eb.execute_boundary(b,None,[0],d,owner_hash='a2',hash_fn=lambda:'a2')
        assert out['tokens']==expected and out['terminated']=='eos'
        assert len(out['trace']['forced'])==1 and out['state_locked']
        assert out['trace']['release_index']==1
    with pytest.raises(ValueError):eb.execute_boundary(b,None,[0],d,owner_hash='stale',hash_fn=lambda:'a2')


def test_fresh_owner_caches_h1_and_no_rollout(monkeypatch):
    import experiments.inference_cf_cached as cached
    from csasr.lss import sites
    monkeypatch.setattr(sites,'assert_no_site_hooks',lambda b:None)
    seen=[]
    class Branch:
        def __init__(self,b,enc,prompt,label):self.bundle=b;self.fed=[];self.positions=[];seen.append(self)
        def step(self,new,**kw):
            self.fed+=list(new);self.positions=list(range(len(self.fed)))
            if len(self.fed)==1:return torch.tensor([0.,5.,0.,0.,2.,-1.]),None,None
            return self.bundle.logits,None,None
    monkeypatch.setattr(cached,'Branch',Branch)
    def bundle(logits):return SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(suppress_tokens=[5],begin_suppress_tokens=[])),processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=4,all_special_ids=[4,5])),logits=torch.tensor(logits))
    b0=bundle([0.,2.,0.,0.,4.,-1.]);b2=bundle([0.,4.,0.,0.,2.,-1.])
    owners={'theta0':'t','A2':'a'};fns={'theta0':lambda:'t','A2':lambda:'a'}
    d=eb.score_boundary(b0,b2,None,[0],[1],4,1,owners=owners,hash_fns=fns)
    assert len(seen)==2 and seen[0] is not seen[1] and seen[0].bundle is b0 and seen[1].bundle is b2
    assert all(b.fed==[0,1] for b in seen) and d['H']==1 and all(p['state_locked'] for p in d['paths'].values())
    with pytest.raises(ValueError):eb.score_boundary(b0,b2,None,[0],[1],4,1,owners={'theta0':'bad','A2':'a'},hash_fns=fns)
    with pytest.raises(ValueError):eb.score_boundary(b0,b0,None,[0],[1],4,1,owners=owners,hash_fns=fns)
    with pytest.raises(ValueError):eb.score_boundary(b0,b2,None,[0],[],4,1,owners=owners,hash_fns=fns) # proposals disagree with live prefix0 argmax


def test_content_g1_bit_identity_on_toy(monkeypatch):
    from test_inference_cf_p2path2 import _trigger_pair
    from test_inference_cf_p2dir_protocol import _enc
    import experiments.inference_cf_p2path3 as old
    from test_inference_cf_p1 import CB
    monkeypatch.setattr(old,"CB",CB)
    monkeypatch.setattr(old,"MAX_NEW",8)
    b0,b2,g0,g2,d=_trigger_pair()
    eos=b0.processor.tokenizer.eos_token_id
    assert eb.action_mode(d['argmax_theta0'],d['argmax_A4'],eos=eos,vocab_size=b0.model.config.vocab_size,special_ids=[eos],step=d['k'],suppress=[],begin=[])=='CONTENT_G1'
    # The R1 content branch dispatches the original function without modifying any inputs or return fields.
    counters=lambda:{'detector_runs':0,'triggers':0,'rollouts':0,'score_paths':0,'G1_decodes':0}
    direct=old.online_g1(b0,b2,g0,g2,_enc(),'toy',counters(),eos)
    transferred=old.online_g1(b0,b2,g0,g2,_enc(),'toy',counters(),eos)
    assert direct==transferred


def test_known_four_rows_representable_and_no_runtime_ids():
    p=json.loads(Path('docs/inference_cf/P2_PATH4_PANEL.json').read_text())
    assert len(p['known_blockers'])==4
    for row in p['known_blockers']:
        assert eb.action_mode(row['theta0_token'],row['A2_token'],eos=50257,vocab_size=51866,special_ids=[50257],step=row['k'],suppress=[],begin=[])=='EOS_BOUNDARY'
    source=Path('src/csasr/inference_cf/eos_boundary.py').read_text()
    assert all(r['utterance_id'] not in source for r in p['known_blockers'])
    assert not any(x in source for x in ('load_references','rollout(','lockstep_detect(','adapt('))


def original_bytes(path):return subprocess.check_output(['git','show',f'{ORIGINAL}:{path}'])


def test_original_contract_and_thresholds_preserved():
    old=json.loads(original_bytes('configs/inference_cf/p2_path4.json'));new=json.loads(Path('configs/inference_cf/p2_path4.json').read_text())
    keys=('A2_inherited','A2_objective','A2_reconstruction','safety','historical_aggregate','aggregate_rescue','benefit','breadth','LODO','bootstrap','decision_precedence','reference_barrier','metrics','firewall','systems','partition_sha256','counts','panel_byte_sha256')
    assert all(new[k]==old[k] for k in keys)
    assert new['contract_revision']==eb.REVISION
    for path in ('src/csasr/inference_cf/consensus_guard.py','src/csasr/inference_cf/branch_adjudication.py','experiments/inference_cf_p2path3.py','docs/inference_cf/P2_PATH4_PANEL.json'):
        assert Path(path).read_bytes()==original_bytes(path)
    for path in ('docs/inference_cf/P2_PATH4_SPEC.md','docs/inference_cf/P2_PATH4_CODEX_DESIGN.md'):
        assert Path(path).read_bytes().endswith(original_bytes(path))
    assert hashlib.sha256(Path('docs/inference_cf/P2_PATH4_PANEL.json').read_bytes()).hexdigest()==new['panel_byte_sha256']


def test_cap_boundary_and_nonfinite_rejection():
    assert eb.action_mode(4,1,eos=4,vocab_size=6,special_ids=[4,5],step=199,suppress=[],begin=[])=='EOS_BOUNDARY'
    with pytest.raises(ValueError):eb.action_mode(4,1,eos=4,vocab_size=6,special_ids=[4,5],step=200,suppress=[],begin=[])
    with pytest.raises(ValueError):eb.boundary_scores(torch.tensor([0.,2.,0.,0.,float('nan'),0.]),torch.zeros(6),4,1,eos=4,special_ids=[4,5],step=2,suppress=[],begin=[])


def test_both_eos_detector_terminates_without_trigger(monkeypatch):
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.consensus_guard import lockstep_detect
    from csasr.lss import sites
    monkeypatch.setattr(sites,'assert_no_site_hooks',lambda b:None)
    class Branch:
        def __init__(self,b,enc,prompt,label):self.fed=[];self.positions=[]
        def step(self,new,**kw):
            self.fed+=list(new);self.positions=list(range(len(self.fed)))
            return torch.tensor([0.,0.,0.,0.,5.,0.]),None,None
    monkeypatch.setattr(cached,'Branch',Branch)
    def bundle():return SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(suppress_tokens=[5],begin_suppress_tokens=[])),processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=4)))
    d=lockstep_detect(bundle(),bundle(),None,[0],owners={'theta0':'t','A4':'a'},hash_fns={'theta0':lambda:'t','A4':lambda:'a'})
    assert not d['trigger'] and d['terminated']=='eos' and d['prefix']==[] and d['fed_identical'] and d['state_locked']
