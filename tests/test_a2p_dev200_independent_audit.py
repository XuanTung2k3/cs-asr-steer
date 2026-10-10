"""CPU checks of the independent auditor, using tiny randomly initialized Whisper only."""
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'tests')]
from experiments.a2p_dev200_independent_replay import independent_terms, allowed_lp, candidate_audit
from experiments.a2p_dev200_independent_analysis import groups, shares, one, compare
from csasr.inference_cf import ttls as L
from csasr.inference_cf.episodic_tta import LNGuard, decoder_ln_names, teacher_logits
from test_ttls_r1 import _bundle, _encode


def test_independent_loss_gradients_and_two_step_state():
    from experiments.inference_cf_a2p_dev200 import b3_loss_fn
    b = _bundle(); _, enc = _encode(b)
    guard = LNGuard(b.model, decoder_ln_names(b.model))
    s = {'y_B': [20, 21, 22, 23], 'y_A': [20, 25, 22, 23], 'y_A_valid_mask': [True]*4}
    # Tiny tokenizer prompts are substituted in both independently computed and frozen paths.
    from test_ttls_r1 import CB
    original = L.CB
    L.CB = CB
    try:
        with torch.no_grad():
            q = [v.detach() for v in allowed_lp(teacher_logits(b.model, enc, CB, s['y_B'], None), 4, [], [])]
        ep = L.Episode(b, guard, enc, 'LN', layer=1)
        ctx = SimpleNamespace(suppress=[], begin=[], eos=2, cB=CB)
        ref_fn = b3_loss_fn(ctx, s, {'logq': q}, [0, 2, 3])
        def own_fn(e):
            ce, p = independent_terms(e, s, q, [0, 2, 3], [], [])
            return {'loss': ce+p, 'parts': {'CE': float(ce.detach()), 'P': float(p.detach())}}
        a, c = own_fn(ep), ref_fn(ep)
        assert a['parts']['P'] == c['parts']['P'] == 0
        ga = torch.autograd.grad(a['loss'], list(ep.params.values()))
        gc = torch.autograd.grad(c['loss'], list(ep.params.values()))
        assert all(torch.equal(x,y) for x,y in zip(ga,gc))
        own = ep.run(own_fn,{})
        fresh = L.Episode(b, guard, enc, 'LN', layer=1)
        assert not fresh.opt.state
        primary = fresh.run(ref_fn,{})
        assert {k:v for k,v in own['log'].items() if k!='adapt_sec'} == {k:v for k,v in primary['log'].items() if k!='adapt_sec'}
        assert all(torch.equal(ep.params[k],fresh.params[k]) for k in ep.params)
        assert guard.verify() and all(p.grad is None for p in b.model.parameters())
    finally:
        L.CB = original


def test_candidate_membership_precedes_acoustic_acceptance():
    lp = {k:np.full((4,8), -10.,dtype=np.float64) for k in ('cB_clean','cE_clean','cB_null','cE_null')}
    s={'y_B':[3,3,3,3], 'y_B_valid_mask':[True,True,True,False], 'utf8_ok':[True]*4}
    lp['cB_clean'][:,3]=math.log(.8)
    lp['cE_clean'][1,4]=-.1
    # Candidate does not pass acoustic test: still excluded from preservation S.
    for k in ('cB_null','cE_null'):lp[k][1,4]=lp[k.replace('null','clean')][1,4]
    c = candidate_audit(lp,s,{4},2)
    assert c['M']==[1] and c['accepted']==[] and c['S']==[0,2]


def test_independent_counts_grouping_and_replay_firewall():
    a,b=one('我 go home','我 去'),one('我 go home','我 go home')
    e=compare(a,b,'u','d',False)
    assert a['counts'][0]-b['counts'][0] == sum(v['type']=='correction' for v in e['events'])-sum(v['type']=='corruption' for v in e['events'])
    events=[{'id':'u','dialogue':'d','type':'correction','index':i,'lexical_kind':'genuine_wrong_language','reference':'word','before':'汉','after':'word'} for i in [1,2,4]]
    assert [v['units'] for v in groups(events)]==[2,1]
    assert shares({'u':0})['largest_share'] is None
    src=(ROOT/'experiments/a2p_dev200_independent_replay.py').read_text()
    mods=[n.module or '' for n in ast.walk(ast.parse(src)) if isinstance(n,ast.ImportFrom)]
    assert not any('evaluate' in m or 'analysis' in m or 'evaluation' in m for m in mods)
    saved=json.loads((ROOT/'results/inference_cf/a2p_dev200_independent_audit/independent_metrics.json').read_text())
    assert not saved['published_mismatches']
    for pop in ('NEW200','FIXED100','FULL300'):
        for arm in ('B0','B1','B2','B3'):
            assert sum(r['systems'][arm]['counts'][2] for r in saved[pop]['rows']) == saved[pop]['metrics'][arm]['mixed_errors']
