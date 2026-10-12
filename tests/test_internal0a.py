"""Focused CPU contracts: no pretrained-model inference and no gold inputs."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from csasr.inference_cf import internal0a as I
from csasr.inference_cf.abts0_k1 import token_intervals


def word(branch, view, crop, iv=(10000, 16000), surface="friends", block=(0, 0)):
    return dict(surface=surface, normalized=surface.casefold(), branch=branch, view_id=view,
                crop=list(crop), interval=list(iv), block=list(block), boundary_truncation_possible=False,
                capped=False, valid_utf8=True, association="ASSOCIATED", tok_first=1)


def test_grid_covers_heard_without_reference_timing_or_end_duplication():
    g = I.window_grid(5 * 16000)
    assert [x['short'] for x in g] == [[0, 32000], [16000, 48000], [32000, 64000], [48000, 80000], [64000, 80000]]
    assert g[0]['long'] == [0, 48000]
    assert g[-1]['long'] == [40000, 80000]
    assert len(I.window_grid(480001)) == 30
    assert len(I.window_grid(10000)) == 1
    with pytest.raises(ValueError):
        I.window_grid(0)


def test_dedup_retains_grid_memberships():
    g = [{'grid_index': 0, 'short': [0, 10], 'long': [0, 30]},
         {'grid_index': 1, 'short': [10, 20], 'long': [0, 30]}]
    v = I.physical_views(g)
    assert len(v) == 6
    assert next(x for x in v if x['branch'] == '4_AUTO')['grid_indices'] == [0, 1]


def test_association_accepts_Han_Latin_t0_without_onset_gate():
    u = [{'surface': '学', 'interval': [0, 100], 'tok_first': 0},
         {'surface': 'wrong', 'interval': [100, 200], 'tok_first': 1}]
    a = I.associate([10, 190], u)
    assert a['block'] == [0, 1] and a['query_t'] == 0 and a['association'] == 'ASSOCIATED'
    assert I.associate(None, u)['association'] == 'UNLOCATED'


def test_discovery_keeps_boundary_cap_and_unlocated_words_and_full_audio_offsets():
    v = {'view_id': 'v', 'branch': '2_EN', 'bounds': [16000, 48000], 'tokens': [10, 11], 'terminated': 'cap'}
    t = {'status': 'OK', 'n_timed': 1, 'start': [0], 'end': [3000]}
    w = I.words_from_view(v, [b' friends', b' hello'], t, [])
    assert len(w) == 2 and w[0]['interval'] == [16000, 19000]
    assert w[0]['boundary_truncation_possible'] and w[1]['interval'] is None and w[1]['capped']


def test_union_does_not_require_ZH_agreement_and_selection_is_separate():
    b = [{'surface': '朋', 'tok_first': 2}]
    w = [word('2_EN', 'a', [0, 32000]), word('4_AUTO', 'b', [0, 64000]),
         word('4_ZH', 'c', [0, 64000], surface='other')]
    r = I.summarize_candidates(w, b)
    assert len(r['candidates']) == 2
    assert r['selected'] is not None
    chosen = next(p for p in r['candidates'] if p['candidate_id'] == r['selected'])
    assert chosen['surface'] == 'friends' and '4_ZH' not in chosen['branches']


def test_identical_clipped_audio_is_not_cross_context_support():
    r = I.summarize_candidates([word('2_EN', 'a', [0, 20000]), word('4_AUTO', 'b', [0, 20000])], [{'surface':'朋'}])
    assert len(r['words']) == 2 and r['selected'] is None


def test_ZH_comparator_does_not_veto_primary_block_or_inflate_rank():
    w=[word('2_EN','a',[0,32000]), word('4_AUTO','b',[0,64000]),
       word('4_ZH','c',[0,64000],block=(1,1))]
    r=I.summarize_candidates(w,[{'surface':'朋'},{'surface':'友'}])
    assert r['selected'] is not None
    p=r['candidates'][r['selected']]
    assert p['block']==[0,0] and p['primary_branch_count']==2
    assert p['ZH_comparator_agreement'] and len(r['words'])==3


def test_discovery_members_retain_their_blocks_when_clustering_abstains():
    w=[word('2_EN','a',[0,32000]),word('4_AUTO','b',[0,64000],block=(1,1))]
    r=I.summarize_candidates(w,[{'surface':'朋'},{'surface':'友'}])
    assert r['selected'] is None and r['candidates'][0]['block'] is None
    assert [p['block'] for p in r['words']]==[[0,0],[1,1]]


def test_repeated_occurrence_same_block_abstains():
    r = I.summarize_candidates([word('2_EN','a',[0,32000]), word('4_AUTO','b',[0,64000]),
                               word('2_EN','c',[16000,48000],iv=(20000,25000)),
                               word('4_AUTO','d',[16000,80000],iv=(20000,25000))], [{'surface':'朋'}])
    assert len(r['candidates']) == 2 and r['selected'] is None
    assert all(p['repeated_occurrence_ambiguous'] for p in r['candidates'])


def test_shuffle_is_deterministic_within_dialogue_bijection_and_singletons_reported():
    rows = [{'utterance_id':u, 'dialogue_id':d, 'heard_samples':n} for u,d,n in [('a','d1',100),('b','d1',120),('c','d1',110),('x','d2',100)]]
    m = I.shuffle_map(rows)
    assert m == I.shuffle_map(list(reversed(rows)))
    assert all(m[u] != u for u in ('a','b','c')) and m['x'] == 'x'
    assert set(m.values()) == set(m)


def test_processed_argmax_uses_lowest_ID_not_topk():
    import torch
    from csasr.inference_cf.core_p1 import processed_argmax
    assert processed_argmax(torch.tensor([0., 2., 2., 0.]), 1, [], []) == 1


def test_attention_timing_shape_and_cap_semantics_are_unchanged():
    h = np.ones((3, 10, 100), dtype=np.float32)
    t = token_intervals(h, 3, 'cap', 32000)
    assert t['n_timed'] == 2
    assert all(0 <= a < b <= 32000 for a,b in zip(t['start'], t['end']))


def test_gates_separate_discovery_from_selection():
    g=json.loads((ROOT/'configs/research_review/internal0a.json').read_text())['gates']
    m=dict(evaluable_absent=14,new_occurrence_hits=3,new_hit_dialogues=3,selected=0,selected_dialogues=0,
           precision=0.,selected_genuine=0,selected_genuine_dialogues=0,false_control_utterances=0,precision_minus_shuffled=None)
    assert I.decision(m,g)=='INTERNAL0A_DISCOVERY_ONLY_SELECTION_INSUFFICIENT'
    assert I.decision(m,g,['identity'])=='INTERNAL0A_INVALID'
    assert I.decision(dict(m,evaluable_absent=10),g)=='INTERNAL0A_INSUFFICIENT'
    assert I.decision(dict(m,new_occurrence_hits=0),g)=='INTERNAL0A_NO_NEW_LEXICAL_DISCOVERY'


def test_runner_does_not_import_evaluator_or_gold_provider():
    for p in ['experiments/internal0a.py','src/csasr/inference_cf/internal0a.py']:
        tree=ast.parse((ROOT/p).read_text())
        modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any('evaluate' in x or 'offline' in x or 'r0_analyze' in x for x in modules)
    code=(ROOT/'experiments/internal0a.py').read_text()
    assert '--approved-full-capture' in code and 'lexical_outputs_inspected' in code


def test_config_matches_fixed_module_constants():
    c=json.loads((ROOT/'configs/research_review/internal0a.json').read_text())
    assert c['crops']['sample_rate']==I.SR and c['crops']['heard_max_samples']==I.HEARD
    assert tuple(c['crops']['branches'])==I.BRANCHES
    assert c['selection']['ZH_comparator_only'] and not c['selection']['require_R0']


def test_review_inventory_blinds_branch_selection_and_keeps_occurrence_ids():
    from experiments.internal0a_offline import review_inventory
    w=dict(word('2_EN','secret_branch_view',[0,32000]), tok_last=1)
    cand={'candidate_id':0,'surface':'friends','members':[w],'interval':w['interval']}
    cap={'rows':[{'utterance_id':'u','words':[w],'candidates':[cand],'selected':0,
                  'shuffled':{'candidates':[],'selected':None}}]}
    ep={'rows':[{'utterance_id':'u','reference':'我的 friends','onsets':[{'word':'friends','ref_index':2,'present_anywhere':False}]}]}
    inv=review_inventory(cap,ep,{'u':{'audio_path':'u.wav'}})
    assert len(inv)==3
    assert all(k.startswith('sha256:') for k in inv)
    assert not any('branch' in x or 'selected' in x or 'supported' in x for x in inv.values())
    assert all(x['A']!=None and x['B']!=None for x in inv.values())
