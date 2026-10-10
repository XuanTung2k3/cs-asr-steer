"""Frozen R1 design identity and numerical evidence; no pretrained model or outcome."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from csasr.inference_cf.core import digest, file_hash
from csasr.inference_cf.core_p1 import direction
from csasr.models.hooks import apply_steering
from experiments.inference_cf_p2r import solve_scale

CFG = json.loads(Path('configs/inference_cf/st_prompt_r1.json').read_text())
PANEL = json.loads(Path(CFG['panel']).read_text())


def test_exact_parent_population_and_reference_free_projection():
    old = json.loads(Path('docs/inference_cf/ST_LOC0_PANEL.json').read_text())
    assert PANEL['parent_identity_hash'] == old['identity_hash']
    assert PANEL['identity_hash'] == CFG['panel_identity_hash'] == digest({k:v for k,v in PANEL.items() if k!='identity_hash'})
    assert PANEL['runtime_queries'] == old['runtime_queries']
    assert PANEL['utterances'] == old['utterances']
    assert PANEL['evaluation_membership'] == old['construction_positions']
    assert Counter(x['stratum'] for x in PANEL['evaluation_membership']) == {'EN-confusion':60,'EN-correct':60,'ZH-correct':60}
    assert len(PANEL['runtime_queries']) == 180 and len(PANEL['utterances']) == 80
    assert len({x['dialogue_id'] for x in PANEL['runtime_queries']}) == 20
    assert CFG['firewall']['runtime_projection'] == ['runtime_queries','utterances']
    assert not any('stratum' in q or 'target_ids' in q or 'competitor' in q for q in PANEL['runtime_queries'])
    by={u['utterance_id']:u for u in PANEL['utterances']}
    for q in PANEL['runtime_queries']:
        prefix=by[q['utterance_id']]['baseline_content_tokens'][:q['t']]
        assert q['content_prefix_sha256'] == digest(prefix)
        assert q['absolute_query'] == 4+q['t']-1
        assert q['forced_en_query_input_sha256'] == digest(CFG['conditions']['c_E']+prefix)
        assert q['forced_zh_query_input_sha256'] == digest(CFG['conditions']['c_M']+prefix)


def test_complete_grid_random_seeds_and_sources():
    assert {(a['layer'],a['eta'],a['sign']) for a in CFG['primary_arms']} == set(itertools.product([3,8,16,24],[.15,.30,.45],[1,-1]))
    assert len({a['id'] for a in CFG['primary_arms']}) == 24
    assert len(CFG['controls']['random']) == 12
    for r in CFG['controls']['random']:
        seed=int(hashlib.sha256(r['seed_key'].encode()).hexdigest()[:16],16)
        assert r['seed']==seed
        v=np.random.default_rng(seed).standard_normal(1280);v=(v/np.linalg.norm(v)).astype(np.float32)
        assert 'sha256:'+hashlib.sha256(v.tobytes()).hexdigest()==r['array_bytes_sha256']
    assert all(file_hash(p)==h for p,h in CFG['source_sha256'].items())
    assert CFG['controls']['D2']['eligible_for_selection'] is False


def test_real_relative_chord_and_unreachable_not_alpha():
    r=torch.tensor([1.,0.,0.,0.],dtype=torch.bfloat16)
    d=torch.tensor([0.,1.,0.,0.])
    for eta in CFG['energy']['eta']:
        sol=solve_scale(r,d,eta*float(r.double().norm()))
        assert sol['status']=='ok' and sol['evals']<=8 and sol['rel_sq_err']<=.02
        assert abs(sol['edit_norm']/float(r.double().norm())/eta-1)<=.02
    assert solve_scale(r,r.float(),.15)['status']=='energy_unreachable'
    assert solve_scale(r,d,2.)['status']=='energy_unreachable'
    x=r.view(1,1,-1)
    assert apply_steering(x,d,0.,1.,None,True) is x
    assert torch.equal(apply_steering(x,d,1.,1.,torch.zeros(1,1),True),x)


def test_geometry_evidence_honestly_partial_and_reproducible():
    e=json.loads(Path('docs/inference_cf/ST_PROMPT_R1_REACHABILITY.json').read_text())
    assert e['config_sha256']==file_hash('configs/inference_cf/st_prompt_r1.json')
    assert e['cases']==2160 and e['pending_layers']==[3,8]
    assert not e['complete_four_layer_preflight'] and not e['evaluator_access']
    assert e['GPU_calls']==e['new_model_forwards']==e['new_edited_logits']==0
    assert all(file_hash(p)==h for p,h in e['inputs'].items())
    for row in e['arms'].values():
        assert not row['failure_indices'] and row['max_relative_squared_error']<=.02
        assert 2<=row['max_solver_evaluations']<=8
    # Independent formula check on sealed inputs, no logits/reference targets.
    for path in e['inputs']:
        with np.load(path) as z:
            for b,a in zip(z['H_B'],z['H_E']):
                result=direction(torch.from_numpy(a),torch.from_numpy(b))
                delta=a.astype(np.float64)-b.astype(np.float64)
                np.testing.assert_array_equal(result['d'].numpy(),(delta/(np.linalg.norm(delta)+1e-6)).astype(np.float32))


def test_original_solver_guards_and_selection_family_fixed():
    old=json.loads(Path('configs/inference_cf/st_loc0.json').read_text())
    for key in ('solver_max_evaluations','max_relative_norm_error','max_relative_squared_error','max_pairwise_relative_squared_error','repair_epsilon','depth_rescale','norm_preserve'):
        assert CFG['energy'][key]==old['energy'][key]
    assert all(CFG['historical_reproduction'][k]==old['historical_reproduction'][k] for k in ('baseline_states_logits','zero_restore','D0_vector_max_abs','D2_vector_max_abs','solver_scale_D0_abs'))
    assert CFG['eligibility']['selection_joint_candidate_random_valid']
    b=CFG['bootstrap']
    assert b['replicates']==10000 and b['seed']==240924
    assert b['joint_family_size']==24*2
    assert b['adjusted_quantiles'][0]==.05/(2*48)
    assert CFG['R1_A']['max_selected']==2
    assert CFG['R1_A']['minimum_corrections']==CFG['R1_A']['minimum_correction_dialogues']==3
    assert CFG['R1_A']['maximum_correct_corruptions']==5 and CFG['R1_A']['maximum_ZH_corruptions']==2


def test_conditional_sequence_scope_and_safety():
    b=CFG['R1_B']
    assert 'CAUSAL_PROMISE' in b['run_only_after']
    assert b['utterances']==80 and b['max_selected']==2
    assert b['bootstrap']['adjusted_quantiles'][0]==.05/16
    assert b['safety']==dict(max_MER_increase=.005,max_ZH_CER_increase=.005,min_matrix_retention=.99,min_EN_retention=.95,max_POI_corruption=.05,max_outside_POI_harm=.01,max_added_deletion_errors=5,max_added_caps=0,max_new_severe_truncations=0)
    assert CFG['firewall']['oracle_privilege']==['utterance_id','t']
    assert 'no redistribution' in b['physical_random_budget']
    assert CFG['compute']['Codex_GPU_jobs']==0 and CFG['compute']['hard_ceiling_seconds_per_job']==10800
    assert len(b['precedence'])==5 and b['precedence'][0].endswith('INVALID')


def test_oracle_schedule_identity_uses_locations_not_arm_outcomes():
    by={u['utterance_id']:[] for u in PANEL['utterances']}
    for q in PANEL['evaluation_membership']:
        if q['stratum']=='EN-confusion':
            by[q['utterance_id']].append(q['t'])
    by={u:sorted(set(v)) for u,v in by.items()}
    assert digest(by)==CFG['R1_B']['schedule_identity_hash']
    assert sum(bool(v) for v in by.values())==37
    assert sum(map(len,by.values()))==60 and max(map(len,by.values()))==2
    assert all(t>=1 for v in by.values() for t in v)
    assert CFG['R1_B']['schedule_counts']['no_location_utterances']==43
