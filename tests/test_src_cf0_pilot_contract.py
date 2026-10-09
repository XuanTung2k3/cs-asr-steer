"""CPU-only design contracts, independent opportunity arithmetic and synthetic gates."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
from csasr.inference_cf import cf_pilot_contract as C

ROOT = Path(__file__).resolve().parents[1]
BASE = '20cd36824822c356f910f42c9b5ca3a5fd5f56b6'

def read(p):
    return json.loads((ROOT / p).read_text())

def sha(p):
    h = hashlib.sha256()
    with (ROOT / p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return 'sha256:' + h.hexdigest()

def digest(o):
    return 'sha256:' + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@pytest.fixture
def cfg():
    return read('configs/inference_cf/src_cf0_pilot.json')


def arm():
    return {'qualified_layer': True, 'energy_valid': True, 'paired_rows': 15,
        'paired_dialogues': 9, 'corrections': 3, 'corrected_dialogues': 3,
        'random': {'net_corrections': 3, 'positive_dialogues': 3,
                   'macro_advantage': .2, 'lodo_net': [2] * 9},
        'off_target': {'net_corrections': 3, 'positive_dialogues': 3,
                   'macro_advantage': .2, 'lodo_net': [2] * 9},
        'EN_corruptions': 0, 'ZH_corruptions': 0, 'correct_EOS_promotions': 0,
        'margin_macro': 0., 'margin_adjusted_lower': None}


def decide(cfg, arms, **kw):
    return C.terminal(cfg, integrity=kw.get('integrity', True),
        construction=kw.get('construction', True), specificity=kw.get('specificity', True), arms=arms)


def test_direction_uses_float64_before_subtraction_and_epsilon():
    clean = np.zeros(1280, dtype=np.float32); clean[0] = 2
    mask = clean.copy(); mask[1] = -.125
    d = C.direction(clean, mask)
    expected = ((clean.astype(np.float64) - mask.astype(np.float64)) / (.125 + 1e-6)).astype(np.float32)
    np.testing.assert_array_equal(d['vector'], expected)
    assert d['tangent_ratio'] == 1
    assert d['raw_tangent_norm'] == .125
    assert d['direction_norm'] < 1  # historical epsilon, not an exact unit-vector assertion
    np.testing.assert_array_equal(C.direction(clean, mask)['vector'], d['vector'])


def test_parallel_direction_has_no_tangent_and_degenerate_abstains():
    h = np.ones(1280, dtype=np.float32)
    assert C.direction(h, h)['status'] == 'ABSTAIN_DEGENERATE'
    assert C.direction(h, h / 2)['tangent_ratio'] < 1e-12
    with pytest.raises(ValueError, match='NONFINITE'):
        C.direction(h, np.full(1280, np.nan))
    with pytest.raises(ValueError, match='SHAPE'):
        C.direction(h[:3], h[:3])


def test_random_is_stable_query_layer_specific_and_label_free():
    a = C.random_direction('example', 4, 16)
    np.testing.assert_array_equal(a, C.random_direction('example', 4, 16))
    assert not np.array_equal(a, C.random_direction('example', 4, 24))
    assert not np.array_equal(a, C.random_direction('example', 5, 16))
    assert abs(np.linalg.norm(a.astype(np.float64)) - 1) < 1e-7


def test_three_dialogue_three_correction_signal_is_attainable_without_significance(cfg):
    # Three independent-dialogue flips among fifteen queries, controls zero: prespecified
    # preliminary effect can pass. No fabricated outcome or small-N adjusted CI requirement.
    aa = [arm() for _ in range(8)]
    assert decide(cfg, aa) == 'SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED'
    assert cfg['statistics']['CI_gates_advancement'] is False


def test_construction_thresholds_are_executable_and_Mandarin_not_fake_gate(cfg):
    m = dict(valid_targets=66, valid_off_pairs=54, EN_confusion_target=20,
             EN_confusion_target_dialogues=8, EN_confusion_joint=15,
             EN_confusion_joint_dialogues=6, EN_correct_joint=20,
             EN_correct_joint_dialogues=10, specificity_pairs=12,
             specificity_dialogues=6, median_abs_cosine=.9, median_tangent_ratio=1.25)
    assert C.construction_layer(m, cfg)['qualified']
    m['EN_confusion_joint'] = 14
    assert not C.construction_layer(m, cfg)['construction']
    m['EN_confusion_joint'] = 15; m['median_tangent_ratio'] = None
    assert C.construction_layer(m, cfg)['construction']
    assert not C.construction_layer(m, cfg)['specificity']
    assert cfg['gate_a']['per_layer']['ZH_correct_minimum'].startswith('NONE')


def test_undefined_control_metrics_cannot_pass(cfg):
    a = arm(); a['random']['macro_advantage'] = float('nan')
    assert not C.power_pass(a, cfg['gate_b'])
    a = arm(); a['off_target']['lodo_net'][0] = float('nan')
    assert not C.power_pass(a, cfg['gate_b'])


@pytest.mark.parametrize('mutation', ['no_correction', 'one_dialogue', 'one_control', 'lodo', 'energy', 'layer'])
def test_every_required_power_component_is_binding(cfg, mutation):
    a = arm()
    if mutation == 'no_correction': a['corrections'] = 0
    if mutation == 'one_dialogue': a['corrected_dialogues'] = 1
    if mutation == 'one_control': a['off_target']['net_corrections'] = 1
    if mutation == 'lodo': a['random']['lodo_net'][0] = 0
    if mutation == 'energy': a['energy_valid'] = False
    if mutation == 'layer': a['qualified_layer'] = False
    assert not C.power_pass(a, cfg['gate_b'])


def test_no_oracle_union_or_margin_substitution(cfg):
    aa = [arm() for _ in range(8)]
    for a in aa: a['corrections'] = 1; a['margin_macro'] = .8; a['margin_adjusted_lower'] = .1
    assert decide(cfg, aa) == 'SRC_CF0_PILOT_MARGIN_ONLY'
    for a in aa: a['margin_macro'] = 0
    assert decide(cfg, aa) == 'SRC_CF0_PILOT_CAUSAL_INSUFFICIENT'


def test_damage_invalid_and_construction_precedence(cfg):
    aa = [arm() for _ in range(8)]
    aa[0]['EN_corruptions'] = 12
    assert decide(cfg, aa) == 'SRC_CF0_PILOT_OBSERVED_DAMAGE'
    assert decide(cfg, aa, integrity=False) == 'SRC_CF0_PILOT_INVALID'
    assert decide(cfg, [], construction=False) == 'SRC_CF0_PILOT_CONSTRUCTION_INSUFFICIENT'
    assert decide(cfg, [], specificity=False) == 'SRC_CF0_PILOT_DIRECTION_NONSPECIFIC'
    assert decide(cfg, aa[:1]) == 'SRC_CF0_PILOT_INVALID'
    for a in aa: a['EN_corruptions'] = 0; a['ZH_corruptions'] = 1
    assert decide(cfg, aa) == 'SRC_CF0_PILOT_OBSERVED_DAMAGE'


def test_frozen_scope_and_gates(cfg):
    assert cfg['matrix']['layers'] == [16, 24]
    assert cfg['matrix']['eta'] == [.15, .30]
    assert cfg['matrix']['signs'] == [1, -1]
    assert cfg['matrix']['primary_arms'] == 8
    assert cfg['matrix']['random_arms'] == cfg['matrix']['off_target_arms'] == 8
    assert cfg['energy']['relative_chord_error_max'] == .02
    assert cfg['energy']['relative_squared_energy_error_max'] == .02
    assert cfg['energy']['native_norm_rounding'] == 'DESCRIPTIVE_ONLY'
    assert cfg['gate_a']['fraction_denominators']['minimum_valid_target_all_strata'] == math.ceil(.9 * 73)
    assert cfg['gate_a']['fraction_denominators']['minimum_valid_off_target_all_strata'] == math.ceil(.9 * 59)
    assert cfg['statistics']['primary_family'] == 16
    assert .05 / (2 * 16) == .0015625
    assert cfg['compute']['scientific_jobs_max'] == 2
    assert cfg['compute']['hours_per_job_max'] == 3
    assert cfg['full300_plan']['new_census_authorized'] is False
    assert cfg['region_policy'] == read('configs/inference_cf/s1_acoustic_evidence.json')['regions']
    assert cfg['counterfactual'] == read('configs/inference_cf/s1_acoustic_evidence.json')['counterfactual']


def test_panel_hashes_sources_and_blocked_history_unchanged(cfg):
    p = read(cfg['panel']['path'])
    assert sha(cfg['panel']['path']) == cfg['panel']['file_sha256']
    assert p['identity_hash'] == cfg['panel']['identity_hash']
    assert p['identity_hash'] == digest({k: v for k, v in p.items() if k != 'identity_hash'})
    assert len(p['runtime_queries']) == 180 and len(p['utterances']) == 80
    for k, v in {**cfg['pins'], **cfg['historical_inputs']}.items(): assert sha(k) == v, k
    assert sha(cfg['full300_plan']['inventory']) == cfg['full300_plan']['inventory_sha256']
    handoff = (ROOT / 'docs/inference_cf/SRC_CF0_CLAUDE_HANDOFF.md').read_text()
    assert sha('configs/inference_cf/src_cf0_pilot.json') in handoff
    assert read(cfg['full300_plan']['inventory'])['identity_hash'] in handoff
    for f in ['SRC_CF0_SPEC.md','SRC_CF0_CODEX_DESIGN.md','SRC_CF0_FIREWALL.md','SRC_CF0_PANEL.json']:
        path = 'docs/inference_cf/' + f
        assert (ROOT / path).read_bytes() == subprocess.check_output(['git','show',BASE+':'+path], cwd=ROOT)
    path = 'configs/inference_cf/src_cf0.json'
    assert (ROOT / path).read_bytes() == subprocess.check_output(['git','show',BASE+':'+path], cwd=ROOT)
    for f in ['STATUS.md','CODE_MAP.md','DATA_EXPOSURE.md']:
        path = 'docs/current/' + f
        assert (ROOT / path).read_bytes().startswith(subprocess.check_output(['git','show',BASE+':'+path], cwd=ROOT))


def test_runtime_and_authorization_are_minimal(cfg):
    f = cfg['firewall']
    assert f['primary_region_array_allowlist'] == ['track_heard_intervals']
    forbidden = {'stratum','acceptable_tokens','reference','coverage_review_only','target_token','gold_timing'}
    assert not forbidden.intersection(f['runtime_query_keys'] + f['runtime_utterance_keys'] + f['authorization_keys'])
    assert f['remote_seals_required']
    source = (ROOT/'src/csasr/inference_cf/cf_pilot_contract.py').read_text()
    assert 'torch' not in source and 'transformers' not in source


def test_full300_inventory_bounds_and_prefix_identity():
    v = read('docs/inference_cf/SRC_CF0_FULL300_INVENTORY.json')
    assert v['identity_hash'] == digest({k: x for k, x in v.items() if k != 'identity_hash'})
    assert not v['new_model_inference'] and not v['references_accessed']
    assert not v['exact_full300_intervention_census']
    assert len(v['rows']) == 300
    assert sum(x['structural_query_count'] for x in v['rows']) == 12146
    assert sum(x['region_only_possible_target_queries'] for x in v['rows']) == 10334
    assert v['summary']['certified_target_lower_bound'] == {'queries':73,'utterances':52,'dialogues':18}
    assert v['summary']['certified_paired_lower_bound'] == {'queries':59,'utterances':43,'dialogues':17}
    r = read('docs/inference_cf/R0_PANEL.json'); s = read('docs/inference_cf/S1_PANEL.json')
    r = {x['utterance_id']:x for x in r['rows']}; s = {x['utterance_id']:x for x in s['utterances']}
    for q in v['compatible_s1_queries']:
        uid,t = q['utterance_id'],q['t']
        assert r[uid]['audio']['full_sha256'] == s[uid]['audio_full_file_sha256']
        assert r[uid]['baseline']['content_ids'][:t] == s[uid]['baseline_content_tokens'][:t]
    assert sum(x['target'] for x in v['compatible_s1_queries']) == 73
    assert sum(x['paired'] for x in v['compatible_s1_queries']) == 59


def test_safety_sample_adequacy_is_not_a_fake_iid_or_cluster_bound():
    assert 1 - .05 ** (1/4) > .52
    assert 1 - .05 ** (1/40) > .07
    assert 1 - .05 ** (1/60) < .05
    assert math.ceil(math.log(.05)/math.log(.95)) == 59
    assert 1 - .05 ** (1/20) > .13
    assert 1 - .05 ** (1/10) > .25
