"""Design-only tests: no model loading, forward, outcome evaluation or submission.

The miniature predicate interpreter is a test oracle for synthetic boundary
cases, not experiment decision code. New scientific implementation remains pending.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import operator
from pathlib import Path

import pytest

from experiments.inference_cf_srd2_g0_inventory import identity_projection

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/inference_cf/srd2_g0.json'
C = json.loads(CONFIG.read_text())
P = json.loads((ROOT / C['population']['manifest']).read_text())


def digest(obj):
    return 'sha256:' + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


OPS = {'==': operator.eq, '<=': operator.le, '>=': operator.ge, '>': operator.gt}


def predicate(rule, metrics):
    if 'all' in rule:
        return all(predicate(x, metrics) for x in rule['all'])
    if 'any' in rule:
        return any(predicate(x, metrics) for x in rule['any'])
    value = metrics.get(rule['metric'])
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return OPS[rule['op']](value, rule['value'])


def condition(rule, gates, flags):
    if 'all' in rule:
        return all(condition(x, gates, flags) for x in rule['all'])
    if 'any' in rule:
        return any(condition(x, gates, flags) for x in rule['any'])
    if 'flag' in rule:
        return flags[rule['flag']]
    return gates[rule['gate']] is rule['is']


def test_population_byte_self_roster_and_membership_hashes():
    assert sha(ROOT / C['population']['manifest']) == C['population']['file_sha256']
    assert digest({k: v for k, v in P.items() if k != 'manifest_hash'}) == P['manifest_hash']
    assert digest(P['selected']) == P['selected_hash'] == C['population']['selected_hash']
    assert digest(P['roster']) == P['roster_hash'] == C['population']['roster_hash']
    assert digest([r['utterance_id'] for r in P['selected']]) == C['population']['selected_ids_hash']
    assert sha(P['source']) == P['source_sha256']


def test_full_role_inventory_balanced_selection_and_exposure_exclusion():
    roster = P['roster']
    assert len(roster) == len({r['utterance_id'] for r in roster}) == 7919
    selected = P['selected']
    assert len(selected) == len({r['utterance_id'] for r in selected}) == 400
    assert set(Counter(r['dialogue_id'] for r in selected).values()) == {20}
    assert len({r['dialogue_id'] for r in selected}) == 20
    excluded = {r['utterance_id'] for r in roster if r['exclusion_reasons']}
    assert len(excluded) == 300 and not excluded & {r['utterance_id'] for r in selected}
    full300 = identity_projection(json.loads((ROOT / 'docs/inference_cf/R0_PANEL.json').read_text()))
    assert len(full300) == 300 and full300 <= excluded
    assert sum(not r['exclusion_reasons'] for r in roster) == 7619


def test_independent_role_ID_projection_matches_the_entire_inventory():
    import pyarrow.parquet as pq
    rows = pq.read_table(P['source'], columns=['utterance_id', 'dialogue_id', 'role']).to_pylist()
    assert all(r['role'] == 'D-dev-select' for r in rows)
    assert {(r['utterance_id'], r['dialogue_id']) for r in rows} == \
        {(r['utterance_id'], r['dialogue_id']) for r in P['roster']}


def test_independent_hash_order_recreates_all_selected_ids_without_lexical_features():
    chosen = []
    for dialogue in sorted({r['dialogue_id'] for r in P['roster']}):
        eligible = [r for r in P['roster'] if r['dialogue_id'] == dialogue and not r['exclusion_reasons']]
        def key(r):
            return (hashlib.sha256(f"SRD2-G0-population-v1|240924|{r['utterance_id']}".encode()).hexdigest(), r['utterance_id'])
        chosen.extend(r['utterance_id'] for r in sorted(eligible, key=key)[:20])
    assert chosen == [r['utterance_id'] for r in P['selected']]


def test_reference_text_metrics_and_goodness_cannot_change_identity_projection():
    a = {'rows': [{'utterance_id': 'UID1', 'reference': 'gold', 'correct': True, 'gate': .8}],
         'texts': {'ZH-CN_U0023_S0_664': 'one transcript'}, 'PIER': .4}
    b = {'rows': [{'utterance_id': 'UID1', 'reference': 'other', 'correct': False, 'gate': .0}],
         'texts': {'ZH-CN_U0023_S0_664': 'another transcript'}, 'PIER': .1}
    assert identity_projection(a) == identity_projection(b) == {'UID1', 'ZH-CN_U0023_S0_664'}


def test_source_pins_and_exclusion_registry_have_not_changed():
    for path, expected in C['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
    for source in P['exclusion_sources']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    assert sha(C['environment']['transformers_forward_source']) == C['environment']['transformers_forward_sha256']


def test_selected_audio_is_verified_without_inference():
    for row in P['selected']:
        assert sha(row['audio_path']) == row['audio_full_sha256']
        assert row['sample_rate'] == 16000 and row['channels'] == 1 and row['frames'] > 0
        assert abs(row['frames'] / 16000 - row['source_duration_sec']) < .001
    assert sum(r['source_duration_sec'] > 30 for r in P['selected']) == 23


def test_role_and_runtime_allowlists_exclude_reference_features():
    assert P['source_columns_read'] == ['utterance_id', 'dialogue_id', 'role', 'audio_path', 'audio_sha256', 'duration_sec']
    assert C['firewall']['runtime_input_allowlist'] == ['canonical_index', 'utterance_id', 'audio_path', 'audio_full_sha256']
    assert C['firewall']['authorization_B_fields'] == ['schema', 'job_A_seal_hash', 'config_hash', 'population_hash', 'permutation_hash', 'audit_hash', 'resource_forecast_hash', 'authorized']
    assert not C['firewall']['authorization_contains_per_row_correctness']


def test_model_contract_contains_only_model_metadata_and_actuator_settings():
    assert set(C['model']) == {'dir', 'files', 'precision', 'attention_implementation', 'eval',
        'parameter_requires_grad', 'parameter_updates', 'layer_zero_based', 'site',
        'norm_preserve', 'depth_rescale', 'temperature_readout', 'suppression_hash'}
    assert len(C['model']['files']) == 11
    assert C['model']['layer_zero_based'] == 16
    assert C['model']['parameter_updates'] == 0 and not C['model']['parameter_requires_grad']
    assert C['model']['norm_preserve'] and not C['model']['depth_rescale']
    assert 'positions' not in C['model'] and 'construction' not in C['model']


def test_historical_R2_and_D2_probability_semantics_remain_distinct():
    from csasr.inference_cf.core_r2 import conflict_from_logits, local_support
    from csasr.inference_cf.readout import objective
    import torch
    part = {'embedded_ids': [0], 'matrix_ids': [1], 'ambiguous_ids': [2]}
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    raw = conflict_from_logits(z, part)
    processed = objective(z, 1, [2], [], part)
    assert raw['P_E'] == pytest.approx(float(torch.softmax(z.float(), 0)[0]))
    assert math.exp(float(processed['log_PE'])) > raw['P_E']
    s = local_support(.4, .2, .4, .2)
    # Both APIs are pure numerical functions; no native-LID/model call occurs.
    assert s['E'] == 0.0
    assert 'raw' in C['gate']['conflict_logits'] and 'generation' in C['D2']['probabilities']


def test_gate_scaled_chord_is_not_a_linearly_scaled_coefficient():
    import torch
    from experiments.inference_cf_p2r import solve_scale, emulate_edit_norm
    h = torch.tensor([1.2, 1.6, 0.0], dtype=torch.float64)
    d = torch.tensor([-.8, .6, 0.0], dtype=torch.float64)
    e = C['dose']['e_star']
    full = solve_scale(h, d, e)
    quarter = solve_scale(h, d, e * .25)
    assert quarter['rel_sq_err'] <= .02
    assert abs(emulate_edit_norm(h, d, quarter['s']) / (e * .25) - 1) <= .02
    assert abs(emulate_edit_norm(h, d, full['s'] * .25) / (e * .25) - 1) > .02
    assert C['dose']['zero_target'].startswith('bypass')
    assert 'norm ratio, NOT vector' in C['dose']['consumed_vs_proposed_error_definition']
    assert 'never inspect FFN/logits' in C['dose']['native_consumption_preview']
    assert 'never solve a second time' in C['dose']['solver_calls_per_arm']


def test_shuffle_is_an_outcome_blind_within_utterance_multiset_not_a_reroll():
    # Independent direct construction of the frozen hash permutation, not a runtime provider.
    ts = [1, 2, 4, 5, 8]
    gates = {1: 0.0, 2: .2, 4: 0.0, 5: .7, 8: .4}
    donors = sorted(ts, key=lambda t: (hashlib.sha256(f'SRD2-G0-gate-permutation-v1|240924|UID|{t}'.encode()).hexdigest(), t))
    assigned = dict(zip(ts, (gates[t] for t in donors)))
    assert sorted(assigned.values()) == sorted(gates.values())
    assert sum(x*x for x in assigned.values()) == pytest.approx(sum(x*x for x in gates.values()))
    assert C['shuffle']['zero_gates_in_pool'] and C['shuffle']['within_utterance_only']
    assert 'never reroll' in C['shuffle']['singleton_or_identity_or_constant']
    assert C['shuffle']['assignment_not_changed_by_D2_invalidity']


def test_integer_event_and_rate_boundaries_and_undefined_never_pass():
    gate = C['gate_predicates']['causal_power']
    assert predicate(gate, {'C_B2': 5, 'corrected_dialogues_B2': 3})
    assert not predicate(gate, {'C_B2': 4, 'corrected_dialogues_B2': 20})
    assert not predicate(gate, {'C_B2': 100, 'corrected_dialogues_B2': 2})
    assert not predicate(gate, {'C_B2': None, 'corrected_dialogues_B2': 3})
    assert not predicate(gate, {'C_B2': float('nan'), 'corrected_dialogues_B2': 3})
    assert C['acceptance']['B2_B1_correction_retention_min'] == .6
    assert math.ceil(.6*7) == 5 and math.floor(.5*7) == 3


def test_lower_energy_without_shuffle_advantage_or_positive_net_utility_cannot_pass():
    metrics = {'correction_retention_B2_B1': .6, 'ZH_harm_ratio_B2_B1': .5,
        'H_ZH_B2': 2, 'ZH_corruption_rate_B2': .01, 'ZH_active_corruption_rate_B2': .1,
        'H_EN_B2': 1, 'EN_corruption_rate_B2': .02, 'C_B2_minus_B3': 2,
        'U_B2_minus_B3': 3, 'H_ZH_B2_minus_B3': 0,
        'positive_correction_gain_dialogues': 2, 'positive_utility_gain_dialogues': 3,
        'max_positive_utility_dialogue_share': .6, 'U_B2': 1}
    gate = C['gate_predicates']['acceptance']
    assert predicate(gate, metrics)
    for key, value in [('C_B2_minus_B3', 1), ('U_B2_minus_B3', 2), ('U_B2', 0),
                       ('max_positive_utility_dialogue_share', .61)]:
        assert not predicate(gate, {**metrics, key: value})
    dose = C['gate_predicates']['dose_comparison']
    dm = {'planned_B2_B3_multisets_equal': True, 'energy_ratio_B2_B3': .9,
          'lost_planned_energy_B2': .1, 'lost_planned_energy_B3': .1}
    assert predicate(dose, dm)
    assert not predicate(dose, {**dm, 'energy_ratio_B2_B3': .89})
    assert not predicate(dose, {**dm, 'energy_ratio_B2_B3': None})


def test_sparse_or_zero_active_Mandarin_is_inconclusive_not_harmless():
    gate = C['gate_predicates']['opportunities']
    vals = {r['metric']: r['value'] for r in gate['all']}
    assert predicate(gate, vals)
    assert not predicate(gate, {**vals, 'active_B2_ZH_correct': 4})
    assert not predicate(gate, {**vals, 'active_B2_ZH_dialogues': 9})


def test_adverse_damage_stop_is_not_masked_by_success():
    rule = C['gate_predicates']['severe_damage']
    base = {'H_ZH_B2': 0, 'ZH_corruption_rate_B2': 0,
        'any_dialogue_ZH_damage_events_ge3_AND_active_rate_gt020': False,
        'H_EN_B2': 0, 'EN_corruption_rate_B2': 0, 'new_EOS_proxy_events_B2': 0}
    assert not predicate(rule, base)
    assert predicate(rule, {**base, 'new_EOS_proxy_events_B2': 1})
    assert not predicate(rule, {**base, 'H_ZH_B2': 10, 'ZH_corruption_rate_B2': .03})
    assert predicate(rule, {**base, 'H_ZH_B2': 10, 'ZH_corruption_rate_B2': .031})


def test_terminal_precedence_and_audit_are_complete_and_deterministic():
    assert [r['label'] for r in C['terminal_rules']] == C['terminal_precedence']
    gates = {key: True for key in C['gate_predicates']}
    gates['severe_damage'] = False
    flags = {'critical_failure': False, 'independent_FULL_audit_PASS': True}
    def decide(g, f):
        return next(r['label'] for r in C['terminal_rules'] if condition(r['when'], g, f))
    assert decide(gates, flags) == 'SRD2_G0_SELECTIVE_FEASIBILITY'
    assert decide({**gates, 'severe_damage': True}, flags) == 'SRD2_G0_OBSERVED_DAMAGE'
    assert decide({**gates, 'acceptance': False}, flags) == 'SRD2_G0_NONSELECTIVE'
    assert decide({**gates, 'opportunities': False}, flags) == 'SRD2_G0_OPPORTUNITY_INSUFFICIENT'
    assert decide({**gates, 'dose_comparison': False}, flags) == 'SRD2_G0_CONTROL_COMPARISON_INCONCLUSIVE'
    assert decide(gates, {**flags, 'critical_failure': True}) == 'SRD2_G0_INVALID'
    # Independent audit failure must be represented as a critical failure, never a provisional PASS.
    assert decide(gates, {'critical_failure': True, 'independent_FULL_audit_PASS': False}) == 'SRD2_G0_INVALID'


def test_every_quantitative_gate_is_a_closed_machine_readable_predicate():
    def check(node):
        if 'all' in node or 'any' in node:
            assert len(node) == 1
            values = node.get('all', node.get('any'))
            assert values
            for child in values:
                check(child)
        else:
            assert set(node) == {'metric', 'op', 'value'}
            assert node['metric'] and node['op'] in OPS
            assert isinstance(node['value'], (int, float, bool))
            assert math.isfinite(node['value'])
    for gate in C['gate_predicates'].values():
        check(gate)
    def references(node):
        if 'gate' in node:
            assert node['gate'] in C['gate_predicates'] and isinstance(node['is'], bool)
        for key in ('all', 'any'):
            for child in node.get(key, []):
                references(child)
    for rule in C['terminal_rules']:
        references(rule['when'])


def test_resource_ceiling_does_not_authorize_automatic_subset_or_extra_job():
    r = C['compute']
    assert .18 * 22256 + 800 < 9000 < 10800
    assert .18 * 80000 + 800 > 10800
    assert r['max_scientific_jobs'] == 2 and not r['automatic_population_reduction']
    assert not r['GPU_in_codex'] and not C['scientific_execution_in_codex']


def test_handoff_config_hash_and_independent_audit_contract():
    handoff = (ROOT / 'docs/inference_cf/SRD2_G0_CLAUDE_HANDOFF.md').read_text()
    assert sha(CONFIG) in handoff
    firewall = (ROOT / 'docs/inference_cf/SRD2_G0_FIREWALL.md').read_text()
    assert 'Do not import primary' in firewall
    assert 'PRIMARY' in firewall and 'FULL' in firewall and 'PASS_TO_SRD2_G0' in firewall
    assert C['statistics']['family_size'] == len(C['statistics']['family']) == 4
    assert C['statistics']['family_quantiles'] == [.05/(2*4), 1-.05/(2*4)]
