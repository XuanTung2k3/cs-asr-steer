"""ST-LOC0 pre-outcome contract and site feasibility; no pretrained forward/outcome."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from csasr.inference_cf import unique as U
from csasr.inference_cf.core import digest, file_hash
from csasr.inference_cf.core_p1 import direction
from csasr.models.hooks import apply_steering

CFG = json.loads(Path('configs/inference_cf/st_loc0.json').read_text())
PANEL = json.loads(Path(CFG['panel']).read_text())


def test_population_identity_prefixes_and_reference_boundary():
    assert PANEL['identity_hash'] == CFG['panel_identity_hash'] == digest(
        {k: v for k, v in PANEL.items() if k != 'identity_hash'})
    old = json.loads(Path('results/inference_cf/p2dir/construction_population.json').read_text())
    assert PANEL['construction_positions'] == old['positions']
    assert [p['utterance_id'] for p in PANEL['utterances']] == old['utterances']
    assert Counter(p['stratum'] for p in old['positions']) == {
        'EN-confusion': 60, 'EN-correct': 60, 'ZH-correct': 60}
    assert len(PANEL['runtime_queries']) == 180
    assert len(PANEL['utterances']) == 80
    assert len({p['dialogue_id'] for p in old['positions']}) == 20
    by = {p['utterance_id']: p for p in PANEL['utterances']}
    for q, p in zip(PANEL['runtime_queries'], old['positions']):
        assert (q['utterance_id'], q['dialogue_id'], q['t']) == (
            p['utterance_id'], p['dialogue_id'], p['t'])
        assert set(q) == {'utterance_id', 'dialogue_id', 't', 'absolute_query',
                          'content_prefix_sha256', 'forced_zh_query_input_sha256',
                          'forced_en_query_input_sha256'}
        u = by[q['utterance_id']]
        assert file_hash(u['baseline_row']) == u['baseline_row_sha256']
        b = json.loads(Path(u['baseline_row']).read_text())['systems']['B0M_L16']
        assert u['baseline_content_tokens'] == b['tokens']
        prefix = b['tokens'][:q['t']]
        assert q['absolute_query'] == 4 + q['t'] - 1
        assert q['content_prefix_sha256'] == digest(prefix)
        assert q['forced_zh_query_input_sha256'] == digest(CFG['conditions']['c_M'] + prefix)
        assert q['forced_en_query_input_sha256'] == digest(CFG['conditions']['c_E'] + prefix)
    assert 'stratum' not in CFG['firewall']['runner_projection']
    for name, expected in CFG['source_sha256'].items():
        assert file_hash(name) == expected


def test_full_matrix_controls_and_energy_are_frozen():
    axes = itertools.product(['v_prompt', 'v_unq'], [16, 24], CFG['sites'], [1, -1])
    assert {(a['family'], a['layer'], a['site'], a['sign']) for a in CFG['primary_arms']} == set(axes)
    assert len({a['id'] for a in CFG['primary_arms']}) == 16
    assert len(CFG['controls']['random']) == 4
    assert CFG['controls']['D2']['eligible_for_selection'] is False
    assert CFG['compute']['pulse_cells'] == 180 * 21
    assert CFG['energy']['e_star'] == 1.1260757575454359
    old = json.loads(Path('configs/inference_cf/p2_dir_direction_identification.json').read_text())
    for k in ('e_star', 'max_relative_squared_error', 'max_pairwise_relative_squared_error',
              'solver_max_evaluations'):
        assert CFG['energy'][k] == old['energy'][k]
    assert CFG['energy']['norm_preserve'] and not CFG['energy']['depth_rescale']
    for c in CFG['controls']['random']:
        assert int(hashlib.sha256(c['seed_key'].encode()).hexdigest()[:16], 16) == c['seed']
        v = np.random.default_rng(c['seed']).standard_normal(1280)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        assert U.array_hash(v) == c['array_sha256']


def test_historical_clean_d1_membership_and_all_fold_vectors():
    con = PANEL['construction_positions']
    with np.load('results/inference_cf/p2dir/extract_run1/states.npz') as z:
        hb = z['H_B']
    idx = {(p['utterance_id'], p['t']): i for i, p in enumerate(con)}
    assert CFG['V2']['rank'] == U.RANK == 32
    assert not CFG['V2']['center'] and not CFG['V2']['label_free']
    for d, frozen in PANEL['calibration_membership']['folds'].items():
        members = U.fold_members(con, d)
        assert digest(members) == frozen['membership_sha256']
        for key in members['A'] + members['B']:
            assert con[idx[tuple(key)]]['dialogue_id'] != d
        fitted = U.fit_fold(hb[[idx[tuple(k)] for k in members['A']]],
                            hb[[idx[tuple(k)] for k in members['B']]])
        assert fitted['status'] == 'ok'
        saved = np.load(frozen['historical_vector_path'])
        assert file_hash(frozen['historical_vector_path']) == frozen['historical_vector_file_sha256']
        assert U.array_hash(saved) == frozen['historical_vector_array_sha256']
        np.testing.assert_allclose(fitted['vector'], saved, rtol=0, atol=2e-6)
    assert len(PANEL['calibration_membership']['A']) == len(PANEL['calibration_membership']['B']) == 60


def test_historical_dynamic_d0_all180_and_degenerate_no_edit():
    with np.load('results/inference_cf/p2dir/extract_run1/states.npz') as z:
        hb, he = z['H_B'], z['H_E']
    idx = {(p['utterance_id'], p['t']): i for i, p in enumerate(PANEL['construction_positions'])}
    for i, uid in enumerate(u['utterance_id'] for u in PANEL['utterances']):
        row = json.loads(Path(f'results/inference_cf/p2dir/extract_run1/rows/{i:03d}.json').read_text())
        for t, p in row['positions'].items():
            j = idx[(uid, int(t))]
            v = direction(torch.from_numpy(he[j]), torch.from_numpy(hb[j]))
            assert U.array_hash(v['d'].numpy()) == p['d0']['sha256']
    zero = direction(torch.zeros(1280), torch.zeros(1280))
    assert zero['d'] is None
    assert CFG['V1']['dynamic_per_query']
    assert CFG['V1']['epsilon'] == 1e-6 and CFG['V1']['tiny_norm'] == 1e-4


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_whisper_self_residual_route_and_zero_identity(tiny_bundle, dtype):
    """Test-only adapter prototype: new production runner/hook is deliberately not added."""
    tiny_bundle = copy.deepcopy(tiny_bundle)
    tiny_bundle.model.to(dtype)
    layer = tiny_bundle.model.model.decoder.layers[1]
    torch.manual_seed(9)
    features = torch.randn(2, 8, 100).to(dtype)
    ids = torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]])
    weights = {k: v.detach().clone() for k, v in tiny_bundle.model.state_dict().items()}

    def forward(alpha):
        box = {}
        handles = []
        def before_self(_m, args):
            box['h_in'] = args[0].detach().clone()
        def self_output(_m, _args, output):
            u = output[0]
            q = box['h_in'] + u
            gain = torch.zeros(q.shape[:2], dtype=dtype)
            gain[:, -1] = 1
            v = torch.zeros(q.shape[-1], dtype=torch.float32)
            v[0] = 1
            repaired = apply_steering(q, v, alpha, 1, gain, norm_preserve=True)
            box['q'], box['repaired'] = q.clone(), repaired.clone()
            if alpha == 0:
                box['returned_u'] = u.clone()
                return output
            returned = u + (repaired - q)
            box['returned_u'] = returned.clone()
            return (returned, *output[1:])
        def before_cross(_m, args):
            box['consumed_q'] = args[0].detach().clone()
        def cross_output(_m, _args, out):
            box['cross_u'] = out[0].detach().clone()
        def before_ffn(_m, args):
            box['r'] = args[0].detach().clone()
        try:
            handles.append(layer.self_attn_layer_norm.register_forward_pre_hook(before_self))
            handles.append(layer.self_attn.register_forward_hook(self_output))
            handles.append(layer.encoder_attn_layer_norm.register_forward_pre_hook(before_cross))
            handles.append(layer.encoder_attn.register_forward_hook(cross_output))
            handles.append(layer.final_layer_norm.register_forward_pre_hook(before_ffn))
            with torch.inference_mode():
                logits = tiny_bundle.model(input_features=features, decoder_input_ids=ids,
                                           use_cache=False).logits.clone()
        finally:
            for handle in handles:
                handle.remove()
        return logits, box

    with torch.inference_mode():
        baseline = tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False).logits
    zero, z = forward(0)
    assert torch.equal(zero, baseline)
    changed, x = forward(.2)
    assert not torch.equal(changed, baseline)
    assert torch.equal(x['consumed_q'], x['h_in'] + x['returned_u'])
    assert torch.equal(x['r'], x['consumed_q'] + x['cross_u'])
    assert torch.equal(x['consumed_q'][:, :-1], x['q'][:, :-1])
    assert not torch.equal(x['consumed_q'][:, -1], x['q'][:, -1])
    torch.testing.assert_close(x['consumed_q'], x['repaired'],
                               rtol=4 * torch.finfo(dtype).eps, atol=4 * torch.finfo(dtype).eps)
    restored, _ = forward(0)
    assert torch.equal(restored, baseline)
    assert all(torch.equal(v, weights[k]) for k, v in tiny_bundle.model.state_dict().items())
    assert all(p.grad is None for p in tiny_bundle.model.parameters())
    assert not layer.self_attn._forward_hooks and not layer.encoder_attn._forward_hooks


def test_uncertainty_gate_scope_and_eligibility_are_predeclared():
    b = CFG['bootstrap']
    assert (b['replicates'], b['seed'], b['family_size']) == (10000, 240924, 16)
    assert b['adjusted_quantiles'] == [(.05 / 16) / 2, 1 - (.05 / 16) / 2]
    d = CFG['decision']
    assert d['minimum_EN_confusion_dialogue_macro_delta_margin_nats'] == .5
    assert d['minimum_top1_reference_corrections'] == d['minimum_correction_dialogues'] == 3
    assert d['precedence'] == ['ST_LOC0_INVALID', 'ST_LOC0_NO_FIXED_DIRECTION_LEVER',
                               'ST_LOC0_LOCAL_EFFECT_ONLY', 'ST_LOC0_SITE_FEASIBLE']
    assert not d['correct_state_damage_is_feasibility_disqualifier']
    assert CFG['oracle']['exclude_D2_and_random'] and CFG['oracle']['post_seal_only']
    assert CFG['cell_eligibility']['minimum_valid_per_stratum'] == 57
    assert CFG['cell_eligibility']['invalid_cells_remain_in_denominator']
    assert CFG['compute']['future_max_jobs'] == 1 and CFG['compute']['hard_ceiling_seconds'] == 10800
    assert CFG['audit']['independent_no_import_primary_decision']


def test_installed_site_forward_identity():
    import inspect
    import transformers
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    frozen = CFG['site_forward']
    assert transformers.__version__ == frozen['transformers_version']
    assert 'sha256:' + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest() == frozen['forward_source_sha256']
    assert file_hash(inspect.getfile(WhisperDecoderLayer)) == frozen['module_file_sha256']
    assert frozen['eval_mode_required'] and frozen['d_model'] == 1280
