"""R0 design preflight: sealed inputs and synthetic CPU feasibility, no R0 outcomes."""
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from csasr.inference_cf.core import digest, file_hash
from csasr.lss.sites import DecoderPostCrossAttnRecorder

CFG = json.loads(Path('configs/inference_cf/r0_region_vector.json').read_text())
PANEL = json.loads(Path(CFG['panel']).read_text())


def test_full300_identity_and_label_free_projection():
    assert digest({k: v for k, v in PANEL.items() if k != 'identity_hash'}) == PANEL['identity_hash'] == CFG['panel_identity_hash']
    parent = json.loads(Path(PANEL['parent']).read_text())
    assert file_hash(PANEL['parent']) == PANEL['parent_sha256']
    rows = PANEL['rows']
    assert len(rows) == len({r['utterance_id'] for r in rows}) == 300
    assert Counter(r['dialogue_id'] for r in rows) == Counter(r['dialogue_id'] for r in parent['rows'])
    assert len(Counter(r['dialogue_id'] for r in rows)) == 20
    assert set(Counter(r['dialogue_id'] for r in rows).values()) == {15}
    assert [r['utterance_id'] for r in rows] == [r['utterance_id'] for r in parent['rows']]
    assert PANEL['membership_hash'] == CFG['membership_hash'] == digest([
        {k: r[k] for k in ('utterance_id', 'dialogue_id')} for r in rows])
    assert PANEL['role_source']['independent_identity_check']
    assert PANEL['role_source']['read_columns'] == ['utterance_id', 'dialogue_id', 'role']
    assert file_hash(PANEL['role_source']['path']) == PANEL['role_source']['file_sha256']
    for r in rows:
        assert set(r) == set(CFG['firewall']['runner_projection'])
        assert r['audio']['baseline_heard_samples'] == min(480000, r['audio']['resampled_num_samples'])
        b = r['baseline']
        assert file_hash(b['path']) == b['file_sha256']
        raw = json.loads(Path(b['path']).read_text())
        assert raw['identity'] == r['utterance_id']
        assert raw['theta0']['tokens'] == b['content_ids']
        assert digest(b['content_ids']) == b['content_sha256']
        assert raw['theta0']['terminated'] == b['terminated']
    assert sum(r['audio']['resampled_num_samples'] > 480000 for r in rows) == 27
    assert not PANEL['references_used_for_selection'] and not PANEL['new_R0_outcomes_inspected']


def test_inherited_sources_interfaces_and_provider_identity():
    for name, expected in CFG['source_sha256'].items():
        assert file_hash(name) == expected
    gen = json.loads(Path(CFG['model']['dir'], 'generation_config.json').read_text())
    assert len(CFG['regions']['language_ids']) == 100
    assert sorted(set(gen['lang_to_id'].values())) == CFG['regions']['language_ids']
    assert digest(gen['lang_to_id']) == CFG['regions']['language_map_hash']
    assert gen['lang_to_id']['<|en|>'] == 50259 and gen['lang_to_id']['<|zh|>'] == 50260
    assert gen['alignment_heads'] == CFG['mapping']['alignment_heads']
    from experiments.inference_cf_p0_r2 import native_lid, full_replay
    from csasr.inference_cf import unique
    assert list(inspect.signature(native_lid).parameters) == ['bundle', 'waveform', 'language_ids']
    assert list(inspect.signature(full_replay).parameters) == ['bundle', 'encoded', 'prompt', 'content', 'attention']
    assert unique.RANK == 32
    assert CFG['unique']['candidate_ranks'] == [2, 4, 8]  # deliberately separate method
    assert not CFG['regions']['null_correction_in_primary']
    assert not CFG['unique']['center_moments']
    assert CFG['oracle']['replace_only_membership'] and not CFG['oracle']['gold_teacher_forcing']


def grid(n):
    """Independent normative grid check; production implementation is future work."""
    if n < 16000:
        return [(0, n)]
    starts = list(range(0, n-16000+1, 8000))
    return [(s, s+16000) for s in sorted(set(starts + [n-16000]))]


def test_window_grid_geometry_cost_and_exact_tail():
    assert grid(7999) == [(0, 7999)]
    assert grid(16000) == [(0, 16000)]
    assert grid(24001) == [(0, 16000), (8000, 24000), (8001, 24001)]
    count = 0
    for r in PANEL['rows']:
        n = r['audio']['resampled_num_samples']
        bounds = grid(n)
        assert bounds[0][0] == 0 and bounds[-1][1] == n
        assert len(bounds) == len(set(bounds))
        assert all(0 <= a < b <= n and b-a <= 16000 for a, b in bounds)
        count += len(bounds)
    assert count == 10387
    assert CFG['compute']['max_scientific_gpu_jobs'] == 1
    assert CFG['compute']['hard_seconds'] == 10800


def participation(h):
    s = np.linalg.svd(h-h.mean(axis=0), compute_uv=False)
    w = s*s
    return 0.0 if w.sum() == 0 else float(w.sum()**2 / (w @ w))


def test_count_is_not_effective_rank_and_thin_svd_is_uncentered_moment():
    h = np.tile(np.arange(12, dtype=np.float64), (20, 1))
    assert participation(h) == 0
    assert np.linalg.matrix_rank(h) == 1  # 20 consecutive copies cannot support rank2
    rng = np.random.default_rng(240924)
    h = rng.normal(size=(18, 12)) + np.arange(12)
    _, s, vt = np.linalg.svd(h, full_matrices=False)
    moment = h.T @ h / len(h)
    vals, vec = np.linalg.eigh(moment)
    np.testing.assert_allclose(s*s/len(h), vals[::-1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vt[:2].T @ vt[:2], vec[:, -2:] @ vec[:, -2:].T, atol=1e-10)
    assert participation(h) >= 2
    assert not np.allclose(moment, (h-h.mean(0)).T @ (h-h.mean(0))/len(h))


def test_circular_control_preserves_sizes_and_is_deterministic():
    labels = np.zeros(24000, dtype=np.int8)
    labels[3000:9000] = 1
    labels[11000:21000] = 2
    key = 'R0-shuffle-v1|240924|synthetic'
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    offsets = np.sort(np.random.default_rng(seed).choice(np.arange(1600, len(labels)-1600+1), size=20, replace=False))
    again = np.sort(np.random.default_rng(seed).choice(np.arange(1600, len(labels)-1600+1), size=20, replace=False))
    np.testing.assert_array_equal(offsets, again)
    assert len(set(offsets.tolist())) == 20
    for offset in offsets:
        assert Counter(np.roll(labels, int(offset))) == Counter(labels)
    assert CFG['controls']['shuffles'] == 20
    assert CFG['controls']['boundary_samples'] == 1600


def test_four_exact_sites_and_no_future_decoder_token_access():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    conf = WhisperConfig(vocab_size=40, num_mel_bins=8, encoder_layers=1, decoder_layers=32,
        encoder_attention_heads=2, decoder_attention_heads=20, d_model=40, encoder_ffn_dim=80,
        decoder_ffn_dim=80, max_source_positions=10, max_target_positions=20,
        decoder_start_token_id=1, pad_token_id=0, bos_token_id=1, eos_token_id=2, suppress_tokens=[])
    conf._attn_implementation = 'eager'
    with torch.random.fork_rng():
        torch.manual_seed(240924)
        model = WhisperForConditionalGeneration(conf).eval()
        for p in model.parameters(): p.requires_grad_(False)
        bundle = SimpleNamespace(model=model, decoder_layer=lambda i: model.model.decoder.layers[i])
        features = torch.randn(1, 8, 20)
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8]])
    consumed = {}
    handles = [model.model.decoder.layers[l].final_layer_norm.register_forward_pre_hook(
        lambda _m, args, l=l: consumed.__setitem__(l, args[0].detach().clone())) for l in CFG['layers']]
    try:
        with torch.inference_mode(), DecoderPostCrossAttnRecorder(bundle, CFG['layers']) as rec:
            out = model(input_features=features, decoder_input_ids=ids, use_cache=False,
                        output_attentions=True, return_dict=True)
        for l in CFG['layers']:
            assert torch.equal(rec.states[l], consumed[l])
            assert torch.equal(rec.states[l], rec.q_states[l] + rec.u_source_states[l])
        with torch.inference_mode(), DecoderPostCrossAttnRecorder(bundle, CFG['layers']) as short:
            cut = model(input_features=features, decoder_input_ids=ids[:, :5], use_cache=False,
                        output_attentions=True, return_dict=True)
        for l in CFG['layers']:
            torch.testing.assert_close(rec.states[l][:, :5], short.states[l], rtol=1e-5, atol=1e-6)
        for l, h in CFG['mapping']['alignment_heads']:
            torch.testing.assert_close(out.cross_attentions[l][:, h, :5], cut.cross_attentions[l][:, h], rtol=1e-5, atol=1e-6)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    finally:
        for h in handles: h.remove()


def test_frozen_family_gates_firewall_and_no_automatic_partial_progression():
    assert CFG['layers'] == [3, 8, 16, 24]
    assert CFG['decision_precedence'] == ['R0_INVALID', 'R0_ORACLE_CONSTRUCTION_INSUFFICIENT',
        'R0_PREDICTED_REGION_INSUFFICIENT', 'R0_VECTOR_UNSTABLE', 'R0_REGION_SIGNAL_INSUFFICIENT',
        'R0_PARTIAL_FEASIBILITY', 'R0_READY_FOR_R1']
    b = CFG['bootstrap']
    assert b['family_size'] == len(CFG['layers'])*2
    assert b['lower_quantile'] == b['alpha']/(2*b['family_size'])
    assert b['upper_quantile'] == 1-b['lower_quantile']
    assert b['replicates'] == 10000 and b['minimum_valid_draws'] == 9900
    assert CFG['gates']['ready_layers_min'] == 2 and CFG['gates']['ready_requires_layer_16_or_24']
    assert not any(CFG['firewall'][k] for k in ['oracle_in_runner', 'gold_in_provider', 'steering', 'autograd', 'training'])
    assert CFG['firewall']['seal_then_independent_audit_then_oracle']
    assert CFG['secondary_predictor']['role'].startswith('descriptive')
    spec = Path('docs/inference_cf/R0_REGION_VECTOR_SPEC.md').read_text()
    assert 'NO automatic progression' in spec
    assert 'reference-derived timing oracle' in spec
