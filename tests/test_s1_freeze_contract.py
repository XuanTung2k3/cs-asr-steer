"""CPU-only S1 freeze/provenance tests; no pretrained forward or S1 outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def read(p):
    return json.loads((ROOT / p).read_text())

def sha(p):
    return 'sha256:' + hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

def digest(v):
    return 'sha256:' + hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

CFG = read('configs/inference_cf/s1_acoustic_evidence.json')
PANEL = read(CFG['panel'])


def test_exact_historical_query_order_and_identity():
    parent = read(PANEL['parent'])
    assert PANEL['runtime_queries'] == parent['runtime_queries']
    assert len(PANEL['runtime_queries']) == 180
    assert len(PANEL['utterances']) == 80
    assert len({u['dialogue_id'] for u in PANEL['utterances']}) == 20
    assert PANEL['counts']['strata'] == {'EN-confusion': 60, 'EN-correct': 60, 'ZH-correct': 60}
    assert digest({k: v for k, v in PANEL.items() if k != 'identity_hash'}) == CFG['panel_identity_hash'] == PANEL['identity_hash']
    assert digest(PANEL['runtime_queries']) == CFG['runtime_membership_hash']
    assert sha(PANEL['parent']) == PANEL['parent_file_sha256']


def test_reference_free_queries_and_lac_input_identity():
    allowed = {'utterance_id', 'dialogue_id', 't', 'absolute_query', 'content_prefix_sha256',
               'forced_zh_query_input_sha256', 'forced_en_query_input_sha256'}
    old = read('results/inference_cf/p2sel_lac/candidates_sealed.json')
    old = {(r['utterance_id'], r['t']): r for r in old['rows']}
    utt = {u['utterance_id']: u for u in PANEL['utterances']}
    for q in PANEL['runtime_queries']:
        assert set(q) == allowed
        tokens = utt[q['utterance_id']]['baseline_content_tokens'][:q['t']]
        assert q['absolute_query'] == len(CFG['conditions']['M']) + q['t'] - 1
        assert old[q['utterance_id'], q['t']]['input_ids'] == CFG['conditions']['M'] + tokens
        assert old[q['utterance_id'], q['t']]['query'] == q['absolute_query']


@pytest.mark.parametrize('u', PANEL['utterances'], ids=lambda u: u['utterance_id'])
def test_r0_primary_full_audio_and_seal_sources(u):
    assert 'sha256:' + hashlib.sha256(Path(u['audio_path']).read_bytes()).hexdigest() == u['audio_full_file_sha256']
    assert sha(u['r0_primary_row']) == u['r0_primary_row_sha256']
    assert sha(u['r0_primary_arrays']) == u['r0_primary_arrays_sha256']
    rec = read(u['r0_primary_row'])
    assert rec['audio_sha256'] == u['audio_full_file_sha256']
    seal = read(PANEL['r0_output_seal']['path'])
    assert seal['files'][u['r0_primary_row']] == u['r0_primary_row_sha256']
    assert seal['files'][u['r0_primary_arrays']] == u['r0_primary_arrays_sha256']


def test_lac_archives_cover_all_queries_without_reading_outcomes():
    rows = {r['utterance_id']: r for r in PANEL['lac_sources']}
    for uid, r in rows.items():
        assert sha(r['row']) == r['row_sha256']
        assert sha(r['logits']) == r['logits_sha256']
        # Only array-key metadata, never candidate ranks or S1 outcomes.
        with np.load(ROOT / r['logits'], allow_pickle=False) as a:
            for q in PANEL['runtime_queries']:
                if q['utterance_id'] == uid:
                    assert f"t{q['t']}_masked" in a.files


def test_pinned_sources_and_region_guards():
    for p, h in CFG['source_sha256'].items():
        assert sha(p) == h
    r0 = read('configs/inference_cf/r0_region_vector.json')
    assert CFG['regions']['source_config'] == r0['regions']
    assert CFG['regions']['alignment_heads'] == r0['mapping']['alignment_heads']
    assert CFG['regions']['attention'].find('>=0.5') >= 0
    assert CFG['counterfactual']['off_target']['energy_ratio_range'] == [0.5, 2.0]


def test_candidate_and_scoring_contract_is_not_script_or_gold_selected():
    assert CFG['candidates']['budgets'] == [5, 20]
    assert CFG['candidates']['primary_budget'] == 20
    assert CFG['candidates']['branches'] == ['M', 'E', 'AUTO']
    assert CFG['conditions']['auto_available'] is True
    assert CFG['conditions']['auto_forced_decoder_ids'] is None
    assert CFG['candidates']['masked_candidate_regeneration'] is False
    assert CFG['scoring']['lambda'] == 1.0
    assert CFG['scoring']['tie'] == 'lower token ID'
    assert CFG['firewall']['gold_in_runner'] is False
    assert CFG['firewall']['oracle_regions'] is False
    assert CFG['firewall']['training'] is False
    assert CFG['firewall']['autograd'] is False
    assert CFG['firewall']['steering'] is False
    assert CFG['firewall']['candidate_seal_remote_before_masks'] is True
    assert CFG['firewall']['final_seal_remote_and_PRIMARY_before_references'] is True


def test_gates_and_multiplicity_are_preregistered_not_margin_only():
    h, d, b = CFG['gate_H'], CFG['gate_D'], CFG['bootstrap']
    assert (h['union20_reference_hits_min'], h['union20_hit_dialogues_min'], h['incremental_vs_M20_min'], h['incremental_dialogues_min']) == (12, 6, 5, 3)
    assert (d['paired_accessible_rows_min'], d['paired_dialogues_min']) == (12, 6)
    assert d['net_hit1_gain_vs_M_min'] == 3
    assert d['new_ZH_correct_corruptions_max'] == 2
    assert d['new_wrong_English_promotions_ZH_max'] == 1
    assert b['family_size'] == len(CFG['candidates']['budgets']) * len(d['mrr_macro_gain_min']) == 8
    assert b['lower_quantile'] == b['alpha'] / (2 * b['family_size'])
    assert b['draws'] == 10000 and b['seed'] == 240924
    assert CFG['labels'] == ['S1_INVALID', 'S1_BLOCKED_CANDIDATE_SEMANTICS',
        'S1_CANDIDATE_HEADROOM_INSUFFICIENT', 'S1_ACOUSTIC_DISCRIMINATION_INSUFFICIENT',
        'S1_PARTIAL_FEASIBILITY', 'S1_READY_FOR_S2']
    assert CFG['compute']['jobs_max'] == 2 and CFG['compute']['hard_seconds_per_job'] == 10800


def test_canonical_documents_are_additive_and_design_is_not_a_run():
    for p in ('STATUS', 'CODE_MAP', 'DATA_EXPOSURE'):
        text = (ROOT / f'docs/current/{p}.md').read_text()
        assert 'ST-PROMPT-R1' in text and 'S1' in text
    assert CFG['status'] == 'DESIGN_FROZEN_IMPLEMENTATION_PENDING'
    assert not (ROOT / 'docs/inference_cf/S1_REPORT.md').exists()


@pytest.mark.parametrize('detected', [50259, 50260, 50265])
def test_installed_native_auto_same_prefix_prompt_with_stub_no_model_forward(detected):
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
    gen_path = Path(CFG['model']['dir']) / 'generation_config.json'
    gen = GenerationConfig.from_dict(json.loads(gen_path.read_text()))
    gen.language, gen.task, gen.return_timestamps, gen.forced_decoder_ids = None, 'transcribe', False, None
    calls = []
    def detect(**kwargs):
        calls.append(kwargs)
        return torch.tensor([detected])
    stub = SimpleNamespace(device=torch.device('cpu'), detect_language=detect)
    encoder = object()
    tokens = WhisperGenerationMixin._retrieve_init_tokens(stub, None, 1, gen,
        SimpleNamespace(forced_decoder_ids=None), 3000, {'encoder_outputs': encoder})
    assert tokens.tolist() == [[50258, detected, 50360, 50364]]
    assert len(calls) == 1 and calls[0]['encoder_outputs'] is encoder
    assert set(gen.lang_to_id.values()) == set(CFG['regions']['source_config']['language_ids'])
    installed = Path(CFG['conditions']['installed_generation_source'])
    assert 'sha256:' + hashlib.sha256(installed.read_bytes()).hexdigest() == CFG['conditions']['installed_generation_file_sha256']


def test_complete_canonical_original_contents_preserved():
    import subprocess
    for name in ('STATUS', 'CODE_MAP', 'DATA_EXPOSURE'):
        path = f'docs/current/{name}.md'
        old = subprocess.check_output(['git', 'show', f"{CFG['starting_head']}:{path}"], cwd=ROOT)
        assert (ROOT / path).read_bytes().startswith(old)
