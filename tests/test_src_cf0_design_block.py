"""CPU-only provenance / coverage review. No model loading or new scientific outcomes."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = 'da10689870ef44ee2a382c25a4483a3d125895d6'


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return 'sha256:' + hashlib.sha256(s.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return 'sha256:' + h.hexdigest()


def test_block_is_not_execution_authorization():
    c = read('configs/inference_cf/src_cf0.json')
    assert c['status'] == 'SRC_CF0_DESIGN_BLOCKED'
    assert c['execution_authorized'] is False
    assert c['scientific_gates_frozen'] is False
    assert c['next_authorized_scientific_action'] is None
    assert c['requested_constraints']['primary_arms'] == 8
    assert c['requested_constraints']['layers'] == [16, 24]
    assert c['requested_constraints']['eta'] == [.15, .30]


def test_panel_identity_and_unmodified_membership():
    p = read('docs/inference_cf/SRC_CF0_PANEL.json')
    par = read(p['parent'])
    assert p['identity_hash'] == digest({k: v for k, v in p.items() if k != 'identity_hash'})
    assert p['runtime_queries'] == par['runtime_queries']
    assert p['utterances'] == par['utterances']
    assert p['runtime_membership_hash'] == par['runtime_membership_hash']
    assert p['audio_membership_hash'] == par['audio_membership_hash']
    assert len(p['runtime_queries']) == 180
    assert len(p['utterances']) == 80
    assert len({q['dialogue_id'] for q in p['runtime_queries']}) == 20
    assert p['new_model_inference'] is False
    assert p['new_direction_or_pulse_outcomes'] is False
    assert read('configs/inference_cf/src_cf0.json')['panel_identity_hash'] == p['identity_hash']


@pytest.mark.parametrize('stratum,target,td,paired,pd', [
    ('EN-confusion', 28, 12, 20, 9), ('EN-correct', 39, 17, 35, 16), ('ZH-correct', 6, 6, 4, 4)])
def test_independent_sealed_coverage(stratum, target, td, paired, pd):
    p = read('docs/inference_cf/SRC_CF0_PANEL.json')
    membership = read('docs/inference_cf/ST_PROMPT_R1_PANEL.json')['evaluation_membership']
    raw = {}
    for f in sorted((ROOT / 'results/inference_cf/s1/run1/candidates').glob('*.json')):
        d = json.loads(f.read_text())
        for q in d['queries']:
            assert q['j'] not in raw
            raw[q['j']] = (d['utterance_id'], q)
    assert sorted(raw) == list(range(180))
    for kind, expected, dialogues in [('target', target, td), ('paired', paired, pd)]:
        ids = []
        for j, m in enumerate(membership):
            uid, q = raw[j]
            assert (uid, q['t']) == (m['utterance_id'], m['t'])
            ok = q['region']['target']['status'] == 'OK'
            if kind == 'paired':
                ok = ok and q['region']['off_target']['status'] == 'OK'
            if m['stratum'] == stratum and ok:
                ids.append(j)
        assert len(ids) == expected
        assert len({membership[j]['dialogue_id'] for j in ids}) == dialogues
        rec = p['coverage_review_only'][stratum][kind]
        assert rec['query_indices'] == ids
        assert (rec['rows'], rec['dialogues']) == (expected, dialogues)


def test_inventory_is_exact_s1_region_record():
    p = read('docs/inference_cf/SRC_CF0_PANEL.json')
    for r in p['sealed_region_inventory']:
        assert file_sha(r['source']) == r['source_sha256']
        d = read(r['source'])
        q = next(q for q in d['queries'] if q['j'] == r['j'])
        assert r['target'] == q['region']['target']
        assert r['off_target'] == q['region']['off_target']
        assert r['paired_available'] == q['region']['paired_available']
        assert r['absolute_query'] == q['query'] == 4 + r['t'] - 1


@pytest.mark.parametrize('seal_path', ['results/inference_cf/s1/run1/candidates_sealed.json',
                                      'results/inference_cf/s1/output_seal.json'])
def test_existing_seal_files(seal_path):
    seal = read(seal_path)
    assert seal['status'] == 'SEALED'
    assert seal['references_used'] is False
    for path, expected in seal['files'].items():
        assert file_sha(path) == expected, path


def test_pinned_sources_and_history_preserved():
    p = read('docs/inference_cf/SRC_CF0_PANEL.json')
    for path, expected in p['sources'].items():
        assert file_sha(path) == expected
        original = subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT)
        assert (ROOT / path).read_bytes() == original
    status = (ROOT / 'docs/current/STATUS.md').read_bytes()
    old = subprocess.check_output(['git', 'show', f'{BASE}:docs/current/STATUS.md'], cwd=ROOT)
    assert status.startswith(old)


def test_review_has_no_runtime_or_fake_positive_label():
    assert not (ROOT / 'experiments/inference_cf_src_cf0.py').exists()
    assert not (ROOT / 'slurm/inference_cf_src_cf0.sbatch').exists()
    assert not (ROOT / 'docs/inference_cf/SRC_CF0_REPORT.md').exists()
    assert 'scientific gates are not frozen' in (ROOT / 'docs/inference_cf/SRC_CF0_SPEC.md').read_text()
