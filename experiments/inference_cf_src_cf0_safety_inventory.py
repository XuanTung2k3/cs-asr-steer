"""CPU-only source/region bounds; never an intervention or correctness census.

Run from the repository root. Only the R0 PRIMARY heard-region array is loaded;
full-replay attention, residuals, oracle regions and lexical outcomes are unused.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INITIAL_HEAD = '20cd36824822c356f910f42c9b5ca3a5fd5f56b6'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return 'sha256:' + h.hexdigest()


def digest(obj):
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)
    return 'sha256:' + hashlib.sha256(data.encode()).hexdigest()


def count(queries):
    return {'queries': len(queries),
            'utterances': len({q['utterance_id'] for q in queries}),
            'dialogues': len({q['dialogue_id'] for q in queries})}


def build_inventory():
    r = read('docs/inference_cf/R0_PANEL.json')
    s = read('docs/inference_cf/S1_PANEL.json')
    seal = read('results/inference_cf/r0/output_seal.json')
    s1_seal_path = 'results/inference_cf/s1/run1/candidates_sealed.json'
    s1_seal = read(s1_seal_path)
    assert r['identity_hash'] == digest({k: v for k, v in r.items() if k != 'identity_hash'})
    assert len(r['rows']) == 300
    assert len({q['dialogue_id'] for q in r['rows']}) == 20
    su = {u['utterance_id']: u for u in s['utterances']}
    rq = {(q['utterance_id'], q['t']): q for q in s['runtime_queries']}
    regions = {}
    for path in sorted(Path('results/inference_cf/s1/run1/candidates').glob('*.json')):
        assert sha(path) == s1_seal['files'][str(path)]
        d = read(path)
        for q in d['queries']:
            key = (d['utterance_id'], q['t'])
            assert key not in regions
            regions[key] = q['region']

    rows, compatible = [], []
    for row in r['rows']:
        i, uid = row['canonical_index'], row['utterance_id']
        jp = f'results/inference_cf/r0/run1/rows/{i:03d}.json'
        npz = jp[:-5] + '.npz'
        for path in (jp, npz):
            assert sha(path) == seal['files'][path]
        assert sha(row['audio']['path']) == row['audio']['full_sha256']
        d = read(jp)
        assert uid == d['utterance_id']
        assert d['content_sha256'] == row['baseline']['content_sha256']
        with np.load(npz, allow_pickle=False) as arrays:
            intervals = arrays['track_heard_intervals'].tolist()
        ts = [q['t'] for q in d['queries'] if q['eligible']]
        en = [a for a in intervals if a[2] == 1 and a[1] - a[0] >= 4000]
        rows.append({
            'canonical_index': i, 'utterance_id': uid,
            'dialogue_id': row['dialogue_id'],
            'audio_sha256': row['audio']['full_sha256'],
            'content_sha256': row['baseline']['content_sha256'],
            'r0_json': jp, 'r0_json_sha256': sha(jp),
            'r0_arrays': npz, 'r0_arrays_sha256': sha(npz),
            'structural_query_count': len(ts),
            'structural_query_membership_hash': digest(ts),
            'primary_heard_intervals_hash': digest(intervals),
            'eligible_EN_interval_count': len(en),
            'region_only_possible_target_queries': len(ts) if en else 0,
        })
        for t in ts:
            if (uid, t) not in rq:
                continue
            u = su[uid]
            assert u['audio_full_file_sha256'] == row['audio']['full_sha256']
            if u['baseline_content_tokens'][:t] != row['baseline']['content_ids'][:t]:
                continue
            region = regions[(uid, t)]
            target = region['target']['status'] == 'OK'
            paired = target and region['off_target']['status'] == 'OK'
            assert paired == region['paired_available']
            compatible.append({'utterance_id': uid, 'dialogue_id': row['dialogue_id'],
                               't': t, 'target': target, 'paired': paired})

    out = {
        'schema': 'SRC_CF0_FULL300_CPU_INVENTORY_V1', 'initial_head': INITIAL_HEAD,
        'new_model_inference': False, 'references_accessed': False,
        'exact_full300_intervention_census': False,
        'parent': 'docs/inference_cf/R0_PANEL.json',
        'parent_sha256': sha('docs/inference_cf/R0_PANEL.json'),
        'parent_identity_hash': r['identity_hash'],
        'parent_membership_hash': r['membership_hash'],
        'r0_seal_sha256': sha('results/inference_cf/r0/output_seal.json'),
        's1_panel_sha256': sha('docs/inference_cf/S1_PANEL.json'),
        's1_candidate_seal_sha256': sha(s1_seal_path),
        'rows': rows, 'compatible_s1_queries': compatible,
        'summary': {
            'rows': len(rows), 'dialogues': len({x['dialogue_id'] for x in rows}),
            'baseline_content_tokens': sum(len(x['baseline']['content_ids']) for x in r['rows']),
            'structural_queries': sum(x['structural_query_count'] for x in rows),
            'compatible_s1_queries': count(compatible),
            'certified_target_lower_bound': count([x for x in compatible if x['target']]),
            'certified_paired_lower_bound': count([x for x in compatible if x['paired']]),
            'region_only_target_upper_bound': {
                'queries': sum(x['region_only_possible_target_queries'] for x in rows),
                'utterances': sum(x['region_only_possible_target_queries'] > 0 for x in rows),
                'dialogues': len({x['dialogue_id'] for x in rows
                                  if x['region_only_possible_target_queries'] > 0}),
            },
        },
        'limitations': [
            'S1 lower bounds require exactly unchanged cached-query model/prompt/attention/preprocessing semantics; no full-replay attention substituted.',
            'Region-only upper bound ignores heard mass, attention association, waveform energy and off-target constraints.',
            'No hidden-state, reachability or correct-Mandarin membership census exists under the future cached path.',
        ],
    }
    out['identity_hash'] = digest(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/inference_cf/SRC_CF0_FULL300_INVENTORY.json')
    args = parser.parse_args()
    inventory = build_inventory()
    Path(args.output).write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(inventory['summary'], indent=2))
    print(inventory['identity_hash'])
