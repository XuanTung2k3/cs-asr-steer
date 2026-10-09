"""CPU-only SRD2-G0 identity inventory; never imports a model or evaluator.

This is design preparation, not the scientific runner. Historical files are
projected to utterance identifiers only. Role parquet reads use an explicit
metadata column allowlist. Selection never uses durations or lexical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLE = Path('/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet')
SEED = 240924
COLUMNS = ('utterance_id', 'dialogue_id', 'role', 'audio_path', 'audio_sha256', 'duration_sec')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return 'sha256:' + h.hexdigest()


def digest(obj):
    return 'sha256:' + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def order_key(uid):
    return hashlib.sha256(f'SRD2-G0-population-v1|{SEED}|{uid}'.encode()).hexdigest()


def identity_projection(obj):
    """Only declared identity fields and UID-shaped mapping keys affect exclusions."""
    ids = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ('utterance_id', 'identity') and isinstance(value, str):
                ids.add(value)
            elif key in ('utterance_ids', 'utterances') and isinstance(value, list):
                ids.update(x for x in value if isinstance(x, str))
            if isinstance(key, str) and key.startswith('ZH-CN_U'):
                ids.add(key)
            if isinstance(value, (list, dict)):
                ids.update(identity_projection(value))
    elif isinstance(obj, list):
        for value in obj:
            ids.update(identity_projection(value))
    return ids


def exclusions():
    paths = [ROOT / 'docs/inference_cf/R0_PANEL.json',
             ROOT / 'docs/inference_cf/ST_LOC0_PANEL.json',
             ROOT / 'results/dg04/results/B0.json',
             ROOT / 'results/dg03/screen/dg03_screen_L16.json',
             ROOT / 'results/dg03/screen/dg03_screen_L24.json',
             ROOT / 'results/basis_a4/panels/cs_dialogue_300.json']
    paths += sorted((ROOT / 'results/round1/job_a/cells').glob('*.json'))
    paths += [ROOT / f'results/job_a/F{i}_hypotheses.jsonl' for i in (3, 4, 5)]
    registry, sources = {}, []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == '.jsonl':
            with path.open() as f:
                ids = {json.loads(line)['utterance_id'] for line in f if line.strip()}
        else:
            ids = identity_projection(json.loads(path.read_text()))
        rel = str(path.relative_to(ROOT))
        sources.append({'path': rel, 'sha256': sha(path), 'identity_count': len(ids)})
        for uid in ids:
            registry.setdefault(uid, []).append(rel)
    registry.setdefault('ZH-CN_U0091_S0_68', []).append('docs/current/DATA_EXPOSURE.md:DG-02')
    sources.append({'path': 'docs/current/DATA_EXPOSURE.md',
                    'sha256': sha(ROOT / 'docs/current/DATA_EXPOSURE.md'), 'identity_count': 1})
    return registry, sources


def build():
    import pyarrow.parquet as pq
    import soundfile as sf
    rows = pq.read_table(ROLE, columns=list(COLUMNS)).to_pylist()
    if len({r['utterance_id'] for r in rows}) != len(rows) or any(r['role'] != 'D-dev-select' for r in rows):
        raise ValueError('role identity failure')
    excluded, sources = exclusions()
    roster, groups = [], {}
    for r in sorted(rows, key=lambda x: x['utterance_id']):
        uid, dialogue = r['utterance_id'], r['dialogue_id']
        reasons = list(excluded.get(uid, []))
        if not Path(r['audio_path']).is_file():
            reasons.append('missing_audio')
        item = {'utterance_id': uid, 'dialogue_id': dialogue,
                'exclusion_reasons': reasons, 'order_key': order_key(uid)}
        roster.append(item)
        if not reasons:
            groups.setdefault(dialogue, []).append(r)
    if len(groups) != 20 or any(len(v) < 20 for v in groups.values()):
        raise ValueError('cannot select 20 per dialogue')
    selected = []
    for dialogue in sorted(groups):
        for r in sorted(groups[dialogue], key=lambda x: (order_key(x['utterance_id']), x['utterance_id']))[:20]:
            info = sf.info(r['audio_path'])
            selected.append({'canonical_index': len(selected), 'utterance_id': r['utterance_id'],
                'dialogue_id': dialogue, 'audio_path': r['audio_path'],
                'audio_full_sha256': sha(r['audio_path']), 'role_audio_sha256': r['audio_sha256'],
                'source_duration_sec': float(r['duration_sec']), 'sample_rate': info.samplerate,
                'channels': info.channels, 'frames': info.frames, 'order_key': order_key(r['utterance_id'])})
    full300 = identity_projection(json.loads((ROOT / 'docs/inference_cf/R0_PANEL.json').read_text()))
    if len(full300) != 300 or full300 & {r['utterance_id'] for r in selected}:
        raise ValueError('FULL300 exclusion failure')
    manifest = {'schema': 'srd2_g0_population_v1', 'role': 'D-dev-select', 'seed': SEED,
        'source': str(ROLE), 'source_sha256': sha(ROLE), 'source_columns_read': list(COLUMNS),
        'selection_rule': '20 per sorted dialogue; smallest SHA256(tag|seed|UID), UID tie-break',
        'selection_tag': 'SRD2-G0-population-v1', 'exclusion_sources': sources,
        'roster_count': len(roster), 'eligible_count': sum(not r['exclusion_reasons'] for r in roster),
        'excluded_count': sum(bool(r['exclusion_reasons']) for r in roster),
        'eligible_per_dialogue': {d: len(v) for d, v in sorted(groups.items())},
        'roster': roster, 'roster_hash': digest(roster), 'selected': selected,
        'selected_hash': digest(selected), 'selected_ids_hash': digest([r['utterance_id'] for r in selected]),
        'FULL300_ids_hash': digest(sorted(full300)), 'FULL300_overlap': 0,
        'selection_uses_reference_or_outcome': False,
        'exposure_limit': 'known intervention UID sources inventoried; no untouched-dialogue claim'}
    manifest['manifest_hash'] = digest(manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError('never overwrite a frozen population')
    result = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({k: result[k] for k in ('roster_count', 'eligible_count', 'excluded_count',
                    'selected_hash', 'selected_ids_hash', 'manifest_hash')}))


if __name__ == '__main__':
    main()
