"""New A5000 roles from original TRAIN only; not historical H100 panels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reft.common import fillers
from reft.data_audit import round_robin
from reft.text import target_text, utterance_stratum

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT/'configs/reft/a5000_e1.yaml'


def config():
    return yaml.safe_load(CONF.read_text())


def output():
    return Path(config()['output_root']).expanduser()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)+'\n')
    temp.replace(path)


def runtime():
    return yaml.safe_load((ROOT/config()['runtime']).read_text())


def original_manifest():
    return Path(runtime()['output_root']).expanduser()/'e0/official_train.parquet'


def integrity(roles):
    names = list(roles)
    for name, rows in roles.items():
        assert len({r['utterance_id'] for r in rows}) == len(rows)
        assert all(r['official_split'] == 'train' for r in rows)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            for key in ['utterance_id', 'dialogue_id', 'speaker_id', 'audio_full_sha256']:
                assert not ({r[key] for r in roles[a]} & {r[key] for r in roles[b]}), (a,b,key)
    return True


def prepare():
    from transformers import WhisperTokenizer
    C = config()
    dest = ROOT/C['manifest']
    if dest.exists():
        raise FileExistsError('Versioned role manifest already exists; do not resample it')
    source = original_manifest()
    df = pd.read_parquet(source)
    assert set(df.official_split) == {'train'}
    tok = WhisperTokenizer.from_pretrained(str(Path(runtime()['model']['root']).expanduser()), local_files_only=True)
    eligible = []
    # Remove every member of duplicate prefix-hash groups before assigning roles.
    unique = df[~df.audio_sha256.duplicated(keep=False)]
    for r in unique.to_dict('records'):
        ids = tok.encode(target_text(r['transcript_raw']), add_special_tokens=False)
        if not (C['data']['duration'][0] <= r['duration_sec'] <= C['data']['duration'][1]):
            continue
        if not ids or len(ids) > C['data']['max_content_tokens']:
            continue
        st = utterance_stratum(r['transcript_raw'], fillers())
        if st['stratum'] == 'OTHER':
            continue
        eligible.append({**r, **st, 'target_tokens_incl_eos': len(ids)+1})
    dialogues = sorted({r['dialogue_id'] for r in eligible})
    rng = np.random.default_rng(C['seed'])
    dialogues = list(np.array(dialogues)[rng.permutation(len(dialogues))])
    assert len(dialogues) == sum(C['data']['dialogue_counts'].values())
    offset, roles, summaries = 0, {}, {}
    for role, n_dialogues in C['data']['dialogue_counts'].items():
        allocated = dialogues[offset:offset+n_dialogues]
        offset += n_dialogues
        rows, missing = round_robin([r for r in eligible if r['dialogue_id'] in allocated],
                                    C['data']['quotas'][role], C['seed'])
        assert not any(missing.values()), (role, missing)
        selected = []
        for r in sorted(rows, key=lambda r: r['utterance_id']):
            selected.append({k: r[k] for k in ['utterance_id', 'dialogue_id', 'speaker_id', 'conversation_id',
                'corpus_speaker_id', 'official_split', 'duration_sec', 'audio_sha256', 'stratum',
                'n_sub_en', 'n_zh', 'target_tokens_incl_eos']} |
                {'audio_full_sha256': sha(r['audio_path']),
                 'target_text_sha256': hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest()})
        roles[role] = selected
        summaries[role] = {'count': len(selected), 'dialogues': len({r['dialogue_id'] for r in selected}),
            'speakers': len({r['speaker_id'] for r in selected}), 'seconds': sum(r['duration_sec'] for r in selected),
            'strata': {s: sum(r['stratum'] == s for r in selected) for s in C['data']['quotas'][role]},
            'substantive_english_units': sum(r['n_sub_en'] for r in selected), 'allocated_dialogues': allocated}
    integrity(roles)
    atomic(dest, {'schema': 'a5000_train_only_roles_v1', 'seed': C['seed'],
        'source_manifest_sha256': sha(source), 'dataset_revision': runtime()['data']['revision'],
        'historical_replication': False, 'roles': roles, 'summaries': summaries,
        'sampling': 'seed42 permutation of 70 dialogues; 42/14/14; per-stratum round-robin over seeded within-dialogue orders'})
    print(json.dumps({'manifest': str(dest), 'sha256': sha(dest), 'summaries': summaries}, indent=1))


def load_rows(role):
    if role not in ['A5000_T394', 'A5000_V106']:
        raise ValueError('C150 is reserved; no initial-campaign reader/evaluator may open it')
    spec = json.loads((ROOT/config()['manifest']).read_text())
    assert sha(original_manifest()) == spec['source_manifest_sha256']
    df = pd.read_parquet(original_manifest()).set_index('utterance_id')
    rows = []
    for ident in spec['roles'][role]:
        r = df.loc[ident['utterance_id']].to_dict() | ident
        assert r['official_split'] == 'train' and sha(r['audio_path']) == r['audio_full_sha256']
        assert hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest() == r['target_text_sha256']
        rows.append(r)
    return rows


if __name__ == '__main__':
    prepare()
