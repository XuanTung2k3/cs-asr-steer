"""New TRAIN-only roles, retaining previous dialogue allocation and C150 seal."""
import hashlib
from pathlib import Path
from .state import sha, atomic, read

ROLE_NAMES=('A5000_T2000','A5000_V300','A5000_C500')


def validate(roles, expected=None):
    if set(roles)!=set(ROLE_NAMES): raise ValueError('Unexpected/missing roles')
    for name,rows in roles.items():
        if expected and len(rows)!=expected[name]['n']: raise ValueError('Role count mismatch')
        if not rows or any(r.get('official_split')!='train' for r in rows): raise ValueError('Only original TRAIN allowed')
        for key in ['utterance_id','audio_full_sha256','waveform_sha256']:
            if any(not r.get(key) for r in rows) or len({r[key] for r in rows})!=len(rows):
                raise ValueError('Missing or duplicate '+key)
    for i,a in enumerate(ROLE_NAMES):
        for b in ROLE_NAMES[i+1:]:
            for key in ['utterance_id','dialogue_id','speaker_id','corpus_speaker_id','audio_full_sha256','waveform_sha256']:
                if {r[key] for r in roles[a]} & {r[key] for r in roles[b]}: raise ValueError('Role leakage: '+key)
    return True


def prepare(cfg,root):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from transformers import WhisperTokenizer
    from reft.text import target_text,utterance_stratum
    from reft.common import fillers
    from reft.data_audit import round_robin
    root=Path(root); p=root/'data_manifest.json'
    if p.exists():
        doc=read(p); validate(doc['roles'],cfg['data']['roles']); return doc
    if sha(cfg['original_manifest'])!=cfg['original_manifest_sha256']: raise ValueError('Original manifest identity changed')
    # Earlier completion automation must not have exposed the C150 dialogue set.
    prior=Path(cfg['prior_pilot_root'])
    if (prior/'confirmation').exists() and any((prior/'confirmation').rglob('*predictions*')):
        raise ValueError('Prior C150 exposure: cannot silently relabel those dialogues fresh')
    old=read(cfg['prior_roles']); df=pd.read_parquet(cfg['original_manifest'])
    if set(df.official_split)!={'train'}: raise ValueError('Nontraining manifest')
    tok=WhisperTokenizer.from_pretrained(cfg['model_root'],local_files_only=True)
    eligible=[]; exclusions={}; seed=cfg['data']['seed']
    duplicate_prefixes=set(df.loc[df.audio_sha256.duplicated(keep=False),'audio_sha256'])
    for r in df.to_dict('records'):
        text=target_text(r['transcript_raw']); tokens=tok.encode(text,add_special_tokens=False)
        why='duplicate_source_audio_prefix' if r['audio_sha256'] in duplicate_prefixes else None
        if not cfg['data']['duration'][0]<=r['duration_sec']<=cfg['data']['duration'][1]: why='canonical_duration_1_to_30'
        elif not tokens or len(tokens)>cfg['data']['max_content_tokens']: why='canonical_content_token_cap190'
        st=utterance_stratum(r['transcript_raw'],fillers())
        if st['stratum']=='OTHER': why='OTHER_no_lexical_units'
        if why: exclusions[why]=exclusions.get(why,0)+1; continue
        eligible.append({**r,**st,'target_tokens_incl_eos':len(tokens)+1})
    roles={}; summaries={}
    for name,spec in cfg['data']['roles'].items():
        dialogues=set(old['summaries'][spec['prior']]['allocated_dialogues'])
        candidates=[r for r in eligible if r['dialogue_id'] in dialogues]
        picked,missing=round_robin(candidates,spec['quotas'],seed)
        if any(missing.values()): raise ValueError(f'Insufficient strata for {name}: {missing}')
        rows=[]
        for r in sorted(picked,key=lambda r:r['utterance_id']):
            audio,sr=sf.read(r['audio_path'],dtype='int16',always_2d=True)
            if sr!=16000: raise ValueError('Unexpected sampling rate')
            waveform=hashlib.sha256(audio.tobytes()).hexdigest()
            keys=['utterance_id','dialogue_id','speaker_id','corpus_speaker_id','conversation_id','official_split',
                  'duration_sec','audio_path','stratum','n_sub_en','n_zh','target_tokens_incl_eos']
            rows.append({k:r[k] for k in keys}|{'audio_full_sha256':sha(r['audio_path']),
                'waveform_sha256':waveform,'target_text_sha256':hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest()})
        roles[name]=rows
        summaries[name]={'n':len(rows),'dialogues':len({r['dialogue_id'] for r in rows}),
            'speakers':len({r['speaker_id'] for r in rows}),'strata':spec['quotas'],
            'audio_seconds':sum(r['duration_sec'] for r in rows),'english_units':sum(r['n_sub_en'] for r in rows)}
    validate(roles,cfg['data']['roles'])
    doc={'schema':'new_a5000_2000_300_500_v1','historical_replication':False,'dataset_revision':cfg['dataset_revision'],
        'source_manifest_sha256':sha(cfg['original_manifest']),'prior_dialogue_allocation_sha256':sha(cfg['prior_roles']),
        'sampling':'Prior 42/14/14 dialogue allocation preserved; seeded per-stratum dialogue round robin',
        'seed':seed,'eligibility_exclusions':exclusions,'roles':roles,'summaries':summaries}
    atomic(p,doc); return doc


def rows(cfg,root,role,confirmation=None):
    if role not in ROLE_NAMES: raise ValueError('Locked/unrecognized role')
    if role=='A5000_C500':
        if confirmation is None or not Path(confirmation).is_file(): raise ValueError('C500 sealed')
        freeze=read(confirmation)
        if freeze.get('data_sha256')!=sha(Path(root)/'data_manifest.json') or not freeze.get('candidates'):
            raise ValueError('Invalid holdout freeze')
    import pandas as pd
    from reft.text import target_text
    spec=read(Path(root)/'data_manifest.json'); validate(spec['roles'],cfg['data']['roles'])
    if sha(cfg['original_manifest'])!=spec['source_manifest_sha256']: raise ValueError('Source data changed')
    df=pd.read_parquet(cfg['original_manifest']).set_index('utterance_id'); result=[]
    for ident in spec['roles'][role]:
        r=df.loc[ident['utterance_id']].to_dict()|ident
        if sha(r['audio_path'])!=r['audio_full_sha256']: raise ValueError('Changed audio')
        if hashlib.sha256(target_text(r['transcript_raw']).encode()).hexdigest()!=r['target_text_sha256']: raise ValueError('Changed transcript')
        result.append(r)
    return result
