"""Read-only metadata admission. Never reads audio or opens an unverified role manifest.

Role documents must have role, provenance, and rows. Each row needs utterance_id, dialogue_id,
speaker_id, audio_sha256 and partition. Identity pins must come from recovered historical evidence,
not from a newly sampled split. Full500 deliberately contains Train394 and Inner106.
"""
import hashlib
import json
from pathlib import Path

ROLES={'Train394':394,'Inner106':106,'Full500':500,'Dev150':150}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_roles(spec):
    docs={};problems=[]
    for role,count in ROLES.items():
        entry=spec.get(role,{})
        if not entry.get('path') or not entry.get('sha256'):
            problems.append(f'{role}: missing historical manifest path/SHA256');continue
        p=Path(entry['path']).expanduser()
        if not p.is_file():problems.append(f'{role}: manifest unavailable: {p}');continue
        if digest(p)!=entry['sha256']:
            problems.append(f'{role}: manifest SHA256 mismatch');continue
        doc=json.loads(p.read_text())
        if doc.get('role')!=role or not doc.get('provenance'):
            problems.append(f'{role}: historical role/provenance binding absent');continue
        rows=doc.get('rows',[])
        if len(rows)!=count:problems.append(f'{role}: expected {count} rows, found {len(rows)}');continue
        fields={'utterance_id','dialogue_id','speaker_id','audio_sha256','partition'}
        if any(not fields.issubset(r) or any(not r[k] for k in fields) for r in rows):
            problems.append(f'{role}: missing identity metadata');continue
        if any(r['partition']!='train' for r in rows):
            problems.append(f'{role}: only original TRAIN rows admitted without additional exposure evidence');continue
        if len({r['utterance_id'] for r in rows})!=count or len({r['audio_sha256'] for r in rows})!=count:
            problems.append(f'{role}: duplicate utterance/audio identity');continue
        docs[role]=rows
    if len(docs)==4:
        def ids(role,key):return {r[key] for r in docs[role]}
        for key in ['utterance_id','dialogue_id','speaker_id','audio_sha256']:
            if ids('Train394',key)&ids('Inner106',key):problems.append(f'Train394/Inner106 overlap: {key}')
            if ids('Full500',key)&ids('Dev150',key):problems.append(f'Full500/Dev150 overlap: {key}')
        def mapping(role):return {r['utterance_id']:r for r in docs[role]}
        if mapping('Full500')!={**mapping('Train394'),**mapping('Inner106')}:
            problems.append('Full500 does not exactly equal Train394 union Inner106 with matching metadata')
    return problems


def admit(config):
    problems=validate_roles(config.get('required_roles',{}))
    if config.get('objective')!='standard_ASR_CE_only':problems.append('Objective is not CE-only')
    if config.get('locked_test_access') is not False:problems.append('Locked test firewall missing')
    if not config.get('resource_admitted'):problems.append('Throughput/encoder-LoRA feasibility and mandatory-matrix resource coverage not admitted')
    if not config.get('scientific_training_implemented'):problems.append('Full R0–R6 scientific executor not implemented')
    return problems
