"""RC7-0C continuation of the failed core7h-20261020-v2 N4 stage (diagnostics only).

The archived run root is read-only input. A NEW continuation root receives copies of the frozen plan,
data manifest, sealed B0 directory and N1/N2 records, then the UNCHANGED `reft.a5000_campaign.worker.n4`
runs there with the repaired NumPy boundary. Under the frozen N2_NO_GO branch N4 trains nothing: it
teacher-forces T2000[:64] and V300[:64] through frozen Whisper and the selected LoReFT CE adapter (D1
lexical decomposition, D2 controlled readout). C500 is never loaded (`rows()` keeps it sealed).

Engineering-only plan overrides (recorded in continuation.json): created_epoch=now, campaign_root=new root,
resources capped to the RC7-0 diagnostic ceiling. source_commit/data_sha256 keep the archived binding
because they key the reused encoder cache; the executing source commit is recorded separately.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ARCHIVED_FILES = ['frozen_plan.json', 'data_manifest.json', 'status.json', 'environment.json', 'worker_usage.json',
                  'worker_heartbeat.json', 'N1_selection.json', 'N2_selection.json', 'N2_decision.json',
                  'B0/identity.json', 'B0/SEALED.json', 'B0/metrics.json', 'jobs/N4/result.json', 'logs/N4.log']
COPIED = ['data_manifest.json', 'N1_selection.json', 'N2_selection.json', 'N2_decision.json']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str)); os.replace(tmp, p)


def fingerprint(source):
    """Hash + mtime of every archived file the legacy runtime could touch, plus absence of N4_analysis.json."""
    source = Path(source); out = {}
    for rel in ARCHIVED_FILES:
        p = source / rel
        out[rel] = [sha(p), p.stat().st_mtime_ns] if p.exists() else None
    out['N4_analysis.json'] = (source / 'N4_analysis.json').exists()
    out['B0/predictions_count'] = len(list((source / 'B0' / 'predictions').glob('*.json')))
    return out


def continuation_plan(plan, root, gpu_hours, wall_hours, now):
    if gpu_hours <= 0 or gpu_hours > 2:
        raise ValueError('RC7-0 diagnostic ceiling is 2 allocated GPU-hours')
    new = copy.deepcopy(plan); c = new['config']
    overrides = {'created_epoch': [plan['created_epoch'], now],
                 'config.campaign_root': [c.get('campaign_root'), str(root)],
                 'config.resources.gpu_hours': [c['resources']['gpu_hours'], gpu_hours],
                 'config.resources.reserve_gpu_hours': [c['resources']['reserve_gpu_hours'], 0],
                 'config.resources.wall_hours': [c['resources']['wall_hours'], wall_hours]}
    new['created_epoch'] = now; c['campaign_root'] = str(root)
    c['resources'].update(gpu_hours=gpu_hours, reserve_gpu_hours=0, wall_hours=wall_hours)
    return new, overrides


def prepare(source, root, gpu_hours=1.0, wall_hours=2.0):
    source, root = Path(source).resolve(), Path(root).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Continuation root must be separate from the archived run root')
    if root.exists():
        raise FileExistsError('Continuation root already exists; continuation IDs are immutable')
    plan = json.loads((source / 'frozen_plan.json').read_text())
    if json.loads((source / 'N2_decision.json').read_text())['decision'] != 'N2_NO_GO':
        raise ValueError('This continuation is defined only for the archived N2_NO_GO branch (D1/D2, no training)')
    if (source / 'confirmation_freeze.json').exists() or (source / 'confirmation').exists():
        raise ValueError('Unexpected confirmation package in archived root')
    root.mkdir(parents=True)
    for rel in COPIED:
        shutil.copy2(source / rel, root / rel)
    shutil.copytree(source / 'B0', root / 'B0')
    new, overrides = continuation_plan(plan, root, gpu_hours, wall_hours, time.time())
    write(root / 'frozen_plan.json', new)
    copies = {rel: sha(root / rel) for rel in COPIED}
    if any(copies[rel] != sha(source / rel) for rel in COPIED) or sha(root / 'data_manifest.json') != plan['data_sha256']:
        raise ValueError('Copied input hash mismatch')
    record = {'schema': 'rc7_0_n4_continuation_v1', 'archived_root': str(source), 'archived_campaign': plan['campaign_id'],
              'archived_source_commit': plan['source_commit'], 'archived_plan_sha256': sha(source / 'frozen_plan.json'),
              'archived_config_sha256': plan['config_sha256'], 'data_sha256': plan['data_sha256'],
              'overrides': overrides, 'copied_inputs_sha256': copies,
              'B0_sealed_sha256': sha(root / 'B0' / 'SEALED.json'), 'archived_fingerprint_before': fingerprint(source),
              'scope': 'N4 D1/D2 diagnostics only; no training; no C500; no official test'}
    write(root / 'continuation.json', record)
    return record


def run(root, gpu=0):
    root = Path(root).resolve(); rec = json.loads((root / 'continuation.json').read_text())
    repo = Path(__file__).resolve().parents[2]
    rec['executing_source_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                                                    text=True, check=True).stdout.strip()
    rec['executing_source_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo,
                                                        capture_output=True, text=True, check=True).stdout.strip())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    lock = open(f'/home/ubuntu/tung.nx/a5000-campaign-gpu{gpu}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from reft.a5000_campaign.worker import Runtime, n4
    from reft.provenance import weights_hash
    start = time.monotonic(); rec['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write(root / 'continuation.json', rec)
    R = Runtime(root)
    try:
        report = n4(R)
        rec['status'] = 'COMPLETE'
    except Exception as e:  # recorded, never hidden
        rec['status'] = 'FAILED'; rec['error'] = f'{type(e).__name__}: {e}'; report = None
        raise
    finally:
        rec['backbone_unchanged'] = weights_hash(R.b.model) == R.initial_hash
        rec['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        rec['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        rec['resident_seconds'] = time.monotonic() - start
        rec['allocated_gpu_hours'] = rec['resident_seconds'] / 3600
        rec['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        before = rec['archived_fingerprint_before']; after = fingerprint(rec['archived_root'])
        rec['archived_fingerprint_after'] = after; rec['archived_root_unchanged'] = before == after
        if (root / 'N4_analysis.json').exists():
            rec['N4_analysis_sha256'] = sha(root / 'N4_analysis.json')
        write(root / 'continuation.json', rec)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--source', default='/home/ubuntu/tung.nx/cs-asr-reft/a5000_experiments/energy_v2/core7h-20261020-v2')
    p.add_argument('--root', required=True)
    p.add_argument('--gpu-hours', type=float, default=1.0)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    out = prepare(a.source, a.root, a.gpu_hours) if a.command == 'prepare' else run(a.root, a.gpu)
    print(json.dumps({'status': 'ok', 'root': a.root} if a.command == 'run' else out, default=str)[:4000])
