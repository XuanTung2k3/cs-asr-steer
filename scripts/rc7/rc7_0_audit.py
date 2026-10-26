"""RC7-0 read-only evidence and holdout-exposure audit.

Subcommands print JSON with counts, relative paths and hashes only. They never print utterance IDs,
transcripts or predictions, never decode or score, and never write inside a run root.

  archives  -- revalidate core7h-20261020-v2 / lra-gate-20261021-v2 output seals and checkpoint hashes
  exposure  -- role overlap and search of experiment roots for any holdout-named or holdout-mentioning file
  history   -- overlap of steering-era tracked artifacts (results/, out/, docs/) with A5000 roles, at
               utterance and inferred-dialogue level

Run on A5000 with the campaign interpreter, e.g.
  CUDA_VISIBLE_DEVICES= .../reft-a5000/bin/python -I scripts/rc7/rc7_0_audit.py archives
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import re
from pathlib import Path

BASE = Path('/home/ubuntu/tung.nx/cs-asr-reft/a5000_experiments')
CORE = BASE / 'energy_v2/core7h-20261020-v2'
LRA = BASE / 'lra_gate_sprint_v1/lra-gate-20261021-v2'
REPO = Path(__file__).resolve().parents[2]
ROLES_V1 = REPO / 'configs/reft/a5000_manifests/roles_v1.json'
SCAN_ROOTS = [BASE, Path('/home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1'),
              Path('/home/ubuntu/shared/tung.nx/a5000-validation'), Path('/home/ubuntu/tung.nx/a5000-bootstrap')]
MANIFEST_NAMES = {'data_manifest.json', 'roles_v1.json', 'role_identities.json', 'manifest.json', 'roles.json'}
UTT = re.compile(r'(?:ZH-CN|EN)_U\d{4}_S\d_\d+')
_SHA = {}


def sha(p):
    p = Path(p)
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def roles():
    m = json.load(open(CORE / 'data_manifest.json'))
    r = dict(m['roles'])
    r.update(json.load(open(ROLES_V1))['roles'])
    return r


def archives():
    out = {}
    st = json.load(open(CORE / 'status.json'))
    ok = bad = missing = 0
    problems = []
    for j in st['jobs'].values():
        for rel, digest in (j.get('outputs') or {}).items():
            p = CORE / rel
            if not p.exists():
                missing += 1; problems.append(rel)
            elif sha(p) == digest:
                ok += 1
            else:
                bad += 1; problems.append(rel)
    bind = lambda key: sorted({v.get('input_hashes', {}).get(key) for v in st['jobs'].values()} - {None})
    out['core'] = {'job_status_counts': dict(collections.Counter(j['status'] for j in st['jobs'].values())),
                   'gpu_seconds_recorded': st.get('gpu_seconds'), 'output_hashes_ok': ok, 'mismatch': bad,
                   'missing': missing, 'problem_paths': problems,
                   'non_complete_jobs': {k: {'status': v['status'], 'exit_code': v.get('exit_code')}
                                         for k, v in st['jobs'].items() if v['status'] != 'COMPLETE'},
                   'source_bindings': bind('source'), 'config_bindings': bind('config'), 'data_bindings': bind('data'),
                   'confirmation_present': (CORE / 'confirmation').exists() or (CORE / 'confirmation_freeze.json').exists(),
                   'N4_analysis_present': (CORE / 'N4_analysis.json').exists()}
    rows = list(csv.DictReader(open(CORE / 'reports/development_metrics.csv')))
    out['core_selected_checkpoints'] = {
        f"{r['stage']}:{r['method']}:{r['objective']}": {
            'sha256': r['checkpoint_sha256'], 'relative_path': str(Path(r['checkpoint']).relative_to(CORE)),
            'sha_ok': Path(r['checkpoint']).exists() and sha(r['checkpoint']) == r['checkpoint_sha256'],
            'bytes': Path(r['checkpoint']).stat().st_size if Path(r['checkpoint']).exists() else None} for r in rows}
    files = ['data_manifest.json', 'frozen_plan.json', 'environment.json', 'status.json', 'N1_selection.json',
             'N2_selection.json', 'N2_panel.json', 'N2_scalar.json', 'N2_decision.json', 'N3_selection.json',
             'CORE_COMPLETE.json', 'resource_usage.csv', 'worker_usage.json', 'admitted.json', 'launch_receipt.json',
             'reports/development_metrics.csv', 'reports/development_table.tex', 'reports/SCIENTIFIC_FINDINGS.md',
             'reports/MANUSCRIPT_RESULTS.md', 'reports/CAMPAIGN_SUMMARY.md', 'reports/pareto.json', 'logs/N4.log',
             'jobs/N4/job.json', 'jobs/N4/result.json', 'cpu-tests.xml']
    out['core_file_sha256'] = {f: sha(CORE / f) if (CORE / f).exists() else None for f in files}
    out['core_N4_result'] = json.load(open(CORE / 'jobs/N4/result.json'))
    plan = json.load(open(CORE / 'frozen_plan.json'))
    out['core_plan'] = {k: plan[k] for k in ['campaign_id', 'config_sha256', 'data_sha256', 'job_graph_sha256', 'source_commit']}
    found = collections.defaultdict(list)
    for p in LRA.rglob('*.pt'):
        found[sha(p)].append(str(p.relative_to(LRA)))
    rows2 = list(csv.DictReader(open(LRA / 'reports/combined.csv')))
    out['lra_checkpoints'] = {r['system']: {'sha256': r['checkpoint_sha256'],
                                            'present_in_sprint_root': r['checkpoint_sha256'] in found,
                                            'relative_path': (found.get(r['checkpoint_sha256']) or [None])[0]}
                              for r in rows2 if r['checkpoint_sha256']}
    out['lra_file_sha256'] = {f: sha(LRA / f) if (LRA / f).exists() else None for f in
                              ['frozen_plan.json', 'data_manifest.json', 'admitted.json', 'status.json',
                               'reports/decision.json', 'reports/combined.csv', 'reports/accounting.json',
                               'reports/FINAL_REPORT.md', 'reports/pareto.json', 'parent_cost_audit.json',
                               'run_clock.json', 'cpu-tests.xml', 'launch.json']}
    lp = json.load(open(LRA / 'frozen_plan.json'))
    out['lra_plan'] = {k: lp[k] for k in ['schema', 'config_sha256', 'data_sha256', 'source_commit']}
    import torch
    for r in rows:
        if r['stage'] == 'N1' and r['method'] in ('R', 'L'):
            s = torch.load(r['checkpoint'], map_location='cpu', weights_only=True)
            out[f"checkpoint_state_keys_{r['method']}_CE"] = {k: [list(v.shape), str(v.dtype), v.numel()] for k, v in s.items()}
    return out


def exposure():
    r = roles()
    ids = {k: {x['utterance_id'] for x in v} for k, v in r.items()}
    audio = {k: {x['audio_full_sha256'] for x in v} for k, v in r.items()}
    names = sorted(r)
    out = {'role_counts': {k: len(v) for k, v in ids.items()},
           'C150_subset_of_C500': ids['A5000_C150'] <= ids['A5000_C500'],
           'C500_membership_sha256_of_sorted_ids': hashlib.sha256(json.dumps(sorted(ids['A5000_C500'])).encode()).hexdigest(),
           'official_splits': {k: sorted({x.get('official_split') for x in v}) for k, v in r.items()},
           'pairwise_overlap': {f'{a}&{b}': {'ids': len(ids[a] & ids[b]), 'audio': len(audio[a] & audio[b])}
                                for i, a in enumerate(names) for b in names[i + 1:]}}
    hold = ids['A5000_C500'] | ids['A5000_C150']
    hold_audio = audio['A5000_C500'] | audio['A5000_C150']
    named, mentions, scanned = collections.Counter(), {}, 0
    for root in SCAN_ROOTS:
        if not root.exists():
            out.setdefault('missing_roots', []).append(str(root)); continue
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.split('.')[0] in hold:
                    named[dp] += 1
                if f in MANIFEST_NAMES or not f.endswith(('.json', '.jsonl', '.csv', '.log', '.md', '.txt')):
                    continue
                p = Path(dp) / f
                try:
                    if p.stat().st_size > 8_000_000:
                        continue
                    s = p.read_text(errors='ignore')
                except OSError:
                    continue
                scanned += 1
                n = len(set(UTT.findall(s)) & hold) + sum(1 for a in hold_audio if a in s)
                if n:
                    mentions[str(p)] = n
    out.update({'scan_roots': [str(x) for x in SCAN_ROOTS], 'text_files_scanned': scanned,
                'holdout_named_files_by_dir': dict(named), 'holdout_mentions_by_file': mentions})
    return out


def history():
    r = roles()
    m = json.load(open(CORE / 'data_manifest.json'))['roles']
    where = {x['utterance_id']: (k, x) for k, v in m.items() for x in v}
    c150 = {x['utterance_id'] for x in r['A5000_C150']}
    conv2dlg = {x['conversation_id']: x['dialogue_id'] for v in m.values() for x in v}
    dlg2role = {x['dialogue_id']: k for k, v in m.items() for x in v}
    # Steering-era populations as documented in docs/current/DATA_EXPOSURE.md.
    sources = {'job_a_D-dev-confirm_subset_decoded': sorted(glob.glob(str(REPO / 'results/job_a/F*_hypotheses.jsonl'))),
               'round1_D-dev-select_subset_decoded': [str(REPO / 'results/round1/job_a/baseline_beam5.json')]
               + sorted(glob.glob(str(REPO / 'out/_determinism_probe_*.json'))),
               'job_b_loc_util_train_steering_training_list': [str(REPO / 'results/job_b/train_utterance_ids.json')],
               'docs_mentions': sorted(glob.glob(str(REPO / 'docs/*.md')))}
    out, per_dlg = {}, collections.defaultdict(set)
    union = set()
    for name, files in sources.items():
        u = set()
        for f in files:
            if os.path.exists(f):
                u |= set(UTT.findall(open(f, errors='ignore').read()))
        c5 = {x for x in u if x in where and where[x][0] == 'A5000_C500'}
        if name != 'docs_mentions':
            union |= c5
        for x in u:
            d = conv2dlg.get('_'.join(x.split('_')[:2]))
            if d and name != 'docs_mentions':
                per_dlg[d].add(name)
        out[name] = {'files': [os.path.relpath(f, REPO) for f in files if os.path.exists(f)], 'distinct_ids': len(u),
                     'by_A5000_role': dict(collections.Counter(where[x][0] for x in u if x in where)),
                     'C500_utterances': len(c5), 'C150_utterances': len(c5 & c150),
                     'C500_dialogues': len({where[x][1]['dialogue_id'] for x in c5}),
                     'C500_strata': dict(collections.Counter(where[x][1]['stratum'] for x in c5))}
    out['C500_union'] = {'utterances': len(union), 'fraction': len(union) / 500,
                         'dialogues': len({where[x][1]['dialogue_id'] for x in union}), 'C150_utterances': len(union & c150)}
    table = collections.defaultdict(collections.Counter)
    for d, s in per_dlg.items():
        table[dlg2role[d]]['+'.join(sorted(s))] += 1
    out['A5000_dialogues_by_inferred_steering_source'] = {k: dict(v) for k, v in table.items()}
    out['A5000_dialogues_without_any_historical_utterance'] = {
        k: sum(1 for d, rr in dlg2role.items() if rr == k and d not in per_dlg) for k in m}
    out['dialogues_with_multiple_steering_sources'] = sum(len(s) > 1 for s in per_dlg.values())
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['archives', 'exposure', 'history'])
    a = p.parse_args()
    print(json.dumps({'archives': archives, 'exposure': exposure, 'history': history}[a.command](), indent=1, default=str))
