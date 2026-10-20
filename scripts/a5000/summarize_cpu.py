"""Consolidate actual JUnit results, preserving raw-run counts and repaired/blocked identities."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[2]
C = yaml.safe_load((ROOT/'configs/a5000/runtime.yaml').read_text())
out = Path(C['output_root']).expanduser()/'cpu'
names = ['cpu-final.xml', 'cpu-uroman.xml', 'cpu-repaired.xml', 'tokenizer.xml']
merged, raw = {}, {}
for name in names:
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0}
    for case in ET.parse(out/name).getroot().iter('testcase'):
        state = 'passed'
        if case.find('failure') is not None:
            state = 'failed'
        elif case.find('error') is not None:
            state = 'error'
        elif case.find('skipped') is not None:
            state = 'skipped'
        counts[state] += 1
        merged[(case.get('classname'), case.get('name'))] = {'state': state, 'evidence': name,
            'reason': case.find('skipped').get('message') if state == 'skipped' else None}
    raw[name] = counts
counts = {s: sum(v['state'] == s for v in merged.values()) for s in ['passed', 'failed', 'skipped', 'error']}
payload = {'method': 'Union of actual full-suite and focused regression JUnit results; later focused results replace '
                     'the same testcase. Not a claim that the raw full-suite run had zero failures.',
           'unique_tests': len(merged), 'resolved_counts': counts, 'raw_runs': raw,
           'unresolved': [{'classname': k[0], 'test': k[1], **v} for k, v in merged.items() if v['state'] != 'passed']}
(out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload, indent=2))
assert counts['failed'] == 0 and counts['error'] == 0
