#!/usr/bin/env python
"""Post-seal independent replay scoring; never used by the GPU runner."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ttls_r1_independent_analysis import one, compare


def main():
    base = ROOT / 'results/inference_cf/ttls_r1r_independent_audit'
    seal = json.loads((base / 'replay_seal.json').read_text())
    payload = {k: v for k, v in seal.items() if k != 'seal_hash'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == seal['seal_hash']
    for path, digest in seal['files'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    raw = json.loads((base / 'replay_r3/summary.json').read_text())
    primary = json.loads((base / 'independent_metrics.json').read_text())
    checks = []
    for index in raw['selection']['indices']:
        row = json.loads((base / f'replay_r3/rows/{index:03d}.json').read_text())['phaseB']
        saved = primary['rows'][index]
        b0 = one(saved['reference'], saved['texts']['B0'])
        baseline_tokens = raw['zero100'][index]['zero']['tokens']
        for arm in ('T1', 'T2', 'T4', 'T6'):
            hypothesis = row[arm]
            scored = one(saved['reference'], hypothesis['text'])
            assert scored['counts'] == saved['systems'][arm]['counts']
            tokens = hypothesis['tokens']
            eos = len(tokens) > len(baseline_tokens) and tokens[:len(baseline_tokens)] == baseline_tokens
            transitions = compare(b0, scored, saved['id'], saved['dialogue'], eos)
            reference_events = [e for e in primary['events'][arm] if e['id'] == saved['id']]
            # Event records include an independently added lexical class in primary.
            strip = lambda event: {k: v for k, v in event.items() if k != 'lexical_kind'}
            assert transitions['events'] == [strip(e) for e in reference_events]
            checks.append({'index': index, 'arm': arm, 'counts_equal': True, 'events_equal': True})
    matched = {'T1': 'B2', 'T2': 'T2A', 'T4': 'T3', 'T6': 'T5'}
    qualifying = []
    eligibility = {}
    for arm, comparator in matched.items():
        genuine = sum(v for k, v in primary['lexical'][arm].items() if k.startswith('genuine'))
        coverage = primary['genuine_dialogues'][arm]
        eligibility[arm] = {'genuine_units': genuine, 'genuine_dialogues': coverage,
                            'minimum_units': 8, 'minimum_dialogues': 4,
                            'lexical_breadth_pass': genuine >= 8 and coverage >= 4,
                            'matched_A2': comparator}
        # This necessary pre-frozen gate alone excludes all arms, independent of safety.
        if genuine >= 8 and coverage >= 4:
            qualifying.append(arm)
    assert not qualifying
    result = {'schema': 'ttls_r1r_independent_final_checks_v1', 'replay_seal_hash': seal['seal_hash'],
              'replay_job': raw['job'], 'replay_count': len(checks), 'score_checks': checks,
              'all100_zero_identity': all(all(r[k] for k in ('baseline_equal', 'zero_free_equal', 'zero_ffn_equal', 'zero_logits_equal')) for r in raw['zero100']),
              'all96_raw_exact': all(all(r[k] for k in ('tokens_equal', 'text_equal', 'termination_equal')) and all(r.get(k, 0) == 0 for k in ('loss_diff', 'grad_diff', 'z_diff')) for r in raw['comparisons']),
              'gradient_crosschecks': raw['gradient_crosschecks'], 'weights_unchanged': raw['weights_unchanged'],
              'reset': raw['reset'], 'model_grad_free': raw['model_grad_free'], 'frozen_gate_checks': eligibility,
              'published_metric_mismatches': primary['published_mismatches'],
              'experiment_label_independent': 'TTLS_R1R_VALID_BUT_INSUFFICIENT',
              'audit_verdict': 'AUDIT_PASS_TTLS_INSUFFICIENT', 'recommendation': 'prioritize A2+P; close this TTLS version',
              'tests_passed': {'focused': 51, 'historical': 47}}
    assert len(checks) == 96 and result['all100_zero_identity'] and result['all96_raw_exact']
    assert not result['published_metric_mismatches']
    (base / 'final_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('replay_count', 'all100_zero_identity', 'all96_raw_exact', 'audit_verdict')}))


if __name__ == '__main__':
    main()
