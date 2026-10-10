#!/usr/bin/env python
"""CPU-only ST-PROMPT-R1 engineering review of sealed unedited paired states.

No model loading, reference targets, edited logits, GPU or scientific decisions.
L3/L8 are explicitly pending passive paired capture; a mean prompt vector is not a substitute.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from csasr.inference_cf.core import digest, file_hash
from csasr.inference_cf.core_p1 import direction
from experiments.inference_cf_p2r import solve_scale

CONFIG = 'configs/inference_cf/st_prompt_r1.json'


def review() -> dict:
    cfg = json.loads(Path(CONFIG).read_text())
    panel = json.loads(Path(cfg['panel']).read_text())
    assert digest({k: v for k, v in panel.items() if k != 'identity_hash'}) == cfg['panel_identity_hash']
    assert all(file_hash(p) == h for p, h in cfg['source_sha256'].items())
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    source = 'sha256:' + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest()
    assert source == cfg['site_forward']['forward_source_sha256']
    summary = {}
    inputs = {}
    start = time.perf_counter()
    for layer in (16, 24):
        path = f'results/inference_cf/st_loc0/run1/calibration/states_L{layer}_CROSS.npz'
        inputs[path] = file_hash(path)
        with np.load(path, allow_pickle=False) as z:
            hb, he = z['H_B'], z['H_E']
        assert hb.shape == he.shape == (180, 1280)
        for arm in [a for a in cfg['primary_arms'] if a['layer'] == layer]:
            errors, evaluations, failures, angles = [], [], [], []
            for i, (b, e) in enumerate(zip(hb, he)):
                r = torch.from_numpy(b).to(torch.bfloat16)
                assert torch.equal(r.float(), torch.from_numpy(b))  # lossless sealed native state
                d = direction(torch.from_numpy(e), r.float())
                assert d['d'] is not None
                target = arm['eta'] * float(r.double().norm())
                sol = solve_scale(r, arm['sign'] * d['d'], target, max_eval=8)
                angles.append(sol['phi'])
                evaluations.append(sol['evals'])
                err = sol.get('rel_sq_err')
                if err is not None:
                    errors.append(err)
                if sol['status'] != 'ok' or err is None or err > .02:
                    failures.append(i)
            summary[arm['id']] = dict(cases=180, failure_indices=failures,
                max_relative_squared_error=max(errors), max_solver_evaluations=max(evaluations),
                phi_min=min(angles), phi_max=max(angles))
    return dict(schema='st_prompt_r1_cpu_geometry_v1', config_sha256=file_hash(CONFIG),
        panel_identity_hash=panel['identity_hash'], inputs=inputs, forward_source_sha256=source,
        device='cpu', dtype='native bf16 repair, float64 geometry', reviewed_layers=[16, 24],
        pending_layers=[3, 8], pending_reason='sealed R0 archives retain only mean prompt vectors, not query-paired E states',
        cases=sum(x['cases'] for x in summary.values()), arms=summary,
        complete_four_layer_preflight=False, new_model_forwards=0, new_edited_logits=0,
        evaluator_access=False, GPU_calls=0, elapsed_seconds=time.perf_counter()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit('refuse overwrite of existing geometry evidence')
    result = review()
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ('cases','reviewed_layers','pending_layers','complete_four_layer_preflight','elapsed_seconds')}))


if __name__ == '__main__':
    main()
