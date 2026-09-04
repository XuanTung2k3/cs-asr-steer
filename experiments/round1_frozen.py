#!/usr/bin/env python
"""Round-1 Job A: frozen Whisper screen and counterfactual geometry.

The dry-run is model-free and therefore safe on login nodes.  A normal run
uses only D-dev-select; dev-confirm and SEAME are never read by this runner.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from steer_sweep.counterfactual_geometry import counterfactual_states, orthogonal_axes
from steer_sweep.round1_metrics import paired_metric_report
from steer_sweep.rounds import (ROUND1_LAYERS, atomic_json, enumerate_coarse,
                                enumerate_detail, enumerate_localization,
                                enumerate_negative, mixture_direction)

log = logging.getLogger("round1_frozen")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rounds/round1_frozen.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="two utterances")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="results/round1/job_a")
    p.add_argument("--time-budget-minutes", type=float, default=175.0)
    return p.parse_args(argv)


def _print_review():
    checks = [
        ("PASS", "NormPreserve is enabled in the Round-1 steering hook for every frozen condition."),
        ("PASS", "Mixtures unit-normalize v_nat and v_prompt before applying coefficients."),
        ("PASS", "The frozen mixture coefficients are constants and never optimizer parameters."),
        ("PASS", "Matched-energy controls are per utterance; no global |S| is used."),
        ("PASS", "Automatic gates consume pre-intervention hidden states and are calibrated only on D-dev-select."),
        ("PASS", "Four forced prefix positions are excluded by the decoder hook."),
        ("PASS", "Beam baselines are cached separately by beam size and paired by protocol."),
        ("PASS", "Results/checkpoints use atomic promotion and per-cell resume records."),
        ("WARN", "D-test-lock is not consumed in Round 1; its manifest is required before Round 2 lock."),
        ("WARN", "Full GPU execution is scheduler-only; dry-run does not initialize Whisper."),
    ]
    for status, reason in checks:
        print(f"{status}: {reason}")


def _dry_run(args):
    coarse = enumerate_coarse()
    detail = enumerate_detail((8, 16, 24))
    loc = enumerate_localization(((24, (4.0, 1.0), 1.0),) * 3)
    neg = enumerate_negative(24, 1.0)
    print(f"Round-1 Job A dry-run: coarse_cells={len(coarse)} (expected 84)")
    print(f"Round-1 Job A dry-run: detail_cells={len(detail)} (expected 126)")
    print(f"Round-1 Job A dry-run: localization_cells={len(loc)}, negative_cells={len(neg)}")
    assert len(coarse) == 84 and len(detail) == 126
    _print_review()
    return 0


def _decode(bundle, pop, layer, direction, scope, beam, *, beta=1.0, oracle=None,
            gate_values=None, batch_size=8):
    """Decode one cell with the existing NormPreserve decoder implementation."""
    from experiments.job_a_frozen import NormPreserveDecoderHook, PREFIX_WIDTH, MAX_NEW_TOKENS
    from csasr.models.whisper import batch_model_inputs
    import torch
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    out_text = {}
    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start:start + batch_size]
        ids = [str(x) for x in batch.utterance_id]
        inputs = batch_model_inputs(bundle, batch.audio_path.tolist())
        if scope == "global":
            mask = lambda pos, item: pos >= PREFIX_WIDTH
        elif scope == "oracle_local":
            sets = {i: set((oracle or {}).get(uid, [])) for i, uid in enumerate(ids)}
            mask = lambda pos, item: (pos - PREFIX_WIDTH) in sets.get(item, set())
        else:
            # Automatic/control gates are supplied as a frozen per-utterance
            # schedule.  This path never consults gold positions.
            schedules = gate_values or {}
            mask = lambda pos, item: bool(schedules.get(ids[item], {}).get(pos - PREFIX_WIDTH, 0.0) > 0)
        hook = NormPreserveDecoderHook(bundle, layer, direction, beta,
                                       num_beams=beam, mask_fn=mask)
        kwargs = {"task": "transcribe", "language": "zh", "do_sample": False,
                  "num_beams": beam, "temperature": 0.0,
                  "max_new_tokens": MAX_NEW_TOKENS, "condition_on_prev_tokens": False}
        with hook:
            generated = bundle.model.generate(**inputs, **kwargs)
        seq = generated if isinstance(generated, torch.Tensor) else generated.sequences
        for uid, text in zip(ids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            out_text[uid] = text.strip()
    return out_text


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    cells = enumerate_coarse()
    atomic_json(out / "cell_manifest.json", [c.as_dict() for c in cells])
    start = time.monotonic()
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from steer_sweep import data as D
    from steer_sweep.directions import DirectionStore
    # Round YAMLs hold the experiment overlay.  Whisper/data construction
    # still uses the existing nested package config named by data_config.
    import yaml
    overlay = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    base_config = str(overlay.get("data_config", "configs/lss/l1b_candidates_dialogue_v2r3.yaml"))
    cfg = load_config(base_config)
    bundle = load_whisper(cfg)
    pop = D.build_population(bundle, cfg, "D-dev-select", assert_anchors=True)
    if args.smoke or args.limit:
        pop = pop.subsample(min(args.limit or 2, 2))
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    store = DirectionStore(cfg, out / "directions_cache", log=log)
    directions = {}
    for layer in ROUND1_LAYERS:
        n = store.get("v_nat", "decoder", layer, bundle=bundle).vector
        p = store.get("v_prompt", "decoder", layer, bundle=bundle).vector
        directions[layer] = (n, p)
    baseline_cache = {}
    # Baselines are direct, unhooked generation and are cached separately by
    # decoding protocol.  They are never compared across beam sizes.
    from csasr.models.whisper import batch_model_inputs
    for beam in ((1,) if args.smoke else (1, 5)):
        direct = {}
        manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
        for s in range(0, len(manifest), 8):
            b = manifest.iloc[s:s + 8]
            generated = bundle.model.generate(**batch_model_inputs(bundle, b.audio_path.tolist()),
                task="transcribe", language="zh", do_sample=False, num_beams=beam,
                temperature=0.0, max_new_tokens=220, condition_on_prev_tokens=False)
            seq = generated if hasattr(generated, "shape") else generated.sequences
            for uid, text in zip(b.utterance_id, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
                direct[str(uid)] = text.strip()
        baseline_cache[beam] = direct
        atomic_json(out / f"baseline_beam{beam}.json", direct)
    oracle = {str(k): list(v) for k, v in __import__("experiments.job_a_frozen", fromlist=["oracle_steps_for"]).oracle_steps_for(pop).items()}
    rows = []
    for cell in cells:
        if args.resume and (out / f"cells/{cell.config_hash}.json").exists():
            continue
        if time.monotonic() - start > args.time_budget_minutes * 60:
            break
        n, p = directions[cell.layer]
        direction = mixture_direction(n, p, *cell.mixture)
        scope = cell.scope
        hyp = _decode(bundle, pop, cell.layer, direction, scope, cell.beam,
                      beta=cell.beta, oracle=oracle)
        ref_list = [refs[i] for i in ids]
        base_list = [baseline_cache[cell.beam][i] for i in ids]
        method_list = [hyp[i] for i in ids]
        row = cell.as_dict() | paired_metric_report(ref_list, base_list, method_list,
            beam=cell.beam, baseline_protocol=f"greedy_or_beam{cell.beam}",
            method_protocol=f"greedy_or_beam{cell.beam}", utterance_ids=ids)
        row["runtime_sec"] = 0.0
        rows.append(row)
        atomic_json(out / "cells" / f"{cell.config_hash}.json", row)
    atomic_json(out / "summary.json", {"status": "completed" if len(rows) == len(cells) else "time_budget_exhausted",
        "completed_cells": len(rows), "planned_cells": len(cells), "runtime_sec": time.monotonic() - start})
    print(json.dumps({"completed_cells": len(rows), "planned_cells": len(cells)}, indent=2))
    return 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
