#!/usr/bin/env python
"""Round-1 Job B entry point with an auditable fixed training protocol."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from steer_sweep.rounds import ROUND1_LAYERS, atomic_json


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rounds/round1_training.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="results/round1/job_b")
    p.add_argument("--time-budget-minutes", type=float, default=175.0)
    return p.parse_args(argv)


def dry_run(args):
    settings = {
        "epochs": 3, "micro_batch": 8, "gradient_accumulation": 2,
        "effective_batch": 16, "optimizer": "AdamW", "weight_decay": 0.0,
        "betas": [0.9, 0.99], "eps": 1e-6, "gradient_clip": 1.0,
        "seed": args.seed, "train_utterances": 500, "prefix_positions_ignored": 4,
        "selection": "D-dev-select greedy MER", "beam5": "best checkpoint only",
        "screen_layers": list(ROUND1_LAYERS), "bf16": True,
    }
    print("Round-1 Job B dry-run: fixed settings")
    print(json.dumps(settings, indent=2))
    print("dry_run_cell_count=7 T1 + 7 GlobalDecoderMatched + SALSA-E + LoRA(priority)")
    return 0


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "training_protocol.json", {
        "status": "running", "seed": args.seed, "layers": list(ROUND1_LAYERS),
        "epochs": 3, "micro_batch": 8, "gradient_accumulation": 2,
        "effective_batch": 16, "optimizer": "AdamW", "weight_decay": 0.0,
        "betas": [0.9, 0.99], "eps": 1e-6, "gradient_clip": 1.0,
        "train_utterances": 500, "selection_split": "D-dev-select",
        "dev_confirm_used": False, "output_root": str(out),
    })
    # The existing tested trainer is reused for the fixed T1/T2/T3 kernels.
    # Its output root is redirected before invocation, so independent method
    # checkpoints remain under this Job-B namespace.
    import yaml
    overlay = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    base_config = str(overlay.get("data_config", "configs/lss/l1b_candidates_dialogue_v2r3.yaml"))
    import experiments.job_b_training as legacy
    legacy.RESULTS_DIR = out
    legacy.CONFIG_PATH = base_config
    legacy.SEED = int(args.seed)
    legacy.BUDGET_HOURS = float(args.time_budget_minutes) / 60.0
    rc = legacy.main([])
    atomic_json(out / "training_status.json", {"status": "completed" if rc == 0 else "failed", "returncode": rc})
    return int(rc)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    return dry_run(args) if args.dry_run else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
