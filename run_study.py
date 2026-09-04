#!/usr/bin/env python
"""Two-track steering study orchestrator.

    python run_study.py --track {A|B|all} --all [--smoke] [--budget-hours N] [--resume]

Track A pins to one GPU and runs the inference-only sweep; Track B pins to
another and runs training. They are independent processes and MUST NOT share a
GPU: two arms on one device makes wall-clock and peak-memory numbers
meaningless.

On this cluster jobs are submitted with `sbatch`, so the two commands in the
task specification are wrapped by `sbatch/study_track_a.sh` and
`sbatch/study_track_b.sh`. The Python interface below is unchanged and is what
those scripts call.

Everything written is stamped `development_only_diagnostic`.
`load_split("D-test")` raises. Track A Confirm is never run automatically: the
single selected configuration is printed and the job stops for human approval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from steer_sweep import config as C           # noqa: E402
from steer_sweep.store import Budget          # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", choices=["A", "B", "all"], required=True)
    p.add_argument("--all", action="store_true",
                   help="run the full programme for the selected track")
    p.add_argument("--smoke", action="store_true",
                   help=f"{C.SMOKE['utterances']} utterances and "
                        f"{C.SMOKE['train_steps']} training steps, including "
                        f"every Track B arm and the init diagnostics")
    p.add_argument("--resume", action="store_true",
                   help="skip cells whose config hash is already in the JSONL")
    p.add_argument("--budget-hours", type=float, default=None,
                   help="stop cleanly and write a partial summary")
    p.add_argument("--out", default=str(C.DEFAULT_OUT))
    p.add_argument("--config", default=C.BASE_CONFIG)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--full-weighted-grid", action="store_true",
                   help="Track A Stage B: run the full 3x3 weighted grid from "
                        "section 3.2 instead of the three-mix default that "
                        "keeps Stage B at section 3.6's ~8 runs")
    p.add_argument("--site", default=None,
                   help="Track B: override the intervention site")
    p.add_argument("--layers", default=None,
                   help="Track B: comma-separated layer override")
    p.add_argument("--skip-preflight", action="store_true",
                   help="NOT for a real run; preflight is mandatory and this "
                        "flag exists only for harness development")
    p.add_argument("--summary-only", action="store_true",
                   help="regenerate out/SUMMARY.md from existing rows")
    return p.parse_args(argv)


def resolve_out_dir(out: str | Path, smoke: bool) -> Path:
    """Isolate smoke output from a full run's `--out` directory.

    `Cell.config_hash()` and Track B's config-hash key describe an
    intervention/arm CONFIGURATION, not the population it ran on, so an
    8-utterance smoke cell and its 300-utterance full-scale counterpart share
    the same identity. Sharing one directory between them would let
    `--resume` silently treat a smoke-scale result as if it were the real
    one. The repository's own v2r3 convention for this exact problem is to
    give smoke output its own generation namespace, refused by name as an
    input everywhere else -- mirrored here rather than trying to make one
    shared JSONL safely serve both scales.
    """
    out_dir = Path(out)
    return (out_dir / "smoke") if smoke else out_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = resolve_out_dir(args.out, args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)

    from steer_sweep.summary import write_summary

    if args.summary_only:
        print(f"wrote {write_summary(out_dir)}")
        return 0

    if args.track == "all":
        print("--track all runs both tracks IN ONE PROCESS on ONE GPU, which "
              "makes wall-clock and peak-memory numbers meaningless. Submit "
              "the two jobs separately:\n"
              "  sbatch sbatch/study_track_a.sh\n"
              "  sbatch sbatch/study_track_b.sh", file=sys.stderr)
        return 64

    from csasr.utils.config import load_config
    from csasr.utils.logging import setup_logging
    from csasr.models.whisper import load_whisper

    log = setup_logging(level="INFO")
    cfg = load_config(args.config)
    budget = Budget(args.budget_hours)

    log.info("track %s | smoke=%s resume=%s budget=%s | CUDA_VISIBLE_DEVICES=%r",
             args.track, args.smoke, args.resume, args.budget_hours,
             os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    bundle = load_whisper(cfg)
    _assert_geometry(bundle)

    started = time.time()
    try:
        if args.track == "A":
            from steer_sweep.study import run_track_a
            run_track_a(bundle, cfg, out_dir=out_dir, smoke=args.smoke,
                        resume=args.resume, budget=budget,
                        batch_size=args.batch_size, log=log,
                        full_weighted_grid=args.full_weighted_grid,
                        skip_preflight=args.skip_preflight)
        else:
            from steer_sweep.study import run_track_b
            layers = ([int(v) for v in args.layers.split(",")]
                      if args.layers else None)
            run_track_b(bundle, cfg, out_dir=out_dir, smoke=args.smoke,
                        resume=args.resume, budget=budget,
                        batch_size=args.batch_size, log=log,
                        skip_preflight=args.skip_preflight,
                        site=args.site, layers=layers)
    finally:
        path = write_summary(out_dir)
        log.info("track %s finished in %.1f min; summary at %s",
                 args.track, (time.time() - started) / 60.0, path)
    return 0


def _assert_geometry(bundle) -> None:
    """The context of section 0, checked rather than assumed."""
    checks = [
        ("encoder layers", bundle.num_encoder_layers, C.EXPECTED_ENCODER_LAYERS),
        ("decoder layers", bundle.num_decoder_layers, C.EXPECTED_DECODER_LAYERS),
        ("d_model", bundle.d_model, C.EXPECTED_D_MODEL),
        ("encoder frames", bundle.max_encoder_frames, C.EXPECTED_ENCODER_FRAMES),
    ]
    bad = [(n, o, e) for n, o, e in checks if int(o) != int(e)]
    if bad:
        raise SystemExit(
            "model geometry does not match the study's stated context:\n"
            + "\n".join(f"  {n}: observed {o}, expected {e}" for n, o, e in bad))
    frames_per_sec = 1.0 / bundle.encoder_step_sec
    if abs(frames_per_sec - C.FRAMES_PER_SECOND) > 1e-6:
        raise SystemExit(f"encoder frame rate {frames_per_sec} != "
                         f"{C.FRAMES_PER_SECOND} fps")


if __name__ == "__main__":
    raise SystemExit(main())
