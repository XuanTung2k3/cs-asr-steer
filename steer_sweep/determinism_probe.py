#!/usr/bin/env python
"""Fresh-process leg of the section 4.1 determinism check.

Invoked as a subprocess (``python -m steer_sweep.determinism_probe ...``) from
``stage_d_ext.run_determinism_check`` so the comparison spans a brand-new
Python interpreter and CUDA context, not just a second call inside the same
process. Decodes one steered cell and writes its hypotheses to ``--result``
as JSON; the caller diffs them against its own in-process runs.

Deliberately minimal: no preflight, no baselines, no analysis. Its only job
is to reproduce one decode exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True,
                        help="the SAME out dir as the parent process, so the "
                             "direction cache is reused rather than rebuilt")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beams", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from steer_sweep import data as D
    from steer_sweep import config as C
    from steer_sweep.directions import DirectionStore
    from steer_sweep.stage_d_ext import _decode_probe_cell

    cfg = load_config(args.config)
    bundle = load_whisper(cfg)
    out_dir = Path(args.out_dir)
    pop = D.build_population(bundle, cfg, C.DEV_SELECT, assert_anchors=True)
    store = DirectionStore(cfg, out_dir / "directions_cache")

    texts = _decode_probe_cell(bundle, pop, store, layer=args.layer,
                               alpha=args.alpha, num_beams=args.beams,
                               batch_size=args.batch_size, log=None)

    Path(args.result).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(texts)} hypotheses to {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
