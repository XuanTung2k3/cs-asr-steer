#!/usr/bin/env python
"""Post-run serialization wrapper for the frozen P2-RJ-E analysis (added after the run, disclosed).

The frozen ``inference_cf_p2rje_analyze.analyze`` is called unchanged. Its result can contain
+inf (rho = +inf when gap <= 0, as the frozen P2-RJ rule defines), which strict JSON rejects.
Non-finite floats are written as the strings "inf" / "-inf" / "nan". No statistic, threshold or
decision is touched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json
import experiments.inference_cf_p2rje_analyze as ana


def finite_json(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = finite_json(ana.analyze(ROOT / args.run))
    atomic_json(ROOT / args.out, res)
    print(json.dumps({k: res[k] for k in ("diagnosis", "validity", "overlap_reproduction_all_original", "Q50", "leverage", "precision", "counts")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
