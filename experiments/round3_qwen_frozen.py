#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from steer_sweep.backends import LazyQwenBackend
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--dry-run", action="store_true"); p.add_argument("--execute", action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print("Round 3 Qwen frozen dry-run: lazy backend, no Qwen initialization"); return 0
    if not a.execute: raise SystemExit("Round 3 is implemented but not executable without --execute")
    LazyQwenBackend().load(allow_initialize=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
