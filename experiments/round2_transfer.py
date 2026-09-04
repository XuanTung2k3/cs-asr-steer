#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--lock", required=True); p.add_argument("--dry-run", action="store_true"); p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    if a.dry_run: print("Round 2 transfer dry-run: lock required; D-test-lock/SEAME are evaluation-only"); return 0
    if not a.execute: raise SystemExit("Round 2 is implemented but not executable without --execute")
    raise SystemExit("Round 2 transfer runner requires a validated Round-1 lock and is not submitted")
if __name__ == "__main__": raise SystemExit(main())
