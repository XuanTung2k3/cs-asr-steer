#!/usr/bin/env python
from __future__ import annotations
import argparse
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--dry-run", action="store_true"); p.add_argument("--execute", action="store_true"); a=p.parse_args(argv)
    if a.dry_run: print("Round 3 Qwen training dry-run: lazy backend, mocked tests only"); return 0
    if not a.execute: raise SystemExit("Round 3 is implemented but not executable without --execute")
    raise SystemExit("Round 3 training is implemented but not submitted")
if __name__ == "__main__": raise SystemExit(main())
