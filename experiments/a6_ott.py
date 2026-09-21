"""A6-OTT compact entry point.

The current repository has reusable exact-site callbacks but no contract-
conformant end-to-end free-decoding backend.  This entry point therefore
exposes the CPU preflight/dry-run and refuses to silently fall back to the
superseded atlas executor.  Physical execution can be added behind the same
compact enumeration once a human authorizes the backend.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A6-OTT compact preflight")
    parser.add_argument("--dry-run", action="store_true", help="validate reuse and print the compact matrix")
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("A6-OTT physical execution is not submitted by preflight; use --dry-run")
    from scripts.a6_ott_preflight import main as preflight_main
    return preflight_main()


if __name__ == "__main__":
    raise SystemExit(main())
