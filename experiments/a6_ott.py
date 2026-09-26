"""A6-OTT compact preflight and physical shard entry point."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A6-OTT compact preflight")
    parser.add_argument("--dry-run", action="store_true", help="validate reuse and print the compact matrix")
    parser.add_argument("--physical", action="store_true", help="run one physical compact shard")
    parser.add_argument("--model")
    parser.add_argument("--dataset", choices=("cs_dialogue", "ascend"))
    parser.add_argument("--side", choices=("encoder", "decoder"))
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-stop", type=int)
    parser.add_argument("--acceptance", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        if args.physical or args.acceptance:
            from csasr.experiments.a6_ott import main as physical_main
            physical_args = list(sys.argv[1:] if argv is None else argv)
            physical_args = [x for x in physical_args if x not in {"--physical"}]
            return physical_main(physical_args)
        raise SystemExit("use --dry-run, --physical, or --acceptance")
    from scripts.a6_ott_preflight import main as preflight_main
    return preflight_main()


if __name__ == "__main__":
    raise SystemExit(main())
