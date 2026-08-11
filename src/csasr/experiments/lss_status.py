"""Print what has run and what is runnable next.

    python -m csasr.experiments.lss_status --config configs/lss/base.yaml
    python -m csasr.experiments.lss_status --stage-status l1a_diag   # one word

Orchestration reads `--stage-status`, never a stage's exit code: exit codes say
whether a command worked, status files say what it found. See
`csasr.lss.gates.STATUS_EXIT_CODES`.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ..lss.prereq import prerequisite_report
from ..utils.config import artifacts_root, load_config
from ..utils.status import LSS_STAGES, read_status, write_overview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSS pipeline status")
    parser.add_argument("--config", default="lss/base.yaml")
    parser.add_argument("--report", action="store_true",
                        help="also print each stage's gate criteria")
    parser.add_argument("--stage-status", metavar="STAGE", default=None,
                        help="print only this stage's status word, for scripts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = artifacts_root(cfg)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage_status:
        if args.stage_status not in LSS_STAGES:
            raise SystemExit(f"unknown stage {args.stage_status!r}")
        print(read_status(root, args.stage_status).get("status", "pending"))
        return 0

    overview = write_overview(root, stages=LSS_STAGES)
    table = prerequisite_report(root)

    if args.json:
        print(json.dumps({"overview": overview,
                          "prerequisites": table.to_dict(orient="records")},
                         indent=2))
        return 0

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(f"artifacts root: {root}\n")
        print(table.to_string(index=False))

    if args.report:
        for stage in LSS_STAGES:
            payload = read_status(root, stage)
            gate = payload.get("gate")
            if not gate:
                continue
            print(f"\n{stage}: {payload.get('status')}"
                  + (f"  next: {payload['next_action']}"
                     if payload.get("next_action") else "")
                  + (f"  blocked_on: {','.join(payload['blocked_reasons'])}"
                     if payload.get("blocked_reasons") else ""))
            for c in gate.get("criteria", []):
                if c.get("comparison") == "report":
                    print(f"  [    ] {c['name']}: {c['value']}")
                else:
                    flag = "PASS" if c["passed"] else "FAIL"
                    print(f"  [{flag}] {c['name']}: {c['value']} "
                          f"{c['comparison']} {c['threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
