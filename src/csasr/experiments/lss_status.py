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


def stage_reusable(root, stage: str, *, config: str) -> str:
    """`reusable`, or `stale:<reason>` -- may a chain skip this stage?

    A chain skips a stage that already passed so an allocation that ran out of
    wall clock is resumed instead of restarted. "Already passed" is not enough
    on its own: it must have passed under the *same decisions*, or the skip
    silently keeps an answer to a different question.

    The comparison is the recorded `resolved_config_hash`. It reproduces exactly
    when the same config path is loaded again -- verified against job 38502 --
    but it is a hash of the config *path string* among other things, so an
    invocation that spells the path differently reads as stale. That errs toward
    re-running, which is the safe direction.

    `source_snapshot_hash` is deliberately **not** compared, even though a code
    change is just as capable of invalidating a pass. It hashes
    `environment.lock.txt`, which `cs_asr_lss.sh preflight` rewrites with a fresh
    timestamp and Slurm job id at the start of every job, so it differs on every
    single run. Requiring it to match would mean never skipping anything, which
    is the same as deleting resumption -- and resumption is what protects the
    one expensive stage. Code changes are handled by `--overwrite`, which the
    caller now actually honours.
    """
    from ..utils.config import load_config as _load_config
    from ..utils.provenance import resolved_config_hash

    payload = read_status(root, stage)
    status = str(payload.get("status", "pending"))
    if status != "passed":
        return f"stale:status={status}"
    recorded = str((payload.get("provenance") or {}).get("resolved_config_hash", ""))
    if not recorded:
        return "stale:no_recorded_config_hash"
    current = resolved_config_hash(_load_config(config))
    if recorded != current:
        return f"stale:config_hash {recorded[:12]}!={current[:12]}"
    return "reusable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSS pipeline status")
    parser.add_argument("--config", default="lss/base.yaml")
    parser.add_argument("--report", action="store_true",
                        help="also print each stage's gate criteria")
    parser.add_argument("--stage-status", metavar="STAGE", default=None,
                        help="print only this stage's status word, for scripts")
    parser.add_argument("--stage-reusable", metavar="STAGE", default=None,
                        help="print 'reusable' if this stage passed under the "
                             "same resolved config, else 'stale:<reason>'")
    parser.add_argument("--stage-config", metavar="PATH", default=None,
                        help="the config --stage-reusable compares against; pass "
                             "it exactly as the stage is invoked with")
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

    if args.stage_reusable:
        if args.stage_reusable not in LSS_STAGES:
            raise SystemExit(f"unknown stage {args.stage_reusable!r}")
        print(stage_reusable(root, args.stage_reusable,
                             config=args.stage_config or args.config))
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
