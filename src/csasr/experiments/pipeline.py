"""Pipeline orchestration with prerequisite checks and gate enforcement.

    python -m csasr.experiments.pipeline --config configs/base.yaml \
        --from-stage e1 --to-stage e5 --stop-on-failed-gate --resume

Every stage is also independently runnable; this only sequences them, validates
expected artifacts and refuses to continue past a failed gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..utils.config import artifacts_root, load_config
from ..utils.logging import setup_logging
from ..utils.status import STAGES, read_status, write_overview

ORDER = [
    "p0",
    "e1",
    "e2_pilot",
    "e3_pilot",
    "e4_pilot",
    "e2_full",
    "e3_full",
    "e4_refine",
    "e4_confirm",
    "e5",
]

MODULES = {
    "p0": ("csasr.experiments.p0_baseline", "experiments/e1_alignment.yaml", []),
    "e1": ("csasr.experiments.e1_alignment", "experiments/e1_alignment.yaml", []),
    "e2_pilot": ("csasr.experiments.e2_directions", "experiments/e2_directions.yaml", []),
    "e2_full": ("csasr.experiments.e2_directions", "experiments/e2_directions.yaml",
                ["--set", "directions.subset=train_direction_full"]),
    "e3_pilot": ("csasr.experiments.e3_separability", "experiments/e3_separability.yaml",
                 ["--phase", "pilot"]),
    "e3_full": ("csasr.experiments.e3_separability", "experiments/e3_separability.yaml",
                ["--phase", "full"]),
    "e4_pilot": ("csasr.experiments.e4_oracle", "experiments/e4_oracle.yaml", ["--phase", "pilot"]),
    "e4_refine": ("csasr.experiments.e4_oracle", "experiments/e4_oracle.yaml", ["--phase", "refine"]),
    "e4_confirm": ("csasr.experiments.e4_oracle", "experiments/e4_oracle.yaml", ["--phase", "confirm"]),
    "e5": ("csasr.experiments.e5_boundaries", "experiments/e5_boundaries.yaml", []),
}

EXPECTED_ARTIFACTS = {
    "p0": ["manifests/dev_select.parquet", "baselines/primary_baseline.json",
           "reports/p0_baseline_audit.md"],
    "e1": ["alignments/dev_select.parquet", "metrics/e1_alignment_metrics.json"],
    "e2_pilot": ["metrics/e2_pilot_direction_statistics.json"],
    "e2_full": ["metrics/e2_full_direction_statistics.json"],
    "e3_pilot": ["metrics/e3_pilot_layer_metrics.parquet",
                 "reports/e3_pilot_selected_layers.json"],
    "e3_full": ["metrics/e3_full_layer_metrics.parquet",
                "reports/e3_full_selected_layers.json"],
    "e4_pilot": ["metrics/e4_all_runs.parquet"],
    "e4_refine": ["metrics/e4_selected_config.json"],
    "e4_confirm": ["metrics/e4_all_runs.parquet"],
    "e5": ["metrics/e5_mask_results.parquet", "metrics/e5_summary.json"],
}

def stage_status_key(stage: str) -> str:
    return stage


def run_stage(stage: str, args, log) -> int:
    module, default_cfg, extra = MODULES[stage]
    mod = __import__(module, fromlist=["main"])
    argv = ["--config", args.config_for(stage, default_cfg)] + extra
    if args.resume:
        argv.append("--resume")
    if args.overwrite:
        argv.append("--overwrite")
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    for ov in args.overrides:
        argv += ["--set", ov]
    if stage == "p0" and args.build_manifest:
        argv.append("--build-manifest")
    log.info("=== running %s: python -m %s %s", stage, module, " ".join(argv))
    return int(mod.main(argv))


def validate_artifacts(cfg, stage: str, log) -> bool:
    root = artifacts_root(cfg)
    missing = [p for p in EXPECTED_ARTIFACTS.get(stage, []) if not (root / p).exists()]
    if missing:
        log.error("%s finished but expected artifacts are missing: %s", stage, missing)
        return False
    return True


class _Args:
    def __init__(self, ns):
        self.__dict__.update(vars(ns))

    def config_for(self, stage: str, default: str) -> str:
        override = getattr(self, f"config_{stage}", None)
        return override or default


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="P0-E5 pipeline orchestration")
    p.add_argument("--config", default="base.yaml",
                   help="base config (used to locate the artifacts root)")
    p.add_argument("--from-stage", default="p0", choices=ORDER)
    p.add_argument("--to-stage", default="e5", choices=ORDER)
    p.add_argument("--only", nargs="*", default=None, choices=ORDER)
    p.add_argument("--stop-on-failed-gate", action="store_true", default=True)
    p.add_argument("--no-stop-on-failed-gate", dest="stop_on_failed_gate",
                   action="store_false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--build-manifest", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--status", action="store_true", help="print stage status and exit")
    ns = p.parse_args(argv)
    args = _Args(ns)

    cfg = load_config(args.config, args.overrides)
    root = artifacts_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(root / "runs" / "pipeline", cfg.get("runtime", {}).get("log_level", "INFO"))

    if args.status:
        print(json.dumps(write_overview(root), indent=2))
        for s in STAGES:
            st = read_status(root, s)
            if st.get("gate"):
                print(f"\n{s}: {st['status']}")
                for c in st["gate"].get("criteria", []):
                    flag = "PASS" if c["passed"] else "FAIL"
                    print(f"  [{flag}] {c['name']}: {c['value']} {c['comparison']} {c['threshold']}")
        return 0

    stages = args.only or ORDER[ORDER.index(args.from_stage): ORDER.index(args.to_stage) + 1]
    log.info("pipeline stages: %s", stages)
    overall = 0
    for stage in stages:
        try:
            rc = run_stage(stage, args, log)
        except Exception:
            log.exception("stage %s raised", stage)
            rc = 1
        ok_artifacts = validate_artifacts(cfg, stage, log) if rc in (0, 2) else False
        st = read_status(root, stage_status_key(stage))
        gate_passed = st.get("gate", {}).get("passed", rc == 0)
        log.info("stage %s: rc=%d gate_passed=%s artifacts_ok=%s",
                 stage, rc, gate_passed, ok_artifacts)
        if rc == 1 or not ok_artifacts or not gate_passed:
            overall = rc or 2
            if args.stop_on_failed_gate:
                log.error("stopping after %s (completed outputs are preserved)", stage)
                break
    log.info("pipeline status: %s", json.dumps(write_overview(root)))
    return overall


if __name__ == "__main__":
    sys.exit(main())
