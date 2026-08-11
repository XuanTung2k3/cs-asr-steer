"""Shared CLI plumbing for every stage: config, seed, run dir, metadata, gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.config import REPO_ROOT, art, load_config, run_dir
from ..utils.logging import setup_logging, write_run_metadata
from ..utils.provenance import stage_provenance
from ..utils.seed import set_seed
from ..utils.hashing import sha256_file
from ..utils.status import write_status


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="path to the stage config")
    p.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    p.add_argument("--output-dir", default=None, help="override experiment.output_root")
    p.add_argument("--resume", action="store_true", help="reuse completed work")
    p.add_argument("--overwrite", action="store_true", help="recompute and replace artifacts")
    p.add_argument("--dry-run", action="store_true", help="validate config and inputs only")
    p.add_argument("--limit", type=int, default=None, help="cap utterances (debug/smoke runs)")
    p.add_argument("--force-prereq", action="store_true",
                   help="run despite an unmet gate (documented diagnostic only)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="dotted config override")
    return p


def _reset_gpu_stats() -> None:
    """Start each stage's peak-memory measurement from zero (guide section 55)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:      # instrumentation must never break a stage
        pass


def gpu_stats() -> dict:
    """Peak GPU memory for this stage, so batch sizes can be tuned on evidence."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        gb = 1024 ** 3
        return {
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 2),
            "device_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / gb, 2),
        }
    except Exception:
        return {}


def prepare(args, stage: str, tag: str | None = None, *,
            claim: bool = True) -> tuple[dict, Path, Any]:
    """Load config, seed everything, set up the run directory and logging.

    ``claim=False`` skips the `running` status write. Use it when the caller
    acquires `StageLock` itself: writing `running` before the lock is held lets
    a second job that is about to be refused the lock overwrite the active run's
    status on its way out.
    """
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.output_dir is not None:
        cfg["experiment"]["output_root"] = args.output_dir
        cfg["data"]["manifest"] = str(Path(args.output_dir) / "manifests" / "cs_dialogue.parquet")
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    rdir = run_dir(cfg, stage, tag)
    log = setup_logging(rdir, cfg.get("runtime", {}).get("log_level", "INFO"))
    log.info("stage=%s seed=%d config=%s", stage, seed, cfg.get("_config_path"))
    _reset_gpu_stats()
    write_run_metadata(rdir, cfg, REPO_ROOT)
    if claim:
        write_status(cfg["experiment"]["output_root"], stage, "running",
                     run_dir=str(rdir), artifacts=[])
    global _LAST_PREPARE
    _LAST_PREPARE = (cfg, rdir, log)
    return cfg, rdir, log


#: the last (cfg, run_dir, logger) `prepare` produced, so a stage wrapper can
#: record a crash that happened after setup without threading them out by hand
_LAST_PREPARE: tuple[dict | None, Path | None, Any] = (None, None, None)


def last_prepare_context() -> tuple[dict | None, Path | None, Any]:
    return _LAST_PREPARE


def claim_stage(cfg: dict, stage: str, rdir: Path) -> None:
    """Mark the stage as running. Call this *after* the lock is held."""
    write_status(cfg["experiment"]["output_root"], stage, "running",
                 run_dir=str(rdir), artifacts=[])


def terminal_on_exception(cfg: dict, stage: str, rdir: Path, log: Any,
                          exc: BaseException) -> int:
    """Record an unhandled exception as `failed` and return its exit code.

    Without this a crash leaves the status at `running`, which every
    prerequisite check reads as "a job is still working on it" -- so the stage
    is neither runnable nor reported as broken, and the pipeline simply stalls.
    """
    import traceback

    detail = traceback.format_exc(limit=30)
    log.error("stage %s raised %s: %s", stage, type(exc).__name__, exc)
    try:
        (rdir / "exception.txt").write_text(detail, encoding="utf-8")
    except OSError:                                     # pragma: no cover
        pass
    # A run that could not take the lock never owned this stage, so it must not
    # write its status: doing so would let a job that was correctly refused
    # overwrite the state of the run that is actually working.
    if "is locked by" in str(exc):
        log.error("another run holds this stage; leaving its status untouched")
        return 2
    write_status(cfg["experiment"]["output_root"], stage, "failed",
                 run_dir=str(rdir), complete=False,
                 error=f"{type(exc).__name__}: {exc}",
                 traceback=detail[-4000:])
    return 2


def finish(cfg: dict, stage: str, rdir: Path, metrics: dict, gate: dict | None,
           artifacts: list[str] | None = None, status_override: str | None = None,
           **extra: Any) -> dict:
    """Persist metrics + status and return the status payload.

    ``extra`` is written verbatim into the status payload; it is how a stage
    records run-level facts a downstream stage must see (for example that it
    ran with an unmet prerequisite).
    """
    provenance = stage_provenance(cfg, stage)
    stats = gpu_stats()
    if stats:
        (rdir / "gpu_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        metrics = {**metrics, "_gpu": stats}
    metrics = {**metrics, "_provenance": provenance}
    (rdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (rdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    status = status_override or ("passed" if (gate is None or gate["passed"]) else "failed")
    payload = write_status(cfg["experiment"]["output_root"], stage, status, gate=gate,
                           run_dir=str(rdir), artifacts=artifacts or [],
                           provenance=provenance,
                           complete=bool(gate is None or gate["passed"]),
                           **extra)
    manifest_path = Path(cfg.get("data", {}).get("manifest", ""))
    if manifest_path.is_file():
        (rdir / "data_hashes.json").write_text(json.dumps({
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }, indent=2), encoding="utf-8")
    (rdir / "status.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    return payload


def save_report(path: str | Path, title: str, sections: list[tuple[str, str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.rstrip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", max_rows: int = 200) -> str:
    """Small dependency-free markdown table renderer."""
    if df is None or len(df) == 0:
        return "_(empty)_"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        else:
            d[c] = d[c].astype(str)
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in d.astype(str).values]
    return "\n".join([header, sep] + rows)


def load_units(cfg: dict, subset: str) -> pd.DataFrame:
    path = art(cfg, "alignments", f"{subset}.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"alignment table missing for {subset}: {path}. Run E1 first.")
    return pd.read_parquet(path)


def primary_baseline_id(cfg: dict) -> str:
    """The baseline frozen by P0 (falls back to B1_ZH if P0 has not run)."""
    path = art(cfg, "baselines", "primary_baseline.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["system"]
    return "B1_ZH"


def baseline_predictions(cfg: dict, subset: str, system: str | None = None) -> pd.DataFrame:
    system = system or primary_baseline_id(cfg)
    path = art(cfg, "baselines", system, f"{subset}.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"cached baseline predictions missing: {path}. Run P0 (or E2's baseline "
            f"pass for training subsets) first.")
    return pd.read_parquet(path)
