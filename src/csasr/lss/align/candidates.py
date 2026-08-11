"""Running the aligner families and caching their candidate tables.

Thin wrappers over `csasr.nat5h.aligners`, which is reused unchanged. The cache
is identity-guarded: a candidate table is reused only when it was produced by
the same code, config and model, exactly as the NAT5H pipeline does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from ...nat5h.schema import (
    RunIdentity,
    artifact_matches_identity,
    assert_schema_v2,
    empty_candidates,
)
from ...utils.logging import get_logger
from ..artifacts import write_parquet

log = get_logger(__name__)

SUPPORTED_FAMILIES = ("existing_ctc", "whisper_dtw", "qwen_forced_aligner")


def load_or_run(path: str | Path, builder: Callable[[], pd.DataFrame],
                identity: RunIdentity, *, overwrite: bool = False) -> pd.DataFrame:
    """Reuse a cached candidate table only if it matches the current identity."""
    path = Path(path)
    if path.is_file() and not overwrite:
        cached = pd.read_parquet(path)
        if len(cached) and artifact_matches_identity(cached, identity):
            log.info("reusing cached candidates: %s (%d rows)", path, len(cached))
            return cached
        log.info("cached candidates at %s do not match this run's identity; rebuilding",
                 path)
    table = builder()
    write_parquet(table, path)
    return table


def run_families(manifest: pd.DataFrame, cfg: dict, geometry, identity: RunIdentity,
                 families: Sequence[str], *, bundle=None,
                 out_dir: Path | None = None,
                 overwrite: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run each configured aligner family; a family that fails is recorded."""
    from ...nat5h.aligners import run_existing_ctc, run_whisper_dtw

    frames: list[pd.DataFrame] = []
    report: dict[str, Any] = {"families": {}, "active": []}

    for family in families:
        if family not in SUPPORTED_FAMILIES:
            report["families"][family] = {"state": "unsupported"}
            continue
        if family == "qwen_forced_aligner":
            # The probe validates the family in its own process; the sweep
            # consumes what the probe persisted. Never run inline: a hang here
            # would take the whole sweep, and the probe's process-group deadline
            # is the only thing that can stop it.
            cached = (Path(out_dir) / f"candidates_{family}.parquet"
                      if out_dir is not None else None)
            if cached is not None and cached.is_file():
                table = pd.read_parquet(cached)
                if len(table) and artifact_matches_identity(table, identity):
                    frames.append(table)
                    report["families"][family] = {
                        "state": "ok", "rows": int(len(table)),
                        "valid_rows": int(table["is_valid"].sum())
                        if "is_valid" in table else None,
                        "source": "persisted_by_probe"}
                    report["active"].append(family)
                    continue
                report["families"][family] = {
                    "state": "unavailable",
                    "error": "probe output does not match this run's identity"}
                continue
            report["families"][family] = {"state": "deferred_to_probe"}
            continue

        def builder(family=family):
            if family == "existing_ctc":
                return run_existing_ctc(manifest, cfg, geometry, identity)
            if bundle is None:
                raise RuntimeError("whisper_dtw requires a loaded model bundle")
            table, meta = run_whisper_dtw(bundle, manifest, cfg, identity)
            report["families"].setdefault(family, {})["mapping"] = meta
            return table

        try:
            if out_dir is not None:
                table = load_or_run(Path(out_dir) / f"candidates_{family}.parquet",
                                    builder, identity, overwrite=overwrite)
            else:
                table = builder()
            frames.append(table)
            report["families"].setdefault(family, {}).update({
                "state": "ok", "rows": int(len(table)),
                "valid_rows": int(table["is_valid"].sum()) if "is_valid" in table else None,
            })
            report["active"].append(family)
        except Exception as exc:
            log.warning("aligner family %s unavailable: %s", family, exc)
            report["families"].setdefault(family, {}).update({
                "state": "unavailable", "error": repr(exc)})

    if not frames:
        return empty_candidates(), report
    out = pd.concat(frames, ignore_index=True)
    assert_schema_v2(out, artifact="lss candidate table")
    return out, report


# The decoder-query convention is swept by `csasr.lss.align.bias.pred_start_sweep`,
# which calls `align_batch` directly and scores against synthetic ground truth.
# It deliberately does not go through `nat5h.aligners.run_whisper_dtw`: that
# function does not thread `pred_start_offset`, and `csasr.nat5h` is read-only
# here so its recorded artifacts stay reproducible. If the sweep selects the
# other convention, adopting it for production alignment is a config change plus
# a one-line pass-through in that function -- a decision for the validation
# stage, not a diagnostic.
