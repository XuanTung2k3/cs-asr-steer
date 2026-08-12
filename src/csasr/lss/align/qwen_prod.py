"""Production alignment with the third family, over a whole manifest.

The L1a probe answers "can this environment run the Qwen aligner at all" on eight
utterances. It is a capability check and nothing else, but it was also the only
Qwen code path that existed, so:

* `candidates.run_families` consumed whatever the probe had persisted;
* the probe's table is keyed to eight *corpus* utterances and stamped with L1a's
  config identity, so L1b's identity check refused it -- job 38573's
  `candidates_all.parquet` has no Qwen rows at all;
* nothing could ever score Qwen against synthetic boundaries, because those
  utterance ids do not exist in the corpus.

This module is the production path: the same adapter, over an arbitrary manifest,
in its own process group, in bounded chunks, published atomically only when it
covered everything it was asked to cover.

What each rule is protecting against
------------------------------------
* **Its own deadline.** 1,800 utterances is 225x the probe's eight, so the probe's
  30-minute bound says nothing about it. `production_deadline_minutes` is separate
  and is not allowed to default to the probe's.
* **A killable process group.** `start_new_session=True` plus `killpg` is what
  reaches dataloader workers and CUDA helpers. A checkpoint stuck in a native call
  ignores SIGALRM, so an in-process alarm cannot bound this.
* **No reusable partial artifact.** Chunks are written inside an attempt-private
  directory that the cache cannot see. A timeout discards the whole attempt, so
  the next run finds either a complete table or nothing -- never a prefix that
  looks complete because it is a valid parquet.
* **Universe coverage.** A table missing a chunk is discarded rather than
  published: a family that aligned 1,400 of 1,800 utterances would score a
  perfect validity *rate* on what it did produce. Per-utterance aligner failures
  are different -- the adapter emits invalid rows for those, so they are covered
  and are counted against the family, which is correct.
* **One frozen language variant.** `qwen_probe.PROBE_LANGUAGE` records the
  measurement behind that choice; two variants of one family are one independence
  class and `nat5h.consensus._family_representatives` would discard the second
  anyway.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ...utils.logging import get_logger
from .qwen_probe import (
    INFRASTRUCTURE_ERRORS,
    PROBE_LANGUAGE,
    _adapter_class,
    _release_gpu,
    _terminate_group,
)

log = get_logger(__name__)

FAMILY = "qwen_forced_aligner"
CANDIDATES_FILE = f"candidates_{FAMILY}.parquet"

#: entry point the child runs. A fresh interpreter, so a hang inside native CUDA
#: code is a hung *process* the parent can kill.
_CHILD_MAIN = "csasr.lss.align.qwen_prod"

#: default utterances per chunk. Small enough that a stall is bounded by one
#: chunk's worth of progress reporting, large enough that the per-chunk
#: bookkeeping is negligible against a 0.6B forward pass per utterance.
DEFAULT_CHUNK_SIZE = 200


@dataclass
class QwenProductionResult:
    """Terminal in every path, and never a scientific claim it did not measure."""

    state: str                        # "ok" | "blocked" | "completed_no_go" | "failed"
    reason: str | None = None
    purpose: str = ""
    language: str = PROBE_LANGUAGE
    model_dir: str = ""
    package_version: str | None = None
    expected_utterances: int = 0
    covered_utterances: int = 0
    universe_complete: bool = False
    candidate_rows: int = 0
    valid_rows: int = 0
    chunks_expected: int = 0
    chunks_written: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    elapsed_seconds: float = 0.0
    deadline_seconds: float = 0.0
    timed_out: bool = False
    published: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    process_group: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def production_deadline_seconds(cfg: Mapping[str, Any], n_utterances: int) -> float:
    """The wall clock this run is allowed, and why it is not the probe's.

    Configured as an absolute number of minutes plus a per-utterance allowance, so
    a manifest 225x the probe's size does not inherit a bound that was set for
    eight items. Both are recorded in the result.
    """
    qcfg = ((cfg.get("alignment") or {}).get("qwen") or {})
    absolute = float(qcfg.get("production_deadline_minutes", 90)) * 60.0
    per_utterance = float(qcfg.get("production_seconds_per_utterance", 0.0))
    return max(absolute, per_utterance * max(int(n_utterances), 0))


def chunk_bounds(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    size = max(int(chunk_size), 1)
    return [(i, min(i + size, int(n_rows))) for i in range(0, int(n_rows), size)]


def _attempt_dir(out_dir: str | Path) -> Path:
    directory = Path(out_dir) / f".qwen_prod_{os.getpid()}_{int(time.time())}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _discard(attempt: Path | None) -> list[str]:
    if attempt is None or not attempt.is_dir():
        return []
    contents = [str(p) for p in sorted(attempt.rglob("*")) if p.is_file()]
    shutil.rmtree(attempt, ignore_errors=True)
    if contents:
        log.warning("discarded %d file(s) from an incomplete Qwen production "
                    "attempt: %s", len(contents), contents[:5])
    return contents


def _read_chunks(attempt: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(attempt.glob("chunk_*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:                            # pragma: no cover
            log.warning("unreadable chunk %s: %s", path, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_qwen_production(cfg: dict, manifest: pd.DataFrame, *,
                        out_dir: str | Path, purpose: str = "natural",
                        stage: str = "l1b_valid", run_dir: str | Path | None = None,
                        parents: Sequence[Mapping[str, Any]] = (),
                        taint_reasons: Sequence[str] = (),
                        deadline_seconds: float | None = None,
                        chunk_size: int | None = None,
                        python: str | None = None) -> QwenProductionResult:
    """Align every utterance in ``manifest`` with the Qwen family.

    Terminal in every path. Publishes `candidates_qwen_forced_aligner.parquet`
    with an authenticating manifest **only** when every chunk completed and every
    requested utterance is represented; otherwise nothing is published and the
    state is `blocked`.
    """
    from ...lss import manifest as manifest_mod
    from .candidates import request_manifest_record

    qcfg = ((cfg.get("alignment") or {}).get("qwen") or {})
    model_dir = str(qcfg.get("local_model_dir", ""))
    language = str(qcfg.get("probe_language", PROBE_LANGUAGE))
    size = int(chunk_size or qcfg.get("production_chunk_size", DEFAULT_CHUNK_SIZE))
    expected = sorted(set(manifest["utterance_id"].astype(str))) if len(manifest) else []
    request_record = request_manifest_record(manifest)
    if deadline_seconds is None:
        deadline_seconds = production_deadline_seconds(cfg, len(expected))
    started = time.monotonic()
    bounds = chunk_bounds(len(manifest), size)

    base = QwenProductionResult(
        state="blocked", purpose=str(purpose), language=language,
        model_dir=model_dir, expected_utterances=len(expected),
        chunks_expected=len(bounds), chunk_size=size,
        deadline_seconds=float(deadline_seconds))

    if not len(manifest):
        base.reason = "empty_manifest"
        return base
    if not model_dir or not Path(model_dir).exists():
        base.reason = "checkpoint_missing"
        return base

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = _attempt_dir(out_dir)
    request = attempt / "request.json"
    result_path = attempt / "result.json"
    manifest_path = attempt / "manifest.parquet"
    manifest.reset_index(drop=True).to_parquet(manifest_path, index=False)
    request.write_text(json.dumps({
        "cfg": cfg, "manifest": str(manifest_path), "result": str(result_path),
        "out_dir": str(attempt), "chunk_size": size, "language": language,
        # Per-utterance aligner failures go somewhere that survives: they are the
        # diagnostic for *why* a family's invalid rate is what it is, and writing
        # them into the attempt directory meant they were deleted by both the
        # success path and the discard path.
        "diagnostics_dir": str(Path(out_dir) / "qwen_diagnostics"),
    }, default=str), encoding="utf-8")

    process = subprocess.Popen(
        [python or sys.executable, "-m", _CHILD_MAIN, "--request", str(request)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        output, _ = process.communicate(timeout=float(deadline_seconds))
        code = int(process.returncode)
    except subprocess.TimeoutExpired:
        kill = _terminate_group(process)
        try:
            output, _ = process.communicate(timeout=5.0)
        except Exception:                                   # pragma: no cover
            output = ""
        written = len(list(attempt.glob("chunk_*.parquet")))
        base.reason = "deadline_exceeded"
        base.timed_out = True
        base.chunks_written = written
        base.elapsed_seconds = time.monotonic() - started
        base.process_group = kill
        base.quarantined = _discard(attempt)
        base.traceback = (output or "")[-2000:] or None
        log.warning("Qwen production exceeded its %.0f s deadline after %d/%d "
                    "chunks; process group %s terminated=%s killed=%s; nothing "
                    "published", deadline_seconds, written, len(bounds),
                    kill.get("pgid"), kill.get("terminated"), kill.get("killed"))
        return base

    base.elapsed_seconds = time.monotonic() - started
    if code != 0 or not result_path.is_file():
        base.reason = f"production_process_exit_{code}"
        base.chunks_written = len(list(attempt.glob("chunk_*.parquet")))
        base.quarantined = _discard(attempt)
        base.traceback = (output or "")[-2000:] or None
        return base

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    base.package_version = payload.get("package_version")
    base.chunks_written = int(payload.get("chunks_written", 0))
    if payload.get("state") != "ok":
        # a child that ran and reported a defect keeps that classification: a bug
        # in our code must not be laundered into a missing-aligner story
        base.state = str(payload.get("state", "blocked"))
        base.reason = str(payload.get("reason") or "child_reported_failure")
        base.traceback = payload.get("traceback") or (output or "")[-2000:] or None
        base.quarantined = _discard(attempt)
        return base

    table = _read_chunks(attempt)
    covered = sorted(set(table["utterance_id"].astype(str))) if len(table) else []
    base.candidate_rows = int(len(table))
    base.valid_rows = int(table["is_valid"].sum()) if len(table) and "is_valid" in table else 0
    base.covered_utterances = len(covered)
    base.universe_complete = bool(set(covered) == set(expected))

    if base.chunks_written != len(bounds) or not base.universe_complete:
        base.reason = ("incomplete_universe: "
                       f"{len(covered)}/{len(expected)} utterances, "
                       f"{base.chunks_written}/{len(bounds)} chunks")
        base.quarantined = _discard(attempt)
        log.warning("Qwen production covered %d of %d utterances; discarding "
                    "rather than publishing a partial table", len(covered),
                    len(expected))
        return base

    target = out_dir / CANDIDATES_FILE
    base.manifest = manifest_mod.publish_frame(
        target, table.reset_index(drop=True), stage=stage, cfg=cfg,
        run_dir=run_dir, parents=[p for p in parents if p],
        taint_reasons=list(taint_reasons),
        key_columns=("utterance_id", "reference_unit_index", "aligner_family",
                     "aligner_variant"),
        schema="nat5h_candidates_v2",
        extra={"request_manifest": request_record, "qwen_production": {
            "purpose": str(purpose), "language": language,
            "expected_utterances": len(expected),
            "covered_utterances": len(covered),
            "universe_complete": True,
            "chunks": len(bounds), "chunk_size": size,
            "deadline_seconds": float(deadline_seconds),
            "package_version": base.package_version,
        }})
    base.published = [str(target)]
    base.state = "ok"
    base.reason = None
    # quiet cleanup: on the success path the chunks were consumed, not discarded,
    # and logging them as discarded reads as data loss
    shutil.rmtree(attempt, ignore_errors=True)
    log.info("Qwen production: %d rows over %d utterances in %.1f s (deadline "
             "%.0f s), published %s", base.candidate_rows, len(covered),
             base.elapsed_seconds, deadline_seconds, target)
    return base


def _child_main(argv: list[str] | None = None) -> int:
    """Align the requested manifest chunk by chunk. Never used by the parent."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen production (child process)")
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))

    from ...nat5h.coordinates import EncoderGeometry
    from ...nat5h.schema import RunIdentity

    cfg = request["cfg"]
    manifest = pd.read_parquet(request["manifest"])
    out_dir = Path(request["out_dir"])
    size = int(request.get("chunk_size", DEFAULT_CHUNK_SIZE))
    language = str(request.get("language", PROBE_LANGUAGE))
    diagnostics_dir = str(request.get("diagnostics_dir") or out_dir)
    qcfg = ((cfg.get("alignment") or {}).get("qwen") or {})

    result: dict[str, Any] = {"state": "blocked", "chunks_written": 0,
                             "language": language}
    adapter = None
    try:
        adapter_cfg = dict(cfg)
        adapter_cfg["qwen_aligner"] = {
            **(cfg.get("qwen_aligner") or {}),
            "model_id": qcfg.get("model_id"),
            "local_model_dir": qcfg.get("local_model_dir"),
            "dtype": qcfg.get("dtype", "bfloat16"),
            "device": (cfg.get("model") or {}).get("device", "cuda"),
        }
        adapter = _adapter_class()(adapter_cfg)
        adapter.load()
        result["package_version"] = getattr(adapter, "package_version", None)
        declared = list(adapter.supported_languages() or [])
        result["declared_languages"] = declared
        if language.lower() not in {str(x).lower() for x in declared}:
            result["reason"] = (f"language_unsupported: {language} is not in "
                                f"{declared}")
            _write(request, result)
            return 0

        geometry = EncoderGeometry.from_values()
        identity = RunIdentity.from_cfg(cfg)
        written = 0
        for index, (lo, hi) in enumerate(chunk_bounds(len(manifest), size)):
            chunk = manifest.iloc[lo:hi].reset_index(drop=True)
            table = adapter.run(chunk, geometry=geometry, language=language,
                                identity=identity,
                                diagnostics_dir=diagnostics_dir)
            if table is None or not len(table):
                # the adapter ran and produced nothing for a whole chunk: a
                # measurement, not infrastructure, and the parent must not
                # publish a table with a hole in it
                result["state"] = "completed_no_go"
                result["reason"] = f"chunk_{index}_returned_no_candidates"
                _write(request, result)
                return 0
            # written whole, then renamed: a chunk file the parent can see is a
            # chunk that finished
            tmp = out_dir / f".chunk_{index:05d}.parquet.tmp"
            table.to_parquet(tmp, index=False)
            os.replace(tmp, out_dir / f"chunk_{index:05d}.parquet")
            written += 1
            result["chunks_written"] = written
            (out_dir / "progress.json").write_text(
                json.dumps({"chunks_written": written, "utterances": int(hi)}),
                encoding="utf-8")
        result["state"] = "ok"
    except BaseException as exc:                            # noqa: BLE001
        result["state"] = "blocked" if isinstance(exc, INFRASTRUCTURE_ERRORS) else "failed"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=20)
        if not isinstance(exc, Exception):                  # pragma: no cover
            _write(request, result)
            raise
    finally:
        _release_gpu(adapter)
    _write(request, result)
    return 0


def _write(request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    Path(request["result"]).write_text(json.dumps(dict(result), default=str),
                                      encoding="utf-8")


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(_child_main())
