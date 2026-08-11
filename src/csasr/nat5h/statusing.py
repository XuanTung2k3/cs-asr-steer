"""Atomic stage status and artifact helpers for NAT5H."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

TERMINAL_STATES = {
    "passed",
    "completed_no_go",
    "insufficient_data",
    "technical_failed",
    "blocked",
    "partial_completed",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


@contextmanager
def atomic_output_path(path: str | Path, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a temporary sibling path and atomically publish it on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}{suffix}")
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"temporary artifact was not written: {tmp}")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


class StageTimer:
    def __init__(self) -> None:
        self.started_at = utc_now()
        self.started = time.monotonic()

    def elapsed_seconds(self) -> float:
        return float(time.monotonic() - self.started)


def status_path(output_root: str | Path, stage: str) -> Path:
    return Path(output_root) / "status" / f"{stage}.json"


def read_status(output_root: str | Path, stage: str) -> dict[str, Any] | None:
    path = status_path(output_root, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def terminal_state(output_root: str | Path, stage: str) -> str | None:
    status = read_status(output_root, stage)
    if not status:
        return None
    state = status.get("state")
    return state if state in TERMINAL_STATES else None


def stage_status_payload(
    *,
    stage: str,
    state: str,
    timer: StageTimer | None = None,
    started_at: str | None = None,
    config_hash: str | None = None,
    code_commit: str | None = None,
    input_manifest_hashes: dict[str, str] | None = None,
    output_paths: list[str] | None = None,
    gate_evidence: dict[str, Any] | None = None,
    next_permitted_stage: str | None = None,
    failure_or_no_go_reason: str | None = None,
    exploratory: bool = True,
) -> dict[str, Any]:
    finished_at = utc_now()
    return {
        "stage": stage,
        "state": state,
        "started_at": timer.started_at if timer else (started_at or finished_at),
        "finished_at": finished_at,
        "elapsed_seconds": timer.elapsed_seconds() if timer else 0.0,
        "config_hash": config_hash,
        "code_commit": code_commit,
        "input_manifest_hashes": input_manifest_hashes or {},
        "output_paths": output_paths or [],
        "gate_evidence": gate_evidence or {},
        "next_permitted_stage": next_permitted_stage,
        "failure_or_no_go_reason": failure_or_no_go_reason,
        "exploratory": bool(exploratory),
    }


def write_stage_status(output_root: str | Path, payload: dict[str, Any]) -> Path:
    stage = str(payload["stage"])
    path = status_path(output_root, stage)
    atomic_write_json(path, payload)
    write_overview(output_root)
    return path


def write_overview(output_root: str | Path) -> Path:
    root = Path(output_root)
    statuses = {}
    for path in sorted((root / "status").glob("n*.json")):
        if path.name == "overview.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"state": "technical_failed", "read_error": repr(exc)}
        statuses[path.stem] = payload
    overview = {
        "experiment": "natural_consensus_5h",
        "exploratory": True,
        "updated_at": utc_now(),
        "statuses": statuses,
    }
    path = root / "status" / "overview.json"
    atomic_write_json(path, overview)
    return path
