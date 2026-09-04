"""Result checkpointing and resume.

One JSON row per cell in `out/results_A.jsonl` / `out/results_B.jsonl`, written
after every cell so an interrupted allocation loses at most one cell. `--resume`
skips cells whose config hash is already present.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from . import config as C


def git_commit(repo: Path = C.REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty(repo: Path = C.REPO_ROOT) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True


def environment() -> dict[str, Any]:
    import torch
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


class ResultStore:
    """Append-only JSONL with a config-hash index."""

    def __init__(self, path: Path, *, track: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.track = track
        self.done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("config_hash"):
                    self.done[row["config_hash"]] = row
        self._stamp = {
            "track": track,
            "taint": C.TAINT,
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "environment": environment(),
            "seeds": C.SEEDS,
        }

    def has(self, config_hash: str) -> bool:
        return config_hash in self.done

    def get(self, config_hash: str) -> dict[str, Any] | None:
        return self.done.get(config_hash)

    def rows(self) -> list[dict[str, Any]]:
        return list(self.done.values())

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = {**self._stamp, "written_at": time.time(), **row}
        payload["development_only_diagnostic"] = True
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        if payload.get("config_hash"):
            self.done[payload["config_hash"]] = payload
        return payload


class Budget:
    """Wall-clock budget with a clean stop."""

    def __init__(self, hours: float | None):
        # `None` means unlimited; 0.0 means "already exhausted". Treating a
        # zero budget as unlimited would silently run the whole programme.
        self.hours = None if hours is None else float(hours)
        self.started = time.monotonic()

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started) / 3600.0

    def exhausted(self) -> bool:
        return bool(self.hours is not None and self.elapsed_hours >= self.hours)

    def remaining_hours(self) -> float:
        return float("inf") if self.hours is None else max(
            0.0, self.hours - self.elapsed_hours)
