"""Persistent model-cache validation and locked downloads for NAT5H."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .statusing import atomic_write_json, utc_now

QWEN_REQUIRED_CONFIGS = ("config.json",)
QWEN_WEIGHT_PATTERNS = ("model.safetensors", "pytorch_model.bin", ".safetensors.index.json")


@dataclass(frozen=True)
class FileMetadata:
    path: str
    size: int
    sha256: str


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def checkpoint_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def qwen_checkpoint_complete(path: str | Path) -> tuple[bool, list[str]]:
    root = Path(path)
    missing = []
    for name in QWEN_REQUIRED_CONFIGS:
        if not (root / name).is_file():
            missing.append(name)
    files = checkpoint_files(root)
    has_weight = any(
        p.name in {"model.safetensors", "pytorch_model.bin"}
        or p.name.endswith(".safetensors")
        or p.name.endswith(".bin")
        or p.name.endswith(".safetensors.index.json")
        for p in files
    )
    if not has_weight:
        missing.append("weight file or weight index")
    return (len(missing) == 0), missing


def collect_file_metadata(path: str | Path, max_files: int = 64) -> list[dict[str, Any]]:
    root = Path(path)
    rows = []
    for p in checkpoint_files(root)[:max_files]:
        rel = str(p.relative_to(root))
        rows.append(asdict(FileMetadata(path=rel, size=p.stat().st_size, sha256=sha256_file(p))))
    return rows


class FileLock:
    """Small filesystem lock based on atomic O_EXCL creation."""

    def __init__(self, path: str | Path, timeout_seconds: int = 7200, poll_seconds: float = 10.0):
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()} created_at={utc_now()}\n".encode("utf-8"))
                return self
            except FileExistsError:
                if time.monotonic() - start > self.timeout_seconds:
                    raise TimeoutError(f"timed out waiting for model download lock: {self.path}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def read_completion_marker(marker_path: str | Path) -> dict[str, Any] | None:
    path = Path(marker_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def ensure_qwen_checkpoint(
    *,
    model_id: str,
    local_model_dir: str | Path,
    cache_dir: str | Path,
    marker_path: str | Path | None = None,
    allow_download: bool = True,
    lock_timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Ensure Qwen forced-aligner weights exist in a persistent local path.

    No Python packages are installed or upgraded here.  If software imports are
    incompatible, callers should report that during preflight.
    """
    local_dir = Path(local_model_dir)
    cache_dir = Path(cache_dir)
    marker = Path(marker_path) if marker_path else local_dir / ".nat5h_complete.json"
    complete, missing = qwen_checkpoint_complete(local_dir)
    marker_payload = read_completion_marker(marker)
    if complete and marker_payload:
        return {
            "state": "present",
            "model_id": model_id,
            "local_model_dir": str(local_dir),
            "marker": str(marker),
            "metadata": marker_payload,
        }

    if complete:
        payload = {
            "model_id": model_id,
            "local_model_dir": str(local_dir),
            "created_at": utc_now(),
            "files": collect_file_metadata(local_dir),
            "downloaded": False,
        }
        atomic_write_json(marker, payload)
        return {
            "state": "present_marker_created",
            "model_id": model_id,
            "local_model_dir": str(local_dir),
            "marker": str(marker),
            "metadata": payload,
        }

    if not allow_download:
        raise FileNotFoundError(
            f"Qwen checkpoint incomplete at {local_dir}; missing {missing}; download disabled"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = local_dir.parent / f".{local_dir.name}.download.lock"
    with FileLock(lock_path, timeout_seconds=lock_timeout_seconds):
        complete, missing = qwen_checkpoint_complete(local_dir)
        if not complete:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        complete, missing = qwen_checkpoint_complete(local_dir)
        if not complete:
            raise FileNotFoundError(
                f"Qwen checkpoint still incomplete at {local_dir}; missing {missing}"
            )
        payload = {
            "model_id": model_id,
            "local_model_dir": str(local_dir),
            "created_at": utc_now(),
            "files": collect_file_metadata(local_dir),
            "downloaded": True,
        }
        atomic_write_json(marker, payload)
    return {
        "state": "downloaded",
        "model_id": model_id,
        "local_model_dir": str(local_dir),
        "marker": str(marker),
        "metadata": read_completion_marker(marker),
    }
