"""Content hashing for manifests, audio and configs (reproducibility logging)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, max_bytes: int | None = None) -> str:
    """Hash a file. If ``max_bytes`` is given only that prefix is hashed.

    The prefix mode exists because hashing 21 GB of audio on every manifest
    build is wasteful; the size is mixed in so truncation still detects changes.
    """
    path = Path(path)
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        remaining = max_bytes if max_bytes is not None else -1
        while True:
            chunk_size = 1 << 20 if remaining < 0 else min(1 << 20, remaining)
            if chunk_size == 0:
                break
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining > 0:
                remaining -= len(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_bytes(payload.encode("utf-8"))


def sha256_strings(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def manifest_hash(df, columns: Iterable[str] = ("utterance_id",)) -> str:
    """Stable hash of a manifest's identity columns (order-independent)."""
    cols = [c for c in columns if c in df.columns]
    values = sorted("|".join(str(r[c]) for c in cols) for _, r in df[cols].iterrows())
    return sha256_strings(values)
