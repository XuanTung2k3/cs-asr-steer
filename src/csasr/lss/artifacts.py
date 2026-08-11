"""Atomic artifact writes and content-addressed references.

Every artifact a stage produces is referenced downstream by hash, not by path,
so a stage cannot silently consume a file that was rewritten after the
producing stage passed its gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..nat5h.statusing import atomic_output_path, atomic_write_json
from ..utils.hashing import sha256_file, sha256_obj


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a parquet atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    return path


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def write_report(path: str | Path, title: str,
                 sections: Sequence[tuple[str, str]]) -> Path:
    """Markdown report in the same shape every other stage produces."""
    from ..experiments._common import save_report      # lazy: avoids a cycle

    return save_report(path, title, list(sections))


def artifact_ref(path: str | Path, *, schema: str | None = None,
                 rows: int | None = None) -> dict[str, Any]:
    """Hash-addressed reference to one artifact, for freeze files and statuses."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot reference a missing artifact: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": None if rows is None else int(rows),
        "schema": schema,
    }


def artifact_refs(paths: Sequence[str | Path], **kwargs: Any) -> list[dict[str, Any]]:
    return [artifact_ref(p, **kwargs) for p in paths]


def verify_refs(refs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Re-hash every referenced artifact. Returns a per-path verdict."""
    results = []
    for ref in refs:
        path = Path(ref["path"])
        if not path.is_file():
            results.append({"path": str(path), "ok": False, "reason": "missing"})
            continue
        actual = sha256_file(path)
        results.append({
            "path": str(path),
            "ok": actual == ref.get("sha256"),
            "reason": "" if actual == ref.get("sha256") else "sha256_mismatch",
            "expected_sha256": ref.get("sha256"),
            "actual_sha256": actual,
        })
    return {"all_ok": all(r["ok"] for r in results), "results": results}


def read_frozen(path: str | Path, expected_sha: str) -> pd.DataFrame:
    """Read a parquet only if it still hashes to what the producer recorded."""
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(
            f"frozen artifact changed since it was produced: {path}\n"
            f"  expected sha256 {expected_sha}\n  actual   sha256 {actual}")
    return pd.read_parquet(path)


def payload_hash(payload: Any) -> str:
    return sha256_obj(payload)
