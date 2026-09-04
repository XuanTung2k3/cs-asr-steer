"""Explicit Round-2 lock serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .rounds import atomic_json, stable_hash


def lock_round1(source_root: Path, output: Path, *, selected: dict, rationale: str) -> dict:
    if not selected or not rationale:
        raise ValueError("--lock requires an explicit selection and rationale")
    source_files = sorted(p for p in Path(source_root).rglob("*") if p.is_file())
    hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files}
    payload = {"selected": selected, "selection_rationale": rationale,
               "source_root": str(source_root), "source_result_hashes": hashes,
               "configuration_hash": stable_hash(selected), "locked": True}
    atomic_json(output, payload)
    return payload
