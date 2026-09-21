"""Deterministic A6-TT per-sample direction cache and bundle hashing."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils.hashing import sha256_bytes, sha256_obj
from .directions import direction_hash


@dataclass(frozen=True)
class DirectionRecord:
    model: str
    dataset: str
    utterance_id: str
    decode_analysis_mode: str
    side: str
    layer: int
    method: str
    alignment_hash: str
    n_A: int
    n_B: int
    rank: int | None
    eligible: bool
    direction_hash: str
    vector: np.ndarray

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("vector", None)
        return value


def cache_key(*, model: str, dataset: str, utterance_id: str, decode_analysis_mode: str,
              side: str, layer: int, method: str, alignment_hash: str) -> str:
    payload = {"model": model, "dataset": dataset, "utterance_id": utterance_id,
               "decode_analysis_mode": decode_analysis_mode, "side": side,
               "layer": int(layer), "method": method, "alignment_hash": alignment_hash}
    return sha256_obj(payload)


class DirectionCache:
    def __init__(self) -> None:
        self._records: dict[str, DirectionRecord] = {}

    def put(self, *, model: str, dataset: str, utterance_id: str, decode_analysis_mode: str,
            side: str, layer: int, method: str, alignment_hash: str, n_A: int, n_B: int,
            rank: int | None, eligible: bool, vector: np.ndarray) -> DirectionRecord:
        arr = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
        if eligible and (arr.ndim != 1 or not np.isfinite(arr).all() or not np.isclose(np.linalg.norm(arr), 1.0, atol=2e-5)):
            raise ValueError("eligible cached directions must be finite unit vectors")
        rec = DirectionRecord(model, dataset, utterance_id, decode_analysis_mode, side, int(layer),
                              method, alignment_hash, int(n_A), int(n_B), None if rank is None else int(rank),
                              bool(eligible), direction_hash(arr), arr)
        key = cache_key(model=model, dataset=dataset, utterance_id=utterance_id,
                        decode_analysis_mode=decode_analysis_mode, side=side, layer=layer,
                        method=method, alignment_hash=alignment_hash)
        if key in self._records and self._records[key].direction_hash != rec.direction_hash:
            raise ValueError("direction cache key collision with different vector")
        self._records[key] = rec
        return rec

    def records(self) -> list[DirectionRecord]:
        return sorted(self._records.values(), key=lambda r: (r.utterance_id, r.model, r.dataset,
                          r.decode_analysis_mode, r.side, r.layer, r.method))

    def bundle_hash(self, *, model: str, dataset: str, side: str, method: str,
                    decode_analysis_mode: str) -> str:
        rows = [r for r in self.records() if (r.model, r.dataset, r.side, r.method,
                r.decode_analysis_mode) == (model, dataset, side, method, decode_analysis_mode)]
        payload = [{"utterance_id": r.utterance_id, "direction_hash": r.direction_hash,
                    "eligible": r.eligible} for r in rows]
        return "sha256:" + sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

    def write(self, root: str | Path) -> dict[str, Any]:
        root = Path(root); vectors = root / "vectors"; vectors.mkdir(parents=True, exist_ok=True)
        metadata = []
        for rec in self.records():
            name = rec.direction_hash.removeprefix("sha256:") + ".npy"
            np.save(vectors / name, rec.vector)
            metadata.append({**rec.metadata(), "vector_file": str(Path("vectors") / name)})
        bundles = {}
        for key in {(r.model, r.dataset, r.side, r.method, r.decode_analysis_mode) for r in self.records()}:
            bundles["|".join(map(str, key))] = self.bundle_hash(model=key[0], dataset=key[1], side=key[2], method=key[3], decode_analysis_mode=key[4])
        payload = {"schema_version": "basis_a6_tt_direction_cache_v1", "records": metadata, "bundles": bundles}
        (root / "DIRECTION_CACHE.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload
