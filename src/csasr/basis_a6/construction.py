"""Model-backend boundary for corpus-level A6-F direction construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .directions import build_fixed_source, direction_hash
from .fixed import fixed_direction_key


@dataclass(frozen=True)
class ConstructionSample:
    utterance_id: str
    group_a: np.ndarray
    group_b: np.ndarray
    conditioning_all: np.ndarray | None = None
    conditioning_cs: np.ndarray | None = None


def construct_corpus_layer(*, source: str, model: str, side: str, layer: int,
                           samples: Iterable[ConstructionSample]) -> dict:
    """Aggregate every frozen construction sample exactly once for one layer.

    The model-specific capture callback is outside this pure function. It must
    provide only the current sample's accepted group states; this function
    never reads evaluation panels or another source.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("empty frozen construction sample set")
    a = [np.asarray(s.group_a, dtype=np.float64) for s in samples]
    b = [np.asarray(s.group_b, dtype=np.float64) for s in samples]
    all_deltas = [np.asarray(s.conditioning_all, dtype=np.float64) for s in samples if s.conditioning_all is not None]
    cs_deltas = [np.asarray(s.conditioning_cs, dtype=np.float64) for s in samples if s.conditioning_cs is not None]
    built = build_fixed_source(groups_a=a, groups_b=b,
                               conditioning_all=np.concatenate(all_deltas) if all_deltas else None,
                               conditioning_cs=np.concatenate(cs_deltas) if cs_deltas else None)
    directions = built["directions"]
    records = []
    for method, vector in directions.items():
        records.append({"key": fixed_direction_key(source=source, model=model, side=side, layer=layer, method=method),
                        "direction_hash": direction_hash(vector), "n_A": built["n_A"], "n_B": built["n_B"],
                        "construction_utterance_ids": sorted(s.utterance_id for s in samples)})
    return {"source": source, "model": model, "side": side, "layer": int(layer),
            "directions": directions, "records": records, "N_utterances": len(samples),
            "n_A": built["n_A"], "n_B": built["n_B"]}
