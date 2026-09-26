"""Cross-source A6-F geometry within each model/side/layer."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .directions import direction_hash


def cross_source_geometry(directions: Mapping[tuple[str, str, str, int, str], np.ndarray]) -> list[dict[str, Any]]:
    """Compare ``(source, model, side, layer, method)`` vectors."""
    rows = []
    methods = ("raw", "add_unique", "minus_shared", "unique_minus_shared", "conditioning_cs", "conditioning_all")
    for model in sorted({k[1] for k in directions}):
        for side in ("encoder", "decoder"):
            layers = sorted({k[3] for k in directions if k[1] == model and k[2] == side})
            for layer in layers:
                for method in methods:
                    cs = directions.get(("cs_dialogue", model, side, layer, method))
                    ac = directions.get(("ascend", model, side, layer, method))
                    if cs is None or ac is None:
                        continue
                    if cs.shape != ac.shape:
                        raise ValueError("cross-source geometry mixed hidden dimensions")
                    rows.append({"model": model, "side": side, "layer": layer, "method": method,
                                 "cosine": float(np.dot(cs, ac) / (np.linalg.norm(cs) * np.linalg.norm(ac))),
                                 "cs_norm": float(np.linalg.norm(cs)), "ascend_norm": float(np.linalg.norm(ac)),
                                 "cs_hash": direction_hash(cs), "ascend_hash": direction_hash(ac)})
    return rows


def write_geometry_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "side", "layer", "method", "cosine", "cs_norm", "ascend_norm", "cs_hash", "ascend_hash"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
