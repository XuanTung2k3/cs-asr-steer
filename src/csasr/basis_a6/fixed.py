"""A6-F direction-source contract and read-only CS/A5 reuse."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .directions import build_fixed_source, direction_hash, normalize

REPO = Path(__file__).resolve().parents[3]
CS_SOURCE = "cs_dialogue"
ASCEND_SOURCE = "ascend"
METHOD_TO_A5_FILE = {"add_unique": "pos_unique.npy", "minus_shared": "neg_shared.npy",
                     "unique_minus_shared": "unique_minus_shared.npy"}


def fixed_direction_key(*, source: str, model: str, side: str, layer: int, method: str) -> tuple[str, str, str, int, str]:
    if source not in {CS_SOURCE, ASCEND_SOURCE}:
        raise ValueError(source)
    if method.startswith("conditioning") and side != "decoder":
        raise ValueError("conditioning directions are decoder-only")
    return source, model, side, int(layer), method


def _a4_path(model: str, side: str, layer: int, method: str) -> Path:
    if model == "whisper":
        if method == "conditioning_all":
            return REPO / "results/basis_frozen_layer_atlas/directions" / f"conditioning_L{layer}.npy"
        if side == "encoder":
            return REPO / "results/basis_a3_raw_cond_scope_depth/directions/raw_encoder" / f"raw_encoder_L{layer}.npy"
        return REPO / "results/basis_frozen_layer_atlas/directions" / f"raw_L{layer}.npy"
    if method == "conditioning_all":
        return REPO / f"results/basis_a4/qwen3_asr_1p7b/directions/conditioning_L{layer}.npy"
    name = "raw_encoder" if side == "encoder" else "raw_decoder"
    return REPO / f"results/basis_a4/qwen3_asr_1p7b/directions/{name}_L{layer}.npy"


def load_reusable_cs_direction(*, model: str, side: str, layer: int, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an accepted CS vector only after its A4/A5 namespace is resolved.

    This is intentionally read-only. Missing or ambiguous artifacts fail
    loudly; callers must construct the missing vector rather than substituting
    a legacy direction.
    """
    if method in METHOD_TO_A5_FILE:
        path = REPO / "results/basis_a5_unique_shared/directions" / model / side / f"L{layer:02d}" / METHOD_TO_A5_FILE[method]
        definition = "A5 uncentered second moments/principal angles/sign rules"
    elif method in {"raw", "conditioning_all"}:
        path = _a4_path(model, side, layer, method)
        definition = "A4 raw difference-in-means" if method == "raw" else "A4 all-content paired conditioning"
    else:
        raise ValueError(f"no accepted reusable CS artifact for {method}")
    if not path.is_file():
        raise FileNotFoundError(f"accepted CS vector missing: {path}")
    vec = normalize(np.load(path), name=method)
    return vec, {"source": CS_SOURCE, "model": model, "side": side, "layer": int(layer),
                 "method": method, "definition": definition, "path": str(path.relative_to(REPO)),
                 "direction_hash": direction_hash(vec), "reused_read_only": True}


def construct_ascend_direction(*, groups_a: list[np.ndarray], groups_b: list[np.ndarray],
                               conditioning_all: np.ndarray | None = None,
                               conditioning_cs: np.ndarray | None = None) -> dict[str, Any]:
    built = build_fixed_source(groups_a=groups_a, groups_b=groups_b,
                               conditioning_all=conditioning_all, conditioning_cs=conditioning_cs)
    return {"source": ASCEND_SOURCE, **built}
