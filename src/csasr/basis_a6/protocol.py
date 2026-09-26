"""Frozen BASIS-A6 constants and configuration-level enumeration."""
from __future__ import annotations

from itertools import product
from typing import Iterator

MODELS = {
    "whisper": {"encoder_layers": tuple(range(32)), "decoder_layers": tuple(range(32)), "encoder_dim": 1280, "decoder_dim": 1280},
    "qwen3_asr_1p7b": {"encoder_layers": tuple(range(24)), "decoder_layers": tuple(range(28)), "encoder_dim": 1024, "decoder_dim": 2048},
}
METHODS = ("raw", "add_unique", "minus_shared", "unique_minus_shared", "conditioning_cs", "conditioning_all")
NON_CONDITIONING = METHODS[:4]
CONDITIONING = METHODS[4:]
PANELS = ("cs_dialogue_dev_select", "seame_dev_man", "seame_dev_sge", "ascend_eval")
DECODE_MODES = ("greedy", "official_standard")
RHOS = (0.5, 1.0, 2.0, 4.0, 6.0)
SOURCES = ("cs_dialogue", "ascend")


def _sites(model: str) -> Iterator[tuple[str, int, str]]:
    spec = MODELS[model]
    for method in NON_CONDITIONING:
        for side in ("encoder", "decoder"):
            for layer in spec[f"{side}_layers"]:
                yield method, layer, side
    for method in CONDITIONING:
        for layer in spec["decoder_layers"]:
            yield method, layer, "decoder"


def enumerate_one_regime(*, branch: str, source: str | None = None) -> list[dict]:
    if branch not in {"fixed", "oracle_tt"}:
        raise ValueError(branch)
    if branch == "fixed" and source not in SOURCES:
        raise ValueError("A6-F requires one of the frozen construction sources")
    rows = []
    for model in MODELS:
        for method, layer, side in _sites(model):
            for rho, decode_mode, panel in product(RHOS, DECODE_MODES, PANELS):
                rows.append({"branch": branch, "construction_source": source,
                             "model": model, "dataset": panel, "side": side,
                             "layer": layer, "method": method, "rho": rho,
                             "decode_mode": decode_mode})
    return rows


def enumerate_a6_f() -> list[dict]:
    return [row for source in SOURCES for row in enumerate_one_regime(branch="fixed", source=source)]


def enumerate_a6_tt() -> list[dict]:
    return enumerate_one_regime(branch="oracle_tt")


def expected_counts() -> dict[str, int]:
    f = enumerate_a6_f(); tt = enumerate_a6_tt()
    return {"a6_f": len(f), "a6_tt": len(tt), "total": len(f) + len(tt), "baselines": len(MODELS) * len(PANELS) * len(DECODE_MODES),
            "one_regime": len(tt)}
