"""Model-agnostic A6 execution orchestration.

Concrete Whisper/Qwen model backends can implement the small callback protocol
without duplicating the scientific ordering. This module does not load models;
it makes the forbidden data flow impossible at the orchestration boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from .cache import DirectionCache
from .directions import build_method_directions, dynamic_rank
from .guards import make_gold_leakage_trace


@dataclass(frozen=True)
class OracleTTCallbacks:
    baseline_decode: Callable[[Any, str], Mapping[str, Any]]
    align_oracle_to_hypothesis: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    extract_analysis_states: Callable[[Any, Mapping[str, Any], str], Mapping[str, Any]]
    extract_conditioning_deltas: Callable[[Any, Mapping[str, Any], str], Mapping[str, Any]]
    steered_decode: Callable[[Any, Mapping[str, Any], Mapping[str, np.ndarray], float, str], Mapping[str, Any]]


def run_oracle_tt_sample(*, sample: Any, model: str, dataset: str, utterance_id: str,
                         decode_mode: str, side: str, layers: list[int], callbacks: OracleTTCallbacks,
                         cache: DirectionCache) -> dict[str, Any]:
    """Run phases A–E once and return reusable per-sample state.

    The callbacks receive the current ``sample`` on every phase. A callback
    cannot request a corpus direction here; only its own hypothesis/alignment
    and current-sample states are passed onward.
    """
    started = perf_counter()
    baseline = callbacks.baseline_decode(sample, decode_mode)                 # A
    alignment = callbacks.align_oracle_to_hypothesis(sample, baseline)         # B
    hypothesis_hash = str(baseline["hypothesis_hash"])
    alignment_hash = str(alignment["alignment_hash"])
    states = callbacks.extract_analysis_states(sample, baseline, side)         # C
    cond = callbacks.extract_conditioning_deltas(sample, baseline, side) if side == "decoder" else {}
    directions: dict[str, np.ndarray] = {}
    records = []
    for layer in layers:
        state = states[layer]
        group_a, group_b = np.asarray(state["A"]), np.asarray(state["B"])
        methods = build_method_directions(
            group_a, group_b,
            conditioning_deltas=None if side != "decoder" else np.asarray(cond[layer]["all"]),
            conditioning_cs_positions=None if side != "decoder" else np.asarray(cond[layer]["cs_positions"], dtype=int),
        )
        rank = dynamic_rank(len(group_a), len(group_b), group_a.shape[-1])
        raw_eligible = "raw" in methods
        us_eligible = rank >= 2
        for method, vector in methods.items():
            eligible = method == "raw" or us_eligible
            # Conditioning eligibility is represented by the callback's
            # non-empty selection; build_method_directions omits empty ones.
            cache.put(model=model, dataset=dataset, utterance_id=utterance_id,
                      decode_analysis_mode=decode_mode, side=side, layer=layer,
                      method=method, alignment_hash=alignment_hash, n_A=len(group_a),
                      n_B=len(group_b), rank=rank if method in {"add_unique", "minus_shared", "unique_minus_shared"} else None,
                      eligible=eligible, vector=vector)
            directions[f"{method}:{layer}"] = vector
            records.append({"layer": layer, "method": method, "n_A": len(group_a), "n_B": len(group_b),
                            "rank": rank, "raw_eligible": raw_eligible, "unique_shared_eligible": us_eligible,
                            "conditioning_eligible": method.startswith("conditioning")})
    trace = make_gold_leakage_trace(utterance_id=utterance_id, hypothesis_hash=hypothesis_hash,
                                    oracle_alignment_hash=alignment_hash,
                                    analysis_passes=int(1 + (2 if side == "decoder" else 0)))
    return {"baseline": baseline, "alignment": alignment, "directions": directions,
            "records": records, "gold_leakage_trace": trace,
            "construction_latency_sec": perf_counter() - started,
            "callbacks": callbacks}


def steer_cached_sample(state: Mapping[str, Any], *, sample: Any, callbacks: OracleTTCallbacks,
                        rho: float, method: str, side: str, layer: int, decode_mode: str) -> Mapping[str, Any]:
    key = f"{method}:{layer}"
    if key not in state["directions"]:
        return {"status": "INELIGIBLE", "method": method, "layer": layer, "rho": rho}
    return callbacks.steered_decode(sample, state["alignment"], {key: state["directions"][key]}, rho, decode_mode)
