"""Hard acceptance guards for A6 oracle information and local steering."""
from __future__ import annotations

from typing import Any, Iterable


class GoldTokenLeakageError(RuntimeError):
    pass


def validate_decoder_analysis_provenance(trace: dict[str, Any]) -> None:
    required = {
        "sequence_source": "baseline_hypothesis",
        "gold_token_hidden_states": False,
        "gold_next_token_logits": False,
        "target_embedding_direction": False,
    }
    for key, expected in required.items():
        if trace.get(key) != expected:
            raise GoldTokenLeakageError(f"decoder analysis violates gold-token guard: {key}={trace.get(key)!r}")
    if trace.get("oracle_information") not in {"location_only", "location_only_alignment"}:
            raise GoldTokenLeakageError("oracle information must be location-only")


def assert_no_external_fixed_direction(direction_source: str | None) -> None:
    """A6-TT must never load an A6-F corpus direction."""
    if direction_source in {"cs_dialogue", "ascend", "A6-F", "fixed"}:
        raise AssertionError("A6-TT accessed an external A6-F corpus direction")


def make_gold_leakage_trace(*, utterance_id: str, hypothesis_hash: str,
                            oracle_alignment_hash: str, analysis_passes: int = 1) -> dict[str, Any]:
    trace = {"schema_version": "basis_a6_gold_leakage_trace_v1", "utterance_id": utterance_id,
             "sequence_source": "baseline_hypothesis", "hypothesis_hash": hypothesis_hash,
             "oracle_alignment_hash": oracle_alignment_hash, "oracle_information": "location_only_alignment",
             "gold_token_hidden_states": False, "gold_next_token_logits": False,
             "target_embedding_direction": False, "analysis_passes": int(analysis_passes)}
    validate_decoder_analysis_provenance(trace)
    return trace


def validate_local_mask(mask: Iterable[int] | Iterable[bool], allowed: Iterable[int],
                        forbidden: Iterable[int] = ()) -> None:
    selected = {int(i) for i, value in enumerate(mask) if bool(value)}
    allowed_set, forbidden_set = set(map(int, allowed)), set(map(int, forbidden))
    if not selected <= allowed_set:
        raise AssertionError(f"local mask edits outside oracle-local set: {sorted(selected - allowed_set)[:5]}")
    if selected & forbidden_set:
        raise AssertionError(f"local mask edits forbidden positions: {sorted(selected & forbidden_set)[:5]}")


def rho_zero_identity(*, baseline: Any, steered: Any, rho: float) -> None:
    if float(rho) == 0.0 and baseline != steered:
        raise AssertionError("rho=0 failed exact baseline identity")
