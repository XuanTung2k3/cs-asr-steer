"""Frozen A6 decode configurations and cache-safe hypothesis utilities."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..utils.hashing import sha256_obj


def decode_config(model: str, mode: str) -> dict[str, Any]:
    if model == "whisper":
        if mode not in {"greedy", "official_standard"}: raise ValueError(mode)
        return {"backend": "transformers_whisper", "do_sample": False, "temperature": 0.0,
                "num_beams": 1 if mode == "greedy" else 5, "task": "transcribe", "language": "zh",
                "condition_on_prev_tokens": False, "max_new_tokens": 200,
                "return_timestamps": False, "mode": mode}
    if model == "qwen3_asr_1p7b":
        if mode not in {"greedy", "official_standard"}: raise ValueError(mode)
        return {"backend": "transformers_qwen3_asr", "do_sample": False, "temperature": 1e-6,
                "num_beams": 1, "force_language": "Chinese", "context": "", "max_new_tokens": 200,
                "eos_token_ids": [151643, 151645], "pad_token_id": 151643,
                "mode": mode, "provenance_equivalent_to": "greedy" if mode == "official_standard" else None}
    raise ValueError(model)


def decode_config_hash(model: str, mode: str) -> str:
    return "sha256:" + sha256_obj(decode_config(model, mode))


def register_hypothesis(*, model: str, dataset: str, mode: str, hypothesis: list[int],
                        baseline_hash: str | None = None) -> dict[str, Any]:
    value = dict(model=model, dataset=dataset, decode_mode=mode,
                 config=decode_config(model, mode), config_hash=decode_config_hash(model, mode),
                 hypothesis=hypothesis)
    value["hypothesis_hash"] = baseline_hash or ("sha256:" + sha256_obj(hypothesis))
    return value


def replicate_hypotheses(hypothesis: list[int], num_hypotheses: int) -> list[list[int]]:
    if num_hypotheses < 1: raise ValueError(num_hypotheses)
    return [list(hypothesis) for _ in range(num_hypotheses)]


def reorder_cache(cache: Mapping[str, Any], beam_indices: list[int]) -> dict[str, Any]:
    """Reorder batch-leading cache entries for beam search without mutation."""
    out = {}
    for key, value in cache.items():
        if isinstance(value, list): out[key] = [deepcopy(value[i]) for i in beam_indices]
        else: out[key] = value
    return out


def local_mask_indices(mask: list[bool], *, batch_index: int = 0, hypothesis_index: int = 0) -> list[tuple[int, int, int]]:
    return [(batch_index, hypothesis_index, i) for i, selected in enumerate(mask) if selected]
