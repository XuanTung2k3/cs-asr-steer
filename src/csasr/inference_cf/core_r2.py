"""Pure, reference-free P0-R2 repairability gate primitives.

This module contains no steering hook, reference alignment, or learned parameter.
Historical P0/P0-R1 modules remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

import numpy as np
import torch

from .core import digest

VERSION_R2 = "p0r2_repairability_v1"
EPS = 1e-12
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
WINDOW_FRAMES = 50
MAX_FRAMES = 1500
LATIN = re.compile(rb" ?'?[A-Za-z]+(?:['-][A-Za-z]+)*\Z")


def _is_han(ch: str) -> bool:
    n = ord(ch)
    return (0x3400 <= n <= 0x9FFF) or (0xF900 <= n <= 0xFAFF)


def token_bytes(tokenizer, token_id: int, byte_decoder: dict[str, int]) -> bytes:
    """Decode Whisper's reversible byte alphabet without replacement characters."""
    if int(token_id) in tokenizer.all_special_ids:
        raise ValueError("special token has no content bytes")
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return bytes(byte_decoder[c] for c in token)


def prefix_utf8_complete(tokenizer, content_prefix: Sequence[int], byte_decoder: dict[str, int]) -> bool:
    try:
        b = b"".join(token_bytes(tokenizer, i, byte_decoder) for i in content_prefix)
        b.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError, KeyError):
        return False
    return True


def tokenizer_partition(tokenizer) -> dict:
    """Freeze disjoint Han/Han-lead and ASCII-Latin vocabulary ID sets."""
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    matrix, embedded, ambiguous = [], [], []
    special = set(tokenizer.all_special_ids)
    for token_id in range(len(tokenizer)):
        if token_id in special:
            ambiguous.append(token_id)
            continue
        b = token_bytes(tokenizer, token_id, decoder)
        if LATIN.fullmatch(b):
            embedded.append(token_id)
            continue
        try:
            s = b.decode("utf-8", errors="strict").lstrip(" ")
            if s and all(_is_han(c) or c in "，。？！；：、" for c in s) and any(_is_han(c) for c in s):
                matrix.append(token_id)
                continue
        except UnicodeDecodeError as exc:
            # A leading Han UTF-8 byte fragment is M-like even before completion.
            lead = b.lstrip(b" ")
            if exc.reason == "unexpected end of data" and lead and 0xE4 <= lead[0] <= 0xE9:
                matrix.append(token_id)
                continue
        ambiguous.append(token_id)
    if len(matrix) + len(embedded) + len(ambiguous) != len(tokenizer):
        raise AssertionError("partition is not exhaustive")
    payload = {"version": "whisper_han_ascii_v1", "matrix_ids": matrix,
               "embedded_ids": embedded, "ambiguous_ids": ambiguous}
    return {**payload, "hash": digest(payload)}


@dataclass(frozen=True)
class Window:
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    attention_mass: float
    point_argmax_frame: int
    point_start_sample: int
    point_end_sample: int
    heard_samples: int

    def record(self) -> dict:
        return {**asdict(self), "duration_sec": (self.end_sample - self.start_sample) / SAMPLE_RATE,
                "start_sec": self.start_sample / SAMPLE_RATE, "end_sec": self.end_sample / SAMPLE_RATE,
                "center_sec": (self.start_sample + self.end_sample) / (2 * SAMPLE_RATE),
                "point_center_sec": (self.point_start_sample + self.point_end_sample) / (2 * SAMPLE_RATE)}


def max_attention_window(head_attention: np.ndarray, heard_samples: int) -> Window:
    """Maximum 50-frame mass; earliest tie; point argmax is diagnostic only.

    Input is current-query attention with shape [frozen alignment heads, encoder frames].
    No other decoder query or reference time enters this function.
    """
    if heard_samples <= 0:
        raise ValueError("localizer_fail:empty_audio")
    a = np.asarray(head_attention, dtype=np.float64)
    valid = min(MAX_FRAMES, (int(heard_samples) + FRAME_SAMPLES - 1) // FRAME_SAMPLES)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < valid:
        raise ValueError("localizer_fail:attention_shape")
    a = a[:, :valid]
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("localizer_fail:invalid_attention")
    mean = a.mean(axis=0)
    total = float(mean.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("localizer_fail:zero_attention")
    mean /= total
    point = int(np.argmax(mean))
    width = min(WINDOW_FRAMES, valid)
    masses = np.convolve(mean, np.ones(width, dtype=np.float64), mode="valid")
    start = int(np.argmax(masses))
    end = start + width
    heard = min(int(heard_samples), MAX_FRAMES * FRAME_SAMPLES)
    size = min(WINDOW_FRAMES * FRAME_SAMPLES, heard)
    sample_start = min(start * FRAME_SAMPLES, heard - size)
    sample_end = sample_start + size
    point_start = min(max(0, point * FRAME_SAMPLES - size // 2), heard - size)
    return Window(start, end, sample_start, sample_end, float(masses[start]), point,
                  point_start, point_start + size, heard)


def local_support(pi_e: float, pi_m: float, null_e: float, null_m: float) -> dict:
    vals = [float(x) for x in (pi_e, pi_m, null_e, null_m)]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in vals):
        raise ValueError("nonfinite_signal:language_posterior")
    pe, pm, ne, nm = vals
    odds = math.log((pe + EPS) / (pm + EPS))
    null_odds = math.log((ne + EPS) / (nm + EPS))
    margin = odds - null_odds
    return {"pi_e": pe, "pi_m": pm, "pair_mass": pe + pm,
            "l_t": odds, "l_null": null_odds, "A": margin,
            "sigmoid_A": 1 / (1 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1 + math.exp(margin)),
            "E": max(0.0, math.tanh(margin / 2))}


def conflict_from_logits(logits: torch.Tensor, partition: dict) -> dict:
    """BC-B on a full-vocabulary next-token distribution, float32 log-softmax."""
    if logits.ndim != 1 or logits.numel() != sum(len(partition[k]) for k in
                                                   ("matrix_ids", "embedded_ids", "ambiguous_ids")):
        raise ValueError("baseline_provider_fail:vocabulary_shape")
    scores = logits.detach().float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite_signal:logits")
    logp = torch.log_softmax(scores, dim=-1)
    def mass(ids):
        return float(torch.exp(torch.logsumexp(logp[ids], dim=0))) if ids else 0.0
    pm = mass(partition["matrix_ids"])
    pe = mass(partition["embedded_ids"])
    q = pm + pe
    r = max(0.0, pm - pe)
    if not all(math.isfinite(x) for x in (pm, pe, q, r)) or not (0 <= r <= q <= 1.000001):
        raise ValueError("nonfinite_signal:conflict")
    return {"P_M": pm, "P_E": pe, "Q": q, "R": r,
            "ambiguous_mass": max(0.0, 1.0 - q),
            "conditional_margin": max(0.0, (pm - pe) / (q + 2 * EPS)),
            "log_odds": math.log((pm + EPS) / (pe + EPS))}


def gate_values(support: dict, baseline: dict, ecf: dict) -> dict:
    e, rb, re = (float(support["E"]), float(baseline["R"]), float(ecf["R"]))
    if not all(math.isfinite(x) for x in (e, rb, re)) or not (0 <= e <= 1 and 0 <= rb <= 1 and 0 <= re <= 1):
        raise ValueError("nonfinite_signal:gate_input")
    d = max(0.0, rb - re)
    return {"D": d, "g_old": e * rb, "g_cf": e * d}


def terminated_by_eos(sequence: Sequence[int], content: Sequence[int], eos_id: int,
                      max_new_tokens: int) -> bool:
    """Whether greedy decoding stopped on EOS (so an EOS gap slot exists).

    The installed Whisper ``generate`` strips a final EOS from its returned sequence, so
    ``eos in sequence`` is not sufficient. Short-form greedy decoding stops only on EOS or at
    ``max_new_tokens``; a content length below the cap therefore implies an emitted EOS.
    """
    return int(eos_id) in list(sequence) or len(content) < int(max_new_tokens)


def same_prefix_inputs(conditions: dict, content_prefix: Sequence[int]) -> tuple[list[int], list[int]]:
    b, e = list(conditions["cB"]), list(conditions["cE"])
    if not b or len(b) != len(e) or any(b[i] != e[i] for i in range(len(b)) if i != 1):
        raise ValueError("conditions must differ only in language token")
    if b[1] == e[1] or any(not isinstance(x, int) for x in content_prefix):
        raise ValueError("invalid language condition or content prefix")
    return b + list(content_prefix), e + list(content_prefix)


def _check_branch(branch: dict) -> None:
    if not all(math.isfinite(branch[k]) for k in ("P_M", "P_E", "Q", "R")):
        raise ValueError("nonfinite conflict component")
    if not 0 <= branch["P_M"] <= 1 or not 0 <= branch["P_E"] <= 1:
        raise ValueError("probability mass bound")
    if not math.isclose(branch["Q"], branch["P_M"] + branch["P_E"], abs_tol=1e-6):
        raise ValueError("classified mass identity")
    if not math.isclose(branch["R"], max(0., branch["P_M"] - branch["P_E"]), abs_tol=1e-6):
        raise ValueError("conflict formula")
    if not 0 <= branch["R"] <= branch["Q"] + 1e-6:
        raise ValueError("conflict bound")


def validate_gate_row(row: dict) -> None:
    required = ("schema", "utterance_id", "logical_position", "prefix_ids", "fallback_reason",
                "baseline", "ecf", "gate", "window", "manifest_hash")
    if row.get("schema") != VERSION_R2 or any(k not in row for k in required):
        raise ValueError("invalid R2 row schema")
    # Conflict invariants hold on every computed branch, including structural-fallback rows.
    for branch in (row["baseline"], row["ecf"]):
        if branch is not None:
            _check_branch(branch)
    diag = row.get("xcf")
    if diag is not None:
        # Wrong-language diagnostic: never enters either gate.
        _check_branch(diag["conflict"])
        if row["baseline"] is not None and not math.isclose(
                diag["D_X"], max(0., row["baseline"]["R"] - diag["conflict"]["R"]), abs_tol=1e-6):
            raise ValueError("wrong-language repairability formula")
    if row["fallback_reason"] is not None and row.get("eligibility_status") == "cf_ineligible":
        # Ecf-branch failure only: the counterfactual gate is zero, the old gate is preserved.
        g = row["gate"]
        if row["fallback_reason"] != "ecf_provider_fail" or g["D"] != 0.0 or g["g_cf"] != 0.0:
            raise ValueError("Ecf-only fallback must zero D and g_cf")
        if row["local_support"] is None or row["baseline"] is None or not math.isclose(
                g["g_old"], row["local_support"]["E"] * row["baseline"]["R"], abs_tol=1e-6):
            raise ValueError("Ecf-only fallback must preserve g_old = E*R_B")
    elif row["fallback_reason"] is not None:
        if row["gate"] != {"D": 0.0, "g_old": 0.0, "g_cf": 0.0}:
            raise ValueError("fallback must zero both gates")
    else:
        b, e, g = row["baseline"], row["ecf"], row["gate"]
        if not math.isclose(g["D"], max(0., b["R"] - e["R"]), abs_tol=1e-6):
            raise ValueError("repairability formula")
        if not 0 <= g["D"] <= b["R"] + 1e-6:
            raise ValueError("repairability bound")
        if not math.isclose(g["g_cf"], row["local_support"]["E"] * g["D"], abs_tol=1e-6):
            raise ValueError("gate formula")
        if not math.isclose(g["g_old"], row["local_support"]["E"] * b["R"], abs_tol=1e-6):
            raise ValueError("old gate formula")
        if not 0 <= g["g_cf"] <= g["g_old"] + 1e-6:
            raise ValueError("pointwise non-increase")
