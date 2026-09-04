"""Decoding for Track A: baselines and steered cells, with decode health.

One decode path serves both, so a steered run and its baseline differ only in
the hooks that were installed. `alpha=0` therefore has to reproduce the
baseline bit-identically, which is preflight check 1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as C
from .hooks import (SteeringPlan, SteeringSession, UtterancePlan,
                    energy_report, planned_positions)


@dataclass
class DecodeResult:
    """Per-utterance decode output plus the audit fields the gates need."""

    texts: dict[str, str]
    num_tokens: dict[str, int]
    hit_max_tokens: dict[str, bool]
    audit: dict[str, Any] = field(default_factory=dict)
    wall_clock_sec: float = 0.0

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "utterance_id": list(self.texts),
            "hypothesis": [self.texts[u] for u in self.texts],
            "num_tokens": [self.num_tokens.get(u, 0) for u in self.texts],
            "hit_max_tokens": [self.hit_max_tokens.get(u, False) for u in self.texts],
        })


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

def build_utterance_plans(bundle, pop, *, baseline_tokens: dict[str, int] | None,
                          utterance_ids: Sequence[str]) -> dict[str, UtterancePlan]:
    """One coverage plan per utterance.

    `valid_frames` is ceil(duration_sec * 50) clamped to 1500 -- requirement
    1.1, and the reason `valid_frame_ratio` is logged for every utterance.
    `planned_decoder_positions` comes from the BASELINE decode length, so |S|
    and therefore alpha_eff are fixed before the steered decode runs rather
    than discovered during it.
    """
    from . import data as D

    durations = D.duration_of(pop)
    frames: dict[str, list[tuple[int, int]]] = {}
    steps: dict[str, list[int]] = {}
    for utterance, group in pop.targets.groupby("utterance_id"):
        u = str(utterance)
        frames[u] = sorted({(int(lo), int(hi)) for lo, hi in
                            zip(group["span_frame_lo"], group["span_frame_hi"])})
        # the oracle decoder step of the earliest target, exactly as Day 5
        steps[u] = [max(0, int(group["token_index"].min()) - 1)]

    plans: dict[str, UtterancePlan] = {}
    for u in utterance_ids:
        u = str(u)
        duration = float(durations.get(u, 0.0))
        planned = int((baseline_tokens or {}).get(u, 0))
        plans[u] = UtterancePlan(
            utterance_id=u,
            duration_sec=duration,
            valid_frames=int(bundle.valid_frames(duration)),
            encoder_frames=frames.get(u, []),
            decoder_steps=steps.get(u, []),
            planned_decoder_positions=planned,
        )
    return plans


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def _generate_kwargs(*, language: str | None, num_beams: int,
                     max_new_tokens: int) -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": language,
        "do_sample": False,
        "num_beams": int(num_beams),
        "temperature": 0.0,
        "max_new_tokens": int(max_new_tokens),
        "condition_on_prev_tokens": False,
    }


@torch.inference_mode()
def decode_population(bundle, pop, *, language: str | None, num_beams: int = 1,
                      plan: SteeringPlan | None = None,
                      utterance_plans: dict[str, UtterancePlan] | None = None,
                      coefficient: float = 0.0, batch_size: int = 8,
                      max_new_tokens: int = 200, log=None) -> DecodeResult:
    """Decode every utterance of `pop`, optionally under a steering plan."""
    from csasr.models.generation import batch_model_inputs

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts: dict[str, str] = {}
    tokens: dict[str, int] = {}
    hit_max: dict[str, bool] = {}
    audits: list[dict[str, Any]] = []
    kwargs = _generate_kwargs(language=language, num_beams=num_beams,
                              max_new_tokens=max_new_tokens)
    started = time.monotonic()
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0

    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
        ids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        if plan is None:
            out = bundle.model.generate(**inputs, **kwargs)
            session_audit = None
        else:
            ordered = [utterance_plans[u] for u in ids]
            with SteeringSession(bundle, plan, ordered, coefficient) as session:
                out = bundle.model.generate(**inputs, **kwargs)
                session_audit = session.audit()
                if session.decoder is not None:
                    session.decoder.assert_every_beam_steered()
            audits.append(session_audit)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(sequences, skip_special_tokens=True)
        pad = bundle.model.config.pad_token_id
        eos = bundle.processor.tokenizer.eos_token_id
        for i, utterance in enumerate(ids):
            row = sequences[i]
            body = row[row != pad]
            generated = int(max(0, len(body) - C.PREFIX_WIDTH))
            texts[utterance] = decoded[i].strip()
            tokens[utterance] = generated
            hit_max[utterance] = bool(generated >= max_new_tokens
                                      and int(row[-1]) != int(eos))
        if log and (bi + 1) % 10 == 0:
            log.info("  decoded %d/%d utterances", min((bi + 1) * batch_size,
                                                       len(manifest)), len(manifest))

    audit = _merge_audits(audits)
    return DecodeResult(texts=texts, num_tokens=tokens, hit_max_tokens=hit_max,
                        audit=audit, wall_clock_sec=time.monotonic() - started)


#: `decoder_min_absolute_position` is `None` for any batch where local
#: coverage steered nothing (e.g. no oracle step fell in that batch); summing
#: is also the wrong merge for a "minimum position" field regardless -- the
#: meaningful cross-batch value is the min of the per-batch mins, which is
#: exactly what this diagnostic exists to catch (a forced-prefix position
#: getting touched, requirement 1.2/1.3).
MIN_MERGE_KEYS = frozenset({"decoder_min_absolute_position"})


def _merge_audits(audits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not audits:
        return {}
    out: dict[str, Any] = {}
    ratios: list[float] = []
    mins: dict[str, list[int]] = {k: [] for k in MIN_MERGE_KEYS}
    seen_min_keys: set[str] = set()
    for a in audits:
        for k, v in a.items():
            if k == "valid_frame_ratio":
                ratios.extend(v)
            elif k in MIN_MERGE_KEYS:
                seen_min_keys.add(k)
                if v is not None:
                    mins[k].append(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if k.endswith("_mean") or k.startswith("expected") or k == "num_beams":
                    out[k] = v
                else:
                    out[k] = out.get(k, 0) + v
            else:
                out[k] = v
    for k in seen_min_keys:
        out[k] = min(mins[k]) if mins[k] else None
    out["valid_frame_ratio_min"] = float(np.min(ratios)) if ratios else float("nan")
    out["valid_frame_ratio_mean"] = float(np.mean(ratios)) if ratios else float("nan")
    out["valid_frame_ratio_max"] = float(np.max(ratios)) if ratios else float("nan")
    out["utterances_fully_padded_free"] = int(sum(1 for r in ratios if r >= 1.0))
    return out


# ---------------------------------------------------------------------------
# decode health (3.7) -- a failing run is INVALID, never negative
# ---------------------------------------------------------------------------

def decode_health(pop, result: DecodeResult, *, max_new_tokens: int = 200) -> dict[str, Any]:
    """Median output-length ratio, empty-output rate, max_new_tokens hit rate."""
    from csasr.data.normalize import normalize_and_segment
    from . import data as D

    references = D.reference_of(pop)
    ratios: list[float] = []
    empty = 0
    for utterance, hypothesis in result.texts.items():
        reference = references.get(utterance, "")
        _, ref_units = normalize_and_segment(reference)
        _, hyp_units = normalize_and_segment(hypothesis)
        if len(ref_units):
            ratios.append(len(hyp_units) / len(ref_units))
        if not str(hypothesis).strip():
            empty += 1
    n = max(1, len(result.texts))
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    empty_rate = empty / n
    hit_rate = sum(1 for v in result.hit_max_tokens.values() if v) / n

    lo, hi = C.DECODE_HEALTH["median_length_ratio_range"]
    checks = {
        "median_length_ratio": median_ratio,
        "median_length_ratio_ok": bool(lo <= median_ratio <= hi),
        "empty_output_rate": empty_rate,
        "empty_output_rate_ok": bool(empty_rate < C.DECODE_HEALTH["empty_output_rate_max"]),
        "max_new_tokens_hit_rate": hit_rate,
        "max_new_tokens_hit_rate_ok": bool(
            hit_rate < C.DECODE_HEALTH["max_new_tokens_hit_rate_max"]),
    }
    checks["healthy"] = bool(checks["median_length_ratio_ok"]
                             and checks["empty_output_rate_ok"]
                             and checks["max_new_tokens_hit_rate_ok"])
    return checks


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def run_cell(bundle, pop, *, plan: SteeringPlan, utterance_plans: dict[str, UtterancePlan],
             language: str | None, batch_size: int = 8, max_new_tokens: int = 200,
             log=None) -> tuple[DecodeResult, dict[str, Any]]:
    """Decode the population under one steering plan, with energy accounting."""
    plan.validate()
    ordered = [utterance_plans[str(u)] for u in pop.manifest["utterance_id"]]
    n_positions = planned_positions(plan, ordered)
    energy = energy_report(plan, n_positions)
    result = decode_population(
        bundle, pop, language=language, num_beams=plan.num_beams, plan=plan,
        utterance_plans=utterance_plans, coefficient=energy["alpha_eff"],
        batch_size=batch_size, max_new_tokens=max_new_tokens, log=log)
    energy["S_observed"] = int(result.audit.get("steered_positions", 0))
    energy["E_total_observed"] = (
        float(energy["alpha_eff"] ** 2) * float(energy["direction_norm"]) ** 2
        * float(energy["S_observed"]))
    return result, energy
