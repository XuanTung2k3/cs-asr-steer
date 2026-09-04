"""Track A preflight. Six checks. Any failure halts the run with a nonzero exit.

Check 2 is the one that matters most: it reproduces the prior oracle-localized
single-step decoder run -- 489 rows, exactly 7 corrections. If it does not
match, this harness is NOT the same experiment as the prior programme and every
downstream comparison is void. It is therefore run through the ORIGINAL frozen
code path (`v2r3_day5_expansion.free_decode` with `language=None`, the raw
un-normalised direction and rho=1), not through this package's sweep path,
because reproducing a prior result means running the prior configuration.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import config as C
from .decode import build_utterance_plans, decode_population
from .hooks import (BeamCoverageError, PromptDirectionAtEncoderError,
                    SteeringPlan, energy_report)


class PreflightFailure(AssertionError):
    """A preflight check failed. Observed vs expected is in the message."""


def _result(name: str, passed: bool, observed: Any, expected: Any,
            detail: str = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "observed": observed,
            "expected": expected, "detail": detail}


# ---------------------------------------------------------------------------
# 1. alpha = 0 reproduces the baseline bit-identically
# ---------------------------------------------------------------------------

def check_alpha_zero(bundle, pop, store, *, batch_size: int, log=None) -> list[dict[str, Any]]:
    """Every site x coverage combination must be a no-op at alpha = 0."""
    from .decode import decode_population as run

    baseline = run(bundle, pop, language="zh", num_beams=1,
                   batch_size=batch_size, log=log)
    plans = build_utterance_plans(bundle, pop, baseline_tokens=baseline.num_tokens,
                                  utterance_ids=pop.utterance_ids)
    v_nat_enc = store.get(C.KIND_V_NAT, C.SITE_ENCODER, C.MID_ENCODER_LAYER,
                          bundle=bundle)
    v_nat_dec = store.get(C.KIND_V_NAT, C.SITE_DECODER, C.MID_DECODER_LAYER,
                          bundle=bundle)
    out: list[dict[str, Any]] = []
    for site in (C.SITE_ENCODER, C.SITE_DECODER, C.SITE_BOTH):
        for coverage in (C.COVERAGE_GLOBAL, C.COVERAGE_LOCAL):
            plan = SteeringPlan(
                site=site, coverage=coverage, alpha=0.0, matched_energy=False,
                encoder_layers=(C.MID_ENCODER_LAYER,) if site != C.SITE_DECODER else (),
                decoder_layers=(C.MID_DECODER_LAYER,) if site != C.SITE_ENCODER else (),
                encoder_direction=v_nat_enc if site != C.SITE_DECODER else None,
                decoder_direction=v_nat_dec if site != C.SITE_ENCODER else None)
            steered = run(bundle, pop, language="zh", num_beams=1, plan=plan,
                          utterance_plans=plans, coefficient=0.0,
                          batch_size=batch_size, log=log)
            identical = all(baseline.texts[u] == steered.texts.get(u)
                            for u in baseline.texts)
            differing = [u for u in baseline.texts
                         if baseline.texts[u] != steered.texts.get(u)]
            out.append(_result(
                f"1.alpha_zero[{site}/{coverage}]", identical,
                f"{len(differing)} utterances differ", "0 utterances differ",
                detail=("; ".join(differing[:3]) if differing else
                        "bit-identical to the unhooked baseline")))
    return out


# ---------------------------------------------------------------------------
# 2. reproduce the prior oracle-localized single-step decoder run
# ---------------------------------------------------------------------------

def check_reproduce_prior(bundle, pop, cfg, *, log=None,
                          limit: int | None = None) -> list[dict[str, Any]]:
    """489/489 rows and exactly 7 corrections, through the original code path.

    The prior programme decoded with `language=None` -- Whisper chose its own
    language token and the prefill width was measured, not assumed -- and
    steered the post-cross-attention residual at decoder layer 16 with the RAW
    direction (norm 2.4146) at rho = 1 under norm-preserving renormalisation.
    Reproducing it means running exactly that, so this check calls the frozen
    Day-5 routines rather than this package's sweep path.
    """
    import numpy as np
    from csasr.experiments import v2r3_day5_expansion as D5
    from .directions import store_path

    directions = dict(np.load(store_path(cfg)))
    layer = int(D5.FROZEN_ACTION["decoder_layer"])
    rho = float(D5.FROZEN_ACTION["rho"])
    vector = directions[f"site_d_layer{layer}"]

    targets = pop.targets
    utterances = pop.manifest[
        pop.manifest["utterance_id"].isin(set(targets["utterance_id"]))
    ].drop_duplicates("utterance_id").reset_index(drop=True)
    if limit:
        utterances = utterances.head(int(limit))
        targets = targets[targets["utterance_id"].isin(set(utterances["utterance_id"]))]

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for i, (_, row) in enumerate(utterances.iterrows()):
        utterance = str(row["utterance_id"])
        group = targets[targets["utterance_id"] == utterance]
        step = max(0, int(group["token_index"].min()) - 1)
        baseline_text, _ = D5.free_decode(bundle, cfg, row["audio_path"])
        steered_text, fired = D5.free_decode(
            bundle, cfg, row["audio_path"], direction=vector, layer=layer,
            rho=rho, target_step=step)
        for _, t in group.iterrows():
            rates = D5.outcome_rates(row["transcript_raw"], baseline_text,
                                     steered_text, int(t["reference_unit_index"]))
            rows.append({"utterance_id": utterance,
                         "dialogue_id": str(t["dialogue_id"]),
                         "reference_unit_index": int(t["reference_unit_index"]),
                         "hook_fired": int(fired), **rates})
        if log and (i + 1) % 20 == 0:
            log.info("  preflight-2: %d/%d utterances", i + 1, len(utterances))

    frame = pd.DataFrame(rows)
    corrections = int(frame["corrected"].sum()) if len(frame) else 0
    dialogues = sorted(frame.loc[frame["corrected"], "dialogue_id"].unique().tolist()) \
        if len(frame) else []
    elapsed = time.monotonic() - started

    if limit:
        return [_result("2.reproduce_prior[SMOKE]", True,
                        {"rows": len(frame), "corrections": corrections},
                        "not asserted under --smoke",
                        detail=f"limited to {limit} utterances; the full check "
                               f"is the only one that binds ({elapsed:.0f}s)")]
    checks = [
        _result("2.reproduce_prior.rows", len(frame) == C.ANCHOR_TARGETS,
                len(frame), C.ANCHOR_TARGETS),
        _result("2.reproduce_prior.corrections",
                corrections == C.ANCHOR_ORACLE_CORRECTIONS,
                corrections, C.ANCHOR_ORACLE_CORRECTIONS,
                detail=f"correcting dialogues {dialogues}; prior programme had "
                       f"{list(C.ANCHOR_ORACLE_CORRECTION_DIALOGUES)} "
                       f"({elapsed:.0f}s)"),
    ]
    return checks


# ---------------------------------------------------------------------------
# 3. hook fire count equals steered layers per forward pass
# ---------------------------------------------------------------------------

def check_hook_fire_count(bundle, pop, store, *, batch_size: int,
                          log=None) -> list[dict[str, Any]]:
    from .decode import decode_population as run

    small = pop.subsample(min(4, len(pop.manifest)))
    baseline = run(bundle, small, language="zh", batch_size=batch_size, log=log)
    plans = build_utterance_plans(bundle, small, baseline_tokens=baseline.num_tokens,
                                  utterance_ids=small.utterance_ids)
    enc_layers = (8, 15)
    dec_layers = (8, 16, 24)
    plan = SteeringPlan(
        site=C.SITE_BOTH, coverage=C.COVERAGE_GLOBAL, alpha=1.0,
        encoder_layers=enc_layers, decoder_layers=dec_layers,
        encoder_direction=store.get(C.KIND_V_NAT, C.SITE_ENCODER, enc_layers[0],
                                    bundle=bundle),
        decoder_direction=store.get(C.KIND_V_NAT, C.SITE_DECODER, dec_layers[0],
                                    bundle=bundle))
    energy = energy_report(plan, 1000)
    steered = run(bundle, small, language="zh", plan=plan, utterance_plans=plans,
                  coefficient=energy["alpha_eff"], batch_size=batch_size, log=log)
    audit = steered.audit
    n_batches = math.ceil(len(small.manifest) / batch_size)
    enc_expected = n_batches * len(enc_layers)
    dec_forwards = int(audit.get("decoder_forward_calls", 0))
    dec_expected = dec_forwards * len(dec_layers)
    return [
        _result("3.hook_fires.encoder",
                int(audit.get("encoder_hook_fires", 0)) == enc_expected,
                int(audit.get("encoder_hook_fires", 0)), enc_expected,
                detail=f"{n_batches} encoder forwards x {len(enc_layers)} layers"),
        _result("3.hook_fires.decoder",
                int(audit.get("decoder_hook_fires", 0)) == dec_expected,
                int(audit.get("decoder_hook_fires", 0)), dec_expected,
                detail=f"{dec_forwards} decoder forwards x {len(dec_layers)} layers"),
    ]


# ---------------------------------------------------------------------------
# 4. encoder padding mask active
# ---------------------------------------------------------------------------

def check_encoder_padding(bundle, pop) -> list[dict[str, Any]]:
    """valid_frame_ratio == ceil(duration*50)/1500 and is < 1 for short clips."""
    from . import data as D

    durations = D.duration_of(pop)
    bad: list[str] = []
    short = 0
    for utterance, duration in durations.items():
        expected = min(C.EXPECTED_ENCODER_FRAMES,
                       max(1, int(math.ceil(duration * C.FRAMES_PER_SECOND))))
        observed = int(bundle.valid_frames(duration))
        if observed != expected:
            bad.append(f"{utterance}: {observed} != {expected}")
        if expected < C.EXPECTED_ENCODER_FRAMES:
            short += 1
    ratios = [min(C.EXPECTED_ENCODER_FRAMES,
                  max(1, int(math.ceil(d * C.FRAMES_PER_SECOND))))
              / C.EXPECTED_ENCODER_FRAMES for d in durations.values()]
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    return [
        _result("4.encoder_padding.frame_count", not bad,
                f"{len(bad)} mismatches", "0 mismatches",
                detail="; ".join(bad[:3]) if bad else
                       "valid_frames == ceil(duration*50) clamped to 1500"),
        _result("4.encoder_padding.short_utterances_masked", short > 0,
                f"{short}/{len(durations)} utterances below 1500 frames",
                ">0 utterances below 1500 frames",
                detail=f"mean valid_frame_ratio {mean_ratio:.4f}; without this "
                       f"mask roughly {100 * (1 - mean_ratio):.0f}% of injected "
                       f"encoder energy would land in padding"),
    ]


# ---------------------------------------------------------------------------
# 5. beam search steers every beam
# ---------------------------------------------------------------------------

def check_beams(bundle, pop, store, *, batch_size: int, num_beams: int = 5,
                log=None) -> list[dict[str, Any]]:
    from .decode import decode_population as run

    small = pop.subsample(min(4, len(pop.manifest)))
    baseline = run(bundle, small, language="zh", num_beams=num_beams,
                   batch_size=min(batch_size, 2), log=log)
    plans = build_utterance_plans(bundle, small, baseline_tokens=baseline.num_tokens,
                                  utterance_ids=small.utterance_ids)
    plan = SteeringPlan(
        site=C.SITE_DECODER, coverage=C.COVERAGE_GLOBAL, alpha=1.0,
        decoder_layers=(C.MID_DECODER_LAYER,), num_beams=num_beams,
        decoder_direction=store.get(C.KIND_V_NAT, C.SITE_DECODER,
                                    C.MID_DECODER_LAYER, bundle=bundle))
    energy = energy_report(plan, 1000)
    try:
        steered = run(bundle, small, language="zh", num_beams=num_beams, plan=plan,
                      utterance_plans=plans, coefficient=energy["alpha_eff"],
                      batch_size=min(batch_size, 2), log=log)
    except BeamCoverageError as exc:
        return [_result("5.beam_coverage", False, str(exc),
                        "steered_positions % num_beams == 0")]
    positions = int(steered.audit.get("decoder_steered_positions", 0))
    return [_result(
        "5.beam_coverage", positions > 0 and positions % num_beams == 0,
        {"steered_positions": positions, "num_beams": num_beams,
         "remainder": positions % num_beams},
        "steered_positions > 0 and divisible by num_beams",
        detail="with num_beams=k the decoder batch is B*k and the hook touches "
               "every row, so positions stay synchronised across beams")]


# ---------------------------------------------------------------------------
# 6. matched-energy invariance
# ---------------------------------------------------------------------------

def check_matched_energy() -> list[dict[str, Any]]:
    """E_total must be constant across |S| in {1, 10, 100, 1000}."""
    plan = SteeringPlan(site=C.SITE_DECODER, coverage=C.COVERAGE_GLOBAL, alpha=2.0,
                        matched_energy=True, decoder_layers=(16,),
                        decoder_direction=None)
    totals = {}
    for s in (1, 10, 100, 1000):
        report = energy_report(plan, s)
        totals[s] = round(report["E_total"], 10)
    values = set(totals.values())
    return [_result("6.matched_energy_invariance", len(values) == 1, totals,
                    {s: round(plan.alpha ** 2, 10) for s in totals},
                    detail="E_total = alpha_eff^2 ||v||^2 |S| with "
                           "alpha_eff = alpha / sqrt(|S|)")]


# ---------------------------------------------------------------------------
# 1.6 guard
# ---------------------------------------------------------------------------

def check_prompt_site_guard(store, bundle) -> list[dict[str, Any]]:
    """A prompt direction at an encoder site must raise, not return a number."""
    try:
        store.get(C.KIND_V_PROMPT, C.SITE_ENCODER, C.MID_ENCODER_LAYER, bundle=bundle)
    except PromptDirectionAtEncoderError as exc:
        return [_result("1.6.prompt_at_encoder_raises", True,
                        type(exc).__name__, "PromptDirectionAtEncoderError",
                        detail=str(exc)[:160])]
    return [_result("1.6.prompt_at_encoder_raises", False,
                    "returned a direction", "PromptDirectionAtEncoderError")]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_preflight_a(bundle, pop, cfg, store, *, batch_size: int = 8,
                    smoke: bool = False, log=None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    results += check_matched_energy()
    results += check_encoder_padding(bundle, pop)
    results += check_prompt_site_guard(store, bundle)
    results += check_alpha_zero(bundle, pop.subsample(min(4, len(pop.manifest))),
                                store, batch_size=batch_size, log=log)
    results += check_hook_fire_count(bundle, pop, store, batch_size=batch_size, log=log)
    results += check_beams(bundle, pop, store, batch_size=batch_size, log=log)
    results += check_reproduce_prior(bundle, pop, cfg, log=log,
                                     limit=4 if smoke else None)
    passed = all(r["passed"] for r in results)
    return {"track": "A", "passed": passed, "smoke": bool(smoke),
            "checks": results,
            "n_passed": sum(1 for r in results if r["passed"]),
            "n_total": len(results)}


def format_preflight(report: dict[str, Any]) -> str:
    lines = [f"=== PREFLIGHT TRACK {report['track']} "
             f"({report['n_passed']}/{report['n_total']} passed"
             f"{', SMOKE' if report.get('smoke') else ''}) ==="]
    for r in report["checks"]:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{mark}] {r['check']}")
        lines.append(f"       observed: {r['observed']}")
        lines.append(f"       expected: {r['expected']}")
        if r.get("detail"):
            lines.append(f"       {r['detail']}")
    return "\n".join(lines)
