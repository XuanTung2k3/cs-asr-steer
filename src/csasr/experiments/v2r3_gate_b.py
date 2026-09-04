"""Gate B evidence: free decoding under oracle and NON-ORACLE localization.

**Purpose.**  Every effect measured so far lives in teacher-forced logprob space
and is localized by an oracle that hands the intervention the exact decoder step.
Neither condition exists at inference.  This stage asks whether the frozen action
produces a net practical improvement on transcripts when the step must be found
from the model's own first-pass output.

**Boundary.**  This stage measures and reports.  It selects no tier, layer, or
strength, evaluates no gate criterion, and touches neither `D-dev-confirm` nor
any intervened condition on `D-test`.  `D-test` appears only as reference-only
counts and a C00 baseline, which are not intervention outcomes.

**Localization is the crux.**  Site E is null, so the problem is not "find the
acoustic span" but "find the decoder step".  Two arms are run in one job so they
share a model, a batch order, and a baseline decode:

  (a) ORACLE -- the gold step, reproducing Day 5 as the reference arm.
  (b) NON-ORACLE -- the transport mechanism, from first-pass output only.

**Transport.**  For decoder step ``q`` and encoder frame ``t``,

    r_q = sum_t A_bar[q, t] * g_t

where ``A_bar`` is the head-averaged cross-attention of the final decoder layer
and ``g_t`` indicates the candidate embedded-English region.  Because each
attention row sums to one over frames, ``r_q`` is the *fraction of step q's
attention mass landing on the candidate region* -- a quantity in [0, 1] with a
meaning.  It is deliberately NOT renormalised: renormalising would discard
exactly the information that distinguishes a step attending mostly to the
candidate region from one that barely looks at it.

The steering strength at step ``q`` is ``rho * r_q``, so the effective
intervention energy is

    E_eff = rho^2 * ||Delta^D||^2 * sum_q r_q^2

and the matched-energy control is exact rather than nominal: it reuses the same
``r_q`` and the same ``rho`` with a random direction of identical norm.

**No gold may enter the non-oracle arm.**  ``g_t`` and the step weights are
derived from the first-pass token sequence alone.  The attention matrix is
obtained by re-forwarding *that generated sequence*; for a causal decoder this
reproduces each step's attention exactly, and it introduces no gold text.  The
gold transcript is read only to score outcomes after both decodes are complete.

Every parameter below was frozen before the run and is recorded in the manifest.
None was chosen by looking at a result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..evaluation.mer import corpus_mer
from ..evaluation.pier import pier, unit_status
from ..data.language_tags import EN, tag_unit
from ..data.normalize import normalize_and_segment
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from . import v2r3_cross_attention_spans as X2
from . import v2r3_day5_expansion as D5
from . import v2r3_directions as D3
from . import v2r3_oracle_screen as OS

STAGE = "v2r3_gate_b"

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION
# ---------------------------------------------------------------------------

#: Recorded, not re-derived.
FROZEN_ACTION: dict[str, Any] = dict(D5.FROZEN_ACTION)

#: Eligibility tiers, conservative first.
FLOORS_MS = (400.0, 300.0, 200.0)

#: Transport reads the same cross-attention Day 2 used: final decoder layer,
#: every head averaged.  Reused rather than re-chosen.
TRANSPORT_LAYER_INDEX = X2.DECODER_LAYER_INDEX
TRANSPORT_HEAD_SET = X2.HEAD_SET

#: Abstention.  ``r_q`` is attention mass on the candidate region, and the scale
#: of that mass is set by the encoder: Whisper pads every input to 1500 frames
#: and attends into the padding, so even a step aligned to a 400 ms region holds
#: only a small fraction of its row there.  A fixed cut such as 0.5 therefore
#: abstains always and makes the arm vacuous by construction rather than by
#: evidence -- confirmed on a 3-utterance smoke where max_q r_q reached 0.114.
#:
#: The rule used instead is parameter-free: the selected step must attend to the
#: candidate region MORE than uniform chance, where chance is the region's share
#: of the valid frames.  Nothing here is fitted to an outcome; ``r_max``, the
#: chance level, and the enrichment are recorded per utterance so the abstention
#: rate at any threshold can be read off afterwards without re-running.
ABSTENTION_ENRICHMENT = 1.0

#: A region covering the whole utterance yields enrichment of exactly 1, and
#: float noise then decides the abstention arbitrarily.  Ties abstain: a region
#: that is the entire audio carries no localization information.
ABSTENTION_TIE_TOLERANCE = 1e-9

#: The stricter fixed-mass variant, reported alongside rather than used to gate.
ABSTENTION_TAU_REPORTED = 0.5

#: A transport "hit" is a selected step within this many steps of the oracle
#: step.  Preregistered; widening it would inflate the non-oracle arm.
TRANSPORT_HIT_TOLERANCE = 1

#: Statistics.  10k resamples as required; the shared module default is 2000, so
#: this is passed explicitly rather than by mutating a constant other stages use.
BOOTSTRAP_DRAWS = 10_000

#: Null-draw noise calibrated on Day 6 from four identically-constructed cells
#: differing only in seed.  Differences smaller than this are reported as
#: indistinguishable rather than as distinctions.
NULL_DRAW_NOISE = 0.03

#: Controls under free decoding.
CONTROL_RANDOM_DRAWS = 5

FROZEN_CONFIG: dict[str, Any] = {
    "action": FROZEN_ACTION,
    "floors_ms": list(FLOORS_MS),
    "transport": {
        "formula": "r_q = sum_t A_bar[q, t] * g_t",
        "attention": {"decoder_layer_index": TRANSPORT_LAYER_INDEX,
                      "head_set": TRANSPORT_HEAD_SET},
        "renormalised": False,
        "renormalisation_note": (
            "r_q is left as attention mass on the candidate region; each "
            "attention row sums to 1 over frames, so r_q in [0, 1] is a "
            "fraction with a meaning that renormalising would destroy"),
        "energy": "E_eff = rho^2 * ||Delta^D||^2 * sum_q r_q^2",
        "g_t_source": "first-pass generated tokens only; no gold",
        "attention_source": (
            "the first-pass generated sequence re-forwarded; exact for a causal "
            "decoder and free of gold text"),
    },
    "abstention": {
        "rule": ("abstain if there is no candidate region, or if the best step's "
                 "attention on it does not exceed uniform chance "
                 "(enrichment = r_max / (|region| / |valid frames|) <= 1)"),
        "enrichment_threshold": ABSTENTION_ENRICHMENT,
        "tie_tolerance": ABSTENTION_TIE_TOLERANCE,
        "ties_abstain": True,
        "parameter_free": True,
        "reported_variant_tau": ABSTENTION_TAU_REPORTED,
        "tuned": False,
        "specification_note": (
            "an earlier fixed cut of max_q r_q < 0.5 was replaced before the "
            "full run: Whisper's 1500-frame padded attention puts r_q on a "
            "scale where 0.5 abstains always, which is a property of the "
            "encoder rather than of the localizer. r_max is recorded so any "
            "threshold can be evaluated post hoc"),
    },
    "transport_hit_tolerance_steps": TRANSPORT_HIT_TOLERANCE,
    "bootstrap": {"kind": "wild_cluster", "weights": "rademacher",
                  "critical_values": "t(G-1)", "cluster": "dialogue_id",
                  "draws": BOOTSTRAP_DRAWS, "ci": OS.BOOTSTRAP_CI,
                  "seed": OS.BOOTSTRAP_SEED},
    "null_draw_noise": NULL_DRAW_NOISE,
    "selects_anything": False,
    "evaluates_gate": False,
    "d_test_use": "reference-only counts and C00 baseline; no intervened condition",
}


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

class TransportSteering:
    """Steer every generation step by ``rho * r_q``.

    Day 5's hook fires at exactly one oracle step.  Transport instead spreads the
    intervention over steps in proportion to the attention mass each pays to the
    candidate region, so the hook needs a weight per step rather than a target.
    The oracle arm is the special case ``r_q = 1`` at the gold step and 0
    elsewhere, which keeps both arms inside one energy accounting.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, rho: float,
                 weights: Sequence[float], norm_preserve: bool = True):
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction
        self.rho = float(rho)
        self.weights = np.asarray(weights, dtype=float)
        self.norm_preserve = norm_preserve
        self.step = -1
        self.fired = 0
        self.prefill_width = 0
        self.applied_sq = 0.0            # sum of r_q^2 over steps that fired
        self._residual: torch.Tensor | None = None
        self._handles: list = []

    def _pre_hook(self, _mod, args):
        self._residual = args[0]
        return None

    def _hook(self, _mod, _inp, output):
        from ..lss.sites import _first, _rebuild
        from ..models.hooks import apply_steering

        if self._residual is None:
            raise RuntimeError("cross-attention ran without its layer-norm pre-hook")
        attn_out = _first(output)
        site = self._residual + attn_out
        if site.shape[1] > 1:                     # prefill
            self.prefill_width = int(site.shape[1])
            return output
        self.step += 1
        weight = float(self.weights[self.step]) if self.step < len(self.weights) else 0.0
        if weight <= 0.0:
            return output
        gain = torch.ones(site.shape[:2], dtype=torch.float32)
        steered = apply_steering(site, self.direction, self.rho * weight, 1.0,
                                 gain, self.norm_preserve)
        self.fired += 1
        self.applied_sq += weight * weight
        return _rebuild(output, attn_out + (steered - site))

    def __enter__(self) -> "TransportSteering":
        D5.assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(
            layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._residual = None
        return False


class PrefillProbe:
    """Record the prefill width and the number of generation steps.

    The mapping from a hook step to an attention row depends on how many tokens
    the model forced before generating, and with ``language=None`` that prefix is
    chosen by the model rather than by us.  Measuring it is safer than assuming
    a width of four.
    """

    def __init__(self, bundle, layer: int):
        self.bundle, self.layer = bundle, int(layer)
        self.prefill_width, self.steps = 0, 0
        self._handles: list = []

    def _hook(self, _mod, _inp, output):
        from ..lss.sites import _first
        out = _first(output)
        if out.shape[1] > 1:
            self.prefill_width = int(out.shape[1])
        else:
            self.steps += 1
        return output

    def __enter__(self) -> "PrefillProbe":
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


def free_decode_probed(bundle, audio_path: str, layer: int) -> tuple[list[int], int, int]:
    """Unsteered first pass, returning its token ids, prefill width, and steps."""
    from ..models.generation import batch_model_inputs

    inputs = batch_model_inputs(bundle, [audio_path])
    kwargs = {"task": "transcribe", "language": None, "do_sample": False,
              "num_beams": 1, "temperature": 0.0, "max_new_tokens": 200}
    with PrefillProbe(bundle, layer) as probe:
        with torch.inference_mode():
            out = bundle.model.generate(**inputs, **kwargs)
    return [int(v) for v in out[0]], probe.prefill_width, probe.steps


def candidate_region(bundle, tokens: Sequence[int], attention: np.ndarray,
                     duration_sec: float, floor_ms: float) -> dict[str, Any]:
    """``g_t`` over encoder frames, from the FIRST-PASS tokens only.

    Reuses the Day 2 derivation -- assign each frame the class of the token that
    attends to it most, mode-filter, take contiguous runs -- with the generated
    sequence in place of the gold one.  A candidate must clear the tier floor.
    """
    tokenizer = bundle.processor.tokenizer
    classes = X2.token_classes(tokenizer, tokens)
    step = float(bundle.encoder_step_sec)
    valid = max(1, min(X2.MAX_ENCODER_FRAMES, int(round(duration_sec / step))))
    window = attention[: len(tokens), :valid]
    if window.size == 0:
        return {"g": np.zeros(0, dtype=float), "spans": [], "valid_frames": valid}
    argmax_token = window.argmax(axis=0)
    raw = [classes[int(t)] for t in argmax_token]
    labels = X2.mode_filter(raw, X2.SMOOTHING_WINDOW_FRAMES)
    all_en = [s for s in X2.spans_from_labels(labels, step_sec=step)
              if str(s.get("language")) == X2.CLASS_EN]
    spans = [s for s in all_en
             if float(s["end_sec"] - s["start_sec"]) * 1000.0 >= float(floor_ms)]
    # Whether the first pass produced ANY English at all is the distinction that
    # separates "the localizer missed it" from "there was nothing to localize":
    # when the baseline error is a dropped or translated English word, the
    # hypothesis contains no English and transport has no signal by construction.
    has_english_token = any(c == X2.CLASS_EN for c in classes)
    g = np.zeros(valid, dtype=float)
    for span in spans:
        lo = max(0, int(round(float(span["start_sec"]) / step)))
        hi = min(valid, int(round(float(span["end_sec"]) / step)))
        if hi > lo:
            g[lo:hi] = 1.0
    return {"g": g, "spans": spans, "valid_frames": valid,
            "en_spans_any": len(all_en), "en_spans_above_floor": len(spans),
            "hypothesis_has_english": bool(has_english_token)}


def step_valid_mass(attention: np.ndarray, valid_frames: int, *, prefill_width: int,
                    steps: int) -> np.ndarray:
    """Attention mass each step places inside the audio, not the padding.

    Whisper pads every input to 30 s and applies no encoder attention mask, so a
    decoder step spends most of its row outside the audio.  ``r_q`` is a share of
    the whole row, so the chance level it must beat has to be measured on the
    same denominator -- otherwise the padding deflates every ``r_q`` while the
    chance level is computed inside the audio, and no step can ever clear it.
    """
    out = np.zeros(max(0, steps), dtype=float)
    for q in range(steps):
        index = int(prefill_width) + q
        if 0 <= index < attention.shape[0]:
            out[q] = float(attention[index, :valid_frames].sum())
    return out


def transport_weights(attention: np.ndarray, g: np.ndarray, *, prefill_width: int,
                      steps: int) -> np.ndarray:
    """``r_q = sum_t A_bar[q, t] g_t`` for each generation step ``q``.

    Prefill of width ``P`` consumes positions ``0..P-1`` and produces the token
    at position ``P``; hook step ``q`` is then the single-token forward whose
    attention row is position ``P + q``.  Rows past the sequence contribute zero.
    """
    if g.size == 0 or steps <= 0:
        return np.zeros(max(0, steps), dtype=float)
    rows = attention[:, : g.size]
    out = np.zeros(steps, dtype=float)
    for q in range(steps):
        index = int(prefill_width) + q
        if 0 <= index < rows.shape[0]:
            out[q] = float(np.dot(rows[index], g))
    return out


def effective_energy(rho: float, direction_norm: float, weights: np.ndarray) -> float:
    """``E_eff = rho^2 * ||Delta||^2 * sum_q r_q^2``."""
    return float(rho ** 2) * float(direction_norm) ** 2 * float(np.sum(np.asarray(weights) ** 2))


def oracle_weights(step: int, steps: int) -> np.ndarray:
    """The oracle arm as a transport vector: unit mass at the gold step."""
    out = np.zeros(max(0, steps), dtype=float)
    if 0 <= step < len(out):
        out[step] = 1.0
    return out


def steered_decode(bundle, audio_path: str, direction: np.ndarray, *, layer: int,
                   rho: float, weights: np.ndarray) -> tuple[str, int, float, int]:
    """Free decoding with the transport-weighted intervention."""
    from ..models.generation import batch_model_inputs

    inputs = batch_model_inputs(bundle, [audio_path])
    kwargs = {"task": "transcribe", "language": None, "do_sample": False,
              "num_beams": 1, "temperature": 0.0, "max_new_tokens": 200}
    vector = torch.as_tensor(direction, dtype=torch.float32, device=bundle.device)
    with TransportSteering(bundle, layer, vector, rho, weights) as hook:
        with torch.inference_mode():
            out = bundle.model.generate(**inputs, **kwargs)
        fired, applied, prefill = hook.fired, hook.applied_sq, hook.prefill_width
    text = bundle.processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text, fired, applied, prefill


# ---------------------------------------------------------------------------
# outcomes
# ---------------------------------------------------------------------------

def monolingual_retention(reference: str, baseline: str, steered: str) -> dict[str, int]:
    """Retention on Mandarin (matrix-language) reference units.

    The intervention is aimed at embedded English; matrix material is what it
    must not damage.  Counted over units that were correct in the baseline.
    """
    norm, units = normalize_and_segment(reference)
    before, after = unit_status(reference, baseline), unit_status(reference, steered)
    kept = lost = total = 0
    for index, unit in enumerate(units):
        if tag_unit(unit) == EN:
            continue
        was = bool(before.get(index, (False, ""))[0])
        if not was:
            continue
        total += 1
        if bool(after.get(index, (False, ""))[0]):
            kept += 1
        else:
            lost += 1
    return {"monolingual_baseline_correct": total,
            "monolingual_retained": kept, "monolingual_lost": lost}


def corruption_on_correct(reference: str, baseline: str, steered: str,
                          target_index: int) -> dict[str, int]:
    """Corruption over baseline-CORRECT units, which targets are not.

    Day 5 could only report corruption inside the target set, and every target
    there is a baseline error by construction, so its corruption rate had no
    denominator.  This counts the units that had something to lose.
    """
    before, after = unit_status(reference, baseline), unit_status(reference, steered)
    at_risk = corrupted = 0
    for index in set(before) | set(after):
        if index == target_index:
            continue
        if not bool(before.get(index, (False, ""))[0]):
            continue
        at_risk += 1
        if not bool(after.get(index, (False, ""))[0]):
            corrupted += 1
    return {"units_at_risk": at_risk, "units_corrupted": corrupted}


def cluster_ci(values: np.ndarray, clusters: Sequence[str]) -> dict[str, Any]:
    """Wild cluster bootstrap at this stage's draw count."""
    return OS.wild_cluster_ci(np.asarray(values, dtype=float), list(clusters),
                              draws=BOOTSTRAP_DRAWS, seed=OS.BOOTSTRAP_SEED,
                              ci=OS.BOOTSTRAP_CI)


def paired_ci(left: pd.DataFrame, right: pd.DataFrame, column: str,
              keys: Sequence[str] = ("utterance_id", "reference_unit_index"),
              ) -> dict[str, Any]:
    """CI on a paired per-unit difference (left minus right), clustered."""
    merged = left.merge(right, on=list(keys), suffixes=("_l", "_r"))
    if merged.empty:
        return {"G": 0, "n": 0, "mean": None}
    diff = merged[f"{column}_l"].astype(float) - merged[f"{column}_r"].astype(float)
    dialogue = merged["dialogue_id_l"] if "dialogue_id_l" in merged else merged["dialogue_id"]
    return cluster_ci(diff.to_numpy(), dialogue.astype(str).tolist())


def resolve_output(output: str | Path) -> Path:
    """Destination guards, identical to Day 5's."""
    target = Path(output)
    if target.is_symlink():
        raise SystemExit(f"refusing a symlink destination: {target}")
    target = target.resolve()
    if "artifacts_lss" in str(target):
        raise SystemExit(f"refusing to write inside the v1 root: {target}")
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise SystemExit(f"refusing an existing destination: {target}")
    return target


def resolve_directions(root: str | Path) -> Path:
    """A smoke generation is refused BY NAME as an input, as on Days 4-6."""
    source = Path(root).resolve()
    if "smoke" in source.name:
        raise SystemExit(f"refusing a smoke generation as input: {source}")
    return source


# ---------------------------------------------------------------------------
# D-test headroom -- reference-only counts and a C00 baseline, never intervened
# ---------------------------------------------------------------------------

def dtest_headroom(bundle, cfg: dict, role_path: Path, out_path: Path, *,
                   batch_size: int, limit: int | None, log) -> dict[str, Any]:
    """Counts and a C00 baseline on the locked split.

    Reference-only counts need no model.  The baseline is C00 -- the unmodified
    system -- which is not an intervention outcome; no steered condition is run
    or scored here, and nothing on this split informs any choice.
    """
    from ..models.generation import decode_manifest

    dtest = pd.read_parquet(role_path)
    if limit:
        dtest = dtest.head(int(limit))
    reference_only: dict[str, Any] = {
        "utterances": int(len(dtest)),
        "dialogues": int(dtest["dialogue_id"].nunique()),
        "audio_hours": float(dtest["duration_sec"].sum() / 3600.0),
    }
    embedded = 0
    total_units = 0
    per_utterance_embedded = []
    for _, row in dtest.iterrows():
        _, units = normalize_and_segment(row["transcript_raw"])
        n_en = sum(1 for u in units if tag_unit(str(u)) == EN)
        total_units += len(units)
        embedded += n_en
        per_utterance_embedded.append(n_en)
    reference_only.update({
        "reference_units": int(total_units),
        "embedded_english_units": int(embedded),
        "embedded_share_of_units": (embedded / total_units) if total_units else None,
        "utterances_with_embedded_english": int(sum(1 for v in per_utterance_embedded if v)),
    })
    log.info("D-test reference-only: %d utterances, %d embedded EN units",
             reference_only["utterances"], reference_only["embedded_english_units"])

    log.info("D-test C00 baseline decode (no intervention) ...")
    decoded = decode_manifest(bundle, dtest, cfg, system="C00",
                              config_id="v2r3_gate_b_dtest_baseline",
                              out_path=out_path, language=None,
                              batch_size=batch_size)
    # decode_manifest returns the raw and normalized hypothesis under their own
    # names; scoring uses the raw text, as the development free-decoding path does
    merged = dtest.merge(decoded[["utterance_id", "hypothesis_raw"]].rename(
        columns={"hypothesis_raw": "hypothesis"}), on="utterance_id", how="inner")

    rows = []
    for _, row in merged.iterrows():
        reference = str(row["transcript_raw"])
        hypothesis = str(row["hypothesis"])
        _, units = normalize_and_segment(reference)
        status = unit_status(reference, hypothesis)
        for index, unit in enumerate(units):
            ok = bool(status.get(index, (False, ""))[0])
            rows.append({"utterance_id": str(row["utterance_id"]),
                         "dialogue_id": str(row["dialogue_id"]),
                         "reference_unit_index": int(index),
                         "embedded_english": tag_unit(unit) == EN,
                         "baseline_error": (not ok)})
    units_df = pd.DataFrame(rows)
    errors = units_df[units_df["baseline_error"]]
    embedded_errors = errors[errors["embedded_english"]]
    baseline = {
        "scored_utterances": int(merged["utterance_id"].nunique()),
        "scored_units": int(len(units_df)),
        "baseline_error_units": int(len(errors)),
        "embedded_english_error_units": int(len(embedded_errors)),
        "embedded_share_of_error_mass": (
            len(embedded_errors) / len(errors) if len(errors) else None),
        "PIER_C00": pier(merged["transcript_raw"].tolist(),
                         merged["hypothesis"].tolist()),
        "MER_C00": corpus_mer(merged["transcript_raw"].tolist(),
                              merged["hypothesis"].tolist()),
    }

    # Minimum detectable effect at this split's G.  The intervention has not been
    # run here and must not be, so the per-cluster variance of the *baseline*
    # embedded-error indicator stands in for the effect's variance.  That is an
    # assumption, stated: a real paired difference could be more or less
    # variable than the baseline rate it is computed from.
    per_dialogue = (units_df[units_df["embedded_english"]]
                    .groupby("dialogue_id")["baseline_error"].mean())
    mde = None
    if len(per_dialogue) > 1:
        from scipy import stats
        G = int(len(per_dialogue))
        se = float(per_dialogue.std(ddof=1) / np.sqrt(G))
        t_crit = float(stats.t.ppf(0.5 + OS.BOOTSTRAP_CI / 2, df=G - 1))
        mde = {"G": G, "se_of_cluster_means": se,
               "t_critical_G_minus_1": t_crit,
               "minimum_detectable_effect": t_crit * se,
               "assumption": ("baseline embedded-error rate variance across "
                              "dialogues stands in for the paired effect's "
                              "variance; no intervened condition was run")}
    return {"reference_only": reference_only, "c00_baseline": baseline,
            "minimum_detectable_effect": mde,
            "intervened_conditions_evaluated": 0}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtest-limit", type=int, default=None)
    parser.add_argument("--skip-dtest", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target_dir = resolve_output(args.output_dir)
    directions_root = resolve_directions(args.directions)

    # every prior artifact is verified by BYTES; identity is not required for an
    # artifact consumed by newly written code (runbook, identity section)
    npz_path = directions_root / "directions.npz"
    side = manifest_mod.load(npz_path)
    fmt, _ = manifest_mod._identity_format(dict(side.get("identity") or {}))
    byte_check = manifest_mod.verify(npz_path, cfg=None, require_identity=False)
    if not byte_check["ok"]:
        raise SystemExit(f"directions failed byte verification: {byte_check}")
    directions = dict(np.load(npz_path))
    identity_note = {"classification": fmt, "identity_required": False,
                     "byte_verified_sha256": side["sha256"],
                     "reason": "consumed by newly written code; bytes are the "
                               "property that matters for reading"}

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    for path in [candidate_path] + [role_root / f"role_{r}.parquet"
                                    for r in conv.DEVELOPMENT_ROLES] + \
                [poi_root / f"poi_{r}.parquet" for r in conv.DEVELOPMENT_ROLES]:
        check = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not check["ok"]:
            raise SystemExit(f"{path.name} failed byte verification: {check}")

    candidates, verdict = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)
    manifest = pd.concat([pd.read_parquet(role_root / f"role_{r}.parquet")
                          for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    poi = pd.concat([pd.read_parquet(poi_root / f"poi_{r}.parquet").rename(
        columns={"poi_index": "reference_unit_index"})
        for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))
    dtest_path = role_root / "locked" / "role_D-test.parquet"

    tiers = {}
    for floor in FLOORS_MS:
        spans = D5._spans_at_floor(candidates, dialogue_of, poi, floor)
        develop = spans[spans["role"] == "D-dev-select"]
        tiers[f"{int(floor)}ms"] = {"spans": int(len(develop)),
                                    "error_units": int(develop["units_error"].sum()),
                                    "dialogues": int(develop["dialogue_id"].nunique())}
    plan = {"frozen_configuration": FROZEN_CONFIG, "tiers": tiers,
            "directions_identity": identity_note,
            "d_test": {"path": str(dtest_path), "exists": dtest_path.exists(),
                       "use": "reference-only counts and C00 baseline only"},
            "destination": str(target_dir)}
    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    started = time.time()
    bundle = load_whisper(cfg)
    layer = int(FROZEN_ACTION["decoder_layer"])
    rho = float(FROZEN_ACTION["rho"])
    vector = directions[f"site_d_layer{layer}"]
    direction_norm = float(np.linalg.norm(vector))

    # nulls for Task 3, rebuilt through the identical Day 3 pipeline
    construct_spans = D5._spans_at_floor(candidates, dialogue_of, poi, 400.0)
    construct = construct_spans[(construct_spans["role"] == "D-construct")
                                & construct_spans["all_correct"]]
    build_manifest = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
    d_meta, d_raw = D3.site_d_contrasts(bundle, build_manifest, construct, [layer],
                                        batch_size=4, log=log)
    permuted = [n["direction"] for n in OS.label_permuted_directions(
        np.vstack(d_raw["vectors"][layer]), d_meta,
        draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + 500 + layer)]
    random_dirs = OS.matched_norm_random(vector, draws=CONTROL_RANDOM_DRAWS,
                                         seed=OS.CONTROL_SEED + 600)

    # ---- first pass, once per utterance, shared by every tier and both arms --
    widest = D5._spans_at_floor(candidates, dialogue_of, poi, min(FLOORS_MS))
    widest = widest[widest["role"] == "D-dev-select"]
    widest_targets = OS.build_targets(bundle, candidates, widest, poi, manifest)
    if args.limit:
        keep = list(dict.fromkeys(widest_targets["utterance_id"]))[: int(args.limit)]
        widest_targets = widest_targets[widest_targets["utterance_id"].isin(keep)]
    utts = manifest[manifest["utterance_id"].isin(set(widest_targets["utterance_id"]))]
    utts = utts.drop_duplicates("utterance_id").reset_index(drop=True)
    log.info("first pass over %d utterances", len(utts))

    from ..models.generation import teacher_forced_forward
    first_pass: dict[str, dict[str, Any]] = {}
    for _, row in utts.iterrows():
        utterance = str(row["utterance_id"])
        tokens, prefill, steps = free_decode_probed(bundle, row["audio_path"], layer)
        out, _, _ = teacher_forced_forward(bundle, [row["audio_path"]], [tokens],
                                           output_attentions=True)
        attn = out.cross_attentions[TRANSPORT_LAYER_INDEX].float().mean(dim=1)
        attention = attn[0].cpu().numpy()
        del out, attn
        text = bundle.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        first_pass[utterance] = {"tokens": tokens, "prefill": prefill, "steps": steps,
                                 "attention": attention, "baseline_text": text,
                                 "duration_sec": float(row["duration_sec"]),
                                 "audio_path": row["audio_path"],
                                 "reference": str(row["transcript_raw"])}
    log.info("first pass complete (%.1f s)", time.time() - started)

    # ---- both arms, every tier ----------------------------------------------
    records: list[dict[str, Any]] = []
    arm_runtime: dict[str, float] = {}
    for floor in FLOORS_MS:
        spans = D5._spans_at_floor(candidates, dialogue_of, poi, floor)
        spans = spans[spans["role"] == "D-dev-select"]
        tier_targets = OS.build_targets(bundle, candidates, spans, poi, manifest)
        tier_targets = tier_targets[tier_targets["utterance_id"].isin(first_pass)]
        tier_utts = sorted(set(tier_targets["utterance_id"].astype(str)))
        log.info("tier %.0f ms: %d targets over %d utterances",
                 floor, len(tier_targets), len(tier_utts))
        for utterance in tier_utts:
            cache = first_pass[utterance]
            group = tier_targets[tier_targets["utterance_id"] == utterance]
            region = candidate_region(bundle, cache["tokens"], cache["attention"],
                                      cache["duration_sec"], floor)
            r = transport_weights(cache["attention"], region["g"],
                                  prefill_width=cache["prefill"], steps=cache["steps"])
            has_region = bool(region["spans"])
            valid = max(1, int(region["valid_frames"]))
            share = float(region["g"].sum()) / float(valid) if region["g"].size else 0.0
            mass = step_valid_mass(cache["attention"], valid,
                                   prefill_width=cache["prefill"], steps=cache["steps"])
            # step selection is by r_q, as specified; the chance level for the
            # abstention test is that step's own in-audio mass times the region's
            # share of the audio, so both sides carry the same denominator
            selected_step = int(r.argmax()) if r.size else -1
            r_max = float(r.max()) if r.size else 0.0
            chance = (float(mass[selected_step]) * share) if (0 <= selected_step < len(mass)) else 0.0
            enrichment = (r_max / chance) if chance > 0 else 0.0
            abstained = (not has_region) or (
                enrichment <= ABSTENTION_ENRICHMENT + ABSTENTION_TIE_TOLERANCE)

            oracle_step = max(0, int(group["token_index"].min()) - 1)
            oracle_exists = oracle_step < int(cache["steps"])
            o_weights = oracle_weights(oracle_step, cache["steps"])

            arms: dict[str, dict[str, Any]] = {}
            t0 = time.time()
            o_text, o_fired, o_sq, _ = steered_decode(
                bundle, cache["audio_path"], vector, layer=layer, rho=rho,
                weights=o_weights)
            arm_runtime["oracle"] = arm_runtime.get("oracle", 0.0) + (time.time() - t0)
            arms["oracle"] = {"text": o_text, "fired": o_fired, "sum_sq": o_sq,
                              "energy": effective_energy(rho, direction_norm, o_weights)}

            t0 = time.time()
            if abstained:
                n_text, n_fired, n_sq = cache["baseline_text"], 0, 0.0
            else:
                n_text, n_fired, n_sq, _ = steered_decode(
                    bundle, cache["audio_path"], vector, layer=layer, rho=rho,
                    weights=r)
            arm_runtime["non_oracle"] = arm_runtime.get("non_oracle", 0.0) + (time.time() - t0)
            arms["non_oracle"] = {"text": n_text, "fired": n_fired, "sum_sq": n_sq,
                                  "energy": effective_energy(rho, direction_norm,
                                                             r if not abstained else np.zeros(0))}

            for _, t in group.iterrows():
                unit = int(t["reference_unit_index"])
                for arm, payload in arms.items():
                    rates = D5.outcome_rates(cache["reference"], cache["baseline_text"],
                                             payload["text"], unit)
                    risk = corruption_on_correct(cache["reference"],
                                                 cache["baseline_text"],
                                                 payload["text"], unit)
                    keep = monolingual_retention(cache["reference"],
                                                 cache["baseline_text"], payload["text"])
                    records.append({
                        "arm": arm, "floor_ms": float(floor),
                        "utterance_id": utterance,
                        "reference_unit_index": unit,
                        "dialogue_id": str(t["dialogue_id"]),
                        "category": str(t["category"]),
                        "is_deletion": bool(t["is_deletion"]),
                        "reference": cache["reference"],
                        "baseline_text": cache["baseline_text"],
                        "steered_text": payload["text"],
                        "hook_fired": int(payload["fired"]),
                        "sum_r_squared": float(payload["sum_sq"]),
                        "effective_energy": float(payload["energy"]),
                        "abstained": bool(abstained) if arm == "non_oracle" else False,
                        "has_candidate_region": has_region,
                        "r_max": r_max,
                        "chance_level": chance,
                        "region_share_of_audio": share,
                        "selected_step_valid_mass": float(
                            mass[selected_step]) if 0 <= selected_step < len(mass) else 0.0,
                        "enrichment": enrichment,
                        "abstain_at_tau_0p5": bool((not has_region) or (r_max < ABSTENTION_TAU_REPORTED)),
                        "hypothesis_has_english": bool(region["hypothesis_has_english"]),
                        "en_spans_any": int(region["en_spans_any"]),
                        "en_spans_above_floor": int(region["en_spans_above_floor"]),
                        "region_frames": float(region["g"].sum()),
                        "valid_frames": int(valid),
                        "selected_step": selected_step,
                        "oracle_step": oracle_step,
                        "oracle_step_exists": bool(oracle_exists),
                        "transport_hit": bool(
                            oracle_exists and (not abstained)
                            and abs(selected_step - oracle_step) <= TRANSPORT_HIT_TOLERANCE),
                        "n_steps": int(cache["steps"]),
                        **rates, **risk, **keep})
        log.info("  tier %.0f ms done (%.1f s elapsed)", floor, time.time() - started)

    free = pd.DataFrame(records)

    # ---- Task 3: controls under free decoding, conservative tier, non-oracle -
    log.info("controls under free decoding (conservative tier, non-oracle arm)")
    control_rows: list[dict[str, Any]] = []
    conservative = D5._spans_at_floor(candidates, dialogue_of, poi, 400.0)
    conservative = conservative[conservative["role"] == "D-dev-select"]
    c_targets = OS.build_targets(bundle, candidates, conservative, poi, manifest)
    c_targets = c_targets[c_targets["utterance_id"].isin(first_pass)]
    controls = ([(f"label_permuted_{k}", v) for k, v in enumerate(permuted)]
                + [(f"matched_energy_random_{k}", v) for k, v in enumerate(random_dirs)])
    for utterance in sorted(set(c_targets["utterance_id"].astype(str))):
        cache = first_pass[utterance]
        group = c_targets[c_targets["utterance_id"] == utterance]
        region = candidate_region(bundle, cache["tokens"], cache["attention"],
                                  cache["duration_sec"], 400.0)
        r = transport_weights(cache["attention"], region["g"],
                              prefill_width=cache["prefill"], steps=cache["steps"])
        valid = max(1, int(region["valid_frames"]))
        share = float(region["g"].sum()) / float(valid) if region["g"].size else 0.0
        mass = step_valid_mass(cache["attention"], valid,
                               prefill_width=cache["prefill"], steps=cache["steps"])
        pick = int(r.argmax()) if r.size else -1
        chance = (float(mass[pick]) * share) if (0 <= pick < len(mass)) else 0.0
        enrich = (float(r.max()) / chance) if chance > 0 else 0.0
        if (not region["spans"]) or enrich <= ABSTENTION_ENRICHMENT + ABSTENTION_TIE_TOLERANCE:
            continue                      # the arm abstains; controls abstain with it
        for name, control in controls:
            text, fired, sq, _ = steered_decode(bundle, cache["audio_path"], control,
                                                layer=layer, rho=rho, weights=r)
            for _, t in group.iterrows():
                unit = int(t["reference_unit_index"])
                rates = D5.outcome_rates(cache["reference"], cache["baseline_text"],
                                         text, unit)
                control_rows.append({
                    "control": name, "utterance_id": utterance,
                    "reference_unit_index": unit,
                    "dialogue_id": str(t["dialogue_id"]),
                    "category": str(t["category"]),
                    "hook_fired": int(fired), "sum_r_squared": float(sq),
                    "effective_energy": effective_energy(
                        rho, float(np.linalg.norm(control)), r),
                    "reference": cache["reference"],
                    "baseline_text": cache["baseline_text"],
                    "steered_text": text, **rates})
    control_df = pd.DataFrame(control_rows)

    # ---- D-test headroom -----------------------------------------------------
    headroom: dict[str, Any] = {"skipped": True}
    if not args.skip_dtest and dtest_path.exists():
        headroom = dtest_headroom(bundle, cfg, dtest_path,
                                  target_dir / "dtest_c00_baseline.parquet",
                                  batch_size=args.batch_size,
                                  limit=args.dtest_limit, log=log)

    target_dir.mkdir(parents=True, exist_ok=True)
    free.to_parquet(target_dir / "gate_b_free_decoding.parquet", index=False)
    if not control_df.empty:
        control_df.to_parquet(target_dir / "gate_b_controls.parquet", index=False)
    payload = {
        "frozen_configuration": FROZEN_CONFIG,
        "plan": plan,
        "direction_norm": direction_norm,
        "runtime_seconds": {**arm_runtime, "total": time.time() - started},
        "rows": int(len(free)),
        "control_rows": int(len(control_df)),
        "dtest_headroom": headroom,
        "selects_anything": False,
        "evaluates_gate": False,
    }
    (target_dir / "gate_b_report.json").write_text(
        json.dumps(payload, indent=2, default=str))
    parent = verdict.get("manifest")
    for name in ["gate_b_free_decoding.parquet", "gate_b_controls.parquet",
                 "gate_b_report.json", "dtest_c00_baseline.parquet"]:
        path = target_dir / name
        if path.exists():
            manifest_mod.publish(path, cfg=cfg, stage=STAGE,
                                 parents=[parent] if parent else (),
                                 schema="lss_v2r3_gate_b_v1",
                                 extra={"frozen_configuration": FROZEN_CONFIG,
                                        "logical_path": str(path),
                                        "evaluates_gate": False})
    print(json.dumps({"state": "completed", "rows": int(len(free)),
                      "destination": str(target_dir)}, indent=2, default=str))
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
