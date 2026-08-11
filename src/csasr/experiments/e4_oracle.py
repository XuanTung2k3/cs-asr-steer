"""E4 — oracle-span local steering: the decisive go/no-go experiment.

Run:
    python -m csasr.experiments.e4_oracle --config configs/experiments/e4_oracle.yaml \
        --phase pilot --resume
    python -m csasr.experiments.e4_oracle --config configs/experiments/e4_oracle.yaml \
        --phase confirm --resume
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.alignment_checks import frame_rms
from ..data.splits import load_subset
from ..directions.encoder import load_direction
from ..evaluation.bootstrap import paired_bootstrap
from ..evaluation.correction_harm import (
    correction_harm_table,
    per_category_correction,
    summarize_correction_harm,
)
from ..evaluation.mer import corpus_mer
from ..evaluation.pier import LANGUAGE_CONFUSION, annotate_unit_status, pier
from ..models.generation import decode_manifest
from ..models.whisper import load_audio, load_whisper
from ..steering.decoder_hook import DecoderAllLayerSteering
from ..steering.encoder_hook import EncoderLocalSteering
from ..steering.masks import derive_jitter_seed, spec_from_kind
from ..utils.config import art
from ..utils.logging import get_logger
from ..utils.status import StageLock, criterion, gate, require_passed
from ._common import (
    base_parser,
    baseline_predictions,
    finish,
    load_units,
    md_table,
    prepare,
    primary_baseline_id,
    save_report,
)

STAGE = "e4"
BASELINE_LANGUAGE = {"B0_AUTO": None, "B1_ZH": "zh"}
log = get_logger(__name__)


def phase_stage(phase: str) -> str:
    return f"{STAGE}_{phase}"


def phase_metric_limits(ecfg: dict, phase: str) -> dict:
    """Return fraction-scale absolute degradation limits for an E4 phase."""
    strict = phase in {"refine", "confirm"}
    return {
        "max_mer_degradation_abs": float(
            ecfg.get("confirm_max_mer_degradation_abs" if strict else
                     "pilot_max_mer_degradation_abs",
                     ecfg.get("max_mer_degradation_abs", 0.002 if strict else 0.005))),
        "max_zh_cer_degradation_abs": float(
            ecfg.get("confirm_max_zh_cer_degradation_abs" if strict else
                     "pilot_max_zh_cer_degradation_abs",
                     ecfg.get("max_zh_cer_degradation_abs", 0.002 if strict else 0.005))),
        "max_outside_region_edit_rate": float(
            ecfg.get("max_outside_region_edit_rate",
                     ecfg.get("max_outside_edit_rate", 0.05))),
    }


def fraction_delta(baseline: float, result: float) -> float:
    """Metrics are stored as fractions; 0.2342 - 0.2322 == 0.002."""
    return float(result) - float(baseline)


def percentage_point_label(delta: float) -> str:
    return f"{float(delta) * 100:+.2f} percentage points"


def degradation_within_limit(baseline: float, result: float, limit: float) -> bool:
    return fraction_delta(baseline, result) <= float(limit) + 1e-12


def correction_ratio_gate_pass(metrics: dict | pd.Series, min_ratio: float) -> bool:
    corrected = int(metrics.get("num_corrected", metrics.get("raw_corrected_count", 0)))
    corrupted = int(metrics.get("num_corrupted", metrics.get("raw_corrupted_count", 0)))
    ratio = metrics.get("correction_to_corruption_ratio", float("nan"))
    if corrupted == 0:
        return corrected > 0
    return bool(np.isfinite(float(ratio)) and float(ratio) > float(min_ratio))


# ---------------------------------------------------------------------------
# evaluation groups
# ---------------------------------------------------------------------------
def poi_candidates(cfg, subset: str, log) -> pd.DataFrame:
    """Every aligned English POI on ``subset`` with its baseline status."""
    manifest = load_subset(cfg, subset)
    preds = baseline_predictions(cfg, subset)
    units = annotate_unit_status(load_units(cfg, subset), manifest, preds)
    en = units[(units["language_tag"] == "EN") & units["is_content"]
               & (units["baseline_status"] != "")].copy()

    order = units.sort_values(["utterance_id", "unit_id"])
    prev_end, next_start = {}, {}
    for utt, grp in order.groupby("utterance_id"):
        g = grp.reset_index(drop=True)
        for i, r in g.iterrows():
            prev_end[(utt, int(r["unit_id"]))] = int(g.iloc[i - 1]["start_frame"]) if i > 0 else int(r["start_frame"])
            next_start[(utt, int(r["unit_id"]))] = int(g.iloc[i + 1]["end_frame"]) if i + 1 < len(g) else int(r["end_frame"])

    en["context_start"] = [prev_end[(u, int(i))] for u, i in zip(en["utterance_id"], en["unit_id"])]
    en["context_end"] = [next_start[(u, int(i))] for u, i in zip(en["utterance_id"], en["unit_id"])]
    zh_spans = {
        utt: [(int(r["start_frame"]), int(r["end_frame"]))
              for _, r in grp.iterrows()]
        for utt, grp in units[(units["language_tag"] == "ZH") & units["is_content"]].groupby("utterance_id")
    }
    en["zh_spans"] = [zh_spans.get(u, []) for u in en["utterance_id"]]
    en = en.merge(manifest[["utterance_id", "speaker_id", "audio_path", "duration_sec",
                            "transcript_raw"]], on="utterance_id", how="left")
    en = en.merge(preds[["utterance_id", "hypothesis_raw", "avg_logprob"]],
                  on="utterance_id", how="left")
    en["poi_index"] = en["unit_id"].astype(int)
    en["poi_duration_sec"] = en["end_sec"] - en["start_sec"]
    en["poi_position"] = en.groupby("utterance_id").cumcount()
    log.info("%s: %d aligned EN POIs (%d incorrect)", subset, len(en),
             int((en["baseline_status"] == "incorrect").sum()))
    return en


def _match_controls(wrong: pd.DataFrame, correct: pd.DataFrame, n: int,
                    seed: int) -> pd.DataFrame:
    """Greedy nearest-neighbour matching on the covariates of section 36."""
    feats = ["poi_duration_sec", "poi_position", "duration_sec", "avg_logprob"]
    if not len(correct) or not len(wrong):
        return correct.head(n)
    pool = correct.copy().reset_index(drop=True)
    mu = pd.concat([wrong[feats], pool[feats]]).astype(float)
    mean, std = mu.mean(), mu.std().replace(0, 1.0)
    W = ((wrong[feats].astype(float) - mean) / std).to_numpy()
    P = ((pool[feats].astype(float) - mean) / std).to_numpy()
    spk_w = wrong["speaker_id"].to_numpy()
    spk_p = pool["speaker_id"].to_numpy()

    taken = np.zeros(len(pool), dtype=bool)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(W))
    picks = []
    for i in order[:n]:
        cost = np.linalg.norm(P - W[i], axis=1)
        cost = np.where(np.isnan(cost), 1e6, cost)
        cost = cost + (spk_p != spk_w[i]) * 1.0     # prefer the same speaker
        cost[taken] = np.inf
        j = int(np.argmin(cost))
        if not np.isfinite(cost[j]):
            break
        taken[j] = True
        picks.append(j)
    return pool.iloc[picks].reset_index(drop=True)


def build_groups(cfg, subset: str, n_wrong: int, n_correct: int, seed: int,
                 log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One target POI per utterance; controls matched to the error set."""
    cand = poi_candidates(cfg, subset, log)
    cand = cand.sample(frac=1.0, random_state=seed).drop_duplicates("utterance_id")
    wrong_pool = cand[cand["baseline_status"] == "incorrect"]
    correct_pool = cand[cand["baseline_status"] == "correct"]
    diagnostics = {
        "subset": subset,
        "one_poi_per_utterance_pool": int(len(cand)),
        "eligible_wrong_after_one_poi_per_utterance": int(len(wrong_pool)),
        "eligible_correct_after_one_poi_per_utterance": int(len(correct_pool)),
        "requested_wrong": int(n_wrong),
        "requested_correct": int(n_correct),
    }
    wrong = wrong_pool.sample(min(n_wrong, len(wrong_pool)), random_state=seed).reset_index(drop=True)
    correct = _match_controls(wrong, correct_pool, n_correct, seed)
    diagnostics.update({
        "selected_wrong": int(len(wrong)),
        "selected_correct": int(len(correct)),
        "shortfall_reason": (
            "one-POI-per-utterance plus baseline-error filtering left fewer "
            "eligible wrong POIs than requested"
            if len(wrong_pool) < n_wrong else ""
        ),
    })
    wrong.attrs["target_diagnostics"] = diagnostics
    correct.attrs["target_diagnostics"] = diagnostics
    return wrong, correct


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------
def _mask_fraction(mask: np.ndarray | None, start: int, width: int) -> float:
    if mask is None or width <= 0 or start < 0 or start + width > len(mask):
        return 0.0
    return float(np.asarray(mask[start:start + width], dtype=bool).mean())


def randloc_candidate_starts(n_valid: int, width: int, excluded_start: int,
                             excluded_end: int, *,
                             speech_mask: np.ndarray | None = None,
                             preferred_mask: np.ndarray | None = None) -> tuple[list[int], list[int]]:
    """Candidate control starts, with a preferred language-region subset."""
    candidates: list[int] = []
    preferred: list[int] = []
    for c in range(0, max(0, n_valid - width + 1)):
        if not (c + width <= excluded_start or c >= excluded_end):
            continue
        if speech_mask is not None and _mask_fraction(speech_mask, c, width) < 1.0:
            continue
        candidates.append(c)
        if preferred_mask is not None and _mask_fraction(preferred_mask, c, width) >= 1.0:
            preferred.append(c)
    return candidates, preferred


def _zh_mask_from_spans(spans, n_valid: int) -> np.ndarray | None:
    if not spans:
        return None
    mask = np.zeros(n_valid, dtype=bool)
    for s, e in spans:
        mask[max(0, int(s)): min(n_valid, int(e))] = True
    return mask


def _speech_mask_for_row(row: pd.Series, n_valid: int) -> np.ndarray | None:
    path = str(row.get("audio_path", ""))
    if not path:
        return None
    try:
        audio = load_audio(path)
        rms = frame_rms(audio, 16000)
    except Exception:
        return None
    if not len(rms):
        return None
    thr = float(0.05 * np.percentile(rms, 95))
    mask = np.asarray(rms > thr, dtype=bool)
    if len(mask) < n_valid:
        mask = np.pad(mask, (0, n_valid - len(mask)), constant_values=False)
    return mask[:n_valid]


def relocated_spans(bundle, targets: pd.DataFrame, seed: int,
                    min_gap_frames: int = 0,
                    shoulder_frames: int = 0) -> tuple[dict, int]:
    """Equal-width spans placed elsewhere in the same utterance (ENC-RANDLOC-1L).

    The control holds direction, strength and mask width fixed and moves only the
    *location*. If steering the aligned span beats steering an identical span
    somewhere else, the span location carries real information -- direct evidence
    on the task itself that the alignment points at the right frames.

    Returns the span map and the number of utterances too short to relocate.
    """
    out: dict[str, tuple[int, int]] = {}
    unrelocated = 0
    for _, r in targets.iterrows():
        utt = str(r["utterance_id"])
        s, e = int(r["start_frame"]), int(r["end_frame"])
        width = max(1, e - s)
        n_valid = bundle.valid_frames(float(r["duration_sec"]))
        rng = np.random.default_rng(derive_jitter_seed(seed, utt))
        excluded_start = max(0, s - max(min_gap_frames, shoulder_frames))
        excluded_end = min(n_valid, e + max(min_gap_frames, shoulder_frames))
        speech_mask = _speech_mask_for_row(r, n_valid)
        preferred_mask = _zh_mask_from_spans(r.get("zh_spans", []), n_valid)
        candidates, preferred = randloc_candidate_starts(
            n_valid, width, excluded_start, excluded_end,
            speech_mask=speech_mask, preferred_mask=preferred_mask)
        pool = preferred or candidates
        if pool:
            start = int(rng.choice(pool))
        else:                       # utterance leaves no disjoint room; keep as-is
            start = max(0, min(s, max(0, n_valid - width)))
            unrelocated += 1
        out[utt] = (start, start + width)
    return out, unrelocated


def encoder_hook_for(bundle, cfg, system: str, layer: int, alpha: float,
                     targets: pd.DataFrame, mask_kind: str, seed: int) -> EncoderLocalSteering:
    ecfg = cfg["e4"]
    dir_root = art(cfg, "directions")
    variant = ecfg.get("direction_variant", "within_utterance_correct_only")
    dseed = int(ecfg.get("direction_seed", 42))
    if system == "ENC-WRONG-1L":
        rec = load_direction(dir_root, "encoder", layer, "wrong_sign", dseed)
    elif system == "ENC-RANDLOC-1L":          # primary direction, wrong place
        rec = load_direction(dir_root, "encoder", layer, variant, dseed)
    elif system.startswith("ENC-RAND-1L"):
        rec = load_direction(dir_root, "encoder", layer, f"random_s{seed}", dseed)
    else:
        rec = load_direction(dir_root, "encoder", layer, variant, dseed)

    if system == "ENC-RANDLOC-1L":
        spans, unrelocated = relocated_spans(
            bundle, targets, int(cfg["experiment"]["seed"]),
            min_gap_frames=int(ecfg.get("randloc_min_gap_frames", 0)),
            shoulder_frames=int(round((float(ecfg.get("shoulder_ms", 100)) / 1000.0)
                                      / bundle.encoder_step_sec)))
        if unrelocated:
            log.warning("ENC-RANDLOC-1L: %d/%d utterances too short to relocate the "
                        "span disjointly; those keep the original location and make "
                        "the control conservative", unrelocated, len(targets))
    else:
        spans = {r["utterance_id"]: (int(r["start_frame"]), int(r["end_frame"]))
                 for _, r in targets.iterrows()}
    contexts = {r["utterance_id"]: (int(r["context_start"]), int(r["context_end"]))
                for _, r in targets.iterrows()}
    kind = "GLOBAL" if system == "ENC-GLOBAL-1L" else mask_kind
    spec = spec_from_kind(kind, bundle.encoder_step_sec,
                          shoulder_ms=float(ecfg.get("shoulder_ms", 100)))
    return EncoderLocalSteering(bundle, layer, rec["direction"], alpha,
                                float(rec["projection_std"]), spec, spans, contexts,
                                norm_preserve=bool(ecfg.get("norm_preserve", True)))


def decoder_hook_for(bundle, cfg, alpha: float) -> DecoderAllLayerSteering:
    dir_root = art(cfg, "directions")
    dseed = int(cfg["e4"].get("direction_seed", 42))
    directions, scales = {}, {}
    for k in range(bundle.num_decoder_layers):
        try:
            rec = load_direction(dir_root, "decoder", k, "decoder_within_utterance", dseed)
        except FileNotFoundError:
            continue
        directions[k] = rec["direction"]
        scales[k] = float(rec["projection_std"])
    return DecoderAllLayerSteering(bundle, directions, scales, alpha,
                                   norm_preserve=bool(cfg["e4"].get("norm_preserve", True)))


def run_system(bundle, cfg, targets: pd.DataFrame, system: str, layer: int | None,
               alpha: float, mask_kind: str, seed: int, phase: str, log,
               resume: bool, overwrite: bool) -> pd.DataFrame:
    config_id = f"L{layer}_a{alpha}_{mask_kind}" if layer is not None else f"a{alpha}"
    if system.startswith("ENC-RAND-1L"):
        config_id += f"_s{seed}"
    out_path = art(cfg, "predictions", "e4", system, phase, f"{config_id}.parquet")
    if system == "DEC-ALL-GLOBAL":
        builder = decoder_hook_for(bundle, cfg, alpha)
    else:
        builder = encoder_hook_for(bundle, cfg, system, layer, alpha, targets, mask_kind, seed)
    language = BASELINE_LANGUAGE.get(primary_baseline_id(cfg))
    preds = decode_manifest(bundle, targets, cfg, system=system, config_id=config_id,
                            out_path=out_path, language=language, hook_builder=builder,
                            resume=resume, overwrite=overwrite)
    if hasattr(builder, "applied_meta") and builder.applied_meta:
        pd.DataFrame(builder.applied_meta.values()).to_parquet(
            out_path.with_name(out_path.stem + "_masks.parquet"), index=False)
    return preds


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_system(cfg, wrong: pd.DataFrame, correct: pd.DataFrame,
                 preds: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    from ..evaluation.correction_harm import EMPTY_SUMMARY

    targets = pd.concat([wrong, correct]).reset_index(drop=True)
    if not len(preds) or not len(targets):
        return ({**EMPTY_SUMMARY, "pier": float("nan"), "pier_baseline": float("nan"),
                 "pier_delta": float("nan"), "pier_relative_gain": float("nan"),
                 "mer": float("nan"), "mer_baseline": float("nan"),
                 "mer_delta": float("nan"), "en_wer": float("nan"),
                 "en_wer_baseline": float("nan"), "zh_cer": float("nan"),
                 "zh_cer_baseline": float("nan"), "zh_cer_delta": float("nan"),
                 "paired_poi_error_diff": float("nan"), "paired_ci_low": float("nan"),
                 "paired_ci_high": float("nan"), "paired_net_error_reduction": float("nan"),
                 "paired_net_reduction_ci_low": float("nan"),
                 "paired_net_reduction_ci_high": float("nan"),
                 "paired_p_value": float("nan"),
                 "per_category_correction": {}, "num_scored": 0},
                pd.DataFrame())
    merged = targets.merge(preds[["utterance_id", "hypothesis_raw"]], on="utterance_id",
                           how="inner", suffixes=("_base", "_steered"))
    rows = [{
        "utterance_id": r["utterance_id"],
        "speaker_id": r["speaker_id"],
        "reference": r["transcript_raw"],
        "poi_index": int(r["poi_index"]),
        "baseline_hypothesis": r["hypothesis_raw_base"],
        "steered_hypothesis": r["hypothesis_raw_steered"],
        "baseline_correct": r["baseline_status"] == "correct",
        "baseline_category": r["baseline_error_type"],
    } for _, r in merged.iterrows()]
    table = correction_harm_table(rows)
    summary = summarize_correction_harm(table)

    refs = merged["transcript_raw"].tolist()
    base_hyp = merged["hypothesis_raw_base"].tolist()
    steer_hyp = merged["hypothesis_raw_steered"].tolist()
    poi_idx = [[int(i)] for i in merged["poi_index"]]
    base_mer, steer_mer = corpus_mer(refs, base_hyp), corpus_mer(refs, steer_hyp)
    base_pier = pier(refs, base_hyp, poi_idx)
    steer_pier = pier(refs, steer_hyp, poi_idx)

    base_err = np.array([0.0 if c else 1.0 for c in table["baseline_correct"]])
    steer_err = np.array([0.0 if c else 1.0 for c in table["steered_correct"]])
    boot = paired_bootstrap(base_err, steer_err, table["speaker_id"].to_numpy(),
                            n_resamples=bootstrap, seed=seed)
    net_reduction = -float(boot["diff"])
    return {
        **summary,
        "pier": steer_pier["pier"], "pier_baseline": base_pier["pier"],
        "pier_delta": steer_pier["pier"] - base_pier["pier"],
        "pier_relative_gain": ((base_pier["pier"] - steer_pier["pier"]) / base_pier["pier"]
                               if base_pier["pier"] else float("nan")),
        "mer": steer_mer["mer"], "mer_baseline": base_mer["mer"],
        "mer_delta": steer_mer["mer"] - base_mer["mer"],
        "en_wer": steer_mer["en_wer"], "en_wer_baseline": base_mer["en_wer"],
        "zh_cer": steer_mer["zh_cer"], "zh_cer_baseline": base_mer["zh_cer"],
        "zh_cer_delta": steer_mer["zh_cer"] - base_mer["zh_cer"],
        "paired_poi_error_diff": boot["diff"],
        "paired_ci_low": boot["ci_low"], "paired_ci_high": boot["ci_high"],
        "paired_net_error_reduction": net_reduction,
        "paired_net_reduction_ci_low": -float(boot["ci_high"])
        if np.isfinite(float(boot["ci_high"])) else float("nan"),
        "paired_net_reduction_ci_high": -float(boot["ci_low"])
        if np.isfinite(float(boot["ci_low"])) else float("nan"),
        "paired_p_value": boot["p_value_two_sided"],
        "per_category_correction": per_category_correction(table),
        "num_scored": int(len(table)),
    }, table


def feasible(row: dict, cfg: dict, random_mean: float | None, phase: str = "pilot") -> bool:
    e = cfg["e4"]
    limits = phase_metric_limits(e, phase)
    if random_mean is None or not np.isfinite(random_mean):
        return False
    return bool(
        row["pier_relative_gain"] >= e["min_relative_pier_gain"]
        and correction_ratio_gate_pass(row, e["min_correction_corruption_ratio"])
        and row["mer_delta"] <= limits["max_mer_degradation_abs"]
        and row["zh_cer_delta"] <= limits["max_zh_cer_degradation_abs"]
        and row["outside_edit_rate"] <= limits["max_outside_region_edit_rate"]
        and row["correction_rate"] > random_mean)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("E4 oracle-span local steering", "experiments/e4_oracle.yaml")
    parser.add_argument("--phase", choices=["pilot", "refine", "confirm"], default="pilot")
    args = parser.parse_args(argv)
    stage_key = phase_stage(args.phase)
    cfg, rdir, log = prepare(args, stage_key)
    root = cfg["experiment"]["output_root"]
    ecfg = cfg["e4"]
    phase = args.phase
    seed = int(cfg["experiment"]["seed"])
    resume, overwrite = args.resume or not args.overwrite, args.overwrite

    with StageLock(root, stage_key):
        if not args.force_prereq:
            prereqs = {
                "pilot": ["e1", "e2_pilot", "e3_pilot"],
                "refine": ["e1", "e2_full", "e3_full", "e4_pilot"],
                "confirm": ["e1", "e2_full", "e3_full", "e4_refine"],
            }[phase]
            require_passed(root, prereqs)
        selected_path = art(cfg, "reports", "e3_selected_layers.json")
        e3_layers = json.loads(selected_path.read_text(encoding="utf-8"))["selected_layers"]

        subset = "dev_confirm" if phase == "confirm" else "dev_select"
        n_wrong = ecfg["pilot_wrong"] if phase == "pilot" else ecfg["confirm_wrong"]
        n_correct = ecfg["pilot_correct"] if phase == "pilot" else ecfg["confirm_correct"]
        wrong, correct = build_groups(cfg, subset, n_wrong, n_correct, seed, log)
        target_diagnostics = dict(wrong.attrs.get("target_diagnostics", {}))
        if args.limit:          # smoke-only knob; never set in production runs
            wrong, correct = wrong.head(args.limit), correct.head(args.limit)
            target_diagnostics["limit_applied"] = int(args.limit)
        for name, df in (("wrong", wrong), ("correct", correct)):
            df.to_parquet(art(cfg, "manifests", f"e4_{phase}_{name}.parquet"), index=False)
        targets = pd.concat([wrong, correct]).reset_index(drop=True)
        log.info("phase=%s subset=%s targets=%d (%d wrong / %d correct)",
                 phase, subset, len(targets), len(wrong), len(correct))
        if args.dry_run:
            return 0

        bundle = load_whisper(cfg)
        mask_kind = ecfg.get("primary_mask", "EXACT_TAPER")
        rows: list[dict] = []
        tables: dict[str, pd.DataFrame] = {}

        def evaluate(system, layer, alpha, s_seed=0):
            preds = run_system(bundle, cfg, targets, system, layer, alpha, mask_kind,
                               s_seed, phase, log, resume, overwrite)
            metrics, table = score_system(
                cfg, wrong, correct, preds,
                int(ecfg.get("bootstrap_resamples", 10000)),
                int(ecfg.get("bootstrap_seed", 342)))
            key = f"{system}|L{layer}|a{alpha}|s{s_seed}"
            tables[key] = table
            rows.append({"phase": phase, "system": system, "layer": layer,
                         "alpha": alpha, "control_seed": s_seed, "mask": mask_kind,
                         **metrics})
            log.info(
                "%s: correction=%.3f corruption=%.3f ratio=%.2f pier=%.4f mer_delta=%+.4f",
                key, metrics["correction_rate"], metrics["corruption_rate"],
                metrics["correction_to_corruption_ratio"], metrics["pier"],
                metrics["mer_delta"])
            return metrics

        if phase in ("confirm", "refine"):
            frozen_path = art(cfg, "metrics", "e4_selected_config.json")
            if not frozen_path.exists():
                raise SystemExit(
                    f"phase={phase} needs a configuration frozen by the previous E4 phase, "
                    f"but {frozen_path} does not exist. The earlier phase found no feasible "
                    "configuration; there is nothing to refine or confirm.")
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            layers = [frozen["layer"]]
            alphas = [frozen["alpha"]] if phase == "confirm" else ecfg["encoder_alphas_refine"]
        else:
            layers, alphas = e3_layers, ecfg["encoder_alphas_pilot"]

        for layer in layers:
            for alpha in alphas:
                evaluate("ENC-LOCAL-1L", layer, float(alpha))

        # Matched controls for every grid point used in feasibility selection.
        for layer in layers:
            for alpha in alphas:
                for system in ("ENC-GLOBAL-1L", "ENC-WRONG-1L", "ENC-RANDLOC-1L"):
                    if system in ecfg["systems"]:
                        evaluate(system, layer, float(alpha))
                if "ENC-RAND-1L" in ecfg["systems"]:
                    for rs in ecfg["random_seeds"]:
                        evaluate("ENC-RAND-1L", layer, float(alpha), s_seed=int(rs))
        if "DEC-ALL-GLOBAL" in ecfg["systems"]:
            for a in ecfg["decoder_alphas"]:
                evaluate("DEC-ALL-GLOBAL", None, a)

        results = pd.DataFrame(rows)
        # per-category counts are a variable-key dict: store as JSON so the
        # parquet schema stays stable across runs
        if "per_category_correction" in results.columns:
            results["per_category_correction"] = results["per_category_correction"].map(
                lambda v: json.dumps(v, default=str))
        all_path = art(cfg, "metrics", "e4_all_runs.parquet")
        if all_path.exists() and resume:
            prev = pd.read_parquet(all_path)
            results = pd.concat([prev[prev["phase"] != phase], results], ignore_index=True)
        results.to_parquet(all_path, index=False)
        for key, tb in tables.items():
            safe = key.replace("|", "__").replace("/", "_")
            tb.to_parquet(art(cfg, "predictions", "e4", "_tables", phase, f"{safe}.parquet"),
                          index=False)

        # ---- configuration selection ----------------------------------
        cur = results[results["phase"] == phase]
        local = cur[cur["system"] == "ENC-LOCAL-1L"].copy()

        def matching_random_mean(row):
            matched = cur[(cur["system"] == "ENC-RAND-1L")
                          & (cur["layer"] == row["layer"])
                          & np.isclose(cur["alpha"].astype(float), float(row["alpha"]))]
            return float(matched["correction_rate"].mean()) if len(matched) else None

        local["random_control_mean"] = [matching_random_mean(r) for _, r in local.iterrows()]
        local["feasible"] = [feasible(r, cfg, r["random_control_mean"], phase)
                             for _, r in local.iterrows()]
        feas = local[local["feasible"]].sort_values(
            ["pier", "net_corrected_pois", "corruption_rate", "alpha", "layer"],
            ascending=[True, False, True, True, False])

        selected = None
        if len(feas):
            top = feas.iloc[0]
            selected = {"layer": int(top["layer"]), "alpha": float(top["alpha"]),
                        "mask": mask_kind, "phase": phase,
                        "pier": float(top["pier"]),
                        "correction_rate": float(top["correction_rate"]),
                        "corruption_rate": float(top["corruption_rate"]),
                        "selected_on": subset}
            if phase != "confirm":
                art(cfg, "metrics", "e4_selected_config.json").write_text(
                    json.dumps(selected, indent=2, default=str), encoding="utf-8")

        # ---- gate: evaluate the selected feasible configuration only --
        best = feas.iloc[0] if len(feas) else None
        # Guide section 36 states the pilot size firmly ("100 utterances with one
        # baseline-incorrect EN POI") but bounds confirmation loosely ("*up to* 300").
        # A confirmation phase must therefore not fail merely because the corpus
        # offers fewer than 300 eligible POIs. The floor is the pilot size: a
        # confirmation run has to be at least as well powered as the pilot it confirms.
        if phase == "pilot":
            required_wrong = int(n_wrong)
        else:
            required_wrong = int(ecfg.get("min_confirm_wrong", n_wrong))
        # Controls are matched 1:1 against the error set, so more controls than
        # targets is impossible by construction; requiring n_correct outright made
        # this criterion unsatisfiable whenever wrong < n_correct.
        required_correct = min(n_correct, len(wrong))
        criteria = [
            criterion("baseline_incorrect_target_count", len(wrong), required_wrong,
                      len(wrong) >= required_wrong),
            criterion("matched_correct_target_count", len(correct), required_correct,
                      len(correct) >= required_correct),
            criterion("required_e4_systems_present",
                      sorted(set(cur["system"])) if "system" in cur else [],
                      ["DEC-ALL-GLOBAL", "ENC-GLOBAL-1L", "ENC-LOCAL-1L",
                       "ENC-RAND-1L", "ENC-RANDLOC-1L", "ENC-WRONG-1L"],
                      {"DEC-ALL-GLOBAL", "ENC-GLOBAL-1L", "ENC-LOCAL-1L",
                       "ENC-RAND-1L", "ENC-RANDLOC-1L", "ENC-WRONG-1L"}.issubset(
                          set(cur["system"])) if "system" in cur else False, "contains"),
            criterion("five_random_control_seeds",
                      int(cur[cur["system"] == "ENC-RAND-1L"]["control_seed"].nunique())
                      if "system" in cur else 0,
                      5,
                      int(cur[cur["system"] == "ENC-RAND-1L"]["control_seed"].nunique())
                      >= 5 if "system" in cur else False),
        ]
        if best is not None:
            same = lambda system: cur[(cur["system"] == system)
                                      & (cur["layer"] == best["layer"])
                                      & np.isclose(cur["alpha"].astype(float),
                                                   float(best["alpha"]))]
            wrong_rows = same("ENC-WRONG-1L")
            global_rows = same("ENC-GLOBAL-1L")
            dec_rows = cur[cur["system"] == "DEC-ALL-GLOBAL"]
            wrong_rate = float(wrong_rows["correction_rate"].mean()) if len(wrong_rows) else np.nan
            random_mean = float(best["random_control_mean"])
            collateral_limit = min(
                float(global_rows["outside_edit_rate"].min()) if len(global_rows) else np.inf,
                float(dec_rows["outside_edit_rate"].min()) if len(dec_rows) else np.inf)
            local_outside = float(best["outside_edit_rate"])
            limits = phase_metric_limits(ecfg, phase)
            criteria += [
                criterion("relative_pier_gain", float(best["pier_relative_gain"]),
                          ecfg["min_relative_pier_gain"],
                          best["pier_relative_gain"] >= ecfg["min_relative_pier_gain"]),
                criterion("correction_to_corruption_ratio",
                          (float(best["correction_to_corruption_ratio"])
                           if np.isfinite(float(best["correction_to_corruption_ratio"]))
                           else best.get("correction_to_corruption_ratio_note", "undefined")),
                          ecfg["min_correction_corruption_ratio"],
                          correction_ratio_gate_pass(best, ecfg["min_correction_corruption_ratio"]), ">"),
                criterion("beats_wrong_sign", float(best["correction_rate"]),
                          wrong_rate, bool(np.isfinite(wrong_rate)
                                           and best["correction_rate"] > wrong_rate)),
                criterion("beats_matched_random_controls", float(best["correction_rate"]),
                          random_mean, best["correction_rate"] > random_mean),
                criterion("less_collateral_than_global_controls", local_outside,
                          collateral_limit, bool(np.isfinite(collateral_limit)
                                                 and local_outside <= collateral_limit), "<="),
                criterion("overall_mer_delta", float(best["mer_delta"]),
                          limits["max_mer_degradation_abs"],
                          best["mer_delta"] <= limits["max_mer_degradation_abs"], "<="),
                criterion("outside_region_edit_rate", local_outside,
                          limits["max_outside_region_edit_rate"],
                          local_outside <= limits["max_outside_region_edit_rate"], "<="),
                criterion("matrix_language_preserved (ZH-CER delta)",
                          float(best["zh_cer_delta"]),
                          limits["max_zh_cer_degradation_abs"],
                          best["zh_cer_delta"] <= limits["max_zh_cer_degradation_abs"], "<="),
            ]
        g = gate(f"e4_{phase}", best is not None and all(c["passed"] for c in criteria), criteria,
                 "If E4 fails, do not proceed to a router: report the negative result, and "
                 "inspect layer/strength/mask/direction before any further scaling.")
        g["selected_config"] = selected
        g["target_diagnostics"] = target_diagnostics
        g["phase_metric_limits"] = phase_metric_limits(ecfg, phase)

        # ---- span-localization evidence (advisory, not gated) ----------
        # Under the no-human alignment protocol this is the empirical answer to
        # "how do you know the spans are in the right place?".
        localization: dict = {}
        if best is not None and "ENC-RANDLOC-1L" in ecfg["systems"]:
            rl = cur[(cur["system"] == "ENC-RANDLOC-1L")
                     & (cur["layer"] == best["layer"])
                     & np.isclose(cur["alpha"].astype(float), float(best["alpha"]))]
            if len(rl):
                rl_rate = float(rl["correction_rate"].mean())
                localization = {
                    "local_correction_rate": float(best["correction_rate"]),
                    "random_location_correction_rate": rl_rate,
                    "localization_advantage": float(best["correction_rate"]) - rl_rate,
                    "local_outside_edit_rate": float(best["outside_edit_rate"]),
                    "random_location_outside_edit_rate": float(rl["outside_edit_rate"].mean()),
                    "beats_random_location": bool(best["correction_rate"] > rl_rate),
                    "note": ("identical direction, strength and mask width applied at a "
                             "random disjoint span in the same utterance. An advantage "
                             "here is direct evidence that the aligned span location "
                             "carries information, independent of any boundary audit. "
                             "Advisory: not part of the guide section 45 gate."),
                }
                log.info("span localization: local=%.3f random-location=%.3f (advantage %+.3f)",
                         localization["local_correction_rate"], rl_rate,
                         localization["localization_advantage"])
        g["span_localization"] = localization

        report_cols = ["system", "layer", "alpha", "pier", "mer", "correction_rate",
                       "corruption_rate", "correction_to_corruption_ratio",
                       "num_corrected", "num_corrupted", "net_corrected_pois",
                       "correction_rate_minus_corruption_rate",
                       "outside_edit_rate", "zh_cer_delta"]
        save_report(
            art(cfg, "reports", f"e4_oracle_steering_{phase}.md"),
            f"E4 — oracle-span local steering ({phase}, {subset})",
            [
                ("Primary systems", md_table(cur[cur["system"] == "ENC-LOCAL-1L"][report_cols])),
                ("Location and confound controls",
                 md_table(cur[cur["system"] != "ENC-LOCAL-1L"][report_cols])),
                ("Selected configuration",
                 "```json\n" + json.dumps(selected, indent=2, default=str) + "\n```"),
                ("Target diagnostics",
                 "```json\n" + json.dumps(target_diagnostics, indent=2, default=str) + "\n```"),
                ("Metric units and gates",
                 "```json\n" + json.dumps({
                     "stored_as": "fractions",
                     "phase_limits": phase_metric_limits(ecfg, phase),
                     "example_0.2322_to_0.2342_delta": fraction_delta(0.2322, 0.2342),
                     "example_label": percentage_point_label(fraction_delta(0.2322, 0.2342)),
                 }, indent=2, default=str) + "\n```"),
                ("Span localization (alignment evidence, advisory)",
                 "```json\n" + json.dumps(localization or
                                          {"note": "ENC-RANDLOC-1L not run"},
                                          indent=2, default=str) + "\n```"),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, stage_key, rdir, {"phase": phase,
                                      "results": cur.to_dict(orient="records"),
                                      "selected": selected,
                                      "target_diagnostics": target_diagnostics}, g)
        log.info("E4 (%s) gate: %s", phase, "PASSED" if g["passed"] else "FAILED")
        return 0 if g["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
