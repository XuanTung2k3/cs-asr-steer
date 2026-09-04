"""Causal transport probe using baseline, gold, and partially edited hypotheses.

Arms B and C are oracle conditions for causal diagnosis. They are not practical
methods, are not selectable, and evaluate no gate. The target set, action,
transport, and abstention rule are inherited unchanged from Gate B.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.language_tags import EN, tag_unit
from ..data.normalize import normalize_and_segment, normalize_text
from ..evaluation.mer import align_tokens, corpus_mer, mer
from ..evaluation.pier import pier, unit_status
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..models.generation import teacher_forced_forward
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from . import v2r3_day5_expansion as D5
from . import v2r3_directions as D3
from . import v2r3_gate_b as GB
from . import v2r3_oracle_screen as OS

STAGE = "v2r3_hypothesis_substitution"
FLOOR_MS = 400.0
ARMS = ("A_baseline", "B_gold_substituted", "C_partial_substitution")

FROZEN_CONFIG: dict[str, Any] = {
    "action": dict(GB.FROZEN_ACTION),
    "floor_ms": FLOOR_MS,
    "target_population": "Gate B conservative-tier D-dev-select baseline errors",
    "arms": {
        "A_baseline": "g_t from generated first-pass hypothesis; Gate B reference",
        "B_gold_substituted": "g_t from gold transcript; oracle causal probe",
        "C_partial_substitution": (
            "g_t from first-pass text after minimal insertion of each target "
            "reference English unit at its deterministic aligned boundary; "
            "oracle causal probe"
        ),
    },
    "partial_construction": {
        "alignment": (
            "deterministic Levenshtein backtrace; insert before an aligned "
            "substitution or at the alignment cursor for a deletion"
        ),
        "edit": (
            "insert the reference English unit with separating spaces into raw "
            "first-pass text while preserving every original character"
        ),
        "validation": (
            "normalized units must equal the original hypothesis units plus "
            "exactly the requested insertions"
        ),
        "skip": (
            "missing or non-English reference unit, missing alignment boundary, "
            "missing raw-text boundary, or failed exact validation"
        ),
    },
    "transport": {
        **dict(GB.FROZEN_CONFIG["transport"]),
        "g_t_source": {
            "A_baseline": "generated first-pass token sequence",
            "B_gold_substituted": "gold transcript token sequence (oracle)",
            "C_partial_substitution": (
                "minimally edited first-pass token sequence containing the "
                "reference English units (oracle)"
            ),
        },
        "only_experimental_change": (
            "the hypothesis token sequence used to derive g_t and its "
            "cross-attention"
        ),
    },
    "abstention": dict(GB.FROZEN_CONFIG["abstention"]),
    "transport_hit_tolerance_steps": GB.TRANSPORT_HIT_TOLERANCE,
    "bootstrap": dict(GB.FROZEN_CONFIG["bootstrap"]),
    "controls": {
        "rule": "run only for an arm with at least one target correction",
        "label_permuted_draws": OS.CONTROL_LABEL_PERMUTED_DRAWS,
        "matched_energy_random_draws": GB.CONTROL_RANDOM_DRAWS,
        "seeds": "identical to Gate B",
    },
    "oracle_conditions_not_methods": ["B_gold_substituted", "C_partial_substitution"],
    "null_draw_noise": GB.NULL_DRAW_NOISE,
    "selects_anything": False,
    "evaluates_gate": False,
    "roles_scored": ["D-dev-select"],
    "held_out_roles_read": [],
}


def _surfaces(text: str) -> list[str]:
    return [u.surface for u in normalize_and_segment(text)[1]]


def _raw_boundary(text: str, units: Sequence[str], boundary: int) -> int | None:
    """Return a deterministic raw character cut for a normalized-unit boundary."""
    left, right = list(units[:boundary]), list(units[boundary:])
    hits = [p for p in range(len(text) + 1)
            if _surfaces(text[:p]) == left and _surfaces(text[p:]) == right]
    return max(hits) if hits else None


def construct_partial_hypothesis(reference: str, baseline: str,
                                 target_indices: Sequence[int]) -> dict[str, Any]:
    """Insert all well-defined target units once in an utterance."""
    _, ref_units = normalize_and_segment(reference)
    _, hyp_units = normalize_and_segment(baseline)
    ref = [u.surface for u in ref_units]
    hyp = [u.surface for u in hyp_units]
    wanted = sorted(set(int(i) for i in target_indices))
    status = {
        i: {"reference_unit_index": i, "well_defined": False,
            "reason": "alignment_boundary_missing"} for i in wanted
    }

    boundary_for: dict[int, int] = {}
    cursor = 0
    for op in align_tokens(ref, hyp):
        if op.ref_idx in status:
            boundary_for[int(op.ref_idx)] = (
                int(op.hyp_idx) if op.hyp_idx is not None else cursor)
        if op.hyp_idx is not None:
            cursor = int(op.hyp_idx) + 1

    insertions: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for index in wanted:
        if index < 0 or index >= len(ref_units):
            status[index]["reason"] = "reference_unit_missing"
            continue
        if tag_unit(ref_units[index]) != EN:
            status[index]["reason"] = "reference_unit_not_english"
            continue
        boundary = boundary_for.get(index)
        if boundary is None or not 0 <= boundary <= len(hyp):
            continue
        raw_pos = _raw_boundary(baseline, hyp, boundary)
        if raw_pos is None:
            status[index]["reason"] = "raw_text_boundary_missing"
            continue
        surface = ref_units[index].surface
        insertions[raw_pos].append((index, surface, boundary))
        status[index].update({
            "well_defined": True, "reason": "ok",
            "hypothesis_unit_boundary": int(boundary),
            "raw_character_boundary": int(raw_pos),
            "inserted_surface": surface,
        })

    edited = str(baseline)
    for pos in sorted(insertions, reverse=True):
        chunk = " ".join(x[1] for x in sorted(insertions[pos]))
        edited = edited[:pos] + " " + chunk + " " + edited[pos:]

    expected = list(hyp)
    by_boundary: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for rows in insertions.values():
        for index, surface, boundary in rows:
            by_boundary[boundary].append((index, surface))
    for boundary in sorted(by_boundary, reverse=True):
        expected[boundary:boundary] = [x[1] for x in sorted(by_boundary[boundary])]
    actual = _surfaces(edited)
    valid = actual == expected
    if not valid:
        edited = baseline
        for item in status.values():
            if item["well_defined"]:
                item.update({"well_defined": False, "reason": "exact_validation_failed"})
    return {
        "text": edited,
        "normalized_text": normalize_text(edited),
        "baseline_units": hyp,
        "expected_units": expected,
        "actual_units": actual,
        "validation_ok": bool(valid),
        "targets": [status[i] for i in wanted],
        "well_defined": sum(bool(status[i]["well_defined"]) for i in wanted),
        "skipped": sum(not bool(status[i]["well_defined"]) for i in wanted),
    }


def _substituted_tokens(tokenizer, baseline_tokens: Sequence[int], prefill: int,
                        text: str) -> list[int]:
    """Preserve the generated special-token envelope and replace its text."""
    base = [int(v) for v in baseline_tokens]
    prefix = base[:max(0, int(prefill))]
    eos = int(tokenizer.eos_token_id)
    body = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)[:220]]
    return prefix + body + [eos]


def _attention(bundle, audio_path: str, tokens: Sequence[int]) -> np.ndarray:
    out, _, _ = teacher_forced_forward(
        bundle, [audio_path], [list(tokens)], output_attentions=True)
    layer = out.cross_attentions[GB.TRANSPORT_LAYER_INDEX].float().mean(dim=1)
    value = layer[0].cpu().numpy()
    del out, layer
    return value


def _localize(bundle, tokens: Sequence[int], attention: np.ndarray, *,
              duration: float, prefill: int, steps: int) -> dict[str, Any]:
    region = GB.candidate_region(bundle, tokens, attention, duration, FLOOR_MS)
    weights = GB.transport_weights(
        attention, region["g"], prefill_width=prefill, steps=steps)
    valid = max(1, int(region["valid_frames"]))
    share = float(region["g"].sum()) / valid if region["g"].size else 0.0
    mass = GB.step_valid_mass(
        attention, valid, prefill_width=prefill, steps=steps)
    selected = int(weights.argmax()) if weights.size else -1
    r_max = float(weights.max()) if weights.size else 0.0
    chance = float(mass[selected]) * share if 0 <= selected < len(mass) else 0.0
    enrichment = r_max / chance if chance > 0 else 0.0
    has_region = bool(region["spans"])
    abstained = (not has_region) or (
        enrichment <= GB.ABSTENTION_ENRICHMENT + GB.ABSTENTION_TIE_TOLERANCE)
    return {
        "weights": weights, "abstained": bool(abstained),
        "has_candidate_region": has_region,
        "hypothesis_has_english": bool(region["hypothesis_has_english"]),
        "en_spans_any": int(region["en_spans_any"]),
        "en_spans_above_floor": int(region["en_spans_above_floor"]),
        "region_frames": float(region["g"].sum()), "valid_frames": valid,
        "selected_step": selected, "r_max": r_max,
        "chance_level": chance, "enrichment": enrichment,
    }


def transport_state(*, oracle_exists: bool, hypothesis_has_english: bool,
                    has_candidate_region: bool, abstained: bool) -> str:
    if not oracle_exists:
        return "no_step_at_all"
    if not hypothesis_has_english:
        return "step_exists_no_english_in_hypothesis"
    if not has_candidate_region:
        return "english_present_below_floor"
    if abstained:
        return "region_found_not_enriched"
    return "localized"


def _utterance_outcome(reference: str, baseline: str, steered: str,
                       targets: set[int]) -> dict[str, Any]:
    before, after = unit_status(reference, baseline), unit_status(reference, steered)
    at_risk = [i for i, value in before.items() if bool(value[0])]
    corrupted = [i for i in at_risk if not bool(after.get(i, (False, ""))[0])]
    outside = [
        i for i in set(before) | set(after)
        if i not in targets
        and bool(before.get(i, (False, ""))[0])
        != bool(after.get(i, (False, ""))[0])
    ]
    improved = [
        i for i in outside
        if not bool(before.get(i, (False, ""))[0])
        and bool(after.get(i, (False, ""))[0])
    ]
    return {
        "at_risk": at_risk, "corrupted_indices": corrupted,
        "units_at_risk": len(at_risk), "units_corrupted": len(corrupted),
        "spillover_changed": len(outside), "spillover_improved": len(improved),
        "spillover_damaged": len(outside) - len(improved),
    }


def _ci(values: Sequence[float], clusters: Sequence[str]) -> dict[str, Any]:
    return GB.cluster_ci(np.asarray(values, dtype=float), clusters)


def _metric_block(frame: pd.DataFrame) -> dict[str, Any]:
    refs = frame["reference"].astype(str).tolist()
    base = frame["baseline_text"].astype(str).tolist()
    hyp = frame["steered_text"].astype(str).tolist()
    clusters = frame["dialogue_id"].astype(str).tolist()
    deltas = {"pier": [], "mer": [], "wer": []}
    for r, b, h in zip(refs, base, hyp):
        deltas["pier"].append(pier([r], [h])["pier"] - pier([r], [b])["pier"])
        deltas["mer"].append(mer(r, h)["rate"] - mer(r, b)["rate"])
        deltas["wer"].append(
            D5.corpus_word_error_rate([r], [h])["wer"]
            - D5.corpus_word_error_rate([r], [b])["wer"])
    return {
        "PIER": {
            "baseline": pier(refs, base), "arm": pier(refs, hyp),
            "delta_arm_minus_baseline_mean_utterance_ci": _ci(deltas["pier"], clusters),
        },
        "MER": {
            "baseline": corpus_mer(refs, base), "arm": corpus_mer(refs, hyp),
            "delta_arm_minus_baseline_mean_utterance_ci": _ci(deltas["mer"], clusters),
        },
        "WER": {
            "baseline": D5.corpus_word_error_rate(refs, base),
            "arm": D5.corpus_word_error_rate(refs, hyp),
            "delta_arm_minus_baseline_mean_utterance_ci": _ci(deltas["wer"], clusters),
        },
    }


def _summary_for(arm: str, target_df: pd.DataFrame, utterance_df: pd.DataFrame,
                 risk_df: pd.DataFrame) -> dict[str, Any]:
    t = target_df[target_df["arm"] == arm]
    u = utterance_df[utterance_df["arm"] == arm]
    risk = risk_df[risk_df["arm"] == arm]
    corrected, corrupted = int(t["corrected"].sum()), int(risk["corrupted"].sum())
    states = Counter(t["transport_state"].astype(str))
    result = {
        "targets": int(len(t)), "utterances": int(len(u)),
        "dialogues": int(t["dialogue_id"].nunique()),
        "corrections": corrected,
        "correction_rate": corrected / len(t) if len(t) else None,
        "correction_rate_ci": _ci(t["corrected"].astype(float), t["dialogue_id"]),
        "baseline_correct_units_at_risk": int(len(risk)),
        "corruptions": corrupted,
        "corruption_rate": corrupted / len(risk) if len(risk) else None,
        "corruption_rate_ci": _ci(risk["corrupted"].astype(float), risk["dialogue_id"]),
        "corrected_to_corrupted_ratio": (
            corrected / corrupted if corrupted else ("inf" if corrected else None)),
        "transport_abstention": {
            k: {"count": int(v), "rate": v / len(t)} for k, v in sorted(states.items())
        },
        "effective_energy": {
            "mean_over_targets": float(t["effective_energy"].mean()),
            "mean_over_utterances": float(u["effective_energy"].mean()),
            "mean_localized_targets": (
                float(t.loc[t["transport_state"] == "localized", "effective_energy"].mean())
                if (t["transport_state"] == "localized").any() else None),
        },
        "spillover": {
            "changed": int(u["spillover_changed"].sum()),
            "improved": int(u["spillover_improved"].sum()),
            "damaged": int(u["spillover_damaged"].sum()),
            "utterances_with_any": int((u["spillover_changed"] > 0).sum()),
            "per_utterance_distribution_descending": sorted(
                u["spillover_changed"].astype(int).tolist(), reverse=True),
        },
        "error_metrics": _metric_block(u),
        "descriptive_below_20_events": bool(corrected < 20 or corrupted < 20),
    }
    strata = {}
    for category in sorted(t["category"].astype(str).unique()):
        tc = t[t["category"] == category]
        utterance_ids = set(tc["utterance_id"].astype(str))
        uc = u[u["utterance_id"].astype(str).isin(utterance_ids)]
        rc = risk[risk["utterance_id"].astype(str).isin(utterance_ids)]
        c, d = int(tc["corrected"].sum()), int(rc["corrupted"].sum())
        strata[category] = {
            "targets": int(len(tc)), "utterances": int(len(uc)),
            "dialogues": int(tc["dialogue_id"].nunique()),
            "corrections": c, "correction_rate": c / len(tc),
            "correction_rate_ci": _ci(tc["corrected"].astype(float), tc["dialogue_id"]),
            "at_risk_units_on_unique_utterances": int(len(rc)),
            "corruptions_on_unique_utterances": d,
            "corruption_rate": d / len(rc) if len(rc) else None,
            "corruption_rate_ci": _ci(rc["corrupted"].astype(float), rc["dialogue_id"]),
            "corrected_to_corrupted_ratio": c / d if d else ("inf" if c else None),
            "transport_abstention": dict(Counter(tc["transport_state"].astype(str))),
            "effective_energy_mean_targets": float(tc["effective_energy"].mean()),
            "spillover_per_utterance": sorted(
                uc["spillover_changed"].astype(int).tolist(), reverse=True),
            "error_metrics_on_unique_utterances": _metric_block(uc),
            "descriptive_below_20_events": bool(c < 20 or d < 20),
            "overlap_note": (
                "category utterance subsets may overlap when one utterance has "
                "targets in multiple categories"
            ),
        }
    result["by_error_category"] = strata
    return result


def resolve_output(output: str | Path) -> Path:
    return GB.resolve_output(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target_dir = resolve_output(args.output_dir)
    directions_root = GB.resolve_directions(args.directions)
    direction_path = directions_root / "directions.npz"
    direction_sidecar = manifest_mod.load(direction_path)
    check = manifest_mod.verify(direction_path, cfg=None, require_identity=False)
    if not check["ok"]:
        raise SystemExit(f"directions failed byte verification: {check}")

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    input_paths = [candidate_path]
    for role in conv.DEVELOPMENT_ROLES:
        input_paths += [
            role_root / f"role_{role}.parquet",
            poi_root / f"poi_{role}.parquet",
        ]
    for path in input_paths:
        verified = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not verified["ok"]:
            raise SystemExit(f"{path.name} failed byte verification: {verified}")

    candidates, verdict = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)
    manifest = pd.concat(
        [pd.read_parquet(role_root / f"role_{r}.parquet")
         for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    poi_table = pd.concat(
        [pd.read_parquet(poi_root / f"poi_{r}.parquet").rename(
            columns={"poi_index": "reference_unit_index"})
         for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    dialogue_of = dict(zip(
        manifest["conversation_id"].astype(str),
        manifest["dialogue_id"].astype(str)))
    spans = D5._spans_at_floor(candidates, dialogue_of, poi_table, FLOOR_MS)
    spans = spans[spans["role"] == "D-dev-select"]
    plan = {
        "frozen_configuration": FROZEN_CONFIG,
        "candidate_spans": int(len(spans)),
        "directions_sha256": direction_sidecar["sha256"],
        "destination": str(target_dir),
    }
    if not args.execute:
        print(json.dumps({
            "state": "validated_no_execution", "models_loaded": False,
            "artifacts_written": [], "plan": plan,
        }, indent=2))
        return 0

    log = setup_logging(level="INFO")
    started = time.time()
    bundle = load_whisper(cfg)
    directions = dict(np.load(direction_path))
    layer = int(GB.FROZEN_ACTION["decoder_layer"])
    rho = float(GB.FROZEN_ACTION["rho"])
    vector = directions[f"site_d_layer{layer}"]
    direction_norm = float(np.linalg.norm(vector))
    targets = OS.build_targets(bundle, candidates, spans, poi_table, manifest)
    if args.limit:
        keep = list(dict.fromkeys(
            targets["utterance_id"].astype(str)))[:int(args.limit)]
        targets = targets[targets["utterance_id"].astype(str).isin(keep)]
    utterance_manifest = manifest[
        manifest["utterance_id"].isin(set(targets["utterance_id"]))]
    utterance_manifest = utterance_manifest.drop_duplicates(
        "utterance_id").reset_index(drop=True)
    log.info("causal probe: %d targets over %d utterances",
             len(targets), len(utterance_manifest))

    target_rows: list[dict[str, Any]] = []
    utterance_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    construction_rows: list[dict[str, Any]] = []
    arm_cache: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in ARMS}
    tokenizer = bundle.processor.tokenizer

    for number, (_, row) in enumerate(utterance_manifest.iterrows(), 1):
        utterance = str(row["utterance_id"])
        group = targets[targets["utterance_id"].astype(str) == utterance]
        reference = str(row["transcript_raw"])
        audio_path = str(row["audio_path"])
        base_tokens, prefill, steps = GB.free_decode_probed(
            bundle, audio_path, layer)
        baseline_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
        base_attention = _attention(bundle, audio_path, base_tokens)
        indices = sorted(set(group["reference_unit_index"].astype(int)))
        partial = construct_partial_hypothesis(
            reference, baseline_text, indices)
        construction_by_index = {
            int(x["reference_unit_index"]): x for x in partial["targets"]}
        for item in partial["targets"]:
            construction_rows.append({
                "utterance_id": utterance,
                "dialogue_id": str(group.iloc[0]["dialogue_id"]),
                **item, "partial_text": partial["text"],
                "validation_ok": partial["validation_ok"],
            })

        gold_tokens = _substituted_tokens(
            tokenizer, base_tokens, prefill, normalize_text(reference))
        partial_tokens = _substituted_tokens(
            tokenizer, base_tokens, prefill, partial["text"])
        token_sets = {
            "A_baseline": base_tokens,
            "B_gold_substituted": gold_tokens,
            "C_partial_substitution": partial_tokens,
        }
        attentions = {
            "A_baseline": base_attention,
            "B_gold_substituted": _attention(
                bundle, audio_path, gold_tokens),
            "C_partial_substitution": _attention(
                bundle, audio_path, partial_tokens),
        }
        oracle_step = max(0, int(group["token_index"].min()) - 1)
        oracle_exists = oracle_step < int(steps)
        outcomes: dict[str, dict[str, Any]] = {}

        for arm in ARMS:
            loc = _localize(
                bundle, token_sets[arm], attentions[arm],
                duration=float(row["duration_sec"]),
                prefill=prefill, steps=steps)
            if loc["abstained"]:
                text_out, fired, applied = baseline_text, 0, 0.0
                delivered = np.zeros(0, dtype=float)
            else:
                text_out, fired, applied, _ = GB.steered_decode(
                    bundle, audio_path, vector, layer=layer, rho=rho,
                    weights=loc["weights"])
                delivered = loc["weights"]
            state = transport_state(
                oracle_exists=oracle_exists,
                hypothesis_has_english=loc["hypothesis_has_english"],
                has_candidate_region=loc["has_candidate_region"],
                abstained=loc["abstained"])
            energy = GB.effective_energy(
                rho, direction_norm, delivered)
            outcomes[arm] = {
                **loc, "text": text_out, "hook_fired": fired,
                "sum_r_squared": applied, "transport_state": state,
                "effective_energy": energy,
                "transport_hit": bool(
                    oracle_exists and not loc["abstained"]
                    and abs(loc["selected_step"] - oracle_step)
                    <= GB.TRANSPORT_HIT_TOLERANCE),
            }
            arm_cache[arm][utterance] = {
                **outcomes[arm], "audio_path": audio_path,
                "reference": reference, "baseline_text": baseline_text,
                "group": group.copy(),
            }

        for arm, outcome in outcomes.items():
            uo = _utterance_outcome(
                reference, baseline_text, outcome["text"], set(indices))
            dialogue = str(group.iloc[0]["dialogue_id"])
            utterance_rows.append({
                "arm": arm, "utterance_id": utterance,
                "dialogue_id": dialogue, "reference": reference,
                "baseline_text": baseline_text,
                "steered_text": outcome["text"],
                "effective_energy": outcome["effective_energy"],
                **{k: uo[k] for k in (
                    "units_at_risk", "units_corrupted",
                    "spillover_changed", "spillover_improved",
                    "spillover_damaged")},
            })
            corrupted = set(uo["corrupted_indices"])
            for index in uo["at_risk"]:
                risk_rows.append({
                    "arm": arm, "utterance_id": utterance,
                    "dialogue_id": dialogue,
                    "reference_unit_index": int(index),
                    "corrupted": int(index in corrupted),
                })
            for _, target in group.iterrows():
                index = int(target["reference_unit_index"])
                rates = D5.outcome_rates(
                    reference, baseline_text, outcome["text"], index)
                construction = construction_by_index.get(index, {})
                target_rows.append({
                    "arm": arm, "utterance_id": utterance,
                    "reference_unit_index": index,
                    "dialogue_id": str(target["dialogue_id"]),
                    "category": str(target["category"]),
                    "reference": reference,
                    "baseline_text": baseline_text,
                    "steered_text": outcome["text"],
                    "transport_state": outcome["transport_state"],
                    "abstained": outcome["abstained"],
                    "hypothesis_has_english": outcome["hypothesis_has_english"],
                    "has_candidate_region": outcome["has_candidate_region"],
                    "en_spans_any": outcome["en_spans_any"],
                    "en_spans_above_floor": outcome["en_spans_above_floor"],
                    "r_max": outcome["r_max"],
                    "chance_level": outcome["chance_level"],
                    "enrichment": outcome["enrichment"],
                    "selected_step": outcome["selected_step"],
                    "oracle_step": oracle_step,
                    "oracle_step_exists": oracle_exists,
                    "transport_hit": outcome["transport_hit"],
                    "effective_energy": outcome["effective_energy"],
                    "hook_fired": outcome["hook_fired"],
                    "partial_construction_well_defined": construction.get(
                        "well_defined"),
                    "partial_construction_reason": construction.get("reason"),
                    **rates,
                })
        if number % 20 == 0 or number == len(utterance_manifest):
            log.info("processed %d/%d utterances (%.1f s)", number,
                     len(utterance_manifest), time.time() - started)

    target_df = pd.DataFrame(target_rows)
    utterance_df = pd.DataFrame(utterance_rows)
    risk_df = pd.DataFrame(risk_rows)
    construction_df = pd.DataFrame(construction_rows)

    nonzero = [
        arm for arm in ARMS
        if int(target_df.loc[
            target_df["arm"] == arm, "corrected"].sum()) > 0
    ]
    control_rows: list[dict[str, Any]] = []
    if nonzero:
        log.info("non-zero corrections in %s; running controls", nonzero)
        construct_spans = D5._spans_at_floor(
            candidates, dialogue_of, poi_table, FLOOR_MS)
        construct = construct_spans[
            (construct_spans["role"] == "D-construct")
            & construct_spans["all_correct"]]
        build_manifest = manifest[
            manifest["utterance_id"].isin(set(construct["utterance_id"]))]
        d_meta, d_raw = D3.site_d_contrasts(
            bundle, build_manifest, construct, [layer],
            batch_size=4, log=log)
        permuted = [
            item["direction"] for item in OS.label_permuted_directions(
                np.vstack(d_raw["vectors"][layer]), d_meta,
                draws=OS.CONTROL_LABEL_PERMUTED_DRAWS,
                seed=OS.CONTROL_SEED + 500 + layer)
        ]
        random_directions = OS.matched_norm_random(
            vector, draws=GB.CONTROL_RANDOM_DRAWS,
            seed=OS.CONTROL_SEED + 600)
        controls = (
            [("label_permuted", k, v) for k, v in enumerate(permuted)]
            + [("matched_energy_random", k, v)
               for k, v in enumerate(random_directions)]
        )
        for arm in nonzero:
            for utterance, cache in arm_cache[arm].items():
                if cache["abstained"]:
                    continue
                for family, draw, control in controls:
                    text_out, fired, applied, _ = GB.steered_decode(
                        bundle, cache["audio_path"], control,
                        layer=layer, rho=rho, weights=cache["weights"])
                    for _, target in cache["group"].iterrows():
                        index = int(target["reference_unit_index"])
                        rates = D5.outcome_rates(
                            cache["reference"], cache["baseline_text"],
                            text_out, index)
                        control_rows.append({
                            "arm": arm, "control_family": family,
                            "draw": int(draw),
                            "utterance_id": utterance,
                            "reference_unit_index": index,
                            "dialogue_id": str(target["dialogue_id"]),
                            "category": str(target["category"]),
                            "effective_energy": GB.effective_energy(
                                rho, float(np.linalg.norm(control)),
                                cache["weights"]),
                            "hook_fired": fired,
                            "sum_r_squared": applied, **rates,
                        })
    else:
        log.info("no arm corrected a target; controls are not warranted")
    control_df = pd.DataFrame(control_rows)

    arm_summary = {
        arm: _summary_for(
            arm, target_df, utterance_df, risk_df) for arm in ARMS
    }
    construction_meta = construction_df.merge(
        targets[[
            "utterance_id", "reference_unit_index", "category"
        ]].drop_duplicates(),
        on=["utterance_id", "reference_unit_index"], how="left")
    construction_summary = {
        "targets": int(len(construction_df)),
        "well_defined": int(construction_df["well_defined"].sum()),
        "skipped": int((~construction_df["well_defined"]).sum()),
        "skip_reasons": dict(Counter(
            construction_df.loc[
                ~construction_df["well_defined"], "reason"].astype(str))),
        "by_category": {},
    }
    for category, group in construction_meta.groupby("category"):
        construction_summary["by_category"][str(category)] = {
            "targets": int(len(group)),
            "well_defined": int(group["well_defined"].sum()),
            "skipped": int((~group["well_defined"]).sum()),
        }

    b_nonzero = arm_summary["B_gold_substituted"]["corrections"] > 0
    c_nonzero = arm_summary["C_partial_substitution"]["corrections"] > 0
    if b_nonzero and c_nonzero:
        answer = "B_and_C_nonzero_token_absence_demonstrated_as_cause"
    elif b_nonzero:
        answer = "B_nonzero_C_zero_token_presence_necessary_not_sufficient"
    elif not c_nonzero:
        answer = "B_and_C_zero_no_english_rate_is_correlate_not_cause"
    else:
        answer = "C_nonzero_B_zero_unanticipated_asymmetry_requires_description"
    for arm in ("B_gold_substituted", "C_partial_substitution"):
        difference = (
            arm_summary[arm]["correction_rate"]
            - arm_summary["A_baseline"]["correction_rate"])
        arm_summary[arm]["correction_rate_difference_vs_A"] = difference
        arm_summary[arm]["below_day6_null_draw_noise_0p03"] = (
            abs(difference) < GB.NULL_DRAW_NOISE)

    controls_summary: dict[str, Any] = {
        "warranted_for_arms": nonzero,
        "reason_if_none": (
            None if nonzero else
            "No arm had a non-zero correction count, so a direction control "
            "cannot distinguish an absent effect."),
        "rows": int(len(control_df)),
    }
    if not control_df.empty:
        draws = {}
        for (arm, family, draw), group in control_df.groupby(
                ["arm", "control_family", "draw"]):
            treated = target_df[target_df["arm"] == arm]
            paired = treated[[
                "utterance_id", "reference_unit_index", "corrected"
            ]].merge(
                group[[
                    "utterance_id", "reference_unit_index",
                    "corrected", "dialogue_id"
                ]],
                on=["utterance_id", "reference_unit_index"],
                suffixes=("_arm", "_control"))
            contrast = (
                paired["corrected_arm"].astype(float)
                - paired["corrected_control"].astype(float))
            draws[f"{arm}/{family}/{int(draw)}"] = {
                "n": int(len(group)),
                "corrections": int(group["corrected"].sum()),
                "mean_effective_energy": float(
                    group["effective_energy"].mean()),
                "arm_minus_control_correction_ci": _ci(
                    contrast, paired["dialogue_id"]),
            }
        controls_summary["draws"] = draws

    summary = {
        "frozen_configuration": FROZEN_CONFIG,
        "plan": plan, "runtime_seconds": time.time() - started,
        "arms": arm_summary,
        "partial_construction": construction_summary,
        "controls": controls_summary,
        "plain_answer_code": answer,
        "selects_anything": False, "evaluates_gate": False,
        "oracle_conditions_not_methods": [
            "B_gold_substituted", "C_partial_substitution"],
    }

    target_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "hypothesis_substitution_targets.parquet": target_df,
        "hypothesis_substitution_utterances.parquet": utterance_df,
        "partial_construction.parquet": construction_df,
    }
    if not control_df.empty:
        frames["hypothesis_substitution_controls.parquet"] = control_df
    for name, frame in frames.items():
        frame.to_parquet(target_dir / name, index=False)
    report_path = target_dir / "hypothesis_substitution_summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for path in [target_dir / name for name in frames] + [report_path]:
        manifest_mod.publish(
            path, cfg=cfg, stage=STAGE,
            parents=[parent] if parent else (),
            schema="lss_v2r3_hypothesis_substitution_v1",
            extra={
                "frozen_configuration": FROZEN_CONFIG,
                "logical_path": str(path),
                "oracle_conditions_not_methods": True,
                "selects_anything": False, "evaluates_gate": False,
            })
    print(json.dumps({
        "state": "completed", "targets": int(len(targets)),
        "plain_answer_code": answer,
        "destination": str(target_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
