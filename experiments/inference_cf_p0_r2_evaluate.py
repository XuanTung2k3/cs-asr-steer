#!/usr/bin/env python
"""CPU-only evaluator for frozen P0-R2 shards; imports no inference runner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from csasr.data.normalize import normalize_text, segment_units
from csasr.evaluation.mer import align_tokens, DEL, MATCH, SUB
from csasr.evaluation.pier import evaluate_pois, LANGUAGE_CONFUSION, pier
from csasr.evaluation.mer import corpus_mer
from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_r2 import VERSION_R2, validate_gate_row


def unit_to_token_positions(tokenizer, content: list[int], hypothesis: str) -> list[int | None]:
    """Checked normalized-prefix offset map; ambiguous shifts remain unalignable."""
    final = normalize_text(hypothesis)
    units = segment_units(final)
    lengths = []
    previous = 0
    for k in range(1, len(content) + 1):
        decoded = tokenizer.decode(content[:k], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        prefix = normalize_text(decoded)
        if not final.startswith(prefix) or len(prefix) < previous:
            return [None] * len(units)
        previous = len(prefix)
        lengths.append(previous)
    if (lengths[-1] if lengths else 0) != len(final):
        return [None] * len(units)
    mapped = []
    for unit in units:
        candidates = [i for i, length in enumerate(lengths) if length > unit.char_start]
        mapped.append(candidates[0] if candidates else None)
    return mapped


def evaluator_labels(reference: str, hypothesis: str, token_positions: list[int | None],
                     *, eos_slot: int | None, heard_sec: float,
                     ctc_by_index: dict[int, dict]) -> tuple[list[dict], dict[int, str]]:
    """Canonical ref↔hyp alignment; positions are evaluator-only output."""
    ref = segment_units(normalize_text(reference))
    hyp = segment_units(normalize_text(hypothesis))
    ops = align_tokens([u.surface for u in ref], [u.surface for u in hyp])
    pois = {p.poi_index: p for p in evaluate_pois(reference, hypothesis)}
    by_ref = {op.ref_idx: (i, op) for i, op in enumerate(ops) if op.ref_idx is not None}
    records = []
    by_position = defaultdict(set)
    for index, unit in enumerate(ref):
        category = None
        if unit.kind == "latin":
            p = pois.get(index)
            if p is not None:
                category = ("EN-correct" if p.correct else
                            "EN-confusion" if p.category in LANGUAGE_CONFUSION else
                            "EN-deletion-slot" if p.category == "deletion" else
                            "EN-same-language-sub" if p.category == "same_language_substitution" else
                            "unalignable:" + p.category)
        elif unit.kind == "han" and index in by_ref and by_ref[index][1].op == MATCH:
            category = "ZH-correct"
        if category is None:
            continue
        position = None
        reason = None
        op_i, op = by_ref.get(index, (-1, None))
        if category.startswith("unalignable:"):
            reason = category
        elif op is None:
            reason = "unalignable:missing_alignment"
        elif op.op == DEL:
            following = next((later.hyp_idx for later in ops[op_i + 1:]
                              if later.ref_idx is not None and later.hyp_idx is not None), None)
            if following is not None:
                # Gap slot = query of the next aligned hypothesis unit; an unmapped offset
                # stays unalignable rather than falling through to the EOS slot.
                position = token_positions[following]
                if position is None:
                    reason = "unalignable:offset"
            else:
                position = eos_slot
                if position is None:
                    reason = "unalignable:truncated_gap"
        elif op.op in (MATCH, SUB) and op.hyp_idx is not None:
            position = token_positions[op.hyp_idx]
            if position is None:
                reason = "unalignable:offset"
        else:
            reason = "unalignable:alignment"
        timing = ctc_by_index.get(index)
        midpoint = None
        if timing and timing.get("is_valid") and timing.get("surface") == unit.surface:
            midpoint = (float(timing["start_sec"]) + float(timing["end_sec"])) / 2
            if midpoint > heard_sec:
                position, reason = None, "unalignable:beyond_heard_audio"
        if position is None and reason is None:
            reason = "unalignable:offset"
        record = {"reference_unit_index": index, "surface": unit.surface,
                  "language": "EN" if unit.kind == "latin" else "ZH",
                  "poi_category": pois[index].category if index in pois else None,
                  "stratum": category, "position": position, "reason": reason,
                  "ctc_midpoint_sec": midpoint}
        records.append(record)
        if position is not None and reason is None:
            by_position[position].add(category)
    unique = {}
    for position, strata in by_position.items():
        unique[position] = next(iter(strata)) if len(strata) == 1 else "unalignable:stratum_collision"
    for record in records:
        position = record["position"]
        if position is not None and unique.get(position) == "unalignable:stratum_collision":
            record["candidate_position"] = position
            record["position"] = None
            record["reason"] = "unalignable:stratum_collision"
    return records, unique


def auc_ap(pos: list[float], neg: list[float]) -> tuple[float | None, float | None]:
    if not pos or not neg:
        return None, None
    from sklearn.metrics import average_precision_score, roc_auc_score
    scores = np.asarray(pos + neg, dtype=np.float64)
    labels = np.asarray([1] * len(pos) + [0] * len(neg))
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def contrast(rows: list[dict], positive: set[str], negative: set[str], field: str) -> dict:
    pos = [r[field] for r in rows if r["stratum"] in positive and r[field] is not None]
    neg = [r[field] for r in rows if r["stratum"] in negative and r[field] is not None]
    a, p = auc_ap(pos, neg)
    return {"auroc": a, "auprc": p, "positive_positions": len(pos),
            "negative_positions": len(neg),
            "positive_dialogues": len({r["dialogue_id"] for r in rows if r["stratum"] in positive and r[field] is not None}),
            "negative_dialogues": len({r["dialogue_id"] for r in rows if r["stratum"] in negative and r[field] is not None})}


def cluster_intervals(rows: list[dict], positive: set[str], negative: set[str],
                      fields: tuple[str, ...], *, replicates: int, seed: int) -> dict:
    by_dialogue = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_id"]].append(row)
    groups = sorted(by_dialogue)
    rng = np.random.default_rng(seed)
    values = {f: {"auroc": [], "auprc": []} for f in fields}
    paired = []
    for _ in range(replicates):
        sample = [r for j in rng.choice(len(groups), size=len(groups), replace=True)
                  for r in by_dialogue[groups[j]]]
        scores = {f: contrast(sample, positive, negative, f) for f in fields}
        for f, pair in scores.items():
            for metric in ("auroc", "auprc"):
                if pair[metric] is not None and math.isfinite(pair[metric]):
                    values[f][metric].append(pair[metric])
        if len(fields) == 2 and all(scores[f]["auroc"] is not None for f in fields):
            paired.append(scores[fields[0]]["auroc"] - scores[fields[1]]["auroc"])
    def interval(xs):
        return [float(np.quantile(xs, .025)), float(np.quantile(xs, .975))] if xs else None
    return {"field_intervals": {f: {metric: interval(values[f][metric])
                                   for metric in ("auroc", "auprc")} for f in fields},
            "valid_replicates": {f: len(values[f]["auroc"]) for f in fields},
            "paired_delta_interval": interval(paired),
            "paired_valid_replicates": len(paired), "replicates": replicates}


def estimable(c: dict, criteria: dict) -> bool:
    return (c["positive_positions"] >= criteria["min_positions_per_stratum"] and
            c["negative_positions"] >= criteria["min_positions_per_stratum"] and
            c["positive_dialogues"] >= criteria["min_dialogues_per_stratum"] and
            c["negative_dialogues"] >= criteria["min_dialogues_per_stratum"])


def auc_status(c: dict, lower: list[float] | None, criteria: dict) -> str:
    """Frozen AUROC rule; an under-powered contrast is NOT_ESTIMABLE, never a pass."""
    if not estimable(c, criteria) or c["auroc"] is None or lower is None:
        return "NOT_ESTIMABLE"
    return ("PASS" if c["auroc"] >= criteria["auc_min"] and lower[0] > criteria["bootstrap_lower_chance"]
            else "FAIL")


def permutation_diagnostic(rows: list[dict], positive: set[str], negative: set[str], *, seed: int) -> dict:
    """Offline negative control: shuffle E or D among non-fallback rows, recompute E*D.

    Descriptive only. It must never be used to choose or tune a formula.
    """
    rng = np.random.default_rng(seed)
    live = [i for i, r in enumerate(rows) if r["fallback_reason"] is None and r["E"] is not None]
    out = {}
    for name in ("E", "D"):
        values = [rows[i][name] for i in live]
        shuffled = list(rng.permutation(values)) if values else []
        permuted = [dict(r) for r in rows]
        for i, v in zip(live, shuffled):
            permuted[i][name] = float(v)
            permuted[i]["g_perm"] = permuted[i]["E"] * permuted[i]["D"]
        for r in permuted:
            r.setdefault("g_perm", r["g_cf"])
        out[f"shuffle_{name}"] = contrast(permuted, positive, negative, "g_perm")
    return out


def _description(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": float(np.median(values)),
            "iqr": [float(np.quantile(values, .25)), float(np.quantile(values, .75))],
            "quantiles": {str(q): float(np.quantile(values, q)) for q in (.05, .5, .95)}}


def _json_safe(value):
    """Missing metrics stay explicit nulls when an incomplete run is blocked."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def main() -> None:
    import pyarrow.parquet as pq
    from transformers import WhisperProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/inference_cf/p0_r2")
    args = parser.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != VERSION_R2 or digest({k: v for k, v in manifest.items()
                                                     if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid manifest")
    for path, expected in manifest["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    panel = json.loads((out / "evaluation_panel.json").read_text())
    inference_panel = json.loads((out / "inference_panel.json").read_text())
    if digest(panel) != manifest["evaluation_panel_hash"] or digest(inference_panel) != manifest["panel_hash"]:
        raise ValueError("panel hash mismatch")
    processor = WhisperProcessor.from_pretrained(manifest["local_model"] if "local_model" in manifest
                                                 else "/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    eval_by_id = {r["utterance_id"]: r for r in panel["rows"]}
    ctc_rows = pq.read_table(manifest["ctc_evaluator_only"],
                             columns=["utterance_id", "reference_unit_index", "surface",
                                      "start_sec", "end_sec", "is_valid"]).to_pylist()
    ctc = defaultdict(dict)
    for row in ctc_rows:
        if row["utterance_id"] in eval_by_id:
            ctc[row["utterance_id"]][int(row["reference_unit_index"])] = row
    auto_rows = pq.read_table(manifest["comparator"]["path"],
                              columns=["utterance_id", "hypothesis_raw"]).to_pylist()
    auto = {r["utterance_id"]: r["hypothesis_raw"] for r in auto_rows}
    data, units, failures = [], [], []
    violations, replay_agreement = [], []
    eos_slots = 0
    baseline_texts = {}
    for i, item in enumerate(inference_panel["rows"]):
        uid = item["utterance_id"]
        path = out / "rows" / f"{i:03d}.json"
        if not path.exists():
            failures.append({"utterance_id": uid, "reason": "missing_shard"})
            continue
        shard = json.loads(path.read_text())
        if shard.get("identity") != uid or shard.get("manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("stale shard")
        if shard.get("status") != "ok":
            failures.append({"utterance_id": uid, "reason": shard.get("reason", "unknown")})
            continue
        baseline_texts[uid] = shard["baseline_text"]
        ev = eval_by_id[uid]
        token_positions = unit_to_token_positions(processor.tokenizer, shard["baseline_content_ids"],
                                                   shard["baseline_text"])
        eos_slot = len(shard["baseline_content_ids"]) if shard["has_eos"] else None
        labeled_units, position_labels = evaluator_labels(
            ev["reference"], shard["baseline_text"], token_positions, eos_slot=eos_slot,
            heard_sec=min(30., ev["duration_sec"]), ctc_by_index=ctc[uid])
        units.extend([{**u, "utterance_id": uid, "dialogue_id": ev["dialogue_id"]}
                      for u in labeled_units])
        eos_slots += int(bool(shard["has_eos"]))
        if shard.get("replay_argmax_agreement") is not None:
            replay_agreement.append(shard["replay_argmax_agreement"])
        for row in shard["rows"]:
            try:
                validate_gate_row(row)
            except ValueError as exc:
                violations.append({"utterance_id": uid, "position": row.get("logical_position"),
                                   "error": str(exc)})
                continue
            t = row["logical_position"]
            stratum = position_labels.get(t)
            local = row["local_support"] or {}
            control = (row["control"] or {}).get("support") or {}
            window = row["window"] or {}
            xcf = row.get("xcf") or {}
            d_x = xcf.get("D_X")
            e_value = local.get("E")
            ineligible = (row["fallback_reason"] is not None and
                          row.get("eligibility_status") != "cf_ineligible")
            g_x = (0.0 if ineligible else
                   e_value * d_x if e_value is not None and d_x is not None else None)
            data.append({"utterance_id": uid, "dialogue_id": ev["dialogue_id"],
                         "position": t, "stratum": stratum,
                         "fallback_reason": row["fallback_reason"],
                         "is_eos_slot": row.get("is_eos_slot"),
                         "replay_match": row.get("replay_argmax_matches_baseline"),
                         "same_prefix_ok": row.get("same_prefix_validation") is True,
                         "eligibility_status": row.get("eligibility_status"),
                         "D_X": d_x, "g_X": g_x,
                         "R_X": (xcf.get("conflict") or {}).get("R"),
                         "E": local.get("E"), "E_control": control.get("E"),
                         "P_M_B": (row["baseline"] or {}).get("P_M"),
                         "P_E_B": (row["baseline"] or {}).get("P_E"),
                         "Q_B": (row["baseline"] or {}).get("Q"),
                         "R_B": (row["baseline"] or {}).get("R"),
                         "P_M_Ecf": (row["ecf"] or {}).get("P_M"),
                         "P_E_Ecf": (row["ecf"] or {}).get("P_E"),
                         "Q_Ecf": (row["ecf"] or {}).get("Q"),
                         "R_Ecf": (row["ecf"] or {}).get("R"),
                         "D": row["gate"]["D"], "g_old": row["gate"]["g_old"],
                         "g_cf": row["gate"]["g_cf"],
                         "center_sec": window.get("center_sec"),
                         "point_center_sec": window.get("point_center_sec"),
                         "start_sec": (window.get("start_sample") / 16000 if window else None),
                         "end_sec": (window.get("end_sample") / 16000 if window else None),
                         "point_start_sec": (window.get("point_start_sample") / 16000 if window else None),
                         "point_end_sec": (window.get("point_end_sample") / 16000 if window else None)})
    row_by_key = {(r["utterance_id"], r["position"]): r for r in data}
    timing = []
    for u in units:
        if u["position"] is None or u["ctc_midpoint_sec"] is None:
            continue
        row = row_by_key.get((u["utterance_id"], u["position"]))
        if row is None:
            continue
        mid = u["ctc_midpoint_sec"]
        timing.append({"stratum": u["stratum"], "dialogue_id": row["dialogue_id"],
                       "max_error": abs(row["center_sec"] - mid) if row["center_sec"] is not None else None,
                       "point_error": abs(row["point_center_sec"] - mid) if row["point_center_sec"] is not None else None,
                       "max_covers": bool(row["start_sec"] <= mid <= row["end_sec"]) if row["start_sec"] is not None else False,
                       "point_covers": bool(row["point_start_sec"] <= mid <= row["point_end_sec"]) if row["point_start_sec"] is not None else False})
    criteria = manifest["resolved_config"]["feasibility"]
    target = {"EN-confusion"}
    comparisons = {}
    for name, neg in (("confusion_vs_zh", {"ZH-correct"}),
                      ("confusion_vs_en_correct", {"EN-correct"})):
        old = contrast(data, target, neg, "g_old")
        new = contrast(data, target, neg, "g_cf")
        intervals = cluster_intervals(data, target, neg, ("g_cf", "g_old"),
                                      replicates=manifest["resolved_config"]["bootstrap_replicates"],
                                      seed=manifest["seed"])
        comparisons[name] = {"old": old, "cf": new,
                             "delta_auroc": (new["auroc"] - old["auroc"]
                                             if new["auroc"] is not None and old["auroc"] is not None else None),
                             "bootstrap": intervals,
                             # Diagnostic only: same local support times the wrong-language
                             # (cX) conflict reduction. Never a pass rule or gate.
                             "wrong_language_diagnostic": contrast(data, target, neg, "g_X"),
                             # Descriptive decomposition: which factor carries the ranking.
                             # R_B nearly separates confusion from correct English by the
                             # emitted script; E must carry confusion-vs-Mandarin.
                             "component_diagnostic": {f: contrast(data, target, neg, f)
                                                      for f in ("E", "R_B", "R_Ecf", "D", "D_X")},
                             "permutation_diagnostic": permutation_diagnostic(
                                 data, target, neg, seed=manifest["seed"])}
    ls_target = {"EN-correct", "EN-confusion"}
    ls_neg = {"ZH-correct"}
    support = contrast(data, ls_target, ls_neg, "E")
    support_ci = cluster_intervals(data, ls_target, ls_neg, ("E",),
                                   replicates=manifest["resolved_config"]["bootstrap_replicates"],
                                   seed=manifest["seed"])
    paired = [r for r in data if r["E"] is not None and r["E_control"] is not None]
    control_ci = cluster_intervals(paired, ls_target, ls_neg, ("E", "E_control"),
                                   replicates=manifest["resolved_config"]["bootstrap_replicates"],
                                   seed=manifest["seed"])
    paired_real = contrast(paired, ls_target, ls_neg, "E")
    paired_control = contrast(paired, ls_target, ls_neg, "E_control")
    fallback = Counter(r["fallback_reason"] for r in data if r["fallback_reason"])
    stratum_summary = {s: {field: _description([r[field] for r in data if r["stratum"] == s and r[field] is not None])
                           for field in ("E", "P_M_B", "P_E_B", "Q_B", "R_B",
                                         "P_M_Ecf", "P_E_Ecf", "Q_Ecf", "R_Ecf",
                                         "D", "g_old", "g_cf", "R_X", "D_X", "g_X")}
                       for s in ("EN-confusion", "EN-deletion-slot", "EN-same-language-sub",
                                 "EN-correct", "ZH-correct")}
    for s, entry in stratum_summary.items():
        segment = [r for r in data if r["stratum"] == s]
        entry["fallback_rate"] = (sum(r["fallback_reason"] is not None for r in segment) /
                                  len(segment) if segment else None)
        entry["provider_valid_rate"] = (sum(r["E"] is not None for r in segment) / len(segment)
                                        if segment else None)
        entry["positions"] = len(segment)
        entry["dialogues"] = len({r["dialogue_id"] for r in segment})
    en_units = [u for u in units if u["language"] == "EN"]
    nonboundary = [u for u in en_units if u["poi_category"] != "boundary_error"]
    # ALIGN measures mapping: a unit with a uniquely determined token query or gap slot is
    # mapped even if that query also hosts another stratum. Such stratum collisions are
    # excluded from the primary contrasts (spec §3), which is a label rule, not a map failure.
    def _mapped(u):
        return (u["position"] is not None and u["reason"] is None) or \
            u["reason"] == "unalignable:stratum_collision"
    alignment_coverage = (sum(_mapped(u) for u in nonboundary) / len(nonboundary)
                          if nonboundary else 0.0)
    alignment_coverage_contrast_eligible = (
        sum(u["position"] is not None and u["reason"] is None for u in nonboundary)
        / len(nonboundary) if nonboundary else 0.0)
    collision_by_stratum = dict(Counter(u["stratum"] for u in units
                                        if u["reason"] == "unalignable:stratum_collision"))
    timely = [x for x in timing if x["max_error"] is not None]
    localizer_rate = sum(x["max_error"] is not None and x["max_error"] <= .5 for x in timing) / len(timing) if timing else 0.0
    pre_provider = [r for r in data if r["fallback_reason"] != "mid_character"]
    support_coverage = sum(r["E"] is not None for r in pre_provider) / len(pre_provider) if pre_provider else 0.0
    mapped_finite = [r for r in data if r["stratum"] in ls_target | ls_neg and r["E"] is not None]
    paired_coverage = len([r for r in mapped_finite if r["E_control"] is not None]) / len(mapped_finite) if mapped_finite else 0.0
    for c in comparisons.values():
        c["status"] = auc_status(c["cf"], c["bootstrap"]["field_intervals"]["g_cf"]["auroc"], criteria)
        c["old_gate_status_descriptive"] = auc_status(
            c["old"], c["bootstrap"]["field_intervals"]["g_old"]["auroc"], criteria)
    ls_bound = support_ci["field_intervals"]["E"]["auroc"]
    control_bound = control_ci["paired_delta_interval"]
    ls_auc_status = auc_status(support, ls_bound, criteria)
    ls_status = ("NOT_ESTIMABLE" if ls_auc_status == "NOT_ESTIMABLE" or control_bound is None else
                 "PASS" if (ls_auc_status == "PASS" and
                            support_coverage >= criteria["support_finite_coverage_min"] and
                            paired_coverage >= criteria["paired_audio_coverage_min"] and
                            control_bound[0] > 0) else "FAIL")
    ls_ok = ls_status == "PASS"
    localizer_en = {x["dialogue_id"] for x in timing if x["stratum"].startswith("EN-")}
    localizer_zh = {x["dialogue_id"] for x in timing if x["stratum"] == "ZH-correct"}
    localizer_estimable = (sum(x["stratum"].startswith("EN-") for x in timing) >= criteria["min_positions_per_stratum"] and
                           sum(x["stratum"] == "ZH-correct" for x in timing) >= criteria["min_positions_per_stratum"] and
                           len(localizer_en) >= criteria["min_dialogues_per_stratum"] and
                           len(localizer_zh) >= criteria["min_dialogues_per_stratum"])
    localizer_status = ("NOT_ESTIMABLE" if not localizer_estimable else
                        "PASS" if localizer_rate >= criteria["localizer_within_half_second_min"] else "FAIL")
    localizer_ok = localizer_status == "PASS"
    bc_ok = (not violations and
             all(r["R_B"] is not None and r["R_Ecf"] is not None for r in pre_provider))
    alignment_ok = alignment_coverage >= criteria["alignment_coverage_min"]
    # R2-ECF: exact same-prefix identity on every position, finite Ecf masses wherever the
    # prefix is complete, and zero formula/bound violations (checked in validate_gate_row).
    ecf_ok = (not violations and all(r["same_prefix_ok"] for r in data) and
              all(r["R_Ecf"] is not None for r in pre_provider))
    gate_ok = all(c["status"] == "PASS" for c in comparisons.values())
    cf_preferred = (gate_ok and
                    comparisons["confusion_vs_en_correct"]["bootstrap"]["paired_delta_interval"] is not None and
                    comparisons["confusion_vs_en_correct"]["bootstrap"]["paired_delta_interval"][0] > 0 and
                    comparisons["confusion_vs_zh"]["delta_auroc"] is not None and
                    comparisons["confusion_vs_zh"]["delta_auroc"] >= 0)
    refs = [eval_by_id[uid]["reference"] for uid in baseline_texts]
    base_hyps = [baseline_texts[uid] for uid in baseline_texts]
    auto_hyps = [auto[uid] for uid in baseline_texts if uid in auto]
    auto_refs = [eval_by_id[uid]["reference"] for uid in baseline_texts if uid in auto]
    by_timing_stratum = {s: {"n": len(xs),
                             "max_valid_rate": sum(x["max_error"] is not None for x in xs) / len(xs),
                             "point_valid_rate": sum(x["point_error"] is not None for x in xs) / len(xs),
                             "max_error_sec": _description([x["max_error"] for x in xs if x["max_error"] is not None]),
                             "point_error_sec": _description([x["point_error"] for x in xs if x["point_error"] is not None]),
                             "max_within_half_second": sum(x["max_error"] is not None and x["max_error"] <= .5 for x in xs) / len(xs) if xs else 0.,
                             "point_within_half_second": sum(x["point_error"] is not None and x["point_error"] <= .5 for x in xs) / len(xs) if xs else 0.,
                             "max_midpoint_coverage": sum(x["max_covers"] for x in xs) / len(xs) if xs else 0.,
                             "point_midpoint_coverage": sum(x["point_covers"] for x in xs) / len(xs) if xs else 0.}
                         for s in ("ZH-correct", "EN-correct", "EN-confusion", "EN-deletion-slot")
                         if (xs := [x for x in timing if x["stratum"] == s])}
    null_probs = json.loads((out / "runtime.json").read_text())["null_probs"]
    en_id, zh_id = manifest["conditions"]["language_token_ids"]
    from csasr.inference_cf.core_r2 import local_support
    null_identity = local_support(null_probs[str(en_id)], null_probs[str(zh_id)],
                                  null_probs[str(en_id)], null_probs[str(zh_id)])
    null_ok = abs(null_identity["A"]) <= criteria["numeric_tolerance"] and abs(null_identity["E"]) <= criteria["numeric_tolerance"]
    pointwise_nonincrease = all(r["g_cf"] <= r["g_old"] + criteria["numeric_tolerance"] for r in data)
    prerequisites = all((alignment_ok, localizer_ok, ls_ok, bc_ok, ecf_ok, null_ok,
                         pointwise_nonincrease))
    summary = {"schema": VERSION_R2, "manifest_hash": manifest["manifest_hash"],
               "status": "R2_CF_PREFERRED" if cf_preferred and prerequisites else
                         "R2_CF_FEASIBLE_NOT_PREFERRED" if gate_ok and prerequisites else
                         "R2_BLOCKED",
               "checks": {"align": alignment_ok, "localizer": localizer_ok, "ls": ls_ok and null_ok,
                          "bc": bc_ok, "ecf": ecf_ok, "gate": gate_ok, "cf_preferred": cf_preferred,
                          "null_identity": null_ok,
                          "pointwise_mandarin_nonincrease": pointwise_nonincrease},
               "check_status": {"localizer": localizer_status, "ls": ls_status,
                                "gate_confusion_vs_zh": comparisons["confusion_vs_zh"]["status"],
                                "gate_confusion_vs_en_correct": comparisons["confusion_vs_en_correct"]["status"]},
               "counts": {"expected_utterances": len(inference_panel["rows"]),
                          "completed_utterances": len(baseline_texts), "failures": failures,
                          "positions": len(data), "fallbacks": dict(fallback),
                          "invariant_violations": violations,
                          "eos_gap_slots": eos_slots,
                          "reference_units": len(units),
                          "mapped_units": sum(_mapped(u) for u in units),
                          "contrast_eligible_units": sum(u["position"] is not None and u["reason"] is None
                                                         for u in units),
                          "stratum_collision_units_by_stratum": collision_by_stratum,
                          "unalignable_reasons": dict(Counter(u["reason"] for u in units
                                                               if u["reason"] is not None))},
               "alignment_coverage": alignment_coverage,
               "alignment_coverage_contrast_eligible_descriptive": alignment_coverage_contrast_eligible,
               "replay_consistency": {"utterance_argmax_agreement": _description(replay_agreement),
                                      "position_agreement_rate": (sum(bool(r["replay_match"]) for r in data)
                                                                  / len(data) if data else None)},
               "localizer": {"within_half_second": localizer_rate, "timed_units": len(timing),
                             "max_window": _description([x["max_error"] for x in timely]),
                             "point_argmax": _description([x["point_error"] for x in timing if x["point_error"] is not None]),
                             "max_window_midpoint_coverage": sum(x["max_covers"] for x in timing) / len(timing) if timing else 0,
                             "point_window_midpoint_coverage": sum(x["point_covers"] for x in timing) / len(timing) if timing else 0,
                             "by_stratum": by_timing_stratum},
               "support": {"real": support, "real_bootstrap": support_ci,
                           "paired_real": paired_real, "paired_control": paired_control,
                           "control_bootstrap": control_ci,
                           "finite_coverage": support_coverage, "paired_coverage": paired_coverage},
               "gate_comparisons": comparisons, "strata": stratum_summary,
               "baseline_metrics": {"B0_forced_zh": {"pier": pier(refs, base_hyps),
                                                     "mer": corpus_mer(refs, base_hyps)},
                                    "B0_AUTO": {"pier": pier(auto_refs, auto_hyps),
                                                "mer": corpus_mer(auto_refs, auto_hyps),
                                                "matched_utterances": len(auto_hyps)}}}
    # Never silently declare a panel pass when any scheduled utterance is absent.
    if failures or len(auto_hyps) != len(inference_panel["rows"]):
        summary["status"] = "R2_BLOCKED"
        summary["checks"]["complete_population"] = False
    else:
        summary["checks"]["complete_population"] = True
    atomic_json(out / "evaluation_units.json", units)
    atomic_json(out / "summary.json", _json_safe(summary))
    print(json.dumps({"status": summary["status"], "manifest_hash": manifest["manifest_hash"],
                      "completed": len(baseline_texts)}, indent=2))


if __name__ == "__main__":
    main()
