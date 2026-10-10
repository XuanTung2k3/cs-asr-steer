"""Physical A6-OTT Phase-B/Phase-C runner.

The model, analysis, direction, hook, generation, and metric primitives are
imported from the accepted compact Phase-A runner.  This module supplies only
the phase manifests, frozen settings, method-specific eligibility, result
namespace, and Qwen greedy/official aliasing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "src")]
ROOT = REPO / "results/a6_ott_upper_bound"
ELIG = ROOT / "eligibility"
MODEL_NAMES = {"whisper": "whisper", "qwen": "qwen3_asr_1p7b", "qwen3_asr_1p7b": "qwen3_asr_1p7b"}
SITES = {"whisper": {"encoder": 32, "decoder": 32}, "qwen3_asr_1p7b": {"encoder": 24, "decoder": 28}}
METHODS = (("add_unique", "encoder"), ("add_unique", "decoder"), ("conditioning_cs", "decoder"))

from csasr.experiments import a6_ott as base


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def phase_for(phase: str) -> str:
    phase = phase.upper()
    if phase not in {"B", "C"}:
        raise ValueError("phase must be B or C")
    return phase


def canonical_key(*, phase: str, model: str, dataset: str, uid: str, family: str, side: str,
                  layer: int, rho: float, decode_mode: str) -> str:
    site = "encoder_post_self_attn_residual_pre_ffn" if side == "encoder" else "decoder_post_cross_attn_residual"
    return "|".join(map(str, (phase, model, dataset, uid, family, side, int(layer), float(rho), decode_mode, site)))


def row_path(phase: str, key: str) -> Path:
    parts = key.split("|")
    _phase, model, dataset, uid, family, side, layer, rho, mode, _site = parts
    return ROOT / ("confirm" if phase == "B" else "transfer") / "rows" / model / dataset / family / side / f"L{int(layer):02d}" / f"rho{float(rho):g}" / mode / (hashlib.sha256(uid.encode()).hexdigest()[:24] + ".json")


def accepted(phase: str, key: str) -> dict[str, Any] | None:
    path = row_path(phase, key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS" or payload.get("canonical_key") != key:
            return None
        p = payload.get("provenance") or {}
        if p.get("phase") != phase or p.get("regime") != "oracle_tt":
            return None
        if p.get("fixed_direction_artifact") is not None:
            return None
        return {"reused": True, "source_artifact": str(path.relative_to(REPO)), "source_hash": _file_sha(path), "payload": payload}
    except Exception:
        return None


def _method_manifest(dataset: str, family: str, side: str | None = None, model: str | None = None) -> dict[str, Any]:
    names = {"cs_dialogue_confirm": "CS_CONFIRM", "ascend_confirm": "ASCEND_CONFIRM",
             "seame_dev_man": "SEAME_MAN", "seame_dev_sge": "SEAME_SGE"}
    method_key = f"{family}_{side}" if side else family
    suffix = {"add_unique_encoder": "ADD_UNIQUE_ENCODER",
              "add_unique_decoder": "ADD_UNIQUE_DECODER",
              "conditioning_cs_decoder": "CONDITIONING_CS_DECODER"}[method_key]
    path = ELIG / f"{('WHISPER' if model == 'whisper' else 'QWEN3_ASR_1P7B')}_{names[dataset]}_{suffix}.json" if model else ELIG / f"{names[dataset]}_{suffix}.json"
    if not path.is_file() and model is not None:
        path = ELIG / f"{names[dataset]}_{suffix}.json"
    if not path.is_file():
        raise RuntimeError(f"missing validated method eligibility manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS" or payload.get("steering_outcomes_consulted"):
        raise RuntimeError(f"invalid eligibility manifest: {path}")
    return payload


def _panel_name(dataset: str) -> str:
    return {"cs_dialogue_confirm": "cs_dialogue_dev_select", "ascend_confirm": "ascend_eval",
            "seame_dev_man": "seame_dev_man", "seame_dev_sge": "seame_dev_sge"}[dataset]


def _dataset_rows(phase: str, dataset: str, families: set[tuple[str, str]], model: str) -> tuple[list[dict[str, Any]], str]:
    from csasr.basis_a6.panels import load_panel
    rows, fp = load_panel(_panel_name(dataset), require_alignment=True)
    ids: set[str] = set()
    for family, side in families:
        ids.update(_method_manifest(dataset, family, side, model)["ids"])
    by_id = {str(row["utterance_id"]): dict(row) for row in rows}
    if ids - set(by_id):
        raise RuntimeError(f"{dataset}: missing IDs in panel: {sorted(ids - set(by_id))[:5]}")
    # Transcript-only oracle rows remain acoustic-empty by design.  This field
    # is consumed only by decoder analysis; encoder masks still read spans.
    # The small method manifests intentionally contain only IDs.  The full
    # audit beside them owns the per-utterance oracle metadata used to attach
    # transcript-only regions without manufacturing acoustic spans.
    audit_names = {"cs_dialogue_confirm": "CS_CONFIRM", "ascend_confirm": "ASCEND_CONFIRM",
                   "seame_dev_man": "SEAME_MAN", "seame_dev_sge": "SEAME_SGE"}
    audit = json.loads((ELIG / f"{audit_names[dataset]}_METHODS.json").read_text())
    entries: dict[str, dict[str, Any]] = {}
    for entry in audit.get("entries", {}).values():
        entries[str(entry["utterance_id"])] = entry
    for uid in ids:
        entry = entries.get(uid)
        if entry and entry.get("oracle_transcript_spans"):
            by_id[uid]["oracle_transcript_spans"] = entry["oracle_transcript_spans"]
        elif entry:
            by_id[uid]["oracle_transcript_spans"] = []
    return sorted((by_id[uid] for uid in ids), key=lambda row: str(row["utterance_id"])), fp


def _settings(model: str, phase: str) -> list[dict[str, Any]]:
    if phase == "B":
        path = ROOT / "confirm" / f"PHASE_B_SETTINGS_{model}.json"
        if not path.is_file():
            selections = json.loads((ROOT / "search" / ("WHISPER_SELECTION.json" if model == "whisper" else "QWEN_SELECTION.json")).read_text())
            settings = []
            for family_side, items in selections["top2_by_family"].items():
                family, side = family_side.split("/")
                for item in items:
                    settings.append({"family": family, "side": side, "layer": int(item["layer"]), "rho": float(item["rho"])})
            _atomic(path, {"schema_version": "a6_ott_phase_b_settings_v1", "status": "FROZEN_FROM_PHASE_A_SELECTION", "model": model, "settings": settings, "phase_a_selection_sha256": _file_sha(ROOT / "search" / ("WHISPER_SELECTION.json" if model == "whisper" else "QWEN_SELECTION.json"))})
        return json.loads(path.read_text())["settings"]
    path = ROOT / "confirm" / f"PHASE_B_FINAL_SETTINGS_{model}.json"
    if not path.is_file():
        raise RuntimeError(f"final Phase-B settings are not frozen: {path}")
    return json.loads(path.read_text())["settings"]


def _baseline(model: str, dataset: str, mode: str) -> dict[str, Any]:
    old = {"cs_dialogue_confirm": "cs_dialogue_dev_select", "ascend_confirm": "ascend_eval",
           "seame_dev_man": "seame_dev_man", "seame_dev_sge": "seame_dev_sge"}[dataset]
    actual_mode = "greedy" if model == "qwen3_asr_1p7b" else mode
    path = ROOT.parent / "basis_a6_expanded" / "baselines" / model / old / f"{actual_mode}.json"
    if not path.is_file():
        raise RuntimeError(f"validated baseline missing: {path}")
    return json.loads(path.read_text())


def _rows_by_method(dataset: str, settings: list[dict[str, Any]], model: str) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for family, side in sorted({(str(s["family"]), str(s["side"])) for s in settings}):
        manifest = _method_manifest(dataset, family, side, model)
        out[(family, side)] = manifest["ids"]
    return out


def _selected_for_side(settings: list[dict[str, Any]], side: str) -> dict[int, dict[str, list[float]]]:
    out: dict[int, dict[str, list[float]]] = {}
    for s in settings:
        if s["side"] != side:
            continue
        out.setdefault(int(s["layer"]), {}).setdefault(str(s["family"]), []).append(float(s["rho"]))
    return out


def _run_selected_sample(bundle, model: str, phase: str, dataset: str, row: dict[str, Any], baseline: dict[str, Any],
                         side: str, selected: dict[int, dict[str, list[float]]], mode: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    layers = sorted(selected)
    masks = base._whisper_masks(bundle, row, baseline) if model == "whisper" else base._qwen_masks(bundle, row, baseline)
    analysis = (base._whisper_analysis(bundle, row, baseline, masks, layers, side) if model == "whisper" else base._qwen_analysis(bundle, row, baseline, masks, layers, side))
    from csasr.basis_a6.cache import DirectionCache
    cache = DirectionCache(); stats = {"direction_constructions": 0, "rho_cache_hits": 0, "new_rows": 0, "reused_rows": 0, "ineligible": []}
    directions = {}
    effective_selected = {layer: dict(families) for layer, families in selected.items()}
    for layer in layers:
        try:
            directions[layer] = base._direction_set(model, dataset, str(row["utterance_id"]), mode, side, layer, analysis[layer], cache, stats)
        except ValueError as exc:
            if "conditioning position selection is empty" not in str(exc) or "conditioning_cs" not in effective_selected[layer]:
                raise
            reduced = dict(analysis[layer]); reduced["conditioning"] = None
            directions[layer] = base._direction_set(
                model, dataset, str(row["utterance_id"]), mode, side, layer,
                reduced, cache, stats, wanted_methods={"add_unique"})
            for rho in sorted(set(effective_selected[layer]["conditioning_cs"])):
                stats["ineligible"].append({"canonical_key": canonical_key(phase=phase, model=model, dataset=dataset, uid=str(row["utterance_id"]), family="conditioning_cs", side=side, layer=layer, rho=rho, decode_mode=mode), "status": "INELIGIBLE", "reason": "conditioning_cs_position_selection_empty", "outcome_consulted": False})
            effective_selected[layer].pop("conditioning_cs", None)
        if "conditioning_cs" in effective_selected[layer] and "conditioning_cs" not in directions[layer]:
            for rho in sorted(set(effective_selected[layer]["conditioning_cs"])):
                stats["ineligible"].append({"canonical_key": canonical_key(phase=phase, model=model, dataset=dataset, uid=str(row["utterance_id"]), family="conditioning_cs", side=side, layer=layer, rho=rho, decode_mode=mode), "status": "INELIGIBLE", "reason": "conditioning_cs_position_selection_empty", "outcome_consulted": False})
            effective_selected[layer].pop("conditioning_cs", None)
    outputs = []
    from csasr.evaluation import canonical
    for layer in layers:
        for family, rhos in effective_selected[layer].items():
            for rho in sorted(set(rhos)):
                key = canonical_key(phase=phase, model=model, dataset=dataset, uid=str(row["utterance_id"]), family=family, side=side, layer=layer, rho=rho, decode_mode=mode)
                if accepted(phase, key) is not None:
                    stats["reused_rows"] += 1; continue
                started = time.monotonic()
                hook, _hook_masks = (base._whisper_hook(bundle, row, baseline, side, layer, directions[layer][family], rho) if model == "whisper" else base._qwen_hook(bundle, row, baseline, side, layer, directions[layer][family], rho))
                steered = base._generate(bundle, model, row, mode, hook)
                records = getattr(hook, "records", [])
                edited = sum(int(getattr(x, "active_positions", getattr(x, "active_frames", int(bool(getattr(x, "steered", False)))))) for x in records)
                energy = sum(float(getattr(x, "edit_norm", getattr(x, "edit_norm_sum", 0.0))) for x in records)
                if rho > 0 and edited <= 0:
                    raise RuntimeError(f"positive-rho local edit was zero for {key}")
                payload = {
                    "status": "PASS", "phase": phase, "regime": "oracle_tt", "model": model, "dataset": dataset,
                    "utterance_id": str(row["utterance_id"]), "reference": row["reference"], "baseline_hypothesis": baseline["text"],
                    "steered_hypothesis": steered["text"], "method_family": family, "method": family, "side": side,
                    "layer": int(layer), "rho": float(rho), "decode_mode": mode,
                    "intervention_site": "encoder_post_self_attn_residual_pre_ffn" if side == "encoder" else "decoder_post_cross_attn_residual",
                    "direction_hash": base._sha(directions[layer][family].tolist()),
                    "rank": int(analysis[layer]["n_A"] if family == "conditioning_cs" else base.dynamic_rank(analysis[layer]["n_A"], analysis[layer]["n_B"], directions[layer][family].shape[0])),
                    "eligible": True, "local_edited_position_count": edited,
                    "perturbation_diagnostics": {"edit_norm_sum": energy, "records": [getattr(x, "__dict__", {}) for x in records]},
                    "target_logit_diagnostics": None, "runtime_sec": time.monotonic() - started, "baseline_runtime_sec": baseline.get("runtime_sec"),
                    "metrics": canonical.corpus_metrics([str(row["reference"])], [str(steered["text"])]),
                    "rho_cache_reused": True, "analysis_passes": analysis[layer]["analysis_passes"],
                    "gold_leakage": {"analysis_sequence": "baseline_hypothesis", "gold_target_hidden_states": False, "gold_next_token_logits": False, "target_embedding_direction": False},
                    "provenance": {"phase": phase, "regime": "oracle_tt", "git_commit": base._git(), "source_run_id": os.environ.get("SLURM_JOB_ID", "local"), "model_revision": getattr(bundle, "revision", None), "panel_fingerprint": None, "fixed_direction_artifact": None, "direction_construction": "per_sample_dynamic_rank", "rho_cache": "one_direction_all_rho", "analysis_layer_collection": "all_requested_layers_one_pass"},
                    "canonical_key": key,
                }
                _atomic(row_path(phase, key), payload); outputs.append(payload); stats["new_rows"] += 1
    return outputs, stats


def _alias_qwen_standard(phase: str, rows: list[dict[str, Any]], stats: dict[str, int]) -> None:
    for greedy in rows:
        if greedy.get("model") != "qwen3_asr_1p7b" or greedy.get("decode_mode") != "greedy":
            continue
        key = canonical_key(phase=phase, model=greedy["model"], dataset=greedy["dataset"], uid=greedy["utterance_id"], family=greedy["method_family"], side=greedy["side"], layer=greedy["layer"], rho=greedy["rho"], decode_mode="official_standard")
        if accepted(phase, key) is not None:
            stats["reused_rows"] += 1; continue
        payload = dict(greedy); payload["decode_mode"] = "official_standard"; payload["canonical_key"] = key
        payload["provenance"] = dict(greedy["provenance"], decode_equivalence="official_standard == greedy", reused_from=greedy["canonical_key"], equivalence_hash=_sha({"baseline": greedy["baseline_hypothesis"], "steered": greedy["steered_hypothesis"], "direction_hash": greedy["direction_hash"]}))
        _atomic(row_path(phase, key), payload); stats["new_rows"] += 1


def run(model_arg: str, phase_arg: str, dataset: str, side: str, mode: str) -> int:
    model = MODEL_NAMES[model_arg]; phase = phase_for(phase_arg)
    settings = _settings(model, phase)
    selected = _selected_for_side(settings, side)
    if not selected:
        raise RuntimeError(f"no settings for {model}/{phase}/{side}")
    families = {(f, side) for layer in selected.values() for f in layer}
    # The explicit side-aware form above prevents encoder and decoder
    # Add-Unique eligibility manifests from being conflated.
    rows, panel_fp = _dataset_rows(phase, dataset, families, model)
    eligible = _rows_by_method(dataset, settings, model)
    allowed_ids = set().union(*(set(eligible[f]) for f in families))
    rows = [r for r in rows if str(r["utterance_id"]) in allowed_ids]
    shard_count = int(os.environ.get("A6_OTT_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("A6_OTT_SHARD_INDEX", "0"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise RuntimeError("invalid A6_OTT_SHARD_INDEX/A6_OTT_SHARD_COUNT")
    if shard_count > 1:
        rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    base_payload = _baseline(model, dataset, mode)
    bundle = base._model_bundle(model); bundle.model.eval()
    started = time.monotonic(); totals = {"direction_constructions": 0, "rho_cache_hits": 0, "new_rows": 0, "reused_rows": 0, "ineligible": []}
    all_new: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row["utterance_id"]); baseline = base_payload["rows"].get(uid)
        if baseline is None: raise RuntimeError(f"baseline missing {dataset}/{uid}")
        # A row can be eligible for one decoder family but not encoder.  The
        # side-specific settings/family filter is the scientific eligibility boundary.
        row_selected = {layer: {f: rs for f, rs in fams.items() if uid in set(eligible[(f, side)])} for layer, fams in selected.items()}
        row_selected = {layer: fams for layer, fams in row_selected.items() if fams}
        if not row_selected: continue
        # Result-first resumability is checked before any model analysis.  A
        # fully accepted sample must not pay for another all-layer analysis
        # pass merely to discover that every requested physical cell exists.
        pending_keys = []
        for layer, fams in row_selected.items():
            for family, rhos in fams.items():
                for rho in sorted(set(rhos)):
                    pending_keys.append(canonical_key(
                        phase=phase, model=model, dataset=dataset, uid=uid,
                        family=family, side=side, layer=layer, rho=rho,
                        decode_mode=mode))
        if pending_keys and all(accepted(phase, key) is not None for key in pending_keys):
            totals["reused_rows"] += len(pending_keys)
            continue
        try:
            out, stats = _run_selected_sample(bundle, model, phase, dataset, row, baseline, side, row_selected, mode)
        except RuntimeError as exc:
            if "empty oracle A/B group" not in str(exc):
                raise
            # A validated acoustic span can still fail the current model's
            # causal A/B mapping.  Keep the sample in the frozen population,
            # but resolve every requested method cell explicitly as
            # outcome-blind method-specific INELIGIBLE.
            stats = {"direction_constructions": 0, "rho_cache_hits": 0,
                     "new_rows": 0, "reused_rows": 0, "ineligible": []}
            for layer, fams in row_selected.items():
                for family, rhos in fams.items():
                    for rho in sorted(set(rhos)):
                        stats["ineligible"].append({
                            "canonical_key": canonical_key(
                                phase=phase, model=model, dataset=dataset,
                                uid=uid, family=family, side=side,
                                layer=layer, rho=rho, decode_mode=mode),
                            "status": "INELIGIBLE",
                            "reason": "empty_oracle_ab_group",
                            "outcome_consulted": False,
                        })
            out = []
        all_new.extend(out)
        for k in ("direction_constructions", "rho_cache_hits", "new_rows", "reused_rows"):
            totals[k] += stats.get(k, 0)
        totals["ineligible"].extend(stats.get("ineligible", []))
    if model == "qwen3_asr_1p7b" and mode == "greedy":
        # Alias every accepted greedy key in this shard, including keys from
        # an interrupted earlier attempt.  This keeps official-standard
        # materialization resumable without a second physical generation.
        alias_sources = list(all_new)
        for row in rows:
            uid = str(row["utterance_id"])
            row_selected = {layer: {f: rs for f, rs in fams.items() if uid in set(eligible[(f, side)])} for layer, fams in selected.items()}
            for layer, fams in row_selected.items():
                for family, rhos in fams.items():
                    for rho in sorted(set(rhos)):
                        key = canonical_key(phase=phase, model=model, dataset=dataset, uid=uid, family=family, side=side, layer=layer, rho=rho, decode_mode="greedy")
                        accepted_row = accepted(phase, key)
                        if accepted_row is not None:
                            alias_sources.append(accepted_row["payload"])
        _alias_qwen_standard(phase, alias_sources, totals)
    if totals["ineligible"]:
        payload = {"schema_version": "a6_ott_ineligible_v1", "status": "PASS", "phase": phase, "model": model, "dataset": dataset, "side": side, "mode": mode, "rows": totals["ineligible"]}
        _atomic(ROOT / ("confirm" if phase == "B" else "transfer") / "ineligible" / f"{model}_{dataset}_{side}_{mode}.json", payload)
        if model == "qwen3_asr_1p7b" and mode == "greedy":
            aliases = []
            for item in totals["ineligible"]:
                key = str(item["canonical_key"]); parts = key.split("|"); parts[8] = "official_standard"
                aliases.append({**item, "canonical_key": "|".join(parts), "equivalence_provenance": "official_standard == greedy"})
            target_phase = "confirm" if phase == "B" else "transfer"
            _atomic(ROOT / target_phase / "ineligible" / f"{model}_{dataset}_{side}_official_standard.json", {**payload, "mode": "official_standard", "rows": aliases})
    _atomic(ROOT / ("confirm" if phase == "B" else "transfer") / "runtime" / f"{model}_{dataset}_{side}_{mode}.json", {"status": "PASS", "phase": phase, "model": model, "dataset": dataset, "side": side, "mode": mode, "panel_fingerprint": panel_fp, **totals, "runtime_sec": time.monotonic() - started, "gpu_job_id": os.environ.get("SLURM_JOB_ID", "local")})
    print(json.dumps({"status": "PASS", "phase": phase, "model": model, "dataset": dataset, "side": side, "mode": mode, **totals, "runtime_sec": time.monotonic() - started}, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("whisper", "qwen", "qwen3_asr_1p7b"))
    ap.add_argument("--phase", required=True, choices=("B", "C"))
    ap.add_argument("--dataset", required=True, choices=("cs_dialogue_confirm", "ascend_confirm", "seame_dev_man", "seame_dev_sge"))
    ap.add_argument("--side", required=True, choices=("encoder", "decoder"))
    ap.add_argument("--mode", required=True, choices=("greedy", "official_standard"))
    args = ap.parse_args()
    return run(args.model, args.phase, args.dataset, args.side, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
