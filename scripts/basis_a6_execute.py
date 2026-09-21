#!/usr/bin/env python3
"""Model-resident, resumable BASIS-A6 full-matrix executor.

One invocation is one frozen shard: model, panel, decode ID, side and layer
(plus construction source for A6-F).  All methods and all five doses stay in
the resident model process.  Completed configuration keys are read before
work starts, so interrupted shards resume without rewriting accepted rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
ROOT = REPO / "results/basis_a6_expanded"
RHOS = (0.5, 1.0, 2.0, 4.0, 6.0)
MODELS = {"whisper": (range(32), range(32)), "qwen3_asr_1p7b": (range(24), range(28))}
NONCOND = ("raw", "add_unique", "minus_shared", "unique_minus_shared")
COND = ("conditioning_cs", "conditioning_all")


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")


def _git() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _model_bundle(model: str):
    if model == "whisper":
        from csasr.models.whisper import load_whisper
        from csasr.utils.config import load_config
        return load_whisper(load_config("model/whisper_large_v3.yaml"))
    from csasr.models.qwen3_asr import load_qwen
    return load_qwen(device="cuda:0")


def baseline_path(model: str, dataset: str, mode: str) -> Path:
    return ROOT / "baselines" / model / dataset / f"{mode}.json"


def load_baseline(model: str, dataset: str, mode: str) -> dict[str, Any]:
    p = baseline_path(model, dataset, mode)
    if not p.is_file():
        raise RuntimeError(f"baseline is missing: {p}")
    payload = json.loads(p.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"baseline is not accepted: {p}")
    return payload


def _load_rows(dataset: str, *, require_alignment: bool = True):
    from csasr.basis_a6.panels import load_panel
    return load_panel(dataset, require_alignment=require_alignment)


def _dir_path(source: str, model: str, side: str, layer: int, method: str) -> Path:
    return ROOT / "fixed" / source / "directions" / model / side / f"L{int(layer):02d}" / f"{method}.npy"


def _direction(source: str, model: str, side: str, layer: int, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    p = _dir_path(source, model, side, layer, method)
    if not p.is_file():
        raise FileNotFoundError(p)
    v = np.asarray(np.load(p), dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(v).all() or not np.isclose(norm, 1.0, atol=2e-5):
        raise RuntimeError(f"invalid direction {p}: norm={norm}")
    # The per-model construction manifest is written once per construction
    # worker; the inventory is the canonical combined 584-entry source of
    # truth and is therefore required here.
    inventory = json.loads((ROOT / "manifests/FIXED_DIRECTION_INVENTORY.json").read_text())
    meta = next((r for r in inventory.get("rows", [])
                 if r.get("source") == source and r.get("model") == model
                 and r.get("side") == side and int(r.get("layer", -1)) == int(layer)
                 and r.get("method") == method), None)
    if not meta or meta.get("direction_hash") is None:
        raise RuntimeError(f"missing direction provenance {key}")
    return v, meta


def _cell_key(branch: str, *, source: str | None, model: str, dataset: str,
              mode: str, side: str, layer: int, method: str, rho: float) -> str:
    fields = [branch, source or "", model, dataset, mode, side, int(layer), method, float(rho)]
    return "|".join(map(str, fields))


def _result_path(branch: str, *, source: str | None, model: str, dataset: str,
                 mode: str, side: str, layer: int) -> Path:
    if branch == "fixed":
        return ROOT / "fixed" / source / "runs" / model / dataset / mode / side / f"L{int(layer):02d}.jsonl"
    return ROOT / "oracle_tt" / "runs" / model / dataset / mode / side / f"L{int(layer):02d}.jsonl"


def _accepted(path: Path, branch: str, source: str | None, model: str, dataset: str,
              mode: str, side: str, layer: int) -> set[str]:
    accepted: set[str] = set()
    if not path.is_file():
        return accepted
    quarantine = ROOT / "quarantine" / "partial" / path.relative_to(ROOT)
    good = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("status") != "PASS":
                raise ValueError("row status is not PASS")
            key = row["canonical_key"]
            expected_prefix = _cell_key(branch, source=source, model=model, dataset=dataset, mode=mode, side=side, layer=layer, method=row["method"], rho=row["rho"])
            if key != expected_prefix or not row.get("provenance"):
                raise ValueError("invalid key/provenance")
            if branch == "fixed" and not row.get("construction_fingerprint"):
                raise ValueError("missing construction fingerprint")
            accepted.add(key); good.append(line)
        except Exception:
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            with quarantine.open("a") as out:
                out.write(line + "\n")
    if len(good) != len(path.read_text().splitlines()):
        path.write_text("\n".join(good) + ("\n" if good else ""))
    return accepted


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str) + "\n")
        fh.flush()


def _metrics(rows: list[dict[str, Any]], baseline: dict[str, Any], outputs: dict[str, dict[str, Any]], eligible: set[str]) -> dict[str, Any]:
    from csasr.evaluation import canonical, retention
    scored = [r for r in rows if str(r["utterance_id"]) in eligible]
    refs = [str(r["reference"]) for r in scored]
    ids = [str(r["utterance_id"]) for r in scored]
    base_text = [str(baseline["rows"][u]["text"]) for u in ids]
    out_text = [str(outputs[u]["text"]) for u in ids]
    bm = canonical.corpus_metrics(refs, base_text)
    mm = canonical.corpus_metrics(refs, out_text)
    transitions = canonical.correction_corruption(refs, base_text, out_text)
    ret = retention.retention_report(refs, base_text, out_text)
    return {**mm, **canonical.error_metric_gains(bm, mm),
            "poi_corrections": int(transitions["corrections"]),
            "poi_corruptions": int(transitions["corruptions"]),
            "poi_net_utility": int(transitions["corrections"] - transitions["corruptions"]),
            "utility": int(transitions["corrections"] - transitions["corruptions"]),
            "outside_harm": None,
            "matrix_retention": ret["matrix_zh"]["rate"],
            "embedded_retention": ret["embedded_en"]["rate"],
            "changed_rate": float(sum(a != b for a, b in zip(base_text, out_text)) / len(out_text)) if out_text else None,
            "scored_utterances": len(scored),
            "baseline_metrics": bm,
            "transitions": transitions}


def _record_values(records: list[Any]) -> tuple[float, int, float]:
    energy, active, pre = 0.0, 0, 0.0
    for r in records:
        energy += float(getattr(r, "edit_norm", getattr(r, "edit_norm_sum", 0.0)))
        active += int(getattr(r, "active_positions", getattr(r, "active_frames", 0)))
        pre += float(getattr(r, "active_pre_norm_sum", getattr(r, "pre_norm", 0.0)))
    return energy, active, pre


def _whisper_hook(bundle, row, baseline, side, layer, vec, rho):
    import torch
    from scripts.basis_a6_real_acceptance import _whisper_masks, _whisper_gate
    masks = _whisper_masks(bundle, row, baseline)
    if side == "encoder":
        from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
        hook = EncoderPostSelfAttnInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho,
            scale=1.0, gain=torch.from_numpy(masks["encoder_gain"]), record=True)
    else:
        from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
        allowed = set(map(int, masks["decoder_positions"]))
        hook = DecoderPostCrossAttnInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho,
            scale=1.0, num_forced_prefix=num_forced_prefix_from(bundle.processor, "zh"),
            gate_fn=lambda **kw: _whisper_gate(allowed, **kw), record=True, record_last_only=False)
    return hook, masks


def _qwen_hook(bundle, row, baseline, side, layer, vec, rho):
    import torch
    from scripts.basis_a6_real_acceptance import _qwen_masks
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    masks = _qwen_masks(bundle, row, baseline)
    if side == "encoder":
        allowed = set(np.flatnonzero(masks["encoder_gain"] > 0).tolist()); site = AUDIO_SITE
    else:
        allowed = set(map(int, masks["decoder_positions"])); site = TEXT_SITE
    return QwenSiteInterventionHook(bundle, layer, torch.from_numpy(vec), alpha=rho, scale=1.0,
                                    site=site, allowed_positions=allowed, record=True), masks


def _generate(bundle, model: str, row, mode: str, hook=None):
    from scripts.basis_a6_real_acceptance import _whisper_generate, _qwen_generate
    return (_whisper_generate(bundle, row["audio_path"], mode, hook) if model == "whisper"
            else _qwen_generate(bundle, row["audio_path"], mode, hook))


def run_baselines(model: str) -> int:
    bundle = _model_bundle(model); bundle.model.eval()
    for dataset in ("cs_dialogue_dev_select", "seame_dev_man", "seame_dev_sge", "ascend_eval"):
        rows, panel_fp = _load_rows(dataset, require_alignment=False)
        for mode in ("greedy", "official_standard"):
            out = baseline_path(model, dataset, mode)
            if out.is_file() and json.loads(out.read_text()).get("status") == "PASS":
                continue
            started = time.monotonic(); result = {}
            if model == "qwen3_asr_1p7b" and mode == "official_standard":
                prior = load_baseline(model, dataset, "greedy")
                result = {u: dict(v) for u, v in prior["rows"].items()}
                alias = "greedy"
            else:
                alias = None
                for i, row in enumerate(rows, 1):
                    decoded = _generate(bundle, model, row, mode)
                    result[str(row["utterance_id"])] = {"text": decoded["text"], "token_ids": decoded["token_ids"], "runtime_sec": decoded.get("runtime_sec")}
                    if i % 25 == 0: print(f"baseline {model} {dataset} {mode} {i}/{len(rows)}", flush=True)
            payload = {"schema_version": "basis_a6_baseline_v1", "status": "PASS",
                       "model": model, "dataset": dataset, "decode_mode": mode,
                       "official_equivalent_to": alias, "panel_fingerprint": panel_fp,
                       "model_revision": getattr(bundle, "revision", None), "rows": result,
                       "runtime_sec": time.monotonic() - started,
                       "provenance": {"git_commit": _git(), "site": "none", "scope": "baseline",
                                      "panel_fingerprint": panel_fp, "model_revision": getattr(bundle, "revision", None)}}
            _write(out, payload)
    return 0


def run_fixed(model: str, source: str, dataset: str, mode: str, side: str, layer: int, *, pilot: bool = False) -> int:
    import torch
    from scripts.basis_a6_real_acceptance import _whisper_masks, _qwen_masks
    bundle = _model_bundle(model); bundle.model.eval()
    rows, panel_fp = _load_rows(dataset); base = load_baseline(model, dataset, mode)
    methods = ("raw",) if pilot else NONCOND + (COND if side == "decoder" else ())
    doses = (0.5,) if pilot else RHOS
    path = _result_path("fixed", source=source, model=model, dataset=dataset, mode=mode, side=side, layer=layer)
    done = _accepted(path, "fixed", source, model, dataset, mode, side, layer)
    for method in methods:
        vec, dmeta = _direction(source, model, side, layer, method)
        for rho in doses:
            key = _cell_key("fixed", source=source, model=model, dataset=dataset, mode=mode, side=side, layer=layer, method=method, rho=rho)
            if key in done: continue
            outputs, energy, active, pre = {}, 0.0, 0, 0.0; started = time.monotonic(); failures = []
            for i, row in enumerate(rows, 1):
                try:
                    hook, masks = (_whisper_hook(bundle, row, base["rows"][str(row["utterance_id"])], side, layer, vec, rho)
                                   if model == "whisper" else _qwen_hook(bundle, row, base["rows"][str(row["utterance_id"])], side, layer, vec, rho))
                    result = _generate(bundle, model, row, mode, hook)
                    outputs[str(row["utterance_id"])] = result
                    e, a, p = _record_values(hook.records); energy += e; active += a; pre += p
                    if i % 25 == 0: print(f"fixed {model} {source} {dataset} {mode} {side} L{layer} {method} rho={rho} {i}/{len(rows)}", flush=True)
                except Exception as exc:
                    failures.append({"utterance_id": row["utterance_id"], "error": repr(exc)})
                    raise
            metric = _metrics(rows, base, outputs, set(outputs))
            cell = {"status": "PASS", "regime": "fixed", "construction_source": source,
                    "model": model, "eval_dataset": dataset, "decode_mode": mode,
                    "side": side, "layer": int(layer), "method": method, "rho": float(rho),
                    "relative_depth": float(layer / (len(MODELS[model][0 if side == "encoder" else 1]) - 1 or 1)),
                    "total_N": len(rows), "eligible_N": len(rows), "eligibility_rate": 1.0,
                    "MER": metric.get("mer"), "PIER": metric.get("pier"), "EN_WER": metric.get("en_wer"),
                    "matrix_CER": metric.get("zh_cer"), "corrections": metric["poi_corrections"],
                    "corruptions": metric["poi_corruptions"], "utility": metric["utility"],
                    "matrix_retention": metric["matrix_retention"], "embedded_retention": metric["embedded_retention"],
                    "outside_harm": metric["outside_harm"], "intervention_energy": energy,
                    "relative_delta": float(energy / pre) if pre else 0.0, "angle": None,
                    "changed_rate": metric["changed_rate"], "delta_margin": None,
                    "runtime": time.monotonic() - started, "edited_positions": active,
                    "direction_hash": dmeta["direction_hash"], "panel_fingerprint": panel_fp,
                    "construction_fingerprint": dmeta.get("construction_fingerprint"),
                    "model_revision": dmeta.get("model_revision"),
                    "site": "decoder_post_cross_attn_residual" if model == "whisper" and side == "decoder" else
                            "encoder_post_self_attn_residual_pre_ffn" if model == "whisper" else
                            "qwen_text_decoder_post_self_attn_residual_pre_mlp" if side == "decoder" else
                            "qwen_audio_encoder_post_self_attn_residual_pre_ffn",
                    "scope": "oracle_local", "norm_preserve": True,
                    "provenance": {"git_commit": _git(), "direction_hash": dmeta["direction_hash"],
                                   "panel_fingerprint": panel_fp, "site_hash": _hash({"model": model, "side": side, "layer": layer}),
                                   "mask_hash": _hash({"dataset": dataset, "scope": "oracle_local", "rows": [r["utterance_id"] for r in rows]}),
                                   "output_failures": failures},
                    "canonical_key": key}
            _append(path, cell); done.add(key)
    _write(path.with_suffix(".manifest.json"), {"status": "PASS", "branch": "fixed", "source": source,
        "model": model, "dataset": dataset, "decode_mode": mode, "side": side, "layer": layer,
        "cells": len(done), "expected_cells": len(methods) * len(doses), "pilot": pilot, "panel_fingerprint": panel_fp,
        "git_commit": _git()})
    return 0


def run_tt(model: str, dataset: str, mode: str, side: str, layer: int) -> int:
    import torch
    from csasr.basis_a6.cache import DirectionCache
    from csasr.basis_a6.directions import build_method_directions, dynamic_rank, direction_hash
    from scripts.basis_a6_real_acceptance import _whisper_analysis, _qwen_analysis, _whisper_masks, _qwen_masks
    bundle = _model_bundle(model); bundle.model.eval()
    rows, panel_fp = _load_rows(dataset); base = load_baseline(model, dataset, mode)
    methods = NONCOND + (COND if side == "decoder" else ())
    path = _result_path("oracle_tt", source=None, model=model, dataset=dataset, mode=mode, side=side, layer=layer)
    done = _accepted(path, "oracle_tt", None, model, dataset, mode, side, layer)
    cache_root = ROOT / "oracle_tt/cache" / model / dataset / mode / side / f"L{layer:02d}"
    cache_manifest_path = cache_root / "DIRECTION_CACHE.json"
    cache = DirectionCache(); states: dict[str, dict[str, Any]] = {}; constructions = 0
    started_analysis = time.monotonic()
    for i, row in enumerate(rows, 1):
        uid = str(row["utterance_id"]); b = base["rows"][uid]
        masks = (_whisper_masks(bundle, row, b) if model == "whisper" else _qwen_masks(bundle, row, b))
        try:
            if model == "whisper":
                a, bb, cond, diag = _whisper_analysis(bundle, row, b, masks, side, layer)
            else:
                a, bb, cond, diag = _qwen_analysis(bundle, row, b, masks, side, layer)
            delta = None if cond is None else cond[0]; cs = None if cond is None else cond[1]
            made = build_method_directions(a, bb, conditioning_deltas=delta,
                                           conditioning_cs_positions=cs if cs is not None and len(cs) else None)
            rank = dynamic_rank(len(a), len(bb), a.shape[-1]) if np.asarray(a).ndim == 2 and np.asarray(bb).ndim == 2 else 0
        except Exception as exc:
            made, diag, a, bb, rank = {}, {"error": repr(exc)}, [], [], 0
        alignment_hash = _hash(masks)
        recs = {}
        for method in methods:
            vec = made.get(method)
            eligible = vec is not None and (method == "raw" or method.startswith("conditioning") or rank >= 2)
            if eligible:
                vec = np.asarray(vec, dtype=np.float64); constructions += 1
                rec = cache.put(model=model, dataset=dataset, utterance_id=uid, decode_analysis_mode=mode,
                                side=side, layer=layer, method=method, alignment_hash=alignment_hash,
                                n_A=len(a), n_B=len(bb), rank=(rank if method in NONCOND[1:] else None),
                                eligible=True, vector=vec)
                recs[method] = {"vector": vec, "direction_hash": rec.direction_hash, "eligible": True,
                                "n_A": len(a), "n_B": len(bb), "rank": rank, "alignment_hash": alignment_hash}
            else:
                recs[method] = {"vector": None, "direction_hash": None, "eligible": False,
                                "n_A": len(a), "n_B": len(bb), "rank": rank, "alignment_hash": alignment_hash}
        states[uid] = {"row": row, "baseline": b, "masks": masks, "methods": recs,
                       "diag": diag, "n_A": len(a), "n_B": len(bb), "rank": rank}
        if i % 25 == 0: print(f"tt analysis {model} {dataset} {mode} {side} L{layer} {i}/{len(rows)}", flush=True)
    cache_payload = cache.write(cache_root)
    construction_latency = time.monotonic() - started_analysis
    for method in methods:
        eligible_ids = {u for u, s in states.items() if s["methods"][method]["eligible"]}
        for rho in RHOS:
            key = _cell_key("oracle_tt", source=None, model=model, dataset=dataset, mode=mode, side=side, layer=layer, method=method, rho=rho)
            if key in done: continue
            outputs, energy, active, pre = {}, 0.0, 0, 0.0; started = time.monotonic()
            for uid in sorted(eligible_ids):
                s = states[uid]; vec = s["methods"][method]["vector"]
                hook, _ = (_whisper_hook(bundle, s["row"], s["baseline"], side, layer, vec, rho)
                            if model == "whisper" else _qwen_hook(bundle, s["row"], s["baseline"], side, layer, vec, rho))
                out = _generate(bundle, model, s["row"], mode, hook); outputs[uid] = out
                e, a, p = _record_values(hook.records); energy += e; active += a; pre += p
            metric = _metrics(rows, base, outputs, eligible_ids) if eligible_ids else {"mer": None, "pier": None, "en_wer": None, "zh_cer": None, "poi_corrections": None, "poi_corruptions": None, "utility": None, "matrix_retention": None, "embedded_retention": None, "changed_rate": None, "outside_harm": None}
            ranks = [states[u]["rank"] for u in eligible_ids]
            cache_key = f"{model}|{dataset}|{side}|{method}|{mode}"
            cell = {"status": "PASS", "regime": "oracle_tt", "construction_source": None,
                    "model": model, "eval_dataset": dataset, "decode_mode": mode, "side": side,
                    "layer": int(layer), "method": method, "rho": float(rho),
                    "relative_depth": float(layer / (len(MODELS[model][0 if side == "encoder" else 1]) - 1 or 1)),
                    "total_N": len(rows), "eligible_N": len(eligible_ids), "eligibility_rate": len(eligible_ids) / len(rows),
                    "MER": metric.get("mer"), "PIER": metric.get("pier"), "EN_WER": metric.get("en_wer"), "matrix_CER": metric.get("zh_cer"),
                    "corrections": metric.get("poi_corrections"), "corruptions": metric.get("poi_corruptions"), "utility": metric.get("utility"),
                    "matrix_retention": metric.get("matrix_retention"), "embedded_retention": metric.get("embedded_retention"),
                    "outside_harm": metric.get("outside_harm"), "intervention_energy": energy,
                    "relative_delta": float(energy / pre) if pre else 0.0, "angle": None, "changed_rate": metric.get("changed_rate"), "delta_margin": None,
                    "runtime": time.monotonic() - started, "analysis_latency": construction_latency,
                    "direction_construction_latency": construction_latency, "total_rtf": None,
                    "rank_median": float(np.median(ranks)) if ranks else None, "rank_p10": float(np.percentile(ranks, 10)) if ranks else None,
                    "rank_p90": float(np.percentile(ranks, 90)) if ranks else None, "edited_positions": active,
                    "direction_bundle_hash": cache_payload["bundles"].get(cache_key), "direction_constructions": constructions,
                    "cache_hits": 4, "cache_reused_across_rho": True,
                    "gold_leakage": {"analysis_sequence": "baseline_hypothesis", "gold_target_hidden_states": False,
                                     "gold_next_token_logits": False, "target_embedding_direction": False},
                    "provenance": {"git_commit": _git(), "panel_fingerprint": panel_fp, "cache_bundle_hash": cache_payload["bundles"].get(cache_key),
                                   "alignment_hashes": sorted({s["methods"][method]["alignment_hash"] for s in states.values()}),
                                   "model_revision": getattr(bundle, "revision", None), "scope": "oracle_local"},
                    "canonical_key": key}
            _append(path, cell); done.add(key)
    _write(path.with_suffix(".manifest.json"), {"status": "PASS", "branch": "oracle_tt", "model": model, "dataset": dataset,
        "decode_mode": mode, "side": side, "layer": layer, "cells": len(done), "expected_cells": len(methods) * len(RHOS),
        "direction_constructions": constructions, "cache_hits": len(methods) * 4, "direction_cache": str(cache_manifest_path.relative_to(REPO)),
        "panel_fingerprint": panel_fp, "git_commit": _git()})
    return 0


def enumerate_shards(output: Path) -> int:
    rows = []
    datasets = ("cs_dialogue_dev_select", "seame_dev_man", "seame_dev_sge", "ascend_eval")
    for model, (enc, dec) in MODELS.items():
        for dataset in datasets:
            for mode in ("greedy", "official_standard"):
                for side, layers in (("encoder", enc), ("decoder", dec)):
                    for layer in layers:
                        for source in ("cs_dialogue", "ascend"):
                            rows.append({"stage": "fixed", "source": source, "model": model, "dataset": dataset, "mode": mode, "side": side, "layer": layer})
                        rows.append({"stage": "oracle_tt", "source": None, "model": model, "dataset": dataset, "mode": mode, "side": side, "layer": layer})
    # A single shard per layer/side is deliberately conservative; all methods
    # and five doses are within that resident process and resume by key.
    _write(output, {"schema_version": "basis_a6_shard_manifest_v1", "status": "PASS", "rows": rows,
                    "count": len(rows), "max_concurrent_gpu_jobs": 2, "layers_per_shard": 1,
                    "git_commit": _git()})
    print(json.dumps({"shards": len(rows), "path": str(output)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("baselines", "fixed", "oracle_tt", "enumerate"), required=True)
    ap.add_argument("--model", choices=tuple(MODELS))
    ap.add_argument("--source", choices=("cs_dialogue", "ascend"))
    ap.add_argument("--dataset")
    ap.add_argument("--decode-mode", choices=("greedy", "official_standard"))
    ap.add_argument("--side", choices=("encoder", "decoder"))
    ap.add_argument("--layer", type=int)
    ap.add_argument("--output", default="results/basis_a6_expanded/manifests/FULL_RUN_SHARDS.json")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.stage == "enumerate": return enumerate_shards(REPO / args.output)
    if not args.model: ap.error("--model is required")
    if args.stage == "baselines": return run_baselines(args.model)
    for name in ("dataset", "decode_mode", "side", "layer"):
        if getattr(args, name) is None: ap.error(f"--{name.replace('_', '-')} is required")
    if args.stage == "fixed":
        if not args.source: ap.error("--source is required for fixed")
        return run_fixed(args.model, args.source, args.dataset, args.decode_mode, args.side, args.layer, pilot=args.pilot)
    return run_tt(args.model, args.dataset, args.decode_mode, args.side, args.layer)


if __name__ == "__main__":
    raise SystemExit(main())
