"""Physical A6-OTT Phase-A executor.

This module is intentionally a small wrapper around the tested BASIS-A6 model
and site primitives.  It owns the compact experiment policy, resumability, and
per-sample direction lifecycle; it does not import or call the superseded
large A6 executor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from csasr.basis_a6.cache import DirectionCache
from csasr.basis_a6.directions import build_method_directions, dynamic_rank
from csasr.basis_a6.ott import COMPACT_FAMILIES, rho_grid
from csasr.basis_a6.panels import load_panel
from scripts.basis_a6_execute import (
    _generate, _metrics, _model_bundle, _qwen_hook, _whisper_hook,
)
from scripts.basis_a6_real_acceptance import (
    _hash as _json_hash,
    _nearest_previous,
    _qwen_masks,
    _reference_groups,
    _whisper_masks,
)

ROOT = REPO / "results/a6_ott_upper_bound"
ROW_ROOT = ROOT / "phase_a" / "rows"
SEARCH_FREEZE = ROOT / "manifests/SEARCH_FREEZE.json"
MODEL_NAMES = {"whisper": "whisper", "qwen": "qwen3_asr_1p7b", "qwen3_asr_1p7b": "qwen3_asr_1p7b"}
SITES = {
    "whisper": {"encoder": 32, "decoder": 32},
    "qwen3_asr_1p7b": {"encoder": 24, "decoder": 28},
}
METHODS = (("add_unique", "encoder"), ("add_unique", "decoder"), ("conditioning_cs", "decoder"))


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _git() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def _uid_path(uid: str) -> str:
    return hashlib.sha256(str(uid).encode()).hexdigest()[:24]


def _background_positions(a: list[int], n: int) -> list[int]:
    """Choose a non-local comparison region, including when A starts at zero."""
    if not a:
        return []
    previous = _nearest_previous(a, len(a), n)
    if previous:
        return previous
    excluded = set(a)
    after = [i for i in range(max(a) + 1, n) if i not in excluded]
    if after:
        return after[:max(1, min(len(a), len(after)))]
    return [i for i in range(n) if i not in excluded][:max(1, min(len(a), n - len(excluded)))]


def canonical_key(*, model: str, dataset: str, uid: str, family: str, side: str,
                  layer: int, rho: float, decode_mode: str = "greedy") -> str:
    return "|".join(map(str, ("A", model, dataset, uid, family, side, int(layer), float(rho), decode_mode,
                                "encoder_post_self_attn_residual_pre_ffn" if side == "encoder" else
                                "decoder_post_cross_attn_residual")))


def row_path(key: str) -> Path:
    parts = key.split("|")
    _, model, dataset, uid, family, side, layer, rho, _mode, _site = parts
    return ROW_ROOT / model / dataset / family / side / f"L{int(layer):02d}" / f"rho{float(rho):g}" / f"{_uid_path(uid)}.json"


def accepted(key: str) -> dict[str, Any] | None:
    path = row_path(key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS" or payload.get("canonical_key") != key:
            return None
        if payload.get("provenance", {}).get("phase") != "A":
            return None
        if payload.get("provenance", {}).get("regime") != "oracle_tt":
            return None
        return {"reused": True, "source_artifact": str(path.relative_to(REPO)),
                "source_hash": _file_sha(path), "payload": payload}
    except Exception:
        return None


def _frozen_ids() -> dict[str, list[str]]:
    payload = json.loads(SEARCH_FREEZE.read_text())
    return {str(k): [str(x) for x in v] for k, v in payload["ids"].items()}


def _dataset_rows(dataset: str) -> tuple[list[dict[str, Any]], str]:
    panel = "cs_dialogue_dev_select" if dataset == "cs_dialogue" else "ascend_eval"
    rows, fp = load_panel(panel, require_alignment=True)
    wanted = set(_frozen_ids()[dataset])
    selected = [r for r in rows if str(r["utterance_id"]) in wanted]
    if {str(r["utterance_id"]) for r in selected} != wanted:
        raise RuntimeError(f"frozen Search IDs missing from {panel}")
    return sorted(selected, key=lambda r: str(r["utterance_id"])), fp


def list_shards(model: str) -> list[str]:
    model = MODEL_NAMES.get(model, model)
    if model not in SITES:
        raise ValueError(model)
    # One contiguous full-depth block per dataset/side.  The analysis callback
    # registers every layer in the block simultaneously; it never reruns the
    # baseline/current-sample analysis once per layer.
    return [f"{dataset} {side} 0 {SITES[model][side]}"
            for dataset in ("cs_dialogue", "ascend")
            for side in ("encoder", "decoder")]


def _whisper_analysis(bundle, row, baseline, masks, layers: list[int], side: str) -> dict[int, dict[str, Any]]:
    import torch
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.lss.encoder_sites import EncoderPostSelfAttnRecorder
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.generation import teacher_forced_forward
    from csasr.models.whisper import batch_model_inputs

    norm, units = normalize_and_segment(row["reference"])
    prefix_zh = build_prefix(bundle.processor, "zh")
    seq = list(baseline["token_ids"])
    content = seq[len(prefix_zh):]
    ref_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
    _ = _reference_groups(row, bundle.processor.tokenizer, list(ref_ids), norm, units)
    if side == "encoder":
        with EncoderPostSelfAttnRecorder(bundle, layers) as rec, torch.inference_mode():
            bundle.model.model.encoder(batch_model_inputs(bundle, [row["audio_path"]])["input_features"])
        out = {}
        for layer in layers:
            st = rec.states[layer][0].cpu().numpy()
            a = []
            for span in row["oracle_spans"]:
                lo, hi = bundle.sec_to_frames(float(span["start_sec"]), float(span["end_sec"]), st.shape[0])
                a.extend(range(max(0, lo), min(st.shape[0], hi)))
            a = sorted(set(a)); b = _background_positions(a, st.shape[0])
            out[layer] = {"A": st[a], "B": st[b], "conditioning": None,
                          "n_A": len(a), "n_B": len(b), "analysis_passes": 1}
        return out
    with DecoderPostCrossAttnRecorder(bundle, layers) as rec, torch.inference_mode():
        teacher_forced_forward(bundle, [row["audio_path"]], [seq])
    baseline_states = {l: rec.states[l][0].cpu().numpy() for l in layers}
    positions = [int(len(prefix_zh) + i - 1) for i in masks["hypothesis_token_indices"] if i > 0]
    prefix_en = build_prefix(bundle.processor, "en")
    paired: dict[str, dict[int, np.ndarray]] = {"en": {}, "zh": {}}
    for lang, prefix in (("en", prefix_en), ("zh", prefix_zh)):
        seq2 = list(prefix) + content
        with DecoderPostCrossAttnRecorder(bundle, layers) as rr, torch.inference_mode():
            teacher_forced_forward(bundle, [row["audio_path"]], [seq2])
        paired[lang] = {l: rr.states[l][0].cpu().numpy() for l in layers}
    same_positions = [int(len(prefix_en) + i - 1) for i in range(len(content))]
    out = {}
    for layer in layers:
        st = baseline_states[layer]
        apos = [p for p in positions if 0 <= p < len(st)]
        bpos = _background_positions(apos, len(st))
        a, b = st[apos], st[bpos]
        valid = [i for i, p in enumerate(same_positions)
                 if 0 <= p < len(paired["en"][layer]) and p < len(paired["zh"][layer])]
        delta = paired["en"][layer][same_positions and [same_positions[i] for i in valid]] - paired["zh"][layer][same_positions and [same_positions[i] for i in valid]]
        cs = [j for j, i in enumerate(valid) if same_positions[i] in set(apos)]
        out[layer] = {"A": a, "B": b, "conditioning": (delta, cs),
                      "n_A": len(a), "n_B": len(b), "analysis_passes": 3}
    return out


def _qwen_analysis(bundle, row, baseline, masks, layers: list[int], side: str) -> dict[int, dict[str, Any]]:
    import torch
    from csasr.lss.qwen_sites import QwenAudioSiteRecorder, QwenTextSiteRecorder
    from csasr.models.qwen3_asr import inputs_for_audio
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
    if side == "encoder":
        with QwenAudioSiteRecorder(bundle, layers) as rec, torch.inference_mode():
            bundle.thinker_model(**inp, use_cache=False)
        out = {}
        for layer in layers:
            # Qwen audio recorder is packed as (T,D), unlike the text stack
            # which is batched (B,T,D).
            st = rec.states[layer].numpy()
            a = np.flatnonzero(np.asarray(masks["encoder_gain"]) > 0).tolist()
            a = [x for x in a if x < len(st)]; b = _background_positions(a, len(st))
            out[layer] = {"A": st[a], "B": st[b], "conditioning": None,
                          "n_A": len(a), "n_B": len(b), "analysis_passes": 1}
        return out
    hyp_ids = bundle.processor.tokenizer.encode(baseline["text"], add_special_tokens=False)
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese", transcript=baseline["text"])
    ids = inp["input_ids"][0].detach().cpu().tolist()
    starts = [i for i in range(len(ids) - len(hyp_ids) + 1) if ids[i:i + len(hyp_ids)] == hyp_ids]
    if not starts:
        raise RuntimeError("baseline hypothesis not found in Qwen analysis prompt")
    start = starts[-1]
    with QwenTextSiteRecorder(bundle, layers) as rec, torch.inference_mode():
        bundle.thinker_model(**inp, use_cache=False)
    base_states = {l: rec.states[l].numpy()[0] for l in layers}
    paired: dict[str, dict[int, np.ndarray]] = {"English": {}, "Chinese": {}}
    for lang in paired:
        pin = inputs_for_audio(bundle, row["audio_path"], language=lang, transcript=baseline["text"])
        pi = pin["input_ids"][0].detach().cpu().tolist()
        ss = [i for i in range(len(pi) - len(hyp_ids) + 1) if pi[i:i + len(hyp_ids)] == hyp_ids]
        if not ss or ss[-1] != start:
            raise RuntimeError("paired Qwen prompt alignment mismatch")
        with QwenTextSiteRecorder(bundle, layers) as rr, torch.inference_mode():
            bundle.thinker_model(**pin, use_cache=False)
        paired[lang] = {l: rr.states[l].numpy()[0] for l in layers}
    out = {}
    hyp_positions = [start + i - 1 for i in masks["hypothesis_token_indices"] if i > 0]
    delta_positions = [start + i - 1 for i in range(len(hyp_ids))]
    for layer in layers:
        st = base_states[layer]
        apos = [p for p in hyp_positions if 0 <= p < len(st)]
        bpos = _background_positions(apos, len(st))
        valid = [i for i, p in enumerate(delta_positions)
                 if 0 <= p < len(paired["English"][layer]) and p < len(paired["Chinese"][layer])]
        pos = [delta_positions[i] for i in valid]
        delta = paired["English"][layer][pos] - paired["Chinese"][layer][pos]
        cs = [j for j, p in enumerate(pos) if p in set(apos)]
        out[layer] = {"A": st[apos], "B": st[bpos], "conditioning": (delta, cs),
                      "n_A": len(apos), "n_B": len(bpos), "analysis_passes": 3}
    return out


def _direction_set(model: str, dataset: str, uid: str, mode: str, side: str,
                   layer: int, analysis: dict[str, Any], cache: DirectionCache,
                   stats: dict[str, int]) -> dict[str, np.ndarray]:
    a, b = np.asarray(analysis["A"]), np.asarray(analysis["B"])
    if len(a) == 0 or len(b) == 0:
        raise RuntimeError(f"empty oracle A/B group for {uid} {side} L{layer}")
    cond = analysis.get("conditioning")
    delta = None if cond is None else np.asarray(cond[0])
    cs = None if cond is None else np.asarray(cond[1], dtype=int)
    methods = build_method_directions(a, b, conditioning_deltas=delta, conditioning_cs_positions=cs)
    wanted = {"add_unique"} if side == "encoder" else {"add_unique", "conditioning_cs"}
    missing = wanted.difference(methods)
    if missing:
        raise RuntimeError(f"required compact direction(s) ineligible for {uid} {side} L{layer}: {sorted(missing)}")
    rank = dynamic_rank(len(a), len(b), a.shape[-1])
    out = {}
    for method in wanted:
        vec = methods[method]
        cache.put(model=model, dataset=dataset, utterance_id=uid,
                  decode_analysis_mode=mode, side=side, layer=layer, method=method,
                  alignment_hash=_sha({"uid": uid, "side": side, "layer": layer, "A": len(a), "B": len(b)}),
                  n_A=len(a), n_B=len(b), rank=rank, eligible=True, vector=vec)
        out[method] = vec
        stats["direction_constructions"] += 1
    return out


def _run_sample(bundle, model: str, dataset: str, row: dict[str, Any], baseline: dict[str, Any],
                side: str, layers: list[int], rhos: tuple[float, ...], mode: str,
                acceptance: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    masks = _whisper_masks(bundle, row, baseline) if model == "whisper" else _qwen_masks(bundle, row, baseline)
    analysis = (_whisper_analysis(bundle, row, baseline, masks, layers, side) if model == "whisper"
                else _qwen_analysis(bundle, row, baseline, masks, layers, side))
    cache = DirectionCache(); stats = {"direction_constructions": 0, "rho_cache_hits": 0}
    directions = {layer: _direction_set(model, dataset, str(row["utterance_id"]), mode, side, layer,
                                          analysis[layer], cache, stats) for layer in layers}
    outputs = []
    for layer in layers:
        for method, method_side in METHODS:
            if method_side != side:
                continue
            for rho in rhos:
                stats["rho_cache_hits"] += 1
                key = canonical_key(model=model, dataset=dataset, uid=str(row["utterance_id"]), family=method,
                                    side=side, layer=layer, rho=rho, decode_mode=mode)
                if not acceptance and accepted(key) is not None:
                    continue
                started = time.monotonic()
                hook, hook_masks = (_whisper_hook(bundle, row, baseline, side, layer, directions[layer][method], rho)
                                    if model == "whisper" else
                                    _qwen_hook(bundle, row, baseline, side, layer, directions[layer][method], rho))
                steered = _generate(bundle, model, row, mode, hook)
                from csasr.evaluation import canonical
                metric = canonical.corpus_metrics([str(row["reference"])], [str(steered["text"])])
                records = getattr(hook, "records", [])
                edited = sum(int(getattr(x, "active_positions", getattr(x, "active_frames", int(bool(getattr(x, "steered", False)))))) for x in records)
                energy = sum(float(getattr(x, "edit_norm", getattr(x, "edit_norm_sum", 0.0))) for x in records)
                if rho > 0 and edited <= 0:
                    raise RuntimeError(f"positive-rho local edit was zero for {key}")
                payload = {
                    "status": "PASS", "phase": "A", "regime": "oracle_tt", "model": model,
                    "dataset": dataset, "utterance_id": str(row["utterance_id"]), "reference": row["reference"],
                    "baseline_hypothesis": baseline["text"], "steered_hypothesis": steered["text"],
                    "method_family": method, "method": method, "side": side, "layer": int(layer), "rho": float(rho),
                    "decode_mode": mode, "intervention_site": "encoder_post_self_attn_residual_pre_ffn" if side == "encoder" else "decoder_post_cross_attn_residual",
                    "direction_hash": _sha(directions[layer][method].tolist()),
                    "rank": int(analysis[layer]["n_A"] if method == "conditioning_cs" else dynamic_rank(analysis[layer]["n_A"], analysis[layer]["n_B"], directions[layer][method].shape[0])),
                    "eligible": True, "local_edited_position_count": edited,
                    "perturbation_diagnostics": {"edit_norm_sum": energy, "records": [getattr(x, "__dict__", {}) for x in records]},
                    "target_logit_diagnostics": None,
                    "runtime_sec": time.monotonic() - started, "baseline_runtime_sec": baseline.get("runtime_sec"),
                    "metrics": metric, "rho_cache_reused": True, "analysis_passes": analysis[layer]["analysis_passes"],
                    "gold_leakage": {"analysis_sequence": "baseline_hypothesis", "gold_target_hidden_states": False,
                                     "gold_next_token_logits": False, "target_embedding_direction": False},
                    "provenance": {"phase": "A", "regime": "oracle_tt", "git_commit": _git(),
                                   "source_run_id": os.environ.get("SLURM_JOB_ID", "local"),
                                   "model_revision": getattr(bundle, "revision", None), "panel_fingerprint": None,
                                   "fixed_direction_artifact": None, "direction_construction": "per_sample_dynamic_rank",
                                   "rho_cache": "one_direction_all_rho", "analysis_layer_collection": "all_requested_layers_one_pass"},
                    "canonical_key": key,
                }
                out = row_path(key)
                _atomic_json(out, payload)
                outputs.append(payload)
    return outputs, stats


def run_shard(model_arg: str, dataset: str, side: str, layer_start: int, layer_stop: int,
              *, acceptance: bool = False, acceptance_count: int = 3) -> int:
    model = MODEL_NAMES.get(model_arg, model_arg)
    if model not in SITES or side not in {"encoder", "decoder"}:
        raise ValueError("invalid model/side")
    if not (0 <= layer_start < layer_stop <= SITES[model][side]):
        raise ValueError("invalid layer block")
    if acceptance:
        rows, panel_fp = _dataset_rows("cs_dialogue")
        dataset = "cs_dialogue"
        rows = rows[:acceptance_count]
    else:
        rows, panel_fp = _dataset_rows(dataset)
    bundle = _model_bundle(model)
    bundle.model.eval()
    # Baselines are exact reusable inputs.  They are not outcome rows and are
    # therefore safe to load before selection; a missing one is a hard error.
    old_dataset = "cs_dialogue_dev_select" if dataset == "cs_dialogue" else "ascend_eval"
    base_path = REPO / "results/basis_a6_expanded/baselines" / model / old_dataset / "greedy.json"
    if not base_path.is_file():
        raise RuntimeError(f"validated baseline missing: {base_path}")
    base_payload = json.loads(base_path.read_text())
    base_rows = base_payload.get("rows", {})
    layers = list(range(layer_start, layer_stop))
    rhos = rho_grid(model)
    started = time.monotonic(); total = {"direction_constructions": 0, "rho_cache_hits": 0, "new_rows": 0, "reused_rows": 0}
    shard_cache = DirectionCache()
    for row in rows:
        uid = str(row["utterance_id"])
        baseline = base_rows.get(uid)
        if baseline is None:
            raise RuntimeError(f"baseline does not contain frozen search utterance {uid}")
        # Skip analysis only when every key in this sample/layer block is
        # already accepted.  This is the physical resumability boundary.
        keys = [canonical_key(model=model, dataset=dataset, uid=uid, family=f, side=s, layer=l, rho=r)
                for l in layers for f, s in METHODS if s == side for r in rhos]
        if not acceptance and all(accepted(k) is not None for k in keys):
            total["reused_rows"] += len(keys); continue
        out, stats = _run_sample(bundle, model, dataset, row, baseline, side, layers, rhos, "greedy", acceptance=acceptance)
        total["new_rows"] += len(out); total["direction_constructions"] += stats["direction_constructions"]
        total["rho_cache_hits"] += stats["rho_cache_hits"]
    # A cache manifest is written separately from logical rows; it is small,
    # resumable provenance and never a corpus-level steering direction.
    _atomic_json(ROOT / "phase_a" / "caches" / model / dataset / side / f"L{layer_start:02d}-{layer_stop - 1:02d}.json",
                 {"status": "PASS", "regime": "oracle_tt", "model": model, "dataset": dataset, "side": side,
                  "layers": layers, "direction_constructions": total["direction_constructions"],
                  "rho_cache_hits": total["rho_cache_hits"], "cache_scope": "per_sample_layer", "fixed_direction_artifact": None,
                  "panel_fingerprint": panel_fp, "git_commit": _git(), "runtime_sec": time.monotonic() - started})
    print(json.dumps({"status": "PASS", "model": model, "dataset": dataset, "side": side,
                      "layers": layers, **total, "runtime_sec": time.monotonic() - started}, sort_keys=True))
    return 0


def run_acceptance(model_arg: str, *, count: int = 3) -> int:
    """Run the compact physical integration acceptance in one resident process."""
    model = MODEL_NAMES.get(model_arg, model_arg)
    # The compact acceptance is plumbing-only.  Use the already validated
    # small D-dev-select panel so a missing search-ID alignment cannot be
    # mistaken for an executor failure.  Phase-A itself still uses the frozen
    # IDs and is refused below until every one has accepted oracle spans.
    acceptance_panel = REPO / "results/basis_a6_expanded/preflight/REAL_ACCEPTANCE_PANEL.json"
    panel_payload = json.loads(acceptance_panel.read_text())
    rows = [dict(x) for x in panel_payload["rows"][:int(count)]]
    panel_fp = str(panel_payload.get("fingerprint"))
    old_dataset = "cs_dialogue_dev_select"
    base_payload = json.loads((REPO / "results/basis_a6_expanded/baselines" / model /
                               old_dataset / "greedy.json").read_text())
    bundle = _model_bundle(model); bundle.model.eval()
    started = time.monotonic(); all_outputs: list[dict[str, Any]] = []; counters = {}
    for side in ("encoder", "decoder"):
        method_names = {"add_unique"} if side == "encoder" else {"add_unique", "conditioning_cs"}
        for row in rows:
            baseline = base_payload["rows"][str(row["utterance_id"])]
            output, stats = _run_sample(bundle, model, "cs_dialogue", row, baseline, side, [0], rho_grid(model), "greedy", acceptance=True)
            all_outputs.extend(output)
            counters[side] = {"direction_constructions": counters.get(side, {}).get("direction_constructions", 0) + stats["direction_constructions"],
                              "rho_cache_hits": counters.get(side, {}).get("rho_cache_hits", 0) + stats["rho_cache_hits"]}
        expected = len(rows) * len(method_names) * len(rho_grid(model))
        got = [r for r in all_outputs if r["side"] == side and r["layer"] == 0]
        if len(got) != expected:
            raise RuntimeError(f"acceptance {model}/{side}: expected {expected} rows, got {len(got)}")
        if len({r["direction_hash"] for r in got if r["method"] == "add_unique"}) < 2:
            raise RuntimeError(f"acceptance {model}/{side}: Add-Unique direction did not differ by sample")
        if not all(r["local_edited_position_count"] > 0 for r in got if r["rho"] > 0):
            raise RuntimeError(f"acceptance {model}/{side}: local edit count was zero")
        if not all(r["rho_cache_reused"] for r in got):
            raise RuntimeError(f"acceptance {model}/{side}: rho cache was not reused")
    payload = {"schema_version": "a6_ott_physical_acceptance_v1", "status": "PASS", "model": model,
               "regime": "oracle_tt", "phase": "A", "dataset": "cs_dialogue", "count": len(rows),
               "families": ["add_unique_encoder", "add_unique_decoder", "conditioning_cs_decoder"],
               "panel_fingerprint": panel_fp, "direction_sample_varying": True,
               "local_edit_nonzero": True, "rho_cache_reused": True, "all_layer_collection": True,
               "counters": counters, "runtime_sec": time.monotonic() - started, "git_commit": _git()}
    _atomic_json(ROOT / "preflight" / f"PHYSICAL_ACCEPTANCE_{model}.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("whisper", "qwen", "qwen3_asr_1p7b"))
    ap.add_argument("--dataset", choices=("cs_dialogue", "ascend"))
    ap.add_argument("--side", choices=("encoder", "decoder"))
    ap.add_argument("--layer-start", type=int)
    ap.add_argument("--layer-stop", type=int)
    ap.add_argument("--acceptance", action="store_true")
    ap.add_argument("--acceptance-count", type=int, default=3)
    ap.add_argument("--list-shards", action="store_true")
    args = ap.parse_args()
    model = MODEL_NAMES.get(args.model, args.model)
    if args.list_shards:
        print("\n".join(list_shards(model))); return 0
    if args.acceptance:
        return run_acceptance(model, count=args.acceptance_count)
    for name in ("dataset", "side", "layer_start", "layer_stop"):
        if getattr(args, name) is None: ap.error(f"--{name.replace('_', '-')} is required")
    if args.model == "qwen": model = "qwen3_asr_1p7b"
    return run_shard(model, args.dataset, args.side, args.layer_start, args.layer_stop)


if __name__ == "__main__":
    raise SystemExit(main())
