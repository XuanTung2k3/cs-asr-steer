#!/usr/bin/env python
"""BASIS-A3 frozen scope/depth/cross-corpus runner.

The runner is additive and intentionally does not import legacy steering hooks.
Scientific decoder intervention uses the DG-02 exact-site hook; scientific
encoder intervention uses ``lss.encoder_sites``.  All conditions are written
as independent JSON blocks and can be resumed safely.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import pandas as pd
import torch

from csasr.experiments.basis_a3_protocol import (
    A2, ALL_RHO, COND_LAYERS, DATA_ROOT, DATASETS, DECODER_LAYERS, ENCODER_LAYERS, R1_RHO, R2_RHO,
    RESULTS, SCOPES, canonical_sha, freeze_panels, geometry_rows, write_geometry,
    write_json,
)

GEN = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
           temperature=0.0, max_new_tokens=200, condition_on_prev_tokens=False)
MODEL_CFG = "model/whisper_large_v3.yaml"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _panel(dataset: str) -> dict:
    return _load_json(RESULTS / "panels" / f"{dataset}_300.json")


def _panel_frame(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(_panel(dataset)["rows"])


def _direction_meta(layer: int, key: str) -> tuple[np.ndarray, float, str]:
    meta = _load_json(A2 / "directions.json")["layers"][str(int(layer))]
    path = REPO / meta["files"][key]
    return np.asarray(np.load(path), dtype=np.float32), float(meta["scale_s_l"]), meta["hashes"][key]


def _baseline_path(dataset: str) -> Path:
    return RESULTS / "baselines" / f"{dataset}.json"


def _ensure_baseline(bundle, dataset: str, frame: pd.DataFrame) -> dict[str, str]:
    """Load or decode one baseline per panel, using an inter-process lock."""
    path = _baseline_path(dataset); lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return _load_json(path)["texts"]
        # The CS-Dialogue panel is exactly the canonical DG-04/BASIS-A2
        # D-dev-select panel.  Its frozen free-decoding baseline is already
        # cached in B0; reusing it avoids a second baseline decode and keeps
        # the baseline transcript byte-for-byte identical to the established
        # panel cache.
        if dataset == "cs_dialogue":
            prior = _load_json(REPO / "results/dg04/results/B0.json")["texts"]
            expected = {str(x) for x in frame["utterance_id"]}
            if set(prior) != expected:
                raise RuntimeError("canonical CS baseline IDs do not match BASIS-A3 panel")
            texts = {str(k): str(v) for k, v in prior.items()}
            write_json(path, {"schema_version": "basis_a3_baseline_v1", "dataset": dataset,
                              "count": len(texts), "texts": texts,
                              "panel_fingerprint": _panel(dataset)["fingerprint"],
                              "source": "results/dg04/results/B0.json"})
            return texts
        from csasr.models.whisper import batch_model_inputs
        texts = {}
        for row in frame.sort_values("duration_sec").to_dict(orient="records"):
            inp = batch_model_inputs(bundle, [row["audio_path"]])
            with torch.inference_mode():
                out = bundle.model.generate(**inp, **GEN)
            seq = out if isinstance(out, torch.Tensor) else out.sequences
            texts[str(row["utterance_id"])] = bundle.processor.batch_decode(
                seq, skip_special_tokens=True)[0].strip()
        write_json(path, {"schema_version": "basis_a3_baseline_v1", "dataset": dataset,
                          "count": len(texts), "texts": texts,
                          "panel_fingerprint": _panel(dataset)["fingerprint"]})
        return texts


def _decoder_positions(bundle, reference: str) -> list[int]:
    """Reference-aligned embedded-English query positions (oracle diagnostic)."""
    from csasr.data.alignment import build_prefix, token_char_offsets
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.experiments.v2r3_directions import first_token_for_unit, unit_char_spans
    norm, units = normalize_and_segment(reference)
    tags = [tag_unit(u) for u in units]
    if EN not in tags or ZH not in tags:
        return []
    ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
    offsets = token_char_offsets(bundle.processor.tokenizer, ids)
    spans = unit_char_spans(norm, units)
    prefix = build_prefix(bundle.processor, language="zh", task="transcribe")
    positions = []
    for i, tag in enumerate(tags):
        if tag != EN or i >= len(spans):
            continue
        ti = first_token_for_unit(offsets, spans[i][0])
        if ti is not None:
            positions.append(len(prefix) + int(ti) - 1)
    return sorted(set(positions))


def _decoder_gate(allowed: set[int]):
    def gate_fn(*, abs_pos, r, **_kwargs):
        vals = torch.as_tensor([1.0 if int(x) in allowed else 0.0 for x in abs_pos],
                               dtype=torch.float32, device=r.device)
        # The canonical hook accepts (B,T); generation here is batch=1, but
        # preserve the row dimension explicitly so a prefill sequence is not
        # mistaken for a per-row (B,) gate.
        return vals.view(1, -1).expand(r.shape[0], -1)
    return gate_fn


def _encoder_span(dataset: str, row: dict, bundle) -> tuple[int, int] | None:
    """Return the frozen reference-aligned target span in encoder frames."""
    if dataset.startswith("seame"):
        uid = str(row["utterance_id"])
        seg = DATA_ROOT / "audio/segments" / uid / "segments.json"
        if not seg.is_file():
            return None
        boundaries = _load_json(seg).get("boundaries", [])
        if len(boundaries) < 2:
            return None
        b = boundaries[1]
        return bundle.sec_to_frames(float(b["start_seconds"]), float(b["end_seconds"]),
                                    bundle.valid_frames(row["duration_sec"]))
    # The CS panel's accepted alignment is already stored in candidates_all;
    # this branch is populated in _cs_encoder_spans below.
    return None


def _encoder_gain(bundle, dataset: str, row: dict, *, scope: str, cs_spans: dict) -> torch.Tensor:
    valid = bundle.valid_frames(row["duration_sec"])
    gain = torch.zeros((1, bundle.max_encoder_frames), device=bundle.device)
    if scope == "global":
        gain[0, :valid] = 1.0
        return gain
    span = _encoder_span(dataset, row, bundle)
    if span is None and dataset == "cs_dialogue":
        sec = cs_spans.get(str(row["utterance_id"]))
        span = bundle.sec_to_frames(*sec, valid) if sec is not None else None
    if span is not None:
        gain[0, span[0]:span[1]] = 1.0
    return gain


def _load_cs_construct_for_encoder():
    from experiments.dg03_build_basis import _load_construct
    from csasr.experiments.v2r3_directions import build_spans, matrix_runs
    from csasr.lss.align import conventions as conv
    cfg, manifest, construct = _load_construct()
    root = Path(cfg["experiment"]["output_root"])
    candidates, _ = conv.read_development_candidates(root / "alignments/candidates_all.parquet")
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str), manifest["dialogue_id"].astype(str)))
    spans = build_spans(candidates, dialogue_of=dialogue_of)
    return cfg, manifest, construct, spans, matrix_runs(candidates)


def _cs_eval_spans() -> dict[str, tuple[float, float]]:
    """Accepted reference-aligned CS spans for the frozen 300-ID panel."""
    from csasr.utils.config import load_config
    from csasr.experiments.v2r3_directions import build_spans
    from csasr.lss.align import conventions as conv
    cfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    role = pd.read_parquet(Path(cfg["v2_namespace"]["role_root"]) / "role_D-dev-select.parquet")
    root = Path(cfg["experiment"]["output_root"])
    candidates, _ = conv.read_development_candidates(root / "alignments/candidates_all.parquet")
    dialogue_of = dict(zip(role["conversation_id"].astype(str), role["dialogue_id"].astype(str)))
    spans = build_spans(candidates, dialogue_of=dialogue_of)
    ids = set(_panel("cs_dialogue")["rows"][i]["utterance_id"] for i in range(_panel("cs_dialogue")["count"]))
    spans = spans[spans["utterance_id"].astype(str).isin(ids)]
    return {str(uid): (float(g["start_sec"].min()), float(g["end_sec"].max()))
            for uid, g in spans.groupby("utterance_id") if len(g)}


@torch.inference_mode()
def build_encoder_directions(bundle) -> dict:
    """Build one exact-site Raw direction per encoder layer from D-construct."""
    from csasr.experiments.v2r3_directions import control_frames, pool_weights, pooled, PRIMARY_POOLING, _frame_range
    from csasr.lss.encoder_sites import EncoderPostSelfAttnRecorder
    cfg, manifest, construct, all_spans, matrix = _load_cs_construct_for_encoder()
    construct = construct[construct["all_correct"]].copy()
    by_uid = {str(u): g for u, g in construct.groupby("utterance_id")}
    wanted = manifest[manifest["utterance_id"].astype(str).isin(by_uid)].sort_values("duration_sec")
    contrasts = {l: [] for l in ENCODER_LAYERS}; norms = {l: [] for l in ENCODER_LAYERS}
    en_counts = {l: 0 for l in ENCODER_LAYERS}; zh_counts = {l: 0 for l in ENCODER_LAYERS}
    from csasr.models.whisper import batch_model_inputs
    for _, row in wanted.iterrows():
        uid = str(row["utterance_id"]); features = batch_model_inputs(bundle, [row["audio_path"]])["input_features"]
        with EncoderPostSelfAttnRecorder(bundle, ENCODER_LAYERS) as rec:
            bundle.model.model.encoder(features)
        valid = bundle.valid_frames(row["duration_sec"])
        groups = by_uid[uid]
        for l in ENCODER_LAYERS:
            block = rec.states[l][0].float().cpu().numpy()
            norms[l].append(float(np.linalg.norm(block[:valid], axis=1).mean()))
            for _, span in groups.iterrows():
                lo, hi = _frame_range(span["start_sec"], span["end_sec"], bundle.encoder_step_sec, valid)
                if hi - lo < 3: continue
                control = control_frames(lo, hi,
                    matrix[matrix["utterance_id"].astype(str) == uid] if len(matrix) else pd.DataFrame(),
                    bundle.encoder_step_sec, valid)
                if control is None: continue
                clo, chi = control
                en = block[lo:hi]; zh = block[clo:chi]
                contrasts[l].append(np.asarray(pool_weights(len(en), PRIMARY_POOLING) @ en -
                                               pool_weights(len(zh), PRIMARY_POOLING) @ zh, dtype=np.float64))
                en_counts[l] += len(en); zh_counts[l] += len(zh)
        del features
    out = {"schema_version": "basis_a3_raw_encoder_direction_manifest_v1", "source_role": "D-construct",
           "site": "encoder_post_self_attn_residual_pre_ffn", "layers": {},
           "alignment": {"family": "existing_ctc", "convention": "blank_to_preceding",
                         "boundary_tolerance_ms": 80, "padding_excluded": True}}
    for l in ENCODER_LAYERS:
        if not contrasts[l]: raise RuntimeError(f"no encoder contrasts for layer {l}")
        v = np.mean(np.stack(contrasts[l]), axis=0); raw_norm = float(np.linalg.norm(v))
        if not np.isfinite(raw_norm) or raw_norm == 0: raise RuntimeError(f"degenerate encoder Raw L{l}")
        unit = (v / raw_norm).astype(np.float32)
        path = RESULTS / "directions/raw_encoder" / f"raw_encoder_L{l}.npy"; path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, unit)
        from csasr.experiments.basis_a3_protocol import sha256_file
        out["layers"][str(l)] = {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path),
            "raw_norm": raw_norm, "direction_norm": float(np.linalg.norm(unit)),
            "scale": float(np.mean(norms[l])), "scale_definition": "mean exact-site norm over valid D-construct frames",
            "n_spans": len(contrasts[l]), "n_english_frames": en_counts[l], "n_mandarin_frames": zh_counts[l]}
    write_json(RESULTS / "directions/raw_encoder/manifest.json", out)
    return out


def _ensure_encoder_directions(bundle) -> None:
    """Build the shared encoder artifacts once when two workers start together."""
    lock_path = RESULTS / "directions/raw_encoder/.build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (RESULTS / "directions/raw_encoder/manifest.json").is_file():
            build_encoder_directions(bundle)


def _decode_condition(bundle, dataset: str, side: str, layer: int, direction_key: str,
                      scope: str, rho: float, *, cs_spans: dict | None = None) -> dict:
    from csasr.models.whisper import batch_model_inputs
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
    frame = _panel_frame(dataset); baseline = _ensure_baseline(bundle, dataset, frame)
    cs_spans = cs_spans or {}
    if side == "decoder":
        direction, scale, dhash = _direction_meta(layer, direction_key)
    else:
        meta = _load_json(RESULTS / "directions/raw_encoder/manifest.json")["layers"][str(layer)]
        direction = np.load(REPO / meta["path"]); scale = float(meta["scale"]); dhash = meta["sha256"]
    texts = {}; edit_norms = []; edited = 0; start = time.monotonic()
    for row in frame.sort_values("duration_sec").to_dict(orient="records"):
        uid = str(row["utterance_id"]); inp = batch_model_inputs(bundle, [row["audio_path"]])
        if side == "decoder":
            allowed = set(_decoder_positions(bundle, row["reference"])) if scope == "oracle_local" else None
            gate = _decoder_gate(allowed) if allowed is not None else None
            hook = DecoderPostCrossAttnInterventionHook(
                bundle, layer, torch.from_numpy(direction), alpha=rho, scale=scale,
                num_forced_prefix=num_forced_prefix_from(bundle.processor), gate_fn=gate,
                norm_preserve=True, mode="steer", record=True, record_last_only=False)
        else:
            gain = _encoder_gain(bundle, dataset, row, scope=scope, cs_spans=cs_spans)
            hook = EncoderPostSelfAttnInterventionHook(bundle, layer, torch.from_numpy(direction),
                alpha=rho, scale=scale, gain=gain, norm_preserve=True, record=True)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inp, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        texts[uid] = bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if side == "decoder":
            active = [r for r in hook.records if r.steered]
            edited += len(active); edit_norms.extend(float(r.edit_norm) for r in active)
        else:
            for rec in hook.records:
                edited += rec.active_frames; edit_norms.append(float(rec.edit_norm))
    refs = [str(x) for x in frame["reference"]]; ids = [str(x) for x in frame["utterance_id"]]
    base = [baseline[x] for x in ids]; method = [texts[x] for x in ids]
    from csasr.evaluation import canonical, retention
    bm, mm = canonical.corpus_metrics(refs, base), canonical.corpus_metrics(refs, method)
    transitions = canonical.correction_corruption(refs, base, method)
    metrics = dict(mm); metrics.update(canonical.error_metric_gains(bm, mm))
    metrics["transitions"] = transitions
    metrics["poi_corrections"] = transitions["corrections"]
    metrics["poi_corruptions"] = transitions["corruptions"]
    metrics["poi_net_utility"] = transitions["net_corrections"]
    metrics["candidate_level_utility"] = None
    metrics["retention"] = retention.retention_report(refs, base, method)
    metrics["outside_harm"] = None
    metrics["edited_positions_or_frames"] = int(edited)
    metrics["total_intervention_energy"] = float(np.sum(edit_norms) if edit_norms else 0.0)
    metrics["mean_intervention_norm"] = float(np.mean(edit_norms) if edit_norms else 0.0)
    metrics["runtime_seconds"] = time.monotonic() - start
    metrics["scope"] = scope; metrics["side"] = side
    return {"schema_version": "basis_a3_condition_v1", "dataset": dataset, "data_role": _panel(dataset)["data_role"],
            "direction": direction_key.title() if direction_key != "conditioning" else "Conditioning",
            "layer": int(layer), "scope": scope, "rho": float(rho), "direction_hash": dhash,
            "scale": scale, "panel_fingerprint": _panel(dataset)["fingerprint"], "baseline_reused": True,
            "metrics": metrics, "texts": texts, "baseline_texts": baseline}


def _preflight_child(spec: tuple[str, str, int]) -> str:
    """One fixed plumbing decode in an independent model replica."""
    dataset, side, layer = spec
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    from csasr.lss.encoder_sites import EncoderPostSelfAttnInterventionHook
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
    bundle = load_whisper(load_config(MODEL_CFG)); bundle.model.eval()
    row = _panel_frame(dataset).sort_values("duration_sec").iloc[0].to_dict()
    inp = batch_model_inputs(bundle, [row["audio_path"]])
    direction = torch.zeros(bundle.d_model, dtype=torch.float32); direction[0] = 1.0
    if side == "decoder":
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, layer, direction, alpha=.5, scale=1.0,
            num_forced_prefix=num_forced_prefix_from(bundle.processor), norm_preserve=True)
    else:
        valid = bundle.valid_frames(row["duration_sec"])
        gain = torch.zeros((1, bundle.max_encoder_frames), device=bundle.device)
        gain[0, :valid] = 1.0
        hook = EncoderPostSelfAttnInterventionHook(
            bundle, layer, direction, alpha=.5, scale=1.0, gain=gain, norm_preserve=True)
    with hook, torch.inference_mode():
        out = bundle.model.generate(**inp, **GEN)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    return bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()


def preflight(args) -> int:
    """Required short 1-worker vs 2-worker equivalence check for encoder path."""
    ctx = mp.get_context("spawn")
    observations = []
    for side in ("encoder", "decoder"):
        spec = ("cs_dialogue", side, 24)
        t0 = time.monotonic()
        with ctx.Pool(processes=1) as pool:
            one = pool.map(_preflight_child, [spec])[0]
        with ctx.Pool(processes=2) as pool:
            two = pool.map(_preflight_child, [spec, spec])
        observations.append({"side": side, "layer": 24, "rho": .5,
                             "one_worker_seconds": time.monotonic() - t0,
                             "two_worker_transcripts_identical": len(set(two)) == 1,
                             "one_vs_two_identical": one == two[0],
                             "transcript_sha256": canonical_sha(one)})
    passed = all(x["two_worker_transcripts_identical"] and x["one_vs_two_identical"] for x in observations)
    write_json(RESULTS / "runtime/encoder_worker_preflight.json",
               {"schema_version": "basis_a3_worker_preflight_v1", "status": "PASS" if passed else "FAIL",
                "observations": observations, "selected_workers": 2 if passed else 1,
                "generation_batch_size": 1, "note": "fixed plumbing conditions only"})
    if not passed:
        raise RuntimeError("BASIS-A3 worker equivalence preflight failed")
    return 0


def run_grid(args) -> int:
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.utils.provenance import code_config_snapshot_hash, test_snapshot_hash
    from csasr.utils.logging import git_state
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    worker_index, num_workers = int(args.worker_index), int(args.num_workers)
    if not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    worker_tag = f"_worker{worker_index}" if num_workers > 1 else ""
    job_manifest_path = RESULTS / "manifests" / f"job_{job_id}{worker_tag}.json"
    job_manifest = {"schema_version": "basis_a3_job_manifest_v1", "status": "RUNNING",
        "slurm_job_id": job_id, "hostname": os.uname().nodename,
        "argv": list(sys.argv), "dataset": args.dataset, "side": args.side,
        "direction": args.direction, "stage": args.stage,
        "worker_index": worker_index, "num_workers": num_workers,
        "panel_fingerprint": _panel(args.dataset)["fingerprint"],
        "code_config_sha256": code_config_snapshot_hash(), "tests_sha256": test_snapshot_hash(),
        "git": git_state(str(REPO)), "started_at": time.time()}
    write_json(job_manifest_path, job_manifest)
    bundle = load_whisper(load_config(MODEL_CFG)); bundle.model.eval()
    if args.side == "encoder":
        _ensure_encoder_directions(bundle)
    cs_spans = _cs_eval_spans() if args.side == "encoder" and args.dataset == "cs_dialogue" else {}
    stages = ("r2", "conditioning") if args.stage == "combined" else (args.stage,)
    result_roots = []
    for stage in stages:
        direction_name = "conditioning" if stage == "conditioning" else args.direction
        layers = ENCODER_LAYERS if args.side == "encoder" else DECODER_LAYERS
        if direction_name == "conditioning":
            if args.side != "decoder":
                raise ValueError("Conditioning is decoder-only")
            layers = COND_LAYERS
        if stage == "r2":
            selection = _load_json(RESULTS / "selection/r2_layers.json")["selected"]
            layers = tuple(selection[args.side]["layers"])
        rhos = R1_RHO if stage == "r1" else (ALL_RHO if stage == "conditioning" else R2_RHO)
        out_root = RESULTS / ("raw_r1" if stage == "r1" else
                              ("conditioning" if stage == "conditioning" else "raw_r2")) / args.dataset
        result_roots.append(str(out_root.relative_to(REPO)))
        conditions = [(layer, scope, rho) for layer in layers
                      for scope in SCOPES for rho in rhos]
        for condition_index, (layer, scope, rho) in enumerate(conditions):
            if condition_index % num_workers != worker_index:
                continue
            key = f"{direction_name}_{args.side}_L{layer}_{scope}_rho{rho:g}.json"
            path = out_root / f"L{layer:02d}" / key
            if path.is_file(): continue
            dkey = "conditioning" if direction_name == "conditioning" else "raw"
            result = _decode_condition(bundle, args.dataset, args.side, layer, dkey, scope, rho, cs_spans=cs_spans)
            result["run_manifest"] = str(job_manifest_path.relative_to(REPO))
            write_json(path, result)
            print(f"completed {path.relative_to(REPO)}", flush=True)
    job_manifest.update({"status": "COMPLETED", "finished_at": time.time(),
                         "result_roots": result_roots})
    write_json(job_manifest_path, job_manifest)
    return 0


def freeze(args) -> int:
    panels = freeze_panels(RESULTS / "panels")
    write_geometry(RESULTS / "geometry")
    from csasr.utils.config import load_config
    from csasr.utils.provenance import stage_provenance, code_config_snapshot_hash, test_snapshot_hash
    cfg = load_config(MODEL_CFG)
    payload = {"schema_version": "lss_spec_freeze_v1", "status": "PLANNED_PRE_RUN",
        "study": "BASIS-A3", "spec": "docs/current/BASIS_A3_RAW_COND_SCOPE_DEPTH_SPEC.md",
        "provenance": stage_provenance({**cfg, "experiment": {"output_root": str(RESULTS), "seed": 42}}, "basis_a3"),
        "code_config_sha256": code_config_snapshot_hash(), "tests_sha256": test_snapshot_hash(),
        "model": cfg["model"], "datasets": {k: {"count": v["count"], "fingerprint": v["fingerprint"]} for k,v in panels.items()},
        "directions": {"families": ["Raw", "Conditioning"], "decoder_source": str(A2 / "directions.json"),
                       "encoder_source": "built from D-construct in first encoder job"},
        "sites": {"decoder": "decoder_post_cross_attn_residual", "encoder": "encoder_post_self_attn_residual_pre_ffn"},
        "grid": {"r1_raw": {"layers": {"encoder": list(ENCODER_LAYERS), "decoder": list(DECODER_LAYERS)}, "rho": [.5], "scopes": list(SCOPES)},
                 "conditioning": {"layers": list(COND_LAYERS), "rho": list(ALL_RHO), "scopes": list(SCOPES)},
                 "r2_raw_additional_rho": list(R2_RHO)},
        "forbidden_splits": list(("D-dev-confirm", "D-test")),
        "execution": {"partition": "mig", "gpu": "nvidia_h100_80gb_hbm3_3g.40gb:1", "workers": 2, "generation_batch_size": 1,
                      "max_concurrent_jobs": 2}, "sha256": ""}
    from csasr.lss.specfreeze import SpecFreeze, seal
    freeze_path = RESULTS / "spec/basis_a3_spec_freeze_v1.json"
    payload["sha256"] = ""
    seal(SpecFreeze(payload), freeze_path)
    write_json(RESULTS / "manifests/pre_run_manifest.json", payload | {"freeze_path": str(freeze_path.relative_to(REPO))})
    print(json.dumps({"status": "PLANNED_PRE_RUN", "freeze": str(freeze_path),
                      "panels": {k: (v["count"], v["fingerprint"]) for k,v in panels.items()}}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("freeze")
    sub.add_parser("preflight")
    g = sub.add_parser("grid")
    g.add_argument("--dataset", choices=DATASETS, required=True)
    g.add_argument("--side", choices=("encoder", "decoder"), required=True)
    g.add_argument("--direction", choices=("Raw", "conditioning"), default="Raw")
    g.add_argument("--stage", choices=("r1", "r2", "conditioning", "combined"), default="r1")
    g.add_argument("--worker-index", type=int, default=0)
    g.add_argument("--num-workers", type=int, default=1)
    x = sub.add_parser("geometry")
    sub.add_parser("select-r2")
    args = ap.parse_args(argv)
    if args.mode == "freeze": return freeze(args)
    if args.mode == "preflight": return preflight(args)
    if args.mode == "geometry":
        write_geometry(RESULTS / "geometry"); return 0
    if args.mode == "select-r2":
        from csasr.experiments.basis_a3_protocol import select_r2_layers
        select_r2_layers(); return 0
    try:
        return run_grid(args)
    except Exception as exc:
        # Keep an interrupted/failed scientific shard auditable rather than
        # leaving a manifest falsely marked RUNNING.
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        worker_index = int(getattr(args, "worker_index", 0))
        num_workers = int(getattr(args, "num_workers", 1))
        worker_tag = f"_worker{worker_index}" if num_workers > 1 else ""
        path = RESULTS / "manifests" / f"job_{job_id}{worker_tag}.json"
        if path.is_file():
            failed = _load_json(path)
            failed.update({"status": "FAILED", "finished_at": time.time(),
                           "error_type": type(exc).__name__, "error": str(exc)})
            write_json(path, failed)
        raise


if __name__ == "__main__": raise SystemExit(main())
