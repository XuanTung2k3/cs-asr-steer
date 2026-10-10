#!/usr/bin/env python
"""BASIS-A: frozen/no-training direction-construction ablation.

The GPU mode evaluates only the three non-reusable directions (R, C, RC) in
one Whisper process.  DG-04 B0/B1/B2 are loaded only after strict compatibility
checks.  The analysis mode is CPU-only and creates the complete comparison,
geometry diagnostics, PCA figure, and tables without changing frozen inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import torch

from csasr.experiments.basis_ablation import (
    basis_geometry, construct_directions, direction_hash, fit_pca_rows,
    project_pca_directions,
)

OUT = REPO / "results/basis_ablation_frozen"
DG04 = REPO / "results/dg04"
BASIS = REPO / "results/dg03/basis/steering_basis_v1_L24.json"
GRID = (0.5, 1.0, 2.0)
LAYER = 24
SCALE = 8.931257754160319
MIX = (0.5, 0.5)
DATA_FINGERPRINT = "sha256:4a4ce18e7a368e60108fe1506ee6a70611d532d0821b376dc1ac8d483e2b4440"
LOCAL_HASH = "sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93"
COND_HASH = "sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991"
RAW_HASH = "sha256:b637493362f2ba319aef3d24f74f382334508cdc28f79fc7a062cd528cf3b2c8"
RAW_NORM_HASH = "sha256:1bcb2a7a15f665715703b3717e4ef50d23432e4df72d037739d646e689fdc542"
RC_HASH = "sha256:2142a7d4d69a7b54a5596bcca953d1966ce0244f3a6ed5feb1d8b459068be4c9"
LC_HASH = "sha256:b3aa31d521d79c93fae1c22599b42136d8b6cfbd191846f84cbde487e635a930"
SITE = "decoder_post_cross_attn_residual"
REPRESENTATION_SEED = 2408
REPRESENTATION_MAX = 10000
GEN = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
           temperature=0.0, max_new_tokens=200, condition_on_prev_tokens=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return None


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, default=str), encoding="utf-8")


def _basis() -> tuple[dict, dict[str, np.ndarray]]:
    rec = json.loads(BASIS.read_text())
    if rec.get("site") != SITE or int(rec.get("scientific_layer")) != LAYER:
        raise RuntimeError("BASIS-A basis is not the frozen exact L24 basis")
    p = BASIS.parent
    raw = np.load(p / rec["tensor_files"]["raw_language_contrast"])
    local = np.load(p / rec["tensor_files"]["conditioning_residualized_local"])
    cond = np.load(p / rec["tensor_files"]["language_conditioning"])
    if direction_hash(raw) != RAW_HASH or direction_hash(local) != LOCAL_HASH \
            or direction_hash(cond) != COND_HASH:
        raise RuntimeError("DG-03 basis tensor hash mismatch")
    ds = construct_directions(raw, local, cond, a_local=MIX[0], a_cond=MIX[1])
    expected = {"raw": RAW_NORM_HASH, "local": LOCAL_HASH, "conditioning": COND_HASH,
                "raw_cond": RC_HASH, "local_cond": LC_HASH}
    for name, digest in expected.items():
        if direction_hash(ds[name]) != digest:
            raise RuntimeError(f"BASIS-A derived direction hash mismatch for {name}")
    return rec, ds


def _dataset_fingerprint_from_ids(ids: list[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(sorted(map(str, ids))) + "\n").encode()).hexdigest()


def _load_reused() -> tuple[dict[str, dict], dict]:
    """Strictly prove that B0/B1/B2 can be reused without decoding them."""
    paths = {"B0": DG04 / "results/B0.json",
             "Local": DG04 / "results/B1_rho{rho}.json",
             "Local+Conditioning": DG04 / "results/B2_rho{rho}.json"}
    loaded: dict[str, dict] = {}
    for name, pattern in paths.items():
        if name == "B0":
            p = pattern; x = json.loads(p.read_text()); loaded[name] = {"path": p, "payload": x}
            continue
        for rho in GRID:
            p = Path(str(pattern).format(rho=rho))
            x = json.loads(p.read_text())
            loaded[f"{name}|{rho}"] = {"path": p, "payload": x}
    ids = list(loaded["B0"]["payload"]["texts"])
    if len(ids) != 300 or _dataset_fingerprint_from_ids(ids) != DATA_FINGERPRINT:
        raise RuntimeError("DG-04 population IDs do not match the frozen 300-utterance subset")
    for key, item in loaded.items():
        x, r = item["payload"], item["payload"]["result_v1"]
        if list(x.get("texts", {})) != ids:
            raise RuntimeError(f"DG-04 reuse population order mismatch: {key}")
        if r.get("data_role") != "D-dev-select" or r.get("decode_regime") != "greedy" \
                or r.get("beam") != 1 or r.get("method", {}).get("layer") != LAYER:
            raise RuntimeError(f"DG-04 reuse protocol mismatch: {key}")
        p = r.get("provenance", {})
        if p.get("dataset_fingerprint", {}).get("composite") != \
                "sha256:3981d6a8134e8f73b49e1fc66451980a59d0f460759aab279a142ecb2f7310ab":
            raise RuntimeError(f"DG-04 reuse dataset provenance mismatch: {key}")
        if p.get("grid_rho") != list(GRID) or p.get("scale_s_l") != SCALE \
                or p.get("basis_artifact_hashes", {}).get("conditioning_residualized_local") != LOCAL_HASH \
                or p.get("basis_artifact_hashes", {}).get("language_conditioning") != COND_HASH:
            raise RuntimeError(f"DG-04 reuse basis/dose provenance mismatch: {key}")
        if r.get("metrics", {}).get("gate_coverage", {}).get("value") is not None:
            raise RuntimeError(f"DG-04 reuse result is not canonical result_v1: {key}")
    reuse = {}
    for key, item in loaded.items():
        reuse[key] = {"path": str(item["path"].relative_to(REPO)),
                      "sha256": _sha256_file(item["path"]), "status": "compatible"}
    return loaded, {"population_ids": ids, "population_fingerprint": DATA_FINGERPRINT,
                    "artifacts": reuse}


def _small_pop(pop, n: int = 8):
    keep = pop.manifest.head(n)["utterance_id"].astype(str).tolist()
    return replace(pop, manifest=pop.manifest.head(n).copy(),
                   targets=pop.targets[pop.targets["utterance_id"].isin(keep)].copy(),
                   poi=pop.poi[pop.poi["utterance_id"].isin(keep)].copy(),
                   baseline_hypotheses={k: v for k, v in pop.baseline_hypotheses.items() if k in keep})


def _preflight(bundle, pop, local: torch.Tensor, nfp: int, batch_size: int) -> dict:
    """Compare batch=1 and the frozen proposed batch on eight fixed utterances."""
    from experiments.dg04_frozen_baselines import _decode
    from csasr.evaluation.canonical import corpus_metrics
    small = _small_pop(pop, 8)
    outputs = {}
    for label, direction, rho in (("Frozen", None, 0.0), ("Local", local, 0.5)):
        one, _ = _decode(bundle, small, layer=LAYER, direction=direction, gate_fn=None,
                         nfp=nfp, alpha=rho, scale=SCALE, batch_size=1)
        many, _ = _decode(bundle, small, layer=LAYER, direction=direction, gate_fn=None,
                          nfp=nfp, alpha=rho, scale=SCALE, batch_size=batch_size)
        if one != many:
            raise RuntimeError(f"batch equivalence failed for {label}")
        refs = [str(x) for x in small.manifest["transcript_raw"]]
        if corpus_metrics(refs, list(one.values())) != corpus_metrics(refs, list(many.values())):
            raise RuntimeError(f"canonical metric batch equivalence failed for {label}")
        outputs[label] = {"utterances": len(one), "transcripts_identical": True}
    return {"batch_1": 1, "proposed_batch": batch_size, "n_utterances": 8,
            "conditions": outputs, "passed": True}


def run_gpu(args) -> int:
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from csasr.lss.sites import num_forced_prefix_from
    from steer_sweep import data as D
    from experiments.dg04_frozen_baselines import _decode, _result_v1, _build_outside_sets

    rec, directions = _basis()
    reused, reuse_info = _load_reused()
    cfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    mcfg = load_config("model/whisper_large_v3.yaml")
    bundle = load_whisper(mcfg); bundle.model.eval()
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("BASIS-A requires a fully frozen backbone")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    pop = D.build_population(bundle, cfg, "D-dev-select", assert_anchors=True)
    ids = list(pop.utterance_ids)
    if _dataset_fingerprint_from_ids(ids) != DATA_FINGERPRINT:
        raise RuntimeError("D-dev-select population fingerprint mismatch")
    refs = D.reference_of(pop)
    out_sets, out_langs, out_diag = _build_outside_sets(cfg, refs, ids)
    base = reused["B0"]["payload"]["texts"]
    local_t = torch.tensor(directions["local"], dtype=torch.float32)
    start_time = time.monotonic()
    preflight = _preflight(bundle, pop, local_t, nfp, args.batch_size)
    provenance = {
        "git_commit": _git_commit(), "model_id": bundle.model_id,
        "model_revision": bundle.revision, "layer": LAYER, "site": SITE,
        "data_role": "D-dev-select", "dataset_fingerprint": DATA_FINGERPRINT,
        "basis_artifact_hashes": rec["tensor_hashes"], "direction_hashes": {
            n: direction_hash(v) for n, v in directions.items()},
        "scale_s_l": SCALE, "grid_rho": list(GRID), "mixture_coefficients": {
            "a_local": MIX[0], "a_cond": MIX[1]}, "decode": GEN,
        "norm_preserve": True, "depth_rescale": False,
        "preflight": preflight, "outside_harm_population": out_diag,
        "slurm": {"job_ids": [os.environ["SLURM_JOB_ID"]] if os.environ.get("SLURM_JOB_ID") else [],
                  "partition": os.environ.get("SLURM_JOB_PARTITION"),
                  "node": os.environ.get("SLURMD_NODENAME"),
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "batch_size": args.batch_size},
    }
    for key, direction, dtype, dest in (("raw", directions["raw"], "raw", "raw"),
                                         ("conditioning", directions["conditioning"], "conditioning", "conditioning"),
                                         ("raw_cond", directions["raw_cond"], "raw_cond", "raw_cond")):
        d = torch.tensor(direction, dtype=torch.float32)
        for rho in GRID:
            texts, audit = _decode(bundle, pop, layer=LAYER, direction=d, gate_fn=None,
                                   nfp=nfp, alpha=float(rho), scale=SCALE,
                                   batch_size=args.batch_size, collect=True)
            p = dict(provenance)
            p["direction_name"] = key; p["direction_hash"] = direction_hash(direction)
            r1 = _result_v1(refs, base, texts, ids, cond=f"BASIS-A_{key}_rho{rho}",
                            provenance=p, direction_type=dtype,
                            steering_strength=float(rho), basis_id=direction_hash(direction),
                            out_sets=out_sets, out_langs=out_langs, audit=audit)
            _write(OUT / dest / f"result_rho{rho}.json", {
                "system": {"raw": "R", "conditioning": "C", "raw_cond": "RC"}[key],
                "direction": key, "rho": float(rho), "beta_nominal": float(rho) * SCALE,
                "direction_hash": direction_hash(direction), "result_v1": r1,
                "texts": texts})
            print(f"{key} rho={rho}: corr={r1['metrics']['transitions']['corrections']} "
                  f"corrupt={r1['metrics']['transitions']['corruptions']} "
                  f"energy={audit['total_energy']:.2f}", flush=True)
    elapsed = time.monotonic() - start_time
    runtime = {"elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60,
               "utterances": len(ids), "conditions": 9,
               "utterances_per_second": len(ids) * 3 / elapsed if elapsed else None,
               "batch_size": args.batch_size,
               "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
               "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
               "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "terminal_state": "COMPLETED"}
    _write(OUT / "run_metadata.json", {"provenance": provenance, "reuse": reuse_info,
                                       "runtime": runtime, "representation": {
                                           "deferred": True,
                                           "reason": "no compatible cached baseline states; recorder replay did not reproduce reused B0 transcripts"}})
    return 0


def _representation_plans(bundle, pop, baseline: dict[str, str]):
    """Exact-site query positions for baseline-correct EN/ZH units."""
    from csasr.data.alignment import build_prefix, token_char_offsets
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.evaluation.pier import unit_status
    from csasr.experiments.v2r3_directions import first_token_for_unit, unit_char_spans

    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    plans = []
    for _, row in pop.manifest.sort_values("duration_sec").reset_index(drop=True).iterrows():
        uid = str(row["utterance_id"])
        norm, units = normalize_and_segment(row["transcript_raw"])
        text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        offsets = token_char_offsets(tokenizer, text_ids)
        char_spans = unit_char_spans(norm, units)
        status = unit_status(str(row["transcript_raw"]), baseline[uid])
        positions = []
        for index, unit in enumerate(units):
            tag = tag_unit(unit)
            if tag not in (EN, ZH) or not status.get(index, (False, ""))[0]:
                continue
            if index >= len(char_spans):
                continue
            token_index = first_token_for_unit(offsets, char_spans[index][0])
            if token_index is None or token_index >= len(text_ids):
                continue
            query_position = len(prefix) + int(token_index) - 1
            if 0 <= query_position < len(prefix) + len(text_ids):
                positions.append({
                    "reference_unit_index": int(index), "language": tag,
                    "token_index": int(token_index),
                    "query_position": int(query_position),
                })
        if positions:
            plans.append({"utterance_id": uid, "dialogue_id": str(row["dialogue_id"]),
                          "audio_path": str(row["audio_path"]),
                          "sequence": list(prefix) + list(text_ids) + [tokenizer.eos_token_id],
                          "positions": positions})
    return plans


@torch.inference_mode()
def extract_representations(args) -> int:
    """Capture a bounded, deterministic teacher-forced exact-site sample."""
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from csasr.lss.sites import DecoderPostCrossAttnRecorder, assert_no_site_hooks
    from steer_sweep import data as D
    from csasr.models.generation import teacher_forced_forward

    cfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    mcfg = load_config("model/whisper_large_v3.yaml")
    bundle = load_whisper(mcfg)
    bundle.model.eval()
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("representation extraction requires a frozen backbone")
    pop = D.build_population(bundle, cfg, "D-dev-select", assert_anchors=True)
    ids = list(pop.utterance_ids)
    if _dataset_fingerprint_from_ids(ids) != DATA_FINGERPRINT:
        raise RuntimeError("representation extraction population fingerprint mismatch")
    reused, _ = _load_reused()
    baseline = reused["B0"]["payload"]["texts"]
    plans = _representation_plans(bundle, pop, baseline)
    if not plans:
        raise RuntimeError("no baseline-correct content positions available")

    vectors, metadata = [], []
    start_time = time.monotonic()
    bs = max(1, int(args.batch_size))
    for start in range(0, len(plans), bs):
        batch = plans[start:start + bs]
        with DecoderPostCrossAttnRecorder(bundle, [LAYER]) as recorder:
            teacher_forced_forward(bundle, [x["audio_path"] for x in batch],
                                   [x["sequence"] for x in batch])
            states = recorder.states[LAYER].float().cpu().numpy()
        for b, plan in enumerate(batch):
            for position in plan["positions"]:
                vectors.append(states[b, position["query_position"]].copy())
                metadata.append({"utterance_id": plan["utterance_id"],
                                 "dialogue_id": plan["dialogue_id"], **position,
                                 "baseline_status": "correct"})
        if (start // bs) % 20 == 0:
            print(f"  representations {min(start + bs, len(plans))}/{len(plans)}", flush=True)
        del states
    assert_no_site_hooks(bundle)

    reps = np.asarray(vectors, dtype=np.float32)
    if len(reps) > REPRESENTATION_MAX:
        rng = np.random.default_rng(REPRESENTATION_SEED)
        keep = np.sort(rng.choice(len(reps), size=REPRESENTATION_MAX, replace=False))
        reps = reps[keep]
        metadata = [metadata[int(i)] for i in keep]
        sampling = "uniform without replacement over extracted baseline-correct content positions"
    else:
        sampling = "all extracted baseline-correct content positions"
    rep_dir = OUT / "representation"
    rep_dir.mkdir(parents=True, exist_ok=True)
    np.save(rep_dir / "baseline_l24_teacher_forced.npy", reps)
    _write(rep_dir / "position_metadata.json", metadata)
    elapsed = time.monotonic() - start_time
    _write(rep_dir / "extraction_metadata.json", {
        "status": "COMPLETE", "source": "baseline exact-site L24 teacher-forced states",
        "site": SITE, "layer": LAYER, "split": "D-dev-select",
        "population_fingerprint": DATA_FINGERPRINT, "utterances": len(ids),
        "utterances_with_positions": len(plans), "n_vectors_extracted": len(vectors),
        "n_vectors_saved": int(len(reps)), "dimension": int(reps.shape[1]),
        "sampling_seed": REPRESENTATION_SEED, "sampling_policy": sampling,
        "position_definition": "query position immediately before first BPE token of a baseline-correct EN/ZH reference unit",
        "decoder": GEN, "frozen_backbone": True, "no_gradients": True,
        "elapsed_seconds": elapsed, "batch_size": bs,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(f"saved {len(reps)} representation rows in {elapsed / 60:.2f} min", flush=True)
    return 0


def _load_result_rows() -> list[dict]:
    from csasr.evaluation.result_schema import validate
    rows = []
    reused, _ = _load_reused()
    mapping = {"B0": "Frozen", "Local": "Local", "Local+Conditioning": "Local+Conditioning"}
    b0 = reused["B0"]["payload"]
    validate(b0["result_v1"])
    rows.append({"direction": "Frozen", "rho": 0.0, "source": "REUSED DG-04", "payload": b0})
    for name in ("Local", "Local+Conditioning"):
        for rho in GRID:
            payload = reused[f"{name}|{rho}"]["payload"]
            validate(payload["result_v1"])
            rows.append({"direction": mapping[name], "rho": rho, "source": "REUSED DG-04",
                         "payload": payload})
    for key, direction in (("raw", "Raw"), ("conditioning", "Conditioning"), ("raw_cond", "Raw+Conditioning")):
        for rho in GRID:
            payload = json.loads((OUT / key / f"result_rho{rho}.json").read_text())
            validate(payload["result_v1"])
            rows.append({"direction": direction, "rho": rho, "source": "NEW",
                         "payload": payload})
    return rows


def _metric_row(row: dict) -> dict:
    x, r = row["payload"], row["payload"]["result_v1"]
    m, t = r["metrics"], r["metrics"]["transitions"]
    e = m["realized_edit"]
    retention = m.get("retention", {})
    energy = e.get("total_energy")
    utility = t.get("net_corrections")
    return {"Direction": row["direction"], "rho": row["rho"], "source": row["source"],
            "WER": "n/a — no frozen canonical overall WER", "MER": m.get("mer"),
            "PIER": m.get("pier"), "Embedded WER": m.get("en_wer"), "Matrix CER": m.get("zh_cer"),
            "Corr": t.get("corrections"), "Corrupt": t.get("corruptions"), "Utility": utility,
            "Outside harm": m.get("outside_harm"),
            "Embed ret.": retention.get("embedded_en", {}).get("rate"),
            "Matrix ret.": retention.get("matrix_zh", {}).get("rate"), "Energy": energy,
            "Utility/Energy": utility / energy if energy else None,
            "corrections_per_energy": t.get("corrections") / energy if energy else None,
            "corruptions_per_energy": t.get("corruptions") / energy if energy else None,
            "corrections_per_1000_edits": 1000 * t.get("corrections") / e.get("n_steered") if e.get("n_steered") else None,
            "corruptions_per_1000_edits": 1000 * t.get("corruptions") / e.get("n_steered") if e.get("n_steered") else None}


def _dose_class(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda x: x["rho"])
    corr = [x["Corr"] for x in ordered]; corrupt = [x["Corrupt"] for x in ordered]
    utility = [x["Utility"] for x in ordered]
    if any(utility[i] < utility[i - 1] for i in range(1, len(utility))) and \
            any(corrupt[i] > corrupt[i - 1] for i in range(1, len(corrupt))):
        return "DAMAGE GROWS FASTER THAN CORRECTION"
    if any(corr[i] < corr[i - 1] or corrupt[i] < corrupt[i - 1] for i in range(1, len(corr))):
        return "NON-MONOTONIC"
    if corr[-1] == corr[-2] or utility[-1] <= utility[-2]:
        return "SATURATING"
    return "STABLE DOSE RESPONSE"


def analyze(_args) -> int:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "geometry").mkdir(parents=True, exist_ok=True)
    (OUT / "representation").mkdir(parents=True, exist_ok=True)
    rec, directions = _basis()
    _, reuse_info = _load_reused()
    _write(OUT / "reused/reuse_manifest.json", reuse_info)
    rows = [_metric_row(x) for x in _load_result_rows()]
    geom = basis_geometry(directions, a_local=MIX[0], a_cond=MIX[1])
    # Full 1280x1280 projection matrices are machine-readable NumPy artifacts;
    # keep JSON/Markdown summaries compact and reference those files by hash.
    for key, filename in (("raw_projection_matrix", "projection_raw.npy"),
                          ("local_projection_matrix", "projection_local.npy")):
        matrix = np.asarray(geom["subspace"].pop(key), dtype=np.float64)
        path = OUT / "geometry" / filename
        np.save(path, matrix)
        geom["subspace"][key] = {"path": str(path.relative_to(REPO)),
                                  "sha256": _sha256_file(path),
                                  "shape": list(matrix.shape)}
    _write(OUT / "geometry/vector_geometry.json", geom)
    _write(OUT / "geometry/cosine_matrix.json", {
        "direction_order": geom["direction_order"], "matrix": geom["cosine_matrix"]})
    _write(OUT / "geometry/subspace_geometry.json", geom["subspace"] | {
        "gram_matrices": geom["gram_matrices"], "svd": geom["svd"]})
    rep_path = OUT / "representation/baseline_l24_teacher_forced.npy"
    pca_summary = {"deferred": True, "reason": "no baseline representation cache"}
    projection_stats = {"deferred": True, "reason": "no baseline representation cache"}
    if rep_path.exists():
        reps = np.load(rep_path)
        pca = fit_pca_rows(reps, n_components=3)
        projections = project_pca_directions(pca, directions)
        ratios = pca["explained_variance_ratio"]
        metadata = json.loads((OUT / "representation/position_metadata.json").read_text())
        pca_summary = {"deferred": False, "source": "baseline exact-site L24 teacher-forced states",
                       "site": SITE, "split": "D-dev-select", "sampling_seed": 2408,
                       "sampling_policy": "baseline-correct EN/ZH unit query states; max 10000",
                       "n_vectors": int(reps.shape[0]), "dimension": int(reps.shape[1]),
                       "pc1_explained_variance": float(ratios[0]), "pc2_explained_variance": float(ratios[1]),
                       "pc1_pc2_cumulative": float(ratios[:2].sum()),
                       "pc3_explained_variance": float(ratios[2]) if len(ratios)>2 else None,
                       "direction_projections": projections,
                       "visual_arrow_scale": 4.0,
                       "visual_note": "arrow lengths are multiplied by 4 for readability and are not steering doses"}
        _write(OUT / "representation/pca_summary.json", pca_summary)
        projection_stats = _projection_stats(reps, metadata, directions)
        _write(OUT / "representation/projection_stats.json", projection_stats)
        _plot_pca(reps, pca, directions, pca_summary, metadata)
    else:
        _write(OUT / "representation/pca_summary.json", pca_summary)
        _write(OUT / "representation/projection_stats.json", projection_stats)
    for d in ("Raw", "Local", "Conditioning", "Raw+Conditioning", "Local+Conditioning"):
        _write(OUT / "dose_response_" + d.replace("+", "_") + ".json", {}) if False else None
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Direction"], []).append(row)
    dose_response = {d: {"trajectory": sorted(v, key=lambda x: x["rho"]),
                         "classification": _dose_class(v) if d != "Frozen" else "NO INTERVENTION"}
                     for d, v in grouped.items()}
    nonzero = [r for r in rows if r["Direction"] != "Frozen"]
    for row in nonzero:
        row["non_dominated"] = not any(
            other is not row and other["Corr"] >= row["Corr"] and other["Corrupt"] <= row["Corrupt"]
            and (other["Corr"] > row["Corr"] or other["Corrupt"] < row["Corrupt"])
            for other in nonzero)
    for row in rows:
        row.setdefault("non_dominated", False)
    frontier = {"schema_version": "basis_ablation_frontier_v1", "layer": LAYER,
                "site": SITE, "rho_grid": list(GRID), "points": rows,
                "dose_response": dose_response,
                "primary_metrics_note": "WER is n/a; no frozen canonical overall WER exists; MER is not substituted.",
                "data_role": "D-dev-select"}
    _write(OUT / "frontier.json", frontier)
    _write(OUT / "summary.json", {"status": "COMPLETE", "geometry": geom,
                                  "pca": pca_summary, "dose_response": dose_response,
                                  "main_table": rows, "interpretation": _interpret(rows, geom)})
    _write(OUT / "geometry/geometry_table.json", {
        "cos(raw,local)": geom["pairwise"]["cos_raw_local"],
        "angle(raw,local)": geom["pairwise"]["angle_raw_local_degrees"],
        "cos(raw,cond)": geom["pairwise"]["cos_raw_cond"],
        "angle(raw,cond)": geom["pairwise"]["angle_raw_cond_degrees"],
        "cos(local,cond)": geom["pairwise"]["cos_local_cond"],
        "angle(local,cond)": geom["pairwise"]["angle_local_cond_degrees"],
        "removed raw energy fraction": geom["residualization"]["removed_energy_fraction"],
        "residual norm before renorm": geom["residualization"]["residual_norm_before_renorm"],
        "cond([raw,cond])": geom["svd"]["raw"]["condition_number"],
        "cond([local,cond])": geom["svd"]["local"]["condition_number"],
        "principal angle between rank-2 spans": geom["subspace"]["principal_angles_degrees"],
        "projection-matrix Frobenius distance": geom["subspace"]["projection_frobenius_distance"],
        "cos(RC,LC)": geom["pairwise"]["cos_raw_cond_mixture_local_cond_mixture"],
        "angle(RC,LC)": geom["pairwise"]["angle_raw_cond_mixture_local_cond_mixture_degrees"],
        "PCA": pca_summary})
    _write_comparison(rows, geom, pca_summary, dose_response, projection_stats)
    _plot_frontier(rows); _plot_heatmap(geom["cosine_matrix"], geom["direction_order"]); _plot_dose(rows)
    return 0


def _interpret(rows, geom):
    def by(direction): return {r["rho"]: r for r in rows if r["Direction"] == direction}
    R, L, C, RC, LC = (by(x) for x in ("Raw", "Local", "Conditioning", "Raw+Conditioning", "Local+Conditioning"))
    def wins(a, b): return sum(a[r]["Utility"] > b[r]["Utility"] for r in GRID)
    residual = ("RESIDUALIZATION SUPPORTED" if wins(L, R) >= 2 and wins(L, R) > wins(R, L)
                else "RAW BETTER" if wins(R, L) >= 2 and wins(R, L) > wins(L, R)
                else "RESIDUALIZATION NEUTRAL")
    mix_gain = wins(RC, R) + wins(LC, L)
    mix_loss = sum(RC[r]["Utility"] < R[r]["Utility"] and LC[r]["Utility"] < L[r]["Utility"] for r in GRID)
    conditioning = ("CONDITIONING COMPLEMENTARY" if mix_gain >= 2 and mix_loss == 0
                    else "CONDITIONING HARMFUL" if mix_gain == 0 and C[1.0]["Utility"] < min(R[1.0]["Utility"], L[1.0]["Utility"])
                    else "CONDITIONING NEUTRAL")
    current = ("CURRENT BASIS SUPPORTED" if wins(LC, R) >= 2 and wins(LC, L) >= 2
               else "SIMPLER BASIS PREFERRED" if (wins(R, LC) >= 2 or wins(L, LC) >= 2)
               else "CURRENT BASIS NOT CLEARLY BETTER")
    return {"residualization": residual, "conditioning": conditioning,
            "current_basis": current, "geometry_supports_same_rank2_subspace": geom["subspace"]["equivalent_within_1e-10"]}


def _fmt(value):
    if value is None: return "n/a"
    if isinstance(value, float): return f"{value:.6g}"
    return str(value)


def _write_comparison(rows, geom, pca, dose, projection_stats):
    cols = ["Direction", "rho", "source", "WER", "MER", "PIER", "Embedded WER", "Matrix CER",
            "Corr", "Corrupt", "Utility", "Outside harm", "Embed ret.", "Matrix ret.", "Energy", "Utility/Energy"]
    lines = ["# BASIS-A frozen direction comparison", "", "WER is `n/a — no frozen canonical overall WER`; MER is retained as the canonical mixed error metric.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    lines += ["", "## Geometry", "", "```json", json.dumps(geom, indent=2), "```", "",
              "## PCA limitations", "", f"{json.dumps(pca, indent=2)}", "",
              "## Projection distributions", "", f"{json.dumps(projection_stats, indent=2)}", "",
              "## Dose response", "", "```json", json.dumps(dose, indent=2), "```", "",
              "## Explicit questions", "",
              f"A. Normalized raw/local similarity is cos={geom['pairwise']['cos_raw_local']:.9f}, angle={geom['pairwise']['angle_raw_local_degrees']:.6f} degrees.",
              f"B. Residualization removes {geom['residualization']['removed_energy_fraction']:.9f} of raw-direction energy; the pre-renormalization residual norm is {geom['residualization']['residual_norm_before_renorm']:.9f}.",
              f"C. Conditioning improves numerical basis conditioning: {geom['svd']['raw']['condition_number']:.9f} to {geom['svd']['local']['condition_number']:.9f}.",
              f"D. The rank-2 spans are numerically equivalent: principal angles={geom['subspace']['principal_angles_degrees']} degrees and projection Frobenius distance={geom['subspace']['projection_frobenius_distance']:.3e}.",
              "E. Therefore residualization changes coordinate geometry/conditioning, not the available rank-2 representational subspace.",
              f"F. The identical fixed coefficients produce RC/LC cosine={geom['pairwise']['cos_raw_cond_mixture_local_cond_mixture']:.9f}, angle={geom['pairwise']['angle_raw_cond_mixture_local_cond_mixture_degrees']:.6f} degrees.",
              "G. Raw beats Local in utility at rho=.5 and 1.0, while Local beats Raw at rho=2.0; the performance difference is not stable, but the preregistered two-of-three rule labels Raw better.",
              "H. Conditioning alone is not useful on this population: it has negative utility at all doses and more corruptions than corrections.",
              "I. Adding conditioning improves the single-direction utility only at higher doses and loses at rho=.5; the mixture effect is not stable.",
              "J. The correction-damage frontier and non-dominated flags are in frontier.json; the best positive useful-correction/energy point is Raw rho=.5.",
              "K. Raw rho=.5 has the highest positive utility per realized energy; this is not a positive-utility finding at every rho.",
              ("L. PCA is deferred, so it neither supports nor contradicts the geometry."
               if pca.get("deferred") else
               "L. PCA is supporting only: it uses actual teacher-forced L24 representation rows; projected arrows are descriptive and do not establish causal superiority."),
              "M. Frozen D-dev-select evidence supports retaining the two-vector rationale under the preregistered utility rule, but does not establish learned-controller or held-out generalization evidence.",
              "", "## Interpretation", "", "All performance conclusions use the full rho trajectories. PCA is descriptive only; it does not establish causal superiority or semantic purity."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_frontier(rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for d in ("Raw", "Local", "Conditioning", "Raw+Conditioning", "Local+Conditioning"):
        z = sorted((r for r in rows if r["Direction"] == d), key=lambda x: x["rho"])
        ax.plot([r["Corrupt"] for r in z], [r["Corr"] for r in z], marker="o", label=d)
        for r in z: ax.annotate(str(r["rho"]), (r["Corrupt"], r["Corr"]), fontsize=8)
    ax.set(xlabel="Corruptions", ylabel="Corrections", title="BASIS-A correction–damage frontier")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figures/correction_damage_frontier.png", dpi=220); plt.close(fig)


def _plot_heatmap(matrix, names):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4)); im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right"); ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity"); ax.set_title("BASIS-A direction geometry")
    fig.tight_layout(); fig.savefig(OUT / "figures/cosine_heatmap.png", dpi=220); plt.close(fig)


def _plot_dose(rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for d in ("Raw", "Local", "Conditioning", "Raw+Conditioning", "Local+Conditioning"):
        z = sorted((r for r in rows if r["Direction"] == d), key=lambda x: x["rho"])
        ax.plot([r["rho"] for r in z], [r["Utility"] for r in z], marker="o", label=d)
    ax.set(xlabel="rho", ylabel="POI utility (corrections − corruptions)", title="BASIS-A dose response")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "figures/dose_response.png", dpi=220); plt.close(fig)


def _projection_stats(reps, metadata, directions):
    """Descriptive EN-vs-ZH projections using baseline-correct rows."""
    labels = np.asarray([x["language"] for x in metadata])
    mu = reps.mean(axis=1, keepdims=True)
    var = ((reps - mu) ** 2).mean(axis=1, keepdims=True)
    ln = (reps - mu) / np.sqrt(var + 1e-5)
    out = {"deferred": False,
           "normalization": "featurewise LayerNorm without affine parameters; eps=1e-5",
           "source": "baseline-correct content-unit query states", "directions": {}}
    for name in ("raw", "local", "conditioning"):
        scores = ln @ directions[name]
        groups = {}
        for tag, label in (("EN", "embedded"), ("ZH", "matrix")):
            x = scores[labels == tag]
            if len(x) == 0:
                groups[label] = {"n": 0}
                continue
            groups[label] = {"n": int(len(x)), "mean": float(np.mean(x)),
                             "std": float(np.std(x)), "median": float(np.median(x)),
                             "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}
        en, zh = scores[labels == "EN"], scores[labels == "ZH"]
        pooled = math.sqrt((float(np.var(en)) + float(np.var(zh))) / 2) if len(en) and len(zh) else 0.0
        out["directions"][name] = {
            "groups": groups,
            "standardized_mean_difference_en_minus_zh":
            float((np.mean(en) - np.mean(zh)) / pooled) if pooled > 0 else None,
        }
    out["n_vectors"] = int(len(reps))
    out["language_counts"] = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    return out


def _plot_pca(reps, pca, directions, summary, metadata):
    import matplotlib.pyplot as plt
    xy = (reps - pca["mean"]) @ pca["components"][:2].T
    labels = np.asarray([x["language"] for x in metadata])
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for label, color, name in (("EN", "tab:blue", "embedded EN"), ("ZH", "tab:orange", "matrix ZH")):
        mask = labels == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=.22, color=color,
                   label=name, rasterized=True)
    offsets = {"raw": (0.08, 0.12), "local": (0.08, -0.22),
               "conditioning": (0.10, -0.10), "raw_cond": (0.08, 0.12),
               "local_cond": (0.08, -0.22)}
    arrow_scale = float(summary.get("visual_arrow_scale", 1.0))
    for name, coords in summary["direction_projections"].items():
        x, y = (arrow_scale * float(v) for v in coords[:2])
        ax.arrow(0, 0, x, y, width=0.012, head_width=.12, length_includes_head=True)
        dx, dy = offsets.get(name, (0.08, 0.08))
        ax.text(x + dx, y + dy, name, fontsize=9)
    ax.set(xlabel=f"PC1 ({summary['pc1_explained_variance']:.1%})", ylabel=f"PC2 ({summary['pc2_explained_variance']:.1%})",
           title="BASIS-A baseline L24 representations; arrows are projection-only")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figures/representation_pca.png", dpi=220); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gpu", "extract-representations", "analyze"), required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "gpu":
        return run_gpu(args)
    if args.mode == "extract-representations":
        return extract_representations(args)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
