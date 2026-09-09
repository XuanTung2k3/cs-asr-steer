#!/usr/bin/env python
"""Frozen full-dev prerequisite screen (Workstream A).

No training. Runs frozen (ungated, fixed-direction) exact-site steering on the FULL
300-utterance D-dev-select candidate population, at the predeclared candidate layers
{L24 anchor, L26, L27, L31 negative control}, for all five direction families
{Raw, Local, Conditioning, Raw+Cond, Local+Cond}, at rho = 0.5 only.

Purpose: test whether the 10-utterance-atlas Conditioning signal at L26/L27 survives
on the canonical 300-utt population, with bounded damage. The gate is CAUSAL
CORRECTION HEADROOM, not positive frozen net utility (a learned controller may
selectively rescue a globally-too-broad direction — cf. L24 frozen vs DG-05/06).

Reuses: frozen per-layer directions from the BASIS-A2 atlas (hash-checked,
constructed canonically on D-construct), the frozen DG-02 exact-site hook, the DG-04
300-utt population + B0 baseline + outside-harm partition, and the canonical
metrics / result_v1. Never modifies DG-00..DG-08 or BASIS-A/A2 artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj

from experiments.dg05_adaptive_controller import (
    GEN, _atomic_json, _git_commit, _load_reused_dg04, _sha256_file, _slurm_metadata,
)

ATLAS_DIR = REPO / "results/basis_frozen_layer_atlas"
ATLAS_DIRECTIONS = ATLAS_DIR / "directions.json"
OUTPUT_ROOT = REPO / "results/learned_expansion_prereq_frozen"
# Frozen pre-registered screen (do not change after results):
LAYERS = (24, 26, 27, 31)
DIRECTIONS = ("raw", "local", "conditioning", "raw_cond", "local_cond")
RHO = 0.5


def _atlas_layer(layer: int) -> dict[str, Any]:
    rec = json.loads(ATLAS_DIRECTIONS.read_text(encoding="utf-8"))
    return rec["layers"][str(int(layer))]


def _load_direction(layer: int, name: str) -> tuple[torch.Tensor, str]:
    rec = _atlas_layer(layer)
    rel = Path(rec["files"][name])
    fpath = rel if rel.exists() else REPO / rel
    arr = np.asarray(np.load(fpath), dtype=np.float64)
    import hashlib
    digest = "sha256:" + hashlib.sha256(np.ascontiguousarray(arr, np.float64).tobytes()).hexdigest()
    if digest != rec["hashes"][name]:
        raise ValueError(f"atlas direction hash mismatch for {name} at L{layer}")
    return torch.from_numpy(arr.astype(np.float32)), digest


def _decode_fixed(bundle, pop, *, layer, direction, nfp, alpha, scale, batch_size=8):
    """Frozen fixed-direction steering at an arbitrary layer. Returns (texts, audit)."""
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts, energies = {}, []
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0
    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size:(bi + 1) * batch_size]
        uids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, int(layer), direction, alpha=float(alpha), scale=float(scale),
            num_forced_prefix=nfp, norm_preserve=True, mode="steer",
            record=True, record_last_only=False, enforce_contract_layer=False)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inputs, **GEN)
        for rec in hook.records:
            if rec.steered:
                energies.append(rec.edit_norm)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for u, t in zip(uids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[u] = t.strip()
    audit = {"n_steered": len(energies),
             "total_energy": float(np.sum(energies)) if energies else 0.0,
             "mean_energy": float(np.mean(energies)) if energies else 0.0}
    return texts, audit


def _emit(refs, base, method_texts, ids, *, layer, direction_name, rho, scale,
          dir_hash, audit, out_sets, out_langs, bundle, cfg) -> dict[str, Any]:
    from csasr.evaluation import canonical
    from csasr.evaluation import retention as ret
    from csasr.evaluation.dg03_outside_harm import corpus_outside_harm
    from csasr.evaluation.result_schema import CanonicalResult, MethodConfig, validate
    from csasr.lss.manifest import run_id as make_run_id

    r = [refs[i] for i in ids]; b = [base[i] for i in ids]; m = [method_texts[i] for i in ids]
    bm = canonical.corpus_metrics(r, b); mm = canonical.corpus_metrics(r, m)
    metrics = dict(mm)
    metrics.update(canonical.error_metric_gains(bm, mm))
    metrics["transitions"] = canonical.correction_corruption(r, b, m)
    metrics["retention"] = ret.retention_report(r, b, m)
    oh = corpus_outside_harm(r, b, m, out_sets, out_langs)
    metrics["outside_harm"] = int(oh["outside_harm"])
    metrics["outside_harm_accounting"] = oh
    metrics["candidate_utility"] = float(oh["utility"])
    metrics["realized_edit"] = audit
    cond = f"L{layer}_{direction_name}_rho{rho}"
    res = CanonicalResult(
        run_id=make_run_id(f"learned_expansion_prereq/{cond}"),
        system_name=f"prereq_{cond}",
        model_id=bundle.model_id, model_revision=bundle.revision,
        data_role="D-dev-select", decode_regime="greedy", beam=1,
        method=MethodConfig(layer=int(layer), direction_artifact_id=dir_hash,
                            direction_type=f"frozen_{direction_name}", gate_type="frozen",
                            steering_strength=float(rho)),
        metrics=metrics,
        provenance={"layer": int(layer), "direction": direction_name, "rho": float(rho),
                    "scale_s_l": float(scale), "direction_hash": dir_hash,
                    "config_hash": sha256_obj(dict(cfg)), "git_commit": _git_commit(),
                    "decode": GEN, "slurm": _slurm_metadata(),
                    "source": "NEW FROZEN FULL-DEV PREREQUISITE"})
    d = res.to_dict(); validate(d)
    return d


def run_screen(cfg: Mapping[str, Any], layers: Sequence[int], batch_size: int = 8) -> dict[str, Any]:
    started = time.time()
    bundle = load_whisper({"model": dict(cfg["model"])})
    bundle.model.eval()
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    data_cfg = load_config(cfg["data"]["candidate_config"])
    from steer_sweep import data as D
    from experiments.dg04_frozen_baselines import _build_outside_sets
    pop = D.build_population(bundle, data_cfg, "D-dev-select", assert_anchors=True)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    a0 = _load_reused_dg04("B0", ids)
    out_sets, out_langs, out_diag = _build_outside_sets(data_cfg, refs, ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in layers:
        scale = float(_atlas_layer(layer)["scale_s_l"])
        for name in DIRECTIONS:
            direction, dir_hash = _load_direction(layer, name)
            texts, audit = _decode_fixed(bundle, pop, layer=int(layer), direction=direction,
                                         nfp=nfp, alpha=RHO, scale=scale, batch_size=batch_size)
            result = _emit(refs, a0["texts"], texts, ids, layer=int(layer), direction_name=name,
                           rho=RHO, scale=scale, dir_hash=dir_hash, audit=audit,
                           out_sets=out_sets, out_langs=out_langs, bundle=bundle, cfg=cfg)
            cell_dir = OUTPUT_ROOT / f"L{layer}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(cell_dir / f"{name}_rho{RHO}.json",
                         {"result_v1": result, "texts": texts, "audit": audit})
            t = result["metrics"]["transitions"]
            rate = (result["metrics"].get("retention", {}).get("matrix_zh", {}) or {}).get("rate")
            rows.append({
                "layer": int(layer), "direction": name, "rho": RHO, "scale_s_l": scale,
                "mer": result["metrics"].get("mer"), "pier": result["metrics"].get("pier"),
                "pier_gain": result["metrics"].get("pier_gain"),
                "en_wer": result["metrics"].get("en_wer"), "matrix_cer": result["metrics"].get("zh_cer"),
                "corrections": int(t["corrections"]), "corruptions": int(t["corruptions"]),
                "utility": int(t["net_corrections"]),
                "outside_harm": result["metrics"].get("outside_harm"),
                "matrix_retention": float(rate) if rate is not None else None,
                "total_energy": float(audit.get("total_energy", 0.0))})
            print(f"L{layer} {name}: corr={t['corrections']} corrupt={t['corruptions']} "
                  f"U={t['net_corrections']} pier_gain={result['metrics'].get('pier_gain'):.4f} "
                  f"mret={rate} oh={result['metrics'].get('outside_harm')}", flush=True)
    summary = {
        "schema_version": "learned_expansion_prereq_v1",
        "population": {"role": "D-dev-select", "n_utterances": len(ids), "outside_diag": out_diag},
        "rho": RHO, "layers": list(layers), "directions": list(DIRECTIONS),
        "rows": rows, "runtime_sec": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else None,
        "slurm": _slurm_metadata(), "git_commit": _git_commit(),
        "source": "NEW FROZEN FULL-DEV PREREQUISITE"}
    _atomic_json(OUTPUT_ROOT / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expansion/learned_expansion_run.yaml")
    parser.add_argument("--layers", default="24,26,27,31")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    layers = tuple(int(x) for x in str(args.layers).split(","))
    for layer in layers:
        if str(layer) not in json.loads(ATLAS_DIRECTIONS.read_text())["layers"]:
            raise SystemExit(f"no frozen atlas directions for L{layer}")
    if not args.run:
        print(json.dumps({"ready": True, "layers": list(layers), "directions": list(DIRECTIONS),
                          "rho": RHO, "gpu_run": False}, indent=2))
        return 0
    run_screen(cfg, layers, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
