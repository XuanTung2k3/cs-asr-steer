#!/usr/bin/env python
"""DG-08 locked D-test / timing decoding for the frozen finalists.

Decodes F0 (frozen Whisper), F1 (DG-04 frozen steering), F2 (SALSA global),
F3 (matched-budget LoRA), and F4 (M* adaptive controller) on the locked D-test
manifest under a fixed decode regime (greedy or beam-5), and emits a text-only
canonical ``result_v1`` per system.  Outside-harm / candidate-utility are
recorded as NOT AVAILABLE on D-test (no candidate/POI alignment artifacts exist
for the locked split); every text-derived metric is computed from transcripts.

This runner never trains, never selects checkpoints, and only reads the frozen
lock artifact's checkpoints.  It is inert without ``--run``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.steering.controller import FixedBasisAdaptiveController, load_frozen_basis
from csasr.steering.dg07_variants import ExactLayerQvLoRA, GlobalVector
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj

LAYER = 24
BETA = 4.465628877080159
BASIS_PATH = REPO / "results/dg03/basis/steering_basis_v1_L24.json"
LOCAL_HASH = "sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93"
COND_HASH = "sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991"
LORA_RANK = 9
LORA_ALPHA = 9.0
BATCH_SIZE = 8


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _slurm() -> dict[str, Any]:
    return {"job_ids": [os.environ["SLURM_JOB_ID"]] if os.environ.get("SLURM_JOB_ID") else [],
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME")}


def load_dtest_manifest(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], str]:
    """Read the locked D-test parquet into a decode manifest + reference/dialogue maps."""
    path = Path(cfg["data"]["dtest_locked_parquet"])
    fingerprint = _sha256_file(path)
    df = pd.read_parquet(path)
    ref_col = cfg["data"]["reference_column"]
    df = df[["utterance_id", "audio_path", "duration_sec", "dialogue_id", ref_col]].copy()
    df["utterance_id"] = df["utterance_id"].astype(str)
    references = {r.utterance_id: str(getattr(r, ref_col)) for r in df.itertuples()}
    dialogues = {str(r.utterance_id): str(r.dialogue_id) for r in df.itertuples()}
    return df, references, dialogues, fingerprint


def load_timing_manifest(bundle: Any, cfg: Mapping[str, Any], n: int = 100) -> pd.DataFrame:
    """Frozen deterministic timing subset: first N D-dev-select utts by duration."""
    root = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3")
    import pyarrow.parquet as pq
    df = pq.read_table(root / "manifests/roles/role_D-dev-select.parquet",
                       columns=["utterance_id", "audio_path", "duration_sec"],
                       filters=[("role", "==", "D-dev-select")]).to_pandas()
    df = df.sort_values(["duration_sec", "utterance_id"]).head(int(n)).reset_index(drop=True)
    df["utterance_id"] = df["utterance_id"].astype(str)
    return df


def _basis_and_scale() -> tuple[torch.Tensor, dict[str, Any], float]:
    basis, record = load_frozen_basis(BASIS_PATH, expected_layer=LAYER,
                                      expected_local_hash=LOCAL_HASH, expected_cond_hash=COND_HASH)
    s_l = float(record["provenance"]["mean_site_norm_s_l"])
    return basis, record, s_l


def build_system(bundle: Any, basis: torch.Tensor, s_l: float, name: str,
                 checkpoint: str | None) -> dict[str, Any]:
    """Return a decode spec for one finalist (optionally at one seed's checkpoint)."""
    if name == "F0_FROZEN_WHISPER":
        return {"name": name, "mode": "baseline"}
    if name == "F1_DG04_FROZEN_STEERING":
        v_local = basis[:, 0].detach().clone().to(bundle.device)
        return {"name": name, "mode": "fixed_dir", "direction": v_local,
                "alpha": 0.5, "scale": s_l}
    if name == "F2_SALSA":
        module = GlobalVector(bundle.d_model).to(bundle.device)
        state = torch.load(checkpoint, map_location=bundle.device, weights_only=False)["state_dict"]
        module.load_state_dict(state, strict=True)
        module.eval()
        return {"name": name, "mode": "action", "module": module, "alpha": 1.0, "scale": 1.0}
    if name == "F3_LORA":
        module = ExactLayerQvLoRA(bundle.decoder_layer(LAYER), LORA_RANK, alpha=LORA_ALPHA,
                                  target_layer=LAYER).to(bundle.device)
        state = torch.load(checkpoint, map_location=bundle.device, weights_only=False)["state_dict"]
        module.load_state_dict(state, strict=True)
        module.eval()
        return {"name": name, "mode": "lora", "module": module}
    if name == "F4_MSTAR":
        module = FixedBasisAdaptiveController(bundle.d_model, basis, 32).to(bundle.device)
        state = torch.load(checkpoint, map_location=bundle.device, weights_only=False)["state_dict"]
        module.load_state_dict(state, strict=True)
        module.eval()
        return {"name": name, "mode": "action", "module": module, "alpha": BETA, "scale": 1.0}
    raise ValueError(f"unknown system {name}")


def decode_system(bundle: Any, manifest: pd.DataFrame, spec: Mapping[str, Any],
                  gen: Mapping[str, Any], nfp: int) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = manifest.sort_values(["duration_sec", "utterance_id"]).reset_index(drop=True)
    texts: dict[str, str] = {}
    energies: list[float] = []
    n_batches = math.ceil(len(manifest) / BATCH_SIZE) if len(manifest) else 0
    for bi in range(n_batches):
        batch = manifest.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        uids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        mode = spec["mode"]
        if mode == "baseline":
            with torch.inference_mode():
                out = bundle.model.generate(**inputs, **gen)
        elif mode == "lora":
            with spec["module"], torch.inference_mode():
                out = bundle.model.generate(**inputs, **gen)
        else:
            if mode == "fixed_dir":
                hook = DecoderPostCrossAttnInterventionHook(
                    bundle, LAYER, spec["direction"], alpha=float(spec["alpha"]),
                    scale=float(spec["scale"]), num_forced_prefix=nfp, gate_fn=None,
                    norm_preserve=True, mode="steer", record=True, record_last_only=False,
                    enforce_contract_layer=True)
            else:  # action (SALSA global / M* controller)
                hook = DecoderPostCrossAttnInterventionHook(
                    bundle, LAYER, direction=None, alpha=float(spec["alpha"]),
                    scale=float(spec["scale"]), action_fn=spec["module"].action,
                    num_forced_prefix=nfp, norm_preserve=True, mode="steer", record=True,
                    record_last_only=False, enforce_contract_layer=True)
            with hook, torch.inference_mode():
                out = bundle.model.generate(**inputs, **gen)
            energies.extend(float(r.edit_norm) for r in hook.records if r.steered)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for uid, text in zip(uids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[uid] = str(text).strip()
    if spec["mode"] in ("baseline", "lora"):
        audit = {"applicability": "not_applicable" if spec["mode"] == "lora" else "baseline"}
    else:
        audit = {"applicability": "exact_site_intervention", "n_steered": len(energies),
                 "total_energy": float(np.sum(energies)) if energies else 0.0,
                 "mean_energy": float(np.mean(energies)) if energies else 0.0}
    return texts, audit


def score_dtest(references: Mapping[str, str], base_texts: Mapping[str, str],
                method_texts: Mapping[str, str], ids: Sequence[str], *, system: str,
                seed: int | None, regime: str, provenance: Mapping[str, Any],
                audit: Mapping[str, Any]) -> dict[str, Any]:
    """Text-only canonical result_v1 for D-test (outside-harm N/A)."""
    from csasr.evaluation import canonical, retention as ret
    from csasr.evaluation.result_schema import CanonicalResult, MethodConfig, validate
    from csasr.lss.manifest import run_id as make_run_id
    r = [references[i] for i in ids]
    b = [base_texts[i] for i in ids]
    m = [method_texts[i] for i in ids]
    bm = canonical.corpus_metrics(r, b)
    mm = canonical.corpus_metrics(r, m)
    metrics = dict(mm)
    metrics.update(canonical.error_metric_gains(bm, mm))
    metrics["transitions"] = canonical.correction_corruption(r, b, m)
    metrics["retention"] = ret.retention_report(r, b, m)
    metrics["outside_harm"] = None
    metrics["outside_harm_available"] = False
    metrics["outside_harm_note"] = ("D-test carries no candidate/POI alignment artifacts; "
                                    "canonical outside-harm/candidate-utility are not computable "
                                    "on the locked split and are reported on D-dev-select only")
    metrics["net_correction_utility"] = int(metrics["transitions"]["net_corrections"])
    metrics["realized_edit"] = dict(audit)
    beam = int(provenance["decode"].get("num_beams", 1))
    res = CanonicalResult(
        run_id=make_run_id(f"dg08/{regime}/{system}" + (f"_seed{seed}" if seed is not None else "")),
        system_name=f"dg08_{system}" + (f"_seed{seed}" if seed is not None else ""),
        model_id=provenance.get("model_id"), model_revision=provenance.get("model_revision"),
        data_role="D-test", decode_regime=("greedy" if beam == 1 else f"beam{beam}"), beam=beam,
        method=MethodConfig(layer=(LAYER if system not in ("F0_FROZEN_WHISPER",) else None),
                            direction_artifact_id=None, direction_type=system,
                            gate_type=system, steering_strength=None),
        metrics=metrics, provenance=dict(provenance))
    d = res.to_dict()
    d["seed"] = seed
    validate(d)
    return d


def resolve_targets(spec_list: Sequence[str], lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn 'F2:13,F4:42,F0' tokens into (system, seed, checkpoint) targets from the lock."""
    ckpts = lock["checkpoints"]
    targets: list[dict[str, Any]] = []
    for token in spec_list:
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, seed_s = token.split(":")
            seed = int(seed_s)
            ck = ckpts[name][str(seed)]["checkpoint"]
            targets.append({"system": name, "seed": seed, "checkpoint": ck})
        else:
            targets.append({"system": token, "seed": None, "checkpoint": None})
    return targets


def run(cfg: Mapping[str, Any], *, regime: str, systems: Sequence[str], out_dir: Path,
        task: str) -> dict[str, Any]:
    started = time.time()
    lock_path = REPO / "results/dg08/DG08_TEST_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("DG08_TEST_LOCK.json missing; lock finalists before any D-test decode")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gen = dict(cfg["decode"]["greedy" if regime == "greedy" else "beam5"])
    bundle = load_whisper({"model": dict(cfg["model"])})
    bundle.model.eval()
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    basis, basis_record, s_l = _basis_and_scale()
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")

    if task == "timing":
        return run_timing(cfg, bundle, basis, s_l, nfp, gen, lock, out_dir, started)

    df, references, dialogues, fingerprint = load_dtest_manifest(cfg)
    if fingerprint != lock["dtest_manifest_fingerprint"]:
        raise RuntimeError("D-test manifest fingerprint does not match the lock")
    ids = [str(u) for u in df.sort_values(["duration_sec", "utterance_id"])["utterance_id"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir.parent / "dtest_manifest.json",
                 {"n_utterances": len(ids), "n_dialogues": len(set(dialogues.values())),
                  "fingerprint": fingerprint, "ids": ids, "references": references,
                  "dialogues": dialogues})

    provenance_base = {"git_commit": _git_commit(), "model_id": bundle.model_id,
                       "model_revision": bundle.revision, "decode": gen, "regime": regime,
                       "data_role": "D-test", "test_lock_commit": lock.get("git_commit"),
                       "dtest_manifest_fingerprint": fingerprint,
                       "basis_artifact_hashes": {"local": LOCAL_HASH, "cond": COND_HASH},
                       "slurm": _slurm()}

    # F0 baseline for this regime is required to score every method.
    f0_path = out_dir / "F0_FROZEN_WHISPER.json"
    if f0_path.exists():
        base_texts = json.loads(f0_path.read_text(encoding="utf-8"))["texts"]
    else:
        spec = build_system(bundle, basis, s_l, "F0_FROZEN_WHISPER", None)
        base_texts, audit = decode_system(bundle, df, spec, gen, nfp)
        result = score_dtest(references, base_texts, base_texts, ids, system="F0_FROZEN_WHISPER",
                             seed=None, regime=regime, provenance=provenance_base, audit=audit)
        _atomic_json(f0_path, {"system": "F0_FROZEN_WHISPER", "seed": None, "regime": regime,
                               "result_v1": result, "texts": base_texts})

    written = ["F0_FROZEN_WHISPER"]
    for target in resolve_targets(systems, lock):
        name, seed, ckpt = target["system"], target["seed"], target["checkpoint"]
        if name == "F0_FROZEN_WHISPER":
            continue
        tag = f"{name}" + (f"_seed{seed}" if seed is not None else "")
        dest = out_dir / f"{tag}.json"
        if dest.exists():
            written.append(tag)
            continue
        if ckpt is not None and _sha256_file(REPO / ckpt if not os.path.isabs(ckpt) else Path(ckpt)) \
                != lock["checkpoints"][name][str(seed)]["sha256"]:
            raise RuntimeError(f"{tag} checkpoint hash does not match the lock")
        spec = build_system(bundle, basis, s_l, name,
                            str(REPO / ckpt) if ckpt and not os.path.isabs(ckpt) else ckpt)
        texts, audit = decode_system(bundle, df, spec, gen, nfp)
        prov = dict(provenance_base)
        prov["checkpoint"] = ckpt
        prov["checkpoint_sha256"] = (lock["checkpoints"][name][str(seed)]["sha256"]
                                     if seed is not None else None)
        result = score_dtest(references, base_texts, texts, ids, system=name, seed=seed,
                             regime=regime, provenance=prov, audit=audit)
        _atomic_json(dest, {"system": name, "seed": seed, "regime": regime,
                            "result_v1": result, "texts": texts})
        written.append(tag)
    summary = {"schema_version": "dg08_dtest_eval_v1", "regime": regime, "task": task,
               "written": written, "n_utterances": len(ids),
               "runtime_sec": time.time() - started, "slurm": _slurm()}
    _atomic_json(out_dir / f"_eval_summary_{regime}.json", summary)
    return summary


def run_timing(cfg: Mapping[str, Any], bundle: Any, basis: torch.Tensor, s_l: float, nfp: int,
               gen: Mapping[str, Any], lock: Mapping[str, Any], out_dir: Path,
               started: float) -> dict[str, Any]:
    """Inference timing on the frozen D-dev-select subset for F0/F2/F3/F4 (seed 42)."""
    df = load_timing_manifest(bundle, cfg, n=100)
    total_audio = float(df["duration_sec"].astype(float).sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    plan = [("F0_FROZEN_WHISPER", None)]
    for name in ("F2_SALSA", "F3_LORA", "F4_MSTAR"):
        plan.append((name, 42))
    for name, seed in plan:
        ckpt = (str(REPO / lock["checkpoints"][name][str(seed)]["checkpoint"])
                if seed is not None else None)
        spec = build_system(bundle, basis, s_l, name, ckpt)
        # one warmup batch
        warm = df.head(BATCH_SIZE)
        decode_system(bundle, warm, spec, gen, nfp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        decode_system(bundle, df, spec, gen, nfp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        records[name] = {"n_utterances": int(len(df)), "total_audio_sec": total_audio,
                         "decode_sec": elapsed, "utt_per_sec": len(df) / elapsed,
                         "rtf": elapsed / total_audio}
    f0 = records["F0_FROZEN_WHISPER"]["decode_sec"]
    for name, rec in records.items():
        rec["overhead_vs_F0"] = rec["decode_sec"] / f0
    payload = {"schema_version": "dg08_timing_v1", "regime": "greedy",
               "timing_subset": "D-dev-select first 100 by duration", "records": records,
               "gpu": os.environ.get("SLURM_JOB_PARTITION"), "runtime_sec": time.time() - started,
               "slurm": _slurm()}
    _atomic_json(out_dir / "timing.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dg08_locked_eval.yaml")
    parser.add_argument("--task", choices=["dtest", "timing"], default="dtest")
    parser.add_argument("--regime", choices=["greedy", "beam5"], default="greedy")
    parser.add_argument("--systems", default="",
                        help="comma list, e.g. 'F1_DG04_FROZEN_STEERING,F2_SALSA:13,F4_MSTAR:42'")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if not args.run:
        print(json.dumps({"ready": True, "task": args.task, "regime": args.regime,
                          "systems": args.systems, "gpu_run": False}, indent=2))
        return 0
    out_dir = (Path(args.output_dir) if args.output_dir
               else REPO / "results/dg08" / ("timing" if args.task == "timing" else f"dtest/{args.regime}"))
    systems = [s for s in args.systems.split(",") if s.strip()]
    run(cfg, regime=args.regime, systems=systems, out_dir=out_dir, task=args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
