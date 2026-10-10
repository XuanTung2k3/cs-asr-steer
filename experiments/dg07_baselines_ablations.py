#!/usr/bin/env python
"""DG-07A/DG-07B learned baselines and core ablations.

This runner is intentionally inert unless ``--run`` is supplied.  DG-07A
freezes the comparison matrix and prepares this entry point; DG-07B may later
invoke one variant per Slurm job.  The new systems all use the selected DG-06
objective (D1): correction-set CE plus matrix-retention KL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.steering.controller import load_frozen_basis
from csasr.steering.dg06_losses import (
    RetentionSetIndex,
    damage_aware_loss,
    load_retention_set,
)
from csasr.steering.dg07_variants import (
    ExactLayerQvLoRA,
    GateOnlyController,
    GlobalVector,
    assert_frozen_basis,
    lora_parameter_count,
    select_lora_rank,
    trainable_parameter_count,
)
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj
from csasr.utils.provenance import stage_provenance

LAYER = 24
BOTTLENECK = 32
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 3
LR = 5e-4
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
EXPECTED_BATCHES_PER_EPOCH = 1346
EXPECTED_OPTIMIZER_STEPS_PER_EPOCH = 673
BETA = 4.465628877080159
LAMBDA_M = 1.0
TARGET_CONTROLLER_PARAMS = 43_651
BASIS_PATH = REPO / "results/dg03/basis/steering_basis_v1_L24.json"
LOCAL_HASH = "sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93"
COND_HASH = "sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991"
CORRECTION_PATH = REPO / "results/dg05/correction_set_v1.json"
CORRECTION_SHA256 = "sha256:813604876dfb68eb7fc2e8f1864c46f7c57ef25ed4f3c700798c0f88d199a430"
RETENTION_PATH = REPO / "results/dg06/retention_set_v1.json"
RETENTION_SHA256 = "sha256:ad2d38a9b83755b078855c8c16ba03594c759a69cd6b697f9d5e76d22337585d"
TRAIN_BASELINE_PATH = REPO / "results/dg05/controller/training_baseline_hypotheses.json"
GEN = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
           temperature=0.0, max_new_tokens=200, condition_on_prev_tokens=False)

VARIANTS = (
    "LB1_SALSA_EXACT_GLOBAL",
    "LB2_LORA_MATCHED_BUDGET",
    "A1_LOCAL_ONLY",
    "A2_CONDITIONING_ONLY",
    "A3_FIXED_MIXTURE_GATE",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


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


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(cfg: Mapping[str, Any], variant: str, expected_seed: int = SEED) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown DG-07 variant: {variant}")
    basis = cfg.get("basis", {})
    data = cfg.get("data", {})
    train = cfg.get("training", {})
    if int(basis.get("layer", -1)) != LAYER:
        raise ValueError("DG-07 is frozen to the DG-03-selected L24")
    if basis.get("local_hash") != LOCAL_HASH or basis.get("cond_hash") != COND_HASH:
        raise ValueError("DG-07 basis hashes do not match the frozen DG-03 basis")
    if list(data.get("training_roles", [])) != ["loc-train", "util-train"]:
        raise ValueError("DG-07 training roles must be loc-train + util-train")
    if data.get("selection_role") != "D-dev-select":
        raise ValueError("DG-07 selection must use D-dev-select")
    used_roles = [str(x).lower() for x in data.get("training_roles", [])]
    used_roles.append(str(data.get("selection_role", "")).lower())
    if any(term in used_roles for term in ("d-dev-confirm", "d-test")):
        raise ValueError("DG-07 may not use D-dev-confirm or D-test")
    if int(train.get("seed", -1)) != int(expected_seed):
        raise ValueError(f"DG-07 seed must be the expected {expected_seed}")
    for key, expected in (("epochs", EPOCHS),
                          ("batch_size", BATCH_SIZE), ("gradient_accumulation", GRAD_ACCUM)):
        if int(train.get(key, -1)) != expected:
            raise ValueError(f"DG-07 {key} is frozen at {expected}")
    if float(train.get("learning_rate", float("nan"))) != LR:
        raise ValueError("DG-07 LR must match frozen M* LR")
    if float(train.get("beta", float("nan"))) != BETA:
        raise ValueError("DG-07 beta must match frozen M* beta")
    if float(train.get("lambda_m", float("nan"))) != LAMBDA_M:
        raise ValueError("DG-07 lambda_M must be 1.0")
    if int(train.get("expected_batches_per_epoch", -1)) != EXPECTED_BATCHES_PER_EPOCH:
        raise ValueError("DG-07 contributing-batch count must match frozen D1")
    if int(train.get("expected_optimizer_steps_per_epoch", -1)) != EXPECTED_OPTIMIZER_STEPS_PER_EPOCH:
        raise ValueError("DG-07 optimizer-step count must match frozen D1")
    if train.get("objective") != "L_corr + lambda_M L_ret,M":
        raise ValueError("DG-07 learned methods must use the selected D1 objective")
    if cfg.get("slurm", {}).get("partition") != "mig":
        raise ValueError("DG-07 GPU jobs must use partition=mig")
    if int(cfg.get("slurm", {}).get("max_concurrent_gpu_jobs", 0)) > 2:
        raise ValueError("DG-07 permits at most two pending/running GPU jobs")


def make_variant(bundle: Any, basis: torch.Tensor, variant: str, cfg: Mapping[str, Any]):
    """Construct one variant after the frozen seed/model initialization."""
    if variant == "LB1_SALSA_EXACT_GLOBAL":
        module = GlobalVector(bundle.d_model).to(bundle.device)
        return module, "exact_site_global", "all_positions", 1.0, False
    if variant == "LB2_LORA_MATCHED_BUDGET":
        layer = bundle.decoder_layer(LAYER)
        q = layer.self_attn.q_proj
        selected_rank = int(cfg["lora"]["rank"])
        rank, count = select_lora_rank(
            TARGET_CONTROLLER_PARAMS, in_features=q.in_features,
            out_features=q.out_features, target_modules=2)
        if rank != selected_rank or count != int(cfg["lora"]["trainable_parameters"]):
            raise ValueError("frozen LoRA rank/accounting does not match actual Q/V dimensions")
        module = ExactLayerQvLoRA(layer, selected_rank,
                                  alpha=float(cfg["lora"]["alpha"]), target_layer=LAYER).to(bundle.device)
        return module, "decoder_l24_qv_lora", "none", None, True
    if variant == "A1_LOCAL_ONLY":
        direction = basis[:, 0]
    elif variant == "A2_CONDITIONING_ONLY":
        direction = basis[:, 1]
    elif variant == "A3_FIXED_MIXTURE_GATE":
        pi = torch.tensor([0.5, 0.5], dtype=basis.dtype, device=basis.device)
        direction = basis @ pi
    else:
        raise ValueError(variant)
    module = GateOnlyController(direction, BOTTLENECK).to(bundle.device)
    assert_frozen_basis(module)
    return module, "frozen_basis_exact_site", "adaptive_gate", BETA, False


def _load_reused_datasets(bundle: Any, cfg: Mapping[str, Any]):
    from steer_sweep.trackb.data import build_examples
    if _sha256_file(CORRECTION_PATH) != CORRECTION_SHA256:
        raise RuntimeError("DG-05 C_E artifact hash mismatch")
    if _sha256_file(RETENTION_PATH) != RETENTION_SHA256:
        raise RuntimeError("DG-06 retention artifact hash mismatch")
    from csasr.steering.dg05_training import load_correction_set
    correction = load_correction_set(CORRECTION_PATH)
    retention = json.loads(RETENTION_PATH.read_text(encoding="utf-8"))
    if retention.get("baseline_sha256") != _sha256_file(TRAIN_BASELINE_PATH):
        raise RuntimeError("DG-06 retention populations are not tied to frozen p0 baseline")
    r_e = load_retention_set(retention["embedded_retention_R_E"])
    r_m = load_retention_set(retention["matrix_retention_R_M"])
    data_cfg = load_config(cfg["data"]["candidate_config"])
    examples = build_examples(bundle, data_cfg, ("loc-train", "util-train"))
    if not examples:
        raise RuntimeError("DG-07 training pool is empty")
    return examples, correction, r_e, r_m


def _batch_loss(bundle: Any, module: torch.nn.Module, examples: Sequence[Any], correction,
                r_e: RetentionSetIndex, r_m: RetentionSetIndex, *, prefix_width: int,
                is_lora: bool) -> tuple[torch.Tensor, dict[str, float]]:
    # Reuse DG-06's exact C_E/R_M mask construction.  R_E is loaded and
    # provenance-checked but inactive because M*=D1.
    from experiments.dg06_damage_aware import collate_dg06_batch
    batch = collate_dg06_batch(bundle, examples, correction, r_e, r_m,
                               prefix_width=prefix_width, include_embedded=False)
    with torch.no_grad():
        baseline = bundle.model(input_features=batch["input_features"],
                                attention_mask=batch["attention_mask"],
                                decoder_input_ids=batch["decoder_input_ids"],
                                use_cache=False)
    if is_lora:
        with module:
            output = bundle.model(input_features=batch["input_features"],
                                  attention_mask=batch["attention_mask"],
                                  decoder_input_ids=batch["decoder_input_ids"],
                                  use_cache=False)
    else:
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, LAYER, direction=None, alpha=float(getattr(module, "alpha", BETA)),
            scale=1.0, action_fn=module.action, num_forced_prefix=prefix_width,
            norm_preserve=True, mode="train", record=False,
            enforce_contract_layer=True)
        with hook:
            output = bundle.model(input_features=batch["input_features"],
                                  attention_mask=batch["attention_mask"],
                                  decoder_input_ids=batch["decoder_input_ids"],
                                  use_cache=False)
    total, components = damage_aware_loss(
        output.logits.float()[:, :-1], baseline.logits.float()[:, :-1],
        batch["labels"], batch["ce_mask"], batch["rm_mask"], None,
        lambda_m=LAMBDA_M, lambda_e=1.0)
    (total / GRAD_ACCUM).backward()
    return total.detach(), components


def _decode_variant(bundle: Any, pop: Any, module: torch.nn.Module, *, prefix_width: int,
                    is_lora: bool) -> tuple[dict[str, str], dict[str, Any]]:
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts: dict[str, str] = {}
    energies: list[float] = []
    module.eval()
    for start in range(0, len(manifest), BATCH_SIZE):
        batch = manifest.iloc[start:start + BATCH_SIZE]
        ids = [str(x) for x in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        if is_lora:
            context = module
        else:
            context = DecoderPostCrossAttnInterventionHook(
                bundle, LAYER, direction=None,
                alpha=float(getattr(module, "alpha", BETA)), scale=1.0,
                action_fn=module.action, num_forced_prefix=prefix_width,
                norm_preserve=True, mode="steer", record=True,
                record_last_only=False, enforce_contract_layer=True)
        with context, torch.inference_mode():
            out = bundle.model.generate(**inputs, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for uid, text in zip(ids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[uid] = str(text).strip()
        if not is_lora:
            energies.extend(float(r.edit_norm) for r in context.records if r.steered)
    if is_lora:
        return texts, {"applicability": "not_applicable", "reason": "parameter-efficient backbone update"}
    return texts, {
        "applicability": "exact_site_intervention",
        "n_steered": len(energies),
        "total_energy": float(np.sum(energies)) if energies else 0.0,
        "mean_energy": float(np.mean(energies)) if energies else 0.0,
    }


def _select(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [r for r in records if float(r.get("utility", -1e30)) > 0
                and bool(r.get("valid_outside_harm", False))]
    if not eligible:
        return None
    def energy_key(record: Mapping[str, Any]) -> float:
        # LoRA has no exact-site intervention energy; its canonical field is
        # explicitly not applicable and therefore sorts last on this tie-break.
        value = record.get("total_energy")
        return float(value) if value is not None else float("inf")

    return sorted(eligible, key=lambda r: (
        -float(r["utility"]), -float(r.get("pier_gain", -1e30)),
        -float(r.get("matrix_retention", -1.0)),
        energy_key(r),
    ))[0]


def _controller_init_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().float().numpy()).tobytes())
    return "sha256:" + h.hexdigest()


def run_training(cfg: Mapping[str, Any], output_dir: Path, variant: str,
                 seed: int = SEED) -> dict[str, Any]:
    started = time.time()
    validate_config(cfg, variant, expected_seed=seed)
    set_seed(int(seed))
    basis, basis_record = load_frozen_basis(
        BASIS_PATH, expected_layer=LAYER, expected_local_hash=LOCAL_HASH,
        expected_cond_hash=COND_HASH)
    bundle = load_whisper({"model": dict(cfg["model"])})
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("DG-07 requires the Whisper backbone to remain frozen")
    module, direction_type, gate_type, steering_strength, is_lora = make_variant(
        bundle, basis, variant, cfg)
    if not is_lora:
        assert_frozen_basis(module)
    n_params = trainable_parameter_count(module)
    examples, correction, r_e, r_m = _load_reused_datasets(bundle, cfg)
    prefix_width = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    optimizer = torch.optim.AdamW(module.trainable_parameters if hasattr(module, "trainable_parameters")
                                  else [p for p in module.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_provenance(dict(cfg), f"dg07_{variant.lower()}")
    manifest.update({
        "variant": variant, "git_commit": _git_commit(), "layer": LAYER,
        "basis_record": basis_record, "basis_hashes": {"local": LOCAL_HASH, "cond": COND_HASH},
        "beta": steering_strength, "objective": "L_corr + lambda_M L_ret,M",
        "lambda_m": LAMBDA_M, "lambda_e": None,
        "training_roles": ["loc-train", "util-train"], "selection_role": "D-dev-select",
        "seed": int(seed), "trainable_parameter_count": n_params,
        "backbone_frozen": True, "basis_frozen": not is_lora,
        "exact_site": ("decoder post-cross-attention residual, pre-FFN"
                       if not is_lora else "decoder layer 24 self-attention Q/V PEFT target"),
        "controller_init_hash": _controller_init_hash(module),
        "correction_set": {"path": str(CORRECTION_PATH), "sha256": CORRECTION_SHA256,
                           "n_positions": correction.n_positions},
        "retention_set": {"path": str(RETENTION_PATH), "sha256": _sha256_file(RETENTION_PATH),
                          "R_E_positions": r_e.n_positions, "R_M_positions": r_m.n_positions},
        "checkpoint_selection": "positive utility -> PIER gain -> matrix retention -> lower energy",
        "decode": GEN, "slurm": {"partition": "mig", "max_concurrent_gpu_jobs": 2},
        "data_safety": {"forbidden": ["D-dev-confirm", "D-test"],
                        "inference_inputs": ["site_state"] if not is_lora else []},
    })
    if is_lora:
        manifest["lora"] = {"target_modules": [
            "model.model.decoder.layers.24.self_attn.q_proj",
            "model.model.decoder.layers.24.self_attn.v_proj"],
            "rank": module.rank, "alpha": module.alpha,
            "trainable_parameters": module.parameter_count(), "rank_selection": "nearest to 43651"}
    _atomic_json(output_dir / "manifest.json", manifest)

    # Match DG-06's contributing-batch rule: batches with no C_E or R_M
    # position do not consume optimizer budget.
    def batch_has_target(batch_examples: Sequence[Any]) -> bool:
        return any(correction.positions.get(str(e.utterance_id))
                   or r_m.positions.get(str(e.utterance_id))
                   for e in batch_examples)

    batches = [examples[i:i + BATCH_SIZE]
               for i in range(0, len(examples), BATCH_SIZE)
               if batch_has_target(examples[i:i + BATCH_SIZE])]
    if not batches:
        raise RuntimeError("DG-07 has no contributing C_E/R_M batches")
    if len(batches) != EXPECTED_BATCHES_PER_EPOCH:
        raise RuntimeError(f"DG-07 contributing batches changed: {len(batches)}")
    history: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        module.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        aggregate = {"l_corr": 0.0, "l_ret_m": 0.0, "total": 0.0}
        counts = {"n_corr": 0, "n_ret_m": 0}
        optimizer_steps = 0
        for index, batch_examples in enumerate(batches):
            _loss, comp = _batch_loss(bundle, module, batch_examples, correction, r_e, r_m,
                                      prefix_width=prefix_width, is_lora=is_lora)
            aggregate["l_corr"] += comp["l_corr"] * comp["n_corr"]
            aggregate["l_ret_m"] += comp["l_ret_m"] * comp["n_ret_m"]
            aggregate["total"] += comp["total"]
            counts["n_corr"] += comp["n_corr"]
            counts["n_ret_m"] += comp["n_ret_m"]
            if (index + 1) % GRAD_ACCUM == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(module.parameters(), CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if optimizer_steps != EXPECTED_OPTIMIZER_STEPS_PER_EPOCH:
            raise RuntimeError(f"DG-07 optimizer budget changed: {optimizer_steps}")
        components = {
            "raw_l_corr": aggregate["l_corr"] / max(1, counts["n_corr"]),
            "raw_l_ret_m": aggregate["l_ret_m"] / max(1, counts["n_ret_m"]),
            "weighted_l_ret_m": LAMBDA_M * aggregate["l_ret_m"] / max(1, counts["n_ret_m"]),
            "mean_total_loss": aggregate["total"] / max(1, len(batches)),
            "n_C_E": counts["n_corr"], "n_R_M": counts["n_ret_m"],
            "n_batches": len(batches), "n_optimizer_steps": optimizer_steps,
        }
        history.append({"epoch": epoch + 1, **components,
                        "runtime_sec": time.time() - epoch_start,
                        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available() else None})
        _atomic_torch(output_dir / f"checkpoint_epoch{epoch + 1}.pt", {
            "schema_version": "dg07_learned_variant_v1", "variant": variant,
            "epoch": epoch + 1, "state_dict": {k: v.detach().cpu().clone()
                                                   for k, v in module.state_dict().items()},
            "trainable_parameter_count": n_params, "objective": manifest["objective"],
            "lambda_m": LAMBDA_M, "layer": LAYER, "basis_hashes": manifest["basis_hashes"],
            "config_hash": sha256_obj(dict(cfg)), "git_commit": _git_commit(),
        })
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    _atomic_json(output_dir / "training_history.json", {"records": history})

    # Evaluation is deliberately last and only on the frozen selection population.
    from steer_sweep import data as D
    from experiments.dg04_frozen_baselines import _build_outside_sets, _result_v1
    from experiments.dg05_adaptive_controller import _load_reused_dg04
    dcfg = load_config(cfg["data"]["candidate_config"])
    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    ids = list(pop.utterance_ids)
    refs = D.reference_of(pop)
    baseline = _load_reused_dg04("B0", ids)
    out_sets, out_langs, out_diag = _build_outside_sets(dcfg, refs, ids)
    eval_records: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        payload = torch.load(output_dir / f"checkpoint_epoch{epoch}.pt",
                             map_location=bundle.device, weights_only=False)
        module.load_state_dict(payload["state_dict"], strict=True)
        texts, audit = _decode_variant(bundle, pop, module, prefix_width=prefix_width,
                                       is_lora=is_lora)
        result = _result_v1(
            refs, baseline["texts"], texts, ids, cond=f"{variant}_epoch{epoch}",
            provenance={"git_commit": _git_commit(), "model_id": bundle.model_id,
                        "model_revision": bundle.revision, "layer": LAYER,
                        "basis_artifact_hashes": {"local": LOCAL_HASH, "cond": COND_HASH},
                        "config_hash": sha256_obj(dict(cfg)), "decode": GEN,
                        "data_role": "D-dev-select", "training_roles": ["loc-train", "util-train"],
                        "variant": variant, "slurm": manifest.get("slurm")},
            direction_type=direction_type, steering_strength=steering_strength,
            basis_id=LOCAL_HASH if not is_lora else None,
            out_sets=out_sets, out_langs=out_langs, audit=audit)
        result["run_id"] = result["run_id"].replace("dg04/", "dg07/", 1)
        result["system_name"] = result["system_name"].replace("dg04_", "dg07_", 1)
        result["method"]["gate_type"] = gate_type
        result["seed"] = int(seed)
        transitions = result["metrics"]["transitions"]
        energy = audit.get("total_energy")
        record = {"epoch": epoch, "variant": variant,
                  "checkpoint": str(output_dir / f"checkpoint_epoch{epoch}.pt"),
                  "checkpoint_sha256": _sha256_file(output_dir / f"checkpoint_epoch{epoch}.pt"),
                  "result_v1": result, "texts": texts,
                  "utility": float(transitions["net_corrections"]),
                  "pier_gain": float(result["metrics"].get("pier_gain", 0.0)),
                  "matrix_retention": float(result["metrics"]["retention"]["matrix_zh"]["rate"]),
                  "valid_outside_harm": result["metrics"].get("outside_harm") is not None,
                  "total_energy": float(energy) if energy is not None else None,
                  "audit": audit}
        eval_records.append(record)
        _atomic_json(output_dir / f"evaluation_epoch{epoch}.json", record)
    selected = _select(eval_records)
    selection = {"rule": manifest["checkpoint_selection"], "variant": variant,
                 "selected_epoch": selected.get("epoch") if selected else None,
                 "selected_checkpoint": selected.get("checkpoint") if selected else None,
                 "population": {"role": "D-dev-select", "n_utterances": len(ids),
                                "outside_harm": out_diag}}
    if selected:
        shutil.copy2(selected["checkpoint"], output_dir / "selected_checkpoint.pt")
        selection["selected_checkpoint_sha256"] = _sha256_file(output_dir / "selected_checkpoint.pt")
    _atomic_json(output_dir / "selection.json", selection)
    summary = {"schema_version": "dg07_run_v1", "variant": variant, "seed": int(seed),
               "trainable_parameter_count": n_params, "manifest": manifest,
               "history": history, "evaluations": eval_records, "selection": selection,
               "runtime_sec": time.time() - started,
               "checkpoint_size_bytes": (output_dir / "selected_checkpoint.pt").stat().st_size
               if selected else None,
               "optimizer_updates": sum(int(x["n_optimizer_steps"]) for x in history),
               "peak_gpu_memory_bytes": max((x["peak_gpu_memory_bytes"] or 0 for x in history), default=0),
               "artifacts": {"correction_set": str(CORRECTION_PATH),
                             "retention_set": str(RETENTION_PATH),
                             "selected_checkpoint": str(output_dir / "selected_checkpoint.pt")
                             if selected else None}}
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dg07_baselines_ablations.yaml")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="override training seed (DG-08 multi-seed); default keeps the frozen 42")
    parser.add_argument("--run", action="store_true", help="execute one GPU variant")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = SEED if args.seed is None else int(args.seed)
    if seed != SEED:
        # DG-08 varies only the training seed; align the config's seed field.
        cfg = dict(cfg)
        cfg["training"] = {**cfg.get("training", {}), "seed": seed}
    validate_config(cfg, args.variant, expected_seed=seed)
    if not args.run:
        lora_count = int(cfg["lora"]["trainable_parameters"])
        print(json.dumps({"ready": True, "variant": args.variant, "layer": LAYER,
                          "objective": "L_corr + lambda_M L_ret,M", "lambda_m": LAMBDA_M,
                          "lora_rank": cfg["lora"]["rank"], "lora_parameters": lora_count,
                          "gpu_run": False}, indent=2))
        return 0
    output = Path(args.output_dir) if args.output_dir else REPO / "results/dg07" / args.variant
    run_training(cfg, output, args.variant, seed=seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
