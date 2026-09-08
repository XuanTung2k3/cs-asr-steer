#!/usr/bin/env python
"""DG-06 damage-aware correction + retention training path.

This entry point **extends the frozen DG-05 correction-only path** (it reuses the
DG-05 controller, exact-site hook, collation, correction set, free-decode
evaluation, and selection helpers) by adding the MC §8 KL-to-baseline retention
losses. It never modifies the frozen DG-05 runner.

Variants:
- ``d1`` — L = L_corr + lambda_M * L_ret,M   (correction + matrix retention)
- ``d2`` — L = L_corr + lambda_M * L_ret,M + lambda_E * L_ret,E  (full damage-aware)

D0 (correction-only) is reused from DG-05 and is not retrained here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from csasr.lss.sites import num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.steering.controller import FixedBasisAdaptiveController
from csasr.steering.dg05_training import (
    correction_target_mask,
    load_correction_set,
)
from csasr.steering.dg06_losses import (
    RetentionSetIndex,
    baseline_provenance,
    damage_aware_loss,
    load_retention_set,
    retention_position_mask,
    retention_set_payload,
)
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj
from csasr.utils.provenance import stage_provenance

# Reuse the frozen DG-05 constants and helpers verbatim.
from experiments.dg05_adaptive_controller import (
    BATCH_SIZE,
    BOTTLENECK,
    CLIP_NORM,
    DG04_REFERENCE_BETA,
    EPOCHS,
    EXPECTED_COND_HASH,
    EXPECTED_LOCAL_HASH,
    GEN,
    GRAD_ACCUM,
    LAYER,
    LR,
    SEED,
    WEIGHT_DECAY,
    _atomic_json,
    _atomic_torch,
    _decode_controller,
    _git_commit,
    _load_reused_dg04,
    _sha256_file,
    _slurm_metadata,
    controller_hook,
    load_frozen_inputs,
    parameter_report,
    set_seed,
)

CORRECTION_SET_ARTIFACT = REPO / "results/dg05/correction_set_v1.json"
CORRECTION_SET_SHA256 = "sha256:813604876dfb68eb7fc2e8f1864c46f7c57ef25ed4f3c700798c0f88d199a430"
TRAINING_BASELINE = REPO / "results/dg05/controller/training_baseline_hypotheses.json"
RETENTION_ARTIFACT = REPO / "results/dg06/retention_set_v1.json"
LAMBDA_M = 1.0
LAMBDA_E = 1.0


def _controller_state_hash(controller: FixedBasisAdaptiveController) -> str:
    """Deterministic hash of the controller parameters (not the pickle bytes)."""
    h = hashlib.sha256()
    for key, value in sorted(controller.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(
            value.detach().cpu().float().numpy(), dtype=np.float64).tobytes())
    return "sha256:" + h.hexdigest()


def _load_training_baseline(examples: Sequence[Any]) -> dict[str, str]:
    """Reuse the frozen DG-05 free-decoding baseline for the training pool."""
    cached = json.loads(TRAINING_BASELINE.read_text(encoding="utf-8"))
    if (cached.get("schema_version") != "dg05_training_baseline_v1"
            or cached.get("roles") != ["loc-train", "util-train"]
            or cached.get("decode") != GEN
            or not isinstance(cached.get("hypotheses"), dict)):
        raise RuntimeError("frozen DG-05 training baseline is missing or altered")
    texts = {str(k): str(v) for k, v in cached["hypotheses"].items()}
    missing = [str(e.utterance_id) for e in examples if str(e.utterance_id) not in texts]
    if missing:
        raise RuntimeError(f"training baseline does not cover {len(missing)} utterances")
    return texts


def build_retention_sets(bundle: Any, examples: Sequence[Any],
                         baseline_texts: Mapping[str, str], *,
                         output_path: Path, source_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build R_E (baseline-correct EN) and R_M (baseline-correct ZH) positions.

    Uses the same canonical unit alignment / language tagging / reference-token
    overlap as the frozen DG-05 C_E builder, but keeps the baseline-**correct**
    embedded and matrix positions rather than the baseline-wrong embedded ones.
    """
    from csasr.data.alignment import build_prefix, token_char_offsets
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.evaluation.pier import unit_status
    from csasr.steering.dg05_training import LID_EN
    from csasr.steering.dg06_losses import LID_ZH

    tokenizer = bundle.processor.tokenizer
    prefix_width = len(build_prefix(bundle.processor, language="zh", task="transcribe"))
    r_e: dict[str, list[int]] = {}
    r_m: dict[str, list[int]] = {}
    excl_e: list[str] = []
    excl_m: list[str] = []
    n_correct_en = n_correct_zh = 0
    for example in examples:
        uid = str(example.utterance_id)
        norm, units = normalize_and_segment(example.reference)
        text_ids = list(example.token_ids[prefix_width:-1])
        offsets = token_char_offsets(tokenizer, text_ids)
        status = unit_status(example.reference, baseline_texts[uid])
        for unit_idx, unit in enumerate(units):
            tag = tag_unit(unit)
            if tag not in (EN, ZH):
                continue
            ok = bool(status.get(unit_idx, (False, ""))[0])
            if not ok:
                continue                       # retention is over baseline-correct units only
            required = LID_EN if tag == EN else LID_ZH
            hits = [prefix_width + tok_idx for tok_idx, (lo, hi) in enumerate(offsets)
                    if lo < unit.char_end and hi > unit.char_start
                    and (prefix_width + tok_idx) < len(example.lid_token_labels)
                    and int(example.lid_token_labels[prefix_width + tok_idx]) == required]
            if tag == EN:
                n_correct_en += 1
                (r_e.setdefault(uid, []).extend(hits) if hits
                 else excl_e.append(f"{uid}:{unit_idx}:no_token_overlap"))
            else:
                n_correct_zh += 1
                (r_m.setdefault(uid, []).extend(hits) if hits
                 else excl_m.append(f"{uid}:{unit_idx}:no_token_overlap"))
    if not r_m:
        raise RuntimeError("baseline produced an empty R_M (matrix retention) population")
    embedded = retention_set_payload(
        r_e, language="EN",
        source="DG-06 frozen Whisper free-decoding baseline (reused DG-05) on loc-train+util-train",
        roles=("loc-train", "util-train"), exclusions=excl_e)
    matrix = retention_set_payload(
        r_m, language="ZH",
        source="DG-06 frozen Whisper free-decoding baseline (reused DG-05) on loc-train+util-train",
        roles=("loc-train", "util-train"), exclusions=excl_m)
    payload = {
        "schema_version": "dg06_retention_sets_v1",
        "baseline_source": str(TRAINING_BASELINE),
        "baseline_sha256": _sha256_file(TRAINING_BASELINE),
        "baseline_model": bundle.metadata(),
        "baseline_decode": GEN,
        "source_config_hash": sha256_obj(dict(source_config)),
        "construction": "free-decoding baseline + canonical unit_status + reference token overlap",
        "n_baseline_correct_en_units": int(n_correct_en),
        "n_baseline_correct_zh_units": int(n_correct_zh),
        "embedded_retention_R_E": embedded,
        "matrix_retention_R_M": matrix,
    }
    _atomic_json(output_path, payload)
    return payload


def collate_dg06_batch(bundle: Any, examples: Sequence[Any], correction_set,
                       r_e_set: RetentionSetIndex, r_m_set: RetentionSetIndex,
                       *, prefix_width: int, include_embedded: bool) -> dict[str, Any]:
    """Model inputs + shifted C_E / R_M / (R_E) masks, all on the same targets."""
    inputs = batch_model_inputs(bundle, [e.audio_path for e in examples])
    pad_id = bundle.processor.tokenizer.pad_token_id
    width = max(len(e.token_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    lid = torch.full_like(ids, -100)
    for row, example in enumerate(examples):
        n = len(example.token_ids)
        ids[row, :n] = torch.tensor(example.token_ids, dtype=torch.long)
        lid[row, :n] = torch.tensor(example.lid_token_labels, dtype=torch.long)
    uids = [str(e.utterance_id) for e in examples]
    ce_full = _masked_or_empty(correction_target_mask, uids, ids, lid,
                               correction_set, prefix_width, ids)
    rm_full = retention_position_mask(uids, ids, lid, r_m_set, prefix_width=prefix_width)
    re_full = (retention_position_mask(uids, ids, lid, r_e_set, prefix_width=prefix_width)
               if include_embedded else None)
    batch = {
        "input_features": inputs["input_features"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": ids.to(bundle.device),
        "labels": ids[:, 1:].to(bundle.device),
        "ce_mask": ce_full[:, 1:].to(bundle.device),
        "rm_mask": rm_full[:, 1:].to(bundle.device),
        "re_mask": re_full[:, 1:].to(bundle.device) if re_full is not None else None,
    }
    return batch


def _masked_or_empty(fn, uids, ids, lid, correction_set, prefix_width, template):
    """C_E mask that tolerates a batch with no correction position (all-false)."""
    try:
        return fn(uids, ids, lid, correction_set, prefix_width=prefix_width)
    except ValueError as exc:
        if "empty" in str(exc):
            return torch.zeros_like(template, dtype=torch.bool)
        raise


def train_batch_dg06(bundle: Any, controller: FixedBasisAdaptiveController,
                     examples: Sequence[Any], correction_set, r_e_set, r_m_set, *,
                     beta: float, prefix_width: int, include_embedded: bool
                     ) -> tuple[torch.Tensor, dict[str, float]]:
    batch = collate_dg06_batch(bundle, examples, correction_set, r_e_set, r_m_set,
                               prefix_width=prefix_width, include_embedded=include_embedded)
    # Frozen-baseline p_0 under the identical teacher-forced context (no hook).
    with torch.no_grad():
        base_out = bundle.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            use_cache=False)
    baseline_logits = base_out.logits.float()[:, :-1]
    # p_theta with the controller active.
    with controller_hook(bundle, controller, beta=beta, num_forced_prefix=prefix_width):
        out = bundle.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            use_cache=False)
    method_logits = out.logits.float()[:, :-1]
    total, components = damage_aware_loss(
        method_logits, baseline_logits, batch["labels"], batch["ce_mask"],
        batch["rm_mask"], batch["re_mask"] if include_embedded else None,
        lambda_m=LAMBDA_M, lambda_e=LAMBDA_E)
    (total / GRAD_ACCUM).backward()
    return total.detach(), components


def validate_frozen_config(cfg: Mapping[str, Any], variant: str) -> None:
    basis_cfg = cfg.get("basis", {})
    train_cfg = cfg.get("training", {})
    controller_cfg = cfg.get("controller", {})
    data_cfg = cfg.get("data", {})
    if variant not in ("d1", "d2"):
        raise ValueError("DG-06 variant must be d1 or d2")
    if int(basis_cfg.get("layer", -1)) != LAYER:
        raise ValueError("DG-06 only permits the DG-03-selected L24")
    if str(basis_cfg.get("local_hash")) != EXPECTED_LOCAL_HASH or \
            str(basis_cfg.get("cond_hash")) != EXPECTED_COND_HASH:
        raise ValueError("DG-06 basis hashes do not match frozen DG-03 inputs")
    if int(controller_cfg.get("bottleneck", -1)) != BOTTLENECK:
        raise ValueError("DG-06 bottleneck is frozen at 32")
    if float(train_cfg.get("beta", float("nan"))) != DG04_REFERENCE_BETA:
        raise ValueError("DG-06 beta must equal the DG-04 reference beta")
    if list(data_cfg.get("training_roles", [])) != ["loc-train", "util-train"]:
        raise ValueError("DG-06 training roles must remain loc-train and util-train")
    if float(train_cfg.get("lambda_m", float("nan"))) != LAMBDA_M:
        raise ValueError("DG-06 lambda_M is frozen at 1.0")
    if float(train_cfg.get("lambda_e", float("nan"))) != LAMBDA_E:
        raise ValueError("DG-06 lambda_E is frozen at 1.0")
    if bool(train_cfg.get("basis_refinement", True)) or bool(train_cfg.get("gate_penalty", True)):
        raise ValueError("DG-06 core forbids basis refinement and gate penalty")


def _matrix_retention(result: Mapping[str, Any]) -> float:
    rate = (result.get("metrics", {}).get("retention", {})
            .get("matrix_zh", {}).get("rate"))
    return float(rate) if rate is not None else -1.0


def select_checkpoint_dg06(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Utility → PIER gain → matrix retention → lower energy (DG-06 spec §10)."""
    eligible = [r for r in records
                if float(r.get("utility", -float("inf"))) > 0
                and bool(r.get("valid_outside_harm", False))]
    if not eligible:
        return None
    return sorted(eligible, key=lambda r: (
        -float(r["utility"]),
        -float(r.get("pier_gain", -float("inf"))),
        -float(r.get("matrix_retention", -1.0)),
        float(r.get("total_energy", float("inf"))),
    ))[0]


def checkpoint_payload(controller: FixedBasisAdaptiveController, *, epoch: int, variant: str,
                       cfg: Mapping[str, Any], basis_record: Mapping[str, Any],
                       components: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "dg06_damage_aware_v1",
        "variant": variant,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().clone()
                        for k, v in controller.state_dict().items()},
        "architecture": {"d_model": controller.d_model, "bottleneck": controller.bottleneck,
                         "hidden_activation": "GELU", "gate": "sigmoid(output[...,0])",
                         "mixture": "softmax(output[...,1:])", "direction": "normalize(V0 @ pi_t)"},
        "trainable_parameter_count": sum(p.numel() for p in controller.trainable_parameters),
        "basis_hashes": basis_record.get("tensor_hashes"), "basis_layer": LAYER,
        "beta": DG04_REFERENCE_BETA,
        "objective": ("correction CE + matrix KL retention" if variant == "d1"
                      else "correction CE + matrix + embedded KL retention"),
        "lambda_m": LAMBDA_M, "lambda_e": (LAMBDA_E if variant == "d2" else None),
        "training_roles": ["loc-train", "util-train"],
        "epoch_components": dict(components),
        "config_hash": sha256_obj(dict(cfg)), "git_commit": _git_commit(),
    }


def run_training(cfg: Mapping[str, Any], output_dir: Path, variant: str) -> dict[str, Any]:
    started = time.time()
    validate_frozen_config(cfg, variant)
    include_embedded = variant == "d2"
    set_seed(int(cfg.get("training", {}).get("seed", SEED)))
    basis, basis_record = load_frozen_inputs()
    bundle = load_whisper({"model": dict(cfg["model"])})
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("DG-06 requires every Whisper parameter to be frozen")
    if basis.requires_grad:
        raise RuntimeError("DG-06 basis must be a frozen tensor")
    # Reconstruct the exact DG-05/D0 controller init (seed -> load_whisper ->
    # freeze -> construct), with no intervening RNG use.  D1 and D2 must match.
    controller = FixedBasisAdaptiveController(bundle.d_model, basis, BOTTLENECK).to(bundle.device)
    init_hash = _controller_state_hash(controller)
    _atomic_torch(output_dir / "initial_controller.pt",
                  {"schema_version": "dg06_initial_controller_v1", "variant": variant,
                   "state_dict": {k: v.detach().cpu().clone()
                                  for k, v in controller.state_dict().items()},
                   "init_hash": init_hash, "seed": int(cfg["training"]["seed"]),
                   "procedure": "set_seed(42)->load_whisper->freeze->construct (DG-05 D0 procedure)"})
    if init_hash != _controller_state_hash(controller):
        raise RuntimeError("controller init is non-deterministic")
    params = parameter_report(controller, bundle.model)

    data_cfg = load_config(cfg["data"]["candidate_config"])
    from steer_sweep.trackb.data import build_examples
    examples = build_examples(bundle, data_cfg, ("loc-train", "util-train"))
    if not examples:
        raise RuntimeError("DG-06 training pool is empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    # C_E: reuse the frozen DG-05 artifact verbatim (identity checked by hash).
    if _sha256_file(CORRECTION_SET_ARTIFACT) != CORRECTION_SET_SHA256:
        raise RuntimeError("frozen DG-05 C_E artifact hash mismatch")
    correction_set = load_correction_set(CORRECTION_SET_ARTIFACT)

    # R_E / R_M: build once from the reused frozen baseline, then freeze.
    baseline_texts = _load_training_baseline(examples)
    RETENTION_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    if RETENTION_ARTIFACT.exists():
        ret_payload = json.loads(RETENTION_ARTIFACT.read_text(encoding="utf-8"))
        if ret_payload.get("schema_version") != "dg06_retention_sets_v1":
            ret_payload = build_retention_sets(bundle, examples, baseline_texts,
                                               output_path=RETENTION_ARTIFACT, source_config=cfg)
    else:
        ret_payload = build_retention_sets(bundle, examples, baseline_texts,
                                           output_path=RETENTION_ARTIFACT, source_config=cfg)
    r_e_set = load_retention_set(ret_payload["embedded_retention_R_E"])
    r_m_set = load_retention_set(ret_payload["matrix_retention_R_M"])

    prefix_width = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    optimizer = torch.optim.AdamW(controller.trainable_parameters, lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    base_prov = baseline_provenance(
        model_id=bundle.model_id, model_revision=bundle.revision,
        training_role_fingerprint=ret_payload["baseline_sha256"],
        config_hash=sha256_obj(dict(cfg)),
        tokenizer=type(bundle.processor.tokenizer).__name__, git_commit=_git_commit())
    manifest = stage_provenance(dict(cfg), f"dg06_damage_aware_{variant}")
    manifest.update({
        "variant": variant, "git_commit": _git_commit(), "basis_record": basis_record,
        "basis_layer": LAYER, "beta": DG04_REFERENCE_BETA,
        "training_roles": ["loc-train", "util-train"],
        "controller_init_hash": init_hash,
        "correction_set": {"artifact": str(CORRECTION_SET_ARTIFACT),
                           "sha256": CORRECTION_SET_SHA256,
                           "n_utterances": len(correction_set.positions),
                           "n_positions": correction_set.n_positions},
        "retention_sets": {"artifact": str(RETENTION_ARTIFACT),
                           "sha256": _sha256_file(RETENTION_ARTIFACT),
                           "R_E": {"n_utterances": ret_payload["embedded_retention_R_E"]["n_utterances"],
                                   "n_positions": ret_payload["embedded_retention_R_E"]["n_positions"]},
                           "R_M": {"n_utterances": ret_payload["matrix_retention_R_M"]["n_utterances"],
                                   "n_positions": ret_payload["matrix_retention_R_M"]["n_positions"]}},
        "lambda_m": LAMBDA_M, "lambda_e": (LAMBDA_E if include_embedded else None),
        "objective": ("L_corr + lambda_M L_ret,M" if variant == "d1"
                      else "L_corr + lambda_M L_ret,M + lambda_E L_ret,E"),
        "baseline_distribution": base_prov,
        "parameter_report": params, "whisper_frozen": True, "basis_frozen": True,
        "exact_site": "decoder post-cross-attention residual, pre-FFN",
        "retention_losses": {"matrix": True, "embedded": include_embedded,
                             "anchor": False, "gate_penalty": False},
        "slurm": _slurm_metadata(), "gpu_run": True,
    })
    _atomic_json(output_dir / "manifest.json", manifest)

    # Full-pool batches carrying any contributing position for this variant.
    def batch_has_target(batch_examples: Sequence[Any]) -> bool:
        for e in batch_examples:
            uid = str(e.utterance_id)
            if correction_set.positions.get(uid) or r_m_set.positions.get(uid):
                return True
            if include_embedded and r_e_set.positions.get(uid):
                return True
        return False

    batches = [examples[i:i + BATCH_SIZE]
               for i in range(0, len(examples), BATCH_SIZE)
               if batch_has_target(examples[i:i + BATCH_SIZE])]
    history = []
    last_components: dict[str, Any] = {}
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        agg = {"l_corr": 0.0, "l_ret_m": 0.0, "l_ret_e": 0.0, "total": 0.0}
        counts = {"n_corr": 0, "n_ret_m": 0, "n_ret_e": 0}
        n_steps = 0
        grad_norms: list[float] = []
        for index, batch_examples in enumerate(batches):
            _loss, comp = train_batch_dg06(
                bundle, controller, batch_examples, correction_set, r_e_set, r_m_set,
                beta=DG04_REFERENCE_BETA, prefix_width=prefix_width,
                include_embedded=include_embedded)
            agg["l_corr"] += comp["l_corr"] * comp["n_corr"]
            agg["l_ret_m"] += comp["l_ret_m"] * comp["n_ret_m"]
            agg["total"] += comp["total"]
            counts["n_corr"] += comp["n_corr"]
            counts["n_ret_m"] += comp["n_ret_m"]
            if include_embedded:
                agg["l_ret_e"] += comp.get("l_ret_e", 0.0) * comp.get("n_ret_e", 0)
                counts["n_ret_e"] += comp.get("n_ret_e", 0)
            n_steps += 1
            if (index + 1) % GRAD_ACCUM == 0 or index + 1 == len(batches):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    controller.trainable_parameters, CLIP_NORM)
                grad_norms.append(float(grad_norm.detach().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        components = {
            "raw_l_corr": agg["l_corr"] / max(1, counts["n_corr"]),
            "raw_l_ret_m": agg["l_ret_m"] / max(1, counts["n_ret_m"]),
            "weighted_l_ret_m": LAMBDA_M * agg["l_ret_m"] / max(1, counts["n_ret_m"]),
            "mean_total_loss": agg["total"] / max(1, n_steps),
            "n_C_E": counts["n_corr"], "n_R_M": counts["n_ret_m"],
            "n_batches": n_steps, "n_optimizer_steps": len(grad_norms),
        }
        if include_embedded:
            components["raw_l_ret_e"] = agg["l_ret_e"] / max(1, counts["n_ret_e"])
            components["weighted_l_ret_e"] = LAMBDA_E * agg["l_ret_e"] / max(1, counts["n_ret_e"])
            components["n_R_E"] = counts["n_ret_e"]
        last_components = components
        record = {
            "epoch": epoch + 1, "variant": variant, **components,
            "gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "controller_parameter_norms": {
                name: float(param.detach().float().norm().cpu())
                for name, param in controller.named_parameters()},
            "runtime_sec": float(time.time() - epoch_started),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else None,
        }
        history.append(record)
        _atomic_torch(output_dir / f"checkpoint_epoch{epoch + 1}.pt",
                      checkpoint_payload(controller, epoch=epoch + 1, variant=variant, cfg=cfg,
                                         basis_record=basis_record, components=components))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    _atomic_json(output_dir / "training_history.json", {"records": history})

    # Free-decoding checkpoint evaluation on D-dev-select (reuse DG-04/DG-05 path).
    from steer_sweep import data as D
    dcfg = load_config(cfg["data"]["candidate_config"])
    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    a0 = _load_reused_dg04("B0", ids)
    a1 = _load_reused_dg04("B1_rho0.5", ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    from experiments.dg04_frozen_baselines import _build_outside_sets, _result_v1
    out_sets, out_langs, out_diag = _build_outside_sets(dcfg, refs, ids)
    provenance = {
        "git_commit": _git_commit(), "model_id": bundle.model_id,
        "model_revision": bundle.revision, "layer": LAYER,
        "basis_artifact_hashes": basis_record.get("tensor_hashes"),
        "dataset_fingerprint": basis_record.get("provenance", {}).get("dataset_fingerprint"),
        "config_hash": sha256_obj(dict(cfg)), "decode": GEN, "data_role": "D-dev-select",
        "slurm": _slurm_metadata(), "training_roles": ["loc-train", "util-train"],
        "variant": variant, "lambda_m": LAMBDA_M,
        "lambda_e": (LAMBDA_E if include_embedded else None),
        "controller_init_hash": init_hash,
        "selection_rule": "positive utility, then PIER gain, then matrix retention, then lower energy",
    }
    direction_type = ("adaptive_controller_correction_matrix_retention" if variant == "d1"
                      else "adaptive_controller_full_damage_aware")
    eval_records: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        payload = torch.load(output_dir / f"checkpoint_epoch{epoch}.pt",
                             map_location=bundle.device, weights_only=False)
        controller.load_state_dict(payload["state_dict"], strict=True)
        texts, audit = _decode_controller(bundle, pop, controller, nfp=nfp)
        result = _result_v1(
            refs, a0["texts"], texts, ids, cond=f"{variant.upper()}_epoch{epoch}",
            provenance=provenance, direction_type=direction_type,
            steering_strength=DG04_REFERENCE_BETA, basis_id=EXPECTED_LOCAL_HASH,
            out_sets=out_sets, out_langs=out_langs, audit=audit)
        result["run_id"] = result["run_id"].replace("dg04/", "dg06/", 1)
        result["system_name"] = result["system_name"].replace("dg04_", "dg06_", 1)
        result["method"]["gate_type"] = "adaptive_controller"
        transitions = result["metrics"]["transitions"]
        evaluation = {
            "epoch": epoch, "variant": variant,
            "checkpoint": str(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "checkpoint_sha256": _sha256_file(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "result_v1": result, "texts": texts,
            "utility": float(transitions["net_corrections"]),
            "pier_gain": float(result["metrics"].get("pier_gain", 0.0)),
            "matrix_retention": _matrix_retention(result),
            "valid_outside_harm": result["metrics"].get("outside_harm") is not None,
            "total_energy": float(audit.get("total_energy", 0.0)),
            "audit": audit,
        }
        eval_records.append(evaluation)
        _atomic_json(output_dir / f"evaluation_epoch{epoch}.json", evaluation)
    selected = select_checkpoint_dg06(eval_records)
    selection = {
        "rule": "positive canonical utility then PIER gain then higher matrix retention then lower energy",
        "variant": variant,
        "selected_epoch": selected.get("epoch") if selected else None,
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "eligible_epochs": [int(x["epoch"]) for x in eval_records
                            if float(x["utility"]) > 0 and bool(x["valid_outside_harm"])],
        "population": {"role": "D-dev-select", "n_utterances": len(ids), "outside_harm": out_diag},
    }
    if selected:
        import shutil
        shutil.copy2(selected["checkpoint"], output_dir / "selected_checkpoint.pt")
        selection["selected_checkpoint_sha256"] = _sha256_file(output_dir / "selected_checkpoint.pt")
    _atomic_json(output_dir / "selection.json", selection)
    summary = {
        "schema_version": "dg06_run_v1", "variant": variant, "seed": int(cfg["training"]["seed"]),
        "controller_init_hash": init_hash, "manifest": manifest, "history": history,
        "final_components": last_components,
        "trainable_parameters": params["controller_trainable"],
        "a0": a0["result_v1"], "a1": a1["result_v1"],
        "evaluations": eval_records, "selection": selection,
        "runtime_sec": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else None,
        "artifacts": {"correction_set": str(CORRECTION_SET_ARTIFACT),
                      "retention_sets": str(RETENTION_ARTIFACT),
                      "initial_controller": str(output_dir / "initial_controller.pt"),
                      "selected_checkpoint": str(output_dir / "selected_checkpoint.pt")
                      if selected else None},
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dg06_damage_aware.yaml")
    parser.add_argument("--variant", choices=["d1", "d2"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="override training.seed (DG-08 multi-seed); default keeps the config seed")
    parser.add_argument("--run", action="store_true",
                        help="run damage-aware optimization; never implied by import")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        # DG-08 varies only the training seed; all other frozen config is untouched.
        cfg = dict(cfg)
        cfg["training"] = {**cfg.get("training", {}), "seed": int(args.seed)}
    validate_frozen_config(cfg, args.variant)
    output_dir = Path(args.output_dir) if args.output_dir else REPO / "results/dg06" / args.variant
    if not args.run:
        basis, _ = load_frozen_inputs()
        print(json.dumps({"ready": True, "variant": args.variant, "layer": LAYER,
                          "basis_shape": list(basis.shape), "beta": DG04_REFERENCE_BETA,
                          "lambda_m": LAMBDA_M,
                          "lambda_e": LAMBDA_E if args.variant == "d2" else None,
                          "gpu_run": False}, indent=2))
        return 0
    run_training(cfg, output_dir, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
