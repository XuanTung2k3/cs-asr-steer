#!/usr/bin/env python
"""DG-05A correction-only adaptive-controller training path.

This entry point is prepared for a later Slurm run.  It deliberately does not
perform free-decoding selection during optimization and it has no KL,
refinement, SALSA, or LoRA path.  Checkpoints are selected later by the frozen
free-decoding rule in ``DG05_ADAPTIVE_CONTROLLER_SPEC.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
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
from csasr.steering.controller import (
    CONTROLLER_SCHEMA_VERSION,
    FixedBasisAdaptiveController,
    load_frozen_basis,
)
from csasr.steering.dg05_training import (
    CorrectionSetIndex,
    correction_only_loss,
    correction_target_mask,
    load_correction_set,
)
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj
from csasr.utils.provenance import stage_provenance


LAYER = 24
BASIS_PATH = REPO / "results/dg03/basis/steering_basis_v1_L24.json"
EXPECTED_LOCAL_HASH = "sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93"
EXPECTED_COND_HASH = "sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991"
DG04_REFERENCE_BETA = 4.465628877080159
BOTTLENECK = 32
SEED = 42
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 3
LR = 5e-4
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
GEN = dict(task="transcribe", language="zh", do_sample=False,
           num_beams=1, temperature=0.0, max_new_tokens=200,
           condition_on_prev_tokens=False)


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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slurm_metadata() -> dict[str, Any]:
    return {
        "job_ids": ([str(os.environ["SLURM_JOB_ID"])]
                     if os.environ.get("SLURM_JOB_ID") else []),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _decode_baseline_training_pool(bundle: Any, examples: Sequence[Any], *,
                                   batch_size: int = BATCH_SIZE) -> dict[str, str]:
    """Free-decode the training roles to define the frozen-baseline C_E set."""
    texts: dict[str, str] = {}
    from csasr.models.whisper import batch_model_inputs
    with torch.inference_mode():
        for start in range(0, len(examples), int(batch_size)):
            batch = list(examples[start:start + int(batch_size)])
            inputs = batch_model_inputs(bundle, [e.audio_path for e in batch])
            seq = bundle.model.generate(**inputs, **GEN)
            if not isinstance(seq, torch.Tensor):
                seq = seq.sequences
            decoded = bundle.processor.batch_decode(seq, skip_special_tokens=True)
            for example, text in zip(batch, decoded):
                texts[str(example.utterance_id)] = str(text).strip()
    if len(texts) != len(examples):
        raise RuntimeError("training baseline decode did not cover every utterance")
    return texts


def _build_training_correction_set(bundle: Any, examples: Sequence[Any],
                                   baseline_texts: Mapping[str, str],
                                   *, output_path: Path,
                                   source_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the explicit token-position C_E artifact from free-decoded baseline text.

    The baseline correctness is computed with the canonical unit alignment used by
    PIER/MER.  Only baseline-wrong embedded-language units are mapped to reference
    decoder-token positions; no oracle feature is passed to the controller.
    """
    from csasr.data.alignment import token_char_offsets
    from csasr.data.language_tags import EN, tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.evaluation.pier import unit_status
    from csasr.steering.dg05_training import correction_set_payload

    positions: dict[str, list[int]] = {}
    exclusions: list[str] = []
    tokenizer = bundle.processor.tokenizer
    prefix_width = len(__import__("csasr.data.alignment", fromlist=["build_prefix"])
                       .build_prefix(bundle.processor, language="zh", task="transcribe"))
    n_wrong_units = 0
    for example in examples:
        uid = str(example.utterance_id)
        norm, units = normalize_and_segment(example.reference)
        text_ids = list(example.token_ids[prefix_width:-1])
        offsets = token_char_offsets(tokenizer, text_ids)
        status = unit_status(example.reference, baseline_texts[uid])
        for unit_idx, unit in enumerate(units):
            if tag_unit(unit) != EN:
                continue
            ok = bool(status.get(unit_idx, (False, ""))[0])
            if ok:
                continue
            n_wrong_units += 1
            hits = [prefix_width + tok_idx for tok_idx, (lo, hi) in enumerate(offsets)
                    if lo < unit.char_end and hi > unit.char_start]
            if not hits:
                exclusions.append(f"{uid}:{unit_idx}:no_token_overlap")
                continue
            positions.setdefault(uid, []).extend(hits)
    if not positions:
        raise RuntimeError("free-decoded training baseline produced an empty C_E")
    payload = correction_set_payload(
        positions,
        source="DG-05B frozen Whisper free-decoding baseline on loc-train+util-train",
        roles=("loc-train", "util-train"), exclusions=exclusions)
    payload.update({
        "baseline_model": bundle.metadata(),
        "baseline_decode": GEN,
        "baseline_hypotheses_sha256": "sha256:" + hashlib.sha256(
            json.dumps(dict(sorted(baseline_texts.items())), sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_config_hash": sha256_obj(dict(source_config)),
        "n_baseline_wrong_embedded_units": int(n_wrong_units),
        "n_positions_before_dedup": int(sum(len(v) for v in positions.values())),
        "exclusions_count": len(exclusions),
        "construction": "free-decoding baseline + canonical unit_status + reference token overlap",
    })
    _atomic_json(output_path, payload)
    return payload


def _load_reused_dg04(name: str, ids: Sequence[str]) -> dict[str, Any]:
    """Reuse a DG-04 result only when it is the identical D-dev-select population."""
    path = REPO / "results/dg04/results" / f"{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    got = list(payload.get("texts", {}))
    if set(got) != set(map(str, ids)):
        raise RuntimeError(f"DG-04 {name} population does not match D-dev-select")
    if payload.get("result_v1", {}).get("data_role") not in (None, "D-dev-select"):
        raise RuntimeError(f"DG-04 {name} is not a D-dev-select result")
    return payload


def _decode_controller(bundle: Any, pop: Any, controller: FixedBasisAdaptiveController,
                       *, nfp: int, batch_size: int = BATCH_SIZE) -> tuple[dict[str, str], dict[str, Any]]:
    """Free-decode one controller checkpoint and collect gate/mixture diagnostics."""
    from csasr.models.whisper import batch_model_inputs
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts: dict[str, str] = {}
    gate_values: list[np.ndarray] = []
    mixture_values: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    energies: list[float] = []
    controller.eval()
    for start in range(0, len(manifest), int(batch_size)):
        batch = manifest.iloc[start:start + int(batch_size)]
        ids = [str(x) for x in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        def action_fn(*, r, abs_pos, **_):
            gate, pi, direction = controller(r)
            eligible = (abs_pos >= int(nfp)).view(1, -1)
            gate_values.append(gate.detach().float()[eligible.expand_as(gate)].cpu().numpy())
            mixture_values.append(pi.detach().float()[eligible.expand_as(pi[..., 0])].cpu().numpy())
            directions.append(direction.detach().float()[eligible.expand_as(direction[..., 0])].cpu().numpy())
            return gate, direction
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, LAYER, direction=None, alpha=DG04_REFERENCE_BETA, scale=1.0,
            action_fn=action_fn, num_forced_prefix=nfp, norm_preserve=True,
            mode="steer", record=True, record_last_only=False,
            enforce_contract_layer=True)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inputs, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        decoded = bundle.processor.batch_decode(seq, skip_special_tokens=True)
        for uid, text in zip(ids, decoded):
            texts[uid] = str(text).strip()
        energies.extend(float(rec.edit_norm) for rec in hook.records if rec.steered)
    g = np.concatenate(gate_values) if gate_values else np.zeros(0, dtype=np.float64)
    pi = np.concatenate(mixture_values) if mixture_values else np.zeros((0, 2), dtype=np.float64)
    d = np.concatenate(directions) if directions else np.zeros((0, 1), dtype=np.float64)
    audit: dict[str, Any] = {
        "n_steered": len(energies),
        "total_energy": float(np.sum(energies)) if energies else 0.0,
        "mean_energy": float(np.mean(energies)) if energies else 0.0,
        "n_gate_eligible": int(g.size),
    }
    if g.size:
        audit["gate_strength"] = {
            "mean": float(g.mean()), "median": float(np.median(g)),
            "q01": float(np.quantile(g, .01)), "q10": float(np.quantile(g, .10)),
            "q25": float(np.quantile(g, .25)), "q75": float(np.quantile(g, .75)),
            "q90": float(np.quantile(g, .90)), "q99": float(np.quantile(g, .99)),
            "near_zero_fraction": float(np.mean(g <= 0.05)),
            "near_one_fraction": float(np.mean(g >= 0.95)),
        }
    if pi.size:
        audit["mixture"] = {
            "mean": [float(x) for x in pi.mean(axis=0)],
            "variance": [float(x) for x in pi.var(axis=0)],
            "median": [float(x) for x in np.median(pi, axis=0)],
        }
    if d.size:
        # Direction diversity is the mean pairwise cosine to a deterministic
        # sample; this remains descriptive and does not redefine coverage.
        sample = d[::max(1, len(d) // 2048)]
        normed = sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
        audit["direction_diversity"] = {
            "n_sample": int(len(normed)),
            "mean_pairwise_cosine": float(
                (normed @ normed.T)[np.triu_indices(len(normed), 1)].mean())
                if len(normed) > 1 else 1.0,
        }
    return texts, audit


def load_frozen_inputs() -> tuple[torch.Tensor, dict[str, Any]]:
    """Load exactly the DG-04-selected L24 basis and its provenance."""
    return load_frozen_basis(
        BASIS_PATH, expected_layer=LAYER,
        expected_local_hash=EXPECTED_LOCAL_HASH,
        expected_cond_hash=EXPECTED_COND_HASH,
    )


def validate_frozen_config(cfg: Mapping[str, Any]) -> None:
    """Reject silent changes to the DG-05A pre-run choices."""
    basis_cfg = cfg.get("basis", {})
    train_cfg = cfg.get("training", {})
    controller_cfg = cfg.get("controller", {})
    data_cfg = cfg.get("data", {})
    if int(basis_cfg.get("layer", -1)) != LAYER:
        raise ValueError("DG-05A only permits the DG-04-selected L24")
    if str(basis_cfg.get("local_hash")) != EXPECTED_LOCAL_HASH or \
            str(basis_cfg.get("cond_hash")) != EXPECTED_COND_HASH:
        raise ValueError("DG-05A basis hashes do not match frozen DG-03 inputs")
    if int(controller_cfg.get("bottleneck", -1)) != BOTTLENECK:
        raise ValueError("DG-05A bottleneck is frozen at 32")
    if float(train_cfg.get("beta", float("nan"))) != DG04_REFERENCE_BETA:
        raise ValueError("DG-05A beta must equal the DG-04 reference beta")
    if list(data_cfg.get("training_roles", [])) != ["loc-train", "util-train"]:
        raise ValueError("DG-05A training roles must remain loc-train and util-train")
    if bool(train_cfg.get("retention_kl", True)) or bool(train_cfg.get("basis_refinement", True)):
        raise ValueError("DG-05A cannot enable DG-06 retention or basis refinement")


def collate_controller_batch(bundle: Any, examples: Sequence[Any],
                             correction_set: CorrectionSetIndex,
                             *, prefix_width: int) -> dict[str, torch.Tensor | list[Any]]:
    """Build model inputs and the shifted C_E-only loss mask."""
    inputs = batch_model_inputs(bundle, [e.audio_path for e in examples])
    pad_id = bundle.processor.tokenizer.pad_token_id
    width = max(len(e.token_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    lid = torch.full_like(ids, -100)
    for row, example in enumerate(examples):
        n = len(example.token_ids)
        ids[row, :n] = torch.tensor(example.token_ids, dtype=torch.long)
        lid[row, :n] = torch.tensor(example.lid_token_labels, dtype=torch.long)
    full_mask = correction_target_mask(
        [e.utterance_id for e in examples], ids, lid, correction_set,
        prefix_width=prefix_width)
    labels = ids[:, 1:].to(bundle.device)
    target_mask = full_mask[:, 1:].to(bundle.device)
    return {
        "input_features": inputs["input_features"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": ids.to(bundle.device),
        "labels": labels,
        "target_mask": target_mask,
        "examples": list(examples),
    }


def controller_hook(bundle: Any, controller: FixedBasisAdaptiveController,
                    *, beta: float, num_forced_prefix: int):
    """Construct the exact-site train-mode hook; the action reads only r."""
    return DecoderPostCrossAttnInterventionHook(
        bundle, LAYER, direction=None, alpha=float(beta), scale=1.0,
        action_fn=controller.action, num_forced_prefix=num_forced_prefix,
        norm_preserve=True, mode="train", record=False,
        enforce_contract_layer=True,
    )


def parameter_report(controller: FixedBasisAdaptiveController, backbone: Any) -> dict[str, Any]:
    """Report exact controller size and its fraction of the frozen backbone."""
    controller_n = int(sum(p.numel() for p in controller.trainable_parameters))
    backbone_n = int(sum(p.numel() for p in backbone.parameters()))
    return {
        "controller_trainable": controller_n,
        "whisper_total": backbone_n,
        "controller_percent_of_whisper": 100.0 * controller_n / max(1, backbone_n),
    }


def train_batch(bundle: Any, controller: FixedBasisAdaptiveController,
                optimizer: torch.optim.Optimizer, examples: Sequence[Any],
                correction_set: CorrectionSetIndex, *, beta: float,
                prefix_width: int) -> tuple[torch.Tensor, int]:
    batch = collate_controller_batch(bundle, examples, correction_set,
                                     prefix_width=prefix_width)
    with controller_hook(bundle, controller, beta=beta,
                         num_forced_prefix=prefix_width):
        output = bundle.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            use_cache=False,
        )
    logits = output.logits.float()[:, :-1]
    loss = correction_only_loss(logits, batch["labels"], batch["target_mask"])
    (loss / GRAD_ACCUM).backward()
    n_targets = int(batch["target_mask"].sum().item())
    return loss.detach(), n_targets


def checkpoint_payload(controller: FixedBasisAdaptiveController, *, epoch: int,
                       cfg: Mapping[str, Any], basis_record: Mapping[str, Any],
                       train_targets: int) -> dict[str, Any]:
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().clone()
                        for k, v in controller.state_dict().items()},
        "architecture": {
            "d_model": controller.d_model,
            "bottleneck": controller.bottleneck,
            "hidden_activation": "GELU",
            "gate": "sigmoid(output[...,0])",
            "mixture": "softmax(output[...,1:])",
            "direction": "normalize(V0 @ pi_t)",
        },
        "trainable_parameter_count": sum(p.numel() for p in controller.trainable_parameters),
        "basis_hashes": basis_record.get("tensor_hashes"),
        "basis_layer": LAYER,
        "beta": DG04_REFERENCE_BETA,
        "objective": "correction-only CE on C_E",
        "training_roles": ["loc-train", "util-train"],
        "train_targets": int(train_targets),
        "config_hash": sha256_obj(dict(cfg)),
        "git_commit": _git_commit(),
    }


def select_checkpoint(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Apply the frozen free-decoding checkpoint rule to evaluated records."""
    eligible = [r for r in records
                if float(r.get("utility", -float("inf"))) > 0
                and bool(r.get("valid_outside_harm", False))]
    if not eligible:
        return None
    return sorted(eligible, key=lambda r: (
        -float(r["utility"]),
        -float(r.get("pier_gain", -float("inf"))),
        float(r.get("total_energy", float("inf"))),
    ))[0]


def run_training(cfg: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run correction-only optimization when explicitly invoked by Slurm."""
    started = time.time()
    validate_frozen_config(cfg)
    set_seed(int(cfg.get("training", {}).get("seed", SEED)))
    basis, basis_record = load_frozen_inputs()
    model_cfg = dict(cfg["model"])
    bundle = load_whisper({"model": model_cfg})
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("DG-05 requires every Whisper parameter to be frozen")
    if basis.requires_grad:
        raise RuntimeError("DG-05 basis must be a frozen tensor")
    controller = FixedBasisAdaptiveController(bundle.d_model, basis, BOTTLENECK).to(bundle.device)
    params = parameter_report(controller, bundle.model)
    data_cfg = load_config(cfg["data"]["candidate_config"])
    from steer_sweep.trackb.data import build_examples
    examples = build_examples(bundle, data_cfg, ("loc-train", "util-train"))
    if not examples:
        raise RuntimeError("DG-05 training pool is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    correction_path = REPO / cfg["data"]["correction_set_artifact"]
    correction_path.parent.mkdir(parents=True, exist_ok=True)
    # The training-role baseline was not part of DG-04's frozen D-dev-select
    # artifact.  Generate it once with frozen Whisper, then freeze/hash C_E
    # before any controller gradient is computed.
    baseline_path = output_dir / "training_baseline_hypotheses.json"
    baseline_texts = _decode_baseline_training_pool(bundle, examples)
    _atomic_json(baseline_path, {
        "schema_version": "dg05_training_baseline_v1",
        "roles": ["loc-train", "util-train"],
        "decode": GEN,
        "hypotheses": dict(sorted(baseline_texts.items())),
        "sha256": "sha256:" + hashlib.sha256(
            json.dumps(dict(sorted(baseline_texts.items())), sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })
    c_payload = _build_training_correction_set(
        bundle, examples, baseline_texts, output_path=correction_path,
        source_config=cfg)
    correction_set = load_correction_set(correction_path)
    if correction_set.n_positions <= 0:
        raise RuntimeError("validated C_E population is empty")
    prefix_width = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    optimizer = torch.optim.AdamW(controller.trainable_parameters, lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    manifest = stage_provenance(dict(cfg), "dg05_adaptive_controller")
    manifest.update({
        "git_commit": _git_commit(), "basis_record": basis_record,
        "basis_layer": LAYER, "beta": DG04_REFERENCE_BETA,
        "training_roles": ["loc-train", "util-train"],
        "correction_set": {"schema_version": correction_set.schema_version,
                            "source": correction_set.source,
                            "roles": list(correction_set.roles),
                            "n_utterances": len(correction_set.positions),
                            "n_positions": correction_set.n_positions,
                            "artifact": str(correction_path),
                            "artifact_sha256": _sha256_file(correction_path),
                            "n_baseline_wrong_embedded_units": c_payload[
                                "n_baseline_wrong_embedded_units"]},
        "objective": "correction-only CE on C_E",
        "parameter_report": params,
        "whisper_frozen": True,
        "basis_frozen": True,
        "exact_site": "decoder post-cross-attention residual, pre-FFN",
        "retention_losses": {"embedded": False, "matrix": False,
                              "anchor": False, "gate_penalty": False},
        "slurm": _slurm_metadata(),
        "gpu_run": True,
    })
    _atomic_json(output_dir / "manifest.json", manifest)
    history = []
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_targets = 0
        grad_norms: list[float] = []
        batches = [examples[i:i + BATCH_SIZE]
                   for i in range(0, len(examples), BATCH_SIZE)]
        for index, batch_examples in enumerate(batches):
            loss, n_targets = train_batch(
                bundle, controller, optimizer, batch_examples, correction_set,
                beta=DG04_REFERENCE_BETA, prefix_width=prefix_width)
            total_loss += float(loss) * n_targets
            total_targets += n_targets
            if (index + 1) % GRAD_ACCUM == 0 or index + 1 == len(batches):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    controller.trainable_parameters, CLIP_NORM)
                grad_norms.append(float(grad_norm.detach().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = {
            "epoch": epoch + 1,
            "correction_ce": total_loss / max(1, total_targets),
            "n_C_E": total_targets,
            "selection": "pending free-decoding D-dev-select",
            "controller_parameter_norms": {
                name: float(param.detach().float().norm().cpu())
                for name, param in controller.named_parameters()},
            "gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "runtime_sec": float(time.time() - epoch_started),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else None,
        }
        history.append(record)
        _atomic_torch(output_dir / f"checkpoint_epoch{epoch + 1}.pt",
                       checkpoint_payload(controller, epoch=epoch + 1, cfg=cfg,
                                          basis_record=basis_record,
                                          train_targets=total_targets))
        # Reset the peak counter before the next checkpoint's free decode.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    _atomic_json(output_dir / "training_history.json", {"records": history})

    # The exact free-decoding selection population is fixed by DG-04.  A0 and
    # A1 are reused only after an ID/role check; no confirm/test rows are read.
    from steer_sweep import data as D
    from csasr.lss.sites import num_forced_prefix_from
    dcfg = load_config(cfg["data"]["candidate_config"])
    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    a0 = _load_reused_dg04("B0", ids)
    a1 = _load_reused_dg04("B1_rho0.5", ids)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")

    # Import the canonical DG-04 scoring helper.  It only scores supplied text;
    # the A2 text is generated by the DG-05 exact-site controller hook below.
    from experiments.dg04_frozen_baselines import _build_outside_sets, _result_v1
    out_sets, out_langs, out_diag = _build_outside_sets(dcfg, refs, ids)
    provenance = {
        "git_commit": _git_commit(), "model_id": bundle.model_id,
        "model_revision": bundle.revision, "layer": LAYER,
        "basis_artifact_hashes": basis_record.get("tensor_hashes"),
        "dataset_fingerprint": basis_record.get("provenance", {}).get("dataset_fingerprint"),
        "basis_construction_config_hash": basis_record.get("provenance", {}).get(
            "construction_config_hash"),
        "config_hash": sha256_obj(dict(cfg)),
        "decode": GEN, "data_role": "D-dev-select", "slurm": _slurm_metadata(),
        "training_roles": ["loc-train", "util-train"],
        "selection_rule": "positive canonical net correction utility, then PIER gain, then lower energy",
    }
    eval_records: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        payload = torch.load(output_dir / f"checkpoint_epoch{epoch}.pt",
                             map_location=bundle.device, weights_only=False)
        controller.load_state_dict(payload["state_dict"], strict=True)
        texts, audit = _decode_controller(bundle, pop, controller, nfp=nfp)
        result = _result_v1(
            refs, a0["texts"], texts, ids,
            cond=f"A2_epoch{epoch}", provenance=provenance,
            direction_type="adaptive_controller_correction_only",
            steering_strength=DG04_REFERENCE_BETA,
            basis_id=EXPECTED_LOCAL_HASH, out_sets=out_sets,
            out_langs=out_langs, audit=audit)
        transitions = result["metrics"]["transitions"]
        evaluation = {
            "epoch": epoch, "checkpoint": str(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "checkpoint_sha256": _sha256_file(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "result_v1": result,
            "texts": texts,
            "utility": float(transitions["net_corrections"]),
            "pier_gain": float(result["metrics"].get("pier_gain", 0.0)),
            "valid_outside_harm": result["metrics"].get("outside_harm") is not None,
            "total_energy": float(audit.get("total_energy", 0.0)),
            "audit": audit,
        }
        eval_records.append(evaluation)
        _atomic_json(output_dir / f"evaluation_epoch{epoch}.json", evaluation)
    selected = select_checkpoint(eval_records)
    selection = {
        "rule": "positive canonical net correction utility then PIER gain then lower realized intervention energy",
        "selected_epoch": selected.get("epoch") if selected else None,
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "eligible_epochs": [int(x["epoch"]) for x in eval_records
                             if float(x["utility"]) > 0 and bool(x["valid_outside_harm"])],
        "population": {"role": "D-dev-select", "n_utterances": len(ids),
                        "outside_harm": out_diag},
    }
    _atomic_json(output_dir / "selection.json", selection)
    if selected:
        import shutil
        shutil.copy2(selected["checkpoint"], output_dir / "selected_checkpoint.pt")
        selection["selected_checkpoint_sha256"] = _sha256_file(output_dir / "selected_checkpoint.pt")
        _atomic_json(output_dir / "selection.json", selection)
    summary = {
        "schema_version": "dg05b_run_v1", "seed": int(cfg["training"]["seed"]),
        "manifest": manifest, "history": history,
        "trainable_parameters": params["controller_trainable"],
        "a0": a0["result_v1"], "a1": a1["result_v1"],
        "a2_evaluations": eval_records, "selection": selection,
        "runtime_sec": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else None,
        "artifacts": {"correction_set": str(correction_path),
                      "training_baseline": str(baseline_path),
                      "selected_checkpoint": str(output_dir / "selected_checkpoint.pt")
                      if selected else None},
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dg05_adaptive_controller.yaml")
    parser.add_argument("--output-dir", default="results/dg05/controller")
    parser.add_argument("--run", action="store_true",
                        help="run correction-only optimization; never implied by import")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    validate_frozen_config(cfg)
    if not args.run:
        basis, _ = load_frozen_inputs()
        print(json.dumps({"ready": True, "layer": LAYER,
                          "basis_shape": list(basis.shape),
                          "beta": DG04_REFERENCE_BETA,
                          "gpu_run": False}, indent=2))
        return 0
    run_training(cfg, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
