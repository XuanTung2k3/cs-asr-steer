#!/usr/bin/env python
"""DG-05A correction-only adaptive-controller training path.

This entry point is prepared for a later Slurm run.  It deliberately does not
perform free-decoding selection during optimization and it has no KL,
refinement, SALSA, or LoRA path.  Checkpoints are selected later by the frozen
free-decoding rule in ``DG05_ADAPTIVE_CONTROLLER_SPEC.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
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
    validate_frozen_config(cfg)
    set_seed(int(cfg.get("training", {}).get("seed", SEED)))
    basis, basis_record = load_frozen_inputs()
    model_cfg = dict(cfg["model"])
    bundle = load_whisper({"model": model_cfg})
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    controller = FixedBasisAdaptiveController(bundle.d_model, basis, BOTTLENECK).to(bundle.device)
    params = parameter_report(controller, bundle.model)
    data_cfg = load_config(cfg["data"]["candidate_config"])
    from steer_sweep.trackb.data import build_examples
    examples = build_examples(bundle, data_cfg, ("loc-train", "util-train"))
    correction_set = load_correction_set(cfg["data"]["correction_set_artifact"])
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
                            "n_positions": correction_set.n_positions},
        "objective": "correction-only CE on C_E",
        "parameter_report": params,
        "gpu_run": True,
    })
    _atomic_json(output_dir / "manifest.json", manifest)
    history = []
    for epoch in range(EPOCHS):
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_targets = 0
        batches = [examples[i:i + BATCH_SIZE]
                   for i in range(0, len(examples), BATCH_SIZE)]
        for index, batch_examples in enumerate(batches):
            loss, n_targets = train_batch(
                bundle, controller, optimizer, batch_examples, correction_set,
                beta=DG04_REFERENCE_BETA, prefix_width=prefix_width)
            total_loss += float(loss) * n_targets
            total_targets += n_targets
            if (index + 1) % GRAD_ACCUM == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(controller.trainable_parameters, CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = {"epoch": epoch + 1, "correction_ce": total_loss / max(1, total_targets),
                  "n_C_E": total_targets, "selection": "pending free-decoding D-dev-select"}
        history.append(record)
        _atomic_torch(output_dir / f"checkpoint_epoch{epoch + 1}.pt",
                       checkpoint_payload(controller, epoch=epoch + 1, cfg=cfg,
                                          basis_record=basis_record,
                                          train_targets=total_targets))
    _atomic_json(output_dir / "training_history.json", {"records": history})
    return {"manifest": manifest, "history": history,
            "trainable_parameters": params["controller_trainable"]}


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
