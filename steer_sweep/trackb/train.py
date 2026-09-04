"""The shared training loop. Identical budget and protocol for every arm.

`config.TRAIN_BUDGET` is applied without exception: same max_steps, batch size,
gradient accumulation, evaluation cadence, early-stopping rule, warmup,
precision and seeds. What differs between arms is only the trainable module and
the loss terms that module's method defines.

The budget is intentionally small. These are under-trained observation runs and
their absolute MER is not comparable to published numbers.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .. import config as C
from . import data as TD
from .modules import assert_frozen_backbone, backbone_grad_norm


@dataclass
class ArmSpec:
    """One training arm: what is trainable and what the loss contains."""

    name: str
    build: Callable[..., dict[str, Any]]     # -> {"modules": [...], "context": cm}
    use_lid: bool = True
    extra_loss: Callable | None = None       # (outputs, batch, state) -> tensor
    stages: int = 1
    note: str = ""


@dataclass
class TrainResult:
    arm: str
    lr: float
    lambda_lid: float
    seed: int
    steps_run: int = 0
    epochs_run: float = 0.0
    best_dev_mer: float = float("inf")
    best_step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    trainable_parameters: int = 0
    wall_clock_sec: float = 0.0
    peak_memory_bytes: int = 0
    early_stopped: bool = False
    stage_steps: list[int] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _lr_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def forward_losses(bundle, batch: dict[str, Any], *, lid_head, lambda_lid: float,
                   site: str, layers: Sequence[int],
                   extra_loss: Callable | None = None,
                   need_attentions: bool = False) -> dict[str, torch.Tensor]:
    """CE on the transcript plus lambda_lid * LID, identical in every arm."""
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.hooks import ActivationRecorder

    recorder_cm: Any = contextlib.nullcontext()
    if lid_head is not None:
        module = "encoder" if site == C.SITE_ENCODER else "decoder"
        recorder_cm = ActivationRecorder(bundle, [int(layers[0])], module=module,
                                         to_dtype=torch.float32)

    with recorder_cm as recorder:
        out = bundle.model(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            output_attentions=bool(need_attentions),
            use_cache=False,
        )
        states = dict(recorder.states) if recorder is not None else {}

    logits = out.logits
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        batch["ce_labels"].reshape(-1), ignore_index=TD.LID_IGNORE)

    losses = {"ce": ce, "total": ce}
    if lid_head is not None and lambda_lid:
        hidden = states[int(layers[0])]
        labels = (batch["lid_frame_labels"] if site == C.SITE_ENCODER
                  else batch["lid_token_labels"])
        lid = lid_head.loss(hidden, labels)
        losses["lid"] = lid
        losses["total"] = losses["total"] + float(lambda_lid) * lid
    if extra_loss is not None:
        extra = extra_loss(out, batch)
        if extra is not None:
            losses["extra"] = extra
            losses["total"] = losses["total"] + extra
    return losses


@torch.inference_mode()
def evaluate(bundle, examples: Sequence[TD.Example], *, batch_size: int = 8,
             max_new_tokens: int = 200) -> dict[str, Any]:
    """Greedy dev decode scored by OUR scorer only."""
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    from csasr.experiments.v2r3_day5_expansion import corpus_word_error_rate
    from csasr.models.whisper import batch_model_inputs

    references, hypotheses = [], []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        inputs = batch_model_inputs(bundle, [e.audio_path for e in chunk])
        out = bundle.model.generate(
            **inputs, task="transcribe", language="zh", do_sample=False,
            num_beams=1, temperature=0.0, max_new_tokens=max_new_tokens,
            condition_on_prev_tokens=False)
        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        for e, text in zip(chunk, bundle.processor.batch_decode(
                sequences, skip_special_tokens=True)):
            references.append(e.reference)
            hypotheses.append(text.strip())
    mer = corpus_mer(references, hypotheses)
    return {
        "MER": float(mer["mer"]),
        "PIER": float(pier(references, hypotheses)["pier"]),
        "WER": float(corpus_word_error_rate(references, hypotheses)["wer"]),
        "n": len(references),
    }


def train_arm(bundle, arm: ArmSpec, *, train_examples: Sequence[TD.Example],
              dev_examples: Sequence[TD.Example], site: str, layers: Sequence[int],
              lr: float, lambda_lid: float, seed: int, budget: dict[str, Any],
              log=None, smoke: bool = False,
              build_kwargs: dict[str, Any] | None = None,
              final_eval: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
              ) -> TrainResult:
    """Train one arm under the shared budget. Returns metrics and diagnostics."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))

    max_steps = int(budget["max_steps"])
    batch_size = int(budget["batch_size"])
    grad_accum = int(budget["grad_accum"])
    eval_every = int(budget["eval_every"])
    patience = int(budget["early_stop_patience"])
    warmup = max(1, int(budget["warmup_fraction"] * max_steps))

    steps_per_epoch = max(1, math.ceil(len(train_examples) / (batch_size * grad_accum)))
    max_steps = min(max_steps, steps_per_epoch * int(budget["max_epochs"]))

    built = arm.build(bundle=bundle, site=site, layers=layers, seed=seed,
                      **(build_kwargs or {}))
    modules: list[nn.Module] = list(built["modules"])
    lid_head = built.get("lid_head")
    context = built.get("context") or contextlib.nullcontext()
    extra_loss = built.get("extra_loss")
    need_attentions = bool(built.get("need_attentions", False))

    parameters = [p for m in modules for p in m.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=float(lr),
                                  weight_decay=float(budget["weight_decay"]),
                                  betas=tuple(budget["betas"]), eps=float(budget["eps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup=warmup, total=max_steps))

    collate = TD.Collator(bundle, site)
    stream = TD.batches(train_examples, batch_size, seed=seed)
    result = TrainResult(arm=arm.name, lr=float(lr), lambda_lid=float(lambda_lid),
                         seed=int(seed), trainable_parameters=n_trainable)
    if smoke:
        result.notes.append("SMOKE: reduced steps; numbers are not results")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    best, since_best = float("inf"), 0

    with context:
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for _ in range(grad_accum):
                batch = collate(next(stream))
                losses = forward_losses(
                    bundle, batch, lid_head=lid_head if arm.use_lid else None,
                    lambda_lid=lambda_lid if arm.use_lid else 0.0, site=site,
                    layers=layers, extra_loss=extra_loss,
                    need_attentions=need_attentions)
                (losses["total"] / grad_accum).backward()
                for k, v in losses.items():
                    accumulated[k] = accumulated.get(k, 0.0) + float(v.detach()) / grad_accum
            torch.nn.utils.clip_grad_norm_(parameters, float(budget["grad_clip"]))
            optimizer.step()
            scheduler.step()
            result.steps_run = step + 1

            if step == 0:
                # requirement: gradient flows only to intended parameters
                norm = backbone_grad_norm(bundle)
                if norm != 0.0:
                    raise AssertionError(
                        f"arm {arm.name}: frozen-backbone gradient norm {norm} "
                        f"!= 0.0 at step 0. Stopping.")

            if (step + 1) % eval_every == 0 or (step + 1) == max_steps:
                metrics = evaluate(bundle, dev_examples, batch_size=batch_size)
                entry = {"step": step + 1, **accumulated, **metrics,
                         "lr": scheduler.get_last_lr()[0]}
                result.history.append(entry)
                if log:
                    log.info("  %s step %d/%d loss %.4f dev MER %.4f",
                             arm.name, step + 1, max_steps,
                             accumulated.get("total", float("nan")), metrics["MER"])
                if metrics["MER"] < best - 1e-6:
                    best, since_best = metrics["MER"], 0
                    result.best_dev_mer, result.best_step = metrics["MER"], step + 1
                else:
                    since_best += 1
                    if since_best >= patience:
                        result.early_stopped = True
                        if log:
                            log.info("  %s early stop at step %d (patience %d)",
                                     arm.name, step + 1, patience)
                        break

        # the trainable modules are attached only inside `context`; a LoRA arm
        # unloads on exit and an intervention removes its hooks, so anything
        # that must observe the TRAINED model has to run here
        if final_eval is not None:
            built["final_eval"] = final_eval(built)

    result.wall_clock_sec = time.monotonic() - started
    result.epochs_run = result.steps_run / steps_per_epoch
    result.peak_memory_bytes = (int(torch.cuda.max_memory_allocated())
                                if torch.cuda.is_available() else 0)
    result.stage_steps = [result.steps_run]
    result.diagnostics.update(built.get("diagnostics", {}))
    result.diagnostics["modules"] = built.get("module_report", {})
    result.diagnostics["steps_per_epoch"] = steps_per_epoch
    result.diagnostics["max_steps_effective"] = max_steps
    result.diagnostics["backbone_grad_norm_final"] = backbone_grad_norm(bundle)
    built["trained_modules"] = modules
    result.diagnostics["_built"] = built
    return result
