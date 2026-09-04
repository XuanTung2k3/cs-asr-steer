#!/usr/bin/env python
"""Job B: Minimal training comparison — rank-2 local steering, SALSA-D, LoRA-PM.

Runs on one H100 within a 3-hour wall-time budget.

Usage:
    python experiments/job_b_training.py [--dry-run]

Priority ordering (if the 3-hour limit is tight):
    1. T1 (proposed rank-2 local steering) — MUST complete
    2. T2 (SALSA-D)
    3. T3 (LoRA-PM)
    4. Beam-5 final pass on best checkpoint of each completed method
    5. SEAME pilot on completed methods
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Repository path setup ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Configuration ──────────────────────────────────────────────────
DEVICE = "cuda"
CONFIG_PATH = "lss/l1b_candidates_dialogue_v2r3.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "job_b"
DECODER_LAYER = 24
SALSA_LAYERS = (24, 25)
ALPHA = 2.0
WEIGHT_NAT = 2.0
WEIGHT_PROMPT = 0.5
SEED = 42
MAX_EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 2        # effective batch size = 16
LR_T1 = 5e-4
LR_T2 = 5e-4
LR_T3 = 1e-4
TRAIN_UTTERANCES = 500
PREFIX_WIDTH = 4
D_MODEL = 1280
BUDGET_HOURS = 3.0
GRAD_CLIP = 1.0
LID_IGNORE = -100

# ── Logging ────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("job_b")


# ═══════════════════════════════════════════════════════════════════
# Atomic file writing
# ═══════════════════════════════════════════════════════════════════

def atomic_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write to tmp, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
        os.rename(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.rename(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_torch_save(path: Path, data: Any) -> None:
    """Save a checkpoint atomically: write beside the target, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(data, tmp)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ═══════════════════════════════════════════════════════════════════
# Helper: split output tuple from transformer layers
# ═══════════════════════════════════════════════════════════════════

def _split_output(output: Any) -> tuple[torch.Tensor, Any]:
    """Return (hidden_states, rebuild_fn) for transformer layer output."""
    if isinstance(output, tuple):
        rest = output[1:]
        return output[0], (lambda new: (new,) + rest)
    return output, (lambda new: new)


# ═══════════════════════════════════════════════════════════════════
# Budget tracker
# ═══════════════════════════════════════════════════════════════════

class Budget:
    """Wall-clock budget with a clean stop."""

    def __init__(self, hours: float) -> None:
        self.hours = float(hours)
        self.started = time.monotonic()

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started) / 3600.0

    def exhausted(self) -> bool:
        return bool(self.elapsed_hours >= self.hours)

    def remaining_hours(self) -> float:
        return max(0.0, self.hours - self.elapsed_hours)


# ═══════════════════════════════════════════════════════════════════
# Trainable modules
# ═══════════════════════════════════════════════════════════════════

class Rank2LocalSteering(nn.Module):
    """T1: Proposed rank-2 local steering with learned gate.

    Trainable: U (d_model × 2 = 2560), w1, w2, b (3). Total ≈ 2563.
    a = normalize([2.0, 0.5]) is FROZEN.
    Gate uses FIXED v_nat, v_prompt (not columns of U).
    """

    def __init__(self, d_model: int, v_nat: torch.Tensor,
                 v_prompt: torch.Tensor, alpha: float = 2.0) -> None:
        super().__init__()
        # U initialized from v_nat, v_prompt as columns
        self.U = nn.Parameter(torch.stack([v_nat.float(), v_prompt.float()], dim=1))

        # a is FROZEN
        a = torch.tensor([WEIGHT_NAT, WEIGHT_PROMPT], dtype=torch.float32)
        a = a / a.norm()
        self.register_buffer("a", a)

        # Gate references (fixed, not columns of U)
        self.register_buffer("v_nat_fixed", v_nat.float().clone())
        self.register_buffer("v_prompt_fixed", v_prompt.float().clone())

        # Gate parameters
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

        self.alpha = alpha

    def gate(self, h: torch.Tensor) -> torch.Tensor:
        """Compute soft gate m_t ∈ (0, 1) for each position."""
        s_nat = (h * self.v_nat_fixed).sum(dim=-1)
        s_prompt = (h * self.v_prompt_fixed).sum(dim=-1)
        s_t = self.w1 * s_nat + self.w2 * s_prompt
        return torch.sigmoid(s_t + self.b)

    def direction(self) -> torch.Tensor:
        """Column-normalize U, then combine with frozen a."""
        U_hat = self.U / self.U.norm(dim=0, keepdim=True)
        direction = U_hat @ self.a
        return direction / direction.norm().clamp(min=1e-6)  # (D,), unit norm

    def forward(self, h: torch.Tensor, *, force_global: bool = False) -> torch.Tensor:
        """h: (B, T, D) hidden states at layer 24."""
        if force_global:
            m_t = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        else:
            m_t = self.gate(h)

        d = self.direction()
        delta = self.alpha * m_t.unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        h_bar = h + delta

        # NormPreserve
        h_norm = h.norm(dim=-1, keepdim=True)
        h_bar_norm = h_bar.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return h_bar * (h_norm / h_bar_norm)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SALSAD(nn.Module):
    """T2: SALSA-D — parameter-matched decoder steering at layers 24 and 25.

    Trainable: v_24 (1280), v_25 (1280). Total = 2560.
    NormPreserve applied consistently.  The original SALSA comparison does
    not define this same rescaling; it is retained here intentionally for a
    fair T1/T2 comparison.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.v_24 = nn.Parameter(torch.zeros(d_model))
        self.v_25 = nn.Parameter(torch.zeros(d_model))

    def forward_layer(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply steering at one layer with NormPreserve."""
        v = self.v_24 if layer == 24 else self.v_25
        h_bar = h + v.unsqueeze(0).unsqueeze(0)
        h_norm = h.norm(dim=-1, keepdim=True)
        h_bar_norm = h_bar.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return h_bar * (h_norm / h_bar_norm)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ManualLoRA(nn.Module):
    """T3: Manual LoRA on Q and V projections of decoder layer 24 self-attention.

    Rank r=1, alpha=1, dropout=0.
    Each target gets two small matrices: A (r × in_features), B (out_features × r).
    """

    def __init__(self, layer_module: nn.Module, rank: int = 1,
                 alpha: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        # Keep the frozen target as a plain reference, not an nn.Module child:
        # module.to(...).float() must not convert the whole decoder layer or
        # copy it into the LoRA checkpoint state dict.
        object.__setattr__(self, "layer_module", layer_module)

        q_proj = layer_module.self_attn.q_proj
        v_proj = layer_module.self_attn.v_proj

        self.q_A = nn.Parameter(torch.randn(rank, q_proj.in_features) * 0.01)
        self.q_B = nn.Parameter(torch.zeros(q_proj.out_features, rank))
        self.v_A = nn.Parameter(torch.randn(rank, v_proj.in_features) * 0.01)
        self.v_B = nn.Parameter(torch.zeros(v_proj.out_features, rank))

        self._orig_q_forward = q_proj.forward
        self._orig_v_forward = v_proj.forward
        self._handles: list = []

    def _patch_q(self, x: torch.Tensor) -> torch.Tensor:
        base = self._orig_q_forward(x)
        lora = (x.to(self.q_A.dtype) @ self.q_A.T @ self.q_B.T) * self.scaling
        return base + lora.to(base.dtype)

    def _patch_v(self, x: torch.Tensor) -> torch.Tensor:
        base = self._orig_v_forward(x)
        lora = (x.to(self.v_A.dtype) @ self.v_A.T @ self.v_B.T) * self.scaling
        return base + lora.to(base.dtype)

    def attach(self) -> "ManualLoRA":
        """Monkey-patch the forward methods."""
        self.layer_module.self_attn.q_proj.forward = self._patch_q
        self.layer_module.self_attn.v_proj.forward = self._patch_v
        return self

    def detach(self) -> None:
        """Restore original forwards."""
        self.layer_module.self_attn.q_proj.forward = self._orig_q_forward
        self.layer_module.self_attn.v_proj.forward = self._orig_v_forward

    def __enter__(self) -> "ManualLoRA":
        return self.attach()

    def __exit__(self, *exc: Any) -> bool:
        self.detach()
        return False

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# Hook context managers for steering during decoding/training
# ═══════════════════════════════════════════════════════════════════

class T1Hook:
    """Forward hook that applies Rank2LocalSteering at decoder layer 24."""

    def __init__(self, bundle: Any, module: Rank2LocalSteering, *,
                 force_global: bool = False,
                 skip_prefix: bool = True) -> None:
        self.bundle = bundle
        self.module = module
        self.force_global = force_global
        self.skip_prefix = skip_prefix
        self._handle: Any = None
        # For generation, track absolute position
        self._position_tracker: Any = None
        self._position_handle: Any = None

    def _decoder_pre_hook(self, _mod: Any, args: Any, kwargs: Any) -> None:
        from steer_sweep.hooks import cache_length
        self._abs_start = cache_length(kwargs.get("past_key_values"))

    def _hook(self, _mod: Any, _inp: Any, output: Any) -> Any:
        hs, rebuild = _split_output(output)
        steered = self.module(hs.float(), force_global=self.force_global).to(hs.dtype)

        if self.skip_prefix:
            start = getattr(self, "_abs_start", 0)
            mask = torch.ones(hs.shape[:2], dtype=torch.bool, device=hs.device)
            for j in range(hs.shape[1]):
                if start + j < PREFIX_WIDTH:
                    mask[:, j] = False
            if not mask.all():
                steered = torch.where(mask.unsqueeze(-1), steered, hs)

        return rebuild(steered)

    def __enter__(self) -> "T1Hook":
        self._position_handle = self.bundle.model.model.decoder.register_forward_pre_hook(
            self._decoder_pre_hook, with_kwargs=True
        )
        self._handle = self.bundle.decoder_layer(DECODER_LAYER).register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._position_handle is not None:
            self._position_handle.remove()
            self._position_handle = None
        return False


class T2Hook:
    """Forward hooks that apply SALSA-D at decoder layers 24 and 25."""

    def __init__(self, bundle: Any, module: SALSAD, *,
                 skip_prefix: bool = True) -> None:
        self.bundle = bundle
        self.module = module
        self.skip_prefix = skip_prefix
        self._handles: list = []
        self._abs_start = 0

    def _decoder_pre_hook(self, _mod: Any, args: Any, kwargs: Any) -> None:
        from steer_sweep.hooks import cache_length
        self._abs_start = cache_length(kwargs.get("past_key_values"))

    def _make_hook(self, layer: int) -> Any:
        def hook(_mod: Any, _inp: Any, output: Any) -> Any:
            hs, rebuild = _split_output(output)
            steered = self.module.forward_layer(hs.float(), layer).to(hs.dtype)

            if self.skip_prefix:
                start = self._abs_start
                mask = torch.ones(hs.shape[:2], dtype=torch.bool, device=hs.device)
                for j in range(hs.shape[1]):
                    if start + j < PREFIX_WIDTH:
                        mask[:, j] = False
                if not mask.all():
                    steered = torch.where(mask.unsqueeze(-1), steered, hs)

            return rebuild(steered)
        return hook

    def __enter__(self) -> "T2Hook":
        self._handles.append(
            self.bundle.model.model.decoder.register_forward_pre_hook(
                self._decoder_pre_hook, with_kwargs=True
            )
        )
        for layer in SALSA_LAYERS:
            self._handles.append(
                self.bundle.decoder_layer(layer).register_forward_hook(
                    self._make_hook(layer)
                )
            )
        return self

    def __exit__(self, *exc: Any) -> bool:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_directions(bundle: Any, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Load v_nat and v_prompt at decoder layer 24."""
    from steer_sweep.directions import DirectionStore
    store = DirectionStore(cfg, RESULTS_DIR / "directions_cache", log=log)
    v_nat_spec = store.get("v_nat", "decoder", DECODER_LAYER, bundle=bundle)
    v_prompt_spec = store.get("v_prompt", "decoder", DECODER_LAYER, bundle=bundle)
    return v_nat_spec.vector.float(), v_prompt_spec.vector.float()


def select_training_data(bundle: Any, cfg: dict, *,
                         dry_run: bool = False) -> list[Any]:
    """Select exactly TRAIN_UTTERANCES training examples with seed=SEED.

    Uses dialogue-stratified subsampling to ensure balanced representation.
    """
    from steer_sweep.trackb.data import build_examples, subsample

    all_examples = build_examples(bundle, cfg, ("loc-train", "util-train"), log=log)
    log.info("total training pool: %d examples", len(all_examples))

    if dry_run:
        return all_examples[:2]

    if len(all_examples) < TRAIN_UTTERANCES:
        raise RuntimeError(
            f"need exactly {TRAIN_UTTERANCES} training utterances, but only "
            f"{len(all_examples)} examples are available")

    n = TRAIN_UTTERANCES
    fraction = n / len(all_examples) if len(all_examples) > 0 else 1.0
    selected = subsample(all_examples, fraction, seed=SEED)

    # Make the frozen list exactly TRAIN_UTTERANCES even when per-dialogue
    # rounding in the stratified sampler undershoots the requested count.
    rng = np.random.default_rng(SEED)
    if len(selected) != TRAIN_UTTERANCES:
        selected_ids = {e.utterance_id for e in selected}
        if len(selected) > TRAIN_UTTERANCES:
            indices = rng.choice(len(selected), TRAIN_UTTERANCES, replace=False)
            selected = [selected[i] for i in sorted(indices)]
        else:
            remaining = [e for e in all_examples
                         if e.utterance_id not in selected_ids]
            rng.shuffle(remaining)
            selected.extend(remaining[:TRAIN_UTTERANCES - len(selected)])

    if len(selected) != TRAIN_UTTERANCES:
        raise AssertionError(
            f"training selection produced {len(selected)} examples, expected "
            f"{TRAIN_UTTERANCES}")

    log.info("selected %d training utterances (seed=%d)", len(selected), SEED)
    return selected


def load_dev_examples(bundle: Any, cfg: dict, *, dry_run: bool = False) -> list[Any]:
    """Load dev-select examples for evaluation."""
    from steer_sweep.trackb.data import build_examples
    examples = build_examples(bundle, cfg, ("D-dev-select",),
                              limit=2 if dry_run else None, log=log)
    log.info("dev-select: %d examples", len(examples))
    return examples


# ═══════════════════════════════════════════════════════════════════
# Pre-training probe check
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_probe_check(bundle: Any, cfg: dict,
                    v_nat: torch.Tensor, v_prompt: torch.Tensor, *,
                    dry_run: bool = False) -> dict[str, Any]:
    """Compute probe AUROC at decoder layer 24 on dev-select.

    Extracts h_{24,t}, computes projections onto v_nat and v_prompt,
    fits logistic regression → EN/ZH label.
    """
    from csasr.models.hooks import ActivationRecorder
    from csasr.models.generation import teacher_forced_forward
    from steer_sweep.trackb.data import build_examples, LID_EN, LID_ZH

    log.info("=== Pre-training probe check ===")

    # Load dev-select data with LID labels
    examples = build_examples(bundle, cfg, ("D-dev-select",),
                              limit=4 if dry_run else None, log=log)

    all_s_nat: list[float] = []
    all_s_prompt: list[float] = []
    all_labels: list[int] = []
    probe_batch_size = 4

    v_nat_dev = v_nat.to(bundle.device)
    v_prompt_dev = v_prompt.to(bundle.device)

    for start in range(0, len(examples), probe_batch_size):
        chunk = examples[start:start + probe_batch_size]
        audio_paths = [e.audio_path for e in chunk]
        token_seqs = [e.token_ids for e in chunk]

        with ActivationRecorder(bundle, [DECODER_LAYER], module="decoder") as rec:
            _out, _padded, mask = teacher_forced_forward(
                bundle, audio_paths, token_seqs
            )
            h_24 = rec.states[DECODER_LAYER]  # (B, T, D)

        for b, example in enumerate(chunk):
            T = min(h_24.shape[1], len(example.lid_token_labels))
            for t in range(T):
                label = example.lid_token_labels[t]
                if label == LID_EN or label == LID_ZH:
                    h_t = h_24[b, t].float()
                    s_n = float(torch.dot(h_t, v_nat_dev.float()))
                    s_p = float(torch.dot(h_t, v_prompt_dev.float()))
                    all_s_nat.append(s_n)
                    all_s_prompt.append(s_p)
                    all_labels.append(label)

        if (start // probe_batch_size) % 20 == 0:
            log.info("  probe: processed %d/%d examples (%d labeled steps)",
                     min(start + probe_batch_size, len(examples)),
                     len(examples), len(all_labels))

    if len(all_labels) < 10:
        log.warning("too few labeled steps (%d) for probe", len(all_labels))
        return {"auroc": float("nan"), "status": "INSUFFICIENT_DATA",
                "n_labeled": len(all_labels)}

    X = np.column_stack([all_s_nat, all_s_prompt])
    y = np.array(all_labels)

    # EN = LID_EN = 1, ZH = LID_ZH = 0
    en_count = int((y == LID_EN).sum())
    zh_count = int((y == LID_ZH).sum())
    log.info("  probe data: %d EN steps, %d ZH steps", en_count, zh_count)

    # Fit logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr_model.fit(X, y)

    proba = lr_model.predict_proba(X)
    # EN is class index where lr_model.classes_ == LID_EN
    en_class_idx = int(np.where(lr_model.classes_ == LID_EN)[0][0])
    en_scores = proba[:, en_class_idx]

    auroc = float(roc_auc_score(y, en_scores))
    auprc = float(average_precision_score(y == LID_EN, en_scores))

    # English recall at gate budgets
    sorted_scores = np.sort(en_scores)[::-1]
    en_mask = (y == LID_EN)
    recall_at_budget: dict[str, float] = {}
    for budget_pct in (10, 20, 50):
        n_gate = max(1, int(len(y) * budget_pct / 100))
        threshold = sorted_scores[min(n_gate - 1, len(sorted_scores) - 1)]
        predicted_en = en_scores >= threshold
        recall = float(predicted_en[en_mask].sum() / max(1, en_mask.sum()))
        recall_at_budget[f"recall_at_{budget_pct}pct"] = recall

    # Score distributions
    en_s_nat = np.array(all_s_nat)[en_mask]
    zh_s_nat = np.array(all_s_nat)[~en_mask]
    en_s_prompt = np.array(all_s_prompt)[en_mask]
    zh_s_prompt = np.array(all_s_prompt)[~en_mask]

    result = {
        "auroc": auroc,
        "auprc": auprc,
        **recall_at_budget,
        "n_en": en_count,
        "n_zh": zh_count,
        "n_total": len(all_labels),
        "score_distributions": {
            "s_nat_en_mean": float(en_s_nat.mean()) if len(en_s_nat) else float("nan"),
            "s_nat_en_std": float(en_s_nat.std()) if len(en_s_nat) else float("nan"),
            "s_nat_zh_mean": float(zh_s_nat.mean()) if len(zh_s_nat) else float("nan"),
            "s_nat_zh_std": float(zh_s_nat.std()) if len(zh_s_nat) else float("nan"),
            "s_prompt_en_mean": float(en_s_prompt.mean()) if len(en_s_prompt) else float("nan"),
            "s_prompt_en_std": float(en_s_prompt.std()) if len(en_s_prompt) else float("nan"),
            "s_prompt_zh_mean": float(zh_s_prompt.mean()) if len(zh_s_prompt) else float("nan"),
            "s_prompt_zh_std": float(zh_s_prompt.std()) if len(zh_s_prompt) else float("nan"),
        },
        "logistic_coef": lr_model.coef_.tolist(),
        "logistic_intercept": lr_model.intercept_.tolist(),
    }

    # Decision logic
    if auroc >= 0.75:
        decision = "PROCEED: two-projection gate is well-supported"
    elif auroc >= 0.65:
        decision = "PROCEED WITH CAUTION: localization is exploratory"
    else:
        decision = "WARNING: consider full hidden-state gate; proceeding anyway for pilot"

    result["decision"] = decision
    print(f"\n{'='*60}")
    print(f"PROBE AUROC = {auroc:.4f}  AUPRC = {auprc:.4f}")
    print(f"  EN steps: {en_count}, ZH steps: {zh_count}")
    print(f"  {decision}")
    print(f"{'='*60}\n")

    atomic_json(RESULTS_DIR / "probe_auroc.json", result)
    return result


# ═══════════════════════════════════════════════════════════════════
# Training infrastructure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainState:
    """Per-method training state."""
    method: str
    best_mer: float = float("inf")
    best_epoch: int = -1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_results: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.epoch_results is None:
            self.epoch_results = []


def collate_batch(bundle: Any, examples: list[Any]) -> dict[str, torch.Tensor]:
    """Collate a batch of examples for teacher-forced training (CE only)."""
    from csasr.models.whisper import batch_model_inputs

    inputs = batch_model_inputs(bundle, [e.audio_path for e in examples])
    pad_id = bundle.processor.tokenizer.pad_token_id

    width = max(len(e.token_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    ce_labels = torch.full((len(examples), width), LID_IGNORE, dtype=torch.long)
    lid_labels = torch.full((len(examples), width), LID_IGNORE, dtype=torch.long)

    for i, e in enumerate(examples):
        n = len(e.token_ids)
        ids[i, :n] = torch.tensor(e.token_ids, dtype=torch.long)
        # CE targets: next token; ignore prefix
        ce_labels[i, :n - 1] = torch.tensor(e.token_ids[1:], dtype=torch.long)
        ce_labels[i, :PREFIX_WIDTH - 1] = LID_IGNORE
        # LID labels for gate diagnostics
        lid_labels[i, :n] = torch.tensor(e.lid_token_labels, dtype=torch.long)

    return {
        "input_features": inputs["input_features"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": ids.to(bundle.device),
        "ce_labels": ce_labels.to(bundle.device),
        "lid_labels": lid_labels.to(bundle.device),
        "examples": examples,
    }


def epoch_batches(examples: list[Any], batch_size: int, *,
                  seed: int, epoch: int) -> list[list[Any]]:
    """Deterministic epoch batches, shuffled by epoch."""
    rng = np.random.default_rng(seed + epoch * 1000)
    order = np.arange(len(examples))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if len(idx) > 0:
            batches.append([examples[int(i)] for i in idx])
    return batches


@torch.inference_mode()
def compute_teacher_forced_metrics(
    bundle: Any, examples: list[Any], *,
    batch_size: int = 8,
    hook_context: Any = None,
    gate_module: Rank2LocalSteering | None = None,
) -> dict[str, Any]:
    """Compute teacher-forced CE and per-token diagnostics."""
    from csasr.models.hooks import ActivationRecorder
    from steer_sweep.trackb.data import LID_EN, LID_ZH

    total_ce = 0.0
    total_tokens = 0
    cs_ce = 0.0
    cs_tokens = 0
    gold_probs: list[float] = []
    gold_margins: list[float] = []
    gold_ranks: list[int] = []

    # Gate diagnostics (T1 only)
    gate_values: list[float] = []
    gate_en: list[float] = []
    gate_zh: list[float] = []

    cm = hook_context if hook_context is not None else contextlib.nullcontext()

    with cm:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            batch = collate_batch(bundle, chunk)

            if gate_module is not None:
                with ActivationRecorder(bundle, [DECODER_LAYER], module="decoder") as rec:
                    out = bundle.model(
                        input_features=batch["input_features"],
                        attention_mask=batch["attention_mask"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        use_cache=False,
                    )
                    h_24 = rec.states[DECODER_LAYER].float()
                    m_t = gate_module.gate(h_24)  # (B, T)
            else:
                out = bundle.model(
                    input_features=batch["input_features"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    use_cache=False,
                )
                m_t = None

            logits = out.logits.float()
            labels = batch["ce_labels"]
            lid = batch["lid_labels"]

            # Per-token CE
            vocab = logits.shape[-1]
            flat_logits = logits[:, :-1].reshape(-1, vocab)
            flat_labels = labels[:, 1:].reshape(-1)
            flat_lid = lid[:, 1:].reshape(-1) if lid is not None else None

            valid = flat_labels != LID_IGNORE
            if valid.any():
                ce_val = F.cross_entropy(flat_logits[valid], flat_labels[valid],
                                         reduction="sum")
                total_ce += float(ce_val)
                total_tokens += int(valid.sum())

                # CS-token CE (EN tokens only)
                if flat_lid is not None:
                    cs_mask = valid & (flat_lid == LID_EN)
                    if cs_mask.any():
                        cs_ce_val = F.cross_entropy(flat_logits[cs_mask],
                                                     flat_labels[cs_mask],
                                                     reduction="sum")
                        cs_ce += float(cs_ce_val)
                        cs_tokens += int(cs_mask.sum())

                # Gold token probability, margin, rank
                probs = F.softmax(flat_logits[valid], dim=-1)
                gold = flat_labels[valid]
                for i in range(len(gold)):
                    gp = float(probs[i, gold[i]])
                    gold_probs.append(gp)
                    top_prob = float(probs[i].max())
                    gold_margins.append(gp - top_prob if gold[i] != probs[i].argmax()
                                        else 0.0)
                    rank = int((probs[i] >= gp).sum())
                    gold_ranks.append(rank)

            # Gate diagnostics
            if m_t is not None and lid is not None:
                for b in range(m_t.shape[0]):
                    T = min(m_t.shape[1], lid.shape[1])
                    for t in range(T):
                        gv = float(m_t[b, t])
                        lab = int(lid[b, t])
                        if lab == LID_EN:
                            gate_en.append(gv)
                            gate_values.append(gv)
                        elif lab == LID_ZH:
                            gate_zh.append(gv)
                            gate_values.append(gv)

    result: dict[str, Any] = {
        "ce": total_ce / max(1, total_tokens),
        "total_tokens": total_tokens,
        "cs_ce": cs_ce / max(1, cs_tokens) if cs_tokens > 0 else float("nan"),
        "cs_tokens": cs_tokens,
        "gold_prob_mean": float(np.mean(gold_probs)) if gold_probs else float("nan"),
        "gold_margin_mean": float(np.mean(gold_margins)) if gold_margins else float("nan"),
        "gold_rank_mean": float(np.mean(gold_ranks)) if gold_ranks else float("nan"),
    }

    if gate_values:
        gv = np.array(gate_values)
        result["gate_diagnostics"] = {
            "mean_m_t": float(gv.mean()),
            "median_m_t": float(np.median(gv)),
            "p90_m_t": float(np.percentile(gv, 90)),
            "pct_above_0.5": float((gv > 0.5).mean()),
            "mean_m_t_en": float(np.mean(gate_en)) if gate_en else float("nan"),
            "mean_m_t_zh": float(np.mean(gate_zh)) if gate_zh else float("nan"),
            "n_en_steps": len(gate_en),
            "n_zh_steps": len(gate_zh),
        }

    return result


@torch.inference_mode()
def decode_with_context(
    bundle: Any, pop: Any, context: Any, *,
    language: str = "zh", num_beams: int = 1, batch_size: int = 8,
) -> dict[str, str]:
    """Decode population with an active hook context."""
    from csasr.models.whisper import batch_model_inputs

    texts: dict[str, str] = {}
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)

    with context:
        for start in range(0, len(manifest), batch_size):
            batch = manifest.iloc[start:start + batch_size]
            ids = [str(u) for u in batch["utterance_id"]]
            inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
            out = bundle.model.generate(
                **inputs, task="transcribe", language=language,
                do_sample=False, num_beams=num_beams, temperature=0.0,
                max_new_tokens=200, condition_on_prev_tokens=False,
            )
            sequences = out if isinstance(out, torch.Tensor) else out.sequences
            decoded = bundle.processor.batch_decode(sequences,
                                                     skip_special_tokens=True)
            for uid, text in zip(ids, decoded):
                texts[uid] = text.strip()

    return texts


def evaluate_method(
    bundle: Any, pop: Any, hypotheses: dict[str, str],
    baseline_texts: dict[str, str],
) -> dict[str, Any]:
    """Full evaluation: MER, PIER, EN WER, ZH CER, corrections, corruptions."""
    from steer_sweep.metrics import corpus_metrics, target_outcomes
    from steer_sweep.metrics import corruption_and_retention
    from csasr.evaluation.mer import corpus_mer
    from steer_sweep import data as D

    corpus = corpus_metrics(pop, hypotheses)
    outcomes = target_outcomes(pop, baseline_texts, hypotheses)
    risk = corruption_and_retention(pop, baseline_texts, hypotheses)

    # corpus_metrics returns "WER" (overall), not per-language breakdown.
    # Call corpus_mer directly for en_wer and zh_cer.
    references = D.reference_of(pop)
    ids = [u for u in pop.utterance_ids if u in hypotheses]
    refs = [references[u] for u in ids]
    hyps = [hypotheses[u] for u in ids]
    mer_detail = corpus_mer(refs, hyps)

    corrections = int(outcomes["corrected"].sum()) if len(outcomes) else 0
    corruptions = int(risk.get("units_corrupted", 0))
    outside = int(outcomes["spillover_changed"].sum()) if len(outcomes) else 0

    return {
        "MER": float(corpus.get("MER", float("nan"))),
        "PIER": float(corpus.get("PIER", float("nan"))),
        "en_wer": float(mer_detail.get("en_wer", float("nan"))),
        "zh_cer": float(mer_detail.get("zh_cer", float("nan"))),
        "WER": float(corpus.get("WER", float("nan"))),
        "corrections": corrections,
        "corruptions": corruptions,
        "outside_edits": outside,
        "zh_retention": float(risk.get("zh_retention", float("nan"))),
        "scored_utterances": int(corpus.get("scored_utterances", 0)),
    }


# ═══════════════════════════════════════════════════════════════════
# Per-method training
# ═══════════════════════════════════════════════════════════════════

def train_method(
    bundle: Any, cfg: dict, pop: Any,
    train_examples: list[Any], dev_examples: list[Any],
    baseline_texts: dict[str, str],
    v_nat: torch.Tensor, v_prompt: torch.Tensor,
    method: str, budget: Budget, *,
    dry_run: bool = False,
) -> TrainState | None:
    """Train one method (T1, T2, or T3) for up to MAX_EPOCHS.

    Returns TrainState with best checkpoint, or None if budget exhausted.
    """
    if budget.exhausted():
        log.warning("budget exhausted before %s", method)
        return None

    log.info("=" * 60)
    log.info("Training method: %s", method)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Freeze backbone
    for p in bundle.model.parameters():
        p.requires_grad_(False)

    method_dir = RESULTS_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)

    # Build method-specific module and hook
    if method == "T1":
        module = Rank2LocalSteering(
            D_MODEL, v_nat.to(bundle.device), v_prompt.to(bundle.device),
            alpha=ALPHA,
        ).to(bundle.device).float()
        lr = LR_T1

        def make_train_hook():
            return T1Hook(bundle, module, skip_prefix=True)

        def make_eval_hook(*, force_global: bool = False):
            return T1Hook(bundle, module, skip_prefix=True,
                          force_global=force_global)

    elif method == "T2":
        module = SALSAD(D_MODEL).to(bundle.device).float()
        lr = LR_T2

        def make_train_hook():
            return T2Hook(bundle, module, skip_prefix=True)

        def make_eval_hook(*, force_global: bool = False):
            return T2Hook(bundle, module, skip_prefix=True)

    elif method == "T3":
        layer_mod = bundle.decoder_layer(DECODER_LAYER)
        module = ManualLoRA(layer_mod, rank=1, alpha=1.0).to(bundle.device).float()
        lr = LR_T3

        def make_train_hook():
            return module  # uses __enter__/__exit__

        def make_eval_hook(*, force_global: bool = False):
            return module

    else:
        raise ValueError(f"Unknown method: {method}")

    n_params = module.n_trainable()
    log.info("  %s: %d trainable parameters, lr=%g", method, n_params, lr)

    # Optimizer
    params = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0,
                                   betas=(0.9, 0.99), eps=1e-6)

    state = TrainState(method=method)
    steps_per_epoch = max(1, math.ceil(len(train_examples) / (BATCH_SIZE * GRAD_ACCUM)))

    for epoch in range(MAX_EPOCHS):
        if budget.exhausted():
            log.warning("budget exhausted at epoch %d of %s", epoch, method)
            break

        epoch_started = time.monotonic()
        log.info("  %s epoch %d/%d", method, epoch + 1, MAX_EPOCHS)

        # Training
        module.train()
        batches = epoch_batches(train_examples, BATCH_SIZE, seed=SEED, epoch=epoch)
        epoch_loss = 0.0
        epoch_tokens = 0
        step_in_epoch = 0

        with make_train_hook():
            for bi, batch_examples in enumerate(batches):
                batch = collate_batch(bundle, batch_examples)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = bundle.model(
                        input_features=batch["input_features"],
                        attention_mask=batch["attention_mask"],
                        decoder_input_ids=batch["decoder_input_ids"],
                        use_cache=False,
                    )

                logits = out.logits.float()
                labels = batch["ce_labels"]
                vocab = logits.shape[-1]
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab),
                    labels[:, 1:].reshape(-1),
                    ignore_index=LID_IGNORE,
                )

                (loss / GRAD_ACCUM).backward()
                epoch_loss += float(loss.detach()) * int((labels[:, 1:] != LID_IGNORE).sum())
                epoch_tokens += int((labels[:, 1:] != LID_IGNORE).sum())

                if (bi + 1) % GRAD_ACCUM == 0 or (bi + 1) == len(batches):
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_in_epoch += 1

        train_ce = epoch_loss / max(1, epoch_tokens)
        log.info("    train CE: %.4f (%d tokens, %d steps)",
                 train_ce, epoch_tokens, step_in_epoch)

        # Evaluation
        module.eval()

        # Teacher-forced metrics on dev
        tf_metrics = compute_teacher_forced_metrics(
            bundle, dev_examples[:50] if not dry_run else dev_examples,
            batch_size=BATCH_SIZE,
            hook_context=make_eval_hook(),
            gate_module=module if method == "T1" else None,
        )
        log.info("    dev CE: %.4f  CS-CE: %.4f  gold-prob: %.4f",
                 tf_metrics["ce"], tf_metrics["cs_ce"],
                 tf_metrics["gold_prob_mean"])

        # Free-decoding greedy evaluation on dev-select
        with torch.inference_mode():
            hypotheses = decode_with_context(
                bundle, pop, make_eval_hook(),
                language="zh", num_beams=1, batch_size=BATCH_SIZE,
            )

        eval_metrics = evaluate_method(bundle, pop, hypotheses, baseline_texts)
        log.info("    dev-select MER: %.4f  PIER: %.4f  corr: %d  corrupt: %d",
                 eval_metrics["MER"], eval_metrics["PIER"],
                 eval_metrics["corrections"], eval_metrics["corruptions"])

        # Epoch result
        epoch_result: dict[str, Any] = {
            "epoch": epoch + 1,
            "method": method,
            "train_ce": train_ce,
            "train_tokens": epoch_tokens,
            **{f"tf_{k}": v for k, v in tf_metrics.items() if k != "gate_diagnostics"},
            **eval_metrics,
            "wall_clock_sec": time.monotonic() - epoch_started,
        }
        if "gate_diagnostics" in tf_metrics:
            epoch_result["gate_diagnostics"] = tf_metrics["gate_diagnostics"]

        state.epoch_results.append(epoch_result)
        atomic_json(method_dir / f"epoch_{epoch + 1}.json", epoch_result)

        # Checkpoint selection: lowest dev-select MER under FREE DECODING
        if eval_metrics["MER"] < state.best_mer:
            state.best_mer = eval_metrics["MER"]
            state.best_epoch = epoch + 1
            state.best_state = {k: v.detach().cpu().clone()
                                for k, v in module.state_dict().items()}
            atomic_torch_save(method_dir / "best_checkpoint.pt", {
                "method": method,
                "epoch": state.best_epoch,
                "state_dict": state.best_state,
            })
            log.info("    *** new best MER: %.4f at epoch %d", state.best_mer,
                     state.best_epoch)

    elapsed = time.monotonic() - started
    log.info("  %s completed in %.1f min (best epoch %d, MER %.4f)",
             method, elapsed / 60, state.best_epoch, state.best_mer)

    # Save checkpoint info
    peak_allocated = (int(torch.cuda.max_memory_allocated())
                      if torch.cuda.is_available() else 0)
    peak_reserved = (int(torch.cuda.max_memory_reserved())
                     if torch.cuda.is_available() else 0)
    log.info("  %s peak CUDA memory: allocated=%.2f GiB reserved=%.2f GiB",
             method, peak_allocated / 2**30, peak_reserved / 2**30)
    atomic_json(method_dir / "training_summary.json", {
        "method": method,
        "n_params": n_params,
        "lr": lr,
        "best_epoch": state.best_epoch,
        "best_mer": state.best_mer,
        "epochs_completed": len(state.epoch_results),
        "wall_clock_sec": elapsed,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "micro_batch": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "checkpoint": str(method_dir / "best_checkpoint.pt")
        if state.best_state is not None else None,
    })

    return state


# ═══════════════════════════════════════════════════════════════════
# Beam-5 evaluation
# ═══════════════════════════════════════════════════════════════════

def beam5_eval(
    bundle: Any, cfg: dict, pop: Any,
    baseline_texts: dict[str, str],
    method: str, state: TrainState,
    v_nat: torch.Tensor, v_prompt: torch.Tensor,
) -> dict[str, Any] | None:
    """Run beam-5 on the best checkpoint of a completed method."""
    if state.best_state is None:
        return None

    log.info("Beam-5 evaluation for %s (best epoch %d)", method, state.best_epoch)
    started = time.monotonic()

    # Rebuild module and load best state
    if method == "T1":
        module = Rank2LocalSteering(
            D_MODEL, v_nat.to(bundle.device), v_prompt.to(bundle.device),
            alpha=ALPHA,
        ).to(bundle.device).float()
        module.load_state_dict(state.best_state)
        module.eval()
        ctx = T1Hook(bundle, module, skip_prefix=True)
    elif method == "T2":
        module = SALSAD(D_MODEL).to(bundle.device).float()
        module.load_state_dict(state.best_state)
        module.eval()
        ctx = T2Hook(bundle, module, skip_prefix=True)
    elif method == "T3":
        layer_mod = bundle.decoder_layer(DECODER_LAYER)
        module = ManualLoRA(layer_mod, rank=1, alpha=1.0).to(bundle.device).float()
        module.load_state_dict(state.best_state)
        module.eval()
        ctx = module
    else:
        return None

    with torch.inference_mode():
        hypotheses = decode_with_context(
            bundle, pop, ctx,
            language="zh", num_beams=5, batch_size=4,
        )

    metrics = evaluate_method(bundle, pop, hypotheses, baseline_texts)
    elapsed = time.monotonic() - started
    metrics["wall_clock_sec"] = elapsed
    metrics["decoding"] = "beam5"
    log.info("  %s beam-5: MER %.4f PIER %.4f (%.1fs)",
             method, metrics["MER"], metrics["PIER"], elapsed)

    atomic_json(RESULTS_DIR / method / "beam5_metrics.json", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════
# T1 forced-global evaluation
# ═══════════════════════════════════════════════════════════════════

def t1_forced_global_eval(
    bundle: Any, pop: Any,
    baseline_texts: dict[str, str],
    state: TrainState,
    v_nat: torch.Tensor, v_prompt: torch.Tensor,
) -> dict[str, Any] | None:
    """Evaluate T1 best checkpoint with m_t = 1 for all steps."""
    if state.best_state is None:
        return None

    log.info("T1 forced-global evaluation (best epoch %d)", state.best_epoch)
    started = time.monotonic()

    module = Rank2LocalSteering(
        D_MODEL, v_nat.to(bundle.device), v_prompt.to(bundle.device),
        alpha=ALPHA,
    ).to(bundle.device).float()
    module.load_state_dict(state.best_state)
    module.eval()

    ctx = T1Hook(bundle, module, skip_prefix=True, force_global=True)
    with torch.inference_mode():
        hypotheses = decode_with_context(
            bundle, pop, ctx,
            language="zh", num_beams=1, batch_size=BATCH_SIZE,
        )

    metrics = evaluate_method(bundle, pop, hypotheses, baseline_texts)
    elapsed = time.monotonic() - started
    metrics["wall_clock_sec"] = elapsed
    metrics["mode"] = "forced_global"
    log.info("  T1 forced-global: MER %.4f PIER %.4f", metrics["MER"], metrics["PIER"])

    atomic_json(RESULTS_DIR / "T1" / "forced_global_metrics.json", metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════════
# SEAME transfer
# ═══════════════════════════════════════════════════════════════════

SEAME_PATHS = [
    Path("/mnt/data/tungnx/seame"),
    Path("/mnt/data/seame"),
    Path("/data/seame"),
]


def find_seame() -> Path | None:
    """Check if SEAME data is available."""
    for p in SEAME_PATHS:
        if p.exists():
            return p
    return None


def seame_pilot(
    bundle: Any, method: str, state: TrainState,
    v_nat: torch.Tensor, v_prompt: torch.Tensor,
    seame_root: Path, budget: Budget,
) -> dict[str, Any] | None:
    """Evaluate a method on SEAME dev_man and dev_sge (300 utterances each)."""
    if state.best_state is None or budget.exhausted():
        return None
    log.info("SEAME pilot for %s", method)
    # SEAME evaluation would go here if data is available
    # For now, return None since SEAME data layout needs to be discovered
    log.warning("SEAME pilot not implemented (data layout not confirmed)")
    return None


# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════

def print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    """Print and save the final summary table."""
    print("\n" + "=" * 100)
    print(f"{'Method':<25s} {'Params':>7s} {'MER':>8s} {'PIER':>8s} "
          f"{'Corr':>6s} {'Corrupt':>8s} {'Mand CER':>9s} {'Decode':>7s}")
    print("-" * 100)

    for label, metrics in sorted(results.items()):
        params = metrics.get("n_params", 0)
        mer = metrics.get("MER", float("nan"))
        pier_val = metrics.get("PIER", float("nan"))
        corr = metrics.get("corrections", 0)
        corrupt = metrics.get("corruptions", 0)
        zh_cer = metrics.get("zh_cer", float("nan"))
        decode = metrics.get("decoding", "greedy")

        params_str = f"{params/1000:.1f}K" if params > 0 else "0"
        print(f"{label:<25s} {params_str:>7s} {mer:>8.4f} {pier_val:>8.4f} "
              f"{corr:>6d} {corrupt:>8d} {zh_cer:>9.4f} {decode:>7s}")

    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 utterances per system to verify pipeline")
    parser.add_argument("--budget-hours", type=float, default=BUDGET_HOURS,
                        help="Wall-time budget in hours (default: 3)")
    parser.add_argument("--config", default=CONFIG_PATH,
                        help="Config YAML path")
    args = parser.parse_args(argv)

    dry_run = args.dry_run
    budget = Budget(args.budget_hours)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log.info("*** DRY RUN MODE: 2 utterances per system ***")

    # ── Load model and config ──────────────────────────────────
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper

    cfg = load_config(args.config)
    log.info("loading whisper-large-v3")
    bundle = load_whisper(cfg)

    assert bundle.d_model == D_MODEL, f"d_model mismatch: {bundle.d_model} != {D_MODEL}"
    assert bundle.num_decoder_layers == 32, "expected 32 decoder layers"

    # ── Load directions ────────────────────────────────────────
    log.info("loading directions (v_nat, v_prompt) at decoder layer %d", DECODER_LAYER)
    v_nat, v_prompt = load_directions(bundle, cfg)
    log.info("  v_nat shape: %s  v_prompt shape: %s", v_nat.shape, v_prompt.shape)
    cos = float(torch.dot(v_nat, v_prompt))
    log.info("  cos(v_nat, v_prompt) = %.4f", cos)

    # ── Pre-training probe check ──────────────────────────────
    probe_result = run_probe_check(bundle, cfg, v_nat, v_prompt, dry_run=dry_run)
    if budget.exhausted():
        log.warning("budget exhausted after probe check")
        return 0

    # ── Build evaluation population (dev-select) ──────────────
    from steer_sweep.data import build_population

    log.info("building dev-select population")
    pop = build_population(bundle, cfg, "D-dev-select", assert_anchors=not dry_run)
    if dry_run:
        pop = pop.subsample(2)
    log.info("  population: %s", pop.counts)

    # ── Baseline decode ────────────────────────────────────────
    log.info("running baseline decode")
    from steer_sweep.decode import decode_population
    baseline_result = decode_population(
        bundle, pop, language="zh", num_beams=1, batch_size=BATCH_SIZE, log=log,
    )
    baseline_texts = baseline_result.texts
    log.info("  baseline: %d utterances decoded in %.1fs",
             len(baseline_texts), baseline_result.wall_clock_sec)

    # Baseline metrics
    baseline_metrics = evaluate_method(bundle, pop, baseline_texts, baseline_texts)
    baseline_metrics["n_params"] = 0
    baseline_metrics["decoding"] = "greedy"
    log.info("  baseline MER: %.4f  PIER: %.4f", baseline_metrics["MER"],
             baseline_metrics["PIER"])
    atomic_json(RESULTS_DIR / "baseline_metrics.json", baseline_metrics)

    # ── Select training data ──────────────────────────────────
    train_examples = select_training_data(bundle, cfg, dry_run=dry_run)
    dev_examples = load_dev_examples(bundle, cfg, dry_run=dry_run)

    # Save selected IDs
    train_ids = [e.utterance_id for e in train_examples]
    atomic_json(RESULTS_DIR / "train_utterance_ids.json", {
        "seed": SEED,
        "n": len(train_ids),
        "utterance_ids": train_ids,
    })

    # ── Train methods in priority order ────────────────────────
    all_results: dict[str, dict[str, Any]] = {
        "Whisper baseline (greedy)": baseline_metrics,
    }
    trained: dict[str, TrainState] = {}
    methods = ["T1", "T2", "T3"]

    for method in methods:
        if budget.exhausted():
            log.warning("budget exhausted before %s", method)
            break

        state = train_method(
            bundle, cfg, pop, train_examples, dev_examples,
            baseline_texts, v_nat, v_prompt, method, budget,
            dry_run=dry_run,
        )

        if state is not None and state.best_epoch >= 0:
            trained[method] = state

            # Get best epoch result for summary
            best_epoch_result = state.epoch_results[state.best_epoch - 1]
            n_params = best_epoch_result.get("n_params", 0)
            if method == "T1":
                n_params = Rank2LocalSteering(D_MODEL, v_nat, v_prompt).n_trainable()
            elif method == "T2":
                n_params = SALSAD(D_MODEL).n_trainable()
            elif method == "T3":
                # Compute LoRA params: 2 * (rank * in + rank * out) for q and v
                q_in = bundle.decoder_layer(DECODER_LAYER).self_attn.q_proj.in_features
                q_out = bundle.decoder_layer(DECODER_LAYER).self_attn.q_proj.out_features
                v_in = bundle.decoder_layer(DECODER_LAYER).self_attn.v_proj.in_features
                v_out = bundle.decoder_layer(DECODER_LAYER).self_attn.v_proj.out_features
                n_params = 1 * q_in + q_out * 1 + 1 * v_in + v_out * 1

            label_map = {
                "T1": "T1: Ours-local",
                "T2": "T2: SALSA-D",
                "T3": "T3: LoRA-PM",
            }
            label = label_map.get(method, method) + " (greedy)"
            result = {**best_epoch_result, "n_params": n_params, "decoding": "greedy"}
            all_results[label] = result

    # ── T1 forced-global evaluation ────────────────────────────
    if "T1" in trained and not budget.exhausted():
        fg_result = t1_forced_global_eval(
            bundle, pop, baseline_texts, trained["T1"], v_nat, v_prompt,
        )
        if fg_result is not None:
            fg_result["n_params"] = Rank2LocalSteering(D_MODEL, v_nat, v_prompt).n_trainable()
            fg_result["decoding"] = "greedy"
            all_results["T1: Ours-global (greedy)"] = fg_result

    # ── Beam-5 final pass ──────────────────────────────────────
    for method in methods:
        if method not in trained or budget.exhausted():
            continue
        beam_result = beam5_eval(
            bundle, cfg, pop, baseline_texts, method, trained[method],
            v_nat, v_prompt,
        )
        if beam_result is not None:
            label_map = {
                "T1": "T1: Ours-local",
                "T2": "T2: SALSA-D",
                "T3": "T3: LoRA-PM",
            }
            label = label_map.get(method, method) + " (beam5)"
            # Get param count from greedy entry
            greedy_label = label_map.get(method, method) + " (greedy)"
            n_params = all_results.get(greedy_label, {}).get("n_params", 0)
            beam_result["n_params"] = n_params
            all_results[label] = beam_result

    # ── Beam-5 baseline ────────────────────────────────────────
    if not budget.exhausted():
        log.info("Beam-5 baseline decode")
        beam_baseline = decode_population(
            bundle, pop, language="zh", num_beams=5, batch_size=4, log=log,
        )
        beam_base_metrics = evaluate_method(bundle, pop, beam_baseline.texts,
                                            baseline_texts)
        beam_base_metrics["n_params"] = 0
        beam_base_metrics["decoding"] = "beam5"
        all_results["Whisper baseline (beam5)"] = beam_base_metrics

    # ── SEAME pilot ────────────────────────────────────────────
    seame_root = find_seame()
    if seame_root is not None and not budget.exhausted():
        log.info("SEAME data found at %s", seame_root)
        for method in methods:
            if method in trained and not budget.exhausted():
                seame_result = seame_pilot(
                    bundle, method, trained[method], v_nat, v_prompt,
                    seame_root, budget,
                )
                # Would add SEAME-M and SEAME-S columns if available

    # ── Summary ────────────────────────────────────────────────
    print_summary_table(all_results)

    # Save summary
    summary = {
        "probe": probe_result,
        "baseline": baseline_metrics,
        "methods": {},
        "budget_elapsed_hours": budget.elapsed_hours,
    }
    for method in methods:
        if method in trained:
            st = trained[method]
            summary["methods"][method] = {
                "best_epoch": st.best_epoch,
                "best_mer": st.best_mer,
                "epochs_completed": len(st.epoch_results),
                "epoch_results": st.epoch_results,
            }

    summary["results_table"] = {k: {kk: vv for kk, vv in v.items()
                                     if kk != "gate_diagnostics"}
                                for k, v in all_results.items()}

    atomic_json(RESULTS_DIR / "summary.json", summary)
    log.info("results saved to %s", RESULTS_DIR)
    log.info("total wall time: %.1f min", budget.elapsed_hours * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
