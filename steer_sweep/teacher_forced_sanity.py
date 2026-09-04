"""Small, model-agnostic teacher-forced loss diagnostic.

The model-specific runner supplies logits and labels.  Keeping the arithmetic
here makes the once-only shift and four-token ignore convention testable on CPU.
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F


def shifted_targets(labels: torch.Tensor, prefix_width: int = 4) -> torch.Tensor:
    out = labels.clone()
    out[..., :int(prefix_width)] = -100
    return out


def manual_loss(logits: torch.Tensor, labels: torch.Tensor, prefix_width: int = 4) -> torch.Tensor:
    """CE for logits already aligned to the same label positions."""
    target = shifted_targets(labels, prefix_width)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)


def top1_diagnostics(logits: torch.Tensor, labels: torch.Tensor, prefix_width: int = 4) -> dict[str, Any]:
    target = shifted_targets(labels, prefix_width)
    valid = target != -100
    logp = logits.log_softmax(-1)
    gold = target.clamp_min(0).unsqueeze(-1)
    gold_logp = logp.gather(-1, gold).squeeze(-1)
    top2 = logits.topk(2, dim=-1).values
    rank = (logits > logits.gather(-1, gold)).sum(-1) + 1
    return {
        "valid_tokens": int(valid.sum()),
        "top1_accuracy": float(((logits.argmax(-1) == target) & valid).sum() / valid.sum().clamp_min(1)),
        "gold_probability": float((gold_logp.exp()[valid]).mean()) if bool(valid.any()) else float("nan"),
        "gold_margin": float((logits.gather(-1, gold).squeeze(-1) - top2[..., 1])[valid].mean()) if bool(valid.any()) else float("nan"),
        "gold_rank": float(rank[valid].float().mean()) if bool(valid.any()) else float("nan"),
        "ignored_prefix": int((~valid).sum()),
    }


def compare_manual_builtin(logits: torch.Tensor, labels: torch.Tensor,
                           builtin_loss: torch.Tensor, prefix_width: int = 4,
                           tolerance: float = 1e-5) -> dict[str, Any]:
    manual = manual_loss(logits, labels, prefix_width)
    diff = float(abs(manual.detach() - builtin_loss.detach()))
    if diff > tolerance:
        raise AssertionError(f"manual/built-in CE disagreement {diff} > {tolerance}")
    return {"manual_loss": float(manual), "builtin_loss": float(builtin_loss),
            "absolute_difference": diff, **top1_diagnostics(logits, labels, prefix_width)}
