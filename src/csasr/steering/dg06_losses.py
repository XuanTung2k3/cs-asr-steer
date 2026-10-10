"""DG-06 damage-aware retention populations and losses.

This module adds the MC §8 retention machinery on top of the frozen DG-05
correction-only path:

- ``RetentionSetIndex`` — baseline-**correct** embedded (``R_E``) or matrix
  (``R_M``) token positions, keyed by utterance id (same alignment/population
  semantics as the frozen DG-05 ``C_E``);
- ``retention_position_mask`` — a ``(B,T)`` mask for exactly one retention
  population, empty-safe (no correction fired there);
- ``kl_retention_loss`` — ``D_KL(p_0 ‖ p_theta)`` (this exact direction) at the
  masked positions, gathered before the softmax for memory safety;
- ``damage_aware_loss`` — the D1/D2 loss composition over a shared method
  forward pass and a frozen-baseline forward pass;
- ``baseline_provenance`` — the provenance descriptor for ``p_0``.

The correction-set-only cross-entropy is reused unchanged from ``dg05_training``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .dg05_training import IGNORE_INDEX, LID_EN, correction_only_loss

RETENTION_SET_SCHEMA_VERSION = "dg06_retention_set_v1"
LID_ZH = 0
LID_LABEL = {"EN": LID_EN, "ZH": LID_ZH}


@dataclass(frozen=True)
class RetentionSetIndex:
    """Immutable baseline-correct positions for one retention population."""

    positions: Mapping[str, tuple[int, ...]]
    language: str                       # "EN" (R_E) or "ZH" (R_M)
    roles: tuple[str, ...]
    source: str
    schema_version: str = RETENTION_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.language not in LID_LABEL:
            raise ValueError(f"retention language must be EN or ZH, got {self.language}")

    @property
    def required_label(self) -> int:
        return LID_LABEL[self.language]

    @property
    def n_positions(self) -> int:
        return sum(len(v) for v in self.positions.values())


def retention_position_mask(utterance_ids: Sequence[str], token_ids: torch.Tensor,
                            lid_token_labels: torch.Tensor, retention_set: RetentionSetIndex,
                            *, prefix_width: int) -> torch.Tensor:
    """Return a ``(B,T)`` mask for exactly one retention population.

    Positions are indexed in the example's full ``token_ids`` sequence and must
    carry the population's language label; the forced prefix is excluded. Unlike
    ``correction_target_mask`` an **empty** population is permitted (a batch may
    contain no retained unit for this language) and yields an all-false mask.
    The mask is meant to be applied to shifted targets at ``[:, 1:]``.
    """
    if token_ids.ndim != 2 or lid_token_labels.shape != token_ids.shape:
        raise ValueError("token_ids and lid_token_labels must have the same (B,T) shape")
    if len(utterance_ids) != token_ids.shape[0]:
        raise ValueError("utterance_ids do not match batch size")
    required = retention_set.required_label
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for row, uid in enumerate(utterance_ids):
        for pos in retention_set.positions.get(str(uid), ()):
            if pos >= token_ids.shape[1]:
                raise ValueError(f"retention position {pos} exceeds padded width for {uid}")
            if pos < int(prefix_width):
                raise ValueError(f"retention set contains forced-prefix position {pos} for {uid}")
            if int(lid_token_labels[row, pos]) != required:
                raise ValueError(
                    f"retention position {pos} for {uid} is not a {retention_set.language} token")
            mask[row, pos] = True
    return mask


def kl_retention_loss(method_logits: torch.Tensor, baseline_logits: torch.Tensor,
                      target_mask: torch.Tensor) -> torch.Tensor:
    """``mean_t D_KL(p_0,t ‖ p_theta,t)`` at masked positions (this direction).

    ``method_logits``/``baseline_logits`` are ``(B,T,V)`` aligned so that index
    ``t`` predicts target ``t`` (i.e. already sliced ``[:, :-1]``); ``target_mask``
    is ``(B,T)``. Positions are gathered **before** the softmax so only the
    handful of retained rows ever materialise a vocab-wide distribution.
    ``D_KL(p‖p)=0`` and an empty mask returns a differentiable zero.
    """
    if method_logits.ndim != 3 or baseline_logits.shape != method_logits.shape:
        raise ValueError("expected method/baseline logits of equal shape (B,T,V)")
    if target_mask.shape != method_logits.shape[:2]:
        raise ValueError("target_mask must be (B,T) matching the logits")
    if not bool(target_mask.any()):
        return method_logits.sum() * 0.0
    m = method_logits[target_mask]                    # (N, V) — carries controller grad
    b = baseline_logits[target_mask].detach()         # (N, V) — frozen baseline p_0
    log_p_theta = F.log_softmax(m, dim=-1)
    log_p0 = F.log_softmax(b, dim=-1)
    p0 = log_p0.exp()
    kl = (p0 * (log_p0 - log_p_theta)).sum(dim=-1)    # KL(p0 || p_theta) per position
    return kl.mean()


def damage_aware_loss(method_logits: torch.Tensor, baseline_logits: torch.Tensor,
                      labels: torch.Tensor, correction_mask: torch.Tensor,
                      matrix_mask: torch.Tensor,
                      embedded_mask: torch.Tensor | None,
                      *, lambda_m: float, lambda_e: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose the D1 (embedded_mask=None) or D2 loss over shared forward passes.

    Returns ``(total, components)`` where ``components`` holds the raw/unweighted
    ``l_corr``, ``l_ret_m`` and (D2) ``l_ret_e`` plus population counts, so the
    relative influence of ``lambda=1`` terms stays interpretable. A batch without
    any ``C_E`` position contributes zero correction CE (no all-token fallback).
    """
    if bool(correction_mask.any()):
        l_corr = correction_only_loss(method_logits, labels, correction_mask)
    else:
        l_corr = method_logits.sum() * 0.0
    l_ret_m = kl_retention_loss(method_logits, baseline_logits, matrix_mask)
    total = l_corr + float(lambda_m) * l_ret_m
    components: dict[str, float] = {
        "l_corr": float(l_corr.detach()),
        "l_ret_m": float(l_ret_m.detach()),
        "n_corr": int(correction_mask.sum()),
        "n_ret_m": int(matrix_mask.sum()),
        "lambda_m": float(lambda_m),
    }
    if embedded_mask is not None:
        l_ret_e = kl_retention_loss(method_logits, baseline_logits, embedded_mask)
        total = total + float(lambda_e) * l_ret_e
        components["l_ret_e"] = float(l_ret_e.detach())
        components["n_ret_e"] = int(embedded_mask.sum())
        components["lambda_e"] = float(lambda_e)
    components["total"] = float(total.detach())
    return total, components


def baseline_provenance(*, model_id: str, model_revision: str | None,
                        training_role_fingerprint: str, config_hash: str,
                        tokenizer: str, git_commit: str) -> dict[str, Any]:
    """Provenance descriptor for the frozen-Whisper baseline distributions p_0."""
    return {
        "schema_version": "dg06_baseline_distribution_v1",
        "semantics": "frozen unsteered Whisper next-token distribution under identical "
                     "teacher-forced context; recomputed per batch (no on-disk full-vocab cache)",
        "kl_direction": "p0 || p_theta",
        "model_id": str(model_id),
        "model_revision": model_revision,
        "training_role_fingerprint": str(training_role_fingerprint),
        "config_hash": str(config_hash),
        "tokenizer": str(tokenizer),
        "git_commit": str(git_commit),
    }


def retention_set_payload(positions: Mapping[str, Sequence[int]], *, language: str,
                          source: str, roles: Sequence[str] = ("loc-train", "util-train"),
                          exclusions: Sequence[str] = ()) -> dict[str, Any]:
    """Deterministic JSON payload for one retention population."""
    if language not in LID_LABEL:
        raise ValueError("retention language must be EN or ZH")
    normalized = {str(uid): sorted(set(int(x) for x in vals))
                  for uid, vals in sorted(positions.items()) if vals}
    return {
        "schema_version": RETENTION_SET_SCHEMA_VERSION,
        "language": language,
        "roles": sorted(str(x) for x in roles),
        "source": str(source),
        "baseline_correct_positions": normalized,
        "n_utterances": len(normalized),
        "n_positions": sum(len(v) for v in normalized.values()),
        "exclusions": list(exclusions),
        "semantics": (f"R_{'E' if language == 'EN' else 'M'} = baseline-correct "
                      f"{language} positions; labels are not controller inputs"),
    }


def load_retention_set(payload: Mapping[str, Any], *,
                       allowed_roles: Sequence[str] = ("loc-train", "util-train")) -> RetentionSetIndex:
    """Load and validate one retention population from its JSON payload."""
    if payload.get("schema_version") != RETENTION_SET_SCHEMA_VERSION:
        raise ValueError("retention-set payload schema is not DG-06 v1")
    language = str(payload.get("language"))
    roles = tuple(sorted(str(x) for x in payload.get("roles", [])))
    if roles != tuple(sorted(str(x) for x in allowed_roles)):
        raise ValueError(f"retention-set roles {roles} do not match {tuple(allowed_roles)}")
    raw = payload.get("baseline_correct_positions")
    if not isinstance(raw, dict):
        raise ValueError("missing baseline_correct_positions mapping")
    positions: dict[str, tuple[int, ...]] = {}
    for uid, values in raw.items():
        vals = tuple(sorted(set(int(v) for v in values)))
        if any(v < 0 for v in vals):
            raise ValueError(f"negative token position for {uid}")
        positions[str(uid)] = vals
    return RetentionSetIndex(positions=positions, language=language, roles=roles,
                             source=str(payload.get("source", "")),
                             schema_version=payload["schema_version"])
