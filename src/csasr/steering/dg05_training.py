"""DG-05 correction-only training data and loss helpers.

The baseline-wrong labels are an explicit upstream artifact.  This avoids
silently deriving a training target from teacher-forced predictions or falling
back to all-token CE.  The artifact is expected to be made from the frozen
backbone's free-decoding baseline plus the canonical token/candidate alignment
on ``loc-train ∪ util-train``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


CORRECTION_SET_SCHEMA_VERSION = "dg05_correction_set_v1"
LID_EN = 1
IGNORE_INDEX = -100


@dataclass(frozen=True)
class CorrectionSetIndex:
    """Immutable baseline-wrong embedded positions keyed by utterance id."""

    positions: Mapping[str, tuple[int, ...]]
    roles: tuple[str, ...]
    source: str
    schema_version: str = CORRECTION_SET_SCHEMA_VERSION

    @property
    def n_positions(self) -> int:
        return sum(len(v) for v in self.positions.values())


def load_correction_set(path: str | Path, *, allowed_roles: Sequence[str] = ("loc-train", "util-train")) -> CorrectionSetIndex:
    """Load and validate the explicit ``C_E`` artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CORRECTION_SET_SCHEMA_VERSION:
        raise ValueError("correction-set artifact schema is not DG-05 v1")
    roles = tuple(sorted(str(x) for x in payload.get("roles", [])))
    if roles != tuple(sorted(str(x) for x in allowed_roles)):
        raise ValueError(f"correction-set roles {roles} do not match {tuple(allowed_roles)}")
    raw = payload.get("baseline_wrong_embedded_positions")
    if not isinstance(raw, dict):
        raise ValueError("missing baseline_wrong_embedded_positions mapping")
    positions: dict[str, tuple[int, ...]] = {}
    for uid, values in raw.items():
        vals = tuple(sorted(set(int(v) for v in values)))
        if any(v < 0 for v in vals):
            raise ValueError(f"negative token position for {uid}")
        positions[str(uid)] = vals
    return CorrectionSetIndex(positions=positions, roles=roles,
                              source=str(payload.get("source", path)),
                              schema_version=payload["schema_version"])


def correction_target_mask(utterance_ids: Sequence[str], token_ids: torch.Tensor,
                           lid_token_labels: torch.Tensor, correction_set: CorrectionSetIndex,
                           *, prefix_width: int) -> torch.Tensor:
    """Return a ``(B,T)`` mask for exactly baseline-wrong embedded targets.

    Positions are indexed in the example's full ``token_ids`` sequence.  The
    forced prefix, unknown-language tokens, padding, and all non-C_E positions
    are excluded.  This mask is used on shifted model targets at ``[:, 1:]``.
    """
    if token_ids.ndim != 2 or lid_token_labels.shape != token_ids.shape:
        raise ValueError("token_ids and lid_token_labels must have the same (B,T) shape")
    if len(utterance_ids) != token_ids.shape[0]:
        raise ValueError("utterance_ids do not match batch size")
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for row, uid in enumerate(utterance_ids):
        for pos in correction_set.positions.get(str(uid), ()):
            if pos >= token_ids.shape[1]:
                raise ValueError(f"C_E position {pos} exceeds padded width for {uid}")
            if pos < int(prefix_width):
                raise ValueError(f"C_E contains forced-prefix position {pos} for {uid}")
            label = int(lid_token_labels[row, pos])
            if label != LID_EN:
                raise ValueError(
                    f"C_E position {pos} for {uid} is not an embedded-language token")
            mask[row, pos] = True
    if not bool(mask.any()):
        raise ValueError("C_E is empty after embedded-language/prefix filtering")
    return mask


def correction_only_loss(logits: torch.Tensor, labels: torch.Tensor,
                         target_mask: torch.Tensor) -> torch.Tensor:
    """Compute CE only on ``C_E``; no retention or all-token fallback."""
    if logits.ndim != 3 or labels.ndim != 2 or target_mask.shape != labels.shape:
        raise ValueError("expected logits (B,T,V), labels/mask (B,T)")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence shapes differ")
    if not bool(target_mask.any()):
        raise ValueError("correction-only loss received an empty C_E mask")
    selected = target_mask & (labels != IGNORE_INDEX)
    if not bool(selected.any()):
        raise ValueError("C_E contains no valid target labels")
    return F.cross_entropy(logits[selected], labels[selected])


def correction_set_payload(positions: Mapping[str, Sequence[int]], *, source: str,
                           roles: Sequence[str] = ("loc-train", "util-train"),
                           exclusions: Sequence[str] = ()) -> dict[str, Any]:
    """Create the deterministic JSON payload expected by the loader."""
    normalized = {str(uid): sorted(set(int(x) for x in vals))
                  for uid, vals in sorted(positions.items())}
    return {
        "schema_version": CORRECTION_SET_SCHEMA_VERSION,
        "roles": sorted(str(x) for x in roles),
        "source": str(source),
        "baseline_wrong_embedded_positions": normalized,
        "n_utterances": len(normalized),
        "n_positions": sum(len(v) for v in normalized.values()),
        "exclusions": list(exclusions),
        "semantics": "C_E = baseline-wrong embedded-language positions; labels are not controller inputs",
    }


def build_correction_set(records: Sequence[Mapping[str, Any]], *, source: str,
                         allowed_roles: Sequence[str] = ("loc-train", "util-train")) -> dict[str, Any]:
    """Construct ``C_E`` from canonical baseline/alignment records.

    Each record must explicitly carry its role, full decoder token position,
    embedded-language label, and baseline correctness.  Missing fields are an
    error rather than an invitation to treat every token as a correction.
    """
    allowed = set(str(x) for x in allowed_roles)
    positions: dict[str, list[int]] = {}
    exclusions: list[str] = []
    for row in records:
        uid = str(row.get("utterance_id", ""))
        if not uid or "role" not in row or "token_position" not in row:
            raise ValueError("C_E records require utterance_id, role, and token_position")
        role = str(row["role"])
        if role not in allowed:
            raise ValueError(f"C_E record for {uid} has disallowed role {role}")
        if "baseline_correct" not in row or "language" not in row:
            raise ValueError("C_E records require baseline_correct and language labels")
        if bool(row["baseline_correct"]) is False and str(row["language"]).upper() == "EN":
            pos = int(row["token_position"])
            if pos < 0:
                raise ValueError("C_E token_position must be non-negative")
            positions.setdefault(uid, []).append(pos)
        else:
            exclusions.append(f"{uid}:{int(row['token_position'])}")
    if not positions:
        raise ValueError("canonical baseline/alignment records produced an empty C_E")
    return correction_set_payload(positions, source=source,
                                  roles=tuple(sorted(allowed)), exclusions=exclusions)
