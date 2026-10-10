"""Deterministic ASCEND subset freezing and manifest validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .data import CanonicalUtterance, ELIGIBILITY_RULE_HASH, ELIGIBILITY_RULE_VERSION, eligibility_record
from ..utils.hashing import sha256_obj

SEED = 42
STRATIFICATION_KEYS = ("speaker_id", "session_id", "topic")


def _sort_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def deterministic_stratified_select(items: Iterable[CanonicalUtterance], target: int,
                                    *, seed: int = SEED) -> list[CanonicalUtterance]:
    eligible = [x for x in items if x.cs_eligible]
    if target < 0:
        raise ValueError(target)
    if not eligible or target == 0:
        return []
    target = min(int(target), len(eligible))
    groups: dict[tuple[str, str, str], list[CanonicalUtterance]] = {}
    for item in eligible:
        key = tuple(getattr(item, name) or "<missing>" for name in STRATIFICATION_KEYS)
        groups.setdefault(key, []).append(item)
    for values in groups.values():
        values.sort(key=lambda x: _sort_key(seed, x.utterance_id))
    # Round-robin over deterministic stratum order gives every available
    # speaker/session/topic stratum a chance before filling remaining slots.
    strata = sorted(groups, key=lambda k: _sort_key(seed, "|".join(k)))
    selected: list[CanonicalUtterance] = []
    cursor = 0
    while len(selected) < target:
        progressed = False
        for stratum in strata:
            values = groups[stratum]
            if cursor < len(values):
                selected.append(values[cursor]); progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected, key=lambda x: x.utterance_id)


def build_subset_manifest(items: Iterable[CanonicalUtterance], *, split: str, role: str,
                          target: int, seed: int = SEED, dataset_fingerprint: str | None = None) -> dict:
    items = list(items)
    rows = deterministic_stratified_select(items, target, seed=seed)
    ids = [x.utterance_id for x in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate ASCEND IDs")
    payload = {
        "schema_version": "basis_a6_ascend_subset_manifest_v1", "role": role,
        "source_split": split, "seed": seed, "stratification_keys": list(STRATIFICATION_KEYS),
        "eligibility_rule_version": ELIGIBILITY_RULE_VERSION, "eligibility_rule_hash": ELIGIBILITY_RULE_HASH,
        "dataset_fingerprint": dataset_fingerprint, "utterance_ids": ids,
        "items": [eligibility_record(x) for x in rows],
        "aggregate": {"N": len(rows), "N_scanned": len(items),
                      "duration_sec": sum(float(x.duration_sec or 0) for x in rows),
                      "en_units": sum(x.en_units for x in rows), "zh_units": sum(x.zh_units for x in rows)},
    }
    payload["fingerprint"] = "sha256:" + hashlib.sha256(json.dumps(
        {"utterance_ids": ids, "eligibility_rule_hash": ELIGIBILITY_RULE_HASH},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def assert_disjoint(construct: dict, evaluation: dict) -> None:
    a = set(construct["utterance_ids"]); b = set(evaluation["utterance_ids"])
    if a & b:
        raise AssertionError(f"ASCEND construct/eval overlap: {sorted(a & b)[:5]}")
    if construct.get("source_split") != "train" or evaluation.get("source_split") != "validation":
        raise AssertionError("ASCEND roles must be train-only construct and validation-only eval")


def write_manifests(output_dir: str | Path, construct: dict, evaluation: dict) -> None:
    assert_disjoint(construct, evaluation)
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    for name, value in (("ASCEND_CONSTRUCT_MANIFEST.json", construct), ("ASCEND_EVAL_MANIFEST.json", evaluation)):
        (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
