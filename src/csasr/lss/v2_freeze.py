"""Build the dialogue-atomic v2 spec payload from immutable evidence."""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

from ..utils.config import REPO_ROOT
from ..utils.hashing import sha256_file
from .align.devselect import scientific_amendment
from .artifacts import artifact_ref
from .balance import RNG_DERIVATION
from .specfreeze import SpecFreeze, load
from .v2_namespace import V2Paths, configured_paths


class V2FreezeError(RuntimeError):
    """The v2 role evidence cannot support the requested supersession."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V2FreezeError(f"cannot read required v2 evidence: {path}") from exc
    if not isinstance(value, dict):
        raise V2FreezeError(f"required v2 evidence is not an object: {path}")
    return value


def _role_artifacts(paths: V2Paths, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    roles = dict(report.get("roles") or {})
    for role in sorted(roles):
        path = paths.role_root / ("locked" if role == "D-test" else "") \
            / f"role_{role}.parquet"
        if not path.is_file():
            raise V2FreezeError(f"role artifact named by report is missing: {path}")
        refs.append(artifact_ref(
            path, schema="lss_role_manifest_v2_dialogue",
            rows=int((roles.get(role) or {}).get("utterances", 0))))
    return refs


def build_v2_spec(cfg: Mapping[str, Any]) -> SpecFreeze:
    """Clone v1 decisions, replacing only the formally superseded sections."""
    paths = configured_paths(cfg)
    old = load(paths.superseded_freeze)
    role_report_path = paths.role_root / "dialogue_role_report.json"
    generation_path = paths.role_root / "generation_complete.manifest.json"
    dialogue_manifest = paths.namespace_root / "manifests" / \
        "cs_dialogue_with_dialogue_id.parquet"
    pairing_evidence = paths.namespace_root / "manifests" / \
        "dialogue_pairing_evidence.json"
    report = _read_json(role_report_path)
    generation = _read_json(generation_path)

    if not generation.get("complete") or generation.get("state") != "completed":
        raise V2FreezeError("dialogue role generation is not complete")
    for key in ("role_assignment_hash", "role_partition_fingerprint"):
        report_key = "assignment_hash" if key == "role_assignment_hash" else \
            "partition_fingerprint"
        if generation.get(key) != report.get(report_key):
            raise V2FreezeError(f"role generation/report disagree on {key}")
    structure = dict(report.get("structure") or {})
    required_structure = (
        "dialogue_disjoint", "conversation_disjoint",
        "every_dialogue_in_exactly_one_role", "d_test_is_official_test_only",
    )
    if not all(bool(structure.get(key)) for key in required_structure):
        raise V2FreezeError(
            "role evidence does not satisfy every dialogue/split invariant")

    dialogue_sidecar = _read_json(Path(str(dialogue_manifest) + ".manifest.json"))
    derivation_fingerprint = str(
        dialogue_sidecar.get("dialogue_derivation_fingerprint") or "")
    if not derivation_fingerprint:
        raise V2FreezeError("dialogue derivation fingerprint is missing")

    balance = dict(report.get("balance") or {})
    partition = dict(report.get("partition_fingerprint_payload") or {})
    balance_spec = dict(partition.get("balance") or {})
    # These values are constraints, not knobs to make the observed allocation
    # pass. They must agree in config, fingerprint, and the accepted report.
    configured_balance = dict((cfg.get("roles") or {}).get("balance") or {})
    for key in ("max_abs_smd", "max_categorical_tv"):
        if float(balance_spec.get(key)) != float(configured_balance.get(key)):
            raise V2FreezeError(f"configured and fingerprinted {key} disagree")
    if float(balance_spec.get("ungated_weight")) != float(
            (((cfg.get("roles") or {}).get("dialogue_roles") or {})
             .get("partition") or {}).get("ungated_weight")):
        raise V2FreezeError("configured and fingerprinted ungated_weight disagree")

    amendment = scientific_amendment(cfg)
    amendment_path = Path(amendment["document"])
    if not amendment_path.is_absolute():
        amendment_path = REPO_ROOT / amendment_path

    payload = copy.deepcopy(old.payload)
    payload["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload["statistics"] = {
        **dict(payload.get("statistics") or {}),
        "cluster_key": "dialogue_id",
        "cluster_justification": (
            "CS-Dialogue has 100 two-party recording sessions and 200 speaker "
            "channels. The paired channels share topic, room, time, and "
            "mutually conditioned speech; dialogue_id, derived from the integer "
            "Speaker ID pairing and validated 100/100 on group size, topic set, "
            "and official split, is therefore the independence unit."),
    }
    targets = {role: int(values["dialogues"])
               for role, values in sorted((report.get("roles") or {}).items())}
    payload["roles"] = {
        "version": str(report.get("role_version")),
        "allocation_unit": "dialogue",
        "targets": targets,
        "assignment_hash": str(report.get("assignment_hash")),
        "dialogue_derivation_fingerprint": derivation_fingerprint,
        "role_partition_fingerprint": str(report.get("partition_fingerprint")),
        "role_partition_fingerprint_payload": partition,
        "structure": structure,
        "balance": {
            "constraints": {
                "max_abs_smd": float(balance_spec["max_abs_smd"]),
                "max_categorical_tv": float(balance_spec["max_categorical_tv"]),
                "gated_categorical": list(balance_spec["gated_categorical"]),
            },
            "objective": str(balance_spec["scoring_formula"]),
            "ungated_weight": float(balance_spec["ungated_weight"]),
            "proposal_mechanism": balance.get("accepted_proposal_mechanism"),
            "feasible_draws": int(balance.get("feasible_draws", 0)),
            "draws": int(balance.get("draws", 0)),
            "accepted_max_abs_smd": float(balance.get("accepted_max_abs_smd")),
            "accepted_max_categorical_tv": float(
                balance.get("accepted_max_categorical_tv")),
            "acceptance_rule": (
                "all gated criteria are feasibility constraints; the unchanged "
                "objective ranks feasible draws only; no infeasible fallback"),
            # The acceptance rule was not the only thing that changed. The
            # proposal machinery was rebuilt, and with it the draw stream, so
            # the same seed no longer enumerates the same candidates.
            "rng_derivation": dict(RNG_DERIVATION),
            "what_changed_from_v1": (
                "two things, not one: (1) the draw-acceptance rule was corrected "
                "so gated SMD and TV are feasibility constraints rather than "
                "post-hoc reported values, and (2) the proposal machinery was "
                "rebuilt -- a stratified deal was added alongside the uniform "
                "permutation, and the RNG derivation changed from "
                "default_rng(seed) to SeedSequence([seed, proposal_index, "
                "0xC5A5]). No threshold, scoring formula, or ungated weight "
                "changed."),
            # The request called this an amended threshold, but the authorized
            # repair explicitly kept 0.15 and 0.25. A freeze must not invent a
            # nonexistent amendment document or describe unchanged values as a
            # post-hoc relaxation.
            "threshold_amendment": {
                "status": "none_thresholds_unchanged",
                "document": None,
                "max_abs_smd": float(balance_spec["max_abs_smd"]),
                "max_categorical_tv": float(balance_spec["max_categorical_tv"]),
                "reason": (
                    "No balance threshold was amended. v2 changes the "
                    "draw-acceptance rule and rebuilds the proposal machinery "
                    "-- including the RNG derivation -- not a numeric "
                    "criterion. The v2 allocation is therefore a fresh "
                    "allocation, not a counterfactual reproduction of v1 under "
                    "a different threshold."),
            },
        },
    }
    payload["scientific_amendments"] = {
        "gate_a_automatic_instrument": amendment,
    }
    payload["dialogue_derivation"] = {
        "fingerprint": derivation_fingerprint,
        "fingerprint_payload": dialogue_sidecar.get(
            "dialogue_derivation_fingerprint_payload"),
        "pairing_evidence": str(pairing_evidence),
    }
    payload["output_namespace"] = {
        "version": "lss-dialogue-v2",
        **paths.to_dict(),
        "candidate_generation_is_write_once": True,
        "candidate_generation_only": True,
    }
    payload["candidate_generation"] = copy.deepcopy(
        dict(cfg.get("candidate_generation") or {}))

    refs = [
        artifact_ref(dialogue_manifest,
                     schema="cs_dialogue_manifest_with_dialogue_id_v1"),
        artifact_ref(pairing_evidence, schema="dialogue_pairing_evidence_v1"),
        artifact_ref(role_report_path, schema="lss_dialogue_role_report_v2"),
        artifact_ref(generation_path, schema="lss_dialogue_role_generation_v1"),
        artifact_ref(amendment_path, schema="scientific_amendment_markdown"),
        *_role_artifacts(paths, report),
    ]
    payload["referenced_artifacts"] = refs
    # `supersedes` carries the v1 lineage and is sealed by supersede_at().  A
    # replacement v2 freeze also supersedes the *previous v2 freeze*, which that
    # one-step pointer cannot express, so the second link is recorded here.
    previous_v2 = str((cfg.get("v2_namespace") or {}).get("superseded_v2_freeze") or "")
    if previous_v2:
        previous_path = Path(previous_v2)
        if not previous_path.is_file():
            raise V2FreezeError(
                f"superseded v2 freeze not found: {previous_path}")
        superseded = load(previous_path)
        payload["supersedes_v2_freeze"] = {
            "path": str(previous_path),
            "sha256": superseded.sha256,
            "file_sha256": sha256_file(previous_path),
            "role_partition_fingerprint": (superseded.payload.get("roles") or {}
                                           ).get("role_partition_fingerprint"),
            "reason": (
                "the previous v2 freeze predates the rng_derivation record, so "
                "it carries the pre-record partition fingerprint. The allocation "
                "is unchanged and was verified bit-identical; only the recorded "
                "identity of the partition rule differs."),
            "superseded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    payload["supersedes"] = None  # filled and sealed only by supersede_at()
    payload["sha256"] = ""
    return SpecFreeze(payload=payload)
