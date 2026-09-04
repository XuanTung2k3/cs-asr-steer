"""The dialogue-atomic v2 execution cannot alias or mutate v1."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csasr.lss.align.devselect import scientific_amendment
from csasr.lss.roles import role_path
from csasr.lss.specfreeze import SPEC_FREEZE_SCHEMA, SpecFreeze, seal
from csasr.lss.specfreeze import load as load_spec
from csasr.lss.v2_freeze import build_v2_spec
from csasr.lss.v2_namespace import (
    NamespaceIsolationError,
    configured_paths,
    validate_namespace,
)
from csasr.utils.config import REPO_ROOT, load_config
from csasr.utils.hashing import sha256_file


def _cfg(tmp_path: Path) -> dict:
    cfg = load_config("lss/l1b_candidates_dialogue_v2.yaml")
    v1 = tmp_path / "artifacts_lss"
    v2 = tmp_path / "artifacts_dialogue_v2"
    generation = v2 / "candidate_generations" / "generation_001"
    cfg["experiment"]["output_root"] = str(generation)
    cfg["experiment"]["role_manifests_root"] = str(v2 / "manifests" / "roles")
    cfg["experiment"]["spec_freeze_path"] = str(
        v2 / "freeze" / "spec_freeze_v2.json")
    cfg["v2_namespace"] = {
        "version": "lss-dialogue-v2",
        "root": str(v2),
        "protected_v1_root": str(v1),
        "superseded_freeze": str(v1 / "freeze" / "spec_freeze_v1.json"),
        "role_root": str(v2 / "manifests" / "roles"),
        "candidate_generation_root": str(generation),
        "candidate_path": str(generation / "alignments" /
                              "candidates_all.parquet"),
        "freeze_path": str(v2 / "freeze" / "spec_freeze_v2.json"),
    }
    return cfg


def _role_generation(cfg: dict) -> Path:
    root = Path(cfg["v2_namespace"]["role_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "generation_complete.manifest.json"
    path.write_text(json.dumps({
        "complete": True,
        "state": "completed",
        "role_assignment_hash": "assignment",
        "role_partition_fingerprint": "partition",
    }), encoding="utf-8")
    return path


def test_real_production_config_resolves_the_real_amendment_document():
    """No temporary override may hide a missing production document."""
    cfg = load_config("lss/l1b_valid.yaml")
    record = scientific_amendment(cfg)
    path = Path(record["document"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    assert path.is_file()
    assert record["sha256"] == sha256_file(path)
    assert record["status"] == "superseded_pending_genuine_lexical_reference"


def test_default_and_v2_role_and_candidate_paths_are_disjoint():
    default = load_config("lss/l1b_valid.yaml")
    versioned = load_config("lss/l1b_candidates_dialogue_v2.yaml")
    paths = configured_paths(versioned)

    assert default["experiment"]["output_root"].endswith("/artifacts_lss")
    assert role_path(default, "D-construct") == Path(
        default["experiment"]["output_root"]) / "manifests/roles" / \
        "role_D-construct.parquet"
    assert role_path(versioned, "D-construct") == paths.role_root / \
        "role_D-construct.parquet"
    assert paths.candidate_path == paths.candidate_generation_root / \
        "alignments/candidates_all.parquet"
    assert paths.protected_v1_root not in paths.candidate_path.parents


def test_namespace_refuses_a_candidate_destination_under_v1(tmp_path):
    cfg = _cfg(tmp_path)
    v1 = Path(cfg["v2_namespace"]["protected_v1_root"])
    malicious = v1 / "candidate_generations" / "generation_001"
    cfg["v2_namespace"]["candidate_generation_root"] = str(malicious)
    cfg["v2_namespace"]["candidate_path"] = str(
        malicious / "alignments/candidates_all.parquet")
    cfg["experiment"]["output_root"] = str(malicious)
    with pytest.raises(NamespaceIsolationError, match="candidate_generation_root"):
        configured_paths(cfg)


def test_namespace_refuses_relative_and_symlink_aliases(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["v2_namespace"]["candidate_path"] = "relative/candidates_all.parquet"
    with pytest.raises(NamespaceIsolationError, match="absolute path"):
        configured_paths(cfg)

    cfg = _cfg(tmp_path)
    v1 = Path(cfg["v2_namespace"]["protected_v1_root"])
    v1.mkdir(parents=True)
    alias = tmp_path / "v1-alias"
    alias.symlink_to(v1, target_is_directory=True)
    cfg["v2_namespace"]["candidate_generation_root"] = str(alias / "generation_001")
    cfg["v2_namespace"]["candidate_path"] = str(
        alias / "generation_001/alignments/candidates_all.parquet")
    cfg["experiment"]["output_root"] = str(alias / "generation_001")
    with pytest.raises(NamespaceIsolationError, match="symlink or alias"):
        configured_paths(cfg)


def test_write_once_candidate_generation_refuses_even_a_partial_retry(tmp_path):
    cfg = _cfg(tmp_path)
    _role_generation(cfg)
    freeze = Path(cfg["v2_namespace"]["freeze_path"])
    freeze.parent.mkdir(parents=True)
    freeze.write_text("sealed", encoding="utf-8")
    assert validate_namespace(cfg, for_execution=True)["ok"]

    generation = Path(cfg["v2_namespace"]["candidate_generation_root"])
    generation.mkdir(parents=True)
    (generation / "partial.tmp").write_text("interrupted", encoding="utf-8")
    with pytest.raises(NamespaceIsolationError, match="immutable even when incomplete"):
        validate_namespace(cfg, for_execution=True)


def _seed_v2_evidence(cfg):
    """Minimal on-disk evidence a v2 freeze needs: v1 freeze, manifest, roles."""
    paths = configured_paths(cfg)
    paths.superseded_freeze.parent.mkdir(parents=True)

    # A real schema-valid v1 freeze is enough: v2 replaces the evidence-bearing
    # role/statistics sections while retaining every unrelated frozen decision.
    seal(SpecFreeze({
        "schema_version": SPEC_FREEZE_SCHEMA,
        "spec_version": "v1",
        "statistics": {"cluster_key": "conversation_id"},
        "roles": {"version": "v1"},
        "alignment_prereg": {},
        "referenced_artifacts": [],
        "supersedes": None,
        "sha256": "",
    }), paths.superseded_freeze)

    manifest = paths.namespace_root / "manifests/cs_dialogue_with_dialogue_id.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"dialogue manifest fixture")
    Path(str(manifest) + ".manifest.json").write_text(json.dumps({
        "dialogue_derivation_fingerprint": "derivation",
        "dialogue_derivation_fingerprint_payload": {"version": "fixture"},
    }), encoding="utf-8")
    (manifest.parent / "dialogue_pairing_evidence.json").write_text(
        json.dumps({"all_checks_pass": True}), encoding="utf-8")

    paths.role_root.mkdir(parents=True)
    roles = {"D-construct": {"dialogues": 20, "utterances": 1},
             "D-test": {"dialogues": 15, "utterances": 1}}
    for role in roles:
        path = paths.role_root / ("locked" if role == "D-test" else "") / \
            f"role_{role}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
    partition = {
        "balance": {
            "max_abs_smd": 0.25,
            "max_categorical_tv": 0.15,
            "gated_categorical": ["gender", "device"],
            "scoring_formula":
                "max(max_SMD, max_gated_TV) + ungated_weight*ungated_TV",
            "ungated_weight": 0.3,
        }
    }
    report = {
        "role_version": "dialogue-v2",
        "roles": roles,
        "assignment_hash": "assignment",
        "partition_fingerprint": "partition",
        "partition_fingerprint_payload": partition,
        "structure": {
            "dialogue_disjoint": True,
            "conversation_disjoint": True,
            "every_dialogue_in_exactly_one_role": True,
            "d_test_is_official_test_only": True,
        },
        "balance": {
            "accepted_proposal_mechanism": "stratified_deal",
            "feasible_draws": 2,
            "draws": 10,
            "accepted_max_abs_smd": 0.24,
            "accepted_max_categorical_tv": 0.14,
        },
    }
    (paths.role_root / "dialogue_role_report.json").write_text(
        json.dumps(report), encoding="utf-8")
    _role_generation(cfg)
    return paths


def test_v2_freeze_records_truthful_threshold_and_amendment_semantics(tmp_path):
    cfg = _cfg(tmp_path)
    paths = _seed_v2_evidence(cfg)

    spec = build_v2_spec(cfg)
    assert spec.get("statistics", "cluster_key") == "dialogue_id"
    assert spec.get("roles", "assignment_hash") == "assignment"
    assert spec.get("roles", "balance", "threshold_amendment", "status") == \
        "none_thresholds_unchanged"
    assert spec.get("roles", "balance", "threshold_amendment", "document") is None
    assert spec.get("scientific_amendments", "gate_a_automatic_instrument",
                    "document").endswith("GATE_A_AUTOMATIC_INSTRUMENT_AMENDMENT_2026-08-11.md")
    assert spec.get("output_namespace", "candidate_path") == str(paths.candidate_path)


def test_v2_freeze_records_the_previous_v2_freeze_it_replaces(tmp_path):
    """A replacement v2 freeze supersedes two things, and `supersedes` holds one.

    `supersede_at` records the v1 lineage and enforces vN -> vN+1, so it cannot
    also express "this v2 replaces that v2". Without the second pointer the
    chain from the pre-record artifacts to their correction would be unwritten.
    """
    from csasr.lss.balance import RNG_DERIVATION
    from csasr.lss.v2_freeze import V2FreezeError

    cfg = _cfg(tmp_path)
    _seed_v2_evidence(cfg)

    # no previous v2 freeze configured: the pointer is simply absent
    spec = build_v2_spec(cfg)
    assert "supersedes_v2_freeze" not in spec.payload
    assert spec.get("roles", "balance", "rng_derivation") == RNG_DERIVATION

    previous = tmp_path / "old_v2" / "spec_freeze_v2.json"
    previous.parent.mkdir(parents=True)
    seal(SpecFreeze({
        "schema_version": SPEC_FREEZE_SCHEMA,
        "spec_version": "v2",
        "statistics": {"cluster_key": "dialogue_id"},
        "roles": {"version": "dialogue-v2",
                  "role_partition_fingerprint": "the-old-fingerprint"},
        "alignment_prereg": {},
        "referenced_artifacts": [],
        "supersedes": None,
        "sha256": "",
    }), previous)
    cfg["v2_namespace"]["superseded_v2_freeze"] = str(previous)

    spec = build_v2_spec(cfg)
    pointer = spec.get("supersedes_v2_freeze")
    assert pointer["path"] == str(previous)
    assert pointer["sha256"] == load_spec(previous).sha256
    assert pointer["file_sha256"] == sha256_file(previous)
    assert pointer["role_partition_fingerprint"] == "the-old-fingerprint"

    # a configured pointer that does not resolve must stop the freeze
    cfg["v2_namespace"]["superseded_v2_freeze"] = str(tmp_path / "absent_v2.json")
    with pytest.raises(V2FreezeError, match="superseded v2 freeze not found"):
        build_v2_spec(cfg)
