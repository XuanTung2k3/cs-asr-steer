"""The v2r3 baseline/POI driver: what it refuses, and what it must not produce.

The driver exists so that `p0_baseline`'s stage behaviour -- v1 subsets, v1 write
paths, primary-baseline selection, and the P0 gate -- is not dragged into a run
that must do none of those things. These tests pin exactly that.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from csasr.experiments import v2r3_baseline_poi as driver


def test_only_the_two_permitted_roles_are_accepted():
    assert driver.resolve_roles(["D-construct"]) == ("D-construct",)
    assert driver.resolve_roles(["D-construct", "D-dev-select", "D-construct"]) == \
        ("D-construct", "D-dev-select")


@pytest.mark.parametrize("role", ["D-dev-confirm", "D-test", "loc-train",
                                  "util-train", "router-calib", "dev_select"])
def test_every_other_role_is_refused_by_name(role):
    """A held-out split must fail loudly, never be quietly skipped."""
    with pytest.raises(driver.BaselinePoiError, match="not permitted"):
        driver.resolve_roles([role])
    with pytest.raises(driver.BaselinePoiError, match="not permitted"):
        driver.resolve_roles(["D-construct", role])


def test_no_roles_is_refused():
    with pytest.raises(driver.BaselinePoiError, match="at least one role"):
        driver.resolve_roles([])


def test_an_existing_destination_is_refused(tmp_path):
    existing = tmp_path / "generation_001"
    existing.mkdir()
    with pytest.raises(driver.BaselinePoiError, match="write-once"):
        driver.resolve_output(existing)
    assert driver.resolve_output(tmp_path / "fresh") == (tmp_path / "fresh").resolve()


def test_a_symlink_destination_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(driver.BaselinePoiError, match="symlink"):
        driver.resolve_output(link)


def test_scoring_is_the_unmodified_pier_function():
    """`poi_table` must be pier's own object, not a wrapper or a copy."""
    from csasr.evaluation import pier

    assert driver.poi_table is pier.poi_table
    assert driver.poi_table.__module__ == "csasr.evaluation.pier"
    source = (__import__("pathlib").Path(driver.__file__)).read_text(encoding="utf-8")
    # no local re-implementation of alignment or categorisation
    for banned in ("def poi_table", "def evaluate_pois", "def _categorize",
                   "align_tokens(", "normalize_text("):
        assert banned not in source, f"driver must not reimplement {banned}"


def _manifest(n=3):
    return pd.DataFrame({
        "utterance_id": [f"ZH-CN_U0001_S0_{i}" for i in range(n)],
        "transcript_raw": ["我 like 这个 project"] * n,
        "speaker_id": ["ZH-CN_U0001_S0"] * n,
        "conversation_id": ["ZH-CN_U0001"] * n,
        "dialogue_id": ["CSD0001"] * n,
        "duration_sec": [3.0] * n,
    })


def test_poi_rows_carry_role_and_cluster_provenance():
    manifest = _manifest()
    predictions = pd.DataFrame({
        "utterance_id": manifest["utterance_id"],
        "hypothesis_raw": ["我 like 这个 project"] * len(manifest),
    })
    table = driver.build_poi(manifest, predictions, "D-construct")
    assert len(table)
    assert set(table["role"]) == {"D-construct"}
    for column in ("dialogue_id", "conversation_id", "speaker_id"):
        assert column in table.columns
    counts = driver.summarise(table, manifest)
    assert counts["lexical_units"] == len(table)
    assert counts["correct"] + counts["baseline_error"] == counts["lexical_units"]
    assert counts["unmatched_units"] == 0
    assert counts["dialogues"] == 1


def test_an_undecoded_utterance_is_refused_rather_than_dropped():
    manifest = _manifest(3)
    predictions = pd.DataFrame({
        "utterance_id": manifest["utterance_id"].tolist()[:2],
        "hypothesis_raw": ["我 like 这个 project"] * 2,
    })
    with pytest.raises(driver.BaselinePoiError, match="received no"):
        driver.build_poi(manifest, predictions, "D-dev-select")


def test_baseline_is_hard_bound_to_b0_auto_and_no_selection_exists():
    """Structural, not textual: the docstring may *promise* no primary baseline.

    A substring scan would trip over the prose that documents the guarantee, so
    the check is on what the module imports and calls.
    """
    import ast
    import pathlib

    assert driver.SYSTEM == "B0_AUTO"
    assert driver.LANGUAGE is None

    tree = ast.parse(pathlib.Path(driver.__file__).read_text(encoding="utf-8"))
    # docstrings describe the guarantee ("writes no primary_baseline.json"), so
    # they must not be scanned as if they were code
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    imported, called, literals = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) in docstrings:
            continue
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            target = node.func
            called.add(target.id if isinstance(target, ast.Name)
                       else getattr(target, "attr", ""))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)

    for banned in ("gate", "criterion", "StageLock", "write_status", "corpus_mer"):
        assert banned not in imported, f"driver must not import {banned}"
        assert banned not in called, f"driver must not call {banned}"
    # the P0 selection artifact is never named as a path, and there is no second
    # baseline to select between. `selects_primary_baseline: False` in the
    # completion payload is a declaration that it did not happen, not a write.
    assert not any("primary_baseline.json" in s for s in literals)
    assert "B1_ZH" not in literals
    assert driver.SYSTEM not in ("B1_ZH",)


def test_completion_payload_declares_no_gate_and_no_selection(tmp_path, monkeypatch):
    """The marker must state, in the artifact itself, what this run did not do."""
    manifest = _manifest()
    predictions = pd.DataFrame({
        "utterance_id": manifest["utterance_id"],
        "hypothesis_raw": ["我 like 这个 project"] * len(manifest),
    })
    table = driver.build_poi(manifest, predictions, "D-construct")
    counts = driver.summarise(table, manifest)
    payload = {"schema": "lss_v2r3_baseline_poi_v1", "counts": {"D-construct": counts},
               "evaluates_gate": False, "selects_primary_baseline": False}
    assert payload["evaluates_gate"] is False
    assert payload["selects_primary_baseline"] is False
    assert json.loads(json.dumps(payload))["counts"]["D-construct"]["lexical_units"] > 0
