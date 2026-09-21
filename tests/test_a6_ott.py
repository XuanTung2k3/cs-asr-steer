from __future__ import annotations

import json

import pytest

from csasr.basis_a6.ott import (
    COMPACT_FAMILIES,
    PhaseFirewallError,
    assert_compact_row,
    assert_phase_a_selection_rows,
    compact_sites,
    enumerate_phase_a,
    freeze_search_ids,
    phase_a_logical_total,
)


def test_compact_matrix_counts_and_methods() -> None:
    assert len(compact_sites("whisper")) == 96
    assert len(compact_sites("qwen3_asr_1p7b")) == 80
    for model, expected in (("whisper", 11_520), ("qwen3_asr_1p7b", 12_800)):
        rows = enumerate_phase_a(model)
        assert len(rows) == expected
        assert all((row["family"], row["side"]) in COMPACT_FAMILIES for row in rows)
        assert {row["decode_mode"] for row in rows} == {"greedy"}
        assert all("construction_source" not in row for row in rows)
        assert phase_a_logical_total(model) == expected


def test_compact_boundary_rejects_legacy_and_fixed_rows() -> None:
    base = {"phase": "A", "data_role": "search", "family": "add_unique", "side": "encoder"}
    assert_compact_row(base)
    for row in (
        {**base, "family": "raw"},
        {**base, "family": "conditioning_all", "side": "decoder"},
        {**base, "branch": "fixed"},
        {**base, "construction_source": "cs_dialogue"},
    ):
        with pytest.raises(PhaseFirewallError):
            assert_compact_row(row)


def test_phase_a_selector_rejects_confirm_and_transfer_rows() -> None:
    good = {"phase": "A", "data_role": "search", "family": "add_unique", "side": "encoder"}
    assert_phase_a_selection_rows([good])
    for bad in (
        {**good, "phase": "B", "data_role": "confirm"},
        {**good, "phase": "C", "data_role": "transfer"},
        {**good, "poi_net_utility": 1},
    ):
        with pytest.raises(PhaseFirewallError):
            assert_phase_a_selection_rows([bad])


def test_search_freeze_is_metadata_only_and_deterministic() -> None:
    records = [
        {"dataset": dataset, "utterance_id": f"{dataset}-{i}", "dialogue_id": f"d{i}"}
        for dataset in ("cs_dialogue", "ascend") for i in range(20)
    ]
    first = freeze_search_ids(records)
    second = freeze_search_ids(list(reversed(records)))
    assert first == second
    assert len(first["ids"]["cs_dialogue"]) == 20
    assert len(first["ids"]["ascend"]) == 20
    with pytest.raises(PhaseFirewallError):
        freeze_search_ids([{"dataset": "cs_dialogue", "utterance_id": "x", "PIER": 0}] * 20 + records[20:])


def test_selection_loader_rejects_confirm_path_before_read(tmp_path) -> None:
    path = tmp_path / "phase_b" / "confirm.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"phase": "A", "rows": []}))
    from csasr.basis_a6.ott import load_selection_rows

    with pytest.raises(PhaseFirewallError):
        load_selection_rows(path, result_root=tmp_path)


def test_resumability_requires_identity_and_provenance(tmp_path) -> None:
    from csasr.basis_a6.ott import load_accepted_cell

    path = tmp_path / "phase_a" / "cell.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"status": "PASS", "canonical_key": "k", "provenance": {"run": "r"}}))
    accepted = load_accepted_cell(path, canonical_key="k", phase="A", result_root=tmp_path)
    assert accepted and accepted["reused"] is True and accepted["source_hash"].startswith("sha256:")
    assert load_accepted_cell(path, canonical_key="wrong", phase="A", result_root=tmp_path) is None


def test_physical_runner_exposes_only_compact_model_resident_shards() -> None:
    from csasr.experiments.a6_ott import list_shards

    assert len(list_shards("whisper")) == 4
    assert len(list_shards("qwen")) == 4
    assert all("encoder" in x or "decoder" in x for x in list_shards("whisper"))
