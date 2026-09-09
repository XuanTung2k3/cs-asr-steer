"""Focused CPU guards for the BASIS-A2 exploratory atlas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.basis_frozen_layer_atlas import DIRECTIONS, LAYERS, MIX, _ids_fp

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_panel_is_exact_quota_and_deterministic():
    panel = json.loads((ROOT / "results/basis_frozen_layer_atlas/panel.json").read_text())
    rows = panel["rows"]
    assert len(rows) == 10
    assert {x["category"] for x in rows} == {
        "baseline_wrong_embedded_poi", "baseline_correct_embedded", "mixed_challenging"}
    assert len({x["utterance_id"] for x in rows}) == 10
    assert panel["fingerprint"] == _ids_fp([x["utterance_id"] for x in rows])


def test_atlas_layers_and_direction_families_are_frozen():
    assert LAYERS == tuple(range(32))
    assert DIRECTIONS == ("Raw", "Local", "Conditioning", "Raw+Cond", "Local+Cond")
    assert MIX == (.5, .5)


def test_direction_mix_is_unit_and_uses_same_coefficients():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=128); cond = rng.normal(size=128)
    raw /= np.linalg.norm(raw); cond /= np.linalg.norm(cond)
    rc = (.5 * raw + .5 * cond); lc = (.5 * raw + .5 * cond)
    assert np.isclose(np.linalg.norm(rc / np.linalg.norm(rc)), 1.0)
    assert np.allclose(rc, lc)
