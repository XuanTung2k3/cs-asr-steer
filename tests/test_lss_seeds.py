"""LSS unit test: no module may draw randomness from an unregistered source."""
from __future__ import annotations

import pytest

from csasr.lss.seeds import REQUIRED_PURPOSES, SeedMap, seed_for, seeds_for

CFG = {"seeds": {p: (7 if p != "jitter" else [1, 2, 3]) for p in REQUIRED_PURPOSES}}


def test_seed_map_requires_every_purpose():
    incomplete = {"seeds": {"bootstrap": 1}}
    with pytest.raises(KeyError):
        SeedMap.from_cfg(incomplete)
    assert SeedMap.from_cfg(CFG).to_dict()["values"]["bootstrap"] == 7


def test_unregistered_purpose_raises():
    with pytest.raises(KeyError, match="unregistered seed purpose"):
        seed_for(CFG, "some_new_idea")


def test_scalar_and_sequence_purposes_are_distinguished():
    assert seed_for(CFG, "bootstrap") == 7
    assert seeds_for(CFG, "bootstrap") == (7,)
    assert seeds_for(CFG, "jitter") == (1, 2, 3)
    with pytest.raises(TypeError):
        seed_for(CFG, "jitter")


def test_serialization_is_sorted_and_stable():
    a = SeedMap.from_cfg(CFG).to_dict()
    b = SeedMap.from_cfg(CFG).to_dict()
    assert a == b
    assert list(a["values"]) == sorted(a["values"])


def test_synthetic_dev_and_gate_are_separate_purposes():
    # The Gate-A synthetic set must never be the one used to pick a config.
    assert "synthetic_dev" in REQUIRED_PURPOSES
    assert "synthetic_gate" in REQUIRED_PURPOSES
