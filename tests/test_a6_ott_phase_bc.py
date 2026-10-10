from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from csasr.basis_a6.cache import DirectionCache
from csasr.experiments import a6_ott
from csasr.experiments.a6_ott_phase_bc import canonical_key, row_path


def test_phase_bc_mode_is_part_of_result_namespace() -> None:
    common = dict(phase="B", model="whisper", dataset="cs_dialogue_confirm",
                  uid="u1", family="add_unique", side="decoder", layer=4, rho=0.5)
    greedy = canonical_key(**common, decode_mode="greedy")
    standard = canonical_key(**common, decode_mode="official_standard")
    assert greedy != standard
    assert row_path("B", greedy) != row_path("B", standard)
    assert row_path("B", "B|whisper|cs_dialogue_confirm|u1|add_unique|decoder|4|0.5|greedy|decoder_post_cross_attn_residual").parts[-2] == "greedy"


def test_confirmation_eligibility_has_no_outcome_fields() -> None:
    root = Path("results/a6_ott_upper_bound/eligibility")
    payload = json.loads((root / "CONFIRMATION_ELIGIBILITY_SUMMARY.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["source"]["outcome_artifacts_loaded"] is False


def test_empty_conditioning_positions_keep_add_unique_decoder_eligible() -> None:
    rng = np.random.default_rng(7)
    analysis = {
        "A": rng.normal(size=(3, 4)),
        "B": rng.normal(size=(3, 4)),
        "conditioning": (rng.normal(size=(3, 4)), np.array([], dtype=int)),
    }
    out = a6_ott._direction_set(
        "qwen3_asr_1p7b", "seame_dev_man", "u1", "greedy", "decoder", 0,
        analysis, DirectionCache(), {"direction_constructions": 0},
    )
    assert set(out) == {"add_unique"}
