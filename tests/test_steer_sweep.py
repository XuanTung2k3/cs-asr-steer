"""CPU regression tests for the two-track steering harness.

Each test asserts a scientific semantic that would silently produce wrong
numbers if it regressed, not merely that a file exists.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from steer_sweep import config as C                      # noqa: E402
from steer_sweep import metrics as M                     # noqa: E402
from steer_sweep import sweep as S                       # noqa: E402
from steer_sweep.hooks import (BeamCoverageError,        # noqa: E402
                               DecoderSteering, DirectionSpec,
                               PromptDirectionAtEncoderError, SteeringPlan,
                               UtterancePlan, apply_edit, cache_length,
                               energy_report, planned_positions)


# ---------------------------------------------------------------------------
# 1.2 cache-length reading
# ---------------------------------------------------------------------------

class _TupleCache(tuple):
    pass


class _ObjCache:
    def __init__(self, n):
        self._n = n

    def get_seq_length(self):
        return self._n


def test_cache_length_handles_legacy_tuple_and_cache_object():
    legacy = _TupleCache([(torch.zeros(2, 4, 7, 8), torch.zeros(2, 4, 7, 8))])
    assert cache_length(legacy) == 7
    assert cache_length(_ObjCache(11)) == 11
    assert cache_length(None) == 0
    assert cache_length(()) == 0


# ---------------------------------------------------------------------------
# 1.4 matched energy
# ---------------------------------------------------------------------------

def test_matched_energy_is_invariant_to_coverage():
    plan = SteeringPlan(site=C.SITE_DECODER, coverage=C.COVERAGE_GLOBAL,
                        alpha=2.0, matched_energy=True, decoder_layers=(16,))
    totals = {s: energy_report(plan, s)["E_total"] for s in (1, 10, 100, 1000)}
    assert len(set(round(v, 10) for v in totals.values())) == 1
    assert math.isclose(totals[1], 4.0)


def test_unmatched_energy_scales_with_coverage():
    """Without matched energy a global cell delivers more dose than a local one,
    which is exactly the confound requirement 1.4 exists to remove."""
    plan = SteeringPlan(site=C.SITE_DECODER, coverage=C.COVERAGE_GLOBAL,
                        alpha=2.0, matched_energy=False, decoder_layers=(16,))
    assert energy_report(plan, 100)["E_total"] > energy_report(plan, 1)["E_total"]


def test_planned_positions_sums_over_layers_and_sites():
    u = UtterancePlan(utterance_id="u", duration_sec=10.0, valid_frames=500,
                      encoder_frames=[(10, 30)], decoder_steps=[3],
                      planned_decoder_positions=40)
    both = SteeringPlan(site=C.SITE_BOTH, coverage=C.COVERAGE_GLOBAL, alpha=1.0,
                        encoder_layers=(8, 15), decoder_layers=(16,))
    assert planned_positions(both, [u]) == 500 * 2 + 40 * 1
    local = SteeringPlan(site=C.SITE_BOTH, coverage=C.COVERAGE_LOCAL, alpha=1.0,
                         encoder_layers=(8, 15), decoder_layers=(16,))
    assert planned_positions(local, [u]) == 20 * 2 + 1 * 1


# ---------------------------------------------------------------------------
# 1.5 / alpha = 0
# ---------------------------------------------------------------------------

def test_alpha_zero_edit_is_bit_identical():
    h = torch.randn(2, 5, 16)
    v = torch.randn(16)
    v = v / v.norm()
    mask = torch.ones(2, 5, dtype=torch.bool)
    out, edited, _ = apply_edit(h, v, 0.0, mask, C.SCALE_ACT_NORM)
    assert edited == 0
    assert torch.equal(out, h)


def test_unmasked_positions_are_bit_identical():
    h = torch.randn(2, 5, 16)
    v = torch.nn.functional.normalize(torch.randn(16), dim=0)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1] = True
    out, edited, _ = apply_edit(h, v, 1.0, mask, C.SCALE_UNIT)
    assert edited == 1
    assert torch.equal(out[0, 0], h[0, 0])
    assert torch.equal(out[1], h[1])
    assert not torch.equal(out[0, 1], h[0, 1])


def test_act_norm_scaling_uses_mean_norm_of_steered_positions():
    h = torch.ones(1, 3, 4) * 2.0                 # every row has norm 4
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.ones(1, 3, dtype=torch.bool)
    out, _, scale = apply_edit(h, v, 0.5, mask, C.SCALE_ACT_NORM)
    assert math.isclose(scale, 4.0, rel_tol=1e-5)
    assert math.isclose(float(out[0, 0, 0]), 2.0 + 0.5 * 4.0, rel_tol=1e-5)


def test_unit_scale_mode_ignores_activation_norm():
    h = torch.ones(1, 2, 4) * 100.0
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    out, _, scale = apply_edit(h, v, 0.5, mask, C.SCALE_UNIT)
    assert scale == 1.0
    assert math.isclose(float(out[0, 0, 0]), 100.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 1.6 v_prompt is decoder-only
# ---------------------------------------------------------------------------

def test_prompt_direction_at_encoder_site_raises():
    with pytest.raises(PromptDirectionAtEncoderError):
        DirectionSpec(kind=C.KIND_V_PROMPT, site=C.SITE_ENCODER, layer=15,
                      vector=torch.zeros(4), source="derived")


def test_prompt_direction_at_decoder_site_is_allowed():
    spec = DirectionSpec(kind=C.KIND_V_PROMPT, site=C.SITE_DECODER, layer=16,
                         vector=torch.zeros(4), source="derived")
    assert spec.site == C.SITE_DECODER


def test_plan_validation_rejects_prompt_at_encoder():
    plan = SteeringPlan(
        site=C.SITE_ENCODER, coverage=C.COVERAGE_GLOBAL, alpha=1.0,
        encoder_layers=(15,),
        encoder_direction=DirectionSpec(kind=C.KIND_V_NAT, site=C.SITE_ENCODER,
                                        layer=15, vector=torch.zeros(4),
                                        source="loaded"))
    plan.encoder_direction.kind = C.KIND_V_PROMPT      # simulate a substitution
    with pytest.raises(PromptDirectionAtEncoderError):
        plan.validate()


# ---------------------------------------------------------------------------
# 1.3 beams
# ---------------------------------------------------------------------------

class _FakeTracker:
    def __init__(self, start):
        self.start_position = start
        self.forward_calls = 1


def _decoder_hook(num_beams, coverage=C.COVERAGE_GLOBAL, steps=None):
    plan = SteeringPlan(site=C.SITE_DECODER, coverage=coverage, alpha=1.0,
                        decoder_layers=(16,), num_beams=num_beams,
                        decoder_direction=DirectionSpec(
                            kind=C.KIND_V_NAT, site=C.SITE_DECODER, layer=16,
                            vector=torch.nn.functional.normalize(torch.randn(8), dim=0),
                            source="loaded"))
    u = UtterancePlan(utterance_id="u", duration_sec=1.0, valid_frames=50,
                      decoder_steps=steps if steps is not None else [2],
                      planned_decoder_positions=10)
    return DecoderSteering(None, plan, [u], 1.0, _FakeTracker(C.PREFIX_WIDTH))


def test_every_beam_is_steered():
    hook = _decoder_hook(num_beams=5)
    hidden = torch.randn(5, 1, 8)                 # B=1, k=5
    mask = hook._build_mask(hidden, C.PREFIX_WIDTH)
    assert bool(mask.all())
    hook.positions = int(mask.sum())
    hook.fired = 1                                # the hook was actually invoked
    hook.assert_every_beam_steered()


def test_beam_misalignment_raises():
    hook = _decoder_hook(num_beams=5)
    hook.fired = 1
    hook.positions = 7                            # not divisible by 5
    with pytest.raises(BeamCoverageError):
        hook.assert_every_beam_steered()


def test_zero_positions_is_not_an_error_for_local_coverage():
    """Regression: local coverage legitimately steers zero positions in any
    batch whose utterances carry no oracle target (only 116/300 D-dev-select
    utterances have one at all) -- caught when Stage D-extended's beam=5
    determinism check hit the first several duration-sorted batches, which
    contain none. 0 % k == 0 holds trivially; only a partial beam group (a
    nonzero count not divisible by num_beams) is a real failure."""
    hook = _decoder_hook(num_beams=5, coverage=C.COVERAGE_LOCAL, steps=[2])
    hook.fired = 1                                # the hook fired, found nothing to steer
    hook.positions = 0
    hook.assert_every_beam_steered()               # must not raise


def test_hook_never_firing_at_all_still_raises():
    """A batch where the hook was genuinely never invoked (wrong layer, hook
    never attached) is a real failure, distinct from a legitimate
    zero-position batch where the hook fired but found nothing to steer."""
    hook = _decoder_hook(num_beams=5)
    hook.fired = 0
    hook.positions = 0
    with pytest.raises(BeamCoverageError):
        hook.assert_every_beam_steered()


def test_forced_prefix_positions_are_never_steered():
    hook = _decoder_hook(num_beams=1)
    hidden = torch.randn(1, 6, 8)
    mask = hook._build_mask(hidden, 0)            # prefill: positions 0..5
    assert not bool(mask[0, :C.PREFIX_WIDTH].any())
    assert bool(mask[0, C.PREFIX_WIDTH:].all())


def test_local_coverage_steers_only_the_oracle_step():
    hook = _decoder_hook(num_beams=1, coverage=C.COVERAGE_LOCAL, steps=[2])
    hidden = torch.randn(1, 8, 8)
    mask = hook._build_mask(hidden, 0)
    expected = C.PREFIX_WIDTH + 2
    assert bool(mask[0, expected])
    assert int(mask.sum()) == 1


# ---------------------------------------------------------------------------
# the fixed selection rule (3.6)
# ---------------------------------------------------------------------------

def _row(cell_id, **kw):
    base = {"cell_id": cell_id, "verdict": "NO-GO", "zh_retention": 1.0,
            "dialogues_with_correction": 0, "corrections_to_corruptions": 0.0,
            "delta_pier": 0.0}
    base.update(kw)
    return base


def test_selection_drops_invalid_decode_before_ranking():
    rows = [_row("bad", verdict="INVALID", dialogues_with_correction=99),
            _row("good", dialogues_with_correction=1)]
    top, trace = M.select(rows, top=1)
    assert [r["cell_id"] for r in top] == ["good"]
    assert trace["dropped_unhealthy_decode"] == ["bad"]


def test_selection_drops_low_zh_retention():
    rows = [_row("leaky", zh_retention=0.98, dialogues_with_correction=99),
            _row("clean", dialogues_with_correction=1)]
    top, trace = M.select(rows, top=1)
    assert [r["cell_id"] for r in top] == ["clean"]
    assert trace["dropped_zh_retention_below_0.99"] == ["leaky"]


def test_dialogue_coverage_outranks_delta_pier():
    """The prior programme's 7 corrections came from 2 of 20 dialogues. Ranking
    on delta PIER alone would prefer exactly that failure mode."""
    concentrated = _row("concentrated", dialogues_with_correction=2,
                        corrections_to_corruptions=9.0, delta_pier=0.05)
    spread = _row("spread", dialogues_with_correction=7,
                  corrections_to_corruptions=2.0, delta_pier=0.01)
    top, _ = M.select([concentrated, spread], top=1)
    assert top[0]["cell_id"] == "spread"


def test_ratio_breaks_ties_before_delta_pier():
    a = _row("a", dialogues_with_correction=5, corrections_to_corruptions=3.0,
             delta_pier=0.001)
    b = _row("b", dialogues_with_correction=5, corrections_to_corruptions=1.0,
             delta_pier=0.900)
    top, _ = M.select([a, b], top=1)
    assert top[0]["cell_id"] == "a"


# ---------------------------------------------------------------------------
# gates (3.7)
# ---------------------------------------------------------------------------

import pandas as pd                                        # noqa: E402

HEALTHY = {"healthy": True}
UNHEALTHY = {"healthy": False}


def _outcomes(corrected_dialogues):
    return pd.DataFrame([{"corrected": True, "dialogue_id": d}
                         for d in corrected_dialogues]
                        or [{"corrected": False, "dialogue_id": "d0"}])


def test_all_four_gates_pass_gives_go():
    gates = M.evaluate_gates(
        delta={"delta_pier": 0.01, "ci_low": 0.002, "ci_high": 0.02},
        outcomes=_outcomes([f"d{i}" for i in range(6)]),
        risk={"units_corrupted": 3, "zh_retention": 0.999,
              "zh_baseline_correct": C.ANCHOR_BASELINE_CORRECT_ZH_UNITS},
        health=HEALTHY)
    assert gates["verdict"] == "GO"
    assert not gates["gates_failed"]


def test_ci_touching_zero_fails_g1():
    gates = M.evaluate_gates(
        delta={"delta_pier": 0.01, "ci_low": -0.001, "ci_high": 0.02},
        outcomes=_outcomes([f"d{i}" for i in range(6)]),
        risk={"units_corrupted": 1, "zh_retention": 1.0, "zh_baseline_correct": 1},
        health=HEALTHY)
    assert gates["G1_delta_pier"] is False
    assert gates["verdict"] == "NO-GO"


def test_prior_programme_result_fails_g2_and_g3():
    """7 corrections vs 13 corruptions from 2 dialogues: the exact prior
    outcome must not pass."""
    gates = M.evaluate_gates(
        delta={"delta_pier": 0.0143, "ci_low": 0.0033, "ci_high": 0.0254},
        outcomes=pd.DataFrame(
            [{"corrected": True, "dialogue_id": "CSD0502"}] * 6
            + [{"corrected": True, "dialogue_id": "CSD0014"}]),
        risk={"units_corrupted": C.ANCHOR_ORACLE_CORRUPTIONS,
              "zh_retention": 0.9979,
              "zh_baseline_correct": C.ANCHOR_BASELINE_CORRECT_ZH_UNITS},
        health=HEALTHY)
    assert gates["corrections"] == C.ANCHOR_ORACLE_CORRECTIONS
    assert gates["dialogues_with_correction"] == 2
    assert gates["G2_correction_ratio"] is False
    assert gates["G3_dialogue_coverage"] is False
    assert gates["verdict"] == "NO-GO"


def test_unhealthy_decode_is_invalid_not_negative():
    gates = M.evaluate_gates(
        delta={"delta_pier": 0.5, "ci_low": 0.4, "ci_high": 0.6},
        outcomes=_outcomes([f"d{i}" for i in range(20)]),
        risk={"units_corrupted": 0, "zh_retention": 1.0, "zh_baseline_correct": 1},
        health=UNHEALTHY)
    assert gates["verdict"] == "INVALID"


def test_zero_corruption_with_corrections_is_infinite_ratio():
    gates = M.evaluate_gates(
        delta={"delta_pier": 0.01, "ci_low": 0.001, "ci_high": 0.02},
        outcomes=_outcomes(["d1", "d2", "d3", "d4", "d5"]),
        risk={"units_corrupted": 0, "zh_retention": 1.0, "zh_baseline_correct": 1},
        health=HEALTHY)
    assert math.isinf(gates["corrections_to_corruptions"])
    assert gates["G2_correction_ratio"] is True


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------

def test_stage_a_is_the_six_site_by_coverage_cells():
    cells = S.stage_a_cells()
    assert len(cells) == 6
    assert {(c.site, c.coverage) for c in cells} == {
        (s, cov) for s in (C.SITE_ENCODER, C.SITE_DECODER, C.SITE_BOTH)
        for cov in (C.COVERAGE_GLOBAL, C.COVERAGE_LOCAL)}
    assert all(c.alpha == 1.0 and c.matched_energy for c in cells)
    assert all(c.direction_kind == C.KIND_V_NAT for c in cells)


def test_stage_b_keeps_the_run_count_at_section_3_6():
    cells = S.stage_b_cells([{"site": C.SITE_DECODER, "coverage": C.COVERAGE_LOCAL}])
    assert len(cells) == 6                    # + top-2 carried from A = 8 rows
    assert sum(1 for c in cells if c.direction_kind == C.KIND_V_PROMPT) == 2
    assert sum(1 for c in cells if c.combo_mode == C.COMBO_WEIGHTED) == 3
    assert sum(1 for c in cells if c.combo_mode == C.COMBO_RANK2) == 1
    assert all(c.site == C.SITE_DECODER for c in cells)


def test_full_weighted_grid_is_three_by_three():
    cells = S.stage_b_cells([{"site": C.SITE_DECODER, "coverage": C.COVERAGE_LOCAL}],
                            full_weighted_grid=True)
    assert sum(1 for c in cells if c.combo_mode == C.COMBO_WEIGHTED) == 9


def test_stage_c_is_the_five_alpha_grid():
    best = {"site": C.SITE_DECODER, "coverage": C.COVERAGE_LOCAL,
            "direction_kind": C.KIND_V_NAT, "alpha": 1.0, "decoder_layers": [16]}
    cells = S.stage_c_cells(best)
    assert [c.alpha for c in cells] == list(C.ALPHA_GRID)
    assert all(c.stage == "C" for c in cells)


def test_stage_d_sweeps_singles_then_multi_for_the_winning_site():
    best = {"site": C.SITE_DECODER, "coverage": C.COVERAGE_LOCAL,
            "direction_kind": C.KIND_V_NAT, "alpha": 2.0, "decoder_layers": [16]}
    cells = S.stage_d_cells(best)
    singles = [c for c in cells if len(c.decoder_layers) == 1]
    multi = [c for c in cells if len(c.decoder_layers) > 1]
    assert {c.decoder_layers[0] for c in singles} == set(C.DECODER_SINGLE_LAYERS)
    assert {c.decoder_layers for c in multi} == {
        tuple(m) for m in C.DECODER_MULTI_LAYERS}
    assert not any(c.encoder_layers for c in cells)


def test_cell_hash_is_stable_and_discriminating():
    a = S.stage_a_cells()[0]
    b = S.stage_a_cells()[0]
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != S.stage_a_cells()[1].config_hash()


# ---------------------------------------------------------------------------
# split discipline
# ---------------------------------------------------------------------------

def test_d_test_loader_raises():
    from steer_sweep.data import LockedSplitError, load_split
    with pytest.raises(LockedSplitError):
        load_split("D-test", {"v2_namespace": {"role_root": "/nonexistent"}})


def test_d_dev_confirm_requires_explicit_authorisation():
    from steer_sweep.data import UnauthorizedSplitError, load_split
    with pytest.raises(UnauthorizedSplitError):
        load_split("D-dev-confirm", {"v2_namespace": {"role_root": "/nonexistent"}})


def test_population_anchor_mismatch_raises():
    from steer_sweep.data import Population, assert_population_anchors
    pop = Population(manifest=None, targets=None, poi=None,
                     baseline_hypotheses={}, split=C.DEV_SELECT,
                     counts={"utterances": 300, "dialogues": 20, "targets": 488,
                             "utterances_with_targets": 116})
    with pytest.raises(AssertionError, match="489"):
        assert_population_anchors(pop)


# ---------------------------------------------------------------------------
# Track B modules
# ---------------------------------------------------------------------------

def test_intervention_parameter_counts_match_the_analytic_formula():
    from steer_sweep.trackb.modules import AdditiveIntervention, LoReftIntervention
    d, r = 1280, 4
    assert sum(p.numel() for p in LoReftIntervention(d, r).parameters()) == 2 * r * d + r
    assert sum(p.numel() for p in AdditiveIntervention(d, r).parameters()) == 2 * r * d


class _FakeBundle:
    def __init__(self, d_model):
        self.d_model = d_model


def test_intervention_module_subspace_reads_the_first_layer():
    """Regression: InterventionModule.subspace() used to be missing entirely,
    so every B0/B1/B2 4.3 diagnostic call crashed with AttributeError before
    a single training row could be written (caught by the Track B smoke run,
    job 42649)."""
    from steer_sweep.trackb.modules import InterventionModule

    module = InterventionModule(_FakeBundle(16), site=C.SITE_DECODER,
                                layers=(16,), rank=4, variant="loreft")
    module.initialize(informed=None, seed=1)
    basis = module.subspace()
    assert basis.shape == (4, 16)
    assert np.allclose((basis @ basis.T).numpy(), np.eye(4), atol=1e-5)


def test_intervention_module_subspace_uses_the_first_of_multiple_layers():
    from steer_sweep.trackb.modules import InterventionModule

    module = InterventionModule(_FakeBundle(16), site=C.SITE_DECODER,
                                layers=(8, 16), rank=2, variant="loreft")
    module.initialize(informed=None, seed=1)
    first_block_basis = module.blocks["8"].subspace()
    assert torch.equal(module.subspace(), first_block_basis)


def test_informed_basis_spans_the_constructed_directions():
    from steer_sweep.trackb.modules import informed_basis
    rng = np.random.default_rng(0)
    v_nat = rng.standard_normal(64)
    v_prompt = rng.standard_normal(64)
    basis = informed_basis(v_nat, v_prompt, 2, seed=1).numpy().astype(np.float64)
    assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-5)
    residual = v_nat - basis.T @ (basis @ v_nat)
    assert np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(v_nat)) + 1e-6


def test_informed_basis_fills_higher_rank_with_orthogonal_random():
    from steer_sweep.trackb.modules import informed_basis
    rng = np.random.default_rng(3)
    basis = informed_basis(rng.standard_normal(64), rng.standard_normal(64), 4,
                           seed=7).numpy().astype(np.float64)
    assert basis.shape == (4, 64)
    assert np.allclose(basis @ basis.T, np.eye(4), atol=1e-5)


def test_random_and_informed_bases_are_both_orthonormal():
    """B0/B1 must differ from B2 in DIRECTION only, never in magnitude."""
    from steer_sweep.trackb.modules import informed_basis, random_basis_matched
    rng = np.random.default_rng(11)
    a = informed_basis(rng.standard_normal(128), rng.standard_normal(128), 4,
                       seed=5).numpy().astype(np.float64)
    b = random_basis_matched(4, 128, seed=5).numpy().astype(np.float64)
    assert np.allclose(a @ a.T, np.eye(4), atol=1e-5)
    assert np.allclose(b @ b.T, np.eye(4), atol=1e-5)


def test_principal_angles_zero_for_identical_and_ninety_for_random():
    from steer_sweep.trackb.diagnostics import principal_angles
    basis = np.eye(2, 256)
    assert max(principal_angles(basis, basis)) < 1e-6
    rng = np.random.default_rng(2)
    angles = principal_angles(basis, rng.standard_normal((2, 256)))
    assert min(angles) > 75.0


def test_seed_range_and_beats_range():
    from steer_sweep.trackb.diagnostics import beats_range, seed_range
    rng = seed_range([0.30, 0.32, 0.34])
    assert rng["n"] == 3 and math.isclose(rng["mean"], 0.32)
    assert beats_range(0.28, rng) == "BEATS_RANGE"
    assert beats_range(0.33, rng) == "WITHIN_RANGE"
    assert beats_range(0.40, rng) == "WORSE_THAN_RANGE"


def test_aga_adapter_fraction_is_within_a_factor_of_two_of_the_paper():
    d, divisor = C.EXPECTED_D_MODEL, C.AGA["adapter_bottleneck_divisor"]
    bottleneck = d // divisor
    per_adapter = d * bottleneck + bottleneck + bottleneck * d + d + 2 * d
    blocks = C.EXPECTED_ENCODER_LAYERS + C.EXPECTED_DECODER_LAYERS
    total = 2 * per_adapter * blocks
    fraction = total / C.BACKBONE_PARAMS_NOMINAL
    target = C.AGA["paper_trainable_fraction"]
    assert target / 2.0 <= fraction <= target * 2.0


def test_aga_target_pattern_sends_each_language_to_its_own_slot():
    from steer_sweep.trackb import aga as AGA
    from steer_sweep.trackb import data as TD
    labels = torch.tensor([[TD.LID_IGNORE, TD.LID_IGNORE, TD.LID_IGNORE,
                            TD.LID_IGNORE, TD.LID_ZH, TD.LID_EN]])
    pattern, mask = AGA.target_pattern({"lid_token_labels": labels}, 6, c_val=0.6)
    assert math.isclose(float(pattern[0, 4, 0]), 0.6, rel_tol=1e-5)
    assert float(pattern[0, 4, 1]) == 0.0
    assert float(pattern[0, 5, 0]) == 0.0
    assert math.isclose(float(pattern[0, 5, 1]), 0.6, rel_tol=1e-5)
    # the two prompt slots anchor themselves
    assert math.isclose(float(pattern[0, AGA.ZH_SLOT, 0]), 0.6, rel_tol=1e-5)
    assert math.isclose(float(pattern[0, AGA.EN_SLOT, 1]), 0.6, rel_tol=1e-5)
    assert bool(mask[0, AGA.ZH_SLOT]) and bool(mask[0, AGA.EN_SLOT])


def test_lid_supervision_ignores_unlabelled_positions():
    from steer_sweep.trackb.modules import LIDHead
    head = LIDHead(8)
    hidden = torch.randn(2, 5, 8)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    labels[0, 2] = 1
    loss = head.loss(hidden, labels)
    assert torch.isfinite(loss)


def test_lora_model_preserves_bundle_model_identity():
    """Regression: `get_peft_model` wraps in a `PeftModel`, and
    `bundle.model.model.decoder` -- which every WhisperBundle helper relies on
    -- resolves one level short through that wrapper's attribute proxy
    (caught by the Track B smoke run, job 42651). `LoraModel` injects in place
    instead and must never change `bundle.model`'s object identity, before or
    after `.unload()`."""
    from peft import LoraConfig, LoraModel

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)

    class DecoderLayer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.self_attn = Attn(d)

    class Inner(nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([DecoderLayer(d) for _ in range(n)])

    class ToyWhisper(nn.Module):
        def __init__(self, d=8, n=2):
            super().__init__()
            self.model = Inner(d, n)

    toy = ToyWhisper()
    cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"])
    tuner = LoraModel(toy, {"default": cfg}, "default")
    assert tuner.model is toy
    assert toy.model.decoder.layers[0] is not None      # the bundle-style path

    trainable = {n for n, p in toy.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    frozen = all(not p.requires_grad for n, p in toy.named_parameters()
                if "lora_" not in n)
    assert frozen

    restored = tuner.unload()
    assert restored is toy
    assert isinstance(toy.model.decoder.layers[0].self_attn.q_proj, nn.Linear)
    assert not hasattr(toy.model.decoder.layers[0].self_attn.q_proj, "lora_A")


def test_backbone_grad_norm_ignores_trainable_params_embedded_in_bundle_model():
    """Regression: for B4, `LoraModel` injects lora_A/lora_B AS CHILDREN of
    `bundle.model`. The old `backbone_grad_norm` summed ALL of
    `bundle.model.parameters()` regardless of `requires_grad`, so a correctly
    working LoRA arm's own gradient was misreported as backbone leakage
    (caught by the Track B smoke run, job 42653: 'frozen-backbone gradient
    norm 0.176 != 0.0'). A frozen param that somehow got a nonzero .grad must
    still be caught."""
    from steer_sweep.trackb.modules import backbone_grad_norm

    class Bundle:
        def __init__(self, model):
            self.model = model

    frozen = nn.Linear(4, 4)
    frozen.weight.requires_grad_(False)
    trainable = nn.Linear(4, 4)          # e.g. an injected lora_A/lora_B

    model = nn.Module()
    model.frozen, model.trainable = frozen, trainable
    bundle = Bundle(model)

    trainable.weight.grad = torch.ones_like(trainable.weight)
    assert backbone_grad_norm(bundle) == 0.0, \
        "a legitimately trainable param embedded in bundle.model must not " \
        "count as backbone leakage"

    frozen.weight.grad = torch.ones_like(frozen.weight)
    assert backbone_grad_norm(bundle) > 0.0, \
        "a frozen param that somehow received a gradient must still be caught"


# ---------------------------------------------------------------------------
# checkpoint / resume
# ---------------------------------------------------------------------------

def test_result_store_resume_skips_completed_cells(tmp_path):
    from steer_sweep.store import ResultStore
    store = ResultStore(tmp_path / "results_A.jsonl", track="A")
    store.append({"config_hash": "abc123", "cell_id": "x"})
    again = ResultStore(tmp_path / "results_A.jsonl", track="A")
    assert again.has("abc123")
    assert not again.has("def456")
    assert again.get("abc123")["cell_id"] == "x"


def test_every_row_carries_the_diagnostic_stamp(tmp_path):
    from steer_sweep.store import ResultStore
    store = ResultStore(tmp_path / "results_B.jsonl", track="B")
    row = store.append({"config_hash": "h", "arm": "B1"})
    assert row["development_only_diagnostic"] is True
    assert row["taint"] == C.TAINT
    assert row["git_commit"]
    assert "seeds" in row and "environment" in row


def test_budget_reports_exhaustion():
    """A zero budget must mean 'stop now', not 'run forever'."""
    from steer_sweep.store import Budget
    assert not Budget(None).exhausted()
    assert Budget(None).remaining_hours() == float("inf")
    assert Budget(0.0).exhausted()
    assert not Budget(10.0).exhausted()


# ---------------------------------------------------------------------------
# SUMMARY.md must report current state, not every historical attempt stacked
# ---------------------------------------------------------------------------

def test_summary_rows_deduplicate_reruns_keeping_the_latest(tmp_path):
    """Regression: a cell rerun without --resume (a corrected code path
    re-executed against a shared out/ dir, exactly what happened fixing the
    Track B smoke jobs 42649/42651/42653/42656) appends a new JSONL row
    rather than overwriting the old one. The summary must report the LATEST
    row per config_hash, not stack every historical attempt into the table."""
    from steer_sweep.summary import _rows

    path = tmp_path / "results_B.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for mer in (0.90, 0.85, 0.31):        # three attempts at the same cell
            fh.write(json.dumps({"config_hash": "h1", "arm": "B0", "mer": mer}) + "\n")
        fh.write(json.dumps({"config_hash": "h2", "arm": "B1", "mer": 0.50}) + "\n")

    rows = _rows(path)
    assert [r["config_hash"] for r in rows] == ["h1", "h2"]
    assert rows[0]["mer"] == 0.31          # the latest attempt, not the first


def test_summary_rows_keep_untracked_rows_without_dropping_them(tmp_path):
    from steer_sweep.summary import _rows

    path = tmp_path / "results_A.jsonl"
    path.write_text(json.dumps({"cell_id": "legacy", "stage": "baseline"}) + "\n",
                    encoding="utf-8")
    rows = _rows(path)
    assert len(rows) == 1 and rows[0]["cell_id"] == "legacy"


def test_trackb_dev_eval_pool_is_capped_before_training():
    """Regression: the training-time dev-eval set defaulted to the full
    `router-calib` role (3,031 utterances). With eval_every=250 over ~14
    training invocations (lambda search x3, lr search x3, B0, B1 x3 seeds,
    B2, B4, B3-reimpl x2 stages), evaluating the full role at every one of
    up to 8 checkpoints per invocation is roughly 13 hours of decode overhead
    at Track A's observed rate -- caught before it burned real GPU time
    (job 42668 was cancelled at 5 minutes, before its first eval, once this
    was traced through). config.TRACKB_DEV_EVAL_LIMIT must be small enough
    that a full run's dev-eval decode stays proportionate to a 'fast
    observation pass', and dialogue-stratified subsampling must actually
    reduce a pool larger than the cap."""
    from steer_sweep import config as C
    from steer_sweep.trackb import data as TD

    assert C.TRACKB_DEV_EVAL_LIMIT < 500, \
        "the cap must stay far below router-calib's 3,031 utterances"

    examples = [
        TD.Example(utterance_id=f"u{i}", dialogue_id=f"d{i % 8}",
                  audio_path="", duration_sec=1.0, reference="", token_ids=[],
                  lid_token_labels=[], en_spans_sec=[])
        for i in range(3031)
    ]
    fraction = C.TRACKB_DEV_EVAL_LIMIT / len(examples)
    capped = TD.subsample(examples, fraction, seed=C.SEEDS["dev_eval_subsample"])
    assert len(capped) <= C.TRACKB_DEV_EVAL_LIMIT * 1.5   # per-dialogue rounding
    assert len(capped) < len(examples) / 10
    assert {e.dialogue_id for e in capped} == {f"d{i}" for i in range(8)}


def test_merge_audits_handles_a_none_min_position_across_batches():
    """Regression: `decoder_min_absolute_position` is None for any batch
    where local coverage steered nothing. A later batch with a real value hit
    `out.get(k, 0) + v` = `None + int` -> TypeError, caught only at full
    scale (300 utterances / multiple batches; the 8-utterance smoke run never
    produced a second batch after a None one). Summing a 'minimum position'
    across batches is also the wrong merge regardless of the crash -- the
    correct cross-batch value is the min of the per-batch mins."""
    from steer_sweep.decode import _merge_audits

    audits = [
        {"hook_fire_count": 3, "decoder_min_absolute_position": None,
         "valid_frame_ratio": [1.0, 1.0]},
        {"hook_fire_count": 2, "decoder_min_absolute_position": 7,
         "valid_frame_ratio": [0.5]},
        {"hook_fire_count": 4, "decoder_min_absolute_position": 4,
         "valid_frame_ratio": [1.0]},
    ]
    merged = _merge_audits(audits)          # must not raise
    assert merged["hook_fire_count"] == 9
    assert merged["decoder_min_absolute_position"] == 4


def test_merge_audits_reports_none_when_every_batch_is_none():
    from steer_sweep.decode import _merge_audits

    merged = _merge_audits([{"decoder_min_absolute_position": None,
                             "valid_frame_ratio": [1.0]}])
    assert merged["decoder_min_absolute_position"] is None


def test_smoke_output_is_isolated_from_full_run_output():
    """Regression: a Cell's config_hash (and Track B's arm-config hash) does
    not depend on population/step-count scale, so an 8-utterance smoke cell
    and its 300-utterance full-scale counterpart share the same identity.
    Sharing one out/ directory would let --resume silently treat smoke
    numbers as the real result (caught mid-session: a full Track A run,
    job 42657, and an earlier Track A smoke, job 42648, both wrote into the
    same out/ before this fix existed)."""
    import run_study as RS

    assert RS.resolve_out_dir("out", smoke=False) == Path("out")
    assert RS.resolve_out_dir("out", smoke=True) == Path("out/smoke")
    assert RS.resolve_out_dir("/mnt/data/x", smoke=True) == Path("/mnt/data/x/smoke")


def test_diagnostics_section_excludes_hyperparameter_search_rows():
    """lambda_lid and lr search both call train('B1', ...) at the SAME seed as
    B1's real run and log the same init_diagnostics as a side effect. Only the
    arm actually being compared belongs in the 4.3 table."""
    from steer_sweep.summary import _diagnostics_section

    diag = {"cos_theta_final_v_nat": 0.03, "cos_theta_final_v_prompt": 0.02,
            "theta_drift_l2": 0.35, "principal_angles_deg": [85.8, 87.9]}
    rows = [
        {"arm": "B1", "tag": "lambda_search_0.1", "tuning": True, "seed": 1,
         "init_diagnostics": diag},
        {"arm": "B1", "tag": "lr_search_0.0005", "tuning": True, "seed": 1,
         "init_diagnostics": diag},
        {"arm": "B1", "tag": "seed_1", "tuning": False, "seed": 1,
         "init_diagnostics": diag},
    ]
    text = "\n".join(_diagnostics_section(rows, {}))
    assert text.count("| B1 |") == 1


# ---------------------------------------------------------------------------
# Stage D-extended: dense (layer x alpha x beam) landscape map
# ---------------------------------------------------------------------------

def test_phase1_cells_are_70_unique_and_beam_matched():
    from steer_sweep import stage_d_ext as DX

    cells = DX.phase1_cells()
    assert len(cells) == 70
    assert len({c.config_hash() for c in cells}) == 70
    layers = {c.decoder_layers[0] for c in cells}
    alphas = {c.alpha for c in cells}
    beams = {c.num_beams for c in cells}
    assert layers == set(DX.PHASE1_LAYERS)
    assert alphas == set(DX.PHASE1_ALPHAS)
    assert beams == set(DX.PHASE1_BEAMS)
    for c in cells:
        assert c.site == C.SITE_DECODER
        assert c.coverage == C.COVERAGE_LOCAL
        assert c.combo_mode == C.COMBO_WEIGHTED
        assert c.weight_nat == 2.0 and c.weight_prompt == 0.5
        assert c.matched_energy is True
        assert c.scale_mode == C.SCALE_ACT_NORM


def test_baseline_for_beam_parity():
    from steer_sweep import stage_d_ext as DX

    assert DX.baseline_for(1) == C.PRIMARY_BASELINE
    assert DX.baseline_for(5) == C.BEAM_BASELINE


def test_assert_beam_parity_raises_on_mismatch():
    """Regression: a beam=5 cell must NEVER be scored against greedy C00, and
    vice versa. Section 3 requires this to be a hard error, not a warning."""
    from steer_sweep import stage_d_ext as DX

    cell = DX.phase1_cell(16, 1.0, 5)
    with pytest.raises(DX.BeamBaselineMismatchError):
        DX.assert_beam_parity(cell, C.PRIMARY_BASELINE)
    DX.assert_beam_parity(cell, C.BEAM_BASELINE)      # does not raise

    cell1 = DX.phase1_cell(16, 1.0, 1)
    with pytest.raises(DX.BeamBaselineMismatchError):
        DX.assert_beam_parity(cell1, C.BEAM_BASELINE)


def _dx_row(layer, alpha, beams, delta_mer, **extra):
    row = {
        "cell_id": f"L{layer}a{alpha}b{beams}", "layer": layer, "alpha": alpha,
        "num_beams": beams, "delta_mer": delta_mer,
        "PIER": 0.47 - delta_mer * 1.5, "PIER_baseline": 0.47,
        "verdict": "NO-GO",
    }
    row.update(extra)
    return row


def test_connected_components_finds_a_synthetic_block():
    from steer_sweep import stage_d_ext as DX

    rows = []
    for layer in DX.PHASE1_LAYERS:
        for alpha in DX.PHASE1_ALPHAS:
            for beams in DX.PHASE1_BEAMS:
                in_block = layer in (20, 24, 28) and alpha in (1.0, 2.0) and beams == 1
                rows.append(_dx_row(layer, alpha, beams, 0.003 if in_block else -0.001))

    comps = DX.connected_components(rows, 1)
    assert len(comps) == 1
    assert comps[0]["size"] == 6
    assert comps[0]["layer_range"] == [20, 28]
    assert comps[0]["alpha_range"] == [1.0, 2.0]
    assert DX.connected_components(rows, 5) == []      # no positive cells at beams=5


def test_connected_components_treats_isolated_point_as_size_one():
    from steer_sweep import stage_d_ext as DX

    rows = [_dx_row(layer, alpha, 1,
                    0.003 if (layer == 16 and alpha == 1.0) else -0.001)
           for layer in DX.PHASE1_LAYERS for alpha in DX.PHASE1_ALPHAS]
    comps = DX.connected_components(rows, 1)
    assert len(comps) == 1 and comps[0]["size"] == 1


def test_plateau_score_zero_for_isolated_point():
    from steer_sweep import stage_d_ext as DX

    rows = [_dx_row(layer, alpha, 1,
                    0.003 if (layer == 16 and alpha == 1.0) else -0.001)
           for layer in DX.PHASE1_LAYERS for alpha in DX.PHASE1_ALPHAS]
    scores = DX.plateau_scores(rows, 1)
    assert len(scores) == 1
    assert scores[0]["plateau_score"] == 0
    assert scores[0]["isolated"] is True


def test_q3_chance_count_matches_nominal_rate():
    from steer_sweep import stage_d_ext as DX

    rows = [_dx_row(4, 0.25, 1, 0.0, verdict="GO") for _ in range(1)] + \
          [_dx_row(4, 0.5, 1, 0.0, verdict="NO-GO") for _ in range(69)]
    q3 = DX.q3_chance_count(rows)
    assert q3["n_cells"] == 70
    assert math.isclose(q3["expected_false_positives"], 3.5)
    assert q3["observed_go"] == 1


def test_q2_paired_analysis_independent_when_gains_match():
    """If gain_b1 == gain_b5 everywhere, composition must read INDEPENDENT,
    not SUBSTITUTES or COMPLEMENTS from noise."""
    from steer_sweep import stage_d_ext as DX

    rows = []
    for layer in DX.PHASE1_LAYERS:
        for alpha in DX.PHASE1_ALPHAS:
            for beams in DX.PHASE1_BEAMS:
                pier = 0.47 - 0.01          # identical PIER regardless of beams
                rows.append({**_dx_row(layer, alpha, beams, 0.0), "PIER": pier,
                            "PIER_baseline": 0.47})
    q2 = DX.q2_paired_analysis(rows)
    assert q2["composition"] == "INDEPENDENT"
    assert math.isclose(q2["mean_difference"], 0.0, abs_tol=1e-9)


def test_phase2_layers_deduplicated_across_alphas():
    """Regression: a connected component spans multiple alphas, so its
    `members` list each layer once per positive alpha. phase2a/phase2b must
    deduplicate before computing gaps or the sqrt(k) energy correction --
    caught before this: a 3-layer, 2-alpha component produced decoder_layers
    (20, 20, 24, 24, 28, 28) and a sqrt(6) correction instead of sqrt(3)."""
    from steer_sweep import stage_d_ext as DX

    component = {
        "size": 6, "layer_range": [20, 28], "alpha_range": [1.0, 2.0],
        "members": [
            {"layer": 20, "alpha": 1.0, "cell_id": "a", "delta_mer": 0.0028},
            {"layer": 20, "alpha": 2.0, "cell_id": "b", "delta_mer": 0.0031},
            {"layer": 24, "alpha": 1.0, "cell_id": "c", "delta_mer": 0.0035},
            {"layer": 24, "alpha": 2.0, "cell_id": "d", "delta_mer": 0.0033},
            {"layer": 28, "alpha": 1.0, "cell_id": "e", "delta_mer": 0.0029},
            {"layer": 28, "alpha": 2.0, "cell_id": "f", "delta_mer": 0.0032},
        ],
    }
    fine = DX.phase2a_cells(component)
    assert {c.decoder_layers[0] for c in fine} == {21, 22, 23, 25, 26, 27}

    pairs = DX.phase2b_cells(component)
    assert len(pairs) == len(DX.PHASE1_BEAMS)
    for matched, per_layer in pairs:
        assert matched.decoder_layers == (20, 24, 28)
        assert per_layer.decoder_layers == (20, 24, 28)
        assert matched.matched_energy is True
        assert per_layer.matched_energy is False
        # per-layer alpha undoes the 1/sqrt(k) split from matched_energy=True
        assert math.isclose(per_layer.alpha, matched.alpha * math.sqrt(3), rel_tol=1e-9)


def test_phase2b_returns_nothing_for_a_single_layer_component():
    from steer_sweep import stage_d_ext as DX

    component = {"size": 1, "layer_range": [16, 16], "alpha_range": [1.0, 1.0],
                "members": [{"layer": 16, "alpha": 1.0, "cell_id": "a", "delta_mer": 0.003}]}
    assert DX.phase2b_cells(component) == []


def test_small_alpha_sanity_uses_alpha_eff_as_the_ratio():
    from steer_sweep import stage_d_ext as DX

    rows = [
        {"cell_id": "ok", "layer": 16, "alpha": 0.25, "num_beams": 1,
         "delta_over_activation_ratio": 0.023, "decoder_act_norm_mean": 12.0},
        {"cell_id": "bad", "layer": 20, "alpha": 0.25, "num_beams": 1,
         "delta_over_activation_ratio": 0.09, "decoder_act_norm_mean": 12.0},
        {"cell_id": "not_alpha_25", "layer": 16, "alpha": 1.0, "num_beams": 1,
         "delta_over_activation_ratio": 0.5, "decoder_act_norm_mean": 12.0},
    ]
    report = DX.check_small_alpha_sanity(rows)
    assert report["passed"] is False
    assert len(report["entries"]) == 2          # only the alpha=0.25 rows
    assert len(report["failing"]) == 1
    assert report["failing"][0]["layer"] == 20


def test_s_audit_flags_deviation_from_expected_pattern():
    from steer_sweep import stage_d_ext as DX

    rows = [{"cell_id": "ok", "S_planned": 116, "S_observed": 114, "num_beams": 1},
           {"cell_id": "bad", "S_planned": 116, "S_observed": 100, "num_beams": 1}]
    report = DX.check_s_audit(rows)
    assert report["passed"] is False
    assert report["n_deviations"] == 1
    assert report["deviations"][0]["cell_id"] == "bad"


def test_s_audit_scales_with_beams_and_layer_count():
    """Regression: |S| observed legitimately scales with num_beams (each
    steered position is counted once per beam-row) AND with layer count, and
    beams=5 loses exactly one utterance's oracle hit relative to beams=1 (114
    vs 113 out of 116) -- discovered when the naive beams=1-only expectation
    of 114 flagged all 35 real beam=5 Phase 1 cells as 'deviations' even
    though every one of them showed the identical, internally consistent
    S_observed=565=113*5 (job 42882). The audit must not re-misreport this."""
    from steer_sweep import stage_d_ext as DX

    rows = [
        {"cell_id": "single_b1", "S_planned": 116, "S_observed": 114, "num_beams": 1},
        {"cell_id": "single_b5", "S_planned": 116, "S_observed": 565, "num_beams": 5},
        {"cell_id": "two_layer_b1", "S_planned": 232, "S_observed": 228, "num_beams": 1},
        {"cell_id": "two_layer_b5", "S_planned": 232, "S_observed": 1130, "num_beams": 5},
        {"cell_id": "four_layer_b1", "S_planned": 464, "S_observed": 456, "num_beams": 1},
    ]
    report = DX.check_s_audit(rows)
    assert report["passed"] is True
    assert report["n_deviations"] == 0

    # a genuinely wrong beams=5 value (the old bare 114*1 expectation) must
    # still be caught
    bad = [{"cell_id": "wrong", "S_planned": 116, "S_observed": 570, "num_beams": 5}]
    bad_report = DX.check_s_audit(bad)
    assert bad_report["passed"] is False
    assert bad_report["deviations"][0]["expected_observed"] == 565


def test_heatmap_gain_difference_shape_and_values():
    from steer_sweep import stage_d_ext as DX

    rows = []
    for layer in DX.PHASE1_LAYERS:
        for alpha in DX.PHASE1_ALPHAS:
            for beams in DX.PHASE1_BEAMS:
                pier = 0.40 if (layer == 16 and alpha == 1.0 and beams == 5) else 0.47
                rows.append({**_dx_row(layer, alpha, beams, 0.0), "PIER": pier,
                            "PIER_baseline": 0.47})
    hm = DX.heatmap_gain_difference(rows)
    values = hm["values"]
    li, ai = DX.PHASE1_LAYERS.index(16), DX.PHASE1_ALPHAS.index(1.0)
    assert values[li][ai] > 0                    # gain_b5 > gain_b1 there
    other_li, other_ai = DX.PHASE1_LAYERS.index(4), DX.PHASE1_ALPHAS.index(0.25)
    assert math.isclose(values[other_li][other_ai], 0.0, abs_tol=1e-9)
