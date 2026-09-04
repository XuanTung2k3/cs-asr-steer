"""LSS unit test: data roles are disjoint, balanced, deterministic and locked."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csasr.lss.balance import (
    BalanceSpec,
    NoFeasibleAssignment,
    ProposalSpec,
    assignment_hash,
    balanced_assignment,
    imbalance,
)
from csasr.lss.roles import ROLES, RoleError, assert_role, assert_roles_disjoint

SPEC = BalanceSpec(continuous=("hours", "cs_rate", "embedded_en_units"),
                   categorical=("device",), distributional=("topics",), draws=400)


def _features(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "conversation_id": [f"C{i:03d}" for i in range(n)],
        # deliberately correlated with the identifier, which is what makes
        # sorted slicing unsafe
        "hours": np.linspace(0.1, 1.3, n) + rng.normal(0, 0.02, n),
        "cs_rate": np.linspace(0.03, 0.4, n),
        "embedded_en_units": np.arange(n) * 3 + 5,
        "device": ["mic" if i % 3 else "phone" for i in range(n)],
        "topics": [["study"] if i % 2 else ["work", "travel"] for i in range(n)],
    })


def _targets(n=40):
    return {"D-construct": n // 2, "loc-train": n // 4,
            "util-train": n // 8, "router-calib": n - (n // 2 + n // 4 + n // 8)}


def test_assignment_is_deterministic_for_a_seed():
    f = _features()
    a1, _ = balanced_assignment(f, _targets(), SPEC, seed=11)
    a2, _ = balanced_assignment(f, _targets(), SPEC, seed=11)
    assert a1 == a2
    assert assignment_hash(a1) == assignment_hash(a2)


def test_a_different_seed_gives_a_different_assignment():
    f = _features()
    a1, _ = balanced_assignment(f, _targets(), SPEC, seed=11)
    a2, _ = balanced_assignment(f, _targets(), SPEC, seed=12)
    assert a1 != a2


def test_every_conversation_is_assigned_exactly_once_with_exact_sizes():
    f = _features()
    targets = _targets()
    assignment, _ = balanced_assignment(f, targets, SPEC, seed=5)
    assert set(assignment) == set(f["conversation_id"])
    counts = pd.Series(assignment).value_counts().to_dict()
    assert counts == targets


def test_target_sizes_must_cover_the_corpus():
    with pytest.raises(ValueError, match="sum to"):
        balanced_assignment(_features(), {"D-construct": 3}, SPEC, seed=1)


def test_chosen_assignment_beats_the_random_median():
    """The point of rerandomization: better than what chance hands you."""
    _, report = balanced_assignment(_features(), _targets(), SPEC, seed=7)
    assert report["accepted_score"] < report["null_score_median"]
    assert report["beats_random_median"] is True


def test_balanced_assignment_beats_sorted_slicing():
    """Plan v1 sliced a sorted list; on an identifier-correlated corpus that is
    systematically worse than a balanced draw."""
    f = _features()
    targets = _targets()
    sliced: dict[str, str] = {}
    start = 0
    for role, size in targets.items():
        for cid in f["conversation_id"][start:start + size]:
            sliced[cid] = role
        start += size
    sliced_max = imbalance(f, sliced, SPEC)["smd"].abs().max()

    assignment, _ = balanced_assignment(f, targets, SPEC, seed=3)
    balanced_max = imbalance(f, assignment, SPEC)["smd"].abs().max()
    assert balanced_max < sliced_max


def test_infeasible_draws_are_never_returned_as_a_fallback():
    spec = BalanceSpec(continuous=("hours",), categorical=(), distributional=(),
                       max_abs_smd=-0.01, max_categorical_tv=0.15, draws=25)
    with pytest.raises(NoFeasibleAssignment) as caught:
        balanced_assignment(_features(), _targets(), spec, seed=3)
    report = caught.value.report
    assert report["feasible_draws"] == 0
    assert report["feasibility_rate"] == 0.0
    assert report["accepted_proposal_mechanism"] is None
    assert "accepted_score" not in report


def test_every_returned_assignment_satisfies_every_gate():
    assignment, report = balanced_assignment(_features(), _targets(), SPEC, seed=9)
    table = imbalance(_features(), assignment, SPEC)
    assert bool(table[table["gated"]]["passed"].all())
    assert report["accepted_max_abs_smd"] <= SPEC.max_abs_smd
    assert report["accepted_max_categorical_tv"] <= SPEC.max_categorical_tv


def test_scoring_formula_and_predeclared_weight_are_unchanged():
    spec = BalanceSpec(continuous=("hours", "cs_rate", "embedded_en_units"),
                       categorical=("device",), distributional=("topics",),
                       draws=400, ungated_weight=0.3)
    _, report = balanced_assignment(_features(), _targets(), spec, seed=7)
    expected = max(report["accepted_max_abs_smd"],
                   report["accepted_max_categorical_tv"]) + \
        0.3 * report["accepted_max_ungated_tv"]
    assert report["accepted_score"] == pytest.approx(expected)
    assert report["ungated_weight"] == 0.3


def test_stratified_proposal_is_random_and_seeded():
    f = _features()
    proposal = [ProposalSpec("stratified_deal", 200, ("device",))]
    first, first_report = balanced_assignment(
        f, _targets(), SPEC, seed=11, proposals=proposal)
    again, _ = balanced_assignment(f, _targets(), SPEC, seed=11, proposals=proposal)
    other, _ = balanced_assignment(f, _targets(), SPEC, seed=12, proposals=proposal)
    assert first == again
    assert first != other
    assert first_report["proposal_reports"][0]["mechanism"] == "stratified_deal"
    assert first_report["proposal_reports"][0]["feasible_draws"] > 0


def test_imbalance_is_zero_on_a_perfectly_balanced_corpus():
    f = pd.DataFrame({
        "conversation_id": [f"C{i}" for i in range(8)],
        "hours": [1.0] * 8, "cs_rate": [0.2] * 8, "embedded_en_units": [10] * 8,
        "device": ["mic"] * 8, "topics": [["study"]] * 8,
    })
    assignment = {f"C{i}": ("a" if i < 4 else "b") for i in range(8)}
    table = imbalance(f, assignment, SPEC)
    assert float(table["smd"].abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(table["tv"].max(skipna=True)) == pytest.approx(0.0, abs=1e-12)
    assert bool(table["passed"].all())


def test_assignment_hash_changes_when_the_assignment_changes():
    a = {"C1": "D-construct", "C2": "loc-train"}
    b = {"C1": "loc-train", "C2": "D-construct"}
    assert assignment_hash(a) != assignment_hash(b)
    assert assignment_hash(a) == assignment_hash(dict(reversed(list(a.items()))))


def _role_frame(role, convs, split="train"):
    return pd.DataFrame([{
        "utterance_id": f"{c}_{i}", "conversation_id": c, "speaker_id": c,
        "official_split": split, "role": role, "duration_sec": 5.0,
        "contains_code_switch": True, "transcript_raw": "我 用 machine 做",
    } for c in convs for i in range(2)])


def test_disjointness_check_accepts_clean_roles_and_rejects_overlap():
    roles = {"D-construct": _role_frame("D-construct", ["C1", "C2"]),
             "loc-train": _role_frame("loc-train", ["C3"])}
    report = assert_roles_disjoint(roles)
    assert report["conversation_disjoint"] is True

    roles["loc-train"] = _role_frame("loc-train", ["C2", "C3"])
    with pytest.raises(RoleError, match="overlap"):
        assert_roles_disjoint(roles)


def test_test_rows_may_not_appear_in_another_role():
    roles = {"D-construct": _role_frame("D-construct", ["C1"], split="test"),
             "D-test": _role_frame("D-test", ["C9"], split="test")}
    with pytest.raises(RoleError, match="test-split rows"):
        assert_roles_disjoint(roles)


def test_assert_role_guards_training_entry_points():
    frame = _role_frame("loc-train", ["C1"])
    assert_role(frame, "loc-train")
    assert_role(frame, ["loc-train", "util-train"])
    with pytest.raises(RoleError):
        assert_role(frame, "D-construct")
    with pytest.raises(RoleError, match="no `role` column"):
        assert_role(frame.drop(columns=["role"]), "loc-train")


def test_role_names_are_the_ones_the_proposal_defines():
    assert ROLES == ("D-construct", "loc-train", "util-train", "router-calib",
                     "D-dev-select", "D-dev-confirm", "D-test")
