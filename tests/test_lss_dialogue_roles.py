"""Dialogue-atomic role allocation.

The four structural requirements are the reason this partition exists, so each
one is tested with a fixture that violates it: a check that only ever sees valid
input proves nothing about what it would reject.
"""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.lss.balance import BalanceSpec
from csasr.lss.dialogue_roles import (ALLOCATED_ROLES, DIALOGUE_ROLES, TEST_ROLE,
                                      allocate_dialogues, allocation_targets,
                                      assert_dialogue_partition,
                                      assert_no_dialogue_straddles,
                                      build_dialogue_roles, dialogue_features,
                                      role_overlap_matrix)
from csasr.lss.roles import RoleError

TOPICS = ["personal topics", "entertainment", "education", "job"]


def _manifest(n_dialogues: int = 20, n_test: int = 3, per_conversation: int = 4
              ) -> pd.DataFrame:
    """A corpus of two-sided dialogues, the last `n_test` officially test."""
    rows = []
    for d in range(1, n_dialogues + 1):
        split = "test" if d > n_dialogues - n_test else ("dev" if d % 7 == 0 else "train")
        for side in (0, 1):
            number = 2 * (d - 1) + side + 1
            conversation = f"ZH-CN_U{number:04d}"
            for u in range(per_conversation):
                rows.append({
                    "utterance_id": f"{conversation}_S0_{u}",
                    "conversation_id": conversation,
                    "speaker_id": f"{conversation}_S0",
                    "dialogue_id": f"CSD{d:04d}",
                    "corpus_speaker_id": number,
                    "official_split": split,
                    "duration_sec": 3.0 + (d % 5),
                    "transcript_raw": "我 like 这个 project",
                    "transcript_normalized": "我 like 这个 project",
                    "contains_code_switch": True,
                    "contains_en": True,
                    "contains_zh": True,
                    "gender": "F" if (d + side) % 2 else "M",
                    "device": "Android" if side == 0 else "iphone",
                    "region": ["Fujian", "Beijing", "Sichuan"][d % 3],
                    "topic": TOPICS[(d + u) % len(TOPICS)],
                })
    return pd.DataFrame(rows)


def _cfg(targets: dict[str, int] | None = None) -> dict:
    return {
        "roles": {"dialogue_roles": {"version": "dialogue-test",
                                     "targets": targets or {
                                         "D-construct": 5, "loc-train": 3,
                                         "util-train": 3, "router-calib": 2,
                                         "D-dev-select": 3, "D-dev-confirm": 1}},
                  "balance": {"draws": 200}},
        "seeds": {"role_assignment": 20260810},
    }


def _features(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    return dialogue_features(manifest if manifest is not None else _manifest())


# --------------------------------------------------------------------------
# features


def test_dialogue_features_pool_both_sides():
    features = _features()
    assert len(features) == 20
    assert set(features["conversations"]) == {2}
    assert features["utterances"].sum() == 20 * 2 * 4
    # composition, not one side's value
    assert set(features["device"]) == {"Android/iphone"}
    assert set(features["gender"]) <= {"FF", "FM", "MM"}


def test_dialogue_features_reject_a_split_straddle():
    manifest = _manifest()
    victim = manifest["conversation_id"] == "ZH-CN_U0001"
    manifest.loc[victim, "official_split"] = "test"
    with pytest.raises(RoleError, match="straddle official splits"):
        dialogue_features(manifest)


# --------------------------------------------------------------------------
# the four required assertions, each with a violating fixture


def _assignment(features: pd.DataFrame) -> dict[str, str]:
    pool = features[features["official_split"] != "test"]["dialogue_id"].tolist()
    targets = _cfg()["roles"]["dialogue_roles"]["targets"]
    out, i = {}, 0
    for role, n in targets.items():
        for dialogue in pool[i:i + n]:
            out[dialogue] = role
        i += n
    return out


def test_assertion_1_every_dialogue_in_exactly_one_role():
    features = _features()
    good = _assignment(features)
    assert assert_dialogue_partition(good, features)["union_equals_corpus"] is True

    missing = dict(good)
    missing.pop(sorted(missing)[0])
    with pytest.raises(RoleError, match="union does not equal"):
        assert_dialogue_partition(missing, features)


def test_assertion_2_role_sets_are_pairwise_disjoint():
    """A dict cannot express an overlap, so the frames are checked too."""
    manifest = _manifest()
    roles, _, _ = build_dialogue_roles(_cfg(), manifest)
    assert assert_no_dialogue_straddles(roles)["dialogue_disjoint"] is True

    shared = dict(roles)
    stolen = roles["loc-train"].head(4)
    shared["D-construct"] = pd.concat([roles["D-construct"], stolen], ignore_index=True)
    with pytest.raises(RoleError, match="roles overlap"):
        assert_no_dialogue_straddles(shared)


def test_assertion_3_d_test_is_official_test_only():
    features = _features()
    good = _assignment(features)
    intruder = dict(good)
    non_test = sorted(features.loc[features["official_split"] != "test", "dialogue_id"])[0]
    intruder[non_test] = TEST_ROLE
    with pytest.raises(RoleError, match="non-test dialogue"):
        assert_dialogue_partition(intruder, features)

    # and the reverse: an official-test dialogue handed to another role
    hijacked = dict(good)
    test_dialogue = sorted(features.loc[features["official_split"] == "test",
                                        "dialogue_id"])[0]
    hijacked[test_dialogue] = "D-construct"
    with pytest.raises(RoleError, match="D-test is the official test split"):
        assert_dialogue_partition(hijacked, features)


def test_assertion_4_union_is_exactly_the_corpus():
    features = _features()
    foreign = dict(_assignment(features))
    foreign["CSD9999"] = "D-construct"
    with pytest.raises(RoleError, match="union does not equal"):
        assert_dialogue_partition(foreign, features)


def test_an_unknown_role_raises():
    features = _features()
    bad = dict(_assignment(features))
    bad[sorted(bad)[0]] = "D-mystery"
    with pytest.raises(RoleError, match="unknown role"):
        assert_dialogue_partition(bad, features)


# --------------------------------------------------------------------------
# determinism and coverage


def test_same_seed_reproduces_the_identical_allocation():
    features = _features()
    spec = BalanceSpec.from_cfg({"draws": 200})
    targets = allocation_targets(_cfg())
    first, report_a, _ = allocate_dialogues(features, targets, spec, seed=20260810)
    again, report_b, _ = allocate_dialogues(features, targets, spec, seed=20260810)
    assert first == again
    assert report_a["accepted_score"] == report_b["accepted_score"]

    other, _, _ = allocate_dialogues(features, targets, spec, seed=20260811)
    assert other != first          # the seed is doing work, not decoration


def test_no_dialogue_straddles_roles_after_a_real_build():
    roles, report, _ = build_dialogue_roles(_cfg(), _manifest())
    seen: dict[str, str] = {}
    for role, frame in roles.items():
        for dialogue in frame["dialogue_id"].unique():
            assert dialogue not in seen, f"{dialogue} in {seen.get(dialogue)} and {role}"
            seen[dialogue] = role
    assert len(seen) == 20
    assert report["structure"]["dialogues_per_role"][TEST_ROLE] == 3
    # both sides of every dialogue travel together
    for frame in roles.values():
        assert set(frame.groupby("dialogue_id")["conversation_id"].nunique()) == {2}


def test_targets_must_cover_the_non_test_pool_exactly():
    short = _cfg({"D-construct": 4, "loc-train": 3, "util-train": 3,
                  "router-calib": 2, "D-dev-select": 3, "D-dev-confirm": 1})
    with pytest.raises(RoleError, match="targets sum to"):
        build_dialogue_roles(short, _manifest())


def test_targets_are_configured_not_hard_coded():
    with pytest.raises(RoleError, match="not configured"):
        allocation_targets({"roles": {}})
    with pytest.raises(RoleError, match="missing"):
        allocation_targets({"roles": {"dialogue_roles": {
            "targets": {"D-construct": 17}}}})
    assert set(allocation_targets(_cfg())) == set(ALLOCATED_ROLES)


def test_official_test_dialogues_are_never_in_the_allocation_pool():
    features = _features()
    targets = allocation_targets(_cfg())
    assignment, _, _ = allocate_dialogues(features, targets,
                                          BalanceSpec.from_cfg({"draws": 50}),
                                          seed=1)
    test_dialogues = set(features.loc[features["official_split"] == "test", "dialogue_id"])
    assert not (set(assignment) & test_dialogues)


# --------------------------------------------------------------------------
# balance


def test_balance_failures_are_reported_and_never_swallowed():
    """Balance is a reported diagnostic, not a gate -- but it must be visible.

    On the real corpus four gated covariates exceed `max_categorical_tv`, because
    a dialogue's gender and device become three-level *compositions* over roles
    of 8-20 units. That is a finding for a human, so the one thing the code must
    never do is drop it.
    """
    skewed = _manifest()
    # make gender composition track the dialogue index, so any split is skewed
    skewed["gender"] = ["F" if int(d[3:]) <= 10 else "M" for d in skewed["dialogue_id"]]
    _, report, diagnostics = build_dialogue_roles(_cfg(), skewed)
    failures = report["balance_failures"]
    over = diagnostics[diagnostics["gated"] & diagnostics["tv"].notna()
                       & (diagnostics["tv"] > BalanceSpec().max_categorical_tv)]
    assert len(over), "fixture no longer produces an over-threshold covariate"
    assert len(failures) >= len(over)
    assert report["max_categorical_tv"] == pytest.approx(float(over["tv"].max()))


def test_topic_and_gender_balance_stay_within_the_configured_tolerance():
    spec = BalanceSpec.from_cfg({"draws": 400})
    cfg = _cfg({"D-construct": 10, "loc-train": 6, "util-train": 6,
                "router-calib": 4, "D-dev-select": 6, "D-dev-confirm": 2})
    cfg["roles"]["balance"]["draws"] = 400
    roles, report, diagnostics = build_dialogue_roles(cfg, _manifest(n_dialogues=40,
                                                                     n_test=6))
    gated = diagnostics[diagnostics["gated"] & diagnostics["tv"].notna()]
    assert not gated.empty
    assert float(gated["tv"].max()) <= spec.max_categorical_tv
    assert float(diagnostics["smd"].abs().max()) <= spec.max_abs_smd
    # topics are optimized as a distributional covariate and reported per role
    assert all(r["topics"] for r in report["roles"].values())
    # rerandomization must beat what an arbitrary draw would give
    assert report["balance"]["beats_random_median"] is True


def test_overlap_matrix_counts_shared_dialogues():
    """The before/after table must be able to *show* an overlap, not just zero."""
    dialogue_of = {"c1": "d1", "c2": "d1", "c3": "d2"}
    split = role_overlap_matrix({"c1": "A", "c2": "B", "c3": "B"}, dialogue_of)
    assert int(split.loc["A", "B"]) == 1          # d1 straddles A and B
    whole = role_overlap_matrix({"c1": "A", "c2": "A", "c3": "B"}, dialogue_of)
    assert int(whole.loc["A", "B"]) == 0


def test_report_states_the_unit_and_the_seed():
    _, report, _ = build_dialogue_roles(_cfg(), _manifest())
    assert report["allocation_unit"] == "dialogue"
    assert report["seed"] == 20260810
    assert report["seed_purpose"] == "role_assignment"
    assert len(report["assignment_hash"]) == 64
    assert set(report["structure"]["dialogues_per_role"]) == set(DIALOGUE_ROLES)


def test_output_guard_refuses_protected_trees_and_existing_manifests(tmp_path):
    from csasr.experiments.dialogue_roles_build import resolve_output

    with pytest.raises(SystemExit, match="protected tree"):
        resolve_output("/mnt/data/x/artifacts_lss/manifests/roles")
    (tmp_path / "role_D-construct.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="immutable"):
        resolve_output(tmp_path)
    assert resolve_output(tmp_path / "fresh") == tmp_path / "fresh"
