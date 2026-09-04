"""Dialogue-atomic role allocation.

The four structural requirements are the reason this partition exists, so each
one is tested with a fixture that violates it: a check that only ever sees valid
input proves nothing about what it would reject.
"""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.lss.balance import BalanceSpec, NoFeasibleAssignment
from csasr.lss.dialogue_roles import (ALLOCATED_ROLES, DIALOGUE_ROLES, TEST_ROLE,
                                      allocate_dialogues, allocation_targets,
                                      assert_dialogue_partition,
                                      assert_no_dialogue_straddles,
                                      build_dialogue_roles, dialogue_features,
                                      partition_parameters, role_overlap_matrix,
                                      role_partition_fingerprint)
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
                                         "D-dev-select": 3, "D-dev-confirm": 1},
                                     "partition": {
                                         "version": "dialogue-test-partition-v1",
                                         "ungated_weight": 0.3,
                                         "proposals": [{
                                             "mechanism": "uniform_permutation",
                                             "draws": 200,
                                         }],
                                     }},
                  # Structural tests use a vacuous balance design; tests below
                  # exercise the real gated covariates and forced no-go path.
                  "balance": {"continuous": [], "categorical": [],
                              "distributional": [], "draws": 200}},
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
    spec = BalanceSpec.from_cfg({"continuous": [], "categorical": [],
                                 "distributional": [], "draws": 200})
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
                                          BalanceSpec.from_cfg({"continuous": [],
                                                                "categorical": [],
                                                                "distributional": [],
                                                                "draws": 50}),
                                          seed=1)
    test_dialogues = set(features.loc[features["official_split"] == "test", "dialogue_id"])
    assert not (set(assignment) & test_dialogues)


# --------------------------------------------------------------------------
# balance


def test_gated_balance_failure_refuses_role_construction():
    """An infeasible draw budget is a no-go, never a publishable diagnostic."""
    skewed = _manifest()
    cfg = _cfg()
    cfg["roles"]["balance"] = {
        "continuous": ["hours"], "categorical": [], "distributional": [],
        "max_abs_smd": 0.0, "max_categorical_tv": 0.15, "draws": 20,
    }
    cfg["roles"]["dialogue_roles"]["partition"]["proposals"][0]["draws"] = 20
    with pytest.raises(NoFeasibleAssignment) as caught:
        build_dialogue_roles(cfg, skewed)
    assert caught.value.report["feasible_draws"] == 0
    assert caught.value.report["accepted_proposal_mechanism"] is None


def test_topic_and_gender_balance_stay_within_the_configured_tolerance():
    spec = BalanceSpec.from_cfg({"continuous": ["hours", "utterances"],
                                 "categorical": ["gender", "device", "region"],
                                 "distributional": ["topics"], "draws": 400,
                                 "ungated_weight": 0.3})
    cfg = _cfg({"D-construct": 10, "loc-train": 6, "util-train": 6,
                "router-calib": 4, "D-dev-select": 6, "D-dev-confirm": 2})
    cfg["roles"]["balance"] = {
        "continuous": ["hours", "utterances"],
        "categorical": ["gender", "device", "region"],
        "distributional": ["topics"], "gated_categorical": ["gender", "device"],
        "max_abs_smd": 0.25, "max_categorical_tv": 0.15, "draws": 400,
    }
    cfg["roles"]["dialogue_roles"]["partition"]["proposals"] = [{
        "mechanism": "stratified_deal", "draws": 400,
        "stratify_by": ["gender", "device"],
    }]
    roles, report, diagnostics = build_dialogue_roles(cfg, _manifest(n_dialogues=40,
                                                                     n_test=6))
    gated = diagnostics[diagnostics["gated"] & diagnostics["tv"].notna()]
    assert not gated.empty
    assert float(gated["tv"].max()) <= spec.max_categorical_tv
    assert float(diagnostics["smd"].abs().max()) <= spec.max_abs_smd
    # topics are optimized as a distributional covariate and reported per role
    assert all(r["topics"] for r in report["roles"].values())
    # the accepted draw is the objective argmin within the declared proposal
    assert report["balance"]["beats_proposal_median"] is True
    assert report["balance"]["beats_random_median"] is None


def test_partition_parameters_require_explicit_weight_and_version():
    cfg = _cfg()
    partition = cfg["roles"]["dialogue_roles"]["partition"]
    partition.pop("ungated_weight")
    with pytest.raises(RoleError, match="ungated_weight is required"):
        partition_parameters(cfg)

    cfg = _cfg()
    cfg["roles"]["dialogue_roles"]["partition"].pop("version")
    with pytest.raises(RoleError, match="partition.version is required"):
        partition_parameters(cfg)


def test_partition_fingerprint_binds_weight_proposals_and_source():
    cfg = _cfg()
    version, spec, proposals = partition_parameters(cfg)
    targets = allocation_targets(cfg)
    first, payload = role_partition_fingerprint(
        version=version, targets=targets, seed=20260810, spec=spec,
        proposals=proposals, source_manifest_sha256="source-a")
    source_changed, _ = role_partition_fingerprint(
        version=version, targets=targets, seed=20260810, spec=spec,
        proposals=proposals, source_manifest_sha256="source-b")
    weighted = BalanceSpec.from_cfg({**cfg["roles"]["balance"],
                                     "ungated_weight": 0.4})
    weight_changed, _ = role_partition_fingerprint(
        version=version, targets=targets, seed=20260810, spec=weighted,
        proposals=proposals, source_manifest_sha256="source-a")
    assert first not in {source_changed, weight_changed}
    assert payload["balance"]["ungated_weight"] == 0.3
    assert payload["proposals"][0]["mechanism"] == "uniform_permutation"


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
    from csasr.lss.manifest import manifest_path

    with pytest.raises(SystemExit, match="protected tree"):
        resolve_output("/mnt/data/x/artifacts_lss/manifests/roles")
    (tmp_path / "role_D-construct.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="immutable"):
        resolve_output(tmp_path)
    sidecar_only = tmp_path / "sidecar-only"
    manifest_path(sidecar_only).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="sidecar"):
        resolve_output(sidecar_only)
    assert resolve_output(tmp_path / "fresh") == tmp_path / "fresh"


def test_cli_gated_no_go_publishes_no_role_generation(tmp_path, monkeypatch, capsys):
    from csasr.experiments import dialogue_roles_build as command

    source = tmp_path / "source.parquet"
    _manifest().to_parquet(source, index=False)
    cfg = _cfg()
    cfg["roles"]["balance"] = {
        "continuous": ["hours"], "categorical": [], "distributional": [],
        "max_abs_smd": -0.01, "max_categorical_tv": 0.15, "draws": 10,
    }
    cfg["roles"]["dialogue_roles"]["partition"]["proposals"][0]["draws"] = 10
    monkeypatch.setattr(command, "load_config", lambda _: cfg)
    destination = tmp_path / "roles-v2"
    rc = command.main([
        "--config", "ignored.yaml", "--manifest", str(source),
        "--output-dir", str(destination),
    ])
    payload = __import__("json").loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["state"] == "completed_no_go"
    assert payload["published"] is False
    assert payload["balance"]["feasible_draws"] == 0
    assert not destination.exists()


def test_interrupted_generation_never_exposes_partial_destination(tmp_path, monkeypatch):
    from csasr.experiments import dialogue_roles_build as command

    roles, report, diagnostics = build_dialogue_roles(_cfg(), _manifest())
    original = command.manifest_mod.publish_frame
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(command.manifest_mod, "publish_frame", interrupted)
    destination = tmp_path / "roles-v2"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        command.publish_generation(
            out_dir=destination, roles=roles, report=report,
            diagnostics=diagnostics, overlap={"before": {}, "after": {}},
            cfg=_cfg(), parent={"path": "source", "sha256": "abc"})
    assert not destination.exists()
    assert not list(tmp_path.glob(".roles-v2.attempt-*"))


def test_successful_generation_is_complete_and_manifests_use_final_paths(tmp_path):
    from csasr.experiments import dialogue_roles_build as command

    roles, report, diagnostics = build_dialogue_roles(_cfg(), _manifest())
    destination = tmp_path / "roles-v2"
    command.publish_generation(
        out_dir=destination, roles=roles, report=report,
        diagnostics=diagnostics, overlap={"before": {}, "after": {}},
        cfg=_cfg(), parent={"path": "source", "sha256": "abc"})
    complete = __import__("json").loads(
        (destination / command.GENERATION_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert complete["complete"] is True
    assert complete["role_partition_fingerprint"] == report["partition_fingerprint"]
    role_path = destination / "role_D-construct.parquet"
    sidecar = command.manifest_mod.load(role_path)
    assert sidecar is not None
    assert sidecar["path"] == str(role_path)
    assert sidecar["role_partition_fingerprint"] == report["partition_fingerprint"]


# --------------------------------------------------------------------------
# the RNG derivation, and the allocation it actually produced


def test_fingerprint_records_the_rng_derivation():
    """The draw stream is part of the allocation rule, so it is part of identity.

    Two derivations enumerate different candidates from the same seed. A
    fingerprint that omitted the stream would call those two partitions
    identical, which is exactly the drift that went unnoticed between v1 and v2.
    """
    from csasr.lss.balance import RNG_DERIVATION

    cfg = _cfg()
    version, spec, proposals = partition_parameters(cfg)
    _, payload = role_partition_fingerprint(
        version=version, targets=allocation_targets(cfg), seed=20260810,
        spec=spec, proposals=proposals, source_manifest_sha256="source-a")

    assert payload["fingerprint_version"].endswith("-v2")
    recorded = payload["rng_derivation"]
    assert recorded == RNG_DERIVATION
    assert "SeedSequence" in recorded["current"]
    assert recorded["previous"] == "numpy.random.default_rng(seed)"
    # the record must say the two allocations are not draw-for-draw comparable
    assert "NOT a counterfactual reproduction" in recorded["comparability"]
    assert "not comparable draw-for-draw" in recorded["comparability"]


def test_changing_the_rng_derivation_changes_the_fingerprint(monkeypatch):
    """A future stream change must not be able to reuse this partition identity."""
    import csasr.lss.dialogue_roles as module

    cfg = _cfg()
    version, spec, proposals = partition_parameters(cfg)
    targets = allocation_targets(cfg)
    before, _ = role_partition_fingerprint(
        version=version, targets=targets, seed=20260810, spec=spec,
        proposals=proposals, source_manifest_sha256="source-a")
    monkeypatch.setattr(module, "RNG_DERIVATION",
                        {**module.RNG_DERIVATION, "current": "something else"})
    after, _ = role_partition_fingerprint(
        version=version, targets=targets, seed=20260810, spec=spec,
        proposals=proposals, source_manifest_sha256="source-a")
    assert before != after


#: The published dialogue-aware corpus manifest. It lives outside the repo, so
#: the pinning test below is skipped where the corpus artifact is unavailable.
V2_MANIFEST = ("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2/manifests/"
               "cs_dialogue_with_dialogue_id.parquet")

#: The allocation the project actually uses, under the gate it actually applies.
ACCEPTED_V2_ASSIGNMENT_HASH = \
    "313ae12472bab761a3d439880f93bbb89598e7ac7569915335b9c4e97a0fcbb5"


@pytest.mark.skipif(not __import__("pathlib").Path(V2_MANIFEST).is_file(),
                    reason="published dialogue manifest is not present")
def test_current_configuration_reproduces_the_accepted_v2_allocation():
    """Pin the accepted allocation: TV 0.15, SMD 0.25, seed 20260810, both proposals.

    This is the reproducibility guarantee with scientific value -- the actual
    allocation under the actual gate, not a rejected candidate under a threshold
    the project examined and declined. It runs the real 2x20000-draw search over
    the real corpus and takes roughly three minutes.

    The partition *fingerprint* is deliberately not pinned: it hashes the prose
    of the RNG record, so pinning it would freeze documentation text. The
    assignment hash pins the thing that matters, the allocation itself.
    """
    import json
    from pathlib import Path

    from csasr.utils.config import load_config

    cfg = load_config("lss/l1b_candidates_dialogue_v2.yaml")
    manifest = pd.read_parquet(V2_MANIFEST)
    sidecar = json.loads(Path(V2_MANIFEST + ".manifest.json").read_text(encoding="utf-8"))

    # the gate this allocation was accepted under, asserted rather than assumed
    balance = dict(cfg["roles"]["balance"])
    assert float(balance["max_categorical_tv"]) == 0.15
    assert float(balance["max_abs_smd"]) == 0.25
    assert cfg["seeds"]["role_assignment"] == 20260810
    mechanisms = [p["mechanism"] for p in
                  cfg["roles"]["dialogue_roles"]["partition"]["proposals"]]
    assert mechanisms == ["uniform_permutation", "stratified_deal"]

    _, report, diagnostics = build_dialogue_roles(
        cfg, manifest, source_manifest_sha256=sidecar["sha256"])

    assert report["assignment_hash"] == ACCEPTED_V2_ASSIGNMENT_HASH
    # and it reproduces as a *passing* allocation, not merely a repeatable one
    assert not len(diagnostics[~diagnostics["passed"]])
    assert float(report["balance"]["accepted_max_categorical_tv"]) <= 0.15
    assert float(report["balance"]["accepted_max_abs_smd"]) <= 0.25
    assert report["structure"]["every_dialogue_in_exactly_one_role"] is True


# --------------------------------------------------------------------------
# republication: correcting an identity record must never move a dialogue


def _published(tmp_path, name="gen-a"):
    """Publish a generation and return (destination, roles, report, diagnostics)."""
    from csasr.experiments import dialogue_roles_build as command

    roles, report, diagnostics = build_dialogue_roles(_cfg(), _manifest())
    destination = tmp_path / name
    command.publish_generation(
        out_dir=destination, roles=roles, report=report, diagnostics=diagnostics,
        overlap={"before": {}, "after": {}}, cfg=_cfg(),
        parent={"path": "source", "sha256": "abc"})
    return destination, roles, report, diagnostics


def test_republication_refuses_a_changed_assignment_hash(tmp_path):
    from csasr.experiments import dialogue_roles_build as command

    destination, roles, report, _ = _published(tmp_path)
    with pytest.raises(SystemExit, match="allocation changed: assignment hash"):
        command.assert_allocation_unchanged(
            roles, destination, report=report,
            expected_assignment_hash="0" * 64)


def test_republication_refuses_a_moved_dialogue_even_at_equal_size(tmp_path):
    """Equal role sizes are not equality: a swap must be caught."""
    from csasr.experiments import dialogue_roles_build as command

    destination, roles, report, _ = _published(tmp_path)
    swapped = {role: frame.copy() for role, frame in roles.items()}
    a = sorted(set(swapped["D-construct"]["dialogue_id"]))[0]
    b = sorted(set(swapped["loc-train"]["dialogue_id"]))[0]
    take = lambda frame, d: frame[frame["dialogue_id"] == d]
    drop = lambda frame, d: frame[frame["dialogue_id"] != d]
    swapped["D-construct"] = pd.concat(
        [drop(swapped["D-construct"], a), take(swapped["loc-train"], b)])
    swapped["loc-train"] = pd.concat(
        [drop(swapped["loc-train"], b), take(roles["D-construct"], a)])
    # sizes are preserved by construction; only membership moved
    for role in ("D-construct", "loc-train"):
        assert swapped[role]["dialogue_id"].nunique() == \
            roles[role]["dialogue_id"].nunique()
    with pytest.raises(SystemExit, match="per-role dialogue membership differs"):
        command.assert_allocation_unchanged(
            swapped, destination, report=report,
            expected_assignment_hash=report["assignment_hash"])


def test_republication_accepts_the_identical_allocation(tmp_path):
    from csasr.experiments import dialogue_roles_build as command

    destination, roles, report, _ = _published(tmp_path)
    equivalence = command.assert_allocation_unchanged(
        roles, destination, report=report,
        expected_assignment_hash=report["assignment_hash"])
    assert equivalence["identical_dialogue_by_dialogue"] is True
    assert equivalence["roles_compared"] == len(DIALOGUE_ROLES)
    assert equivalence["dialogues_compared"] == 20


def test_a_republished_generation_points_at_what_it_replaces(tmp_path):
    import json

    from csasr.experiments import dialogue_roles_build as command

    first, roles, report, diagnostics = _published(tmp_path, "gen-a")
    previous = first / command.GENERATION_MANIFEST_NAME
    record = command.supersession_record(previous, reason="corrected fingerprint")
    second = tmp_path / "gen-b"
    command.publish_generation(
        out_dir=second, roles=roles, report=report, diagnostics=diagnostics,
        overlap={"before": {}, "after": {}}, cfg=_cfg(),
        parent={"path": "source", "sha256": "abc"}, supersedes=record,
        equivalence={"identical_dialogue_by_dialogue": True})

    published = json.loads(
        (second / command.GENERATION_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert published["supersedes"]["path"] == str(previous)
    assert published["supersedes"]["role_assignment_hash"] == report["assignment_hash"]
    assert published["allocation_equivalence"]["identical_dialogue_by_dialogue"] is True
    # the superseded generation is a record, not a workspace
    assert previous.is_file()
    assert json.loads(previous.read_text(encoding="utf-8"))["complete"] is True


def test_supersession_record_refuses_a_missing_generation(tmp_path):
    from csasr.experiments import dialogue_roles_build as command

    with pytest.raises(SystemExit, match="superseded generation manifest not found"):
        command.supersession_record(tmp_path / "absent.json", reason="x")
