"""Allocate whole dialogues to data roles.

`csasr.lss.roles` partitions *conversations*, which in CS-Dialogue is one side of
a two-party session.  A conversation-disjoint partition therefore still lets the
two halves of one recorded dialogue land in different roles: under the frozen v1
assignment 62 of 85 dialogues are split that way, including 8 of the 15
development dialogues, whose two sides sit in `D-dev-select` and `D-dev-confirm`
respectively.  Selection and confirmation then share sessions, topics, and
interlocutors, and a conversation-block bootstrap counts two correlated halves as
two independent clusters.

This module allocates at the dialogue level instead.  `D-test` is exactly the
official test split, so the locked evaluation batch stays comparable with
published CS-Dialogue numbers; every other role is drawn from the remaining
dialogues by seeded rerandomization. The declared stratified proposal deals the
gender-by-device composition across roles; topic and other ungated covariates
remain terms in the unchanged objective and are reported separately.

The v2 partition differs from v1 in two ways, not one: the draw-acceptance rule
was corrected so gated criteria constrain feasibility rather than being reported
after the fact, *and* the proposal machinery was rebuilt, including the RNG
derivation (`csasr.lss.balance.RNG_DERIVATION`). v2 is therefore a fresh
allocation rather than v1's re-judged under a different threshold, and the two
are not comparable draw-for-draw. The derivation is recorded in the partition
fingerprint so a future stream change cannot inherit this partition's identity.

The four structural requirements are enforced here rather than reported: a
dialogue belongs to exactly one role, role dialogue sets are pairwise disjoint,
`D-test` holds official-test dialogues only, and the roles cover the corpus
exactly.  Each raises `RoleError`.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from ..utils.hashing import manifest_hash, sha256_obj
from ..utils.logging import get_logger
from .balance import (RNG_DERIVATION, BalanceSpec, ProposalSpec, assignment_hash,
                      balanced_assignment, imbalance)
from .eligibility import conversation_unit_counts, unit_table
from .roles import ROLE_EXTRA_COLUMNS, RoleError, TEST_ROLE
from .seeds import seed_for

log = get_logger(__name__)

#: The role a dialogue may hold.  `D-test` is not allocated: it *is* the
#: official test split, which this module never redraws.
ALLOCATED_ROLES = ("D-construct", "loc-train", "util-train", "router-calib",
                   "D-dev-select", "D-dev-confirm")
DIALOGUE_ROLES = ALLOCATED_ROLES + (TEST_ROLE,)

OFFICIAL_TEST = "test"


class RoleBalanceError(RoleError):
    """A purported accepted allocation still violates a gated criterion."""

    def __init__(self, report: Mapping[str, Any], diagnostics: pd.DataFrame):
        self.report = dict(report)
        self.diagnostics = diagnostics.copy()
        failed = diagnostics[~diagnostics["passed"]]
        super().__init__(
            f"dialogue-role balance gate failed for {len(failed)} role/covariate "
            "rows; no role artifacts may be published")


def dialogue_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-dialogue covariates: both sides of the session pooled.

    Continuous covariates are summed over the pair because the pair is the unit
    being allocated.  `gender` and `device` become the *composition* of the pair
    ("FM", "Android/iphone"), which is what stratification can act on once the
    two participants can no longer be separated.
    """
    for column in ("dialogue_id", "conversation_id", "official_split"):
        if column not in manifest.columns:
            raise RoleError(f"manifest lacks `{column}`; build it with "
                            "csasr.experiments.dialogue_manifest first")

    per_conversation = conversation_unit_counts(manifest)
    lookup = manifest.drop_duplicates("conversation_id").set_index("conversation_id")
    per_conversation["dialogue_id"] = per_conversation["conversation_id"].map(
        lookup["dialogue_id"])
    per_conversation["official_split"] = per_conversation["conversation_id"].map(
        lookup["official_split"])
    if per_conversation["dialogue_id"].isna().any():
        raise RoleError("some conversations carry no dialogue_id")

    grouped = per_conversation.groupby("dialogue_id")
    features = grouped.agg(
        conversations=("conversation_id", "nunique"),
        utterances=("utterances", "sum"),
        hours=("hours", "sum"),
        cs_utterances=("cs_utterances", "sum"),
        embedded_en_units=("embedded_en_units", "sum"),
        en_units=("en_units", "sum"),
        zh_units=("zh_units", "sum"),
    ).reset_index()
    features["cs_rate"] = features["cs_utterances"] / features["utterances"].clip(lower=1)
    features["mean_duration_sec"] = (features["hours"] * 3600.0
                                     / features["utterances"].clip(lower=1))

    def composition(column: str) -> pd.Series:
        return grouped[column].agg(lambda s: "/".join(sorted(str(v) for v in s)))

    if "gender" in per_conversation.columns:
        features["gender"] = features["dialogue_id"].map(
            grouped["gender"].agg(lambda s: "".join(sorted(str(v) for v in s))))
    for column in ("device",):
        if column in per_conversation.columns:
            features[column] = features["dialogue_id"].map(composition(column))
    if "region" in per_conversation.columns:
        features["region"] = features["dialogue_id"].map(
            grouped["region"].agg(lambda s: sorted({str(v) for v in s})))
    if "topics" in per_conversation.columns:
        features["topics"] = features["dialogue_id"].map(
            grouped["topics"].agg(lambda s: sorted({t for entry in s for t in entry})))

    splits = grouped["official_split"].agg(lambda s: sorted(set(s)))
    straddling = {d: v for d, v in splits.items() if len(v) > 1}
    if straddling:
        raise RoleError(
            f"{len(straddling)} dialogue(s) straddle official splits: "
            f"{dict(list(straddling.items())[:3])}. The official split is "
            "pair-atomic in this corpus; a straddle means the pairing is wrong.")
    features["official_split"] = features["dialogue_id"].map(
        {d: v[0] for d, v in splits.items()})
    return features.sort_values("dialogue_id").reset_index(drop=True)


def allocation_targets(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Dialogues per allocated role, from config.  Never hard-coded here."""
    block = ((cfg.get("roles") or {}).get("dialogue_roles") or {})
    targets = dict(block.get("targets") or {})
    if not targets:
        raise RoleError("roles.dialogue_roles.targets is not configured")
    unknown = sorted(set(targets) - set(ALLOCATED_ROLES))
    if unknown:
        raise RoleError(f"unknown role(s) in dialogue_roles.targets: {unknown}")
    missing = [r for r in ALLOCATED_ROLES if r not in targets]
    if missing:
        raise RoleError(f"roles.dialogue_roles.targets is missing {missing}")
    return {r: int(targets[r]) for r in ALLOCATED_ROLES}


def partition_parameters(cfg: Mapping[str, Any]) -> tuple[str, BalanceSpec,
                                                          tuple[ProposalSpec, ...]]:
    """Load the explicit, versioned dialogue-partition parameters.

    The ungated weight is load-bearing and was previously a corpus-tuned module
    constant.  A dialogue allocation must now predeclare it alongside the
    proposal mechanisms, so an old config cannot silently reproduce the old
    selection rule.
    """
    roles_cfg = dict(cfg.get("roles") or {})
    dialogue = dict(roles_cfg.get("dialogue_roles") or {})
    partition = dict(dialogue.get("partition") or {})
    version = str(partition.get("version", "")).strip()
    if not version:
        raise RoleError("roles.dialogue_roles.partition.version is required")
    if "ungated_weight" not in partition:
        raise RoleError(
            "roles.dialogue_roles.partition.ungated_weight is required; the "
            "load-bearing corpus-tuned value may not be implicit")
    proposal_cfg = partition.get("proposals")
    if not isinstance(proposal_cfg, list) or not proposal_cfg:
        raise RoleError("roles.dialogue_roles.partition.proposals must be a nonempty list")

    balance_cfg = dict(roles_cfg.get("balance") or {})
    balance_cfg["ungated_weight"] = float(partition["ungated_weight"])
    spec = BalanceSpec.from_cfg(balance_cfg)
    try:
        proposals = tuple(ProposalSpec.from_cfg(p, default_draws=spec.draws)
                          for p in proposal_cfg)
    except (TypeError, ValueError) as exc:
        raise RoleError(f"invalid dialogue partition proposal: {exc}") from exc
    return version, spec, proposals


def role_partition_fingerprint(*, version: str, targets: Mapping[str, int],
                               seed: int, spec: BalanceSpec,
                               proposals: Sequence[ProposalSpec],
                               source_manifest_sha256: str) -> tuple[str, dict[str, Any]]:
    """Focused identity for the exact allocation rule and exact source bytes."""
    payload = {
        # v2 adds `rng_derivation`: the draw stream determines which candidates
        # exist, so leaving it out let two allocations with different streams
        # share one identity. Adding it necessarily changes the fingerprint value
        # for an otherwise identical rule -- that is the point.
        "fingerprint_version": "dialogue-role-partition-fingerprint-v2",
        "partition_version": str(version),
        "allocation_unit": "dialogue",
        "targets": {str(k): int(v) for k, v in targets.items()},
        "seed": int(seed),
        "balance": {
            "continuous": list(spec.continuous),
            "categorical": list(spec.categorical),
            "distributional": list(spec.distributional),
            "gated_categorical": list(spec.gated_categorical),
            "max_abs_smd": float(spec.max_abs_smd),
            "max_categorical_tv": float(spec.max_categorical_tv),
            "ungated_weight": float(spec.ungated_weight),
            "scoring_formula": "max(max_SMD, max_gated_TV) + ungated_weight*ungated_TV",
        },
        "proposals": [{"mechanism": p.mechanism, "draws": int(p.draws),
                       "stratify_by": list(p.stratify_by)} for p in proposals],
        "rng_derivation": dict(RNG_DERIVATION),
        "source_manifest_sha256": str(source_manifest_sha256),
    }
    return sha256_obj(payload), payload


def allocate_dialogues(features: pd.DataFrame, targets: Mapping[str, int],
                       spec: BalanceSpec, *, seed: int,
                       proposals: Sequence[ProposalSpec] | None = None
                       ) -> tuple[dict[str, str], dict[str, Any], pd.DataFrame]:
    """Assign non-test dialogues to roles by seeded rerandomization.

    `balance.balanced_assignment` keys on a column named `conversation_id`; the
    frame handed to it holds one row per *dialogue*, so the column is renamed at
    that boundary and mapped straight back.  Reusing the tested rerandomization
    is worth the rename: it is the same estimator, applied one level up.
    """
    pool = features[features["official_split"] != OFFICIAL_TEST].copy()
    if sum(targets.values()) != len(pool):
        raise RoleError(
            f"dialogue targets sum to {sum(targets.values())} but the non-test "
            f"pool holds {len(pool)} dialogues")

    as_balance = pool.rename(columns={"dialogue_id": "conversation_id"})
    assignment, report = balanced_assignment(
        as_balance, targets, spec, seed=seed, proposals=proposals)
    diagnostics = imbalance(as_balance, assignment, spec)
    failures = diagnostics[~diagnostics["passed"]]
    if len(failures):
        # This is an independent postcondition over the returned assignment. It
        # prevents a future proposal or scoring refactor from bypassing the draw
        # acceptance constraint and publishing an over-threshold artifact.
        raise RoleBalanceError(report, diagnostics)
    report["unit"] = "dialogue"
    report["pool"] = "official train + dev, test excluded"
    return assignment, report, diagnostics


def assert_dialogue_partition(assignment: Mapping[str, str],
                              features: pd.DataFrame) -> dict[str, Any]:
    """The four structural requirements.  Any violation raises.

    These are the properties the whole re-partition exists to obtain, so they
    are checked against the dialogue universe rather than against the frames
    that were just built from `assignment` -- a check derived from the same
    dictionary it is checking would pass by construction.
    """
    universe = set(features["dialogue_id"])
    test_dialogues = set(features.loc[features["official_split"] == OFFICIAL_TEST,
                                      "dialogue_id"])
    full = dict(assignment)
    for dialogue in test_dialogues:
        if full.get(dialogue) not in (None, TEST_ROLE):
            raise RoleError(
                f"official-test dialogue {dialogue} was allocated to "
                f"{full[dialogue]!r}; D-test is the official test split and is "
                "never redrawn")
        full[dialogue] = TEST_ROLE

    # 1. every dialogue in exactly one role
    members: dict[str, set[str]] = {}
    for dialogue, role in full.items():
        if role not in DIALOGUE_ROLES:
            raise RoleError(f"dialogue {dialogue} carries unknown role {role!r}")
        members.setdefault(role, set()).add(dialogue)
    counts = pd.Series(list(full.values())).value_counts().to_dict()

    # 2. pairwise disjoint role sets
    names = sorted(members)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = members[a] & members[b]
            if shared:
                raise RoleError(
                    f"roles {a} and {b} share {len(shared)} dialogue(s): "
                    f"{sorted(shared)[:5]}")

    # 3. D-test holds official-test dialogues only
    allocated_test = members.get(TEST_ROLE, set())
    intruders = allocated_test - test_dialogues
    if intruders:
        raise RoleError(
            f"{len(intruders)} non-test dialogue(s) reached D-test: "
            f"{sorted(intruders)[:5]}")
    missing_test = test_dialogues - allocated_test
    if missing_test:
        raise RoleError(
            f"{len(missing_test)} official-test dialogue(s) are absent from "
            f"D-test: {sorted(missing_test)[:5]}")

    # 4. the union is exactly the corpus
    covered = set(full)
    if covered != universe:
        unassigned = sorted(universe - covered)
        foreign = sorted(covered - universe)
        raise RoleError(
            f"role union does not equal the dialogue universe; "
            f"{len(unassigned)} unassigned {unassigned[:5]}, "
            f"{len(foreign)} unknown {foreign[:5]}")

    return {
        "dialogues": len(universe),
        "every_dialogue_in_exactly_one_role": True,
        "role_sets_pairwise_disjoint": True,
        "d_test_is_official_test_only": True,
        "union_equals_corpus": True,
        "dialogues_per_role": {r: int(counts.get(r, 0)) for r in DIALOGUE_ROLES},
    }


def build_dialogue_roles(cfg: Mapping[str, Any], manifest: pd.DataFrame, *,
                         source_manifest_sha256: str | None = None
                         ) -> tuple[dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame]:
    """One manifest per role, allocated whole dialogues at a time."""
    roles_cfg = cfg.get("roles") or {}
    block = dict(roles_cfg.get("dialogue_roles") or {})
    version = str(block.get("version", "dialogue-v1"))
    partition_version, spec, proposals = partition_parameters(cfg)

    features = dialogue_features(manifest)
    targets = allocation_targets(cfg)
    # `role_assignment` is the registered purpose for "which unit lands in which
    # data role"; this is that draw at dialogue granularity, so it needs no new
    # seed purpose and stays inside the frozen seed map
    seed = seed_for(cfg, "role_assignment")
    assignment, balance_report, diagnostics = allocate_dialogues(
        features, targets, spec, seed=seed, proposals=proposals)

    structure = assert_dialogue_partition(assignment, features)

    full = dict(assignment)
    for dialogue in features.loc[features["official_split"] == OFFICIAL_TEST,
                                 "dialogue_id"]:
        full[dialogue] = TEST_ROLE
    a_hash = assignment_hash(full)
    source_sha = str(source_manifest_sha256 or manifest_hash(
        manifest, columns=("utterance_id", "dialogue_id")))
    partition_sha, partition_payload = role_partition_fingerprint(
        version=partition_version, targets=targets, seed=seed, spec=spec,
        proposals=proposals, source_manifest_sha256=source_sha)

    sub_split = dict(block.get("router_calib_split") or {})
    sub_assignment: dict[str, str] = {}
    if sub_split:
        calib = sorted(d for d, r in full.items() if r == "router-calib")
        if sum(sub_split.values()) != len(calib):
            raise RoleError(
                f"dialogue_roles.router_calib_split sums to {sum(sub_split.values())} "
                f"but router-calib holds {len(calib)} dialogues")
        calib_features = features[features["dialogue_id"].isin(calib)].rename(
            columns={"dialogue_id": "conversation_id"})
        sub_assignment, sub_report = balanced_assignment(
            calib_features, sub_split, spec, seed=seed + 1)
        balance_report["router_calib_split"] = sub_report

    roles: dict[str, pd.DataFrame] = {}
    for role in DIALOGUE_ROLES:
        wanted = [d for d, r in full.items() if r == role]
        frame = manifest[manifest["dialogue_id"].isin(wanted)].copy()
        frame["role"] = role
        frame["sub_role"] = frame["dialogue_id"].map(sub_assignment).fillna("")
        frame["role_version"] = version
        frame["role_assignment_hash"] = a_hash
        frame["role_partition_fingerprint"] = partition_sha
        roles[role] = frame.reset_index(drop=True)

    report = dialogue_role_report(roles, features, full, balance_report,
                                  diagnostics, structure,
                                  version=version, assignment_sha=a_hash, seed=seed)
    report["partition_version"] = partition_version
    report["partition_fingerprint"] = partition_sha
    report["partition_fingerprint_payload"] = partition_payload
    report["source_manifest_sha256"] = source_sha
    return roles, report, diagnostics


def assert_no_dialogue_straddles(roles: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """No dialogue -- and so no conversation or utterance -- spans two roles."""
    dialogues = {r: set(df["dialogue_id"]) for r, df in roles.items()}
    conversations = {r: set(df["conversation_id"]) for r, df in roles.items()}
    overlaps: dict[str, list[str]] = {}
    names = sorted(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for label, sets in (("dialogue", dialogues), ("conversation", conversations)):
                shared = sets[a] & sets[b]
                if shared:
                    overlaps[f"{label}:{a}|{b}"] = sorted(shared)[:10]
    if overlaps:
        raise RoleError(f"roles overlap after allocation: {overlaps}")

    leaked = {r: int((df["official_split"] == OFFICIAL_TEST).sum())
              for r, df in roles.items() if r != TEST_ROLE}
    if any(leaked.values()):
        raise RoleError(f"official-test rows reached non-test roles: {leaked}")
    return {"dialogue_disjoint": True, "conversation_disjoint": True,
            "test_rows_in_non_test_roles": 0}


def role_overlap_matrix(assignment_by_conversation: Mapping[str, str],
                        dialogue_of: Mapping[str, str]) -> pd.DataFrame:
    """Dialogues shared by each ordered role pair, for a before/after table."""
    per_role: dict[str, set[str]] = {}
    for conversation, role in assignment_by_conversation.items():
        dialogue = dialogue_of.get(conversation)
        if dialogue is not None:
            per_role.setdefault(role, set()).add(dialogue)
    names = sorted(per_role)
    rows = [{"role": a, **{b: len(per_role[a] & per_role[b]) if a != b else 0
                           for b in names}} for a in names]
    return pd.DataFrame(rows).set_index("role")


def dialogue_role_report(roles: Mapping[str, pd.DataFrame], features: pd.DataFrame,
                         assignment: Mapping[str, str], balance: Mapping[str, Any],
                         diagnostics: pd.DataFrame, structure: Mapping[str, Any], *,
                         version: str, assignment_sha: str, seed: int) -> dict[str, Any]:
    per_role: dict[str, Any] = {}
    for role, df in roles.items():
        units = unit_table(df)
        embedded = int(units["is_embedded_english"].sum()) if len(units) else 0
        indexed = features.set_index("dialogue_id")
        mine = indexed.loc[sorted(set(df["dialogue_id"]))]
        topics: dict[str, int] = {}
        for entry in mine.get("topics", pd.Series(dtype=object)):
            for topic in entry or []:
                topics[topic] = topics.get(topic, 0) + 1
        per_role[role] = {
            "dialogues": int(df["dialogue_id"].nunique()),
            "conversations": int(df["conversation_id"].nunique()),
            "utterances": int(len(df)),
            "hours": round(float(df["duration_sec"].sum()) / 3600.0, 3),
            "cs_utterances": int(df["contains_code_switch"].sum()),
            "embedded_en_units": embedded,
            "gender_composition": {str(k): int(v) for k, v in
                                   mine["gender"].value_counts().items()},
            "device_composition": {str(k): int(v) for k, v in
                                   mine["device"].value_counts().items()},
            "topics": dict(sorted(topics.items())),
            "manifest_hash": manifest_hash(df),
            "sub_roles": {k: int(v) for k, v in
                          df["sub_role"].value_counts().to_dict().items() if k},
        }
    failed = diagnostics[~diagnostics["passed"]] if len(diagnostics) else diagnostics
    gated = diagnostics[diagnostics.get("gated", True)] if len(diagnostics) else diagnostics

    def _max(frame: pd.DataFrame, column: str) -> float:
        if not len(frame) or column not in frame or not frame[column].notna().any():
            return 0.0
        return float(frame[column].max(skipna=True))

    return {
        "role_version": version,
        "allocation_unit": "dialogue",
        "seed": int(seed),
        "seed_purpose": "role_assignment",
        "assignment_hash": assignment_sha,
        "assignment": dict(sorted(assignment.items())),
        "structure": dict(structure),
        "balance": dict(balance),
        "balance_failures": failed.to_dict(orient="records") if len(failed) else [],
        "max_abs_smd": float(diagnostics["smd"].abs().max()) if len(diagnostics) else 0.0,
        "max_categorical_tv": _max(gated, "tv"),
        "max_categorical_tv_reported": _max(diagnostics, "tv"),
        "roles": per_role,
        "dialogues": int(len(features)),
    }
