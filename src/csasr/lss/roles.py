"""The five data roles the proposal defines, built once and frozen.

    D-construct     build the fixed directions, nuisance models, prompt subspace
    loc-train       train the temporal localizer
    util-train      generate intervention-utility labels, train the selector
    router-calib    probability calibration and the abstention threshold
    D-dev-select    layer/dose selection, early oracle headroom
    D-dev-confirm   specificity controls, freeze the action, dry-run
    D-test          one locked evaluation batch

Two deviations from plan v1, both forced by the implementation review:

* `router-calib` holds 20 conversations, not 10, and is split into a
  probability-calibration half and a threshold-selection half. Fitting isotonic
  calibration, choosing an operating threshold, and then reporting the gate
  result on the same ten conversations would make the calibration error
  optimistic and the gate a tuned-set number.
* conversations are assigned by balanced rerandomization rather than by slicing
  a sorted list (see `csasr.lss.balance`).

`conversation_id` and `speaker_id` are 1:1 in this corpus (140/140 train,
20/20, 10/10, 30/30), so conversation-disjoint roles are also speaker-disjoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..data.manifest import assert_no_test_data
from ..utils.config import art, artifacts_root
from ..utils.hashing import manifest_hash
from ..utils.logging import get_logger
from .artifacts import artifact_ref, write_json, write_parquet
from .balance import BalanceSpec, assignment_hash, balanced_assignment, imbalance
from .eligibility import conversation_unit_counts, unit_table
from .seeds import seed_for

log = get_logger(__name__)

TRAIN_ROLES = ("D-construct", "loc-train", "util-train", "router-calib")
PASSTHROUGH_ROLES = ("D-dev-select", "D-dev-confirm", "D-test")
ROLES = TRAIN_ROLES + PASSTHROUGH_ROLES
TEST_ROLE = "D-test"

#: named unions used by later stages (e.g. the two LoRA data-matching arms)
DERIVED_VIEWS: dict[str, tuple[str, ...]] = {
    "router-all": ("loc-train", "util-train", "router-calib"),
    "full-method-data": ("D-construct", "loc-train", "util-train"),
    "lora-router-matched": ("loc-train", "util-train"),
}

ROLE_EXTRA_COLUMNS = ("role", "sub_role", "role_version", "role_assignment_hash")


class RoleError(RuntimeError):
    """A role manifest was requested or built in a way that is not permitted."""


def _source_root(cfg: Mapping[str, Any]) -> Path:
    root = (cfg.get("experiment") or {}).get("source_artifacts_root")
    if not root:
        raise RoleError("experiment.source_artifacts_root is not configured")
    return Path(root)


def _load_source(cfg: Mapping[str, Any], name: str) -> pd.DataFrame:
    path = _source_root(cfg) / "manifests" / f"{name}.parquet"
    if not path.is_file():
        raise RoleError(f"source manifest missing: {path}")
    return pd.read_parquet(path)


def _load_test(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(str((cfg.get("data") or {}).get("manifest", "")))
    if not path.is_file():
        raise RoleError(f"corpus manifest missing: {path}")
    full = pd.read_parquet(path)
    return full[full["official_split"] == "test"].reset_index(drop=True)


def role_targets(cfg: Mapping[str, Any]) -> dict[str, int]:
    targets = dict((cfg.get("roles") or {}).get("train_conversation_targets") or {})
    missing = [r for r in TRAIN_ROLES if r not in targets]
    if missing:
        raise RoleError(f"roles.train_conversation_targets is missing {missing}")
    return {r: int(targets[r]) for r in TRAIN_ROLES}


def build_roles(cfg: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Assign conversations to roles and return one manifest per role."""
    roles_cfg = cfg.get("roles") or {}
    version = str(roles_cfg.get("version", "v1"))
    spec = BalanceSpec.from_cfg(roles_cfg.get("balance") or {})

    train = _load_source(cfg, "train")
    assert_no_test_data(train)
    features = conversation_unit_counts(train)
    targets = role_targets(cfg)
    if sum(targets.values()) != len(features):
        raise RoleError(
            f"role targets sum to {sum(targets.values())} but the train split has "
            f"{len(features)} conversations")

    assignment, balance_report = balanced_assignment(
        features, targets, spec, seed=seed_for(cfg, "role_assignment"))
    diagnostics = imbalance(features, assignment, spec)

    # router-calib is split again so calibration and threshold selection never
    # share conversations with each other or with the gate evaluation
    sub_assignment: dict[str, str] = {}
    calib_split = dict(roles_cfg.get("router_calib_split") or {})
    if calib_split:
        calib_ids = sorted(c for c, r in assignment.items() if r == "router-calib")
        calib_features = features[features["conversation_id"].isin(calib_ids)]
        if sum(calib_split.values()) != len(calib_ids):
            raise RoleError(
                f"router_calib_split sums to {sum(calib_split.values())} but "
                f"router-calib holds {len(calib_ids)} conversations")
        sub_assignment, calib_balance = balanced_assignment(
            calib_features, calib_split, spec, seed=seed_for(cfg, "role_assignment") + 1)
        balance_report["router_calib_split"] = calib_balance

    a_hash = assignment_hash(assignment)
    roles: dict[str, pd.DataFrame] = {}
    for role in TRAIN_ROLES:
        members = [c for c, r in assignment.items() if r == role]
        sub = train[train["conversation_id"].isin(members)].copy()
        sub["role"] = role
        sub["sub_role"] = sub["conversation_id"].map(sub_assignment).fillna("")
        roles[role] = sub.reset_index(drop=True)

    passthrough = {
        "D-dev-select": lambda: _load_source(cfg, "dev_select"),
        "D-dev-confirm": lambda: _load_source(cfg, "dev_confirm"),
        TEST_ROLE: lambda: _load_test(cfg),
    }
    for role, loader in passthrough.items():
        sub = loader().copy()
        sub["role"] = role
        sub["sub_role"] = ""
        roles[role] = sub.reset_index(drop=True)

    for role, frame in roles.items():
        frame["role_version"] = version
        frame["role_assignment_hash"] = a_hash

    report = role_report(roles, features, balance_report, diagnostics,
                         version=version, assignment=assignment,
                         assignment_sha=a_hash)
    return roles, report


def assert_roles_disjoint(roles: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Conversations and utterances may belong to exactly one role."""
    conversations = {r: set(df["conversation_id"]) for r, df in roles.items()}
    utterances = {r: set(df["utterance_id"]) for r, df in roles.items()}
    overlaps: dict[str, list[str]] = {}
    names = sorted(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_conv = conversations[a] & conversations[b]
            shared_utt = utterances[a] & utterances[b]
            if shared_conv:
                overlaps[f"conversation:{a}|{b}"] = sorted(shared_conv)[:10]
            if shared_utt:
                overlaps[f"utterance:{a}|{b}"] = sorted(shared_utt)[:10]
    if overlaps:
        raise RoleError(f"roles overlap: {overlaps}")

    leaked = {r: int((df["official_split"] == "test").sum())
              for r, df in roles.items() if r != TEST_ROLE}
    if any(leaked.values()):
        raise RoleError(f"test-split rows reached non-test roles: {leaked}")
    return {
        "conversation_disjoint": True,
        "utterance_disjoint": True,
        "test_rows_in_non_test_roles": 0,
        "roles": {r: len(conversations[r]) for r in names},
    }


def role_report(roles: Mapping[str, pd.DataFrame], features: pd.DataFrame,
                balance: Mapping[str, Any], diagnostics: pd.DataFrame, *,
                version: str, assignment: Mapping[str, str],
                assignment_sha: str) -> dict[str, Any]:
    per_role: dict[str, Any] = {}
    for role, df in roles.items():
        units = unit_table(df)
        embedded = int(units["is_embedded_english"].sum()) if len(units) else 0
        per_role[role] = {
            "conversations": int(df["conversation_id"].nunique()),
            "speakers": int(df["speaker_id"].nunique()),
            "utterances": int(len(df)),
            "hours": round(float(df["duration_sec"].sum()) / 3600.0, 3),
            "cs_utterances": int(df["contains_code_switch"].sum()),
            "embedded_en_units": embedded,
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
        "assignment_hash": assignment_sha,
        "assignment": dict(sorted(assignment.items())),
        "balance": dict(balance),
        "balance_failures": failed.to_dict(orient="records") if len(failed) else [],
        "max_abs_smd": float(diagnostics["smd"].abs().max()) if len(diagnostics) else 0.0,
        # gated covariates only; high-cardinality ones are reported separately
        # because a role of 20 conversations cannot cover 28 regions
        "max_categorical_tv": _max(gated, "tv"),
        "max_categorical_tv_reported": _max(diagnostics, "tv"),
        "roles": per_role,
        "train_conversations": int(len(features)),
    }


def write_roles(cfg: Mapping[str, Any], roles: Mapping[str, pd.DataFrame],
                report: Mapping[str, Any], diagnostics: pd.DataFrame,
                *, stage: str = "l0_freeze", run_dir=None,
                taint_reasons: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Persist role manifests; the test role goes to a separate locked folder.

    Each manifest is published with an artifact manifest, so a downstream stage
    can tell a role table produced by this run apart from one left behind by a
    different configuration.
    """
    from .manifest import publish_frame

    refs: dict[str, dict[str, Any]] = {}
    for role, df in roles.items():
        if role == TEST_ROLE:
            path = art(cfg, "manifests", "roles", "locked", f"role_{role}.parquet")
        else:
            path = art(cfg, "manifests", "roles", f"role_{role}.parquet")
        publish_frame(path, df, stage=stage, cfg=cfg, run_dir=run_dir,
                      taint_reasons=taint_reasons,
                      key_columns=("utterance_id",),
                      schema="lss_role_manifest_v1")
        refs[role] = artifact_ref(path, schema="lss_role_manifest_v1", rows=len(df))
    write_json(dict(report), art(cfg, "manifests", "roles", "role_report.json"))
    write_json({"assignment": report["assignment"],
                "assignment_hash": report["assignment_hash"],
                "balance": report["balance"]},
               art(cfg, "manifests", "roles", "assignment.json"))
    write_parquet(diagnostics, art(cfg, "manifests", "roles",
                                   "balance_diagnostics.parquet"))
    return refs


def role_path(cfg: Mapping[str, Any], name: str) -> Path:
    root = artifacts_root(cfg) / "manifests" / "roles"
    if name == TEST_ROLE:
        return root / "locked" / f"role_{name}.parquet"
    return root / f"role_{name}.parquet"


def load_role(cfg: Mapping[str, Any], name: str, *,
              allow_test: bool = False) -> pd.DataFrame:
    """Load one role manifest. The test role needs two independent unlocks."""
    if name not in ROLES:
        raise RoleError(f"unknown role {name!r}; expected one of {list(ROLES)}")
    if name == TEST_ROLE:
        prohibited = bool((cfg.get("experiment") or {}).get("prohibit_test_split", True))
        if not allow_test or prohibited:
            raise RoleError(
                "D-test is locked. Loading it requires allow_test=True *and* "
                "experiment.prohibit_test_split=false in a config written for the "
                "single locked evaluation batch.")
    path = role_path(cfg, name)
    if not path.is_file():
        raise RoleError(f"role manifest missing: {path}. Run l0_freeze first.")
    df = pd.read_parquet(path)
    if name != TEST_ROLE:
        assert_no_test_data(df)
    return df


def load_view(cfg: Mapping[str, Any], name: str) -> pd.DataFrame:
    """Load a named union of roles (never includes the test role)."""
    members = DERIVED_VIEWS.get(name)
    if members is None:
        raise RoleError(f"unknown view {name!r}; expected one of {sorted(DERIVED_VIEWS)}")
    frames = [load_role(cfg, r) for r in members]
    out = pd.concat(frames, ignore_index=True)
    out["view"] = name
    return out


def assert_role(df: pd.DataFrame, expected: str | Sequence[str]) -> None:
    """Guard at the entry of any training function."""
    allowed = {expected} if isinstance(expected, str) else set(expected)
    if "role" not in df.columns:
        raise RoleError("frame carries no `role` column; load it with load_role()")
    present = set(df["role"].unique())
    if not present <= allowed:
        raise RoleError(f"frame contains roles {sorted(present)}, expected {sorted(allowed)}")


def role_of(cfg: Mapping[str, Any], utterance_id: str) -> str:
    for role in ROLES:
        path = role_path(cfg, role)
        if not path.is_file():
            continue
        ids = pd.read_parquet(path, columns=["utterance_id"])["utterance_id"]
        if (ids == utterance_id).any():
            return role
    raise RoleError(f"utterance {utterance_id!r} belongs to no role manifest")
