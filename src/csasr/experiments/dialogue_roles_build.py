"""Build dialogue-atomic LSS role manifests at a new path.

Reads the dialogue-aware corpus manifest, allocates whole two-party sessions to
roles, and publishes one manifest per role with its balance diagnostics and a
before/after overlap table against the frozen conversation-level assignment.

Existing role manifests are never read for output, never modified, and never
overwritten: the stage refuses any output directory that already holds a role
table, and refuses to write inside the production roles tree at all.  It writes
no status, allocates no gate generation, and freezes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.balance import NoFeasibleAssignment
from ..lss.dialogue_roles import (DIALOGUE_ROLES, TEST_ROLE, assert_no_dialogue_straddles,
                                  RoleBalanceError, build_dialogue_roles,
                                  role_overlap_matrix)
from ..nat5h.statusing import atomic_write_json
from ..lss.roles import RoleError
from ..utils.config import load_config
from ..utils.hashing import sha256_file

STAGE = "dialogue_roles"
REPORT_NAME = "dialogue_role_report.json"
ASSIGNMENT_NAME = "assignment.json"
DIAGNOSTICS_NAME = "balance_diagnostics.parquet"
OVERLAP_NAME = "role_overlap_before_after.json"
GENERATION_MANIFEST_NAME = "generation_complete.manifest.json"

#: Never write here, whatever the caller passes.
PROTECTED = ("/artifacts_lss/status", "/artifacts_lss/freeze",
             "/artifacts_lss/synthetic", "/artifacts_lss/alignments",
             "/artifacts_lss/manifests")


def resolve_output(output: str | Path) -> Path:
    """Refuse protected, symlinked, or already occupied destinations."""
    requested = Path(output)
    if requested.is_symlink():
        raise SystemExit(f"refusing a symlink destination: {requested}")
    target = requested.resolve()
    for guard in PROTECTED:
        if guard in str(target):
            raise SystemExit(f"refusing to write inside a protected tree: {target}")
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise SystemExit(
            f"refusing to overwrite an existing destination artifact or sidecar: "
            f"{target}; role generations are immutable")
    return target


def _logical_path(attempt: Path, final: Path, artifact: Path) -> Path:
    return final / artifact.relative_to(attempt)


def role_dialogue_sets(roles_dir: Path) -> dict[str, set[str]]:
    """`role -> set(dialogue_id)` as published under `roles_dir`."""
    out: dict[str, set[str]] = {}
    for path in sorted(roles_dir.glob("role_*.parquet")) + \
            sorted(roles_dir.glob("locked/role_*.parquet")):
        frame = pd.read_parquet(path, columns=["role", "dialogue_id"])
        for role, dialogue in zip(frame["role"], frame["dialogue_id"]):
            out.setdefault(str(role), set()).add(str(dialogue))
    return out


def assert_allocation_unchanged(roles: dict[str, pd.DataFrame], previous: Path, *,
                                expected_assignment_hash: str,
                                report: dict[str, Any]) -> dict[str, Any]:
    """Refuse to republish unless the allocation is bit-identical to `previous`.

    A republication exists to correct an identity record, never to move a
    dialogue. Equal role *sizes* would hide a swap, so the comparison is
    dialogue-by-dialogue within each role. A mismatch means something other than
    the fingerprint payload moved, and nothing may be promoted until a human has
    seen what.
    """
    if report["assignment_hash"] != expected_assignment_hash:
        raise SystemExit(
            "allocation changed: assignment hash is "
            f"{report['assignment_hash']} but {expected_assignment_hash} was "
            "required; nothing was published")

    previous_sets = role_dialogue_sets(previous)
    if not previous_sets:
        raise SystemExit(f"no previous role manifests to compare under: {previous}")
    current_sets = {role: set(frame["dialogue_id"].astype(str))
                    for role, frame in roles.items()}
    if set(previous_sets) != set(current_sets):
        raise SystemExit(
            f"role sets differ: {sorted(previous_sets)} vs {sorted(current_sets)}")
    differences = {
        role: {"added": sorted(current_sets[role] - previous_sets[role]),
               "removed": sorted(previous_sets[role] - current_sets[role])}
        for role in sorted(previous_sets)
        if current_sets[role] != previous_sets[role]
    }
    if differences:
        raise SystemExit(
            "allocation changed: per-role dialogue membership differs from "
            f"{previous}: {json.dumps(differences)[:800]}; nothing was published")

    return {
        "compared_with": str(previous),
        "assignment_hash": expected_assignment_hash,
        "roles_compared": len(previous_sets),
        "dialogues_compared": sum(len(v) for v in previous_sets.values()),
        "identical_dialogue_by_dialogue": True,
    }


def supersession_record(previous_generation: Path, *, reason: str) -> dict[str, Any]:
    """Point at the generation this one replaces, by path and by hash.

    The superseded generation is never opened for writing: it stays on disk as
    the record of the pre-correction state, and this pointer is what makes the
    chain between them navigable.
    """
    if not previous_generation.is_file():
        raise SystemExit(
            f"superseded generation manifest not found: {previous_generation}")
    payload = json.loads(previous_generation.read_text(encoding="utf-8"))
    return {
        "path": str(previous_generation),
        "generation_root": str(previous_generation.parent),
        "file_sha256": sha256_file(previous_generation),
        "role_assignment_hash": payload.get("role_assignment_hash"),
        "role_partition_fingerprint": payload.get("role_partition_fingerprint"),
        "reason": reason,
        "superseded_at": _now(),
    }


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def publish_generation(*, out_dir: Path, roles: dict[str, pd.DataFrame],
                       report: dict[str, Any], diagnostics: pd.DataFrame,
                       overlap: dict[str, Any], cfg: dict[str, Any],
                       parent: dict[str, Any],
                       supersedes: dict[str, Any] | None = None,
                       equivalence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Publish a complete immutable role generation with one atomic rename.

    All artifacts and sidecars are written below an attempt-private sibling.
    The generation-complete manifest is the final write. Only then is the whole
    directory renamed into the requested final path. A killed writer can leave
    a private attempt, but never a partial production-looking generation.
    """
    if out_dir.exists() or manifest_mod.manifest_path(out_dir).exists():
        raise SystemExit(f"refusing occupied immutable destination: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(
        prefix=f".{out_dir.name}.attempt-", dir=str(out_dir.parent)))
    lock = out_dir.parent / f".{out_dir.name}.publish.lock"
    promoted = False
    try:
        (attempt / "locked").mkdir()
        refs: dict[str, Any] = {}
        for role in DIALOGUE_ROLES:
            frame = roles[role]
            artifact = (attempt / "locked" / f"role_{role}.parquet") \
                if role == TEST_ROLE else (attempt / f"role_{role}.parquet")
            final_artifact = _logical_path(attempt, out_dir, artifact)
            published = manifest_mod.publish_frame(
                artifact, frame, stage=STAGE, cfg=cfg, parents=[parent],
                key_columns=("utterance_id",),
                schema="lss_role_manifest_v2_dialogue",
                logical_path=final_artifact,
                extra={"allocation_unit": "dialogue",
                       "dialogues": int(frame["dialogue_id"].nunique()),
                       "role_assignment_hash": report["assignment_hash"],
                       "role_partition_fingerprint": report["partition_fingerprint"]})
            refs[role] = {"path": str(final_artifact), "rows": int(len(frame)),
                          "sha256": published["sha256"]}

        payloads = (
            (REPORT_NAME, report),
            (ASSIGNMENT_NAME, {"assignment": report["assignment"],
                               "assignment_hash": report["assignment_hash"],
                               "seed": report["seed"],
                               "balance": report["balance"],
                               "partition_fingerprint": report["partition_fingerprint"]}),
            (OVERLAP_NAME, overlap),
        )
        for name, payload in payloads:
            artifact = attempt / name
            atomic_write_json(artifact, payload)
            manifest_mod.publish(
                artifact, stage=STAGE, cfg=cfg, parents=[parent],
                logical_path=_logical_path(attempt, out_dir, artifact),
                extra={"role_partition_fingerprint": report["partition_fingerprint"]})

        diagnostics_path = attempt / DIAGNOSTICS_NAME
        manifest_mod.publish_frame(
            diagnostics_path, diagnostics, stage=STAGE, cfg=cfg, parents=[parent],
            schema="lss_dialogue_role_balance_diagnostics_v2",
            logical_path=_logical_path(attempt, out_dir, diagnostics_path),
            extra={"role_partition_fingerprint": report["partition_fingerprint"]})

        # This is intentionally the final write inside the attempt. Its
        # presence means every listed artifact and sidecar was durable before
        # the directory became visible at its final name.
        inventory = []
        for artifact in sorted(p for p in attempt.rglob("*") if p.is_file()):
            inventory.append({
                "path": str(_logical_path(attempt, out_dir, artifact)),
                "relative_path": str(artifact.relative_to(attempt)),
                "sha256": sha256_file(artifact),
                "bytes": int(artifact.stat().st_size),
            })
        generation = {
            "schema": "lss_dialogue_role_generation_v1",
            "complete": True,
            "state": "completed",
            "published": True,
            "role_partition_fingerprint": report["partition_fingerprint"],
            "role_assignment_hash": report["assignment_hash"],
            "source_parent": {"path": parent.get("path"),
                              "sha256": parent.get("sha256")},
            "supersedes": supersedes,
            "allocation_equivalence": equivalence,
            "artifacts": inventory,
        }
        atomic_write_json(attempt / GENERATION_MANIFEST_NAME, generation)

        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(lock_fd)
        try:
            if out_dir.exists() or manifest_mod.manifest_path(out_dir).exists():
                raise SystemExit(f"destination appeared during publication: {out_dir}")
            os.rename(attempt, out_dir)
            promoted = True
        finally:
            lock.unlink(missing_ok=True)
        return {"refs": refs, "generation": generation}
    finally:
        if not promoted and attempt.exists():
            shutil.rmtree(attempt)


def previous_overlap(roles_dir: Path, dialogue_of: dict[str, str]) -> dict[str, Any]:
    """Dialogue overlaps in the frozen conversation-level assignment.

    Read-only, and absent-tolerant: the point of the table is the comparison, so
    a missing baseline is reported rather than raised.
    """
    files = sorted(roles_dir.glob("role_*.parquet")) + \
        sorted(roles_dir.glob("locked/role_*.parquet"))
    if not files:
        return {"available": False, "reason": f"no role manifests under {roles_dir}"}
    assignment: dict[str, str] = {}
    for path in files:
        frame = pd.read_parquet(path, columns=["conversation_id", "role"])
        for conversation, role in zip(frame["conversation_id"], frame["role"]):
            assignment[str(conversation)] = str(role)
    matrix = role_overlap_matrix(assignment, dialogue_of)
    return {"available": True, "source": str(roles_dir),
            "matrix": json.loads(matrix.to_json(orient="index")),
            "total_shared_pairs": int((matrix.to_numpy() > 0).sum() // 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l0_freeze.yaml")
    parser.add_argument("--manifest", required=True,
                        help="dialogue-aware corpus manifest (has dialogue_id)")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for the dialogue-atomic role manifests")
    parser.add_argument("--compare-with", default=None,
                        help="frozen roles directory for the before/after table")
    parser.add_argument("--supersedes", default=None,
                        help="generation_complete.manifest.json of the generation "
                             "this one replaces; its role directory must hold the "
                             "identical allocation")
    parser.add_argument("--expect-assignment-hash", default=None,
                        help="refuse to publish unless the allocation hashes to "
                             "exactly this value")
    parser.add_argument("--supersession-reason", default="",
                        help="why the previous generation is being replaced")
    args = parser.parse_args(argv)
    if args.supersedes and not args.expect_assignment_hash:
        raise SystemExit(
            "--supersedes requires --expect-assignment-hash: a republication "
            "must state the allocation it is required to reproduce")

    cfg = load_config(args.config)
    source = Path(args.manifest)
    if not source.is_file():
        raise SystemExit(f"manifest not found: {source}")
    out_dir = resolve_output(args.output_dir)

    source_sha = sha256_file(source)
    parent = manifest_mod.load(source) or {"path": str(source), "sha256": source_sha}
    if parent.get("sha256") != source_sha:
        raise SystemExit(
            f"source manifest sidecar does not authenticate current bytes: {source}")

    manifest = pd.read_parquet(source)
    try:
        roles, report, diagnostics = build_dialogue_roles(
            cfg, manifest, source_manifest_sha256=source_sha)
    except (NoFeasibleAssignment, RoleBalanceError) as exc:
        no_go = {
            "state": "completed_no_go",
            "published": False,
            "output": str(out_dir),
            "reason": str(exc),
            "balance": exc.report,
        }
        print(json.dumps(no_go, indent=2, default=str))
        return 0
    report["structure"].update(assert_no_dialogue_straddles(roles))

    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))
    after = role_overlap_matrix(
        {str(c): str(r) for role, df in roles.items()
         for c, r in zip(df["conversation_id"], df["role"])}, dialogue_of)
    if int(after.to_numpy().sum()):
        raise RoleError(f"dialogues still shared between roles: {after.to_dict()}")
    overlap = {
        "unit": "dialogues shared between a role pair",
        "before": previous_overlap(Path(args.compare_with), dialogue_of)
        if args.compare_with else {"available": False, "reason": "not requested"},
        "after": {"matrix": json.loads(after.to_json(orient="index")),
                  "total_shared_pairs": 0},
    }

    # Every gate below is checked before anything is promoted: a republication
    # that moved a dialogue must leave no artifact behind at all.
    equivalence = supersedes = None
    if args.expect_assignment_hash:
        previous_root = Path(args.supersedes).parent if args.supersedes \
            else Path(args.compare_with or out_dir)
        equivalence = assert_allocation_unchanged(
            roles, previous_root, report=report,
            expected_assignment_hash=args.expect_assignment_hash)
    if args.supersedes:
        supersedes = supersession_record(
            Path(args.supersedes), reason=args.supersession_reason)

    published = publish_generation(
        out_dir=out_dir, roles=roles, report=report, diagnostics=diagnostics,
        overlap=overlap, cfg=cfg, parent=parent, supersedes=supersedes,
        equivalence=equivalence)
    refs = published["refs"]

    print(json.dumps({
        "output": str(out_dir),
        "dialogues_per_role": report["structure"]["dialogues_per_role"],
        "utterances_per_role": {r: refs[r]["rows"] for r in DIALOGUE_ROLES},
        "assignment_hash": report["assignment_hash"],
        "max_abs_smd": round(report["max_abs_smd"], 4),
        "max_gated_categorical_tv": round(report["max_categorical_tv"], 4),
        "balance_failures": len(report["balance_failures"]),
        "state": "completed",
        "published": True,
        "accepted_proposal_mechanism":
            report["balance"]["accepted_proposal_mechanism"],
        "feasibility": report["balance"]["proposal_reports"],
        "partition_fingerprint": report["partition_fingerprint"],
        "dialogues_shared_between_roles": 0,
        "allocation_equivalence": equivalence,
        "supersedes": supersedes,
    }, indent=2))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
