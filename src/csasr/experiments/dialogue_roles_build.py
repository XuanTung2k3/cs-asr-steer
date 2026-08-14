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
from pathlib import Path
from typing import Any

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.dialogue_roles import (DIALOGUE_ROLES, TEST_ROLE, assert_no_dialogue_straddles,
                                  build_dialogue_roles, role_overlap_matrix)
from ..lss.roles import RoleError
from ..utils.config import load_config
from ..utils.hashing import sha256_file

STAGE = "dialogue_roles"
REPORT_NAME = "dialogue_role_report.json"
ASSIGNMENT_NAME = "assignment.json"
DIAGNOSTICS_NAME = "balance_diagnostics.parquet"
OVERLAP_NAME = "role_overlap_before_after.json"

#: Never write here, whatever the caller passes.
PROTECTED = ("/artifacts_lss/status", "/artifacts_lss/freeze",
             "/artifacts_lss/synthetic", "/artifacts_lss/alignments",
             "/artifacts_lss/manifests")


def resolve_output(output: str | Path) -> Path:
    """Refuse a destination that is protected or already holds role manifests."""
    target = Path(output).resolve()
    for guard in PROTECTED:
        if guard in str(target):
            raise SystemExit(f"refusing to write inside a protected tree: {target}")
    existing = sorted(target.glob("role_*.parquet")) + \
        sorted(target.glob("locked/role_*.parquet"))
    if existing:
        raise SystemExit(
            f"refusing to overwrite {len(existing)} existing role manifest(s) in "
            f"{target}; role manifests are immutable")
    return target


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
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = Path(args.manifest)
    if not source.is_file():
        raise SystemExit(f"manifest not found: {source}")
    out_dir = resolve_output(args.output_dir)

    manifest = pd.read_parquet(source)
    roles, report, diagnostics = build_dialogue_roles(cfg, manifest)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "locked").mkdir(exist_ok=True)
    parent = manifest_mod.load(source) or {"path": str(source),
                                           "sha256": sha256_file(source)}
    refs: dict[str, Any] = {}
    for role in DIALOGUE_ROLES:
        frame = roles[role]
        path = (out_dir / "locked" / f"role_{role}.parquet") if role == TEST_ROLE \
            else (out_dir / f"role_{role}.parquet")
        published = manifest_mod.publish_frame(
            path, frame, stage=STAGE, cfg=cfg, parents=[parent],
            key_columns=("utterance_id",), schema="lss_role_manifest_v2_dialogue",
            extra={"allocation_unit": "dialogue",
                   "dialogues": int(frame["dialogue_id"].nunique()),
                   "role_assignment_hash": report["assignment_hash"]})
        refs[role] = {"path": str(path), "rows": int(len(frame)),
                      "sha256": published["sha256"]}

    for name, payload in ((REPORT_NAME, report),
                          (ASSIGNMENT_NAME, {"assignment": report["assignment"],
                                             "assignment_hash": report["assignment_hash"],
                                             "seed": report["seed"],
                                             "balance": report["balance"]}),
                          (OVERLAP_NAME, overlap)):
        (out_dir / name).write_text(json.dumps(payload, indent=2, default=str),
                                    encoding="utf-8")
        manifest_mod.publish(out_dir / name, stage=STAGE, cfg=cfg, parents=[parent])
    diagnostics.to_parquet(out_dir / DIAGNOSTICS_NAME, index=False)
    manifest_mod.publish(out_dir / DIAGNOSTICS_NAME, diagnostics, stage=STAGE,
                         cfg=cfg, parents=[parent])

    print(json.dumps({
        "output": str(out_dir),
        "dialogues_per_role": report["structure"]["dialogues_per_role"],
        "utterances_per_role": {r: refs[r]["rows"] for r in DIALOGUE_ROLES},
        "assignment_hash": report["assignment_hash"],
        "max_abs_smd": round(report["max_abs_smd"], 4),
        "max_gated_categorical_tv": round(report["max_categorical_tv"], 4),
        "balance_failures": len(report["balance_failures"]),
        "dialogues_shared_between_roles": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
