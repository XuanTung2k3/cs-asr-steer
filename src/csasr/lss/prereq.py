"""The stage dependency graph, declared once and enforced mechanically.

Plan v1 carried a cycle: the baseline stage could not pass until the oracle
stage supplied an effect size, while the oracle stage consumed baseline
outputs. Declaring the graph in one place and asserting acyclicity at import
means that class of mistake fails at test time instead of at week six.

Two properties this adds over `csasr.utils.status.require_passed`:

* artifacts are checked, not just statuses -- a stage that passed but whose
  outputs were deleted or rewritten does not satisfy a prerequisite;
* `--force-prereq` is contagious. A forced run is *tainted*: its status carries
  `taint.diagnostic_only`, every descendant inherits the union of its parents'
  taint reasons, and production gates refuse tainted evidence. A documented
  diagnostic can never quietly become the basis of a confirmatory result.

Taint is sticky by construction. `inherited_taint` unions the parents' reasons
with whatever the stage last recorded for itself, so re-running a tainted stage
without recomputing cannot launder its artifacts; clearing it requires
`--overwrite`, which recomputes them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..utils.hashing import sha256_file
from ..utils.status import LSS_STAGES, read_status

#: stage -> stages that must have passed first
STAGE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "l0_freeze": (),
    "l1a_diag": ("l0_freeze",),
    "l1b_valid": ("l0_freeze", "l1a_diag"),
    "l1c_labels": ("l1b_valid",),
    "l2a_prompts": ("l0_freeze",),
    "l2b_cache": ("l0_freeze", "l2a_prompts"),
    "l3_directions": ("l1c_labels", "l2b_cache"),
    "l4_oracle": ("l3_directions",),
    "l5_transport": ("l2b_cache", "l3_directions"),
    "l6_localizer": ("l1c_labels", "l2b_cache"),
    "l7a_util_labels": ("l4_oracle", "l6_localizer", "l2b_cache"),
    "l7b_selector": ("l7a_util_labels",),
    "l7c_calibrate": ("l7b_selector",),
    "l8_baselines": ("l0_freeze", "l4_oracle"),
    "l9a_eval": ("l7c_calibrate", "l8_baselines"),
    "l9b_report": ("l9a_eval",),
    "l10_test": ("l9b_report",),
}

#: stage -> artifact paths (relative to the artifacts root) that must exist
ARTIFACT_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "l1a_diag": (
        "freeze/spec_freeze_v1.json",
        "manifests/roles/role_D-construct.parquet",
    ),
    "l1b_valid": (
        "freeze/spec_freeze_v1.json",
        "manifests/roles/role_D-construct.parquet",
        "metrics/l1a_family_language_summary.parquet",
    ),
    "l1c_labels": (
        "alignments/consensus_spans_v1.parquet",
        "freeze/l1b_spans_freeze.json",
    ),
}

#: statuses that satisfy a prerequisite. `completed_no_go`, `blocked`,
#: `completed_roles_only` and `failed` are all terminal and none of them
#: unlocks anything: a valid negative is still a negative.
SATISFYING_STATUSES = ("passed",)

#: taint reason recorded when a stage ran with an unmet prerequisite
FORCED_PREREQUISITE = "forced_prerequisite"

#: taint reason recorded when a stage's provenance says its artifacts are not
#: production artifacts (smoke runs, `--limit`ed runs)
SMOKE_ARTIFACT = "smoke_artifact"


class PrerequisiteError(RuntimeError):
    """A stage was asked to run before its inputs were ready."""


def empty_taint() -> dict[str, Any]:
    return {"diagnostic_only": False, "taint_reasons": [], "parent_run_ids": []}


def taint_of(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The taint a status payload carries, in the repository's status schema.

    Reads the explicit `taint` block when present and reconstructs it from the
    older top-level markers otherwise, so a status file written before taint
    blocks existed is still recognised as diagnostic.
    """
    block = dict(payload.get("taint") or {})
    reasons = {str(r) for r in (block.get("taint_reasons") or [])}
    if payload.get("forced_prereq"):
        reasons.add(FORCED_PREREQUISITE)
    if (payload.get("provenance") or {}).get("production_artifact") is False:
        reasons.add(SMOKE_ARTIFACT)
    parents = [str(p) for p in (block.get("parent_run_ids") or [])]
    return {
        "diagnostic_only": bool(reasons) or bool(block.get("diagnostic_only")),
        "taint_reasons": sorted(reasons),
        "parent_run_ids": sorted(set(parents)),
    }


def artifact_taint(artifacts_root: str | Path, stage: str) -> dict[str, Any]:
    """Taint carried by the *bytes* this stage's declared inputs consist of.

    Status taint is mutable; a manifest is not. An artifact published by a
    forced run keeps its manifest even after the producing stage is re-run
    cleanly, so this is what stops `--overwrite` from laundering reused files.
    """
    from .manifest import load as load_manifest, merge_taint

    root = Path(artifacts_root)
    manifests = [load_manifest(root / rel)
                 for rel in ARTIFACT_PREREQUISITES.get(stage, ())]
    return merge_taint([m for m in manifests if m])


def inherited_taint(artifacts_root: str | Path, stage: str, *,
                    forced: bool = False, sticky: bool = True) -> dict[str, Any]:
    """Union of every parent's taint, the taint bound to the input artifacts,
    this run's own forcing, and (by default) whatever this stage recorded before.

    ``sticky=False`` is a full recompute (`--overwrite`): the stage's own status
    history is dropped, but neither its parents' taint nor the taint recorded in
    its inputs' manifests is -- that would be laundering. An overwrite therefore
    only produces a clean stage when the inputs themselves are clean.
    """
    root = Path(artifacts_root)
    reasons: set[str] = set()
    parents: set[str] = set()

    for parent in STAGE_PREREQUISITES.get(stage, ()):
        payload = read_status(root, parent)
        parent_taint = taint_of(payload)
        if parent_taint["diagnostic_only"]:
            reasons.update(parent_taint["taint_reasons"])
            parents.update(parent_taint["parent_run_ids"])
            run_id = payload.get("run_dir") or payload.get("slurm_job_id")
            if run_id:
                parents.add(str(run_id))

    # bytes outrank status: this survives --overwrite by construction
    from_artifacts = artifact_taint(root, stage)
    reasons.update(from_artifacts["taint_reasons"])
    parents.update(from_artifacts["parent_run_ids"])

    if sticky:
        own = taint_of(read_status(root, stage))
        reasons.update(own["taint_reasons"])
        parents.update(own["parent_run_ids"])

    if forced:
        reasons.add(FORCED_PREREQUISITE)

    return {"diagnostic_only": bool(reasons), "taint_reasons": sorted(reasons),
            "parent_run_ids": sorted(parents)}


def assert_acyclic(graph: Mapping[str, tuple[str, ...]] | None = None) -> None:
    """Raise if the declared dependency graph contains a cycle or unknown stage."""
    declared = graph is None
    graph = STAGE_PREREQUISITES if declared else graph
    if declared:
        # only the pipeline's own graph must use registered stage names; a
        # caller-supplied graph is checked for shape alone
        unknown = {s for s in graph} - set(LSS_STAGES)
        if unknown:
            raise ValueError(f"dependency graph names unknown stages: {sorted(unknown)}")
    for stage, prereqs in graph.items():
        bad = [p for p in prereqs if p not in graph]
        if bad:
            raise ValueError(f"stage {stage!r} depends on undeclared stage(s) {bad}")

    state: dict[str, int] = {}          # 0 = unvisited, 1 = on stack, 2 = done

    def visit(node: str, path: list[str]) -> None:
        mark = state.get(node, 0)
        if mark == 2:
            return
        if mark == 1:
            cycle = " -> ".join(path + [node])
            raise ValueError(f"dependency cycle: {cycle}")
        state[node] = 1
        for parent in graph[node]:
            visit(parent, path + [node])
        state[node] = 2

    for stage in graph:
        visit(stage, [])


def topological_order(graph: Mapping[str, tuple[str, ...]] | None = None) -> list[str]:
    """Stages in an order where every stage follows its prerequisites."""
    graph = STAGE_PREREQUISITES if graph is None else graph
    assert_acyclic(graph)
    order: list[str] = []
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        for parent in graph[node]:
            visit(parent)
        seen.add(node)
        order.append(node)

    for stage in graph:
        visit(stage)
    return order


def _status_problem(payload: dict[str, Any]) -> str | None:
    status = payload.get("status", "pending")
    if status not in SATISFYING_STATUSES:
        return status
    if not payload.get("complete", True):
        return "passed-but-incomplete"
    # a partial run that stopped early is terminal but must never unlock a
    # production stage, even though its own limited checks all passed.
    # `full_l0_pass` is L0's spelling of it; `full_stage_pass` is the generic one.
    if any(payload.get(k) is False for k in ("full_l0_pass", "full_stage_pass")):
        return "partial-run"
    taint = taint_of(payload)
    if taint["diagnostic_only"]:
        return "diagnostic-only(" + ",".join(taint["taint_reasons"]) + ")"
    return None


def require_prerequisites(artifacts_root: str | Path, stage: str, *,
                          force: bool = False, overwrite: bool = False,
                          log: Any = None) -> dict[str, Any]:
    """Check every declared prerequisite of ``stage``; raise unless satisfied.

    The returned report carries a ``taint`` block. A stage must write it into
    its own status via ``finish(..., taint=report["taint"])`` so the marker
    propagates to its descendants.
    """
    if stage not in STAGE_PREREQUISITES:
        raise KeyError(f"stage {stage!r} has no declared prerequisites")
    root = Path(artifacts_root)
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    for parent in STAGE_PREREQUISITES[stage]:
        payload = read_status(root, parent)
        problem = _status_problem(payload)
        checks.append({"kind": "stage", "name": parent,
                       "status": payload.get("status", "pending"),
                       "ok": problem is None, "problem": problem})
        if problem is not None:
            problems.append(f"{parent}={problem}")

    from .manifest import verify as verify_artifact

    for rel in ARTIFACT_PREREQUISITES.get(stage, ()):
        path = root / rel
        if not path.is_file():
            checks.append({"kind": "artifact", "name": rel, "ok": False,
                           "sha256": None, "problem": "missing"})
            problems.append(f"{rel}=missing")
            continue
        # Hashing the file and comparing it to itself proves nothing. The
        # producer's manifest is the recorded expectation, so a file rewritten
        # after its stage passed is detected here rather than consumed.
        verdict = verify_artifact(path)
        checks.append({
            "kind": "artifact", "name": rel, "ok": bool(verdict["ok"]),
            "sha256": sha256_file(path),
            "expected_sha256": (verdict.get("manifest") or {}).get("sha256"),
            "problem": None if verdict["ok"] else verdict["verdict"],
            "detail": verdict.get("detail", ""),
        })
        if not verdict["ok"]:
            problems.append(f"{rel}={verdict['verdict']}")

    forced_run = bool(force) and bool(problems)
    report = {
        "stage": stage,
        "satisfied": not problems,
        "forced": bool(force),
        "forced_run": forced_run,
        "checks": checks,
        "problems": problems,
        "taint": inherited_taint(root, stage, forced=forced_run,
                                 sticky=not bool(overwrite)),
    }
    if problems:
        message = (f"prerequisites for {stage} not satisfied: {', '.join(problems)}. "
                   "Run them first, or pass --force-prereq for a documented "
                   "diagnostic run (which marks this stage's output non-production).")
        if not force:
            raise PrerequisiteError(message)
        if log is not None:
            log.warning("FORCED: %s", message)
    return report


def prerequisite_report(artifacts_root: str | Path) -> pd.DataFrame:
    """One row per stage: its status, whether it is runnable, and why not."""
    root = Path(artifacts_root)
    rows = []
    for stage in topological_order():
        payload = read_status(root, stage)
        blockers = []
        for parent in STAGE_PREREQUISITES[stage]:
            problem = _status_problem(read_status(root, parent))
            if problem is not None:
                blockers.append(f"{parent}={problem}")
        for rel in ARTIFACT_PREREQUISITES.get(stage, ()):
            if not (root / rel).is_file():
                blockers.append(f"{rel}=missing")
        taint = taint_of(payload)
        rows.append({
            "stage": stage,
            "status": payload.get("status", "pending"),
            "runnable": not blockers,
            "diagnostic_only": bool(taint["diagnostic_only"]),
            "taint": ",".join(taint["taint_reasons"]),
            "blocked_by": ";".join(blockers),
            "prerequisites": ";".join(STAGE_PREREQUISITES[stage]),
        })
    return pd.DataFrame(rows)


assert_acyclic()
