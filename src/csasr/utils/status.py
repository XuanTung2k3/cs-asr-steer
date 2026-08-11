"""Machine-readable stage status and gate bookkeeping (guide section 3)."""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

E1E5_STAGES = [
    "p0",
    "e1",
    "e2_pilot",
    "e2_full",
    "e3_pilot",
    "e3_full",
    "e4_pilot",
    "e4_refine",
    "e4_confirm",
    "e5",
]

# Localize-Select-Steer pipeline (docs/proposal_arr). It lives in its own
# artifacts root, so its stages must not appear in the E1-E5 overview and vice
# versa; `STAGE_GROUPS` is what keeps the two separate.
LSS_STAGES = [
    "l0_freeze",
    "l1a_diag",
    "l1b_valid",
    "l1c_labels",
    "l2a_prompts",
    "l2b_cache",
    "l3_directions",
    "l4_oracle",
    "l5_transport",
    "l6_localizer",
    "l7a_util_labels",
    "l7b_selector",
    "l7c_calibrate",
    "l8_baselines",
    "l9a_eval",
    "l9b_report",
    "l10_test",
]

STAGE_GROUPS = {"e1_e5": E1E5_STAGES, "lss": LSS_STAGES}
STAGE_GROUP_ORDER = ("e1_e5", "lss")

STAGES = E1E5_STAGES + LSS_STAGES

# `completed_no_go` is a scientifically valid negative that is not a crash: the
# stage ran correctly and the evidence says do not proceed. `insufficient_data`
# is the same shape for "not enough material to decide". `completed_roles_only`
# is the terminal state of a partial run that deliberately stopped early.
# All of them are terminal and all of them block downstream stages, exactly like
# `failed`; they exist so the distinction survives into the reports. Vocabulary
# borrowed from `csasr.nat5h.statusing.TERMINAL_STATES`.
# `completed` is a run that did its work and decided nothing -- an evidence
# preparation step. `awaiting_manual_verdicts` is the one state that waits on a
# person, and only the explicitly requested manual workflow may enter it.
ALLOWED = {"pending", "running", "passed", "failed", "blocked",
           "completed_no_go", "insufficient_data", "completed_roles_only",
           "completed", "awaiting_manual_verdicts"}

#: statuses a stage can end in. Anything else means the run is still in flight.
TERMINAL_STATUSES = frozenset(ALLOWED - {"pending", "running"})


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def status_path(artifacts_root: str | Path, stage: str) -> Path:
    p = Path(artifacts_root) / "status" / f"{stage}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_status(artifacts_root: str | Path, stage: str) -> dict[str, Any]:
    p = status_path(artifacts_root, stage)
    if not p.exists():
        return {"stage": stage, "status": "pending"}
    return json.loads(p.read_text(encoding="utf-8"))


def write_status(artifacts_root: str | Path, stage: str, status: str,
                 gate: dict | None = None, **extra: Any) -> dict[str, Any]:
    if status not in ALLOWED:
        raise ValueError(f"status must be one of {sorted(ALLOWED)}, got {status!r}")
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": _now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if gate is not None:
        payload["gate"] = gate
    payload.update(extra)
    p = status_path(artifacts_root, stage)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(p)
    write_overview(artifacts_root, stages=overview_stages(artifacts_root, stage))
    return payload


def group_for_stage(stage: str) -> str | None:
    for name, members in STAGE_GROUPS.items():
        if stage in members:
            return name
    return None


def overview_stages(artifacts_root: str | Path, stage: str | None = None) -> list[str]:
    """Which stage names belong in this artifacts root's overview.

    A root holds one pipeline, so listing every stage of every pipeline would
    report a permanent wall of `pending` for stages that will never run here.
    A group is included when the stage being written belongs to it, or when the
    root already contains one of its status files. An empty root falls back to
    E1-E5 so existing behaviour is unchanged.
    """
    groups: list[str] = []
    current = group_for_stage(stage) if stage is not None else None
    if current is not None:
        groups.append(current)
    for name, members in STAGE_GROUPS.items():
        if name in groups:
            continue
        if any(status_path(artifacts_root, s).exists() for s in members):
            groups.append(name)
    if not groups:
        groups = ["e1_e5"]
    return [s for name in STAGE_GROUP_ORDER if name in groups for s in STAGE_GROUPS[name]]


def write_overview(artifacts_root: str | Path,
                   stages: list[str] | None = None) -> dict[str, str]:
    stages = overview_stages(artifacts_root) if stages is None else stages
    overview = {s: read_status(artifacts_root, s).get("status", "pending") for s in stages}
    p = Path(artifacts_root) / "status" / "overview.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    tmp.replace(p)
    return overview


def require_passed(artifacts_root: str | Path, stages: list[str]) -> None:
    """Raise unless every prerequisite stage has passed."""
    bad = []
    for s in stages:
        payload = read_status(artifacts_root, s)
        st = payload.get("status", "pending")
        prov = payload.get("provenance") or {}
        complete = payload.get("complete", st == "passed")
        production = prov.get("production_artifact", True)
        if st != "passed" or not complete or production is False:
            reason = st
            if st == "passed" and not complete:
                reason = "passed-but-incomplete"
            if production is False:
                reason = "smoke-artifact"
            bad.append((s, reason))
    if bad:
        msg = ", ".join(f"{s}={st}" for s, st in bad)
        raise RuntimeError(
            f"prerequisite stage(s) not passed: {msg}. "
            "Run them first, or pass --force-prereq for a documented diagnostic run."
        )


class StageLock:
    """Simple advisory lock so two jobs cannot write the same stage."""

    def __init__(self, artifacts_root: str | Path, stage: str):
        self.path = Path(artifacts_root) / "status" / f"{stage}.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"stage {self.stage} is locked by: {holder.strip()}. "
                f"Remove {self.path} if that run is dead."
            )
        os.write(fd, f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID')} at={_now()}".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def gate(name: str, passed: bool, criteria: list[dict[str, Any]],
         decision_note: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "criteria": criteria,
        "note": decision_note,
        "evaluated_at": _now(),
    }


def criterion(name: str, value: Any, threshold: Any, passed: bool,
              comparison: str = ">=") -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
    }
