"""Gate construction: criteria, grouping, and the status a gate implies.

Gates in this pipeline are grouped because the group that fails determines the
*response*. An alignment stage that fails on measured boundary accuracy is a
different event from one that passes accuracy but cannot find enough reliable
spans: the first says stop and repair, the second triggers the pre-registered
fallback ladder. That distinction is preserved in ``gate["groups"]``.

The *status* a gate implies is a separate question, and answers only this: did
the experiment run correctly?

    mechanical / reporting fail  ->  `failed`
        The artifacts are malformed or a required report was not written. That
        is an implementation defect, not a result.
    external / jitter / coverage fail  ->  `completed_no_go`
        The experiment ran correctly and the measurement did not clear a
        pre-registered scientific threshold. That is a result.
    evidence unavailable  ->  `blocked`
        Something the gate needs does not exist yet (a second aligner, the
        synthetic calibration, human verdicts in manual mode). Nothing is
        broken and nothing has been measured.

`completed_no_go` is emphatically **not** a licence to continue: no terminal
status other than `passed` satisfies a downstream prerequisite, and no
threshold is ever relaxed to turn one into the other.

Exit codes follow from that, and separate command health from scientific
outcome (see ``STATUS_EXIT_CODES``): a job that produced a truthful
`completed_no_go` or `blocked` did its work and exits 0, so Slurm `afterok`
does not report the allocation as failed. Only `failed` -- an implementation
defect -- is a nonzero exit. Orchestration reads the status file, never the
exit code, to decide whether the next stage may run.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..utils.status import criterion, gate

#: criterion groups, in the order they are reported
GROUP_ORDER = ("mechanical", "external", "jitter", "coverage", "reporting")

#: groups whose failure means "the experiment ran and the answer is no", as
#: opposed to "the experiment is broken". Membership decides `completed_no_go`
#: vs `failed`; it does not decide the response, which is keyed off the group
#: name itself in the pre-registered `failure_response` config.
NO_GO_GROUPS = frozenset({"external", "jitter", "coverage"})

#: process exit code per terminal status. 0 means the command completed and
#: wrote a truthful terminal status; the science is in the status file.
STATUS_EXIT_CODES = {
    "passed": 0,
    "completed": 0,
    "completed_no_go": 0,
    "completed_roles_only": 0,
    "awaiting_manual_verdicts": 0,
    "blocked": 0,
    "insufficient_data": 0,
    "failed": 2,
}


def exit_code(status: str) -> int:
    """Process exit code for a terminal stage status.

    Unknown statuses are treated as failures: a stage that ends in a state this
    contract does not describe has not demonstrated that it worked.
    """
    return int(STATUS_EXIT_CODES.get(status, 2))


def finite(value: Any, default: float = float("nan")) -> float:
    """Coerce to float; anything non-numeric or non-finite becomes NaN.

    NaN never satisfies a comparison, so a missing measurement fails its
    criterion instead of silently passing it.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def check(name: str, value: Any, threshold: Any, comparison: str = ">=",
          *, group: str = "mechanical") -> dict[str, Any]:
    """A criterion whose pass/fail is derived from ``comparison``."""
    if comparison == "report":
        return reported(name, value, group=group)
    numeric = finite(value)
    limit = finite(threshold)
    if comparison == ">=":
        passed = numeric >= limit
    elif comparison == "<=":
        passed = numeric <= limit
    elif comparison == "==":
        passed = numeric == limit
    elif comparison == ">":
        passed = numeric > limit
    elif comparison == "<":
        passed = numeric < limit
    else:
        raise ValueError(f"unknown comparison {comparison!r}")
    out = criterion(name, value, threshold, bool(passed), comparison)
    out["group"] = group
    return out


def reported(name: str, value: Any, *, group: str = "reporting") -> dict[str, Any]:
    """Evidence that is recorded but never gates anything."""
    out = criterion(name, value, "reported_only", True, "report")
    out["group"] = group
    return out


def group_results(criteria: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Per-group pass/fail; a group with no criteria is vacuously true."""
    out: dict[str, bool] = {}
    for c in criteria:
        if c.get("comparison") == "report":
            continue
        g = str(c.get("group", "mechanical"))
        out[g] = out.get(g, True) and bool(c.get("passed"))
    return out


def classify(criteria: Sequence[Mapping[str, Any]], *,
             blocked: bool = False,
             no_go_groups: frozenset[str] = NO_GO_GROUPS) -> str:
    """Map criterion outcomes onto a stage status.

    ``blocked`` is set by the caller when the gate cannot conclude because
    evidence it requires does not exist, not because anything is wrong.

    Precedence is `failed` > `blocked` > `completed_no_go` > `passed`:

    * a malformed artifact must be repaired before any reading of it means
      anything, so it outranks everything;
    * a gate that never saw its required evidence has not run the experiment,
      so it may not report the experiment's *result* -- `completed_no_go` would
      claim a measurement that was not made. Any thresholds that did fail are
      still recorded in ``gate["groups"]`` and in the criteria table, so
      nothing is hidden by this ordering.
    """
    groups = group_results(criteria)
    failed = {g for g, ok in groups.items() if not ok}
    if failed and not failed <= set(no_go_groups):
        return "failed"
    if blocked:
        return "blocked"
    return "completed_no_go" if failed else "passed"


def evaluate(criteria: Sequence[Mapping[str, Any]], *, name: str, note: str = "",
             blocked: bool = False, blocked_reasons: Sequence[str] = (),
             no_go_groups: frozenset[str] = NO_GO_GROUPS) -> tuple[dict[str, Any], str]:
    """Build the gate payload and the status it implies.

    ``blocked_reasons`` are machine-readable codes naming the evidence that is
    missing (for example ``blocked_missing_synthetic_calibration``). Passing any
    reason implies ``blocked``, so a caller cannot record a reason and still
    report a pass.
    """
    reasons = list(dict.fromkeys(str(r) for r in blocked_reasons))   # dedupe, ordered
    status = classify(criteria, blocked=blocked or bool(reasons),
                      no_go_groups=no_go_groups)
    payload = gate(name, status == "passed", list(criteria), note)
    payload["groups"] = group_results(criteria)
    payload["status"] = status
    payload["blocked_reasons"] = reasons
    payload["exit_code"] = exit_code(status)
    return payload, status


def failed_criteria(gate_payload: Mapping[str, Any]) -> list[str]:
    return [c["name"] for c in gate_payload.get("criteria", [])
            if c.get("comparison") != "report" and not c.get("passed")]
