"""LSS unit test: grouped gates map evidence onto the right stage status."""
from __future__ import annotations

import math

from csasr.lss.gates import check, classify, evaluate, failed_criteria, reported
from csasr.utils.status import ALLOWED


def _criteria(mechanical=True, external=True, coverage=True):
    return [
        check("m", 1 if mechanical else 0, 1, "==", group="mechanical"),
        check("x", 0.95 if external else 0.10, 0.90, ">=", group="external"),
        check("c", 800 if coverage else 12, 500, ">=", group="coverage"),
        reported("observed_rate", 0.167),
    ]


def test_all_groups_pass_gives_passed():
    assert classify(_criteria()) == "passed"


def test_only_coverage_failing_is_a_valid_negative():
    assert classify(_criteria(coverage=False)) == "completed_no_go"


def test_a_missed_scientific_threshold_is_a_result_not_a_crash():
    """External accuracy is never weakened -- but failing it is a measurement,
    so it is `completed_no_go`, not `failed`. The failing *group* is what
    selects the pre-registered response, and that survives in the payload."""
    assert classify(_criteria(external=False, coverage=False)) == "completed_no_go"
    assert classify(_criteria(external=False)) == "completed_no_go"
    payload, _ = evaluate(_criteria(external=False), name="gate_a")
    assert payload["groups"]["external"] is False
    assert payload["passed"] is False


def test_a_malformed_artifact_is_a_defect():
    assert classify(_criteria(mechanical=False)) == "failed"
    assert classify(_criteria(mechanical=False, coverage=False)) == "failed"


def test_blocked_is_only_reachable_when_nothing_failed():
    assert classify(_criteria(), blocked=True) == "blocked"
    assert classify(_criteria(mechanical=False), blocked=True) == "failed"


def test_blocked_reasons_imply_blocked_and_forbid_a_pass():
    payload, status = evaluate(_criteria(), name="gate_a",
                               blocked_reasons=["blocked_missing_synthetic_calibration"])
    assert status == "blocked"
    assert payload["passed"] is False
    assert payload["blocked_reasons"] == ["blocked_missing_synthetic_calibration"]


def test_exit_codes_separate_command_health_from_scientific_outcome():
    from csasr.lss.gates import exit_code

    for status in ("passed", "completed", "completed_no_go", "blocked",
                   "completed_roles_only", "awaiting_manual_verdicts"):
        assert exit_code(status) == 0, f"{status} completed; it must exit 0"
    assert exit_code("failed") == 2
    assert exit_code("something_unheard_of") == 2


def test_every_status_the_gate_can_emit_is_an_allowed_status():
    for evidence in (_criteria(), _criteria(coverage=False), _criteria(external=False)):
        assert classify(evidence) in ALLOWED
    assert "blocked" in ALLOWED


def test_nan_never_passes():
    c = check("missing_measurement", float("nan"), 0.9, ">=")
    assert c["passed"] is False
    c = check("missing_measurement", None, 0.9, "<=")
    assert c["passed"] is False


def test_reported_criteria_never_gate():
    only_reports = [reported("a", 1), reported("b", float("nan"))]
    assert classify(only_reports) == "passed"


def test_criterion_shape_matches_the_repo_contract():
    c = check("x", 1.0, 0.5, ">=")
    assert set(c) == {"name", "value", "threshold", "comparison", "passed", "group"}
    assert c["comparison"] in {">=", "<=", "==", ">", "<", "report"}


def test_evaluate_returns_gate_payload_and_status():
    payload, status = evaluate(_criteria(coverage=False), name="l1b_valid",
                               note="grouped Gate A")
    assert status == "completed_no_go"
    assert payload["passed"] is False
    assert payload["groups"]["external"] is True
    assert payload["groups"]["coverage"] is False
    assert failed_criteria(payload) == ["c"]
