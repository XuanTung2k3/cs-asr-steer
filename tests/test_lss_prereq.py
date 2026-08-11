"""LSS unit test: the stage graph is acyclic and prerequisites are enforced."""
from __future__ import annotations

import pytest

from csasr.lss.prereq import (
    STAGE_PREREQUISITES,
    PrerequisiteError,
    assert_acyclic,
    prerequisite_report,
    require_prerequisites,
    topological_order,
)
from csasr.utils.status import LSS_STAGES, write_status


def test_declared_graph_is_acyclic():
    assert_acyclic()


def test_cycle_is_detected():
    with pytest.raises(ValueError, match="cycle"):
        assert_acyclic({"a": ("b",), "b": ("a",)})


def test_declared_graph_only_names_registered_stages():
    assert set(STAGE_PREREQUISITES) == set(LSS_STAGES)


def test_undeclared_parent_is_rejected():
    with pytest.raises(ValueError, match="undeclared"):
        assert_acyclic({"a": ("ghost",)})


def test_topological_order_places_every_stage_after_its_prerequisites():
    order = topological_order()
    assert set(order) == set(LSS_STAGES)
    position = {s: i for i, s in enumerate(order)}
    for stage, prereqs in STAGE_PREREQUISITES.items():
        for p in prereqs:
            assert position[p] < position[stage], f"{p} must precede {stage}"


def test_prompt_baselines_do_not_depend_on_the_oracle_stage():
    """Regression for the plan-v1 cycle: baselines waited on an oracle effect
    size while the oracle consumed baseline outputs."""
    order = topological_order()
    assert "l4_oracle" not in STAGE_PREREQUISITES["l2a_prompts"]
    assert order.index("l2a_prompts") < order.index("l4_oracle")


def test_pending_prerequisite_blocks_and_force_overrides(tmp_path):
    with pytest.raises(PrerequisiteError, match="l0_freeze=pending"):
        require_prerequisites(tmp_path, "l1a_diag")
    report = require_prerequisites(tmp_path, "l1a_diag", force=True)
    assert report["forced"] is True and report["satisfied"] is False


def test_completed_no_go_upstream_does_not_satisfy(tmp_path):
    write_status(tmp_path, "l0_freeze", "completed_no_go", complete=False)
    with pytest.raises(PrerequisiteError, match="completed_no_go"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_forced_upstream_is_rejected_downstream(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True, forced_prereq=True)
    with pytest.raises(PrerequisiteError, match="forced"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_missing_artifact_blocks_even_when_the_stage_passed(tmp_path):
    write_status(tmp_path, "l0_freeze", "passed", complete=True)
    with pytest.raises(PrerequisiteError, match="spec_freeze_v1.json=missing"):
        require_prerequisites(tmp_path, "l1a_diag")


def test_prerequisite_report_lists_every_stage(tmp_path):
    table = prerequisite_report(tmp_path)
    assert set(table["stage"]) == set(LSS_STAGES)
    first = table[table["stage"] == "l0_freeze"].iloc[0]
    assert bool(first["runnable"]) is True
    later = table[table["stage"] == "l1b_valid"].iloc[0]
    assert bool(later["runnable"]) is False and "l0_freeze" in later["blocked_by"]
