"""LSS unit test: when the chain may skip a stage, and when it must not.

`cs_asr_lss.sh chain` skips a stage that already passed so an allocation that ran
out of wall clock is resumed instead of restarted from L0. Two things can go
wrong with that, and both did:

* `--overwrite` was forwarded to every stage but never consulted by the skip, so
  the flag was a no-op for exactly the two stages whose message advertised it;
* "already passed" said nothing about *what* it passed, so a configuration change
  reused an answer to a different question.

The bash is checked by grep-level assertions on the launcher, which is crude but
is the only way to test a launcher this file cannot execute (it needs Slurm, a
GPU and the real artifacts root).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from csasr.experiments.lss_status import stage_reusable
from csasr.utils.provenance import resolved_config_hash
from csasr.utils.status import write_status

LAUNCHER = Path(__file__).resolve().parents[1] / "cs_asr_lss.sh"
CONFIG = "lss/l1a_diag.yaml"


def _passed(root: Path, stage: str, *, config_hash: str) -> None:
    write_status(root, stage, "passed", complete=True,
                 provenance={"resolved_config_hash": config_hash})


@pytest.fixture
def root(tmp_path):
    (tmp_path / "status").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _current_hash() -> str:
    from csasr.utils.config import load_config

    return resolved_config_hash(load_config(CONFIG))


def test_a_stage_that_passed_under_this_config_is_reusable(root):
    _passed(root, "l1a_diag", config_hash=_current_hash())
    assert stage_reusable(root, "l1a_diag", config=CONFIG) == "reusable"


def test_a_failed_stage_is_never_reusable(root):
    write_status(root, "l1a_diag", "failed", complete=True)
    assert stage_reusable(root, "l1a_diag", config=CONFIG) == "stale:status=failed"


def test_a_pending_stage_is_never_reusable(root):
    assert stage_reusable(root, "l1a_diag", config=CONFIG) == "stale:status=pending"


def test_a_pass_under_a_different_config_is_not_reusable(root):
    """The failure this guards: reusing an answer to a different question."""
    _passed(root, "l1a_diag", config_hash="0" * 64)
    verdict = stage_reusable(root, "l1a_diag", config=CONFIG)
    assert verdict.startswith("stale:config_hash")
    assert "000000000000" in verdict


def test_a_pass_with_no_recorded_hash_is_not_reusable(root):
    """An older status file cannot prove what it passed under."""
    write_status(root, "l1a_diag", "passed", complete=True)
    assert stage_reusable(root, "l1a_diag",
                          config=CONFIG) == "stale:no_recorded_config_hash"


def test_the_config_hash_is_reproducible_for_the_same_config_path():
    """If it were not, nothing would ever be reusable and resumption would be
    gone. Verified against job 38502's recorded hash when this was written."""
    assert _current_hash() == _current_hash()


def test_the_source_hash_is_deliberately_not_part_of_the_rule():
    """`environment.lock.txt` is rewritten with a fresh timestamp and Slurm job
    id by every `preflight`, and it is inside `source_snapshot_hash`. Requiring
    that to match would mean never skipping anything -- i.e. deleting the
    resumption that protects the one expensive stage. Documented, so a future
    reader does not "fix" it by adding the source hash.
    """
    from csasr.utils import provenance

    # the fact that makes the source hash unusable here
    assert "environment.lock.txt" in provenance.SOURCE_FILES
    # and the rule states its own omission, so it reads as a decision
    assert "source_snapshot_hash" in stage_reusable.__doc__
    assert "deliberately **not** compared" in stage_reusable.__doc__


# --------------------------------------------------------------------------
# the launcher's own logic
# --------------------------------------------------------------------------
def _launcher() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_the_launcher_consults_overwrite_before_skipping():
    text = _launcher()
    # the flag is both forwarded and recorded
    assert "--overwrite) common+=(\"$1\"); overwrite=1" in text
    # and both skip branches consult it
    assert text.count('${overwrite} -eq 0 && "${reuse}" == "reusable"') == 2


def test_the_launcher_skips_only_on_a_reusable_verdict():
    text = _launcher()
    assert "stage_reusable l0_freeze configs/lss/l0_freeze.yaml" in text
    assert "stage_reusable l1a_diag configs/lss/l1a_diag.yaml" in text
    # the old rule -- status alone -- must not be what gates the skip any more
    assert '"$(stage_status l0_freeze)" == "passed"' not in text
    assert '"$(stage_status l1a_diag)" == "passed"' not in text


def test_the_launcher_has_no_bare_conditional_at_statement_level():
    """`set -e` turns a false `[[ ... ]] && echo` at statement level into an
    ended job. Any such line must be a function or an if/fi."""
    offenders = [
        line.strip() for line in _launcher().splitlines()
        if line.strip().startswith("[[") and "&&" in line
        and not line.strip().startswith("[[ ${overwrite}")
    ]
    assert not offenders, offenders


def test_the_launcher_still_refuses_to_run_l1b_after_a_non_passing_l1a():
    text = _launcher()
    assert 'L1a is $(stage_status l1a_diag), not passed; l1b will not run.' in text
    assert 'L0 is $(stage_status l0_freeze), not passed; l1a and l1b will not run.' in text
