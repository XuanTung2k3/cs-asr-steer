"""`role` has to survive consensus, or every per-role threshold is a lie.

`build_unit_consensus_v2` copies a fixed list of columns from the winning
candidate and `role` is not on it. On job 38573 that produced 32,086 accepted
spans and this:

    "per_role_spans": {"D-construct": {"spans": 0, "en_spans": 0, ...},
                       "loc-train":   {"spans": 0, ...}, ... all six roles}

so `min_loc_train_en_spans`, `min_dev_select_targets`, `min_dev_confirm_targets`
and `construct_bilingual_utterances` all failed on bookkeeping. A read-only
reconstruction of the same spans gives D-construct 1,124 / loc-train 1,370 /
util-train 1,952 / router-calib 2,006 / D-dev-select 1,322 / D-dev-confirm 1,274.
Those still miss the preregistered counts -- the point is that the gate must
report the real numbers and fail on them, not on a dropped column.

The mapping is joined back on in `csasr.lss.align.consensus_prod` rather than by
editing `csasr.nat5h`, whose recorded artifacts have to stay reproducible.
"""
from __future__ import annotations

import pandas as pd
import pytest

from csasr.lss.align import consensus_prod
from csasr.nat5h.schema import ALIGNMENT_SCHEMA_VERSION

SR = 16000
ROLES = ("D-construct", "loc-train", "util-train", "router-calib",
         "D-dev-select", "D-dev-confirm")


def _candidates(roles=ROLES, *, per_role: int = 3, agree: bool = True,
                families=("existing_ctc", "whisper_dtw")) -> pd.DataFrame:
    rows = []
    for role in roles:
        for i in range(per_role):
            for family in families:
                shift = 0.0 if agree or family == "existing_ctc" else 5.0
                for unit, language in enumerate(("ZH", "EN")):
                    start = unit * 0.5 + shift
                    rows.append({
                        "schema_version": ALIGNMENT_SCHEMA_VERSION,
                        "utterance_id": f"{role}_u{i}",
                        "conversation_id": f"c{i}", "split": "train",
                        "role": role,
                        "reference_unit_index": unit,
                        "reference_language": language,
                        "reference_text": "我" if language == "ZH" else "machine",
                        "aligner_family": family,
                        "aligner_variant": f"{family}/default",
                        "start_sec": start, "end_sec": start + 0.4,
                        "audio_duration_sec": 4.0,
                        "start_sample": int(start * SR),
                        "end_sample": int((start + 0.4) * SR),
                        "sample_rate": SR, "waveform_num_samples": int(4.0 * SR),
                        "is_valid": True, "failure_code": "", "failure_detail": "",
                        "source_token_count": 1, "mapping_method": "test",
                        "model_id": "fake", "model_revision": "0",
                        "code_commit": "unknown", "config_hash": "test",
                        "speaker": f"s{i}",
                    })
    return pd.DataFrame(rows)


def _config() -> consensus_prod.ConsensusConfig:
    return consensus_prod.ConsensusConfig.from_cfg(
        {"min_families": 2, "bins": {"high": 50, "medium": 100, "low": 200},
         "primary_tolerance_ms": 200.0})


def test_role_survives_consensus_for_every_configured_role():
    candidates = _candidates()
    accepted, rejected, _ = consensus_prod.build(candidates, _config())

    assert "role" in accepted.columns
    counts = accepted["role"].value_counts().to_dict()
    assert set(counts) == set(ROLES)
    assert all(v == 6 for v in counts.values()), counts       # 3 utterances x 2 units


def test_rejected_units_carry_their_role_too():
    """Coverage is accepted+rejected against the expected universe, so a rejected
    row without a role breaks the partition just as quietly."""
    candidates = _candidates(agree=False)
    accepted, rejected, _ = consensus_prod.build(candidates, _config())
    assert not len(accepted)
    assert "role" in rejected.columns
    assert set(rejected["role"]) == set(ROLES)


def test_nonempty_input_roles_cannot_become_all_zero_output_counts():
    """The exact regression: spans exist, and every role counts zero."""
    from csasr.experiments.lss_l1b_valid import _role_span_counts

    candidates = _candidates()
    accepted, _, _ = consensus_prod.build(candidates, _config())
    counts = _role_span_counts(accepted, list(ROLES))

    assert sum(v["spans"] for v in counts.values()) == len(accepted)
    assert not all(v["spans"] == 0 for v in counts.values()), \
        "spans exist but every role counts zero: the role column was dropped"
    for role in ROLES:
        assert counts[role]["spans"] == 6
        assert counts[role]["en_spans"] == 3 and counts[role]["zh_spans"] == 3
        assert counts[role]["utterances"] == 3


def test_a_span_table_without_a_role_fails_the_schema_check():
    """Loud rather than zero: `spans.SPAN_COLUMNS` declares `role`,
    `spans.load_spans` filters on it, and every per-role threshold reads it."""
    from csasr.experiments.lss_l1b_valid import _span_schema_ok

    accepted, _, _ = consensus_prod.build(_candidates(), _config())
    accepted = accepted.assign(confidence_bin="high", spec_freeze_sha256="x",
                               unit_id=accepted["reference_unit_index"])
    assert _span_schema_ok(accepted) is True
    assert _span_schema_ok(accepted.drop(columns=["role"])) is False


def test_an_utterance_claimed_by_two_roles_raises_rather_than_picking_one():
    candidates = _candidates(roles=("D-construct",))
    other = candidates.copy()
    other["role"] = "loc-train"
    with pytest.raises(ValueError, match="more than one role"):
        consensus_prod.utterance_roles(pd.concat([candidates, other],
                                                ignore_index=True))


def test_a_span_whose_utterance_is_not_in_the_mapping_raises():
    """A silent NaN there is what produced six zero-span roles."""
    accepted, _, _ = consensus_prod.build(_candidates(roles=("D-construct",)),
                                          _config())
    with pytest.raises(ValueError, match="no role in the frozen"):
        consensus_prod.attach_roles(accepted, {"someone_else": "loc-train"},
                                    what="accepted span")


def test_candidates_without_a_role_leave_the_spans_without_one():
    """Not silently invented. The schema check is what turns it into a failure."""
    candidates = _candidates(roles=("D-construct",)).drop(columns=["role"])
    accepted, _, _ = consensus_prod.build(candidates, _config())
    assert len(accepted) and "role" not in accepted.columns


def test_the_tolerance_sweep_still_works_with_roles_attached():
    """`tolerance_sweep` calls `build` per tolerance; the join must not disturb
    the counts it reports."""
    sweep = consensus_prod.tolerance_sweep(_candidates(), _config(), [50, 200])
    assert list(sweep["tolerance_ms"]) == [50.0, 200.0]
    assert (sweep["accepted"] == 6 * len(ROLES)).all()
