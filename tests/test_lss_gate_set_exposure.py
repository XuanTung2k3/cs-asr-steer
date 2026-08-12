"""A confirmatory gate is only confirmatory once.

Job 38573's synthetic gate scores were read during review -- the per-family
medians, the p90s, the -700 ms bias -- and the tolerance-selection rule was then
changed in response to them. Those 100 items can still be scored; they can no
longer answer "does the configuration we just chose hold on data nobody had
looked at". Reusing them would be the oldest way to make a gate pass: tune, look,
tune again, report the last number.

So exposure is recorded, not remembered, and the record is what the next gate
generation is drawn against.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csasr.lss.align import exposure


def _items(purpose: str, ids, generation=None) -> pd.DataFrame:
    label = purpose if generation in (None, 0) else f"{purpose}_g{generation}"
    return pd.DataFrame([{
        "pair_id": f"{label}_syn_{i:04d}",
        "zh_utterance_id": f"zh_{i}", "en_utterance_id": f"en_{i}",
        "purpose": purpose,
    } for i in ids])


def _pool(n: int, offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame({"utterance_id": [f"zh_{i}" for i in range(offset, offset + n)]})


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------
def test_an_empty_root_has_an_empty_ledger(tmp_path):
    ledger = exposure.load_ledger(tmp_path)
    assert ledger["schema_version"] == exposure.LEDGER_SCHEMA
    assert ledger["generations"] == []
    assert exposure.exposed_source_ids(ledger) == set()
    assert exposure.current_generation(ledger, "gate") == 1


def test_recording_a_generation_persists_its_sources_and_a_fingerprint(tmp_path):
    frame = _items("gate", range(3), generation=1)
    exposure.record_generation(tmp_path, frame, purpose="gate", generation=1,
                              reason="rendered for a Gate A evaluation")

    payload = json.loads(exposure.ledger_path(tmp_path).read_text())
    entry = payload["generations"][0]
    assert entry["purpose"] == "gate" and entry["generation"] == 1
    assert entry["exposed"] is False
    assert set(entry["source_utterance_ids"]) == {"zh_0", "zh_1", "zh_2",
                                                 "en_0", "en_1", "en_2"}
    assert len(entry["source_fingerprint"]) == 64
    assert entry["pair_ids"] == ["gate_g1_syn_0000", "gate_g1_syn_0001",
                                 "gate_g1_syn_0002"]


def test_an_unexposed_generation_is_reused_and_an_exposed_one_is_not(tmp_path):
    """Re-rendering an unexposed set from the same sources is waste, not
    freshness. Once a gate evaluation has read it, the number moves on."""
    exposure.record_generation(tmp_path, _items("gate", range(3), 1),
                              purpose="gate", generation=1, reason="rendered")
    assert exposure.current_generation(exposure.load_ledger(tmp_path), "gate") == 1

    exposure.mark_exposed(tmp_path, purpose="gate", generation=1,
                          reason="read by a Gate A evaluation that concluded blocked")
    ledger = exposure.load_ledger(tmp_path)
    assert exposure.current_generation(ledger, "gate") == 2
    entry = exposure.find_generation(ledger, "gate", 1)
    assert entry["exposed"] is True and entry["exposed_at"]
    assert "concluded blocked" in entry["exposed_reason"]


def test_re_rendering_a_generation_cannot_un_expose_it(tmp_path):
    """The items have been seen. Rewriting the record must not launder that."""
    exposure.record_generation(tmp_path, _items("gate", range(3), 1),
                              purpose="gate", generation=1, reason="rendered")
    exposure.mark_exposed(tmp_path, purpose="gate", generation=1, reason="read")
    exposure.record_generation(tmp_path, _items("gate", range(3), 1),
                              purpose="gate", generation=1, reason="re-rendered")

    ledger = exposure.load_ledger(tmp_path)
    assert exposure.find_generation(ledger, "gate", 1)["exposed"] is True
    assert len(exposure.generations_for(ledger, "gate")) == 1, "no duplicate entry"


def test_a_generation_number_cannot_be_rebound_to_different_sources(tmp_path):
    """A retry may allocate a new generation, but it may not rewrite history."""
    original = _items("gate", range(3), 1)
    exposure.record_generation(
        tmp_path, original, purpose="gate", generation=1,
        reason="first attempt", alignment_request_sha256="a" * 64,
        item_set_sha256="c" * 64)

    with pytest.raises(exposure.GenerationConflictError, match="immutable"):
        exposure.record_generation(
            tmp_path, _items("gate", range(10, 13), 1), purpose="gate",
            generation=1, reason="retry", alignment_request_sha256="b" * 64,
            item_set_sha256="d" * 64)

    entry = exposure.find_generation(exposure.load_ledger(tmp_path), "gate", 1)
    assert entry["source_utterance_ids"] == sorted(
        {"zh_0", "zh_1", "zh_2", "en_0", "en_1", "en_2"})
    assert entry["alignment_request_sha256"] == "a" * 64
    assert entry["item_set_sha256"] == "c" * 64


def test_an_interrupted_generation_is_retired_not_recycled(tmp_path):
    """A failed attempt reserves its audio and its generation number forever."""
    frame = _items("gate", range(3), 1)
    exposure.record_generation(
        tmp_path, frame, purpose="gate", generation=1,
        reason="attempt started", alignment_request_sha256="a" * 64,
        item_set_sha256="b" * 64)
    before = exposure.find_generation(exposure.load_ledger(tmp_path), "gate", 1)
    immutable_before = {
        key: before[key]
        for key in ("source_utterance_ids", "source_fingerprint", "pair_ids",
                    "alignment_request_sha256", "item_set_sha256")
    }

    exposure.abandon_unexposed(
        tmp_path, purpose="gate", reason="interrupted before Gate A")
    ledger = exposure.load_ledger(tmp_path)
    retired = exposure.find_generation(ledger, "gate", 1)

    assert retired["abandoned"] is True and retired["abandoned_at"]
    assert {key: retired[key] for key in immutable_before} == immutable_before
    assert exposure.next_generation(ledger, "gate") == 2
    assert exposure.current_generation(ledger, "gate") == 2
    eligible, report = exposure.eligible_sources(_pool(6), ledger)
    assert report["excluded"] == 3
    assert set(eligible["utterance_id"]) == {"zh_3", "zh_4", "zh_5"}


# --------------------------------------------------------------------------
# excluding what has been seen
# --------------------------------------------------------------------------
def test_a_new_gate_generation_excludes_every_exposed_source(tmp_path):
    exposure.record_generation(tmp_path, _items("gate", range(5), 1),
                              purpose="gate", generation=1, reason="rendered")
    exposure.mark_exposed(tmp_path, purpose="gate", generation=1, reason="read")

    eligible, report = exposure.eligible_sources(_pool(10),
                                                exposure.load_ledger(tmp_path))
    assert report["pool"] == 10 and report["excluded"] == 5
    assert set(eligible["utterance_id"]) == {f"zh_{i}" for i in range(5, 10)}
    assert report["exposed_fingerprint"]


def test_a_set_that_reuses_exposed_audio_raises(tmp_path):
    exposure.record_generation(tmp_path, _items("gate", range(5), 1),
                              purpose="gate", generation=1, reason="rendered")
    exposure.mark_exposed(tmp_path, purpose="gate", generation=1, reason="read")
    ledger = exposure.load_ledger(tmp_path)

    with pytest.raises(AssertionError, match="fresh confirmatory gate"):
        exposure.assert_unexposed(_items("gate", range(3), 2), ledger)

    fresh = exposure.assert_unexposed(_items("gate", range(5, 8), 2), ledger)
    assert fresh["fresh"] is True and fresh["reused_exposed"] == []


def test_re_rendering_the_same_generation_is_allowed_explicitly(tmp_path):
    """Its own sources are in the ledger, so the guard has to be told which
    generation is being rebuilt or it would refuse every re-run."""
    frame = _items("gate", range(3), 1)
    exposure.record_generation(tmp_path, frame, purpose="gate", generation=1,
                              reason="rendered")
    ledger = exposure.load_ledger(tmp_path)
    own = exposure.find_generation(ledger, "gate", 1)["source_utterance_ids"]
    assert exposure.assert_unexposed(frame, ledger, allow=own)["fresh"] is True


# --------------------------------------------------------------------------
# the sets that predate the ledger
# --------------------------------------------------------------------------
def test_sets_rendered_before_the_ledger_are_bootstrapped_as_exposed(tmp_path):
    """Job 38573 left `l1b_synthetic_{dev,gate}_items.parquet` behind. Without
    this, its gate sources would look unseen and could be drawn again."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    _items("dev", range(0, 4)).to_parquet(
        metrics / "l1b_synthetic_dev_items.parquet", index=False)
    _items("gate", range(4, 8)).to_parquet(
        metrics / "l1b_synthetic_gate_items.parquet", index=False)

    ledger = exposure.bootstrap(tmp_path)
    exposure.save_ledger(tmp_path, ledger)

    generations = {(g["purpose"], g["generation"]): g
                   for g in ledger["generations"]}
    assert set(generations) == {("dev", 0), ("gate", 0)}
    assert all(g["exposed"] for g in generations.values())
    assert "38573" in generations[("gate", 0)]["reason"]
    # the first fresh gate this root can produce is generation 1, and it cannot
    # use anything either legacy set drew on
    assert exposure.current_generation(ledger, "gate") == 1
    exposed = exposure.exposed_source_ids(ledger)
    assert {"zh_4", "en_4", "zh_0", "en_0"} <= exposed
    _, report = exposure.eligible_sources(_pool(12), ledger)
    assert report["excluded"] == 8


def test_bootstrapping_twice_does_not_duplicate_the_record(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    _items("gate", range(4)).to_parquet(
        metrics / "l1b_synthetic_gate_items.parquet", index=False)
    first = exposure.bootstrap(tmp_path)
    exposure.save_ledger(tmp_path, first)
    second = exposure.bootstrap(tmp_path)
    assert len(second["generations"]) == len(first["generations"]) == 1


def test_a_corrupt_ledger_fails_closed_and_is_not_replaced(tmp_path):
    path = exposure.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(exposure.ExposureLedgerError, match="unreadable"):
        exposure.load_ledger(tmp_path)
    assert path.read_text(encoding="utf-8") == "{not json"
    path.write_text(json.dumps({"schema_version": "something_older",
                                "generations": [{"purpose": "gate"}]}),
                    encoding="utf-8")
    with pytest.raises(exposure.ExposureLedgerError, match="expected schema"):
        exposure.load_ledger(tmp_path)
    assert json.loads(path.read_text())["schema_version"] == "something_older"


def test_the_storage_label_versions_a_generation(tmp_path):
    """A fresh generation must not read the cached candidate tables of the one it
    replaces; they are different items under the same family names."""
    from csasr.lss.align.synthetic import set_label

    assert set_label("dev") == "dev"
    assert set_label("gate", None) == "gate"
    assert set_label("gate", 0) == "gate"
    assert set_label("gate", 2) == "gate_g2"
