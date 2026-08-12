"""LSS unit test: alignment diagnostics measure, never repair by assumption."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csasr.lss.align.bias import (
    disagreement_regression,
    offset_correction_counterfactual,
    paired_edges,
    signed_bias_summary,
)
from csasr.lss.align.ladder import run_ladder, verdict
from csasr.lss.align.qwen_probe import families_after_probe, probe_qwen
from csasr.lss.align.repair import (
    OverlapRepairPolicy,
    overlap_inventory,
    repair_adjacent_overlaps,
)
from csasr.lss.align.romanization import cause_counts, probe_uroman
from csasr.lss.align.summaries import (
    family_language_summary,
    relabel_rejections,
    reproduction_check,
)

SR = 16000


def _candidate(utt, unit, family, start_s, end_s, language="ZH", valid=True,
               failure=""):
    return {
        "schema_version": "nat5h_alignment_v2",
        "utterance_id": utt, "reference_unit_index": unit,
        "reference_language": language, "reference_text": "x",
        "aligner_family": family, "aligner_variant": f"{family}/default",
        "start_sample": int(start_s * SR), "end_sample": int(end_s * SR),
        "start_sec": start_s, "end_sec": end_s,
        "is_valid": valid, "failure_code": failure,
    }


def test_one_frame_overlap_is_benign():
    """sec_to_frames floors starts and ceils ends, so 20 ms is a grid artifact."""
    frame = pd.DataFrame([
        _candidate("u1", 0, "whisper_dtw", 0.0, 0.52),
        _candidate("u1", 1, "whisper_dtw", 0.50, 1.0),
    ])
    inventory = overlap_inventory(frame)
    assert len(inventory) == 1
    assert inventory.iloc[0]["overlap_ms"] == pytest.approx(20.0, abs=1e-6)
    assert bool(inventory.iloc[0]["benign"]) is True


def test_overlap_inventory_includes_rows_flagged_invalid():
    """The overlapping rows are exactly the ones marked invalid; excluding them
    would report zero overlaps on a table full of them."""
    frame = pd.DataFrame([
        _candidate("u1", 0, "whisper_dtw", 0.0, 0.8, valid=False,
                   failure="adjacent_overlap"),
        _candidate("u1", 1, "whisper_dtw", 0.5, 1.0, valid=False,
                   failure="adjacent_overlap"),
    ])
    inventory = overlap_inventory(frame)
    assert len(inventory) == 1
    assert inventory.iloc[0]["overlap_ms"] == pytest.approx(300.0, abs=1e-6)
    assert bool(inventory.iloc[0]["benign"]) is False


def test_rows_without_timestamps_are_skipped():
    frame = pd.DataFrame([
        _candidate("u1", 0, "existing_ctc", 0.0, 0.5),
        {**_candidate("u1", 1, "existing_ctc", 0.0, 0.0, valid=False,
                      failure="mapping_failed"),
         "start_sample": -1, "end_sample": -1},
    ])
    assert len(overlap_inventory(frame)) == 0


def test_cross_language_overlap_truncates_only_the_earlier_span():
    frame = pd.DataFrame([
        _candidate("u1", 0, "whisper_dtw", 0.0, 0.8, language="ZH"),
        _candidate("u1", 1, "whisper_dtw", 0.5, 1.0, language="EN"),
    ])
    repaired, report = repair_adjacent_overlaps(frame, OverlapRepairPolicy())
    assert report["repaired"] == 1
    assert repaired.iloc[0]["end_sample"] == int(0.5 * SR)     # earlier one moved
    assert repaired.iloc[1]["start_sample"] == int(0.5 * SR)   # later one untouched
    assert report["median_boundary_movement_ms"] == pytest.approx(300.0, abs=1e-6)


def test_same_language_overlap_is_reported_not_truncated():
    """csasr.data.alignment.resolve_boundary_overlaps only handles cross-language
    pairs, so same-language collisions must stay visible."""
    frame = pd.DataFrame([
        _candidate("u1", 0, "whisper_dtw", 0.0, 0.8, language="ZH"),
        _candidate("u1", 1, "whisper_dtw", 0.5, 1.0, language="ZH"),
    ])
    repaired, report = repair_adjacent_overlaps(frame, OverlapRepairPolicy())
    assert report["repaired"] == 0
    assert report["skipped_same_language"] == 1
    assert repaired.iloc[0]["end_sample"] == int(0.8 * SR)


def test_repair_never_produces_empty_or_reversed_spans():
    frame = pd.DataFrame([
        _candidate("u1", 0, "whisper_dtw", 0.4, 0.9, language="ZH"),
        _candidate("u1", 1, "whisper_dtw", 0.2, 1.0, language="EN"),
    ])
    repaired, _ = repair_adjacent_overlaps(frame, OverlapRepairPolicy())
    assert (repaired["end_sample"] > repaired["start_sample"]).all()


def _paired_frame(n=40, offset_ms=-300.0, slope_ms_per_s=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = 0.5 + i * 0.25
        delta = offset_ms + slope_ms_per_s * t + rng.normal(0, 5)
        rows.append(_candidate(f"u{i//4}", i % 4, "existing_ctc", t, t + 0.3))
        rows.append(_candidate(f"u{i//4}", i % 4, "whisper_dtw",
                               t + delta / 1000.0, t + 0.3 + delta / 1000.0))
    return pd.DataFrame(rows)


def test_paired_edges_reports_signed_deltas():
    pairs = paired_edges(_paired_frame(offset_ms=-300.0))
    assert len(pairs) == 40
    assert pairs["dstart_ms"].median() == pytest.approx(-300.0, abs=15)


def test_regression_detects_a_true_scale_error():
    pairs = paired_edges(_paired_frame(offset_ms=0.0, slope_ms_per_s=40.0))
    report = disagreement_regression(pairs)
    assert report["theil_slope_ms_per_s"] == pytest.approx(40.0, rel=0.2)
    assert abs(report["r"]) >= 0.3
    assert report["scale_error_evidence"] is True


def test_regression_calls_a_constant_offset_a_constant_offset():
    """A flat, noisy relationship must not be read as a rate error, which is
    what a bare OLS slope on this corpus does."""
    pairs = paired_edges(_paired_frame(offset_ms=-500.0, slope_ms_per_s=0.0))
    report = disagreement_regression(pairs)
    assert abs(report["theil_slope_ms_per_s"]) < 5.0
    assert report["intercept_ms"] == pytest.approx(-500.0, abs=40)
    assert report["scale_error_evidence"] is False


def _ladder(canonical_ms: float, legacy_ms: float) -> pd.DataFrame:
    return pd.DataFrame([{"rung": "unit_edge_canonical", "language": "ALL",
                          "edge": "start", "median_abs_ms": canonical_ms},
                         {"rung": "switch_midpoint_legacy", "language": "ALL",
                          "edge": "midpoint", "median_abs_ms": legacy_ms}])


WEAK = {"theil_slope_ms_per_s": 19.0, "r": 0.13, "scale_error_evidence": False}
STRONG = {"theil_slope_ms_per_s": 19.0, "r": 0.8, "scale_error_evidence": True}


def test_verdict_requires_correlated_evidence_for_a_scale_error():
    """A noisy slope with r=0.13 is not a rate error, whatever OLS reports."""
    assert verdict(_ladder(360.0, 370.0), WEAK) == "refuted"


def test_a_statistic_effect_that_leaves_a_large_residual_is_only_partial():
    """E1 compared switch midpoints (370 ms); unit edges give 200 ms on the same
    data. That halves the number without making the alignment usable."""
    assert verdict(_ladder(200.0, 380.0), WEAK) == "partially_supported"


def test_full_support_needs_the_residual_to_clear_the_bar_too():
    assert verdict(_ladder(60.0, 380.0), WEAK) == "supported"
    assert verdict(_ladder(60.0, 65.0), STRONG) == "supported"
    assert verdict(_ladder(360.0, 365.0), STRONG) == "partially_supported"


def test_ladder_emits_rungs_and_a_recorded_verdict():
    table, evidence = run_ladder(_paired_frame(offset_ms=-400.0))
    assert evidence["rungs_completed"] >= 1
    assert evidence["verdict"] in {"supported", "refuted", "inconclusive"}
    assert set(table["rung"]) <= {"unit_edge_canonical", "switch_midpoint_legacy",
                                  "unit_edge_first_only"}


def test_signed_bias_summary_splits_by_language_and_edge():
    frame = _paired_frame(offset_ms=-200.0)
    summary = signed_bias_summary(paired_edges(frame), by=("language",))
    assert set(summary["edge"]) == {"start", "end"}
    assert (summary["n"] > 0).all()


def test_offset_counterfactual_is_marked_not_applied():
    """It shows what a correction would buy, precisely so nobody applies one
    fitted to inter-aligner agreement."""
    table = offset_correction_counterfactual(paired_edges(_paired_frame(offset_ms=-300.0)))
    assert not table["applied"].any()
    none = table[(table["correction"] == "none") & (table["tolerance_ms"] == 100)]
    median = table[(table["correction"] == "global_median") & (table["tolerance_ms"] == 100)]
    assert float(median["acceptance_rate"].iloc[0]) > float(none["acceptance_rate"].iloc[0])


def test_family_summary_handles_a_family_with_no_english():
    frame = pd.DataFrame([
        _candidate("u1", 0, "existing_ctc", 0.0, 0.5, language="ZH"),
        _candidate("u1", 1, "existing_ctc", 0.5, 1.0, language="ZH", valid=False,
                   failure="mapping_failed"),
    ])
    summary = family_language_summary(frame)
    assert len(summary) == 1
    assert summary.iloc[0]["valid_unit_coverage"] == pytest.approx(0.5)
    assert summary.iloc[0]["top_failure_code"] == "mapping_failed"


def test_rejection_codes_are_relabelled_to_the_applied_tolerance():
    """nat5h hard-codes `_gt_200ms` regardless of the configured tolerance."""
    rejected = pd.DataFrame([{"rejection_code": "start_disagreement_gt_200ms",
                              "reference_language": "EN"}])
    out = relabel_rejections(rejected, 100.0)
    assert out.iloc[0]["rejection_code"] == "start_disagreement_gt_100ms"
    assert out.iloc[0]["tolerance_ms"] == 100.0


def test_reproduction_check_tolerates_small_drift_only():
    assert reproduction_check({"consensus_spans": 222}, {"consensus_spans": 222})["reproduced"]
    assert reproduction_check({"consensus_spans": 224}, {"consensus_spans": 222})["reproduced"]
    assert not reproduction_check({"consensus_spans": 150},
                                  {"consensus_spans": 222})["reproduced"]


def test_uroman_probe_measures_expansion_and_vocabulary_loss():
    probe = probe_uroman(["machine", "我"], vocab={c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")})
    assert len(probe) == 2
    han = probe[probe["surface"] == "我"].iloc[0]
    assert han["n_chars_in"] == 1
    assert han["n_chars_out"] >= 2          # one Han character becomes a syllable
    assert han["expansion_ratio"] > 1.0


def test_cause_counts_summarizes_an_empty_table():
    assert cause_counts(pd.DataFrame()) == {}


def _qwen_cfg(model_dir):
    return {"alignment": {"qwen": {"local_model_dir": str(model_dir),
                                   "model_id": "x", "probe_utterances": 2}},
            "model": {"device": "cpu"}}


def _explode(monkeypatch, exc):
    class Boom:
        def __init__(self, *a, **k):
            raise exc

    monkeypatch.setattr("csasr.nat5h.aligners.Qwen3ForcedAlignerAdapter", Boom)


def test_qwen_probe_never_raises(monkeypatch, tmp_path):
    """A third aligner that explodes must not be able to take a stage down."""
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    _explode(monkeypatch, RuntimeError("adapter exploded"))
    result = probe_qwen(_qwen_cfg(model_dir),
                        pd.DataFrame({"utterance_id": ["u1"]}), None, None)
    # a RuntimeError is a defect in our own code, not a missing aligner, so it
    # must not be laundered into `blocked`
    assert result.state == "failed"
    assert "adapter exploded" in (result.reason or "")
    assert result.traceback


def test_infrastructure_failure_is_blocked_not_a_defect(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    _explode(monkeypatch, ImportError("no qwen runtime installed"))
    result = probe_qwen(_qwen_cfg(model_dir),
                        pd.DataFrame({"utterance_id": ["u1"]}), None, None)
    assert result.state == "blocked"
    assert result.timed_out is False


def test_probe_timeout_is_bounded_frees_the_model_and_quarantines_partials(
        monkeypatch, tmp_path):
    """A hung checkpoint load must not consume the allocation, must not leave
    the model resident, and must not leave a partial table the cache would
    reuse. A timeout is `blocked` -- never a scientific no-go."""
    import time

    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    out_dir = tmp_path / "alignments"
    out_dir.mkdir()
    stale = out_dir / "candidates_qwen_forced_aligner.parquet"
    stale.write_bytes(b"truncated")
    unloaded = []

    class Hang:
        def __init__(self, *a, **k):
            pass

        def load(self):
            time.sleep(30)

        def run(self, *a, **k):                     # pragma: no cover
            return pd.DataFrame()

        def unload(self):
            unloaded.append(True)

    monkeypatch.setattr("csasr.nat5h.aligners.Qwen3ForcedAlignerAdapter", Hang)
    started = time.monotonic()
    result = probe_qwen(_qwen_cfg(model_dir), pd.DataFrame({"utterance_id": ["u1"]}),
                        None, None, deadline_seconds=1, out_dir=out_dir)

    assert time.monotonic() - started < 10, "the deadline did not bound the probe"
    assert result.state == "blocked" and result.timed_out is True
    assert result.reason == "deadline_exceeded"
    assert result.state != "completed_no_go", "a timeout is not a measurement"
    assert unloaded, "the adapter must be released even when the alarm fires mid-load"
    assert not stale.exists(), "the partial table is still where the cache looks"
    assert result.quarantined and "quarantined" in result.quarantined[0]


def test_a_timed_out_probe_drops_the_family_from_the_consensus():
    from csasr.lss.align.qwen_probe import QwenProbeResult

    timed_out = QwenProbeResult(state="blocked", reason="deadline_exceeded",
                                timed_out=True)
    assert families_after_probe(["existing_ctc", "whisper_dtw",
                                 "qwen_forced_aligner"], timed_out) \
        == ["existing_ctc", "whisper_dtw"]


def test_missing_qwen_checkpoint_is_blocked_not_an_error():
    cfg = {"alignment": {"qwen": {"local_model_dir": "/nonexistent/path"}}}
    result = probe_qwen(cfg, pd.DataFrame(), None, None)
    assert result.state == "blocked" and result.reason == "checkpoint_missing"


# ---------------------------------------------------------------------------
# The probe must call the adapter the way the adapter is actually declared.
#
# Job 38502 died with `Qwen3ForcedAlignerAdapter.run() missing 1 required
# positional argument: 'language'`. Every existing probe test used a fake
# adapter whose `run(self, *a, **k)` accepts anything, so no test could see it,
# and the production path (`probe_qwen_subprocess`) had no test at all.
# ---------------------------------------------------------------------------
def _call_contract(func):
    """A signature reduced to what decides whether a call succeeds.

    Names, kinds and which parameters are optional -- not annotations, which
    cannot make a call fail and which the stub has no reason to mirror.
    """
    import inspect

    return [(p.name, p.kind, p.default is inspect.Parameter.empty)
            for p in inspect.signature(func).parameters.values()]


def _real_run_contract():
    from csasr.nat5h.aligners import Qwen3ForcedAlignerAdapter

    return _call_contract(Qwen3ForcedAlignerAdapter.run)


class StubAdapter:
    """Stands in for the real adapter with its *real* `run` signature.

    Defined at module scope so a subprocess can import it by name.
    """

    languages = ["chinese", "english"]

    def __init__(self, cfg):
        self.cfg = cfg
        self.package_version = "stub-0"
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def supported_languages(self):
        return list(self.languages)

    def run(self, manifest, geometry, language, identity, diagnostics_dir=None):
        assert self.loaded, "run() before load()"
        rows = []
        for _, row in manifest.iterrows():
            rows.append(_candidate(str(row["utterance_id"]), 0,
                                   "qwen_forced_aligner", 0.0, 0.5))
            rows[-1]["aligner_variant"] = f"qwen_forced_aligner/{language}"
        return pd.DataFrame(rows)


def test_the_stub_adapter_matches_the_real_adapters_run_signature():
    """The guard that makes every other stub test meaningful.

    A stub with a permissive `run(*a, **k)` proves nothing about the call the
    probe makes. This pins the stub to the real declaration, so a change to
    `Qwen3ForcedAlignerAdapter.run` breaks here instead of on the cluster.
    """
    stub = _call_contract(StubAdapter.run)
    real = _real_run_contract()
    assert stub == real, (
        f"StubAdapter.run{stub} no longer matches "
        f"Qwen3ForcedAlignerAdapter.run{real}; update the stub and check that "
        "probe_qwen still calls it correctly")


def test_the_probe_calls_run_with_the_language_it_requires(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    out_dir = tmp_path / "alignments"
    monkeypatch.setattr("csasr.nat5h.aligners.Qwen3ForcedAlignerAdapter", StubAdapter)

    result = probe_qwen(_qwen_cfg(model_dir),
                        pd.DataFrame({"utterance_id": ["u1", "u2"]}),
                        None, None, out_dir=out_dir)

    assert result.state == "ok", result.reason
    assert result.candidate_rows == 2
    assert result.package_version == "stub-0"
    # exactly one variant, and the one the frozen consensus order ranks first
    assert result.language_diagnostics[0]["variants"] == \
        ["qwen_forced_aligner/Chinese"]
    table = pd.read_parquet(out_dir / "candidates_qwen_forced_aligner.parquet")
    assert set(table["aligner_variant"]) == {"qwen_forced_aligner/Chinese"}


def test_one_variant_because_the_consensus_would_discard_a_second():
    """`PROBE_LANGUAGE` must be the variant `_family_representatives` picks.

    A probe that persisted a variant the consensus ranks second would spend GPU
    time on rows that are then dropped in favour of a variant that does not
    exist in the table.
    """
    from csasr.lss.align.qwen_probe import PROBE_LANGUAGE
    from csasr.nat5h.schema import ALIGNER_VARIANT_ORDER

    qwen = [v for v in ALIGNER_VARIANT_ORDER if v.startswith("qwen_forced_aligner/")]
    assert qwen, "the frozen variant order no longer mentions qwen"
    assert qwen[0] == f"qwen_forced_aligner/{PROBE_LANGUAGE}"


def test_a_language_the_checkpoint_does_not_declare_is_blocked(monkeypatch, tmp_path):
    """A capability change must not be run as if it were supported."""
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()

    class OtherLanguages(StubAdapter):
        languages = ["japanese"]

    monkeypatch.setattr("csasr.nat5h.aligners.Qwen3ForcedAlignerAdapter",
                        OtherLanguages)
    result = probe_qwen(_qwen_cfg(model_dir),
                        pd.DataFrame({"utterance_id": ["u1"]}), None, None)
    assert result.state == "blocked"
    assert "probe_language_unsupported" in (result.reason or "")


def test_the_production_subprocess_path_publishes_a_successful_probe(
        monkeypatch, tmp_path):
    """End to end through the real child process, with a stub adapter.

    `probe_qwen_subprocess` is the production path and nothing exercised it: not
    the request round-trip, not `publish_attempt`, not the `state == "ok"`
    branch. A monkeypatch cannot reach a fresh interpreter, so the child is
    pointed at the stub through `ADAPTER_ENV_VAR`.
    """
    import os
    import sys

    from csasr.lss.align.qwen_probe import ADAPTER_ENV_VAR, probe_qwen_subprocess

    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    out_dir = tmp_path / "alignments"
    out_dir.mkdir()

    monkeypatch.setenv(ADAPTER_ENV_VAR,
                       f"{StubAdapter.__module__}:{StubAdapter.__qualname__}")
    # the child is a fresh interpreter: it needs this process's import paths,
    # including the directory this test module lives in
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(Path(__file__).resolve().parent)]
        + [p for p in sys.path if p]))

    result = probe_qwen_subprocess(
        _qwen_cfg(model_dir), pd.DataFrame({"utterance_id": ["u1", "u2"]}),
        deadline_seconds=300, out_dir=out_dir)

    assert result.state == "ok", (result.reason, result.traceback)
    assert result.candidate_rows == 2
    published = out_dir / "candidates_qwen_forced_aligner.parquet"
    assert published.is_file(), "a successful probe must publish its candidates"
    assert not list(out_dir.glob(".qwen_attempt_*")), \
        "the attempt directory must be cleaned up after publishing"
    assert result.published == [str(published)]


def test_families_fall_back_to_two_when_qwen_is_blocked():
    configured = ["existing_ctc", "whisper_dtw", "qwen_forced_aligner"]
    from csasr.lss.align.qwen_probe import QwenProbeResult

    blocked = QwenProbeResult(state="blocked", reason="checkpoint_missing")
    assert families_after_probe(configured, blocked) == ["existing_ctc", "whisper_dtw"]
    ok = QwenProbeResult(state="ok")
    assert families_after_probe(configured, ok) == configured


# --------------------------------------------------------------------------
# why Whisper-DTW's spans overlap: attribution, not repair
# --------------------------------------------------------------------------
def _dtw_row(utterance, index, text, start, end, language="ZH"):
    return {"utterance_id": utterance, "reference_unit_index": index,
            "reference_text": text, "reference_language": language,
            "aligner_family": "whisper_dtw", "aligner_variant": "whisper_dtw/zh_median7",
            "start_sample": int(start * 16000), "end_sample": int(end * 16000),
            "is_valid": True, "failure_code": ""}


def test_two_units_sharing_one_whisper_token_are_attributed_not_repaired():
    """The measured mechanism behind Whisper-DTW's 17.1% invalid rate.

    On job 38573, 15,112 of 88,246 candidates were flagged `adjacent_overlap`, all
    Mandarin, and in 100.0% of them the two spans were byte-identical -- the
    overlap equalled the unit's own duration to the millisecond (mean 391.747 ms
    on both). That is only possible if both reference units were assigned the same
    token span, which is what Whisper's BPE produces for `他们`: one token, two
    characters, no sub-token resolution to recover.

    Nothing repairs it. Truncating the earlier span would invent a boundary inside
    one token from no acoustic evidence, and buy exactly the coverage number the
    0.95 threshold exists to measure.
    """
    from csasr.lss.align.repair import attribute_overlaps

    candidates = pd.DataFrame([
        _dtw_row("u1", 5, "他", 1.66, 1.98),
        _dtw_row("u1", 6, "们", 1.66, 1.98),          # identical span
        _dtw_row("u1", 7, "说", 1.98, 2.38),
    ])
    table, report = attribute_overlaps(candidates)

    assert report["overlaps"] == 1
    assert report["causes"] == {"shared_source_token": 1}
    assert report["identical_span_fraction"] == 1.0
    assert report["cross_language_fraction"] == 0.0
    assert table["previous_reference_text"].iloc[0] == "他"
    assert table["overlap_ms"].iloc[0] == pytest.approx(320.0)
    assert table["duration_ms"].iloc[0] == pytest.approx(320.0)
    assert "not repairable by truncation" in report["note"]


def test_a_one_frame_overlap_is_told_apart_from_a_shared_token():
    """`sec_to_frames` floors starts and ceils ends, so a single 20 ms overlap is
    an artifact of the grid rather than a collision."""
    from csasr.lss.align.repair import attribute_overlaps

    candidates = pd.DataFrame([
        _dtw_row("u1", 0, "对", 0.00, 0.74),
        _dtw_row("u1", 1, "我", 0.72, 0.96),          # 20 ms, different span
    ])
    _, report = attribute_overlaps(candidates)
    assert report["causes"] == {"frame_grid_artifact": 1}
    assert report["identical_span_fraction"] == 0.0


def test_a_genuine_partial_overlap_is_its_own_cause():
    from csasr.lss.align.repair import attribute_overlaps

    candidates = pd.DataFrame([
        _dtw_row("u1", 0, "对", 0.00, 0.90),
        _dtw_row("u1", 1, "我", 0.60, 1.20),          # 300 ms, distinct spans
    ])
    _, report = attribute_overlaps(candidates)
    assert report["causes"] == {"partial_overlap": 1}


def test_attribution_never_moves_a_boundary():
    """It is a diagnostic. A repair that buys validity by moving an edge would
    have to be argued for on external truth, not asserted here."""
    from csasr.lss.align.repair import attribute_overlaps

    candidates = pd.DataFrame([
        _dtw_row("u1", 5, "他", 1.66, 1.98),
        _dtw_row("u1", 6, "们", 1.66, 1.98),
    ])
    before = candidates.copy(deep=True)
    attribute_overlaps(candidates)
    pd.testing.assert_frame_equal(candidates, before)


def test_attribution_of_an_empty_or_single_family_table_is_empty():
    from csasr.lss.align.repair import attribute_overlaps

    table, report = attribute_overlaps(pd.DataFrame())
    assert not len(table) and report["overlaps"] == 0
    table, report = attribute_overlaps(
        pd.DataFrame([_dtw_row("u1", 0, "对", 0.0, 0.5)]))
    assert not len(table) and report["overlaps"] == 0


def test_overlaps_are_counted_within_an_utterance_not_across_them():
    """The last unit of one utterance cannot overlap the first of the next."""
    from csasr.lss.align.repair import attribute_overlaps

    candidates = pd.DataFrame([
        _dtw_row("u1", 0, "对", 0.0, 3.0),
        _dtw_row("u2", 0, "我", 0.0, 0.5),
    ])
    _, report = attribute_overlaps(candidates)
    assert report["overlaps"] == 0
