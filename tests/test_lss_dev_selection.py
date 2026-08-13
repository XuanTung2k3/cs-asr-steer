"""Configuration selection: development evidence only, and it must actually run.

Two failures these cover.

**The pred_start experiment was unreachable.** `bias.pred_start_sweep` needed
synthetic development data; that data was built by l1b; l1b requires l1a. So the
one licensed test of the ~700 ms Whisper bias was stranded across a dependency
boundary and L1a logged "deferred" instead. A test here has to fail if L1a merely
says that again -- which is why the assertions are on the artifact and the frozen
selection, not on a log line.

**The tolerance needed a human.** `spec.yaml` names the human audit as the
instrument for a rule about the absolute boundary error of accepted spans, so
automatic L1b blocked and told the user to annotate. The estimand is unchanged
here and so are the numbers; the instrument is synthetic development boundaries,
and the gate set is never allowed near a selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csasr.experiments import lss_l1a_diag
from csasr.lss import manifest as manifest_mod
from csasr.lss.align import devselect
from csasr.lss.align.consensus_prod import ConsensusConfig
from csasr.nat5h.schema import ALIGNMENT_SCHEMA_VERSION, RunIdentity

SR = 16000


# --------------------------------------------------------------------------
# fixtures: rendered items and the candidates a family would produce for them
# --------------------------------------------------------------------------
def _items(n: int, purpose: str = "dev") -> pd.DataFrame:
    return pd.DataFrame([{
        "pair_id": f"{purpose}_syn_{i:04d}",
        "utterance_id": f"{purpose}_syn_{i:04d}",
        "audio_path": f"/nonexistent/{purpose}_{i}.wav",
        "zh_text": "我 用", "en_text": "machine learning",
        "duration_sec": 4.0, "purpose": purpose,
        "zh_end_sec": 2.0, "en_start_sec": 2.0, "true_boundary_sec": 2.0,
        "reference_kind": "constructed_exact_lexical",
        "reference_semantics": "test_fixture_exact_lexical_edges",
        "zh_utterance_id": f"{purpose}_zh_{i}", "en_utterance_id": f"{purpose}_en_{i}",
    } for i in range(n)])


def _candidates(items: pd.DataFrame, families=("existing_ctc", "whisper_dtw"),
                *, offsets: dict | None = None) -> pd.DataFrame:
    """Four units per item: ZH, ZH, EN, EN, with the seam after the second."""
    offsets = offsets or {}
    rows = []
    for family in families:
        shift = float(offsets.get(family, 0.0))
        for _, item in items.iterrows():
            seam = 2.0 + shift
            for unit, (language, start, end) in enumerate([
                    ("ZH", 0.2, 1.0), ("ZH", 1.0, seam),
                    ("EN", seam, seam + 0.6), ("EN", seam + 0.6, seam + 1.2)]):
                rows.append({
                    "schema_version": ALIGNMENT_SCHEMA_VERSION,
                    "utterance_id": item["pair_id"], "conversation_id": "synthetic",
                    "split": f"synthetic_{item['purpose']}",
                    "reference_unit_index": unit, "reference_language": language,
                    "reference_text": "x", "aligner_family": family,
                    "aligner_variant": f"{family}/default",
                    "start_sec": start, "end_sec": end, "audio_duration_sec": 4.0,
                    "start_sample": int(start * SR), "end_sample": int(end * SR),
                    "sample_rate": SR, "waveform_num_samples": int(4.0 * SR),
                    "is_valid": True, "failure_code": "", "failure_detail": "",
                    "source_token_count": 1, "mapping_method": "test",
                    "model_id": "fake", "model_revision": "0",
                    "code_commit": "unknown", "config_hash": "test",
                })
    return pd.DataFrame(rows)


def _config() -> ConsensusConfig:
    return ConsensusConfig.from_cfg({"min_families": 2, "estimator": "both_edges",
                                     "bins": {"high": 50, "medium": 100, "low": 200}})


# --------------------------------------------------------------------------
# the operating tolerance
# --------------------------------------------------------------------------
def test_an_accurate_configuration_selects_the_most_generous_tolerance():
    """Accuracy selects, coverage never does, and among the tolerances that meet
    accuracy the widest is taken because tightening cannot create agreement."""
    items = _items(100)
    evidence = devselect.tolerance_accuracy(
        items, _candidates(items), _config(), [50, 100, 200, 300])
    record = devselect.select_operating_tolerance(evidence)

    assert list(evidence["tolerance_ms"]) == [50.0, 100.0, 200.0, 300.0]
    assert (evidence["median_abs_error_ms"] == 0).all()
    assert record["selected_tolerance_ms"] == 300.0
    assert record["instrument"] == "synthetic_dev"
    assert record["thresholds"] == {"max_median_abs_error_ms": 100.0,
                                    "max_p90_abs_error_ms": 200.0}
    assert "never coverage" in record["selected_by"]


def test_a_family_that_is_500ms_out_qualifies_at_no_tolerance():
    """The measured state: Whisper-DTW is ~700 ms early, so consensus seams are
    far outside 100/200 ms and no operating point exists to report a result at."""
    items = _items(100)
    candidates = _candidates(items, offsets={"whisper_dtw": -0.5})
    evidence = devselect.tolerance_accuracy(items, candidates, _config(),
                                            [50, 100, 200, 300])
    record = devselect.select_operating_tolerance(evidence)

    assert record["selected_tolerance_ms"] is None
    assert record["qualifying_tolerances_ms"] == []
    # and it says which threshold each tolerance missed, per tolerance
    assert record["rejection_reasons"]
    reasons = " ".join(sum(record["rejection_reasons"].values(), []))
    assert "median_abs_error_ms" in reasons or "items_scored" in reasons


def test_a_tolerance_scored_on_three_items_cannot_be_selected():
    """A median over a handful of items that happened to agree is not a
    selection, and missing measurements must never help."""
    items = _items(4)
    evidence = devselect.tolerance_accuracy(items, _candidates(items), _config(),
                                            [200], min_boundaries=50)
    assert bool(evidence["eligible"].iloc[0]) is False
    record = devselect.select_operating_tolerance(evidence)
    assert record["selected_tolerance_ms"] is None
    assert "items_scored=4<50" in " ".join(record["rejection_reasons"]["200"])


def test_the_selection_refuses_gate_rows():
    items = _items(100, purpose="gate")
    with pytest.raises(devselect.DevelopmentOnlyError):
        devselect.tolerance_accuracy(items, _candidates(items), _config(), [200])


def test_erosion_and_padding_come_from_the_same_measurement():
    """`spec.yaml`'s erosion rule with the development instrument in place of the
    human one; the fields that record which instrument it was are not optional."""
    items = _items(100)
    candidates = _candidates(items, offsets={"whisper_dtw": 0.08})
    evidence = devselect.tolerance_accuracy(items, candidates, _config(),
                                            [100, 200, 300])
    record = devselect.select_operating_tolerance(evidence)
    assert record["selected_tolerance_ms"] is not None

    applied = devselect.apply_selection(_config(), record)
    assert applied.tolerance_selected is True
    assert applied.measurement_instrument == "synthetic_dev"
    assert applied.erosion_ms == pytest.approx(
        record["measured_at_selected"]["median_abs_error_ms"])
    assert applied.union_padding_ms == pytest.approx(
        record["measured_at_selected"]["p90_abs_error_ms"])
    # and the rule it was selected by travels with it
    assert "synthetic_dev" in applied.selection_rule


def test_measuring_the_error_does_not_unselect_the_tolerance():
    """`with_measured` rebuilt the object field by field and dropped
    `tolerance_selected`, so measuring the error un-made the decision."""
    selected = ConsensusConfig.from_cfg({"primary_tolerance_ms": 100.0})
    assert selected.tolerance_selected is True
    measured = selected.with_measured(median_error_ms=40.0, p90_error_ms=90.0,
                                      instrument="synthetic_dev")
    assert measured.tolerance_selected is True
    assert measured.tolerance_ms == 100.0
    assert measured.selection_rule == selected.selection_rule
    assert measured.measurement_instrument == "synthetic_dev"


def test_an_unselected_record_leaves_the_configuration_untouched():
    """No silent 200 ms: a record with nothing selected changes nothing, so the
    caller has to notice and block."""
    base = _config()
    unchanged = devselect.apply_selection(base, {"selected_tolerance_ms": None})
    assert unchanged is base
    assert unchanged.tolerance_selected is False


def test_consensus_spans_are_what_gets_scored_not_candidates():
    """The operating tolerance governs the *consensus* seam, which is what a
    steering mask is built from."""
    spans = pd.DataFrame([{
        "utterance_id": "dev_syn_0000", "reference_unit_index": 1,
        "reference_language": "ZH", "consensus_start_second": 1.0,
        "consensus_end_second": 1.9,
    }])
    reshaped = devselect.consensus_as_candidates(spans, estimator="both_edges")
    assert reshaped["aligner_family"].iloc[0] == "consensus"
    assert reshaped["aligner_variant"].iloc[0] == "consensus/both_edges"
    assert reshaped["end_sec"].iloc[0] == pytest.approx(1.9)
    assert bool(reshaped["is_valid"].iloc[0]) is True
    # and it is not an aligner family, so it cannot be counted as one
    from csasr.lss.align.autoevidence import INDEPENDENCE_CLASS

    assert devselect.CONSENSUS_FAMILY not in INDEPENDENCE_CLASS


# --------------------------------------------------------------------------
# the decoder-query convention
# --------------------------------------------------------------------------
def test_the_convention_with_the_smaller_development_error_is_selected():
    sweep = pd.DataFrame([
        {"pred_start_offset": -1, "n_boundaries": 100,
         "median_abs_error_ms": 700.0, "p90_abs_error_ms": 1000.0,
         "median_signed_error_ms": -700.0, "within_100ms": 0.05},
        {"pred_start_offset": 0, "n_boundaries": 100,
         "median_abs_error_ms": 120.0, "p90_abs_error_ms": 300.0,
         "median_signed_error_ms": -100.0, "within_100ms": 0.45},
    ])
    record = devselect.select_whisper_variant(sweep)
    assert record["pred_start_offset"] == 0
    assert record["aligner_variant"] == "whisper_dtw/zh_median7_pred0"
    assert record["changed_the_default"] is True
    assert record["instrument"] == "synthetic_dev"
    assert record["measured"]["median_abs_error_ms"] == 120.0


def test_audio_splice_sweep_selects_coordinates_without_lexical_error_names(
        monkeypatch):
    """The production RMS/VAD development set is a seam diagnostic.  The L1a
    convention sweep must remain executable after reference typing, but it may
    not relabel the seam offset as lexical absolute error."""
    items = _items(60).copy()
    items["reference_kind"] = "audio_splice"
    items["reference_semantics"] = "known_audio_seam_not_lexical_boundary"

    def fake_run(bundle, manifest, cfg, identity, *, pred_start_offset=None):
        shift = -0.7 if int(pred_start_offset) == -1 else -0.1
        return (_candidates(items, families=("whisper_dtw",),
                            offsets={"whisper_dtw": shift}),
                {"unit_coverage": 1.0})

    monkeypatch.setattr("csasr.nat5h.aligners.run_whisper_dtw", fake_run)
    sweep = devselect.whisper_variant_sweep(
        _Bundle(), items, {"alignment": {}},
        RunIdentity(model_id="fake", model_revision="0", code_commit="test",
                    config_hash="test"),
        offsets=(-1, 0))

    assert set(sweep["metric_semantics"]) == {
        "audio_seam_relative_not_lexical_accuracy"}
    assert "median_abs_error_ms" not in sweep.columns
    assert "median_absolute_splice_edge_offset_ms" in sweep.columns
    selected = devselect.select_whisper_variant(sweep, min_boundaries=50)
    assert selected["pred_start_offset"] == 0
    assert selected["metric_semantics"] == \
        "audio_seam_relative_not_lexical_accuracy"
    assert "median_abs_error_ms" not in selected["measured"]
    assert selected["measured"]["median_absolute_splice_edge_offset_ms"] \
        == pytest.approx(100.0)


def test_the_default_convention_keeps_its_historical_variant_name():
    """Recorded artifacts were produced under offset -1 and must keep meaning
    what they meant."""
    from csasr.nat5h.aligners import dtw_variant_label

    assert dtw_variant_label(-1) == "whisper_dtw/zh_median7"
    assert dtw_variant_label(0) == "whisper_dtw/zh_median7_pred0"
    sweep = pd.DataFrame([
        {"pred_start_offset": -1, "n_boundaries": 100, "median_abs_error_ms": 50.0,
         "p90_abs_error_ms": 90.0, "median_signed_error_ms": 10.0,
         "within_100ms": 0.9},
        {"pred_start_offset": 0, "n_boundaries": 100, "median_abs_error_ms": 400.0,
         "p90_abs_error_ms": 600.0, "median_signed_error_ms": 400.0,
         "within_100ms": 0.1},
    ])
    record = devselect.select_whisper_variant(sweep)
    assert record["pred_start_offset"] == -1
    assert record["changed_the_default"] is False


def test_a_convention_with_too_few_boundaries_cannot_be_selected():
    sweep = pd.DataFrame([
        {"pred_start_offset": -1, "n_boundaries": 3, "median_abs_error_ms": 1.0,
         "p90_abs_error_ms": 2.0, "median_signed_error_ms": 0.0,
         "within_100ms": 1.0},
    ])
    record = devselect.select_whisper_variant(sweep, min_boundaries=50)
    assert record["pred_start_offset"] is None
    assert "n_boundaries=3<50" in " ".join(record["rejected"]["-1"])


def test_the_selected_convention_is_the_one_production_alignment_runs():
    """`run_whisper_dtw` reads the frozen offset and puts it in the variant label
    and the row metadata, so a table cannot be silently from the other one."""
    from csasr.nat5h.aligners import selected_pred_start_offset

    assert selected_pred_start_offset({}) == -1
    assert selected_pred_start_offset(
        {"alignment": {"dtw_selected": {"pred_start_offset": 0}}}) == 0


def test_two_conventions_do_not_share_a_candidate_cache():
    """Both resolve to the same `cfg`, so `artifact_matches_identity` cannot tell
    their tables apart -- the sweep would score one convention twice."""
    from csasr.lss.align.candidates import _cache_name

    assert _cache_name("whisper_dtw", -1) == "candidates_whisper_dtw.parquet"
    assert _cache_name("whisper_dtw", 0) == "candidates_whisper_dtw_pred0.parquet"
    assert _cache_name("existing_ctc", 0) == "candidates_existing_ctc.parquet"


# --------------------------------------------------------------------------
# the L1a artifact, through the real stage helper
# --------------------------------------------------------------------------
class _Bundle:
    model_id = "fake"
    revision = "0"
    encoder_step_sec = 0.02
    sample_rate = SR


@pytest.fixture
def l1a_root(tmp_path, monkeypatch):
    """An artifacts root where L1a's selection helper can run without a GPU."""
    root = tmp_path / "artifacts_lss"
    (root / "status").mkdir(parents=True, exist_ok=True)
    items = _items(100)

    monkeypatch.setattr("csasr.lss.align.devselect.build_dev_set",
                        lambda cfg, **kw: (items, {
                            "path": str(root / devselect.DEV_ITEMS_FILE),
                            "sha256": "devsha", "pairs": 100,
                            "fingerprint": "fp", "settings": {"gap_ms": 0.0},
                            "source_utterances": sorted(items["zh_utterance_id"])}))
    monkeypatch.setattr("csasr.models.whisper.load_whisper", lambda cfg: _Bundle())

    def fake_run_whisper_dtw(bundle, manifest, cfg, identity, *,
                             pred_start_offset=None):
        # offset 0 is accurate, -1 is 700 ms early: the hypothesis under test
        shift = 0.0 if int(pred_start_offset) == 0 else -0.7
        table = _candidates(items, families=("whisper_dtw",),
                            offsets={"whisper_dtw": shift})
        return table, {"unit_coverage": 1.0}

    monkeypatch.setattr("csasr.nat5h.aligners.run_whisper_dtw", fake_run_whisper_dtw)
    return root, items


def _cfg_for(root: Path) -> dict:
    return {
        "experiment": {"output_root": str(root), "seed": 42},
        "model": {"id": "fake"},
        "seeds": {"synthetic_dev": 201, "synthetic_gate": 202},
        "synthetic": {"num_pairs_dev": 100, "source_role": "D-construct"},
        "alignment": {"dtw_variants": {"pred_start_offsets": [-1, 0]}},
        "gate_a": {"min_pred_start_boundaries": 50},
    }


def test_l1a_writes_the_sweep_artifact_and_freezes_a_selection(l1a_root, caplog):
    """The assertion that "deferred" can never satisfy: the artifact exists, has
    a row per configured offset, and the freeze names the chosen variant."""
    root, _ = l1a_root
    cfg = _cfg_for(root)
    out = lss_l1a_diag._select_aligner_configuration(
        cfg, lss_l1a_diag.log if hasattr(lss_l1a_diag, "log") else _Logger(),
        None, {"taint_reasons": []}, qwen=None)

    sweep_path = root / devselect.PRED_START_SWEEP_TABLE
    assert sweep_path.is_file(), "l1a must produce metrics/l1a_pred_start_sweep.parquet"
    sweep = pd.read_parquet(sweep_path)
    assert sorted(sweep["pred_start_offset"]) == [-1, 0]
    assert set(sweep["purpose"]) == {"dev"}
    assert set(sweep["instrument"]) == {"synthetic_dev"}
    # the sweep measured what it claims to measure
    early = sweep[sweep["pred_start_offset"] == -1].iloc[0]
    assert early["median_signed_error_ms"] == pytest.approx(-700.0, abs=1.0)

    selection = json.loads((root / devselect.ALIGNER_SELECTION_FILE).read_text())
    assert selection["schema_version"] == devselect.ALIGNER_SELECTION_SCHEMA
    assert selection["pred_start_offset"] == 0
    assert selection["development_artifact"]["sha256"] == "devsha"
    assert selection["seeds"]["synthetic_dev"] == 201
    assert out["whisper"]["changed_the_default"] is True


def test_l1b_reads_the_frozen_selection_rather_than_the_default(l1a_root):
    root, _ = l1a_root
    lss_l1a_diag._select_aligner_configuration(_cfg_for(root), _Logger(), None,
                                               {"taint_reasons": []}, qwen=None)
    loaded = devselect.load_aligner_selection(root)
    assert loaded["available"] is True
    assert loaded["pred_start_offset"] == 0


def test_an_unmanifested_selection_is_not_believed(l1a_root):
    """A JSON file with no sidecar could be a leftover from any configuration."""
    root, _ = l1a_root
    lss_l1a_diag._select_aligner_configuration(_cfg_for(root), _Logger(), None,
                                               {"taint_reasons": []}, qwen=None)
    manifest_mod.manifest_path(root / devselect.ALIGNER_SELECTION_FILE).unlink()

    loaded = devselect.load_aligner_selection(root)
    assert loaded["available"] is False
    assert loaded["reason"] == "unauthenticated"


def test_a_tainted_selection_is_refused(l1a_root):
    """A `--force-prereq` run's configuration choice cannot govern production."""
    root, _ = l1a_root
    lss_l1a_diag._select_aligner_configuration(
        _cfg_for(root), _Logger(), None,
        {"taint_reasons": ["forced_prerequisite"]}, qwen=None)
    loaded = devselect.load_aligner_selection(root)
    assert loaded["available"] is False and loaded["reason"] == "tainted"


def test_a_dry_run_cannot_satisfy_l1bs_prerequisite():
    """A dry run skips the development set and the sweep, so whatever its own
    checks say it has not done the work l1b depends on."""
    import inspect

    source = inspect.getsource(lss_l1a_diag._run)
    assert "not args.dry_run" in source
    assert "full_stage_pass=bool(status == \"passed\" and not args.only" in source
    assert "and not args.dry_run)" in source


class _Logger:
    """Minimal logger: these helpers log, and a test should not care how."""

    def info(self, *a, **k):
        pass

    warning = error = debug = info


def test_the_development_set_is_published_with_its_own_schema(tmp_path,
                                                             monkeypatch):
    """L1a publishes it and L1b verifies it; identity is deliberately not part of
    that check because the two stages resolve different config files."""
    root = tmp_path / "artifacts_lss"
    root.mkdir(parents=True)
    items = _items(10)
    cfg = {"experiment": {"output_root": str(root)}, "model": {"id": "fake"}}
    manifest_mod.publish_frame(root / devselect.DEV_ITEMS_FILE, items,
                               stage="l1a_diag", cfg=cfg,
                               key_columns=("pair_id",),
                               schema=devselect.DEV_ITEMS_SCHEMA)

    frame, report = devselect.load_dev_set(root)
    assert report["available"] is True and len(frame) == 10
    assert report["sha256"]

    # a table published under a different schema is refused
    manifest_mod.publish_frame(root / devselect.DEV_ITEMS_FILE, items,
                               stage="l1a_diag", cfg=cfg,
                               key_columns=("pair_id",), schema="something_else")
    frame, report = devselect.load_dev_set(root)
    assert report["available"] is False and report["reason"] == "schema_mismatch"
    assert not len(frame)


def test_a_missing_development_set_is_reported_not_invented(tmp_path):
    frame, report = devselect.load_dev_set(tmp_path)
    assert not len(frame)
    assert report["available"] is False and report["reason"] == "missing"
    assert "run l1a" in report["detail"]


def test_a_rewritten_development_set_fails_authentication(tmp_path):
    """Bytes changed after publication: the manifest is the producer's recorded
    expectation, so hashing the file against itself would prove nothing."""
    root = tmp_path / "artifacts_lss"
    root.mkdir(parents=True)
    cfg = {"experiment": {"output_root": str(root)}, "model": {"id": "fake"}}
    manifest_mod.publish_frame(root / devselect.DEV_ITEMS_FILE, _items(10),
                               stage="l1a_diag", cfg=cfg,
                               key_columns=("pair_id",),
                               schema=devselect.DEV_ITEMS_SCHEMA)
    _items(5).to_parquet(root / devselect.DEV_ITEMS_FILE, index=False)

    frame, report = devselect.load_dev_set(root)
    assert report["available"] is False
    assert report["reason"] == "unauthenticated"
    assert "sha256_mismatch" in report["detail"]


def test_missing_measurements_cannot_make_a_tolerance_qualify():
    """A NaN median must never satisfy `<= 100 ms`."""
    evidence = pd.DataFrame([{
        "tolerance_ms": 200.0, "purpose": "dev", "instrument": "synthetic_dev",
        "accepted_spans": 0, "rejected_units": 400, "items_scored": 0,
        "min_boundaries": 50, "eligible": False, "n": 0,
        "median_abs_error_ms": float("nan"), "p90_abs_error_ms": float("nan"),
        "median_signed_error_ms": float("nan"), "within_100ms": float("nan"),
    }])
    record = devselect.select_operating_tolerance(evidence)
    assert record["selected_tolerance_ms"] is None
    assert np.isnan(float(evidence["median_abs_error_ms"].iloc[0]))
