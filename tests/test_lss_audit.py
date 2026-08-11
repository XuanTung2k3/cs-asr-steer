"""LSS unit test: the audit is blinded, powered, and immutable."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from csasr.lss.align.jitter import jitter_spans, jitter_stability, mask_iou
from csasr.lss.audit.pack import AuditPackSpec, build_pack, item_id, sample_audit_units
from csasr.lss.audit.verdicts import (
    adjudicate,
    agreement,
    boundary_error,
    ingest,
    unblind,
    usable_fraction,
)

SR = 16000


def _spans(n_en=12, n_zh=12):
    rows = []
    for i in range(n_en):
        rows.append({"utterance_id": f"u{i}", "unit_id": 1, "language": "EN",
                     "surface": "machine", "confidence_bin": "high" if i % 2 else "medium",
                     "consensus_start_sample": int(0.5 * SR),
                     "consensus_end_sample": int(0.9 * SR),
                     "n_families": 2, "sample_rate": SR, "safe_interior_ms": 200.0})
    for i in range(n_zh):
        rows.append({"utterance_id": f"z{i}", "unit_id": 2, "language": "ZH",
                     "surface": "我", "confidence_bin": "high" if i % 2 else "medium",
                     "consensus_start_sample": int(1.0 * SR),
                     "consensus_end_sample": int(1.14 * SR),
                     "n_families": 2, "sample_rate": SR, "safe_interior_ms": 60.0})
    return pd.DataFrame(rows)


def _manifest(spans):
    return pd.DataFrame({"utterance_id": spans["utterance_id"].unique(),
                         "audio_path": ["/nonexistent.wav"] * spans["utterance_id"].nunique()})


def test_sample_is_class_balanced():
    spec = AuditPackSpec(n_en=6, n_zh=6)
    sampled = sample_audit_units(_spans(), spec, seed=1)
    assert (sampled["language"] == "EN").sum() >= 4
    assert (sampled["language"] == "ZH").sum() >= 4


def test_sampling_is_deterministic():
    spec = AuditPackSpec(n_en=6, n_zh=6)
    a = sample_audit_units(_spans(), spec, seed=3)
    b = sample_audit_units(_spans(), spec, seed=3)
    assert list(a["utterance_id"]) == list(b["utterance_id"])


def test_item_ids_are_salted_and_stable():
    assert item_id("salt", "u1", 1) == item_id("salt", "u1", 1)
    assert item_id("salt", "u1", 1) != item_id("other", "u1", 1)
    assert len(item_id("salt", "u1", 1)) == 16


def test_pack_leaks_no_system_information_to_the_annotator(tmp_path):
    spec = AuditPackSpec(n_en=6, n_zh=6, perturbed_fraction=0.2, duplicate_fraction=0.2)
    spans = _spans()
    summary = build_pack(spans, _manifest(spans), tmp_path, spec,
                         seed=1, blinding_seed=2, perturb_seed=3, render_audio=False)
    items = [json.loads(line) for line in
             (tmp_path / "items.jsonl").read_text().splitlines()]
    forbidden = {"language", "confidence_bin", "n_families", "aligner_family",
                 "utterance_id", "unit_id", "is_decoy", "applied_offset_ms"}
    for item in items:
        assert not (forbidden & set(item)), item
    assert summary["decoys"] > 0
    assert summary["duplicates"] > 0
    # the mapping exists, but in a separate file that is not part of the pack
    key = json.loads((tmp_path / "blinding_key.json").read_text())
    assert {"utterance_id", "is_decoy", "applied_offset_ms"} <= set(key[0])


def test_guide_and_template_are_written(tmp_path):
    spec = AuditPackSpec(n_en=4, n_zh=4)
    spans = _spans()
    build_pack(spans, _manifest(spans), tmp_path, spec, seed=1, blinding_seed=2,
               perturb_seed=3, render_audio=False)
    guide = (tmp_path / "ANNOTATION_GUIDE.md").read_text()
    assert "corrected_start_sec" in guide and "adjudication" in guide
    template = pd.read_csv(tmp_path / "verdict_template.csv")
    assert "corrected_start_sec" in template.columns


def _verdict_rows(tmp_path, rows, annotator="a1"):
    raw = tmp_path / "verdicts_raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / f"{annotator}.csv", index=False)
    return raw


def test_ingest_copies_raw_files_immutably(tmp_path):
    raw = _verdict_rows(tmp_path, [{"audit_item_id": "abc", "verdict": "usable"}])
    table, report = ingest(raw, tmp_path)
    assert len(table) == 1
    copies = list((tmp_path / "verdicts_immutable").glob("*.csv"))
    assert len(copies) == 1 and report["files"][0]["sha256"]


def test_unknown_item_ids_are_rejected(tmp_path):
    (tmp_path / "blinding_key.json").write_text(json.dumps(
        [{"audit_item_id": "known", "utterance_id": "u1", "unit_id": 1}]))
    table = pd.DataFrame([{"audit_item_id": "ghost", "verdict": "usable"}])
    with pytest.raises(ValueError, match="unknown audit items"):
        unblind(table, tmp_path / "blinding_key.json")


def _unblinded(rows):
    return pd.DataFrame(rows)


def test_boundary_error_excludes_decoys_and_splits_by_language():
    rows = []
    for i in range(4):
        rows.append({"audit_item_id": f"e{i}", "annotator_id": "a1", "verdict": "usable",
                     "corrected_start_sec": 0.5, "corrected_end_sec": 0.9,
                     "true_start_sec": 0.55, "true_end_sec": 0.95, "clip_offset_sec": 0.0,
                     "applied_offset_ms": 0.0, "is_decoy": False, "is_duplicate": False,
                     "language": "EN", "utterance_id": f"u{i}", "unit_id": 1})
    rows.append({"audit_item_id": "d0", "annotator_id": "a1", "verdict": "boundary_off",
                 "corrected_start_sec": 0.5, "corrected_end_sec": 0.9,
                 "true_start_sec": 0.5, "true_end_sec": 0.9, "clip_offset_sec": 0.0,
                 "applied_offset_ms": 250.0, "is_decoy": True, "is_duplicate": False,
                 "language": "EN", "utterance_id": "u9", "unit_id": 1})
    report = boundary_error(_unblinded(rows))
    assert report["n"] == 4                      # the decoy is not scored
    assert report["median_abs_error_ms"] == pytest.approx(50.0, abs=1e-6)
    assert report["decoys_excluded"] == 1
    assert "EN" in report["by_language"]


def test_agreement_measures_decoy_detection():
    rows = [{"audit_item_id": f"d{i}", "annotator_id": "a1",
             "verdict": "boundary_off" if i < 8 else "usable",
             "is_decoy": True, "is_duplicate": False,
             "utterance_id": f"u{i}", "unit_id": 1, "corrected_start_sec": None}
            for i in range(10)]
    assert agreement(_unblinded(rows))["decoy_detection_rate"] == pytest.approx(0.8)


def test_alpha_is_one_when_annotators_agree_and_low_when_they_do_not():
    same = [{"audit_item_id": "x", "annotator_id": a, "verdict": "usable",
             "is_decoy": False, "is_duplicate": False, "utterance_id": "u1",
             "unit_id": 1, "corrected_start_sec": None} for a in ("a1", "a2")]
    assert agreement(_unblinded(same))["inter_annotator_alpha"] == pytest.approx(1.0)
    differ = [{"audit_item_id": "x", "annotator_id": "a1", "verdict": "usable",
               "is_decoy": False, "is_duplicate": False, "utterance_id": "u1",
               "unit_id": 1, "corrected_start_sec": None},
              {"audit_item_id": "x", "annotator_id": "a2", "verdict": "wrong_unit",
               "is_decoy": False, "is_duplicate": False, "utterance_id": "u1",
               "unit_id": 1, "corrected_start_sec": None}]
    assert agreement(_unblinded(differ))["inter_annotator_alpha"] < 0.5


def test_adjudication_flags_disagreement_without_touching_raw_rows():
    rows = [{"utterance_id": "u1", "unit_id": 1, "annotator_id": "a1",
             "verdict": "usable", "corrected_start_sec": 0.50},
            {"utterance_id": "u1", "unit_id": 1, "annotator_id": "a2",
             "verdict": "boundary_off", "corrected_start_sec": 0.90}]
    frame = _unblinded(rows)
    table = adjudicate(frame)
    assert bool(table.iloc[0]["needs_adjudication"]) is True
    assert table.iloc[0]["start_spread_ms"] == pytest.approx(400.0)
    assert len(frame) == 2      # unchanged


def test_usable_fraction_ignores_decoys():
    rows = [{"verdict": "usable", "is_decoy": False},
            {"verdict": "usable", "is_decoy": False},
            {"verdict": "boundary_off", "is_decoy": True}]
    assert usable_fraction(_unblinded(rows)) == pytest.approx(1.0)


def test_mask_iou_is_analytic():
    assert mask_iou((0.0, 400.0), (100.0, 500.0)) == pytest.approx(300 / 500)
    assert mask_iou((0.0, 100.0), (200.0, 300.0)) == 0.0


def test_jitter_shifts_both_edges_deterministically():
    spans = _spans(n_en=3, n_zh=0)
    a = jitter_spans(spans, offset_ms=100.0, seed=7)
    b = jitter_spans(spans, offset_ms=100.0, seed=7)
    assert list(a["jittered_start_sample"]) == list(b["jittered_start_sample"])
    delta = (a["jittered_start_sample"] - spans["consensus_start_sample"]).abs()
    assert (delta <= int(0.1 * SR)).all()


def test_jitter_stability_reports_every_offset():
    cfg = {"jitter": {"offsets_ms": [50, 100]}, "seeds": {"jitter": [1, 2]}}
    table, summary = jitter_stability(_spans(n_en=5, n_zh=5), cfg)
    assert set(table["offset_ms"]) == {50.0, 100.0}
    assert "offset_100ms" in summary
    assert 0.0 <= summary["offset_100ms"]["median_mask_iou"] <= 1.0
