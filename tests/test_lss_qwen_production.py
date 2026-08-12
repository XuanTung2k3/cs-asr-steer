"""Production Qwen alignment: the whole manifest, or nothing published.

These drive the **real** subprocess runner. `run_qwen_production` spawns a fresh
interpreter, so a `monkeypatch` in this process cannot reach the adapter it
loads; the child is pointed at a stub through `qwen_probe.ADAPTER_ENV_VAR`, the
same seam the probe tests use. That matters because everything worth testing here
is in the parent/child protocol -- the chunk files, the universe check, the
process-group kill, the atomic publish -- and a fake that replaced
`run_qwen_production` itself would exercise none of it.

The stub adapters below are pinned to the real `Qwen3ForcedAlignerAdapter.run`
signature by `test_the_stub_adapters_match_the_real_adapter_signature`. A
permissive `run(*a, **k)` would make every other test here vacuous: that is
exactly how a call missing the required `language` argument reached the cluster.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from csasr.lss.align import qwen_prod
from csasr.lss.align.qwen_probe import ADAPTER_ENV_VAR
from csasr.nat5h.schema import (
    ALIGNMENT_SCHEMA_VERSION,
    RunIdentity,
    artifact_matches_identity,
    assert_schema_v2,
)

SR = 16000


def _cfg(root: Path, model_dir: Path, **qwen) -> dict:
    return {
        "experiment": {"output_root": str(root)},
        "model": {"id": "fake", "device": "cpu"},
        "alignment": {"qwen": {"local_model_dir": str(model_dir), "model_id": "x",
                               "probe_language": "Chinese", **qwen}},
    }


def _manifest(n: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "utterance_id": f"u{i:04d}",
        "conversation_id": f"c{i // 5}",
        "audio_path": f"/nonexistent/u{i}.wav",
        "duration_sec": 3.0,
        "transcript_raw": "我 machine",
        "split": "train",
    } for i in range(n)])


def _rows(manifest: pd.DataFrame, identity: RunIdentity, language: str,
          *, skip: int = 0) -> pd.DataFrame:
    """Schema-v2 candidate rows for a chunk, stamped with the run identity.

    The identity stamp is the point: the probe's table carried L1a's
    `config_hash`, so L1b refused it and `candidates_all.parquet` had no Qwen rows
    at all. A production table has to be stamped by the run that consumes it.
    """
    rows = []
    for offset, (_, row) in enumerate(manifest.iterrows()):
        if offset < skip:
            continue
        for unit, (language_tag, start) in enumerate([("ZH", 0.0), ("EN", 0.6)]):
            rows.append({
                "schema_version": ALIGNMENT_SCHEMA_VERSION,
                "utterance_id": str(row["utterance_id"]),
                "conversation_id": str(row.get("conversation_id", "c")),
                "split": str(row.get("split", "train")),
                "reference_unit_index": unit,
                "reference_text": "我" if language_tag == "ZH" else "machine",
                "reference_language": language_tag,
                "aligner_family": "qwen_forced_aligner",
                "aligner_variant": f"qwen_forced_aligner/{language}",
                "start_sec": start, "end_sec": start + 0.5,
                "audio_duration_sec": 3.0,
                "start_sample": int(start * SR), "end_sample": int((start + 0.5) * SR),
                "sample_rate": SR, "waveform_num_samples": int(3.0 * SR),
                "is_valid": True, "failure_code": "", "failure_detail": "",
                "source_token_count": 1, "mapping_method": "qwen_sequential_surface",
                "model_id": "x", "model_revision": None,
                "code_commit": identity.code_commit,
                "config_hash": identity.config_hash,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stub adapters, importable by a fresh interpreter
# --------------------------------------------------------------------------
class StubAdapter:
    """Aligns every utterance it is given. Real `run` signature, on purpose."""

    languages = ["Chinese", "English"]
    skip_per_chunk = 0

    def __init__(self, cfg):
        self.cfg = cfg
        self.package_version = "stub-prod-0"
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def supported_languages(self):
        return list(self.languages)

    def run(self, manifest, geometry, language, identity, diagnostics_dir=None):
        assert self.loaded, "run() before load()"
        return _rows(manifest, identity, language, skip=self.skip_per_chunk)


class DropsUtterances(StubAdapter):
    """Returns rows for all but the first utterance of every chunk."""

    skip_per_chunk = 1


class HangsForever(StubAdapter):
    def run(self, manifest, geometry, language, identity, diagnostics_dir=None):
        while True:                                     # pragma: no cover - killed
            time.sleep(0.05)


class DeclaresOnlyJapanese(StubAdapter):
    """Module scope, not a local class: the child imports it by qualified name."""

    languages = ["Japanese"]


class FailsOnTheSecondChunk(StubAdapter):
    def run(self, manifest, geometry, language, identity, diagnostics_dir=None):
        marker = Path(os.environ["CSASR_TEST_CHUNK_MARKER"])
        seen = int(marker.read_text()) if marker.is_file() else 0
        marker.write_text(str(seen + 1))
        if seen >= 1:
            raise ValueError("chunk 1 is broken")
        return _rows(manifest, identity, language)


def _child_env(monkeypatch, adapter) -> None:
    monkeypatch.setenv(ADAPTER_ENV_VAR,
                       f"{adapter.__module__}:{adapter.__qualname__}")
    # a fresh interpreter needs this process's import paths, including the
    # directory this test module lives in
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(Path(__file__).resolve().parent)] + [p for p in sys.path if p]))


@pytest.fixture
def paths(tmp_path):
    model = tmp_path / "qwen"
    model.mkdir()
    out = tmp_path / "alignments"
    out.mkdir()
    return tmp_path, model, out


# --------------------------------------------------------------------------
# the contract that makes the rest meaningful
# --------------------------------------------------------------------------
def _contract(func):
    return [(p.name, p.kind, p.default is inspect.Parameter.empty)
            for p in inspect.signature(func).parameters.values()]


def test_the_stub_adapters_match_the_real_adapter_signature():
    from csasr.nat5h.aligners import Qwen3ForcedAlignerAdapter

    real = _contract(Qwen3ForcedAlignerAdapter.run)
    for adapter in (StubAdapter, DropsUtterances, HangsForever,
                    FailsOnTheSecondChunk):
        assert _contract(adapter.run) == real, (
            f"{adapter.__name__}.run no longer matches the real adapter; the "
            "production runner's call would not be tested")


def test_the_production_runner_calls_the_adapter_the_way_the_adapter_declares():
    """`run_qwen_production` must pass `language` positionally-or-by-name and a
    geometry and an identity. The probe once omitted `language` entirely."""
    source = inspect.getsource(qwen_prod._child_main)
    assert "language=language" in source
    assert "geometry=geometry" in source and "identity=identity" in source


# --------------------------------------------------------------------------
# success over a whole manifest
# --------------------------------------------------------------------------
def test_a_full_manifest_is_aligned_chunked_and_published(monkeypatch, paths):
    root, model, out = paths
    _child_env(monkeypatch, StubAdapter)
    cfg = _cfg(root, model)
    manifest = _manifest(25)

    result = qwen_prod.run_qwen_production(
        cfg, manifest, out_dir=out, purpose="natural", deadline_seconds=300,
        chunk_size=10)

    assert result.state == "ok", (result.reason, result.traceback)
    assert result.chunks_expected == 3 and result.chunks_written == 3
    assert result.expected_utterances == 25 and result.covered_utterances == 25
    assert result.universe_complete is True
    assert result.candidate_rows == 50
    assert result.package_version == "stub-prod-0"

    published = out / qwen_prod.CANDIDATES_FILE
    assert published.is_file()
    table = pd.read_parquet(published)
    assert_schema_v2(table, artifact="published qwen table")
    assert set(table["aligner_variant"]) == {"qwen_forced_aligner/Chinese"}, \
        "one frozen language variant, or two variants of one family double-vote"

    # authenticated, and identity-matching for the run that asked for it
    assert result.manifest["schema"] == "nat5h_candidates_v2"
    assert result.manifest["qwen_production"]["universe_complete"] is True
    assert artifact_matches_identity(table, RunIdentity.from_cfg(cfg))
    # nothing left behind
    assert not list(out.glob(".qwen_prod_*"))


def test_run_families_consumes_the_production_table(monkeypatch, paths):
    """The gap this closes: the sweep asked for Qwen, the probe had written eight
    corpus utterances under L1a's identity, and `candidates_all.parquet` came out
    with no Qwen rows at all."""
    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    root, model, out = paths
    _child_env(monkeypatch, StubAdapter)
    cfg = _cfg(root, model)
    manifest = _manifest(6)
    identity = RunIdentity.from_cfg(cfg)

    def runner(m, *, out_dir, purpose="natural"):
        return qwen_prod.run_qwen_production(
            cfg, m, out_dir=out_dir, purpose=purpose, deadline_seconds=300,
            chunk_size=4)

    table, report = cand.run_families(
        manifest, cfg, EncoderGeometry.from_values(), identity,
        ["qwen_forced_aligner"], out_dir=out, qwen_runner=runner)

    assert report["active"] == ["qwen_forced_aligner"]
    assert report["families"]["qwen_forced_aligner"]["source"] == "production_runner"
    assert set(table["utterance_id"]) == set(manifest["utterance_id"])


def test_without_a_runner_the_family_is_deferred_never_run_inline(paths):
    """A hang in the sweep's own process cannot be bounded, so the family is
    dropped rather than attempted."""
    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    root, model, out = paths
    cfg = _cfg(root, model)
    table, report = cand.run_families(
        _manifest(3), cfg, EncoderGeometry.from_values(),
        RunIdentity.from_cfg(cfg), ["qwen_forced_aligner"], out_dir=out)
    assert report["families"]["qwen_forced_aligner"]["state"] == "deferred_to_probe"
    assert not len(table)


# --------------------------------------------------------------------------
# the ways it must refuse to publish
# --------------------------------------------------------------------------
def test_an_incomplete_universe_is_discarded_not_published(monkeypatch, paths):
    """A family that aligned 1,400 of 1,800 utterances would score a perfect
    validity *rate* on what it produced, so the table must not exist at all."""
    root, model, out = paths
    _child_env(monkeypatch, DropsUtterances)

    result = qwen_prod.run_qwen_production(
        _cfg(root, model), _manifest(12), out_dir=out, deadline_seconds=300,
        chunk_size=4)

    assert result.state == "blocked"
    assert "incomplete_universe" in (result.reason or "")
    assert result.covered_utterances == 9 and result.expected_utterances == 12
    assert result.universe_complete is False
    assert not (out / qwen_prod.CANDIDATES_FILE).exists()
    assert result.quarantined, "the discarded chunks must be recorded"
    assert not list(out.glob(".qwen_prod_*"))


def test_a_failed_chunk_discards_the_chunks_that_did_finish(monkeypatch, paths):
    """Chunk 0 wrote a perfectly valid parquet. A prefix of the work is not the
    work, and a reusable prefix is the failure mode chunking exists to avoid."""
    root, model, out = paths
    _child_env(monkeypatch, FailsOnTheSecondChunk)
    monkeypatch.setenv("CSASR_TEST_CHUNK_MARKER", str(root / "chunks.txt"))

    result = qwen_prod.run_qwen_production(
        _cfg(root, model), _manifest(8), out_dir=out, deadline_seconds=300,
        chunk_size=4)

    # a ValueError from the adapter is a defect, not a missing aligner
    assert result.state == "failed"
    assert "chunk 1 is broken" in (result.reason or "")
    assert not (out / qwen_prod.CANDIDATES_FILE).exists()
    assert any("chunk_00000" in path for path in result.quarantined), \
        "the chunk that did finish must be named in what was discarded"
    assert not list(out.glob(".qwen_prod_*"))


def test_a_timeout_kills_the_process_group_and_publishes_nothing(monkeypatch, paths):
    root, model, out = paths
    _child_env(monkeypatch, HangsForever)

    started = time.monotonic()
    result = qwen_prod.run_qwen_production(
        _cfg(root, model), _manifest(4), out_dir=out, deadline_seconds=3,
        chunk_size=2)
    elapsed = time.monotonic() - started

    assert result.state == "blocked", result.reason
    assert result.reason == "deadline_exceeded"
    assert result.timed_out is True
    # `blocked`, never `completed_no_go`: a timeout measured nothing
    assert result.state != "completed_no_go"
    assert elapsed < 60, "the deadline did not bound the run"
    assert result.process_group.get("pgid"), \
        "the whole process group must be signalled, not just the child"
    assert result.process_group.get("terminated") or result.process_group.get("killed")
    assert not (out / qwen_prod.CANDIDATES_FILE).exists()
    assert not list(out.glob(".qwen_prod_*"))


def test_a_missing_checkpoint_is_blocked_before_anything_is_spawned(paths):
    root, model, out = paths
    cfg = _cfg(root, model)
    cfg["alignment"]["qwen"]["local_model_dir"] = str(root / "absent")
    result = qwen_prod.run_qwen_production(cfg, _manifest(2), out_dir=out)
    assert result.state == "blocked" and result.reason == "checkpoint_missing"
    assert not list(out.iterdir())


def test_a_language_the_checkpoint_does_not_declare_is_refused(monkeypatch, paths):
    root, model, out = paths
    _child_env(monkeypatch, DeclaresOnlyJapanese)
    result = qwen_prod.run_qwen_production(
        _cfg(root, model), _manifest(2), out_dir=out, deadline_seconds=120)
    assert result.state == "blocked"
    assert "language_unsupported" in (result.reason or "")
    assert not (out / qwen_prod.CANDIDATES_FILE).exists()


# --------------------------------------------------------------------------
# the deadline is its own, not the probe's
# --------------------------------------------------------------------------
def test_the_production_deadline_is_not_the_probe_deadline():
    """1,800 utterances is 225x the probe's eight; inheriting a bound set for
    eight is how a run gets killed for being the size it was asked to be."""
    cfg = {"alignment": {"qwen": {"probe_deadline_minutes": 30,
                                  "production_deadline_minutes": 90,
                                  "production_seconds_per_utterance": 4.0}}}
    # the absolute floor holds for a small manifest
    assert qwen_prod.production_deadline_seconds(cfg, 8) == 90 * 60
    # and it scales with the work once the per-utterance allowance dominates
    assert qwen_prod.production_deadline_seconds(cfg, 1800) == 7200.0
    # the probe's own bound never enters
    bare = {"alignment": {"qwen": {"probe_deadline_minutes": 1}}}
    assert qwen_prod.production_deadline_seconds(bare, 1800) == 90 * 60


def test_chunk_bounds_cover_every_row_exactly_once():
    bounds = qwen_prod.chunk_bounds(25, 10)
    assert bounds == [(0, 10), (10, 20), (20, 25)]
    covered = [i for lo, hi in bounds for i in range(lo, hi)]
    assert covered == list(range(25))
    assert qwen_prod.chunk_bounds(0, 10) == []


def test_an_empty_manifest_is_blocked_not_a_published_empty_table(paths):
    root, model, out = paths
    result = qwen_prod.run_qwen_production(_cfg(root, model), pd.DataFrame(),
                                           out_dir=out)
    assert result.state == "blocked" and result.reason == "empty_manifest"
    assert not (out / qwen_prod.CANDIDATES_FILE).exists()


def test_the_probe_table_is_not_reused_as_production_evidence(monkeypatch, paths):
    """A 147-row probe table under a different identity must not be consumed.

    The cache is only reused when it matches this run's identity; the probe's does
    not, so the production runner decides, and it either covers the universe or
    publishes nothing.
    """
    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    root, model, out = paths
    cfg = _cfg(root, model)
    identity = RunIdentity.from_cfg(cfg)
    other = RunIdentity(code_commit="unknown", config_hash="a-different-config")
    _rows(_manifest(2), other, "Chinese").to_parquet(
        out / qwen_prod.CANDIDATES_FILE, index=False)

    _child_env(monkeypatch, DropsUtterances)          # cannot cover the universe

    def runner(m, *, out_dir, purpose="natural"):
        return qwen_prod.run_qwen_production(
            cfg, m, out_dir=out_dir, purpose=purpose, deadline_seconds=300,
            chunk_size=4)

    table, report = cand.run_families(
        _manifest(8), cfg, EncoderGeometry.from_values(), identity,
        ["qwen_forced_aligner"], out_dir=out, qwen_runner=runner)

    assert not len(table), "the mismatched probe table must not be consumed"
    record = report["families"]["qwen_forced_aligner"]
    assert record["state"] == "blocked"
    assert "incomplete_universe" in record["error"]


def test_qwen_cache_is_bound_to_the_exact_requested_manifest(paths):
    """The same config and IDs with changed inputs are a different request."""
    from csasr.lss import manifest as manifest_mod
    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    root, model, out = paths
    cfg = _cfg(root, model)
    identity = RunIdentity.from_cfg(cfg)
    old_request = _manifest(3)
    cache = out / qwen_prod.CANDIDATES_FILE
    old_table = _rows(old_request, identity, "Chinese")
    manifest_mod.publish_frame(
        cache, old_table, stage="l1b_valid", cfg=cfg,
        key_columns=("utterance_id", "reference_unit_index",
                     "aligner_family", "aligner_variant"),
        schema="nat5h_candidates_v2",
        extra={"request_manifest": cand.request_manifest_record(old_request)})

    changed_request = old_request.copy()
    changed_request.loc[0, "transcript_raw"] = "different transcript"
    called = []

    def runner(manifest, *, out_dir):
        called.append(manifest.copy())
        return {"state": "blocked", "reason": "deliberate test stop"}

    table, report = cand.run_families(
        changed_request, cfg, EncoderGeometry.from_values(), identity,
        ["qwen_forced_aligner"], out_dir=out, qwen_runner=runner)

    assert called, "a same-identity cache for different input must not be reused"
    assert not len(table)
    assert report["families"]["qwen_forced_aligner"]["state"] == "blocked"


def test_non_qwen_cache_is_bound_to_the_exact_requested_manifest(
        monkeypatch, paths):
    from csasr.lss import manifest as manifest_mod
    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    root, model, out = paths
    cfg = _cfg(root, model)
    identity = RunIdentity.from_cfg(cfg)
    old_request = _manifest(3)
    old_table = _rows(old_request, identity, "Chinese").assign(
        aligner_family="existing_ctc", aligner_variant="existing_ctc/test")
    cache = out / "candidates_existing_ctc.parquet"
    manifest_mod.publish_frame(
        cache, old_table, stage="l1b_valid", cfg=cfg,
        key_columns=("utterance_id", "reference_unit_index",
                     "aligner_family", "aligner_variant"),
        schema="nat5h_candidates_v2",
        extra={"request_manifest": cand.request_manifest_record(old_request)})

    changed_request = old_request.copy()
    changed_request.loc[0, "audio_path"] = "/different/audio.wav"
    rebuilt = _rows(changed_request, identity, "Chinese").assign(
        aligner_family="existing_ctc", aligner_variant="existing_ctc/rebuilt")
    calls = []

    def fake_ctc(manifest, cfg, geometry, identity):
        calls.append(manifest.copy())
        return rebuilt

    monkeypatch.setattr("csasr.nat5h.aligners.run_existing_ctc", fake_ctc)
    table, report = cand.run_families(
        changed_request, cfg, EncoderGeometry.from_values(), identity,
        ["existing_ctc"], out_dir=out)

    assert calls, "the exact requested inputs, not config identity, control reuse"
    assert set(table["aligner_variant"]) == {"existing_ctc/rebuilt"}
    assert report["families"]["existing_ctc"]["source"] == "rebuilt"
    published = manifest_mod.load(cache)
    assert published["request_manifest"] == cand.request_manifest_record(
        changed_request)


def test_the_result_json_round_trips_every_field_the_parent_reports(monkeypatch,
                                                                   paths):
    """The child writes JSON and the parent rebuilds a dataclass from it; a field
    the child cannot express is a field that silently reads as its default."""
    root, model, out = paths
    _child_env(monkeypatch, StubAdapter)
    result = qwen_prod.run_qwen_production(
        _cfg(root, model), _manifest(3), out_dir=out, deadline_seconds=300,
        chunk_size=2)
    payload = json.loads(json.dumps(result.to_dict(), default=str))
    for field in ("state", "purpose", "language", "expected_utterances",
                  "covered_utterances", "universe_complete", "chunks_expected",
                  "chunks_written", "chunk_size", "deadline_seconds",
                  "elapsed_seconds", "published", "manifest"):
        assert field in payload, field
    assert payload["language"] == "Chinese"
