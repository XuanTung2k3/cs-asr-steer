"""End-to-end Gate-A runs: every way a false pass could happen, closed.

These call `lss_l1b_valid.main()` against a temporary artifacts root with mocked
aligner output, so they exercise the production assembly -- which criteria are
constructed, which blockers fire, what status and exit code come out, and
whether `l1c` is unlocked afterwards -- rather than the helpers underneath it.

The helpers are tested in `test_lss_gate_a_auto.py`; a helper that behaves
correctly while production never calls it is exactly the gap this file covers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from csasr.experiments import lss_l1b_valid
from csasr.lss import manifest as manifest_mod
from csasr.lss.align import autoevidence
from csasr.lss.prereq import PrerequisiteError, require_prerequisites
from csasr.nat5h.schema import ALIGNMENT_SCHEMA_VERSION
from csasr.utils.status import read_status, write_status

SR = 16000

#: The fixture's transcript must segment into exactly the units the fixture
#: mocks candidates for, or "full coverage" in these tests is not full coverage.
#: `unit_table("我 machine 用 learning")` gives ZH, EN, ZH, EN -- the same four
#: units, in the same order, as `_candidate_rows` emits. It previously read
#: "我 用 machine learning 做 这个", which segments into *seven* units, so a
#: fixture claiming every unit was covered actually covered four of seven and the
#: per-family coverage floor could not be tested at all.
TRANSCRIPT = "我 machine 用 learning"
UNITS_PER_UTT = 4

#: Roles the fixture publishes candidates for. `roles_to_label` in the real
#: config names six; the coverage denominator spans all of them, so a fixture
#: that aligns one role would be judged against the others' units too.
LABELLED_ROLES = ["D-construct"]


# --------------------------------------------------------------------------
# a minimal but complete artifacts root
# --------------------------------------------------------------------------
def _manifest(n_utterances: int, role: str = "D-construct") -> pd.DataFrame:
    return pd.DataFrame([{
        "utterance_id": f"{role}_u{i}",
        "conversation_id": f"c{i // 5}",
        "speaker_id": f"c{i // 5}",
        "dialogue_id": f"d{i // 5}",
        "audio_path": f"/nonexistent/{role}_u{i}.wav",
        "duration_sec": 4.0,
        "transcript_raw": TRANSCRIPT,
        "contains_code_switch": True,
        "role": role,
    } for i in range(n_utterances)])


def _candidate_rows(manifest: pd.DataFrame, families, *, offset_ms=0.0,
                    invalid_family=None, drop_utterances=0, languages=("ZH", "EN")):
    rows = []
    keep = manifest.iloc[drop_utterances:]
    for family in families:
        for _, utt in keep.iterrows():
            for unit in range(UNITS_PER_UTT):
                shift = offset_ms / 1000.0 if family == "whisper_dtw" else 0.0
                start = unit * 0.5 + shift
                rows.append({
                    "schema_version": ALIGNMENT_SCHEMA_VERSION,
                    "utterance_id": utt["utterance_id"],
                    "conversation_id": utt["conversation_id"],
                    "split": "train",
                    "role": utt["role"],
                    "reference_unit_index": unit,
                    "reference_language": languages[unit % len(languages)],
                    "reference_text": "machine" if unit % 2 else "我",
                    "aligner_family": family,
                    "aligner_variant": "default",
                    "start_sec": float(start),
                    "end_sec": float(start + 0.4),
                    "audio_duration_sec": 4.0,
                    "start_sample": int(start * SR),
                    "end_sample": int((start + 0.4) * SR),
                    "sample_rate": SR,
                    "waveform_num_samples": int(4.0 * SR),
                    "coordinate_policy": "short_wav_no_crop_v1",
                    "is_valid": family != invalid_family,
                    "failure_code": "" if family != invalid_family else "no_alignment",
                    "failure_detail": "",
                    "source_token_count": 1,
                    "mapping_method": "test",
                    "model_id": "fake", "model_revision": "0",
                    "code_commit": "unknown", "config_hash": "test",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """An artifacts root with L0 and L1a passed and their artifacts published."""
    artifacts = tmp_path / "artifacts_lss"
    (artifacts / "status").mkdir(parents=True, exist_ok=True)
    # The production amendment is an immutable repository document, not test
    # infrastructure. Keep these integration tests self-contained so an
    # unrelated documentation move/deletion cannot make every Gate-A assertion
    # abort before reaching the behavior it is meant to exercise.
    (tmp_path / "scientific_amendment.md").write_text(
        "# Test-only scientific amendment fixture\n", encoding="utf-8")
    cfg_stub = {"experiment": {"output_root": str(artifacts)}, "model": {"id": "fake"}}

    write_status(artifacts, "l0_freeze", "passed", complete=True, full_l0_pass=True)
    write_status(artifacts, "l1a_diag", "passed", complete=True, full_stage_pass=True)

    freeze = artifacts / "freeze" / "spec_freeze_v1.json"
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps({"sha256": "x"}), encoding="utf-8")
    manifest_mod.publish(freeze, None, stage="l0_freeze", cfg=cfg_stub)

    roles_dir = artifacts / "manifests" / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    for role in ("D-construct", "loc-train", "D-dev-select", "D-dev-confirm"):
        manifest_mod.publish_frame(roles_dir / f"role_{role}.parquet",
                                   _manifest(30, role), stage="l0_freeze",
                                   cfg=cfg_stub, key_columns=("utterance_id",))

    summary = artifacts / "metrics" / "l1a_family_language_summary.parquet"
    summary.parent.mkdir(parents=True, exist_ok=True)
    manifest_mod.publish_frame(summary, pd.DataFrame([{"aligner_family": "existing_ctc",
                                                       "valid_units": 100}]),
                               stage="l1a_diag", cfg=cfg_stub)
    return artifacts


def _OVERRIDES(root: Path) -> list[str]:
    """Config overrides shared by the runner and the identity of what it reads."""
    return [f"experiment.output_root={root}",
            "gate_a.automatic_instrument_amendment.document="
            f"{root.parent / 'scientific_amendment.md'}",
            f"roles_to_label={json.dumps(LABELLED_ROLES)}"]


def _resolved_cfg(root: Path) -> dict:
    """The exact config L1b will load, so the identity check is a real test.

    Publishing with a hand-made stub would always mismatch, which would make
    every one of these tests pass for the wrong reason.
    """
    from csasr.utils.config import load_config

    # the same overrides `_run` passes, or the identity check fails on
    # `config_sha256` and every test blocks on unauthenticated candidates
    return load_config("lss/l1b_valid.yaml", _OVERRIDES(root))


def _publish_candidates(root: Path, frame: pd.DataFrame, *, cfg=None,
                        with_manifest: bool = True, taint_reasons=()):
    path = root / "alignments" / "candidates_all.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_manifest:
        manifest_mod.publish_frame(
            path, frame, stage="l1b_valid", cfg=cfg or _resolved_cfg(root),
            taint_reasons=taint_reasons,
            key_columns=lss_l1b_valid.CANDIDATE_KEYS)
    else:
        frame.to_parquet(path, index=False)
    return path


def _argv(root: Path, *extra: str) -> list[str]:
    """Command line for L1b against this root.

    Every call site goes through here. A hand-built argv that omitted one of
    `_OVERRIDES` resolved a different config than `_publish_candidates` recorded,
    and the run then blocked on `identity_mismatch: config_sha256` -- passing the
    test for a reason unrelated to what it claims to check.
    """
    argv = ["--config", "lss/l1b_valid.yaml"]
    for override in _OVERRIDES(root):
        argv += ["--set", override]
    return argv + list(extra)


DEFAULT_PARTS = "consensus,evidence,jitter,coverage,gate"


def _run(root: Path, *extra: str, only: str = DEFAULT_PARTS) -> tuple[int, dict]:
    """Run L1b's real entry point against this root, and read what it wrote.

    `only` names the parts; passing a second `--only` in `extra` would rely on
    argparse's last-wins, which reads as a bug at the call site.
    """
    code = lss_l1b_valid.main(_argv(root, "--only", only, *extra))
    return code, read_status(root, "l1b_valid")


def _assert_not_a_pass(status: dict, code: int):
    assert status["status"] != "passed", status.get("blocked_reasons")
    assert (status.get("gate") or {}).get("passed") is not True
    assert code in (0, 2)


def _assert_l1c_locked(root: Path):
    with pytest.raises(PrerequisiteError):
        require_prerequisites(root, "l1c_labels")


# --------------------------------------------------------------------------
# synthetic exact-boundary scoring, through the production assembly
# --------------------------------------------------------------------------
def _fake_rendered(purpose: str, n: int, *, offset_sec: float = 0.0):
    """A rendered synthetic set and the candidates a family would produce for it.

    `build_set` needs real source audio and the aligners need a GPU, so both are
    substituted. Everything between them -- the manifest, the seam mapping, the
    score table, publishing it, and Gate A reading it back -- is the real code.
    """
    items, candidates = [], []
    for i in range(n):
        pair_id = f"{purpose}_syn_{i:04d}"
        zh_end = 2.0
        items.append({
            "pair_id": pair_id, "utterance_id": pair_id,
            "audio_path": f"/nonexistent/{pair_id}.wav",
            "zh_text": "我 用", "en_text": "machine learning",
            "duration_sec": 4.0, "purpose": purpose,
            "zh_end_sec": zh_end, "en_start_sec": zh_end,
            "true_boundary_sec": zh_end,
            "reference_kind": "constructed_exact_lexical",
            "reference_semantics": "test_fixture_exact_lexical_edges",
            # source ids namespaced per purpose, so the disjointness check is real
            "zh_utterance_id": f"{purpose}_zh_{i}",
            "en_utterance_id": f"{purpose}_en_{i}",
        })
        for family in ("existing_ctc", "whisper_dtw"):
            shift = offset_sec if family == "whisper_dtw" else 0.0
            for unit, (language, start, end) in enumerate([
                    ("ZH", 0.2, 1.0), ("ZH", 1.0, zh_end + shift),
                    ("EN", zh_end + shift, zh_end + 0.6),
                    ("EN", zh_end + 0.6, zh_end + 1.2)]):
                candidates.append({
                    "schema_version": ALIGNMENT_SCHEMA_VERSION,
                    "utterance_id": pair_id, "conversation_id": "synthetic",
                    "split": f"synthetic_{purpose}",
                    "reference_unit_index": unit,
                    "reference_language": language, "reference_text": "x",
                    "aligner_family": family,
                    "aligner_variant": f"{family}/default",
                    "start_sec": float(start), "end_sec": float(end),
                    "audio_duration_sec": 4.0,
                    "start_sample": int(start * SR), "end_sample": int(end * SR),
                    "sample_rate": SR, "waveform_num_samples": int(4.0 * SR),
                    "coordinate_policy": "short_wav_no_crop_v1",
                    "is_valid": True, "failure_code": "", "failure_detail": "",
                    "source_token_count": 1, "mapping_method": "test",
                    "model_id": "fake", "model_revision": "0",
                    "code_commit": "unknown", "config_hash": "test",
                })
    return pd.DataFrame(items), pd.DataFrame(candidates)


@pytest.fixture
def synthetic_stubs(root, monkeypatch):
    """Substitute only the GPU and the source audio, never the scoring.

    The development set now comes from L1a, published under
    `devselect.DEV_ITEMS_FILE`, so this fixture publishes it the way L1a would
    rather than letting L1b render it. That hand-off is the thing under test in
    `test_the_development_set_comes_from_l1a_...`; here it is a precondition.
    """
    from csasr.lss.align import devselect
    from csasr.nat5h.coordinates import EncoderGeometry

    sets = {"dev": _fake_rendered("dev", 120),
            "gate": _fake_rendered("gate", 120, offset_sec=-0.030)}
    manifest_mod.publish_frame(
        root / devselect.DEV_ITEMS_FILE, sets["dev"][0], stage="l1a_diag",
        cfg={"experiment": {"output_root": str(root)}, "model": {"id": "fake"}},
        key_columns=("pair_id",), schema=devselect.DEV_ITEMS_SCHEMA)

    def fake_build_set(manifest, cfg, *, n_pairs, seed, out_dir, purpose,
                       sources=None, generation=None):
        frame = sets[purpose][0]
        return frame, {"purpose": purpose, "pairs": int(len(frame)),
                       "generation": generation,
                       "set_label": f"{purpose}_g{generation}" if generation
                       else purpose}

    def fake_run_families(manifest, cfg, geometry, identity, families, **kwargs):
        purpose = str(manifest["split"].iloc[0]).replace("synthetic_", "")
        table = sets[purpose][1]
        table = table[table["aligner_family"].isin(list(families))]
        return table.reset_index(drop=True), {"families": {
            f: {"state": "ok"} for f in families}}

    monkeypatch.setattr("csasr.lss.align.synthetic.build_set", fake_build_set)
    monkeypatch.setattr("csasr.lss.align.candidates.run_families",
                        fake_run_families)
    monkeypatch.setattr("csasr.models.whisper.load_whisper",
                        lambda cfg: object())
    monkeypatch.setattr(EncoderGeometry, "from_bundle",
                        classmethod(lambda cls, bundle: cls.from_values()))
    return sets


def test_synthetic_scoring_turns_known_boundaries_into_measured_error(
        root, synthetic_stubs):
    """The gap that made automatic Gate A block: rendering was not scoring.

    The gate set's families are placed 30 ms early by construction, so the
    published table must say 30 ms -- and Gate A must read it rather than
    reporting missing calibration.
    """
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    scores = pd.read_parquet(root / autoevidence.SYNTHETIC_SCORES_FILE)
    gate_rows = scores[scores["purpose"] == "gate"]
    assert len(gate_rows), "the gate set was not scored"
    assert set(gate_rows["family"]) == {"existing_ctc", "whisper_dtw"}
    # only the convention the pipeline consumes reaches the gate's table
    assert set(scores["convention"]) == {"unit_edge_canonical"}
    dtw = gate_rows[gate_rows["family"] == "whisper_dtw"]
    assert dtw["median_abs_error_ms"].max() == pytest.approx(30.0, abs=1e-3)

    names = {c["name"]: c for c in status["gate"]["criteria"]}
    assert "synthetic_boundaries_scored" in names
    assert names["synthetic_boundaries_scored"]["value"] == 120
    assert names["synthetic_absolute_error_within_100ms"]["passed"] is True
    # the block that motivated all of this is gone
    assert autoevidence.BLOCKED_MISSING_SYNTHETIC not in status["blocked_reasons"]
    assert code == 0


def test_the_operating_tolerance_is_selected_from_development_evidence(
        root, synthetic_stubs):
    """The automatic path no longer needs annotation to choose its tolerance.

    `spec.yaml` states the rule on the absolute boundary error of the spans a
    tolerance accepts and names the human audit as the instrument. The estimand is
    unchanged and so are the numbers -- 100 ms and 200 ms; only the instrument is
    the one the automatic path has, and the record says which one it was.
    """
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    point = json.loads((root / "freeze" / "l1b_operating_point.json").read_text())
    assert point["selected_tolerance_ms"] == 300.0        # dev error is 0 ms here
    assert point["instrument"] == "synthetic_dev"
    assert point["thresholds"] == {"max_median_abs_error_ms": 100.0,
                                   "max_p90_abs_error_ms": 200.0}
    # every configuration choice the gate rests on, in one artifact
    assert point["seeds"]["synthetic_dev"] and point["seeds"]["synthetic_gate"]
    assert point["development_artifact"]["sha256"]
    assert "aligner_selection" in point
    assert point["consensus"]["measurement_instrument"] == "synthetic_dev"
    amendment = point["selection"]["scientific_amendment"]
    assert amendment["id"] == \
        "gate-a-automatic-instrument-amendment-2026-08-11"
    # The repository's real RMS/VAD development set is only an audio seam, so
    # the amendment stays pending even though this exact-lexical fixture is
    # sufficient to exercise the future supported path.
    assert amendment["status"] == \
        "superseded_pending_genuine_lexical_reference"
    assert len(amendment["sha256"]) == 64
    assert "not an unchanged execution" in amendment["disclosure"]

    # neither blocker survives, and the reported instrument is not the human one
    assert autoevidence.BLOCKED_UNSELECTED_TOLERANCE not in status["blocked_reasons"]
    assert autoevidence.BLOCKED_NO_QUALIFYING_TOLERANCE not in status["blocked_reasons"]
    names = {c["name"]: c["value"] for c in status["gate"]["criteria"]}
    assert names["consensus_tolerance_ms"] == 300.0
    assert names["consensus_tolerance_instrument"] == "synthetic_dev"

    evidence = pd.read_parquet(root / "metrics" / "l1b_tolerance_selection.parquet")
    assert set(evidence["purpose"]) == {"dev"}, \
        "the gate set must never appear in the evidence a selection is made on"


def test_the_selection_never_sees_a_gate_row(root, synthetic_stubs):
    """The leakage guard, at the function that would have to be bypassed."""
    from csasr.lss.align import devselect

    _, gate_candidates = synthetic_stubs["gate"]
    gate_items = synthetic_stubs["gate"][0].assign(purpose="gate")
    with pytest.raises(devselect.DevelopmentOnlyError):
        devselect.tolerance_accuracy(
            gate_items, gate_candidates,
            devselect.ConsensusConfig.from_cfg({"min_families": 2}), [50, 100])


def test_no_qualifying_tolerance_blocks_rather_than_defaulting_to_200ms(
        root, monkeypatch):
    """A development set no tolerance can satisfy must not silently fall back.

    With no selected operating point every number downstream describes a
    configuration nobody chose, so this is `blocked` and carries its own code --
    distinct from "the selection never ran".
    """
    from csasr.lss.align import devselect
    from csasr.nat5h.coordinates import EncoderGeometry

    # both families 500 ms out on the development seam: nothing meets 100/200 ms
    sets = {"dev": _fake_rendered("dev", 120, offset_sec=-0.5),
            "gate": _fake_rendered("gate", 120, offset_sec=-0.5)}
    manifest_mod.publish_frame(
        root / devselect.DEV_ITEMS_FILE, sets["dev"][0], stage="l1a_diag",
        cfg={"experiment": {"output_root": str(root)}, "model": {"id": "fake"}},
        key_columns=("pair_id",), schema=devselect.DEV_ITEMS_SCHEMA)
    monkeypatch.setattr(
        "csasr.lss.align.synthetic.build_set",
        lambda manifest, cfg, **kw: (sets[kw["purpose"]][0],
                                     {"purpose": kw["purpose"], "pairs": 120}))
    monkeypatch.setattr(
        "csasr.lss.align.candidates.run_families",
        lambda manifest, cfg, geometry, identity, families, **kw: (
            sets[str(manifest["split"].iloc[0]).replace("synthetic_", "")][1],
            {"families": {f: {"state": "ok"} for f in families}}))
    monkeypatch.setattr("csasr.models.whisper.load_whisper", lambda cfg: object())
    monkeypatch.setattr(EncoderGeometry, "from_bundle",
                        classmethod(lambda cls, bundle: cls.from_values()))

    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    assert autoevidence.BLOCKED_NO_QUALIFYING_TOLERANCE in status["blocked_reasons"]
    assert autoevidence.BLOCKED_UNSELECTED_TOLERANCE not in status["blocked_reasons"]
    point = json.loads((root / "freeze" / "l1b_operating_point.json").read_text())
    assert point["selected_tolerance_ms"] is None
    assert point["selection"]["rejection_reasons"], \
        "the record must say why each tolerance failed"
    # and it says so as a repair action, not as "go and annotate"
    actions = status["next_actions"]
    assert any("repair alignment accuracy" in a["action"] for a in actions)
    _assert_not_a_pass(status, code)
    _assert_l1c_locked(root)


def test_the_selection_is_blocked_not_defaulted_when_it_never_ran(
        root, synthetic_stubs):
    """`--evaluate-gate` without the `synthetic` part has no frozen operating
    point, which is a different gap from "nothing qualified" and says so."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root)                       # DEFAULT_PARTS: no `synthetic`

    assert autoevidence.BLOCKED_UNSELECTED_TOLERANCE in status["blocked_reasons"]
    assert autoevidence.BLOCKED_NO_QUALIFYING_TOLERANCE not in status["blocked_reasons"]
    assert not (root / "freeze" / "l1b_operating_point.json").exists()


def test_the_published_score_table_is_authenticated(root, synthetic_stubs):
    """Gate A refuses an unmanifested table, so writing one without a manifest
    would leave the block in place while the file existed."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)

    from csasr.lss import manifest as mm

    verdict = mm.verify(root / autoevidence.SYNTHETIC_SCORES_FILE,
                        cfg=_resolved_cfg(root), require_identity=True)
    assert verdict["ok"], verdict
    assert verdict["manifest"]["schema"] == autoevidence.SYNTHETIC_SCORES_SCHEMA
    amendment = verdict["manifest"]["scientific_amendment"]
    assert amendment["id"] == \
        "gate-a-automatic-instrument-amendment-2026-08-11"
    assert len(amendment["sha256"]) == 64


def test_the_convention_comparison_is_written_for_a_reviewer(root, synthetic_stubs):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)

    comparison = pd.read_parquet(
        root / "metrics" / "l1b_synthetic_convention_comparison.parquet")
    assert set(comparison["convention"]) == {"unit_edge_canonical",
                                            "midpoint_legacy"}
    assert {"reference_kind", "metric_semantics",
            "median_signed_offset_ms", "median_absolute_offset_ms"} \
        <= set(comparison.columns)
    per_item = pd.read_parquet(
        root / autoevidence.SYNTHETIC_ITEMS_FILE)
    assert set(per_item["purpose"]) == {"dev", "gate"}
    assert per_item["scorable"].all()
    assert set(per_item["reference_kind"]) == {"constructed_exact_lexical"}
    assert {"signed_start_error_ms", "signed_end_error_ms",
            "absolute_boundary_error_ms"} <= set(per_item.columns)


# --------------------------------------------------------------------------
# the coverage denominator must be the set the stage actually aligns
# --------------------------------------------------------------------------
def test_the_swept_sample_and_the_coverage_denominator_are_the_same_set(root):
    """The two were written twice and drifted.

    The sweep and denominator once used different universes: a sampled numerator
    was compared with every unit in the full role. They must now share the exact
    dialogue-stratified sample, or coverage can fail on bookkeeping rather than
    an aligner's behavior.
    """
    from csasr.utils.config import load_config

    cfg = load_config("lss/l1b_valid.yaml", _OVERRIDES(root))
    sample = lss_l1b_valid.sweep_sample(cfg, LABELLED_ROLES)
    universe = lss_l1b_valid._expected_unit_universe(cfg, LABELLED_ROLES)

    assert len(sample) == 30
    assert set(universe["utterance_id"]) == set(sample["utterance_id"]), \
        "coverage is being measured against utterances the sweep never aligns"
    assert len(universe) == 30 * UNITS_PER_UTT
    # and only units a family actually emits candidates for
    assert set(universe["language"]) <= {"EN", "ZH"}


def test_full_coverage_of_the_swept_sample_reads_as_full_coverage(root):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root)
    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    coverage = evidence["unit_coverage"]
    assert coverage["denominator"] == "expected_universe"
    assert coverage["coverage"] == pytest.approx(1.0)
    assert coverage["missing_units"] == 0


def test_a_probe_sized_family_cannot_count_as_an_independent_aligner(root):
    """The Qwen probe writes `probe_utterances` (8) of the swept 300.

    `valid_unit_coverage` is a rate over the rows a family produced, so those 8
    utterances score 1.0 and used to qualify as a full independent aligner --
    enough to clear `blocked_insufficient_independent_aligners` on 3% of the
    units. A family must now also cover the universe coverage is measured
    against.
    """
    manifest = _manifest(30)
    full = _candidate_rows(manifest, ["existing_ctc", "whisper_dtw"])
    probe = _candidate_rows(manifest.head(2), ["qwen_forced_aligner"])
    _publish_candidates(root, pd.concat([full, probe], ignore_index=True))

    code, status = _run(root)
    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    independence = evidence["independence"]

    assert "qwen_forced_aligner" not in independence["qualifying_families"]
    assert "qwen_forced_aligner" in independence["rejected_families"]
    reason = " ".join(independence["rejection_reasons"]["qwen_forced_aligner"])
    assert "valid_units" in reason
    # Target coverage is measured against the full expected target universe;
    # a perfect candidate-relative rate can no longer appear here.
    validity = {row["aligner_family"]: row for row in evidence["family_validity"]}
    assert validity["qwen_forced_aligner"]["valid_unit_coverage"] == \
        pytest.approx(2 / 30)
    # the two real families are unaffected
    assert independence["n_independent_valid_families"] == 2
    _assert_not_a_pass(status, code)


# --------------------------------------------------------------------------
# the false-pass paths
# --------------------------------------------------------------------------
def test_two_qualifying_aligners_with_insufficient_paired_overlap_cannot_pass(root):
    """Both families clear 95% target coverage but their paired rate misses the
    explicitly tightened test requirement. This is the paired-evidence blocker,
    unlike the one-qualifying-family case, and it never claims disjoint inputs."""
    manifest = _manifest(30)
    ctc = _candidate_rows(manifest.iloc[:-1], ["existing_ctc"])
    dtw = _candidate_rows(manifest.iloc[1:], ["whisper_dtw"])
    override = "gate_a.min_paired_target_rate=0.95"
    from csasr.utils.config import load_config
    configured = load_config("lss/l1b_valid.yaml", _OVERRIDES(root) + [override])
    _publish_candidates(root, pd.concat([ctc, dtw], ignore_index=True),
                        cfg=configured)

    code, status = _run(root, "--set", override)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS not in status["blocked_reasons"]
    assert autoevidence.BLOCKED_NO_PAIRED_AGREEMENT in status["blocked_reasons"]
    evidence = json.loads(
        (root / "diagnostics/l1b_automatic_evidence.json").read_text())
    pair = evidence["pair_selection"]
    assert pair["reason"] == "insufficient_paired_target_overlap"
    assert pair["pair_diagnostics"][0]["paired_target_objects"] > 0
    _assert_l1c_locked(root)


def test_empty_consensus_cannot_pass(root):
    """Nothing reaches the primary bins, so jitter, coverage and span-schema
    criteria have nothing to run on. Omitting them was a vacuous pass."""
    manifest = _manifest(30)
    # 5 s apart: no unit will ever reach consensus
    _publish_candidates(root, _candidate_rows(manifest, ["existing_ctc", "whisper_dtw"],
                                              offset_ms=5000.0))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert status["blocked_reasons"]
    assert status["status"] == "blocked" and code == 0
    _assert_l1c_locked(root)


def test_an_unauthenticated_candidate_table_cannot_pass(root):
    """A nonempty parquet with no manifest could be anything."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]),
                        with_manifest=False)
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_UNAUTHENTICATED_CANDIDATES in status["blocked_reasons"]
    assert status["status"] == "blocked" and code == 0
    _assert_l1c_locked(root)


def test_tainted_candidates_cannot_pass(root):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]),
                        taint_reasons=("forced_prerequisite",))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_TAINTED_INPUTS in status["blocked_reasons"]
    assert status["taint"]["diagnostic_only"] is True
    _assert_l1c_locked(root)


def test_four_percent_invalid_rows_cannot_pass(root):
    """Above the proposal's 1% limit, below the 0.95 coverage floor."""
    frame = _candidate_rows(_manifest(30), ["existing_ctc", "whisper_dtw"])
    rows = frame.index[frame["aligner_family"] == "whisper_dtw"][:5]
    frame.loc[rows, "is_valid"] = False
    _publish_candidates(root, frame)
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    names = {c["name"] for c in status["gate"]["criteria"]}
    assert "target_invalid_rate_whisper_dtw" in names
    _assert_l1c_locked(root)


def test_rejected_third_family_is_diagnostic_after_a_pair_is_selected(root):
    """A configured experimental aligner may fail qualification without vetoing
    the deterministic CTC+Whisper production pair. Its failure is retained, but
    it is not a gate criterion and cannot contaminate that pair's result."""
    frame = _candidate_rows(
        _manifest(30),
        ["existing_ctc", "whisper_dtw", "qwen_forced_aligner"],
        invalid_family="qwen_forced_aligner")
    _publish_candidates(root, frame)

    code, status = _run(root)

    _assert_not_a_pass(status, code)  # lexical calibration is still unavailable
    criteria = {c["name"]: c for c in status["gate"]["criteria"]}
    assert criteria["target_invalid_rate_qwen_forced_aligner"]["comparison"] == "report"
    assert criteria["family_role_qwen_forced_aligner"]["value"] \
        == "diagnostic_rejected_not_selected_pair"
    assert criteria["target_invalid_rate_whisper_dtw"]["comparison"] == "<="
    _assert_l1c_locked(root)


def test_omitted_utterances_are_visible_in_coverage(root):
    """Half the intended universe never aligned. Candidate-relative coverage
    would read 1.0; the expected-universe denominator reads 0.5."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              drop_utterances=15))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    coverage = [c for c in status["gate"]["criteria"]
                if c["name"] == "alignment_target_object_coverage"]
    assert coverage and coverage[0]["passed"] is False
    structural = [c for c in status["gate"]["criteria"]
                  if c["name"] == "accepted_rejected_partition_structurally_valid"]
    assert structural and structural[0]["passed"] is True
    _assert_l1c_locked(root)


def test_one_language_missing_cannot_pass(root):
    """An EN-ZH comparison with only one language compares nothing."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              languages=("ZH",)))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_MISSING_LANGUAGE in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_the_previously_observed_one_valid_aligner_case_remains_a_no_pass(root):
    """existing_ctc valid, whisper_dtw not. The recorded production state."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"],
                                              invalid_family="whisper_dtw"))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    assert autoevidence.BLOCKED_INSUFFICIENT_ALIGNERS in status["blocked_reasons"]
    _assert_l1c_locked(root)


def test_perfect_agreement_still_cannot_pass_without_absolute_error(root):
    """The end-to-end version of "a shared bias agrees perfectly and is wrong":
    two families, identical spans, full coverage -- and still no accuracy
    evidence, so the gate blocks rather than passing."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code, status = _run(root)
    _assert_not_a_pass(status, code)
    # the two families agree to the sample, every unit is covered, and there is
    # still no source of absolute error
    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    assert evidence["independence"]["n_independent_valid_families"] == 2
    assert evidence["cross_aligner_agreement"][
        "cross_aligner_boundary_disagreement_ms"] == 0.0
    assert autoevidence.BLOCKED_MISSING_SYNTHETIC in status["blocked_reasons"]
    _assert_l1c_locked(root)
    assert not (root / "freeze" / "l1b_spans_freeze.json").exists()


def test_a_blocked_gate_is_a_completed_command_not_a_failed_job(root):
    """Exit 0 for a gate that could not conclude: the job did its work."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code = lss_l1b_valid.main(_argv(root, "--only", "consensus,evidence,jitter,gate"))
    status = read_status(root, "l1b_valid")
    assert status["status"] == "blocked", \
        [c["name"] for c in status["gate"]["criteria"] if not c["passed"]]
    assert code == 0
    assert autoevidence.BLOCKED_MISSING_SYNTHETIC in status["blocked_reasons"]
    _assert_l1c_locked(root)


# --------------------------------------------------------------------------
# workflow, status and exit-code behaviour on the production path
# --------------------------------------------------------------------------
def test_automatic_preparation_exits_zero_and_never_opens_a_verdict_file(root, monkeypatch):
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))

    def explode(*a, **k):                      # pragma: no cover - must not run
        raise AssertionError("automatic mode read a human verdict file")

    monkeypatch.setattr("csasr.lss.audit.verdicts.ingest", explode)
    code = lss_l1b_valid.main(_argv(root, "--prepare-audit",
                                     "--only", "consensus,evidence"))
    status = read_status(root, "l1b_valid")
    assert code == 0
    assert status["status"] == "completed"
    assert status["mode"] == "automatic"
    assert status["next_action"] == "evaluate_gate_a"
    assert status.get("gate") is None, "a preparation run decides nothing"


def test_manual_pack_in_automatic_mode_is_refused(root):
    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    with pytest.raises(SystemExit, match="manual workflow"):
        lss_l1b_valid.main(_argv(root, "--only", "pack"))


def test_an_unhandled_exception_is_recorded_as_failed_not_left_running(root, monkeypatch):
    """A crash used to leave `running`, which reads as an in-flight job forever."""
    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    monkeypatch.setattr(lss_l1b_valid.consensus_prod, "tolerance_sweep",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = lss_l1b_valid.main(_argv(root, "--only", "consensus,gate"))
    status = read_status(root, "l1b_valid")
    assert code == 2
    assert status["status"] == "failed"
    assert "boom" in status["error"]
    _assert_l1c_locked(root)


def test_the_lock_is_taken_before_the_status_is_claimed(root):
    """A second job that cannot get the lock must not overwrite the active
    run's status on its way out."""
    from csasr.utils.status import StageLock

    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    write_status(root, "l1b_valid", "passed", complete=True)
    with StageLock(root, "l1b_valid"):
        code = lss_l1b_valid.main(_argv(root, "--only", "consensus"))
    assert code == 2, "the refused run reports that it did not do its work"
    assert read_status(root, "l1b_valid")["status"] == "passed", \
        "the refused run must not overwrite the active run's status"


# --------------------------------------------------------------------------
# role survives all the way to the per-role criteria
# --------------------------------------------------------------------------
def test_per_role_counts_are_real_numbers_not_zeros(root):
    """Job 38573 accepted 32,086 spans and reported 0 for all six roles, so three
    per-role span diagnostics failed on a dropped column rather than on data.
    Data sufficiency itself is now independently counted from the full L0 role
    manifest."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root)

    spans = pd.read_parquet(root / "alignments" / "consensus_spans_v1.parquet")
    assert "role" in spans.columns
    assert set(spans["role"]) == {"D-construct"}

    coverage = json.loads((root / "diagnostics" / "l1b_coverage.json").read_text())
    counts = coverage["per_role_spans"]["D-construct"]
    assert counts["spans"] > 0 and counts["utterances"] == 30
    assert counts["en_spans"] > 0 and counts["zh_spans"] > 0

    names = {c["name"]: c for c in read_status(root, "l1b_valid")["gate"]["criteria"]}
    assert names["role_counts_available"]["passed"] is True
    # the full role-manifest count, not the audit sample or span count, is what
    # the data-sufficiency threshold is evaluated against.
    criterion = names["data_sufficiency_construct_bilingual_utterances"]
    assert criterion["value"] == 30
    assert coverage["data_sufficiency"]["data_sufficiency_universe"] == \
        "full_l0_role_manifest"


def test_a_span_table_without_role_fails_the_schema_criterion(root, monkeypatch):
    """Loud, not silent: a mechanical failure, so the status is `failed`."""
    real_build = lss_l1b_valid.consensus_prod.build

    def drop_role(candidates, config, **kwargs):
        accepted, rejected, pairwise = real_build(candidates, config, **kwargs)
        if len(accepted) and "role" in accepted:
            accepted = accepted.drop(columns=["role"])
        return accepted, rejected, pairwise

    monkeypatch.setattr(lss_l1b_valid.consensus_prod, "build", drop_role)
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    code, status = _run(root)

    names = {c["name"]: c for c in status["gate"]["criteria"]}
    assert names["span_schema_valid"]["passed"] is False
    assert status["status"] == "failed" and code == 2
    _assert_l1c_locked(root)


# --------------------------------------------------------------------------
# the corresponding-pair requirement, through the production assembly
# --------------------------------------------------------------------------
def test_a_naturally_qualifying_family_without_synthetic_calibration_blocks(
        root, synthetic_stubs):
    """CTC and Qwen qualify on natural speech; only CTC and Whisper are scored on
    the gate set. Two classes on each side, a different pair on each side."""
    manifest = _manifest(30)
    rows = _candidate_rows(manifest, ["existing_ctc", "qwen_forced_aligner"])
    _publish_candidates(root, rows)
    code, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    assert evidence["pair_selection"]["selected_pair"] == [
        "existing_ctc", "qwen_forced_aligner"]
    calibration = evidence["synthetic_calibration"]
    assert calibration["available"] is False
    assert "qwen_forced_aligner" in calibration["detail"]
    assert autoevidence.BLOCKED_UNCALIBRATED_FAMILIES in status["blocked_reasons"]

    names = {c["name"]: c for c in status["gate"]["criteria"]}
    assert names["synthetic_calibration"]["value"] == \
        autoevidence.BLOCKED_UNCALIBRATED_FAMILIES
    _assert_not_a_pass(status, code)
    _assert_l1c_locked(root)


def test_aligner_sweep_publishes_candidate_parents_and_transitive_taint(
        root, monkeypatch):
    """The merged table is the evidence Gate A reads, so provenance attached
    only to per-family caches or a status payload is insufficient."""
    import logging

    from csasr.lss.align import candidates as cand
    from csasr.nat5h.coordinates import EncoderGeometry

    cfg = _resolved_cfg(root)
    sample = _manifest(3)
    sample.attrs["sampling_report"] = {
        "schema_version": lss_l1b_valid.SAMPLING_REPORT_SCHEMA,
        "design": {"sample_dialogues": "all", "utterances_per_dialogue": 15,
                   "sample_stratify_field": "dialogue_id", "sample_seed": 303},
        "roles": {"D-construct": {"selected_dialogues": 1,
                                    "selected_utterances": 3,
                                    "dialogue_shortfall": 0}},
    }
    table = _candidate_rows(sample, ["existing_ctc", "whisper_dtw"])
    prerequisite_parent = {
        "path": "freeze/spec_freeze_v1.json", "sha256": "freeze-sha",
        "producing_run_id": "l0/run-1", "diagnostic_only": True,
        "taint_reasons": ["l0_forced"], "parent_run_ids": ["source/run-0"],
    }
    cache_parent = {
        "path": "alignments/candidates_existing_ctc.parquet",
        "sha256": "cache-sha", "producing_run_id": "l1b/cache-attempt",
        "diagnostic_only": True, "taint_reasons": ["cached_parent_taint"],
        "parent_run_ids": ["l0/run-1"],
    }
    calls = []

    def fake_run_families(manifest, cfg, geometry, identity, families, **kwargs):
        calls.append(kwargs)
        return table, {
            "families": {family: {"state": "ok"} for family in families},
            "cache_manifests": {"existing_ctc": cache_parent},
            "request_manifest": cand.request_manifest_record(manifest),
        }

    monkeypatch.setattr(lss_l1b_valid, "sweep_sample",
                        lambda cfg, roles, missing_ok=False: sample)
    monkeypatch.setattr("csasr.lss.align.devselect.load_aligner_selection",
                        lambda root: {"available": False, "reason": "test"})
    monkeypatch.setattr("csasr.models.whisper.load_whisper", lambda cfg: object())
    monkeypatch.setattr(EncoderGeometry, "from_bundle",
                        classmethod(lambda cls, bundle: cls.from_values()))
    monkeypatch.setattr("csasr.lss.align.candidates.run_families",
                        fake_run_families)

    report = lss_l1b_valid._run_aligner_sweep(
        cfg, logging.getLogger("candidate-provenance-test"), run_dir="test-run",
        roles=LABELLED_ROLES, parents=[prerequisite_parent],
        taint={"diagnostic_only": True,
               "taint_reasons": ["forced_prerequisite"]})

    assert calls and calls[0]["parents"] == [prerequisite_parent]
    assert calls[0]["taint_reasons"] == ["forced_prerequisite"]
    published = report["manifest"]
    assert published["diagnostic_only"] is True
    assert set(published["taint_reasons"]) == {
        "forced_prerequisite", "l0_forced", "cached_parent_taint"
    }
    assert {p["path"] for p in published["parent_artifacts"]} == {
        prerequisite_parent["path"], cache_parent["path"]
    }
    assert published["sampling"]["design"]["sample_seed"] == 303
    assert report["sampling"] == sample.attrs["sampling_report"]
    verdict = manifest_mod.verify(
        root / "alignments" / "candidates_all.parquet", cfg=cfg,
        require_identity=True)
    assert verdict["ok"] and verdict["manifest"] == published


def test_the_disagreement_is_measured_between_the_corresponding_families(
        root, synthetic_stubs):
    """Naming one pair and measuring another is how an accuracy claim ends up
    being about an aligner that produced none of the spans."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)

    evidence = json.loads(
        (root / "diagnostics" / "l1b_automatic_evidence.json").read_text())
    pair = evidence["pair_selection"]["selected_pair"]
    agreement = evidence["cross_aligner_agreement"]
    assert set(pair) == {"existing_ctc", "whisper_dtw"}
    assert {agreement["family_a"], agreement["family_b"]} == set(pair)


# --------------------------------------------------------------------------
# a gate set is confirmatory once
# --------------------------------------------------------------------------
def test_gate_item_fingerprint_includes_truth_not_sent_to_the_aligner():
    """Audio inputs alone cannot authenticate absolute-error ground truth."""
    from csasr.lss.align import candidates as candidate_cache
    from csasr.lss.align import synthetic

    items, _ = _fake_rendered("gate", 2)
    changed_truth = items.copy()
    changed_truth.loc[0, "true_boundary_sec"] += 0.25
    changed_truth.loc[0, "zh_end_sec"] += 0.25

    assert candidate_cache.request_manifest_fingerprint(
        synthetic.aligner_manifest(items)) == \
        candidate_cache.request_manifest_fingerprint(
            synthetic.aligner_manifest(changed_truth))
    assert candidate_cache.request_manifest_fingerprint(items) != \
        candidate_cache.request_manifest_fingerprint(changed_truth)


def test_evaluating_the_gate_exposes_the_generation_it_read(root, synthetic_stubs):
    from csasr.lss.align import exposure

    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    ledger = exposure.load_ledger(root)
    entry = exposure.find_generation(ledger, "gate", 1)
    assert entry is not None and entry["exposed"] is True
    assert status["status"] in entry["exposed_reason"]
    # so the next evaluation must build a different generation
    assert exposure.current_generation(ledger, "gate") == 2


def test_a_second_evaluation_cannot_reuse_the_exposed_sources(root,
                                                             synthetic_stubs):
    """The stub re-renders the same audio for generation 2. That is what a real
    exhausted pool looks like, and it must block rather than quietly confirm a
    configuration on items whose scores have already been read."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)
    code, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    assert autoevidence.BLOCKED_EXPOSED_GATE_SET in status["blocked_reasons"]
    # blocked, not failed: nothing is malformed, there is simply nothing left to
    # judge with, and a mechanical failure would outrank the blocker and call an
    # exhausted source pool an implementation defect
    assert status["status"] == "blocked" and code == 0
    names = {c["name"]: c["value"] for c in status["gate"]["criteria"]}
    assert names["synthetic_gate_set_is_fresh"] == 0
    assert any("unexposed source audio" in a["action"]
               for a in status["next_actions"])
    _assert_l1c_locked(root)


def test_interrupted_synthetic_preparation_retries_with_a_new_generation(
        root, monkeypatch):
    """A crash after scoring but before evaluation must not authenticate stale
    scores on retry.

    The first invocation raises immediately after score publication. Generation
    1 is therefore recorded but unexposed and its score table is complete while
    the stage status is failed. The retry must retire that attempt, reserve its
    sources, allocate generation 2, rebuild every candidate cache under
    ``--overwrite``, and bind the replacement score manifest to generation 2's
    exact alignment request.
    """
    from csasr.lss.align import devselect, exposure
    from csasr.nat5h.coordinates import EncoderGeometry

    dev_items, dev_candidates = _fake_rendered("dev", 120)
    manifest_mod.publish_frame(
        root / devselect.DEV_ITEMS_FILE, dev_items, stage="l1a_diag",
        cfg={"experiment": {"output_root": str(root)}, "model": {"id": "fake"}},
        key_columns=("pair_id",), schema=devselect.DEV_ITEMS_SCHEMA)

    generated: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    overwrite_seen: list[tuple[str, bool]] = []

    def fake_build_set(manifest, cfg, *, n_pairs, seed, out_dir, purpose,
                       sources=None, generation=None):
        assert purpose == "gate" and generation is not None
        # Namespacing the source and pair IDs makes a generation replacement
        # observable. Reusing generation 1's IDs here would not exercise the
        # immutable ledger or exact-request checks.
        items, candidates = _fake_rendered(
            f"gate_g{generation}", 120, offset_sec=-0.010 * generation)
        items["purpose"] = "gate"
        generated[int(generation)] = (items, candidates)
        return items, {"purpose": "gate", "pairs": int(len(items)),
                       "generation": int(generation),
                       "set_label": f"gate_g{generation}"}

    def fake_run_families(manifest, cfg, geometry, identity, families, **kwargs):
        first_id = str(manifest["utterance_id"].iloc[0])
        if first_id.startswith("dev_syn_"):
            purpose, table = "dev", dev_candidates
        else:
            generation = next(
                g for g in generated if first_id.startswith(f"gate_g{g}_syn_"))
            purpose, table = f"gate_g{generation}", generated[generation][1]
        overwrite_seen.append((purpose, bool(kwargs.get("overwrite"))))
        table = table[table["aligner_family"].isin(list(families))]
        return table.reset_index(drop=True), {
            "families": {f: {"state": "ok"} for f in families},
            "cache_manifests": {},
        }

    monkeypatch.setattr("csasr.lss.align.synthetic.build_set", fake_build_set)
    monkeypatch.setattr("csasr.lss.align.candidates.run_families",
                        fake_run_families)
    monkeypatch.setattr("csasr.models.whisper.load_whisper", lambda cfg: object())
    monkeypatch.setattr(EncoderGeometry, "from_bundle",
                        classmethod(lambda cls, bundle: cls.from_values()))
    _publish_candidates(root, _candidate_rows(
        _manifest(30), ["existing_ctc", "whisper_dtw"]))

    real_select = devselect.select_operating_tolerance
    selections = 0

    def interrupt_once(*args, **kwargs):
        nonlocal selections
        selections += 1
        if selections == 1:
            raise RuntimeError("simulated interruption after score publication")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(devselect, "select_operating_tolerance", interrupt_once)

    first_code, first_status = _run(root, only="synthetic")
    first_ledger = exposure.load_ledger(root)
    first_entry = exposure.find_generation(first_ledger, "gate", 1)
    first_score_manifest = manifest_mod.load(
        root / autoevidence.SYNTHETIC_SCORES_FILE)
    immutable_first = {
        key: first_entry[key]
        for key in ("source_utterance_ids", "source_fingerprint", "pair_ids",
                    "alignment_request_sha256", "item_set_sha256")
    }
    assert first_code == 2 and first_status["status"] == "failed"
    assert first_entry["exposed"] is False
    assert first_score_manifest["synthetic_gate_generation"] == 1
    assert first_score_manifest["synthetic_gate_request_sha256"] == \
        first_entry["alignment_request_sha256"]
    assert first_score_manifest["synthetic_gate_item_sha256"] == \
        first_entry["item_set_sha256"]

    second_code, second_status = _run(
        root, "--overwrite", only="synthetic," + DEFAULT_PARTS)
    second_ledger = exposure.load_ledger(root)
    retired = exposure.find_generation(second_ledger, "gate", 1)
    second_entry = exposure.find_generation(second_ledger, "gate", 2)
    second_score_manifest = manifest_mod.load(
        root / autoevidence.SYNTHETIC_SCORES_FILE)

    assert second_code == 0 and second_status["status"] in {
        "blocked", "completed_no_go"
    }
    assert retired["abandoned"] is True
    assert {key: retired[key] for key in immutable_first} == immutable_first
    assert second_entry is not None and second_entry["exposed"] is True
    assert second_entry["source_fingerprint"] != retired["source_fingerprint"]
    assert second_score_manifest["synthetic_gate_generation"] == 2
    assert second_score_manifest["synthetic_gate_request_sha256"] == \
        second_entry["alignment_request_sha256"]
    assert second_score_manifest["synthetic_gate_item_sha256"] == \
        second_entry["item_set_sha256"]
    assert second_score_manifest["synthetic_gate_request_sha256"] != \
        first_score_manifest["synthetic_gate_request_sha256"]
    assert overwrite_seen == [
        ("dev", False), ("gate_g1", False),
        ("dev", True), ("gate_g2", True),
    ], "--overwrite must reach every synthetic family dispatch on retry"

    # Even an otherwise authentic score artifact is unavailable when its exact
    # gate request does not match the generation the evaluator is about to read.
    stale = autoevidence.synthetic_calibration_status(
        root, selected_pair=["existing_ctc", "whisper_dtw"],
        cfg=_resolved_cfg(root), require_authentication=True,
        expected_gate_generation=2,
        expected_gate_request_sha256=second_entry["alignment_request_sha256"],
        expected_gate_item_sha256=first_entry["item_set_sha256"])
    assert stale["available"] is False
    assert stale["reason"] == autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC


# --------------------------------------------------------------------------
# the status contract
# --------------------------------------------------------------------------
def test_a_terminal_blocked_run_is_recorded_as_complete(root):
    """`complete` answers "did this run finish the work it set out to do"; the
    status already says whether the gate passed. Recording a truthful terminal
    `blocked` as incomplete gave it the same shape as a job killed mid-write."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root)
    assert status["status"] == "blocked"
    assert status["complete"] is True
    # and it still satisfies no prerequisite
    _assert_l1c_locked(root)


def test_a_failed_run_is_recorded_as_incomplete(root, monkeypatch):
    monkeypatch.setattr(lss_l1b_valid.consensus_prod, "tolerance_sweep",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _publish_candidates(root, _candidate_rows(_manifest(30), ["existing_ctc"]))
    code, status = _run(root, only="consensus,gate")
    assert status["status"] == "failed" and code == 2
    assert status.get("complete") is False


def test_the_next_actions_never_recommend_annotation_as_the_primary_action(
        root, synthetic_stubs):
    """The old `_next_action` returned "run --prepare-manual-audit" for the
    commonest blocker, which repairs none of what is actually wrong."""
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root, only="synthetic," + DEFAULT_PARTS)

    actions = status["next_actions"]
    assert actions[0]["kind"] != "manual"
    assert "annotate" not in actions[0]["action"].lower()
    # annotation is present exactly once, last, and marked optional
    assert actions[-1]["kind"] == "optional"
    assert sum(1 for a in actions if a["kind"] in ("manual", "optional")) == 1
    assert status["next_action"] == actions[0]["action"]


def test_a_score_table_from_an_exposed_generation_is_refused(root, synthetic_stubs):
    """The hole this closes: authentic bytes, correct schema, right configuration
    -- and the items' scores had already been read.

    A second evaluation would otherwise re-report the first one's numbers as if
    they were a fresh confirmation, which is the whole failure the ledger exists
    to prevent.
    """
    from csasr.lss.align import exposure

    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)

    scores = root / autoevidence.SYNTHETIC_SCORES_FILE
    verdict = manifest_mod.verify(scores, cfg=_resolved_cfg(root),
                                  require_identity=True)
    assert verdict["ok"], "the table itself is authentic; that is the point"
    assert verdict["manifest"]["synthetic_gate_generation"] == 1
    assert exposure.current_generation(exposure.load_ledger(root), "gate") == 2

    status = autoevidence.synthetic_calibration_status(
        root, selected_pair=["existing_ctc", "whisper_dtw"],
        cfg=_resolved_cfg(root), expected_gate_generation=2)
    assert status["available"] is False
    assert status["reason"] == autoevidence.BLOCKED_EXPOSED_GATE_SET
    assert "already been read" in status["detail"]

    # and evaluating from what is on disk reports it rather than reusing it
    code, second = _run(root, only=DEFAULT_PARTS)
    assert autoevidence.BLOCKED_EXPOSED_GATE_SET in second["blocked_reasons"]
    assert second["status"] == "blocked" and code == 0
    _assert_l1c_locked(root)


def test_a_truncated_score_table_cannot_pass_even_with_a_manifest(root,
                                                                 synthetic_stubs):
    """A truncated parquet is still a valid parquet with fewer rows.

    `expected_keys_sha256` is what detects it: the manifest records the row keys
    the producer promised to write, so a prefix of them is refused rather than
    measured.
    """
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _run(root, only="synthetic," + DEFAULT_PARTS)

    scores_path = root / autoevidence.SYNTHETIC_SCORES_FILE
    scores = pd.read_parquet(scores_path)
    assert len(scores) > 2
    scores.head(2).to_parquet(scores_path, index=False)     # same manifest, fewer rows

    verdict = manifest_mod.verify(scores_path, cfg=_resolved_cfg(root),
                                  require_identity=True,
                                  frame=pd.read_parquet(scores_path))
    assert verdict["ok"] is False
    assert verdict["verdict"] in ("sha256_mismatch", "expected_keys_mismatch")

    status = autoevidence.synthetic_calibration_status(
        root, selected_pair=["existing_ctc", "whisper_dtw"],
        cfg=_resolved_cfg(root))
    assert status["available"] is False
    assert status["reason"] == autoevidence.BLOCKED_UNAUTHENTICATED_SYNTHETIC


def test_a_missing_measurement_cannot_satisfy_a_gate_criterion(root,
                                                              synthetic_stubs,
                                                              monkeypatch):
    """NaN never satisfies a comparison, so evidence that could not be measured
    fails its criterion instead of silently passing it."""
    real = lss_l1b_valid.jitter_mod.jitter_stability

    def nan_jitter(primary, cfg):
        table, summary = real(primary, cfg)
        for key in list(summary):
            if isinstance(summary[key], dict):
                summary[key] = {k: float("nan") for k in summary[key]}
        return table, summary

    monkeypatch.setattr(lss_l1b_valid.jitter_mod, "jitter_stability", nan_jitter)
    _publish_candidates(root, _candidate_rows(_manifest(30),
                                              ["existing_ctc", "whisper_dtw"]))
    _, status = _run(root)

    jitter = [c for c in status["gate"]["criteria"]
              if c["name"].startswith("jitter_") and c["comparison"] != "report"]
    assert jitter, "the jitter criteria must be constructed, not omitted"
    assert all(c["passed"] is False for c in jitter)
