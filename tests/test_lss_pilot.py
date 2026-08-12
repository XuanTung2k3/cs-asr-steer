"""LSS unit test: the compute projections the L0 budget gate is made of."""
from __future__ import annotations

import pytest

from csasr.lss.pilot import PilotResult, extrapolate


def test_each_family_is_costed_at_its_own_rate():
    """Charging every family the slowest family's rate overstates L1 threefold.

    Measured on this corpus: existing_ctc 0.33 utt/s, whisper_dtw 35 utt/s.
    The old rule (n_items * n_families / min_rate) reported 12.2 GPU-h for
    4900 utterances and failed the budget guard; the actual cost of running
    both families over those utterances is 4.1 h.
    """
    results = [
        PilotResult(mode="align_existing_ctc", n=20, seconds=59.8, rate=0.3346),
        PilotResult(mode="align_whisper_dtw", n=20, seconds=0.6, rate=35.18),
    ]
    projected = extrapolate(results, {}, {"alignment_utterances": 4900,
                                          "alignment_families": 2})
    assert projected["l1_alignment_gpu_hours"] == pytest.approx(4.11, abs=0.05)
    assert projected["l1_alignment_gpu_hours_worst_case"] == pytest.approx(8.14, abs=0.05)
    assert projected["l1_alignment_gpu_hours"] < projected["l1_alignment_gpu_hours_worst_case"]


def test_an_untimed_family_is_charged_at_the_slowest_measured_rate():
    """A configured family nobody timed must not be free."""
    results = [
        PilotResult(mode="align_existing_ctc", n=20, seconds=59.8, rate=0.3346),
        PilotResult(mode="align_whisper_dtw", n=20, seconds=0.6, rate=35.18),
    ]
    two = extrapolate(results, {}, {"alignment_utterances": 4900,
                                    "alignment_families": 2})
    three = extrapolate(results, {}, {"alignment_utterances": 4900,
                                      "alignment_families": 3})
    assert three["l1_alignment_gpu_hours"] > two["l1_alignment_gpu_hours"]
    # the third is billed at the slowest rate, i.e. one more CTC pass
    assert (three["l1_alignment_gpu_hours"] - two["l1_alignment_gpu_hours"]
            == pytest.approx(4900 / 0.3346 / 3600, abs=0.05))


def test_a_family_that_cannot_load_is_not_budgeted():
    """Weights on disk are not enough; the architecture has to exist."""
    from csasr.lss.pilot import runnable_families

    cfg = {"alignment": {"families": ["whisper_dtw", "qwen_forced_aligner"],
                         "qwen": {"local_model_dir": "/nonexistent/path"}}}
    families = runnable_families(cfg)
    assert families["whisper_dtw"]["runnable"] is True
    assert families["qwen_forced_aligner"]["runnable"] is False
    assert families["qwen_forced_aligner"]["reason"] == "checkpoint_missing"


def test_qwen_capability_is_judged_against_the_runtime_the_adapter_uses(tmp_path):
    """`transformers` is the wrong thing to ask.

    `Qwen3ForcedAlignerAdapter.load` goes through `qwen_asr`, which vendors its
    own `Qwen3ASRForConditionalGeneration`. Asking whether the *installed
    transformers* implements that class declared the family unloadable while the
    aligner loaded perfectly well -- job 38502 reached `adapter.run()`, which
    only happens after `adapter.load()` returns -- and a whole family's worth of
    GPU hours was dropped from the L1 budget on that false negative.
    """
    import json

    from csasr.lss.pilot import QWEN_RUNTIME_MODULES, _qwen_runnable

    assert QWEN_RUNTIME_MODULES[0] == "qwen_asr.core.transformers_backend", \
        "the adapter's own runtime must be consulted first"

    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"architectures": ["NoSuchArchitecture"]}), encoding="utf-8")
    ok, reason = _qwen_runnable({"local_model_dir": str(model_dir)})
    assert ok is False
    assert "architecture_unavailable" in reason
    # the reason must name every runtime that was asked, not just transformers
    for module_name in QWEN_RUNTIME_MODULES:
        assert module_name in reason


def test_a_vendored_architecture_counts_as_runnable(tmp_path, monkeypatch):
    """The real checkpoint's architecture lives in qwen_asr, not transformers."""
    import json
    import sys
    import types

    from csasr.lss import pilot

    backend = types.ModuleType("fake_qwen_backend")
    backend.Qwen3ASRForConditionalGeneration = object
    monkeypatch.setitem(sys.modules, "fake_qwen_backend", backend)
    monkeypatch.setattr(pilot, "QWEN_RUNTIME_MODULES", ("fake_qwen_backend",))

    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3ASRForConditionalGeneration"]}), encoding="utf-8")
    ok, reason = pilot._qwen_runnable({"local_model_dir": str(model_dir)})
    assert ok is True
    assert "fake_qwen_backend" in reason


def test_an_unimportable_runtime_is_reported_not_swallowed(tmp_path, monkeypatch):
    import json

    from csasr.lss import pilot

    monkeypatch.setattr(pilot, "QWEN_RUNTIME_MODULES", ("no_such_module_at_all",))
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"architectures": ["Whatever"]}), encoding="utf-8")
    ok, reason = pilot._qwen_runnable({"local_model_dir": str(model_dir)})
    assert ok is False
    assert "unimportable" in reason and "ModuleNotFoundError" in reason
