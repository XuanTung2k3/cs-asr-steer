"""Cached-equivalence harness + CE1-CE8 evaluator on a tiny real-architecture Whisper."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_ce as ce  # noqa: E402
from experiments.inference_cf_ce_accept import evaluate  # noqa: E402
from csasr.inference_cf.core import atomic_json, digest  # noqa: E402
from transformers.models.whisper.tokenization_whisper import bytes_to_unicode  # noqa: E402

COND = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}


def test_ce_harness_and_evaluator_pass_on_tiny_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    b = tiny_bundle()
    bd = {v: k for k, v in bytes_to_unicode().items()}
    m = {"schema": ce.SCHEMA, "utterances": ["u0"], "layers": [16, 24], "alphas": [0.0, 1.0]}
    m["manifest_hash"] = digest(m)
    atomic_json(tmp_path / "manifest.json", m)
    for layer in (16, 24):
        for alpha in (0.0, 1.0):
            res, comps, _ = ce.ce_decode(
                b, enc=encoded(), waveform=np.ones(FRAMES * 320, dtype=np.float32), uid="u0",
                layer=layer, alpha=alpha, conditions=COND, partition=PART,
                null_probs={4: .5, 5: .5}, language_ids=(4, 5), nfp=4, suppress=[], begin=[],
                byte_decoder=bd, lid_cache={}, max_new_tokens=8)
            assert len(comps) == len(res["steps"])
            greedy = cached.cached_greedy(b, encoded(), CB, max_new_tokens=8)
            atomic_json(tmp_path / "rows" / f"00_L{layer}_a{alpha}.json",
                        {"identity": f"u0|L{layer}|a{alpha}", "layer": layer, "alpha": alpha,
                         "tokens": res["tokens"], "lineage_ok": res["lineage_ok"],
                         "distinct_caches": res["distinct_caches"],
                         "cached_greedy_tokens": greedy["tokens"], "r2_generate_tokens": [],
                         "steps": res["steps"], "comparisons": comps})
    atomic_json(tmp_path / "runtime.json", {"status": "completed", "elapsed_sec": 1.0,
                                            "peak_vram_bytes": 1, "cached_sec": 1.0, "audio_sec": 1.0})
    s = evaluate(tmp_path)
    assert s["verdict"] == "CACHED_EQUIVALENCE: PASS", s["checks"]
    assert s["ce6"]["cached_edits"] > 0 and s["ce6"]["non_near_tie_disagreements"] == 0
    # A missing shard blocks.
    (tmp_path / "rows" / "00_L16_a1.0.json").unlink()
    assert evaluate(tmp_path)["verdict"] == "CACHED_EQUIVALENCE: BLOCK"
