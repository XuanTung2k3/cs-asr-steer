"""Focused CPU tests for P1 causal acceptance (tiny random Whisper with a real layer 24)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.core_p1 import (DIRECTION_TOL, LAYER, VERSION_P1, direction,
                                        processed_argmax, reference_edit, selected_gate,
                                        step_inputs)
from csasr.inference_cf.core_r2 import gate_values
from csasr.lss.sites import (DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder,
                             assert_no_site_hooks)
import experiments.inference_cf_p1 as p1
from experiments.inference_cf_p1_accept import evaluate

VOCAB, EOS, FRAMES = 64, 2, 100
CB, CE = [1, 5, 6, 7], [1, 4, 6, 7]
PART = {"matrix_ids": list(range(8, 44)), "embedded_ids": list(range(44, 54)),
        "ambiguous_ids": list(range(0, 8)) + list(range(54, VOCAB))}


class Tok:
    eos_token_id = EOS
    all_special_ids = [EOS]

    def convert_ids_to_tokens(self, i):
        return "abcdefghijklmnopqrstuvwxyz"[int(i) % 26]

    def decode(self, ids, **kwargs):
        return "".join(self.convert_ids_to_tokens(i) for i in ids)


def tiny_bundle(seed=0):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(seed)
    cfg = WhisperConfig(vocab_size=VOCAB, num_mel_bins=8, d_model=16, encoder_layers=1,
                        decoder_layers=25, encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=FRAMES,
                        max_target_positions=64, pad_token_id=0, bos_token_id=1, eos_token_id=EOS,
                        decoder_start_token_id=1, dropout=0.0, attention_dropout=0.0)
    model = WhisperForConditionalGeneration(cfg).eval()
    model.set_attn_implementation("eager")
    model.generation_config.alignment_heads = [[1, 0], [24, 1]]
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=Tok()), device="cpu",
                           d_model=16, num_decoder_layers=25, sample_rate=16000,
                           decoder_layer=lambda i: model.model.decoder.layers[i])


def encoded(seed=1):
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(seed)
    return BaseModelOutput(last_hidden_state=torch.randn((1, FRAMES, 16)))


def run_decode(bundle, alpha, monkeypatch, max_new_tokens=8):
    monkeypatch.setattr(p1, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    return p1.decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(),
                     conditions={"cB": CB, "cE": CE, "language_token_ids": [4, 5]},
                     partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                     alpha=alpha, max_new_tokens=max_new_tokens, num_forced_prefix=4)


# ---- pure math -----------------------------------------------------------------------------

def test_direction_unit_norm_sign_and_fallbacks():
    hb, he = torch.tensor([1., 2., 3.]), torch.tensor([2., 0., 3.])
    r = direction(he, hb)
    assert r["status"] == "ok" and torch.linalg.vector_norm(r["d"]) == pytest.approx(1, abs=1e-5)
    assert torch.allclose(r["d"], (he - hb) / torch.linalg.vector_norm(he - hb), atol=1e-5)  # sign = E - M
    assert direction(hb + DIRECTION_TOL / 10, hb)["status"] == "direction_fail:tiny"
    assert direction(torch.tensor([float("nan"), 0., 0.]), hb)["status"] == "direction_fail:nonfinite"


def test_reference_edit_normpreserve_sign_and_zero_bypass():
    h, d = torch.tensor([3., 4., 0.]), torch.tensor([0., 0., 1.])
    e = reference_edit(h, d, 1.0, .5)
    assert float(torch.linalg.vector_norm(e["h_prime"])) == pytest.approx(5.0, rel=1e-6)
    assert e["cos_edit_d"] > 0 and e["applied"]
    for a, g in ((0.0, .5), (1.0, 0.0)):
        z = reference_edit(h, d, a, g)
        assert not z["applied"] and torch.equal(z["h_prime"], h.double()) and z["edit_norm"] == 0


def test_selected_gate_is_frozen_r2_old_gate():
    s, b = {"E": .7}, {"R": .6}
    assert selected_gate(s, b) == gate_values(s, b, {"R": .1})["g_old"] == pytest.approx(.42)


def test_processed_argmax_suppression_and_prefix_inputs():
    x = torch.tensor([0., 5., 1., 4.])
    assert processed_argmax(x, 0, [], []) == 1
    assert processed_argmax(x, 0, [1], []) == 3
    assert processed_argmax(x, 0, [], [3, 1]) == 2 and processed_argmax(x, 1, [], [3, 1]) == 1
    assert step_inputs(CB, [9, 10]) == CB + [9, 10]
    with pytest.raises(ValueError):
        step_inputs(CB, [1.5])


# ---- hook semantics on a real (tiny) Whisper ---------------------------------------------

def _forward(bundle, tokens, hook=None, layers=(23, 24)):
    ids = torch.tensor([tokens])
    ctx = hook if hook is not None else torch.no_grad()
    with ctx:
        with DecoderPostCrossAttnRecorder(bundle, list(layers)) as rec:
            out = bundle.model(encoder_outputs=encoded(), decoder_input_ids=ids,
                               use_cache=False, return_dict=True)
    return out.logits[0].detach(), {k: v[0].clone() for k, v in rec.states.items()}


def _hook(bundle, n, pos, alpha, g=0.8):
    torch.manual_seed(3)
    d = torch.nn.functional.normalize(torch.randn(16), dim=0)
    gain, dirs = torch.zeros((1, n)), torch.zeros((1, n, 16))
    gain[0, pos], dirs[0, pos] = g, d
    return DecoderPostCrossAttnInterventionHook(
        bundle, LAYER, None, alpha=alpha, num_forced_prefix=4, norm_preserve=True, record=True,
        record_last_only=False, action_fn=lambda q, u_source, r, abs_pos: (gain, dirs)), d


def test_alpha_zero_and_gate_zero_are_bitwise_identity():
    b = tiny_bundle()
    tokens = CB + [9, 10, 11]
    base, _ = _forward(b, tokens)
    hook0, _ = _hook(b, len(tokens), 5, alpha=0.0)
    assert torch.equal(_forward(b, tokens, hook0)[0], base)
    hook_g0, _ = _hook(b, len(tokens), 5, alpha=1.0, g=0.0)
    assert torch.equal(_forward(b, tokens, hook_g0)[0], base)
    assert_no_site_hooks(b)


def test_only_layer24_site_at_intended_position_is_edited_with_normpreserve():
    b = tiny_bundle()
    tokens = CB + [9, 10, 11]
    base_logits, base = _forward(b, tokens)
    hook, d = _hook(b, len(tokens), 5, alpha=1.0)
    steered_logits, st = _forward(b, tokens, hook)      # recorder installed after the hook sees r'
    assert torch.equal(st[23], base[23])                 # earlier layer untouched
    other = [i for i in range(len(tokens)) if i != 5]
    assert torch.equal(st[24][other], base[24][other])   # other positions' site untouched
    assert not torch.equal(st[24][5], base[24][5])
    assert float(st[24][5].norm()) == pytest.approx(float(base[24][5].norm()), rel=1e-5)
    assert float(torch.dot(st[24][5] - base[24][5], d)) > 0   # sign of d
    assert torch.equal(steered_logits[:5], base_logits[:5])    # causal: earlier queries unchanged
    assert not torch.equal(steered_logits[5], base_logits[5])  # decoder still valid, changed
    assert torch.isfinite(steered_logits).all()


def test_forced_prefix_position_is_never_edited():
    b = tiny_bundle()
    tokens = CB + [9]
    base, _ = _forward(b, tokens)
    hook, _ = _hook(b, len(tokens), 3, alpha=1.0)          # query at final prompt token (t=0)
    assert torch.equal(_forward(b, tokens, hook)[0], base)


# ---- end-to-end decode with the real replay/hook path -----------------------------------

def test_decode_alpha0_identity_and_isolation(monkeypatch):
    b = tiny_bundle()
    r = run_decode(b, 0.0, monkeypatch)
    assert r["steps"] and all(s["f3_bitwise_equals_f1"] and s["steered_next"] == s["unsteered_next"]
                              and not s["edit_applied"] for s in r["steps"])
    assert all(s["input_is_prompt_plus_prefix"] for s in r["steps"])
    assert r["edited_positions"] == []
    assert_no_site_hooks(b)


def test_decode_alpha1_real_edit_site_sign_normpreserve_and_serialization(monkeypatch, tmp_path):
    b = tiny_bundle()
    r = run_decode(b, 1.0, monkeypatch)
    edits = [s for s in r["steps"] if s["edit_applied"]]
    assert edits, "expected at least one real eligible edit"
    for s in edits:
        a = s["hook_audit"]
        assert a["layer"] == LAYER and a["abs_pos"] == s["query_index"] >= 4
        assert a["edit_norm"] > 0 and abs(a["post_norm"] - a["pre_norm"]) <= 1e-4 * a["pre_norm"]
        assert s["reference_cos_edit_d"] > 0
        assert abs(a["edit_norm"] - s["reference_edit_norm"]) <= 1e-3 * s["reference_edit_norm"]
        assert s["g"] == pytest.approx(selected_gate(s["local_support"], s["baseline"]))
    assert r["steps"][0]["query_index"] == 3 and not r["steps"][0]["edit_applied"]   # t=0 blocked
    assert all(p >= 4 for p in r["edited_positions"])
    varying = [s["cos_to_previous_direction"] for s in r["steps"]
               if s.get("cos_to_previous_direction") is not None]
    assert varying and min(varying) < 0.99                                    # sample-varying d
    assert r["terminated"] in ("eos", "cap") and isinstance(r["text"], str)
    assert_no_site_hooks(b)
    atomic_json(tmp_path / "r.json", r)                                        # serializable
    assert json.loads((tmp_path / "r.json").read_text())["steps"][1]["query_index"] == 4


def test_tiny_direction_prevents_edit(monkeypatch):
    b = tiny_bundle()
    monkeypatch.setattr(p1, "direction", lambda he, hb: {"status": "direction_fail:tiny",
                                                         "norm": 0.0, "d": None})
    r = run_decode(b, 1.0, monkeypatch)
    assert not any(s["edit_applied"] for s in r["steps"])
    assert all(s["f3_bitwise_equals_f1"] for s in r["steps"])


def test_acceptance_evaluator_passes_on_valid_synthetic_run(monkeypatch, tmp_path):
    b = tiny_bundle()
    config = json.loads(Path("configs/inference_cf/p1_causal_acceptance.json").read_text())
    config = {**config, "population_indices": [0]}
    panel = {"schema": VERSION_P1, "rows": [{"utterance_id": "u0", "audio_path": "x",
                                             "audio_sha256": "s", "duration_sec": 1.0}]}
    manifest = {"schema": VERSION_P1, "resolved_config": config, "panel_hash": digest(panel)}
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(tmp_path / "panel.json", panel)
    atomic_json(tmp_path / "runtime.json", {"elapsed_sec": 1.0, "peak_vram_bytes": 1})
    for a in config["alphas"]:
        r = run_decode(b, float(a), monkeypatch)
        r.update(status="ok", identity=f"u0|alpha={float(a)}", utterance_id="u0",
                 alpha=float(a), manifest_hash=manifest["manifest_hash"])
        atomic_json(tmp_path / "rows" / f"000_alpha{float(a)}.json", r)
    s = evaluate(tmp_path)
    assert s["verdict"] == "P1_CAUSAL_ACCEPTANCE_PASS", s["checks"]
    # A missing record blocks.
    (tmp_path / "rows" / "000_alpha1.0.json").unlink()
    assert evaluate(tmp_path)["verdict"] == "P1_BLOCKED"


def test_config_frozen_and_no_reference_or_tuning_api():
    config = json.loads(Path("configs/inference_cf/p1_causal_acceptance.json").read_text())
    assert config["alphas"] == [0.0, 1.0] and config["site"]["layer"] == LAYER == 24
    assert config["selected_gate"] == "g_old = E * R_B"
    assert config["population_indices"] == list(range(0, 300, 30))
    assert digest(config) == digest(json.loads(json.dumps(config)))
    names = inspect.signature(p1.decode).parameters
    assert not any(x in names for x in ("reference", "oracle", "ctc", "stratum", "layer", "alphas"))


# ---- v1.1 A6 instrument (after attempt 1, job 54770) ------------------------------------

def test_hook_realized_edit_equals_site_precision_replica(monkeypatch):
    b = tiny_bundle()
    r = run_decode(b, 1.0, monkeypatch)
    edits = [s for s in r["steps"] if s["edit_applied"]]
    assert edits
    for s in edits:
        assert s["hook_audit"]["edit_norm"] == pytest.approx(s["replica_edit_norm"], rel=1e-6)
        assert s["hook_audit"]["post_norm"] == pytest.approx(s["replica_post_norm"], rel=1e-6)


def test_bf16_sub_resolution_edit_breaks_float64_reference_but_not_replica():
    from csasr.models.hooks import apply_steering
    torch.manual_seed(0)
    h = (torch.randn(1280) * 9 / 1280 ** .5).to(torch.bfloat16)
    d = torch.nn.functional.normalize(torch.randn(1280), dim=0)
    g = 0.0042                                               # attempt-1 failing edit size
    site = h.view(1, 1, -1)
    realized = apply_steering(site, d.to(torch.bfloat16).view(1, 1, -1), 1.0, 1.0,
                              torch.tensor([[g]], dtype=torch.bfloat16), True)
    realized_norm = float((realized - site).float().norm())
    f64 = reference_edit(h.float(), d, 1.0, g)["edit_norm"]
    assert abs(realized_norm - f64) > 0.05 * f64            # v1 instrument would fail
    replica = apply_steering(h.view(1, 1, -1), d.to(torch.bfloat16).view(1, 1, -1), 1.0, 1.0,
                             torch.tensor([[g]], dtype=torch.bfloat16), True)
    assert float((replica - site).float().norm()) == realized_norm   # v1.1 instrument is exact


def test_v11_config_versioning():
    config = json.loads(Path("configs/inference_cf/p1_causal_acceptance.json").read_text())
    assert config["acceptance_version"] == "v1.1"
    assert config["acceptance"]["a6_reference"] == "site_precision_replica"
    assert config["acceptance"]["reference_edit_rel_tol"] == 0.05           # tolerance unchanged
    assert config["supersedes_attempt"]["job"] == "54770"
