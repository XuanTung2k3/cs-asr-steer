"""Focused CPU checks for the versioned, unsteered P0-R2 contract."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from csasr.inference_cf.core import atomic_json, digest
from csasr.inference_cf.core_r2 import (
    VERSION_R2, conflict_from_logits, gate_values, local_support,
    max_attention_window, prefix_utf8_complete, same_prefix_inputs,
    tokenizer_partition, validate_gate_row,
)
import experiments.inference_cf_p0_r2 as runner
from experiments.inference_cf_p0_r2_evaluate import (
    _json_safe, auc_ap, cluster_intervals, evaluator_labels, unit_to_token_positions,
)


PART = {"matrix_ids": [0], "embedded_ids": [1], "ambiguous_ids": [2]}


def test_ls_b_null_and_signed_margin():
    null = local_support(.3099, .0199, .3099, .0199)
    assert null["A"] == 0 and null["E"] == 0
    assert null["sigmoid_A"] == .5
    assert local_support(.1, .9, .3099, .0199)["E"] == 0
    positive = local_support(.99, .01, .3099, .0199)
    assert 0 < positive["E"] < 1
    assert positive["pair_mass"] == pytest.approx(1)


@pytest.mark.parametrize("logits", [[0., 2., -1.], [2., 0., -1.], [-20., -25., 20.]])
def test_both_branch_conflict_invariants(logits):
    m = conflict_from_logits(torch.tensor(logits, dtype=torch.bfloat16), PART)
    assert 0 <= m["R"] <= m["Q"] <= 1.000001
    assert m["Q"] == pytest.approx(m["P_M"] + m["P_E"])
    assert m["R"] == pytest.approx(max(0, m["P_M"] - m["P_E"]))
    if m["P_M"] <= m["P_E"]:
        assert m["R"] == 0


def test_repairability_and_old_gate_reproduction():
    e = {"E": .8}
    b = {"R": .6}
    ecf = {"R": .2}
    g = gate_values(e, b, ecf)
    assert g == pytest.approx({"D": .4, "g_old": .48, "g_cf": .32})
    assert 0 <= g["D"] <= b["R"]
    assert gate_values(e, b, {"R": .7})["D"] == 0
    assert g["g_cf"] <= g["g_old"]


def test_max_window_integrates_mass_and_ties_earliest():
    a = np.zeros((2, 100))
    a[:, 10] = .4
    a[:, 70:75] = .12
    w = max_attention_window(a, 100 * 320)
    assert w.start_frame == 25 and w.end_frame == 75
    assert w.point_argmax_frame == 10  # point peak differs from best mass window
    assert w.end_sample - w.start_sample == 16000
    tie = max_attention_window(np.ones((2, 100)), 100 * 320)
    assert tie.start_frame == 0


def test_unheard_tail_and_partial_last_frame_cannot_enter_window():
    a = np.zeros((1, 1500))
    a[0, 1499] = 1000
    a[0, 1] = 1
    w = max_attention_window(a, 200 * 320 + 17)
    assert w.end_frame <= 201
    assert w.end_sample <= 200 * 320 + 17
    assert w.end_sample - w.start_sample == 16000
    short = max_attention_window(np.ones((1, 1500)), 7000)
    assert short.start_sample == 0 and short.end_sample == 7000
    with pytest.raises(ValueError, match="localizer_fail"):
        max_attention_window(np.zeros((1, 50)), 0)


class ByteTok:
    all_special_ids = [99]

    def convert_ids_to_tokens(self, i):
        return {10: "ä", 11: "¸", 12: "Ń", 13: "Ġ", 14: "H", 15: "i"}[i]


def test_prefix_utf8_uses_raw_bytes_not_replacement_text():
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    chars = bytes_to_unicode()
    tok = ByteTok()
    # Euro-sign UTF-8 is E2 82 AC; first byte alone is incomplete.
    tok.convert_ids_to_tokens = lambda i: chars[{10: 0xE2, 11: 0x82, 12: 0xAC}[i]]
    bd = {v: k for k, v in chars.items()}
    assert not prefix_utf8_complete(tok, [10], bd)
    assert not prefix_utf8_complete(tok, [10, 11], bd)
    assert prefix_utf8_complete(tok, [10, 11, 12], bd)


def test_same_prefix_exact_condition_and_no_future_token_parameter():
    cond = {"cB": [1, 2, 4, 5], "cE": [1, 3, 4, 5]}
    b, e = same_prefix_inputs(cond, [11, 12])
    assert b == [1, 2, 4, 5, 11, 12]
    assert e == [1, 3, 4, 5, 11, 12]
    assert b[4:] == e[4:]
    with pytest.raises(ValueError):
        same_prefix_inputs({"cB": [1, 2, 4, 5], "cE": [1, 3, 4, 6]}, [11])
    names = inspect.signature(runner.inference_utterance).parameters
    assert not any(x in names for x in ("reference", "oracle", "ctc", "stratum", "true_future", "poi"))


def test_dense_runner_uses_baseline_prefix_for_both_branches(monkeypatch):
    from types import SimpleNamespace

    class Tokenizer:
        eos_token_id = 99
        all_special_ids = [99]

        def convert_ids_to_tokens(self, token_id):
            return {14: "H", 15: "i"}[token_id]

        def decode(self, ids, **kwargs):
            return "".join({14: "H", 15: "i"}[x] for x in ids)

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[14, 15, 99]])

    class Encoder:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.zeros((1, 100, 4)))

    model = Model()
    model.model = SimpleNamespace(encoder=Encoder())
    bundle = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=Tokenizer()),
                             sample_rate=16000)
    monkeypatch.setattr(runner, "load_audio", lambda path, rate: np.ones(32000))
    monkeypatch.setattr(runner, "batch_model_inputs", lambda bundle, paths:
                        {"input_features": torch.zeros((1, 80, 3000)),
                         "attention_mask": torch.ones((1, 3000))})
    captures = []

    def replay(bundle, encoded, prompt, content, *, attention):
        captures.append((list(prompt), list(content), attention))
        logits = torch.tensor([[4., 0., -5.]] * (len(prompt) + len(content)))
        if not attention:
            logits[:, 0], logits[:, 1] = 0., 4.
            return logits, None
        heads = torch.zeros((1, len(prompt) + len(content), 100))
        heads[0, len(prompt) - 1, 10] = 1.
        heads[0, len(prompt), 70] = 1.
        heads[0, len(prompt) + 1, 90] = 1.
        return logits, heads

    monkeypatch.setattr(runner, "full_replay", replay)
    monkeypatch.setattr(runner, "native_lid", lambda bundle, waveform, ids: {3: .9, 2: .1})
    conditions = {"cB": [1, 2, 4, 5], "cM": [1, 2, 4, 5],
                  "cE": [1, 3, 4, 5], "language_token_ids": [3, 2]}
    result = runner.inference_utterance(
        bundle, audio_path="real", donor_audio_path="donor", audio_sha256="sha",
        conditions=conditions, partition=PART, null_probs={3: .5, 2: .5},
        language_ids=(2, 3),
        manifest={"current_utterance_id": "u", "manifest_hash": "h",
                  "decode": {"max_new_tokens": 200}})
    assert captures == [([1, 2, 4, 5], [14, 15], True),
                        ([1, 3, 4, 5], [14, 15], False)]
    assert [r["prefix_ids"] for r in result["rows"]] == [[], [14], [14, 15]]
    assert [r["prediction_query_index"] for r in result["rows"]] == [3, 4, 5]
    assert result["rows"][2]["is_eos_slot"]
    assert all(r["fallback_reason"] is None for r in result["rows"])
    assert all(r["gate"]["g_cf"] <= r["gate"]["g_old"] for r in result["rows"])


def _valid_row():
    b = {"P_M": .65, "P_E": .05, "Q": .7, "R": .6}
    e = {"P_M": .3, "P_E": .1, "Q": .4, "R": .2}
    support = {"E": .8}
    return {"schema": VERSION_R2, "utterance_id": "u", "logical_position": 0,
            "prefix_ids": [], "fallback_reason": None, "baseline": b, "ecf": e,
            "local_support": support, "gate": gate_values(support, b, e),
            "window": {"start_frame": 0, "end_frame": 50}, "manifest_hash": "h"}


def test_schema_roundtrip_and_fallback_zero(tmp_path):
    row = _valid_row()
    validate_gate_row(row)
    dest = tmp_path / "r.json"
    atomic_json(dest, row)
    reloaded = json.loads(dest.read_text())
    assert reloaded == row and digest(reloaded) == digest(row)
    for reason in ("mid_character", "localizer_fail", "baseline_provider_fail",
                   "ecf_provider_fail", "nonfinite_signal"):
        fallback = {**row, "fallback_reason": reason,
                    "gate": {"D": 0.0, "g_old": 0.0, "g_cf": 0.0}}
        validate_gate_row(fallback)
        with pytest.raises(ValueError):
            validate_gate_row({**fallback, "gate": row["gate"]})
    with pytest.raises(ValueError):
        validate_gate_row({**row, "gate": {**row["gate"], "D": .5}})


def test_partition_is_frozen_and_exactly_reused():
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    part = tokenizer_partition(processor.tokenizer)
    assert [len(part[k]) for k in ("matrix_ids", "embedded_ids", "ambiguous_ids")] == [1667, 37858, 12341]
    assert part["hash"] == tokenizer_partition(processor.tokenizer)["hash"]
    assert not (set(part["matrix_ids"]) & set(part["embedded_ids"]))


def test_evaluator_offset_and_deletion_gap_are_separate_from_inference():
    class Tok:
        def decode(self, ids, **kwargs):
            return "".join({1: "你", 2: "它"}[i] for i in ids)
    positions = unit_to_token_positions(Tok(), [1, 2], "你它")
    assert positions == [0, 1]
    records, labels = evaluator_labels("你hello好", "你它", positions,
                                       eos_slot=None, heard_sec=5., ctc_by_index={})
    deletion = [r for r in records if r["surface"] == "hello"]
    assert deletion and deletion[0]["stratum"] == "EN-deletion-slot"
    assert deletion[0]["position"] == 1


def test_alignment_collision_is_counted_unalignable():
    # The deleted English unit and the following correct Han unit share one query.
    records, labels = evaluator_labels("hello你", "你", [0], eos_slot=None,
                                       heard_sec=5., ctc_by_index={})
    assert labels[0] == "unalignable:stratum_collision"
    assert all(r["position"] is None for r in records)
    assert all(r["reason"] == "unalignable:stratum_collision" for r in records)


def test_sparse_panel_metrics_are_serializable_and_do_not_pass_by_nan(tmp_path):
    assert auc_ap([.8], []) == (None, None)
    rows = [{"dialogue_id": "d1", "stratum": "EN-confusion", "g_cf": .8,
             "g_old": .9}]
    intervals = cluster_intervals(rows, {"EN-confusion"}, {"ZH-correct"},
                                  ("g_cf", "g_old"), replicates=8, seed=7)
    assert intervals["field_intervals"]["g_cf"] == {"auroc": None, "auprc": None}
    atomic_json(tmp_path / "metrics.json", intervals)
    assert _json_safe({"mer": float("nan"), "n": 0}) == {"mer": None, "n": 0}


def test_whisper_config_has_frozen_alignment_heads_and_no_steering_api():
    config = json.loads(Path("/mnt/data/tungnx/whisper-large-v3/generation_config.json").read_text())
    assert len(config["alignment_heads"]) == 10
    source = inspect.getsource(runner)
    assert "DecoderPostCrossAttnInterventionHook" not in source
    assert "apply_steering" not in source


def test_whisper_full_replay_query_equals_prefix_only_query():
    """Future B0 tokens in an offline replay cannot affect an earlier query."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16,
                        encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=10, max_target_positions=20,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        decoder_start_token_id=1,
                        dropout=0.0, attention_dropout=0.0)
    model = WhisperForConditionalGeneration(cfg).eval()
    model.set_attn_implementation("eager")
    enc = BaseModelOutput(last_hidden_state=torch.randn((1, 10, 16)))
    prefix = torch.tensor([[1, 2, 3]])
    future = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        short = model(encoder_outputs=enc, decoder_input_ids=prefix,
                      use_cache=False, output_attentions=True, return_dict=True)
        full = model(encoder_outputs=enc, decoder_input_ids=torch.cat((prefix, future), dim=1),
                     use_cache=False, output_attentions=True, return_dict=True)
    assert torch.allclose(short.logits[0, -1], full.logits[0, 2], atol=1e-5)
    assert torch.allclose(short.cross_attentions[0][0, :, -1],
                          full.cross_attentions[0][0, :, 2], atol=1e-5)


# ---- Added after recovery of the interrupted session -------------------------------------

from csasr.inference_cf.core_r2 import terminated_by_eos
from experiments.inference_cf_p0_r2_evaluate import auc_status, estimable, permutation_diagnostic

CONFIG = json.loads(Path("configs/inference_cf/p0_r2_repairability.json").read_text())
CRIT = CONFIG["feasibility"]


def test_ls_b_strictly_monotone_for_positive_margin_and_bounded():
    null = (.3099, .0199)
    es = [local_support(p, 1 - p, *null)["E"] for p in (.95, .97, .99, .999, .9999)]
    assert all(0 < a < b < 1 for a, b in zip(es, es[1:]))
    assert local_support(0., 1., *null)["E"] == 0.0          # negative evidence -> exactly zero


def test_conflict_is_float32_log_softmax_of_given_logits():
    logits = torch.tensor([3.1, 2.9, -1.0, 0.5], dtype=torch.bfloat16)
    part = {"matrix_ids": [0], "embedded_ids": [1, 3], "ambiguous_ids": [2]}
    got = conflict_from_logits(logits, part)
    ref = torch.softmax(logits.float(), dim=-1).double()
    assert got["P_M"] == pytest.approx(float(ref[0]), abs=1e-6)
    assert got["P_E"] == pytest.approx(float(ref[1] + ref[3]), abs=1e-6)
    assert got["R"] == pytest.approx(max(0., float(ref[0] - ref[1] - ref[3])), abs=1e-6)


def test_eos_slot_exists_when_generate_strips_eos_but_not_at_cap():
    # Installed Whisper generate() strips the final EOS from the returned sequence.
    assert terminated_by_eos([14, 15], [14, 15], 99, 200)
    assert terminated_by_eos([14, 15, 99], [14, 15], 99, 200)
    capped = [14] * 200
    assert not terminated_by_eos(capped, capped, 99, 200)
    assert terminated_by_eos([], [], 99, 200)                  # immediate EOS -> one EOS slot


def _fake_runner_setup(monkeypatch, sequence, lid_counter):
    from types import SimpleNamespace

    class Tokenizer:
        eos_token_id = 99
        all_special_ids = [99]

        def convert_ids_to_tokens(self, token_id):
            return {14: "H", 15: "i"}[token_id]

        def decode(self, ids, **kwargs):
            return "".join({14: "H", 15: "i"}[x] for x in ids)

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([sequence])

    class Encoder:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.zeros((1, 100, 4)))

    model = Model()
    model.model = SimpleNamespace(encoder=Encoder())
    bundle = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=Tokenizer()),
                             sample_rate=16000)
    monkeypatch.setattr(runner, "load_audio", lambda path, rate: np.ones(32000))
    monkeypatch.setattr(runner, "batch_model_inputs", lambda bundle, paths:
                        {"input_features": torch.zeros((1, 80, 3000)),
                         "attention_mask": torch.ones((1, 3000))})
    prompts = []

    def replay(bundle, encoded, prompt, content, *, attention):
        prompts.append((list(prompt), list(content)))
        n = len(prompt) + len(content)
        logits = torch.tensor([[4., 0., -5.]] * n)
        if prompt[1] == 3:           # English counterfactual lowers matrix mass
            logits[:, 0], logits[:, 1] = 1., 3.
        if prompt[1] == 7:           # wrong-language diagnostic
            logits[:, 0], logits[:, 1], logits[:, 2] = 1., 0., 3.
        heads = torch.zeros((1, n, 100))
        heads[0, :, 40] = 1.         # every query attends to the same frame -> same window
        return logits, (heads if attention else None)

    def lid(bundle, waveform, ids):
        lid_counter.append(len(waveform))
        return {3: .9, 2: .1}

    monkeypatch.setattr(runner, "full_replay", replay)
    monkeypatch.setattr(runner, "native_lid", lid)
    conditions = {"cB": [1, 2, 4, 5], "cM": [1, 2, 4, 5], "cE": [1, 3, 4, 5],
                  "cX": [1, 7, 4, 5], "wrong_language": "ru", "language_token_ids": [3, 2]}
    return bundle, conditions, prompts


def test_runner_eos_slot_after_stripped_eos_wrong_language_and_lid_cache(monkeypatch):
    calls = []
    bundle, conditions, prompts = _fake_runner_setup(monkeypatch, [14, 15], calls)
    result = runner.inference_utterance(
        bundle, audio_path="real", donor_audio_path="donor", audio_sha256="sha",
        conditions=conditions, partition=PART, null_probs={3: .5, 2: .5}, language_ids=(2, 3),
        manifest={"current_utterance_id": "u", "manifest_hash": "h", "decode": {"max_new_tokens": 200}})
    # Same baseline content for B, Ecf and the diagnostic X branch; prompts differ only at index 1.
    assert [p[1] for p in prompts] == [[14, 15]] * 3
    assert {tuple(p[0][:1] + p[0][2:]) for p in prompts} == {(1, 4, 5)}
    assert result["has_eos"] and not result["eos_in_returned_sequence"]
    assert [r["is_eos_slot"] for r in result["rows"]] == [False, False, True]
    for r in result["rows"]:
        assert r["xcf"]["D_X"] == pytest.approx(max(0., r["baseline"]["R"] - r["xcf"]["conflict"]["R"]))
        assert r["gate"]["g_cf"] == pytest.approx(r["local_support"]["E"] * r["gate"]["D"])
        assert r["gate"]["g_cf"] <= r["gate"]["g_old"] + 1e-12   # X never enters a gate
    # Identical window -> one real and one control LID evaluation, the rest exact cache hits.
    assert result["lid_calls"] == {"real": 1, "control": 1, "cache_hits": 4}
    assert len(calls) == 2


def test_runner_no_eos_slot_when_capped(monkeypatch):
    calls = []
    bundle, conditions, _ = _fake_runner_setup(monkeypatch, [14, 15], calls)
    result = runner.inference_utterance(
        bundle, audio_path="real", donor_audio_path="donor", audio_sha256="sha",
        conditions=conditions, partition=PART, null_probs={3: .5, 2: .5}, language_ids=(2, 3),
        manifest={"current_utterance_id": "u", "manifest_hash": "h", "decode": {"max_new_tokens": 2}})
    assert not result["has_eos"] and result["truncated_at_cap"]
    assert not any(r["is_eos_slot"] for r in result["rows"])


def test_deletion_gap_with_unmapped_next_offset_is_not_moved_to_eos():
    records, _ = evaluator_labels("你hello好", "你好", [0, None], eos_slot=2,
                                  heard_sec=5., ctc_by_index={})
    deletion = [r for r in records if r["surface"] == "hello"][0]
    assert deletion["position"] is None and deletion["reason"] == "unalignable:offset"


def test_not_estimable_is_never_a_pass():
    small = {"auroc": .99, "positive_positions": 29, "negative_positions": 500,
             "positive_dialogues": 12, "negative_dialogues": 20}
    assert not estimable(small, CRIT)
    assert auc_status(small, [.9, 1.], CRIT) == "NOT_ESTIMABLE"
    ok = {**small, "positive_positions": 30}
    assert auc_status(ok, [.6, .9], CRIT) == "PASS"
    assert auc_status(ok, [.49, .9], CRIT) == "FAIL"
    assert auc_status({**ok, "auroc": .69}, [.6, .9], CRIT) == "FAIL"


def test_permutation_diagnostic_is_descriptive_and_deterministic():
    rows = [{"dialogue_id": f"d{i % 12}", "stratum": "EN-confusion" if i % 2 else "ZH-correct",
             "fallback_reason": None, "E": float(i % 2), "D": .5, "g_cf": .5 * (i % 2)}
            for i in range(80)]
    a = permutation_diagnostic(rows, {"EN-confusion"}, {"ZH-correct"}, seed=240924)
    b = permutation_diagnostic(rows, {"EN-confusion"}, {"ZH-correct"}, seed=240924)
    assert a == b and set(a) == {"shuffle_E", "shuffle_D"}
    assert rows[1]["E"] == 1.0                                 # inputs are not mutated


def test_config_and_partition_hash_reproducible():
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    assert tokenizer_partition(processor.tokenizer)["hash"] == CONFIG["tokenizer_partition_hash"]
    assert digest(CONFIG) == digest(json.loads(json.dumps(CONFIG)))
    assert CONFIG["wrong_language_diagnostic"] not in ("en", "zh")
    assert CONFIG["bootstrap_replicates"] == 2000 and CONFIG["seed"] == 240924
    assert CRIT["auc_min"] == .7 and CRIT["min_positions_per_stratum"] == 30
    assert CRIT["min_dialogues_per_stratum"] == 10


def test_ecf_failure_zeroes_only_counterfactual_gate_and_preserves_old(monkeypatch):
    calls = []
    bundle, conditions, _ = _fake_runner_setup(monkeypatch, [14, 15], calls)
    base_replay = runner.full_replay

    def replay(bundle_, encoded, prompt, content, *, attention):
        logits, heads = base_replay(bundle_, encoded, prompt, content, attention=attention)
        if prompt[1] == 3:                       # corrupt the Ecf branch at query of t=1
            logits[len(prompt), :] = float("nan")
        return logits, heads

    monkeypatch.setattr(runner, "full_replay", replay)
    result = runner.inference_utterance(
        bundle, audio_path="real", donor_audio_path="donor", audio_sha256="sha",
        conditions=conditions, partition=PART, null_probs={3: .5, 2: .5}, language_ids=(2, 3),
        manifest={"current_utterance_id": "u", "manifest_hash": "h", "decode": {"max_new_tokens": 200}})
    bad, good = result["rows"][1], result["rows"][0]
    assert bad["fallback_reason"] == "ecf_provider_fail" and bad["eligibility_status"] == "cf_ineligible"
    assert bad["gate"]["D"] == 0 and bad["gate"]["g_cf"] == 0
    assert bad["gate"]["g_old"] == pytest.approx(bad["local_support"]["E"] * bad["baseline"]["R"])
    assert bad["gate"]["g_old"] > 0
    assert good["eligibility_status"] == "eligible" and good["fallback_reason"] is None
    assert all(r["same_prefix_validation"] and r["replay_mode"].startswith("full_same_prefix")
               for r in result["rows"])
    # A row claiming Ecf-only fallback cannot silently zero or alter the old gate.
    with pytest.raises(ValueError):
        validate_gate_row({**bad, "gate": {"D": 0.0, "g_old": 0.0, "g_cf": 0.0}})


def test_language_swap_invalidates_kv_cache_so_full_replay_is_required():
    """Changing the language token changes every later decoder state, so KV entries built
    under cB are not valid under cE; the runner therefore replays the full prefix per branch."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=1,
                        decoder_layers=2, encoder_attention_heads=2, decoder_attention_heads=2,
                        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=10,
                        max_target_positions=20, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        decoder_start_token_id=1, dropout=0.0, attention_dropout=0.0)
    model = WhisperForConditionalGeneration(cfg).eval()
    enc = BaseModelOutput(last_hidden_state=torch.randn((1, 10, 16)))
    content = [11, 12, 13]
    with torch.inference_mode():
        b = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([[1, 5, 7] + content]),
                  use_cache=False, output_hidden_states=True, return_dict=True)
        e = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([[1, 6, 7] + content]),
                  use_cache=False, output_hidden_states=True, return_dict=True)
    # Content-position states after layer 1 differ -> cached keys/values would be stale.
    assert not torch.allclose(b.decoder_hidden_states[1][0, 3:], e.decoder_hidden_states[1][0, 3:])
    source = inspect.getsource(runner.full_replay)
    assert "use_cache=False" in source and "past_key_values" not in source
