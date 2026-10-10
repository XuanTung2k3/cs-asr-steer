"""Focused DG-08 preflight tests: frozen finalists, seed handling, exact site,
LoRA budget, canonical D-test scoring, lock guard, decoder config, bootstrap."""
from __future__ import annotations

import json

import pytest
import torch

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
from csasr.steering.dg07_variants import (
    ExactLayerQvLoRA,
    GlobalVector,
    lora_parameter_count,
    select_lora_rank,
)
from csasr.utils.config import load_config

import experiments.dg07_baselines_ablations as dg07
import experiments.dg08_dtest_eval as dg08
import experiments.dg08_lock as lock
import experiments.dg08_stats as stats


# --- frozen protocol config ----------------------------------------------------

def test_config_freezes_seeds_and_decode_regimes():
    cfg = load_config("configs/dg08_locked_eval.yaml")
    assert cfg["seeds"] == [13, 42, 73]
    assert cfg["decode"]["greedy"]["num_beams"] == 1
    assert cfg["decode"]["greedy"]["temperature"] == 0.0
    assert cfg["decode"]["beam5"]["num_beams"] == 5
    assert cfg["bootstrap"]["unit"] == "dialogue"
    assert cfg["bootstrap"]["repetitions"] == 2000
    assert cfg["bootstrap"]["seed"] == 42


def test_lock_seed42_hashes_match_frozen_finalists():
    assert lock.EXPECTED_SEED42_SHA["F4_MSTAR"].startswith("sha256:b9c45a1e")
    assert lock.EXPECTED_SEED42_SHA["F2_SALSA"].startswith("sha256:03cc0bf6")
    assert lock.EXPECTED_SEED42_SHA["F3_LORA"].startswith("sha256:7861cdf8")
    for system in ("F2_SALSA", "F3_LORA", "F4_MSTAR"):
        assert set(lock.CHECKPOINTS[system]) == {13, 42, 73}


# --- seed handling -------------------------------------------------------------

def test_lora_init_is_seed_dependent_and_deterministic(tiny_bundle):
    layer = tiny_bundle.model.model.decoder.layers[0]

    def build():
        return ExactLayerQvLoRA(layer, rank=2, alpha=2.0, target_layer=0)

    dg07.set_seed(13)
    a = build().q_A.detach().clone()
    dg07.set_seed(13)
    b = build().q_A.detach().clone()
    dg07.set_seed(73)
    c = build().q_A.detach().clone()
    assert torch.allclose(a, b)              # same seed -> identical init
    assert not torch.allclose(a, c)          # different seed -> different init


def test_dg07_validate_config_accepts_only_expected_seed():
    cfg = load_config("configs/dg07_baselines_ablations.yaml")
    cfg = dict(cfg); cfg["training"] = {**cfg["training"], "seed": 13}
    dg07.validate_config(cfg, "LB1_SALSA_EXACT_GLOBAL", expected_seed=13)
    with pytest.raises(ValueError, match="seed must be"):
        dg07.validate_config(cfg, "LB1_SALSA_EXACT_GLOBAL", expected_seed=42)


# --- LoRA matched budget -------------------------------------------------------

def test_lora_rank_and_budget_are_frozen():
    rank, count = select_lora_rank(43651, in_features=1280, out_features=1280, target_modules=2)
    assert (rank, count) == (9, 46080)
    assert lora_parameter_count(9, in_features=1280, out_features=1280) == 46080


def test_lora_targets_q_and_v_only(tiny_bundle):
    layer = tiny_bundle.model.model.decoder.layers[0]
    module = ExactLayerQvLoRA(layer, rank=2, alpha=2.0, target_layer=0)
    assert set(dict(module.named_parameters())) == {"q_A", "q_B", "v_A", "v_B"}


# --- exact-site SALSA + M* action ---------------------------------------------

def test_salsa_global_vector_uses_exact_site(tiny_bundle):
    torch.manual_seed(0)
    module = GlobalVector(tiny_bundle.d_model)
    with torch.no_grad():
        module.vector += 0.1
    features = torch.randn(1, 8, 100)
    ids = torch.tensor([[1, 5, 7, 9]])
    hook = DecoderPostCrossAttnInterventionHook(
        tiny_bundle, 1, direction=None, alpha=1.0, scale=1.0, action_fn=module.action,
        num_forced_prefix=0, norm_preserve=True, mode="steer", record=True)
    with hook, torch.inference_mode():
        out = tiny_bundle.model(input_features=features, decoder_input_ids=ids, use_cache=False)
    assert out.logits.shape[0] == 1
    assert hook.steered_calls == 1


def test_build_system_baseline_and_fixed_dir(monkeypatch):
    # baseline needs no checkpoint or basis
    spec = dg08.build_system(bundle=type("B", (), {"d_model": 16})(), basis=torch.zeros(16, 2),
                             s_l=8.9, name="F0_FROZEN_WHISPER", checkpoint=None)
    assert spec["mode"] == "baseline"


# --- canonical D-test scoring (text-only, outside-harm N/A) ---------------------

def test_score_dtest_validates_and_marks_outside_harm_unavailable():
    ids = ["u1", "u2"]
    refs = {"u1": "hello world", "u2": "谢谢 你"}
    base = {"u1": "hello word", "u2": "谢谢 你"}
    method = {"u1": "hello world", "u2": "谢谢 你"}
    prov = {"model_id": "m", "model_revision": None, "decode": {"num_beams": 1},
            "test_lock_commit": "abc"}
    result = dg08.score_dtest(refs, base, method, ids, system="F4_MSTAR", seed=13,
                              regime="greedy", provenance=prov, audit={"applicability": "x"})
    assert result["schema_version"] == "result_v1"
    assert result["metrics"]["outside_harm"] is None
    assert result["metrics"]["outside_harm_available"] is False
    assert "net_correction_utility" in result["metrics"]
    assert result["seed"] == 13
    assert result["data_role"] == "D-test"


def test_resolve_targets_maps_systems_and_seeds():
    fake_lock = {"checkpoints": {
        "F2_SALSA": {"13": {"checkpoint": "a", "sha256": "sha256:x"}},
    }}
    tg = dg08.resolve_targets(["F0_FROZEN_WHISPER", "F2_SALSA:13"], fake_lock)
    assert tg[0] == {"system": "F0_FROZEN_WHISPER", "seed": None, "checkpoint": None}
    assert tg[1] == {"system": "F2_SALSA", "seed": 13, "checkpoint": "a"}


# --- lock guard ----------------------------------------------------------------

def test_build_lock_rejects_wrong_seed42_hash(tmp_path, monkeypatch):
    fake = tmp_path / "ck.pt"
    fake.write_bytes(b"not-the-frozen-checkpoint")
    monkeypatch.setitem(lock.CHECKPOINTS, "F2_SALSA",
                        {13: str(fake), 42: str(fake), 73: str(fake)})
    cfg = load_config("configs/dg08_locked_eval.yaml")
    with pytest.raises(RuntimeError, match="seed 42 checkpoint hash"):
        lock.build_lock(cfg)


# --- bootstrap sufficient statistics -------------------------------------------

def test_corpus_from_components_matches_micro_average():
    from csasr.evaluation import canonical
    refs = ["hello world", "谢谢 你 好"]
    base = ["hello world", "谢谢 你 好"]
    hyp = ["hello word", "谢谢 你 好"]
    ids = ["a", "b"]
    rmap = {"a": refs[0], "b": refs[1]}; bmap = {"a": base[0], "b": base[1]}
    hmap = {"a": hyp[0], "b": hyp[1]}
    _, tot = stats._arm_components(ids, rmap, bmap, hmap)
    corpus = stats._corpus_from_components(tot)
    direct = canonical.corpus_metrics(refs, hyp)
    assert corpus["mer"] == pytest.approx(direct["mer"], abs=1e-9)
    assert corpus["pier"] == pytest.approx(direct["pier"], abs=1e-9)


def test_dialogue_aggregate_reproduces_full_corpus():
    ids = ["a", "b", "c"]
    rmap = {"a": "hello world", "b": "谢谢 你", "c": "good morning"}
    bmap = dict(rmap); hmap = {"a": "hello word", "b": "谢谢 你", "c": "good morning"}
    dialogues = {"a": "D1", "b": "D1", "c": "D2"}
    per_utt, tot = stats._arm_components(ids, rmap, bmap, hmap)
    dstats = stats._dialogue_aggregate(per_utt, dialogues)
    full = stats._metric_from_dialogues(dstats, ["D1", "D2"], "mer")
    assert full == pytest.approx(stats._corpus_from_components(tot)["mer"], abs=1e-9)
