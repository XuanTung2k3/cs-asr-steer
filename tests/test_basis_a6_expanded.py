import json

import numpy as np
import pytest

from csasr.basis_a6.cache import DirectionCache
from csasr.basis_a6.construction import ConstructionSample, construct_corpus_layer
from csasr.basis_a6.decoding import decode_config, decode_config_hash, local_mask_indices, register_hypothesis, reorder_cache
from csasr.basis_a6.data import adapt_ascend_row
from csasr.basis_a6.directions import dynamic_rank, raw_direction, unique_shared_direction
from csasr.basis_a6.eligibility import common_eligible_manifest, eligibility_table
from csasr.basis_a6.geometry import cross_source_geometry
from csasr.basis_a6.guards import GoldTokenLeakageError, assert_no_external_fixed_direction, make_gold_leakage_trace, validate_decoder_analysis_provenance, validate_local_mask
from csasr.basis_a6.protocol import enumerate_a6_f, enumerate_a6_tt, expected_counts
from csasr.basis_a6.subsets import assert_disjoint, deterministic_stratified_select
from csasr.basis_a6.runners import OracleTTCallbacks, run_oracle_tt_sample, steer_cached_sample


def row(text, id_, split="train", **extra):
    return {"id": id_, "audio": {"array": np.zeros(8), "sampling_rate": 16000},
            "transcription": text, "duration": 1.0, "original_speaker_id": extra.get("speaker", "s1"),
            "session_id": extra.get("session", "x"), "topic": extra.get("topic", "t"), "language": "mixed"}


def test_ascend_adapter_uses_canonical_mixed_segmentation():
    zh = adapt_ascend_row(row("你好 世界", "zh"), split="train")
    en = adapt_ascend_row(row("hello world", "en"), split="train")
    mixed = adapt_ascend_row(row("你好 hello", "mixed"), split="train")
    assert (zh.en_units, zh.zh_units, zh.cs_eligible) == (0, 4, False)
    assert (en.en_units, en.zh_units, en.cs_eligible) == (2, 0, False)
    assert (mixed.en_units, mixed.zh_units, mixed.cs_eligible) == (1, 2, True)
    with pytest.raises(ValueError):
        adapt_ascend_row(row("你好 hello", "test"), split="test")


def test_subset_selection_and_split_disjointness():
    train = [adapt_ascend_row(row("你好 hello", f"tr{i}", speaker=f"s{i%3}"), split="train") for i in range(8)]
    val = [adapt_ascend_row(row("你好吗 hello", f"va{i}", speaker=f"s{i%3}"), split="validation") for i in range(5)]
    a = deterministic_stratified_select(train, 4)
    b = deterministic_stratified_select(val, 3)
    assert [x.utterance_id for x in a] == [x.utterance_id for x in deterministic_stratified_select(train, 4)]
    assert set(x.utterance_id for x in a).isdisjoint(x.utterance_id for x in b)
    assert_disjoint({"source_split": "train", "utterance_ids": [x.utterance_id for x in a]},
                    {"source_split": "validation", "utterance_ids": [x.utterance_id for x in b]})


def test_directions_dynamic_rank_and_no_rank1_fallback():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(3, 7)), rng.normal(size=(2, 7))
    assert dynamic_rank(3, 2, 7) == 2
    us = unique_shared_direction(a, b)
    assert us is not None and us.rank == 2
    assert np.isclose(np.linalg.norm(raw_direction(a, b)), 1)
    assert unique_shared_direction(a[:1], b) is None


def test_fixed_construction_source_is_pooled_once_and_conditioning_is_decoder_only():
    rng = np.random.default_rng(8)
    samples = [ConstructionSample(f"u{i}", rng.normal(size=(4, 6)) + 1, rng.normal(size=(5, 6)),
                                  conditioning_all=rng.normal(size=(3, 6)), conditioning_cs=rng.normal(size=(1, 6)))
               for i in range(3)]
    built = construct_corpus_layer(source="ascend", model="whisper", side="decoder", layer=16, samples=samples)
    assert built["N_utterances"] == 3 and built["n_A"] == 12 and built["n_B"] == 15
    assert set(built["directions"]) == {"raw", "add_unique", "minus_shared", "unique_minus_shared", "conditioning_all", "conditioning_cs"}
    assert all(r["key"][0] == "ascend" for r in built["records"])


def test_cache_key_bundle_hash_and_cross_source_geometry():
    rng = np.random.default_rng(2)
    v = raw_direction(rng.normal(size=(4, 5)), rng.normal(size=(4, 5)))
    cache = DirectionCache()
    for uid in ("u2", "u1"):
        cache.put(model="whisper", dataset="d", utterance_id=uid, decode_analysis_mode="greedy",
                  side="decoder", layer=16, method="raw", alignment_hash="a", n_A=2, n_B=2,
                  rank=None, eligible=True, vector=v)
    h1 = cache.bundle_hash(model="whisper", dataset="d", side="decoder", method="raw", decode_analysis_mode="greedy")
    assert h1.startswith("sha256:")
    rows = cross_source_geometry({("cs_dialogue", "whisper", "decoder", 16, "raw"): v,
                                  ("ascend", "whisper", "decoder", 16, "raw"): -v})
    assert len(rows) == 1 and np.isclose(rows[0]["cosine"], -1)


def test_gold_leakage_guard_and_local_mask():
    trace = make_gold_leakage_trace(utterance_id="u", hypothesis_hash="h", oracle_alignment_hash="a")
    validate_decoder_analysis_provenance(trace)
    bad = dict(trace); bad["sequence_source"] = "gold_reference"
    with pytest.raises(GoldTokenLeakageError):
        validate_decoder_analysis_provenance(bad)
    validate_local_mask([False, True, False], [1], [0, 2])
    with pytest.raises(AssertionError):
        validate_local_mask([True, False], [1], [0])
    with pytest.raises(AssertionError):
        assert_no_external_fixed_direction("cs_dialogue")
    assert_no_external_fixed_direction(None)


def test_tt_runner_uses_same_hypothesis_and_reuses_direction_across_rho():
    calls = []
    callbacks = OracleTTCallbacks(
        baseline_decode=lambda sample, mode: {"hypothesis_hash": "h", "tokens": [1, 2]},
        align_oracle_to_hypothesis=lambda sample, baseline: {"alignment_hash": "a", "local": [1]},
        extract_analysis_states=lambda sample, baseline, side: {16: {"A": np.ones((3, 4)), "B": np.zeros((3, 4))}},
        extract_conditioning_deltas=lambda sample, baseline, side: {16: {"all": np.ones((3, 4)), "cs_positions": [0]}},
        steered_decode=lambda sample, alignment, directions, rho, mode: calls.append((sample, rho, tuple(directions))) or {"rho": rho},
    )
    cache = DirectionCache()
    state = run_oracle_tt_sample(sample="same", model="whisper", dataset="d", utterance_id="u",
                                 decode_mode="greedy", side="decoder", layers=[16], callbacks=callbacks, cache=cache)
    assert state["gold_leakage_trace"]["sequence_source"] == "baseline_hypothesis"
    steer_cached_sample(state, sample="same", callbacks=callbacks, rho=.5, method="raw", side="decoder", layer=16, decode_mode="greedy")
    steer_cached_sample(state, sample="same", callbacks=callbacks, rho=6, method="raw", side="decoder", layer=16, decode_mode="greedy")
    assert [x[1] for x in calls] == [.5, 6]
    assert len([r for r in cache.records() if r.method == "raw"]) == 1


def test_eligibility_tables_and_common_manifest():
    records = [{"model": "m", "dataset": "d", "side": "decoder", "layer": 16, "utterance_id": "u1",
                "method": "add_unique", "n_A": 2, "n_B": 3, "rank": 2, "eligible": True},
               {"model": "m", "dataset": "d", "side": "decoder", "layer": 16, "utterance_id": "u1",
                "method": "minus_shared", "n_A": 2, "n_B": 3, "rank": 2, "eligible": True},
               {"model": "m", "dataset": "d", "side": "decoder", "layer": 16, "utterance_id": "u1",
                "method": "unique_minus_shared", "n_A": 2, "n_B": 3, "rank": 2, "eligible": True}]
    assert eligibility_table(records)[0]["N_total"] == 3
    assert common_eligible_manifest(records)["groups"]["m|d|decoder|16"] == ["u1"]


def test_full_matrix_counts_exactly():
    assert expected_counts() == {"a6_f": 46720, "a6_tt": 23360, "total": 70080, "baselines": 16, "one_regime": 23360}
    assert len(enumerate_a6_f()) == 46720
    assert len(enumerate_a6_tt()) == 23360
    assert len({tuple(sorted(r.items())) for r in enumerate_a6_f()}) == 46720


def test_frozen_decode_modes_and_beam_cache_helpers():
    assert decode_config("whisper", "greedy")["num_beams"] == 1
    assert decode_config("whisper", "official_standard")["num_beams"] == 5
    qg, qo = decode_config("qwen3_asr_1p7b", "greedy"), decode_config("qwen3_asr_1p7b", "official_standard")
    assert {k: v for k, v in qg.items() if k not in {"mode", "provenance_equivalent_to"}} == {k: v for k, v in qo.items() if k not in {"mode", "provenance_equivalent_to"}}
    assert decode_config_hash("whisper", "greedy").startswith("sha256:")
    assert register_hypothesis(model="whisper", dataset="d", mode="greedy", hypothesis=[1])["hypothesis_hash"].startswith("sha256:")
    assert reorder_cache({"past": ["a", "b"]}, [1, 0])["past"] == ["b", "a"]
    assert (replicate := local_mask_indices([False, True], batch_index=2, hypothesis_index=3))
    assert replicate == [(2, 3, 1)]
