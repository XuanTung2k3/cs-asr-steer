import json
import math
from pathlib import Path

import pytest
from csasr.inference_cf.core import (aligned_input, atomic_json, cache_key,
                                      condition_tokens, conditions_identical, generated_content,
                                      digest, permutation, support, validated_row)


class Tok:
    def convert_tokens_to_ids(self, names):
        return [{"<|startoftranscript|>": 1, "<|zh|>": 2, "<|en|>": 3,
                 "<|transcribe|>": 4, "<|notimestamps|>": 5}[x] for x in names]


class Proc:
    tokenizer = Tok()


def test_conditions_are_exact_and_baseline_equals_matrix():
    c = condition_tokens(Proc())
    assert c == {"c0": [1, 2, 4, 5], "cM": [1, 2, 4, 5], "cE": [1, 3, 4, 5]}
    assert conditions_identical(c)
    assert not conditions_identical({**c, "cM": c["cE"]})


def test_content_alignment_first_and_later_position():
    c = condition_tokens(Proc())
    for content in ([], [41, 42]):
        ins = [aligned_input(c[k], content) for k in c]
        assert all(x[-len(content):] == content for x in ins) if content else all(len(x)==4 for x in ins)
        assert len(set(len(x) for x in ins)) == 1


def test_hf_generate_omits_forced_prompt_regression():
    assert generated_content([101, 102, 99, 103], 99) == [101, 102]
    assert generated_content([101, 102], 99) == [101, 102]
    assert generated_content([], 99) == []


def test_score_algebra_collision_finite_and_candidate_parity():
    s = {"sE_yE": -1., "sE_yM": -3., "sM_yE": -4., "sM_yM": -2.}
    x = support(s, 10, 11)
    assert x == {"collision": False, "q_own": 1., "q_cross": 2.}
    assert support(s, 10, 10)["q_cross"] is None
    with pytest.raises(ValueError):
        support({**s, "sM_yE": math.nan}, 10, 11)


def test_no_reference_in_inference_api():
    import inspect
    from experiments.inference_cf_p0 import inference_row
    names = inspect.signature(inference_row).parameters
    assert not any(x in names for x in ("reference", "true_future", "oracle", "stratum", "poi"))


def test_shuffle_cache_and_resume(tmp_path):
    p = permutation(["a", "b", "c"])
    assert p == {"a": "b", "b": "c", "c": "a"}
    assert all(k != v for k,v in p.items())
    kwargs = dict(model_revision="r", audio_sha256="a", condition="cE", prompt_tokens=[1,3],
                  content_prefix=[42], logical_position=1, layer=24, site="site",
                  beam_lineage="greedy:0", scorer_version="v1", decode_config={"beam":1})
    h = cache_key(**kwargs)
    for field, new in (("model_revision", "r2"), ("audio_sha256", "b"), ("condition", "cM"),
                       ("prompt_tokens", [1,2]), ("content_prefix", [43]),
                       ("logical_position", 2), ("layer", 16), ("site", "other"),
                       ("beam_lineage", "beam:1"), ("scorer_version", "v2"),
                       ("decode_config", {"beam":5})):
        assert cache_key(**{**kwargs, field:new}) != h
    f = tmp_path / "row.json"
    atomic_json(f, {"identity":"a", "manifest_hash":"h"})
    assert validated_row(f, "a", "h")
    with pytest.raises(ValueError): validated_row(f, "a", "other")
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_hash_canonical_and_schema():
    assert digest({"a":1,"b":2}) == digest({"b":2,"a":1})
    with pytest.raises(ValueError):
        aligned_input([], [1])
