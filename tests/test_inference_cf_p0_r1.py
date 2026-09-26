"""Focused CPU tests for P0-R1 (K=3 short-continuation crossed support)."""
import math

import pytest
import torch

from csasr.inference_cf.core import VERSION, permutation
from csasr.inference_cf.core_r1 import (K, VERSION_R1, longest_common_prefix, sequence_logprob,
                                        support_k3, truncate_continuation, digest)
import experiments.inference_cf_p0_r1 as r1


# ---- pure contracts ---------------------------------------------------------

def test_truncate_continuation_eos_cap_and_no_special():
    assert truncate_continuation([4, 5, 9], 9, 3) == [4, 5]      # EOS truncates
    assert truncate_continuation([9, 5, 6], 9, 3) == []          # immediate EOS -> empty
    assert truncate_continuation([4, 5, 6, 7], 9, 3) == [4, 5, 6]  # cap at K
    assert truncate_continuation([4], 9, 3) == [4]
    assert K == 3


def test_longest_common_prefix():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [3, 4]) == 0
    assert longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_sequence_logprob_token_average():
    assert sequence_logprob([-1.0, -2.0, -3.0]) == -2.0
    assert sequence_logprob([-0.5]) == -0.5
    with pytest.raises(ValueError):
        sequence_logprob([])
    with pytest.raises(ValueError):
        sequence_logprob([-1.0, math.nan])


def test_support_k3_full_partial_and_noncollision_algebra():
    s = {"sE_yE": -1., "sE_yM": -3., "sM_yE": -4., "sM_yM": -2.}
    # non-collision: q_cross = .5[(sE_yE-sE_yM)+(sM_yM-sM_yE)] = .5[2+2] = 2
    x = support_k3(s, [4, 5, 6], [4, 7, 8])
    assert x["collision"] is False and x["q_cross"] == 2.0 and x["q_own"] == 1.0 and x["lcp"] == 1
    # partial collision (shared prefix, not identical) is still scored, not a collision
    p = support_k3(s, [4, 5, 6], [4, 5, 9])
    assert p["collision"] is False and p["q_cross"] == 2.0 and p["lcp"] == 2
    # full collision -> uninformative q_cross null, lcp == length
    f = support_k3(s, [4, 5, 6], [4, 5, 6])
    assert f["collision"] is True and f["q_cross"] is None and f["lcp"] == 3
    with pytest.raises(ValueError):
        support_k3({**s, "sM_yE": math.nan}, [4], [5])


# ---- executor mechanics with a deterministic fake forward -------------------

def _fake_forward_factory():
    V, EOS = 12, 9
    # per-position peak token: pos2->4, pos3->5, pos4->EOS(9); others->0
    peaks = {2: 4, 3: 5, 4: 9}

    def fake_forward(bundle, enc, tokens, capture=False):
        logits = torch.full((len(tokens), V), -10.0)
        for i in range(len(tokens)):
            logits[i, peaks.get(i, 0)] = 5.0
        last_logp = torch.log_softmax(logits[-1], dim=-1)
        h = [float(len(tokens))] * 4 if capture else None
        return last_logp, h, logits
    return fake_forward, EOS


def test_k3_continuation_serialization_and_capture(monkeypatch):
    fake, EOS = _fake_forward_factory()
    monkeypatch.setattr(r1, "forward_logits", fake)
    counter = [0]
    cand, state = r1.continue_k3(None, None, [1, 2], [3], EOS, counter, capture=True)
    assert cand == [4, 5]              # greedy [4,5,EOS] -> truncated
    assert state == [3.0, 3.0, 3.0, 3.0]   # captured at step 0 (input length 3)
    assert counter[0] == 3            # three forwards then EOS break


def test_autoregressive_cross_scoring_positions_and_alignment(monkeypatch):
    fake, EOS = _fake_forward_factory()
    monkeypatch.setattr(r1, "forward_logits", fake)
    counter = [0]
    # candidate spanning all three continuation positions; prompt len2, shared len1 -> base 3
    step_logprobs, align = r1.score_sequence(None, None, [1, 2], [3], [4, 5, 6], counter)
    assert align["continuation_query_indices"] == [2, 3, 4]      # base+j-1 for j=0,1,2
    assert align["prompt_length"] == 2 and align["content_prefix_length"] == 1
    assert align["candidate_length"] == 3 and align["input_length"] == 6
    assert align["decoder_mask"] == [1] * 6
    assert align["cache_positions_full_replay"] == list(range(6))
    assert align["cache_used"] is False and align["beam_lineage"] == "greedy:0"
    # logits[2] peaks at 4, logits[3] at 5 -> those steps high; logits[4] peaks at 9 not 6 -> low
    assert step_logprobs[0] == pytest.approx(step_logprobs[1])
    assert step_logprobs[2] < step_logprobs[0]
    assert sequence_logprob(step_logprobs) == pytest.approx(sum(step_logprobs) / 3)


def test_cross_scores_four_components(monkeypatch):
    fake, EOS = _fake_forward_factory()
    monkeypatch.setattr(r1, "forward_logits", fake)
    counter = [0]
    conditions = {"c0": [1, 2, 0, 0], "cM": [1, 2, 0, 0], "cE": [1, 3, 0, 0]}
    scores, steps, aligns = r1.cross_scores(None, None, conditions, [3], [4, 5], [4, 6], counter)
    assert set(scores) == {"sE_yE", "sE_yM", "sM_yE", "sM_yM"}
    assert set(steps) == set(aligns) == set(scores)
    assert counter[0] == 4           # four teacher-forced scoring forwards


def test_no_reference_in_inference_api():
    import inspect
    names = inspect.signature(r1.inference_row).parameters
    assert not any(x in names for x in ("reference", "true_future", "oracle", "stratum", "poi"))


def test_deterministic_audio_permutation():
    p = permutation(["a", "b", "c"])
    assert p == {"a": "b", "b": "c", "c": "a"}
    assert all(k != v for k, v in p.items())


def test_manifest_version_and_hash_separation():
    assert VERSION_R1 != VERSION and VERSION_R1 == "p0r1_cross_k3_v1"
    base = {"panel_hash": "sha256:same", "conditions": {"x": 1}}
    p0 = {**base, "schema": VERSION, "candidate_k": 1}
    p0r1 = {**base, "schema": VERSION_R1, "candidate_k": 3}
    assert digest(p0) != digest(p0r1)   # same panel, different scientific version -> different hash
