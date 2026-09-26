"""Cached steering path: equivalence to the accepted P1 replay, isolation, identity, policies."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import experiments.inference_cf_cached as cached
import experiments.inference_cf_p1 as p1
from csasr.lss.sites import assert_no_site_hooks
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, encoded, tiny_bundle  # noqa: E402

COND = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}


def _run(bundle, monkeypatch, **kw):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    args = dict(waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(),
                conditions=COND, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                layer=24, alpha=1.0, max_new_tokens=8, num_forced_prefix=4)
    args.update(kw)
    return cached.cached_decode(bundle, **args)


def _replay(bundle, monkeypatch, alpha):
    monkeypatch.setattr(p1, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    return p1.decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(),
                     conditions=COND, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                     alpha=alpha, max_new_tokens=8, num_forced_prefix=4)


def test_zero_dose_cached_identity_and_matched_cached_baseline(monkeypatch):
    b = tiny_bundle()
    r = _run(b, monkeypatch, alpha=0.0)
    assert all(s["f3_bitwise_equals_b"] and s["steered_next"] == s["unsteered_next"]
               and not s["edit_applied"] for s in r["steps"])
    g = cached.cached_greedy(b, encoded(), CB, max_new_tokens=8)
    assert r["tokens"] == g["tokens"] and r["terminated"] == g["terminated"]
    assert r["lineage_ok"] and r["distinct_caches"]
    assert_no_site_hooks(b)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_cached_equals_accepted_replay_decode_fp32(monkeypatch, alpha):
    b = tiny_bundle()
    c = _run(b, monkeypatch, alpha=alpha)
    r = _replay(b, monkeypatch, alpha)
    assert c["tokens"] == r["tokens"] and c["terminated"] == r["terminated"]
    for sc, sr in zip(c["steps"], r["steps"]):
        assert sc["query_index"] == sr["query_index"] and sc["fallback_reason"] == sr["fallback_reason"]
        assert sc["g"] == pytest.approx(sr["g"], abs=1e-6)
        assert sc["direction_norm"] == pytest.approx(sr["direction_norm"], rel=1e-5)
        assert sc["edit_applied"] == sr["edit_applied"]
        if sc["edit_applied"]:
            assert sc["hook_audit"]["edit_norm"] == pytest.approx(sr["hook_audit"]["edit_norm"], rel=1e-4)
            assert sc["hook_audit"]["abs_pos"] == sr["hook_audit"]["abs_pos"]
    if alpha:
        assert any(s["edit_applied"] for s in c["steps"])


def test_branch_isolation_steered_cache_never_touches_B_or_E():
    b = tiny_bundle()
    enc = encoded()
    B = cached.Branch(b, enc, CB, "B")
    S = cached.Branch(b, enc, CB, "S")
    B.step(CB, capture_layer=24)
    S.step(CB)
    before = [t.clone() for layer in B.cache.self_attention_cache.layers for t in (layer.keys, layer.values)]
    d = torch.nn.functional.normalize(torch.randn(16), dim=0)
    hook = cached._edit_hook(b, 24, 4, 1.0, 0.9, d, 4)
    S.step([9], hook=hook)
    B.step([9], capture_layer=24)
    after = [t for layer in B.cache.self_attention_cache.layers for t in (layer.keys, layer.values)]
    assert all(torch.equal(x, y[..., :x.shape[-2], :]) for x, y in zip(before, after))
    assert hook.records[-1].edit_norm > 0 and hook.records[-1].abs_pos == 4
    assert B.cache is not S.cache and B.positions == S.positions == [0, 1, 2, 3, 4]
    assert_no_site_hooks(b)


def test_gate_policies_dose_maps_and_direction_sign(monkeypatch):
    b = tiny_bundle()
    base = _run(b, monkeypatch)
    for policy in cached.GATE_POLICIES:
        r = _run(b, monkeypatch, gate_policy=policy)
        for s in r["steps"]:
            if s["fallback_reason"] is None:
                comp = {"ER": s["g_ER"], "one": 1.0, "E": s["local_support"]["E"],
                        "R": s["baseline"]["R"],
                        "cf": s["local_support"]["E"] * max(0.0, s["baseline"]["R"] - (s["ecf"] or {"R": 0})["R"])}
                assert s["g"] == pytest.approx(comp[policy])
    sq = _run(b, monkeypatch, dose="sqrt")
    s0 = [s for s in sq["steps"] if s["fallback_reason"] is None][0]
    assert s0["dose"] == pytest.approx(np.sqrt(s0["g"]))
    assert cached.dose_map(0.0, "sqrt") == 0.0 and cached.dose_map(.25, "sqrt") == .5
    neg = _run(b, monkeypatch, direction_sign=-1.0)
    e_pos = [s for s in base["steps"] if s["edit_applied"]]
    e_neg = [s for s in neg["steps"] if s["edit_applied"]]
    assert e_pos and e_neg
    # Same first edited step, dose and pre-edit state; NormPreserve makes realized norms for
    # +d and -d differ unless h is orthogonal to d, so compare the edited outcome instead.
    assert e_neg[0]["t"] == e_pos[0]["t"] and e_neg[0]["dose"] == pytest.approx(e_pos[0]["dose"])
    assert e_neg[0]["hook_audit"]["pre_norm"] == pytest.approx(e_pos[0]["hook_audit"]["pre_norm"])
    assert e_neg[0]["steered_top2_margin"] != e_pos[0]["steered_top2_margin"]
    with pytest.raises(ValueError):
        _run(b, monkeypatch, gate_policy="min")
    with pytest.raises(ValueError):
        cached.dose_map(.5, "square")


def test_positions_monotonic_and_prefix_lineage(monkeypatch):
    b = tiny_bundle()
    r = _run(b, monkeypatch, max_new_tokens=6)
    qs = [s["query_index"] for s in r["steps"]]
    assert qs == list(range(3, 3 + len(qs)))                 # no skipped/reset position
    assert r["lineage_ok"]
