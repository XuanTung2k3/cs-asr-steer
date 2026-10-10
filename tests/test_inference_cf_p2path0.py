"""P2-PATH0 focused tests (freeze fc3602d): exact C/U membership and frozen sites (primary and independent auditor
reconstruction from raw sealed token arrays); decision streams (logical EOS only for eos), zero-based content indexing,
L=1/3 effective lengths incl. EOS; clamp decoder == historical forced_decode without overrides, self-clamp identity,
forced token consumed before the next decision with cache advancement, suppression check, forced EOS stops;
suffix slicing after the forced span; edit distance; DIRECT_ONLY; frozen thresholds/precedence; reference firewall."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2tta0 import EOS_T, _bundle  # noqa: E402

import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import csasr.inference_cf.path_decode as pd  # noqa: E402
import experiments.inference_cf_p2path0 as run  # noqa: E402
import experiments.inference_cf_p2path0_analyze as an  # noqa: E402
import experiments.inference_cf_p2path0_audit as au  # noqa: E402
import experiments.inference_cf_p2tta0_audit as tau  # noqa: E402

C = json.loads(Path(run.CONFIG).read_text())
S = json.loads(Path(run.SITES).read_text())


# ---- population / sites ------------------------------------------------------------------------------------------

def test_sites_hash_population_and_independent_reconstruction():
    assert hashlib.sha256(Path(run.SITES).read_bytes()).hexdigest() == C["sites_byte_sha256"]
    plan4 = json.loads(Path(run.A4_PLAN).read_text())
    a4 = [json.loads(Path(s["sources"]["A4"]["path"]).read_text())["A4"] for s in S["rows"]]
    p4 = [plan4["rows"][s["parent_A3_index"]] for s in S["rows"]]
    mine = au.own_sites(p4, a4)
    for s, r, a, m in zip(S["rows"], p4, a4, mine):
        prim = run.compute_sites(r["y_B"], r["y_B_terminated"], r["y_A"], r["y_A_terminated"], a["tokens"], a["terminated"])
        assert (prim["role"], prim["site_index"], prim["B0_token"], prim["alternative_token"], prim["common_prefix"], prim["self_clamp"]) == \
            (s["role"], s["site_index"], s["B0_token"], s["alternative_token"], s["common_prefix"], s["self_clamp"])
        assert (m["role"], m["k"], m["B0"], m["alt"], m["prefix"], m["self"]) == (s["role"], s["site_index"], s["B0_token"], s["alternative_token"],
                                                                               s["common_prefix"], s["self_clamp"])
        for L in ("1", "3"):
            assert prim["clamps"][L]["tokens"] == s["clamps"][L]["tokens"] == m["cl"][L][0]
            assert prim["clamps"][L]["suffix_baseline_distance"] == s["clamps"][L]["suffix_baseline_distance"] == m["cl"][L][2]
        assert run.tok_hash(s["common_prefix"]) == s["common_prefix_sha256"]
    roles = [s["role"] for s in S["rows"]]
    assert roles.count("RESCUE") == 5 and roles.count("INDUCE") == 7
    elig = lambda R, L: sum(s["role"] == R and s["clamps"][L]["suffix_eligible"] for s in S["rows"])
    assert (elig("RESCUE", "1"), elig("RESCUE", "3"), elig("INDUCE", "1"), elig("INDUCE", "3")) == (4, 4, 6, 5)
    u1003 = next(s for s in S["rows"] if s["utterance_id"] == "ZH-CN_U1003_S0_61")
    assert u1003["clamps"]["3"]["tokens"][-1] == 50257 and u1003["clamps"]["3"]["effective_length"] == 3
    assert not u1003["clamps"]["3"]["suffix_eligible"]


def test_streams_divergence_edit_distance_and_suffix_slicing():
    assert pd.stream([5, 6], "eos") == [5, 6, 50257] and pd.stream([5, 6], "cap") == [5, 6]
    assert pd.first_divergence([1, 2, 3], [1, 2, 4]) == 2 and pd.first_divergence([1, 2], [1, 2, 3]) == 2
    assert pd.first_divergence([1, 2], [1, 2]) is None
    assert pd.first_divergence(pd.stream([1], "eos"), pd.stream([1, 2], "eos")) == 1      # shorter EOS stream diverges at its EOS
    assert pd.edit_distance([1, 2, 3], [1, 3]) == 1 == tau.own_lev([1, 2, 3], [1, 3])
    assert pd.suffix_distance([9, 9, 1, 2], [0, 0, 1, 2], 2) == 0                         # forced-span replacement earns nothing
    assert pd.suffix_distance([9, 9, 1, 2], [0, 0, 1, 2], 1) == 1


# ---- clamp decoder on the tiny model --------------------------------------------------------------------------------

def _b(supp=()):
    b = _bundle(suppress=list(supp))
    b.processor.tokenizer.eos_token_id = EOS_T
    return b


def test_clamp_decoder_matches_forced_decode_and_self_identity():
    b = _b()
    ref = tta.forced_decode(b, _enc(), CB, max_new_tokens=8)
    free = pd.clamp_decode(b, _enc(), CB, max_new_tokens=8)
    assert (free["tokens"], free["terminated"]) == (ref["tokens"], ref["terminated"]) and free["trace"]["positions_ok"]
    k = 2
    st = ref["tokens"] + ([EOS_T] if ref["terminated"] == "eos" else [])
    selfc = pd.clamp_decode(b, _enc(), CB, site=k, forced=[st[k]], expected_prefix=st[:k], max_new_tokens=8)
    assert selfc["tokens"] == ref["tokens"] and selfc["trace"]["prefix_ok"] and selfc["trace"]["release_index"] == k + 1
    assert selfc["trace"]["forced"][0]["greedy"] == st[k]


def test_forced_token_consumed_cache_advances_and_suppression_flag():
    b = _b(supp=[33])
    ref = tta.forced_decode(b, _enc(), CB, max_new_tokens=8)
    alt = 20 if ref["tokens"][1] != 20 else 21
    out = pd.clamp_decode(b, _enc(), CB, site=1, forced=[alt, 22, 23], expected_prefix=ref["tokens"][:1], max_new_tokens=8)
    assert out["tokens"][:4] == [ref["tokens"][0], alt, 22, 23] and out["trace"]["release_index"] == 4
    assert [f["cache_len"] for f in out["trace"]["forced"]] == [5, 6, 7]                 # prompt 4 + one consumed token per step
    assert out["trace"]["suppression_ok"] and out["trace"]["positions_ok"]
    bad = pd.clamp_decode(b, _enc(), CB, site=1, forced=[33], expected_prefix=ref["tokens"][:1], max_new_tokens=8)
    assert not bad["trace"]["suppression_ok"]
    wrong = pd.clamp_decode(b, _enc(), CB, site=2, forced=[20], expected_prefix=[ref["tokens"][0] + 1, 0], max_new_tokens=8)
    assert not wrong["trace"]["prefix_ok"]
    stop = pd.clamp_decode(b, _enc(), CB, site=1, forced=[EOS_T], expected_prefix=ref["tokens"][:1], max_new_tokens=8)
    assert stop["tokens"] == ref["tokens"][:1] and stop["terminated"] == "eos"


# ---- endpoints / criteria / precedence ------------------------------------------------------------------------------

def test_row_endpoints_and_direct_only():
    B0, AUTO = [1, 2, 3, 4, 50257], [7, 8, 9, 50257]
    FREE = [5, 6, 3, 9, 50257]
    e = an.row_endpoint("RESCUE", 1, B0, AUTO, FREE, [1, 2, 3, 4, 50257])
    assert e["d0"] == 2 and e["d_target"] == 0 and e["R"] == 1.0
    e2 = an.row_endpoint("RESCUE", 1, B0, AUTO, FREE, [1, 6, 3, 9, 50257])            # forced token alone, suffix unchanged
    assert e2["d_target"] == 2 and e2["R"] == 0.0
    i = an.row_endpoint("INDUCE", 1, B0, AUTO, B0, [7, 8, 9, 50257])
    assert i["d0"] == 3 and i["d_target"] == 0 and i["dB"] == 3
    d = an.row_endpoint("INDUCE", 3, [1, 2, 50257], [7, 8, 50257], [1, 2, 50257], [7, 8, 50257])
    assert not d["eligible"] and d["R"] is None


def test_criteria_thresholds_and_label_precedence():
    r = lambda R, dlg, d0=10, dB=1: {"dialogue_id": dlg, "d0": d0, "d_target": d0 * (1 - R), "dB": dB, "eligible": d0 > 0, "R": R if d0 else None}
    ok = an.criterion("RESCUE", [r(0.6, "a"), r(0.5, "b"), r(0.7, "a"), r(0.0, "c")], C)
    assert ok["required"] == 3 and ok["successes"] == 3 and ok["pooled_reduction"] == pytest.approx(0.45) and not ok["pass"]   # pooled < 0.5
    assert an.criterion("RESCUE", [r(0.6, "a"), r(0.5, "b"), r(0.7, "a"), r(0.3, "c")], C)["pass"]
    assert not an.criterion("RESCUE", [r(0.9, "a"), r(0.9, "a"), r(0.9, "a"), r(0.0, "a")], C)["pass"]                      # one dialogue
    assert not an.criterion("RESCUE", [r(0.9, "a"), r(0.9, "b"), r(0.0, "c", d0=0), r(0.0, "d", d0=0)], C)["pass"]          # n_eligible 2 < 3
    ind = [r(0.5, "a"), r(0.5, "b"), r(0.6, "c"), r(-0.5, "d"), r(-0.2, "e"), r(0.0, "f")]
    c1 = an.criterion("INDUCE", ind, C)
    assert c1["required"] == 3 and c1["successes"] == 3 and c1["pooled_reduction"] == pytest.approx(0.15) and not c1["pass"]
    assert an.criterion("INDUCE", [r(0.5, "a"), r(0.5, "b"), r(0.6, "c"), r(0.1, "d"), r(0.1, "e")], C)["pass"]
    assert not an.criterion("INDUCE", [r(0.5, "a", dB=0), r(0.5, "b"), r(0.6, "c"), r(0.1, "d"), r(0.1, "e")], C)["pass"]
    P = lambda **k: {**{f"{a}_{l}_PASS": False for a in ("RESCUE", "INDUCE") for l in ("1", "3")}, **k}
    assert an.decide(False, P()) == "P2_PATH0_INVALID"
    assert an.decide(True, P(RESCUE_1_PASS=True, INDUCE_1_PASS=True)) == "P2_PATH0_SINGLE_TOKEN_CAUSAL"
    assert an.decide(True, P(RESCUE_3_PASS=True, INDUCE_3_PASS=True, RESCUE_1_PASS=True)) == "P2_PATH0_SHORT_PREFIX_CAUSAL"
    assert an.decide(True, P(RESCUE_1_PASS=True, INDUCE_3_PASS=True)) == "P2_PATH0_ASYMMETRIC"
    assert an.decide(True, P()) == "P2_PATH0_NO_LOCAL_BRANCH_CAUSALITY"


def test_reference_firewall_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path0/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path0/run1")
    assert not called
    src = Path("experiments/inference_cf_p2path0_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path0_analyze|path_decode|inference_cf_p2path0\b)", src, re.M) is None
    rsrc = Path("experiments/inference_cf_p2path0.py").read_text()
    assert "load_references" not in rsrc and "logit_bias" not in rsrc
