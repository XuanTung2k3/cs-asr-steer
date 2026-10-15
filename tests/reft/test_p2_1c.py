"""CPU tests for P2-1c: token-boundary anchoring, divergence types, continuation scoring, beam-equivalent score, the
begin-suppression semantics of forced-prefix replay, mechanism flags, cell aggregation and the verdict mapping."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.p2_1c_continuation_audit import (aggregate_cells, anchor_at, anchor_inside, beam_equivalent, boundaries, cand_logps,  # noqa: E402
                                           decide, en_region, first_divergences, flags_for, hyp_after, hyp_before,
                                           inference_manifest_clean, local_disfavor, seq_preference, status_of)
from reft.text import alignment, tokenize  # noqa: E402

R = {"recovered_min_retention": 0.8, "dropped_max_retention": 0.2, "local_drop_nats": 1.0, "seed_consistent_min": 2,
     "dominance_fraction": 0.6, "dominance_min_cases": 3, "min_valid_damaged_anchors": 3}


def b(*xs):
    return [x.encode() if isinstance(x, str) else x for x in xs]


def test_boundaries_han_multi_char_and_split_word():
    # tokens: "我们" (two Han chars in one token), " use", " Py", "thon", "，", "然后" -> units 我 们 use python 然 后
    toks = b("我们", " use", " Py", "thon", "，", "然后")
    ms, full = boundaries(toks)
    assert full == tokenize("我们 use Python，然后")
    assert ms[0] == 0 and ms[1] == 2 and ms[2] == 3
    assert ms[3] is None              # inside the word "Python" (split across tokens): not a clean boundary
    assert ms[4] == 4 and ms[5] == 4 and ms[6] == 6
    a = anchor_at(ms, 4)
    assert a == {"k": 5, "units": 4, "exact": True, "shortfall": 0}   # largest clean boundary: after the punctuation


def test_boundaries_partial_utf8_and_straddling_token():
    han = "中文".encode()
    toks = [han[:2], han[2:], b" ok"]   # a token boundary inside a UTF-8 character
    ms, _ = boundaries(toks)
    assert ms[1] is None and ms[2] == 2 and ms[3] == 3
    # one token spanning the end of unit 0 and start of unit 1 ("了我" as one token after "好")
    ms2, _ = boundaries(b("好了", "我"))
    a = anchor_at(ms2, 1)          # no boundary with exactly 1 unit -> nearest earlier, approximate
    assert a["exact"] is False and a["k"] == 0 and a["shortfall"] == 1


def test_anchor_inside_span_skips_straddling_token():
    # units 我 们 中 文 好 吗 ; the token "中文" straddles span units 2 and 3 -> first clean boundary with >= 3 units is 4
    ms, _ = boundaries(b("我", "们", "中文", "好", "吗"))
    assert ms == [0, 1, 2, 4, 5, 6]
    assert anchor_inside(ms, 3, 5) == {"k": 3, "units": 4, "exact": True, "shortfall": 0}
    assert anchor_inside(ms, 3, 3) is None


def test_hyp_positions_and_en_region():
    ref, hyp = ["我", "要", "use", "python", "然", "后"], ["我", "要", "use", "pi", "然", "后"]
    ops = alignment(ref, hyp)
    assert hyp_before(ops, 4) == 4 and hyp_after(ops, 3) == 4
    ops_del = alignment(ref, ["我", "要", "use", "后"])          # python and 然 deleted
    assert hyp_before(ops_del, 4) == 3 and hyp_after(ops_del, 4) == 3
    tags = ["ZH", "ZH", "EN", "EN", "ZH", "ZH"]
    assert en_region(tags, ref, 4, set()) == [2, 3]
    assert en_region(["ZH", "EN", "EN", "ZH"], ["我", "um", "ok", "然"], 3, {"um"}) == [1, 2]
    assert en_region(["ZH", "ZH"], ["我", "要"], 1, set()) is None


def test_format_vs_lexical_divergence():
    b0_units = tokenize("我们 use Python 然后")
    s_units = tokenize("我们 use python 然后")       # case-only difference: no lexical divergence
    ms = [0, 2, 3, 4, 6, 6]
    d = first_divergences([1, 2, 3, 4, 9], [1, 2, 7, 4, 9], b0_units, s_units, ms)
    assert d["token_k"] == 2 and d["lexical_unit"] is None and d["format_only_first"]
    d2 = first_divergences([1, 2, 3, 4, 9], [1, 2, 3, 9], b0_units, tokenize("我们 use Python"), ms)
    assert d2["token_k"] == 3 and d2["lexical_unit"] == 4 and d2["token_div_b0_units"] == 4 and not d2["format_only_first"]
    assert first_divergences([1, 9], [1, 9], ["a"], ["a"], [0, 1])["token_k"] is None


def test_candidate_shift_prompt_not_scored_and_beam_equivalent():
    V = 7
    lp = torch.log_softmax(torch.arange(6 * V, dtype=torch.float32).reshape(6, V).sin(), -1)
    P = [0, 1, 2, 3]                              # prefix of length 4 (rows 0..2 score prefix tokens: never used)
    c = [5, 6, 2]
    got = cand_logps(lp, len(P), c)
    assert got == [float(lp[3, 5]), float(lp[4, 6]), float(lp[5, 2])]
    assert abs(beam_equivalent(got, c, set()) - sum(got) / 3) < 1e-9
    assert beam_equivalent(got, c, {6}) is None   # a canonically suppressed token -> constrained score undefined
    assert abs(beam_equivalent(got, c, set(), 0.5) - sum(got) / 3 ** 0.5) < 1e-9


def test_begin_suppression_not_rearmed_after_forced_prefix():
    from transformers.generation.logits_process import SuppressTokensAtBeginLogitsProcessor
    proc = SuppressTokensAtBeginLogitsProcessor([220, 50257], begin_index=4, device="cpu")
    s = torch.zeros(1, 50300)
    at4 = proc(torch.zeros(1, 4, dtype=torch.long), s.clone())
    at9 = proc(torch.zeros(1, 9, dtype=torch.long), s.clone())
    assert torch.isinf(at4[0, 50257]) and torch.isinf(at4[0, 220])
    assert not torch.isinf(at9).any()              # EOS stays available after a forced content prefix


def test_status_local_and_sequence_preference():
    assert status_of(0.85, R) == "recovered" and status_of(0.2, R) == "dropped" and status_of(0.5, R) == "partial"
    assert status_of(None, R) is None
    assert local_disfavor(2, -0.1, -0.1, R) and local_disfavor(1, -2.5, -1.0, R) and not local_disfavor(1, -1.5, -1.0, R)
    cands = [{"status": "recovered", "scores": {"S": {"beam_eq": -0.30}}},
             {"status": "dropped", "scores": {"S": {"beam_eq": -0.20}}},
             {"status": "partial", "scores": {"S": {"beam_eq": -0.01}}},
             {"status": "dropped", "scores": {"S": {"beam_eq": None}}}]
    q, m = seq_preference(cands, "S")
    assert q is True and abs(m - (-0.10)) < 1e-12
    assert seq_preference(cands[:1], "S") == (None, None)


def test_flags():
    # recovers under B0 prefix, drops under own prefix -> prefix history
    assert flags_for(True, "recovered", "dropped", True, False, False, False) == ["H1_PREFIX_HISTORY"]
    # identical prefixes make the B0/own comparison void
    assert flags_for(True, "recovered", "dropped", True, True, False, False) == []
    assert flags_for(True, "dropped", "dropped", True, False, True, None) == ["H2_LOCAL_SWITCHBACK"]
    assert flags_for(True, "dropped", "dropped", True, False, False, True) == ["H3_SEQUENCE_PATH"]
    # restart from its own prefix recovers the span the original search dropped -> search/path, may co-occur with H2
    assert flags_for(True, "dropped", "recovered", True, False, True, None) == ["H2_LOCAL_SWITCHBACK", "H3_SEQUENCE_PATH"]
    assert flags_for(True, "partial", "dropped", True, False, False, None) == []
    assert flags_for(False, "recovered", "recovered", True, False, False, None) == []


def rows_for(case, arm, flags_per_seed, drops=(True, True, True)):
    return [{"case": case, "system": f"{arm}{s}", "orig_drop": d, "flags": f}
            for s, d, f in zip((42, 43, 44), drops, flags_per_seed)]


def test_cells_and_decision():
    H1, H2 = ["H1_PREFIX_HISTORY"], ["H2_LOCAL_SWITCHBACK"]
    rows = []
    for c in ("d0", "d1", "d2", "d3", "d4"):
        rows += rows_for(c, "CE", [H1, H1, []])
    for c in ("d0", "d3", "d4"):
        rows += rows_for(c, "KL", [H1, [], H1])
    rows += rows_for("d1", "KL", [[], [], []], drops=(False, False, False))
    cells = aggregate_cells(rows, R)
    assert cells["d1/KL"]["applicable"] is False and cells["d0/CE"]["supported"] == ["H1_PREFIX_HISTORY"]
    assert cells["d0/CE"]["counterexamples"]["H1_PREFIX_HISTORY"] == ["CE44"]
    d = decide(True, 5, cells, R)
    assert d["verdict"] == "P2_1C_DIAGNOSTIC_COMPLETE" and d["recommendation"] == "GO_P2_2_ON_POLICY_PRESERVATION_DESIGN"
    assert decide(False, 5, cells, R)["recommendation"] == "HOLD_FOR_ENGINEERING_OR_ALIGNMENT_REPAIR"
    assert decide(True, 2, cells, R)["verdict"] == "P2_1C_INCONCLUSIVE"
    # split evidence: H1 in 3 cells, H2 in 3 cells of 8 -> partially identifiable, tie -> smallest architectural screen
    rows = []
    for c in ("d0", "d1", "d2"):
        rows += rows_for(c, "CE", [H1, H1, H1])
    for c in ("d3", "d4"):
        rows += rows_for(c, "CE", [H2, H2, []])
    rows += rows_for("d0", "KL", [H2, H2, H2]) + rows_for("d3", "KL", [[], [], []]) + rows_for("d4", "KL", [[], [], []])
    d = decide(True, 5, aggregate_cells(rows, R), R)
    assert d["verdict"] == "P2_1C_PARTIALLY_IDENTIFIABLE" and d["recommendation"] == "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING"
    rows = [r for c in ("d0", "d1", "d2", "d3", "d4") for r in rows_for(c, "CE", [[], [], []])]
    d = decide(True, 5, aggregate_cells(rows, R), R)
    assert d["verdict"] == "P2_1C_INCONCLUSIVE" and d["recommendation"] == "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING"


def test_inference_manifest_has_no_reference_fields():
    ok = {"cases": [{"key": "d0", "uid": "u", "prompt": [1, 2, 3, 4], "b0_anchors": {"start": 0, "post_en": 5},
                     "own_anchors": {"CE42": 7, "KL42": None}}]}
    assert inference_manifest_clean(ok)
    bad = {"cases": [dict(ok["cases"][0], span_units=[3, 4])]}
    assert not inference_manifest_clean(bad)
    bad2 = {"cases": [dict(ok["cases"][0], b0_anchors={"post_en": "我"})]}
    assert not inference_manifest_clean(bad2)
