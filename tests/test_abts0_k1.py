"""ABTS-0 K1 tests (CPU; synthetic values and a random tiny Whisper only; no pretrained forward, no K1 outcome)."""
import dataclasses
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf import abts0_k1 as K  # noqa: E402

CFG = json.loads((ROOT / "configs/inference_cf/abts0_k1.json").read_text())
HAN_XUE, HAN_XIAO = "学".encode(), "校".encode()


# ---- frozen constants match the config ----------------------------------------------------------------------------

def test_module_constants_match_frozen_config():
    assert K.WORD_RE.pattern == CFG["candidates"]["word_regex"]
    assert K.MIN_COVER == CFG["candidates"]["min_han_cover"]
    assert K.ONSET_TOL == CFG["candidates"]["onset_tolerance_samples"] == 3200
    assert K.PRESENT_TOL == CFG["candidates"]["present_tolerance_samples"]
    assert K.R0_EN_MIN == CFG["region"]["r0_en_min_fraction"] and K.CROP_COVER_MIN == CFG["region"]["crop_word_cover_min"]
    assert K.HEARD_MASS_MIN == CFG["region"]["heard_mass_min"]
    assert math.isclose(K.LN2, CFG["scoring"]["threshold_nats"], rel_tol=0, abs_tol=1e-15)
    assert CFG["engine"]["cB"] == [50258, 50260, 50360, 50364] and CFG["engine"]["cE"] == [50258, 50259, 50360, 50364]


# ---- units ----------------------------------------------------------------------------------------------------------

def test_utf8_chars_keeps_exact_byte_offsets_and_flags_invalid_bytes():
    data = HAN_XUE + b"a" + b"\xe6\xa0" + b"b"
    out = K.utf8_chars(data)
    assert out[0] == ("学", 0, 3, True) and out[1] == ("a", 3, 4, True)
    assert [o[3] for o in out[2:4]] == [False, False] and out[4] == ("b", 6, 7, True)


def test_content_units_map_split_han_characters_and_latin_words_to_token_spans():
    toks = [HAN_XUE[:2], HAN_XUE[2:] + HAN_XIAO, b" go", b"-to", b" school"]
    cu = K.content_units(toks)
    kinds = [(u.kind, u.surface, u.tok_first, u.tok_last) for u in cu["units"]]
    assert kinds == [("HAN", "学", 0, 1), ("HAN", "校", 1, 1), ("LATIN", "go-to", 2, 3), ("LATIN", "school", 4, 4)]
    assert cu["text"] == "学校 go-to school"


# ---- DTW timing -----------------------------------------------------------------------------------------------------

def _diag_heads(rows: int, frames: int, heads: int = 3) -> np.ndarray:
    h = np.full((rows, heads, frames), 1e-4, dtype=np.float32)
    for r in range(rows):
        c = int((r + 0.5) * frames / rows)
        h[r, :, max(0, c - 1):c + 2] = 1.0
    return h


def test_token_intervals_monotonic_eos_and_cap_rows():
    heard = 40 * 320
    h = _diag_heads(5, 40)
    eos = K.token_intervals(h, 4, "eos", heard)
    assert eos["status"] == "OK" and eos["n_timed"] == 4
    assert all(a < b for a, b in zip(eos["start"], eos["end"]))
    assert all(b <= a for a, b in zip(eos["start"][1:], eos["end"][:-1]))
    assert eos["start"] == sorted(eos["start"]) and eos["end"][-1] <= heard
    cap = K.token_intervals(h, 5, "cap", heard)            # 5 rows, capped: last token has no EOS row -> untimed
    assert cap["n_timed"] == 4
    with pytest.raises(ValueError):
        K.token_intervals(h[:, :, :10], 4, "eos", heard)
    assert K.token_intervals(h, 0, "eos", heard)["status"] == "NO_TOKENS"


# ---- association --------------------------------------------------------------------------------------------------

def _b0(intervals, toks=None):
    toks = toks or [b"\xe6\x88\x91", HAN_XUE, HAN_XIAO, b"\xe4\xba\x86"]       # 我 学 校 了
    cu = K.content_units(toks)
    timing = {"status": "OK", "n_timed": len(toks), "start": [a for a, _ in intervals], "end": [b for _, b in intervals]}
    return cu, timing


def _word(surface="school", tok=3):
    return K.Unit("LATIN", surface, 0, len(surface), 0, tok, tok)


IV = [(0, 3200), (3200, 6400), (6400, 9600), (9600, 12800)]


def test_associate_contiguous_block_target_query_and_surface():
    b0, tm = _b0(IV)
    r = K.associate(_word(), (3300, 9500), b0, tm, 16000)
    assert r["status"] == "ASSOCIATED" and r["t"] == 1 and r["b_surface"] == "学校" and r["block"] == [1, 2]
    assert r["han_cover"] == 1.0 and r["onset_diff"] == -100


@pytest.mark.parametrize("iv,status", [((0, 3000), "CONTENT_T0"), ((3300, 9500), "ASSOCIATED"), (None, "EN_UNTIMED"),
                                       ((20000, 21000), "BEYOND_HEARD"), ((9000, 9000), "ZERO_DURATION")])
def test_associate_statuses(iv, status):
    b0, tm = _b0(IV)
    heard = 16000 if status != "BEYOND_HEARD" else 19000
    assert K.associate(_word(), iv, b0, tm, heard)["status"] == status


def test_associate_rejects_low_cover_onset_mismatch_present_and_gap():
    b0, tm = _b0([(0, 3200), (3200, 4000), (9000, 9600), (9600, 12800)])
    assert K.associate(_word(), (3200, 8000), b0, tm, 16000)["status"] == "LOW_HAN_COVER"
    b0, tm = _b0([(0, 1000), (1000, 9000), (9000, 9600), (9600, 12800)])
    r = K.associate(_word(), (6000, 9500), b0, tm, 16000)
    assert r["status"] == "ONSET_MISMATCH" and r["onset_diff"] == -5000
    b0p, tmp = _b0(IV, [b"\xe6\x88\x91", HAN_XUE, b" School", HAN_XIAO])
    assert K.associate(_word("school"), (3300, 9500), b0p, tmp, 16000)["status"] == "PRESENT_IN_B0"
    b0g, tmg = _b0(IV, [b"\xe6\x88\x91", HAN_XUE, b" ok", HAN_XIAO])
    assert K.associate(_word("school"), (3300, 12000), b0g, tmg, 16000)["status"] == "NONCONTIGUOUS_BLOCK"
    b0, tm = _b0(IV)
    assert K.associate(_word(), (13000, 15000), b0, tm, 16000)["status"] == "NO_HAN_OVERLAP"


def test_associate_onset_tolerance_is_exactly_200ms():
    b0, tm = _b0([(0, 1000), (1000, 9000), (9000, 9600), (9600, 12800)])
    assert K.associate(_word(), (4200, 9500), b0, tm, 16000)["status"] == "ASSOCIATED"
    assert K.associate(_word(), (4201, 9500), b0, tm, 16000)["status"] == "ONSET_MISMATCH"


def test_associate_mid_character_onset_abstains():
    toks = [b"\xe6\x88\x91", "了".encode() + HAN_XUE[:1], HAN_XUE[1:], HAN_XIAO]   # token 1 = 了 + first byte of 学
    b0, tm = _b0([(0, 1000), (1000, 3000), (3000, 6400), (6400, 9600)], toks)
    r = K.associate(_word(), (3500, 9500), b0, tm, 16000)
    assert r["status"] == "MID_CHARACTER_ONSET" and r["b_surface"] == "学校" and r["t"] == 1


# ---- regions, selection, scoring rule -----------------------------------------------------------------------------

def test_en_fraction_and_circular_shift():
    track = np.array([[0, 1000, 0], [1000, 3000, 1], [3000, 4000, 2]])
    assert K.en_fraction(track, (500, 1500), 4000) == 0.5
    sh = K.shift_track(track, 4000)
    t = np.zeros(4000, int)
    for a, b, c in sh:
        t[a:b] = c
    assert t[3000:4000].tolist() == [1] * 1000 and t[0:1000].tolist() == [1] * 1000 and (t == 1).sum() == 2000


def test_choose_earliest_onset_then_token_then_surface():
    P = [{"status": "ELIGIBLE", "en_interval": [500, 900], "en_first_token": 4, "surface": "b"},
         {"status": "ELIGIBLE", "en_interval": [500, 900], "en_first_token": 4, "surface": "a"},
         {"status": "NOT_IN_R0_EN", "en_interval": [100, 200], "en_first_token": 1, "surface": "z"},
         {"status": "ELIGIBLE", "en_interval": [800, 900], "en_first_token": 2, "surface": "c"}]
    assert K.choose(P)["surface"] == "a"
    assert K.choose([P[2]]) is None


def _F(xc, xb, tc, tb, oc, ob):
    return {"x": {"c": xc, "b": xb}, "target": {"c": tc, "b": tb}, "off": {"c": oc, "b": ob}}


def test_acoustic_rule_thresholds_and_controls():
    r = K.acoustic_rule(_F(-8.0, -0.1, -9.0, -0.2, -8.1, -0.1))   # A_t(c)=1, A_t(b)=.1, A_o(c)=.1, A_o(b)=0 -> C=.8
    assert r["accept"] and math.isclose(r["C"], 0.8) and not r["off_only_accept"]
    r = K.acoustic_rule(_F(-8.0, -0.1, -8.5, -0.2, -8.1, -0.1))   # A_t(c)=.5 < ln2
    assert not r["accept"] and r["status"] == "ABSTAIN"
    r = K.acoustic_rule(_F(-8.0, -0.1, -9.0, -1.0, -8.0, -0.1))   # C = (1-.9) - 0 = .1
    assert not r["accept"]
    r = K.acoustic_rule(_F(-8.0, -0.1, -9.0, -0.1, -9.0, -0.1))   # off-target identical effect
    assert r["off_only_accept"] and not r["accept"]
    assert K.acoustic_rule(_F(-8.0, float("-inf"), -9.0, -0.1, -9.0, -0.1))["status"] == "NONFINITE"


def test_valid_pieces():
    assert K.valid_pieces([10, 20], 50257, [20]) is False
    assert K.valid_pieces([], 50257, []) is False
    assert K.valid_pieces([10, 50257], 50257, []) is False
    assert K.valid_pieces([10, 11], 50257, [5]) is True


# ---- matched engine on a random tiny Whisper (CPU float32) --------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_engine():
    """Random tiny Whisper with EAGER attention (cross-attention weights) and alignment heads on layers 1-2."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.models.whisper import WhisperBundle, inspect_modules
    cfg = WhisperConfig(vocab_size=200, num_mel_bins=8, encoder_layers=2, decoder_layers=3, encoder_attention_heads=2,
                        decoder_attention_heads=2, d_model=16, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=50, max_target_positions=32, decoder_start_token_id=1, pad_token_id=0,
                        bos_token_id=1, eos_token_id=2, suppress_tokens=[], attn_implementation="eager")
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    model.requires_grad_(False)
    model.generation_config.alignment_heads = [[1, 0], [2, 1]]
    model.generation_config.suppress_tokens, model.generation_config.begin_suppress_tokens = [], []
    b = WhisperBundle(model=model, processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2, decode=lambda *a, **k: "")),
                      config=cfg, device="cpu", dtype=torch.float32, model_id="tiny-k1", revision="test",
                      encoder_step_sec=0.02, max_encoder_frames=50, num_encoder_layers=2, num_decoder_layers=3, d_model=16,
                      sample_rate=16000, chunk_sec=1.0, module_report=inspect_modules(model))
    torch.manual_seed(1)
    with torch.inference_mode():
        enc = BaseModelOutput(last_hidden_state=model.model.encoder(torch.randn(1, 8, 100)).last_hidden_state)
    return b, enc


def test_matched_decode_equals_forced_decode_and_keeps_heads(tiny_engine):
    from csasr.inference_cf.episodic_tta import forced_decode
    b, enc = tiny_engine
    d = K.matched_decode(b, enc, [1], eos=2, suppress=[], begin=[], max_new_tokens=12, capture_layer=1)
    ref = forced_decode(b, enc, [1], max_new_tokens=12, capture_layer=1)
    assert d["tokens"] == ref["tokens"] and d["terminated"] == ref["terminated"]
    S = len(d["tokens"]) + (d["terminated"] == "eos")
    assert d["heads"].shape == (S, 2, enc.last_hidden_state.shape[1]) and len(d["logits_sha256"]) == S
    assert np.isfinite(d["heads"]).all() and (d["heads"] >= 0).all()


def test_score_word_replays_the_capture_logits_bitwise(tiny_engine):
    b, enc = tiny_engine
    d = K.matched_decode(b, enc, [1], eos=2, suppress=[], begin=[], max_new_tokens=12, capture_layer=1)
    if len(d["tokens"]) < 3:
        pytest.skip("random tiny model terminated too early")
    t = 1
    s = K.score_word(b, enc, [1], d["tokens"][:t], d["tokens"][t:t + 2], suppress=[], begin=[], capture_layer=1)
    assert s["query_logits_sha256"] == d["logits_sha256"][t] and s["query_argmax"] == d["tokens"][t]
    assert len(s["logp"]) == 2 and all(v <= 0 for v in s["logp"]) and math.isfinite(s["F"])
    s2 = K.score_word(b, enc, [1], d["tokens"][:t], [5, 7], suppress=[], begin=[], capture_layer=1)
    assert s2["query_logits_sha256"] == s["query_logits_sha256"]           # same prefix, independent cold branch
    with pytest.raises(ValueError):
        K.score_word(b, enc, [1], d["tokens"][:t], [], suppress=[], begin=[], capture_layer=1)


# ---- firewall / evaluator logic -----------------------------------------------------------------------------------

def test_runner_and_module_never_touch_references():
    for p in ("src/csasr/inference_cf/abts0_k1.py", "experiments/abts0_k1.py"):
        src = (ROOT / p).read_text()
        body = re.sub(r"FORBIDDEN = \(.*?\)\n", "", src.split('"""', 2)[2], flags=re.S)   # the firewall's own deny-list
        for bad in ("load_references", "evaluation_units", "oracle_tracks", "abts0_k1_evaluate", "transcript_raw",
                    "candidates_existing_ctc", "p0_r2/evaluation"):
            assert bad not in body, (p, bad)


def test_runner_sources_exist_and_firewall_detects_forbidden_paths():
    import experiments.abts0_k1 as R
    assert all((ROOT / s).exists() for s in R.SOURCES)
    R.OPENED.clear()
    R.OPENED.add("/mnt/data/tungnx/whisper-large-v3/model.safetensors")
    assert R.firewall_check()
    R.OPENED.add(str(ROOT / "results/inference_cf/p0_r2/evaluation_units.json"))
    with pytest.raises(RuntimeError):
        R.firewall_check()
    R.OPENED.clear()


def test_evaluator_auroc_and_decision_precedence():
    import experiments.abts0_k1_evaluate as E
    assert E.auroc([2.0, 3.0], [1.0]) == 1.0 and E.auroc([1.0], [1.0]) == 0.5 and E.auroc([], [1.0]) is None
    go = CFG["evaluator"]["go_MM"]
    m = {"invalid": False, "reachable_genuine_runs": 20, "reachable_dialogues": 8, "auroc_dialogues": 3, "accepted": 12,
         "accepted_dialogues": 6, "precision": 0.75, "genuine_recoverable_words": 6, "genuine_recoverable_dialogues": 4,
         "reachable_recall": 0.2, "auroc_macro": 0.65}
    assert E.decide(m, go) == "K1_PASS"
    assert E.decide(dict(m, precision=0.74), go) == "K1_FAIL"
    assert E.decide(dict(m, accepted=11), go) == "K1_FAIL"
    assert E.decide(dict(m, auroc_dialogues=2), go) == "K1_INSUFFICIENT"
    assert E.decide(dict(m, reachable_genuine_runs=11), go) == "K1_INSUFFICIENT"
    assert E.decide(dict(m, invalid=True), go) == "K1_BLOCKED"


def test_evaluator_literal_labelling_uses_timing_overlap():
    import experiments.abts0_k1_evaluate as E
    rc = {"tokens": ["我", "school", "了", "school"], "is_en": [False, True, False, True], "timing": [(0, 3000), (3000, 8000),
          (8000, 9000), (20000, 25000)], "heard": 30000}
    st = [None, {"correct": False, "category": "wrong_language_substitution", "artifact": None, "genuine": True},
          None, {"correct": True, "category": "correct", "artifact": None, "genuine": False}]
    assert E.label_candidate(rc, st, "School", (3500, 7000))["label"] == "LITERAL"
    assert E.label_candidate(rc, st, "school", (8100, 8900))["label"] == "EN_FROM_ZH_SPEECH"
    assert E.label_candidate(rc, st, "college", (3500, 7000))["label"] == "WRONG_ENGLISH"
    lab = E.label_candidate(rc, st, "school", (9100, 11000))
    assert lab["label"] == "UNTIMED_SPAN" and not lab["literal"]
    assert E.label_candidate(rc, st, "school", (8100, 8900))["literal_pm200"] is True


def test_structural_pools_end_to_end_with_unchanged_s1_crops():
    s1cfg = json.loads((ROOT / "configs/inference_cf/s1_acoustic_evidence.json").read_text())
    heard, F = 64000, 200
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(80000) * 0.1).astype(np.float32)
    track = np.array([[0, 16000, 2], [16000, 24000, 1], [24000, 64000, 2]])
    b0 = K.content_units([b"\xe6\x88\x91", HAN_XUE, HAN_XIAO, "了".encode()])
    b0_t = {"status": "OK", "n_timed": 4, "start": [0, 16000, 20000, 24000], "end": [16000, 20000, 24000, 64000]}
    en = K.content_units([b" I", b" go", b" school", b" now"])
    en_t = {"status": "OK", "n_timed": 4, "start": [0, 5000, 16100, 24100], "end": [5000, 16100, 23900, 40000]}
    heads = np.full((5, 3, F), 1e-6, dtype=np.float32)
    heads[:, :, 50:75] = 0.9 / 25                                      # every query attends to the EN region
    out = K.structural_pools(en=en, en_timing=en_t, b0=b0, b0_timing=b0_t, heard=heard, b0_heads=heads, track=track,
                             x=x, s1cfg=s1cfg)
    st = {p["surface"]: p["status"] for p in out["proposals"]}
    assert st["school"] == "ELIGIBLE" and st["I"] == "CONTENT_T0" and st["now"] == "NOT_IN_R0_EN"
    ch = out["chosen"]
    assert ch["surface"] == "school" and ch["t"] == 1 and ch["b_surface"] == "学校"
    a, b = ch["target"]["bounds"]
    assert 16000 <= a < b <= 24000 and ch["target"]["word_cover"] >= 0.5
    oa, ob = ch["off_target"]["bounds"]
    assert ob - oa == b - a and (ob <= a or oa >= b) and ch["off_target"]["en_fraction"] <= 0.1
    assert {u["surface"] for u in out["U0"]} == {"I", "go", "school", "now"}
    shuf = K.structural_pools(en=en, en_timing=en_t, b0=b0, b0_timing=b0_t, heard=heard, b0_heads=heads,
                              track=K.shift_track(track, heard), x=x, s1cfg=s1cfg)
    assert {p["surface"]: p["status"] for p in shuf["proposals"]}["school"] == "NOT_IN_R0_EN"
