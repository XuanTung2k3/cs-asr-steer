"""R0 implementation tests (reference-free provider, constructor, controls, gates, firewall, auditor agreement).
No pretrained science: the end-to-end smoke uses a tiny RANDOM Whisper with large-v3 decoder geometry."""
from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from csasr.inference_cf import r0_regions as R
from csasr.inference_cf import r0_unique as Q

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/inference_cf/r0_region_vector.json").read_text())
PANEL = json.loads((ROOT / "docs/inference_cf/R0_PANEL.json").read_text())
LANG = CFG["regions"]["language_ids"]


def probs(pe, pm):
    p = np.full(100, (1 - pe - pm) / 98, dtype=np.float64)
    p[0], p[1] = pe, pm
    return p


# 1. window classification -------------------------------------------------------------------------------------------
def test_window_rule_thresholds_and_abstentions():
    rc = CFG["regions"]
    loud = np.full(16000, 0.1, np.float32)
    assert R.classify_window(probs(0.85, 0.1), LANG, loud, rc)["label"] == "EN"
    assert R.classify_window(probs(0.1, 0.85), LANG, loud, rc)["label"] == "ZH"
    amb = R.classify_window(probs(0.4, 0.3), LANG, loud, rc)
    assert amb["label"] == "U" and amb["reasons"] == ["ambiguous_pair"]
    low = R.classify_window(probs(0.3, 0.1), LANG, loud, rc)          # Q = .4 < .5
    assert low["label"] == "U" and "low_pair_mass" in low["reasons"]
    sil = R.classify_window(probs(0.9, 0.05), LANG, np.zeros(16000, np.float32), rc)
    assert sil["label"] == "U" and "low_rms" in sil["reasons"]
    short = R.classify_window(probs(0.9, 0.05), LANG, np.full(7999, 0.1, np.float32), rc)
    assert short["label"] == "U" and set(short["reasons"]) >= {"short_crop"}
    both = R.classify_window(probs(0.3, 0.1), LANG, np.zeros(100, np.float32), rc)
    assert set(both["reasons"]) == {"short_crop", "low_rms", "low_pair_mass"}
    for bad in (np.full(100, np.nan), probs(0.5, 0.2) * 1.01, np.ones(99) / 99):
        with pytest.raises(ValueError):
            R.classify_window(bad, LANG, loud, rc)


# 2. grid, sweep, masks -----------------------------------------------------------------------------------------------
def test_grid_sweep_matches_bruteforce_and_frame_masks():
    assert R.window_grid(7999) == [(0, 7999)] and R.window_grid(24001) == [(0, 16000), (8000, 24000), (8001, 24001)]
    assert sum(len(R.window_grid(r["audio"]["resampled_num_samples"])) for r in PANEL["rows"]) == 10387
    rng = np.random.default_rng(1)
    for n in (5000, 16000, 47123, 80000):
        b = R.window_grid(n)
        codes = list(rng.integers(0, 3, len(b)))
        iv = R.sample_intervals(b, codes, n, 0.75)
        tr = R.intervals_to_track(iv, n)
        for x in rng.integers(0, n, 300):
            cov = [c for (s, e), c in zip(b, codes) if s <= x < e]
            fe, fz = cov.count(1) / len(cov), cov.count(2) / len(cov)
            assert tr[x] == (1 if fe >= .75 else (2 if fz >= .75 else 0))
        assert all(iv[k][2] != iv[k + 1][2] for k in range(len(iv) - 1)) and iv[0][0] == 0 and iv[-1][1] == n
    tr = np.zeros(330, np.int8)
    tr[:160] = 1
    tr[320:] = 2
    we, wm = R.frame_masks(tr)
    assert we.tolist() == [0.5, 0.0] and wm.tolist() == [0.0, 1.0]          # partial last frame uses its 10 samples


# 3. mapping and assignment --------------------------------------------------------------------------------------------
def test_mapping_mass_normalization_and_shared_assignment():
    h = np.zeros((10, 3, 4))
    h[:, 0, :] = [0.3, 0.3, 0.2, 0.2]          # real frames = 2: raw .6 -> mapped
    h[:, 1, :] = [0.1, 0.1, 0.4, 0.4]          # raw .2 -> U
    h[:, 2, :] = [0.0, 0.5, 0.25, 0.25]        # raw .5 -> mapped, argmax frame 1
    R.check_attention(h)
    mp = R.query_mapping(h[:, :, :2], 0.5)
    assert mp["mapped"].tolist() == [True, False, True]
    np.testing.assert_allclose(mp["a"][0], [0.5, 0.5])
    assert mp["argmax_frame"].tolist() == [0, 0, 1]
    asg = R.assign_queries(mp["a"], mp["mapped"], np.array([True, True, False]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), CFG["mapping"])
    assert asg["label"].tolist() == [0, 0, 0]                                 # q0: qE=.5 <.7; q2 ineligible
    asg = R.assign_queries(np.array([[0.75, 0.25]]), np.array([True]), np.array([True]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), CFG["mapping"])
    assert asg["label"].tolist() == [0]                                       # qM=.25 > .20 opposite cap
    with pytest.raises(ValueError):
        R.check_attention(np.full((10, 2, 3), np.nan))
    with pytest.raises(ValueError):
        R.check_attention(np.zeros((10, 2, 3)))


def test_query_eligibility_rules():
    from transformers import WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from experiments.inference_cf_r0 import eligibility
    tok = WhisperProcessor.from_pretrained(CFG["model"]["dir"], local_files_only=True).tokenizer
    bd = {v: k for k, v in bytes_to_unicode().items()}
    han = tok.encode("挺", add_special_tokens=False)
    content = [5620] + han + [50300, 220]
    el = eligibility(tok, content, 50257, [], bd)
    assert el[0] == ["t0_forced_prefix_query"] and "eos_slot" in el[len(content)]
    assert "special_or_invalid_target" in el[1 + len(han)]
    mid = [b for b in range(1, 50000) if not __import__("csasr.inference_cf.core_r2", fromlist=["x"]).prefix_utf8_complete(tok, [b], bd)][:1]
    el2 = eligibility(tok, [5620] + mid + [220], 50257, [], bd)
    assert "incomplete_utf8_prefix" in el2[2]


# 4. constructor ------------------------------------------------------------------------------------------------------
def explicit_reference(HE, HM, r):
    """Explicit uncentered moment eigendecomposition reference for a given rank."""
    CE, CM = HE.T @ HE / len(HE), HM.T @ HM / len(HM)
    wE, VE = np.linalg.eigh(CE)
    wM, VM = np.linalg.eigh(CM)
    UE, UM = VE[:, ::-1][:, :r], VM[:, ::-1][:, :r]
    P, s, Qt = np.linalg.svd(UE.T @ UM)
    a = UE @ P
    sc = np.array([a[:, i] @ CE @ a[:, i] for i in range(r)]) * (1 - np.clip(s, 0, 1) ** 2)
    v = a[:, int(np.argmax(sc))]
    c = HE.mean(0) - HM.mean(0)
    return (v if v @ c > 0 else -v) / np.linalg.norm(v)


def test_thin_svd_constructor_matches_explicit_moments_and_rank_policy():
    rng = np.random.default_rng(240924)
    n_ok = 0
    for k in range(25):
        d = 48
        HE = rng.normal(size=(30, d)) @ np.diag(rng.uniform(.2, 3, d)) + 2
        HM = rng.normal(size=(28, d)) @ np.diag(rng.uniform(.2, 3, d)) - 1
        rec = Q.construct(HE, HM, np.arange(30) % 12, np.arange(28) % 12, CFG["unique"])
        assert set(rec["candidates"]) == {"2", "4", "8"}
        if rec["status"] == "ok":
            n_ok += 1
            ref = explicit_reference(HE, HM, rec["rank"])
            assert np.max(np.abs(rec["vector"].astype(np.float64) - ref)) <= 1e-6
            assert abs(np.linalg.norm(rec["vector"].astype(np.float64)) - 1) <= 2e-6
            again = Q.construct(HE, HM, np.arange(30) % 12, np.arange(28) % 12, CFG["unique"])
            assert again["vector_sha256"] == rec["vector_sha256"]
    assert n_ok >= 15


def test_rank_selection_abstentions_no_fallback():
    rng = np.random.default_rng(3)
    H = rng.normal(size=(40, 32))
    rec = Q.construct(np.tile(H[:1], (20, 1)), H[:20], np.arange(20), np.arange(20), CFG["unique"])     # repeated rows
    assert rec["status"] == "invalid" and rec["reasons"][0] == "NO_SUPPORTED_RANK" and rec["vector"] is None and rec["rank"] is None
    assert "E:participation" in rec["reasons"] and "E:numerical_rank" in rec["reasons"]
    few = Q.construct(H[:5], H[5:30], np.arange(5), np.arange(25), CFG["unique"])
    assert few["status"] == "invalid" and "E:count" in few["reasons"]
    bins = Q.construct(H[:20], H[20:40], np.zeros(20), np.arange(20), CFG["unique"])
    assert bins["status"] == "invalid" and "E:bins" in bins["reasons"]
    empty = Q.construct(np.zeros((0, 32)), H[:20], np.zeros(0), np.arange(20), CFG["unique"])
    assert empty["status"] == "invalid" and empty["n_E"] == 0
    # largest supported rank is chosen before winner guards (8 with plenty of diverse rows)
    sc = 0.93 ** np.arange(32)                       # separated spectrum, participation ~14 >= 8, eigen gap ~13% > 5%
    big = Q.construct(rng.normal(size=(800, 32)) * sc + 0.01, rng.normal(size=(800, 32)) * sc[::-1], np.arange(800) % 20, np.arange(800) % 20, CFG["unique"])
    assert big["rank"] == 8
    iso = Q.construct(rng.normal(size=(60, 32)) + 1, rng.normal(size=(60, 32)), np.arange(60) % 20, np.arange(60) % 20, CFG["unique"])
    assert iso["rank"] is None and "M:eigen_gap" in iso["reasons"]          # isotropic noise: no identifiable rank


def test_independent_auditor_constructor_agrees():
    from experiments.inference_cf_r0_audit import agree, construct_i
    rng = np.random.default_rng(11)
    agreeing = 0
    for k in range(30):
        HE = rng.normal(size=(int(rng.integers(6, 60)), 64)) * rng.uniform(.3, 2, 64) + rng.normal(size=64)
        HM = rng.normal(size=(int(rng.integers(6, 60)), 64)) * rng.uniform(.3, 2, 64)
        bE, bM = rng.integers(0, 15, len(HE)), rng.integers(0, 15, len(HM))
        p, m = Q.construct(HE, HM, bE, bM, CFG["unique"]), construct_i(HE, HM, bE, bM, CFG["unique"])
        agreeing += agree(p, m, p["vector"], CFG["unique"])[0]
    assert agreeing == 30


# 5. controls ---------------------------------------------------------------------------------------------------------
def test_shuffle_erode_dilate_controls():
    key = CFG["controls"]["shuffle_seed_key"]
    o = R.shuffle_offsets("U1", 48000, 20, 1600, key)
    assert o == sorted(set(o)) and len(o) == 20 and all(1600 <= x <= 48000 - 1600 for x in o)
    assert o == R.shuffle_offsets("U1", 48000, 20, 1600, key) and o != R.shuffle_offsets("U2", 48000, 20, 1600, key)
    assert R.shuffle_offsets("U1", 3218, 20, 1600, key) is None                  # 19 legal offsets
    from experiments.inference_cf_r0_audit import offsets
    assert offsets("U1", 48000, 20, 1600, key) == o
    tr = np.zeros(20000, np.int8)
    tr[1000:5000] = 1
    tr[5000:6000] = 2
    tr[9000:10000] = 1
    for off in o:
        assert Counter(np.roll(tr, off).tolist()) == Counter(tr.tolist())
    er = R.erode(tr, 1600)
    assert er[2600:3400].tolist() == [1] * 800 and er[:2600].sum() == 0 and (er == 2).sum() == 0 and er[9000:10000].sum() == 0
    di = R.dilate(tr, 1600)
    assert di[3400:6600].tolist() == [0] * 3200                                   # expanded EN/ZH overlap -> U
    assert di[0:1000].tolist() == [1] * 1000 and di[7400:7600].tolist() == [0] * 200 and di[7600:11600].tolist() == [1] * 4000
    from experiments.inference_cf_r0_audit import dilate_i, erode_i
    assert np.array_equal(erode_i(tr, 1600), er) and np.array_equal(dilate_i(tr, 1600), di)


# 6/10. end-to-end engineering smoke on a tiny RANDOM Whisper with real large-v3 geometry ---------------------------------
@pytest.fixture(scope="module")
def tiny_large_geometry():
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
    from csasr.models.whisper import WhisperBundle, inspect_modules
    conf = WhisperConfig(vocab_size=51866, num_mel_bins=128, encoder_layers=1, decoder_layers=32, encoder_attention_heads=2,
                         decoder_attention_heads=20, d_model=40, encoder_ffn_dim=40, decoder_ffn_dim=40, max_source_positions=1500,
                         max_target_positions=448, decoder_start_token_id=50258, pad_token_id=50257, bos_token_id=50257, eos_token_id=50257)
    conf._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(240924)
        model = WhisperForConditionalGeneration(conf).eval().to(torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained(CFG["model"]["dir"], local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    proc = WhisperProcessor.from_pretrained(CFG["model"]["dir"], local_files_only=True)
    return WhisperBundle(model=model, processor=proc, config=conf, device="cpu", dtype=torch.bfloat16, model_id="tiny-random", revision=None,
                         encoder_step_sec=0.02, max_encoder_frames=1500, num_encoder_layers=1, num_decoder_layers=32, d_model=40,
                         sample_rate=16000, chunk_sec=30.0, module_report=inspect_modules(model))


def _ctx(bundle):
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import tokenizer_partition
    part = tokenizer_partition(bundle.processor.tokenizer)
    return {"tok": bundle.processor.tokenizer, "partition": {"embedded": set(part["embedded_ids"]), "matrix": set(part["matrix_ids"])},
            "suppress": list(bundle.model.generation_config.suppress_tokens), "begin": list(bundle.model.generation_config.begin_suppress_tokens),
            "byte_decoder": {v: k for k, v in bytes_to_unicode().items()},
            "counters": Counter({k: 0 for k in ("lid_calls", "encoder_passes", "decoder_forwards", "constructions")})}


def _synthetic(monkeypatch):
    """Engineering-only: synthetic LID (EN first half, ZH second half) and peaked bf16 alignment attention so that the
    group / construction / control paths execute on the tiny model's REAL recorded states."""
    import experiments.inference_cf_p0_r2 as p0
    orig = p0.full_replay
    calls = {"n": 0}

    def lid(bundle, waveform, language_ids):
        calls["n"] += 1
        en = calls["n"] <= 9                     # row 0 has 18 windows: first half EN, second half ZH
        p = probs(0.9, 0.05) if en else probs(0.05, 0.9)
        return {i: float(np.float32(x)) for i, x in zip(language_ids, p / p.sum())}

    def replay(bundle, encoded, prompt, content, *, attention):
        logits, heads = orig(bundle, encoded, prompt, content, attention=attention)
        if heads is not None:
            L = heads.shape[1]
            h = torch.zeros_like(heads)
            for q in range(L):
                f = min(468, int(q / L * 460))
                h[:, q, f] = 1.0
            heads = h.to(torch.bfloat16).float()
        return logits, heads
    monkeypatch.setattr(p0, "native_lid", lid)
    monkeypatch.setattr(p0, "full_replay", replay)


def test_end_to_end_tiny_smoke_integrity_firewall_and_auditor_agreement(tiny_large_geometry, monkeypatch):
    from experiments.inference_cf_r0 import assert_no_hooks, process_row, runtime_projection
    from experiments.inference_cf_r0_audit import recompute_row
    from experiments.inference_cf_p2dir_analyze import jsonable
    from csasr.inference_cf.core import canonical
    bundle = tiny_large_geometry
    w0 = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    rows = runtime_projection(PANEL)
    row = dict(rows[0])
    row["audio"] = dict(row["audio"])
    _synthetic(monkeypatch)
    import sys
    opened = []
    hook = lambda e, a: opened.append(str(a[0])) if e == "open" and a and isinstance(a[0], (str, bytes)) else None
    sys.addaudithook(hook)
    rec, arrays = process_row(bundle, row, CFG, _ctx(bundle))
    rec = jsonable(rec)
    canonical(rec)                                                        # JSON-safe, no NaN
    assert_no_hooks(bundle)
    assert all(torch.equal(v, w0[k]) for k, v in bundle.model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in bundle.model.parameters())
    assert rec["integrity"]["site_equals_ffn_input"] and rec["integrity"]["repeat_replay_bitwise"] and rec["integrity"]["finite"]
    assert rec["integrity"]["primary_repeat_identical"]
    assert set(rec["constructions"]) == {"3", "8", "16", "24"} and len(rec["queries"]) == rec["T"] + 1
    assert rec["counts"]["EN"] > 0 and rec["counts"]["ZH"] > 0
    assert any(rec["constructions"][l]["primary"]["status"] == "ok" for l in rec["constructions"])
    assert len(rec["controls"]["shuffle"]["16"]) == 20
    assert not any(x for x in opened if any(f in x for f in ("candidates_existing", "/roles/", "p2rj", "alignments")))
    # independent auditor reproduces the row from the saved arrays
    z = {k: np.asarray(v) for k, v in arrays.items()}
    chk, res = recompute_row(rec, z, CFG, PANEL["rows"][0])
    bad = {k: v for k, v in chk.items() if not v}
    assert not bad and not res["disagree"], (bad, res["disagree"][:5])
    assert res["n"] == 4 * (1 + 2 + 20)


def test_end_to_end_tiny_real_lid_path_all_uncertain_is_complete(tiny_large_geometry):
    from experiments.inference_cf_r0 import process_row, runtime_projection
    bundle = tiny_large_geometry
    row = runtime_projection(PANEL)[0]
    rec, arrays = process_row(bundle, row, CFG, _ctx(bundle))
    assert len(rec["windows"]) == len(R.window_grid(row["audio"]["resampled_num_samples"]))
    assert arrays["lid_probs"].shape == (len(rec["windows"]), 100)
    for l in rec["constructions"].values():                                # invalid vectors keep complete records
        assert l["primary"]["status"] in ("ok", "invalid") and "candidates" in l["primary"]


# 7. gates, bootstrap, label precedence -------------------------------------------------------------------------------
def test_nested_gate_precedence_and_partial_does_not_advance():
    from experiments.inference_cf_r0_analyze import decide
    g = CFG["gates"]
    L = lambda O=True, P=True, S=True, I=True: {"O": O, "P": P, "S": S, "I": I}
    assert decide(False, {"16": L()}, g)["label"] == "R0_INVALID"
    assert decide(True, {"3": L(O=False), "16": L(O=False)}, g)["label"] == "R0_ORACLE_CONSTRUCTION_INSUFFICIENT"
    assert decide(True, {"3": L(P=False, S=False, I=False)}, g)["label"] == "R0_PREDICTED_REGION_INSUFFICIENT"
    assert decide(True, {"3": L(S=False, I=False)}, g)["label"] == "R0_VECTOR_UNSTABLE"
    assert decide(True, {"3": L(I=False)}, g)["label"] == "R0_REGION_SIGNAL_INSUFFICIENT"
    assert decide(True, {"3": L(), "8": L(), "16": L(I=False), "24": L(I=False)}, g)["label"] == "R0_PARTIAL_FEASIBILITY"
    assert decide(True, {"3": L(), "16": L(I=False)}, g)["label"] == "R0_PARTIAL_FEASIBILITY"
    assert decide(True, {"3": L(), "24": L()}, g)["label"] == "R0_READY_FOR_R1"


def test_bootstrap_shared_draws_bonferroni8_and_empty_strata():
    from experiments.inference_cf_r0_analyze import draws, macro_boot
    b = CFG["bootstrap"]
    assert b["lower_quantile"] == 0.05 / 16 and b["family_size"] == 8
    keys = [f"D{i:02d}" for i in range(20)]
    W = draws(keys, 2000, 240924)
    assert np.array_equal(W, draws(keys, 2000, 240924)) and np.all(W.sum(axis=1) == 20)
    one = macro_boot([("D00", 1.0), ("D00", 3.0)], keys, W, b["lower_quantile"], b["upper_quantile"])
    assert one["estimate"] == 2.0 and one["usable_draws"] < 2000                 # draws missing D00 are unusable
    assert macro_boot([], keys, W, .025, .975)["estimate"] is None
    full = macro_boot([(k, float(i)) for i, k in enumerate(keys)], keys, W, .025, .975)
    assert full["estimate"] == 9.5 and full["usable_draws"] == 2000


def test_stability_counts_invalid_perturbations_against_all_primary_valid():
    from experiments.inference_cf_r0_analyze import gate_S, stability
    rows = [{"erode_cos": .95, "dilate_cos": .97}] * 8 + [{"erode_cos": None, "dilate_cos": .99}] * 2
    st = stability(rows, CFG["gates"])
    assert st["both_valid_fraction"] == .8 and st["stable_fraction"] == .8 and gate_S(st, CFG["gates"])
    st2 = stability(rows[:7] + [{"erode_cos": .5, "dilate_cos": .99}] + rows[8:], CFG["gates"])
    assert not gate_S(st2, CFG["gates"])                                      # no survivor-only pass


# 8/6. firewall and gating --------------------------------------------------------------------------------------------
def test_provider_firewall_and_oracle_gating():
    from experiments.inference_cf_r0_audit import provider_hits
    for rel in ("src/csasr/inference_cf/r0_regions.py", "src/csasr/inference_cf/r0_unique.py", "experiments/inference_cf_r0.py"):
        assert provider_hits(rel) == [], rel
    an = (ROOT / "experiments/inference_cf_r0_analyze.py").read_text()
    prim = an[an.index("def primary("):an.index("def oracle_tracks(")]
    assert "parquet" not in prim and "oracle_tracks" not in prim
    assert an.index("committed(SEAL)") < an.index("committed(PRIMARY_AUDIT)") < an.index("oracle_tracks([r")
    au = (ROOT / "experiments/inference_cf_r0_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(r0_regions|r0_unique|inference_cf_r0\b|inference_cf_r0_analyze)", au, re.M) is None
    from experiments.inference_cf_r0_analyze import oracle
    if not (ROOT / "results/inference_cf/r0/output_seal.json").exists():
        with pytest.raises((PermissionError, FileNotFoundError)):
            oracle("results/inference_cf/r0/run1")


def test_boundary_matching_one_to_one():
    from experiments.inference_cf_r0_analyze import boundary_match
    n, errs = boundary_match(np.array([100, 5000]), np.array([90, 120, 9000]), 3200)
    assert n == 1 and errs == [10]
