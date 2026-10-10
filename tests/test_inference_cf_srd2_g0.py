"""SRD2-G0 focused CPU tests: frozen pins, raw-R2 vs processed-D2 semantics, query identity, causal replay,
DG-02 site consumption, D2 probe/tangent, native-BF16 NormPreserve pulses (tiny random Whisper in bfloat16 on
CPU), exact no-op/restore, shuffle multiset, firewall/allowlists, lossless storage, evaluator mapping and
label precedence on synthetic fixtures, and auditor independence. No real model forward, no reference text,
no GPU."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import experiments.inference_cf_srd2_g0 as runner  # noqa: E402
import experiments.inference_cf_srd2_g0_audit as aud  # noqa: E402
import experiments.inference_cf_srd2_g0_evaluate as ev  # noqa: E402
from csasr.inference_cf import srd2_g0 as S  # noqa: E402
from csasr.inference_cf.core import atomic_json, digest, file_hash  # noqa: E402
from csasr.inference_cf.core_r2 import conflict_from_logits, gate_values, local_support, max_attention_window  # noqa: E402

CFG = json.loads((ROOT / runner.CONFIG).read_text())
VOCAB, EOS, FRAMES = 64, 59, 100
PROMPT = [60, 61, 62, 63]
PART = {"matrix_ids": list(range(8, 34)), "embedded_ids": list(range(34, 45)),
        "ambiguous_ids": list(range(0, 8)) + list(range(45, VOCAB))}


class Tok:
    eos_token_id = EOS
    all_special_ids = [59, 60, 61, 62, 63]

    def convert_ids_to_tokens(self, i):
        return "abcdefghijklmnopqrstuvwxyz"[int(i) % 26]

    def decode(self, ids, **kwargs):
        return "".join(self.convert_ids_to_tokens(i) for i in ids)


def tiny_bundle(dtype=torch.bfloat16, seed=0):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(seed)
    cfg = WhisperConfig(vocab_size=VOCAB, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=25,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=FRAMES, max_target_positions=64, pad_token_id=0, bos_token_id=60,
                        eos_token_id=EOS, decoder_start_token_id=60, dropout=0.0, attention_dropout=0.0)
    model = WhisperForConditionalGeneration(cfg).eval().to(dtype)
    model.set_attn_implementation("eager")
    model.requires_grad_(False)
    model.generation_config.alignment_heads = [[1, 0], [24, 1]]
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=Tok()), device="cpu", dtype=dtype,
                           d_model=16, num_decoder_layers=25, sample_rate=16000,
                           decoder_layer=lambda i: model.model.decoder.layers[i])


def encoded(dtype=torch.bfloat16, seed=1):
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(seed)
    return BaseModelOutput(last_hidden_state=torch.randn((1, FRAMES, 16)).to(dtype))


def make_ctx(bundle, max_new=8):
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    ctx = object.__new__(runner.Ctx)
    ctx.bundle, ctx.cfg, ctx.tok, ctx.eos = bundle, CFG, bundle.processor.tokenizer, EOS
    ctx.suppress, ctx.begin = [60, 61, 62, 63], []
    ctx.partition, ctx.nfp, ctx.language_ids, ctx.en_id, ctx.zh_id = PART, 4, (4, 5), 4, 5
    ctx.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    ctx.e_star, ctx.max_new, ctx.layer, ctx.prompt = float(CFG["dose"]["e_star"]), max_new, 16, list(PROMPT)
    return ctx


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """One tiny-model Job-A capture (bf16 CPU) reused by the pulse tests."""
    import experiments.inference_cf_p0_r2 as r2
    import csasr.models.whisper as W
    tmp = tmp_path_factory.mktemp("srd2")
    audio = tmp / "u.wav"
    audio.write_bytes(b"synthetic-audio-bytes")
    bundle = tiny_bundle()
    ctx = make_ctx(bundle)
    enc = encoded()
    mp = pytest.MonkeyPatch()
    mp.setattr(runner, "encode", lambda b, path: (enc, {"features_sec": 0.0, "encoder_sec": 0.0,
                                                         "encoder_sha256": S.tensor_sha(enc.last_hidden_state)}))
    mp.setattr(W, "load_audio", lambda path, sr=16000: np.ones(FRAMES * 320, dtype=np.float32))
    lid_calls = []

    def fake_lid(b, w, ids):
        lid_calls.append(len(w))
        return {4: .9, 5: .1}
    mp.setattr(r2, "native_lid", fake_lid)
    row = {"canonical_index": 0, "utterance_id": "UTEST", "audio_path": str(audio), "audio_full_sha256": file_hash(audio)}
    prims = {"max_attention_window": max_attention_window, "conflict_from_logits": conflict_from_logits,
             "local_support": local_support, "gate_values": gate_values, "reason": r2._reason}
    (tmp / "gate").mkdir()
    res = runner.capture_utterance(ctx, row, {4: .5, 5: .5}, prims, tmp / "gate", {"manifest_hash": "sha256:test"})
    with np.load(tmp / "gate" / res["arrays"]["file"]) as z:
        arr = {k: z[k] for k in z.files}
    yield SimpleNamespace(res=res, arr=arr, ctx=ctx, row=row, tmp=tmp, bundle=bundle, mp=mp, lid_calls=lid_calls)
    mp.undo()


# ---- 1. frozen pins -----------------------------------------------------------------------------------

def test_frozen_config_freeze_and_population_hashes():
    assert file_hash(ROOT / runner.CONFIG) == runner.CONFIG_SHA
    fr = json.loads((ROOT / runner.FREEZE).read_text())
    assert all(file_hash(ROOT / p) == h for p, h in fr["files"].items())
    P = json.loads((ROOT / runner.POPULATION).read_text())
    assert digest([r["utterance_id"] for r in P["selected"]]) == CFG["population"]["selected_ids_hash"]
    assert runner.DESIGN_COMMIT.startswith("3168508e") and CFG["dose"]["e_star"] == 1.1260757575454359


def test_runtime_projection_is_exactly_four_keys_and_rejects_extras():
    P = json.loads((ROOT / runner.POPULATION).read_text())
    panel = S.runtime_projection(P)
    assert len(panel["rows"]) == 400 and all(set(r) == set(S.RUNTIME_KEYS) for r in panel["rows"])
    S.validate_runtime_panel(panel)
    for mutate in (lambda p: p["rows"][3].update(stratum="EN-confusion"), lambda p: p.update(dialogues=[1]),
                   lambda p: p["rows"][0].update(audio_path={"nested": "x"}), lambda p: p["rows"][1].pop("audio_path")):
        bad = json.loads(json.dumps(panel))
        mutate(bad)
        bad["runtime_hash"] = digest({k: v for k, v in bad.items() if k != "runtime_hash"})
        with pytest.raises(ValueError):
            S.validate_runtime_panel(bad)
    assert CFG["firewall"]["runtime_input_allowlist"] == list(S.RUNTIME_KEYS)


# ---- 2/3/4. raw R2 vs processed D2 semantics, partition, suppression ----------------------------------------

def test_r2_gate_uses_raw_logits_and_d2_uses_processed_logits():
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.readout import objective
    torch.manual_seed(0)
    z = torch.randn(VOCAB) * 2
    z[40] = 6.0                                       # a suppressed embedded (Latin) token
    prims = {"max_attention_window": max_attention_window, "conflict_from_logits": conflict_from_logits,
             "local_support": local_support, "gate_values": gate_values, "reason": r2._reason}
    heads = np.random.default_rng(0).random((2, FRAMES))
    g = S.r2_gate(utf8_complete=True, heads_q=heads, heard_samples=FRAMES * 320, raw_logits=z, partition=PART,
                  lid=lambda a, b: {4: .9, 5: .1}, null_probs={4: .5, 5: .5}, en_id=4, zh_id=5, primitives=prims)
    p = torch.softmax(z.double(), 0)
    assert g["baseline"]["P_E"] == pytest.approx(float(p[PART["embedded_ids"]].sum()), rel=1e-5)
    pr = objective(z, 3, [40], [], PART)
    assert math.exp(float(pr["log_PE"])) < g["baseline"]["P_E"] - .05        # suppression changes D2's objective only
    assert g["g"] == pytest.approx(g["E"] * max(0, g["baseline"]["P_M"] - g["baseline"]["P_E"]), abs=1e-12)
    assert "raw" in CFG["gate"]["conflict_logits"] and "generation-suppressed" in CFG["D2"]["probabilities"]


def test_r2_gate_fallback_precedence_and_zero():
    import experiments.inference_cf_p0_r2 as r2
    prims = {"max_attention_window": max_attention_window, "conflict_from_logits": conflict_from_logits,
             "local_support": local_support, "gate_values": gate_values, "reason": r2._reason}
    z = torch.randn(VOCAB)
    calls = []
    g = S.r2_gate(utf8_complete=False, heads_q=np.zeros((2, FRAMES)), heard_samples=FRAMES * 320, raw_logits=z, partition=PART,
                  lid=lambda a, b: calls.append(1) or {4: .9, 5: .1}, null_probs={4: .5, 5: .5}, en_id=4, zh_id=5, primitives=prims)
    assert g["fallback_reason"] == "mid_character" and g["g"] == 0.0 and not calls      # zero attention also fails; precedence
    z[3] = float("nan")
    g = S.r2_gate(utf8_complete=True, heads_q=np.ones((2, FRAMES)), heard_samples=FRAMES * 320, raw_logits=z, partition=PART,
                  lid=lambda a, b: {4: .9, 5: .1}, null_probs={4: .5, 5: .5}, en_id=4, zh_id=5, primitives=prims)
    assert g["fallback_reason"] == "nonfinite_signal" and g["g"] == 0.0


def test_tokenizer_partition_and_suppression_pins():
    from transformers import WhisperTokenizer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    tok = WhisperTokenizer.from_pretrained(CFG["model"]["dir"], local_files_only=True)
    part = tokenizer_partition(tok)
    assert part["hash"] == CFG["gate"]["partition_hash"]
    gen = json.loads((Path(CFG["model"]["dir"]) / "generation_config.json").read_text())
    assert digest({"suppress": gen["suppress_tokens"], "begin": gen["begin_suppress_tokens"]}) == CFG["model"]["suppression_hash"]
    assert gen["alignment_heads"] == CFG["gate"]["alignment_heads"]


# ---- 5/6. query identity, EOS/cap/UTF-8 ------------------------------------------------------------------

def test_inventory_query_index_eos_and_cap():
    assert S.inventory([5, 6, 7], "eos") == [0, 1, 2, 3] and S.inventory([5, 6, 7], "cap") == [0, 1, 2]
    assert S.inventory([], "eos") == [0]
    assert [S.query_index(t) for t in (0, 1, 5)] == [3, 4, 8]
    assert S.feed_tokens([9, 8, 7], 0) == list(S.CB) and S.feed_tokens([9, 8, 7], 2) == [8]
    assert S.expected_action([9, 8], 2) == S.EOS and S.expected_action([9, 8], 1) == 8
    with pytest.raises(ValueError):
        S.inventory([1], "replay_mismatch")


def test_structural_eligibility_reasons():
    assert S.structural(1, [5], True, True, True) == (True, [])
    assert S.structural(0, [], True, True, True)[1] == ["forced_prompt_query_t0"]
    assert "mid_character_prefix" in S.structural(2, [5, 6], False, True, True)[1]
    assert "unexpected_special_content_token" in S.structural(2, [5, 50365], True, True, True)[1]
    assert "nonfinite_native_logits" in S.structural(2, [5, 6], True, True, False)[1]


def test_prefix_utf8_complete_on_split_han_character():
    from transformers import WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import prefix_utf8_complete
    tok = WhisperTokenizer.from_pretrained(CFG["model"]["dir"], local_files_only=True)
    bd = {v: k for k, v in bytes_to_unicode().items()}
    ids = tok.encode("我们用租母开会", add_special_tokens=False)
    assert prefix_utf8_complete(tok, ids[:2], bd) and not prefix_utf8_complete(tok, ids[:3], bd) and prefix_utf8_complete(tok, ids[:4], bd)


# ---- 7/8/9. tiny-model Job A: cached vs full replay, causality, DG-02 consumption --------------------------

def test_capture_schema_inventory_and_cached_replay_bitwise(captured):
    res, arr = captured.res, captured.arr
    inv = res["inventory"]
    assert res["status"] == "ok" and inv == S.inventory(res["baseline"]["content_ids"], res["baseline"]["terminated"])
    assert res["baseline"]["replay_bitwise"] and res["baseline"]["lineage_ok"]
    assert arr["cached_logits"].shape == (len(inv), VOCAB) and arr["heads"].shape == (len(inv), 2, FRAMES)
    assert [q["query"] for q in res["queries"]] == [3 + t for t in inv]
    for q in res["queries"]:
        assert q["cached_argmax"] == q["expected_action"]
        if q["structural"]["eligible"]:
            c = q["compat"]
            assert c["TV"] < .05 and c["site_cos"] > .99 and c["PE_abs_diff"] < .02
    assert any(q["structural"]["eligible"] for q in res["queries"])
    assert res["full_replay"]["future_mass_max"] <= 1e-7
    assert all(p["TV"] <= .05 and p["attention_L1"] <= .02 for p in res["probes"])
    # lossless: the stored bf16 bits reproduce the recorded gate's raw script masses
    i = next(k for k, q in enumerate(res["queries"]) if q["gate"]["baseline"] is not None)
    lf = torch.from_numpy(S.from_bf16_bits(arr["full_logits"][i]))
    assert conflict_from_logits(lf, PART)["P_E"] == res["queries"][i]["gate"]["baseline"]["P_E"]


def test_lid_cache_reuses_identical_crops_only(captured):
    res = captured.res
    keys = {(q["gate"]["lid"]["start_sample"], q["gate"]["lid"]["end_sample"]) for q in res["queries"] if q["gate"]["lid"]}
    assert res["lid"]["calls"] == len(keys) == len(captured.arr["lid_keys"])


def test_full_replay_future_token_independence():
    import experiments.inference_cf_p0_r2 as r2
    bundle = tiny_bundle(torch.float32)
    enc = encoded(torch.float32)
    a, ha = r2.full_replay(bundle, enc, PROMPT, [5, 6, 7, 8], attention=True)
    b, hb = r2.full_replay(bundle, enc, PROMPT, [5, 6, 30, 31], attention=True)
    assert torch.allclose(a[:6], b[:6], atol=1e-5) and torch.allclose(ha[:, :6], hb[:, :6], atol=1e-6)


def test_native_site_ffn_input_equals_site_on_clean_step():
    import experiments.inference_cf_p2r as p2r
    bundle = tiny_bundle()
    br = p2r.DiagBranch(bundle, encoded(), PROMPT, "B")
    obs = S.NativeSite(bundle, 16)
    with torch.inference_mode():
        br.step(PROMPT, capture_layer=None, attention=True, hook=obs)
    assert torch.equal(obs.ffn, obs.r) and obs.r.dtype == torch.bfloat16
    S.assert_no_hooks_anywhere(bundle)


# ---- 10-17. D2, pulses, no-op, restore (tiny bf16 Job B) ----------------------------------------------------

@pytest.fixture(scope="module")
def pulsed(captured):
    res = captured.res
    gates = {q["t"]: q["g"] for q in res["queries"] if q["structural"]["eligible"]}
    perm = S.permute("UTEST", gates)
    eng = runner.Engine(captured.ctx)
    (captured.tmp / "pulses").mkdir(exist_ok=True)
    with torch.inference_mode():
        out = runner.pulse_utterance(eng, captured.row, res, captured.arr, perm, captured.tmp / "pulses", {"manifest_hash": "sha256:test"})
    with np.load(captured.tmp / "pulses" / out["arrays"]["file"]) as z:
        arr = {k: z[k] for k in z.files}
    with np.load(captured.tmp / "pulses" / out["logits"]["file"]) as z:
        lg = {k: z[k] for k in z.files}
    return SimpleNamespace(out=out, arr=arr, lg=lg, perm=perm, eng=eng)


def test_job_b_clean_state_matches_job_a_bitwise_and_d2_scratch_identity(pulsed):
    out = pulsed.out
    assert out["status"] == "ok"
    for q in out["queries"]:
        assert q["clean"]["logits_bitwise_vs_A"] and q["clean"]["site_bitwise_vs_A"]
        assert set(q["arms"]) == {"B0", "B1", "B2", "B3"}
        if q["structural"]:
            d = q["d2"]
            assert d["status"] == "ok" and d["scratch_logits_bitwise"] and d["scratch_site_bitwise"]
            assert d["counters"] == {"autograd_calls": 1, "scratch_forwards": 1}
            assert d["unit_error"] <= 2e-6 and d["radial_dot"] <= 1e-6
        else:
            assert all(not q["arms"][a]["executed"] and q["arms"][a]["top1"] == q["arms"]["B0"]["top1"] for a in S.EDIT_ARMS)


def test_no_parameter_gradients_and_no_hooks_after_job_b(captured, pulsed):
    assert all(p.grad is None and not p.requires_grad for p in captured.bundle.model.parameters())
    S.assert_no_hooks_anywhere(captured.bundle)


def test_executed_pulses_consume_the_native_preview_bitwise(pulsed):
    out = pulsed.out
    executed = [(q, a) for q in out["queries"] if q["structural"] for a in S.EDIT_ARMS if q["arms"][a]["executed"]]
    assert executed, "B1 at e* must execute on the tiny model"
    for q, a in executed:
        r = q["arms"][a]
        assert r["integrity_ok"] and r["solver_calls"] == 1 and r["ffn_equals_preview_bitwise"] and r["q_bitwise"]
        assert r["guards"]["pass_all"] and r["consumed_chord_actual"] == pytest.approx(r["target"], rel=.02)
        assert r["solver"]["rel_sq_err"] <= .02 and r["solver_matches_hook"]
    assert all(q["restore_bitwise"] for q in out["queries"] if q["structural"] and q["restore_bitwise"] is not None)
    assert len(pulsed.lg["exec_logits"]) == len(executed) == len(pulsed.arr["exec_ffn"])


def test_targets_are_gate_scaled_chords_from_sealed_gates_and_shuffle(captured, pulsed):
    e = CFG["dose"]["e_star"]
    pairs = {p["recipient_t"]: p for p in pulsed.perm["pairs"]}
    gq = {q["t"]: q for q in captured.res["queries"]}
    for q in pulsed.out["queries"]:
        if not q["structural"]:
            continue
        g, gs = gq[q["t"]]["g"], pairs[q["t"]]["g_shuffled"]
        assert q["arms"]["B1"]["target"] == e
        assert q["arms"]["B2"]["target"] == (0.0 if g == 0 else e * g)
        assert q["arms"]["B3"]["target"] == (0.0 if gs == 0 else e * gs)


def test_zero_target_bypass_and_tiny_target_no_floor(captured):
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.readout import readout_direction
    ctx = captured.ctx
    eng = runner.Engine(ctx)
    content = captured.res["baseline"]["content_ids"]
    t = next(q["t"] for q in captured.res["queries"] if q["structural"]["eligible"])
    with torch.inference_mode():
        br = p2r.DiagBranch(ctx.bundle, encoded(), PROMPT, "B")
        for s in range(t):
            eng.step(br, S.feed_tokens(content, s, PROMPT))
        new = S.feed_tokens(content, t, PROMPT)
        L = br.length
        d2 = readout_direction(ctx.bundle, layer=16, cache=br.cache, encoded=encoded(), new_tokens=new, start=L, step=t,
                               suppress=ctx.suppress, begin=ctx.begin, partition=PART)
        l0, o0, _ = eng.step(br, new)
        r0, la, _ = eng.arm(br, L, new, t, L + len(new) - 1, d2["direction"], 0.0, o0)
        assert r0["status"] == "zero_target" and r0["solver"] is None and la is None
        r1, la, _ = eng.arm(br, L, new, t, L + len(new) - 1, None, 0.5, o0)
        assert r1["status"] == "invalid_direction" and la is None
        rt, la, _ = eng.arm(br, L, new, t, L + len(new) - 1, d2["direction"], 1e-5, o0)
        assert not rt["executed"] and rt["target"] == 1e-5 and rt["status"] in (
            "solver_target_unattainable", "native_consumption_unattainable", "energy_unreachable")
        # zero-gain DG-02 hook is value-identical (bitwise) to the clean step
        from csasr.inference_cf.loc0_sites import cross_pulse_hook
        br.crop(L)
        hk = cross_pulse_hook(ctx.bundle, 16, lambda q=None, u_source=None, r=None, abs_pos=None: (
            torch.zeros((1, r.shape[1]), dtype=r.dtype), torch.zeros_like(r)), 4)
        lz, oz, _ = eng.step(br, new, hook=hk)
        assert torch.equal(lz, l0) and torch.equal(oz.ffn, o0.ffn)


def test_cached_solver_refuses_second_distinct_solve():
    r = torch.randn(16).to(torch.bfloat16)
    v = torch.randn(16)
    cs = S.CachedSolver(r, v, .5, {"status": "ok", "s": 1.0})
    assert cs(r.clone(), v, .5)["s"] == 1.0
    with pytest.raises(RuntimeError):
        cs(r + 1, v, .5)
    with pytest.raises(RuntimeError):
        cs(r, v, .25)


def test_chord_guard_boundaries_and_norm_ratio():
    d = CFG["dose"]
    assert S.chord_guards(1.0, 1.0, 1.0, d)["pass_all"]
    assert not S.chord_guards(1.0, 1.0, 1.006, d)["pass_all"]           # .005 norm-ratio guard
    assert not S.chord_guards(1.0, 1.03, 1.03, d)["pass_all"]           # >2 % chord / squared energy
    assert S.chord_guards(1.0, 1.004, 1.008, d)["pass_all"]
    assert not S.chord_guards(0.0, 1.0, 1.0, d)["pass_all"]


# ---- 18/19. shuffle -------------------------------------------------------------------------------------

def test_permutation_is_hash_ordered_within_utterance_and_preserves_multiset():
    gates = {1: 0.0, 2: .2, 4: 0.0, 5: .7, 8: .4}
    p = S.permute("UID", gates)
    donors = sorted(gates, key=lambda t: (hashlib.sha256(f"SRD2-G0-gate-permutation-v1|240924|UID|{t}".encode()).hexdigest(), t))
    assert [x["donor_t"] for x in p["pairs"]] == donors and [x["recipient_t"] for x in p["pairs"]] == sorted(gates)
    assert S.bit_multiset([x["g"] for x in p["pairs"]]) == S.bit_multiset([x["g_shuffled"] for x in p["pairs"]])
    assert S.planned_sq_total([x["g"] for x in p["pairs"]]) == S.planned_sq_total([x["g_shuffled"] for x in p["pairs"]])
    assert S.permute("UID", gates) == p                                              # deterministic, no reroll
    assert S.permute("U", {3: .5})["singleton"] and S.permute("U", {3: .5})["uninformative"]
    c = S.permute("U", {1: .3, 2: .3, 9: .3})
    assert c["constant"] and c["uninformative"]
    assert S.permute("U", {})["n"] == 0 and not S.permute("U", {})["uninformative"]
    assert [x for x in aud.permutation_ref("UID", gates)] == [(x["recipient_t"], x["donor_t"], x["g_shuffled"]) for x in p["pairs"]]


# ---- 20/21. firewall, serialization ----------------------------------------------------------------------

def test_runner_and_mechanics_import_no_reference_code():
    for rel in (runner.__file__, S.__file__):
        tree = ast.parse(Path(rel).read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and (m.startswith("csasr.evaluation") or "evaluate" in m or "population" in m or "pyarrow" in m
                              or m.endswith("srd2_g0_audit") or "p2rj" in m) for m in mods), rel
    s = aud.static_scan("experiments/inference_cf_srd2_g0.py")
    assert not [c for c in s["constants"] if any(b in c for b in (".parquet", "transcript", "target_set", "stratum", "/roles/"))]


def test_auditor_never_imports_primary_code():
    s = aud.static_scan("experiments/inference_cf_srd2_g0_audit.py")
    assert not {"experiments.inference_cf_srd2_g0", "experiments.inference_cf_srd2_g0_evaluate", "csasr.inference_cf.srd2_g0"} & set(s["imports"])


def test_cli_rejects_other_config_or_root(monkeypatch):
    for argv in (["x", "prepare", "--config", "configs/other.json"], ["x", "prepare", "--out", "results/elsewhere"]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            runner.main()


def test_lossless_bf16_storage_and_json_finite(tmp_path):
    x = torch.randn(1000).to(torch.bfloat16).float()
    assert np.array_equal(S.from_bf16_bits(S.bf16_bits(x)), x.numpy())
    assert np.array_equal(aud.unbf16(S.bf16_bits(x)), x.numpy())
    with pytest.raises(ValueError):
        S.bf16_bits(torch.tensor([1.0000001]))
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "x.json", {"a": float("nan")})


# ---- 22. evaluator mapping on synthetic fixtures (real tokenizer, synthetic text) -------------------------

@pytest.fixture(scope="module")
def wtok():
    from transformers import WhisperTokenizer
    return WhisperTokenizer.from_pretrained(CFG["model"]["dir"], local_files_only=True)


def test_mapping_confusion_target_set_and_mandarin_dedup(wtok):
    content = wtok.encode("我们用租母开会", add_special_tokens=False)
    qs, led = ev.map_utterance(wtok, "U1", "我们用zoom开会", content, "eos", 5.0)
    by = {q["t"]: q for q in qs}
    # canonical alignment substitutes 'zoom' with the later transliteration unit (母, token 4); 租 is an insertion
    assert by[4]["stratum"] == "EN-confusion" and content[4] not in by[4]["Y"] and 8863 in by[4]["Y"]
    zh = [q for q in qs if q["stratum"] == "ZH-correct"]
    assert [q["t"] for q in zh] == [0, 1, 5, 6]                        # 我/们 share token 0 -> one query
    assert all(q["Y"] == [content[q["t"]]] for q in zh)
    assert led["latin_units"] == 1 and led["latin_mapped"] == 1
    own, lt, lm = aud.own_map(wtok, "我们用zoom开会", content, "eos", 5.0)
    assert {t: (s, Y) for t, (s, Y) in own.items()} == {q["t"]: (q["stratum"], q["Y"]) for q in qs} and (lt, lm) == (1, 1)


def test_mapping_heard_scope_and_english_correct(wtok):
    content = wtok.encode("我们用zoom开会", add_special_tokens=False)
    qs, led = ev.map_utterance(wtok, "U2", "我们用zoom开会", content, "eos", 31.0)
    assert qs == [] and led["latin_mapped"] == 0 and led["latin_units"] == 1
    qs, _ = ev.map_utterance(wtok, "U2", "我们用zoom开会", content, "eos", 5.0)
    en = [q for q in qs if q["stratum"] == "EN-correct"]
    assert len(en) == 1 and content[en[0]["t"]] in en[0]["Y"]
    own, _, _ = aud.own_map(wtok, "我们用zoom开会", content, "eos", 5.0)
    assert {t: s for t, (s, _) in own.items()} == {q["t"]: q["stratum"] for q in qs}


def test_mapping_deletion_gap_collides_with_mandarin_and_is_excluded(wtok):
    content = wtok.encode("我们用开会", add_special_tokens=False)
    qs, _ = ev.map_utterance(wtok, "U3", "我们用zoom开会", content, "eos", 5.0)
    own, _, _ = aud.own_map(wtok, "我们用zoom开会", content, "eos", 5.0)
    t_open = 2                                                        # '开' query = deletion gap slot of 'zoom'
    assert content[t_open] == wtok.encode("开", add_special_tokens=False)[0]
    assert t_open not in {q["t"] for q in qs} and t_open not in own
    assert {q["t"] for q in qs} == set(own) == {0, 1, 3}


def test_outcomes_tally_and_utility(wtok):
    content = wtok.encode("我们用租母开会", add_special_tokens=False)
    qs, _ = ev.map_utterance(wtok, "U1", "我们用zoom开会", content, "eos", 5.0)
    zoom = next(q for q in qs if q["stratum"] == "EN-confusion")

    def arms(top_b1, top_b2, top_b3, b0):
        return {"B0": {"top1": b0}, **{a: {"top1": x, "executed": True, "status": "executed"} for a, x in
                                        (("B1", top_b1), ("B2", top_b2), ("B3", top_b3))}}
    pulses = {"U1": {"queries": [{"t": q["t"], "structural": True,
                                   "arms": arms(8863 if q is zoom else 1, 8863 if q is zoom else content[q["t"]],
                                                content[q["t"]], content[q["t"]])} for q in qs]}}
    rows = ev.outcomes(qs, pulses, {"U1": "D1"})
    T = ev.tally(rows, ["D1"])["D1"]
    assert T["B1_C"] == 1 and T["B2_C"] == 1 and T["B3_C"] == 0
    assert T["B1_H_ZH"] == 4 and T["B2_H_ZH"] == 0 and T["B1_act_ZH"] == 4


# ---- 23. label precedence -----------------------------------------------------------------------------------

def test_terminal_precedence_evaluator_and_auditor_agree():
    rng = np.random.default_rng(0)
    names = list(CFG["gate_predicates"])
    for _ in range(300):
        gates = {k: bool(rng.integers(0, 2)) for k in names}
        flags = {"critical_failure": bool(rng.random() < .2), "independent_FULL_audit_PASS": bool(rng.random() < .8)}
        assert ev.decide(CFG, gates, flags) == aud.first_label(CFG, gates, flags)
    allpass = {k: True for k in names}
    allpass["severe_damage"] = False
    ok = {"critical_failure": False, "independent_FULL_audit_PASS": True}
    assert ev.decide(CFG, allpass, ok) == "SRD2_G0_SELECTIVE_FEASIBILITY"
    assert ev.decide(CFG, {**allpass, "resource": False}, ok) == "SRD2_G0_COMPUTE_BLOCKED"
    assert ev.decide(CFG, {**allpass, "severe_damage": True, "opportunities": False}, ok) == "SRD2_G0_OBSERVED_DAMAGE"
    assert ev.decide(CFG, {**allpass, "causal_power": False, "control_opportunities": False}, ok) == "SRD2_G0_CAUSAL_POWER_INSUFFICIENT"
    assert ev.decide(CFG, {**allpass, "energy_execution": False}, ok) == "SRD2_G0_INVALID"


def test_predicates_never_pass_on_missing_and_agree():
    rule = CFG["gate_predicates"]["acceptance"]
    m = {r["metric"]: r["value"] for r in rule["all"]}
    assert ev.predicate(rule, m) and aud.pred(rule, m)
    for k in m:
        bad = {**m, k: None}
        assert not ev.predicate(rule, bad) and not aud.pred(rule, bad)
    leaves = ev.leaf_report(rule, {**m, "U_B2": None})
    assert any(x["metric"] == "U_B2" and x["status"] == "NOT_ESTIMABLE" for x in leaves)


def test_bootstrap_is_deterministic_and_paired():
    T = {d: {"n_C": 2, "n_EN": 3, "n_ZH": 10, **{f"{a}_{k}": (i % 3) for a in ev.ARMS for k in ("C", "H_EN", "H_ZH", "act_ZH")}}
         for i, d in enumerate(f"D{i:02d}" for i in range(20))}
    b1 = ev.bootstrap(T, sorted(T), CFG)
    b2 = ev.bootstrap(T, sorted(T), CFG)
    assert b1 == b2 and b1["family"]["C_B2-C_B3"]["pct95"] == [0.0, 0.0]      # identical arms -> zero paired difference
    assert b1["replicates"] == 10000


# ---- 24. independent recomputation helpers ---------------------------------------------------------------

def test_auditor_window_and_support_match_historical_primitives():
    rng = np.random.default_rng(3)
    for heard in (FRAMES * 320, 77 * 320 + 13, 4000, 1500 * 320 + 9999):
        frames = max(FRAMES, min(1500, (heard + 319) // 320))
        a = rng.random((10, frames))
        a[:, 5:9] = 3.0
        w = max_attention_window(a, heard).record()
        assert aud.window(a, heard) == {k: w[k] for k in ("start_frame", "end_frame", "start_sample", "end_sample")}
    for pe, pm in ((.9, .05), (.2, .7), (.31735, .019)):
        assert aud.support_E(pe, pm, .3173545002937317, .01905861869454384) == \
            local_support(pe, pm, .3173545002937317, .01905861869454384)["E"]


def test_auditor_script_mass_matches_r2_conflict():
    t = object.__new__(aud.Tok)
    t.E, t.M = np.asarray(PART["embedded_ids"]), np.asarray(PART["matrix_ids"])
    z = torch.randn(VOCAB).to(torch.bfloat16).float()
    pe, pm = t.masses(z.numpy())
    c = conflict_from_logits(z, PART)
    assert abs(pe - c["P_E"]) < 1e-6 and abs(pm - c["P_M"]) < 1e-6


# ---- independent auditor on the real tiny-model arrays ----------------------------------------------------

def fake_audit_tok():
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    T = object.__new__(aud.Tok)
    T.tok, T.E, T.M = Tok(), np.asarray(PART["embedded_ids"]), np.asarray(PART["matrix_ids"])
    T.suppress, T.begin, T.lang, T.en, T.zh = [60, 61, 62, 63], [], [4, 5], 4, 5
    T.bdec = {v: k for k, v in bytes_to_unicode().items()}
    T.special, T.eos = set(Tok.all_special_ids), EOS
    return T


def test_audit_a_recomputes_tiny_gate_row_without_errors(captured):
    rr, info = aud.recompute_gate_row(fake_audit_tok(), captured.res, captured.arr, {"EN": .5, "ZH": .5})
    assert info["errors"] == {} and info["inventory_ok"] and info["query_index_ok"]
    assert [x["eligible"] for x in rr] == [q["structural"]["eligible"] for q in captured.res["queries"]]
    bad = json.loads(json.dumps(captured.res))
    k = next(i for i, q in enumerate(bad["queries"]) if q["g"] > 0)
    bad["queries"][k]["g"] += 1e-3
    _, info2 = aud.recompute_gate_row(fake_audit_tok(), bad, captured.arr, {"EN": .5, "ZH": .5})
    assert info2["errors"].get("g") == 1


def test_primary_recompute_agrees_with_tiny_pulse_matrix(captured, pulsed):
    perm = {"utterances": [pulsed.perm], "uninformative_queries": pulsed.perm["n"] if pulsed.perm["uninformative"] else 0,
            "structural_queries": pulsed.perm["n"]}
    rec = aud.primary_recompute(CFG, fake_audit_tok(), [captured.row], {"UTEST": captured.res}, {"UTEST": pulsed.out}, perm,
                                run=captured.tmp)
    assert rec["errors"] == {}, rec["errors"]
    assert rec["energy_execution"]["solver_hook_identity_pass"] and rec["energy_execution"]["zero_no_edit_restore_bitwise"]
    assert rec["autograd_calls"] == sum(1 for q in pulsed.out["queries"] if q["structural"])
    integ = ev.integrity(CFG, {"UTEST": pulsed.out}, {"UTEST": captured.res}, perm)
    for k, v in rec["energy_execution"].items():
        assert aud.close(v, integ["energy_execution"][k], 1e-12), k
    for k, v in rec["dose_comparison"].items():
        assert aud.close(v, integ["dose_comparison"][k], 1e-12), k


# ---- evaluator vs independent FULL core on a synthetic 20-dialogue matrix -------------------------------------

def _synthetic(wtok):
    e = CFG["dose"]["e_star"]
    conf = wtok.encode("我们用租母开会", add_special_tokens=False)
    corr = wtok.encode("我们用zoom开会", add_special_tokens=False)
    sel, refs, gate, pulses = [], {}, {}, {}
    for i in range(20):
        uid = f"U{i:02d}"
        content = conf if i % 2 == 0 else corr
        sel.append({"utterance_id": uid, "dialogue_id": f"D{i:02d}", "source_duration_sec": 5.0 if i != 19 else 31.0})
        refs[uid] = "我们用zoom开会"
        gate[uid] = {"baseline": {"content_ids": list(content), "terminated": "eos"}}
        qs = []
        for t in range(len(content) + 1):
            b0 = content[t] if t < len(content) else 50257
            top = {"B1": b0, "B2": b0, "B3": b0}
            if i % 2 == 0 and t == 4:
                top["B1"] = 8863
                top["B2"] = 8863 if i % 4 == 0 else b0
                top["B3"] = 8863 if i == 2 else b0
            if t == 1:
                top["B1"] = 1 if i % 3 == 0 else b0
                top["B3"] = 1 if i % 5 == 0 else b0
            if i == 1 and t == 2:
                top["B1"] = 1
            if i == 3 and t == 2:
                top["B2"] = 50257                                    # early-EOS proxy only if >=10 tokens remain (not here)
            st = t >= 1
            arms = {"B0": {"top1": b0, "executed": False, "status": "none"}}
            for a, tg in (("B1", e), ("B2", e * .5), ("B3", e * .5)):
                arms[a] = ({"top1": top[a], "executed": True, "status": "executed", "target": tg, "consumed_chord_actual": tg,
                            "integrity_ok": True, "guards": {"proposed_chord": tg, "proposed_rel_sq_err": 0.0, "proposed_rel_chord_err": 0.0}}
                           if st else {"top1": b0, "executed": False, "status": "structurally_ineligible", "target": 0.0})
            q = {"t": t, "structural": st, "arms": arms, "clean": {"logits_bitwise_vs_A": True, "site_bitwise_vs_A": True},
                 "restore_bitwise": True if st else None}
            if st:
                q["d2"] = {"scratch_logits_bitwise": True, "scratch_site_bitwise": True}
            qs.append(q)
        pulses[uid] = {"utterance_id": uid, "status": "ok", "queries": qs}
    return sel, refs, gate, pulses


def test_evaluator_and_independent_full_core_agree_on_synthetic_matrix(wtok):
    sel, refs, gate, pulses = _synthetic(wtok)
    n_struct = sum(1 for p in pulses.values() for q in p["queries"] if q["structural"])
    perm = {"uninformative_queries": 0, "structural_queries": n_struct}
    jobA = {"rows_complete": 400}
    side = {"gate_seal": {"metrics": jobA, "critical": {"x": True}}, "forecast": {"metrics": {"formula_forecast_seconds": 1.0}},
            "pulse_runtime": {"status": "completed", "weights_unchanged": True, "parameters_with_grad": 0, "parameters_requiring_grad": 0},
            "apparatus": {"ok": True}}
    res = ev.evaluate(CFG, pulses, gate, perm, refs, {"selected": sel}, wtok, side)
    integ = res["integrity"]
    rec = {"energy_execution": integ["energy_execution"], "dose_comparison": integ["dose_comparison"],
           "uninformative_queries": 0, "structural_queries": n_struct}
    T = fake_audit_tok()
    T.tok, T.eos = wtok, 50257
    core = aud.full_core(CFG, T, sel, gate, pulses, refs, rec, jobA, {"formula_forecast_seconds": 1.0})
    m, a = res["metrics"], core["metrics"]
    assert set(m) == set(a)
    assert all(aud.close(m[k], a[k], 1e-12) if isinstance(m[k], float) else m[k] == a[k] for k in m), \
        {k: (m[k], a[k]) for k in m if m[k] != a[k]}
    assert res["gates"] == {k: v for k, v in core["gates"].items()}
    assert res["counts"]["C"] == core["C"] == {"B1": 10, "B2": 5, "B3": 1}            # U19 (odd) is the heard-scope row
    assert res["counts"]["H_ZH"] == core["HZ"] and core["HZ"]["B2"] == 0 and core["HZ"]["B1"] == 7 and core["HZ"]["B3"] == 4
    assert res["counts"]["H_EN"]["B1"] == 1 and m["mapped_EN_confusion"] == 10 and m["mapped_EN_correct"] == 9
    assert res["denominators"]["heard_scope_rows"] == 1 and m["English_mapping_fraction"] == pytest.approx(19 / 20)
    for d in core["dialogues"]:
        assert all(int(res["per_dialogue"][d].get(k, 0)) == int(core["per"][d].get(k, 0)) for k in set(res["per_dialogue"][d]) | set(core["per"][d]))
    assert ev.decide(CFG, res["gates"], {"critical_failure": False, "independent_FULL_audit_PASS": True}) == \
        aud.first_label(CFG, core["gates"], {"critical_failure": False, "independent_FULL_audit_PASS": True})
    b = res["bootstrap"]["family"]["C_B2-C_B3"]["bonferroni_98.75"]
    assert b[0] <= 4 <= b[1]


def test_job_a_seal_metrics_equal_independent_audit_metrics(captured, monkeypatch):
    from collections import Counter
    P = json.loads((ROOT / runner.POPULATION).read_text())
    rows = [{"utterance_id": r["utterance_id"]} for r in P["selected"]]
    gate_rows = {r["utterance_id"]: {**captured.res, "utterance_id": r["utterance_id"]} for r in rows}
    hist = CFG["gate"]
    runtime = {"null": {"EN_abs_error": abs(.5 - hist["null_EN_historical"]), "ZH_abs_error": abs(.5 - hist["null_ZH_historical"])}}
    monkeypatch.setattr(S, "CB", tuple(PROMPT))
    monkeypatch.setattr(aud, "CB", list(PROMPT))
    mine = runner.job_a_metrics(CFG, rows, gate_rows, runtime)
    rr, info = aud.recompute_gate_row(fake_audit_tok(), captured.res, captured.arr, {"EN": .5, "ZH": .5})
    recs = {u: (g, rr) for u, g in gate_rows.items()}
    flags = {u: info for u in gate_rows}
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    theirs = aud.job_a_metrics_independent(CFG, rows, recs, flags, info["probes"] * len(rows), {"EN": .5, "ZH": .5}, dlg, Counter())
    assert set(mine) == set(theirs)
    for k in mine:
        assert aud.close(mine[k], theirs[k], 1e-12) if not isinstance(mine[k], bool) else mine[k] == theirs[k], (k, mine[k], theirs[k])
    assert mine["rows_complete"] == 400 and mine["dialogues_complete"] == 20 and mine["identity_fraction"] == 1.0
    for gate in ("job_A_coverage", "compatibility"):
        assert {r["metric"] for r in CFG["gate_predicates"][gate]["all"]} <= set(mine)


def test_resource_forecast_uses_entire_structural_inventory_and_blocks_missing_throughput(captured):
    row = json.loads(json.dumps(captured.res))
    row["encoder"]["encoder_sec"] = .05                       # the fixture's fake encoder reports 0 s
    gate = {f"U{i}": row for i in range(400)}
    n = 400 * sum(q["structural"]["eligible"] for q in captured.res["queries"])
    zero = runner.resource_forecast(CFG, {f"U{i}": captured.res for i in range(400)}, {"model_load_sec": 30.0})
    assert zero["observed_component_forecast_seconds"] is None and not zero["resource_pass"]   # nonpositive timing cannot pass
    fc = runner.resource_forecast(CFG, gate, {"model_load_sec": 30.0})
    assert fc["inputs"]["actual_structural_queries"] == n
    assert fc["formula_forecast_seconds"] == pytest.approx(.18 * n + 800)
    assert fc["observed_component_per_query"] >= .18 and fc["observed_component_fixed"] >= 800
    assert set(fc["metrics"]) == {r["metric"] for r in CFG["gate_predicates"]["resource"]["all"]}
    bad = runner.resource_forecast(CFG, gate, {"model_load_sec": None})
    assert bad["observed_component_forecast_seconds"] is None and not bad["resource_pass"]
