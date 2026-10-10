"""TTLS-R1 focused tests (CPU; tiny real-class Whisper, no pretrained weights): DG-02 edit masking and causality,
differentiation (float64 finite differences), clean identity, per-utterance reset, frozen candidate / stable rules,
an end-to-end process_row on the tiny model, reference-leakage firewall and config/module agreement."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf import ttls as L  # noqa: E402
from csasr.inference_cf.episodic_tta import LNGuard, decoder_ln_names, forced_decode, teacher_logits  # noqa: E402

LAYER = 1
CB = [1, 10, 11, 12]
CE = [1, 13, 11, 12]


class _Tok:
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in tokens)


class _Proc:
    tokenizer = _Tok()


def _bundle(dtype=torch.bfloat16):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from csasr.models.whisper import WhisperBundle, inspect_modules
    cfg = WhisperConfig(vocab_size=200, num_mel_bins=8, encoder_layers=2, decoder_layers=3, encoder_attention_heads=2,
                        decoder_attention_heads=2, d_model=16, encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=50,
                        max_target_positions=64, decoder_start_token_id=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        suppress_tokens=[])
    cfg._attn_implementation = "eager"                     # as the real decode path (alignment-head attentions)
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval().to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.alignment_heads = [[1, 0]]
    return WhisperBundle(model=model, processor=_Proc(), config=cfg, device="cpu", dtype=dtype, model_id="ttls-tiny",
                         revision="test", encoder_step_sec=0.02, max_encoder_frames=50, num_encoder_layers=2,
                         num_decoder_layers=3, d_model=16, sample_rate=16000, chunk_sec=1.0, module_report=inspect_modules(model))


def _encode(bundle, seed=1):
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(seed)
    feats = torch.randn(1, 8, 100).to(bundle.dtype)
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=feats).last_hidden_state
    return BaseModelOutput(last_hidden_state=h), BaseModelOutput(last_hidden_state=h.clone())


@pytest.fixture(scope="module")
def tb():
    return _bundle()


# ---- site edit ---------------------------------------------------------------------------------------------------

def test_edit_only_at_masked_steps_and_causal(tb):
    _, enc = _encode(tb)
    y = [20, 21, 22, 23, 24, 25]
    z = torch.randn(16) * 3
    with torch.no_grad():
        a = teacher_logits(tb.model, enc, CB, y, None)
        with L.ttls_hook(tb, z, {2}, layer=LAYER, mode="steer", record=True) as h:
            b = teacher_logits(tb.model, enc, CB, y, None)
    assert torch.equal(a[:2], b[:2])                       # content steps before the edit are bitwise unchanged
    assert not torch.equal(a[2], b[2])                     # the edited query (abs 3 + 2) changes
    steered = [r.abs_pos for r in h.records if r.steered]
    assert steered == [L.PROMPT_LEN - 1 + 2]
    rec = next(r for r in h.records if r.steered)
    assert math.isclose(rec.pre_norm, rec.post_norm, rel_tol=2e-2)   # NormPreserve (bf16)


def test_step0_never_edited_and_empty_mask_identity(tb):
    _, enc = _encode(tb)
    y = [20, 21, 22]
    z = torch.randn(16) * 3
    with torch.no_grad():
        a = teacher_logits(tb.model, enc, CB, y, None)
        with L.ttls_hook(tb, z, {0}, layer=LAYER, mode="steer"):
            b = teacher_logits(tb.model, enc, CB, y, None)
        with L.ttls_hook(tb, z, set(), layer=LAYER, mode="steer"):
            c = teacher_logits(tb.model, enc, CB, y, None)
        with L.ttls_hook(tb, z, L.ALL, layer=LAYER, mode="steer"):
            d = teacher_logits(tb.model, enc, CB, y, None)
    assert torch.equal(a, b) and torch.equal(a, c)         # forced prefix (t = 0) and empty mask: exact no-op
    assert torch.equal(a[0], d[0]) and not torch.equal(a[1:], d[1:])


def test_gradient_matches_finite_differences_float64():
    b64 = _bundle(torch.float64)
    _, enc = _encode(b64)
    y = [20, 21, 22, 23]
    z = torch.nn.Parameter(torch.randn(16, dtype=torch.float64) * 0.3)

    def f(zz):
        with L.ttls_hook(b64, zz, {1, 2}, layer=LAYER, mode="train"):
            lg = teacher_logits(b64.model, enc, CB, y, None)
        return -torch.log_softmax(lg[2].double(), -1)[50]
    with torch.enable_grad():
        (g,) = torch.autograd.grad(f(z), z)
    u = torch.randn(16, dtype=torch.float64)
    u /= u.norm()
    eps = 1e-3                                             # smaller steps are dominated by sub-float64 forward rounding
    with torch.no_grad():
        fd = (f(z + eps * u) - f(z - eps * u)) / (2 * eps)
    assert float(g.norm()) > 0
    assert abs(float(fd) - float(g @ u)) <= 1e-3 * max(abs(float(fd)), 1e-3)


# ---- episodes / reset -----------------------------------------------------------------------------------------------

def _ce_loss(y, mask):
    def fn(ep):
        _, loss = L.ce_terms(ep, y, mask, [], [], 2, prompt=CB)
        return {"loss": loss, "parts": {"CE": float(loss)}}
    return fn


def test_ttls_episode_reset_and_budget(tb):
    _, enc = _encode(tb)
    y, mask = [20, 21, 22, 23], [True] * 4
    logs = []
    for _ in range(2):
        ep = L.Episode(tb, None, enc, "TTLS", steps=L.ALL, layer=LAYER)
        assert torch.count_nonzero(ep.z) == 0 and ep.trainable == 16
        r = ep.run(_ce_loss(y, mask), {})
        logs.append(r["log"])
        assert len(r["log"]["losses"]) == 3 and len(r["log"]["grad_l2"]) == 2 and r["log"]["grad_l2"][0] > 0
        assert float(ep.z.norm()) <= L.E_STAR + 1e-6
    assert logs[0]["losses"] == logs[1]["losses"]          # fresh variable every episode: no carry-over
    ep = L.Episode(tb, None, enc, "TTLS", steps=L.ALL, layer=LAYER)
    with torch.no_grad():
        ep.z.fill_(10.0)
    assert ep.project() and math.isclose(float(ep.z.norm()), L.E_STAR, rel_tol=1e-6)
    assert all(p.grad is None and not p.requires_grad for p in tb.model.parameters())


def test_ln_episode_reuses_a2_actuator_and_resets(tb):
    _, enc = _encode(tb)
    guard = LNGuard(tb.model, decoder_ln_names(tb.model))
    ep = L.Episode(tb, guard, enc, "LN", layer=LAYER)
    r = ep.run(_ce_loss([20, 21, 22], [True] * 3), {})
    assert r["log"]["master_delta_l2"] > 0 and guard.verify()
    eff = ep.effective()
    guard.materialize(eff)
    guard.restore()
    assert guard.verify() and guard.current_hash() == guard.theta0_hash


# ---- frozen rules -----------------------------------------------------------------------------------------------------

def _lp(T, V=8):
    return {k: np.full((T + 1, V), -10.0) for k in ("cB_clean", "cB_null", "cE_clean", "cE_null")}


def test_candidate_rule():
    y_B = [5, 6, 7, 3]
    emb = {1, 2}                                           # "embedded-Latin" ids
    lp = _lp(4)
    # t=1: cE proposes Latin 1 over b=6 with prompt-robust support (E = 7 under both prompts)
    lp["cB_clean"][1, 1], lp["cB_null"][1, 1], lp["cB_clean"][1, 6], lp["cB_null"][1, 6] = -4.0, -12.0, -1.0, -2.0
    lp["cE_clean"][1, 1], lp["cE_null"][1, 1], lp["cE_clean"][1, 6], lp["cE_null"][1, 6] = -0.5, -8.5, -3.0, -4.0
    # t=2: cE argmax is the baseline token itself -> no proposal
    lp["cE_clean"][2, 7] = -0.1
    # t=3: cE tie between Latin 1 and 2 -> lowest ID 1; zero evidence -> proposal but rejected
    lp["cE_clean"][3, [1, 2]] = -0.1
    lp["cE_null"][3, 1], lp["cE_clean"][3, 3], lp["cE_null"][3, 3] = -1.1, -9.0, -10.0
    lp["cB_clean"][3, 1], lp["cB_null"][3, 1], lp["cB_clean"][3, 3], lp["cB_null"][3, 3] = -4.0, -5.0, -1.0, -2.0
    out = L.select_candidates(lp, y_B, emb, [True] * 4, eos=0)
    assert out["M"] == [1, 3] and out["accepted"] == [1] and out["t_star"] == 1 and out["c_star"] == 1
    r1 = next(r for r in out["records"] if r["t"] == 1)
    assert r1["evidence"]["cB"]["E"] == pytest.approx(7.0) and r1["evidence"]["cE"]["E"] == pytest.approx(7.0)
    r3 = next(r for r in out["records"] if r["t"] == 3)
    assert r3["e"] == 1 and not r3["accepted"]
    lp2 = copy.deepcopy(lp)
    lp2["cE_null"][1, 6] = -9.0                            # cE evidence collapses (A(b) = 6 > A(e) - tau) -> rejected
    assert L.select_candidates(lp2, y_B, emb, [True] * 4, eos=0)["t_star"] is None
    assert L.select_candidates(lp, y_B, emb, [True, False, True, True], eos=0)["M"] == [3]   # UTF-8 prefix guard
    assert L.select_candidates(lp, [5, 1, 7, 3], emb, [True] * 4, eos=0)["M"] == [3]          # Latin baseline never a target
    assert L.select_candidates(lp, y_B, emb, [True] * 4, eos=6)["M"] == [3]                   # EOS baseline never a target


def test_stable_positions():
    assert L.stable_positions([0.9, 0.4, 0.8, 0.95], [True, True, True, False], [2]) == [0]


# ---- end-to-end on the tiny model -------------------------------------------------------------------------------------

def test_process_row_tiny_end_to_end(tb, monkeypatch):
    import experiments.inference_cf_ttls_r1 as R
    import csasr.inference_cf.core_r2 as core_r2
    monkeypatch.setattr(core_r2, "prefix_utf8_complete", lambda *a, **k: True)
    guard = LNGuard(tb.model, decoder_ln_names(tb.model))
    enc_inf, enc_tr = _encode(tb)
    y_B = forced_decode(tb, enc_inf, CB, max_new_tokens=8, capture_layer=LAYER)["tokens"]
    assert len(y_B) >= 3
    y_A = list(y_B[:1]) + [int(y_B[1]) + 1 if int(y_B[1]) + 1 != 2 else 5] + list(y_B[2:])
    emb = set(range(3, 200)) - set(int(t) for t in y_B)
    legal = np.ones(200, dtype=bool)
    legal[:3] = False
    ctx = R.Ctx(tb, guard, suppress=[], begin=[], eos=2, embedded=emb, legal_mask=legal, byte_decoder={}, cB=CB, cE=CE, layer=LAYER,
                encode=lambda s: _encode(tb), encode_null=lambda: _encode(tb, seed=9), detect_lang=lambda s: CB[1],
                a2_effective=lambda s: {n: guard.theta0[n].clone() for n in guard.names},
                auto_prompt=lambda lang: [CB[0], lang, CB[2], CB[3]], tau=-1e9)
    s = {"utterance_id": "u", "dialogue_id": "d", "group": "D", "a3_group": None, "y_B": y_B, "y_B_terminated": "cap",
         "y_B_text": "", "y_B_valid_mask": [True] * len(y_B), "y_A": y_A, "y_A_text": "", "y_A_valid_mask": [True] * len(y_A),
         "utf8_ok": [True] * len(y_B), "A2": {"tokens": y_B, "terminated": "cap"}}
    monkeypatch.setattr(R, "MAX_NEW", 8)
    real = L.select_candidates

    def forced_candidate(*a, **k):                         # the degenerate tiny model proposes nothing: inject one
        out = real(*a, **k)                                # candidate to exercise the AC adaptation / decode plumbing
        if out["t_star"] is None:
            out.update(t_star=1, c_star=50, accepted=[1], M=sorted(set(out["M"]) | {1}))
        return out
    monkeypatch.setattr(L, "select_candidates", forced_candidate)
    row = R.process_row(ctx, s, 0)
    assert row["status"] == "ok" and row["B0"]["tokens_equal_S0"]
    assert row["B2"]["reproduces_historical_A2"]           # theta0 "masters" reproduce B0
    assert row["candidates"]["t_star"] is not None and row["candidates"]["t_star"] >= 1
    for name, var, obj, P in R.ARMS:
        a = row[name]
        assert a["status"] == "ok" and a["reset_ok"] and len(a["log"]["losses"]) == 3
        if var == "TTLS":
            assert a["log"]["trainable_scalars"] == 16 and math.sqrt(sum(x * x for x in a["z_effective"])) <= L.E_STAR * 1.01
        if P:
            assert "P" in a["log"]["parts"][0] and a["log"]["parts"][0]["P"] == pytest.approx(0.0, abs=1e-6)
    assert row["validated"]["clean_identity"]["empty_mask_decode_equals_B0"]
    assert row["validated"]["ttls_grad_flow_CE"]["pass"] and row["validated"]["ttls_grad_flow_AC"]["pass"]
    assert row["ACSUB"]["tokens"][:row["candidates"]["t_star"] + 1] == y_B[:row["candidates"]["t_star"]] + [row["candidates"]["c_star"]]
    assert guard.verify() and all(p.grad is None and not p.requires_grad for p in tb.model.parameters())
    json.dumps(row)                                        # JSON-serializable


# ---- firewall / config ------------------------------------------------------------------------------------------------

FORBIDDEN = ("csasr.evaluation", "inference_cf_p2_evaluate", "p0_r2", "load_references", "ttls_r1_evaluate")


@pytest.mark.parametrize("rel", ["src/csasr/inference_cf/ttls.py", "experiments/inference_cf_ttls_r1.py"])
def test_no_reference_or_evaluator_input(rel):
    tree = ast.parse((ROOT / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not any(f in m for m in mods for f in FORBIDDEN), (rel, mods)
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("load_references", "reference"), rel
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.lower()
            assert "reference" not in v or v == "references_used" or v.startswith("no reference") or "\n" in v, (rel, node.value)


def test_config_matches_module_and_pins():
    c = json.loads((ROOT / "configs/inference_cf/ttls_r1.json").read_text())
    assert c["prompts"]["cB"] == L.CB and c["prompts"]["cE"] == L.CE_PROMPT
    assert c["variables"]["TTLS"]["budget"]["E_STAR"] == L.E_STAR and c["variables"]["TTLS"]["optimizer"]["lr"] == L.ttls_lr(1280)
    assert c["candidate_rule"]["tau"] == L.TAU and c["objectives"]["LAMBDA_P"] == L.LAMBDA_P and c["stable_set"]["p_min"] == L.STABLE_P_MIN
    assert c["site"]["layer"] == L.LAYER and c["variables"]["steps"] == 2
    import experiments.inference_cf_ttls_r1 as R
    assert [a[0] for a in R.ARMS] == ["T1", "T2", "T2A", "T3", "T4", "T5", "T6"]
    assert set(c["arms"]) == {"B0", "B1", "B2"} | {a[0] for a in R.ARMS}
    for p, h in c["source_sha256"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p


def test_population_is_exposed_fixed100():
    from csasr.inference_cf.exposure_registry import exposed_ids
    t1 = json.loads((ROOT / "results/inference_cf/p2tta_funnel/tta1/plan_sealed.json").read_text())
    a3 = json.loads((ROOT / "docs/inference_cf/P2_TTA_A3_PANEL.json").read_text())
    ids = t1["ids"]
    assert len(ids) == 100 and len({r["dialogue_id"] for r in t1["rows"]}) == 20
    assert set(ids) <= set(exposed_ids(ROOT)["ids"])
    assert {r["utterance_id"] for r in a3["rows"]} <= set(ids)


# ---- frozen evaluator accounting (toy transcripts; no panel references) -------------------------------------------------

def test_evaluator_transitions_and_events():
    import experiments.inference_cf_ttls_r1_evaluate as E
    genuine = {"wrong_language_substitution", "phonetic_transliteration_or_script", "same_language_substitution", "other"}
    R = ["我们用zoom开会", "今天天气很好"]
    B = ["我们用祖母开会", "今天天气很好"]
    M = ["我们用zoom开会", "今天天气不好"]
    tr = E.poi_transitions(R, B, M, [False, False], ["d1", "d2"], ["u1", "u2"], genuine)
    assert tr["corrections"] == 1 and tr["genuine_substitution_corrections"] == 1 and tr["corruptions"] == 0
    assert tr["genuine_dialogue_count"] == 1
    zh = E.zh_changes(R, B, M)
    assert zh["new_zh_errors"] >= 1                         # 很 -> 不 is a newly introduced Mandarin error
    Tk = {"B0": [[1, 2], [5, 6, 7]], "X": [[1, 2, 3], [5]]}
    Te = {"B0": ["eos", "eos"], "X": ["eos", "eos"]}
    ev = E.row_events(Tk, Te, "X")
    assert ev["eos_recovery_rows"] == 1 and ev["premature_eos_rows"] == 1 and ev["changed_rows"] == 2


def test_evaluator_label_rule():
    import experiments.inference_cf_ttls_r1_evaluate as E
    c = json.loads((ROOT / "configs/inference_cf/ttls_r1.json").read_text())

    def arm(g, dl, net, newzh, safe=True):
        return {"transitions": {"genuine_substitution_corrections": g, "genuine_dialogue_count": dl, "net": net},
                "zh": {"new_zh_errors": newzh}, "safe": safe}
    base = {k: arm(0, 0, 0, 0) for k in ("B2", "T2A", "T3", "T5", "T1", "T2", "T4", "T6")}
    assert E.decide(False, base, c)["label"] == "TTLS_R1_INVALID"
    assert E.decide(True, base, c)["label"] == "TTLS_R1_NOT_SUPPORTED"
    assert E.decide(True, {**base, "T4": arm(3, 2, 1, 5)}, c)["label"] == "TTLS_R1_MIXED"
    assert E.decide(True, {**base, "T4": arm(8, 4, 6, 0), "T3": arm(8, 4, 6, 3)}, c)["label"] == "TTLS_R1_PROMISING"
    assert E.decide(True, {**base, "T4": arm(8, 4, 6, 4), "T3": arm(8, 4, 6, 3)}, c)["label"] == "TTLS_R1_MIXED"   # more ZH damage than A2
    assert E.decide(True, {**base, "T4": arm(8, 4, 6, 0, safe=False)}, c)["label"] == "TTLS_R1_MIXED"
