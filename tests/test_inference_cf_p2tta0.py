"""P2-TTA0 focused tests (contract eb0da2a): panel identity; exact decoder-LN trainables (real config, meta device);
fp32-master-only gradients through bf16 functional_call; theta0 functional logits bitwise = ordinary forward; exact
reset (incl. on error) and fresh optimizer per objective; exactly 2 AdamW steps / 3 loss forwards; A1 entropy and A2
NLL formulas (suppression at t0 vs later, EOS in vocabulary, final query excluded); empty/interior-special teachers;
live independent auditor agreement; forced decode = historical cached path without hooks; deterministic labels,
confirmation bias and selection (incl. exact tie -> A2); runner reference/steering-free; auditor independence."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, VOCAB, Tok  # noqa: E402
from test_inference_cf_p2dir_protocol import _bf16_bundle, _enc  # noqa: E402

import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2tta0 as run  # noqa: E402
import experiments.inference_cf_p2tta0_analyze as an  # noqa: E402
import experiments.inference_cf_p2tta0_audit as au  # noqa: E402

CFG = json.loads(Path(run.CONFIG).read_text())
EOS_T = VOCAB - 1                         # tiny tokenizer: content < EOS_T, special >= EOS_T


class TokT(Tok):
    eos_token_id = EOS_T
    all_special_ids = [EOS_T]


def _bundle(seed=0, suppress=(), begin=()):
    b = _bf16_bundle(seed)
    b.processor = SimpleNamespace(tokenizer=TokT())
    b.model.generation_config.suppress_tokens = list(suppress)
    b.model.generation_config.begin_suppress_tokens = list(begin)
    b.model.requires_grad_(False)
    return b


def _setup(seed=0, suppress=(), begin=()):
    b = _bundle(seed, suppress, begin)
    names = tta.decoder_ln_names(b.model)
    return b, names, tta.LNGuard(b.model, names)


def _adapt(b, guard, y, kind, mask=None, counters=None, keep=False):
    g = b.model.generation_config
    mask = mask if mask is not None else tta.valid_mask(y, g.suppress_tokens, g.begin_suppress_tokens, {EOS_T}, EOS_T)
    return tta.adapt(b.model, guard, _enc(), CB, y, mask, kind, suppress=g.suppress_tokens, begin=g.begin_suppress_tokens,
                     eos=EOS_T, partition=PART, counters=counters, keep_grad0=keep)


# ---- panel / trainables ---------------------------------------------------------------------------------------

def test_panel_exact_first_per_dialogue():
    p = json.loads(Path(run.PANEL).read_text())
    assert hashlib.sha256(Path(run.PANEL).read_bytes()).hexdigest() == CFG["panel"]["byte_sha256"]
    assert hashlib.sha256(Path(run.PARENT).read_bytes()).hexdigest() == p["parent_byte_sha256"] == CFG["source_sha256"][run.PARENT]
    parent = json.loads(Path(run.PARENT).read_text())
    seen, first = set(), []
    for r in parent["rows"]:
        if r["dialogue_id"] not in seen:
            seen.add(r["dialogue_id"])
            first.append(r["utterance_id"])
    ids = [r["utterance_id"] for r in p["rows"]]
    assert ids == first == CFG["panel"]["ids"] and len(ids) == 20 and len({r["dialogue_id"] for r in p["rows"]}) == 20


def test_trainables_exact_on_real_config_meta():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    with torch.device("meta"):
        m = WhisperForConditionalGeneration(WhisperConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True))
    names = tta.decoder_ln_names(m)
    params = dict(m.named_parameters())
    assert names == [p["name"] for p in CFG["trainables"]["parameters"]] and len(names) == 194
    assert [list(params[n].shape) for n in names] == [p["shape"] for p in CFG["trainables"]["parameters"]]
    assert sum(params[n].numel() for n in names) == 248320 and sum(p.numel() for p in m.parameters()) == 1543490560
    assert not any(n.startswith("model.encoder") for n in names)


# ---- gradient flow / bitwise theta0 / reset ----------------------------------------------------------------------

def test_theta0_functional_logits_bitwise_and_master_only_grads():
    b, names, guard = _setup()
    y = [10, 11, 12, 13]
    ordinary = tta.teacher_logits(b.model, _enc(), CB, y, None)
    masters = guard.fresh_masters()
    with torch.enable_grad():
        func = tta.teacher_logits(b.model, _enc(), CB, y, {n: m.to(torch.bfloat16) for n, m in masters.items()})
        assert torch.equal(func.detach(), ordinary)
        func.sum().backward()
    assert all(m.grad is not None and m.grad.dtype == torch.float32 for m in masters.values())
    assert any(float(m.grad.abs().sum()) > 0 for m in masters.values())
    assert all(p.grad is None and not p.requires_grad for p in b.model.parameters())


def test_adapt_two_steps_three_losses_reset_and_fresh_optimizer():
    b, names, guard = _setup()
    y = [10, 11, 12, 13, 14]
    c = {}
    r1 = _adapt(b, guard, y, "A1", counters=c)
    assert r1["log"]["steps"] == 2 and len(r1["log"]["losses"]) == 3 and len(r1["log"]["grad_l2"]) == 2
    assert c == {"teacher_forwards": 3, "backwards": 2, "optimizer_steps": 2}
    assert r1["log"]["master_delta_l2"] > 0 and guard.verify()           # resident params untouched by adaptation
    r2 = _adapt(b, guard, y, "A1")                                        # fresh masters/optimizer -> identical result
    assert r2["log"]["losses"] == r1["log"]["losses"] and all(torch.equal(r1["final_masters"][n], r2["final_masters"][n]) for n in names)
    guard.materialize(r1["effective"])
    assert not guard.verify()
    guard.restore()
    assert guard.verify() and guard.current_hash() == guard.theta0_hash


def test_reset_on_error_and_other_params_immutable(monkeypatch):
    b, names, guard = _setup()
    v0 = guard.other_versions()

    def boom(*a, **k):
        raise RuntimeError("decode failure")
    monkeypatch.setattr(tta, "forced_decode", boom)
    with pytest.raises(RuntimeError):
        run.run_objective(b, guard, "A1", _enc(), _enc(), [10, 11, 12], [True] * 3, [10, 11, 12], [True] * 3,
                          suppress=[], begin=[], eos=EOS_T, partition=PART, counters={}, cfg_prompt=CB, max_new_tokens=6)
    assert guard.verify() and guard.other_versions() == v0


def test_run_objective_final_forced_decode_and_zero_update_identity():
    b, names, guard = _setup()
    base = tta.forced_decode(b, _enc(), CB, max_new_tokens=6)
    r = run.run_objective(b, guard, "A2", _enc(), _enc(), [], [], base["tokens"], [True] * len(base["tokens"]),
                          suppress=[], begin=[], eos=EOS_T, partition=PART, counters={}, cfg_prompt=CB, max_new_tokens=6)
    lg = r["log"]
    assert lg["no_valid_content"] and lg["losses"] == [0.0, 0.0, 0.0] and lg["grad_l2"] == [0.0, 0.0] and lg["steps"] == 2
    assert lg["master_delta_l2"] == 0 and lg["effective_changed_scalars"] == 0
    assert r["decode"]["tokens"] == base["tokens"] and lg["reset_ok"] and lg["start_hash"] == lg["end_hash"] == guard.theta0_hash
    assert "theta2" in r["common_y_B"]


def test_forced_decode_equals_historical_cached_path(monkeypatch):
    b = _bundle()
    d = tta.forced_decode(b, _enc(), CB, max_new_tokens=8)
    monkeypatch.setattr(cached, "native_lid", lambda bb, w, ids: {4: .9, 5: .1})
    with torch.inference_mode():
        h = cached.cached_decode(b, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=_enc(),
                                 conditions={"cB": CB, "cE": CE, "language_token_ids": [4, 5]}, partition=PART,
                                 null_probs={4: .5, 5: .5}, language_ids=(4, 5), layer=16, alpha=0.0, max_new_tokens=8,
                                 num_forced_prefix=4)
    assert d["tokens"] == h["tokens"] and d["terminated"] == h["terminated"]


# ---- objective math ------------------------------------------------------------------------------------------------

def test_position_terms_entropy_nll_suppression_and_eos():
    torch.manual_seed(0)
    V, y = 12, [3, 5]
    L = torch.randn(3, V)
    sup, beg, eos = [7, 8], [5, 9], 11
    t = tta.position_terms(L, y, sup, beg, eos)
    for i in range(2):
        allowed = [v for v in range(V) if v not in sup and not (i == 0 and v in beg)]
        lp = torch.log_softmax(L[i, allowed].double(), -1)
        assert float(t["entropy"][i]) == pytest.approx(float(-(lp.exp() * lp).sum()), abs=1e-6)
        assert eos in allowed and t["eos_prob"][i] == pytest.approx(float(lp.exp()[allowed.index(eos)]), abs=1e-6)
    lp1 = torch.log_softmax(L[1, [v for v in range(V) if v not in sup]].double(), -1)
    assert float(t["target_logprob"][1]) == pytest.approx(float(lp1[[v for v in range(V) if v not in sup].index(5)]), abs=1e-6)
    assert not math.isnan(float(t["target_logprob"][0]))                   # y[0]=3 allowed at t0
    t2 = tta.position_terms(L, [5, 3], sup, beg, eos)                      # 5 is begin-suppressed at t0 only
    assert math.isnan(float(t2["target_logprob"][0])) and len(t["entropy"]) == 2 and len(t["eos_prob"]) == 3
    assert tta.valid_mask([5, 3], sup, beg, {eos}, eos) == [False, True]
    assert tta.valid_mask([3, 7, 11, 4], sup, beg, {eos}, eos) == [True, False, False, True]


def test_objective_losses_on_tiny_model_and_live_auditor_agree():
    b, names, guard = _setup(suppress=[20, 21], begin=[10])
    y = [10, 12, 20, 13, 14]                                                # t0 begin-suppressed, t2 suppressed
    g = b.model.generation_config
    mask = tta.valid_mask(y, g.suppress_tokens, g.begin_suppress_tokens, {EOS_T}, EOS_T)
    assert mask == [False, True, False, True, True]
    for kind in ("A1", "A2"):
        r = _adapt(b, guard, y, kind, mask=mask, keep=True)
        la = au.live_objective_check(b.model, names, _enc(), CB, y, kind, suppress=g.suppress_tokens, begin=g.begin_suppress_tokens,
                                     tokenizer=TokT())
        assert la["valid"] == 3 and abs(la["auditor_loss"] - r["log"]["losses"][0]) <= 1e-5
        gd = torch.linalg.vector_norm(r["grad0"].double() - la["_grad"].double())
        assert float(gd) <= max(1e-8, 1e-3 * float(torch.linalg.vector_norm(la["_grad"].double())))
        pos = r["log"]["positions"][0]
        man = np.mean([pos["entropy"][i] for i in (1, 3, 4)]) if kind == "A1" else -np.mean([pos["target_logprob"][i] for i in (1, 3, 4)])
        assert abs(man - r["log"]["losses"][0]) < 1e-5
        assert guard.verify()


def test_teacher_cleaning():
    sp = {EOS_T}
    assert tta.clean_teacher([10, 11, EOS_T], CB, EOS_T, sp) == [10, 11]
    real = [50258, 50260, 50360, 50364]
    assert tta.clean_teacher(real + [100, 200, 50257], real, 50257, {50257, 50258}) == [100, 200]   # verified prompt + EOS stripped
    with pytest.raises(tta.TTAInvalid):
        tta.clean_teacher([10, EOS_T, 11], CB, EOS_T, sp)
    long = list(range(3, 13)) * 20
    assert tta.clean_teacher(long, CB, EOS_T, sp) == long                 # cap content preserved, no EOS invented


def test_severe_truncation_and_levenshtein():
    assert tta.severe_truncation(10, 5, "eos") and not tta.severe_truncation(10, 6, "eos")
    assert not tta.severe_truncation(9, 0, "eos") and not tta.severe_truncation(10, 5, "cap")
    assert tta.levenshtein([1, 2, 3], [1, 3]) == 1 == au.own_lev([1, 2, 3], [1, 3])
    assert tta.compare([1, 2], [1, 2, 3]) == {"equal": False, "first_difference": 2, "levenshtein": 1}


# ---- labels / selection ----------------------------------------------------------------------------------------------

BASEPT = {"mer_increase": 0.0, "zh_cer_increase": 0.0, "zh_retention": 0.99, "en_retention": 0.96, "outside_harm_rate": 0.02,
          "poi_corruption_rate": 0.02, "added_caps": 0, "new_severe_truncations": 0, "net_poi_error_reduction": 2,
          "improved_utterances": 0, "degraded_utterances": 0}


def test_labels_confirmation_bias_and_selection():
    S = lambda **k: an.safety_predicates({**BASEPT, **k})
    U = lambda **k: an.useful_signal({**BASEPT, **k})
    assert all(S().values()) and U()["useful"]
    assert not S(mer_increase=0.0101)["MER"] and S(mer_increase=0.01)["MER"] and not S(zh_cer_increase=0.0151)["ZH_CER"]
    assert not S(zh_retention=0.9799)["matrix_ZH_retention"] and S(zh_retention=None)["matrix_ZH_retention"]
    assert not S(outside_harm_rate=0.0301)["outside_harm"] and not S(added_caps=2)["caps"] and not S(new_severe_truncations=1)["severe_truncation"]
    assert not U(net_poi_error_reduction=1)["useful"]
    assert U(net_poi_error_reduction=0, improved_utterances=3, degraded_utterances=2)["useful"]
    assert not U(net_poi_error_reduction=0, improved_utterances=3, degraded_utterances=3)["useful"]
    assert not U(net_poi_error_reduction=0, improved_utterances=2, degraded_utterances=0)["useful"]
    bad = S(mer_increase=0.02)
    cb = an.confirmation_bias(1.0, 0.99, bad)
    assert cb["flag"] and an.label_objective("A1", True, bad, U(), cb) == "TTA0_EM_CONFIRMATION_BIAS"
    assert not an.confirmation_bias(1.0, 0.991, bad)["flag"] and not an.confirmation_bias(0.0, 0.0, bad)["flag"]
    assert not an.confirmation_bias(1.0, 0.5, S())["flag"]
    assert an.label_objective("A1", True, S(), U(), an.confirmation_bias(1.0, 0.5, S())) == "TTA0_EM_VIABLE"
    assert an.label_objective("A2", True, bad, U(), None) == "TTA0_AC_NOT_VIABLE"
    assert an.label_objective("A2", False, S(), U(), None) == "P2_TTA0_INVALID"
    M = {"A1": {"mer": 0.2, "pier": 0.3, "zh_cer": 0.1}, "A2": {"mer": 0.2, "pier": 0.3, "zh_cer": 0.1}}
    both = {"A1": "TTA0_EM_VIABLE", "A2": "TTA0_AC_VIABLE"}
    assert an.select(both, M, {"A1": 1.0, "A2": 1.0}, True)["selected"] == "A2"            # exact tie -> A2
    assert an.select(both, M, {"A1": 0.5, "A2": 1.0}, True)["selected"] == "A1"            # update-norm key
    assert an.select(both, {**M, "A1": {"mer": 0.19, "pier": 0.4, "zh_cer": 0.2}}, {"A1": 1, "A2": 1}, True)["selected"] == "A1"
    assert an.select({"A1": "TTA0_EM_CONFIRMATION_BIAS", "A2": "TTA0_AC_NOT_VIABLE"}, M, {"A1": 1, "A2": 1}, True)["stage_label"] == "P2_TTA0_NO_VIABLE_OBJECTIVE"
    assert an.select(both, M, {"A1": 1, "A2": 1}, False)["stage_label"] == "P2_TTA0_INVALID"


def test_outside_harm_vs_harmless_edit():
    from experiments.inference_cf_p2seq_analyze import outside
    ref = "我们 去 school 吃饭"
    assert outside(ref, ref, "我们 去 school 吃面")["outside_harm"] >= 1
    assert outside(ref, "我门 去 school 吃饭", ref)["outside_harm"] == 0


def test_runner_reference_steering_free_and_auditor_independent():
    assert au.runner_clean()["ok"]
    src = Path("experiments/inference_cf_p2tta0_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta)", src, re.M) is None
    for rel, h in CFG["source_sha256"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == h, rel
    assert tta.OPTIM == {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False, "foreach": False,
                         "fused": False, "maximize": False} and tta.STEPS == 2


def test_invalid_record_and_invalid_audit_are_reference_free():
    """INVALID supersedes all labels: the terminal record/audit must not load references or compute metrics."""
    import ast
    import inspect
    for fn in (an.invalid_record, au.cmd_invalid):
        src = inspect.getsource(fn)
        calls = {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "") for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call)}
        assert not calls & {"load_references", "corpus_metrics", "correction_corruption", "counts", "transitions", "outside"}
