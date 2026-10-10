"""A2P-DEV200 focused tests (CPU; tiny real-class Whisper, no pretrained weights): the B3 arm is the TTLS-R1 T2A arm
bit-for-bit (oracle = TTLS-R1 process_row on the same inputs), initial P is exactly 0, A2-only episode reproduces an
archived A2 state, LN reset; frozen roster / oracle-selection rules; evaluator event grouping, concentration and the
frozen decision rule; reference firewall; config agreement."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]

from csasr.inference_cf import ttls as L  # noqa: E402
from csasr.inference_cf.episodic_tta import OPTIM, STEPS, LNGuard, decoder_ln_names, forced_decode  # noqa: E402
from test_ttls_r1 import CB, CE, LAYER, _bundle, _encode  # noqa: E402

import experiments.inference_cf_a2p_dev200 as A  # noqa: E402
import experiments.inference_cf_a2p_dev200_evaluate as AE  # noqa: E402


@pytest.fixture(scope="module")
def tb():
    return _bundle()


def test_b3_is_ttls_r1_t2a_and_a2_reproduces_archive(tb, monkeypatch, tmp_path):
    import experiments.inference_cf_ttls_r1 as R1
    import experiments.inference_cf_ttls_r1r as RR
    import csasr.inference_cf.core_r2 as core_r2
    monkeypatch.setattr(core_r2, "prefix_utf8_complete", lambda *a, **k: True)
    monkeypatch.setattr(R1, "MAX_NEW", 8)
    monkeypatch.setattr(A, "MAX_NEW", 8)
    guard = LNGuard(tb.model, decoder_ln_names(tb.model))
    enc_inf, enc_tr = _encode(tb)
    d0 = forced_decode(tb, enc_inf, CB, max_new_tokens=8, capture_layer=LAYER)
    y_B = d0["tokens"]
    y_A = list(y_B[:1]) + [int(y_B[1]) + 1 if int(y_B[1]) + 1 != 2 else 5] + list(y_B[2:])
    emb = set(range(3, 200)) - set(int(t) for t in y_B)
    legal = np.ones(200, dtype=bool)
    legal[:3] = False
    # historical A2 archive for this tiny utterance (adapt() = the TTA1 A2 implementation)
    from csasr.inference_cf.episodic_tta import adapt
    rec = adapt(tb.model, guard, enc_tr, CB, y_A, [True] * len(y_A), "A2", suppress=[], begin=[], eos=2)
    npz = tmp_path / "a2.npz"
    np.savez(npz, **{f"A2_p{j:03d}": rec["final_masters"][n].numpy() for j, n in enumerate(guard.names)})
    guard.materialize(rec["effective"])
    try:
        dA2 = forced_decode(tb, enc_inf, CB, max_new_tokens=8, capture_layer=LAYER)
    finally:
        guard.restore()
    s = {"utterance_id": "u", "dialogue_id": "d", "group": "D", "a3_group": None, "y_B": y_B, "y_B_terminated": d0["terminated"],
         "y_B_text": d0["text"], "y_B_valid_mask": [True] * len(y_B), "y_A": y_A, "y_A_text": "", "y_A_valid_mask": [True] * len(y_A),
         "utf8_ok": [True] * len(y_B), "A2": {"tokens": dA2["tokens"], "terminated": dA2["terminated"], "masters_npz": str(npz)}}
    ctx1 = R1.Ctx(tb, guard, suppress=[], begin=[], eos=2, embedded=emb, legal_mask=legal, byte_decoder={}, cB=CB, cE=CE, layer=LAYER,
                  encode=lambda s: _encode(tb), encode_null=lambda: _encode(tb, seed=9), detect_lang=lambda s: CB[1],
                  a2_effective=lambda s: {n: guard.theta0[n].clone() for n in guard.names}, auto_prompt=lambda lang: [CB[0], lang, CB[2], CB[3]])
    r1 = R1.process_row(ctx1, dict(s), 0)
    oracle = {"candidates": {k: r1["candidates"][k] for k in A.CAND_KEYS}, "T2A": {k: r1["T2A"][k] for k in ("tokens", "terminated", "text")},
              "T2A_log": {k: r1["T2A"]["log"][k] for k in ("losses", "parts", "grad_l2", "master_delta_l2", "effective_delta_l2")}}
    ctx = RR.Ctx(tb, guard, suppress=[], begin=[], eos=2, embedded=emb, cB=CB, cE=CE, layer=LAYER,
                 encode=lambda s: _encode(tb), encode_null=lambda: _encode(tb, seed=9))
    ctx.counters = {}
    row = A.process_row(ctx, {**s, "oracle": oracle}, 0, regression=True)
    assert row["status"] == "ok" and all(row["gates"].values()), row["gates"]
    assert row["B3"]["log"]["parts"][0]["P"] == 0.0 and row["B3"]["log"]["trainable_scalars"] == sum(p.numel() for p in guard.theta0.values())
    assert row["A2_only"]["step0_CE_equal_B3"]
    assert guard.verify() and guard.current_hash() == guard.theta0_hash
    assert all(p.grad is None and not p.requires_grad for p in tb.model.parameters())
    # NEW200 path (step-0 CE identity against a recorded A2 loss; AUTO replay)
    s2 = {**s, "AUTO": {"tokens": None, "text": None, "terminated": None}, "A2": {**s["A2"], "loss0": rec["log"]["losses"][0], "grad0": rec["log"]["grad_l2"][0]}}
    ctx.detect_lang, ctx.auto_prompt = (lambda s: CB[1]), (lambda lang: [CB[0], lang, CB[2], CB[3]])
    auto = forced_decode(tb, enc_inf, [CB[0], CB[1], CB[2], CB[3]], max_new_tokens=8, capture_layer=LAYER)
    s2["AUTO"] = {"tokens": auto["tokens"], "text": auto["text"], "terminated": auto["terminated"]}
    row2 = A.process_row(ctx, s2, 0)
    assert all(row2["gates"].values()) and row2["B3"]["tokens"] == row["B3"]["tokens"]
    json.dumps(row)
    json.dumps(row2)


def test_regression_oracle_rule():
    p = [{"y_B": [1]}, {"y_B": [2]}, {"y_B": [3]}, {"y_B": [4]}, {"y_B": [5]}]
    r = [{"B2": {"tokens": [1]}, "T2A": {"tokens": [1]}}, {"B2": {"tokens": [9]}, "T2A": {"tokens": [2]}},
         {"B2": {"tokens": [3]}, "T2A": {"tokens": [3]}}, {"B2": {"tokens": [4]}, "T2A": {"tokens": [8]}}, {"B2": {"tokens": [5]}, "T2A": {"tokens": [5]}}]
    assert A.regression_indices(r, p) == [0, 1, 2, 3]


def test_roster_membership_hashes_match_path5():
    c = json.loads((ROOT / A.CONFIG).read_text())["population"]
    panel = json.loads((ROOT / c["panel"]).read_text())
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in panel["rows"]}
    for k, h in c["membership_sha256"].items():
        assert A.membership_hash(k, panel["partitions"][k]["ids"], dlg) == h
    new = panel["partitions"]["NEW200"]["ids"]
    assert len(new) == 200 and not set(new) & set(panel["partitions"]["FIXED100"]["ids"])


def test_config_matches_inherited_constants():
    c = json.loads((ROOT / A.CONFIG).read_text())["B3_definition"]
    o = c["optimizer"]
    assert (o["lr"], o["weight_decay"], tuple(o["betas"]), o["eps"], o["updates"]) == (OPTIM["lr"], OPTIM["weight_decay"], OPTIM["betas"], OPTIM["eps"], STEPS)
    assert L.LAMBDA_P == 1.0 and L.STABLE_P_MIN == 0.5 and L.TAU == math.log(10.0)


def test_event_grouping_and_shares():
    ev = [{"j": 0, "utterance_id": "u", "dialogue_id": "d", "poi_index": i, "type": "correction", "kind": "genuine_wrong_language", "poi": w, "B0": "x", "system": w}
          for i, w in ((3, "go"), (4, "back"), (5, "to"), (6, "school"))]
    ev += [{"j": 0, "utterance_id": "u", "dialogue_id": "d", "poi_index": 9, "type": "correction", "kind": "deletion_or_boundary", "poi": "ok", "B0": "", "system": "ok"}]
    g = AE.group_events(ev)
    assert len(g) == 2 and g[0]["units"] == 4 and g[0]["wrong_language"] and not g[1]["genuine"]
    s = AE.shares({"a": 20, "b": 2, "c": 1, "d": 0, "e": -1})
    assert s["n_positive"] == 3 and s["n_negative"] == 1 and math.isclose(s["largest_share"], 20 / 23) and s["top3_share"] == 1.0


def _fake(N2, N3, pos_u, pos_d, neg_u, largest, lodo, excl, ret, mer2, mer3, mer1, zh3, zh1):
    ev = {"new_severe_truncations": 0, "added_caps": 0}
    return {"metrics": {"B1": {"mer": mer1, "zh_cer": zh1, "pier": .4}, "B2": {"mer": mer2, "pier": .45, "zh_cer": .2}, "B3": {"mer": mer3, "pier": .45, "zh_cer": zh3}},
            "newzh": {"N2": N2, "N3": N3, "D": N2 - N3, "D_non_termination_rows": excl},
            "retention_B2_by_B3": {"B2_genuine": 10, "genuine_rate": ret, "B2_corrections": 20, "rate": ret},
            "breadth": {"zh_avoided_utt": {"n_positive": pos_u, "n_negative": neg_u, "largest_share": largest}, "zh_avoided_dlg": {"n_positive": pos_d}},
            "lodo": {"D_positive_panels": lodo}, "vs_B0": {"B3": {"events": ev}, "B2": {"events": ev}}}


def test_decision_rule():
    ok = _fake(40, 10, 8, 6, 1, .3, 20, 12, .9, .25, .248, .26, .2, .23)
    assert AE.decide(True, ok, {})["label"] == "A2P_DEV200_BROAD_SUPPORT"
    assert AE.decide(True, _fake(40, 10, 8, 6, 1, .7, 20, 12, .9, .25, .248, .26, .2, .23), {})["label"] == "A2P_DEV200_CONCENTRATED_OR_MIXED"
    assert AE.decide(True, _fake(5, 2, 3, 3, 0, .4, 20, 3, .9, .25, .248, .26, .2, .23), {})["label"] == "A2P_DEV200_CONCENTRATED_OR_MIXED"
    assert AE.decide(True, _fake(10, 10, 3, 3, 3, .4, 10, 0, .9, .25, .248, .26, .2, .23), {})["label"] == "A2P_DEV200_NOT_SUPPORTED"
    assert AE.decide(True, _fake(40, 10, 8, 6, 1, .3, 20, 12, .4, .25, .248, .26, .2, .23), {})["label"] == "A2P_DEV200_NOT_SUPPORTED"
    assert AE.decide(True, _fake(40, 10, 8, 6, 1, .3, 20, 12, .9, .25, .2561, .26, .2, .23), {})["label"] == "A2P_DEV200_NOT_SUPPORTED"
    assert AE.decide(True, _fake(40, 10, 8, 6, 1, .3, 20, 12, .9, .25, .248, .24, .2, .23), {})["label"] == "A2P_DEV200_CONCENTRATED_OR_MIXED"
    assert AE.decide(False, ok, {})["label"] == "A2P_DEV200_INVALID"


FORBIDDEN = ("csasr.evaluation", "inference_cf_p2_evaluate", "p0_r2", "load_references", "evaluate")


def test_runner_has_no_reference_or_evaluator_input():
    tree = ast.parse((ROOT / "experiments/inference_cf_a2p_dev200.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            assert not any(f in m for m in mods for f in FORBIDDEN), mods
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("load_references", "reference")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.lower()
            assert "reference" not in v or v == "references_used" or v.startswith("no reference") or "\n" in v, node.value
