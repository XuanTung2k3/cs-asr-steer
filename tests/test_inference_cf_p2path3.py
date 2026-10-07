"""P2-PATH3 focused tests (freeze 5853a8d): exact fixed100/partition/order hashes and nesting (refill/duplicate rejected);
parent A4/G1 helper hashes unchanged; replay12 exact projection (1-ulp logprob change and changed tokens detected, owner
labels ignored); barrier before remaining 88; unchanged G1 controller on a toy two-instance fixture (no-trigger G1 = live
prefix, null decision; trigger -> winner first token only, one event, counters) and no G2 in the runner; safety
None/cap-delta semantics; rescue/benefit/breadth/LODO boundaries; exhaustive label precedence; deterministic shared
bootstrap draws; reference barrier and auditor independence."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB as CB_T  # noqa: E402
from test_inference_cf_p2dir_protocol import _enc  # noqa: E402
from test_inference_cf_p2path2 import _pair  # noqa: E402
from test_inference_cf_p2tta0 import EOS_T  # noqa: E402

import experiments.inference_cf_p2path3 as run  # noqa: E402
import experiments.inference_cf_p2path3_analyze as an  # noqa: E402
import experiments.inference_cf_p2path3_audit as au  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / run.CONFIG).read_text())
P = json.loads((ROOT / run.PANEL).read_text())


def _lists():
    par = json.loads((ROOT / run.PARENT).read_text())
    full = [r["utterance_id"] for r in par["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in par["rows"]}
    dev = [r["utterance_id"] for r in json.loads((ROOT / run.A3_PANEL).read_text())["rows"]]
    p12 = [r["utterance_id"] for r in json.loads((ROOT / run.P2_PANEL).read_text())["rows"]]
    return full, dev, p12, dlg


def test_fixed100_partitions_order_and_hashes():
    full, dev, p12, dlg = _lists()
    assert run.sha_file(ROOT / run.PARENT) == C["fixed100_parent_byte_sha256"] and run.sha_file(ROOT / run.PANEL) == C["panel_byte_sha256"]
    parts = run.partitions(full, dev, p12)
    assert {k: len(v) for k, v in parts.items()} == C["counts"]
    assert all(run.membership_hash(k, v, dlg) == C["partition_sha256"][k] for k, v in parts.items())
    assert set(parts["NOVEL76"]) == set(full) - set(dev) and parts["NOVEL76"] == [u for u in full if u in set(parts["NOVEL76"])]
    order = run.execution_order(full, p12)
    assert order == P["execution_order"] and order[:12] == p12 == C["PATH2_barrier"]["first_ids"] and sorted(order) == sorted(full)
    assert run.row_fp(order) == P["execution_order_sha256"]
    assert [r["canonical_index"] for r in P["rows"]] == list(range(100))
    with pytest.raises(ValueError):
        run.partitions(full, dev + [dev[0]], p12)                         # duplicate
    with pytest.raises(ValueError):
        run.partitions(full, dev, p12 + ["ZH-CN_NOT_IN_DEV24"])            # refill / not nested
    swapped = list(parts["NOVEL76"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert run.membership_hash("NOVEL76", swapped, dlg) != C["partition_sha256"]["NOVEL76"]
    assert run.membership_hash("NOVEL", parts["NOVEL76"], dlg) != C["partition_sha256"]["NOVEL76"]      # renamed role


def test_parent_helpers_and_a4_inheritance_pinned():
    for p in ("src/csasr/inference_cf/consensus_guard.py", "src/csasr/inference_cf/branch_adjudication.py", "src/csasr/inference_cf/script_safe_tta.py",
              "src/csasr/inference_cf/episodic_tta.py", "src/csasr/inference_cf/path_decode.py", "experiments/inference_cf_cached.py"):
        assert run.sha_file(ROOT / p) == C["source_sha256"][p]
    T = C["A4_inherited"]
    assert T["trainables"]["tensor_count"] == len(T["trainables"]["parameters"]) == 194
    assert sum(p["numel"] for p in T["trainables"]["parameters"]) == T["trainables"]["scalar_count"] == 248320
    assert T["optimization"]["steps"] == 2 and T["optimization"]["lr"] == 1e-3 and T["optimization"]["weight_decay"] == 0
    from csasr.inference_cf.episodic_tta import OPTIM, STEPS
    assert STEPS == 2 and OPTIM["lr"] == 1e-3 and OPTIM["weight_decay"] == 0 and tuple(OPTIM["betas"]) == (0.9, 0.999)
    assert run.RUNTIME_KEYS == ("utterance_id", "audio_path", "audio_fingerprint_64k_sizeprefixed", "audio_full_sha256", "y_A", "y_A_valid_mask", "classes")


def _p2row(i):
    return json.loads((ROOT / f"results/inference_cf/p2path2/run1/rows/{i:02d}.json").read_text())


def test_replay_projection_exact_and_injected_mismatch():
    for i in range(12):
        r = _p2row(i)
        assert run.replay_mismatches(json.loads(json.dumps(r)), r) == []
    r = _p2row(0)                                                         # triggered row
    live = json.loads(json.dumps(r))
    for v in live["score_paths"].values():
        v["owner"]["condition"] = "renamed"                             # owner labels ignored
    live["G2"] = None
    assert run.replay_mismatches(live, r) == []
    lp = live["decision"]["scores"]["A4_b0"]["logprobs"]
    lp[0] = float(np.nextafter(lp[0], 0.0))                               # 1 ulp
    assert "decision" in run.replay_mismatches(live, r)
    live2 = json.loads(json.dumps(r))
    live2["G1"]["tokens"] = live2["G1"]["tokens"][:-1]
    assert run.replay_mismatches(live2, r) == ["G1"]
    nt = _p2row(4)
    assert nt["decision"] is None
    live3 = json.loads(json.dumps(nt))
    live3["G0"]["terminated"] = "cap"
    assert "G0" in run.replay_mismatches(live3, nt)


def test_barrier_precedes_remaining88_and_no_g2():
    src = (ROOT / "experiments/inference_cf_p2path3.py").read_text()
    code = src[src.index("def cmd_run"):src.index("def cmd_seal")]
    i_bar = code.index('raise TTAInvalid("PATH2 replay barrier failed (remaining 88 not entered)')
    assert code.index("for i, uid in enumerate(order[:12])") < code.index("replay_mismatches(row, p2row)") < i_bar < code.index("enumerate(order[12:], start=12)")
    assert code.index('raise TTAInvalid("replay phase-1 barrier failed') < code.index("online_g1(")
    assert "g2_forced" not in src and "G2_replay" not in src
    ctrl = src[src.index("def online_g1"):src.index("def _norm")]
    assert not any(t in ctrl for t in au.AUDIT_TOKENS)


def _toy(monkeypatch, perturb):
    monkeypatch.setattr(run, "CB", CB_T)
    monkeypatch.setattr(run, "MAX_NEW", 8)
    return _pair(perturb)


def _counters():
    return {"detector_runs": 0, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0}


def test_online_g1_no_trigger_identity(monkeypatch):
    b0, b4, g0, g4 = _toy(monkeypatch, 0.0)
    cnt = _counters()
    rec = run.online_g1(b0, b4, g0, g4, _enc(), "toy", cnt, EOS_T)
    assert rec["fails"] == [] and rec["decision"] is None and rec["G1"] == {"tokens": rec["detector"]["prefix"], "terminated": rec["detector"]["terminated"]}
    assert cnt == {"detector_runs": 1, "triggers": 0, "rollouts": 0, "score_paths": 0, "G1_decodes": 0}


def test_online_g1_trigger_one_token_commit(monkeypatch):
    for pert in (0.3, 0.6, 1.0, 2.0):
        b0, b4, g0, g4 = _toy(monkeypatch, pert)
        cnt = _counters()
        try:
            rec = run.online_g1(b0, b4, g0, g4, _enc(), "toy", cnt, EOS_T)
        except Exception as e:                                              # EOS-first branch is the frozen INVALID path
            assert "EOS-first" in str(e)
            continue
        if rec["decision"] is None:
            continue
        d = rec["decision"]
        w = d["b0"] if d["winner"] == "theta0" else d["b4"]
        assert rec["G1"]["forced"] == [w["tokens"][0]] and rec["G1"]["tokens"][:d["k"]] == d["prefix"] and rec["G1"]["tokens"][d["k"]] == w["tokens"][0]
        assert cnt == {"detector_runs": 1, "triggers": 1, "rollouts": 2, "score_paths": 4, "G1_decodes": 1}
        S = {k: sum(v["logprobs"]) / len(v["logprobs"]) for k, v in d["scores"].items()}
        m = 0.5 * S["theta0_b0"] + 0.5 * S["A4_b0"] - (0.5 * S["theta0_b4"] + 0.5 * S["A4_b4"])
        assert d["margin"] == m and d["winner"] == ("theta0" if m >= -1e-12 else "A4")
        assert [f["token"] for f in rec["G1"]["trace"]["forced"]] == rec["G1"]["forced"] and rec["fails"] == []
        return
    pytest.skip("no usable disagreement on tiny model")


def _pt(**kw):
    p = {"mer_increase": 0.0, "zh_cer_increase": 0.0, "zh_retention": 1.0, "en_retention": 1.0, "outside_harm_rate": 0.0,
         "poi_corruption_rate": 0.0, "cap_delta": 0, "new_severe_truncations": 0}
    p.update(kw)
    return an.safety(p, C)["pass"]


def test_safety_boundaries_and_none_semantics():
    assert _pt() and _pt(mer_increase=0.01) and not _pt(mer_increase=0.0100001) and _pt(zh_cer_increase=0.015) and not _pt(zh_cer_increase=0.0151)
    assert _pt(zh_retention=0.98) and not _pt(zh_retention=0.9799) and _pt(en_retention=0.95) and not _pt(en_retention=0.9499)
    assert _pt(outside_harm_rate=0.03) and not _pt(outside_harm_rate=0.0301) and _pt(poi_corruption_rate=0.05) and not _pt(poi_corruption_rate=0.0501)
    assert _pt(cap_delta=1) and not _pt(cap_delta=2) and _pt(cap_delta=-1) and not _pt(new_severe_truncations=1)
    assert _pt(zh_retention=None, en_retention=None, outside_harm_rate=None, poi_corruption_rate=None)       # vacuous, reported not-assessable
    p = {"mer_increase": 0, "zh_cer_increase": 0, "zh_retention": None, "en_retention": 1, "outside_harm_rate": 0, "poi_corruption_rate": 0,
         "cap_delta": 0, "new_severe_truncations": 0}
    assert an.safety(p, C)["not_assessable"] == ["matrix_ZH_retention"]


def test_rescue_and_benefit_boundaries():
    assert an.rescue(10, 10, 9, C)["required_R_ZH"] == 1 and an.rescue(10, 10, 9, C)["pass"] and not an.rescue(10, 10, 10, C)["pass"]
    assert an.rescue(10, 5, 4, C)["X_ZH"] == 0 and an.rescue(10, 5, 4, C)["pass"] and not an.rescue(10, 5, 6, C)["pass"]     # no waiver absent excess
    r = an.rescue(18, 41, 29, C)
    assert r["X_ZH"] == 23 and r["required_R_ZH"] == 12 and r["pass"] and not an.rescue(18, 41, 30, C)["pass"]
    assert an.rescue(0, 1, 0, C)["required_R_ZH"] == 1
    b = an.benefit(119, 101, 105, 0.3, 0.3, True, C)
    assert b["I_A4"] == 18 and b["I_G1"] == 14 and b["retention"] == 14 / 18 and b["BENEFIT_RETENTION_PASS"] and b["USEFUL_GAIN_PASS"]
    b1 = an.benefit(119, 101, 106, 0.3, 0.3, True, C)
    assert b1["I_G1"] == 13 and not b1["BENEFIT_RETENTION_PASS"] and b1["USEFUL_GAIN_PASS"]                     # OVERCONSERVATIVE case
    assert an.benefit(120, 100, 105, 0.3, 0.3, True, C)["BENEFIT_RETENTION_PASS"]                                # 15/20 = 0.75 inclusive
    b2 = an.benefit(100, 101, 99, 0.3, 0.3, True, C)
    assert b2["retention"] is None and b2["BENEFIT_RETENTION_PASS"] and not an.benefit(100, 101, 100, 0.3, 0.3, True, C)["BENEFIT_RETENTION_PASS"]
    assert an.benefit(100, 90, 95, 0.305, 0.3, True, C)["USEFUL_GAIN_PASS"] and not an.benefit(100, 90, 95, 0.30501, 0.3, True, C)["USEFUL_GAIN_PASS"]
    assert not an.benefit(100, 90, 95, 0.3, 0.3, False, C)["USEFUL_GAIN_PASS"] and not an.benefit(100, 90, 100, 0.3, 0.3, True, C)["USEFUL_GAIN_PASS"]


def test_breadth_concentration_and_lodo():
    single = an.breadth([20] + [0] * 75, ["d0"] + [f"d{i % 20}" for i in range(1, 76)], C)
    assert not single["BREADTH_PASS"] and single["C_utt"] == 1.0
    same = an.breadth([2, 2, 2], ["a", "a", "a"], C)
    assert not same["BREADTH_PASS"] and same["N_rescue_utt"] == 3 and same["N_rescue_dialogue"] == 1
    ok = an.breadth([2, 1, 1], ["a", "b", "c"], C)
    assert ok["C_utt"] == 0.5 and ok["C_dlg"] == 0.5 and ok["BREADTH_PASS"]                     # inclusive 0.50
    dl = an.breadth([2, 1, 1, 1], ["a", "a", "b", "c"], C)
    assert dl["C_dlg"] == 0.6 and dl["BREADTH_PASS"]                                             # inclusive 0.60
    assert not an.breadth([3, 1, 1, 1], ["a", "a", "b", "c"], C)["BREADTH_PASS"]
    harm = an.breadth([2, 1, 1, -4], ["a", "b", "c", "d"], C)
    assert harm["R_plus"] == 4 and harm["R_minus"] == 4 and harm["R_net_76"] == 0 and not harm["BREADTH_PASS"] and harm["C_utt"] == 0.5
    zero = an.breadth([0, 0, -1], ["a", "b", "c"], C)
    assert zero["C_utt"] == zero["C_dlg"] == 1.0 and not zero["BREADTH_PASS"]
    dd = [f"d{i:02d}" for i in range(20)]
    lo = an.lodo([1] * 18 + [0, 0], dd)
    assert lo["dialogues"] == 20 and lo["required"] == 18 and lo["count_positive"] == 20 and lo["LODO_ROBUST"]
    lo2 = an.lodo([5, 0, -1] + [0] * 17, dd)
    assert lo2["min"] == -1 and lo2["count_positive"] == 19 and lo2["LODO_ROBUST"]
    lo3 = an.lodo([1, 1] + [0] * 18, dd)
    assert lo3["count_positive"] == 20
    lo4 = an.lodo([1] + [0] * 19, dd)
    assert lo4["count_positive"] == 19 and lo4["LODO_ROBUST"]
    lo5 = an.lodo([1, -1, 1] + [0] * 17, dd)
    assert lo5["count_positive"] == 18 and lo5["LODO_ROBUST"]                                  # exactly 18 of 20 removals
    lo6 = an.lodo([2, -2] + [0] * 18, dd)
    assert lo6["count_positive"] == 1 and not lo6["LODO_ROBUST"] and lo6["median"] == 0


def test_label_precedence_exhaustive():
    import itertools
    for v, s, u, r, b in itertools.product((False, True), repeat=5):
        lab = an.decide(v, s, u, r, b)
        assert lab == au.own_label(v, s, u, r, b)
        exp = ("P2_PATH3_INVALID" if not v else "P2_PATH3_SEQUENCE_DAMAGE" if not s else "P2_PATH3_NO_USEFUL_GAIN" if not u
               else "P2_PATH3_OVERCONSERVATIVE" if not r else "P2_PATH3_SIGNAL_CONCENTRATED" if not b else "P2_PATH3_SUPPORTED")
        assert lab == exp and lab in an.LABELS


def test_bootstrap_deterministic_shared_draws():
    from experiments.inference_cf_p2seq_analyze import draws
    D = [f"d{i // 5:02d}" for i in range(100)]
    a, b = draws(D, reps=2000, seed=240924), draws(D, reps=2000, seed=240924)
    assert len(a) == 2000 and all(np.array_equal(x, y) for x, y in zip(a, b)) and all(len(x) == 100 for x in a)
    ca, cb = np.ones((100, 8)), np.ones((100, 8))
    ca[:, 0] = 2
    r = an.count_delta_ci(ca, cb, 0, a)
    assert r["delta"] == 100 and r["ci95"] == [100.0, 100.0] and r["valid_draws"] == 2000


def test_reference_firewall_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2path3/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2path3/run1")
    assert not called
    asrc = (ROOT / "experiments/inference_cf_p2path3_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2path3_analyze|inference_cf_p2path3\b|consensus_guard|branch_adjudication|path_decode)", asrc, re.M) is None
    fsrc = (ROOT / "experiments/inference_cf_p2path3_analyze.py").read_text()
    assert "load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
    rsrc = (ROOT / "experiments/inference_cf_p2path3.py").read_text()
    assert "load_references" not in rsrc and "num_poi" not in rsrc and "utt_counts" not in rsrc
    assert math.ceil(0.9 * 20) == 18
