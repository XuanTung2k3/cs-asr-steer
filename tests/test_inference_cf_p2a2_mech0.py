"""P2-A2-MECH0 focused tests (freeze 9d89640): panel identity and reference-free strata/CONTROL40/MECH_PANEL reconstruction; exact A2
settings and untouched A2 source; step observer off/on exact (losses, gradients, final masters) with post-step timing and two distinct
snapshots; 194-tensor SELF/CROSS/POST partition; float64 squared norm aggregation (master and bf16-effective); exact family drop with no
reoptimization; teacher-forced geometry (STEP1 prefix mismatch accepted, processed logp, suppressed EOS -> None, H3 continuation, H0);
action retention / 0.25 sensitivity boundary; mechanism-label precedence vs the independent auditor; readiness; reference barrier;
auditor independence."""
from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p2tta0 import _adapt, _setup  # noqa: E402

import csasr.inference_cf.episodic_tta as tta  # noqa: E402
import experiments.inference_cf_p2a2_mech0 as run  # noqa: E402
import experiments.inference_cf_p2a2_mech0_analyze as an  # noqa: E402
import experiments.inference_cf_p2a2_mech0_audit as au  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / run.CONFIG).read_text())
P = json.loads((ROOT / run.PANEL).read_text())


def _sealed():
    B0, A2, AUTO = {}, {}, {}
    for r in P["rows"]:
        p5 = json.loads((ROOT / r["PATH5_output"]["path"]).read_text())
        B0[r["utterance_id"]], A2[r["utterance_id"]] = p5["theta0"], p5["A2_free"]
        AUTO[r["utterance_id"]] = json.loads((ROOT / r["B0_AUTO"]["path"]).read_text())["systems"]["B0_AUTO"]
    return B0, A2, AUTO


def test_panel_strata_controls_exact():
    ids = [r["utterance_id"] for r in P["rows"]]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["rows"]}
    fixed = [r["utterance_id"] for r in json.loads((ROOT / run.FIXED).read_text())["rows"]]
    B0, A2, AUTO = _sealed()
    parts, div = run.strata(ids, dlg, B0, A2, AUTO, fixed)
    assert {k: len(parts[k]) for k in run.STRATA} == C["counts"]
    assert all(run.membership_hash(k, parts[k], dlg) == C["partition_sha256"][k] for k in run.STRATA)
    assert all(div[r["utterance_id"]] == r["divergence"] for r in P["rows"])
    assert len(parts["CONTROL40"]) == 40 and all(sum(dlg[u] == d for u in parts["CONTROL40"]) == 2 for d in set(dlg.values()))
    assert set(parts["MECH_PANEL"]) == set(parts["A2_DELTA"]) | set(parts["CONTROL40"]) and not set(parts["A2_DELTA"]) & set(parts["CONTROL40"])
    assert run.divergence({"tokens": [1, 2], "terminated": "cap"}, {"tokens": [1, 2, 3], "terminated": "eos"})["kind"] == "OTHER_INVALID"
    assert run.divergence({"tokens": [1, 2], "terminated": "eos"}, {"tokens": [1, 2], "terminated": "cap"})["kind"] == "OTHER_INVALID"
    assert run.divergence({"tokens": [1], "terminated": "eos"}, {"tokens": [1, 2], "terminated": "eos"})["kind"] == "EOS_RECOVERY"
    assert run.divergence({"tokens": [1, 2], "terminated": "eos"}, {"tokens": [1], "terminated": "eos"})["kind"] == "EOS_REGRESSION"
    short = {u: A2[u] for u in ids}
    keep = [u for u in ids if dlg[u] == "CSD0012"]
    for u in keep:                                                            # force a control shortage in one dialogue
        short[u] = {"tokens": list(B0[u]["tokens"]) + [7], "terminated": "eos"}
    with pytest.raises(ValueError):
        run.strata(ids, dlg, B0, short, AUTO, fixed)


def test_a2_settings_and_untouched_source():
    c5 = json.loads((ROOT / "configs/inference_cf/p2_path5.json").read_text())
    assert C["A2_inherited"] == c5["A2_inherited"] and C["suppression"] == c5["suppression"]
    assert tta.STEPS == 2 and tta.OPTIM["lr"] == 1e-3 and tta.OPTIM["weight_decay"] == 0 and tuple(tta.OPTIM["betas"]) == (0.9, 0.999)
    assert all(run.sha_file(ROOT / p) == h for p, h in C["instrumentation_provenance"]["original_source_baseline_sha256"].items())


def test_family_partition_exact():
    names = [p["name"] for p in C["A2_inherited"]["trainables"]["parameters"]]
    fam = run.family_partition(names, C["families"])
    assert [sum(v == f for v in fam.values()) for f in run.FAMILIES] == [64, 64, 66] and len(fam) == 194
    bad = {**C["families"], "POST": C["families"]["POST"][:-1]}
    with pytest.raises(ValueError):
        run.family_partition(names, bad)
    dup = {**C["families"], "SELF": C["families"]["SELF"] + [C["families"]["CROSS"][0]]}
    with pytest.raises(ValueError):
        run.family_partition(names, dup)
    assert run.layer_of("model.decoder.layer_norm.weight") == "decoder_final" and run.layer_of("model.decoder.layers.7.final_layer_norm.bias") == "layer07"


def _toy_families(names):
    return {"SELF": [n for n in names if ".self_attn_layer_norm." in n], "CROSS": [n for n in names if ".encoder_attn_layer_norm." in n],
            "POST": [n for n in names if ".final_layer_norm." in n or n.startswith("model.decoder.layer_norm.")]}


def test_observer_off_on_exact_and_timing():
    b, names, g = _setup(3)
    y = [10, 11, 12, 13, 50, 51]
    off = _adapt(b, g, y, "A2")
    with run.step_observer() as snaps:
        on = _adapt(b, g, y, "A2")
    assert off["log"]["losses"] == on["log"]["losses"] and off["log"]["grad_l2"] == on["log"]["grad_l2"]
    assert all(torch.equal(off["final_masters"][n], on["final_masters"][n]) for n in names)
    assert len(snaps) == 2 and all(len(s) == len(names) for s in snaps)
    S1, S2 = dict(zip(names, snaps[0])), dict(zip(names, snaps[1]))
    assert all(torch.equal(S2[n], on["final_masters"][n]) for n in names) and any(not torch.equal(S1[n], S2[n]) for n in names)
    assert any(not torch.equal(S1[n], g.theta0[n].float().cpu()) for n in names)          # snapshot is post-step, not theta0
    after = _adapt(b, g, y, "A2")
    assert len(snaps) == 2 and after["log"]["losses"] == off["log"]["losses"]               # hook removed on exit
    fam = run.family_partition(names, _toy_families(names)) if len(names) == 194 else {n: f for f, v in _toy_families(names).items() for n in v}
    th32 = {n: g.theta0[n].float().cpu() for n in names}
    th16 = {n: g.theta0[n].cpu() for n in names}
    nm = run.delta_norms(th32, S2, names, fam, effective=False)
    assert math.isclose(nm["total"] ** 2, sum(v ** 2 for v in nm["family"].values()), rel_tol=1e-12)
    assert math.isclose(nm["total"] ** 2, sum(v ** 2 for v in nm["layer"].values()), rel_tol=1e-12)
    assert math.isclose(nm["total"], math.sqrt(sum(float(((S2[n].double() - th32[n].double()) ** 2).sum()) for n in names)), rel_tol=1e-12)
    ne = run.delta_norms(th32, S2, names, fam, effective=True, theta0_bf16=th16)
    assert math.isclose(ne["total"], math.sqrt(sum(float(((S2[n].to(torch.bfloat16).float() - th16[n].float()).double() ** 2).sum()) for n in names)), rel_tol=1e-12)
    zero = run.delta_norms(th32, th32, names, fam, effective=False)
    assert zero["total"] == 0 and zero["relative"] == 0


def test_drop_family_exact_no_reoptimization():
    names = [p["name"] for p in C["A2_inherited"]["trainables"]["parameters"]]
    final = {n: torch.full((2,), float(i), dtype=torch.bfloat16) for i, n in enumerate(names)}
    th0 = {n: torch.full((2,), -1.0, dtype=torch.bfloat16) for n in names}
    for f in run.FAMILIES:
        st = run.drop_family(final, th0, C["families"][f])
        assert all(torch.equal(st[n], th0[n]) for n in C["families"][f])
        assert all(st[n] is final[n] for n in names if n not in set(C["families"][f]))
    src = (ROOT / "experiments/inference_cf_p2a2_mech0.py").read_text()
    ph2 = src[src.index("Phase 2: MECH_PANEL84"):src.index("def cmd_seal")]
    assert "run_objective" not in ph2 and "adapt(" not in ph2 and "AdamW" not in ph2


class _FakeBranch:
    def __init__(self, bundle, enc, prompt, label):
        self.bundle, self.fed, self.positions = bundle, [], []

    def step(self, new, **kw):
        self.fed += list(new)
        self.positions = list(range(len(self.fed)))
        t = len(self.fed) - 4
        return self.bundle.logits_at(t), None, None


def test_geometry_teacher_forced_prefix_and_continuation(monkeypatch):
    import experiments.inference_cf_cached as cached
    from csasr.lss import sites
    monkeypatch.setattr(sites, "assert_no_site_hooks", lambda b: None)
    monkeypatch.setattr(cached, "Branch", _FakeBranch)

    def logits_at(t):                                   # greedy always token 3; vocabulary 8; EOS id 7
        x = torch.zeros(8)
        x[3] = 5.0
        x[7] = 2.0 + t
        return x
    bundle = SimpleNamespace(logits_at=logits_at)
    guard = SimpleNamespace(current_hash=lambda: "h")
    monkeypatch.setattr(run, "CB", [0, 1, 2, 4])
    res = run.geometry_path(bundle, guard, None, [5, 6], 2, {"EOS": 7, "B0": 7, "A2": 3, "AUTO": None}, [3, 3, 3], "h", [6], [7])
    assert res["prefix_greedy_mismatch_positions"] == [0, 1] and res["fed_ok"] and res["state_locked"]   # non-greedy fixed prefix accepted
    lp = torch.log_softmax(torch.tensor([0, 0, 0, 5.0, 0, 0, -float("inf"), 4.0]), -1)
    assert math.isclose(res["cands"]["A2"]["logp"], float(lp[3]), rel_tol=1e-6) and res["cands"]["AUTO"]["logp"] is None
    assert res["continuation"]["H_eff"] == 3 and len(res["continuation"]["logprobs"]) == 3
    assert math.isclose(res["continuation"]["C_H"], sum(res["continuation"]["logprobs"]) / 3, rel_tol=1e-12)
    r0 = run.geometry_path(bundle, guard, None, [], 0, {"EOS": 7, "B0": 3, "A2": 3}, None, "h", [6], [7])
    assert r0["cands"]["EOS"]["logp"] is None and r0["cands"]["EOS"]["allowed"] is False and "continuation" not in r0
    with pytest.raises(ValueError):
        run.geometry_path(bundle, SimpleNamespace(current_hash=lambda: "other"), None, [], 0, {}, None, "h", [], [])


def test_retention_sensitivity_and_actions():
    assert run.action_class(5, 4, 5) == "A2" and run.action_class(4, 4, 5) == "B0" and run.action_class(9, 4, 5) == "third"
    assert an.retention(["A2"] * 3 + ["B0"]) == 0.75 and an.retention([]) is None
    assert an.sensitive(0.75, 1.0, C) and not an.sensitive(0.76, 0.76, C) and an.sensitive(1.0, 0.75, C) and not an.sensitive(None, None, C)


def test_mechanism_precedence_vs_auditor():
    g = lambda rows, dlg, zh, mx, prow, pdlg: {"rows": rows, "dialogues": dlg, "net_ZH": zh, "net_mixed": mx, "positive_rows": prow, "positive_dialogues": pdlg}
    a = lambda gg: {"rows": gg["rows"], "dlg": gg["dialogues"], "zh": gg["net_ZH"], "mx": gg["net_mixed"], "prow": gg["positive_rows"], "pdlg": gg["positive_dialogues"]}
    cases = []
    for (er, ed), F, dm, fr, (ez, nz), (np_, nd) in itertools.product(((12, 10), (2, 2)), ((0.5, 0.5), (0.49, 0.6), (0.2, 0.24), (0.3, 0.1), (None, 0.6), (0.9, 0.9)),
                                                              (0.1, 0.0), (0.75, 0.74), ((10, 6), (0, 0), (5, 1)), ((3, 3), (2, 2), (1, 1))):
        eg, ng = g(er, ed, ez, 0, min(er, 2), min(ed, 2)), g(32, 10, nz, 0, np_, nd)
        lab = an.decide_mechanism(True, eg, ng, F[0], F[1], dm, fr, C)
        assert lab == au.own_mech(True, a(eg), a(ng), F[0], F[1], dm, fr) and lab in an.LABELS
        cases.append(lab)
    assert {"P2_A2_MECH0_TERMINATION_DOMINANT", "P2_A2_MECH0_NONTERMINATION_DOMINANT", "P2_A2_MECH0_MIXED", "P2_A2_MECH0_NO_CLEAR_MECHANISM"} <= set(cases)
    assert an.decide_mechanism(False, g(12, 10, 9, 9, 9, 9), g(32, 10, 9, 9, 9, 9), 0.9, 0.9, 1, 1, C) == "P2_A2_MECH0_INVALID"
    assert an.decide_mechanism(True, g(12, 10, 9, 9, 9, 9), g(32, 10, 9, 9, 9, 9), None, None, 1, 1, C) == "P2_A2_MECH0_NO_CLEAR_MECHANISM"
    assert an.readiness(True, 0) == "YES" and an.readiness(True, 1) == "NO" and an.readiness(False, 0) == "NO"
    assert an.fraction(3, 0) is None and an.fraction(0, 4) == 0


def test_reference_barrier_and_auditor_independence(monkeypatch):
    monkeypatch.setattr(an, "SEAL", "results/inference_cf/p2a2_mech0/__no_seal__.json")
    called = []
    monkeypatch.setattr("experiments.inference_cf_p2_evaluate.load_references", lambda: called.append(1))
    with pytest.raises(PermissionError):
        an.secondary("results/inference_cf/p2a2_mech0/run1")
    assert not called
    fsrc = (ROOT / "experiments/inference_cf_p2a2_mech0_analyze.py").read_text()
    assert "load_references" not in fsrc[fsrc.index("def primary("):fsrc.index("def secondary(")]
    rsrc = (ROOT / "experiments/inference_cf_p2a2_mech0.py").read_text()
    assert "load_references" not in rsrc and "lockstep_detect" not in rsrc and "online_g1" not in rsrc and "adapt_a4" not in rsrc
    asrc = (ROOT / "experiments/inference_cf_p2a2_mech0_audit.py").read_text()
    assert re.search(r"^\s*(from|import)\s+\S*(p2a2_mech0|consensus_guard|branch_adjudication|path_decode|eos_boundary)", asrc, re.M) is None
