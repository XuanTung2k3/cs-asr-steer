"""P2-R mechanism diagnosis: frozen random direction, energy solver, pulse isolation, replay
equivalence with the deployable cached path, relocation rules, oracle isolation, decisions."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, FRAMES, PART, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2r as p2r  # noqa: E402
import experiments.inference_cf_p2r_analyze as an  # noqa: E402
from csasr.models.hooks import apply_steering  # noqa: E402

CFG = json.loads(Path("configs/inference_cf/p2_r_mechanism_diagnosis.json").read_text())
COND = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}


def _ctx(bundle, target=None, layer=16):
    cfg = json.loads(json.dumps(CFG))
    cfg["conditions"] = COND
    cfg["site"]["layer"] = layer
    cfg["site"]["max_new_tokens"] = 8
    if target is not None:
        cfg["energy"]["target_edit_norm"] = target
    return p2r.Ctx(bundle, cfg, PART, {4: .5, 5: .5}, (4, 5), 4)


def _baseline(bundle, monkeypatch, alpha=0.0, max_new_tokens=8):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    return cached.cached_decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=encoded(),
                                conditions=COND, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                                layer=16, alpha=alpha, max_new_tokens=max_new_tokens, num_forced_prefix=4)


def test_config_frozen_values():
    assert CFG["site"]["layer"] == 16 and CFG["d2"]["alpha"] == 2.0 and CFG["site"]["min_step"] == 1
    assert CFG["energy"]["target_edit_norm"] == pytest.approx(math.sqrt(1203.3762345332195 / 949))
    assert CFG["population"]["positions_per_stratum"] == 60 and CFG["population"]["max_utterances"] == 80
    assert CFG["analysis"]["bootstrap"]["replicates"] == 10000 and len(an.PRIMARY) == len(CFG["analysis"]["primary_family"])
    assert list(an.PRIMARY) == CFG["analysis"]["primary_family"]


def test_random_direction_is_frozen_unit_and_orthogonal():
    r = torch.randn(1280, dtype=torch.float32)
    v1, s1 = p2r.random_direction("U1", 7, r)
    v2, _ = p2r.random_direction("U1", 7, r)
    v3, _ = p2r.random_direction("U1", 8, r)
    assert s1 == "ok" and torch.equal(v1, v2) and not torch.equal(v1, v3)
    assert float(v1.norm()) == pytest.approx(1.0, abs=1e-12)
    assert abs(float(torch.dot(v1, r.double()))) < 1e-9 * float(r.norm())
    seed = int(hashlib.sha256(b"P2R-random-v1|U1|7").hexdigest()[:16], 16)
    assert p2r.random_seed("U1", 7) == seed
    z = torch.from_numpy(np.random.default_rng(seed).standard_normal(1280))
    rh = r.double() / r.double().norm()
    ref = z - torch.dot(z, rh) * rh
    assert torch.allclose(v1, ref / ref.norm())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_solver_hits_target_with_exact_hook_arithmetic(dtype):
    torch.manual_seed(0)
    site = (torch.randn(1280) * 0.3).to(dtype)
    target = 0.12 * float(site.float().norm())
    for v in (torch.randn(1280), -torch.randn(1280)):
        sol = p2r.solve_scale(site, v, target)
        assert sol["status"] == "ok"
        e = p2r.emulate_edit_norm(site, v.double() / v.double().norm(), sol["s"])
        assert e == sol["edit_norm"]
        assert abs(e * e / target ** 2 - 1) <= (2e-3 if dtype == torch.float32 else 0.02)
    # a direction parallel to the state cannot rotate it: unreachable, never silently relaxed
    assert p2r.solve_scale(site, site.double(), target)["status"] == "energy_unreachable"
    # the emulation is the hook's own function
    vv = torch.randn(1280, dtype=torch.float64); vv /= vv.norm()
    x = site.reshape(1, 1, -1)
    st = apply_steering(x, (0.5 * vv).to(dtype).reshape(1, 1, -1), 1.0, 1.0, torch.ones((1, 1), dtype=dtype), True)
    assert p2r.emulate_edit_norm(site, vv, 0.5) == float((st - x).float().norm())


def test_pulse_pass_same_state_energy_matched_restore_and_continuation(monkeypatch):
    b = tiny_bundle()
    base = _baseline(b, monkeypatch)
    toks = base["tokens"]
    assert len(toks) >= 3
    ctx = _ctx(b, target=0.03)
    jobs = {t: {"target_ids": [toks[t]], "stratum": "ZH-correct",
                "arms": {a: {"dir": a, "energy": 0.03, "continue": t == 2} for a in p2r.ARMS}} for t in (1, 2)}
    with torch.inference_mode():
        out = p2r.pulse_pass(ctx, encoded=encoded(), uid="u", tokens=toks, jobs=jobs)
    assert "_baseline_mismatch" not in out
    for t in (1, 2):
        rec = out[t]
        assert rec["restore_bitwise"] and rec["dir"] == "ok"
        es = []
        for a in p2r.ARMS:
            x = rec["arms"][a]
            assert x["solver"]["status"] == "ok" and x["solver_matches_hook"]
            es.append(x["edit_norm"] ** 2)
            assert abs(x["edit_norm"] ** 2 / 0.03 ** 2 - 1) <= 0.02
            assert x["summary"]["kl_to_none"] is not None
        assert max(es) - min(es) <= 0.02 * 0.03 ** 2
        assert ("continuation" in rec["arms"]["plus_d"]) == (t == 2)
    # +d and -d really are opposite at the identical position/state
    assert out[1]["arms"]["plus_d"]["solver"]["phi"] == pytest.approx(math.pi - out[1]["arms"]["minus_d"]["solver"]["phi"], abs=1e-6)


def test_current_replay_reproduces_deployable_cached_decode(monkeypatch):
    b = tiny_bundle()
    ref = _baseline(b, monkeypatch, alpha=2.0)
    ctx = _ctx(b)
    with torch.inference_mode():
        rep = p2r.replay(ctx, encoded=encoded(), waveform=np.ones(FRAMES * 320, dtype=np.float32), uid="",
                         tokens=ref["tokens"], terminated=ref["terminated"], lid_cache={}, mode="current",
                         plan=None, record_at={})
    assert rep["lineage_ok"]
    assert len(rep["steps"]) == len(ref["steps"])
    for a, s in zip(rep["steps"], ref["steps"]):
        assert a["g"] == s["g"] and a["fb"] == s["fallback_reason"] and a["edit"] == s["edit_applied"]
        audit = s["hook_audit"]
        assert a["edit_norm"] == (audit["edit_norm"] if s["edit_applied"] else 0.0)
        assert a["s_argmax"] == s["steered_next"]


def test_relocation_plan_frozen_rules():
    steps = [{"t": t, "edit": e > 0, "edit_norm": e, "dir": "ok"} for t, e in
             enumerate([0, .9, 0, .5, 0, 1.2, 0, .5, 0])]
    targets = [{"t": 6}, {"t": 3}, {"t": 2}, {"t": 8}]            # frozen hash order
    plan = p2r.relocation_plan({"steps": steps}, targets)
    assert plan["retained"] == [3]
    # donors by descending energy (ties by position): 5 (1.2), 1 (.9), 7 (.5); open targets 6, 2, 8
    assert [(m["from"], m["to"], m["energy"]) for m in plan["moves"]] == [(5, 6, 1.2), (1, 2, .9), (7, 8, .5)]
    assert [m["source_after_target"] for m in plan["moves"]] == [False, False, False]
    assert plan["unmatched_donors"] == [] and plan["unfilled_targets"] == []


def test_oracle_and_reference_information_cannot_enter_deployable_path():
    import ast
    banned = ("p2r", "target", "oracle", "reference", "evaluation_units", "ctc")
    for mod in ("experiments/inference_cf_cached.py", "experiments/inference_cf_p2.py"):
        tree = ast.parse(Path(mod).read_text())
        idents = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                idents.add(node.id)
            elif isinstance(node, ast.Attribute):
                idents.add(node.attr)
            elif isinstance(node, ast.arg):
                idents.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                idents.update(a.name for a in node.names)
                idents.add(getattr(node, "module", "") or "")
        assert not [i for i in idents if any(b in i.lower() for b in banned)], mod
    params = set(inspect.signature(cached.cached_decode).parameters)
    assert not params & {"plan", "targets", "target_ids", "positions", "reference"}
    runner = Path("experiments/inference_cf_p2r.py").read_text()
    assert "load_references" not in runner and "evaluation_units" not in runner and "transcript_raw" not in runner
    assert "inference_cf_p2r_population" not in runner


def test_population_reproducible_from_frozen_config():
    pytest.importorskip("pyarrow")
    import experiments.inference_cf_p2r_population as popm
    frozen = Path("results/inference_cf/p2r/population.json")
    if not frozen.exists():
        pytest.skip("population not frozen yet")
    saved = json.loads(frozen.read_text())
    pop = popm.build(json.loads(Path("configs/inference_cf/p2_r_mechanism_diagnosis.json").read_text()))
    from csasr.inference_cf.core import digest
    assert digest(pop) == saved["population_hash"]
    assert all(saved["estimable"].values())
    for s, lst in saved["d1"].items():
        assert len(lst) == 60 and max(__import__("collections").Counter(c["utterance_id"] for c in lst).values()) <= 2
        assert all(c["t"] >= 1 and c["target_ids"] for c in lst)
    assert len(saved["utterances"]) <= 80


def _ci(a, b):
    return {"estimate": (a + b) / 2, "ci": [a, b]}


def _prim(**kw):
    base = {k: _ci(-0.1, 0.1) for k in an.PRIMARY}
    base["D3.P_plus_random"] = base["D3.P_minus_random"] = _ci(-0.001, 0.001)
    base["D2.P_oracle_current"] = _ci(-0.001, 0.001)
    base.update({k.replace("__", "."): v for k, v in kw.items()})
    return base


def _facts(**kw):
    comp = {a: {"target_recovered": 0} for a in p2r.ARMS}
    f = {"companion": comp, "delta_p": 1 / 60, "delta_p_d2": 1 / 200, "random_top1_flips": 5,
         "random_companion_changed": 1, "d2_pulses": {"current_recovered": 0, "oracle_recovered": 0},
         "plus_d_confusion_correction_rate": 0.0, "edits_nonzero": True, "d2_current_net_correction": 0,
         "validity_ok": True, "d2_energy_ok": True}
    f.update(kw)
    return f


def _sec():
    z = _ci(-0.1, 0.1)
    return {"D1.L_random": z, "D1.dPE_plus": z, "D2.Lex_oracle_current": z, "D2.L_current_none": z,
            "D1.L_plus_EN-correct": z}


def test_decision_rules_frozen_profiles():
    # generic perturbation: random acts, no sign-specific advantage, equivalence at delta_p
    d = an.decide(_prim(), _sec(), _facts())
    assert d["D3"] == "GENERIC_SUPPORTED" and d["D1"] == "DIRECTION_UNINFORMATIVE"
    assert d["D2"].startswith("D2-B") and d["primary_diagnosis"] == "GENERIC_PERTURBATION"
    # sign contradiction -> direction issue (also blocks the generic profile via G3)
    p = _prim(D1__L_minus=_ci(.1, .5), D1__L_plus_minus=_ci(-.6, -.1), D1__Lex_plus_minus=_ci(-.5, -.05))
    d = an.decide(p, _sec(), _facts())
    assert d["D1"] == "DIRECTION_SIGN_CONTRADICTION" and d["primary_diagnosis"] == "DIRECTION_ISSUE"
    # wide equivalence intervals: nonsignificance is not equivalence -> ambiguous
    d = an.decide(_prim(D3__P_plus_random=_ci(-.2, .2)), _sec(), _facts())
    assert d["D3"] == "GENERIC_NOT_ESTABLISHED" and d["primary_diagnosis"] == "MECHANISM_STILL_AMBIGUOUS"
    # positive semantics + oracle rescue -> localization
    p = _prim(D1__L_plus=_ci(.1, .4), D1__L_plus_minus=_ci(.1, .5), D1__Lex_plus_minus=_ci(.05, .3),
              D2__L_oracle_current=_ci(.2, .6), D2__L_oracle_none=_ci(.3, .8), D2__P_oracle_current=_ci(.01, .1))
    s = _sec(); s["D2.Lex_oracle_current"] = _ci(.01, .2)
    d = an.decide(p, s, _facts())
    assert d["D1"] == "DIRECTION_POSITIVE_SEMANTICS" and d["primary_diagnosis"] == "LOCALIZATION_ISSUE"
    # invalid experiment never yields a single-cause diagnosis
    assert an.decide(_prim(), _sec(), _facts(validity_ok=False))["primary_diagnosis"] == "MECHANISM_STILL_AMBIGUOUS"
    # D2 energy mismatch cannot support any D2 reading
    assert an.decide(_prim(), _sec(), _facts(d2_energy_ok=False))["D2"] == "D2_UNRESOLVED_ENERGY_MISMATCH"


def test_dialogue_bootstrap_averages_within_dialogue_first():
    vals = [("a", 1.0), ("a", 1.0), ("a", 1.0), ("b", 0.0)]
    r = an.dialogue_bootstrap(vals, 2000, 1, 0.05)
    assert r["estimate"] == pytest.approx(0.5) and r["dialogues"] == 2 and r["positions"] == 4


def test_run_utterance_end_to_end_tiny_oracle_replay_and_serialization(monkeypatch):
    b = tiny_bundle()
    base = _baseline(b, monkeypatch, max_new_tokens=8)
    toks = base["tokens"]
    ctx = _ctx(b, target=0.03)
    ctx.cfg["d2"]["alpha"] = 0.05                        # tiny states (|r| ~ 0.14) need small packets
    real_gate = p2r.gate_step

    def gate_zero_at_targets(ctx_, waveform, prefix, *a, **k):   # make targets unedited -> relocations
        out = real_gate(ctx_, waveform, prefix, *a, **k)
        if len(prefix) in (1, 3):
            out = {**out, "g": 0.0}
        return out
    monkeypatch.setattr(p2r, "gate_step", gate_zero_at_targets)
    d1 = [{"utterance_id": "u", "t": t, "stratum": s, "target_ids": [toks[t]], "companion": t == 1}
          for t, s in ((1, "EN-confusion"), (2, "ZH-correct"))]
    d2t = [{"utterance_id": "u", "t": t, "target_ids": [toks[t]], "stratum": "EN-confusion",
            "ctc_midpoint_sec": None} for t in (3, 1)]
    d2c = [{"utterance_id": "u", "t": 2, "target_ids": [toks[2]], "stratum": "ZH-correct"}]
    budget = []
    with torch.inference_mode():
        res = p2r.run_utterance(ctx, encoded=encoded(), waveform=np.ones(FRAMES * 320, dtype=np.float32), uid="u",
                                tokens=toks, terminated=base["terminated"], lid_cache={}, d1_positions=d1,
                                d2_targets=d2t, d2_controls=d2c, pulse_budget=budget,
                                oracle_crop=lambda t, gs: {"status": "no_ctc_midpoint"})
    json.dumps(res)                                              # serializable
    assert res["current"]["lineage_ok"] and res["oracle"]["lineage_ok"]
    assert all(s["baseline_argmax_ok"] for p in ("current", "oracle") for s in res[p]["steps"])
    assert [(a["g"], a["fb"]) for a in res["current"]["steps"]] == [(a["g"], a["fb"]) for a in res["oracle"]["steps"]]
    assert res["baseline_mismatch"] == []
    plan = res["plan"]
    moved = {m["to"]: m for m in plan["moves"]}
    ora = {s["t"]: s for s in res["oracle"]["steps"]}
    assert len(plan["moves"]) == 2 and plan["retained"] == []
    for m in plan["moves"]:
        assert ora[m["to"]]["solver"]["status"] == "ok" and ora[m["to"]]["edit"]
        assert ora[m["from"]]["kind"] == "moved_away" and not ora[m["from"]]["edit"]
        assert ora[m["to"]]["kind"] == "relocated_in"
        if ora[m["to"]]["solver"]["status"] == "ok":
            assert abs(ora[m["to"]]["edit_norm"] ** 2 / m["energy"] ** 2 - 1) <= 0.02
    for t in (1, 2):
        assert res["pulses"][str(t)]["restore_bitwise"]
    for s in res["current"]["steps"]:
        if s["t"] in (1, 2, 3):
            assert "none" in s and "arm" in s
    assert len(budget) == len(res["pulse_companions"]) == len([m for m in plan["moves"]][:20])


def test_pulse_at_final_eos_step_is_supported(monkeypatch):
    """A P2 donor edit can sit on the final step that emits EOS (t == len(tokens))."""
    b = tiny_bundle()
    toks = _baseline(b, monkeypatch)["tokens"][:3]          # truncated: step 3 plays the EOS-step role
    ctx = _ctx(b, target=0.03)
    jobs = {3: {"target_ids": None, "stratum": None,
                "arms": {"pulse_current_9": {"dir": "plus_d", "energy": 0.03, "continue": True}}}}
    with torch.inference_mode():
        out = p2r.pulse_pass(ctx, encoded=encoded(), uid="u", tokens=toks, jobs=jobs)
    assert out[3]["restore_bitwise"] and out[3]["arms"]["pulse_current_9"]["solver"]["status"] == "ok"
    assert "continuation" in out[3]["arms"]["pulse_current_9"]
    assert out.get("_baseline_mismatch", []) in ([], [3])    # tiny model need not emit EOS there
