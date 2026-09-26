"""P2-RJ-E leverage expansion: complete deterministic population, identity with P2-RJ definitions,
competitor rule, overlap identity, frozen Q50 / terminal labels, role exclusion, no gradient
steering, serialization and manifest inputs."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inference_cf_p1 import CB, CE, PART, VOCAB, encoded, tiny_bundle  # noqa: E402

import experiments.inference_cf_cached as cached  # noqa: E402
import experiments.inference_cf_p2r as p2r  # noqa: E402
import experiments.inference_cf_p2rj as rj  # noqa: E402
import experiments.inference_cf_p2rj_analyze as rja  # noqa: E402
import experiments.inference_cf_p2rj_prepare as rjprep  # noqa: E402
import experiments.inference_cf_p2rje as rje  # noqa: E402
import experiments.inference_cf_p2rje_analyze as ana  # noqa: E402
from csasr.inference_cf.core import digest, file_hash  # noqa: E402

CFG = json.loads(Path(rje.CONFIG).read_text())
RJ_CFG = json.loads(Path("configs/inference_cf/p2_rj_jacobian_diagnosis.json").read_text())
POS = json.loads(Path(rje.POSITIONS).read_text())
RJ_POS = json.loads(Path(rje.P2RJ_POSITIONS).read_text())


def test_protocol_identical_to_p2rj():
    assert CFG["energy"]["e_star"] == RJ_CFG["energy"]["e_star"]
    assert CFG["site"]["layer"] == RJ_CFG["site"]["layer"] == 16
    assert CFG["site"]["num_forced_prefix"] == RJ_CFG["site"]["num_forced_prefix"]
    assert CFG["conditions"] == RJ_CFG["conditions"]
    assert CFG["baseline_run"] == RJ_CFG["baseline_run"] and CFG["baseline_system"] == RJ_CFG["baseline_system"]
    b, rb = CFG["analysis"]["bootstrap"], RJ_CFG["analysis"]["bootstrap"]
    assert (b["replicates"], b["seed"], b["unit"]) == (rb["replicates"], rb["seed"], rb["unit"])
    assert CFG["analysis"]["alpha_per_bound"] == ana.ALPHA_PER_BOUND == 0.05 / len(rja.DECISION_FAMILY)
    assert CFG["analysis"]["leverage"] == RJ_CFG["analysis"]["leverage"]
    # the reused P2-RJ code is byte-identical to what produced P2-RJ run1
    man = json.loads(Path(CFG["p2rj_reference"]["run"], "manifest.json").read_text())
    assert man["manifest_hash"] == CFG["p2rj_reference"]["run_manifest_hash"]
    for rel in ("experiments/inference_cf_p2rj.py", "experiments/inference_cf_p2rj_analyze.py", "experiments/inference_cf_p2r.py",
                "experiments/inference_cf_cached.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py",
                "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py"):
        assert file_hash(Path(rel)) == man["sources"][rel], rel


def test_terminal_labels_frozen():
    assert ana.final_label(True, "SUBSTANTIAL", "SUBSTANTIAL", True) == "P2_RJ_E_DIRECTION_ISSUE"
    assert ana.final_label(True, "WEAK", "WEAK", True) == "P2_RJ_E_DIRECTION_AND_SITE_ISSUE"
    assert ana.final_label(True, "UNRESOLVED", "UNRESOLVED", True) == "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED"
    assert ana.final_label(True, "WEAK", "UNRESOLVED", True) == "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED"
    assert ana.final_label(True, "SUBSTANTIAL", "SUBSTANTIAL", False) == "P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED"
    assert ana.final_label(False, "SUBSTANTIAL", "SUBSTANTIAL", True) == "INVALID_RUN_NO_DIAGNOSIS"
    b = lambda l, h: {"ci": [l, h]}
    assert rja.leverage_class(b(0.51, 0.7)) == "SUBSTANTIAL" and rja.leverage_class(b(0.2, 0.49)) == "WEAK"
    # Q50 = mean over dialogues of within-dialogue fraction rho >= 0.5
    recs = [{"dialogue_id": "A", "rho": 0.6, "kappa_plus_d": 0.0}, {"dialogue_id": "A", "rho": 0.4, "kappa_plus_d": 0.0},
            {"dialogue_id": "B", "rho": 0.5, "kappa_plus_d": 0.0}]
    for r in recs:
        r["stratum"] = "EN-confusion"
    q = rja.leverage_alignment(recs, 100, 0, ana.ALPHA_PER_BOUND)["Q50"]
    assert q["estimate"] == pytest.approx((0.5 + 1.0) / 2)


def test_population_complete_unique_and_contains_original_60():
    assert POS["positions_hash"] == digest({k: v for k, v in POS.items() if k != "positions_hash"})
    assert POS["config_hash"] == digest(CFG)
    ps = POS["positions"]
    keys = [(p["utterance_id"], p["t"]) for p in ps]
    assert len(set(keys)) == len(keys) == 227 and keys == sorted(keys)
    p2r_pop = json.loads(Path("results/inference_cf/p2r/population.json").read_text())
    assert len(ps) == p2r_pop["ledger"]["EN-confusion|valid"]
    assert POS["counts"] == {"total": 227, "original": 60, "new": 167, "utterances": 43, "dialogues": 17}
    orig = {(p["utterance_id"], p["t"]): p for p in RJ_POS["positions"] if p["stratum"] == "EN-confusion"}
    got = {(p["utterance_id"], p["t"]): p for p in ps if p["original"]}
    assert set(got) == set(orig)
    for k, p in got.items():
        o = orig[k]
        assert (p["target_ids"], p["baseline_token"], p["competitor"], p["p2r"], p["dialogue_id"]) == \
            (o["target_ids"], o["baseline_token"], o["competitor"], o["p2r"], o["dialogue_id"])
    assert all(p["competitor"] is None for p in ps if not p["original"])
    assert all(p["baseline_token"] not in p["target_ids"] for p in ps)          # EN-confusion by construction
    panel = {r["utterance_id"] for r in json.loads(Path("results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    assert set(POS["utterances"]) <= panel and len(panel) == 300                  # D-dev-select only


def test_population_rebuild_is_deterministic_with_unchanged_p2r_eligibility():
    pytest.importorskip("pyarrow")
    cands, ledger, rebuilt = rje.eligible_en_confusion()
    assert rebuilt == CFG["p2r_reference"]["population_hash"]                   # eligibility code unchanged
    assert rje.build_positions(cands, RJ_POS["positions"]) == POS["positions"]
    assert ledger == POS["ledger_en_confusion"]


def test_competitor_rule_equals_frozen_p2rj_derivation():
    torch.manual_seed(3)
    for step in (1, 5):
        z = torch.randn(VOCAB) * 3
        tgt = [int(i) for i in torch.topk(z, 3).indices[1:]]                    # make targets high-ranked
        s, _ = p2r.summarize(z, step, [0, 2], [], PART, tgt, None)
        assert rje.competitor_rule(z, step, [0, 2], [], tgt)["competitor"] == rjprep.competitor(s, tgt)
    z = torch.zeros(VOCAB); z[10] = z[11] = 5.0                                 # exact tie is recorded
    assert rje.competitor_rule(z, 1, [], [], [44])["tied_non_reference"] == 1


def _baseline(bundle, monkeypatch):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    return cached.cached_decode(bundle, waveform=np.ones(64 * 320, dtype=np.float32), encoded=encoded(),
                                conditions={"cB": CB, "cE": CE, "language_token_ids": [4, 5]}, partition=PART,
                                null_probs={4: .5, 5: .5}, language_ids=(4, 5), layer=16, alpha=0.0,
                                max_new_tokens=8, num_forced_prefix=4)


def test_prepass_competitor_and_overlap_identity_with_p2rj_runner(monkeypatch):
    b = tiny_bundle()
    b.model.requires_grad_(False)
    toks = _baseline(b, monkeypatch)["tokens"]
    cfg = json.loads(json.dumps(CFG)); cfg["conditions"] = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}
    cfg["energy"]["e_star"] = 0.03
    ctx = rj.Ctx(b, cfg, PART, 4)
    plist = [{"t": t, "stratum": "EN-confusion", "target_ids": [44, 45], "competitor": None, "baseline_token": toks[t],
              "original": False} for t in (1, 2)]
    comp = rje.competitor_prepass(ctx, encoded=encoded(), tokens=toks, positions=plist)
    pc, check = rje.with_competitors(plist, comp)
    recs, vecs = rj.run_utterance(ctx, encoded=encoded(), uid="u", tokens=toks, positions=pc, mode="bf16")
    # the same positions with an explicitly supplied competitor give bitwise-identical P2-RJ outputs
    recs2, vecs2 = rj.run_utterance(ctx, encoded=encoded(), uid="u", tokens=toks,
                                    positions=[dict(p) for p in pc], mode="bf16")
    assert json.dumps(recs, sort_keys=True) == json.dumps(recs2, sort_keys=True)
    assert all(np.array_equal(vecs[k], vecs2[k]) for k in vecs)
    for p in pc:
        idt = recs[str(p["t"])]["identity"]
        assert idt["logits_bitwise_equal"] and idt["site_equals_recorder"]
        assert idt["logp_competitor"] == idt["max_other_logp"]                   # c* is the best non-reference token
        assert recs[str(p["t"])]["stats"]["A"] == pytest.approx(
            float(np.linalg.norm(rj.tangent(torch.from_numpy(vecs[f"t{p['t']}_g_margin"]).double(),
                                            torch.from_numpy(vecs[f"t{p['t']}_r"]).double()))) * 0.03, rel=1e-9)
    assert check["1"]["rule_equals_frozen"] is None


def test_gradient_never_applied_and_no_forbidden_roles():
    src = Path("experiments/inference_cf_p2rje.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in {"apply_steering", "hook_edit", "solve_scale", "chord_edit", "scaled_direction",
                                "DecoderPostCrossAttnInterventionHook", "pulse_hook", "_edit_hook", "cached_decode"}, name
    assert "hook=" not in src
    low = (src + Path("experiments/inference_cf_p2rje_analyze.py").read_text()).lower()
    for w in ("dev-confirm", "dev_confirm", "d-test", "d_test", "dtest", "router-calib", "router_calib", "seame", "fleurs",
              "vimed", "ascend", "qwen"):
        assert w not in low, w


def test_sources_and_serialization():
    for p in rje.SOURCES:
        if p != "docs/inference_cf/P2_RJ_E_PRE_RUN_AUDIT.md":
            assert Path(p).exists(), p
    assert rje.CONFIG in rje.SOURCES and rje.POSITIONS in rje.SOURCES and "experiments/inference_cf_p2rje_analyze.py" in rje.SOURCES
    assert json.loads(json.dumps(POS)) == POS
