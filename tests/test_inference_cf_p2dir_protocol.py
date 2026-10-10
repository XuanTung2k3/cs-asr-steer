"""P2-DIR protocol (spec sections 3, 6, 10, 11): frozen config/population/construction, runner
sealing and pulse semantics on a bf16 tiny Whisper, D0 == historical P2-R +d pulse, energy matching,
restore/zero-dose identity, lossless logit storage, manifest/firewall rejection."""
from __future__ import annotations

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
import experiments.inference_cf_p2dir as run  # noqa: E402
import experiments.inference_cf_p2dir_prepare as prep  # noqa: E402
import experiments.inference_cf_p2r as p2r  # noqa: E402
from csasr.inference_cf import unique as U  # noqa: E402
from csasr.inference_cf.core import digest, file_hash  # noqa: E402
from csasr.inference_cf.directions import UniqueDirection  # noqa: E402

CFG = json.loads(Path(prep.CONFIG).read_text())
P2R_CFG = json.loads(Path("configs/inference_cf/p2_r_mechanism_diagnosis.json").read_text())
RJ_CFG = json.loads(Path("configs/inference_cf/p2_rj_jacobian_diagnosis.json").read_text())
COND = {"cB": CB, "cE": CE, "language_token_ids": [4, 5]}
E_TINY = 0.03


# ---- frozen values ------------------------------------------------------------------------------

def test_config_frozen_values():
    assert CFG["candidates"] == ["D0", "D1", "D2"] and CFG["layer"] == 16 == run.LAYER
    assert CFG["norm_preserve"] is True and CFG["depth_rescale"] is False
    assert CFG["energy"]["e_star"] == P2R_CFG["energy"]["target_edit_norm"] == RJ_CFG["energy"]["e_star"]
    assert CFG["energy"]["e_star"] == pytest.approx(math.sqrt(1203.3762345332195 / 949))
    assert CFG["bootstrap"]["replicates"] == 10000 and CFG["bootstrap"]["seed"] == 240924
    assert CFG["bootstrap"]["family_size_exp1"] == 10
    assert run.CB == P2R_CFG["conditions"]["cB"] and run.CE == P2R_CFG["conditions"]["cE"]
    assert run.BASELINE_RUN == P2R_CFG["baseline_run"] and run.BASELINE_SYSTEM == P2R_CFG["baseline_system"]
    assert CFG["D1"]["rank_A"] == CFG["D1"]["rank_B"] == U.RANK == 32 and CFG["D1"]["minimum_count_per_group"] == U.MIN_COUNT
    assert CFG["D1"]["degeneracy_relative_tolerance"] == U.REL_TOL and CFG["D1"]["center"] is False
    assert CFG["D2"]["epsilon"] == 1e-12 and CFG["D2"]["tiny_tangent_norm"] == 1e-8
    assert CFG["D0"]["epsilon"] == 1e-6 and CFG["D0"]["tiny_norm"] == 1e-4


def test_population_and_construction_projection_frozen():
    pos = json.loads(Path(prep.POSITIONS).read_text())
    assert pos["positions_hash"] == CFG["positions_hash"] == digest({k: v for k, v in pos.items() if k != "positions_hash"})
    assert json.loads(Path(prep.P2R_POPULATION).read_text())["population_hash"] == CFG["p2r_parent_population_hash"]
    con = json.loads(Path(prep.CONSTRUCTION).read_text())
    assert con["construction_hash"] == digest({k: v for k, v in con.items() if k != "construction_hash"})
    assert all(set(p) == set(U.ALLOWED_FIELDS) for p in con["positions"])
    assert [(p["utterance_id"], p["t"], p["stratum"], p["dialogue_id"]) for p in con["positions"]] == \
        [(p["utterance_id"], p["t"], p["stratum"], p["dialogue_id"]) for p in pos["positions"]]
    assert len(con["positions"]) == 180 and all(p["t"] >= 1 for p in con["positions"])
    assert len(con["folds"]) == 20
    for k, f in con["folds"].items():
        assert f == U.fold_members(con["positions"], k)
        assert len(f["A"]) >= 32 and len(f["B"]) >= 32
    assert con["firewall"]["role"] == "D-dev-select"


def test_firewall_rejects_non_panel_utterance():
    with pytest.raises(ValueError):
        prep.firewall(["ZH-CN_NOT_A_PANEL_UTTERANCE"])


# ---- runner semantics on a bf16 tiny model --------------------------------------------------------

def _bf16_bundle(seed=0):
    b = tiny_bundle(seed)
    b.model.to(torch.bfloat16)
    return b


def _baseline_tokens(bundle, monkeypatch):
    monkeypatch.setattr(cached, "native_lid", lambda b, w, ids: {4: .9, 5: .1})
    dec = cached.cached_decode(bundle, waveform=np.ones(FRAMES * 320, dtype=np.float32), encoded=_enc(),
                               conditions=COND, partition=PART, null_probs={4: .5, 5: .5}, language_ids=(4, 5),
                               layer=16, alpha=0.0, max_new_tokens=6, num_forced_prefix=4)
    return dec["tokens"]


def _enc():
    e = encoded()
    e.last_hidden_state = e.last_hidden_state.to(torch.bfloat16)
    return e


def _d1():
    v = np.random.default_rng(0).standard_normal(16)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    return UniqueDirection(v, {"status": "ok", "vector_sha256": U.array_hash(v), "dialogue": "Dx"})


def _run(monkeypatch, ts=(1, 2), seed=0):
    bundle = _bf16_bundle(seed)
    toks = _baseline_tokens(bundle, monkeypatch)
    ctx = run.Ctx(bundle, PART, 4, E_TINY, cB=CB, cE=CE)
    with torch.inference_mode():
        xrec, states = run.extract_utterance(ctx, encoded=_enc(), tokens=toks, ts=list(ts))
        recs, vecs, logits = run.exp1_utterance(ctx, encoded=_enc(), tokens=toks, ts=list(ts), d1=_d1(), extracted=states)
    return bundle, ctx, toks, xrec, states, recs, vecs, logits


def test_exp1_seals_three_directions_and_pulses_at_matched_energy(monkeypatch):
    bundle, ctx, toks, xrec, states, recs, vecs, logits = _run(monkeypatch)
    for t in ("1", "2"):
        r = recs[t]
        assert all(r["identity"][k] for k in ("extracted_hb_bitwise", "extracted_he_bitwise", "d2_logits_bitwise", "d2_site_bitwise"))
        assert r["restore_bitwise"]
        assert set(r["arms"]) == set(r["directions"]) == {"D0", "D1", "D2"}
        assert all(r["directions"][a]["unchanged_after_pulses"] for a in run.ARMS)
        es = []
        for a in run.ARMS:
            x = r["arms"][a]
            assert r["directions"][a]["status"] == "ok" and x["steered"] and x["solver"]["status"] == "ok"
            assert abs(x["edit_norm"] ** 2 / E_TINY ** 2 - 1) <= CFG["energy"]["max_relative_squared_error"]
            es.append(x["edit_norm"] ** 2)
            assert f"t{t}_post_{a}" in vecs and f"t{t}_{a}" in logits
        assert max(es) / min(es) - 1 <= CFG["energy"]["max_pairwise_relative_squared_error"]
        assert r["directions"]["D2"]["counters"]["autograd_calls"] == 1
        assert r["directions"]["D1"]["sha256"] == _d1().fold["vector_sha256"]
    assert_no = __import__("csasr.lss.sites", fromlist=["assert_no_site_hooks"]).assert_no_site_hooks
    assert_no(bundle)


def test_d0_arm_equals_historical_p2r_plus_d_pulse(monkeypatch):
    bundle, ctx, toks, xrec, states, recs, vecs, logits = _run(monkeypatch, seed=1)
    cfg = json.loads(json.dumps(P2R_CFG))
    cfg["conditions"] = COND
    pctx = p2r.Ctx(bundle, cfg, PART, {4: .5, 5: .5}, (4, 5), 4)
    jobs = {t: {"target_ids": None, "arms": {"plus_d": {"dir": "plus_d", "energy": E_TINY, "continue": False}}} for t in (1, 2)}
    with torch.inference_mode():
        hist = p2r.pulse_pass(pctx, encoded=_enc(), uid="u", tokens=toks, jobs=jobs)
    for t in (1, 2):
        h, n = hist[t], recs[str(t)]
        assert n["arms"]["D0"]["solver"]["s"] == h["arms"]["plus_d"]["solver"]["s"]          # bitwise solver regression
        assert n["arms"]["D0"]["edit_norm"] == h["arms"]["plus_d"]["edit_norm"]
        assert n["identity"]["cos_d_state"] == h["cos_d_state"]
        assert xrec[str(t)]["cos_d_state"] == h["cos_d_state"]
        assert n["arms"]["D0"]["argmax"] == h["arms"]["plus_d"]["summary"]["argmax"] or \
            run.unpack_bf16(logits[f"t{t}_D0"])[h["arms"]["plus_d"]["summary"]["argmax"]] == run.unpack_bf16(logits[f"t{t}_D0"]).max()


def test_invalid_direction_is_zero_edit_bitwise(monkeypatch):
    bundle = _bf16_bundle()
    toks = _baseline_tokens(bundle, monkeypatch)
    ctx = run.Ctx(bundle, PART, 4, E_TINY, cB=CB, cE=CE)
    bad = UniqueDirection(None, {"status": "invalid", "reason": "numerical_rank", "dialogue": "Dx"})
    with torch.inference_mode():
        _, st = run.extract_utterance(ctx, encoded=_enc(), tokens=toks, ts=[1])
        recs, vecs, logits = run.exp1_utterance(ctx, encoded=_enc(), tokens=toks, ts=[1], d1=bad, extracted=st)
    r = recs["1"]
    assert r["directions"]["D1"]["status"] == "invalid" and not r["arms"]["D1"]["steered"]
    assert r["arms"]["D1"]["edit_norm"] == 0.0 and "t1_post_D1" not in vecs and "t1_D1" not in vecs
    assert np.array_equal(logits["t1_D1"], logits["t1_none"])          # zero dose -> bitwise identity


def test_extraction_identity_and_query_index(monkeypatch):
    bundle, ctx, toks, xrec, states, recs, vecs, logits = _run(monkeypatch)
    for t in ("1", "2"):
        assert xrec[t]["argmax"] == xrec[t]["expected_token"]
        assert recs[t]["query"] == xrec[t]["query"] == len(CB) + int(t) - 1
        assert np.array_equal(vecs[f"t{t}_hb"], states[t][0].numpy())


def test_bf16_logit_storage_lossless():
    x = torch.randn(1000).to(torch.bfloat16).float()
    a = run.pack_bf16(x)
    assert a.dtype == np.int16 and torch.equal(run.unpack_bf16(a), x)
    import experiments.inference_cf_p2dir_analyze as an
    assert np.array_equal(an.unpack_bf16(a), x.numpy())
    with pytest.raises(ValueError):
        run.pack_bf16(torch.tensor([1.0 + 2 ** -12]))


def test_manifest_rejection(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"schema": run.SCHEMA, "stage": "exp1", "manifest_hash": "sha256:0",
                                                   "sources": {}, "git_commit": "x"}))
    with pytest.raises(ValueError):
        run.verify_manifest(out, "exp1")
    m = {"schema": run.SCHEMA, "stage": "extract", "sources": {}, "git_commit": "x"}
    m["manifest_hash"] = digest(m)
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        run.verify_manifest(out, "exp1")                     # wrong stage
    m = {"schema": run.SCHEMA, "stage": "exp1", "sources": {prep.CONFIG: "sha256:bad"}, "git_commit": "x"}
    m["manifest_hash"] = digest(m)
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="source changed"):
        run.verify_manifest(out, "exp1")


def test_exp1_runner_never_reads_references():
    import ast
    src = Path("experiments/inference_cf_p2dir.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("extract_utterance", "exp1_utterance"):
            seg = ast.get_source_segment(src, node)
            for f in ("target_ids", "competitor", "positions.json", "p2rj"):
                assert f not in seg
    # module-level imports exclude the evaluator Jacobian module (imported lazily in eval only)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("p2rj" in (getattr(n, "module", "") or "") or any("p2rj" in a.name for a in n.names) for n in top)
