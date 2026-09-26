#!/usr/bin/env python
"""Independent CPU audit of immutable P0-R1 (K=3) row shards -> summary.json.

Reapplies the frozen G3 feasibility logic from the P0 spec, unchanged, to the K=3
token-average sequence support. G1/G2 are re-checked on the reused panel for continuity.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from csasr.inference_cf.core_r1 import atomic_json, digest, file_hash, sequence_logprob, support_k3


def describe(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0}
    xs.sort()
    return {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs),
            "min": xs[0], "max": xs[-1], "q25": xs[(len(xs)-1)//4],
            "q75": xs[(3*(len(xs)-1))//4]}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + .5*(a == b) for a in pos for b in neg)/(len(pos)*len(neg))


def aggregate(out):
    manifest = json.loads((out / "manifest.json").read_text())
    panel = json.loads((out / "panel.json").read_text())
    perm = json.loads((out / "audio_permutation.json").read_text())
    assert digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) == manifest["manifest_hash"]
    assert digest(panel) == manifest["panel_hash"] and digest(perm) == manifest["permutation_hash"]
    assert len(panel["rows"]) == len(perm) and all(k != v for k, v in perm.items())
    for p, h in manifest["sources"].items():
        assert file_hash(p) == h, p
    shards = sorted((out / "rows").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in shards]
    expected = {r["identity"] for r in panel["rows"]}
    assert len(rows) == len(expected) and {r["identity"] for r in rows} == expected
    assert all(r["manifest_hash"] == manifest["manifest_hash"] for r in rows)
    panel_by_id = {r["identity"]: r for r in panel["rows"]}
    assert set(perm) == expected and set(perm.values()) == expected
    for r in rows:
        assert r["shuffled_identity"] == perm[r["identity"]]
        assert r["audio_sha256"] == panel_by_id[r["identity"]]["audio_sha256"]
        assert r["shuffled_audio_sha256"] == panel_by_id[perm[r["identity"]]]["audio_sha256"]
        if r["status"] != "ok":
            assert r.get("reason")
            continue
        y = r["candidates"]
        yE, yM = y["yE"], y["yM"]
        # support recompute from saved token-average scores
        assert support_k3(r["real_scores"], yE, yM) == r["real_support"]
        assert support_k3(r["shuffled_scores"], yE, yM) == r["shuffled_support"]
        # token-average recompute from per-step logprobs
        for key in ("sE_yE", "sE_yM", "sM_yE", "sM_yM"):
            assert math.isclose(sequence_logprob(r["real_step_logprobs"][key]), r["real_scores"][key], abs_tol=1e-9)
            assert math.isclose(sequence_logprob(r["shuffled_step_logprobs"][key]), r["shuffled_scores"][key], abs_tol=1e-9)
        # candidate length / K / EOS-free
        assert 1 <= len(yE) <= y["K"] and 1 <= len(yM) <= y["K"]
        assert not y["yE_has_special"] and not y["yM_has_special"]
        assert y["full_sequence_collision"] == (list(yE) == list(yM))
        # conditions identity
        assert r["conditions"]["c0"] == r["conditions"]["cM"]
        assert r["conditions"] == manifest["conditions"]["prompt_tokens"]
        # scoring alignment across all continuation positions
        for name, cand in (("sE_yE", yE), ("sE_yM", yM), ("sM_yE", yE), ("sM_yM", yM)):
            a = r["real_scoring_alignment"][name]
            cond = "cE" if name.startswith("sE") else "cM"
            base = len(r["conditions"][cond]) + len(r["shared_content_prefix"])
            assert a["prompt_length"] == len(r["conditions"][cond])
            assert a["content_prefix_length"] == len(r["shared_content_prefix"])
            assert a["candidate_length"] == len(cand)
            assert a["continuation_query_indices"] == [base + j - 1 for j in range(len(cand))]
            assert a["decoder_mask"] == [1]*a["input_length"]
            assert a["cache_positions_full_replay"] == list(range(a["input_length"]))
            assert not a["cache_used"] and a["beam_lineage"] == "greedy:0"
        # geometry recompute (G1/G2 continuity)
        assert all(len(r["states"][c]) == len(r["states"]["c0"]) for c in ("cM", "cE"))
        assert r["geometry"]["finite"]
        h0, hm, he = (np.asarray(r["states"][c], dtype=np.float64) for c in ("c0", "cM", "cE"))
        delta = he-hm
        dn = np.linalg.norm(delta)
        d = delta/(dn+1e-6)
        derived = {"h0_norm": np.linalg.norm(h0), "hM_norm": np.linalg.norm(hm),
                   "hE_norm": np.linalg.norm(he), "delta_norm": dn,
                   "direction_norm": np.linalg.norm(d),
                   "c0_cM_residual_norm": np.linalg.norm(h0-hm),
                   "rho": 2*np.dot(h0-(hm+he)/2, d)/(dn+1e-6)}
        for key, value in derived.items():
            assert math.isclose(float(value), r["geometry"][key], abs_tol=1e-6), (r["identity"], key)
        assert r["logical_position"] == len(r["shared_content_prefix"])
    kept = [r for r in rows if r["status"] == "ok"]
    by = {s: [r for r in kept if r["stratum"] == s] for s in ("english_wrong", "english_correct", "mandarin")}
    coll = [r for r in kept if r["real_support"]["collision"]]
    usable = [r for r in kept if not r["real_support"]["collision"]]
    wrong = [r["real_support"]["q_cross"] for r in by["english_wrong"] if not r["real_support"]["collision"]]
    matrix = [r["real_support"]["q_cross"] for r in by["mandarin"] if not r["real_support"]["collision"]]
    auc = auroc(wrong, matrix) if len(wrong) >= 10 and len(matrix) >= 10 else None
    audio_abs = [abs(r["real_support"]["q_cross"]-r["shuffled_support"]["q_cross"]) for r in usable]
    audio_signed = [r["real_support"]["q_cross"]-r["shuffled_support"]["q_cross"] for r in usable]
    residuals = [r["geometry"]["c0_cM_residual_norm"] for r in kept]
    if kept and all(r["c0_equals_cM_by_tokens"] for r in kept) and max(residuals) <= .02:
        g1 = "DEGENERATE_BY_CONSTRUCTION"
    elif kept and all(not r["c0_equals_cM_by_tokens"] for r in kept) and min(residuals) > .02:
        g1 = "IDENTIFIABLE"
    else:
        g1 = "BROKEN"
    g2 = "PASS" if kept and all(r["real_scoring_alignment"] and r["continuation_alignment"] for r in kept) else "BLOCKED"
    coverage = len(kept)/len(rows) if rows else 0
    collision_rate = len(coll)/len(kept) if kept else 1
    # FROZEN G3 logic from P0 spec, unchanged
    g3 = "SUPPORTED" if (coverage >= .9 and collision_rate <= .5 and auc is not None and auc >= .60
                         and len(audio_abs) >= 10 and st.mean(audio_abs) >= .05) else "WEAK/BLOCKED"
    runtime = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    summary = {"schema": manifest["schema"], "manifest_hash": manifest["manifest_hash"],
               "supersedes_manifest": manifest.get("supersedes_manifest"),
               "candidate_k": manifest["candidate_k"], "n_panel": len(rows),
               "statuses": dict(Counter(r["status"] for r in rows)),
               "skip_reasons": dict(Counter(r.get("reason") for r in rows if r["status"] != "ok")),
               "finite_coverage": coverage, "candidate_collision_rate": collision_rate,
               "full_sequence_collisions": len(coll), "usable_rows": len(usable),
               "lcp_distribution": describe([r["candidates"]["longest_common_prefix"] for r in kept]),
               "candidate_length": {"yE": describe([r["candidates"]["yE_length"] for r in kept]),
                                    "yM": describe([r["candidates"]["yM_length"] for r in kept])},
               "strata": {s: {"n": len(v), "usable": sum(not r["real_support"]["collision"] for r in v),
                              "collisions": sum(r["real_support"]["collision"] for r in v),
                              "q_cross": describe([r["real_support"]["q_cross"] for r in v]),
                              "q_own": describe([r["real_support"]["q_own"] for r in v])} for s, v in by.items()},
               "auroc_wrong_english_vs_mandarin": auc,
               "audio_control_abs_q_cross_change": describe(audio_abs),
               "audio_control_signed_q_cross_change": describe(audio_signed),
               "geometry": {k: describe([r["geometry"][k] for r in kept]) for k in
                            ("h0_norm", "hM_norm", "hE_norm", "delta_norm", "direction_norm",
                             "c0_cM_residual_norm", "rho", "collapse_factor_diagnostic")},
               "representative": {s: [r["identity"] for r in sorted(v, key=lambda x: x["identity"])[:3]] for s, v in by.items()},
               "runtime": runtime, "processed_audio_sec": sum(r["duration_sec"] for r in rows),
               "output_tokens": sum(r.get("output_tokens", 0) for r in rows),
               "model_evaluations": {"encoder": 2*len(rows), "baseline_generate": len(rows),
                                     "real_decoder_forwards": sum(r.get("model_evaluations", {}).get("real_decoder_forwards", 0) for r in kept),
                                     "shuffled_decoder_forwards": sum(r.get("model_evaluations", {}).get("shuffled_decoder_forwards", 0) for r in kept)},
               "G1": g1, "G2": g2, "G3": g3}
    atomic_json(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", default="results/inference_cf/p0_r1")
    a = p.parse_args()
    print(json.dumps(aggregate(ROOT / a.out), indent=2))
