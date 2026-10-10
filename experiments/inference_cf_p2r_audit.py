#!/usr/bin/env python
"""Independent P2-R post-run audit (CPU). Does not import the P2-R analysis module.

Recomputes the primary contrasts, their Bonferroni dialogue-bootstrap intervals and the D2
energy balance from raw rows. It also checks completeness, pairing, energy matching,
random-direction geometry, provenance and data roles, and that nothing changed after the freeze.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np


def canon(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()


def git_blob_hash(commit: str, rel: str) -> str:
    blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def interval(pairs, reps, seed, alpha):
    groups = defaultdict(list)
    for g, v in pairs:
        groups[g].append(v)
    names = sorted(groups)
    m = np.array([sum(groups[n]) / len(groups[n]) for n in names])
    draws = m[np.random.default_rng(seed).integers(0, len(names), size=(reps, len(names)))].mean(axis=1)
    return float(m.mean()), [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    issues, checks = [], {}
    checks["manifest_hash"] = canon({k: v for k, v in man.items() if k != "manifest_hash"}) == man["manifest_hash"]
    src = {}
    for p, h in man["sources"].items():
        rel = str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p
        src[rel] = git_blob_hash(man["git_commit"], rel) == h
    checks["sources_match_manifest_commit"] = all(src.values())
    # post-freeze changes: which frozen sources differ between the manifest commit and the working tree?
    changed = [rel for rel in src if (ROOT / rel).exists() and
               "sha256:" + hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != man["sources"][str(ROOT / rel)]]
    checks["sources_changed_after_run"] = changed
    cfg = json.loads((ROOT / man["config"]).read_text())
    pop = json.loads((ROOT / man["population"]).read_text())
    checks["population_hash"] = canon({k: v for k, v in pop.items() if k != "population_hash"}) == man["population_hash"]
    checks["config_hash"] = canon(cfg) == man["config_hash"]
    rt = json.loads((run / "runtime.json").read_text())
    rows = {}
    for i, uid in enumerate(pop["utterances"]):
        r = json.loads((run / "rows" / f"{i:03d}.json").read_text())
        rows[uid] = r
    checks["rows"] = {"expected": len(pop["utterances"]), "ok": sum(r["status"] == "ok" for r in rows.values()),
                      "identity_ok": all(r["identity"] == u and r["manifest_hash"] == man["manifest_hash"]
                                         for u, r in rows.items()),
                      "runtime": rt["status"], "job": rt["job_id"]}
    # data roles
    import pyarrow.parquet as pq
    role = {r["utterance_id"]: r["role"] for r in pq.read_table(
        "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet",
        columns=["utterance_id", "role"]).to_pylist()}
    checks["all_D_dev_select"] = all(role.get(u) == "D-dev-select" for u in pop["utterances"])
    # D1 positions: presence, pairing, energy, random geometry
    e_star = cfg["energy"]["target_edit_norm"]
    tol = cfg["energy"]["max_pairwise_squared_energy_mismatch"]
    seen, pairs, energy_bad, rand_phi_bad = 0, defaultdict(list), 0, 0
    pm_phi_bad = 0
    for s, lst in pop["d1"].items():
        for c in lst:
            rec = rows[c["utterance_id"]]["pulses"].get(str(c["t"]))
            if rec is None:
                issues.append(f"missing D1 position {c['utterance_id']}:{c['t']}")
                continue
            seen += 1
            arms = rec["arms"]
            e2 = [arms[a]["edit_norm"] ** 2 for a in ("plus_d", "minus_d", "random")]
            if not all(abs(x / e_star ** 2 - 1) <= tol for x in e2) or max(e2) - min(e2) > tol * e_star ** 2:
                energy_bad += 1
            if abs(arms["random"]["solver"]["phi"] - math.pi / 2) > 1e-6:
                rand_phi_bad += 1
            if abs(arms["plus_d"]["solver"]["phi"] + arms["minus_d"]["solver"]["phi"] - math.pi) > 1e-6:
                pm_phi_bad += 1
            if s != "EN-confusion":
                continue
            n = rec["none"]["logp_ref"]
            L = {a: arms[a]["summary"]["logp_ref"] - n for a in arms if a in ("plus_d", "minus_d", "random")}
            P = {a: math.exp(arms[a]["summary"]["logp_ref"]) - math.exp(n) for a in L}
            lx = {a: (arms[a]["summary"]["logp_ref"] - math.log(arms[a]["summary"]["P_E_processed"]))
                  - (n - math.log(rec["none"]["P_E_processed"])) for a in L}
            dlg = c["dialogue_id"]
            pairs["D1.L_plus"].append((dlg, L["plus_d"]))
            pairs["D1.L_minus"].append((dlg, L["minus_d"]))
            pairs["D1.L_plus_minus"].append((dlg, L["plus_d"] - L["minus_d"]))
            pairs["D1.Lex_plus_minus"].append((dlg, lx["plus_d"] - lx["minus_d"]))
            pairs["D3.L_plus_random"].append((dlg, L["plus_d"] - L["random"]))
            pairs["D3.L_minus_random"].append((dlg, L["minus_d"] - L["random"]))
            pairs["D3.P_plus_random"].append((dlg, P["plus_d"] - P["random"]))
            pairs["D3.P_minus_random"].append((dlg, P["minus_d"] - P["random"]))
    checks["d1_positions_present"] = seen
    checks["d1_energy_unmatched"] = energy_bad
    checks["random_not_orthogonal"] = rand_phi_bad
    checks["plus_minus_not_opposite"] = pm_phi_bad
    # D2
    tk = {(c["utterance_id"], c["t"]): c["dialogue_id"] for c in pop["d2_targets"]}
    ec = eo = 0.0
    d2_seen = 0
    for uid, r in rows.items():
        ec += sum(s["edit_norm"] ** 2 for s in r["current"]["steps"])
        eo += sum(s["edit_norm"] ** 2 for s in r["oracle"]["steps"])
        cur = {s["t"]: s for s in r["current"]["steps"]}
        ora = {s["t"]: s for s in r["oracle"]["steps"]}
        for m in r["plan"]["moves"]:
            if ora[m["from"]]["edit"] or not ora[m["to"]]["edit"] or cur[m["to"]]["edit"]:
                issues.append(f"relocation not executed as planned {uid}:{m}")
        for t, s in cur.items():
            if (uid, t) in tk:
                d2_seen += 1
                c_, o_, n_ = s["arm"]["logp_ref"], ora[t]["arm"]["logp_ref"], s["none"]["logp_ref"]
                pairs["D2.L_oracle_current"].append((tk[(uid, t)], o_ - c_))
                pairs["D2.L_oracle_none"].append((tk[(uid, t)], o_ - n_))
                pairs["D2.P_oracle_current"].append((tk[(uid, t)], math.exp(o_) - math.exp(c_)))
    checks["d2_targets_present"] = [d2_seen, len(pop["d2_targets"])]
    checks["d2_energy"] = {"current": ec, "oracle": eo, "ratio": eo / ec}
    # recompute primary family
    bs = cfg["analysis"]["bootstrap"]
    fam = cfg["analysis"]["primary_family"]
    recompute, agree = {}, {}
    for k in fam:
        est, ci = interval(pairs[k], bs["replicates"], bs["seed"], 0.05 / len(fam))
        recompute[k] = {"estimate": est, "ci": ci}
        a = ana["primary"][k]
        agree[k] = abs(a["estimate"] - est) < 1e-9 and all(abs(x - y) < 1e-9 for x, y in zip(a["ci"], ci))
    checks["primary_recomputed_agrees"] = agree
    if not all(agree.values()):
        issues.append("primary recomputation disagrees")
    for k, v in checks.items():
        if v is False:
            issues.append(f"check failed: {k}")
    if checks["rows"]["ok"] != checks["rows"]["expected"] or not checks["rows"]["identity_ok"]:
        issues.append("incomplete rows")
    if energy_bad or rand_phi_bad or pm_phi_bad:
        issues.append("D1 energy/geometry")
    if abs(checks["d2_energy"]["ratio"] - 1) > cfg["d2"]["total_energy_tolerance"]:
        issues.append("D2 energy")
    allowed_post_run = {"experiments/inference_cf_p2r_analyze.py"}
    if set(changed) - allowed_post_run:
        issues.append(f"frozen sources changed after run: {sorted(set(changed) - allowed_post_run)}")
    out = {"schema": "p2r_audit_v1", "verdict": "P2_R_AUDIT: PASS" if not issues else "P2_R_AUDIT: BLOCK",
           "issues": issues, "checks": checks, "recomputed_primary": recompute}
    (ROOT / args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"verdict": out["verdict"], "issues": issues,
                      "changed_after_run": changed, "d2_energy": checks["d2_energy"],
                      "d1": [seen, energy_bad, rand_phi_bad, pm_phi_bad]}, indent=2))


if __name__ == "__main__":
    main()
