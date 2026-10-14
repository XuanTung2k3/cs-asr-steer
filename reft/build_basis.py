#!/usr/bin/env python
"""P1-B -- weighted gradient SVD, frozen rank-4 basis and controls (CPU, construction partition only).

Policy (frozen in P0P1_PROTOCOL.md section 5): basis rows = A if the construction partition has >= 100 A rows over >= 10
dialogues, else A + B. Unit-normalised q; per-utterance energy 1; EN_ONLY / ZH_ONLY energy capped at 20% each; SVD in
float64; V = top-4 right singular vectors, signs aligned with the weighted mean correction gradient.
Controls: R1..R5 random orthonormal (seeds 42..46), PCA4 of site activations, SCRIPT1 mean script-readout gradient.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, cfg, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import check_freeze, git_state, now, sha  # noqa: E402


def stratum_weights(strata: list[str], base_w, caps: dict):
    """Scale capped strata uniformly so each holds at most its cap of the total squared weight (solved jointly)."""
    import torch
    e = collections.Counter()
    for s, x in zip(strata, base_w.tolist()):
        e[s] += x * x
    active: set = set()
    while True:  # active set: strata held exactly at their cap; total T solves T = free + sum(cap_s) * T
        T = sum(v for s, v in e.items() if s not in active) / (1 - sum(caps[s] for s in active))
        new = {s for s in caps if s in e and s not in active and e[s] > caps[s] * T + 1e-12}
        if not new:
            break
        active |= new
    f = {s: math.sqrt(caps[s] * T / e[s]) if s in active else 1.0 for s in e}
    w = base_w * torch.tensor([f[s] for s in strata], dtype=base_w.dtype)
    tot = float((w ** 2).sum())
    shares = collections.Counter()
    for s, x in zip(strata, w.tolist()):
        shares[s] += x * x / tot
    return w, dict(shares)


def orth_random(d, k, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    Qm, Rm = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return Qm * torch.sign(torch.diagonal(Rm))  # unique QR (positive diagonal)


def effective_rank(sv):
    import torch
    p = sv ** 2 / (sv ** 2).sum()
    p = p[p > 0]
    return {"entropy_effective_rank": float(torch.exp(-(p * p.log()).sum())),
            "participation_ratio": float((sv ** 2).sum() ** 2 / (sv ** 4).sum())}


def principal_cosines(A, B):
    import torch
    return torch.linalg.svdvals(A.T @ B).tolist()


def select_rows(meta, P1):
    con_a = [m for m in meta if m["partition"] == "construction" and m["source"] == "A"]
    n_dlg = len({m["dialogue"] for m in con_a})
    pol = P1["basis_policy"]
    primary = len(con_a) >= pol["a_primary_min_examples"] and n_dlg >= pol["a_primary_min_dialogues"]
    srcs = ("A",) if primary else ("A", "B")
    rows = [m for m in meta if m["partition"] == "construction" and m["source"] in srcs]
    return rows, {"construction_A_rows": len(con_a), "construction_A_dialogues": n_dlg,
                  "policy": "A_only" if primary else "A_plus_B_fallback", "sources": list(srcs), "n_rows": len(rows)}


def main():
    import torch
    C = cfg()
    P1 = C["p1"]
    k = P1["rank"]
    fz = check_freeze("p1")
    meta = read_jsonl(art("p1/gradients/rows.jsonl"))
    gm = read_json(art("p1/gradient_manifest.json"))
    tp = art("p1/gradients/tensors.pt")
    assert sha(tp) == gm["tensors_sha256"], "gradient tensors changed"
    T = torch.load(tp)
    Q = T["q"].double()
    Hs = T["h"].double()
    G = T["gs"].double()
    d = Q.shape[1]
    rows, policy = select_rows(meta, P1)
    idx = torch.tensor([m["row"] for m in rows])
    Qr = Q[idx]
    Qn = Qr / Qr.norm(dim=1, keepdim=True)
    per_utt = collections.Counter(m["id"] for m in rows)
    w0 = torch.tensor([1 / math.sqrt(per_utt[m["id"]]) for m in rows], dtype=torch.float64)
    strata = [m["stratum"] for m in rows]
    w, shares = stratum_weights(strata, w0, P1["basis_policy"]["stratum_energy_caps"])
    WQ = w[:, None] * Qn
    U, sv, Vh = torch.linalg.svd(WQ, full_matrices=False)
    V = Vh[:k].T.clone()
    # sign: aligned with the weighted mean correction gradient; tie -> first nonzero coordinate positive
    for j in range(k):
        s = float((w * (Qn @ V[:, j])).sum())
        if abs(s) < 1e-12:
            s = float(V[V[:, j].abs().argmax(), j])
        if s < 0:
            V[:, j] = -V[:, j]
    # independent reconstruction: eigen-decomposition of the weighted Gram matrix
    Gm = WQ.T @ WQ
    ev, evec = torch.linalg.eigh(Gm)
    top = evec[:, ev.argsort(descending=True)[:k]]
    pc = principal_cosines(V, top)
    eig_top = ev.sort(descending=True).values[:k]
    checks = {"orthonormal_max_dev": float((V.T @ V - torch.eye(k, dtype=torch.float64)).abs().max()),
              "finite": bool(torch.isfinite(V).all()), "shape": list(V.shape),
              "eigh_principal_cosines": pc, "eig_vs_sv2_max_rel_dev": float(((eig_top - sv[:k] ** 2).abs() / sv[:k] ** 2).max())}
    checks["ok"] = (checks["orthonormal_max_dev"] < 1e-6 and checks["finite"] and checks["shape"] == [d, k]
                    and min(pc) > 1 - 1e-6 and checks["eig_vs_sv2_max_rel_dev"] < 1e-6)
    # spectrum and representation
    energy = (sv ** 2) / (sv ** 2).sum()
    rho_in = ((Qn @ V) ** 2).sum(1)
    per_stratum = {}
    for key in ("stratum", "category", "lang", "source"):
        grp = collections.defaultdict(list)
        for m, r in zip(rows, rho_in.tolist()):
            grp[m[key]].append(r)
        per_stratum[key] = {g: {"n": len(v), "mean_rho_in_sample": sum(v) / len(v)} for g, v in sorted(grp.items())}
    # raw-gradient diagnostic (unnormalised, unweighted)
    _, sv_raw, Vh_raw = torch.linalg.svd(Qr, full_matrices=False)
    raw = {"top_singular_values": sv_raw[:16].tolist(), "energy_top4": float((sv_raw[:k] ** 2).sum() / (sv_raw ** 2).sum()),
           "principal_cosines_with_V": principal_cosines(V, Vh_raw[:k].T)}
    # controls (built now, before any evaluation)
    rnd = {s: orth_random(d, k, s) for s in P1["random_seeds"]}
    Hc = Hs[idx] - Hs[idx].mean(0, keepdim=True)
    _, _, Vh_h = torch.linalg.svd(Hc, full_matrices=False)
    pca = Vh_h[:k].T.clone()
    con_all = torch.tensor([m["row"] for m in meta if m["partition"] == "construction"])
    Gn = G[con_all] / G[con_all].norm(dim=1, keepdim=True)
    dscript = Gn.mean(0)
    dscript = (dscript / dscript.norm())[:, None]
    cbar = (w[:, None] * (Qn @ V)).sum(0)
    cbar = cbar / cbar.norm()
    cbar_r = {}
    for s, R in rnd.items():
        c = (w[:, None] * (Qn @ R)).sum(0)
        cbar_r[s] = c / c.norm()
    basis_path, ctrl_path = art("p1/basis_rank4.pt"), art("p1/controls.pt")
    for p in (basis_path, ctrl_path):
        assert not p.exists(), p
    torch.save({"V": V, "singular_values": sv, "cbar": cbar, "layer": P1["layer"], "rank": k,
                "site": P1["site"]}, basis_path)
    torch.save({"random": rnd, "pca4": pca, "script1": dscript, "cbar_random": cbar_r}, ctrl_path)
    cos_script = (V.T @ dscript)[:, 0].tolist()
    spec = {"schema": "reft_p1_svd_spectrum_v1", "utc": now(), "top64_singular_values": sv[:64].tolist(),
            "total_energy": float((sv ** 2).sum()), "energy_fraction_top": [float(x) for x in energy[:16]],
            "energy_top4": float(energy[:k].sum()), **effective_rank(sv), "n_rows": len(rows),
            "in_sample_mean_rho": float(rho_in.mean()), "per_group_in_sample": per_stratum,
            "raw_gradient_diagnostic": raw}
    write_json(art("p1/svd_spectrum.json"), spec)
    man = {"schema": "reft_p1_basis_manifest_v1", "utc": now(), **fz, "git": git_state(), "policy": policy,
           "weights": {"normalization": "unit q", "per_utterance": "1/sqrt(n_rows)",
                       "stratum_caps": P1["basis_policy"]["stratum_energy_caps"], "energy_shares_after": shares},
           "basis_sha256": sha(basis_path), "controls_sha256": sha(ctrl_path), "checks": checks,
           "sign_convention": "sum_i w_i (q_hat_i . v_j) >= 0; order by descending singular value",
           "svd": "torch.linalg.svd float64 on W*Q_hat (rows = gradients), V = Vh[:4].T",
           "cos_V_with_script1": cos_script, "cos_cbar_V_with_script1": float(V @ cbar @ dscript[:, 0]),
           "source_gradients_sha256": gm["tensors_sha256"]}
    write_json(art("p1/basis_manifest.json"), man)
    write_json(art("p1/random_controls.json"), {
        "schema": "reft_p1_controls_v1", "random_seeds": P1["random_seeds"],
        "construction": "QR of N(0,1) 1280x4 float64 (torch.Generator(seed)), positive-diagonal R",
        "pca4": "top-4 principal components of centred site activations h of the basis rows",
        "script1": "normalize(mean unit script-readout gradient g_s over all construction rows)",
        "full": "unprojected q (oracle ceiling, not a control)", "controls_sha256": sha(ctrl_path),
        "random_vs_V_principal_cosines": {s: principal_cosines(V, R) for s, R in rnd.items()},
        "pca4_vs_V_principal_cosines": principal_cosines(V, pca)})
    print(json.dumps({"policy": policy, "checks": checks, "energy_top4": spec["energy_top4"],
                      "top8_sv": spec["top64_singular_values"][:8], "effective_rank": effective_rank(sv),
                      "cos_V_script1": cos_script, "basis_sha256": man["basis_sha256"]}, indent=1))
    if not checks["ok"]:
        raise SystemExit("basis checks FAILED")


if __name__ == "__main__":
    main()
