#!/usr/bin/env python
"""DG-04 frozen-steering baselines + correction-damage frontier (Whisper-large-v3 × CS-Dialogue, L24).

Systems (frozen DG-03 basis; canonical DG-02 exact-site hook; NormPreserve):
  B0  frozen baseline (no intervention)
  B1  global fixed local steering: d = v_local, all eligible non-prefix positions
  B2  global fixed local+conditioning mixture: d = normalize(V0 @ [0.5,0.5])
  B3  projection_gated_frozen: g_t = sigmoid((<LN(r_t), v_local> - tau)/T), d = v_local
Dose grid rho in {0.5,1.0,2.0} (nominal beta = rho * s_l), applied as alpha=rho, scale=s_l.
Every system×dose is a complete validated result_v1 with DG-01 canonical metrics + canonical
outside harm (reuses csasr.evaluation.dg03_outside_harm) + realized-energy audit. No adaptive
controller, no training, no beam-5. See docs/current/DG04_FROZEN_BASELINES_SPEC.md.

Modes:
  (default)          decode + score the requested --systems, write per-system result_v1 files
  --build-frontier   CPU-only: merge result files -> frontier.json + reference.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import torch

GEN = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
           temperature=0.0, max_new_tokens=200, condition_on_prev_tokens=False)
GRID = (0.5, 1.0, 2.0)
LAYER = 24
BASIS = "results/dg03/basis/steering_basis_v1_L24.json"


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def _data_root(dcfg) -> Path:
    return Path(dcfg["v2_namespace"]["role_root"]).parent.parent


def _b3_gate(v_local_t, tau, T):
    """Parameter-free LayerNorm projection gate g_t = sigmoid((<LN(r), v_local> - tau)/T)."""
    def gate_fn(*, q, u_source, r, abs_pos):
        mu = r.mean(dim=-1, keepdim=True)
        var = r.var(dim=-1, unbiased=False, keepdim=True)
        ln = (r - mu) / torch.sqrt(var + 1e-5)
        v = v_local_t.to(r.device, r.dtype)
        s = (ln * v).sum(dim=-1)                     # (B, T)
        return torch.sigmoid((s - tau) / T)
    return gate_fn


def _decode(bundle, pop, *, layer, direction, gate_fn, nfp, alpha, scale,
            batch_size=8, collect=False):
    """Batched free decoding. direction=None -> baseline. Returns (texts, audit)."""
    from csasr.models.whisper import batch_model_inputs
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts, energies, gates = {}, [], []
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0
    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size:(bi + 1) * batch_size]
        uids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        if direction is None:
            with torch.inference_mode():
                out = bundle.model.generate(**inputs, **GEN)
        else:
            hook = DecoderPostCrossAttnInterventionHook(
                bundle, layer, direction, alpha=alpha, scale=scale,
                num_forced_prefix=nfp, gate_fn=gate_fn, norm_preserve=True,
                mode="steer", record=collect, record_last_only=False,
                enforce_contract_layer=True)
            with hook, torch.inference_mode():
                out = bundle.model.generate(**inputs, **GEN)
            if collect:
                for rec in hook.records:
                    if rec.steered:
                        energies.append(rec.edit_norm)
                    if not rec.is_forced_prefix:
                        gates.append(rec.gate)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for u, t in zip(uids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[u] = t.strip()
    audit = {"n_steered": len(energies),
             "total_energy": float(np.sum(energies)) if energies else 0.0,
             "mean_energy": float(np.mean(energies)) if energies else 0.0,
             "n_gate_eligible": len(gates)}
    if gates:
        g = np.asarray(gates, dtype=float)
        audit["gate_strength"] = {
            "mean": float(g.mean()), "median": float(np.median(g)),
            "q10": float(np.quantile(g, 0.1)), "q90": float(np.quantile(g, 0.9)),
            "n_gt_0.5": int((g > 0.5).sum())}
    return texts, audit


def _build_outside_sets(dcfg, references: dict, ids: list):
    """Frozen D-dev-select inside/outside candidate partition (reuse DG-03 machinery)."""
    import pyarrow.parquet as pq
    from experiments.dg03_repair_outside_harm import _frozen_candidates
    root = _data_root(dcfg)
    role_path = root / "manifests/roles/role_D-dev-select.parquet"
    cand_path = root / "candidate_generations/generation_001/alignments/candidates_all.parquet"
    poi_path = root / "baselines/generation_001/poi_D-dev-select.parquet"
    cand = pq.read_table(cand_path, columns=[
        "utterance_id", "reference_unit_index", "reference_language",
        "aligner_family", "start_sec", "end_sec", "role"],
        filters=[("role", "==", "D-dev-select")]).to_pandas()
    cand = cand[cand["aligner_family"].astype(str) == "existing_ctc"]
    poi = pq.read_table(poi_path, columns=["utterance_id", "poi_index", "correct"],
                        filters=[("role", "==", "D-dev-select")]).to_pandas().rename(
        columns={"poi_index": "reference_unit_index"})
    refs_ordered = {uid: references[uid] for uid in ids}
    sets, languages, diag = _frozen_candidates(cand, poi, refs_ordered)
    return sets, languages, diag


def _result_v1(refs, base, method_texts, ids, *, cond, provenance, direction_type,
               steering_strength, basis_id, out_sets, out_langs, audit):
    from csasr.evaluation import canonical
    from csasr.evaluation import retention as ret
    from csasr.evaluation.dg03_outside_harm import corpus_outside_harm
    from csasr.evaluation.result_schema import CanonicalResult, MethodConfig, validate
    from csasr.lss.manifest import run_id as make_run_id

    r = [refs[i] for i in ids]
    b = [base[i] for i in ids]
    m = [method_texts[i] for i in ids]
    bm = canonical.corpus_metrics(r, b)
    mm = canonical.corpus_metrics(r, m)
    metrics = dict(mm)
    metrics.update(canonical.error_metric_gains(bm, mm))
    metrics["transitions"] = canonical.correction_corruption(r, b, m)
    metrics["retention"] = ret.retention_report(r, b, m)
    oh = corpus_outside_harm(r, b, m, out_sets, out_langs)
    metrics["outside_harm"] = int(oh["outside_harm"])
    metrics["outside_harm_accounting"] = oh
    metrics["candidate_utility"] = float(oh["utility"])
    metrics["realized_edit"] = audit
    res = CanonicalResult(
        run_id=make_run_id(f"dg04/{cond}"), system_name=f"dg04_{cond}",
        model_id=provenance.get("model_id"), model_revision=provenance.get("model_revision"),
        data_role="D-dev-select", decode_regime="greedy", beam=1,
        method=MethodConfig(layer=LAYER, direction_artifact_id=basis_id,
                            direction_type=direction_type, gate_type="frozen",
                            steering_strength=steering_strength),
        metrics=metrics, provenance=provenance)
    d = res.to_dict()
    validate(d)
    return d


def _calibrate_b3(bundle, layer, v_local_t, dcfg, nfp, *, n=300, log=print):
    """Freeze tau=median, T=std of s_t=<LN(r),v_local> on router-calib (n shortest, TF)."""
    import pyarrow.parquet as pq
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.models.generation import teacher_forced_forward
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    root = _data_root(dcfg)
    rc = pq.read_table(root / "manifests/roles/role_router-calib.parquet",
                       columns=["utterance_id", "audio_path", "transcript_raw", "duration_sec"],
                       filters=[("role", "==", "router-calib")]).to_pandas()
    rc = rc.sort_values("duration_sec").head(int(n)).reset_index(drop=True)
    tok = bundle.processor.tokenizer
    eot = tok.eos_token_id
    prefix = build_prefix(bundle.processor, language="zh")
    plen = len(prefix)
    v = v_local_t.float()
    proj = []
    for i in range(0, len(rc), 4):
        b = rc.iloc[i:i + 4]
        seqs, audios = [], []
        for _, row in b.iterrows():
            norm, _u = normalize_and_segment(row["transcript_raw"])
            tids = tok.encode(norm, add_special_tokens=False)[:220]
            if not tids:
                seqs.append(None); continue
            seqs.append(list(prefix) + list(tids) + [eot]); audios.append(row["audio_path"])
        keep = [(s, a) for s, a in zip(seqs, [r for r in b["audio_path"]]) if s is not None]
        if not keep:
            continue
        seqs2 = [s for s, _ in keep]; audios2 = [a for _, a in keep]
        with DecoderPostCrossAttnRecorder(bundle, [layer]) as recr, torch.inference_mode():
            teacher_forced_forward(bundle, audios2, seqs2)
            st = recr.states[layer].float().cpu()
        for bi2, seq in enumerate(seqs2):
            content = st[bi2, plen:len(seq) - 1]           # content positions (exclude prefix + eot)
            if content.shape[0] == 0:
                continue
            mu = content.mean(dim=-1, keepdim=True)
            var = content.var(dim=-1, unbiased=False, keepdim=True)
            ln = (content - mu) / torch.sqrt(var + 1e-5)
            proj.append((ln @ v).numpy())
        if (i // 4) % 20 == 0:
            log(f"  b3 calib {i}/{len(rc)}")
    s = np.concatenate(proj) if proj else np.zeros(1)
    tau = float(np.median(s)); T = float(np.std(s)) or 1.0
    return {"tau": tau, "T": T, "n_utterances": int(len(rc)), "n_positions": int(s.size),
            "s_mean": float(s.mean()), "s_std": float(np.std(s)),
            "role": "router-calib", "subset": "300 shortest by duration"}


# --------------------------------------------------------------------------
def run_systems(args):
    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from csasr.lss.sites import num_forced_prefix_from
    from steer_sweep import data as D

    def log(m): print(m, flush=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    out_dir = Path(args.output_dir); (out_dir / "results").mkdir(parents=True, exist_ok=True)

    rec = json.loads((REPO / BASIS).read_text())
    assert int(rec["scientific_layer"]) == LAYER
    d_local = np.load(REPO / "results/dg03/basis" / rec["tensor_files"]["conditioning_residualized_local"])
    d_cond = np.load(REPO / "results/dg03/basis" / rec["tensor_files"]["language_conditioning"])
    s_l = float(rec["provenance"]["mean_site_norm_s_l"])
    basis_id = rec["tensor_hashes"]["conditioning_residualized_local"]
    v_local = torch.tensor(d_local, dtype=torch.float32)
    mix = 0.5 * d_local + 0.5 * d_cond
    v_mix = torch.tensor(mix / np.linalg.norm(mix), dtype=torch.float32)

    dcfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    mcfg = load_config("model/whisper_large_v3.yaml") if Path("configs/model/whisper_large_v3.yaml").exists() else None
    bundle = load_whisper(mcfg if mcfg and "model" in mcfg else dcfg); bundle.model.eval()
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")

    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    if args.limit:
        pop = pop.subsample(args.limit)
    refs = D.reference_of(pop); ids = list(pop.utterance_ids)
    out_sets, out_langs, out_diag = _build_outside_sets(dcfg, refs, ids)
    log(f"D-dev-select: {len(ids)} utts; outside-harm population {out_diag}")

    provenance = {
        "git_commit": _git_commit(), "model_id": bundle.model_id, "model_revision": bundle.revision,
        "layer": LAYER, "basis_artifact_hashes": rec["tensor_hashes"],
        "basis_dataset_fingerprint": rec["provenance"]["dataset_fingerprint"],
        "basis_construction_config_hash": rec["provenance"]["construction_config_hash"],
        "scale_s_l": s_l, "grid_rho": list(GRID)}

    # baseline (needed to score every system)
    base, base_audit = _decode(bundle, pop, layer=LAYER, direction=None, gate_fn=None,
                               nfp=nfp, alpha=0.0, scale=s_l, batch_size=args.batch_size)
    b0 = _result_v1(refs, base, base, ids, cond="B0", provenance=provenance,
                    direction_type=None, steering_strength=0.0, basis_id=basis_id,
                    out_sets=out_sets, out_langs=out_langs, audit=base_audit)
    _write(out_dir / "results" / "B0.json", {"system": "B0", "rho": 0.0, "beta_nominal": 0.0,
           "result_v1": b0, "texts": {i: base[i] for i in ids}})
    log(f"B0: MER={b0['metrics']['mer']:.4f} PIER={b0['metrics']['pier']:.4f} "
        f"zh_cer={b0['metrics']['zh_cer']:.4f}")

    b3_calib = None
    if "B3" in systems:
        b3_calib = _calibrate_b3(bundle, LAYER, v_local, dcfg, nfp, log=log)
        _write(out_dir / "b3_calibration.json", {**b3_calib, "git_commit": provenance["git_commit"]})
        log(f"B3 calib: tau={b3_calib['tau']:.4f} T={b3_calib['T']:.4f} "
            f"(n_pos={b3_calib['n_positions']})")

    plans = []
    for sysname in systems:
        if sysname == "B1":
            plans += [("B1", rho, v_local, None, "local") for rho in GRID]
        elif sysname == "B2":
            plans += [("B2", rho, v_mix, None, "local_cond_mix_fixed") for rho in GRID]
        elif sysname == "B3":
            gate = _b3_gate(v_local, b3_calib["tau"], b3_calib["T"])
            plans += [("B3", rho, v_local, gate, "projection_gated_frozen") for rho in GRID]

    for sysname, rho, direction, gate_fn, dtype in plans:
        texts, audit = _decode(bundle, pop, layer=LAYER, direction=direction, gate_fn=gate_fn,
                               nfp=nfp, alpha=float(rho), scale=s_l, batch_size=args.batch_size,
                               collect=True)
        r1 = _result_v1(refs, base, texts, ids, cond=f"{sysname}_rho{rho}", provenance=provenance,
                        direction_type=dtype, steering_strength=float(rho), basis_id=basis_id,
                        out_sets=out_sets, out_langs=out_langs, audit=audit)
        t = r1["metrics"]["transitions"]
        _write(out_dir / "results" / f"{sysname}_rho{rho}.json",
               {"system": sysname, "rho": float(rho), "beta_nominal": float(rho) * s_l,
                "result_v1": r1, "texts": {i: texts[i] for i in ids}})
        log(f"{sysname} rho={rho}: U={t['net_corrections']} corr={t['corrections']} "
            f"corrupt={t['corruptions']} outside_harm={r1['metrics']['outside_harm']} "
            f"pier_gain={r1['metrics'].get('pier_gain')} zh_cer={r1['metrics']['zh_cer']:.4f} "
            f"n_steered={audit['n_steered']} energy={audit['total_energy']:.1f}")
    return 0


def build_frontier(args):
    out_dir = Path(args.output_dir); res_dir = out_dir / "results"
    files = sorted(res_dir.glob("*.json"))
    b0 = json.loads((res_dir / "B0.json").read_text())["result_v1"]
    zh_cer_b0 = b0["metrics"]["zh_cer"]
    points = []
    for f in files:
        e = json.loads(f.read_text()); r = e["result_v1"]; m = r["metrics"]; t = m["transitions"]
        points.append({
            "system": e["system"], "rho": e["rho"], "beta_nominal": e.get("beta_nominal"),
            "n_steered": m["realized_edit"]["n_steered"],
            "total_energy": m["realized_edit"]["total_energy"],
            "mean_energy": m["realized_edit"]["mean_energy"],
            "corrections": t["corrections"], "corruptions": t["corruptions"],
            "net_corrections": t["net_corrections"],
            "correction_rate": t["correction_rate"], "corruption_rate": t["corruption_rate"],
            "outside_harm": m["outside_harm"], "candidate_utility": m.get("candidate_utility"),
            "pier_gain": m.get("pier_gain"), "mer_gain": m.get("mer_gain"),
            "en_wer_gain": m.get("en_wer_gain"), "zh_cer": m["zh_cer"],
            "matrix_retention_ok": bool(m["zh_cer"] <= 1.5 * zh_cer_b0),
            "gate_strength": m["realized_edit"].get("gate_strength"),
        })
    # non-dominated on (corrections up, corruptions down) among non-zero points
    nz = [p for p in points if p["system"] != "B0"]
    for p in nz:
        p["non_dominated"] = not any(
            q is not p and q["corrections"] >= p["corrections"] and q["corruptions"] <= p["corruptions"]
            and (q["corrections"] > p["corrections"] or q["corruptions"] < p["corruptions"])
            for q in nz)
    # reference selection (spec §12)
    elig = [p for p in nz if p["net_corrections"] > 0 and p["corrections"] > p["corruptions"]
            and p["outside_harm"] is not None and p["matrix_retention_ok"]]
    reference = None
    if elig:
        elig.sort(key=lambda p: (-p["net_corrections"],
                                 -(p["pier_gain"] or -1e9), p["total_energy"]))
        reference = elig[0]
    front = {"git_commit": _git_commit(), "layer": LAYER, "baseline_zh_cer": zh_cer_b0,
             "eligibility_rule": "U>0 & corr>corrupt & outside_harm present & zh_cer<=1.5*zh_cer_B0",
             "selection_rule": "max net_corrections; tie PIER gain; then lower total_energy",
             "points": sorted(points, key=lambda p: (p["system"], p["rho"])),
             "n_eligible": len(elig)}
    _write(out_dir / "frontier.json", front)
    ref_payload = ({"result": "NO POSITIVE FROZEN OPERATING POINT"} if reference is None
                   else {"result": f"SELECT {reference['system']} rho={reference['rho']}",
                         "reference": reference})
    _write(out_dir / "reference.json", ref_payload)
    print(json.dumps(ref_payload, indent=2, default=str))
    return 0


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="B1,B2")
    ap.add_argument("--output-dir", default=str(REPO / "results/dg04"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--build-frontier", action="store_true")
    args = ap.parse_args(argv)
    return build_frontier(args) if args.build_frontier else run_systems(args)


if __name__ == "__main__":
    raise SystemExit(main())
