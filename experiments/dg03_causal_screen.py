#!/usr/bin/env python
"""DG-03 causal/specificity screen at one layer (free decoding, D-dev-select).

Conditions (spec §6): C0 baseline, C1 v_local @ oracle embedded steps, C2 -v_local,
C3 matched-norm random (seeds 0,1,2), C4 v_local @ count-matched matrix-continuation
steps. Oracle steps are an oracle diagnostic / non-deployable gate (NOT the final
method). Every condition is scored against C0 with the DG-01 canonical surface and
emitted as a complete result_v1 (MER/PIER/embedded-WER/matrix-CER + gains + POI
transitions + the three retention populations + canonical outside-harm accounting
and reserved gate_coverage). Net correction-damage utility U = net_corrections drives
selection (spec §8).

Edit audit (spec §5): per condition the number of steered edits, total and mean
realized edit energy, and the intended edit count are recorded; C4 is count-matched
to C1 so the location control has a matched edit budget. See
docs/current/DG03_BASIS_CAUSAL_SPEC.md.
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
RANDOM_SEEDS = (0, 1, 2)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return None


def _matched_matrix_sets(oracle: dict[str, list[int]]) -> dict[str, set[int]]:
    """Count-matched matrix-continuation steps (spec §6 C4).

    For each oracle embedded step s take the first matrix continuation step
    (s+1, s+2, ...) that is not itself an oracle step, so |matrix set| == |oracle|
    per utterance (matched edit budget). Records nothing here; the runner reports
    intended vs realized counts.
    """
    out = {}
    for uid, steps in oracle.items():
        oset = set(steps)
        used: set[int] = set()
        for s in steps:
            cand = s + 1
            while cand in oset or cand in used:
                cand += 1
            used.add(cand)
        out[uid] = used
    return out


def _decode(bundle, pop, *, layer, direction, sets, nfp, alpha, scale,
            batch_size=8, record_energy=False):
    """Batched free decoding; steer `direction` where gen_step in sets[uid]. direction=None -> C0.

    Returns (texts, energy_stats) where energy_stats has n_steered, total_energy,
    mean_energy (realized ‖r̃−r‖ over steered positions).
    """
    from csasr.models.whisper import batch_model_inputs
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook

    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts: dict[str, str] = {}
    energies: list[float] = []
    n_batches = math.ceil(len(manifest) / batch_size) if len(manifest) else 0
    for bi in range(n_batches):
        batch = manifest.iloc[bi * batch_size:(bi + 1) * batch_size]
        uids = [str(u) for u in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())
        if direction is None:
            with torch.inference_mode():
                out = bundle.model.generate(**inputs, **GEN)
        else:
            row_sets = [set(sets.get(u, set())) for u in uids]

            def gate_fn(*, q, u_source, r, abs_pos):
                B, T = r.shape[0], r.shape[1]
                pos = [int(x) for x in abs_pos.detach().reshape(-1).tolist()][:T]
                g = torch.zeros(B, T, dtype=r.dtype, device=r.device)
                for t, ap in enumerate(pos):
                    j = ap - nfp
                    if j < 0:
                        continue
                    for b in range(B):
                        if j in row_sets[b]:
                            g[b, t] = 1.0
                return g

            hook = DecoderPostCrossAttnInterventionHook(
                bundle, layer, direction, alpha=alpha, scale=scale,
                num_forced_prefix=nfp, gate_fn=gate_fn, norm_preserve=True,
                mode="steer", record=record_energy, record_last_only=False,
                enforce_contract_layer=True)
            with hook, torch.inference_mode():
                out = bundle.model.generate(**inputs, **GEN)
            if record_energy:
                energies += [rec.edit_norm for rec in hook.records if rec.steered]
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for u, t in zip(uids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[u] = t.strip()
    stats = {"n_steered": len(energies),
             "total_energy": float(np.sum(energies)) if energies else 0.0,
             "mean_energy": float(np.mean(energies)) if energies else 0.0}
    return texts, stats


def _result_v1(refs, base, method_texts, ids, *, layer, cond, provenance,
               direction_type, steering_strength, basis_id):
    """Build a complete, validated result_v1 record for one condition vs C0."""
    from csasr.evaluation import canonical
    from csasr.evaluation import retention as ret
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
    res = CanonicalResult(
        run_id=make_run_id(f"dg03_screen/{cond}/L{layer}"),
        system_name=f"dg03_{cond}_L{layer}",
        model_id=provenance.get("model_id"),
        model_revision=provenance.get("model_revision"),
        data_role="D-dev-select", decode_regime="greedy", beam=1,
        method=MethodConfig(layer=layer, direction_artifact_id=basis_id,
                            direction_type=direction_type, gate_type="oracle_diagnostic",
                            steering_strength=steering_strength),
        metrics=metrics, provenance=provenance)
    d = res.to_dict()
    validate(d)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, choices=(16, 24))
    ap.add_argument("--basis-dir", default="results/dg03/basis")
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    layer = int(args.layer)
    out_path = Path(args.output or str(REPO / f"results/dg03/screen/dg03_screen_L{layer}.json"))

    def log(m): print(m, flush=True)

    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.directions.controls import random_direction, wrong_sign
    from steer_sweep import data as D
    from experiments.job_a_frozen import oracle_steps_for

    rec = json.loads((Path(args.basis_dir) / f"steering_basis_v1_L{layer}.json").read_text())
    v_local_np = np.load(Path(args.basis_dir) / rec["tensor_files"]["conditioning_residualized_local"])
    s_l = float(rec["provenance"]["mean_site_norm_s_l"])
    dose = rec["provenance"]["diagnostic_dose"]
    alpha = float(dose["alpha"])
    basis_id = rec["tensor_hashes"]["conditioning_residualized_local"]
    log(f"L{layer}: s_l={s_l:.3f} alpha={alpha} nominal_update={dose['nominal_update_norm']:.3f}")

    mcfg_path = "configs/model/whisper_large_v3.yaml"
    mcfg = load_config("model/whisper_large_v3.yaml") if Path(mcfg_path).exists() else None
    dcfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    bundle = load_whisper(mcfg if mcfg and "model" in mcfg else dcfg)
    bundle.model.eval()
    v_local = torch.tensor(v_local_np, dtype=torch.float32)

    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    if args.limit:
        pop = pop.subsample(args.limit)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    oracle = oracle_steps_for(pop)
    matrix = _matched_matrix_sets(oracle)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    intended_oracle = sum(len(v) for v in oracle.values())
    intended_matrix = sum(len(v) for v in matrix.values())
    log(f"D-dev-select screen: {len(ids)} utts, intended oracle edits={intended_oracle}, "
        f"matched matrix edits={intended_matrix}, nfp={nfp}")

    provenance = {
        "git_commit": _git_commit(), "model_id": bundle.model_id,
        "model_revision": bundle.revision, "dataset_role": "D-dev-select",
        "basis_artifact_hashes": rec["tensor_hashes"],
        "basis_dataset_fingerprint": rec["provenance"].get("dataset_fingerprint"),
        "basis_construction_config_hash": rec["provenance"].get("construction_config_hash"),
        "diagnostic_dose": dose, "oracle_diagnostic": True, "non_deployable": True,
        "outside_harm_note": ("canonical n_corrupted_outside is reconstructed from the "
                              "frozen D-dev-select existing_ctc candidate population by "
                              "the CPU-only dg03_repair_outside_harm post-emission step; "
                              "it is correctness-flip harm, not transcript edit count."),
    }

    base, _ = _decode(bundle, pop, layer=layer, direction=None, sets=None, nfp=nfp,
                      alpha=alpha, scale=s_l, batch_size=args.batch_size)
    base_r1 = _result_v1(refs, base, base, ids, layer=layer, cond="C0_baseline",
                         provenance=provenance, direction_type=None,
                         steering_strength=0.0, basis_id=basis_id)

    conditions = {}
    texts_store = {"C0_baseline": {i: base[i] for i in ids}}

    def run(name, direction, sets, direction_type, intended):
        texts, estats = _decode(bundle, pop, layer=layer, direction=direction, sets=sets,
                                nfp=nfp, alpha=alpha, scale=s_l,
                                batch_size=args.batch_size, record_energy=True)
        r1 = _result_v1(refs, base, texts, ids, layer=layer, cond=name,
                        provenance=provenance, direction_type=direction_type,
                        steering_strength=alpha, basis_id=basis_id)
        t = r1["metrics"]["transitions"]
        estats["intended_edits"] = int(intended)
        conditions[name] = {"result_v1": r1, "edit_audit": estats}
        texts_store[name] = {i: texts[i] for i in ids}
        log(f"  {name}: U={t['net_corrections']} corr={t['corrections']} corrupt={t['corruptions']} "
            f"pier_gain={r1['metrics'].get('pier_gain')} zh_cer={r1['metrics']['zh_cer']:.4f} "
            f"n_steered={estats['n_steered']} total_energy={estats['total_energy']:.1f}")
        return r1

    run("C1_local", v_local, oracle, "conditioning_residualized_local", intended_oracle)
    run("C2_sign", wrong_sign(v_local), oracle, "conditioning_residualized_local_sign_reversed", intended_oracle)
    c3 = [run(f"C3_random_seed{sd}", random_direction(v_local.numel(), sd), oracle,
              "matched_norm_random", intended_oracle) for sd in RANDOM_SEEDS]
    run("C4_wrongloc", v_local, matrix, "conditioning_residualized_local_wrong_location", intended_matrix)

    def _U(r1): return r1["metrics"]["transitions"]["net_corrections"]
    def _pg(r1): return r1["metrics"].get("pier_gain")
    c3_U = [_U(r) for r in c3]
    c3_pg = [_pg(r) for r in c3 if _pg(r) is not None]
    result = {
        "layer": layer, "git_commit": provenance["git_commit"], "dataset_role": "D-dev-select",
        "n_utterances": len(ids), "num_forced_prefix": nfp, "diagnostic_dose": dose,
        "basis_artifact": rec["tensor_hashes"],
        "basis_dataset_fingerprint": rec["provenance"].get("dataset_fingerprint"),
        "baseline_result_v1": base_r1,
        "conditions": conditions,
        "C3_random_aggregate": {"U_mean": float(np.mean(c3_U)), "U_per_seed": c3_U,
                                "pier_gain_mean": float(np.mean(c3_pg)) if c3_pg else None},
        "selection_inputs": {
            "U_C1": _U(conditions["C1_local"]["result_v1"]),
            "U_C2": _U(conditions["C2_sign"]["result_v1"]),
            "U_C3_mean": float(np.mean(c3_U)),
            "U_C4": _U(conditions["C4_wrongloc"]["result_v1"]),
            "zh_cer_C0": base_r1["metrics"]["zh_cer"],
            "zh_cer_C1": conditions["C1_local"]["result_v1"]["metrics"]["zh_cer"],
        },
        "texts": texts_store,
        "model": bundle.metadata(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    # Candidate accounting is deterministic CPU post-processing over the stored
    # hypotheses; it does not change any decoded output or selection input.
    from experiments.dg03_repair_outside_harm import repair as repair_outside_harm
    data_root = Path(dcfg["experiment"]["output_root"]).parents[1]
    repair_outside_harm(out_path, data_root)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
