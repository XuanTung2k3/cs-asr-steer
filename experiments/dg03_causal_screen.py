#!/usr/bin/env python
"""DG-03 causal/specificity screen at one layer (free decoding, D-dev-select).

Conditions (spec §6): C0 baseline, C1 v_local @ oracle embedded steps, C2 -v_local,
C3 matched-norm random (seeds 0,1,2), C4 v_local @ matrix-continuation steps.
Oracle steps are an oracle diagnostic / non-deployable gate (NOT the final method).
Scores every condition against C0 with DG-01 canonical metrics (paired_corpus_report)
and reports net correction-damage utility U = net_corrections. See
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


def _matrix_sets(oracle: dict[str, list[int]]) -> dict[str, set[int]]:
    """Matrix-continuation steps: s+1 for each oracle embedded step s (spec §6 C4)."""
    out = {}
    for uid, steps in oracle.items():
        s = set(steps)
        out[uid] = {j + 1 for j in steps if (j + 1) not in s}
    return out


def _decode(bundle, pop, *, layer, direction, sets, nfp, alpha, scale,
            batch_size=8, record_energy=False):
    """Batched free decoding; steer `direction` where gen_step in sets[uid]. direction=None -> C0."""
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
    mean_energy = float(np.mean(energies)) if energies else 0.0
    return texts, mean_energy


def _score(refs, base_texts, method_texts, ids):
    from csasr.evaluation.canonical import paired_corpus_report
    r = [refs[i] for i in ids]
    b = [base_texts[i] for i in ids]
    m = [method_texts[i] for i in ids]
    rep = paired_corpus_report(r, b, m)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, choices=(16, 24))
    ap.add_argument("--basis-dir", default="results/dg03/basis")
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    layer = int(args.layer)
    out_path = Path(args.output or f"results/dg03/screen/dg03_screen_L{layer}.json")

    def log(m): print(m, flush=True)

    from csasr.utils.config import load_config
    from csasr.models.whisper import load_whisper
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.directions.controls import random_direction, wrong_sign
    from steer_sweep import data as D
    from experiments.job_a_frozen import oracle_steps_for

    # basis artifact for this layer
    rec = json.loads((Path(args.basis_dir) / f"steering_basis_v1_L{layer}.json").read_text())
    v_local_np = np.load(Path(args.basis_dir) / rec["tensor_files"]["conditioning_residualized_local"])
    s_l = float(rec["provenance"]["mean_site_norm_s_l"])
    dose = rec["provenance"]["diagnostic_dose"]
    alpha = float(dose["alpha"])
    log(f"L{layer}: s_l={s_l:.3f} alpha={alpha} nominal_update={dose['nominal_update_norm']:.3f}")

    cfg = load_config("model/whisper_large_v3.yaml") if Path(
        "configs/model/whisper_large_v3.yaml").exists() else load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    bundle = load_whisper(cfg if "model" in cfg else load_config("lss/l1b_candidates_dialogue_v2r3.yaml"))
    bundle.model.eval()
    v_local = torch.tensor(v_local_np, dtype=torch.float32)

    dcfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    pop = D.build_population(bundle, dcfg, "D-dev-select", assert_anchors=True)
    if args.limit:
        pop = pop.subsample(args.limit)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    oracle = oracle_steps_for(pop)
    matrix = _matrix_sets(oracle)
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    log(f"D-dev-select screen: {len(ids)} utts, {sum(len(v) for v in oracle.values())} oracle steps, nfp={nfp}")

    # C0 baseline
    base, _ = _decode(bundle, pop, layer=layer, direction=None, sets=None, nfp=nfp,
                      alpha=alpha, scale=s_l, batch_size=args.batch_size)
    conditions = {}

    def run(name, direction, sets):
        texts, energy = _decode(bundle, pop, layer=layer, direction=direction, sets=sets,
                                nfp=nfp, alpha=alpha, scale=s_l,
                                batch_size=args.batch_size, record_energy=True)
        rep = _score(refs, base, texts, ids)
        rep["realized_edit_energy_mean"] = energy
        t = rep["transitions"]
        conditions[name] = rep
        log(f"  {name}: U={t['net_corrections']} corr={t['corrections']} corrupt={t['corruptions']} "
            f"pier_gain={rep['pier_gain']} zh_cer={rep['method']['zh_cer']} energy={energy:.3f}")
        return rep

    run("C1_local", v_local, oracle)
    run("C2_sign", wrong_sign(v_local), oracle)
    c3 = []
    for sd in RANDOM_SEEDS:
        c3.append(run(f"C3_random_seed{sd}", random_direction(v_local.numel(), sd), oracle))
    run("C4_wrongloc", v_local, matrix)

    # aggregate C3
    c3_U = [r["transitions"]["net_corrections"] for r in c3]
    c3_pier = [r["pier_gain"] for r in c3 if r["pier_gain"] is not None]
    result = {
        "layer": layer, "git_commit": _git_commit(), "dataset_role": "D-dev-select",
        "n_utterances": len(ids), "num_forced_prefix": nfp,
        "diagnostic_dose": dose, "basis_artifact": rec["tensor_hashes"],
        "baseline_metrics": conditions["C1_local"]["baseline"],
        "conditions": conditions,
        "C3_random_aggregate": {"U_mean": float(np.mean(c3_U)),
                                "U_per_seed": c3_U,
                                "pier_gain_mean": float(np.mean(c3_pier)) if c3_pier else None},
        "model": bundle.metadata(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
