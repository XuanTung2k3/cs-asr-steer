#!/usr/bin/env python
"""Learned-expansion runner: layer- and basis-mode-configurable damage-aware training.

Additive, one-seed exploratory training over the frozen DG-02 exact site. It reuses
the frozen DG-05/DG-06/DG-04 stack verbatim (exact-site hook, correction set C_E,
matrix-retention set R_M, D1 damage-aware loss, free-decode selection, canonical
result_v1 + outside-harm scoring) and only generalises two axes:

* **basis mode** — raw_only / local_only / cond_only / raw_cond / local_cond
  (via ``csasr.steering.expansion``), mapping rank-2 → FixedBasisAdaptiveController
  (the M* controller) and rank-1 → GateOnlyController;
* **decoder layer** — any predeclared valid layer (L24 authorized now; L26/L27
  only after the frozen prerequisite gate passes).

It never modifies the frozen DG-05/06/07 runners or their artifacts. Two-site is
intentionally NOT runnable here (single-site only). No multi-seed.

Objective is the selected DG-06 **D1**: ``L = L_corr(C_E) + lambda_M L_ret,M``,
lambda_M = 1.0, beta = DG-04 reference, seed 42, 3 epochs, greedy D-dev-select
checkpoint selection. Forbidden roles: D-dev-confirm, D-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, num_forced_prefix_from
from csasr.models.whisper import batch_model_inputs, load_whisper
from csasr.steering.controller import FixedBasisAdaptiveController
from csasr.steering.dg05_training import load_correction_set
from csasr.steering.dg06_losses import damage_aware_loss, load_retention_set
from csasr.steering.expansion import (
    build_steering_module,
    load_basis_mode,
    load_basis_mode_from_atlas,
)
from csasr.steering.expansion.controller_factory import assert_basis_frozen
from csasr.utils.config import load_config
from csasr.utils.hashing import sha256_obj
from csasr.utils.provenance import stage_provenance

# Reuse frozen DG-05/DG-06 constants + helpers verbatim.
from experiments.dg05_adaptive_controller import (
    BATCH_SIZE, BOTTLENECK, CLIP_NORM, DG04_REFERENCE_BETA, EPOCHS, GEN,
    GRAD_ACCUM, LR, SEED, WEIGHT_DECAY, _atomic_json, _atomic_torch, _git_commit,
    _load_reused_dg04, _sha256_file, _slurm_metadata, parameter_report, set_seed,
)
from experiments.dg06_damage_aware import (
    CORRECTION_SET_ARTIFACT, CORRECTION_SET_SHA256, LAMBDA_M, RETENTION_ARTIFACT,
    collate_dg06_batch, select_checkpoint_dg06,
)

# Layers this exploratory branch may touch: frozen anchors + the frozen-screen
# candidate band + negative control. A typo cannot silently steer elsewhere.
ALLOWED_LAYERS = (16, 24, 26, 27, 31)
DG03_BASIS_L24 = REPO / "results/dg03/basis/steering_basis_v1_L24.json"
ATLAS_DIRECTIONS = REPO / "results/basis_frozen_layer_atlas/directions.json"
MODES = ("raw_only", "local_only", "cond_only", "raw_cond", "local_cond")


def _controller_state_hash(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().float().numpy(),
                                      dtype=np.float64).tobytes())
    return "sha256:" + h.hexdigest()


def load_basis_for(layer: int, mode: str):
    """Frozen basis mode for (layer, mode): DG-03 artifact at L24 else atlas columns."""
    if int(layer) == 24:
        return load_basis_mode(DG03_BASIS_L24, mode=mode, layer=24)
    return load_basis_mode_from_atlas(ATLAS_DIRECTIONS, mode=mode, layer=int(layer))


def layered_train_hook(bundle: Any, module: torch.nn.Module, *, layer: int,
                       beta: float, num_forced_prefix: int):
    """Train-mode exact-site hook at an arbitrary allowed layer (action reads only r)."""
    return DecoderPostCrossAttnInterventionHook(
        bundle, int(layer), direction=None, alpha=float(beta), scale=1.0,
        action_fn=module.action, num_forced_prefix=num_forced_prefix,
        norm_preserve=True, mode="train", record=False,
        enforce_contract_layer=False)


def train_batch_d1(bundle: Any, module: torch.nn.Module, examples: Sequence[Any],
                   correction_set, r_m_set, *, layer: int, beta: float,
                   prefix_width: int) -> tuple[torch.Tensor, dict[str, float]]:
    """D1 damage-aware step (correction CE + matrix KL) at the configured layer."""
    batch = collate_dg06_batch(bundle, examples, correction_set, r_m_set, r_m_set,
                               prefix_width=prefix_width, include_embedded=False)
    with torch.no_grad():
        base_out = bundle.model(input_features=batch["input_features"],
                                attention_mask=batch["attention_mask"],
                                decoder_input_ids=batch["decoder_input_ids"],
                                use_cache=False)
    baseline_logits = base_out.logits.float()[:, :-1]
    with layered_train_hook(bundle, module, layer=layer, beta=beta,
                            num_forced_prefix=prefix_width):
        out = bundle.model(input_features=batch["input_features"],
                           attention_mask=batch["attention_mask"],
                           decoder_input_ids=batch["decoder_input_ids"],
                           use_cache=False)
    method_logits = out.logits.float()[:, :-1]
    total, components = damage_aware_loss(
        method_logits, baseline_logits, batch["labels"], batch["ce_mask"],
        batch["rm_mask"], None, lambda_m=LAMBDA_M, lambda_e=1.0)
    (total / GRAD_ACCUM).backward()
    return total.detach(), components


def decode_module(bundle: Any, pop: Any, module: torch.nn.Module, *, layer: int,
                  nfp: int, batch_size: int = BATCH_SIZE) -> tuple[dict[str, str], dict[str, Any]]:
    """Free-decode a checkpoint; collect gate (+ rank-2 mixture) + energy diagnostics."""
    is_rank2 = isinstance(module, FixedBasisAdaptiveController)
    manifest = pop.manifest.sort_values("duration_sec").reset_index(drop=True)
    texts: dict[str, str] = {}
    gate_v: list[float] = []
    picond_v: list[float] = []
    energies: list[float] = []
    module.eval()
    for start in range(0, len(manifest), int(batch_size)):
        batch = manifest.iloc[start:start + int(batch_size)]
        ids = [str(x) for x in batch["utterance_id"]]
        inputs = batch_model_inputs(bundle, batch["audio_path"].tolist())

        def action_fn(*, r, abs_pos, **_):
            eligible = (abs_pos >= int(nfp)).view(1, -1)
            if is_rank2:
                gate, pi, direction = module(r)
                m = eligible.expand_as(gate)
                gate_v.extend(gate.detach().float()[m].cpu().numpy().tolist())
                picond_v.extend(pi.detach().float()[..., 1][m].cpu().numpy().tolist())
                return gate, direction
            gate = module(r)
            m = eligible.expand_as(gate)
            gate_v.extend(gate.detach().float()[m].cpu().numpy().tolist())
            return module.action(r=r)

        hook = DecoderPostCrossAttnInterventionHook(
            bundle, int(layer), direction=None, alpha=DG04_REFERENCE_BETA, scale=1.0,
            action_fn=action_fn, num_forced_prefix=nfp, norm_preserve=True,
            mode="steer", record=True, record_last_only=False,
            enforce_contract_layer=False)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inputs, **GEN)
        for rec in hook.records:
            if rec.steered:
                energies.append(rec.edit_norm)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        for uid, text in zip(ids, bundle.processor.batch_decode(seq, skip_special_tokens=True)):
            texts[uid] = text.strip()

    audit: dict[str, Any] = {"n_steered": len(energies),
                             "total_energy": float(np.sum(energies)) if energies else 0.0,
                             "mean_energy": float(np.mean(energies)) if energies else 0.0,
                             "n_gate_eligible": len(gate_v)}
    if gate_v:
        g = np.asarray(gate_v, dtype=float)
        audit["gate_strength"] = {
            "mean": float(g.mean()), "median": float(np.median(g)),
            "std": float(g.std()), "q10": float(np.quantile(g, 0.1)),
            "q90": float(np.quantile(g, 0.9)),
            "near_zero_frac": float((g <= 0.05).mean()),
            "near_one_frac": float((g >= 0.95).mean())}
    if is_rank2 and picond_v:
        p = np.asarray(picond_v, dtype=float)
        audit["pi_cond"] = {
            "mean": float(p.mean()), "std": float(p.std()), "median": float(np.median(p)),
            "q25": float(np.quantile(p, 0.25)), "q75": float(np.quantile(p, 0.75)),
            "frac_gt_0.1": float((p > 0.1).mean()), "frac_gt_0.2": float((p > 0.2).mean()),
            "note": "overall eligible-position distribution (free decode); per-outcome "
                    "split deferred (needs per-position reference alignment)"}
    return texts, audit


def run_training(cfg: Mapping[str, Any], output_dir: Path, *, layer: int, mode: str,
                 seed: int) -> dict[str, Any]:
    started = time.time()
    if int(layer) not in ALLOWED_LAYERS:
        raise ValueError(f"layer {layer} not in predeclared {ALLOWED_LAYERS}")
    if mode not in MODES:
        raise ValueError(f"mode {mode} not in {MODES}")
    for role in ("D-dev-confirm", "D-test"):
        if role in (cfg.get("data", {}).get("training_roles", []) or []):
            raise ValueError(f"forbidden role {role} in training roles")
    set_seed(int(seed))

    basis = load_basis_for(int(layer), mode)
    bundle = load_whisper({"model": dict(cfg["model"])})
    bundle.model.eval()
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("backbone must be frozen")

    module, spec = build_steering_module(basis, d_model=bundle.d_model, layer=int(layer),
                                         bottleneck=BOTTLENECK)
    module = module.to(bundle.device)
    assert_basis_frozen(module)
    init_hash = _controller_state_hash(module)
    params = parameter_report(module, bundle.model) if isinstance(
        module, FixedBasisAdaptiveController) else {
        "controller_trainable": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
        "whisper_total": int(sum(p.numel() for p in bundle.model.parameters())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frozen C_E (hash-checked) + R_M (matrix retention) reused verbatim; both are
    # layer-independent (token positions / baseline correctness), so valid at any layer.
    if _sha256_file(CORRECTION_SET_ARTIFACT) != CORRECTION_SET_SHA256:
        raise RuntimeError("frozen DG-05 C_E artifact hash mismatch")
    correction_set = load_correction_set(CORRECTION_SET_ARTIFACT)
    ret_payload = json.loads(RETENTION_ARTIFACT.read_text(encoding="utf-8"))
    if ret_payload.get("schema_version") != "dg06_retention_sets_v1":
        raise RuntimeError("frozen DG-06 retention artifact missing/altered")
    r_m_set = load_retention_set(ret_payload["matrix_retention_R_M"])

    data_cfg = load_config(cfg["data"]["candidate_config"])
    from steer_sweep.trackb.data import build_examples
    examples = build_examples(bundle, data_cfg, ("loc-train", "util-train"))
    if not examples:
        raise RuntimeError("training pool is empty")
    prefix_width = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")

    manifest = stage_provenance(dict(cfg), f"learned_expansion_L{layer}_{mode}")
    manifest.update({
        "workstream": "C_L24_authorized" if int(layer) == 24 else "D_conditional_layer",
        "layer": int(layer), "mode": mode, "basis_rank": basis.rank,
        "basis_provenance": dict(basis.provenance), "controller_class": spec.controller_class,
        "trainable_parameters": spec.trainable_parameters, "beta": DG04_REFERENCE_BETA,
        "objective": "L_corr(C_E) + lambda_M L_ret,M", "lambda_m": LAMBDA_M,
        "seed": int(seed), "controller_init_hash": init_hash,
        "training_roles": ["loc-train", "util-train"],
        "correction_set_sha256": CORRECTION_SET_SHA256,
        "retention_set_sha256": _sha256_file(RETENTION_ARTIFACT),
        "git_commit": _git_commit(), "slurm": _slurm_metadata(),
        "exact_site": "decoder post-cross-attention residual, pre-FFN",
        "forbidden_roles": ["D-dev-confirm", "D-test"],
    })
    _atomic_json(output_dir / "manifest.json", manifest)

    def batch_has_target(bx: Sequence[Any]) -> bool:
        return any(correction_set.positions.get(str(e.utterance_id))
                   or r_m_set.positions.get(str(e.utterance_id)) for e in bx)

    batches = [examples[i:i + BATCH_SIZE] for i in range(0, len(examples), BATCH_SIZE)
               if batch_has_target(examples[i:i + BATCH_SIZE])]
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(EPOCHS):
        epoch_started = time.time()
        module.train()
        optimizer.zero_grad(set_to_none=True)
        agg = {"l_corr": 0.0, "l_ret_m": 0.0, "total": 0.0}
        counts = {"n_corr": 0, "n_ret_m": 0}
        grad_norms: list[float] = []
        for index, bx in enumerate(batches):
            _loss, comp = train_batch_d1(bundle, module, bx, correction_set, r_m_set,
                                         layer=int(layer), beta=DG04_REFERENCE_BETA,
                                         prefix_width=prefix_width)
            agg["l_corr"] += comp["l_corr"] * comp["n_corr"]
            agg["l_ret_m"] += comp["l_ret_m"] * comp["n_ret_m"]
            agg["total"] += comp["total"]
            counts["n_corr"] += comp["n_corr"]
            counts["n_ret_m"] += comp["n_ret_m"]
            if (index + 1) % GRAD_ACCUM == 0 or index + 1 == len(batches):
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.requires_grad], CLIP_NORM)
                grad_norms.append(float(gn.detach().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        history.append({
            "epoch": epoch + 1,
            "raw_l_corr": agg["l_corr"] / max(1, counts["n_corr"]),
            "raw_l_ret_m": agg["l_ret_m"] / max(1, counts["n_ret_m"]),
            "mean_total_loss": agg["total"] / max(1, len(batches)),
            "n_C_E": counts["n_corr"], "n_R_M": counts["n_ret_m"],
            "gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "runtime_sec": float(time.time() - epoch_started),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else None,
        })
        _atomic_torch(output_dir / f"checkpoint_epoch{epoch + 1}.pt", {
            "schema_version": "learned_expansion_checkpoint_v1", "layer": int(layer),
            "mode": mode, "epoch": epoch + 1,
            "state_dict": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
            "trainable_parameters": spec.trainable_parameters, "beta": DG04_REFERENCE_BETA,
            "basis_provenance": dict(basis.provenance), "init_hash": init_hash})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    _atomic_json(output_dir / "training_history.json", {"records": history})

    # Free-decode checkpoint evaluation on D-dev-select (reuse DG-04 scoring).
    from steer_sweep import data as D
    pop = D.build_population(bundle, data_cfg, "D-dev-select", assert_anchors=True)
    refs = D.reference_of(pop)
    ids = list(pop.utterance_ids)
    a0 = _load_reused_dg04("B0", ids)
    nfp = prefix_width
    from experiments.dg04_frozen_baselines import _build_outside_sets
    out_sets, out_langs, out_diag = _build_outside_sets(data_cfg, refs, ids)
    eval_records: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        payload = torch.load(output_dir / f"checkpoint_epoch{epoch}.pt",
                             map_location=bundle.device, weights_only=False)
        module.load_state_dict(payload["state_dict"], strict=True)
        texts, audit = decode_module(bundle, pop, module, layer=int(layer), nfp=nfp)
        result = _emit_result(refs, a0["texts"], texts, ids, layer=int(layer), mode=mode,
                              basis=basis, audit=audit, out_sets=out_sets, out_langs=out_langs,
                              seed=int(seed), bundle=bundle, cfg=cfg)
        transitions = result["metrics"]["transitions"]
        rate = (result["metrics"].get("retention", {}).get("matrix_zh", {}) or {}).get("rate")
        evaluation = {
            "epoch": epoch, "layer": int(layer), "mode": mode,
            "checkpoint": str(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "checkpoint_sha256": _sha256_file(output_dir / f"checkpoint_epoch{epoch}.pt"),
            "result_v1": result, "texts": texts,
            "utility": float(transitions["net_corrections"]),
            "pier_gain": float(result["metrics"].get("pier_gain", 0.0)),
            "matrix_retention": float(rate) if rate is not None else -1.0,
            "valid_outside_harm": result["metrics"].get("outside_harm") is not None,
            "total_energy": float(audit.get("total_energy", 0.0)),
            "audit": audit}
        eval_records.append(evaluation)
        _atomic_json(output_dir / f"evaluation_epoch{epoch}.json", evaluation)
    selected = select_checkpoint_dg06(eval_records)
    selection = {
        "rule": "positive utility then PIER gain then matrix retention then lower energy",
        "layer": int(layer), "mode": mode,
        "selected_epoch": selected.get("epoch") if selected else None,
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "eligible_epochs": [int(x["epoch"]) for x in eval_records
                            if float(x["utility"]) > 0 and bool(x["valid_outside_harm"])],
        "population": {"role": "D-dev-select", "n_utterances": len(ids), "outside_harm": out_diag}}
    if selected:
        import shutil
        shutil.copy2(selected["checkpoint"], output_dir / "selected_checkpoint.pt")
        selection["selected_checkpoint_sha256"] = _sha256_file(output_dir / "selected_checkpoint.pt")
    _atomic_json(output_dir / "selection.json", selection)
    summary = {
        "schema_version": "learned_expansion_run_v1", "layer": int(layer), "mode": mode,
        "basis_rank": basis.rank, "seed": int(seed), "controller_init_hash": init_hash,
        "manifest": manifest, "history": history, "trainable_parameters": spec.trainable_parameters,
        "a0_baseline": a0["result_v1"], "evaluations": eval_records, "selection": selection,
        "runtime_sec": float(time.time() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available() else None,
        "source": "NEW 1-SEED EXPLORATORY"}
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _emit_result(refs, base, method_texts, ids, *, layer, mode, basis, audit,
                 out_sets, out_langs, seed, bundle, cfg) -> dict[str, Any]:
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
    cond = f"L{layer}_{mode}"
    res = CanonicalResult(
        run_id=make_run_id(f"learned_expansion/{cond}"),
        system_name=f"learned_expansion_{cond}",
        model_id=bundle.model_id, model_revision=bundle.revision,
        data_role="D-dev-select", decode_regime="greedy", beam=1, seed=int(seed),
        method=MethodConfig(layer=int(layer),
                            direction_artifact_id=json.dumps(basis.provenance.get("frozen_tensor_hashes")),
                            direction_type=f"adaptive_{mode}",
                            gate_type="adaptive_controller",
                            steering_strength=DG04_REFERENCE_BETA),
        metrics=metrics,
        provenance={"layer": int(layer), "mode": mode, "basis_rank": basis.rank,
                    "basis_provenance": dict(basis.provenance), "decode": GEN,
                    "config_hash": sha256_obj(dict(cfg)), "git_commit": _git_commit(),
                    "slurm": _slurm_metadata(), "source": "NEW 1-SEED EXPLORATORY"})
    d = res.to_dict()
    validate(d)
    return d


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expansion/learned_expansion_run.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run", action="store_true", help="execute training (never implied)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if int(args.layer) not in ALLOWED_LAYERS:
        raise SystemExit(f"layer {args.layer} not in predeclared {ALLOWED_LAYERS}")
    out = (Path(args.output_dir) if args.output_dir
           else REPO / "results/learned_expansion/proposed" / f"L{args.layer}_{args.mode}")
    if not args.run:
        basis = load_basis_for(int(args.layer), args.mode)
        print(json.dumps({"ready": True, "layer": args.layer, "mode": args.mode,
                          "basis_rank": basis.rank, "basis_source": basis.provenance["basis_source"],
                          "beta": DG04_REFERENCE_BETA, "seed": args.seed, "gpu_run": False}, indent=2))
        return 0
    run_training(cfg, out, layer=int(args.layer), mode=args.mode, seed=int(args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
