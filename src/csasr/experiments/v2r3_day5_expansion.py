"""Day 5: dose extension, eligibility expansion, and the first free decoding.

Three questions, one run:

1. **Does the dose curve saturate?**  Day 4 left Site D layers 8 and 16 monotone
   and still rising at rho=2.  An effect that never saturates is harder to
   separate from a generic magnitude effect, so the grid is extended to rho 3
   and 4 -- diagnostically.  The frozen action is not re-derived from it.
2. **How much correctability survives as span quality degrades?**  The frozen
   action is rerun at the 300 ms and 200 ms floors beside the 400 ms tier, and
   effect is reported against coverage.
3. **Does the deletion advantage survive free decoding?**  Under teacher forcing
   Site D moved deletions *more* than substitutions, because teacher forcing
   supplies a decoder step for every gold token.  Free decoding removes that:
   a word the baseline omits has no emitted token and no step to intervene at.
   The named prediction is that the deletion advantage collapses.  If it does
   not, the mechanism story is wrong.

Free decoding here is **oracle-localized** -- gold span, gold decoder step -- so
it is an upper bound, not the practical result.  Day 7 runs the non-oracle
version.  Nothing here selects a layer, a strength, or an eligibility floor, and
no gate is evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..evaluation.mer import corpus_mer, error_rate
from ..evaluation.pier import pier, unit_status
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..lss.sites import assert_dropout_disabled
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from . import v2r3_directions as D3
from . import v2r3_oracle_screen as OS

STAGE = "v2r3_day5_expansion"

# ---------------------------------------------------------------------------
# FROZEN ACTION -- decided upstream, recorded here, never re-derived
# ---------------------------------------------------------------------------

FROZEN_ACTION: dict[str, Any] = {
    "site": "D",
    "decoder_layer": 16,
    "rho": 1.0,
    "rationale": (
        "layer 16 beat the label-permuted null at every rho in every stratum; "
        "its dose-response is clean and monotone; layer 31 turns down at rho=2 "
        "and layer 24 collapses. rho=1 is one centroid displacement -- the "
        "interpretable unit -- and sits mid-grid rather than at an edge."),
    "decided_by": "human, before this session",
    "re_derived_here": False,
}

#: Diagnostic extension only.  The frozen rho is not revisited on this curve.
EXTENDED_RHO = (3.0, 4.0)
DOSE_LAYERS = (8, 16)

#: Expansion tiers.  400 ms is the existing conservative tier and is rerun here
#: so all three are measured by the same code in the same job.
FLOORS_MS = (400.0, 300.0, 200.0)

FROZEN_CONFIG: dict[str, Any] = {
    "frozen_action": FROZEN_ACTION,
    "extended_rho": list(EXTENDED_RHO),
    "dose_layers": list(DOSE_LAYERS),
    "floors_ms": list(FLOORS_MS),
    "free_decoding": {"localization": "oracle (gold span, gold decoder step)",
                      "is_upper_bound": True,
                      "non_oracle_deferred_to": "Day 7"},
    "selects_layer_strength_or_floor": False,
    "evaluates_gate": False,
    "site_e_pooling": OS.SITE_E_POOLING,
    "bootstrap": {"kind": "wild cluster", "cluster": "dialogue_id",
                  "critical_values": "t(G-1)", "draws": OS.BOOTSTRAP_DRAWS,
                  "seed": OS.BOOTSTRAP_SEED},
}


class StepGatedDecoderSteering:
    """Steer the post-cross-attention residual at one *generation step*.

    `DecoderPostCrossAttnSteeringHook` gates on a position mask supplied up
    front, which suits teacher forcing where every position is known. Free
    decoding produces its own sequence, so the intervention has to be gated on
    the step counter instead: the oracle supplies *which* step, and this hook
    fires on exactly that one.
    """

    def __init__(self, bundle, layer: int, direction: torch.Tensor, alpha: float,
                 target_step: int, norm_preserve: bool = True):
        self.bundle = bundle
        self.layer = int(layer)
        self.direction = direction
        self.alpha = float(alpha)
        self.target_step = int(target_step)
        self.norm_preserve = norm_preserve
        self.step = -1              # prefill is step -1; first generated step is 0
        self.fired = 0
        self._residual: torch.Tensor | None = None
        self._handles: list = []

    def _pre_hook(self, _mod, args):
        self._residual = args[0]
        return None

    def _hook(self, _mod, _inp, output):
        from ..lss.sites import _first, _rebuild
        from ..models.hooks import apply_steering

        if self._residual is None:
            raise RuntimeError("cross-attention ran without its layer-norm pre-hook")
        attn_out = _first(output)
        site = self._residual + attn_out
        if site.shape[1] > 1:                    # prefill, not a generation step
            return output
        self.step += 1
        if self.step != self.target_step:
            return output
        gain = torch.ones(site.shape[:2], dtype=torch.float32)
        steered = apply_steering(site, self.direction, self.alpha, 1.0, gain,
                                 self.norm_preserve)
        self.fired += 1
        return _rebuild(output, attn_out + (steered - site))

    def __enter__(self) -> "StepGatedDecoderSteering":
        assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(
            layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._residual = None
        return False


def free_decode(bundle, cfg: dict, audio_path: str, *,
                direction: np.ndarray | None = None, layer: int = 16,
                rho: float = 1.0, target_step: int | None = None) -> tuple[str, int]:
    """Greedy free decoding, optionally steered at one oracle-given step."""
    from ..models.generation import batch_model_inputs

    inputs = batch_model_inputs(bundle, [audio_path])
    kwargs = {"task": "transcribe", "language": None, "do_sample": False,
              "num_beams": 1, "temperature": 0.0, "max_new_tokens": 200}
    if direction is None or target_step is None:
        with torch.inference_mode():
            out = bundle.model.generate(**inputs, **kwargs)
        fired = 0
    else:
        vector = torch.as_tensor(direction, dtype=torch.float32, device=bundle.device)
        with StepGatedDecoderSteering(bundle, layer, vector, rho, target_step) as hook:
            with torch.inference_mode():
                out = bundle.model.generate(**inputs, **kwargs)
            fired = hook.fired
    text = bundle.processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return text, fired


def outcome_rates(reference: str, baseline: str, steered: str,
                  unit_index: int) -> dict[str, Any]:
    """Correction, corruption, and spillover for one target unit."""
    before = unit_status(reference, baseline)
    after = unit_status(reference, steered)
    b_ok = bool(before.get(unit_index, (False, ""))[0])
    a_ok = bool(after.get(unit_index, (False, ""))[0])
    changed_elsewhere = sum(
        1 for k in set(before) | set(after)
        if k != unit_index
        and bool(before.get(k, (False, ""))[0]) != bool(after.get(k, (False, ""))[0]))
    improved_elsewhere = sum(
        1 for k in set(before) | set(after)
        if k != unit_index
        and not bool(before.get(k, (False, ""))[0])
        and bool(after.get(k, (False, ""))[0]))
    return {
        "baseline_correct": b_ok, "steered_correct": a_ok,
        "corrected": (not b_ok) and a_ok,
        "corrupted": b_ok and (not a_ok),
        "spillover_changed": int(changed_elsewhere),
        "spillover_improved": int(improved_elsewhere),
        "spillover_damaged": int(changed_elsewhere - improved_elsewhere),
    }


def corpus_word_error_rate(references: Sequence[str],
                           hypotheses: Sequence[str]) -> dict[str, Any]:
    """Corpus WER: per-utterance edits and reference lengths, summed.

    Concatenating the corpus into one string before aligning would let an error
    in one utterance be repaired by tokens from the next.
    """
    edits = length = 0
    for reference, hypothesis in zip(references, hypotheses):
        counts = error_rate(str(reference).split(), str(hypothesis).split())
        # csasr.evaluation.mer.error_rate names these sub/del/ins
        edits += int(counts["sub"] + counts["del"] + counts["ins"])
        length += max(1, len(str(reference).split()))
    return {"wer": edits / length, "edits": int(edits), "reference_tokens": int(length)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def resolve_output(output: str | Path) -> Path:
    target = Path(output)
    if target.is_symlink():
        raise SystemExit(f"refusing a symlink destination: {target}")
    target = target.resolve()
    if "artifacts_lss" in str(target):
        raise SystemExit(f"refusing to write inside the v1 root: {target}")
    if target.exists() or manifest_mod.manifest_path(target).exists():
        raise SystemExit(f"refusing an existing destination: {target}")
    return target


def _spans_at_floor(candidates, dialogue_of, poi, floor_ms: float):
    from .v2r3_headroom_diagnostic import eligible_spans, structurally_eligible

    adjusted = conv.apply_convention(candidates, D3.CTC_CONVENTION,
                                     family=D3.SPAN_FAMILY)
    spans = eligible_spans(adjusted, family=D3.SPAN_FAMILY)
    spans = spans[structurally_eligible(spans)].copy()
    spans = spans[spans["span_duration_ms"] >= float(floor_ms)].reset_index(drop=True)
    frame = adjusted[adjusted["aligner_family"].astype(str) == D3.SPAN_FAMILY]
    edges = {}
    for (utterance, index), group in frame.groupby(["utterance_id", "reference_unit_index"]):
        edges[(str(utterance), int(index))] = (
            float(pd.to_numeric(group["start_sec"], errors="coerce").min()),
            float(pd.to_numeric(group["end_sec"], errors="coerce").max()))
    starts, ends = [], []
    for utterance, indices in zip(spans["utterance_id"], spans["unit_indices"]):
        pairs = [edges[(str(utterance), int(i))] for i in indices
                 if (str(utterance), int(i)) in edges]
        starts.append(min(p[0] for p in pairs) if pairs else float("nan"))
        ends.append(max(p[1] for p in pairs) if pairs else float("nan"))
    spans["start_sec"], spans["end_sec"] = starts, ends
    spans["dialogue_id"] = [str(dialogue_of.get(str(c), ""))
                            for c in spans["conversation_id"]]
    spans = spans[spans["start_sec"].notna()].reset_index(drop=True)
    return D3.attach_baseline_status(spans, poi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1b_candidates_dialogue_v2r3.yaml")
    parser.add_argument("--directions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    target_dir = resolve_output(args.output_dir)
    directions_root = Path(args.directions)
    if "smoke" in directions_root.name:
        raise SystemExit(f"refusing a smoke generation: {directions_root}")

    # every prior artifact is verified by BYTES; identity is not required for an
    # artifact consumed by newly written code (runbook, identity section)
    npz_path = directions_root / "directions.npz"
    side = manifest_mod.load(npz_path)
    fmt, _ = manifest_mod._identity_format(dict(side.get("identity") or {}))
    byte_check = manifest_mod.verify(npz_path, cfg=None, require_identity=False)
    if not byte_check["ok"]:
        raise SystemExit(f"directions failed byte verification: {byte_check}")
    directions = dict(np.load(npz_path))
    identity_note = {"classification": fmt, "identity_required": False,
                     "byte_verified_sha256": side["sha256"],
                     "reason": "consumed by newly written code; bytes are the "
                               "property that matters for reading"}

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    for path in [candidate_path] + [role_root / f"role_{r}.parquet"
                                    for r in conv.DEVELOPMENT_ROLES] + \
                [poi_root / f"poi_{r}.parquet" for r in conv.DEVELOPMENT_ROLES]:
        check = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not check["ok"]:
            raise SystemExit(f"{path.name} failed byte verification: {check}")

    candidates, verdict = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)
    manifest = pd.concat([pd.read_parquet(role_root / f"role_{r}.parquet")
                          for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    poi = pd.concat([pd.read_parquet(poi_root / f"poi_{r}.parquet").rename(
        columns={"poi_index": "reference_unit_index"})
        for r in conv.DEVELOPMENT_ROLES], ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))

    tiers = {}
    for floor in FLOORS_MS:
        spans = _spans_at_floor(candidates, dialogue_of, poi, floor)
        develop = spans[spans["role"] == "D-dev-select"]
        tiers[f"{int(floor)}ms"] = {"spans": int(len(develop)),
                                    "error_units": int(develop["units_error"].sum()),
                                    "dialogues": int(develop["dialogue_id"].nunique())}
    plan = {"frozen_configuration": FROZEN_CONFIG, "tiers": tiers,
            "directions_identity_note": identity_note,
            "destination": str(target_dir)}
    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    bundle = load_whisper(cfg)
    construct_spans = _spans_at_floor(candidates, dialogue_of, poi, 400.0)
    construct = construct_spans[(construct_spans["role"] == "D-construct")
                                & construct_spans["all_correct"]]
    build_manifest = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
    d_meta, d_raw = D3.site_d_contrasts(bundle, build_manifest, construct,
                                        list(DOSE_LAYERS), batch_size=4, log=log)

    all_scores: list[pd.DataFrame] = []

    # ---- Task 1: extended dose grid, layers 8 and 16, rho 3 and 4 -------------
    develop400 = _spans_at_floor(candidates, dialogue_of, poi, 400.0)
    develop400 = develop400[develop400["role"] == "D-dev-select"]
    targets400 = OS.build_targets(bundle, candidates, develop400, poi, manifest)
    if args.limit:
        keep = list(dict.fromkeys(targets400["utterance_id"]))[: int(args.limit)]
        targets400 = targets400[targets400["utterance_id"].isin(keep)]
    utt400 = manifest[manifest["utterance_id"].isin(set(targets400["utterance_id"]))]
    utt400 = utt400.drop_duplicates("utterance_id").reset_index(drop=True)
    positions = {str(u): [int(v) for v in g["query_position"]]
                 for u, g in targets400.groupby("utterance_id")}
    log.info("task 1: extended dose over %d targets", len(targets400))
    for layer in DOSE_LAYERS:
        vector = directions[f"site_d_layer{layer}"]
        nulls = OS.label_permuted_directions(
            np.vstack(d_raw["vectors"][layer]), d_meta,
            draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + 100 + layer)
        for rho in EXTENDED_RHO:
            variants = [("correct", vector)] + [
                (f"label_permuted_{k}", n["direction"]) for k, n in enumerate(nulls)]
            for kind, vec in variants:
                per = []
                for start in range(0, len(utt400), args.batch_size):
                    batch = utt400.iloc[start:start + args.batch_size]
                    sub = targets400[targets400["utterance_id"].isin(
                        set(batch["utterance_id"].astype(str)))]
                    per.append(OS.score_batch(bundle, batch, sub,
                                              decoder=(layer, vec, rho),
                                              decoder_positions=positions))
                frame = pd.concat(per, ignore_index=True)
                frame["task"], frame["layer"] = "dose", str(layer)
                frame["rho"], frame["kind"], frame["floor_ms"] = rho, kind, 400.0
                all_scores.append(frame)
        log.info("  dose layer %d done", layer)

    # ---- Task 2: eligibility expansion at the frozen action -------------------
    layer, rho = int(FROZEN_ACTION["decoder_layer"]), float(FROZEN_ACTION["rho"])
    vector = directions[f"site_d_layer{layer}"]
    nulls = OS.label_permuted_directions(
        np.vstack(d_raw["vectors"][layer]), d_meta,
        draws=OS.CONTROL_LABEL_PERMUTED_DRAWS, seed=OS.CONTROL_SEED + 100 + layer)
    for floor in FLOORS_MS:
        spans = _spans_at_floor(candidates, dialogue_of, poi, floor)
        spans = spans[spans["role"] == "D-dev-select"]
        tier_targets = OS.build_targets(bundle, candidates, spans, poi, manifest)
        if args.limit:
            keep = list(dict.fromkeys(tier_targets["utterance_id"]))[: int(args.limit)]
            tier_targets = tier_targets[tier_targets["utterance_id"].isin(keep)]
        utts = manifest[manifest["utterance_id"].isin(set(tier_targets["utterance_id"]))]
        utts = utts.drop_duplicates("utterance_id").reset_index(drop=True)
        pos = {str(u): [int(v) for v in g["query_position"]]
               for u, g in tier_targets.groupby("utterance_id")}
        log.info("task 2: floor %.0f ms, %d targets, %d utterances",
                 floor, len(tier_targets), len(utts))
        for kind, vec in [("correct", vector)] + [
                (f"label_permuted_{k}", n["direction"]) for k, n in enumerate(nulls)]:
            per = []
            for start in range(0, len(utts), args.batch_size):
                batch = utts.iloc[start:start + args.batch_size]
                sub = tier_targets[tier_targets["utterance_id"].isin(
                    set(batch["utterance_id"].astype(str)))]
                per.append(OS.score_batch(bundle, batch, sub,
                                          decoder=(layer, vec, rho),
                                          decoder_positions=pos))
            frame = pd.concat(per, ignore_index=True)
            frame["task"], frame["layer"] = "tier", str(layer)
            frame["rho"], frame["kind"], frame["floor_ms"] = rho, kind, float(floor)
            all_scores.append(frame)
        baseline = []
        for start in range(0, len(utts), args.batch_size):
            batch = utts.iloc[start:start + args.batch_size]
            sub = tier_targets[tier_targets["utterance_id"].isin(
                set(batch["utterance_id"].astype(str)))]
            baseline.append(OS.score_batch(bundle, batch, sub))
        frame = pd.concat(baseline, ignore_index=True)
        frame["task"], frame["layer"] = "tier", str(layer)
        frame["rho"], frame["kind"], frame["floor_ms"] = 0.0, "C00", float(floor)
        all_scores.append(frame)

    scores = pd.concat(all_scores, ignore_index=True)
    meta = pd.concat([targets400, tier_targets])[
        ["utterance_id", "reference_unit_index", "dialogue_id", "category",
         "is_deletion"]].drop_duplicates()
    scores = scores.merge(meta, on=["utterance_id", "reference_unit_index"], how="left")

    # ---- Task 3: first free decoding, oracle-localized ------------------------
    log.info("task 3: oracle-localized free decoding over %d utterances", len(utt400))
    prefix_len = 4
    free_rows = []
    for _, row in utt400.iterrows():
        utterance = str(row["utterance_id"])
        group = targets400[targets400["utterance_id"] == utterance]
        step = int(group["token_index"].min()) - 1     # gold step for the earliest target
        baseline_text, _ = free_decode(bundle, cfg, row["audio_path"])
        steered_text, fired = free_decode(
            bundle, cfg, row["audio_path"], direction=vector, layer=layer,
            rho=rho, target_step=max(0, step))
        for _, t in group.iterrows():
            rates = outcome_rates(row["transcript_raw"], baseline_text, steered_text,
                                  int(t["reference_unit_index"]))
            free_rows.append({"utterance_id": utterance,
                              "reference_unit_index": int(t["reference_unit_index"]),
                              "dialogue_id": str(t["dialogue_id"]),
                              "category": str(t["category"]),
                              "is_deletion": bool(t["is_deletion"]),
                              "hook_fired": int(fired),
                              "reference": row["transcript_raw"],
                              "baseline_text": baseline_text,
                              "steered_text": steered_text, **rates})
    free = pd.DataFrame(free_rows)

    utterance_level = free.drop_duplicates("utterance_id")
    corpus = {
        "PIER_baseline": pier(utterance_level["reference"].tolist(),
                              utterance_level["baseline_text"].tolist()),
        "PIER_steered": pier(utterance_level["reference"].tolist(),
                             utterance_level["steered_text"].tolist()),
        "MER_baseline": corpus_mer(utterance_level["reference"].tolist(),
                                   utterance_level["baseline_text"].tolist()),
        "MER_steered": corpus_mer(utterance_level["reference"].tolist(),
                                  utterance_level["steered_text"].tolist()),
        # corpus WER sums per-utterance edits and reference lengths; aligning one
        # concatenated string would let errors migrate across utterance boundaries
        "WER_baseline": corpus_word_error_rate(utterance_level["reference"],
                                               utterance_level["baseline_text"]),
        "WER_steered": corpus_word_error_rate(utterance_level["reference"],
                                              utterance_level["steered_text"]),
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(target_dir / "day5_scores.parquet", index=False)
    free.to_parquet(target_dir / "day5_free_decoding.parquet", index=False)
    payload = {"frozen_configuration": FROZEN_CONFIG, "plan": plan,
               "corpus_metrics": corpus,
               "rows_teacher_forced": int(len(scores)),
               "rows_free_decoding": int(len(free)),
               "selects_layer_strength_or_floor": False, "evaluates_gate": False}
    (target_dir / "day5_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for name in ("day5_scores.parquet", "day5_free_decoding.parquet",
                 "day5_report.json"):
        manifest_mod.publish(target_dir / name, stage=STAGE, cfg=cfg,
                             parents=[parent] if parent else (),
                             schema="lss_v2r3_day5_expansion_v1",
                             extra={"frozen_configuration": FROZEN_CONFIG,
                                    "evaluates_gate": False})
    print(json.dumps({"state": "completed", "output": str(target_dir),
                      "teacher_forced_rows": int(len(scores)),
                      "free_rows": int(len(free)),
                      "corpus_metrics": corpus}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
