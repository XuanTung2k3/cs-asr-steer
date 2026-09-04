"""Day 4: the teacher-forced oracle screen with its control battery.

Decisive experiment.  It measures whether the frozen directions move the gold
token at the places the localizer says they should, against four controls that
each remove one thing the hypothesis depends on.

Teacher-forced scoring only: one forward pass per utterance per condition, gold
transcripts fixed.  **No free decoding** -- that is Day 7.  Nothing here selects
a layer or a strength, and no gate is evaluated.

The four controls, and what each one kills:

* wrong location -- same direction and strength at a random position in the same
  utterance.  If this matches the correct location, the span means nothing.
* matched-norm random direction at the correct location.  If this matches, the
  effect is perturbation, not direction.
* opposite sign at the correct location.  If this matches, the axis is not
  directional.
* label-permuted -- language labels shuffled within nuisance-matched strata and
  the direction rebuilt through the *identical* pipeline, residualisation
  included.  If this matches, the language pairing carries nothing.

Pair permutation is deliberately **not** implemented.  For a paired mean
difference it is algebraically vacuous: sum_i (h+_i - h-_pi(i)) / N equals
mean(h+) - mean(h-) for every permutation pi, so it cannot be a null.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.language_tags import EN, ZH
from ..lss import manifest as manifest_mod
from ..lss.align import conventions as conv
from ..lss.sites import (DecoderPostCrossAttnSteeringHook, assert_no_site_hooks)
from ..models.hooks import EncoderSteeringHook
from ..models.whisper import load_whisper
from ..utils.config import load_config
from ..utils.logging import setup_logging
from . import v2r3_directions as D3

STAGE = "v2r3_oracle_screen"

# ---------------------------------------------------------------------------
# FROZEN CONFIGURATION
# ---------------------------------------------------------------------------

RHO_GRID = (0.25, 0.5, 1.0, 2.0)

#: Site-E pooling used for the screen: central mass, as specified for the
#: conservative subset.  The other two variants stay for Day 6.
SITE_E_POOLING = "central60"

#: C11 pairs one encoder layer with one decoder layer at matched depth rather
#: than crossing all 16 combinations: the factorial is over sites, not over
#: layer pairs, and 16 crossings would quadruple the run for no extra contrast.
C11_LAYER_PAIRS = ((15, 8), (23, 16), (27, 24), (31, 31))

CONTROL_RANDOM_DRAWS = 5
CONTROL_LABEL_PERMUTED_DRAWS = 5
CONTROL_SEED = 20260816

#: Wild cluster bootstrap, Rademacher weights, clustered on dialogue_id, with
#: t(G-1) critical values.  G is small by corpus construction and is reported
#: with every interval.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_CI = 0.95

#: Strata reported for every outcome.  (c) is descriptive only: at G=12 the wild
#: bootstrap admits 2^12 = 4096 sign vectors, so the smallest attainable p-value
#: is about 2.4e-4 -- adequate to detect an effect, close to uninformative as a
#: null.
STRATA = ("all_errors", "substitution_only", "wrong_language_substitution")
DESCRIPTIVE_ONLY = ("wrong_language_substitution",)

FROZEN_CONFIG: dict[str, Any] = {
    "rho_grid": list(RHO_GRID),
    "site_e_pooling": SITE_E_POOLING,
    "c11_layer_pairs": [list(p) for p in C11_LAYER_PAIRS],
    "conditions": ["C00", "C10", "C01", "C11"],
    "controls": ["wrong_location", "matched_norm_random", "opposite_sign",
                 "label_permuted"],
    "control_random_draws": CONTROL_RANDOM_DRAWS,
    "control_label_permuted_draws": CONTROL_LABEL_PERMUTED_DRAWS,
    "control_seed": CONTROL_SEED,
    "pair_permutation_implemented": False,
    "pair_permutation_note": (
        "algebraically vacuous for a paired mean difference; labels are permuted "
        "within nuisance-matched strata instead"),
    "teacher_forced_only": True,
    "free_decoding": False,
    "bootstrap": {"kind": "wild cluster", "weights": "Rademacher",
                  "cluster": "dialogue_id", "critical_values": "t(G-1)",
                  "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                  "ci": BOOTSTRAP_CI},
    "strata": list(STRATA),
    "descriptive_only": list(DESCRIPTIVE_ONLY),
    "selects_layer_or_strength": False,
    "evaluates_gate": False,
}


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------

def build_targets(bundle, candidates: pd.DataFrame, spans: pd.DataFrame,
                  poi: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per baseline-error unit: where to steer and what to score."""
    from ..data.alignment import build_prefix, token_char_offsets
    from ..data.normalize import normalize_and_segment

    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    categories = {(str(u), int(i)): str(c) for u, i, c in
                  zip(poi["utterance_id"], poi["reference_unit_index"], poi["category"])}
    correct = {(str(u), int(i)): bool(c) for u, i, c in
               zip(poi["utterance_id"], poi["reference_unit_index"], poi["correct"])}
    rows: list[dict[str, Any]] = []
    by_utterance = {str(u): g for u, g in spans.groupby("utterance_id")}
    step = float(bundle.encoder_step_sec)

    for _, row in manifest.iterrows():
        utterance = str(row["utterance_id"])
        if utterance not in by_utterance:
            continue
        norm, units = normalize_and_segment(row["transcript_raw"])
        text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        offsets = token_char_offsets(tokenizer, text_ids)
        char_spans = D3.unit_char_spans(norm, units)
        valid = bundle.valid_frames(float(row["duration_sec"]))
        for _, span in by_utterance[utterance].iterrows():
            lo, hi = D3._frame_range(span["start_sec"], span["end_sec"], step, valid)
            for unit_index in span["unit_indices"]:
                key = (utterance, int(unit_index))
                if correct.get(key) is not False:
                    continue                       # baseline-error units only
                if int(unit_index) >= len(char_spans):
                    continue
                token_index = D3.first_token_for_unit(offsets, char_spans[int(unit_index)][0])
                if token_index is None or token_index >= len(text_ids):
                    continue
                rows.append({
                    "utterance_id": utterance,
                    "dialogue_id": str(span["dialogue_id"]),
                    "reference_unit_index": int(unit_index),
                    "category": categories.get(key, "unknown"),
                    "is_deletion": categories.get(key) == "deletion",
                    "gold_token_id": int(text_ids[token_index]),
                    "token_index": int(token_index),
                    "query_position": int(len(prefix) + token_index - 1),
                    "span_frame_lo": int(lo), "span_frame_hi": int(hi),
                    "valid_frames": int(valid),
                    "span_duration_ms": float(span["span_duration_ms"]),
                })
    return pd.DataFrame(rows)


def stratum_mask(targets: pd.DataFrame, stratum: str) -> pd.Series:
    if stratum == "all_errors":
        return pd.Series(True, index=targets.index)
    if stratum == "substitution_only":
        return targets["category"] != "deletion"
    if stratum == "wrong_language_substitution":
        return targets["category"] == "wrong_language_substitution"
    raise ValueError(f"unknown stratum {stratum!r}")


# ---------------------------------------------------------------------------
# the label-permuted null: rebuilt through the identical pipeline
# ---------------------------------------------------------------------------

def label_permuted_directions(contrasts: np.ndarray, meta: pd.DataFrame, *,
                              draws: int, seed: int,
                              basis: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Shuffle EN/ZH labels within nuisance strata, then rebuild identically.

    A label swap turns a pair's contribution from (positive - negative) into
    (negative - positive), i.e. a sign flip of the stored contrast.  Everything downstream -- ridge residualisation,
    clipping, dialogue-balanced averaging, and the prompt projection when one
    applies -- is the same code the real direction went through, so the only
    difference between the null and the real direction is the labelling.
    """
    strata = pd.qcut(meta["span_duration_sec"].rank(method="first"), 4,
                     labels=False, duplicates="drop").to_numpy()
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for draw in range(draws):
        signs = np.ones(len(meta), dtype=float)
        for stratum in np.unique(strata):
            index = np.flatnonzero(strata == stratum)
            flip = rng.permutation(len(index)) < (len(index) // 2)
            signs[index[flip]] = -1.0
        permuted = np.asarray(contrasts, dtype=float) * signs[:, None]
        assembled = D3.assemble(permuted, meta, basis=basis)
        out.append({"draw": int(draw), "direction": assembled["direction"],
                    "flipped": int((signs < 0).sum()),
                    "ridge_energy_removed": assembled["ridge"]["energy_fraction_removed"]})
    return out


def matched_norm_random(direction: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    """Random directions with the same norm as `direction`."""
    rng = np.random.default_rng(seed)
    norm = float(np.linalg.norm(direction))
    out = []
    for _ in range(draws):
        vector = rng.standard_normal(direction.shape[0])
        out.append(vector / max(np.linalg.norm(vector), 1e-12) * norm)
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _gain(shape: tuple[int, int], positions: Sequence[Sequence[int]]) -> torch.Tensor:
    gain = torch.zeros(shape, dtype=torch.float32)
    for row, where in enumerate(positions):
        for index in where:
            if 0 <= int(index) < shape[1]:
                gain[row, int(index)] = 1.0
    return gain


@torch.inference_mode()
def score_batch(bundle, rows: pd.DataFrame, targets: pd.DataFrame, *,
                encoder: tuple[int, np.ndarray, float] | None = None,
                decoder: tuple[int, np.ndarray, float] | None = None,
                encoder_frames: dict[str, tuple[int, int]] | None = None,
                decoder_positions: dict[str, list[int]] | None = None
                ) -> pd.DataFrame:
    """One teacher-forced forward for one condition; outcomes at each target.

    C11 ordering is structural rather than scheduled: the encoder hook fires
    inside the encoder forward, cross-attention then consumes the steered
    encoder states unmodified, and the decoder hook fires afterwards on the
    post-cross-attention residual. Encoder -> normal cross-attention -> decoder
    is therefore the only order this composition can run in.
    """
    from ..data.alignment import build_prefix
    from ..data.normalize import normalize_and_segment
    from ..models.generation import teacher_forced_forward

    tokenizer = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language="zh")
    eot = tokenizer.eos_token_id
    sequences, order = [], []
    for _, row in rows.iterrows():
        norm, _ = normalize_and_segment(row["transcript_raw"])
        text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
        sequences.append(list(prefix) + list(text_ids) + [eot])
        order.append(str(row["utterance_id"]))

    max_len = max(len(s) for s in sequences)
    frame_gain = None
    if encoder is not None:
        layer, vector, rho = encoder
        n_frames = int(bundle.max_encoder_frames)
        spans = [list(range(*(encoder_frames or {}).get(u, (0, 0)))) for u in order]
        frame_gain = _gain((len(order), n_frames), spans)
    position_gain = None
    if decoder is not None:
        position_gain = _gain((len(order), max_len),
                              [(decoder_positions or {}).get(u, []) for u in order])

    contexts = []
    if encoder is not None:
        layer, vector, rho = encoder
        direction = torch.as_tensor(vector, dtype=torch.float32, device=bundle.device)
        contexts.append(EncoderSteeringHook(bundle, int(layer), direction,
                                            alpha=float(rho), scale=1.0,
                                            gain=frame_gain))
    if decoder is not None:
        layer, vector, rho = decoder
        direction = torch.as_tensor(vector, dtype=torch.float32, device=bundle.device)
        contexts.append(DecoderPostCrossAttnSteeringHook(
            bundle, int(layer), direction, alpha=float(rho), scale=1.0,
            gain=position_gain, steer_prefill=True))

    entered = []
    try:
        for context in contexts:
            entered.append(context.__enter__())
        out, padded, _mask = teacher_forced_forward(
            bundle, rows["audio_path"].tolist(), sequences)
        logits = out.logits.float()
    finally:
        for context in reversed(entered):
            context.__exit__(None, None, None)
    assert_no_site_hooks(bundle)

    logprobs = torch.log_softmax(logits, dim=-1)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
    index = {u: i for i, u in enumerate(order)}
    records = []
    for _, target in targets.iterrows():
        b = index.get(str(target["utterance_id"]))
        if b is None:
            continue
        q = int(target["query_position"])
        if q < 0 or q >= logprobs.shape[1]:
            continue
        row_lp = logprobs[b, q]
        gold = int(target["gold_token_id"])
        gold_lp = float(row_lp[gold])
        ranked = torch.argsort(row_lp, descending=True)
        rank = int((ranked == gold).nonzero()[0, 0]) + 1
        top = row_lp.clone()
        top[gold] = -float("inf")
        records.append({
            "utterance_id": str(target["utterance_id"]),
            "reference_unit_index": int(target["reference_unit_index"]),
            "gold_logprob": gold_lp,
            "gold_rank": rank,
            "margin": gold_lp - float(top.max()),
            "entropy": float(entropy[b, q]),
        })
    del out, logits, logprobs
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# wild cluster bootstrap
# ---------------------------------------------------------------------------

def wild_cluster_ci(values: np.ndarray, clusters: Sequence[str], *,
                    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                    ci: float = BOOTSTRAP_CI) -> dict[str, Any]:
    """Wild cluster bootstrap of a mean, Rademacher weights, t(G-1) critical values."""
    from scipy import stats

    y = np.asarray(values, dtype=float)
    labels = np.asarray([str(c) for c in clusters])
    groups = sorted(set(labels))
    G = len(groups)
    if G < 2 or not len(y):
        return {"G": G, "n": int(len(y)), "mean": float(np.mean(y)) if len(y) else None,
                "ci_low": None, "ci_high": None, "draws": 0}
    mean = float(np.mean(y))
    index = {g: np.flatnonzero(labels == g) for g in groups}
    residual = y - mean
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(draws, dtype=float)
    for d in range(draws):
        weights = rng.choice([-1.0, 1.0], size=G)
        drawn = y.copy()
        for k, g in enumerate(groups):
            drawn[index[g]] = mean + residual[index[g]] * weights[k]
        cluster_means = np.array([drawn[index[g]].mean() for g in groups])
        se = cluster_means.std(ddof=1) / np.sqrt(G)
        stats_boot[d] = (drawn.mean() - mean) / se if se > 0 else 0.0
    cluster_means = np.array([y[index[g]].mean() for g in groups])
    se = cluster_means.std(ddof=1) / np.sqrt(G)
    critical = float(np.nanquantile(np.abs(stats_boot), ci))
    t_critical = float(stats.t.ppf(0.5 + ci / 2, df=G - 1))
    return {"G": G, "n": int(len(y)), "mean": mean, "se": float(se),
            "t_statistic": float(mean / se) if se > 0 else None,
            "bootstrap_critical_value": critical,
            "t_critical_value_G_minus_1": t_critical,
            "ci_low": mean - critical * se, "ci_high": mean + critical * se,
            "excludes_zero": bool(abs(mean) > critical * se) if se > 0 else False,
            "draws": int(draws), "seed": int(seed),
            "min_attainable_p": float(2.0 ** (-G))}


def paired_contrast(treated: pd.DataFrame, control: pd.DataFrame, outcome: str,
                    clusters: pd.DataFrame) -> dict[str, Any]:
    """Per-unit difference treated-minus-control, bootstrapped on dialogues."""
    key = ["utterance_id", "reference_unit_index"]
    merged = treated[key + [outcome]].merge(
        control[key + [outcome]], on=key, suffixes=("_t", "_c"))
    merged = merged.merge(clusters[key + ["dialogue_id"]], on=key, how="left")
    difference = merged[f"{outcome}_t"] - merged[f"{outcome}_c"]
    return wild_cluster_ci(difference.to_numpy(), merged["dialogue_id"])


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


def _wrong_frames(lo: int, hi: int, valid: int, rng) -> tuple[int, int]:
    """A same-length window elsewhere in the same utterance."""
    width = max(1, hi - lo)
    if valid <= width:
        return 0, width
    for _ in range(20):
        start = int(rng.integers(0, max(1, valid - width)))
        if abs(start - lo) >= width:
            return start, start + width
    return 0, width


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
        raise SystemExit(f"refusing a smoke generation as a direction source: {directions_root}")

    # Day 3 directions are SPLIT format and are verified by BYTES, not identity.
    #
    # Identity asks whether an artifact was produced by the code now running.
    # For an artifact consumed by newly written code that is necessarily false:
    # writing this module is itself what changes code_config_sha256. The check
    # would be self-defeating, so the format is classified and recorded, the
    # bytes are authenticated against the sidecar, and identity is not required
    # -- the same standard `lss_v2_candidates._parents` applies to role
    # manifests. The split gate stays in force for what this session publishes.
    npz_path = directions_root / "directions.npz"
    side = manifest_mod.load(npz_path)
    split_format, format_problem = manifest_mod._identity_format(
        dict(side.get("identity") or {}))
    if split_format != "split":
        raise SystemExit(f"direction artifact is not split format: "
                         f"{split_format} {format_problem}")
    byte_check = manifest_mod.verify(npz_path, cfg=None, require_identity=False)
    if not byte_check["ok"]:
        raise SystemExit(f"direction artifact failed byte verification: {byte_check}")
    identity_note = {
        "classification": split_format,
        "byte_verified_sha256": side["sha256"],
        "identity_required": False,
        "reason": ("identity asks whether the artifact was produced by the code "
                   "now running; writing this consuming module is what changed "
                   "code_config_sha256, so the check would be self-defeating. "
                   "Bytes are the property that matters for reading."),
        "recorded_code_config_sha256": side["identity"].get("code_config_sha256"),
    }
    directions = dict(np.load(npz_path))

    root = Path(cfg["experiment"]["output_root"])
    candidate_path = root / "alignments" / "candidates_all.parquet"
    role_root = Path(cfg["v2_namespace"]["role_root"])
    poi_root = root.parent.parent / "baselines" / "generation_001"
    legacy = [candidate_path]
    for role in conv.DEVELOPMENT_ROLES:
        legacy += [role_root / f"role_{role}.parquet", poi_root / f"poi_{role}.parquet"]
    for path in legacy:
        check = manifest_mod.verify(path, cfg=None, require_identity=False)
        if not check["ok"]:
            raise SystemExit(f"{path.name} failed byte verification: {check}")

    candidates, verdict = conv.read_development_candidates(candidate_path)
    conv.assert_development_only(candidates)
    manifests = [pd.read_parquet(role_root / f"role_{r}.parquet")
                 for r in conv.DEVELOPMENT_ROLES]
    pois = [pd.read_parquet(poi_root / f"poi_{r}.parquet").rename(
        columns={"poi_index": "reference_unit_index"}) for r in conv.DEVELOPMENT_ROLES]
    manifest = pd.concat(manifests, ignore_index=True)
    poi = pd.concat(pois, ignore_index=True)
    dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                           manifest["dialogue_id"].astype(str)))
    spans = D3.attach_baseline_status(
        D3.build_spans(candidates, dialogue_of=dialogue_of), poi)
    develop = spans[spans["role"] == "D-dev-select"].copy()
    construct = spans[(spans["role"] == "D-construct") & spans["all_correct"]].copy()

    subset = {
        "spans": int(len(develop)),
        "error_units": int(develop["units_error"].sum()),
        "dialogues": int(develop["dialogue_id"].nunique()),
        "span_duration_ms": {
            "min": float(develop["span_duration_ms"].min()),
            "median": float(develop["span_duration_ms"].median()),
            "p90": float(develop["span_duration_ms"].quantile(0.90)),
            "max": float(develop["span_duration_ms"].max())},
    }
    plan = {"frozen_configuration": FROZEN_CONFIG,
            "subset_D_dev_select": subset,
            "directions_source": str(directions_root),
            "directions_identity_format": split_format,
            "directions_identity_note": identity_note,
            "smoke_excluded": True,
            "input_identity_format": "legacy (byte-verified; identity never attempted)",
            "output_identity_format": "split",
            "destination": str(target_dir)}

    if not args.execute:
        print(json.dumps({"state": "validated_no_execution", "models_loaded": False,
                          "artifacts_written": [], "plan": plan}, indent=2, default=str))
        return 0

    log = setup_logging(level="INFO")
    bundle = load_whisper(cfg)
    targets = build_targets(bundle, candidates, develop, poi, manifest)
    if args.limit:
        keep = list(dict.fromkeys(targets["utterance_id"]))[: int(args.limit)]
        targets = targets[targets["utterance_id"].isin(keep)]
    utterances = manifest[manifest["utterance_id"].isin(set(targets["utterance_id"]))]
    utterances = utterances.drop_duplicates("utterance_id").reset_index(drop=True)
    log.info("screen over %d targets in %d utterances, G=%d",
             len(targets), len(utterances), targets["dialogue_id"].nunique())

    # per-pair contrasts for the label-permuted null, rebuilt through D3's pipeline
    runs = D3.matrix_runs(candidates)
    build_manifest = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
    e_meta, e_raw = D3.site_e_contrasts(bundle, build_manifest, construct, runs,
                                        [15, 23, 27, 31], batch_size=4, log=log)
    d_meta, d_raw = D3.site_d_contrasts(bundle, build_manifest, construct,
                                        [8, 16, 24, 31], batch_size=4, log=log)

    rng = np.random.default_rng(CONTROL_SEED)
    conditions: list[dict[str, Any]] = [{"condition": "C00", "kind": "baseline"}]
    for layer in (15, 23, 27, 31):
        vector = directions[f"site_e_layer{layer}_{SITE_E_POOLING}"]
        nulls = label_permuted_directions(
            np.vstack(e_raw["vectors"][(layer, SITE_E_POOLING)]), e_meta,
            draws=CONTROL_LABEL_PERMUTED_DRAWS, seed=CONTROL_SEED + layer)
        randoms = matched_norm_random(vector, CONTROL_RANDOM_DRAWS, CONTROL_SEED + layer)
        for rho in RHO_GRID:
            conditions.append({"condition": "C10", "site": "E", "layer": layer,
                               "rho": rho, "kind": "correct", "encoder": vector})
            conditions.append({"condition": "C10", "site": "E", "layer": layer,
                               "rho": rho, "kind": "wrong_location", "encoder": vector})
            conditions.append({"condition": "C10", "site": "E", "layer": layer,
                               "rho": rho, "kind": "opposite_sign", "encoder": -vector})
            for k, r in enumerate(randoms):
                conditions.append({"condition": "C10", "site": "E", "layer": layer,
                                   "rho": rho, "kind": f"random_{k}", "encoder": r})
            for k, n in enumerate(nulls):
                conditions.append({"condition": "C10", "site": "E", "layer": layer,
                                   "rho": rho, "kind": f"label_permuted_{k}",
                                   "encoder": n["direction"],
                                   "null_norm_ratio": float(
                                       np.linalg.norm(n["direction"]) / max(np.linalg.norm(vector), 1e-12))})
    for layer in (8, 16, 24, 31):
        vector = directions[f"site_d_layer{layer}"]
        nulls = label_permuted_directions(
            np.vstack(d_raw["vectors"][layer]), d_meta,
            draws=CONTROL_LABEL_PERMUTED_DRAWS, seed=CONTROL_SEED + 100 + layer)
        randoms = matched_norm_random(vector, CONTROL_RANDOM_DRAWS, CONTROL_SEED + 100 + layer)
        for rho in RHO_GRID:
            conditions.append({"condition": "C01", "site": "D", "layer": layer,
                               "rho": rho, "kind": "correct", "decoder": vector})
            conditions.append({"condition": "C01", "site": "D", "layer": layer,
                               "rho": rho, "kind": "wrong_location", "decoder": vector})
            conditions.append({"condition": "C01", "site": "D", "layer": layer,
                               "rho": rho, "kind": "opposite_sign", "decoder": -vector})
            for k, r in enumerate(randoms):
                conditions.append({"condition": "C01", "site": "D", "layer": layer,
                                   "rho": rho, "kind": f"random_{k}", "decoder": r})
            for k, n in enumerate(nulls):
                conditions.append({"condition": "C01", "site": "D", "layer": layer,
                                   "rho": rho, "kind": f"label_permuted_{k}",
                                   "decoder": n["direction"],
                                   "null_norm_ratio": float(
                                       np.linalg.norm(n["direction"]) / max(np.linalg.norm(vector), 1e-12))})
    for enc_layer, dec_layer in C11_LAYER_PAIRS:
        e_vector = directions[f"site_e_layer{enc_layer}_{SITE_E_POOLING}"]
        d_vector = directions[f"site_d_layer{dec_layer}"]
        for rho in RHO_GRID:
            conditions.append({"condition": "C11", "site": "ED",
                               "layer": f"{enc_layer}+{dec_layer}", "rho": rho,
                               "kind": "correct", "encoder": e_vector,
                               "decoder": d_vector})
            conditions.append({"condition": "C11", "site": "ED",
                               "layer": f"{enc_layer}+{dec_layer}", "rho": rho,
                               "kind": "wrong_location", "encoder": e_vector,
                               "decoder": d_vector})

    frames = {str(t["utterance_id"]): (int(t["span_frame_lo"]), int(t["span_frame_hi"]))
              for _, t in targets.iterrows()}
    positions = {}
    for utterance, group in targets.groupby("utterance_id"):
        positions[str(utterance)] = [int(v) for v in group["query_position"]]
    wrong_frames, wrong_positions = {}, {}
    for utterance, group in targets.groupby("utterance_id"):
        first = group.iloc[0]
        wrong_frames[str(utterance)] = _wrong_frames(
            int(first["span_frame_lo"]), int(first["span_frame_hi"]),
            int(first["valid_frames"]), rng)
        span_positions = [int(v) for v in group["query_position"]]
        wrong_positions[str(utterance)] = [max(0, p - 7) for p in span_positions]

    results: list[pd.DataFrame] = []
    total = len(conditions)
    for ci, condition in enumerate(conditions):
        wrong = condition["kind"] == "wrong_location"
        per_batch = []
        for start in range(0, len(utterances), args.batch_size):
            batch = utterances.iloc[start:start + args.batch_size]
            keys = set(batch["utterance_id"].astype(str))
            sub = targets[targets["utterance_id"].isin(keys)]
            scored = score_batch(
                bundle, batch, sub,
                encoder=((condition["layer"] if condition["condition"] != "C11"
                          else int(str(condition["layer"]).split("+")[0]),
                          condition["encoder"], condition["rho"])
                         if "encoder" in condition else None),
                decoder=((condition["layer"] if condition["condition"] != "C11"
                          else int(str(condition["layer"]).split("+")[1]),
                          condition["decoder"], condition["rho"])
                         if "decoder" in condition else None),
                encoder_frames=(wrong_frames if wrong else frames),
                decoder_positions=(wrong_positions if wrong else positions))
            per_batch.append(scored)
        frame = pd.concat(per_batch, ignore_index=True)
        for key in ("condition", "site", "layer", "rho", "kind", "null_norm_ratio"):
            value = condition.get(key)
            # `layer` is an int for C10/C01 and "enc+dec" for C11; keep one dtype
            frame[key] = str(value) if key == "layer" and value is not None else value
        results.append(frame)
        if ci % 25 == 0:
            log.info("  condition %d/%d: %s %s rho=%s %s", ci + 1, total,
                     condition["condition"], condition.get("layer"),
                     condition.get("rho"), condition["kind"])

    scores = pd.concat(results, ignore_index=True)
    scores = scores.merge(targets[["utterance_id", "reference_unit_index",
                                   "dialogue_id", "category", "is_deletion"]],
                          on=["utterance_id", "reference_unit_index"], how="left")
    target_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(target_dir / "oracle_scores.parquet", index=False)
    targets.to_parquet(target_dir / "oracle_targets.parquet", index=False)
    payload = {"frozen_configuration": FROZEN_CONFIG, "plan": plan,
               "conditions_run": int(total), "targets": int(len(targets)),
               "utterances": int(len(utterances)),
               "selects_layer_or_strength": False, "evaluates_gate": False}
    (target_dir / "oracle_screen_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    parent = verdict.get("manifest")
    for name in ("oracle_scores.parquet", "oracle_targets.parquet",
                 "oracle_screen_report.json"):
        manifest_mod.publish(target_dir / name, stage=STAGE, cfg=cfg,
                             parents=[parent] if parent else (),
                             schema="lss_v2r3_oracle_screen_v1",
                             extra={"frozen_configuration": FROZEN_CONFIG,
                                    "evaluates_gate": False})
    print(json.dumps({"state": "completed", "output": str(target_dir),
                      "conditions": total, "targets": int(len(targets)),
                      "rows": int(len(scores))}, indent=2, default=str))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
