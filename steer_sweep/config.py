"""Frozen constants for the two-track study.

Everything a result depends on is declared here, before any result is seen.
The staging selection rule (`SELECTION_RULE`) and the four Track A gates are
hard-coded so they cannot be changed after looking at a table.

Where a value in the task specification disagrees with what the repository
actually contains, BOTH are recorded and the disagreement is surfaced in
`SPEC_DEVIATIONS`. Nothing is silently substituted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "out"

TAINT = "development_only_diagnostic"

# ---------------------------------------------------------------------------
# model and geometry -- asserted at load, never assumed
# ---------------------------------------------------------------------------

BASE_CONFIG = "lss/l1b_candidates_dialogue_v2r3.yaml"

EXPECTED_ENCODER_LAYERS = 32
EXPECTED_DECODER_LAYERS = 32
EXPECTED_D_MODEL = 1280
EXPECTED_ENCODER_FRAMES = 1500
FRAMES_PER_SECOND = 50.0
BACKBONE_PARAMS_NOMINAL = 1_550_000_000      # 1.55 B, the reporting denominator

#: The four forced prefix tokens of section 0.
PREFIX_TOKENS = ("<|startoftranscript|>", "<|{lang}|>", "<|transcribe|>",
                 "<|notimestamps|>")
PREFIX_WIDTH = 4

# ---------------------------------------------------------------------------
# population and anchors -- asserted before any run
# ---------------------------------------------------------------------------

#: Verified against
#: artifacts_dialogue_v2r3/gate_b/generation_001/gate_b_free_decoding.parquet
#: on 2026-08-18: 489 rows at floor 400 ms in the oracle arm, 7 corrections,
#: units_at_risk 6,264, and monolingual (ZH) baseline-correct 5,711 when the
#: CURRENT source is used to recompute retention from the stored text.
ANCHOR_TARGETS = 489
ANCHOR_BASELINE_CORRECT_UNITS = 6_264
ANCHOR_BASELINE_CORRECT_ZH_UNITS = 5_711
ANCHOR_ORACLE_CORRECTIONS = 7
ANCHOR_ORACLE_CORRUPTIONS = 13
ANCHOR_ORACLE_CORRECTION_DIALOGUES = ("CSD0014", "CSD0502")
ANCHOR_DEV_SELECT_UTTERANCES = 300
ANCHOR_DEV_SELECT_DIALOGUES = 20
ANCHOR_DEV_SELECT_TARGET_UTTERANCES = 116
ANCHOR_DEV_CONFIRM_UTTERANCES = 150
ANCHOR_DEV_CONFIRM_DIALOGUES = 10

#: Target eligibility floor. The frozen v2r3 conservative tier.
TARGET_FLOOR_MS = 400.0

DEV_SELECT = "D-dev-select"
DEV_CONFIRM = "D-dev-confirm"
CONSTRUCT = "D-construct"
LOC_TRAIN = "loc-train"
UTIL_TRAIN = "util-train"
LOCKED_SPLITS = ("D-test",)

# ---------------------------------------------------------------------------
# Track A -- sites, directions, coverage, alpha, layers
# ---------------------------------------------------------------------------

SITE_ENCODER = "encoder"
SITE_DECODER = "decoder"
SITE_BOTH = "encoder+decoder"

KIND_V_NAT = "v_nat"
KIND_V_NAT_RELAXED = "v_nat_relaxed"
KIND_V_PROMPT = "v_prompt"

#: v_prompt is a decoder-only object: Whisper's encoder never sees the language
#: token, so a prompt-derived direction has no defined meaning at an encoder
#: site. Requesting one raises (requirement 1.6).
DECODER_ONLY_KINDS = (KIND_V_PROMPT,)

COVERAGE_GLOBAL = "global"
COVERAGE_LOCAL = "local"

SCALE_ACT_NORM = "act_norm"
SCALE_UNIT = "unit"
DEFAULT_SCALE_MODE = SCALE_ACT_NORM

ALPHA_GRID = (0.25, 0.5, 1.0, 2.0, 3.0)

#: Layer grids as specified in the task.
ENCODER_SINGLE_LAYERS = (8, 15, 20, 27)
ENCODER_MULTI_LAYERS = ((15, 20), (8, 15, 20, 27))
DECODER_SINGLE_LAYERS = (8, 16, 24, 28)
DECODER_MULTI_LAYERS = ((16, 20), (8, 16, 24, 28))

#: Stage A mid layers.
MID_ENCODER_LAYER = 15
MID_DECODER_LAYER = 16

#: Combination modes for site x direction cell 5.
COMBO_WEIGHTED = "weighted"
COMBO_RANK2 = "rank2"
#: Coarse 3x3 grid over (alpha on v_nat, beta on v_prompt).
COMBO_WEIGHT_GRID = (0.5, 1.0, 2.0)

# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

BASELINES: tuple[dict[str, Any], ...] = (
    {"name": "C00", "language": "zh", "num_beams": 1,
     "note": "default decode, greedy, forced <|zh|>"},
    {"name": "C00_beam5", "language": "zh", "num_beams": 5,
     "note": "as C00 with num_beams=5; the ONLY valid comparator for beam runs"},
    {"name": "force_en", "language": "en", "num_beams": 1,
     "note": "forced <|en|> prefix -- the ceiling of Whisper's native "
             "language control and the reviewer challenge to any prompt result"},
    {"name": "force_zh", "language": "zh", "num_beams": 1,
     "note": "forced <|zh|> prefix, stated explicitly"},
)
PRIMARY_BASELINE = "C00"
BEAM_BASELINE = "C00_beam5"

# ---------------------------------------------------------------------------
# gates (3.7) -- thresholds, fixed
# ---------------------------------------------------------------------------

GATES = {
    "G1_delta_pier": {"rule": "delta_pier > 0 AND bootstrap ci_low > 0"},
    "G2_correction_ratio": {"rule": "corrections : corruptions >= 2 : 1",
                            "threshold": 2.0},
    "G3_dialogue_coverage": {"rule": "dialogues with >=1 correction >= 5",
                             "threshold": 5, "of": ANCHOR_DEV_SELECT_DIALOGUES},
    "G4_zh_retention": {"rule": "ZH retention >= 0.99",
                        "threshold": 0.99,
                        "denominator": ANCHOR_BASELINE_CORRECT_ZH_UNITS},
}

#: Decode health. A run that fails any of these is INVALID, not negative, and
#: its PIER must not be interpreted.
DECODE_HEALTH = {
    "median_length_ratio_range": (0.7, 1.3),
    "empty_output_rate_max": 0.01,
    "max_new_tokens_hit_rate_max": 0.02,
}

#: Between-stage selection, hard-coded so it cannot move after seeing results.
#: The PRIMARY key is dialogue coverage, not delta PIER: the prior run's seven
#: corrections came from two of twenty dialogues, so its effective cluster
#: count was two. Rank against that failure mode.
SELECTION_RULE = {
    "step_1": "drop runs with unhealthy decode (INVALID)",
    "step_2": "drop runs with zh_retention < 0.99",
    "step_3": ("sort by dialogues_with_correction DESC, "
               "then corrections_to_corruptions DESC, "
               "then delta_pier DESC"),
    "primary_key": "dialogues_with_correction",
}

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_CI = 0.95

# ---------------------------------------------------------------------------
# Track B -- training budget, identical across every arm (4.5)
# ---------------------------------------------------------------------------

TRAIN_BUDGET = {
    "max_steps": 2000,
    "max_epochs": 3,
    "batch_size": 8,
    "grad_accum": 2,
    "eval_every": 250,
    "early_stop_patience": 3,
    "early_stop_metric": "dev_mer",
    "lr_grid": (1e-4, 5e-4, 1e-3),
    "lr_selection": "best dev MER, same protocol every arm",
    "warmup_fraction": 0.10,
    "precision": "bfloat16 autocast, fp32 master weights",
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "betas": (0.9, 0.99),
    "eps": 1e-6,
}

LAMBDA_LID_GRID = (0.1, 0.3, 0.5)
LAMBDA_TUNED_ON = "B1"
LR_TUNED_ON = "B1"
DEFAULT_RANK = 4
RANK_SWEEP = (1, 8)
B1_SEEDS = (20260818, 20260819, 20260820)

#: Training-time early-stopping / lr-selection dev set, dialogue-stratified
#: from `router-calib` (3,031 utterances). Section 4.5's `eval_every=250`
#: means up to 8 full greedy-decode passes over this set PER training
#: invocation, and there are ~14 of them across lambda search, lr search, and
#: the arms -- evaluating the FULL role at every one of those would turn "a
#: fast observation pass" into roughly 13 hours of eval overhead alone at
#: this model's decode rate. 150 utterances keeps each pass in the tens of
#: seconds while still drawing from all 8 router-calib dialogues. The FINAL
#: per-arm report (4.7) is unaffected: it always scores the full D-dev-select
#: population, once, after training.
TRACKB_DEV_EVAL_LIMIT = 150
DATA_EFFICIENCY_FRACTIONS = (0.10, 0.25, 0.50, 1.00)

#: Site/layer for B0-B2. Track A Stage D supplies this; if Track A has not
#: finished, this default is used and the reason is logged (4.2).
TRACKB_DEFAULT_SITE = SITE_DECODER
TRACKB_DEFAULT_LAYERS = (16,)
TRACKB_DEFAULT_REASON = ("Track A Stage D unavailable at Track B launch; "
                         "decoder layer 16 is the prior causal-leverage site "
                         "(delta gold logprob +0.1057 [+0.0466, +0.1649], G=19)")

ARMS = ("B0", "B1", "B2", "B3-reimpl", "B4")

#: AGA reimplementation constants, read from the reference repository
#: (bobbiaditya/Attention-Guided-Adaptation-for-Code-Switching-Speech-Recognition)
#: and the paper (arXiv:2312.08856). See steer_sweep/trackb/aga.py for the
#: fidelity note.
AGA = {
    "adapter_bottleneck_divisor": 4,
    "adapter_sites": ("post_self_attn", "post_mlp"),
    "adapter_has_layernorm": True,
    "stage1": {"encoder_adapters": True, "decoder_adapters": False,
               "cs_weight": 0.0},
    "stage2": {"encoder_adapters": True, "decoder_adapters": True,
               "cs_weight": 0.01, "c_val_attention": 0.6,
               "head_percentage": 100.0},
    "paper_trainable_fraction": 0.056,
    "fidelity_factor_tolerance": 2.0,
    "head_selection": {
        "criterion": ("per decoder self-attention head, per utterance: "
                      "sum of attention mass on prompt key positions 1:3 "
                      "versus mass on position 0 plus positions 3:; the head "
                      "is counted when the former exceeds the latter"),
        "ranking": "descending count over the training split",
        "reference_pickle": "attention_count_whispernoft_new.pkl",
        "computed_on": "CS-Dialogue loc-train (NOT SEAME)",
    },
    "time_box_days": 3,
}

# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

SEEDS = {
    "global": 20260818,
    "bootstrap": BOOTSTRAP_SEED,
    "direction_random_init": 20260821,
    "data_subsample": 20260822,
    "dev_eval_subsample": 20260823,
}

# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

SMOKE = {"utterances": 8, "train_steps": 20, "eval_every": 10,
         "bootstrap_draws": 200}

# ---------------------------------------------------------------------------
# Where the specification and the repository disagree. Recorded, not resolved
# silently; reproduced verbatim into out/SUMMARY.md.
# ---------------------------------------------------------------------------

SPEC_DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "no_steer_sweep_directory",
        "spec": "a partial harness exists at steer_sweep/ (adapters, hooks, "
                "metrics, sweep, preflight, run.py)",
        "repository": "no such directory exists anywhere on this filesystem. "
                      "The real partial harness is src/csasr/steering/ "
                      "(encoder_hook, decoder_hook, masks), src/csasr/models/"
                      "hooks.py, src/csasr/lss/sites.py and the frozen v2r3 "
                      "experiment drivers.",
        "resolution": "this package is written at steer_sweep/ as asked and "
                      "builds on the existing modules; none of them is rewritten.",
    },
    {
        "id": "direction_store_lacks_v_prompt",
        "spec": "v_prompt is loaded from the existing direction store",
        "repository": "directions.npz holds only site_e_layer{15,23,27,31}_"
                      "{hann,central60,uniform} and site_d_layer{8,16,24,31}. "
                      "There is no prompt direction: the rank-1 prompt subspace "
                      "was PROJECTED OUT of every v_nat contrast at "
                      "construction (prompt_subspace_rank=1).",
        "resolution": "v_prompt is constructed here by the specified method "
                      "(difference-in-means over prompt language-token "
                      "states, <|en|> prefix minus <|zh|> prefix) and stored "
                      "as a NEW artifact under its own kind. It is never "
                      "conflated with v_nat.",
    },
    {
        "id": "layer_grid_not_in_store",
        "spec": "encoder singles {8,15,20,27}, decoder singles {8,16,24,28}",
        "repository": "the store holds encoder {15,23,27,31} and decoder "
                      "{8,16,24,31}. Only encoder 15/27 and decoder 8/16/24 "
                      "overlap the requested grid.",
        "resolution": "missing layers are constructed on D-construct through "
                      "the frozen Day-3 pipeline (v2r3_directions) with the "
                      "identical recipe, and every direction records whether "
                      "it was loaded or reconstructed.",
    },
    {
        "id": "directions_are_not_unit_norm",
        "spec": "unit-norm float32, loaded from the existing direction store",
        "repository": "stored directions are float64 with norms 0.75-5.54 "
                      "(||site_d_layer16|| = 2.414639416389413). The prior "
                      "runs used the RAW norm with rho=1 and norm-preserving "
                      "renormalisation.",
        "resolution": "the sweep unit-normalises to float32 and carries the "
                      "dose in alpha under scale_mode. Preflight check 2 "
                      "instead reproduces the prior convention exactly, "
                      "because its job is to reproduce the prior experiment.",
    },
    {
        "id": "prior_run_did_not_force_zh",
        "spec": "decoder prefix is 4 forced tokens including <|zh|>",
        "repository": "every v2r3 free-decoding stage ran with language=None, "
                      "so Whisper chose the language token itself and the "
                      "prefill width was MEASURED rather than assumed.",
        "resolution": "preflight check 2 replicates language=None because it "
                      "must reproduce the prior 7/489. The study's own "
                      "baselines and every steered cell force the language "
                      "token as specified, and C00 is the comparator.",
    },
    {
        "id": "aga_is_a_loss_not_a_parameter_attachment",
        "spec": "attach trainable parameters to the selected heads; match the "
                "paper's ~5.6% trainable-parameter fraction",
        "repository": "the reference implementation attaches bottleneck "
                      "adapters (Linear d->d/4 -> GELU -> Linear d/4->d, "
                      "residual, + LayerNorm) at EVERY encoder and decoder "
                      "block, and uses head selection to mask an AUXILIARY "
                      "attention-guidance MSE loss "
                      "(loss = cs_weight * loss_cs + loss_att), not to place "
                      "parameters. Adapters at d/4 over 24 blocks of "
                      "whisper-small reproduce ~5.9% -- which is where the "
                      "~5.6% figure comes from.",
        "resolution": "B3-reimpl follows the reference mechanism: adapters "
                      "everywhere, head selection masks the guidance loss. "
                      "The parameter-fraction check is applied to the adapter "
                      "set. Recorded in the fidelity note.",
    },
    {
        "id": "reference_repo_disables_its_own_head_selection",
        "spec": "mirror the head-selection procedure",
        "repository": "in the released code the selected-head mask is "
                      "commented out and replaced by a hard-coded 50% "
                      "`random_onezero` 12x12 mask; the flattening loop also "
                      "swaps the layer and head keys, which is invisible only "
                      "because the matrix is square for whisper-small.",
        "resolution": "the procedure documented in code_util/head_selection.md "
                      "is implemented (not the disabled code path), computed "
                      "on CS-Dialogue, with the key-order defect corrected. "
                      "Both facts are in the fidelity note.",
    },
    {
        "id": "reference_repo_has_no_license",
        "spec": "check the repository for a LICENSE file",
        "repository": "no LICENSE file is present at the repository root "
                      "(checked 2026-08-18).",
        "resolution": "recorded; we reimplement rather than vendor, so the "
                      "obligation is citation of arXiv:2312.08856, which is "
                      "discharged in the summary.",
    },
    {
        "id": "gate_b_artifact_carries_pre_fix_retention",
        "spec": "denominator 5,711 baseline-correct Mandarin units",
        "repository": "the stored gate_b parquet carries 6,264 / 6,251 / 13 "
                      "for monolingual retention because it predates the "
                      "Unit-tagging repair; recomputing from the stored text "
                      "with the CURRENT source gives 5,711 / 5,699 / 12 "
                      "(99.79%), matching the dated document.",
        "resolution": "G4 uses 5,711 and the harness recomputes retention "
                      "with current code rather than reading the stale field.",
    },
)
