# D2 — pre-registered decision rules

These rules were written before any D2 calibration or training outcome existed. They are implemented in
`reft/d2_edit_locality_evaluate.py` (`components`, `decide`, `RECOMMEND`, reusing `primary_gain`, `en_guard`, `mer_guard`,
`component_gain`, `tw_support`, `strong` from `reft/d1_regionkl_evaluate.py`) and unit-tested in
`tests/reft/test_d2_edit_locality.py`. The constants are in `configs/reft/d2_edit_locality_frozen.yaml` (`decision`).

## Inputs

**Checkpoints:**
- D2_s: selected by the unchanged P2-1 rule;
- GoldKL_s: the historical P2-1b `KL_seed{s}` selected checkpoints, recomputed with the same code. Recomputation must
  reproduce `p2_1b/evaluation.json` at every epoch.

**Evaluation:** dev150, frozen-LID beam-5, against the pinned B0. s ∈ {42, 43, 44}.

**Values** are read from the evaluation artifacts at full precision, never from rounded tables. For orientation only, the
historical GoldKL values are new Mandarin errors 53/48/50, long switch-back dropped units 30/31/26, English net 34/41/35
(sum 110) and genuine-CS MER 0.1124/0.1105/0.1103.

**Definitions:**

| Term | Definition |
|---|---|
| `zh_new` | new Mandarin errors (B0-correct substantive Mandarin reference units now wrong); retention = 1 − zh_new / 4950 |
| `en_net` | substantive English corrections minus corruptions against B0 |
| `cs_mer` | genuine-CS MER |
| `tw_new` | new errors among B0-correct units of the target window (first 7 substantive Mandarin units after each EN→ZH switch) |
| `sb` | long switch-back dropped units (deletion runs ≥ 5 starting after substantive English) |

mean(·) is over the three seeds.

## A — preservation gate (both required)

| Condition | Requirement |
|---|---|
| A1 | mean zh_new(D2) ≤ mean zh_new(GoldKL) − 5 |
| A2 | zh_new(D2_s) < zh_new(GoldKL_s) in ≥ 2 of 3 matched seeds |

## B — English guardrail

Σ_s en_net(D2_s) / Σ_s en_net(GoldKL_s) ≥ 0.90 (≥ 99 of 110 under the current control). If the GoldKL sum is ≤ 0, B fails.

## C — genuine-CS MER guardrail (strict)

mean_s cs_mer(D2_s) ≤ mean_s cs_mer(GoldKL_s), full precision, no tolerance.

## D — switch-back evidence (reported, outside the promotion rule)

TW new errors and long-SB dropped units with denominators and per-seed consistency. TW support holds iff mean tw_new(D2) <
mean tw_new(GoldKL) and D2 is lower in ≥ 2 seeds.

## E — strong preservation (reported only)

zh_new ≤ 49 (retention ≥ 0.99) per seed, for D2 and GoldKL.

## Secondary preservation components (used only for MIXED)

For key ∈ {tw_new, sb}: passes iff mean(D2) ≤ mean(GoldKL) − 5 and D2 is strictly lower in ≥ 2 seeds.

## Representation measures

Edit-energy, edit-norm, selectivity and teacher-forced KL changes are descriptive. A reduction of representation norms alone
can never trigger promotion or MIXED.

## Integrity (all required, else `D2_INVALID`)

- the D2.0 calibration verdict is `D2_CALIBRATION_OK` (not a dry run) and every run used the frozen μ;
- the newest preflight is `D2_READY_FOR_TRAINING` (not a dry run);
- verification: 3 runs × 3 epochs reload and re-decode 20/20, hashes match, 5,122 parameters;
- three runs with epochs 1–3 and 96 updates each;
- every run starts from its seed-specific identity (W = R, b = 0) with `base` equal to the GoldKL run of the same seed;
- zero-init identity 150/150 for every run (D2 and GoldKL);
- backbone weights unchanged;
- GoldKL recomputation reproduces `p2_1b/evaluation.json`;
- per epoch, D2 mask positions, supervised tokens and updates equal the matched GoldKL run;
- orthogonality error ≤ 1e-5 at every step;
- the mask-metadata hash equals the frozen hash in every run;
- the edit term is active in every epoch (edit-mask positions > 0 and μ·L_edit > 0);
- the learning rate is constant at 5e-4.

## Terminal verdict (strict precedence)

1. `D2_INVALID`: integrity fails.
2. `D2_INCREMENTAL_PRESERVATION_GAIN`: A ∧ B ∧ C.
3. `D2_MIXED_TRADEOFF`: A holds but B or C fails; or A fails but a secondary preservation component (TW or SB) passes.
4. `D2_NO_INCREMENTAL_GAIN`: otherwise.

**Fixed-epoch-3 comparisons** apply the same component functions descriptively and never change the verdict. Thresholds
are never changed after outputs are seen; the verdict is never upgraded by hand. This is a pilot-screening verdict on an
exposed panel, not independent confirmation. Locality-specific causality is not claimed without a matched
generic-shrinkage control.

## Recommendation (one; it is not authorization)

| Verdict | Recommendation |
|---|---|
| INCREMENTAL_PRESERVATION_GAIN | `GO_LOCALITY_SPECIFICITY_CONTROL`: a generic-shrinkage control (training-only-calibrated inference attenuation h + αΔh, or a global edit penalty with the same gradient-scale rule), then independent confirmation |
| MIXED_TRADEOFF | `CHARACTERIZE_CORRECTION_PRESERVATION_TRADEOFF`: state which side moved; no automatic follow-up |
| NO_INCREMENTAL_GAIN | `STOP_GOLD_PREFIX_LOCALITY_OBJECTIVE_EXPANSION`: no further weighted KL, locality coefficient or architecture variant by default; the next direction is chosen jointly with a competitive LoRA benchmark and the prefix-history findings |
| INVALID | `HOLD_FOR_ENGINEERING_REPAIR`: name the repair needed |
