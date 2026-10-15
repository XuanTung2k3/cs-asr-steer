# D1 — pre-registered decision rules

These rules were written before any D1 training outcome existed. They are implemented in `reft/d1_regionkl_evaluate.py`
(`primary_gain`, `en_guard`, `mer_guard`, `component_gain`, `tw_support`, `strong`, `decide`, `RECOMMEND`) and
unit-tested in `tests/reft/test_d1_regionkl.py`. The constants are in `configs/reft/d1_regionkl_frozen.yaml`
(`decision`).

## Inputs

**Checkpoints:**
- D1_s: selected by the unchanged P2-1 rule;
- GoldKL_s: the historical P2-1b `KL_seed{s}` selected checkpoints, recomputed with the same code. Recomputation must
  reproduce `p2_1b/evaluation.json` at every epoch.

**Evaluation:** dev150, frozen-LID beam-5, against the pinned B0. s ∈ {42, 43, 44}.

**Values** are read from the evaluation artifacts at full precision, never from rounded tables or hard-coded numbers.
For orientation only, the historical GoldKL values are:

| Seed | zh_new | sb | en_net | CS MER |
|---|---|---|---|---|
| 42 | 53 | 30 | 34 | 0.1124 |
| 43 | 48 | 31 | 41 | 0.1105 |
| 44 | 50 | 26 | 35 | 0.1103 |

The GoldKL English-net sum is 110.

**Definitions:**

| Term | Definition |
|---|---|
| `zh_new` | new Mandarin errors (B0-correct substantive Mandarin reference units now wrong); retention = 1 − zh_new / 4950 |
| `en_net` | substantive English corrections minus corruptions against B0 (`paired(...)["net"]`) |
| `cs_mer` | genuine-CS MER |
| `tw_new` | new errors among B0-correct units of the target window (first 7 substantive Mandarin units after each EN→ZH switch) |
| `sb` | long switch-back dropped units (P2-1D/P2-1b definition: deletion runs ≥ 5 starting after substantive English) |

mean(·) is over the three seeds.

## A — primary preservation improvement (both required)

| Condition | Requirement |
|---|---|
| A1 | mean zh_new(D1) ≤ mean zh_new(GoldKL) − 5 |
| A2 | zh_new(D1_s) < zh_new(GoldKL_s) in ≥ 2 of 3 matched seeds |

## B — English guardrail

Σ_s en_net(D1_s) / Σ_s en_net(GoldKL_s) ≥ 0.90. Under the current control that means ≥ 99 of 110. If the GoldKL sum is
≤ 0, the ratio is undefined and B fails.

## C — genuine-CS MER guardrail (strict)

mean_s cs_mer(D1_s) ≤ mean_s cs_mer(GoldKL_s), at full precision with no tolerance. The earlier +0.005 allowance does not
apply.

## D — switch-back evidence (reported; mechanism-relevant, outside the promotion rule)

- TW new errors and long-SB dropped units are reported with their denominators (TW eligible and B0-correct units) and
  per-seed consistency.
- **TW support** holds iff mean tw_new(D1) < mean tw_new(GoldKL) and D1 is lower in ≥ 2 seeds.
- A D1 result with an aggregate preservation gain but without TW support is never described as confirming
  switch-back-specific protection.

## E — strong preservation (reported only)

zh_new ≤ 49 (retention ≥ 0.99 of 4,950) per seed, for D1 and GoldKL.

## Secondary preservation components (used only for MIXED)

For key ∈ {tw_new, sb}: the component passes iff mean(D1) ≤ mean(GoldKL) − 5 and D1 is strictly lower in ≥ 2 seeds.

## Integrity (all required, else `D1_INVALID`)

- the newest preflight is `D1_READY_FOR_TRAINING` (not a dry run);
- verification passes: 3 runs × 3 epochs reload and re-decode 20/20, hashes match, 5,122 parameters;
- three runs with epochs 1–3 and 96 updates each;
- zero-init identity 150/150 for every run (D1 and GoldKL);
- backbone weights unchanged;
- GoldKL recomputation reproduces `p2_1b/evaluation.json` (selected epoch; per epoch zh_new, sb, worst dialogue, EN
  net, CS MER, MER);
- per epoch, D1 mask positions, supervised tokens and updates equal the matched GoldKL run (the teacher mask does not
  depend on the student);
- the initial parametrization `base` equals the GoldKL run of the same seed;
- orthogonality error ≤ 1e-5 at every step;
- the region-metadata hash equals the frozen hash in every run;
- region weights are active in every epoch;
- the learning rate is constant at 5e-4.

## Terminal verdict (strict precedence)

1. `D1_INVALID`: integrity fails.
2. `D1_INCREMENTAL_PRESERVATION_GAIN`: A ∧ B ∧ C.
3. `D1_MIXED_TRADEOFF`: either A holds but B or C fails (preservation improved at a correction or MER cost), or A fails
   but a secondary preservation component (TW or SB) passes (a consistent component gain that conflicts with the primary
   rule).
4. `D1_NO_INCREMENTAL_GAIN`: otherwise, i.e. no sufficiently consistent preservation improvement.

**Fixed-epoch-3 comparisons** apply the same component functions descriptively and never change the verdict.

**Guards:**
- Thresholds are never changed after outputs are seen, and the verdict is never upgraded by hand.
- This is a pilot-screening verdict on an exposed panel, not independent confirmation.
- Region-specific causality is not claimed without a matched non-switch-region control.

## Recommendation (one; it is not authorization)

| Verdict | Recommendation |
|---|---|
| INCREMENTAL_PRESERVATION_GAIN | `GO_MATCHED_REGION_SPECIFICITY_CONTROL`: the same total reweighting on a matched non-switch Mandarin region, before any claim of switch-back specificity |
| MIXED_TRADEOFF | `CHARACTERIZE_CORRECTION_PRESERVATION_TRADEOFF`: state which side moved; no automatic follow-up |
| NO_INCREMENTAL_GAIN | `STOP_FIXED_SWITCHBACK_WEIGHTING`: stop this fixed weighting rule |
| INVALID | `HOLD_FOR_ENGINEERING_REPAIR`: name the repair needed |

In every case, D2 (representation edit-locality) remains a separate research candidate and is not started
automatically. A strong optimized LoRA benchmark should be prepared as an independent protocol, not launched within D1.
