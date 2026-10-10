# P2-TTA0-R — numerical audit repair and sealed-output evaluation (report)

**Stage label: `P2_TTA0_R_OBJECTIVE_SELECTED` — A2 AUTO-CONSISTENCY.** Independent post-audit
`P2_TTA0_R_AUDIT: PASS`. Per-objective original TTA0 labels: A1 `TTA0_EM_CONFIRMATION_BIAS`, A2 `TTA0_AC_VIABLE`.
Immutable selection: `results/inference_cf/p2tta_funnel/tta0_r/selected_objective.json`. MAP is skipped; next is TTA1.
This stage used no GPU, made no optimizer step or backward pass, and generated no transcript.

## Repair and gate

The funnel contract (`ff75e1a`) overrides one check only: the live complete-gradient criterion becomes
‖g_p − g_a‖ ≤ max(1e-8, 0.02·‖g_a‖), up from 1e-3. Everything else is unchanged: loss agreement 1e-5, finite
gradients, float64 equivalence, fp32 repeat determinism, trainable/reset/output/manifest identities, and all original
TTA0 thresholds.

The gate result is **`PASS_TO_P2_TTA0_R_EVALUATION`** (`f7d9878`, committed and pushed before references were opened):
- All 50 sealed TTA0 hashes, baseline20, the panel and the source anchors match.
- The decision section is an exact copy of the original.
- The independent recomputation passes: updates and losses from the saved masters/terms, theta0 bf16 hash, resets,
  outputs and the 82/80/142 counts.
- Live gradient: A1 1.025e-2 and A2 0, both within 2e-2. Loss differences are 7e-7 and 6e-8.
- Recorded diagnostic: float64 primary and auditor gradients are bitwise equal, the fp32 repeat is identical, and the
  largest fp32 discrepancy is 1.04e-2.

Two engineering attempts are preserved, and neither involved a sealed input or a tolerance:
- Gate attempt 1 (`gate_audit_attempt1_testfile_scope_bug.json`) and post-audit attempt 1
  (`post_audit_attempt1_testfile_scope_bug.json`) both failed on one cause.
- They treated `tests/test_inference_cf_p2tta0.py` as a runtime source. I extended that test file after run1, in
  `2b57f8a`.
- The fix restricts the scope; the runner, `episodic_tta.py`, the cached decoder and the model I/O are still required
  to be byte-identical. Tests were added for the fix.

## Results (panel20, original TTA0 rules)

| System | PIER | MER | EN-WER | ZH-CER | POI err | S / D / I | caps |
|---|---|---|---|---|---|---|---|
| B0-FORCED | 0.4302 | 0.3120 | 0.4469 | 0.2839 | 77/179 | 116 / 149 / 91 | 1 |
| B0-AUTO | 0.3408 | 0.2989 | 0.3575 | 0.2850 | 61 | 103 / 147 / 91 | 1 |
| A1 GREEDY-EM | 0.3743 | 0.2244 | 0.3911 | 0.1900 | 67 | 67 / 178 / 11 | 0 |
| A2 AUTO-CONSISTENCY | 0.4190 | 0.3015 | 0.4358 | 0.2735 | 75 | 116 / 137 / 91 | 1 |

| vs B0-FORCED | A1 | A2 |
|---|---|---|
| Net POI reduction (corrections / corruptions) | +10 (10 / 0) | +2 (2 / 0) |
| ΔPIER [95% CI] | −0.0559 [−0.091, −0.017] | −0.0112 [−0.033, 0.000] |
| ΔMER [95% CI] | −0.0876 [−0.244, +0.001] | −0.0105 [−0.026, 0.000] |
| ΔZH-CER [95% CI] | −0.0939 [−0.270, +0.010] | −0.0104 [−0.025, 0.000] |
| Matrix-ZH / embedded-EN retention | 0.988 / 1.000 | 1.000 / 1.000 |
| Outside-POI lexical harm | 0.0116 (9/774) | 0 (0/774) |
| Utterances improved / degraded (mixed errors) | 5 / 3 | 2 / 0 |
| Added caps / **new severe truncations** | 0 / **1** | 0 / 0 |
| Safety (original) | **fails severe truncation** | all pass |
| Useful | yes | yes (net POI = 2) |

Versus B0-AUTO (95% CI):
- A2: ΔPIER +0.078 [−0.040, +0.286], ΔMER +0.003.
- A1: ΔPIER +0.034 [−0.095, +0.259], ΔMER −0.075.

So both remain above AUTO on PIER. Bootstrap: 2000 paired draws, seed 240924, 2000/2000 valid; descriptive only.

**A1.** Entropy on the fixed y_B path fell 0.627 → 0.294 (−53%). The one safety failure is the new severe truncation
on `U1003_S0_141`: its baseline hit the 200-token cap, and A1 ends it with EOS at 79 tokens. Under the original precedence, ≥1% entropy decrease plus any safety failure gives
**`TTA0_EM_CONFIRMATION_BIAS`**. The rule fires because the frozen truncation bound fails. The point metrics
otherwise improve strongly; that is a descriptive observation and changes no label. Other observations:
- Sequences shortened: mean length 52.35 → 46.5, insertions 91 → 11.
- 10/20 transcripts changed, all among the 17 where forced and AUTO agree; none moved toward AUTO.

**A2.** The NLL fell 0.365 → 0.156 in 20/20 utterances. 3/20 transcripts changed, all in the forced = AUTO group:
2 POI corrections, 0 corruptions, and no safety cost. The three cases where AUTO and forced differ (`U0029_S0_472`,
`U0102_S0_80`, `U1079_S0_23`) were **unchanged** under both A1 and A2; their distances to AUTO stayed 4, 30 and 1. So
there was no movement toward AUTO. On this panel A2's gain comes from self-consistency on rows where the forced
output already agrees with AUTO.

**Selection.** A2 is the only viable objective, so it is selected; no tie-break was needed.
