# P2-A2-MECH0 — episodic A2 adaptation dynamics and termination mechanism (report, terminal)

**Mechanism verdict: `P2_A2_MECH0_TERMINATION_DOMINANT`.** **`A2_CONFIRMATION_READY = YES`.** Independent audit
`P2_A2_MECH0_AUDIT: PASS` (primary phase before references, then full); the auditor's label and readiness agree.

**Scope:** exposed-development mechanism diagnosis on the already-exposed D-dev-select FULL300. It is not a method search,
not fresh validation, and makes no significance claim. A2 is unchanged: exact historical TTA1/PATH5.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, panel, config) | `9d89640` | pre-outcome |
| Implementation + plan | `ab40674` (+ auditor fixture fix) | 9 focused tests; regression 47/47 |
| Pre-run audit | `ff3141b` | `PASS_TO_P2_A2_MECH0` |
| Manifest | `5c2e9e4` | pushed before the job |
| Run | Slurm 58084 | 12 min 12 s (714 s runtime), peak 4.25 GB |
| LN state archives + rows + primary + seal | 3 archive commits, then `ae69635` | pushed before any reference |
| Primary-phase audit | `9ed40e7` | `P2_A2_MECH0_AUDIT: PASS` |
| Secondary + full audit + report | this commit | TERMINATION_DOMINANT; readiness YES |

**Instrumentation.** STEP1 snapshots used a global `torch.optim` step post-hook, registered only around each A2 episode.
It clones the detached fp32 masters to CPU after each original AdamW step. `episodic_tta.py` and `inference_cf_p2tta0.py`
were **not modified**: their baseline hashes match. Synthetic observer-off/on runs gave exact losses, gradients and final
masters. The step-2 snapshot equalled the final masters on 300/300 rows.

## Panel, strata, reconstruction

FULL300: 300 rows, 20 dialogues, hash `bc806239…`. The strata were rebuilt independently from token arrays and match the
frozen panel exactly:

| A2_SAME | A2_DELTA | EOS_RECOVERY | EOS_REGRESSION | CONTENT_DIVERGENCE | AUTO_SAME | AUTO_DELTA | CONTROL40 | MECH_PANEL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 44 | 12 (10 dialogues) | 0 | 32 | 264 | 36 | 40 (2 × 20) | 84 |

The barrier passed on 300/300:
- theta0 matches PATH5 canonical B0;
- final A2 tokens, termination and text match PATH5 A2;
- the effective bf16 state hash matches PATH5;
- FIXED100 fp32 final masters match the historical archives on 100/100.

## Dynamics, steps and geometry (reference-free)

**Loss and parameter movement (FULL300 medians).**
- Losses: L0 0.194 → L1 0.106 → L2 0.071.
- Master delta: 0.498 (step 1) → 0.901 (step 2); relative 0.0033 → 0.0059; effective bf16 delta 0.658 → 0.928.
- Per family at step 2: SELF 0.517, CROSS 0.519, POST 0.527.

**STEP1 vs STEP2 (= final A2).**

| Measure | Changed | Controls | EOS_RECOVERY | CONTENT_DIVERGENCE |
|---|---|---|---|---|
| STEP1 transcript = A2 | 29/44 | 39/40 | 10/12 | 19/32 |
| STEP1 first action = A2 | 37/44 | — | 11/12 | 26/32 |

Sanity checks: the STEP0 action equals B0 on 44/44 rows and the STEP2 action equals A2 on 44/44. Most of the trajectory
shift appears after step 1; step 2 completes it, mainly on content rows.

**First-divergence margins** (processed log-probabilities on the fixed historical prefix):

| Group | Margin | Median STEP0 | STEP1 | STEP2 | ΔM1 | ΔM2 | ΔM2 > 0 |
|---|---|---:|---:|---:|---:|---:|---|
| EOS_RECOVERY (12) | M_stop = logp(A2 content) − logp(EOS) | −0.156 | +0.219 | +0.438 | +0.406 | **+0.625** | 12/12 (10 dialogues); ΔM1 > 0 12/12 |
| CONTENT_DIVERGENCE (32) | M_branch | −0.313 | +0.188 | +0.438 | +0.719 | +0.969 | 32/32 |

**H3 continuation support** (median mean token log-prob C_H, STEP0 → STEP1 → STEP2):

| Group | STEP0 | STEP1 | STEP2 | Rows increased by STEP2 |
|---|---:|---:|---:|---|
| EOS_RECOVERY | −0.647 | −0.449 | −0.419 | 12/12 |
| CONTENT_DIVERGENCE | −1.017 | −0.695 | −0.466 | 30/32 |

**AUTO alignment** (sealed AUTO action at the same absolute k):
- **EOS_RECOVERY: AUTO itself stops at k on 12/12 rows** (B0 action = AUTO; A2 action = AUTO 0/12).
- Changed rows with matched prefix: A2 action = AUTO on 0/21; B0 action = AUTO on 17/21.
- All assessable changed rows: A2 = AUTO on 8/43 (all in AUTO_DELTA content rows).

**Family state ablations** (MECH_PANEL; action retention at the historical prefix):

| Condition | Overall | EOS | Content | Free decode = A2 (changed) | Sensitive (≥ 0.25 rule) |
|---|---:|---:|---:|---:|---|
| DROP_SELF | 0.659 | 0.667 | 0.656 | 21/44 | **yes** (ΔRET 0.34 / 0.33) |
| DROP_CROSS | 0.864 | 0.917 | 0.844 | 34/44 | no |
| DROP_POST | 0.795 | 0.917 | 0.750 | 31/44 | no |
| A2_STEP1 | 0.841 | 0.917 | 0.813 | 29/44 | — |

All controls are unchanged under every drop. These are necessity tests of the final adapted family states, not tests of
independent sufficiency. No subset is selected.

## Reference effect (post-seal)

**FULL300 by system** (the PATH5 aggregates reproduce exactly):

| System | ZH | POI | Mixed | S / D / I | PIER | MER | ZH-CER |
|---|---:|---:|---:|---|---:|---:|---:|
| B0 | 3309 | 1074 | 4456 | 1087 / 3012 / 357 | 0.4735 | 0.2647 | 0.2282 |
| AUTO | 3416 | 887 | 4380 | 976 / 3066 / 338 | 0.3911 | 0.2602 | 0.2356 |
| A2 | 3037 | 1012 | 4126 | 1079 / 2679 / 368 | 0.4462 | 0.2451 | 0.2095 |

**B0 → A2 net improvement by group** (positive = A2 better):

| Group | Rows | ZH | POI | Mixed | Deletions |
|---|---:|---:|---:|---:|---:|
| EOS_RECOVERY | 12 | **+300** | +23 | +323 | **+331** |
| CONTENT_DIVERGENCE | 32 | −28 (positive-only +11) | **+39** | +7 | +2 |
| A2_SAME | 256 | 0 | 0 | 0 | 0 |
| FULL300 | 300 | +272 | +62 | +330 | +333 |

- **F_ZH_EOS = 0.965** (300/311 of the positive ZH rescue); bootstrap [0.758, 0.993].
- **F_DEL_EOS = 0.971** (331/341); bootstrap [0.768, 0.995].
- **AUTO relation:** AUTO_SAME rows give ZH +296 and deletions +329 (all EOS recovery is AUTO_SAME); AUTO_DELTA rows give
  ZH −24 and POI +38.

**Robustness (descriptive).**
- Dialogues improved / worsened / tied: ZH 7/5/8, mixed 9/3/8, POI 9/1/10, deletions 9/3/8.
- Largest positive-rescue shares: row U1003_S0_119 has 0.395 of ZH and 0.375 of deletion; dialogue CSD0502 has 0.441 of
  ZH and 0.457 of deletion.
- LODO: ZH stays positive on 20/20 removals (min +135); mixed, POI and deletions are also 20/20.
- Bootstrap (A2 − B0): PIER −0.027 [−0.044, −0.012], MER −0.020 [−0.041, −0.003], ZH-CER −0.019 [−0.042, −0.0001].
- Bootstrap (B0 − A2): ZH +272 [2, 640], deletions +333 [45, 741].

## Mechanism label

Frozen criteria for TERMINATION_DOMINANT:
- EOS_RECOVERY has ≥ 3 rows and ≥ 3 dialogues: **12 rows, 10 dialogues** ✓;
- both fractions ≥ 0.50: **0.965 / 0.971** ✓;
- median ΔM2 > 0: **+0.625** ✓;
- ≥ 75% of EOS rows with ΔM2 > 0: **100%** ✓.

Result: **`P2_A2_MECH0_TERMINATION_DOMINANT`**.

## Scientific interpretation

1. **How much of A2's positive ZH effect comes from B0 premature-EOS rows?** 96.5% of the positive ZH rescue (300/311).
   This group also exceeds the whole FULL300 net (+272), because content rows are net −28.
2. **How much of the deletion reduction comes from those rows?** 97.1% (331/341 positive).
3. **Does adaptation move log p(content) − log p(EOS) toward continuation?** Yes, on 12/12 EOS rows (median ΔM2 +0.63).
   Continuation support also rises on 12/12. Because the group is defined by B0 EOS vs A2 content, this is a consistency
   check rather than independent evidence.
4. **Does the movement appear after step 1 or need step 2?** It already appears after step 1: ΔM1 > 0 on 12/12, the STEP1
   action matches A2 on 11/12, and the STEP1 transcript matches on 10/12. Step 2 strengthens it and completes most content
   changes (STEP1 transcript match 19/32). This is descriptive only; no one-step method is recommended.
5. **Does A2 follow AUTO at the divergence?** No. On every EOS-recovery row, AUTO stops exactly where B0 does. On prefix-
   matched changed rows, A2 takes AUTO's action 0/21 times. Termination recovery is not AUTO imitation. It is consistent
   with the hard-NLL objective on content tokens only (EOS is excluded from the loss) sharpening continuation over stopping.
6. **Does A2 help where AUTO == B0?** Yes. Every EOS recovery is an AUTO_SAME row (ZH +296, deletions +329).
7. **Which LN families are mechanically necessary?** SELF (self-attention LayerNorm) is the only family whose removal
   reverts at least 25% of divergence actions (overall and EOS). CROSS and POST are below the rule. These are necessity
   results for the final state, not sufficiency. Descriptively, DROP_SELF also has lower ZH error on changed rows than
   FULL_A2; this is not used for any selection.
8. **Are gains distributed or concentrated?** The ZH gain is positive under every LODO removal and improves 7 dialogues,
   but it is concentrated: one row holds 0.40 and one dialogue 0.44 of the positive ZH rescue. POI gains are broader
   (9 dialogues improved, 1 worsened).
9. **Does any finding invalidate A2 as the confirmation candidate?** No. Reconstruction and artifacts are exact and both
   audits PASS. There are 0 FULL_A2 severe truncations vs B0. Concentration and the small net ZH loss on content rows are
   reported, not disqualifying. The frozen readiness rule gives **YES**.

## Method decision

- A2 remains frozen exactly as historical TTA1/PATH5. No change, subset, step count or EOS objective is introduced.
- **NEXT STAGE:** a separately frozen exact-A2 confirmation. Nothing is run here (no D-dev-confirm, D-test or P3).

**Data:**
- Used: already-exposed D-dev-select FULL300 only.
- Not used: fresh validation, D-dev-confirm, D-test, P3, transfer.

**Compute:**
- One job: 12 min 12 s; peak 4.25 GB allocated / 4.66 GB reserved.
- A2: 600 updates; 600 + 1 backward passes; 600 observer snapshots.
- Decodes: 936 free (300 theta0 + 300 A2 + 84 STEP1 + 252 DROP).
- Scoring: 132 geometry paths and 132 DROP current-query paths.
- LN archives: theta0 + 300 STEP1 + 200 STEP2 (391 MB), pushed before the references were opened.
