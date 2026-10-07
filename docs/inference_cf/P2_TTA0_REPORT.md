# P2-TTA0 — episodic decoder-LN TTA objective viability screen (report, terminal)

**Terminal: `P2_TTA0_INVALID`.** Independent reference-free audit `P2_TTA0_AUDIT: PASS` (INVALID confirmed).
Neither objective is labelled or selected, and no objective advances. No TTA1 handoff was written. The sealed
adapted outputs were **not evaluated**: no reference was loaded and no canonical metric was computed.

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `eb0da2a` | spec/design/config/panel20 |
| Implementation + pseudo seal | `503c874` | `episodic_tta.py`, runner, analysis, independent auditor, Slurm, 14 tests; y_B/y_A sealed before adaptation |
| Pre-run audit | `6d408d2` | `PASS_TO_P2_TTA0` (33/33) |
| Resolved manifest pushed | `6a954f5` | `sha256:22ed38d9…`, before the job |
| Run | Slurm 57868 | 20/20 × {A1, A2} complete, 68 s; first-row live A1 gradient check FAILED |
| INVALID record + audit + diagnostic | `2b57f8a` | `P2_TTA0_AUDIT: PASS`, label `P2_TTA0_INVALID` |

## Setup (as frozen)

- **Panel:** 20 IDs, the first row of each of 20 dialogues from the fixed-100 panel (sha256 `3c752d7a…`).
- **Trainables:** 194 decoder LayerNorm affine tensors, 248,320 of 1,543,490,560 scalars (0.0160882%).
- **Optimizer:** fresh fp32 masters and a fresh AdamW (lr 1e-3, wd 0) per objective and utterance; 2 steps, 3 loss
  evaluations; bf16 `functional_call`. Exact theta0 reset was verified after every objective.
- **Teachers:** y_B = P2-SEQ S0 (the theta0 forced-ZH decode reproduced it bitwise in all 20/20); y_A = historical
  AUTO (reused, not regenerated).
- **Compute:** 20 encoder passes, 20 theta0 integrity decodes and 40 final forced-ZH decodes; 82 backward passes
  (80 adaptation + 2 audit), 80 optimizer steps, 142 teacher forwards.
- **Runtime:** wall 68 s (setup 12.7 s); A1 adapt 5.1 s + decode 11.0 s; A2 adapt 5.3 s + decode 12.4 s.
- **Peak VRAM:** 4.29 GB allocated, 4.65 GB reserved.

## Why INVALID

The frozen first-row live check runs at theta0, before any update. It recomputes each objective with an independent
formula: full-vocabulary masking, `Categorical` entropy for A1 and `cross_entropy` for A2. It requires loss
|diff| ≤ 1e-5 and ‖g_primary − g_auditor‖ ≤ max(1e-8, 1e-3·‖g_auditor‖) on the full fp32 master gradient.

| Objective | Loss |diff| | Gradient relative diff | Result |
|---|---|---|---|
| A1 GREEDY-EM | 7.2e-7 | **1.02e-2** | fail |
| A2 AUTO-CONSISTENCY | 6.0e-8 | 0 (bitwise) | pass |

Under the frozen precedence, a failed live check makes the stage technically INVALID and supersedes every label.

**Engineering diagnostic** (`experiments/inference_cf_p2tta0_live_diag.py` → `run1_live_audit_diagnostic.json`). This
ran on CPU only, on row 0's sealed teachers and the real bf16 model, and read no reference or adapted output. It
compares the gradient with each formula computed in float32 and in float64.

| Relative gradient difference | A1 | A2 |
|---|---|---|
| primary vs auditor, float64 logits | 0 (bitwise) | 0 (bitwise) |
| primary float32 vs float64 | 7.7e-3 | 1.04e-2 |
| auditor float32 vs float64 | 7.5e-3 | 1.04e-2 |
| primary float32, repeated | 0 | 0 |

What this shows:
- **The objective code is correct.** Both formulas define the same gradient exactly.
- **The tolerance is the problem.** The float32-logit / bf16-backward pipeline the contract mandates only reproduces
  that gradient to about 1%, for any float32 implementation. That includes A2, whose check passed only because its
  two float32 formulas happen to produce bitwise-identical logit gradients.
- **So the frozen 1e-3 tolerance cannot be met** by an implementation that is numerically independent of the primary
  one.
- **Repair is not possible without a human decision.** Rerunning would reproduce the failure deterministically. The
  only fixes are loosening the frozen tolerance, changing the frozen precision, or making the auditor non-independent,
  and all of these are contract changes. No retry, tolerance change or second job was made.

## Reference-free mechanics (descriptive only; no evaluator)

| | A1 GREEDY-EM | A2 AUTO-CONSISTENCY |
|---|---|---|
| Mean loss L0 → L1 → L2 | 0.953 → 0.600 → 0.435 | 0.365 → 0.215 → 0.156 |
| Loss decreased | 20/20 | 20/20 |
| Mean grad L2 (step 1, 2) | 1.50, 0.95 | 0.70, 0.36 |
| Mean fp32 master update L2 (relative) | 0.902 (0.59%) | 0.896 (0.59%) |
| Effective bf16 scalars changed (mean of 248,320) | 233,757 | 233,511 |
| Token-weighted entropy on common y_B path, theta0 → theta2 | 0.627 → 0.294 (−53%) | 0.627 → 0.330 (−47%) |
| Mean EOS prob at content positions | 0.0090 → 0.0022 | 0.0090 → 0.0014 |
| Transcripts changed vs B0-FORCED | 10/20 | 3/20 |
| Mean length (B0 52.35) | 46.5 | 52.85 |
| Equal to B0-AUTO (B0-FORCED: 17/20) | 7/20 | 14/20 |
| Token distance to AUTO, sum (B0: 35) | 207 | 49 |
| Closer / farther from AUTO than B0 | 0 / 10 | 0 / 3 |
| Added caps / new severe truncations | 0 / 1 (`U1003_S0_141`: capped 200 → EOS at 79) | 0 / 0 |

Did A1's entropy actually decrease? Yes. Its objective optimized cleanly: −53% on the common path and the loss fell in
20/20 utterances. Whether that improved PIER or hurt Mandarin retention is **unknown**: the stage is INVALID, so no
reference-based evaluation was run, and the confirmation-bias label cannot be applied. Reference-free signs:
- 10/20 transcripts changed and the mean length fell.
- One severe-truncation event, on the one baseline that had hit the 200-token cap.
- EOS probability fell along the teacher path while Mandarin-script mass rose slightly (P_M 0.713 → 0.729).

Did A2 move forced-ZH output toward AUTO? No. Its output changed only on 3 utterances, and in all 3 the AUTO teacher
already equals the forced output (17/20 panel utterances are identical under forced-ZH and AUTO). On the 3 utterances
where AUTO differs, A2 left the forced-ZH output unchanged. On this panel, the AUTO-consistency objective is mostly
self-training on the forced path. Whether those 3 changes help is unknown, again because there was no evaluation.

## Next action

Stop. **A new human decision is required.** Options include re-freezing the live-check gradient tolerance at a level
calibrated to the mandated float32/bf16 precision (the diagnostic measures about 1e-2). That would happen before any
outcome is looked at, and a frozen evaluation of the already-sealed run1 outputs could then be authorized without a
rerun, since every other check passed. No TTA1, no full 100/300, no new LR/steps/objective/subset, P3 HELD.
Data used: the already-exposed D-dev-select panel20 only.
