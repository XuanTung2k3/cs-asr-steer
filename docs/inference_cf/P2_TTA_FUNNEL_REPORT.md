# P2-TTA-FUNNEL — master report (actual path)

Frozen contract `ff75e1a` (`P2_TTA_FUNNEL_SPEC.md`, `P2_TTA_FUNNEL_CODEX_DESIGN.md`,
`configs/inference_cf/p2_tta_funnel.json`).

**Path taken: TTA0-R → A2 selected → TTA-MAP skipped → TTA1 → `P2_TTA1_SUPPORTED` → STOP.**

| Stage | Executed | Label | Audit | GPU |
|---|---|---|---|---|
| TTA0-R | yes (CPU only, sealed run1) | `P2_TTA0_R_OBJECTIVE_SELECTED` (A2) | gate `PASS_TO_P2_TTA0_R_EVALUATION`; `P2_TTA0_R_AUDIT: PASS` | 0 jobs |
| TTA-MAP | **no** (R selected an objective; the frozen branch skips MAP) | — | — | 0 jobs |
| TTA1 | yes (A2, fixed 100) | `P2_TTA1_SUPPORTED` vs forced-ZH | `PASS_TO_P2_TTA1`; `P2_TTA1_AUDIT: PASS` | 1 job (57871) |

Reports: `P2_TTA0_R_REPORT.md` and `P2_TTA1_REPORT.md`. No MAP report exists because MAP did not run, and the frozen
MAP panel stays unused.

## Key numbers

**TTA0-R (panel20)**
- The only numerical repair was the live-gradient tolerance, 1e-3 → 2e-2. Measured live discrepancies: A1 1.025e-2,
  A2 0.
- A1 is `TTA0_EM_CONFIRMATION_BIAS`. Entropy fell 53% and one new severe truncation occurred, on a baseline that had
  hit the 200-token cap. Its point metrics otherwise improved strongly: MER −0.088, PIER −0.056.
- A2 is `TTA0_AC_VIABLE`: net POI +2, PIER −0.011, every safety bound passes. It is the only viable objective, so it is
  selected.

**TTA1 (fixed 100, A2)**

| | PIER | MER | ZH-CER |
|---|---|---|---|
| B0-FORCED | 0.4957 | 0.2570 | 0.2193 |
| B0-AUTO | 0.4080 | 0.2591 | 0.2336 |
| A2 TTA | 0.4540 | 0.2537 | 0.2211 |

- Vs forced-ZH: net POI +29 (31/2), PIER −0.042 [−0.091, −0.002], MER −0.003, matrix-ZH retention 0.994, outside harm
  0.006, all safety bounds pass → `P2_TTA1_SUPPORTED`.
- Vs AUTO: PIER +0.046, so still below AUTO; MER −0.005 and ZH-CER −0.013, slightly better than AUTO.
- 14/100 transcripts changed. Two utterances carry 25 of the 29 net POI corrections.

## Scientific conclusion (exposed development only)

- **Objective:** the problem-specific objective, AUTO-consistency NLL with a fixed AUTO teacher, was the one that
  survived. Entropy minimization (A1) optimized strongly but tripped the frozen severe-truncation safety bound, so it
  falls under confirmation bias.
- **Capacity:** 0.016% of parameters (decoder-LN affine) and 2 AdamW steps are enough to change outputs safely
  (14/100) with a material POI gain over forced-ZH.
- **Conditioning:** A2 barely moves forced-ZH output toward AUTO; in the disagreement group the summed distance to
  AUTO goes 149 → 146. Most changes are self-consistency on rows where AUTO equals the forced output, and the method
  does not recover the AUTO language-choice advantage (about 47% of the PIER gap is closed). MAP, which would have
  measured D_cond and run the soft-KL A3, was not run, so the conditioning gap itself is unmeasured.
- **Sequence stability:** at 100 utterances there is no new cap, no severe truncation and no retention loss. This
  contrasts with P2-SEQ steering (MER +0.028, deletions/early EOS).
- **Best surviving method:** episodic decoder-LN TTA with A2 AUTO-consistency. It beats forced-ZH on PIER/POI and
  roughly ties AUTO on MER, but it does not beat ordinary Whisper AUTO on PIER.

## Compute and data

- One new scientific GPU job: 57871, 2 m 16 s allocation, 117 s runtime. Peak VRAM 3.87 GB allocated / 4.22 GB
  reserved. TTA0-R and MAP used 0 jobs.
- Data: only already-exposed D-dev-select (sealed panel20 run1, fixed 100). No fresh validation, full-300, P3 or
  transfer.
- Engineering notes: the TTA0-R gate and post-audit each had one preserved attempt that failed on the same
  test-file source-scope bug (`*_attempt1_testfile_scope_bug.json`). Both were fixed with tests; no sealed input,
  tolerance or decision changed.

## Next action

Stop. A new human decision is required for any confirmation on unexposed data or any further method change. The frozen
funnel authorizes nothing more.
