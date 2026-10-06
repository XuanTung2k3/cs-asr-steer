# P2-SEL-XA — report (terminal)

**Terminal: `P2_SEL_XA_INVALID`** (XA0 finite-difference validity failed; material subset 4/30 rows).
`P2_SEL_XA_AUDIT: PASS (XA0)` (attempt 1 BLOCKed on an auditor tolerance defect: float64-vs-float32 J compared at
1e-6 on derived ratios; fixed mechanically in `02d9128`; label unchanged; BLOCK file preserved). Final session audit
`P2_SEL_XA_AUDIT: PASS`. No gate selected, so XA1 and XA2 were not run.

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `595e174` | spec/design/config |
| Implementation | `83845ee` | S/C/F helper, XA0 runner, analysis, auditor, Slurm, 9 (+1) tests |
| Pre-XA0 audit | `e6cf957` | `PASS_TO_P2_SEL_XA_XA0` |
| XA0 | Slurm 57844 | valid engineering; FD subset too small → INVALID |
| XA0 audit | `b467ae5` | PASS (attempt 2) |
| XA1 / XA2 | — | NOT RUN |

Details: `P2_SEL_XA_XA0_REPORT.md`.

## Bottleneck update (descriptive)

At L16, the current query's cross-attention source contribution is almost orthogonal to the reference-free
script-readout gradient (|cos| ≈ 0.02). The source-gain sensitivity S_src is therefore far below 1 nat per unit gain
for essentially every row, and below the bf16 resolution of a 5% finite difference. As a gate factor, F_src
would also have been near zero for every true positive. Taken together with P2-SEL-E (window artifacts) and
P2-SEL-T (token-local LID), three reference-free acoustic evidence signals have now failed to separate the 5
ZH-correct false positives from EN-confusion true positives under frozen criteria. Under the frozen
INVALID label no claim about source information is made.

## Next action

Stop. A new human decision and separately frozen pre-registration are required, for example lexical/phonetic token
compatibility (named by the P2-SEL-T contract), or accepting the P2-SEL tradeoff. No XA1/XA2/full-300/P3.
