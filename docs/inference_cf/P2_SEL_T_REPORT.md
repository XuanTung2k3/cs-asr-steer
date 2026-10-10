# P2-SEL-T — report (terminal)

**Terminal: `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`** (T0). `P2_SEL_T_AUDIT: PASS (T0)`; final session audit
`P2_SEL_T_AUDIT: PASS`. No R_TOK selection, so T1 (causal screen) and T2 (mini decode) were not run (frozen stop).

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `fdace84` | spec/design/config |
| Implementation | `d36d7f6` | runner, analysis, independent auditor, Slurm, 13 tests |
| Pre-T0 audit | `50004bd` | `PASS_TO_P2_SEL_T_T0`; CENTER == E0 short crop at 180/180 |
| T0 | Slurm 57838 | valid; 358 L/R LID calls; no steering/D2/autograd |
| T0 audit | see `results/inference_cf/p2sel_t/t0_run1_audit.json` | PASS; label reproduced |
| T1 / T2 | — | NOT RUN |

Details: `P2_SEL_T_T0_REPORT.md`.

## Bottleneck update

Token-localizing the existing native LID to the decoder's most-attended 0.5 s crop does not separate the
ZH-correct false positives from EN-confusion true positives: E_tok(TP) − E_tok(FP) = +0.010. Four of five false
positives keep strong English evidence in the attended crop. Together with P2-SEL-E (window-level artifacts ruled
out), the residual selectivity limit is not where the LID window is placed or how its score is transformed. The
contract names the next scientific family, phonetic/lexical acoustic compatibility rather than another LID
transform. This report does not claim that no information exists.

## Next action

Stop. A new human decision and separately frozen pre-registration are required, e.g. phonetic/lexical
compatibility evidence, or accepting the P2-SEL tradeoff. No T1/T2/full-300/P3/fresh validation is authorized.
