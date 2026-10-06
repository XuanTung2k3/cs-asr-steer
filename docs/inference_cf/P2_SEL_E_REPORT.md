# P2-SEL-E — report (terminal)

**Terminal state: `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`** (E0). `P2_SEL_E_AUDIT: PASS (E0)`; final session audit
`P2_SEL_E_AUDIT: PASS`. No repair was selected, so E1 (repair screen) and E2 (mini decode) were not run (frozen stop rule).

| Step | Commit / job | Result |
|---|---|---|
| Original freeze | `11f4e63` | pre-E0 BLOCK (`de8d038`): two-token Q premise false; preserved |
| Revised freeze | `b693b8d` | 100-way Q semantics, H_E3/R3 activated |
| Implementation | `0703523` | E0/E1 runner, analysis, independent auditor, Slurm, 16 tests |
| Pre-E0 audit r1 | `7964adb` | `PASS_TO_P2_SEL_E_E0` (R3-vs-E1 conflict pre-registered) |
| E0 | Slurm 57809 | valid; groups 42/18/55/5; all hypotheses fail -> AMBIGUOUS |
| E0 audit | `c7bc7e6` | `P2_SEL_E_AUDIT: PASS (E0)` |
| E1 / E2 | — | NOT RUN |

Details: `P2_SEL_E_E0_REPORT.md`. Compute: one 61 s MIG job; no steering or autograd.

## Bottleneck update (descriptive)

E's residual ZH-correct false positives carry strong, temporally persistent, high-confidence local English
language-ID evidence. None of the four pre-registered artifact mechanisms (window scale, temporal spikes, low
pair mass, null correction) explains them. They are therefore not repairable by any of the frozen
reference-free E transforms. The remaining selectivity gap appears to lie in what a window-level LID signal can
represent: a 1 s acoustic window can contain genuine English evidence near a Mandarin-correct token. This is
not a defect in how E is computed. No localizer or new-signal claim is made, since evaluator-centred oracle
support for FN recovery stayed below the frozen threshold.

## Next action

Stop. Any further work, such as a token-level rather than window-level evidence signal, or accepting the P2-SEL
tradeoff, needs a new human decision and a separately frozen pre-registration. No E1, E2, full-300, P3 or fresh
validation is authorized.
