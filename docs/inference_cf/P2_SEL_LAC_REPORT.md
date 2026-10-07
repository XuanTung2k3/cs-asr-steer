# P2-SEL-LAC — report (terminal)

**Terminal: `P2_SEL_LAC_NOT_DISCRIMINATIVE`** (LAC0). `P2_SEL_LAC_AUDIT: PASS (LAC0)`; final session audit
`P2_SEL_LAC_AUDIT: PASS`. No F_lex selected, so LAC1 (gate screen) and LAC2 (mini decode) were not run (frozen stop).

| Step | Commit / job | Result |
|---|---|---|
| Freeze | `be460ce` | spec/design/config |
| Implementation | `4e6062c` | helpers, runner, analysis, independent auditor, Slurm, 8 tests |
| Candidate seal | committed before masking | 180 pairs from unmasked logits only |
| Pre-LAC0 audit | `92b81c6` | `PASS_TO_P2_SEL_LAC_LAC0` (seal 180/180, feature adapter 80/80) |
| LAC0 | Slurm 57866 | valid; forward-only; not discriminative |
| LAC1 / LAC2 | — | NOT RUN |

Details: `P2_SEL_LAC_LAC0_REPORT.md`.

## Bottleneck update (descriptive)

The local acoustic counterfactual is informative where the decoder's choice is already audio-driven: correct
Mandarin S −4.0 and correct English S +2.3. On EN-confusion true positives the English candidate's margin
barely depends on the attended audio (median S +0.08; 6/42 strong). The decoder prefers Mandarin there by about
7.8 nats regardless of whether that half-second is present, and the ZH-correct false positives look the same
(S ≈ 0). So TP and FP are again indistinguishable, here because the English candidate's deficit at confusion
points is not carried by the attended local audio. Across P2-SEL-E (window LID), P2-SEL-T (token LID),
P2-SEL-XA (source gradient) and LAC (local occlusion), no reference-free local acoustic signal has passed the
frozen safe-separation criteria. This does not prove the information is absent.

## Next action

Stop. A new human decision and separately frozen pre-registration are required, for example explicit
phonetic/acoustic-token compatibility (named by this contract), or accepting the P2-SEL gate tradeoff.
No LAC1/LAC2/full-300/P3 is authorized.
