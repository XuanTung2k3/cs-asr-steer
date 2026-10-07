# P2-SEL-LAC LAC0 — local acoustic counterfactual lexical compatibility (forward-only)

Contract `be460ce`; implementation `4e6062c`; candidate seal committed/pushed before masking; pre-LAC0
`PASS_TO_P2_SEL_LAC_LAC0` (`92b81c6`).

LAC0 job Slurm 57866, manifest at `92b81c6`:

| Item | Value |
|---|---|
| Wall time | 72 s |
| Counterfactual evaluations | 180 |
| Masked encoder calls | 179 (one identical-mask reuse) |
| Decoder prefix steps | 5,662 |
| Backward / D2 / LID / steering calls | 0 |
| Peak VRAM | 3.95 GB allocated / 4.30 GB reserved |

The unmasked logits were reused from the sealed P2-DIR NONE files.

**Frozen label: `P2_SEL_LAC_NOT_DISCRIMINATIVE`. `P2_SEL_LAC_AUDIT: PASS (LAC0)`.** No F_lex selected; LAC1/LAC2 not run.

## Integrity (all pass)

At 180/180 rows:
- candidates reproduced from the unmasked logits only; the masked run never re-selects;
- W* equals the P2-SEL-T bounds[j], inside W_1s and heard audio, with length min(8000, |W_1s|);
- outside-window bytes are identical, and the inside is positive float32 zero;
- every window had nonzero energy and changed (0 no-ops);
- the replayed prefix equals prompt + baseline tokens[:t], and the query index matches;
- masked logits are finite, with full vocabulary;
- suppression is disjoint from V_E ∪ V_M;
- the forward-only counters hold.

The auditor independently recomputed every mask and hash from the source audio, the candidates, S and the label.
Groups: 42/18/55/5/60.

## Frozen predicates

| Criterion | Observed | Pass |
|---|---|---|
| ≥ 34/42 EN TP with S ≥ log(17/3) = 1.7346 (≥ 3 dialogues) | **6/42** (6 dialogues) | no |
| ≥ 4/5 ZH FP with S ≤ log(11/9) = 0.2007 (≥ 3 dialogues) | 4/5 (4 dialogues) | yes |
| mean S(TP) − S(FP) ≥ 0.50 nat, one-sided 80% lower > 0 | +0.948, lower −0.022 (90% [−0.437, +2.871]) | no |
| Recall: ≥ 9/18 EN FN with S ≥ log(17/3) | 4/18 | no |

Precedence: `P2_SEL_LAC_NOT_DISCRIMINATIVE`; stop, with no alternate mask, candidate or transform.

## Descriptive (median [IQR]; not decision inputs)

| Group | S_lex | M0 | Mmask | Δ_E | Δ_M | F_lex | removed energy |
|---|---|---|---|---|---|---|---|
| EN TP (42) | +0.078 [−0.766, 0.781] | −7.84 | −8.03 | +0.40 | +0.22 | 0.04 | 0.058 |
| EN FN (18) | −0.094 [−0.672, 0.453] | −7.06 | −7.55 | +0.39 | +0.47 | 0.00 | 0.043 |
| ZH TN (55) | −4.00 [−6.41, −2.02] | −12.38 | −8.44 | +2.50 | +7.06 | 0.00 | 0.024 |
| ZH FP (5) | +0.062 [−0.125, 0.188] | −9.19 | −7.94 | −0.06 | −0.25 | 0.03 | 0.030 |
| EN-correct (60) | +2.31 [0.64, 4.08] | +9.38 | +5.88 | +4.53 | +2.73 | 0.82 | 0.023 |

On all 42 EN-confusion TP the unmasked top-1 is the sealed Mandarin candidate (M0 ≈ −7.8). Masking the attended
0.5 s barely changes the English-vs-Mandarin candidate margin (|S| < 1 for most rows; 22/42 positive, 6/42 ≥ 1.73).
Where the decoder already follows the audio, occlusion is strongly diagnostic: correct Mandarin positions lose
Mandarin support (Δ_M +7.1, S −4.0), and correct English positions lose English support (S +2.3). The 5 ZH FP
(candidates e.g. "Jiang"/江, "school"/学, "I"/我) also sit near S ≈ 0, unlike TN. So the signal separates FP from
correct Mandarin but not from EN-confusion TP.

Firewall: exposed 180 D-dev-select positions only; references used only for historical grouping. P3 HELD.
