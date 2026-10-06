# P2-SEL-T T0 — token-local English evidence diagnosis (no steering)

Contract `fdace84`; implementation `d36d7f6`; pre-T0 `PASS_TO_P2_SEL_T_T0` (`50004bd`). T0 job Slurm 57838,
manifest at `50004bd`: 62 s, 358 new `native_lid` calls (LEFT/RIGHT only; 2 deduplicated), CENTER, E_1s, W and null
reused from P2-SEL-E E0; 0 steering / D2 / autograd; peak VRAM 3.70 GB allocated / 4.01 GB reserved.

**Frozen label: `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`. `P2_SEL_T_AUDIT: PASS (T0)`. No repair; T1/T2 not run.**

## Integrity (all pass)

Live current-query W equal to the stored W at 180/180, W attention mass (and its re-sum from the saved frame vector)
within 1e-6, E_1s reproduced within 1e-6, CENTER bounds == E0 short bounds and raw probabilities reused bitwise,
all crops inside W and heard audio, new LID 100-way sums within 1e-6, masses and j* reproduced, groups
**EN-confusion TP 42 / FN 18, ZH-correct TN 55 / FP 5, EN-correct 60**. Bootstrap 9,967 valid draws (≥ 9,900).

## Frozen predicates (applied once)

| Branch | Criterion | Observed | Pass |
|---|---|---|---|
| TOKEN_LOCALIZATION | ≥4/5 FP: E_tok ≤ 0.10 and E_1s − E_tok ≥ 0.25 (≥3 dlg) | 1/5 (1 dialogue) | no |
| | ≥34/42 TP: E_tok ≥ 0.20 (≥3 dlg) | 36/42 (15 dialogues) | yes |
| | mean E_tok(TP) − E_tok(FP) ≥ 0.25, one-sided 80% lower > 0 | +0.010, lower −0.167 (90% [−0.278, +0.354]) | no |
| CONTEXT_CONTRAST | ≥4/5 FP C ≤ −0.25; ≥26/42 TP C ≥ 0; mean C(TP) − C(FP) ≥ 0.25, lower > 0 | 1/5; 30/42; +0.116, lower −0.051 | no |
| RECALL_ONLY | ≥9/18 FN E_tok ≥ 0.20 (≥3 dlg) and FP suppression fails | 8/18 (8 dialogues) | no |

Precedence -> `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE` -> stop (no contrast or recall fallback).

## Descriptive (median [IQR]; selected crop fraction L/C/R)

| Group | E_1s | E_L | E_C | E_R | E_tok | E_ctx | C_tokctx | a_tok | L/C/R |
|---|---|---|---|---|---|---|---|---|---|
| EN TP (42) | 0.893 | 0.748 | 0.835 | 0.889 | 0.912 [0.644, 0.966] | 0.615 | +0.112 | 0.509 | .24/.48/.29 |
| EN FN (18) | 0 | 0 | 0.047 | 0.140 | 0.093 [0, 0.502] | 0.277 | −0.070 | 0.491 | .22/.56/.22 |
| ZH TN (55) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.732 | .15/.60/.25 |
| ZH FP (5) | 0.925 | 0.934 | 0.616 | 0.688 | 0.934 [0.688, 0.966] | 0.622 | +0.224 | 0.359 | .60/0/.40 |
| EN-correct (60) | 0.792 | 0.343 | 0.813 | 0.781 | 0.768 | 0.488 | 0 | 0.614 | .23/.55/.22 |

All 180 crops: C 96, R 45, L 39. ZH-FP rows (E_1s → E_tok, chosen crop): U1027_S0_3 0.977 → 0.966 (L);
U1063_S0_54 0.981 → 0.994 (R); U1076_S0_24 0.748 → 0.688 (R); U0011_S0_780 0.893 → **0.000** (L; the only
suppressed row); U0029_S0_95 0.925 → 0.934 (L).

For 4/5 ZH-correct false positives, strong English evidence persists in the most-attended 0.5 s crop, so it is not
explained by surrounding context outside the attended region. Their selected-crop attention is more diffuse (a_tok
median 0.36 vs 0.51 TP, 0.73 TN), and none selects CENTER. This is descriptive only and is not a frozen criterion.
Firewall: exposed 180 D-dev-select positions only; evaluator timing not used. P3 HELD.
