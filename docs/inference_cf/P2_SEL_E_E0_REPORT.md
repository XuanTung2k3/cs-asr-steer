# P2-SEL-E E0 — local-English-evidence diagnosis (no steering)

Revised contract `b693b8d`; implementation `0703523`; pre-E0 audit r1 `PASS_TO_P2_SEL_E_E0` (`7964adb`).
E0 job: Slurm 57809, manifest at `7964adb`, 61 s, 361 `native_lid` calls (180 long + 180 short + 1 null),
0 autograd/readout calls, peak VRAM 3.70 GB allocated / 4.01 GB reserved.

**Frozen diagnosis: `P2_SEL_E_DIAGNOSIS_AMBIGUOUS`. `P2_SEL_E_AUDIT: PASS (E0)`. No repair selected; E1 and E2 not run.**

## Integrity (all pass)

180/180 rows. The recomputed 1 s E reproduces P2-R run3 E exactly (max |Δ| = 0.0 ≤ 1e-6). Windows and query keys
equal P2-R. The 100-way softmax sums are within 1e-6 (null 1.00000003). Provider identity (generation config and
mapping hashes, 100 IDs, EN/ZH 50259/50260) verified. Short-crop bounds independently reproduced. Frozen groups
reproduced exactly: **EN-confusion TP 42, FN 18; ZH-correct TN 55, FP 5; EN-correct 60.**
Null: π_null(E) = 0.317, π_null(M) = 0.019, so ell_null = +2.81 at every row.

## Frozen hypotheses (criteria applied once, in precedence order)

| Hypothesis | Frozen criterion | Observed | Pass |
|---|---|---|---|
| H_E1 static scale | ≥4/5 FP with E_short ≤ 0.10 and E_long − E_short ≥ 0.25; ≥34/42 TP with E_short ≥ 0.20 | 0/5 FP; 29/42 TP | no |
| H_E1 oracle FN | ≥9/18 FN with E_oracle ≥ 0.20 and E_oracle − E_long ≥ 0.20 | 6/18 (6 dialogues) | no |
| H_E1 short-only FN | ≥9/18 FN with E_short ≥ 0.20, E_long = 0 | 8/18 (7 dialogues) | no |
| H_E2 temporal spikes | ≥4/5 FP isolated; ≥26/42 TP persistent; one-sided 80% lower bound > 0 | 1/5 isolated; 32/42 persistent; diff −0.066, lower −0.255 | no |
| H_E3 low pair-mass Q | ≥4/5 FP with Q ≤ 0.2169; ≤10/42 TP; diff ≥ 0.5; lower > 0 | 0/5; 0/42; diff 0.0 | no |
| H_E4 null pathology | ≥4/5 FP with ell_local ≤ 0, A > 0, null share ≥ 0.8; TP median ell_local > 0 | 0/5 (TP median ell_local +5.69) | no |

Precedence → `P2_SEL_E_DIAGNOSIS_AMBIGUOUS` → stop (no fallback repair).

## Descriptive table (median [IQR]; not decision inputs)

| Group | n / dialogues | E (1 s) | E_short | Q | ell_local | A | E_(t−1) | attention mass | E_oracle | timing error (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| EN-conf TP | 42 / 16 | 0.893 [0.658, 0.954] | 0.835 [0.109, 0.963] | 0.930 [0.822, 0.976] | 5.69 [4.39, 6.56] | 2.88 [1.58, 3.75] | 0.754 [0, 0.941] | 0.674 | 0.798 [0.503, 0.982] | 0.43 [0.13, 1.19] |
| EN-conf FN | 18 / 11 | 0 | 0.047 [0, 0.660] | 0.721 [0.428, 0.841] | 0.31 [−0.67, 1.58] | −2.50 [−3.48, −1.23] | 0 | 0.699 | 0.000 [0, 0.400] | 0.27 [0.16, 0.58] |
| ZH-corr TN | 55 / 19 | 0 | 0 | 0.987 [0.953, 0.994] | −4.25 [−5.56, −3.34] | −7.06 [−8.38, −6.16] | 0 | 0.873 | — | — |
| ZH-corr FP | 5 / 5 | 0.925 [0.893, 0.977] | 0.616 [0.303, 0.959] | 0.908 [0.676, 0.966] | 6.06 [5.69, 7.25] | 3.25 [2.88, 4.44] | 0.798 [0.774, 0.925] | 0.673 | — | — |
| EN-correct | 60 / 20 | 0.792 [0, 0.962] | 0.813 [0.047, 0.955] | 0.954 [0.849, 0.989] | 4.97 [−0.97, 6.75] | 2.16 [−3.78, 3.94] | 0.767 [0, 0.960] | 0.842 | — | — |

The five ZH-correct false positives (each in a different dialogue) are not window-scale, spike, low-Q or null
artifacts. Their local windows carry strong English language-ID evidence: ell_local 4.75–7.44, Q 0.50–0.99, and
4/5 have positive E at the preceding steps. Their profile is indistinguishable from EN-confusion TPs. The FN side is
mixed: only 6/18 have evaluator-centred oracle support and 8/18 have short-window support, both below the
frozen 9/18. The FN rows sit at weakly English ell_local (median +0.31), below the constant null offset +2.81.
These observations are descriptive; under the frozen tree they do not select any repair.

Firewall: only the exposed 180 D-dev-select positions; P2-R oracle timing used diagnostically only. P3 HELD.
