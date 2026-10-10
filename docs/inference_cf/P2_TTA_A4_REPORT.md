# P2-TTA-A4 — script-preserving AUTO distillation (report, terminal)

**Verdict: `P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`.** Independent post-audit `P2_TTA_A4_AUDIT: PASS` (first attempt).
Exposed-development debug on the enriched A3 24-panel. It is not a population estimate, and no significance claim is
made.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, config) | `37441eb` | before any A4 output |
| Implementation + plan seal | `9488a69` | `script_safe_tta.py`, runner, analysis, auditor, sbatch, 9 tests |
| Pre-run audit | `d32676b` | `PASS_TO_P2_TTA_A4` |
| Manifest pushed | `b349f12` | before the job |
| Run | Slurm 57937 | 24 rows, 53 s allocation (34 s runtime) |
| Output seal | `e506e74` | pushed before references |
| Evaluation + post-audit | this commit | `SEQUENCE_SAFETY_NOT_RESCUED`, `P2_TTA_A4_AUDIT: PASS` |

## Contract

- **Panel:** the exact A3 panel (24 rows: 12 D / 12 A, 20 dialogues, sha256 `6422741b…`).
- **Actuator, identical to A2/A3:** 194 decoder-LN tensors / 248,320 scalars; fresh fp32 masters + AdamW (lr 1e-3,
  wd 0, betas 0.9/0.999, eps 1e-8); 2 steps; bf16 forward; exact reset; forced-ZH greedy final decode.
- **Safe teacher:** q_A (theta0 AUTO) at y_A positions whose target is in V_E, and q_B (theta0 forced-ZH, same prefix)
  everywhere else.
- **Loss:** forward KL(q_SAFE ‖ p_FORCED) over the full allowed vocabulary.
- **Partition:** canonical `core_r2.tokenizer_partition` (`whisper_han_ascii_v1`); V_E = `embedded_ids`,
  V_M = `matrix_ids`.

**Integrity** (all checks pass):
- theta0 forced decode = S0 on 24/24, and the detected AUTO language equals A3's sealed value on 24/24;
- D_ANCHOR0 = 0.0 exactly on every row, confirming the theta0 anchor identity;
- live independent check: loss diff 1.5e-8, gradient 0.89% (≤ 2%);
- exact no-op on 13 rows: all 12 controls (`<|zh|>`) and `U0039_S0_80`, which has no EN position;
- 48 optimizer steps, 49 backward passes;
- peak VRAM 4.49 GB allocated / 4.89 GB reserved.

**Token counts on valid positions:**
- D group: EN 148, MATRIX 24, OTHER 26.
- A controls: EN 94, MATRIX 423, OTHER 58 (all no-ops).

## Mechanics (11 eligible D rows)

| Quantity | Step 0 | Step 2 |
|---|---|---|
| D_E, token-weighted | 0.761 | 0.355 |
| D_M, token-weighted | 0.0 | 0.003 |
| D_ANCHOR, token-weighted | 0.0 | 0.062 |

- **English transfer:** D_E decreased in **11/11** eligible rows, median R_E **0.465** (required ≥ 9 rows and
  ≥ 0.25) → **EN_TRANSFER_PASS.**
- **Anchor:** G_E = 0.406, so **ρ_anchor = 0.152** (≤ 0.25; row-wise median 0.077) → **ANCHOR_PRESERVED.** The
  largest anchor drift is on `U2004_S0_104` (`<|ms|>`): D_ANCHOR2 = 0.918.

The script-preserving teacher works as intended under teacher forcing. English-position AUTO information transfers
nearly as strongly as in A3 (median 0.465 vs 0.514). The non-English forced-ZH distributions stay locally
anchored: matrix-position drift is only 0.003 nats. There is no strong evidence of shared-parameter interference at
the teacher-forced level.

## Sequence movement (reference-free)

- Changed vs B0: D 5/12, A 0/12.
- Changed vs A2: 4 rows. Changed vs A3: 4 rows.
- Summed D distance to AUTO: B0 149, A2 146, A3 132, **A4 140**. Movement vs B0: 2 closer, 8 same, 2 farther.
- No caps, no severe truncations; mean D length 16.6.

## ASR (24-panel, enriched)

| System | PIER | MER | EN-WER | ZH-CER | POI err | S / D / I |
|---|---|---|---|---|---|---|
| B0-FORCED | 0.5863 | 0.2861 | 0.5863 | 0.1848 | 146 | 73 / 215 / 3 |
| B0-AUTO | 0.3414 | 0.2979 | 0.3414 | 0.2792 | 85 | 58 / 242 / 3 |
| A2 | 0.5141 | 0.2920 | 0.5181 | 0.2149 | 128 | 81 / 203 / 13 |
| A3 | 0.5141 | 0.2930 | 0.5181 | 0.2163 | 128 | 83 / 211 / 4 |
| A4 | 0.5141 | 0.2920 | 0.5181 | 0.2149 | 128 | 78 / 215 / 4 |

A4 vs B0-FORCED, against the frozen TTA1 safety bounds:

| Quantity | A4 | Bound | Pass |
|---|---|---|---|
| POI corrections / corruptions | 18 / 0 (net +18) | — | — |
| ΔPIER | −0.0723 | — | — |
| ΔMER | +0.0059 | ≤ 0.01 | yes |
| ΔZH-CER | **+0.0301** | ≤ 0.015 | **no** |
| Matrix-ZH retention | **0.9632** | ≥ 0.98 | **no** |
| Embedded-EN retention | 1.000 | ≥ 0.95 | yes |
| Outside harm | **0.0367** | ≤ 0.03 | **no** |
| POI corruption | 0 | ≤ 0.05 | yes |
| Caps / severe truncations | 0 / 0 | ≤ 1 / 0 | yes |

**Matrix safety rescue vs A3** (fraction of the A3-to-bound gap closed):

| Measure | Fraction closed |
|---|---|
| ZH-CER | 0.080 |
| Retention | 0.087 |
| Outside harm | 0.192 |

None of the three is half-rescued, so there is **no partial rescue and no full rescue**.

**Benefit vs A2:**
- POI 128 vs 128 (difference 0), MER difference 0.0000, no severe truncation → retained.
- The corpus totals coincide with A2's only through group offsets (verified per row):
  - D group: A4 101 POI / 143 mixed errors, A2 103 / 154.
  - A group: A4 = B0 (27 / 154); A2 reaches 25 / 143.

**A4 vs A3:** POI equal (128); MER −0.0010; ZH-CER −0.0013; retention +0.0016; outside harm −0.0016. The gains are
marginal.

Descriptive bootstrap: A4−B0 ZH-CER +0.030 [0.000, +0.097]; A4−A2 PIER 0.000 [−0.024, +0.023].

The frozen precedence goes EN transfer pass → anchor pass → full rescue **fails**, giving
**`P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED`**.

## Interpretation

| Question | Answer |
|---|---|
| 1. Can A4 still transfer AUTO's English information? | **Yes:** 11/11 rows, median R_E 0.465. |
| 2. Does forced-ZH anchoring keep non-English teacher-forced distributions stable? | **Yes:** ρ 0.15, D_M2 0.003. |
| 3. Does that stability survive free decoding? | **No.** Mandarin safety is essentially unchanged from A3 (ZH-CER +0.030 vs +0.031). |
| 4. Does A4 repair the A3 Mandarin problem? | **No:** at most 19% of any gap closed. |
| 5. How much of A2's POI benefit is retained? | All of it on this panel: POI 128 = A2, but no better. |

**Supported bottleneck: sequence / free-decoding coupling, not teacher construction or teacher-forced LN
interference.**
- The Mandarin damage sits in rows where A2, A3 and A4 all emit the *same* degraded transcript. The main case is
  `U0023_S0_664`, where mixed errors go 19 → 39 for every method.
- Its frozen AUTO path is entirely Latin script (18 EN, 0 MATRIX positions). This was recorded before outcomes: AUTO
  writes Latin script for Mandarin speech in most D rows.
- So on such rows the "English" positions *are* AUTO's language switch, and script-level anchoring cannot separate
  useful code-switched English from AUTO's translation/romanization of Mandarin.
- Once the free decoder starts on that Latin path, the locally anchored Mandarin distributions are never reached.

Script identity of y_A tokens is therefore not a sufficient safety signal for which AUTO content to trust.

## Next action

Stop. Do not tune LR, steps, trainables or the teacher formula. Any next step needs a new human decision and a separately
frozen contract. The implicated mechanism is sequence-level path selection (the first-divergence/language-switch
decision during free decoding), not teacher-forced distribution matching. Data: exposed D-dev-select only. No fresh
validation, full-100 A4, full-300 or P3.
