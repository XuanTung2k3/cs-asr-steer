# P2 compact development — report

**Verdict: `P2_VALID_CONFIGURATION_SELECTED` → `READY_FOR_P3_LOCKED_EVALUATION`.** This follows
the frozen rules of `P2_COMPACT_DEVELOPMENT_SPEC.md` (v1 + v1.1 §11).

**Selected, frozen:** layer 16, α = 2.0, φ(g) = g, g = E·R_B, +d.

**`OPEN DECISION` (human, before P3).** The effect is +2 POI errors of 2,268, and the
predeclared P2-B mechanism controls do not support the proposed mechanism (§P2-B). P3 is not
started.

## Setup

- **Role:** D-dev-select, 300 R2-panel utterances in 20 dialogues (already exposed). No
  confirm/test/transfer data.
- **Execution:** `experiments/inference_cf_cached.py`, bf16, greedy, 200 tokens, H100 MIG 3g.40gb.
- **Δ reference (v1.1):** `B0M_Lℓ`, the α = 0 run of the identical `cached_decode` path in the
  same job. It is identical across all four runs and both layers: 300/300 token sequences,
  zero-dose bitwise, `lineage_ok`.
- **Standalone `cached_greedy` B0 is diagnostic only.** It differs from `B0M` on 59/300
  utterances (near-tie bf16 flips between code paths); this is why attempt 1 was invalid.

| Run | Job | Commit | Manifest | Time | Peak VRAM |
|---|---|---|---|---|---|
| P2-A r1 L16 | 54821 | `04d2383` | `60bb8115…` | 55.8 min | 3.93 GiB |
| P2-A r1 L24 | 54822 | `04d2383` | `0d377be5…` | 54.9 min | 3.93 GiB |
| P2-B | 54824 | `816b385` | `bb07a502…` | 79.6 min | 3.93 GiB |
| P2-C | 54825 | `816b385` | `64d30a6f…` | 47.2 min | 3.93 GiB |

**P2-A attempt 1** (jobs 54803/54804) is `P2_BLOCKED_INVALID_EXPERIMENT`, preserved and unused
(`P2_A_ATTEMPT1_INVALID_BASELINE.md`).

## Baselines (300 IDs)

| System | PIER (errors/2268) | MER | EN-WER | ZH-CER | caps |
|---|---|---|---|---|---|
| **B0M_L16/L24** (matched Δ reference) | 0.4735 (1074) | 0.2647 | 0.4771 | 0.2282 | 2 |
| B0 `cached_greedy` forced-ZH (diagnostic) | 0.4771 (1082) | 0.2606 | 0.4806 | 0.2229 | 1 |
| **B0_AUTO** (`generate`, language=None) | **0.3911** (887) | 0.2602 | 0.3964 | 0.2356 | 2 |
| B1 forced-EN | 0.3241 (735) | 0.7949 | 0.3320 | 0.8666 | 0 |

**Beating B0 alone ≠ beating ordinary Whisper.** B0_AUTO has a PIER 8.2 points lower than any
steered configuration.

## P2-A — efficacy / layer / strength (g = E·R_B, φ_id)

Δ is computed against the matched B0M; for ΔPIER, positive = fewer errors. Validity rules
V1–V5 are in §5 of the spec.

| Config | ΔPIER err | ΔZH-CER (+ = worse) | ΔMER (+ = worse) | ZH retention | Corrections / corruptions | Realized energy | Valid |
|---|---|---|---|---|---|---|---|
| L16 α0.5 | −1 | −0.0002 | −0.0001 | 0.9996 | 0 / 1 | 74.9 | no (V5) |
| L16 α1 | 0 | −0.0001 | −0.0002 | 0.9996 | 0 / 0 | 301.4 | no (V5) |
| **L16 α2** | **+2** | −0.0009 | −0.0009 | 0.9997 | 2 / 0 | 1203.4 | **yes** |
| L24 α0.5 | 0 | −0.0001 | −0.0002 | 0.9996 | 0 / 0 | 75.3 | no (V5) |
| L24 α1 | −1 | −0.0003 | −0.0002 | 0.9997 | 0 / 1 | 298.1 | no (V5) |
| L24 α2 | 0 | −0.0003 | −0.0003 | 0.9993 | 2 / 2 | 1206.3 | no (V5) |

**Edit behaviour.**

- About 950 edits per configuration; the gate is positive at about 1,045 steps out of about
  12,450 eligible.
- Mean relative edit size is 0.03–0.12.
- Utterances that diverge from B0M: 12–28 per configuration.
- Every divergence is preceded by an applied edit (the v1.1 validity check).

**Outcome:** `P2_A_VALID_CONFIG_EXISTS` → (ℓ*, α*) = (16, 2.0), with α_c = 0.3110.

## P2-C — dose map φ(g) = √g at L16

| Config | ΔPIER err | ΔZH-CER | ΔMER | ZH retention | Corrections / corruptions | Energy | Valid |
|---|---|---|---|---|---|---|---|
| √ α0.5 | 0 | −0.0003 | −0.0003 | 0.9999 | 0 / 0 | 103.9 | no (V5) |
| √ α1 | +2 | −0.0002 | −0.0003 | 0.9993 | 2 / 0 | 416.6 | yes |
| √ α2 | 0 | −0.0006 | −0.0005 | 0.9994 | 2 / 2 | 1679.8 | no (V5) |

**Final selection** (§5, over P2-A and P2-C): L16 α2 id and √ α1 tie at +2. The tie-break
**lower ZH-CER increase** (−0.0009 vs −0.0002) keeps **L16 α2 φ_id**. Redistributing strength
with √g does not improve the correction–damage trade-off.

**Selected configuration vs matched B0M** (dialogue-cluster bootstrap, 2000 resamples, seed
240924; positive = selected better):

| Metric | Δ | 95% CI |
|---|---|---|
| PIER | +0.00088 | [0, 0.0027] |
| MER | +0.00089 | [0, 0.0023] |
| ZH-CER | +0.00090 | [−0.0002, 0.0026] |

**Selected configuration vs B0_AUTO:**

| Metric | Δ | 95% CI | Reading |
|---|---|---|---|
| PIER | −0.0816 | [−0.133, −0.034] | worse |
| MER | −0.0036 | [−0.014, 0.006] | no clear difference |
| ZH-CER | +0.0083 | [0.0008, 0.020] | better |

## P2-B — mechanism controls at L16 α2 (predeclared, not selectable)

| Control | ΔPIER err | ΔZH-CER | ZH retention | Corrections / corruptions | Energy | Edited steps | V1–V5 |
|---|---|---|---|---|---|---|---|
| g = E·R_B, +d (selected) | +2 | −0.0009 | 0.9997 | 2 / 0 | 1203 | 949 | valid |
| g = 1, α* = 2 | **+18** | −0.0151 | **0.9855** | 38 / 20 | 48718 | 12166 | V4 fail |
| g = 1, α_c = 0.311 (energy-matched) | +5 | −0.0073 | 0.9987 | 9 / 4 | 1176 | 12339 | valid |
| g = E | +8 | +0.0004 | 0.9980 | 18 / 10 | 5676 | 2192 | valid |
| g = R_B | +10 | −0.0167 | 0.9914 | 20 / 10 | 35795 | 10184 | valid |
| g_cf = E·[R_B − R_Ecf]_+ | +2 | −0.0057 | 0.9994 | 2 / 0 | 73 | 679 | valid |
| g = E·R_B, **−d** | **+11** | −0.0140 | 0.9985 | 11 / 0 | 1089 | 966 | valid |

Answers to the predeclared questions (§7), restricted to them:

- **Does selective placement add value over broad steering?** Not supported.
  - The energy-matched constant gate (realized energy 1176 vs 1203, within 2%) gives +5 vs +2.
  - The same-α constant gate gives +18 but breaks retention (V4).
- **Does baseline conflict protect already-correct English?** g = E alone has 10 corruptions
  vs 0 for E·R_B. This is consistent with R_B limiting harm, but E·R_B also makes almost no
  corrections.
- **Does acoustic support protect legitimate Mandarin?** Not supported. g = R_B has *lower*
  ZH-CER than E·R_B, with retention 0.9914.
- **Does the R2-rejected g_cf restriction add value?** No. g_cf ties the selected configuration
  (+2) at 6% of its energy.
- **Does the direction sign matter causally?** Yes, but opposite to the hypothesis. −d gives
  +11 with 0 corruptions vs +2 for +d.

**Post-hoc descriptive intervals** (`results/inference_cf/p2_B_descriptive_intervals.json`,
not predeclared, not used for selection):

- Versus the selected configuration, every control's PIER CI includes 0. The data cannot rank
  them; all effects are small relative to the dialogue-level variance at n = 300.
- Versus B0M, −d has PIER CI [0, 0.0126] and ZH-CER CI [0.0003, 0.031].

**Mechanism reading (development evidence only).**

- The frozen detector-gated +d intervention has essentially no recognition effect at L16/L24:
  about 2 POI corrections in 2,268.
- Where edits help at all, the controls suggest generic perturbation effects rather than
  detector-selective, direction-specific repair.
- This matches R2's weak localization on EN-confusion (0.464). It is not a claim about
  locked-data performance.

## Validity and audit

- **Experiment validity (v1.1)**, all runs: 0 failures; B0M zero-dose bitwise with lineage;
  **every divergence edit-attributed** (0 unattributed); all configurations complete.
- **Independent audit** (`experiments/inference_cf_p2_audit.py`, which does not import the
  evaluator): `AUDIT_PASS` on the P2-A r1, final and P2-B summaries. See `P2_AUDIT.md`.

## Frozen artifact

`results/inference_cf/p2_frozen_configuration.json` records the configuration, method,
comparators, manifests, jobs, commits and all attempted settings.
