# P2-TTA-A4 — Script-preserving AUTO distillation (pre-outcome specification)

Frozen before any A4 output. Machine contract `configs/inference_cf/p2_tta_a4.json`. Start HEAD `ab59778`, clean, no jobs.
Separate exposed-development debug after `P2_TTA_A3_TEACHER_UNSAFE`. Historical TTA0/TTA1/A3 outcomes are unmodified.
Core v6 is unchanged.

## Question

Can AUTO's useful English-token information be transferred **without** AUTO's Mandarin degradation?

## Panel

Exactly the A3 panel `docs/inference_cf/P2_TTA_A3_PANEL.json` (sha256 `6422741b…`): 24 rows (12 D, 12 A), 20 dialogues.
No new panel.

## Actuator (identical to A2/A3)

| Setting | Value |
|---|---|
| Trainables | 194 decoder-LN affine tensors / 248,320 scalars |
| Optimizer | fresh fp32 masters + AdamW per utterance (lr 1e-3, wd 0, betas 0.9/0.999, eps 1e-8) |
| Steps | 2 |
| Forward | bf16 |
| Reset | exact |
| Final decode | forced-ZH greedy |

The A3 encoder path, vocabulary and suppression are unchanged.

## Token classes

The canonical partition is `core_r2.tokenizer_partition` (`whisper_han_ascii_v1`, hash `sha256:7daa5009…`):
- V_E = `embedded_ids` (37,858);
- V_M = `matrix_ids` (1,667);
- the 12,341 remaining IDs are ambiguous;
- the sets are disjoint and exhaustive.

Each **valid** y_A position (the A3 mask) is classified by its target y_A,t only: **EN** if in V_E, **MATRIX** if in V_M,
**OTHER** otherwise. EOS and prefix positions are not loss positions.

Pre-outcome counts on D rows: EN 148, MATRIX 24, OTHER 26. On A rows: 94 / 423 / 58.

Recorded limitation: in D rows AUTO mostly emits Latin script, so the EN rule imports AUTO at most of those positions.

## Teachers and objective

Both teachers are theta0, detached and float32 on the same y_A prefix:
- q_A under the AUTO prompt cA. This is the historical `detect_language` prompt, and it must equal A3's sealed `lang_id`.
- q_B under forced-ZH cB.

The safe teacher is q_SAFE,t = q_A,t if y_A,t ∈ V_E, otherwise q_B,t. There is no interpolation, lambda, gate or threshold.

The student p_θ is forced-ZH on y_A. The loss is L_A4 = mean over valid t of KL(q_SAFE,t ‖ p_θ,t): forward KL over the
full allowed vocabulary at temperature 1.

**No-op rule (exact).** A row runs exactly 2 zero-gradient AdamW steps if any of these holds:
- cA = cB;
- the row has no valid EN position;
- the row has no valid position.

In each case q_SAFE = q_B = p_θ0, so the objective is identically zero. The output must equal B0-FORCED. This covers all
12 A controls (A3 detected `<|zh|>`, so q_A = q_B bitwise) and D row `U0039_S0_80` (no EN position).

## Mechanistic diagnostics (eligible D rows = 11 with at least one EN position)

| Quantity | Definition |
|---|---|
| D_E | mean KL(q_A ‖ p_θ) over EN positions |
| D_M | mean KL(q_B ‖ p_θ) over MATRIX positions |
| D_ANCHOR | mean KL(q_B ‖ p_θ) over all non-EN positions |
| R_E | (D_E0 − D_E2)/max(D_E0, 1e-12) |

All are recorded at steps 0, 1 and 2.

- **EN_TRANSFER_PASS:** at least **9/11** eligible D rows have D_E2 < D_E0, AND median R_E ≥ 0.25.
- **ANCHOR_PRESERVED:** ρ = D_ANCHOR2_tw / max(G_E_tw, 1e-12) ≤ 0.25, AND finite, AND no integrity failure. Here
  G_E_tw = Σ n_E·(D_E0 − D_E2) / Σ n_E and D_ANCHOR2_tw = Σ n_nonE·D_ANCHOR2 / Σ n_nonE, both over the eligible D rows.
  The row-wise median is reported descriptively.
- **Theta0 anchor integrity:** |D_ANCHOR0| ≤ 1e-5 nats on every row; otherwise INVALID.

## Safety, rescue and benefit

**Absolute safety** is the authoritative TTA1 vocabulary, copied exactly and identical to TTA0. It is evaluated for A4 vs
B0-FORCED on 24 rows:

| Bound | Value |
|---|---|
| ΔMER | ≤ 0.01 |
| ΔZH-CER | ≤ 0.015 |
| Matrix-ZH retention | ≥ 0.98 |
| Embedded-EN retention | ≥ 0.95 |
| Outside-POI harm | ≤ 0.03 |
| POI corruption | ≤ 0.05 |
| Added caps | ≤ 1 |
| New severe truncations | 0 |

The float guard is 1e-12. **FULL_MATRIX_RESCUE** means all of these pass.

**PARTIAL_MATRIX_RESCUE** is descriptive only. The fraction of the A3-to-bound gap closed is computed per measure, with
A3 references taken from the sealed A3 evaluation:
- ZH-CER: (0.031455 − x)/(0.031455 − 0.015);
- retention: (x − 0.9616)/(0.98 − 0.9616);
- outside harm: (0.038339 − x)/(0.038339 − 0.03).

A measure counts as half-rescued if its fraction is ≥ 0.5. Partial rescue holds if at least 2/3 measures are
half-rescued and none is worse than A3 by more than 0.005 absolute.

**Benefit retention vs A2** (A2 is reused): POI(A4) ≤ POI(A2) + 2, AND MER(A4) − MER(A2) ≤ 0.005, AND no new severe
truncation.

## Labels (precedence)

1. **INVALID**: engineering, panel, reset, teacher, lang mismatch, theta0 anchor, live-check, reuse, leakage, audit or
   numerical failure.
2. **ENGLISH_TRANSFER_WEAK**: EN_TRANSFER fails.
3. **SHARED_PARAMETER_INTERFERENCE**: ANCHOR_PRESERVED fails.
4. **SEQUENCE_SAFETY_NOT_RESCUED**: FULL_MATRIX_RESCUE fails.
5. **SAFE_BUT_NO_GAIN**: benefit retention fails.
6. **PROMISING**: everything above passes, plus the independent audit PASS.

All labels are prefixed `P2_TTA_A4_`.

## Integrity, compute and firewall

- **Live check:** on row 0 (a D row with EN positions), an independent formula must agree with loss ≤ 1e-5 and gradient
  ≤ max(1e-8, 0.02‖g‖).
- **Other integrity:** theta0 forced = S0; references only after the committed output seal; bootstrap descriptive.
- **Compute:** one sbatch job, 24 rows, 48 optimizer steps, ≤ 49 backward passes; target < 10 min, hard limit 30 min.
- **Reuse:** B0/AUTO/A2/A3 are reused with hash checks against the A3 seal. A3 outputs are taken from A3 run1 rows.
- **Not run:** A1/A2/A3 reruns, full-100/300, P3.
- **Forbidden data:** D-dev-confirm, D-test, router-calib new role, new panel, transfer corpora, fresh validation.
