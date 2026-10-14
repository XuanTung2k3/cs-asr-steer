# P1-D — Conditional Controllability Audit (frozen protocol)

**Status.** FROZEN before any P1-D probe is fit or scored.
- Freeze record: `/mnt/data/tungnx/cs-asr-reft/freeze/p1d.json`.
- Constants: `configs/reft/p1d.yaml`.
- Code: `reft/conditional_audit.py`.

**Type.** This is a new **exploratory** CPU-only diagnostic. The P1 (layer 31) audit outcomes were already examined.
P1-D is therefore not an independent confirmation, and it does not replace the sealed P1 verdict
(`P1_SUBSPACE_WEAK_OR_NARROW`).

**What P1-D never does:**
- no GPU work, no Whisper forward pass, no new data;
- no change to V, the layer or the rank;
- no preservation-aware subspace, no P2 controller.

## 1. Question

The proposed P2 controller reads only `z = Vᵀ Norm(h)`, where V ∈ ℝ^{1280×4} is the frozen layer-31 gradient-SVD basis
(sha256 `f020e93b…`). P1-D asks whether those four numbers carry enough information:

- **D1 — error-state separability.** Can z tell lexical errors (A rows) from already-correct decisions (P rows)?
- **D2 — correction-coordinate predictability.** Given an error, can z predict the oracle in-subspace correction
  direction `c* = Vᵀq / ‖Vᵀq‖`?

Neither is free-decoding evidence. D1 does not imply D2.

## 2. Inputs and integrity (`audit` step; any failure → `P1D_INCONCLUSIVE`)

**Source artifacts** (layer 31 only, never layer 24), each hash-checked against its own manifest:
- `p1/l31/gradients/{rows.jsonl, tensors.pt}`
- `p1/l31/basis_rank4.pt` and `controls.pt`
- `p1/l31/basis_manifest.json`
- `p1/panel.json` (durations)

**Checks:**
- V sha256 equals `f020e93b…`;
- layer 31 and rank 4 (basis file and manifests);
- tensor rows equal metadata rows (`row` index = position; N = 1,758);
- partitions unchanged against `p1/panel.json`;
- construction and audit dialogues are disjoint;
- per-partition A/B/P counts equal the gradient manifest (construction 175/586/604, audit 45/149/199);
- A rows are free-decoding first divergences (`competitor` = AUTO's emitted token); P rows have `correct_is_argmax`;
- all features finite;
- the predictor inputs come from an allowlist (§3) with no reference-derived column.

## 3. Features (controller-visible input)

**Normalization.** `Norm(h) = h / sqrt(mean_j h_j² + 1e-6)`: per-position RMS, parameter-free, ε = 1e-6. This becomes
the `Norm` of the future P2 controller.

**Feature sets:**
- `z_V = Vᵀ Norm(h)`: 4 features (primary);
- `z_Rs = R_sᵀ Norm(h)`: 4 features each, for the P1 random orthonormal bases with seeds 42–46;
- `Norm(h)`: 1,280 features; the full-state, exploratory information upper bound, not parameter-matched.

**Nuisance features (D1 control).** These are reference-free metadata available at decoding time:
- content position t, and log(1 + t);
- utterance audio duration (s);
- t / duration;
- whether Auto's detected language is `zh`.

**Standardization.** Every model standardizes its inputs with mean and standard deviation from the **construction rows
of its own training population only**, then applies them unchanged to audit rows.

**Never an input** (allowed only for supervision, grouping or post-hoc reporting):
- reference or competitor tokens and their text;
- correctness / source label;
- q, g_s and c*;
- category, unit language, stratum;
- margin m₀;
- utterance or dialogue ID;
- target length (reference-derived).

## 4. Populations

| | Construction | Audit (6 dialogues) |
|---|---|---|
| D1 primary: A (label 1) vs P (label 0) | 175 vs 604 | 45 vs 199 |
| D1 secondary: B vs P (B = teacher-forced gold-prefix errors, separately) | 586 vs 604 | 149 vs 199 |
| D2 primary: A | 175 | 45 |
| D2 secondary: B | 586 | 149 |

A and B are never pooled. A row is excluded from D2 if `‖Vᵀq‖ / ‖q‖ < 1e-6`; the same tolerance applies in each random
subspace. Excluded counts are reported.

## 5. D1 — error-state separability

**Probe.** L2 logistic regression `s = wᵀz + b` (scikit-learn `LogisticRegression`, lbfgs, `class_weight="balanced"`,
max_iter 5000) on standardized inputs.

**Regularization.**
- C is chosen by `GroupKFold(5)` over construction dialogues, maximizing pooled out-of-fold (OOF) AUROC.
- Ties go to the smaller C.
- Grids: low-dimensional inputs {1e-3, 1e-2, 1e-1, 1, 10, 100}; full state {1e-4, 1e-3, 1e-2, 1e-1}.
- The model is then refit on all construction rows.

**Thresholds** come from the selected configuration's construction OOF scores only:
- τ₅ / τ₁₀: the 95th / 90th percentile of OOF scores on construction P rows (5% / 10% construction FPR);
- τ_bal: the threshold maximizing OOF balanced accuracy.

None is tuned on the audit set.

**Models**, all with the identical procedure:
- V;
- R42–R46;
- NUIS (nuisance only);
- V+NUIS (9 features);
- FULL (`Norm(h)`, 1,280 features).

**Audit metrics**, each computed once:
- AUROC;
- average precision (prevalence 45/244 = 0.184; the sampled population is not deployment precision);
- recall of A at τ₅ / τ₁₀ with the achieved audit FPR on P;
- balanced accuracy at τ_bal;
- recall on CS A rows and on EN / ZH A rows (by unit language);
- for FULL, construction in-sample vs audit AUROC.

**Position confound.**
- P rows come only from matched prefixes before the first divergence, so positions differ systematically from A rows.
- Positions are binned with frozen bins: t ∈ {0}, {1–2}, {3–5}, {6–10}, {11–20}, {21+}.
- The position-stratified AUROC is the pair-weighted mean of within-bin AUROCs over bins containing both classes.
- Supported A rows are the audit A rows in such bins.
- Also reported: AUROC(V+NUIS) − AUROC(NUIS).

**Uncertainty.**
- Dialogue-cluster bootstrap over the 6 audit dialogues (2,000 resamples, seed 42) for AUROC_V, AUROC_V − mean_s
  AUROC_Rs, and AUROC_V − AUROC_NUIS.
- Leave-one-dialogue-out (LODO) AUROC_V.
- A resample lacking a class is skipped and counted.

## 6. D2 — correction-coordinate predictability

**Target.** `c* = Vᵀq / ‖Vᵀq‖` (supervision only).

**Probe.** Ridge regression `ĉ = M z + b` (M 4×4, b ∈ ℝ⁴; scikit-learn `Ridge`) on standardized z.
- α is chosen by `GroupKFold(5)` over construction dialogues, maximizing the mean OOF cosine(ĉ, c*).
- Ties go to the larger α.
- Grids: 4-dimensional input {1e-2, 1e-1, 1, 10, 100, 1000}; full state {1, 10, 100, 1e3, 1e4, 1e5}.
- Refit on all construction rows.
- `û = ĉ / ‖ĉ‖`; if ‖ĉ‖ = 0, û = 0, which counts as zero alignment.

**Baselines:**
- **CONST:** `c̄ = normalize(mean of construction c*)`, the same for every row.
- **R42–R46:** the same pipeline inside each random subspace (input `R_sᵀ Norm(h)`, target `R_sᵀq / ‖R_sᵀq‖`).
- **FULL:** `Norm(h)` (1,280 features) → c* in V. Exploratory, not capacity-matched.

**Audit metrics:**
- **Alignment:**
  - mean and median cos(û, c*);
  - fraction > 0 and fraction < 0;
  - the same for CONST.
- **First-order gain** at ε = 0.10:
  - `g = ε·‖h‖·qᵀ(V û)` in nats, against the oracle ceiling `ε·‖h‖·‖Vᵀq‖`;
  - the ratio of mean gains;
  - this is not a measured token flip.
- **Random subspaces:** each is scored against its own ceiling, and its absolute gain is also reported.
- **Opposite-sign targets at similar states:** for each audit A row, its nearest construction A row in standardized z;
  report the fraction of nearest-neighbour pairs with cos(c*_i, c*_nn) < 0, and the mean.
- **Breakdowns** (counts shown): CS, translation-like, en lexical substitution, zh lexical substitution, unit language
  EN / ZH, per dialogue.
- **Uncertainty:** dialogue bootstrap (2,000 resamples, seed 42) for mean cos_V − mean cos_CONST and for the mean
  cos_V; LODO for the difference against CONST.
- **Secondary:** B rows, reported separately and never in the headline.

## 7. Combined selectivity (offline information audit, audit rows only)

D1-V scores audit A and P rows. Rows with score ≥ τ₁₀ (primary; τ₅ is also reported) are selected, and D2-V gives û for
every selected row.

**Report:**
- fraction of A selected; fraction of P selected;
- on selected A rows: the sign of cos(û, c*) and the mean predicted first-order gain;
- on selected P rows: the first-order change of the correct margin `ε·‖h‖·q_Pᵀ(V û)`, where `q_P` is the stored
  gradient of the correct-vs-runner-up margin, as a mean and the fraction that is negative (predicted damage);
- coverage by dialogue and by category.

No oracle quantity or label drives the selection, and nothing is run through Whisper.

## 8. Decision rules (prospective; primary = A-row populations, V features)

**D1 core**, all required:
1. AUROC_V ≥ 0.65, and the bootstrap lower bound of AUROC_V > 0.5.
2. AUROC_V > max_s AUROC_Rs, and the bootstrap lower bound of AUROC_V − mean_s AUROC_Rs > 0.
3. Recall of A at τ₁₀ ≥ 0.25, and greater than the maximum random-seed recall at its own τ₁₀.
4. Every LODO AUROC_V ≥ 0.60.

**D1 position control:**
- **Support** requires ≥ 20 supported audit A rows in ≥ 2 bins with both classes. Without support, D1 is
  **CONFOUNDED_UNRESOLVED**.
- **Pass** requires position-stratified AUROC_V ≥ 0.60 and AUROC(V+NUIS) − AUROC(NUIS) ≥ 0.02.

**D1 status:**
- PASS if the core and the position control pass;
- FAIL if the core fails, or the core passes but the supported position control fails (the signal is explained by
  position);
- CONFOUNDED_UNRESOLVED if the core passes but support is insufficient.

**D2 PASS** requires all of the following:
1. Mean cos_V > mean cos_CONST, with the bootstrap lower bound of the difference > 0.
2. Mean cos_V > the maximum over seeds of mean cos_Rs.
3. frac(cos_V > 0) ≥ 0.60.
4. Every LODO difference against CONST > 0.
5. On CS A rows, mean cos_V > mean cos_CONST (point estimate).

**Verdict:**
- integrity fails → `P1D_INCONCLUSIVE`;
- D1 PASS and D2 PASS → `P1D_CONDITIONAL_INFORMATION_SIGNAL`;
- D1 PASS and D2 not PASS → `P1D_ERROR_DETECTION_ONLY`;
- D1 FAIL or CONFOUNDED_UNRESOLVED, and D2 PASS → `P1D_DIRECTION_ONLY`;
- D1 FAIL and D2 not PASS → `P1D_NO_CONVINCING_SIGNAL`;
- D1 CONFOUNDED_UNRESOLVED and D2 not PASS → `P1D_INCONCLUSIVE`.

The verdict never turns the failed P1 preservation gate into a pass. The FULL, B-row and combined analyses are reported,
but they do not enter the verdict.

## 9. Limits fixed in advance

- 6 audit dialogues, 45 A rows. Only 11 of them are English decisions and 4 are English lexical substitutions.
- The A/P sampling design differs: first divergence vs matched-prefix correct positions.
- Gradients and gains are first-order, from teacher-forced local passes.
- There is no free decoding.
