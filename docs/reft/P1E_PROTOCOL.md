# P1-E — Decoupled Conditioning/Correction Subspace Feasibility (frozen protocol)

**Status.** FROZEN before the conditioning basis U is built and before any P1-E audit metric is computed.
- Freeze record: `/mnt/data/tungnx/cs-asr-reft/freeze/p1e.json`.
- Constants: `configs/reft/p1e.yaml`.
- Code: `reft/decoupled_audit.py`.

**Type.** Exploratory model development, CPU only.
- The six audit dialogues were already examined in P1 and P1-D. They are **not** an independent confirmation set.
- No GPU, Whisper pass, gradient extraction, change to V, rank or layer search, or P2.
- Earlier freezes and results are untouched.

## 1. Question and architecture

The proposed controller reads through a **conditioning (read) basis U** and writes through the **frozen correction
(write) basis V**:

```
z = Uᵀ x,                x = RMSNorm(h) = h / sqrt(mean h² + 1e-6)
h̃ = h + α · V · tanh(A z + b),   with A ∈ ℝ^{4×4}, b ∈ ℝ^4
```

The site is decoder layer 31, post-cross-attention and pre-FFN. V is the P1 layer-31 basis (sha256 `f020e93b…`).

**Question:** does a compact read basis U, built from construction data, carry enough held-out WHEN (error vs correct)
and HOW (in-V correction direction) information for a selective, preservation-aware action?

## 2. Data roles and integrity

- **Sources:** P1 layer-31 rows and tensors, V, the random controls R42–R46 and the panel (through the P1-D loader,
  which re-checks every hash).
- **P1-D artifacts:** hashes are checked against `p1d/manifest.json`.
- **Partitions:**
  - construction: 21 dialogues, A/B/P = 175/586/604;
  - audit: 6 dialogues, 45/149/199.
- **Primary population:** A vs P. B is reported separately and never pooled. **U is built from construction rows
  only.**
- **Predictor inputs:** x (through Uᵀ x, Vᵀ x, Rᵀ x, or the full x) and the P1-D nuisance metadata, all
  reference-free. Never inputs: labels, q, c*, categories, margins, IDs or target length.

## 3. U construction (primary, single method)

1. **WHEN reader.**
   - The P1-D FULL logistic on standardized x (balanced class weights, C = 0.001), fit on construction A vs P.
   - It must reproduce the P1-D construction statistics within 1e-9: grouped OOF AUROC 0.776632, in-sample AUROC
     0.952460. Otherwise the run stops (`P1E_INCONCLUSIVE`).
2. **HOW reader.**
   - The P1-D FULL ridge on standardized x (α = 1000) → `c* = Vᵀq / ‖Vᵀq‖`, fit on construction A rows.
   - It must reproduce the grouped OOF mean cosine 0.288104 within 1e-9.
3. **Unscaling to raw x coordinates.**
   - Coefficients: `w_raw = w_std / σ` and `W_raw = W_std / σ` (column-wise), with σ the construction scaler
     standard deviations.
   - Intercepts: `b_raw = b − Σ_j w_std,j μ_j / σ_j`.
   - The intercepts are recorded, but they do not enter U.
4. **Block normalization.** `B = [w_raw / ‖w_raw‖₂ ; W_raw / ‖W_raw‖_F]` ∈ ℝ^{5×1280}. WHEN and HOW get equal total
   energy, with no outcome-dependent weighting.
5. **Basis.**
   - SVD `B = P Σ Rᵀ`, with **U = the first 4 right singular vectors** (rank fixed at 4).
   - Sign: the largest-magnitude coordinate of each column is positive.
   - Checks: UᵀU = I (< 1e-10), finite, shape [1280, 4]. The hash is recorded.
6. **Caveat.** B has rank ≤ 5, so a rank-4 U is not evidence of a hidden low-dimensional mechanism. Only held-out
   performance counts.

## 4. E1 — conditioning-subspace diagnostics (reported, not gated)

- singular values of B and the energy retained by rank 4;
- principal cosines of U vs V and of U vs R42–R46;
- how much of each reader's raw weight vector or matrix lies in span(U);
- z_U distributions by partition and class;
- construction→audit standardized mean shift;
- per-coordinate point-biserial correlation with the A/P label on construction rows (label-axis dominance).

## 5. Probes (construction-only fitting; P1-D conventions)

- **WHEN probe.** `LogisticRegression` (lbfgs, balanced, max_iter 5000) on standardized 4-d features.
  - C is chosen from {1e-3 … 100} by `GroupKFold(5)` over construction dialogues, maximizing pooled OOF AUROC; ties go to
    the smaller C.
  - The model is refit on all construction rows.
  - Thresholds τ₅ / τ₁₀ are the 95th / 90th percentiles of OOF scores on construction P rows; τ_bal maximizes OOF
    balanced accuracy.
- **HOW probe.** `Ridge` (4 → 4, standardized inputs) to c* in **V coordinates**.
  - α is chosen from {1e-2 … 1e3} by `GroupKFold(5)`, maximizing the mean OOF cosine; ties go to the larger α.
  - The model is refit on all construction rows.
- **Read sets** (the write basis is **always V**):
  - U (primary);
  - V (the coupled P1-D design);
  - R42–R46 (random read);
  - FULL x (the reconstructed WHEN / HOW readers; exploratory, not capacity-matched);
  - NUIS and U+NUIS (WHEN only);
  - CONST, the mean construction c* (HOW only).

## 6. E2 — WHEN (audit A vs P)

**Reported:**
- AUROC, average precision (prevalence 0.184), and recall at τ₅ / τ₁₀ with the achieved FPR;
- balanced accuracy at τ_bal;
- dialogue-cluster bootstrap (2,000 resamples, seed 42) of AUROC_U, AUROC_U − mean_s AUROC_Rs and AUROC_U − AUROC_NUIS;
- leave-one-dialogue-out (LODO) AUROC_U;
- position-stratified AUROC (P1-D bins), and AUROC(U+NUIS) − AUROC(NUIS);
- recall on CS, unit-EN, unit-ZH and ZH_ONLY errors.

**E2 PASS** requires all of the following:
1. AUROC_U ≥ 0.65, with bootstrap lower bound > 0.5.
2. AUROC_U > AUROC_V and > max_s AUROC_Rs, with bootstrap lower bound of AUROC_U − mean_s AUROC_Rs > 0.
3. Bootstrap lower bound of AUROC_U − AUROC_NUIS > 0.
4. Recall at τ₁₀ ≥ 0.25, and greater than the maximum random-seed recall at its own τ₁₀.
5. Every LODO AUROC_U ≥ 0.60.
6. Position support (≥ 20 A rows in ≥ 2 bins), stratified AUROC_U ≥ 0.60, and AUROC(U+NUIS) − AUROC(NUIS) ≥ 0.02.

## 7. E3 — HOW (audit A rows; targets c* in V)

**Reported:**
- mean / median cos(û, c*) and the fractions > 0 and < 0;
- first-order gain `g = ε‖h‖ · qᵀ V û` (ε = 0.10), against the fixed-V oracle ceiling `ε‖h‖ · ‖Vᵀq‖`, and their ratio;
- improvement over CONST, with the dialogue bootstrap and LODO;
- breakdowns: CS, translation-like, non-translation, en / zh lexical substitution, unit-EN / ZH;
- the fraction of audit A rows whose nearest construction A row in standardized z_U has an opposite-signed target.

**E3 PASS** requires all of the following:
1. Mean cos_U > mean cos_CONST, with bootstrap lower bound of the difference > 0.
2. Mean cos_U > max_s mean cos(R_s read → V).
3. Mean cos_U > mean cos_V (the coupled P1-D design).
4. frac(cos_U > 0) ≥ 0.60.
5. Every LODO difference against CONST > 0.
6. CS rows: mean cos_U > CONST.
7. Non-translation rows (excluding translation-like and zh_to_latin): mean cos_U > CONST.

## 8. E4 — combined offline selectivity

**Shared definitions** (first-order, saved q and h only):
- For a selected row with unit V-coefficient direction û, the update is `ε‖h‖ V û`.
- A gain: `ε‖h‖ q_Aᵀ V û`.
- P change: `ε‖h‖ q_Pᵀ V û`, where q_P is the gradient of the correct-vs-runner-up margin. A negative value is predicted
  damage.
- **Net first-order benefit:** `NB = (Σ_selected A gain + Σ_selected P change) / (n_A + n_P)` over audit A ∪ P.

**E4-A (separate readers; reported, not gated).** The U WHEN probe selects rows at τ₅ / τ₁₀, and the U HOW probe gives
û. It is compared with the same pipeline for V (P1-D). Its readout parameters are 5 + 20.

**E4-B (joint 20-parameter offline action head; gated).**
- Head: `a = tanh(A z̃ + b)`, where z̃ is the construction-standardized z. The standardization folds into A and b, so the
  controller still has 20 parameters.
- Targets: c* for A rows and 0 for P rows.
- Loss: `Σ_i w_i ‖a_i − target_i‖² + λ‖A‖_F²`, with w = 0.5/n_A per A row and 0.5/n_P per P row within each fitted
  set.
- Fit: float64 L-BFGS (strong-Wolfe line search) from A = 0, b = 0. It is deterministic.
- λ is chosen from {1e-4, 1e-3, 1e-2, 0.1, 1} by `GroupKFold(5)`, minimizing the pooled OOF weighted loss; ties go to
  the larger λ.
- Selection: ‖a‖ ≥ τ, with τ₅ / τ₁₀ the 95th / 90th percentiles of OOF ‖a‖ on construction P rows. Direction
  û = a / ‖a‖.
- Read sets: U (primary), V (coupled), R42–R46 (random read). All write V.
- Also reported:
  - CONST applied to every row;
  - CONST applied at the U-head's selected rows;
  - the magnitude-scaled variant `ε‖h‖ V a` (the controller's actual output magnitude, ‖a‖ ≤ 2) over all rows.

**E4-B PASS** requires all of the following, at τ₁₀ on audit:
1. **J1:** recall of A ≥ 0.25 and achieved FPR on P ≤ 0.15.
2. **J2:** selected A rows have frac(cos > 0) ≥ 0.60 and mean gain > 0.
3. **J3:** NB_U > 0, with dialogue-bootstrap lower bound > 0.
4. **J4:** NB_U > NB_V and NB_U > max_s NB_Rs, and recall_U − recall_V ≥ 0.10.
5. **J5:** at least 3 selected non-translation A rows with cos > 0, and selected A rows spanning ≥ 3 dialogues.

## 9. Verdict (frozen)

| Condition | Verdict |
|---|---|
| Integrity or reader reproduction fails | `P1E_INCONCLUSIVE` |
| E2, E3 and E4-B pass | `P1E_DECOUPLED_FEASIBLE_EXPLORATORY` |
| E2 passes, but E3 or E4-B does not | `P1E_WHEN_ONLY` |
| E3 passes and E2 does not | `P1E_HOW_ONLY` |
| Otherwise | `P1E_NO_CONVINCING_SIGNAL` (this includes E4-B passing alone, which cannot occur without E2 or E3 evidence and would be reported as an anomaly) |

No verdict authorizes P2. The P1 preservation gate failure and the P1-D verdict stand.

## 10. Cost accounting (reported)

- **Fitted to build U:**
  - WHEN reader: 1,281 parameters;
  - HOW reader: 4 × 1,280 + 4 = 5,124 parameters;
  - their scaler statistics;
  - CPU time.
- **Stored bases:** U and V, 5,120 coefficients each.
- **Controller:** A and b, 20 parameters.
- **Probes:** WHEN 5, HOW 20, joint head 20.

The total supervised adaptation cost is not "20 parameters".

## 11. Limits

- 6 previously examined audit dialogues and 45 A rows. Only 11 are English-unit errors, 4 of them English lexical
  substitutions.
- A and P come from different sampling designs.
- All quantities are first-order and teacher-forced. There is no decoding claim.
