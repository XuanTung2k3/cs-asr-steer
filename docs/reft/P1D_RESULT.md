# P1-D — Conditional Controllability Audit: result

**Frozen verdict: `P1D_NO_CONVINCING_SIGNAL`.**

The proposed controller would read only z = V₃₁ᵀ·RMSNorm(h). Those four numbers do not tell when to intervene: error
vs correct AUROC is 0.564, the same as random rank-4 projections and a position-only baseline. They also do not tell how
to intervene: the predicted correction direction is no better than one constant direction. The full 1,280-dimensional
state carries clearly more information (AUROC 0.752, direction cosine 0.212). Within this protocol, that information is
not available through the frozen four-dimensional read.

**Status of this result.**
- It is an exploratory CPU-only diagnostic. The P1 audit outcomes had already been examined.
- It does not change the sealed P1 verdict (`P1_SUBSPACE_WEAK_OR_NARROW`), and it does not revise the failed preservation
  gate.
- No GPU, no Whisper pass, no change to V, layer or rank.

Protocol `P1D_PROTOCOL.md` was frozen before any fit (`freeze/p1d.json`, commit `403d900`). All probes, scalers,
hyperparameters and thresholds were fitted on construction rows and written to `p1d/frozen_models.json` before the
single audit scoring (Slurm 59182). Runtime: 39 s CPU.

## 1. Inputs and integrity: 17/17 checks pass (Slurm 59181)

- **Hashes:** rows, tensors, basis, controls and decode all match their manifests. V sha256 is `f020e93b…`, layer 31,
  rank 4, orthonormal.
- **Rows:** row order equals tensor order (N = 1,758). Partitions are unchanged, and construction (21 dialogues) and
  audit (6) are disjoint.
- **Counts:** construction A/B/P = 175/586/604; audit = 45/149/199.
- **Row definitions:**
  - every A competitor is AUTO's emitted token;
  - every P row's correct token is the argmax;
  - every B competitor is the argmax.
- **Inputs:** the predictor allowlist is h (through z), t, audio duration and the Auto-detected language. Labels, q,
  c*, categories, margins and IDs are never inputs. All features are finite.
- **Normalization:** RMSNorm(h) = h / sqrt(mean h² + 1e-6), the same for every projection.
- **Standardization:** construction rows only.

## 2. D1 — error-state separability

Primary population: audit A errors vs P correct decisions, 45 vs 199 (prevalence 0.184), 6 dialogues. Thresholds come
from construction out-of-fold scores.

| Probe (inputs) | Construction CV AUROC (C) | Audit AUROC | Average precision | Recall at τ₅ (achieved FPR) | Recall at τ₁₀ (achieved FPR) | Balanced accuracy |
|---|---|---:|---:|---|---|---:|
| **V (4)** | 0.581 (0.01) | **0.564** | 0.247 | 0.089 (0.050) | 0.178 (0.075) | 0.527 |
| R42–R46 (4 each) | 0.535–0.577 | 0.444 / 0.532 / 0.569 / 0.621 / 0.483 (mean 0.530) | 0.186–0.266 | 0.00–0.16 | 0.09–0.24 | 0.49–0.57 |
| NUIS (position, duration, LID: 5) | 0.602 | 0.546 | 0.270 | 0.089 (0.035) | 0.156 (0.080) | 0.536 |
| V+NUIS (9) | 0.613 | 0.578 | 0.309 | 0.156 | 0.200 | 0.565 |
| FULL Norm(h) (1,280; exploratory, not capacity-matched) | 0.777 (0.001) | **0.752** | 0.467 | 0.311 (0.050) | 0.400 (0.131) | 0.669 |

**Uncertainty.** Dialogue-cluster bootstrap with 2,000 resamples:
- AUROC_V: [0.48, 0.64];
- AUROC_V − mean random: [−0.05, +0.10];
- AUROC_V − NUIS: [−0.05, +0.07].

Leave-one-dialogue-out AUROC_V ranges from 0.53 to 0.59.

**Position confound.** It is weak in the primary population:
- median position is 3 for errors and 4 for correct decisions;
- support: 45 A rows across 6 bins;
- position-stratified AUROC: V 0.573, NUIS 0.606, random 0.45–0.67.

So the V probe has almost no error signal, with or without position control. That is a clear negative, not an
unresolved confound.

**Subgroup recall at τ₁₀**, computed on sealed predictions:

| Subgroup | n | Recall |
|---|---:|---:|
| CS errors | 36 | 0.19 |
| English-unit errors | 11 | 0.27 |
| Mandarin-unit errors | 32 | 0.16 |
| ZH_ONLY | 9 | 0.11 |

**FULL generalization.** In-sample AUROC 0.952 vs audit 0.752. The gap shows strong overfitting, but the held-out
signal is real.

**D1 core criteria (all fail):**

| Criterion | Required | Observed |
|---|---|---|
| AUROC | ≥ 0.65 | 0.564 |
| Beyond random | AUROC_V > every seed | R45 = 0.621 is higher |
| Recall at 10% FPR | ≥ 0.25 | 0.178 |
| LODO | every value ≥ 0.60 | 0.53–0.59 |

**→ D1 FAIL.**

**Secondary: B vs P** (gold-prefix teacher-forced errors; not pooled with A).
- V: AUROC 0.615.
- NUIS alone: AUROC 0.844. B errors sit late in the sequence (median t = 17 vs 4).
- V+NUIS: 0.844. Position-stratified V: 0.599.

This population is dominated by the sampling and position confound.

## 3. D2 — correction-coordinate predictability

Primary population: audit A errors, n = 45 (CS 36), 6 dialogues. No row was excluded for a negligible projected
gradient. Targets are c* = Vᵀq / ‖Vᵀq‖. Gains are first order at ε = 0.10·‖h‖, where the oracle in-V ceiling is
1.585 nats per row.

| Predictor | CV mean cos (α) | Audit mean cos (median) | cos > 0 | cos < 0 | First-order gain (nats) | Gain / ceiling |
|---|---|---|---:|---:|---:|---:|
| **V ridge (4 → 4)** | 0.162 (0.01) | **0.135** (0.152) | 0.62 | 0.38 | 0.238 | 0.150 |
| CONST (mean construction direction) | — | 0.110 (0.238) | 0.51 | 0.49 | 0.183 | 0.116 |
| R42–R46 (own subspace and targets) | 0.02–0.14 | −0.039 / 0.127 / 0.158 / 0.060 / 0.015 | 0.49–0.62 | | −0.03…0.05 (ceilings ≈ 0.35–0.39) | −0.07…0.14 |
| FULL Norm(h) → c* in V (exploratory) | 0.288 (1,000) | 0.212 (0.202) | 0.60 | 0.40 | 0.195 | 0.123 |
| Oracle P_V q (P1 Test 2 at ε = 0.10) | | 1 | | | 1.585 | 1 |

**Uncertainty.**
- mean cos_V: [−0.02, +0.30];
- mean cos_V − CONST: **[−0.06, +0.09]**;
- leave-one-dialogue-out differences against CONST: +0.004 to +0.062.

The point estimate is above CONST, but the interval contains zero.

**Opposite-sign targets at similar states.** For 31% of audit errors, the nearest construction error in z-space has an
oppositely signed target (mean target cosine 0.23). Nearby projected states often need different corrections.

**Breakdown** (tiny subsets, no strong claims):

| Subgroup | n / dialogues | Mean cos V | Mean cos CONST | cos > 0 |
|---|---|---:|---:|---:|
| CS | 36 / 6 | 0.121 | 0.060 | 0.58 |
| English-unit | 11 / 5 | 0.229 | 0.172 | 0.55 |
| Translation-like | 5 / 3 | 0.560 | 0.259 | 0.80 |
| **English lexical substitution** | **4 / 3** | **−0.179** | 0.176 | 0.25 |
| Mandarin lexical substitution | 16 / 5 | 0.145 | 0.354 | 0.69 |
| Mandarin-unit | 32 / 6 | 0.070 | 0.090 | 0.63 |

The only clearly positive slice is the 5 translation-like rows, the language-preference case. For English→English and
Mandarin substitutions, z does worse than a constant direction.

**Per audit dialogue** (anonymized as AD1–AD6), mean cos V vs CONST:

| Dialogue | n | V | CONST |
|---|---:|---:|---:|
| AD1 | 8 | 0.01 | 0.16 |
| AD2 | 11 | 0.19 | 0.14 |
| AD3 | 2 | 0.07 | 0.14 |
| AD4 | 7 | 0.15 | 0.01 |
| AD5 | 9 | −0.11 | −0.12 |
| AD6 | 8 | 0.46 | 0.37 |

**D2 criteria:**

| Criterion | Result | Observed |
|---|---|---|
| Beats CONST with bootstrap lower bound > 0 | **fail** | lower bound −0.06 |
| Beats every random seed | **fail** | R44 = 0.158 > 0.135 |
| cos > 0 for ≥ 60% of rows | pass | 0.62 |
| Every LODO difference vs CONST > 0 | pass | |
| CS beats CONST | pass | |

**→ D2 not PASS.**

**Secondary B** (n = 149): V 0.118 vs CONST 0.123 (difference [−0.09, +0.07]); FULL 0.258.

## 4. Combined selectivity (offline information audit; no Whisper execution)

D1-V selects rows by its construction-frozen threshold, and D2-V supplies the update direction.

| Threshold | A selected | P selected | Selected A: cos + / − | Selected A: mean first-order gain | Selected P: predicted change in correct margin | Selected P: fraction predicted damaged |
|---|---|---|---|---:|---:|---:|
| τ₁₀ | 8/45 (4 dialogues) | 15/199 | 6 / 2 | +0.50 nats | −0.26 nats | 0.40 |
| τ₅ | 4/45 (3 dialogues) | 10/199 | 4 / 0 | +1.08 nats | −0.45 nats | 0.50 |

At either threshold, the policy touches about twice as many correct positions as errors, and it predicts damage on 40–50%
of the correct positions it touches. Selected errors by category at τ₁₀: Mandarin substitution 4, English substitution
2, translation-like 1, Mandarin omission 1.

## 5. Interpretation

1. **When:** the four controller-visible coordinates contain essentially no error-state information beyond random
   rank-4 projections or position.
2. **How:** they predict the in-V correction direction no better than a single fixed direction. A fixed direction is
   exactly the global language-preference edit that P1 already found to damage correct tokens.
3. **Information exists elsewhere in h.** The exploratory FULL readout reaches AUROC 0.752 and direction cosine 0.212
   (0.258 on B). The information needed to choose when to act, and somewhat how, lies outside span(V₃₁) at layer 31.
   It is not shown to be usable without overfitting: the FULL in-sample AUROC is 0.95.
4. **Preservation.** P1's failed gate stands. P1-D adds that the planned 4-dimensional read gives the controller no
   evident means of avoiding that damage: at matched low FPR, the selected updates still push correct margins down.

**Limits.**
- 6 audit dialogues and 45 A rows. Only 11 are English-unit errors, and 4 are English lexical substitutions.
- First-order, teacher-forced quantities.
- A and P come from different sampling designs (first divergence vs matched prefix). The position control shows little
  confound for A vs P, but strong confound for B vs P.

## 6. Recommended next action (one)

**Do not train the projected-only 20-parameter controller (`h + α·V·tanh(A·Vᵀ·Norm(h) + b)`).** The single next step is
a project-owner decision on whether to re-scope P2 to a **decoupled read/write controller**:
- it reads conditioning information from the full `Norm(h)` (the only input with held-out signal here) through a read
  capacity fixed in advance in a new protocol;
- it writes only through the frozen V₃₁;
- it carries the P1 preservation monitoring forward.

That would be a new pre-registered experiment. It does not continue P1 or P1-D, and none of it has been run.
