# P1-E — Decoupled Conditioning/Correction Subspace Feasibility: result

**Frozen verdict: `P1E_WHEN_ONLY`.**

A read basis U (rank 4), built from construction-only full-state readers, recovers the full-state error signal through
four numbers:
- held-out error-vs-correct AUROC 0.757 (V-read 0.564, random 0.44–0.62, position-only 0.546, full state 0.752).

It does **not** recover a reliable correction direction in the frozen write basis V:
- cosine 0.168 vs a constant direction's 0.110 (difference CI [−0.02, +0.15]);
- only 53% positive.

The single 20-parameter action head is not selective enough:
- 7/45 errors and 15/199 correct positions selected;
- net first-order benefit CI includes 0;
- it does not beat the random-read heads.

Under the frozen rules, the existing tiny-controller architecture does not proceed.

**Status of this result.**
- Exploratory, CPU only. The six audit dialogues were already examined in P1 and P1-D, so this is not a confirmation.
- No GPU, Whisper pass, new gradients, change to V, rank/layer search or P2.
- P1 (`P1_SUBSPACE_WEAK_OR_NARROW`, preservation gate failed) and P1-D (`P1D_NO_CONVINCING_SIGNAL`) stand.

**Provenance.**
- Protocol `P1E_PROTOCOL.md` frozen at commit `b32d2d6` (`freeze/p1e.json`) before U was built.
- Slurm jobs: integrity 59199, build 59200, single scoring run 59202.
- CPU time: 3.3 s to build U, 23.7 s for the run.

## 1. Inputs and integrity: all checks pass

- **P1 layer-31 artifacts:**
  - rows, tensors, basis and controls are hash-matched;
  - V sha256 `f020e93b…`, layer 31, rank 4, orthonormal;
  - row order and partitions unchanged; construction (21 dialogues) and audit (6) disjoint;
  - A/B/P counts 175/586/604 and 45/149/199, with the A/B/P definitions re-verified.
- **P1-D outputs:** all hashes match. The P1-D FULL readers' coefficients had not been serialized, so they were refit
  with the frozen rules (C = 0.001, α = 1000). They reproduced the recorded construction statistics **exactly** (error
  0.0): grouped OOF AUROC 0.7766, in-sample 0.9525, OOF HOW cosine 0.2881.
- **Inputs:** predictors read only RMSNorm(h) projections or nuisance metadata. Layer-24 artifacts were not read.

## 2. U construction (E1)

| Item | Value |
|---|---|
| Method | WHEN logistic (construction A vs P, 779 rows) and HOW ridge (construction A → c* in V, 175 rows). Coefficients unscaled to raw RMSNorm(h) coordinates (max reconstruction error 3e-15). B = [w / ‖w‖ ; W / ‖W‖_F] ∈ ℝ^{5×1280}. U = top-4 right singular vectors. |
| Singular values of B | 1.002, 0.619, 0.508, 0.428, 0.413. Rank 5; rank 4 keeps 91.5% of energy (B has rank ≤ 5 by construction, so this is not evidence of structure). |
| U sha256 | `583214de840702ba4b34e33e69f750d52cd9967cdbda21f67cbcec7cf6c85324`; UᵀU = I to 1.6e-15 |
| U vs V principal cosines | 0.148, 0.081, 0.042, 0.032: U is almost orthogonal to V; read and write are nearly disjoint |
| U vs random R42–R46 | top cosines 0.07–0.11 |
| Label axis | point-biserial correlations of z_U with A/P (construction): −0.68, 0.01, 0.06, 0.10. **One axis carries the WHEN signal.** |
| Construction → audit shift of z_U | standardized mean shifts −0.13 to +0.11 |

**Supervised extraction cost.**
- 1,281 WHEN reader parameters and 5,124 HOW reader parameters, a total of **6,405 fitted parameters**, plus 5,120
  scaler statistics.
- 3.3 s CPU.
- Stored artifact: U 5,120 + V 5,120 + A, b 20 = **10,260 coefficients**.

The adaptation procedure is therefore not a 20-parameter method: 20 is only the controller.

## 3. E2 — WHEN (audit: 45 A errors vs 199 correct; prevalence 0.184; 6 dialogues)

| Read features | Audit AUROC | Average precision | Recall at τ₅ (achieved FPR) | Recall at τ₁₀ (achieved FPR) | Balanced accuracy |
|---|---:|---:|---|---|---:|
| **U (4)** | **0.757** | 0.461 | 0.356 (0.085) | **0.467 (0.146)** | 0.703 |
| V (4; P1-D) | 0.564 | 0.247 | 0.089 (0.050) | 0.178 (0.075) | 0.527 |
| R42–R46 (4) | 0.44–0.62 | 0.19–0.27 | 0.00–0.16 | 0.09–0.24 | 0.49–0.57 |
| NUIS (position, duration, LID) | 0.546 | 0.270 | 0.089 (0.035) | 0.156 (0.080) | 0.536 |
| U+NUIS | 0.765 | 0.469 | 0.333 | 0.511 | 0.708 |
| FULL (1,280; exploratory) | 0.752 | 0.467 | 0.311 (0.050) | 0.400 (0.131) | 0.669 |

**Uncertainty.** Dialogue-cluster bootstrap with 2,000 resamples:
- AUROC_U: [0.66, 0.85];
- U − mean random: [+0.16, +0.30];
- U − NUIS: [+0.08, +0.31];
- U − V: [+0.10, +0.27].

Leave-one-dialogue-out AUROC_U ranges from 0.73 to 0.80.

**Position control.** Position-stratified AUROC_U is 0.780 over 45 supported A rows in 6 bins (V 0.573, NUIS 0.606).
U+NUIS − NUIS = +0.22. **E2 passes all six criteria.**

**Subgroup recall at τ₁₀:**

| Subgroup | n | Recall |
|---|---:|---:|
| CS errors | 36 | 0.56 |
| English-unit errors | 11 | 0.73 |
| Mandarin-unit errors | 32 | 0.38 |
| ZH_ONLY | 9 | 0.11 |

Caveat: the τ₁₀ threshold produced 14.6% audit FPR rather than about 10%. This is the miscalibration explained in §6.

## 4. E3 — HOW (audit: 45 A errors; targets c* in V; oracle ceiling 1.585 nats at ε = 0.10)

| Read → V coefficients | Mean cos (median) | cos > 0 / < 0 | First-order gain (nats) | Gain / ceiling |
|---|---|---|---:|---:|
| **U** | **0.168** (0.209) | 0.53 / 0.47 | 0.210 | 0.133 |
| V (coupled, P1-D) | 0.135 (0.152) | 0.62 / 0.38 | 0.238 | 0.150 |
| CONST (mean direction) | 0.110 (0.238) | 0.51 / 0.49 | 0.183 | 0.116 |
| R42–R46 read, write V | 0.042 / 0.156 / 0.070 / 0.140 / 0.153 | | 0.04–0.34 | 0.02–0.21 |
| FULL (exploratory) | 0.212 (0.202) | 0.60 / 0.40 | 0.195 | 0.123 |

**Uncertainty.**
- U − CONST: [−0.02, +0.15]; U − V: [−0.08, +0.16].
- Leave-one-dialogue-out U − CONST: +0.03 to +0.09.
- For 40% of audit errors, the nearest construction error in z_U needs an opposite-signed correction.

**Breakdown:**

| Slice | n | cos U | cos V | cos CONST | cos FULL |
|---|---:|---:|---:|---:|---:|
| CS | 36 | 0.172 | 0.121 | 0.060 | 0.212 |
| Translation-like | 5 | 0.520 | 0.560 | 0.259 | 0.506 |
| Non-translation | 37 | 0.119 | 0.095 | **0.140** | 0.185 |
| English lexical substitution | **4** | −0.086 | −0.179 | 0.176 | 0.045 |
| Mandarin lexical substitution | 16 | 0.184 | 0.145 | **0.354** | 0.234 |
| English-unit | 11 | 0.302 | 0.229 | 0.172 | 0.315 |

**E3 criteria:**
- **Fail:**
  - beats CONST with a positive bootstrap lower bound (lower bound −0.02);
  - positive fraction ≥ 0.60 (observed 0.53);
  - non-translation rows beat CONST (0.119 < 0.140).
- **Pass:** beats random read, beats coupled V, LODO, CS.

**→ E3 FAIL.** Outside the 5 translation-like rows, a constant direction is as good or better. Even the full state predicts
the in-V direction only weakly (0.212).

## 5. E4 — combined offline selectivity (first-order; ε = 0.10 normalized budget)

Net first-order benefit (NB) is the sum of first-order margin changes over selected rows, divided by the number of audit
A ∪ P rows. Damage on correct rows is negative.

| Policy (audit, τ₁₀ unless noted) | Errors selected | Correct selected (FPR) | Selected errors cos + / − | Mean gain on selected errors | Mean change on selected correct | Correct predicted damaged | NB [95% CI] |
|---|---|---|---|---:|---:|---:|---|
| **E4-B U joint 20-param head** | **7/45** | 15/199 (0.075) | 6 / 1 | +1.06 | −0.23 | 0.60 | **+0.016 [−0.008, +0.034]** |
| E4-B V joint head (coupled) | 5/45 | 9/199 (0.045) | 4 / 1 | +0.10 | −0.47 | 0.44 | −0.015 [−0.035, +0.004] |
| E4-B R42–R46 joint heads (write V) | 1–6 /45 | 7–15 | | | | | −0.020 … **+0.031** (R42 [+0.009, +0.057]) |
| E4-B CONST at U-head-selected rows | 7/45 | 15/199 | 4 / 3 | +0.29 | −0.14 | 0.60 | −0.001 |
| CONST at **every** row (no selection) | 45/45 | 199/199 | 23 / 22 | +0.18 | +0.03 | 0.47 | +0.059 [+0.013, +0.122] |
| E4-A separate U readers (25 params) | 21/45 | 29/199 (0.146) | 12 / 9 | +0.38 | +0.07 | 0.48 | +0.041 [+0.007, +0.064] |
| E4-A separate V readers (P1-D) | 8/45 | 15/199 | 6 / 2 | +0.50 | −0.26 | 0.40 | +0.000 |
| E4-B U head, τ₅ | 5/45 | 8/199 | 5 / 0 | +1.28 | −0.56 | 0.75 | +0.008 [−0.007, +0.020] |

- **Actual controller magnitudes.** Run unselected at every row with its own output magnitude, the U head has mean
  ‖a‖ 0.41 on errors and 0.35 on correct rows. The head barely differentiates them. Its mean gain is +0.16 on errors and
  −0.02 on correct rows, with 54% of correct rows damaged.
- **Per audit dialogue** (U head, τ₁₀): 0–2 errors selected per dialogue, and net benefit is negative in 2 of 6
  dialogues.

**E4-B criteria:**
- **Fail:**
  - J1: recall 0.156 < 0.25;
  - J3: NB lower bound < 0;
  - J4: R42's random-read head has a higher NB (0.031), and recall gain over V is only +0.04.
- **Pass:**
  - J2: 6/7 selected errors positive;
  - J5: 5 non-translation positive selections across 5 dialogues.

**→ E4-B FAIL.**

**Context.** On this first-order NB metric, an unselective constant direction scores at least as high as every learned
policy. NB averages away per-row harm. In P1's measured local test, the same kind of basis-aligned perturbations flipped
13.6% of correct tokens.

## 6. Methodological caveat (found after scoring; disclosed, the verdict is not changed)

U was built from readers fitted on **all** construction rows. The protocol then selected the U-probe hyperparameters and
thresholds by grouped CV **inside the same construction rows**, so the validation folds had already shaped U.

- **Inflated construction CV:**
  - U WHEN CV AUROC 0.953, against 0.777 for the full-state reader it came from;
  - U HOW CV cosine 0.863, against 0.288.
- **Consequences:**
  - weak regularization was chosen (HOW α = 0.01; joint head λ = 0.001, while every other head chose λ = 1);
  - thresholds were miscalibrated (E2 τ₁₀ gave 14.6% audit FPR; the U head's τ₁₀ is far above its audit P
    magnitudes).
- **What stays valid:** the audit evaluation itself has no leakage, because U, the readers and the probes never saw
  audit rows.

The E3 and E4-B failures may be partly worsened by this miscalibration. But the full-state reader, which has no such
leakage, also predicts direction only weakly (cos 0.212, positive 60%). So weak HOW information is not an artifact of
the leak. A clean design would cross-fit U inside each fold. That was not run: it would be a new protocol, and the
instruction is not to keep adding complexity.

## 7. Interpretation

1. **WHEN.** The when-to-act information that P1-D found in the full state does fit in four read coordinates. U
   essentially compresses the full-state logistic, with one dominant label axis and the same audit AUROC (0.757 vs
   0.752). This is expected from the construction, not a new mechanism.
2. **HOW.** The how-to-act information, meaning the in-V correction direction, is weak at layer 31 even in the full
   state. A compact read does not create it. The few positive slices are translation-like (n = 5) or tiny (English
   substitutions: n = 4, negative).
3. **Joint selectivity.** A single 20-parameter map cannot turn good WHEN into selective correction. The separate
   two-reader diagnostic (25 parameters) does better on NB but still does not beat an unselective constant direction.
4. **Preservation.** The P1 preservation failure is not resolved. Nothing here measures token flips, only first-order
   margins.

**Limits.**
- 6 previously examined audit dialogues and 45 A rows. Only 11 are English-unit errors, 4 of them English substitutions.
- A and P come from different sampling designs. The position control is satisfied for E2.
- Everything is first-order and teacher-forced.
- The CV-leakage caveat in §6 applies.

## 8. Recommended next action (one)

**Stop the frozen-subspace tiny-controller line at decoder layer 31.** That covers V-read (P1-D) and U-read / V-write
(P1-E). Do not add rank, modules or layers.

The single next step is a project-owner decision on whether to continue adaptation with a conventionally trained
baseline under Whisper Auto beam-5: an ACL-0-style residual editor, LoReFT or LoRA, trained with an ASR loss and
pre-registered, with preservation monitoring. P0 showed that such an editor still beats Auto on genuine-CS utterances
(MER 0.083 vs 0.121; unmatched comparison).
