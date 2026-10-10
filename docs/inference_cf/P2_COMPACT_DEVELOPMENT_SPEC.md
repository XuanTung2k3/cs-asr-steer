# P2 compact-development specification — v1 (frozen before any P2 recognition outcome)

**Prerequisites.**

- `CACHED_EQUIVALENCE: PASS` (`PRE_P2_CACHED_EQUIVALENCE_SPEC.md`);
- Pre-P2 audit `PASS_TO_P2_DEVELOPMENT`;
- R2 frozen (`g = E·R_B`, job 54758) and P1 passed (job 54781).

No P2 recognition output existed when this spec was committed.

## 1. Frozen method (not searched in P2)

- **Detector:** `g_t = E_t·R_t^B`. LS-B on the 1.0 s max-attention alignment-head window; BC-B
  with partition `sha256:7daa5009…`; R2 structural fallbacks. **DETECTOR FORMULA: FROZEN.**
  Nothing about the detector changes in P2: no LocalSupport, null, window, heads, token-class,
  BaselineConflict or fallback change; no min/weighted/exponent/learned/threshold combination;
  no new counterfactual language.
- **Direction:** `d = norm(hE − hB)` (tiny/nonfinite fallback 1e-4) from isolated unsteered
  cached branches.
- **Edit:** canonical DG-02 hook at the post-cross-attention / pre-FFN site,
  `h' = NormPreserve(h + s_t·d)`, zero-dose bypass, positions < 4 never edited.
- **Dose:** `s_t = α·φ(g_t)`. Reference `φ_id(g) = g`; the **only** alternative in this session
  is `φ_√(g) = √g` (zero-preserving, bounded, strictly increasing, ranking-preserving,
  parameter-free). No clipping or thresholding of `g`; no third mapping; linear rescaling is
  redundant with α. **SCORE-TO-DOSE RELATION: validated here.**
- **Execution:** cached steering path (`experiments/inference_cf_cached.py`), bf16, H100 MIG
  3g.40gb, greedy, `max_new_tokens = 200`, `generate()` suppression.

## 2. Population and roles

- **Primary:** all 300 D-dev-select utterances (20 dialogues) of the frozen R2 panel. This is the
  project's declared selection role; it was already exposed by P0/R2/P1 and is labelled as such.
- **Excluded from P2:** D-dev-confirm, D-test, SEAME, CS-FLEURS, ViMedCSS and Qwen.
- **Calibration:** `φ_√` is parameter-free, so no calibration role is needed.

## 3. Baselines (same job regime, matched 300 IDs)

| System | Definition |
|---|---|
| **B0** | Cached greedy forced-ZH (`cached_greedy`, cB). All Δ are computed against B0. |
| **B1** | Cached greedy forced-EN, no steering. |
| **B0_AUTO** | Ordinary Whisper `generate()` with `language=None`, batch 1. The R2 B0_AUTO parquet is also reported. |

**Rule:** beating B0 alone does not mean beating ordinary Whisper. The P2 report must show B0_AUTO
beside every selected result. No deployment-superiority claim is made in P2.

## 4. Metrics (Δ = baseline − method; positive = improvement)

Every system and configuration reports:

- **Recognition:** PIER (canonical `pier`) with numerator/denominator and per-category counts;
  MER, EN-WER and ZH-CER (`corpus_mer`) with counts.
- **Per-POI flips:** corrections (B0-wrong → method-correct) and corruptions (B0-correct →
  method-wrong).
- **Retention and harm:** matrix ZH retention and embedded EN retention (`retention.py`); matrix
  damage count (B0-correct ZH units lost; outside-target harm).
- **Decode health:** failures and 200-token caps.
- **Gate and edit coverage:** gate-positive steps, edited steps, direction failures, fallback
  counts.
- **Edit energy:** nominal dose; realized post-NormPreserve edit energy `Σ‖h' − h‖²`; mean
  relative edit.
- **Cost:** runtime, VRAM, forward and LID counts.

Per-step records distinguish: gate zero, structural fallback, direction invalid, edit applied
without a token change, and token changed. Corrected vs corrupted is attributed per POI on the
CPU side. References stay evaluator-only.

## 5. Validity constraints and selection (frozen; derived from pre-existing project conventions)

A configuration is **valid** only if **all** of the following hold against B0 on the 300 IDs:

| # | Constraint | Value and source |
|---|---|---|
| V1 | Decode health | 0 failures, and 200-token-capped decodes ≤ B0's count + 3 (1% of 300). Repetition/hallucination guard. |
| V2 | Mandarin damage | ZH-CER(method) − ZH-CER(B0) ≤ **0.005** absolute. `configs/base.yaml: pilot_max_zh_cer_degradation_abs = 0.005`, the project's pre-registered development tolerance. |
| V3 | Overall damage | MER(method) − MER(B0) ≤ **0.005** absolute. `configs/base.yaml: pilot_max_mer_degradation_abs = 0.005`. |
| V4 | Mandarin retention | Baseline-correct CS-Mandarin retention (`matrix_zh_retention`) ≥ **0.99**. Derived from V2 on this panel: 0.005 × ~14.5k ZH reference units ≈ 72 units ≈ 0.62% of the ~11.6k B0-correct ZH units, rounded to the conservative two-decimal bound. |
| V5 | Efficacy | ΔPIER = PIER(B0) − PIER(method) > 0. |

**Selection.** Among valid configurations, maximize ΔPIER. PIER is a ratio of integer counts, so
ties are exact count equalities. Deterministic tie-break:

1. higher ΔPIER;
2. lower ZH-CER increase;
3. lower realized edit energy;
4. simpler dose map (`id` before `√`);
5. lower α.

Correction/corruption ratio, EN-WER and dialogue-cluster intervals are co-reported but are not
selection terms. **None of these numbers change after P2 outcomes.**

## 6. P2-A — efficacy, layer and strength screen

- **Grid:** `g = E·R_B`, `φ_id`, ℓ ∈ {16, 24}, α ∈ {0.5, 1.0, 2.0}. Six configurations. α = 0 is
  identity (validated by CE1) and not selectable.
- **Outcome:** `P2_A_VALID_CONFIG_EXISTS` (select (ℓ*, α*) by §5) or `P2_A_NO_VALID_CONFIG`.
- **No extensions:** no extra layers or α values, and no change to g, localizer or direction.
- **If `P2_A_NO_VALID_CONFIG`:** run only the minimal diagnostics below, then freeze the negative
  result and stop. P2-C is not run.
  - **Configuration:** the P2-A configuration with the largest ΔPIER (ties by §5), or ℓ = 24,
    α = 1.0 if none has ΔPIER > 0.
  - **Diagnostic 1:** `g = 1` (does selectivity fail?).
  - **Diagnostic 2:** `−d` with `g = E·R_B` (does the direction sign matter?).

## 7. P2-B — mechanism controls (only if P2-A is valid)

All at (ℓ*, α*), `φ_id`, not selectable:

| Control | Question |
|---|---|
| `g = 1` (same α*) | Does selective placement add value over broad steering? |
| `g = 1` at **α_c** (energy-matched, see below) | Same question at matched realized energy. |
| `g = E` | Does baseline conflict protect already-correct English? |
| `g = R_B` | Does acoustic support protect legitimate Mandarin? |
| `g = E·R_B` | Reused from P2-A. |
| `g_cf = E·[R_B − R_Ecf]_+` | Does the R2-rejected restriction add intervention-level value? |
| `g = E·R_B` with `−d` | Does the direction sign matter causally? |

**Energy-matched constant gate (frozen formula, development only).**
`α_c = sqrt(Σ_edits ‖h' − h‖² / N_elig)`, where:

- the sum runs over all edited steps of the selected P2-A configuration;
- `N_elig` = the number of its steps with no structural fallback and query ≥ 4 (the steps a
  constant gate would edit).

With d unit-norm, a constant gate at α_c realizes about the same total energy. The realized
energy of the α_c run is reported to confirm this.

Both same-recipe (α*) and realized-energy comparisons are reported. No gate formula beyond these
is added. Conclusions are restricted to these predeclared questions, with no optimality claim.

## 8. P2-C — one dose-map comparison (only if P2-A is valid)

- **Grid:** `φ_√` at ℓ*, α ∈ {0.5, 1.0, 2.0}. Same detector and the same α budget.
- **Final selection:** by §5 over **all valid configurations of P2-A (both layers) and P2-C**. The
  tie rule prefers `φ_id`.
- **Question:** does redistributing strength under the same detector ranking improve the
  correction–damage trade-off? It is not a search for an optimal nonlinearity.

## 9. Uncertainty and audit

- **Intervals:** paired dialogue-cluster bootstrap (2,000 resamples, seed 240924) for ΔPIER,
  ΔMER, ΔZH-CER and ΔEN-WER of the selected configuration vs B0 and vs B0_AUTO. Descriptive only;
  selection uses point estimates by §5.
- **Audit:** an independent audit recomputes headline metrics from the saved hypotheses.

**Outcomes:**

- `P2_VALID_CONFIGURATION_SELECTED` → `READY_FOR_P3_LOCKED_EVALUATION`;
- `P2_NO_VALID_CONFIGURATION`, a valid negative result;
- `P2_BLOCKED_INVALID_EXPERIMENT`.

## 10. Resources

- **Jobs:** `sbatch` on MIG, at most two concurrent GPU jobs. P2-A is split by layer into two
  parallel jobs, each writing matched baselines once.
- **LID reuse:** LID posteriors are reused only for identical (utterance, crop) inputs.
- **Reruns:** only for concrete infrastructure or implementation invalidity, into versioned
  directories.

## 11. v1.1 amendment — matched baseline and experiment validity (after P2-A attempt 1)

This amendment follows P2-A attempt 1 (jobs 54803/54804). That attempt is recorded as
`P2_BLOCKED_INVALID_EXPERIMENT`; see `P2_A_ATTEMPT1_INVALID_BASELINE.md`. The fix is to
execution, not to any rule.

- **Δ reference (replaces §3 B0 as the reference):** for each layer ℓ in a run, `B0M_Lℓ` is the
  α = 0 run of the identical `cached_decode` path (gate ER, dose id) in the same job. By CE1 it is
  bitwise its own B branch.
  - Every Δ, V1–V5 check, flip, retention value, selection and "vs B0" interval uses the
    `B0M_Lℓ` of the configuration's own run and layer.
- **Diagnostics:** B0 (`cached_greedy`), B1 and B0_AUTO remain reported baselines. B0_AUTO
  intervals are still reported.
- **Experiment validity (frozen before the rerun; all must hold, otherwise
  `P2_BLOCKED_INVALID_EXPERIMENT`):**
  1. 0 failed rows.
  2. Every `B0M_Lℓ` step is bitwise equal to its B branch (`zero_dose_bitwise`) and has
     `lineage_ok`.
  3. **Divergence attribution:** for every configuration and utterance where the tokens differ
     from `B0M_Lℓ`, let k be the first differing token index. Some step t ≤ k must have an
     applied edit. Step t emits token t, and an edit at t can affect only tokens ≥ t.
  4. All configurations are complete.
- **Also reported:** whether `B0M_Lℓ` hypotheses agree across layers and jobs.
- **Unchanged:** population, detector, direction, site, dose maps, layers, α grid, V1–V5
  thresholds, selection and tie rule, P2-B/P2-C/A_diag definitions, bootstrap.
- **Reruns:** P2-A reruns into `p2_A_r1_L16` and `p2_A_r1_L24`.
