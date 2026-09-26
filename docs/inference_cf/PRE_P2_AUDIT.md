# Pre-P2 independent audit (cached equivalence + P2 contract freeze)

**CACHED_EQUIVALENCE: PASS** (spec v1.1, job 54802).
**Verdict: `PASS_TO_P2_DEVELOPMENT`.**

## A. Cached equivalence

**Runs.**

| Attempt | Job | Commit | Verdict |
|---|---|---|---|
| 1 (v1) | 54801 | `c7e2369` | `BLOCK`, one CE6 clause, preserved |
| 2 (v1.1) | 54802 | `6e40916` | `PASS` |

Both runs produced identical CE1–CE5 statistics (deterministic).

**Evidence (v1.1).**

| # | Evidence |
|---|---|
| CE1 | Bitwise S = B logits at every α = 0 step. α = 0 cached tokens equal plain cached greedy on 20/20 decodes and equal the R2 `generate()` B0 on **10/10** utterances (full replay managed 9/10), so the replay-vs-`generate()` confound is removed. |
| CE2 | `lineage_ok` on every decode; consecutive query indices; no skipped or reset positions. |
| CE3 | Three distinct cache objects per decode. Unit test: S edits never alter B's cache. |
| CE4 | 736 compared steps. Fallback identical 100%; window identical 97.3%; R_B and g within 0.02 on 99.5%; E within 0.02 on 100% of same-window steps. |
| CE5 | Direction status identical 100%; cosine ≥ 0.99 on 100% (minimum 0.9925); norm within 2% on 100%. |
| CE6 | 40 cached edits; `edit_applied` identical 100%. Sign, own-replica (both paths) and NormPreserve all pass. Specified edits agree within 5% on 38/38 common edits. **Zero non-near-tie argmax disagreements**; the only disagreement is the known U0029 t = 6 tie (margins 0.0625 / 0.0). Maximum \|Δlogit\| 0.31. |
| CE7 | Hook absence asserted around B/E at every step; none leaked. |
| CE8 | 80.6 s job: cached 40.9 s and probe 39.2 s for 104 s of audio × 4 decodes; peak 4.19 GB. |

**Adversarial review of the v1.1 correction (post-outcome, disclosed).**

The v1 clause compared the two paths' realized bf16 edit norms. The v1.1 clause compares their
specified edits.

- **Is it a loosening?** It removes a cross-path check on realized outputs. Both paths call the
  identical `apply_steering`, and each path's realized edit is verified equal to its own
  same-precision replica. A realized cross-path difference can therefore only come from
  (i) different inputs — h_B, g, d, α, site and position, all independently checked by
  CE4/CE5/CE6 — or (ii) bf16 rounding inside the shared function. Rounding (ii) says nothing
  about cache semantics.
- **Is the rounding explanation real?** A CPU simulation of the shared function with identical
  math reproduces >5% cross-realization differences in 39% of draws at g ≈ 0.004 and 4% at
  g ≈ 0.1. That matches the two observed exceedances, both small edits.
- **What recognition depends on:** token-level agreement. It held under v1 already: 0
  non-near-tie disagreements over 736 steps.
- **Judgment: accepted.** The correction changes the measurand to the property the spec intends
  (equivalence of the intervention); the tolerance and share are unchanged. The v1 `BLOCK`
  remains on record.

## B. Documents and provenance

- **Canonical plan and STATUS** (`7ea8b19`): reconciled. Forward-looking text names
  `g = E·R_B` as the main method; `g_cf` is the R2-rejected ablation (new C3b row). Figure M1
  describes E + R_B → g, and hE − hB → d. R2/P1 checklist items are complete; the stage table
  has Pre-P2 and P2 rows; the next-task text is current. Historical R2 text is labelled as
  history.
- **R2 runner:** the post-R2 working-tree change was classified C (useful, unapplied). It is
  preserved as a patch and the committed bytes are restored (`fa10d9f`). Frozen R2/P1 manifests
  verify at HEAD; the living plan verifies at each manifest's own commit.
- **History:** R2 and P1 are preserved unchanged (runs, attempt-1 records, reports, audits).

## C. P2 contract (`P2_COMPACT_DEVELOPMENT_SPEC.md` + config), frozen before any P2 outcome

| Item | Frozen value | Verified |
|---|---|---|
| Detector | `g = E·R_B` (formula frozen; no detector search) | ✅ config and runner only use `ER` for selectable configs |
| Dose maps | `φ(g) = g`; single alternative `√g` (P2-C only) | ✅ `DOSE_MAPS = ("id", "sqrt")`; no other map exists |
| Layers | {16, 24} | ✅ prepare refuses other layers |
| α grid | {0.5, 1.0, 2.0} (α = 0 only as identity, via CE1) | ✅ |
| Population | 300 D-dev-select utterances, 20 dialogues (R2 panel, hash-checked) | ✅ no confirm, test or transfer |
| Constraints | caps ≤ B0 + 3; ΔZH-CER ≤ +0.005; ΔMER ≤ +0.005 (`configs/base.yaml` pilot tolerances); matrix ZH retention ≥ 0.99; ΔPIER > 0 | ✅ implemented with a float-representation guard only |
| Selection / tie rule | max ΔPIER; then lower ZH-CER increase, lower realized energy, `id` < `√`, lower α | ✅ unit-tested |
| P2-B | g = 1 (α*), g = 1 energy-matched α_c (frozen formula), g = E, g = R_B, g_cf, −d; not selectable | ✅ |
| P2-C | `√g` × {0.5, 1, 2} at ℓ*, only if P2-A valid | ✅ prepare refuses otherwise |
| Negative P2-A | only the g = 1 and −d diagnostics at the predeclared configuration | ✅ `A_diag` stage |
| Baselines | B0 cached forced-ZH (Δ reference), B1 forced-EN, B0_AUTO (`generate`, language=None) | ✅ same job regime |
| Leakage | references read only by the CPU evaluator; the runner reads the reference-free panel | ✅ |
| Outcomes inspected | none (no P2 run exists) | ✅ |

**Tests:** 73 inference-CF tests pass (P0/P0-R1/R2/P1/cached/CE/P2), run with
`OMP_NUM_THREADS=4`.

**Residual risks (measured, not blockers):**

- The detector's weak localization on EN-confusion (R2) may limit efficacy.
- B0_AUTO is the stronger unsteered baseline.
- D-dev-select is heavily exposed, so P2 is development evidence only.

**PASS_TO_P2_DEVELOPMENT**
