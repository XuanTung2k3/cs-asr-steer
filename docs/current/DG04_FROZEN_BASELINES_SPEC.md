# DG-04 — Frozen-Steering Baselines & Correction–Damage Frontier: Pre-Run Specification

**Status:** FROZEN pre-run (2026-09-07), **before** any DG-04 ASR outcome was observed. Authority:
v6 proposal (`…Method_First_Proposal_v6.md`) §7–§10, `METHOD_CONTRACT.md` §1–§7, and the frozen
DG-03 selection (`DG03_BASIS_CAUSAL_SPEC.md`). Every scientific rule below is pre-registered; **no
rule is changed after seeing DG-04 outcomes.** DG-04 runs **no** adaptive controller, damage-aware
training, basis refinement, SALSA/LoRA, SEAME/ViMedCSS/multilingual/Qwen, and **no beam-5**.

## 1. Model, site, layer (frozen by DG-02/DG-03)
- **Model:** `openai/whisper-large-v3`, bf16, frozen backbone.
- **Site:** decoder post-cross-attention, pre-FFN — FROZEN (DG-02); hook
  `csasr.lss.sites.DecoderPostCrossAttnInterventionHook` (forced-prefix exclusion, cache position,
  NormPreserve, per-row/token gate, audit recording).
- **Layer:** **L24** — the single layer selected & frozen by DG-03. Not reselected.

## 2. Frozen basis (loaded, never refit)
`results/dg03/basis/steering_basis_v1_L24.json`:
`v_local` = `sha256:2459a635…` (conditioning_residualized_local), `v_cond` = `sha256:319951b5…`
(language_conditioning); dataset_fingerprint `sha256:3981d6a8…`, construction_config_hash
`sha256:1c021fe9…`; scale `s_ℓ = 8.931257754160319`. No fitting/normalization/residualization change
in DG-04; no legacy `v_nat`/`v_prompt`.

## 3. Data roles
- **Evaluation:** `D-dev-select` only — the deterministic `build_population(bundle, cfg,
  "D-dev-select")` candidate population (same 300-utt population DG-03 scored). No new subset.
- **B3 calibration:** `router-calib` only — a fixed deterministic subsample of **300 shortest by
  duration** (sorted), teacher-forced, projections at content (non-prefix) positions. Not
  D-dev-confirm, not D-test.
- Exposure recorded in `DATA_EXPOSURE.md`.

## 4. Systems (canonical names)
Dose applied through the frozen hook as `alpha=ρ`, `scale=s_ℓ`, unit `direction`, NormPreserve on.
- **B0** — frozen baseline, no intervention (common baseline for every gain).
- **B1** — global fixed **local** steering: `d = v_local`, gate = **all eligible non-prefix
  positions** (no learned gate, no oracle mask). `r̃_t = NormPreserve(r_t + ρ·s_ℓ·d)`.
- **B2** — global fixed **local+conditioning** mixture: `d = normalize(V^0 π_fixed)` with the
  pre-registered **`π_fixed = [0.5, 0.5]`** (equal weighting; since `v_local⊥v_cond` unit,
  `d = (v_local+v_cond)/√2`). Same global gate and dose grid as B1. No mixture search.
- **B3** — exact-site **projection-gated frozen** steering (`projection_gated_frozen`, NOT the
  adaptive controller): `s_t = ⟨LN(r_t), v_local⟩` (parameter-free LayerNorm over the feature dim,
  eps 1e-5), `g_t = σ((s_t − τ)/T)`, `d = v_local`, `r̃_t = NormPreserve(r_t + ρ·s_ℓ·g_t·d)`.
  Inference-safe (only `LN(r)·v_local`); no oracle CS location.

## 5. B3 calibration (frozen procedure)
τ, T are frozen from `router-calib` (300 shortest, teacher-forced) `s_t = ⟨LN(r_t), v_local⟩` over
all content positions: **`τ = median(s_t)`, `T = std(s_t)`** (population std; if 0, fall back to
1.0). Deployed gate uses only the inference-safe projection. Written to
`results/dg04/b3_calibration.json` before D-dev-select evaluation. One formulation only; no gate search.

## 6. Dose / strength grid (pre-registered; not tuned to outcomes)
`ρ ∈ {0.5, 1.0, 2.0}` (low / medium=DG-03 diagnostic dose / high), plus `ρ=0` ≡ B0. Same grid for
B1, B2, B3. No values added after inspecting the frontier; no grid/Bayesian optimization.

## 7. Decoding (inherited from DG-03)
Free decoding, greedy, `temperature=0`, `num_beams=1`, `max_new_tokens=200`, `language=zh`,
`task=transcribe`, `condition_on_prev_tokens=False`. No beam-5. No teacher-forced metric selects the
reference operating point.

## 8. Realized-intervention accounting (spec §10)
Per system×dose record nominal `ρ` (and `β_nominal = ρ·s_ℓ`) **and** realized: `n_steered` (edited
positions), `total_energy = Σ‖r̃−r‖`, `mean_energy`. B3 additionally: `n_gate_eligible`
(non-prefix positions), gate-strength mean/median/quantiles, and `n_gate_gt_0.5`. Systems are
compared on realized intervention, not nominal ρ alone.

## 9. Gate coverage (deferred)
The canonical `gate_coverage` denominator is **not** frozen (DG-01 §4 / MC §6). DG-04 does **not**
populate canonical `gate_coverage`; it reports only descriptive gate statistics (§8). result_v1
keeps the reserved null gate_coverage.

## 10. Metrics (DG-01 canonical) & artifacts
Per system×dose emit a complete validated `result_v1`: MER, PIER, embedded-WER (`en_wer`),
matrix-CER (`zh_cer`), gains (`baseline−method`), POI transitions (corrections/corruptions/rates),
three retention populations, reserved gate_coverage, **canonical outside harm**
(`csasr.evaluation.dg03_outside_harm.corpus_outside_harm` over the frozen D-dev-select existing_ctc
candidate population — `n_corrupted_outside`, correctness-flip harm), realized energy, and per-utterance
texts. Provenance: git commit, model/revision, layer, basis hashes, dataset fingerprint, config hash,
Slurm metadata. No legacy `delta_pier`.

## 11. Correction–damage frontier
`results/dg04/frontier.json`: one row per operating point (B0 + every system×dose) with system, ρ,
`β_nominal`, realized edit count + total/mean energy, corrections, corruptions, canonical outside
harm, correction/corruption rates, PIER gain, MER gain, embedded-WER gain, matrix-CER (retention).
Non-dominated points flagged mechanically on (corrections↑, corruptions↓); **no point is discarded**.

## 12. Reference operating-point rule (frozen; = ticket §15)
Primary utility `U = net_corrections = corrections − corruptions` (POI, η=κ=1; DG-03-consistent).
**Eligibility:** (1) `U > 0`; (2) `corrections > corruptions`; (3) canonical outside harm present &
valid; (4) matrix retention not catastrophic, pre-defined as `zh_cer_system ≤ 1.5 × zh_cer_B0`;
(5) numerically valid. **Selection among eligible:** max `U`; tie-break higher PIER gain; then lower
realized `total_energy`. Result → `results/dg04/reference.json` (`DG04_REFERENCE_FROZEN_BASELINE`),
may be B1/B2/B3 (B3 is not forced to win). If no non-zero point is eligible → **NO POSITIVE FROZEN
OPERATING POINT** (flagged as a DG-05 risk; stage still valid if the frontier is reproducible).

## 13. Slurm plan (mig only; ≤2 concurrent)
- **Job A** (`--systems B1,B2`): B0 + B1 grid + B2 grid on D-dev-select (one model load).
- **Job B** (`--systems B3`): B3 calibration on router-calib + B3 grid on D-dev-select (B0 re-decoded
  locally for scoring; deterministic → identical).
- Partition **`mig`** (never `main`); record job id, resource, terminal state, runtime, log, artifact.
- CPU merge (`--build-frontier`) assembles the frontier + reference after both jobs complete.

## 14. Results (filled after terminal completion)
- **Job A:** 50483, `mig`, COMPLETED, 00:09:16 — B0 + B1/B2 full grid.
- **Job B:** 50484, `mig`, COMPLETED, 00:05:03 — router-calib + B3 full grid.
- All 10 operating points are present and validate as `result_v1`; no point was discarded.
- Positive frozen points under §12 are B1 `ρ=0.5` (`U=38`) and B3 `ρ=0.5` (`U=33`).
- **Reference:** `B1 ρ=0.5`, `β_nominal=4.465628877080159`, selected by maximum frozen POI
  utility. It has 135 corrections, 97 corruptions, outside harm 1306, PIER gain +0.01675,
  matrix CER 0.2630, and realized energy 82007.08 total / 4.0743 mean.
- B3 calibration is fixed from `router-calib` (300 shortest, τ=1.2037518, T=1.9582404).
- No non-zero point is discarded despite the high-dose negative results. The candidate-level
  utility remains diagnostic; §12 selection uses the frozen POI `U = corrections − corruptions`.
- Frontier: `results/dg04/frontier.json`; reference: `results/dg04/reference.json`.
- The scientific runner was committed as `33799da`; audit hardening and frontier provenance were
  committed as `a4f14be`. Focused DG-04 tests: **7 passed**.
