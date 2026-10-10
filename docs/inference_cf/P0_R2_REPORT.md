# P0-R2 feasibility report — frozen repairability run

**Final R2 verdict: `R2_CF_FEASIBLE_NOT_PREFERRED`.**
**Selected gate for P1: `g_old = E·R_B`.** `g_cf` is not selected.
**Post-run audit:** `POST_R2_AUDIT: PASS` (`P0_R2_POST_RUN_AUDIT.md`).

## Run identity

| Item | Value |
|---|---|
| Spec / audit | `P0_R2_REPAIRABILITY_SPEC.md` v1.1; pre-run `P0_R2_REPAIRABILITY_AUDIT.md` (`PASS_TO_R2_RUN`) |
| Git commit (manifest) | `fe960e47731abbe8b3dac2f3714292084ccc9f40` (clean; equal to `origin/cs-asr-steer-inf` at freeze) |
| Manifest hash | `sha256:e8869f0c8be31783187f3976b905def858751978a0631118e094ead901af1c25` |
| Config hash | `sha256:8bcc7849e7c7358cbf609e6f398ed3caf6afd01155898ad29e7043f31700f6d6` (unchanged after the run) |
| Tokenizer partition | `sha256:7daa5009…` (1667 / 37858 / 12341) |
| Population | DG-04 B0 D-dev-select, 300 utterances, 20 dialogues; panel `sha256:f8fe606d…` |
| Slurm | job **54758**, `COMPLETED` 0:0, wall 24:04, program 1,428 s, H100 MIG 3g.40gb, peak VRAM 3.40 GiB (3,645,013,504 B), 0.40 GPU-h |
| Coverage | 300/300 shards `ok`; 13,859 dense positions |
| Fallbacks | `mid_character` 1,115 (8.0%); no localizer, LocalSupport, baseline or Ecf failure |
| EOS / cap | 298 EOS slots; 2 utterances capped at 200 tokens |
| Replay consistency | argmax agreement with the actual decode 99.8% |
| Invariants | same-prefix validation on every row; 0 invariant violations |
| Cost | LID evaluations: 11,855 real + 8,750 control, plus 4,883 exact-window cache hits |

Evaluator: the checked-in `experiments/inference_cf_p0_r2_evaluate.py`, unmodified. It verified all
23 source hashes. Bootstrap: 2,000 dialogue resamples, seed 240924. The headline metrics were
reproduced by independent code (post-run audit).

## Frozen checks

| Check | Result | Evidence |
|---|---|---|
| ALIGN | **PASS** | 0.962 of non-boundary English POIs mapped (floor 0.80). Contrast-eligible share 0.793 is descriptive. Unalignable: 453 stratum collisions, 98 beyond heard audio, 29 boundary, 7 other. |
| LOCALIZER | **PASS** | 0.890 of 13,218 timed units within 0.5 s (floor 0.60). Median absolute error: max-window 0.222 s, point-argmax 0.136 s. Midpoint coverage: max-window 0.890, point 0.868. |
| R2-LS | **PASS** | AUROC(E; English vs ZH-correct) **0.820** [0.783, 0.849] (1,439 vs 7,925 positions, 20/20 dialogues). Finite coverage 1.00. Mismatched-audio AUROC 0.503; paired Δ(real−mismatch) CI [0.269, 0.360] > 0. Null identity holds. |
| R2-BC / Ecf | **PASS** | All masses finite; `0≤R≤Q`, `0≤D≤R_B`, `g_cf≤g_old` with 0 violations; exact same-prefix identity on every row. |
| R2-GATE, EN-confusion vs ZH-correct | **PASS** | `g_cf` AUROC **0.710** [0.664, 0.757], AUPRC 0.133 (316 vs 8,901; 18/20 dialogues) |
| R2-GATE, EN-confusion vs EN-correct | **PASS** | `g_cf` AUROC **0.726** [0.679, 0.774], AUPRC 0.552 (316 vs 1,168; 18/20 dialogues) |
| CF preference | **FAIL** | Paired ΔAUROC(`g_cf`−`g_old`) = **−0.110** [−0.145, −0.065] vs ZH and **−0.110** [−0.144, −0.067] vs EN-correct |

`g_old` on the same rows: AUROC **0.820** [0.768, 0.856] vs ZH (AUPRC 0.330) and **0.836**
[0.782, 0.873] vs EN-correct (AUPRC 0.741). It meets the same frozen bars.

## Localizer by stratum (evaluator timing only)

| Stratum | n | Max-window within 0.5 s | Point within 0.5 s | Max median abs error |
|---|---:|---:|---:|---:|
| ZH-correct | 11,443 | 0.919 | 0.898 | 0.204 s |
| EN-correct | 1,168 | 0.844 | 0.806 | 0.306 s |
| EN-confusion | 349 | **0.464** | 0.461 | 0.576 s |
| EN-deletion-slot | 177 | **0.181** | 0.198 | 6.36 s |

The pooled pass is dominated by Mandarin. On the primary target stratum fewer than half the
windows are centred within 0.5 s, and deletion slots are essentially not localized. The
max-window was frozen before the run and is retained; the point diagnostic is no better on the
target strata. No localizer switch is made.

## Components (medians by stratum)

| Stratum | positions / dialogues | fallback | E | R_B | R_Ecf | D | R_X (ru) | D_X | g_old | g_cf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EN-confusion | 316 / 18 | 0.142 | 0.787 | 0.872 | 0.862 | **0.002** | 0.000 | 0.734 | 0.415 | 0.000 |
| EN-correct | 1,168 / 20 | 0.000 | 0.798 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| EN-deletion-slot | 48 / 16 | 0.104 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| EN-same-language-sub | 81 / 18 | 0.000 | 0.486 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ZH-correct | 8,901 / 20 | 0.110 | 0.000 | 0.997 | 0.996 | 0.000 | 0.948 | 0.016 | 0.000 | 0.000 |

Component AUROCs:

| Component | vs ZH-correct | vs EN-correct |
|---|---:|---:|
| E | 0.876 | 0.536 |
| R_B | 0.206 | 0.928 |
| R_Ecf | 0.224 | 0.902 |
| D | 0.571 | 0.784 |
| D_X | 0.784 | 0.928 |
| g_X = E·D_X | 0.823 | 0.836 |
| shuffled-E → E·D | 0.497 | 0.551 |
| shuffled-D → E·D | 0.675 | 0.488 |

## Interpretation (strictly within the claim boundary)

1. **Local acoustics carry excess English evidence.**
   - E separates English from Mandarin positions (0.82 overall; 0.88 at confusion vs correct
     Mandarin, where both strata emit Han).
   - The signal collapses to chance on mismatched local audio (0.50).
   - This is the strongest positive result.
2. **Baseline matrix-script conflict is measurable.**
   - Its separation of confusion from correct English (R_B 0.93) is largely the emitted script
     (Latin ⇒ R_B≈0), as predicted.
   - That contrast therefore mostly confirms the gate stays quiet on correct English.
3. **The same-prefix English counterfactual does not reduce the conflict where it matters.**
   - At confusion positions R_Ecf≈R_B (median D 0.002). Given a Chinese prefix, flipping
     `<|zh|>`→`<|en|>` rarely moves Han mass (consistent with the P0/P0-R1 collisions).
   - The problem is prompt insensitivity rather than prompt susceptibility. Multiplying by D
     discards most of `g_old`'s ranking signal (ΔAUROC −0.11, CI excludes 0).
4. **The forced-Russian diagnostic behaves unexpectedly.**
   - It removes Han mass at confusion positions (R_X 0.00) but not at Mandarin ones (0.95), so
     D_X and g_X rank far better than D and g_cf.
   - Diagnostic only, reported and **not adopted**. A different counterfactual condition would be
     a new, separately versioned experiment, never a post-hoc swap.
5. **What the selected gate targets.** It detects *English speech rendered in the matrix script at
   the current decode step* (confusion). It does not reach deletions (E≈0 at deletion slots) or
   same-language substitutions.
6. **Recognition baselines, recomputed on the fresh R2 decode.**
   - **B0 (forced ZH):** PIER 0.4735 (1,074 errors), MER 0.2647, EN-WER 0.477, ZH-CER 0.228;
     339 wrong-language substitutions, 32 transliterations, 538 deletions, 129 same-language.
   - **B0_AUTO (matched 300):** PIER 0.3898 (884), MER 0.2597, EN-WER 0.395, ZH-CER 0.235.
   - B0_AUTO is stronger on PIER and EN-WER. No later result that beats B0 alone may be called
     better than ordinary Whisper.

## Frozen R2 outcome

- **Verdict:** `R2_CF_FEASIBLE_NOT_PREFERRED` under the rule frozen before the run (spec §4).
- **P1 gate:** the reconciled `g_old = E_t·R_t^B` (LS-B × BC-B, max-window localizer, fallbacks
  as specified; the Ecf branch plays no role).
- **Unchanged after the outcome:** formulas, thresholds, window, partition, strata, seed and
  replicate count were not changed after seeing results.

**Stored artifacts.** `results/inference_cf/p0_r2/` holds the manifest, inference panel (audio
identity only), runtime, summary, row shards and Slurm logs. `evaluation_panel.json` (reference
transcripts) and `evaluation_units.json` (reference surfaces) are dataset-derived. They are
kept out of Git and regenerated by the evaluator from the hashed role parquet.
