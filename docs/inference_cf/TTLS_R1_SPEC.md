# TTLS-R1 — unified test-time learned steering pilot (frozen specification)

**Status:** `FROZEN_BEFORE_RUN`. The machine-readable authority is `configs/inference_cf/ttls_r1.json`. Any discrepancy
between the implementation and that config blocks the run.

**Scope.** This is an exploratory research ticket on branch `feature/ttls-r1`, started from `c69038ad`. It is not a
change to the core-v6 method (`METHOD_CONTRACT.md`). Every historical method, result and verdict is unchanged.
Only new files are added; no existing module is modified.

## 1. Question

Can a temporary, localized activation variable, optimized per utterance at the frozen DG-02 site, correct embedded-English
lexical errors while preserving correctly recognized Mandarin better than updating shared decoder LayerNorm parameters
(historical A2), under the same test-time objective?

The two motivations, both tested and neither assumed:
1. Direct representation-level constraints may give a better correction–preservation trade-off.
2. Activation-level interventions are easier to analyse causally.

## 2. Population, data and model

| Item | Value |
|---|---|
| Panel | The fixed-100 D-dev-select panel `docs/inference_cf/P2_SEL_MINI_PANEL.json` (sha256 `266ea7ea…`); 20 dialogues × 5 |
| Teacher plan | The sealed TTA1 plan `results/inference_cf/p2tta_funnel/tta1/plan_sealed.json`: y_B (P2-SEQ S0 forced-ZH), y_A (historical AUTO), valid masks, audio identities |
| Subgroup | The A3 24-panel (`P2_TTA_A3_PANEL.json`), a subset of the 100. It is reported separately and is not a population estimate |
| Exposure | Already-exposed FULL300 subset (test-checked against `exposure_registry.exposed_ids()`). No new exposure and no addendum |
| Never touched | D-dev-confirm, D-test, router-calib, P3, transfer, and any unexposed utterance |
| Model | Whisper-large-v3, BF16, pinned files; seed 240924; greedy; at most 200 new tokens |

**Designer knowledge.** The designer has read the historical TTA1, A2-MECH0, A3, A4 and DIR-SPRINT0 reports on this
exposed panel. It is exposed development data, not confirmation.

## 3. Design-time finding that changes the proposed localization (reference-free)

Before writing the method, I inspected only inference outputs (B0 and AUTO token sequences, no reference):
- Forced-ZH and AUTO differ on **12/100** rows.
- On **12/12**, the first differing token is at **content step 0**.
- The cause is either a leading-space artifact of the English prompt (`The` vs ` The`, `I` vs ` I`, `我想` vs ` 我想`) or
  a whole-utterance translation (for example U0021, U0023_664, U0102 and U2004 → Malay).
- Content step 0 is a DG-02 forced-prefix position (`num_forced_prefix = 4`) and is **never edited**.

The proposed "first AUTO/forced disagreement region" is therefore empty on 88 rows and uneditable on the other 12. It
cannot serve as the TTLS location mask. The deviation is recorded in the config (`masks.deviation`):
- **CE arms** use the all-editable-steps mask.
- **AC arms** are localized by the frozen acoustic-lexical candidate rule (section 5).

## 4. Adaptation variables (both reset per utterance)

Both variables share these settings:
- 2 AdamW steps and 3 loss evaluations;
- bf16 forward;
- free forced-ZH greedy decode after adaptation.

**LN (historical A2 actuator, reused unchanged from `episodic_tta`).**
- Trainables: 194 decoder LayerNorm affine tensors, 248,320 scalars.
- Fresh fp32 masters used through `functional_call`.
- AdamW: lr 1e-3, wd 0, betas (0.9, 0.999), eps 1e-8.
- Exact `LNGuard` restore with bitwise verification.

**TTLS.**
- **Variable:** one temporary fp32 vector `z ∈ R^1280` (1,280 scalars).
- **Edit:** `r~_t = NormPreserve(r_t + g_t z)` at the decoder layer-16 post-cross-attention / pre-FFN residual.
- **Hook:** the canonical `DecoderPostCrossAttnInterventionHook` with `apply_steering` (alpha = scale = 1, no depth
  rescale). It returns `u_source + (r~ − r)`. Content step 0 is never edited.
- **Optimizer:** z starts at 0; AdamW with lr = E\*/(2√1280) = 0.0157374, wd 0, same betas and eps.
- **Budget:** after every step, `z ← z·min(1, E*/‖z‖)` with E\* = 1.1260757575454359. This is the frozen DG-02 L16
  realized chord, about 15% of ‖r‖. Under NormPreserve the realized per-position chord is therefore at most about E\*.
- **Reset:** a new zero z and a new optimizer per utterance and per arm. The hook exists only inside one forward or one
  decode call, so nothing carries over between utterances.

The update and memory comparison is reported as measured. Equal optimizer steps are **not** equal compute.

## 5. Objectives

All objectives use the forced-ZH prompt cB = `<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>`.

### CE — matched historical A2 supervision

`L_CE = mean_{valid t} −log p_θ(y_A[t] | cB, y_A[:t])` on the historical AUTO pseudo-transcript path. This is exactly
`episodic_tta.position_terms` with `objective_loss('A2')`.

| Variable | Mask | Arm |
|---|---|---|
| LN | (no mask) | B2, the historical A2, reused |
| TTLS | ALL editable steps | T1 |

### AC — acoustic-conditioned lexical supervision

The candidate is computed once per utterance before any adaptation, from frozen θ0 teacher-forced passes on the B0 path
y_B. There are four branches, all sharing the same prefix:
- clean and null audio, where null = 30 s of zeros through the same extractor and encoder (the DIR-SPRINT0 D3 null);
- crossed with the prompts cB and cE = `<|en|>` transcribe.

For each content step `1 ≤ t < |y_B|`:

| Stage | Rule |
|---|---|
| Structural | UTF-8-complete prefix; b_t = y_B[t] is not EOS and not an embedded-Latin token (`core_r2.tokenizer_partition`) |
| Proposal | e_t = argmax of the clean-cE log-probability, lowest ID on ties. A language-disagreement step is one where e_t is embedded-Latin. M = all such steps |
| Evidence | A^c(v) = l^c_clean(v) − max(l^c_null(v), log 1e-12); E^c = A^c(e_t) − A^c(b_t) for c ∈ {cB, cE} |
| Accept | E^cB ≥ τ **and** E^cE ≥ τ, with τ = log 10 |

Requiring the evidence under both prompts makes it prompt-robust: the language prior of neither prompt can create
support on its own. A Latin-script proposal is never accepted for its script alone.

**Target.** t\* is the first accepted step and c\* = e_t\*. If no step is accepted, the arm abstains: an exact no-op whose
output is B0.

**Loss.** `L_AC = −log p_θ(c* | cB, y_B[:t*])`.

**Mask (TTLS).** `{t*}`. Training and free decoding are consistent: the decoded prefix up to t\* equals y_B[:t\*]
because nothing before t\* is edited.

Neither the adapted model nor any reference redefines the candidate. Every variable uses the same candidate and the same
evidence.

### +P — preservation

`+ λ · mean_{t∈S} KL(p_θ0,t ‖ p_θ,t)` on the B0 path under cB, over the allowed vocabulary, with λ = 1.
- **Stable set S:** valid content steps of y_B with p_θ0(b_t) ≥ 0.5 that are not in M.
- **Same definition for both variables.** For that reason the A2-CE+P arm (T2A) is added, so that P is never a
  TTLS-only advantage.
- **Known limit.** KL to θ0 has zero gradient at θ0, so with 2 steps P can act only on the second update. This is
  recorded rather than tuned.

## 6. Experiment matrix

| Arm | Variable | Objective | Mask | Source |
|---|---|---|---|---|
| B0 | — | forced-ZH | — | reused S0; reproduced in-job |
| B1 | — | AUTO | — | reused; replayed in-job with the historical `detect_language` prompt |
| B2 | LN | CE | — | reused TTA1/TTA0-R A2; decoded in-job from the archived fp32 masters |
| T1 | TTLS | CE | ALL | new |
| T2 | TTLS | CE + P | ALL | new |
| T2A | LN | CE + P | — | new (matched comparator for T2) |
| T3 | LN | AC | — | new |
| T4 | TTLS | AC | {t\*} | new |
| T5 | LN | AC + P | — | new |
| T6 | TTLS | AC + P | {t\*} | new |
| CD | — | DIR-SPRINT0 D3 rule at every step ≥ 1 (UTF-8-complete prefix, non-EOS greedy) | — | no-update comparator |
| ACSUB | — | c\* substituted at t\*, then ordinary greedy | — | no-update comparator |

**Matched pairs.** CE: B2–T1. CE+P: T2A–T2. AC: T3–T4. AC+P: T5–T6.

Token-level DIR-SPRINT0 counts are never mixed with these sequence metrics.

## 7. Execution

**Before the job (CPU):**
- `prepare` writes the plan with hashes, UTF-8 prefix flags, the archived A2 masters and A3 membership.
- 14 focused tests cover: edit masking and causality, forced-prefix no-op, float64 finite-difference gradient, episode
  reset and budget, LN actuator reuse, the frozen candidate and stable rules, an end-to-end `process_row` on a tiny
  real-class Whisper, the reference firewall, config/module agreement, exposure, and the evaluator accounting and label
  rule.
- Sources are committed; then `manifest` records the commit, sources, plan, environment, model files and seed.

**One Slurm job** (MIG 3g.40gb, ≤ 3 h). Within it:
- On the first row, the job checks that a TTLS hook with an empty mask decodes exactly to B0, and that TTLS and LN
  gradient flow is finite and nonzero. If either check fails, it stops immediately; if both pass, it continues.
- Reuse identities B0, B1 and B2 are checked on every row.
- Each row is written atomically, so the run is resumable.

**After the job:**
- `seal` hashes every output; the seal is committed and pushed **before** any reference is opened.
- `inference_cf_ttls_r1_evaluate.py` then opens references and computes everything.

The candidate rule, losses, thresholds and label rule are frozen here. No objective, threshold or arm changes after
outputs are seen. There are no smoke jobs and no sweeps.

## 8. Evaluation (free decoding)

**Metrics.** Canonical `csasr.evaluation`, computed against B0 and against B1 (AUTO), for:
- PIER, MER, EN-WER and ZH-CER;
- POI corrections and corruptions, with corrections split into:
  - **genuine lexical substitutions** (B0 POI category: wrong-language, transliteration, same-language or other);
  - **deletion-type** (deletion, boundary, insertion-near-POI);
  - **EOS-recovery rows** (B0 stopped with EOS and is a strict prefix of the system output);
- matrix-ZH retention and newly introduced Mandarin errors (B0-correct ZH unit → incorrect), plus ZH repairs;
- outside-POI harm, prefix preservation before the editable region, and active-edit harm rate;
- premature-EOS rows, new severe truncations and caps.

**Descriptives.**
- Candidate coverage, abstention, acceptance, support distribution and candidate correctness (post-seal).
- Loss trajectories, gradient norms, z norms and projection.
- L16 site displacement and KL on all and on stable positions.
- Adapt and decode time, backward counts, peak memory and trainable scalars.

**Uncertainty.** Dialogue-cluster bootstrap with 2,000 draws, seed 240924; descriptive only.

**Subgroups:** D (12), A (88), the A3 24-panel, and the AC-accepted rows.

## 9. Exploratory label (frozen)

The labels are checked in this order: `TTLS_R1_INVALID` → `PROMISING` → `MIXED` → `NOT_SUPPORTED`.

| Label | Condition |
|---|---|
| **INVALID** | Any integrity failure: a missing row, B0/B1/B2 reuse mismatch, a failed in-job validation, a reset failure or a non-LN change |
| **PROMISING** | Some TTLS arm X ∈ {T1, T2, T4, T6} meets all three conditions below |
| **MIXED** | Not PROMISING, and some TTLS arm has ≥ 3 genuine-substitution corrections with net POI corrections > 0 |
| **NOT_SUPPORTED** | Otherwise |

The three PROMISING conditions, each judged against B0 or against X's matched A2 arm Y:
- **(a) Corrections:** ≥ 8 genuine-substitution POI corrections against B0, spread over ≥ 4 dialogues.
- **(b) Safety against B0** (all TTA1 bounds):
  - ΔMER ≤ 0.01 and ΔZH-CER ≤ 0.015;
  - matrix-ZH retention ≥ 0.98;
  - outside harm ≤ 0.03 and POI corruption ≤ 0.05;
  - no new severe truncation and at most +1 cap.
- **(c) Dominance over Y:** at least as many genuine-substitution corrections as Y, and no more newly introduced Mandarin
  errors than Y.

The AUTO (B1) comparison is reported for every arm but does not enter the label.

## 10. Limits stated in advance

- **Small opportunity.** With 12 AUTO-disagreement rows, CE supervision carries a lexical signal on few rows.
- **First region only.** AC is first-candidate-only, so at most one candidate correction is targeted per utterance.
- **Weak preservation.** P can act only through the second step.
- **Exposed panel.** This is exposed development data: no confirmation, no significance claim and no generalization
  beyond the panel.
