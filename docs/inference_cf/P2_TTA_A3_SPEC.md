# P2-TTA-A3 — Soft AUTO-KL conditioning-gap debug (pre-outcome specification)

Frozen before any A3 adaptation output. Machine contract: `configs/inference_cf/p2_tta_a3.json`; panel
`docs/inference_cf/P2_TTA_A3_PANEL.json`; implementation plan `P2_TTA_A3_CLAUDE_DESIGN.md`.

This is a separate, compact, exposed-development **debug** stage after the completed P2-TTA-FUNNEL
(`P2_TTA1_SUPPORTED`, A2 AUTO-consistency). Historical TTA0/TTA0-R/TTA1 outcomes, reports and seals are not modified.
The core v6 METHOD_CONTRACT is unchanged. Starting HEAD: `59788c869a97a48645fe9522b1c63ad79c7c6acb`, clean, no jobs.

## Question

A2, which fits the hard AUTO transcript, beats forced-ZH, but on the 12 fixed-100 utterances where theta0 AUTO ≠
FORCED it barely moves output toward AUTO (summed token distance 149 → 146). Is the hard AUTO transcript too weak a
signal? A3 replaces it with the full soft AUTO next-token distribution and measures:

1. whether the teacher-forced conditioning gap actually closes under the unchanged A2 actuator;
2. whether free decoding then moves toward AUTO;
3. whether that movement is safe and useful compared with A2.

The aim is to separate these failure modes: objective/actuator coupling, free-decode sequence leverage, AUTO
teacher safety, conditioning not being the bottleneck, or A3 promising.

## Panel (reference-free, frozen)

The parent is the exact fixed-100 panel (`P2_SEL_MINI_PANEL.json`, sha256 `266ea7ea…`).
- **D:** stored theta0 B0-AUTO decoded text ≠ stored theta0 B0-FORCED (P2-SEQ S0) decoded text, compared exactly with
  no normalization. Reproduced independently: |D| = 12 across 9 dialogues; 88 A. This matches the funnel's frozen split.
- **All 12 D rows** are included, since |D| ≤ 16.
- **12 A controls**, filled in this order:
  1. one row per dialogue *not* represented in D, taking dialogues in first-appearance order and rows in parent order
     (11 rows);
  2. round-robin over D-represented dialogues that have fewer than 2 selected rows (1 row: CSD0011, `U0022_S0_91`);
  3. any dialogue with fewer than 2, then any dialogue only if still required (not needed).
- **Result:** 24 utterances, 20 dialogues, sha256 `6422741b91e64ff007458b7a04d2f559e8ed8ffb592a087a538adf63632b2206`.
  The maximum of 3 per dialogue occurs only because CSD0012 has 3 D rows, which must all be included.
- **Inputs used:** only hash-anchored theta0 transcripts. No reference, error count, POI count, duration, A2 outcome or
  quality score. This is an enriched diagnostic panel, not a population estimate.

## Fixed actuator, inherited exactly from A2/TTA1

| Setting | Value |
|---|---|
| Trainables | the identical 194 decoder LayerNorm affine tensors (248,320 scalars), same names/shapes |
| Masters and optimizer | fresh fp32 masters and fresh AdamW per utterance: lr 1e-3, wd 0, betas (0.9, 0.999), eps 1e-8, no amsgrad/foreach/fused |
| Steps | exactly 2 updates; 3 loss evaluations |
| Precision / forward | bf16 eager eval forward through `functional_call` casts |
| Reset | exact bf16 theta0 restore with bitwise verification after every utterance |
| Encoder | detached theta0 encoder computed once per utterance |
| Caches | fresh KV cache for every decode |
| Final decode | forced-ZH greedy (`cB = [50258, 50260, 50360, 50364]`) on the unsteered cached path, 200-token cap, unchanged suppression |
| Excluded | AUTO condition, steering, D2 and LID during the final decode |

No sweep or tuning; no A1/A2 rerun.

## A3 objective

1. **Frozen AUTO path.** y_A is the historical theta0 AUTO greedy content IDs, reused unchanged (not regenerated).
2. **AUTO condition.** Recovered at theta0 exactly as historical `generate(language=None)` did:
   `model.detect_language(input_features=<historical features>, generation_config=<model generation_config>)` gives
   lang. Then cA = [SOT 50258, lang, transcribe 50360, notimestamps 50364]. EN is never substituted; a ZH result is
   allowed.
3. **Teacher-condition integrity.** The theta0 cached-path greedy decode under cA must reproduce the historical AUTO
   decoded text for every panel row. Any mismatch → INVALID. This is an integrity check only; the replay output is
   never used as a teacher.
4. **Common path.** Both the teacher (cA + y_A) and the student (cB + y_A) are teacher-forced on the same y_A content
   prefix, with no KV cache. Content step t uses absolute query 3 + t; the prefix and the final EOS query are not loss
   positions.
5. **Vocabulary.** Generation-allowed only: all IDs minus `suppress_tokens`, minus `begin_suppress_tokens` at t = 0
   only. EOS stays in the vocabulary. Teacher and student use the same allowed set at each t.
6. **Valid positions.** The original A2 y_A mask: a non-special content ID whose target is allowed at step t. It is
   fixed before the updates.
7. **Teacher distribution.** q_t = softmax over the allowed IDs of the theta0 teacher logits, computed in float32 at
   temperature 1. It is computed once before any update and detached. The full allowed vocabulary is used, with no
   top-k. Teacher arrays are transient and never saved.
8. **Loss.** `L_A3(θ) = D_cond(θ) = (1/T_valid) Σ_valid t KL(q_t || p_θ,t)`, where p_θ is the forced-ZH student,
   computed in float32 via stable log_softmax. This is the **forward KL** with the AUTO teacher first. There is no hard
   NLL, entropy, anchor, reference, lambda, temperature or weighting term.
9. **Identical-condition rule** (pre-registered, exact).
   - If cA equals cB, then q equals p_θ0 bitwise. D_cond is then identically zero along the trajectory, so its exact
     gradient is zero.
   - Such a row runs exactly 2 no-op AdamW steps on a scalar zero connected to the masters (the TTA0 empty-teacher
     convention). It records D_cond = 0, and its final decode must equal B0-FORCED.
   - The reason: Adam's first step is scale-invariant, so the fp32 rounding noise of a mathematically zero gradient
     would otherwise become an update of nearly full lr size.
   - Rows with no valid content follow the same no-op rule.

Per-utterance logs:
- D_cond0/1/2 and the relative reduction (D_cond0 − D_cond2)/max(D_cond0, 1e-12);
- loss0/1/2, gradient norms before each update, fp32 master and effective bf16 update norms, finite flags;
- student entropy, P_E/P_M and EOS probability on the y_A path at steps 0 and 2;
- the final transcript, its length and cap/EOS;
- d_BA, d_A2A, d_A3A and closer/same/farther;
- runtime, VRAM, and forward/backward counts.

## Frozen mechanistic diagnostics (group D)

- **GAP_CLOSED** requires both:
  - at least ceil(0.75·|D|) = 9 of 12 D rows with D_cond2 < D_cond0;
  - median relative gap reduction over all D rows ≥ 0.25.
- **Distances** use the TTA1 token Levenshtein: d_BA = d(B0-FORCED, B0-AUTO), d_A2A = d(A2, AUTO), d_A3A = d(A3, AUTO).
  A3 is CLOSER if d_A3A < d_BA, SAME if equal, FARTHER if greater.
- **Pooled distance reduction:** R_dist_X = 1 − Σ_D d_XA / Σ_D d_BA, reported for A2 and A3.
- **MOVEMENT** passes if either (a) at least 4 D rows are CLOSER and at most 1 is FARTHER, or (b) R_dist_A3 ≥ 0.15.

These are diagnostic materiality screens, not significance tests. A-control D_cond, the change counts and the safety
contribution are reported separately.

## Frozen safety (copied exactly from frozen TTA0/TTA1)

These are A3 versus matched B0-FORCED on all 24 rows:

| Bound | Value |
|---|---|
| MER increase | ≤ 0.01 |
| ZH-CER increase | ≤ 0.015 |
| Matrix-ZH retention | ≥ 0.98 |
| Embedded-EN retention | ≥ 0.95 |
| Outside-POI lexical harm | ≤ 0.03 |
| POI corruption rate | ≤ 0.05 |
| Added caps | ≤ 1 |
| New severe truncations (B0 length ≥ 10 and A3 EOS length ≤ floor(0.5·B0 length)) | 0 |

- An empty baseline-correct denominator means the bound is not assessable and passes vacuously. The float guard is
  1e-12.
- **Provenance:** `configs/inference_cf/p2_tta0.json` (decision thresholds) and `configs/inference_cf/p2_tta_funnel.json`
  (`TTA1.thresholds`). Their safety bounds are identical (they differ only in the benefit rule), so there is no conflict.
  The TTA1 confirmation safety vocabulary is used as the authority; it includes POI corruption.
- **"Materially worse than B0-FORCED on the canonical safety outcome"** means exactly that any one of these bounds
  fails. No additional criterion is introduced.

## A3 vs A2 (frozen before outcomes)

- **A2 reference:** the audited TTA1 SELECTED-TTA output for the same IDs, reused and never rerun.
- **NO_ADVANTAGE** means either:
  - A3 POI errors > A2 POI errors; or
  - equal POI errors and MER(A3) − MER(A2) > 0.01, the TTA safety-scale MER tolerance.
- **PROMISING** needs both POI(A3) ≤ POI(A2) and MER(A3) − MER(A2) ≤ 0.01 + 1e-12.
- **Resolution of a spec gap:** if A3 has *fewer* POI errors than A2 but MER worse by more than 0.01, the A2 condition
  of PROMISING fails, so the label is CONDITIONING_ONLY. CONDITIONING_ONLY is defined as gap + movement + safety pass
  but PROMISING not met.

## Label precedence (exactly one)

1. **`P2_TTA_A3_INVALID`**: engineering, reset, panel, objective, numerical, live-audit, reuse, AUTO-replay, leakage,
   incomplete or audit failure.
2. **`P2_TTA_A3_GAP_NOT_CLOSED`**: GAP_CLOSED fails.
3. **`P2_TTA_A3_SEQUENCE_LEVERAGE_LIMIT`**: the gap closes but MOVEMENT fails.
4. **`P2_TTA_A3_TEACHER_UNSAFE`**: gap and movement pass but any frozen safety bound fails.
5. **`P2_TTA_A3_CONDITIONING_ONLY`**: gap, movement and safety pass but the PROMISING A2 condition fails.
6. **`P2_TTA_A3_PROMISING`**: all of the above pass, POI(A3) ≤ POI(A2), MER not worse than A2 by more than 0.01, no
   new severe truncation, and the independent post-audit passes.

PROMISING authorizes only a recommendation for a separately frozen A3-on-100 comparison. It is never run here.

**Interpretation map:**

| Outcome | Reading |
|---|---|
| Gap not closed | objective/actuator coupling failure |
| Gap closed, no movement | free-decode sequence leverage limit |
| Movement but unsafe | AUTO teacher / objective safety failure |
| Safe movement but A2 better | conditioning is not the remaining bottleneck |
| A3 ≥ A2 | the soft conditioning target is promising |

## Integrity and numerics

- **Before the run:** tiny-model tests check KL orientation, mask, detachment, float64 identity with an independent
  formula, fp32 repeat, exactly 2 steps, LN-only gradients, reset, fresh optimizer, forced-ZH final decode, no
  steering, and references closed.
- **During the run:** on the first panel row (a D row), an independent full-vocabulary masked forward-KL formula must
  agree with the primary at theta0 within loss ≤ 1e-5 and gradient ≤ max(1e-8, 0.02·‖g_auditor‖). This is the funnel
  convention for numerically independent fp32/bf16 gradients.
- **INVALID triggers:** any nonfinite value, reset failure, theta0 forced decode ≠ S0, AUTO replay ≠ historical AUTO
  text, missing row, or count mismatch.
- **Evaluation gate:** references open only after the committed output seal.
- **Bootstrap:** dialogue-block, 2000 draws, seed 240924, descriptive only.

## Compute and firewall

- One sbatch job on one H100 MIG 3g.40gb, 4 CPU, 48 GB.
- 24 utterances, 48 optimizer steps, ≤ 49 backward passes (48 + 1 live check).
- Target under 10 minutes; hard limit 30 minutes. At most one pending/running job.
- No A1/A2 rerun, no 100-panel A3, no full-300, no P3.
- Allowed data: already-exposed D-dev-select fixed-100 artifacts only.
- Forbidden: D-dev-confirm, D-test, router-calib new role, full-300, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND, transfer
  corpora, fresh validation.

## Audit-repair protocol

An auditor implementation defect may be repaired only if all four hold: the scientific output is unchanged; frozen
data, settings and thresholds are unchanged; the failed attempt is preserved; and the fix is tested and documented. A
genuine frozen scientific or audit failure stops the stage.
