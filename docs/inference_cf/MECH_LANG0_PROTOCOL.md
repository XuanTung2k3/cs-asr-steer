# MECH-LANG0 — retrospective analysis protocol (exploratory, post-hoc; NOT a confirmatory freeze)

Written and committed before any MECH-LANG0 derived quantity is computed: no new cosine, tangent alignment, gap,
realized-edit alignment or feature contrast. The historical reports and their headline outcomes were already known
when this was written (P2-DIR, ST-LOC0, ST-PROMPT-R1-A, S1, SRC-CF0-P). Every MECH-LANG0 result is therefore
**exploratory and post-hoc**. Nothing in it selects a direction, dose, sign, layer, threshold or gate for execution.

CPU only:
- Allowed: NumPy reads of sealed artifacts plus the model's `generation_config.json`, read as JSON for the frozen
  suppression list.
- Not allowed: model or tokenizer loading, forwards, autograd, Slurm, new vectors built from outcomes, new data or
  roles.

## 0. Population and identity (all analyses)

- **Master order.** The 180 queries j = 0..179 of the frozen ST-LOC0 runtime panel, keyed by
  (utterance_id, t, dialogue_id, absolute_query, content_prefix_sha256).
- **Panel equality.** This tuple must be equal, in order, in:
  - the ST-PROMPT-R1, S1 and SRC-CF0 runtime panels;
  - the SRC-CF0-P runtime projection;
  - `p2rj/positions.json`, on the subset of keys it carries.
- **Per-utterance artifacts.** These include the P2-DIR, ST-LOC0, ST-PROMPT-R1 and SRC-CF0 rows and the S1 rows. Each
  is joined by (file identity utterance_id, t). The query index stored in the file must equal j. Joins never rely on
  array order alone.
- **State identity at L16 DG-02** (post-cross-attention / pre-FFN, forced-ZH, B0M_L16 prefix). These must be bitwise
  equal:
  - P2-DIR exp1 `hb`;
  - ST-LOC0 `states_L16_CROSS.H_B`;
  - ST-PROMPT-R1 `states_L16.H_M`;
  - the SRC-CF0-P `clean_L16` (bf16 bits);
  - P2-DIR `extract_run1/states.npz` H_B (joined by its own order check).
- **State identity at L24.** ST-LOC0 `states_L24_CROSS.H_B`, ST-PROMPT-R1 `states_L24.H_M` and SRC-CF0-P `clean_L24`.
- **Baseline logits.** These must be bitwise equal: ST-LOC0 `baseline_logits`, ST-PROMPT-R1 `none_logits` and
  SRC-CF0-P `clean_r1_raw`.
- **Any failed identity check** removes the dependent calculation, with a reason. No substitution.

## 1. Analysis A — direction geometry (L16 primary; L24 limited)

- **Vectors.** All are the sealed historical float32 unit vectors.
  - v_prompt = normalize(h_E − h_M) (ST-LOC0 `v1_vectors` `_CROSS`; cross-checked against ST-PROMPT-R1 `v_prompt`).
  - v_AC = normalize(h_clean − h_mask) (SRC-CF0-P `target_L{l}_v`; 73 queries).
  - D2: the L16 reference-free readout tangent (ST-LOC0 `d2_vectors`; cross-checked against P2-DIR exp1 `D2`).
  - Random: the SRC-CF0-P per-query PCG64 `random_L{l}_v`.
- **Lexical gradient.** g = P2-DIR exp1 evaluator `g_margin`: the raw gradient of m = lse z(Y_ref) − z(c*) with
  respect to the L16 DG-02 state h = `hb`.
  - Tangent: g_tan = g − h (h·g)/(h·h), in float64.
  - Evaluator-only (reference-derived). It exists only at L16. **At L24 no lexical gradient exists. Every L24
    lexical-alignment quantity is MISSING, and no surrogate is used.** D2 exists only at L16.
- **Formulas.** cos(a, b) = a·b / (‖a‖‖b‖), in float64. Signed and absolute values are reported for:
  - (v_prompt, D2), (v_AC, D2), (v_prompt, v_AC) at L16;
  - (v_prompt, v_AC) at L24;
  - (d, g_tan) for d ∈ {v_prompt, v_AC, D2, random} at L16.
- **Sign conventions.** v_prompt + points toward the forced-EN state, and the historically favorable orientation was
  minus. v_AC + points toward "audio present". D2 + is the applied orientation. A negative cos(v_prompt, g_tan)
  therefore means the minus pulse moves along the lexical gradient.
- **Summaries,** by stratum (evaluator label used for description only) and over all queries:
  - n, mean, median, min, max;
  - fraction > 0;
  - equal-dialogue macro mean (with the number of dialogues);
  - coverage and counts of missing reasons.
- **Random reference.** The empirical SRC random cosine and the analytic isotropic reference in d = 1280:
  E|cos| = sqrt(2/(π d)) ≈ 0.0223, sd ≈ 1/sqrt(d) ≈ 0.0280.
- **Realized-edit alignment (descriptive).** Applies to L16 arms whose stored before/consumed native states exist:
  - ST-LOC0 v_prompt ±, D2 and random;
  - ST-PROMPT-R1 L16;
  - SRC-CF0-P L16.

  Definitions:
  - edit = consumed − before;
  - κ = g_tan·edit / (‖g_tan‖ ‖edit‖);
  - first-order Δm_lin = g·edit, compared with the observed Δm.
- **Comparisons not made.** No cross-layer cosine is compared as if intervention meanings were shared.

## 2. Analysis B — lexical decision gap

- **Inputs.** Processed logits: float32 from stored bf16 bits, then float64; frozen suppression → −∞ (begin-suppression
  only at t = 0, which does not occur). Y_ref = P2-RJ `target_ids` (all < EOS).
- **Gap and top-1.**
  - G(z) = max_{k∉Y} z_k − max_{k∈Y} z_k.
  - Top-1 = argmax with the lowest token ID on ties.
  - Hence top-1 ∈ Y iff G < 0, or G = 0 and the lowest tied ID is in Y. This is verified on every row.
- **Per arm and query.**
  - G_base, G_post and ΔG = G_post − G_base.
  - Δrank of the best reference token (lowest-ID tie rule).
  - Correction: base top-1 ∉ Y and post top-1 ∈ Y.
  - Corruption: the reverse, on correct strata.
  - Other top-1 changes.
  - Near-crossing: uncorrected EN-confusion rows with 0 < G_post ≤ 1 nat (also reported at ≤ 0.5), against the same
    count at baseline.
  - Fraction of the baseline gap closed: −ΔG / G_base, for EN-confusion with G_base > 0.
- **Arms** (DG-02 only; stored logits):
  - ST-LOC0 v_prompt L16/L24 ± at e* = 1.12608 (absolute chord) and same-site random.
  - ST-PROMPT-R1 v_prompt L3/L8/L16/L24 × η {.15, .30, .45} × ± (relative dose) and matched random. L3/L8 are
    supplementary.
  - SRC-CF0-P v_AC L16/L24 × η {.15, .30} × ±, with the off-target and random controls (relative dose, active rows
    only).
  - D2 L16 at e*.
- **Reporting.** Each family is reported separately on its own denominators.
- **Dose-aligned descriptive slice.** L16 at relative energy ≈ 0.15 on the v_AC-active rows:
  - v_prompt R1 η.15 ±;
  - v_AC η.15 ±;
  - D2 at e* (relative ≈ 0.148–0.152);
  - R1 random η.15.

  This is still not a controlled matched-energy comparison. Directions, random draws and row populations differ
  between studies, and this limitation is stated explicitly.
- **First-order ceiling (descriptive).** A = ‖g_tan‖ · ‖edit‖ against the baseline fixed-competitor deficit −m_base.
  It separates "weak alignment" (κ ≈ 0) from "insufficient movement even if aligned" (A < −m_base).

## 3. Analysis C — D2 correction vs Mandarin damage

- **Groups** (D2 L16 e*; evaluator labels; the reconstructed totals are checked against the historical 5 / 8 / 1):
  - D2 EN-confusion corrections;
  - ZH-correct corruptions;
  - EN-correct corruptions;
  - harmless ZH-correct, harmless EN-correct and uncorrected EN-confusion.
- **S1 eligibility.** `target_status == OK` and `paired_available`, from the sealed S1 candidates record, cross-checked
  against the SRC-CF0-P runtime regions. The historical overlap is 3 of 5 corrections and 3 of 8 Mandarin corruptions;
  this is reconstructed, not assumed.
- **Pre-declared features.** Exactly these six; no others are searched. All are inference-available, being recorded by
  reference-free runners.
  1. F1: S1 predicted-English target region available (`target_status == OK`).
  2. F2: query→region attention integral (S1 `region.target.integral`; present only when a region crop exists).
  3. F3: raw heard attention mass (S1 `region.raw_heard_mass`).
  4. F4: utterance native LID log-odds, logit(en) − logit(zh) (S1 `auto.language_logits`; one call per utterance).
  5. F5: baseline decoder script log-odds J0 = log P_E − log P_M (P2-DIR D2 runtime provenance).
  6. F6: baseline decoder top-1 probability (NONE processed distribution).
- **Reporting.** Per group: n, number of dialogues, median, IQR and min/max. One descriptive rank statistic
  P(X_a > X_b) + ½ P(=) per contrast, for two contrasts: corrections vs ZH corruptions, and ZH corruptions vs harmless
  ZH. No threshold, classifier, fitted gate or optimization. With n = 5 vs 8, nothing is claimed as discrimination.
- **Classifying information.** For every quantity the report states whether it is:
  - (a) inference-available;
  - (b) evaluator-only;
  - (c) unavailable at inference.

## 4. Analysis D — hypothesis assessment

- **H1 (transcription vs translation task contrast).** Search the existing archives for task-token (50358 / 50359)
  activation pairs at the DG-02 site. If none exist, construction feasibility is UNKNOWN. No evidence of lexical
  alignment is inferred.
- **H2 (pooled language/script direction).** Reference-free geometry of the per-query v_prompt at L16/L24 only:
  - mean resultant length ‖mean_i v_i‖;
  - leave-one-dialogue-out cos(v_i, normalize(mean of the other dialogues' v));
  - median pairwise cosine.

  No pooled direction is evaluated against any lexical label or gradient.
- **H3 (selective readout).** Uses only Analyses A–C.
- **Ranking criteria.** Lexical influence, compatibility with code-switching confusions, reference-free information,
  Mandarin risk, compute and novelty. One advisory label is chosen from the four allowed.

## 5. Outputs and checks

- **Files.**
  - `experiments/inference_cf_mech_lang0.py` (single script);
  - `tests/test_inference_cf_mech_lang0.py`;
  - `results/inference_cf/mech_lang0/analysis.json`;
  - `results/inference_cf/mech_lang0/manifest.json` (hashes of every consumed artifact);
  - `docs/inference_cf/MECH_LANG0_REPORT.md`.
- **Historical sanity checks.**
  - P2-DIR D2: 5 corrections, 8 ZH-correct and 1 EN-correct corruption, 43 top-1 changes.
  - ST-LOC0 v_prompt: 0 corrections at L16/L24.
  - ST-PROMPT-R1: at most 1 correction per arm.
  - SRC-CF0-P: 0 corrections in all 8 arms.
  - Historical P2-DIR κ and cos(d, g_tan) medians reproduced.
- **Failure handling.** A mismatch is reported, never silently patched.
