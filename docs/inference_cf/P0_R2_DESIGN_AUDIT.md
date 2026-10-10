# P0-R2 design audit — `LocalSupport × BaselineConflict`

**Status: `R2_DESIGN_READY_WITH_REQUIRED_REVISIONS`.** Design audit only. No production code,
no GPU job, no intervention, no P1. Historical P0/P0-R1 artifacts untouched.

The proposed gate is coherent and worth testing **only after seven minimal revisions (§10)**. One
of them fixes a would-be inference-validity blocker: the plan's "validated token/audio alignment"
(Whisper DTW timestamps) depends on future tokens. Once the revisions are frozen, the next pass
freezes the P0-R2 feasibility spec from §12.

---

## 1. Evidence inspected

| Item | Value |
|---|---|
| Worktree / branch | `/home/tungnx/cs-asr-steer-inf`, `feature/inference-cf-steering` (old path `/home/tungnx/cs_asr_steer_inf` is a compatibility symlink) |
| HEAD | `08643eb` (P0-R1 final contract); history `55faf76 → cba7784 → e4bc24b → ee6ffd4 → 7b1931d → 0809930 → 08643eb`; tracked tree clean |
| Revised plan | `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md`, **untracked**, 446 lines, sha256 `365837fb…f8ad29` (see MINOR-5) |
| Governing docs read | `AGENTS.md`, `CLAUDE.md`, `FEASIBILITY.md`, `P0_INDEPENDENT_AUDIT.md`, `docs/inference_cf/{P0_FEASIBILITY_SPEC,P0_R1_EVIDENCE_SPEC,P0_R1_REPORT}.md`, `docs/current/{STATUS,CODE_MAP,DATA_EXPOSURE}.md`, `docs/V2R3_CROSS_ATTENTION_SPANS_2026-08-15.md`, `docs/RUNBOOK_V2_2026-08-14.md` (`whisper_dtw` sections) |
| Code read | `transformers 4.57.6` `WhisperGenerationMixin.detect_language`, `_extract_token_timestamps`, `WhisperEncoder` length check; local `generation_config.json` (`alignment_heads`, `lang_to_id`); `csasr.inference_cf.{core,core_r1}`, `experiments/inference_cf_p0*.py`, `csasr.lss.sites.DecoderPostCrossAttnRecorder`, `csasr.data.alignment` (`token_char_offsets`, `build_prefix`), `csasr.evaluation.pier` (`evaluate_pois`, `LANGUAGE_CONFUSION`), `experiments/dg04_frozen_baselines.py` |
| Data read (all `D-dev-select`, already exposed) | `results/dg04/results/B0.json`; `B0_AUTO/D-dev-select.parquet`; `poi_D-dev-select.parquet`; P0/P0-R1 row shards and manifests |
| CPU probes run (no GPU, no dev-outcome of any gate) | full-vocabulary script partition; script classes of the 2,895 actual baseline tokens saved in P0 rows; canonical POI categories for the two baselines; `R_t` algebra; **native LID on an all-zero waveform** (fp32, CPU) |

## 2. Historical constraints — confirmed, not rescued

1. `cB == cM`, so `hB == hM`, **for the forced-`zh` baseline actually used by P0 and DG-04 B0**.
   Verified in P0 (`[50258,50260,50360,50364]` for both; bitwise-identical L24 states). `rho/c` is
   retired. *This identity does not hold for an auto-LID baseline (MAJOR-4).*
2. Same-content-prefix, mask, cache and lineage plumbing passed (P0 G2, 0 violations in 57 rows).
3. K=1 and K=3 continuation evidence collided (87.7%, 73.7%) and failed frozen G3. It is retired
   and not revisited here.
4. P0 showed only that `hE−hM` is finite and nontrivial (median `‖hE−hM‖` 2.75). It gives no
   evidence that steering along it helps ASR. **New observation (exploratory, P0-R1 rows):**
   `‖hE−hM‖` is the same size across strata (AUROC wrong-English vs Mandarin 0.49), so the
   direction carries no localization information. That is consistent with separating "how" from
   "when".

## 3. The decomposition and the actual failure mode

The when/how split is logically sound. The `E`-high/`R`-high contradiction is the right shape for
the failure the method targets. **It does not target all baseline-wrong English.**

The canonical POI categories were recomputed with `csasr.evaluation.pier` on the baseline R2 must
actually decode (DG-04 B0, `language="zh"`, PIER 0.470, which reproduces its recorded 1,066 errors):

| Error category (of 1,066 English POI errors) | Share | What `R_t` does | In scope? |
|---|---:|---|---|
| deletion | 49.4% | no generated token of its own; at the gap slot `R` reflects the next emitted token (usually Han) | **conditional**: only if the localizer's window still covers the skipped English audio |
| wrong_language_substitution (EN → Han) | 31.9% | high | **yes** |
| same_language_substitution (EN → other EN) | 12.2% | low (Latin mass) | **no, by construction** |
| phonetic_transliteration_or_script | 3.1% | high | **yes** |
| boundary_error / other (digits etc.) | 3.4% | ambiguous or ineligible | no |

**The scientific target must be narrowed** to *embedded-language errors associated with
matrix-language confusion*: the canonical `LANGUAGE_CONFUSION` set = wrong-language substitution
+ transliteration (35.0% of errors), plus deletions reported as a separate, conditional stratum.
Consequences:

- **PIER stays the headline metric**, but the method's reachable ceiling is the confusion (plus
  part of the deletion) subset. Report category-resolved corrections and corruptions.
  Same-language substitutions that stay wrong are *expected*, not method failures.
- **R2 strata must be split**: `EN-confusion`, `EN-deletion-slot`, `EN-same-language-sub`,
  `EN-correct`, `ZH-correct` (matrix). Gate criteria are defined on `EN-confusion` (primary) and
  `EN-deletion-slot` (secondary); `EN-same-language-sub` is reported separately and never counted
  as a miss.
- **Manuscript claim**: "reduces matrix-language confusion of embedded English", not "improves
  English recognition in general".
- **Consequence for evaluating R**: on the baseline's own greedy path, `R_t` is close to a
  function of the emitted token's script (greedy emits the argmax). Its separation of confusion
  from correct English is therefore close to **tautological**; it confirms coverage but proves
  nothing further. The substantive discrimination is **among positions where the baseline emits
  Han**: English actually spoken (confusion) vs Mandarin actually spoken (correct matrix). That
  burden falls almost entirely on `E_t`. R2's headline criterion must test exactly that
  (§12, `R2-GATE`).

## 4. `LocalSupport` audit

### A. Is token-local native Whisper LID meaningful?

- **Implementation**: `detect_language` runs one decoder step with only `<|startoftranscript|>`
  and reads the logits of the 100 language tokens. It is segment-level by design.
- **Fixed 30 s input**: `WhisperEncoder` raises unless the mel input is exactly 3,000 frames
  (30 s). A cropped window must be **zero-padded to 30 s**, so a 1 s window is 96.7% padding.
- **No encoder attention mask**: Whisper applies none. Repository evidence (v2r3
  cross-attention study): on average 80% of cross-attention mass lands on padding beyond the audio,
  up to 94%.
- **Measured padding-template prior (CPU, fp32, all-zero waveform)**: π(en)=0.3099,
  π(zh)=0.0199, next ru/ja/es/fr. Pairwise **E = 0.940**; pair mass 0.33. The 1 s-padded and 30 s
  silent inputs are identical after feature extraction.

**So a mostly-padded window is pulled toward "English" by a non-acoustic template.** That is
exactly the false-activation direction on Mandarin positions. The raw pairwise posterior is not
"local acoustic support". Short-window validity is unproven and is what R2-LS must test.

### B. Can `x_t` be built reference-free and causally?

- **The plan's "validated token/audio alignment" is future-dependent.**
  `_extract_token_timestamps` needs the completed generated sequence. It z-scores every frame
  column across **all output tokens** (`dim=-2`), median-filters, then runs one global monotonic
  DTW over the whole token×frame matrix. Token `t`'s timestamp depends on later tokens. It is
  **post-hoc** and **cannot define a deployable current-step gate** (CRITICAL-1). It is also
  historically weak here: `whisper_dtw` had 17.4% invalid units, all Mandarin.
- **Current-step cross-attention is causal.** At decode step `t`, the baseline forward has the
  query that predicts `y_t`. Its cross-attention over encoder frames depends only on `x`, `cB` and
  `y_<t`. The local model ships 10 timing `alignment_heads`
  (`[7,0],[10,17],[12,18],[13,12],[16,1],[17,14],[19,11],[21,4],[24,1],[25,6]`, including one at
  L24).
- **Minimal inference-valid localizer** (§12). Average those heads' softmax weights for the
  current query only. Restrict to the audio's own frames (`F = min(1500, ceil(seen_duration/0.02))`,
  derived from the waveform). Renormalize and take `c_t = argmax`. No cross-token normalization,
  no DTW, no future tokens.
- **Offline equivalence**: under causal self-attention, a teacher-forced full replay of the fixed
  baseline sequence gives, for every `t`, exactly the causal quantities (future tokens cannot
  affect earlier queries). R2 may therefore compute everything in one replay per utterance.
- **Risks R2 must measure, not assume**: attention sinks (raw final-layer attention sent 93% of
  frames to special tokens in v2r3); peaks at the wrong place; and self-localization by a failing
  decoder. On a deletion, the attention may already have moved past the English audio, so `E` stays
  low and the error is systematically missed. Localizer quality is a feasibility verdict in its own
  right, validated against evaluator-only MMS-FA CTC reference-unit times (§7).
- Utterances over 30 s (6/60 in the P0 panel): only the first 30 s is heard. Windows are clipped
  to the seen audio, and reference targets beyond it are unalignable.

### C. Cost

- **Encoder passes**: each gated position needs its own padded 30 s encoder pass plus one decoder
  step. Baseline content is about 51 tokens per utterance, so ~45 eligible positions cost ~45
  extra full encoder passes, versus 1 for ordinary decoding. That is roughly 45× the baseline
  encoder FLOPs (≈2 TFLOP per pass; estimate, not measured).
- **Valid savings**:
  - exact caching of **identical** windows, keyed by `(audio sha, start_frame, end_frame,
    provider version, precision)`; consecutive tokens of one word often share a window;
  - batching all windows of an utterance offline in R2;
  - batching across utterances in deployment.
- **Invalid savings**: reusing the full-utterance encoder output and masking frames. Those states
  are contextualized by the whole utterance, so that would be a **different provider** and could
  only be introduced as one, versioned.
- **Verdict on cost**: heavy but testable. It must be disclosed as a main cost result. It is not a
  blocker for R2 (dense D-dev-select ≈ 15k padded encoder passes, one MIG job).

### D. Verdict — `LOCAL_SUPPORT_VALID_WITH_REQUIRED_REVISION`

The factor is inference-valid only with the **causal current-step localizer** (not DTW) and a
**padding-template correction** of the language log-odds (§10 revisions 1–2). Whether it actually
separates English from Mandarin speech on short windows is an open empirical question for R2.

## 5. `BaselineConflict` audit

### A. Mathematics

- **Identity holds**: for masses `a, b > 0`, `(a−b)/(a+b) = tanh(½ log(a/b))`.
  - Measured on 10⁵ random pairs with ε=1e-12: the plan's form `(P_M−P_E)/(P_M+P_E+ε)` differs
    from `tanh(C_t/2)` by ≤5.7e-11.
  - The form `(P_M−P_E)/(P_M+P_E+2ε)` equals `tanh(C_t/2)` to 3e-16. Freeze that one (revision 3).
- **Properties**: bounded in [0,1] after `[·]_+`; zero when `P_E ≥ P_M`; 0/0 → 0. It is a
  principled posterior-odds margin (the pairwise-renormalized matrix-over-embedded probability
  margin), not an arbitrary normalization. The positive part correctly makes it one-sided: an
  English preference is never "conflict".
- **Numerics**: compute `P_M`, `P_E` by `logsumexp` over class log-probs in float32, from bf16
  logits upcast before `log_softmax`.

### B. Tokenizer feasibility (51,866 ids)

| Class | Count | Proposed set |
|---|---:|---|
| ASCII-Latin word tokens (letters, optional leading space, `'`/`-`) | 37,858 | `V_E` |
| pure Han (CJK Unified/Ext A/compat, optionally with CJK punctuation) | 1,365 | `V_M` |
| UTF-8 fragments whose lead byte is `0xE4–0xE9` (Han range) | 302 | `V_M` |
| continuation-only byte fragments | 249 | ambiguous |
| other-lead fragments (kana/Hangul/accents…) | 925 | ambiguous |
| other scripts (Cyrillic, accented Latin, kana…) | 8,600 | ambiguous |
| digits / punctuation-symbol / whitespace / control bytes | 427 / 408 / 11 / 99 | ambiguous |
| Han+kana mixed | 13 | ambiguous |
| special/control (EOS, SOT, language, task, timestamps) | 1,609 | ambiguous |

**Actual baseline output (P0 rows, 2,895 tokens):**

| Token class | Share |
|---|---:|
| Han | 62.6% |
| Latin | 19.3% |
| Han-lead fragment | 6.8% |
| continuation fragment | 6.8% |
| punctuation | 4.4% |
| digit / whitespace | 0.2% |

**6.8% of decode steps start mid-character.** The prefix ends in an incomplete UTF-8 sequence, so
the next token must be a continuation byte and the language is already decided. `R_t` is
meaningless there, and so is an edit. **4 of the 57 P0 diagnostic positions were mid-character**
(for example `yE = yM = '�,但'`), a previously unflagged P0 detail.

**A deterministic partition is feasible**, with three caveats:

- Han-lead fragments must go into `V_M`, or Mandarin mass is undercounted.
- Mid-character steps must be structurally ineligible.
- `V_E` is **ASCII-Latin script, not English**: pinyin, romanized names and other Latin languages
  count as E. That is acceptable only because the matrix language is Han-script.

### C. Classified mass `Q_t = P_M + P_E`

`R_t` depends only on the ratio of the two masses. Measured: `P_M=1e-9, P_E=1e-10 → R=0.82`;
`P_M=0.02, P_E=0.0005 (Q=0.02) → R=0.95`. Punctuation, EOS or byte steps can therefore show
"maximal conflict" built from noise-level masses.

- **Logging alone is insufficient**: the quantity is numerically undefined as `Q→0`, not merely
  noisy.
- **A bounded reliability multiplier (`Q·R`) is not the minimal fix**: it changes `R` from a
  conditional preference into a joint mass and adds a factor.
- **A minimum-mass fallback is required and sufficient** (revision 4): if `Q_t < ½` (most
  next-token mass is not language-bearing), `R_t` is undefined, the gate falls back to `g_t = 0`
  (no repair), and the reason is recorded. `½` is a majority rule, not a tuned value.
- `Q_t·R_t` may be reported **descriptively only**; it cannot replace the primary after outcomes.
- R2 must report `Q_t` by stratum. If abstention hits deletion slots (next token = punctuation),
  that is a finding, not a reason to lower the threshold.

### D. What `R_t` detects

`R_t` measures **the baseline's next-token script preference (Han vs ASCII-Latin), restricted to
language-bearing mass.** It is matrix-*script* preference, not lexical language and not error.

| Case | `R_t` behavior |
|---|---|
| Mandarin token | high |
| correct English | low |
| English → Mandarin substitution / transliteration | high (targeted) |
| English → another English word | **low (not detectable)** |
| deletion gap slot | usually high (next emitted Han), but depends on the localizer |
| punctuation / digit | ineligible or ambiguous |
| English named entity rendered in Han / pinyin names | ambiguous either way |

**Claimable scope**: "baseline matrix-script preference at positions with language-bearing next-token
mass, for a Han-matrix / Latin-embedded pair".

### E. Multilingual limitation

Han-vs-Latin is a first instantiation only. It is **invalid for same-script CS-FLEURS pairs**
(e.g., Spanish–English, both Latin) and weak for pairs where either language uses Latin
transliteration. A future `BaselineConflict` provider must satisfy all of:

- **Input**: `p_t^B` over the full vocabulary, with `M`, `E` given as inputs.
- **Outputs**: `P_M`, `P_E`, uncovered mass `1−Q`, and `R_t ∈ [0,1]`, with `R=0` when E-mass ≥
  M-mass.
- **No references**: no reference or target lexicon from evaluation data.
- **Development-only freeze**: defined and frozen on development data only, before transfer.
- **Script-independent**: language affinity must not rely on script.
- **Degeneracy check**: must reduce to the Han/Latin partition, or be validated against it, on the
  primary pair.

A conditioned-branch affinity (e.g., `log pE(v) − log pM(v)`-weighted mass) is one candidate. It
is not designed here.

### F. Verdict — `BASELINE_CONFLICT_VALID_WITH_REQUIRED_REVISION`

Required: the 2ε epsilon form; the `V_M ⊇` Han-lead-fragment partition with explicit ambiguous
classes; mid-character ineligibility; and `Q_t ≥ ½` abstention (§10 revisions 3–5).

## 6. Product gate `g = E·R`

- **Soft AND**: the product t-norm is a sensible soft AND of two independently interpretable
  factors. It is high only when both are high. No sigmoid or temperature is needed.
- **Which factor dominates where**:
  - On correct English, `R≈0` → `g≈0`: R carries the veto.
  - On Mandarin, `R≈1` → `g≈E`: **Mandarin false activation is governed entirely by `E`'s floor.**
  - Hence the padding-template bias (§4A) is the single largest threat to the product. The `g=R`
    ablation (C3) fires on essentially every Mandarin position by design.
- **Collapse toward zero**: the product of two soft scores is small in absolute terms, but ranking
  (AUROC/PR) is unaffected. Absolute scale is absorbed by `α` and the planned gate-RMS dose
  matching in P2.
- **Calibration**: **not required for R2 feasibility.** Treat `g_t` as a **ranking / repair-need
  score**, never a calibrated error probability. Any calibration would be a declared P2 parameter,
  not an R2 choice.
- **Discrimination by construction**: matrix → low (via E); correct embedded → low (via R);
  confusion → high only if E is high where English was spoken.
- **Cases where both factors are high but steering is not warranted**:
  1. Mandarin speech whose short window is pulled toward English by the padding template.
  2. English-origin names or loanwords that the references (canonically) write in Han.
     Steering would corrupt a correct Han transcription.
  3. Localization bleed: a Mandarin step whose window centers on an adjacent English word,
     risking English insertion or duplication.
  4. Hallucination or repetition loops with collapsed attention.
  5. Mid-character steps (handled by revision 5).
  6. English words pronounced with Mandarin phonology.

  Items 1–3 must be measured in R2 (Mandarin false-activation rate; neighbor-of-English Mandarin
  positions reported separately).
- **Verdict — `GATE_COMPOSITION_VALID_WITH_REQUIRED_REVISION`**. The revision is only the explicit
  eligibility and fallback rule: `g_t = 0` with a reason when mid-character, `Q_t<½`, localizer
  failure, or nonfinite. The product itself stands.

## 7. Diagnostic labels and alignment

**Findings:**

- **Label-provenance mismatch (MAJOR-3).** P0's wrong/correct strata came from `poi_D-dev-select`,
  which was computed on **B0_AUTO** (`language=None`, batch 16). P0 decoded **DG-04 B0**
  (`language="zh"`).
  - The transcripts differ in **46/300 utterances**.
  - POI errors are 884 vs 1,066; wrong-language substitutions 175 vs 340.
  - It does not change P0/P0-R1 verdicts, which failed on collision, but R2 labels must be
    recomputed against the exact baseline R2 decodes.
- **The P0 character-fraction mapping cannot be reused.** It mislabels (an "English" row landed on
  `来`) and ignores deletions.

**Existing infrastructure suffices** for an evaluator-only mapping:

- `pier.evaluate_pois` / `align_tokens`: canonical ref↔hyp unit alignment and categories.
- `alignment.token_char_offsets`: generated token ↔ hypothesis character offsets.
- `candidates_existing_ctc.parquet`: MMS-FA reference-unit times, used only to validate the
  localizer.

**Smallest defensible mapping (evaluator-only, deterministic):**

1. Decode the exact R2 baseline (`cB`, frozen config). Keep its token ids.
2. Normalize reference and hypothesis with the canonical normalizer. Compute canonical unit
   alignment and category for **every** reference unit.
3. Map each hypothesis unit to its **first** generated token index via `token_char_offsets` over
   the normalization offset map. Any non-unique or offset-inconsistent mapping becomes
   `unalignable:offset`.
4. Assign positions by case:
   - **correct or substituted unit**: `t` = step predicting the first token of its aligned
     hypothesis unit.
   - **deleted unit**: `gap slot` = step predicting the first token of the next aligned hypothesis
     unit, or EOS; consecutive deletions share one slot.
   - **boundary errors, targets past 30 s of audio, truncation at 200 tokens**: `unalignable:*`.
5. Nothing is forced. Unalignable units stay in the counts and are excluded from AUROC
   denominators.
6. Localizer validation: compare `c_t` to the CTC midpoint of the aligned reference unit.

**Panel**: **reconstruct, do not reuse.** Build a new versioned panel from the **same development
population** (`D-dev-select`, already exposed). Evaluate **every eligible generated position and
gap slot** in each included utterance (dense). Membership is decided by label availability only,
never by gate values.

- **Preferred**: all 300 D-dev-select utterances (20 dialogues). This gives ≈373 confusion POIs;
  intervals should be dialogue-cluster bootstrap.
- **Alternative**: the 60 P0 utterances, flagged as a subset.
- No new data role is touched.

## 8. Paper flow, multilingual and Qwen

| Check | Result |
|---|---|
| R2 has no steering, alpha or layer optimization | ✅ plan §3.7, §4 |
| Direction usefulness not claimed before P1 | ✅ §3.1, §9.7 |
| C1/C2/C3 = constant gate / `g=E` / `g=R`; M = full `E·R` | ✅ §9.2 |
| Retired `rho/c`, `q_cross` not main-method components | ✅ §1, §4, Figure M1 note |
| Training-free / reference-free at inference | ✅ **only with the causal localizer** (CRITICAL-1) |
| CS-FLEURS same-script not supported by Han/Latin | ✅ stated in §3.4, §9.1, A1 |
| Qwen revalidates conditioning, local support and conflict | ✅ A3. Also Qwen3-ASR may expose no native per-window LID equivalent; the provider must be revalidated or declared ineligible. |

**Contradictions to fix in the R2 spec (plan not edited; the spec supersedes on these points):**

1. §3.3 "validated token/audio alignment" means post-hoc DTW, which conflicts with inference-time
   validity → causal localizer.
2. §1/§3.1 state `cB==cM` as a Whisper property. It holds only for the forced-`zh` baseline.
   **Which baseline is B0 is undeclared** (MAJOR-4): the auto-LID baseline is *stronger* (PIER
   0.390 vs 0.470) and has half the wrong-language substitutions. If B0 = forced-`zh`, then
   B0_AUTO must be a reported comparator: "Is steering better than letting Whisper auto-detect?"
   If B0 = auto-LID, then `cB ≠ cM` on English-detected utterances, and R/P0 geometry must be
   re-derived.
3. §3.7 R2-BC ("R larger for baseline-wrong than baseline-correct English") and §9.3 D1 category
   (a) use generic "baseline-wrong". Restrict both to the confusion stratum (§3).
4. §8 names `docs/inference_cf/CONTRACT.md`, which does not exist. The header folder name is
   stale.

## 9. Findings

**CRITICAL**
- **C-1 (inference validity).** The plan's localizer ("validated token/audio alignment or
  cross-attention timing") resolves to Whisper DTW timestamps, which use all future tokens and
  cannot define a current-step gate. *Disposition*: fixed by revision 1 (causal current-step
  alignment-head localizer). Blocking if unrevised.

**MAJOR**
- **M-1** Padding-template bias: the native LID of the zero-padded template itself gives E=0.94. A
  raw short-window posterior is not acoustic support and biases toward Mandarin false activation.
  → revision 2.
- **M-2** Scope: the gate targets matrix-language confusion (35.0% of B0 errors) plus
  conditionally deletions (49.4%). Same-language substitutions (12.2%) are out of scope. `R`'s
  separation of confusion from correct English is near-tautological. → narrowed claim, split
  strata, headline criterion on E within Han-emitting positions.
- **M-3** Label provenance: P0 strata came from B0_AUTO, not the decoded forced-`zh` B0 (46/300
  transcripts differ). → recompute labels against the exact R2 baseline.
- **M-4** Baseline declaration: forced-`zh` vs auto-LID is unresolved. `cB==cM` holds only for
  forced-`zh`; auto-LID is a stronger baseline that must at least be a comparator. → freeze in the
  R2 spec.
- **M-5** `Q_t` degeneracy: `R` becomes near-maximal from noise-level masses. → revision 4.
- **M-6** Mid-character steps (6.8%) have no language decision to make. → revision 5.

**MINOR**
- **m-1** `V_E` is Latin script, not English (pinyin, names, other Latin languages).
- **m-2** Cost: ~45× baseline encoder FLOPs for local LID. Disclose; exact-window caching only.
- **m-3** Six of 60 panel utterances exceed 30 s: clip windows, mark targets unalignable.
- **m-4** 4 of 57 P0 diagnostic positions were mid-character (historical, no verdict change).
- **m-5** The plan is an untracked, mutable file. The P0/P0-R1 manifests hash a version (sha256
  `f3062e4e…`) that no longer exists anywhere. → commit the revised plan as a versioned file with
  the R2 spec and stop hashing a mutable path.
- **m-6** Plan references a non-existent `CONTRACT.md`; header folder name is stale.
- **m-7** Prior repository evidence of weak attention localization (93% sink frames; `whisper_dtw`
  17.4% invalid): the localizer may fail R2. This is an accepted, measured risk, not a design
  error.

## 10. Minimal required revisions

1. **Localizer (causal).** `x_t` comes from the current-step head-mean attention of the shipped
   `alignment_heads`, restricted to the audio's own frames. No DTW, no cross-token normalization,
   no future tokens. One primary window `W = 1.0 s`.
2. **Template-corrected LocalSupport.**
   `E_t = σ(ℓ_t − ℓ_∅)`, where `ℓ_t = log π_t(E) − log π_t(M)` and `ℓ_∅` is the same log-odds for
   an all-zero 30 s waveform. It is a model/precision constant, measured in-run and recorded
   (CPU fp32 reference: ℓ_∅ = ln(0.3099/0.0199) = 2.745). This turns the posterior odds into a
   likelihood ratio against the model's own no-speech template. It is a constant shift, not a
   tuned temperature. Raw `π_E`, `π_M`, pair mass and raw pairwise E are saved.
3. **Exact epsilon policy.** `R_t = [(P_M − P_E)/(P_M + P_E + 2ε)]_+ ≡ [tanh(C_t/2)]_+`, with
   `C_t = log((P_M+ε)/(P_E+ε))` and ε = 1e-12.
4. **Mass-eligibility fallback.** `Q_t = P_M + P_E`. If `Q_t < ½`, then `R_t` is undefined and
   `g_t = 0` with reason `low_language_mass`.
5. **Structural eligibility.** The token partition as in §5B. Steps whose prefix ends in an
   incomplete UTF-8 sequence are ineligible (`g_t = 0`, reason `mid_character`).
6. **Baseline declaration.** Freeze `cB` = forced-`zh` DG-04 B0 (the setting in which `cB==cM` was
   established) and add B0_AUTO as a required comparator. Any switch to auto-LID is a new versioned
   contract.
7. **Labels / strata.** Use the evaluator-only mapping of §7 on the exact R2 baseline, with the
   five strata of §3. Reconstruct the panel; do not reuse P0 labels or positions.

## 11. Final verdict

| Component | Verdict |
|---|---|
| LocalSupport | `LOCAL_SUPPORT_VALID_WITH_REQUIRED_REVISION` |
| BaselineConflict | `BASELINE_CONFLICT_VALID_WITH_REQUIRED_REVISION` |
| `g = E·R` | `GATE_COMPOSITION_VALID_WITH_REQUIRED_REVISION` |
| **P0-R2 design** | **`R2_DESIGN_READY_WITH_REQUIRED_REVISIONS`** |

The design is mathematically coherent, scientifically motivated for a narrowed target,
reference-free and causal with revision 1, computationally testable, and aligned with a
narrowed paper claim. Whether it *works* is left to R2.

## 12. Contract the P0-R2 feasibility spec should freeze

**Fixed**:

- Whisper-large-v3 (weights sha256 `a8e94b85…`), bf16, eager attention; decoder L24 site for
  direction diagnostics only.
- `cB = cM = [50258,50260,50360,50364]`, `cE = [50258,50259,50360,50364]`.
- Greedy, `max_new_tokens=200`, forced `zh`.
- `D-dev-select` only; no steering, no alpha, no layer choice.

**Per utterance** (one teacher-forced causal replay of the frozen baseline sequence under `cB`):

```text
for each step t (query predicting y_t; prefix y_<t):
  eligible_t   = prefix bytes decode as complete UTF-8          else g_t=0 ("mid_character")
  A_t(f)       = mean_{(l,h) in alignment_heads} attn_{l,h}(query_t -> f),  f < F
                 F = min(1500, ceil(min(duration,30)/0.02)); renormalize over f < F
  c_t          = argmax_f A_t(f)   (ties -> smallest f)          else g_t=0 ("localizer_fail")
  window_t     = [0.02*c_t - 0.5, 0.02*c_t + 0.5] s, shifted inside [0, min(duration,30)];
                 whole seen audio if shorter than 1.0 s
  pi_t         = softmax over the 100 language-token logits at <|startoftranscript|>
                 given encoder(pad_to_30s(x[window_t]))
  l_t          = log pi_t(en) - log pi_t(zh);   l_null = same for all-zero 30 s input
  E_t          = sigmoid(l_t - l_null)
  pB_t         = softmax(logits_t under cB)  (bf16 logits upcast to float32)
  P_M = sum_{V_M} pB_t,  P_E = sum_{V_E} pB_t,  Q_t = P_M + P_E
                 V_M = pure-Han tokens ∪ Han-lead (0xE4–0xE9) fragments
                 V_E = ASCII-Latin word tokens;  all else ambiguous (class masses logged)
  if Q_t < 0.5:  g_t = 0 ("low_language_mass")
  C_t = log((P_M+eps)/(P_E+eps));  R_t = max(0, (P_M-P_E)/(P_M+P_E+2*eps)),  eps = 1e-12
  g_t = E_t * R_t
direction (logged, unused for gating):  d_t = (hE_t - hM_t)/(||hE_t - hM_t|| + 1e-6)  at L24
```

**Saved per row**: token ids/text; `eligible` + reason; `c_t`, `A_t` peak mass and entropy;
window bounds; `π_en`, `π_zh`, pair mass, `ℓ_t`, `ℓ_∅`, raw and corrected E; `P_M`, `P_E`,
per-class ambiguous masses, `Q_t`, `C_t`, `R_t`, `g_t`; `‖hE−hM‖`; evaluator-only `{stratum,
category, alignment status, CTC midpoint}`; cache keys (audio sha, window frames, provider
version, precision, prefix hash, condition).

**Audio control**: same windows (same seconds) cut from the deterministic cyclic-permuted
utterance's audio, and the same R. Report ΔE and the change in AUROC.

**Strata**: `EN-confusion` (primary), `EN-deletion-slot`, `EN-same-language-sub`, `EN-correct`,
`ZH-correct`, `unalignable:*`.

**Predeclared criteria** (recommended values; the spec must freeze them before the run):

| Gate | Criterion |
|---|---|
| **ALIGN** | ≥ 80% of non-boundary English POIs receive a position or gap slot; unalignable counted by reason |
| **LOCALIZER** | on mapped units with CTC times, ≥ 60% have `|0.02·c_t − CTC midpoint| ≤ 0.5 s`; median error reported. **Fail ⇒ `R2-LS` blocked; no window search.** |
| **R2-LS** | finite `E` on ≥ 90% of eligible positions; IQR(E) ≥ 0.10; AUROC(E; English-spoken aligned positions [`EN-correct ∪ EN-confusion`] vs `ZH-correct`) ≥ 0.70; audio control mean \|ΔE\| ≥ 0.05. **Fallback W = 2.0 s** allowed only if, label-free, ≥ 50% of W=1.0 s windows have pair mass below the template's (0.33). |
| **R2-BC** (coverage / sanity, not discovery) | `R ≥ ½` on ≥ 90% of `EN-confusion` and `R ≤ ½` on ≥ 90% of `EN-correct` eligible positions; `Q_t` abstention rate reported per stratum |
| **R2-GATE** (headline) | AUROC(g; `EN-confusion` vs `ZH-correct`) ≥ 0.70 **and** AUROC(g; `EN-confusion` vs `EN-correct`) ≥ 0.70; `ZH-correct` false-activation rate at g ≥ ½ reported, plus Mandarin positions adjacent to English separately; ≥ 30 positions per compared stratum; dialogue-cluster bootstrap intervals; `EN-deletion-slot` reported secondarily |

**Other rules**: one physical GPU job; reruns only for concrete defects; thresholds not weakened
after outcomes. **P1 remains blocked until R2 passes.**
