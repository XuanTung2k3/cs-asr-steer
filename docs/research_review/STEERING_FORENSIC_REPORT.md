# Steering forensic report — inference-time steering for Mandarin–English CS-ASR

**Base:** `7c8401e7` (`audit/a2p-dev200-independent`) · **Branch:** `research/steering-forensics` · 2026-10-11.

**Scope.** Retrospective, CPU-only diagnosis. No GPU job, model forward, training, threshold change, frozen-contract edit or
cleanup action was performed.

**Data use.** References were read only for the already-exposed FULL300 D-dev-select population, through the canonical
`experiments.inference_cf_p2_evaluate.load_references`, and only for offline diagnosis. No D-dev-confirm, D-test,
router-calib, P3 or transfer data was accessed. SRD2-400 and DIR-240 references were not reopened.

**Labels.** **[F]** marks a fact: either recomputed here from sealed artifacts, or taken from an independently audited
record that is named. **[H]** marks a hypothesis.

Companion files: [NEXT_STEERING_DIRECTION.md](NEXT_STEERING_DIRECTION.md), [REPO_CLEANUP_PLAN.md](REPO_CLEANUP_PLAN.md),
[cleanup_manifest.json](cleanup_manifest.json).

---

## 1. Bottom line — why steering failed

1. **The program optimized against the wrong target population [F].** The 1,074 forced-ZH English errors on FULL300 mix
   four different problems:

   | Problem | Errors | Share | What fixes it |
   |---|---|---|---|
   | Whole-utterance language-ID failures (English-dominant utterances, 30 of them) | 310 | 29% | AUTO already fixes these |
   | Scoring and annotation artifacts (`a i` vs `AI`, missing spaces such as `dangerousfrom`, fillers, disfluent spellings) | ≈246 | 23% | Nothing to steer |
   | Tail deletions | 151 | 14% | 139 lie in audio beyond the 30 s window, so nothing to steer |
   | Genuine intra-sentential errors (remainder) | — | — | The real target |

   In Mandarin-matrix utterances (<30 s) the genuine target is small: **45 English runs rendered in Han** plus some genuine
   omissions and wrong English words. The local panels drew mostly from the wrong problems:

   | Panel | From English-dominant utterances | Mid-span (prefix already translated) | Deletions targeted |
   |---|---|---|---|
   | S1 | 32% | 78% | none |
   | P2-RJ-E | 52% | 89% | none |

2. **The English word is decodable by the frozen model, but only along an English-history trajectory [F].** On S1's 60
   confusion positions:

   | Condition | Reference word available |
   |---|---|
   | Same forced-ZH prefix, top-20 under M/E/AUTO prompts | 11/60 |
   | Plain forced-EN free decode, correct at the aligned unit | **43/60** |

   Across all 1,074 forced-ZH errors, forced-EN is correct on **556 (52%)**. A chance control is far lower: 139/1,074
   reference words appear in a *shuffled* utterance's forced-EN text, against 663/1,074 in the matched one. Swapping only
   the prompt does not help. Even the translate prompt, given the Chinese prefix, still emits Han at 676/693 queries
   (DIR-SPRINT0). Candidate *generation* is therefore a conditioning/history problem, not acoustic absence.
   Candidate *selection* is the open problem: raw forced-EN words absent from forced-ZH are spoken English only
   **6.8%** of the time in Mandarin-matrix utterances.
3. **Bounded-energy pulses only flip near-ties, and no tested direction carries word identity [F].**
   - In P2-DIR, all 5 corrections (D2) are the 5 smallest-gap positions of 60 (gaps 0.29–2.62 nats; median 10.6).
     MECH-LANG0 found the same, and this review reproduced it.
   - The ideal L16 lexical gradient at e* gives a median shift of 4.9 nats, against a 10.6-nat gap. Only 27/227 positions
     are bridgeable (P2-RJ-E).
   - The gap is just as large at Mandarin-matrix span onsets: median 10.6 nats, reference log-prob −11.5.
   - What each direction actually encodes:

     | Direction | Encodes |
     |---|---|
     | D0 | Prompt conditioning; random-level cosine with the lexical tangent |
     | D2 | Script class: κ 0.79, but 38/43 top-1 changes go to non-reference tokens |
     | D3 | Acoustically *supported* semantics (171/197 proposals are Han, i.e. translations) |
     | D4, D5 | Lexically inert |

4. **Any token change re-routes the whole continuation, and teacher-forced KL cannot see it [F].**
   - P2-SEQ deletions went from 1,026 to 1,299.
   - U0027_S0_116 was cut from 75 to 31 tokens.
   - U0021_S0_513 translated Mandarin into English.
   - U0086_S0_185 lost "process" and "practical experience".
   - Inducing the AUTO basin with one token works 77% of the time; rescuing back works 2/4 (PATH0).
   - KL on the B0 prefix does not bind once the path diverges (A3/A4, A2P-DEV200: D = 0).
5. **The test-time objectives contained no corrective information for code-switching [F].**
   - AUTO's output is *identical* to forced-ZH on **211/214 Mandarin-matrix utterances**. AUTO-pseudo-label CE (A2,
     TTLS-CE) is self-training on the baseline's own errors there.
   - Their real effects are whole-utterance English switches and premature-EOS recovery: A2-MECH0 F_ZH_EOS 0.965, and
     on NEW200 4 rows give −274 of −281 ZH errors.
   - TTLS-R1R's only genuine phrase (U0086_S0_185, "go back to school") is so near a tie that a rounding change flipped it.

Two secondary facts:
- **BF16 kernel [F].** The historical BF16 NormPreserve kernel is a verified defect at near-zero edits. It is fixed for
  TTLS (R1R), but the frozen DG-02 hook still calls it. Its impact on the direction studies is minor (§4.6).
- **Content step 0 [F].** Content t=0 is never editable: 49/269 wrong English runs (<30 s) start there. This is a
  design limitation of the frozen contract.

## 2. Executable path (verified by code reading)

```text
wav → models/whisper (16 kHz mono, 30 s window, frozen large-v3, BF16) → encoder
prompt [SOT, <|zh|>, transcribe, notimestamps] (absolute positions 0–3)
decoder layer ℓ: residual q → encoder_attn_layer_norm → cross-attn u → site r = q + u (DG-02, pre-FFN)
  edit: return u + (NormPreserve(r + Δ) − r) via lss/sites.py → models/hooks.apply_steering (historical BF16 kernel)
→ FFN → … → logits → suppression → greedy → per-branch KV cache (inference_cf_cached.Branch, explicit cache_position)
eligible queries: absolute ≥ 4 ⇒ content token t=0 (predicted at q=3) is never editable
scoring: evaluation/{canonical,mer,pier}: Han chars + English words, DP alignment; PIER = English POIs, insertions excluded
TTA: inference_cf/episodic_tta (194 decoder LN tensors, AdamW ×2, fresh FP32 masters) · ttls/ttls_r1r (one L16 z vector)
```

Verified properties:
- **Cache isolation [F].** Per-branch caches are isolated and hooks are removed in `finally`.
- **Gradient flow [F].** The site query is not detached in the intervention path.
- **Numerical-noise floor [F].** Two code paths for the same forced-ZH decode differ on **59/300** utterances through BF16
  near-tie flips (P2 report). The decoding is chaotic at near-ties, so any small correction count must be compared with a
  matched-perturbation null.

## 3. Historical approaches

Local results (single next-token pulses at a fixed teacher-forced prefix) and free-decoding results are kept in separate
tables. Populations differ, so rows must not be pooled.

### 3.1 Local / teacher-forced token-level studies

| Study (population) | Mechanism → expected reason | Result | Most plausible failure reason | Status |
|---|---|---|---|---|
| P2-R, P2-RJ, P2-RJ-E (180 → 227 confusions) | D0 = mean(h_EN − h_ZH) at L16. Expected that language conditioning carries English lexical preference. | cos with lexical tangent −0.01 (random 0.022). Ideal direction gives 4.9 nats against a 10.6-nat gap; 12% bridgeable. | D0 is prompt/condition information, not lexical. The gap is ~11 nats. | **F** (audited) |
| P2-DIR D0/D1/D2 (60/60/60) | D1 = dialogue-cross-fitted unique subspace; D2 = script readout gradient. | D1 +0.23 nat. D2 +4.5 nat with 5 corrections, but ZH margin −2.0 and corruption 7.7%. | D2 raises Latin mass as a class. Corrections occur only at gap < 3.3 nats. | **F** |
| P2-SEL, -E, -T, -XA, -LAC (local panel) | Gate D2 by need: LID, ambiguity, cross-attention sensitivity, occlusion. | No gate separates the 5 ZH false positives; XA signal is below BF16 resolution. | The false positives carry strong genuine English-LID evidence. Need ≠ repairability. | **F** |
| ST-LOC0, ST-PROMPT-R1 (60/60/60, L3/8/16/24) | Prompt/unique directions at self/cross sites and doses; L24 expected to have more leverage. | Best arm 1 correction. L24 margin ≤ +1.5 nat. Matched random at L24 also large (nonlinear above η≈0.3). | Direction lacks word identity. Large-dose effects are nonspecific. | **F** |
| S1 (60 confusions, no edit) | Acoustic evidence should expose English candidates. | Union@20 11/60; region masking raises rank 4/4 but gives 0 top-1. | Same-prefix candidate generation fails. Positions are mid-translation (47/60). | **F** |
| R0 (FULL300) | Per-utterance unique direction from predicted EN regions. | Oracle-valid 16–17/300 per layer. | Few decoder queries attend to EN audio in a ZH decode (state starvation). | **F** |
| SRC-CF0-P (20 pairs; 8 arms) | Local audio-mask contrast as an acoustic residual direction. | 0 corrections; max +0.47 nat. | Acoustic contrast is not lexical form; gaps are large. | **F** |
| SRD2-G0 (400 exposed) | D2 + selective gate + permuted control, calibrated native energy. | Ungated 4 corr / 446 ZH harms; gated 1/9; permuted 1/4. 519/606 top-1 changes at confusion queries went to other Han tokens. | Script push with no lexical target; gate no better than shuffle. | **F** |
| DIR-SPRINT0 D2–D5 (240 exposed) | D3 clean/null acoustic prior; D4 transcribe−translate; D5 phone concepts. | D3 proposes Han 171/197; D3/D2 2 corrections each; D4/D5 0; gated arms 0. Direct D3 substitution: 4 corr / 96 ZH harms. | Clean-vs-null likelihood supports *translations*. Task/phone axes are lexically orthogonal. | **F** |

### 3.2 Free-decoding (full-ASR) studies

| Study (population) | Mechanism | Result | Most plausible failure reason | Status |
|---|---|---|---|---|
| P2 (FULL300) | E·R_B-gated D0 at L16 α2. | 2 corr / 0 corrupt vs matched. AUTO PIER 0.391 vs forced-ZH 0.474; P2-B controls do not support the mechanism. | Effect within noise of a lexically void direction. | **F** |
| P2-SEQ (fixed100) | Gated D2 local pulses with free continuation. | 20 corr / 14 corrupt. Outside-POI harm 244/4,055. Deletions 1,026 → 1,299. | Local flips re-route continuations: EOS, mode switches, synonym drift. | **F** |
| PATH0–PATH5 (fixed100, NEW200) | Branch/path control, consensus token, EOS protection. | Induction easy, rescue rare. Gains confined to U0023_S0_664 and the enriched panel. NEW200: 0 rescues / 2 harms. | Sequence basins. Gains concentrated in one utterance. | **F** |
| A2 / A3 / A4 TTA | AUTO-CE on 194 LN tensors; soft-AUTO KL; script-preserving teacher. | A2 fixed100: 31/2 corrections, 15 genuine units in one utterance, 26 new ZH. A3/A4: teacher-forced safe but free decoding unsafe. | AUTO = forced-ZH on 211/214 Mandarin-matrix rows. Gains are EOS recovery (F_ZH_EOS 0.965). | **F** |
| TTLS-R1 → R1R (fixed100) | One L16 vector; AUTO CE ± stable-prefix KL; acoustic candidate. | R1 confounded by BF16 zero-edit (6/100 sequences). R1R: T1 has 4 genuine units, one phrase; AC = direct substitution 100/100. | Same uninformative objective, global actuator, near-tie success. | **F** |
| A2+P (fixed100 → NEW200) | A2 + KL on stable B0 positions. | fixed100 26 → 3 new ZH (20 from one row). NEW200: 9 = 9 (D = 0). | KL on B0 path non-binding off-path; fixed100 effect was one utterance. | **F** |
| Core v6 DG03–DG08 (trained) | Contrastive basis + trained controller + damage-aware CE/KL. | Trained positive historical results (see CODE_MAP). | Outside the training-free target. Not re-audited here. | not re-verified |

## 4. New forensic analyses (CPU, sealed artifacts, FULL300)

Inputs:
- `results/inference_cf/p2_A_r1_L16/rows/*.json`: matched B0M_L16 forced-ZH, B0_AUTO, B1 forced-EN free decodes.
- P2-RJ, P2-RJ-E and P2-DIR position records.
- `results/inference_cf/s1/analysis.json`, P2-SEQ rows, and the P2-A2-MECH0 panel.

Scoring uses the canonical `evaluate_pois` and `align_tokens`.

### 4.1 Composition of the forced-ZH English error mass (1,074 errors / 2,268 POIs)

Diagnostic taxonomy, first matching class (not a frozen metric):

| Class | All | Mandarin-matrix (<0.2 EN) | Mixed (0.2–0.5) | English-dominant (≥0.5) | forced-EN recovers (all) | AUTO recovers (all) |
|---|---:|---:|---:|---:|---:|---:|
| translation (Han substitution) | 334 | 79 | 86 | 169 | 252 | 133 |
| internal deletion | 280 | 168 | 71 | 41 | 122 | 15 |
| tail deletion (139 in >30 s audio) | 151 | 77 | 29 | 45 | 35 | 17 |
| boundary join (`dangerousfrom`) | 112 | 65 | 38 | 9 | 63 | 4 |
| acronym letter (`a i` vs `AI`) | 90 | 49 | 15 | 26 | 41 | 16 |
| wrong English word | 63 | 40 | 16 | 7 | 12 | 1 |
| filler (`okay`, `yeah`) | 44 | 20 | 11 | 13 | 31 | 6 |
| **total** | **1,074** | **498** | **266** | **310** | **556** | **192** |

What the table shows:
- **AUTO adds nothing in Mandarin-matrix utterances.** It recovers 3 of their 498 errors, and none outside forced-EN's set.
  Its PIER advantage comes from the 30 English-dominant utterances.
- **The span-level view shrinks the genuine target further** (Mandarin-matrix, <30 s; 161 wrong English runs):

  | Run rendering in forced-ZH | Runs | Share | Note |
  |---|---|---|---|
  | Rendered in Han | 45 | 28% | Translations; forced-EN recovers 28 |
  | Partial Latin | 66 | 41% | Mostly spacing, acronym, inflection or annotation |
  | True omission | 38 | 24% | Many fillers and disfluent repeats |

  The genuine lexical target is roughly 45–90 runs across 214 utterances and 20 dialogues. This explains why every study
  observed 1–5 corrections.
- **Reference noise exists [F].** Examples: `socisaedy` (U0023_S0_671), `link link king` for LinkedIn (U0028_S0_120),
  `hash tax` for hashtags (U0027_S0_116).

### 4.2 Candidate availability (bottleneck A)

| Same 60 S1 confusion positions | Available |
|---|---:|
| reference first token in top-20, same forced-ZH prefix, M / E / AUTO prompts (union) | 11 |
| — of which span-initial / mid-span | 4/13 / 7/47 |
| forced-EN whole-sequence free decode correct at the aligned unit | **43** |
| — of these, not first-token accessible | 38 |
| AUTO free decode correct | 16 |
| word appears anywhere in forced-EN text | 50 |
| reference word needs >1 BPE (leading-space form) | 3 |

- **BPE is not the bottleneck:** 57/60 targets are single tokens.
- **Deletions were never targeted.** By construction, "EN-confusion" means substitution only (`SRD2_G0_SPEC` §evaluation),
  and deletions are 50% of errors.
- **The candidate source decides success.** TTLS-AC and D3 drew candidates from the same Han prefix and failed; the
  whole-sequence forced-EN decode succeeds on most of the same units.
- **Selection is unsolved.** In Mandarin-matrix utterances, forced-EN words absent from forced-ZH are mostly translations
  (precision 6.8%, content words 7.5%). English-dominant: 0.81; mixed: 0.40.

### 4.3 What directions encode and why they miss (bottleneck B)

**Direction geometry and response:**
- **Linear response is confirmed:** observed Δm tracks g·edit with r 0.92–0.95 (MECH-LANG0).
- **Only D2 aligns with the lexical tangent** (cos 0.79). D0 −0.009 and v_AC +0.016 sit at the random level of 0.022.
- **D0 is condition information.** It has a strong common component (resultant 0.64 at L16), meaning near-constant
  prompt conditioning.
- **D2 is query-specific** (resultant 0.18) but **class-level:** the reference rank improves on 31/60 rows and 38/43 top-1
  changes go to non-reference tokens (MECH-LANG0).

**Who gets corrected** (P2-DIR per-position records):

| utterance | t | reference | pre-edit gap (nats) | span-initial |
|---|---:|---|---:|---|
| U1080_S0_181 | 26 | is | 0.29 | no |
| U1004_S0_310 | 9 | team | 1.38 | no |
| U1063_S0_17 | 43 | lay | 1.46 | no |
| U0021_S0_513 | 1 | do | 2.61 | no |
| U1002_S0_488 | 37 | getting | 2.62 | yes |

- **Corrections are predicted by the gap, not the direction.** The 10th percentile of the gap is 2.97 nats and the median
  10.6. On this shared 60-position panel, no arm of P2-DIR, ST-LOC0, ST-PROMPT-R1 or SRC-CF0 corrected a confusion with
  a gap above 3.3 nats (MECH-LANG0 §C). Gaps were not recomputed for the SRD2-400 or DIR-240 populations.
- **Why D3 fails [F, mechanism H].** Clean-vs-null acoustic likelihood rewards tokens that are semantically licensed by
  the audio. Whisper's forced-ZH decoder renders English speech as its Chinese translation, so the translation is
  "acoustically supported": 171/197 D3 proposals at English-confusion queries are Han.

### 4.4 Site and timing (bottleneck C)

- **Information, not site, is the dominant deficit [F].** At matched L16 energy:

  | Direction | Gap closed | Corrections |
  |---|---|---|
  | Generic directions | ≤ 5% | 0 |
  | D2 | 27% | 5 |
  | Ideal lexical gradient (offline) | ~50% (4.9 of 10.6 nats) | — |

  Leverage is a secondary ceiling.
- **L24 is unresolved [F].** L24 gives larger margins for prompt directions (+1.48 nat), but matched random grows too.
  L24 lexical alignment was never measured, so it is unknown, not zero.
- **The lock is the Han history, not just a translated prefix [F].** At the 17 Mandarin-matrix span-onset positions in
  P2-RJ-E, the prefix is correct Mandarin, yet the reference log-prob is still −11.5 (median gap 10.6). [H] The
  English surface form is not readable from a Han-history state at any tested site.
- **Content step 0 [F].** 49/269 wrong English runs (<30 s) begin at t=0 (13/43 in English-dominant utterances), which
  the frozen forced-prefix exclusion makes unreachable. PATH0 shows the first divergent token decides the basin.

### 4.5 Sequence damage (bottleneck D) — raw P2-SEQ pairs

- **Scale.** 41/100 transcripts changed; 90/237 edits flipped the token at the edited step.
- **Edit size.** 40/41 changed transcripts first diverge at a full-size edit (norm ≥ 0.05). 29/237 edits were below BF16
  resolution, and at most one transcript change is attributable to them.

| Utterance | Damage |
|---|---|
| U0027_S0_116 | Premature EOS: 75 → 31 tokens |
| U0021_S0_513 | Mode flip to translation: Mandarin rendered as English |
| U0086_S0_185 | Reverse mode flip: correct English ("process", "practical experience") rendered as Han |
| U0011_S0_503 | Synonym drift: 社会技巧 → 社交技能 (still Han) |

- **Why KL preservation fails [F].** It constrains distributions only along the protected B0 prefix. A2P-DEV200 shows
  identical new-ZH indices with and without P (9 = 9), and A3/A4 kept the teacher-forced anchor while free decoding stayed
  unsafe.

### 4.6 Implementation defect check (BF16 NormPreserve)

- `models/hooks.apply_steering` computes `steered / (new_norm + EPS) * orig_norm` in the input dtype (`hooks.py:110`). It
  is called by the frozen DG-02 hook (`lss/sites.py:39,252,502`).
- **Synthetic CPU check** (state norm 6.6, D = 1280, BF16):

  | Intended edit norm | Realized change vs intent |
  |---|---|
  | 0 | Noise floor of norm 0.0096 on every row |
  | ≤ 0.01 | Dominated by rounding; cosine with the intended edit ≤ 0.52 |
  | ≥ 0.1 | Accurate (cosine ≥ 0.99) |

- **Consequences.** The TTLS zero-edit confound is real (fixed by the versioned R1R ratio-first kernel). Any gated method
  with small nonzero gains is partly a noise injector.
- **Classification:** VERIFIED LATENT DEFECT. Primary for TTLS-R1, minor for the direction studies. Not changed here:
  DG-02 is frozen.

### 4.7 Test-time objectives (bottleneck E)

| Stratum | AUTO text == forced-ZH | AUTO == forced-EN |
|---|---:|---:|
| Mandarin-matrix (214) | **211** | 1 |
| mixed (56) | 46 | 10 |
| English-dominant (30) | 7 | 17 |

- **AUTO-CE cannot correct code-switching.** On Mandarin-matrix utterances the pseudo-label is the baseline itself.
- **What the AUTO-CE methods actually did:**
  - adopted whole-utterance English decodes (A2's NEW200 genuine corrections: 3 utterances, all AUTO copies);
  - extended premature EOS (A2-MECH0: 12 EOS-recovery rows in 10 dialogues carry 300/311 of the ZH rescue).
- **The one genuine TTLS phrase went against its own objective [F].** For U0086_S0_185 the AUTO pseudo-label *equals* B0
  and contains the error 回到学校. CE toward that target cannot carry "go back to school", so the T1 correction ran
  against its objective: a near-tie side effect. The same holds for TTLS-AC's 烤烧→cooking (U1034_S0_53, AUTO = B0); there
  the correct word appears in the whole forced-EN decode ("and cooking at home").
- **The acoustic pseudo-target also fails.** TTLS-AC (same-prefix forced-EN argmax + clean/null ratio) abstains 94% and
  equals direct substitution 100/100.
- **The actuator is too global [H].** A single z or global LN update cannot express position-specific lexical changes.
  This is secondary to the missing information.
- **Premature EOS is broader than A2's rows [F].** Some 30 s utterances end after 5–12 units (U0039_S0_80, U1001_S0_388,
  U0040_S0_141, U1003_S0_119). A2 recovers only some of them, and a whole-utterance token rate ranks only 2/12 EOS
  rows near the top.

## 5. Successes vs failures — what separates them

| Kind | Cases | Separating property |
|---|---|---|
| Local corrections | 5 D2 positions in §4.3; ST-LOC/ST-PROMPT singletons inside the same six near-boundary rows | Pre-edit gap < 3.3 nats (near-tie); not direction specificity |
| TTLS genuine phrase | U0086_S0_185 "go back to school" (B0 "回到学校"; forced-EN correct) | Near-tie phrase decision; R1R's rounding change alone flipped it. The pseudo-label pointed at the error (AUTO = B0) |
| A2 genuine | NEW200: 3 whole-utterance English switches copying AUTO | English-dominant utterance; AUTO already right |
| A2+P fixed100 "protection" | U0023_S0_664 (20 of 23 prevented errors) | English speech: forced-ZH translates the whole utterance and AUTO = forced-EN is right. Single-utterance effect; absent on NEW200 |
| Typical failures | U0011_S0_265 (English-dominant, translated: my→有, to→去); U1081_S0_65 watermelon (E-branch rank 1,045); U0011_S0_503 social skills→社會技巧 | Gap ≈ 10–11 nats under Han history, yet the forced-EN free decode is correct |

## 6. Confirmed issues vs hypotheses

| Classification | Item |
|---|---|
| VERIFIED IMPLEMENTATION DEFECT | BF16 divide-then-multiply NormPreserve at near-zero edits (§4.6). Fixed only in versioned R1R; DG-02 hook unchanged. |
| VERIFIED DESIGN LIMITATION | Content t=0 not editable (frozen forced-prefix exclusion). |
| VERIFIED DESIGN LIMITATION | Confusion panels exclude deletions and select mid-span positions. Targets inherit a wrong, translated prefix. |
| VERIFIED DESIGN LIMITATION | AUTO pseudo-labels = baseline on 211/214 Mandarin-matrix rows. |
| VERIFIED EVALUATION ISSUE | Error pools mix utterance-level LID, artifacts and >30 s truncation with genuine CS errors. Some references are noisy. |
| EMPIRICALLY SUPPORTED | Lexical form is reachable along English-history trajectories (43/60, 556/1,074) but not from Han-history states (gap ≈ 11 nats). |
| EMPIRICALLY SUPPORTED | Bounded single-site pulses correct only near-ties. Generic directions carry no word identity. |
| EMPIRICALLY SUPPORTED | Teacher-forced KL/gating does not bound free-decoding damage. |
| HYPOTHESIS | An input-specific English-history state, transplanted at a time-aligned onset, can deliver the word where generic directions could not. |
| HYPOTHESIS | Reference-free acoustic localization can lift forced-EN candidate precision from 7% to a usable level. |
| UNKNOWN | L24 lexical alignment. Encoder-site steering was not tested in the inference program. Safety on fresh dialogues. |

## 7. Limits and reproduction

- **Population.** All quantities come from exposed development data (FULL300 = fixed100 + NEW200; 20 dialogues). They
  are diagnostic, not confirmatory.
- **Taxonomy.** The §4.1 taxonomy and span classes are heuristic, written for this review, and not frozen.
- **Coverage.** This is not a line-by-line audit of every legacy file. GPU-side tensors were not replayed.
- **Reproduction.** The scripts (≈10 short Python files) live in the session scratchpad. Their exact inputs and
  definitions are stated above.

  | Analysis | Script | Input |
  |---|---|---|
  | Availability | `a_avail` | p2_A_r1_L16 |
  | S1 join + chance control | `b_s1join` | S1 analysis rows ↔ `p2rj/positions.json` ↔ `p0_r2/evaluation_units.json` |
  | Strata, deletions | `c_runs`, `d_del` | — |
  | Taxonomy | `e_taxonomy` | — |
  | Candidate precision + AUTO identity | `g_pseudo_cand` | — |
  | Gaps | `h_gap` | p2rje rows |
  | Kept words | `i_kept` | — |
  | Spans | `j_spans` | — |
  | BF16 | `k_bf16` | synthetic |
  | EOS rate | `l_eos` | — |

  They are not committed, because outputs are reference-derived and the deliverable set was fixed to four files. They can
  be added under `experiments/` on request.
- **Prior draft.** An untracked earlier draft of this file existed in the worktree; this report supersedes it. A copy is
  kept outside the repo.
