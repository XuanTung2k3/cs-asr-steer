# DIR-SPRINT0 — three-family steering-direction discovery (frozen specification)

Status: **DESIGN_FROZEN**. Machine-readable authority: `configs/inference_cf/dir_sprint0.json`; any implementation
discrepancy blocks execution. Core proposal v6, METHOD_CONTRACT and every historical verdict are unchanged.

Feasibility history:
- `DIR_SPRINT0_D5_PROVIDER_BLOCKED` (72727f34).
- `DIR_SPRINT0_D5_PROVIDER_READY` (930bca0d). Audit `DIR_SPRINT0_D5_PROVIDER_AUDIT: PASS` (28/28).

## 1. Question and limits

**Question.** Does any of three new direction families produce actual English first-token corrections? It must
- beat matched D2 and random controls;
- keep Mandarin, English and EOS safety acceptable;
- do so at the frozen L16 DG-02 site and realized chord e* = 1.1260757575454359.

**The three families.**
- **D3 acoustic-prior contrast:** original audio versus null audio.
- **D4 transcription-fidelity contrast:** transcribe versus translate.
- **D5 phonetic-concept steering:** an external frozen phone model, frozen articulatory concepts and an unlabeled
  calibration bank.

The main outcome is **actual English first-token correction**, not script mass or logit movement.

**Design constraints.**
- Single-query pulses from the pristine baseline-generated query. Persistent history advances with B0 tokens only.
- One positive orientation per new family.
- No sign, dose, layer, threshold or gate search, and no new gate design.
- No weight training, test-time optimizer or full autoregressive steered decoding.

**If no family shows credible promise, DIR-SPRINT0 closes.** No second direction-discovery programme is automatic.

## 2. Population (frozen before any model inference)

`docs/inference_cf/DIR_SPRINT0_POPULATION.json` (selector `experiments/inference_cf_dir_sprint0_population.py`,
identity only):

| Item | Value |
|---|---|
| Role roster | 7,919 D-dev-select utterances, 20 dialogues (six metadata columns only) |
| Exclusion | `exposure_registry.exposed_ids()`, hash-checked: documented FULL300 registry (300) + A1 SRD2-G0 (400) = 700; plus missing audio |
| Eligible | 7,219 (min 92 per dialogue) |
| Rule | per sorted dialogue, sort eligible UIDs by `SHA256("DIR-SPRINT0-population-v1|240924|UID")`, then UID; take 12 |
| Selected | 240 = 12 × 20; 2,851.4 s audio; 15 longer than 30 s; all mono 16 kHz |
| Selected-ID hash | `sha256:5b62f378fa0ba671723a85dd4836b21ba785a71dab8067ada4d08cc28c3371c2` |
| Roster hash | recorded in the config (`population.roster_hash`) |
| Calibration bank (D5 only) | ALL FULL300 documented-exposure utterances (300, 5,269 s), unlabeled, no correctness filtering; disjoint from the 240 |

**Selection inputs.** Only identity, role, dialogue, audio existence and audio format. No correctness, language,
reference, baseline prediction, phonetic score or gate value.

**What these utterances are.** New development utterances from the same 20 previously used dialogues. They are not
independent confirmation.

**Roles never touched.** D-dev-confirm, D-test, router-calib, P3 and every transfer corpus.

## 3. Model, baseline and queries (unchanged SRD2-G0 contract)

**Model and decoding.**
- Frozen Whisper-large-v3, with the pinned model, tokenizer and preprocessing files.
- BF16, eager attention, eval mode, no gradients on parameters.
- Forced-ZH transcribe prompt `[50258,50260,50360,50364]`; greedy with the explicit lowest-ID processed argmax; at most
  200 new tokens.

**Query inventory.**
- Queries t = 0..T−1, plus t = T only if B0 stopped on EOS. The absolute query index is 4+t−1.
- Structural eligibility: t ≥ 1, a complete UTF-8 prefix, no special content token, and a finite native site and raw
  logits.
- Every other query is a recorded no-op on every arm.

**R2 gate.** The original R2 `g_old = E·R_B` is used unchanged:
- full-replay frozen ten-head window;
- raw float32 script mass;
- native LID on the crop, with the recomputed null;
- the original fallbacks.

**Compatibility.** The SRD2-G0 full-replay versus cached compatibility bounds are frozen identically.

## 4. Branches (same audio, same B0 content prefix, separate caches)

| Branch | Prompt / audio | Use |
|---|---|---|
| B0 | transcribe `[50258,50260,50360,50364]`, original audio | baseline; the only persistent history |
| E | `[50258,50259,50360,50364]` | D0 only |
| TL | `[50258,50260,50359,50364]` (`<|translate|>` = `task_to_id.translate`) | D4 only |
| NULL | B0 prompt, `np.zeros(30·16000)` through the identical feature extractor / encoder | D3 only |

**Identity requirements for every branch:**
- its own `DiagBranch` and cache;
- fed tokens == prompt + content[:t];
- the same absolute query index.

Branch identity and finiteness must be 1.0 across Job A.

## 5. New families

### D3 — acoustic-prior contrast (frozen candidate rule)

**Log-probabilities.** l_B and l_N are float32 `log_softmax` values of the generation-processed B0 and NULL logits
(suppress_tokens → −inf), upcast to float64.

**Contrast and score.**
- A(v) = l_B(v) − max(l_N(v), log 1e-12). The floor applies after the log-softmax.
- S(v) = l_B(v) + 1.0·A(v).

**Legal set.** Content IDs below EOS that are not suppressed, not special, and UTF-8-boundary legal: the first byte is
not a continuation byte, and the bytes form a valid sequence or a valid prefix.
- Frozen: 49,695 tokens, hash `sha256:bfcdc254…`.
- Excluded: 82 suppressed, 466 continuation-start, 13 invalid, 1 empty.
- Per query, legal additionally requires finite l_B and l_N.

**Plausibility.** l_B(v) ≥ max_w l_B(w) + log(1e-3). The maximum is taken over the full processed vocabulary.
- Why 1e-3: log 1e-3 = −6.91 nats covers the first-order reach of an e* edit. The historical P2-DIR value is
  A = ‖g_tan‖e* = 5.72 [5.18, 6.30] nats.
- The window adapts to entropy.
- It is not S1's fixed Top-20 (which missed most English-confusion targets), and it is not tuned on any D3 outcome.

**Candidate.** c_AP = argmax over the plausible legal set of S, lowest token ID on exact ties. EOS may be b_t but is
never c_AP.

**Abstentions:**
- `nonfinite_branch` and `no_plausible_legal_candidate` → no-edit;
- c_AP == b_t → `baseline_is_candidate`, an exact no-edit with no gradient call.

**Direction.**
- J = z[c_AP] − z[b_t] on the scratch query's float32 raw logits.
- One autograd call through the historical zero-probe scratch mechanics (`readout.ZeroProbe`, `clone_scratch`,
  `cache_fingerprint`).
- d_AP = `readout.tangent_unit` of the gradient. Sign +.

**Comparator D3CD.** Direct contrastive token selection: top-1 := c_AP where D3 status is `candidate`, else B0. It is a
decision only, never a pulse, and not energy-matched.

### D4 — transcription-fidelity contrast

**Direction.** d_TF = `readout.tangent_unit(h_TR − h_TL, h_TR)`, with h_TR the B0 L16 DG-02 site and h_TL the TL-branch
site at the same query. Sign: transcribe minus translate, never reversed.

**Entanglement disclosure.** `<|translate|>` means X→English, so TR−TL partly encodes "do not output English". It is
reported, not resolved by choosing a sign.

### D5 — phonetic-concept steering

**Provider.** `facebook/wav2vec2-xlsr-53-espeak-cv-ft` at revision 2c733782.
- Manifest digest `sha256:93ae874c…`; feature table digest `sha256:8e1735af…`.
- Run once per utterance on the full original waveform: CPU float32, 4 threads, never on crops.
- **Job A apparatus.** Each child process first reruns `sample1.flac` and `aishell_example_mandarin.wav`. Requirements:
  - an in-job repeat must be bitwise identical;
  - the output must be bitwise equal to the sealed engineering hash, or within max |Δlog p| ≤ 1e-3 with argmax
    agreement 1.0.

**Concepts.** Frozen concept table `results/inference_cf/dir_sprint0/design/d5_concept_table.json` (digest
`sha256:cbd856f3…`): for each provider symbol, the fraction of its panphon segments in each class.

| Concept | Segment rule (panphon +1/0/−1) |
|---|---|
| NAS | nas=+1 |
| STOP | son=−1 ∧ cont=−1 (stops and affricates) |
| FRIC | son=−1 ∧ cont=+1 |
| LAB | cons=+1 ∧ lab=+1 |
| COR | cons=+1 ∧ cor=+1 |
| DOR | cons=+1 ∧ cor=−1 ∧ lab=−1 ∧ hi=+1 |

- **Excluded from the primary concepts (descriptive only):**
  - voiced obstruents (son=−1 ∧ voi=+1);
  - aspiration (sg=+1).

  espeak's English and Mandarin conventions confound these with language: English p/t/k are never marked aspirated,
  and Mandarin b/d/g are written p/t/k.
- **Outside every primary class:** panphon codes glottals h/ɦ as [+son, −hi], so they fall in none. Uvulars (−hi) are
  likewise excluded, and neither EN nor ZH espeak uses them.
- **Tone** is recorded descriptively and never converted.

**Local evidence at query t.** The window is the R2 window of the query: the gate's own full-replay ten-head window,
[s0, s1).
- Weights: w_j = |[320j, 320j+400) ∩ [s0, s1)| / 400 (`phone_provider.interval_weights`).
- Emitting frame: s_j = Σ_{segmental v} p_jv ≥ 0.5.
- Evidence: q_k = Σ_j w_j e_j Σ_v p_jv C_vk / Σ_j w_j e_j s_j, the fraction of emitting phone mass in class k.

**Evidence validity.** The first failing rule abstains; nothing is ever imputed.

| Rule | Abstention |
|---|---|
| No R2 window | `no_window` |
| Provider failure for the utterance | `provider_failure` |
| Fewer than 20 positive-weight frames | `too_few_window_frames` |
| Nonfinite posteriors | `nonfinite_posteriors` |
| Emitting weight E < 2.0 | `low_emitting_weight` |
| Excluded-symbol share > 0.1 of (segmental + excluded) emitting mass | `excluded_mass` |
| Zero valid mass | `zero_valid_mass` |

- **Silence:** silent windows have no emitting frame.
- **Justification** (`d5_threshold_engineering.json`; public clips and synthetic signals only):
  - Per-frame segmental mass is bimodal. Speech emission frames are at ≥ 0.98, and only 2.7–5.1% of frames lie in
    (0.2, 0.8).
  - Silence and noise frames never exceed 0.13.
  - In 1-second speech windows the emitting-weight 5th percentile is 2.6–8.2; silence is 0.
  - A 1-second window holds 50–51 frames, so 20 frames = 0.4 s.

**Calibration prototypes** (FULL300, all utterances; canonical B0 regenerated in Job A).
- **Inputs:** every structural bank query with valid evidence.
- **States:** unit-normalized L16 DG-02 states h/‖h‖.
- **Contrast:** High_k = {q_k ≥ Q_0.8} and Low_k = {q_k ≤ Q_0.2}, using pooled numpy linear quantiles and requiring
  Q_0.8 > Q_0.2.
- **Dialogue balancing:**
  - a dialogue enters a concept only with ≥ 5 High and ≥ 5 Low queries;
  - Δ_d = mean_High,d − mean_Low,d;
  - v_k = normalize(mean_d Δ_d), giving each dialogue equal weight.
- **Coverage requirements:** ≥ 15 dialogues, and ≥ 500 High and ≥ 500 Low queries overall.
- **Normalization:** μ_k is the dialogue-macro mean, and s_k is the pooled population SD, which must exceed 1e-6.
  No low-rank normalization is used.
- **Labels:** none are used. Any failing concept makes D5 `PROVIDER_OR_CONSTRUCTION_BLOCKED`.

**Diagnostics (descriptive only).**
- Position Spearman correlations.
- Concept means by the class (Latin, Han, EOS) of the baseline token b_t.
- Dialogue η².
- Prototype cosines, and cosine with the bank's baseline-token script axis.

**Direction.**
- z_k = (q_k − μ_k)/s_k.
- d_PHON = `readout.tangent_unit(Σ_k z_k v_k, h_B)`; abstain if ‖z‖ < 1e-6. Sign +.
- D5 is not Whisper-only: it needs the external phone model, the frozen tables and the FULL300 bank.

**Shuffled control (D5SH, D5SHG).** A within-utterance derangement over the D5-valid queries.
- Ordering: sort by (`SHA256("DIR-SPRINT0-d5-shuffle-v1|240924|UID|t")`, t).
- Assignment: position i receives the z vector of position (i+1) mod n. There is no fixed point when n ≥ 2.
- What stays fixed: the prototypes, normalization, tangent rule, the recipient's own h_B, and e*.
- Singletons are flagged and still pulsed. Degenerate shuffles (identical z) and cos(d_D5, d_SH) > 0.9 are reported.
- No oracle or reference enters.

## 6. Historical controls (common-query, same site and chord)

| Arm | Construction |
|---|---|
| D0 | `core_p1.direction(h_E, h_B)`, sign + (historical P2-DIR D0) |
| D1 | sealed NEW_P2_DIR_CROSSFIT_V1 fold vector of the utterance's dialogue (fold d fitted without d), sign +, no refit |
| D2 | unchanged `ReadoutDirection` (J = log P_E − log P_M), sign + |
| VAC | v_AC when constructible: <br>• R0 window-grid LID track (unchanged R0 rule); <br>• S1 association on the cached B0 query attention (≥ 0.10 association, ≥ 0.5 heard mass); <br>• hard-zero mask; cold forced-ZH masked replay; <br>• `src_cf0_pilot.direction(h_clean, h_mask)`, sign + |
| RND | isotropic PCG64 unit vector per UID/t/layer (tag `DIR-SPRINT0-random-v1`) |

Previous aggregates on other populations are never substituted for these common-query controls.

## 7. Arm matrix (frozen)

**Pulse arms (15).**

| Group | Arms | Target |
|---|---|---|
| Ungated | D0, D1, D2, VAC, RND, D3, D4, D5, D5SH | e* |
| R2-gated | D2G, D3G, D4G, D5G, D5SHG, RNDG | e*·g_old, solved as its own native chord |

**Non-pulse comparators:** B0 NONE and D3CD.

**Energy execution** uses the unchanged SRD2-G0 dose machinery:
- historical NormPreserve;
- `p2r.solve_scale`, with at most 8 native-BF16 emulations;
- native consumption preview, cached one-solve hook, and bitwise FFN-input equals preview;
- guards: ≤ 2% relative squared energy, ≤ 2% chord, norm ratio ≤ 0.005.

**No-edit cases:** a zero target, an invalid direction and an unreachable target each give an explicit exact no-edit.

**Energy validity:**
- ungated arms: matched fraction ≥ 0.95 of valid-direction nonzero requests;
- gated arms: lost planned squared energy ≤ 0.10;
- if D2 or RND fails, the experiment is INVALID;
- if a new-family variant fails, that variant is ineligible.

## 8. Jobs (at most two scientific GPU jobs, each ≤ 3 h)

**Job A (no pulse, no readout autograd, no reference).**
- B0 inventory with the teacher-replay bitwise check, the R2 full replay and gate, compatibility, and probes.
- E, TL and NULL branches.
- D3 decisions.
- v_AC track, targets and masked states.
- Bank B0 states and R2 windows.
- Two concurrent CPU phone-provider children.
- The deterministic CPU construct stage: D5 evidence, prototypes, and every non-autograd direction.

**After Job A.** Seal and push; then the independent `DIR_SPRINT0_AUDIT_A: PASS`. Then the complete planned-pulse recost
(both forecasts ≤ 9,000 s, else `DIR_SPRINT0_COMPUTE_BLOCKED`) and the minimal global authorization.

**Job B.**
- Old30 historical apparatus.
- Per structural query, from the pristine pre-step cache: D2 and D3 autograd, the clean step (bitwise against Job A),
  then the 15 arms, restore, and advance with the B0 token.
- Seal and push. `DIR_SPRINT0_AUDIT_PRIMARY: PASS` is required before references.

**No third job, subset, selective rerun or dose optimization.**

## 9. Evaluation (post-seal, evaluator process only)

**Mapping.** The unchanged SRD2-G0 mapping: canonical normalize/segment/align, R2 `unit_to_token_positions`, P2-R
`target_set`, and the singleton baseline target for Mandarin. Long audio (> 30 s) is primary-unalignable.

**Events.**
- Correction: B0 top-1 ∉ Y and arm top-1 ∈ Y (EN-confusion).
- Corruption: B0 top-1 ∈ Y and arm top-1 ∉ Y (EN-correct, ZH-correct).
- Utility: U = C − H_ZH − H_EN.

**Severe EOS.** Arm top-1 = EOS, B0 top-1 ≠ EOS, and T − t ≥ 10.

**Mechanistic gap.**
- G = max_{k∉Y} zp_k − max_{k∈Y} zp_k on processed logits; Δ = G_B0 − G_arm.
- The mechanistic statistic is the dialogue-macro mean of Δ_v − Δ_RND(matched) over mapped structural EN-confusion
  queries.

**Reporting.** Every arm, on full-population, structural, constructible, edited and common-eligible denominators, with
per-dialogue tallies.

## 10. Frozen predicates

| Block | Predicate |
|---|---|
| Opportunity | mapped EN-confusion ≥ 150 in ≥ 10 dialogues; EN-correct ≥ 50 in ≥ 10; ZH-correct ≥ 500 in ≥ 15; English mapping ≥ 0.8; Job A coverage |
| Coverage (family) | constructible fraction ≥ 0.5 of structural queries **and** ≥ 0.5 of mapped structural EN-confusion queries |
| Power (variant) | C ≥ 8; correction dialogues ≥ 4; max dialogue share ≤ 0.5 |
| Advantage (variant) | C − C_D2(matched) ≥ 4; C − C_RND(matched) ≥ 4; ≥ 3 dialogues with positive paired gain over D2. D5: also C − C_SH ≥ 4 and ≥ 3 positive-gain dialogues versus the shuffle. D3: also U − U_D3CD ≥ 1 |
| Safety estimability | actively edited ZH-correct ≥ 40 in ≥ 8 dialogues |
| Safety | U ≥ 1; unconditional ZH corruption ≤ 0.01; active ZH corruption ≤ 0.05; H_EN ≤ 3 and ≤ 0.02; severe EOS = 0; no dialogue with ≥ 3 ZH corruptions and active rate > 0.20 |
| Mechanistic | point ≥ 0.5 nat and Bonferroni (m = 6) lower bound > 0 |

**Matched comparators:**
- D3, D4, D5 against D2 and RND;
- the gated variants against D2G and RNDG;
- D5 against D5SH, and D5G against D5SHG.

**Family status.**
- **Validity:** PROVIDER_OR_CONSTRUCTION_BLOCKED, INSUFFICIENT_COVERAGE or VALID_AND_TESTED.
- **Outcome**, first match over the ungated then the gated variant:

  | Outcome | Condition |
  |---|---|
  | PROMISING_FEASIBILITY | energy, power, advantage, estimability and safety all pass |
  | SAFETY_FAILED | energy, power, advantage and estimability pass; safety fails |
  | INSUFFICIENT_COVERAGE | energy, power and advantage pass; estimability fails |
  | LEXICAL_POWER_INSUFFICIENT | otherwise |

A scientific negative is not a provider failure. A non-constructible family is never credited with zero harm.

## 11. Statistics

**Bootstrap.** Paired dialogue-cluster bootstrap: 10,000 draws of the 20 sorted dialogues, seed 240924, the same draws
for every arm.

**Primary family (m = 30), Bonferroni quantiles 0.05/60.**
- For v ∈ {D3, D3G, D4, D4G, D5, D5G}, against matched D2 and RND: C_v − C_D2, C_v − C_RND, U_v − U_D2, U_v − U_RND.
- D5 and D5G versus the shuffle, on C and U.
- D3 and D3G utility versus D3CD.

**Mechanistic family:** m = 6.

**Role of the intervals.** Count and breadth gates, not interval significance, decide. Either controller variant may
qualify its family; this multiplicity is disclosed. Positive outcomes on few queries are exploratory.

## 12. Terminal precedence (first match)

| # | Label | Condition |
|---|---|---|
| 1 | `DIR_SPRINT0_INVALID` | Critical failure (see below), or all three families blocked |
| 2 | `DIR_SPRINT0_COMPUTE_BLOCKED` | Post-A recost > 9,000 s |
| 3 | `DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT` | Opportunity predicates fail, or no family is VALID_AND_TESTED |
| 4 | `DIR_SPRINT0_STEERING_FEASIBILITY_SIGNAL` | ≥ 1 family PROMISING_FEASIBILITY (with FULL audit PASS) |
| 5 | `DIR_SPRINT0_CAUSAL_POWER_WITH_DAMAGE` | ≥ 1 family SAFETY_FAILED |
| 6 | `DIR_SPRINT0_MECHANISTIC_EFFECT_ONLY` | ≥ 1 VALID family with a mechanistic effect |
| 7 | `DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE` | Remaining |

**Critical failures for rule 1:**
- provenance, firewall, compatibility, identity or integrity failures;
- an incomplete matrix;
- an old30 failure;
- D2 or RND energy failure;
- an audit failure.

**What labels authorize.** Only rule 4 may recommend a separately frozen full-decoding study. No label authorizes full
ASR decoding, A2-TTA integration, test-role use, confirmation or transfer.

## 13. Audits

| Barrier | Label |
|---|---|
| Pre-run | `PASS_TO_DIR_SPRINT0` |
| After Job A | `DIR_SPRINT0_AUDIT_A: PASS` |
| Before references | `DIR_SPRINT0_AUDIT_PRIMARY: PASS` |
| After evaluation | `DIR_SPRINT0_AUDIT_FULL: PASS` |

**Independent recomputation.** The auditor (`experiments/inference_cf_dir_sprint0_audit.py`) never imports the runner,
the mechanics module or the evaluator. It independently recomputes:
- population and exclusion;
- the D3 legal set and candidates;
- the gate;
- D0, D1, D4, RND, D5, D5SH and v_AC;
- the D5 evidence and prototypes;
- provider posteriors, for a hash-ordered sample;
- energies and decisions;
- the mapping, counts, bootstrap, predicates and label.

**Preservation.** Failed attempts are preserved. No outcome-dependent amendment is allowed.
