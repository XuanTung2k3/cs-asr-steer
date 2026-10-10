# MECH-LANG0 — Offline Mechanistic Diagnosis

**Status: EXPLORATORY / POST-HOC.**
- This is a retrospective, CPU-only analysis of already-sealed artifacts on the already-exposed D-dev-select panel:
  180 queries, 80 utterances, 20 dialogues.
- Nothing here is confirmatory. Nothing selects a direction, dose, sign, layer, threshold or gate for execution.
- The recommendation in section F is advisory. It is not an authorization.

Protocol: `docs/inference_cf/MECH_LANG0_PROTOCOL.md`, committed (`b60a03a`) before any derived quantity was computed.
Outputs: `results/inference_cf/mech_lang0/analysis.json` and `manifest.json`. Script:
`experiments/inference_cf_mech_lang0.py`. Tests: `tests/test_inference_cf_mech_lang0.py`.

## A. Provenance

| Item | Value |
|---|---|
| Original HEAD (= `origin/cs-asr-steer-inf`, clean) | `e5e4ea6291ff835bb37dc288ff61ae0181349e76` |
| Protocol commit | `b60a03a` |
| Code commit (the analysis ran at this HEAD) | `df9f6b5` (script `sha256:bb0984c4…`) |
| Final HEAD | the commit that adds this report and the outputs; see `git log` |
| Runtime | 27 s wall on CPU, peak RSS 1.5 GB |
| Model / GPU / Slurm / forward / autograd | none / none / none / 0 / 0 |
| Tokenizer | not loaded. `generation_config.json` was read as JSON for the frozen suppression list (hash = pinned config hash; suppression digest = SRC-CF0-P runtime) |
| Artifacts consumed | 1,539 files, each sha256-hashed in `manifest.json` (manifest hash `sha256:1156ff02…`) |
| Tests | `tests/test_inference_cf_mech_lang0.py`: 12/12 pass. No existing code was modified |

**Artifacts used.**
- **P2-DIR exp1.** `rows/*`:
  - `hb`, `he`, `D0`, `D2`, `g_readout` and `post_*`;
  - the evaluator `g_margin`;
  - D2 runtime provenance;
  - `exp1_run1_analysis.json`;
  - `extract_run1/states.npz`.
- **ST-LOC0.** `calibration/*`: states, `v1_vectors`, `d2_vectors` and `baseline_logits`. `barrier/*` and `pulses/*`:
  logits, before-states and consumed states.
- **ST-PROMPT-R1-A.** `capture/*` (states, `v_prompt`, `none_logits`) and `pulses/*`.
- **SRC-CF0-P.** `construction/*` (`clean_*`, `target_L*_v`, `random_L*_v`, region records), `pulses/*` and
  `runtime.json`.
- **S1.** `candidates/*`: region records and native-LID logits.
- **Panels.** The ST-LOC0, ST-PROMPT-R1, S1 and SRC-CF0 panels, and `p2rj/positions.json`.

**Query identity checks (all PASS).**
- **Panel order.** The master order is the ST-LOC0 runtime panel. It equals, on every shared key, the ST-PROMPT-R1, S1
  and SRC-CF0 panels, the SRC-CF0-P runtime projection and `p2rj/positions.json`. The keys are utterance, t, dialogue,
  absolute query = 4+t−1, content-prefix sha256 and forced-ZH query-input sha256.
- **Row joins.** Every per-utterance artifact was joined by (utterance_id, t), with the stored query index required to
  equal j: 180/180 for P2-DIR, ST-LOC0, ST-PROMPT-R1, SRC-CF0-P and S1. No join relies on array order.
- **L16 DG-02 state** (post-cross-attention / pre-FFN, forced-ZH, B0M_L16 prefix). All of these are bitwise identical
  on 180/180:
  - P2-DIR `hb`;
  - ST-LOC0 `states_L16_CROSS.H_B`;
  - ST-PROMPT-R1 `H_M`;
  - SRC-CF0-P `clean_L16`;
  - P2-DIR extract `H_B`, whose order was verified by byte matching.

  The 180 states are pairwise distinct. The L24 states are identical across ST-LOC0, ST-PROMPT-R1 and SRC-CF0-P.
- **Baseline logits.** The NONE logits are bitwise equal across ST-LOC0, ST-PROMPT-R1 and SRC-CF0-P.
- **Vectors.**
  - ST-LOC0 v_prompt equals ST-PROMPT-R1 v_prompt at L16 and L24, and equals P2-DIR D0 at L16.
  - ST-LOC0 D2 equals P2-DIR D2.
  - v_AC is present exactly where the S1 target is OK (73/180).
  - All vectors have shape 1280.
- **Reproduction of P2-DIR.** The sealed P2-DIR per-position analysis was reproduced from the raw arrays on 180/180
  positions: cos(d, g_tan), κ, D2/D0 Δm and ‖g_tan‖. The maximum absolute error was 1.8e-15.

**Historical totals reconstructed independently (all PASS).**
- **D2.**
  - 5 EN-confusion corrections, 8 ZH-correct and 1 EN-correct corruptions, and 43 EN-confusion top-1 changes.
  - S1-eligible: 3 of the 5 corrections and 3 of the 8 ZH corruptions.
- **ST-LOC0 v_prompt** (L16/L24 ±): 0 corrections, and the published corruption counts.
- **ST-PROMPT-R1-A**, all 24 arms: correction counts match (1 each at L24 η.30 −, L24 η.45 + and L24 η.45 −), and so do
  the published corruption counts.
- **SRC-CF0-P**: 0 corrections in all 8 v_AC arms, and the published per-arm corruptions on active rows.

**Available and missing.**

| Quantity | L16 | L24 |
|---|---|---|
| v_prompt (per query) | 180 | 180 |
| v_AC (per query; S1 target OK) | 73. Missing 107: LOW_ASSOCIATION 80, LOW_HEARD_MASS 14, NO_EN_REGION 13 | 73 |
| D2 | 180 | **MISSING**: never constructed at L24 |
| Reference-lexical gradient `g_margin` (evaluator-only, raw) | 180 (P2-DIR exp1) | **MISSING**: never saved at L24. All L24 lexical-alignment quantities are omitted; no surrogate is used |
| Edited logits | ST-LOC0 (e*), R1 (η), SRC-CF0-P (η, active rows) | same |
| Transcription-vs-translation task activations | **none in any archive** (5,910 JSON files scanned for a `<|translate|>` 50359 prompt; 0 found) | none |

## B. Direction geometry (L16 primary)

Notation:
- g_tan = g − h(h·g)/(h·h), where g is the gradient of m = lse z(Y_ref) − z(c*) at the L16 DG-02 state h.
- v_prompt + points to the forced-EN state; the historically useful sign was **minus**.
- v_AC + means "audio present"; D2 + is the applied sign.
- κ is the cosine of the **realized** edit (consumed − before) with g_tan.

### Alignment with the lexical tangent, EN-confusion (60 rows, 17 dialogues; v_AC 28 rows, 12 dialogues)

| Direction | mean cos | median | range | frac > 0 | median \|cos\| | dialogue means (range; # > 0) | realized κ (sign used) |
|---|---|---|---|---|---|---|---|
| v_prompt | −0.009 | −0.015 | [−0.075, +0.068] | 0.40 | 0.023 | −0.060…+0.022; 5/17 | minus +0.009, plus −0.010 |
| v_AC | +0.016 | +0.022 | [−0.109, +0.085] | 0.75 | 0.046 | −0.029…+0.056; 9/12 | plus +0.017, minus −0.017 |
| D2 | **+0.788** | +0.817 | [+0.443, +0.974] | 1.00 | 0.817 | +0.681…+0.939; 17/17 | **+0.785** |
| random (SRC PCG64) | −0.007 | −0.010 | [−0.091, +0.055] | 0.43 | 0.022 | −0.034…+0.015; 8/17 | −0.005 … +0.009 |

- **Isotropic reference** in d = 1280: E|cos| = 0.0223, sd = 0.028. Measured against it:
  - v_prompt is at random level;
  - v_AC is about 2× random in |cos|, with a consistent but tiny positive sign (75% positive);
  - D2 is about 35× random.
- **Other strata.**
  - EN-correct: v_prompt +0.012, v_AC +0.012, D2 +0.468.
  - ZH-correct: v_prompt −0.009, v_AC +0.025 (n = 6), D2 **−0.587**, with 0/60 positive. In ZH-correct rows the
    gradient points toward the Mandarin reference, so D2 points against it.
- **Mutual geometry.** Over all rows, median |cos| for each pair:
  - v_prompt vs D2: 0.029 (mean −0.000), at random level.
  - v_AC vs D2: 0.031 (mean +0.006), at random level.
  - v_prompt vs v_AC: mean −0.074 at L16 (85% negative) and **−0.141 at L24** (92% negative). This is a small but
    consistent anti-alignment: removing the predicted-English audio moves the state slightly toward the forced-EN-prompt
    side. It is descriptive only. The two layers are not compared as if their intervention meanings were the same.
- **L24.** No lexical gradient and no D2 exist there, so L24 lexical alignment is **unknown**, not zero.
- **Linear response holds.** For every L16 arm with stored native states, the first-order prediction g·edit tracks the
  observed Δm across EN-confusion rows:

  | Arm | corr(observed, linear) | mean observed Δm | mean linear Δm |
  |---|---|---|---|
  | D2 | 0.92 | +4.48 | +4.59 |
  | v_prompt e* | 0.95 | | |
  | v_AC η.15/.30 | 0.92–0.99 | | |
  | random | 0.77–0.95 | | |

  The exception is large relative doses. v_prompt η.45 − gives an observed +0.62 against a linear +0.19, and random
  η.30 gives +0.26 against +0.12. Above η ≈ 0.3 a non-specific nonlinear component appears, matching the large L24
  random effects reported in ST-PROMPT-R1.

## C. Lexical decision gap

The gap is G = max_{k∉Y} z_k − max_{k∈Y} z_k on processed logits (lowest-ID tie rule, verified on every row). G > 0 means
the strongest non-acceptable token wins.

**Baseline.**
- EN-confusion G median **11.10 nats** (IQR 8.95–12.90, range 0.81–16.84).
- Only **1/60** rows are within 1 nat of the boundary, 3/60 within 2 and **6/60 within 3.25**. The other 54 lie above
  5 nats.

| Family / study | Arm | n | ΔG median (macro) | median gap closed | ref-rank improved | corrections | other top-1 changes | EN-corr / ZH-corr corruptions |
|---|---|---|---|---|---|---|---|---|
| v_prompt ST-LOC0 (e*) | L16 − | 60 | −0.13 (−0.10) | 1.1% | 33 | 0 | 3 | 0 / 0 |
| | L24 − | 60 | −0.16 (−0.21) | 1.5% | 34 | 0 | 6 | 1 / 1 |
| v_prompt R1 (η) | L16 .45 − | 60 | −0.42 (−0.52) | 4.0% | 40 | 0 | 8 | 2 / 1 |
| | L24 .30 − | 60 | −0.50 (−0.65) | 4.9% | 34 | 1 | 10 | 1 / 2 |
| | L24 .45 − | 60 | −1.08 (−1.19) | 8.8% | 35 | 1 | 14 | 5 / 3 |
| random R1 | L24 .45 | 60 | −0.76 (−0.88) | 6.9% | 27 | 1 | 10 | 0 / 1 |
| v_AC SRC (η, active rows) | L16 .30 + | 28 | −0.23 (−0.27) | 2.7% | 14 | 0 | 1 | 1 / 1 |
| | L24 .30 + | 28 | −0.43 (−0.63) | 4.8% | 17 | 0 | 2 | 1 / 1 |
| | L24 .30 − | 28 | +0.10 (+0.21) | −1.0% | 4 | 0 | 6 | 3 / 2 |
| D2 (e*) | L16 + | 60 | **−2.81 (−2.98)** | **27.0%** | 31 | **5** | 38 | 1 / 8 (ZH ΔG median +2.09) |

- **Coverage.** All 67 arms are in `analysis.json` (`B_gap_by_arm`): 28 v_prompt (including R1 L3/L8), 8 v_AC,
  8 off-target, 22 random and D2.
- **Near-crossing.** No v_prompt or v_AC arm has more than **1** uncorrected confusion within 1 nat of the boundary
  after the edit. That is the same as the baseline count of 1/60. The post-edit G lower quartile stays at or above
  7.5 nats in every one of these arms.
- **Who gets corrected.**
  - D2's 5 corrections have baseline G of 3.25, 3.06, 1.63, 0.81 and 1.81. That is **5 of the 6** confusions within
    3.25 nats, and none of the 54 above 5 nats.
  - The single v_prompt correction (j = 25, CSD0501, baseline G 3.06) is one of the same six.
- **D2 acts at the script level.** It improves the reference rank on only 31/60 rows, and 38 of its 43 top-1 changes go
  to non-reference tokens. The historical P2-DIR record shows Δlog P_E +1.99 against Δlog p_ref +2.15. D2 raises
  English as a class, not the specific English word.

**Weak alignment versus too little movement.** First-order reach is A = ‖g_tan‖·‖edit‖, with median ‖g_tan‖ = 4.50 on
EN-confusion rows. A perfectly aligned edit covers the baseline deficit −m_base in:

| Dose | Rows covered (of 60) | Median κ required |
|---|---|---|
| e* or η .15 | 8–9 | **1.70** (impossible) |
| η .30 | 39 | 0.85 |
| η .45 | 52 | 0.57 |

Realized κ is 0.01–0.02 for v_prompt and v_AC, about **50–100× below** what η .30–.45 would need, and 0.79 for D2.

Hence:
- For v_prompt and v_AC the binding failure is **alignment**. Their margin movements are what the linear readout
  predicts from κ ≈ 0.01–0.02. They are not blocked by nonlinearity.
- For every direction, including D2, the tested dose leaves most confusions **out of first-order reach**. A large dose
  brings in non-specific nonlinear effects and damage (ST-PROMPT-R1).

**Comparability limitation.** The three studies used different dose regimes:
- ST-LOC0 and D2: an absolute chord e* = 1.126, about η 0.148–0.152 at L16 and 0.126 at L24.
- ST-PROMPT-R1 and SRC-CF0-P: relative η.

They also used different row sets (SRC-CF0-P: 28 active EN-confusion rows) and independent random draws. The table is
therefore **not** a matched-energy comparison across studies.

The closest descriptive slice is L16 at relative energy ≈ 0.15 on the 28 v_AC-active EN-confusion rows (median ΔG):

| Arm | median ΔG | corrections |
|---|---|---|
| v_prompt η.15 − | −0.13 | 0 |
| v_AC η.15 + | −0.15 | 0 |
| random | +0.05 / +0.06 | 0 |
| D2 | −2.20 | 3 |

This slice is retrospective and selects no dose or sign.

## D. D2 selectivity

| Group (D2 L16 e*) | n (dialogues) | S1 target OK | S1 paired | S1 status |
|---|---|---|---|---|
| EN-confusion corrections | 5 (5) | **3** | 3 | OK 3, LOW_ASSOCIATION 2 |
| ZH-correct corruptions | 8 (7) | **3** | 3 | OK 3, LOW_ASSOCIATION 3, NO_EN_REGION 2 |
| EN-correct corruption | 1 (1) | 1 | 1 | OK |
| ZH-correct harmless | 52 (20) | 3 | 1 | LOW_ASSOCIATION 47 |
| EN-correct harmless | 59 (20) | 38 | 34 | |
| EN-confusion uncorrected | 55 (17) | 25 | 17 | |

The historical overlap is reproduced: 3 of 5 corrections and 3 of 8 Mandarin corruptions are S1-eligible.

- Corrected rows: j 1 (CSD0011), 25 (CSD0501), 44 (CSD0532), 45 (CSD0540) and 54 (CSD0502).
- Corrupted ZH rows: j 120 (CSD0006), 128 (CSD0051), 135 (CSD0538), 139 (CSD1006), 163 (CSD0015), 165 (CSD0043),
  174 (CSD0541) and 176 (CSD1006).
- EN-correct corruption: j 94 (CSD0015).

**Pre-declared inference-available features.** These are the six frozen in the protocol; no other was searched. Values
are medians. P(A>B) is a descriptive pairwise rank probability. It is not a test, and no threshold is derived from it.

| Feature | Source (a: inference-available) | Corrections (5) | ZH corruptions (8) | ZH harmless (52) | P(corr > ZH-corr) | P(ZH-corr > ZH-harmless) |
|---|---|---|---|---|---|---|
| F1 predicted-EN region available | S1 runner | 3/5 | 3/8 | 3/52 | 0.61 | 0.66 |
| F2 query→region attention | S1 exact integral | 0.46 | 0.09 (n = 6) | 0.007 (n = 50) | 0.63 | 0.86 |
| F3 raw heard mass | S1 alignment heads | 0.85 | 0.92 | 0.94 | 0.35 | 0.43 |
| F4 utterance LID logit(en) − logit(zh) | native detect_language | −3.38 | −3.16 | −3.00 | 0.41 | 0.57 |
| F5 decoder script log-odds log P_E − log P_M | D2 runtime provenance | **−1.03** | **−3.62** | **−7.78** | **0.93** | **0.91** |
| F6 decoder top-1 probability | NONE distribution | **0.29** | **0.60** | **0.99** | 0.10 | 0.12 |

**Information classes.**
- (a) All six features are recorded by reference-free runners. They are computed for all 180 rows before any outcome
  label is attached, and the feature function cannot reach the references (enforced by a test).
- (b) The groups themselves are evaluator-only labels: strata, Y_ref, c* and D2 correction or corruption.
- (c) Unavailable at inference: whether a query is a code-switch confusion, the reference token, gold language labels
  and MMS-FA/CTC timing.

**Observed.**
- Both D2 corrections and D2 Mandarin corruptions sit on **low-confidence, near-boundary** states:
  - corrections: baseline G 0.8–3.25;
  - ZH corruptions: baseline G −0.19 to −5.69, with 6/8 within 1.5 nats;
  - F6 medians 0.29 and 0.60 for these two groups, against 0.99 for harmless ZH.

  Uncertainty marks where D2 can flip the top-1. It does not mark which way is correct.
- The acoustic and LID features (F3, F4) are essentially uninformative.
- F2 (attention to a predicted-English region) is higher for corrupted than for harmless Mandarin states.
- F5 is the only feature that orders corrections above Mandarin corruptions. But it is **D2's own objective J**, so
  gating on it is close to tautological. With n = 5 vs 8, post-hoc, no discrimination is established.
- **No selective gate has been learned or validated, and no threshold is proposed.**

**Is the unchanged S1 gate sufficient?** No. This is a retrospective restriction, counts only:

| Gate | Corrections kept | ZH-correct rows passing | Corrupted among passing | Rate |
|---|---|---|---|---|
| S1 target OK | 3/5 | 6/60 | 3 | **50%** |
| S1 paired | 3/5 | 4/60 | 3 | 75% |
| Ungated D2 | 5/5 | 60/60 | 8 | 13% |

- The S1 gate cuts Mandarin exposure from 60 rows to 6, so absolute ZH corruptions fall from 8 to 3. But the Mandarin
  states it lets through are exactly the ambiguous ones, with English-like acoustic evidence and attention.
- It also drops 2 of the 5 corrections (LOW_ASSOCIATION).
- The Mandarin-risk problem is therefore not solved by acoustic region availability.

## E. Next-hypothesis comparison

**H1 — transcription-vs-translation fidelity contrast**,
d_faithful = normalize(mean h(x, transcribe) − mean h(x, translate)).

- **Supporting:** conceptual only. The failure mode is English speech rendered as Mandarin, which is fidelity-like.
- **Contradictory, by analogy only:** the one prompt-token contrast tested (v_prompt, language token) is lexically
  orthogonal (|cos| ≈ random, κ ≈ 0.01). Whisper's translate task means X → English, so this contrast is entangled with
  output-language selection again.
- **Unknowns:** everything empirical. **No task-contrast activations exist in any archive. Construction feasibility is
  UNKNOWN.**
- **Risk:** high. It needs new GPU capture, it may be another conditioning axis, and its Mandarin effect is unknown.
- **Minimal experiment:** a capture-only geometry pass (no pulses). Record transcribe/translate DG-02 states on positions
  that already have a stored L16 g_margin, and measure cos(d_faithful, g_tan) against random.

**H2 — pooled language/script concept direction**, d_lang = normalize(mean_i[h_E(i) − h_M(i)]).

- **Distinction.** The tested per-query v_prompt is one prompt-conditioning contrast per query. A pooled direction is one
  corpus-level vector.
- **Four properties must not be conflated:**
  - prompt conditioning: which language token is forced;
  - script preference: Latin vs Han output mass;
  - language selection: which language model is active;
  - acoustic lexical identity: which word was heard.

  The ACL 2026 transliteration result concerns *script* representations. v_prompt measures *prompt conditioning*.
- **Geometry, reference-free.**
  - Per-query v_prompt shares a strong common component: mean resultant length **0.64 at L16** and 0.51 at L24.
    Leave-one-dialogue-out cos(v_i, pooled) has median **0.66** at L16 and 0.51 at L24. Pairwise median is 0.41.
  - A pooled d_lang would therefore stay close to the already-tested per-query vectors, which are lexically orthogonal
    and orthogonal to D2.
  - The lexically effective D2 is **query-specific**: resultant 0.18, pairwise median 0.006, LODO median 0.11. A fixed
    vector cannot track it.
- **Supporting:** consistency makes a pooled direction easy and cheap to build, and the transliteration literature
  motivates a script axis.
- **Contradictory:** the geometry above, and the fact that a fixed direction pushes every query the same way (D2-like
  Mandarin exposure without query specificity).
- **Unknowns:** a *script* direction (from decoder states emitting Latin vs Han tokens rather than from prompt tokens)
  has not been built or tested. Its alignment with g_tan is unknown. Per protocol, no pooled direction was evaluated
  against any lexical label here.
- **Risk:** high that it reproduces v_prompt's inertness or D2's ungated damage.
- **Minimal experiment:** an offline geometry check of a frozen, reference-free script contrast against the stored L16
  g_tan.

**H3 — selective readout steering**, h' = NormPreserve(h + g_conf · β · d_D2).

- **Supporting.**
  - The only direction with demonstrated lexical influence: κ 0.79, about 35× random, with 5/60 corrections at e*.
  - Linear response holds.
  - Fully reference-free construction (one activation-gradient call, no parameter gradient).
  - It corrects 5 of the 6 near-boundary confusions, consistent with S1's finding that the reference is accessible only
    on a minority of queries.
- **Contradictory.**
  - D2 is a script-class push, not lexical identity: rank improved on only 31/60, 38 off-reference top-1 changes.
  - It is anti-aligned with Mandarin-correct gradients (−0.59) by construction.
  - The S1 gate concentrates rather than removes Mandarin risk.
  - Uncertainty features flag corrections and corruptions alike.
  - The upside at e* is capped at near-boundary confusions (≈ 6/60 here).
- **Unknowns.** Whether any *pre-frozen*, reference-free gate separates beneficial from harmful firings on positions
  not used here, and its cost in corrections.
- **Risk.**
  - High Mandarin safety risk.
  - The only promising-looking feature (F5) is D2's own objective, and was seen post-hoc on these 180 rows.
  - Correction power is bounded at small doses.
- **Minimal experiment:** see section F.

**Ranking** (lexical influence, code-switch compatibility, reference-free information, Mandarin risk, compute, novelty):
1. **H3.** The only option with measured lexical leverage and a reference-free construction. Its single, well-defined
   open problem is selectivity.
2. **H1.** No evidence either way; feasibility unknown.
3. **H2.** Reference-free geometry argues against a fixed pooled prompt-language axis. A distinct *script* axis is
   untested.

This ranking rests on mechanism (alignment, linear response, consistency, gap ceiling), not on post-hoc performance
alone. The H3 selectivity evidence itself is inconclusive.

## F. Recommendation

**`NEXT_SELECTIVE_READOUT_FEASIBILITY`** (advisory only; not an experimental authorization and not a confirmed finding).

**Smallest discriminating follow-up (proposal; not implemented, not run):**

1. **Separate pre-outcome freeze** (human-authorized design pass). Fix one reference-free gate rule built from
   decoder-internal NONE quantities, and fix its firing budget by a reference-free rule.
   - Disclosure: F5/F6 were inspected post-hoc here, so the gate is informed by this exposure. **The 180 panel positions
     must not be used to evaluate it.**
   - No threshold is proposed here.
2. **Population.** New teacher-forced positions from the already-exposed D-dev-select FULL300 utterances **outside**
   the 80-utterance panel (development evidence only; no confirm/test/P3/transfer).
   - A reference-free census comes first, because FULL300 opportunities are unknown.
   - The status `FULL300_SAFETY_FEASIBILITY_UNKNOWN` is unchanged.
3. **Arms.** One single-pulse matrix:
   - NONE;
   - ungated D2 at L16 e* (unchanged P2-DIR recipe);
   - gated D2;
   - a random gate matched in firing rate on the same positions.
4. **Pre-registered endpoints.** EN-confusion corrections, ZH-correct and EN-correct corruptions, and the firing rate
   by stratum, with dialogue counts.
5. **The discriminating question.** Does the frozen gate lower the Mandarin corruption rate among fired ZH-correct
   states below ungated D2 *and* the matched random gate, while retaining corrections? The expected ceiling is small:
   only near-boundary confusions are reachable at e*.

## Established vs hypothesis

**Established** (exact, descriptive, on exposed data):
- At L16, v_prompt and v_AC are near-orthogonal to the reference-lexical tangent (|cos| 0.02–0.05; κ 0.01–0.02) and to
  D2. D2 is strongly aligned (0.79) on EN-confusion and anti-aligned (−0.59) on ZH-correct rows.
- The single-step margin response is linear in g·edit.
- The EN-confusion decision gap is large (median 11 nats). Only 6/60 are within 3.25 nats, and D2 corrects 5 of them.
- D2's Mandarin corruptions are low-confidence, near-boundary Mandarin states.
- The S1 gate keeps 3 of 5 corrections and 3 of 8 corruptions, and 50% of the ZH rows it lets through are corrupted.

**Hypotheses** (not established):
- That a reference-free gate can make D2 selective.
- That a script, rather than prompt, axis would align lexically.
- That a task-fidelity contrast is feasible or useful.
- Any statement about L24 lexical alignment.

Historical verdicts and seals are unchanged: P2-DIR, ST-LOC0, ST-PROMPT-R1-A, S1 and SRC-CF0-P.
`FULL300_SAFETY_FEASIBILITY_UNKNOWN` remains.

## Data exposure

- No new data, role, split or sample. No D-dev-confirm, D-test, router-calib, P3 or transfer access.
- Inputs were already-exposed D-dev-select artifacts only.
- Evaluator-only references were used retrospectively for evaluator-style description: P2-RJ acceptable sets,
  competitors and strata, and the P2-DIR evaluator gradient.
- No model forward, autograd, vector construction from outcomes, GPU or Slurm.
