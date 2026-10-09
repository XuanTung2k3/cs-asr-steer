# SRD2-G0 final report — acoustic-conflict-gated readout steering feasibility

**Terminal label: `SRD2_G0_OBSERVED_DAMAGE`.** It is the first-matching label under the frozen precedence, and
the independent FULL audit agrees: `SRD2_G0_AUDIT_FULL: PASS`, no critical failure.

- **What triggered it:** the only severe-damage disjunct that fired is `new_EOS_proxy_events_B2 ≥ 1`, at 1.
  The study **STOPS**.
- **What the label does not authorize:** a follow-up stage, full decoding, a gate/dose/direction revision,
  confirmation, test-role use or transfer.
- **Scope:** all results are local, single-query, development evidence from 400 new utterances. Those utterances
  belong to the same 20 already-used dialogues, so this is not independent confirmation.

## 1. Repository and provenance

| Item | Value |
|---|---|
| Frozen design commit | `3168508e` (config `sha256:218b3be9…`; design/test bytes verified against `SRD2_G0_FREEZE.json`) |
| Implementation | `42705ed4`: runner, mechanics module, evaluator, auditor, Slurm script, 38 focused tests |
| Prepare / manifest | `bf060725` (runtime hash `sha256:167f1c84…`) / `ed835d8b` (manifest hash `sha256:08940ec9…`, 61 pinned sources) |
| Pre-run audit | `70b8e042`: `PASS_TO_SRD2_G0` on the first attempt, 37/37 checks |
| Job A | Slurm 58300; seal commit `fc23d259` (`gate_seal` `sha256:6d56128a…`) |
| Audit A | attempt 1 **BLOCK** preserved, plus auditor amendment A1 (`a651bb3b`); `SRD2_G0_AUDIT_A: PASS` with r1 (`dfdf461e`) |
| Permutation / authorization | `ba129235` / `a88f3d38` (`authorization_B` `sha256:7a717645…`) |
| Job B | Slurm 58314; seal commit `6bdaa493` (`pulse_seal` `sha256:40a5b228…`) |
| PRIMARY audit | `b8180514`: `SRD2_G0_AUDIT_PRIMARY: PASS` |
| Evaluation + FULL audit + terminal | `211d1c2c` (`audit_FULL` `sha256:08e0e54f…`) |

**Population.** Selected-ID hash `sha256:ff2e3054…`, roster hash `sha256:28aa458b…`. The 400 utterances are 20 per
dialogue across 20 dialogues, with zero overlap with FULL300 or with any other known intervention-exposed ID.

**Model.** Whisper-large-v3: all 11 model, tokenizer and preprocessing hashes match the freeze. BF16, eager
attention. Torch 2.10.0+cu128, Transformers 4.57.6. Weights were unchanged across both jobs (state digest
compared at start and end) and no parameter gradients were created.

## 2. Scientific integrity

**Audits.**
- Pre-run: `PASS_TO_SRD2_G0`.
- A: attempt 1 `BLOCK`, preserved; then PASS with r1.
- PRIMARY: PASS, before any reference was opened.
- FULL: PASS. Evaluator and auditor agree on every metric, gate, integer count, per-dialogue tally, bootstrap
  interval and the label.

**Auditor amendment A1** (recorded in `results/inference_cf/srd2_g0/run1/audit_r1/AMENDMENT_A1.json`; it touched
only the auditor):

1. *Script-mass precision.* The auditor recomputed raw script masses in float64. The frozen R2 definition is a
   float32 full-vocabulary log-softmax, so the two differ by up to 4.0e-5, which exceeded the auditor's own
   1e-6 tolerance. An independent float32 recomputation reproduces all 1,167 sampled values exactly.
2. *Opened-path classification.* The runner hashes, and never imports, the pinned historical source
   `inference_cf_p2r_population.py`. The auditor flagged that file by name before checking whether it was
   pinned.
3. *What did not change:* runner, evaluator, configuration, thresholds and Job A data.
4. *Where r1 lives.* r1 is stored in the run directory so that the pre-audited runner stayed byte-identical for
   Job B.
5. *Auditor of record.* r1 is the auditor of record for audits A, PRIMARY and FULL. The canonical auditor file
   is kept at its manifest-pinned bytes so the source pins remain verifiable (see `AMENDMENT_A1_ADDENDUM.json`).

**Authorization serialization.**
- The runner wrote `authorization_B.json` with keys sorted, but the same runner's Job-B guard checks field order.
- The runner output is preserved verbatim as `authorization_B_runner_sorted.json`.
- The identical content was re-serialized in the frozen field order. Content equality was verified, and the
  runner was not changed.

**Reference firewall.**
- *Runner inputs:* the four-key runtime projection only. Unexpected or extra fields are rejected recursively.
- *File-open logs:* the Job A and Job B logs contain only audio, model, pinned-source and run-directory files,
  plus, for Job B, the hash-pinned old30 apparatus artifacts. No reference, role parquet, CTC or evaluation path
  was opened.
- *Reference access:* references (`transcript_raw`) were read only by the separate evaluator and FULL auditor.
  Both ran after the pushed pulse seal and PRIMARY PASS, and both apply a pushdown filter to the 400 frozen IDs.
- *Unit ledger:* `evaluation_units.json` contains reference surfaces, so it is kept out of Git; its hash is in
  `evaluation.json`.

**Historical files.** No historical file, result, seal, contract or verdict was modified.

**Jobs.** Two scientific GPU jobs in total (17.3 min and 15.8 min), with no reruns.

## 3. Population and coverage

**Utterances.**
- 400/400 rows over 20 dialogues. 399 baselines stopped on EOS and 1 hit the 200-token cap.
- Median length is 16 content tokens (mean 28.3).
- 23 utterances longer than 30 s were run in all four arms. Their lexical attribution is primary-unalignable
  (`heard_scope`), as frozen.

**Queries.**
- 11,720 inventory queries in total.
- 10,267 are structurally eligible.
- The other 1,453 are recorded four-arm no-ops: 400 t=0 forced-prompt queries and 1,053 queries with a
  mid-character prefix. No query had an unexpected special token or a non-finite native state.

**Mapping coverage.**
- English unit mapping: 2,416 of 2,917 normalized Latin units, 0.828 (frozen floor 0.8).
- Mapped primary queries, all in the frozen denominators:

  | Stratum | Mapped queries | Dialogues | Structurally eligible |
  |---|---|---|---|
  | EN-confusion | 857 | 19 | 840 |
  | EN-correct | 504 | 18 | 477 |
  | ZH-correct | 5,292 | 20 | 4,471 |

- Pre-registered target-set interpretation, fixed before any reference was opened: an empty historical
  `target_set`, or a baseline token inconsistent with the stratum, makes the unit unalignable.

**Abstentions and fallbacks.**
- No gate-provider fallback on any structural query.
- D2 was valid on 10,267/10,267 structural queries.
- B2 abstentions: 210 cells on tiny gates (70 solver unattainable, 140 native consumption unattainable),
  losing 0.05% of planned energy.
- B3 abstentions: 194 cells (70 solver unattainable, 124 native consumption unattainable), losing 0.04%.

## 4. Gate performance (R2 `g_old`)

**Distribution over structural queries.**
- E > 0 on 26.9%; R_B > 0 on 89.4%.
- g > 0 on 1,968 queries (19.2%): median of nonzero g 0.735, IQR 0.38–0.88. Every dialogue has gate-positive
  queries (2–246 per dialogue).

**Exposure by stratum** (descriptive):

| Stratum | Share of structural queries with g > 0 |
|---|---|
| EN-confusion | 88.6% |
| EN-correct | 0.2% |
| ZH-correct | 2.2% |

**Full-replay versus cached compatibility.** All frozen bounds pass:
- raw TV median 0.00035, p99 0.0275, max 0.0750;
- processed argmax agreement 0.99786;
- site relative L2 p99 0.0169, max 0.0420; site cosine minimum 0.99912;
- causal-probe TV and attention-L1 maxima 0.0321 / 0.0119; future self-attention mass 0;
- null LID pins exact.

**Shuffled gate (B3).**
- Within-utterance hash order. Multisets and planned squared energy are identical to B2 (1211.0137 each).
- 4,373 queries (42.6%) sit in uninformative utterances: 241 identity permutations including 85 singletons and
  233 constant-gate utterances. This is within the frozen 50% cap.

**Energy.**
- Realized B2/B3 squared-energy ratio is 0.99987.
- B2 executed chords: median 0.875, IQR 0.62–1.01.

## 5. D2 intervention

**Apparatus.** The old30 historical apparatus reproduced 30/30 queries bitwise: clean logits and site, D2
direction (max error 0), J within 1e-6, e* pulse logits and consumed state, zero-dose hook, and restore.

**New population checks.**
- Clean cached logits and site are bitwise equal to Job A at every query.
- D2 scratch logits and site are bitwise equal to the clean step.
- Every pulse restore is bitwise.

**Solver accuracy.**
- B1 matched 10,267/10,267 queries (1.000).
- Executed relative squared-energy error max 0.0114; chord error max 0.0057.
- Consumed/proposed norm ratio max 0.00498 (guard 0.005).
- The FFN input equals the native preview bitwise in every executed cell, and every cell used exactly one solve.

**Resources.** Median D2 time 0.033 s per query; median pulse 0.012 s; peak VRAM 3.7 GB.

**Raw arrays.** 2.8 GB (Job A) and 1.6 GB (Job B), archived content-addressed under
`/mnt/data/tungnx/cs-asr-steer/archives/srd2_g0/run1/`.

## 6. Four-arm results

All 400 rows. Counts are unique (UID, t) queries; harm dialogues count dialogues with any correct-state
corruption.

| Arm | EN corrections | EN corruptions | ZH corruptions | Correction dialogues | Harm dialogues | Utility U |
|---|---|---|---|---|---|---|
| B0 NONE | — | 0 | 0 | — | — | 0 |
| B1 UNGATED D2 (e\*) | 4 | 1 | 446 | 3 | 20 | −443 |
| B2 R2-GATED D2 | 1 | 0 | 9 | 1 | 7 | −8 |
| B3 SHUFFLED-GATE D2 | 1 | 0 | 4 | 1 | 2 | −3 |

**Active (actually edited) ZH-correct queries:**

| Arm | Edited | Corrupted | Active corruption rate |
|---|---|---|---|
| B1 | 4,471 | 446 | 10.0% |
| B2 | 89 | 9 | 10.1% |
| B3 | 98 | 4 | 4.1% |

**Unconditional ZH corruption rates:** B1 0.084, B2 0.0017, B3 0.0008.

**EOS-risk proxy** (new EOS with ≥ 10 baseline content tokens remaining):

| Arm | Events | All new-EOS proposals |
|---|---|---|
| B1 | 85 | 126 |
| B2 | 1 | 17 |
| B3 | 2 | 15 |

The decisive B2 event is `ZH-CN_U1063_S0_4`, t=12, 11 tokens remaining, g = 0.791. B3 proposed EOS at the same
query.

**Top-1 changes over all queries:** B1 2,209, B2 875, B3 781.

## 7. Main causal comparisons

**B2 versus B1 (gating).**
- Correction retention is 1/4 = 0.25 (needs ≥ 0.6).
- Mandarin corruptions fall from 446 to 9 (ratio 0.020): Bonferroni 98.75% interval for the reduction
  [315, 570].
- Gating removes about 98% of Mandarin **exposure** (89 versus 4,471 edited ZH-correct queries), not the per-edit
  harm: the active corruption rate stays at about 10% in both arms.

**B2 versus B3 (the primary selectivity test), paired by dialogue:**

| Quantity | Point | Bonferroni 98.75% interval |
|---|---|---|
| C_B2 − C_B3 | 0 | [0, 0]: the single correction is the same query in both arms |
| U_B2 − U_B3 | −5 | [−14, 5] |
| H_ZH_B3 − H_ZH_B2 | −5 | [−14, 5] |

Breadth: 0 dialogues have a positive correction gain, and 1 has a positive utility gain.

**Equal-cost utility** is negative in every arm.

**Does selectivity exceed dose reduction?** No. With identical planned-dose multisets, B2 is no better than B3 at
equal realized energy (ratio 0.9999). The gate's apparent harm reduction against B1 is dose reduction.

## 8. Terminal decision (first match)

| # | Rule | Outcome |
|---|---|---|
| 1 | INVALID | Not matched. Critical flags clear; compatibility and energy execution PASS. |
| 2 | COMPUTE_BLOCKED | Not matched. Resource PASS: 2,648 s forecast, ≤ 9,000. |
| 3 | **OBSERVED_DAMAGE** | **Matched** by `new_EOS_proxy_events_B2 = 1 ≥ 1` (see below). |

The other severe-damage disjuncts did not fire:
- `H_ZH_B2 = 9` is not ≥ 10, and the rate 0.0017 is not > 0.03.
- No dialogue has ≥ 3 ZH events with an active rate > 0.20.
- `H_EN_B2 = 0`.

**Later predicates.** These cannot change the label; they are reported because every one is frozen.
- *Opportunities: PASS.* 857/19, 504/18, 5,292/20; active B2 ZH-correct 89 across 19 dialogues; mapping 0.828.
- *Causal power: FAIL.* `C_B2 = 1` (needs ≥ 5); 1 dialogue (needs ≥ 3).
- *Control opportunities: FAIL.* `C_B1 = 4` (needs ≥ 5). These pass: `H_ZH_B1 = 446`, harmed dialogues 20,
  uninformative fraction 0.426.
- *Dose comparison: PASS.* Multisets equal; ratio 0.99987; lost energy 0.00053 / 0.00036.
- *Acceptance: FAIL on 10 of 14 predicates.* The 4 that pass are `ZH_harm_ratio_B2_B1`, `H_ZH_B2 ≤ 10`, the
  unconditional ZH rate, and both EN criteria. The failures are:
  - retention 0.25;
  - active ZH rate 0.1011 > 0.10;
  - C_B2 − C_B3 = 0;
  - U_B2 − U_B3 = −5;
  - H_ZH_B2 − H_ZH_B3 = +5;
  - correction-gain dialogues 0;
  - utility-gain dialogues 1;
  - maximum utility share 1.0;
  - U_B2 = −8.

**Robustness of the conclusion.** Without the single EOS event the next matching label would have been
`SRD2_G0_CAUSAL_POWER_INSUFFICIENT`. The study's conclusion does not hinge on the EOS proxy.

## 9. Scientific interpretation

**Established on this development population** (local, single-query):

- **The R2 gate is a strong exposure selector.** It fires on 89% of English-confusion queries and on only 2% of
  correct-Mandarin and 0.2% of correct-English queries.
- **D2 at the frozen budget has almost no lexical correction power on this population.**
  - Ungated D2 changes the top-1 at 606 of 840 English-confusion queries but corrects only 4.
  - Of those 606 changes, 519 move to another Han token and only 9 to Latin script. The readout push reshuffles
    within-Han ranking far more often than it crosses the script boundary.
  - This is consistent with MECH-LANG0's large median lexical gap: about 11 nats at historical positions, against
    a first-order reach of about 5 nats.
- **Gating reduces damage only by reducing exposure.** About 10% of edited correct-Mandarin states are
  corrupted whether or not the edit is gated. At matched planned dose the true gate is not better than a
  within-utterance shuffle; numerically it is slightly worse.
- **D2 at e\* frequently proposes premature EOS.** 85 ungated events and 1 gated event had ≥ 10 tokens
  remaining. The frozen early-stop risk proxy treats this as observed damage.

**What worked:**
- the reference-free pipeline and identity checks: bitwise apparatus, cached and pulse identities, exact energy
  control;
- the gate's exposure selectivity;
- adequate opportunity coverage.

**What failed:**
- causal correction power: 1 B2 correction;
- selectivity beyond dose reduction;
- utility, negative in all arms;
- the frozen EOS safety proxy.

**Does the acoustic gate truly help D2?** Not in a way that yields corrections. It suppresses harm by suppressing
edits.

**Does the evidence justify further work on this combination?** No.

**Limitations.**
- New utterances from the same 20 already-used dialogues; no independent-dialogue evidence.
- First-token, single-query outcomes only, with no free decoding. The EOS proxy is not a measured truncation.
- 23 long-audio rows are unalignable for lexical outcomes.
- Mandarin endpoints are next-token preservation outcomes.
- The shuffle control is weak inside utterances with uniformly high gates (42.6% uninformative).
- Correction counts are tiny (4, 1, 1), so intervals are wide.

## 10. Next research recommendation (advisory; not executed)

**Close the gated-readout (D2) line.** No SRD2-G1, and no dose, gate, layer, sign or direction variants of this
actuator, because the bottleneck is the actuator's lexical reach at confusion states, not the gate's targeting.

The single most justified next step is the project's **already separately frozen main-line exact-A2
confirmation**: P2-A2-MECH0 reported `A2_CONFIRMATION_READY = YES` and that confirmation has not been run. It
requires its own explicit human authorization.

SRD2-G0 adds one consistent observation: local readout pushes readily induce early-EOS proposals, and A2's
mechanism is termination-dominant EOS recovery. That link is descriptive only, not evidence for A2.

## 11. Final project state and data exposure

- **New exposure.** All 400 SRD2-G0 `D-dev-select` utterances (20 dialogues) now have their references opened
  for lexical evaluation. Treat them as **exposed development data**; they are no longer unexposed candidates
  for a future fresh population.
- **Apparatus inputs.** The old30 P2-DIR engineering queries come from historical, already-exposed FULL300
  utterances and were used only for engineering reproduction, outside every new denominator.
- **No other roles touched.** No D-dev-confirm, D-test, router-calib, P3 or transfer data.
- **DATA_EXPOSURE.md is not edited.** Its bytes are pinned by the frozen SRD2-G0 contract test
  (`tests/test_srd2_g0_freeze.py`), so this exposure is recorded in `docs/current/STATUS.md` and in this report.
  A human should carry it into `DATA_EXPOSURE.md` at the next authorized freeze.
- **Historical results unchanged.** Core v6, METHOD_CONTRACT and all historical verdicts are unchanged.
- **Descriptive tables.** `results/inference_cf/srd2_g0/run1/descriptive.json` (post-terminal, non-decisional).
