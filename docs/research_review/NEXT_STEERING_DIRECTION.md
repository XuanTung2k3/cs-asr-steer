# Next steering direction — ranked causes, candidate directions, one decisive experiment

Companion to [STEERING_FORENSIC_REPORT.md](STEERING_FORENSIC_REPORT.md). Evidence tags follow that report: **[F]** fact,
**[H]** hypothesis.

**Status.** Nothing here is authorized or run. The experiment below needs:
- a separately frozen spec;
- a reference-free implementation audit;
- explicit human approval before any GPU allocation.

Core v6 (`METHOD_CONTRACT.md`) is unchanged. The proposal stays within the DG-02 site and layers {16, 24}.

## 1. Ranked failure causes

Each cause lists: (1) evidence, (2) type, (3) what it explains, (4) fixable?, (5) one solution, (6) why it could work where
earlier methods failed, (7) the smallest falsifying test.

### C1 — The lexical information is absent at the point of intervention

1. **Evidence [F].**
   - Generic directions lie at random cosine to the lexical tangent: D0 −0.009 against random 0.022; D3/D4/D5 are
     orthogonal or Han-pointing.
   - D2 is class-level: 38/43 top-1 changes go to non-reference tokens.
   - Under a Han history the reference token sits at log-prob ≈ −11. This holds even at Mandarin-matrix span onsets
     (gap 10.6 nats, n = 17).
   - The same units are produced by the plain forced-EN whole decode: 43/60, and 556/1,074 overall, against 139/1,074
     in a shuffled control.
2. **Type.** Design limitation of every tested direction family, backed by empirical evidence.
3. **Explains.** D0/D1/D3/D4/D5, ST-LOC0, ST-PROMPT-R1, SRC-CF0, S1, R0, and why D2 corrects only near-ties.
4. **Fixable?** Plausibly. The information exists inside the same frozen model, along a different (English-history)
   trajectory.
5. **Solution.** Take the intervention vector from the model's *own* English-history state for *this* word: the DG-02
   site state of the forced-EN branch at the query that emits it. Transplant it, time-aligned, into the forced-ZH decode
   at the span onset.
6. **Why it could work.** Earlier vectors were averages over conditions (D0), class gradients (D2) or contrasts (D3–D5),
   so none encoded the word. This vector is computed from the audio by a model that demonstrably emits that word.
7. **Smallest falsification.** The XBT-L24 vs D0-L24 vs matched-random arms of XBT-0 (§3). C1 fails if the transplant
   does not beat generic and random edits of the same norm at time-aligned onsets.

### C2 — The target population was mis-specified

1. **Evidence [F].** Of the 1,074 forced-ZH English errors:
   - 29% are in English-dominant utterances, where AUTO already fixes them;
   - about 23% are scoring or annotation artifacts;
   - 139 are tail deletions beyond the 30 s window.

   The panels were 32–52% English-dominant and 78–89% mid-span, and deletions (50% of errors) were never targeted. The
   genuine Mandarin-matrix target is about 45–90 runs in 20 dialogues.
2. **Type.** Verified evaluation-design issue.
3. **Explains.**
   - small and fragile correction counts everywhere;
   - why AUTO looks strong;
   - why A2's "genuine" gains are whole-utterance switches;
   - why A2+P's fixed100 result came from one English utterance (U0023_S0_664).
4. **Fixable?** Yes, on CPU.
5. **Solution.** Freeze a genuine-target registry before any new outcome:
   - English runs in utterances ≤ 30 s whose reference is < 0.5 English;
   - run-level classes (Han-rendered / omitted / wrong English);
   - acronym, boundary-join, filler and disfluent-repeat units excluded;
   - English-dominant utterances reported separately as an LID problem.
6. **Why it could work.** It removes noise that no steering method can or should fix, and concentrates power on the
   actual scientific question.
7. **Smallest falsification.** A CPU re-tabulation of all historical correction events by registry class. C2 is wrong if
   most historical corrections are genuine Mandarin-matrix lexical recoveries away from near-ties.

### C3 — Candidate localization (selection) is unsolved

1. **Evidence [F].** Forced-EN words absent from forced-ZH are spoken English only 6.8% of the time in Mandarin-matrix
   utterances (0.81 in English-dominant). The rest are translations of Mandarin.
2. **Type.** Open research problem; the raw number is verified.
3. **Explains.** Why a naive "use the English branch" would be catastrophic, and why TTLS-AC abstained 94%.
4. **Fixable?** Unknown. This is the riskiest assumption.
5. **Solution.** Accept a forced-EN word only if three conditions hold:
   - its alignment-head time span overlaps a frozen R0 predicted-EN region (R0 rows already store intervals for FULL300;
     EN precision 0.967, recall 0.584);
   - the forced-ZH tokens aligned to that span are Han or absent;
   - the word is not already in the forced-ZH output there.
6. **Why it could work.** It localizes on *acoustic language evidence*, not on clean/null likelihood. The latter rewards
   translations (D3: 171/197 Han).
7. **Smallest falsification.** XBT-0 Stage K1 needs only alignment forwards: precision < 0.40 or recall < 0.25 kills it.

### C4 — The test-time objectives were uninformative

1. **Evidence [F].**
   - AUTO = forced-ZH on 211/214 Mandarin-matrix utterances.
   - TTLS's only genuine phrase (U0086_S0_185) came on a row whose pseudo-label *is* the error.
   - A2's NEW200 gains: 97.5% of the ZH reduction from 4 EOS rows; genuine units are AUTO copies.
2. **Type.** Verified design flaw of the AUTO-CE objectives.
3. **Explains.** A2, A3, A4, TTLS-R1/R1R, A2+P and PATH*.
4. **Fixable?** Only by importing new information, e.g. the C1/C3 candidates as pseudo-targets. More steps, LR, KL
   weights or LN subsets cannot add information.
5. **Solution.** Retire AUTO-CE TTA for code-switching. If TTA is revisited, use C3-localized forced-EN spans as sparse
   targets.
6. **Why it could work.** The target would then contain the word.
7. **Smallest falsification.** CPU check: every genuine AUTO-CE correction should lie on a row where AUTO ≠ B0, or at a
   near-tie. Already observed for TTLS-R1R and A2-NEW200.

### C5 — Autoregressive divergence is not controlled off-path

1. **Evidence [F].**
   - P2-SEQ: 41 changed transcripts, deletions 1,026 → 1,299, mode flips both ways (U0021_S0_513, U0086_S0_185), early
     EOS (U0027_S0_116).
   - PATH0: inducing the AUTO basin works 77% of the time; rescue works 2/4.
   - KL on the B0 prefix is non-binding (A2P-DEV200: D = 0).
2. **Type.** Empirical property of greedy autoregressive decoding.
3. **Explains.** P2-SEQ, SRD2-G0 harms, A3/A4, A2+P, PATH4 over-protection.
4. **Fixable?** Partly. One intervention at a verified onset, with the forced-ZH prompt unchanged, lets the model
   return to Mandarin natively. Forced-ZH already code-switches correctly for 1,194 English units.
5. **Solution.** No continuous pulses: edit one onset query per accepted candidate, then measure return-to-Mandarin and
   EOS explicitly as primary endpoints.
6. **Why it could work.** Damage scales with the number of edited positions and their misplacement. Here edits are
   sparse and only fire where acoustic English evidence and a disagreement both exist.
7. **Smallest falsification.** The ORACLE arm of XBT-0: if even oracle onset forcing causes as many new ZH errors as it
   recovers, onset-local correction is unsafe in principle.

### C6 — Content step 0 is outside the contract

1. **Evidence [F].** 49/269 wrong runs (<30 s) start at t=0. The forced-prefix exclusion (DG-02 frozen) makes them
   unreachable.
2. **Type.** Verified design limitation.
3. **Explains.** Part of the English-dominant failures.
4. **Fixable?** Yes, but only with a contract decision. For English-dominant utterances, AUTO LID is the simpler fix.
5. **Solution.** Record `OPEN DECISION: edit query 3 for content t=0` in a future spec. Report t=0 candidates separately
   in XBT-0.
6. **Why it could work.** The first token decides the basin (PATH0).
7. **Smallest falsification.** A count of XBT-0 candidates at t=0 and their FORCE outcomes. No contract change is needed
   to measure this.

### C7 — BF16 NormPreserve kernel at near-zero edits

1. **Evidence [F].** Noise floor of norm ≈0.0096 at state norm 6.6. Edits ≤ 0.01 are mostly rounding. The DG-02 hook
   still calls `hooks.apply_steering`.
2. **Type.** Verified latent implementation defect.
3. **Explains.** The TTLS-R1 confound (fixed in R1R). It is a minor contaminant for gated arms: in P2-SEQ, 29/237 edits
   were sub-resolution and at most 1 transcript change is attributable to them.
4. **Fixable?** Yes.
5. **Solution.** Any new arm uses the versioned ratio-first kernel (`ttls_r1r.apply_steering_ratio_first`) and hard-zero
   gating. Record an `IMPLEMENTATION GAP` proposal for the DG-02 hook; do not edit frozen code.
6. **Why it could work.** It gives exact identity at zero and realized edit = intended edit.
7. **Smallest falsification.** Already falsified on CPU (§4.6 of the report). Real-model identity is checked at
   XBT-0 start.

**Not root causes** (insufficient evidence): universal L16/L24 incapacity, cache leakage, gradient disconnection, BPE
fragmentation (57/60 targets are single tokens).

## 2. Candidate research directions (maximum three)

| Rank | Direction | Plausibility | Existing evidence | Novelty | Training-free | Difficulty | GPU | Publishability |
|---|---|---|---|---|---|---|---|---|
| 1 | **Cross-branch lexical transplant (XBT):** acoustically localized forced-EN candidates; onset-local DG-02 state transplant into the forced-ZH decode | Medium | Strong for availability (43/60; 52%); none for localization or delivery | Medium–high: input-specific counterfactual-condition activation transplant vs generic directions | Yes; reference-free at deployment | Medium (alignment heads + R0 regions + one hook) | Low (≤ 2 h on MIG 3g) | Positive result is a method paper; a negative one still gives a clean information-vs-leverage analysis |
| 2 | **Coverage-guided termination control:** suppress EOS when cross-attention has not covered the heard speech | High for MER/ZH, none for English lexis | Strong: A2's gains are EOS (F_ZH_EOS 0.965); 30 s utterances ending after 5–12 units | Low–medium | Yes | Low | Very low | Component/finding: "Whisper CS TTA gains are termination effects" |
| 3 | **Span-local conditioning switch:** language-token KV/prompt patch during predicted-EN regions, no lexical candidate | Low | Mostly against: same-prefix EN prompt 10/60; translate prompt emits Han 676/693 | Medium | Yes | Low–medium | Low | Low unless surprising |

Notes:
- **Rank 3 is not a separate study.** It enters only as a cheap control arm inside the XBT experiment.
- **Rejected for now**, each with evidence:
  - more generic direction discovery (DIR-SPRINT0);
  - gates on D2 (SRD2-G0: gated ≈ permuted);
  - any AUTO-CE TTA variant or LR/step/KL sweep (C4);
  - D3-style clean/null candidates (rewards translations);
  - larger single-site doses (nonspecific above η≈0.3; random L24 grows too).

**Critical caveat on Direction 1.** If XBT works only as well as forcing the candidate token, the contribution is the
*information pipeline* (localized English-history candidates), and the deployable form is constrained decoding, not
steering. The experiment is built to tell these apart. Claiming steering credit requires the transplant to approach
forcing *and* beat matched generic and random edits.

## 3. The single decisive experiment — XBT-0

**Question.** At time-aligned span onsets, does a reference-free, English-history DG-02 state transplant restore the
English word that generic directions could not, and is the result sequence-safe? This holds candidate quality fixed
across delivery arms.

**Population.** Exposed D-dev-select only.
- **Primary:** FULL300 utterances ≤ 30 s whose reference is < 0.5 English, scored against the frozen genuine-target
  registry (C2).
- **Optional power extension**, to be decided *before* the freeze: SRD2-400 and DIR-240, already exposed.
- **Separate reporting:** English-dominant utterances and t=0 candidates.
- **Never:** D-dev-confirm, D-test, router-calib, P3 or transfer data.

**Stage K1 — candidate localization** (forward-only, reference-free; scored offline):
1. Reuse the sealed forced-ZH (`B0M_L16`) and forced-EN (`B1`) decodes and R0 region intervals. Verify bitwise replay on
   a subset.
2. Run teacher-forced forced-EN and forced-ZH passes and take word time spans from the 10 large-v3 `alignment_heads`
   (cross-attention DTW).
3. Apply the frozen C3 rule (region overlap ≥ 0.5, Han-or-empty in forced-ZH, word absent in forced-ZH).
4. The onset query q* is the first forced-ZH query whose aligned time is at or after the candidate start.
5. **Endpoint:** precision and recall of the candidates against the registry, by stratum and dialogue.

**Stage K2 — delivery at q*** (single intervention, forced-ZH prompt unchanged, greedy free decoding thereafter, BF16,
versioned ratio-first kernel):

| Arm | Intervention at q* | Role |
|---|---|---|
| NONE | — | Baseline (bitwise B0M_L16) |
| FORCE | Emit the candidate's first token | Information-only ceiling for the reference-free pipeline |
| XBT-L24 / XBT-L16 | Replace the DG-02 site state with the forced-EN branch's site state at the query emitting the candidate's first token, rescaled to the forced-ZH norm | **Steering hypothesis** |
| D0-L24 | Per-query prompt difference at the same realized edit norm | Generic-information control (tests C1) |
| RAND-L24 | Random direction at the same realized edit norm | Perturbation null (59/300 code-path flip floor) |
| KVSWAP | Language-token KV switched to `<|en|>` for queries inside the predicted-EN region | Direction-3 control |
| ORACLE | Force the reference word's first token at the reference-aligned onset | Diagnostic only, non-deployable; delivery and return ceiling |

**Endpoints** (offline, reference-based):
- genuine run recovery (full and partial);
- new ZH errors (utterance-wide and within ±10 units);
- return-to-Mandarin (first reference ZH unit after the run correct);
- EOS/length change, MER and PIER;
- dialogue-bootstrap CIs and counts per dialogue.

**Cost.** About 600 teacher-forced passes plus (candidates ≈ 150–400) × 7 short free decodes. One Slurm job of ≤ 2 h
on MIG 3g.40gb. No training, no sweeps, one fixed configuration per arm.

**Decision rule** (frozen before outcomes):

| Code | Condition | Conclusion |
|---|---|---|
| **K1-FAIL** | Candidate precision < 0.40 or recall < 0.25 on primary registry runs | Localization is the bottleneck. Stop delivery work. Any further study is localization-only. |
| **K2-FAIL** | ORACLE recovers < 50% of registry runs, or ORACLE's new ZH errors ≥ recovered units | Onset-local intervention is unsafe even with perfect information. Close onset-local CS steering. |
| **K3-STEERING-NULL** | XBT-L24 recoveries ≤ RAND-L24 + 2, or < 0.5 × FORCE | Representation transplant adds nothing beyond its information. Reframe as localized constrained decoding. |
| **C1 test** | XBT-L24 ≥ 3 × D0-L24 recoveries and greater gap closure at matched norm | Earlier failures were information, not leverage (confirms C1). If XBT ≈ D0, leverage/site is the bottleneck: L24 vs L16 then becomes the next question. |
| **PROMOTE** (to a separately frozen expanded study, never confirmation) | All of: reference-free FORCE or XBT-L24 ≥ 15 genuine run recoveries in ≥ 8 dialogues; new ZH errors ≤ 0.5 × recovered units; MER not worse than B0 (dialogue-bootstrap upper bound ≤ +0.002); for a *steering* claim, XBT-L24 ≥ 0.7 × FORCE | — |

**Expected result [H, stated in advance]:**
- **K1:** recall 0.3–0.5. Precision is the main uncertainty, plausibly 0.3–0.6, limited by R0's EN recall of 0.58 and
  alignment-head jitter on short words.
- **ORACLE:** ≥ 60% single-word recovery with mostly native return to Mandarin; lower for multi-word runs.
- **FORCE:** ≈ ORACLE × precision.
- **XBT:** XBT-L24 between RAND and FORCE, possibly close to FORCE (late-layer readout); XBT-L16 weaker (overwritten by
  layers 17–31 attending to the Han history).
- **Controls:** D0-L24 ≈ RAND-L24; KVSWAP ≈ NONE outside English-dominant rows.

**Explicit failure conditions:** K1-FAIL, K2-FAIL and K3-STEERING-NULL above. Also:
- any NONE/zero-edit replay that is not bitwise;
- any reference access before the output seal;
- more than 5% of candidates whose q* cannot be mapped (alignment failure) → INVALID, not negative.

## 4. Fallback (only if K1 fails, or Direction 1 is closed) — EOS-COV0

**Question.** Is A2's measurable benefit reproducible without any parameter update, by a reference-free termination
rule?

**Rule.** When EOS is the argmax at step k, suppress it once if the alignment-head coverage of the forced-ZH decode ends
more than δ seconds before the heard speech end. δ is frozen, e.g. 1.5 s, with a frozen energy-VAD on the remainder.

**Arms.**
- B0;
- A2 (reused);
- EOS-COV;
- a matched random EOS-suppression control at the same firing rate.

**Population.** FULL300, including A2-MECH0's 12 EOS-recovery rows.

**Kill conditions.**
- EOS-COV recovers < 70% of A2's EOS-attributable ZH gain;
- or insertions/hallucinated tails rise by > 10%;
- or any change on rows where B0 is not early-stopped (these must be bitwise).

**Scope.** This answers the "safer continuation" question. It does not touch English lexis.

## 5. Prerequisites and contract notes

- **CPU, before any GPU work.**
  - Freeze the genuine-target registry (C2) and the candidate rule (C3).
  - Write the XBT-0 spec, firewall and decision rule.
  - Add focused tests: transplant identity at zero edit, norm rescaling, onset mapping and KV isolation.
- **OPEN DECISION** (human):
  - whether the transplanted vector, a per-utterance state from a counterfactual branch, is admissible as an inference_cf
    study outside the core-v6 basis (it uses the frozen DG-02 site and layers 16/24);
  - whether t=0 editing is ever allowed;
  - whether SRD2-400 and DIR-240 may be reused for power.
- **IMPLEMENTATION GAP** (record only; do not patch frozen code): the DG-02 hook uses the historical BF16 NormPreserve
  kernel. New studies should call the versioned ratio-first kernel.
- **Labeling.** All results remain exposed-development evidence. No fresh-validation or confirmation claim is possible
  without a separately authorized held-out stage.
