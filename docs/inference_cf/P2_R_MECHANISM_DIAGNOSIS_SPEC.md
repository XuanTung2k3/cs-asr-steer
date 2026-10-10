# P2-R mechanism diagnosis — frozen spec v1

**Stage:** P2-R is a new, versioned **diagnostic** stage. It is not a method-selection stage.

**Project state while it runs:** `P2_COMPLETE_MECHANISM_UNRESOLVED`, **P3 HELD**.

**Frozen before any P2-R outcome.** The following are the frozen artifacts:

| Artifact | Path |
|---|---|
| Config | `configs/inference_cf/p2_r_mechanism_diagnosis.json` |
| Population | `results/inference_cf/p2r/population.json` |
| Code | `experiments/inference_cf_p2r{,_population,_prepare,_analyze}.py` |
| Slurm | `slurm/inference_cf_p2r.sbatch` |
| Tests | `tests/test_inference_cf_p2r.py` |

**Design input:** the Codex contract `P2_R_CODEX_CONTRACT.md`, archived verbatim.

## 1. What P2-R does not change

- **P2 stays as it is.** Its standing is `P2_VALID_CONFIGURATION_SELECTED` and `P2_AUDIT: PASS`.
  - The selected configuration stays L16, α = 2, φ_id, g = E·R_B, +d.
  - Its thresholds, grids and reports are unchanged.
- **P2-R changes nothing retroactively.** It only asks why the P2 mechanism did not show
  selective lexical repair.
- **Opposite-sign superiority is not a premise.** The P2 report's descriptive "−d +11 vs +d +2"
  is not an established ranking, and `−d` is never adopted.

## 2. Data roles (firewall)

| Role | P2-R use |
|---|---|
| `D-dev-select`: the 300-utterance R2/P2 panel, 20 dialogues | **Diagnostic only.** It is already exposed, so it supports exploratory causal diagnosis, never confirmation or generalization. |
| `router-calib`: 3,031 utterances, 8 dialogues; its 300 shortest were used for DG-04 calibration | **Not used.** At most a *conditional* future calibration/replication candidate, and only with explicit human authorization. It is not untouched validation data. |
| `D-dev-confirm`, `D-test`, P3 populations, transfer corpora (SEAME, CS-FLEURS, ViMedCSS, Qwen) | **No access.** `D-test` already has historical DG-08 exposure and stays prohibited for development. |

**Consequence:** no fresh repair-validation role exists. Any repair hypothesis frozen after P2-R
can only be `REPAIR_NOT_VALIDATED_NO_FRESH_ROLE` in this session.

## 3. Population (frozen, built from baseline and evaluator information only)

- **Source.** The matched baseline `B0M_L16` (`results/inference_cf/p2_A_r1_L16`) supplies the
  token sequences. It is token-identical to the R2 baselines on 300/300 utterances.
- **Labels.** The frozen R2 evaluator alignment (`results/inference_cf/p0_r2/evaluation_units.json`)
  supplies strata and positions. The role references (`transcript_raw`) supply the target words.
- **Strata:** `EN-confusion` (primary), `EN-correct`, `ZH-correct`.

**Structural eligibility** (exclusions are counted in `ledger`):

- R2 alignable.
- Position t ≥ 1. The first content token is structurally non-editable (forced prefix).
- UTF-8-complete prefix.
- The hypothesis unit begins exactly at token t.
- For English strata, only one reference unit maps to the position.

**Reference-consistent target set** (English strata; a minimal correction, §9):

- **Variants:** {raw-case, lower, Capitalized, UPPER} × {no space, one space}, each appended to the
  decoded **exact generated prefix**. A variant counts only if encoding keeps the generated prefix
  IDs unchanged.
- **Distinguishing beginning:** a candidate's next token must, when stripped and lower-cased, be a
  prefix of the word, of length ≥ min(2, |word|).
- **Log-probability:** `log p(ref)` is the log of the summed probability over the set. Rank is the
  best rank in the set.
- **By stratum:**
  - `EN-correct` requires that the baseline token belongs to the set.
  - `EN-confusion` requires that it does not.
  - `ZH-correct` uses the baseline token as its target.

**Selection** (Codex, exact):

- 60 positions per stratum, chosen round-robin over dialogues.
- Within a dialogue, positions are ordered by `SHA-256("P2R-v1|uid|t|stratum")`.
- At most 2 positions per utterance per stratum.
- Correct-stratum controls are taken from already-selected utterances first.
- The union is capped at 80 utterances.
- Minimum interpretable population: 40 positions and 12 dialogues per stratum. Below that, the
  stratum is reported as not estimable and nothing is expanded.
- The first 20 selected positions per stratum are **companion** positions.

**Frozen outcome of selection:**

| Stratum | Positions | Utterances | Dialogues |
|---|---|---|---|
| EN-confusion | 60 | 37 | 17 (13 span-initial) |
| EN-correct | 60 | 49 | 20 |
| ZH-correct | 60 | 43 | 20 |

- **Union:** 80 utterances.
- **D2:** 209 target positions (all valid EN-confusion in the 80 utterances) and 2,507 control
  positions.

**Secondary stratification (descriptive, pre-registered).** A pre-outcome structural inspection
found that 188 of 232 structurally valid EN-confusion units sit *inside* English spans that the
forced-ZH baseline translated into Mandarin: the preceding reference unit is itself a
mis-recognized English word. EN-confusion results are therefore also reported separately for:

- **span-initial** positions, where the previous reference unit is MATCHed or absent;
- **within-span** positions.

## 4. Site, directions and energy (D1/D3)

- **Site:** L16, the exact DG-02 post-cross-attention / pre-FFN residual, through the canonical
  hook, with NormPreserve, bf16 and eager attention. Suppression is identical to `generate()`, and
  `num_forced_prefix` = 4.
- **Arms at each D1 position** (each a single pulse on the query that predicts token t):

| Arm | Direction |
|---|---|
| no-edit | the unedited B branch |
| `+d` | d = norm(hE − hB), with the frozen `core_p1.direction` fallback (fallback → excluded) |
| `−d` | −d |
| **random** | exactly one Gaussian vector per position |

- **Random direction:** seed `int(sha256("P2R-random-v1|uid|t")[:16], 16)`, drawn with
  `numpy.default_rng(seed).standard_normal(1280)`.
  - Project it orthogonal to the pre-edit site state, then normalize.
  - Degeneracy fallback: e₀ projected and normalized.
  - No seed search and no retries.
- **Identical state:** every arm starts from the same unedited B-branch cache and baseline prefix.
  The cache is cropped back after each arm. The restored unedited step is verified bitwise.
- **Energy.** Gate magnitude is removed. The target realized edit norm is
  **e\* = sqrt(1203.3762345332195 / 949) = 1.12608**: the RMS hook-audited edit norm of the
  selected P2 run. It is not a tuned dose.
  - **Solver:** it runs inside the hook action on the pre-edit site state r.
    - Analytic chord: `s = |r| sin θ* / sin(φ − θ*)`, with `θ* = 2 asin(e*/(2|r|))`.
    - Then a secant refinement that re-evaluates the hook's own `apply_steering` in the site dtype,
      at most 8 evaluations.
    - It never reads logits or correctness.
    - The direction is passed pre-scaled (s·v, gain 1), so the scalar is not rounded to bf16.
  - **Unreachable** (φ ≤ θ*): the arm is recorded as `energy_unreachable`, gets no edit, and the
    position leaves matched comparisons.
  - **Matching:** a position is *matched* only if all three arms satisfy |e²/e\*² − 1| ≤ 0.02 and
    are pairwise within 0.02·e\*². Tolerances are never relaxed.
  - **Energy measures:** matching uses the hook-audited ‖steered − site‖, the P2 energy
    definition. The downstream-consumed edit, (q + u′) − r measured by the recorder, is
    co-reported.
- **Companions:** after the pulse, greedy free continuation with no further edits, up to the
  original 200-token cap.

**Metrics per arm** (processed distribution = the decode rule; raw-logit quantities co-reported):

- `L = log p_arm(ref) − log p_none(ref)` and `P = p_arm(ref) − p_none(ref)`.
- Lexical specificity: `Lex = Δ[log p(ref) − log P_E,processed]`.
- ΔP_E and ΔP_M (raw BC-B masses), ambiguous mass, and processed masses.
- Reference rank.
- Margin: `log p(ref) − max_{v∉ref} log p(v)`.
- Log-odds of the reference against the baseline token.
- Top-1 change and whether the new top-1 is in the reference set.
- KL and TV to the no-edit distribution.
- Realized and consumed edit norms, relative squared energy, `cos(edit, v)`, `cos(d, h)` and ‖hE − hB‖.

## 5. D2 — localization

- **Current:** a controlled baseline-prefix replay of the frozen P2 schedule on the 80 utterances.
  - B and E are teacher-forced on the matched-baseline tokens.
  - One S branch applies the P2 edit, using the same frozen R2 providers and the same hook as
    `cached_decode` (α = 2, g = E·R_B, +d), with persistent cache history.
- **Verification:** on every prefix identical to the saved P2 selected run, the replayed
  (g, edit, edit_norm) must equal the saved P2 steps.
- **Oracle:** energy-packet relocation (Codex, exact).
  - Keep current edits already at targets.
  - Order donor edits at other positions by descending realized energy, breaking ties by
    position.
  - Assign each donor once to an unedited, direction-valid target in frozen hash order, within the
    same utterance.
  - Remove the donor edit.
  - At the destination: +d_t = norm(hE_t − hB_t), with energy equal to the donor's realized edit
    norm via the §4 solver.
  - Unmatched donors and all other edits stay unchanged.
  - Whether the source comes after the target is recorded.
- **Energy check:** total realized energy must match between current and oracle within 2%.
  Otherwise D2 is `D2_UNRESOLVED_ENERGY_MISMATCH`.
- **No-edit arm:** the B branch.
- **Damage:** at all valid EN-correct and ZH-correct positions, the change in log p of the
  baseline token and top-1 changes.
- **Pulse companions:** the first 20 relocations in plan order. For each, run a single pulse at
  the donor's position and one at the oracle target, both from the clean prefix, both with free
  continuation.
- **Acoustic annotation (diagnostic only):** at targets with a valid CTC midpoint, apply the
  unchanged LocalSupport provider to one evaluator-centred 1.0 s window, clipped to the heard
  audio. Report the timing error of the current window and E(oracle) − E(current). No searched
  window and no extra decoder arm.

## 6. Analysis (frozen)

- **Bootstrap:** dialogue resampling with 10,000 replicates and seed 240924. Positions are averaged
  within each dialogue, then across dialogues.
- **Primary family:** 11 contrasts, reported with Bonferroni-simultaneous two-sided 95% intervals
  (per-contrast α = 0.05/11).
  - `D1.L_plus`, `D1.L_minus`, `D1.L_plus_minus`, `D1.Lex_plus_minus`.
  - `D3.L_plus_random`, `D3.L_minus_random`, `D3.P_plus_random`, `D3.P_minus_random`.
  - `D2.L_oracle_current`, `D2.L_oracle_none`, `D2.P_oracle_current`.
  - These are on EN-confusion matched positions; the D2 contrasts are on D2 targets.
- **Secondary:** everything else, with pointwise 95% intervals and descriptive status. This covers
  all strata, span-initial vs within-span, ΔP_E/ΔP_M, flips, companions, damage and acoustics.
- **Equivalence resolution:**
  - D3 uses δ_p = 1/N (matched EN-confusion positions).
  - D2-B materiality uses δ_p,D2 = 1/N (D2 targets).

## 7. Decision rules (frozen; implemented in `inference_cf_p2r_analyze.decide`)

**Notation.** `pos(x)` means the simultaneous lower bound is above 0; `neg(x)` means the upper
bound is below 0. "Recovered" means the target POI is correct in a companion transcript.

**D1 (direction semantics):**

- **`DIRECTION_POSITIVE_SEMANTICS`** requires all of the following:
  - pos(L_plus);
  - pos(L_plus_minus);
  - pos(Lex_plus_minus);
  - EN-confusion companions do not recover fewer targets under +d than under −d.
- **`DIRECTION_SIGN_CONTRADICTION`** requires all of the following:
  - pos(L_minus);
  - neg(L_plus_minus);
  - neg(Lex_plus_minus);
  - no companion contradiction.
- **`DIRECTION_UNSTABLE`:** otherwise, if any of the four D1 contrasts excludes 0.
- **`DIRECTION_UNINFORMATIVE`:** otherwise.

A script-mass increase alone never satisfies these conditions, because Lex is required.

**D3 (generic perturbation)** is `GENERIC_SUPPORTED` only if G1, G2 and G3 all hold:

| Condition | Requirement |
|---|---|
| G1 | The random direction changes the processed top-1 at ≥ 3 matched D1 positions, or changes at least one companion transcript. |
| G2 | Neither pos(L_plus_random) nor pos(L_minus_random). |
| G3 | The simultaneous intervals of P_plus_random and P_minus_random lie inside [−δ_p, δ_p], **and** L_plus_minus and Lex_plus_minus contain 0. Nonsignificance is not equivalence. |

**D2 (localization):**

- **`D2-A`** requires all of the following:
  - pos(L_oracle_current);
  - pos(L_oracle_none);
  - the pointwise estimate of Lex_oracle_current is > 0;
  - oracle pulse companions recover ≥ current pulse companions.
- **`D2-B`:** the simultaneous upper bound of P_oracle_current is < δ_p,D2.
- **Otherwise:** unresolved, including an energy mismatch.

**Final diagnosis.** Each profile is evaluated. The primary diagnosis is the single satisfied
profile, and only when validity holds. Zero profiles, or more than one, give
**`MECHANISM_STILL_AMBIGUOUS`**.

- **`DIRECTION_ISSUE`** holds if either branch holds, **and** D2 ≠ D2-A:
  - D1 = SIGN_CONTRADICTION; or
  - +d moves script mass without lexical recovery, **while** a control shows a usable response:
    - the pointwise ΔP_E(+d) lower bound is > 0;
    - not pos(L_plus) and not pos(Lex_plus_minus);
    - and either pos(L_minus) or the pointwise L_random lower bound is > 0.
- **`LOCALIZATION_ISSUE`:** D1 = POSITIVE and D2 = D2-A.
- **`GENERIC_PERTURBATION`:** D3 = GENERIC_SUPPORTED and D2 ≠ D2-A.
- **`SITE_OR_SENSITIVITY_ISSUE`:** D1 = POSITIVE, D2 = D2-B, the +d top-1 correction rate at
  matched EN-confusion is < 10%, and every matched edit is nonzero. This is interpreted narrowly,
  as L16 at the P2 energy.
- **`CURRENT_MECHANISM_SUPPORTED`** requires all of the following:
  - D1 = POSITIVE and pos(L_plus_random);
  - D2 = D2-B;
  - the +d correction rate is ≥ 10%;
  - the pointwise lower bound of L_current_none is > 0;
  - D2 current net correction is > 0, counted as (newly correct targets) − (corrupted controls);
  - EN-correct L_plus is not pointwise-negative.

**Validity** (any failure means `MECHANISM_STILL_AMBIGUOUS`):

- All 80 rows are ok.
- Replay baseline-argmax identity and lineage hold in both passes.
- The gate schedule is identical across the current and oracle passes.
- The current replay equals the P2 steps on every compared step.
- Pulse restores are bitwise, with 0 pulse baseline mismatches.
- There are ≥ 40 matched EN-confusion positions across ≥ 12 dialogues.

## 8. Stopping, compute, repair policy

**Compute:**

- One Slurm job, batch 1, on H100 MIG 3g.40gb, with a 2-hour limit.
- A resumable row ledger. If the cap prevents completion, all rows are kept and the diagnosis is
  reported as incomplete; nothing is extended automatically.

**Reruns** happen only for recorded infrastructure or implementation invalidity. They go into
versioned directories and preserve the invalid attempt. There is never a rerun because a result is
unfavourable.

**Repair policy** (after diagnosis): at most **one** separately versioned hypothesis, following
Codex's conditional policy.

| Diagnosis | Allowed next step |
|---|---|
| Direction | Characterize first; never `−d` by default. |
| Localization | One causal, reference-free localizer/eligibility change; direction, site and dose stay frozen. |
| Generic | Stop tuning. |
| Site | One downstream-sensitivity diagnostic; no layer/α sweep. |
| Ambiguous | The single most informative follow-up only. |

No repair is executed automatically. With no fresh role, any repair is
`REPAIR_NOT_VALIDATED_NO_FRESH_ROLE`.

**Forbidden during P2-R:**

- Relabeling P2.
- Layer, α, gate, localizer, condition or direction searches.
- Random-seed selection.
- Oracle or reference information in a deployable API.
- Carrying oracle schedules across divergent histories.
- Treating probability or Latin mass as transcript recovery.
- Treating nonsignificance as equivalence.
- Any confirm, test, P3 or transfer data.
- `…_REPAIRABILITY.md`.
- Starting P3.

## 9. Reconciliation with the Codex contract (disclosed corrections, all made before outcomes)

1. **Target token.** A single token is ill-defined under Whisper's case- and space-specific
   vocabulary: references are case-normalized for evaluation, and spacing after Han varies.
   - **Correction:** the reference-consistent *set* of §3. It keeps Codex's exact-prefix,
     prefix-ID-preservation and distinguishing-beginning rules.
   - Positions where several reference words map to one token are excluded as ambiguous targets.
2. **Energy measure.** e\* comes from P2's hook-audited ‖steered − site‖, so matching uses that
   same measure. Codex's "downstream-consumed state" is measured and reported, and the audit
   compares the two.
3. **Solver.** Codex's geometric solve is kept. Because bf16 rounding would otherwise break the 2%
   tolerance, a secant refinement is added that uses **the hook's own arithmetic on the site
   state**. It never reads logits.
4. **D2 targets.** All valid EN-confusion positions in the 80 utterances, not only the 60 D1
   positions, so that relocation has destinations. The primary D2 outcome is over this frozen list.
   Targets where the direction falls back are *blocked*, not filled.
5. **Operationalized thresholds**, which the Codex text left open:
   - the G1 non-inertness counts;
   - the 10% correction-rate boundary that separates SITE from CURRENT;
   - D2-B materiality δ_p,D2 = 1/N_D2;
   - Bonferroni as the simultaneous method.
6. **Span stratification.** Added as *secondary, descriptive*, because pre-outcome inspection
   showed that most EN-confusion positions are inside translated spans. It is not a decision input.
7. **No-edit companion.** The no-edit continuation is the matched-baseline transcript. This is
   guaranteed by the bitwise restore check, so it is not re-decoded.
