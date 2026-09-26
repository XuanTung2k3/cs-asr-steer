> **Provenance (archived verbatim, 2026-09-25).** This is the Codex P2-R design contract. It was produced in the
> Codex session `~/.codex-persistent/sessions/2026/09/25/rollout-2026-09-25T09-42-13-01a0d7f1-….jsonl` (final
> message, read-only design task at remote HEAD `43a73c7`). It is the *input* to P2-R. The executable, frozen
> reconciliation is `P2_R_MECHANISM_DIAGNOSIS_SPEC.md`, which records every deviation from this text. The
> absolute links below point at the original checkout.

# P2-R MECHANISM DIAGNOSIS CONTRACT

## CURRENT EVIDENCE

**P2 status:** Valid, completed, and audited. Preserve its selection and results.

The live remote `cs-asr-steer-inf` HEAD is:

```text
43a73c735a19a719aec738cce479b9a458677344
```

The local HEAD matches. The local branch is named `feature/inference-cf-steering`; the working tree contains a pre-existing modification to `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md`. This audit uses the **committed canonical plan at the verified HEAD**. Nothing was modified or launched.

The inference-time project is explicitly separate from the older trained-controller project in `CODE_MAP.md`; their methods and evidence must not be conflated.

**Primary unresolved contradiction:** A finite, correctly applied conditioned displacement and a gate that ranks confusion cases do not produce convincing selective lexical repair.

| P2 condition | Net POI errors removed |
|---|---:|
| Selected `ER_B,+d` | 2 |
| Approximately energy-matched constant gate | 5 |
| `E` | 8 |
| `R_B` | 10 |
| `g_cf` | 2 |
| `ER_B,−d` | 11 |

These are descriptive comparisons, not an established ranking. In particular, the report’s stronger sentence about opposite-sign superiority should not become P2-R’s premise.

**What is valid:**

- Selection of L16, α=2, identity dose map, `ER_B,+d`, under P2’s frozen rule.
- Matched-baseline identity, intervention attribution, and computational validity.
- A very small observed development improvement: `2/2268`.
- Weak localization specifically on confusion and deletion cases.
- P2’s failure to demonstrate the intended mechanism. [P2 report](/home/tungnx/cs-asr-steer-inf/docs/inference_cf/P2_REPORT.md), [independent audit](/home/tungnx/cs-asr-steer-inf/docs/inference_cf/P2_AUDIT.md)

**What is unsupported:**

- Useful English-repair semantics of `+d`.
- Added value from `ER_B` selectivity.
- Superiority of `−d`, another gate, or generic perturbation.
- Generalization or readiness to execute P3.

**Recommendation:** Three diagnostic comparisons, implemented as **two computational blocks**: D1+D3 share controlled-position inference; D2 tests placement.

## HYPOTHESES

**H1 — Direction semantic failure:** The conditioned displacement represents prompt conditioning, script preference, or inconsistent lexical changes rather than useful repair. Sign reversal is one possible symptom.

**H2 — Localization/eligibility failure:** Useful interventions are suppressed or placed away from repairable decisions. Distinguish acoustic-window error, gate suppression, and structurally unreachable positions.

**H3 — Generic perturbation:** Corrections reflect nonspecific boundary crossing rather than information carried by the conditioned direction.

**H4 — Site/downstream sensitivity failure:** At L16 and the P2 operating energy, available state changes have insufficient useful influence on the lexical decision.

These explanations can coexist. Conflicting evidence must produce `MECHANISM_STILL_AMBIGUOUS`, not a forced single-cause diagnosis.

## DATA ROLES

**Diagnostic population:** Only the existing 300-utterance `D-dev-select` R2/P2 panel.

CPU inspection established:

- All **300 R2 baseline token sequences match P2’s `B0M_L16` sequences**.
- Before stricter reference-token validation, unique structurally eligible positions include **262 EN-confusion positions across 46 utterances and 17 dialogues**.
- Correct-English and correct-Mandarin controls are plentiful.

This supports a substantially smaller GPU panel.

**Repair-selection data:** `D-dev-select` retains its selection role. Its extensive historical exposure means P2-R can support exploratory causal diagnosis, not independent confirmation or generalization.

**Repair-validation population:** No untouched population is established by the ledger.

`router-calib` is a **conditional candidate for a later calibration/replication check**: 3,031 utterances across eight dialogues, with documented use of its 300 shortest utterances for DG-04 calibration. Before future use:

- Audit actual prior-use IDs and dialogue coverage.
- Freeze the proposed population without consulting repair outcomes.
- Preserve its calibration role unless a human explicitly authorizes a different role.
- Do not describe leftover utterances from exposed dialogues as independent, untouched validation.

The unselected remainder of `D-dev-select` is also not dialogue-independent confirmation. Training and construction roles have their own substantial exposure.

**Locked population:** No P2-R access to `D-dev-confirm`, `D-test`, P3 evaluation populations, or transfer outcomes.

`D-dev-confirm` already has historical exposure and cannot become repair-selection data. The exposure ledger also records historical DG-08 intervention exposure on `D-test`; it is therefore not globally untouched, while remaining prohibited for new development. Only this exposure fact is relevant here. [Data exposure ledger](/home/tungnx/cs-asr-steer-inf/docs/current/DATA_EXPOSURE.md)

## D1 — DIRECTION SEMANTICS

**Population:**

Freeze **60 positions per stratum**:

```text
EN-confusion
EN-correct
ZH-correct
```

Select deterministically using baseline/evaluator information only:

1. Round-robin across dialogues.
2. Within dialogue, order by SHA-256 of `P2R-v1 | utterance_id | position | stratum`.
3. Initially allow at most two positions per utterance per stratum.
4. Prefer correct controls from already-selected utterances.
5. Cap the union at **80 utterances**.

Target at least 15 dialogues per stratum. The minimum interpretable population is **40 valid positions across 12 dialogues per stratum**. These are coverage requirements, not a guarantee of statistical power.

Resolve eligibility and freeze IDs before intervention outcomes. If quotas cannot be met within the cap, report limited estimability; do not expand after seeing effects.

Oracle target positions come from evaluator alignment, explicitly labeled **offline diagnostic positions**.

**Reference-token validity:**

- Use the exact generated baseline prefix, never a gold prefix.
- Verify that the reference continuation can be tokenized without changing the existing prefix token IDs.
- Require an unambiguous next-token target.
- Preserve exclusions for ambiguous alignment, shared/non-distinguishing token beginnings, incomplete UTF-8, and undefined targets.
- A correct first subtoken is not equivalent to recovery of the complete English word.

Deletions remain a separately reported reachability population, not part of the primary direction test.

**Conditions:**

```text
Matched no-edit baseline
+d
−d
One deterministic random direction — shared with D3
```

Each intervention starts from the **same unedited cache and same baseline prefix**, with no previous intervention.

**Site:**

L16, exact decoder post-cross-attention residual, pre-FFN; canonical hook; NormPreserve; frozen model, bf16, eager attention, suppression rules, and cache semantics.

Keep forced-prefix exclusion. The first content token is not editable under the current contract.

**Dose matching:**

Remove gate magnitude from this experiment.

Freeze one target realized edit norm from existing P2 records:

\[
e_*=\sqrt{1203.3762345332195/949}\approx1.126.
\]

This is the selected P2 run’s RMS edit norm over edited steps—not a new tuned dose.

For each direction \(v\), solve geometrically for the smallest nonnegative scalar \(s\) satisfying:

\[
\left\|\operatorname{NormPreserve}(h+s v)-h\right\|_2=e_*.
\]

The solver may inspect states and realized edits, **never logits or correctness**. Different nominal scalars are energy calibration, not an alpha search.

Measure the actual downstream-consumed state after dtype rounding. Require pairwise squared-energy mismatch ≤2%; unmatched cases remain reported and cannot support a matched-energy comparison. Do not silently relax this tolerance.

**Metrics:**

Primary, on EN-confusion:

\[
L_i^a=\log p_i^a(y_{\rm ref})-\log p_i^0(y_{\rm ref}).
\]

Primary comparisons:

```text
+d versus baseline
−d versus baseline
+d versus −d
```

Co-report:

- Baseline and reference token IDs/text.
- Raw and suppression-processed distributions, clearly distinguished.
- Baseline top-1/top-2 margin.
- Reference-versus-baseline-token log odds.
- Reference-versus-best-competitor margin.
- Reference rank, with deterministic tie handling.
- `P_E`, `P_M`, ambiguous mass, and their changes.
- Top-1 changes and whether the changed token is correct.
- Absolute edit norm and relative squared energy.
- `||hE−hB||`, `cos(d,h)`, and cosine between requested direction and realized edit.

For reference tokens in the frozen English vocabulary partition, report:

\[
\Delta\log\frac{p(y_{\rm ref})}{P_E}.
\]

This distinguishes lexical evidence from an indiscriminate increase in Latin mass.

**Transcript companion:** Preselect 20 positions per stratum. After the single intervention, continue greedy decoding freely, with no further edits, to the original total 200-token cap. Report complete-word recovery, POI corrections/corruptions, and matrix damage. These are baseline-prefix-conditioned continuations, not end-to-end method performance.

**Primary decision rule:**

- **D1-A:** `+d` improves reference evidence and exceeds `−d`, with uncertainty excluding zero. Lexical specificity and continuation outcomes must not contradict the interpretation.
- **D1-B:** The corresponding evidence favors `−d`: direction sign/interpretation is suspect. **Do not adopt `−d`.**
- **D1-C:** Neither sign establishes consistent useful reference evidence: `hE−hB` remains unvalidated as a repair direction.

A script-mass increase alone cannot satisfy D1-A or D1-B.

## D2 — LOCALIZATION

**Population:** The same diagnostic utterances, with all valid baseline-prefix positions available for placement accounting. Primary outcomes use the frozen EN-confusion targets.

**Current condition:**

Recompute the frozen `E`, `R_B`, and `2E R_B` schedule on the matched baseline trajectory. Apply that schedule while feeding the **same baseline token sequence** to every condition.

This is a controlled baseline-prefix replay of current placement—not a replacement P2 free-decoding run.

**Oracle condition:**

Use an **energy-packet relocation intervention**:

1. Retain edits already assigned to oracle EN-confusion targets.
2. Identify positive-energy edits at other positions.
3. Order donor packets by descending realized energy, breaking ties by position.
4. Assign them to unedited, structurally eligible oracle confusion targets in the frozen hash order.
5. Move each packet once; remove its original edit.
6. Leave unmatched packets and other edits unchanged.

At the destination, use the same frozen direction rule:

\[
d_t=\operatorname{normalize}(h_t^E-h_t^B).
\]

Match each packet’s absolute realized energy using the D1 geometric solver. Match total realized energy within 2%; report relative-energy distributions because state norms differ across positions.

This preserves the edit count and energy budget. It does **not** claim that the transported gate value equals `E_tR_t` at its destination: overriding eligibility is precisely the evaluator-only manipulation.

“Same direction” means the same position-dependent direction field and sign—not transporting a vector constructed at a different position.

**Important controls:**

- The two arms share baseline prefixes, unsteered B/E states, site, decoder, direction rule, and packet energies.
- S-branch caches remain isolated and retain their own earlier interventions.
- Keep genuinely ineligible positions in the coverage ledger; do not bypass forced-prefix or malformed-prefix guards.
- Record moves from before/after each target. An edit occurring after a target cannot explain its immediate token probability.

**Acoustic localization annotation:**

At the frozen target positions, compare the current window with one evaluator-centered, boundary-clipped **1.0-second** window using existing valid CTC timing. Apply the unchanged LocalSupport provider to that single oracle crop.

Report timing error and changes in `E`. This is an acoustic diagnostic only; it adds no searched window and no additional decoder treatment arm.

This separates evidence of a bad acoustic crop from evidence that the resulting intervention schedule misses useful decoder positions.

**Metrics:**

Primary:

\[
L_{\rm placement}
=\operatorname{mean}\left[
\log p^{O}(y_{\rm ref})-\log p^{C}(y_{\rm ref})
\right].
\]

Also report oracle versus no edit, current versus no edit, reference margins, target energy coverage, correct-position damage, and source/destination energy accounting.

For a frozen subset of up to 20 relocated packets, additionally compare single-pulse current-location versus oracle-location free continuations. No subsequent oracle schedule is applied after trajectory divergence.

**Primary decision rule:**

- **D2-A:** Oracle placement improves reference evidence over current placement **and over no edit**, with supporting lexical/continuation evidence: localization contributes to failure.
- **D2-B:** A sufficiently precise upper bound rules out a material oracle rescue: localization alone does not explain failure.
- Otherwise: localization remains unresolved.

A nonsignificant oracle comparison is not D2-B.

## D3 — GENERIC PERTURBATION

**Required: YES.**

Opposite signs alone cannot distinguish meaningful conditioning from generic boundary crossing. D3 adds one direction to the existing D1 computation.

**Random construction:**

For each frozen position, generate exactly one Gaussian vector using a documented PRNG and seed derived from:

```text
SHA256("P2R-random-v1" | utterance_id | position)
```

Project it orthogonally to the baseline state and normalize. Freeze a deterministic numerical-degeneracy fallback. No seed selection, retries based on effects, or random-vector family.

**Energy matching:** Identical to D1, at identical states and positions.

**Metrics:**

- Paired reference log-probability contrasts: `+d−random`, `−d−random`.
- Reference probability changes.
- Lexical-within-English evidence.
- Reference margins, top-1 flips, corrections/corruptions.
- Distributional change and realized energy.

**Decision rule:**

`GENERIC_PERTURBATION` requires:

1. Random perturbations produce real lexical/output changes.
2. Neither conditioned sign establishes a specific lexical advantage.
3. Equivalence is demonstrated at a declared resolution—not inferred from nonsignificance.

Freeze the reference-probability equivalence margin as:

\[
\delta_p=1/N_{\rm confusion}.
\]

At 60 targets, this is 0.0167: one expected correct next-token draw across the diagnostic panel under categorical sampling. This supplies an interpretable diagnostic resolution; it is **not** a greedy-ASR efficacy threshold.

Require paired intervals for conditioned-versus-random mean probability effects to lie inside \([-\delta_p,\delta_p]\), with no contradictory sign-specific log-probability or lexical evidence.

All three being inert does not establish generic repair.

## COMPUTE PLAN

**Jobs:** One future Slurm GPU job, two blocks:

1. D1+D3: shared states, baseline, and three directions.
2. D2: current and oracle placement replays; bounded continuation companions.

Use one orchestration runner. Keep **batch size 1** unless numerical equivalence of another batching policy is separately established; P2 already demonstrated sensitivity to execution-path differences.

**Estimated conditions:**

- Up to 180 controlled positions × three nonzero interventions.
- Up to 80 utterances × two placement schedules.
- Up to 180 D1/D3 free continuations.
- Up to 40 D2 pulse continuations.
- Matched baselines and shared B/E computations.

**Reuse:**

- Saved baseline hypotheses, exact token IDs, evaluator alignments, provenance, and audio identities.
- Local-LID results only for identical waveform crops and provider/model/precision identities.
- States and caches generated within the new job, before intervention forks.

P2’s saved matched-baseline rows contain token sequences and identity flags, **not complete per-step gates, hidden tensors, or logits**. Selected-run steps are compact scalars. These artifacts cannot reconstruct D1 without model computation.

R2 transcript identity permits alignment reuse; it does not prove numerical identity of R2 replay logits or gates with the cached path.

Never reuse a state, gate, or cache across a changed prefix or intervention history.

**Maximum GPU workload:** One H100 MIG 3g.40gb allocation, **two GPU-hours**, 80 utterances maximum, original 200-token cap. Planning estimate: approximately **45–90 minutes**, extrapolated from P2 throughput, not a benchmark.

If the cap prevents completion, retain all records and report incomplete diagnosis. No automatic full-corpus rerun or result-driven extension.

**Uncertainty and frozen analysis:**

- Dialogue is the resampling unit; all paired conditions stay together.
- Average positions within dialogue, then average dialogues for primary mechanism quantities.
- Use 10,000 fixed-seed dialogue-bootstrap resamples.
- Freeze the complete primary contrast family before execution; use simultaneous 95% intervals for diagnostic decisions.
- Report pointwise intervals and distributions descriptively.
- Recompute transcript numerator/denominator counts within each resample.
- Coverage minima do not establish equivalence precision. Wide intervals end in ambiguity.

## FINAL DIAGNOSIS STATES

Apply these as evidence profiles. If incompatible profiles survive, return `MECHANISM_STILL_AMBIGUOUS`.

**DIRECTION_ISSUE:**

- D1-B establishes reverse-sign lexical advantage; or
- `+d` reliably produces script movement without useful lexical recovery, while a control demonstrates usable downstream response.
- Oracle placement does not explain away that failure.

**LOCALIZATION_ISSUE:**

- D1 supports useful `+d` semantics.
- D2 oracle placement improves target evidence over both current placement and baseline at matched energy.
- Continuations support recovery rather than merely Latin output.
- Record whether acoustic-window evidence, eligibility suppression, or both explain the placement failure.

**GENERIC_PERTURBATION:**

- D3 meets its nonzero-effect and equivalence requirements.
- Neither sign has a specific lexical advantage.
- D2 does not supply contradictory evidence for selective conditioned repair.

This conclusion is limited to the frozen random construction and diagnostic population.

**SITE_OR_SENSITIVITY_ISSUE:**

- Useful, direction-specific reference evidence exists, but realized changes are insufficient to overcome baseline lexical margins.
- Oracle placement provides no material rescue at the declared resolution.
- Nonzero site edits and downstream response are verified.

Interpret this narrowly as **insufficient useful sensitivity at L16 and the P2 operating energy**. These experiments cannot prove that L16 is intrinsically incapable of repair or distinguish every possible direction failure from site failure.

If every tested direction is inert and those explanations remain confounded, return ambiguity.

**CURRENT_MECHANISM_SUPPORTED:**

- `+d` has useful lexical advantages over both `−d` and random.
- Current placement itself improves target evidence and produces supported net correction.
- Oracle relocation has no material advantage.
- Correct-English and correct-Mandarin behavior does not contradict selectivity.

This supports the mechanism on exposed development diagnostics only. It does not overturn P2’s tiny end-to-end effect or authorize P3.

**MECHANISM_STILL_AMBIGUOUS:**

- Insufficient precision, poor target validity, energy mismatch, mixed causes, or conflicting probability/transcript evidence.
- No statistically significant difference is not enough to select another state.

## CONDITIONAL ONE-REPAIR POLICY

**If direction:** Permit one separately versioned direction-revision hypothesis only after characterizing whether the current displacement encodes sign reversal, script preference, or lexically inconsistent prompt movement. Do not automatically choose `−d`.

**If localization:** Permit one separately versioned localization/eligibility repair hypothesis. Preserve direction, site, dose rule, and other gate components. Choose acoustic-window repair only if the corresponding acoustic diagnostic supports it.

**If generic:** Stop tuning the current direction/gate. Any representation redesign requires a separate scientific decision.

**If site/sensitivity:** Permit one separately specified downstream-sensitivity diagnostic—not a layer or alpha sweep.

**If ambiguous:** Allow one follow-up addressing the unresolved contrast:

- For H1 versus H4 with inert interventions: measure the evaluator-only reference-margin Jacobian at the same site and states, including its tangent norm and alignment with realized edits. Do not apply the gradient as a steering direction.
- If uncertainty alone dominates: add a predeclared sample for that single contrast from the existing diagnostic role, without changing conditions.

Select only one follow-up after identifying the actual ambiguity.

## FORBIDDEN DURING P2-R

- Retroactively invalidating, relabeling, or reselecting P2.
- Layer, alpha, gate, localizer, language-condition, or direction searches.
- Selecting a winning random seed or adopting `−d`.
- Reference text, timing, target masks, or future tokens entering a deployable inference API.
- Carrying oracle schedules across divergent free-decoding histories.
- Treating reference-token probability, Latin mass, or first-subtoken correctness as transcript recovery.
- Treating nonsignificance as equivalence.
- Consuming confirmation, test, P3, or transfer outcomes.
- Creating or using the prohibited replacement-plan filename.
- Automatic repair execution or P3 progression.

## P3 STATUS

**HELD**

P3 remains untouched during and immediately after P2-R. Any progression requires a subsequent human scientific decision.

## EXACT INSTRUCTIONS FOR CLAUDE IMPLEMENTER

1. Treat this response as a diagnostic design, **not current authorization to edit or execute**.
2. On a later implementation request, verify remote HEAD and preserve the existing canonical-plan modification.
3. Build an additive diagnostic harness reusing `core_r2`, `core_p1`, cached branch semantics, and the canonical exact-site hook. Preserve P2 artifacts and production behavior.
4. Keep evaluator labels and oracle placement in an explicitly separate diagnostic interface.
5. Freeze IDs, target-token validity, random construction, energy solver, contrasts, bootstrap, and compute limits before intervention outcomes.
6. Write a complete provenance manifest covering resolved configuration, source/model/tokenizer hashes, environment, Git state, data identities, and completion status.
7. Verify matched-baseline identity, cache isolation, prefix exclusion, realized-energy matching, and complete exclusion accounting before interpreting effects.
8. Save compact sufficient statistics and bounded diagnostic tensors so analysis can be independently reproduced without another model run.
9. Report all three diagnostic comparisons and exactly one final diagnosis state, including ambiguity when warranted.
10. Stop after the P2-R report. Do not implement a repair or start P3 automatically.

P2-R_PROTOCOL_READY