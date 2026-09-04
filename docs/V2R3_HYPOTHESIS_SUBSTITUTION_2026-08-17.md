# V2R3 hypothesis-substitution causal probe

**Date:** 2026-08-17  
**Job:** Slurm 41736, COMPLETED, exit 0, 5 min 11 s  
**Full destination:** /mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/hypothesis_substitution/generation_001  
**Smoke-only destination:** /mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/hypothesis_substitution/smoke_001

> **Arms B and C are oracle conditions for establishing causation. They are not
> practical mitigation methods, were not eligible for selection, and must not
> be presented as proposed methods.**

This run selected nothing, declared no gate, and changed no threshold,
tolerance, comparison operator, or criterion. It did not read or touch
D-dev-confirm or D-test.

## Plain answer

**No. The correction effect does not return when the hypothesis contains the
English token.**

Arm A reproduced Day 7 with 0/489 corrections. Arm B derived transport from the
complete gold transcript and also produced 0/489. Arm C inserted the reference
English unit into the first-pass hypothesis while leaving all original
first-pass characters unchanged and again produced 0/489.

The intervention corrected no target even though:

- the Arm-A “step exists but no English in the hypothesis” state was eliminated
  in B and C;
- delivered transport coverage increased from 57/489 in A to 147/489 in B and
  96/489 in C; and
- all 489/489 Arm-C insertions were well-defined and none was skipped.

Therefore the earlier no-English rate is a **correlate of failure, not its
cause**. English-token presence affects whether transport can be formed, but it
is not sufficient to restore a transcript correction. Something downstream of,
or orthogonal to, transport input is blocking the effect.

This is the third preregistered interpretation, not a tuned narrative. Every
correction count is below 20 events and is descriptive.

## Frozen design

The frozen action was recorded, not re-derived:

- Site D;
- decoder layer 16;
- rho = 1.

The experiment used the same 400 ms conservative-tier D-dev-select target
population as Day 7: 489 baseline-error English units in 116 utterances and 20
dialogues. The transport formula, final-decoder-layer all-head attention,
non-renormalized weights, enrichment abstention rule, and transport hit
tolerance were unchanged.

Only the hypothesis token sequence used to derive the English region and its
cross-attention changed:

| Arm | Transport input |
|---|---|
| A — baseline | generated first-pass hypothesis, exactly as Day 7 |
| B — gold-substituted | complete gold transcript; oracle causal condition |
| C — partial substitution | minimally edited first-pass hypothesis containing the target reference English units; oracle causal condition |

## Arm C construction

Construction was performed once per utterance for every target reference unit
in that utterance:

1. Normalize and segment the reference and first-pass hypothesis with the
   frozen MER/PIER unitization.
2. Compute the existing deterministic Levenshtein backtrace.
3. For a substitution, insert the reference English unit immediately before
   its aligned hypothesis unit. For a deletion, insert it at the current
   hypothesis-side alignment cursor.
4. Map that unit boundary back to a raw first-pass character boundary and
   insert the reference unit with separating spaces.
5. Preserve every original character of the first-pass output. In particular,
   do not replace or delete the wrong token.
6. Re-normalize and require the resulting units to equal exactly the original
   hypothesis units plus the requested insertions. Otherwise mark that target
   skipped and expose the reason.

Construction was well-defined for **489/489 targets (100%)** and skipped for
**0/489**.

| Error category | Well-defined / targets | Skipped |
|---|---:|---:|
| deletion | 290 / 290 | 0 |
| wrong-language substitution | 90 / 90 | 0 |
| same-language substitution | 77 / 77 | 0 |
| phonetic/transliteration | 17 / 17 | 0 |
| boundary error | 12 / 12 | 0 |
| other | 3 / 3 | 0 |

## Overall outcomes

Corruption is counted once per utterance over the 6,264 reference units that
were correct in the first pass. PIER, MER, and WER are corpus metrics over the
116 unique utterances. E_eff is mean delivered energy over all 489 target rows;
the active-only value conditions on non-abstention.

| Arm | Corrections / 489 | Corruptions / 6,264 | Corrected : corrupted | PIER | MER | WER | E_eff all | Active targets; E_eff active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 0 | 0 | 0 : 0 | 0.524914 | 0.285039 | 0.868394 | 2.0386 | 57; 17.4889 |
| B gold | 0 | 1 (0.0160%) | 0 : 1 | 0.524914 | 0.285154 | 0.868394 | 3.9999 | 147; 13.3058 |
| C partial | 0 | 2 (0.0319%) | 0 : 2 | 0.524914 | 0.285269 | 0.867876 | 2.6805 | 96; 13.6537 |

The active-only energy shows that the null is not explained by failure to
deliver energy. B and C delivered energy on substantially more targets than A
and still corrected none. The energy distributions differ because substituted
queries change attention mass; rho, direction, weighting, and abstention logic
did not change.

### Overall intervals

Intervals use 10,000-draw Rademacher wild cluster bootstrap with dialogue_id,
and t(G-1) critical-value reporting.

- Correction rate: A, B, and C are each 0.0000 [0.0000, 0.0000], G=20,
  n=489.
- Corruption rate:
  - A: 0.000000 [0.000000, 0.000000], G=20, n=6,264.
  - B: 0.000160 [-0.000058, 0.000377], G=20, n=6,264.
  - C: 0.000319 [-0.000081, 0.000720], G=20, n=6,264.
- Mean per-utterance PIER change relative to baseline:
  - A, B, C: 0.000000 [0.000000, 0.000000], G=20, n=116.
- Mean per-utterance MER change:
  - A: 0.000000 [0.000000, 0.000000], G=20, n=116.
  - B: +0.000059 [-0.000021, +0.000138], G=20, n=116.
  - C: +0.000238 [-0.000305, +0.000781], G=20, n=116.
- Mean per-utterance WER change:
  - A and B: 0.000000 [0.000000, 0.000000], G=20, n=116.
  - C: -0.000575 [-0.001355, +0.000206], G=20, n=116.

The zero correction intervals are degenerate because every observed outcome is
zero; they are not binomial upper confidence bounds and do not establish an
exactly zero population rate. All arm differences are far smaller than Day 6's
approximately +/-0.03 calibrated null-draw noise and must not be interpreted as
distinctions.

## Transport abstention and localization

The first five columns below form a target partition. “Delivered despite no
gold step” is shown separately because gold-substituted transport can localize
an earlier generated query even when the baseline decode ends before the
target's gold step.

| Arm | No step at all | Step exists, no English | English below 400 ms | Region found, not enriched | Step exists, localized | Delivered despite no gold step | Total delivered |
|---|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 84 | 161 | 122 | 65 | 57 | 0 | 57 (11.7%) |
| B gold | 84 | 0 | 162 | 135 | 108 | 39 | 147 (30.1%) |
| C partial | 84 | 0 | 214 | 95 | 96 | 0 | 96 (19.6%) |

For the two categories motivating this run:

- wrong-language substitution: Arm A has 73/90 = **81.1%** step-existing
  targets with no English in the hypothesis; B and C have 0/90;
- deletion: after excluding 83 targets with no gold step, Arm A has
  78/207 = **37.7%** with no English; B and C have 0/207.

Thus the substitution did exactly what the causal probe required. It removed
the diagnosed input absence and raised coverage, but it did not restore a
correction.

### Transport by primary error category

Entries are counts in the order:
no step / step-no-English / below-floor / not-enriched / step-localized.
The last column is total non-abstaining transport and may additionally include
the noted no-step transport in Arm B.

| Arm | Category (n; G) | Partition counts | Delivered |
|---|---|---:|---:|
| A | deletion (290; G=19) | 83 / 78 / 70 / 29 / 30 | 30 |
| A | wrong-language substitution (90; G=12) | 0 / 73 / 12 / 0 / 5 | 5 |
| A | same-language substitution (77; G=17) | 0 / 0 / 31 / 28 / 18 | 18 |
| B | deletion (290; G=19) | 83 / 0 / 71 / 61 / 75 | 114, including 39 no-step |
| B | wrong-language substitution (90; G=12) | 0 / 0 / 38 / 47 / 5 | 5 |
| B | same-language substitution (77; G=17) | 0 / 0 / 38 / 19 / 20 | 20 |
| C | deletion (290; G=19) | 83 / 0 / 106 / 46 / 55 | 55 |
| C | wrong-language substitution (90; G=12) | 0 / 0 / 55 / 24 / 11 | 11 |
| C | same-language substitution (77; G=17) | 0 / 0 / 37 / 19 / 21 | 21 |

## Primary-category outcomes

Category-wide corruption and corpus metrics use unique utterances containing at
least one target of that category. Category utterance sets can overlap, so their
denominators and corruptions must not be summed.

| Arm | Category | Corrections | Corruptions / at risk | E_eff all; active | PIER | MER | WER |
|---|---|---:|---:|---:|---:|---:|---:|
| A | deletion | 0/290, G=19 | 0/4,210 | 2.0758; 20.0659 | 0.550499 | 0.336144 | 0.862340 |
| A | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0213; 0.3832 | 0.781513 | 0.389051 | 0.935484 |
| A | same-language sub. | 0/77, G=17 | 0/3,520 | 4.4960; 19.2331 | 0.335017 | 0.169789 | 0.802495 |
| B | deletion | 0/290, G=19 | 1/4,210 | 6.5385; 16.6331 | 0.550499 | 0.336303 | 0.862340 |
| B | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0174; 0.3137 | 0.781513 | 0.389051 | 0.935484 |
| B | same-language sub. | 0/77, G=17 | 0/3,520 | 0.5963; 2.2957 | 0.335017 | 0.169789 | 0.802495 |
| C | deletion | 0/290, G=19 | 1/4,210 | 2.4251; 12.7869 | 0.550499 | 0.336303 | 0.861626 |
| C | wrong-language sub. | 0/90, G=12 | 0/867 | 0.0255; 0.2087 | 0.781513 | 0.389051 | 0.935484 |
| C | same-language sub. | 0/77, G=17 | 1/3,520 | 6.3370; 23.2357 | 0.335017 | 0.170026 | 0.802495 |

Every category correction rate is 0.0000 [0.0000, 0.0000] under the specified
bootstrap: G=19 for deletion, G=12 for wrong-language substitution, and G=17
for same-language substitution. These are again descriptive degenerate
intervals based on zero observed events.

The remaining 32 targets—17 phonetic/transliteration, 12 boundary errors, and 3
other—also have zero corrections in every arm. Their full category metrics,
energies, abstention counts, G values, and intervals are retained in the
authenticated summary artifact.

## Spillover

| Arm | Outside-target units changed | Improved | Damaged | Utterances with any | Per-utterance distribution |
|---|---:|---:|---:|---:|---|
| A | 0 | 0 | 0 | 0/116 | 116 zeros |
| B | 1 | 0 | 1 | 1/116 | [1, then 115 zeros] |
| C | 3 | 1 | 2 | 3/116 | [1, 1, 1, then 113 zeros] |

Affected utterances:

- B: ZH-CN_U1004_S0_221 — one damaged unit;
- C: ZH-CN_U0028_S0_120 — one damaged unit;
- C: ZH-CN_U1021_S0_68 — one improved unit;
- C: ZH-CN_U2004_S0_281 — one damaged unit.

Spillover is sparse and concentrated, but unlike Day 7 it is not uniformly
zero in the oracle input conditions. It does not include a target correction.

## Controls

No arm had a non-zero correction rate. Therefore neither the label-permuted
null nor the matched-energy random control was warranted under the
predeclared rule: when the treated direction produces no target correction,
direction controls cannot distinguish an absent transcript effect. Zero
control rows were generated.

This was decided mechanically after all three arms completed. No result was
used to tune the action, transport, abstention, or target set.

## Statistics and interpretation limits

- Wild cluster bootstrap: Rademacher weights, 10,000 resamples.
- Independent cluster: dialogue_id.
- Overall G=20; primary-category G values are stated with the estimates above.
- t(G-1) critical values are recorded in every interval object.
- All correction results and all one- or two-corruption results rest on fewer
  than approximately 20 events and are descriptive.
- Category corpus metrics use overlapping unique-utterance subsets and are not
  additive.
- PIER, MER, and WER interval entries are paired mean per-utterance changes;
  the displayed metric values are corpus micro-aggregates.
- Every observed B-minus-A and C-minus-A transcript-metric difference is much
  smaller than the Day 6 +/-0.03 null-draw noise calibration.

The result supports only this causal statement: **putting English into the
transport input is insufficient to recover transcript corrections under the
frozen mechanism.** It does not prove that transport input never matters, and
it does not identify the downstream blocker.

## Execution, identity, and immutability

The single job first wrote smoke_001 for 8 targets over 3 utterances,
authenticated its report, and recorded smoke_used_as_input=false. The smoke was
never promoted or consumed.

Frozen identities matched before smoke, between smoke and full execution, and
after the full run:

- Git HEAD: fdd441a75b0d66f56d8e5efec62f01649f7b9ae9;
- code/config snapshot:
  b9b950becc9cec2f572d53b46aa4a6ff7d58ddea5d5d4175438e1a92004045ba;
- test snapshot:
  61f8ba27ba19e2a2235d94fb4859d1742a2b5f62190b2de33970945d0ca75389.

Post-verification authenticated every new full artifact with identity required.
Arm A matched Day 7 exactly on all 489 paired rows for correction, abstention,
selected step, transport hit, effective energy, and steered text.

No existing artifact was overwritten, deleted, or modified. In particular,
the run did not write to production status/, freeze/, the exposure ledger,
existing candidate/role/POI artifacts, or any prior v2r3 generation. It
allocated or consumed no gate generation and changed no production Gate-A
state.

## New artifacts

Under hypothesis_substitution/generation_001:

- hypothesis_substitution_targets.parquet;
- hypothesis_substitution_utterances.parquet;
- partial_construction.parquet;
- hypothesis_substitution_summary.json;
- one authentication sidecar for each file.

There is no controls parquet because controls were not warranted.

## Verification performed

- 43 focused v2r3 tests passed before submission.
- The full suite collected 701 tests and ran without failures through 71%, then
  reached the explicit 900-second verification timeout; it is incomplete, not
  reported as a full-suite pass. The login host required preloading only the
  environment's compatible libstdc++ to avoid mutually incompatible default
  CUDA/SciPy loader paths; this changed no installed file or environment state.
- New module and tests passed Ruff.
- New module passed Python compilation.
- Submission script passed bash -n.
- Artifact/authentication dry-run loaded no model and wrote nothing.
- Slurm 41736 completed with exit 0.
- Smoke report passed identity-required verification.
- Every full artifact passed identity-required verification.
- Frozen code/test hashes matched at all three checkpoints.
- Arm A reproduced Day 7 exactly for n=489.
- D-dev-confirm and D-test were not read or touched.

No gate was evaluated or declared.
