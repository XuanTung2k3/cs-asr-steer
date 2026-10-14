# P2-1D — correction–damage diagnostic audit: result

**Verdict: `P2_1D_DIAGNOSTIC_COMPLETE`.**

This is an exploratory, post-hoc, CPU-only, observational audit of the existing P2-1 dev150 decodes.
- dev150 is exposed and also chose the P2-1 checkpoints.
- No GPU, model, decoding or training was used, and no locked role was opened.
- No P2-0/P2-1 protocol, freeze, configuration, result or verdict changed. `P2_1_CORRECTION_PRESERVATION_FAILURE`
  stands.
- All distances are reference-unit distances, not acoustic time. No causal claim is made.

## A. Executive conclusion

**Corrections.** R2's English gains are mostly **recovery of omitted English (33 repairs) and of wrong-language
renderings (15)**. Fine English → English lexical discrimination does not improve:
- 5 of 39 baseline substitutions are repaired;
- 5 new substitutions are created from correct units, and 5 more from former omissions.

**Mandarin damage.** It is real and highly concentrated:
- 73 new errors in 13 utterances and 9 dialogues; the top 3 utterances hold 63% and the top 3 dialogues 84%.
- 63 of the 73 are deletions. 55 of those lie in contiguous dropped spans of 5 or more units that start
  **immediately after an English unit**.

The dominant failure is a **switch-back omission**: after English, the following Mandarin span is skipped.
- Mandarin damage rate: about 3% within 7 units of English, 0.4% farther away, and 0.1% in Mandarin-only utterances.

**Damage vs repair.** Damage is mostly **not** coupled to an English repair in the same utterance: 70% (51/73) falls in
10 utterances with no English repair (dialogue bootstrap [0.33, 1.0]). It still occurs next to English already present.

**Robustness.** The finding survives an alternative alignment tie-break: 72 of 73 Mandarin corruptions are stable.
The epoch-3 MER regression is a single looping Mandarin-only utterance.

**Unproven:** whether the cause is a global English-continuation bias, a missing acoustic check, or exposure-bias in
free decoding.

**Highest-information next experiment:** `GO_P2_1b_PAIRED_CE_KL` (§I).

## B. Canonical reproduction (G1, G2): every check passes

| Check | Result |
|---|---|
| Pinned B0 beam-5 and greedy hashes; selected epochs R2 = 2, L1 = 3 | pass |
| `paired(B0, R2 epoch 2)` = `p2_1/report.json` = `p2_1/evaluation.json`, exactly (including bootstrap CIs) | pass |
| `metrics` = report (B0 and R2) | pass |
| 150 utterances / 20 dialogues / 97 CS | pass |
| English corrections / corruptions | 62 / 25 |
| New Mandarin errors / repairs; B0-correct non-filler Mandarin units | 73 / 103; 4,950 |
| Outside-target damage | 76 / 4,971 |
| MER (B0 → R2) | 0.11986 → 0.10771 |
| CS MER | 0.12272 → 0.10992 |
| Mandarin CER | 0.09622 → 0.08889 |
| Event table (one event per canonical reference unit) independently reproduces every count above | pass |
| Coverage exact: no duplicates, none missing | pass |
| Insertions separate; per-utterance Δmixed = Δwrong reference units + Δinsertions, all 150 | pass |

## C. Mandarin damage localization (D1)

**Concentration.**
- 73 new errors in 13 utterances (9 dialogues).
- Per affected utterance: 18, 15, 13, 7, 7, 4, 2, 2, 1, 1, 1, 1, 1. Median 2, 90th percentile 14.6, maximum 18.
- Top-1/3/5 utterance shares: 0.25 / 0.63 / 0.82.
- Per dialogue: 29, 25, 7, 4, 2, 2, 2, 1, 1, then 0 × 11. Top-1/3 dialogue shares: 0.40 / 0.84.

**By stratum** (B0-correct non-filler Mandarin units):

| Stratum | B0-correct units | New errors | Rate | Wilson 95% |
|---|---:|---:|---:|---|
| CS | 4,098 | 72 | 1.76% | [1.40, 2.21]% |
| Mandarin-only | 852 | 1 | 0.12% | [0.02, 0.66]% |

**By distance** (reference units). The dialogue-bootstrap intervals are wide because damage sits in 13 utterances:

| Distance bin | Rate by distance to nearest substantive English unit (n) | Rate by distance to nearest EN↔ZH switch (n) |
|---|---|---|
| 0–1 | 3.0% (364) | 3.1% (743) |
| 2–3 | 3.0% (701) | 3.3% (634) |
| 4–7 | 3.2% (974) | 2.5% (841) |
| ≥ 8 | 0.44% (2,059) | 0.37% (1,880) |
| no English / no switch | 0.12% (852) | 0.12% (852) |

**Operation.** Of the 73 new errors, 63 are deletions, 6 Han substitutions and 4 Latin substitutions. The 103 Mandarin
repairs came from B0 deletions (56) and substitutions (47).

**Span structure** (supplementary analysis, added after the main run and disclosed):
- The 63 deletions form 13 dropped runs.
- 55 deletions are in runs of 5 or more units, and 55 are in runs that begin right after a substantive English unit.
- 48 are internal skips and 15 tail truncations.
- Total insertions do not rise (B0 38 Han + 22 Latin; R2 30 + 19). The dropped spans are skipped, not replaced by
  English.

**Dialogue-level decomposition.** Mixed-error change = wrong-reference-unit change + insertion change, per dialogue.
The +30 dialogue is +30 wrong reference units (29 of them new Mandarin errors), no English change, and 0 from
insertions. The +21
dialogue is +20 wrong units and +1 insertion.

## D. English repair × Mandarin damage (D2)

| Cell (all 150) | Utterances | Dialogues | English corrections | English corruptions | New Mandarin errors | Mandarin repairs | Net mixed-error change |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN_REPAIR & ZH_DAMAGE | 3 | 3 | 10 | 5 | 22 | 2 | +16 |
| EN_REPAIR & no damage | 19 | 13 | 52 | 6 | 0 | 46 | −100 |
| no EN_REPAIR & ZH_DAMAGE | 10 | 7 | 0 | 2 | 51 | 1 | +52 |
| neither | 118 | 20 | 0 | 12 | 0 | 54 | −49 |

**CS only (97 utterances):**

| Cell | Utterances | Dialogues | English corrections | New Mandarin errors | Net |
|---|---:|---:|---:|---:|---:|
| EN_REPAIR & ZH_DAMAGE | 3 | 3 | 10 | 22 | +16 |
| EN_REPAIR & no damage | 15 | 11 | 44 | 0 | −93 |
| no EN_REPAIR & ZH_DAMAGE | 9 | 7 | 0 | 50 | +51 |
| neither | 70 | 19 | 0 | 0 | −42 |

**Severity.**
- With an English repair: damage per utterance is 18, 2, 2, then 0 × 19.
- Without an English repair (damaged utterances only): 15, 13, 7, 7, 4, 1, 1, 1, 1, 1.
- Damage-only utterances carry **70% of all damage** (dialogue bootstrap [0.33, 1.0]).

**"No English repair" is not "no English change".** 17 such utterances have English corruptions (14 units) or
wrong-to-wrong English changes (10 units).

**English net vs Mandarin net per utterance:**
- 94 of 150 utterances have both nets at zero;
- 9 improve both, 1 trades English gain for Mandarin loss, 7 lose Mandarin only;
- descriptive utterance correlation +0.32 (not a significance claim).

**Implication.** Most English gains come with no Mandarin damage (19 utterances, −100 net). Most damage comes in
utterances that gain no English. The cost is not a per-utterance trade-off. It is a separate failure mode located
around English.

## E. Epoch dynamics (D3; R2, beam-5; epoch 2 stays the selected checkpoint)

| R2 epoch | MER | CS MER | English WER / PIER | Mandarin CER | English corrections / corruptions | New Mandarin / repairs (retention) | Utterances improved / worsened | Dialogues improved / worsened | Loops / non-terminated | Training CE / gradient norm |
|---|---:|---:|---|---:|---|---|---|---|---|---|
| 0 (= B0) | 0.1199 | 0.1227 | 0.211 / 0.197 | 0.0962 | — | — | — | — | 0 / 1 | — |
| 1 | 0.1133 | 0.1158 | 0.174 / 0.157 | 0.0966 | 63 / 19 | 77 / 67 (0.984) | 38 / 22 | 14 / 4 | 0 / 1 | 0.396 / 0.355 |
| **2** | **0.1077** | **0.1099** | 0.180 / 0.164 | **0.0889** | 62 / 25 | 73 / 103 (0.985) | 47 / 22 | 15 / 2 | 0 / 2 | 0.360 / 0.382 |
| 3 | 0.1379 | 0.1114 | 0.177 / 0.161 | 0.1263 | 60 / 20 | 83 / 103 (0.983) | 49 / 24 | 14 / 5 | 1 / 3 | 0.347 / 0.378 |

**Paired transitions between epochs.**
- Epoch 1 → 2: English +7 corrections / +14 corruptions; Mandarin 5 new / **45 repairs**; ΔMER −0.0056.
- Epoch 2 → 3: English 12 / 9; Mandarin 15 new / 5 repairs; ΔMER +0.030.

**Reading.**
- English correction volume saturates at epoch 1 (about 60 corrections every epoch).
- The epoch-2 gain over epoch 1 comes from Mandarin repairs, not English.
- **Mandarin damage recurs at a near-constant level (73–83 units; retention 0.983–0.985)** at every epoch. It is a
  stable property of the adaptation, not an epoch-specific accident.
- Lower training CE does not track safer decoding.

**Epoch-3 loop.** One Mandarin-only utterance loops (+196 mixed errors). It accounts for more than all of the epoch-3
MER regression (163%; the rest improves).
- Descriptive-only MER with that utterance excluded: B0 0.1192, R2 epoch 2 0.1071, epoch 3 0.1078. Not an official
  score.
- That utterance had no damage at epoch 2.

**Greedy MER (secondary):** epochs 0 / 1 / 2 / 3 = 0.149 / 0.162 / 0.133 / 0.131.

## F. English lexical specificity (D4; substantive English units, B0 → R2 epoch 2)

| B0 category (n) | → correct | same category | → other wrong |
|---|---:|---:|---:|
| omission (89) | **33** | 49 | 7 (5 → lexical substitution) |
| translation-like (28) | **15** | 6 | 7 (5 → omission, 1 → lexical substitution, 1 → boundary) |
| boundary (26) | 6 | 19 | 1 |
| **English → English lexical substitution (39)** | **5** | 34 | 0 |
| segmentation artifact (37) | 2 | 35 | 0 |
| other (3) | 1 | 1 | 1 |

**Corruptions of B0-correct English (25)**, by category: 10 segmentation artifact, 8 omission, 5 lexical substitution,
2 boundary.

**Reading.**
- "Translation-like 28 → 6" overstates repair. Only **15** units were repaired. The other 7 left the category by turning
  into omissions (5) or other errors.
- English → English lexical errors: 5 repaired against 5 newly created from correct units plus 5 converted from
  omissions. The net is worse (39 → 45).
- R2's effect is language-preference and omission recovery, not finer lexical discrimination. This describes outputs
  only; it does not show that acoustic evidence was ignored.
- **L1 (diagnostic control):** 5 omission repairs and nothing else, with 0 corruptions.
- **R2 epochs 1 and 3** show the same structure (lexical substitutions repaired: 4 and 4; translation-like repaired: 16
  and 16).

## G. Alignment and decoding robustness (D5)

**Alternative minimum-cost alignment** (opposite tie-break; equal edit cost on all 150 utterances; sensitivity only):

| Events | Stable under the alternative |
|---|---|
| Mandarin corruptions | 72/73 (98.6%) |
| English corruptions | 24/25 (96%) |
| English repairs | 55/62 (89%) |

The damage finding is alignment-robust. About 11% of English repair attributions depend on tie-breaking. The aggregate
is unaffected.

**Ten most-damaged utterances.** All are CS and none loops at epoch 2.
- R2 hypothesis lengths are 0.17–0.58 of B0 for the top 6; the damage is deletions; insertions are unchanged.
- These are genuine dropped spans in free decoding, not scorer artifacts.

**Decoding instability.** It exists (the epoch-3 loop; non-termination 1 → 2 → 3 over epochs 0, 2, 3), but it does not
explain the selected-epoch damage.

## H. Mechanistic hypotheses (evidence, not conclusions)

| | Supporting | Contradicting | Unavailable | Distinguishing experiment |
|---|---|---|---|---|
| **H1 — global adaptation not conservative** | damage recurs at every epoch at constant retention (0.983–0.985); a systematic pattern (Mandarin dropped after English) across 9 dialogues; 70% of damage is unrelated to any English repair, as a global bias would produce | damage is concentrated in 13 utterances, not diffuse; net Mandarin improves (CER −0.007, 103 repairs) | whether teacher-forced preservation at baseline-correct positions constrains free-running span skipping | P2-1b: same R2 plus a fixed-weight KL-to-frozen term on baseline-correct gold-prefix positions, 3 seeds; does switch-back dropping fall while omission and translation recovery remain? |
| **H2 — selectivity is the main limit** | damage-only utterances carry 70% of damage; ORACLE_ONLY per-utterance choice gives MER 0.0944 with 2 new Mandarin errors | damage sits at the same switch points where corrections act, so selectivity would need sub-utterance precision; P1-D/P1-E found weak deployable WHEN signals; the oracle uses references | any deployable signal that predicts span-skip risk | a selectivity feasibility study (reference-free predictors of switch-back drops), only after H1 is tested |
| **H3 — acoustic lexical grounding insufficient** | English → English substitutions are not repaired (5 vs 10 created); gains are translation and omission recovery; dropped spans are acoustically present speech the decoder skipped | output taxonomy is not internal evidence; ACL-0's audio-conditioned editor added nothing over the state-only editor; nothing shows acoustic information is missing from h | acoustic counterfactuals (shuffle or removal) | P2-2 M1 vs R3 with acoustic-shuffle controls, if authorized later |
| **H4 — scoring or decoding artifact** | the epoch-3 regression is one loop; 11% of English repair events are tie-sensitive | 98.6% of Mandarin damage is alignment-stable; damage is real span deletion; integrity is exact | — | not needed as a primary next step (evaluation is sound) |

## I. Research decision (recommendation only; not authorized)

**Primary: `GO_P2_1b_PAIRED_CE_KL`.**

**Rationale.**
- The dominant, stable, alignment-robust failure is preservation (switch-back span omission). It recurs at every epoch
  and mostly occurs where R2 makes no English gain.
- A fixed-weight KL-to-frozen-backbone term on baseline-correct gold-prefix positions, including the Mandarin positions
  that follow English spans, is the smallest change that targets exactly that failure.
  - It reuses the frozen P2 infrastructure: about 25 minutes per run, 3 seeds.
  - It answers both open P2-1 questions: preservation, and seed stability of the CS gain.
- If KL fails to reduce switch-back drops while keeping omission and translation recovery, that failure is itself
  diagnostic: free-running exposure bias or the need for selectivity (H2).

**Suggested pre-registration.**
- Keep the R2 configuration.
- Fix λ_KL before running.
- Add switch-back Mandarin span deletions as a pre-declared secondary outcome.
- Judge with the existing P2-1 decision rules.

**Why not the alternatives first.**
- `GO_SELECTIVITY_FEASIBILITY_DESIGN`: the oracle advantage is large but references-based and unreachable as shown. The
  damage co-locates with correction sites, and deployable WHEN detection has been weak at this site (P1-D/P1-E). It is
  higher risk before a global constraint has been tried.
- `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING`: H3 is plausible for fine lexical errors, but it does not address the
  primary failure (preservation). P2-2 was conditional on a positive P2-1. Acoustic conditioning added nothing in
  ACL-0.
- `HOLD_FOR_EVALUATION_REPAIR`: evaluation is exact and alignment-robust.
- `INSUFFICIENT_EVIDENCE`: the damage pattern is clear, even if its cause is not.

## J. Reproducibility

**Code.** `reft/p2_1d_audit.py` (main run; `supplement` adds the deletion-span analysis, added post hoc and disclosed),
`tests/reft/test_p2_1d_audit.py`, `slurm/reft_p2_1d.sbatch`. The protocol, code and tests were committed (`40cc766`)
before the data run.

**Commands:**
- `sbatch slurm/reft_p2_1d.sbatch test`
- `sbatch slurm/reft_p2_1d.sbatch run`
- `sbatch slurm/reft_p2_1d.sbatch supplement`

**Tests:** 61/61 in the full CPU suite before the run; 14/14 P2-1D tests including the supplement test.

**Runtime:** main run 26.8 s CPU (Slurm 59243, 29 s wall); supplement a few seconds. No GPU, model load, decoding or
training.

**Inputs.** Hashes are in `p2_1d/manifest.json`. The pinned baselines, report, evaluation, selections, manifests and
every decode file were hash-checked.

**Outputs** (private; sha256 prefixes):

| File | sha256 |
|---|---|
| `analysis.json` | `38c11e8b…` |
| `integrity.json` | `3ccac670…` |
| `events_private.jsonl` | `53cef263…` |
| `utterance_summary_private.jsonl` | `85629fe1…` |
| `dialogue_summary_private.jsonl` | `827d364d…` |
| `analysis_supplement_deletions.json` | `01e83834…` |
| `manifest.json` | `7bd3ae45…` |

**Figures:** `p2_1d/figures/F1…F5` (anonymized aggregates, kept local).

**Location.** Private artifacts are under `/mnt/data/tungnx/cs-asr-reft/p2_1d/` and contain no IDs, transcripts or audio
paths in Git. The P2-0 freeze chain verifies unchanged (`p2_0_r1.json`), and the P2-1 report verdict is unchanged.
