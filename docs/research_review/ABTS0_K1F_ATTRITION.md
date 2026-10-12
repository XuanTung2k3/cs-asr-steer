# ABTS-0 K1-F — candidate attrition over the 39 reachable MM onsets (CPU-only, retrospective)

**K1 verdict is unchanged: `K1_INSUFFICIENT` (STRUCTURAL)** at commit `6856f5b7`. Before reading anything, this
diagnosis re-verifies the capture seal, the candidate seal and the gate hash. It writes only
`results/inference_cf/abts0_k1f/attrition.json`.

How it was produced:
- **Script:** `experiments/abts0_k1f_attrition.py`, an isolated reference-side script. The runner never imports it.
- **Labels:** the frozen `align_tokens` alignment of whole hypotheses to the reference, plus the approved A1 block label.
  No timing is used for correctness.
- **Unchanged:** no GPU job, no K1 scoring, no K2, and no change to any filter or threshold.
- **R0 values:** the R0 fractions shown are recomputed descriptively for every proposal. They are not new filters.

## 1. Attrition funnel (39 reachable genuine MM run onsets, 13 dialogues)

| Stage, in frozen pipeline order | Correct onset candidates left | Lost here |
|---|---:|---:|
| Reachable onsets | 39 | |
| Onset word present **anywhere** in the forced-EN decode | 25 | 14 absent |
| Forced-EN word **at that reference position** (localized correspondence) | 20 | 5 present but elsewhere |
| EN timing, heard, not already in B0 | 20 | 0 |
| Contiguous B0 Han block (≥ 0.5 coverage) | 19 | 1 |
| Onset within 200 ms | 12 | 7 (off by 220, 240, 240, 240, 500, 680, 860 ms) |
| Character boundary, t ≥ 1 | 10 | 2 at t = 0 |
| **R0 predicted-English fraction ≥ 0.5** | **0** | **10** (9 of them at R0 fraction 0.0) |
| Crops, eligible, earliest choice, A1-literal | 0 | — |

By cause:

| Cause | Onsets | Share |
|---|---:|---:|
| Generation (absent) | 14 | 36% |
| Generation (misplaced) | 5 | 13% |
| Localization | 8 | 21% |
| t = 0 | 2 | 5% |
| R0 region | 10 | 26% |
| Selection | 0 | 0% |

Headroom independent of filter order:
- Only **2 of the 20** localized correct onset proposals have R0 fraction ≥ 0.5 at all: "unvisited" (rejected on
  onset) and "they" (rejected at t = 0).
- For the 10 that did pass association (all later rejected by R0), the associated B0 block is A1-LITERAL at the onset
  in only 3 cases: friends, cool and new.
- So removing the 200 ms, t = 0 and earliest-pair losses entirely would still leave **at most 2 onsets** that pass
  R0, against the 8 the frozen gate needs.

## 2. Main failure cause

1. **The R0 region filter is the binding constraint on onsets.** Every onset that survived association was rejected
   because R0 does not mark the span as English. Examples (correct word, then B0's Han rendering, then R0 fraction):
   - `friends` → 们友, R0 0.0
   - `cool` → 酷, R0 0.0
   - `summer` → 天夏, R0 0.0
   - `new` (in "new singers") → 歌新, R0 0.0
   - `social` (in "social skills") → 技, R0 0.0

   The same filter is also the main safety barrier. Among MM proposals that reached it, it passed 18 of 55
   correct-genuine words (33%) but only 15 of 2,473 English-over-Mandarin words (0.6%). The 2,458 translations it
   rejected outnumber the 37 correct-genuine words it rejected by 66 to 1.

   *Interpretation (not established):* the onsets B0 renders as Han are exactly the spans where the model's own
   language evidence says "Mandarin". A decoder-derived English-region signal therefore fails on the same spans.
2. **Generation is the second limit.** For 19 of the 39 onsets, the forced-English decode never produces the word at
   that position. In 14 the word is absent entirely (`development`, `suitable`, `phone`, `knowledge`, `information`);
   in 5 it appears only elsewhere (`improving`, `explore`, `secret`).
3. **Localization is a minor factor.** The 200 ms tolerance loses 7. Four of those are 220–240 ms away, so a 400 ms
   tolerance would recover them, but only one of the four ("unvisited") passes R0, and its A1 label is ambiguous.

## 3. False positives and safety (the 17 MM chosen pairs; nothing was scored)

**Chosen pairs.** Labelled by what the English word aligns to:

| English word aligns to | Pairs |
|---|---:|
| Mandarin speech (a translation) | 10 |
| A genuine English word | 5 |
| A different English word | 2 |

By A1 label of the B0 block:

| A1 label | Pairs |
|---|---:|
| `EN_ON_MANDARIN_SPEECH` | 10 |
| `AMBIGUOUS_ALIGNMENT` | 3 |
| `WRONG_OCCURRENCE` | 1 |
| `WRONG_ENGLISH` | 1 |
| `LITERAL` | **2** ("my", "one") |

So pre-acoustic precision is 2/17 (12%), or 5/17 (29%) if you count English correctness only.

Typical translations:
- abroad ← 在国
- I ← 我觉得
- a ← 一个
- and ← 还有
- before ← 就是
- that ← 那个

**13 of the 17** chosen blocks contain Mandarin that B0 already got right. Steering at these targets would mostly
damage correct Mandarin.

**Correct candidates that do survive.** 11 correct-genuine words are eligible. None is a reachable onset; they are
later words in runs, such as "really" in "they don't really", or "agree". Earliest-pair selection keeps 5 of them;
each of the other 6 loses to an earlier eligible pair in the same utterance.

## 4. Recommendation: **stop ABTS-0 (forced-English onset transplant) for MM**

No single change to the existing pipeline gets close to the frozen requirement of 8 onset hits:
- With R0 kept: localized onset headroom is at most 2 of 39.
- With R0 relaxed: every correct-genuine word that reached the R0 stage would come with about 66 English-over-Mandarin
  proposals. Before any acoustic scoring, 13 of 17 chosen blocks already overwrite correct Mandarin.
- Widening onset tolerance or allowing t = 0: these add at most 1–2 candidates, all still blocked by R0.
- Changing the earliest-pair choice: it loses no reachable onsets.

Reopening this route would need language evidence that is independent of the decoder and actually marks these spans
as English. That would be a new method with its own preregistration, not a K1 adjustment.

## 5. Limits of inference

- **Small sample.** There are 39 onsets in 13 dialogues of one exposed panel (FULL300 D-dev-select), and the counts
  are small; no confidence intervals are attached. MIX/END and other datasets are not covered.
- **Alignment-based labels.**
  - Correctness comes from whole-hypothesis Levenshtein alignment to the reference.
  - Forced-English decodes of MM speech are largely translations, so "misplaced" and "absent" depend on that alignment.
  - Ambiguous cases are conservative: they count as not correct.
  - The CTC timing proxy is unused, so localization labels are lexical, not acoustic.
- **The acoustic rule was never run.** Nothing here says how well it would discriminate.
- **The R0 explanation is an interpretation.** The mechanism (shared language evidence between B0 and R0) is inferred,
  not tested.
- **Blinding is gone.** After this diagnosis, the K1 candidates on FULL300 are reference-inspected. Any revised
  candidate method evaluated on this panel is exploratory, not confirmatory. The sealed K1 verdict itself is
  unaffected.
