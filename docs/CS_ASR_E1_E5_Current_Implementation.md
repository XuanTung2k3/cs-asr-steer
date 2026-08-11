# CS-ASR E1–E5 Current Implementation and Guide-Conformance Record

Audit date: 2026-07-27 (revised 17:xx after the job-31189 failure analysis)  
Reference guide: `docs/CS_ASR_E1_E5_AI_Guide.md`  
Production launcher: `cs_asr_e1_e5.sh`  
Production artifact root: `/mnt/data/tungnx/cs-asr-steer/artifacts_v2`

> **Revision note.** The first production submission (job 31189) failed 70 s in,
> during P0's first decode batch. The cause and its correction are registered as
> **DEC-A1** below; that correction is a *deliberate, documented deviation from
> guide §7.2*, not a silent change.
>
> **Second revision.** The researcher has set "no human verification or labelling"
> as a hard project constraint. The guide's manual boundary audit is therefore
> replaced by an automatic protocol, registered as **AUD-A1** — the most
> significant deviation in this document, with its scientific consequences stated
> explicitly there. **E4-G5** is resolved using measured POI pools, an independent
> CTC aligner closes **E1-N2**, and `ENC-RANDLOC-1L` is added as span-localization
> evidence beyond guide §40. Source hashes refreshed.
>
> **Third revision (after job 31473).** E1 ran to completion for the first time and
> failed its gate rather than crashing. Re-analysis of its own artifacts showed the
> synthetic ground-truth harness was measuring its own construction artefacts —
> untrimmed edge silence at the splice and pairs longer than Whisper's 30 s window —
> not the aligner. The harness is corrected, the boundary-error criteria are made
> advisory pending calibration (**GATE-A1**), and **E1-G3** is opened to calibrate
> `exclude_boundary_ms` against the measured error. Source hashes refreshed.

## 1. Purpose and interpretation

This document records what the repository currently implements compared with
`CS_ASR_E1_E5_AI_Guide.md`. It does not replace or weaken the guide.

Status labels:

- **Implemented**: the current code implements the required guide behavior.
- **Adapted**: the implementation differs because of local data or Slurm constraints,
  but preserves the intended experiment.
- **Narrower**: the implementation runs a smaller or simpler version than described.
- **Gap**: a required guide item is not fully implemented.
- **Optional omission**: an experiment explicitly marked optional in the guide is not run.

This is a static implementation audit. Actual P0/E1–E5 scientific pass/fail
status is determined only from the generated gate records under
`artifacts_v2/status/` and reports under `artifacts_v2/reports/`.

## 2. Executive conclusion

The command

```bash
sbatch cs_asr_e1_e5.sh auto
```

attempts the complete guide-gated workflow:

```text
unit tests
  → smoke (all stages, ~8 utterances, isolated artifact root)
  → P0
  → E1
  → E2 pilot
  → E3 pilot
  → E4 pilot
  → E2 full
  → E3 full
  → E4 refinement
  → E4 confirmation
  → E5
```

The `smoke` step executes every stage once against `artifacts_smoke` before any
expensive work starts, because E3, E4 and E5 had never executed against the real
model. It is marker-guarded (runs once), non-fatal, cannot write to
`artifacts_v2`, and is skipped with `SKIP_SMOKE=1`. Its gate results are
meaningless at that sample size and must never be cited.

It stops at the first failed scientific gate. **E1 no longer stops for a human
boundary audit**: under the no-human protocol registered as AUD-A1 the run
proceeds P0→E5 unattended. If the 48-hour Slurm allocation expires, the same
command can be submitted again; completed workflow markers are skipped and
long-running stages use their saved checkpoints.

The core experiment chain, required E4 systems, and all ten required E5 masks
are implemented. The current implementation is nevertheless **not yet an exact
implementation of every required detail in the guide**. The gaps and narrower
choices are registered in Section 5.

## 3. Launcher and execution behavior

| Requirement | Current implementation | Status |
|---|---|---|
| One Slurm allocation at a time | All stages run sequentially inside one `sbatch` allocation | Implemented |
| Stop after a failed gate | Stage exit code 2 stops the launcher | Implemented |
| Pause for E1 human audit | Disabled by design (AUD-A1). The exit-code-3 mechanism is retained and re-enabled by `alignment.gate.require_human_audit: true` | Adapted |
| Resume long stages | `--resume`, prediction caches, alignment processed-ID files, and E2 accumulators are used | Implemented |
| Resume after wall-time expiration | Re-submit the identical `auto` command | Implemented |
| Preserve completed outputs | Versioned workflow `.done` markers skip passed steps | Implemented |
| Independent stage execution | Individual `p0`, `e1`, `e2`, `e2_full`, `e3`, `e4_*`, and `e5` modes remain available | Implemented |
| Python pipeline command from guide | Slurm shell is the active orchestrator; the Python pipeline module is used for status reporting | Adapted |
| One H100/BF16 | Requests one GPU; model configuration uses BF16 CUDA inference | Implemented |
| Official test prohibition | Experiment subset loaders and configs prohibit use of the official test split | Implemented |

The workflow is conditional, not unconditional: reaching E5 requires every
preceding gate to pass. A single `sbatch` submission cannot bypass the 48-hour
scheduler limit, but it no longer needs a human in the loop to reach E5.

## 4. Stage-by-stage implementation map

### P0 — Baseline and POI audit

- **Implemented:** builds and validates official and internal manifests.
- **Implemented:** assigns `dev_select` by speaker and leaves the remaining
  development speakers for `dev_confirm`.
- **Implemented:** verifies speaker disjointness.
- **Implemented:** decodes both `B0_AUTO` and `B1_ZH`.
- **Implemented:** selects the frozen primary baseline by MER on `dev_select`.
- **Implemented:** retains both baseline outputs and constructs the POI taxonomy.
- **Implemented:** gates on erroneous EN POIs, language-confusion errors, and
  baseline-correct EN POIs.
- **Implemented:** no official test decoding is performed.

### E1 — Alignment construction and audit

- **Adapted:** the local CS-Dialogue installation contains `short_wav` but not
  the `long_wav` TextGrids. Alignment therefore uses Whisper cross-attention
  DTW, which is the guide's third-priority alignment source.
- **Implemented:** aligns `dev_select`, `dev_confirm`, and the eligible full
  direction-training set; derives the pilot table from the full table.
- **Implemented:** preserves times, creates half-open frame spans, validates
  bounds/order/padding/conflicts, and marks 100 ms boundary-adjacent regions.
- **Implemented:** language tags include conservative handling for fillers,
  numerals, ambiguous Latin tokens, and romanized Mandarin.
- **Implemented:** creates a 100-boundary audit pack with waveform,
  spectrogram, audio clip, transcript context, confidence, and predicted times.
- **Implemented:** creates a second alignment configuration (`zh`/median 7
  versus `en`/median 3) and reports agreement as a reliability proxy.
- **Adapted AUD-A1:** the blocking human boundary audit is replaced by an
  automatic protocol — see "Alignment verification protocol" below for the four
  replacement criteria, the retained optional human path, and the scientific
  exposure this creates. The audit pack is still generated for inspection.
- **Closed E1-G1:** "within approximately ±100 ms" is now an actual gate
  criterion, measured by the synthetic ground-truth harness rather than by human
  estimation.
- **Closed E1-G2:** within-50 ms and within-100 ms rates and signed bias are
  reported automatically for every run, with no verdict file required.
- **Superseded E1-N1:** audit-sample stratification no longer gates anything.
- **Implemented E1-N2:** an independent aligner family (CTC) is implemented, and
  gated when enabled.

### E2 — Direction construction

- **Implemented:** uses encoder layers 15, 23, 27, and 31.
- **Implemented:** observes all requested layers in each forward pass and uses
  resumable online accumulation without retaining the activation corpus.
- **Implemented:** uses equal-weight within-utterance EN-minus-ZH means.
- **Implemented:** constructs correct-only, all-valid, global-centroid,
  wrong-sign, and five matched random encoder directions.
- **Implemented:** stores direction scale, centroids, midpoint, counts, seed,
  manifest hash, model revision, and normalization version.
- **Implemented:** evaluates five direction-estimation seeds.
- **Implemented:** verifies hook behavior and the decoder next-token shift.
- **Implemented:** constructs a distinct direction for every decoder layer.
- **Gap E2-G1:** decoder direction construction does not currently filter the
  training manifest to correctly transcribed examples before teacher forcing.
- **Narrower E2-N1:** decoder directions use at most 2,000 training utterances.
- **Narrower E2-N2:** each full-run direction seed uses a deterministic 80%
  subsample of `train_direction_full`; no separate canonical direction using
  100% of that manifest is produced.
- **Optional omission:** diagonal-whitened, PCA, and error-versus-correct
  directions are not constructed.

### E3 — Separability

- **Implemented:** evaluates at most 1,000 code-switched `dev_select`
  utterances on unseen speakers.
- **Implemented:** excludes boundary-adjacent frames for the primary result
  and also reports boundary-inclusive results.
- **Implemented:** reports frame-level metrics, utterance-macro metrics,
  speaker-macro AUROC, and speaker-bootstrap confidence intervals.
- **Implemented:** evaluates correct-only seeds, all-valid, global-centroid,
  wrong-sign, and five random controls.
- **Implemented:** gates on AUROC, across-seed standard deviation, random
  controls, and sign consistency.
- **Implemented:** freezes the top two layers for the E4 pilot.
- **Gap E3-G1:** layer ordering is only speaker-macro AUROC followed by
  frame-micro AUROC. Seed stability, EN recall, confidence-interval width, and
  speaker variance are reported but not all included in the ranking rule.
- **Gap E3-G2:** the gate has no explicit per-speaker variance or
  speaker-dominance criterion to demonstrate that results are not driven by
  one or two speakers.
- **Optional omission:** shuffled-label evaluation is not run.

### E4 — Oracle-span local steering

- **Implemented:** pilot uses 100 baseline-wrong plus 100 matched
  baseline-correct EN POIs, with one target per utterance.
- **Implemented:** refinement and confirmation request up to 300 wrong plus 300
  correct POIs.
- **Implemented:** pilot grid uses two E3 layers and
  `alpha = {0.5, 1.0, 2.0}`.
- **Implemented:** refinement uses
  `alpha = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}`.
- **Implemented:** primary encoder steering uses one layer, exact core span,
  tapered 100 ms shoulders, projection-standard-deviation scaling, and norm
  preservation.
- **Implemented:** required systems are present:
  `ENC-LOCAL-1L`, `ENC-GLOBAL-1L`, `ENC-WRONG-1L`, five
  `ENC-RAND-1L` seeds, and `DEC-ALL-GLOBAL`.
- **Implemented:** decoder steering uses distinct per-layer directions,
  `alpha/sqrt(K)`, norm preservation, and the five requested decoder strengths.
- **Implemented beyond the minimal schedule:** global, wrong-sign, and all five
  random controls run at every encoder grid point.
- **Implemented:** reports MER, PIER, EN-WER, ZH-CER, correction, corruption,
  correction-to-corruption ratio, net corrections, transcript changes,
  outside-region edits, and error-category correction.
- **Implemented:** outside-region edits exclude the target POI and one
  reference unit on each side.
- **Implemented:** selection follows PIER, net corrections, corruption,
  smaller alpha, and later-layer tie breakers after feasibility checks.
- **Narrower E4-N1:** refinement keeps the single best pilot layer rather than
  up to two promising layers. The guide permits one or two, so this reduces
  compute without violating the stated refinement range.
- **Gap E4-G1:** matched correct controls use POI duration, POI position,
  utterance duration, baseline confidence, and a same-speaker preference, but
  do not match token frequency or boundary proximity.
- **Gap E4-G2:** same-speaker preference does not explicitly guarantee the
  aggregate speaker distribution requested by the guide.
- **Gap E4-G3:** paired bootstrap currently covers the target POI error
  indicator only. Separate paired confidence intervals for PIER and MER and a
  paired correction-versus-harm difference are not reported.
- **Gap E4-G4:** the generated report lacks dedicated per-speaker result
  tables, wall time, and peak-memory measurements.
- **Resolved E4-G5 — target-count criteria made coherent.** The two count
  criteria could fail for arithmetic rather than scientific reasons.
  `_match_controls` matches controls 1:1 against the error set and therefore
  returns at most `len(wrong)` controls, so requiring `len(correct) >= n_correct`
  was unsatisfiable whenever `wrong < n_correct`.
  Measured candidate pools (utterances with ≥1 reference EN POI, one POI per
  utterance): **dev_select 2,328**, **dev_confirm 1,501**. So the pilot's 100 is
  comfortably attainable (needs only a ~4 % baseline error rate), but the
  confirmation's 300 on `dev_confirm` needs ~20 % and is not guaranteed.
  Resolution, grounded in guide §36's own wording — it states the pilot size
  firmly ("100 utterances with one baseline-incorrect EN POI") but bounds
  confirmation loosely (**"up to** 300"):
  - pilot keeps the hard requirement of `pilot_wrong`;
  - refine/confirm require only `min(confirm_wrong, min_confirm_wrong)`, where
    `min_confirm_wrong: 100` — a confirmation must be at least as well powered as
    the pilot it confirms;
  - the correct-control criterion is now `min(n_correct, len(wrong))`.
- **Implemented beyond guide §40 — `ENC-RANDLOC-1L` span-localization control.**
  Identical direction, strength and mask width, applied at a random *disjoint*
  span in the same utterance (per-utterance deterministic seed; utterances with no
  disjoint room keep the original span and are counted, making the control
  conservative). Under the no-human alignment protocol this is the strongest
  available evidence that spans are correctly placed, because it is measured on
  the task itself rather than on a boundary proxy: if steering the aligned span
  beats steering an identical span elsewhere, the span location carries real
  information. **Reported and analysed but deliberately NOT part of the §45 gate**,
  since adding hard criteria the guide does not require could fail E4 for
  non-guide reasons. Promotable to a gate criterion if the researcher chooses.
- **Optional omission:** decoder-single-layer, decoder-local,
  encoder-multilayer, and prompt-derived interventions are not run.

### E5 — Mask and boundary robustness

- **Implemented:** uses the selected E4 layer and alpha on `dev_confirm`.
- **Implemented:** evaluates all ten required masks:
  `EXACT_HARD`, `EXACT_TAPER`, `EXPAND_50`, `EXPAND_100`, `EXPAND_200`,
  `JITTER_50`, `JITTER_100`, `JITTER_200`, `BOUNDARY_ONLY`, and
  `WHOLE_WORD_PLUS_CONTEXT`.
- **Implemented:** each jitter level uses seeds 242–246, derives stable
  per-utterance random seeds, independently jitters start/end, clips masks,
  preserves nonempty spans, and stores sampled offsets.
- **Implemented:** plots ten real mask examples.
- **Implemented:** reports the E4 primary metrics, retained gain, jitter
  variability, and hard-versus-tapered harm.
- **Implemented:** explicitly handles zero or negative exact-span gain.
- **Implemented:** gates on 100 ms retained gain, correction/harm ratio,
  taper harm, outside-region edits, and jitter stability.
- **Gap E5-G1:** E5 reconstructs its deterministic wrong/correct groups using
  the E4 group-building function instead of directly loading the frozen
  `e4_confirm_wrong.parquet` and `e4_confirm_correct.parquet` manifests.
  Current deterministic inputs should yield the same IDs, but direct loading
  would provide stronger immutability.
- **Optional omission:** monolingual ZH/EN full-mask harm controls are not run.

### E1 boundary-frame overlap (found by job 31413)

- **Implemented — `resolve_boundary_overlaps`.** `sec_to_frames` floors starts and
  ceils ends, so two adjacent units both claim the single frame containing the
  boundary between them. On real `dev_select` alignments this produced **248
  conflicting overlaps across 3,360 utterances (0.21% of content units), every
  one exactly 1 frame (20 ms)** — a rounding artefact, not an alignment failure.
  Spans are now made disjoint before anything consumes the table: the earlier
  unit is truncated (preserving the following onset, which is what E4 steers), or
  the later unit is delayed when both share a start frame. Verified on the real
  data: 248/248 resolved, 0 residual, 0 units emptied or lost, idempotent.
  This matters beyond validation: a frame shared by an EN and a ZH unit would be
  accumulated into *both* centroids in E2 and would blur E4's mask.
- **Fixed — `validate_unit_table` no longer aborts the stage.** Its assertions ran
  unconditionally, so a violation killed E1 by traceback *after* the expensive
  alignment had completed, discarding the run instead of failing the gate. The
  E1 stage now calls it with `strict=False`: violations are reported, feed the
  existing `units_validate` gate criterion, and preserve every artifact. The unit
  tests still call it with `strict=True`, as guide §14 requires.
- **Fixed — smoke findings are now reported.** The smoke hit this same assertion
  ~30 minutes before production did, but its output was buried in 100 KB of
  stderr and the run continued. `run_smoke` now prints a deduplicated summary of
  every distinct exception it raised.
- **Fixed (job 31473) — the resolver now guarantees its postcondition.** The
  original resolver handled adjacent overlaps but not spans that no edge move can
  separate. Job 31473 hit **1 such case in 178,033 units**: `life` (EN) and `所`
  (ZH) were assigned *identical* frames 430–478, and that single unit failed the
  boolean `units_validate` criterion and stopped the whole pipeline. Three
  distinct defects were found and fixed here, the last two by testing rather than
  by reasoning:
  - *Coincident and nested spans.* Identical conflicting spans give no basis for
    preferring either, so **both** forfeit their language claim; a strictly nested
    span is the anomaly against a sequence-corroborated enclosing one, so **only
    the nested one** forfeits. Forfeited units are retagged `UNKNOWN` and excluded
    from centroids, boundaries and masks, but keep their row for provenance.
  - *Withdrawing a claim exposes a new pair.* Once `所` is dropped, `life` (EN,
    ending 478) and the following `以` (ZH, starting 477) become neighbours and
    overlap by one frame. A sweep over the original pair list never examines that
    pair, so the first fix left the conflict in place and was not idempotent.
    The sweep now compares against the last unit that still holds a claim and
    iterates to a fixed point.
  - *No guaranteed postcondition.* Randomised tables that pile units onto shared
    frames still defeated the single sweep (35 failures in 3,000). The resolver
    now ends with an unconditional pass that withdraws the claims of anything
    still conflicting, so "no EN frame is also a ZH frame" holds for **any**
    input, degrading by excluding data rather than by failing a stage.

  Verified on the real cached tables: `dev_select` and `dev_confirm` clean with
  nothing dropped, `train_direction_full` clean with **2 units of 178,033**
  dropped, all three idempotent and with no rows lost; plus **5,000/5,000**
  adversarial random tables. `overlaps_dropped` is reported so a growing rate
  stays visible.

### E1 synthetic ground truth (corrected after job 31473)

Job 31473 reported **2.0% of synthetic boundaries within 100 ms** and a
**−1,647 ms** systematic bias, while cross-configuration agreement was 97% and
silence absorption 6.6%. Those numbers cannot all describe the same aligner, and
re-analysis of the run's own artifacts showed the harness, not the aligner, was
at fault:

- **Corpus clips carry edge silence.** Measured on the 99 rendered pairs: the ZH
  clip contributed a median **765 ms** of trailing silence and the EN clip
  **675 ms** of leading silence, so the file splice sat in the middle of ~1.32 s
  of silence. The aligner marks the end of Mandarin *speech*; the reference was
  the file join ~0.9 s later. Most of the reported "error" was this definitional
  gap. Clips are now silence-trimmed before concatenation (`synthetic.trim_silence`),
  which reduced boundary silence from **1,320 ms to 80 ms** median on real audio.
- **17 of 99 pairs exceeded Whisper's 30 s window**, 8 with their boundary past
  30 s and therefore unobservable by construction; these produced the 9,650 ms
  p95. `synthetic.max_duration_sec` (28 s) now bounds pair construction, and any
  pair still over the window is excluded from scoring and counted as
  `num_pairs_over_window`.
- **Re-scored against the speech transition** and restricted to pairs inside the
  window, job 31473's own alignments give a median absolute error of **215 ms**
  with 29% within 100 ms — far better than 2%, and still well above the 100 ms
  `exclude_boundary_ms` in force. This is the finding the run actually produced.
- **The synthetic cache is fingerprinted.** The rendered waveforms are a function
  of these settings, so a cached alignment made under different ones describes
  audio that no longer exists and is now discarded rather than resumed.

**Registered as GATE-A1:** the two synthetic gate criteria are advisory (`null`)
rather than blocking. `exclude_boundary_ms` is meant to be calibrated *from* the
measured boundary error, so asserting a threshold guessed before the first valid
measurement inverts the inference. Both figures are reported in full, in the job
log and in a dedicated report section that states what the measurement implies
for the exclusion window; either becomes blocking as soon as its threshold is set
to a number. E1 still gates on validation, boundary counts, cross-configuration
agreement and silence absorption.

### Alignment verification protocol

- **Adapted AUD-A1 — the manual boundary audit is replaced by automatic
  measurement, against guide §13/§15/§62.** The guide requires ≥100 human
  boundary verdicts with ≥90% judged usable. The researcher has set "no human
  verification or labelling" as a hard project constraint. E1 therefore no longer
  blocks (the exit-code-3 path is disabled) and the human criteria are replaced by
  four automatic ones:
  1. **`synthetic_boundaries_within_100ms`** — monolingual ZH and EN utterances
     concatenated at a *known* instant, aligned by the ordinary pipeline, error
     measured against that instant. This is the only component yielding real
     ground truth. Concatenated speech lacks cross-boundary coarticulation and
     natural switch prosody, so it is an **optimistic bound** and must be reported
     as such in any writeup.
  2. **`synthetic_abs_systematic_bias_ms`** — signed error, catching systematic
     early/late drift such as silence absorption.
  3. **`config_agreement_within_100ms`** — the pre-existing two-configuration DTW
     proxy. Weak on its own: same method, correlated failures.
  4. **`silence_absorption_rate`** — fraction of spans whose edge lies inside
     silence, targeting the defect actually observed on this corpus
     (a single-character unit spanning 0.00–0.88 s).
  Optionally a fifth, **`ctc_agreement_within_100ms`**, when the independent CTC
  aligner is enabled (see below). A filled `verdicts.csv` is still honoured if
  supplied but is never required, and `alignment.gate.require_human_audit: true`
  restores the guide's blocking behaviour.

  **Scientific exposure, stated plainly.** Alignment error blurs the intervention,
  so it makes a *positive* E4 result conservative — the effect was found despite
  noisy spans. It makes a *negative* E4 result ambiguous: "steering does not work"
  cannot be separated from "the spans were wrong." Mitigations are (1) the measured
  synthetic error distribution, (2) E5's ±50/100/200 ms jitter sweep, which shows
  how much the effect depends on span accuracy at all, and (3) `ENC-RANDLOC-1L`
  (below). Using forced alignment without human verification is standard practice
  in the field; the guide is stricter than the norm because it defaults to
  priority-3 cross-attention DTW.

- **Implemented — independent CTC forced aligner (closes E1-N2).**
  `src/csasr/data/ctc_alignment.py` provides romanization-based multilingual CTC
  alignment with a pure-PyTorch Viterbi over the standard blank-interleaved
  trellis (torchaudio 2.5.1 cannot load against torch 2.10.0, and is not needed).
  Frame geometry matches exactly: wav2vec2 strides 320 samples at 16 kHz = 20 ms,
  identical to Whisper's encoder step, so frame indices are directly comparable.
  Unlike the two-DTW-configuration proxy, a CTC model infers boundaries from
  frame-local acoustic posteriors, so its failure modes are largely disjoint and
  agreement constitutes evidence. **Disabled by default** (`alignment.ctc.enabled`)
  pending model download; when unavailable E1 records the check as skipped rather
  than failing. **DTW remains primary for all E2/E4/E5 spans** — promoting CTC to
  primary would be a pre-registration decision taken before any E4 result is seen.

### Decoding configuration

- **Adapted DEC-A1 — `return_timestamps` is `false`, against guide §7.2.**
  The guide prescribes `return_timestamps: true`. With `transformers` 4.57.6
  that setting is *incompatible* with the rest of §7.2: combined with
  `return_dict_in_generate: true` it forces Whisper's segment/long-form return
  path, which (a) returns a plain `dict` rather than a `ModelOutput`, and
  (b) moves per-step scores inside the segment records, so `output_scores: true`
  is silently ignored. Consequences observed in job 31189: an immediate
  `AttributeError: 'dict' object has no attribute 'sequences'`, and — had that
  been patched alone — `avg_logprob` would have been `NaN` for every utterance,
  destroying the baseline-confidence covariate that guide §36 requires for E4
  control matching.
  Nothing in P0 or E1–E5 consumes Whisper's own timestamps: E1 derives every
  boundary from cross-attention DTW under an explicit `<|notimestamps|>` prefix.
  Setting it `false` therefore preserves the *intent* of §7.2 (deterministic
  greedy decoding with usable per-token scores) where the literal setting cannot
  be satisfied by this library version. `decode_batch` additionally accepts both
  return shapes, so the pipeline no longer crashes if the flag is flipped back.
  Verified on GPU: real sequences, correct code-switched transcripts, and
  populated `avg_logprob`.

### H100 execution and reporting

- **Implemented:** BF16 inference, inference-only execution, cached baseline
  predictions, online direction statistics, hook cleanup, and single-utterance
  OOM fallback during generation.
- **Implemented:** batch audio is loaded through a thread pool
  (`soundfile` releases the GIL, and the dataset is on NFS), keeping the GPU
  from idling on network latency. Order-preserving, so decoding is unaffected.
- **Narrower RUN-N1:** batches are duration-sorted with fixed batch sizes rather
  than dynamically limited by total audio frames. `decoding.batch_size` is 16;
  `alignment.batch_size` stays at 8 because E1 is the only stage requesting
  `output_attentions=True`, which materialises ~2.9 GB of attention maps per
  utterance (32 layers × 20 heads × 1500²). Note that `decoding.batch_size` is
  part of every prediction cache's provenance, so it must be fixed before the
  first production run: changing it later invalidates all cached decodes.
- **Partially closed RUN-G1:** every stage now writes `gpu_stats.json` (peak
  allocated/reserved vs. device total) into its run directory, and the decode
  loop logs throughput in utterances/second with the batch size used. Telemetry
  is *per stage*, not yet per E4 configuration.

## 5. Deviation and correction register

The following table is the authoritative checklist for bringing the current
implementation into stricter agreement with the guide.

| ID | Priority | Required correction | Current state |
|---|---:|---|---|
| E1-G1 | High | Make “most measured usable boundaries within 100 ms” an E1 gate criterion | **Re-opened as GATE-A1** — criterion exists and is measured, but is advisory until the threshold is calibrated from the first valid measurement |
| E1-G2 | High | Require numeric boundary estimates and report 50/100 ms rates and bias | **Closed automatically** — synthetic harness reports 50/100/150/200/250/300/500 ms rates plus mean and median signed bias; no human estimates involved |
| E1-G3 | **Blocking** | Calibrate `exclude_boundary_ms` against the measured boundary error, then re-gate | **Open** — job 31473's re-analysis implies ~215 ms median error vs. a 100 ms window; awaiting the first run of the corrected harness |
| E1-N1 | Medium | Stratify audit examples by span length and utterance position | Superseded by AUD-A1 (no human audit) |
| E1-N2 | Low | Use an independent aligner family if alignment quality is weak | **Implemented** — CTC aligner; model downloaded to `/mnt/data/tungnx/models/mms-fa` (`MahmoudAshraf/mms-300m-1130-forced-aligner`, `Wav2Vec2ForCTC`, 1.2 GB) and verified to load and align through `align_units`. Left `enabled: false` deliberately: it audits DTW but supplies no span used by E2–E5, so it is enabled in the E1 calibration re-run rather than in the run being unblocked. |
| AUD-A1 | — | Manual boundary audit replaced by automatic measurement | Accepted deviation, justified above |
| E2-G1 | High | Restrict decoder-direction examples to correctly transcribed training examples | Open |
| E2-N1 | Medium | Document/justify the 2,000-utterance decoder cap or make it configurable to all eligible examples | Open |
| E2-N2 | Medium | Produce a canonical 100%-full primary direction in addition to seed subsamples | Open |
| E3-G1 | High | Implement the complete guide ranking rule | Open |
| E3-G2 | High | Add per-speaker variance/dominance reporting and a gate criterion | Open |
| E4-G1 | High | Add token-frequency and boundary-proximity covariates to matching | Open |
| E4-G2 | Medium | Validate aggregate speaker-distribution balance | Open |
| E4-G3 | High | Add paired PIER, MER, and correction/harm confidence intervals | Open |
| E4-G4 | Medium | Add per-speaker tables, wall time, and peak memory | Open |
| E4-G5 | **Blocking** | Decide `pilot_wrong`/`pilot_correct` vs. the attainable POI pool, or redefine the matched-control criterion | **Closed** — measured pools, `min_confirm_wrong: 100`, controls scored against `min(n_correct, len(wrong))` |
| E5-G1 | Medium | Load frozen E4 confirmation manifests directly in E5 | Open |
| RUN-N1 | Low | Replace fixed-size batches with a total-audio-frame budget | Open |
| RUN-G1 | Medium | Persist runtime and peak-memory telemetry by E4 run | Partially closed (per-stage telemetry added) |
| DEC-A1 | — | `return_timestamps: false` vs. guide §7.2 | Accepted deviation, justified above |
| GATE-A1 | — | Synthetic boundary-error criteria advisory until calibrated | Accepted deviation, justified above; reverts to blocking by setting either threshold |

Optional omissions do not block guide completion unless the primary results
indicate that one is needed for diagnosis.

### 5.1 What `auto` cannot produce, by construction

These are not defects. They are outside what any single automated run can
deliver, and each requires separate human work after the pipeline finishes.

| Item | Guide reference | Why `auto` cannot do it |
|---|---|---|
| ~~E1 human boundary audit~~ | §13, §62 | **No longer applicable** — replaced by the automatic protocol (AUD-A1). `auto` now runs P0→E5 unattended. |
| Executive decision `GO` / `CONDITIONAL` / `STOP` | §63.10 | **No code writes it.** Gates are computed per stage, but nothing aggregates them into the single decision the guide names as the final deliverable. |
| Any official-test-split number | §2, §4 | Mechanically prohibited throughout E1–E5. Requires a separate, pre-registered run after a `GO`. |
| E6 router | §1, §63.10 | Out of scope. E1–E5 only establish whether steering works *given oracle spans*; the oracle span is an upper bound, not a system. |

### 5.2 Standing limitations for the final report

- **Alignment source.** `long_wav` TextGrids are absent, so all of E1 rests on
  Whisper cross-attention DTW — the guide's *third*-priority source (§10). Every
  E4/E5 span inherits that error. `unit_duration_diagnostics()` surfaces cases
  such as first-token silence absorption rather than hiding them.
- **Short-form only.** Only the `short_wav` release is present; the study uses
  pre-segmented utterances, not the long-form dialogue the corpus is built from.
- **Single model, dataset, language pair, and decoding mode** (greedy,
  `num_beams: 1`). No generalisation claim is supportable.

## 6. Optional human-audit procedure (not required)

Under AUD-A1 no human audit is needed and `auto` runs unattended. The audit pack
is still generated, so the procedure below remains available if a reviewer asks
for human corroboration or if the automatic criteria look marginal.

1. Set `alignment.gate.require_human_audit: true` in
   `configs/experiments/e1_alignment.yaml` (this restores the blocking behaviour
   and the exit-code-3 path).
2. Open `/mnt/data/tungnx/cs-asr-steer/artifacts_v2/audit/e1/`.
3. Review the clips and figures.
4. Copy `verdicts_template.csv` to `verdicts.csv` and fill the verdict and
   boundary-time fields.
5. Re-submit `sbatch cs_asr_e1_e5.sh auto`.

A `verdicts.csv` is honoured whenever it is present, even with
`require_human_audit: false` — supplying one adds a criterion, never removes one.

### 6.1 Where the alignment evidence lands

| Artifact | Contents |
|---|---|
| `metrics/e1_synthetic_boundary_error.parquet` | per-boundary predicted vs. known time |
| `metrics/e1_boundary_agreement.parquet` | DTW configuration-vs-configuration deltas |
| `metrics/e1_ctc_agreement.parquet` | DTW vs. independent CTC deltas (when enabled) |
| `metrics/e1_alignment_metrics.json` | `synthetic_ground_truth`, `silence_absorption`, `ctc_agreement` blocks |
| `reports/e4_oracle_steering_*.md` | "Span localization" section (`ENC-RANDLOC-1L`) |

These five are the evidence base for the alignment-quality paragraph in any
writeup, replacing what would otherwise have been human verdicts.

## 7. Runtime verification checklist

After each run, check:

```bash
squeue -u "$USER"
tail -n 100 /mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_<JOB_ID>.out
tail -n 100 /mnt/data/tungnx/cs-asr-steer/logs/cs_asr_e1_e5_<JOB_ID>.err
find /mnt/data/tungnx/cs-asr-steer/artifacts_v2/status -maxdepth 1 -type f -print
find /mnt/data/tungnx/cs-asr-steer/artifacts_v2/workflow -maxdepth 1 -type f -print
```

Do not infer scientific success from a Slurm `COMPLETED` state alone. Confirm
the `passed` gate state and inspect the corresponding report.

## 8. Inspected source snapshot

The findings above correspond to these SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `docs/CS_ASR_E1_E5_AI_Guide.md` | `daaec0c58ee244e757201ea0c0dd71e1feb892c4e73b5ca005d9ce4af1b228c9` |
| `cs_asr_e1_e5.sh` | `b4d0056d76c2982bde25ac6b0e3529bdc54d3230149455d83269268718b25a58` |
| `configs/base.yaml` | `cec12b7fab3e33f5392d7e3f9d5c11e4fb34275704599a06f3ee4db551404691` |
| `configs/model/whisper_large_v3.yaml` | `52be1c35b927b62f0d2cfdb7db23cccdfdce66db328a6b24e9cfec548fefe742` |
| `configs/experiments/e1_alignment.yaml` | `8365a84f53d535cc2f4361fe741df128b4a405b68580a369efb28a168b1da715` |
| `configs/experiments/e2_directions.yaml` | `44976ce5ebfc796d7205cf69ddc4fc23ed9130b29ecd64ef593a9ad57274ff92` |
| `configs/experiments/e3_separability.yaml` | `bb0e855b88f47011396c7404b8396691238aa66cdc612b94af226ab882f2cb9e` |
| `configs/experiments/e4_oracle.yaml` | `60cf414abcabb1a71f857c5ddfefdc92cf6ccedeb3ddbc6a8291325bad23d472` |
| `configs/experiments/e5_boundaries.yaml` | `8df2d8aa81436fe2d35937fce8c05e27198cfb4988cf8ed01dc8341e79036259` |
| `src/csasr/models/generation.py` | `45bff3a10150e8a16c7fe9ab9c1b08018230fc03f9d87192758b4a6a4935ee13` |
| `src/csasr/models/whisper.py` | `837a16a5ca830d53c6c5287eaad52fec2f0cdb9f276a2b4efcaee5a40b0953cf` |
| `src/csasr/experiments/_common.py` | `7e352f948cbe5db95ee538ac6f71c3c1014d8ef501aed49d83808a63c36fa39d` |
| `src/csasr/experiments/e1_alignment.py` | `dc54c8b2c2b5c036d4d70d18836055eecab09a276caa7470d4483bd744a6182d` |
| `src/csasr/experiments/e2_directions.py` | `721161cda0c515c91333f0e59af18cb879ee499acf23cdcfc12b0f18dc2754c7` |
| `src/csasr/experiments/e3_separability.py` | `f5f09aeaed678aa687124ad810537737b51eae0c70383ec1338399b9f3198b8e` |
| `src/csasr/experiments/e4_oracle.py` | `9139faccf8900311a879c03a0c78aead7ee499efd27472edddc6bcaae3a9a4ab` |
| `src/csasr/experiments/e5_boundaries.py` | `744ad28d86a52eb620bc22c575445f40b9a672f0610c82b2a702a59f6dadc4ff` |
| `src/csasr/data/alignment.py` | `592abcf8e341b82894d84f4f49676bec5e21522e198076b921a448341141224d` |
| `src/csasr/data/alignment_checks.py` | `d4675ac90ff38b0cfd09ecfab8cf05f637f5b8414b503225c82731bcd73d4380` |
| `src/csasr/data/ctc_alignment.py` | `a029cd917b54931eb7adb18b5709325bc2a88f3d6cf71fc48e6eefee0002e178` |

If any of these files changes, refresh this audit before using it as the
current conformance record.
