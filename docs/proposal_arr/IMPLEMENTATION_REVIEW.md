# Independent implementation review: CS-ASR ARR proposal v5 / plan v1

**Review date:** 10 August 2026  
**Reviewed proposal:** `CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`  
**Reviewed plan:** `CS_ASR_ARR_October_2026_Implementation_Plan_v1.md`  
**Code and evidence inspected:** `src/csasr/`, `tests/`, current configs and launchers,
`docs/CS_ASR_E1_E5_*`, and the recorded P0/E1/NAT5H artifacts under
`/mnt/data/tungnx/cs-asr-steer/`.

## Executive verdict

The plan is thoughtful, substantially grounded in the repository, and represents most of
the proposal at the level of work-package names. It is **not implementation-ready**. Several
blocking problems would either prevent the pipeline from running or make the main result
scientifically invalid:

1. the planned attention cache cannot produce the decoder transport tensor;
2. the proposed decoder site is not the site exposed by the existing hooks;
3. the planned candidate-ID refactor does not fix duplicate-candidate loss in the decoding
   cache;
4. at least one selector feature uses reference-alignment information unavailable at test
   time;
5. selector calibration, threshold choice, and Gate-D evaluation reuse the same ten
   conversations;
6. the utility and outside-harm definitions do not match either the candidate geometry or
   the current metric implementation;
7. steering scale is omitted from the freeze schema and energy equations even though the
   current hook multiplies by `projection_std`;
8. LoRA and learned-steering baselines receive less data than the proposed method, so the
   advertised matched-data comparison is not matched;
9. the main evaluation is budgeted over CS utterances only, yet claims overall MER/WER and
   monolingual retention; and
10. the declared stage graph contains circular and omitted dependencies.

No locked test run should occur until the blocking items below are resolved in the plan and
validated in an end-to-end development dry run. Gate A remains genuinely unresolved: P0 is
valid, production E1 failed badly, and the NAT5H smoke result establishes mechanical
two-family output—not boundary accuracy.

## Experiment coverage audit

| Proposal experiment | Plan coverage | Review |
|---|---|---|
| Exp. 0 — alignment, baseline, headroom | WP1/WP2 | **Partial.** Four prompts, counts, alignment audit, and MDE are named. Boundary-jitter stability required by proposal §4.1 is absent. The L2 MDE gate depends on the later WP4 oracle gain, creating a gate cycle. The timestamp-dependent heuristic is inconsistent with the frozen no-timestamp baseline. |
| Exp. 1 — oracle steering and action selection | WP3/WP4/WP5 | **Partial and currently non-executable for D/ED.** Encoder controls are largely covered. Decoder post-cross-attention states/hooks, D-step oracle semantics, D-layer/dose search, ED search, and action-overlap reporting are not specified consistently. |
| Exp. 2 — CS-span localizer | WP6 | **Nominally covered.** All four named systems and three neural seeds appear. Evaluation-split ownership, treatment of unaligned English spans, and a genuine frame-LID/diarization reference are unresolved. |
| Exp. 3 — utility selector and abstention | WP7 | **Represented but statistically invalid as written.** The five arms are named, but labels/features are underspecified and calibration, threshold optimization, and evaluation share `router-calib`. |
| Exp. 4 — main non-oracle comparison | WP8/WP9/WP10 | **Nominally covered.** Prompts, LoRA, global steering, learned steering, proposed, and oracle systems are listed. Data matching, matched-coverage comparisons, full-corpus scoring, monolingual retention data, and oracle-upper-bound semantics are missing or inconsistent. |
| Exp. 5 — explanatory ablations | WP6/WP7/WP9 | **Incomplete.** Missing: utility features with/without language margin; an ablation that isolates outcome supervision; oracle-mask versus predicted-mask at fixed utility policy; predicted mask with oracle utility; and automatic E/D/ED comparison if transport passes. “Oracle localization + oracle utility” changes two factors and is not a mask ablation. |
| Exp. 6 — preservation and efficiency | WP0/WP8/WP9 | **Named, not fully executable.** Metrics are listed, but the planned runs include only CS utterances, no monolingual evaluation manifests or sample counts are defined, and current outside-span code measures changes rather than newly introduced errors. |
| Exp. 7 — optional SEAME transfer | Schedule only | **Not an implementation task.** There is a date and drop rule but no work package, data schema, license check, preprocessing, transfer/reconstruction freeze, metric compatibility check, compute budget, or gate. This is acceptable only if the plan explicitly labels Exp. 7 “not planned unless activated” and defines a separate post-core work package. |

The claim in proposal §12.1 that every P0 experiment yields a complete manuscript is not yet
supported by plan v1. Exp. 5 and Exp. 6 need concrete run matrices, while Exp. 7 needs either
an executable optional branch or an explicit exclusion.

## Blocking findings

### B1. The WP2 cache is insufficient for WP5 transport

WP5 requires

\[
r_q=\sum_t \bar A_{q,t}g_t,
\]

which needs attention mass over acoustic frame `t` for every relevant decoder step `q`.
WP2 stores only per-step argmax frame, top-1 mass, and entropy. Those three summaries cannot
reconstruct `A[q, t]`, compute `r_q` for an arbitrary candidate mask, or compute on-span /
off-span mass. Appendix A compounds the mismatch by declaring
`aggregate_attention(summaries, ...) -> ndarray` as if the summaries contained the matrix.

Required correction to the plan: cache selected-head attention rows (possibly quantized or
candidate-window sparse), or rerun the first pass after candidates are known and directly
cache candidate-conditioned `r_q`. Storage and compute estimates must be redone. A round-trip
test must compare cached/reconstructed `r_q` with full eager attention on real Whisper
generation, not only a toy matrix.

### B2. The proposed decoder intervention site does not match the current hook site

The proposal defines decoder states **after cross-attention**. Current
`ActivationRecorder(module="decoder")` and `DecoderSteeringHook` attach to the output of a
whole `WhisperDecoderLayer`, after its feed-forward block. `inspect_modules()` records this
as “residual stream output of decoder block k.” WP3 says it will collect post-cross-attention
states but does not add a recorder for that sublayer; WP5's new hook is described as “like”
the current whole-block hook.

This is not a naming issue: direction construction and intervention would occur in different
representations from the proposal. The plan must choose one exact tensor and use it for
natural contrasts, prompt-subspace projection, steering, random/sign controls, and hook
tests. Implementing true post-cross-attention residual steering requires capturing the
residual around `decoder.layers[k].encoder_attn`, not merely its attention output.

### B3. Candidate batching is broken beyond `EncoderLocalSteering`

WP7 R1 changes the span lookup from `utterance_id` to `candidate_id`, but the shared decoding
pipeline still:

- hashes targets by `utterance_id`;
- stores no `candidate_id` in `PREDICTION_COLUMNS`;
- builds the resume map as one row per `utterance_id`; and
- filters pending work using `utterance_id`.

Multiple candidates for one utterance will therefore collide or disappear on resume even if
the hook applies the right mask. The row identity must become an explicit `example_id`
(candidate ID for label generation, utterance ID elsewhere) through provenance, cache,
resume, OOM fallback, outputs, and scoring. Add a resume test with two candidates from the
same utterance; the proposed hook-only test is insufficient.

### B4. Selector feature leakage and inference-interface contradictions

WP7 lists “boundary confidence” sourced from consensus edge disagreement. Consensus comes
from reference-unit forced alignment and is unavailable to the automatic test-time system.
It cannot be in `phi_k`. The Appendix-A `build_features(..., spans, ...)` signature makes
this leakage easy to introduce.

Separately, §11.2 says `features.py` must raise on every test row. That would prevent the
locked system from constructing legitimate inference-only features on `D-test`. The guard
must reject forbidden **sources/columns** (reference, gold unit, consensus, baseline error
label), not the test split itself.

Each feature needs a source contract with three fields: inference availability, mapping to a
candidate when no transcript token exists, and missing-value behavior. Token confidence,
script, encoder–decoder disagreement, term surface/frequency, and attention concentration
are currently undefined for deletions and candidates not aligned to a first-pass token.

### B5. Gate D evaluates on the calibration/tuning set

`router-calib` is used to fit isotonic calibration, maximize net utility over `tau`, compute
AUROC/ECE, and establish a paired-bootstrap advantage over baselines. This makes ECE
optimistic and Gate D a tuned-set result. Ten conversation clusters are also too few for
stable isotonic calibration plus threshold selection plus a 95% gate comparison.

Use one of the following before implementation:

- split `router-calib` by conversation into probability-calibration and operating-threshold
  subsets, then evaluate Gate D once on `D-dev-confirm`; or
- use grouped nested cross-fitting for calibration and threshold selection, producing
  out-of-fold predictions, with `D-dev-confirm` reserved for the single final development
  confirmation.

Do not use the same rows to fit isotonic regression and claim ECE <= 0.05. Calibration
should include a reliability diagram, Brier/log loss, positive count, and uncertainty, not
ECE alone.

### B6. Utility mathematics and the planned table schema are inconsistent

The assertion that `N_corrected` is in `{0,1}` because `K=1` is false. `K=1` means one
candidate region, not one lexical unit; thresholding and merge gaps can create a region that
overlaps multiple English or Mandarin units. A false-positive Mandarin candidate has no
single `poi_index`, yet the current correction/harm code and Appendix-C schema assume one
`baseline_correct` value and one baseline category.

The four proposed `candidate_class` values are not mutually exclusive: a candidate can be
an English error and also cause outside harm. Store independent boolean/multi-count fields,
not one categorical value. Define candidate-local reference-unit sets using frozen
evaluation alignment only for training labels/scoring, never features.

There is also an objective mismatch. The model estimates `P(U > 0 | phi)` but `tau` is chosen
to maximize a utility whose negative magnitude varies. Ranking by probability of positive
utility is not generally the same as ranking by expected utility. Either regress
`E[U | phi]`, predict benefit and harm components separately, or justify a binary target by
showing utility magnitude is effectively constant conditional on sign.

### B7. “Outside-span harm” is not implemented by the current metric

`evaluation/correction_harm.py:outside_region_edits` counts transcript differences between
baseline and steered outputs outside a fixed reference-token radius. It does not determine
whether those changes are new errors; beneficial outside corrections are counted as harm.
It also uses `poi_index +/- 1`, not the set of lexical units overlapping the selected
acoustic candidate.

Define mutually exclusive outcome sets relative to the reference:

- newly corrected units inside the candidate;
- newly corrupted units inside the candidate;
- newly corrupted units outside the candidate; and
- optionally, corrected units outside the candidate as a separate diagnostic.

Insertions need an explicit reference-boundary attribution rule. Unit tests should cover
deletion recovery, multi-unit candidates, insertions, beneficial outside changes, and false
positive Mandarin candidates.

### B8. Steering strength and energy omit the existing activation scale

The existing hook applies `alpha * scale * gain * direction`; production E4 supplies
`projection_std` as `scale`. The proposal and plan write `rho * gain * Delta`, the freeze
file stores no scale, and the effective-energy equation omits `scale^2`. Thus `rho=1` is not
reproducible from the planned freeze artifact and energy-matched controls will not actually
be matched.

Freeze the direction normalization convention and site/layer-specific scale artifact. The
pre-normalization energy is

\[
E_{pre}=(\rho s)^2\lVert\Delta\rVert^2\sum_t g_t^2.
\]

With norm preservation, the realized delta depends on the hidden vector and direction, so a
closed-form global gain only matches pre-normalization energy. If the claim is realized
post-renormalization matching, gains require calibration or the comparison must report the
realized energy distributions and a tolerance. The freeze file must record `scale`, its
estimator, artifact hash, and norm-preservation convention.

### B9. Matched-data baselines are not matched to the complete method

The proposed method uses `D-construct` to estimate both fixed directions, nuisance models,
and prompt subspace, then uses `loc-train` and `util-train` for learned routing. WP8 trains
LoRA and learned global steering only on `loc-train U util-train`. Consequently the proposed
method receives 70 additional construction conversations. The learned-vector versus
closed-form comparison is especially confounded by this data mismatch.

At minimum report two supervised baselines:

1. proposal-defined router-matched LoRA on `loc-train U util-train`; and
2. full-method-data-matched LoRA and learned steering on
   `D-construct U loc-train U util-train`, with exact utterances/audio hours matched.

If only one can be primary, full-method-data matching is the fairer scientific comparison;
the proposal-defined version can be a sensitivity result. Match supervision rules and
hyperparameter-selection budget as well as conversations. “Same conversation set” is not
enough if one method uses all utterances and another only CS utterances.

### B10. Exp. 6 and the main metrics cannot be produced from the planned run set

WP9 and WP10 budget roughly `10 systems x CS utterances`. That supports embedded-English
PIER but not corpus-wide MER/WER or monolingual Mandarin/English retention. Define and run:

- all utterances in each evaluation role for MER/WER;
- frozen monolingual-ZH and monolingual-EN subsets for retention;
- the embedded-CS subset for PIER and local intervention outcomes; and
- all utterances as the denominator for second-pass coverage/latency.

Global steering and LoRA must be evaluated on the same full sets. Parameter/latency metrics
cannot meaningfully be “stratified by deletion/substitution”; only correction outcomes
should be. Recompute the W5/W6 compute budget after adding these runs.

### B11. The stage graph is internally inconsistent

- L2 cannot pass because its MDE threshold depends on WP4 oracle gain, yet WP3/WP4
  logically consume L2 outputs.
- WP5 consumes an attention tensor WP2 does not store and a high-confidence subset from
  WP1.
- WP4 D/ED requires the WP5 routing semantics and exact decoder hook, while WP5 is described
  as both a W1 pilot and W2 decision.
- WP6's correctable-span metric depends on WP4 outputs and its chosen encoder feature layer
  may depend on WP3, despite being called parallel.
- WP7 requires passed/frozen WP4 and WP6 artifacts.
- WP9/Gate F requires WP8, so WP8 is on the locked-test critical path even though §8.2 calls
  it parallel.
- WP3 also requires Gate A and the relevant activation cache, so WP2 is on the critical
  path.

Move MDE finalization into WP4 or create a non-blocking L2 status followed by an L4 power
criterion. Write an explicit prerequisite list for every stage and test it in the launcher;
the current plan only says stages will use `require_passed()`.

### B12. Direction construction and the oracle grid are under-specified

WP3 does not explicitly enforce the proposal's **baseline-correct English** construction
filter, nor define how each English span is matched to one nearby Mandarin control, how ties
are resolved, whether controls may be reused, or what happens when no valid match exists.
Nuisance residualization cannot substitute for a specified pairing procedure.

The decoder contrast requires a matched Mandarin continuation under the same prefix, but its
matching rule is absent. “Cross-seed cosine, seeds 42–46” is not meaningful unless the plan
defines bootstrap/subsample draws; the paired mean itself is deterministic.

WP4's stated grid is also arithmetically unclear. Four encoder layers times three strengths
already gives 12 E configurations; D adds four decoder layers times three strengths, and a
Cartesian ED grid is much larger. The compute estimate refers to “~12 configs” while also
promising E, D, ED, controls, selection, and confirmation. Use a staged design: select E and
D independently, evaluate one preregistered ED combination, then apply controls only to the
selected action(s). Record the exact multiplicity policy and selection rule before running.

## High-priority scientific and evaluation issues

### H1. The no-timestamp baseline and timestamp heuristic are incompatible

The frozen baseline explicitly uses `return_timestamps: false`; current comments note that
timestamps change Whisper's generation path. WP6 nevertheless assumes first-pass token
timestamps. Define a timestamp source that does not change the baseline transcript, or run a
separate timestamp extraction pass and prove transcript/token identity. Otherwise the
heuristic receives information from a different decoder and is not a clean first-pass
baseline. The bilingual prompt likewise needs an executable generation interface and a test
showing that custom decoder IDs do not conflict with Whisper's forced-language machinery.

### H2. Localizer labels can silently turn unaligned English into negatives

“Positive on consensus EN spans; ignore low-confidence spans” is insufficient. English
units with no consensus span are neither known negatives nor merely low-confidence spans.
Training on their frames as Mandarin/background will teach the localizer to suppress the
hard cases and can inflate evaluation on the alignable subset. Use an unknown/ignore mask
for any region whose reference units lack trustworthy boundaries, or restrict training to
utterances with demonstrably complete unit coverage. Report label coverage and included vs
excluded span characteristics.

### H3. Consensus restriction creates selection bias

The local primary endpoint is restricted to high/medium consensus spans. Alignment
agreement likely correlates with duration, pronunciation, SNR, language, and baseline
correctness. Report unrestricted embedded-English PIER alongside the consensus-restricted
endpoint, the fraction excluded, and included/excluded distributions for duration, error
type, baseline confidence, and conversation. Never change the denominator by system.

### H4. Sorted conversation slicing is not a safe role assignment

Taking conversations 1–70, 71–105, etc. after sorting IDs may preserve collection order,
topic, device, region, or chronology. Use a seeded group assignment stratified or balanced
on hours, CS prevalence, English-unit count, topic/device, and available demographics.
Freeze and hash the assignment. Report balance; determinism alone is not sufficient.

### H5. The MMS baseline is not a frame-LID/diarization model

`facebook/mms-lid-126` is a 1B-parameter utterance-level
`Wav2Vec2ForSequenceClassification` audio classifier, not a frame classifier or language
diarizer. Sliding a 0.5 s window is a useful heuristic but should be named “sliding-window
utterance LID,” not the proposal's off-the-shelf frame-level baseline. Its official model
card also uses a CC-BY-NC license and should be recorded in the dependency/license audit:
[MMS-LID-126 model card](https://huggingface.co/facebook/mms-lid-126).

Either add a genuine code-switch LID/diarization system, or explicitly narrow the baseline
claim and report window/hop sensitivity. Whisper `detect_language` on 0.5 s clips is not an
equivalent fallback without validation. The 1 GPU-hour estimate for heavily overlapping
windows through a 1B model also needs measurement in WP0.

### H6. Gate-T diagnostics are internally contradictory

Proposal §2.8 begins with baseline-correct spans with known first-BPE positions, then asks
for results stratified by deletion, substitution, and correct cases. A baseline-correct span
is neither a baseline deletion nor substitution, and a deleted word has no first-pass BPE
step. Define distinct diagnostic cohorts:

- correct/substituted spans with an identifiable first-pass token step; and
- deletion spans evaluated by attention mass near a gold acoustic region, without claiming
  first-BPE hit rate.

Also distinguish oracle gold-step routing from inference-legal first-pass routing. WP4's
D-only oracle needs a precise rule for what happens when no decoder step exists.

### H7. Action selection under an oracle mask does not directly transfer to a soft mask

WP4 selects dose/action using an oracle binary/tapered mask, while the automatic gate is
`g_t = 1[t in S] m_t`. Its mass and energy depend on localizer calibration. Freezing the same
`rho` does not freeze the same intervention strength. Define whether candidate scores are
renormalized, clipped, calibrated, or left as probabilities, and report energy shift from
oracle to predicted masks. Add binary-predicted-mask versus soft-predicted-mask as an
ablation or calibrate only a globally fixed scale before `D-dev-confirm`.

### H8. Statistical claims need paired curve differences and multiplicity control

Pointwise marginal bands for two frontiers are not a test of their paired difference.
“Non-overlap somewhere” after sweeping thresholds is a multiple-selection rule. For Gate F,
predefine a common axis (matched coverage or matched collateral harm), interpolate each
system within each conversation bootstrap replicate, and form a paired band for the
difference. Use the frozen operating point for confirmatory inference; label the full curve
descriptive unless a simultaneous-band procedure is specified.

Gate B also tests several layers/doses, five permutation seeds, five random seeds, sign,
location, global, and sites. Specify whether null seeds are averaged into a control family,
which single comparison is confirmatory on `D-dev-confirm`, and how family-wise selection is
handled. Ten `D-dev-confirm` conversations make CIs fragile; report cluster-level effects
and leave the final claim to the 30-conversation locked test.

### H9. Test rerun policy is not truly a one-batch lock

The plan permits fixing a W6 bug and rerunning test after another development dry run. Once
partial test outputs have been visible, that is no longer the original blind one-batch
evaluation. Define failures before test:

- infrastructure failure with no readable predictions may be resumed from immutable
  checkpoints;
- deterministic scoring bugs discovered without inspecting outcomes may be corrected with
  a signed incident record; and
- any outcome-informed change invalidates the confirmatory run and must be disclosed as a
  second analysis.

The “empty `l10_test.json`” check is also unsafe because the stage normally writes `running`
at startup. Use an immutable test-attempt ledger rather than existence/overwrite semantics.

## Missing baselines and ablations

### Required to support the proposal's stated claims

1. **Outcome-supervision control.** Train the same selector architecture on the same full
   features to predict baseline ASR error/correctness rather than intervention utility. The
   current uncertainty-only and localizer-only arms change both labels and features/model
   capacity, so they do not prove utility labels are necessary.
2. **Factorized mask/utility upper bounds.** Evaluate at least:
   oracle mask + predicted utility, predicted mask + oracle utility, and oracle mask +
   oracle utility. These isolate “where” from “whether.”
3. **Language-margin ablation.** Proposal Exp. 5 explicitly requires full utility features
   with and without the language margin; WP7 omits the run.
4. **Matched-policy local/global control.** Apply the same selected-utterance/candidate
   policy to local fixed steering and utterance-global fixed steering at matched realized
   energy. An always-on global baseline versus selectively gated local steering confounds
   locality and selection.
5. **Automatic E/D/ED ablation if Gate T passes.** Oracle E/D/ED in WP4 does not establish
   the contribution of transported decoder steering in the complete predicted pipeline.
6. **True LID/diarization reference or an honestly relabeled sliding-window LID baseline.**

### Strongly recommended for fairness/interpretation

7. **Full-data-matched LoRA and learned steering**, in addition to the proposal-defined
   router-only matching described in B9.
8. **Learned local vector baseline.** A supervised learned vector with the same predicted
   local mask separates closed-form versus learned direction without changing locality.
9. **Joint nonlearned selector.** A preregistered score such as localizer confidence times
   ASR uncertainty tests whether the GBM is learning more than a simple conjunction.
10. **Second-pass execution control.** On a small development fixture, run an unsteered
    second pass through the identical code path to verify bit-identical or metric-identical
    output. This is primarily a systems control, not a paper baseline.

### Baseline naming risk

The proposed B2 is one vector plus one scalar at one selected layer. SALSA learns
layer-wise steering vectors with a supervised objective; the paper itself reports encoder
layer-wise adaptation. B2 is a reasonable cheap learned-global baseline, but it should not
be called “SALSA-equivalent” without matching the relevant layer-wise formulation. Cite it
as a deliberately smaller fallback and keep the distinction explicit:
[SALSA](https://arxiv.org/abs/2606.00460).

The proposal identifies Attention-Guided Adaptation as the closest parameter-efficient
competitor, but the plan supplies only a literature comparison. If an empirical
reimplementation on the same data is infeasible, state that limitation explicitly and
avoid claiming empirical superiority to the closest method. Its published method guides
language-expressive attention heads and trains additional parameters:
[Attention-Guided Adaptation](https://arxiv.org/abs/2312.08856).

## Reproducibility and provenance gaps

1. `git init` does not make earlier artifacts reproducible and may accidentally commit an
   already-dirty snapshot. Record an initial content manifest before changes, then commit
   intentionally. Preserve source-snapshot hashes for backward traceability.
2. Current provenance hashes `src`, `tests`, `configs`, and only
   `cs_asr_e1_e5.sh`; plan v1 does not explicitly add `cs_asr_lss.sh`, environment files,
   normalization resources, or report code dependencies to `SOURCE_FILES`.
3. `pyproject.toml` uses lower bounds rather than a lock. Export the exact Conda/pip
   environment, CUDA/driver/cuDNN versions, Slurm resource request, and sklearn/transformers
   versions used to train serialized models.
4. Hash model weights **and** processor/tokenizer/generation config. The current local model
   revision hashes one weight file and tokenizer JSON but does not fully identify all model
   assets.
5. Candidate-label artifacts need localizer checkpoint/config hash, label-threshold hash,
   pass-1 cache hash, metric version, and normalization hash—not only `action_hash`.
6. Freeze-file self-hashing needs a specified canonical serialization that excludes or
   blanks its own `sha256` field, atomic sealing, schema versioning, and hashes of every
   referenced artifact.
7. Seed ownership is unclear for role assignment, TCN initialization, minibatch order,
   permutation controls, random directions, LoRA, bootstrap, and human-audit sampling.
   Record a seed map. Setting `PYTHONHASHSEED` after interpreter startup does not make Python
   hash iteration deterministic; avoid depending on it.
8. Record deterministic-algorithm settings and document any nondeterministic GPU kernels.
   Re-run at least the selected neural localizer/selector/LoRA configurations across seeds
   or clearly separate selection seeds from final retraining seeds.
9. Human audit needs an annotation guide, blinding to aligner/system, auditor count,
   adjudication procedure, inter-annotator agreement, and immutable raw verdicts. “4–6 human
   hours” is not a reproducibility protocol.
10. Data-role reports need exact utterance/audio-hour hashes and balance diagnostics. Sorted
    identifier slicing is not enough.
11. Model/download licenses must be recorded. MMS-LID is CC-BY-NC; SEAME availability and
    redistribution restrictions need a go/no-go checklist before it enters a reproducibility
    package.

## Task sizing and complexity review

### Tasks that are too large as written

- **WP1** combines Qwen runtime debugging, coordinate diagnosis, DTW repair, a three-aligner
  production sweep, synthetic validation, a 200-unit human audit, and production label
  generation in one gate/week. Split it into WP1a diagnostic correctness, WP1b external
  validation, and WP1c production alignment/coverage. Do not scale to every training role
  before 1a/1b demonstrate boundary validity.
- **WP2** combines four full prompt decodes, a new per-step score path, attention capture,
  multi-layer activation cache, metrics, and power. Split prompt baselines from cache
  infrastructure. Cache exact required layers only after resolving the WP3/WP6 layer
  dependency, or budget a deliberate recache.
- **WP4** has no credible run count. Stage E and D selection independently, decide transport,
  run one ED combination, then run control families for only the frozen action. Publish the
  exact matrix before estimating GPU hours.
- **WP9** combines system integration, every development system, all metrics, bootstrap
  frontiers, six tables, a figure, freeze sealing, claim selection, and a full dry run in one
  week. Separate integration validation from report generation so a plotting failure cannot
  be confused with a scientific gate.

### Unnecessary or premature complexity

- Isotonic calibration is high variance with ten conversations; logistic/Platt calibration
  is the safer primary unless calibration data are enlarged.
- Nuisance residualization with conversation one-hot, demographics, bins, duration, and
  position is a large addition to an otherwise paired-mean estimator. Predefine the design
  matrix, missing-value policy, ridge objective, and an unresidualized paired-mean ablation.
- A self-hashing incrementally mutated freeze file is more fragile than immutable,
  stage-specific manifests plus one final lock manifest.
- A 1B sliding-window LID model with 80% overlap is expensive and not truly frame-level. A
  genuine diarization baseline or a cheaper validated acoustic-LID baseline is cleaner.
- Building decoder transport before the encoder oracle action shows headroom is defensible
  only as a small diagnostic. It should not delay the E-only critical path.

## Recommended dependency structure

The plan should use the following dependency graph rather than the current nominal week
parallelism:

```text
L0 roles + metric/spec freeze + real pilot
  |\
  | +--> L2a prompt baselines / score-cache interface tests
  |
  +----> L1a coordinate + aligner diagnostics
          -> L1b synthetic + blinded human validation (Gate A)
          -> L1c production consensus labels
                |\
                | +--> L6a localizer train/selection -> frozen localizer
                |
                +----> L3 exact-site E/D directions
                         |\
                         | +--> L5 transport diagnostic (Gate T)
                         |        -> exact decoder routing interface
                         |
                         +----> L4 staged E, D, then optional ED oracle screen (Gate B)
                                  -> frozen action

frozen localizer + frozen action + inference-safe pass-1 cache
  -> L7a candidate utility labels
  -> L7b selector training
  -> L7c grouped calibration/threshold selection

full method data/spec + frozen action layer
  -> L8 fair LoRA / learned-direction baselines

L6 + L7 + L8 + metric validation
  -> L9a complete D-dev-confirm evaluation (Gates C/D/E/F)
  -> L9b report pipeline dry run + immutable seal
  -> L10 one locked full-corpus test attempt
```

Move final MDE relative to oracle gain into L4/L9a. Treat WP8 as a locked-test dependency.
If Gate T fails, prune all practical D dependencies immediately. If Gate B fails, define
which localization metric remains meaningful without a nonempty oracle-correctable set.

## Minimum acceptance checklist before implementation starts

The implementation plan should be revised until all of the following have unambiguous
answers:

- [ ] exact tensor and hook site for E and D, with shapes and token/frame index conventions;
- [ ] cache schema sufficient to compute every planned feature and `r_q`;
- [ ] candidate-aware decode/resume identity end to end;
- [ ] feature allowlist proving no reference/alignment field reaches inference;
- [ ] missing-token behavior for every selector feature;
- [ ] multi-unit utility and true newly-introduced-harm definitions;
- [ ] mutually non-overlapping outcome fields instead of one `candidate_class`;
- [ ] scale/norm/energy convention stored in the freeze artifact;
- [ ] disjoint training, calibration, threshold-selection, development-confirmation, and
      test roles, or grouped out-of-fold predictions where data are reused;
- [ ] balanced conversation role assignment rather than sorted slicing;
- [ ] fair data and tuning budgets for LoRA and learned steering;
- [ ] explicit run matrices for Exp. 5 and full manifests for Exp. 6;
- [ ] boundary-jitter validation and consensus-selection-bias reporting;
- [ ] valid timestamp source for the heuristic baseline;
- [ ] paired frontier comparison at matched coverage/harm;
- [ ] exact stage prerequisites with no L2/L4 cycle;
- [ ] realistic configuration counts and revised compute/storage estimates;
- [ ] immutable provenance, environment lock, seed map, license record, and test-attempt
      policy; and
- [ ] an end-to-end `D-dev-confirm` dry run that exercises real Whisper hooks, duplicate
      candidates, zero-candidate utterances, deletions, false positives, full-corpus metrics,
      monolingual retention, and all report artifacts.

## Final assessment

The project has a credible research direction and a useful existing infrastructure base,
but plan v1 currently overstates executability and experimental completeness. The safest
path is to make the encoder-only pipeline the primary critical path, repair alignment,
formalize candidate-level outcomes and leakage-safe features, and treat decoder transport as
a genuinely conditional branch. After the blocking interface and evaluation issues are
resolved, the proposal's central “where / whether / how” decomposition can be tested
cleanly. In its current form, implementing the plan verbatim risks producing an apparently
complete locked experiment whose selector has leaked alignment information, whose decoder
mechanism is applied at the wrong tensor, and whose headline harm/selectivity metrics do not
measure the quantities claimed.

---

# L0 → L1a → L1b Alignment Gate-A implementation review

**Review date:** 11 August 2026  
**Verdict:** `REQUEST CHANGES`  
**Review type:** review-only inspection of the current implementation

The current implementation correctly separates automatic and manual workflows and
truthfully blocks when synthetic calibration is absent. However, several paths can still
produce scientifically invalid Gate-A passes through stale or unauthenticated artifacts,
missing mandatory criteria, incorrect coverage populations, and non-transitive
artifact-level taint.

## Review scope limitation

This checkout has no `.git` directory. `git status` and `git log` both report "not a git
repository," so staged, unstaged, committed, and Claude-specific diffs cannot be
reconstructed. This review covers the current implementation, files modified during the
inferred 11 August 01:35–02:28 fix window, and older components those files call. No
repository files were modified while performing the review.

## Actionable findings

### P0 — Forced-prerequisite taint exists only in mutable status, not artifact provenance

**Relevant regions:** `src/csasr/lss/prereq.py:91`,
`src/csasr/experiments/_common.py:94`, `src/csasr/lss/artifacts.py:42`, and
`src/csasr/utils/provenance.py:54`.

**Failure scenario.** A forced L1a/L1b run writes diagnostic artifacts. A later
`--overwrite` invocation clears sticky status taint, but does not necessarily rebuild every
artifact. Untainted status can then point at tainted candidate, consensus, or synthetic
evidence.

**Evidence.** `finish()` passes `taint` only as a top-level status field.
`stage_provenance()` receives no upstream provenance or taint. `artifact_ref()` contains
only path, hash, size, row count, and schema. `sticky=False` clears a stage's historical
taint without verifying which artifacts were recomputed. Existing tests synthesize status
payloads; they do not inspect real artifact manifests.

**Minimal correction.** Create immutable per-artifact manifests containing taint reasons,
producing run ID, parent artifact hashes/run IDs, source/config/data/model hashes, and
completion state. Merge taint across all parents. An overwrite may clear taint only for
artifacts atomically rebuilt in that run.

**Required test.** Force L1a, generate a tainted cached artifact, establish clean L0, retry
L1a/L1b without rebuilding that artifact, and assert Gate A remains blocked. Repeat with
two independently tainted parents and verify the union survives cache reuse.

### P0 — Candidate and synthetic inputs can be stale, partial, or fabricated

**Relevant regions:** `src/csasr/experiments/lss_l1b_valid.py:76`,
`src/csasr/lss/align/candidates.py:28`, `src/csasr/nat5h/schema.py:52`, and
`src/csasr/lss/align/autoevidence.py:249`.

**Failure scenario.** A nonempty `candidates_all.parquet` or five-column synthetic-score
parquet from an earlier configuration is accepted and can contribute to a production pass.

**Evidence.** `_candidates()` blindly reads `candidates_all.parquet` or the exploratory
NAT5H fallback. The combined candidate file bypasses `artifact_matches_identity()`.
Per-family cache identity uses `code_commit="unknown"` by default and ignores model
ID/revision and manifest/sample identity. There is no completion marker or expected-key
count; a truncated nonempty cache can be reused. Synthetic loading checks only five column
names. It does not verify schema version, known-boundary manifest hash, gate-set
fingerprint, configuration, units, finite/range constraints, independent families, or
taint. `num_boundaries` is summed across rows, double-counting the same boundaries across
families or conventions.

**Minimal correction.** Require atomic completion manifests for candidate and synthetic
evidence, verify exact expected unit keys/counts, producer identity, model/family identity,
data/config/source hashes, taint, and rendered gate-set fingerprint. Never use the NAT5H
fallback for production Gate A.

**Required test.** Inject stale, truncated, duplicate-counted, renamed-family,
wrong-config, and tainted artifacts and verify blocked/failed outcomes, never passed.

### P0 — Gate A omits mandatory evidence and can pass vacuously

**Relevant regions:** `src/csasr/lss/align/autoevidence.py:76`,
`src/csasr/experiments/lss_l1b_valid.py:306`, and
`src/csasr/experiments/lss_l1b_valid.py:471`.

**Failure scenario.** Two families each cover disjoint unit sets. Both qualify
independently, candidate-relative coverage is 1.0, agreement has zero pairs, consensus is
empty, and all natural, jitter, coverage, and span-schema criteria are skipped. A
superficially valid synthetic file can then let the remaining criteria pass.

**Evidence.** Missing paired natural agreement does not add a blocker. Jitter, coverage,
and span-schema criteria exist only under `len(spans)`. `unit_coverage()` defines its
denominator from candidate rows, not the frozen expected unit universe. Per-family invalid
rate is not gated: the 0.95 validity floor permits 5% invalidity despite the proposal's 1%
limit. A focused probe confirmed two families with 4% invalid rows both qualify. Automatic
usable-item rate is absent. EN–ZH uses `<= 30`, while the proposal says "under 30." Tests
check that a manually constructed missing-jitter criterion fails, but not that production
always constructs the criterion.

**Minimal correction.** Explicitly require nonzero paired natural evidence, nonempty
accepted primary spans, both jitter measurements, expected-universe coverage, automatic
usable-item rate, per-family invalid and nonmonotonic rates at most 0.01, and EN/ZH
eligibility. Missing mandatory evidence must be blocked, not omitted.

**Required test.** Run the actual Gate-A assembly with disjoint family keys, empty
consensus, omitted units, 4% invalidity, one missing language, and absent jitter. Every case
must be non-pass.

### P0 — Coverage is measured on the wrong role and several configured thresholds are unused

**Relevant regions:** `src/csasr/experiments/lss_l1b_valid.py:103`,
`src/csasr/experiments/lss_l1b_valid.py:502`, `configs/lss/l1b_valid.yaml:9`, and
`configs/lss/audit.yaml:77`.

**Failure scenario.** D-construct spans are counted and reported as
`loc_train_en_spans_high_medium`, allowing a claim about loc-train without aligning
loc-train. Dev-select and dev-confirm target coverage is never evaluated.

**Evidence.** The aligner sweep uses `D-construct` and `head(300)`. Coverage uses
`unit_table(D-construct.head(2000))`. `roles_to_label` is unused. Only
`min_loc_train_en_spans` is referenced. `min_construct_bilingual_utterances`,
`min_dev_select_targets`, and `min_dev_confirm_targets` are never applied. The candidate
and coverage universes differ, explaining the stored
`accepted_rejected_partition_exact=false`.

**Minimal correction.** Define and freeze each role's expected unit universe,
align/evaluate the configured roles separately, and gate every declared absolute-count
threshold under its actual role. Split the work into resumable per-role tasks instead of
one monolithic sweep.

**Required test.** Fixtures with distinguishable role IDs must demonstrate that
D-construct rows cannot satisfy loc-train or development thresholds.

### P0 — L0 can pass with a stale freeze; prerequisite hash checks compare against nothing

**Relevant regions:** `src/csasr/experiments/lss_l0_freeze.py:352`,
`src/csasr/lss/specfreeze.py:248`, `src/csasr/lss/prereq.py:240`, and
`src/csasr/experiments/lss_l1b_valid.py:254`.

**Failure scenario.** The live configuration or roles change, but L0 leaves an existing
freeze untouched, verifies only that old freeze's self-consistency, and writes a new passed
status.

**Evidence.** L0 builds a current spec but discards it when the freeze already exists.
`verify()` confirms the old file's self-hash and referenced files, not equality to the new
spec. `assert_matches()` is never called and checks only a small subset of frozen settings.
Prerequisites compute the current SHA but have no producer-recorded expected SHA to compare
against. L1b writes `spec_freeze_sha256=""` into spans.

**Minimal correction.** Compare the complete live specification with the frozen payload
and require explicit versioned supersession for any change. Status manifests must record
expected artifact hashes; downstream prerequisites must compare against them.

**Required test.** Mutate a frozen decision or artifact after L0 passes and verify L1a
refuses it. Run L0 with changed configuration and verify it cannot report passed against
the old freeze.

### P0 — Synthetic calibration is incomplete, interface-inconsistent, and not guaranteed disjoint

**Relevant regions:** `src/csasr/lss/align/synthetic.py:34`,
`src/csasr/lss/align/synthetic.py:68`, `src/csasr/data/alignment_checks.py:181`, and
`src/csasr/lss/align/autoevidence.py:264`.

**Failure scenario.** Tuning and gate calibration reuse source utterances, or the
documented `score_family()` helper produces a file the Gate-A reader rejects.

**Evidence.** Different random seeds do not guarantee disjoint source utterances or pairs;
there is no overlap assertion. Both purposes reuse pair IDs such as `syn_0000`.
`score_family()` emits `pct_within_100ms`; the reader requires `within_100ms`. The helper
emits p95, while the proposal requires p90. Start-edge, end-edge, and combined errors are
not represented consistently. Synthetic aligner execution and scoring are still explicitly
unimplemented. Absence currently blocks correctly, but automatic production cannot
complete.

**Minimal correction.** Construct a single deterministic partition before rendering and
enforce zero source-content overlap between dev and gate. Implement per-boundary
predictions for both edges, consistent median/p90/within-100 metrics, and an authenticated
gate-score manifest.

**Required test.** Assert dev/gate source hashes are disjoint; serialize actual
`score_family()` output and load it through Gate A; verify start, end, combined, median,
p90, units, and eligible denominators against hand-computed fixtures.

### P0 — Consensus/tolerance and jitter do not implement the frozen selection policy

**Relevant regions:** `src/csasr/lss/align/consensus_prod.py:41`,
`src/csasr/lss/align/consensus_prod.py:185`,
`src/csasr/experiments/lss_l1b_valid.py:216`, and
`src/csasr/lss/align/jitter.py:80`.

**Failure scenario.** Outputs are labeled as coming from a selected consensus
estimator/tolerance even though the selection or estimator ablation never affected
computation; jitter results are then computed over low-confidence spans excluded from
primary claims.

**Evidence.** `primary_tolerance_ms: null` becomes 200 ms through `or max(bins)`.
`select_primary_tolerance()` is unused. `ConsensusConfig.estimator` is recorded in output
but does not change the consensus algorithm; configured alternates are ignored. Jitter
receives all accepted spans, while only coverage filters high/medium. "Safe-interior
survival" compares jittered full-span duration against the original safe-interior duration;
it does not determine whether the safe interior survives the shifted boundaries.

**Minimal correction.** Select and freeze tolerance using authenticated external accuracy,
implement or remove estimator alternatives, and apply all primary Gate-A robustness checks
to the exact frozen high/medium subset. Correct the survival definition.

**Required test.** Calibration should select the expected tolerance; estimator variants
must change computation or be rejected; low-confidence spans must not affect primary
jitter; a shifted mask that loses the safe interior must fail survival.

### P1 — `probe_deadline_minutes` does not enforce process-tree termination

**Relevant regions:** `src/csasr/lss/align/qwen_probe.py:62`,
`src/csasr/lss/align/qwen_probe.py:157`, and `tests/test_lss_align_diag.py:283`.

**Failure scenario.** Checkpoint loading hangs inside native/CUDA code or starts workers.
SIGALRM may not interrupt it, and child/GPU processes can remain after the parent reports
or exits.

**Evidence.** The default is correctly 30 minutes and the alarm encloses import, load, and
run. The implementation explicitly runs in-process and has no process group. In a non-main
thread it warns and proceeds unbounded. `unload()` and `empty_cache()` do not terminate
children or guarantee release after an uninterruptible native call. Quarantine can move an
older valid Qwen cache because it is not scoped to the current attempt. The test uses
interruptible `time.sleep()` and never creates a child/grandchild or checks process/GPU
cleanup.

**Minimal correction.** Run the probe in a new subprocess session/process group, write
results to an attempt-specific temporary directory, terminate the group on timeout,
escalate to SIGKILL, and publish outputs only after successful validation.

**Required test.** A mocked child that spawns a grandchild and writes a partial artifact
must leave neither process alive, must not publish/reuse the partial result, and must record
blocked/deadline/elapsed fields.

### P1 — The second-aligner repair path is not executable in the declared stage graph

**Relevant regions:** `src/csasr/experiments/lss_l1a_diag.py:226`,
`src/csasr/experiments/lss_l1a_diag.py:259`,
`src/csasr/lss/align/candidates.py:54`, and `src/csasr/lss/prereq.py:33`.

**Failure scenario.** Whisper remains below 0.95, Qwen probe succeeds, but Gate A still has
no second production aligner.

**Evidence.** Qwen probe retains only row counts; the returned table is discarded.
Production `run_families()` always reports Qwen as deferred and never includes it. The
pred-start experiment is reported as deferred because it needs L1b's synthetic dev set.
L1b cannot run normally until L1a passes, creating an experimental dependency cycle. The
configured consensus-estimator alternatives are also not run.

**Minimal correction.** Move synthetic development-set construction into a prerequisite
utility/stage available before L1a, execute the pred-start diagnostic there, and persist
validated Qwen candidates for the production sweep.

**Required test.** With Whisper invalid and Qwen probe valid, L1b must consume Qwen
candidates and count two independent families. The pred-start sweep must execute without
forcing or tainting L1b.

### P1 — Stage lifecycle and wrapper behavior can leave misleading states

**Relevant regions:** `src/csasr/experiments/_common.py:64`,
`src/csasr/experiments/lss_l0_freeze.py:227`,
`src/csasr/experiments/lss_l0_freeze.py:235`, and `cs_asr_lss.sh:181`.

**Failure scenarios.** `prepare()` writes `running` before acquiring `StageLock`, so lock
contention overwrites the active run's state. `--verify-only` returns without a terminal
`finish()`, leaving L0 `running`. Unexpected exceptions generally leave `running`, not
`failed`. Automatic preparation always returns `completed`, even when `blocked_reasons`
says required evidence was missing. `chain "$@"` forwards options such as
`--force-prereq` and `--overwrite` only to L0, not L1a/L1b. The single eight-hour chain
contains L0, a 30-minute probe, synthetic rendering, and an alignment workload documented
as 6–10 GPU-hours.

**Minimal correction.** Acquire the lock before writing status, wrap stages in a
terminal-status exception handler, finish verify-only truthfully, classify unsuccessful
preparation as blocked, propagate explicitly supported chain flags, and split the chain
into status-aware resumable jobs.

**Required test.** Lock contention must preserve the active status; injected exceptions
must yield `failed`; verify-only must terminate; missing preparation evidence must not say
completed; wrapper tests must inspect argument forwarding and status-based continuation.

### P2 — Tests validate helpers but miss the production false-pass paths

**Relevant regions:** `tests/test_lss_gate_a_auto.py:1` and
`tests/test_lss_align_diag.py:283`.

**Evidence.** Automatic/manual tests call `_resolve_parts()`, not `main()`. Taint tests
write synthetic statuses rather than real artifacts/manifests. Synthetic tests
intentionally accept the under-specified five-column file. No test covers empty consensus,
missing agreement, omitted expected units, wrong roles, config drift, cache completion,
actual Qwen candidate consumption, or wrapper chaining. The timeout test cannot prove
process-tree cleanup.

**Minimal correction.** Add small end-to-end stage tests using temporary artifact roots and
mocked aligners, including every false-pass path listed above.

**Required test.** One parameterized production-path suite should assert final status and
exit code, artifact provenance, exact criteria present, and downstream prerequisite
behavior.

## Requirement matrix

| Requirement | Implemented | Correct | Evidence | Remaining action |
|---|---:|---:|---|---|
| Automatic L0→L1a→L1b prep→auto Gate A | Partial | No | Default L1b is automatic; synthetic scoring and second-aligner repair are incomplete | Implement missing evidence paths |
| Automatic mode never reads human verdicts | Yes | Yes | `_resolve_parts()` discards `verdicts`; tests pass | Add production-main test |
| Manual workflow explicitly selected | Yes | Mostly | Manual prepare/evaluate modes are separate | Decide whether human evidence may substitute for a second auto aligner |
| Two independent valid natural aligners | Partial | No | Hard-coded independence classes; no artifact/model identity validation | Authenticate family implementations and paired coverage |
| One valid aligner cannot pass | Yes | Yes | Fewer than two adds `blocked_insufficient_independent_aligners`; focused test passes | Preserve |
| Aliases/copies cannot pass as independent | Partial | No | Names map to classes, but arbitrary candidate files are trusted | Validate producer/model/algorithm provenance |
| Natural metrics named disagreement | Yes | Yes | Names and guard are appropriate | Add end-to-end schema assertion |
| Synthetic absolute-boundary calibration | No | No | Rendering exists; scoring is documented as unimplemented | Implement and authenticate scoring |
| Missing synthetic evidence blocks | Yes | Mostly | A missing file blocks correctly | Reject stale/fabricated files too |
| Start/end/combined boundary errors | Partial | No | Natural computes both; synthetic interface does not | Standardize definitions |
| Median/p90 in milliseconds | Partial | No | Natural has median/p90; synthetic helper emits median/p95 | Add p90 and unit validation |
| Coverage at least 0.95 | Partial | No | Candidate-relative denominator can omit expected units | Use frozen expected universe |
| Invalid/nonmonotonic at most 0.01 | Partial | No | Nonmonotonic gated; invalidity effectively allowed to 0.05 | Gate both at 0.01 |
| Usable items at least 90% | Manual only | No | No automatic usable-item criterion | Add automatic usable rate |
| EN/ZH asymmetry | Partial | No | Natural disagreement check exists; no guaranteed language eligibility; uses `<=30` | Require both subsets and authoritative comparison |
| ±50/±100 ms jitter required | Partial | No | Both offsets gated only when spans exist | Make mandatory on primary subset |
| Frozen high-confidence consensus subset | Partial | No | Bins exist and downstream defaults filter; provenance/selection is incomplete | Freeze exact selection and evidence |
| Configured absolute role counts | Partial | No | Three thresholds and `roles_to_label` are unused | Implement per-role accounting |
| Confidence filtering frozen/recorded | Partial | No | Bins recorded, but source/config/provenance incomplete | Include full criteria/config hashes |
| Thresholds recorded in evidence | Partial | Mostly | Gate criteria serialize thresholds | Link criteria artifact to freeze/run manifest |
| Timeout defaults to 30 minutes | Yes | Yes | Config and fallback both use 30 | Add default-value test |
| Timeout covers checkpoint loading | Yes | Mostly | Alarm wraps adapter construction/load/run | Move the same scope into a child process |
| Timeout kills complete process tree | No | No | Explicitly in-process; non-main-thread path unbounded | Use process-group termination |
| Timeout quarantines partial cache | Partial | No | Renames matching files indiscriminately; no completion marker | Attempt-scoped temp plus atomic publish |
| Timeout is blocked with reason/elapsed | Yes | Yes | Result fields and classification are correct | Test process path |
| Unrelated exceptions remain failures | Yes | Yes | Non-infrastructure exceptions become `failed` | Preserve |
| Forced taint transitive/multi-parent | Status only | Partial | Status union logic and tests exist | Put it in immutable artifact manifests |
| Taint survives caching/retry | Partial | No | Sticky status exists; overwrite/cache can launder | Bind taint to every cached artifact |
| Tainted Gate A cannot pass | Current status only | Partial | L1b adds blocked reason from prerequisite status | Reject tainted artifacts independently |
| Roles-only terminal truth | Yes | Yes | `completed_roles_only`, exit 0, `full_l0_pass=false` | Add wrapper integration test |
| Roles-only cannot satisfy L1a | Yes | Yes | Only full `passed` plus full flag satisfies prerequisite | Preserve |
| Auto preparation terminal success/exit 0 | Yes | Partial | Successful path is `completed`, exit 0, next `evaluate_gate_a` | Block when preparation evidence is missing |
| `completed_no_go` exit 0 | Yes | Yes | Python map and wrapper contract agree | Preserve |
| `blocked` versus `failed` semantics | Partial | No | Gate classification is sound; uncaught exceptions leave `running` | Add terminal exception handling |
| Slurm chain reads gate status | Yes | Mostly | Chain checks L0/L1a status and reports Gate A | Propagate flags and split oversized chain |
| Atomic status writes | Yes | Yes | Temp plus replace | Preserve |
| Atomic run manifests/reports | Partial | No | Parquet/JSON helpers atomic; `finish()` metrics/provenance and reports are not | Make all terminal artifacts atomic |
| Run IDs/config/data/parent provenance | Partial | No | Run dir and hashes exist; upstream list empty and artifacts unbound | Add complete lineage manifests |
| Stale cache invalidation | Partial | No | Default code commit is `unknown`; combined cache unchecked | Strengthen cache identity/completion |
| Deterministic audit/sampling | Partial | No | Seeds recorded; `head()` sampling and dev/gate overlap remain | Freeze sampled IDs and enforce disjointness |
| Task sizing/dependencies | No | No | L1a needs an L1b-produced dev set; eight-hour monolithic chain | Split prerequisite and per-role tasks |
| Previous CTC-valid/Whisper-invalid case remains no-pass | Yes | Yes | Current code blocks with fewer than two qualifying families; regression test passes | Preserve |

## Verification performed

The following checks were performed:

- Searched for `AGENTS.md`, `CLAUDE.md`, and `CONTRIBUTING.md`; none were found in
  this repository.
- Ran `git status --short` and `git log --oneline -8`; both failed because no Git
  repository exists.
- Inspected timestamps to infer the 11 August change window.
- Read proposal v5, implementation plans v1/v2, the state/exit contract, configs,
  schemas, L0/L1a/L1b CLIs, prerequisite and taint logic, aligner/cache/synthetic/
  consensus/jitter code, status/provenance utilities, tests, and `cs_asr_lss.sh`.
- Ran 106 focused tests from `test_lss_gates.py`, `test_lss_gate_a_auto.py`,
  `test_lss_align_diag.py`, `test_lss_prereq.py`, and `test_lss_specfreeze.py`.
  All 106 passed.
- Ran `bash -n cs_asr_lss.sh`; it passed.
- Ran CLI `--help` for L0, L1a, L1b, and the status module; all passed.
- Focused in-memory probes confirmed that two families with 4% invalid rows both
  qualify, candidate-relative coverage reports 1.0 when half the intended universe is
  omitted, and `score_family()` emits `pct_within_100ms` while the loader expects
  `within_100ms`.
- Read existing production status files without mutation. The recorded state is L0
  `running`, L1a `passed`, and an older L1b `failed`; it is not a Gate-A pass.

The two tests explicitly documented as regressions—automatic preparation previously
producing a pass, and only ±100 ms jitter being gated—would fail the described earlier
behavior. Without Git history, it is not possible to confirm independently that all 106
tests fail on Claude's pre-change revision. The passing tests around taint and timeout are
insufficient to establish the stronger artifact/process-tree requirements.

The review did not run models, GPU jobs, datasets, Slurm submission, network operations,
production stage commands, or model downloads. L1a `--dry-run` was not run because its
current default path can still execute the Qwen probe/model load. Full integration tests
against real artifacts were not run to avoid mutating experiment state.

## Exact remaining edit checklist

1. Introduce immutable, atomic artifact manifests with taint and complete parent
   provenance.
2. Enforce manifest/hash/completion validation for candidates, consensus spans,
   synthetic scores, roles, and the L0 freeze.
3. Prevent `--overwrite` or retry from clearing taint unless every reused artifact is
   proven clean or recomputed.
4. Make all mandatory Gate-A evidence explicit; missing agreement, consensus, jitter,
   role coverage, language subsets, or usable-rate evidence must be non-pass.
5. Gate invalid rate and nonmonotonic rate independently at 0.01 using a frozen
   expected-unit denominator.
6. Implement per-role alignment and coverage; remove the D-construct-as-loc-train
   mislabel and apply all configured counts.
7. Implement authenticated synthetic scoring with disjoint dev/gate sources and
   consistent start/end/combined median/p90 metrics.
8. Implement and freeze the preregistered tolerance/estimator selection, then run jitter
   on the exact primary subset.
9. Move Qwen probing into a process group with timeout escalation and attempt-scoped
   atomic outputs.
10. Persist successful Qwen candidates and resolve the L1a↔synthetic-dev dependency
    cycle.
11. Fix stage terminal-state handling, lock ordering, preparation failure status, atomic
    run outputs, and chain argument propagation.
12. Split the monolithic chain into resumable status-aware jobs.
13. Add production-path tests for every false-pass scenario and shell-wrapper behavior.
14. Restore Git metadata or provide Claude's base/head revisions before attributing
    individual findings to Claude's patch.

## Readiness decision

It is **not safe** to run:

```text
L0 → L1a → L1b automatic preparation → automatic Gate A
```

The earliest production acceptance point that must be stopped is **L0**: an existing stale
freeze can be self-consistent yet inconsistent with the live configuration, and the
recorded L0 status is currently `running`. L1a additionally lacks a process-safe Qwen
deadline and depends on synthetic development evidence produced downstream. L1b must not
be treated as production-ready until artifact taint/provenance, synthetic scoring,
mandatory criteria, correct role coverage, and cache validation are fixed.

The current code does correctly avoid passing the known one-valid-aligner case and
correctly blocks when synthetic evidence is genuinely absent, but those protections are
not sufficient against stale or malformed evidence.
