# P2-SEQ — Sequence-Level Viability Screen

Pre-outcome design freeze, 2026-10-07. Authority: this specification plus
`configs/inference_cf/p2_seq.json`; disagreement blocks execution. This is a bounded, legacy
inference-time development program, separate from the active v6/core METHOD_CONTRACT. No core
method changes, fresh validation, full300, P3, or adaptation are authorized.

## Question and accepted evidence

Does the existing E*R_B*D2 configuration produce useful free-decoding behavior on the fixed100
mini-panel? D2 is powerful at L16; gating retains local EN benefit and removes most Mandarin
harm, but residual false positives remain. Accept terminal E DIAGNOSIS_AMBIGUOUS, T
TOKEN_LID_NOT_DISCRIMINATIVE, XA INVALID, and LAC NOT_DISCRIMINATIVE. Do not rerun or reinterpret
these terminal decisions as successful repairs. This stage tests the existing configuration,
not another mechanism or score. No OLD-direction/BROAD/ungated/alternative gate/beam/TTA arm.

Start: branch feature/inference-cf-steering, local=remote
`42af08631700329d7402ad33ceedbc157a775d43`, clean, empty queue. No local work discarded.

## Exact panel and three systems

Use the existing `docs/inference_cf/P2_SEL_MINI_PANEL.json` rows in their saved order, all100
unique IDs,20 dialogues,5 each. Byte SHA256:
`266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`.
The internal panel_sha256 is a separate payload digest, not this byte hash. Resolve audio by
joining the existing parent P2 panel IDs; no reselect/refill/drop/outcome-based ordering.
Role is already-exposed D-dev-select; refs are evaluator-only and must never enter the runner.

- S0 B0-FORCED: same sequence driver as STEER, alpha0, forced-ZH no-edit B/E/S topology.
- S1 B0-AUTO: historical ordinary Whisper generate(language=None), practical comparator.
- S2 STEER: unchanged p2dir_readout_v1, L16 zero-based DG-02 post-cross-attention/pre-FFN,
  E*R_B, alpha2, identity phi, NormPreserve, no depth rescale, current R2 localizer.

All use the frozen local Whisper-large-v3 snapshot/tokenizer/preprocessing, eval mode, bf16/eager,
batch1, greedy, no sampling/beam, max_new_tokens200. Preserve old audio mono/resample/feature
padding/truncation; heard audio remains offline available first30s, no streaming claim. Model-file
identity comes from P2-DIR model.files; weights SHA256 in config. Preserve suppression tokens and
begin suppression at step0, exact processed_argmax (lowest-ID tie), EOS handling and decoding.
Forced prompts cB=[50258,50260,50360,50364], cE=[50258,50259,50360,50364], length4. No edit at
absolute query3/t0; eligible queries start4/t1. S0 and S2 have identical S forward/hook/query/cache/
attention/argmax topology; only dose/direction intervention differs. No plain cached_greedy causal
baseline: historical P2 recorded59/300 near-tie transcript differences against its matched B0M.

## Reuse decision (pre-outcome, independent audit)

Config seals100 historical row paths/byte hashes from p2_A_r1_L16. Each is complete with AUTO and
B0M_L16, zero-dose/lineage true; old source hashes match current files. No new mini outcomes were
computed in this design session.

S0 MUST run in the new same-driver zero-dose path: an audited D2-runtime zero-dose execution
artifact does not yet exist. Historical B0M is a diagnostic lineage anchor only; matching prompt
or text is insufficient execution-path evidence. STEER must newly decode all100.

Prefer reuse AUTO for all100. Before outcomes verify every sealed row/manifest/hash/status and
exact IDs/audio input bytes/feature config, snapshot/tokenizer/generation_config, historical
inference_cf_p2.py generate call, same torch2.10.0+cu128/transformers4.57.6/Python3.11.9 environment,
precision/eager/200/greedy settings and R2 model/provider anchor. Missing proof, changed input or
changed library semantics rejects reuse for the entire AUTO system; compute all100 AUTO in the
same single allocation, no separate baseline job. The choice is engineering-only and must be
sealed/pushed in reuse_audit.json before outcomes. No selective per-utterance reuse based on error.
No fabricated per-system historical runtime/VRAM; AUTO efficiency unavailable if reused.

Exact AUTO call remains model.generate(**batch_model_inputs, task='transcribe', language=None,
do_sample=False, num_beams=1, max_new_tokens=200, condition_on_prev_tokens=False), unchanged library
defaults. Preserve historical first-EOS truncation and tokenizer.decode(skip_special_tokens=True).
Keep original returned IDs, language/prompt and termination metadata if available. Do not reroute
AUTO through forced cached branches. Historical AUTO outputs retain original saved cap flags;
where raw EOS evidence is absent, a length200 result is conservatively a cap. This practical
comparator may differ in execution path; it NEVER becomes the causal delta reference.

## Runtime D2 coupling: implementation contract, no new scientific actuator

Reuse cached.Branch, _edit_hook and gate pipeline; add one additive sequence driver with fixed
D2 provider, supporting alpha0/2 only. Independent B/E/S KV caches share immutable original encoder
output and consume their own prompts followed by the SAME emitted S content tokens. B/E are
unsteered even after S diverges. Do not feed S0/reference tokens into S2; do not use180 teacher-forced
directions or gate values at new prefixes. No reset/hot swap/crop of persistent caches.

At t>=1 snapshot the PRE-step B cache via readout.clone_scratch before B.step updates it. Clean
B.step captures current-query L16 site/current frozen alignment-head attention, E.step remains
unsteered (attentionFalse). Compute historical prefix_utf8_complete, max_attention_window50frames,
RAW-logit conflict_from_logits R_B and 100-language native_lid local_support E with unchanged
zero480000-waveform null and epsilon1e-12. Cache LID only by identical audio/window/provider key,
not by reference or outcomes. No masked audio, token-local crop, LAC/XA/temporal/Q repair.

For eligible positive effective dose only, call unchanged readout_direction on the pre-step
snapshot, exact new IDs/start/step/encoder/partition/suppression. It uses its existing isolated
zero-leaf autograd, float32 processed J, float64 tangent geometry and float32 returned unit axis.
Verify scratch logits AND site bitwise equal to clean B. The gradient is at B, applied unchanged
to S's current site via DG-02 hook alpha2*g/NormPreserve; do NOT recompute at S or optimize parameters.
Release snapshot/scratch every step. Do not wrap the entire driver in a decorator that prevents
existing readout inference_mode(False)/enable_grad behavior. No model parameter grads retained;
flags restored, persistent caches/encoder untouched by scratch. Zero-dose S0 skips D2/snapshot
without changing its model-forward path. Structural and existing tiny/nonfinite direction
fallbacks are logged no-edits; provider/site/cache isolation mismatches invalidate, no alternate fix.

S0 must have B/S logits bitwise equal EVERY step, exact next token identity, isolated cache lineage.
For S2 require the same identity until first actual edit. After prior S edits, a current zero gate
means exact unchanged current S state, NOT equality to the counterfactual B branch. Every first
S2-vs-S0 transcript divergence (including EOS/length) must be preceded by an actual edit at or
before that step. No edit anywhere with identical transcript is valid NO_USEFUL_GAIN, not INVALID.

## Canonical metrics and diagnostics

Use csasr.evaluation.canonical.corpus_metrics, correction_corruption and canonical retention
primitives. Canonical PIER/MER/EN-WER/ZH-CER, POI corrections/corruptions (must satisfy net corrections
=forced POI errors-steered errors), matrix-ZH and embedded-EN retention, exact counts/denominators,
and cap count. Normalize/tag/align once through existing canonical definitions; no whitespace WER.
Use load_references' existing frozen R2 evaluation hash, subset fixed100 only.

Define lexical outside-target population explicitly: reference indices OUTSIDE all canonical
English POIs. Report outside transcript edits using two canonical reference-anchored align_tokens
projections, including insertion lists assigned to preceding reference index; leading insertions
at-1 separately. Report correctness-flip outside harm using unit_status, baseline-correct to
STEER-wrong, denominator baseline-correct outside units. Do not count beneficial edits as harm.
This lexical outside-POI population is NOT acoustic outside-candidate harm; the latter is N/A
without additional trusted alignment. No new oracle/audio alignment is required or authorized.

Record utterances with actual edits (count/fraction); eligible steps (query>=4 and structural
fallback absent); positive gates; valid-direction steps; actual state-changing edits (hook norm>0);
sum squared realized edit norms and relative energy; mean/max raw gate over all decoded steps
and separately eligible; D2 provider/scratch/autograd counts; LID/cache calls. Preserve hook pre/post
norms and edited-site bf16 vectors in compact NPZ for independent energy checks. S0 D2 calls=0.
Separate requested action from realized nonzero edit. Record per-system walltime/throughput and
peak allocated/reserved VRAM, excluding model-load setup but including branch/cache/LID/D2 costs.
Reset peaks only after previous system caches freed; also report whole-job totals/setup separately.

## Frozen development decision and uncertainty

Error deltas always STEER-minus-FORCED (positive=worse); also emit canonical *_gain=FORCED-minus-
STEER. Corpus micro counts/ratios, not unweighted mean utterance WER or dialogue mean PIER.
Point thresholds (float representation guard1e-12 only):

| Predicate | Frozen value |
|---|---|
| Material benefit | >=5 fewer POI errors AND PIER gain>=0.005 (0.5 percentage point) |
| MER increase | <=0.005 absolute |
| ZH-CER increase | <=0.005 absolute |
| Matrix-ZH retention | >=0.99 |
| Embedded-EN retention | >=0.95 |
| POI corruption rate | <=0.05 of baseline-correct POIs |
| Lexical outside-POI harm | <=0.01 of baseline-correct outside units |
| Added capped utterances | <=1 vs matched forced |

MER/ZH/99% retention follow historical P2 budgets. A net5 correction/half-point gain excludes
old near-inert +2/2268 behavior; this is practical screening, not a statistical discovery threshold.
EN95% retention/POI5% corruption preserve prior correct-state safety scale. Outside1% makes the
Mandarin-dominated outside budget explicit. No effect/predicate is tuned from the new100 outcomes.
Empty baseline-correct retention/corruption populations yield None, raw denominator0 and an explicit
not-assessable note (vacuously passes that rate rule). Nonempty POI, MER, EN and ZH reference
populations and finite primary error metrics are required.

Use existing paired dialogue-block bootstrap machinery:2000 draws, seed240924, NumPy default_rng,
sorted20 dialogue IDs, draw20 with replacement and retain all5 utterances per draw. Recompute corpus
ratios from count sums. Shared draws for forced/AUTO comparisons and rate uncertainty; pointwise95%
percentile intervals [0.025,0.975], omit empty-denominator endpoint draws and report valid counts,
>=1980 valid for each primary error endpoint. Intervals are DESCRIPTIVE, not qualification gates;
no multiple-family or confirmatory claim. This mirrors P2's computationally cheap block bootstrap.

Exact precedence:
1. P2_SEQ_INVALID: panel/path/reuse/provider/model/manifest/cache/autograd/evaluator/integrity failure,
   missing outputs (all100 required per system), unattributed divergence or audit failure. STOP.
2. P2_SEQ_SEQUENCE_DAMAGE: any point safety/cap bound fails, irrespective of corrections. STOP.
3. P2_SEQ_PROMISING: safety passes AND both material benefit predicates pass. STOP.
4. P2_SEQ_NO_USEFUL_GAIN: safety passes but at least one benefit predicate fails. STOP.

Always report STEER-minus-AUTO PIER/MER/ZH-CER and intervals separately. AUTO cannot change the
causal label. If PIER difference>=+0.005, explicitly say below AUTO on the primary practical metric;
95% lower>0 strengthens clearly-worse wording. Report conflicting Mandarin/MER tradeoffs without
collapsing them. PROMISING vs forced may coexist with below AUTO; no SOTA or deployment claim.

## Stopping, audit, compute, provenance and firewall

Before job: focused tests, resolved manifest/reuse seal and independent PASS_TO_P2_SEQ must be
committed/pushed. Auditor cannot import primary decision/analysis/bootstrap; may share locked
canonical metric primitives but independently aggregates counts/draws/labels. After run require
P2_SEQ_AUDIT: PASS before scientific conclusion. Recompute all IDs/provider/reuse/current-prefix
cache/zero-dose/first-divergence/energy/metric/ratio/CI/rule checks. Reused rows stay immutable;
record source paths/hash mapping rather than silently rewriting original outcomes.

Use provenance/specfreeze patterns: resolved config, git clean state/tree/commit, environment,
model/tokenizer/input/generation hashes, job/GPU metadata/status, every output shard/hash, parent
and reuse lineage. Runner receives audio/IDs/inference providers only, no references/evaluator data.
Compact per-step rows plus edited pre/post vectors; no full KV or high-dimensional logits dump.
Preserve exceptions/partial rows, never filter/refill. Expected200 new system-utterance passes
(S0+S2),300 only if AUTO reuse fails pre-run proof. One sbatch MIG/H100 allocation, max3h,
4CPU/48G, at most ONE P2-SEQ job pending/running, no automatic retry. Incomplete budget =>INVALID;
engineering-only repair needs a separately recorded rerun decision, no second job in this screen.

PROMISING recommends separately frozen larger development confirmation, no automatic300.
NO_USEFUL_GAIN/DAMAGE closes further local-gate repairs for now; create evidence-only
P2_SEQ_TTA_HANDOFF.md recommending P2-TTA0 test-time adaptation objective viability. Do not choose
adaptation objectives/coefficients/steps/hyperparameters or implement/run adaptation here.
INVALID authorizes only engineering repair, preserving the attempt and fixed scientific rules.

Allowed fixed exposed D-dev-select100 and previous P2/R/DIR/SEL/E/T/XA/LAC artifacts. Forbidden new
router-calib role, D-dev-confirm, D-test, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND, transfer/fresh
validation. Stage ends after100. E/T/XA/LAC remain terminal. Tests/check -> inspect diff -> commit
-> push origin HEAD:cs-asr-steer-inf for intentional changes, no force/main merge/PR. No scientific
job or runner implementation in this Codex design session.
