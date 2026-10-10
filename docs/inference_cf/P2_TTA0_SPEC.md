# P2-TTA0 — Episodic Test-Time Adaptation Objective Viability

Pre-outcome freeze, 2026-10-07. Specification/config must agree or execution blocks. This is a
separate bounded exposed-development experiment, not a revision of the v6/core METHOD_CONTRACT,
a final TTA algorithm, a literature replication or confirmation claim. Accept P2_SEQ_SEQUENCE_DAMAGE
and all terminal local-selectivity stages. No steering, D2, local-gate repair or continual adaptation.

Start: feature/inference-cf-steering, local=origin/cs-asr-steer-inf
`a8b34ff490835600d4a264f2d7a7c7f3c1f65f5f`, clean, empty queue. P2-SEQ shows small POI gain with
MER+0.028/ZH-CER+0.033/retention0.940/outside harm0.060, mainly deletion/early ending; AUTO is a
larger practical lever. Question: can two tiny episodic decoder updates give a useful change
without the sequence damage? This is an objective screen only, not a hyperparameter search.

## Panel, pseudo teachers and baseline compatibility

P2_TTA0_PANEL20.json takes the FIRST encountered row per dialogue in the saved100-panel order,
retaining encounter order. Exactly20 dialogues/20 IDs, no duration/transcript/language/error-based
selection. Parent byte SHA256266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5.
New20 byte SHA2563c752d7a2e247a77f05956839e3aa396ae04818fea7440f74b87778af5fa9eb4.
Exact IDs are enumerated in panel/config. Role: already-exposed D-dev-select. No refill/drop.

Systems exactly B0-FORCED, B0-AUTO, A1 GREEDY-EM, A2 AUTO-CONSISTENCY. Reuse sealed20 P2-SEQ S0
forced outputs and historical AUTO under P2-SEQ's audited reuse proof. Config records exact20
row hashes. Verify current model/tokenizer/preprocessing/audio/env/source/IDs/EOS/generation
identity before run; incomplete proof BLOCKS rather than choosing another teacher. Respect the
existing size-prefixed64KiB audio fingerprint, not an incompatible raw-SHA substitute; also record
full input hashes for the new manifest. Do not regenerate AUTO merely for convenience.

Before updates commit/push pseudo_sealed.json with exact y_B/y_A token IDs, source row/manifest,
model/tokenizer hashes, EOS/cap and content/mask metadata. No reference text. Preserve byte-fragment
content tokens, never retokenize decoded text. Strip only a verified leading Whisper prompt and
terminal EOS if present; unexpected interior special/control token INVALID. Historical returned
content lists are used unchanged when already stripped. No teacher selection after adaptation.

Forced inference uses cached.Branch.step (prompt then one emitted token), attention=True and
passive L16 recorder, core_p1.processed_argmax, unchanged suppression and lowest-ID tie, no edit
hook/gate/E/LID/D2. Freeze bf16/eager/eval/batch1, greedy beam1/no sampling,200 emitted-token cap,
cB=[50258,50260,50360,50364]. AUTO teacher remains ordinary historical generate(language=None,
task=transcribe,do_sample=False,num_beams=1,max_new_tokens=200,condition_on_prev_tokens=False).
In the ONE allocation the original theta0 no-hook forced path MUST reproduce all20 P2-SEQ S0
token lists and termination before that utterance adapts; mismatch =>INVALID, not silent baseline
replacement. Parameter/master data at theta0 is identical bf16 to these baselines. No separate
GPU equivalence job. All final adapted outputs decode FORCED-ZH, never AUTO or forced-EN.

## Exact trainables and precision/reset architecture

Meta-device inspection of the actual local Whisper-large-v3 config and installed class found:
- model.decoder.layers.i.{self_attn_layer_norm,encoder_attn_layer_norm,final_layer_norm}.
  {weight,bias} for EVERY i=0..31;
- model.decoder.layer_norm.{weight,bias}.

97 LayerNorm modules,194 tensors, each1280 scalars;248320 adapted scalars of1543490560 unique model
parameters (fraction0.00016088209830062065,0.0160882%). Complete194-name/shape enumeration in config.
No encoder LN/attention/FFN/projection/embedding/token/prompt tensor is adapted.

All resident model parameters stay bf16 and requires_grad=False. At each objective start create
fresh fp32 master torch.nn.Parameters for exactly those194 names from original bf16.theta0.float().
Optimizer and grads belong ONLY to these masters. During differentiable teacher forcing use
 torch.func.functional_call(model, {name:master.to(torch.bfloat16)}, args/kwargs,
 strict=False, tie_weights=True).
The casts stay attached to the graph; never detach, use .data or copy_ to transfer master values
BEFORE the backward. Frozen model params remain gradient-free. This preserves bf16 forward
weights/activations at theta0 and accumulates small updates in fp32. Model remains eval (dropout
and LayerDrop off); no autocast/GradScaler. Synthetic LayerNorm check already confirms theta0
forward equality and fp32-master-only gradient flow; full tiny-Whisper tests are required later.

After TWO steps, copy rounded bf16 master values into only resident decoder-LN parameters for
final no-hook forced decode. In a finally block restore original bf16 bytes, all flags/grads and
assert bitwise hashes. Check fresh next-objective masters equal theta0.float bitwise. Discard
optimizer moments, masters, graph and ALL decoder caches after EACH objective. Reset before next
objective and utterance, even on exceptions. Never reuse KV from theta0 or previous weights for
adapted decode. Model non-LN tensors immutable; encoder always original theta0.

Compute encoder once per utterance under frozen parameters using original preprocessing. Clone
its detached output into ordinary non-inference tensors outside inference_mode for adaptation;
retain original bf16 values bitwise. All teacher forwards/final decodes share this encoder for
that utterance only. No encoder graph/update or cross-utterance carry. A1 and A2 both start theta0,
not A2 from A1. Fresh AdamW per objective: lr1e-3,weight_decay0,steps2,betas(.9,.999),eps1e-8,
amsgrad=False,foreach=False,fused=False,maximize=False; no scheduler, accumulation or clipping.
No precision/optimizer/LR/subset fallback if nonfinite: INVALID and STOP.

## Teacher-forced indexing, suppression and two objectives

Explicit decoder input cB+y (T generated content IDs), no EOS appended; use_cache=False, no past KV,
no labels/automatic shift. At content step t the predictor is logits[:,3+t,:] for t=0..T-1.
The final query3+T predicts next/EOS for diagnostics only. Prefix queries0..2 and final query3+T
are excluded from loss. EOS is NOT a target/loss position but stays in the allowed predictive
vocabulary at every content position. This exposes early-EOS risk rather than hiding it.

Use float32 logits, temperature1, exact generation suppress_tokens every step plus begin_suppress
only t0. Gather finite ALLOWED vocabulary IDs and compute log_softmax, avoiding zero*(-infinity).
Fixed valid teacher mask: non-special generated content ID whose target is allowed at its step;
log excluded positions and keep the same mask across both updates. No reference/POI/UTF8-completion
mask or inferred correctness label. Teacher cap: use all up-to200 generated content tokens, do not
invent an EOS label. Empty/no-valid transcript: retain utterance, exactly2 no-op AdamW steps on
zero loss connected to masters (zero gradients), log no_valid_content; output must equal baseline.
No replacement teacher, additional generation or sampling.

A1 GREEDY-EM: y=y_B from theta0 forced baseline; minimize mean over valid content queries of
H=-sum_{v allowed}p(v)log p(v). It does NOT optimize pseudo-token CE or sample/beam/policy gradient.
A2 AUTO-CONSISTENCY: y=y_A from theta0 ordinary AUTO; under cB minimize mean valid-query
-log p(y_A[t]). Teacher detached/frozen, no reference or evaluator/Jacobian, no second teacher,
regularizer/lambda or mixed loss. Thus both fit a fixed model-generated trajectory, independently.

Perform L0 forward/backward/update1; L1 forward/backward/update2; L2 forward without update.
Exactly2 optimizer updates,3 loss evaluations per objective. Teacher logit loss is manually indexed
because built-in labels CE would include wrong prompt/EOS/suppression semantics. No whole-driver
inference_mode decorator around differentiable adaptation. Final inference remains no-grad.

## Compact diagnostics and canonical outcomes

Record L0/L1/L2, content/masked counts, gradient global L2 before EACH update, finite flags, fp32
master delta L2/relative L2 (denominator initial LN L2, epsilon1e-12), effective bf16 delta L2 and
changed-scalar count. Rounding away an update is valid scientific no-change, not INVALID.
On the COMMON fixed y_B trajectory record before/after each objective float32 entropy, script
P_E/P_M from unchanged partition and EOS probabilities at matched positions plus final query.
Token-weighted corpus mean entropy is used for the stated A1 confirmation-bias diagnostic only;
script mass/EOS traces are descriptive, never extra objective/selection features. A1 reuses its
three loss forwards; A2 may add one theta2 common-y_B forward, theta0 common baseline reused.

Final generated length/EOS/cap, token equality/first difference and token-level Levenshtein distance
vs forced/AUTO. Severe truncation: baseline content length>=10 AND adapted EOS-terminated length
<=floor(0.5*baseline length). No new severe-truncation event allowed; cap and length-ratio reported.
Canonical PIER/MER/EN-WER/ZH-CER, corrections/corruptions, matrix-ZH/embedded-EN retention, S/D/I,
P2-SEQ lexical outside-POI edits and baseline-correct-to-wrong harm (NOT acoustic outside-candidate
harm). Reference inputs stay evaluator-side AFTER all adaptation outputs sealed. Per-system
runtime, throughput, backward/optimizer/forward counts, peak allocated/reserved VRAM; reused
baseline timings unavailable rather than fabricated. Setup separately, episode/update/decode costs
included. No large vocab/activation/KV dumps. Tiny compressed LN/master checkpoints allowed for
independent update/reset audit; about1MB per objective/utterance before compression.

## Viability, confirmation bias and selection

Per objective, point safety ALL required (float guard1e-12):
MER increase<=.010; ZH-CER increase<=.015; matrix-ZH retention>=.98;
embedded-EN retention>=.95; lexical outside-POI harm<=.03; POI corruption rate<=.05;
added capped utterances<=1; NEW severe-truncation count0.
Empty baseline-correct rate denominators=>None and not-assessable/vacuous rate predicate; nonempty
primary reference-language/POI counts and finite main metrics required. No uncertainty-sign gate.

Useful signal: net POI error reduction>=2 OR at least3 utterances have fewer mixed errors than
B0-FORCED AND the count improved STRICTLY exceeds the count with more mixed errors. Equal error
counts are ties, not improvements; same reference denominator means error-count and MER ordering
are equivalent within each utterance. Count changes from canonical mixed alignment, not edit distance
between hypotheses. Both useful AND safety =>VIABLE. Otherwise NOT_VIABLE, except A1 below.

A1 confirmation-bias label takes precedence over ordinary NOT_VIABLE when common-y_B token-weighted
entropy falls>=1% from positive finite theta0 value AND ANY stated sequence/safety predicate fails.
This is entropy going down while real safety deteriorates; script-confidence rise alone is not
confirmation bias. A zero/empty entropy denominator makes this predicate false, explicitly logged.
Labels TTA0_EM_VIABLE/TTA0_EM_NOT_VIABLE/TTA0_EM_CONFIRMATION_BIAS and
TTA0_AC_VIABLE/TTA0_AC_NOT_VIABLE. Technical nonfinite/mismatch/incomplete failures supersede ALL
labels with P2_TTA0_INVALID; never classify a failed/missing episode as merely NOT_VIABLE.

At most1 advances: only one viable=>select it; neither=>P2_TTA0_NO_VIABLE_OBJECTIVE; both=>lower
corpus MER, then lower PIER, then lower ZH-CER, then lower mean per-utterance fp32-master update L2.
Treat floats within1e-12 as tied at each key, then exact tie choose A2 (fixed problem-specific
preference). All ties/keys are logged. Stage precedence INVALID, then OBJECTIVE_SELECTED iff at
least1 viable, else NO_VIABLE_OBJECTIVE. No retrospective step/LR/objective selection.

Descriptive paired dialogue bootstrap2000 draws seed240924, sorted20 dialogue IDs (one utterance
each), corpus count sums then ratios, pointwise95% quantiles. Shared draws across both objectives
and baselines; omit empty-denominator draws, disclose valid counts. No significance requirement,
confirmatory claim or attempt to generalize20 as100/full-panel performance.

## Audit, execution ceiling and stopping

Independent PASS_TO_P2_TTA0 before job, P2_TTA0_AUDIT: PASS afterward. No primary decision import;
reuse canonical primitives but own aggregation/bootstrap/classification/selection. Pre-run checks
panel/hash/names/optimizer/reset/suppression/teachers/no-leakage/tests/model/env/reuse seals and no
outcomes. In same allocation FIRST panel utterance, for EACH objective at theta0, independently
compute its loss/gradient on the same fixed teacher with an independent formula implementation:
loss abs<=1e-5 and norm(g_primary-g_auditor)<=max(1e-8,1e-3*norm(g_auditor)) on the complete
fp32 master-gradient vector (bf16 backward tolerance), no extra optimizer update.
Store comparison scalars, not full logits. This adds2 validation backwards/forwards, not new arms.
Post audit independently sums saved position entropy/target-logprob into L0/L1/L2, checks gradients,
master/effective updates, theta0 reset hashes, non-LN identity evidence, complete20x2 outputs,
canonical metrics/rules/selection. Tiny checkpoints may be used; do not infer reset from a success
boolean alone. Existing baseline/pseudo/input/provider provenance stays immutable.

One sbatch H100 MIG3g.40gb allocation,4CPU/48G; at most1 pending/running. Target<60min, hard120min.
20 encoder passes,40 adapted final decodes,20 theta0 forced-integrity decodes,80 optimizer updates,
120 objective teacher forwards plus<=20 A2 diagnostics+2 audit forwards;82 backwards including
2 independent checks. No new AUTO sampling, LID or D2. Model/optimizer/teacher graphs released
per episode. Exhaustion/nonfinite=>preserve partial artifacts,INVALID,STOP, no automatic extra job.

Selected=>P2_TTA0_TTA1_HANDOFF.md recommending a separately frozen100-screen; DO NOT execute TTA1.
Neither=>STOP, no more steps/LR/encoderLN/EM-tok/RL/prompt candidate. INVALID=>engineering-only
correction preserving attempt, no hyperparameter fallback. Every intentional reviewed edit:
check/test -> inspect diff -> commit -> push origin HEAD:cs-asr-steer-inf. Freeze/resolved manifest/
pseudo seal/preaudit remote before scientific outcomes; no force/main merge/PR.
Allowed exposed D-dev-select20 from existing100 and inference_cf artifacts. Forbidden new
router-calib role/confirm/test/P3/full100TTA/full300/SEAME/CS-FLEURS/ViMedCSS/ASCEND/transfer/fresh
validation. No outcomes/runner implementation in this Codex session. Core authority unchanged.
