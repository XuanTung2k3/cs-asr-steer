# P2-SEL-LAC — Local Acoustic Counterfactual Lexical Compatibility

**Pre-LAC0 outcome freeze, 2026-10-06.** Base `6421b0cae58e3e2836c8ff9eaeeb8e41668c0550`.
Config: `configs/inference_cf/p2_sel_lac.json`; implementation handoff: P2_SEL_LAC_CODEX_DESIGN.md.
No LAC candidate extraction, S_lex computation, masked inference or scientific GPU outcome in the design
session. This human-authorized bounded development stage accepts the terminal E ambiguous, T token-LID
not-discriminative and XA INVALID results and their independent audits. Never reopen them.
Separate from the core METHOD_CONTRACT; no core-method, confirmation or final ASR-improvement claim.

## Question, scope and fixed components

**Does actual local audio preferentially support the current English lexical candidate rather than the
competing Mandarin candidate?** This is direct forward-only local occlusion, not another LID formula or
small first-order source-vector probe. D2/L16 leverage remains accepted. Freeze D2 READOUT, L16 DG-02
post-cross-attention/pre-FFN, alpha2, identity dose, NormPreserve, E_1s/R_B, native LID and T window/j*.
No learning, different directions/layers/alphas/gates/masks/window lengths/transforms/threshold sweeps.

Exactly180 ordered existing P2-RJ positions,80 utterances/20 dialogues. Historical old-E groups:
EN-confusion TP42/FN18, ZH-correct TN55/FP5, EN-correct60. Unique uid/t/absolute predicting-query join;
never reclassify with new signal, refill or add a data role. Reference/evaluator labels only enter the
historical CPU group/evaluation joins, never candidate/window/mask/S/F construction.

Interpretation is candidate-specific next-token dependence on an occluded waveform region. Canonical
V_M includes Han-lead byte fragments and V_E ASCII-Latin subwords; these are not guaranteed complete
words, correct lexical alternatives, or a reference-consistent pair. Masking can affect neighboring
features, global feature normalization and contextual decoder states. A positive score is differential
model dependence, not proof of phonetic correctness. No outside-waveform samples change.

## Unmasked baseline and candidate seal

Reuse lossless P2-DIR Exp1 NONE logits, whose files are pinned by directions_sealed.json and whose
live cached-B equality was independently confirmed on all180 in XA0. Verify that audit PASS despite
its terminal scientific INVALID, all source/model/tokenizer/partition/suppression/prefix hashes, raw
NPZ seals, query keys and old C0/C2 lineage. No redundant unmasked GPU replay is needed. Missing or
mismatched baseline lineage invalidates; do not silently use approximate/reconstructed logits.

Canonical `core_r2.tokenizer_partition`, whisper_han_ascii_v1, hash
`sha256:7daa50091056677286dd30933300cb9b1245ca778f969df2a95ccc06016dac02`:
V_E=37858 embedded_ids (ASCII-Latin content); V_M=1667 matrix_ids (Han/Han-lead content). Same model
and partition, no filtering/rank cutoff/word completion. Current suppress_tokens/begin_suppress_tokens
have EMPTY intersection with both sets, verified from provider metadata in design. Pre-run audit must
reproduce this fact; a changed provider is INVALID, not an alternate candidate rule.

From raw unmasked float32 logits z (no rounding, probability or transcript):

    c_E = lowest token ID attaining max(z[V_E])
    c_M = lowest token ID attaining max(z[V_M])
    M0  = z[c_E] - z[c_M]

Candidates are selected ONLY from unmasked z. Sort IDs numerically before deterministic argmax; do not
assume incidental array ordering. Prepare `results/inference_cf/p2sel_lac/candidates_sealed.json` on CPU
with all180 candidate integer pairs, uid/t/query, complete input-prefix IDs/hash, W*/j hash, model/
partition/suppression and unmasked-file hashes. Verify serialization round-trip and commit/push before
ANY masked output. This is construction from exposed baseline outputs, not masked-outcome selection.
The masked evaluator receives sealed IDs and has no candidate-selection API.

Save candidate probabilities/ranks and top1 on both distributions descriptively. Probabilities/ranks
use generate-suppressed full-vocabulary scores, suppression unchanged, begin only at t0, temperature1;
rank=1+number of greater allowed scores+equal-score lower-ID tokens, top1 lowest ID on tie. Sealed
candidates remain unsuppressed. Primary margins use finite raw logits, not probabilities or ranks.
These descriptive fields cannot orient/select the repair.

## Counterfactual waveform and full-model replay

Load original x exactly via `csasr.models.whisper.load_audio`: soundfile float32, stereo mean if needed,
unchanged scipy resample to16000Hz if needed. Mask after resampling. W*=[a,b) is exactly
`p2sel_t/t0_run1/rows/{i:03d}.json positions[t].bounds[positions[t].j]`, half-open sample indices.
j* is the historical current-query attention argmax with earliest L/C/R ties. Do not recompute/select
j* from masked audio, references, evaluator timing or candidate outcomes. Same bounds inside W_1s;
length=min(8000,W_1s length). Never extend a short window.

ONE mask:

    x_mask = x.copy()
    x_mask[a:b] = np.float32(0.0)

No taper, noise, fade, cropping, signal normalization, file export, sample deletion or length change.
Hard in-window silence is the simplest exact occlusion. Boundary discontinuities and off-manifold
silence are limitations of this frozen intervention, not grounds for a second mask after outcomes.
Keep x[:a] and x[b:] byte-identical, including unused waveform tail. Preserve float32 shape/length,
finite values. Record original/local energy in float64, removed-energy fraction
`sum(x[a:b]^2)/(sum(x[:heard_samples]^2)+1e-12)` and waveform/mask/outside-slice hashes. If original local energy>0, at least one numeric sample MUST change; all masked samples are
positive float32 zero. Zero-energy/no-op windows are VALID. If x_mask bytes equal x, reused unmasked
logits and S=0 are allowed; do not omit such positions.

Heard audio is the existing OFFLINE available waveform, capped at480000 samples by Whisper's ordinary
30s feature path: 0<=a<b<=heard_samples=min(len(x),480000), matching T. This is not a streaming method
or a token-time-based hearing boundary. No waveform outside the existing input is accessed or appended;
no future DECODER token beyond prefix y_<t enters either forward.

Features must use the same `batch_model_inputs` processor call semantics on a waveform list:
`feature_extractor([x_mask],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)`;
unchanged default padding/truncation/log-mel normalization and model-device bf16 cast. Carry its real
attention_mask into the encoder. A tiny in-memory adapter is permissible in implementation, but before
LAC0 a CPU check on all80 original waveforms MUST reproduce batch_model_inputs unmasked features and
attention masks bitwise. Do not write masked WAVs and re-read with quantization, zero mel/encoder frames,
or reuse original encoded states. Derived features outside W* may change naturally.

Run the whole masked encoder, then an isolated unsteered forced-ZH B branch with fresh self/cross KV
cache. Identical prefix is cB=[50258,50260,50360,50364]+original baseline tokens[:t]. Use existing cached
B.step call sequence: prompt first, then one baseline token at a time until the predicting query.
Use attention=True on every cached step, as the frozen baseline.
No token t/reference/future target and no masked generation. Earlier decoder states must be recomputed
under masked audio: **never transplant unmasked self/cross KV caches or hot-swap their encoder**.

One counterfactual evaluation per position includes one masked encode plus the necessary prefix steps;
not a promise of one cheap current-query decoder call. Identical (original waveform hash,bounds,
model/provider hash) can share an encoder. Use a bounded LRU of at most2 masked encoder/branch entries
per utterance; a reused branch's fed IDs must be an exact prefix of current input IDs. Otherwise discard
and cold-rebuild it. No cross-utterance or cross-system cache transport. Deduplicate only identical
counterfactual inputs; record actual encoder/decoder/counterfactual call counts. Do not use full-prefix
one-call replay with a different numerical kernel to compare against cached baseline for convenience.

## LAC0 signal, descriptive report and integrity

At the SAME sealed candidates:

    Mmask   = z_mask[c_E] - z_mask[c_M]
    Delta_E = z[c_E] - z_mask[c_E]
    Delta_M = z[c_M] - z_mask[c_M]
    S_lex   = M0 - Mmask = Delta_E - Delta_M
    F_lex   = max(0, tanh(S_lex/2))

CPU float64 arithmetic from lossless float32 logits; identity absolute tolerance1e-10. Positive S means
local removal hurts the English candidate relative to Mandarin; negative means the reverse. Only S_lex
may construct a repair. No alternative aggregate-script score, KL, Delta_E-only or rank gate.

Report per historical group counts/dialogues, median/IQR and dialogue-equal means of S/M0/Mmask/
Delta_E/Delta_M/F; candidate IDs/texts/probabilities/ranks/top1, old E_1s/E_tok/R_B, and removed-energy
fraction descriptively. Token texts are tokenizer-rendered labels, never transcript inputs.
Minimal zero-cost controls are precisely these fields; no feature search.

Validity: exact population/group/W*/j/query, finite waveform/logits with frozen vocabulary shape,
exact outside-waveform bytes, required in-window zero/change, model/tokenizer/partition/suppression/
feature/prefix identity, baseline seals/replay evidence, serialization seal and absent reference/future/
autograd/steering/LID. Model params frozen, eval, torch.inference_mode; no parameter grads/hooks/cache
transport. No invalidation for small S, small removed energy, small logit change, candidate probability,
rank or absent finite-difference materiality. Unlike XA, no derivative/linearity prerequisite.

## Frozen LAC0 statistical decision

Raw nat thresholds are chosen before masks. Strong English effect is
`s_strong=2*atanh(0.70)=log(17/3)=1.7346010553881064`; weak/non-English is
`s_weak=2*atanh(0.10)=log(11/9)=0.20067069546215124`. These are inverse images of fixed70% retention/
10% residual factor values under the sole natural log-odds/tanh mapping, not thresholds fitted to5FP.
Group separation≥0.50 nat retains the prior materiality scale, well above the0.05–0.1 near-inert regime.

Bootstrap10000 shared draws, numpy.random.default_rng(240924), sorted fixed20 dialogue IDs sampled20
with replacement. Mean within each group/dialogue, then equal represented drawn-dialogue means retaining
multiplicity; omit empty group cells. TP-minus-FP uses shared draws and group-specific denominators.
Require>=9900 valid contrast draws; pointwise90% descriptive intervals, one-sided80% support lower is
20th percentile. This is conditional exposed-development uncertainty, not independent confirmation.

Exact precedence:
1. **P2_SEL_LAC_INVALID** for integrity/provider/population/required-data/audit failure or insufficient
   valid draws. STOP; never silently refill, reselect or change mask.
2. **P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED** iff >=34/42 EN TP have S>=s_strong across>=3
   dialogues, >=4/5 ZH FP have S<=s_weak across>=3 dialogues, and dialogue-equal mean S(TP)-S(FP)
   >=0.50 nat with one-sided80% lower>0. Engineering/audit valid. Select ONLY F_lex above.
3. **P2_SEL_LAC_RECALL_ONLY** iff not supported and >=9/18 historical EN FN have
   S>=s_strong across>=3 dialogues. STOP. This characterizes differential
   dependence, not actual recovery: multiplicative unchanged E remains zero on these rows.
4. **P2_SEL_LAC_NOT_DISCRIMINATIVE** otherwise. STOP. Candidate-specific dependence is insufficient
   under frozen safe-separation criteria; report partial predicates, not proof of absent information.
   Next family would be explicit phonetic/acoustic-token compatibility, not another script/LID signal.

No outcome-driven threshold/formula/mask changes. Only supported plus `P2_SEL_LAC_AUDIT: PASS (LAC0)`
permits immutable lac0_selection.json (sole formula, candidate/seal/row/source/config hashes, all
predicates and independent audit links), committed/pushed BEFORE LAC1 outcomes.

## LAC1 — one conditional causal repair

Only supported/audited/pushed selection. Exactly C0 NONE, C_OLD=E_1s*R_B*D2,
C_NEW=E_1s*R_B*F_lex*D2. Frozen L16/site/alpha2/NormPreserve and original unmasked states. Reuse P2-SEL
C0/C2 (not C1 old direction), sealed D2 and original reference-margin evaluator after exact lineage
checks. Only C_NEW pulses are new. Reuse sealed LAC0 F; no masked steering branch or masked prefix
cache enters the actuator. Invalid lineage stops, no rerun search. No BROAD/other factor/composition.
E=0 remains zero. Require0<=g_new<=g_old<=1 within1e-7 and zero effective gate bitwise NONE.

Metrics: dialogue-equal EN-confusion/EN-correct/ZH-correct reference-margin delta, correct-state top1
corruption, new/old benefit retention, absolute ZH harm ratio and paired ZH improvement, F/gate/edit
coverage, squared realized and relative post-NormPreserve energy; historical5FP/18FN descriptive only.
Exact matched old denominators, not rounded+3.520/-0.195 anchors. Reference data are evaluation-only.

Use10000 shared paired dialogue draws, seed240924, >=9900 valid; six-endpoint two-sided90% Bonferroni
simultaneous intervals alpha=0.10/6, quantiles alpha/2 and1-alpha/2. Family: new three-stratum margins,
paired new-minus-old ZH margin, EN and ZH corruption rates. ALL success requirements:
- benefit retention>=0.70 (about+2.464 nat), confusion simultaneous lower>0;
- ZH absolute harm ratio<=0.50; paired ZH improvement>=+0.10 nat with simultaneous lower>0;
- each correct margin lower>=-0.25 nat, observed dialogue-equal corruption<=0.05, simultaneous
  upper<=0.10; EN-correct new-minus-old mean>=-1e-6 and zero new corruptions;
- state/cache/zero-gate/provenance/independent audit PASS.

The10% upper uncertainty allowance is small-sample development-only, not acceptable deployment damage.
Nonfinite/zero old denominator invalidates. Precedence: P2_SEL_LAC_INVALID; then
P2_SEL_LAC_GATE_STILL_UNSAFE if any correct-state safety/no-EN-regression rule fails; then
P2_SEL_LAC_GATE_TOO_CONSERVATIVE if retention<0.70; then P2_SEL_LAC_NO_MATERIAL_GAIN if any remaining
benefit-CI/harm-ratio/paired-ZH-improvement/paired-CI condition fails; otherwise
P2_SEL_LAC_GATE_SUPPORTED. Every non-supported label STOP. No safety relaxation after outcomes.

## LAC2 — conditional development mini decoding

Only gate supported and `P2_SEL_LAC_AUDIT: PASS (LAC1)`. Existing P2_SEL_MINI_PANEL.json100 IDs/20
dialogues/5 each; byte SHA256 `266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`.
No new panel/refill. Exactly B0, OLD E*R_B*D2, NEW E*R_B*F_lex*D2. No BROAD/AUTO/full300.
Greedy beam1/max_new_tokens200, bf16, frozen generation suppression, cB/cE as P2, no forced-prefix edit.
Baseline/OLD reuse only with exact output IDs/model/prefix/generation/method lineage; otherwise generate
these required systems in the single LAC2 job. Historical old-direction P2 outputs are NOT OLD-D2.

For live NEW prefix, compute unmasked B logits/E/R_B/attention first. If old gate is zero, exact no-edit
and skip masked counterfactual/D2, record factor not evaluated. Otherwise seal current c_E/c_M, choose
W* with UNCHANGED T current-query rule on original audio, mask/re-encode/rebuild prefix and compute
F. Never transport stored180 teacher-forced F/W/candidates onto a diverged transcript or choose W/candidates
from masked outputs. Discard masked trajectory after feature extraction; only original B/E/S and original
D2 scratch enter steering. New gate zero gives no edit and may skip D2. Counterfactual itself is forward-only.
Bounded2-entry cache follows the exact prefix/hash rule above; no stale unmasked/OLD branch transport.

Canonical PIER/MER/EN-WER/ZH-CER, corrections/corruptions, EN/ZH retention, outside harm, gate/edit
coverage, realized/relative energy, counterfactual calls/unique encodes/decoder steps, D2 autograd and
native_lid calls, runtime, utterances/sec and audio-sec/sec throughput, peak allocated/reserved VRAM.
Report setup/warmup separately; include counterfactual cost in end-to-end NEW runtime. Corpus-ratio metrics
are recomputed from summed utterance counts per shared20-dialogue draw; pointwise90% paired development
intervals,10000 draws seed240924. No LAC2 pass/selection claim: always STOP even positive, recommend a
separately frozen full300 confirmation, never run it here.

## Tests, independent audit, provenance, compute and firewall

Focused tests in design/config cover population/W*, raw unmasked candidate selection/ties/seal, original
feature-adapter identity, mask bounds/outside bytes/energy/no-op, natural masked encoder and fresh prefix
cache, S identity/F bounds/reference-free forward-only behavior, D2/E/R_B/site/alpha regression,
zero gate identity, one repair/no fallback, and exact100 panel/live-prefix construction.

Before masks require `PASS_TO_P2_SEL_LAC_LAC0`: accepted parents/PASS, all180 baseline/T/W/old lineage,
provider/partition/suppression identity, CPU unmasked-feature equality, candidates_sealed.json already
pushed, constants/firewall/no LAC outcomes. Independent auditor must NOT import primary analysis or
decision code. After LAC0 independently reconstruct candidates/W/mask/outside hashes/prefix/cache
lineage, S/F, group summaries/bootstrap/predicates/label: `P2_SEL_LAC_AUDIT: PASS (LAC0)`. Before LAC1
verify pushed immutable selection/old lineage; post recompute exact bf16 edit energy, evaluator margins/
corruptions/CIs/verdict: `P2_SEL_LAC_AUDIT: PASS (LAC1)`. Before LAC2 panel; post canonical metrics/counts/
live-input lineage/cost: `P2_SEL_LAC_AUDIT: PASS (LAC2)`. No scientific conclusion without stage audit PASS.
Mask/prefix/cache isolation checked independently from source waveform hashes, processor calls and tests;
auditor uses compact lossless logits, never imports primary decision code. No extra GPU audit job required.

Every run manifest resolved config/spec/design/code/model/tokenizer/processor/environment/git/input/
output hashes, prefix/candidates/W*, parent manifests, status/runtime/call/VRAM counts. Use existing
provenance/specfreeze utilities. Save compact rows plus lossless bf16 raw logits and per-file hashes;
source audio stays in place, no masked WAV/high-dimensional encoder/cache dumps. Keep binary artifacts
available locally with committed hashes and all scientific scalar/seal/selection/audit state on GitHub.

At most one short15min LAC0 sbatch H100/MIG job; one conditional15min LAC1; one conditional120min
LAC2; max2 pending/running. Few GPU hours total; preserve partial artifacts and STOP on exhaustion, no
automatic additional job. LAC0: at most180 counterfactual evaluations, zero backward/D2/LID/steering;
actual prefix/encode counts reported, exact dedup encouraged. No scientific jobs in this design session.

Allowed only exposed D-dev-select/P2-R/RJ/DIR/SEL/SEL-E/SEL-T/SEL-XA artifacts/existing100 panel.
Forbidden router-calib new role, D-dev-confirm, D-test, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND,
Qwen/other transfer, fresh validation. References never choose candidates/W/mask/S/F. P3 HELD.
Every reviewed edit test/check→diff→commit→push origin HEAD:cs-asr-steer-inf; no force/main/PR/discard.
Freeze tested implementation/pre-run audit on clean pushed sources before masked outcomes.
