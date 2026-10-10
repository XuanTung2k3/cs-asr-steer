# P2-SEL-T — Token-Localized English Evidence Diagnosis

Pre-T0 outcome freeze, 2026-10-06; starting commit `2810cf263854751d5cb68469879ab7a8cb2877c0`.
Machine contract: `configs/inference_cf/p2_sel_t.json`. No T0/T1/T2 outcome was computed in this design session.
This human-authorized stage follows the valid, terminal `P2_SEL_E_DIAGNOSIS_AMBIGUOUS` and its independent
PASS. It is bounded exposed-development work, separate from the active core method in
`docs/current/METHOD_CONTRACT.md`; no core-method, confirmation, test, or final ASR claim is authorized.
Historical reports and their stopping rules remain intact.

Question: **Does confident English evidence in the current 1 s window describe surrounding context,
rather than the acoustic region most attended by this decoder query?** Diagnose first; do not search
algebraic E transforms. The sole conditional repair is `R_TOK: E_new=E_tok`. Contrast is diagnostic only.

## Frozen components and sources

D2 is unchanged `p2dir_readout_v1` in `src/csasr/inference_cf/readout.py`: generate-suppressed float32
script-mass log-odds gradient at the unedited current B L16 state; existing float64 tangent projection,
epsilon normalization, float32 direction, cache isolation and invalid-direction no-edit fallback.
D2 script masses are distinct from acoustic language-token probabilities. Retain L16 DG-02
post-cross-attention/pre-FFN, alpha 2, phi identity, NormPreserve, no depth rescale, frozen R_B and
baseline `max_attention_window`. No direction/layer/alpha/null/Q/temporal/E-threshold search.

Acoustic probabilities use the exact 100-way `native_lid`, all frozen language IDs, EN/ZH 50259/50260.
Use `core_r2.local_support` unchanged: eps=1e-12,
`A=log((pi_E+eps)/(pi_M+eps))-log((null_E+eps)/(null_M+eps))`, `E=max(0,tanh(A/2))`.
Null is the frozen 480000-sample zero-waveform result. Q is relevant-pair mass, not a two-way softmax.
Provider/config/code, population, sealed D2, old-run and panel hashes are pinned in the config.
Do not change historical source functions to implement this stage.

Population is exactly the ordered 180 P2-RJ keys (80 utterances/20 dialogues), joined uniquely by
utterance, content step t and absolute predicting query. Retain old-E groups: EN-confusion TP42/FN18,
ZH-correct TN55/FP5, EN-correct60. Never regroup with E_tok or refill. Old gated D2 dialogue-equal
margins +3.520/-0.195 nat, 3 ZH corruptions, and the E0 ambiguous diagnosis are motivation only.

## T0 — diagnostic, no intervention or autograd

Reuse E_1s, exact W_1s and centered short-window probabilities from audited
`results/inference_cf/p2sel_e/e0_run1/rows/*.json`, plus its manifest/null and E0 analysis lineage.
The E0 files have no full attention vectors. Replay only unedited forced-ZH B prefixes through selected
queries using `p2r.DiagBranch.step(attention=True)`; encoder caching per utterance is allowed.
Feed only prompt plus baseline tokens preceding the predicted token t. Neither token t's reference
nor later tokens enter its current-query feature. Heard audio means the actual waveform made available
to the existing offline decoder (capped at 30 s), not evaluator token timing or an invented streaming
prefix. Every crop stays inside that already-available waveform and W_1s.

Sample rate 16000, encoder frame 320 samples. Given W_1s=[s0,s1), set n=min(8000,s1-s0):

| Candidate | Half-open sample bounds |
|---|---|
| LEFT | [s0,s0+n) |
| CENTER | [floor((s0+s1-n)/2),floor((s0+s1-n)/2)+n) |
| RIGHT | [s1-n,s1) |

When W is shorter than 0.5 s all three equal W; deduplicate identical LID calls. Do not pad or extend
crop bounds; the native provider's ordinary feature padding remains unchanged. The CENTER is exactly
P2-SEL-E `short_bounds` whenever its recorded bounds match this formula. A mismatch in the exposed180
is INVALID, not permission to substitute a new center. Reuse its raw probabilities and old null,
derive E_C with the same local_support. Only LEFT/RIGHT new LID calls are needed (at most360); no
new long/center/null inference. Verify the live current-query W matches stored W exactly, old mass
within 1e-6, query/key/provider identity, and E_1s reproduction within 1e-6.

Attention source is exactly `generation_config.alignment_heads`, all frozen pairs, current query only,
as returned by the existing cached branch. Average heads in float64 over the first
`valid=min(1500,ceil(heard_samples/320))` frames and normalize by their sum exactly as core_r2.
For candidate [a,b), mass is the sum over normalized frame weights times
`length([a,b) intersect [320*f,min(320*(f+1),heard_samples)))/length(that heard frame)`.
This explicit fractional-frame rule handles sample clipping at utterance ends and gives a whole heard
frame its whole mass. It changes no baseline W choice. Invalid/empty/nonfinite/negative/zero-total
attention invalidates T0. Persist normalized mean vectors (maximum1500 floats/key, small auditable NPZ)
and hashes, not all heads/hidden states; auditor independently recomputes masses and optionally
checks a source replay. Record all frozen alignment-head pairs in the manifest.

Select j*=first exact maximum in order L,C,R (no near-tie tolerance). E_tok=E_j*,
E_ctx=(sum of other two E)/2, C_tokctx=E_tok-E_ctx. The masses choose the crop BEFORE inspecting LID
values. E_tok is the only possible runtime repair. Also report a_tok and
`a_tok/(a_L+a_C+a_R+1e-12)` descriptively: overlapping candidates are not a probability partition.

Report each historical group's counts/dialogues, median/IQR E_1s/E_L/E_C/E_R/E_tok/E_ctx/C_tokctx,
a_L/a_C/a_R/a_tok, L/C/R selected fractions, positive E_tok rate; retain row-level values. Existing
P2-R evaluator timing may be displayed only diagnostically. No new alignment/oracle inference.

## T0 statistics and fixed decision

Use 10000 shared draws from numpy.random.default_rng(240924): sample20 IDs with replacement from
sorted fixed20-dialogue universe. Within each group average position values/rates per dialogue,
then average represented sampled dialogues equally, respecting multiplicity; omit empty group cells.
Compare groups with shared draws, each group's own represented denominator. Require >=9900 valid
draws for each required contrast; otherwise INVALID. Report pointwise90% percentile intervals;
decision support is one-sided80% lower (20th percentile), explicitly exploratory development uncertainty.
These are conditional exposed-data comparisons, not confirmation. Thresholds borrow historical
0.10 suppression/0.20 material-evidence scales and 0.25 contrast, not future L/R results.

Apply this precedence once:

1. **INVALID**: any population/group/provider/bounds/reuse/causality/hash/required-field/audit or
   bootstrap-validity failure. Stop.
2. **TOKEN_LOCALIZATION_SUPPORTED** iff all hold: at least4/5 historical ZH FP have E_tok<=0.10
   AND E_1s-E_tok>=0.25 spanning>=3 dialogues; at least34/42 EN TP have E_tok>=0.20 spanning>=3
   dialogues; dialogue-equal mean E_tok(TP)-E_tok(FP)>=0.25 with one-sided80% lower>0.
   Select only R_TOK.
3. **CONTEXT_CONTRAST_ONLY** iff at least4/5 FP have C_tokctx<=-0.25 spanning>=3 dialogues,
   at least26/42 TP have C_tokctx>=0 spanning>=3 dialogues, and dialogue-equal mean C(TP)-C(FP)
   >=0.25 with one-sided80% lower>0. Stop; a contrast gate needs another freeze.
4. **RECALL_ONLY** iff at least9/18 historical FN have E_tok>=0.20 across>=3 dialogues AND the
   ZH suppression predicate in step2 fails. Stop; recall alone does not solve safety.
5. **TOKEN_LID_NOT_DISCRIMINATIVE** otherwise. Stop; no supported safe token gate/contrast under
   these material criteria. Report partial predicates; this is not proof of no information. Next
   scientific family would be phonetic/lexical acoustic compatibility, not another LID transform.

No stopped label permits extra diagnostic jobs, repairs, threshold changes or contrast fallback.
Only supported plus `P2_SEL_T_AUDIT: PASS (T0)` permits an immutable selection artifact with all
row/source/attention/config hashes, exact formula and decision predicates. Commit/push it BEFORE T1.

## T1 — conditional single repair causal screen

Arms: C0 NONE; C_OLD=E_1s*R_B*D2; C_NEW=E_tok*R_B*D2. Same180 states/prefixes/evaluator. Reuse
P2-SEL C0/C2 (not C1 OLD direction) and sealed D2 after key/state/direction/logit/hook hash audit.
Only C_NEW pulses are new. A demonstrated old-lineage mismatch is INVALID and stops, not a rerun search.
No BROAD. Zero effective gate must be bitwise NONE. Cache crop/restore, parameter gradient isolation,
site reconstruction and bfloat16 NormPreserve energy use existing audited code.

Report dialogue-equal EN-confusion reference-margin delta, EN/ZH-correct margin delta and top1
corruption, positive E/gate/edit coverage, realized squared post-NormPreserve energy, relative energy,
benefit retention vs C_OLD, absolute ZH harm ratio vs C_OLD; historical5FP/18FN subgroup results
are descriptive only. References/evaluator gradients are evaluation-only.

Shared paired dialogue bootstrap as above; T1 six-endpoint two-sided90% Bonferroni simultaneous
intervals alpha=0.10/6 (percentiles alpha/2 and1-alpha/2), >=9900 valid draws. Family: new confusion,
new EN-correct, new ZH-correct margins; paired ZH new-minus-old; EN and ZH corruption rates.
All point rates are dialogue-equal; also report raw counts. Exact matched old margin denominators
are independently recomputed; rounded3.520/-0.195 are anchors, not substituted denominators.

All supported criteria:
- retention>=0.70 (about+2.464 nat); confusion simultaneous lower>0;
- absolute ZH harm ratio<=0.50; paired ZH improvement>=+0.10 nat and simultaneous lower>0;
- each correct margin simultaneous lower>=-0.25 nat; each observed corruption<=0.05 and simultaneous
  upper<=0.10; EN-correct new-minus-old mean>=-1e-6 and zero new corruptions;
- integrity and independent audit PASS. R_B=0 rows retain exact no-edit identity.

The 10% interval upper allowance is a small-sample development uncertainty budget, not acceptable
deployment damage. Freeze it now; never relax after outcomes. Invalid/nonfinite/zero old denominator
invalidates the comparison. Exact label precedence: P2_SEL_T_INVALID; then
P2_SEL_T_TOKEN_GATE_STILL_UNSAFE if any correct-state safety/no-EN-regression rule fails; then
P2_SEL_T_TOKEN_GATE_TOO_CONSERVATIVE if retention<0.70; then P2_SEL_T_NO_MATERIAL_GAIN if any remaining
confusion-CI/harm-ratio/paired-improvement/paired-CI criterion fails; otherwise
P2_SEL_T_TOKEN_GATE_SUPPORTED. All other labels STOP.

## T2 — conditional development mini decode

Only T1 supported plus `P2_SEL_T_AUDIT: PASS (T1)`. Reuse byte-identical P2_SEL_MINI_PANEL.json,
100 existing IDs/20 dialogues/5 each, SHA256
`266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`; do not reselect/refill.
Systems exactly B0 matched forced-ZH NONE; OLD=E_1s*R_B*D2; NEW=E_tok*R_B*D2.
Use existing cached B/E/S matched-current-prefix semantics and P2 frozen generation parameters.
At each live query NEW chooses its own W and attention-selected crop using only currently available
B-prefix state/audio. Do not transplant teacher-forced per-position E_tok into free decoding or
reuse OLD feature trajectories on diverged NEW prefixes. D2 uses the same live-state provider.
Identical crop LID outputs may cache by waveform identity/bounds/provider; no future token access.
AUTO only exact compatible cached result, never regenerate. No BROAD/full300.

Canonical PIER/MER/EN-WER/ZH-CER, corrections/corruptions, EN/ZH retention, outside harm, gate/edit
coverage, realized/relative energy, runtime, native_lid calls, D2 autograd calls, peak allocated/reserved
VRAM. Use existing canonical metrics/role-reference joins and dialogue uncertainty. Report paired OLD/NEW
pointwise90% development intervals with same20-dialogue bootstrap; no T2 success/selection test.
Always STOP even favorable; recommend separately frozen full300 confirmation, never execute it here.

## Audit, compute, firewall and GitHub

Before T0 require `PASS_TO_P2_SEL_T_T0`: exact parent terminal/PASS, source hashes, population/group,
provider100/head-map identity, crop CENTER compatibility, null and old reuse, no T outcomes, firewall.
Independent auditor must not import primary decision code. After T0 independently derive E, attention
masses, chosen crop, groups, bootstrap/predicates/label: `P2_SEL_T_AUDIT: PASS (T0)`. Before T1 verify
pushed selection and old reuse. After T1 independently recompute exact bfloat16 hook energy, margins,
corruption, intervals and verdict: `P2_SEL_T_AUDIT: PASS (T1)`. Before T2 verify panel; post independently
recompute metrics/coverage/counts/live feature lineage: `P2_SEL_T_AUDIT: PASS (T2)`. No unaudited conclusion.

Every run writes resolved config/spec/design/code/model/generation/provider/head-map/environment/git/
input/output hashes, parent manifests, runtime/call counts/VRAM/status using existing provenance utilities.
Compact rows + bounded mean-attention NPZ suffice; no redundant states/full heads. Focused tests are
listed in config/design and must pass before outcome jobs. Selection artifact immutable after outcome.

T0 at most one short H100/MIG sbatch job (no D2/autograd/steering); T1 at most one small job;
T2 at most one conditional job; at most2 pending/running. Target total below a few GPU hours;
request T0/T1 15-minute limits, T2 2-hour limit; on exhaustion preserve partial results and STOP
for review, no automatic additional job. No scientific jobs in this design session.

Allowed only exposed D-dev-select/P2-R/P2-RJ/P2-DIR/P2-SEL/P2-SEL-E artifacts/existing100 panel.
Forbidden router-calib new role, D-dev-confirm, D-test, P3, SEAME, CS-FLEURS, ViMedCSS, ASCEND,
Qwen/other transfer, fresh validation. Evaluator timing is descriptive only, never selection input.
P3 HELD. Test/check → inspect diff → commit → push origin HEAD:cs-asr-steer-inf for each reviewed
change; no force/main merge/PR/reset. Freeze implementation/tests and pre-run PASS before T0.
