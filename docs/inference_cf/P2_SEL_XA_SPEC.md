# P2-SEL-XA — Cross-Attention Acoustic Compatibility Diagnosis

**Pre-XA0 outcome freeze (2026-10-06).** Base: `5186b5caba06ae1bd4ede7a340e10088a0fd541c`.
Machine contract: `configs/inference_cf/p2_sel_xa.json`; implementation map: P2_SEL_XA_CODEX_DESIGN.md.
No XA0/XA1/XA2 scientific outcome or experiment implementation in this design session.
This separately human-authorized bounded development stage preserves the valid terminal P2-SEL-E
`P2_SEL_E_DIAGNOSIS_AMBIGUOUS` and P2-SEL-T `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`. Do not rerun them.
It is not the core method in METHOD_CONTRACT.md and makes no core-method, confirmation or test claim.

## Scientific question and frozen scope

Four of five historical Mandarin-correct false positives retain strong English native-LID evidence even
in the query's most-attended half-second crop. Ask instead: **Does the actual cross-attention source
contribution used by this query locally support the downstream English rather than Mandarin decision?**
The diagnostic is a directional derivative with respect to source gain, not a pure causal attribution
of acoustic language identity: u_source is query-conditioned and includes the frozen output projection/
bias; downstream J can reflect decoder context. The fixed finite-difference check validates only a
local source-gain response, not source removal, lexical correctness or global attribution.

Keep D2 `p2dir_readout_v1`, L16 DG-02 post-cross-attention/pre-FFN, R_B, E_1s, alpha2, identity dose,
NormPreserve, native100 LID/null/epsilon, localizer and P2-SEL-T windows unchanged. No alternate direction,
gradient, script partition, source, head/layer/window/alpha/lambda/threshold search or new LID transform.
XA0 has no D2 steering. Its sole nonzero intervention is the prespecified diagnostic scratch source
scaling; no changed state/logit/token/cache is ever propagated into the baseline trajectory.

Exactly180 existing ordered positions (80 utterances/20 dialogues), historical old-E grouping:
EN-confusion TP42/FN18, ZH-correct TN55/FP5, EN-correct60. Unique uid/t/absolute predicting-query joins;
never regroup by compatibility, refill or use a fresh role. Config pins source/population/model-provider
lineage. References, strata, evaluator gradients, CTC/timing, outcomes and future tokens never construct,
orient or select S_src/C_src/F_src. Groups are used only in CPU diagnosis after feature construction.

## XA0 — construction and exact reuse

Capture q from encoder_attn_layer_norm pre-hook and u_source directly from encoder_attn output[0], at
the predicting query in the unedited forced-ZH B branch. Under eval/dropout disabled, r is the actual
model-dtype q+u_source before FFN. Reuse DecoderPostCrossAttnRecorder; do not approximate u by h differences
or use post-FFN states. Retain native-bfloat16 values and float32 lossless upcasts for serialization.
Require native-dtype sum bitwise equal r and sealed h_B. Float64 q+u versus saved r may differ through
rounding: relative max error `max|q64+u64-r64|/max(max|r64|,1e-12)` must be at most
`max(1e-3,4*finfo(model_dtype).eps)` (0.03125 for bf16), the frozen DG-02 tolerance family.

J is exactly `readout.objective`: log(P_E+1e-12)-log(P_M+1e-12). P_E/P_M are full-vocabulary
script masses from whisper_han_ascii_v1 after unchanged generate suppression (begin suppression only
at step0), float32 logits, temperature1, stable logsumexp/logaddexp. They are NOT native language-token
probabilities. g_J is the existing zero-probe raw float32 autograd gradient, before tangent projection,
unit normalization or NormPreserve geometry. Never substitute D2's unit/tangent direction or g_ref.

All180 raw `t{t}_g_readout` vectors already exist, float32 length1280, hash-sealed in P2-DIR Exp1 and
audited there. XA0 MUST reuse these, not run another backward. Verify sealed row/NPZ/logit hashes,
query/prefix/model/suppression/partition identity, and live B site/NONE logits bitwise against that seal.
Missing/mismatched gradient lineage is INVALID; no alternate extraction or gradient fallback.
Only replay B once per utterance to capture current q/u and perform the fixed validation forward.
E_1s/R_B come from P2-R/P2-SEL; E_tok only from audited P2-SEL-T compact rows. No LID inference.

In CPU float64 from exact captured u and raw float32 g upcast:

    S_src = dot(g_J, u_source)
    C_src = S_src / (norm(g_J)*norm(u_source) + 1e-12)
    F_src = min(1, max(0, S_src))

Zero norms imply all three zero; no norm threshold, fitted scale, coefficient or fallback. Nonfinite
inputs/results invalidate. Require C in [-1,1] within1e-6 and matching signs of S and C. **S is the sole
selection/gate signal. C is a descriptive scale-free companion**, not a second route. Different rankings
are possible because norms vary; report them without changing F or decisions. No sign disagreement is
mathematically valid given the positive denominator; such a mismatch is numerical INVALID.

The factor is the simplest bounded rectifier of the signed, dimensionless nat-valued source-gain
sensitivity: zero for non-English support, full weight once sensitivity reaches1 nat per unit gain.
The cap is fixed now, not a coefficient chosen from five false positives. XA0 evidence is allowed to
reject it; do not optimize its mapping or switch to a cosine gate. It is uncalibrated development evidence.

## Fixed finite-difference validation

Only lambda1 and lambda0.95. Preserve q and all other queries/layers; replace this query's u with
`(0.95*u).to(model_dtype)` in one encoder_attn output hook. This is a scratch diagnostic, not a repair;
no NormPreserve/alpha/D2 update in this check. Clone the exact pre-step B cache/encoder tensors with
readout.clone_scratch, discard the scratch cache/graph/outputs after computing J. Keep persistent
B unchanged and matched baseline tokens. A lambda1 identity test is CPU/synthetic; no extra baseline
GPU forward beyond the ordinary clean B step. Save r1/r0.95 and lossless raw logits for audit.

Per key:

    observed = J(r0.95) - J(r1)
    P_nominal = -0.05 * S_src
    P_realized = dot(g_J, r0.95-r1)

P_realized accounts for bf16 source scaling and residual-add rounding. Both predictions must be
reported. Use one label-free material subset: |P_nominal|>=0.02 AND |P_realized|>=0.02 nat.
Require at least30 rows spanning>=10 dialogues. On this subset require:
- observed and P_realized signs agree for>=90% of rows (observed zero counts as disagreement);
- P_nominal and P_realized signs agree for>=90%;
- median signed observed/P_realized lies in [0.5,1.5];
- relative RMS error `sqrt(sum((observed-P_realized)^2)/sum(P_realized^2))<=0.50`.

All checks must pass for diagnostic validity. Report subset size, dialogue coverage, row residuals,
nominal and realized predictions on all groups; no stratum-specific exclusion. Failure/insufficient
material rows means INVALID for this first-order interpretation; it need not prove a code bug. STOP,
no smaller lambda, float32 rerun or alternate derivative in this stage. These broad tolerances address
a cheap bf16 local check; not a claim of a globally linear model.

## Group summaries, uncertainty and decision

Primary42TP vs5FP; supporting18FN/55TN/60EN-correct. Report counts/dialogues, median/IQR and
dialogue-equal means of S/C/norm(u)/norm(g)/J/E_1s/E_tok/R_B, plus F and validation quantities.
No high-dimensional feature search or additional evaluator-gradient criterion. Source scaling NEVER
creates a recognition outcome used to select F. Reused evaluator information may be displayed only
after independently constructed features, never orientation or an inference feature.

Use10000 shared draws, numpy.random.default_rng(240924), sorted fixed20-dialogue universe sampled20
with replacement. Average group values/rates within dialogue, then equally over represented drawn
dialogues retaining multiplicity; omit empty group cells, require>=9900 valid draws for each required
contrast. Pointwise90% percentile descriptive intervals; one-sided80% lower=20th percentile for the
single decision contrast. This is conditional exploratory development uncertainty, not confirmation.

Apply once in this exact precedence:
1. **INVALID**: integrity/site/provider/gradient/population/finite-difference/audit failure or insufficient
   valid draws. STOP. No repair or silent replacement.
2. **SOURCE_COMPATIBILITY_SUPPORTED** iff at least4/5 historical ZH FP have F_src<=0.10 across>=3
   dialogues, at least34/42 historical EN TP have F_src>=0.70 across>=3 dialogues, and dialogue-equal
   mean F(TP)-F(FP)>=0.50 with one-sided80% lower>0. All validity checks pass. The0.10/0.70 bounded-factor
   criteria are fixed strong suppression/large-majority retention requirements, not fitted XA scales.
3. **SOURCE_COMPATIBILITY_RECALL_ONLY** iff not supported, the preceding FP safety predicate fails,
   and at least9/18 historical FN have F>=0.70 across>=3 dialogues. STOP. This means available English
   compatibility among misses, NOT actual recovery: unchanged multiplicative E is zero on these rows.
4. **SOURCE_COMPATIBILITY_NOT_DISCRIMINATIVE** otherwise. STOP. No materially supported safe
   discrimination under the frozen rule; report partial predicates, do not claim proof of no information.
   Next family would be lexical/phonetic token compatibility rather than another script/LID transform.

Only supported plus `P2_SEL_XA_AUDIT: PASS (XA0)` permits selection. Immutable
`results/inference_cf/p2sel_xa/xa0_selection.json` records sole F clip formula, all row/vector/probability/
source/config hashes, predicates and audit links. Commit/push BEFORE XA1 outcomes. No C gate or fallback.

## XA1 — conditional one-gate causal screen

Only supported/audited/pushed selection. Exactly NONE, OLD=E_1s*R_B*D2, NEW=E_1s*R_B*F_src*D2.
Same180 B states/prefixes and frozen reference lexical-margin evaluator. Reuse P2-SEL C0/C2 (not C1
historical old direction), sealed D2, same hook/bf16 NormPreserve and exact key/state/logit identities. Require
0<=g_new<=g_old<=1 within1e-7.
Only NEW pulses are new. Invalid old lineage stops, not a repeat search. No BROAD or alternative arm.
E=0 remains zero, so this chosen repair cannot recover historical18FN; report this limitation.

Dialogue-equal EN-confusion and EN/ZH-correct margins, correct-state top1 corruption, benefit retention,
absolute ZH harm ratio, paired ZH improvement, F/gate/edit coverage and squared realized/relative energy.
The historical5FP/18FN results are descriptive only. Use exact recomputed matched old denominators;
+3.520/-0.195 nat are rounded anchors, not substituted exact values.

Use10000 shared paired dialogue draws as above, six-endpoint two-sided90% Bonferroni simultaneous
intervals alpha=0.10/6, quantiles alpha/2 and1-alpha/2, >=9900 valid draws. Family: new confusion,
EN-correct, ZH-correct margins, paired new-minus-old ZH margin, EN/ZH corruption rates.
Success requires ALL:
- benefit retention>=0.70 (about+2.464 nat), confusion simultaneous lower>0;
- ZH absolute harm ratio<=0.50; paired ZH improvement>=+0.10 nat with simultaneous lower>0;
- each correct margin simultaneous lower>=-0.25 nat; observed dialogue-equal corruption<=0.05,
  simultaneous upper<=0.10; EN-correct new-minus-old mean>=-1e-6 and zero new corruptions;
- exact no-edit at zero gate, state/cache/provenance and independent audit PASS.

The10% upper uncertainty allowance is development-only, not permissible deployment damage. Invalid/
zero/nonfinite denominator invalidates; never relax budgets. Precedence: P2_SEL_XA_INVALID; then
P2_SEL_XA_GATE_STILL_UNSAFE if any correct-state safety/no-EN-regression check fails; then
P2_SEL_XA_GATE_TOO_CONSERVATIVE if benefit retention<0.70; then P2_SEL_XA_NO_MATERIAL_GAIN if any
remaining benefit-CI/harm-ratio/paired-ZH-improvement/paired-CI condition fails; else
P2_SEL_XA_GATE_SUPPORTED. Every non-supported result stops.

## XA2 — conditional development mini decoding

Only gate supported plus `P2_SEL_XA_AUDIT: PASS (XA1)`. Existing P2_SEL_MINI_PANEL.json,100 IDs/20
dialogues/5 each, byte SHA256 `266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5`.
No selection/refill. Exactly B0 forced-ZH NONE, OLD=E*R_B*D2, NEW=E*R_B*F_src*D2; no BROAD/AUTO arm.
No full300. In free decoding recompute features from each system's own live shared B/E/S content
prefix: reuse one reference-free D2 scratch gradient to obtain both unchanged direction and S_src.
Never wrap readout in an active recorder: obtain gradient from pre-step scratch first, then capture
current unedited q/u in a separate clean B step and close its recorder.
Do not use teacher-forced/stored per-position F on diverged transcripts. Lambda validation is absent
at deployment; no extra backward beyond the live D2 readout. Existing invalid-D2 fallback remains
no-edit; nonfinite feature/provider rows fail the run/audit, never use another compatibility factor.

Canonical PIER/MER/EN-WER/ZH-CER, corrections/corruptions, EN/ZH retention, outside harm, gate/edit
coverage, realized/relative energy, runtime, autograd/native_lid counts, peak allocated/reserved VRAM.
Greedy beam1, max_new_tokens200, bf16, cB=[50258,50260,50360,50364], cE=[50258,50259,50360,50364],
frozen generate suppression and no forced-prefix edit, as P2. Recompute corpus ratios from summed
utterance-level counts in each dialogue draw (not mean utterance error rates). Use canonical
evaluators and shared20-dialogue pointwise90% development intervals for paired OLD/NEW.
No success/selection label from XA2: always STOP even positive; separately frozen full300 design needed.

## Engineering, tests, audit, compute, firewall and GitHub

Focused tests are frozen in the design/config: exact180 group/key joins, direct q/u/site capture and
rounding reconstruction; unchanged J/raw-gradient provenance; S exact dot/C normalization/F clip;
fixed source scaling/query isolation/finite-difference synthetic sanity; absent model parameter grads,
cache mutation, reference/evaluator/future leakage; zero-gate identity; one formula/no sweep/fallback;
exact mini-panel/live-prefix behavior. Reuse existing D2/site/P2-SEL/P2-SEL-T tests.

Before XA0 require `PASS_TO_P2_SEL_XA_XA0`: parent terminal/PASS, sealed gradient inventory and hashes,
source/panel/model/partition/suppression identity, population/group/old lineage, constants, no XA
outcomes and firewall. Independent auditor must not import primary decision code. XA0 auditor independently
recomputes q+u, J from saved logits (absolute tolerance1e-4, inherited P2-DIR CPU/GPU allowance), S/C/F, finite-difference predictions/metrics, groups, bootstrap and
selection: `P2_SEL_XA_AUDIT: PASS (XA0)`. Before XA1 verify pushed immutable selection/old lineage;
post independently recompute exact bf16 edit energy, margins, corruption/intervals/label:
`P2_SEL_XA_AUDIT: PASS (XA1)`. Before XA2 panel identity; post independently recompute canonical metrics,
coverage/calls and live feature lineage: `P2_SEL_XA_AUDIT: PASS (XA2)`. No unaudited scientific conclusion.

Manifest resolved config/spec/design/environment/git/model/source/parent/input/output hashes, status,
query/precision/provider metadata, counters and runtime using existing provenance/specfreeze utilities.
Save q/u/r1/r0.95/g vectors and lossless raw logits1/0.95 for180 rows (bounded few MB, not cache/head dumps);
record gradient-reuse hashes. Analysis reproducible on CPU. Include exceptions/termination, do not omit rows.

At most one small sbatch MIG/H100 job XA0 (clean replay plus one.95 scratch/key, zero new backward/LID),
one conditional XA1, one conditional XA2; max2 pending/running. Limits15/15/120minutes, fewGPUhours total.
Exhaustion preserves partial artifacts and stops for review, no automatic extra job. No scientific
job in this Codex session. Allowed only exposed D-dev-select, P2-R/RJ/DIR/SEL/SEL-E/SEL-T artifacts and
existing100 panel. Forbidden router-calib new role, confirm/test/P3/transfer/fresh validation. P3 HELD.
Every reviewed edit: test/check → inspect diff → commit → push origin HEAD:cs-asr-steer-inf. Freeze
implementation/tests and pre-run audit on clean pushed sources before any XA0 outcome; no force/main/PR.
