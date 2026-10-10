# P2-A2-MECH0 — Episodic A2 adaptation dynamics and termination mechanism

**PRE_OUTCOME_FREEZE; development mechanism/robustness diagnosis, not a method search.**
Starting local/remote HEAD `fb302dc875c1cbc0749381f810e1cdc272999f1b`, branch
`feature/inference-cf-steering`, clean tree, empty user queue. Latest commit restores current docs
truncated in the prior PATH5 reporting commit; preserve that restored history. PATH5 is audited
`P2_PATH5_SAFE_NO_ADDED_VALUE`: controller line CLOSED, main candidate A2. This separate inference_cf
supporting stage does not revise the core v6 METHOD_CONTRACT or reopen PATH/E/T/XA/LAC.
No runner, adaptation, scientific decoding, reference scoring, or GPU job in this design session.

## Question and evidence limits

Does exact historical A2 stabilize forced-ZH decoding, substantially by recovering premature EOS and
excessive deletion? PATH5 exposed NEW200 A2 ZH rescue281, including274 on eight theta0-EOS/A2-content
rows; deletions fell1986→1694. These are motivating exposed-development observations, not new MECH0
measurements or an independently validated causal termination effect. We freeze state and parameter-family
interventions now, before any new diagnostic output. No result changes A2 or selects a one-step/subset method.

## Panel and reference-free strata

Exact PATH5 FULL300: 300 rows,20 dialogues×15, canonical original parent order (utterance ID),
source `results/inference_cf/p2_A_r1_L16/panel.json` byte SHA256
`a1eca10644a400034cfc14ccb44bea863e6957b3606ab38d266eb29b8ad5987e`.
FULL300 membership/order hash
`bc8062390705cf133cbd42813d01b65f7ff3d5db836c5b94d9fc602275db6c5c`.
`P2_A2_MECH0_PANEL.json` fingerprints IDs/dialogues/order, all300 audio, sealed AUTO teachers and masks,
PATH5 canonical live B0/A2 tokens/text/termination, all300 final effective-state hashes,100 available
historical fp32 final-master archives, plan/model/seal/source provenance. No reselection/new role.
Do not use old P2-A B0: its execution path differed; authoritative B0 is PATH5 row `theta0`.

Token equality includes EOS/cap termination: compare generated content IDs plus one logical EOS iff
terminated=eos; never compare decoded text only. Exclude forced prompt/control prefix; stored content
has no EOS. Identical tokens with different termination are a difference, not silently SAME. Strict
prefix ending EOS gives the appropriate first EOS/content divergence. Cap is not EOS; unavailable
first action, control/special incompatibility or invalid structure is OTHER_INVALID, requiring STOP.
Generation-valid content uses actual tokenizer/generation identity; EOS is a separate legal action.

Independently recovered groups: A2_SAME256, A2_DELTA44, EOS_RECOVERY12 (10 dialogues),
EOS_REGRESSION0, CONTENT_DIVERGENCE32, OTHER_INVALID0. AUTO_SAME264/AUTO_DELTA36 compare
sealed AUTO output to canonical B0 by the same token/termination rule. Cross categories:
AUTO_SAME×A2_SAME247; AUTO_SAME×A2_DELTA17; AUTO_DELTA×A2_SAME9; AUTO_DELTA×A2_DELTA27.
All memberships use only token arrays/IDs; references, error/POI/language labels, durations and losses
cannot define a stratum. Parent FIXED100 and NEW200 are supporting exposure partitions, not new roles.

MECH_CHANGED=all44 A2_DELTA. CONTROL40=first2 A2_SAME rows per dialogue in FULL300 canonical
order; each dialogue has7–15 available. Require exactly2 per dialogue; shortage means STOP, no refill.
MECH_PANEL84 is disjoint union in FULL300 order, hash
`760bc3d7e69dc7c167937ec820417ad8d5050b22868e865fe35da5818036c8bf`.
Panel byte hash is pinned in config; all membership hash domains use compact sorted-key UTF8 JSON
`{partition,rows:[{utterance_id,dialogue_id}]}`. Freeze/push memberships before outcomes.

## A2 remains exact historical TTA1

Teacher: sealed theta0 AUTO greedy content IDs and masks from PATH5 plan, detached; no regenerated
AUTO teacher or native LID. Student forced-ZH prompt `[50258,50260,50360,50364]`.
Loss mean valid-position hard NLL `-log p_theta(y_A[t]|x,cB,y_A[:t])`.
Exact trainables:194 decoder LayerNorm weight/bias tensors,248,320 scalars (full enumeration config).
32 decoder layers each self-attn/encoder-attn/final LayerNorm, plus final decoder LayerNorm.
AdamW lr1e-3, wd0, betas(.9,.999), eps1e-8, amsgrad/foreach/fused/maximize False;
exactly2 updates; no clipping, accumulation, scheduler, scaler or autocast; seed240924.
Fresh fp32 masters/optimizer from exact theta0 each utterance, bf16/eager eval functional_call with
masters cast bf16, resident parameters requires_grad=False, no non-LN updates. Reset original bf16
bytes/flags exactly in finally; no optimizer/cache/state transfer between utterances.

Teacher input cB+y_A, no appended EOS, use_cache=False, no past KV; content query index3+t.
Prefix/EOS/special targets do not enter loss; EOS remains in allowed predictive vocabulary.
General suppression every absolute content step, begin suppression only0, float32 log_softmax over
allowed IDs, temperature1. Original sealed masks fixed for both steps. Empty valid set gives exactly2
connected-zero no-op updates, never drop/refill. Preserve raw IDs/UTF8 fragments, no text retokenization.
Frozen encoder once per utterance; detached clone outside inference_mode for adaptation. No encoder grads.
Ordinary decode identical cached.Branch prompt-once/one-token semantics, passive L16 recorder/attention=True,
greedy/batch1/beam1/no sampling/bf16 eager/max200 output decisions including EOS. EOS50257, no action hook/D2.
No A4 teacher/no-op rule, EOS loss, controller or new objective.

## All300 reconstruction barrier and step observer

Phase1: canonical300 theta0 decode and uninterrupted original2-step A2, logging/snapshotting only.
Require theta0 tokens/termination/text equal PATH5 canonical B0; final A2 tokens/termination/text equal
PATH5 A2 exactly300/300. Require final bf16 LN-state hash equal sealed A2_state_hash on every row;
compare all194 final fp32 arrays exactly wherever original archives exist (FIXED100100); compare effective
bf16 bytes too. Historical archive file hashes must match; new archive container bytes need not match.
Non-LN bytes/flags, model/tokenizer/preprocessing/prompt/teacher/reset/environment/cache ownership pinned.
Mismatch → INVALID, abort before STEP1 free decoding or causal ablations; do not alter A2 to match.

Existing `episodic_tta.adapt` exposes L0/L1/L2 and final masters but no intermediate weights. Future
implementation may add only an optional snapshot observer after each original optimizer step. The
observer clones detached fp32 masters to CPU; no RNG, forward, gradient, optimizer mutation, resident
materialization, loss adjustment or extra step. The original optimizer trajectory remains uninterrupted.
Observer-off/on must produce exact losses/gradients/final-master/effective-state equality on synthetic
regression, and exact all300 historical final barrier in the real run. Source changes must be documented
as non-outcome-bearing instrumentation, with old original-source hash and new resolved-source hash.
Do not run STEP1 decode before update2. STEP0 is exact theta0, STEP1 after first original update,
STEP2 after second. STEP0 loss is on teacher y_A, not a reference transcript.

After phase1 all300 PASS, use saved states/MECH_PANEL encoder outputs for phase2. One resident model
is sufficient: materialize chosen LN state, create fresh decoder KV, run diagnostic, destroy path,
restore theta0 in finally. Never switch weights under a retained KV cache. Frozen encoder outputs may
be offloaded losslessly to CPU after phase1 for84 rows and returned with exact dtype/bytes; no re-encoding.

## A — optimization dynamics, all300

Record original losses L0/L1/L2, gradient norms/finiteness, total fp32 and effective-bf16 deltas after
update1/update2, relative delta `L2(delta)/(L2(theta0.float)+1e-12)`. Reduction uses float64 squares of
actual fp32 differences; effective delta compares bf16 values converted to float32. STEP0 delta is0.
Per tensor, LayerNorm module (weight+bias), decoder layer and family, report L2 norms. Aggregate by
sqrt(sum squared member norms), never sum of norms. Final decoder LN is a distinct `decoder_final`
module, not invented layer32. Partition194 tensors exactly:
SELF=self_attn_layer_norm weight/bias64 (81,920 scalars);
CROSS=encoder_attn_layer_norm64 (81,920);
POST=per-layer final_layer_norm plus final decoder layer_norm66 (84,480).

Seal compressed LN-only fp32 STEP1 snapshots for300 and STEP2 forNEW200200, reusing archived final
FIXED100 snapshots; one immutable theta0 snapshot. This supports independent norm/state recomputation.
Each full LN snapshot is under1MiB uncompressed; no full-model/encoder/KV/vocabulary logit dumps.
Commit compact rows and snapshot archives/hashes to the project artifacts before reference access.

On MECH_PANEL only decode STEP1, using STEP2/FULL_A2 from phase1. Report token+termination identity
STEP1 vsSTEP2 for changed,controls,EOS_RECOVERY,CONTENT_DIVERGENCE, denominators explicitly;
empty group=None. At each historical changed-row common-prefix site record whether STEP1's processed
argmax equals final A2 action. This never selects a one-step method.

## B — first-divergence geometry, changed44 only

Replay identical historical forced-ZH common prefix under STEP0/1/2 using clean state-owned
cached.Branch, feeding prompt once then prefix token-by-token. **STEP1 common prefix need not be
its greedy path**: teacher-force that fixed history, record any mismatch descriptively, not INVALID.
This differs from PATH branch-adjudication's greedy-prefix sanity; do not blindly reuse its rejection.
Validate actual fed sequence/cache positions/state ownership, not STEP1 prefix greediness.

At k compute canonical float32 processed log probabilities (same suppression/temperature1/full allowed
vocabulary). Save EOS, B0/A2 actions and valid sealed AUTO action at k, entropy over finite allowed
probabilities, processed top1-top2 gap (nat). Mandatory B0/A2 action logp finite. Optional suppressed
EOS/AUTO score is null with allowed=False, never infinity arithmetic. Nonfinite raw logits are INVALID.

EOS_RECOVERY: `M_stop(s)=logp_s(c_A2)-logp_s(EOS)`; ΔM1/2=M(s1/2)−M(s0).
Report rows, median, strict-positive counts/rate and dialogue distribution. The final stop-margin crossing is
partly implied by defining EOS_RECOVERY from B0 EOS/A2 content; it is a consistency check, not independent
mechanism evidence. Interpret step1, continuation support and causal family ablations together. CONTENT_DIVERGENCE:
`M_branch(s)=logp_s(c2)-logp_s(c0)`, same step shifts. EOS_REGRESSION, if present, analogous reversed
stop/content quantity descriptive only, not a new label feature.

H=3 final A2 content tokens `[k:k+3]`, shortened independently at EOS/cap. Teacher-force that exact
continuation from the same fixed prefix under each state; mean token processed logp at k+j. The same
owned path may reuse current-query logits/cache for this diagnostic continuation, under unchanged
weights. No branch selection or repair. EOS not scored here; immediate-EOS A2 gives H_eff0 and
C_H=None, not INVALID. Donors include only generation-valid content; no horizon sweep.

AUTO alignment uses **sealed AUTO token/EOS action at absolute k**, not a new AUTO forward on another
history. Compare A2/B0 action equality and exact-length H_eff A2 continuation equality with AUTO segment.
Record AUTO-prefix-equals-common flag; positional agreement with different AUTO history is descriptive,
not evidence of a same-history next-token distribution. Report all assessable rows plus matched-prefix
subset rates, for EOS/content/AUTO_SAME/AUTO_DELTA, with numerator/denominator/unavailable counts.
Missing/invalid AUTO action or empty A2 continuation is not-assessable. No alignment statistic chooses a site/state.

## C — causal state-family ablations, MECH_PANEL84 only

Reuse B0/FULL_A2; STEP1 plus DROP_SELF/DROP_CROSS/DROP_POST are diagnostic outputs, not candidates.
For each DROP start from complete saved final effective bf16 A2, replacing exactly selected family
with exact theta0 bf16 bytes. Others remain bitwise final A2. No optimizer/new update, mixing of
step states, rescaling or tuning. Fresh KV per materialized state; reset always. Free decode ordinary
forced-ZH; record tokens/text/termination and EOS decision-stream ED to FULL_A2/B0.

At the historical changed-row common prefix separately query each condition's processed argmax:
B0 action / FULL_A2 action / third. These are same-history causal state tests, not the condition's
new first-divergence site. Controls have no historical divergence and no action-retention denominator.
RET_FULL=1. RET_DROP=count matched FULL_A2 action / changed-row group size, separately all44/EOS12/content32;
empty group=None. ΔRET=1−RET_DROP. MECHANISM_SENSITIVE if overall OR EOS ΔRET>=.25;
multiple families may pass. Content sensitivity is supporting only. No trainable subset selection.

## D — post-seal reference analysis

Before references: seal/push panel/strata/control hashes, all300 reconstruction, step states/losses/norms,
STEP1 outputs, geometry/margins/continuation/AUTO alignment, ablation outputs/state/cache provenance,
manifest/source/model/config hashes. Confirm immutable seal on remote; independent primary audit PASS.
Only then open historically exposed FULL300 references. No row errors/POI labels enter construction.

Canonical metrics for B0/AUTO/A2 on FULL300; diagnostic conditions on identical MECH_PANEL denominators.
Per row ΔZH/ΔPOI/ΔMixed/ΔS/ΔD/ΔI=B0−A2, positive improvement. Aggregate EOS_RECOVERY,
EOS_REGRESSION,CONTENT_DIVERGENCE,A2_SAME,AUTO relation/cross categories/dialogue and supporting
FIXED100/NEW200. Group net changes must sum exactly to FULL300 totals. Post-seal baseline aggregates
must reproduce PATH5: B0 ZH3309/POI1074/mixed4456; A2 ZH3037/POI1012/mixed4126.

P_ZH=max(ΔZH,0), P_D=max(B0_D−A2_D,0).
F_ZH_EOS=sum EOS_RECOVERY P_ZH / sum FULL300 P_ZH;
F_DEL_EOS=sum EOS_RECOVERY P_D / sum FULL300 P_D.
Zero denominator gives undefined/null, never0. Also report net contributions/harms: positive-only fractions
are not a net-effect causal decomposition. Dialogue counts improving/harming ZH/mixed/POI; max utterance/
dialogue positive-rescue shares for ZH/deletion (undefined at zero denominator). LODO across all20
net ZH/mixed/POI/deletion rescue, min/median/max/count/proportion>0, descriptive only.

Paired dialogue bootstrap repository convention:2000 draws,seed240924,numpy default_rng/PCG64,
sorted20 dialogue IDs, sample20 with replacement, same cluster draws for systems, sum counts then
canonical rates,95% percentile intervals. A2−B0 PIER/MER/EN-WER/ZH-CER; B0−A2 net errors/deletion;
F_ZH_EOS/F_DEL_EOS recomputed with repeated-cluster weighting. Omit/report zero-denominator fraction
replicates. Bootstrap does not select method, mechanism label or confirmation readiness.

## Frozen interpretive labels and exact precedence

First applicable, point statistics only, no threshold changes:
1. P2_A2_MECH0_INVALID: any identity/method/stratum/state/reset/cache/reconstruction/reference/evaluation/
   audit/budget failure. STOP, do not relax tolerances to proceed.
2. P2_A2_MECH0_TERMINATION_DOMINANT: EOS_RECOVERY≥3 rows/≥3 dialogues;
   both fractions defined and≥.50; median ΔM2>0; ≥75% EOS rows have strict-positive ΔM2.
3. P2_A2_MECH0_NONTERMINATION_DOMINANT: both fractions defined and<.25;
   nonEOS changed (CONTENT_DIVERGENCE∪EOS_REGRESSION) has net ZH>0 OR net mixed>0,
   and ≥3 rows with ΔZH>0 OR ΔMixed>0 spanning≥3 dialogues.
4. P2_A2_MECH0_MIXED: both fractions defined, and either:
   (a) EOS and nonEOS changed **each** have net ZH≥5 OR net mixed≥5, with≥2 positive-improvement
       rows spanning≥2 dialogues per group; or
   (b) at least one fraction lies [.25,.50), and each group has net ZH>0 OR net mixed>0.
   Five errors is a fixed small material count anchored to prior TTA benefit convention; do not fit it.
5. P2_A2_MECH0_NO_CLEAR_MECHANISM: valid and none above. Undefined fractions cannot pass2–4.

Family sensitivity and group outcome associations cannot override this precedence. Labels are interpretive;
only state-family resets provide causal evidence for those frozen parameter-state interventions. No statement
that EOS itself causes every measured error reduction without a corresponding EOS intervention.

## Confirmation readiness, tests, audit and scope

A2_CONFIRMATION_READY=YES only if complete exact reconstruction/artifact reproducibility and independent
primary/post-audit PASS, with no missed FULL_A2 severe truncation relative to B0. NO for instability,
non-reproducible required artifact, audit/integrity failure, or≥1 missed FULL_A2 severe truncation.
Original severe rule: B0 length≥10, A2 EOS termination, A2 length≤floor(.5*B0 length). Investigate
canonical token/EOS/cap semantics independently; no invented extra safety threshold. Diagnostic STEP1/DROP
severe failures do not alone disqualify unchanged A2. Mechanism concentration alone does not change A2.
Even YES only recommends a separately frozen exact-A2 confirmation; no confirmation/test run here or automatic
additional A2 development experiment. A valid NO-readiness result may coexist with a mechanism label.

Focused tests: exact300 IDs/order/audio/teacher/seal hashes; token/termination strata and CONTROL40;194-family
partition/disjointness; original A2 settings; observer-off/on exact loss/gradient/final regression; update1 snapshot
timing; all300 final barrier; fp32/bf16 norm math and squared aggregation; no reoptimization for DROP; exact
family replacement/complement identity; fresh KV/state ownership/reset; teacher-forced STEP1 prefix mismatch valid;
processed EOS/candidate logp, margins, H3 including empty continuation; AUTO positional/matched-prefix distinction;
reference barrier; positive/net decomposition and undefined denominators; label/sensitivity boundaries/readiness;
no G1/G1A/A3/A4/controller or confirmation. Reuse CPU A2/TTA1 tests; new focused implementation tests are future work.

Independent auditor: PASS_TO_P2_A2_MECH0 before sbatch; P2_A2_MECH0_AUDIT: PASS after run, primary phase before
references. Auditor must not import primary decision code; independently reconstruct panel/strata/controls,
state snapshots/families/norms/logit processing/margins/continuation/ablation identities/retention,
reference barrier/canonical errors/fractions/concentration/bootstrap/label/readiness. Canonical metric primitives
may be shared. First independent live loss tolerance1e-5, gradient relative.02 with floor1e-8 unchanged.

At most one sbatch MIG/H100 job, pending/running≤1. All300 reconstruction600 optimizer updates;
max601 backwards including live audit.600 reconstruction free decodes +84 STEP1 +252 DROP=936 total;
132 STEP0/1/2 geometry paths (current query plus H3 continuation) and132 DROP current-query paths.
No extra optimization or AUTO/LID/D2/controller calls. Target<30min, hard3h; timeout/partial →INVALID,
no expansion/resubmission. Keep all300 barrier before phase2 diagnostics.
Allowed exact already-exposed D-dev-select FULL300 and sealed historical artifacts only. No new role,
D-dev-confirm/D-test/P3/router-calib new role/SEAME/CS-FLEURS/ViMedCSS/ASCEND/transfer.
No A2 objective/teacher/LR/steps/trainables/precision changes, EOS loss, router, G1/G1A/G2 or A3/A4.
No automatic confirmation. Checks→diff→commit→push origin HEAD:cs-asr-steer-inf; clean local=remote;
no force/merge-main/PR/history rewriting.
