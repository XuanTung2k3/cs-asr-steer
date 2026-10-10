# ST-LOC0 — Frozen-Direction, Multi-Site Causal Feasibility

Status: **PRE-OUTCOME DESIGN FROZEN; IMPLEMENTATION / INDEPENDENT PRE-RUN AUDIT PENDING**.
No ST-LOC0 scientific outcomes, pretrained model forwards, or GPU jobs were run during this freeze.
The executable numerical contract is `configs/inference_cf/st_loc0.json`; population and calibration
membership are frozen in `ST_LOC0_PANEL.json`. Freeze base: `8c1298333597d970daac6d79a1f3278bf6080800`.

## 1. Question, authority, and claim boundary

Can either **v_prompt** or **v_unq** correct lexical decisions under one matched-energy pulse at
L16/L24, immediately after self-attention or immediately after cross-attention / before FFN?
This is a bounded exposed-development feasibility stage, before any test-time location optimizer.
The user explicitly authorizes the additional self-attention diagnostic site. This supporting
`inference_cf` experiment does not change the locked core v6 method, its DG-02 site, training,
selection claims, or `METHOD_CONTRACT.md`. Do not relabel this grid as the core paper method.

Historical conclusions remain: P2-DIR D0 had near-zero useful L16 lexical leverage; clean D1 gave
modest safe EN-confusion margin benefit without top-1 corrections; D2 had strong lexical power
with Mandarin damage. P2-SEQ had sequence damage. A2-MECH0 termination recovery is a separate TTA
finding; A2 is not an actuator here. No previous selectivity, TTA, or PATH stage is reopened.

Only V1/V2 are eligible. No v_unp, v_nat, new learned direction, main-candidate readout gradient,
new objective, gate, location optimizer, persistent steering, or full ASR evaluation is permitted.

## 2. Frozen panel and separation of inputs

Exactly the P2-R/P2-DIR 180 positions: 60 EN-confusion, 60 EN-correct, 60 ZH-correct;
80 utterances, 20 dialogues. Preserve original position order, utterance order, audio hashes,
query indices, prefixes, and evaluator definitions. EN-confusion spans 17 dialogues; the two
correct strata each span all 20. This is already-exposed **D-dev-select**, not fresh validation.

Historical identity fingerprints:

- P2-R population: `sha256:6a880d2b20049200dbd7e58ab6d34c55238a772e7be616fac337c7fff8b743bb`.
- P2-RJ positions: `sha256:c1e016c0fd5c6c4544a806194541ce4fb1b0b94a5aa3660d07807a3d870a54d4`.
- Clean D1 construction: `sha256:47a0f6ca66182a330903055024d0ea6afdc1520ae05e9eda9da524c30a30ede2`.

`ST_LOC0_PANEL.json` has a separate content-addressed `identity_hash`, calculated using
`core.digest` over every field except `identity_hash`. It pins audio sources, baseline rows,
construction membership, historical vectors, manifests, and source files. Membership is unchanged.

The matched baseline is specifically `results/inference_cf/p2_A_r1_L16`, system **B0M_L16**;
do not substitute an ordinary historical B0 or a later PATH/TTA baseline. At generated content
step t, use that sealed baseline's tokens `[:t]`, query index `4+t-1`. Prefixes:
`c_M=[50258,50260,50360,50364]`, `c_E=[50258,50259,50360,50364]`. SAME audio, SAME content prefix,
only condition tokens differ. No independently generated English prefix, reference prefix, or
future content token is used. Inherit historical audio preprocessing/encoder semantics unchanged.
EOS=50257, historical cap=200, batch=1, greedy, temperature=1, bf16/eager, frozen pretrained weights.
Pin model/tokenizer/config files, processed suppression and script partition hashes from the
historical manifest. A checkpoint/model revision or different cache path is not an equivalent baseline.

The runtime consumes only the panel's `runtime_queries` and `utterances` projection and sealed
site/dialogue vectors. `construction_positions`, its stratum labels, and calibration membership
are offline-builder inputs only. Reference target tokens/competitors stay in the separate
historical evaluator artifact. Runtime schema and import tests must enforce this separation.

## 3. V1: dynamic same-prefix prompt contrast

For each site/layer/current query, collect unedited h_E and h_M under the paired prefixes above.
V1 is constructed dynamically per query, not one corpus-global vector:

`delta = float64(h_E) - float64(h_M)`

`v_prompt = float32(delta / (||delta||_2 + 1e-6))`.

This is the exact `core_p1.direction` / `OldDirection` D0 provider, including subtraction AFTER
upcast, CPU float64 norm, float32 serialization, orientation E minus M. Nonfinite delta/norm or
norm <1e-4 yields an exact no-edit with reason. Do not substitute normalize-then-subtract,
centering, bf16 subtraction, a learned sign, or another fallback. Minus arm negates the sealed
float32 vector; solver unit normalization is inherited and distinct from provider epsilon.
L16 DG-02 must reproduce the original D0 semantics and historical vectors.

## 4. V2: clean two-group unique-subspace calibration

Use **NEW_P2_DIR_CROSSFIT_V1** in `unique.py`, not the legacy A5 Add-Unique artifact.
A = frozen EN-correct positions, B = frozen ZH-correct positions, both unedited forced-ZH states
at the matched site. Exclude ALL rows in the evaluation dialogue, including every stratum and
utterance. Sorting is `(dialogue_id, utterance_id, t)`. Calibrate separately at each of the four
physical sites using identical A/B membership; never transplant an L16 numerical vector elsewhere.
Labels are allowed ONLY for this offline historical calibration. This is a fixed calibration-derived
representation direction, **not wholly label-free zero-shot**. Runtime reads the held-out dialogue's
sealed float32 vector and carries no correctness feature.

For each fold, CPU float64 uncentered moments, symmetrized as historical code:

`C_A=mean(h h^T over A); C_B=mean(h h^T over B)`.

Top-r eigenvectors `U_A,U_B`, r=32; `U_A^T U_B=P Sigma Q^T`;
`a_i=U_A P[:,i]`; `score_i=(a_i^T C_A a_i)*(1-sigma_i^2)`.
Pick the greatest score, lowest index on exact ties subject to the historical winner-separation
guard. Orient toward `mean_A-mean_B`, normalize in float64, serialize float32.

Inherit ALL guards, not just the formula: counts and matrix ranks >=32; positive relative
32nd eigenvalue and isolated 32nd/33rd boundary; relative tolerance 1e-10; singular values clipped
to [0,1]; material and isolated winning score; isolated chosen principal singular value; principal-pair
orthogonality/identity tolerance 1e-8; nondegenerate sign contrast and sign projection; serialized
unit error <=2e-6. Exact details and hash format remain those of the pinned `unique.py`.
No alternate rank/group/eigenvector/global direction or sign fallback.

Original clean groups/formula unrecoverable: **ST_LOC0_BLOCKED_DIRECTION_PROVENANCE**, stop before
execution. This freeze's CPU-only historical refit reproduced all 20 clean fold vectors exactly
(max absolute difference 0), all 180 D0 vector hashes exactly, and verified all 240 sealed Exp-1
files. This checks historical provenance only, not new-site outcomes.

A5 differs: English-overlapping span positions versus preceding positions (not clean ZH-correct),
reference-prefix extraction, and a production top-64 intermediate construction. Its historical
manifest and retained A6 Add-Unique copy are provenance, not a substitute for D1. Do not equate them.

## 5. Physical sites and residual semantics

Layer indices are **zero-based** HuggingFace `decoder.layers[16]` and `[24]`, dimension 1280.
`WhisperDecoderLayer.forward` in the installed Transformers **4.57.6** implementation computes
(the complete module and forward-source hashes are pinned in config `site_forward`):

`q = h_in + dropout(self_attn(LN_self(h_in)))`

`r = q + dropout(encoder_attn(LN_cross(q), encoder_states))`

`h_out = r + FFN(LN_final(r))`.

Eval mode disables both attention dropouts. Neither site is the block's post-FFN output.

**POST_SELF_ATTENTION:** q, residual after self-attention, before cross-attention.
Capture h_in with a self-attention-LN pre-hook and u_self with the self-attention output hook;
sum in native dtype before converting for storage. A stage-scoped residual-aware hook returns
`u_self + (repair(q)-q)` in output element 0, preserving other tuple elements. The layer's residual
addition then feeds the edited q to BOTH cross-attention LN and its residual bypass. Audit the
actual q consumed at the cross-attention-LN pre-hook. Editing only that LN's argument is WRONG:
the cross-attention residual variable was already assigned and would bypass the intervention.

**POST_CROSS_ATTENTION_PRE_FFN:** r, exact DG-02. Reuse
`DecoderPostCrossAttnRecorder` / `DecoderPostCrossAttnInterventionHook`: record q/u_source/r,
modify the source output to realize repaired residual, audit r at final-layer-LN input.
L16 must exactly reproduce historical DG-02. Both sites feed the same frozen decoder tail and
normal output projection; no changed readout or new gradient objective is needed.

Synthetic CPU tests on the real Whisper layer class establish the self-site route is technically
valid. A production adapter and both cached/full-prefix CPU regression paths must pass before
scientific execution; no runner is delivered by this freeze. Unsupported model forward geometry
or unavailable exact hook: STOP; do not shrink/replace the site grid.

## 6. Matrix and controls

16 primary arms = `{v_prompt,v_unq}` × `{16,24}` × two sites × `{+,-}`.
IDs in config fully enumerate the matrix. ALL 180 positions must have records for every arm.

Controls: NONE; FOUR random controls (one fixed Gaussian unit vector per physical layer/site);
D2 READOUT at L16 DG-02 only, historical positive orientation. Random PCG64 seeds are derived
from SHA256 of `ST_LOC0-random-v1|240924|L{layer}|{site}` (first 16 hex digits as integer), then
float64 draw/normalization and float32 serialization. Seeds and vector hashes are frozen in config;
no resampling on failure, no random sign sweep. D2 reuses hashed historical `t{t}_D2` vectors,
with independently verified original readout construction. D2 cannot be selected or counted in
the V1/V2 feasibility oracle. Total: 21 pulse arms / 3780 cells, plus 180 shared NONE distributions
and zero-dose checks at each site. No BROAD, gate, LID, free branch selection, or A2.

## 7. Single-pulse matched energy and eligibility

Every arm starts from the identical unedited forced-ZH prefix and pre-query cache. Exactly ONE
query pulse, no earlier edit, no continuation edit. Use unchanged P2-R `solve_scale`, e*=
**1.1260757575454359**: CPU float64 analytic chord geometry, native bf16 repair emulation,
analytic initial scale then rescale/secant, at most 8 evaluations. No independent dose tuning.
Repair is `apply_steering`, NormPreserve ON, epsilon 1e-6, no depth rescale. API alpha=scale=1;
the solved vector multiplier supplies the fixed target chord, not a new alpha search.

Both proposed and actually consumed chord norms must be within 2% of e*. Also inherit the
stronger historical squared-energy error <=0.02 and pairwise squared-energy ratio excess <=0.02
among valid arms at the same query (across physical sites too). Consumed/proposed chord norm
difference <=0.005 relative (historical auditor); solver/hook norm agreement <=1e-6*max(1,norm).
Store native before, proposed-after and consumed-after states, direction/norm/sign, solver history,
realized chord/relative magnitude, site reconstruction and cache lineage. Require normal
NormPreserve implementation unchanged; report consumed-state norm deviations from bf16 rounding.

Predetermined mathematical no-edit cells: V1 tiny/nonfinite delta; a new-site D1 guard-invalid
fold; geometrically unreachable energy; solver unable to meet target in eight evaluations.
Record reason, zero pulse, exact baseline logits; no substitute/fallback. These cells remain in
all 60-row stratum denominators with zero delta, not dropped or silently rerun. An arm is eligible
for statistical/feasibility qualification only if >=57/60 valid pulse cells in EACH stratum and
valid cells span >=12 dialogues in EACH stratum. Report all arms even when ineligible.
Original clean D1 provenance failure is the earlier STOP, not this new-site guard case.
V1-only nonfinite delta/norm retains its historical no-edit fallback; a nonfinite baseline B
state or emitted distribution is an engineering failure and takes INVALID precedence.
Nonfinite baseline B state or emitted logits, incorrect hook, unexpected consumed energy/cache corruption, absent
planned record, or altered fallback is ST_LOC0_INVALID. Missing cells cannot be excused by eligibility.

## 8. Historical barrier BEFORE new-site pulses

Replay historical L16 DG-02 NONE / plus-D0 / plus-D1 / D2 on ALL 180 positions first; retain
these outputs as their matrix cells. Before any new-site pulse:

1. Matched baseline tokens, packed bf16 logits and captured H_B/H_E match sealed historical states
   bitwise; passive multi-site observation must not change them.
2. D0 matches historical normalization/fallback (independent vector maxabs <=1e-6); D1 fold vectors
   maxabs <=2e-6 and construction status/index/guards match. Original CPU refit was exact.
3. D0/D1 historical pulse consumed states and logits reproduce bitwise at the pinned execution
   path, with historical solver/energy checks; D0 P2-R scale agreement <=1e-9.
4. D2 original vector provenance: independent gradient vector maxabs <=2e-6, cosine >=1-1e-6;
   J agreement <=1e-4, independent scratch spot check <=1e-5. Reusing the sealed control vectors
   avoids redundant 180-query autograd construction; it does not create a new D2.
5. D2 pulse output reproduces historical packed logits/state and top-1 decisions; post-seal
   evaluator reproduces old outcomes/scalars within 1e-8. D0/D1 outcomes also match old report.
6. Zero-dose at every site and fresh restored NONE replay are bitwise equal to baseline.
   Cache hashes/length/positions before and after each arm prove the next arm is unedited.

A genuine mismatch is **ST_LOC0_INVALID**, STOP before new-site outcome interpretation (and
before remaining pulses if detected in the barrier). Never relax tolerances after seeing results.

## 9. Seal, outcome definitions, and uncertainty

Before pulses, seal/push site-specific unedited states, all 80 site/dialogue fold records, V1
per-query vectors, random controls and D2 references with manifest/source/config hashes. All new
fits use original membership and pinned math only. Fitting can follow passive extraction inside
one allocation, but its immutable calibration seal must be on remote BEFORE pulse outcomes.
Historical reuse requires exact state/hash agreement, never a semantically similar substitute.

After ALL pulse outputs and audit scalars are sealed and pushed, open the historical evaluator
positions. Reuse `inference_cf_p2dir_analyze.logit_metrics` and the same processed-logit rules:
`m=logsumexp(z[Y_ref])-z[c_fixed]`. Target set and competitor are fixed by historical evaluator,
not repicked after editing. A top-1 correction is baseline outside Y_ref and edited top1 inside;
corruption is baseline inside and edited outside. Top-1 ties use lowest token ID.

For EACH arm, all three strata: margin change (nats), top-1 corrections/changes/corruptions,
reference rank movement, English/Mandarin token mass changes and invalid-cell counts. Report
position-average margins and the authoritative dialogue-macro estimate. The **dialogue-macro
estimate** controls the +0.50 gate, paired with the matching interval (no mixed estimands).

Paired dialogue-cluster bootstrap, **B=10000, seed=240924**, matches authoritative P2-DIR
rather than the suggested 2000. Sorted dialogue IDs, shared RNG draws across arms, mean positions
within dialogue then mean represented dialogues; historical missing-stratum-draw handling retained.
Fixed calibration folds are not refitted in this descriptive conditional bootstrap.
Primary multiplicity family: all 16 planned V1/V2 EN-confusion margin comparisons, INCLUDING
ineligible arms in family size. Bonferroni two-sided alpha=0.05/16; percentile quantiles
0.0015625 and 0.9984375. Safety and control intervals are pointwise 95%, explicitly descriptive.
Report counts and dialogue coverage, not a huge family of additional significance tests.

## 10. Evaluator-only oracle envelope

After the full seal, for each confusion identify tested combinations yielding reference-correct
top1. Primary/restricted candidate-only oracle uses EXACTLY the 16 V1/V2 arms; D2/random excluded.
Report correctable count/fraction, dialogue coverage, per-family/site/sign positive/negative
responses and correct-stratum corruptions without selection. Any extended control envelope must
be separately labeled and cannot affect this primary diagnostic. This is reference-dependent,
nondeployable, not a selector/ASR system/training target, and cannot enter a live provider.

## 11. Exhaustive terminal decision and next step

First test INVALID (historical/site/causal integrity, leakage, audit or incomplete matrix).
Otherwise, consider only eligible primary arms:

- **ST_LOC0_NO_FIXED_DIRECTION_LEVER:** none has multiplicity-adjusted EN-confusion margin
  lower bound >0.
- **ST_LOC0_LOCAL_EFFECT_ONLY:** at least one has adjusted lower >0, but none meets ALL of
  +0.50 dialogue-macro nat, adjusted lower >0, >=3 top-1 reference corrections across >=3
  dialogues, and matched-energy/integrity PASS. This also covers positive effects below the
  materiality cutoff, making the labels exhaustive. No automatic LOC1/full decode.
- **ST_LOC0_SITE_FEASIBLE:** at least one meets ALL those gates and independent post-audit PASS.
  Correct-state damage is fully reported but does not automatically disqualify causal feasibility.
  This label makes NO safety/deployability/sequence-benefit claim.

Precedence: INVALID → NO_FIXED_DIRECTION_LEVER → LOCAL_EFFECT_ONLY → SITE_FEASIBLE, applying
these disjoint predicates. Among feasible arms rank: more corrected confusions, then more correction
dialogues, then higher dialogue-macro confusion delta, then fewer total EN/ZH correct corruptions,
then ascending configuration ID. Selection is exposed-development only.

SITE_FEASIBLE recommends a **separately frozen ST-LOC1** reference-free location-optimization
experiment. NO_FIXED_DIRECTION_LEVER closes this V1/V2 site-search family. No energy/rank/layer/
teacher/direction/threshold changes or new method follow automatically from any result.

## 12. Compute, audit, and firewall

Future execution: preferably ONE sbatch H100/MIG allocation, at most one pending/running job,
absolute wall-time **3h**. Existing timings (85.686 s extraction; 113.328 s for 540 pulses plus
D2 gradients; CPU 20-fold provenance refit ~13.4 s) suggest **15–30 min** for 3780 pulse cells,
new site fitting and sealing overhead, subject to pre-run conservative estimate. No new LID,
no main-candidate backward, no redundant old extraction/direction fitting. If the complete
16-arm-plus-control matrix is predicted not to fit 3h, STOP for explicit compute revision BEFORE
submission. Never reduce arms. No throughput/model outcome was generated in this Codex session.

Before GPU require **PASS_TO_ST_LOC0** from an independent auditor not importing primary decision
logic. Verify panel/folds/allowlists, both site routes, exact original reproduction contract, 16
arms/controls/random hashes, dose, eligibility, estimand/multiplicity, reference barrier and compute.
Post-run require **ST_LOC0_AUDIT: PASS**, independently recomputing fit guards, energies, cache
integrity, full-matrix metrics/intervals/oracle/winner and hashes. No continuation on failed audit.
Every phase writes a manifest with resolved config, environment, git state, model metadata,
source/data/vector hashes and status using existing provenance/specfreeze machinery.

Allowed: historical exposed D-dev-select panel and P2-R/P2-RJ/P2-DIR artifacts. Forbidden:
D-dev-confirm, D-test, router-calib as a new role, P3, transfer datasets. Calibration labels are
only offline group membership. No evaluator/reference/oracle information in provider/runner;
no outcome-dependent live selection before seal. Historical reports stay byte-unchanged.
