# P2-DIR — Steering-Direction Identification, frozen v1

Status: pre-outcome design freeze, 2026-10-06. Implementation and scientific execution pending.
Authority for this separate inference_cf development stage: this specification and its JSON config.
The user's three-direction instruction supersedes the older handoff's one-construction limit.
DG-03R/v6 core-paper basis/controller/training contracts and DG-08 evidence remain unchanged.
This is an exploratory development stage, not independent confirmation or a revised core method.
No GPU experiment or new actuator was implemented during the design session.

## 1. Starting evidence and scope

Actual root `/home/tungnx/cs-asr-steer-inf`; branch `feature/inference-cf-steering`;
local and fetched origin/cs-asr-steer-inf HEAD both
`ae6d10b9f7194a7e090b52f95dc49ab63a407a18`; clean tree, empty Slurm queue.
No local-only work, restoration, merge, reset or PR. Terminal diagnosis accepted verbatim:
`P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`. Do not reopen P2-R/P2-RJ or expand to
227 positions. Only direction changes: D0 OLD, D1 UNIQUE, D2 READOUT. No fourth fix,
shared vector, hybrid, different layer, alpha sweep, gate sweep or localizer redesign.
Inherited random and minus-old results can be cited descriptively, never newly selected.
Question: at the same state/site/realized energy, which direction moves the lexical decision
and preserves correct English/Mandarin states? Teacher-forced results are mechanism screens.

## 2. Historical unique/shared recovery and limitations

| Field | Recovered provenance |
|---|---|
| Existing implementation | `src/csasr/experiments/basis_a5_unique_shared.py:construct_unique_shared`; extraction `experiments/basis_a5_unique_shared.py:construct_whisper` |
| Numerical-core last commit | `e00385d306717d21f19e294ce2c922ff44f80538` |
| Artifact provenance commit | `2b307645b05f91b080302d8f33f755212446dca1` (manifest refresh, not necessarily extraction execution) |
| A | D-construct English tokens overlapping `all_correct` eligible spans |
| B actual code | `_preceding_indices(emb)`: up to len(emb) positions before the first embedded token; no explicit Mandarin/baseline-correct filter |
| Site/timing | L16 DG-02 pre-FFN, forced-ZH reference teacher forcing, index prefix_length+i-1 predicting content token i; text truncated at 220 |
| Centering | Uncentered, pooled positions, CPU float64 moments |
| Rank actual code | rank requested 32; production top_vectors=64; SVD on all 64 basis columns; argmax limited to first 32 principal modes |
| Sign | Positive dot with mean_A-minus-mean_B; raw sign retained if anchor indeterminate |
| Data role | D-construct construction; historical oracle-local D-dev-select/SEAME evaluation |
| Artifact | `results/basis_a5_unique_shared/directions/whisper/manifest.json`, layers.decoder.16; A=453 B=378; v_unique hash `f82f7790b6df3ee2a9b8cacda1a98c91a35cdb7e894008e58dde15314f9f7bff` |
| Availability | Original A5 L16 .npy/moment .npz paths are absent, but A6 fixed CS copy `results/basis_a6_expanded/fixed/cs_dialogue/directions/whisper/decoder/L16/add_unique.npy` is recovered: float64 unit vector, file SHA256 `93d681722d8831460ef241e27a77377036d7dc179be9e13893ffe1be376e9830`; float32 .npy reserialization exactly matches A5 hash above. A6 fixed.py explicitly reuses A5 CS; ASCEND/per-sample A6 constructions differ and are not substituted. |
| Prior outcomes | A5 atlas L16 Whisper CS oracle-local rho=.5: Add-Unique MER .242856, PIER .451940, corrections/corruptions 60/19; Minus-Shared .266203/.569665, 21/247; composite .249985/.507937, 39/125. Energies differ (5144/6473/5791), so these are descriptive, not energy-matched proof. |
| Leakage/interpretation | Evaluation panel already exposed; oracle-local masks; A/B role disjointness alone does not prove a language-specific direction. Historical rank stability failed and was made diagnostic by an explicit historical amendment. |
| Reusable unchanged? | Historical vector can be recovered unchanged from A6 copy, but is NOT selected for P2-DIR because its population/rank semantics do not meet the frozen top32/correct-state definition. Historical artifacts remain unchanged; reuse numerical formula patterns/tests only. |

IMPLEMENTATION GAP (historical/off P2-DIR path): A5 prose top-r and Mandarin-correct B do not
match actual top-64/first-32 and preceding-position extraction. No historical repair or relabeling
is authorized. P2-DIR uses a NEW frozen definition below and does not depend on resolving A5.
A-unique is a representation-derived hypothesis, not automatically language-specific. Prior
outcomes motivated this family but cannot count as unseen validation.

## 3. Shared states and D0

Use byte-identical population from `results/inference_cf/p2rj/positions.json` (180 positions,
60 per stratum; parent P2-R population hash in config), original B0M_L16 tokens and prompt.
B prompt [50258,50260,50360,50364], E prompt [50258,50259,50360,50364]. Both consume
the same generated content prefix; cache query is the query predicting t. Model/eager/bf16,
suppression and 200-token cap match P2. Eligibility t>=1, absolute query>=prefix length;
all forced-prefix positions excluded. Exact DG-02 L16 index16, eval/dropout off; no depth scale.

D0 calls `core_p1.direction(h_e,h_b)` unchanged. Detached states converted to float64 BEFORE
subtraction; norm float64; nonfinite or norm<1e-4 -> no edit; delta/(norm+1e-6) returned float32.
B/E independent caches contain only common-prefix tokens; S has its own edited history in
free decoding. Gate uses frozen R2 B-branch signals, not processed S logits. Do not rename
D0's epsilon-normalized vector to an exactly unit vector in deployment. Exp-1 solver's internal
unit normalization is inherited unchanged from P2-R. Record both supplied and solver directions.

## 4. D1 — NEW_P2_DIR_CROSSFIT_V1

Freeze ONE leakage rule: leave-one-dialogue-out throughout Exp-1/2/3. For evaluation dialogue k,
A=all 60 frozen EN-correct positions excluding k; B=all 60 frozen ZH-correct positions excluding k.
Use unedited B site vectors at precisely those baseline-prefix queries, extracted once, before
any arm outcomes. No confusion states or reference-margin gradients enter fitting. Labels are
construction/evaluator metadata, never inference features. The static provider gets only a sealed
fold vector; dialogue-to-fold assignment is fixed by panel IDs, not predictions. These are
supervised development-derived frozen representations, so describe D1 as development-fitted,
training-free model weights, not wholly data-free or zero-shot inference. General deployment to
unseen dialogues is OUT OF SCOPE; no global refit in this stage.

Every fold requires >=32 observations in EACH group and numerical rank>=32. No reduced-rank
fallback. Pool positions uniformly (no new dialogue balancing), sort by (dialogue_id,utterance_id,t),
accumulate sum(h) and sum(hh^T) in CPU float64, h from bf16 site converted losslessly.
C_A=sum(hh^T)/n_A and C_B likewise; no centering/LayerNorm/residualization. Symmetrize;
`numpy.linalg.eigh`, descending eigenvalue order, retain EXACTLY 32 columns per group.
Require smallest retained eigenvalue > 1e-10 times largest. Require boundary eigenvalue gap
(lambda32-lambda33) > 1e-10 times largest (zero lambda33 if unavailable).
Pin NumPy/LAPACK environment in manifest. `numpy.linalg.svd(U_A.T@U_B,full_matrices=False)`;
sigma clipped [0,1], descending order; a=U_A P, b=U_B Q; E_A(i)=a_i^T C_A a_i;
s(i)=E_A(i)(1-sigma_i^2). Argmax over all 32; exact score ties lowest index. Require winner score
>1e-10*max(E_A), and gap to second score >1e-10*max(E_A). Require selected sigma isolated
from every other sigma by >1e-10. These guards avoid arbitrary orientation of degenerate modes.
Require orthonormality max absolute error<=1e-8 for U/A/B and principal-pair identity<=1e-8.
Sign: dot(a_selected,mu_A-mu_B)>0; flip if negative; if absolute dot<=1e-10*norm(mean contrast)
or mean contrast norm<=1e-10, fold invalid. Normalize exactly in float64; serialize float32 unit
vector (.npy, allow_pickle=False); require norm error<=2e-6. No shared direction computed as an arm.

Invalid fold is a direction failure for its dialogue, zero edit, counted; never substitute D0,
readout, A5/A6 or another rank. Serialize every fold's included/excluded IDs, counts, moment hashes,
eigenspectra, bases, principal vectors/sigma/energies/scores, selected index, sign dot/flip,
vector hashes, model/site/config/source/environment hashes. Auditors can reconstruct from moments.
Do not select cross-fit versus global from results.

## 5. D2 — reference-free readout

At unedited B site h_t define J=log(P_E+1e-12)-log(P_M+1e-12). Partition exactly
`core_r2.tokenizer_partition`, version whisper_han_ascii_v1: embedded ASCII Latin regex,
matrix Han/Han-lead UTF8 fragments, special/mixed/ambiguous excluded from both. Hash complete
ID arrays and verify against P2/R2 manifest. P is full-vocabulary softmax after generate suppression
(suppress_tokens and begin_suppress_tokens only at t=0), temperature 1, no sampling temperature.
Use float32 logits and stable logsumexp/logaddexp(logmass,log(1e-12)), preserving exact objective.
Raw logits/masses co-reported, not construction. No reference set, competitor, transcript, alignment,
future token or evaluator Jacobian in provider signature, imports or serialization.

Autograd boundary: frozen model eval; all parameters requires_grad=False, grads None before/after.
A float32 zero leaf delta cast to bf16 is added ONLY at current B query's u_source, value-identical
forward. Gradient of J wrt delta obtained with autograd.grad once, no backward()/optimizer;
then remove probe hooks, detach everything, discard graph and scratch cache. Forward bf16,
gradient float32, tangent/unit math CPU float64. hhat=h/||h||;
gperp=g-hhat*dot(hhat,g); d=gperp/(||gperp||+1e-12), return float32.
Nonfinite objective/state/gradient, state norm<=1e-8 or tangent norm<1e-8 -> no edit with reason.
Do not flip direction based on results. Unit error<=2e-6, radial dot<=1e-6 in float64;
if epsilon normalization fails either bound, no edit with normalization_fail reason.

Important cache boundary: snapshot the B cache BEFORE feeding the current token(s), deep-copy
scratch tensor caches into ordinary detached non-inference tensors under inference_mode(False),
likewise encoder outputs/input tensors. Grad-enabled forward must use exactly the same token(s),
cache_position, encoder output and prefix as B. Its logits/site must match clean no-grad B bitwise.
The existing Branch.step has an inference_mode decorator: wrapping it in enable_grad is insufficient.
Build a dedicated scratch forward under inference_mode(False), enable_grad; never advance B/E/S
or retain a graph in their caches. Do not import diagnostic SiteGradientProbe in deployable code;
implement a narrowly reference-free equivalent with dedicated tests. Restore any changed flags
on exit; exceptions remove hooks and discard scratch caches. One backward-equivalent autograd call
per eligible query, zero when alpha=0/g=0/prefix-invalid (gate known before direction computation).
For Exp-1 compute at every eligible frozen position regardless of gate.

D2 is B-state-derived even when S has prior edits, as D0 was. Apply resulting direction at current
S exact site in deployment; log B/S difference. This is fixed, not a choice to differentiate S.
Evaluator useful tangent gradient is forbidden as direction, target, distillation, selection or
orientation signal. Evaluator may read it only AFTER all three independent direction artifacts
are sealed. Shared readout/evaluator autograd plumbing is not permission to pass reference IDs.

## 6. Exp-1: direction-only causal screen

All D0/D1/D2 plus no-edit, single pulse only at t from identical unedited B cache. Use
P2-R DiagBranch/pulse_hook/solve_scale/scaled_direction/apply_steering exactly; clean cache cropped
or cloned between arms; arm order D0,D1,D2 fixed, no earlier edits or continuation arm selection.
Each realized post-repair chord norm target e*=1.1260757575454359
(sqrt(1203.3762345332195/949)). Solver max8; gain=alpha=1, s*v passed to hook.
Target and pairwise squared-energy relative mismatch <=.02. Unreachable/failure recorded zero-edit.
Never tune dose from logits. Primary common three-way intersection; preserve all failures with
intention-to-screen zero change tables alongside valid-only tables. Require >=95% valid per arm,
>=40 matched positions AND >=12 dialogues per stratum. More than 5% historical state identity
mismatches invalidates run. No refill/replacement/resampling. Reuse P2-RJ V3 tie-aware identity,
logp/margin tolerance1e-3, norm relative1e-6 and D0 geometry1e-6; D0 solver regression1e-9.

Primary endpoint: Delta m_ref with FIXED baseline competitor c* and frozen reference set from
P2-RJ positions; m=logsumexp(z_processed[Y_ref])-z_processed[c*]. Do not redefine competitor
after edit. Report delta logp(ref), P_E/P_M, best reference rank (full vocabulary, deterministic
first-index ties), top1 correction, raw quantities, cosine with evaluator tangent gradient,
kappa=dot(gref_tan,realized_edit)/(||gref_tan||*e*), cosine with readout gradient, chord norm,
relative edit. Correct strata: delta fixed correct-token margin, top1 corruption, P_E/P_M.
Also co-report post-edit strongest non-reference margin descriptively (never selection).

Statistic: mean positions per dialogue, then equal mean dialogues within each stratum; paired
contrasts use the same dialogue draws and matched rows. 10000 dialogue bootstrap draws with
replacement, numpy default_rng(240924), sorted dialogue IDs; percentile intervals. If a stratum
has no rows in a draw, omit that draw and record count (require >=9900 valid draws). No normal
approximation for 60 positions. Refit D1 in each replicate? NO: inference conditional on sealed
cross-fit directions; overlapping training folds make unconditional direction-estimation uncertainty
unresolved. Disclose conditional development CIs and dependence; no confirmatory generalization claim.

Frozen family10: for each D1/D2: confusion delta margin, paired delta vsD0, EN margin, ZH margin,
and one pooled correct corruption rate (dialogue average over correct positions of BOTH strata).
Each interval at alpha=.05/10, percentiles .0025/.9975. Additionally report pointwise95% per-stratum
corruption CIs; for pass require simultaneous pooled upper<=.05 AND each stratum observed rate<=.05.
For each candidate qualify iff confusion point>=.5 nat AND lower>0; paired-vsD0 lower>0;
EN and ZH margin lower>=-.25 nat; corruption constraints above; all engineering validity passes.
.5 nat is approximately10% of exposed median available first-order leverage4.9-5.07 nats and
5-10 times the old inert range. It is a screen for material local power, not expected error closure.
Safety budgets are deliberately tighter than benefit and do not relax for this small sample;
wide safety CIs can stop the stage. All thresholds strict as specified, equality follows >=/<=.

Selection: neither qualifies -> P2_DIR_NO_NEW_DIRECTION_SUPPORTED, STOP.
One qualifies -> select it. Both qualify -> paired D2-minusD1 confusion contrast, pointwise95%
CI from same bootstrap; choose D2 only if lower>.25 nat; otherwise D1 (static/cheaper).
This tie contrast does not qualify candidates and cannot rescue failures. Terminal selection
labels P2_DIR_UNIQUE_SELECTED / P2_DIR_READOUT_SELECTED. Do not select empirical lowest ASR error.

## 7. Exp-2: gate-coupled screen

Only selected new direction continues. Same 180 rows/strata, baseline-prefix controlled persistent
S replay for NONE, OLD and NEW, frozen g=E*R_B, alpha2, phi(g)=g, current localizer and repair.
Reuse P2-R replay CURRENT mechanics; no oracle relocation. Each arm S has its own edited KV history
but feeds the identical baseline tokens; B/E unedited, same prefixes. At designated positions report
cumulative logits movement vs clean B; this tests actual schedule, not gate-scaled isolated pulses.
Energy naturally differs, report it; do not energy-match this experiment or alter gate for a vector.

Family5: NEW confusion margin, NEW-minusOLD confusion margin, NEW EN margin, NEW ZH margin,
NEW pooled corruption; same bootstrap at alpha=.05/5. Pass iff confusion point>=.10 nat AND
>=.20*its Exp-1 point AND lower>0; NEW-minusOLD lower>0; margins lower>=-.25 nat; pooled
corruption upper<=.05 and observed each-stratum<=.05; direction valid>=95%; engineering passes.
If fail benefit with safety passed: P2_DIR_GATE_COUPLING_SUSPECTED, stop before free decoding.
If fail safety: P2_DIR_CAUSAL_POWER_WITH_DAMAGE, stop. Invalid engineering -> P2_DIR_INVALID,
no scientific verdict. Gate coverage can be low but cannot change thresholds or denominator.

## 8. Exp-3: conditional free-decoding development screen

Only after audited Exp-2 pass. Same R2/P2 300 utterances /20 dialogues D-dev-select. Greedy,
forced-ZH, eager/bf16, max_new_tokens200, independent utterance decode, no previous-utterance context.
B0 matched cached no-edit forced-ZH; OLD original P2 cached ER/alpha2/id/+D0; NEW same with selected
provider; AUTO ordinary generate(language=None) with original P2 AUTO settings, no edit.
D1 uses same sealed leave-dialogue-out vector for each of all300 utterances; no refit/global vector.
D2 uses current unedited B state on shared currently generated prefix and scratch pre-step cache.
No ref transcripts/labels in any decoder. Existing correct-state/POI annotations evaluator-only.

BROAD: g=1 selected direction, eligible UTF8-complete content queries, forced-prefix exclusion;
no LocalSupport/conflict abstention (would defeat broad control). Match NEW total realized squared
energy PER UTTERANCE, not nominal alpha. Run NEW first and seal reference-free energy budget
Q_u=sum ||edit_NEW||^2. Let N_u=count eligible UTF8-complete queries in B0's cached no-edit decode
(t>=1); packet target=sqrt(Q_u/N_u). BROAD allocates equal squared-energy packets sequentially
at its own eligible queries, using P2-R solver on its own S pre-edit state, until remaining Q_u
is exhausted; last packet capped by remaining budget; decrement by ACTUAL realized squared
energy, clamp remaining budget at zero after a rounding overshoot within tolerance; once
exhausted all later edits zero.
If Q_u=0 or N_u=0, no edit. Direction failure/unreachable consumes no budget. Stop at EOS/cap;
never delay EOS, add positions or redistribute from correctness. No outcome-based alpha search.
Match aggregate Q within2% for positive-budget utterances and report each unmatched utterance;
BROAD comparison is invalid if aggregate mismatch>2% or >5% positive-budget utterances mismatch>2%.
Do not change packet rule on failure. Retain NEW/OLD result, label broad energy comparison invalid.
This broad control uses a precomputed same-audio NEW budget, extra passes and completed-decode
metadata; it is a diagnostic control, not a claimed online deployable algorithm. Direction itself
uses no future tokens. BROAD need not have alpha2: dose determined by frozen geometric energy rule.

Metrics: canonical PIER/MER/EN-WER/ZH-CER, corrections/corruptions, correct-EN and correct-ZH
retention, outside-target edits AND aligned harm; retain transition-identity checks. Gate coverage=
positive g / all eligible content queries; direction-valid rate=valid/requested (co-report fraction
of all eligible queries); edit coverage=nonzero realized edits/eligible; report RMS/total chord
energy, edit/prestate norm, per-stratum distributions. Runtime includes all B/E/S, LID, readout,
BROAD budget passes; forwards, autograd counts, peak allocated/reserved VRAM, wall time, throughput.
Canonical retention/outside-harm denominators are fixed baseline-correct aligned reference units;
reuse canonical evaluation definitions, do not interpret string edit counts as harm.

ASR bootstrap: dialogue block resampling, sum numerator/denominator counts then ratio (not mean
per-utterance WER), paired common draws10000/seed240924. Gains baseline-minus-method.
Family10: NEW PIER gain vsB0 and vsOLD; NEW damage vsB0 in MER/ZH-CER/EN-WER;
NEW retention loss EN/ZH and outside-target harm rate; BROAD-minusNEW PIER gain and outside-target
harm difference. Bonferroni alpha=.05/10. Safety: damage upper<=.005 MER, <=.005 ZH-CER,
<=.01 EN-WER, retention loss upper<=.01 each language, outside-target harm upper<=.005.
NEW material success iff PIER gain point>=.005 and lower>0 vs BOTH B0 and OLD, safety passes.
This is roughly >=12 net errors of2268 vs exposed OLD2, a development-screen threshold only.
AUTO all metrics/pointwise CIs mandatory; report NEW-minusAUTO with pointwise95% uncertainty,
never claim superiority to ordinary Whisper unless supported. No selection based on AUTO.

## 9. Frozen interpretation, in precedence order

Engineering invalid -> P2_DIR_INVALID; audit not PASS -> no scientific label finalization.
Exp1 neither qualifies -> P2_DIR_NO_NEW_DIRECTION_SUPPORTED (construction unresolved).
Local benefit candidate fails safety -> report P2_DIR_CAUSAL_POWER_WITH_DAMAGE alongside
no selection; no gate repair. Exp2 benefit failure -> P2_DIR_GATE_COUPLING_SUSPECTED.
After Exp1/2 passes, Exp3 safety failure -> P2_DIR_CAUSAL_POWER_WITH_DAMAGE.
Exp3 NEW material success -> P2_DIR_OLD_DIRECTION_PRIMARY_BOTTLENECK_SUPPORTED.
Else valid BROAD beats NEW: BROAD-minusNEW gain point>=.005 and lower>0, BROAD meets all same
absolute safety constraints (report pointwise95% intervals for additional BROAD safety metrics),
outside-harm difference simultaneous upper<=.005 -> P2_DIR_SELECTIVE_GATE_LIMIT_SUSPECTED.
Else -> P2_DIR_SITE_OR_SEQUENCE_LEVERAGE_SUSPECTED. This is the frozen trigger to prioritize a
later site/representation investigation: locally useful directions at frozen energy and current
gate still fail material free-decoding benefit. It does NOT establish site failure, because
sequence drift/dose also remain possible. Report multiple supportive observations but one primary
label by precedence. No next-component redesign, no P3, no fresh confirmation authorized here.

## 10. Test, manifest and independent audit contracts

Before GPU: implement focused tests and run existing inference_cf/site suites; engineering real-model
preflight may occur only in a later explicitly authorized implementation/execution session via Slurm.
D0 unchanged function outputs and cached OLD logits/tokens/edits versus historical P2 (exact, including
zero/tiny/nonfinite); D1 synthetic known subspaces, energy scores, top32 boundary, tied/degenerate
rank/SVD guards, sign flips, orthonormality, deterministic serialization, excluded-dialogue IDs
including all utterances of k, fold reconstruction and hashes; no global refit option.
D2 finite differences J at tiny perturbations, no reference API/import, suppression and epsilon,
tangent/unit norm, tiny/nonfinite no-op, grads None/flags restore, detached cache values/identities
unchanged, grad/no-grad bitwise equivalence, current-prefix-only and no future tokens, exception cleanup.
Common: DG-02 state/site reconstruction, dropout guard, no rescale, zero-dose bitwise identity,
NormPreserve tolerance inherited DG-02; solver target/pairwise2% energy, unreachable reporting,
all180 population keys/hash identity, firewall, missing/corrupt manifest rejection. Statistical
fixtures hand-computed, paired bootstrap deterministic, all threshold/equality/selection/stop cases,
zero-denominator handling, BROAD exhaustion/early-EOS/mismatch, canonical PIER identity.

Every extraction and Exp1/2/3 run writes an immutable resolved config/spec freeze via
csasr.utils.provenance / csasr.lss.specfreeze plus existing inference_cf hashing: clean committed
Git HEAD/tree, env/python/torch/transformers/numpy/LAPACK/CUDA/GPU/Slurm, local model revision and
weight/tokenizer/generation config hashes, population/role/panel IDs and hashes, source hashes,
fold hashes, partition hash, precision, decoder settings, input/output file hashes, RUNNING then
COMPLETE/INVALID/FAILED status and reasons. No overwrites; preserve invalid attempts. Lock artifacts
before evaluator references are loaded. Config source hashes are design-base compatibility anchors;
future authorized code changes have their own run source hashes, never silently replace anchors.

Independent auditor is a separate CPU executable, does not import primary analysis/decide/bootstrap
functions. May reuse immutable schemas/canonical tokenization/metrics, but independently reconstruct
population IDs, fold exclusions/moments/eigen/SVD/sign/vector hashes, D0/readout objective provenance,
realized apply_steering energy from saved before/after states, every primary statistic/damage,
bootstrap/thresholds/selection and Exp2/3 verdict. Recompute D2 objective gradient via independent
scratch-forward spotchecks in the same authorized GPU allocation (first10 positions per stratum
sorted by sha256("P2DIR-audit-v1|"+utterance_id+"|"+str(t)); CPU checks all saved objective/probe/cache
records); not evaluator-derived direction. Independent reconstructed scalar statistics/CIs must
agree within1e-8; vector unit/cosine and energy tolerances are the frozen construction/hook bounds.
Selections and terminal labels must agree exactly.
Auditor must reject reference/evaluator artifacts in D0/D2 construction or any runtime provider
inputs. D1 offline fitting may use ONLY approved position IDs, dialogue IDs and correct-stratum
labels from the frozen population plus unedited states; reference token sets, competitors, margins,
Jacobians and outcomes are forbidden fitting inputs. Seal a minimal construction-only projection
of population metadata and audit its field allowlist. Historical label provenance is disclosed,
not mistaken for reference-free fitting. Save full
per-position vectors/states/processed evaluator logits or lossless required logit slices PLUS
full rank information; no top20 truncation prevents reconstruction. Site vectors can stay local
large artifacts but manifest hashes and scalar summaries are committed.
Pre-run audit label PASS_TO_P2_DIR_RUN; after EACH experiment `P2_DIR_AUDIT: PASS` with stage suffix
and independently recomputed decision. Next experiment requires preceding PASS. No scientific
conclusion without this exact PASS. On disagreement stop, preserve outputs, fix only mechanical
bugs with tests/new commit/new manifest; no threshold/arm/population changes after outcomes.

## 11. Firewall, compute and execution stop rules

Only already-exposed D-dev-select, P2-R/P2-RJ states/artifacts and read-only historical direction
metadata/results used here. No new historical D-construct extraction is authorized or needed.
Router-calib, D-dev-confirm, D-test/P3, SEAME/CS-FLEURS/ViMedCSS/ASCEND/Qwen transfer forbidden.
Firewall checks allowed panel IDs from frozen artifact, not read forbidden role content.
One H100 MIG3g.40gb, batch1, sbatch, at most2 pending/running GPU jobs, aim<=4 GPU-hours total.
Exp1 extraction/three pulses cheap in1h job; Exp2 conditional1h; Exp3 conditional2h. If budget
insufficient stop as COMPUTE_BLOCKED, do not truncate population or alter precision. Independent
readout spotchecks fit same jobs. No GPU run in design session. CPU numerical/test work allowed.
Commit spec/config/tests/run matrix/audit contract BEFORE scientific outcomes; every reviewed edit:
check -> inspect diff -> commit -> push origin HEAD:cs-asr-steer-inf; never force-push or merge main.
