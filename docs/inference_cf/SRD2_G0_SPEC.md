# SRD2-G0 — Acoustic-conflict-gated readout steering feasibility

Status: **DESIGN_FROZEN_IMPLEMENTATION_PENDING**. This is the separately human-authorized,
prospective D-dev-select diagnostic. The machine-readable authority is
`configs/inference_cf/srd2_g0.json`; an implementation discrepancy blocks execution.
No scientific inference, pulse, reference evaluation or Slurm submission occurred in this design
session. Core proposal v6, METHOD_CONTRACT, and all historical verdicts remain unchanged.

## 1. Question and limits

Can the original R2 `g_old=E*R_B` preserve D2 English first-token corrections while suppressing
correct-Mandarin changes, beyond an utterance-matched shuffled-dose control?
MECH-LANG0 is retrospective motivation: historical D2 had five English corrections, eight
Mandarin corruptions and one English corruption. Its S1-region subset retained three corrections
and three Mandarin corruptions. Neither that subset nor R2 detection AUROC demonstrates causal
safety. P2-DIR ended `P2_DIR_NO_NEW_DIRECTION_SUPPORTED`; SRC-CF0-P ended
`SRC_CF0_PILOT_CAUSAL_INSUFFICIENT`. Those records are closed.

This study uses a frozen readout direction, one layer, one positive orientation, one absolute
maximum chord, one continuous gate and four arms. It is a next-token intervention screen from
model-generated prefixes. There is no intervened autoregressive decoding, transcript-quality
claim, controller training, parameter update, direction/dose search or automatic confirmation.

## 2. Population frozen before inference

The complete canonical role has 7,919 utterances in 20 dialogues. The CPU identity inventory
excludes 300 known intervention-exposed IDs, leaving 7,619 eligible IDs. The registry includes
FULL300, DG-03/04, all Round-1 cell ID lists, historical Job-A F3/F4/F5 IDs, BASIS-A4's dialogue
panel, ST-LOC0, and the explicitly documented DG-02 acceptance utterance. Their canonical-role
union is the same 300 IDs. Sources, byte hashes and every roster exclusion reason are recorded in
`SRD2_G0_POPULATION.json`. This is a bounded inventory of documented exposure, not proof that an
otherwise undocumented historical run never touched another utterance.

For each sorted dialogue, sort eligible IDs by
`SHA256("SRD2-G0-population-v1|240924|" + UID)`, then UID; select the first 20. Concatenate sorted
dialogues in that order. This freezes 400 IDs, 20 per dialogue, zero FULL300 overlap. No duration,
reference, POI, prediction, gate, gradient, outcome or correctness feature participates in ranking.
Only identity, role, audio metadata/existence and known exposure determine eligibility. All 400
selected files are readable mono 16 kHz audio; full byte hashes are separate from historical
metadata audio hashes. The role parquet was read with the six-column metadata allowlist only.

Selected-ID hash:
`sha256:ff2e3054ed25ecf6ffc944552a98f368878568125cdb2ef446f64763408137ae`.
Selected execution-record hash:
`sha256:9a58ef115adb63924bb621557e348264f8273832941efbd0ef554cf5b2a276e5`.
Complete roster hash:
`sha256:28aa458ba2bbbc03b69087952075595d14ca2c655e1712ef3691930ccd29a566`.

These are new utterance-level development outcomes on the same previously used 20 dialogues.
They are not independent confirmation, untouched-speaker evidence or independent-dialogue
generalization. No confirm/test/router-calib/P3/transfer role is accessed.

## 3. Canonical baseline and every query

Frozen Whisper-large-v3, pinned historical model/tokenizer/preprocessing files, BF16, eager
attention, eval/dropout disabled. Original waveform preprocessing and 30-second heard horizon
are those of R2/Whisper. Forced-ZH transcribe prompt `[50258,50260,50360,50364]`; greedy, beam one,
no sampling/timestamps/previous-utterance conditioning; maximum 200 new tokens.

Use `cached.Branch`, `_processed` and `core_p1.processed_argmax` for a single pristine cached B0
trajectory. The explicit argmax uses the lowest token ID on ties, as the R2 generation/D2
readout convention does. `cached_greedy` is a loop reference, not an unexamined API substitution:
its `topk` next-ID selection does not guarantee that tie ordering. No other processor changes.
Suppress lists and their hash are those of the pinned generation config.

Let T be the number of emitted non-EOS content IDs. Inventory every prediction t=0..T-1 and
also t=T iff B0 actually stopped on EOS. The latter is a genuine EOS opportunity. A capped
trajectory gets no artificial terminal slot. Prefix is content[:t]; absolute query is 4+t-1.
For t>=1, the pre-step cache contains prompt+content[:t-1], and the input is content[t-1].

Forced-prompt query t=0 is logged and gated but is never edited under DG-02. Pulse eligibility
requires t>=1, a strict complete-UTF8 content prefix, no unexpected special content token, and
finite native residual/raw logits. All other inventory rows have explicit four-arm no-ops.
EOS as the predicted action is not itself a veto. UTF8, cap and special-token reasons are saved;
no reference-based token-position sampling occurs. Gate-provider failure sets g=0 with the
original reason, but does not remove an otherwise valid D2 query from the shuffle pool.

## 4. Exact R2 gate and full-replay/cached compatibility

The authoritative gate attention comes from unchanged R2 `full_replay(use_cache=False)` on the
exact cached-B0 content sequence and the same clean encoder output. Read query 4+t-1. Decoder
causal masking prevents later baseline tokens from influencing it. At predetermined first,
middle and last eligible queries per utterance, verify against prefix-only replay; no future
token enters the gate API. **Do not use cached attention to replace this source.**

The ten heads are `[[7,0],[10,17],[12,18],[13,12],[16,1],[17,14],[19,11],[21,4],[24,1],[25,6]]`.
Use original `core_r2.max_attention_window`: average heads, restrict to actually heard frames,
renormalize, select earliest maximum 50-frame/one-second contiguous mass window. Preserve the
original short-audio and <=319-sample partial-terminal left shift. No S1 association threshold.
Native language readout is SOT-only, full 100-language-token softmax on the actual crop padded to
30 seconds. Recompute the all-zero 30-second null once with identical settings.

`A=log((pi_EN+eps)/(pi_ZH+eps))-log((pi_null_EN+eps)/(pi_null_ZH+eps))`,
`E=max(0,tanh(A/2))`, `R_B=max(0,P_M_B-P_E_B)`, `g=E*R_B`, eps=1e-12.
The original R2 script mass uses **raw**, unsuppressed float32 full-vocabulary log-softmax.
D2's objective uses generation-processed logits. Preserve this intentional historical
distinction. Both share tokenizer partition
`sha256:7daa50091056677286dd30933300cb9b1245ca778f969df2a95ccc06016dac02`.
Do not renormalize EN/ZH language pairs, threshold the gate, add low-Q vetoes, substitute g_cf,
use S1 regions, or calibrate from retrospective corrections. Ecf/Russian/donor branches are
unneeded for g_old and are not executed. Original required-input fallback precedence is in config.

Cross-path comparison is numerical, not bitwise. Every source/prefix/query/encoder identity must
match. On all structurally eligible queries, frozen checks are: raw-distribution TV median<=.01,
p99<=.05, maximum<=.10; p99 absolute difference in each raw script mass<=.02; processed argmax
agreement>=.995; site relative L2 p99<=.02 and maximum<=.05; site cosine>=.999 everywhere.
Quantiles use NumPy linear interpolation, with no selected subset. Prefix-only/full replay TV<=.05
and heard-normalized mean cross-attention L1<=.02 at the fixed probes; decoder self-attention
future mass<=1e-7. Null EN/ZH probabilities match historical pins within absolute1e-6.
These are prospective engineering bounds for BF16 kernel-path variation, not loosened historical
identity tests or gate calibration. Large distribution/site differences indicate a different
query mechanism and block the study. All checks must pass before Job B. No alternate path follows
a failure. Cached replay versus its own cached baseline remains bitwise.
TV is half the full-vocabulary L1 distance between float32 raw-logit softmaxes; site relative L2
uses the cached site norm as denominator and float64 vector geometry. Probe attention compares
the heard-masked, renormalized ten-head mean. Zero site norm or missing critical evidence fails.

## 5. Unchanged D2 and four pristine arms

At L16 DG-02, `J=log(P_E+1e-12)-log(P_M+1e-12)`. Use the existing `ReadoutDirection` /
`readout_direction`: native-BF16 scratch query, zero float32 residual probe, one
`autograd.grad(J,probe)`, then CPU-float64 radial projection and historical normalization.
Every model parameter is frozen, without gradient or optimizer updates. Direction/gradient
guards and scratch isolation are unchanged. Scratch site/logits must equal the ordinary cached
query bitwise. Construct once per structural query, even when g=0; reuse this exact direction
across the intervention arms. Tiny/invalid direction means recorded no-edit on all arms.

| Arm | Direction | Desired realized chord |
|---|---|---|
| B0 NONE | none | 0 |
| B1 D2_UNGATED | original +D2 | e*=1.1260757575454359 |
| B2 D2_R2_GATE | same +D2 | e* g_t |
| B3 D2_SHUFFLED_GATE | same +D2 | e* g_shuffled_t |

After Job A, hash-sort eligible donor indices by
`SHA256("SRD2-G0-gate-permutation-v1|240924|UID|t")`, then t. Assign their gates in that order to
recipient indices in ascending t, within the same utterance only. Zero gates stay in the pool.
Structurally invalid rows remain zero and outside it. Singletons, constant gates, and identity
permutations are retained and flagged; no reroll or exclusion. Direction invalidity discovered in
Job B does not alter the frozen permutation. Commit/push this assignment before pulses.

Each arm starts from the same pristine pre-query cache/encoder/history. Deep-copy scratch caches
or crop/replay with the existing restoration checks. Never advance an intervention output as
another arm's history. Only the clean baseline token advances the persistent replay. B0 may reuse
the clean query output; duplicate exact no-op outputs may reference the same lossless blob.

## 6. Realized dose and quantization

Use historical `apply_steering`/NormPreserve and `p2r.solve_scale` with `loc0_sites.pulse_action`.
The **solver target** is e*, e*g or e*g_shuffled, not the old coefficient times g. CPU-float64
chord geometry, at most eight native-BF16 emulations, unchanged epsilon/repair, no depth scaling.
Zero target bypasses division/solver/edit and is bitwise B0. Unreachable or off-tolerance targets
produce explicit no-edit, without a gate floor, dose substitution, redistribution or new strength.

Executed edits require <=2% relative squared-energy error for proposed and consumed chords,
<=2% chord error, original solver/hook absolute tolerance1e-6*max(1,edit_norm), and
abs(consumed_chord_norm/proposed_chord_norm-1)<=.005. This is the historical **norm-ratio** guard,
not a new relative vector-L2 test. Native norm rounding remains descriptive.
Before a pulse, record pristine native q,u,r=q+u and predict the actual repair consumption
`q+(u+(proposed_r-r))` using the unchanged DG-02 arithmetic on the same native device/dtype.
If that fixed solver result cannot pass the consumed-energy/norm guards, record
`native_consumption_unattainable` and exact no-edit, retaining its planned dose. No new scale,
gate floor or numerical threshold is introduced. Solve once, at most eight emulations; the hook
can consume a hash-verified cached solver result. Actual FFN input must match this prediction
bitwise. No downstream logits or correctness enter the preview. This prevents a tiny BF16 dose
from being knowingly executed off-target and then credited as a matched intervention.
Exact zero, pristine-cache restoration, no lingering hooks and unchanged weights are mandatory.
At least95% of nonzero B1 requests must be matched; otherwise INVALID. For B2/B3 report every
abstention, planned and realized squared-energy totals, relative errors, and per-utterance ratios.
Each may lose at most10% of its planned squared energy; their global realized-energy ratio must
lie [.90,1.10]. Planned bit multisets/totals must match exactly; sum sorted desired squared norms
with math.fsum so permutation does not introduce summation-order error. A zero total or failed realized
matching makes the selectivity comparison INCONCLUSIVE; it never permits an energy-normalized
post-hoc winner. Report per-dialogue and per-stratum distributions without claiming exact matching
within those strata (the gate deliberately redistributes dose there).

## 7. Conditional jobs and reference-free stop checks

Job A captures baseline, complete inventory, full-replay gates and compatibility evidence only.
It requires400/400 rows,20 dialogues, exact source/query identity, all compatibility checks,
>=1,000 structural queries across20 dialogues, >=20 nonzero gates across>=5 dialogues, and
finite required gate providers on>=99% of complete-prefix queries. Structural UTF8 fallbacks do
not count as provider failures. Failures stop globally, never select rows for advancement.
Seal/push, independently audit, seal the shuffle, and issue a minimal global authorization.

Job B is conditional on that authorization and the measured query-count cost forecast<=9,000s.
It verifies historical apparatus on the already-audited P2-DIR spot set, outside new outcome
denominators, then computes all four arms on **every** frozen query. Original clean state/logit,
zero-dose, D2 scratch identity and restoration criteria apply; no looser historical reproduction
rule is introduced. Do not import historical target/stratum fields for these engineering checks.
Seal/push the complete pulse matrix before references. A hard time limit aborts and preserves
the incomplete attempt; it does not authorize another scientific job or a smaller panel.

## 8. Post-seal evaluator and denominators

Use canonical normalize/segment_units/align_tokens and R2 `unit_to_token_positions`. Reference
units map by checked normalized hypothesis offsets to the first generated-token query. Missing
offsets, capped gaps, target-set failures and collisions are explicit unalignable reasons. Use
historical `target_set` for English; ZH-correct retains its historical singleton baseline token.
Different-stratum collisions or multiple English units sharing a query are excluded from primary
contrasts with counts; same-category Mandarin mappings deduplicate to one query. No CTC/MMS-FA,
oracle timing or gold teacher forcing is used by the runner or needed by the short-audio evaluator.
The 23 selected utterances longer than30s still receive all four arms; without a frozen heard-scope
timing source their whole-reference lexical attribution is conservatively primary-unalignable,
not silently credited to the heard segment. They remain in400-row and mapping-coverage ledgers.

Primary EN-confusion is wrong-language substitution/script transliteration, not every English
error. Deletion/EOS opportunities, same-language substitutions and full-word recovery are
descriptive only. Correction: B0 top1 notin acceptable first-token set Y, arm top1 in Y.
Corruption: B0 top1 in Y, arm top1 notin Y. Lowest-ID ties. Mandarin singleton changes are the
historical next-token preservation endpoint, not verified full-word/transcript damage.

Report all400 IDs/all inventory queries; every mapping failure; mapped stratum denominators;
structural-eligible, gate-positive and actually-edited subsets; unconditional and active corruption
rates; correction/harm dialogues; gate and planned/realized energy distributions; EOS risk proxies.
No-edit/ineligible positions remain in mapped-stratum denominators, with conditional active
results beside them. Multiple positions within a dialogue do not supply independent samples.

## 9. Frozen feasibility criteria

Minimum opportunities: mapped EN-confusion>=30 and EN-correct>=30, each>=10 dialogues;
ZH-correct>=100 across>=10 dialogues; actually edited B2 ZH-correct>=40 across>=10 dialogues;
unique query/gap mapping of>=80% of normalized Latin reference units on the400 rows, before
target-set/stratum filtering (unheard-scope rows count unmapped). Insufficient opportunities are
INCONCLUSIVE, never safety passes. No correctness label influences Job A or position inclusion.

Causal power: B2 has>=5 actual EN-confusion corrections across>=3 dialogues. Margin or script
mass alone cannot pass. Control opportunities require B1>=5 corrections and>=4 ZH corruptions
across>=3 harmed dialogues, valid energy matching, and at most50% of structural queries in
uninformative shuffle utterances. A near-zero harm comparator cannot establish harm reduction.

Let C_a be corrected English-confusion queries, H_ZH_a/H_EN_a correct-state corruptions,
and U_a=C_a-H_ZH_a-H_EN_a. Equal event costs are predeclared, not optimized. Require ALL:

- C_B2>=ceil(.60*C_B1); H_ZH_B2<=floor(.50*H_ZH_B1).
- H_ZH_B2<=10, unconditional ZH corruption<=.01 and active ZH corruption<=.10.
- H_EN_B2<=3 and unconditional English-correct corruption<=.02.
- C_B2-C_B3>=2; U_B2-U_B3>=3; H_ZH_B2<=H_ZH_B3.
- U_B2>=1, requiring positive net correction-minus-correct-state-harm count at the frozen event costs.
- Positive paired correction gain in>=2 dialogues; positive paired utility gain in>=3 dialogues;
  largest dialogue share of positive paired utility gain<=.60.

This requires retained useful corrections, absolute protection and an advantage over the same
planned-dose distribution, rather than lower energy alone. It is a preliminary effect-size/
breadth criterion, not a significance claim or evidence of broad deployable safety.

Conservative adverse stop on B2 if ANY: >=10 ZH corruptions AND unconditional rate>.03;
any dialogue>=3 ZH corruptions AND its active rate>.20; >=5 English-correct corruptions AND
unconditional rate>.05; or any newly proposed EOS where B0 has>=10 remaining content tokens.
The EOS condition is an early-stop risk proxy on a single query, not a measured sequence truncation.
Control-arm damage is fully reported but does not itself invalidate the candidate arm.

## 10. Statistics and terminal precedence

Paired dialogue bootstrap: seed240924,10,000 draws of20 sorted dialogues with replacement, all
queries/utterances retained and the same draw for every arm. Report pooled counts/rates and
equal-dialogue macro rates; percentile95% intervals descriptively. Four predeclared primary
comparisons are C_B2-C_B3, U_B2-U_B3, H_ZH_B3-H_ZH_B2, H_ZH_B1-H_ZH_B2; report Bonferroni
98.75% marginal intervals (quantiles.00625/.99375). Undefined rates are not zero; rate intervals
need>=9,500 defined draws. Count/breadth gates, not CI significance, control this bounded feasibility
label. A favorable point label with a CI spanning zero is disclosed as uncertain development
evidence. No arm/comparator/metric is chosen after outcomes; no superiority or safety hypothesis
test is claimed from a sparse count gate.

Apply exactly, first matching condition:

1. `SRD2_G0_INVALID`: any critical provenance, compatibility, site/cache/model, firewall, incomplete
   matrix, energy integrity, evaluator or audit failure. STOP.
2. `SRD2_G0_COMPUTE_BLOCKED`: no critical invalidity, but pre-job projection>9,000s or hard ceiling
   cannot be met. STOP for an explicit human resource revision; no subset/extra job.
3. `SRD2_G0_OBSERVED_DAMAGE`: any adverse B2 condition above. STOP.
4. `SRD2_G0_OPPORTUNITY_INSUFFICIENT`: Job A coverage or evaluator opportunities fail. STOP.
5. `SRD2_G0_CAUSAL_POWER_INSUFFICIENT`: B2 correction/breadth gate fails. STOP.
6. `SRD2_G0_CONTROL_COMPARISON_INCONCLUSIVE`: comparator opportunities, shuffle informativeness
   or planned/realized dose-distribution checks fail. STOP.
7. `SRD2_G0_NONSELECTIVE`: remaining acceptance condition fails. STOP.
8. `SRD2_G0_SELECTIVE_FEASIBILITY`: every preceding gate passes, independent FULL audit PASS.

Only the last label may recommend a separately human-frozen next stage. It does not authorize
full decoding, direction/gate revision, confirmation, test, transfer or an automatic experiment.
Both a failed and a positive study remain development evidence on the same20 dialogues.

## 11. Resources and audit

At most two scientific H100/MIG-compatible jobs, each<=3h, no extra layer/dose/sign. Reference-free
forecast: Job A approximately25–55min, Job B approximately35–80min, conservatively4,807s for Job B.
This is an estimate from historical timings, not a new throughput measurement. The absolute
80,000-query worst case does not fit; Job A must recost its complete inventory before Job B.
`SRD2_G0_COMPUTE_FEASIBILITY.md` defines the margin and stop formula.

Independent pre-run `PASS_TO_SRD2_G0`, post-A PASS, PRIMARY pulse PASS before references, and FULL
PASS after reference scoring are required. Auditor may reuse mechanical hashing/storage tools
but cannot import primary gate/label decision code. It recomputes g/shuffle/energies, mapped
correction/corruption counts, rates, every terminal predicate and audit barriers independently.
See FIREWALL and CLAUDE_HANDOFF for exact stages and required future interfaces. Historical
data/results/contracts stay unchanged; scientific implementation is still pending.
