# SRC-CF0-P — construction and preliminary causal feasibility

Status: **DESIGN_FROZEN_IMPLEMENTATION_PENDING**. Human-authorized narrower hypothesis,
frozen before any new model forward, direction outcome or lexical scoring. Initial HEAD
`20cd36824822c356f910f42c9b5ca3a5fd5f56b6`. The original `SRC_CF0_DESIGN_BLOCKED`
specification, panel, configuration and record remain unchanged. This pilot does not reopen
S1, R0, ST-LOC0, ST-PROMPT-R1, P2-DIR or the closed controller line; core v6 is unchanged.
Machine-readable authority: `configs/inference_cf/src_cf0_pilot.json`. Implementation and
independent runnable audit are still required. No scientific execution in this Codex freeze.

## Question and claim boundary

Does the original-minus-predicted-English-masked decoder residual cause actual English
first-token corrections beyond matched-energy random and off-target acoustic directions?
The maximum positive verdict is `SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED`. It supports
proposing a separately frozen expanded-development safety study only. It cannot authorize
CF1, TTO-CF, full-sequence steering, confirmation/test data or speech-LLM transfer.
A margin, norm, cosine or English probability increase alone cannot establish correction power.

## Frozen population and counterfactuals

Reuse `SRC_CF0_PANEL.json` by file and identity hash; its runtime queries and utterances are
exactly the S1/P2-R/ST-LOC0 panel: 180 positions, 80 utterances, 20 dialogues; 60 per stratum.
Identity `sha256:2f66af05f61a73ff8dd7df825f86c6a7dc69389f3be807f312399a242464d371`;
runtime membership `sha256:1bbaaa1c665c13335fc329ef1cb93fe061d176cdc6129988b37e454c21763236`.
No selection, replacement, resampling or label-dependent runtime eligibility.

Historical target/paired opportunity ceilings are EN-confusion 28/20 (12/9 dialogues),
EN-correct 39/35 (17/16), and ZH-correct 6/4 (6/4). Totals 73 targets and 59 pairs.
These are existing S1 region opportunities, not newly measured direction validity.
Numerical guards can only reduce them. The English cohort can support a small preliminary
paired screen; the four paired Mandarin queries cannot answer comprehensive active safety.

Inherit the complete S1 region and counterfactual objects verbatim in the new config:
R0 PRIMARY native-LID predicted EN intervals; original forced-ZH cached-query attention
from ten frozen alignment heads, raw heard mass >=0.50; integrate fractional 320-sample
frames over actual heard duration. Heard horizon min(N,480000), sample rate 16000.
EN interval >=4000 samples; crop length min(interval,32000); deterministic 320-sample
starts plus right anchor; highest attention then earliest start/end; association >=0.10,
RMS >=1e-4. Off-target is disjoint and equal duration, EN fraction <=0.10,
attention <=min(0.10,target_mass/2), RMS >=1e-4, waveform energy ratio [0.5,2].
Off-target order: lowest attention, nearest absolute log energy ratio, earliest start.
Hard positive float32 zero over [a,b), no taper; all other waveform bytes unchanged.
Whole features/encoder recomputed for each distinct masked waveform; cold model-owned
forced-ZH decoder cache. No absent-region fallback, guard relaxation, oracle timing or mask sweep.

## Direction, site and numerical guards

Frozen Whisper-large-v3, eval, no autograd or optimizer. Layers [16,24], zero-based;
DG-02 post-cross-attention residual before FFN, dimension 1280. Same frozen forced-ZH
transcribe prompt [50258,50260,50360,50364], generated content[:t], absolute query 4+t-1,
model, tokenizer, audio preprocessing and decoder-site tensor for clean and masked passes.
No independent masked transcript or gold teacher forcing.

`delta = float64(h_clean) - float64(h_mask)` after lossless upcast of native BF16 states.
`v = float32(delta / (norm(delta)+1e-6))`. Preserve historical epsilon semantics rather than
forcing serialization to have norm exactly one. Solver separately normalizes the vector.
Nonfinite/incorrect-shape inputs or clean norm <1e-8 are critical INVALID. Raw delta norm
<1e-4 is ABSTAIN_DEGENERATE/no edit. Off-target uses precisely the same algorithm.

Report raw norm, vector norm/hash, signed and absolute target/off cosine, direction/residual
cosine, `v_perp=v-h*dot(h,v)/dot(h,h)`, tangent ratio `norm(v_perp)/norm(v)`, and raw tangent
norm. Geometry is descriptive except the predeclared validity gates below. Construction
distinction from off-target is not proof of lexical or phonetic specificity.

Each available clean/target/off input is recomputed twice using independent cold encoder
and decoder caches; require bitwise native residuals and raw logits, and identical vector
bytes. Record both layers in the same passive pass. Unexpected repeat mismatch is INVALID,
not a statistical tolerance adjustment. No cache computed from one audio/state can be used
for another; restore original model/hooks and pristine query caches after every pulse.

## Dose and complete matrix

P-B: layers [16,24] x eta [0.15,0.30] x signs [+1,-1] = eight primary arms.
Eight off-target arms and eight random arms mirror the same configurations. NONE and zero
are mandatory. All 180 rows exist in every arm; absent/degenerate/unreachable inputs emit
exact baseline with an explicit no-edit reason. No arm or stratum is dropped.

Use unchanged historical `apply_steering` through the DG-02 hook: NormPreserve enabled,
depth scaling disabled. CPU float64 P2-R geometry solves nonnegative scale for signed unit
v and target eta*norm(native BF16 h); exact native BF16 repair emulation, max 8 evaluations.
Require realized eta relative error <=0.02 AND relative squared-energy error <=0.02;
paired squared-energy difference <=0.02; consumed/proposed relative error <=0.005;
solver/hook absolute edit-norm difference <=1e-6*max(1,edit_norm). Unreachable means no edit,
not cap reinterpretation or dose substitution. BF16 norm-rounding is descriptive only:
ST-PROMPT-R1 amendment A1 did not freeze an additional 0.5% norm-preservation validity gate.

Random: one PCG64 isotropic 1280-dimensional standard-normal float64 vector per UID/t/layer,
normalized, serialized float32. Seed is first 16 hex SHA256 digits of
`SRC_CF0_P-random-v1|240924|UID|t|layer`; same vector for both signs/doses. No projection,
resampling or outcome selection. Random is compared only on matching target eligibility;
off-target comparison uses identical joint-valid positions, directions and realized energies.

## P-A construction gate

Critical identities, repeatability, complete180x2 records, reference firewall, model/site/cache
restoration and historical reproduction must all pass. After direction seal and independent
construction audit, a separate evaluator joins existing strata for coverage only.

Per layer require ALL construction coverage/geometry conditions:

- Numerical target validity >=90% of 73 raw targets (>=66); off-target validity >=90% of
  59 raw pairs (>=54). Tiny vectors abstain; nonfinite inputs invalidate the run.
- >=20 numerically valid EN-confusion targets across >=8 dialogues.
- >=15 joint-valid EN-confusion pairs across >=6 dialogues, and >=20 EN-correct pairs
  across >=10 dialogues. Joint means target/off/random tangent ratio >=0.25 and native
  reachability/energy validity for BOTH signs at BOTH doses. No minimum ZH count is claimed.
- On all numerically valid target/off EN-confusion pairs before energy filtering: >=12
  pairs across >=6 dialogues; median absolute target/off cosine <=0.90 and median
  target/off RAW tangent-norm ratio >=1.25. Zero off tangent denominator abstains, never infinity.

At least one layer must pass coverage/geometry AND construction specificity. If none meets
coverage/geometry -> CONSTRUCTION_INSUFFICIENT. If some meets those but none specificity ->
DIRECTION_NONSPECIFIC. Otherwise freeze qualified layer IDs in a global authorization;
P-B still records the COMPLETE eight-arm matrix. Only already-qualified layers can supply
a selected arm. These thresholds preserve most of the 20 paired opportunities, prevent a
one-dialogue construction result and require measurable distinction from generic masking.
They do not establish that a vector carries the correct word.

## P-B lexical gate and mandatory damage screen

Require complete eight-primary/eight-random/eight-off matrix, NONE/zero, energy and causal
integrity. Actual valid edits must retain >=90% of the P-A sealed joint-reachable prediction
per arm; unexpected repeatability or consumed-site mismatch is INVALID. No newly eligible
row is substituted. Full-stratum denominators remain 60; also report edited, paired and
unreachable denominators separately. Pair cohort is fixed by reference-free validity only,
not target availability or success.

A single prequalified arm has correction power iff ALL:

- >=15 paired EN-confusion rows across >=6 dialogues.
- >=3 actual top-1 corrections on that pair cohort across >=3 dialogues.
- Net corrected-position advantage >=2 over EACH matched control, positive net advantage
  in >=2 dialogues for each, and dialogue-macro corrected-indicator advantage >=0.10 each.
- For each control, net advantage remains strictly positive after removing EACH represented
  pair-cohort dialogue. This tests dominance, not formal significance.

Correction: clean processed top1 outside the historical acceptable first-token set, edited
top1 inside. Corruption reverses that condition on clean-correct positions. Full vocabulary,
canonical generation processing, float32 readout; equal processed values tie by lower token ID.
No candidate union, first-token accessibility filter or gold token enters inference.

A power-qualified arm additionally needs observed EN-correct corruptions <=3, ZH-correct
corruptions=0 and correct-state new EOS promotions=0, counted on ALL its target-active
opportunities (including target-only). This is an adverse-event screen, NOT broad safety.
Global serious-damage stop if ANY primary arm has >=12 EN-correct corruptions, >=3
ZH-correct corruptions or >=2 correct-state EOS promotions. Evaluate this only after sealing;
never use references to modify a live run. All active Mandarin changes must be enumerated.
No-edit54/60 Mandarin rows cannot be counted as active safety evidence.

## Statistics, selection and interpretation

Pooled counts are primary; conditional edited/pair rates and equal-dialogue macro rates are
separate. Use 10,000 paired dialogue-cluster bootstrap draws, seed240924, sampling20
canonical sorted dialogue IDs with replacement, shared indices across systems. Macro means
over represented cohort dialogues, counting repeated sampled dialogues with multiplicity.
Empty draws undefined; >=9900 finite draws needed for a reported interval.
16 primary contrasts =8 arms x2 correction-indicator controls; Bonferroni simultaneous95%
interval quantiles .0015625/.9984375. Report counts, macro differences and intervals; these
intervals DO NOT gate progression. Small paired cohorts cannot support a compulsory
adjusted-significance claim from three flips. Prespecified counts, breadth, control advantage
and LODO consistency define preliminary evidence, never statistical confirmation.

Separate eight-arm margin-only diagnostic family: mean dialogue-macro EN-confusion margin
>=0.50 nat and Bonferroni lower95% >0 (quantiles .003125/.996875), target minus NONE on
matching valid cohort. A positive margin without the full correction gate cannot advance.
Correct-state harm uncertainty is descriptive clustered uncertainty; a zero-event percentile
bootstrap is not a valid upper harm bound. Report zero events and opportunity counts explicitly.

Select at most one qualifying configuration: more paired corrections; more corrected
dialogues; larger minimum net control advantage; fewer ZH corruptions; fewer EN corruptions;
lower eta; lower layer; lexicographic arm ID. No oracle union of arms can advance. Sign key
must be stable (`L16_eta0.15_minus`, `..._plus`, etc.).

## Exact terminal precedence

1. `SRC_CF0_PILOT_INVALID`: any critical identity/firewall/coverage-matrix/audit failure.
2. `SRC_CF0_PILOT_CONSTRUCTION_INSUFFICIENT`: no layer passes coverage/geometry.
3. `SRC_CF0_PILOT_DIRECTION_NONSPECIFIC`: coverage/geometry passes but no layer passes specificity.
4. `SRC_CF0_PILOT_OBSERVED_DAMAGE`: global serious-damage stop met.
5. `SRC_CF0_PILOT_SIGNAL_SAFETY_UNRESOLVED`: at least one single arm passes power AND
   observed damage screen; full independent audit PASS required.
6. `SRC_CF0_PILOT_OBSERVED_DAMAGE`: power-qualified arms exist but all fail observed screen.
7. `SRC_CF0_PILOT_MARGIN_ONLY`: no passing power arm; at least one prequalified valid arm
   meets the adjusted margin diagnostic.
8. `SRC_CF0_PILOT_CAUSAL_INSUFFICIENT`: otherwise.

Failed audits override every scientific label. Undefined quantities cannot satisfy a gate.
No negative/partial label progresses. Even the strongest label leaves Mandarin safety
unresolved and recommends only a new safety freeze. Construction failures do not run P-B.

## Reproduction, compute and publication

Exact180 panel/prefix/audio/model/tokenizer/generation hashes. Clean raw logits bitwise S1 M
and historical NONE; native post-cross residuals match ST-LOC0 archives where present;
pre-FFN input confirms actual consumption. All180 L16 D0+/- and sealed D2 apparatus at
old e*=1.1260757575454359 reproduce old pulse logits/consumed states bitwise. Inherit old
D0 vector abs1e-6, D2 abs2e-6 and D0 scale1e-9*max(1,s) tolerances; do not loosen them.
No reference-gradient computation for the apparatus. Zero pulse bitwise at BOTH layers,
hook restoration/cache isolation/model-weight identity verified. Any scientific mismatch stops.

At most two scientific jobs later: P-A construction, conditional P-B pulses, <=3h EACH,
target5–20min each. Conservative construction estimate408 encoder evaluations and13,318
cached decoder steps for independent repeats of80 originals+124 distinct S1 masks; record
both layers together. Pulse upper bound4,320 primary/control query forwards plus <=540
NONE/zero,80 clean encoders and3,159 clean-prefix steps, with pristine query cache snapshots.
Historical apparatus adds at most540 query forwards. Group repeated queries/masks by exact
hash. Ineligible rows skip forward but remain complete no-edit records. Forecast must be
validated by future reference-free engineering throughput before production. If >3h STOP
for explicit compute revision; no matrix/population reduction. No jobs/model forwards here.
