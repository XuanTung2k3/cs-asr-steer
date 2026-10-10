# ST-PROMPT-R1 — layer × relative-dose causal feasibility

PRE-OUTCOME SCIENTIFIC DESIGN FROZEN; runners and independent runnable audits pending.
Base commit: `0687da04de8f4f3685c884197f4f9d96f64ee324`. JSON configuration is the numerical contract.
No GPU/model forwards, new edited logits, new lexical outcomes, or new R1 reference evaluation occurred in this freeze.
CPU bf16 repair of sealed states is engineering evidence only.

## Authority and scope

This human-authorized study is separate from terminal ST-LOC0 and R0. Layers3/8 are expressly
additional supporting diagnostic layers, not replacements for core METHOD_CONTRACT/v6 layers16/24.
No core method, historical report or earlier stop rule changes. V2R3 dose response used a D-construct
mean-shift direction and gold teacher forcing; it motivates checking dose but does not identify
v_prompt or transfer its alpha scale. P2-SEQ deletion/EOS damage motivates the conditional sequence screen.
No training, unique direction, controller, TTA, new site or automatic R2 execution.

## Population and direction

180 exact historical positions,60 per EN-confusion/EN-correct/ZH-correct;80 utterances,20 dialogues.
Panel identity `sha256:41168f57a2ef443ab73c2979e3e95f16b1a57b3cf6f4ee93aed4d968cc088b6d`; parent identity `sha256:fa9d33c90f54f873dac2f2662878957be8313b9b97f43f7bd2caf3c22f2a6f69`.
Preserve historical order, B0M_L16 tokens, audio hashes, query t, fixed reference set and competitor.
Runtime receives ONLY runtime_queries and utterances; evaluator membership/targets never travel with it.
Conditions: M=[50258,50260,50360,50364], E=[50258,50259,50360,50364], task transcribe.
At step t use identical generated prefix[:t], absolute query4+t-1. No future token as input.
Historical eligibility t>=1 means the predicting query is outside the four-token forced prefix.
EOS50257; cap200; batch1; bf16/eager frozen eval model; suppression at every step and begin suppression
only at step0; lowest-ID processed argmax ties. Model/tokenizer/partition/suppression hashes inherit ST-LOC0.

Dynamic direction is exactly core_p1.direction(h_E,h_M): upcast each native state to float64 BEFORE
subtraction, delta=E-M; float32(delta/(norm(delta)+1e-6)); tiny norm<1e-4/nonfinite returns no-edit.
Historical fallback is abstention, never another vector. Signs +/- negate this same vector.
Solver unit-normalization is historical and separate from the epsilon-normalized provider.

## Physical site and energy

Zero-based decoder layers3,8,16,24. Only post-cross-attention residual r=q+u_source, before FFN.
Recorder hooks encoder_attn_layer_norm input and encoder_attn output[0]; final_layer_norm pre-hook
independently verifies consumed r. Intervention returns u_source+(repaired_r-r); measure the residual
ACTUALLY consumed after the native addition, including bf16 rounding. Dropout disabled, no post-FFN edit,
no depth scaling. Installed Transformers4.57.6 forward source is pinned in config.site_forward.

For each signed vector and native bf16 r: target=eta*norm(float64(r)), eta=0.15,0.30,0.45.
Use unchanged P2-R solve_scale/scaled_direction/emulate_edit_norm and apply_steering. CPUfloat64 angle
phi=acos(dot(r,vhat)/norm(r)); theta=2asin(target/(2norm(r))); reject target>=2norm(r) or
phi<=theta+1e-9. Start s=norm(r)*sin(theta)/sin(phi-theta); native repair/secant, maximum8 evaluations.
alpha=scale=gain=1 at queried position; norm repair epsilon1e-6. No outcome enters scalar solving.
Actual eta relative error<=0.02 AND squared-energy relative error<=0.02; same-target candidate/random
pairwise squared-energy mismatch<=0.02. Solver/hook norm disagreement<=1e-6*max(1,realized_hook_edit_norm), consumed/proposed
relative norm deviation<=0.005. Other positions zero gain and bitwise unchanged.
Geometric/tiny/solver failures produce explicit no-edit placeholders, zero outcome change, original denominators.
Unexpected consumed-energy, nonfinite baseline, cache or hook violations INVALID rather than silent abstention.

CPU review on exact sealed L16/L24 paired states covers2160 primary geometric cases; evidence artifact
ST_PROMPT_R1_REACHABILITY.json. L3/L8 R0 archives contain M states but only an averaged prompt comparator,
not paired per-query E states: do not substitute that mean or claim complete preflight. The later allocation
must passively extract all four layers on the EXACT180 prefixes, then verify all36 dose cells/query BEFORE
new pulses. If a dose has insufficient engineering coverage for selection, preserve every arm/cell; stop
before outcomes for explicit human dose revision rather than silently pruning the ladder. Required coverage
for this pre-pulse check is>=54/60 and>=12 dialogues in EACH stratum for EVERY primary/random arm.
This is a frozen execution condition, not permission for a scientific outcome-based redesign.

## R1-A matrix, barriers and measures

24 primary arms;12 fixed PCG64 Gaussian random vectors, one per layer/dose, seeds/hashes in config;
NONE180; D2L16 at original e*=1.1260757575454359, apparatus only; zero720.
All pulses start from pristine B0 caches and identical prefixes; crop/clone scratch caches after each pulse.
No prior edit, continuation or persistent state. E/M extraction may record four layers in one paired replay.
Reuse only exact prefix/model/cache/input compatible states. Do not replace the canonical baseline with R0/PATH B0.

Before new pulses reproduce L16 states/NONE logits bitwise; historical D0 vectors maxabs1e-6;
ST-LOC0 L16 plus/minus pulses and D2 at old energy bitwise consumed states/logits; old solver scale
abs1e-9; D2 sealed vector maxabs2e-6/provenance; all-layer zero bitwise and full weight/cache restoration.
Historical tolerances in config are binding. Barrier failure=>INVALID, stop before new interpretation.

Seal lossless full logits, paired vectors, native before/after consumed states, solver traces/status,
actual eta, cache lineage, zero/restoration proofs, complete cell manifest. Then push + independent
reference-free audit, ONLY THEN evaluator accesses P2-RJ frozen reference sets and competitors.
Primary margin m=logsumexp(z_processed[Y_ref])-z_processed[c_fixed]; c never recomputed after edits.
Correction: confusion baseline wrong, edited argmax in Y_ref; corruption: correct baseline in Y_ref,
edited argmax outside Y_ref. Rank lowest-ID ties, target logprob, top1changes, processed EN/ZH mass,
KL(P_edit||P_NONE) over generation-valid support, all strata/dialogues/utterances reported. No top-k loss.

## R1-A uncertainty and deterministic decision

10000 shared paired draws, numpy.default_rng(240924),20 sorted dialogue clusters with replacement.
Mean positions within dialogue, then mean represented dialogues. Require>=9900 valid draws/endpoint.
Two qualifying endpoints per24 arms (vs NONE, vs matched random): Bonferroni family48,95% simultaneous
percentiles0.0005208333333333333/0.9994791666666667. This corrects jointly across all24 arms and both
claims; do not use unadjusted controls to qualify an arm. Correct-state CIs pointwise/descriptive.
Ineligible no-edit cells retained with delta0, valid-only tables secondary; eligible arm>=54valid/stratum
and>=12 represented valid dialogues/stratum. For the random contrast require the SAME minimum
joint-valid candidate/random counts and dialogues; invalid pairs contribute contrast0, and full intention-
to-screen candidate-vsNONE statistics remain unchanged. Report joint and valid-only tables explicitly.
Every planned cell required, missing cell invalidates stage.

Full positive gate: corrections>=3 in>=3 dialogues, macro confusion delta>=0.50nat,
adjusted lower vsNONE>0 AND paired vsrandom point>0/lower>0; correct corruptions<=5 total,
ZHcorruptions<=2, arm eligibility and all integrity/audits pass.
Precedence: INVALID; CAUSAL_PROMISE if any full positive arm; POWER_WITH_DAMAGE if any eligible arm
passes ALL benefit gates but fails corruption budgets; MARGIN_ONLY if any eligible arm has both
adjusted lower bounds>0 but no full benefit/safety pass; otherwise NO_CORRECTION_POWER.
Thus isolated corrections without full evidence cannot advance. Labels carry ST_PROMPT_R1_A_ prefix.
Select at most2 full-pass arms: corrections descending, corrected dialogues descending, ZH corruptions
ascending, EN corruptions ascending, adjusted vsNONE lower descending, eta ascending, configID lexical.
Exact float comparisons; no fitted threshold/tie tolerance. Push immutable selection before R1-B.

## Conditional R1-B schedules and continuation

Run ONLY after A_CAUSAL_PROMISE, pushed A seal/selection and independent A audit.80 original utterances,
including no-query no-op rows. Oracle builder uses ONLY existing EN-confusion locations from180 metadata:
unique sorted t per utterance. SINGLE earliest t; MULTI all these t. Exactly37 active utterances,
43 no-location no-op controls;37 SINGLE and60 MULTI locations, maximum2 pulses/utterance.
The full80 UID-to-index schedule hash is pinned in config.R1_B.schedule_identity_hash. No choice from A corrections, target
identity, rank, dose response or subsequent transcript. Schedule sealed/pushed before new decodes; runner
receives UID+t only. No replacement of skipped positions. Query indices remain decoder steps after
transcript divergence, not reference alignment/retargeting. Report this limitation and selection bias.

Matched B0/zero must reproduce B0M_L16. SINGLE replays authentic B0 prefix to earliest t, then one pulse
and releases greedily; before pulse tokens/caches/logits identical to B0. MULTI starts prompt and greedily
decodes its own trajectory, activating only frozen scheduled indices. At every activation replay CURRENT
emitted prefix unedited under E/M using independently owned scratch caches. Recompute direction, edit S
at selected layer/sign/eta, discard scratch; S alone owns edited history. No gold teacher forcing or future
query insertion. EOS stops immediately; cap200 preserved. A skipped/unreachable query gets no edit.

Per-utterance normalized squared-chord budget=N_scheduled*eta^2, one eta^2 packet/index, native tolerance
as above; actual physical squared energy summed/logged separately. Unspent packets never redistributed.
For each mode run signal first and seal reference-free per-index consumed-energy ledger BEFORE random.
Random at the same fixed indices uses precisely these chord targets on its OWN state/history; no signal
transcript, logits, reference identity or future tokens enter random direction construction. Zero companions
use same branching/schedule machinery without an edit. Signal/random budget transport is a diagnostic
control, not a deployable claim. EOS can prevent spending: never extend decode to match energy. Require
aggregate squared energy within2%,<=5% positive-budget utterances unmatched by>2%; otherwise comparison
INVALID and no sequence advancement (preserve transcript evidence). No hidden fairness exception.

Each selected config: B0, SINGLE, MULTI, RANDOM_SINGLE, RANDOM_MULTI, ZERO_SINGLE, ZERO_MULTI. No additional
selection between doses/signs/configs after A. At most4 signal sequence conditions; both modes reported.
Seal/push all outputs/cache/energy proofs, independent primary audit, then canonical evaluator on80.
Report PIER/MER/EN-WER/ZH-CER, POI transitions/retentions/outside-POI edits versus actual harm, S/D/I,
EOS/caps/severe truncations, edits/changed utterances/dialogues, total energy, latency/forwards/VRAM.

## R1-B safety, inference and stop

Corpus-rate paired dialogue bootstrap10000/seed240924; same draws. Family8 endpoints: PIER gain vs B0
and matched random for each of at most4 config/modes. Even one selected config retains8 slots;
Bonferroni percentiles0.003125/0.996875. No-edit control bitwise matched; its metric equality mandatory.
Benefit ALL:>=3 net POI-error reduction vsB0,>=3 dialogues each with positive net reduction,
PIER gains vsB0/random point>0 AND adjusted lower>0. Safety ALL: MERincrease<=.005, ZHCERincrease<=.005,
matrixretention>=.99, ENretention>=.95, POIcorruption<=.05, outside-POIharm<=.01, added deletion errors<=5,
added caps<=0, new severe truncations0. Rates corpus/canonical baseline-correct denominators; undefined
required rate prevents supported label. Severe=EOS output <50% B0 content length AND>=5 fewer tokens;
also report existing canonical EOS/truncation diagnostics without changing this fixed new-collapse rule.

Final precedence: INVALID; ORACLE_SEQUENCE_FEASIBLE if any condition passes benefit AND safety;
ORACLE_BENEFIT_WITH_DAMAGE if any passes benefit but none passes both; SEQUENCE_DAMAGE if no benefit
passes and any signal fails safety; otherwise NO_RELIABLE_SEQUENCE_BENEFIT. Prefix ST_PROMPT_R1_B_.
Every condition's safety outcome remains explicit even if another qualifies. Supported recommends ONLY
a separately frozen R2 reference-free location-optimization study. All other labels STOP. No automatic
multipulse-first experiment, retuning, new directions, extra layers/doses or full300/confirmation/test.

## Compute and auditing

One future A job, one conditional B, sequential, sbatch H100/MIG, hard10800seconds/job. A~40000forward
upper forecast,15–30min; B conservative320000forwards (worst2 configs,80*200steps and repeated E/M
scratch prefixes),20–60min pending CPU scheduling/limited non-outcome throughput estimate. Estimate
from ST-LOC0 17496forwards/385seconds; allow ample overhead. Require measured conservative forecast
below3h before submission; stop for human compute revision otherwise, never shrink matrix/population.
No GPU science in this session. Manifest via provenance/specfreeze records resolved config, git clean
HEAD, model/input/source hashes, env/dtypes, scheduler/counters, immutable RUNNING→COMPLETE/INVALID.
Require PASS_TO_ST_PROMPT_R1_A and ST_PROMPT_R1_A_AUDIT: PASS, then conditional analogous B gates.
Independent auditor must not import primary analysis/decision/bootstrap; recompute counts, shared draws,
confidence bounds, decisions, site/energy/cache identities and schedule projection. Audit failure INVALID.
