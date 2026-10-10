# S1 — Acoustic Evidence Feasibility (frozen v1)

Status: DESIGN_FROZEN_IMPLEMENTATION_PENDING. Separate supporting inference_cf study,
not a revision of the v6 core method. No new model inference or S1 outcomes in the freeze.
Machine-readable numerical authority: `configs/inference_cf/s1_acoustic_evidence.json`.
Both lexical headroom H and acoustic discrimination D are necessary. A positive result is
first-token acoustic-evidence feasibility, not a direction, lexical-word recovery, ASR
improvement, or permission to deploy steering. A2 and prior terminal stages remain unchanged.

## 1. Evidence and scientific distinction

ST-LOC0 ended LOCAL_EFFECT_ONLY. ST-PROMPT-R1-A ended MARGIN_ONLY, with at most two unique
EN-confusion corrections across 24 arms. R0 ended ORACLE_CONSTRUCTION_INSUFFICIENT: native
acoustic regions passed its quality gate, but the baseline supplied too few EN-associated
hidden states for per-utterance SVD. S1 uses acoustic regions without constructing vectors
or requiring R0's decoder-group size, q_E>=.70, opposite-mass, or rank eligibility.

LAC0 ended P2_SEL_LAC_NOT_DISCRIMINATIVE. It sealed raw-logit argmax c_E/c_M from canonical
script partitions, zeroed the P2-SEL-T most-attended **0.5 s** W*, and measured
S_lex=(z_E-z_M)-(z_mask_E-z_mask_M). Strong EN-TP evidence occurred on 6/42 (needed 34);
TP-minus-FP lower80 was -0.022; recall 4/18. No gate repair advanced. S1 does not reopen it.
S1 instead seals same-prefix multilingual top-K unions, selects an **R0-predicted English**
region, and tests fixed-candidate full-vocabulary log-probability reranking against a matched
non-English off-target mask. The hard-zero transform is reused; region selection, candidate
population, evidence normalization, and decision endpoints differ. Attention alone is not
an English detector. Occlusion is a model sensitivity diagnostic, not proof of phonetic truth.

## 2. Population and privileged inputs

Exact parent ST_PROMPT_R1_PANEL: 180 positions, 80 utterances, 20 dialogues; 60 EN-confusion,
60 EN-correct, 60 ZH-correct. Preserve query order, t, absolute query=4+t-1, historical
B0M_L16 content prefixes, audio, and P2-RJ acceptable-token sets/fixed competitors. The
S1 panel fingerprints its parent and adds only verified source provenance. Original audio
identity uses size-prefixed first-64KiB hashes; R0 uses full-file SHA256. Both definitions
are recorded and verified, not incorrectly compared as identical hash algorithms.

Runtime preparation projects UID, audio metadata and **content[:t]**, t/query and model
metadata. Dialogue is a reporting key only. No per-row strata, target sets, competitors,
alignment, gold timing, error counts or goodness scores enter any provider. Full baseline
sequences are archived provenance, not permission to feed future tokens. The historical
population excludes structurally unsupported queries; publish original eligibility/omission
limitations. S1 does not add replacement queries for deleted English words. First-token
availability at these fixed queries cannot demonstrate recovery of an omitted full word.

## 3. Original-audio candidate pass

Frozen Whisper-large-v3 at `/mnt/data/tungnx/whisper-large-v3`, bf16/eager, batch1, eval,
inference_mode, unchanged weights/tokenizer/feature extractor. Task transcribe, no timestamps,
greedy historical suppression and cap200. M prompt [50258,50260,50360,50364]; E prompt
[50258,50259,50360,50364]. Each branch owns a cold cache: consume prompt, then one identical
B0-generated token per step, to the exact query. No separately generated E transcript.

AUTO is AVAILABLE: installed WhisperGenerationMixin.detect_language uses original audio's
encoder and one SOT query, masks to all generation_config.lang_to_id tokens, and selects the
lowest-ID argmax. Detect once per original utterance; freeze its language token. Construct
[50258,detected_language_ID,50360,50364] and replay the **same B0 prefix**. Explicit language=None,
task=transcribe, return_timestamps=False, forced_decoder_ids=None avoids legacy implicit forced
IDs. The detected language can be neither EN nor ZH and may duplicate M/E; do not restrict it.
AUTO here means detected-language same-prefix scoring, not an AUTO-generated trajectory.
Audit native prompt construction with a stub detector before a job. Incompatible installed
semantics blocks execution; never substitute SOT-only scoring or a different task.

Require M raw-logit bitwise identity to sealed historical NONE on all180. Reuse old M logits
only after that audit. No verified full E/AUTO distributions are presumed available; replay
these branches. Fresh M attention must follow the matched cached path (R0 attention follows
PATH5 prefixes/full replay and is not assumed compatible). No encoder or KV cache from a
counterfactual may be used to generate candidates.

For K=5,20, take finite allowed processed-logit TopK separately in M/E/AUTO. Sort descending
value, ascending token ID ties; union by ID, stored ascending IDs with origin bits and source
ranks. Size <=3K. No script-based restriction, no manual gold insertion. EOS is a legal
candidate and tagged TERMINATE, never an acceptable lexical hit. Any other generation-valid
control/timestamp token is retained as OTHER_ACTION, never a lexical reference hit; do not add
processors absent from the historical cached path or silently filter these candidates. Use exactly
cached._processed(raw,t,suppress,begin): always-suppressed IDs=-inf; begin suppression at t=0
only. Temperature1; full allowed-vocabulary float32 log_softmax. Archive lossless raw bf16,
processed float32 distributions, legal IDs and input hashes. Full distributions allow identical
scoring later, including candidates first proposed by E/AUTO. Push candidate seal before masks.

## 4. Predicted target region and mapping

Reuse only R0 **primary** native100-way LID windows/track_heard_intervals by UID and verified
full audio/model/config/source hashes. All80 sources and their arrays have been byte-verified.
R0 windows=1s, stride=.5s plus right anchor; crop>=.5s, RMS>=1e-4, EN+ZH mass>=.5,
conditional EN or ZH mass>=.8, sample vote>=.75 including uncertain votes in the denominator.
No gap filling. No oracle/MMS-FA/script-region/vector arrays may be imported. S1 adds no LID call.

Decode sees heard=min(resampled samples,480000), at16kHz. Clip all intervals to this horizon;
never mask later audio. Use the ten frozen alignment heads in config, arithmetic mean at the
current M query; require original attention mass on real heard frames >=.5. Renormalize the
heard-frame mean in float64. Each320-sample encoder frame distributes its mass uniformly across
its actual heard samples (short last frame denominator=its actual samples). Interval attention
integral is the sum of this density on the half-open sample interval.

Enumerate maximal EN intervals of at least4000 samples. In each choose length=min(interval
length,32000); enumerate interval-left+n*320 starts fitting inside plus unique right-anchored
last start. Pick the interval/crop with greatest query-attention integral, earliest start/end
ties. Require integral>=.10 and RMS>=1e-4. Never shorten a low-energy winner to make it qualify.
This gives a .25–2s predicted-English target, rather than fixed attended .5s W*. The thresholds
are conservative reference-free association/energy guards, not fitted region-quality claims.
No candidate's identity or probability participates in region choice.

Absent EN, low heard mass, low association or low energy are explicit abstentions. No R0
hard query-to-EN group is required. The R0 region can cover English omitted from the baseline,
but absent decoder queries cannot be conjured by the mapping.

## 5. One target mask, one matched off-target mask

Hard-zero finite float32 waveform copy at [a,b); positive zeros, no taper. Everything outside
is byte-identical, including unheard samples. Reload no WAV intermediate. Recompute whole
feature extractor/attention mask and acoustic encoder; global feature normalization can change
features beyond the masked interval, and that is part of the frozen waveform counterfactual.
Replay a cold M decoder cache through the identical prefix, attention=True, no steering.

Off-target length equals target length exactly. Starts=0+n*320 plus unique right-anchored last
start within heard audio. Require disjoint target, predicted EN sample fraction<=.10,
query-attention mass<=min(.10,target_mass/2), RMS>=1e-4, and sum-square energy ratio to target
in [.5,2]. Rank by lower attention integral, then smaller absolute log energy ratio, then
start. Never inspect effects for placement. If no match: NO_MATCHED_OFFTARGET. Target and
matched-control scores jointly abstain to original M ranking for advancement/unconditional
endpoints. Still archive target-only diagnostics if computed. No fallback crop formulation,
mask length sweep, energy scaling, taper comparison, oracle substitution, or random audio.
Original/counterfactual feature, waveform, region, and cache lineage hashes are mandatory.

## 6. Fixed-candidate scoring and controls

For every c in the SEALED C_K, let l0=logp_M_original(c), lm=logp_M_mask(c).
Support=l0-lm; EN_REGION score=l0+Support=2*l0-lm; lambda=1. No candidate renormalization,
clipping, null fitting or outcome-dependent scaling. Compute log_softmax float32 over the full
allowed vocabulary, then subtraction/ranking float64, lower token ID score ties. A constant
full-softmax normalizer shift does not change the within-union contrast ranking.

Scorers on exactly the same union: M_ORIGINAL; E_ORIGINAL; AUTO_ORIGINAL (their own original
prompt distributions, never a mixture of incompatible probabilities); EN_REGION; OFF_TARGET
(same formula); SHUFFLED_SUPPORT; LAC_UNION; NO_CONTRAST=M_ORIGINAL. Shuffle a single EN support
vector among union IDs with PCG64 seeded by first16 hex digits of SHA256 of
`s1-shuffle-v1|UID|t|K`; do not seek a favorable seed. It tests candidate specificity and costs
no forward. E/AUTO comparators are descriptive, cannot replace EN_REGION after outcomes.

All180 historical LAC masked arrays/prefixes have been inventoried and hashed. Reuse them only
with exact original M raw logits, processing, model, input and W* provenance. LAC_UNION applies
the NEW fixed score to the NEW sealed union with the OLD masked distribution. It is not the
historical best-E/best-M S_lex. Separately report original LAC0 quantities and terminal label
unchanged. If exact reuse audit fails, INVALID; no convenient rerun/replacement comparator.
No-match rows keep original M ranking in unconditional metrics; distinguish candidate coverage,
region coverage and paired-control coverage explicitly. Every stratum uses identical inference.

## 7. Evaluator endpoints and Gate H

Only after full seal and PRIMARY audit, load historical acceptable reference-token sets.
Hit=highest-ranked union token belongs to acceptable set. Reference rank is best rank of an
acceptable union token; MRR=1/rank, or0 if none. Natural inclusion alone defines candidate
accessibility; this is evaluator conditioning, never an inference-time oracle selector.
Report M/E/AUTO Top5/20 and union first-token recall, incremental E beyond M, incremental AUTO
beyond M+E, union-minus-M, source ranks, union sizes, dialogues and all correct-state ambiguity.
Report pooled /60 and dialogue-macro metrics separately. Missing-query English words are
structurally untested; quantify using historical evaluator metadata only, without adding queries.

H passes iff complete valid original outputs180 AND union20 hit count>=12/60 across>=6 dialogues
AND union20-minus-M20 hits>=5 across>=3 dialogues. These modest feasibility counts require a
nontrivial 20% confusion headroom and breadth beyond a single exposed case. EOS never counts.

## 8. Gate D, controls and protection

Primary K=20; K=5 secondary, never a replacement advancement budget. Paired-accessible P_K =
EN-confusion queries naturally containing an acceptable token in C_K AND having valid target
and matched off-target, plus valid historical LAC comparator. Membership is determined only
post-seal; retain every query in unconditional /60 summaries with fallback/no-hit outcomes.

D point/coverage/protection gates ALL:
- P20>=12 rows across>=6 dialogues, and >=50% of all union20-accessible confusion queries.
- EN_REGION net Hits@1 improvement vs M>=3, gross new hits in>=3 dialogues, dialogue-macro
  paired Hits@1 improvement>=.10. Net means new correct minus new incorrect, not gross flips.
- Net Hits@1 improvement vs EACH OFF_TARGET/LAC_UNION/SHUFFLED_SUPPORT>=2 on that same P20.
- Dialogue-macro paired MRR gain >=.10 vs M and >=.05 vs each of those three controls.
- On the original correct-state populations60+60 (fallback included), new corruptions vs M
  <=3 EN, <=2 ZH, total<=5. Newly wrong ASCII-Latin English-script top1 promotions on ZH<=1,
  using the unchanged canonical tokenizer/script partition, not a new language classifier.

D inference additionally requires every adjusted lower MRR gain bound vs the four controls>0
at K20. These gates demand specificity beyond removing arbitrary audio, old LAC evidence and
support permutation. E/AUTO ranking and comparison are always reported; they do not change the
frozen acoustic hypotheses. Do not claim dominance over these comparators unless measured.
If paired coverage is too small the acoustic hypothesis is unsupported, not excused by a
conditional high accuracy. Report all correct-state mass/rank/support changes and EOS promotions.

## 9. Statistical contract

Seed240924, B=10000; sample20 fixed dialogue IDs with replacement, retain paired candidate
rankings/strata/availability. For conditional statistics average per-row paired effects within
each represented eligible dialogue, then average those dialogue means, weighted by sampled
multiplicity. Never impute absent eligible dialogues as zero. Empty draws are undefined; require
>=9900 finite draws per contrast; no repeat-draw rescue. Paired pooled effects are descriptive.

Bonferroni two-sided95% simultaneous family: 2 budgets x4 EN_REGION-minus-control MRR contrasts
=8; percentile quantiles .003125/.996875, linear interpolation. Compute all eight regardless
of which looks best; K20 alone can advance. Pointwise95% bootstrap for coverage/Hit/E/A and
other endpoints is descriptive. Count gates do not require significance. No candidate/source/
mask/score selection after reference access. Conditioning on headroom is explicitly disclosed;
this is enriched, heavily exposed development evidence, not independent validation.

## 10. Exact terminal precedence

1. S1_INVALID: provenance, panel, missing required matrix, candidate-mutability, cache/model,
   numerical, leakage, comparator-integrity or audit failure. Expected region abstention is not
   invalidity. Any nonfinite legal score is invalid, not silently suppressed.
2. S1_BLOCKED_CANDIDATE_SEMANTICS: pre-run branch/prompt/legal-token semantics cannot implement
   the frozen contract; stop before outcomes. It cannot excuse a post-run integrity failure.
3. S1_CANDIDATE_HEADROOM_INSUFFICIENT: valid H fails. D is descriptive; no S2.
4. S1_ACOUSTIC_DISCRIMINATION_INSUFFICIENT: H passes but any D point/coverage/protection gate fails.
5. S1_PARTIAL_FEASIBILITY: H and ALL D point/coverage/protection gates pass, but adjusted inference
   (including finite-draw requirement) fails. Acoustic evidence remains unsupported for progression.
6. S1_READY_FOR_S2: H and ALL D gates pass, FULL independent audit PASS. Recommend only a separately
   frozen S2 direction-construction study. No automatic S2 job or ASR/steering claim.

## 11. Compute, execution and stop

Two sequential Slurm jobs max, one pending/running. Candidate allocation: <=80 encoders,
<=9477 original cached branch steps plus<=80 native language queries (less when AUTO duplicates).
Acoustic allocation: <=360 masked encoders and<=11692 prompt/token steps; group shared masks by
UID/bounds/model/preprocessing, retain query-specific lineage. Historical LAC and R0 reuse mean
zero new LAC masks/LID. Both no backward/no train/no edits/no free generation.
Historical LAC180 forwards took72s and R0 FULL300 including10387 LID windows took11.9min;
conservative forecast3–15min/job, H100/MIG, existing acl1 interpreter, hard3h/job. This is a
forecast, not a new throughput measurement. CPU preparation and synthetic tests now; future
limited reference-free cost smoke belongs to the immutable first production job. >3h estimate
requires explicit pre-outcome compute revision, never population/control reduction.

Push specification/config/panel before implementation science. Implement, test and independently
obtain PASS_TO_S1, then push resolved plan/manifest. Candidate job -> immutable candidate seal
push -> S1_AUDIT: PASS (CANDIDATES) -> acoustic job -> immutable full seal push -> S1_AUDIT: PASS
(PRIMARY) -> exposed-reference evaluation -> independently recomputed FULL audit and terminal
label. No new outcome report in this freeze. D-dev-confirm/D-test/router-calib new role/P3/
transfer forbidden. Historical failures and all prior reports stay unchanged.
