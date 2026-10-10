# R0 — predicted regions and per-utterance unique direction

Status: **PRE-OUTCOME FREEZE**, 2026-10-08. No R0 model forward, GPU experiment,
region outcome or reference-boundary analysis in this design session. Starting local and
fetched remote HEAD: `02138e5e87c7fc9bd5018f9b7feee03115efeb23`; clean tree, empty queue.
Authority for this separate supporting study: this specification, its config, panel and
firewall. User-authorized observational layers [3,8,16,24] do not revise core v6/MC
candidate steering layers, basis, controller or training. No intervention is performed.

## 1. Question, historical separation and claims

Can frozen Whisper predict acoustic EN/ZH membership and construct an identifiable,
stable, informative per-input-utterance unique vector from its own baseline decoder
states? This is nonstreaming, with a preliminary analysis pass and no parameter training.
"Zero-shot" refers to absence of gold membership/transcripts in the provider, not to
an absence of historical development-informed design choices. No PIER, correction,
causal lexical-power or ASR-improvement claim is possible here.

Three distinct constructions must remain named separately:

| Construction | Membership / numerics | R0 role |
|---|---|---|
| A5 `v_unique` | D-construct pooled reference-prefix states. Actual decoder B: positions preceding first embedded span, without explicit Mandarin-correct filtering. Requested32; production eigensystem64/SVD64 with winner among first32. Rank-stability FAIL preserved, later diagnostic-only amendment. | Historical motivation, never a fallback |
| D1/ST-LOC0 `v_unq` | Annotated EN-correct vs ZH-correct B states, whole-dialogue-excluded, uncentered float64, exact32 subspaces, guarded principal-pair winner | Tested numerical concepts; not zero-shot calibration |
| R0 `v_unq_pred(x,l)` | Only this utterance's predicted acoustic membership, own baseline-prefix states, guarded small rank | Sole new primary method |

A5 prose and actual code disagree on B/rank; use the recovered implementation described
in P2-DIR §2, not the idealized prose. Do not repair historical artifacts. ST-LOC0 remains
`ST_LOC0_LOCAL_EFFECT_ONLY`, no LOC1 authorized. Local-LID selectivity failures, P2-SEQ
damage and A2 termination recovery remain unchanged; no A2 state enters R0.

## 2. Population and immutable inputs

`R0_PANEL.json`: exact historical FULL300, canonical utterance-ID order, 300 IDs,
20 dialogues ×15, D-dev-select only. Parent: `P2_A2_MECH0_PANEL.json`, whose source is
PATH5's canonical FULL300. Audio bytes were independently full-hashed. The panel carries
only ID/dialogue/source/baseline provenance, never reference surfaces, timings or strata.
Config pins identity and ordered membership hashes; per-row canonical PATH5 `theta0`
content IDs and termination are reused exactly, with full source-file fingerprints.
No new decode is needed to obtain this generated forced-ZH hypothesis. ID/dialogue/role
columns of the current role parquet were independently checked for all300; its hash is
pinned in the panel. All11 current model/tokenizer/preprocessing files match historical
PATH5 pins, including a fresh full weight-file hash, without loading model weights.

Whisper-large-v3, eval, all parameters frozen, bf16/eager, batch1, encoder detached;
forced-ZH `[50258,50260,50360,50364]`; paired EN comparator changes only50260→50259.
Historical greedy suppression/begin-suppression, temperature1 logits, no sampling/beam,
200-decision cap, EOS50257. No cross-utterance conditioning. `load_audio` mono averaging,
float32 soundfile, scipy resampling if needed; normal feature extraction, no denoising,
amplitude normalization or inferred future content.

**Audio horizon:** original waveform grid covers all resampled samples. 27/300 originals
exceed30s (max59.478s); historical baseline encoder consumes first min(N,480000) samples.
Keep those rows. Seal full N and baseline-heard N independently. Mapping/constructibility
and primary oracle comparisons use the heard portion only. Full-waveform region quality
and English beyond the baseline horizon are separately reported. No decoder state is
created for unconsumed audio. READY is limited to this baseline horizon; it does not
establish long-form full-utterance ASR feasibility.

## 3. Primary acoustic predictor — fixed raw native LID

Use `experiments.inference_cf_p0_r2.native_lid(bundle,waveform,language_ids)` unchanged:
single SOT readout, softmax across exactly the100 sorted native language IDs, float32
logits/softmax. Freeze complete map/hash; EN50259/ZH50260. For N>0, W=16000, stride8000:
if N<W one crop [0,N); otherwise starts0,8000,... through last full regular crop, then
append N−W iff absent. Sort/deduplicate bounds. Every crop is inside original audio;
native `_features` independently pads it to30s. No donor or query-selected crop.

Save the entire100-way distribution. Let Q=P_E+P_M, pE_pair=P_E/(Q+1e-12) and
pM_pair=P_M/(Q+1e-12). Q need not equal1. Label:

* UNCERTAIN if duration<8000 samples, RMS<1e-4, or Q<0.50;
* EN if otherwise pE_pair>=0.80;
* ZH if otherwise pM_pair>=0.80;
* UNCERTAIN otherwise. Retain all simultaneously applicable abstention reasons.

Nonfinite/malformed probability vectors or sum error>1e-6 are engineering INVALID,
not a low-confidence abstention. Keep Q/odds and entropy for diagnosis. No silent use
of LocalSupport E or its positive threshold. Silence's historical ell_null≈+2.81 was
designed for a one-sided repair gate, not symmetric segmentation; null correction is
excluded from the primary classifier. Fixed RMS check is an abstention check, not a VAD.

At each sample compute the fraction of all covering windows labeled EN and ZH,
including UNCERTAIN windows in the denominator. Assign EN if EN fraction>=.75, ZH if
ZH fraction>=.75, otherwise U. Merge maximal adjacent identical sample labels; preserve
U, no bridge/gap-fill/span removal. Implementation may use interval sweeps rather than
materialize sample arrays. Encoder-frame masks wE,wM are exact sample-duration fractions
inside each320-sample frame, with partial last-frame denominator its actual duration.
This avoids pretending overlapping windows are independent evidence.

## 4. Baseline state extraction and common mapping

One forced-ZH **causal full replay** of sealed prompt+generated content, use_cache=False,
captures all four DG-02 states and shipped alignment-head weights simultaneously. Reuse
the encoder once. Recorder reconstructs r=q+u_source in native bf16 before detached f32
storage. Capture final-FFN-LN input for a synthetic/site audit; not post-FFN hidden states.
No model modification, steering hook, gradient, optimizer or gold teacher forcing.

For content decision t, query index=len(prompt)+t−1. Eligible:1<=t<T, target is a
non-special generation-valid token and baseline prior bytes are complete UTF8. Exclude
t=0 forced-prefix query, EOS/control/timestamp targets, invalid byte-prefix queries.
Retain exclusions. EOS termination is separate metadata; cap is not EOS. Never remove
an utterance for empty/short/capped generation: its vector can abstain.

Full replay causally masks decoder self-attention; no future baseline tokens influence
the queried state. This differs numerically from cached decoding. Audit baseline token
source exactly; do not require full-replay argmax to reproduce every cached near-tie.
Record disagreements descriptively; repeat identical full replay must be bit-identical.
No switching of numerical execution path to improve agreement.

Frozen alignment heads (same for every extraction layer):
`[(7,0),(10,17),(12,18),(13,12),(16,1),(17,14),(19,11),(21,4),(24,1),(25,6)]`.
Use their arithmetic mean current-query attention, with no DTW, attention-head search,
temporal filtering or reference-to-hypothesis alignment. Keep raw mass over actual heard
frames. If raw real mass<.50, query is U. Otherwise truncate to heard frames and normalize
that mean to sum1 in float64. Nonfinite/negative/zero malformed attention => INVALID.

`qE(t)=sum_f a(t,f) wE(f)`; qM likewise. EN iff qE>=.70 and qM<=.20; ZH iff
qM>=.70 and qE<=.20; else U. This identical assignment indexes all four layers. Each
group state appears once; no duplicate/soft-weighted states. Earliest argmax real frame
defines temporal-diversity bin floor(argmax_frame*320/3200), shared across layers. Bin
counts are an information proxy, not proof of independent observations; the separate
centered participation-ratio guard screens redundant states.

A gold EN acoustic span may have no corresponding generated query: omission, broad or
wrong attention, first-token exclusion, and encoder truncation are separate reasons.
Oracle membership cannot invent queries or hidden states for it.

## 5. Per-utterance small-rank numerical contract

For each layer independently, H_E/H_M consist only of eligible assigned rows in query
order. Fit CPU float64, raw uncentered moments C=H.T H/n. No state normalization,
LayerNorm transform, centering or cross-utterance pooling in the construction.
Efficient thin SVD of H is mathematically equivalent to C eigendecomposition: eigenvalues
singular_value²/n and eigenvectors right singular vectors. Pad missing eigenvalues with0.
Pin NumPy/BLAS/thread count; compare against explicit C on synthetic CPU fixtures.

Choose the **largest** r from[8,4,2] supported in BOTH groups by all these pre-winner
conditions (record evidence for every candidate):

1. n>=2r+2 and occupied200ms attention bins>=r+1.
2. Numerical rank from singular values>1e-4 times largest is>=r.
3. Effective information: centered diagnostic participation ratio
   `(sum lambda_centered)^2/sum(lambda_centered^2)`>=r. Zero variance fails.
   Centering is used ONLY for this redundancy guard, not C/U/SVD/sign means.
4. lambda_r>1e-5 lambda_1 and (lambda_r−lambda_(r+1))>.05 lambda_r.

If none supports, INVALID_VECTOR:NO_SUPPORTED_RANK; preserve distinct count/bin/rank/
participation/eigen-floor/eigen-gap reasons. Rank is not selected by oracle cosine,
outcomes or a winner's later behavior. Once chosen, **no lower-rank retry**.

For chosen r, `U_E.T U_M=P diag(sigma) Q.T`, a=U_E P;
score_i=(a_i.T C_E a_i)*(1−sigma_i²); winner=lowest index on exact argmax ties.
Guard largest score>1e-6 max(E_E), winner−runnerup>.05 winner, selected sigma isolated
by>1e-4 from all others. Raw sigma outside[0,1] by>1e-8 fails; otherwise clip roundoff.
Orthonormality/principal-pair max error<=1e-8. Any materially negative C eigenvalue below
−1e-10 lambda1 fails. Mean contrast norm>1e-10; |cos(a_winner,meanE−meanM)|>.05.
Flip toward predicted meanE−meanM; f64 normalize, f32 serialize unit error<=2e-6.
All failure reasons are explicit, never an alternate vector. Same rank policy and guards
apply independently to predicted, oracle, shuffled and perturbed memberships; ranks may
differ, reported. Do not force oracle rank into prediction.

Seal states, group IDs/bins, counts, centered diagnostic spectra, C spectra, numerical
ranks, every candidate-rank guard, U/P/Q/sigma, energies, winner/index/gap, orientation,
vector and tensor/source hashes. Invalid vectors remain in construction denominators.

## 6. Reference-free controls and secondary predictor

Twenty same-utterance controls: on the **heard sample mask**, circularly shift the complete
EN/ZH/U label track; preserves exact sample counts and cyclic region structure. Key
`R0-shuffle-v1|240924|<uid>`, SHA256 first16 hex integer seeds PCG64; choose20 distinct
integer offsets without replacement from[1600,N_heard−1600], sorted before use. If fewer
than20 legal offsets, control-unavailable, no substitute. This is region-label shuffling,
not state-value shuffling. Re-map/fill guards without gold. Save seeds/offsets/control
validity/vectors before seal. Circular wrap is intentional; do not claim natural timing.

Two fixed boundary controls, heard mask only: ERODE_100MS shrinks every maximal E/M
interval by1600 samples at each end (empty→U); DILATE_100MS expands by1600, clips toheard
bounds, conflicting expanded E/M overlap→U. No moving hidden states or retesting LID.
Evaluate independently under the same rank rules, retain failures.

Numerical repeats: refit every primary vector twice same environment; status/rank/index
and vector hash identical; independent alternative thin-SVD/explicit-moment synthetic
agreement maxabs<=1e-6 after same mean orientation. Numerical nonreproducibility =>R0_INVALID.

Baseline-script secondary is **diagnostic only**: canonical tokenizer partition labels
eligible generated tokens EN/M/U, never gold; accumulate a_t(f) votes per class over
eligible queries, normalize by all token votes per frame. Fraction>=.75 determines E/M,
else U (zero vote U). Refit small-rank vectors descriptively, never replace primary, add
candidate layers, or rescue a failed label. The baseline can omit English. After seal,
separately count reference EN units deleted by canonical ref/hyp alignment and acoustic
predictor coverage of their CTC intervals; lexical alignment serves evaluation only.

One paired forced-EN full replay of the same baseline prefix captures all layers for
`v_prompt` descriptive comparison: historical `core_p1.direction` per eligible query;
also apply it to mean querywise delta. Tiny/nonfinite historical fallback unchanged.
Report availability/norm and signed cosine with primary directions where valid; no
steering, selector, candidate substitution or agreement threshold based on v_prompt.

## 7. Oracle source and post-seal construction

Evaluator-only `candidates_existing_ctc.parquet`: complete300/300 structural source
coverage, 16766 records, valid record in each utterance. MMS-FA reference-unit Viterbi
timing, short_wav_sample0,16k sample coordinates, recorded emission-to-second conversion
and normalization hash. This is an approximate **reference-derived timing oracle**, NOT
manual ground-truth acoustic segmentation. Precision/recall and vector agreement are
against this proxy; disclose timing uncertainty and unidentified aligner weight revision.

Current file hash matches historical P0-R2's frozen source hash `sha256:af4f4127…`.
Sidecar's old `sha256` field differs; retain its bytes as provenance but **do not treat
that field as current byte identity**. The current file/P0-R2 manifest hashes and stored
coordinate/provider metadata are authoritative for reuse. No fresh CTC run or silently
substituted aligner. Pre-run auditor checks these exact sources. Read-only design
inspection used schema/IDs/validity/provider metadata, not language memberships, surfaces
or boundary values. It did not calculate oracle constructibility or R0 agreement.

After pushed primary seal and `R0_AUDIT: PASS (PRIMARY)`, evaluator filters strictly to
the300 IDs, reads valid finite in-bounds sample intervals and reference-unit language/
surface consistency under historical normalization. Duplicates must agree; malformed
structural source is R0_INVALID. Invalid/uncovered intervals remainU, conflicting EN/M
overlaps U, no inferred speech filling. Build EN/M sample unions and frame masks.
Use exactly sealed baseline queries/states/attention/mapping/guards; replace membership
ONLY. No gold transcript replay or new encoder/model pass. Unknown language/reference
units are U, not Mandarin by default. Missing source/provenance at preflight =>
`R0_DESIGN_BLOCKED:ORACLE_SOURCE`; inadequate valid vectors =>oracle-insufficient outcome.

## 8. Measures and quantitative gates

All300 ×4 records required; abstention is a scientific outcome, not missing data.
Report full/heard EN/M/U durations; known-support duration-weighted precision/recall,
IoU and boundary matches within200ms (one-to-one nearest matching, earliest tie);
boundary errors; omitted-English interval/query coverage; query mass/count/bin coverage;
spectra/rank/angles/gaps/sign; signed and absolute oracle cosine separately; predicted vs
shuffle cosine; shuffle vs oracle cosine; perturb validity and signed cosine; per-layer
and paired cross-layer cosine/norm statistics. No best-of-rank or best-of-shuffle scoring.

Region quality on known oracle support inside baseline-heard horizon: pooled EN precision
>=.70, EN recall>=.50, ZH precision>=.80, ZH recall>=.50, classified coverage>=.50.
Need known EN and ZH support in>=60 rows spanning>=12 dialogues. Undefined ratio fails;
U is a missed class for recall, excluded from prediction-positive precision denominator.
Predictions on oracle-U scored separately, not assigned a false correctness label.

Per-layer nested gates:

* O: oracle-valid>=60 utterances, >=12 dialogues.
* P within O: global region quality passes; paired primary/oracle-valid>=30, >=10
  dialogues; paired/oracle-valid>=.50.
* S within P: among ALL primary-valid vectors, both perturbations valid on>=.80;
  >=.80 have both valid and minimum signed cosine>=.80; median of minimum cosine over
  both-valid rows>=.90. Invalid perturbation fails that row, no survivor-only pass.
* I within S: signal-evaluable paired rows (>=10 valid shuffled vectors/20 each)>=.80
  of paired; still>=30 rows/>=10 dialogues. Macro signed pred–oracle mean>=.50 and
  adjusted lower>.25; mean advantage over the **mean** valid shuffle–oracle signed
  cosine>=.20 and adjusted lower>0. Report all20 control failures, no zero imputation.

Paired dialogue-cluster bootstrap: B10000, seed240924, sorted20 IDs, shared draws across
layers/endpoints. Compute per-dialogue mean then mean represented sampled dialogues;
empty strata draws undefined. Require>=9900 usable draws/endpoint, else signal gate
fails as insufficient evidence, not engineering INVALID. No vector refit in bootstrap.
Bonferroni family8=4layers×2 signed signal endpoints, two-sided adjusted quantiles
.003125/.996875. Other rates/cosines/durations get descriptive pointwise95% intervals.
No significance claim about utterance independence or ASR improvement.

## 9. Terminal precedence — deterministic, no post-outcome revision

1. **R0_INVALID:** panel/model/site/state/attention/hash/firewall/integrity/audit failure,
   incomplete300×4 records or nonreproducible numerics. Halt.
2. **R0_ORACLE_CONSTRUCTION_INSUFFICIENT:** no layer passes O.
3. **R0_PREDICTED_REGION_INSUFFICIENT:** O exists, no O-layer passes P.
4. **R0_VECTOR_UNSTABLE:** P exists, no P-layer passes S.
5. **R0_REGION_SIGNAL_INSUFFICIENT:** S exists, no S-layer passes I.
6. **R0_PARTIAL_FEASIBILITY:** I exists but fewer than2 layers pass I, or neither16 nor24
   passes I. Mechanistic evidence only; NO automatic progression.
7. **R0_READY_FOR_R1:** >=2 layers pass all nested gates, including16 or24, and full
   independent audit PASS. Recommend a separately frozen R1 only; no experiment launch,
   steering selection, dose/location optimization or ASR claim authorized by R0 itself.

Keep all successful layers, no post-outcome single-layer winner tuning. Gold proxy
concentration/timing limitations accompany either positive label. No fallback predictor,
vector, rank or additional experiments after failure.

## 10. Execution, compute and audits

Design CPU preflight independently hashed300 audio and baseline sources and read audio
geometry: 5269.3300625s original audio,10387 deterministic LID windows. No new R0 LID,
baseline states or vectors computed. Historical R2 (~11855 real LID calls plus controls,
24min) and ST-LOC0 passive80-row extraction71s support a provisional15–40min allocation
estimate. It is an estimate, not a measured R0 throughput result.

Future one sbatch H100/MIG allocation, max1 pending/running, hard3h. All four layers in
one forced-ZH replay; EN comparator one additional replay, no generation unless baseline
source proves unusable (then STOP for design revision). LID exact input/model/grid hashes
required for cache reuse; no assumed reuse of current-query R2/T crops. Thin-SVD CPU
fit/control jobs need memory bounded per row, single BLAS thread, no1280² eigensolver
per shuffle. Future first canonical-row smoke inside the allocation checks only shapes,
finite outputs/repeatability and timing; no references or design changes. Conservative
full300 cost projection>3h =>STOP for explicit compute revision, never reduce population.

Required future gates: `PASS_TO_R0` before production; output seal+push verified against
remote; independent `R0_AUDIT: PASS (PRIMARY)` before oracle; post-oracle independent
`R0_AUDIT: PASS (FULL)`. Auditor does not import primary decision code, independently
recomputes region masks/mapping/rank/score/orientation/bootstrap/label from sealed arrays.
Every run uses provenance/specfreeze manifest: resolved config/source/model/environment/
git/split/output hashes, trainable/edit counters all zero, timing/VRAM/LID/cache counts.
All historical files remain unchanged. Confirmation/test/P3/transfer forbidden.
