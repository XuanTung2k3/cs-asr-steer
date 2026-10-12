# ABTS-0 — Acoustic-Branch Test-Time Lexical Steering

**Design proposal only.** Base: `research/steering-forensics`, `5c504c4e0754d3a8b71bf8b66f4120c88967956f`. No execution authorized here. Separate hypothesis; historical contracts/verdicts remain authoritative for their experiments. No offline training, entropy objective, layer/dose search, or controller.

## Evidence and question

Established retrospective evidence: English-history decoding supplies aligned correct words for 43/60 historical confusions; S1's same-Mandarin-prefix union supplies acceptable first tokens for 11/60. These word/token measurements are not interchangeable. Unfiltered English-only proposals have approximately 6.8% precision in Mandarin-matrix speech. S1 masking produced no ranking corrections; TTLS-R1R established numerical repair, not broad lexical efficacy; P2-SEQ demonstrated sequence damage.

**Unproven:** predicted acoustic regions plus candidate-specific masking can distinguish spoken English from Mandarin translation. Audio dependence alone cannot establish this. K1 tests information/localization; K2 tests delivery and damage, holding selected information fixed.

## Population, provenance and baseline

Use all exposed D-dev-select FULL300: 300 utterances, 20 dialogues ×15. Reuse ordered membership from `R0_PANEL.json`, identity `sha256:ff28efc645150d122410991d3b2449edc239b26111e97fda09ddf31e9340e2cb`; file SHA256 `51255fcb9b8e9ab7c73200797a338bd07fcf71f9961cce9425a7031f79d7d042`. Retain long utterances, but restrict candidate spans to the actual first-30-second heard horizon. No confirm/test/router-calib/P3/transfer access.

Pin model/tokenizer/audio/config/source hashes in a future execution manifest. Initial sources: P2-A `p2_A_r1_L16/rows` forced-ZH/forced-EN/AUTO and R0 PRIMARY heard-region intervals. Reuse only after provenance and numerical compatibility checks.

Freeze one engine: TTLS-R1R `greedy_decode`/cached `Branch`, BF16/eager/batch1, historical prompts, suppression, greedy ties, cap200 and EOS. The 59/300 historical path discrepancies cannot become exclusions: if another path supplied historical outputs, capture and seal matched-engine B0/EN/AUTO for **all300** before selection; retain old outputs descriptively. Every delivery arm uses this baseline's prefixes. R0 regions may be reused by audio hash; R0 replay attention is not substituted for cached-query attention.

## K1 — fixed reference-free candidate algorithm

1. Align each branch's **generated tokens**, using cached attention from its own history: frozen ten alignment heads, heard frames, per-frame standardization, median7, mean-head DTW, prediction-query offset−1. Group tokens by decoded character offsets. Do not call the reference-reading `align_batch` wrapper. This is nonstreaming analysis, not an online guarantee.
2. Extract forced-EN words matching `[A-Za-z]+(?:['’-][A-Za-z]+)*`. Require positive-duration heard spans. Associate each with the contiguous block of B0 Han units intersecting its interval; require their union to cover ≥0.5 of the EN-word interval and the first unit's onset to differ by ≤200ms. Reject a word already present there, ignoring case. Its target is that block's first prediction query, with complete UTF-8 prefix and content index t≥1. Encode the proposal with one leading space, without special tokens; require generation-valid pieces. Zero-duration, missing-query, EOS and t=0 cases abstain. Deleted English with no overlapping Han query remains untested.
3. Require ≥0.5 of the EN-word interval inside R0 predicted-EN regions. Invoke unchanged S1 `heard_attention`, `select_target` and `select_offtarget`: heard mass≥0.5, target association≥0.10, duration4000–32000 samples, RMS≥1e−4; matched off-target duration, EN fraction≤0.10, attention≤min(0.10,target/2), energy ratio[0.5,2]. Require the target crop to cover ≥0.5 of the word interval. Hard-zero transforms preserve outside waveform bytes.
4. Choose the earliest structurally eligible pair per utterance **before acoustic scoring**; ties: onset, EN-token index, surface string. If verification fails, abstain without trying later proposals. Retain rejected candidates and reasons for coverage accounting.
5. Let c be the EN surface and b the overlapping B0 Han surface, separately tokenized under the same B0 prefix p. Define F_a(w)=mean_j log P_M(w_j|a,p,w_<j), using processed full-vocabulary probabilities. Set A_R(w)=F_x(w)−F_MaskR(w), and

   **C=[A_target(c)−A_target(b)]−[A_off(c)−A_off(b)].**

   Accept iff A_target(c)≥log2 and C≥log2; otherwise abstain. These prospective thresholds are twofold sensitivity screens, not calibrated probabilities. Length normalization is fixed for verification; no candidate regeneration under masking.

Seal all pools, chosen queries, scores, accepted/abstained selections and hashes before references. Controls: unfiltered EN-not-in-B0 proposals; off-target-only acceptance using A_off(c)−A_off(b)≥log2; shuffled association using R0 tracks circularly shifted by floor(heard_samples/2), rebuilding localization without extra acoustic scoring. Report controls' differing coverage explicitly.

**Evaluator only:** precision means literal spoken-English word at the associated span, not a translation or merely appearing elsewhere. Report candidate recall, onset reachability, wrong-span/mistranslation rates, and Han-rendered versus omitted/wrong-English targets separately. Primary genuine Mandarin-matrix group has reference English fraction<0.2; mixed/English-dominant results are separate. Recall denominators include all genuine baseline-wrong heard English onsets, with a separate reachable denominator. Registry excludes spacing/acronym-normalization/filler artifacts under a sealed taxonomy; no runtime registry access. Existing reference timing is an evaluator proxy, not alignment ground truth; report ±200ms sensitivity.

**K1 GO (Mandarin-matrix):** ≥12 accepted selections in ≥6 dialogues; precision≥0.75; ≥6 genuine baseline-wrong recoverable words in ≥4 dialogues; reachable-onset recall≥0.20; C separates literal candidates from all nonliteral proposals, including mistranslations, with dialogue-macro AUROC≥0.65 on the pre-scoring pool. AUROC averages only dialogues containing both classes; require ≥3 such dialogues. Missing classes/coverage means insufficient evidence. Otherwise stop K2. Compare precision against controls; bootstrap intervals are descriptive. Gold evaluation authorizes only a global hashed GO, never row selection.

## K2 — candidate-conditioned delivery

Run every K1-accepted selection, including incorrect candidates, never an evaluator-filtered subset. One fresh FP32 z∈R^1280, L16 DG-02 post-cross-attention/pre-FFN, one onset pulse. Frozen weights; ratio-first R1R repair:

**r̃=(r+BF16(z))·(||r||/max(||r+BF16(z)||,epsilon)).**

Candidate loss is **L(z)=−(1/m)Σ_j log P_z(c_j|x,p,c_<j)**. Encode the EN surface with one leading space, without special tokens; reject invalid/suppressed tokenizations. All BPE pieces are scored autoregressively, never baseline future tokens. Candidate conditioning is model-generated pseudo-supervision, not gold teacher forcing.

Use two AdamW updates from z=0: lr=e*/(2√1280), wd0, betas(.9,.999), eps1e−8; e*=1.1260757575454359. Project ||z||≤e* after each update; require consumed chord≤1.02e* and norm-ratio error≤0.02. No amplitude escalation.

Loss calls need a gradient-enabled cached counterpart to `Branch.step` (which currently uses inference mode): clone pristine prefix caches, feed the onset query, edit once, then candidate pieces with graph-connected downstream KV. Detach only the pre-edit cache; never z-dependent continuations. Verify z=0 logits match the inference engine at every scored step. Free decoding rebuilds a fresh cache, applies final z once, then ordinary greedy continuation; no candidate tokens forced in STEER.

| Arm | Delivery |
|---|---|
| B0 | Matched engine; active-mask zero-vector identity also verified |
| AUTO | Matched unsteered native-language decoding |
| FORCE | Same B0 prefix; force all selected candidate BPE pieces, resume greedy |
| STEER | Optimized onset pulse; fully free continuation |
| RANDOM | One SHA256(`ABTS0|UID|t|240924`)-seeded tangent Gaussian, chord matched to STEER within2%, same onset |

## Decisions, integrity and minimum work

Primary benefit: complete genuine Mandarin-matrix words recovered at selected slots, counted once per word, not per BPE piece. Safety covers all300: correct-Mandarin/English corruptions, outside-span harm, return-to-Mandarin, EOS/deletion/cap damage. Define utility U=G−H−I: G is primary word recovery; H is all new correct→incorrect canonical reference units; I is added insertion count versus B0. Report canonical PIER/MER/ZH-CER, dialogue breadth, full300 denominators and active denominators. Teacher-forced loss/margins are diagnostic only.

Precedence: INVALID → K1_INSUFFICIENT → K2_DAMAGE → K2_NO_POWER → SIGNAL_NO_STEERING_ADVANTAGE → STEERING_FEASIBILITY. Damage: any new severe truncation/cap, net mixed-error increase, or >2 new Mandarin corruptions versus B0. Power requires ≥3 full lexical repairs in ≥3 dialogues and net lexical utility≥RANDOM+3. Steering advantage additionally requires ≥80% of FORCE's repairs, ≥2 fewer outside-candidate corruptions than FORCE, and higher net utility. Otherwise credit candidate discovery/constrained decoding only. AUTO remains the practical comparator. No label authorizes confirmation or full steering optimization.

Use paired dialogue bootstrap, seed240924/B2000, including empty dialogues; report undefined ratios and LODO sensitivity. Counts are pilot screening thresholds, not significance claims or broad safety evidence.

Invalidate for zero-vector/clean-token identity failure, gradients absent/nonfinite, wrong FFN consumption, stale caches, model changes, leakage or unresolved matched-baseline discrepancies. Require independent pre-run, K1-seal and K2-seal audits before respective reference scoring.

Dependencies: cached engine, R1R hook/episode primitives, S1 masks, R0 PRIMARY intervals, alignment DTW helpers, independent canonical scoring. Random chord solving must emulate the ratio-first kernel: P2-R's historical emulator uses the older repair and cannot be reused unchanged. No shared-kernel edits. Proposed cost: two future jobs, each≤3h; K1≤600 alignment captures, ≤600 masked encoders and1800 short candidate scorings, plus any all300 matched-baseline regeneration; K2≤300 episodes/600 backwards plus three continuation arms and controls. Allow 30–120min/job provisionally; bounded dry-run forecast must fit before execution. Main risks: missing cached attention, uncertain alignment, uninformative masking, gradient-engine identity and inadequate edit leverage. No new inference, implementation or commit accompanies this document.
