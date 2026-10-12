# ABTS-0 K1 — preflight, implementation and decision gate

**Label: `K1_BLOCKED`.** K1 needs fresh GPU forward passes that are not authorized: matched-engine forced-EN decoding with
cached alignment attention, and masked-audio rescoring. No saved artifact contains them. This is an engineering/data
gate, **not** a scientific verdict. K2 is not eligible, and nothing in K2 was implemented or run.

**Update 2026-10-12 (§10 governs):** after the final scientific preflight, the status is **`READY_FOR_K1_CAPTURE`**,
with amendments A1–A3/G1 approved, implemented and sealed. Only GPU authorization is pending; nothing has been submitted.

Scope: design `docs/research_review/ABTS0_DESIGN.md` (pinned `sha256:bf6cf5c2…`); forensic base `5c504c4e`. Data:
exposed D-dev-select FULL300 only (20 dialogues). No confirm/test/router-calib/P3/transfer access, no GPU job, no
training, no edits to historical code, results or contracts. Every change since `5c504c4e` is a new file.

## 1. Bottom line

1. **The design is coherent enough to run K1.** I found no conceptual error that blocks K1. Twelve underspecified clauses
   needed explicit interpretations (§2.3); they are frozen in code and config before any K1 output exists.
2. **The 59/300 discrepancy is a tie-breaking policy, not numerical chaos [established].** `cached_greedy` picks with
   `torch.topk`; the matched path uses `argmax`. On exact BF16 logit ties these choose different tokens. This corrects my
   forensic report's "near-tie BF16 flips / chaotic decoding" wording.
3. **Historical forced-EN outputs are on the mismatched path.** P2-A `B1` came from `cached_greedy`, so forced-EN must be
   recaptured with the matched engine. That capture is also the only way to obtain the cached attention K1 needs.
4. **K1's gates are measurable, but the reachable target is small [established on references].**
   - Mandarin-matrix (MM) utterances hold 186 genuine baseline-wrong heard English runs.
   - Only **39 runs (13 dialogues)** have a Han-rendered onset at content t ≥ 1, the only kind ABTS can address. Omissions
     (89) and wrong-English (22) generally cannot be associated, by design.
   - The upper-bound proxy for literal recoveries is **14 utterances in 11 dialogues**. So the reachable-recall gate (≥ 8 of
     39) needs at least 57% of that ceiling to survive alignment, choice, crops and acoustic acceptance.
5. **The frozen acoustic rule is weakly discriminative, so PASS rests on the structural filters [pre-run evidence,
   small n].** I expected a floor/ceiling bias in C: b starts near log-prob 0, c near −11 nats. In the relevant case it
   did **not** appear. I applied the rule at token level to S1's sealed masked log-probs at positions where B0 chose
   Han:
   - **Literal English tokens:** 5 of 8 accepted; median C = +1.83; A_target(c) ≥ ln 2 in 8/8.
   - **Other Latin tokens:** 7 of 19 accepted.
   - Accept-rate likelihood ratio ≈ 1.7; token-level precision ≈ 0.36; only 3 dialogues have both classes.

   K1's acoustic step can therefore add only modest precision. Reaching the 0.75 gate needs a structural pool (U2) that
   is already about 64% literal. Caveats: this is single-token, same-prefix and S1's population, a weak forecast only.

## 2. Phase A — preflight

### 2.1 The eight checks

| # | Check | Finding | Status |
|---|---|---|---|
| 1 | English-history vs same-prefix branch | K1 takes candidate *words* from a full forced-EN decode (forensics: 43/60 aligned words) and scores them under the B0 *prefix* (forensics: 11/60 first tokens). The design keeps these separate. Scoring c under a Han history raised a floor-effect concern; the S1 token-level preview does not support it (§4). | VERIFIED |
| 2 | Word alignment across divergent histories | Each branch is aligned with its own cached attention and associated only by audio time. On B0, applying the frozen DTW to R0's saved attention, against MMS-FA CTC onsets of 11,534 matched Han characters: median \|error\| **0.22 s**, p90 0.84 s, **45% within 200 ms**, mean signed **−0.39 s** (DTW earlier). Forced-EN alignment of *translated* output (reordered words under a monotonic DTW) cannot be measured before capture. | PARTLY VERIFIED |
| 3 | Model / tokenizer / decoder baseline | All 11 model files match the S1 pins by full SHA-256. The matched engine (`Branch.step` + passive recorder + `processed_argmax`) equals B0M on **300/300**: TTLS-R1 and A2P checked tokens and termination at run time (100 + 200), and the plan's `y_B` = B0M on 300/300. R0's baseline = B0M on 300/300. Historical `B1` (forced-EN) and standalone `B0` came from `cached_greedy` and are **not** matched. AUTO is HF `generate`; K1 recaptures matched AUTO. | VERIFIED; EN must be recaptured |
| 4 | BF16 zero-edit identity, gradients | Not exercised by K1, which makes no edit. R1R's ratio-first kernel is bitwise at z = 0 on its full-sequence Episode path (100/100), with nonzero gradient. **K2 gaps:** (a) no gradient-enabled cached decoder step; (b) `Branch` caches are inference-mode tensors that autograd refuses, so they must be cloned outside inference mode; (c) z = 0 identity must be re-proven on that new path; (d) no ratio-first RANDOM chord emulator (P2-R's uses the old kernel). | K1 n/a; K2 GAP |
| 5 | Candidate likelihood and BPE | c is encoded as `' ' + surface`; b is re-tokenized canonically (not B0's generated pieces) for symmetry. F is the mean processed log-prob over pieces. Pieces must be < EOS and unsuppressed. Forced-EN capitalization is kept, so `' School'` and `' school'` have different priors; this mostly cancels inside A. | IMPLEMENTED |
| 6 | First-token and deletion inaccessibility | t = 0 abstains. English with no overlapping B0 Han (omissions) is untested. MM genuine heard runs by B0 onset rendering: Han 43 (23%), omitted 89 (48%), wrong-English 22 (12%), correct onset with later error 28 (15%). | VERIFIED limitation |
| 7 | Leakage, exposure, reproducibility | Runner imports no reference code; a static test forbids it, and GPU/CPU stages log opened paths against a deny-list. Evaluator is a separate script that refuses to run without the output seal. CTC parquet is read with a FULL300 row filter. The plan seals 300 audio file hashes and waveform hashes, R0 rows against R0's output seal, model files and source hashes. Raw attention goes to a content-addressed archive on `/mnt/data`, not `/home`. | VERIFIED |
| 8 | K1 gates measurable with existing data | Reachable genuine MM runs: 39 in 13 dialogues (≥ 12 / ≥ 6 needed), so K1 is not INSUFFICIENT by construction. AUROC needs ≥ 3 dialogues with both classes, which is likely but unknown. **The gates cannot be computed from saved data:** matched forced-EN words, their timing and masked scores do not exist. | MEASURABLE only after GPU |

### 2.1a OPEN DECISIONs raised by the alignment diagnostic (human, before any K1 output)

These are pre-outcome measurements on B0 and references only; no K1 candidate was seen. I did not change frozen values.

1. **Onset tolerance (design step 2; `ONSET_TOL = 3200`).** Typical DTW onset error (median 0.22 s) exceeds 200 ms. Two
   branches aligned independently will often miss the tolerance on genuine pairs, so expect many `ONSET_MISMATCH`
   abstentions and lower recall. Options: keep it as designed (conservative), or widen to about 400 ms before the run.
2. **Evaluator literal criterion (I12; primary = overlap at tolerance 0).** DTW intervals sit 0.39 s earlier than CTC on
   average. Short literal words can then fail to overlap and be labelled non-literal, which biases precision down. The
   ±200 ms variant is already computed. Recommendation: before evaluation, declare the ±200 ms variant (or an
   offset-corrected one) primary, or report both with the tolerance-0 value labelled conservative.

### 2.2 The 59/300 forced-ZH discrepancy

- **Facts.**
  - `cached_greedy` (P2 standalone `B0`, and also P2 `B1`) and the matched path both use `Branch.step` with
    `attention=True`. The recorder is read-only, and B0M's steered logits equal its unhooked branch bitwise (P2 audit).
  - The engines differ only in the final choice: `torch.topk(x, 2).indices[0]` vs `torch.argmax(x)`.
  - At all **59/59** first divergences, B0M holds the *lower* token ID. Under symmetric numerical noise this would happen
    with p = 2⁻⁵⁹.
  - On this torch (2.10.0) the CPU `topk` returns the higher index for exact ties in **11/11** synthetic BF16 ties;
    `argmax` returns the lowest.
  - 6 of the 59 divergences are at content t = 0. The median first divergence is t = 11, and 45/59 then differ in length.
- **Conclusion [established].** Exact BF16 top-1 ties are common: at least 59/300 utterances hit one that changes the
  transcript. Two engines with bitwise-identical logits still give different transcripts because they break ties
  differently. Bitwise logit proof at each divergence would need a GPU replay; it is not needed for the decision.
- **Matched-engine recommendation.** Use `episodic_tta.forced_decode` ≡ `ttls.greedy_decode` (no hook) ≡ B0M (lowest-ID
  ties) for every causal comparison. Never mix it with `cached_greedy`. K1's `matched_decode` is that loop, also keeping
  attention and per-step logits hashes. Its equality with `forced_decode` is tested.
- **K2 implication.** Ties are fragile: any perturbation of one BF16 ulp can flip them. That makes the RANDOM matched-chord
  arm and the zero-vector identity check necessary, not optional. Historical outputs were not altered.

### 2.3 Interpretations frozen before outputs (I1–I12)

| ID | Design clause | Frozen interpretation |
|---|---|---|
| I1 | "freeze one engine" | `matched_decode` = `forced_decode` numerics; `cached_greedy` excluded |
| I2 | Per-frame standardization | Over the available prediction rows (rows T+1 for EOS). `align_batch` also standardizes over prompt rows, which a cached decode does not produce. |
| I3 | Capped decodes | Last token untimed (no EOS row) |
| I4 | "Han units" / words | One unit per Han character (`is_han`); Latin words by the design regex over the decoded byte stream. A unit's interval spans its covering tokens. |
| I5 | "contiguous block" | Han units with positive overlap must be consecutive among B0 content units; otherwise `NONCONTIGUOUS_BLOCK` |
| I6 | "Reject a word already present there" | A casefold-equal B0 Latin word whose interval intersects the word ±200 ms (or is untimed) |
| I7 | Target query | First token of the block's first character. `MID_CHARACTER_ONSET` if that token starts before the character. t ≥ 1. |
| I8 | b tokenization | B0 decoded text from the first to the last block character, tokenized canonically without a leading space |
| I9 | "structurally eligible" | Steps 2–3, including R0 ≥ 0.5 and the unchanged S1 crops on B0's cached attention at query t. Step 4 picks before any scoring. |
| I10 | Scoring replay | Fresh cold `Branch` per (condition, word), one token per step. The query logits must hash-equal the capture logits (x condition), or the row is INVALID. |
| I11 | Controls | U0 = forced-EN words absent from B0; U1 = associated; U2 = all eligible; off-target-only rule; shuffled R0 track (shift ⌊heard/2⌋), steps 3–4 rebuilt with no scoring |
| I12 | Evaluator | Literal = canonical-unit equality plus CTC-interval overlap (±200 ms sensitivity). Groups by canonical EN fraction. Artifact taxonomy: frozen filler list, adjacent single letters, boundary join. Reachable = genuine run with a Han-substituted onset, B0 index > 0, heard. Precedence: BLOCKED > INSUFFICIENT (reachable < 12, or < 6 dialogues, or < 3 AUROC dialogues) > FAIL > PASS. |

## 3. Phase B — what was implemented

| File | Content |
|---|---|
| `configs/inference_cf/abts0_k1.json` | Frozen K1 parameters, pins, evaluator gates and labels |
| `src/csasr/inference_cf/abts0_k1.py` | Reference-free core: `matched_decode`, `score_word`, `token_intervals` (frozen `dtw` / `token_frame_spans`), `content_units`, `associate`, `region_check` (unchanged S1 `heard_attention` / `select_target` / `select_offtarget`), `choose`, `acoustic_rule`, `structural_pools`, `shift_track` |
| `experiments/abts0_k1.py` | `prepare` / `forecast` (CPU), `capture` (GPU), `select` (CPU, seal before scoring), `score` (GPU), `seal` (CPU). Path audit and deny-list. Archive objects content-addressed under `/mnt/data/.../archives/abts0_k1`. |
| `experiments/abts0_k1_evaluate.py` | Reference side only. `registry` (pre-run denominators and ceiling from historical artifacts) and `evaluate` (requires the output seal) |
| `slurm/abts0_k1.sbatch` | Two phases (`capture`, `score`), 1.5 h each. **Not submitted.** |
| `tests/test_abts0_k1.py` | **24 CPU tests, all passing** (final run at the commit below). Cover config–constant agreement; UTF-8/unit spans; DTW EOS/cap rows; every association status incl. exact 200 ms tolerance and mid-character onset; R0 fraction and circular shift; choice order; accept-rule thresholds and controls. A random eager tiny Whisper shows `matched_decode` ≡ `forced_decode` and that `score_word` replays capture logits bitwise. Plus an end-to-end `structural_pools` run through the unchanged S1 crops, a runner reference firewall, evaluator labelling and AUROC/decision precedence. |

**Engineering notes.**
- Pure-Python DTW costs about 2.6 s per typical branch and about 19 s per long capped branch. Selection therefore runs on
  CPU between the two GPU allocations, never inside one. It uses a process pool when POSIX semaphores are available
  (this host's 64 MB `/dev/shm` is full of other users' files), otherwise the identical serial path.
- Neighbouring regression: `test_s1_impl` passed. `test_r0_impl`, `test_ttls_r1r`, `test_inference_cf_cached` and
  `test_dg02_site` were stopped unfinished, because shared-NFS I/O stalled them for more than 45 min. No pre-existing
  file was modified (`git diff --diff-filter=M 5c504c4e` is empty). `test_r0_impl` also needs `PYTHONPATH=src`: the
  installed `csasr` resolves to another checkout. That is pre-existing.

## 4. Pre-run evidence (CPU, saved artifacts; no K1 output exists)

`results/inference_cf/abts0_k1/registry_prerun.json` comes from historical B0M, historical B1 and R0 PRIMARY tracks,
with MMS-FA timing.

| Group (reference EN fraction) | EN runs | Heard | Genuine B0-wrong heard | Han onset | Omitted | Wrong EN | **Reachable** (dialogues) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MM (< 0.2), primary | 550 | 521 | 186 | 43 | 89 | 22 | **39 (13)** |
| MIX (0.2–0.5) | 136 | 134 | 65 | 18 | 26 | 4 | 15 (9) |
| END (≥ 0.5) | 58 | 58 | 39 | 15 | 17 | 0 | 15 (8) |

The literal-recovery ceiling proxy is a genuine heard word that is in historical forced-EN and ≥ 0.5 inside R0 EN (±200
ms). At one selection per utterance:

| Group | Han-rendered: utterances (dialogues) | Other B0-wrong (mostly deletions next to Han): utterances (dialogues) |
|---|---:|---:|
| MM | 14 (11) | 38 (15) |
| MIX | 8 (6) | 21 (13) |
| END | 11 (8) | 11 (10) |

**S1 crop attrition** (historical, same unchanged S1 functions): of S1's 180 queries, only 59 obtained both a target and
an off-target crop. 80 failed `LOW_ASSOCIATION` and 14 `NO_MATCHED_OFFTARGET`. K1 targets sit at R0-EN onsets, so their
association should be higher, but it is unmeasured.

**Token-level preview of the frozen rule on S1** (diagnostic only). This uses S1's sealed clean, target-masked and
off-target-masked log-probs over Top-20 unions at the 59 paired queries. Labels are S1's reference-acceptable
first-token sets. c is any Latin union token other than B0's token b, and F is that single token's log-prob.

| Positions | Candidates | n | Accepted | Median A_target(c) | Median A_target(b) | Median C |
|---|---|---:|---:|---:|---:|---:|
| B0 token Han, EN-confusion | literal (acceptable EN) | 8 | **5** | 4.52 | 2.91 | **+1.83** |
| B0 token Han, EN-confusion | other Latin | 19 | **7** | 2.05 | 1.71 | +0.38 |
| B0 token Han, ZH-correct | other Latin | 3 | 2 | 0.70 | −0.24 | +1.25 |
| B0 token Latin (EN-correct; not K1's case) | literal / other | 78 / 592 | 25 / 50 | 3.69 / 0.06 | 3.87 / 4.89 | −0.17 / −4.89 |

In the Han-history case:
- precision of accepted = 5/14 ≈ 0.36;
- dialogue-macro AUROC(C) = 0.795, but only 3 dialogues have both classes;
- the floor/ceiling concern is not borne out, since literal c loses *more* under the target mask than b.

Accepting with likelihood ratio ≈ 0.63/0.37 ≈ 1.7, precision ≥ 0.75 needs a pre-acoustic literal rate of about 64%.
The unfiltered forced-EN rate in MM is 6.8% (forensics). R0-EN and association filtering will raise it by an unknown
amount.

## 5. Proposed GPU work — needs explicit authorization

| Step | Where | Work | Forecast |
|---|---|---|---|
| A `capture` | GPU job, `sbatch --export=ALL,PHASE=capture slurm/abts0_k1.sbatch` | 300 encoders; matched B0 (cB) and EN (cE) decodes with cached alignment attention; matched AUTO | About 48k decoder steps (forecast EN length 1.5× B0), about 12–15 min at S1's measured 14 ms/step. Peak about 4 GiB. Archive about 0.3–0.6 GB on `/mnt/data`. |
| `select` | CPU | DTW, association, R0 + S1 crops, earliest choice, controls; `candidates_sealed.json` committed and pushed | 30–50 min serial |
| B `score` | GPU job, `PHASE=score` | Only sealed chosen pairs (≤ 300): 3 encoders + 6 cold replays each | ≤ 103k steps worst case, realistically ≤ 30k. About 10–25 min, about 4 GiB. |
| `seal`, then `evaluate` | CPU | Output seal, then reference evaluation and label | Minutes |

**Total ≤ 1 GPU-hour** in two allocations on an H100 MIG 3g.40gb, each capped at 1.5 h. The design itself requires an
independent pre-run audit and a K1-seal audit before reference scoring.

**Plan.** Sealed at `b1e96dde` (`plan_hash sha256:5f3c3ae7…`), with all model files and audio verified. Re-run
`python experiments/abts0_k1.py prepare --hash-model` only if a pinned source changes; the GPU stages refuse a stale
plan.

**Missing evidence the jobs supply:**
- matched-engine forced-EN tokens;
- cached alignment attention for B0 and EN;
- hard-zero target/off-target log-probs for the chosen c and b.

## 6. K1 metrics

None computed. No K1 output exists, and the evaluator refuses to run without the output seal. Once the jobs have run,
`evaluate` reports, per group:
- precision (primary and ±200 ms) with a descriptive dialogue bootstrap;
- accepted, rejected and abstained counts by reason;
- unalignable (`UNTIMED_SPAN`) counts;
- genuine recoverable words and dialogues;
- reachable and all-genuine recall;
- dialogue-macro AUROC of C;
- U0, U1, U2, chosen, off-target-only and shuffled precision;
- literal / wrong-English / wrong-span / EN-from-Mandarin labels, and Han-rendered vs deletion-adjacent b blocks;
- a per-dialogue table and examples.

## 7. Reusable infrastructure for K2

- **Already there.**
  - `matched_decode` (baseline arms, AUTO, zero-vector identity reference).
  - `score_word` (cold cached replays for candidate log-probs; the template for the gradient path).
  - Sealed selections with `t`, prefix, `c_ids` and `b_ids`.
  - Capture `logits_sha256` per step for bitwise identity checks.
  - R1R `apply_steering_ratio_first` / `RatioFirstInterventionHook`.
  - Evaluator canonical scoring.
- **To build for K2.**
  - A gradient-enabled cached step: clone the pristine prefix cache outside inference mode, apply the edit once at the
    onset, keep candidate-piece KV graph-connected.
  - A z = 0 bitwise check against `matched_decode` logits at every scored step.
  - A ratio-first RANDOM chord solver.
  - FORCE / STEER / RANDOM free decodes.
- **K2 estimate.** About 400 decoder steps per accepted selection, including 3 loss evaluations, 2 backward passes and 3
  free decodes plus an identity decode. At most 300 selections, realistically 12–60 in MM. That gives **≤ 30–45 min,
  peak ≈ 6 GiB** (autograd over ≤ 10 tokens through 32 decoder layers with ≤ 200-token KV). It fits one MIG 3g.40gb
  allocation of ≤ 1 h.

## 8. Is K2 scientifically justified?

**Not yet.** K2 needs `K1_PASS`, and `K1_PASS` is unmeasured.
- **Against PASS:**
  - the small reachable target (39 runs, ceiling 14 Han-rendered utterances);
  - S1's crop attrition (59/180 paired);
  - an acoustic rule that separates literal from nonliteral tokens only weakly (LR ≈ 1.7 at token level; n = 27, 3
    dialogues).
- **For it:**
  - K1 targets sit at R0-EN onsets, a better case than S1's generic queries;
  - literal English loses more under masking than b (median C +1.83), so the rule is not biased against literal
    candidates.

The decision rests on the K1 run itself, about one GPU-hour. Spending it is a human call. The expected value is modest:
a `K1_FAIL` would be informative, because it closes the candidate-transplant route.

## 9. Reproduction

```bash
export PYTHONPATH=src:. LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib
python experiments/abts0_k1.py prepare --hash-model && python experiments/abts0_k1.py forecast   # CPU
python experiments/abts0_k1_evaluate.py registry                                                 # CPU, references
python -m pytest -q tests/test_abts0_k1.py
# after authorization only: PHASE=capture job -> select -> commit/push seal -> PHASE=score job -> seal -> evaluate
```

Diagnostics in §2.2 and §4 (tie analysis, B0 DTW vs CTC, S1 rule preview) are short scratch scripts reading only the
named sealed artifacts. Their outputs are quoted here; they are not K1 evidence.


## 10. Final scientific preflight — 2026-10-12

**Decision: `READY_FOR_K1_CAPTURE`, not submitted.** The runtime path (capture → select → score) is final. It is
covered by the sealed plan `sha256:5f3c3ae7…`, and nothing below changes it.

**Approved and implemented (§10.6).** A1–A3 and G1 were approved on 2026-10-12 with seven conditions. They are
implemented additively and sealed in `amendment_seal.json` before any K1 output existed. The only thing still required
before the job runs is **GPU authorization**.

K2 remains ineligible. Sections 1–9 are kept as the original record; where they differ, this section governs. An
uncommitted draft of this section, from an earlier pass, was reviewed, verified and folded in here.

### 10.1 Remaining problems and resolutions

| # | Issue | Verified finding | Resolution |
|---|---|---|---|
| 1 | Denominators | The registry counts **maximal reference English runs**, not words. MM has 186 genuine, B0-wrong, heard runs. 43 have a Han-rendered onset and 39 are reachable (onset Han-substituted, aligned B0 index > 0, heard), in 13 dialogues. The ceiling of 14 utterances (30 words) is a *word*-level proxy. It uses the mismatched historical EN branch and CTC ±200 ms, so it is not a bound. | Separate denominators (§10.2). The frozen 0.20 gate applies to the 39 reachable onsets: **≥ 8 onset hits**, which is only 4.3% of 186. With one selection per utterance, the 14-utterance proxy would allow at most 14 hits (35.9% of 39). Hitting 8 uses 57% of that proxy. |
| 2 | Preview precision (§4) | "5/14 ≈ 0.36" pooled two populations. On EN-confusion positions, 5/8 = 62.5% of literal candidates were accepted against 7/19 = 36.8% of other Latin candidates. Precision is 5/12 = 41.7% and the likelihood ratio is 1.70, so 75% precision would need a pre-acoustic literal rate of 63.9%. Adding the 3 ZH-correct candidates (2 accepted) gives 5/14 = 35.7%, a likelihood ratio of 1.53 and a required rate of 66.3%. | The definitions are corrected here. Units are single tokens on S1 queries, not whole-word K1 pairs. Only 3 dialogues are involved, so the result is descriptive only. **No threshold changes:** A_target(c) ≥ ln 2 and C ≥ ln 2 stay. |
| 3 | Timing | Comparing B0 DTW to the CTC proxy over 11,534 Han units, the median absolute error is 0.222 s and 44.8% fall within 200 ms. The 0.389 s figure is the **mean signed** error. With the median absolute at 0.22 s, it reflects a skewed tail, not a constant offset. Runtime association compares **two DTW estimates** (EN against B0) from the same heads on the same audio. Shared bias cancels there, and their relative error is unmeasured. | **Keep the 200 ms runtime tolerance** (`ONSET_TOL = 3200`, design step 2). Widening it, or applying a 390 ms shift, would be an untested, post-design criterion change that admits more false associations. A tight tolerance only costs coverage, and low coverage now reports as INSUFFICIENT (A3/G1), not FAIL. As a descriptive sensitivity check only, report how many `ONSET_MISMATCH` proposals have \|onset_diff\| ≤ 6400. This is already recorded in `selection.json`, and those rows are neither scored nor used for the decision. |
| 4 | Evaluator literal label | The v1 primary criterion requires the candidate's EN-word DTW interval to overlap the reference unit's CTC interval at tolerance 0. That interval carries the error from item 3 directly, so v1 under-counts literal candidates and biases toward K1_FAIL. The design defines precision **lexically**: "literal spoken-English word at the associated span … not merely appearing elsewhere". | **A1**: timing-free lexical labelling (§10.3). The v1 labels at 0 and ±200 ms become localization diagnostics. |
| 5 | Onset recall | v1 credits a reachable run when *any* genuine literal word inside it is accepted. That is run coverage, but the design's gate is "reachable-onset recall". | **A2**: the numerator counts onset hits only. v1's run coverage is kept as a secondary metric. |
| 6 | Opportunity count | v1 returns INSUFFICIENT only if reachable < 12 or < 6 dialogues, or if < 3 AUROC dialogues. With 39 reachable runs in 13 dialogues, a structurally thin candidate pool would read as FAIL. The design says "Missing classes/coverage means insufficient evidence". | **A3** (evaluator precedence) and **G1** (an enforced reference-free gate before scoring). |
| 7 | Runtime stops | `capture` records `B0.equals_B0M` but does not stop on a mismatch. `score` checks only that the candidate seal exists. | **G1** enforces integrity and capacity before `PHASE=score`, and the sbatch score branch requires its PASS file. |
| 8 | Reproducibility | Re-verified at HEAD `3c19e654`: all 13 source pins match; config `sha256:5befb72a…` and design `sha256:bf6cf5c2…` match the plan; the panel has 300 unique IDs (20 dialogues × 15); 11/11 model files were verified at prepare; `references_used = false`; no Slurm jobs. B0M equals the R0 baseline on 300/300, and B0M equals the matched engine on 300/300 by historical run-time checks. Neither is a fresh ABTS GPU replay. `processed_argmax` breaks ties to the lowest ID; `cached_greedy`'s `topk` does not (§2.2). | Capture must reproduce B0 tokens and termination on **300/300**. Any mismatch → `K1_BLOCKED`, with no exclusions. The evaluator is not in the plan pins, so A1–A3/G1 need no re-`prepare`. |
| 9 | Stale config prose | `compute.job` says "one Slurm job, 2 h". The authorized sequence uses two allocations of ≤ 1.5 h. | Documented here, but the config is not edited: it is pinned, and editing would invalidate the plan. |

### 10.2 K1 eligibility and recall definitions (MM is primary; MIX and END are reported separately)

Each measurement below is kept separate.

| Measurement | Definition | Reference-free? |
|---|---|---|
| **Structural eligibility** | The funnel of proposals by association/region status, then U0 ⊇ U1 ⊇ U2 ⊇ chosen, then encoding-valid chosen. | yes |
| **Opportunity count** | MM encoding-valid chosen pairs, and their dialogues (G1, before scoring). After scoring: MM pairs with status `SCORED` and a finite rule (A3). | group label only |
| **Candidate precision** | Primary: verified `LITERAL` / all accepted MM, with `AMBIGUOUS` counted as non-literal. Upper bound: (`LITERAL` + `AMBIGUOUS`) / accepted. Also: pre-acoustic precision of the chosen set; U0/U1/U2 pool precision; off-target-only and shuffled-track controls. | no |
| **Reachable-onset recall** (gate) | Distinct reachable MM runs whose **first** reference unit is the aligned onset `i` of an accepted, `LITERAL`, genuine candidate, divided by the reachable count from captured B0. This should be 39; any other value means B0 changed, so `K1_BLOCKED`. | no |
| All-genuine onset recall | The same numerator / genuine heard MM runs (186). Includes deletions and wrong-English runs, which are unaddressable by design. | no |
| Reachable-run coverage | The v1 numerator (a genuine literal anywhere in the run) / 39. Secondary only. | no |

### 10.3 Frozen rules

**Selection (unchanged; pinned in the plan).** Design steps 1–5 and I1–I11 apply exactly as frozen:
- regex units and association at ≥ 0.5 coverage and ≤ 200 ms onset;
- reject words already present in B0; require t ≥ 1 and a complete UTF-8 prefix;
- R0 EN fraction ≥ 0.5, S1 crops unchanged, crop word-cover ≥ 0.5;
- one earliest eligible pair per utterance, abstaining if verification fails;
- accept iff A_target(c) ≥ ln 2 and C ≥ ln 2.

The runner opens no reference or evaluator file, and its firewall enforces this. Nothing in A1–A3/G1 can reach
selection or threshold choice.

**Evaluation (approval requested; additive `experiments/abts0_k1_evaluate_v2.py`, v1 kept and also reported):**

- **A1, timing-free lexical label.** Let H be the chosen pair's B0 Han block.
  1. Map H's raw K1 units to canonical B0 units (`segment_units(normalize_text(B0 text))`) by exact monotone surface
     match. Any unmapped unit → `AMBIGUOUS_UNMAPPED`.
  2. Let R be the reference indices that the frozen `csasr.evaluation.mer.align_tokens(reference, B0)` aligns
     (MATCH/SUB) to H. This is the same alignment that defines "reachable", so numerator and denominator share it.
     If R is empty → `AMBIGUOUS_NO_ALIGNMENT`.
  3. Let `i = min(R)` and let `cu` be the candidate's canonical units. If `rt[i:i+|cu|] == cu` and all those units are
     English → **`LITERAL`**. The frozen `unit_status` at `i` then gives the genuine/artifact flags and the B0
     rendering. `LITERAL_ONSET` means `i` is the first unit of a reachable run.
  4. Otherwise:
     - `cu` starts at some j ∈ R or at i − 1 → `AMBIGUOUS_BOUNDARY`;
     - else `cu` occurs elsewhere in the reference → `WRONG_OCCURRENCE`;
     - else `rt[i]` is English → `WRONG_ENGLISH`;
     - else → `EN_ON_MANDARIN_SPEECH` (translation or hallucination; translation cannot be told apart lexically).

  No timing and no human adjudication enter the label. AUROC uses `LITERAL` against non-ambiguous non-literal pairs.
  The v1 CTC-overlap labels (0 and ±200 ms) and their agreement with A1 are reported as diagnostics only.

  Reference-side reporting stays separate. Genuine heard runs are split by B0 onset rendering: Han substitution
  (addressable), deletion/omission (unaddressable), wrong-English, correct onset with a later error, and
  boundary/other.
- **A2, onset recall.** Use the numerator in §10.2. All numeric GO thresholds are unchanged: ≥ 12 accepted in
  ≥ 6 dialogues; precision ≥ 0.75; ≥ 6 genuine recoverable words in ≥ 4 dialogues; onset recall ≥ 0.20; dialogue-macro
  AUROC ≥ 0.65 over ≥ 3 dialogues.
- **A3, precedence.** Checks run in order and the first match decides:
  1. `K1_BLOCKED`: any integrity failure, `INVALID_ENGINE_IDENTITY`, a seal mismatch or a reachable count ≠ 39.
  2. `K1_INSUFFICIENT`, with a reason code:
     - `STRUCTURAL`: G1 failed;
     - `COVERAGE`: < 12 scored MM pairs or < 6 such dialogues;
     - `CLASSES`: < 3 MM dialogues with both A1 classes;
     - `REACHABLE`: reachable < 12 or < 6 dialogues.
  3. `K1_FAIL`: any GO condition fails.
  4. `K1_PASS`: every GO condition holds. This makes K2 eligible for human review only.

  The COVERAGE and STRUCTURAL thresholds are the logical minimum the frozen gates already need. Accepted pairs are a
  subset of scored pairs, which are a subset of chosen pairs. An AUROC dialogue needs ≥ 2 scored pairs. So these are
  not new tuning parameters. Both INSUFFICIENT and FAIL stop K2. INSUFFICIENT (STRUCTURAL/COVERAGE) is still a
  negative feasibility finding for this candidate generator at panel size. It is simply not evidence about the
  acoustic rule.
- **G1, pre-score gate.** Additive `experiments/abts0_k1_gate.py`, CPU, writes `results/inference_cf/abts0_k1/gate_prescore.json`.
  - **Integrity** (any failure → `K1_BLOCKED`):
    - 300/300 ordered capture rows carry the plan hash;
    - B0 `equals_B0M` holds on 300/300;
    - all archived attention objects match their hashes;
    - `runtime_capture.json` shows weights unchanged, no grads, no forbidden opens and no site hooks;
    - the selection and candidate seal hashes match, and both are committed **and pushed**.
  - **Capacity** (any failure → `K1_INSUFFICIENT/STRUCTURAL`, and **no scoring**):
    - ≥ 12 MM encoding-valid chosen pairs;
    - in ≥ 6 dialogues;
    - ≥ 3 dialogues each holding ≥ 2 of them.

  MM membership comes from `groups_prerun.json`, a per-utterance group and dialogue list written from references
  **before capture** and committed with its hash. G1 opens no reference or label. The `PHASE=score` branch of the
  sbatch script adds one `test` that the gate status is `PASS`.

### 10.4 Execution sequence (not run; each step waits for the previous one)

```bash
cd /home/tungnx/cs-asr-steer-inf
export PYTHONPATH="$PWD/src:$PWD" LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
PY=/home/tungnx/miniconda3/envs/acl1/bin/python; R=results/inference_cf/abts0_k1
# 0  DONE (fe213d17): v2 evaluator, G1 gate, sbatch guards, groups_prerun.json; amendment_seal.json READY_FOR_K1_CAPTURE.
"$PY" experiments/abts0_k1_gate.py verify-amendments          # must print the seal hash; the sbatch re-checks it
# 1  GPU capture (authorized). Resumes at row level if re-submitted after a wall-time stop.
mkdir -p $R/run1
sbatch --output=$R/run1/slurm-%j.out --error=$R/run1/slurm-%j.err --export=ALL,PHASE=capture slurm/abts0_k1.sbatch
# 2  CPU selection (falls back to serial DTW automatically if /dev/shm is full), then seal the candidates in git.
"$PY" experiments/abts0_k1.py select --workers 16
git add $R/run1/capture $R/run1/runtime_capture.json $R/run1/selection.json $R/run1/candidates_sealed.json
git commit -m "ABTS-0 K1: capture + sealed candidates (pre-scoring)" && git push origin research/steering-forensics
# 3  Gate. If not PASS: record K1_BLOCKED / K1_INSUFFICIENT(STRUCTURAL) in k1_status, commit, STOP.
"$PY" experiments/abts0_k1_gate.py check && git add $R/gate_prescore.json && git commit -m "ABTS-0 K1: pre-score gate" && git push
# 4  GPU scoring of sealed chosen pairs only (the sbatch refuses to score without gate PASS)
sbatch --output=$R/run1/slurm-%j.out --error=$R/run1/slurm-%j.err --export=ALL,PHASE=score slurm/abts0_k1.sbatch
# 5  Output seal (also re-verifies archive hashes), commit + push BEFORE opening references
"$PY" experiments/abts0_k1.py seal
git add $R/run1/scores $R/run1/runtime_score.json $R/output_seal.json && git commit -m "ABTS-0 K1: output seal" && git push
# 6  Reference evaluation, v2 primary (approved) with v1 alongside; then a new sealed k1_status. STOP for human review.
"$PY" experiments/abts0_k1_evaluate_v2.py evaluate
```

### 10.5 GPU cost and stop conditions

Both phases run on an H100 MIG 3g.40gb with a wall limit of 1.5 h each and a peak of about 4 GiB. The time
estimates come from the saved forecast (15–30 ms per decoder step).

| Phase | Work | Expected | Worst case |
|---|---|---|---|
| Capture | 300 encoders, B0 + EN with attention, plus AUTO; about 48,088 decoder steps; 0.3–0.6 GB archive on `/mnt/data` | 15–40 min, including I/O on the congested filesystem | Hard cap 180,000 steps, 45–90 min |
| Select | CPU DTW over 600 branches | 30–50 min | Several hours, serial |
| Score | Sealed chosen pairs only: 3 encoders + 6 cold replays each | 10–25 min | ≤ 102,966 steps, 26–52 min |

The expected total is about 0.5–1 GPU-hour. At most 3 GPU-hours can be allocated.

**Stop conditions.**
- Any exception in a GPU phase stops the run. This covers source/plan/config pins, waveform hashes, eager attention,
  alignment heads, masks changing bytes outside the crop, and the firewall.
- A wall-time stop allows **one** resubmission of the same phase, with identical code and row-level resume. An
  incomplete phase after that → `K1_BLOCKED`.
- Any B0 mismatch, missing row, archive hash failure or changed weight probe → `K1_BLOCKED`. No exclusions.
- If G1 capacity fails → `K1_INSUFFICIENT`, and scoring is not submitted.
- Any `INVALID_ENGINE_IDENTITY` or seal mismatch → `K1_BLOCKED`.
- Population, thresholds, 200 ms association, rules and the evaluator must not change after step 1.
- No K2, entropy/RL, optimizer or layer sweep. A `K1_PASS` only queues human review.

### 10.6 Approved amendments: implementation and final checks

All seven approval conditions are met:

| Condition | How it is met |
|---|---|
| 1. No false lexical matches | `lexical_label` uses the frozen `align_tokens` path **and** the exact union of all co-optimal alignments (forward/backward DP, checked against brute force). These cases are AMBIGUOUS: the word repeated inside the alignable window (`REPEAT`); an optimal alignment that assigns the word outside the block or deletes it (`ALIGNMENT`); a block that starts on a different English run (`RUN`); an adjacent occurrence (`BOUNDARY`); an unmappable or non-Han block (`UNMAPPED`); a block aligned to nothing (`NO_ALIGNMENT`). Primary precision counts all AMBIGUOUS cases as incorrect, and `precision_optimistic` is reported separately. |
| 2. Recall reporting | `onset_recall` = onset hits / reachable (39). `genuine_run_onset_coverage` = onset hits / genuine heard runs (186). Also reported: `reachable_run_coverage_a1`, and the original v1 metric `reachable_run_coverage_v1`. |
| 3. Safety | Every labelled pair carries `block_b0_correct_mandarin` and `over_correct_mandarin` (a non-literal proposal whose block contains Mandarin units that B0 got right). Accepted, chosen and `EN_ON_MANDARIN_SPEECH` counts are reported per group. |
| 4. 200 ms runtime tolerance | `ONSET_TOL = 3200` is unchanged. `sens400_*` counts and labels `ONSET_MISMATCH` proposals with \|onset_diff\| ≤ 6400; these are never scored or used in the decision. |
| 5. Additive changes | New files are `experiments/abts0_k1_evaluate_v2.py`, `experiments/abts0_k1_gate.py` and `tests/test_abts0_k1_v2.py`. The v1 evaluator is untouched; v2 runs it and stores `v1_label` and v1 metrics alongside. The sbatch changed by two guard lines. |
| 6. Hashes, seals, plan consistency | See below. The runtime source bundle is unchanged, so the plan did not need re-preparing. |
| 7. Selection is reference-free | The runner and `abts0_k1.py` are untouched, and a test asserts they never reference the gate, v2 or `groups_prerun`. The gate opens no reference file, and its audit hook raises on forbidden paths. `groups_prerun.json` (MM 214 / MIX 56 / END 30) was frozen before any capture output existed; `groups` refuses to run once one does. |

**Final hashes.**
- Commit: `fe213d17` (amendments). The amendment seal is committed on top of it.
- Plan `sha256:5f3c3ae7410df3d53d8f751aa623a88d615f50b737f65a76ae183fa031887951`. Its self-hash is valid, all 13 source pins match, and its source list equals the runner's `SOURCES`.
- Config `sha256:5befb72ae3a5e41a3f557be66dd2c22a697543334ec3ebe5454d80db437a392f`.
- Design `sha256:bf6cf5c2f328cb6d057f19df2dd61ece44f456d33733e479f6704d4953ca7c17`. This matches the config pin.
- Groups `sha256:2cf68cb6ec06b7d8e4da32fd8a64cf93218a257064f0c02751119087f579d2a8`.
- Amendment seal `sha256:c4368812d9cd42709b39ba7589df597e65aad1aea41cb2a3fc4d6ff4d458e93e`, status `READY_FOR_K1_CAPTURE`. It pins:
  - the v1 and v2 evaluators, the gate and the sbatch;
  - the groups, registry and plan files, the config and the design;
  - `mer.py`, `pier.py`, `normalization.py`;
  - both test files.

  `verify-amendments` passes, and the capture phase refuses to run if any pinned file changes.

**Focused tests.** 46/46 passed (exit 0): `tests/test_abts0_k1_v2.py` (22 new) and `tests/test_abts0_k1.py` (the
original 24). No regression sweep was run.

**Final K1 status: `READY_FOR_K1_CAPTURE`.** `k1_status.json` (`K1_BLOCKED`, pre-amendment) is kept as the historical
record.
