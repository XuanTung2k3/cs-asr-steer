# ABTS-0 K1 — preflight, implementation and decision gate

**Label: `K1_BLOCKED`.** K1 needs fresh GPU forward passes that are not authorized: matched-engine forced-EN decoding with
cached alignment attention, and masked-audio rescoring. No saved artifact contains them. This is an engineering/data
gate, **not** a scientific verdict. K2 is not eligible, and nothing in K2 was implemented or run.

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
