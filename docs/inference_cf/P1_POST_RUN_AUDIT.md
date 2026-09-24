# P1 post-run independent audit

**POST_P1_AUDIT: PASS**

| Question | Answer and evidence |
|---|---|
| Did α = 0 reproduce baseline? | **Yes.** F3 with the hook installed is bitwise equal to unsteered F1 at all 190 steps, and tokens equal the unsteered argmax. Against `generate()`: 9/10 identical; the 1 divergence sits at the exact position where R2's independent replay of that `generate()` sequence already disagreed with `generate()` (cached-vs-replay bf16 numerics, hook-independent). |
| Was the nonzero edit applied only at the specified state? | **Yes.** Every audit record is layer 24 at the current query position ≥ 4. Tests on a real 25-layer model show layer 23 unchanged, other positions' layer-24 sites unchanged, and earlier-query logits unchanged. |
| Was d really hE − hM? | **Yes.** `hE`/`hB` come from the recorder at the layer-24 site of cE/cB replays of the identical prefix; `cB = cM` per the manifest. The sign is confirmed by `cos(h' − hB, d)` ≥ 0.872 on all 20 edits. |
| Was the gate unchanged from R2? | **Yes.** `g` is recomputed as `E·R_B` from the logged components via unchanged `core_r2` on every step, with the same partition hash and fallbacks. The Ecf conflict is unused. |
| Was the nonzero α frozen before outcome? | **Yes.** α = 1.0 is in spec v1 (commit `113c48e`, before attempt 1) and unchanged in v1.1. |
| Was NormPreserve exact? | **Yes, within bf16.** Maximum relative norm change 0.38%. The realized edit equals the site-precision replica of the specified formula exactly. |
| Did branch computation contaminate the decode? | **No.** No hooks during F1/F2 (asserted every step), no hooks after the run, α = 0 bitwise identity, and deterministic identical statistics across two separate jobs. |
| Did reference/evaluator information enter? | **No.** The decode API takes audio, prompts, partition, null odds and the dose only. R2 `generate()` tokens are used only by the CPU evaluator, as a diagnostic. |
| All positions and utterances accounted for? | **Yes.** 20/20 records, 374 steps, fallbacks (20 `mid_character`, 3 forced-prefix-blocked `g>0` steps) recorded. |
| Runtime/memory honest? | **Yes.** Program 37.0 s, wall 53 s, peak 3.42 GB, from `runtime.json` and `sacct`. |
| Any layer/α selection from recognition? | **No.** A single layer and a single nonzero α; no sweep code exists. |
| Was the attempt-1 rerun legitimate? | **Yes.** Attempt 1 failed only the norm-matching clause of A6, on 2 edits below bf16 resolution, because the instrument compared bf16 with float64. The fix replicates the specified edit at the actual precision and keeps the tolerance and sign clause. Nothing about the intervention changed, attempt 1 is preserved, and the rerun reproduced attempt 1's per-step statistics identically. The decision could not have been influenced by recognition outcomes: the recognition diagnostic was identical in both attempts and plays no role in any criterion. |

**Claim boundary.** P1 establishes only the computational and causal validity of the intervention
pipeline. It does not establish ASR improvement, an optimal direction, layer or dose, or
paper-level effectiveness. B0_AUTO remains the stronger unsteered baseline.

**POST_P1_AUDIT: PASS** → `READY_FOR_P2_COMPACT_DEVELOPMENT`. P2 is not started.
