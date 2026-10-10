# P0 inference-time counterfactual feasibility — provisional

**Status: `BLOCKED_G3` (K=1) → `P0_BLOCKED_EVIDENCE` after the one authorized K=3 repair. P1 has not started.** The frozen P0 rules were committed before GPU work in `55faf76ca6d16173efc5eabeedea1bda0c3f4242`. An independent audit (`P0_INDEPENDENT_AUDIT.md`) reproduced every K=1 number from the row shards and confirmed the gate verdicts; the P0-R1 K=3 revision (`docs/inference_cf/P0_R1_EVIDENCE_SPEC.md`, report `docs/inference_cf/P0_R1_REPORT.md`, job 54712, manifest `sha256:e4514a8229c34c119e74cdf0da2d4fd324c5310e5b3e7f8ff89b4b16042ee9eb`) reduced candidate collision 87.7%→73.7% and doubled usable rows 7→15 but still fails the frozen G3 support criteria (collision >50%; Mandarin usable 3≪10; AUROC not estimable). See the P0-R1 section below. A narrow parser correction after physical job 54702 was committed as `cba7784` before the only corrected run, job 54703. The corrected manifest is `sha256:0287abb2f6537f1d88f179fa62310424e411b1baee47258670c87d781badfb6c`. The original checkout is `/home/tungnx/cs-asr-steer` (`learned-expansion`, HEAD/base `ce5d225f41d66c5d8aadc9df3c734fbc608646db`) with unrelated A7 edits untouched. This worktree is `/home/tungnx/cs_asr_steer_inf`, branch `feature/inference-cf-steering`.

## Fixed configuration and data

- Frozen model: `openai/whisper-large-v3`, local weights SHA-256 `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`; tokenizer JSON SHA-256 `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca`. bf16, eager attention, frozen/eval. Zero-indexed decoder L24, post-cross-attention residual `r=q+u_source`, pre-FFN. This is the DG-02 validated site and DG-03 engineering anchor; no new layer selection.
- Ordinary baseline: `task=transcribe`, `language=zh`, greedy beam 1, `do_sample=false`, temperature 0, `max_new_tokens=200`, `condition_on_prev_tokens=false`. K=1 raw next-token argmax candidates and single-token `log_softmax` cross-scores. No alpha or temperature search, no steering edit.
- Exact prompt tokens: `c0` ordinary baseline Mandarin `[50258,50260,50360,50364]`; `cM` explicit Mandarin matrix `[50258,50260,50360,50364]`; `cE` explicit English embedded `[50258,50259,50360,50364]`. Tokens are `<|startoftranscript|>`, language, `<|transcribe|>`, `<|notimestamps|>`. `c0==cM` by serialized IDs.
- Only dialogue-v2r3 `D-dev-select` was read: frozen DG-04 B0 outputs, the role manifest and evaluator-only POI/reference units. Deterministic seed 240924 and SHA-256 ordering selected 20 wrong-English positions from 884 available POIs/187 utterances, 20 correct-English from 1,384/232, and 20 Mandarin from 300/300. One utterance per selected row. Three text mismatches on replay were explicitly skipped; retained counts are 18 wrong English, 19 correct English and 20 Mandarin. No test or confirmation outcome was read by P0.

## Position and scoring validity

For every retained row, the ordinary model generated its actual content token IDs. The installed `generate` API omits the forced prompt from `sequences`; the corrected parser treats them as content only, strips first EOS and takes the shared prefix before the offline-selected diagnostic position. All three conditions concatenate their exact prompt and **identical content prefix**. The L24 recorder captures the last query, which predicts the same logical next content token. Five retained rows are at the first content token, where the final prompt token is the query. Every decoder mask is all ones; all branches use full causal replay with `use_cache=false` and positions `0..input_length-1`, with no exchanged KV history. All are greedy beam lineage 0. The same real encoder representation is reused for `c0/cM/cE`; the processor supplies the real audio padding mask. The CPU audit checked prompt lengths, content prefix lengths, logical query indices, mask lengths, replay positions, branch lineage and raw state-vector dimensions for all 57 retained rows.

The real `yE` and `yM` come from their respective current-audio branch distributions only. Their IDs remain fixed for the two shuffled-audio scorers. All four real and four shuffled log probabilities, candidate IDs, EOS/special flags and row vectors are saved. Neither branch chose EOS or another special token in the retained panel. The deterministic cyclic permutation has 60 no-self audio pairs. Each row retains its own identity, prefix, candidates and conditions; only encoder audio changes. The CPU auditor verified the permutation and recomputed `q_own` and `q_cross` from all saved scores. A collision sets `q_cross=null` by the frozen rule.

## Gate decisions

| Gate | Provisional verdict | Evidence |
|---|---|---|
| G1 | `DEGENERATE_BY_CONSTRUCTION` | `c0` and `cM` prompts are token-identical; all 57 `||h0-hM||` are exactly 0. `||hE-hM||` median 2.749, range 1.058–7.744; direction norm median 0.9999996. `rho` median −0.9999993 and diagnostic collapse factor median 0.9999993 are mechanical. The original geometry/collapse **need** factor is non-identifiable and cannot survive into P1. |
| G2 | `PASS` | 57/60 retained positions passed the full replay alignment audit; three baseline text mismatches have explicit skips. Prompt/mask/cache/query/lineage, L24 site and the three raw vectors are saved per retained row. |
| G3 | `WEAK/BLOCKED` | Finite retained coverage 57/60 = 95%, but candidates collide in 50/57 = 87.7%. Only 3 wrong-English, 3 correct-English and 1 Mandarin rows have `q_cross`; the predeclared 50% collision and at-least-10-per-class/AUROC criteria fail. AUROC is not estimable. |

The distinct-state result does **not** rescue G3: a direction exists, but the frozen one-token evidence contrast is usually undefined. The q-cross median is 8.047 (n=3) for wrong English, 3.031 (n=3) for correct English, and 5.383 (n=1) for Mandarin. Those sparse values cannot establish separation. Diagnostic `q_own` means over all retained rows are −0.0139 wrong English, −0.0186 correct English and −0.0465 Mandarin; it is not the gate. Among the seven noncolliding rows, paired absolute real/shuffled q-cross change averages 1.994 (median 1.961), with signed mean 1.895. This shows a response to audio on those seven comparisons but fails the frozen minimum of ten and does not establish a usable score across the panel. Baseline-correct English is reported separately because false activation there would risk damage.

Deterministic representative rows (lexicographically first retained row in each stratum, plus the first noncollision where different):

| Stratum | Row | Diagnostic position / evaluator-only target | `yE`, `yM` | `q_cross` real / shuffled |
|---|---|---|---|---|
| Wrong English | `english_wrong:ZH-CN_U0011_S0_111:0` | 0 / `okay` | `Okay`, `所以` | 8.531 / 5.160 |
| Correct English | `english_correct:ZH-CN_U0011_S0_265:27` | 22 / `been` | same token ID | collision / collision |
| Correct English, first noncollision | `english_correct:ZH-CN_U0091_S0_104:2` | 2 / `quite` | `这`, `它` | 0.188 / −0.266 |
| Mandarin | `mandarin:ZH-CN_U0011_S0_166:18` | 14 / `呢` | `的`, `的` | collision / collision |
| Mandarin, first noncollision | `mandarin:ZH-CN_U0023_S0_666:1` | 0 / `就` | `Many`, `对` | 5.383 / 5.729 |

These are diagnostic locations selected offline from references, not oracle inputs to inference. The character-fraction-to-token mapping is coarse and does not prove exact word-boundary localization. Three skipped rows had a fresh single-item baseline text differing from the frozen batched B0 text; the frozen skip rule was kept. Do not reinterpret this panel as a recognition evaluation.

## Execution and artifacts

Both jobs used one `mig` allocation, `NVIDIA H100 80GB HBM3 MIG 3g.40gb` on `worker-mig-3g40gb-0`; each terminal state is `COMPLETED` exit 0. Job **54702** (61 s allocation, 43.47 s measured program time) produced only invalid state rows: 57 prompt-parser skips and 3 frozen-baseline-text skips. Its entire `results/inference_cf/p0/` directory, including manifest and Slurm stdout/stderr, is preserved. The minimal parser regression and spec correction were committed before corrected job **54703** (59 s allocation, 45.76 s measured program time), whose output is `results/inference_cf/p0_retry/`. The two allocated GPU times sum to **0.0333 GPU-hours**; corrected run alone is 0.0164 GPU-hours. Peak CUDA allocation was 3,426,093,568 bytes (3.19 GiB). The 60 selected audio files total 1,150.579 s; retained scored rows total 1,094.833 s. Retained baseline content totals 2,895 tokens. Corrected-job model accounting: 120 encoder passes (real plus shuffled), 60 baseline generations, 171 real full-replay decoder passes and 114 shuffled full-replay passes. Loading, diagnostic and baseline costs are all included in job time; this is not a deployment latency measurement.

The CPU audit verified the corrected Git commit/manifest and all pinned source hashes, 60 unique identities, panel/permutation hashes, all skip reasons, exact prompts, condition equality, full-replay alignment, raw state-vector geometry, all crossed scores, candidate collisions and no-self shuffled map. `summary.json` is independently recomputed from row shards. It is not trusted as the GPU output.

Artifacts: `docs/inference_cf/P0_FEASIBILITY_SPEC.md` (pre-run freeze), `results/inference_cf/p0_retry/{manifest.json,panel.json,audio_permutation.json,runtime.json,rows/*.json,summary.json,slurm-54703.out,slurm-54703.err}`, and preserved failed attempt `results/inference_cf/p0/`. The panel includes evaluator-only surfaces for audit; the inference API accepts no reference, target token or oracle switch position.

**Next contract:** stop before P1. If later scientific work repairs G3, it must freeze a new evidence definition and run under a new authorized stage. Because G1 degenerates, any future gate would have to be explicitly evidence-only; the original `e*c` geometry need interpretation is invalid. **P1 has not started.**

## P0-R1 revised evidence (K=3), 2026-09-24

The one authorized scientific-definition repair — K=1→K=3 short greedy continuations scored by
token-average log probability, everything else frozen and the byte-identical panel reused — was
run once (Slurm **54712**, H100 MIG `3g.40gb`, COMPLETED, 69 s; manifest `e4514a8…` supersedes the
P0 manifest `0287abb…`). Independent CPU recomputation from raw per-step log-probs reproduces the
committed `summary.json` exactly.

| Gate | K=1 (54703) | K=3 (54712) |
|---|---|---|
| G1 | `DEGENERATE_BY_CONSTRUCTION` (`‖h0−hM‖=0`) | `DEGENERATE_BY_CONSTRUCTION` (unchanged) |
| G2 | `PASS` | `PASS` (0 violations across all 3 continuation positions) |
| G3 | `WEAK/BLOCKED` (collision 87.7%, usable 7, AUROC n/e) | `WEAK/BLOCKED` (collision 73.7%, usable 15, AUROC n/e) |

K=3 met the audio-dependence sub-criterion (abs Δq_cross mean 0.485 over 15 rows ≥ 0.05/≥10) but
failed the collision (≤50%) and ≥10-per-class/AUROC sub-criteria; collisions concentrate in the
Mandarin stratum (17/20). Frozen thresholds were **not** weakened. Final: **`P0_BLOCKED_EVIDENCE`**
— the candidate-continuation support formulation has failed P0; a new support definition must be a
separately versioned experiment. The direction `d = normalize(hE − hM)` at L24 is well-defined but
has **no** validated evidence gate. Full detail: `docs/inference_cf/P0_R1_REPORT.md`. **P1 has not
started.**
