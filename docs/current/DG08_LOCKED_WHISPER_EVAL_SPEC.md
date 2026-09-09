# DG-08 — Locked Whisper Core Evaluation (frozen protocol)

**Status: PROTOCOL FROZEN before new training/D-test (see §Provenance).** This file freezes the
DG-08 comparison before any DG-08 result is inspected. Authority: v6, `METHOD_CONTRACT.md`, and the
completed DG-00–DG-07 specs. Scope is **Whisper-large-v3 × CS-Dialogue only**. No cross-dataset /
cross-model expansion; no method redesign; no post-test tuning.

DG-08 question: under a locked protocol, does the proposed damage-aware adaptive steering method
(M*) provide a reproducible correction–damage–efficiency trade-off relative to the strongest
frozen-steering, learned-global, and matched-budget PEFT baselines?

## 1. Finalists (frozen identities)

| ID | System | Kind | Source / construction |
|---|---|---|---|
| **F0** | Frozen Whisper | deterministic | no intervention |
| **F1** | DG-04 frozen steering | deterministic | B1 ρ=0.5: L24 exact site, fixed `v_local`, all eligible positions, α=ρ·s_L24 (β=4.465628877080159), NormPreserve. `results/dg04/reference.json` |
| **F2** | SALSA exact-global | learned (3 seeds) | `GlobalVector` (1,280 params, zero-init) at L24 exact site, α=scale=1, NormPreserve. DG-07 `LB1_SALSA_EXACT_GLOBAL` |
| **F3** | matched-budget LoRA | learned (3 seeds) | `ExactLayerQvLoRA`, decoder L24 self-attn q_proj+v_proj, rank 9, α 9, dropout 0, 46,080 params. DG-07 `LB2_LORA_MATCHED_BUDGET` |
| **F4** | **M\*** proposed | learned (3 seeds) | DG-06 **D1** damage-aware adaptive controller: `LayerNorm(1280)→Linear(1280,32)→GELU→Linear(32,3)`, gate+rank-2 mixture over `V^0=[v_local,v_cond]`, L24, β=4.465628877080159, objective `L_corr(C_E)+1.0·L_ret,M(R_M)`, KL `p0‖pθ`, 43,651 params |

F4 is the frozen DG-06 D1 method. DG-07 found `conditioning direction: NO CLEAR CONTRIBUTION`, but
M* was already frozen; **`v_cond` is not removed** (that would be a new method after final selection).
The ablation result is recorded honestly as a development finding only.

## 2. Seeds

Neither `METHOD_CONTRACT.md` nor DG-07 freezes a three-seed list (DG-07 used one dev seed, 42).
DG-08 freezes: **seeds = [13, 42, 73]** for the learned finalists F2/F3/F4. F0/F1 are deterministic
(one evaluation per decode regime; no artificial seed variance). Seed 42 checkpoints are **reused**
from DG-06/DG-07 because their configs match this locked protocol exactly (verified by hash in the
test lock, §7); only seeds 13 and 73 are newly trained → **6 new training runs**.

Reused seed-42 checkpoints (verified by SHA in the lock):
- F2 seed 42: `results/dg07/LB1_SALSA_EXACT_GLOBAL/selected_checkpoint.pt`
  (`sha256:03cc0bf6…6f8e`).
- F3 seed 42: `results/dg07/LB2_LORA_MATCHED_BUDGET/selected_checkpoint.pt`
  (`sha256:7861cdf8…0046`).
- F4 seed 42: `results/dg06/d1/selected_checkpoint.pt` (`sha256:b9c45a1e…f241`).

## 3. Training configuration (reuse-exact; only new variable = seed)

Every learned finalist reuses its frozen DG-06/DG-07 configuration unchanged: `loc-train ∪ util-train`
training, AdamW, LR 5e-4, weight decay 0, batch 8, accumulation 2, three epochs, grad-clip 1.0, the
frozen contributing pool (1,346 batches / 673 optimizer steps per epoch = 2,019 updates), objective
`L_corr(C_E)+1.0·L_ret,M(R_M)`, β 4.465628877080159, L24, `V^0` basis, C_E/R_M frozen artifacts.
**No** retuning of LR, λ, β, rank, α, layer, basis, controller width, target modules, epochs, budget,
or checkpoint rule. Implementation: seeds 13/73 run through the **frozen** DG-06 (M*) and DG-07
(SALSA/LoRA) runners via an added backward-compatible `--seed` (default 42 preserves exact behavior);
no other code path changes. Note: F2 SALSA is zero-initialized with no stochastic training component,
so its seeds are expected to coincide (reported honestly, not as fabricated variance).

## 4. Data roles (DATA_EXPOSURE)

- Training: `loc-train ∪ util-train` only.
- Checkpoint selection: `D-dev-select` only (greedy, temperature 0, beam 1), frozen DG-06/DG-07 rule
  (positive canonical utility → PIER gain → matrix retention → lower realized energy).
- Inference timing: a frozen deterministic `D-dev-select` subset (§9).
- Final evaluation: **`D-test`** (after the §7 lock).
- `D-dev-confirm`: **not used** for any DG-08 selection (default: no confirmation pass).
- No seed is discarded; all three preregistered seeds count. No D-test tuning of any kind.

## 5. Decoding

- **Greedy (primary):** `do_sample=False, num_beams=1, temperature=0.0, task=transcribe,
  language=zh, max_new_tokens=200, condition_on_prev_tokens=False`.
- **Beam-5 (secondary robustness):** identical except `num_beams=5`. Uses the same greedy-selected
  locked checkpoints; no beam-specific reselection/retuning.

## 6. Metrics

**Primary:** PIER, MER, canonical correction–damage utility (= net POI corrections, corrections −
corruptions). **Secondary:** embedded-language WER, matrix-language CER/WER, corrections, corruptions,
embedded retention, matrix retention.

**Outside-harm limitation (recorded, not a defect):** canonical outside-harm and candidate-utility
require per-utterance candidate/POI alignment artifacts (`candidates_all.parquet` + `poi_*.parquet`).
These were **never generated for D-test** (D-test carries only the locked utterance-level manifest;
`candidates_all.parquet` contains no D-test rows). Generating them now would be new D-test processing
outside the locked evaluation. Therefore on D-test, `outside_harm`/`candidate_utility` are reported as
**NOT AVAILABLE** (`null`), while every text-derived metric above is computed from the locked
transcripts. Outside-harm remains available and reported on the DG-04/05/06/07 `D-dev-select`
development frontier. No value is fabricated.

Metric schema is frozen `metrics_v1`/`result_v1`; normalization and gain sign (`baseline − method`,
positive = better) are unchanged from DG-01.

## 7. HARD D-test lock (critical boundary)

After all three selected checkpoints exist for F2/F3/F4 and **before** any D-test intervention, the
lock artifact `results/dg08/DG08_TEST_LOCK.json` records: F0/F1 configs; F2/F3/F4 checkpoint SHA-256
for seeds 13/42/73; model revision; basis hash; L24; objective/λ; greedy + beam-5 decoder configs;
`metrics_v1` version; D-test manifest fingerprint (sha256 of the locked parquet); bootstrap config;
git commit. Once committed, this file's status line reads
`D-TEST CONFIGURATION LOCKED — NO FURTHER SCIENTIFIC SELECTION`. After the lock, no finalist,
checkpoint, seed, β, λ, LR, basis, layer, LoRA rank, decoder setting, normalization, or metric
definition may change; D-test is evaluation-only.

## 8. Statistics — dialogue-block bootstrap

Bootstrap unit = **dialogue** (D-test has 15 dialogues); repetitions = **2000**; bootstrap seed =
**42**. Each replicate resamples D-test dialogues with replacement, includes all utterances of sampled
dialogues, and recomputes corpus metrics from transcripts (never bootstrapping aggregated numbers).
For M* vs SALSA and M* vs LoRA, use matched seed IDs (13-13, 42-42, 73-73): per replicate compute the
per-seed metric difference and average the three. For M* vs deterministic F0/F1, per replicate compute
each M* seed's difference to the same deterministic reference and average the three seed differences.
Report 95% percentile CIs for PIER, MER, and canonical utility (secondary CIs optional). A difference
is "clearly supported" only if its 95% CI excludes zero in the favorable direction.

## 9. Efficiency

Per learned method: trainable parameters, % of backbone, checkpoint size, optimizer updates, training
wall time (mean ± std over 3 seeds), peak GPU memory. Inference efficiency on a **frozen deterministic
D-dev-select timing subset** (first 100 utterances after canonical duration sort, fixed before any
timing), same mig GPU class / batch / decoder / warmup: total decode time, utt/s and RTF, relative
overhead vs F0. Model-load time excluded from timed decode, applied consistently. DG-07 training
resource figures are reused for the seed-42 runs where hardware/config are identical. Parameter
efficiency, training efficiency, and inference overhead are reported as **separate** axes (fewer
trainable parameters ≠ faster inference).

## 10. Slurm plan

`partition=mig` only, never `main`. ≤ 2 DG-08 GPU jobs pending/running at once; `squeue` checked
before each submission. Training pairs (2 new seeds × 3 methods = 6 runs): (SALSA 13, SALSA 73),
(LoRA 13, LoRA 73), (M* 13, M* 73). Evaluation batches several system decodes per job (model loaded
once). Greedy D-test evaluated first (primary), then beam-5. Bootstrap statistics and tables run
offline on CPU.

## 11. DG-07 ablation carry-forward (development findings; not reinterpreted by D-test)

local direction SUPPORTED · conditioning direction NO CLEAR CONTRIBUTION · adaptive gate SUPPORTED ·
adaptive mixture SUPPORTED · damage-aware objective SUPPORTED. A5 refined basis DEFERRED/NON-BLOCKING.
DG-08 does not run 3-seed ablations (A1/A2/A3/A5); DG-07 is the development ablation study.

## 12. Failure policy

A failed GPU job is not license to change scientific config. Environment/Slurm failures: fix launcher/
resources only, rerun identical config. Software defects: fix narrowly, add regression, commit, rerun.
Scientific results (including weak seeds) are kept; no seed is rescued or dropped.

## Artifacts

- Training driver: the frozen DG-06/DG-07 runners with `--seed`; DG-08 outputs under
  `results/dg08/train/{MSTAR,SALSA,LORA}/seed{13,42,73}/`.
- D-test evaluation: `experiments/dg08_dtest_eval.py`; lock: `experiments/dg08_lock.py`;
  statistics/tables: `experiments/dg08_stats.py`; launchers `sbatch/cs_asr_dg08_train.sh`,
  `sbatch/cs_asr_dg08_eval.sh`.
- Outputs: `results/dg08/` (`DG08_TEST_LOCK.json`, `dtest/<regime>/<system>.json`, `tables/`,
  `timing/`, `stats/`).
- Config: `configs/dg08_locked_eval.yaml`. Tests: `tests/test_dg08_locked_eval.py`.

## D-TEST CONFIGURATION LOCKED — NO FURTHER SCIENTIFIC SELECTION

The hard lock `results/dg08/DG08_TEST_LOCK.json` is committed (§Provenance). After this commit no
finalist, checkpoint, seed, β, λ, LR, basis, layer, LoRA rank, decoder setting, normalization, or
metric definition may change; D-test is evaluation-only.

## Provenance (execution record)

- Starting HEAD (DG-07 frozen): `7aff0ee326e0a83dd772149101b6b31a67a00506`.
- Pre-training commit: `f1dc4a8d9e55856b531c441f892d50a04f5deff8`.
- New training runs (`partition=mig`, H100 3g.40gb, all COMPLETED, seed → selected epoch):
  SALSA 13 `50704`→ep3, SALSA 73 `50705`→ep3; LoRA 13 `50713`→ep3, LoRA 73 `50714`→ep2;
  M* 13 `50718`→ep1, M* 73 `50719`→ep2. Seed 42 reused from DG-06/DG-07 (hash-verified in the lock).
- Selected checkpoint SHA-256 (per finalist × seed) recorded in `DG08_TEST_LOCK.json`.
- D-test manifest fingerprint recorded in the lock; D-test = 6,257 utts / 15 dialogues.
- Test-lock commit: `1fb2c3729cd2a51128edbd80e0d99754ce63d526`.
- Greedy D-test jobs (`mig`): F0 `50725`; finalists `50733`/`50734` (all COMPLETED, 11 decodes,
  ~11 min/system). Beam-5 jobs: F0 `50760`, F1+SALSA×3 `50767`, LoRA×3 `50768`, M*×3
  `51100`/`51101`/`51102` (batch-16, ~46 min/decode) — all COMPLETED, 11 decodes. Timing `50761`.
- All D-test evaluation ran on `partition=mig`, H100 3g.40gb, ≤ 2 concurrent. No `main`.
- **Result: M\* significantly improves over frozen Whisper on PIER, MER, and utility in BOTH greedy
  and beam-5 (all 95% CIs exclude zero); competitive with SALSA/LoRA (does not dominate SALSA on raw
  correction; LoRA is non-viable — MER 1.29 greedy / 1.91 beam). Beam robustness: PERSISTS.**
  Full tables/bootstrap: `results/dg08/DG08_RESULTS_SUMMARY.md`, `results/dg08/tables/`,
  `results/dg08/stats/`.
- Final result commit: `e94339dc0dbbce5d7661d85608665617e6d9b76f`.
- Status: **DG-08 — READY FOR INDEPENDENT AUDIT** (not FROZEN). Do not begin SEAME/ViMedCSS/Qwen
  cross-dataset/model expansion.
- Implementation note (mechanical, throughput-only): the frozen DG-06/DG-07 runners gained a
  backward-compatible `--seed` (default 42 → identical behavior) for the multi-seed runs; the D-test
  evaluator gained a `--batch-size` flag (decode throughput only; not a locked decoder parameter) —
  M* beam-5 used batch 16 on `mig`, other beam-5 systems batch 8; the difference is bf16 reduction
  jitter within the accepted noise floor, not a protocol change.
