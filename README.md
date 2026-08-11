# CS-ASR Local Language Activation Steering — P0 & E1–E5

Frozen-backbone, gradient-free feasibility study: can `whisper-large-v3` correct
embedded-English code-switching errors by adding **one rank-one English-associated
direction** to **one local encoder-frame region**, while causing substantially
fewer new errors?

Implements `docs/CS_ASR_E1_E5_AI_Guide.md`. **The official test split is never
touched during E1–E5** — it is filtered out at manifest load and re-checked by
`assert_no_test_data()` on every code path.

---

## Resolved environment

| Item | Value |
|---|---|
| Code | `/home/tungnx/cs-asr-steer` |
| Generated data | `/mnt/data/tungnx/cs-asr-steer` (artifacts, logs, cache) |
| Python env | `/home/tungnx/miniconda3/envs/acl1` (3.11, torch 2.10, transformers 4.57.6) |
| Model | `/mnt/data/tungnx/whisper-large-v3` (local, offline) |
| Dataset | `/mnt/data/tungnx/CS-Dialogue` (`short_wav` release) |
| Cluster | Slurm, partition `main`, H100 80 GB |

Derived from the actual model config, not hardcoded: 32 encoder / 32 decoder
layers, `d_model` 1280, **encoder step 0.020 s**, 1500 max encoder frames.

### Verified split statistics

| Split | Speakers | Utterances | Hours | Role |
|---|---:|---:|---:|---|
| train | 140 | 26,239 | 68.6 | direction construction |
| dev_select | 20 | 3,360 | 11.5 | layer/strength selection |
| dev_confirm | 10 | 2,826 | 6.8 | E4 confirmation, E5 |
| test | 30 | 6,257 | — | **locked** |

Speaker-disjointness across all three internal splits is asserted, not assumed.

---

## Running

```bash
sbatch cs_asr_e1_e5.sh              # full P0..E5, stops at the first failed gate
sbatch cs_asr_e1_e5.sh tests        # unit tests only
sbatch cs_asr_e1_e5.sh p0 e1 e2     # selected stages, in order
sbatch cs_asr_e1_e5.sh e4_pilot
```

Plumbing check before committing GPU hours (writes to a separate
`artifacts_smoke/` root, gates are expected to fail at that sample size):

```bash
srun --partition=main --gres=gpu:1 --cpus-per-task=6 --mem=64G --time=01:30:00 \
     bash /mnt/data/tungnx/cs-asr-steer/tmp/smoke_full.sh
```

Env overrides: `LIMIT=100`, `OVERWRITE=1`, `SEED=42`, `BUILD_MANIFEST=1`.

### Individual stages

```bash
python -m csasr.experiments.p0_baseline   --config configs/experiments/e1_alignment.yaml --build-manifest --resume
python -m csasr.experiments.e1_alignment  --config configs/experiments/e1_alignment.yaml --seed 42 --resume
python -m csasr.experiments.e2_directions --config configs/experiments/e2_directions.yaml --seeds 42 43 44 45 46 --resume
python -m csasr.experiments.e3_separability --config configs/experiments/e3_separability.yaml --resume
python -m csasr.experiments.e4_oracle     --config configs/experiments/e4_oracle.yaml --phase pilot --resume
python -m csasr.experiments.e5_boundaries --config configs/experiments/e5_boundaries.yaml --resume
```

Every command supports `--config --seed --output-dir --resume --overwrite
--dry-run --limit --set key=value`. Exit code `2` means *gate failed* (a valid
scientific result); `1` means the stage crashed.

### Status and orchestration

```bash
python -m csasr.experiments.pipeline --config configs/base.yaml --status
python -m csasr.experiments.pipeline --config configs/base.yaml \
    --from-stage e1 --to-stage e5 --stop-on-failed-gate --resume
```

---

## Stage map

| Stage | What it does | Gate |
|---|---|---|
| **P0** | manifest, splits, `B0_AUTO`/`B1_ZH` baselines, POI taxonomy, primary baseline frozen | ≥300 erroneous EN POIs, ≥100 language-confusion, ≥300 correct POIs |
| **E1** | reference-unit → encoder-frame alignment, audit pack | spans validate; ≥90 % boundaries usable; enough bilingual training utterances |
| **E2** | online EN−ZH direction accumulation at layers 15/23/27/31 + per-decoder-layer control directions | directions finite/unit-norm, stable across 5 seeds, hook + decoder-shift tests pass |
| **E3** | projection separability on unseen dev speakers | best AUROC ≥ 0.80, seed SD ≤ 0.03, beats random, sign consistent |
| **E4** | oracle-span local steering vs. wrong-sign / random / global-encoder / global-decoder controls | ≥5 % relative PIER gain, correction:corruption > 2, beats all controls |
| **E5** | mask and boundary robustness (expand / jitter / taper) | ≥70 % of exact-span gain retained at ±100 ms |

A failed gate stops the pipeline; completed outputs are preserved and every
stage resumes at utterance granularity.

---

## Key implementation decisions

**Alignment source.** CS-Dialogue's `long_wav` TextGrids are *not* present in the
local release, so dataset-provided word boundaries (guide §10 priority 1) are
unavailable. E1 uses priority 3: Whisper cross-attention DTW, teacher-forced on
the reference, restricted to the published `alignment_heads`. Reliability is
*measured*, not assumed — every switch boundary is aligned twice under two
independent configurations (zh prefix/median 7 vs en prefix/median 3) and the
gate is stated on their agreement. An audit pack (spectrograms, clips, context)
is emitted; drop a filled `verdicts.csv` next to `verdicts_template.csv` and
rerun E1 to replace the proxy with human judgements.

**Hook convention.** The residual-stream *output* of block `l`
(`model.model.encoder.layers[l]`, zero-indexed) — recorded in
`model_metadata.json` after inspecting the real module tree. Decoder directions
are collected through the *same* hooks used for steering, so a direction always
lives in the space it will be added to.

**No activation corpus.** Per layer we accumulate only the running sum of
per-utterance (μ_EN − μ_ZH), global centroid sums, and first/second frame
moments (D and D×D, float64) — enough to recover `s_l = sqrt(dᵀ Cov d)` exactly
in a single pass. Accumulators checkpoint and are idempotent on resume.

**Equal utterance weighting.** A 100-frame utterance and a 2-frame utterance
contribute equally to the direction (unit-tested).

**Steering.** `h_t ← h_t + α·s_l·g_t·d_l` with norm preservation. Zero-gain
positions are returned bit-identically and `α=0` is a hard no-op, so the
"changes nothing" tests are exact rather than approximate.

**Normalization.** One versioned pipeline (`v1`) used everywhere: baseline error
detection, correct-only filtering, MER, PIER and correction/corruption. Chinese
is scored per character, English per word. Not every Latin token is English —
fillers (`uh`), romanized Mandarin (`weibo`), bare numerals and stray letters are
tagged `OTHER`/`UNKNOWN` and excluded from direction construction.

---

## Layout

```
configs/            base + data/model/experiment configs (paths already resolved)
src/csasr/
  data/             manifest, normalize, language_tags, alignment, splits
  models/           whisper (frozen loader), hooks, hook_tests, generation
  directions/       accumulators, encoder, decoder, controls
  steering/         masks, encoder_hook, decoder_hook
  evaluation/       normalization, mer, pier, correction_harm, bootstrap
  experiments/      p0_baseline, e1..e5, pipeline
  utils/            config, hashing, logging, seed, status
tests/              8 test modules (60 tests, CPU-only, tiny random model)
cs_asr_e1_e5.sh     Slurm entry point
```

Artifacts (all under `/mnt/data/tungnx/cs-asr-steer/artifacts`):
`manifests/ alignments/ baselines/ directions/ predictions/ metrics/ reports/
status/ audit/ runs/`. Each run directory carries `config_resolved.yaml`,
`environment.json`, `git_state.json`, `model_metadata.json`, `run.log`,
`metrics.json`, `status.json`.

---

## Tests

```bash
LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH \
  /home/tungnx/miniconda3/envs/acl1/bin/python -m pytest -q
```

The `LD_LIBRARY_PATH` prefix is required on this machine: the system
`libstdc++` predates the `CXXABI_1.3.15` that scipy's compiled extensions need.
`cs_asr_e1_e5.sh` sets it automatically.

Covered: split disjointness and test-split blocking · normalization and language
tagging · frame mapping, padding, boundary trim, DTW · all eight hook-correctness
requirements · mask exactness/taper/expand/jitter/clipping · equal utterance
weighting, projection statistics, accumulator resumability, random/wrong-sign
controls · MER and per-language breakdown · POI taxonomy, PIER, correction and
corruption accounting.

---

## Scope discipline

Not implemented here, deliberately: LID router training, two-pass inference
without gold spans, LoRA or backbone fine-tuning, Qwen3-ASR replication,
multilingual generalization, and any test-set evaluation. Oracle-span steering
is an **upper bound**, not a deployable system — E4/E5 results must not be
presented as one.
