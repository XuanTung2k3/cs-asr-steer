# AI Execution Guide: E1–E5 for Local Language Activation Steering in Code-Switched ASR

Version: 1.0  
Primary model: `openai/whisper-large-v3`  
Primary dataset: CS-Dialogue, Mandarin–English  
Hardware assumption: one NVIDIA H100 with up to 80 GB VRAM  
Scope: frozen-backbone, gradient-free feasibility experiments only  
Final-test status: **PROHIBITED during E1–E5**

---

## 0. Purpose of this guide

This document is an operational specification for an AI coding/research agent. The agent must use it to:

1. inspect or create a reproducible experiment codebase;
2. prepare CS-Dialogue without speaker leakage;
3. align reference language spans to Whisper encoder frames;
4. estimate language-associated activation directions without gradients;
5. determine whether those directions separate Mandarin and English;
6. test whether oracle-local steering corrects embedded-English errors;
7. test whether the effect remains under realistic boundary error;
8. stop early when a required feasibility gate fails.

This document covers:

- **E1 — Alignment Construction and Audit**
- **E2 — Layer-wise Activation Extraction and Direction Construction**
- **E3 — Language-Direction Separability Evaluation**
- **E4 — Oracle-Span Local Steering**
- **E5 — Mask and Boundary Robustness**

It does **not** cover:

- training a deployable LID router;
- selective two-pass inference without gold spans;
- LoRA or backbone fine-tuning;
- Qwen3-ASR replication;
- multilingual generalization;
- final test-set evaluation.

Those are later stages and must not be allowed to obscure the initial go/no-go result.

---

## 1. Central feasibility question

The agent must answer:

> Can a frozen Whisper-large-v3 model correct a meaningful subset of embedded-English code-switching errors by adding one rank-one English-associated direction to one local encoder-frame region, while causing substantially fewer new errors?

The method is successful at the E1–E5 stage only if:

1. the direction separates EN and ZH frames on unseen speakers;
2. correct-sign local steering outperforms random and wrong-sign controls;
3. oracle steering corrects more errors than it creates;
4. local steering is safer than utterance-global steering;
5. most of the gain survives approximately ±100 ms boundary uncertainty.

The agent must not claim:

- that all CS-ASR errors arise from language confusion;
- that frame-language decodability proves causal use;
- that an EN–ZH direction is free from script information;
- that oracle-span steering is a deployable system;
- that a negative E3/E4 result can be rescued merely by training a more complex router.

---

## 2. Non-negotiable experiment constraints

The agent must obey all of the following:

1. Freeze every Whisper parameter.
2. Use `torch.inference_mode()` for activation extraction and decoding.
3. Do not optimize the steering vector with gradients.
4. Construct directions from official training speakers only.
5. Select layers, strengths, masks, and thresholds using development speakers only.
6. Do not inspect or decode the official test split during E1–E5.
7. Use one target POI/candidate per utterance in the main E4/E5 analysis.
8. Cache the unmodified baseline once.
9. Use the exact same text normalization and decoding configuration across comparable systems.
10. Log the exact model revision, tokenizer revision, data manifest hash, code commit, hook position, direction source, and random seed.
11. Stop after a failed gate unless explicitly instructed to run a diagnostic.
12. Preserve all existing user code and unrelated changes.

---

## 3. Experiment names and dependency flow

| Experiment | Name | Required predecessor |
|---|---|---|
| E1 | Alignment Construction and Audit | P0 baseline prerequisite |
| E2 | Layer-wise Activation Extraction and Direction Construction | E1 |
| E3 | Language-Direction Separability Evaluation | E2 |
| E4 | Oracle-Span Local Steering | E3 pass |
| E5 | Mask and Boundary Robustness | E4 pass |

Execution logic:

```text
P0 baseline
  └── E1 alignment
        └── E2 directions
              └── E3 separability
                    ├── FAIL → stop and diagnose
                    └── PASS → E4 oracle steering
                                  ├── FAIL → stop; do not build a router
                                  └── PASS → E5 boundary robustness
                                                ├── FAIL → method is oracle-bound
                                                └── PASS → ready for E6 router work
```

The agent must maintain a machine-readable stage status:

```json
{
  "p0": "pending",
  "e1": "pending",
  "e2": "pending",
  "e3": "pending",
  "e4": "pending",
  "e5": "pending"
}
```

Allowed values are:

- `pending`
- `running`
- `passed`
- `failed`
- `blocked`

---

## 4. Dataset choice and split setup

### 4.1 Primary dataset

Use CS-Dialogue as the only primary dataset for E1–E5.

Expected official speaker-independent split statistics:

| Official split | Speakers | Utterances | Duration | E1–E5 role |
|---|---:|---:|---:|---|
| Train | 140 | 26,428 | 68.97 h | Direction construction |
| Dev | 30 | 6,196 | 18.30 h | Selection and confirmation |
| Test | 30 | 6,293 | 16.74 h | Locked; do not use |

Before implementation, verify that:

- audio is actually accessible;
- transcripts and speaker IDs are present;
- official split membership is unambiguous;
- audio paths resolve;
- sample rate and channel metadata are known.

If CS-Dialogue audio is unavailable, stop and report the blocker. Do not silently replace it. Suggested replacements, requiring an explicit decision, are:

1. licensed SEAME if the actual audio is available;
2. an open natural Mandarin–English corpus such as ASCEND;
3. CS-FLEURS only as a controlled fallback, with a clear synthetic/read-speech limitation.

### 4.2 Internal development split

The official development set must be divided by speaker:

- `dev_select`: approximately 20 speakers;
- `dev_confirm`: approximately 10 speakers.

Procedure:

1. collect unique official development speaker IDs;
2. sort them deterministically;
3. shuffle with seed 42;
4. assign the first 20 speakers to `dev_select`;
5. assign the remainder to `dev_confirm`;
6. persist the split manifest;
7. assert that speaker intersections are empty.

Use:

- `dev_select` to choose candidate layers and steering strengths;
- `dev_confirm` to confirm E4 and run E5;
- official test only in a later locked experiment.

### 4.3 Training direction subsets

Create:

- `train_direction_pilot`: 2,000–5,000 eligible bilingual utterances;
- `train_direction_full`: every eligible bilingual training utterance;
- five bootstrap/subsample manifests with seeds 42–46.

An eligible utterance:

- contains at least one aligned EN content unit;
- contains at least one aligned ZH content unit;
- has valid audio and transcript;
- passes alignment-quality checks;
- has sufficient non-boundary frames in both languages.

Sample speakers approximately evenly for the pilot. Do not let a few speakers or long utterances dominate.

### 4.4 Required manifest schema

Create a tabular manifest, preferably Parquet, with at least:

```text
utterance_id
conversation_id
speaker_id
audio_path
sample_rate
duration_sec
official_split
internal_split
transcript_raw
transcript_normalized
contains_en
contains_zh
contains_code_switch
audio_sha256
```

Create a separate word/span table:

```text
utterance_id
unit_id
surface
normalized_surface
language_tag
start_sec
end_sec
start_frame
end_frame
is_content
is_boundary_adjacent
alignment_source
alignment_confidence
baseline_status
baseline_error_type
```

Allowed `language_tag` values:

- `EN`
- `ZH`
- `OTHER`
- `UNKNOWN`

Exclude `OTHER` and `UNKNOWN` from direction construction.

---

## 5. Repository and module contract

If a codebase already exists, adapt its structure without unnecessarily moving user files. If no codebase exists, create:

```text
cs_asr_steering/
├── pyproject.toml
├── README.md
├── configs/
│   ├── base.yaml
│   ├── data/
│   │   └── cs_dialogue.yaml
│   ├── model/
│   │   └── whisper_large_v3.yaml
│   └── experiments/
│       ├── e1_alignment.yaml
│       ├── e2_directions.yaml
│       ├── e3_separability.yaml
│       ├── e4_oracle.yaml
│       └── e5_boundaries.yaml
├── src/csasr/
│   ├── data/
│   │   ├── manifest.py
│   │   ├── normalize.py
│   │   ├── language_tags.py
│   │   ├── alignment.py
│   │   └── splits.py
│   ├── models/
│   │   ├── whisper.py
│   │   ├── hooks.py
│   │   └── generation.py
│   ├── directions/
│   │   ├── accumulators.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── controls.py
│   ├── steering/
│   │   ├── masks.py
│   │   ├── encoder_hook.py
│   │   └── decoder_hook.py
│   ├── evaluation/
│   │   ├── normalization.py
│   │   ├── mer.py
│   │   ├── pier.py
│   │   ├── correction_harm.py
│   │   └── bootstrap.py
│   ├── experiments/
│   │   ├── p0_baseline.py
│   │   ├── e1_alignment.py
│   │   ├── e2_directions.py
│   │   ├── e3_separability.py
│   │   ├── e4_oracle.py
│   │   ├── e5_boundaries.py
│   │   └── pipeline.py
│   └── utils/
│       ├── hashing.py
│       ├── logging.py
│       └── seed.py
├── tests/
│   ├── test_splits.py
│   ├── test_language_tags.py
│   ├── test_frame_mapping.py
│   ├── test_hooks.py
│   ├── test_masks.py
│   ├── test_directions.py
│   ├── test_mer.py
│   └── test_pier.py
└── artifacts/
    ├── manifests/
    ├── alignments/
    ├── baselines/
    ├── directions/
    ├── predictions/
    ├── metrics/
    ├── reports/
    └── status/
```

All commands must support:

- a config file;
- a deterministic seed;
- an output directory;
- `--resume`;
- `--overwrite` only when explicitly supplied;
- a dry-run or validation mode where practical.

---

## 6. Environment and reproducibility

### 6.1 Required implementation properties

Use:

- Python 3.10–3.13 as supported by the selected packages;
- PyTorch with CUDA support;
- Hugging Face Transformers or an equivalent Whisper implementation exposing hidden states;
- BF16 inference on H100;
- float32 or float64 accumulators for centroid statistics;
- Parquet/JSONL for manifests and predictions;
- immutable configuration snapshots per run.

### 6.2 Seed policy

Use:

```text
primary seed: 42
direction stability seeds: 42, 43, 44, 45, 46
random direction seeds: 142, 143, 144, 145, 146
boundary jitter seeds: 242, 243, 244, 245, 246
bootstrap seed: 342
```

Set seeds for:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- dataset sampling.

### 6.3 Run metadata

Each run directory must include:

```text
config_resolved.yaml
environment.json
git_state.json
data_hashes.json
model_metadata.json
run.log
metrics.json
status.json
```

Record dirty working-tree state without modifying unrelated files.

---

## 7. Common model and decoding configuration

### 7.1 Primary model

```yaml
model:
  id: openai/whisper-large-v3
  revision: null
  dtype: bfloat16
  device: cuda
  freeze_backbone: true
  output_hidden_states: true
  use_cache: true
```

Resolve and record the exact revision before the first run.

### 7.2 Baseline decoding

Use deterministic decoding for initial experiments:

```yaml
decoding:
  task: transcribe
  temperature: 0.0
  do_sample: false
  num_beams: 1
  return_dict_in_generate: true
  output_scores: true
  return_timestamps: true
```

Run at least:

- `B0_AUTO`: automatic language selection;
- `B1_ZH`: global Mandarin language token.

Choose the primary baseline on `dev_select` using MER, while retaining both outputs for analysis. Once chosen, freeze it for E1–E5.

Do not mix greedy and beam decoding inside one layer/strength comparison. Beam-search replication is later.

### 7.3 Text normalization

Implement one versioned normalization pipeline:

- Unicode normalization;
- consistent simplified/traditional Chinese policy;
- lowercase English if required by the chosen metric protocol;
- punctuation normalization/removal;
- consistent whitespace;
- Chinese character segmentation;
- English word tokenization;
- preservation of named entities and acronyms.

The exact same normalization must be used for:

- baseline error detection;
- correctness filtering for direction construction;
- MER;
- PIER;
- correction/corruption evaluation.

---

## 8. P0 prerequisite — baseline and POI audit

P0 is required even though the requested numbered experiments begin at E1.

### 8.1 Tasks

1. Validate all manifests and split disjointness.
2. Decode `dev_select` and `dev_confirm` with B0 and B1.
3. Compute MER, EN-WER, ZH-CER, and PIER.
4. Align hypothesis units to reference units.
5. Mark each embedded-English POI as baseline-correct or baseline-incorrect.
6. Assign error categories.
7. Select the primary baseline.
8. Cache all outputs and token scores.

### 8.2 Error categories

Use:

```text
wrong_language_substitution
phonetic_transliteration_or_script
same_language_substitution
deletion
boundary_error
insertion_near_poi
other
```

The main E4 result should report:

- all erroneous EN POIs;
- a frozen language-confusion subset;
- each error category separately where sample size permits.

The error taxonomy is for analysis. The oracle intervention may use the gold POI span but must not change categories after seeing steering results.

### 8.3 Gate P0

Recommended minimum:

- at least several hundred erroneous EN POIs across development;
- at least approximately 100 credible language-confusion errors;
- enough baseline-correct EN POIs for matched harm controls.

If insufficient:

1. report exact counts;
2. do not fabricate a balanced sample;
3. stop and request a dataset/scope decision.

### 8.4 Outputs

```text
artifacts/baselines/B0_AUTO/
artifacts/baselines/B1_ZH/
artifacts/manifests/poi_dev_select.parquet
artifacts/manifests/poi_dev_confirm.parquet
artifacts/reports/p0_baseline_audit.md
```

---

# E1 — Alignment Construction and Audit

## 9. E1 objective

Create reliable reference word/span-to-encoder-frame mappings for:

- training-time direction construction;
- oracle-span E4 intervention;
- E5 boundary perturbation.

Gold alignment is allowed in E1–E5 because this stage estimates an upper bound. It must not later be presented as deployable inference.

## 10. E1 alignment source priority

Use:

1. dataset-provided word or language boundaries;
2. a multilingual forced aligner audited on mixed speech;
3. Whisper cross-attention/timestamp alignment;
4. manual annotation for a small diagnostic subset.

If using an external forced aligner:

- run it in a separate process from Whisper if GPU memory or dependency conflicts exist;
- record model name and revision;
- never assume mixed-language reliability merely from monolingual support.

## 11. E1 language tagging

Assign reference units to EN/ZH/OTHER/UNKNOWN.

Initial rules:

- Chinese Han characters → `ZH`;
- ordinary Latin alphabet words → usually `EN`;
- punctuation → `OTHER`;
- numerals, symbols, fillers, acronyms, romanized Mandarin, and named entities → audit rules.

Create an exception lexicon. Do not silently label every Latin token as English.

The agent must generate:

- tag counts;
- frequent ambiguous units;
- 100 random tagged examples;
- 100 switch-boundary examples.

## 12. E1 time-to-frame mapping

Do not hardcode a 20 ms stride without checking the actual model configuration.

Derive:

```python
frame_start = floor(start_sec / encoder_step_sec)
frame_end = ceil(end_sec / encoder_step_sec)
```

Then:

- clamp to valid encoder indices;
- use half-open spans `[start_frame, end_frame)`;
- assert `start_frame < end_frame`;
- exclude padding frames;
- preserve the original seconds for audit.

For direction construction, mark frames within 100 ms of a language boundary:

```text
is_boundary_adjacent = true
```

Exclude those frames from the primary direction. Retain them for E5.

## 13. E1 manual audit

Create an audit pack containing at least 100 switch boundaries:

- waveform or spectrogram image;
- playable local audio segment if supported;
- reference words around the boundary;
- predicted start/end boundaries;
- alignment source and confidence.

Sample across:

- speakers;
- short and long embedded spans;
- high and low confidence;
- utterance positions;
- both switch directions if available.

Record:

- boundary accepted/rejected;
- approximate absolute boundary error;
- percentage within 50 ms;
- percentage within 100 ms;
- systematic early/late bias.

## 14. E1 unit tests

Required:

1. split speakers do not overlap;
2. all audio paths exist;
3. all frame spans are in range;
4. padding frames are never labeled;
5. unit ordering is monotonic;
6. overlapping conflicting language spans are rejected;
7. boundary exclusion produces the expected frame counts;
8. a known synthetic 1-second example maps correctly.

## 15. E1 acceptance gate

E1 passes when:

- manifests and frame spans validate;
- at least 90% of audited boundaries are judged usable, or an equivalent documented threshold;
- most usable boundaries are within approximately ±100 ms;
- no speaker leakage exists;
- enough eligible bilingual training utterances remain.

If alignment is weak:

1. compare at least one alternative aligner;
2. test expanded windows;
3. report whether error is systematic;
4. do not proceed to claim representation failure.

## 16. E1 outputs

```text
artifacts/alignments/train_direction_pilot.parquet
artifacts/alignments/train_direction_full.parquet
artifacts/alignments/dev_select.parquet
artifacts/alignments/dev_confirm.parquet
artifacts/reports/e1_alignment_audit.md
artifacts/metrics/e1_alignment_metrics.json
artifacts/status/e1.json
```

Suggested command contract:

```bash
python -m csasr.experiments.e1_alignment \
  --config configs/experiments/e1_alignment.yaml \
  --seed 42 \
  --resume
```

---

# E2 — Layer-wise Activation Extraction and Direction Construction

## 17. E2 objective

Estimate reproducible encoder language-associated directions from training speakers without storing the complete activation corpus.

Also construct a decoder-all-layer direction baseline for use only in E4.

## 18. Exact hook convention

Primary encoder hook:

> residual-stream output after encoder block \(l\), before block \(l+1\).

Candidate encoder layers, zero-indexed:

\[
\mathcal L_{\mathrm{enc}}=\{15,23,27,31\}.
\]

The agent must inspect the actual model module names and record:

- module path;
- input shape;
- output shape;
- whether output is a tensor or tuple;
- whether layer numbering is zero- or one-indexed.

Do not silently hook attention-only or MLP-only tensors.

## 19. Hook correctness tests

Before extracting directions:

1. a forward pass with no hook equals the reference;
2. an observation-only hook leaves outputs identical;
3. an \(\alpha=0\) steering hook leaves logits and text identical;
4. an all-zero mask leaves outputs identical;
5. a full-one mask equals the global-steering implementation;
6. reversing the direction reverses the projection shift;
7. norm preservation retains frame norms within tolerance;
8. hooks are removed after each run.

Use strict equality where deterministic implementation permits; otherwise define and log numerical tolerance.

## 20. Primary encoder direction

For each eligible bilingual training utterance \(i\) and encoder layer \(l\):

\[
\mu^{(l)}_{i,\mathrm{EN}}
=
\frac{1}{|T_{i,\mathrm{EN}}|}
\sum_{t\in T_{i,\mathrm{EN}}} h^{(l)}_{i,t},
\]

\[
\mu^{(l)}_{i,\mathrm{ZH}}
=
\frac{1}{|T_{i,\mathrm{ZH}}|}
\sum_{t\in T_{i,\mathrm{ZH}}} h^{(l)}_{i,t}.
\]

Construct:

\[
\Delta_i^{(l)}
=
\mu^{(l)}_{i,\mathrm{EN}}
-
\mu^{(l)}_{i,\mathrm{ZH}},
\]

\[
d_l
=
\operatorname{normalize}
\left(
\frac{1}{N}\sum_{i=1}^{N}\Delta_i^{(l)}
\right).
\]

Every utterance receives equal weight.

### Correct-only filtering

Primary direction:

- use only reference units correctly recognized by the frozen primary baseline;
- exclude boundary-adjacent frames;
- exclude `OTHER` and `UNKNOWN`;
- require a minimum number of EN and ZH frames.

Also construct an `all_valid_spans` direction as an ablation.

## 21. Direction scale

For each layer:

\[
s_l
=
\operatorname{Std}
\left[
d_l^\top h^{(l)}_{i,t}
\right]
\]

on eligible training frames.

Store:

- unit direction \(d_l\);
- projection standard deviation \(s_l\);
- EN/ZH projected centroids;
- midpoint;
- sample counts;
- direction-estimation seed;
- data-manifest hash.

## 22. Online accumulation

Do not retain all frame states.

For each batch:

1. run one encoder forward pass;
2. observe all candidate layers;
3. pool EN and ZH frames within each utterance;
4. update per-layer sum/count/statistics in float32 or float64;
5. discard hidden states;
6. periodically checkpoint accumulators.

The accumulator must be resumable and idempotent. Track processed utterance IDs to prevent double counting.

## 23. Required encoder direction variants

Construct:

1. `within_utterance_correct_only` — primary;
2. `within_utterance_all_valid`;
3. `global_centroid_correct_only`;
4. five matched random directions per candidate layer;
5. wrong-sign direction generated as `-d_l`.

Optional only after primary completion:

- diagonal-whitened direction;
- PCA direction over \(\Delta_i^{(l)}\);
- error-versus-correct direction.

Random controls:

```python
r = torch.randn_like(d, generator=g)
r = r / r.norm()
```

Match intervention scale with the same \(s_l\). Do not multiply random vectors by a different norm.

## 24. Decoder-all-layer baseline direction

This baseline is required for E4 analysis but is not the primary method.

### 24.1 Activation collection

Use teacher forcing on correctly transcribed training examples.

For every decoder layer \(k\):

- identify the hidden state that produces the logit for reference token \(y_j\);
- assign the language label of \(y_j\) to that state;
- exclude special prefix tokens, language/task tokens, timestamp tokens, EOS, and punctuation;
- mean-pool EN and ZH token states within the same utterance.

The agent must explicitly verify the decoder shift convention. Do not assume decoder input position \(j\) predicts token \(y_j\) without a unit test.

Construct:

\[
d_k^{\mathrm{dec}}
=
\operatorname{normalize}
\left[
\frac{1}{N}
\sum_i
\left(
\mu^{(k)}_{i,\mathrm{EN}}
-
\mu^{(k)}_{i,\mathrm{ZH}}
\right)
\right].
\]

Construct one direction for every decoder layer. Never reuse an encoder direction in the decoder or one decoder layer’s direction in another layer.

### 24.2 Decoder baseline interpretation

`DEC_ALL_GLOBAL` tests whether a broad decoder language/script bias can produce the apparent improvement.

It is expected to be less selective and may damage Mandarin output. It is a confound/control, not the principal method.

## 25. E2 artifact schema

Each direction file must contain:

```python
{
    "model_id": str,
    "model_revision": str,
    "module": "encoder" | "decoder",
    "layer_index": int,
    "hook_location": str,
    "direction_type": str,
    "direction": Tensor,
    "direction_norm": float,
    "projection_std": float,
    "centroid_en_projection": float,
    "centroid_zh_projection": float,
    "midpoint_projection": float,
    "num_utterances": int,
    "num_en_frames_or_tokens": int,
    "num_zh_frames_or_tokens": int,
    "seed": int,
    "manifest_hash": str,
    "normalization_version": str
}
```

## 26. E2 acceptance gate

E2 passes when:

- primary directions exist for all candidate encoder layers;
- all direction tensors are finite and unit norm;
- estimates are reproducible across seeds;
- per-language sample counts are adequate;
- decoder per-layer directions exist for the E4 baseline;
- all hook and decoder-shift tests pass;
- no development/test examples contributed to direction construction.

E2 does not require good separability; that is tested in E3.

## 27. E2 outputs

```text
artifacts/directions/encoder/layer_15/
artifacts/directions/encoder/layer_23/
artifacts/directions/encoder/layer_27/
artifacts/directions/encoder/layer_31/
artifacts/directions/decoder/all_layers/
artifacts/reports/e2_direction_construction.md
artifacts/metrics/e2_direction_statistics.json
artifacts/status/e2.json
```

Suggested command:

```bash
python -m csasr.experiments.e2_directions \
  --config configs/experiments/e2_directions.yaml \
  --seeds 42 43 44 45 46 \
  --resume
```

---

# E3 — Language-Direction Separability Evaluation

## 28. E3 objective

Determine whether the training-derived encoder direction separates EN and ZH frames on unseen development speakers.

E3 is a diagnostic gate. It does not yet demonstrate that intervention improves ASR.

## 29. Evaluation data

Use `dev_select` only.

Exclude:

- padding;
- `OTHER` and `UNKNOWN`;
- boundary-adjacent frames in the primary result.

Report an additional boundary-inclusive result.

To reduce imbalance:

- report frame-micro metrics;
- report utterance-macro metrics;
- report speaker-macro metrics;
- optionally subsample equal EN/ZH frames per utterance.

## 30. Projection score

For layer \(l\):

\[
a_t^{(l)}
=
d_l^\top h_t^{(l)}.
\]

Higher scores should correspond to English.

Use the training-derived midpoint as the default threshold:

\[
b_l
=
\frac{1}{2}
\left(
\bar a_{\mathrm{EN}}^{\mathrm{train}}
+
\bar a_{\mathrm{ZH}}^{\mathrm{train}}
\right).
\]

Do not fit a high-capacity classifier in E3.

## 31. E3 metrics

For every candidate layer and direction seed:

- AUROC;
- AUPRC;
- macro F1;
- EN precision/recall/F1;
- ZH precision/recall/F1;
- balanced accuracy;
- projected centroid gap;
- standardized effect size;
- boundary versus non-boundary performance;
- speaker-bootstrap 95% confidence interval.

Controls:

- global-centroid direction;
- all-valid-spans direction;
- five random directions;
- wrong sign, verifying AUROC transformation;
- optional shuffled language labels.

## 32. Layer selection

Rank layers primarily by:

1. speaker-macro AUROC;
2. stability across direction seeds;
3. EN recall;
4. narrow confidence interval;
5. low speaker variance.

Keep the best two layers for the E4 pilot.

Do not choose the final intervention layer solely from E3. E4 correction-versus-harm determines the final layer.

## 33. E3 gate

Recommended pass:

- at least one layer has held-out AUROC around 0.80 or better;
- standard deviation across direction seeds is no more than approximately 0.03;
- the primary direction clearly exceeds matched random controls;
- sign behaves consistently;
- results are not driven by one or two speakers.

Interpretation:

- `AUROC >= 0.80`: pass;
- `0.70 <= AUROC < 0.80`: exploratory/conditional; inspect alignment and direction variants;
- `AUROC < 0.70`: fail unless a documented alignment defect explains the result.

If E3 fails:

1. check E1 alignment;
2. compare correct-only and all-valid directions;
3. inspect per-layer projection distributions;
4. test neighboring layers;
5. stop before large E4 sweeps if no direction is credible.

## 34. E3 outputs

```text
artifacts/metrics/e3_layer_metrics.parquet
artifacts/metrics/e3_summary.json
artifacts/reports/e3_separability.md
artifacts/reports/e3_selected_layers.json
artifacts/status/e3.json
```

Suggested command:

```bash
python -m csasr.experiments.e3_separability \
  --config configs/experiments/e3_separability.yaml \
  --resume
```

---

# E4 — Oracle-Span Local Steering

## 35. E4 objective

Test whether adding the correct EN-associated direction to a gold embedded-English frame span improves its transcription.

E4 is the decisive go/no-go experiment.

## 36. E4 evaluation groups

From the frozen primary baseline on `dev_select`, build:

### Pilot

- 100 utterances with one baseline-incorrect EN POI;
- 100 utterances with one baseline-correct EN POI.

### Confirmation

- up to 300 baseline-incorrect EN POIs;
- a matched set of up to 300 baseline-correct EN POIs.

Use one target POI per utterance.

Match correct controls approximately on:

- speaker distribution;
- POI duration;
- token frequency;
- POI position;
- boundary proximity;
- baseline confidence;
- utterance duration.

Persist selected IDs before running steering.

## 37. Primary encoder intervention

For target region \(C\), layer \(l\), and unit direction \(d_l\):

\[
\bar h_t^{(l)}
=
h_t^{(l)}
+
\alpha s_l g_t(C)d_l.
\]

Apply norm preservation:

\[
\tilde h_t^{(l)}
=
\frac{\bar h_t^{(l)}}
{\|\bar h_t^{(l)}\|_2+\epsilon}
\|h_t^{(l)}\|_2.
\]

Use \(\epsilon\) only for numerical safety.

For E4 primary:

- use an exact gold span;
- add a tapered 100 ms shoulder unless E4 explicitly compares hard versus tapered;
- steer one encoder layer;
- steer one target POI.

## 38. E4 pilot grid

Use the top two E3 layers:

\[
\alpha_{\mathrm{enc}}\in\{0.5,1.0,2.0\}.
\]

Run:

```text
2 layers × 3 strengths × 200 pilot utterances
= approximately 1,200 steered decodes
```

The baseline is cached and must not be decoded again.

Select up to two promising configurations for refinement.

## 39. E4 refinement grid

For the best one or two layers:

\[
\alpha_{\mathrm{enc}}
\in
\{0.25,0.5,0.75,1.0,1.5,2.0\}.
\]

Use the larger confirmation subset.

## 40. Required E4 systems

| ID | System | Description |
|---|---|---|
| BASE | No steering | Cached frozen baseline |
| ENC-LOCAL-1L | Primary | Correct-sign direction, one encoder layer, gold local span |
| ENC-GLOBAL-1L | Location control | Same direction/layer over all valid encoder frames |
| ENC-WRONG-1L | Sign control | \(-d_l\) on the gold local span |
| ENC-RAND-1L-S1..S5 | Perturbation control | Five matched random directions |
| DEC-ALL-GLOBAL | Decoder confound | Per-layer decoder directions applied globally during generation |

Optional after required systems:

- decoder single-layer steering;
- decoder-all local token-position steering;
- encoder multi-layer steering;
- script prompt-derived direction.

## 41. Decoder-all-layer steering implementation

At every decoder layer \(k\), during each content-generation step:

\[
\bar h_{k,j}
=
h_{k,j}
+
\frac{\alpha_{\mathrm{dec}}}{\sqrt{K}}
s_k^{\mathrm{dec}}d_k^{\mathrm{dec}}.
\]

Then preserve the norm of \(h_{k,j}\).

Use:

\[
\alpha_{\mathrm{dec}}
\in
\{0.05,0.1,0.2,0.4,0.8\}.
\]

Requirements:

- use a distinct direction for each decoder layer;
- do not steer the fixed initial special-token prefix;
- steer the last-token activation used for each next-token prediction;
- remove hooks after generation;
- report matrix-language corruption prominently.

`DEC-ALL-GLOBAL` is not expected to be selective. It tests whether an apparent gain is merely a decoder-wide language/script shift.

## 42. E4 metrics

Let \(E_0\) be baseline-incorrect POIs and \(C_0\) baseline-correct POIs.

\[
\mathrm{CorrectionRate}
=
\frac{
|\{j\in E_0:\text{steered output is correct at }j\}|
}{
|E_0|
}.
\]

\[
\mathrm{CorruptionRate}
=
\frac{
|\{j\in C_0:\text{steered output becomes wrong at }j\}|
}{
|C_0|
}.
\]

Report:

- MER;
- PIER;
- EN-WER;
- ZH-CER;
- CorrectionRate;
- CorruptionRate;
- correction-to-corruption ratio;
- net corrected POIs;
- outside-region edit rate;
- percentage with any transcript change;
- error-category-specific correction;
- per-speaker results;
- wall time and peak memory.

For outside-region edits, align baseline and steered hypotheses to the reference and exclude the target POI plus one neighboring unit on each side.

## 43. E4 configuration selection

On `dev_select`, define a feasible configuration as:

- correct-sign result exceeds the mean random control;
- correction-to-corruption ratio is greater than 2;
- overall MER degradation is no more than +0.2 absolute;
- outside-region edits are acceptably rare.

Among feasible configurations:

1. prefer the lowest PIER;
2. break ties using higher net corrections;
3. then lower CorruptionRate;
4. then smaller \(\alpha\);
5. then a single later encoder layer for simplicity.

Freeze the selected layer and strength before `dev_confirm`.

## 44. E4 statistical analysis

Use paired bootstrap resampling by utterance or conversation:

- 10,000 resamples if computationally cheap;
- 95% confidence intervals;
- paired difference in PIER and MER;
- paired correction/harm difference.

Random controls should report mean and dispersion over five seeds.

## 45. E4 gate

Recommended pass:

1. at least approximately 5% relative PIER improvement or another clearly meaningful POI correction gain;
2. correction-to-corruption ratio greater than 2;
3. correct-sign steering outperforms wrong-sign steering;
4. correct-sign steering exceeds matched random controls;
5. local steering has less collateral damage than global encoder and global decoder steering;
6. no meaningful matrix-language degradation outside the region.

If E4 fails:

- do not proceed to train a router;
- inspect whether the issue is layer, strength, mask, or direction;
- report the negative result honestly;
- optionally run a small neighboring-layer diagnostic;
- stop large-scale work if no configuration passes.

## 46. E4 outputs

```text
artifacts/manifests/e4_pilot_wrong.parquet
artifacts/manifests/e4_pilot_correct.parquet
artifacts/manifests/e4_confirm_wrong.parquet
artifacts/manifests/e4_confirm_correct.parquet
artifacts/predictions/e4/<system>/<config>/
artifacts/metrics/e4_all_runs.parquet
artifacts/metrics/e4_selected_config.json
artifacts/reports/e4_oracle_steering.md
artifacts/status/e4.json
```

Suggested commands:

```bash
python -m csasr.experiments.e4_oracle \
  --config configs/experiments/e4_oracle.yaml \
  --phase pilot \
  --resume
```

```bash
python -m csasr.experiments.e4_oracle \
  --config configs/experiments/e4_oracle.yaml \
  --phase confirm \
  --resume
```

---

# E5 — Mask and Boundary Robustness

## 47. E5 objective

Determine whether the selected E4 intervention remains useful when the target span is not perfectly aligned.

E5 must use:

- the frozen selected encoder layer;
- the frozen selected \(\alpha\);
- `dev_confirm`;
- no official test examples.

## 48. Required E5 masks

Let the aligned core span be:

\[
C=[t_s,t_e).
\]

Test:

1. `EXACT_HARD`: exact binary core span;
2. `EXACT_TAPER`: exact core with tapered shoulders;
3. `EXPAND_50`: expand both sides by 50 ms;
4. `EXPAND_100`: expand both sides by 100 ms;
5. `EXPAND_200`: expand both sides by 200 ms;
6. `JITTER_50`: independently jitter start/end within ±50 ms;
7. `JITTER_100`: independently jitter start/end within ±100 ms;
8. `JITTER_200`: independently jitter start/end within ±200 ms;
9. `BOUNDARY_ONLY`: steer only neighborhoods around the two switch boundaries;
10. `WHOLE_WORD_PLUS_CONTEXT`: POI plus one neighboring unit on each side.

Clip all masks to valid non-padding frames.

## 49. Tapered mask

Use a triangular or Hann shoulder.

Example:

```text
0 outside expanded region
smoothly rises from 0 to 1 before the core
1 inside the core
smoothly falls from 1 to 0 after the core
```

The shoulder duration should be a config value, initially 100 ms.

Test mask functions independently with known spans and plot at least ten examples.

## 50. Boundary jitter protocol

For each jitter level:

- generate five deterministic jitter versions using seeds 242–246;
- preserve `start < end`;
- clip to valid frames;
- store actual sampled offsets;
- report mean and variance.

Expansion and jitter answer different questions:

- expansion tests uncertainty handled by a wider mask;
- jitter tests mislocalized boundaries.

Do not describe expansion alone as alignment error.

## 51. E5 evaluation groups

Use `dev_confirm`:

- baseline-incorrect POIs;
- matched baseline-correct POIs;
- optional monolingual ZH/EN utterances for full-mask harm controls.

Do not reselect examples based on whether steering succeeds.

## 52. E5 metrics

Report the same primary metrics as E4:

- PIER;
- MER;
- EN-WER;
- ZH-CER;
- CorrectionRate;
- CorruptionRate;
- outside-region edit rate;
- net corrections.

Additionally report:

- percentage of exact-mask gain retained;
- performance versus boundary error magnitude;
- variance across jitter seeds;
- hard-versus-tapered harm difference.

Define retained gain:

\[
\mathrm{RetainedGain}_{m}
=
\frac{
\mathrm{PIER}_{\mathrm{base}}-\mathrm{PIER}_{m}
}{
\mathrm{PIER}_{\mathrm{base}}-\mathrm{PIER}_{\mathrm{exact}}
}.
\]

Handle a zero or negative exact gain explicitly rather than dividing silently.

## 53. E5 gate

Recommended pass:

- approximately 70% or more of exact-span PIER gain remains at ±100 ms;
- correction-to-corruption remains favorable;
- tapered masks have equal or lower harm than hard masks;
- outside-region edits remain low;
- results are stable across jitter seeds.

If exact steering succeeds but ±100 ms fails:

- mark the method as oracle-bound;
- do not claim practical localization robustness;
- consider wider soft masks or a more accurate aligner before E6.

## 54. E5 outputs

```text
artifacts/predictions/e5/<mask>/<seed>/
artifacts/metrics/e5_mask_results.parquet
artifacts/metrics/e5_summary.json
artifacts/reports/e5_boundary_robustness.md
artifacts/status/e5.json
```

Suggested command:

```bash
python -m csasr.experiments.e5_boundaries \
  --config configs/experiments/e5_boundaries.yaml \
  --resume
```

---

## 55. H100 execution strategy

One H100 80 GB is sufficient.

### General rules

- use BF16 model weights;
- use float32/float64 statistics;
- begin with conservative dynamic batches;
- batch by total audio frames, not only utterance count;
- cache baseline predictions;
- never retain gradients;
- release unused hooks and tensors;
- empty caches between model/aligner stages only when useful;
- do not load a large aligner and Whisper simultaneously unless necessary.

### Suggested initial batch strategy

Start with:

- activation extraction: 8–16 short utterances per batch;
- generation: 4–8 utterances per batch;
- reduce batch size for long audio;
- increase only after measuring peak memory.

Do not optimize for maximum VRAM usage before correctness tests pass.

### Avoid terabyte-scale activation caches

Store:

- per-utterance pooled EN/ZH means if needed;
- online sums/counts;
- projection summaries;
- selected local windows for debugging.

Do not store:

- all frames;
- all 32 encoder layers;
- all training utterances in full precision.

---

## 56. Pilot and confirmation schedule

### Pilot

Run:

1. P0 on `dev_select`;
2. E1 audit of 100 boundaries;
3. E2 on 2,000–5,000 training utterances;
4. E3 on 500–1,000 `dev_select` utterances;
5. E4 on 100 wrong + 100 correct POIs;
6. stop or continue based on gates.

### Confirmation

After pilot pass:

1. E2 full direction accumulation;
2. repeat E3 direction stability;
3. E4 refinement on up to 300 wrong + 300 correct POIs;
4. confirm selected configuration on `dev_confirm`;
5. run E5.

Do not scale every configuration. Reduce the search after each gate.

---

## 57. Pipeline orchestration

Provide:

```bash
python -m csasr.experiments.pipeline \
  --config configs/base.yaml \
  --from-stage e1 \
  --to-stage e5 \
  --stop-on-failed-gate \
  --resume
```

The pipeline must:

1. validate prerequisites;
2. acquire a stage lock;
3. snapshot resolved configuration;
4. set status to `running`;
5. run the stage;
6. validate expected artifacts;
7. compute gate decision;
8. set status to `passed` or `failed`;
9. stop after failure;
10. preserve all completed outputs.

Every stage must also be independently runnable.

---

## 58. Configuration example

```yaml
experiment:
  name: csasr_e1_e5
  seed: 42
  output_root: artifacts
  prohibit_test_split: true

data:
  dataset: cs_dialogue
  root: /PATH/TO/CS_DIALOGUE
  manifest: artifacts/manifests/cs_dialogue.parquet
  dev_select_speakers: 20
  split_seed: 42
  direction_pilot_utterances: 5000

model:
  id: openai/whisper-large-v3
  revision: null
  dtype: bfloat16
  device: cuda
  freeze_backbone: true

decoding:
  task: transcribe
  temperature: 0.0
  do_sample: false
  num_beams: 1
  return_timestamps: true
  output_scores: true

alignment:
  source: auto
  exclude_boundary_ms: 100
  audit_boundaries: 100

directions:
  encoder_layers: [15, 23, 27, 31]
  variants:
    - within_utterance_correct_only
    - within_utterance_all_valid
    - global_centroid_correct_only
  seeds: [42, 43, 44, 45, 46]
  random_seeds: [142, 143, 144, 145, 146]

e3:
  pass_auroc: 0.80
  conditional_auroc: 0.70
  max_seed_std: 0.03
  keep_top_layers: 2

e4:
  pilot_wrong: 100
  pilot_correct: 100
  confirm_wrong: 300
  confirm_correct: 300
  encoder_alphas_pilot: [0.5, 1.0, 2.0]
  encoder_alphas_refine: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
  decoder_alphas: [0.05, 0.1, 0.2, 0.4, 0.8]
  norm_preserve: true
  primary_mask: tapered
  shoulder_ms: 100
  min_relative_pier_gain: 0.05
  min_correction_corruption_ratio: 2.0
  max_mer_degradation_abs: 0.2

e5:
  expand_ms: [50, 100, 200]
  jitter_ms: [50, 100, 200]
  jitter_seeds: [242, 243, 244, 245, 246]
  min_retained_gain_at_100ms: 0.70
```

The agent must replace `/PATH/TO/CS_DIALOGUE` only after resolving the actual dataset root.

---

## 59. Required result tables

### E3

| Layer | Direction | AUROC | AUPRC | Macro F1 | EN Recall | Seed SD | Random AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|

### E4 primary

| System | Layer | Alpha | PIER | MER | Correction | Corruption | Corr/Harm | Outside edits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### E4 location/confound

| System | Target locality | PIER change | ZH-CER change | Outside edits | Interpretation |
|---|---|---:|---:|---:|---|

### E5

| Mask | Error magnitude | PIER | Retained gain | Correction | Corruption | Outside edits |
|---|---:|---:|---:|---:|---:|---:|

---

## 60. AI decision and reporting protocol

After every stage, the AI agent must report:

1. what was run;
2. exact data subset;
3. artifacts produced;
4. validation tests;
5. metrics;
6. gate decision;
7. next allowed action;
8. any deviation from this guide.

The agent must pause for user direction when:

- dataset access is missing;
- official splits cannot be verified;
- speaker IDs are absent;
- alignment audit fails;
- E3 is below the conditional threshold;
- E4 fails the correction-versus-harm gate;
- required package/model access needs new authorization;
- a change would expand scope to model training or another dataset.

The agent may autonomously:

- fix implementation bugs;
- reduce batch size after OOM;
- resume incomplete stages;
- add unit tests;
- improve logging;
- rerun a failed command with the same scientific configuration.

The agent must not autonomously:

- use test data;
- change the main model;
- substitute a dataset;
- tune thresholds on `dev_confirm` and then call it independent confirmation;
- remove failed control results;
- reinterpret decoder script shifts as language correction;
- train the backbone.

---

## 61. Failure diagnosis map

| Failure | Likely causes | First actions |
|---|---|---|
| E1 poor boundaries | aligner not CS-capable; wrong time stride | audit stride; compare aligner; expand masks |
| E2 unstable direction | few EN frames; speaker imbalance; boundary contamination | equal-weight utterances; filter boundaries; increase data |
| E3 low AUROC | no linear direction; bad alignment; wrong hook | verify hook; inspect distributions; test neighboring layers |
| E3 high but E4 no correction | language is decodable but not intervention-effective | vary layer/alpha; inspect error types; do not claim causal control |
| E4 high corruption | alpha too large; mask too broad; global bias | lower alpha; taper mask; change layer |
| Random equals correct direction | generic perturbation or decoding instability | verify deterministic baseline; match scales; inspect sign |
| Decoder-all looks strong | likely global language/script forcing | inspect ZH harm and outside edits; keep as confound |
| E5 exact works, jitter fails | alignment-sensitive/oracle-bound method | wider taper; better boundary detector; restrict claims |
| OOM | retained hidden states; excessive generation batch | detach/discard states; reduce batch; split model stages |

---

## 62. Completion criteria for E1–E5

The E1–E5 package is complete only when:

- [ ] official train/dev/test manifests are validated;
- [ ] test usage is mechanically blocked;
- [ ] `dev_select` and `dev_confirm` are speaker-disjoint;
- [ ] baseline predictions and POI taxonomy are frozen;
- [ ] E1 audit and frame mappings pass;
- [ ] encoder directions exist for layers 15, 23, 27, and 31;
- [ ] five direction-estimation seeds are evaluated;
- [ ] decoder directions exist for every decoder layer;
- [ ] E3 metrics and random controls are reported;
- [ ] top two E3 layers are frozen for the E4 pilot;
- [ ] E4 includes wrong-sign, random, global-encoder, and global-decoder controls;
- [ ] correction and corruption are both reported;
- [ ] one E4 configuration is selected or the idea is stopped;
- [ ] E5 includes expansion and true boundary jitter;
- [ ] E5 reports retained gain at 100 ms;
- [ ] every run is reproducible from saved configs;
- [ ] final report clearly states pass/fail and limitations.

---

## 63. Final deliverables expected from the AI agent

The AI agent must produce:

1. working E1–E5 code;
2. unit and integration tests;
3. resolved configuration files;
4. immutable data split manifests;
5. alignment audit pack;
6. encoder and decoder direction artifacts;
7. E3 layer-selection report;
8. E4 oracle-steering report;
9. E5 boundary-robustness report;
10. one concise executive decision:
    - `GO: build E6 router`;
    - `CONDITIONAL: fix specified issue`;
    - `STOP: core intervention not supported`.

---

## 64. Master instruction to give an AI coding agent

Copy the following instruction together with this file:

> Read `CS_ASR_E1_E5_AI_Execution_Guide.md` completely before changing code. Inspect the existing repository, environment instructions, and data availability first. Implement a resumable, testable pipeline for P0 and E1–E5 using frozen `openai/whisper-large-v3` and CS-Dialogue. Preserve official speaker splits, subdivide development speakers into `dev_select` and `dev_confirm`, and mechanically prohibit access to the official test split. Use training speakers only to estimate within-utterance EN-minus-ZH directions at encoder layers 15, 23, 27, and 31. Construct separate per-layer decoder directions for the `DEC-ALL-GLOBAL` E4 control. Run each experiment sequentially and enforce the documented gates. Do not proceed to E4 if E3 fails, and do not proceed to E5 if E4 fails. Use BF16 inference on one H100, online activation accumulation, cached baselines, deterministic configs, and complete artifact logging. Never fine-tune the Whisper backbone. At each checkpoint, report evidence, pass/fail status, and the next permitted action. Do not silently change the dataset, model, split, hook location, metrics, or thresholds.

---

## 65. References

- [CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition](https://arxiv.org/abs/2502.18913)
- [Whisper](https://github.com/openai/whisper)
- [Activation Steering for Accent Adaptation in Large Audio Language Models](https://arxiv.org/abs/2603.05813)
- [Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration](https://arxiv.org/abs/2601.02906)
- [SALSA: Speech Aware LLM Adaptation via Learned Steering Activation Vectors](https://arxiv.org/abs/2606.00460)
- [Reducing Language Confusion for Code-Switching Speech Recognition with Token-Level Language Diarization](https://arxiv.org/abs/2210.14567)
- [Aligning Speech to Languages to Enhance Code-Switching Speech Recognition](https://arxiv.org/abs/2403.05887)
- [PIER: A Novel Metric for Evaluating What Matters in Code-Switching Speech Recognition](https://arxiv.org/abs/2501.09512)

