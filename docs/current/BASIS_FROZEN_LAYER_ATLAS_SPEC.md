# BASIS-A2 — Frozen All-Layer Steering Response Atlas

Status: frozen pre-run exploratory/mechanistic addendum. This document does
not reopen DG-00–DG-08 or BASIS-A, and does not select a production layer.

## Scope and data

- Model: Whisper-large-v3, frozen, bf16; decoder layers 0–31 are included as
  exploratory atlas layers only.
- Site: `decoder_post_cross_attn_residual`, the post-cross-attention residual
  before the FFN, using `DecoderPostCrossAttnInterventionHook` and
  `NormPreserve`; no depth rescaling.
- Evaluation: exactly the frozen 10-utterance D-dev-select panel in
  `results/basis_frozen_layer_atlas/panel.json`. Panel selection uses baseline
  POI metadata only and has quota 4 baseline-wrong embedded POIs, 3
  baseline-correct embedded cases, and 3 mixed/challenging cases.
- Direction construction: each layer independently uses the canonical DG-03
  aggregate-then-residualize definitions from D-construct. The L24 vectors are
  never copied to another layer. RC and LC use the DG-04 coefficients
  `(a_local, a_cond)=(0.5,0.5)` and are normalized after mixing.
- Free decoding: greedy, language zh, task transcribe, temperature 0,
  beam 1, max 200 new tokens, `condition_on_prev_tokens=false`; generation
  batch size is 1 per worker. Doses are `rho in {0.25,0.5,1,2}` times each
  layer's frozen D-construct scale `s_l`.

## Diagnostics

Teacher-forced diagnostics use the same panel, canonical reference token
alignment, and exact site. They report gold NLL/probability/rank/top-1,
entropy, KL(p0||psteer), site displacement, relative displacement, intended
direction cosine, and LN-site projections. A logit-lens is labelled
intermediate and never called a final decoder probability.

The fixed mixture diagnostic is teacher-forced at rho=.5 for
`lambda={-1,-.5,0,.25,.5,1,2}` for both `normalize(raw+lambda*cond)` and
`normalize(local+lambda*cond)` at every layer. Lambda is exploratory and is
not promoted to a canonical setting.

The linear probe is one fixed logistic probe per layer, trained on
D-construct dialogues and evaluated on disjoint D-dev-select dialogues, using
actual exact-site reference states and EN/ZH labels. No D-dev-confirm or
D-test data is read.

## Predeclared interpretation

All-layer results are descriptive. Layer-band candidates require agreement
among utility, PIER, retention, consistency across panel cases, token
diagnostics, and geometry; no final best layer may be claimed from this
10-utterance panel. Candidate outputs are labelled exploratory, with L24 as an
anchor and at most three Raw/Local, three Conditioning, and one weak negative
control layer proposed for later 300-utterance validation. PCA is not run
unless compatible larger cached states already exist.

## Operations and provenance

The planned workloads are: a bounded fixed-condition worker preflight, one
GPU direction/teacher-forced/probe extraction job, and one all-layer free
decode job using deterministic resumable layer shards. At most two GPU jobs
may run simultaneously; batch-size or worker changes are operational only.
Every condition is keyed by utterance/layer/direction/rho/diagnostic and is
written without overwriting DG04/BASIS-A artifacts. Runtime telemetry records
Slurm, GPU, workers, batch sizes, memory where available, throughput, runtime,
completed conditions, and failures.
