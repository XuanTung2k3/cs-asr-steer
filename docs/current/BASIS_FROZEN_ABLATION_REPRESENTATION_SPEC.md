# BASIS-A Supporting Representation Analysis Specification

This is an additive post-run analysis of BASIS-A. It does not alter the
frozen intervention matrix, directions, doses, decoder, or result files.

## Frozen analysis definition

- Population: the exact 300-utterance `D-dev-select` subset used by DG-04,
  fingerprint `sha256:4a4ce18e7a368e60108fe1506ee6a70611d532d0821b376dc1ac8d483e2b4440`.
- Model/site: Whisper-large-v3, decoder L24 post-cross-attention residual,
  pre-FFN (`decoder_post_cross_attn_residual`).
- State source: one teacher-forced forward on each reference transcript,
  using the existing tokenizer and unit alignment utilities. No free-decoding
  transcript is substituted for a hidden state.
- Position sample: first-BPE-token query position for baseline-correct EN/ZH
  reference units only. Baseline correctness is taken from the reused frozen
  DG-04 B0 transcripts and canonical `unit_status`.
- PCA: fit to actual saved representation rows, never to the five direction
  vectors; deterministic seed 2408; at most 10,000 rows.
- Direction projections: `Raw`, `Local`, and `Conditioning`; scores are
  `<LN(r_t), d>` using featurewise LayerNorm without affine parameters and
  epsilon `1e-5`, matching the existing frozen projection-gate semantics.
- Distribution groups: baseline-correct embedded-English (`EN`) and
  matrix-language (`ZH`) positions. Report n, mean, standard deviation,
  median, quartiles, and standardized EN-minus-ZH mean difference.

## Operational policy

The extraction is analysis-only, uses one frozen model load, no trainable
parameters and no gradients, and reads no `D-dev-confirm` or `D-test` data.
The GPU launcher uses Slurm partition `main`; batch size is fixed before the
run and is not tuned from the results. PCA and projections remain descriptive:
they cannot establish causal superiority, semantic purity, or held-out
generalization.
