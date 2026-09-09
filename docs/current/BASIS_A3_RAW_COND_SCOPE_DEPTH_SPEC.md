# BASIS-A3 — Raw/Conditioning Scope × Depth × Cross-Corpus

Status: additive frozen explanatory study. This spec does not reopen DG-00–DG-08,
change the locked D-test artifacts, or introduce a learned controller.

## Frozen scientific panel

Only `Raw` and `Conditioning` direction families are used. `Raw` is the frozen
English-minus-Mandarin baseline-correct contrast and `Conditioning` is the
frozen English-prefix-minus-Mandarin-prefix contrast. Decoder vectors are the
existing BASIS-A2 exact-site artifacts, one vector per layer, and are reused
without reconstruction. Encoder Raw vectors are built separately for every
encoder layer from D-construct only using the exact encoder site below.

Evaluation uses the exact prior 300-ID D-dev-select panel for CS-Dialogue. For
SEAME, the only accepted local fixed panel is the frozen `eval_100` artifact
(`50 dev-man + 50 dev-sge`); it is reused exactly and the 300-per-split target
shortfall is recorded. No SEAME dev row is used to construct or tune a vector.
`D-dev-confirm` and `D-test` are forbidden.

## Intervention sites and scopes

Decoder steering is at `decoder_post_cross_attn_residual`, reconstructed as
`q + cross_attention_output` immediately before the decoder FFN, using the
canonical DG-02 hook and `NormPreserve`, with no depth rescaling. `Global`
steers every eligible non-prefix decode position. `Oracle-local` uses the same
vector, layer, and dose only at reference-aligned embedded-English positions;
it is a diagnostic upper bound, not deployable inference.

Encoder steering is at `encoder_post_self_attn_residual_pre_ffn`, reconstructed
as `q_enc + self_attention_output` immediately before the encoder FFN. `Global`
steers all valid acoustic frames. `Oracle-local` steers only the accepted
reference-aligned embedded-English acoustic span. Padding frames are always
zero-gain. The fixed acoustic boundary tolerance is 80 ms (the existing SEAME
panel segment-boundary convention); it is not tuned from outcomes.

## Frozen grid

R1 runs Raw on all 32 encoder and all 32 decoder layers at `rho=0.5`, both
scopes, on each panel. R2 adds only `rho in {0.25, 1.0}` for at most three
mechanically selected encoder layers and the fixed decoder anchors L0, L24,
and L27 (plus one negative control only if the predeclared selection rule
requires it). Conditioning is decoder-only at L24, L26, L27, and L31,
`rho in {0.25, 0.5, 1.0}`, both scopes, on each panel. R1 `rho=0.5` cells are
reused; they are never decoded again.

All generation is free decoding, greedy, `language=zh`, `task=transcribe`,
temperature 0, beam 1, max 200 new tokens, `condition_on_prev_tokens=false`,
generation batch size 1 per worker. The default Slurm execution is one MIG
GPU with two independent batch-1 workers, at most two concurrent jobs.

## Direction provenance and scales

Raw encoder construction records the D-construct population, accepted frame
alignment, English/Mandarin frame counts, per-layer vector hash, and the
normalization scale. Decoder Raw and Conditioning records point to their
existing BASIS-A2 arrays and hashes; no copy is made. The study never reads
legacy Local, Raw+Conditioning, Local+Conditioning, learned mixtures, or
controller artifacts as a steering condition.

## Metrics and audit

Headline transitions are explicitly POI transitions:
`poi_corrections` and `poi_corruptions`, with
`poi_net_utility = poi_corrections - poi_corruptions`. Candidate-level
outside-harm accounting, when available, remains separately named and is not
substituted into the POI fields. MER, PIER, embedded-English WER, matrix CER,
retention, edit counts, realized energy, and runtime use the canonical
`metrics_v1`/`result_v1` surfaces.

## Selection and claims

R1 is descriptive. R2 layer selection is mechanical from R1 only: balanced
correction/retention score, then intermediate response, then weak/damaging
response. No SEAME outcome changes the grid. Oracle-local results diagnose
direction quality versus scope/localization quality; they do not establish a
deployable localizer. Cross-corpus results are exploratory development
evidence, not held-out test evidence.

## Required provenance

The protocol freeze, panel fingerprints, resolved configuration, model
metadata, environment, git state, source/config hashes, direction hashes,
Slurm metadata, failed-cell quarantine, and completion status live under
`results/basis_a3_raw_cond_scope_depth/`. The protocol freeze is sealed before
the first scientific GPU result.
