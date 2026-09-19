# BASIS-A5 Protocol Amendment: Rank Stability Is Diagnostic

**Status:** ACTIVE exploratory-A5 amendment, 2026-09-19.

This amendment applies only to the exploratory BASIS-A5 Unique/Shared local
steering atlas. It does not rewrite, relabel, or delete the previous rank
stability gate.

## Amendment

- The primary construction rank remains **`r=32`** for every model, side, and
  layer.
- `r=32` was fixed before any downstream steering outcome was observed.
- Rank stability is now a **scientific diagnostic**, not a blocking criterion
  for this exploratory atlas.
- No MER, PIER, correction, corruption, retention, or other steering result
  may be used to select the construction rank.
- Every atlas result carries `rank_stability_status` with one of `STABLE`,
  `UNSTABLE`, or `NOT_TESTED` where the construction-only diagnostic is
  available. Results from unstable layers remain explicitly labeled
  `UNSTABLE` and are analyzed separately; they are not relabeled as passes.

## Preserved failure

The prior failure is preserved exactly at
`results/basis_a5_unique_shared/manifests/rank_stability.json`. Its recorded
`status=FAIL`, protocol hash, layer rows, and values are not overwritten. The
rank-gate Slurm job was **53047**; the prior protocol commits are
`de3a3e0` and `9d31b10`. The SHA-256 of the preserved JSON at amendment time
is `sha256:9610c7d2f57aa051a729a038dba57b8525b98cd95d1fa2e3fa6ac192e5ee817c`.

The old anchor policy did not include any Qwen audio-encoder layer. That fact
is recorded as a coverage limitation of the old diagnostic, not as evidence
that Qwen encoder directions are absent. Qwen encoder direction artifacts are
audited independently across E0--E23.

## Construction and interpretation boundary

Directions remain frozen rank-32 constructions from the corpus-aggregated
CS-Dialogue `D-construct` population only. The three atlas methods are
localized `+Unique`, `-Shared`, and normalized `Unique-minus-Shared` at
`rho=0.5`; the latter is not a difference-in-means direction. The atlas uses
the accepted A4 local masks and does not read `D-dev-confirm` or `D-test`.

This amendment authorizes execution and post-hoc stable/unstable/untested
comparisons. It does not authorize choosing a rank after inspecting those
comparisons or claiming that instability is harmless from a favorable single
layer.
