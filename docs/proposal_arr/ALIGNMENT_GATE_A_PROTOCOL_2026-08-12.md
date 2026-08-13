# Alignment Gate-A reliability protocol

This document describes the repaired implementation after Slurm job 38745. It
is a task-specific reliability protocol for span-local representation and
intervention experiments in this repository. It is not a universal forced-
alignment standard, and it does not amend the proposal's numerical thresholds.

## Objects and representations

Gate A retains three traceable levels:

1. raw aligner items, including every invalid timestamp and failure reason;
2. normalized reference units;
3. same-language runs, embedded-English spans, and EN↔ZH switch edges.

Natural family qualification uses level 3 because these outer boundaries are
what Site-E masks and span-local analyses consume. Raw failures remain published
in `metrics/l1b_raw_unit_diagnostics.parquet`; aggregation never rewrites their
timestamps. An internal zero-duration same-language unit may sit inside a run
whose two outer edges are supported, while the same defect at a required outer
edge invalidates that target. Same-language shared tokenizer intervals may be
unioned. Cross-language overlap, reversed switch order, missing switch edges,
and invalid outer boundaries remain invalid.

## Distinct evidence questions

Full-role data sufficiency is counted from the complete authenticated L0 role
manifests. The deterministic alignment audit sample measures boundary
reliability and cannot cap a role-count requirement. Evidence records both the
full-role source manifest and `alignment_audit_sample_size_per_role`; startup
validation rejects a configured count that is mathematically unreachable from
its declared universe.

Raw positive-duration validity and temporal nonmonotonicity have separate
denominators. A missing, non-finite, zero, or reversed interval is invalid
duration. It is not also counted as nonmonotonic. Ordering is measured only
among otherwise positive-duration objects.

Natural cross-aligner differences are disagreement measurements. They are not
ground-truth boundary error. Absolute lexical start/end/boundary error may be
claimed only for `manual_lexical`, `existing_gold_lexical`, or a
`constructed_exact_lexical` reference that genuinely supplies exact lexical
edges. An RMS/VAD-trimmed concatenation is `audio_splice`: it supports signed
ZH-end and EN-start offsets and the gap around its known audio seam, but not a
lexical absolute-error claim.

## Pairing and calibration

A natural family must clear target coverage, target-invalid, and target-
nonmonotonic thresholds. Independence is determined by estimator-family
provenance, not display names. The configured `aligner_pair_priority` chooses
the first naturally qualifying independent pair with sufficient paired target
objects. Gate or error results never choose the pair.

Calibration uses paired eligible reference items from that same selected pair.
A rejected third family cannot improve or worsen it. If fewer than two families
qualify, the blocker is `blocked_insufficient_independent_aligners`. Only when
at least two qualify but eligible overlap is insufficient is
`blocked_no_paired_cross_aligner_evidence` emitted; raw and target-filtered
overlap counts remain visible in either case.

Development and gate candidate pools are larger than the required 100 usable
paired boundaries. A fresh held-out generation is not allocated when the
development reference kind cannot support the selection. Gate generations,
source IDs, request fingerprints, and item/truth fingerprints are immutable;
interrupted attempts are abandoned and their sources stay reserved.

## Jitter and status

±50 and ±100 ms geometric window survival remains reported under the original
thresholds. Results are additionally stratified by duration and normalized by
span duration because raw safe-interior survival is duration-confounded.
Scientific conclusion stability (directions, probes, layer choice, Site-E
effects, correction, and corruption) is a separate sub-gate and is explicitly
unavailable until those downstream artifacts exist.

`completed` means preparation succeeded; `completed_no_go` means compatible
evidence was measured and a threshold failed; `blocked` means required or
scientifically compatible evidence is unavailable; `failed` means corrupt
input/schema or an unexpected implementation error. All truthful terminal
outcomes return zero except `failed`. Only `passed`, together with an authentic
frozen-span artifact, can unlock L1c.

## Safe development re-evaluation

The following command reads authenticated cached candidates and recomputes CPU
diagnostics without running aligners, allocating/exposing a gate generation,
writing a production status, freezing spans, or unlocking L1c:

```bash
cd /home/tungnx/cs-asr-steer
PYTHONPATH=src /home/tungnx/miniconda3/envs/acl1/bin/python \
  -m csasr.experiments.lss_alignment_dev_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output /tmp/csasr_gate_a_dev_diagnostic
```

The JSON and Markdown outputs have manifest sidecars carrying
`development_only_diagnostic` taint. Before another full chain, the diagnostic
must show a credible deterministic natural pair and reachable full-role counts,
and a genuine lexical calibration source must be configured and validated.

The future production command is therefore recorded only as:

```bash
# DO NOT RUN YET while genuine lexical calibration is unavailable
sbatch cs_asr_lss.sh chain --overwrite
```
