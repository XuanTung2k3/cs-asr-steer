# ST-PROMPT-R1 implementation design

Design freeze only. No new scientific runner, sequence outcome or GPU job in this session.
Use the spec and JSON as complete pre-outcome authority. Core v6 and historical conclusions unchanged.

## Verified reusable interfaces

- core_p1.direction: exact same-prefix E-M provider, epsilon/tiny/nonfinite status, float64 subtraction.
- models.hooks.apply_steering: native-dtype norm-preserving repair, zero object/bitwise identity.
- lss.sites.DecoderPostCrossAttnRecorder and DecoderPostCrossAttnInterventionHook(action_fn): DG-02;
  explicit supporting layers3/8 supported with enforce_contract_layer=False, not core candidate change.
  Independently verify final_layer_norm input and installed WhisperDecoderLayer.forward source hash.
- inference_cf_p2r.solve_scale/scaled_direction/emulate_edit_norm/DiagBranch.crop: unchanged geometry.
  Do not import its evaluator-bearing main runner plan. New relative wrapper computes target from r.
- inference_cf_cached.Branch.step and _processed: model-owned cache, cache_position, fixed prefix,
  no sharing between E/M/S. cached_greedy and processed_argmax have different tie implementations:
  R1 uses historical ST-LOC0/P2-DIR lowest-ID processed_argmax; reject baseline replay differences,
  never silently substitute cached_greedy topk on near/exact ties.
- inference_cf_st_loc0.runtime_projection, preparation/hash/barrier patterns, full-logit packing,
  FFN consumption audit and immutable seal patterns. Do not duplicate its V2 calibration machinery.
- inference_cf_p2dir_analyze.logit_metrics: evaluator only, fixed competitor/reference sets.
  P2-R summarize uses a different competitor possibility: not the primary endpoint here.
- canonical P2 evaluator/POI alignment and sequence evaluation routines: reuse untouched; no text
  edit distance used as correctness or retention. Independent auditor may reuse tokenization/schema,
  never primary metrics aggregates, bootstrap or decide implementation.

## Proposed isolated implementation (pending Claude)

1. src/csasr/inference_cf/prompt_r1.py: reference-free paired-query adapter and relative pulse,
   strict runtime projection, exact solver/consumed residual evidence. No evaluator imports.
2. experiments/inference_cf_st_prompt_r1.py: CPU prepare/manifest; future A passive4-layer capture,
   geometry/reproduction barrier, pulses; B model-owned continuation, energy ledgers/zero/random;
   CPU seal. Separate phase allowlists prevent references being loaded in runner.
3. experiments/inference_cf_st_prompt_r1_analyze.py: post-seal-only fixed endpoint/48-family A
   statistics/selection;8-family B statistics; deterministic label logic and every condition table.
4. experiments/inference_cf_st_prompt_r1_schedule.py: evaluator privilege, minimal UID+t oracle
   projection sealed BEFORE B; no targets/text/error scores cross runtime boundary.
5. experiments/inference_cf_st_prompt_r1_audit.py: independent formulas, full-grid/count/energy
   reconstruction and reference barrier/remote seal audit. No import primary decision/analysis.
6. slurm/inference_cf_st_prompt_r1.sbatch: one phase per allocation, interpreter from config,
   3h ceiling, Slurm only. B refuses missing audited/pushed selected config and schedule.

These paths are proposed, not claimed implemented. Implement/test/commit/push before pre-run PASS.
The committed preflight helper is CPU geometry only; it never loads model weights or logits/references.

## Required meaningful tests

Exact180/order/hash/80/20 and runtime-field allowlist; same-input paired E/M, no future prefixes;
all4 exact DG-02 captures/FFN input, zero identity, weight/hook restoration including exceptions;
float64-before-subtract direction equivalence/tiny/nonfinite/sign; relative dose versus raw coefficient,
reachability guards, max8 evals, native consumed energy and pairwise checks; complete24+12 controls;
historical D0/pulse/D2 barrier exact; deterministic random seed/hash; all decision boundary/equality
cases, family48/8 draws, absent-dialogue handling, selection ties, fixed competitor and canonical POI
identity; oracle earliest/all schedules independent of A success; prefix ownership after divergence;
EOS/cap/unreachable/missing schedule, unspent budget never moved; energy-matched random fairness and
invalid mismatches; no gold/provider leakage; immutable push barriers and manifests, zero sequence equal.
Run existing p1/cached/p2r/p2dir/st_loc0/DG-02 CPU suites. Later independent runnable audits mandatory.

## Preflight limitations and evidence

Sealed exact L16/L24 paired180 arrays enable CPU dose review; R0 mean prompt vectors are NOT
per-query E/M pairs at L3/L8. Document this limitation; passive four-layer extraction/complete geometry
coverage is mandatory before any new pulse in later A job. Frozen ladder never silently changed.
Historical timing gives a conservative forecast, not a measured R1 throughput claim. No pretrained
CPU or GPU model invocation is needed in the freeze. Geometry status and source hashes in the evidence
artifact can be independently reproduced. Do not let numerical emulation access downstream logits.
