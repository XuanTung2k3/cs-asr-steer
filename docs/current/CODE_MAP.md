# CODE MAP — where the method lives

Companion to `METHOD_CONTRACT.md`. Every repository path presented as a location below exists. A
component with no contract-conformant implementation is described as **absent** without inventing
a path.

Classification: **canonical** means the implementation satisfies the current contract;
**reusable** means a tested primitive can be used by a future canonical runner; **legacy** means
preserved pilot/current-execution code that does not satisfy the finalized contract. There is
currently no end-to-end canonical free-decoding runner.

---

## 1. Entry points and commands

| Entry point | Path | Classification and verified behavior |
|---|---|---|
| Round-1 frozen sweep | `experiments/round1_frozen.py` | **legacy for this contract / current execution wrapper**. Runs 84 greedy `D-dev-select` cells over seven layers and imports `NormPreserveDecoderHook` from Job A. It does not dispatch Job A and does not consume its YAML `confirm_split`. The hook is post-FFN. |
| Prior frozen Job A | `experiments/job_a_frozen.py` | **legacy pilot**. Fixed layer 24, beam 5, six F0–F5 systems on a 150-utterance / 10-dialogue candidate population. F1–F5 use a custom post-FFN hook; F5 is a single projection sigmoid, not the contract gate. |
| Round-1 training wrapper | `experiments/round1_training.py` | **legacy for this contract / current execution wrapper**. Delegates `experiments.job_b_training.main([])` and does not implement the advertised seven-layer/T1/global/SALSA/LoRA matrix. |
| Prior Job B trainer | `experiments/job_b_training.py` | **legacy pilot**. Fixed layer-24 rank-2 T1, layers 24+25 T2, and layer-24 LoRA T3. T1/T2 hook whole decoder-block outputs and training CE covers all non-prefix tokens. |
| Round audit printer | `experiments/round_audit.py` | **diagnostic only**. Prints hard-coded review assertions; it is not a repository or artifact verifier. |
| DG-02 real acceptance | `experiments/dg02_real_acceptance.py` (launcher `sbatch/cs_asr_dg02_real_acceptance.sh`) | **DG-02 verification helper**. One-utterance real-Whisper plumbing check at L16 & L24 (β=0 identity, `r=q+u`, exact site, prefix/cache/norm). Needs a GPU/compute node; writes `results/dg02_real_acceptance.json`. No layer/direction/β selection, no ASR-quality claim. |
| Round-2 / Round-3 | `experiments/round2_combinations.py`, `experiments/round2_transfer.py`, `experiments/round3_qwen_frozen.py`, `experiments/round3_qwen_training.py` | **stubs/guards, proposed behavior unimplemented**. Dry-runs exist; non-dry Round-2 paths terminate as not submitted, and Round-3 contains no experiment pipeline. |
| P0 + E1–E5 | `src/csasr/experiments/p0_baseline.py`, `src/csasr/experiments/e1_alignment.py`, `src/csasr/experiments/e2_directions.py`, `src/csasr/experiments/e3_separability.py`, `src/csasr/experiments/e4_oracle.py`, `src/csasr/experiments/e5_boundaries.py`, `src/csasr/experiments/pipeline.py` | **legacy** feasibility pipeline. |
| V2R3 family | `src/csasr/experiments/v2r3_baseline_poi.py`, `src/csasr/experiments/v2r3_cross_attention_spans.py`, `src/csasr/experiments/v2r3_day5_expansion.py`, `src/csasr/experiments/v2r3_directions.py`, `src/csasr/experiments/v2r3_gate_b.py`, `src/csasr/experiments/v2r3_headroom_diagnostic.py`, `src/csasr/experiments/v2r3_hypothesis_substitution.py`, `src/csasr/experiments/v2r3_invariance.py`, `src/csasr/experiments/v2r3_oracle_screen.py` | **legacy pipeline with reusable exact-site direction/teacher-forced pieces**. Results are documented in `docs/RESULTS_RUN_7DAYS.md` and `docs/STEERING_EXPERIMENT_FULL_RECORD_2026-08-17.md`. |
| NAT5H / LSS staging | `src/csasr/experiments/nat5h_pipeline.py`, `src/csasr/experiments/lss_alignment_dev_diagnostic.py`, `src/csasr/experiments/lss_l0_freeze.py`, `src/csasr/experiments/lss_l1a_diag.py`, `src/csasr/experiments/lss_l1b_valid.py`, `src/csasr/experiments/lss_preflight.py`, `src/csasr/experiments/lss_status.py`, `src/csasr/experiments/lss_v2_candidates.py`, `src/csasr/experiments/lss_v2_spec_freeze.py` | **legacy/reusable** staging. |
| Track-A/B drivers | `run_study.py`, `run_stage_d_ext.py`, `steer_sweep/study.py`, `steer_sweep/sweep.py`, `steer_sweep/stage_d_ext.py`, `steer_sweep/summary.py`, `steer_sweep/summary_d_ext.py` | **legacy** orchestration. |

Verified model-free commands:

```bash
python experiments/round1_frozen.py --dry-run
python experiments/round1_training.py --dry-run
python experiments/round_audit.py
```

Round-1 Slurm wrappers exist at `sbatch/cs_asr_round1_frozen.sh` and
`sbatch/cs_asr_round1_training.sh`; they invoke the two Round-1 wrappers above with
`configs/rounds/round1_frozen.yaml` and `configs/rounds/round1_training.yaml`. Guard/stub wrappers
exist at `sbatch/cs_asr_round2_combinations.sh`, `sbatch/cs_asr_round2_transfer.sh`,
`sbatch/cs_asr_round3_qwen_frozen.sh`, and `sbatch/cs_asr_round3_qwen_training.sh`. Legacy launchers
are `sbatch/study_track_a.sh`, `sbatch/study_track_b.sh`, `sbatch/study_stage_d_ext.sh`,
`cs_asr_e1_e5.sh`, `cs_asr_lss.sh`, and `cs_asr_nat5h.sh`.

---

## 2. Method component → location

| Contract component | Location | Classification / evidence |
|---|---|---|
| Exact decoder post-cross-attention site, recorder, hook | `src/csasr/lss/sites.py` — `DecoderPostCrossAttnInterventionHook` (canonical DG-02 hook), `DecoderPostCrossAttnRecorder` (now stores `q`/`u_source`/`r`), `AuditRecord`, `num_forced_prefix_from`, `CONTRACT_DECODER_LAYERS=(16,24)`, `DecoderPostCrossAttnSteeringHook` (legacy reconstruction), `assert_dropout_disabled`, `assert_site_reconstruction`, `assert_no_site_hooks` | **canonical primitive** (DG-02). Exact site, `r=q+u_source`, cache-position via layer `cache_position` pre-hook, dynamic forced-prefix exclusion, per-row `gate_fn`/`gain`, `steer`/`train` modes, detached audit records. **FROZEN** — CPU/synthetic matrix green + real-model acceptance PASS at L16 & L24 (Slurm job 50369; `results/dg02_real_acceptance.json`) |
| Whole-decoder-block steering | `src/csasr/models/hooks.py` — `DecoderSteeringHook`; `steer_sweep/hooks.py` — `DecoderSteering`; `experiments/job_a_frozen.py` — `NormPreserveDecoderHook`; `experiments/job_b_training.py` — `T1Hook`, `T2Hook` | **legacy/rejected site**; only the first also applies `sqrt(num_layers)` depth rescaling |
| Nominal update and norm repair | `src/csasr/models/hooks.py` — `apply_steering` | **canonical primitive** used by the exact-site hook and encoder hook; Round-1 runners reimplement it with different scaling |
| Encoder steering | `src/csasr/models/hooks.py` — `EncoderSteeringHook`; builders in `src/csasr/steering/encoder_hook.py` and masks in `src/csasr/steering/masks.py`; direction in `src/csasr/directions/encoder.py` | reusable / legacy for the decoder-site contract |
| Exact-site decoder contrasts | `src/csasr/experiments/v2r3_directions.py` — `site_d_contrasts`, `assemble` | reusable exact-site construction from a legacy pipeline |
| Post-FFN decoder direction construction | `src/csasr/directions/decoder.py`, with generic accumulation in `src/csasr/directions/accumulators.py` | **legacy for this contract** |
| Conditioning projection | `src/csasr/experiments/v2r3_directions.py` — `orthonormal_basis`, `project_out`, `assemble` | reusable for `v_local`/`v_cond`, but operation order differs from MC §4; dedicated canonical component absent |
| **Canonical steering-basis builder** (`v_raw`, `v_cond`, `v_local`, `V^0`; MC §4) | `src/csasr/directions/steering_basis.py` — `build_basis`, `raw_language_contrast`, `language_conditioning`, `conditioning_residualized_local`, `write_basis_artifact` (`steering_basis_v1`); construction runner `experiments/dg03_build_basis.py` | **canonical (DG-03, FROZEN).** v6 aggregate-then-residualize order; finite/unit/orthogonal validation + deterministic tensor hashing; artifacts carry real `dataset_fingerprint` (sha256 of D-construct candidates+role+POI parquets) and `construction_config_hash`. Built at L16/L24 (jobs 50452/50470). Not a relabel of legacy `v_nat` |
| **DG-03 causal screen** (C0–C4, free decoding) | `experiments/dg03_causal_screen.py` + CPU repair `experiments/dg03_repair_outside_harm.py` (launcher `sbatch/cs_asr_dg03_screen.sh`) | **canonical (DG-03, FROZEN).** Reuses frozen DG-02 hook + `job_a_frozen.oracle_steps_for` (oracle diagnostic gate); every condition (incl. C0) is a complete validated `result_v1` (metrics+gains+POI transitions+3 retention populations+canonical `outside_harm`+reserved gate_coverage) with per-utterance texts; per-condition edit-count/total-energy recorded, C4 count-matched to C1. Selected **L24** (jobs 50471/50472; `results/dg03/screen/`). Outside harm is reconstructed from the frozen D-dev-select candidate spans; no GPU rerun is required |
| **DG-04 frozen baseline frontier** (B0–B3) | `experiments/dg04_frozen_baselines.py`, `sbatch/cs_asr_dg04_frozen_baselines.sh`, `results/dg04/` | **canonical (DG-04, FROZEN).** L24 and `steering_basis_v1` are guarded; B1/B2 use fixed directions and B3 is a frozen projection gate calibrated on `router-calib`. Full `ρ={0.5,1.0,2.0}` matrix ran on D-dev-select in jobs 50483/50484 using `mig`; `result_v1`, canonical outside harm, config/dataset hashes, Slurm metadata, frontier, and reference are emitted. Reference is B1 `ρ=0.5`; no adaptive training |
| **Adaptive controller** `f_θ(LN(r_t)) → (g_t, π_t)` (MC §6) | **Absent** | **DG-05 gap.** Consumes the frozen DG-02 hook's per-row site state; `d_{ℓ,t}=normalize(V^0 π_t)`. Legacy F5/T1 gates are **not** this controller |
| **Damage-aware training** (correction-set-only CE + KL retention `𝓛_ret,E`/`𝓛_ret,M` + optional anchor; MC §8) | **Absent** | **DG-06 gap.** Job-B all-token CE is `LEGACY DESIGN` |
| **Exact-site scientific free-decoding runner** (DG-05+) | **Absent for post-DG-04 stages** | **gap.** DG-03/DG-04 canonical runners are implemented above; later runners must remain on `sites.DecoderPostCrossAttnInterventionHook` (FROZEN) and write a full provenance manifest; legacy runners steer post-FFN |
| Random/sign controls | `src/csasr/directions/controls.py` — `random_direction`, `wrong_sign`, `orthogonalized_random` | reusable; no contract-complete paired control runner (DG-03 controls) |
| Inference-safe feature contract | `src/csasr/lss/features_contract.py` — `FEATURE_ALLOWLIST`, `assert_inference_safe` | canonical leakage guard; the controller's only input `LN(r_t)` trivially passes it |
| Temporal localizer | Absent | `LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE` (superseded by the controller, MC §5–§6) |
| Outcome-supervised utility selector and abstention | Absent | `LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE`; Job-B logistic regression was only an EN/ZH diagnostic probe |
| Factorized gate | Absent | `LEGACY DESIGN` / `NOT IN CURRENT CORE SCOPE`; Job-A F5 / Job-B T1 are monolithic projection gates, not the controller |
| Encoder–decoder disagreement score | Absent | `OPTIONAL SUPPORTING ANALYSIS` only; no longer a decision gate (MC §5) |
| Candidate correction/harm/utility | `src/csasr/lss/outcomes.py` — `candidate_unit_sets`, `newly_introduced_errors`, `utility` | **canonical primitive** correctness-flip accounting |
| Legacy target/outside accounting | `src/csasr/evaluation/correction_harm.py`; `steer_sweep/metrics.py` — `target_outcomes`, `corruption_and_retention` | reusable/legacy; `outside_region_edits` is transcript difference, not harm |
| Contract correction + retention losses | Absent | **DG-06 gap.** Need correction-set-only CE on `𝒞_E` + explicit KL-to-baseline retention on `ℛ_E`/`ℛ_M` (MC §8); Job B's `F.cross_entropy` at `experiments/job_b_training.py:1016` is all-token CE (legacy) |
| Gate status machinery | `src/csasr/lss/gates.py`, `src/csasr/utils/status.py` | canonical |
| Speaker-role partition | `configs/lss/roles.yaml`, `src/csasr/lss/roles.py` | legacy v1; conversation-disjoint but not dialogue-disjoint |
| Dialogue-atomic partition | `configs/lss/roles.yaml`, `src/csasr/lss/dialogue_roles.py`, `src/csasr/lss/balance.py`, `src/csasr/lss/seeds.py`, `src/csasr/experiments/dialogue_roles_build.py` | canonical partition implementation; dialogue-v2 `router-calib` has no configured sub-split |
| Alignment | `src/csasr/lss/align/`, `src/csasr/lss/spans.py`, `src/csasr/data/alignment.py`, `src/csasr/data/ctc_alignment.py`, `src/csasr/data/alignment_checks.py`, `src/csasr/nat5h/consensus.py` | reusable/legacy |
| Spec freeze / provenance | `src/csasr/lss/specfreeze.py`, `src/csasr/lss/v2_freeze.py`, `src/csasr/lss/v2_namespace.py`, `src/csasr/lss/manifest.py`, `src/csasr/utils/provenance.py`, `src/csasr/utils/hashing.py` | canonical primitives; Round-1 result writers do not use the full manifest surface |

Configuration warning: `configs/lss/spec.yaml` lists decoder layers `[8,16,24,31]` and names the
nonexistent `csasr.directions.accumulators.DirectionAccumulator.projection_std`; the implemented
class is `LayerAccumulator`. `configs/rounds/round1_frozen.yaml` lists seven layers, while the
runner reads the same list from `steer_sweep/rounds.py` rather than from YAML.

---

## 3. Metrics

### 3a. Canonical DG-01 surface (new — the interface future DG stages use)

| Component | Path | Classification / convention |
|---|---|---|
| Canonical scoring facade | `src/csasr/evaluation/canonical.py` — `SCHEMA_VERSION = "metrics_v1"`; `corpus_metrics`, `gain`, `error_metric_gains`, `correction_corruption`, `candidate_outcomes`, `paired_corpus_report` | **canonical**. Reuses `mer.corpus_mer` / `pier.pier` / `lss.outcomes`; no rescoring. `gain(M) = baseline − method` (**positive = improvement**); emits `mer_gain/pier_gain/en_wer_gain/zh_cer_gain`, never the bare `delta_*` |
| Canonical retention | `src/csasr/evaluation/retention.py` — `matrix_zh_retention`, `embedded_en_retention`, `monolingual_retention`, `retention_report` | **canonical**. Three MC §8 populations kept distinct; each `{numerator, denominator, rate}`; empty population → `rate = None` |
| Canonical result schema | `src/csasr/evaluation/result_schema.py` — `RESULT_SCHEMA_VERSION = "result_v1"`, `CanonicalResult`, `validate`, `reserved_gate_coverage`, `populate_gate_coverage` (guard) | **canonical**. Deterministic JSON (`sort_keys`, `allow_nan=False`); `gate_coverage` reserved/null (deferred, §4 of spec) |
| Legacy adapter | `src/csasr/evaluation/legacy_adapter.py` — `adapt_job_a_summary`, `adapt_job_b_summary`, `adapt_round1_paired_report`, `resign_delta` | **canonical (read-only)**. Re-signs legacy `method − baseline` deltas into gains; marks records `legacy-derived`, `production_artifact=False`; never fabricates retention/outside-harm. Job-B adapter reads the flat `results_table` block (its `methods` block nests `epoch_results`) |
| Canonical emission + provenance | `src/csasr/evaluation/result_emit.py` — `build_provenance`, `emit_result`, `emit_baseline_and_method` | **canonical**. Smallest current `result_v1` emitter: scores an already-produced text triple through the facade (no rescoring) and stamps a real manifest via `utils.provenance.stage_provenance` + `utils.logging.git_state` + `lss.manifest.run_id` + `utils.hashing`; unavailable provenance stays `null`. Changes no decoding/steering/training/data |
| Tests | `tests/test_canonical_metrics.py`, `tests/test_retention.py`, `tests/test_result_schema.py`, `tests/test_legacy_adapter.py`, `tests/test_dg01_regression.py` | CPU-only fixtures; regression file round-trips the committed Job-A/Job-B summaries + the emission pair |

### 3b. Legacy/reusable metric surfaces (preserved; feed canonical via reuse or adapter)

| Surface | Path | Verified convention |
|---|---|---|
| v1 PIER/MER (reused by facade) | `src/csasr/evaluation/normalization.py`, `mer.py`, `pier.py`, `correction_harm.py`, `bootstrap.py` | `pier` = POI errors / reference POIs; MER and PIER are lower-better; the **canonical EN-WER/ZH-CER source** (`mer.corpus_mer`) |
| Sweep delta | `steer_sweep/metrics.py` — `delta_pier`, `corruption_and_retention` | `delta_pier = baseline − method` (**positive better**); `zh_retention` is **blended** matrix+monolingual — superseded by `retention.py` |
| Round-1 paired metrics | `steer_sweep/round1_metrics.py` — `pier_transition_counts`, `assert_pier_identity`, `paired_metric_report` | `delta_PIER = method − baseline` (**negative better**); re-signed by the adapter. `assert_pier_identity` is **reused** by the facade |
| Candidate utility (reused by facade) | `src/csasr/lss/outcomes.py` | **canonical outside harm** = `n_corrupted_outside` (correctness flip); insertions excluded by default |
| Legacy WER (not canonical) | `src/csasr/experiments/v2r3_day5_expansion.py` — `corpus_word_error_rate` | whitespace-split blended WER, different alignment, no EN/ZH split — **legacy**, not the canonical WER |
| Legacy retention helper | `src/csasr/experiments/v2r3_gate_b.py` — `monolingual_retention` | counts all non-EN baseline-correct units (blended) — legacy; canonical decomposition is in `retention.py` |
| Outside-edit fields (descriptive only) | `correction_harm.outside_region_edits`; `round1_metrics.outside_edit_rate`; `experiments/job_b_training.py` — `outside_edits` | projection diff outside ±1 radius; whole-transcript edit distance; summed target spillover. **None is canonical outside harm** (that is `outcomes.n_corrupted_outside`) |

DG-01 status: **COMPLETE.** The canonical surface (3a) exists, is unit-tested, and its regression
against the committed Job-A/Job-B summaries passes (`tests/test_dg01_regression.py`); a current
`result_v1` emission path (`result_emit.py`) writes a real reused-infrastructure manifest. Legacy
surfaces (3b) are preserved for historical reproducibility and reached only via reuse or the
read-only adapter; no legacy Round-1 number is merged/compared until it passes through the adapter.
Non-blocking carry-forward: gate coverage remains deferred (spec §4 / MC §6), and the *scientific*
free-decoding runner that would feed `result_emit` is DG-02+ work (G1/G5) — the emitter is built and
tested but no exact-site free-decoding runner exists yet to call it.

---

## 4. Result artifacts

- `results/job_a/summary.json` and `results/job_b/summary.json`: completed **legacy pilot** outputs.
  Job A's target table contains no baseline-correct targets (`num_baseline_correct = 0`), so its
  reported zero target corruptions cannot establish a zero corruption/retention rate.
- `results/round1/job_a/summary.json`: 84/84 current-wrapper cells completed; still post-FFN and
  seven-layer, so not contract evidence.
- `results/round1/job_b/training_status.json` and `results/round1/job_b/summary.json`: completed
  delegated Job-B run; still fixed-layer, rank-2/post-FFN/all-token-CE pilot behavior.
- `out/SUMMARY.md` and `out/SUMMARY_D_EXT.md`: legacy Track-A/B summaries.
- The configured dialogue-v2 role tree is recorded by
  `configs/lss/l1b_candidates_dialogue_v2r3.yaml`; its published locked artifact exists at
  `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/locked/role_D-test.parquet`.

No result above may be relabeled as evidence for the finalized exact-site method.

---

## 5. Tests

- `tests/test_lss_sites.py`: exact-site reconstruction, separation from block output, norm repair,
  dropout/prefill behavior, and hook cleanup on a tiny model (legacy `DecoderPostCrossAttnSteeringHook`).
- `tests/test_dg02_site.py`: DG-02 exact-site matrix (spec §13) — `r=q+u_source`, β=0 bit-identity,
  disabled/removed-hook identity, FFN-consumes-repaired-`r̃`, norm preservation, dynamic forced-prefix
  exclusion + eligible editing, per-token gate, state-dependent per-beam (row-local) gate, cache
  positions advance / agree with full-sequence / no reset / eligibility transition, detached
  recording, gradient-to-trainable-gate with frozen backbone, lifecycle/double-install/exception
  cleanup, single application, L16/L24 mapping + layer guards, and a standalone real
  `WhisperDecoderLayer` reconstruction. CPU-only, real block class, no weight download.
- `tests/test_round1_3_infrastructure.py`: cell determinism, split-disjoint helper, teacher-forced
  prefix masking, and Round-1 PIER transition identity.

Exact-site free-decoding cache position, forced-prefix exclusion, and per-row gating are tested
synthetically (`tests/test_dg02_site.py`) and confirmed on the real model (Slurm job 50369,
`experiments/dg02_real_acceptance.py`, artifact `results/dg02_real_acceptance.json`, PASS at L16 &
L24). Actual beam decoding is validated only synthetically here (state-dependent row-local gate); a
real beam run is deferred to when beam decoding is scientifically evaluated.
