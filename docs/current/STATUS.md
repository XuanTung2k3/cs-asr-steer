# STATUS

**Stage 2 / DG-01 — COMPLETE (canonical metrics + result artifacts; freeze commit pending).**
`metrics_v1` + `result_v1` are frozen. Canonical modules: `src/csasr/evaluation/canonical.py`,
`retention.py`, `result_schema.py`, `legacy_adapter.py`, and the emission path `result_emit.py`.
Regression validated against the committed `results/job_a/summary.json` and
`results/job_b/summary.json` (read-only; MER/PIER reproduced exactly, gains re-signed to
`baseline − method`, legacy blended retention and outside-edits preserved as descriptive-only).
All 31 focused/regression tests pass (`tests/test_canonical_metrics.py`, `test_retention.py`,
`test_result_schema.py`, `test_legacy_adapter.py`, `test_dg01_regression.py`). Gate-coverage
denominator stays a **non-blocking deferred** item (spec §4 / MC §6; resolved with the gate,
DG-03+). **The DG-01 deliverables are currently uncommitted (untracked) in the working tree**; the
working tree must be committed before any hash is treated as the frozen DG-01 state — no freeze
commit is fabricated here (`HEAD` is still the pre-DG-01 code baseline `0975244…`). Next ticket:
**DG-02 — exact post-cross-attention, pre-FFN intervention hook.**

**Stage 1 / DG-00 — COMPLETE (documents finalized; freeze commit pending).** Scientific contract
finalized 2026-09-07 against repository **code baseline** `0975244bfb9563cdb123c4572eca697259ed399a`
(current `HEAD`). That baseline is the code state the contract audited; it does **not** contain the
Stage-1 deliverables. The deliverables themselves — `AGENTS.md`, `CLAUDE.md`, and `docs/current/*` —
are **currently uncommitted (untracked) in the working tree**. The freeze is therefore not yet
captured in an immutable commit; committing these files is a required mechanical step before the
contract can be relied on as locked. No fabricated freeze commit is recorded here. Authority:
`METHOD_CONTRACT.md` (MC). See also `CODE_MAP.md`, `EXPERIMENT_MATRIX.md`, and `DATA_EXPOSURE.md`.

This stage changed documentation only. Existing runnable results are not evidence for the
finalized method unless CODE_MAP classifies the relevant path as canonical.

## Confirmed decisions

1. Intervention site = decoder post-cross-attention residual, pre-FFN
   (`csasr.lss.sites.DECODER_TENSOR`); whole decoder-block output is rejected.
2. No depth rescale at the frozen site; norm-preserving repair is required.
3. `q`, nominal update, repaired `r`, and effective `u^S = r-q` are defined in MC §2 and §7;
   zero-gain positions must be bit-identical and `alpha=0` is a no-op.
4. Decoder candidate layers are `{16, 24}`.
5. Conditioning-residualized `Δ^⊥` is primary; legacy directions (including codebase `v_nat`)
   remain labeled legacy and must not be silently relabeled.
6. The inference-safe feature groups and factorized gate are contract components, not claims about
   current implementation.
7. Headline evidence requires free decoding; teacher-forced results are screening only.
8. `D-test` has permitted reference/C00 exposure but no steered exposure found.
9. Encoder–decoder disagreement is a **conditional** mechanistic hypothesis (MC §5); the contract
   preserves the decision rule and fallback gate (MC §6).
10. Correction, English retention, Mandarin retention, and monolingual retention populations are
    distinguished (MC §8).
11. Metric sign convention: `PIER_gain = PIER_baseline − PIER_method` (positive = improvement).
    Legacy artifacts with opposite signs must be converted before comparison.

## Implementation gaps

- **G1 — exact-site integration:** the reusable exact-site recorder/hook exists, but no current
  free-decoding runner wires it through cache position, forced prefix, and beam expansion. Current
  Job-A/Job-B and Round-1 execution paths intervene post-FFN.
- **G2 — layer sets:** `configs/lss/spec.yaml` lists `[8,16,24,31]`; Round-1 frozen executes seven
  layers; the contract permits decoder layers `{16,24}`.
- **G3 — scaling:** contract scale is construction-time projection standard deviation. Job A/B
  use unit scale; sweep code defaults to activation-norm scale; one legacy hook also divides by
  `sqrt(num_layers)`.
- **G4 — directions:** exact-site v2r3 assembly residualizes each contrast before aggregation,
  whereas MC specifies aggregate raw direction then project it. The dedicated class named by
  `configs/lss/spec.yaml` (`DirectionAccumulator`) does not exist; `LayerAccumulator` does.
- **G5 — gate:** no temporal localizer, outcome-supervised utility selector, abstention component,
  or factorized gate is implemented. Job-A F5 and Job-B T1 are monolithic projection gates.
- **G6 — training:** the current wrapper delegates the fixed-layer legacy trainer. Its T1 is
  rank-2, its T2 spans layers 24/25, and its CE covers every non-prefix token rather than only the
  correction set. Retention is measured, not represented by an explicit loss.
- **G7 — metrics:** implementations use different candidate populations/accounting; three
  outside-edit fields have three non-harm meanings; and delta signs conflict:
  `steer_sweep.metrics.delta_pier` is positive-better while Round-1 `delta_PIER`/`delta_MER` are
  negative-better.
- **G8 — calibration split:** dialogue-v2 `router-calib` has no frozen
  `calib-prob`/`calib-thresh` subdivision.
- **G9 — provenance:** current Round-1 result directories do not contain the complete resolved
  config/environment/git/model/hash manifest required by `AGENTS.md`.

## Open scientific decisions (non-blocking for DG-01)

1. Practical deployable site: E-only, decoder, or E+D (MC §1).
2. Conditioning subspace rank/estimator (MC §4).
3. Trained direction rank: 1 vs 2 (MC §4).
4. Feature-to-score aggregation (MC §5).
5. Exact factorized-gate algebra and transport composition (MC §6).
6. Whether an explicit retention loss and its weight (MC §8).
7. Staged training order: correction-only-first vs joint (MC §8).
8. Dialogue-v2 calibration subdivision (MC §6; DATA_EXPOSURE).

None of these decisions blocks metric canonicalization.

## Artifact state

- `results/job_a/summary.json` and `results/job_b/summary.json` are completed legacy pilots.
- `results/round1/job_a/summary.json` is complete at 84/84 cells, but remains a seven-layer,
  post-FFN current-wrapper result.
- `results/round1/job_b/training_status.json` records successful completion of the delegated legacy
  trainer; `results/round1/job_b/summary.json` contains that trainer's result summary.
- The dialogue-v2 locked test artifact exists as `locked/role_D-test.parquet` with its sidecar.
- `D-test` remains underpowered (MDE approximately 0.0608 at 15 dialogue blocks); this is a
  scientific limitation rather than an implementation blocker.

## Current ticket

**DG-01 — canonical metric and result-schema implementation: COMPLETE.** The spec is frozen at
`docs/current/DG01_METRICS_RESULTS_SPEC.md` (§13 implementation status; §14 regression + emission
completion). Metric schema `metrics_v1`, result schema `result_v1`. The regression-validation pass
ran: the read-only adapter round-trips the committed Job-A and Job-B summaries with MER/PIER
reproduced exactly and deltas re-signed to `baseline − method`; a fixed `adapt_job_b_summary` now
reads the flat `results_table` block (it previously dropped every Job-B method system by reading a
nonexistent `systems` key); and a current emission path (`src/csasr/evaluation/result_emit.py`)
emits `result_v1` with a real reused-infrastructure manifest (git HEAD, `resolved_config_hash`,
dataset fingerprint, schema versions; unavailable provenance stays `null`). No second canonical
metric/schema/gain-sign definition exists (Phase-D scan). Regression fixtures checked:
`results/job_a/summary.json`, `results/job_b/summary.json` (both byte-untouched). Non-blocking
carry-forward: gate-coverage denominator deferred to the gate ticket (DG-03+); the *scientific*
free-decoding runner that would call `result_emit` is DG-02+ (G1/G5) — the emitter exists and is
tested but no exact-site free-decoding runner exists yet to feed it. **Deliverables uncommitted:**
the DG-01 files are untracked; commit the working tree before treating any hash as the frozen DG-01
state (no freeze commit fabricated).

Historical DG-01 build record (retained): the canonical modules exist and pass CPU-only unit tests:
- `src/csasr/evaluation/canonical.py` (`SCHEMA_VERSION = "metrics_v1"`) — reuse facade for MER,
  PIER, EN-WER, ZH-CER, `*_gain` (baseline − method, positive = improvement), POI
  correction/corruption, candidate outside harm; asserts the PIER transition identity.
- `src/csasr/evaluation/retention.py` — three distinct MC §8 retention populations.
- `src/csasr/evaluation/result_schema.py` (`RESULT_SCHEMA_VERSION = "result_v1"`) — deterministic
  JSON, `gate_coverage` reserved/null with a guard.
- `src/csasr/evaluation/legacy_adapter.py` — read-only re-signing adapter (`legacy-derived`,
  `production_artifact=False`).
- Tests: `tests/test_canonical_metrics.py`, `tests/test_retention.py`,
  `tests/test_result_schema.py`, `tests/test_legacy_adapter.py`.

No existing Round-1 number may be merged or treated as comparable until it passes through the
adapter. The regression-validation pass has now run (spec §14); DG-01 is **COMPLETE**. The legacy
execution scripts are intentionally **not** rewired to emit `result_v1` (that is DG-02+ scientific-
runner work, not a DG-01 blocker); the canonical emission helper `result_emit.py` is the current
`result_v1` producer and is validated.

Scoped, non-blocking deferral inside DG-01: **gate coverage denominator** is not fixed by Stage 1
and the gate does not yet exist (MC §6). The canonical schema reserves the `gate_coverage` field
but does not compute it in DG-01; its denominator resolves with the gate (DG-03+). This is recorded
as a blocker on that one metric only, not on the metric/schema/manifest work that proceeds now.

**Reconcile in the implementation pass:** `src/csasr/evaluation/{pier,mer,correction_harm}.py`,
`steer_sweep/metrics.py`, `steer_sweep/round1_metrics.py`, and `src/csasr/lss/outcomes.py` to one
versioned denominator, transition convention, sign convention, and artifact schema, via a reuse
facade (`src/csasr/evaluation/canonical.py`) — not by reimplementing scoring.

## Current ticket (Stage 3)

**DG-02 — exact post-cross-attention/pre-FFN intervention site: implementation + CPU/synthetic
validation COMPLETE; real-model acceptance PENDING on a GPU node; NOT yet frozen.** Spec at
`docs/current/DG02_INTERVENTION_SITE_SPEC.md` (§17 = delivered implementation).

Delivered (implementation pass, working tree, uncommitted):
- `src/csasr/lss/sites.py` — new canonical `DecoderPostCrossAttnInterventionHook` (exact site
  `r=q+u_source`; FFN consumes repaired `r̃`; layer `cache_position` pre-hook; dynamic
  `num_forced_prefix`; per-row `gate_fn`/`gain`; `steer`/`train` modes; detached `AuditRecord`
  emitter; `CONTRACT_DECODER_LAYERS=(16,24)` guard). Recorder now exposes `q`/`u_source`/`r`
  separately. Legacy `DecoderPostCrossAttnSteeringHook`, `apply_steering`, and F5/T1 paths
  untouched. `apply_steering` (`models/hooks.py`) reused verbatim — no `sqrt(num_layers)` rescale.
- `tests/test_dg02_site.py` — spec §13 matrix, **19 tests**; with `tests/test_lss_sites.py` (11)
  → **30 passed** CPU. DG-01 suite (31) re-run **unchanged**.
- `experiments/dg02_real_acceptance.py` + `sbatch/cs_asr_dg02_real_acceptance.sh` — the §14 rung-1
  one-utterance real-Whisper acceptance (β=0 identity, `r=q+u`, exact site, prefix/cache/norm at
  L16 & L24). **Must run on a GPU/compute node to close the real-model gates before freeze.**

Blocking-to-freeze item (environmental, not a code defect): the CPU dev box has 2 GB RAM / no swap
/ no GPU and cannot load whisper-large-v3 (~3 GB), so §14 rung 1 was not executed here. Freeze
DG-02 only after `dg02_real_acceptance.py` reports `PASS` at both layers.

Confirmed architecture facts (installed source + repository, verified this pass):

Confirmed architecture facts (installed source + repository, verified this pass):
- Model is **Whisper-large-v3, 32 decoder layers**; `bundle.decoder_layer(k)` →
  `model.model.decoder.layers[k]`, **0-indexed and direct** — scientific **L16→index 16, L24→index
  24**, both interior, no off-by-one.
- `transformers` `WhisperDecoderLayer.forward` (installed): the **site is line 528**
  `residual + encoder_attn(LN(residual))` (post-cross-attn, pre-FFN); block output (line 537,
  post-FFN) is the **rejected** site. Dropouts (511/527) are identity under `eval()`.
- Operational capture: `q` = `encoder_attn_layer_norm` pre-hook `args[0]` (post-self-attn state);
  `u_source` = `encoder_attn` output[0] (source cross-attn contribution); `r = q + u_source` = the
  site (exact to dtype ULP; validated by `assert_site_reconstruction`). No LayerNorm sits on the
  `q→r` residual path.
- `sites.py` (`DecoderPostCrossAttnRecorder` / `DecoderPostCrossAttnSteeringHook`) is the
  **CANONICAL/REUSABLE** base; it omits the `sqrt(num_layers)` rescale and reuses
  `apply_steering` norm preservation (preserves per-token L2 norm of the site `r`, `EPS=1e-6`).
- Cache position available via layer `cache_position` kwarg and `cache_length(past_key_values)`
  (`steer_sweep/hooks.py:46`); forced prefix = `build_prefix` length (4 here), to be read
  dynamically not hard-coded; per-beam gate is row-local (batch = `batch*num_beams`).

The additive infra `IMPLEMENTATION GAP`s are now **implemented** in `sites.py`: q/u_source/r exposed
separately; decoder-layer pre-hook for `cache_position`; dynamic `num_forced_prefix`; per-row
`gate_fn` + trainable direction/gain + `steer`/`train` mode + detached `AuditRecord`. Still deferred
by design (row-local interface provided, transport later): G-e source-item→beam gate transport;
G-f first-content-token/final-prefill steering choice (kept at the legacy "skip all prefill"
default via position-based exclusion). DG-02 is **not** marked complete until the real-model
acceptance passes. Layer selection (L16 vs L24) and directions are DG-03.

## Next tickets

Immediate next step (to close DG-02): run `experiments/dg02_real_acceptance.py` on a GPU/compute
node (`sbatch/cs_asr_dg02_real_acceptance.sh`); on `PASS` at L16 & L24, mark DG-02 COMPLETE and
freeze. The CPU/synthetic matrix (spec §13) already passes.

Then: **DG-03 — version steering directions and validate layers 16/24 with causal controls.**
Direction construction, gate implementation, and training work follow as separate tickets (DG-03+).
No new scientific run is GO until DG-02 is frozen.
