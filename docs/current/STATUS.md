# STATUS

## Stage roll-up (2026-09-07)

- **DG-00 — COMPLETE.** Scientific contract & guardrails.
- **DG-01 — COMPLETE.** Canonical metrics (`metrics_v1`) + result schema (`result_v1`) + adapter.
- **DG-02 — COMPLETE / FROZEN.** Exact post-cross-attention/pre-FFN site; real-model acceptance PASS
  at L16 & L24 (Slurm job 50369). Committed `db365d8`.
- **DG-03R — COMPLETE / FROZEN.** The reconciled documents define **contrastive basis → adaptive
  controller → damage-aware optimization** and the older disagreement / temporal-localizer /
  utility-selector / factorized-gate roadmap is **superseded for the core paper** (`LEGACY DESIGN`).
  **Prior blocker (now resolved):** the updated method existed only in the contract docs / the DG-03R
  ticket, with no versioned proposal artifact, while `AGENTS.md`/`CLAUDE.md` still pointed to v5.
  **Resolution:** added the versioned proposal
  `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md` as the scientific source of
  record; repointed `AGENTS.md`, `CLAUDE.md`, and `METHOD_CONTRACT.md` to v6; marked v5 (and
  Implementation_Plan/ROUND_1_3/GATE_A) **SUPERSEDED / HISTORICAL**; and confirmed no active
  experiment gate requires the localizer / utility selector / disagreement / factorized gate. No
  code/tests/configs changed; no experiment run. Not yet `COMPLETE` — awaiting the independent final
  audit. See v6, `METHOD_CONTRACT.md` §4–§8/§10, `EXPERIMENT_MATRIX.md`, `CODE_MAP.md`.

- **DG-03 — COMPLETE / FROZEN.** Canonical v6 steering basis built & versioned at L16 and L24 on
  Whisper-large-v3 × CS-Dialogue (`steering_basis_v1`, `results/dg03/basis/`; construction Slurm jobs
  50452 then **50470** with complete provenance). Free-decoding causal screen on D-dev-select (final
  jobs **50471/50472**) selected **L24** by the pre-registered rule (`DG03_BASIS_CAUSAL_SPEC.md` §8,
  §11): intended `v_local` at oracle embedded steps gives U=+33 (61 corrections vs 28 corruptions,
  PIER gain +0.0146, matrix-CER 0.2287 vs baseline 0.2261), beating sign-reversed (−43),
  matched-random (−7), and count-matched wrong-location (+10); L16 showed no useful headroom (U=−6).
  Dose fixed pre-run at ρ=1×s_ℓ (no tuning). **Audit blockers resolved** (repair commits
  `5465e04`/`12f0cac`; final artifact freeze `212a06f`): basis
  carries a real `dataset_fingerprint`/`construction_config_hash`; every screen condition is a
  complete validated `result_v1` (metrics+gains+POI transitions+3 retention populations+reserved
  gate_coverage) with per-utterance texts; per-condition edit-count/total-energy recorded and C4
  count-matched to C1. Canonical `outside_harm = n_corrupted_outside` is now populated in every
  C0–C4 result from the stored hypotheses and frozen D-dev-select candidate spans by CPU-only
  post-processing; it is correctness-flip harm, not outside transcript edits. No GPU rerun was
  needed. The added candidate utility is diagnostic only; the frozen selection U remains POI
  corrections − corruptions. Pre-run commit `749759b`; GPU jobs 50470–50472 completed; focused
  DG-03/DG-01 validation: **34 passed**.

- **DG-04 — COMPLETE / FROZEN.** Frozen exact-site baselines B0/B1/B2/B3 ran on the same 300-utterance
  D-dev-select population at L24 with the pre-registered `ρ={0.5,1.0,2.0}` grid. Jobs **50483** and
  **50484** completed on `mig`; all 10 frontier points validate as `result_v1` with canonical outside
  harm and realized-energy accounting. B3 calibration used 300 shortest `router-calib` utterances
  (τ=1.2037518, T=1.9582404) and no runtime reference information. The frozen POI rule selects
  **B1 ρ=0.5** (`U=38`, 135 corrections / 97 corruptions, PIER gain +0.01675); B3 ρ=0.5 is the
  other positive point (`U=33`). No adaptive/controller/training work was run. Freeze artifacts:
  `results/dg04/frontier.json`, `results/dg04/reference.json`; runner commit `33799da`, audit
  hardening commit `a4f14be`; final freeze commit `2ed6cf1`; focused DG-04 tests: **7 passed**.

- **DG-05 — COMPLETE / FROZEN.** DG-05B trained the fixed L24 controller once with
  seed 42 in job **50507** (`mig`, H100 3g/40GB) after three failed execution attempts
  (50489/50505/50506). Training used only `loc-train ∪ util-train`, correction-only CE on a
  validated 51,227-position `C_E`, and no retention/anchor/gate losses. All three checkpoints
  were free-decoded on the same 300-utterance D-dev-select population as DG-04; epoch 3 was
  selected by the frozen utility → PIER gain → energy rule. A2 epoch 3 has utility +91 (135
  corrections / 44 corruptions), PIER gain +0.04012, embedded WER gain +0.03219, matrix CER
  gain −0.02745, outside harm 1,142, embedded retention 0.9634, matrix retention 0.9040, and
  total energy 63,926.3 (18,812 realized edits; mean energy 3.398). It improves the DG-04 B1 rho=0.5 point (utility +38; 135/97;
  PIER gain +0.01675; energy 82,007.1; matrix retention 0.8930) while showing a present damage
  signal that motivates DG-06. Classification: **VALID ADAPTIVE SIGNAL WITH DAMAGE**. The
  accepted run executed at source commit `6f9c7506d2c39b0723070537c9403f5be9cd935a`, an
  ancestor of audit HEAD `bd533039733971be1860fd75d967cfd26cbd09e4`. Complete
  artifacts/provenance are in `results/dg05/controller/`; selected checkpoint hash is
  `sha256:6d9390dee19097110bbe3f9ed6707ee72e83e93722cf448bd075c8c3f40dfbcc`.

- **DG-06 — COMPLETE / FROZEN.** Damage-aware training added on
  top of the frozen DG-05 controller (`src/csasr/steering/dg06_losses.py`,
  `experiments/dg06_damage_aware.py`; spec `DG06_DAMAGE_AWARE_SPEC.md`). Two `mig` GPU jobs from
  pre-run commit `16f4578` trained the fixed L24 controller from the **identical** reconstructed
  DG-05/D0 init (`sha256:25ebe8e4…156f`, D1==D2), seed 42, β fixed, `λ_M=λ_E=1.0`, C_E reused
  verbatim from DG-05, retention populations `R_E` (11,748 pos) / `R_M` (151,556 pos) built from the
  reused frozen baseline. Checkpoints selected by free decoding on the same 300-utt D-dev-select
  population; both selected epoch 1.
  - **D0** (reused DG-05 epoch 3): net 91 (135 corr / 44 corrupt), PIER gain +0.04012, **MER gain
    −0.01919**, **matrix-CER gain −0.02745**, outside harm 1,142, embed ret 0.9634, matrix ret
    0.9040, energy 63,926 (gate mean 0.823).
  - **D1** correction+matrix retention (job **50558**): net **85** (116/31), PIER gain +0.0375,
    **MER gain +0.0254**, **matrix-CER gain +0.0239**, outside harm **316**, embed ret 0.9742,
    matrix ret **0.9745**, energy 35,575 (gate mean 0.446).
  - **D2** full damage-aware (job **50559**): net 68 (83/15), PIER gain +0.0300, MER gain +0.0250,
    matrix-CER gain +0.0243, outside harm **295**, embed ret **0.9875**, matrix ret 0.9756, energy
    35,920 (gate mean 0.451).
  Both retention variants **flip MER and matrix-CER from degrading (D0) to improving**, cut outside
  harm ~72–74% and corruptions, and roughly halve gate strength/energy. D2's embedded-retention
  objective further raises embedded retention (0.9742→0.9875) and lowers corruption/outside harm but
  sacrifices corrections (116→83), lowering net utility below D1. Frozen selection (highest utility)
  ⇒ **`SELECT D1`** under the predeclared utility → matrix-retention → embedded-retention → energy
  rule. D2 adds measurable embedded and matrix retention benefit while retaining positive utility;
  outcome **`FULL DAMAGE-AWARE SUCCESS`**. Gate penalty and basis refinement remain deferred
  (DG-06 core forbids them). Focused tests: **27 passed** (15 DG-06 + 12 DG-05, CPU), plus the
  exact-site/site-infrastructure checks passed in the audit environment. No D-dev-confirm / D-test
  read; no post-hoc λ/β/LR tuning.

- **DG-07A — COMPLETE / FROZEN PRE-RUN.** The required comparison matrix is frozen in
  `docs/current/DG07_BASELINES_ABLATIONS_SPEC.md`: LB1 exact-site SALSA global, LB2 matched-budget
  Q/V LoRA, A1 local-only, A2 conditioning-only, and A3 fixed-mixture gate. All new methods share
  the selected DG-06 D1 objective (`L_corr(C_E) + λ_M L_ret,M`, λM=1), seed 42, three-epoch budget,
  and greedy `D-dev-select` checkpoint selection. A5 refined basis is deferred because no λA was
  predeclared. Implementations, CPU focused tests, and `mig`-only launch batches are committed;
  no DG-07 GPU job has been submitted.

- **DG-07B — READY FOR INDEPENDENT AUDIT.** The frozen five-run matrix completed on `mig` with
  one development seed and greedy D-dev-select decoding. Accepted jobs are **50592** (LB1),
  **50604** (LB2 exact-config rerun after a mechanical N/A-energy tie-break fix), **50612** (A1),
  **50613** (A2), and **50642** (A3); failed job **50593** was the pre-fix LB2 software attempt.
  All runs used `loc-train ∪ util-train`, seed 42, 2,019 optimizer updates, and the frozen DG-06
  D1 objective. A3 produced valid per-epoch `result_v1` records but no selected checkpoint because
  all three utilities were non-positive under the predeclared positive-utility rule. The compact
  canonical summary is `results/dg07/summary_v1.json`; attempts and full per-method artifacts are
  in `results/dg07/`. Focused DG-07 tests: **17 passed**. No D-dev-confirm or D-test data was read.

**Current ticket:** `DG-07 — READY FOR INDEPENDENT AUDIT` (do not begin DG-08 or DG-07C).

---

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
3. `q`, `u_source`, pre-intervention `r`, repaired `r̃`, and effective `u^S = r̃-r` are defined in
   MC §1–§2 and §7; zero-gain positions must be bit-identical and `β=0` is a no-op.
4. Decoder candidate layers are `{16, 24}`.
5. The frozen basis is `V^0 = [v_local, v_cond]`; legacy directions (including codebase `v_nat`)
   remain labeled legacy and must not be silently relabeled.
6. The inference-safe feature allowlist is canonical; the factorized gate is superseded `LEGACY
   DESIGN`, not a current contract component.
7. Headline evidence requires free decoding; teacher-forced results are screening only.
8. `D-test` has permitted reference/C00 exposure but no steered exposure found.
9. Encoder–decoder disagreement is **optional supporting analysis** only (MC §5), not a decision
   rule or fallback gate.
10. Correction, English retention, Mandarin retention, and monolingual retention populations are
    distinguished (MC §8).
11. Metric sign convention: `PIER_gain = PIER_baseline − PIER_method` (positive = improvement).
    Legacy artifacts with opposite signs must be converted before comparison.

## Implementation gaps

> **DG-03R note:** this list predates the DG-03R reconciliation and is retained as historical
> context. **G1 is resolved** (exact-site hook FROZEN, DG-02). **G5 is superseded** — the temporal
> localizer / utility selector / abstention / factorized gate are `LEGACY DESIGN`, replaced by the
> adaptive controller. The authoritative post-DG-02 gaps are `METHOD_CONTRACT.md` §13 and
> `CODE_MAP.md` §2 (basis builder, controller, damage-aware losses, scientific exact-site runner).

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

## Historical open scientific decisions (pre-DG-03R; retained for provenance only)

> **DG-03R note:** superseded by `METHOD_CONTRACT.md` §12 (reconciled). Items about the
> disagreement gate, factorized-gate algebra, and conditioning-subspace rank are no longer open for
> the core paper; the live open decisions are layer selection (DG-03 screen), controller
> architecture (DG-05), and training weights/schedule (DG-06).

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

**DG-02 — exact post-cross-attention/pre-FFN intervention site: COMPLETE / FROZEN.** Implementation
+ CPU/synthetic validation green, and the real-model acceptance passed at **both L16 and L24** on
whisper-large-v3 (Slurm job **50369**, `mig` H100 3g.40gb, COMPLETED exit 0; artifact
`results/dg02_real_acceptance.json`, verdict PASS). Spec at
`docs/current/DG02_INTERVENTION_SITE_SPEC.md` (§17 delivered implementation, §18 acceptance
evidence). Implementation commit `4e10399`.

Real-model acceptance summary (utterance `ZH-CN_U0091_S0_68`, `D-dev-select`, 2.525 s, bf16):
β=0 token+transcript identity (steered_calls=0), `r=q+u` at bf16 rounding scale (L16 ≤3.1e-3, L24
≤7.8e-3, both ≪ tol), exact-site `err_vs_block_gap ≈ 0.008 ≪ 1` (site not block output),
forced-prefix zero-edit + eligible edit, cache positions `[0..8]` monotonic/no-reset/boundary-aligned,
norm preservation rel dev ≤ 4.6e-4. No layer/direction/β selection (DG-03).

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

Freeze evidence: the real-model acceptance was executed on a GPU node (Slurm job 50369, `mig`
partition) after the CPU dev box proved too small to load the model; it reported `PASS` at both
layers, so DG-02 is frozen. No DG-02 bug surfaced on the real model — no code changed between the
CPU-green state and acceptance.

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

Current post-DG-06 ticket: **DG-07 — learned baselines and core ablations.** The
historical roll-up below is retained for provenance.

DG-02 is frozen (real-model acceptance PASS, job 50369). Next ticket:

**DG-03 — steering basis construction and causal validation: COMPLETE / FROZEN.** Basis
construction and both free-decoding causal screens completed; the independent-audit blockers
(basis provenance, complete `result_v1`+retention, C4 edit-count/energy) are resolved (commit
`eac2e37`) and re-run (construction 50470; screens 50471/50472). **SELECT L24.** Controller and
training work remain separate DG-04+ tickets.
