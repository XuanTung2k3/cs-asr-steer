# DATA EXPOSURE

Companion to `METHOD_CONTRACT.md` and `EXPERIMENT_MATRIX.md`. This records what each role has
already been used for. No inspected development or test role is described as untouched.

Two partition schemes coexist in `configs/lss/roles.yaml`:

- **legacy v1** assigns conversation/speaker halves and passes the historical
  `dev_select`/`dev_confirm`/`test` rows through to the corresponding D-roles. It is not
  dialogue-atomic.
- **dialogue-v2** assigns both sides of each dialogue together. It pools the official train and
  development dialogues for non-test roles and keeps the 15 official-test dialogues as `D-test`.

The schemes provide analogous *roles*, not row-equivalent splits. Counts from one scheme must not
be attached to the other.

## Dialogue-v2 role artifacts

The configured and published role tree is
`/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles`. Its role report records:

| Role | Dialogues | Conversations | Utterances | Hours |
|---|---:|---:|---:|---:|
| `D-construct` | 20 | 40 | 7,169 | 18.589 |
| `loc-train` | 15 | 30 | 6,050 | 16.479 |
| `util-train` | 12 | 24 | 4,718 | 11.634 |
| `router-calib` | 8 | 16 | 3,031 | 7.968 |
| `D-dev-select` | 20 | 40 | 7,919 | 22.692 |
| `D-dev-confirm` | 10 | 20 | 3,538 | 9.560 |
| `D-test` | 15 | 30 | 6,257 | 16.638 |

The locked test artifact exists as `locked/role_D-test.parquet` with its sidecar. A file named
`role_D-test-lock.parquet` is not part of this artifact layout.

## Role use under the updated method (DG-03R, 2026-09-07)

The dialogue-v2 partition is **unchanged** — DG-03R does **not** repartition data and does **not**
rename any physical role artifact. Logical use of each role under the contrastive-basis / adaptive
-controller / damage-aware-training method (MC §4–§8):

| Role | Logical use (updated method) |
|---|---|
| `D-construct` | steering-**basis construction** (`v_raw`, `v_cond`, `v_local`, `V^0`) |
| `loc-train`, `util-train` | **controller-training pool** (`loc-train ∪ util-train`); historical names + row assignments preserved |
| `router-calib` | calibration / hyperparameter support; DG-04 B3 projection-gate calibration only |
| `D-dev-select` | layer (L16 vs L24), steering-strength, model/checkpoint **selection**; incl. DG-03 causal screen and DG-04 frontier |
| `D-dev-confirm` | **confirmation only**, never new selection |
| `D-test` | locked final intervened evaluation **after the entire Whisper pipeline is frozen** |

All prior exposure records below are **preserved and still binding**; no exposed split is described
as untouched. The legacy training exposure (Job-B post-FFN gate/direction training) remains legacy —
the controller / basis builder / damage-aware losses are not yet implemented or trained.

## Per-role exposure

### `D-construct`

- **Role:** direction and conditioning-subspace construction.
- **Exposure:** used by exact-site v2r3 direction construction and prior direction diagnostics.
  **DG-03 (2026-09-07):** used to build the v6 `steering_basis_v1` at L16/L24 — 233 baseline-correct
  embedded spans / 125 utterances / 20 dialogues; 125 conditioning pairs (en vs zh prefix). Slurm
  jobs 50452 and final provenance rerun 50470; artifacts `results/dg03/basis/`. Construction only;
  not a selection or claim split.
- **Status:** touched for construction; not a selection or claim split.

### `loc-train`, `util-train`, `router-calib`

- **Roles (updated method, v6):** `loc-train ∪ util-train` is the **controller-training pool** (row
  assignments and physical names preserved); `router-calib` is calibration/hyperparameter support
  **only if** a calibration step is introduced. The former "localizer training / utility-selector
  training" role labels are `LEGACY DESIGN` — the localizer / utility-selector / disagreement-gate
  design is superseded (v6 §Appendix A; MC §5–§6).
- **Observed use:** the legacy Job-B trainer sampled 500 rows from `loc-train ∪ util-train` to train
  its own post-FFN monolithic gate/direction systems — legacy, not the v6 controller.
- **DG-04 calibration exposure (2026-09-07):** the 300 shortest `router-calib` utterances were used
  for the frozen B3 projection-gate calibration only (teacher-forced exact-site states; Slurm job
  50484). No D-dev-confirm or D-test data was read.
- **DG-05A/B state (2026-09-07):** the fixed-basis adaptive controller was trained once with seed
  42 in Slurm job **50507** (partition `mig`). A frozen Whisper free-decoding baseline over the
  `loc-train ∪ util-train` pool produced and validated `results/dg05/correction_set_v1.json`
  (4,064 utterances / 51,227 token positions); controller gradients used only these `C_E`
  positions. Checkpoint selection used only `D-dev-select` (300 utterances, greedy, temperature
  0, beam 1), selecting epoch 3 by the predeclared utility → PIER gain → energy rule. Jobs
  50489/50505/50506 are failed recovery attempts and opened no additional roles. No
  D-dev-confirm or D-test rows were read. Dialogue-v2 `router-calib` exists, but its role
  report/config has no `calib-prob`/`calib-thresh` sub-roles. Only legacy v1 config defines a
  10/10 calibration split.
- **Status:** the training roles now have DG-05B baseline/training exposure in addition to their
  legacy exposure; physical assignments are unchanged. Whether the v6 controller needs a
  `router-calib` subdivision remains deferred (only if a calibration step is added).

### `D-dev-select`

- **Role:** layer/dose and development-system selection.
- **Exposure:** used by legacy E-series/v2r3 screens and by the current 84-cell Round-1 frozen
  sweep. The sweep population is a 300-utterance candidate subset spanning 20 dialogues, not all
  7,919 dialogue-v2 role rows. Job-A F5 also calibrates its legacy projection gate here.
- **DG-02 acceptance (integration/debug, no claim):** `experiments/dg02_real_acceptance.py` reads
  **one** `D-dev-select` utterance (the shortest in the candidate manifest) purely to verify
  exact-site plumbing (β=0 token identity, `r=q+u`, prefix/cache/norm) at L16 & L24. It makes no
  ASR-quality judgement, no layer/direction/β selection, and no confirmatory measurement. The run
  executed on a GPU node (Slurm job **50369**) on exactly **one** utterance,
  **`ZH-CN_U0091_S0_68`** (duration 2.525 s), recorded in `results/dg02_real_acceptance.json`
  (verdict PASS). This is debug/integration exposure on an already heavily-touched selection split
  — it does not add a new claim surface and is not confirmation/test data.
- **DG-03 intervention exposure (2026-09-07):** used for the free-decoding causal/specificity screen
  that **selected L24** — 300 candidate utterances, conditions C0–C4 with the frozen DG-02 hook at an
  oracle diagnostic gate (initial jobs 50453/50454; final jobs 50471/50472; `results/dg03/screen/`).
  Canonical outside-harm fields were reconstructed offline from these stored hypotheses and the
  same frozen candidate alignments; this added no data exposure and no new subset. This is **intervened**
  development exposure on the selection split (its designated purpose: layer/strength/checkpoint
  selection); no confirmatory or test claim may be read from it.
- **DG-04 frontier exposure (2026-09-07):** the same 300-utterance candidate population was used
  for B0/B1/B2/B3 free-decoding baselines across the frozen `ρ={0.5,1.0,2.0}` grid (Slurm jobs
  50483/50484). This is development frontier/operating-point selection exposure only; no new subset
  was created and no D-dev-confirm or D-test intervention occurred.
- **Status:** heavily touched for selection (now including DG-03 intervened layer selection); no
  confirmatory claim may be read from it.

### `D-dev-confirm`

- **Role:** one confirmation freeze after all selection and calibration.
- **Exposure:** prior E4/E5 and v2r3 analyses used confirmation-role data. The completed legacy
  `results/job_a/summary.json` reports F0–F5 on a 150-utterance candidate subset spanning all 10
  dialogues. The current `round1_frozen.py` wrapper does not consume its YAML `confirm_split` and
  therefore does not establish a new contract-compliant confirmation run.
- **Status:** touched; it must not be reused for new selection.

### `D-test` — LOCKED FOR INTERVENTIONS

- **Role:** final evaluation in one frozen batch.
- **Permitted exposure already completed:** reference-only counts and the frozen C00 baseline were
  used for power analysis: 176,345 reference units, 35,921 embedded-English units (20.37%), C00
  assigns 27.12% of error mass to embedded-English units, and MDE is approximately 0.0608 at
  `G=15` (`docs/RESULTS_RUN_7DAYS.md`). A unit-tagging bug was discovered through those counts.
- **Intervention exposure:** no steered `D-test` result artifact was found. Production/smoke C00
  baseline artifacts are present and are within the permitted exposure.
- **Status:** not untouched, but still locked for intervened evaluation. The observed MDE is about
  four times the largest prior oracle effect, so the split is underpowered for effects of that
  size; this is a scientific limitation, not grounds to redraw or inspect alternatives.

### SEAME

- **Role:** optional second-corpus same-pair transfer.
- **Exposure:** none found; current Round-1 runners do not consume it.
- **Status:** optional and unimplemented in the current Round-1 path.

## Leakage guards and do-not rules

- `csasr.lss.features_contract.assert_inference_safe` rejects forbidden reference-, alignment-,
  gold-, utility-, and outcome-derived columns and rejects unlisted columns. It guards feature
  columns, not a data split.
- Dialogue-v2 keeps both conversation sides of a dialogue in one role. Legacy v1 does not provide
  that dialogue-level guarantee.
- Do not read confirmation or test claims from `D-dev-select`.
- Do not perform new selection on `D-dev-confirm`.
- Do not produce or inspect intervened `D-test` output before the entire pipeline is frozen.
- Before gate calibration, create and freeze dialogue-v2 `calib-prob`/`calib-thresh` sub-roles or
  obtain a scientific decision that replaces that requirement.
