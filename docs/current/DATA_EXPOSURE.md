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

## Per-role exposure

### `D-construct`

- **Role:** direction and conditioning-subspace construction.
- **Exposure:** used by exact-site v2r3 direction construction and prior direction diagnostics.
- **Status:** touched for construction; not a selection or claim split.

### `loc-train`, `util-train`, `router-calib`

- **Roles:** proposed localizer training, utility-selector training, and calibration.
- **Observed use:** the legacy Job-B trainer samples 500 rows from `loc-train ∪ util-train` to train
  its own post-FFN monolithic gate/direction systems. That is not the contract localizer or
  outcome-supervised utility selector.
- **Implementation state:** the contract temporal localizer and selector are not implemented or
  trained. Dialogue-v2 `router-calib` exists, but its role report/config has no
  `calib-prob`/`calib-thresh` sub-roles. Only legacy v1 config defines a 10/10 calibration split.
- **Status:** the training roles have legacy exposure; calibration for the finalized gate remains
  proposed.

### `D-dev-select`

- **Role:** layer/dose and development-system selection.
- **Exposure:** used by legacy E-series/v2r3 screens and by the current 84-cell Round-1 frozen
  sweep. The sweep population is a 300-utterance candidate subset spanning 20 dialogues, not all
  7,919 dialogue-v2 role rows. Job-A F5 also calibrates its legacy projection gate here.
- **DG-02 acceptance (integration/debug, no claim):** `experiments/dg02_real_acceptance.py` reads
  **one** `D-dev-select` utterance (the shortest in the candidate manifest) purely to verify
  exact-site plumbing (β=0 token identity, `r=q+u`, prefix/cache/norm) at L16 & L24. It makes no
  ASR-quality judgement, no layer/direction/β selection, and no confirmatory measurement. The
  concrete `utterance_id` is recorded in `results/dg02_real_acceptance.json` when the run executes
  (on a GPU node; not yet run in the CPU dev environment). This is debug/integration exposure on an
  already heavily-touched selection split — it does not add a new claim surface.
- **Status:** heavily touched for selection; no confirmatory claim may be read from it.

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
