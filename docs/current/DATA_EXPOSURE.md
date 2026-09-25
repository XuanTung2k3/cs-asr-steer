# DATA EXPOSURE

**Inference-time P0 exposure, 2026-09-24:** The separate `feature/inference-cf-steering`
feasibility diagnostic used only the dialogue-v2r3 `D-dev-select` role, frozen DG-04 B0
transcripts, and its evaluator-only POI/reference units to select 60 positions (20 wrong
English, 20 correct English, 20 Mandarin). The corrected K=1 run retained 57; three frozen-baseline
text mismatches were skipped. The one authorized K=3 repair (P0-R1, job 54712) **reused the exact
same 60-position panel and permutation byte-identically** (hash-checked; no new example selection)
and retained the same 57. No P0/P0-R1 confirmation/test inference or P1 run occurred; no new split
or role was touched. See `FEASIBILITY.md` for source hashes and Slurm jobs 54702/54703/54712.
Prior exposure below is unchanged.

**Inference-time P0-R2 exposure (run 2026-09-24, Slurm 54758; outcomes inspected after the frozen run):** the frozen R2 panel is the same
300 already-exposed DG-04 B0 `D-dev-select` utterances (20 dialogues), decoded densely by the
unsteered forced-ZH baseline plus same-prefix forced-EN and diagnostic forced-RU replays. Evaluator
only: role references, MMS-FA/CTC unit times (`candidates_existing_ctc.parquet`) and the B0_AUTO
comparator transcripts. Pre-run design/implementation checks read only these baseline texts,
references and CTC times (no gate value). No D-dev-confirm, D-test, SEAME, ASCEND use. **P1 (jobs 54770, 54781):** 10 of the same already-exposed D-dev-select utterances (R2 panel positions 0,30,…,270), decoded unsteered (α=0) and with the α=1.0 engineering edit; the R2 `generate()` tokens were compared only as a diagnostic. No references entered P1 and no recognition outcome selected any P1 setting. **Pre-P2 CE (jobs 54801, 54802):** the same 10 P1 utterances; tokens/logits/edits only, no references. **P2 compact development (spec v1.1; P2-A attempt 1 jobs 54803/54804 invalid and unused; P2-A r1 jobs 54821/54822; P2-B/P2-C jobs 54824/54825):** the same 300 already-exposed R2-panel `D-dev-select` utterances (20 dialogues). Runner reads the reference-free panel; the CPU evaluator reads role references to compute PIER/MER/ZH-CER/EN-WER/retention **and selects layer, α, dose map from them** (selection exposure). No D-dev-confirm, D-test, SEAME, CS-FLEURS, ViMedCSS or Qwen use. **P2-R mechanism diagnosis (jobs 54855 abort, 54871 invalid, 54874 valid; debug 54861/54870 outside results, rows deleted unread):** 80 of the same already-exposed D-dev-select utterances (20 dialogues). Evaluator-side use of role references, R2 alignment and CTC midpoints to define diagnostic positions, reference-consistent target tokens and oracle placement (offline diagnostic only, never a deployable input); no selection of any method setting. `router-calib` was not used (only its utterance IDs were read for the overlap check, as were D-dev-confirm IDs); no D-test, P3 or transfer data.

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
- **DG-06 state (2026-09-08):** damage-aware controllers D1 (job **50558**) and D2 (job **50559**)
  were trained on the same `loc-train ∪ util-train` pool (`mig`). C_E was **reused verbatim** from the
  frozen DG-05 artifact (hash-checked). The baseline-**correct** retention populations `R_E`
  (2,111 utts / 11,748 positions) and `R_M` (7,023 utts / 151,556 positions) were built once from the
  **reused frozen DG-05 free-decoding baseline** (`results/dg06/retention_set_v1.json`,
  `sha256:ad2d38a9…585d`) — no new decode of the training pool. KL-to-baseline `p_0` was recomputed
  per batch from the frozen backbone under teacher forcing (no new data). Checkpoint selection used
  only `D-dev-select` (300 utts, greedy/temp-0/beam-1). No D-dev-confirm or D-test rows were read.
- **DG-07B state (2026-09-08):** LB1 SALSA, matched-budget LB2 LoRA, A1 local-only, A2
  conditioning-only, and A3 fixed-mixture gate trained on the unchanged `loc-train ∪ util-train`
  pool with seed 42 and the selected DG-06 D1 objective. Accepted Slurm jobs were **50592**,
  **50604**, **50612**, **50613**, and **50642**; job **50593** was a failed LB2 software attempt
  before a mechanical N/A-energy selection fix. Checkpoint selection used only `D-dev-select`
  (greedy, temperature 0, beam 1). No D-dev-confirm or D-test rows were read; no additional
  dataset or repartition was introduced. Full provenance is in `results/dg07/summary_v1.json`.
- **Status:** the training roles now have DG-05B and DG-06 baseline/training exposure in addition to
  their legacy exposure; physical assignments are unchanged. Whether the v6 controller needs a
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
- **DG-06 selection exposure (2026-09-08):** the same 300-utterance candidate population was used to
  free-decode the D1/D2 damage-aware checkpoints for the frozen utility→PIER→matrix-retention→energy
  selection rule (jobs 50558/50559). Development checkpoint-selection exposure only; no new subset;
  no D-dev-confirm or D-test intervention.
- **DG-07 selection exposure (2026-09-08):** the same 300-utterance candidate population was used
  to free-decode all five learned/ablation methods for the frozen checkpoint rule (jobs 50592,
  50604, 50612, 50613, 50642). A3 had no eligible checkpoint because all recorded utilities were
  non-positive; its per-epoch `result_v1` records remain preserved. No new subset was created and
  no D-dev-confirm or D-test intervention occurred.
- **Status:** heavily touched for selection (now including DG-03 intervened layer selection and
  DG-05/DG-06 controller checkpoint selection); no confirmatory claim may be read from it.

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
- **DG-08 FINAL INTERVENED EVALUATION EXPOSED (2026-09-09).** After the hard D-test lock
  (`results/dg08/DG08_TEST_LOCK.json`, commit `1fb2c3`, committed BEFORE any D-test decode), the
  frozen finalists F0–F4 (learned F2/F3/F4 at seeds 13/42/73) were evaluated once on the entire
  locked `D-test` manifest (6,257 utts / 15 dialogues) under greedy and beam-5 decoding
  (`partition=mig`; jobs 50725/50733/50734 greedy, 50760/50767/50768/51100/51101/51102 beam-5). This
  is the sanctioned single final evaluation after full pipeline freeze; no selection, checkpoint,
  seed, hyperparameter, decoder, normalization, or metric was chosen or changed from D-test outcomes.
  D-test carries no candidate/POI alignment artifacts, so outside-harm/candidate-utility were not
  computed on it (text metrics only). **`D-test` must not be used for any later selection or tuning.**
- **Status:** `D-test` is now the exposed final intervened evaluation split; locked, one-shot,
  evaluation-only. The observed MDE (~4× the largest prior oracle effect) means the 15-dialogue
  bootstrap is underpowered; this is a reported scientific limitation, not grounds to redraw the split.

### SEAME

- **Role:** optional second-corpus same-pair transfer.
- **Exposure:** none found; current Round-1 runners do not consume it.
- **Status:** optional and unimplemented in the current Round-1 path.

### ASCEND (CAiRE/ASCEND) — added for BASIS-A6 (2026-09-21)

- **Role:** third corpus for BASIS-A6 (`BASIS_A6_ASCEND_DATA_SPEC.md`). Official HF splits map to
  BASIS-A6 roles as: `train` → `ASCEND-construct` (construction source only, frozen size-matched
  subset), `validation` → `ASCEND-eval` (evaluation panel, target 300 or all eligible),
  `test` → **DO NOT READ / DO NOT USE**.
- **Exposure:** none yet. Protocol frozen; the download script (`scripts/download_ascend.py`) and
  subset manifests (`ASCEND_CONSTRUCT_MANIFEST.json`, `ASCEND_EVAL_MANIFEST.json`) are the
  reproducible data path. No model inference, no eligibility scan, and no subset freeze have run.
- **Guards:** eligibility requires *verified* transcript-level code-switching (≥1 EN content unit
  AND ≥1 ZH content unit) via the canonical `language_tags` pipeline, not the dataset "mixed"
  label. Construction=train only, eval=validation only, disjoint by split. `test` is guarded by
  the download marker `DO_NOT_USE_TEST.marker` and adapter refusal.
- **Status:** frozen protocol, unexposed. ASCEND `test` is locked out of BASIS-A6 entirely.

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
- ASCEND `test` must never be read for BASIS-A6. Construct only from ASCEND `train`; evaluate only
  on ASCEND `validation`. Do not enter any `validation`/`test` utterance into construction.
