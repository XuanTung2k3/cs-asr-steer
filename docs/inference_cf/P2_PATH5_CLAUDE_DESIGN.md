# P2-PATH5 — implementation design (Claude; no Codex stage)

Normative: `P2_PATH5_SPEC.md`, `P2_PATH5_PANEL.json`, `configs/inference_cf/p2_path5.json`. This design is frozen with
the contract, before implementation or any outcome.

## Reuse map (no new scientific primitive)

- **A2:** `experiments/inference_cf_p2tta0.run_objective(bundle, guard, "A2", ...)` is the exact TTA1 sequence: adapt,
  then materialize, ordinary `forced_decode`, and restore in `finally`.
  - y_B for its descriptive common diagnostic is the row's live theta0 decode, with `valid_mask(y_B)`.
  - Independent live check: `inference_cf_p2tta0_audit.live_objective_check` on the first FIXED100 row.
  - Final-master arrays are compared with the FIXED100 archives.
- **Detector:** `consensus_guard.lockstep_detect` (byte-pinned).
- **Content G1:** `inference_cf_p2path3.online_g1` (byte-pinned), with A2 in its legacy adapted A4 slot. Its replayed
  detection must equal the logical detection.
- **Action types:** `eos_boundary.action_mode` (reference-free validation of the live proposals).
  `score_boundary` / `execute_boundary` are NOT called.
- **Infrastructure:** the PATH4 runner provides the two-instance / no-alias / non-LN / reset checks, manifest, and
  plumbing-child-commit pattern.
- **Analysis:** the PATH4 analyzer and the TTA1/P2SEQ primitives provide metrics and the bootstrap.

## New additive files

- `experiments/inference_cf_p2path5.py` — prepare / manifest / run / seal.
- `experiments/inference_cf_p2path5_analyze.py` — reference-free primary + opportunity gate; guarded secondary.
- `experiments/inference_cf_p2path5_audit.py` — independent prerun / post.
- `slurm/inference_cf_p2path5.sbatch` and `tests/test_inference_cf_p2path5.py`.
- Outputs go to `results/inference_cf/p2path5/`, with rows by FULL300 index `000..299`.

## Controller

```
online_g1a(b_t0, b_a2, g_t0, g_a2, enc, owner_row, counters, eos, a2_free):
    det = lockstep_detect(...)                  # one logical event
    if not det.trigger:      -> NO_TRIGGER, output = det prefix/termination (must equal a2_free)
    mode = action_mode(argmax_theta0, argmax_A2, ...)
    CONTENT_G1   -> rec = path3.online_g1(...); require rec.detector == det; output = rec.G1
    EOS_BOUNDARY -> ABSTAIN: output = a2_free (no decoder path created, A2 not mutated); controller disabled
```

`a2_free` is the row's own live ordinary A2 output, a runtime value with no reference or row ID. The controller source
must not contain audit tokens (partition, historical targets, references); the pre-run auditor checks its source.

## Job sequence

1. Load two independent bf16/eager instances once. They must have non-aliased LN storage and identical non-LN weights.
2. **FIXED100 phase 1** (all 100), per row:
   - encoder once, then a detached clone for adaptation;
   - theta0 `forced_decode` must equal the sealed B0;
   - live check on row 0;
   - `run_objective` A2: tokens/termination/text equal the sealed A2; fp32 masters equal the archive; bf16 effective
     state matches;
   - record the A2 state hash; keep the encoder and effective state.
   - Any failure → INVALID abort.
3. **FIXED100 phase 2:** materialize each saved state, then run `online_g1a`.
   - G1A must equal the OPP0 derived target.
   - Content rows must equal the sealed PATH4 `detector` / `decision` / `score_paths` / `G1`.
   - EOS and no-trigger rows must equal A2.
   - Restore.
   - Any failure → INVALID abort before NEW200.
4. **NEW200**, per row in FULL300 order: encoder; theta0 decode (B0); A2 `run_objective`; materialize; `online_g1a`;
   restore; checks.
5. **Runtime file:** barrier status and counters (updates, backwards, detector, triggers, content events, abstentions,
   rollouts, score paths, G1 decodes, changed rows), peak VRAM, timings.

## Analysis

- **Primary (reference-free).** Validity; NEW200 opportunity counts (N_DISAGREE, N_CONTENT_ELIGIBLE, N_EOS_ABSTAIN,
  N_G1A_CHANGED, N_CHANGED_DLG); theta0/A2 winners; the opportunity class for each row; and the gate result
  OPEN_REFERENCES or OPPORTUNITY_SPARSE.
- **Secondary.** Runs only if the seal is committed, the primary audit PASSes, and the gate is open. It computes the
  NEW200 metrics for B0/AUTO/A2/G1A, safety, strong retention, rescue/breadth, precedence, LODO, bootstrap, the
  opportunity-outcome map, the FULL300 secondary aggregate, and all rescue/harm rows.
- **If the gate fires:** a reference-free terminal record, `terminal_sparse.json`, holds the label
  `P2_PATH5_OPPORTUNITY_SPARSE`. No evaluator import.

## Tests (CPU)

- Panel: FULL300/FIXED100/NEW200 relation and hashes.
- A2 inheritance pins.
- G1A dispatch on toy and monkeypatched fixtures: content → original G1, bit-identical; both EOS orientations → abstain
  = A2 with no boundary scorer, no rollout and no extra decoder path; both-EOS / no-trigger; one event; disabled after
  abstention.
- No row IDs in the controller.
- Opportunity gate boundaries.
- 0.90 retention / I_A2 ≤ 0 rule, PIER/MER guards, breadth/concentration, label precedence (all combinations vs the
  auditor).
- Reference firewall, including no evaluator under OPPORTUNITY_SPARSE.
- Auditor independence.
- Plus the historical regression suites: TTA0, funnel, PATH1/2/3/4-R1, PATH4.

## Independent audits

- **Pre-run `PASS_TO_P2_PATH5`:**
  - frozen files at the freeze commit; panel/partitions rebuilt independently;
  - teachers re-derived; A2 pins;
  - controller allowlist and dispatch;
  - no boundary arbitration, G2 or A4;
  - barrier ordering; thresholds and gate arithmetic; truth tables; firewall; compute.
- **Primary phase:**
  - recompute the barrier identities against the sealed artifacts;
  - triggers from the argmax pairs; own dispatch classification;
  - content-G1 arithmetic; abstention identity;
  - counts, the opportunity gate and summaries.
- **Full:** recompute metrics, safety, retention, rescue, breadth, LODO, bootstrap and label with the canonical
  primitives only.
