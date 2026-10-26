# RC7-0E — Next-phase readiness (preparation only; nothing launched)

## 1. Admission summary

| Phase | Admission | Reason |
|---|---|---|
| RC7-1 baseline verification / reuse | **READY** (development scope) | 12/12 selected checkpoints and 187/187 core seals revalidated; source tree-identical; decoder/scorer shared |
| RC7-2 controller pilot | **NOT ADMITTED as a novel method** | novelty decision `PARTIAL_OVERLAP_ABLATION_ONLY` (`RC7_0_NOVELTY_DECISION.json`) |
| RC7-3 functional analysis | READY to specify | N4 D1/D2 repaired; predictions for all arms are sealed |
| RC7-4 multi-seed | BLOCKED on engineering | the only multi-seed path (`worker.n5`) is coupled to opening C500 |
| RC7-5 held-out | BLOCKED on owner decision | C500 partial historical exposure (`RC7_0_DATA_EXPOSURE_AUDIT.md` §6) |

`configs/rc7/rc7_2_proposed_not_executed.yaml` is **intentionally not created**. The candidate failed novelty
admission, and six scientific values remain unspecified (see the gate comparison, §4). Writing the file would
mean inventing them.

## 2. Repository fitness (A5000 implementation reused; no refactor)

| Requirement | State | Location |
|---|---|---|
| Shared base-model loading | yes | `csasr.models.whisper.load_whisper` in both `a5000_campaign.worker.Runtime` and `a5000_lra_sprint.runtime` |
| Explicit intervention interface | yes | `RepresentationAdapter` (`'ereft'` = LoReFT, `'lra'` = DiReFT), `SprintLRA`, `EncoderWeightLoRA` |
| LoReFT / DiReFT as separate named configs | partly | historical IDs `R`/`L` in code; scientific names only in docs (kept for artifact compatibility) |
| Modular controller policies | yes | `Allocation`, `LRAAllocation` (subclass; same operator) |
| Explicit objective | yes | `reft.a5000.objectives.weighted_ce` (CE = ω 1) |
| Canonical decode/score | yes | `Runtime.decode` + `a5000_campaign.evaluation.metrics`; sprint reuses them unchanged |
| Validated data roles | yes | `a5000_campaign.data.rows` (C500 requires a confirmation freeze) |
| Frozen, hashed configs | yes | `frozen_plan.json`, `config_sha256`, admission seals |
| Immutable outputs | yes | per-evaluation `SEALED.json` |
| Safe diagnosis/resume | yes | optimizer-boundary resume; continuation pattern in `reft/rc7/n4_continuation.py` |

Gaps to close before RC7-4/RC7-7. Each is a concrete defect, none needs a broad refactor:

1. **Seed replication without C500.** `n5` trains seeds 43/44 *and* opens C500 in one stage. A replication-only job
   (same recipe and frozen horizon, V300 only) is needed so seeds can be run before the held-out decision.
2. **Minimal adapter export** (LoReFT without the 1280² base buffer) for deployable-bytes measurement.
3. **Training-memory scope** recorded separately from decode memory.
4. **Frozen-model D1** on the same panels, so D1 can be read as adapted − frozen.
5. Two parallel runtimes (campaign, sprint) duplicate setup. Leave as is unless a defect appears.

## 3. RC7-1 specification (ready to review; not executed)

- **Arms:** B0; LoReFT CE `a4719921…`; DiReFT CE `c531a22a…`; LoRA-PM CE `6afd342c…`; LoRA-Strong CE `39c76746…`; plus
  their WCE counterparts (one-seed exploratory).
- **Verification:** re-hash checkpoints; reload; teacher-forced parity on a small T2000 panel; and, if a decode is required, a
  bitwise prediction-parity check on a fixed V300 subset against sealed predictions (the sprint AUDIT job already did
  this for LRA). No retraining. No C500.
- **Budget:** ≤ 2 GPU-h. Output: a versioned RC7-1 record next to, not inside, the archived roots.
- **Not in scope:** new selection or any claim of independent confirmation.

## 4. Recommended path in place of RC7-2

Skip controller expansion and prioritize compact adaptation and functional analysis:

1. RC7-1 as above.
2. Close gap 1, then (with authorization) run seed replication of LoReFT/DiReFT/LoRA-PM/LoRA-Strong CE on V300 to
   quantify seed variance. One seed cannot currently separate the three ~5K methods.
3. RC7-3 functional analysis from sealed predictions: correction/damage taxonomy, per-dialogue gains, the LoReFT CE
   non-CS degradation and loop, edit-magnitude placement. Mostly CPU work on existing predictions.
4. Owner decision on C500 eligibility (§6 of the exposure audit) before any RC7-5 freeze.

If the owner nevertheless wants the EMA-residual test, it must be registered as an **L4 feature ablation**:
- same DiReFT CE parent `c531a22a…`;
- 16 inputs / 289 parameters;
- `energy_coefficients` with ε 0.25, α 1, c 1;
- AdamW lr 5e-4, 125 updates, final checkpoint;
- comparators L4 and the ρ = 0 variant;
- the unchanged 0.5 pp genuine-CS-MER rule against each, ≤ 1 pp non-CS degradation, no new loops, allocation std ≥ 0.001;
- every §4 blocker in the gate comparison resolved first.

It would remain exploratory (V300 exposed, post-hoc).
