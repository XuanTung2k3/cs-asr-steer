# CLAUDE.md — CS-ASR steering

**Start in `docs/current/`. Read `AGENTS.md` — it is the shared contract for every agent and
applies to you in full.** The authoritative files:

- `docs/current/METHOD_CONTRACT.md` — the locked method. If code disagrees, the code is wrong
  until a human revises this file. Never resolve a contradiction in favour of existing code.
- `docs/current/CODE_MAP.md` · `EXPERIMENT_MATRIX.md` · `DATA_EXPOSURE.md` · `STATUS.md`.

Proposal — **current authority is v6** (do not duplicate here):
`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md` (contrastive basis →
adaptive controller → damage-aware). v5 (`…Method_First_Proposal_v5.md`) is **SUPERSEDED /
HISTORICAL** and must not override v6 post-DG-02.

## Non-negotiables (see AGENTS.md for the full list)
- **Scientific invariants:** decoder post-cross-attention pre-FFN site (not post-FFN), **FROZEN
  (DG-02)**; no `sqrt(num_layers)` rescale; norm-preserving repair; candidate layers `{16, 24}`.
  Core method (updated proposal, DG-03R): frozen basis `V^0 = [v_local, v_cond]` → adaptive
  controller `f_θ(LN(r_t)) → (g_t, π_t)` (per-position direction is rank-one) → damage-aware
  training (correction-set CE + KL retention). The disagreement/localizer/utility-selector/
  factorized-gate roadmap is **superseded** (LEGACY). See `METHOD_CONTRACT.md` §4–§8.
- **Data-role separation:** selection→`D-dev-select`, calibration→`router-calib`, one freeze on
  `D-dev-confirm`, **no intervened `D-test`** before full freeze (`D-test` is locked + underpowered).
- **Preserve legacy** files, results, and docs — do not delete or rename.
- **Free-decoding** for claims; teacher-forced is screening only. Honour the safe-claim boundaries.
- Mark `OPEN DECISION` / `IMPLEMENTATION GAP` in `METHOD_CONTRACT.md`; don't guess or silently patch.

## Practical
- Focused changes + focused tests in `tests/`; keep run manifests/provenance intact.
- Tests need the LD_LIBRARY_PATH prefix on this machine (see `README.md` "Tests"):
  `LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH .../python -m pytest -q`.
- No GPU jobs unless asked; Round-1 Slurm only; Round-2/3 are code-only.
- There is no end-to-end canonical free-decoding entry point yet. The exact-site reusable primitive
  is `csasr.lss.sites.DecoderPostCrossAttnSteeringHook`; `experiments/job_a_frozen.py`,
  `experiments/job_b_training.py`, their `round1_*` wrappers, and completed outputs are
  legacy/current-execution paths for this contract. See CODE_MAP before reusing `steer_sweep/`.
