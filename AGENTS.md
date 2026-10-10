# AGENTS.md — CS-ASR steering

Read the contract before touching anything. The scientific and coding contract lives in
**`docs/current/`** and is authoritative:

- `docs/current/METHOD_CONTRACT.md` — the locked method (site, tensors, layers, directions,
  scores, gate, repair, losses, claim boundaries, gates). **If code disagrees with this file, the
  code is wrong until a human revises the file.**
- `docs/current/CODE_MAP.md` — where each component lives; canonical vs reusable vs legacy.
- `docs/current/EXPERIMENT_MATRIX.md` — required / secondary / optional experiments and controls.
- `docs/current/DATA_EXPOSURE.md` — what each split has already been used for.
- `docs/current/STATUS.md` — current stage, gaps, blockers, next ticket.

**Current full scientific proposal (ACTIVE authority):**
`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md` — the
contrastive-basis → adaptive-controller → damage-aware method. The implementation contract that
operationalizes it is `docs/current/METHOD_CONTRACT.md`.
`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`
(+ Implementation Plan v1/v2, ROUND_1_3, GATE_A_*) is **SUPERSEDED / HISTORICAL** — its
localizer / utility-selector / disagreement / factorized-gate roadmap must **not** override v6 for
any post-DG-02 work. Do **not** paste the proposal into code or into these files.

## Invariants you must preserve
1. **Scientific invariants.** Intervention site = decoder post-cross-attention residual, pre-FFN
   (never the post-FFN block output) — **FROZEN (DG-02)**; no `sqrt(num_layers)` rescale;
   norm-preserving repair; candidate layers `{16, 24}`. Core method (updated proposal, DG-03R):
   frozen contrastive basis `V^0 = [v_local, v_cond]` (MC §4) → adaptive controller
   `f_θ(LN(r_t)) → (g_t, π_t)` with a **rank-one per-position** direction `normalize(V^0 π_t)`
   (MC §6) → damage-aware training (correction-set CE + KL retention, MC §8). The older
   disagreement / temporal-localizer / utility-selector / factorized-gate roadmap is `LEGACY
   DESIGN`, superseded for the core paper. Do not silently resolve a contradiction in favour of
   existing code — record it as an `IMPLEMENTATION GAP` in `METHOD_CONTRACT.md` and stop.
2. **Data-role separation.** Selection on `D-dev-select`, calibration on `router-calib`, one
   confirmation freeze on `D-dev-confirm`, and **no intervened `D-test` output** before the whole
   pipeline is frozen. `D-test` is locked and underpowered — see DATA_EXPOSURE. Never weaken a gate
   threshold to stay on schedule.
3. **Legacy outputs.** Do not delete or rename legacy files, results (`results/`, `out/`), or docs.
   Legacy hooks/directions/drivers stay in the tree; just keep them off contract paths.
4. **Teacher-forced ≠ transcript.** Action selection and headline claims come from free-decoding
   only (MC §9). Respect the safe-claim boundaries (MC §11).
5. **Implementation status.** There is no end-to-end contract-conformant free-decoding runner yet.
   `experiments/job_a_frozen.py`, `experiments/job_b_training.py`, and the `round1_*` wrappers are
   legacy/current-execution paths for this contract; their completed results must remain labeled
   legacy. Reuse the exact-site primitives identified in CODE_MAP rather than relabeling these runs.

## How to work
- Make focused changes; extend the focused tests in `tests/` (e.g. hook/site reconstruction, PIER
  identity, feature-allowlist, split disjointness). Prefer the canonical modules in CODE_MAP.
- Every new contract run must write a **manifest** (resolved config, environment, git state, model
  metadata, hashes, status) using `csasr.utils.provenance` / `csasr.lss.specfreeze`. Current
  Round-1 result directories do not meet this requirement and are an implementation gap.
- Do not run GPU jobs unless explicitly asked; Round-1 Slurm only, Round-2/3 are code-only.
- Mark unsettled choices `OPEN DECISION` and code/method mismatches `IMPLEMENTATION GAP` in
  `METHOD_CONTRACT.md`; do not guess.
