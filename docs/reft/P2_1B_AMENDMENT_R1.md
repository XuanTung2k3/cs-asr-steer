# P2-1b amendment r1 — repair of the G3 initial-gradient smoke check

**Status.** This is an engineering repair of one preflight check. Before any P2-1b training run, the first preflight (job
59254, `p2_1b/preflight_59254.json`, kept unchanged) returned `P2_1B_BLOCKED_PREFLIGHT_G3_kl_objective`. The amendment is
recorded in `p2_1b/freeze_r1.json`, which chains to the unchanged `p2_1b/freeze.json` through its SHA-256.

**Unchanged:** the scientific design, arms, loss, λ, mask, data, seeds, selection rule, decision rules, run limit and
evaluation.

## 1. Failure

G3 requires that "the CE value and gradient are identical to the CE path at initialization". The smoke compared:
- the gradient of the KL-arm loss (`kl_step`, λ = 1) on adapter `ad`, with
- the gradient of `train_supervised.batch_loss` on a second adapter `ad2`,

both built by `make_adapter("R2", …, init seed 42)`.

**Results:**
- The CE values were identical.
- The gradients of W and b were identical.
- The gradient of the rotation's stored parameter `rot.parametrizations.weight.original` differed: max |Δ| = 0.0335,
  above the tolerance of rtol 1e-4 and atol 1e-7.

## 2. Diagnosis

The diagnostics were bounded engineering jobs on a GPU compute node. They ran on 4 training utterances, did no training
and wrote nothing under `p2_1b/`.

| Check | Result |
|---|---|
| CE path vs CE path, two fresh adapters in one process | `original` gradient differs (max 0.033–0.041); loss, W and b gradients bitwise equal |
| Same, KL path at λ = 0 vs CE path | same pattern (0.037) |
| Gradient of the KL term alone at initialization (λ = 1 minus λ = 0) | 4.3e-8 (R), 7.3e-9 (W), 0 (b); CE gradient norms are 0.337, 0.169, 0.011 |
| R = `rot.weight`, dL/dR and dL/d(W − R) across fresh adapters | bitwise equal |
| `original` tensor across fresh adapters | bitwise equal (−I pattern: diagonal −1, reflectors 0) |
| Norm of the `original` gradient across adapters | equal (0.33713) |
| Differing entries | 2,556 = all reflector coordinates of the 1,278 complement directions; the in-span entry is equal |

**Cause.** `torch.nn.utils.parametrizations.orthogonal` defaults to Householder with trivialization for a 2 × 1280
weight.
- When R is assigned at initialization, its `right_inverse` sets `base` to R completed to a square orthogonal matrix.
  The completion columns come from `torch.randn`, which draws from the **global** RNG. It then sets `original` to −I,
  so that `R = (base @ householder(original))[:, :2]`.
- R itself is fixed by the adapter's own init-seed generator. The coordinates in which its gradient is expressed, the
  complement columns of `base`, depend on the global RNG state at construction.
- `ad2` was built after `ad` had consumed global RNG. Its `original` gradient is therefore the same Riemannian gradient
  expressed in a different orthonormal basis.

**Conclusion.**
- The KL objective's gradient is mathematically zero at initialization, as observed.
- The CE gradient is identical where it is coordinate-free.
- The check failed because the smoke test constructed its two adapters differently. It does not indicate a defect in
  the objective.

## 3. Why the training runs are unaffected

- `run_train` (unchanged) calls `torch.manual_seed(seed)` before `load_backbone()` and `make_adapter`. No arm-dependent
  code runs between those calls, so CE_s and KL_s receive bitwise-identical `base`, `original`, W and b.
- This is also the path P2-1 used. G2 confirmed it: one full new CE epoch at seed 42 reproduced
  `p2_1/R2/epoch1/adapter.pt` bitwise.
- `base` is a registered buffer, so it is saved in `adapter.pt` and restored on reload.

**Note for interpretation.** Adam is not rotation-invariant. The R trajectory therefore depends on the
global-seed-determined `base` as well as on the init seed. Both are tied to the run seed, so the CE/KL pairing within a
seed is preserved.

## 4. Repair

**`reft/p2_1b_train.py`, `run_smoke` only.**
- `torch.manual_seed(42)` runs before each of the two adapters compared in the initial-gradient check.
- New gating item `init_state_identical_incl_base`: the two state dicts, including `base`, must be bitwise equal.
- Informational fields record that `base` is part of the state dict, both for the new adapters and for P2-1's
  `epoch1/adapter.pt`.
- The tolerance of `init_grads_equal_ce_path` is unchanged.

**`check_freeze` and `freeze-r1`.**
- `check_freeze` verifies the chain head: `freeze_r1.json` if it is present, and its `prev_freeze_sha256` must match the
  unchanged `freeze.json`.
- New write-once command `freeze-r1`, added to `slurm/reft_p2_1b_cpu.sbatch`. It re-hashes the frozen sources plus this
  document, and requires the frozen inputs to be unchanged.

**`tests/reft/test_p2_1b.py`.** New regression test `test_orthogonal_base_follows_global_rng`:
- R is equal across global seeds and `base` differs;
- the same global seed gives bitwise-identical state and gradient;
- across bases the gradient differs but has equal norm.

**Unchanged:** `run_train`, `kl_step`, `kl_masked`, `teacher_mask`, the evaluation, the prefix audit, the configuration
and the decision rules.

## 5. Rerun

The CPU tests, then `freeze-r1`, then one new GPU preflight smoke, written as `p2_1b/preflight_<job>.json`. The submit
script reads the newest preflight. The P2-1b run limit (≤ 6 new training runs) is unaffected, because no training ran.
