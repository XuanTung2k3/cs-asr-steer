# P2-R run 1 (job 54855): infrastructure/implementation failure, no outcome

**What happened:**

- The job used commit `4e9adf2` and manifest `90efa6b2…`, with output in
  `results/inference_cf/p2r/run1/`, which is preserved.
- It passed start-up verification.
- Slurm then recorded `FAILED`, signal 6 (SIGABRT), about 6 s into the first utterance. No row was
  written and the logs were empty.

**Diagnosis:**

- A debug copy (job 54861) ran the same frozen manifest into a directory outside `results/`, with
  `PYTHONFAULTHANDLER=1`.
- It localized the abort to `solve_scale`: float64 vector arithmetic (`torch.dot` and norms) on a
  CUDA tensor inside the hook action, on the MIG device.
- P2 never ran fp64 kernels on the GPU.

**Fix (implementation only; no scientific change):**

- The solver geometry now runs in CPU float64.
- The only device computation left is the exact bf16 `apply_steering` emulation, which is the
  hook's own arithmetic.
- s·v is formed in CPU float64 and rounded to bf16 identically for the emulation and for the hook
  (`scaled_direction`).
- The target energy, tolerances, directions, population and decision rules are unchanged.

**Verification:**

- The P2-R tests pass.
- A working-tree GPU smoke (job 54870, debug manifest outside `results/`) completed 4 utterances.
  Only technical fields were read:
  - solver = hook;
  - realized edit norms 1.1256–1.1269 against e\* = 1.1261;
  - bitwise restores, replay lineage, no baseline mismatch.
- The smoke was then cancelled and its rows deleted unread.

**Rerun:** into the versioned directory `results/inference_cf/p2r/run2/`.
