# P2-1b amendment r2 — repair of the evaluation output serialization

**Status.** This is an engineering repair, made after training and before any evaluation output existed. It is recorded
in `p2_1b/freeze_r2.json`, which chains to `freeze_r1.json` and through it to `freeze.json`.

**Unchanged:** the scientific design, rules, metrics, training runs, checkpoints, selection and prefix audit.

## 1. Failure

The execution chain (amendment r1) ran as follows:
- the five training runs completed (jobs 59265–59269);
- selection completed (59270);
- the prefix audit completed (59273).

The evaluation job **59271** then failed at its final step. `write_json` serializes with `sort_keys=True`, and
`paired_kl_minus_ce` mixes the integer seed keys 42, 43 and 44 with the string key `"seed_mean"`:

```
TypeError: '<' not supported between instances of 'str' and 'int'
```

- No `evaluation.json` was written.
- The dependent jobs, verify (59272) and report (59274), were cancelled by `--kill-on-invalid-dep`. They never ran.
- No result was inspected before this repair.

## 2. Repair

- **`reft/p2_1b_evaluate.py`:** the paired table is written with string keys (`json_keys`). This is the same form JSON
  gives it on reading, and the report and figures do not index this table by seed. No computation changed.
- **`reft/p2_1b_train.py`:** `check_freeze` verifies the whole chain `freeze.json → freeze_r1.json → freeze_r2.json` and
  the sources of its head. `freeze-r1` and `freeze-r2` share one write-once implementation, `run_freeze_amend`. The
  existing `freeze_r1.json` verifies unchanged.
- **`slurm/reft_p2_1b_cpu.sbatch`:** adds the `freeze-r2` step.
- **`tests/reft/test_p2_1b.py`:** new regression test `test_paired_table_serializes`.

**Already-completed jobs keep their r1 provenance.** Training, selection and the prefix audit ran under the r1 head.
`reft/p2_1b_prefix_audit.py` is unchanged by r2.

## 3. Rerun

The CPU tests, then `freeze-r2`, then evaluate, then verify (GPU), then report, as `afterok` jobs with
`--kill-on-invalid-dep`. There are no new training runs, so the run count stays at 5 of 6.
