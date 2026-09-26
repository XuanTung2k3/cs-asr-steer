# P2-R run 2 (job 54871): implementation invalidity (1/80 rows failed), not analyzed

**What happened:**

- The job used commit `7f6837a` and manifest `65b3b6c2…`, with output in
  `results/inference_cf/p2r/run2/`, which is preserved.
- It completed with 79/80 rows `ok`.
- Row 034 failed with `IndexError` in `pulse_pass`.

**Cause:**

- A D2 pulse-companion donor was the replay's final step, the step that emits EOS (t = len(tokens)).
- That step is a legitimate P2 edit site, so it is a valid donor.
- `pulse_pass` indexed `tokens[t]` instead of expecting EOS there.

**Handling:**

- The frozen validity rule requires all 80 rows `ok`, so run 2 is invalid.
- Only row statuses and the traceback were read. No P2-R metric from run 2 was computed or
  inspected.

**Fix (implementation only):** at t = len(tokens), the expected baseline token is EOS, for both the
identity check and the summary. A test covers a pulse on the final step.

**Rerun:** into the versioned directory `results/inference_cf/p2r/run3/`, with no other change.
