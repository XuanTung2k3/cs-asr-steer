# Preserved, unapplied patches

## `post_r2_runner_row_level_fallbacks.patch`

- **Found:** 2026-09-25 (file mtime 01:11 UTC), as an uncommitted working-tree change to
  `experiments/inference_cf_p0_r2.py`. It was made after the R2 run by an external analysis
  session.
- **Content:**
  - Turns a whole-utterance B/Ecf replay failure into per-row `baseline_provider_fail` /
    `ecf_provider_fail` fallbacks.
  - Makes missing alignment-head attention return `None`, which later records as
    `localizer_fail`.
- **Classification:** *C — useful future robustness change.* It is not accidental residue and has
  no scientific effect on completed runs: R2 job 54758 had 0 replay failures.
- **Why it is not applied:**
  - The frozen R2 manifest (`sha256:e8869f0c…`, commit `fe960e4`) and both P1 manifests hash the
    exact bytes of this runner. Applying the change would make those historical manifests
    unverifiable at HEAD.
  - The R2 results were produced by the committed version, never by this modified runner.
- **Applying it later:** do so only in a new, separately versioned runner/manifest.

## Provenance note (verified 2026-09-25)

After restoring the committed runner, every frozen manifest's source hashes verify at HEAD except
the canonical plan. The plan is a living document updated after each freeze. Its hash in each
manifest matches `git show <manifest git_commit>:INFERENCE_STEERING_IMPLEMENTATION_PLAN.md` for
R2 (`fe960e4`), P1 attempt 1 (`c272606`) and P1-r1 (`d704db9`). The P1 attempt-1 manifest also
differs on the files superseded by spec v1.1, as documented in
`P1_ATTEMPT1_INSTRUMENT_DEFECT.md`.
