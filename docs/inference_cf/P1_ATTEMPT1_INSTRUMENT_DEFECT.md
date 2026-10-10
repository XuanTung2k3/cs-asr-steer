# P1 attempt 1 (job 54770) — `P1_BLOCKED` by an acceptance-instrument defect

**Run.** Slurm 54770, `COMPLETED` 0:0, 51 s wall (35.9 s program), H100 MIG 3g.40gb, peak
3.42 GB. Manifest `sha256:4e9d12fa…`, commit `c272606`, output `results/inference_cf/p1/`
(preserved unchanged).

**Frozen-v1 verdict: `P1_BLOCKED`.** A1–A5 and A7–A13 PASS; A6 FAIL.

**Passing evidence:**

- **Identity:** bitwise F3 = F1 on all 190 α = 0 steps.
- **Edits:** 20 real edits, all at layer 24 at the query position (≥ 4), from 23 `g > 0` steps;
  3 were blocked at the forced-prefix position.
- **NormPreserve:** maximum relative error 0.38%.
- **Directions:** valid at every step (norm 1.37–8.15), varying (minimum consecutive cosine 0.095).
- **Gate:** recomputed exactly.
- **Isolation and prefix integrity** hold.

**Why A6 failed.** A6 has two clauses:

1. **Sign:** `cos(h' − hB, d) > 0`. This **passed** for every edit (minimum 0.872).
2. **Norm match:** the hook-recorded realized edit norm must equal a **float64** reference edit
   norm within 5% relative. This failed on 2 of 20 edits: g = 0.0042 (reference 0.0040 vs hook
   0.0168) and g = 0.059 (0.0515 vs 0.0636).

The other 18 edits agree within 0.06–2.5%.

**Diagnosis: instrument defect, not intervention failure.**

- **Where the edit happens:** the model runs in bf16, and `apply_steering` adds, renormalizes and
  rounds all 1,280 site elements in bf16.
- **Resolution floor:** the realized edit (`steered − site`) therefore carries rounding noise of
  order `‖h‖·2⁻⁸ ≈ 0.035` at this site (‖h‖ ≈ 9). The two failing edits are at or below that
  floor. Every larger edit matches.
- **Mis-specified check:** the check compared a bf16 computation against a float64 computation
  with a purely relative tolerance. That is mathematically unattainable for sub-resolution edits
  regardless of correctness.
- **Nothing else implicated:** the sign, the site, the gain and the NormPreserve all check out.

**Handling.**

- **Record:** this attempt stays recorded as `P1_BLOCKED` under v1. Its artifacts are preserved
  and not overwritten.
- **Fix:** P1 spec v1.1 corrects the instrument only. The A6 reference is the **same frozen edit
  computed at the site's actual precision**: `models.hooks.apply_steering` on the recorded bf16
  site value, with the bf16-cast gate and direction, on the same device.
- **Unchanged:** the 5% tolerance and the sign clause. The float64 reference is retained and
  reported descriptively.
- **Everything else frozen:** gate, direction, site, α, population and all other criteria.
- **Rerun:** a new manifest and output directory (`results/inference_cf/p1_r1/`). This rerun is
  justified solely by instrument invalidity, never by an unfavorable outcome.
