# Pre-P2 cached-equivalence attempt 1 (job 54801): `CACHED_EQUIVALENCE: BLOCK` by an instrument defect

**Run.** Slurm 54801, `COMPLETED` 0:0, 100 s wall, commit `c7e2369`. Output
`results/inference_cf/pre_p2_ce/` is preserved unchanged.

**v1 result.**

- **Passing:** CE1–CE5, CE7 and CE8.
  - Zero-dose logits are bitwise identical.
  - α=0 cached tokens equal cached greedy on 20/20 decodes and equal R2 `generate()` on 10/10
    utterances (full replay managed 9/10).
  - Fallback 100%, window 97.3%, g within 0.02 on 99.5% of steps; direction status 100%, minimum
    cosine 0.9925.
- **Failing:** CE6, on a single clause.
  - **Edits:** 40 cached edits; `edit_applied` identical on 100% of steps.
  - **Specification checks:** sign, own-replica and NormPreserve checks all pass.
  - **Tokens:** zero non-near-tie argmax disagreements; the only disagreement is the known
    `U0029` t=6 near-tie with margins 0.0625 / 0.0.
  - **The failing clause:** "cached vs replay *realized* edit norm within 5% on ≥95% of common
    edits" gave **36/38 = 94.7%**.

**The two exceedances.**

| Edit | g | Cached | Replay | Difference |
|---|---:|---:|---:|---:|
| U0011, L24, t=25 | 0.0043 | 0.0156 | 0.0168 | 7.5% |
| U1063, L24, t=35 | 0.1046 | 0.1180 | 0.1065 | 10.9% |

At L16 the same U1063 position agrees within 0.1%. Each path's realized edit equals its own
same-precision replica of the specified edit exactly.

**Diagnosis: instrument defect, not a cache-semantic difference.**

- **Where the noise comes from:** both paths call the identical `models.hooks.apply_steering`.
  Its NormPreserve scalars (`‖h‖`, `‖h̃‖`) are computed in bf16 with about 0.4% relative error
  (P1 had already observed a 0.38% maximum norm error). That injects a radial error of up to about
  `0.004·‖h‖ ≈ 0.036` into any realized edit.
- **What the clause actually compared:** two independently bf16-rounded outputs.
- **CPU simulation of the same function:** identical math, with inputs differing only by
  bf16-level site noise, gives cross-path realized-norm differences above 5% in **39%** of draws
  at g=0.0043 and **4%** at g=0.1046. It gives 0% at g ≥ 0.35.
- **Conclusion:** two small-edit exceedances out of 38 is the expected chance behaviour of the
  shared bf16 arithmetic.
- **Cross-path equivalence is otherwise established:** every input of the edit agrees (g, d, α,
  site, position), and each realized edit equals its specification.

**Handling (same pattern as P1 v1.1).**

- **Record:** the v1 verdict stays `BLOCK` and is preserved.
- **CE spec v1.1 corrects the CE6 measurand only.**
  - The cross-path comparison now uses each path's **specified edit**: the float64 reference edit
    from that path's own logged `h_B`, `g`, `d`.
  - Both paths must match their own same-precision replica.
- **Unchanged:** the 5% tolerance, the ≥95% share and every other criterion.
- **Descriptive only:** realized cross-path norms.
- **Rerun:** `results/inference_cf/pre_p2_ce_r1/`.
- **Post-outcome disclosure:** this correction was made after observing the v1 outcome. It is
  disclosed here and must be explicitly accepted or rejected by the Pre-P2 audit.
