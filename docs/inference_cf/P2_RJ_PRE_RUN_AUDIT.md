# P2-RJ pre-run audit

**Verdict: `PASS_TO_P2_RJ_RUN`.**

- **Audited:** 2026-09-26, before any GPU execution and before any P2-RJ outcome exists.
- **Spec:** `P2_RJ_JACOBIAN_DIAGNOSIS_SPEC.md` v1, committed at `9b921a9`, plus amendment v1.1
  (§12, pre-outcome).

| Check | Result |
|---|---|
| **Population unchanged** | The positions file lists the P2-R `population.json` D1 positions (`sha256:6a880d2b…`) in the same order, with the same targets: 180 positions, all (utterance, t) unique. A test rebuilds it from P2-R run3 (manifest `sha256:33a99f29…`) and gets it byte-for-byte. The utterances are the 80 P2-R `D-dev-select` utterances. |
| **Exact state** | B/E cached branches with P2-R semantics: the same `Branch.step` code, run under `no_grad`. On a tiny model, a test reproduces P2-R `pulse_pass` **bitwise** for cos(d, r), ‖hE − hB‖, the audit pre-norm, log p(ref), the argmax, and the solver scale s for all three arms. The grad-step logits and site are bitwise equal to the unedited step. |
| **Reference definition frozen** | Y_ref is the P2-R `target_ids`. c\* is the first non-Y_ref token in P2-R's saved unedited top-20 list, frozen per position in the positions file before any gradient. For EN-confusion c\* is the baseline token (60/60), and m(r₀) equals the P2-R margin. |
| **Gradient correctness** | Tested against a central finite difference through an additive site edit (test-only, tiny model); agreement is within 2%. The tangent projection is exact (⟂ r, idempotent). A = ‖g_tan‖·e\*. |
| **Gradient never applied** | `SiteGradientProbe` takes no direction argument. It asserts its δ is zero at install and at forward, and refuses a nonzero δ (tested). An AST test checks that no edit function (`apply_steering`, `hook_edit`, `solve_scale`, `chord_edit`, `scaled_direction`) receives a gradient-derived argument, and that the runner constructs no steering hook, `cached_decode` or `pulse_hook`. The arm edits δ_a are state-vector geometry only; the model never sees them. |
| **Deployable path untouched** | Diff against the P2-R run3 commit (`d3c1014`) for `inference_cf_cached.py`, `inference_cf_p2.py`, `inference_cf_p2r.py`, `src/csasr/inference_cf/*`, `sites.py` and `hooks.py`: **empty**. A test also compares their hashes with the P2-R manifest. |
| **No new layer / α / gate / direction / localizer** | Layer 16 comes from config and equals P2-R. e\* equals the P2-R `target_edit_norm` exactly (tested). The directions are the P2-R ±d and the frozen random vector (same function, same seed; tested). There is no gate, α, dose map or localizer in the runner. |
| **Decision thresholds frozen** | Spec §7 and config `analysis`. `decide`, `leverage_class`, `alignment_class` and `label` are unit-tested against every profile. The Bonferroni family is {Q50, K₊}. |
| **Data roles / P3** | The only data inputs are the reference-free inference panel (audio paths), the `B0M_L16` baseline tokens, the P2-R population and run3 rows, and the frozen positions. There is no `router-calib`, D-dev-confirm, D-test, P3 or transfer path in any P2-RJ file (grep). **P3 untouched.** |
| **Pipeline dry-run** | A synthetic random-vector run over the 180 real keys (scratchpad, not an outcome) went through `analyze` and the independent `audit`. They agree on every recomputed statistic, class and diagnosis. The dry-run is also what surfaced the bf16 argmax-tie artifact fixed in amendment v1.1. |
| **Tests** | `tests/test_inference_cf_p2rj.py`, plus the P2-R, P2, cached and earlier inference_cf regressions: all pass. |
| **Compute** | One Slurm job on MIG 3g.40gb with a 1-hour limit; bf16 pass, then fp32 pass. About 360 backward passes in total, with a resumable ledger. |

**Residual risks** (they change nothing in the protocol):

- **bf16 gradient precision.** It is guarded by V7, a float32 replicate: the label must not change.
- **First-order approximation.** Checked by V6 against saved P2-R arm outcomes, which are near
  bf16 resolution; slope and r are descriptive.
- **Decode-rule tie at c\*.** At a tie, the gradient is taken with respect to the frozen P2-R-order
  token; this is disclosed in spec §12.

No conceptual blocker remains.
