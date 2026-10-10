# ST-PROMPT-R1 — final report (terminal)

| Stage | Verdict | Audit |
|---|---|---|
| R1-A single-pulse layer × dose × sign screen | **`ST_PROMPT_R1_A_MARGIN_ONLY`** | `PASS_TO_ST_PROMPT_R1_A`; `ST_PROMPT_R1_A_AUDIT: PASS (PRIMARY)` before references; `ST_PROMPT_R1_A_AUDIT: PASS (FULL)` |
| R1-B conditional oracle-local sequence evaluation | **NOT RUN** — not authorized. The frozen precondition is `ST_PROMPT_R1_A_CAUSAL_PROMISE`; A gave MARGIN_ONLY and an empty selection | — |

Details are in `ST_PROMPT_R1_A_REPORT.md`. No R1-B report exists, because nothing ran.

## Scientific conclusion

1. **Material causal lexical repair from v_prompt? No.**
   - Two of 24 arms raise the fixed-competitor EN-confusion margin with adjusted evidence against both NONE and matched-energy
     random: L16 η 0.45 − (+0.63 nat) and L24 η 0.30 − (+0.75 nat).
   - No arm produces more than one top-1 reference correction. Across all 24 configurations the union is 2 of 60 positions,
     in 2 dialogues.
2. **Does strength explain the previous weak results? No.**
   - The margin rises monotonically with η, up to +1.48 nat at L24 η 0.45 −.
   - Corrections do not rise with it, and correct-state damage and KL grow with dose.
3. **Layer-dependent? Yes.** Effects are strongest at L24. Matched random effects also grow at L24 (+1.01 nat at η 0.45), so
   large-dose L24 shifts are largely non-specific.
4. **Power only with collateral damage?** No power appears even with damage. The largest-margin arm makes 1 correction against
   8 correct-state corruptions.
5. **Transfer of local correction into full decoding:** not tested (R1-B not authorized).

**Established:**
- Exact reproduction of the historical L16 apparatus (states, D0, ST-LOC0 pulses, D2 bitwise).
- Fully reachable, energy-valid relative doses at L3/8/16/24 (6,480 cells).
- A sign-consistent (minus), dose-monotone margin lever that beats random only at two configurations.

**Not established:**
- Lexical correction power.
- Any sequence or ASR benefit.
- Any deployable location policy.

## Final decision

- **Continue to test-time location optimization (R2): NO.**
- **Frozen reason:** R2 requires a positive R1-B (`ST_PROMPT_R1_B_ORACLE_SEQUENCE_FEASIBLE`). R1-B requires
  `ST_PROMPT_R1_A_CAUSAL_PROMISE`, and R1-A is `ST_PROMPT_R1_A_MARGIN_ONLY` with no full benefit-gate pass. Isolated
  corrections without full evidence cannot advance.
- **Next permitted experiment:** none is authorized by this study. Any new direction, dose or layer would need a separate,
  human-authorized pre-outcome freeze.

## Disclosures

- **Amendment A1** (user-authorized after job 58158 and before any reference access): the runner's non-frozen 0.5%
  norm-preservation check flagged 189 of 6,480 cells, at a maximum of 0.68%, which is the bf16 precision of the frozen
  NormPreserve output. It became a recorded diagnostic. Validity uses the frozen criteria, all of which passed on every cell.
  Because of this check, Slurm records the job state as FAILED; all phases completed.
- Exposed development data with selection bias; single-step teacher-forced prefixes; the oracle union is reference-dependent.
- Compute: one job (58158), 6 min 20 s wall time, peak VRAM 3.97 GB allocated / 4.37 GB reserved, 20,736 decoder forwards,
  7,740 pulse calls.
