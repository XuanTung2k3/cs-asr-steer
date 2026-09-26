# P2 independent audit

**Verdict: `P2_AUDIT: PASS`.** The selection `L16_a2_ER_id_pos` is correctly derived under the
frozen rules. The mechanism caveat in `P2_REPORT.md` stands, and P3 needs a human decision.

## Method

The audit script is `experiments/inference_cf_p2_audit.py`. It does not import the P2 evaluator.
It reads the saved rows and does the following:

1. **Provenance.**
   - Recomputes each manifest hash.
   - Checks every manifest source against `git show <manifest commit>:<path>`.
   - Checks that every row is `ok`, carries the matching manifest hash and identity, and has
     `runtime.status = completed`.
2. **Metrics.**
   - MER totals come from its own edit-distance DP over normalized units.
   - PIER, ZH-CER and EN-WER are computed per utterance with the project metric definitions
     and aggregated independently.
   - Validity V1–V5 is re-derived directly from `configs/inference_cf/p2_compact_development.json`.
3. **Divergence attribution.**
   - Takes the first differing token versus `B0M_Lℓ`.
   - Requires an applied edit at a step t ≤ k.
   - Additionally checks B-branch consistency: at k, the method's unsteered argmax equals B0M's
     token.
4. **Selection.** Re-derives the selection with its own sort under the frozen tie rule and
   compares it with the evaluator's.

## Results

| Summary | Runs | Provenance | Metrics agree | Unattributed divergences | Selection |
|---|---|---|---|---|---|
| `p2_A_r1_summary.json` | 54821, 54822 | ok (sources match `04d2383`) | 6/6 configs | 0 | L16 α2 = evaluator |
| `p2_final_summary.json` | + 54825 | ok (`816b385` for C) | 9/9 | 0 | L16 α2 = evaluator |
| `p2_B_summary.json` | 54824 | ok (`816b385`) | 6/6 | 0 | n/a (not selectable) |

**Additional checks.**

- **B-branch consistency at divergence:** 100% in every configuration (584 divergences in
  total). At every first divergence, the unsteered branch still agrees with B0M.
  - **Disclosed audit-instrument fix:** the first audit pass counted 11 cases as inconsistent.
    In all of them B0M had stopped: saved tokens exclude the final EOS, and the unsteered
    argmax at k was EOS (50257).
  - The script now compares against EOS when k equals the B0M length, and the audits were
    rerun.
  - This check is an audit diagnostic, not a spec validity rule.
- **Matched-baseline agreement:**
  - `B0M_L16` and `B0M_L24` are token-identical to each other.
  - Both are token-identical across the A-r1, B and C jobs.
- **Standalone `cached_greedy` B0** differs from B0M on 59/300 utterances. This confirms the
  cause of the attempt-1 invalidity, which is fixed by v1.1.
- **α_c:** recomputed from the selected run's edits, it is 0.3110. The α_c run realized energy
  1176 vs 1203, within 2%.
- **Process:**
  - No grid extension.
  - No rerun because of an unfavorable result; the only rerun, P2-A r1, was for the recorded
    baseline-execution invalidity.
  - The HEAD commits made during the jobs touched only STATUS/CODE_MAP/DATA_EXPOSURE, which are
    not manifest sources, and were made after each job's commit check.

## Scientific caveats (not audit failures)

- **Tiny effect.** ΔPIER is +2/2268, with a lower CI bound of exactly 0.
- **Controls.** The P2-B controls do not support detector selectivity or the +d direction.
- **Stronger comparator.** B0_AUTO beats every steered configuration on PIER by about 8 points.
- **Exposure.** D-dev-select is heavily exposed, so this is development evidence only.
