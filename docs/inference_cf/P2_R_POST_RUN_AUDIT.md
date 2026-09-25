# P2-R post-run audit

**Verdict: `P2_R_AUDIT: PASS`.**

- **Audit script:** `experiments/inference_cf_p2r_audit.py`. It does not import the analysis
  module.
- **Output:** `results/inference_cf/p2r/run3_audit.json`.

| Check | Result |
|---|---|
| Manifest, config and population hashes | ok (`33a99f29…`; population `6a880d2b…`) |
| Sources equal the manifest commit (`d3c1014`) | ok, all 14 |
| Completeness | 80/80 rows ok; identity and manifest hash on every row; runtime `completed` (job 54874) |
| Expected positions | 180/180 D1 positions present; 209/209 D2 targets present |
| Energy matching | 0/180 D1 positions outside the 2% target and pairwise tolerance. D2 total energy ratio 1.0000020 |
| +d/−d pairing | φ(+d) + φ(−d) = π at every position: identical state and position, opposite sign |
| Random direction | φ = π/2 (orthogonal to the state) at every position; seed and construction frozen |
| Relocation | Every planned move executed: donor unedited in the oracle, destination edited, destination unedited in the current placement |
| Independent recomputation | All 11 primary estimates and Bonferroni dialogue-bootstrap intervals, recomputed from raw rows with separate code, agree with the analysis to 1e-9 |
| Oracle leakage | The deployable path (`inference_cf_cached.py`, `inference_cf_p2.py`, core modules) is byte-identical to the P2 freeze. An AST test forbids plan, target or reference identifiers there. References are read only by the CPU analysis and audit. |
| P3 exposure | None. All 80 utterances are `D-dev-select`. No confirm, test, P3 or transfer file was read. |
| Post-hoc changes | See below. |

**Post-hoc changes:**

- **The only post-run source change is in `inference_cf_p2r_analyze.transcript_eval`.** It was a
  disclosed crash fix: a Han companion target is not a POI, so its correctness is now read from
  `unit_status`.
  - Decision inputs, the primary family, thresholds and `decide` are unchanged. The audit
    recomputes the primary family independently.
- **Earlier attempts are preserved**, and no metric was computed from either:
  - run1: abort;
  - run2: 1 failed row.
- **Debug smokes stayed outside `results/`**, and their rows were deleted unread:
  - job 54861 localized the abort;
  - job 54870 did a 4-utterance technical check.
