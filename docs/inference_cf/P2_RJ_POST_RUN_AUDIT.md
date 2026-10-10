# P2-RJ post-run audit

**Verdict: `P2_RJ_AUDIT: PASS`.**

- **Audit script:** `experiments/inference_cf_p2rj_audit.py`. It imports neither the P2-RJ
  analysis module nor the runner.
- **Output:** `results/inference_cf/p2rj/run1_audit.json`.

| Check | Result |
|---|---|
| Manifest, config and positions hashes | ok (`sha256:c020cf62…`, positions `sha256:c1e016c0…`) |
| Sources equal the manifest commit (`a93aa37`) | ok, all 15; **no source changed after the freeze** |
| Completeness | 80/80 utterances × {bf16, fp32} rows ok, with row identity, manifest hash and vector-file hash; runtime `completed` (job 54998). **180/180 positions in each pass.** |
| Population | Identical to the P2-R `D-dev-select` population (80 utterances, 180 D1 positions). The P2-R run3 manifest hash matches. |
| State identity | 180/180 positions reproduce P2-R run3 state: argmax (tie-aware, v1.1), ‖r‖, cos(d, r), and the solver s for all arms (bitwise). The baseline margin is re-derived from raw P2-R rows (max \|Δ\| ≤ 1e-3). The competitor is re-derived from the raw P2-R top-20. |
| Independent recomputation | From the raw vectors and raw P2-R rows (not the positions file's derived values), the audit recomputes margins, ‖g‖, ‖g_tan‖, A, ρ, cosines, κ, first-order predictions and observed P2-R changes. The Bonferroni dialogue-bootstrap Q50 and K₊ intervals (bf16 and fp32), the leverage/alignment classes, the pooled first-order r (0.9185) and the precision cosine (0.99982) all agree with the analysis to 1e-9. The independent diagnosis is `P2_RJ_STILL_AMBIGUOUS`. |
| **Gradient applied as steering** | **NO.** Every probed forward is value-identical to the unedited step: grad-step logits are bitwise equal on 360/360 probed steps. The runner contains no steering hook, `cached_decode` or `pulse_hook` (AST). No edit function receives a gradient-derived argument (AST). |
| Reference leakage | None into inference. The deployable path (`inference_cf_cached.py`, `inference_cf_p2.py`, `core*`, `sites.py`, `hooks.py`) is byte-identical to the P2-R freeze. References enter only as frozen evaluator target sets in the diagnostic namespace. |
| P3 exposure | **None.** All utterances are `D-dev-select`. No `router-calib`, D-dev-confirm, D-test, P3 or transfer file was read. |
| Post-hoc changes | **None after outcomes.** The only amendment (v1.1, tie-aware identity) was committed at `77bf214` before the manifest and before the job. No threshold, metric, population or decision rule changed, and there were no reruns. |
