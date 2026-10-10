# P2-RJ-E post-run audit

**Verdict: `P2_RJ_E_AUDIT: PASS`.**

- **Audit script:** `experiments/inference_cf_p2rje_audit.py`. It imports neither analysis module
  nor any runner.
- **Output:** `results/inference_cf/p2rje/run1_audit.json`.

| Check | Result |
|---|---|
| Manifest, config and positions hashes | ok (`sha256:a6fc9cce…`, positions `sha256:47987132…`) |
| Sources equal the manifest commit (`646179b`) | ok, all 19; no source changed after the freeze |
| **Population size** | 227 = the frozen P2-R ledger `EN-confusion\|valid`; no duplicates |
| **Dialogue membership** | Every position is an R2 EN-confusion evaluation unit, with the recorded dialogue (17 dialogues) |
| Original positions | All 60 P2-RJ EN-confusion positions are included exactly once |
| **Overlap reproduction** | All 60 reproduce P2-RJ run1 raw rows and vectors: gap, ‖g_tan‖, gradient direction; maximum deviation **2.2e-16** |
| Completeness | 43/43 utterances × {bf16, fp32} ok with vector-file hashes; **227/227 positions in both passes**; runtime `completed` (job 55002) |
| Independent recomputation | Margins/gaps, ‖g‖, ‖g_tan‖, A, ρ, the precision cosine (0.99981), median A (4.896), median ρ (0.519), Q50 with its Bonferroni dialogue-bootstrap interval for bf16 (0.5387 [0.4061, 0.6653]) and fp32 (0.5191 [0.3854, 0.6472]), the leverage classes (UNRESOLVED / UNRESOLVED) and the diagnosis all agree with the analysis to 1e-9. **Independent diagnosis: `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`.** |
| **Gradient applied as steering** | **NO.** 454/454 probed forwards are value-identical to the unedited step (bitwise logits). The P2-RJ-E and P2-RJ runners contain no edit function fed by a gradient, and no steering hook or `cached_decode` (AST). |
| Reference leakage | None into inference. The deployable path is byte-identical to the P2-R / P2-RJ freezes. |
| **P3 exposure** | **None.** The utterances are a subset of the 300-utterance D-dev-select panel. No D-dev-confirm, D-test, `router-calib`, P3 or transfer identifier appears in the P2-RJ-E sources. |
| **Post-hoc changes** | One disclosed serialization-only wrapper (`inference_cf_p2rje_write_analysis.py`), added after the run because ρ = +∞ at one gap ≤ 0 position is not strict-JSON. It calls the frozen `analyze` unchanged. There were no threshold, metric, population or decision changes, and no reruns. |
