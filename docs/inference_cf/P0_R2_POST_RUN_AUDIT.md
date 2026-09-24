# P0-R2 post-run independent audit

**POST_R2_AUDIT: PASS**

**Question audited:** do the actual R2 results (job 54758, manifest `e8869f0c…`, commit
`fe960e4`) support the reported frozen conclusion `R2_CF_FEASIBLE_NOT_PREFERRED` and the
selected gate `g_old = E·R_B`?

**Answer:** yes. Headline metrics were recomputed by code written independently of the
evaluator: its own position-label join from saved units, its own Mann–Whitney AUROC with
average-rank ties, and its own paired dialogue bootstrap on a different RNG stream.

## Checks

| Item | Result |
|---|---|
| All 300 IDs accounted for | 300 shards, unique, in panel order, all `status=ok`, all bound to the frozen manifest hash. Empty stderr. |
| Frozen thresholds unchanged | Config file hash equals manifest `config_hash`. The evaluator verified all 23 source hashes before computing. Replicates 2000, seed 240924, AUROC floor 0.70, minima 30 positions / 10 dialogues, all as frozen. |
| Headline reproduction | Independent AUROCs match the evaluator to 4 decimals: g_cf 0.7102 / 0.7261, g_old 0.8200 / 0.8359, E 0.8200. Counts match: 316 confusion, 8,901 ZH-correct, 1,168 EN-correct; 18/20 dialogues. |
| ΔAUROC | Independent paired bootstrap: [−0.145, −0.069] vs ZH and [−0.146, −0.069] vs EN-correct, against evaluator [−0.145, −0.065] and [−0.144, −0.067]. Both exclude 0 on the negative side, so "not preferred" is conservative. |
| NOT_ESTIMABLE handling | Every required contrast meets 30 positions / 10 dialogues. All 2,000 bootstrap replicates valid. |
| Fallback retention | 1,115 `mid_character` rows remain in the panel. Mapped fallbacks stay at g=0 in the gate AUROCs (14% of EN-confusion and 11% of ZH-correct positions), which lowers scores. No abstention was dropped. |
| Unalignable accounting | Every unmapped unit carries an explicit reason. Collision exclusions are itemized per stratum (ZH 75, EN-correct 26, EN-confusion 19, deletion 285, same-language 48). |
| No post-hoc changes | Localizer (max-window) kept, even though it is not better than point-argmax on the target strata. No stratum or mapping redefinition. No new gate threshold. |
| Leakage | The runner reads only the reference-free inference panel. Evaluator-only data (references, CTC, B0_AUTO) is read after inference. |
| Invariants | 0 violations. Same-prefix validation on every row. Replay argmax agreement 99.8%. |
| Component consistency | Component AUROCs agree with stratum medians. At confusion positions D≈0 (R_Ecf≈R_B), which explains g_cf < g_old. Shuffled-E collapses g toward chance (0.50/0.55), confirming E carries confusion-vs-Mandarin. |
| Prompt susceptibility | Exposed and resolved the opposite way from the pre-run worry. The English counterfactual is **prompt-insensitive** at both Mandarin and confusion positions (median D 0.0001 and 0.002). D adds noise, not repair-specific signal. |
| Russian diagnostic | Reported honestly as an unexpected diagnostic (D_X, g_X rank better than D, g_cf). **Not adopted**: it was predeclared diagnostic-only, and adopting it would be outcome-driven selection. |
| g_cf vs g_old | Correctly computed on identical rows and paired by dialogue. The frozen rule (spec §4) gives `R2_CF_FEASIBLE_NOT_PREFERRED` and forbids selecting g_cf by name, so the P1 gate is `g_old`. `g_old` independently meets the same AUROC/interval bars on both contrasts. |
| B0_AUTO visibility | Reported on the matched 300: PIER 0.390 vs B0 0.474, EN-WER 0.395 vs 0.477. |
| Script-specific boundary | Report claims only Mandarin–English, distinct-script findings. |

## Limitations that constrain P1 claims (not blockers)

1. **Weak target-stratum localization.** The LOCALIZER pass is pooled and dominated by Mandarin.
   On EN-confusion only 46% of windows are centred within 0.5 s (median error 0.58 s), yet E
   still ranks confusion above correct Mandarin at 0.88.
2. **Deletions are unreachable** by this gate (median E 0 at deletion slots).
3. **The EN-confusion vs EN-correct contrast is carried by R_B** (emitted script), as predicted.
   It shows the gate stays quiet on correct English, not general error detection.
4. **g_old ranks; it does not predict repair.** It is a *repair-need ranking score*. Whether
   intervening where it is high helps recognition is untested until P1/P2.
5. **B0_AUTO is the stronger unsteered baseline.**

## Verdict

The R2 outcome is valid, complete and reproducible. The frozen conclusion
`R2_CF_FEASIBLE_NOT_PREFERRED` with **selected gate `g_old = E·R_B`** is supported. Progression
to P1 causal-acceptance is permitted under the frozen contract: P1 consumes `g_old` unchanged.

**POST_R2_AUDIT: PASS**
