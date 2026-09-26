# P2-RJ-E pre-run audit

**Verdict: `PASS_TO_P2_RJ_E_RUN`.**

- **Audited:** 2026-09-26, before any P2-RJ-E GPU execution or outcome.
- **Spec:** `P2_RJ_E_LEVERAGE_EXPANSION_SPEC.md` v1, committed at `04395d2`.

| Check | Result |
|---|---|
| **Population frozen** | `results/inference_cf/p2rje/positions.json` (`sha256:47987132…`) holds all **227** eligible EN-confusion positions: 43 utterances, 17 dialogues, sorted, no duplicates. The count equals the frozen P2-R ledger `EN-confusion\|valid = 227`. The 60 P2-RJ positions are included exactly once, with identical targets, baseline tokens, frozen c\* and P2-R reference values, and 167 positions are new. Exclusions are the unchanged P2-R ledger reasons. No outcome-based selection. |
| **Eligibility identical** | Enumeration intercepts `select` in the **unmodified** P2-R population module. The same call re-derives the frozen P2-R population hash (`sha256:6a880d2b…`). A test shows the rebuild is deterministic. |
| **Conditions identical** | Same model, checkpoint and bf16 + fp32 replicate. Same L16 DG-02 site, cached B/E semantics, forced prompts and suppression. The runner calls the **byte-identical** P2-RJ `run_utterance`; the P2-RJ, P2-R, cached, sites and hooks sources hash-equal the P2-RJ run1 manifest (tested). |
| **Only sample count expanded** | Margin, Y_ref, the c\* rule, tangent projection, e\* = 1.1260757575454359, dialogue bootstrap (10,000, seed 240924), α = 0.025 per bound, the Q50 definition, the 0.5 threshold and `leverage_class` are all reused unchanged (tested against the P2-RJ config and code). |
| **c\* for new positions** | The identical rule (P2-R `summarize` top-20 order), applied in a no-grad pre-pass on the identical unedited branch before any gradient. A test shows it equals the P2-RJ derivation. For the original 60, the frozen c\* is used and checked against the rule. |
| **Q50 / threshold unchanged** | Yes. The only new element is the frozen terminal mapping SUBSTANTIAL / WEAK / UNRESOLVED → `P2_RJ_E_DIRECTION_ISSUE` / `P2_RJ_E_DIRECTION_AND_SITE_ISSUE` / `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`, with precision fragility → SITE_UNRESOLVED (tested). |
| **Layer / e\* / direction / gate / α** | Layer 16 and e\* are unchanged. There is no new direction, gate, α, dose map or localizer, and no edited forward. |
| **Gradient evaluator-only** | P2-RJ `SiteGradientProbe` is zero-valued with value identity. An AST test finds no edit function, steering hook or `cached_decode` in the P2-RJ-E runner, and no `hook=`. |
| **P3 / role exposure** | The utterances are a subset of the 300-utterance D-dev-select inference panel. No D-dev-confirm, D-test, `router-calib`, P3 or transfer identifiers appear in the P2-RJ-E sources (tested and audited). **P3 untouched.** |
| **Pipeline dry-run** | A synthetic run (scratchpad; not an outcome) passes through `analyze` and the independent `audit`: agreement on all statistics, overlap checks exercised. The original-60 subset reproduces the P2-RJ Q50 of 0.614 [0.442, 0.761] exactly. |
| **Tests** | `tests/test_inference_cf_p2rje.py`: 8 pass. The P2-RJ suite passed at the P2-RJ freeze, and its code is byte-unchanged. |
| **Compute** | One Slurm job on MIG 3g.40gb with a 1-hour limit; bf16 then fp32 passes. |

**Pre-outcome limitation (spec §3).** All eligible positions lie in the same 17 dialogues, so the
interval may remain wide. If it does, the terminal state is `SITE_UNRESOLVED`.

No blocker.
