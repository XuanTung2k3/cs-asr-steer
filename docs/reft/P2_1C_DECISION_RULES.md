# P2-1c — pre-registered decision rules

These rules were written before any P2-1c GPU output existed. They are implemented in
`reft/p2_1c_continuation_audit.py` (`status_of`, `local_disfavor`, `seq_preference`, `flags_for`, `aggregate_cells`,
`decide`) and unit-tested in `tests/reft/test_p2_1c.py`. Thresholds are in `configs/reft/p2_1c_frozen.yaml` → `rules`.

## Per case × adapted system S (damaged cases)

| Quantity | Definition |
|---|---|
| `orig_drop` | S's original beam-5 output retains ≤ 20% of the span units |
| B | span status of S's beam-5 continuation from **B0's `post_en` prefix** (requires a valid `post_en` anchor) |
| O | span status of S's continuation from **its own prefix** (requires a valid own anchor at the common switch-back location) |
| L (local disfavor) | at B0's `post_en` prefix, B0's switch-back token is not S's argmax, **or** log p_S − log p_B0 ≤ −1.0 nat |
| Q (sequence preference) | in the `post_en` candidate pool, S's best beam-equivalent score among omitting candidates (retention ≤ 0.2) exceeds its best among retaining candidates (≥ 0.8); undefined if either class is empty |

**Status thresholds:** recovered ≥ 0.8; dropped ≤ 0.2; partial otherwise.

## Flags (only when `orig_drop`; several may hold; none → UNRESOLVED)

- **H1_PREFIX_HISTORY:** B = recovered **and** O = dropped, with a valid own anchor and token prefixes that are not
  identical.
  - **Sub-type, from B0 continued from S's prefix:**
    - B0 also drops → "history induces omission even in B0";
    - B0 recovers → "adapter × history".
- **H2_LOCAL_SWITCHBACK:** B = dropped **and** L.
- **H3_SEQUENCE_PATH:** either:
  - B = dropped and not L and Q; **or**
  - O = recovered: restarting from its own prefix recovers the span that the original full search dropped.

## Cells and seed consistency

**Cell.** One damaged case and one arm (CE: CE42/43/44; KL: KL42/43/44).
- **Applicable** when at least 2 seeds of the arm have `orig_drop`.
- **A flag is supported in the cell** when at least 2 of those seeds carry it.

Seeds without the flag are reported as counterexamples. Arms that do not drop a span are reported descriptively only.

## Aggregate

**Dominant flag:**
- supported in ≥ 60% of applicable cells, **and**
- across ≥ 3 distinct damaged cases.

## Terminal verdict (in order)

1. **`P2_1C_ENGINEERING_BLOCKED`** if any of the following fails:
   - an integrity check: T1 validation for any case, leakage, weights, adapter hashes, finiteness or partial run;
   - the canonical text reproduction from output tokens.
2. **`P2_1C_INCONCLUSIVE`** if fewer than 3 damaged cases have a valid `post_en` anchor.
3. **`P2_1C_DIAGNOSTIC_COMPLETE`** if exactly one flag is dominant.
4. **`P2_1C_PARTIALLY_IDENTIFIABLE`** if either:
   - two flags are dominant, or
   - no flag is dominant but some flag is supported in ≥ 2 cells.
5. **`P2_1C_INCONCLUSIVE`** otherwise.

## Recommendation (exactly one; it is not authorization)

| Condition | Recommendation |
|---|---|
| ENGINEERING_BLOCKED, or INCONCLUSIVE because of anchors | `HOLD_FOR_ENGINEERING_OR_ALIGNMENT_REPAIR` |
| COMPLETE, H1 dominant | `GO_P2_2_ON_POLICY_PRESERVATION_DESIGN` |
| COMPLETE, H2 dominant | `GO_P2_2_SWITCHBACK_AWARE_ADAPTATION` |
| COMPLETE, H3 dominant | `GO_P2_2_SEQUENCE_LEVEL_STABILITY_DESIGN` |
| PARTIALLY_IDENTIFIABLE, with a unique leading flag in ≥ 50% of applicable cells and ≥ 3 cases | that flag's GO above |
| PARTIALLY_IDENTIFIABLE otherwise, or INCONCLUSIVE with valid anchors | `GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING` (smallest parameter-matched R3 vs M1 vs E1+D2 comparison; no acoustic-grounding claim) |

**`STOP_CURRENT_REFT_LINE` is not selected by these rules.** This audit does not measure adaptation potential, and the
English-correction gains come from P2-1/P2-1b, which this diagnostic does not revisit.

## Reporting obligations

- Case-level rows are reported before any aggregate.
- Co-occurring flags are reported together.
- All counterexamples are disclosed.
- Recovered spans (d1, d2) are contrasted with persistent spans (d0, d3, d4) descriptively only: no classifier, no
  threshold tuning, no new cases.
