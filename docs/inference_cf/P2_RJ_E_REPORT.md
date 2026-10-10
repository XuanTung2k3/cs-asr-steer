# P2-RJ-E leverage expansion — report (final diagnostic stage)

**Final diagnosis: `P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`** (terminal).

- **Audit:** `P2_RJ_E_AUDIT: PASS` (`P2_RJ_E_POST_RUN_AUDIT.md`).
- **The diagnostic programme is ENDED.** There is no P2-RJ-F, no further Q50 expansion, and no new
  Jacobian, random-control or localization diagnostic.
- **P3:** **HELD.**
- **Next step:** one independent actuator-design pass (`P2_RJ_E_ACTUATOR_REDESIGN_HANDOFF.md`).

**History is unchanged:**

| Stage | Standing |
|---|---|
| P2 | `P2_VALID_CONFIGURATION_SELECTED`, `P2_AUDIT: PASS` |
| P2-R | `P2_R_MECHANISM_STILL_AMBIGUOUS`, `P2_R_AUDIT: PASS` |
| P2-RJ | `P2_RJ_STILL_AMBIGUOUS`, `P2_RJ_AUDIT: PASS`: alignment MISALIGNED, leverage UNRESOLVED |

## Run

| Item | Value |
|---|---|
| Job | 55002 (MIG 3g.40gb): bf16 66 s, fp32 57 s, peak 8.1 GB |
| Commit / manifest | `646179b` / `sha256:a6fc9cce…` |
| Rows | 43/43 utterances × 2 passes ok; **227/227 positions** in both passes |
| Attempts | No failed attempt; nothing rerun |

## Population expansion

| Item | Value |
|---|---|
| Eligible EN-confusion positions | **227**: 60 original P2-RJ + **167 new** |
| Utterances / dialogues | 43 / **17** |
| Eligibility code | Unchanged P2-R; rebuild reproduces the frozen P2-R population hash; count equals the P2-R ledger |
| Selection | none |

**Exclusions** (371 candidate units):

| Reason | Units |
|---|---|
| mid-character prefix | 45 |
| not a token start | 36 |
| multiple reference units | 30 |
| R2-unalignable | 22 |
| forced-prefix first content token | 11 |

The expansion adds **no dialogues**: every eligible position lies in the same 17 dialogues. This
limitation was disclosed before the outcome.

## Protocol identity

**No scientific change from P2-RJ.** Everything below is identical to P2-RJ:

- layer 16, the DG-02 post-cross-attention / pre-FFN site, bf16 plus an fp32 replicate;
- the margin logsumexp z(Y_ref) − z(c\*);
- the zero-valued probe gradient and the tangent projection;
- e\* = 1.1260757575454359;
- the dialogue bootstrap (10,000, seed 240924), α = 0.025 per bound, Q50 at 0.5, and
  `leverage_class`.

**c\* for new positions.** It was produced by the identical rule (P2-R top-20 order) in a no-grad
pre-pass. Two positions have an exact tie for c\*, which m(r₀) ignores.

**Overlap.** The 60 original positions reproduce P2-RJ run1 **bitwise**: maximum deviation
2.2e-16 across gap, ‖g_tan‖ and gradient cosine. The original-subset Q50 re-computes to exactly
0.614 [0.442, 0.761].

**Post-run addition (disclosed).** `inference_cf_p2rje_write_analysis.py` is a serialization-only
wrapper.

- One new position has gap ≤ 0 (m(r₀) = +0.10). Under the frozen rule its ρ = +∞, which strict
  JSON cannot store.
- The frozen `analyze` function is called unchanged, and no statistic or decision changes.

## Leverage — all 227 EN-confusion positions (bf16 primary)

| Quantity | Value |
|---|---|
| Median gap | **10.56 nats** (IQR 8.36–12.23) |
| Median ‖g‖ | 4.35 |
| Median ‖g_tan‖ | **4.35** |
| Median A = ‖g_tan‖·e\* | **4.90 nats** (IQR 3.12–7.38) |
| Median ρ = A/gap | **0.52** (IQR 0.33–0.71) |

**How much of the gap an optimally aligned e\*-edit could close, to first order:**

| Closes | Positions | Dialogue-weighted |
|---|---|---|
| ≥ 10% | 100% | 1.00 |
| ≥ 50% | 52.4% | **Q50 = 0.539**, Bonferroni CI **[0.406, 0.665]** |
| ≥ 100% | 11.9% | 0.147 [0.066, 0.242] |

- **Leverage class: `UNRESOLVED`** in bf16 and in fp32.
  - fp32 Q50 = 0.519 [0.385, 0.647].
  - The median g_tan cosine between precisions is 0.9998.
- **Split:**
  - original 60: Q50 0.614, median ρ 0.58;
  - new 167 (13 dialogues): position-level ρ ≥ 0.5 is 0.485, median ρ 0.48.

The expanded point estimate moved *toward* 0.5, and the interval still contains it.

## H1 confirmation (descriptive; not reopened)

| Quantity | Value on all 227 positions |
|---|---|
| cos(g_tan, +d) | −0.0097 [−0.019, −0.002]; median \|cos\| 0.024 (random: 0.022) |
| κ(+d) | **−0.0098** [−0.019, −0.002] |
| κ(−d) | **+0.0099** [0.002, 0.019] |
| κ(random) | −0.0001 [−0.009, 0.006] |

**Interpretation:**

- d = norm(hE − hB) captures ≈ −1% of the available first-order lexical sensitivity. Its
  |cosine| with the useful direction is at the level of a random direction in 1,280 dimensions.
- **−d is not thereby validated.** It captures only about +1%.
- Neither the random direction nor the reference gradient is promoted.

## Mechanistic conclusion

1. **H1 is established on the complete eligible population.** d = norm(hE − hB) is **not** a useful
   lexical-repair direction at the tested L16 site. It captures approximately none (≈ −1%) of the
   available first-order lexical sensitivity. P2-RJ additionally showed that the useful direction
   is largely a Latin-vs-Han script-readout direction (cos 0.80), which d also misses.
2. **The first-order model is sound.**
   - P2-RJ found r = 0.92 and slope 1.03 against the realized P2-R edits.
   - Precision is robust (cosine 0.9998).
   - The P2/P2-R inertness is therefore geometric, not an artefact.
3. **H4 is unresolved after exhausting the diagnostic role.** At the fixed P2 budget e\* ≈ 15% of
   ‖r‖, an optimally aligned L16 edit would:
   - close a median of about half of the ~10.6-nat lexical gap (ρ 0.52);
   - fully close it at only ~12% of positions.

   Whether "most positions reach half the gap" cannot be decided: Q50 is 0.54 [0.41, 0.67] over
   the only 17 dialogues that contain eligible EN-confusion units. L16 at e\* is neither clearly
   sufficient nor clearly insufficient.
4. **Consequence for redesign.** The **direction is the confirmed defect**. Site adequacy at e\* is
   borderline and unresolved. Even a perfectly aligned first-order edit at e\* would flip only a
   minority of confusions, so a redesign should not assume that L16 at the P2 budget suffices.
   Per the frozen terminal policy, redesign proceeds **direction-first**, with a predeclared
   site-failure criterion.

## Diagnostic programme

**ENDED: YES.** No further P2-R-family diagnostic is authorized.
