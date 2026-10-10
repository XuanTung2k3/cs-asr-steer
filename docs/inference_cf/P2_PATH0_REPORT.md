# P2-PATH0 — first-divergence rescue / induction causal test (report, terminal)

**Primary verdict: `P2_PATH0_ASYMMETRIC`.** INDUCE_1 and INDUCE_3 pass; RESCUE_1 and RESCUE_3 fail.

- Independent audit: `P2_PATH0_AUDIT: PASS`, both primary-phase (before references) and full.
- Secondary descriptive flag: `ASR_HARM_ALIGNMENT = INDUCTION_ONLY`.
- Scope: exposed-development causal diagnostic on the 12 A3/A4 D rows. No significance claim; the cohort is 5 C + 7 U.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, sites, config) | `fc3602d` | sites sha256 `6ca1ada0…` |
| Implementation + plan seal | `d81c6e7` | `path_decode.py`, runner, analysis, auditor, sbatch, 7 tests |
| Pre-run audit | `ce9ff03` | `PASS_TO_P2_PATH0` (independent site reconstruction 12/12) |
| Manifest pushed | `a15a8c8` | before the job |
| Run | Slurm 58004 | 45 s allocation, 23 s runtime |
| Output seal + primary analysis | `ca4a92e` | before references; primary-phase audit PASS next commit |
| Secondary + full audit + report | this commit | INDUCTION_ONLY; `P2_PATH0_AUDIT: PASS` |

## Validity

- **Sites:** C = 5 RESCUE, U = 7 INDUCE, and all sites, prefixes, donor spans, effective lengths and suffix denominators
  were recomputed independently from raw token arrays plus termination. They are exact.
- **Reconstruction:** A4 was rebuilt on 12/12 rows through the unchanged `adapt_a4` (24 optimizer steps, plus 1 live-check
  backward; live gradient 0.89%).
  - The effective bf16 LN state is bitwise equal to the sealed checkpoint.
  - FREE-A4 equals the sealed A4 output token-for-token and in termination.
  - theta0 decode equals B0; the detected language equals the sealed value.
- **Barrier and sanity:** the barrier passed before any clamp. SELF-clamp reproduces FREE on 12/12.
- **Clamp traces:** the greedy prefix equals the frozen common prefix and every forced token was allowed. The cache
  positions are contiguous and the release index is k + effective L.
- **Resets and VRAM:** exact; non-LN weights unchanged. Peak 3.74 GB allocated / 4.05 GB reserved; 60 decode passes.

## Sites

| Row | Dialogue | Role | k | B0 token | Alt token | Eff. L1 / L3 |
|---|---|---|---|---|---|---|
| U0021_S0_513 | CSD0011 | RESCUE | 0 | 4184 | 6780 (A4) | 1 / 3 |
| U0023_S0_666 | CSD0012 | RESCUE | 5 | 13162 | 436 | 1 / 3 |
| U0023_S0_87 | CSD0012 | RESCUE | 15 | 9611 | 750 | 1 / 3 |
| U0023_S0_664 | CSD0012 | RESCUE | 0 | 31148 | 34627 | 1 / 3 |
| U0092_S0_101 | CSD0046 | RESCUE | 10 | 13992 | 9990 | 1 / 3 (DIRECT_ONLY) |
| U0029_S0_472 | CSD0015 | INDUCE | 0 | 9443 | 8239 (AUTO) | 1 / 3 |
| U0039_S0_80 | CSD0020 | INDUCE | 0 | 25246 | 8624 | 1 / 3 |
| U0091_S0_68 | CSD0046 | INDUCE | 0 | 44 | 376 | 1 / 3 |
| U0102_S0_80 | CSD0051 | INDUCE | 0 | 15003 | 4162 | 1 / 3 |
| U1003_S0_61 | CSD0502 | INDUCE | 0 | 6734 | 1079 | 1 / 3 incl. EOS (L3 DIRECT_ONLY) |
| U1079_S0_23 | CSD0540 | INDUCE | 0 | 40 | 286 | 1 / 3 (DIRECT_ONLY) |
| U2004_S0_104 | CSD1002 | INDUCE | 0 | 36757 | 4839 | 1 / 3 |

## Branch state (descriptive)

Margins are the alternative minus the B0 logit, on processed float32 logits at the common prefix.

| Group | Mean margin, theta0 | Mean margin, A4 | Mean shift |
|---|---|---|---|
| Rescue | −1.34 | +1.11 | +2.45 |
| Induction (AUTO token) | −7.11 | −4.58 | +2.54 |

- **Rescue:** A4 flips every rescue site to its alternative. Example: U0021 at k = 0 moves P_E/P_M from 0.38/0.54 to
  0.87/0.09. On U0023_S0_664 the margin goes from −4.25 to +0.44.
- **Induction:** A4 moves toward the AUTO token but stays below it everywhere (margins −0.56 to −11.2), which is why
  those rows keep the B0 path.

## Primary suffix-only endpoints

| Arm | Eligible | Direct-only | Successes / required | Success dialogues | Σd0 → Σd_target | Pooled | Pass |
|---|---|---|---|---|---|---|---|
| RESCUE-1 | 4 | 1 | 2 / 3 | 1 | 48 → 28 | 0.417 | **no** |
| RESCUE-3 | 4 | 1 | 2 / 3 | 1 | 44 → 17 | 0.614 | **no** |
| INDUCE-1 | 6 | 1 | 4 / 3 | 4 | 48 → 11 | 0.771 | **yes** |
| INDUCE-3 | 5 | 2 | 5 / 3 | 5 | 40 → 5 | 0.875 | **yes** |

**Rescue detail (R).** Both rescue successes are in one dialogue (CSD0012), so the ≥2-dialogue rule fails. RESCUE-3
also passes the pooled bar but has too few successes.

| Row | R (L1) | R (L3) | Note |
|---|---|---|---|
| U0023_S0_664 | 0.90 | 0.89 | |
| U0023_S0_666 | 1.0 | 1.0 | d0 = 1 |
| U0021 | 0.0 | 0.41 | forcing B0's first token does not change the AUTO-like English suffix |
| U0023_S0_87 | 0.0 | 0.0 | |

**Induction detail (R; every success also leaves B0, dB > 0).**

| Row | R (L1) | R (L3) |
|---|---|---|
| U0029 | 1.0 | 1.0 |
| U0091 | 1.0 | 1.0 |
| U0102 | 0.93 | 0.93 |
| U0039 | 0.5 | 0.5 |
| U2004 | 0.30 | 0.75 |
| U1003 | 0.0 | direct-only |

Precedence: not both L1, not both L3, but at least one passes, so the label is **`P2_PATH0_ASYMMETRIC`**.

## Secondary reference evaluation (after the seal; does not change the label)

**RESCUE rows (5):**

| System | ZH errors | POI errors | Mixed errors | S / D / I |
|---|---|---|---|---|
| B0 | 4 | 50 | 54 | 28 / 26 / 0 |
| FREE-A4 | 27 | 32 | 60 | 33 / 26 / 1 |
| RESCUE-1 | 7 | 36 | 43 | 16 / 27 / 0 |
| RESCUE-3 | 4 | 43 | 47 | 21 / 26 / 0 |

- **Mandarin effect:** RESCUE removes most of A4's downstream Mandarin damage. ZH errors fall 27 → 7 (L1) and → 4 (L3,
  back to B0). This is concentrated in U0023_S0_664 (22 → 2).
- **POI/English effect:** the cost is A4's English POI gains (POI errors 32 → 36 / 43). This mirrors the
  rescue-to-B0 direction.
- The rescue flag is not aligned: the primary criterion did not pass, only 1–2 rows reduce ZH errors, and POI errors
  increase.

**INDUCTION rows (7):**

| System | ZH errors | POI errors | Mixed errors |
|---|---|---|---|
| FREE-A4 = B0 | 14 | 69 | 83 |
| AUTO | 24 | 54 | 79 |
| INDUCE-1 | 24 | 54 | 79 |
| INDUCE-3 | 24 | 54 | 79 |

INDUCE-1 and INDUCE-3 reproduce AUTO's aggregate errors exactly.

- **Mandarin effect:** entering the AUTO branch creates AUTO's Mandarin damage. Over eligible rows ZH errors rise +10
  (L1, 3 rows) and +9 (L3, 2 rows); U2004 goes 0 → 8.
- **POI/English effect:** POI errors fall 69 → 54.
- Aligned on both L.

**`ASR_HARM_ALIGNMENT = INDUCTION_ONLY`.**

## Interpretation

| Question | Answer |
|---|---|
| Is one token causal? | **In the induction direction, yes.** Forcing the single AUTO first token at the divergence (k = 0 for all U rows) under the fixed A4 state sends 4/6 eligible rows' released suffixes most of the way into the AUTO path (pooled 77%), with AUTO's Mandarin damage following. **In the rescue direction, no:** one B0 token restores B0's suffix in only 2/4 rows, both in one dialogue. |
| Is a short prefix causal? | Induction: yes (L3 5/5, pooled 87.5%). Rescue: L3 reaches pooled 61% but still only 2 successes in one dialogue, so short-prefix rescue is not established on this cohort. |
| Sufficiency | Entering the AUTO branch is **sufficient** to select the AUTO basin (and its Mandarin harm) under A4 weights. |
| Necessity | Undoing A4's branch decision is **not** sufficient to return to B0 in general: U0021 and U0023_S0_87 keep A4's suffix after the B0 tokens are forced. So the branch token is not shown to be the necessary cause of A4's damage there. |
| Is the failure distributed? | Partly. Where rescue works (U0023_S0_664) the damage is branch-local. Where it fails, A4's state keeps pulling the suffix back after release, which is consistent with distributed adapted-state influence. That is a hypothesis, not a demonstration. |

**Most likely current bottleneck.** The basin is entered at early language-switch decisions, which explains how A4's
Mandarin damage appears. But it is not a single reversible token everywhere: the adapted state also biases later
decisions (rescue margins: A4 prefers the alternative at all rescue sites, mean shift +2.45).

## Next action

Stop. No follow-on is implemented. The justified next class of experiment is a **separately frozen branch-aware
path-control diagnostic**: detect and adjudicate early language-switch decisions during free decoding, for example by
comparing the competing branches' continuations. Its rescue arm must also handle the persistent adapted-state bias
(rows where B0-token forcing is not enough). This should be studied without another TTA loss, and it is not a
deployable method. A new human decision is required.

**Data:** exposed D-dev-select 12 rows only. No fresh validation, no full-24 extension, no full-100/300, no P3.
