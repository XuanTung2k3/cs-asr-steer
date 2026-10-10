# A2P-DEV200 — preservation-constrained decoder TTA on NEW200 (FROZEN SPEC)

**Status:** FROZEN before execution, before any NEW200 A2+P output exists, and before any NEW200 reference is opened in
this ticket.

| Item | Value |
|---|---|
| Machine-readable contract | `configs/inference_cf/a2p_dev200.json` |
| Branch | `feature/a2p-dev200`, from `146a5d0e` (`audit/ttls-r1r-independent`) |
| Ticket type | Separately frozen inference_cf exploratory ticket |
| What stays unchanged | Core v6, METHOD_CONTRACT and all earlier reports |
| Evidence status | **Exposed-development generalization study, NOT independent confirmation** |

## 1. Question

Does preserving stable baseline decoder distributions during episodic LayerNorm adaptation reduce new Mandarin
recognition errors while retaining genuine embedded-English lexical corrections? Is that benefit broad, rather than
driven by isolated decoding trajectories?

The answer is assessed against matched A2 and against ordinary Whisper AUTO.

**Fixed100 context (TTLS-R1, sealed; reproduced exactly by this ticket's evaluator before freezing):**

| Arm | POI corrections / corruptions | Genuine wrong-language POI units | New ZH errors (vs B0) | PIER | MER |
|---|---|---|---|---|---|
| A2 | 31 / 2 | 15 | 26 | .4540 | .2537 |
| A2+P | 30 / 2 | 15 | 3 | .4555 | .2495 |

- The 15 genuine units are **2 events in 1 utterance** (U0021_S0_513).
- 20 of the 23 avoided new-ZH errors are on **one utterance** (U0023_S0_664).
- The frozen rule below would label fixed100 `CONCENTRATED_OR_MIXED`. This is reported as context; the rule was not
  tuned to it.

## 2. Population

**NEW200** is the frozen P2-PATH5 roster: FULL300 − FIXED100, 200 utterances, 20 dialogues × 10, in FULL300 order.

- Panel: `docs/inference_cf/P2_PATH5_PANEL.json` (sha256 `4386b1fc…`).
- Membership hash (P2-PATH5 convention): `e23a8ce5…2b40`. It is recomputed and checked in `prepare`, together with the
  FULL300 and FIXED100 hashes.
- It is not resampled.

**Exposure.**
- All FULL300 IDs are in `exposure_registry.exposed_ids()`.
- NEW200 references were previously opened in DG-04/06/07, P0-R2, P2 selection and P2-PATH5 scoring.
- A2+P has never been evaluated on NEW200.
- This is **not fresh validation**. Because these IDs are already registered, no new exposure addendum is required
  (the hash-pinned addenda file is not edited).

**Never accessed:** D-dev-confirm, D-test, router-calib, P3, transfer datasets and any unexposed data.

## 3. Arms

| Arm | Method | Source | In-job identity check (all 200 rows; failure → INVALID) |
|---|---|---|---|
| B0 | Frozen Whisper forced-ZH greedy, cap 200 | P2-PATH5 run1 `theta0` (live canonical cached forced decode, the TTA1/S0 path). The older P2-A `systems.B0` (19/100 fixed100 mismatches) is **not used** | Live forced decode == source tokens and termination |
| B1 | Frozen Whisper AUTO greedy | P2-PATH5 plan `analysis.AUTO` (= sealed P2-A r1 `systems.B0_AUTO`, the TTA1 teacher source) | `detect_language` + `auto_prompt` live decode text == source (TTLS-R1 convention); token equality recorded |
| B2 | Historical A2-CE | P2-PATH5 run1 `A2_free`: the live TTA1 A2 episode on the same theta0 / teacher / prompt, P2-PATH5 audit PASS | B3 step-0 CE == P2-PATH5 `A2_log.losses[0]` **bitwise**. Same theta0, teacher, mask and prompt; this already holds 100/100 on fixed100 for sealed T2A |
| B3 | A2-CE+P | **New GPU execution** | §4 |

Only B3 involves new adaptation. B0, B1 and B2 are not regenerated, only re-verified.

## 4. B3 = TTLS-R1 arm T2A, unchanged

The same code path is used: `ttls.Episode(variable="LN")`, `ttls.ce_terms`, `ttls.preservation_term`,
`ttls.select_candidates`, `ttls.stable_positions` and `ttls.displacement`. The θ0 branches come from
`experiments/inference_cf_ttls_r1r.theta0_branches`, which is byte-identical to the TTLS-R1 code and was verified equal
on 100/100 rows in R1R.

**Settings:**
- Whisper-large-v3, BF16, eager.
- Trainables: 194 decoder LayerNorm affine tensors (248,320 scalars). Fresh fp32 masters per utterance; bf16
  `functional_call`; exact `LNGuard` restore.
- AdamW: lr 1e-3, weight decay 0, betas (0.9, 0.999), eps 1e-8. Two updates, three loss evaluations.

**Loss:**

```
L = L_A2_CE + 1.0 * L_P

L_A2_CE = valid-position mean NLL of y_A under the cB (forced-ZH) prompt
          (episodic_tta.position_terms + objective_loss('A2'); historical EOS / valid-mask conventions)

L_P = mean_{t in S} KL( p_theta0(. | y_B[:t], audio, cB) || p_theta(. | y_B[:t], audio, cB) )
      allowed vocabulary (soft_auto_tta.allowed_log_probs / kl_terms)
      if y_A == y_B the CE logits are reused for P
      empty S -> L_P = 0 * sum(logits)

S = { t : y_B_valid_mask[t]  and  p_theta0(y_B[t]) >= 0.5  and  t not in M }
```

- **y_A** is P2-PATH5 `clean_teacher(AUTO)`, hash-checked against the panel.
- **y_B** is B0.
- **M** follows the frozen TTLS-R1 candidate rule:
  - branches: cB/cE prompts × clean / 30-s null audio;
  - the `<|en|>` argmax is embedded-Latin while b_t is not;
  - the prefix is UTF-8 complete;
  - E^cB ≥ log 10 and E^cE ≥ log 10.

  Everything is θ0-only, with no reference.
- **Final decode:** forced-ZH greedy, cap 200, under the effective bf16 state, then exact restore.

## 5. Regression oracle and integrity gates (one job; phase R runs before NEW200 and stops the job on failure)

**Phase R** uses a FIXED100 subset chosen by a frozen rule: every TTLS-R1 row whose sealed B2 or T2A tokens differ from
y_B, plus the first two unchanged rows. This covers both influential rows, U0021_S0_513 and U0023_S0_664. Exact checks:
1. θ0 candidates and S equal the sealed R1 row.
2. B3 tokens, termination, text, losses, loss parts, grad norms, master Δ and effective Δ equal sealed R1 T2A.
3. An A2-only episode through the same Episode path gives fp32 masters equal to the **archived historical A2 masters**,
   and its decode equals sealed A2. This validates A2 = A2+P − P for this code path.

**Phase N** runs on every NEW200 row and checks:
* B0 identity and B1 replay;
* B3 step-0 CE == P2-PATH5 A2;
* initial P == 0.0 exactly;
* 2 finite updates with grad₀ > 0;
* exact LN reset, and no site hooks.

**At job end:** non-LN bytes unchanged and no model gradients.

**Recorded per row:**
* losses and parts;
* grad norms;
* master and effective Δ;
* the S / M sets;
* displacement (relative L16 site change, KL on all steps and on S);
* tokens and text;
* time and peak memory.

The B3-vs-PATH5-A2 grad₀ relative difference is recorded descriptively. P is exactly 0 at θ0, but its gradient
vanishes only analytically.

## 6. Evaluation (post-seal; `experiments/inference_cf_a2p_dev200_evaluate.py`)

NEW200 references open only after the committed A2P-DEV200 output seal **and** the P2-PATH5 and TTLS-R1 seals verify.

**Metrics.**
* Canonical metrics through the unchanged TTLS-R1 evaluator primitives: PIER, MER, EN-WER, ZH-CER, S/D/I, caps.
* POI corrections and corruptions vs B0, and B3 vs B2.
* New ZH errors and ZH repairs.
* Matrix-ZH retention and outside-POI harm.
* EOS-recovery, premature-EOS, severe-truncation and added-cap rows.

**Lexical split** (frozen TTLS-R1R rule): genuine_wrong_language / genuine_same_language / word_boundary_repair /
deletion_or_boundary.

**Three granularities are reported separately:**
* POI units;
* **lexical events**: maximal runs of consecutive reference POI indices with the same transition, in one utterance;
* utterances and dialogues.

**Retention** is the fraction of B2's corrected POI units, genuine and all, also corrected by B3.

**Breadth** is the number of improved / worsened / tied utterances and dialogues on mixed errors for B3−B0, B3−B2,
B2−B0 and B3−B1.

**Concentration.** Per utterance, d_u = newZH_B2(u) − newZH_B3(u), also aggregated per dialogue. Reported:
* the largest single-utterance and single-dialogue share;
* the top-3 shares;
* the same shares for the mixed-error gain.

**Leave-one-dialogue-out** (20 panels): D, ΔMER and ΔPIER for B3 − B2.

**Dialogue bootstrap.** 2,000 paired dialogue resamples, seed 240924, descriptive only. It covers rate deltas for
B3−B2, B3−B0, B3−B1, B2−B0, B2−B1 and B1−B0, plus count intervals for D and for new-ZH.

**Termination-changed rows:** a row counts if, for B2 or B3, the termination differs from B0, or EOS recovery or
premature EOS holds. D is also computed excluding them.

**Secondary results.**
* FIXED100 from sealed TTLS-R1 (B3 = T2A) with the same analysis.
* A FULL300 aggregate.
* Neither is independent data, and neither enters the rule.

## 7. Frozen decision rule (NEW200 only)

Notation: N2 = new-ZH(B2 vs B0), N3 = new-ZH(B3 vs B0), D = N2 − N3.

1. **`A2P_DEV200_INVALID`**: any integrity failure. This covers regression oracle, B0/B1/B2 identity, reset, non-LN
   change, missing rows, and seals.
2. **`A2P_DEV200_NOT_SUPPORTED`**: any one of the following:
   * D ≤ 0;
   * retention < 0.5 when assessable (genuine retention if B2 has ≥ 3 genuine units, else all-correction retention if B2
     has ≥ 3 corrections);
   * MER(B3) > MER(B2) + 0.005.
3. **`A2P_DEV200_BROAD_SUPPORT`**: all of the following:
   1. Opportunity: N2 ≥ 10.
   2. Magnitude: D ≥ 0.5·N2.
   3. Breadth: at least 5 utterances with d_u > 0, and at least 4 dialogues with a positive dialogue sum. The number of
      utterances with d_u > 0 must be at least twice the number with d_u < 0.
   4. Concentration:
      * largest single-utterance share of Σ_{d_u>0} d_u ≤ 0.5;
      * D > 0 in at least 18 of 20 LODO panels;
      * D on non-termination-changed rows ≥ 5.
   5. Retention ≥ 0.8 when assessable; otherwise it passes and is reported.
   6. Safety vs B2:
      * MER(B3) ≤ MER(B2) + 0.002;
      * PIER(B3) ≤ PIER(B2) + 0.01;
      * new severe truncations and added caps no worse than B2's.
   7. Versus AUTO: MER(B3) < MER(B1) and ZH-CER(B3) < ZH-CER(B1). The PIER / EN-WER gap is reported as the trade-off.
4. **`A2P_DEV200_CONCENTRATED_OR_MIXED`**: otherwise.

Every verdict is exposed-development evidence. **Even BROAD_SUPPORT is not confirmation.** Thresholds will not be
revised after outcomes are seen.

## 8. Compute and process

* **One Slurm job** on `mig`, H100 3g.40gb, ≤ 3 h. No smoke jobs and no sweeps.
* Expected runtime: about 15–25 min. TTLS-R1 took 8.8 min for 100 rows with 7 arms; P2-PATH5 took 13.9 min for 300 rows.
* Order of operations:
  1. Freeze commit: spec, config, plan, code and tests.
  2. Commit and push the manifest.
  3. Submit the job.
  4. Commit and push the outputs and seal.
  5. Run the evaluation.
  6. Write the report.
* No next round starts automatically.

## 9. Deviations from the TTLS-R1 T2A arm

There are none in the method.

**Differences in instrumentation only:**
* Scalar parts are logged via `float(x.detach())`; the value is identical.
* Additional gates and records: regression oracle, step-0 CE identity, AUTO replay, state hash.

**Population:**
* NEW200 rows take y_B from P2-PATH5 `theta0`.
* `y_B_valid_mask` and `utf8_ok` are recomputed with the TTA1 / TTLS-R1 functions (`valid_mask`,
  `prefix_utf8_complete`).
* `group` (D/A) is descriptive only: D means y_A ≠ y_B.
