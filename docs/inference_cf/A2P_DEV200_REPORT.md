# A2P-DEV200 — preservation-constrained decoder TTA on NEW200: REPORT

**Verdict (frozen rule, `A2P_DEV200_SPEC.md` §7): `A2P_DEV200_NOT_SUPPORTED`.**

The Mandarin-protection benefit of A2+P seen on fixed100 (26 → 3 new ZH errors) **does not generalize to NEW200**:
* On NEW200, historical A2 introduces only **9** new Mandarin errors relative to B0, in 4 utterances and 4 dialogues.
* A2+P introduces **exactly the same 9**, on the same 4 rows. Avoided errors D = 0, with bootstrap interval [0, 0].
* A2+P retains every A2 correction (36/36 POI units, 22/22 genuine lexical units).
* But it is otherwise near-identical to A2: it differs from A2 on 5/200 transcripts, with a slightly negative net effect.

Read together with fixed100, the earlier protection result looks like a property of one fixed100 trajectory
(U0023_S0_664), not a broad preservation mechanism.

This is **exposed development evidence**. NEW200 references were opened in earlier stages, so it is not independent
confirmation.

## 1. Provenance

| Item | Value |
|---|---|
| Branch | `feature/a2p-dev200`, from `146a5d0e` (`audit/ttls-r1r-independent`) |
| Frozen config | `25190862`; `configs/inference_cf/a2p_dev200.json` |
| Freeze (spec, plan, runner, evaluator, tests) | `6a642301`; plan hash `sha256:c39d964c9ca460ef0debd905dce11607a0b52f23af1e0d18a0a06b9970c4bd75` |
| Manifest (pushed before job) | `8d94411f`; `sha256:8338c982df0ed5d2c40c5c81bd37102e9eae23c15b906daf242b05b396e09ff0` |
| Outputs + seal (before NEW200 references) | `0dbd52e6`; seal `sha256:aaccd4dda88ac83bc8adc3cc1c075f2c02f15930f0fa80503a44eab925b64a87` (220 files) |
| Slurm | **58397**, COMPLETED, 8 min 26 s (486 s measured), H100 MIG 3g.40gb. One job, no smoke jobs, no sweeps |
| Model | Local Whisper-large-v3, `model.safetensors` `a8e94b85…fd95`. All 11 file hashes are equal to P2-PATH5 and TTLS-R1 (checked in prepare) |
| Environment | torch 2.10.0+cu128, transformers 4.57.6, BF16 eager, greedy, cap 200, seed 240924 |
| Population | NEW200 = P2-PATH5 roster. Membership hash `e23a8ce5…2b40` recomputed and equal; FULL300 / FIXED100 hashes equal; 20 dialogues × 10; all IDs in `exposed_ids()` |
| Data never touched | D-dev-confirm, D-test, router-calib, P3, transfer data |

## 2. What ran and what was reused

| Arm | Source | In-job verification (200/200 unless stated) |
|---|---|---|
| B0 forced-ZH | P2-PATH5 `theta0` (live canonical) | Live decode == source tokens and termination: **200/200** |
| B1 AUTO | P2-PATH5 `analysis.AUTO` (= P2-A r1 `B0_AUTO`) | Live `detect_language` + AUTO prompt replay: text **200/200**, tokens **200/200** |
| B2 A2-CE | P2-PATH5 `A2_free` (live TTA1 A2) | B3 step-0 CE == P2-PATH5 A2 loss **bitwise, 200/200**; median grad₀ relative difference 1.5e-4 (max 2.6e-3, from the P term, which is analytically zero at θ0) |
| **B3 A2-CE+P** | **New**: TTLS-R1 arm T2A code path, unchanged | §3 |

**Regression oracle.** This ran on 16 FIXED100 rows before NEW200, chosen by the frozen rule: every row where sealed A2
or T2A differs from B0, plus 2 controls. It includes U0021_S0_513 and U0023_S0_664. The results were **16/16 bitwise**:
* θ0 candidates and stable set equal the sealed R1 rows;
* B3 tokens, text, termination, all 3 losses, the CE/P parts, both grad norms and the master/effective Δ equal sealed
  R1 T2A;
* an A2-only episode through the same Episode path reproduces the **archived historical A2 fp32 masters exactly**, and
  its decode equals sealed A2.

So within this code path, A2 = A2+P with λ_P set to 0.

**Integrity.**
* All gates passed on every row:
  * initial P = 0.0 exactly;
  * two finite updates with grad₀ > 0;
  * exact LN reset;
  * no leaked hooks.
* Non-LN bytes were unchanged and no model gradients remained at job end.
* The stable set was never empty (mean |S| = 40.8).
* 27/200 rows have a nonempty candidate set M.

## 3. Primary NEW200 results

| System | PIER | MER | EN-WER | ZH-CER | S / D / I | Caps |
|---|---|---|---|---|---|---|
| B0 forced-ZH | .4637 | .2688 | .4656 | .2330 | 748 / 1986 / 239 | 1 |
| B1 AUTO | **.3836** | .2608 | **.3880** | .2367 | 652 / 2013 / 220 | 1 |
| B2 A2-CE | .4427 | **.2406** | .4466 | **.2033** | 728 / 1694 / 240 | 1 |
| **B3 A2-CE+P** | .4421 | .2414 | .4453 | .2043 | 727 / 1699 / 244 | 1 |

| vs B0 | POI corr / corrupt | Genuine lexical units | Genuine events / utterances / dialogues | Boundary / deletion units | New ZH errors (rows / dialogues) | ZH repairs | ZH retention | Outside harm | EOS-recovery rows | Premature EOS / new severe trunc. / added caps |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 AUTO | 129 / 3 | 115 | 26 / 8 / 7 | 1 / 13 | **60** (9 / 7) | 2 | .9920 | .0080 | 0 | 0 / 0 / 0 |
| B2 A2 | 36 / 3 | 22 | 6 / 3 / 3 | 1 / 13 | **9** (4 / 4) | 288 | .9988 | .0012 | 8 | 0 / 0 / 0 |
| B3 A2+P | 37 / 3 | 23 | 6 / 3 / 3 | 1 / 13 | **9** (4 / 4) | 283 | .9988 | .0012 | 8 | 0 / 0 / 0 |

**B3 vs B2 directly.**
* 5/200 transcripts differ.
* +1 POI correction (`am` → `I'm`, U1027_S0_3).
* 0 corruptions, 0 new genuine wrong-language corrections.
* **5 ZH units that A2 had correct become wrong under A2+P; 0 are repaired.**
* On mixed errors: 1 utterance improved, 2 worsened, 197 tied; by dialogue, 1 improved, 2 worsened, 17 tied.

**Dialogue bootstrap.** 2,000 paired draws, seed 240924; descriptive only.

| Contrast | ΔPIER [95%] | ΔMER [95%] | ΔZH-CER [95%] | ΔEN-WER [95%] |
|---|---|---|---|---|
| B3 − B2 | −.0006 [−.0020, .0000] | +.0007 [−.0004, +.0021] | **+.0011 [.0000, +.0027]** | −.0013 [−.0039, .0000] |
| B3 − B0 | −.0216 [−.0419, −.0054] | −.0274 [−.0540, −.0050] | −.0287 [−.0598, −.0034] | −.0204 [−.0389, −.0053] |
| B3 − B1 (AUTO) | **+.0585 [+.0006, +.1238]** | −.0194 [−.0502, +.0085] | **−.0324 [−.0634, −.0068]** | **+.0573 [+.0006, +.1207]** |
| B2 − B0 | −.0210 [−.0404, −.0054] | −.0281 [−.0549, −.0057] | −.0297 [−.0609, −.0043] | −.0191 [−.0363, −.0052] |

New-ZH counts vs B0: B2 9 [1, 18], B3 9 [1, 18]. Avoided errors D = B2 − B3 = 0 [0, 0].

## 4. Frozen decision criteria

| Criterion | Observed | Pass |
|---|---|---|
| NOT_SUPPORTED trigger: D ≤ 0 | N2 = 9, N3 = 9, **D = 0** | **triggered** |
| NOT_SUPPORTED trigger: retention < 0.5 | Genuine retention 22/22 = 1.00 | not triggered |
| NOT_SUPPORTED trigger: MER(B3) > MER(B2) + .005 | +.0007 | not triggered |
| 1 Opportunity: N2 ≥ 10 | 9 | ✗ |
| 2 Magnitude: D ≥ .5·N2 | 0 | ✗ |
| 3 Breadth (≥ 5 utterances, ≥ 4 dialogues with d_u > 0) | 0 / 0 | ✗ |
| 4 Concentration / LODO / non-termination D ≥ 5 | LODO D > 0 in 0/20 panels; non-termination D = 0 | ✗ |
| 5 Retention ≥ .8 | 1.00 | ✓ |
| 6 Safety vs B2 | MER +.0007, PIER −.0006, no new truncations or caps | ✓ |
| 7 vs AUTO: MER and ZH-CER both lower | MER .2414 < .2608; ZH-CER .2043 < .2367 | ✓ |

The label is **`A2P_DEV200_NOT_SUPPORTED`**: the frozen D ≤ 0 trigger fires.

## 5. Breadth and concentration

* **Mandarin protection.** There is nothing to attribute: every per-utterance d_u is 0 (200 tied), and the same holds
  for all 20 dialogues. The 9 A2 new-ZH errors fall on these rows, and A2+P reproduces each one:
  * U0023_S0_469: 1
  * U0040_S0_149: 3
  * U1001_S0_388: 3
  * U1027_S0_3: 2, from rendering 江苏 as "Jiangsu" in a whole-utterance English decode
* **Where the A2/A2+P gain over B0 comes from:**
  * Mixed-error gain: 13 utterances improve and 4 worsen under A2; 12 improve and 5 worsen under A2+P.
  * Under A2+P, the largest single utterance accounts for 40% of the positive gain (U1003_S0_119, −123), and the top 3
    account for 82%.
  * By dialogue: largest 40%, top 3 83%.
  * Four EOS-recovery rows (theta0 stops after 4–52 tokens and A2 continues to 80–112) account for **−274 of A2's −281
    ZH-error reduction** (A2+P: −269 of −271). This is the P2-PATH5 observation again: NEW200's A2 benefit is mostly
    termination recovery.
* **Genuine English lexical recovery is narrow, and identical for A2 and A2+P.**
  * A2 has 22 units in 6 events across 3 utterances / 3 dialogues; A2+P has 23 units in the same 6 events.
  * In all three utterances, A2 reproduces AUTO's English decoding of an almost entirely English utterance:

    | Utterance | Change | Step-0 token changed? |
    |---|---|---|
    | U0029_S0_1085 | `30分鐘…只是慢跑` → `30 minutes … just slow running` | No |
    | U0040_S0_454 | `那些冰箱非常適合你` → `那, That Iceland is very suitable for you.` | No |
    | U1027_S0_3 | Whole-utterance English | Yes (the only step-0 row) |

  * These are wrong-language recoveries, but they are whole-utterance language switches, not localized repairs of
    embedded English words.
* **Where A2+P differs from A2** (all 5 differing rows):

  | Utterance | Effect of A2+P relative to A2 |
  |---|---|
  | U0040_S0_149 | EOS-recovery row; slightly different continuation; mixed +5, ZH +5 |
  | U0091_S0_104 | Reverts toward B0's wording; mixed +5, ZH +5 |
  | U1027_S0_3 | `I'm`; mixed −2 |
  | U0085_S0_108 | Count-neutral |
  | U1080_S0_1 | Capped repetition, count-neutral |

  So preservation acted mostly as a mild pull back toward B0 on rows where A2 was *helping* Mandarin. That is the
  opposite of the hypothesized role.

## 6. Fixed100 vs NEW200

| Quantity | Fixed100 A2 | Fixed100 A2+P | NEW200 A2 | NEW200 A2+P |
|---|---|---|---|---|
| New ZH errors vs B0 | 26 | **3** | 9 | **9** |
| Avoided by P (D) | — | 23 (20 on U0023_S0_664) | — | **0** |
| POI corrections / corruptions | 31 / 2 | 30 / 2 | 36 / 3 | 37 / 3 |
| Genuine lexical units (events / utterances) | 15 (2 / 1) | 15 (2 / 1) | 22 (6 / 3) | 23 (6 / 3) |
| Genuine retention by P | — | 15/15 | — | 22/22 |
| PIER | .4540 | .4555 | .4427 | .4421 |
| MER | .2537 | **.2495** | **.2406** | .2414 |
| ZH-CER | .2211 | **.2161** | **.2033** | .2043 |
| LODO panels with D > 0 | — | 20/20 | — | 0/20 |

Notes:
* The fixed100 rule outcome, computed before freezing and given as context only, was `CONCENTRATED_OR_MIXED`.
* The fixed100 LODO figure counts panels with D > 0. Removing CSD0012 still leaves D = 3.

**FULL300 (secondary, not independent).**

| System | PIER | MER | ZH-CER |
|---|---|---|---|
| B0 | .4735 | .2647 | .2282 |
| AUTO | .3911 | .2602 | .2356 |
| A2 | .4462 | .2451 | .2095 |
| A2+P | .4462 | .2442 | .2084 |

The small FULL300 advantage of A2+P comes entirely from fixed100.

## 7. Comparison against AUTO

On NEW200, A2+P (like A2) trades English for Mandarin relative to ordinary AUTO:

| Metric | A2+P − AUTO [95%] | Better |
|---|---|---|
| PIER | +.059 [+.001, +.124] | AUTO |
| EN-WER | +.057 [+.001, +.121] | AUTO |
| ZH-CER | −.032 [−.063, −.007] | A2+P |
| MER | −.019 [−.050, +.009] | uncertain |

* AUTO makes 129 POI corrections (115 genuine, 26 events over 8 utterances / 7 dialogues) against A2+P's 37, and causes
  60 new ZH errors against 9.
* Of AUTO's corrections, 129/129 are on the 24 rows where its first content token differs from B0. A forced-ZH decoder
  (A2, A2+P, TTLS) can change step 0 on only 1 row.

This is a clear multilingual trade-off, but it is A2's trade-off. Preservation adds nothing to it on NEW200.

## 8. Optimization, compute and memory

**Optimization trajectory (means over 200 rows).**
* Total loss: .3236 → .2253 → .1824.
* CE: .3236 → .1968 → .1495.
* P: 0 → .0285 → .0329.
* grad L2: .594 → .352; master Δ L2 .814.
* KL on S after adaptation, measured on the stable set: .0329.

**Compute and memory.**
* Job: 486 s in total.
  * Setup 13.9 s; regression oracle 45.2 s (16 rows, two episodes each); NEW200 439 s.
  * Within NEW200, B3 adaptation took 41.3 s in total and B3 decoding 112.0 s. The rest is B0/AUTO replay, θ0 branches,
    displacement and checks.
* Work done:
  * B3: 432 updates and backward passes, 648 loss evaluations, 216 decodes.
  * Regression A2-only episodes: 32 backward passes.
* Trainables: 248,320 scalars per episode.
* Peak memory: 3.79 GB allocated / 4.77 GB reserved for the job; per-row B3 peak ≤ 4.37 GB.

## 9. Interpretation

1. **Numerically, A2+P is exactly what was claimed.** With λ_P = 0 the code path reproduces archived A2 bitwise, and P
   is zero at θ0. The NEW200 result is therefore not an implementation artifact.
2. **NEW200 offers little Mandarin damage to protect.** A2's harm here is 9 new ZH errors on 4 rows, against 26 on
   fixed100, most of which were one row. On those 9 errors P does nothing: they occur in decoding branches where the
   teacher-forced KL on the B0 path does not bind. One of them (U1027_S0_3) is a whole-utterance language switch that
   P does not resist.
3. **When P does act, it is not protective.** It changes 5 rows and loses 5 A2 ZH repairs; ZH-CER +.0011 [0, +.0027].
4. **Retention is real but trivial.** P keeps every A2 correction because it barely changes A2.
5. **The fixed100 result was concentrated, not general.** 20 of its 23 avoided errors came from a single trajectory, and
   that pattern does not recur on 200 new utterances from the same 20 dialogues.

Implication: A2+P should not be promoted as a broadly protective variant of A2. If a preservation idea is pursued, it
needs a different mechanism, for example one acting on free-decoding branch or termination decisions rather than
teacher-forced KL on the B0 path, and a new frozen ticket. A2 itself remains the adaptation candidate, with its
NEW200 benefit dominated by EOS recovery (P2-PATH5). No next round has been started.

## 10. Limitations

* Exposed development data. NEW200 references were opened in earlier stages, so this is not independent confirmation.
* Only 20 dialogues (the same 20 as fixed100). Event counts are tiny: 9 new-ZH errors, 3 genuine-lexical utterances.
  Intervals are descriptive.
* The decision rule is a frozen development rule. It fired on D ≤ 0, which is not threshold-sensitive.
* B1 and B2 are reused from P2-PATH5. They were verified by live B0/AUTO replay (200/200) and by step-0 CE identity for
  A2 (200/200), but the NEW200 A2 decodes were not re-executed. P2-PATH5 already audited them, and the 16-row oracle
  confirms that this code path reproduces archived A2 exactly.
* The lexical split is the frozen TTLS-R1R surface rule. "Genuine wrong-language" includes whole-utterance English
  decodes such as U1027_S0_3.
* The canonical POI-unit count and the event-level count can differ by a few units, because an event is counted as
  genuine if any member is (for AUTO: 115 vs 116 units).

## 11. Files and reproduction

**Spec and config:** `docs/inference_cf/A2P_DEV200_SPEC.md`, `configs/inference_cf/a2p_dev200.json`.

**Code:**
* `experiments/inference_cf_a2p_dev200.py` (prepare / manifest / run / seal)
* `experiments/inference_cf_a2p_dev200_evaluate.py`
* `slurm/inference_cf_a2p_dev200.sbatch`
* `tests/test_a2p_dev200.py` (7 tests)

**Outputs, under `results/inference_cf/a2p_dev200/`:**
* `plan_sealed.json`
* `run1/manifest.json`, `run1/runtime.json`
* `run1/regression/*.json`: 16 oracle rows
* `run1/rows/000–199.json`: per-utterance hypotheses, loss / parts / grad trajectories, S / M, displacement, gates,
  timing and peak memory
* `run1/slurm-58397.{out,err}`
* `output_seal.json`
* `evaluation.json`: metrics, transitions, lexical events, breadth, concentration, LODO, bootstrap, changed utterances,
  FIXED100 / FULL300 secondary, compute

```bash
cd /home/tungnx/cs-asr-steer-inf && git checkout feature/a2p-dev200
export PYTHONPATH=src:. LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
$PY -m pytest -q tests/test_a2p_dev200.py tests/test_ttls_r1.py            # CPU
# original execution (plan/manifest exist; never overwrite):
OUT=results/inference_cf/a2p_dev200/run1
sbatch --output=$OUT/slurm-%j.out --error=$OUT/slurm-%j.err --export=ALL,OUT=$OUT slurm/inference_cf_a2p_dev200.sbatch   # Slurm 58397
$PY experiments/inference_cf_a2p_dev200.py seal --run $OUT                                                               # committed 0dbd52e6
$PY experiments/inference_cf_a2p_dev200_evaluate.py --run $OUT --out results/inference_cf/a2p_dev200/evaluation.json
# replay into a NEW directory: commit, then `manifest --out <new>` (the run guard requires the manifest commit to be an
# ancestor of HEAD with only that manifest changed since), then sbatch with OUT=<new>.
```
