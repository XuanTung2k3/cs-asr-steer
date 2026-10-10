# TTLS-R1R — numerical repair, re-execution and revalidation: REPORT

**Label (frozen rule, `TTLS_R1R_SPEC.md` §7): `TTLS_R1R_VALID_BUT_INSUFFICIENT`.**

**Repair.** The ratio-first repair restores an exact clean start. With z = 0 under the actual masks, free decoding
reproduces B0 on 100/100 utterances. Teacher-forced logits and the L16 FFN input are bitwise identical on 100/100 rows,
and the initial preservation KL is exactly 0. The gradient at z = 0 is nonzero on every episode.

**Corrected TTLS.** The corrected arms are therefore causally attributable to optimizing z. They still do not deliver
meaningful English lexical recovery:
* the best arm, T1 TTLS-CE, makes **4 genuine lexical corrections, all in one utterance / one dialogue**
  (`回到学校 → go back to school`);
* its other 8 corrections are deletion, EOS or word-boundary repairs;
* the acoustic arms add nothing over direct candidate substitution.

**Recommendation.** Close this TTLS configuration (utterance-global L16 vector, A2 CE / AC objectives). Do not start
TTLS Round 2.

Exploratory development evidence only. The panel is 100 already-exposed D-dev-select utterances in 20 dialogues. No
D-dev-confirm, D-test, router-calib, P3 or fresh data was used.

## 1. Provenance

| Item | Value |
|---|---|
| Branch | `feature/ttls-r1r-repair`, from audit commit `f99f176b` (`audit/ttls-r1-independent`) |
| Freeze (sources, spec, config, evaluator, tests) | `38723a01` |
| Resolved manifest (pushed before job) | `77e68fa5`, `sha256:6d39fafe02f7423df108a716d1054656528af639a6297dbf36e76359b52c651d` |
| Outputs + seal (before any reference access) | `caa4f3c1`, seal `sha256:6cba364127a815b74fab02eb5cc935faa6a39a36e60ec173e8c9c4bdca172a22` (204 files) |
| Slurm | **58391**, COMPLETED, 6 min 24 s, node `worker-mig-3g40gb-0`, H100 MIG 3g.40gb (one job; no smoke jobs) |
| Config | `configs/inference_cf/ttls_r1r.json` sha256 `d4f378ea…1f83` |
| Inherited plan | TTLS-R1 `plan_sealed.json`, hash `sha256:61ba6a52…e31e`; R1 config `8270f663…`; R1 seal `e0d02378…` (all verified in-job) |
| Model | local Whisper-large-v3, `model.safetensors` sha256 `a8e94b85…fd95`, `tokenizer.json` `6d8cbd7c…21ca` (all 11 files in manifest) |
| Environment | torch 2.10.0+cu128, transformers 4.57.6, numpy 1.26.4, CUDA 12.8, Python 3.11.9; BF16 eager, greedy, cap 200; seed 240924 |
| Evaluation | `results/inference_cf/ttls_r1r/evaluation.json` sha256 `e4f53489…7c22` |

## 2. Repair and what changed relative to TTLS-R1

The historical kernel `models.hooks.apply_steering` divides and then multiplies:
`steered / (new_norm + EPS) * orig_norm`. In BF16 this is not the identity at δ = 0.

The repair, in the new module `src/csasr/inference_cf/ttls_r1r.py` (`NUMERICS = "ratio_first_v1"`), computes the ratio
first: `steered * (orig_norm / new_norm.clamp_min(EPS))`. When δ = 0, `steered` is bitwise `hidden`, so the ratio is
exactly 1.

How the repair is wired in:
* It is used through `RatioFirstInterventionHook`, a subclass of the canonical DG-02 hook that overrides only `_hook`.
* There is no z = 0 branch and no detach.
* `hooks.py`, `sites.py`, `ttls.py`, every R1-pinned source and all R1 outputs are byte-unchanged; a test checks this.
  TTLS-R1 therefore stays replayable.

What is unchanged: the population, y_A / y_B teachers, candidate rule, stable set, masks, L16 site, variable, AdamW
settings, two steps, budget E*, objectives and decoding protocol. The in-job θ0 candidates and stable sets were
**identical to the sealed R1 rows on all 100 rows**.

Other changes (recorded in spec §9):
* Instrumentation: phase A zero census, bitwise probes, zero-KL, per-episode gates and a post-row clean re-decode.
* `float(loss.detach())` logging. The value is identical.
* The evaluator gains the frozen lexical split, the effect decomposition and the R1R label.

## 3. Zero-edit integrity (phase A runs before any optimization; every gate passed)

| Check (z = 0, no optimizer) | Repaired kernel | Historical kernel (contrast) |
|---|---|---|
| Free decode = B0, CE mask ALL | **100/100** | 94/100 (audit census, 6 token changes) |
| Free decode = B0, AC mask {t\*} | **6/6** accepted rows | — |
| Teacher-forced logits bitwise = clean | **100/100**, max \|Δlogit\| **0** | 0/100, max \|Δlogit\| 0.25 |
| L16 FFN input bitwise = clean | **100/100** | max chord 0.0230 |
| Initial preservation KL (stable set) | **0.0** on every row and mask | R1: nonzero on 97/100 T2 and 5/6 T6 episodes (max 4.24e-4) |

Phase B per-episode gates all passed:
* fresh zero z and empty optimizer;
* step-0 gradient finite and > 0 (minimum 0.0263 for CE, 1.19 for AC);
* initial P = 0.0 exactly in T2 and T6;
* the edit is consumed, with FFN-input chord max 1.094 ≤ E\*;
* norm preservation at the FFN input, |‖r̃‖/‖r‖ − 1| ≤ 0.0084 (gate 0.02);
* **exactly 0** FFN-input displacement outside the mask;
* budget held, with the projection never binding;
* no leaked hooks, and LN θ0 intact;
* **post-row clean decode = B0 on 100/100 rows**.

At job end, all parameter bytes were unchanged and the model held no gradients. B0 reproduced the sealed S0 on
100/100 rows.

## 4. Primary results (free decoding, 100 utterances)

Corrections, corruptions and new-ZH errors are counted relative to B0. Lexical split (frozen):
* **gen** = genuine lexical (wrong-language + same-language substitution);
* **bnd** = word-boundary or spacing repair;
* **del** = deletion, boundary error, or insertion.

| System | PIER | MER | EN-WER | ZH-CER | POI corr/corrupt | gen / bnd / del | gen dialogues | new ZH | ZH repairs | EOS-recovery rows |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 forced-ZH | .4957 | .2570 | .5029 | .2193 | – | – | – | – | – | – |
| Z0 zero vector (repaired) | .4957 | .2570 | .5029 | .2193 | 0/0 | 0/0/0 | 0 | 0 | 0 | 0 |
| Z0H zero vector (historical) | .4957 | .2416 | .5029 | .2017 | 0/0 | 0/0/0 | 0 | 3 | 16 | 0 |
| B1 AUTO | **.4080** | .2591 | **.4152** | .2336 | 63/2 | 37/2/24 | 3 | 72 | 0 | 0 |
| B2 historical A2 | .4540 | .2537 | .4626 | .2211 | 31/2 | 15/2/14 | 1 | 26 | 26 | 4 |
| **T1 TTLS-CE (R1R)** | .4784 | .2519 | .4871 | **.2157** | **12/0** | **4**/3/5 | **1** | **0** | 17 | 4 |
| T1 TTLS-CE (R1, original) | .4842 | .2526 | .4928 | .2157 | 8/0 | 0/3/5 | 0 | 0 | 17 | 4 |
| **T2 TTLS-CE+P (R1R)** | .4871 | .2530 | .4957 | .2157 | 6/0 | 0/2/4 | 0 | 0 | 17 | 4 |
| T2A A2-CE+P | .4555 | **.2495** | .4641 | .2161 | 30/2 | 15/2/13 | 1 | 3 | 26 | 4 |
| T3 A2-AC | .4899 | .2611 | .4971 | .2248 | 4/0 | 1/0/3 | 1 | 30 | 1 | 0 |
| **T4 TTLS-AC (R1R)** | .4914 | .2606 | .4986 | .2240 | 3/0 | 1/0/2 | 1 | 25 | 0 | 0 |
| T5 A2-AC+P | .4914 | .2618 | .4986 | .2254 | 3/0 | 1/0/2 | 1 | 32 | 0 | 0 |
| **T6 TTLS-AC+P (R1R)** | .4914 | .2606 | .4986 | .2240 | 3/0 | 1/0/2 | 1 | 25 | 0 | 0 |
| ACSUB direct substitution | .4914 | .2606 | .4986 | .2240 | 3/0 | 1/0/2 | 1 | 25 | 0 | 0 |
| CD contrastive decoding | .4828 | .2746 | .4856 | .2419 | 58/49 | 37/1/20 | 13 | 254 | 46 | 0 |

**Other events.**
* No R1R arm has premature-EOS rows, new caps or new severe truncations. The Z0H historical control has one severe
  truncation flag and fails the safety bounds.
* T1 and T2 keep matrix-ZH retention at 1.000, with outside-POI harm 0/4,055.
* T4 and T6: retention .9938, outside harm .0062.
* All R1R arms are within the R1 safety bounds vs B0.
* Rows that changed vs B0, and their effect on mixed errors:

  | Arm | Rows changed | Better | Worse | Net mixed errors |
  |---|---|---|---|---|
  | T1 | 11 | 8 | 0 | −29 |
  | T2 | 10 | 6 | 0 | −23 |
  | T4 / T6 | 5 | 2 | 2 | +21 |

**Dialogue-cluster bootstrap.** 2,000 draws, seed 240924, method − comparator, descriptive only.

| Contrast | ΔPIER [95%] | ΔMER [95%] | ΔZH-CER [95%] |
|---|---|---|---|
| T1 − B0 | −.0172 [−.0336, −.0041] | −.0050 [−.0088, −.0014] | −.0036 [−.0072, −.0003] |
| T1 − T1(R1) | −.0057 [−.0185, .0000] | −.0007 [−.0021, .0000] | .0000 [.0000, .0000] |
| T1 − B2 | +.0244 [−.0198, +.0774] | −.0017 [−.0131, +.0074] | −.0053 [−.0174, +.0023] |
| T1 − B1 (AUTO) | +.0704 [.0000, +.1482] | −.0071 [−.0180, +.0043] | −.0178 [−.0370, −.0038] |
| T2 − T1 | +.0086 [.0000, +.0241] | +.0010 [.0000, +.0026] | .0000 |
| T2 − T2A | +.0316 [−.0074, +.0828] | +.0035 [−.0013, +.0106] | −.0004 [−.0052, +.0044] |
| T4 − T3 | +.0014 [.0000, +.0046] | −.0005 [−.0025, +.0007] | −.0008 [−.0029, +.0006] |
| T4 − ACSUB, T6 − ACSUB | 0 | 0 | 0 |

**Subgroup PIER.**

| Subgroup | B0 | B1 | B2 | T1 | T2A |
|---|---|---|---|---|---|
| A3-24 | .586 | .341 | .514 | .566 | .514 |
| D12 | .758 | .369 | .656 | .739 | .656 |
| A88 | .419 | .419 | .395 | .403 | .397 |

## 5. Effect decomposition

### 5.1 Effect of the numerical repair

* **Controls.** The historical zero control changed 6 rows. It cut mixed errors by 89, mostly on one capped row,
  U1003_S0_141. That improvement came from rounding, not steering. The repaired zero control changes nothing.
* **T2, T4, T6.** Token-identical to R1 on all 100 rows. The optimized vectors are near-identical: median cosine
  between R1R and R1 z is .991–.998, and the final-norm medians match within 1e-3.
* **T1.** Differs from R1 on only 2 rows.
  * **U0086_S0_185** (CSD0043): `重新回到学校` → `重新go back to school`. This is 4 wrong-language POI corrections and
    −4 mixed errors. Reference: `重新 go back to school`.
  * **U0092_S0_101**: `更加` → `更为`. The normalized text changed, with no metric change.
* **Fragility.** The repair shifted the BF16 rounding of the norm and flipped a whole-phrase code-switch decision on
  U0086_S0_185. This only shows how close that decision sits to a threshold; it is not evidence of a robust mechanism.
* **Optimization trajectories** are near-identical to R1:
  * T1 mean loss .4131 → .3432 → .2936, decreasing on 100/100 rows;
  * gradient norm .1826 → .1430 (R1: .1830 → .1429).
* The R1 qualitative picture is unchanged. What R1R adds is a valid clean baseline.

### 5.2 Effect of optimizing z (R1R arm vs Z0 = B0; now causally clean)

**TTLS-CE (T1).**
* 12 POI corrections and 0 corruptions over 11 changed rows, in 5 dialogues; no row got worse.
* Composition of the corrections:
  * 4 genuine wrong-language corrections, all in U0086_S0_185 (one dialogue);
  * 3 word-boundary repairs (`peoplethey`, `withlike`, `somerelationship`);
  * 5 deletion recoveries; 2 of them (`super topic`) come from EOS recovery.
* 17 ZH units repaired, 0 new ZH errors.
* PIER −.017, MER −.005 and ZH-CER −.004 vs B0, all with intervals excluding 0.
* This is a real but small, mostly non-lexical gain from optimizing z.

**TTLS-AC (T4).**
* Optimization moves the candidate to rank 0 on 5/6 rows: mean margin over b goes −0.87 → +1.49.
* Free decoding gives 3 corrections, 1 of them genuine (`烤烧 → cooking`, U1034_S0_53), plus 25 new ZH errors.
* 23 of those 25 new ZH errors come from U1003_S0_63, where inserting ` or` ends the sentence early.

### 5.3 Effect of preservation

**On TTLS.**
* With the repair, P is **exactly 0 and gradient-free at the first update** and acts only on the second, as R1 claimed
  but R1 did not deliver.
* T2 vs T1: P removes half the corrections (12 → 6), **including all 4 genuine lexical ones**, and prevents no harm: both
  arms have 0 new ZH errors.
* T6 is token-identical to T4. KL(P) at the final step is only 1.1e-4.

**On A2.** A2+P (T2A) remains the useful preservation result. It keeps the same 15 genuine corrections as A2 and cuts
new ZH errors from 26 to 3. These 15 corrections sit on one utterance, U0021_S0_513.

### 5.4 Remaining confounds: global masks and inaccessible first-token decisions

TTLS-R1R is **still not a localized, opportunity-matched actuator comparison**:
* TTLS can never change content step 0 (forced prefix 4).
* The CE mask is utterance-global: one vector at every editable step.
* A2 changes all 194 decoder LayerNorm affine tensors (248,320 scalars), with a different learning-rate rule and a
  different functional budget.

The step-0 census shows that the large lexical gains of AUTO and A2 lie almost entirely where TTLS cannot act:

| System | Rows whose step-0 token ≠ B0 | Corrections on those rows | Genuine lexical on those rows |
|---|---|---|---|
| B1 AUTO | 12 | **63 of 63** | **37 of 37** |
| B2 A2 | 3 | 15 of 31 | **15 of 15** |
| T2A A2+P | 2 | 15 of 30 | **15 of 15** |
| TTLS (all arms) | 0 (structurally impossible) | – | – |

On the rows TTLS can reach, B2 makes 16 corrections with 0 genuine lexical; T1 makes 12, with 4 genuine lexical in one
utterance. This descriptive, single-utterance contrast does not make TTLS a better lexical actuator. It shows that:
* the R1/R1R A2-vs-TTLS comparison mostly measures **access to the whole-utterance first-token language decision**, not
  representation-space vs parameter-space adaptation;
* on reachable positions, neither method produces broad lexical recovery.

## 6. Comparisons asked for

* **Corrected vs original TTLS.** Covered in §5.1: identical except T1 (+4 genuine corrections on one row, PIER −.0057).
* **Corrected TTLS vs same-objective A2.**

  | Pair | Genuine lexical | New ZH | ΔPIER [95%] |
  |---|---|---|---|
  | T1 vs B2 | 4 vs 15 | 0 vs 26 | +.024 [−.020, +.077] |
  | T2 vs T2A | 0 vs 15 | 0 vs 3 | +.032 [−.007, +.083] |
  | T4 vs T3 | 1 vs 1 | 25 vs 30 | — |
  | T6 vs T5 | 1 vs 1 | 25 vs 32 | — |

  TTLS is safer but delivers much less benefit, and A2+P nearly closes the safety gap.
* **Corrected TTLS vs AUTO.** AUTO is better on English: T1 PIER +.070 [.000, +.148]. TTLS is better on Mandarin:
  ZH-CER −.018 [−.037, −.004]. The MER difference is uncertain.
* **TTLS vs direct acoustic substitution.**
  * T4 and T6 are normalized-transcript identical to ACSUB on 99/100 rows; tokens are identical on 98/100.
  * All corpus metrics and correction counts are identical.
  * Adaptation adds no recognition value over substituting the candidate.
* **Compute.**
  * Whole job: 366 s. Phase A census 140 s, phase B 225 s, setup 12.6 s.
  * T1 adaptation: 9.85 s per 100 utterances (R1: 9.45 s), 200 backward passes, 300 loss evaluations, 1,280 trainable
    scalars.
  * T1 decoding: 55.3 s per 100 utterances.
  * T2 adaptation: 10.7 s per 100 utterances.
  * T4 and T6: 12 backward passes each over the 6 active rows.
  * Peak GPU memory ≤ 3.87 GB.
  * Adaptation cost is negligible compared with decoding.

## 7. Decision

The frozen rule requires, for some corrected arm:
* (a) ≥ 8 genuine-lexical POI corrections in ≥ 4 dialogues;
* (b) R1 safety bounds;
* (c) genuine lexical and new-ZH at least as good as the matched A2 arm.

Result:
* T1 has 4 genuine lexical corrections in 1 dialogue, which fails (a) and (c).
* T2 has 0; T4 and T6 have 1 each.
* No arm qualifies, and every gate passed, so the label is **`TTLS_R1R_VALID_BUT_INSUFFICIENT`**.
* For continuity, the R1 frozen rule applied to the corrected arms would still read `TTLS_R1_MIXED` via T1, based on its
  broad "substitution" class. As the spec requires, that rule is descriptive only.

**Recommendation: close this TTLS configuration.**
* The repaired, causally clean result shows TTLS-CE is a low-harm but low-yield edit, dominated by EOS, deletion and
  boundary repairs.
* TTLS-AC is equivalent to direct substitution.
* Preservation has nothing to protect in TTLS and only costs benefit.
* The confounds remain (global mask, no step-0 access, unmatched budgets), but the repair does not change the
  conclusion.

Do not start TTLS Round 2. Any prospective A2+P study needs separate human approval and a separately frozen,
exposure-safe protocol; it has **not** been started.

## 8. Limitations

* The 100 utterances are already exposed and repeatedly reused; there are only 20 dialogue clusters. Intervals are
  descriptive and unadjusted.
* Genuine corrections are extremely concentrated:
  * T1's 4 are on one utterance;
  * A2's 15 are on one utterance;
  * AC's 1 is on one utterance.
* The lexical split is a frozen surface rule over the canonical POI categories. U0086_S0_185 is a case where B0
  translated an English phrase into Mandarin, so "wrong-language" here means recovering a code-switch.
* The repaired kernel differs from the historical one for nonzero edits in BF16 rounding. That is why T1 differs on two
  rows; nonzero results are not bitwise comparable to R1.
* The comparison is not opportunity-matched (§5.4), and no dose- or correction-matched Pareto analysis was run.
* Not rerun, by design: B1, B2, T2A, T3, T5, CD and ACSUB, reused from the sealed R1 outputs (their validity is
  unaffected by the TTLS kernel). Z0H is reused from the audit census.

## 9. Files and reproduction

* Spec and config: `docs/inference_cf/TTLS_R1R_SPEC.md`, `configs/inference_cf/ttls_r1r.json`.
* Code:
  * `src/csasr/inference_cf/ttls_r1r.py`
  * `experiments/inference_cf_ttls_r1r.py`
  * `experiments/inference_cf_ttls_r1r_evaluate.py`
  * `slurm/inference_cf_ttls_r1r.sbatch`
  * `tests/test_ttls_r1r.py` (18 tests)
* Outputs under `results/inference_cf/ttls_r1r/`:
  * `run1/manifest.json` and `run1/runtime.json`;
  * `run1/integrity/000–099.json`: phase A zero census, with historical-kernel contrast probes;
  * `run1/rows/000–099.json`: per-utterance predictions, loss / gradient / z-norm trajectories, z, displacement, probes,
    gates and timing;
  * `run1/slurm-58391.{out,err}`;
  * `output_seal.json`;
  * `evaluation.json`: all tables, transition events, bootstrap, subgroups and changed utterances.

```bash
cd /home/tungnx/cs-asr-steer-inf && git checkout feature/ttls-r1r-repair
export PYTHONPATH=src:. LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
$PY -m pytest -q tests/test_ttls_r1r.py tests/test_ttls_r1.py tests/test_ttls_independent_audit.py   # 42 passed (CPU)
# GPU re-execution into a NEW directory (never overwrite run1):
OUT=results/inference_cf/ttls_r1r/replay; $PY experiments/inference_cf_ttls_r1r.py manifest --out $OUT   # after committing
sbatch --output=$OUT/slurm-%j.out --error=$OUT/slurm-%j.err --export=ALL,OUT=$OUT slurm/inference_cf_ttls_r1r.sbatch
# Post-seal scoring of the sealed run (requires committed seals; writes a new file):
$PY experiments/inference_cf_ttls_r1r_evaluate.py --run results/inference_cf/ttls_r1r/run1 --out <new path>.json
```

The original run used:

```bash
OUT=results/inference_cf/ttls_r1r/run1
python experiments/inference_cf_ttls_r1r.py manifest --out $OUT      # at 38723a01, committed as 77e68fa5
sbatch --output=$OUT/slurm-%j.out --error=$OUT/slurm-%j.err --export=ALL,OUT=$OUT slurm/inference_cf_ttls_r1r.sbatch   # Slurm 58391
python experiments/inference_cf_ttls_r1r.py seal --run $OUT          # committed as caa4f3c1 before scoring
```

The run guard refuses any re-execution unless the manifest commit is an ancestor of HEAD and only the run manifest has
changed since. A replay therefore needs its own committed manifest.
