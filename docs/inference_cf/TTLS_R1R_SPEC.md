# TTLS-R1R — numerical repair, re-execution and revalidation (FROZEN SPEC)

Status: **FROZEN** before execution and before any repaired recognition outcome exists. Machine-readable contract:
`configs/inference_cf/ttls_r1r.json`. Branch `feature/ttls-r1r-repair`, started from the independent-audit commit
`f99f176b60b403da282852957410298d06138ba2` (`audit/ttls-r1-independent`).

This is a strictly scoped **numerical repair** of TTLS-R1, not a new steering-method search. It is an exploratory,
already-exposed development study (100 D-dev-select utterances, 20 dialogues); it is not core-v6, not a confirmation and
uses no D-dev-confirm, D-test, router-calib, P3 or other fresh reserved data.

## 1. The defect being repaired

The independent audit (`TTLS_R1_INDEPENDENT_AUDIT.md`) found that TTLS-R1's zero vector is not a clean start. The
historical DG-02 repair kernel `models.hooks.apply_steering` computes, in BF16,

```python
steered = hidden + delta
steered = steered / (new_norm + EPS) * orig_norm
```

With `delta = 0` and unit gain, divide-then-multiply rounds the residual. On the real model the all-mask zero vector
changes teacher-forced logits by up to 0.25, changes 6/100 free-decoded token sequences, and makes the initial
preservation KL nonzero on 97/100 T2 episodes. R1's in-job check tested only the **empty** mask, which bypasses this
code. TTLS-R1's T1/T2/T4/T6 optimizations therefore started from a perturbed decoder, and their clean-start
attribution is invalid.

## 2. The repair (the only scientific-numerical change)

`src/csasr/inference_cf/ttls_r1r.py`, `NUMERICS = "ratio_first_v1"`:

```python
steered = steered * (orig_norm / new_norm.clamp_min(EPS))      # ratio first, floored denominator
```

* **Zero edit is the clean decoder bitwise.** When `delta == 0`, `steered == hidden` bitwise, so `new_norm == orig_norm`
  bitwise. The ratio is then exactly 1, and the site, the FFN input and every logit are bitwise unchanged. The hook still
  rebuilds `u_source + (steered - site) = u_source + 0`, which is exact.
* **Gradient stays connected.** There is no `z == 0` branch and no detach. At z = 0 the derivative is the projected
  clean gradient Σ_t (I − u_t u_tᵀ) ∂L/∂r̃_t, with u_t = r_t/‖r_t‖. This is nonzero whenever the loss gradient is not
  radial.
* **Nonzero edits.** The edit is still renormalised to ‖r_t‖ and stays differentiable. Its BF16 rounding differs
  slightly from the historical kernel, so outputs for nonzero z are not expected to be bitwise identical to R1.
* **Versioned, not in place.** `apply_steering_ratio_first` and `RatioFirstInterventionHook` are new. The latter is a
  subclass of the canonical `DecoderPostCrossAttnInterventionHook`, with the same capture, absolute positions, forced
  prefix, rebuild and lifecycle; only `_hook` calls the repaired kernel. The following are **byte-unchanged**, so
  TTLS-R1 remains exactly replayable:
  * `models/hooks.py`, `lss/sites.py` and `inference_cf/ttls.py`;
  * every TTLS-R1 pinned source;
  * all R1 outputs and seals.

`ttls_hook_r1r`, `EpisodeR1R` and `displacement_r1r` are the R1 functions with only the hook class swapped.

## 3. What is retained exactly (TTLS-R1 contract)

| Item | Value |
|---|---|
| Population | Sealed R1 plan `plan_sealed.json` (hash `sha256:61ba6a52…`): same 100 ids, same order, same y_B / y_A / masks / UTF-8 flags |
| Model | Whisper-large-v3 local snapshot, BF16, eager, greedy, cap 200 |
| Site | DG-02 decoder L16 post-cross-attention / pre-FFN; forced prefix 4 (content step 0 never edited) |
| Variable | One fresh zero fp32 z ∈ R^1280 per arm per utterance; frozen backbone |
| Optimizer | AdamW lr = E*/(2√1280) = 0.0157374, wd 0, betas .9/.999, eps 1e-8; 2 updates, 3 loss evaluations |
| Budget | Projection ‖z‖ ≤ E* = 1.1260757575454359 after every step |
| CE | Historical A2 loss on the AUTO pseudo-transcript y_A, cB prompt; mask ALL |
| AC | −log p(c* \| y_B[:t*]); frozen θ0-only candidate rule (`<\|en\|>` branch, null audio, τ = log 10); mask {t*} |
| P | λ = 1, mean_{t∈S} KL(p0 ‖ pθ) on the B0 path; frozen stable set S |
| Decoding / evaluation | R1 free decoding; R1 evaluator primitives and safety bounds unchanged |

Candidates (t*, c*, M, accepted, S) are recomputed from θ0 in-job and **must equal the sealed R1 rows**. No teacher,
candidate, mask, learning rate, step count, budget or objective changes.

## 4. Arms

* **Rerun (repaired kernel).** T1 TTLS-CE; T2 TTLS-CE+P; T4 TTLS-AC; T6 TTLS-AC+P.
* **New controls.**
  * B0: clean decode, which must equal sealed S0 on 100/100.
  * Z0_ALL: zero vector, no optimizer, actual CE mask, 100/100.
  * Z0_AC: zero vector at {t*} on every accepted row.
  * Z0H: the historical-kernel zero control. This is the audit census, reused and not rerun.
* **Reused unchanged from sealed R1.** These are unaffected by the defect: B1 AUTO, B2 historical A2, T2A A2-CE+P, T3
  A2-AC, T5 A2-AC+P, ACSUB and CD. The original R1 T1/T2/T4/T6 outputs are kept as `*_R1` for contrast.

## 5. Integrity gates (in the single job; phase A stops the job before any optimization)

**Phase A** covers all 100 rows with no optimizer. Any failed row → job status `stopped_identity`, no TTLS arm runs,
and the label is `TTLS_R1R_INVALID`. Phase A checks:

1. The canonical B0 decode equals the sealed S0 tokens and termination.
2. θ0 candidates and S equal the sealed R1 rows.
3. **Zero-vector free decoding**: Z0_ALL = B0 on 100/100, and Z0_AC = B0 on every accepted row.
4. **Zero-vector teacher-forced identity**: under each actual mask, the logits and the L16 FFN input (a
   `final_layer_norm` pre-hook) are bitwise equal to clean.
5. **Zero initial preservation KL** ≤ 1e-12 (expected exactly 0).

The historical kernel's zero-vector logit and FFN differences are also recorded for contrast; they are not a gate.

**Phase B** covers every TTLS episode. Any failure → job fails → `INVALID`. Phase B checks:

6. A fresh zero z and an empty optimizer at episode start.
7. Step-0 gradient norm is finite and > 0.
8. Initial P ≤ 1e-12 for the P arms.
9. The final edit is consumed: edited-position FFN-input chord > 0 when z ≠ 0.
10. Norm preservation: |‖r̃‖/‖r‖ − 1| ≤ 0.02 at the FFN input.
11. Exactly zero FFN-input displacement outside the mask, on the same teacher-forced history.
12. The budget holds.
13. No site hooks are leaked, and the LN guard still holds θ0.
14. **Post-row clean decode = B0** (exact episodic reset).

At job end, all parameter bytes must be unchanged and the model must hold no gradients. The runner reads only the
sealed plan and the sealed R1 rows' candidate fields (no references); an AST firewall test enforces this.

## 6. Evaluation (post-seal; `experiments/inference_cf_ttls_r1r_evaluate.py`)

References open only after the committed R1R output seal **and** the R1 seal verify. The evaluation reports:

* PIER, MER, EN-WER and ZH-CER;
* POI corrections and corruptions;
* new Mandarin errors and ZH retention;
* outside-POI harm;
* EOS recovery and premature EOS;
* severe truncation and caps;
* dialogue-cluster bootstrap (2,000 draws, seed 240924, descriptive);
* subgroups D12, A88, A3-panel-24 and AC-accepted;
* comparisons against B0, B1 (AUTO), the matched A2 arm, the original R1 arm and ACSUB.

**Frozen lexical split of POI corrections**, using the B0 canonical category at the POI:

| Class | Rule |
|---|---|
| deletion_or_boundary | deletion / boundary_error / insertion_near_poi |
| genuine_wrong_language | wrong_language_substitution / phonetic_transliteration_or_script |
| word_boundary_repair | same_language_substitution / other, and the compact (space-free, lower-case) POI is a substring of, or equal to, the compact B0 surface (e.g. `peoplethey→people`) |
| genuine_same_language | same_language_substitution / other, and not a boundary repair |

`genuine_lexical` = genuine_wrong_language + genuine_same_language.

**Effect decomposition** (descriptive):

1. **Repair.** Z0H vs Z0 (= B0), and each R1 arm vs its R1R arm.
2. **Optimizing z.** R1R arm vs Z0 (= B0).
3. **Preservation.** T2 vs T1, and T6 vs T4.
4. **Step-0 and global-mask opportunity.**
   * For B1, B2 and T2A: how many rows change content step 0, and how many corrections, genuine-lexical corrections
     and new-ZH errors fall on those rows. TTLS structurally cannot change step 0.
   * The CE mask is utterance-global (one vector at every editable step).

TTLS-R1R is therefore still **not** a localized, opportunity-matched actuator comparison.

## 7. Frozen decision

* `TTLS_R1R_INVALID`: any gate fails; any row is missing or failed; B0 is not reproduced; or there is a seal or manifest
  mismatch.
* `TTLS_R1R_VALID_AND_PROMISING`: some corrected arm X ∈ {T1, T2, T4, T6}, with matched A2 arm Y (T1:B2, T2:T2A, T4:T3,
  T6:T5), meets all of:
  * (a) ≥ 8 **genuine_lexical** POI corrections vs B0, in ≥ 4 dialogues;
  * (b) all R1 safety bounds vs B0;
  * (c) genuine_lexical(X) ≥ genuine_lexical(Y), and new-ZH(X) ≤ new-ZH(Y).

  Reduced deletion, EOS or boundary errors alone never qualify.
* `TTLS_R1R_VALID_BUT_INSUFFICIENT`: otherwise. If TTLS remains lexically ineffective, the recommendation is to close
  this TTLS configuration.

The R1 frozen rule applied to the corrected arms, and every AUTO comparison, are descriptive only.

## 8. Compute and process

* **One Slurm job** on `mig`, 1× H100 3g.40gb, ≤ 3 h, with no smoke jobs. R1 ran in 8.8 min.
* Per-row JSON is written atomically and is resumable.
* **Order of operations:**
  1. Commit the sources and spec (this freeze).
  2. Commit and push the resolved manifest.
  3. Submit the job.
  4. Commit the outputs and seal before any reference access.
  5. Run the evaluation.
  6. Write the report.

  No setting may change after repaired outcomes are viewed. No TTLS Round 2 and no A2+P confirmation follow
  automatically.

## 9. Recorded differences from TTLS-R1

1. **Repair kernel** for the TTLS hook: divide-then-multiply → ratio-first (§2). This affects T1/T2/T4/T6 and the zero
   controls only.
2. **Instrumentation**:
   * phase A zero-vector census under the actual masks;
   * teacher-forced logit and FFN-input bitwise probes, plus a historical-kernel contrast probe;
   * zero-KL;
   * per-episode gates 6–14;
   * a post-row clean re-decode.
3. **Scalar logging** uses `float(loss.detach())` instead of `float(loss)`. The value is identical and only the autograd
   warning is removed.
4. **Not rerun**: the LN arms (B2, T2A, T3, T5), AUTO, CD and ACSUB. They do not use the TTLS hook and are reused from
   the sealed R1 outputs.
5. **New evaluator additions**:
   * the frozen lexical split, which replaces R1's broad "genuine substitution" class for the decision;
   * corrected-vs-original, TTLS-vs-ACSUB and step-0 opportunity tables;
   * the R1R label rule.
