# DG-06 — Damage-Aware Correction + Retention Optimization (frozen specification)

**Status:** SPEC FROZEN before scientific results (see §Provenance for the pre-run commit). This
file freezes every DG-06 training/evaluation choice. It must **not** be changed after DG-06
free-decoding results are observed. DG-06 is subordinate to v6,
`METHOD_CONTRACT.md` (MC §6, §8, §9, §10), and the frozen DG-03/DG-04/DG-05 artifacts.

DG-06 question (MC §8, RQ2):

> Can explicit retention objectives preserve useful embedded-language corrections while reducing
> matrix/embedded collateral damage?

DG-05 concluded **VALID ADAPTIVE SIGNAL WITH DAMAGE**: the correction-only adaptive controller
improves the DG-04 frozen correction–damage frontier but still degrades MER / matrix CER and has
material outside harm. DG-06 tests whether the MC §8 KL-to-baseline retention losses reduce that
damage while keeping the corrections.

---

## 1. Frozen scientific inputs (reused exactly; no re-derivation)

| Input | Value / artifact |
|---|---|
| Model | Whisper-large-v3 (`/mnt/data/tungnx/whisper-large-v3`, `openai/whisper-large-v3`), bf16, eager, frozen |
| Corpus | CS-Dialogue (dialogue-v2 roles) |
| Selected layer | **L24** (DG-03 selection; 0-indexed decoder layer) |
| Site | decoder post-cross-attention residual, pre-FFN (DG-02 FROZEN) |
| Repair | `NormPreserve` (DG-02 FROZEN), `norm_preserve=True` |
| Basis | frozen DG-03 `steering_basis_v1_L24`; `V^0 = [v_local, v_cond]`, non-trainable buffer |
| `v_local` hash | `sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93` |
| `v_cond` hash | `sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991` |
| Controller | DG-05A architecture, unchanged: `LayerNorm(1280) → Linear(1280,32) → GELU → Linear(32,3)`; `g_t=sigmoid(logit_0)`; `π_t=softmax(logits[1:])` (rank-2 simplex); `d_t=normalize(V^0 π_t)` |
| β (global strength) | **fixed** `beta = 4.465628877080159` (DG-04 B1 ρ=0.5 nominal β); not trained |
| Bottleneck width | 32 (frozen) |
| Trainable | controller LayerNorm + two Linear layers only; Whisper, basis, all direction buffers `requires_grad=False` |

Prohibited in the DG-06 core (per ticket §2, §16): changing the layer, rebuilding directions,
changing the bottleneck width, changing the mixture parameterization, introducing LoRA/adapters/SALSA,
refining the basis, or adding an anchor loss.

## 2. Training populations (MC §8; v6 §; canonical semantics)

Baseline = frozen backbone, no intervention. All three populations are token-position sets keyed by
utterance id, indexed in the example's full decoder `token_ids` sequence, mapped from
**reference** units by char-overlap with the reference token offsets and filtered by the frozen
`lid_token_labels` (`LID_EN=1`, `LID_ZH=0`, `LID_IGNORE=-100`). Baseline unit correctness uses the
canonical `pier.unit_status` — the same alignment/population semantics already frozen for DG-05 `C_E`.

- **Correction set** `𝒞_E` = baseline-**wrong** embedded-language (EN) positions.
  **Reused verbatim** from the frozen DG-05 artifact `results/dg05/correction_set_v1.json`
  (`dg05_correction_set_v1`, 4,064 utterances / 51,227 positions,
  artifact hash `sha256:813604876dfb68eb7fc2e8f1864c46f7c57ef25ed4f3c700798c0f88d199a430`).
- **Embedded retention set** `ℛ_E` = baseline-**correct** embedded-language (EN) positions.
- **Matrix retention set** `ℛ_M` = baseline-**correct** matrix-language (ZH) positions.

`ℛ_E`/`ℛ_M` are built once by DG-06 from the **same frozen Whisper free-decoding baseline** used by
DG-05 (`results/dg05/controller/training_baseline_hypotheses.json`, `dg05_training_baseline_v1`,
10,768 utterances) and frozen as `results/dg06/retention_set_v1.json` (`dg06_retention_set_v1`)
before any controller gradient. Recorded per population: number of training utterances, population
size, excluded units (no token overlap / language mismatch), and alignment/exclusion counts. The
populations are **not** redefined based on DG-06 outcomes.

## 3. Baseline distributions `p_{0,t}` for retention (MC §8; ticket §4)

Retention KL compares against the **frozen unsteered Whisper** next-token distribution `p_{0,t}`,
computed under **identical teacher-forced context** (the same gold `decoder_input_ids`, the only
difference being that the controller hook is absent). Concretely, per training batch DG-06 runs one
extra forward pass with the frozen backbone and **no** controller hook (`torch.no_grad`) to obtain
the baseline logits, then gathers `p_{0,t}` only at `ℛ`-positions. `p_{0,t}` is never DG-05 adaptive
predictions, steered predictions, or reference one-hot targets.

**Caching decision (recorded):** a full-vocabulary (`V≈51865`) `p_{0,t}` cache over the hundreds of
thousands of retention positions is not a safe efficiency win (hundreds of GB). Baseline logits are
therefore recomputed per batch with the frozen backbone under deterministic teacher forcing; the
recompute is a single extra frozen forward per batch, cheap next to the controller forward+backward.
The baseline-distribution provenance (model id, revision, training-role fingerprint, config hash,
tokenizer, git commit) is stamped into the run manifest.

## 4. Losses (MC §8, exact)

```
𝓛_corr  = (1/|𝒞_E|) Σ_{t∈𝒞_E}  − log p_θ(y_t | x, y_{<t})          # correction-set-only CE
𝓛_ret,E = (1/|ℛ_E|) Σ_{t∈ℛ_E}  D_KL( p_{0,t} ‖ p_{θ,t} )            # KL direction p0 ‖ pθ
𝓛_ret,M = (1/|ℛ_M|) Σ_{t∈ℛ_M}  D_KL( p_{0,t} ‖ p_{θ,t} )            # KL direction p0 ‖ pθ
```

- `𝓛_corr` reuses the frozen DG-05 correction-only CE (`correction_only_loss`); no all-token
  fallback; only `𝒞_E` contributes.
- KL direction is **`p_0 ‖ p_θ`** (not reversed): `Σ_v p_{0,v} (log p_{0,v} − log p_{θ,v})`,
  computed with `log_softmax` for stability. `D_KL(p‖p)=0`. Empty population contributes exactly 0.
- Component losses are logged **raw/unweighted** so relative magnitudes are interpretable.

## 5. λ values (ticket §7; MC §8 leaves them OPEN — predeclared here)

Neither v6 nor `METHOD_CONTRACT.md` freezes numeric λ. DG-06 predeclares the single setting:

> **`λ_M = 1.0`, `λ_E = 1.0`.** `λ_A` (anchor) is inactive (no basis refinement).

One λ setting only. **No λ sweep.** λ is **not** changed after inspecting D-dev-select results.
`λ=1` does not mean equal numerical influence, hence the raw component logging (§4).

## 6. Variants (ticket §8)

| Variant | Objective | Source |
|---|---|---|
| **D0** correction-only | `𝓛 = 𝓛_corr` | **reuse frozen DG-05** epoch-3 controller (not retrained) |
| **D1** + matrix retention | `𝓛 = 𝓛_corr + λ_M 𝓛_ret,M` | DG-06 Job A |
| **D2** full damage-aware | `𝓛 = 𝓛_corr + λ_M 𝓛_ret,M + λ_E 𝓛_ret,E` | DG-06 Job B |

## 7. Initialization fairness (ticket §9)

D1 and D2 start from the **same controller initialization** used for DG-05/D0. The DG-05 procedure
(seed 42; `set_seed(42)` → `load_whisper` → freeze backbone → construct
`FixedBasisAdaptiveController(1280, V^0, 32)`) is replicated verbatim, with no intervening RNG use, so
the reconstructed init is bit-identical to the D0 init. DG-06 saves `initial_controller.pt` and its
sha256; the two jobs must report the **same** init hash. Training is **not** sequential
(D0→D1→D2): D1 and D2 both start from this init and differ **only** in objective.

## 8. Training budget (ticket §10; matches frozen DG-05)

seed 42 · AdamW · LR `5e-4` · weight decay `0.0` · batch size 8 · grad accumulation 2 · **3 epochs**
· grad-clip norm `1.0` · one checkpoint per epoch. No training-budget search; D1/D2 are **not** given
more updates. The contributing batch set for D1/D2 is the full `loc-train ∪ util-train` pool (every
batch carrying any `𝒞_E ∪ ℛ_M ∪ ℛ_E` position), because retention is defined over all
baseline-correct positions — this is a consequence of the objective, not a budget change, and the
per-epoch optimizer-step count is logged. D1 and D2 traverse the identical batch set/order.

## 9. Teacher forcing vs free decoding (MC §9; ticket §12)

Teacher forcing is **optimization only**. Free decoding (greedy, temperature 0, beam 1) is the
**only** basis for checkpoint selection, system comparison, and scientific claims. Variants are never
selected from lowest training loss.

## 10. Checkpoint-selection rule (frozen; identical to DG-05, ticket §13)

Evaluate each epoch checkpoint by free decoding on **`D-dev-select`** (greedy, temperature 0, beam 1),
score with the canonical `result_v1` correction–damage utility, then:

1. require a valid canonical result (valid outside harm) and positive canonical utility;
2. maximize correction–damage utility (`net_corrections`);
3. tie-break by PIER gain;
4. tie-break by higher matrix retention;
5. tie-break by lower realized edit energy.

Applied identically to D1 and D2.

## 11. Data (ticket §14; DATA_EXPOSURE)

- Training: `loc-train ∪ util-train` only.
- Checkpoint selection / development evaluation: `D-dev-select` only.
- **No** `D-dev-confirm`; **no** `D-test`.

## 12. Optional components (frozen decisions)

- **Gate penalty:** `gate penalty deferred unless retention training remains overly broad and the
  active contract authorizes a separate follow-up.` Not in the DG-06 core comparison (ticket §15).
- **Basis refinement / anchor:** `deferred to DG-07 ablation` (ticket §16). Not run.

## 13. Slurm / GPU plan (ticket §20, §21)

`partition=mig` only (never `main`); ≤ 2 DG-06 GPU jobs pending/running; `squeue` checked before each
submission. **Job A** = D1 training + checkpoint free-decoding selection; **Job B** = D2 training +
selection. Each job loads Whisper once, builds/loads `𝒞_E`+`ℛ`, trains 3 epochs, evaluates 3
checkpoints, and writes `result_v1`/provenance. D0 is reused from DG-05.

## 14. Evaluation & reporting (ticket §23–§29)

Per D1/D2 canonical `result_v1`: MER, PIER, embedded-language WER, matrix CER/WER, corrections,
corruptions, correction/corruption rate, canonical outside harm, embedded retention, matrix
retention, canonical utility, edited positions, total/mean edit energy, gate + mixture statistics,
full provenance. Comparison set: A0 (frozen Whisper), F (DG-04 B1 ρ=0.5 reference), D0, D1, D2 — DG-04
artifacts reused where the population/settings match; the DG-04 grid is not rerun.

## 15. Selected-variant & outcome rules (frozen before results; ticket §27, §28)

Eligible D1/D2 must: preserve useful embedded corrections; have positive canonical utility; not
catastrophically collapse matrix retention; be a valid free-decoding result. Among eligible:
(1) highest canonical utility; (2) higher matrix retention if tied; (3) higher embedded retention;
(4) lower realized energy → return `SELECT D1` / `SELECT D2` / `NO DAMAGE-AWARE VARIANT SELECTED`.
Outcome ∈ {`FULL DAMAGE-AWARE SUCCESS`, `MATRIX-RETENTION SUCCESS; EMBEDDED-RETENTION NOT SUPPORTED`,
`NO RETENTION-OBJECTIVE GAIN`}. A negative result is kept — no post-hoc λ/β/LR tuning.

## Artifacts

- Losses / populations: `src/csasr/steering/dg06_losses.py`.
- Runner (extends the DG-05 path): `experiments/dg06_damage_aware.py`.
- Config: `configs/dg06_damage_aware.yaml`.
- Launcher (mig-only, variant arg): `sbatch/cs_asr_dg06_damage_aware.sh`.
- Output roots: `results/dg06/d1/`, `results/dg06/d2/`; retention set `results/dg06/retention_set_v1.json`.
- Focused tests: `tests/test_dg06_damage_aware.py`.

## Provenance (executed run)

- Starting HEAD (DG-05 frozen): `0798355` (`0798355514fd7cb17a8dba0115287500a798067b3`).
- Pre-run commit: `16f4578760eab630098764c954ddd3509dea1f2b`.
- Controller-init hash (D1 == D2, reconstructed DG-05/D0 procedure):
  `sha256:25ebe8e43583bdaed53ab3ab40c3b69b3ff82d0685eaf840bd71e0819ede156f`.
- Config hash: `855c791e691a785bc683dc6c209282fbb0d9ec10e488be6411aed6ac23585fef`.
- Retention-set artifact hash: `sha256:ad2d38a9b83755b078855c8c16ba03594c759a69cd6b697f9d5e76d22337585d`
  (R_E 2,111 utts / 11,748 positions; R_M 7,023 utts / 151,556 positions).
- Trainable parameters: 43,651 (0.0028% of frozen Whisper; identical to DG-05).
- **Job A (D1)** — Slurm `50558`, `partition=mig`, H100 3g.40gb (`worker-mig-3g40gb-0`),
  COMPLETED, 48:57 wall / 2,934 s runtime, peak GPU 5.51 GB. Selected **epoch 1**, checkpoint
  `sha256:b9c45a1e279a5b5acf84c45adff6f48a2eae67f0e1062294b9211c7b77fef241`.
  Result root `results/dg06/d1/`.
- **Job B (D2)** — Slurm `50559`, `partition=mig`, H100 3g.40gb (`worker-mig-3g40gb-0`),
  COMPLETED, 45:41 wall / 2,738 s runtime, peak GPU 5.51 GB. Selected **epoch 1**, checkpoint
  `sha256:bf49dddfa570b2e887c6d6704ed271598ba0856fd363d427c7a698589fb14612`.
  Result root `results/dg06/d2/`.
- **Selected variant: `SELECT D1`** (highest canonical utility among eligible).
- **Outcome: `MATRIX-RETENTION SUCCESS; EMBEDDED-RETENTION NOT SUPPORTED`.**
- Focused tests: `tests/test_dg06_damage_aware.py` 15 + `tests/test_dg05_adaptive_controller.py` 12
  = 27 passed (CPU).
