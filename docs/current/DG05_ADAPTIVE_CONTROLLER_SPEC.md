# DG-05A / DG-05B — Adaptive Controller Specification and Development Run

**Status:** `DG-05 — COMPLETE / FROZEN` (2026-09-07). DG-05A freezes the implementation choices
below; DG-05B has one accepted development-seed training/free-decoding run. The stage is frozen
as **VALID ADAPTIVE SIGNAL WITH DAMAGE** and remains subordinate to v6,
`METHOD_CONTRACT.md`, and the frozen DG-03/DG-04 artifacts.

## 1. Scope and prerequisite

- **Scope:** Whisper-large-v3 × CS-Dialogue only.
- **Prerequisite:** DG-04 is `COMPLETE / FROZEN`; its reference is B1 `rho=0.5`.
- **Selected site/layer:** decoder post-cross-attention residual, pre-FFN, L24.
- **Backbone:** Whisper-large-v3 frozen; no adapters, LoRA, or SALSA.
- **Basis:** frozen DG-03/DG-04 `steering_basis_v1_L24`; no reconstruction or refitting.

Frozen basis inputs:

| column | tensor hash |
|---|---|
| `v_local` (`conditioning_residualized_local`) | `sha256:2459a63576be545325a68d744bd228854bd5f9e6e09bbcf93e5c6122cd7efa93` |
| `v_cond` (`language_conditioning`) | `sha256:319951b5d28f9e49d159169e1e098f8410ed074154a378d64ba24fd35572f991` |

The controller reads `r = q + u_source` from the canonical DG-02 interface and computes
`LN(r)` internally. It receives no reference, alignment, language label, baseline outcome,
oracle location, or future-token feature at inference.

## 2. Controller architecture

One architecture is fixed; no width or capacity sweep is permitted:

```text
LayerNorm(d_model=1280)
→ Linear(1280, 32)
→ GELU
→ Linear(32, 3)
```

- Output 0: `g_t = sigmoid(logit_0)`, so `g_t ∈ [0,1]`.
- Outputs 1–2: `pi_t = softmax(logits[1:])`, a rank-2 simplex mixture.
- Direction: `d_t = normalize(V^0 pi_t)`, with `V^0=[v_local,v_cond]` fixed.
- Intervention: `NormPreserve(r_t + beta * g_t * d_t)` through the frozen exact-site primitive.
- The basis is a non-trainable buffer; only the bottleneck controller parameters are optimized.

## 3. Strength and optimization

- **Fixed beta:** `beta = 4.465628877080159`, the nominal beta of the frozen DG-04 reference
  B1 `rho=0.5` (`rho * s_L24`), with no beta search and no learned beta in DG-05A.
- **Optimizer:** AdamW, learning rate `5e-4`, weight decay `0.0`, gradient clipping `1.0`.
- **Budget:** 3 epochs, batch size 8, gradient accumulation 2, seed 42.
- **Trainable parameters:** controller LayerNorm and two Linear layers only. Whisper parameters,
  basis tensors, and all direction buffers must have `requires_grad=False`.
- **Objective:** correction-only cross-entropy; no KL retention, gate penalty, anchor loss, or
  basis refinement. Those are DG-06 decisions.

## 4. Training data and correction set

- **Training pool:** existing `loc-train ∪ util-train` dialogue-v2 roles, with physical assignments
  unchanged. No D-dev-select, D-dev-confirm, or D-test rows are used for gradient training.
- **`C_E`:** an explicit, versioned `dg05_correction_set_v1` artifact keyed by utterance and full
  decoder-token position. Each listed position must be baseline-wrong and embedded-language under
  the canonical baseline/alignment accounting. The runner validates this artifact against the
  example's embedded-language token labels and rejects mismatches or an empty set.
- The runner refuses to fall back to all-token CE. Teacher forcing supplies optimization context;
  labels are never controller inputs.
- The correction artifact must record its source baseline, roles, utterance count, position count,
  exclusions, and source/config/data fingerprints before training.

## 5. Checkpoints and selection

The runner saves each epoch checkpoint with controller state, architecture, basis hashes, beta,
train-target count, config hash, and git commit. Checkpoint selection is not based on teacher-forced
loss alone. After checkpoints are evaluated by deterministic **free decoding** on `D-dev-select`,
the frozen rule is:

1. require positive canonical correction–damage utility and valid canonical outside harm;
2. maximize utility;
3. tie-break by PIER gain;
4. tie-break by lower realized intervention energy.

The eventual comparison is B0, the frozen DG-04 B1 `rho=0.5` reference, and the adaptive
controller at this fixed beta. This stage does not rerun the DG-04 grid. Teacher-forced loss is
optimization evidence only; transcript claims require free decoding.

## 6. Focused implementation guards

The canonical action path must verify:

- layer 24 and both frozen basis hashes;
- `g` bounds, simplex `pi`, unit `d_t`, and NormPreserve;
- beta zero is a no-op and the frozen prefix is untouched;
- the basis and Whisper backbone receive no gradients while the controller does;
- only `C_E` contributes to the loss;
- the runtime input contract contains only `LN(r)`;
- checkpoint serialization is deterministic and provenance-bearing.

## 7. Slurm plan (prepared, not submitted in DG-05A)

- Launcher: `sbatch/cs_asr_dg05_adaptive_controller.sh`.
- Partition: **`mig` only**, never `main`.
- Maximum concurrent GPU jobs from DG-05: 2.
- DG-05B jobs: 50489 (`FAILED`, runner prefix-scope bug), 50505 (`FAILED`, empty per-batch C_E
  guard), 50506 (`FAILED`, invalid token-position artifact guard), and 50507 (`COMPLETED`). All
  jobs used `partition=mig` and the prepared H100 3g/40GB resource; no `main` job was submitted.

## 8. Artifacts

- Controller: `src/csasr/steering/controller.py`.
- Correction-only helpers: `src/csasr/steering/dg05_training.py`.
- Exact-site training runner: `experiments/dg05_adaptive_controller.py`.
- Frozen config: `configs/dg05_adaptive_controller.yaml`.
- Prepared launcher: `sbatch/cs_asr_dg05_adaptive_controller.sh`.
- Development run output root: `results/dg05/controller/`.
- Core implementation commit: `f282d3b`; parameter-provenance follow-up: `165e12b`.

## 9. DG-05B development result (seed 42; not frozen)

Job 50507 trained on `loc-train ∪ util-train` with the fixed correction-only objective. The
training-role frozen-baseline decode produced `results/dg05/correction_set_v1.json` (4,064
utterances, 51,227 token positions; artifact hash
`sha256:813604876dfb68eb7fc2e8f1864c46f7c57ef25ed4f3c700798c0f88d199a430`). Three epochs were
evaluated by greedy/temperature-0/beam-1 free decoding on the same 300-utterance D-dev-select
population as DG-04. The frozen rule selected epoch 3; selected checkpoint hash is
`sha256:6d9390dee19097110bbe3f9ed6707ee72e83e93722cf448bd075c8c3f40dfbcc`.

The complete audit bundle is under `results/dg05/controller/`: manifest, training history,
epoch checkpoints, per-checkpoint `result_v1` evaluations, `selection.json`, `summary.json`, and
the copied `selected_checkpoint.pt`. The manifest records Whisper/basis freezing, exact-site
use, job/partition metadata, hashes, and the absence of retention/anchor/gate losses.

Selected epoch-3 A2 diagnostics are: utility +91 (135 corrections, 44 corruptions), PIER gain
+0.04012, MER gain −0.01919, embedded WER gain +0.03219, matrix CER gain −0.02745, canonical
outside harm 1,142, embedded retention 0.9634, matrix retention 0.9040, total edit energy
63,926.3 (mean 3.398), and 18,812 realized edits. Gate mean/median are 0.8233/0.9698 (q01
0.0063, q10 0.2775, q25 0.8158, q75 0.9914, q90 0.9960, q99 0.9985; near-0 4.93%, near-1
57.59%). Mean mixture weights are [0.9397, 0.0603], mixture variance [0.0379, 0.0379], and
sampled mean pairwise direction cosine is 0.9119. These are descriptive diagnostics; no
sparsity or new coverage definition is claimed.

The epoch-3 point improves the frozen DG-04 B1 `rho=0.5` frontier point (38 net corrections,
97 corruptions, PIER gain +0.01675, total energy 82,007.1, matrix retention 0.8930). The
nonzero collateral damage is a `PRESENT` damage signal and motivates DG-06 retention losses; it
is not a DG-05 software failure. DG-05B did not read D-dev-confirm or D-test and did not run
SALSA/LoRA.
