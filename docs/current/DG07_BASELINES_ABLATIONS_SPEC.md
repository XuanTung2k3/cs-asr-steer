# DG-07A — Learned Baselines and Core Ablations

**Status: FROZEN PRE-RUN (2026-09-08).** This specification freezes the
development comparison before any DG-07 result is inspected. DG-07A performs
implementation and launch preparation only; DG-07B is the later Slurm run.
Authority remains v6 plus `METHOD_CONTRACT.md` and the completed DG-00–DG-06
specifications.

## 1. Frozen proposed reference

`M* = DG-06 D1`, selected by the predeclared utility-first rule. It is the
damage-aware adaptive controller checkpoint
`results/dg06/d1/selected_checkpoint.pt`, hash
`sha256:b9c45a1e279a5b5acf84c45adff6f48a2eae67f0e1062294b9211c7b77fef241`.

| Input | Frozen value |
|---|---|
| Backbone | Whisper-large-v3, local frozen revision from DG-06 |
| Site | decoder post-cross-attention residual, pre-FFN, DG-02 hook |
| Layer | L24 |
| Basis | DG-03 `V0=[v_local,v_cond]`; local `sha256:2459a635…`, cond `sha256:319951b5…` |
| β | `4.465628877080159` |
| Objective | `L_corr + λ_M L_ret,M`, λM=`1.0`, KL `p0 || pθ` |
| Controller | DG-05/DG-06 fixed-basis trunk, 43,651 trainable parameters |
| M* checkpoint | D1 selected epoch 1; hash above |

The D0 correction-only controller is reused from DG-05 and is not retrained.
DG-04 B1 ρ=0.5, DG-05 D0, DG-06 D1, and the frozen Whisper result are reused
when their D-dev-select population and greedy decoding settings match.

## 2. Scope and data roles

The required new systems are exactly:

1. `LB1_SALSA_EXACT_GLOBAL`
2. `LB2_LORA_MATCHED_BUDGET`
3. `A1_LOCAL_ONLY`
4. `A2_CONDITIONING_ONLY`
5. `A3_FIXED_MIXTURE_GATE`

All train on `loc-train ∪ util-train`, use seed 42, AdamW, LR 5e-4,
weight decay 0, batch 8, accumulation 2, three epochs, gradient clipping 1.0,
and the same per-epoch checkpoint schedule as M*. Checkpoint selection and all
scientific comparison use only `D-dev-select`, free decoding, greedy,
temperature 0, beam 1. `D-dev-confirm` and `D-test` are forbidden.
The frozen contributing pool is 1,346 batches per epoch and 673 optimizer
updates per epoch (2,019 updates across three epochs), matching DG-06 D1.

## 3. Objective fairness

Every required learned system uses the selected D1 objective:

```
L = L_corr(C_E) + 1.0 * L_ret,M(R_M)
```

`C_E` and `R_M` are the frozen DG-05/DG-06 artifacts. `R_E` is loaded and
provenance-checked for shared population accounting but is inactive because M*
is D1; no embedded-retention term is silently added. There is no all-token CE,
anchor loss, gate penalty, or basis refinement. Retention is
`D_KL(p0 || pθ)` from frozen unsteered Whisper under matching teacher-forced
context. Teacher forcing is optimization only; free decoding selects checkpoints.

## 4. Required systems

### LB1 — exact-site SALSA-style global vector

`GlobalVector` trains one raw vector `u ∈ R^1280`, zero-initialized. The exact
DG-02 hook applies the same vector at every eligible non-prefix position at L24,
with `alpha=scale=1` and NormPreserve. It has no gate, mixture, oracle mask, or
reference input. Trainable count: 1,280.

### LB2 — matched-budget LoRA

`ExactLayerQvLoRA` inserts manual LoRA adapters only in decoder layer 24
self-attention `q_proj` and `v_proj`. The backbone remains frozen. The nearest
integer rank to M*'s 43,651 parameters is computed from the actual Q/V
dimensions before results: with 1,280×1,280 Q/V projections,

```
count(r) = 2 * r * (1280 + 1280)
rank = 9, alpha = 9, dropout = 0, trainable count = 46,080
```

No rank sweep or performance-based rank choice is permitted. LoRA intervention
energy is structurally not applicable and is recorded as such, not fabricated.

### A1 — local-only adaptive gate

The DG-05-sized LayerNorm → Linear(1280,32) → GELU trunk predicts only `g_t`.
The direction is frozen `v_local`; no mixture output exists.

### A2 — conditioning-only adaptive gate

The same gate-only trunk predicts `g_t` with frozen direction `v_cond`; no
mixture output exists.

### A3 — fixed-mixture adaptive gate

The same gate-only trunk predicts `g_t` with frozen
`π_fixed=[0.5,0.5]` and
`d_fixed=normalize(V0 π_fixed)`. This is DG-04's frozen B2 mixture and isolates
token-dependent mixture learning from token-dependent gate learning.

## 5. Deferred optional ablation

`A5_REFINED_BASIS` is **DEFERRED / NON-BLOCKING**. DG-06 did not predeclare an
anchor coefficient `λ_A`; DG-07A does not invent one or run a sweep.

## 6. Selection and reporting

For each new method, eligible checkpoints must have a valid canonical result and
positive canonical utility. Selection priority is exactly:

1. maximize canonical utility;
2. PIER gain tie-break;
3. matrix retention tie-break;
4. lower realized energy tie-break.

Every run emits `result_v1` with MER, PIER, embedded WER, matrix CER/WER,
corrections, corruptions, canonical outside harm, both retention populations,
utility, and realized energy when applicable. Efficiency provenance records
trainable parameters, backbone percentage, checkpoint size, training wall time,
peak GPU memory, optimizer updates, and descriptive free-decoding time/RTF when
available.

The primary learned comparison is Whisper, DG-04 B1, LB1, LB2, DG-05 D0, and
M*. The ablation comparison is M*, A1, A2, A3, and reused D0. The scientific
questions are adaptive-versus-global, basis-component necessity, token-dependent
mixture value, retention-objective value, and parameter-efficient comparison.

## 7. Provenance and execution

The DG-07 runner stamps resolved config, code/config snapshot, model metadata,
basis hashes, C_E/R_E/R_M pointers and hashes, checkpoint hashes, dataset-role
metadata, seed, and Slurm metadata. All GPU jobs must use `partition=mig`, never
`main`, and no more than two DG-07 jobs may be pending/running. Launch batches:

| Batch | Systems |
|---|---|
| 1 | LB1, LB2 |
| 2 | A1, A2 |
| 3 | A3; A5 only if separately preregistered |

No launcher submits a job during DG-07A.
