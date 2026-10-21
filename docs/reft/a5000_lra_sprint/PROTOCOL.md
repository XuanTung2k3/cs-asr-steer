# Gated-LRA exploratory sprint v1

This is a new exploratory follow-up after observing `core7h-20261020-v2` results, not a historical replication or independent confirmation. Existing N2_NO_GO is unchanged. N4/N5 will not resume. Original selected baselines and predictions are read-only inputs; no baseline recipe is retrained.

## Source, data and warm start

The complete tracked source, resolved configuration, parent frozen plan, selected checkpoints and development manifest are hashed at remote prepare. Admission binds the frozen plan. Deployment uses an additive commit-addressed snapshot, preserving the original checkout and earlier evidence.

Only original TRAIN-derived A5000_T2000 and V300 are loaded. Speaker, dialogue, utterance and audio-hash separation are checked using existing role metadata. C500 transcript/audio and official test are excluded. The parent manifest contains holdout identifiers/hashes for integrity inventory only. Parquet selection filters to T/V IDs before materializing references. No gate interface receives references, decoder states, token language labels or POIs.

Warm start: selected CE LRA seed42, rank2, epoch3, LR0.0005, SHA256 `c531a22ab1a0fd00464eb4b39c9e2378078c4fc7fe99d046506efd90b67e742d`. Backbone is frozen Whisper-large-v3, post-final-LayerNorm encoder output, BF16 interface and FP32 adaptation arithmetic. Decoder/scorer are reused unchanged from the parent.

## Arms frozen before sprint decoding

|Arm|GPU|Fit|Allocation|
|---|---:|---|---|
|L0|0|Continue 5122 LRA parameters|Always on|
|L1|1|None|Select alpha from 0,0.5,1,1.5,2 on V300|
|L2|2|289 gate parameters; LRA frozen|Generic temporal gate|
|L3|3|289 gate parameters; LRA frozen|Generic energy-normalized gate|
|L4|4|289 gate parameters; LRA frozen|Geometry-aware energy-normalized gate|
|AUDIT|5|None|Old checkpoint, prediction, scoring and cost audit|

Trainable arms use identical fresh AdamW, LR0.0005, betas(0.9,0.99), eps1e-6, weight_decay0, clip1, seed42, one epoch/2000 examples/125 updates. Microbatch1, effective batch16; CE is sum content+EOS NLL divided by all valid targets in each logical batch. No KL, LID or weighted loss. Training order is default_rng(43). Final checkpoint only; no new architecture/LR/checkpoint search. Gate arms use the same selected warm start, 16 active features, width16 and 289 learned scalars (5411 total adapter+gate). L0 is an equal additional CE-update control, not a new baseline search.

L1 reuses exact verified B0 (alpha0) and selected-LRA (alpha1) predictions and performs three new V300 decodes. This explicitly different, predeclared scalar-selection budget has no fitting updates. Tie-break: CS-MER, overall MER, ZH-CER, declared grid order. Report its full screening cost.

## Features and energy

One-second/50-frame blocks; epsilon0.25, alpha1. Shared features describe state energy, temporal variance/velocity, neighboring contrasts and valid duration. Generic gates have eight further state-only statistics. L4 substitutes eight projected-state/update-norm/alignment/neighbor-update statistics. Every MLP has16 active feature inputs; train-only normalization uses valid T2000 blocks.

An orthonormal basis spans the effective affine update `[BA|Bc]`, obtained by QR(B), then an SVD of the small projected affine map. All projected features are invariant norms/trace statistics, not arbitrary A/B coordinates. Equivalent invertible rank-space transformations and orthogonal basis rotations are tested. Numerical rank threshold is max(1e-10,1e-6*smax); rank0/1/2 supported. Exact invariance is not claimed for arbitrarily ill-conditioned floating-point transformations.

For L3/L4, u=1+0.25*tanh(q), a=u*sqrt(E/D), E=sum_valid ||delta||², D=sum_valid u²||delta||². Energy is conserved before BF16 interface casting; cast drift is recorded separately. Zero energy uses scalar1 (no gate-learning opportunity). No coefficient clipping. Zero-output initialization matches the learned LRA exactly and permits gate gradients after the nonzero warm start.

## Frozen interpretation rule

L4 is a useful exploratory signal only if it improves genuine-CS MER by at least0.005 absolute against each of L0/L1/L2/L3, degrades non-CS MER by at most0.01 against each, adds no loops/nontermination, and allocation std is at least0.001. Missing arms/preservation evidence is inconclusive. A smaller L4–L3 benefit is a negative practical-benefit outcome. No gate architecture expansion follows any sprint outcome. Seed42/V300 exposure prevents confirmatory or novelty claims.

## Runtime, recovery and accounting

Admission requires CPU tests, idle six-A5000 allocation, real CE backward/reload for L0/L2/L3/L4, gated free decoding, old LRA token/scoring parity, frozen weights and projected runtime below55min. Archived gate jobs were1059–1110sec; archived five-alpha scalar was3672sec. These are estimates only; measured admission may reject launch.

Detached tmux owns all jobs. A controller lock and per-GPU locks prevent duplicates. Reports at30min and59min; at55min workers receive a cooperative STOP, checked between CUDA operations outside atomic checkpoint replacement. No SIGKILL or interruption during checkpoint writes. A hung CUDA/filesystem operation cannot be guaranteed to finish by60min without unsafe force; final report must disclose any remaining live process. Checkpoints are saved at optimizer boundaries with optimizer/RNG/order/config identity. Resume is restricted to the original window and refuses a STOP or surviving child. No reboot auto-recovery is claimed.

Actual sprint GPU residence is integrated across active children, separately from preflight. Attributed per-method totals include shared parent LR/epoch screening and warm-start cost and must never be summed as campaign consumption. Full model inference timing excludes audio I/O/preprocessing and is labeled accordingly. Old failed/looping outcomes are retained. PIER and word-segmented WER-CS/non-CS stay unavailable without verified annotations/segmentation.

Exports: combined CSV/Markdown/LaTeX, measured nondominated points, PDF/PNG CS-MER versus parameters/attributed GPUh/non-CS MER, accounting, frozen decision and evidence report. Completed rows only; unfinished arms are identified without invented metrics.
