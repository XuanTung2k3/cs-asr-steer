# A5000 E1 reconstruction implementation

Status: implementation and CPU validation in progress; GPU outcomes will be appended after execution.

E-ReFT reuses the published LoReFTIntervention: orthonormal seeded QR R, PyTorch Householder orthogonal
parameterization, W=R and b=0. Forward delta=((W-R)h+b)R is evaluated in FP32 and cast back to BF16.
The encoder adapter hooks the complete Whisper encoder output AFTER final layer_norm, never the last
encoder block and never a decoder site. The cached route computes the frozen encoder without gradients,
then applies the same adapter with gradients enabled before the decoder. Native forward/generate integration
is tested against this route. Duplicate adapter contexts are rejected; masks pass through unchanged.
All 1500 frames are adapted; no undocumented frame masking is introduced.

LRA uses B(Ah+c), A=0,c=0,B=the same seeded orthonormal output basis as E-ReFT. This is exact identity with
nonzero first-step A/c gradients; B gets signal after A/c move. A/B/c count is 2dr+r=5122. Neither a novel
method family nor authenticated historical P2-F1 source: E-ReFT and LRA are closely related affine low-rank
families, with distinct parameterization/optimization dynamics. Single LR equality alone is not equivalence.

Tests cover full-size counts/orthogonality, alpha0, exact no-op, actual post-final-norm site, single application,
native/cached generation, tiny-model gradient isolation, serialization, logical-batch CE, O2 path and O3 limits,
script-proxy ambiguity, D1 NLL decomposition, D2 held-out controls and role leakage rejection.
The active pilot itself imports no LID head and no GoldKL term into training.

Code: reft/a5000/{adapters,data,objectives,diagnostics,runner}.py.
Configuration: configs/reft/a5000_e1.yaml. Artifacts: ~/shared/tung.nx/a5000-e1-pilot-v1.
Full checkpoints include optimizer state, epoch/batch cursor and CPU/CUDA RNG states. Frozen encoder cache
is BF16, keyed by model/processor/manifest/audio hashes; it never caches adapted states. Both arms share it.
