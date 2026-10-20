# A5000 E1 reconstruction implementation

Status: encoder implementations validated on CPU and A5000; matched scientific pilot running.
Frozen active source commit: 61a380ff6d41c1d76005aa75bc1edf501b63366c.
New implementation, not authenticated H100 replication.

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

Full primary CPU suite: 1005 passed, 3 known artifact skips (two model-path tests are separately portable).
Initial focused suite: 230 passed, 2 old model-path skips. New primary-method tests: 19 passed.
The separate follow-up suite adds 18 passing tests, including all nine O1/O2/O3 tiny-model paths and exact
optimizer-boundary resume tests. Remaining private historical allocation check cannot be recreated.

A5000 preflight passed for both encoder methods, including the longest selected target (144 tokens incl EOS):
E-ReFT peak allocated 4,448,610,816 / reserved 5,014,290,432 bytes;
LRA peak allocated 4,452,439,552 / reserved 5,026,873,344 bytes. Both 5122 learned scalars.
Actual exported preflight bytes: E-ReFT 6,576,551 (includes dense orthogonal buffer); LRA 22,424.
Native encoder-hook vs exact frozen-encoder-cache generation tokens match; baseline native Auto matches
frozen-LID cached generation on the preflight utterance. Backbone hash unchanged. No activation checkpointing.

Graceful stop was exercised in E-ReFT LR1e-4 epoch1, resuming at batch21 with optimizer state, target/token
counters and RNG state preserved, no duplicate step keys and unchanged sealed B0 outputs. Actual record:
a5000-e1-pilot-v1/audits/resume_{before,after}.json. This is an internal Codex check, not independent review.
