# CS-ASR-ReFT (branch `cs-asr-reft`)

**Experiment:** Task-Aligned Frozen Subspace Adaptation for Efficient Code-Switching ASR. This sprint covers P0/P1 only:
a feasibility test of whether a frozen rank-4 subspace, extracted from task gradients at Whisper-large-v3 decoder layer
31 (final layer; post-cross-attention, pre-FFN; AMENDMENT L31 — layer 24 was the original choice), has local lexical-correction leverage beyond random directions. No controller is
trained.

**Documents:**
- `docs/reft/SCOPE.md`: scope, independence from the historical steering contract, reused code, data roles and claim
  boundaries.
- `docs/reft/P0P1_PROTOCOL.md`: the frozen protocol and its timestamped amendments.
- `docs/reft/P0P1_RESULT.md`: results and verdicts.
- `docs/reft/CODE_MAP.md`: modules and commands.

**Environment:** `/home/tungnx/miniconda3/envs/acl1` (Python 3.11, torch 2.10.0+cu128, transformers 4.57.6). Run every
step through `slurm/reft_*.sbatch`, not on the login node.

**Artifacts:** All private artifacts live under `/mnt/data/tungnx/cs-asr-reft/` and are never committed. That includes
outputs, references, gradients, bases and sample-level evaluations.
