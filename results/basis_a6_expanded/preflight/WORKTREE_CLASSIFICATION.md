# BASIS-A6 Expanded Preflight — Worktree Classification

Recorded before further repairs. No reset, clean, discard, or automatic stash was performed.

- Repository: `/home/tungnx/cs-asr-steer`
- HEAD at inspection: `a05a7d7879246e482b9e3ce39df06c7e97f164c3`
- Branch: `learned-expansion`
- Existing worktree was dirty before this repair turn.

## A — BASIS-A6 implementation and preflight work

The following paths are in scope for the A6 implementation/preflight commit:

- `.gitignore` — A6 external ASCEND cache/data-byte exclusions.
- `scripts/download_ascend.py`
- `scripts/freeze_ascend_manifests.py`
- `scripts/basis_a6_*.py`
- `src/csasr/basis_a6/`
- `slurm/basis_a6_gpu_acceptance.sbatch`
- `tests/test_basis_a6_expanded.py`
- `results/basis_a6_expanded/`
- `data/external/ASCEND/ASCEND_DOWNLOAD_PROVENANCE.json`
- `data/external/ASCEND/DO_NOT_USE_TEST.marker`

Dataset bytes and model/cache directories remain ignored and are not commit candidates.

## B — Unrelated user work

These paths are outside the A6 repair scope and must remain unstaged:

- `configs/expansion/`
- `docs/current/AGA_REIMPLEMENTATION_AUDIT.md`
- `docs/current/TRAINING_EXPANSION_IMPLEMENTATION.md`
- `docs/reports/`
- `src/csasr/steering/expansion/`
- `tests/test_training_expansion.py`
- `slurm-51490.out`
- `results/basis_a5_unique_shared/`
- `results/basis_frozen_layer_atlas/`
- `results/learned_expansion/`

## C — Ambiguous or overlapping files

The following files contain both pre-existing user work and A6 additions. They must be
reviewed hunk-by-hunk and never staged wholesale:

- `docs/current/CODE_MAP.md`
- `docs/current/METHOD_CONTRACT.md`
- `docs/current/STATUS.md`

The top-level `data/` status also contains ignored ASCEND dataset/cache bytes plus the A6
provenance/marker listed under A. Only the explicit provenance/marker files are candidates.

## Protection decision

The A6 and unrelated changes are separable by path except for the three documentation files
listed under C. Those files will be preserved and selectively staged only when the A6 hunks
are unambiguous. Unrelated paths will not be staged or committed.
