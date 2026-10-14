# CS-ASR-ReFT — scope and research contract

**Experiment:** Task-Aligned Frozen Subspace Adaptation for Efficient Code-Switching ASR.
**Branch:** `cs-asr-reft` (started from `main` @ `db365d8f`). **Workspace:** `/home/tungnx/cs-asr-reft`.

## 1. Independence from the historical steering contract

This repository was cloned from `main`. That branch contains the historical steering contract (`AGENTS.md`, `CLAUDE.md`,
`docs/current/`, `docs/proposal_arr/`). For legacy paths, that contract still applies unchanged: `src/csasr/`,
`experiments/`, `steer_sweep/`, `sbatch/`, `configs/` outside `configs/reft/`, `results/`, `out/`, and the legacy docs.

The ReFT experiment is a separate research question, and its contract is this file plus `P0P1_PROTOCOL.md`:

- The steering contract's method invariants do not define this method: the rank-one direction, conditioning-residualized
  Δ⊥, norm-preserving repair, candidate layers {16, 24} and the gate/repair machinery.
- ReFT code lives under `reft/`, `configs/reft/`, `tests/reft/`, `slurm/reft_*.sbatch` and `docs/reft/`.
- The historical frozen results are not reinterpreted, modified or relabeled. ACL-0 artifacts on
  `research/steering-forensics` are read in place and verified against their own seal.
- Data-role separation is kept from the historical contract:
  - `D-test`, `D-dev-confirm` and `router-calib` are never opened.
  - Selection or confirmation claims are never read from exposed `D-dev-select` data.

## 2. Reused code (selective, with provenance)

| Reused piece | Source path @ commit | Where it lives here |
|---|---|---|
| Whisper loader, audio features | `src/csasr/models/whisper.py` @ main `db365d8f` | imported unchanged |
| Canonical normaliser, unit segmentation, language tags | `src/csasr/data/normalize.py`, `src/csasr/data/language_tags.py` @ `db365d8f` (identical at `3467db86`) | imported unchanged |
| `target_text`, `first_divergence`, token bytes | `cs-asr-steer-inf/src/csasr/inference_cf/acl0.py`, `experiments/acl0.py` @ `3467db86` | copied into `reft/text.py` |
| Unit alignment and error categories (`one`, `category`, `compare`, `ratios`) | `cs-asr-steer-inf/experiments/ttls_r1_independent_analysis.py` @ `3467db86` | copied into `reft/text.py` |
| Loop detector | `cs-asr-steer-inf/experiments/acl0_evaluate.py` @ `3467db86` | copied into `reft/text.py` |
| Layer-24 post-cross-attention site semantics | ACL-0 `SiteHook` (`experiments/acl0.py`), DG-02 `src/csasr/lss/sites.py` | re-implemented in `reft/site.py`, verified by tests and the smoke |
| Fillers | `cs-asr-steer-inf/configs/research_review/iae1.json` | copied into `configs/reft/p0p1.yaml` |
| Slurm pattern | `cs-asr-steer-inf/slurm/acl0_{gpu,cpu}.sbatch` | `slurm/reft_*.sbatch` |

Not transplanted:
- historical fixed-steering directions, routing, phonetic scorers (PCV/PACR/IAE);
- legacy experiment drivers;
- ACL-0 editors.

## 3. Data roles

| Role | Use in ReFT | Exposure class |
|---|---|---|
| ACL-0 dev150 (`D-dev-select`, 20 dialogues) | P0 baseline diagnosis only | Exposed (A3-ACL0-DEV150, used for ACL-0 checkpoint selection). Exploratory, never confirmation. |
| `loc-train ∪ util-train` (27 dialogues) | P1 basis construction (≈80%) and internal feasibility audit (≈20%), dialogue-disjoint | Training data. Both partitions are part of the training budget, not external validation. |
| `D-test`, `D-dev-confirm`, `router-calib` | none (never opened) | locked / exposed / needs authorization |

## 4. Claim boundaries (all stages)

**P1 can establish at most three things:**
1. representation separability;
2. the share of gradient energy retained by a subspace;
3. local oracle controllability.

**P1 does not show:**
- deployable ASR improvement or free-decoding sufficiency;
- that a controller can recover words;
- that a model learns where to intervene;
- causal isolation of language interference;
- superiority over PEFT;
- training-compute savings from fewer parameters.

Basis-extraction cost and later controller-optimization cost are reported separately.

## 5. Agent working rules for ReFT paths

- **Compute:** GPU work runs only through `sbatch` (MIG partition, ≤ 3 h per job). CPU tests and audits also run through
  `sbatch`.
- **Temporary files and logs:** set `TMPDIR=/mnt/data/tungnx/tmp/...`. Slurm logs go to `/mnt/data/tungnx/slurm_logs`.
- **Never in Git:** private artifacts (transcripts, sample-level outputs, gradients, bases, manifests with references)
  stay under `/mnt/data/tungnx/cs-asr-reft/`. Only code, tests, configs and sanitized summaries are committed.
- **Freezes and amendments:** every GPU and evaluation command checks a write-once freeze record
  (`reft.provenance freeze`). An amendment needs a timestamped justification and must come before the outcome it could
  affect.
- **Scope:** do not start P2 (controller training) without explicit authorization.
