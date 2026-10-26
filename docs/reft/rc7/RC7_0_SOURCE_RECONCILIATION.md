# RC7-0 — Source reconciliation (Milestone 1)

Date: 2026-10-26. Executed by Claude Code (Opus 5.5) under the RC7-0 mandate. Read-only inventory except for the new
branch/worktree named below. No historical file, snapshot, run directory or published branch was modified.

## 1. Decision

**Canonical RC7 base: commit `b18dd0c6840191f218f2f45d79499f9b80923d4e`** (tree `f11eb5fc19fdf92ac90383afad0152b32097df04`),
the A5000 Gated-LRA sprint snapshot, carried forward on the new branch `research/rc7-0-20261026`.

The A5000 implementation *is* selected as canonical, but no file transfer was required: every A5000 source snapshot
is a clean Git checkout whose tree hash is byte-identical to a commit already present in the local Mac repository.
The local Mac repository therefore already contained the complete A5000 implementation with its original commit
ancestry; nothing was copied by SCP/rsync and no history was synthesized.

Canonicalization is complete for source. Two data/provenance blockers are *not* source differences and are carried
to the evidence and exposure audits (historical H100 role manifests unrecovered; partial historical C500 exposure).

## 2. Three-way inventory

### 2.1 Local Mac — `~/Code/cs-asr-reft`

| Item | Value |
|---|---|
| Checked-out branch / HEAD | `exp/a5000-lra-gate-sprint-v1` / `b18dd0c` |
| Tracked modifications | none |
| Untracked | `presentations/` only (not research source; left untouched) |
| Remote | `origin https://github.com/XuanTung2k3/cs-asr-steer.git` (no embedded credentials) |
| Local branches | `cs-asr-reft` 000b062 (tracks origin), `a5000/e0-e1-pure-ce` 4843372, `exp/a5000-ce-only-adaptive-reft` 3c8c01e, `exp/a5000-energy-campaign-v1` 627e179, `exp/a5000-energy-campaign-multigpu-v2` 2ca3c2d, `exp/a5000-lra-gate-sprint-v1` b18dd0c |
| History shape | linear; every local branch tip is an ancestor of b18dd0c |

### 2.2 Remote A5000 — host `50c00b4c6d3a`, user `ubuntu`

| Location | Branch / HEAD | Tree | State |
|---|---|---|---|
| `/home/ubuntu/tung.nx/cs-asr-reft` (main checkout) | `a5000/e0-e1-pure-ce` / `6568c44` | `aca70d7f…` = local 6568c44 | clean tracked tree; untracked `a5000_experiments/` (11 GB run outputs, not source) |
| `cs-asr-reft-campaign-snapshots/57851ec…` | exp/a5000-energy-campaign-v1 | `0f6ab3be…` = local | clean |
| `…/950bdf6…` | exp/a5000-energy-campaign-v1 | `303ca9c1…` = local | clean |
| `…/a8c1429…` | exp/a5000-energy-campaign-v1 | `997485e6…` = local | clean |
| `…/627e179…` | exp/a5000-energy-campaign-v1 | `2d5ea3b3…` = local | clean |
| `…/2ca3c2d…` (**core7h-20261020-v2 source**) | exp/a5000-energy-campaign-multigpu-v2 | `315899be…` = local | clean, no ignored source |
| `…/e02d2bb…` | exp/a5000-lra-gate-sprint-v1 | `8942f949…` = local | clean |
| `…/b18dd0c…` (**lra-gate-20261021-v2 source**) | exp/a5000-lra-gate-sprint-v1 | `f11eb5fc…` = local | clean, no ignored source |

"Clean" = `git status --porcelain` empty; `--ignored` shows only `__pycache__`/`.pytest_cache`.
Git bundles for every snapshot exist in `/home/ubuntu/tung.nx/a5000-bootstrap/`; the b18dd0c bundle verifies as a
complete history (sha256 `5f8dc6d6…2690`).

Launch-time generated files in run roots (`supervise.py` sha256 `2721bb23…`, `control.sh` sha256 `281c8eed…`)
are wrappers written by `reft/a5000_campaign/cli.py:113` at launch; they are run artifacts, not divergent source.

Environment (A5000, `/home/ubuntu/tung.nx/reft-a5000`): Python 3.13.3, torch 2.7.1+cu126, CUDA 12.6, cuDNN 90501,
numpy 2.2.0, scikit-learn 1.6.1, transformers 4.57.6, driver 535.309.01, 6× RTX A5000 (24 GB).
Backbone: Whisper-large-v3 revision dir `06f233fe…`, model sha256 `a8e94b85…fd95`.

Active jobs at inventory time: none (all six GPUs at 1 MiB / 0 %; no tmux server; only an interactive `nvitop`).

### 2.3 GitHub — `XuanTung2k3/cs-asr-steer` (**PUBLIC** repository)

| Ref | SHA | Relation |
|---|---|---|
| `main` (default) | db365d8 | steering-project history; not touched |
| `cs-asr-reft` | 000b062 | ancestor of b18dd0c; 17 commits behind |
| `research/steering-forensics`, `audit/*`, `feature/*`, `cs-asr-steer-inf` | various | unrelated; not touched |
| `research/rc7-0-20261026` | — | did not exist; created by this phase |

None of the 17 A5000 commits (ada154e … b18dd0c) had been pushed before RC7-0.
A5000 can read GitHub anonymously (`git ls-remote` succeeds) but holds no credentials (`credential.helper` unset,
no `~/.git-credentials`, no SSH key besides `authorized_keys`) → **push from A5000 is unavailable**; the Mac
(osxkeychain credential helper) is the push path. No server authentication was modified.

## 3. What was imported, retained and created

- **Imported from A5000:** nothing by file copy. The A5000 code identity was proven equal to local commits by tree
  hash; the canonical branch starts from that commit, preserving original ancestry.
- **Retained locally:** all local branches unchanged; untracked `presentations/` untouched; the main checkout's
  working tree untouched (RC7-0 work uses the separate worktree `~/Code/cs-asr-reft-rc7-0`).
- **Created:** branch `research/rc7-0-20261026` at b18dd0c plus RC7-0 commits; A5000 test clone
  `/home/ubuntu/tung.nx/rc7-0-work/repo` (cloned from the verified bundle, outside every snapshot and run root).

## 4. Public-repository content screen (before first push)

Pushing the branch publishes the 17 previously unpushed commits (96 files, +18,125 lines). Screened:

- No credentials/tokens/private keys (pattern scan of the full diff).
- No transcripts: the only CJK text in the diff is a 2-character unit-test literal.
- No checkpoints, audio, parquet or binary archives (all changed files are `.py/.md/.sh/.json/.yaml`).
- `configs/reft/a5000_manifests/roles_v1.json` publishes **identifiers and hashes only** for A5000_T394/V106/C150
  (utterance/dialogue/speaker IDs, audio sha256, target-text sha256, durations, unit counts; no text). C150 ⊂ C500
  membership therefore becomes public. This discloses holdout *identity*, not outcomes; it is recorded so the owner
  can judge it. RC7-0 adds no further holdout membership to the repository.
- Configs contain absolute A5000 server paths (`/home/ubuntu/…`). These are not secrets; RC7-0 documents avoid
  adding new ones where a relative/run-ID reference suffices.

## 5. Validation of the canonical base

Baseline CPU suite run on A5000 in the isolated clone at b18dd0c (GPU hidden):

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD:$PWD/src /home/ubuntu/tung.nx/reft-a5000/bin/python -m pytest \
  tests/reft/a5000_lra_sprint tests/reft/a5000_campaign tests/reft/a5000 tests/reft/a5000_followup -q -p no:cacheprovider
→ 115 tests, 0 failures, 0 errors, 0 skipped (16.6 s); junit sha256 95f93b56…69e2
```

The local Mac has no PyTorch; all torch-dependent tests run on A5000.

## 6. Unresolved items (not source differences)

1. Historical H100 role manifests (Train394/Inner106/Full500/Dev150) and `assignment.json` are absent from every
   inspected location (local repo, `~/Downloads`, A5000 home and shared dirs) — see `RC7_0_DATA_EXPOSURE_AUDIT.md`.
2. The H100 encoder-adapter source was never authenticated (stated in `reft/a5000/adapters.py`); A5000 E-ReFT/LRA
   are new reconstructions, not H100 code.
3. The A5000 main checkout remains at 6568c44 by design (runs bind commit-addressed snapshots). Nothing was moved.
