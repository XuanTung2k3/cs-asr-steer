# Repository cleanup plan — proposal only (nothing executed)

**Status: PROPOSAL.** No file, branch, worktree or history was deleted, moved or rewritten.

**Policy.** `AGENTS.md` invariant 3 forbids deleting or renaming `results/`, `out/` and docs. Every REMOVE, ARCHIVE or
move below therefore needs an explicitly approved, written policy exception. This plan is the input to that decision.

**Machine-readable form.** [cleanup_manifest.json](cleanup_manifest.json) has 3,819 entries. Each records path, bytes,
sha256 (files), reference counts (docs), class, proposed action and restoration source.

## 1. Storage inventory (measured 2026-10-11)

| Location | Size | Composition |
|---|---:|---|
| `/home` filesystem | 10 TB, **95% used**, ~591 GB free (shared) | The pressure point |
| This worktree `cs-asr-steer-inf` | 18 GB | 3.07 GB git-tracked (results 3.05 GB, of which `.npz` 2.34 GB) · **15.30 GB git-ignored raw tensors** · 12 MB caches |
| — ignored `results/inference_cf/dir_sprint0/run1/{bank,capture,construct,phones,pulses}` | 10.49 GB | 1,801 `.npz` |
| — ignored `results/inference_cf/srd2_g0/run1/{gate,pulses}` | 4.64 GB | 1,200 `.npz` |
| — ignored `p2dir` / `p2sel` / `p2sel_xa` / `p2sel_lac` rows and arrays | 0.17 GB | 420 `.npz`, no external copy |
| — ignored reference-derived `evaluation_units.json` / `evaluation_panel.json` | 14 MB | Intentionally out of git |
| Worktrees `cs-asr-steer-a2p-dev200-audit`, `-ttls-audit`, `-ttls-r1r-audit` | 9.22 GB total | Plain checkouts (tracked files only + `__pycache__`); clean; HEAD == origin |
| Main checkout `cs-asr-steer` (branch `learned-expansion`) | 9.1 GB | **Uncommitted changes, no remote branch.** Includes 2.5 GB ignored `data/external/ASCEND` (transfer dataset, not inspected) |
| `.claude/worktrees/agent-*` (2) | 4 MB | Agent worktrees; state not inspected |
| Git object store (shared) | 2.3 GB | 1.55 GiB loose + 0.75 GiB packed. Largest history blobs are `.npz` logits under `st_prompt_r1`, `p2a2_mech0`, `r0`, `st_loc0` (0.6/0.4/0.38/0.38 GB) |
| `/mnt/data/tungnx/cs-asr-steer` (19 TB free) | 19 GB | `archives/dir_sprint0` 9.8 GB + `archives/srd2_g0` 4.4 GB (content-addressed by sha256), models 1.8 GB, providers 1.2 GB, legacy `artifacts*` ≈1.7 GB |

Largest tracked result directories: `st_prompt_r1` 0.63 GB, `p2a2_mech0` 0.44, `r0` 0.40, `st_loc0` 0.39,
`src_cf0_pilot` 0.25, `dir_sprint0` 0.12, `s1` 0.10 GB.

## 2. Key verified finding

All 3,001 git-ignored DIR-SPRINT0/SRD2-G0 tensors in this worktree (**15.12 GB**) are **byte-identical** to the
content-addressed archive on `/mnt/data`. Each local file's sha256 was computed and found as
`archives/<run>/run1/sha256/<sha256>.npz`. The runners and auditors already address that archive by hash:
- `experiments/inference_cf_dir_sprint0.py:56`
- `experiments/inference_cf_srd2_g0.py:51`
- the two `*_audit.py` files

The only ignored files in those runs *not* in the archive are:
- two reference-derived `evaluation_units.json` files (KEEP);
- one `.pyc` (REMOVE).

## 3. Candidates

### REMOVE — verified duplicates or disposable (requires policy exception and approval)

| Item | Bytes | Evidence | Restoration |
|---|---:|---|---|
| 1,801 `dir_sprint0/run1/**.npz` + 1,200 `srd2_g0/run1/**.npz` (local copies; manifest total also includes one 144 kB `.pyc`) | 15,120,767,369 | sha256 ∈ archive; manifest lists each pair | `cp archives/<run>/run1/sha256/<sha>.npz <path>`, re-verify sha256 |
| Worktrees `cs-asr-steer-a2p-dev200-audit`, `-ttls-audit`, `-ttls-r1r-audit` | 9,223,122,066 | `git status` clean; HEAD = `origin/<branch>` (7c8401e7 / f99f176b / 146a5d0e); ignored content is `__pycache__` only | `git worktree add <path> <branch>` |
| `__pycache__/` (13 dirs) and `.pytest_cache/` in this worktree; one `.pyc` under `srd2_g0/run1/audit_r1` | ≈12,000,000 | Interpreter/test caches | Regenerated automatically |

**Estimated recoverable on `/home`: ≈ 24.4 GB.** That is 15.12 + 9.22 + 0.01. No unique scientific byte is lost: every
removed byte has a verified copy in the archive or on origin.

### ARCHIVE — unique, needed for reproducibility, not daily work

| Item | Bytes | Proposal |
|---|---:|---|
| 420 ignored `.npz` in `p2dir/exp1_run1/rows`, `p2dir/folds_run1/arrays`, `p2sel/s1_run1/rows`, `p2sel_xa/xa0_run1/rows`, `p2sel_lac/lac0_run1/rows` | 169,713,707 | Copy to `/mnt/data/tungnx/cs-asr-steer/archives/<run>/sha256/`, verify, then delete locally. Low priority: small gain, and these are the only copies. |
| Tracked `.npz` logits in git (2.34 GB working tree) | — | **Keep in git.** Deleting from the tree saves nothing in `.git` and breaks replay. Future studies should write raw tensors straight to the archive, as DIR-SPRINT0/SRD2 did. |

### CONSOLIDATE — navigability, not space (≈1 MB)

90 documents. They stay in place because many paths are pinned by configs, tests and sealed manifests (reference counts
are in the manifest).

| Group | Count | Proposal |
|---|---:|---|
| Per-study process docs in `docs/inference_cf/`: `*_CODEX_DESIGN`, `*_CLAUDE_DESIGN`, `*_HANDOFF`, `*_PRE_RUN_AUDIT*`, intermediate `*_E0/T0/XA0/LAC0/EXP1/_A_REPORT`, `*ATTEMPT*`/`*INVALID*`/`*ABORT*`, compute-feasibility, implementation notes | 63 | Index in one `docs/inference_cf/LEDGER.md` (one row per study: spec, freeze, manifest, report, audit, label). Optionally move only the unreferenced ones (e.g. `P2_R_RUN1_ABORT.md`, `P2_R_RUN2_INVALID.md`, `P2_SEL_XA_XA0_REPORT.md`, `SRD2_G0_DESIGN.md`) to `docs/archive/inference_cf/`. |
| Legacy narratives `docs/*.md` (E1–E5, V2R3, RUNBOOK, results logs), root `FEASIBILITY.md`, `INFERENCE_STEERING_IMPLEMENTATION_PLAN.md`, `P0_INDEPENDENT_AUDIT.md`, `docs/fix_gateA/`, `docs/skills/` | 27 | Move the 0-reference items to `docs/archive/legacy/`. Leave referenced ones and list them in the ledger. |
| `docs/current/STATUS.md` (986 lines of stacked "previous headers") | 1 | Trim to the current state plus a link to `LEDGER.md`. This needs human approval, because STATUS is authoritative. |

### KEEP

- **Authority:** `AGENTS.md`, `CLAUDE.md`, `README.md`, `docs/current/*` (contract chain), and the v6 proposal (v5 too,
  as a referenced historical record).
- **Frozen inputs:** every `*_SPEC.md`, `*_PANEL.json`, `*_FREEZE.json`, `*_POPULATION.json` and `*_FIREWALL.md`.
- **Results and audits:** terminal reports and independent/post-run audits, and all tracked results (sealed evidence
  addressed by hashes).
- **Reference-derived ignored JSON.**
- **The main checkout** — dirty and unpushed; untouched.
- **Agent worktrees** — not inspected.
- **`/mnt/data` archives and models.**

### Out of scope (REVIEW separately)

`/mnt/data/tungnx/cs-asr-steer/artifacts*` (≈1.7 GB legacy, including `*_cancelled_*` and `artifacts_smoke` 0.8 GB).
That filesystem has 19 TB free, so there is no pressure. These were not hash-checked.

Git history: no rewrite. An optional, non-destructive `git gc --prune=never` repack of the 1.55 GiB of loose objects is
possible after approval. Expected savings are small and unmeasured, since most blobs are compressed `.npz`.

## 4. Proposed compact active documentation structure

```text
AGENTS.md · CLAUDE.md · README.md
docs/current/            METHOD_CONTRACT · CODE_MAP · STATUS (current only) · EXPERIMENT_MATRIX · DATA_EXPOSURE(+ADDENDA) · DG/BASIS specs
docs/proposal_arr/       v6 (active) · v5 (historical)
docs/inference_cf/       LEDGER.md (new index) · per study: SPEC · PANEL/FREEZE/POPULATION · REPORT · INDEPENDENT_AUDIT
docs/research_review/    STEERING_FORENSIC_REPORT · NEXT_STEERING_DIRECTION · REPO_CLEANUP_PLAN · cleanup_manifest.json
docs/archive/            inference_cf/ (unreferenced process docs) · legacy/ (pre-DG narratives) — only after approval
```

Going forward, each study should produce one spec, one report and one independent audit. Design notes and handoffs go
into appendices of the spec. Raw tensors go to the hash-addressed archive, not git.

## 5. Safety and restoration procedure (for after approval)

1. **Approve a policy exception.** Record a one-line exception in `AGENTS.md` (human-written), scoped to "verified
   duplicates and clean, pushed worktrees listed in `docs/research_review/cleanup_manifest.json`".
2. **Re-verify immediately before acting.** For each REMOVE entry of kind `git_ignored_result`:
   - recompute the local sha256;
   - recompute the archive object's sha256;
   - require both to equal `entry.sha256`;
   - abort the whole step on any mismatch.
3. **Remove the duplicates.** Delete only the verified local duplicates. Log each path and hash to a dated JSON under
   `/mnt/data/tungnx/cs-asr-steer/archives/cleanup_logs/`.
4. **Remove the worktrees.** For each audit worktree:
   - re-check `git status --porcelain` is empty and HEAD == `origin/<branch>`;
   - run `git worktree remove <path>` (no `--force`);
   - the branch itself stays.
5. **Clear caches.** Delete `__pycache__` and `.pytest_cache` (no approval subtleties).
6. **Archive (optional).** Copy, verify, then delete locally.
7. **Consolidate.** Create `LEDGER.md` first. Move unreferenced docs only after the reference counts in the manifest
   are recomputed at that commit. Never move a referenced or hashed path.
8. **Check after.**
   - The focused test suite passes (with `LD_LIBRARY_PATH` set).
   - `experiments/inference_cf_dir_sprint0_audit.py` and `inference_cf_srd2_g0_audit.py` can still resolve every
     archived object by hash, checked by a dry-run existence check.

**Restoration.** Every removal is reversible:
- duplicates: by copying from the content-addressed archive;
- worktrees: by `git worktree add` from origin;
- caches: automatically.

Nothing in this plan touches git history, branches, the main checkout, datasets or unique evidence.
