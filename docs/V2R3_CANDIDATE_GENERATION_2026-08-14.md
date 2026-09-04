# v2r3 candidate generation: execution and results

**Date:** 2026-08-14
**Job:** Slurm 40374, `COMPLETED`, exit 0, wall clock **427 s (7 m 10 s)**
**Namespace:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3`
**Status:** The natural candidate table for the dialogue-atomic roles exists and is
complete. No gate was evaluated, no gate generation was allocated, and no gate is
claimed to pass.

---

## 1. Two amendments, and why they were needed

Both were authorization defects, not code defects, and both were resolved by
amendment rather than by changing code.

### Amendment 1 — destination

The original authorization named
`artifacts_dialogue_v2/candidate_generations/generation_001`, while pre-flight
item 6 required the generation to bind to partition fingerprint
`b6c1af90894ec64c3dacc0b110bfe2da4e8bef3a9881863dc6751d2b61786ecf`. Those are
mutually exclusive: the corrected roles exist only in a *different namespace*,
because `csasr.lss.v2_namespace.configured_paths` pins `role_root` to
`<namespace_root>/manifests/roles`. That pinning is exactly why the Session 2e
republication needed a new root in the first place.

The refusal was verified empirically rather than read off the source:

```text
attempt: role_root=v2r2, candidate root under v2
REFUSED: role_root must be /mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2/manifests/roles
```

Binding a GPU artifact to the superseded fingerprint `87d24cb8…` would have
required rerunning the GPU work to correct it — the precise cost Session 2e was
performed to avoid. The destination was amended to v2r3 (see §2).

### Amendment 2 — post-verify 6 restated

The original post-verify asked for confirmation that a partial generation is never
visible at the destination. `lss_v2_candidates.py` does not provide that property:
it writes directly into the destination root and publishes the completion manifest
last. There is no attempt-private directory and no atomic promote.

Adding one immediately before the first real GPU run would have traded a real risk
(modifying a production execution path minutes before executing it) for a cosmetic
gain. The requirement was restated to match the property the implementation
actually has. See §7 for the attestation.

---

## 2. Why v2r3 exists at all

Job **40373**, submitted against v2r2, refused in 2 seconds:

```text
v2 freeze artifact manifest is not authentic for this execution configuration:
identity_mismatch differs in ['source_sha256']
```

Nothing was written; the destination directory was never created.

`manifest.identity()` binds artifacts to `source_snapshot_hash()`, which covers
`src/`, `configs/`, **and `tests/`** plus repo-root launchers. The v2r2 artifacts
were published at 19:41:13 under snapshot `d2812973efaf98f6…`; six regression tests
were added afterwards, moving the tree to `82c9608403e96a21…`. Every v2r2
artifact — freeze and all seven role manifests — consequently failed the
execution-time identity check.

This was a correct refusal. The artifacts were not produced by the tree present at
execution time, and the guard said so before spending any GPU time.

v2r3 republishes the same allocation under the current tree. **Nothing about the
science changed.**

---

## 3. Pre-flight

Items 1–4 were verified before Amendment 1 and were not re-run afterwards, per
instruction.

| # | Check | Result |
|---|---|---|
| 1 | v2 role manifests authenticate against sidecars | **PASS** — all 7, in both namespaces |
| 2 | v2 config resolves to the v2 root; default resolves to v1 | **PASS** |
| 3 | `generation_001` does not exist | **PASS** |
| 4 | Exposure generation 3 unallocated | **PASS** — ledger holds generations 0, 1, 2 only |
| 5 | Session 2b atomicity applies | **FAILED as stated** → resolved by Amendment 2 |
| 6 | Roles carry `b6c1af90…` | **FAILED for the authorized path** → resolved by Amendment 1 |

### Code freeze

| point | `source_snapshot_hash` |
|---|---|
| H1, recorded before publishing v2r3 | `64346ca6852cdf5b588708cd7e4acc2da5dbaf3eb87737ed63161a726811adca` |
| H2, after publishing roles, freeze, marker | identical |
| H3, immediately before `sbatch` | identical |

The v2r3 config had to be created before H1 was recorded, since creating it
changes `configs/`; the freeze window (publish → completion manifest) is
unaffected. All publishing writes landed under `/mnt/data`. The sbatch launcher
lives outside the repository and is not covered by the snapshot, so re-pointing its
destination could not disturb the freeze. The candidate table's recorded
`identity.source_sha256` is `64346ca6…` — the same tree.

---

## 4. Publication mechanisms differ by design

The role generation and the candidate generation use **different** publication
mechanisms. This is deliberate and consumers must not assume otherwise.

| | Role generation | Candidate generation |
|---|---|---|
| Implementation | `dialogue_roles_build.publish_generation` | `lss_v2_candidates._write_complete` |
| Mechanism | attempt-private directory, completion manifest last, **atomic rename** | direct writes into the destination, completion manifest last |
| Partial state visible at final path? | **No** | **Yes, during the run** |
| Guard against resume/reuse | destination must not exist | `validate_namespace(for_execution=True)` refuses a pre-existing destination |

**Consumer contract:** a candidate generation lacking
`generation_complete.manifest.json` must be treated as **absent** — never as a
partial cache to resume, and never as data.

---

## 5. Command

```bash
sbatch /mnt/data/tungnx/cs-asr-steer/jobs/v2r2_candidates/run_candidates.sbatch
```

Resources: `--partition=main --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=20:00:00`.
The launcher runs exactly one command:

```bash
python -m csasr.experiments.lss_v2_candidates \
    --config lss/l1b_candidates_dialogue_v2r3.yaml --execute
```

It never invokes `cs_asr_lss.sh`, never runs L0 or L1b, and writes no status or
freeze artifact. Node `worker-0`, NVIDIA H100 80GB HBM3, torch 2.10.0+cu128,
git SHA `fdd441a75b0d66f56d8e5efec62f01649f7b9ae9`, started 20:41:59Z, ended
20:49:07Z. Models: `openai/whisper-large-v3` (bfloat16) and the MMS-FA CTC aligner.

---

## 6. Realized counts — post-verify 1

Every role matches the read-only Session 3 figures exactly.

| Role | Dialogues | Expected | Utterances | Expected | Match |
|---|---:|---:|---:|---:|:--:|
| D-construct | 20 | 20 | 300 | 300 | OK |
| D-dev-select | 20 | 20 | 300 | 300 | OK |
| loc-train | 15 | 15 | 225 | 225 | OK |
| util-train | 12 | 12 | 180 | 180 | OK |
| router-calib | 8 | 8 | 120 | 120 | OK |
| D-dev-confirm | 10 | 10 | 150 | 150 | OK |
| **Total** | **85** | **85** | **1,275** | **1,275** | **OK** |

**No deviation.** Utterances per dialogue: min 15, max 15. Total shortfall across
all 85 dialogues: **0**. `D-test` contributes **0 rows** — it was not swept.

---

## 7. Post-verify results

### 2. Alignment quality per family — development roles only

Computed with `csasr.lss.align.autoevidence.family_validity` over `D-construct` +
`D-dev-select` only (600 utterances). No statistic here is computed over
`loc-train`, `util-train`, `router-calib`, or `D-dev-confirm`, although candidates
were generated for them.

| Family | Independence class | Units | Utterance coverage | Invalid rate | Invalid-duration rate | Nonmonotonic rate |
|---|---|---:|---:|---:|---:|---:|
| `existing_ctc` | `ctc_forced_alignment` | 29,500 | 600/600 = 1.0000 | 0.000000 | 0.000000 | 0.000000 |
| `qwen_forced_aligner` | `qwen_forced_alignment` | 29,500 | 600/600 = 1.0000 | 0.032949 | 0.032847 | 0.000000 |
| `whisper_dtw` | `whisper_cross_attention_dtw` | 29,433 | 600/600 = 1.0000 | 0.173920 | 0.000000 | 0.000000 |

`invalid_duration` and `nonmonotonic` keep separate denominators, per the repository
invariant: the nonmonotonic denominator is the duration-valid subset (29,500 /
28,531 / 29,433 respectively). All three families reach full utterance coverage.
`whisper_dtw` carries the highest invalid-unit rate at 17.4%, none of it from
duration failures.

These are raw-unit validity statistics. They are **not** alignment accuracy, and
nothing here compares an aligner to truth.

### 3. Generation manifest contents — one gap

Recorded: sampler seed **303**, stratify field **`dialogue_id`**, **K = 15**
utterances per dialogue, `sample_dialogues: all`, realized per-role dialogue counts
with per-dialogue eligible/selected/shortfall detail, the frozen selection
(`pred_start_offset: 0`, `query_at_token`), configured and active families, and a
full artifact inventory with hashes. The candidate sidecar records all seven parent
v2r3 role manifest hashes plus the freeze, `taint_reasons: []`,
`diagnostic_only: false`.

**Gap:** the partition fingerprint `b6c1af90…` is **not** recorded as a literal
field in `generation_complete.manifest.json` or the candidate sidecar. The binding
is real but indirect — the sidecar names each parent role manifest by sha256, and
those role manifests' own sidecars carry the fingerprint (as does the freeze, twice).
Adding a direct field is a one-line change to `_write_complete`; it was not made,
because the code freeze was in force and modifying a production execution path was
out of scope. Recorded here as an open item.

### 4. v1 artifacts unchanged (repository hasher)

| sha256 | mtime | File |
|---|---|---|
| `29d7018d44536abce7afc80e900aa20b21ae5e1508c16f85562b3ccb86e23ce0` | 2026-08-12 00:59:42 | `candidates_all.parquet` |
| `41e3b48386650e5df596694d5a7b5713fc35f335ddd17b9cbdb85dfe29cde049` | 2026-08-11 03:44:59 | `spec_freeze_v1.json` |
| `7f8527a7005a14ea61834f6ed1d9576ce16a4c4f85559e00418772f94265c6cb` | 2026-08-12 00:47:05 | `role_D-construct.parquet` |
| `775fceb6b43a8965be29cb33c193463c2d46cea10a7131a3e8fdafa286dff82c` | 2026-08-12 00:47:06 | `role_D-dev-confirm.parquet` |
| `d2f8c118c33199a05785c616bcf32648f246b927eef6df55742a06b90d6adb6c` | 2026-08-12 00:47:06 | `role_D-dev-select.parquet` |
| `ce48031adfbec30eb61e2a2b8d58cd58108ce3989172826295d5ff8893aa7143` | 2026-08-12 00:47:05 | `role_loc-train.parquet` |
| `b4fe8e1d0395f81cedf607278443f96470969d8386118c4b716de7925eaf53a7` | 2026-08-12 00:47:06 | `role_router-calib.parquet` |
| `b318330d407e85f558af482cff15758e42b988b3e9dad6e3ac724aeec13f95a3` | 2026-08-12 00:47:06 | `role_util-train.parquet` |
| `38fdb289d37281bf5c96eba3c66633e65f59b8994a61bdda8baaae6a347fe478` | 2026-08-12 00:47:07 | `locked/role_D-test.parquet` |

`candidates_all.parquet` matches the expected `29d7018d…`. **Zero files under
`artifacts_lss` were modified today.**

### 5. Superseded namespaces unchanged

| Namespace | Files | Newest mtime |
|---|---:|---|
| `artifacts_dialogue_v2` | 29 | 2026-08-14 17:00:10 |
| `artifacts_dialogue_v2r2` | 29 | 2026-08-14 19:41:13 |

Both predate this session's first write (20:37Z). Both remain write-once and
unmodified; each is marked by a sibling `*.SUPERSEDED_BY.json` pointer, never by a
file placed inside it.

### 6. Write-once-with-completion-gate attestation

I attest the following, and explicitly **not** attempt-private atomic promotion:

- `validate_namespace(for_execution=True)` refuses a pre-existing destination, so
  no interrupted run can be resumed or reused as a cache;
- `generation_complete.manifest.json` is written only after every configured family
  reports `state: ok` **and** the candidate table re-verifies against its manifest;
- the completion manifest is the last write.

A partial directory **is** visible at the final path during the run. A generation
lacking its completion manifest must be treated as absent.

### 7. No gate generation allocated or exposed

Exposure ledger mtime **1786502490** (matches the expected value),
sha256 `46849f5caed622017b8ff9087f8dbc9c1ea6238870e22e903621d071dc207dfb`.
Generations present: 0, 1, 2. **Generation 3 remains unallocated.**

---

## 8. Artifacts

Namespace `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3`:

| sha256 | Bytes | Path |
|---|---:|---|
| `cdc0ff4e59fe18901c8e1a5e03092486…` | 4,901,994 | `candidate_generations/generation_001/alignments/candidates_all.parquet` |
| `9d3b444aa975d313ff7ae4523ac74efe…` | 2,557,933 | `…/alignments/candidates_existing_ctc.parquet` |
| `81a5c385125b5197e090931b7af5518e…` | 1,435,877 | `…/alignments/candidates_whisper_dtw_pred0.parquet` |
| `60ce95f2e778648c984055e3f344345d…` | 1,047,642 | `…/alignments/candidates_qwen_forced_aligner.parquet` |
| `31801a05181fea676be1b88a40ff2953…` | 61,962 | `…/diagnostics/l1b_aligner_sweep.json` |

Each has a published manifest sidecar; the completion manifest inventories all ten
files with hashes. Candidate table: **178,884 rows**, 38 columns, three families —
`existing_ctc` 59,651 rows, `qwen_forced_aligner` 59,651, `whisper_dtw` 59,582 —
each covering all 1,275 utterances.

Supporting artifacts published this session:

- `manifests/cs_dialogue_with_dialogue_id.parquet` — `f8de76a056a2f552…`, bit-identical to v2 and v2r2
- `manifests/roles/` — 7 role manifests, partition fingerprint `b6c1af90…`, assignment hash `313ae124…`
- `freeze/spec_freeze_v2.json` — sealed spec sha256 `e6c38a3ecb4f0a26e50ab930fa7e27936f15d7303ff794dc77581269a87d3062`

**Allocation verified bit-identical to v2r2** before publishing: assignment hash
`313ae124…`, per-role dialogue sets identical dialogue-by-dialogue across 7 roles
and 100 dialogues, four disjointness assertions all true, gated TV 0.1279 ≤ 0.15,
max |SMD| 0.2240 ≤ 0.25, zero balance failures, D-test exactly the official test
split.

Supersession chain, each link carrying path, sealed sha256, and file sha256:

```text
spec_freeze_v1.json (733d9fee…)
   ↑ supersedes
v2   (9bcf07f7…, fingerprint 87d24cb8…)
   ↑ supersedes_v2_freeze
v2r2 (c572d289…, fingerprint b6c1af90…)
   ↑ supersedes_v2_freeze
v2r3 (e6c38a3e…, fingerprint b6c1af90…)
```

---

## 9. What is now stale

Both of the following were computed on the **v1 sample under the old conversation-
level role partition**. Neither describes the current data, and neither should be
quoted going forward without recomputation on this generation.

- **The Run A convention diagnostic**
  (`docs/RUN_A_CONVENTION_DIAGNOSTIC_RESULTS_2026-08-14.md`). Its four blank-frame
  convention results — `blank_excluded` 440/1260 ms, `blank_to_preceding`
  340/1220, `blank_to_following` 680/1460, `blank_midpoint` 480/1000 — and the
  utterance-final partition (600 final / 2,428 non-final) were measured on 3,028
  paired targets drawn from the v1 candidate table under conversation-level roles.
  The population has changed: roles are now dialogue-atomic, the sample is
  dialogue-stratified at 15 utterances per dialogue, and a third aligner family is
  present.
- **The Session B headroom estimate.** Its span counts, tier counts, and cluster
  counts came from the v1 300-utterance-per-role sample. Both its numerator and its
  independence unit have changed: the sample is now 1,275 utterances stratified by
  dialogue, and the cluster is `dialogue_id`, not `conversation_id`.

Also affected, less directly: any statement that quotes v1 conversation-level
cluster counts, and the `D-dev-select` annotation pack's switch universe, which was
enumerated over the v1 sample.

---

## 10. Open items

1. The partition fingerprint is not a literal field in the candidate generation
   manifest (§7.3). One-line fix, deferred out of the code freeze.
2. `router_calib_split` remains unset. The conversation-level 10/10 described 20
   conversations; how 8 dialogues divide between probability calibration and
   threshold selection is a scientific decision, not a rescaling.
3. Identity binding covers `tests/`, so a test-only edit invalidates data-artifact
   identity. That is what forced v2r3. Reported separately; no change made.

**No gate was evaluated. No gate generation was allocated, exposed, or consumed.
Production Gate A is unchanged.**
