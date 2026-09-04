# v2r3 baseline and POI labelling: execution and results

**Date:** 2026-08-15
**Job:** Slurm 40486, `COMPLETED`, exit 0, wall clock **118 s (1 m 59 s)**
**Destination:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/baselines/generation_001`
**Status:** B0_AUTO hypotheses and lexical POI tables exist for `D-construct` and
`D-dev-select` on the v2r3 dialogue-atomic sample. No gate was evaluated, no
baseline was selected, and no gate generation was allocated.

---

## 1. Purpose and scientific boundary

This run produces **baseline labelling only**: what the frozen Whisper baseline
transcribes for each sampled utterance, and whether each embedded-English lexical
unit is reproduced correctly.

It does **not**:

- evaluate any gate, or write any status or freeze artifact;
- select a primary baseline, or write `primary_baseline.json`;
- allocate, expose, or consume a gate generation;
- produce anything for `D-dev-confirm` or `D-test`;
- measure alignment accuracy or compare an aligner against truth.

Why both roles were needed: direction construction consumes baseline-**correct**
units from `D-construct`, and development evaluation consumes baseline-**error**
units from `D-dev-select`. If those two rested on differently-defined baselines,
the contrast between them would be unsound. §5 shows they do not.

---

## 2. A new driver was required

`p0_baseline.py:133` could not be reused, and the authorization forbade adapting
scoring logic, so the blocker was reported before any code was written. Four
independent reasons:

1. Its POI loop is `for subset in DEV_SUBSETS` with
   `DEV_SUBSETS = ("dev_select", "dev_confirm")` — no `D-construct`, and it would
   have generated the held-out `D-dev-confirm`.
2. It loads v1 split manifests via `load_subset`, which resolves to
   `artifacts_root/manifests/{name}.parquet`. It has no notion of the v2r3 role
   manifests or the dialogue-stratified sample.
3. It writes `poi_{subset}.parquet` into the v1 artifacts root — over the
   immutable `poi_dev_select.parquet`.
4. It selects a primary baseline (lines 122–131) and evaluates the P0 gate
   (lines 158–170).

`src/csasr/experiments/v2r3_baseline_poi.py` is a new driver that does none of
those things. **The scoring functions are reused verbatim**: it imports
`csasr.evaluation.pier.poi_table` and calls it unmodified — no wrapper, no
re-implementation, no additional normalization. A regression test asserts
`driver.poi_table is pier.poi_table` and that the module contains no local
definition of `poi_table`, `evaluate_pois`, `_categorize`, and no call to
`align_tokens` or `normalize_text`. `p0_baseline.py` and `pier.py` were not
modified.

The driver refuses any role outside `{D-construct, D-dev-select}` **by name**
rather than skipping it, so requesting a held-out split fails loudly:

```text
BaselinePoiError: role(s) ['D-dev-confirm'] are not permitted here; this driver
generates baselines only for ['D-construct', 'D-dev-select']. D-dev-confirm and
D-test are held out and must not receive baseline labelling.
```

Publication is **attempt-private with atomic promote** — the stronger of the two
mechanisms in the repository. Unlike the candidate generation, no partial
directory is ever visible at the destination.

---

## 3. The overlap finding

The prior `poi_dev_select.parquet` was built on the v1 sample under the
conversation-level partition. Measured against the v2r3 `D-dev-select` sample:

| Level | v1 `poi_dev_select` | v2r3 `D-dev-select` | Shared |
|---|---:|---:|---:|
| Utterances | 1,718 | 300 | **54** (18.0% of new) |
| Conversations | 20 | 40 | 6 |
| Dialogues | 14 | 20 | **4** |
| Lexical units | 23,726 | 2,268 | — |

**Interpretation.** The decisive evidence is not the percentage but where v1's
dev_select dialogues went. Its 14 dialogues are now distributed across **five**
roles under the dialogue-atomic partition:

| Where v1 dev_select's 14 dialogues sit in v2r3 | Count |
|---|---:|
| D-construct | 5 |
| D-dev-select | 4 |
| loc-train | 2 |
| util-train | 2 |
| router-calib | 1 |

That is what a genuine re-partition looks like, and it confirms the premise for
this run: the old table no longer describes the split it names. The conversation
counts fit the same story — v1 held one side of several dialogues (20
conversations from 14 dialogues), v2r3 holds both sides of all 20. The 54 shared
utterances come from the 4 dialogues that stayed.

Overlap was low, so the run proceeded. A high overlap would have meant the
re-partition had not taken effect as the manifests describe, and would have
stopped the run.

---

## 4. Pre-flight

| # | Check | Result |
|---|---|---|
| 1 | v2r3 candidate generation complete | **PASS** — `state: completed`, 10 artifacts, 0 unresolved or hash-mismatched, candidate `cdc0ff4e…` |
| 2 | Old-vs-new dev_select overlap | **PASS (low)** — see §3 |
| 3 | Destination absent | **PASS** — `destination_exists_before=no`, confirmed again inside the job |
| 4 | Exposure generation 3 unallocated | **PASS** — ledger holds 0, 1, 2 |
| 5 | Source snapshot recorded | `64346ca6852cdf5b588708cd7e4acc2da5dbaf3eb87737ed63161a726811adca` at pre-flight, unchanged from the v2r3 publish; docs-only edits are outside the snapshot |

The freeze baseline differs from the pre-flight value because the new driver and
its tests were added after pre-flight and before publication — deliberately, and
in that order, after the lesson recorded in §13 of the runbook. See §8.

---

## 5. Command and configuration equality

```bash
sbatch /mnt/data/tungnx/cs-asr-steer/jobs/v2r3_baseline_poi/run_baseline_poi.sbatch
```

Resources `--partition=main --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=20:00:00`.
The launcher runs exactly one command:

```bash
python -m csasr.experiments.v2r3_baseline_poi \
    --config lss/l1b_candidates_dialogue_v2r3.yaml \
    --roles D-construct D-dev-select \
    --output-dir .../artifacts_dialogue_v2r3/baselines/generation_001 --execute
```

Node `worker-0`, NVIDIA H100 80GB HBM3, `openai/whisper-large-v3` in bfloat16,
git SHA `fdd441a75b0d66f56d8e5efec62f01649f7b9ae9`, started 04:13:09Z, ended
04:15:07Z.

### Post-verify 3 — decoding and normalization, side by side

| Setting | v1 `dev_select` | v2r3 `D-construct` | v2r3 `D-dev-select` |
|---|---|---|---|
| `task` | transcribe | transcribe | transcribe |
| `language` | None | None | None |
| `temperature` | 0.0 | 0.0 | 0.0 |
| `do_sample` | False | False | False |
| `num_beams` | 1 | 1 | 1 |
| `return_dict_in_generate` | True | True | True |
| `output_scores` | True | True | True |
| `return_timestamps` | False | False | False |
| `max_new_tokens` | 200 | 200 | 200 |
| `batch_size` | 16 | 16 | 16 |
| `system` / `config_id` | B0_AUTO | B0_AUTO | B0_AUTO |
| `model_revision` | `sha256:a8e94b859…` | `sha256:a8e94b859…` | `sha256:a8e94b859…` |

**All settings identical, including the model revision hash.** The requirement
that must hold — `D-construct` and `D-dev-select` from identical settings — holds.
Continuity with the v1 run also holds, so no approximation was needed and nothing
had to be reported as a deviation.

Normalization is identical by construction: both roles are scored by
`pier.poi_table` → `evaluate_pois` → `normalize_text` / `segment_units`, the same
functions the v1 run used, imported unmodified.

The B0/B1 selection step was not reproduced. With a single baseline there is
nothing to select, and selection would have written the `primary_baseline.json`
this run must not produce. `language=None` is hard-bound in the driver.

---

## 6. Results

### Post-verify 1 — lexical units, correct, baseline-error

| Role | Lexical units | Correct | Baseline-error | Utterances | Dialogues |
|---|---:|---:|---:|---:|---:|
| D-construct | 2,310 | 1,408 | 902 | 300 | 20 |
| D-dev-select | 2,268 | 1,384 | 884 | 300 | 20 |
| **Total** | **4,578** | **2,792** | **1,786** | 600 | 40 |

Error categories:

| Category | D-construct | D-dev-select |
|---|---:|---:|
| `deletion` | 456 | 499 |
| `wrong_language_substitution` | 224 | 175 |
| `same_language_substitution` | 125 | 138 |
| `boundary_error` | 50 | 34 |
| `phonetic_transliteration_or_script` | 42 | 30 |
| `other` | 5 | 8 |

These are counts. No threshold is applied to them here and no gate is evaluated.

### Post-verify 2 — every eligible unit maps to a POI status

| Role | Rows | Null `category` | Null `correct` | correct + error = rows |
|---|---:|---:|---:|:--:|
| D-construct | 2,310 | **0** | **0** | True |
| D-dev-select | 2,268 | **0** | **0** | True |

**Unmatched count 0**, as expected. The driver additionally refuses to build a
table if any sampled utterance received no hypothesis, so silent row loss cannot
occur.

---

## 7. Artifacts

`artifacts_dialogue_v2r3/baselines/generation_001/`:

| sha256 | Bytes | Path |
|---|---:|---|
| `c903e77fd2657450de4b49e2e7f40a62…` | 178,105 | `baselines/B0_AUTO/D-construct.parquet` |
| `737d3308e15ca2d3f6ca6c5dfef5ee78…` | 208,130 | `baselines/B0_AUTO/D-dev-select.parquet` |
| `334feed9f063ee7d9d9341f0eaa6bac2…` | 34,476 | `poi_D-construct.parquet` |
| `5cebe272022ea66739b45c7171a2b9fd…` | 34,501 | `poi_D-dev-select.parquet` |

Plus the two `.meta.json` decoding-provenance files and
`baseline_poi_complete.manifest.json`, which inventories all six artifacts with
hashes and records `evaluates_gate: false` and `selects_primary_baseline: false`
in the artifact itself.

Provenance binds each published table to four parents by SHA-256:

```text
candidates_all.parquet            cdc0ff4e59fe18901c8e1a5e…   (v2r3 candidate generation)
spec_freeze_v2.json               e961f3295e54468f6a3d0193…   (v2r3 freeze)
role_D-construct.parquet          d61c305ef19f0c3d52daa988…
role_D-dev-select.parquet         c0f3ad12be3e0d66d20bec04…
```

Sidecars carry `taint_reasons: []`, `diagnostic_only: false`, and
`identity.source_sha256 = 4fff0bcf…` — the frozen tree.

---

## 8. Immutability and the freeze

### Post-verify 5 — existing `artifacts_v2` tables unmodified

| mtime | sha256 | File |
|---|---|---|
| 2026-07-27 19:42:01 | `8ab686f5d3d69421f62657c2…` | `manifests/poi_dev_select.parquet` |
| 2026-07-27 19:42:02 | `984a38e383414d3f969845d5…` | `manifests/poi_dev_confirm.parquet` |
| 2026-07-27 19:19:15 | `7ee22b08691028dd757152b7…` | `baselines/B0_AUTO/dev_select.parquet` |
| 2026-07-27 19:26:27 | `57c43ef5b8ed66f005ad113d…` | `baselines/B0_AUTO/dev_confirm.parquet` |
| 2026-07-27 19:41:59 | `2c3b2a0f6fe05da93e404fcc…` | `baselines/primary_baseline.json` |

**Zero files under `artifacts_v2` modified since 2026-08-01.**

### Post-verify 6 — v1 artifacts unchanged (repository hasher)

`candidates_all.parquet` = `29d7018d44536abce7afc80e900aa20b21ae5e1508c16f85562b3ccb86e23ce0`
(expected value), `spec_freeze_v1.json` = `41e3b483…`, and all seven v1 role
manifests unchanged at their 2026-08-12 00:47 mtimes.

### Post-verify 7 — superseded namespaces unchanged

`artifacts_dialogue_v2` newest mtime 2026-08-14 17:00:10;
`artifacts_dialogue_v2r2` newest 2026-08-14 19:41:13. Both as expected.

### Post-verify 8 — no gate generation

Exposure ledger mtime **1786502490**, sha256 `46849f5caed62201…`, generations
0, 1, 2 only. Generation 3 remains unallocated.

### Post-verify 9 — source snapshot across the freeze window

| Checkpoint | Hash |
|---|---|
| Before publication | `4fff0bcf039d0d19b0e3310adf27600f50f408a966429ef82b6f379f89352829` |
| Inside the job (printed by the launcher) | identical |
| After the completion marker | identical |

**All three identical.** Nothing under `src/`, `configs/`, or `tests/` changed
during the window. The full CPU suite was run *before* the freeze began:
**647 passed, 0 failures** (632 + 15 new driver tests).

---

## 9. What is now stale

- **`artifacts_v2/manifests/poi_dev_select.parquet` and `poi_dev_confirm.parquet`.**
  Built on the v1 sample under the conversation-level partition. `poi_dev_select`
  now describes a split that no longer exists in that form — 10 of its 14
  dialogues have moved to other roles. It remains on disk, immutable, as the
  record of the v1 run; it must not be used as `D-dev-select` labelling for
  dialogue-atomic work.
- **`artifacts_v2/baselines/primary_baseline.json`.** Selected on the v1
  dev_select split. No selection has been performed on the v2r3 population.
- Previously recorded and still stale: **the Run A convention diagnostic** and
  **the Session B headroom estimate**, both computed on the v1 sample under the
  old role partition.

No `D-dev-confirm` baseline exists for the v2r3 partition, by design — it is held
out and the driver refuses it.

---

## 10. Open items

1. **Headroom is now computable but not computed.** The conservative subset size
   under the current eligibility rule needs the baseline-error units from this run
   intersected with the eligibility filters. That is the next natural step.
2. Carried forward from the v2r3 generation: the partition fingerprint is not a
   literal field in the candidate generation manifest, and `whisper_dtw` shows
   17.4% invalid units on development roles.
3. Identity granularity covers `tests/`; recorded in the runbook §13, deferred.

**No gate was evaluated. No gate generation was allocated, exposed, or consumed.
Production Gate A is unchanged.**
