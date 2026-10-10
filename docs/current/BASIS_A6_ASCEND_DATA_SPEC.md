# BASIS-A6 — ASCEND Data Protocol (SCIENTIFIC SPEC)

Status: **FROZEN PROTOCOL — data acquisition + canonical adapter + subset freeze rules.**
This document defines how the ASCEND corpus enters BASIS-A6 as (a) a **construction source**
(`ASCEND-construct`, from TRAIN only) and (b) an **evaluation panel** (`ASCEND-eval`, from
VALIDATION only). It changes **no** existing dataset, split, or panel. **No GPU jobs and no
scientific inference are launched by this stage.** Part of the BASIS-A6 freeze
(`BASIS_A6_EXPANDED_SPEC.md`).

Companion documents: `BASIS_A6_FIXED_SPEC.md` (A6-F), `BASIS_A6_ORACLE_TT_SPEC.md` (A6-TT),
`BASIS_A6_EXECUTION_PLAN.md`. Data-role table update in `DATA_EXPOSURE.md`.

---

## 0. Scope in one paragraph

Add the **official** ASCEND corpus (`CAiRE/ASCEND`, HF Hub) as a fourth corpus through the
**same canonical data interface** used by CS-Dialogue and SEAME — one adapter, no
dataset-specific special cases sprinkled through the steering runner. ASCEND `train` supplies
a deterministic frozen `ASCEND-construct` subset (matched in size to CS-Dialogue
`D-construct`); ASCEND `validation` supplies a deterministic frozen `ASCEND-eval` panel
(target 300 eligible code-switched utterances, to match CS `D-dev-select`). **ASCEND `test`
is DO-NOT-READ / DO-NOT-USE.** Eligibility requires *verified* transcript-level
code-switching (≥1 English content unit AND ≥1 Chinese content unit), not the dataset-level
"mixed" label alone.

---

## 1. Corpus + roles (FROZEN)

| ASCEND HF split | Approx N (reference) | BASIS-A6 role | Use |
|---|---:|---|---|
| `train` | 9,869 | `ASCEND-construct` (subset) | **construction source only** |
| `validation` | 1,130 | `ASCEND-eval` (panel) | **evaluation only** |
| `test` | 1,315 | — | **DO NOT READ / DO NOT USE** |

Rules:
- No `validation` or `test` utterance may ever enter construction.
- No `test` utterance may ever be read (the download guard `DO_NOT_USE_TEST.marker` and the
  adapter both enforce this).
- The construction subset and the eval panel are disjoint by construction (different HF
  splits) and are each frozen with an exact-ID manifest + fingerprint before any decode.

---

## 2. Download / preparation path (reproducible)

Canonical script: **`scripts/download_ascend.py`** (already committed with this protocol). It
wraps the basic command below but adds provenance capture and loud failure on incompatible
upstream structure.

```bash
mkdir -p data/external/ASCEND
python scripts/download_ascend.py --out data/external/ASCEND        # optional: --revision <sha>
```

Equivalent minimal form (recorded for transparency; the script is the path of record):

```python
from datasets import load_dataset
ds = load_dataset("CAiRE/ASCEND", cache_dir="data/external/ASCEND/hf_cache")
ds.save_to_disk("data/external/ASCEND/dataset")
```

The script records, in `data/external/ASCEND/ASCEND_DOWNLOAD_PROVENANCE.json`:
- `dataset_repo = CAiRE/ASCEND`, `requested_revision` (pinned sha/tag if given),
- observed split names and **sizes**, and whether they match the protocol reference,
- per-split HF `_fingerprint` and the feature schema,
- `preparation_code_hash` (sha256 of the download script itself),
- the `do_not_use_splits = ["test"]` guard.

**Fail-loud rules (frozen):** missing any of `{train, validation, test}` → FATAL (protocol
incompatible). Split sizes differing from the reference `{train 9869, validation 1130, test
1315}` → FATAL unless a human updates this spec with the new revision + sizes and re-runs with
`--allow-size-mismatch`. Never silently reshape the data to fit the protocol.

**Do NOT commit dataset audio to git.** Git-ignore `data/external/ASCEND/dataset/` and
`data/external/ASCEND/hf_cache/`; only the small JSON provenance + the frozen subset manifests
(§6–§7) are tracked.

---

## 3. Canonical ASCEND adapter (ONE interface)

Implement ASCEND behind the **same canonical data interface** as CS-Dialogue and SEAME (the
adapter yields the project's canonical utterance record: id, audio path/array + sample rate,
normalized reference transcript, duration, and metadata). The steering runner must not learn
that ASCEND exists beyond selecting the adapter — **no dataset-specific branches** in the
steering / extraction / masking code.

Official ASCEND fields consumed (approximate upstream schema):

| ASCEND field | Canonical use |
|---|---|
| `id` | canonical `utterance_id` (namespaced `ascend/<id>`) |
| `audio` (`path` / array / `sampling_rate`) | canonical audio input |
| `transcription` | raw reference → canonical **normalize()** → `Unit` sequence |
| `duration` | reported duration (stratification + reporting) |
| `language` | recorded metadata only (NOT trusted for eligibility) |
| `original_speaker_id` | stratification key (speaker) |
| `session_id` | stratification key (session) |
| `topic` | stratification key (topic) |

Transcript processing reuses the project's canonical language-aware pipeline
(`csasr.data.normalize` → `Unit`s → `csasr.data.language_tags.tag_unit`, tags
`EN / ZH / OTHER / UNKNOWN`). No new tokenizer, no ASCEND-specific normalization.

---

## 4. Eligibility rule (verified code-switching — FROZEN)

An ASCEND utterance is **CS-eligible** iff, under the canonical language-aware transcript
processing, it contains:

- **≥ 1 English content unit** (`tag == EN`), **AND**
- **≥ 1 Chinese content unit** (`tag == ZH`).

Content units exclude `OTHER` (fillers, bare numerals, romanized-Mandarin) and `UNKNOWN`
(mixed-script / non-ASCII-alnum), exactly as the frozen `language_tags` rules define. The
dataset-level `language == "mixed"` label is **not** sufficient and is **not** trusted on its
own; eligibility is decided from the transcript tags. This same rule governs both
`ASCEND-construct` and `ASCEND-eval`.

The number of English content units and Chinese content units per utterance is recorded in
each manifest (they drive the eligibility decision and the reported group-support statistics).

---

## 5. Reference reporting statistics (per subset)

For each frozen subset, record but **do not outcome-tune**:
- number of eligible utterances (and total scanned),
- total duration,
- number of embedded-English content units and matrix-Chinese content units,
- (at construction, once a model pass exists) number of eligible **encoder frames** and
  eligible **decoder positions** — these are populated when the construction pass runs, not at
  freeze time, and are stored alongside the direction diagnostics.

---

## 6. `ASCEND-construct` subset (TRAIN only — FROZEN manifest)

- Source: **ASCEND `train` only.** Deterministic, frozen, exact-ID.
- **Primary matching target:** `N_construct(ASCEND) = N_construct(CS D-construct)` — the
  number of construction **utterances** used for CS-Dialogue direction construction in
  A4/A5. That frozen CS construction population is **125 utterances / 20 dialogues** (A5 §2;
  A4 DG-03 construction manifest). The ASCEND-construct target is therefore **125 eligible
  utterances**. At freeze time the exact CS count is re-read from the frozen A4/A5
  construction manifest; if it differs from 125 the ASCEND target follows the CS count (the
  rule is "match CS", the number is bound to the CS manifest, not hard-coded).
- If fewer than the target number of `train` utterances are CS-eligible (§4), use all eligible
  and document the exact `N`. (ASCEND train is ~9,869 rows, so the 125 target is expected to
  be comfortably reachable; the fallback exists for completeness.)
- **Deterministic stratified selection:** stratify across available `original_speaker_id`,
  `session_id`, and `topic` where practical, then take a seeded deterministic sample to the
  target size. **Canonical seed = 42** (project default; `csasr.data.splits`,
  `csasr.lss.seeds`) unless an authoritative current config says otherwise.
- Also report (not tuned): total duration, embedded-English units, matrix-Chinese units,
  eligible encoder frames, eligible decoder positions (frames/positions filled at pass time).
- Freeze **`ASCEND_CONSTRUCT_MANIFEST.json`** with: schema version, source split, seed,
  stratification keys, exact ordered utterance IDs, per-utterance (EN units, ZH units,
  duration), aggregate stats, eligibility-rule hash, and a subset `fingerprint`
  (sha256 over the ordered ID list + rule hash).

No `validation`/`test` utterance may enter this subset.

---

## 7. `ASCEND-eval` panel (VALIDATION only — FROZEN manifest)

- Source: **ASCEND `validation` only.** Deterministic, frozen, exact-ID.
- **Target: 300 eligible code-switched utterances** (to match CS `D-dev-select = 300`).
- Deterministic stratified selection across `original_speaker_id` / `session_id` / `topic`
  where possible; seed = 42.
- If fewer than 300 satisfy §4, **use all eligible** examples and document the exact `N`
  (validation is ~1,130 rows, so 300 eligible is plausible but not guaranteed; the realized
  `N` is whatever the frozen rule yields, never padded with ineligible rows).
- Freeze **`ASCEND_EVAL_MANIFEST.json`** with the same fields as §6.
- ASCEND `test` remains untouched.

`ASCEND-construct` ∩ `ASCEND-eval` = ∅ (different HF splits); a disjointness assertion is
recorded in both manifests.

---

## 8. Provenance to record (for the freeze commit)

- `dataset_repo`, pinned `revision` (if retrievable), observed split **sizes**, HF
  fingerprints, feature schema (from `ASCEND_DOWNLOAD_PROVENANCE.json`).
- `preparation_code_hash` (download script) and the subset-selection code hash (recorded when
  the selection code lands; the *rule* is frozen here regardless of implementation).
- Both subset manifests + their fingerprints.
- Canonical seed used (42) and the stratification keys actually available.

---

## 9. Scientific risks

1. **Trusting the "mixed" label.** Mitigation: eligibility is decided from canonical
   transcript tags (§4), not the dataset label; EN/ZH unit counts are recorded.
2. **Domain/style mismatch vs CS-Dialogue/SEAME** (topics, spontaneity). Mitigation: ASCEND
   is a **transfer/robustness** panel and a second construction source; cross-corpus reads are
   reported as transfer, never as within-corpus claims.
3. **Size matching is approximate.** Mitigation: primary target = utterance count matched to
   CS D-construct; duration/frame/position budgets are reported, not equalized. A hard
   equalization would be outcome-tuning and is forbidden.
4. **Upstream revision drift.** Mitigation: pin + record the revision and fingerprints;
   fail-loud on structural/size change (§2).
5. **Accidental test exposure.** Mitigation: download guard marker + adapter refusal on the
   `test` split; construction=train, eval=validation, enforced at manifest freeze.

---

## 10. Protocol Status: **PASS**

Corpus, roles, download path, canonical adapter interface, eligibility rule, subset targets,
stratification, seed, manifests, and provenance are determined and frozen. Adapter/subset code
is a later implementation step; no data reshaping, no GPU work, and no scientific inference are
launched by this stage.
