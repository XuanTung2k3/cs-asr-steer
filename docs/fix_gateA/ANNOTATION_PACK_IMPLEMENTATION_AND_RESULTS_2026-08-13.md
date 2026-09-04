# Gate-A lexical-boundary annotation pack: implementation and results

> **SUPERSEDED — 2026-08-14 by `docs/RUNBOOK_V2_2026-08-14.md`.**
>
> Human lexical annotation was **dropped** by explicit decision on 2026-08-13/14.
> The Gate-A criterion moved from absolute boundary accuracy against a human
> reference to a provisional aligner configuration plus an invariance battery,
> so this pack is superseded rather than failed and no annotation pass is
> planned. **The unresolved EN→ZH tier ambiguity recorded below is therefore
> moot** — it no longer blocks anything, and no protocol decision on it is
> required. The pack's switch-universe enumeration is reused by the CTC
> convention diagnostic.
>
> The body below is unaltered and retained as historical evidence of what was
> built, exported, and mechanically verified on 2026-08-13.

**Date:** 2026-08-13

**Repository:** `/home/tungnx/cs-asr-steer`

**Pack:** `/mnt/data/tungnx/cs-asr-steer/gateA_annotation_pack_seed20260813`

**Status:** Export completed and the seven mechanical verification checks passed. One protocol ambiguity must be resolved before annotation begins, and the operational safeguards below must be followed.

## 1. Purpose and scientific boundary

This export prepares development-only audio clips for obtaining human lexical-boundary references that are independent of all automatic aligners. It does not annotate a boundary, run a model, change a production status, allocate a held-out gate generation, freeze spans, or unlock L1c.

The provisional `existing_ctc` boundary is used only to choose the three-second audio window. It is not written to a TextGrid, annotator-facing CSV, audio cue, or label. The analyst manifest necessarily records each clip's source offset, so `manifest.csv` and `manifest.json` must not be given to annotators.

The pack remains diagnostic-only. Its existence does not mean Gate A passes.

## 2. Implementation

### Source files

| File | Purpose |
| --- | --- |
| `src/csasr/lss/align/annotation_pack.py` | Builds the D-dev-select switch universe, assigns duration strata, deterministically samples items, extracts padded clips, and emits empty TextGrids and annotator instructions. |
| `src/csasr/experiments/annotation_pack_export.py` | CLI entry point. Authenticates the cached candidate table, refuses output under the production artifacts root, exports the pack, and writes analyst and annotator manifests. |
| `tests/test_lss_annotation_pack.py` | Regression tests for empty tiers, absence of provisional boundaries, role restriction, deterministic sampling, clip centering/padding, and annotator instructions. |

### Inputs

- Authenticated cached candidates: `/mnt/data/tungnx/cs-asr-steer/artifacts_lss/alignments/candidates_all.parquet`
- Frozen role membership checked against: `/mnt/data/tungnx/cs-asr-steer/artifacts_lss/manifests/roles/role_D-dev-select.parquet`
- Source WAV paths already recorded in the candidate table
- Permitted split: `D-dev-select` only
- Provisional family: `existing_ctc`, for clip centering only
- Seed: `20260813`

No D-construct, D-dev-confirm, D-test, or gate-generation item was used.

### Sampling and export behavior

1. Filter cached candidates to `role == D-dev-select` and `aligner_family == existing_ctc`.
2. Reconstruct language runs in reference-unit order.
3. Enumerate eligible adjacent ZH-to-EN and EN-to-ZH switches.
4. Assign the duration of the associated embedded-English run to each switch.
5. Compute duration terciles and stratify by switch direction × duration tercile.
6. Draw 150 primary and 40 separately flagged items deterministically from seed `20260813`.
7. Extract one three-second, 16-kHz PCM WAV and one empty two-tier TextGrid per selected item.
8. Write separate annotator-facing indexes without timing fields and analyst manifests containing provenance and clip offsets.

The realized tercile cut points were 0.5200 s and 1.2133 s. The eligible universe contained 1,367 boundaries from 298 D-dev-select utterances. Sixteen potential boundaries were excluded because their reference-unit indices were non-contiguous; there were no exclusions for invalid units, non-finite boundaries, or non-positive embedded spans. There were no audio export failures.

### Command used

```bash
cd /home/tungnx/cs-asr-steer
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src

/home/tungnx/miniconda3/envs/acl1/bin/python \
  -m csasr.experiments.annotation_pack_export \
  --config lss/l1b_valid.yaml \
  --output /mnt/data/tungnx/cs-asr-steer/gateA_annotation_pack_seed20260813 \
  --seed 20260813 \
  --primary 150 \
  --double 40
```

## 3. Output inventory

The output root contains exactly 386 files:

| Path | Count | Contents |
| --- | ---: | --- |
| `README.md` | 1 | Annotator instructions |
| `ambiguity_notes.csv` | 1 | Empty ambiguity-note template |
| `manifest.csv` | 1 | Analyst manifest for all 190 items |
| `manifest.json` | 1 | Machine-readable pack metadata and all 190 item records |
| `clips/*.wav` | 150 | Primary clips |
| `clips/*.TextGrid` | 150 | Primary empty label grids |
| `clips/items.csv` | 1 | Primary annotator-facing index without timing |
| `double_annotation/*.wav` | 40 | Separately flagged clips |
| `double_annotation/*.TextGrid` | 40 | Separately flagged empty label grids |
| `double_annotation/items.csv` | 1 | Separate annotator-facing index without timing |

The authoritative full per-item path list is in the `clip_path` and `textgrid_path` fields of `manifest.csv` and `manifest.json`. Together with the six fixed files above, those fields enumerate every written file. All paths are descendants of:

```text
/mnt/data/tungnx/cs-asr-steer/gateA_annotation_pack_seed20260813
```

## 4. Realized sampling

| Stratum | Primary | Separately flagged | Total |
| --- | ---: | ---: | ---: |
| EN→ZH, long | 25 | 7 | 32 |
| EN→ZH, medium | 25 | 7 | 32 |
| EN→ZH, short | 25 | 7 | 32 |
| ZH→EN, long | 25 | 7 | 32 |
| ZH→EN, medium | 25 | 6 | 31 |
| ZH→EN, short | 25 | 6 | 31 |
| **Total** | **150** | **40** | **190** |

Direction totals are 96 EN→ZH and 94 ZH→EN. Duration totals are 64 long, 63 medium, and 63 short. The pack contains 190 unique fingerprints from 128 unique source utterances.

## 5. Seven required verification results

| # | Check | Result | Evidence |
| ---: | --- | --- | --- |
| 1 | Exactly 150 + 40 clips and a matching manifest row for every clip | **PASS** | 190 manifest rows, 190 WAVs, and 190 TextGrids. Every referenced file exists, every filename stem equals its fingerprint, and there are no extra WAV or TextGrid files. |
| 2 | Every source utterance belongs to D-dev-select | **PASS** | Programmatic set-membership comparison used the frozen role manifest's `utterance_id` column. All 128 unique sampled utterance IDs are members of its 3,360 IDs; missing count is zero. Manifest `source_role` is `D-dev-select`. |
| 3 | Same-seed sampling is reproducible | **PASS** | A fresh export to `/tmp/csasr_annotation_repro_20260813.e1LeTt` produced 190 fingerprints identical to the original in the same order, 386 files, and zero failures. |
| 4 | Every TextGrid tier is empty | **PASS** | All 190 TextGrids parse to exactly the tiers `matrix_end` and `embedded_start`; each has one full-clip interval whose label is the empty string. Bad-grid count is zero. |
| 5 | Realized strata reported | **PASS** | Counts are shown in Section 4 and sum to 150 primary + 40 separately flagged items. |
| 6 | Audio duration and pack size reported | **PASS** | Every WAV is exactly 3.0 s. Total audio duration is 570 s (9.5 min). Total size of all 386 files is 18,864,437 bytes (17.99 MiB). |
| 7 | Nothing written outside the requested output directory | **PASS for this execution** | The export reported 386 paths and the filesystem inventory contains exactly those 386 descendants of the output root. The reproducibility check wrote only to the separately named `/tmp` directory. Protected production status/report mtimes remained `1786502490`; `freeze/alignment_freeze.json` remains absent. |

## 6. Tests and static checks

```text
PYTHONPATH=src pytest -q tests/test_lss_annotation_pack.py
14 passed

PYTHONPATH=src pytest -q --ignore=tests/test_lss_l1b_production_paths.py
493 passed (exit 0)

PYTHONPATH=src pytest -q tests/test_lss_l1b_production_paths.py
40 passed (exit 0)

python -m compileall -q src/csasr tests
PASS

bash -n cs_asr_lss.sh
PASS

git diff --check
PASS
```

Together, the two exhaustive pytest invocations cover all 533 collected tests. The same-seed end-to-end re-export also completed successfully with 190 items and zero export failures. No model inference, model download, GPU job, Slurm submission, gate exposure, or production artifact mutation was performed.

## 7. Protocol issues to resolve before annotation

### 7.1 EN→ZH tier semantics are inconsistent

The required tier names and requested instruction are directionally meaningful for a ZH→EN switch: `matrix_end` is the end of the preceding Mandarin word and `embedded_start` is the start of the following English word. They do not describe the two switch-adjacent lexical edges for an EN→ZH switch.

The generated README currently says that, for EN→ZH, `matrix_end` is the point where the following Mandarin word *starts*. That contradicts both the tier name and the requirement to mark a lexical *end*. It can lead annotators to put different kinds of boundaries on the same tier depending on direction, making aggregate error metrics uninterpretable.

**Required decision:** define direction-neutral tiers such as preceding-word end and following-word start, use direction-specific tiers, or restrict this pack to the switch direction for which `matrix_end`/`embedded_start` are scientifically defined. Do not start annotation of EN→ZH items until the protocol, README, schema, and downstream scorer agree.

### 7.2 The 40 designated items still require two independent passes

All 190 fingerprints are unique, matching the requested 150 primary boundaries plus 40 additional boundaries designated for double annotation. The separate 40-item directory contains one copy of each designated item; it does not itself contain two annotation results.

To obtain agreement evidence, each of those same 40 fingerprints must be annotated independently twice—for example by two annotators or in two properly separated sittings. The pack must not be reported as providing agreement data until both completed label sets exist and can be matched by fingerprint.

### 7.3 Keep analyst provenance away from annotators

`manifest.csv` and `manifest.json` contain `clip_start_offset_sec`, source paths, and unit indices. They are necessary analyst provenance but can reveal that the provisional center is 1.5 seconds into the clip. Annotators should receive only `README.md`, the relevant clip/TextGrid directory, its `items.csv`, and the ambiguity template.

## 8. Current Gate-A interpretation

This work prepares missing human evidence; it does not supply it. Gate A remains blocked until:

1. the EN→ZH tier-semantics ambiguity is resolved and the pack is regenerated if necessary;
2. humans provide independent lexical boundary annotations, including two independent passes over every designated 40-item fingerprint;
3. annotation integrity and agreement are evaluated;
4. candidate aligners/conventions are scored against genuine `manual_lexical` references using development evidence only; and
5. the production Gate-A workflow receives a qualifying independent aligner pair and compatible calibration evidence without taint.

No threshold was changed, no convention was selected, and no held-out evidence was consumed.

## 9. Recommended next action

Do not begin EN→ZH annotation yet. First make an explicit protocol decision on its tier semantics. Also assign the designated 40 fingerprints to two independent annotation passes and keep their outputs separate. If the semantic decision changes the schema or instructions, update the exporter/tests and generate a new versioned output directory rather than overwriting this evidence pack.
