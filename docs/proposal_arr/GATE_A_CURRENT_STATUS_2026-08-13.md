# Alignment Gate A: current status and completion plan

> **SUPERSEDED — 2026-08-14 by `docs/RUNBOOK_V2_2026-08-14.md`.**
>
> The current protocol treats Gate A as a development measurement supported by
> a provisional aligner configuration and an invariance battery, not as an
> absolute lexical-accuracy gate. The body below is retained unaltered as
> historical evidence of the status and interpretation recorded on 2026-08-13.

**Status date:** 2026-08-13 UTC  
**Repository:** `/home/tungnx/cs-asr-steer`  
**Branch:** `main`  
**Last production run:** Slurm job `38745`  
**Overall verdict:** **PARTIALLY COMPLETE — do not run the next production Gate A yet**

## 1. Executive summary

The large implementation repair is complete enough to establish a safe baseline:

- the complete CPU test suite passes: **500 passed**;
- cached candidate artifacts authenticate and carry no diagnostic taint;
- full-role data sufficiency is now evaluated from the complete L0 manifests;
- CTC and Whisper qualify as two independent families at the task-specific
  target-object level on the cached job-38745 candidates;
- they have 9,728 paired eligible target objects, so they did not process
  disjoint data;
- raw unit defects remain visible while internal same-language defects can be
  aggregated safely at language-run level;
- production Gate A cannot use an RMS/VAD audio splice as lexical truth;
- no production span freeze exists, and L1c remains locked.

Gate A still cannot pass for two primary reasons:

1. There is no genuine lexical-boundary reference configured. The available
   CS-Dialogue tree contains utterance transcripts and audio, but no obvious
   TextGrid, CTM, EAF, LAB, or word-timestamp files. The current synthetic
   development items have `reference_kind=audio_splice`, which is a known audio
   seam, not a known lexical word boundary.
2. CTC and Whisper disagree far beyond the frozen natural-agreement limits:
   median 440 ms, P90 1,680 ms, and an EN-ZH median difference of 440 ms.

The next work is therefore not another threshold adjustment. It is a large,
scientifically meaningful addition: ingest a genuine lexical reference and use
development-only lexical evidence to resolve the CTC/Whisper boundary convention
before exposing another held-out gate generation.

## 2. Evidence and precedence

Three evidence sources must not be confused:

1. **Job 38745 production status.** The on-disk
   `status/l1b_valid.json` says `blocked`, `complete=true`, exit 0, with the old
   blockers `blocked_no_qualifying_operating_tolerance`,
   `blocked_insufficient_independent_aligners`, and
   `blocked_no_paired_cross_aligner_evidence`. This is a truthful historical
   record of that run, but it predates the current target-object and reference-
   semantics repairs.
2. **Current CPU-only diagnostic over authenticated job-38745 candidates.** This
   reinterprets the same raw candidates using the repaired logic. It is
   diagnostic-only and cannot pass Gate A or freeze spans.
3. **Current unit tests.** These validate code semantics with CPU fixtures and
   mocks. They do not validate real Qwen, Whisper, or CTC model behavior after
   the code changes.

Accordingly, the diagnostic corrects the explanation of job 38745, but it is
not a replacement production Gate-A result.

## 3. Current production state

| Item | Current state | Interpretation |
|---|---:|---|
| Slurm job 38745 | `COMPLETED`, exit `0:0` | The orchestration succeeded; this was not a crash. |
| L0 | passed | Full role manifests and prerequisites exist. |
| L1a | passed historically | Must be rerun after a real lexical-development source is added. |
| L1b | `blocked`, `complete=true` | Required evidence was unavailable; this is not `completed_no_go`. |
| Provenance taint | none | Existing candidate evidence is not diagnostic-tainted. |
| Production frozen spans | absent | Correct: blocked evidence cannot freeze spans. |
| L1c | locked | Correct and required. |
| Exposure ledger | gate generations 0, 1, and 2 exposed | None may be reused as a new confirmatory gate. |
| Gate generation 3 | not allocated | Preserve this until development evidence is credible. |

The current status and report files are historical outputs from job 38745:

```text
/mnt/data/tungnx/cs-asr-steer/artifacts_lss/status/l1b_valid.json
/mnt/data/tungnx/cs-asr-steer/artifacts_lss/reports/l1b_gate_a.md
/mnt/data/tungnx/cs-asr-steer/artifacts_lss/synthetic/exposure_ledger.json
```

## 4. What now passes or is correctly implemented

### 4.1 Code and reliability checks

The following repairs are implemented and covered by the current CPU suite:

- full-role sufficiency and sampled alignment-audit counts use distinct
  universes;
- mathematically unreachable configured count criteria are rejected;
- invalid duration and nonmonotonicity have separate definitions and
  denominators;
- raw aligner items, normalized units, language runs, and target objects retain
  traceable provenance;
- internal zero-duration same-language units do not invent timestamps but may
  leave valid enclosing outer edges;
- Whisper same-language shared-token intervals may be merged without merging
  across a language switch;
- cross-language overlap and invalid switch ordering remain invalid;
- pair selection is deterministic from configured priority, not best gate error;
- a rejected third aligner cannot contaminate the selected pair's calibration;
- insufficient qualifying families and insufficient paired overlap produce
  distinct blockers;
- natural disagreement is not serialized as absolute ground-truth error;
- reference kinds are explicit;
- audio-splice evidence emits seam-relative metrics only;
- genuine lexical fixtures support true start, end, combined, signed, median,
  P90, and within-tolerance error metrics;
- candidate pools are configured above the 100-usable-boundary requirement;
- CTC frame time derives from waveform duration and emission-frame count;
- Whisper receives a real attention mask and generation score options match
  their consumers;
- jitter output separates geometric behavior from unavailable downstream
  scientific-conclusion stability;
- candidate caches and score artifacts are bound to request/item fingerprints;
- exposure generations are immutable and interrupted attempts are not silently
  reused;
- artifact parent provenance and diagnostic taint propagate transitively;
- `blocked`, `completed_no_go`, `failed`, and successful preparation remain
  distinct;
- only an untainted `passed` result can freeze spans and unlock L1c;
- a CPU-only cached diagnostic exists and cannot allocate/expose a gate set,
  write production status, freeze spans, or unlock L1c.

### 4.2 Corrected interpretation of natural evidence

Using the current target-object logic on the authenticated job-38745 candidate
table:

| Family | Valid target objects | Target coverage | Qualification |
|---|---:|---:|---|
| `existing_ctc` | 9,730 / 9,730 | 100.000% | qualifies |
| `whisper_dtw` | 9,728 / 9,730 | 99.979% | qualifies |
| `qwen_forced_aligner` | 8,519 / 9,730 | 87.554% | rejected |

The deterministic selected pair is:

```text
existing_ctc × whisper_dtw
```

It has:

```text
paired target objects: 9,728
paired target rate:     99.979%
minimum required count: 100
minimum required rate:  90%
```

Therefore these repaired requirements pass on cached evidence:

- two genuinely independent natural aligner families;
- sufficient target-object validity and coverage for CTC and Whisper;
- sufficient paired overlap for the selected pair;
- deterministic pair selection;
- Qwen remains visible diagnostically but cannot fail the selected pair.

This also proves that job 38745's old “disjoint sets” explanation was wrong.
The aligners had extensive overlap; paired evidence was suppressed because the
old raw-unit qualification rejected Whisper.

### 4.3 Corrected full-role data sufficiency

The full L0 role manifests, rather than the 300-item-per-role audit sample, give:

| Requirement | Current full-role value | Threshold | Result |
|---|---:|---:|---|
| D-construct bilingual utterances | 2,126 | 500 | pass |
| loc-train embedded-English targets | 4,589 | 1,000 | pass |
| D-dev-select targets | 2,772 | 300 | pass |
| D-dev-confirm targets | 1,618 | 150 | pass |

The previous 298/500 and 847/1,000 failures were sampling/bookkeeping failures,
not evidence that the full frozen role pools were insufficient.

## 5. What still fails scientifically

### 5.1 Natural CTC–Whisper agreement

The selected pair currently measures:

| Criterion | Cached diagnostic | Required | Result |
|---|---:|---:|---|
| Median boundary disagreement | 440 ms | ≤100 ms | fail |
| P90 boundary disagreement | 1,680 ms | ≤200 ms | fail |
| EN median disagreement | 240 ms | diagnostic | — |
| ZH median disagreement | 680 ms | diagnostic | — |
| EN-ZH median difference | 440 ms | <30 ms | fail |
| Within 100 ms | 8.24% | diagnostic | — |

These are natural cross-aligner disagreement measurements, not absolute error.
They show that the two estimators are not presently compatible enough to define
stable production span edges.

The seam-relative diagnostic provides a useful clue:

| Family | Median ZH end minus seam | Median EN start minus seam | Median gap |
|---|---:|---:|---:|
| CTC | -320 ms | +450 ms | ~800 ms |
| Whisper | +220 ms | +300 ms | 0 ms |
| Qwen | -540 ms | +360 ms | ~800 ms |

This is consistent with different ownership of blank/silence frames around
unit boundaries. It is a diagnosis to test, not permission to subtract a
post-hoc constant.

### 5.2 Genuine lexical calibration is unavailable

The current development and gate construction uses RMS/VAD trimming followed
by concatenation. It knows the audio sample at which two clips were joined but
does not independently know the lexical end of the Mandarin word or lexical
start of the English word.

Therefore it can measure:

- `zh_end_minus_splice_ms`;
- `en_start_minus_splice_ms`;
- `gap_around_splice_ms`;
- absolute offset from the audio seam.

It cannot support a claim named lexical absolute-boundary error and cannot
satisfy the 100/200-ms absolute-error sub-gate.

### 5.3 No operating tolerance can currently be selected

The old 100-item development artifact is both undersized relative to the new
150-candidate configuration and semantically incompatible because it contains
`audio_splice` references.

With the current repaired code:

1. the existing 100-item development artifact is rejected before a gate
   generation is allocated because 100 candidates cannot guarantee 100 usable
   boundaries;
2. after L1a rebuilds 150 items, the items still have
   `reference_kind=audio_splice`, so L1b blocks with
   `blocked_missing_genuine_lexical_calibration` before allocating generation 3.

This is the correct safe behavior. No fallback tolerance may be chosen.

### 5.4 Jitter and usable-item evidence remain unresolved

Job 38745 reported failures in safe-interior survival, mask IoU, and usable-item
rate. The implementation now adds duration-stratified diagnostics and separates
geometric survival from scientific-conclusion stability, but no production run
has evaluated those changes with a valid operating point.

These criteria must be rerun after the lexical-reference and boundary-
convention work. The old 90% thresholds have not been weakened. Downstream
conclusion-stability evidence remains explicitly unavailable until its artifacts
exist.

## 6. Remaining implementation gaps and unverified behavior

### P0 — No production lexical-reference ingestion path

The scoring layer supports `manual_lexical`, `existing_gold_lexical`, and
`constructed_exact_lexical`, but the production synthetic builder always emits
`audio_splice`. Merely changing `configs/lss/audit.yaml` would be false labeling.

Required repair:

- add a configured, schema-validated, authenticated loader for an actual lexical
  reference artifact;
- validate finite ordered lexical start/end edges and reference semantics;
- create immutable, source-disjoint development and gate partitions;
- fingerprint the exact items, truth values, sources, and configuration;
- keep at least 150 candidates for at least 100 paired scorable boundaries;
- prevent reference rows from entering both selection and confirmation.

### P0 — CTC/Whisper boundary conventions remain incompatible

The current CTC token spans exclude CTC blank states. The ~800-ms seam gap
suggests that blank-frame ownership may be responsible for much of the natural
disagreement. This has not yet been confirmed with genuine lexical development
truth.

Required repair:

- retain raw token/emission spans;
- implement explicitly named, deterministic candidate conventions for assigning
  the transition between adjacent language runs;
- compare them only on genuine lexical **development** references;
- freeze the chosen convention and its configuration hash;
- apply it unchanged to natural and held-out gate items;
- never fit an offset or choose a convention from held-out gate results.

### P1 — No fresh dev-only aligner runner outside production artifacts

The safe diagnostic command recomputes target objects from cached candidates,
but it cannot exercise the corrected CTC timing or Whisper attention-mask path.
`l1b --align --only evidence` avoids the synthetic/gate parts, but still writes
candidate tables and stage status under the production artifacts root.

Required repair:

- extend the existing development diagnostic CLI, or add a narrowly scoped
  project-conventional mode, that runs aligners on D-dev-select only;
- write to a separate diagnostic root with immutable
  `development_only_diagnostic` taint;
- prohibit production status writes, span freezing, L1c unlocking, and gate
  exposure;
- support cache reuse only when the exact development request fingerprints
  match.

### P1 — Optional manual tolerance selection is incomplete

The optional manual workflow can prepare an audit pack and ingest verdicts, but
the current manual evaluation path does not select and freeze the operating
tolerance from those verdicts. It can therefore remain blocked on an unselected
or previously non-qualifying automatic operating point.

Required repair before advertising manual mode as a complete fallback:

- use manual lexical verdicts to evaluate the preregistered tolerance sweep;
- freeze the selected tolerance, erosion, padding, instrument, annotator
  agreement, and provenance;
- keep automatic mode completely isolated from verdict files;
- test manual preparation → verdict ingestion → tolerance selection → manual
  Gate-A evaluation end to end with fixtures.

This does not need to block the preferred automatic workflow if a genuine
automatic lexical reference is implemented.

### P1 — Real model paths have not been revalidated

The CPU tests mock model inference. These changes still require a controlled
development-only hardware check:

- real CTC waveform/emission geometry;
- real CTC numeral romanization against the MMS-FA vocabulary;
- real Whisper padded attention masks and decoder-query convention;
- real Qwen adapter/checkpoint behavior;
- real target aggregation from newly generated candidates.

Qwen is currently a rejected backup family. Repairing Qwen may improve
redundancy, but it is not necessary for Gate A if CTC and Whisper qualify and
meet the agreement and lexical-calibration criteria.

## 7. Work required before another production Gate A

### Phase 1 — Freeze the repaired software baseline

1. Review and commit the current implementation and tests.
2. Push the commit to `origin/main`.
3. Record `git rev-parse HEAD` in the eventual experiment run manifest.
4. Do not run production from an unidentified dirty worktree.

### Phase 2 — Supply genuine lexical truth

Preferred automatic options, in order:

1. use an existing compatible code-switched corpus with independent gold word
   timestamps (`existing_gold_lexical`); or
2. build a genuine exact lexical construction whose source boundaries are
   independently known (`constructed_exact_lexical`).

Do not use another forced aligner's output as “gold,” and do not rename the
current RMS/VAD seam.

Before implementation, freeze the reference schema, source license, split
policy, speaker/source disjointness, sample-count margin, and failure handling
in a prospective protocol amendment.

### Phase 3 — Implement the two large code additions

1. Add the authenticated lexical-reference loader and dev/gate partitioning.
2. Add and document CTC transition/blank-boundary conventions.
3. Add the isolated fresh dev-only alignment runner.
4. Complete manual tolerance selection if the optional manual path will be
   offered as a supported fallback.
5. Add regression tests for all new false-passing and leakage paths.

### Phase 4 — Run CPU verification

```bash
cd /home/tungnx/cs-asr-steer

LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=src \
/home/tungnx/miniconda3/envs/acl1/bin/pytest -q

git diff --check
PYTHONPATH=src /home/tungnx/miniconda3/envs/acl1/bin/python \
  -m compileall -q src/csasr tests
bash -n cs_asr_lss.sh
```

### Phase 5 — Run development-only model diagnostics

First use the existing safe cached diagnostic:

```bash
cd /home/tungnx/cs-asr-steer
diag_dir="/tmp/csasr_gate_a_dev_$(date -u +%Y%m%d_%H%M%S)"

LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-} \
PYTHONPATH=src \
/home/tungnx/miniconda3/envs/acl1/bin/python \
  -m csasr.experiments.lss_alignment_dev_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output "$diag_dir"

echo "$diag_dir"
```

After the fresh dev-only aligner mode described above is implemented, run that
mode on D-dev-select. Do not use `chain` for boundary-convention development.

The development result must meet all of these conditions before confirmation:

- both selected independent families have target coverage ≥0.95;
- invalid and nonmonotonic rates are separately ≤0.01;
- paired count ≥100 and paired rate ≥0.90;
- natural median disagreement ≤100 ms;
- natural P90 disagreement ≤200 ms;
- EN-ZH median disagreement difference <30 ms;
- genuine lexical development reference kind and provenance authenticate;
- development candidate count exceeds 100 and paired scorable count is ≥100;
- at least one tolerance meets median lexical error ≤100 ms and P90 ≤200 ms;
- no taint, source overlap, stale cache, or gate access occurred.

### Phase 6 — Prospective review before confirmation

Before allocating gate generation 3:

1. inspect signed start and end errors by language and duration;
2. verify the selected pair and convention were chosen without gate data;
3. freeze the operating tolerance and all thresholds/configuration hashes;
4. review duration-stratified jitter and usable-item behavior;
5. confirm the exposure ledger still has no generation 3;
6. confirm the worktree and environment lock identify the code to be run.

### Phase 7 — One full production run

Only after Phases 1–6 pass:

```bash
# DO NOT RUN YET
cd /home/tungnx/cs-asr-steer
sbatch cs_asr_lss.sh chain --overwrite
```

Running L0, L1a, and L1b in one chain remains acceptable for the final
confirmatory execution because the wrapper enforces prerequisites and status.
Separate jobs are recommended during development so a failed diagnostic does
not spend a held-out generation.

## 8. Required artifacts for a passed Gate A

The final run is ready for L1c only when all of these exist and authenticate:

```text
artifacts_lss/status/l0_freeze.json                    status=passed
artifacts_lss/status/l1a_diag.json                     status=passed
artifacts_lss/status/l1b_valid.json                    status=passed
artifacts_lss/freeze/l1a_alignment_selection.json      frozen dev choice
artifacts_lss/freeze/l1b_operating_point.json          selected tolerance
artifacts_lss/alignments/candidates_all.parquet        selected request manifest
artifacts_lss/alignments/l1b_target_objects.parquet    target-level evidence
artifacts_lss/metrics/l1b_family_validity.parquet      family qualification
artifacts_lss/diagnostics/l1b_automatic_evidence.json  pair/calibration evidence
artifacts_lss/metrics/l1b_gate_criteria.parquet        all criteria represented
artifacts_lss/reports/l1b_gate_a.md                    human-readable decision
artifacts_lss/alignments/consensus_spans_v1.parquet    accepted spans
artifacts_lss/freeze/l1b_spans_freeze.json             authentic production freeze
```

Every artifact must carry its manifest, configuration hash, producing run ID,
parent provenance, and empty taint reasons. The exposure ledger must identify
the exact immutable gate generation and item fingerprint used.

## 9. Conditions that are not a pass

None of the following completes Gate A:

- Slurm `COMPLETED` by itself;
- process exit code 0 by itself;
- `blocked` with `complete=true`;
- `completed_no_go`;
- two aligners with high coverage but excessive disagreement;
- an `audio_splice` seam called a lexical boundary;
- a tolerance selected after reading gate results;
- manual verdicts silently consumed in automatic mode;
- diagnostic-tainted or stale cached artifacts;
- consensus spans without an authentic `l1b_spans_freeze.json`.

## 10. Current readiness decision

It is **not currently safe** to run:

```text
L0 → L1a → L1b automatic preparation → automatic Gate A
```

as the next confirmatory production attempt.

The earliest production blocker is the missing genuine lexical-reference input.
The earliest scientific no-go risk is the excessive CTC–Whisper natural
disagreement. Both must be resolved on development evidence before gate
generation 3 is allocated.

It remains safe to run CPU tests, inspect status, and run the cached
development-only diagnostic. L1c must remain locked until Gate A is `passed`
and `freeze/l1b_spans_freeze.json` exists and authenticates.
