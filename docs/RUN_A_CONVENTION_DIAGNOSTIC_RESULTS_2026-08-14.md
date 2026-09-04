# Run A — CTC blank-run boundary convention diagnostic: results

> **STALE POPULATION — 2026-08-14.**
>
> These numbers were computed on the v1 sample under the old role partition.
> Session 8 regenerates the diagnostic on the v2r2 population. The conclusions
> — `blank_to_preceding` best; EN−ZH asymmetry reduced from 440 ms to 140 ms;
> utterance-final targets unaffected by every convention at 700 ms median; and
> blank-run median 220 ms — are expected to hold, but they must be reconfirmed
> before the eligibility rule depends on them. The body below is retained as
> historical evidence of the v1 diagnostic.

**Date:** 2026-08-14

**Repository:** `/home/tungnx/cs-asr-steer`

**Diagnostic root:** `/mnt/data/tungnx/cs-asr-steer/diagnostics/run_a_conventions_2026-08-14`

**Status:** **The Run-A diagnostic reproduction gate passed under the
human-approved Option-1 development-only criterion.** This is a QA gate for the
diagnostic harness, not production Gate A. Production Gate A remains blocked.

The original gate mixed an all-role paired count and P90 with a diagnostic that
is prohibited from reading four of those six roles. On 2026-08-14 the user chose
Option 1: keep the development-only scope and re-baseline the reproduction check
to that population. The reason, exact corrected criterion, and initial result
are preserved in Section 4. The subsequent CPU-only run computed all four
conventions and the utterance-final split; Section 5 contains the complete
common-paired comparison.

---

## 1. Purpose and scientific boundary

This run recomputes CTC-vs-Whisper boundary disagreement under named blank-frame
ownership conventions, over the already-cached candidate table, to test the
hypothesis that CTC token spans exclude blank frames and therefore leave an
unowned gap at a language switch.

**It is diagnostic-only.** It did not:

- run any model or aligner, or request a GPU;
- write, read, or modify any production stage status, freeze file, exposure
  ledger entry, consensus span table, or the cached candidate table;
- allocate, expose, or consume a gate generation;
- build or freeze spans, or unlock L1c;
- change any threshold, tolerance, or comparison operator;
- fit an offset, constant, or correction term.

**No convention was selected.** The published payload records
`convention_selected: null`, and the diagnostic contains no argmin, argmax,
ranking, or recommendation over any metric. Selecting a convention is a human
decision on development evidence, and this document does not make it.

**Completion of this run does not mean Gate A passes.** Gate A remains blocked
for the two reasons recorded in `GATE_A_CURRENT_STATUS_2026-08-13.md`: there is
no genuine lexical-boundary reference, and the two aligners disagree beyond the
frozen natural-agreement limits. Nothing here changes either.

Every artifact this run produced carries the immutable taint
`development_only_diagnostic`, and every one of them is a descendant of the
diagnostic root above. The production artifacts root was verified untouched
(Section 8).

---

## 2. Implementation

### Source files

| File | Status | Lines | Purpose |
| --- | --- | ---: | --- |
| `src/csasr/lss/align/conventions.py` | new (untracked) | 892 | The four named conventions as a derived view: authenticated development-only read, language-run transition inventory with classified blank runs, convention arithmetic, cross-aligner disagreement statistics, and an utterance-final/non-final partition over the same paired frame. |
| `src/csasr/experiments/ctc_convention_diagnostic.py` | new (untracked) | 311 | CPU-only CLI. Authenticates the cached table, refuses output under the production root, builds the payload, publishes four tainted artifacts. |
| `tests/test_lss_ctc_conventions.py` | new (untracked) | 817 | 45 regression tests, including an exact partition check for the utterance-final split. |
| `src/csasr/lss/align/annotation_pack.py` | modified (tracked) | 443 | Behaviour-preserving extraction of the language-run and adjacent-transition helpers so the diagnostic and the human annotation pack share one enumeration. +39 / −10. |

`src/csasr/lss/align/annotation_pack.py` is the only tracked file changed. No
production CTC aligner, Gate-A evaluator, target-object module, or threshold
file was edited.

### Inputs

- Cached candidates: `/mnt/data/tungnx/cs-asr-steer/artifacts_lss/alignments/candidates_all.parquet`
  - sha256 `29d7018d44536abce7afc80e900aa20b21ae5e1508c16f85562b3ccb86e23ce0`
  - produced by `l1b_valid/38745`, manifest taint `[]` (untainted)
  - 265,098 rows total; **76,754 read** (see Section 4.2)
- Permitted roles: `D-construct`, `D-dev-select` only
- Family under convention: `existing_ctc`, variant `existing_ctc/default`
  (the only variant present)
- Reference family: `whisper_dtw`
- No audio, no model, no manifest of any held-out role

### The conventions

For a transition between adjacent language runs with the intervening blank run
spanning `[b_start, b_end]` — `b_start` being the raw end of the preceding run
and `b_end` the raw start of the following one:

| Convention | Preceding run's new end | Following run's new start |
| --- | --- | --- |
| `blank_excluded` | `b_start` (unchanged) | `b_end` (unchanged) |
| `blank_to_preceding` | `b_end` | `b_end` (unchanged) |
| `blank_to_following` | `b_start` (unchanged) | `b_start` |
| `blank_midpoint` | `(b_start + b_end) / 2` | `(b_start + b_end) / 2` |

`conventions.convention_boundary(convention, b_start, b_end)` is the entire rule
and takes nothing else. It cannot consult the reference aligner, a measured
disagreement, or a fitted quantity, because none of them is a parameter.

### Refactoring performed

`annotation_pack._language_runs` became the public `language_runs`, and a new
`adjacent_run_transitions(runs)` returns each consecutive run pair with its
direction and the two units a boundary can come from. `switch_universe` now
sources its neighbour pairs from that helper. Its EN-run-centric outer loop was
deliberately retained so the `non_positive_embedded_span` exclusion counter is
still incremented once per English run rather than once per transition.

Behaviour preservation was verified two ways: the 14 existing annotation-pack
tests pass unchanged, and the pre-refactor function body was reconstructed
verbatim and compared against the refactored one over 400 randomized frames
containing third languages, missing units, invalid units, non-finite edges and
degenerate embedded spans — **identical output frames and identical exclusion
counters on all 400**.

---

## 3. Review findings and resolution

### BLOCKER — held-out candidate rows were loaded before filtering

**Resolved.** The CLI previously called `manifest_mod.read_verified()`, which
materialises the entire parquet, and filtered afterwards.

`conventions.read_development_candidates()` now:

1. authenticates the artifact by its manifest sha256 over the whole file,
   without reading any row;
2. reads with the role predicate pushed into the parquet scan
   (`pd.read_parquet(..., filters=[("role", "in", DEVELOPMENT_ROLES)])`), so a
   held-out row is never materialised into a frame the process can reach;
3. re-asserts `assert_development_only()` on the result, so that if the
   predicate were ever dropped a silent full read could not become a silent full
   analysis.

The manifest's `expected_keys_sha256` is deliberately **not** rechecked, because
recomputing it would require reading every row — including the held-out ones —
to prove something the file sha256 has already proved. The verdict dictionary
records this decision in `key_check_skipped_reason` rather than leaving it
implicit.

Two further changes close the same hole downstream:

- `build_diagnostic()` no longer filters. It calls `assert_development_only()`
  and **raises** on a held-out row. Filtering there would have meant the row had
  already been read, leaving the guarantee resting on that call rather than on
  the read.
- `role_scope()` no longer inspects the source table's role column. The withheld
  roles are named from the **configuration** that produced the table
  (`roles_to_label` in `configs/lss/l1b_valid.yaml`), recorded in the payload as
  `withheld_source: "configuration, not the candidate table's role column"`.

The test that codified post-load filtering was replaced. There are now five
tests covering this: the evaluator refuses a held-out row; the read returns only
permitted roles from a real mixed-role parquet and reports `rows_read` below the
published total; a tampered table is refused before any row is read; withheld
roles are named from configuration; and the end-to-end CLI test publishes a real
mixed-role table and asserts the held-out utterance never reached the
computation.

**This resolution is the direct cause of the gate outcome in Section 4, and the
two requirements are mutually exclusive. See Section 10.**

### RISK — invalid targets falsely counted as moved

**Resolved.** `target_edges_moved()` compared boundaries with `!=`. Because
`NaN != NaN` is true, an invalid target whose boundary was never computed was
reported as moved — including under `blank_excluded`, which by definition moves
nothing. `_same_edge()` now treats two missing edges as equal.

Measured on a fixture with one invalid ZH run, pre-fix versus fixed:

| Convention | Pre-fix | Fixed |
| --- | ---: | ---: |
| `blank_excluded` | **1** | **0** |
| `blank_to_preceding` | 2 | 1 |
| `blank_to_following` | 2 | 1 |
| `blank_midpoint` | 3 | 2 |

The defect was real, not hypothetical. A regression test now drives that fixture
through `target_edges_moved()` and asserts `blank_excluded == 0`.

### RISK — multi-variant CTC inputs corrupt moved-edge and edge-case counts

**Resolved, both halves.**

*Merge.* `target_edges_moved()` reduces both sides with
`representative_targets()` — the same production rule the disagreement
statistics use — then merges on `target_id, utterance_id, language,
aligner_variant` with `validate="one_to_one"`. The merge can no longer go
many-to-many or compare one variant against another, and a violation is a hard
error rather than a wrong number.

*Counts.* `switch_transitions()` now returns its non-transition tallies keyed by
aligner variant, and `edge_case_counts()` and `blank_run_summary()` take a
`variant` argument. The payload restricts both to the variant the metrics
actually use, read back from `representative_targets()` via
`representative_variant()` rather than re-derived, and additionally reports
`family_variant_used` and `family_variants_present`.

Measured on a two-variant fixture, pre-fix versus fixed:

| Quantity | Pre-fix | Fixed | Metric denominator |
| --- | ---: | ---: | ---: |
| `blank_excluded` edges moved | **2** | **0** | — |
| `blank_to_preceding` edges moved | 6 | 1 | — |
| `blank_midpoint` edges moved | 8 | 2 | — |
| `transitions_total` | 2 | 1 | 1 |

Two new tests cover this path, which previously had no coverage.

In this run the point is moot in practice: `family_variants_present` is
`['existing_ctc/default']`, a single variant. The fix matters for correctness of
the code, not for these numbers.

### RISK — reported switch-universe provenance not literally true

**Resolved.** The payload claimed the universe came from both annotation-pack
helpers while the inventory called only `language_runs()`.

`switch_transitions()` now genuinely uses `adjacent_run_transitions()`: for each
consecutive pair of target-language runs it looks up the pack's transition
object and takes the direction and both units from it. Every *actionable*
transition is therefore one of the pack's directly adjacent run pairs, recorded
per row in a new `directly_adjacent_runs` column.

The payload no longer claims identity. It states the relationship precisely in
`switch_universe_difference_from_annotation_pack`: the inventory additionally
enumerates target-language run pairs separated by a third-language run, which
the pack drops silently; those are classified `multiple_blank_runs`, are never
actionable, and exist only so the denominator is visible. The pack also keeps
zero-length and overlapping gaps as eligible items, which this inventory
classifies and excludes from action, so the two populations are related but not
identical.

A new test asserts the general relationship — every actionable transition is in
the pack's universe, all actionable rows have `directly_adjacent_runs = True`,
and the extra rows are exactly `multiple_blank_runs` and all inert. The original
equality test was kept but its docstring now says it holds only for a clean
fixture.

### SCOPE and NOTE findings

The review reported SCOPE clean and raised no NOTE findings. Nothing was
reverted. One item is worth stating explicitly: `--convention` was added to the
CLI during this session because the mandatory gate requires running
`blank_excluded` alone. It always includes `blank_excluded` as the reference,
and the payload records `conventions_computed`, `conventions_available` and
`all_conventions_computed`, so a partial run cannot be mistaken for a complete
one.

### Disagreements left unresolved

None on the findings themselves. All four were accepted and fixed. The earlier
conflict between strict development-only scope and an all-role reproduction
expectation was resolved by the user's explicit Option-1 decision: scope stays
strict, and the Run-A QA gate is defined on the development population.

---

## 4. Run-A reproduction gate — **PASSED on corrected development criterion**

This section records a human decision, not an automatic selection made by the
diagnostic. On 2026-08-14 the user explicitly selected Option 1 from the three
choices in the initial report: retain the strict development-only read and
replace the population-incompatible all-role reproduction criterion.

The corrected criterion is:

1. `blank_excluded` must reproduce the production statistic implementation
   exactly on the identical development-only rows (`n`, median, P90, and
   within-100 ms all equal between the diagnostic and production functions);
2. the overall median, EN median, ZH median, and EN−ZH median difference must
   reproduce the documented central values 440, 240, 680, and 440 ms;
3. every valid development-role target must be paired: the paired count must
   equal the target count derived from the same permitted population.

All three conditions pass: the implementation-equivalence block is entirely
true, the four central values match exactly, and 3,028/3,028 development target
runs are paired. The development-only P90 of 1,260 ms is recorded as the
baseline for future reproduction; it was not retroactively used as a condition
to pass this run. The all-role P90 of 1,680 ms and paired count of 9,728 are
retained below as provenance, but are no longer criteria for a development-only
read.

This does **not** lower a production accuracy or agreement threshold. It does
not make cross-aligner disagreement into accuracy, select a convention, or
change production Gate A's blocked status.

### 4.1 Initial result under the superseded all-role criterion

The initial gate was: run `blank_excluded` only and reproduce median ≈ 440 ms,
P90 ≈ 1680 ms, paired count ≈ 9,728, each within about 5%.

| Criterion | Expected | Obtained | Relative deviation | Verdict (5% tolerance) |
| --- | ---: | ---: | ---: | :---: |
| Median boundary disagreement | 440 ms | **440.0 ms** | 0.00% | **PASS** |
| P90 boundary disagreement | 1,680 ms | **1,260.0 ms** | −25.00% | **FAIL** |
| Paired count | 9,728 | **3,028** | −68.87% | **FAIL** |

Two of three criteria failed in job 39875, so that job correctly stopped before
the other conventions. The result was not changed. Instead, the mismatch was
diagnosed and the user then approved the population-corrected criterion above.

Statistics not part of the gate, against the values recorded in
`GATE_A_CURRENT_STATUS_2026-08-13.md` §5.1:

| Statistic | Documented (all roles) | Obtained (development roles) | Match |
| --- | ---: | ---: | :---: |
| EN median disagreement | 240 ms | 240.0 ms | exact |
| ZH median disagreement | 680 ms | 680.0 ms | exact |
| EN−ZH median difference | 440 ms | 440.0 ms (signed −440.0) | exact |
| Within 100 ms | 8.24% | 8.2232% | 0.02 pp |
| Within 200 ms | not recorded | 23.9432% | — |

### 4.2 What was read

| Quantity | Value |
| --- | ---: |
| Rows in the cached table | 265,098 |
| Rows read | **76,754** (28.9%) |
| — `D-construct` | 30,474 |
| — `D-dev-select` | 46,280 |
| Roles declared by the source stage | `D-construct`, `D-dev-confirm`, `D-dev-select`, `loc-train`, `router-calib`, `util-train` |
| Roles withheld | `D-dev-confirm`, `loc-train`, `router-calib`, `util-train` |
| Utterances with target runs | **600** |
| Target language runs | 3,028 |
| Paired targets available | 3,028 (100% of target runs paired) |

### 4.3 Reason for re-baselining

**The harness is measuring the same quantity as the production path. It is
measuring it over one third of the utterances.**

The evidence that the *quantity* is unchanged:

1. The diagnostic's own pairing was checked against the production function
   `target_objects.cross_aligner_target_agreement` on the identical rows.
   `reproduces_production_statistics` is `true` for every checked statistic
   (`n`, median, P90, within-100 ms) on both the own and common paired sets.
   The headline median, P90 and within-100 ms values in the output *are* the
   production function's return values, not a reimplementation.
2. Four independent statistics reproduce the documented all-role baseline
   exactly: overall median 440 ms, EN median 240 ms, ZH median 680 ms, EN−ZH
   difference 440 ms. A harness measuring a different quantity would not land on
   all four.
3. Within-100 ms reproduces to 0.02 percentage points (8.2232% vs 8.24%).

The evidence that the *population* differs, which is arithmetic and required no
held-out data to establish:

- `configs/lss/align.yaml:88` sets `sample_utterances: 300`, and
  `configs/lss/l1b_valid.yaml:9` labels **six** roles. The candidate table
  therefore covers about 300 × 6 = 1,800 swept utterances.
- This run read two of those six roles and reports exactly **600** utterances
  with target runs — 300 per role, as configured.
- Two roles of six is 33.3% of the utterances. The obtained paired count is
  3,028 of the documented 9,728, i.e. **31.1%**. The shortfall is the split
  restriction, to within the small variation in how many target runs each role's
  utterances contain.

So the paired-count criterion fails **by construction**, not because of a defect.
The expected value of 9,728 comes from
`csasr.experiments.lss_alignment_dev_diagnostic`, which reads every role
including `D-dev-confirm`. A development-only diagnostic cannot reproduce it
without reading held-out rows, which is precisely what the BLOCKER fix in
Section 3 forbids.

The P90 criterion is the one that is not fully explained by arithmetic. The
median is identical to four significant figures while the P90 is 25% lower. Two
things are worth stating and neither is a conclusion:

- The boundary values are quantised to the 20 ms encoder step (440 ms = 22
  frames, 1,260 ms = 63 frames, 1,680 ms = 84 frames). A median sitting in a
  dense central bin is far more stable across subsamples than a P90 sitting in a
  sparse tail, so the two behaving differently under a change of population is
  expected rather than surprising.
- Confirming that the tail difference is attributable to the four withheld roles
  would require measuring those roles. That is exactly the read the BLOCKER fix
  prohibits, so **this run does not confirm it and does not assert it**.

No scientific threshold or production pass condition was changed. The Run-A QA
criterion was re-baselined only after the human chose Option 1, because its
paired-count and tail-quantile expectations came from a different population
that the diagnostic is forbidden to read.

### 4.4 Consequence

After the Option-1 decision, a new CPU-only run computed all four conventions
on the same 3,028-target common paired set. The original stopped run remains
preserved under `gate_blank_excluded/`; it was not overwritten.

---

## 5. Results — four-convention comparison

All rows below use the identical common paired set of 3,028 targets. The own and
common paired sets are identical for every convention, so no convention gains
an advantage by dropping targets. Values are CTC-vs-Whisper disagreement, not
absolute boundary error.

| Convention | n | Median ms | P90 ms | ≤100 ms | ≤200 ms | EN median ms | ZH median ms | EN−ZH ms | Edges moved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blank_excluded` | 3,028 | 440 | 1,260 | 8.2232% | 23.9432% | 240 | 680 | −440 | 0 |
| `blank_to_preceding` | 3,028 | 340 | 1,220 | 10.4029% | 30.4491% | 280 | 420 | −140 | 2,347 |
| `blank_to_following` | 3,028 | 680 | 1,460 | 1.7503% | 8.2563% | 560 | 720 | −160 | 2,347 |
| `blank_midpoint` | 3,028 | 480 | 1,000 | 3.0053% | 12.2523% | 400 | 590 | −190 | 2,979 |

Observation 6's prediction of a modest improvement is confirmed for
`blank_to_preceding`: median disagreement falls 100 ms and P90 falls 40 ms.
The result is not universal: `blank_to_following` worsens both statistics, and
`blank_midpoint` improves P90 while worsening the median. This diagnostic does
not select a convention; selection awaits genuine lexical reference evidence.

### Utterance-final split

The split partitions the same common paired set into exactly one final target
per utterance (600) and 2,428 non-final targets. “Final” is derived from the
maximum language-run ID within each utterance before pairing; it does not alter
any edge.

| Convention | Final n | Final median / P90 ms | Final ≤100 / ≤200 | Non-final n | Non-final median / P90 ms | Non-final ≤100 / ≤200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `blank_excluded` | 600 | 700 / 1,002 | 5.0000% / 12.0000% | 2,428 | 390 / 1,260 | 9.0198% / 26.8946% |
| `blank_to_preceding` | 600 | 700 / 1,002 | 5.0000% / 12.0000% | 2,428 | 280 / 1,220 | 11.7381% / 35.0082% |
| `blank_to_following` | 600 | 780 / 1,600 | 1.3333% / 5.5000% | 2,428 | 660 / 1,460 | 1.8534% / 8.9374% |
| `blank_midpoint` | 600 | 720 / 991 | 2.1667% / 7.1667% | 2,428 | 450 / 1,000 | 3.2125% / 13.5091% |

`blank_to_preceding` leaves the utterance-final block unchanged and accounts
for its improvement in non-final targets, consistent with its moving the end of
the preceding run rather than the start of the following run.

### Edge-case counts

Restricted to `existing_ctc/default`, the variant the statistics use.

| Case | Count |
| --- | ---: |
| `single_blank_run` (actionable) | 2,347 |
| `zero_length_blank_run` | 56 |
| `no_blank_run_units_overlap` | 0 |
| `multiple_blank_runs` | 0 |
| `non_contiguous_reference_unit_index` | 25 |
| `left_unit_cannot_support_an_edge` | 0 |
| `right_unit_cannot_support_an_edge` | 0 |
| `non_finite_boundary` | 0 |
| `transition_at_utterance_boundary` (edges) | 1,200 |
| `same_language_target_runs_merged` | 0 |
| **Transitions total** | **2,428** |
| Target language runs | 3,028 |
| Utterances with target runs | 600 |

The counts reconcile exactly:
`2 × 3,028 − 2 × 2,428 = 1,200`. Every transition is directly adjacent
(`directly_adjacent_runs` is true for all 2,428), split `ZH→EN` 1,233 and
`EN→ZH` 1,195.

### Blank runs at language switches

Over the 2,347 actionable transitions:

| Direction | n | Median blank run | P90 blank run | Total |
| --- | ---: | ---: | ---: | ---: |
| All | 2,347 | 220 ms | 940 ms | 883.0 s |
| `ZH→EN` | 1,199 | 220 ms | 944 ms | 462.8 s |
| `EN→ZH` | 1,148 | 200 ms | 940 ms | 420.2 s |

---

## 6. Verification results

| # | Check | Result | Evidence |
| ---: | --- | --- | --- |
| A | Genuine reassignment, not a fitted offset | **PASS** | `conventions.convention_boundary(convention, b_start, b_end)` takes only the convention name and the blank run's two endpoints; it has no access to the reference aligner, a measured disagreement, or any fitted quantity. A test asserts that two blanks of different width move their edges by different amounts (400 ms and 100 ms), which a constant offset could not do. |
| B | Arithmetic test | **PASS** | Eight parametrized assertions on a hand-computed 400 ms blank spanning [1.0, 1.4]: `blank_excluded` (1.0, 1.4), `blank_to_preceding` (1.4, 1.4), `blank_to_following` (1.0, 1.0), `blank_midpoint` (1.2, 1.2) — tested both on the pure function and end-to-end on the frame. A separate test proves reassignment never chains through a shared run. |
| C | Raw spans preserved | **PASS** | `apply_convention()` copies the input frame; a test asserts frame equality before and after all four conventions. `blank_excluded` is bit-identical to the cached geometry across `start_sec`, `end_sec`, `start_sample`, `end_sample`. The cached candidate table was verified unmodified after the run (Section 8). |
| D | Same paired set | **PASS** | The CLI intersects the paired-target sets across all four conventions and reports both the own set and the common set, labelled `paired_set` in the output table. Every convention has 3,028 targets in both views. |
| E | Per-language subsets | **PASS** | `disagreement_summary()` groups the already-merged paired frame, so language strata partition the overall paired rows. Verified in the output: 1,384 EN + 1,644 ZH = 3,028. A test asserts the strata sum to the total. |
| F | Edge cases | **PASS, with the universe caveat stated** | Eight classifications plus two non-transition cases, all reported with explicit zeros rather than omitted keys (Section 5). The inventory is wider than the annotation pack's eligible universe in one direction only, stated in the payload rather than implied — see the fourth review finding in Section 3. |
| G | Utterance-final partition | **PASS** | The label is computed from the maximum run ID per utterance on the already-built targets. The output partitions 3,028 common paired targets into 600 final and 2,428 non-final rows for every convention; a regression test asserts one final target per utterance and exact denominator reconciliation. |

Two additional verifications performed this session:

- **Reproduction of the production statistic on identical rows.**
  `reproduces_production_statistics` is `true` for `n`, median, P90 and
  within-100 ms, on both paired sets.
- **Both RISK defects shown to be real.** The pre-fix implementations were
  reconstructed and run beside the fixed ones on the fixtures that expose them;
  both reported `blank_excluded` moving edges, which is impossible by definition
  (Section 3).

---

## 7. Tests and static checks

```text
pytest -q
559 tests reached 100%, exit 0

pytest -q tests/test_lss_ctc_conventions.py
45 passed, exit 0

pytest --collect-only
559 tests collected

python -m compileall -q src/csasr tests
PASS

bash -n cs_asr_lss.sh
PASS

bash -n run_diagnostic.sbatch
PASS

git diff --check
PASS
```

The full invocation covers all 559 collected tests. The new test in this update
checks that the utterance-final split is a true partition of the exact paired
set rather than a newly paired subset.

The focused and full suites were run on a frozen source/test working tree. This matters here:
`utils.provenance.source_snapshot_hash()` hashes everything under `src/`,
`configs/` and `tests/`, so editing a source file while the suite runs changes
the snapshot mid-run and causes spurious identity-mismatch failures. Earlier
runs in this session showed exactly that and were discarded.

```text
git status --porcelain
 M src/csasr/lss/align/annotation_pack.py
?? docs/RUN_A_CONVENTION_DIAGNOSTIC_RESULTS_2026-08-14.md
?? docs/fix_gateA/AGENTS.md
?? docs/fix_gateA/CLAUDE.md
?? src/csasr/experiments/ctc_convention_diagnostic.py
?? src/csasr/lss/align/conventions.py
?? tests/test_lss_ctc_conventions.py

git diff --stat
 src/csasr/lss/align/annotation_pack.py | 49 +++++++++++++++++++++++++++-------
 1 file changed, 39 insertions(+), 10 deletions(-)
```

---

## 8. Commands, hashes, and provenance

### Initial single-convention execution

The initial evidence was produced by one CPU-only Slurm job before this update:

```bash
cd /mnt/data/tungnx/cs-asr-steer/diagnostics/run_a_conventions_2026-08-14
sbatch run_diagnostic.sbatch \
  --config lss/l1b_valid.yaml \
  --diagnostic-output /mnt/data/tungnx/cs-asr-steer/diagnostics/run_a_conventions_2026-08-14/gate_blank_excluded \
  --convention blank_excluded
```

which executed, inside the job:

```bash
cd /home/tungnx/cs-asr-steer
export PATH=/home/tungnx/miniconda3/envs/acl1/bin:$PATH
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tungnx/cs-asr-steer/src
export PYTHONHASHSEED=0
python -m csasr.experiments.ctc_convention_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output .../gate_blank_excluded \
  --convention blank_excluded
```

| Item | Value |
| --- | --- |
| Slurm job | `39875`, partition `main`, node `worker-0`, no GPU |
| State / exit | `COMPLETED`, `0:0` |
| Start / end (UTC) | `2026-08-14T04:51:42Z` / `2026-08-14T04:52:12Z` |
| **Wall clock** | **30 s** (Slurm elapsed `00:00:31`) |
| Python | 3.11.9 (`acl1`) |
| Git SHA | `1ced51bd2ec09b2819b66f1576e4a944baa510e0` |
| Working tree at run time | dirty — the five files listed in Section 7 |
| Config hash (`config_sha256`) | `3d02ecd1dbfdebc630c2bc70aa67016e66d947ddde541f4927ba0675b86b38b1` |
| Job log | `logs/ctc_conv_diag_39875.out` (stderr empty) |

The working tree was dirty at run time because the diagnostic and its tests are
not committed. The job log records `git status --porcelain` so the exact
uncommitted set is part of the run's provenance.

### Four-convention execution with utterance-final split

After the user selected Option 1, the complete diagnostic was run directly on
CPU. The pre-existing `all_conventions/` directory was not overwritten; this
run used a new immutable sibling directory.

```bash
cd /home/tungnx/cs-asr-steer
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
export PYTHONHASHSEED=0
/home/tungnx/miniconda3/envs/acl1/bin/python \
  -m csasr.experiments.ctc_convention_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output \
  /mnt/data/tungnx/cs-asr-steer/diagnostics/run_a_conventions_2026-08-14/all_conventions_with_utterance_final
```

| Item | Value |
| --- | --- |
| Execution | direct CPU process; no Slurm submission and no GPU |
| Exit | 0 |
| Wall clock | 1m55.207s |
| Completed (UTC) | 2026-08-14T05:16:20Z |
| Config hash | `3d02ecd1dbfdebc630c2bc70aa67016e66d947ddde541f4927ba0675b86b38b1` |
| Candidate parent | `l1b_valid/38745`, manifest sha256 `29d7018d4453...` |
| Roles read | `D-construct`, `D-dev-select` only; 76,754 rows |
| Conventions | all four; `convention_selected: null` |

### Outputs

The initial stopped run remains under
`.../gate_blank_excluded/`. The complete run is under
`.../all_conventions_with_utterance_final/`:

| Artifact | Rows | Purpose | Taint |
| --- | ---: | --- | --- |
| `ctc_boundary_conventions.json` | — | machine-readable payload, including the positional split | `development_only_diagnostic` |
| `ctc_boundary_conventions.md` | — | rendered evidence report | `development_only_diagnostic` |
| `ctc_convention_disagreement.parquet` | 8 | four conventions × own/common paired views | `development_only_diagnostic` |
| `ctc_switch_transitions.parquet` | 2,428 | immutable transition inventory | `development_only_diagnostic` |

Their repository hash values (the project hash includes file size before the
bytes) are `1de39c9ae848`, `294e87a645d1`, `af430cd44298`, and
`fffeb6272de5`, respectively.

For provenance, the artifacts in the initial stopped run were:

All under `/mnt/data/tungnx/cs-asr-steer/diagnostics/run_a_conventions_2026-08-14/gate_blank_excluded/`:

| Artifact | Rows | sha256 (12) | Taint |
| --- | ---: | --- | --- |
| `ctc_boundary_conventions.json` | — | `6740ee09e196` | `development_only_diagnostic` |
| `ctc_boundary_conventions.md` | — | `74bd47408637` | `development_only_diagnostic` |
| `ctc_convention_disagreement.parquet` | 2 | `b8c72232b492` | `development_only_diagnostic` |
| `ctc_switch_transitions.parquet` | 2,428 | `fffeb6272de5` | `development_only_diagnostic` |

Each carries `diagnostic_only: true`, `production_gate_evaluated: false`,
`held_out_gate_generation_consumed: false`, `production_spans_frozen: false`,
`l1c_unlocked: false`, `convention_selected: null`, and names
`candidates_all.parquet` as its parent artifact, so taint propagates through the
repository manifest contract.

### Production artifacts root — verified untouched

```text
find /mnt/data/tungnx/cs-asr-steer/artifacts_lss -newermt "2026-08-14 00:00" -type f
(no output)

artifacts_lss/synthetic/exposure_ledger.json      mtime 1786502490  (unchanged)
artifacts_lss/alignments/candidates_all.parquet   mtime 1786496382  (unchanged)
```

No file under the production artifacts root was created, modified, or deleted.
Gate generation 3 remains unallocated. `freeze/alignment_freeze.json` remains
absent.

---

## 9. Observations

Stated as measurements. No convention is recommended, and the choice is not made
here.

1. **The reference geometry reproduces on the development subset.** Median
   440 ms, EN 240 ms, ZH 680 ms, EN−ZH 440 ms — all four match the documented
   all-role values exactly. Within-100 ms is 8.2232% against a documented 8.24%.

2. **The gate's paired-count expectation and development-only scope are
   arithmetically incompatible.** The table covers six roles at 300 utterances
   each; this run read two of them and measured 600 utterances and 3,028 paired
   targets against a documented all-role 9,728. 31.1% obtained versus 33.3%
   expected from the role fraction alone.

3. **P90 is the one statistic that moved without an arithmetic explanation.**
   1,260 ms here against 1,680 ms documented, while the median is unchanged.
   Boundary values are quantised to the 20 ms encoder step, so a tail quantile
   is far more sensitive to a change of population than a central one. This run
   neither confirms nor rules out any other cause, because doing so would
   require reading the withheld roles.

4. **The ZH-versus-EN asymmetry is large and directional.** ZH runs disagree
   with Whisper by a median 680 ms against EN's 240 ms; signed EN − ZH is
   −440 ms. The frozen limit on this difference is 30 ms.

5. **Start and end edges disagree by different amounts.** Median start
   disagreement 140 ms, median end disagreement 300 ms, over the same 3,028
   paired targets.

6. **The blank runs are smaller than the hypothesis's ~800 ms.** At the 2,347
   actionable language-run transitions on natural D-construct and D-dev-select
   speech, the median blank run is 220 ms and the P90 is 940 ms; the total
   unowned audio is 883 s. The ~800 ms figure in
   `GATE_A_CURRENT_STATUS_2026-08-13.md` §5.1 is a seam-relative measurement on
   *synthetic spliced* items, which is a different population; the two numbers
   are not directly comparable and this run does not reconcile them. What can be
   said is that on this population, a median blank run of 220 ms is smaller than
   the 440 ms median disagreement it was hypothesised to explain.

7. **The switch universe is almost entirely clean.** Of 2,428 transitions,
   2,347 (96.7%) have exactly one blank run and are actionable; 56 (2.3%) are
   zero-length, where all four conventions coincide; 25 (1.0%) are
   non-contiguous. There were no overlapping runs, no unusable edges, no
   non-finite boundaries, no third-language interruptions, and no merged
   same-language runs. Every transition was directly adjacent, so on this
   population the inventory and the annotation pack's universe do not diverge.

8. **Blank reassignment changes the disagreement, but no single convention
   dominates every statistic.** `blank_to_preceding` has the lowest median
   (340 ms) and improves P90 modestly (1,220 ms); `blank_midpoint` has the
   lowest P90 (1,000 ms) but a worse median than the reference (480 ms).
   `blank_to_following` worsens both. These are not accuracy results and the
   diagnostic selects none.

9. **The utterance-final split localises the preceding-rule improvement.** Its
   final-target block is unchanged at median/P90 700/1,002 ms, while the
   non-final median falls from 390 to 280 ms. This is consistent with the
   convention moving preceding-run ends, but does not establish lexical
   correctness.

---

## 10. Open issues and next action

### 10.1 Option 1 is complete

The development-only reproduction criterion is recorded in Section 4 and
passes. The three additional conventions and utterance-final split are complete.
No all-role diagnostic was run, and no held-out row was read.

### 10.2 A convention still cannot be selected from this evidence

The median blank run at a language switch on this population is 220 ms, while
the median CTC-Whisper disagreement is 440 ms and the ZH median is 680 ms.
Reassigning a 220 ms median blank run cannot by itself account for a 440 ms
median disagreement. Either the mechanism is not the dominant term on this
population, or the ~800 ms seam-relative figure and this 220 ms figure are
measuring different things — the seam figure comes from synthetic spliced items
and this one from natural speech. This is worth resolving before a convention is
selected on the strength of the blank-frame hypothesis.

### 10.3 Unchanged production blockers

Gate A remains blocked for the reasons in
`GATE_A_CURRENT_STATUS_2026-08-13.md`. No genuine lexical reference exists yet;
the annotation pack prepares that evidence but does not supply it, and its EN→ZH
tier-semantics ambiguity is still unresolved. Nothing in this run changes the
production status, and no convention was selected.
