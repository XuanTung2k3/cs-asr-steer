# LSS stage state and exit-code contract

One rule underlies all of it: **an exit code says whether the command worked; a
status file says what it found.** Orchestration reads the status file.

## Statuses

Defined in `csasr.utils.status.ALLOWED`, classified by `csasr.lss.gates.classify`.

| status | meaning | exit |
|---|---|---|
| `passed` | every gate criterion met; this is the only status that unlocks a downstream stage | 0 |
| `completed` | an evidence-preparation run finished. It decided nothing | 0 |
| `completed_no_go` | the experiment ran correctly and a pre-registered scientific threshold was not met. A result, not a crash | 0 |
| `blocked` | evidence the gate requires does not exist or is scientifically incompatible (a missing second aligner, no genuine lexical calibration, a probe timeout). Nothing measured, nothing broken | 0 |
| `awaiting_manual_verdicts` | the *optional* manual audit pack is built and waiting on annotators. The only state that waits on a person | 0 |
| `completed_roles_only` | a partial run (`--roles-only`, `--dry-run`) finished what it was asked to do | 0 |
| `failed` | implementation defect, corrupt artifact, malformed result, or an unhandled exception | 2 |
| `running` / `pending` | not terminal | — |

`complete` in the status file answers **"did this run finish the work it set out to
do"**, not "did the gate pass" — `status` already says that. It is true for every
terminal status except `failed`. It used to be `gate["passed"]`, which recorded a
truthful terminal `blocked` as an incomplete run: the same shape as a job killed
mid-write, which is the case `complete` exists to distinguish. Prerequisites are
unaffected either way, because `prereq._status_problem` rejects any status that is
not `passed` before it looks at `complete`.

Precedence when several apply: **`failed` > `blocked` > `completed_no_go` > `passed`**.

A malformed artifact outranks everything, because no reading of it means
anything until it is repaired. Missing evidence outranks a no-go, because a gate
that never saw its required evidence has not run the experiment and may not
report the experiment's result. The thresholds that *did* fail are still
recorded in `gate["groups"]` and in the criteria table, so the ordering hides
nothing.

**No terminal status other than `passed` satisfies a prerequisite.**
`completed_no_go` is a valid negative, not a licence to continue.

## Exit codes

`csasr.lss.gates.STATUS_EXIT_CODES`.

| code | meaning |
|---|---|
| 0 | ran to completion and wrote a terminal status |
| 2 | `failed` — the only nonzero code a stage produces on purpose |
| 64 | usage error in `cs_asr_lss.sh` |

Consequence for Slurm: a job whose Gate A concluded `completed_no_go` or
`blocked` is reported as **successful**, because it was. `afterok` dependencies
therefore fire on any completed stage; whether the *next* stage may run is
enforced separately and unconditionally by `csasr.lss.prereq`.

## Criterion groups and what a failure in each means

`csasr.lss.gates.GROUP_ORDER`. The group selects the response; the response
ladder is pre-registered in `configs/lss/audit.yaml: failure_response`.

| group | failure means | status |
|---|---|---|
| `mechanical` | artifacts malformed, partition not exact, schema invalid | `failed` |
| `reporting` | a required report was not written | `failed` |
| `external` | measured accuracy/validity threshold not met | `completed_no_go` |
| `jitter` | the mask does not survive ±50/±100 ms boundary error | `completed_no_go` |
| `coverage` | not enough reliable spans to do the science | `completed_no_go` |

## Artifact manifests

Every artifact Gate A depends on is published with a sidecar
`<artifact>.manifest.json` (`csasr.lss.manifest`), written atomically after the
artifact itself:

```json
{"sha256": "…", "complete": true, "rows": 12043,
 "producing_run_id": "l1b_valid/2026-08-11T02:14:41Z",
 "identity": {"source_sha256": "…", "config_sha256": "…",
              "data_sha256": "…", "model_id": "…"},
 "expected_keys_sha256": "…",
 "parent_artifacts": [...], "taint_reasons": [], "diagnostic_only": false}
```

This is what turns "the file is nonempty" into evidence:

* **`sha256`** — a file rewritten after its stage passed is detected, because
  the manifest is the producer's recorded expectation. Hashing a file and
  comparing it to itself proves nothing.
* **`complete` + `expected_keys_sha256`** — a truncated parquet is still a valid
  parquet with fewer rows. The recorded key set is what detects it.
* **`identity`** — a table produced under a different config, model or source
  tree is refused rather than consumed. (`nat5h.RunIdentity` defaults
  `code_commit` to `"unknown"`, so a source snapshot hash is used instead.)
* **`taint_reasons`** — taint is bound to bytes, not to a mutable status.

## Diagnostic taint

`--force-prereq` makes a run **diagnostic**. Both its status and every artifact
it writes carry:

```json
{"diagnostic_only": true,
 "taint_reasons": ["forced_prerequisite"],
 "parent_run_ids": ["/runs/l1a-1"]}
```

* descendants inherit the **union** of their parents' reasons, from statuses
  *and* from the manifests of the artifacts they read;
* `prereq._status_problem` refuses a tainted parent, so a diagnostic can never
  become a production input;
* production Gate A adds `blocked_tainted_inputs` and cannot pass;
* taint is **sticky**, and `--overwrite` cannot launder it: clearing a stage's
  status history does not clear the taint recorded in the artifacts it reuses.
  Only artifacts actually rebuilt come back clean;
* status files written before taint blocks existed are still recognised through
  the legacy `forced_prereq` and `provenance.production_artifact` markers.

## Gate A evidence, and what each source may claim

| source | claim it supports |
|---|---|
| two independent valid aligners | a second opinion exists at all. Counted by *independence class*, so two variants of one estimator are one |
| cross-aligner disagreement on natural speech | how far two estimators differ. **Not boundary error** |
| RMS/VAD synthetic splices (`audio_splice`) | signed offset and gap relative to the known audio seam. **Not lexical absolute error** |
| exact lexical construction (`constructed_exact_lexical`) | true lexical absolute error, only when construction genuinely supplies the lexical start/end edges |
| manual annotation (optional mode) | true absolute boundary error — a person supplied the boundary |
| boundary jitter ±50/±100 ms | whether conclusions survive being wrong by that much |
| frozen high-confidence subset | whether there is enough material |

Two aligners that share a bias agree perfectly and are both wrong; on this data
Whisper-DTW's signed offset relative to the synthetic audio seam was −490 ms while it agreed
with itself to 10 ms across configurations. Cross-aligner agreement is therefore
never substituted for accuracy. The naming rule is enforced in code by
`autoevidence.assert_no_absolute_error_claims`, which fails the run if a
natural-speech criterion is ever named as an error.

Synthetic scoring **is implemented** (`lss_l1b_valid._score_synthetic_sets`), but
job 38573's RMS/VAD-trimmed concatenations establish an audio seam rather than a
lexical word boundary. Their old absolute-error labels were scientifically too
strong. The repaired implementation emits `zh_end_minus_splice_ms`,
`en_start_minus_splice_ms`, `gap_around_splice_ms`, and
`absolute_splice_edge_offset_ms`, and refuses to use those quantities for the
lexical 100/200-ms sub-gate. Exact lexical references remain supported by an
explicit `reference_kind`.

## Configuration selection: development evidence only

Automatic mode never waits on a person. It may choose a configuration only from
**development** items carrying genuine lexical reference edges; the present
`audio_splice` development set cannot make that choice:

| decision | stage | artifact |
|---|---|---|
| Whisper decoder-query convention (`pred_start_offset`) | l1a | `freeze/l1a_alignment_selection.json` + `metrics/l1a_pred_start_sweep.parquet` |
| operating tolerance, erosion, union padding | l1b | requires `manual_lexical`, `existing_gold_lexical`, or genuinely `constructed_exact_lexical`; otherwise blocks before allocating a new held-out generation |

`configs/lss/spec.yaml` literally names a **human** instrument for median ≤ 100
ms and p90 ≤ 200 ms. The proposed automatic substitution is recorded in
`GATE_A_AUTOMATIC_INSTRUMENT_AMENDMENT_2026-08-11.md`; it is valid only for
genuinely known lexical edges and is currently marked superseded/pending such a
reference. **The numbers are unchanged.** `devselect.assert_development_only`
raises if a gate-set row ever reaches selection, and an `audio_splice` row is
ineligible even on development data.

If no swept tolerance meets the rule, the outcome is
`blocked_no_qualifying_operating_tolerance` — never a silent fallback to 200 ms,
because every downstream number would then describe a configuration nobody chose.

## A gate set is confirmatory once

`synthetic/exposure_ledger.json` records the source utterance ids and fingerprint of
every rendered generation. **Evaluating Gate A exposes the generation it read**, so
the next evaluation renders generation *N+1* from sources no earlier generation
used; the score table's manifest records `synthetic_gate_generation` and a later
evaluation refuses a table from an exposed generation however authentic its bytes
are. Sets rendered before the ledger existed are bootstrapped from the item tables
they left behind and recorded as exposed. The **development** set is deliberately
not filtered by exposure: it is meant to be reused, and `partition_sources` keeps
the two pools disjoint.

## Blocked reasons

| code | response |
|---|---|
| `blocked_missing_synthetic_calibration` | re-run the `synthetic` part; the score table is absent |
| `blocked_missing_genuine_lexical_calibration` | obtain manual, existing-gold, or genuinely exact constructed lexical edges; an RMS/VAD audio seam cannot satisfy lexical absolute error |
| `blocked_unauthenticated_synthetic_evidence` | the score table is unsigned, stale or malformed |
| `blocked_missing_development_set` | run l1a; every automatic configuration choice is made on it |
| `blocked_unselected_operating_tolerance` | the selection never ran, or its frozen record is not authentic |
| `blocked_no_qualifying_operating_tolerance` | it *did* run and no tolerance met the rule: repair alignment accuracy |
| `blocked_exposed_gate_set` | no unexposed source audio is left for a fresh confirmatory gate |
| `blocked_uncalibrated_natural_families` | a family qualifies naturally but has no synthetic-gate score |
| `blocked_insufficient_independent_aligners` | repair `whisper_dtw` or bring `qwen_forced_aligner` to full-manifest coverage |
| `blocked_no_candidate_alignments` | run the aligner sweep (`l1b --align`) |
| `blocked_unauthenticated_candidate_evidence` | the candidate table has no valid manifest; re-run the sweep |
| `blocked_exploratory_candidate_source` | only the recorded NAT5H table was available; run the sweep |
| `blocked_no_paired_cross_aligner_evidence` | at least two families qualify naturally, but their eligible target-object overlap is below the recorded count/rate requirement; raw overlap is reported separately |
| `blocked_empty_consensus` | no accepted span reaches the primary confidence bins |
| `blocked_missing_jitter_evidence` | the ±50/±100 ms robustness checks did not run |
| `blocked_missing_language_subset` | one of EN/ZH has no primary spans, so EN−ZH compares nothing |
| `blocked_tainted_inputs` | re-run the prerequisite stages without `--force-prereq` |
| `blocked_missing_manual_verdicts` | annotate the audit pack (manual mode only) |

Every blocker **and** every failing criterion gets an entry in `next_actions`,
ordered by dependency. Annotation appears once, last, marked `optional`: it repairs
a missing human verdict and nothing else, and recommending it for an aligner that
cannot align was misleading.

## Mandatory Gate-A evidence

Every item below is **always constructed** when a gate is evaluated. Evidence
that could not be measured produces a blocker, never an omitted criterion — an
omitted criterion is how a gate passes vacuously.

| evidence | must hold |
|---|---|
| two independent valid aligners | counted by estimator class; a family must clear coverage ≥0.95 **and** invalid ≤0.01 **and** nonmonotonic ≤0.01 |
| paired cross-aligner target objects | ≥ `min_paired_units` and ≥ `min_paired_target_rate` for the deterministic configured pair; raw overlap is diagnostic and distinct |
| alignment coverage | ≥0.95 against the **frozen expected-unit universe** of the roles being labelled, not against the candidate rows |
| lexical absolute error | scored on paired eligible objects for the same naturally qualifying deterministic pair, using a genuine lexical reference from an unexposed gate generation; rejected families cannot contaminate the pair |
| corresponding families | the families that qualify on natural speech and the families with synthetic calibration must be the **same** independence classes, and the disagreement is measured between those |
| jitter | measured and gated at ±50 **and** ±100 ms, on the frozen high/medium subset |
| automatic usable-item rate | ≥0.90 |
| per-role absolute counts | evaluated against the role each threshold is about, from a span table that **carries `role`** |
| span schema | present, monotonic, carrying `role`, and bound to the spec freeze by `spec_freeze_sha256` |
| operating point | selected by the preregistered rule on development items and frozen, or the gate blocks |

## Commands

Automatic production path — no human annotation is required when a genuine
automatic lexical reference exists. It is currently blocked by the available
`audio_splice` reference, so the Slurm commands below must not be run yet:

```
# DO NOT RUN YET: first provide/validate genuine lexical calibration
sbatch cs_asr_lss.sh chain          # l0 -> l1a -> l1b --align, prepare + evaluate
```

`--evaluate-gate` on its own decides from what is on disk, which after a previous
evaluation means the gate generation is already exposed and the run blocks with
`blocked_exposed_gate_set`. To evaluate again, include the `synthetic` part so a
fresh generation is rendered; the default (prepare + evaluate) does.

`chain` is **resumable**: a stage that already passed is skipped, so an
allocation that ran out of wall clock is continued by resubmitting the same
command. It forwards `--force-prereq`, `--overwrite`, `--resume`, `--dry-run`
and `--set` to *every* stage, and rejects any other flag rather than silently
applying it to L0 only.

Optional manual path:

```
sbatch cs_asr_lss.sh l1b --prepare-manual-audit     # -> awaiting_manual_verdicts
#   humans fill artifacts_lss/audit/l1b/verdicts_raw/*.csv
sbatch cs_asr_lss.sh l1b --evaluate-gate --mode manual
```

Automatic mode never opens a verdict file — `--only verdicts` is dropped rather
than honoured. Manual mode never invents one.

Inspecting state:

```
sbatch cs_asr_lss.sh status
python -m csasr.experiments.lss_status --config configs/lss/base.yaml --report
python -m csasr.experiments.lss_status --stage-status l1b_valid   # one word
```

Safe CPU-only development diagnostics from authenticated cached candidates:

```bash
PYTHONPATH=src python -m csasr.experiments.lss_alignment_dev_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output /tmp/csasr_gate_a_dev_diagnostic
```

The command writes diagnostic-tainted JSON/Markdown and manifest sidecars only.
It does not write a stage status, allocate/expose a held-out generation, build or
freeze production spans, or unlock L1c. Before another full chain, this report
must show reachable full-role sufficiency and a credible deterministic natural
pair; a compatible lexical calibration source must also be registered. Gate A
is a task-specific reliability protocol for the span-local experiments in this
project, not a universal forced-alignment standard.
