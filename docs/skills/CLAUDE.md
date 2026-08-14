# Claude Code guide for `cs-asr-steer`

Use this file as reusable project context for Claude Code. At the start of a
new session, prompt Claude with:

```text
Read /home/tungnx/cs-asr-steer/docs/skills/CLAUDE.md completely, then inspect
the repository and perform this task: <task>.
```

This file provides context and guardrails. It does not authorize commits,
pushes, Slurm submissions, production runs, held-out-data access, or destructive
operations. The task prompt must grant those actions explicitly.

## Mission

This repository studies selective test-time intervention for Mandarin–English
code-switching ASR. The intended system localizes embedded-English regions,
decides whether intervention is useful, and applies sparse steering to a frozen
Whisper model while measuring correction, corruption, preservation, and cost.

Scientific validity is more important than obtaining a passing gate or a
positive result. A truthful block or negative result is a successful outcome.

## Authority and source order

Resolve conflicts in this order:

1. the current user request and applicable repository instruction files;
2. the actual Git diff, code, configuration, schemas, tests, and authenticated
   artifacts;
3. `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`;
4. `docs/proposal_arr/CS_ASR_ARR_October_2026_Implementation_Plan_v2.md`;
5. dated run reports such as
   `docs/RUN_A_CONVENTION_DIAGNOSTIC_RESULTS_2026-08-14.md` and
   `docs/fix_gateA/ANNOTATION_PACK_IMPLEMENTATION_AND_RESULTS_2026-08-13.md`;
6. older plans, summaries, chat claims, and logs.

Treat dated status text as a snapshot, not live truth. Never trust a summary
when the implementation or artifact says otherwise. Never restore a deleted
file merely because another document links to it.

## Required start for every task

Run from `/home/tungnx/cs-asr-steer`.

1. Read every applicable `CLAUDE.md`, `AGENTS.md`, or equivalent instruction
   file in the repository path relevant to the task.
2. Run `git status --short --branch`.
3. Inspect staged, unstaged, and relevant untracked changes. Preserve unrelated
   user work and identify the exact task diff.
4. Inspect recent history when the change boundary is ambiguous.
5. Classify the task as review, diagnosis, implementation, diagnostic
   execution, production execution, or documentation. Do not silently broaden
   its authority.
6. State the files, write destinations, data roles, constants, and out-of-scope
   items before a nontrivial implementation.

If the task is review-only, do not edit, commit, push, submit jobs, or mutate
artifacts. If the task is diagnosis-only, identify the cause and report it;
implement a repair only when requested.

## Non-negotiable scientific invariants

### Never manufacture a pass

- Do not lower, widen, remove, reinterpret, or bypass a threshold to fit an
  observed result.
- Do not choose a convention, offset, tolerance, family pair, feature, or model
  on held-out evidence.
- Do not introduce a fallback operating tolerance when none qualifies.
- Do not mark production Gate A passed because a diagnostic QA or reproduction
  check passed.
- Do not freeze spans or unlock L1c unless production Gate A is both `passed`
  and untainted.

A human may explicitly amend a diagnostic criterion. Record the decision,
reason, old criterion, new criterion, population, and consequences. Do not
present that amendment as a production threshold change.

### Keep evidence semantics truthful

- Natural aligner-to-aligner differences are
  `cross_aligner_disagreement`, never absolute error.
- An RMS/VAD concatenation point is `audio_splice`: report seam-relative signed
  offsets, not lexical-boundary error.
- Absolute lexical error requires `manual_lexical`,
  `existing_gold_lexical`, or genuinely `constructed_exact_lexical` evidence.
- Never infer or coerce `reference_kind` from a path, filename, or desired
  outcome.
- The target objects are embedded-English span outer boundaries, EN↔ZH switch
  edges, and language runs needed for Site-E—not every Mandarin character or
  tokenizer piece.

### Require genuinely independent evidence

- Production Gate A requires two genuinely independent automatic estimator
  families on natural speech.
- Two variants, aliases, checkpoints, or copies of one estimator class do not
  count as two families.
- Select pairs deterministically from preregistered/configured priority, never
  by best observed gate error.
- Evaluate calibration on the same corresponding qualifying pair and paired
  eligible items. A rejected third family cannot worsen or rescue that pair.
- Fewer than two qualifying families means
  `blocked_insufficient_independent_aligners`; insufficient paired overlap is a
  separate blocker only after two families qualify.

### Preserve raw failures and target-level usability

Keep all three representations with parent provenance:

1. raw aligner items;
2. normalized reference units;
3. aggregated language runs and target spans.

Raw zero/reversed durations and same-language overlaps remain visible even when
an enclosing run has usable outer edges. Same-language internal overlaps may be
merged at run level; cross-language overlap, invalid switch order, or a damaged
required outer edge remains invalid. `invalid_duration` and `nonmonotonic` have
separate definitions and denominators.

### Preserve status and orchestration meaning

- `passed`: the applicable gate passed.
- `completed`: the requested non-gate operation completed successfully.
- `completed_no_go`: valid scientific evidence was obtained and a threshold
  failed.
- `blocked`: required evidence or infrastructure was unavailable, so no valid
  decision can be made.
- `failed`: implementation error, corrupt input, invalid schema, or unexpected
  exception.

Exit code 0 means the command completed; it does not imply a gate pass. Shell
wrappers and downstream stages must read the status artifact. `roles-only` L0
is terminal for its limited operation but never satisfies L1a's full-L0
production prerequisite.

### Preserve provenance, taint, and cache identity

- Manifests must record exact parent artifacts, run IDs, configuration identity,
  request/item fingerprints, thresholds, and transitive taint.
- `--force-prereq` creates immutable diagnostic taint that combines across
  parents and survives retries and cache reuse.
- A timeout must terminate the complete subprocess group, quarantine incomplete
  output, record reason and elapsed time, and produce `blocked` rather than a
  scientific no-go.
- Publish artifacts atomically. Never authenticate a partially written output.
- Cache reuse requires the exact request fingerprint, not merely config or
  model identity.
- Exposure-ledger generations are immutable. Never replace one generation's
  source IDs or reuse a score artifact for a different gate-item fingerprint.

## Split discipline

The LSS roles are speaker/conversation-disjoint:

- `D-construct`: construction evidence;
- `loc-train`, `util-train`, `router-calib`: router/localizer training and
  calibration with their distinct purposes;
- `D-dev-select`: development selection and the current human annotation pack;
- `D-dev-confirm`: final development confirmation only;
- `D-test`: locked test data; tunes nothing.

Default convention diagnostics may read only `D-construct` and
`D-dev-select`. The annotation exporter may read only `D-dev-select`. Do not
read `D-dev-confirm`, `D-test`, or a gate generation unless the current task and
protocol explicitly authorize that use. Push split predicates into data reads;
filtering after materializing held-out rows is not sufficient.

Automatic L1b must never wait for, open, or silently use manual verdict files.
Manual preparation and evaluation are separate explicitly selected paths.

## Protected paths and command boundaries

Repository:

```bash
REPO=/home/tungnx/cs-asr-steer
ARTIFACTS=/mnt/data/tungnx/cs-asr-steer/artifacts_lss
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
PYTEST=/home/tungnx/miniconda3/envs/acl1/bin/pytest
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
```

Unless explicitly authorized for a production task, never write to:

```text
$ARTIFACTS/status/
$ARTIFACTS/freeze/
$ARTIFACTS/synthetic/exposure_ledger.json
$ARTIFACTS/alignments/consensus_spans_v1.parquet
$ARTIFACTS/alignments/candidates_*.parquet
```

Write diagnostics to a new versioned directory under `/tmp` or an explicitly
provided diagnostic root outside `$ARTIFACTS`. Carry
`development_only_diagnostic` taint on every diagnostic artifact and descendant.
Do not overwrite a prior diagnostic run.

Do not run `sbatch`, full datasets, GPU/model inference, downloads, commits,
pushes, rebases, or destructive Git operations unless the user explicitly asks
for that action. Even then, resolve exact targets first and report what will be
affected.

## Current state snapshot — 2026-08-14

Verify this section before relying on it. The authority for the current
alignment criterion is `docs/proposal_arr/RUNBOOK.md` (Runbook v2, 2026-08-14).

### Gate A as currently defined

- **Gate A is a development measurement, not a held-out confirmatory gate.** It
  is a preprocessing audit reported in Experimental Setup. Exposure ledgers,
  immutable gate generations, and held-out accounting exist to protect a
  preregistered claim evaluated once on `D-test`; Gate A consumes none of that.
- **No absolute lexical-accuracy claim is made.** The corpus has no gold word
  timings, and human lexical annotation was deliberately dropped on 2026-08-13.
  Gate A therefore no longer requires a `manual_lexical` reference. The
  100 ms / 200 ms precision targets no longer apply, and the 30 ms EN−ZH bound
  no longer applies *to cross-aligner disagreement* — that bound was specified
  for signed lexical error against gold, which does not exist here.
- **The criterion is a provisional aligner configuration plus an invariance
  battery.** It is met when (1) an aligner configuration is frozen on
  development evidence with a recorded config hash, and (2) the headline
  conclusion survives boundary jitter (±50/±100/±200 ms, separately for English
  and Mandarin spans), all four CTC blank conventions, all independent aligner
  paths, and the pooling schemes (Hann taper, central-60%, uniform full-span)
  with no sign flip. Boundary statistics are reported by language class
  throughout. This certifies invariance, not accuracy, and the paper claims
  only invariance.
- **Three aligner paths are in scope**: CTC forced alignment under a named
  blank-frame convention (characterised); Whisper DTW (reference for
  disagreement statistics); cross-attention pseudo-labels following Liu et al.,
  IEEE TASLP 2025 (arXiv 2403.05887) (to be built).
- **Eligibility now excludes utterance-final targets** (added 2026-08-14),
  alongside the duration tier floor (≥400 ms conservative; ≥300/≥200 ms
  expansion tiers), the frozen confidence threshold, matrix-language preceding
  context, and the non-contiguous reference-unit exclusion.

Redefining Gate A relaxes no invariant in the sections above. Cross-aligner
disagreement is still `cross_aligner_disagreement` and never error against
truth; `audio_splice` is still not a lexical boundary; a convention, path, or
threshold is still a human choice made on development evidence; and no
diagnostic result may be reported as an accuracy claim or as a production pass.

### Measured state

- The last established production chain evidence is job 38745: L0 passed, L1a
  passed, and L1b completed in a truthful blocked state. No production span
  freeze exists and L1c remains locked.
- The Run-A *diagnostic reproduction* gate was re-baselined by explicit human
  decision to the development-only population and passed. This was a QA check on
  the diagnostic harness, not Gate A.
- Four CTC blank conventions were measured on one common set of 3,028
  CTC–Whisper target objects: excluded median/P90 440/1260 ms; to-preceding
  340/1220; to-following 680/1460; midpoint 480/1000. `blank_to_preceding` is
  the **provisional** convention pending the three-way comparison; it is not
  frozen, and it cuts the EN−ZH asymmetry from −440 ms to −140 ms.
- The utterance-final partition contains 600 final and 2,428 non-final targets.
  The to-preceding improvement occurs in non-final targets; final targets
  disagree at 700 ms median and are unchanged by every convention.
- Median blank run at natural language switches is 220 ms. The ~800 ms figure in
  earlier documents was seam-relative on synthetic spliced items — a different
  population. A residual 280 ms median after the best convention and
  utterance-final exclusion remains unexplained.
- Human annotation is dropped. The blinded D-dev-select annotation pack (150
  primary plus 40 designated double-annotation items) is superseded rather than
  failed; its unresolved EN→ZH tier ambiguity is therefore moot, and its
  switch-universe enumeration is reused by the convention diagnostic.
- Held-out gate generation 3 remains unallocated and was not consumed by the
  diagnostic work.
- The latest verified CPU suite contained 559 tests. Recount and rerun rather
  than assuming this number remains current.

Read `docs/proposal_arr/RUNBOOK.md` and the dated result documents named in the
authority section for complete numbers, the day-by-day plan, and limitations.

## Repository map for LSS work

```text
configs/lss/                         thresholds and stage configuration
src/csasr/experiments/lss_l0_freeze.py
src/csasr/experiments/lss_l1a_diag.py
src/csasr/experiments/lss_l1b_valid.py
src/csasr/experiments/lss_status.py
src/csasr/experiments/annotation_pack_export.py
src/csasr/experiments/ctc_convention_diagnostic.py
src/csasr/lss/align/                 aligner evidence and diagnostics
src/csasr/lss/gates.py               status and exit-code contract
src/csasr/lss/manifest.py            artifact authentication and provenance
src/csasr/lss/prereq.py              stage graph and prerequisite checks
tests/                               CPU regression and integration tests
cs_asr_lss.sh                        Slurm/orchestration wrapper
```

Inspect actual imports and callers before assuming this map is complete.

## Implementation and review discipline

- Trace data end to end: raw output → normalization → validity → natural-family
  qualification → pair selection → reference calibration → tolerance selection
  → status → optional span freeze.
- For changed metrics, trace production, serialization, loading, aggregation,
  eligibility filtering, denominators, units, thresholds, and gate use.
- Test semantics, not file existence. A regression test must fail under the old
  incorrect behavior.
- Check false-pass paths explicitly: one family, alias families, missing lexical
  reference, tainted inputs, timeout residue, roles-only L0, stale cache,
  missing jitter, malformed manifest, accidental manual-file use, and wrappers
  that inspect only exit codes.
- Use deterministic seeds and conversation-level statistical units. Keep all
  pieces of one lexical item together. Final intervals use conversation-block
  bootstrap, not item bootstrap.
- Do not add complexity beyond the current task. Split oversized tasks and make
  dependencies explicit.

For a review, report only actionable evidence-backed findings, ordered by
scientific severity. For implementation, use focused tests first, then the full
CPU suite when proportionate.

## Standard verification

```bash
cd /home/tungnx/cs-asr-steer
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src

/home/tungnx/miniconda3/envs/acl1/bin/pytest -q
/home/tungnx/miniconda3/envs/acl1/bin/python -m compileall -q src/csasr tests
bash -n cs_asr_lss.sh
git diff --check
git status --short --branch
git diff --stat
```

Run only checks relevant to the task and state what was not run. Do not launch
models merely to make verification look comprehensive.

## Required handoff

End with:

1. outcome and whether production Gate A changed;
2. files changed and why;
3. evidence and exact commands run;
4. passed, failed, and unrun checks;
5. current blockers separated into code versus scientific/data blockers;
6. exact safe next command, with dangerous or premature commands clearly marked;
7. confirmation that held-out data, production artifacts, thresholds, and
   provenance were not improperly changed.

Never describe a diagnostic pass, successful exit code, completed annotation
export, or improved disagreement number as a production Gate-A pass.
