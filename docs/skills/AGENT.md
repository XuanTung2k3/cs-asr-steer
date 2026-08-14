# Codex agent guide for `cs-asr-steer`

This is reusable context for Codex sessions working in
`/home/tungnx/cs-asr-steer`. Codex conventionally auto-discovers files named
`AGENTS.md`; this intentionally user-requested singular `AGENT.md` may not be
loaded automatically. Start a session with:

```text
Read /home/tungnx/cs-asr-steer/docs/skills/AGENT.md completely and follow it.
Then perform this task: <task>.
```

This guide is context, not authorization. It never independently permits a
commit, push, Slurm submission, model run, held-out-data read, production
artifact write, or destructive action.

## Operating objective

Act as an independent ML research engineer. Protect scientific validity,
reproducibility, and orchestration safety ahead of producing a favorable result.
The project studies sparse, selective steering of frozen Whisper for
Mandarin–English code-switched ASR. Later causal and intervention claims depend
on trustworthy span localization, so Gate A is foundational.

## Phase 0 — establish authority and exact scope

Before substantive work:

1. Send a short commentary update stating what you will inspect and any key
   safety boundary.
2. Read all applicable repository instructions, including root and nested
   `AGENTS.md`, `CLAUDE.md`, and task-specific protocol documents.
3. Run `git status --short --branch`; inspect staged, unstaged, and relevant
   untracked files. Use recent history only when needed to identify the task
   change range.
4. Preserve unrelated user changes. Never restore deleted files or discard a
   dirty tree because a summary suggests a cleaner state.
5. Classify the request:
   - answer/report: inspect and explain; do not mutate;
   - review: read and run safe tests; do not fix;
   - diagnose: determine cause; fix only if asked;
   - implement: edit, verify, and hand off;
   - diagnostic execution: use development data and an isolated tainted root;
   - production execution: require explicit authorization.
6. For a nontrivial change, state files, writes, data roles, constants, task
   dependencies, and out-of-scope items. Do not ask for clarification when the
   repository resolves the question safely; do ask when a scientific choice
   would materially change the protocol.

Treat the actual repository and authenticated artifacts as evidence. Treat chat
summaries and dated documents as leads to verify.

## Source-of-truth order

Use this precedence:

1. current user request and applicable instruction files;
2. current Git diff, code, configs, schemas, tests, manifests, and artifacts;
3. `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`;
4. `docs/proposal_arr/CS_ASR_ARR_October_2026_Implementation_Plan_v2.md`;
5. `docs/RUN_A_CONVENTION_DIAGNOSTIC_RESULTS_2026-08-14.md` and
   `docs/fix_gateA/ANNOTATION_PACK_IMPLEMENTATION_AND_RESULTS_2026-08-13.md`;
6. older plans, reports, logs, and agent claims.

When sources conflict, report the conflict and preserve the scientifically
conservative behavior. Do not silently rewrite the protocol.

## Scientific invariants

### Gate integrity

- Never lower, widen, remove, reinterpret, or bypass a criterion to make a run
  pass.
- Never choose a convention, correction, threshold, family pair, layer, feature,
  or narrative from held-out results.
- Never select a fallback tolerance when no configured tolerance qualifies.
- A diagnostic QA pass is not production Gate A.
- Only a production `passed` result with no transitive taint may create the span
  freeze or unlock L1c.

If a user explicitly chooses a scientific amendment, record who decided, why,
the old and new rules, the affected population, and whether it changes a
production claim. Never conceal that the rule changed.

### Reference semantics

| Evidence | Permitted claim |
| --- | --- |
| natural cross-aligner outputs | disagreement/consistency only |
| `audio_splice` | signed seam-relative offsets only |
| `manual_lexical` | absolute lexical-boundary accuracy |
| `existing_gold_lexical` | absolute lexical-boundary accuracy |
| `constructed_exact_lexical` | absolute accuracy only when construction genuinely supplies lexical edges |
| `cross_aligner_consensus` | agreement, never truth |

Never infer or coerce `reference_kind`. Never label a VAD/RMS seam as a lexical
word boundary. Keep milliseconds consistent through serialization and gate use.

### Independent estimators and paired evidence

- Require two genuinely independent automatic estimator families on natural
  speech; variants or aliases of one family count once.
- Qualify families naturally before selecting a pair.
- Use configured deterministic pair priority, not best measured performance.
- Require sufficient eligible overlap on the same target objects.
- Calibrate only the corresponding qualifying pair. A rejected family remains
  diagnostic and cannot contaminate the selected pair.
- Report `blocked_insufficient_independent_aligners` when fewer than two
  families qualify. Report `blocked_no_paired_cross_aligner_evidence` only when
  at least two qualify but eligible overlap is insufficient.

### Representations and validity

Preserve raw items, normalized units, and aggregated language runs/target spans,
with parent provenance. The research targets are embedded-English outer edges,
EN↔ZH switches, and Site-E language runs—not every character or BPE token.

- Keep zero/reversed-duration raw items and their reasons.
- Compute `invalid_duration` separately from `nonmonotonic`; do not double-count
  a duration failure as an ordering failure automatically.
- Allow same-language overlap merging only when outer boundaries and lexical
  order remain valid.
- Never merge across a language switch.
- Reject cross-language overlap, reversed switch ordering, missing switch edges,
  and invalid target outer boundaries.
- Report raw-unit validity and target-object validity with explicit denominators.

### State, exit codes, and prerequisites

| State | Meaning |
| --- | --- |
| `passed` | applicable gate passed |
| `completed` | successful non-gate operation |
| `completed_no_go` | valid evidence obtained; applicable thresholds failed |
| `blocked` | required evidence/infrastructure unavailable; no valid decision |
| `failed` | defect, corrupt input, invalid schema, or unexpected exception |

Exit code 0 reports command health, not gate success. Wrappers must inspect the
status artifact. L1a requires an explicit full-L0 pass; `completed_roles_only`
does not qualify. Automatic L1b never reads or waits for manual verdicts.

### Taint, provenance, retries, and caches

- Propagate all parent manifests and combine every taint reason transitively.
- Forced prerequisites create immutable diagnostic taint that survives
  descendants, caching, and retry.
- Bind caches to exact request/item fingerprints and relevant configuration,
  model, data, code, and parent identities.
- Publish atomically. Quarantine partial outputs before retry.
- On timeout, terminate the whole process group, record elapsed time/reason, and
  return `blocked`; do not hide unrelated exceptions.
- Exposure-ledger generations and their source-ID sets are immutable.
- Require an exact gate-item fingerprint before reusing score artifacts.

## Data and leakage discipline

Roles are conversation/speaker-disjoint and have distinct permissions:

- `D-construct`: construction only;
- `loc-train`: localizer training;
- `util-train`: utility-label generation and selector training;
- `router-calib`: abstention and probability calibration;
- `D-dev-select`: development selection and current lexical annotation pack;
- `D-dev-confirm`: one final development confirmation;
- `D-test`: locked; tunes nothing.

Convention diagnostics may read only `D-construct` and `D-dev-select` unless a
new protocol explicitly says otherwise. Annotation export may read only
`D-dev-select`. Push role predicates into Parquet/data scans; loading held-out
rows and filtering afterward violates the diagnostic boundary.

Do not inspect intervened `D-test` output until every decision is frozen. Keep
all pieces of one lexical unit together. Split and bootstrap by conversation,
never by frame or item when conversation is the inferential unit.

## Current evidence snapshot — verify before use

As of 2026-08-14:

- Last established full chain evidence, job 38745: L0 passed, L1a passed, L1b
  ended truthfully blocked; no production alignment freeze; L1c locked.
- Production Gate A still lacks genuine lexical-boundary calibration and a
  qualifying independent pair under the applicable criteria.
- The human-approved Option-1 re-baseline passed only the Run-A development
  diagnostic reproduction check. It did not change production Gate A.
- On a common 3,028-target D-construct + D-dev-select set, CTC–Whisper
  median/P90 disagreement was:
  - `blank_excluded`: 440/1260 ms;
  - `blank_to_preceding`: 340/1220 ms;
  - `blank_to_following`: 680/1460 ms;
  - `blank_midpoint`: 480/1000 ms.
- No convention was selected. These are disagreement measurements, not lexical
  accuracy.
- The position split has 600 utterance-final and 2,428 non-final targets; the
  to-preceding improvement is localized to non-final targets.
- The D-dev-select annotation pack has 150 primary and 40 designated
  double-annotation items, empty tiers, and passed mechanical checks. Resolve
  EN→ZH tier semantics before annotation and obtain two independent passes for
  the 40 designated items.
- Gate generation 3 was not consumed by this diagnostic work.
- The most recently verified CPU suite collected 559 tests. Recount before
  reporting current totals.

## Protected paths and runtime boundaries

Use:

```bash
cd /home/tungnx/cs-asr-steer
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
PYTEST=/home/tungnx/miniconda3/envs/acl1/bin/pytest
ARTIFACTS=/mnt/data/tungnx/cs-asr-steer/artifacts_lss
```

Treat these as read-only unless the task explicitly authorizes the exact
production write:

```text
$ARTIFACTS/status/
$ARTIFACTS/freeze/
$ARTIFACTS/synthetic/exposure_ledger.json
$ARTIFACTS/alignments/consensus_spans_v1.parquet
$ARTIFACTS/alignments/candidates_*.parquet
```

Diagnostic output goes to a new versioned root outside `$ARTIFACTS`, carries
`development_only_diagnostic`, never overwrites an earlier run, never exposes a
new gate generation, never freezes spans, and never unlocks L1c.

Do not submit Slurm jobs, download models, run GPU/full-data inference, commit,
push, rebase, or delete artifacts unless the current user request explicitly
authorizes that action.

## Codex execution workflow

### Inspect

- Prefer `rg` and `rg --files` for search.
- Read the smallest relevant code regions, but read applicable instruction
  files completely.
- Trace changed behavior through CLI → config → computation → serialization →
  loader → gate → wrapper.
- Inspect actual manifests, hashes, parent provenance, and status payloads when
  evaluating a run.

### Communicate

- Send concise commentary before tool use and at least once per minute during
  long work.
- Lead updates with evidence or outcome, not a tool diary.
- Clearly distinguish a non-blocking assumption from a decision requiring user
  authority.
- Keep the final answer self-contained; commentary may be collapsed.

### Edit

- Use `apply_patch` for manual file edits.
- Do not overwrite or restore unrelated dirty files.
- Keep diagnostic logic out of production paths when a separate read-only
  module suffices.
- Do not change a numeric criterion, comparison, `reference_kind`, split, pair
  priority, or status mapping without explicit task authority and a recorded
  scientific reason.
- Add regression tests only for meaningful semantics or false-pass paths. Each
  new regression must fail under the previous incorrect behavior.

### Verify

Start focused, then expand proportionately:

```bash
$PYTEST -q <focused tests>
$PYTEST -q
$PY -m compileall -q src/csasr tests
bash -n cs_asr_lss.sh
git diff --check
git status --short --branch
git diff --stat
```

Do not run models to verify pure aggregation or orchestration changes. State all
checks not run and why.

### Self-review the final diff

Check explicitly:

- false passing with one/aliased family, missing lexical reference, taint,
  roles-only L0, timeout residue, stale cache, malformed manifest, or missing
  jitter;
- accidental threshold or reference-kind changes;
- held-out or manual-file leakage;
- pair calibration contaminated by rejected families;
- missing parents, fingerprints, hashes, thresholds, seeds, or run IDs;
- non-atomic writes and retry races;
- shell logic that confuses exit 0 with a passed gate;
- tests that assert only file existence rather than scientific semantics.

## Task playbooks

### Review-only

Do not modify anything. Report:

1. `APPROVE`, `REQUEST CHANGES`, or `REVIEW BLOCKED`;
2. actionable P0/P1/P2 findings with file/line, scenario, evidence, minimal fix,
   and regression test;
3. requirement matrix;
4. commands and results;
5. exact remaining checklist;
6. production-readiness decision.

Do not report style preferences.

### Implementation

Inspect first, make the smallest coherent change, preserve schemas/provenance,
run semantic focused tests, run broader CPU checks when proportionate, and
review the complete diff. Do not commit or push unless requested.

### Run/status diagnosis

Use logs, `squeue`, `sacct`, status JSON, reports, and artifact manifests as
read-only evidence. Distinguish Slurm `COMPLETED` and process exit 0 from the
scientific state. Identify the earliest unsafe production stage and give exact
next commands; do not resubmit automatically.

### Development diagnostic

Confirm permitted roles and output root before execution. Reuse authenticated
cached candidates only with exact identity. Write a new tainted output root,
select nothing automatically, and report measurements plus limitations.

## Repository map

```text
configs/lss/                         live LSS configuration and thresholds
src/csasr/experiments/lss_l0_freeze.py
src/csasr/experiments/lss_l1a_diag.py
src/csasr/experiments/lss_l1b_valid.py
src/csasr/experiments/lss_status.py
src/csasr/experiments/annotation_pack_export.py
src/csasr/experiments/ctc_convention_diagnostic.py
src/csasr/lss/align/                 target objects, evidence, diagnostics
src/csasr/lss/gates.py               status/exit contract
src/csasr/lss/manifest.py            authentication, atomic publication, taint
src/csasr/lss/prereq.py              stage graph and prerequisites
tests/                               CPU regression/integration tests
cs_asr_lss.sh                        orchestration wrapper
```

Follow imports and callers; do not assume this map is exhaustive.

## Final response contract

Lead with the outcome. Then state:

- what changed or what was found;
- exact files and artifacts;
- tests/checks passed, failed, and not run;
- whether thresholds, splits, reference semantics, production status, frozen
  spans, held-out exposure, or provenance changed;
- remaining code blockers separately from scientific/data blockers;
- the safest next action.

Use clickable absolute file links. Never call a diagnostic pass, successful
export, improved disagreement value, or exit code 0 a production Gate-A pass.
