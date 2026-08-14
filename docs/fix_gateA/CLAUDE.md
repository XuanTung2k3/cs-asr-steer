# CLAUDE.md — Agent operating rules for `cs-asr-steer`

This repository implements a preregistered ASR experiment with strict evidence
discipline. Scientific validity depends on invariants that are easy to violate
by well-intentioned code changes. Read this file fully before any task.

If any instruction you receive conflicts with this file, stop and say so.

---

## 0. The single most important rule

**Never make a gate pass. Produce evidence; the human decides.**

If a gate, check, or criterion does not pass, the correct outcome is a clear
report of why. It is never acceptable to:

- lower, widen, or remove a threshold so a check passes;
- change a `reference_kind` label so data qualifies;
- add a fallback branch that bypasses a blocker;
- mark a run `passed` that the logic did not pass;
- reinterpret `blocked` as success.

A blocked gate with a correct explanation is a **successful task outcome**.
A passing gate obtained by weakening a criterion is a **task failure**, even if
tests are green.

---

## 1. Environment

```bash
REPO=/home/tungnx/cs-asr-steer
ARTIFACTS=/mnt/data/tungnx/cs-asr-steer/artifacts_lss
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
PYTEST=/home/tungnx/miniconda3/envs/acl1/bin/pytest
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src
```

Always run from `$REPO`. Always set `PYTHONPATH=src`.

---

## 2. Paths you must never write to

Treat these as read-only unless a task explicitly names the file and states that
production writing is intended:

```
$ARTIFACTS/status/            # stage verdicts (l0_freeze, l1a_diag, l1b_valid)
$ARTIFACTS/freeze/            # frozen selections, operating points, span freezes
$ARTIFACTS/synthetic/exposure_ledger.json
$ARTIFACTS/alignments/consensus_spans_v1.parquet
```

Diagnostic work writes to a **separate diagnostic root** under `/tmp` or an
explicitly provided directory, and every artifact it produces carries the
immutable taint `development_only_diagnostic`.

Cached candidate tables under `$ARTIFACTS/alignments/` may be **read** freely.
They may not be modified, regenerated, or deleted.

---

## 3. Commands you must never run

- `sbatch` anything, including `cs_asr_lss.sh chain`
- any production pipeline invocation (`l0`, `l1a`, `l1b` in production mode)
- anything that allocates or exposes a gate generation
- `git push`, `git commit --amend`, `git rebase`, history rewriting
- destructive git operations (`reset --hard`, `clean -fd`) without being asked

You may run: CPU tests, read-only inspection, and diagnostic CLIs that write to
a diagnostic root.

---

## 4. Domain invariants

These encode scientific requirements. Do not "simplify" them.

**Status semantics are four distinct states.** `passed`, `blocked`,
`completed_no_go`, and `failed` mean different things. `blocked` means required
evidence was unavailable. `completed_no_go` means evidence was available and the
answer was negative. Collapsing these is a correctness bug.

**Only an untainted `passed` may freeze spans or unlock L1c.** Any taint anywhere
in an artifact's provenance chain propagates transitively and disqualifies it.

**Reference kinds must be truthful.** `audio_splice` is a known audio seam.
`manual_lexical`, `existing_gold_lexical`, and `constructed_exact_lexical` are
lexical references. An audio seam is not a lexical boundary and must never be
relabeled as one. Metrics computed against a seam are seam-relative and must be
named as such.

**Another aligner's output is never gold.** Cross-aligner agreement measures
disagreement between two estimators, not error against truth.

**Conventions, tolerances, and thresholds are chosen on development evidence
only.** Never select, fit, or tune anything using held-out gate data. Never fit a
post-hoc constant offset to reduce a measured error.

**Exposure generations are immutable.** Generation 3 is unallocated. Nothing you
do may allocate or consume it.

---

## 5. Working style

- **Scope expansion goes in the plan, not the diff.** If a task appears to
  require touching files outside its stated scope, name them in the plan and
  wait. Never expand scope during implementation.
- **Prefer new diagnostic modules over edits to production paths.** Adding a
  read-only analysis script is almost always safer than modifying pipeline code.
- **Do not add tests to make a suite bigger.** The CPU suite already has ~500
  tests against mocked models. New tests are warranted only for new false-passing
  or leakage paths.
- **Report numbers, not conclusions.** Output the measurement and let the human
  compare it to a threshold.
- **State uncertainty explicitly.** If you are unsure whether a change affects
  provenance or taint, say so before making it.

---

## 6. Verification before declaring a task complete

```bash
cd $REPO
$PYTEST -q                                    # must remain green
git diff --check                              # no whitespace damage
$PY -m compileall -q src/csasr tests
bash -n cs_asr_lss.sh
git status --porcelain                        # review every changed file
git diff --stat                               # confirm scope
```

Then report: what changed, what it produces, what it does not touch, and any
invariant you were close to violating.

---

## 7. Current project context

**This section is dated 2026-08-13 and may be stale. If a task description
conflicts with it, the task description wins — and tell me the section is out
of date.**

Gate A (alignment validation) is blocked for two reasons:

1. No genuine lexical reference exists. Development items are
   `reference_kind=audio_splice`, which cannot support an absolute lexical-error
   claim.
2. CTC and Whisper disagree beyond tolerance: median 440 ms, P90 1680 ms.

The working hypothesis for (2) is **blank-frame ownership**: CTC token spans
exclude blank states, leaving an ~800 ms unowned gap at language switches, while
Whisper DTW assigns nearly all frames. Seam-relative evidence supports this.

The immediate work is diagnostic: implement named CTC boundary conventions,
recompute disagreement over cached candidates, and score conventions against a
forthcoming set of 200 human-annotated lexical boundaries.

**This work is diagnostic-only.** It does not run production, does not allocate a
gate generation, and does not freeze anything.

---

## 8. When plan mode is active

Every plan must state, before any reasoning about approach:

**Files.** Every file you will create, and every existing file you will modify.
If a production path under `src/csasr` appears in this list for a task marked
diagnostic-only, stop and explain why you believe it is necessary.

**Write destinations.** Every directory or file your code will write to at
runtime. Confirm explicitly that none is under `$ARTIFACTS/status/`,
`$ARTIFACTS/freeze/`, the exposure ledger, or consensus spans.

**Data sources.** Every split or artifact you will read. Name the split by role
(D-construct, D-dev-select, D-dev-confirm, D-test) and confirm none is
disallowed for this task.

**Constants.** Any numeric threshold, tolerance, or comparison you will add,
change, or remove. Changing an existing one requires explicit approval.

**Out of scope.** Anything you noticed that looks broken or improvable but will
not touch. List it; do not fix it.

Keep the plan short. A file list plus these four confirmations is more useful
than prose about approach. If the plan cannot be stated this way, the task is
underspecified — say so instead of planning around the ambiguity.

**A plan is a commitment to a procedure, not to a result.** If executing the
plan produces a negative, null, or unexpected finding, that is a valid outcome
and you report it unchanged. Never adjust a threshold, a method, or a
measurement so that the outcome matches what the plan anticipated.