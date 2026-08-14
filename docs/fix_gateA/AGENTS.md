# Cross-model review prompts

Use after a Claude Code session, before running anything. Give the reviewer a
**fresh context** — no implementation summary, no rationale, no chat history from
the implementing session. It reads the diff against the requirements, cold.

**Review these:** Session 1 (conventions), Session 3 (lexical scoring),
Session 4 (real model paths).
**Spot-check only:** Session 2 (one leakage question, below).
**Skip:** Session 0 (read-only), Session 5 (report).

---

## Universal review preamble

Prepend to every review request.

```
You are reviewing a diff in a preregistered ASR research codebase. Scientific
validity depends on invariants that are easy to violate with clean-looking code.
Read CLAUDE.md in the repo root first.

Your job is NOT general code review. Do not comment on style, naming, docstrings,
or defensive error handling unless they affect correctness of a measurement.

Your job is to find changes that would silently invalidate the science. The
failure mode you are hunting for is code that runs, passes tests, and produces a
number that is wrong or that was obtained by relaxing a constraint.

RULES FOR YOUR REVIEW
1. Verify by running, not by reading. Where a claim can be checked empirically,
   check it. State explicitly which findings you verified and which you inferred.
2. Cite file:line for every finding.
3. If you find nothing wrong in a category, say so explicitly rather than
   omitting it. Silence is ambiguous.
4. Do not propose fixes unless asked. Report findings.
5. If the diff is larger than the task required, say so and list what exceeded
   scope.

Classify each finding:
  BLOCKER  — would invalidate results or violate an invariant
  RISK     — could produce a wrong number under some input
  SCOPE    — outside the task's stated boundary
  NOTE     — worth knowing, not action-forcing

Start by running: git diff --stat, then git diff for every changed file.
```

---

## The seven categories to check on every reviewed session

```
Check each of these explicitly and report on each by name:

1. THRESHOLD INTEGRITY
   Did any numeric threshold, tolerance, comparison operator, or pass condition
   change? Search the diff for changed constants and for flipped or loosened
   comparisons (>= vs >, < vs <=). Any such change is a BLOCKER unless the task
   explicitly requested it.

2. REFERENCE SEMANTICS
   Is reference_kind ever set, inferred, defaulted, or coerced in the diff?
   audio_splice must never become a lexical kind. Metrics computed against a
   seam must be named seam-relative. Report every line that touches
   reference_kind.

3. WRITE BOUNDARIES
   Does anything write to $ARTIFACTS/status/, $ARTIFACTS/freeze/, the exposure
   ledger, consensus spans, or modify cached candidate tables? Trace every file
   write in the diff to its destination. Verify empirically that diagnostic
   output lands outside the production artifacts root.

4. TAINT AND PROVENANCE
   Do all new artifacts carry development_only_diagnostic taint? Does taint
   propagate to derived artifacts? Can any code path produce an untainted
   artifact from a tainted parent?

5. SPLIT DISCIPLINE
   Does any code read from D-dev-confirm, D-test, or a gate generation? Only
   D-dev-select and D-construct are permissible for this work. Trace every data
   source to its split.

6. FITTING AND SELECTION
   Is any constant, offset, correction term, or convention chosen from data
   rather than configured? Search for anything resembling argmin/argmax over a
   metric, or a value derived from measured error. Selection is a human
   decision; the code must only report.

7. SCOPE
   List every file changed. Flag anything outside the task boundary, especially
   edits to production alignment code when the task said diagnostic-only.
```

---

## Session 1 — conventions review

Append after the universal preamble and the seven categories.

```
TASK THAT WAS PERFORMED
Implement four named CTC boundary conventions (blank_excluded,
blank_to_preceding, blank_to_following, blank_midpoint) and recompute
CTC-vs-Whisper boundary disagreement under each over cached candidates.
Diagnostic only. Report numbers; select nothing.

VERIFY EMPIRICALLY
- Run the diagnostic. Confirm blank_excluded reproduces a median disagreement of
  approximately 440 ms and P90 approximately 1680 ms. If it does not, the
  harness is measuring a different quantity than the production path and every
  other number it produces is suspect. This is the single most important check.
- Confirm the four conventions are pure functions of the raw span plus the blank
  run, with no dependence on the measured disagreement.
- Confirm raw token and emission spans are preserved unmutated; conventions must
  be a derived view.
- Construct a small synthetic case by hand (a known blank run between two known
  runs) and verify each convention produces the arithmetically correct boundary.
- Confirm the paired count used is reported and matches the expected ~9,728.

SPECIFIC RISKS FOR THIS TASK
- A convention implemented as an offset subtraction rather than a reassignment
  of blank frames. These differ: one is a fitted correction, the other is a
  principled convention. Report which was implemented.
- Blank-run detection that silently skips cases with no blank run, or with
  multiple blank runs, without reporting how many were skipped.
- Per-language medians computed over different subsets than the overall median.
```

---

## Session 3 — lexical scoring review

```
TASK THAT WAS PERFORMED
Ingest human annotations as reference_kind=manual_lexical and score every CTC
convention and Whisper against genuine lexical truth. Report signed and absolute
errors, language stratification, EN-vs-ZH paired difference with bootstrap CI,
coverage as a function of minimum span duration, and inter-annotator agreement.

VERIFY EMPIRICALLY
- Feed a deliberately malformed annotation (unordered boundaries, missing tier,
  fingerprint mismatch). Confirm it is rejected with a reason, not silently
  dropped or coerced.
- Hand-compute the signed error for one item and confirm the pipeline matches.
- Confirm the bootstrap resamples by conversation, not by item. Item-level
  bootstrap understates the CI because items within a conversation are
  correlated. Report which was implemented.
- Confirm inter-annotator agreement is computed on the 50 double-annotated items
  only, and that those items are not double-counted in the main metrics.
- Confirm sign convention is stated and consistent (aligner minus human).

SPECIFIC RISKS FOR THIS TASK
- reference_kind inferred from file location or format rather than set
  explicitly.
- Annotations silently matched to items by index or filename rather than by
  fingerprint.
- Coverage sweep computed on a different denominator than the error metrics.
- Absolute error reported where signed error is required for the EN-vs-ZH
  asymmetry check — the asymmetry is directional and absolute error hides it.
```

---

## Session 4 — real model path review

```
TASK THAT WAS PERFORMED
Run corrected CTC frame-timing and Whisper attention-mask code paths against
real models on at most 300 D-dev-select utterances, in a dev-only runner writing
to a diagnostic root, and report whether recomputed geometry matches the
cached-candidate diagnostic.

VERIFY EMPIRICALLY
- Confirm the runner cannot write production stage status, freeze spans, unlock
  L1c, or allocate or expose a gate generation. Try to make it do so if the code
  structure permits a test; otherwise trace every write path and report.
- Confirm the empirical ms-per-frame is reported and matches the model's
  expected frame rate.
- Confirm the attention mask actually masks padded regions — inspect a real
  batch, do not trust that the argument is passed.
- Confirm cache reuse requires exact development request fingerprint match, and
  that a deliberately altered fingerprint forces recomputation.
- Confirm the utterance count is capped and the cap is enforced.

SPECIFIC RISKS FOR THIS TASK
- Divergence between real-model geometry and the cached diagnostic being
  reconciled, averaged, or explained away rather than reported.
- Qwen code being repaired or re-enabled. It is a rejected family and explicitly
  out of scope; any Qwen change is SCOPE.
- The runner reusing a production cache path, which would couple diagnostic and
  production artifacts.
```

---

## Session 2 — spot check only

Not a full review. One question:

```
Read CLAUDE.md. Review this diff for exactly one thing: split discipline.

Trace every data source the annotation export reads from. Confirm all 250 items
(200 primary + 50 double-annotated) come from D-dev-select only, and that no
item originates in D-construct, D-dev-confirm, D-test, or any gate generation.

Also confirm the sampling seed is recorded and the sample is reproducible.

Report only on these two points.
```

---

## Handling the review output

**BLOCKER findings:** fix before running anything. Take the fix back to Claude
Code with the specific finding quoted; do not let the reviewer implement it, so
the separation between implementer and checker is preserved.

**RISK findings:** judge yourself. Some are real, some reflect the reviewer not
knowing your invariants.

**SCOPE findings:** usually revert. An agent that touched production alignment
code during a diagnostic task has produced a diff you cannot cleanly reason
about.

**If the review comes back clean on all seven categories:** ask one follow-up —
"which of your findings did you verify by running code, and which did you infer
from reading?" A review that verified nothing is not a review.

**If the reviewer and implementer disagree**, do not arbitrate by asking either
one again. Check the specific claim yourself. This is usually a five-minute
empirical question and both models can be confidently wrong about it.