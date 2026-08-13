# Gate-A Automatic Instrument Amendment — 2026-08-11

Status: superseded; not valid for the current RMS/VAD splice harness.

Amendment ID: `gate-a-automatic-instrument-amendment-2026-08-11`

## Disclosure and timing

This is an explicit post-freeze amendment. It does not alter or reinterpret the
text of the sealed v1 specification. The sealed rule names a human boundary
audit as the instrument used to select the largest consensus tolerance whose
accepted spans meet median absolute boundary error at most 100 ms and 90th
percentile absolute boundary error at most 200 ms.

The amendment was written after inspection of job 38573. Consequently, the
synthetic gate items measured by that job are exposed development evidence and
cannot support a new confirmatory conclusion. The exposure ledger must reserve
their source recordings permanently. Only a newly allocated, unexposed gate
generation may evaluate the amended automatic workflow.

## Scientific correction (2026-08-12)

Inspection of the actual renderer showed that the construction does not provide
the estimand stated below. It uses RMS/VAD energy to trim two source clips and
then concatenates them. The concatenation sample is exactly known as an audio
seam, but neither source clip has an independently known lexical word boundary.
The earlier text's claim that this was an interchangeable instrument for lexical
absolute error was therefore incorrect.

The code now records `reference_kind=audio_splice` and emits only signed
seam-relative ZH-end and EN-start offsets. It cannot select an operating
tolerance or satisfy automatic lexical calibration from those measurements.
Automatic Gate A remains blocked until a `manual_lexical`,
`existing_gold_lexical`, or genuinely `constructed_exact_lexical` reference is
available. The 100/200 ms thresholds are unchanged.

The following rule is retained as historical proposed text, but is not active
for `audio_splice` evidence.

## Superseded proposed automatic rule

For automatic mode only, the proposal had selected the operating tolerance on
the synthetic development set:

1. Sweep the frozen candidate tolerances.
2. Construct consensus spans independently at each tolerance.
3. Score the worse of the start and end unit-edge errors for each eligible
   synthetic-development item.
4. Exclude a tolerance with fewer than the configured minimum number of
   scorable boundaries.
5. Select the largest remaining tolerance with median absolute error at most
   100 ms and 90th-percentile absolute error at most 200 ms.
6. If no tolerance qualifies, report
   `blocked_no_qualifying_operating_tolerance`; do not use a fallback tolerance.

The selected configuration is then judged on a disjoint, freshly allocated
synthetic gate generation. The exact development artifact, gate generation,
alignment-request fingerprints, thresholds, seeds, and this amendment's SHA-256
must be recorded in the operating-point and score manifests.

## What does not change

- The 100 ms median and 200 ms 90th-percentile limits are unchanged.
- Coverage, invalid/nonmonotonic, usable-item, EN–ZH, independence, and
  ±50/±100 ms jitter requirements are unchanged.
- Natural aligner disagreement is not absolute boundary error.
- At least two corresponding independent automatic aligner families must
  qualify on natural speech and be calibrated on the synthetic gate set.
- Missing or tainted evidence remains blocking and cannot be replaced by a
  passing default.
- The optional manual workflow retains the literal human-audit instrument and
  remains a separately requested second opinion.

## Interpretation

Results under this rule must be described as following the
`gate-a-automatic-instrument-amendment-2026-08-11` protocol, not as an unchanged
execution of the original sealed human-instrument rule. Job 38573 remains a
diagnostic predecessor and cannot be reclassified retrospectively.
