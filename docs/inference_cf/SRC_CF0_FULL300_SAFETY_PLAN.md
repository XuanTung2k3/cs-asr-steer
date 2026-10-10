# SRC-CF0 — FULL300 active-safety opportunity planning

This is a CPU-only, outcome-blind inventory and future design plan. It is separate from the
SRC-CF0-P correction-power decision. No new FULL300 cached-attention census, model inference, steering, correctness
analysis or safety experiment was run or authorized by this freeze.

## Sources and exact compatibility

Canonical exposed D-dev-select FULL300:300 utterances,20 dialogues x15, original order in
`R0_PANEL.json`, inherited canonical PATH5 theta0 content IDs, audio SHA256 and heard geometry.
Panel identity `sha256:ff28efc645150d122410991d3b2449edc239b26111e97fda09ddf31e9340e2cb`;
membership `sha256:526aa8ada399730379c7c45aaa48755350ca77d5bd4e12d82939c7e9c25858ec`.
`SRC_CF0_FULL300_INVENTORY.json` fingerprints all300 primary R0 JSON/NPZ files against the
pushed R0 output seal, verifies300 original audio hashes, and records structural membership
hashes and primary heard-interval hashes. Reproduce with the CPU-only inventory script.
It opens only PRIMARY predicted-region arrays; no oracle-region or error analysis.

Available: original audio, baseline-generated content IDs/termination, native-LID PRIMARY
heard EN/ZH/U intervals, structural query eligibility and full-replay attention/states.
R0 used PATH5 baseline full replay with use_cache=False. S1 used stepwise native cached
queries. R0's ~99.8% full-replay argmax agreement is NOT a cached-query identity proof;
matching dimensions, or even matching top1, does not prove attention/state equivalence.
Do not substitute R0 full-replay heads/states for S1 cached-query association/interventions.

All180 existing S1 query prefixes match the canonical FULL300 generated prefix[:t], audio
hash and structural query eligibility. Their EXACT sealed S1 attention-derived region
records can therefore be reused in a future census that explicitly adopts the unchanged S1
cached path/model/prompt/preprocessing/head semantics. This establishes a subset lower bound,
not equivalence of full-replay attention. Model/source/environment identity remains a hard
future pre-run audit condition. Do not reuse S1 masked logits as residuals: no DG-02 states
were archived by that runner.

## CPU-computable bounds

| Quantity | Verified count | Meaning |
|---|---:|---|
| Baseline content tokens |13,561|Already generated, no new decode|
| Structurally eligible query positions |12,146|R0 generated-token/UTF8 rule, not region eligibility|
| Compatible S1 subset |180 /80 utterances /20 dialogues|Exact original-prefix membership|
| Target eligibility lower bound |73 /52 utterances /18 dialogues|Existing S1 records, unchanged cached semantics|
| Paired eligibility lower bound |59 /43 utterances /17 dialogues|Existing S1 target/off-target records|
| Region-only target upper bound |10,334 /253 utterances /20 dialogues|Queries in utterances with an EN interval >=4000 heard samples|

The last bound ignores heard-attention mass, association, waveform RMS/energy, off-target
placement and reachability. Pair eligibility is at most target eligibility, so59<=paired<=10,334
is a valid but very loose count bound; it is not an estimate. Zero reachable edits and zero
correct-Mandarin opportunities are still possible: no nonzero lower bound on either exists.
No FULL300 active-correct-Mandarin count can be computed under intended S1 semantics from
these archives. R0 full-replay labels are NOT a substitute.

## Future minimal reference-free census (separate authorization required)

Freeze all12,146 structurally eligible queries in canonical utterance/t order, unique(UID,t).
Eligibility:1<=t<T, baseline-generated next token non-special/non-suppressed, complete UTF8
prefix. This uses baseline token IDs only, not gold token identity. It is recomputed with the
frozen tokenizer; existing R0 eligibility is merely a membership reproduction check.
All20 dialogues retained, no selection based on baseline script, references, success or harm.

For each utterance: original verified waveform, fresh original encoder, forced-ZH prompt,
stepwise cached baseline content prefix. At eligible query capture exact ten-head cached
attention, apply unchanged S1 target/off-target selection to sealed PRIMARY R0 intervals.
The census records eligibility/reasons/crops and source/attention hashes only. No new LID,
masked waveform forwards, direction construction, pulses or lexical evaluation needed.
Reuse180 exact S1 readouts only after hash/semantics audit, without changing membership.
At most300 encoder passes and13,861 prompt/content step calls (13,561 tokens+300 prompt
calls), plus CPU crop integrals. Target5–20min, hard3h, one proposed GPU job. This is NOT
part of the two-job pilot authorization. No job was submitted here.

Seal and push full query population and census BEFORE any future correctness evaluation.
Report target queries, pairs, unique utterances/dialogues, per-utterance repeat counts and
per-dialogue concentration. Subsequent separately frozen construction geometry determines
actually reachable edits. Evaluator-confirmed correct Mandarin is a later isolated join after
appropriate output seal/audit, never a sampling feature or runtime input.

A future expanded-safety study should freeze its candidate query population from inference-
available association/mask constraints only. Default census retains every eligible query;
no reference-dependent deduplication. If a resource-limited intervention subsample is needed,
use a separately frozen deterministic UID/t hash ordering and utterance caps before references,
not an attempt to fill a correct-Mandarin quota. Report how many eligible rows this excludes.
Additional queries within one utterance or20 dialogues do not provide independent trials.

## What would be adequate active Mandarin evidence?

The requested40–60 active correct-Mandarin opportunities across >=10 dialogues is a planning
range, NOT a frozen acceptance rule. An explicit zero-event calculation illustrates its limits:
for n INDEPENDENT Bernoulli opportunities, one-sided95% Clopper-Pearson upper bound is
`1-0.05**(1/n)`: n40 ->7.22%, n60 ->4.87%; >=59 independent zero-event opportunities are
needed for an upper bound <=5%. Four zero-event opportunities give52.71%, not broad safety.
These iid bounds are not valid when treating correlated decoder queries as independent.

At the independent-dialogue level, zero affected dialogues among10 yields25.89% and among20
yields13.91% upper bound on the probability of ANY observed harm in a sampled dialogue.
That estimand differs from per-active-position corruption probability. A zero-event cluster
bootstrap returns a degenerate interval and cannot certify a small harm risk.

Illustrative design-effect sensitivity for60 opportunities across10 equal-size dialogues:
`n_eff=N/(1+(m-1)*rho)` with m6 gives40 (rho.1),24 (rho.3),10 (rho1). This is a planning
sensitivity calculation, not an estimated correlation or a valid confidence certificate.
A future safety specification must freeze the harm estimand, desired bound, independent units,
cluster distribution and interval method before outcomes; it must distinguish unconditional
no-edit behavior from conditional active harm. Pilot safety observations cannot calibrate a
favorable harm threshold. No statistical method manufactures additional independent dialogues.

## Planning decision and stop rules

**Can FULL300 potentially support a defensible safety study? UNKNOWN.** Region-only necessary
coverage is broad enough to justify proposing a small cached-attention census, but neither
exact target/pair eligibility nor reachable correct-Mandarin exposure is known. A promising
opportunity census does not demonstrate direction efficacy; a positive pilot does not supply
adequate safety coverage. Keep both decisions separate.

If future FULL300 cannot support the desired clustered risk statement, do not relax regions or
sample gold-correct Mandarin. Remaining D-dev-select may contain more opportunities, but this
review has not inventoried a larger authorized corpus or additional independent dialogues.
An explicit human decision and reference-free metadata/population freeze are required before
extending development coverage. No D-dev-confirm, D-test, SEAME, transfer corpus, new role or
fresh validation may be used to solve this planning uncertainty automatically.
