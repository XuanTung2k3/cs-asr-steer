# TTLS-R1 independent scientific audit

**Final verdict: `AUDIT_FAIL_INVALID_RESULTS`.**

The saved free-decoding hypotheses, published metrics, frozen exploratory label and targeted replay are reproducible.
The failure concerns **TTLS's zero-activation baseline and clean-start causal attribution**, not fabricated metrics,
reference leakage, A2's historical validity or corrupted output files. With `z=0` and the actual TTLS ALL mask, the
native BF16 repair alters decoder states and changes 6/100 token sequences, including four normalized transcripts.
That no-optimization control changes 89 mixed/ZH errors. The reported correction–preservation advantage cannot be
attributed solely to optimized representations under a matched clean initialization.

Even setting this defect aside, these results do not establish a TTLS advantage over decoder-LN TTA: TTLS-CE's eight
POI corrections are segmentation/deletion/EOS effects; preservation-regularized A2 keeps substantially more benefit;
and TTLS-AC contributes no normalized-transcript benefit beyond direct candidate substitution.

Original files, outputs, seals and `TTLS_R1_MIXED` verdict remain unchanged. This is a separate audit on branch
`audit/ttls-r1-independent`, worktree `/home/tungnx/cs-asr-steer-ttls-audit`. No Round 2 was started.

## 1. Exact reproducibility state and audit scope

| Item | Verified identity |
|---|---|
| Repository | `https://github.com/XuanTung2k3/cs-asr-steer.git` |
| Completed experimental revision | `800fee26a55651c307a1a33b066fdb47ece22b05` (`feature/ttls-r1`) |
| Scientific freeze | `770ee86e` |
| Executed source revision | `90547f4b1dd827358cb09d0c02f2a143899adacb` |
| Resolved manifest committed | `8a048c1f` |
| Primary job | Slurm `58385`, MIG H100 3g.40gb, runtime 508.278 s |
| Primary output seal commit | `42e6848f`, before report/evaluation commit `800fee26` |
| Manifest hash | `sha256:8c0b5831f601396da829fd1ddb4f7b63a24831019ee71adc39f7368481a28145` |
| Plan hash | `sha256:61ba6a5216e12e683e82ae0a8ca405cbe67a450bfa28e4c13682ef244169e31e` |
| Config SHA256 | `8270f6636674d4397601ad9ba0737628834f707a2c8e710c91c15722a152f764` |
| Fixed100 panel SHA256 | `266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5` |
| Model weights SHA256 | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| Tokenizer JSON SHA256 | `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca` |
| Environment | Torch 2.10.0+cu128, Transformers 4.57.6, NumPy 1.26.4; BF16 eager forward, greedy, cap 200 |
| Seed | 240924 |
| Data | Exactly 100 already-exposed D-dev-select utterances, 20 dialogues × 5; IDs/order in sealed plan |

The model is the local `/mnt/data/tungnx/whisper-large-v3` snapshot. An upstream Hugging Face
commit revision is not recorded in the original manifest; reproducibility is pinned to the complete local-file SHA256
set, rather than an invented remote revision.

All manifest source pins, all 11 local model/tokenizer/preprocessor file hashes, every raw-output seal file,
100 original audio full SHA256 values, and all 100 historical A2 row/archive SHA256 values were checked. Manifest,
plan and seal self-hashes were recomputed independently. Exact IDs and replay IDs are retained in the audit JSONs;
there is no reconstructed/resampled evaluation population.

Reference source: the canonical dialogue-v2r3 `role_D-dev-select.parquet`, filtered to these 100 IDs. Its SHA and a
hash of the exact ID→reference mapping are recorded in `independent_metrics.json`. Dialogue/role identities match
the sealed plan. No D-dev-confirm, D-test, router-calib, P3 or transfer data was used.

I reviewed AGENTS, METHOD_CONTRACT, DATA_EXPOSURE, CODE_MAP, the TTLS spec/config/source/runner/evaluator/manifests,
canonical normalization/alignment/retention primitives, TTA1/A2 and DIR-SPRINT0 D3 source/reports. TTLS is a separately
frozen exploratory study, not an implementation of the core-v6 basis/controller method. The superseded first
manifest and its ancestor-check repair remain preserved; the execution source guard change does not alter the method.
Git history verifies seal-before-evaluation ordering. Historical push timing and the assertion that no earlier
unrecorded reference inspection occurred cannot be established independently from Git commits alone.

## 2. Implementation and fairness findings

### Verified implementation properties

- **Frozen TTLS backbone:** model parameters have `requires_grad=False`; TTLS optimizes only one fresh FP32 1,280-D
  vector. Native BF16 casts preserve an autograd path through the DG-02 intervention. Losses, nonzero finite gradients
  and optimized-vector replay agree exactly. No detached tensor was found blocking TTLS learning.
- **A2 trainables:** `decoder_ln_names` selects exactly 194 decoder LayerNorm affine tensors (248,320 scalars).
  FP32 masters, BF16 `functional_call`, AdamW lr 1e-3, wd 0, betas .9/.999, eps 1e-8, two updates. The historical
  unregularized CE actuator reconstructed from theta0 matches archived effective parameters bitwise on all 24 replay
  rows and reproduces its historical tokens/termination.
- **TTLS optimization:** AdamW lr `1.1260757575454359/(2*sqrt(1280))`, same remaining optimizer settings, two updates,
  three evaluations; fresh zero vector/optimizer per arm/utterance. Post-step projection enforces the vector norm
  budget, but never binds in the saved run. A vector norm constraint is not an exact per-position realized-chord bound.
- **DG-02 site:** captures the incoming cross-attention residual and attention output and edits their sum before
  `final_layer_norm`/FFN, under eval/dropout-off. No post-FFN edit or depth scaling. Independent native FFN pre-hooks
  confirm consumption of the optimized edit, zero direct displacement outside the declared query mask on the same
  teacher-forced history, and maximum measured relative norm-rounding error 0.007513 on replayed states.
- **Positions:** query `3+t` predicts content t; absolute positions below four are excluded. Thus TTLS never edits the
  first content prediction. AC targets one frozen step, CE uses all editable steps including the eventual EOS query.
- **Isolation:** teacher-forced updates use `use_cache=False`; each free decode creates its own fresh Branch/cache.
  Context hooks are removed in finally/exit paths. Resident LN is restored after each LN decode; no cross-branch or
  cross-episode stale cache was found. Replay non-LN hashes, LN reset and parameter-gradient checks pass.
- **Leakage:** pseudo-transcripts, UTF-8 validity, script partition, null-audio support, candidate/stable masks all use
  theta0 outputs. No gold transcript/regions/target labels enter adaptation. A3/D/A memberships are carried as report
  metadata, not used to choose the update. Replay runner has no reference loader.
- **Final decode:** uses the learned vector through the hook at the correct cached absolute positions. This is actual
  free decoding, not a teacher-forced prediction table relabeled as an ASR result.

### Material numerical defect: a zero TTLS vector is not a clean start

The historical kernel computes, in BF16:

```python
steered = hidden + delta
steered = steered / (new_norm + EPS) * orig_norm
```

When `delta=0` but gain is one, divide-then-multiply still rounds the residual. The alpha-zero and empty-mask paths
bypass this code and remain exact; **an empty-mask test does not test the actual ALL-mask zero-vector initialization**.
The primary run logged a max logit difference of 0.125 on its first row but asserted only empty-mask token identity.
It did not stop or measure the all-mask zero-vector free-decoding result.

The independent 24-row replay reproduces the original numerical behavior exactly, while the zero-vector control
changes three row outputs. Teacher-forced logit differences reach 0.25; zero-vector native FFN chords reach 0.022967.
These are systematic extra BF16 operations at identical weights and inputs, not between-run nondeterminism. The
resulting token changes mean they cannot be dismissed as harmless numeric tolerance.

On the full100 **zero-vector, no-optimizer** census, canonical B0 is reproduced first on 100/100. The zero hook then
changes six token sequences and four normalized transcripts. One row, U1003_S0_141, changes from a 200-token capped
repetition to 95 tokens ending in EOS; mixed/ZH errors fall by 87 on this row alone. This is an uncontrolled numerical
perturbation with autoregressive amplification, not a proposed improved method and not evidence of linguistic
steering. It also crosses the existing half-baseline-length severe-truncation marker; fewer mixed errors do not by
themselves establish a safe recovery from the cap.

| No-optimization control | PIER | MER | ZH-CER | Mixed errors | ZH errors | S/D/I |
|---|---:|---:|---:|---:|---:|---|
| Ordinary B0 | .495690 | .256975 | .219295 | 1483 | 1107 | 339/1026/118 |
| TTLS ALL mask, z=0 | .495690 | .241553 | .201664 | 1394 | 1018 | 299/1053/42 |

Zero hook creates 3 baseline-correct ZH corruptions and repairs 16 reference ZH units; it makes no POI correctness
changes. Relative to this omitted control, saved TTLS-CE has **64 more mixed errors and 71 more ZH errors**, although
its eight POI improvements remain. The accidental perturbation does not explain away those eight English
segmentation/deletion corrections; it invalidates attribution of aggregate ASR and preservation differences solely
to the learned representation update against a clean initialization.

**Affected:** T1/T2/T4/T6 optimization starts and their zero-update counterfactuals; initial CE and KL statistics;
claims about clean-start causal adaptation-space comparisons. The saved free hypotheses and their metric counts
are reproducible observations. B0/B1/B2/T2A/T3/T5/CD/ACSUB do not use this TTLS repair and are not invalidated by it.

**Preservation claim correction:** initial P is nonzero on 97/100 T2 episodes (max .000423528) and 5/6 T6 episodes
(max .000156193). Initial gradient norms differ from their no-P counterparts on those same 97 and 5 episodes.
Therefore P acts through **both** updates for TTLS, not only update two. For A2+P, initial P is exactly zero, consistent
with the clean theta0 starting state. The initial TTLS discrepancy is small numerically, but the asserted exact
zero/one-regularized-update mechanism is false.

An unapplied minimal ratio-first repair is documented in `TTLS_R1_ZERO_EDIT_PATCH_PROPOSAL.md`, with CPU identity
and nonzero-gradient tests. It needs a recorded numerical contract revision and corrected TTLS outputs, not silent
replacement. No shared historical kernel or original experiment was patched here, and no alternate objective/LR/mask
was searched.

### Matched objectives, unmatched intervention opportunities

- CE shares the exact historical AUTO pseudo-label path; AC shares the same first accepted frozen candidate,
  audio/null evidence and NLL; +P shares the same stable B0 set, allowed vocabulary and lambda=1. These are genuinely
  matched supervision pairs. Preservation is not a TTLS-only feature because T2A and T5 are included.
- The actuator comparison is nevertheless **space + locality + update budget**, not an isolated parameter-space
  versus representation-space causal factor. TTLS cannot change step zero, is restricted to L16, and has a norm-bounded
  utterance-global vector. LN changes all decoder LN families and may change step zero. The key A2 lexical gains and
  harms occur in a whole-utterance first-token branch that TTLS structurally cannot take. Learning-rate rules and
  parameter budgets differ; equal two steps are not equal functional displacement or compute.
- The design-time localization change is explicit and frozen before execution. It is not outcome-based tuning,
  but CE is **global steering at all editable steps**, not localized token-wise activation optimization.
- The AC proposal is prompt-conditioned **clean versus all-zero 30-second audio**, not an oracle word/region target.
  Accepted candidate prefix matching anywhere in the reference (6/6) is not aligned lexical-target precision. Short
  ` I`, `c`, ` or` can match unrelated words; only one accepted row repairs a genuine wrong-language English error.

Minor reporting issues: the label's taxonomy calls concatenated-word repairs “genuine substitutions”; the report
already caveats this and the audit preserves the frozen rule. “Active-edit harm” is actually a rate conditional on
changed output rows, not all rows physically edited/optimized. It is not a population safety estimate. The stated
“all TTA1 bounds” omit embedded-EN retention from the frozen TTLS gate; the measured retention passes the historical
.95 bound anyway, so this omission does not change this run's label.

## 3. Independent recomputation from raw hypotheses and raw references

The independent script imports only versioned normalization and language tagging. It implements its own
Wagner–Fischer DP with the canonical diagonal > deletion > insertion backtrace, corpus language-aware counts,
reference-unit status, POI taxonomy, corrections/corruptions, CS-Mandarin retention, outside-POI correctness harm,
EOS-prefix classification and dialogue resampling. It never imports Claude's TTLS evaluator, corpus aggregation,
bootstrap, transition or decision functions as its source of truth.

Differential tests compare the independent alignment and per-POI taxonomy to the inspected canonical primitives on
all 1,200 saved transcripts. Corpus totals equal the sum of row counts. PIER error differences equal corrections minus
corruptions for every arm. There are **zero published metric/transition/new-ZH mismatches**. Denominators: 696 POIs,
5,771 mixed reference units, 5,048 ZH units, 4,054 baseline-correct CS ZH units, 351 baseline-correct POIs.

| Arm | PIER | MER | EN-WER | ZH-CER | POI errors | S/D/I | POI corr/corrupt | New ZH errors |
|---|---:|---:|---:|---:|---:|---|---|---:|
| B0 | 0.495690 | 0.256975 | 0.502874 | 0.219295 | 345 | 339/1026/118 | 0/0 | 0 |
| B1 | 0.408046 | 0.259054 | 0.415230 | 0.233558 | 284 | 324/1053/118 | 63/2 | 72 |
| B2 | 0.454023 | 0.253682 | 0.462644 | 0.221078 | 316 | 351/985/128 | 31/2 | 26 |
| T1 | 0.484195 | 0.252643 | 0.492816 | 0.215729 | 337 | 333/1007/118 | 8/0 | 0 |
| T2 | 0.487069 | 0.252989 | 0.495690 | 0.215729 | 339 | 334/1008/118 | 6/0 | 0 |
| T2A | 0.455460 | 0.249523 | 0.464080 | 0.216125 | 317 | 331/983/126 | 30/2 | 3 |
| T3 | 0.489943 | 0.261133 | 0.497126 | 0.224842 | 341 | 333/1057/117 | 4/0 | 30 |
| T4 | 0.491379 | 0.260613 | 0.498563 | 0.224049 | 342 | 334/1053/117 | 3/0 | 25 |
| T5 | 0.491379 | 0.261826 | 0.498563 | 0.225436 | 342 | 334/1060/117 | 3/0 | 32 |
| T6 | 0.491379 | 0.260613 | 0.498563 | 0.224049 | 342 | 334/1053/117 | 3/0 | 25 |
| CD | 0.482759 | 0.274649 | 0.485632 | 0.241878 | 336 | 312/1252/21 | 58/49 | 254 |
| ACSUB | 0.491379 | 0.260613 | 0.498563 | 0.224049 | 342 | 334/1053/117 | 3/0 | 25 |

Correct-Mandarin retention is 1.000000 for T1/T2, 4028/4054 for B2, 4051/4054 for T2A, and 4029/4054 for T4/T6.
Outside-POI harm counts are respectively 0, 26, 3 and 25 over 4,055 baseline-correct outside units. These are saved
hypothesis facts, not evidence that future TTLS active interventions are safe.

The original frozen decision is independently reproduced as **TTLS_R1_MIXED, T1 only**. None is PROMISING. Its three
qualifying canonical same-language substitutions are `peoplethey→people they`, `withlike→with like` and
`somerelationship→some relationship`, across two dialogues. This label does not establish novel lexical recovery.
Under a literal wrong-word/language-confusion interpretation, CE provides no qualifying lexical correction.

## 4. Independent GPU replay and omitted-control census

**Selection:** union of every changed T1 row (11), every changed historical B2 row (14), and all six accepted AC rows:
21 unique utterances. Add the first three canonical unchanged controls from a not-yet-covered dialogue, giving
24 utterances over 14 dialogues. The exact rule and IDs were frozen in `replay_selection.json` before launching.
This outcome-stratified sample is for audit coverage, never a new effectiveness population.

It covers all English correction types, all original A2 harm cases, all accepted AC outcomes including the severe
continuation failure, EOS recovery and the repetitive capped row. No Mandarin-for-English TTLS-CE correction exists
to sample. The absent category is a result, not filled using new rows.

- Slurm **58388**, MIG H100 3g.40gb: 3m32s allocation; measured replay work 187.808 s.
- 240 arm-output comparisons plus 24 candidate-matrix comparisons: **264/264 exact**.
- All selected tokens/termination match. Maximum loss difference, gradient-norm difference and optimized-z difference:
  **zero**. B0/B1 reuse checks and all archived B2 effective-state reconstructions pass.
- Actual DG-02 FFN-input recording confirms the learned z is consumed. Maximum replayed native chord 1.090307,
  below E*=1.126076; zero direct out-of-mask displacement on the fixed replayed history. Indirect autoregressive
  consequences are not constrained by that local identity.
- Model gradients remain absent; non-LN bytes unchanged; exact resident LN reset at completion.
- Slurm **58389** runs only the omitted zero-vector control and clean baseline on fixed100, no optimizer or new
  method: 110.717 s measured work. All 24 overlapping zero outputs match the earlier compact replay exactly;
  all pretrained weights remain unchanged. Full adaptation was not rerun because its targeted replay agrees.
- Audit launch **58387** failed before any model load at an incorrect import in the new audit harness. Its six-second
  failed log is preserved. The import was corrected before 58388; no original experiment source was changed.

The reference-free replay and zero-census outputs were separately sealed, committed and pushed on the audit branch
before their new reference scoring: replay seal `2e6a496a`, zero-control seal `54c2d32f`. The independent primary hypothesis scoring uses the already-sealed historical
outputs. No jobs remain active at audit completion.

## 5. Factorial interpretation: space, preservation and supervision

### Adaptation-space effect

| Matched objective | A2 POI corr / new ZH | TTLS POI corr / new ZH | Interpretation of saved outcomes |
|---|---|---|---|
| CE | 31 / 26 | 8 / 0 | TTLS edits less and avoids the first-token branch; gives substantially less English benefit |
| CE+P | 30 / 3 | 6 / 0 | A2 preservation closes most of the harm gap while retaining the benefit |
| AC | 4 / 30 | 3 / 25 | Six active rows; both fragile, TTLS adds no normalized benefit over substitution |
| AC+P | 3 / 32 | 3 / 25 | No convincing preservation advantage for either AC actuator |

TTLS-CE has worse PIER than A2 by .030172 and worse PIER than A2+P by .028736. Its lower new-ZH count is not measured at
matched correction levels. No dose/correction-matched Pareto experiment was performed. The clean-start defect further
prevents attributing this trade-off to representation optimization itself.

### Preservation effect

A2+P retains **30/31** corrections, with the same 15 wrong-language lexical corrections, while new ZH errors drop
26→3; mixed errors 1464→1440. This is the most useful positive development observation, independent of the TTLS hook
bug. But 20 of the 23 avoided new ZH errors are on U0023_S0_664; it remains highly concentrated. Its 15 genuine
wrong-language English corrections all come from U0021_S0_513. No broad preservation or lexical-generalization
claim is warranted.

TTLS+P loses two POI corrections and has the same 0 new ZH errors. The initial KL premise is numerically false for
TTLS, and the raw effect is small. For AC, +P leaves TTLS normalized outputs and all token arrays unchanged, while LN
+P is slightly worse. Teacher-forced retention on the B0 path does not stop the wrong continuation basin.

### Objective and direct-alternative effect

CE is stronger than this sparse AC rule for both adaptation spaces. AC accepts six of 100 rows, all in the
AUTO=B0 group. It therefore provides no active test on the D12 group. T4 and T6 are token-identical; T4 versus ACSUB
differs in tokens on two rows but is **normalized-transcript identical on all 100**, with all per-row metric counts
identical. Adaptation adds no demonstrated recognition value to the candidate.

On U1034_S0_53, `烧烤→cooking` is one genuine lexical correction shared by TTLS, LN and direct substitution. On
U2011_S0_99, `of course` is deletion recovery. On U1003_S0_63, inserting ` or` causes an earlier sentence ending:
92 baseline content tokens → 64 under T4/ACSUB, +27 deletions, +23 mixed and ZH errors. The altered prefix means the
frozen strict-prefix “premature EOS” counter is zero, and the length remains above the half-length severe threshold.
Those formal zero counters do **not** mean no termination damage. Report this concrete failure separately.

CD/D3 is a distinct full-sequence direct alternative, not the same AC target rule. It is worse on MER and has 254
new ZH errors/49 POI corruptions. No evidence justifies substituting CD for this pilot's candidate source.

### English lexical effects versus termination/deletion

- TTLS-CE: eight corrected POIs, four dialogues. Five are canonical deletion/boundary outcomes and three are
  concatenated-word repairs. Two corrections (`super topic`) occur on strict-prefix EOS recovery; `it's` is an actual
  omitted-English-unit recovery. No Mandarin-for-English wrong-word correction is made.
- T1 has four EOS-recovery rows: they contribute 12 of its 18 net ZH error reductions and 14 of its 19 net deletion
  reductions. Five additional repaired baseline ZH units occur in the capped repetitive row, not robust CS lexical
  recovery. These figures describe the saved method against B0 and do not eliminate the zero-start confound.
- A2's four EOS-recovery rows account for all 26 repaired baseline ZH units and 12 POI improvements; its 15 genuine
  wrong-language POI corrections occur on one different utterance. Both termination and lexical effects exist, with
  strong concentration and some new harms.
- All TTLS-AC Mandarin damage is on two rows; 23/25 new ZH errors are on one dialogue/utterance. It is not a broadly
  evidenced improvement in correction–preservation behavior.

## 6. Dialogue uncertainty and repeated development exposure

Independent paired dialogue bootstrap: 20 dialogue blocks, 2,000 draws, NumPy seed 240924; recompute micro-ratios from
summed counts in every draw. All listed intervals have 2,000 valid draws. Differences below are **method minus
comparator**; lower is better for error rates.

| Contrast | ΔPIER [95%] | ΔMER [95%] | ΔZH-CER [95%] |
|---|---|---|---|
| T1-B2 | +0.030172 [-0.009761, +0.081999] | -0.001040 [-0.012350, +0.008105] | -0.005349 [-0.017388, +0.002255] |
| T2-T2A | +0.031609 [-0.007353, +0.082817] | +0.003466 [-0.001287, +0.010616] | -0.000396 [-0.005204, +0.004445] |
| T4-T3 | +0.001437 [+0.000000, +0.004599] | -0.000520 [-0.002482, +0.000730] | -0.000792 [-0.002941, +0.000574] |
| T6-T5 | +0.000000 [+0.000000, +0.000000] | -0.001213 [-0.003723, +0.000000] | -0.001387 [-0.004245, +0.000000] |
| T1-B1 | +0.076149 [+0.008162, +0.153064] | -0.006411 [-0.016931, +0.004844] | -0.017829 [-0.037006, -0.003846] |
| T2A-B2 | +0.001437 [+0.000000, +0.004386] | -0.004159 [-0.013434, +0.000000] | -0.004952 [-0.015905, +0.000000] |
| T2-T1 | +0.002874 [+0.000000, +0.009759] | +0.000347 [+0.000000, +0.001040] | +0.000000 [+0.000000, +0.000000] |
| T4-ACSUB | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |

These intervals reproduce the published overlapping comparisons. They are descriptive, unadjusted exploratory
intervals, not significance or equivalence tests. They do not account for all historical/adaptive development
choices. Zero-count corruption intervals from resampling would degenerate and cannot establish zero population
risk. Query/word counts are not independent observations; there are only 20 repeatedly used dialogues.

On leaving out the dialogue with the largest POI rescue, T1 net POI rescue remains five, but these remain
segmentation/deletion outcomes. A2+P net ZH rescue remains at least two after removing any one dialogue. Both facts
are suggestive robustness descriptives, not independent confirmation. A3's enriched 24-panel and D12 subgroup must
not be used as population estimates. AUTO remains better on PIER than every adaptation arm; T1−AUTO is +.076149
with descriptive interval [+ .008162, + .153064]. There is no reliable aggregate MER advantage over AUTO.

## 7. Representation mechanism: what is actually identifiable

The optimized variable is one unrestricted temporary **utterance-global additive vector**, reused across all CE
positions (or one AC position). It is not a separately optimized hidden state for each query, a frozen linguistic
basis, a region detector, or a trained controller. It may be called a representation-space actuator, but no
linguistic interpretation follows from its dimensionality or cosine alone.

Available and checked: loss/gradient trajectories, FP32/BF16 vector values and norm budget, same-history native
FFN-input displacements at edited versus unedited positions, candidate log-prob margins/ranks, actual free tokens,
reset/caches and metric consequences. Candidate TF rank improves, but for AC the resulting sequence can be
harmful; a local margin is not correct-word recovery.

Not available as matched sealed quantities: CE reference lexical margins, query-specific lexical-gradient vectors,
or matched D3 gradient directions at these exact TTLS queries. DIR-SPRINT0 D3's directions were constructed on a
different population/target rule; importing their cosines here would be scientifically invalid. No learned-edit
alignment with D3 or gold lexical gradients is established. No new gradient-direction search was launched to make
the interpretation more favorable.

## 8. Reviewer verdict and Round-2 recommendation

**Strengths:** matched pseudo-supervision and explicit A2+P comparator; fresh episodic states; correct DG-02 site;
real free decoding; untouched weights for TTLS; exposed-data disclosure; original negative conclusions are candid;
complete hash-addressed raw outputs permit exact independent replay.

**Principal weaknesses:** nonidentity zero-vector starting path; missing sham-control inference; false initial-P
claim; different masks/step-zero accessibility and functional budgets confound a pure adaptation-space effect;
small and concentrated lexical opportunities; weak accepted-candidate “precision”; sequence harm missed by narrow
EOS counters; no independently demonstrated ASR benefit over AUTO or direct substitution; repeated use of the same
dialogues; measured adaptation-time savings are small beside decoding/setup and do not offset the lost English
benefit.

Answers to the skeptical-review questions:
1. **TTLS better than matched A2?** Not established; CE produces less benefit and the clean-start invariance fails.
2. **Could simpler A2 preservation explain the useful improvements?** Yes on this development panel; A2+P retains
   almost all A2 POI corrections with most observed new-Mandarin harm removed, albeit concentrated.
3. **Lexical advantage beyond EOS?** TTLS-CE has none for wrong English/language substitutions; AC's one genuine
   correction is already obtained by no-update substitution.
4. **Less Mandarin damage at comparable correction?** Not tested at matched correction; raw lower harm accompanies
   substantially fewer corrections and a forbidden first-token edit.
5. **Beats AUTO/direct alternatives?** No on PIER or on AC recognition over substitution; MER differences versus
   AUTO are uncertain.
6. **Justifies complexity?** No demonstrated efficacy gain; roughly half the adaptation time saves only ~10 s over
   100 rows while decoding dominates. Replay timings include diagnostic forwards and are not deployment latency.
7. **Interpretable representation?** A temporary optimization variable with a causal physical site, but no evidence
   of a specific language/phonetic mechanism. Native numeric control is essential before stronger claims.
8. **Development overstatement?** Any broad safety/superiority claim would be overstated on repeatedly exposed
   dialogues and concentrated events. The report's own caution is appropriate but does not cure the numerical flaw.

**Round 2 for TTLS efficacy is not scientifically justified from this run.** First resolve the exact-zero numerical
contract in an explicitly recorded repair revision; test forward identity while preserving gradients; and preserve
original outputs while rerunning only affected TTLS/control components with unchanged supervision/settings. That is
an integrity repair, not a new efficacy result or permission to tune a method. The patch is a proposal here because
changing the frozen shared repair requires a documented scientific revision.

A separately frozen, exposure-safe evaluation of **A2+P** is more defensible as a prospective next question than
expanding TTLS, given its observed trade-off. It would still require explicit human approval, a declared development
versus confirmation role, AUTO comparisons, breadth/concentration reporting and no retrospective threshold tuning.
No Round 2, confirmation, test or transfer execution is authorized or launched by this audit.

## 9. Audit artifacts and reproduction commands

- `experiments/ttls_r1_independent_analysis.py`: independent DP/counts/transitions/bootstrap from raw references.
- `experiments/ttls_r1_independent_replay.py`: exact frozen-method replay plus independent native-consumption probes.
- `experiments/ttls_r1_zero_control_census.py`: no-optimization full100 zero-vector control only.
- `experiments/ttls_r1_audit_summary.py`: provenance, decomposition, independent frozen-label recomputation and verdict.
- `results/inference_cf/ttls_r1_independent_audit/`: all independent scoring, manifests, replay IDs/rows, control census,
  seals, model-file verification and preserved failed-launch logs.
- `docs/inference_cf/TTLS_R1_ZERO_EDIT_PATCH_PROPOSAL.md`: minimal unapplied repair and required revalidation.
- `tests/test_ttls_independent_audit.py`: independent alignment/taxonomy/count identity, zero-vector defect,
  gradient-preserving repair proposal and replay/census integrity checks.

```bash
export PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:$LD_LIBRARY_PATH
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
$PY experiments/ttls_r1_independent_analysis.py
$PY experiments/ttls_r1_audit_summary.py
$PY -m pytest -q tests/test_ttls_independent_audit.py tests/test_ttls_r1.py
```

CPU verification: **71 tests passed**: 10 independent audit tests + 14 existing TTLS tests, and 47 historical
MER/PIER/retention/canonical-metric/episodic-TTA tests. There were no failures. Joblib fell back to serial execution
because `/dev/shm` was full; the existing scalar-loss conversion warning did not interrupt or detach optimization.
Exact commands and counts are in `test_results.json`.

GPU harnesses are separately frozen audit replays, not production runner substitutions. Do not rerun them into
existing output paths or overwrite raw seals. CPU analysis may regenerate derived audit summaries from sealed raw
outputs. Original experiment sources, raw results, config and reports are byte-unchanged at audit completion.

Final verdict remains **`AUDIT_FAIL_INVALID_RESULTS`**, specifically for TTLS causal-validity/clean-start claims.
The numerical tables above remain independently reproduced descriptions of preserved hypotheses; the audit does
not defend TTLS or treat a passing replay as proof of method effectiveness.
