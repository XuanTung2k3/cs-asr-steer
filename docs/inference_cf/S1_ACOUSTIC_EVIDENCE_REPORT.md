# S1 — Acoustic Evidence Feasibility: report (terminal)

**Terminal label: `S1_CANDIDATE_HEADROOM_INSUFFICIENT`** (evaluator and independent FULL audit agree). S2 is not authorized.

| Stage | Result |
|---|---|
| Freeze | `a89067b` (spec / design / firewall / panel / config) |
| Implementation | `4699430`; sealed plan `8adcab7` (23/23 prepare checks) |
| Pre-run audit | attempt 1 `FAIL_PRERUN` (`672268e`; static-check false positive, preserved) → mechanical fix `6c6917f` → `PASS_TO_S1` (`686df02`) |
| Job 1 (candidates) | Slurm 58176, manifest `4332156`; candidate seal `f6c73bb`, pushed before any mask |
| CANDIDATES audit | attempt 1 FAIL (`574b463`; open-log rule flagged the runner's own source-hash reads, preserved) → auditor-rule fix `9a2bb1e` / `9e60ef6` → `S1_AUDIT: PASS (CANDIDATES)` (`5714583`) |
| Job 2 (acoustic) | Slurm 58177, manifest `4af3372`; output seal + reference-free primary analysis `7ffb21b`, pushed before any reference |
| PRIMARY audit | `S1_AUDIT: PASS (PRIMARY)` (`6055fcd`), before any reference was opened |
| Evaluation | `9483d40` (P2-RJ targets opened only now) |
| FULL audit | `S1_AUDIT: PASS (FULL)` (`d5dd94e`); independent label identical |

## 1. Research questions

1. When the forced-Mandarin baseline (M) is wrong at an EN-confusion query, does frozen Whisper-large-v3 already put the
   correct English **first token** among its accessible candidates (M / forced-English E / native-detected AUTO, Top-5/20)?
2. If it does, does model-predicted acoustic evidence — the change in M log-probability when an R0-predicted English
   region is hard-zeroed — rank that token above the incorrect Mandarin candidates, beyond a matched off-target mask,
   the historical LAC mask and a shuffled-support control?

## 2. Reused evidence

- **Historical identity.** The M raw logits at all 180 queries are bitwise identical to the sealed historical NONE
  logits (ST-LOC0 calibration, ST-PROMPT-R1 capture and the P2-DIR source of LAC0).
- **R0 regions.** Only `track_heard_intervals` was read from the R0 primary arrays, after full-file hash checks. No
  oracle, MMS-FA timing or script masks were read.
- **LAC0.** The historical masked logits passed the exact-reuse audit (prefix, query, waveform, W*, unmasked-vector
  hashes) on 180/180 queries, giving 0 new LAC forwards.
- **Inputs.** Audio and baseline B0M_L16 prefixes are panel-pinned. The P2-RJ acceptable first-token sets were used in
  the evaluator only.

## 3. Panel and model

- **Panel.** The exact parent panel: 180 queries (60 EN-confusion / 60 EN-correct / 60 ZH-correct), 80 utterances, 20
  dialogues. Panel identity is `sha256:c78ef1e9…` and runtime membership is `sha256:1bbaaa1c…`. This is already-exposed
  D-dev-select data.
- **Model.** Frozen Whisper-large-v3 (bf16 / eager / batch 1 / inference_mode), with pinned model and installed
  generation-file hashes. Weights were unchanged (probe), with no gradients, no hooks left and no steering.

## 4. Candidate generation

- **Branches.** M = `[50258,50260,50360,50364]` and E = `[50258,50259,50360,50364]`. Each branch has its own cold cache
  fed the identical B0 prefix.
- **AUTO.** The native installed `detect_language` was called once per original utterance (80 SOT queries). The
  lowest-ID argmax was verified independently in all 80 cases. It detected **zh 65 and en 15**, so AUTO is always
  identical to M or E. AUTO was run as an alias with 0 extra steps, and by construction it adds no candidates.
- **Processing.** Exact `_processed` (suppression only, since every query has t ≥ 1), float32 full-vocabulary
  log_softmax, descending-value / ascending-ID Top-K.
- **Unions.** Mean union size was 5.86 at K = 5 and 23.4 at K = 20 (max 30). EOS appears in 24 K5 unions and 54 K20
  unions. OTHER_ACTION tokens appear in 2 K20 unions. Neither ever counts as a hit.
- **Ties.** The smallest gap at the Top-20 boundary was exactly 0, so the frozen lower-ID rule decided some boundary
  ties. This was verified independently.

## 5. S1-A: candidate headroom (EN-confusion, /60)

| Branch | Recall@5 | Recall@20 | dialogues @20 | dialogue-macro @20 |
|---|---|---|---|---|
| Forced-ZH (M) | 3 | 8 | 7 | .147 |
| Forced-EN (E) | 3 | 10 | 8 | .167 |
| AUTO (= M/E alias) | 3 | 10 | 8 | .181 |
| **Union** | **4** | **11** | **9** | **.196** (pointwise 95% .094–.309) |

- **Incremental coverage.** E beyond M@20 is 3 queries in 3 dialogues. AUTO beyond M+E is 0. Union minus M@20 is
  **3 (3 dialogues)**. At K = 5, union minus M is 1.
- **Reference first-token rank over the full distribution** (median; queries with rank ≤ 5 / ≤ 20 / ≤ 100):
  - M: 353; 3 / 8 / 16.
  - E: 316; 3 / 10 / 17.
  - AUTO: 341; 3 / 10 / 16.
  - No query lacks a legal target.
- **Correct-state ambiguity.**
  - Every EN-correct union contains a Latin token (mean 17.8 at K = 20).
  - 19/60 ZH-correct unions contain at least one ASCII-Latin token (mean 0.88).
- **Structural limitation** (historical P2-R ledger, EN-confusion units):
  - 371 candidate units, of which 227 are valid.
  - 22 are `r2_unalignable` (deleted or omitted English with no baseline query).
  - 36 are not at a token start, 45 have a mid-character prefix, 30 share a position, and 11 are forced-prefix.
  - S1 tests only existing queries. First-token availability is not full-word recovery.

**Gate H: FAIL.** The 180 outputs are complete and valid, and union@20 spans 9 ≥ 6 dialogues. But union@20 hits are 11
against a required 12, and the incremental count over M@20 is 3 against a required 5 (its 3 dialogues meet the ≥ 3
requirement).

## 6. Predicted region and counterfactual construction (reference-free)

| target status | all 180 | EN-confusion | EN-correct | ZH-correct |
|---|---|---|---|---|
| OK, matched off-target | 59 | 20 | 35 | 4 |
| OK, NO_MATCHED_OFFTARGET | 14 | 8 | 4 | 2 |
| LOW_ASSOCIATION (< .10 query attention in any EN crop) | 80 | 14 | 16 | 50 |
| LOW_HEARD_MASS (< .5) | 14 | 14 | 0 | 0 |
| NO_EN_REGION | 13 | 4 | 5 | 4 |

- **Target crops.** 73 crops of 4,880–32,000 samples (median 32,000). Query-attention integral was .11–.96 (median .69).
- **Off-target crops.** Same length, integral ≤ .057 (median .0008), energy ratio .50–1.99 (median .91).
- **Integrals.** Computed exactly in rational arithmetic, so the frozen tie rules are applied exactly. The auditor
  reproduced them with an independent Fraction route.
- **Masks.** Hard positive zero, outside bytes identical, whole masked encoder, cold M cache: 124 distinct masks and
  132 query readouts. Zero-change preprocessing was identical on all 80 utterances.

## 7. S1-B: acoustic discrimination (descriptive only, because H failed)

- **Paired-accessible population.** P20 = **4** EN-confusion rows in 4 dialogues (Gate D needs ≥ 12 rows in ≥ 6
  dialogues). That is 36% of the 11 accessible rows (needs ≥ 50%). P5 = 1.
- **Why accessible rows fall out of P20.** Of the 7 accessible rows outside P20:
  - 3 are LOW_ASSOCIATION.
  - 2 are NO_EN_REGION.
  - 2 have a target but no matched off-target.

| K20 on P20 (4 rows) | M | EN_REGION | OFF_TARGET | LAC_UNION | SHUFFLED | E | AUTO |
|---|---|---|---|---|---|---|---|
| Hits@1 | 0 | **0** | 0 | 0 | 0 | 0 | 0 |
| MRR (dialogue-macro) | .098 | **.246** | .098 | .142 | .090 | .181 | .114 |
| mean reference rank | 12.25 | 5.25 | 12.25 | 7.25 | 11.75 | 11.75 | 9.25 |

- **Rank changes under the target mask.** EN_REGION raised the reference rank on all four rows: 21→7, 13→7, 7→2 and
  8→5. The matched off-target mask left ranks essentially unchanged (21, 13, 8, 7). It produced **no** top-1 correction.
- **EN_REGION MRR gain** (Bonferroni family 8, adjusted 99.375% interval, 9,904/10,000 finite draws):

  | vs control | gain | adjusted lower |
  |---|---|---|
  | M | +.148 | +.066 |
  | OFF_TARGET | +.148 | +.057 |
  | SHUFFLED | +.157 | +.060 |
  | LAC_UNION | +.104 | −.024 |

  This is n = 4 conditional evidence on enriched, exposed data, not support for the acoustic hypothesis.
- **Unconditional /60 (fallback to M where unpaired).** EN_REGION gives 0 Hits@1, as does M. LAC_UNION gives 2 and E
  gives 1, all on rows outside P20. These are not advancement tests.
- **Correct-state protection** (K20, with fallback):
  - EN-correct: 1 new corruption.
  - ZH-correct: 2 new corruptions.
  - 0 newly wrong Latin promotions on ZH-correct, and 0 EOS promotions.
  - Shuffled support corrupts 13 EN-correct rows, and E corrupts 2 EN-correct + 5 ZH-correct rows.
- **Gate D** is reported only. Coverage and point gates fail (P20 is too small, with no net Hits@1). Protection and the
  MRR-margin gates pass. The adjusted-inference gate fails against LAC_UNION.

## 8. Historical LAC0 comparison (not pooled)

- **LAC0 terminal label.** `P2_SEL_LAC_NOT_DISCRIMINATIVE` is unchanged (S_TP−FP lower80 −.022; strong TP 6). Its raw
  best-E/best-M S_lex is a different statistic on a different candidate population and fixed 0.5 s W*.
- **LAC_UNION.** This is a new comparator: the S1 fixed score applied to the old masked distributions on the new
  unions. It moves top-1 at 20 queries versus 11 for EN_REGION.

## 9. Statistics and abstentions

- **Bootstrap.** Dialogue-cluster bootstrap, seed 240924, B = 10,000, 20 fixed dialogues. Conditional statistics use the
  mean of dialogue means with multiplicity weighting; empty draws are undefined, never imputed. Bonferroni family 8 at
  quantiles .003125 / .996875. Every headline denominator keeps all 60 (or 180).
- **Abstentions.** Region and control abstentions (121/180 not paired) are expected feasibility limitations, not
  invalidity. They are mainly low query-attention association (80) and low heard-frame attention mass (14).

## 10. Bottleneck diagnosis

- **The primary bottleneck is candidate availability.** Even the frozen multilingual union holds an acceptable English
  first token for only 11/60 EN-confusion queries. E adds only 3 queries beyond M@20, and AUTO adds nothing on this
  panel because native detection always returns zh or en. The median reference rank is about 300–350 in every branch.
- **The secondary bottleneck is region association.** Only 20/60 confusion queries receive a paired target/off-target,
  and only 4 of these are also accessible. Where a pair exists, removing predicted-English audio specifically lowers M's
  preference against the reference (unlike the matched off-target mask). It never lowers it enough to change top-1.

## 11. Implications for S2

- **S2 is not authorized.** The frozen precedence stops at H, and nothing in S1 measures steering, full decoding or ASR.
- **What is supported.** Constrained re-ranking of the model's own Top-K cannot reach most English confusions on this
  panel.
- **The 4-row signal.** Target-specific acoustic sensitivity that beats off-target and shuffled controls but not LAC is
  hypothesis-generating only.
- **Any follow-up needs a new pre-outcome freeze.** That includes larger K, other candidate generators, other masks,
  more data and new queries. None may reuse these outcomes for selection.

## 12. Compute, provenance and audits

- **Jobs.**
  - Slurm 58176 (candidates): 1 min 50 s wall, 92 s runtime, peak 3.69 GB allocated / 3.99 GB reserved.
  - Slurm 58177 (acoustic): 1 min 08 s wall, 52 s runtime, peak 3.42 / 3.78 GB.
  - The in-allocation smoke forecast 4.9 min and 9.0 min.
- **Forward counts.**
  - Original pass: 80 encoders, 6,318 cached decoder steps (M 3,159 + E 3,159, AUTO 0 via alias; allowance 9,477) and
    80 detection queries.
  - Masked pass: 124 encoders and 3,500 steps (allowance 360 / 11,692).
  - No LID, LAC, autograd, optimizer or steering calls.
- **Smoke and tests.**
  - CPU engineering smoke on synthetic audio exercised every path.
  - Tests: `test_s1_impl` (21) + `test_s1_freeze_contract` (91).
  - Inference_cf regressions 208/208 across 12 files.
- **Firewall evidence.** Both jobs logged every opened path. No P2-RJ, panel strata, population or evaluator bytecode
  was opened. The only evaluator/auditor file reads were the runner's own pinned-source hash verification.
- **Failed attempts.** Both are preserved (`prerun_audit_attempt1_FAIL.json`, `candidates_audit_attempt1_FAIL.json`).
  Neither involved outcomes or references.
- **Artifacts.** Machine-readable outputs are in `results/inference_cf/s1/`: `plan_sealed`, `run1/` (manifests,
  runtimes, `candidates/`, `acoustic/`, `candidates_sealed`), `output_seal`, `primary_analysis`, `analysis` and all
  audits.
