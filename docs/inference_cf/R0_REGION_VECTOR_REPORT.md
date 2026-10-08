# R0 — predicted regions and per-utterance unique-direction feasibility (report, terminal)

**Terminal verdict: `R0_ORACLE_CONSTRUCTION_INSUFFICIENT`.** Independent audits:
- Pre-run: `PASS_TO_R0`.
- After the reference-free seal, before oracle access: `R0_AUDIT: PASS (PRIMARY)`.
- After the oracle phase: `R0_AUDIT: PASS (FULL)`.

The full auditor reproduced the label independently.

**Decision:** no layer passes the oracle-constructibility gate O (≥ 60 oracle-valid utterances in ≥ 12 dialogues).
Even reference-derived membership produces only 16–17 valid per-utterance vectors per layer.
- R1 is **not** authorized.
- No new region predictor, vector family, rank or threshold is introduced.
- This is a construction-feasibility result. It makes no ASR, steering or causal claim.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, firewall, panel, config, contract tests) | `a5929c6` | pre-outcome |
| Implementation (isolated additive modules + tests) | `1519f9d` | 15 focused tests; R0 freeze + ST-LOC0/DG-02/LSS/P2-DIR/P2-R regression 120/120 |
| Sealed reference-free plan | `7e2c607` | 12/12 prepare checks |
| Pre-run audit | `9a33d01` | `PASS_TO_R0` (57 checks) |
| Manifest (plumbing child; job ran at `9a33d01`) | `9e05252` | pushed before the job |
| Run | Slurm 58150 | 11 min 51 s, 300/300 rows, 0 failures |
| Output seal (arrays in 3 batches, then records + seal) | `f7b9431`, `51daa5e`, `3134477`, `df75622` | pushed before any oracle access |
| Primary-phase audit | `3929b42` | `R0_AUDIT: PASS (PRIMARY)` (37 checks; 27,600 constructions recomputed, 0 disagreements) |
| Oracle evaluation | `c670386` | ORACLE_CONSTRUCTION_INSUFFICIENT |
| Full audit | `24388cf` | `R0_AUDIT: PASS (FULL)` (17 checks) |

## Method and reproducibility

**Population.** Frozen `R0_PANEL.json` (identity `sha256:ff28efc6…`): the already-exposed D-dev-select FULL300.
- 300 utterances; 20 dialogues × 15.
- 27 long rows are kept. Full waveform N and baseline-heard N = min(N, 480000) are both sealed.
- No D-dev-confirm, D-test, router-calib new role, P3 or transfer data was used.

**Reference-free region predictor.** Unchanged `inference_cf_p0_r2.native_lid` (single SOT readout, 100-way native softmax),
run on the frozen grid: 1 s windows, 0.5 s stride, plus a unique right-anchored final window. That gives 10,387 windows over
the full waveforms, all computed (0 cache reuse).
- Window rule: U if a crop is shorter than 8000 samples, has RMS < 1e-4, or Q = P_E + P_M < 0.50. Otherwise EN if
  pE_pair ≥ 0.80, ZH if pM_pair ≥ 0.80, else U.
- At each sample, EN or ZH needs ≥ 0.75 of all covering windows (U counts in the denominator).
- Maximal runs are merged with no gap filling. Frame masks are exact sample fractions per 320-sample frame.

**Baseline decoder states.**
- Content is the sealed PATH5 theta0 forced-ZH tokens (13,561 tokens; 298 EOS-terminated, 2 capped).
- One causal full replay (`p0_r2.full_replay`, use_cache = False) under a passive 4-layer DG-02 recorder captures
  r = q + u_source (native bf16, stored losslessly) at layers 3/8/16/24 together with the 10 frozen alignment heads.
- Per row, this replay ran twice; both runs were bit-identical on all 300 rows.
- A forced-EN replay of the same prefix feeds only the descriptive v_prompt comparator.
- On every row and layer, the recorded site equals the tensor the FFN consumes (`final_layer_norm` input).
- No edits, gradients, optimizer or teacher forcing on references.
- 12,146 queries are eligible (1 ≤ t < T, non-special target, complete UTF-8 prefix). Replay argmax agrees with the cached
  baseline token on 99.8% of them, recorded descriptively only.

**Mapping.** Arithmetic mean of the 10 alignment heads over heard frames.
- A query with raw real-audio mass < 0.50 is U.
- Otherwise the mean is normalized, and q_E / q_M are the attention-weighted frame-mask masses.
- Assignment: EN iff q_E ≥ 0.70 and q_M ≤ 0.20 (ZH likewise), else U.
- One membership is shared by all four layers. 200 ms argmax bins act as a diversity proxy.

**Construction** (`r0_unique`, isolated and new): raw uncentered float64 moments via thin SVD of each group's own states.
- The largest rank in {8, 4, 2} supported in both groups is chosen before winner tests. Support requires: count ≥ 2r + 2,
  bins ≥ r + 1, numerical rank, centered participation ratio ≥ r, eigen floor, and an eigen gap > 5%.
- Principal pairs → score (a′C_E a)(1 − σ²) → guarded winner (score floor, 5% winner gap, σ isolation).
- Orientation toward mean_E − mean_M; f32 unit vector.
- No lower-rank retry, no fallback (A5, D1, oracle or v_prompt).
- Every primary vector was refit twice with identical status, rank, winner and hash on 300 × 4.

**Controls** (sealed before oracle access):
- 20 circular shifts of the heard label track: offsets in [1600, N − 1600], PCG64 seeded from SHA256 of
  `R0-shuffle-v1|240924|<uid>`. All 300 rows had ≥ 20 legal offsets.
- ERODE_100MS and DILATE_100MS (an expanded EN/ZH overlap becomes U).
- A baseline-script diagnostic (tokenizer partition votes; descriptive only).
- The v_prompt comparator.

**Firewall evidence.**
- The runner reads only the rows projection.
- Its runtime file-open log (46 paths) contains no oracle, CTC, role or evaluator file.
- The oracle-source bytes were verified by the pre-run auditor, not the runner.
- Static scans of the provider modules show no reference identifiers.
- The oracle mode refuses to run without the committed seal and a committed PRIMARY PASS.

**Oracle source** (post-seal, evaluator-only; REFERENCE-DERIVED TIMING PROXY, not manual ground truth):
`candidates_existing_ctc.parquet`. The file hash matches the freeze, and the parquet is filtered to the 300 IDs before any
column is materialized.
- 16,766 records, all used: 2,268 EN and 14,498 ZH unit intervals.
- 0 invalid, 0 out of bounds, 0 duplicates, 0 EN/ZH conflicts.
- Every record's language and surface match the historical `nat5h` reference units rebuilt from the role transcript, and its
  per-utterance normalization hash matches.
- *Mechanical note:* the config's single `normalization_hash` pin (`f781bcc1…`) is canonical row 0's per-utterance hash. It
  was checked on row 0; all other rows were checked against their recomputed historical hash. This interpretation was fixed
  before any boundary or outcome was read.
- Oracle membership replaced only the label track. The sealed queries, states, attention and mapping were reused unchanged:
  the recomputed mapping was bit-identical.

## Results

**Predicted regions (reference-free).**
- Windows: 2,456 EN, 5,854 ZH and 2,077 U (1,587 low pair-mass, 490 ambiguous).
- Heard samples: EN 13.19 M, ZH 34.08 M, U 31.27 M.

**Region quality vs the oracle proxy** (heard horizon, known support; pooled):

| EN precision | EN recall | ZH precision | ZH recall | Classified coverage | Rows / dialogues with known EN + ZH |
|---|---|---|---|---|---|
| 0.967 | 0.584 | 0.981 | 0.725 | 0.700 | 299 / 20 |

The frozen region-quality gate **passes**. Descriptive measures:
- IoU: EN 0.335, ZH 0.425. Proxy U covers 57% of heard audio, because unit spans do not tile the audio.
- On proxy-U audio, 16.0% is predicted EN and 36.8% ZH.
- 200 ms boundary matching is 7.8% (median error 2,117 samples).
- On the full waveform the figures are similar (EN P/R 0.967/0.578). 449,200 proxy-EN samples lie beyond the heard 30 s.

**Construction validity by layer** (300 records per layer, none missing):

| Layer | Predicted-valid (dialogues) | Rank 2/4/8 | Oracle-valid (dialogues) | Paired (dialogues) | Gate O | P | S | I |
|---|---|---|---|---|---|---|---|---|
| 3 | 31 (14) | 20/9/2 | 17 (8) | 12 (6) | **fail** | – | – | – |
| 8 | 30 (14) | 22/4/4 | 16 (8) | 9 (5) | **fail** | – | – | – |
| 16 | 32 (14) | 22/6/4 | 16 (7) | 11 (6) | **fail** | – | – | – |
| 24 | 32 (14) | 20/8/4 | 16 (8) | 10 (5) | **fail** | – | – | – |

**Failure reasons.**
- Predicted: 267–270 of 300 per layer are `NO_SUPPORTED_RANK`. In about 260 of these the EN group fails every rank
  condition (count, bins, numerical rank, participation, eigen floor/gap). In about 103–157 the ZH group also fails.
  Isolated winner-gap failures: 0–1.
- Oracle: 283–284 per layer are `NO_SUPPORTED_RANK`.

**State availability (the bottleneck).**
- The median utterance has 34 eligible queries. Under predicted membership it has 0 EN-assigned queries: 151 rows have none
  and only 48 have ≥ 6.
- Under oracle membership, only 29 rows have ≥ 6 EN-assigned queries, 147 have ≥ 6 ZH-assigned, and 18 have both. Totals
  are 627 EN and 2,529 ZH oracle-assigned queries out of 12,146 eligible.
- Two things combine: English spans are short within code-switched utterances, and the 0.70 attention-mass rule rarely
  places a decoder query inside a single short unit interval.

**Direction stability (predicted-valid rows; reference-free S inputs).**

| Layer | Both perturbations valid | Stable (min signed cos ≥ 0.80) | Median min cos | Reference-free S |
|---|---|---|---|---|
| 3 | 0.839 | 0.677 | 0.969 | fail |
| 8 | 0.867 | 0.833 | 0.979 | pass |
| 16 | 0.750 | 0.688 | 0.987 | fail |
| 24 | 0.875 | 0.875 | 0.975 | pass |

These are not gate outcomes, because P and O fail first.

**Predicted–oracle similarity** (paired rows only; descriptive — the gate never reaches I):

| Layer | Paired | Signed cos, macro [pointwise 95%] | Absolute cos | Signed-evaluable rows | Adjusted signed lower | Advantage over shuffle–oracle [adj. CI] |
|---|---|---|---|---|---|---|
| 3 | 12 | 0.787 [0.605, 0.928] | 0.794 | 10 (4 dialogues) | 0.428 | +0.405 [+0.096, +0.693] |
| 8 | 9 | 0.856 [0.775, 0.906] | 0.856 | 9 (5) | 0.712 | +0.401 [+0.306, +0.642] |
| 16 | 11 | 0.823 [0.743, 0.888] | 0.823 | 9 (5) | 0.665 | +0.363 [+0.300, +0.472] |
| 24 | 10 | 0.837 [0.784, 0.881] | 0.837 | 9 (4) | 0.758 | +0.273 [+0.102, +0.512] |

- Median shuffle–oracle cosine is 0.34–0.60.
- Within the few paired rows, predicted vectors agree with oracle vectors and beat shuffled regions. This rests on 9–12 rows
  from 4–6 dialogues, concentrated in CSD0043/0046/0051/0502/0532/0538. It is not evidence of broad feasibility, and it is not
  gate-qualifying.

**Controls (descriptive).**
- Shuffles: on average 2.6–3.0 of 20 are valid per row; 41–48 rows have ≥ 10 valid.
- Median shuffle-vs-primary cosine on predicted-valid rows: 0.44–0.61.
- Script diagnostic: 30–33 valid per layer, median cosine with primary 0.94–0.96.
- v_prompt mean-delta cosine with primary: median −0.08 to −0.15. The prompt direction is not the region-unique direction.

**English omission** (post-seal; canonical ref/hyp alignment of the role transcript vs the theta0 text):
- 567 reference EN units are deleted by the baseline, across 20 dialogues (most in CSD0020: 106, and CSD0502: 100).
- 376 (66%) are ≥ 50% covered by predicted acoustic EN (median coverage 1.0). 74 lie entirely beyond the 30 s heard horizon.
- Only 123 have any eligible decoder query attending inside them, and 34 have one assigned EN.
- The acoustic predictor often detects omitted English. The baseline decoder has no corresponding English state to build from;
  none is invented.

**Statistics.**
- Dialogue-cluster bootstrap: B = 10000, seed 240924, 20 sorted dialogues, draws shared across layers and endpoints.
- Bonferroni family 8: quantiles 0.003125 / 0.996875.
- Usable draws: 9,895–9,961 per signal endpoint, reported descriptively.
- The independent auditor reproduced the gates, bootstrap and label.

## Scientific interpretation

1. **Primary bottleneck: decoder-state availability.** It is not region prediction, not SVD stability and not oracle mismatch.
   - Region prediction passes its frozen quality gate (EN precision 0.97, recall 0.58).
   - Within one utterance, too few baseline decoder queries map confidently onto English (or, under the proxy, even onto
     Mandarin) acoustic regions to support any rank ≥ 2 with the frozen count, bin and effective-rank guards.
   - The gold-proxy membership is not better: 16–17 valid utterances against the 60 required. The failure is therefore
     structural to per-utterance construction from the baseline's own prefix states, not a predictor defect.
2. **Is construction feasible without references?** For a small subset of utterances, yes. About 30 per layer produce stable
   vectors, and on the 9–12 utterances where the oracle is also constructible those vectors agree with it (signed cosine
   about 0.8). It is not broadly feasible on FULL300.
3. **What R1 may test:** nothing. R0 authorizes no intervention study, no relaxed thresholds or ranks, no pooled or
   cross-utterance variant and no new predictor. Any redesign would need a separate, human-authorized pre-outcome freeze.
4. **What remains unproven:**
   - Any steering usefulness (never measured here).
   - Any ASR or lexical effect.
   - Feasibility under different construction rules.
   - Long-form behaviour beyond the 30 s horizon.
   - Agreement with manual acoustic segmentation (the oracle is an MMS-FA proxy with unknown aligner revision).
5. **Exact terminal label:** `R0_ORACLE_CONSTRUCTION_INSUFFICIENT`.

Historical conclusions are unchanged: ST-LOC0 LOCAL_EFFECT_ONLY, A2/MECH0 and core v6.

## Engineering and compute

- **Job:** one Slurm job, 58150, on an H100 MIG 3g.40gb. Wall time 11 min 51 s (runtime 704 s).
  - Peak VRAM: 3.66 GB allocated / 3.98 GB reserved.
  - In-allocation row-0 smoke: 1.4 s for row 0, projected total 1,069 s; after 10 rows the projection was 871 s.
- **Counters:**
  - 10,387 LID calls (0 cache hits) and 300 encoder passes.
  - 900 decoder full replays (forced-ZH ×2 + forced-EN).
  - 30,000 CPU constructions (single BLAS thread).
  - 0 autograd calls, optimizer steps or edit hooks.
  - Weights unchanged, grads None, eval mode.
- **Outputs:** 381 MB under `results/inference_cf/r0/run1`. Per row, a JSON record plus an npz with packed bf16 states/heads,
  LID probabilities, masks, vectors, bases and spectra.
- **Tests:** `tests/test_r0_impl.py` (15) includes an end-to-end tiny random large-v3-geometry smoke whose outputs the
  independent auditor reproduced exactly; freeze contract (7); regression total 120/120.

Artifacts: `results/inference_cf/r0/` (`plan_sealed.json`, `prerun_audit.json`, `run1/`, `primary_analysis.json`,
`output_seal.json`, `primary_audit.json`, `oracle_analysis.json` + `_vectors.npz`, `final_audit.json`).
