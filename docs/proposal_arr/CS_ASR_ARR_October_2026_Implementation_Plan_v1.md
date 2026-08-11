# Executable Implementation Plan — Selective Test-Time Intervention for CS-ASR

**Plan version:** v1, 10 August 2026
**Derived from:** `docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md`
**Target:** ARR October 2026, submission 12 October 2026
**Workspace:** `/home/tungnx/cs-asr-steer` (code) · `/mnt/data/tungnx/cs-asr-steer` (all generated data)
**Environment:** `/home/tungnx/miniconda3/envs/acl1`, Slurm partition `main`, 1×H100 80 GB
**New artifacts root:** `/mnt/data/tungnx/cs-asr-steer/artifacts_lss`

---

## 0. What this document is

The v5 proposal states *what* to prove and *why*. This document states *what to build, in
what order, with which files, against which measured numbers, and with which pass/fail
tests* — edited against the code and results that already exist in this workspace.

Three rules govern every section below:

1. **Nothing is assumed about the repository.** Every claim about existing code, artifacts
   or data was verified on disk on 2026-08-10 and is cited by path.
2. **Where the proposal's premises are stale, this plan corrects them** (§1.3) instead of
   silently planning around them.
3. **Every work package ends in a machine-checkable gate** written to
   `artifacts_lss/status/<stage>.json` in the existing gate format
   (`src/csasr/utils/status.py:124` `gate()` / `criterion()`), so "did it pass" is never a
   judgement call made after seeing the numbers.

The user runs all GPU/Slurm jobs. This plan is written so that each work package is a
self-contained, resumable, single-command submission.

---

## 1. Verified repository state (2026-08-10)

### 1.1 Code inventory — what already exists and is reusable

| Area | Path | State | Reuse in this plan |
|---|---|---|---|
| Frozen Whisper loading, frame geometry | `src/csasr/models/whisper.py` | Working, `inference_mode` everywhere, records local weight hash | Reuse unchanged |
| Steering hooks (encoder/decoder, norm-preserving) | `src/csasr/models/hooks.py` | Working, hook-leak assertion included | Reuse; **extend** for per-step decoder gain |
| Frame masks (exact/taper/expand/jitter/global) | `src/csasr/steering/masks.py` | Working, 11 mask kinds | Reuse for the frozen action |
| Encoder-local steering builder | `src/csasr/steering/encoder_hook.py` | Working | **Refactor** (§5.WP7 R1): spans keyed by `utterance_id` (line 53) forbids two candidates of one utterance in a batch |
| Batched decoding + prediction cache w/ provenance | `src/csasr/models/generation.py` | Working, OOM fallback, resumable | Reuse for every decode incl. second pass |
| Teacher-forced forward | `src/csasr/models/generation.py:201` | Working | Reuse for directions and transport |
| PIER / POI taxonomy | `src/csasr/evaluation/pier.py` | Working, 7 error categories | Reuse; **redefine eligibility** (§4.3) |
| MER / EN-WER / ZH-CER | `src/csasr/evaluation/mer.py` | Working | Reuse |
| Correction / corruption / outside-span edits | `src/csasr/evaluation/correction_harm.py` | Working | Reuse; extend with utility \(U_k\) |
| Block bootstrap (grouped + paired) | `src/csasr/evaluation/bootstrap.py` | Working, 10 k resamples supported | Reuse; **extend** for frontier bands |
| Direction accumulators (streaming, checkpointed) | `src/csasr/directions/accumulators.py`, `encoder.py`, `decoder.py` | Working | Reuse; **extend** with pooling weights, residualization, prompt subspace |
| Whisper cross-attention DTW aligner | `src/csasr/data/alignment.py` | Working but **boundary-unreliable** (§1.2) | Keep as one of three aligners |
| MMS-FA CTC forced aligner | `src/csasr/data/ctc_alignment.py` | Working (uroman + own Viterbi) | Keep as second aligner |
| Multi-aligner consensus, canonical sample coordinates | `src/csasr/nat5h/{schema,coordinates,consensus,aligners}.py` | Working; schema `nat5h_alignment_v2` | **Promote to the production alignment path** |
| Qwen3-ForcedAligner adapter | `src/csasr/nat5h/aligners.py:349` | Checkpoint downloaded, **runtime blocked** | Debug → third aligner |
| Splits, manifest, test-split block | `src/csasr/data/{splits,manifest}.py` | Working, `assert_no_test_data` enforced | Reuse; add role layer |
| Stage status / gates / locks | `src/csasr/utils/status.py`, `experiments/_common.py` | Working | Reuse; append new stage names |
| Slurm launchers | `cs_asr_e1_e5.sh`, `cs_asr_nat5h.sh` | Working, phase-by-phase | Template for `cs_asr_lss.sh` |
| Unit tests | `tests/` (90 tests) | Passing under launcher env | Extend, keep green |

### 1.2 Artifact state — what has and has not passed

| Stage | Status | Evidence |
|---|---|---|
| P0 baseline + POI audit | **passed** (job 31413) | `artifacts_v2/status/p0.json`, `reports/p0_baseline_audit.md` |
| E1 alignment | **failed** (job 31574) | `artifacts_v2/status/e1.json`; synthetic within-100 ms 0.060, bias −548.6 ms; CTC-vs-DTW within-100 ms 0.070, median 370 ms |
| E2–E5 | never run in production | `artifacts_v2/status/overview.json` |
| NAT5H exploratory (r2, job 32027) | **completed_no_go** at N1 | `artifacts_nat5h_r2/status/n1_align_smoke.json` |

Frozen primary baseline is `B0_AUTO` (Whisper auto language token, greedy, no timestamps),
`artifacts_v2/baselines/primary_baseline.json`.

### 1.3 Corrections to factual premises in proposal v5

These are not editorial nits: three of them change what W0–W1 must actually do.

| # | v5 statement | Verified reality | Consequence for the plan |
|---|---|---|---|
| 1 | §4.1: "`whisper_dtw` failed … the job ended `completed_no_go` because fewer than two aligners were valid" | The **latest** run (`artifacts_nat5h_r2`, job 32027) had **two mechanically valid families** (`existing_ctc`, `whisper_dtw`). The no-go reason was `insufficient_english_coverage`: 16 consensus EN spans against a threshold of 20, on a 20-utterance smoke | The blocker is **English-span consensus rate at scale**, not aligner availability. WP1 targets span-level agreement and sample size, not aligner procurement |
| 2 | §4.1: "`existing_ctc` … identify its training languages … replace it if it is monolingual" | `existing_ctc` = **MMS-FA** (`/mnt/data/tungnx/models/mms-fa`, `Wav2Vec2ForCTC`, vocab 31) driven by **uroman romanization** of both scripts (`src/csasr/data/ctc_alignment.py:95`) | It is *not* monolingual; do **not** replace it. Instead audit the romanization path for Han (one Han char → multi-letter syllable) as a boundary-error source |
| 3 | §4.1: "obtain an independent second source, such as Qwen3-ForcedAligner" | Qwen3-ForcedAligner-0.6B is **already downloaded** (`/mnt/data/tungnx/cs-asr-steer/models/Qwen3-ForcedAligner-0.6B`); its runtime is deferred/blocked (`artifacts_nat5h_r2/diagnostics/qwen_adapter_smoke.json`) | This is a ½-day debugging task (`sbatch cs_asr_nat5h.sh qwen_smoke`), not a procurement risk |
| 4 | §11.1: utility-label generation is "the dominant compute item … larger than LoRA training" | Measured free decoding on this cluster: **6.0–6.5 utt/s** at `batch_size=16` (`artifacts_v2/runs/p0`: 3360 utts / 563.6 s; 2826 utts / 431.8 s) | 10 k–50 k second passes ≈ **0.5–2.5 GPU-hours**. It is the largest *inference* item but is **not** a schedule threat here. The schedule risk is alignment. Budget accordingly (§7) |
| 5 | §4.2: "at least 1,500 eligible embedded-English units; at least 500 baseline English errors" | Measured (§1.4): eligible **embedded**-English units are 6215 (dev_select) / 2149 (dev_confirm) / 7758 (test, reference-only). Baseline errors 2745 / 979 | Power targets are met with margin — **but only under the corrected eligibility definition** in §4.3. The P0 number 23726 is inflated ≈4× by monolingual `<EN>` utterances |
| 6 | §3.1: "Use conversation/session as the highest-level split unit" | In CS-Dialogue short_wav as indexed here, `conversation_id` ↔ `speaker_id` is **1:1** (train 140/140, dev_select 20/20, dev_confirm 10/10, test 30/30). Verified zero conversation overlap across splits | The existing speaker-disjoint splits already satisfy conversation-disjointness. Use `conversation_id` as the bootstrap cluster key; no re-split needed |
| 7 | §8.1 Gate T: "at least 60% practical coverage" | Deletions are **63%** of embedded-English errors on dev_select CS utterances (1736/2745) | Site D is structurally blind to the majority error type. Expect Gate T to matter less than v5 assumes; E-only is the likely practical system. Plan for that outcome (§10, D-2) |

### 1.4 Measured quantities this plan depends on

Data (from `artifacts_v2/manifests/`, recomputed 2026-08-10):

| Split | Utterances | Conversations = speakers | Hours | CS utterances | Embedded-EN units | Baseline-error units |
|---|---:|---:|---:|---:|---:|---:|
| train | 26 239 | 140 | 68.64 | 4 935 | — (decode in WP2) | — |
| dev_select | 3 360 | 20 | 11.46 | 708 | 6 215 | 2 745 |
| dev_confirm | 2 826 | 10 | 6.82 | 420 | 2 149 | 979 |
| test (locked) | 6 257 | 30 | 16.64 | 1 046 | 7 758 (reference-only) | — (locked) |

Baseline-error composition on **CS utterances** (frozen `B0_AUTO`, embedded-EN units):

| Category | dev_select | dev_confirm | Group |
|---|---:|---:|---|
| `deletion` | 1 736 (63%) | 472 (48%) | **deletion** |
| `wrong_language_substitution` | 443 | 262 | substitution |
| `same_language_substitution` | 332 | 138 | substitution |
| `phonetic_transliteration_or_script` | 108 | 63 | substitution |
| `boundary_error` | 107 | 35 | other |
| `other` | 19 | 9 | other |

Throughput / cost anchors:

- free decoding, no hooks, bs 16: **6.0–6.5 utt/s**;
- steered decoding adds a per-batch hook build; treat as **≥5 utt/s** until measured in WP0;
- E1-style teacher-forced alignment with `output_attentions=True`, bs 8, ~10.8 k utterances: **6 h 10 m** (job 31574) — attention materialization dominates;
- disk free on `/mnt/data`: **11.6 TB**.

### 1.5 Environment constraints (all verified, all still true)

1. `LD_LIBRARY_PATH` must prefix `$CONDA_PREFIX/lib` or scipy/sklearn imports fail
   (`CXXABI_1.3.15`). Both launchers already do this; the new launcher must too.
2. `torchaudio` **fails to import** (`libtorchaudio.so`, torch 2.10 vs torchaudio 2.5.1,
   `artifacts_nat5h_r2/status/n0_preflight.json`). No new component may depend on it —
   use `soundfile` + `librosa` + the in-repo Viterbi.
3. `/tmp` is not shared with compute nodes; anything `srun` touches lives under
   `/mnt/data/tungnx`.
4. `cs_asr_e1_e5.sh` sets `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`. Any new checkpoint
   (e.g. an LID model) must be fetched **on the login node** first, following the
   `src/csasr/experiments/ensure_nat5h_models.py` pattern. The hub is reachable from the
   login node (verified HTTP 200).
5. The workspace is **not a git repository**; provenance falls back to source snapshot
   hashes. Recommendation (§11.3): `git init` in W0 so the ARR reproducibility appendix can
   cite a commit.
6. Available and sufficient: `scikit-learn 1.8.0` (`HistGradientBoostingClassifier`,
   `CalibratedClassifierCV`), `peft 0.18.1`, `transformers 4.57.6`, `torch 2.10.0+cu128`,
   `uroman 1.3.1.1`. **No** lightgbm/xgboost — the plan deliberately uses sklearn's HistGBM
   so no new dependency is introduced.
7. Whisper `generation_config.alignment_heads` is present with **10 heads** — the transport
   experiment (§5.WP5) has the head set it needs.

---

## 2. Proposal → implementation gap map

| v5 component | § | Exists today | To build |
|---|---|---|---|
| Reliable EN/ZH span boundaries | 4.1 | DTW + MMS-FA + consensus machinery; **gate failing** | Scale-up, third aligner, human audit, high-confidence subset (WP1) |
| 4 prompt baselines (auto/zh/en/bilingual) | 4.2 | 2 of 4 (`B0_AUTO`, `B1_ZH`) | `B2_EN`, `B3_BILINGUAL` (WP2) |
| First-pass inference-available cache | 2.2 | partial (logprobs only) | entropy, top-2 margin, cross-attn summaries, encoder states (WP2) |
| Fixed encoder direction Δ^E with pooling + residualization | 2.3 | mean-delta accumulators | Hann/central-mass pooling, nuisance residualization (WP3) |
| Prompt-orthogonal decoder direction Δ^{D⊥} | 2.3 | per-layer decoder direction, no projection | first-EN-BPE positions, prompt subspace \(U^{prompt}\), projection (WP3) |
| Oracle E/D/ED screen + free-decoding utility | 5.1 | E4 pilot/refine/confirm structure | C00/C10/C01/C11 harness on utility \(U_k\) (WP4) |
| Label-permutation null | 5.2 #1 | random + wrong-sign only | within-pair EN/ZH label permutation, ≥5 seeds (WP4) |
| Matched-energy global control | 5.2 #5 | `ENC-GLOBAL-1L` (unmatched energy) | energy-matched global (WP4) |
| Soft cross-attention transport \(r_q\) | 2.7, 2.8 | none (DTW path only) | `lss/steer/transport.py` + step-indexed decoder hook (WP5) |
| Read-only temporal localizer | 2.4 | none | heuristic / off-the-shelf LID / linear / TCN, 3 seeds (WP6) |
| Utility labels \(U_k(a^*)\) | 2.5 | none | candidate-level second-pass label generator (WP7) |
| Utility selector + calibration + τ | 2.5 | none | LogReg + HistGBM + isotonic calibration (WP7) |
| Two-pass inference algorithm | 2.9 | none | `lss/steer/pipeline.py` incl. abstention accounting (WP9) |
| Matched-data LoRA | 6-Exp4 | none | peft LoRA on `loc-train ∪ util-train` (WP8) |
| Learned global rank-one steering | 6-Exp4 | none | 1 vector + scale, transcript CE (WP8) — *also* the SALSA fallback |
| Selectivity frontier with bootstrap bands | 7.4 | bootstrap primitives | per-resample frontier recomputation (WP9) |
| Deletion/substitution stratification | 7.2 | categories exist | stratified reporting everywhere (WP9, WP10) |
| Locked test batch, mechanical | 3.3, 10.0 | test-split block | dry-run script + unlock protocol (WP9/WP10) |
| SEAME transfer | 6-Exp7 | none | optional; go/no-go 7 Sep (§8) |

---

## 3. Target architecture

### 3.1 New package layout

Library code lives in a new package; stage entry points follow the existing
`src/csasr/experiments/<stage>.py` convention so `_common.prepare/finish` and the status
contract are reused unchanged.

```
src/csasr/lss/                       # Localize · Select · Steer
  roles.py               data-role construction and role assertions
  spans.py               consensus span table → frames, high-confidence subset, span I/O
  cache.py               pass-1 frozen cache (states, token stats, attention summaries)
  features.py            selector feature vector φ_k, all inference-available
  directions/
    pooling.py           Hann / central-mass pooling weights w_it
    residualize.py       conversation-balanced nuisance residualization
    prompt_subspace.py   U_k^prompt from forced-EN vs forced-ZH runs; projection
    permute.py           within-pair EN/ZH label-permutation null (≥5 seeds)
  localizer/
    labels.py            frame targets from consensus EN spans (+ boundary tolerance)
    heuristic.py         transcript/timestamp baseline
    lid_baseline.py      off-the-shelf frame LID (MMS-LID sliding window)
    linear.py            linear frame classifier + temporal smoothing
    tcn.py               2-layer temporal convolution (primary), ≥3 seeds
    decode.py            threshold → merge → candidate regions S_1..S_K
    evaluate.py          frame F1/AUPRC, span P/R@IoU, correctable-span recall
  utility/
    labels.py            U_k(a*) generation (second pass per candidate)
    selector.py          LogReg / HistGBM / MLP + calibration + τ selection
    evaluate.py          AUROC/AUPRC, ECE, coverage, net utility
  steer/
    action.py            frozen action a* (site, layer, ρ, mask) + freeze-file I/O
    transport.py         r_q = Σ_t Ā_{q,t} g_t, alignment-head aggregation, E_eff
    decoder_local.py     step-indexed decoder steering hook (new)
    pipeline.py          two-pass localize→select→steer→transcript
  baselines/
    prompts.py           B0/B1/B2/B3
    global_steer.py      fixed global steering, energy-matched variant
    lora.py              matched-data LoRA (peft)
    learned_steer.py     learned global rank-one vector (SALSA fallback)
  report/
    metrics.py           embedded-EN PIER, correction/corruption, harm, retention
    frontier.py          τ sweep + block-bootstrap pointwise bands
    tables.py            Tables 1–6, Figure 2

src/csasr/experiments/
  lss_l0_freeze.py  lss_l1_align.py   lss_l2_baseline.py  lss_l3_directions.py
  lss_l4_oracle.py  lss_l5_transport.py lss_l6_localizer.py lss_l7_utility.py
  lss_l8_baselines.py lss_l9_system.py  lss_l10_test.py

configs/lss/{base,align,baseline,directions,oracle,transport,localizer,utility,systems,test}.yaml
cs_asr_lss.sh                          # phase-by-phase Slurm launcher
```

### 3.2 Stage IDs and status contract

Append to `STAGES` in `src/csasr/utils/status.py`:

```
l0_freeze, l1_align, l2_baseline, l3_directions, l4_oracle,
l5_transport, l6_localizer, l7_utility, l8_baselines, l9_system, l10_test
```

Every stage writes `artifacts_lss/status/<stage>.json` with `gate.criteria[]` in the
existing schema, and `require_passed()` blocks downstream stages. Adopt one NAT5H
convention that E1–E5 lacks: the state vocabulary gains `completed_no_go` (a scientifically
valid negative that is *not* a crash) — see `src/csasr/nat5h/statusing.py:13`. Extend
`ALLOWED` in `utils/status.py` accordingly.

### 3.3 The freeze file — the mechanism that makes W6 mechanical

`artifacts_lss/freeze/frozen_pipeline.json` is written incrementally by L3/L4/L5/L6/L7 and
sealed at the end of L9. It records, with a SHA-256 over its own canonical serialization:

```jsonc
{
  "eligibility":   {"unit": "embedded_en", "matrix_language_required": true, ...},
  "normalization": {"version": "v1", "hash": "..."},
  "decoding":      {"task":"transcribe","num_beams":1,"temperature":0.0,"batch_size":16,...},
  "alignment":     {"source":"consensus_v2","min_aligners":2,"tol_ms":100,"subset_hash":"..."},
  "action":        {"site":"E","layer":27,"rho":1.0,"mask":"EXACT_TAPER","shoulder_ms":100,
                    "norm_preserve":true,"direction_artifact":"..."},
  "transport":     {"enabled":false,"heads":[[7,0],...],"agg":"mean","reason":"gate_T_failed"},
  "localizer":     {"model":"tcn","ckpt":"...","threshold":0.42,"min_dur_ms":120,
                    "merge_gap_ms":80,"max_candidates":4},
  "selector":      {"model":"histgbm","ckpt":"...","calibration":"isotonic","tau":0.61,
                    "features":["..."]},
  "metrics":       {"bootstrap":{"n":10000,"seed":342,"cluster":"conversation_id"}},
  "sealed_at": "...", "sha256": "..."
}
```

`lss_l10_test.py` refuses to run unless (a) the freeze file is sealed, (b) its hash matches
the hash recorded by the W5 dry-run, and (c) `--i-am-running-the-locked-test` is passed.
This is the concrete implementation of v5 §3.3 and §10.0.

### 3.4 Launcher

`cs_asr_lss.sh <phase>` mirrors `cs_asr_nat5h.sh`: sets `LD_LIBRARY_PATH`, `PYTHONPATH`,
`TMPDIR` under `/mnt/data`, HF caches, single-active-job guard, `USR1` → clean checkpoint,
and dispatches one stage. Phases: `preflight tests l0 l1 l2 l3 l4 l5 l6 l7 l8 l9 dryrun l10
summary`. No `auto` mode: every stage boundary is an inspection boundary.

---

## 4. Data roles

### 4.1 Role mapping (concrete, with counts)

| v5 role | Source | Conversations | CS utterances | Purpose |
|---|---|---:|---:|---|
| `D-construct` | official `train`, conversations 1–70 | 70 | ≈2 470 | Δ^E, Δ^{D⊥}, nuisance models, prompt subspace, transport head rule |
| `D-router / loc-train` | official `train`, conversations 71–105 | 35 | ≈1 230 | temporal localizer training |
| `D-router / util-train` | official `train`, conversations 106–130 | 25 | ≈880 | candidate generation + utility labels + selector training |
| `D-router / router-calib` | official `train`, conversations 131–140 | 10 | ≈350 | probability calibration + τ |
| `D-dev-select` | `dev_select` | 20 | 708 | layer/dose selection, early oracle headroom |
| `D-dev-confirm` | `dev_confirm` | 10 | 420 | specificity controls, freeze a*, full dry-run |
| `D-test` | official `test` | 30 | 1 046 | one locked batch |

Assignment is deterministic: sort `conversation_id` ascending, then slice. Implemented in
`lss/roles.py:build_roles(cfg) -> dict[str, pd.DataFrame]`, written to
`artifacts_lss/manifests/role_<name>.parquet` plus `role_report.json`. Assertions
(`lss/roles.py:assert_roles_disjoint`): pairwise conversation-disjoint; no `test` row in any
non-`D-test` role; every row's `official_split` consistent with its role.

**Why train is split three ways and not cross-fitted:** v5 §3.2 permits conversation-level
cross-fitting if data are too small. With 140 training conversations and ~4 935 CS
utterances, fixed subsets are affordable and simpler to defend. Cross-fitting stays as the
documented fallback if `util-train` candidate counts come in below 3 000 (checked in WP7's
gate evidence, not after seeing selector results).

### 4.2 Eligible lexical unit — the decision that changes headline numbers

**Problem found.** `evaluation/pier.py:reference_pois` counts *every* English content unit
in the reference. CS-Dialogue's dev split contains 1 710 `<EN>` (monolingual English)
utterances, so P0's 23 726 "EN POIs" on dev_select are **73% monolingual-English units**,
not embedded code-switches. PIER computed that way answers a different question from the
one the paper asks.

**Decision (freeze before any steering result).** An *eligible embedded-English unit* is an
English content unit `u` in reference `Y*` such that:

1. `Y*` contains ≥1 Mandarin content unit (i.e. `contains_code_switch == True`); and
2. `u` lies inside an alignment-consensus span (§5.WP1) when a *local* claim is made
   (oracle/steered results); corpus-level PIER without local intervention reports both the
   consensus-restricted and unrestricted versions.

Implemented as `lss/report/metrics.py:eligible_units(manifest, units, level)` with
`level ∈ {"corpus", "local"}`. Both numbers appear in Table 1; the primary endpoint uses
the embedded definition. Measured sizes are in §1.4 — every v5 §4.2 power target is met.

### 4.3 Deletion / substitution grouping (required by v5 §7.2)

Fixed mapping over the existing taxonomy in `evaluation/pier.py:20`:

- **deletion** ← `deletion`
- **substitution** ← `wrong_language_substitution`, `phonetic_transliteration_or_script`, `same_language_substitution`
- **other** ← `boundary_error`, `insertion_near_poi`, `other` (reported, never pooled into either)

### 4.4 Test lock and unlock protocol

`experiment.prohibit_test_split: true` remains set in every config except
`configs/lss/test.yaml`. Unlocking requires all four of:

1. `freeze/frozen_pipeline.json` sealed and hash-matching the dry-run record;
2. `configs/lss/test.yaml` with `prohibit_test_split: false`;
3. CLI flag `--i-am-running-the-locked-test`;
4. an empty `artifacts_lss/status/l10_test.json` (the stage refuses to overwrite a previous
   locked run — a second test run is a new, explicitly named artifacts root).

Reference-only statistics on `D-test` (unit counts, cluster counts, MDE) are permitted
before the lock and are produced by `lss_l2_baseline.py --test-reference-only`, which loads
transcripts and **never** instantiates the model.

---

## 5. Work packages

Each WP lists: purpose · maps to · inputs · new code · gate · outputs · tests · compute.

---

### WP0 — Freeze, scaffolding, and the measured pilot  *(stage `l0_freeze`, W0)*

**Purpose.** Make every later stage mechanical: fix the decisions v5 §13 items 3–5 require,
create the package skeleton, and replace estimated costs with measured ones.

**Maps to.** v5 §10 W0, §11, §13.3–13.5.

**New code.**

- `src/csasr/lss/roles.py` — `build_roles`, `assert_roles_disjoint`, `role_of(utterance_id)`.
- `src/csasr/lss/steer/action.py` — freeze-file read/write/seal/verify (`FrozenPipeline`
  dataclass, `seal()`, `verify(expected_sha)`).
- `src/csasr/experiments/lss_l0_freeze.py` — writes role manifests, the initial freeze file,
  and runs the 200-utterance compute pilot.
- `configs/lss/base.yaml` (+ the per-stage configs listed in §3.1).
- `cs_asr_lss.sh`.
- `utils/status.py`: append the 11 stage names; add `completed_no_go` to `ALLOWED`.

**Compute pilot (200 utterances, `D-construct`, reported in `metrics/l0_pilot.json`).**
Measure and record: teacher-forced examples/s with recorder hooks; free-decoding RTF and
utt/s at bs 8/16/24; encoder-only forward utt/s; steered-decode utt/s (bs 16); one full
localize→second-pass cycle per candidate; GPU peak memory per mode; bytes/utterance for the
encoder-state cache and the attention summary; NFS read throughput for audio.

**Gate `l0_freeze` (all required).**

| Criterion | Threshold |
|---|---|
| role manifests written, pairwise conversation-disjoint | true |
| no test-split row in any non-test role | true |
| freeze file created, schema-valid, unsealed | true |
| pilot measured on ≥200 utterances | true |
| extrapolated `util-train` label-generation cost | ≤ 6 GPU-h (else shrink `util-train` conversations per v5 §11.1, never K̄) |
| `pytest -q` | 100% pass |

**Outputs.** `manifests/role_*.parquet`, `manifests/role_report.json`,
`freeze/frozen_pipeline.json`, `metrics/l0_pilot.json`, `reports/l0_freeze.md`.

**Tests.** `tests/test_lss_roles.py` (disjointness, determinism, test-block),
`tests/test_lss_freeze.py` (seal/verify round-trip, tamper detection).

**Compute.** ≤ 1 GPU-hour.

---

### WP1 — Alignment repair → **Gate A**  *(stage `l1_align`, W0–W1; critical path)*

**Purpose.** Produce span boundaries good enough that "local" means something, or produce a
defensible restricted subset on which it does. Everything downstream is conditional on this.

**Maps to.** v5 §4.1, §6-Exp0, Gate A, §13.1–13.2, §13.6.

**Inputs.** `role_*` manifests; MMS-FA at `/mnt/data/tungnx/models/mms-fa`;
Qwen3-ForcedAligner at `/mnt/data/tungnx/cs-asr-steer/models/Qwen3-ForcedAligner-0.6B`;
existing NAT5H consensus code.

**Work, in order.**

1. **Qwen unblock.** `sbatch cs_asr_lss.sh l1 --aligner-smoke qwen` (wrapping the existing
   `cs_asr_nat5h.sh qwen_smoke` path). Deliverable: `diagnostics/qwen_adapter_smoke.json`
   with `state: ok` and ≥20 aligned utterances, or a written statement that the third source
   is unavailable and the gate runs 2-of-2.
2. **Coordinate-bug hypothesis test.** The production E1 audit reported CTC-vs-DTW
   +1354 ms systematic offset and 7% within-100 ms, while NAT5H v2 — same two aligners, new
   canonical sample-index coordinates — accepts 39% of units at a 200 ms tolerance. This is
   consistent with a **coordinate-mapping defect in the E1 audit path**, not with aligner
   failure. Recompute the E1 CTC-vs-DTW comparison under `nat5h/coordinates.py` and record
   the delta in `diagnostics/coordinate_hypothesis.json`. If agreement jumps, E1's failure
   is explained and the fix is to retire the E1 audit path in favour of consensus v2.
3. **DTW `adjacent_overlap` repair.** 93/564 units (16.5%) are invalidated by adjacent
   overlap (`diagnostics/whisper_dtw_mapping_summary.json`). Apply the existing resolver
   (`data/alignment.py:366 resolve_boundary_overlaps`) inside the NAT5H candidate path and
   re-measure `valid_unit_coverage`.
4. **Scale-up.** Run 3 aligner families over: 300 CS utterances of `D-dev-select` (gate
   estimation), plus all CS utterances of `D-construct`, `loc-train`, `util-train`,
   `router-calib`, `D-dev-confirm` (label production). Consensus per
   `nat5h/consensus.py:build_unit_consensus_v2` at **±100 ms** (tightened from the smoke's
   200 ms) on both edges, ≥2 families.
5. **Class-balanced human audit, ≥200 units** (100 EN, 100 ZH), sampled stratified by span
   duration. Reuse the E1 audit pack generator (clips + spectrogram figures +
   `verdicts_template.csv`). This is the only *external* boundary-truth source and v5 §4.1
   makes it non-optional.
6. **Synthetic splice check** retained from E1 (silence-trimmed protocol,
   `configs/experiments/e1_alignment.yaml` synthetic block) as a second objective source.
7. **Freeze the high-confidence subset.** `spans.py:high_confidence_subset()` writes
   `alignments/consensus_spans.parquet` with columns
   `utterance_id, unit_id, language, start_sample, end_sample, start_frame, end_frame,
   n_aligners, max_edge_disagreement_ms, confidence_bin, safe_interior_start/end,
   steering_mask_start/end`. Confidence bins: `high` (≥2 aligners ≤50 ms), `medium`
   (≤100 ms), `low` (≤200 ms, excluded from primary claims).

**Gate A (`l1_align`).**

| # | Criterion | Threshold | Note |
|---|---|---|---|
| A1 | unit table validates (`data/alignment.py:309`) | true | |
| A2 | mechanically valid aligner families | ≥2 | 3 preferred |
| A3 | per-family `valid_unit_coverage` | ≥0.95 | currently 0.979 CTC / 0.835 DTW → step 3 must fix DTW |
| A4 | EN units with `high`+`medium` consensus, on the 300-utterance gate sample | ≥0.60 | **reported, and it defines the science subset** |
| A5 | human audit units scored | ≥200, class-balanced | |
| A6 | human "usable" fraction | ≥0.90 | |
| A7 | median absolute boundary error vs human | ≤100 ms | |
| A8 | p90 absolute boundary error vs human | ≤200 ms | |
| A9 | EN-vs-ZH median boundary-error difference | ≤30 ms, else stratified reporting mandatory | v5 §4.1 |
| A10 | synthetic splice within 100 ms | ≥0.90 | E1 measured 0.06 → must be re-measured under consensus |
| A11 | synthetic absolute bias | ≤50 ms | E1 measured 548.6 ms |

**Failure response (pre-decided).** If A4 ≥0.35 but A7/A8 fail on `low` bins only: restrict
all local work to `high`+`medium` bins, record the reduced coverage in Table 1, and state
the restriction as a limitation. If A4 < 0.35 or A7 fails on `high` bins: **stop and repair**
— do not weaken the gate (v5 §10.1). If ≥1 extra week is consumed here, drop the entire
optional list in §8.3 immediately.

**Outputs.** `alignments/consensus_spans.parquet`, `metrics/l1_*`, `audit/l1/{records.jsonl,
clips/,figures/,verdicts_template.csv}`, `reports/l1_alignment.md`, `diagnostics/*`.

**Tests.** `tests/test_lss_spans.py` — consensus tolerance logic, confidence binning,
sample↔frame round-trip (extends `tests/test_nat5h_alignment_v2.py`), safe-interior erosion
never yields an empty mask.

**Compute.** 6–10 GPU-hours for the aligner sweep (DTW dominates); human audit is human
time, ~4–6 h, and can overlap WP2/WP3.

---

### WP2 — First-pass cache, prompt baselines, power  *(stage `l2_baseline`, W0–W1)*

**Purpose.** Everything the test-time system is allowed to see, computed once and cached;
plus the four prompt baselines and the power statement.

**Maps to.** v5 §2.2, §4.2, §6-Exp0.

**New code.**

- `lss/baselines/prompts.py` — `B0_AUTO` (existing), `B1_ZH` (existing), **`B2_EN`**
  (`language="en"`), **`B3_BILINGUAL`** (forced prefix `<|zh|><|en|><|transcribe|>
  <|notimestamps|>`, built via `data/alignment.py:134 build_prefix` extended to accept a
  language *sequence*; this is non-standard and is labelled as such in the paper).
- `lss/cache.py`:
  - `cache_first_pass(bundle, manifest, cfg, out_dir)` → per-utterance parquet + `.npz`:
    - `token_ids`, `token_logprobs` (existing), **`token_entropy`**, **`token_top2_margin`**
      (from `scores` inside `decode_batch`; requires returning per-step distributions —
      extend `models/generation.py:decode_batch` with `collect_step_stats=True` that
      computes entropy and top-2 margin *on device* and keeps only the scalars);
    - encoder states at the preregistered layers (fp16, valid frames only);
    - **cross-attention summaries**: for the 10 alignment heads only, the per-step argmax
      frame, top-1 mass, and entropy over frames — captured by a forward hook on the decoder
      cross-attention modules, **not** by `output_attentions=True` (E1 measured ~2.9 GB per
      utterance for the full tensor; the hook keeps it at ~KB).
- `lss_l2_baseline.py` — runs the 4 prompt systems on `D-dev-select`, `D-dev-confirm`,
  `D-construct`, `loc-train`, `util-train`, `router-calib`; runs the cache; computes
  reference-only test statistics and the MDE.
- `lss/report/metrics.py` — embedded-EN PIER, MER, EN-WER, ZH-CER, correction/corruption,
  outside-span harm, retention; all stratified by deletion/substitution.

**Power statement (v5 §4.2).** Block-bootstrap MDE for ΔPIER on `D-test`, clustered by
`conversation_id`, 10 000 resamples, using the frozen `B0_AUTO` error pattern from
`dev_confirm` as the plug-in. Must satisfy: MDE ≤ ½ of the plausible oracle gain measured in
WP4. Reported before any intervention is run on test.

**Gate `l2_baseline`.**

| Criterion | Threshold |
|---|---|
| 4 prompt systems decoded on all development roles | true |
| eligible embedded-EN units on `D-test` (reference-only) | ≥1 500 (measured: 7 758) |
| conversation clusters on `D-test` | ≥30 (measured: 30) |
| baseline embedded-EN errors on `D-dev-confirm` | ≥500 (measured: 979) |
| cache completeness (states + token stats + attn summary) | ≥0.99 of targeted utterances |
| MDE(ΔPIER) | ≤ ½ oracle gain — *evaluated at end of WP4, recorded here as pending* |

**Outputs.** `baselines/{B0_AUTO,B1_ZH,B2_EN,B3_BILINGUAL}/<role>.parquet`,
`cache/pass1/<role>/*.parquet|npz`, `metrics/l2_power.json`, `reports/l2_baseline.md`.

**Tests.** `tests/test_lss_cache.py` — entropy/margin agree with a slow reference
implementation on a 5-utterance fixture; attention-summary hook produces the same argmax as
`output_attentions=True` on 2 utterances; cache is resumable and provenance-guarded.

**Compute.** 4 systems × ~12 k utterances ÷ 6 utt/s ≈ **2.2 GPU-h**, plus ~1 GPU-h caching.
Storage: encoder states ≈ 0.8 MB/utt fp16 for one layer → ~10 GB per layer over 12 k
utterances; cache **two** layers maximum until WP3 selects one.

---

### WP3 — Fixed directions  *(stage `l3_directions`, W1–W2)*

**Purpose.** Construct Δ^E and Δ^{D⊥} on `D-construct` only, with the estimator v5 §2.3
specifies — not the simple mean the current code implements.

**Maps to.** v5 §2.3, §5.2.

**New code.**

- `lss/directions/pooling.py` — `hann_weights(n)`, `central_mass_weights(n, keep=0.6)`;
  pooled state \(x_{i,l}=\sum_t w_{it} h_{l,t}\).
- `lss/directions/residualize.py` — `fit_nuisance(states, meta)` /
  `apply(states)`; nuisance covariates: conversation id (one-hot, conversation-balanced
  weights \(\omega_i\) = inverse conversation size), speaker gender, span duration bin,
  utterance duration, span position in utterance. Ridge with a fixed λ picked on
  `D-construct` only.
- `lss/directions/prompt_subspace.py` — collect post-cross-attention decoder states at the
  same positions under forced-EN and forced-ZH prompts; \(U_k^{prompt}\) = top-r left
  singular vectors of the stacked prompt-difference matrix, **r = 2 preregistered**;
  `project_out(d, U)`.
- `lss/directions/permute.py` — `label_permuted_direction(pairs, seed)`: swap EN/ZH labels
  *within* nuisance-matched pairs, redo residualization and aggregation, normalize; seeds
  242–246.
- `lss_l3_directions.py` — orchestrates; extends `directions/decoder.py` to select the
  **first English BPE after a Mandarin prefix** (\(q_i^{M\to EN}\)) and a matched Mandarin
  continuation (\(r_i^{M\to M}\)) rather than all EN/ZH token positions.

**Scope discipline (v5 §2.3).** ≤4 encoder layers `[15, 23, 27, 31]` (already the repo
default), ≤4 decoder layers `[8, 16, 24, 31]`, strengths ρ ∈ {0.5, 1, 2}, paired mean only,
rank one only. Anything else is appendix-only and must not consume W2 GPU time.

**Gate `l3_directions`.**

| Criterion | Threshold |
|---|---|
| all directions finite, unit norm | true |
| cross-seed minimum cosine (seeds 42–46) | ≥0.90 |
| EN/ZH frame counts per layer | ≥ config minimum |
| prompt subspace explains ≥80% of forced-EN−forced-ZH variance at r=2 | true |
| \(\|\widetilde d^{D\perp}\|/\|\widetilde d^{D,nat}\|\) | ≥0.30 (if lower, the decoder direction is essentially the prompt effect → Site D claim is dropped, not rescued) |
| label-permuted null directions built for ≥5 seeds | true |
| no `D-dev-*` or test row contributed | true (assertion) |

**Outputs.** `directions/encoder/layer_*/…pt`, `directions/decoder/layer_*/…pt`,
`directions/nulls/*`, `metrics/l3_direction_stats.json`, `reports/l3_directions.md`.

**Tests.** `tests/test_lss_directions.py` — pooling weights sum to 1 and are symmetric;
residualization is idempotent on already-residualized input; `project_out` output is
orthogonal to `U` to 1e-6; permutation null has near-zero cosine with the real direction on
synthetic data with no true signal.

**Compute.** ~2–3 GPU-h (encoder accumulation over `D-construct` CS utterances + decoder
teacher-forced passes ×3 prompt conditions).

---

### WP4 — Oracle screen and specificity → **Gate B**  *(stage `l4_oracle`, W2–W3)*

**Purpose.** Decide whether *any* fixed local action repairs real errors, choose \(a^*\),
and kill trivial explanations. If this fails, the paper pivots (v5 §8.2 fallback claim) and
WP6–WP9 are not built.

**Maps to.** v5 §5.1, §5.2, §5.3, §6-Exp1, Gate B.

**Design.** Targets are eligible embedded-EN units on `D-dev-select` (selection) and
`D-dev-confirm` (confirmation), restricted to the `high`+`medium` alignment bins, one target
unit per utterance (K=1). Conditions C00/C10/C01/C11 × {4 encoder layers} × {ρ ∈ 0.5,1,2}
for selection; a single frozen configuration for confirmation.

**Primary quantity is free-decoding utility**, not teacher-forced likelihood:

\[
U = N_{corrected} - \eta N_{local\ harm} - \kappa N_{outside\ harm},\qquad \eta=\kappa=1
\]

with 0.5 / 2 reported as sensitivity only. Teacher-forced screening may be used to prune the
layer × ρ grid before free decoding, and this pruning is declared in the report.

**Required controls, all at matched realized energy \(E_{eff}=\rho^2\|\Delta\|^2\sum_t g_t^2\):**

| Control | Exists | Action |
|---|---|---|
| within-pair label permutation (≥5 seeds) | no | WP3 `permute.py` |
| matched-norm random directions | yes (`ENC-RAND-1L`) | reuse, 5 seeds |
| opposite sign | yes (`ENC-WRONG-1L`) | reuse |
| wrong location | yes (`ENC-RANDLOC-1L`) | reuse; promote from advisory to **required** |
| global matched-energy | partial (`ENC-GLOBAL-1L` is unmatched) | new: scale \(\rho_{global}=\rho\sqrt{\sum_t g_t^2/\;n_{valid}}\) |
| preservation set | partial | reuse `correction_harm.py` + monolingual ZH/EN WER |

**Note on `norm_preserve`.** `models/hooks.py:apply_steering` renormalizes to the original
norm by default and E4 used that convention. Keep it, record it in the freeze file, and
compute \(E_{eff}\) from the **realized** delta (post-renormalization) so the matched-energy
controls are honest.

**Gate B (`l4_oracle`) — v5 §8.1 statistical form.**

| Criterion | Threshold |
|---|---|
| free-decoding net utility on `D-dev-confirm` | >0 with conversation-block bootstrap 95% CI excluding 0 |
| corrections vs corruptions | corrections > corruptions |
| correct direction vs **every** null (label-permuted, random, sign, wrong-location, global-matched) | paired bootstrap difference CI excludes 0 for each |
| dose stability | effect present at two adjacent non-extreme ρ, or stable ±25% around the selected ρ |
| at least one action in {E, D, ED} with positive net utility | true |
| deletion / substitution stratified results reported | true (reporting requirement, not a threshold) |

**Failure response.** Stop the method pipeline (v5 §5.3). Switch to the fallback manuscript:
oracle-local correctability analysis + the localization-ceiling study, which reuses WP1,
WP2, WP6 only.

**Outputs.** `metrics/l4_all_runs.parquet` (one row per system × layer × ρ × phase),
`metrics/l4_selected_action.json`, `predictions/l4/*`, `reports/l4_oracle.md`; freeze file
gains the `action` block.

**Tests.** `tests/test_lss_oracle.py` — utility arithmetic on hand-built cases; energy
matching within 1% between local and global-matched controls; the gate function returns
`False` when any null is not beaten.

**Compute.** selection: ~12 configs × ~700 targets ≈ 8 400 decodes ≈ 25 min; with controls
and both phases ≈ **3–5 GPU-h**.

---

### WP5 — Soft cross-attention transport → **Gate T**  *(stage `l5_transport`, W1 pilot → W2 decision)*

**Purpose.** Decide early whether the decoder site is reachable, so W3+ is not spent on a
dead branch.

**Maps to.** v5 §2.7, §2.8, Gate T.

**New code.**

- `lss/steer/transport.py`
  - `aggregate_attention(cache, heads, layers, sharpen=None) -> Ā  # (Q, T)`
    using the 10 published alignment heads; aggregation `mean` (preregistered), optional
    temperature sharpening declared before use.
  - `transport_gain(Ā, g) -> r_q` (no unit-mass renormalization, per v5 §2.7).
  - `effective_energy(rho_D, delta, r) -> E_eff`.
  - `diagnostics(...)` → top-1/top-3 hit rate on the gold first-BPE step, ±1 agreement,
    on-span vs off-span attention mass ratio, usable coverage, \(E_{eff}\) distribution,
    all stratified by alignment-confidence bin and by deletion/substitution/correct.
- `lss/steer/decoder_local.py` — `DecoderTransportSteeringHook`: like
  `models/hooks.py:DecoderSteeringHook` but keeps a per-batch step counter (each generation
  call has `T == 1` after prefill) and applies gain `r[b, step]`.

**The circularity that v5 does not address, and its resolution.** \(r_q\) needs
cross-attention, but the second pass's cross-attention does not exist until the second pass
runs. Resolution, preregistered: compute \(\bar A\) from the **first pass** (cached in WP2),
apply \(r_q\) by step index during the second pass, and **stop applying** once the second
pass's token sequence diverges from the first pass's (compare generated ids per step; on
divergence, zero the gain for all later steps). Divergence position and the fraction of
steered steps lost to divergence are reported. This keeps the intervention causal and
inference-legal.

**Gate T (`l5_transport`), run on the `high` alignment bin only** (v5 §2.8 fix #3).

| Criterion | Threshold |
|---|---|
| gold first-BPE step in top-3 under \(r_q\) | ≥0.60 |
| agreement within \(q^*\pm1\) | ≥0.50 |
| on-span / off-span attention mass ratio | ≥2.0 |
| practical decoder coverage | ≥0.60 |
| result reported stratified by alignment confidence | true |

**Failure response (pre-decided).** E-only becomes the practical system; Site D survives
only as a small oracle analysis in the appendix; the freeze file records
`transport.enabled=false` with the reason. Given §1.3 #7 (63% deletions), **plan for this
outcome** — it is the modal case, not the failure case.

**Outputs.** `metrics/l5_transport.parquet`, `reports/l5_transport.md`, freeze-file
`transport` block.

**Tests.** `tests/test_lss_transport.py` — `r_q` equals a hand-computed value on a 3×4 toy
attention matrix; the step-indexed hook applies exactly the intended per-step gains under a
mocked generation loop; divergence detection zeroes the tail.

**Compute.** ~1–2 GPU-h (uses WP2's cached attention summaries; only the diagnostic needs
extra teacher-forced passes on ~300 utterances).

---

### WP6 — Temporal CS localizer → **Gate C**  *(stage `l6_localizer`, W2–W3)*

**Purpose.** Find candidate embedded-English regions from frozen encoder states, including
the regions the first pass **deletes** — which is 63% of the errors here, and the single
strongest argument for the module.

**Maps to.** v5 §2.4, §6-Exp2, Gate C.

**Labels.** `localizer/labels.py:frame_targets(consensus_spans, geometry, tolerance_ms=40)`
— positive on consensus EN spans (all of them: baseline-correct, incorrect, deleted, and
Mandarin-substituted, per v5 §2.4), label-smoothed within ±2 frames of an edge, ignore-mask
on `low`-confidence spans.

**Four localizers.**

1. `heuristic.py` — enumerate spans from the first-pass transcript's English tokens and
   their timestamps. **By construction it cannot propose a deleted region** — this is the
   comparison that carries the module's value claim.
2. `lid_baseline.py` — off-the-shelf frame LID, **no retraining** (v5 fix #2, non-optional).
   Primary: `facebook/mms-lid-126` over sliding windows (0.5 s window, 0.1 s hop) →
   per-frame P(en) by interpolation. Fallback if the checkpoint cannot be staged: Whisper's
   own `detect_language` over the same sliding windows (self-contained, no download).
   **Must be fetched on the login node in W0** (`ensure_lss_models.py`, patterned on
   `ensure_nat5h_models.py`) because compute nodes run with `HF_HUB_OFFLINE=1`.
3. `linear.py` — logistic frame classifier on the frozen layer's states + median smoothing.
4. `tcn.py` — 2-layer temporal convolution (kernel 5, dilation 1/2, hidden 256,
   ≈0.9 M params), weighted BCE or focal loss, balanced span sampling, **3 seeds**
   (v5 fix #8), trained on `loc-train` only, early-stopped on a `loc-train` conversation
   holdout — never on `router-calib`.

**Candidate decoding.** `decode.py:candidates(m, thr, min_dur_ms, merge_gap_ms, max_k)`;
`min_dur_ms`, `merge_gap_ms`, `max_k` frozen on development data; **`max_k = 4`**
preregistered (this is the \(\bar K\) that sets WP7's cost).

**Metrics (v5 §6-Exp2).** Frame F1 and AUPRC; span precision/recall at IoU 0.5 (frozen);
median boundary error; candidates per utterance; recall of baseline-EN errors; recall of
**deleted or Mandarin-substituted** spans; **recall of oracle-correctable spans** (the set
WP4 proved repairable). Learned models report mean ± spread over 3 seeds.

**Gate C (`l6_localizer`).**

| Criterion | Threshold |
|---|---|
| recall of oracle-correctable spans (primary localizer) | ≥0.70 |
| candidates per utterance at that operating point | ≤4 |
| primary localizer beats the transcript heuristic on deleted-span recall | paired bootstrap CI excludes 0 |
| primary localizer vs off-the-shelf LID on correctable-span recall | reported with CI — **either outcome is publishable, ignorance is not** (v5 §2.4) |
| seed spread (3 seeds) of correctable-span recall | ≤0.05 |
| no `util-train`/`router-calib`/dev/test conversation in training | true (assertion) |

**Failure response.** Report localization as the bottleneck (v5 §8.3 row 3); the automatic
claim is limited; keep the oracle result as the paper's core.

**Outputs.** `models/localizer_{linear,tcn}_seed*.pt`, `metrics/l6_localizer.parquet`,
`metrics/l6_candidates_<role>.parquet`, `reports/l6_localizer.md`, freeze-file `localizer`
block.

**Tests.** `tests/test_lss_localizer.py` — label generation matches spans on a fixture;
`candidates()` respects min-duration/merge-gap/max-k; a perfect-score input yields exactly
the gold spans; heuristic returns ∅ for a deleted-word fixture.

**Compute.** Feature extraction is already cached (WP2). Training all four localizers ×3
seeds: **<1 GPU-h**. LID baseline inference over ~5 k utterances: ~1 GPU-h.

---

### WP7 — Utility labels and selector → **Gate D**  *(stage `l7_utility`, W4–W5)*

**Purpose.** Learn *whether* the frozen action helps, from actual intervention outcomes —
including on the localizer's false positives.

**Maps to.** v5 §2.5, §6-Exp3, §11.1, Gate D.

**R1 — required refactor (blocking).** `steering/encoder_hook.py:53` looks up
`self.spans[utterance_id]`, so a batch cannot contain two candidates of the same utterance.
Change `EncoderLocalSteering` to key spans by a **candidate id** column carried on the batch
frame (`candidate_id`), falling back to `utterance_id` when absent so E4/E5 stay green.
Without this, candidate-level second passes serialize and the WP7 cost estimate is wrong.

**Label generation (`utility/labels.py`).**

- Candidates come from running the **frozen** localizer on `util-train` — never from gold
  spans (v5 §2.5). Generate at a **looser** threshold than the operating point
  (`thr_label = thr_op − 0.10`, recorded) so false positives are represented.
- For each candidate: one free-decoding second pass with the frozen action \(a^*\); compute
  \(U_k\) with η=κ=1 via `report/metrics.py`.
- Realized class proportions are **recorded in the gate evidence**: candidates on
  baseline-incorrect EN spans / baseline-correct EN spans / Mandarin or non-CS regions /
  candidates causing outside-span harm.
- Counting convention (v5 §2.5): \(N_{corrected}\in\{0,1\}\) under K=1 while harm can exceed
  1 → the utility distribution is asymmetric by construction. `reports/l7_utility.md` plots
  it in W4; do not discover this in W5.

**Cost (measured, not guessed).** `util-train` ≈880 CS utterances × K̄≤4 ≈ **3 500 second
passes ≈ 10 min** at 6 utt/s. Even including `router-calib` and a 3× safety factor this is
**<1.5 GPU-h**. If WP0's pilot contradicts this by >3×, shrink `util-train` conversations —
never K̄ (v5 §11.1).

**Features φ_k (`lss/features.py`), all inference-available, all sourced concretely:**

| Feature | Source |
|---|---|
| localizer score (max, mean), span duration | WP6 |
| encoder English-language margin | projection of pooled span states on Δ^E (E3-style machinery) |
| first-pass token language/script at the span | first-pass tokens + `data/language_tags.tag_unit` |
| token confidence, entropy, top-2 margin | WP2 cache |
| encoder–decoder disagreement | encoder margin sign vs decoded token language |
| cross-attention concentration | WP2 attention summary / WP5 |
| position in utterance, boundary confidence | span geometry + consensus edge disagreement |
| term-frequency proxy | frequency of the candidate surface in **training** transcripts only; never a test reference |

**Selector (`utility/selector.py`).** `LogisticRegression` (baseline) and
`HistGradientBoostingClassifier` (primary, sklearn 1.8 — no new dependency); 2-layer MLP as a
capacity ablation only. Probability calibration with `CalibratedClassifierCV(method="isotonic")`
fitted on `router-calib`. τ chosen on `router-calib` by maximizing net utility, then swept
for the frontier.

**Comparison arms (v5 §6-Exp3).** (1) steer every detected span; (2) uncertainty-only;
(3) language/localizer-score-only; (4) full utility selector; (5) full selector + calibrated
abstention.

**Gate D (`l7_utility`).**

| Criterion | Threshold |
|---|---|
| utility labels generated | ≥3 000 candidates |
| all four candidate classes present | each ≥5% of labels |
| selector AUROC for \(U_k>0\) on `router-calib` | ≥0.65 |
| expected calibration error | ≤0.05 |
| **net utility at matched coverage vs steer-all and vs uncertainty-only** | positive difference, paired bootstrap CI excludes 0 |
| no `loc-train` conversation leaked into `util-train` | true (assertion) |

**Failure response.** Simplify to local steer-all and weaken the selector claim (v5 §8.1
Gate D response). The paper survives; the "whether" contribution does not.

**Outputs.** `metrics/l7_utility_labels.parquet` (schema in Appendix C),
`models/selector_*.joblib`, `metrics/l7_selector.json`, `reports/l7_utility.md`, freeze-file
`selector` block.

**Tests.** `tests/test_lss_utility.py` — utility arithmetic; label generator never reads a
reference for candidate *selection*; feature builder raises if asked for a test-split row;
calibration improves ECE on a synthetic miscalibrated fixture.

**Compute.** ≤3 GPU-h including feature extraction and the comparison arms.

---

### WP8 — Method baselines: LoRA and learned global steering  *(stage `l8_baselines`, W3–W5)*

**Purpose.** The two comparisons that decide whether the method claim survives review.

**Maps to.** v5 §6-Exp4 (fixes #1 and #9), §9.3-#2, Gate F.

**B1 — matched-data LoRA.** `lss/baselines/lora.py` with `peft` 0.18.1 on
`WhisperForConditionalGeneration`. **Matched data is defined and enforced in code:** training
utterances come from exactly `loc-train ∪ util-train` conversations (v5 fix #1); an assertion
rejects any `D-construct`, `D-dev-*` or test conversation. Targets: `q_proj,v_proj` of
encoder+decoder attention, r=16, α=32, dropout 0.05, bf16, gradient checkpointing, ~3 epochs.
Report for **every** trained system: audio hours, conversation count, trainable parameters,
wall-clock, peak memory.

**B2 — learned global rank-one steering (SALSA fallback, and the recommended primary form).**
`lss/baselines/learned_steer.py`: a single trainable vector `v ∈ R^{1280}` plus a scalar,
added at the frozen selected layer over **all** frames, trained with transcript
cross-entropy on the same matched data. ~1 281 trainable parameters. This isolates
*learned vs closed-form* and *global vs local* as separate axes and reuses the WP4 hook
infrastructure.

**Scheduling decision (deviates from v5 in ordering, not in content).** Build **B2 first**
(≈1 day) and treat a faithful SALSA reimplementation as optional. v5 §6-Exp4 already blesses
this substitution if SALSA is not running by end of W4; doing it in the other order removes
the risk instead of deferring it. The paper states plainly which variant was run.

**Gate `l8_baselines`.**

| Criterion | Threshold |
|---|---|
| LoRA trained only on `loc-train ∪ util-train` conversations | true (assertion) |
| LoRA converged (dev loss decreasing, MER on `D-dev-select` ≤ baseline) | true |
| learned global steering trained and evaluated | true |
| parameter/compute accounting reported for all trained systems | true |
| no test-split data touched | true |

**Outputs.** `models/lora_*`, `models/learned_steer_*.pt`,
`metrics/l8_training_cost.json`, `predictions/l8/*`, `reports/l8_baselines.md`.

**Tests.** `tests/test_lss_baselines.py` — the data-matching assertion fires on a polluted
manifest; the learned-steering module has exactly the advertised trainable parameter count
and the backbone stays frozen (`requires_grad` audit).

**Compute.** LoRA ≈ **6–10 GPU-h** (≈15–20 h of audio × 3 epochs); learned steering ≈1 GPU-h.

---

### WP9 — Complete system, frontier, and the W5 dry-run → **Gates E, F**  *(stage `l9_system`, W5)*

**Purpose.** Assemble the two-pass system, produce the headline evidence on
`D-dev-confirm`, choose primary vs fallback claim, then **freeze everything** and prove the
test script runs end to end.

**Maps to.** v5 §2.9, §6-Exp4/5/6, §7.4, §10.0, Gates E and F.

**New code.**

- `lss/steer/pipeline.py:run_two_pass(bundle, manifest, frozen, cfg)` implementing v5 §2.9
  exactly, including the abstention accounting v5 fix #13 requires:
  **zero-candidate abstentions are counted separately from below-threshold abstentions**,
  because the first is a localizer recall failure and the second is a selector decision.
- `lss/report/frontier.py:frontier(rows, taus, clusters, n_boot=10000)` — sweeps τ and, for
  **each** bootstrap resample of conversations, recomputes the **entire** curve, yielding
  pointwise bands (v5 fix #7). Bands are produced for every comparison system on the same
  axes.
- `lss/report/tables.py` — emits Tables 1–6 and Figure 2 as `.md` + `.csv` + `.pdf`.
- `lss_l9_system.py --dry-run --split D-dev-confirm` → the **identical** script that W6 runs
  with `--split D-test`.

**Systems in the main comparison (v5 §6-Exp4).** frozen Whisper · forced-ZH · forced-EN ·
bilingual-token · global fixed steering · predicted localization + steer-all · **predicted
localization + utility-aware selective steering (proposed)** · oracle localization + oracle
utility (upper bound) · matched-data LoRA · learned global steering.

**Required reporting.** PIER (embedded-EN), MER, WER, correction rate, corruption rate,
outside-span harm, Mandarin retention, monolingual ZH/EN WER, coverage, abstention (split by
cause), trained parameters, training cost, two-pass latency and second-pass usage rate —
**each stratified by deletion vs substitution** (v5 fix #6).

**Gate E (automatic method).**

| Criterion | Threshold |
|---|---|
| ΔPIER on `D-dev-confirm` under full automation | >0 |
| corrections vs corruptions | corrections > corruptions |
| fraction of oracle PIER gain retained | reported (no threshold) |

**Gate F (method comparison).**

| Criterion | Threshold |
|---|---|
| proposed system non-dominated vs LoRA and global steering on (correction, harm) or on retention at matched coverage | true |
| frontier band of proposed vs global steering | non-overlapping over a non-trivial τ range — **if it overlaps everywhere, the selectivity claim is not supported** (v5 §7.4) |

**Dry-run gate (`l9_system`, the W5 deliverable that protects W6).**

| Criterion | Threshold |
|---|---|
| every table and figure generated by script, no manual step | true |
| freeze file sealed; hash recorded | true |
| dry-run wall-clock | ≤ the W6 batch's Slurm allocation |
| script exits 0 with `--split D-dev-confirm` and produces all `EXPECTED_ARTIFACTS` | true |

**Claim selection.** Primary vs fallback (v5 §8.2) is chosen **here**, from `D-dev-confirm`,
and written into `freeze/claim.json` before the test batch.

**Outputs.** `metrics/l9_systems.parquet`, `metrics/l9_frontier.parquet`,
`figures/figure2_frontier.pdf`, `tables/table*.md`, `freeze/frozen_pipeline.json` (sealed),
`freeze/claim.json`, `reports/l9_system.md`.

**Tests.** `tests/test_lss_pipeline.py` — abstention accounting sums to the utterance count;
zero-candidate and below-τ abstentions are distinguishable; the frontier's chosen-τ point
equals the single-threshold evaluation; `l10` refuses to run against a tampered freeze file.

**Compute.** ~10 systems × 420 CS utterances (+ second passes) ≈ **2–3 GPU-h**; bootstrap
frontier is CPU-bound, ~20 min with 10 000 resamples over ~2 000 units.

---

### WP10 — Locked test batch  *(stage `l10_test`, W6)*

**Purpose.** Execute, do not develop.

**Procedure.** One Slurm submission: `sbatch cs_asr_lss.sh l10`. The stage verifies the
freeze file hash, unlocks the test split per §4.4, runs the identical W5 script with
`--split D-test`, and writes the final tables/figures. **No new code may be written during
W6.** If a bug appears, the correct response is to record it, fix it, re-run the **dry-run**
on `D-dev-confirm`, re-seal, and only then re-run the test — accepting the schedule cost.

**Gate `l10_test`.** Artifacts complete; freeze hash matched; no configuration key differs
from the dry-run except `split` and `prohibit_test_split`; a diff of the two invocation
records is written to `reports/l10_invocation_diff.md` and must contain exactly those keys.

**Compute.** ~10 systems × 1 046 CS utterances ≈ **4–6 GPU-h**; request 12 h.

---

## 6. Gate register (one page)

| Gate | Stage | Pass condition (short) | If it fails |
|---|---|---|---|
| **A** Alignment | `l1_align` | ≥2 valid aligners, ≥0.95 valid-unit coverage, ≥200-unit balanced human audit with median ≤100 ms / p90 ≤200 ms, EN−ZH ≤30 ms, synthetic ≥0.90@100 ms, bias ≤50 ms | Restrict to `high`+`medium` consensus subset and report coverage; if `high` bin also fails, stop and repair — never weaken |
| **B** Oracle headroom | `l4_oracle` | Net utility >0 with bootstrap CI excluding 0; corrections > corruptions; beats every null in paired bootstrap; dose-stable | Stop the method pipeline; write the fallback paper |
| **T** Transport | `l5_transport` | top-3 ≥0.60, ±1 ≥0.50, mass ratio ≥2.0, coverage ≥0.60, on `high` bin | E-only practical system; Site D → appendix |
| **C** Localizer recall | `l6_localizer` | ≥0.70 recall of oracle-correctable spans at ≤4 candidates/utterance | Report localization as the bottleneck; limit the automatic claim |
| **D** Utility value | `l7_utility` | Net utility beats steer-all and uncertainty-only at matched coverage, CI excludes 0 | Simplify to steer-all; drop the selector claim |
| **E** Automatic method | `l9_system` | ΔPIER >0 on `D-dev-confirm`, corrections > corruptions | Use the preregistered fallback claim |
| **F** Comparison | `l9_system` | Non-dominated vs LoRA/global steering; frontier bands separate somewhere | Reframe before test; do not run test on a weak claim |

---

## 7. Compute and storage budget

| Work package | GPU-hours (est.) | Notes |
|---|---:|---|
| WP0 pilot | 1 | measurement only |
| WP1 alignment sweep | 6–10 | DTW with attention dominates; MMS-FA and Qwen are cheap |
| WP2 baselines + cache | 3–4 | 4 prompts × ~12 k utterances at 6 utt/s |
| WP3 directions | 2–3 | encoder accumulation + 3 prompt conditions teacher-forced |
| WP4 oracle screen | 3–5 | selection + confirmation + 5 control families |
| WP5 transport | 1–2 | reuses cached attention summaries |
| WP6 localizer (+LID baseline) | 1–2 | training is minutes; LID inference dominates |
| WP7 utility labels + selector | 2–3 | ~3.5 k second passes ≈ 10 min; features dominate |
| WP8 LoRA + learned steering | 7–11 | LoRA is the largest single training item |
| WP9 system + dry-run | 2–3 | |
| WP10 locked test | 4–6 | |
| **Total** | **32–50** | ≈ 2–3 days of exclusive H100 time; fits the schedule with ≥2× slack |

**Correction to v5 §11.1:** utility-label generation is ~2–3 GPU-hours here, *smaller* than
LoRA training and smaller than the alignment sweep. The dominant schedule risk is Gate A,
not compute.

**Storage.** Encoder-state cache (1 layer, fp16, valid frames) ≈0.8 MB/utt → ~10 GB per
layer over 12 k utterances; cache ≤2 layers (~20 GB). Attention summaries ≈ KB/utt.
Predictions and metrics < 5 GB. Total < 50 GB against 11.6 TB free.

---

## 8. Schedule

### 8.1 Week map (v5 §10 weeks × this plan's work packages)

| Period | Work packages | Locked output |
|---|---|---|
| **W0: 10–16 Aug** | WP0 (all), WP1 steps 1–3, WP2 start, login-node model staging | Roles frozen, freeze file created, measured pilot, Qwen unblocked or declared unavailable |
| **W1: 17–23 Aug** | WP1 steps 4–7 (**Gate A**), WP2 finish, WP5 pilot, WP3 start | Gate A decision; transport rule or E-only fallback; caches complete |
| **W2: 24–30 Aug** | WP3 finish, WP4 selection phase, WP6 training | Directions locked; localizer frozen; Gate B preliminary |
| **W3: 31 Aug–6 Sep** | WP4 confirmation (**Gate B**), WP6 (**Gate C**), WP8 B2 then B1 start | Action \(a^*\) frozen — no further direction/site tuning |
| **W4: 7–13 Sep** | WP7 labels + selector, WP8 LoRA finish, Methods+Setup writing; **SEAME go/no-go 7 Sep** | Utility labels audited, selector trained |
| **W5: 14–20 Sep** | WP7 (**Gate D**), WP9 (**Gates E, F**), **full dry-run on `D-dev-confirm`** | Pipeline sealed; claim chosen; dry-run green |
| **W6: 21–27 Sep** | WP10 locked test; tables, CIs, Figure 2; **first complete draft** | Locked results + full draft before travel |
| **28 Sep–1 Oct** | Interspeech — protected, no dependency | — |
| **W7: 2–5 Oct** | Results, Analysis, Related Work (incl. Aditya et al. ICASSP 2024 positioning), limitations, appendix | Submission-ready content |
| **W8: 6–11 Oct** | Internal review, anonymization, formatting, reference audit, reproducibility package | Final anonymous package |
| **12 Oct** | Submit | — |

### 8.2 Critical path

`WP1 (Gate A) → WP3 → WP4 (Gate B) → WP7 (Gate D) → WP9 (Gates E/F) → WP10`.
WP2, WP5, WP6 and WP8 are parallelizable against it. **Gate A is the only item with no
slack**; v5 §4.3 explicitly permits engineering to continue while alignment is repaired, and
this plan uses that permission: WP0, WP2, WP6 scaffolding and WP8-B2 can all be built on the
`high`-confidence subset or on synthetic fixtures before Gate A closes.

### 8.3 Drop order if time slips (from v5 §10.1, made concrete)

Drop, in this order: (1) SEAME (WP-optional); (2) MLP selector capacity ablation (WP7);
(3) adjacent-layer and 4th-dose appendix runs (WP3/WP4); (4) SALSA faithful reimplementation
— keep B2 (WP8); (5) qualitative examples.

**Never drop:** Gate A validation · oracle headroom + label-permutation and matched-random
controls · prompt deconfounding for any Site-D claim · read-only localizer evaluation ·
utility vs steer-all/uncertainty comparison · LoRA and learned-global-steering references ·
locked non-oracle free decoding · correction/corruption and preservation · the selectivity
frontier · conversation-block CIs.

---

## 9. Risk register

| # | Risk | Likelihood | Impact | Mitigation / pre-decided response |
|---|---|---|---|---|
| R1 | Gate A fails again on English spans (EN consensus was 16/20 at smoke scale) | **High** | Blocks everything local | Three aligners; ±100 ms consensus; confidence bins; restrict science to `high`+`medium` and report coverage; long-span-only subset as the last resort (v5 §4.1) |
| R2 | Deletions (63% of errors) are unreachable by any local intervention | Medium-High | Shrinks oracle headroom | Stratify from WP4 onward; if E-steering only repairs substitutions, the honest claim narrows to substitution repair + localization of deletions — decide at Gate B, not at write-up |
| R3 | Gate T fails (attention cannot reach the deleted region) | **High** (by design of R2) | Removes contribution #3 | E-only system; Site D → appendix. Freeze file records the reason |
| R4 | Localizer recall <0.70 | Medium | Automatic claim limited | Fallback paper on the localization ceiling (v5 §8.3) |
| R5 | Selector adds nothing over steer-all | Medium | Removes contribution #2 | Simplify to local steer-all, reframe as "where" only |
| R6 | LoRA dominates every axis | Medium | Method claim weak | Lead with preservation/selectivity frontier; if it dominates accuracy *and* retention *and* parameters *and* latency, say so (v5 §8.3) |
| R7 | Multi-candidate batching bug (R1 refactor missed) | Low | 4× slower labels | Covered by `tests/test_lss_utility.py` batching test |
| R8 | Test-split contamination | Low | Fatal to the paper | `assert_no_test_data` everywhere + role assertions + the 4-condition unlock |
| R9 | Environment drift (torchaudio, libstdc++) | Low | Job failures | Preflight in the launcher asserts imports before any GPU work |
| R10 | W6 test batch hits a bug | Low (after dry-run) | No recovery week | The W5 dry-run is the mitigation; W6 changes only `split` |

---

## 10. Deviations from proposal v5 (and why)

| # | Deviation | Rationale |
|---|---|---|
| D-1 | **Eligibility redefined** to embedded-English units inside a CS utterance (§4.3) | The existing PIER counts 73% monolingual-English units on dev_select; unchanged, the primary endpoint would not measure code-switching |
| D-2 | **E-only is planned as the expected practical system**, Site D as a conditional branch | 63% of embedded-EN errors are deletions, which have no decoder step to steer |
| D-3 | **Learned global steering (B2) is built before any SALSA reimplementation** | v5 already permits the substitution; doing it first removes the schedule risk instead of deferring it to end of W4 |
| D-4 | **Consensus multi-aligner alignment replaces the E1 DTW-only path**; the NAT5H v2 coordinate system becomes production | E1's own audit failed; NAT5H v2 with canonical sample coordinates achieves 200 ms agreement on 39% of units with the same two aligners, implicating the E1 audit's coordinate mapping |
| D-5 | **Gate A adds a consensus-coverage criterion (A4) and confidence bins** | v5's "coverage ≥0.95" is ambiguous between *units having an alignment* and *units having a trustworthy alignment*; both are now measured separately |
| D-6 | **Transport uses first-pass attention with divergence cut-off** | v5 §2.7 leaves the circularity (second-pass attention needed before the second pass) unresolved |
| D-7 | **HistGBM (sklearn) replaces "gradient-boosted trees" generically** | No lightgbm/xgboost in the environment; avoids a new dependency in the final month |
| D-8 | **`max_k = 4` candidates preregistered**, `thr_label = thr_op − 0.10` | v5 asks for a preregistered cap and a looser labelling threshold but does not fix values; fixing them now prevents post-hoc tuning |
| D-9 | **`completed_no_go` added to the status vocabulary** | A failed scientific gate is not a crashed job; the E1–E5 status vocabulary conflates them, the NAT5H one does not |
| D-10 | **Compute framing corrected**: alignment, not utility labels, is the dominant risk | Measured throughput (§1.4) |
| D-11 | **Reference `arXiv:2408.05769` remains unverified and is excluded** until checked against the abstract page | v5 §References flags it; two citation errors already occurred in this lineage |

---

## 11. Testing, QA, reproducibility

### 11.1 Unit tests to add (all must pass under the launcher environment)

`tests/test_lss_roles.py`, `test_lss_freeze.py`, `test_lss_spans.py`, `test_lss_cache.py`,
`test_lss_directions.py`, `test_lss_oracle.py`, `test_lss_transport.py`,
`test_lss_localizer.py`, `test_lss_utility.py`, `test_lss_baselines.py`,
`test_lss_pipeline.py`. Existing 90 tests stay green — the WP7 R1 refactor is the one change
that touches shared code, and it keeps the `utterance_id` fallback for that reason.

Run with:

```bash
cd /home/tungnx/cs-asr-steer
LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib \
PYTHONPATH=src /home/tungnx/miniconda3/envs/acl1/bin/python -m pytest -q
```

### 11.2 Leakage assertions (mechanical, not procedural)

1. `assert_no_test_data` on every manifest load (already enforced repo-wide).
2. `lss/roles.py:assert_role(df, expected_role)` at the entry of every training function.
3. `l8_baselines` asserts the LoRA training conversation set equals `loc-train ∪ util-train`.
4. `l7_utility` asserts `util-train ∩ loc-train = ∅` at the conversation level.
5. `features.py` raises if any feature function receives a test-split row.

### 11.3 Reproducibility package

`git init` in W0 so `stage_provenance` records a real commit instead of
`no_git_repository`; per-stage `provenance.json`, `data_hashes.json`, `gpu_stats.json`
(already emitted by `experiments/_common.py:finish`); the sealed freeze file; and the
invocation diff from WP10. These are the artifacts the ARR appendix cites.

---

## 12. Immediate next actions

Ordered, each independently checkable. Items 1–4 are login-node work; 5–8 are Slurm.

1. **Create the scaffolding** — `src/csasr/lss/` package, `configs/lss/*.yaml`,
   `cs_asr_lss.sh`, stage names in `utils/status.py`. (WP0)
2. **Build and verify the data roles** — `python -m csasr.experiments.lss_l0_freeze
   --config configs/lss/base.yaml --roles-only`; check `manifests/role_report.json` for
   140/20/10/30 conversation counts and zero overlap. (WP0, §4.1)
3. **Stage the LID checkpoint on the login node** — `python -m
   csasr.experiments.ensure_lss_models --config configs/lss/base.yaml` (compute nodes are
   `HF_HUB_OFFLINE=1`). (WP6)
4. **`git init` + first commit** so provenance stops recording `no_git_repository`. (§11.3)
5. **`sbatch cs_asr_lss.sh l0`** — 200-utterance compute pilot; read
   `metrics/l0_pilot.json` and confirm the WP7 cost extrapolation. (WP0)
6. **`sbatch cs_asr_lss.sh l1 --aligner-smoke qwen`** — unblock the third aligner, or
   record it as unavailable and proceed 2-of-2. (WP1 step 1)
7. **Run the coordinate-hypothesis diagnostic** — recompute E1's CTC-vs-DTW comparison under
   `nat5h/coordinates.py`; this single number decides whether Gate A is a bug fix or a
   research problem. (WP1 step 2)
8. **`sbatch cs_asr_lss.sh l1`** — full alignment sweep + audit pack; then complete the
   ≥200-unit class-balanced human audit and re-run `l1` to evaluate Gate A. (WP1)

Parallel, not blocking: start `lss_l2_baseline.py` (WP2) as soon as WP0 lands — it does not
depend on Gate A — and draft the Method and Experimental Setup sections against §3–§5 of
this plan during W4–W5 as v5 §13.11 requires.

---

## Appendix A — New-code manifest (signatures)

```python
# lss/roles.py
def build_roles(cfg: dict) -> dict[str, pd.DataFrame]
def assert_roles_disjoint(roles: Mapping[str, pd.DataFrame]) -> dict
def assert_role(df: pd.DataFrame, expected: str) -> None

# lss/spans.py
def load_consensus(cfg) -> pd.DataFrame
def high_confidence_subset(spans, max_edge_ms: float = 100.0, min_aligners: int = 2) -> pd.DataFrame
def spans_to_frames(spans, geometry: EncoderGeometry) -> pd.DataFrame
def steering_mask(span_row, geometry, pad_ms=100.0, taper_ms=100.0) -> np.ndarray

# lss/cache.py
def cache_first_pass(bundle, manifest, cfg, out_dir: Path, layers: Sequence[int]) -> dict
def load_pass1(role: str, utterance_id: str, what: str) -> Any

# lss/features.py
def build_features(candidates: pd.DataFrame, cache, spans, directions, cfg) -> pd.DataFrame
FEATURE_NAMES: tuple[str, ...]          # frozen order, written into the freeze file

# lss/directions/pooling.py
def hann_weights(n: int) -> np.ndarray
def central_mass_weights(n: int, keep: float = 0.6) -> np.ndarray

# lss/directions/residualize.py
class NuisanceModel:  def fit(X, meta) -> "NuisanceModel";  def apply(X, meta) -> np.ndarray

# lss/directions/prompt_subspace.py
def prompt_subspace(states_en, states_zh, rank: int = 2) -> np.ndarray
def project_out(d: np.ndarray, U: np.ndarray) -> np.ndarray

# lss/directions/permute.py
def label_permuted_direction(pairs, seed: int, residualizer) -> np.ndarray

# lss/localizer/*.py
def frame_targets(spans, geometry, tolerance_ms: float = 40.0) -> np.ndarray
class LinearLocalizer / class TCNLocalizer:  fit(X, y, groups); predict(X) -> np.ndarray
def heuristic_scores(pass1, geometry) -> np.ndarray
def lid_scores(audio, model, window_s=0.5, hop_s=0.1) -> np.ndarray
def candidates(m, thr, min_dur_ms, merge_gap_ms, max_k) -> list[tuple[int, int, float]]
def evaluate_localizer(pred, gold, correctable, iou: float = 0.5) -> dict

# lss/utility/*.py
def generate_utility_labels(bundle, cfg, role: str, frozen) -> pd.DataFrame
class UtilitySelector:  fit(X, y, groups); calibrate(Xc, yc); predict_proba(X); choose_tau(...)
def evaluate_selector(probs, labels, clusters) -> dict

# lss/steer/*.py
class FrozenPipeline:  load/save/seal/verify
def aggregate_attention(summaries, heads, layers, sharpen=None) -> np.ndarray
def transport_gain(A_bar: np.ndarray, g: np.ndarray) -> np.ndarray
def effective_energy(rho: float, delta: np.ndarray, r: np.ndarray) -> float
class DecoderTransportSteeringHook(...)          # step-indexed, divergence-aware
def run_two_pass(bundle, manifest, frozen, cfg) -> pd.DataFrame

# lss/report/*.py
def eligible_units(manifest, units, level: str) -> pd.DataFrame
def system_metrics(rows, clusters, stratify: bool = True) -> dict
def frontier(rows, taus, clusters, n_boot: int = 10000, seed: int = 342) -> pd.DataFrame
def emit_tables(metrics, out_dir: Path) -> list[Path]
```

## Appendix B — Metric definitions as implemented

- **PIER** (`report/metrics.py`) = errors / eligible embedded-EN units, per §4.2; reported
  overall and split by deletion / substitution / other (§4.3).
- **CorrectionRate** = P(steered correct | baseline incorrect); **CorruptionRate** =
  P(steered incorrect | baseline correct) — existing
  `evaluation/correction_harm.py:summarize_correction_harm`.
- **Outside-span harm** = `outside_region_edit_rate` with `exclude_radius=1`
  (`evaluation/correction_harm.py:20`).
- **Utility** \(U_k = N_{corrected} - \eta N_{local\,harm} - \kappa N_{outside\,harm}\),
  η=κ=1 primary; 0.5 / 2 sensitivity only.
- **Retention** = ZH-CER and monolingual EN/ZH WER deltas vs the frozen baseline.
- **Coverage** = fraction of utterances receiving a second pass; **abstention** split into
  `zero_candidate` and `below_tau`.
- **Effective energy** \(E_{eff}=\rho^2\|\Delta\|^2\sum g^2\) computed on the *realized*
  delta after `norm_preserve`.
- **CIs**: block bootstrap over `conversation_id`, 10 000 resamples, seed 342
  (`evaluation/bootstrap.py`).

## Appendix C — Utility-label table schema (`metrics/l7_utility_labels.parquet`)

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | str | `<utterance_id>#<rank>` — the batching key for the WP7 R1 refactor |
| `utterance_id`, `conversation_id` | str | cluster keys |
| `role` | str | must be `util-train` (or `router-calib`) |
| `start_frame`, `end_frame` | int | candidate region |
| `localizer_score_max`, `localizer_score_mean` | float | |
| `candidate_class` | str | `error_en` / `correct_en` / `false_positive` / `outside_harm` |
| `baseline_correct`, `baseline_category` | bool, str | frozen before intervention |
| `steered_correct`, `steered_category` | bool, str | after the frozen action |
| `n_corrected`, `n_local_harm`, `n_outside_harm` | int | utility components |
| `utility` | float | η=κ=1 |
| `utility_eta05`, `utility_eta2` | float | sensitivity |
| `thr_label` | float | localizer threshold used for generation |
| `action_hash` | str | ties the label to the exact frozen action |
