# Executable Implementation Plan v2 — Selective Test-Time Intervention for CS-ASR

**Plan version:** v2, 10 August 2026 — supersedes `CS_ASR_ARR_October_2026_Implementation_Plan_v1.md`
**Inputs:** proposal v5; `IMPLEMENTATION_REVIEW.md` (12 blocking findings against v1)
**Status of this document:** Steps 1–3 are implemented, tested and runnable. Steps 4–11 are design.
**Code:** `src/csasr/lss/`, `src/csasr/experiments/lss_*.py`, `configs/lss/`, `cs_asr_lss.sh`
**Artifacts root:** `/mnt/data/tungnx/cs-asr-steer/artifacts_lss`

---

## 0. Why v2 exists

Plan v1 was reviewed and found not implementation-ready: twelve blocking findings, several of
which would have produced an apparently complete locked experiment measuring the wrong things.
Building Steps 1–3 then produced measurements that changed the plan again. This document is the
merged result. Everything below marked **measured** was computed from artifacts on disk, not
argued.

### 0.1 What the review changed

| Review | Change in v2 |
|---|---|
| B2 decoder site | The proposal's site (post-cross-attention residual) is **not** what `models/hooks.py` touches. Implemented as `lss/sites.py` and verified against a manual re-implementation of `WhisperDecoderLayer.forward` |
| B4 feature leakage | `boundary_confidence ← consensus disagreement` **deleted**; the guard rejects forbidden *sources*, never a data split |
| B5 tuned-set gate | `router-calib` grows 10 → 20 conversations, split `calib-prob` / `calib-thresh` |
| B6/B7 outcomes | Harm redefined as a **newly introduced error**, not a transcript change; candidates map to *sets* of units; six independent booleans replace one categorical |
| B8 steering scale | `scale = projection_std` and `E_pre = (ρ·scale)²‖Δ‖²Σg²` are in the freeze; decoder depth-rescale disabled so ρ=1 means the same at E and D |
| B9 matched data | Two LoRA arms: router-matched and full-method-data-matched |
| B11 stage graph | Declared once in `lss/prereq.py`, acyclicity asserted at import; the L2↔L4 cycle is gone |
| H2 label coverage | Unalignable English becomes `ignore`, never a negative |
| H3 selection bias | Included-vs-excluded distributions reported; both denominators published |
| H4 role assignment | Sorted slicing replaced by seeded rerandomization |
| Freeze design | One mutable freeze file replaced by per-stage immutable freezes |

### 0.2 What the measurements changed

1. **The "coordinate bug" hypothesis is refuted.** Under canonical coordinates the two aligners
   still disagree by 360 ms median on English starts. E1's headline +1354 ms was a tail-inflated
   *mean*; its own median was 370 ms.
2. **The disagreement is a constant offset, not a scale error.** OLS slope swings 102 → 19 ms/s by
   time window while r = 0.14 and the intercept holds at **−499 ms**. Theil–Sen slope 4.1 ms/s.
   The −499 ms intercept matches E1's audio-seam-relative offset of **−490 ms**: Whisper-DTW is
   systematically early relative to that seam by about half a second. This is not lexical error.
3. **The suspect is one line.** `data/alignment.py:200` used the query that *predicts* token k;
   OpenAI's reference slices at the query *at* token k. One decoder step is 220–400 ms here.
   `align_batch` now takes `pred_start_offset` (default unchanged) and `lss/align/bias.py`
   sweeps it against the development audio seam as a coordinate diagnostic.
4. **All 93 DTW overlaps are same-language (ZH→ZH), median 320 ms, none benign.** The existing
   resolver only handles cross-language pairs, so it repairs **zero** of them. Whisper-DTW is
   100% valid on English and 80% on Mandarin — the invalid rate is entirely a Han-character
   problem, and English spans are mechanically fine in both families.
5. **`nat5h.yaml`'s erosion would reject every span.** 250 ms erosion + 200 ms minimum interior
   needs ≥700 ms; measured consensus medians are 225 ms (EN) / 140 ms (ZH), so 0.000 qualify.
   v2 erodes a *fraction* of each span, capped by measured human error.
6. **Tightening the consensus tolerance destroys English coverage.** Measured sweep:

   | tolerance | accepted | EN accepted | EN retention |
   |---:|---:|---:|---:|
   | 50 ms | 2 | 0 | 0.000 |
   | 100 ms | 46 | 1 | 0.010 |
   | 200 ms | 222 | 16 | 0.167 |
   | 300 ms | 295 | 30 | 0.313 |

   The tolerance is therefore *selected* by a preregistered accuracy rule, never fixed a priori.
7. **`region` cannot be balanced.** 28 levels over 140 conversations: the best of 4000 draws for
   a 20-conversation role is TV 0.214, so a 0.15 threshold is unattainable by pigeonhole. It is
   optimized and reported; only the two-level covariates are gated.
8. **Jitter is already failing.** At ±100 ms the median mask IoU is 0.51 (threshold 0.60) and
   safe-interior survival is 0.74 (threshold 0.90) — see §4.

The blunt fact behind all of it: the median embedded-English span is **225 ms** while the two
aligners disagree about its start by **360 ms**. For a typical embedded-English word the
aligners disagree by more than the word is long.

---

## 1. Stage graph

Declared in `src/csasr/lss/prereq.py`; `assert_acyclic()` runs at import and is unit-tested.

| Stage | Prereqs | Gate | Status |
|---|---|---|---|
| `l0_freeze` | — | mechanical | **implemented** |
| `l1a_diag` | l0 | completeness (no science claim) | **implemented** |
| `l1b_valid` | l0, l1a | **Gate A** | **implemented** |
| `l1c_labels` | l1b | coverage | design |
| `l2a_prompts` | l0 | mechanical | design |
| `l2b_cache` | l0, l2a | completeness | design |
| `l3_directions` | l1c, l2b | stability | design |
| `l4_oracle` | l3 | **Gate B** + MDE | design |
| `l5_transport` | l2b, l3 | **Gate T** | design |
| `l6_localizer` | l1c, l2b | **Gate C** | design |
| `l7a/b/c` | l4, l6, l2b | **Gate D** | design |
| `l8_baselines` | l0, l4 | fairness | design |
| `l9a/b` | l7c, l8 | **Gates E, F** | design |
| `l10_test` | l9b | lock | design |

Enforcement goes beyond `require_passed`: artifacts are hash-checked, and `--force-prereq`
is contagious (a forced run marks its own output, and downstream production runs refuse it).

---

## 2. Step 1 — `l0_freeze` (implemented)

**Roles**, measured on the real corpus:

| role | conversations | utterances | hours | CS utts | embedded-EN units |
|---|---:|---:|---:|---:|---:|
| D-construct | 60 | 10 727 | 29.4 | 2 126 | 14 718 |
| loc-train | 35 | 6 968 | 17.1 | 1 220 | 8 334 |
| util-train | 25 | 4 887 | 12.8 | 912 | 6 107 |
| router-calib | 20 | 3 657 | 9.4 | 677 | 4 858 |
| D-dev-select | 20 | 3 360 | 11.5 | 708 | 6 215 |
| D-dev-confirm | 10 | 2 826 | 6.8 | 420 | 2 149 |
| D-test (locked) | 30 | 6 257 | 16.6 | 1 046 | 7 758 |

Balance achieved: **max |SMD| 0.120**, gated categorical TV 0.09, zero failures, objective 0.139
against a random-draw median of 0.385.

**Spec freeze** (`freeze/spec_freeze_v1.json`, sealed, self-hashing, refuses overwrite) records
eligibility, error grouping, both intervention tensors, the steering-scale contract, the feature
allowlist with per-feature source contracts, the outcome definitions, the seed map, and the
bootstrap convention. It is **executable**: the outcome fixtures and the decoder-site
reconstruction run inside the stage as gate criteria.

**Eligibility**, the decision that moves the headline number: an eligible unit is an English
content unit in an utterance that also contains Mandarin. dev_select drops 23 726 → **6 215**;
73% of P0's "English POIs" were monolingual-English utterances.

---

## 3. Step 2 — `l1a_diag` (implemented, passing)

Diagnostics only; its gate checks that every diagnostic ran and reproduces, and reports
coverage without thresholding it. It is structurally barred from improving agreement: the only
correction it computes is an explicit counterfactual marked not-applied, because an offset fitted
to inter-aligner agreement optimizes the statistic it would then be judged by.

Outputs: coordinate ladder (3 rungs), robust disagreement regression, signed-bias summary,
overlap inventory and repair-effect measurement, CTC `mapping_failed` cause buckets, uroman
expansion probe, per-family × language validity, Qwen probe (never raises), and a reproduction
check against the previous run's recorded per-family counts.

**Verdict semantics were split** after the measurement: `supported` / `partially_supported` /
`refuted` / `inconclusive`, because a *statistic* change (midpoint → unit edge, 380 → 200 ms) is
not a repaired aligner. Current verdict: `partially_supported` with a 200 ms residual.

---

## 4. Step 3 — `l1b_valid` (implemented) — Gate A

**Gate A concludes automatically. No new human annotation is on the primary path.**

```
l0 → l1a (automatic aligners) → l1b --prepare-audit → l1b --evaluate-gate
```

Manual boundary annotation exists only as an explicitly requested second opinion:

```
l1b --prepare-manual-audit → humans annotate → l1b --evaluate-gate --mode manual
```

Automatic mode never opens a verdict file (`--only verdicts` is dropped, not honoured);
manual mode never invents one.

### What each evidence source is allowed to claim

| source | supports |
|---|---|
| two **independent** valid aligners | that a second opinion exists. Counted by estimator class, so two `pred_start` variants of DTW are one |
| cross-aligner disagreement, natural speech | how far two estimators differ. **Not boundary error** |
| RMS/VAD synthetic splices | signed offset relative to a known audio seam; **not** lexical absolute error |
| exact lexical fixtures (when available) | true absolute boundary error, only when construction genuinely supplies lexical edges |
| manual annotation (optional mode) | true absolute boundary error — a person supplied the boundary |
| jitter ±50 **and** ±100 ms | whether conclusions survive being wrong by that much (proposal §4.1; only ±100 was gated before) |
| frozen high-confidence subset | whether there is enough material |

Two aligners that share a bias agree perfectly and are both wrong — DTW's signed offset relative
to the synthetic audio seam was −490 ms while it agreed *with itself* to 10 ms. So agreement is never
substituted for accuracy, and the naming rule is enforced in code:
`autoevidence.assert_no_absolute_error_claims` fails the run if a natural-speech criterion is
ever named as an error.

### Grouping and status

The failing group still selects the pre-decided response; the *status* now answers only "did
the experiment run correctly". Full contract in `LSS_STATE_AND_EXIT_CONTRACT.md`.

| group | contents | failure ⇒ |
|---|---|---|
| **M** mechanical | schema, partition exact, reports written | `failed` — a defect |
| **X** external | cross-aligner disagreement ≤100/200 ms and EN−ZH ≤30 ms; nonmonotonic ≤0.01; synthetic ≥0.90@100 ms, \|bias\| ≤50 ms; manual mode adds ≥200 units, ≥0.90 usable, median ≤100 ms, p90 ≤200 ms, α ≥0.67, decoys ≥0.80 | `completed_no_go` — never weakened |
| **J** jitter | mask IoU ≥0.60, contamination ≤0.05, safe-interior survival ≥0.90, at **both** ±50 and ±100 ms | `completed_no_go` |
| **C** coverage | unit coverage ≥0.95, plus absolute counts, not rates | `completed_no_go` |
| **R** reporting | selection bias, label coverage, both denominators | `failed` |

Precedence: **`failed` > `blocked` > `completed_no_go` > `passed`**. Missing evidence outranks
a no-go, because a gate that never saw its evidence has not run the experiment — the thresholds
that did fail stay visible in `gate["groups"]`. No terminal status but `passed` unlocks `l1c`,
and the span freeze `l1c` consumes is written **only** on a passed, untainted gate.

### Two things Gate A cannot currently establish

**Fewer than two independent valid aligners.** On the recorded candidate table `existing_ctc`
reaches 0.979 valid-unit coverage and `whisper_dtw` reaches 0.835, below the 0.95 floor. One
aligner cannot corroborate itself, so automatic Gate A returns
`blocked_insufficient_independent_aligners`. This is the same condition the earlier smoke run
hit, and it remains a no-pass.

**The current synthetic splice is not lexical truth.** The aligner execution,
paired-item scoring, manifests, and exact-reference metric path are implemented.
However, the current renderer trims source clips with RMS/VAD and knows only the
audio concatenation seam. That seam cannot support lexical absolute-error claims
or select a lexical operating tolerance. It now emits seam-relative diagnostics
and automatic Gate A returns `blocked_missing_genuine_lexical_calibration` until
manual, existing-gold, or genuinely exact constructed lexical edges are
available. See `ALIGNMENT_GATE_A_PROTOCOL_2026-08-12.md`.

**Measured jitter (smoke scale, 20 utterances):** median mask IoU **0.507** and safe-interior
survival **0.738** at ±100 ms — the proposal §4.1 check plan v1 omitted. At 140–225 ms median
spans, ±100 ms of boundary error removes half the mask.

The optional manual pack is blinded by construction: `items.jsonl` carries no language,
confidence bin, family count or aligner identity, times are clip-relative, order is shuffled,
10% of items are ±250 ms decoys and 10% are duplicates, and corrected times are mandatory for
≥120 items — E1's template left them optional, which is why its median/p90 criteria were never
evaluable.

---

## 5. Where this leaves the project

Gate A is genuinely open and the evidence is not encouraging. Three things are now measurable
that were not before, and they point the same way:

- English consensus retention is 0.167 at 200 ms and collapses under tightening;
- the mask does not survive the boundary error the corpus actually has;
- the residual DTW–CTC disagreement (200 ms) is comparable to the span length (225 ms).

The pre-decided ladder in `configs/lss/audit.yaml` applies in order: repair DTW (test the
`pred_start` hypothesis first — it is the cheapest and the evidence points at it), then the
long-span subset, then the calibrated arm, then an alternate estimator, then the fallback
manuscript. **The accuracy thresholds are not part of that ladder.**

If the `pred_start` sweep confirms the one-step hypothesis, adopting it in production is a config
change plus a one-line pass-through in `nat5h/aligners.py:run_whisper_dtw` — a decision for the
validation stage, deliberately not taken by a diagnostic.

---

## 6. Running it

```bash
# tests (295 pass, including the 117 pre-existing)
LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib \
PYTHONPATH=src /home/tungnx/miniconda3/envs/acl1/bin/python -m pytest -q

# future whole automatic path; DO NOT RUN while lexical calibration is blocked
# sbatch cs_asr_lss.sh chain --overwrite

# safe CPU-only cached development re-evaluation
PYTHONPATH=src python -m csasr.experiments.lss_alignment_dev_diagnostic \
  --config lss/l1b_valid.yaml \
  --diagnostic-output /tmp/csasr_gate_a_dev_diagnostic

# or one stage at a time
python -m csasr.experiments.lss_l0_freeze --config configs/lss/l0_freeze.yaml --roles-only
sbatch cs_asr_lss.sh l0                     # full: pilot + seal, needs a GPU
sbatch cs_asr_lss.sh l1a --only qwen        # isolated, deadlined
sbatch cs_asr_lss.sh l1a
sbatch cs_asr_lss.sh l1b                    # prepare + evaluate, automatic

sbatch cs_asr_lss.sh status                 # what is runnable, and why not
python -m csasr.experiments.lss_status --stage-status l1b_valid   # one word

# optional: a human boundary audit, only when explicitly wanted
sbatch cs_asr_lss.sh l1b --prepare-manual-audit
#   humans fill artifacts_lss/audit/l1b/verdicts_raw/*.csv per ANNOTATION_GUIDE.md
sbatch cs_asr_lss.sh l1b --evaluate-gate --mode manual
```

Exit code 0 means the command worked, not that Gate A passed; read
`status/l1b_valid.json` (`.status`, `.blocked_reasons`, `.next_action`) for the outcome.

Read `metrics/l1a_pred_start_sweep.parquet` before anything else in Step 2: it decides whether
Whisper-DTW is repairable — and repairing it is what would give Gate A its second independent
aligner.

---

## 7. Steps 4–11 (design unchanged from v1 §5, with the review's corrections)

Carried forward, in dependency order: `l1c_labels` (production consensus labels + ignore masks) ·
`l2a_prompts` (four prompt baselines) · `l2b_cache` (pass-1 cache **including attention rows** —
review B1; the `encoder_attn` hook already implemented for the decoder site yields cross-attention
weights at KB/utterance instead of the ~2.9 GB/utterance `output_attentions=True` path) ·
`l3_directions` · `l4_oracle` (staged E → D → one preregistered ED, then controls on the frozen
action only) · `l5_transport` (first-pass attention with a divergence cut-off) · `l6_localizer` ·
`l7a/b/c` (candidate-level `example_id` threaded through decode, cache, resume, OOM fallback and
outputs — review B3) · `l8_baselines` (two matching arms) · `l9a/b` · `l10_test`.

Ablations the review added and v2 keeps: outcome-supervision control, factorized mask/utility
upper bounds, language-margin ablation, matched-policy local/global control, automatic E/D/ED if
Gate T passes, and an honestly-labelled sliding-window LID baseline (MMS-LID is an
utterance-level classifier under CC-BY-NC, not a frame-level diarizer).
