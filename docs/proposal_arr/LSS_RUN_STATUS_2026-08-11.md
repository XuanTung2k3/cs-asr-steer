# LSS Steps 1–3 — implementation and run status

**Date:** 2026-08-11 · **Branch state:** `6adbb49` + uncommitted changes listed in §9
**Purpose of this document:** a complete, verifiable account of what has been built, what
has been measured, what is broken, and what is *known to be missing*, written so that an
independent reviewer (human or AI) can find problems in it. §10 is an explicit review
checklist with the questions I think are most likely to expose a real defect.

Every number in this document was measured on this machine on this date unless it is
marked **[plan v2]**, which means it was measured in an earlier session and recorded in
`CS_ASR_ARR_October_2026_Implementation_Plan_v2.md` §0.2 and has *not* been re-verified
here.

**Revision, later on 2026-08-11:** job 38502 finished. L0 passed; **L1a failed**. §2, §4,
§5.9–§5.14 and §7 are rewritten around that result, and one environment claim this
document previously made — that the Qwen aligner cannot load — is **refuted** by it (§4.1).

**Revision 3, later on 2026-08-11 — read §0 first.** Job **38573** ran the whole chain to
completion: L0 passed, L1a passed, L1b reached Gate A and terminated truthfully as
`blocked` after 01:13:46, exit 0. It is the first run that produced real Gate-A
measurements, and they are bad. §0 separates what that job **verified** from what has been
**changed since and not yet run**, because those are now two different things and mixing
them is the fastest way to mislead a reader.

---

## 0. What is measured, and what is only implemented

**Measured — job 38573 (COMPLETED, exit 0:0, 01:13:46).** Everything in §6.1 is a number
this job produced. The headline: Gate A is nowhere near passing, and *not* for want of
evidence — the synthetic gate set was scored and the aligners are 670–720 ms out on it.

**Implemented after 38573 and never run — §5.15 to §5.22.** Eight defects, one of which
(the dropped `role` column, §5.15) made four preregistered coverage criteria fail on
bookkeeping rather than on data. **No number in §5.15–§5.22 comes from a GPU run.** The
figures quoted there are either measurements taken from 38573's own artifacts by read-only
analysis on CPU (marked *[from 38573 artifacts]*) or they are absent.

**Unavailable evidence.** No human boundary audit exists, and none is required any more
(§5.16). No Qwen production alignment exists yet — the family has only ever run on the
8-utterance probe. And **the synthetic gate set 38573 used is burned** (§5.18): its
per-family medians, p90s and bias were read during this review and the selection rule was
changed in response, so those 100 items can no longer confirm anything. That is recorded in
`synthetic/exposure_ledger.json`, not just in prose.

---

## 1. Scope

| Stage | State |
|---|---|
| `l0_freeze` | implemented, **passed** (jobs 38502, 38573) |
| `l1a_diag` | implemented, **passed** (job 38573); now also selects the aligner configuration (§5.17), not yet run |
| `l1b_valid` (Gate A) | implemented, ran to a truthful `blocked` (job 38573). Eight fixes since (§5.15–§5.22), not yet run |
| `l1c`–`l10` | design only, not implemented |

Standing constraints in force:

- The user launches every GPU/Slurm job. No job is submitted by the assistant.
- A scientific gate is never weakened to let the pipeline continue. Where a threshold was
  changed (§5.1), the claim is that the *old threshold was unsatisfiable in principle*,
  and the replacement is strictly stronger. That claim is the single most important thing
  to review.

---

## 2. Run history

| Job | Elapsed | Exit | Stopped at | Cause |
|---|---|---|---|---|
| 38110 | 04:25 | 2 | L0 | `decoder_site_reconstruction_rel_err` 1.674e-3 > 1e-3 (§5.1) |
| 38392 | 04:52 | 2 | L0 | `projected_l1_alignment_gpu_hours` 12.203 > 12.0 (§5.4–5.6) |
| 38445 | 23:22 | 2 | L0 | same gate, now ~80 GPU-h — real, caused by §5.7 |
| 38502 | 04:47 | 2 | **L1a** | `TypeError: Qwen3ForcedAlignerAdapter.run() missing 1 required positional argument: 'language'` (§5.9) |
| 38573 | 01:13:46 | **0** | **L1b, truthfully** | Gate A `blocked`; nothing crashed (§6.1) |

(`sacct`: 38502 elapsed 00:04:47 FAILED 2:0; 38573 elapsed 01:13:46 **COMPLETED 0:0**,
11:57:04 → 13:10:50.)

**What job 38502 established.** L0 passed — `L0 gate: PASSED (full_l0_pass=True)` — so
§5.1–§5.8 are confirmed on the real model and the budget guard is satisfied. The chain
then ran L1a, which failed after 10.6 s of a 1800 s probe deadline and correctly refused
to start L1b (`L1a is failed, not passed; l1b will not run`). No stage has ever run on
unvalidated prerequisites.

L0's measured pilot rates in that job, which every projection below uses:

| pilot | rate |
|---|---|
| free decode bs16 | 5.81 utt/s |
| encoder only | 61.07 utt/s |
| teacher-forced | 42.84 utt/s |
| decoder-site capture | 46.34 utt/s |
| steered decode | 5.80 utt/s |
| `align_existing_ctc` | 3.71 utt/s |
| `align_whisper_dtw` | 9.79 utt/s |

The CTC rate is the §5.7/§5.8 fixes landing: 58.7 s per utterance became 0.27 s.

---

## 3. What is verified working

- **Role construction.** 140 train conversations → D-construct 60 / loc-train 35 /
  util-train 25 / router-calib 20, plus dev-select 20, dev-confirm 10, test 30.
  Seeded rerandomisation: `max_abs_smd = 0.120` (limit 0.25), `max_categorical_tv = 0.09`
  (limit 0.15), beats the random-null median. Conversation- and utterance-disjoint,
  0 test rows outside `D-test`.
- **Decoder site.** The post-cross-attention residual is captured and steered correctly.
  Proven on the real model in float32: `rel_err = 1.8e-8` (bound 1e-5). The site is
  demonstrably *not* the decoder block output (`site_vs_block_max_abs_diff = 0.4375`).
- **Spec freeze.** Seals, self-verifies, and re-compares clean across runs
  (`still describes this configuration`) — confirmed again in 38502.
- **The whole L0 gate.** All 24 gated criteria passed in 38502 (27 rows total, 3 of
  them report-only).
- **Test suite:** 380 tests collected, all passing, exit 0, run after every change in §5.

---

## 4. Environment facts that constrain the work

| Fact | Consequence |
|---|---|
| **`qwen_asr` 0.0.6 is installed and the aligner loads** (§4.1) | Qwen is a *real* third aligner, not a blocked one |
| Qwen weights on disk (1.8 GB, `models/Qwen3-ForcedAligner-0.6B`, `architectures: ["Qwen3ASRForConditionalGeneration"]`) | nothing to download |
| `torchaudio 2.5.1` vs `torch 2.10.0` → `libtorch.so: cannot open shared object file` | `torchaudio.functional.forced_align` unusable; hand-written Viterbi required |
| Compute nodes **have** network (HTTP 200 to huggingface.co) | `stage_models` can run inside a job; no login-node step needed |
| `HF_HUB_OFFLINE=1` everywhere except the `stage_models` phase | by design |
| Wall clock limit 20:00:00 | not a constraint at current runtimes |

### 4.1 The Qwen "cannot load" claim was wrong — **this document asserted it; job 38502 refuted it**

The previous revision stated: *"`transformers 4.57.6` lacks
`Qwen3ASRForConditionalGeneration` → Qwen aligner cannot load; probe returns `blocked`;
consensus is 2-of-2."* Both halves are false, and the run proves it.

**The proof.** 38502 failed with `Qwen3ForcedAlignerAdapter.run() missing 1 required
positional argument`. A `TypeError` for a missing argument is raised *at the call site*,
which is reached only after `adapter.load()` has returned. The 0.6 B checkpoint therefore
loaded on the H100, in 10.6 s, inside the probe subprocess.

**The mechanism.** `Qwen3ForcedAlignerAdapter.load` imports `qwen_asr`, and
`qwen_asr.core.transformers_backend` *vendors* its own `Qwen3ASRForConditionalGeneration`.
Verified in the `acl1` environment:

```
qwen-asr 0.0.6
dir(qwen_asr.core.transformers_backend) ->
    ['Qwen3ASRConfig', 'Qwen3ASRForConditionalGeneration', 'Qwen3ASRProcessor']
transformers 4.57.6, hasattr(transformers, 'Qwen3ASRForConditionalGeneration') -> False
```

The adapter never asks `transformers` for that class, so whether `transformers` implements
it was never relevant. The capability check that produced the false negative is §5.10.

**What follows from it.** The consensus is not condemned to 2-of-2 for software reasons,
and `blocked_insufficient_independent_aligners` is not the foregone conclusion §7.2 made
it. It also means the budget in §5.4 was shrunk on a false premise (§5.10).

### 4.2 The Qwen `language` argument is inert for this corpus — measured, not assumed

`Qwen3ForcedAlignerAdapter.run` requires a `language` and labels its output
`qwen_forced_aligner/{language}`, so the probe has to choose one or run both. In
qwen_asr 0.0.6 the argument only selects a *tokenizer*:
`Qwen3ForceAlignProcessor.encode_timestamp` lowercases it, branches to a Japanese or a
Korean tokenizer, and sends everything else — Chinese and English alike — through
`tokenize_space_lang`. `align()` uses it nowhere else. Measured here:

```
encode_timestamp("我们 用 machine learning 来 做", "Chinese")
encode_timestamp("我们 用 machine learning 来 做", "English")
  -> ['我', '们', '用', 'machine', 'learning', '来', '做']   # identical
```

So a second variant is a bit-identical duplicate at twice the GPU cost, and
`nat5h.consensus._family_representatives` would discard it anyway (one vote per *family*,
`ALIGNER_VARIANT_ORDER` ranking Chinese first). **Decision: one variant,
`qwen_forced_aligner/Chinese`** — pinned by a test against `ALIGNER_VARIANT_ORDER`, with
`package_version` recorded in every probe result so the claim is tied to 0.0.6.

Incidentally, the same function splits Han text per character and keeps English words
whole, which is the granularity the reference units already use.

**Deliberately not changed:** upgrading `transformers` would alter the Whisper decoder
internals that `sites.py` reconstructs and would invalidate the sealed freeze and
`environment.lock.txt`. Upgrading `torchaudio` is lower risk but was not needed once the
Viterbi was fixed (§5.8).

---

## 5. Defects found and fixed

§5.1–§5.8 are unchanged from the previous revision and were all confirmed by 38502's L0
pass. §5.9 onward are new.

### 5.1 The decoder-site tolerance was unsatisfiable in bfloat16 — **review this first**

**Symptom:** `rel_err = 1.674e-3` against a `1e-3` bound.

**Evidence.** Same weights, same wiring, three precisions:

| dtype | rel_err | eps | ULP |
|---|---|---|---|
| float32 | 8.0e-7 | 1.2e-7 | 6.7 |
| float16 | 2.9e-3 | 9.8e-4 | 2.9 |
| bfloat16 | 1.15e-2 | 7.8e-3 | 1.5 |

The error tracks the dtype epsilon. On the H100 the measured value is **0.21 ULP** of
bfloat16. The steering identity is exact in real arithmetic — the hook rewrites the
cross-attention output and the layer recomputes `residual + attn_out` in the model dtype —
so the residual is pure rounding.

**Fix.** One criterion became four:

| criterion | bound | measured |
|---|---|---|
| `decoder_site_reconstruction_fp32_rel_err` | ≤ 1e-5 | **1.8e-8** |
| `decoder_site_reconstruction_ulp` | ≤ 4 | **0.21** |
| `decoder_site_err_vs_block_gap` | ≤ 0.25 | **0.0063** |
| `decoder_site_differs_from_block_output` | == 1 | 1 |

`err_vs_block_gap` is the discriminating one: a misplaced hook leaves the site untouched
and scores ≥ 1, while 4 ULP of bf16 against the measured gap is ~0.12.

**Threshold-calibration error I made and corrected:** I first set that bound to 0.01,
which is *tighter* than the ULP bound on the real model and would have failed a correct
run. Caught by the float16 parametrisation of the new test.

**Tests:** `tests/test_lss_sites.py::test_reconstruction_error_is_rounding_not_wiring`
(3 dtypes), `::test_a_misplaced_hook_fails_the_block_gap_check`.

**What a reviewer should challenge:** is a dtype-relative bound a legitimate gate, or is
it a weakened one? My argument is that an absolute bound below the dtype's epsilon states
a requirement no correct implementation can meet, and that the float32 check (1e-5, met
with 500× margin) is the real correctness statement. Disagreement here is reasonable.

### 5.2 The spec freeze pinned measurements, so L0 could never pass twice

`extra={"pilot": {...}}` put wall-clock throughput and the site-check residual inside the
sealed payload, and `compare()` counted them as decisions. Any rerun of an unchanged
configuration would report drift. Fixed by adding `pilot` to `VOLATILE_KEYS`; the numbers
remain in the freeze as provenance.
**Test:** `test_lss_specfreeze.py::test_a_rerun_with_unchanged_decisions_reports_no_drift`,
which also asserts a real decision change is still caught.

### 5.3 `l0_freeze.yaml` did not include `align.yaml`

The CTC pilot failed with `alignment.ctc.model_id is not configured`, so the L1 budget was
projected from whisper_dtw alone. Added to the include chain. A new criterion
`pilot_aligner_families_measured` fails if a configured, timeable family produces no rate.

### 5.4 The projection budgeted a family that cannot run

`alignment_families` counted `[existing_ctc, whisper_dtw, qwen_forced_aligner]` = 3, but
Qwen was believed unable to load. Added `pilot.runnable_families()`. **The capability test
it used was wrong (§5.10) and the belief was wrong (§4.1)**, so this fix was right in
mechanism and wrong in verdict.

### 5.5 The projection charged every family at the slowest family's rate

`hours(n_items * n_families, min_rate)` billed whisper_dtw's 4900 utterances at CTC's
rate — a 3× overstatement (12.20 h vs 4.11 h). Replaced with a per-family sum; the old
worst case is still reported as `l1_alignment_gpu_hours_worst_case`. Families configured
but not timed are charged at the slowest measured rate.

### 5.6 The aligner pilot timed the *shortest* utterances

`sample` is sorted by duration for batching, so `head(20)` selected the 20 shortest:
**mean 1.75 s against a role mean of 9.86 s**. Alignment cost scales with audio length, so
the measured rate was optimistic by several fold. Now the picks are evenly spaced across
the duration range, and a new criterion `pilot_aligner_sample_duration_ratio ≤ 0.25`
fails if the timed audio is unrepresentative.

**This fix is what exposed §5.7.** Under the old sampling the next defect was invisible.

### 5.7 `romanize()` rebuilt uroman's tables on every call — the 80 GPU-hour bug

With representative audio, CTC measured **1174 s for 20 utterances = 58.7 s each**,
projecting ~80 GPU-h. Root cause, `ctc_alignment.py`:

```python
return ur.Uroman().romanize_string(str(text))    # a new Uroman() per call
```

`align_units` romanizes **each reference unit separately** (deliberate — romanising the
whole utterance breaks the unit→token mapping). Measured:

| | cost |
|---|---|
| `Uroman()` construction | **3.08 s** |
| `romanize_string()` on an existing instance | **0.0011 s** |

2,800× overhead per call. A 19-unit utterance → 19 × 3.08 = 58.5 s, against 58.7 s
observed. Fixed with `@lru_cache(maxsize=1)`; output byte-identical.
**Test:** `test_ctc_alignment.py::test_the_romanizer_is_built_once_not_once_per_unit`.

### 5.8 The Viterbi was a pure-Python scalar loop — and my first fix made it slower

`viterbi_align` looped over frames × states in Python, indexing torch tensors one scalar
at a time: 4.38 s for a 550-frame, 301-state trellis.

**A torch-vectorised version measured 6× SLOWER** (27.4 s vs 1.23 s) — on a few-hundred-
element trellis, torch's per-call overhead exceeds the loop it replaces. Redone in numpy:

| trellis | scalar original | numpy | |
|---|---|---|---|
| 550 × 81 | 1.23 s | 0.102 s | 12× |
| 550 × 301 | 4.38 s | 0.211 s | 21× |
| 1400 × 601 | — | 0.427 s | |

**Correctness is guaranteed by an oracle, not by inspection:** the original scalar
implementation is kept verbatim in the test file and the vectorised version must produce
**identical spans** across 5 random seeds plus a repeated-token case.

**Net effect on CTC, confirmed in 38502:** 3.71 utt/s, i.e. 0.27 s per utterance.
Projected L1 alignment budget: **0.51 GPU-h** for two families against a 12 h limit.

### 5.9 The Qwen probe called `run()` without the argument it requires — **this is what killed 38502**

`probe_qwen` called `adapter.run(sample, geometry=geometry, identity=identity)`. The real
signature is `run(self, manifest, geometry, language, identity, diagnostics_dir=None)`.
The probe classified the resulting `TypeError` as a defect rather than as infrastructure —
which is exactly right, and is why L1a reported `failed` instead of quietly proceeding
2-of-2 — but the call was still wrong.

**Why no test caught it.** Every probe test substituted a fake adapter whose
`run(self, *a, **k)` accepts anything, so no test could observe the call. Worse, the
*production* path is `probe_qwen_subprocess`, and `probe_qwen_subprocess`, `_child_main`
and `publish_attempt` had **no test at all** — the tested path was the in-process one.

**Fix:** pass the language (§4.2 decides which); and three new tests —

- a stub adapter whose `run` signature is pinned to the real declaration by comparing
  parameter names, kinds and optionality (annotations excluded, since they cannot make a
  call fail);
- an in-process probe asserting the persisted table carries exactly
  `qwen_forced_aligner/Chinese`;
- **an end-to-end run through the real subprocess**, with the child pointed at the stub
  via `CSASR_QWEN_ADAPTER_CLASS`, asserting the attempt is published and cleaned up.

Verified by reintroducing the bug: two tests fail, including the subprocess one. Restored:
all pass.

### 5.10 L0 asked the wrong runtime whether Qwen can load

`_qwen_runnable` checked `hasattr(transformers, arch)`. The adapter loads through
`qwen_asr` (§4.1), so this was a false negative, and §5.4 removed a whole family from the
L1 budget on the strength of it.

**Fix:** resolve the checkpoint's declared architectures against
`QWEN_RUNTIME_MODULES = ("qwen_asr.core.transformers_backend", "transformers")` — the
adapter's own runtime first, `transformers` retained as a fallback for a future release
that implements the class natively. The reason string now names every runtime asked and
which one supplied the class. Measured cost: 7.5 s of import, one-time, in a ~5 min stage.

**Budget consequence**, from 38502's own rates (4900 utterances, untimed family charged at
the slowest measured rate):

| runnable families | `l1_alignment_gpu_hours` | worst case | limit |
|---|---|---|---|
| 2 (before) | 0.506 | 0.734 | 12.0 |
| **3 (after)** | **0.873** | 1.101 | 12.0 |

The correction does not put the budget gate at risk. **L0 must be re-run for this to take
effect** — see §8.

### 5.11 `--overwrite` was a no-op for exactly the two stages that advertised it

`cs_asr_lss.sh chain` forwarded `--overwrite` to every stage but never consulted it before
the "already passed; skipping" branch, whose message read *"pass `--overwrite` to redo
it"*. The flag could not redo it.

**Fix, and a rejected "improvement".** `--overwrite` now suppresses the skip. The skip
itself was tightened to require that the stage passed *under the same resolved config*, via
a new `lss_status --stage-reusable` that compares the recorded `resolved_config_hash`.
That hash is reproducible — verified against 38502's recorded
`3ad1df9d…`, reproduced exactly by re-loading the same config path.

The obvious next step, also requiring `source_snapshot_hash` to match, was **rejected on
evidence**: `source_snapshot_hash` covers `environment.lock.txt`, which
`cs_asr_lss.sh preflight` rewrites with a fresh timestamp and Slurm job id at the start of
every job. It therefore differs on every run even when no code changed, so requiring it
would mean *never* skipping — deleting the resumption that exists to protect the one
expensive stage. `--overwrite` is the lever for a code change. This is stated in
`stage_reusable.__doc__` and asserted by a test, so it does not get "fixed" later.

A second, smaller defect was introduced and removed during this fix: bare
`[[ … ]] && echo …` at statement level ends the job under `set -e`. There is now a test
that no such line exists in the launcher.

### 5.12 Coverage was measured against a universe 30× larger than the stage aligns

`_run_aligner_sweep` aligns `manifest[contains_code_switch].head(300)` per role.
`_expected_unit_universe` took **every unit of the whole role**. Measured on D-construct:

| | units |
|---|---|
| swept sample (300 CS utterances) | 10,167 |
| denominator coverage was measured against | 309,535 |
| **ceiling on `alignment_unit_coverage`** | **0.033** |

Against a 0.95 floor, that criterion could not pass however good the aligners were, and
Gate A would have reported `completed_no_go` on the coverage group for a bookkeeping
mismatch rather than a measurement.

**Fix:** one `sweep_sample()` defines both the sweep and the denominator, so they cannot
drift again; the universe is restricted to EN/ZH units, which are the only ones any family
emits candidates for. The anti-cheat property is preserved — the selection is
deterministic and independent of aligner output, so a family that silently drops
utterances still shows as missing coverage.

This also exposed that the end-to-end test fixture used a transcript segmenting into
**seven** units while mocking candidates for four, so "full coverage" in those tests was
4/7. The fixture now uses a transcript whose real segmentation is exactly the four units
it mocks.

### 5.13 An 8-utterance probe could qualify as a full independent aligner

`family_validity.valid_unit_coverage` is a rate over *the rows a family produced*, and
`independent_valid_families` was called without `min_units` (default 1). The Qwen probe
writes `probe_utterances` (8) of the swept 300 into
`candidates_qwen_forced_aligner.parquet`, which `run_families` then folds into the sweep.
Those 8 utterances score 1.0 and would have counted as a second independence class —
clearing `blocked_insufficient_independent_aligners` on ~3% of the units.

**Fix:** L1b now passes `min_units = ceil(min_family_valid_unit_coverage × |universe|)`,
so a family must cover the same universe coverage is measured against. Reported as
`min_valid_units_per_family`. Tested end to end with a probe-sized Qwen table alongside two
full families: Qwen is rejected with `valid_units=…<…` while its validity rate is still
1.0, and the two real families are unaffected.

This is a hole that only became reachable because §4.1/§5.10 made Qwen real.

### 5.14 Synthetic exact-boundary scoring — implemented

The largest previously-missing piece. `build_set` rendered splices with known boundaries
and `score_family` could score a prediction, but nothing ran the aligners over the rendered
audio, so no absolute error existed anywhere. Now:

1. `synthetic.aligner_manifest` builds a manifest from the rendered pairs
   (`transcript_raw` = ZH text + EN text, so the constructed instant is a seam *between*
   reference units and no unit straddles it);
2. `_score_synthetic_sets` runs `candidates.run_families` over it, per purpose, into a
   per-purpose directory so the gate set's candidates can never be read as the dev set's;
3. `synthetic.boundary_predictions` reads the seam off the units that touch it — the end
   of the last valid ZH unit against `zh_end_sec`, the start of the first valid EN unit
   against `en_start_sec` (with a gap these are different instants and each is scored
   against the one it actually predicts), plus the legacy `(end + start) / 2` against
   `true_boundary_sec`;
4. `score_rendered_set` writes per-family/edge scores and publishes
   `metrics/l1b_synthetic_scores.parquet` **with a manifest**, because
   `synthetic_calibration_status` authenticates before reading.

Two decisions a reviewer should check:

- **A family that produced no valid unit on one side of the seam is `scorable=False`, not
  a large error.** It made no prediction; charging it for one would read as an accuracy
  failure instead of missing coverage. Counts and reasons go to `synthetic_unscorable`.
- **Only the canonical (unit-edge) convention enters the score table.** Gate A takes the
  worst case over every row in it, and E1 measured the legacy midpoint convention ~490 ms
  out [plan v2], so including it would fail the bias criterion for a convention nothing
  downstream consumes — steering masks are built from unit edges. The legacy convention is
  still scored on the same items and written to
  `metrics/l1b_synthetic_convention_comparison.parquet`, which is the diagnostic the
  "coordinate bug" hypothesis needs. **If you think the gate should judge both
  conventions, this is the line to argue with.**

14 unit tests plus 4 end-to-end tests through the production assembly (GPU and source audio
substituted; manifest, seam mapping, scoring, publishing and Gate A's read are all real).

---

## 5b. Defects found after job 38573 — implemented, **not yet run**

### 5.15 `role` was dropped by consensus, so four coverage criteria failed on bookkeeping

`nat5h.consensus.build_unit_consensus_v2` copies a fixed list of columns from the winning
candidate and `role` is not on it. `candidates_all.parquet` *does* carry it (util-train
35,995 / loc-train 31,096 / D-dev-select 30,852 / router-calib 30,546 / D-dev-confirm 27,876
/ D-construct 20,307), and `consensus_spans_v1.parquet` does not. So `_role_span_counts`
returned zero for all six roles and `min_loc_train_en_spans`, `min_dev_select_targets`,
`min_dev_confirm_targets` and `construct_bilingual_utterances` all failed against zero while
32,086 spans sat on disk.

A read-only reconstruction of the same spans *[from 38573 artifacts]*: D-construct 1,124
spans / 60 EN / 245 utterances, loc-train 1,370 / 36 / 230, util-train 1,952 / 32 / 258,
router-calib 2,006 / 92 / 250, D-dev-select 1,322 / 85 / 252, D-dev-confirm 1,274 / 29 / 234.
**Those still miss the preregistered counts** (loc-train needs 1,000 EN spans and has 36).
The fix does not make them pass; it makes them fail on the data.

Fixed by joining the frozen utterance-to-role mapping back on in
`consensus_prod.build` — not by editing `csasr.nat5h`, whose recorded artifacts have to stay
reproducible. `utterance_roles` raises on an utterance claimed by two roles, `attach_roles`
raises on an unmapped span, `role` is required by `_span_schema_ok` (mechanical → `failed`,
not a silent zero), and `role_counts_available` reports the exact regression: spans exist and
every role counts zero.

### 5.16 Automatic mode needed a human to choose its tolerance — the contradiction, resolved

`spec.yaml alignment_prereg.tolerance_selection_rule` reads *"the largest swept tolerance
whose accepted spans still satisfy **human** median ≤ 100 ms and p90 ≤ 200 ms"*, and
`alignment.consensus.primary_tolerance_ms` is null. So the *primary, automatic* workflow
blocked and its `next_action` told the user to run a manual audit — for a workflow whose
stated purpose is to need none.

**The conflict is real and is documented rather than papered over.** The resolution: the
estimand in that rule is *the absolute boundary error of the spans a tolerance accepts*.
"Human" names the **instrument**, which was the only one implemented when the rule was
written; the same freeze already registers synthetic splices as external truth
(`calibration.fit_on: synthetic`, `corrections_require_external_truth: true`) and their
boundaries are known by construction rather than estimated. So the rule is applied unchanged
— same form, same numbers, 100 ms and 200 ms — against the synthetic **development** set, and
the instrument is recorded in the frozen selection:

    automatic  instrument = synthetic_dev   (development items only)
    manual     instrument = human_audit     (the preregistered validation, still available)

`spec.yaml` is **not edited**: it is sealed into `freeze/spec_freeze_v1.json`, and changing
`alignment_prereg` would make `spec_freeze_matches_live_config` fail and L0 report `failed`.
The automatic thresholds live in `configs/lss/audit.yaml` as
`max_selection_median_abs_error_ms: 100.0` / `max_selection_p90_abs_error_ms: 200.0` with the
conflict written next to them.

Mechanically (`csasr.lss.align.devselect`): for each swept tolerance, build consensus on the
synthetic **dev** candidates, read the seam off the accepted *consensus* spans — which is
what a steering mask is actually built from — and score it against the constructed boundary,
worse edge per item. `assert_development_only` raises if a gate row ever reaches this code.
The result is frozen in `freeze/l1b_operating_point.json` with the selected tolerance, the
rule, the instrument, the development artifact's sha256, the thresholds, the seeds, and L1a's
aligner selection by reference. Erosion and union padding come from the same measurement,
following the preregistered erosion rule with the same instrument swap.

**If nothing qualifies, the outcome is `blocked_no_qualifying_operating_tolerance`, never a
silent 200 ms.** A provisional value recorded as "selected" would make every downstream
number a statement about a configuration nobody chose. Given §6.1's numbers, **this is the
outcome to expect.** It is a distinct code from `blocked_unselected_operating_tolerance`,
which now means only "the selection never ran or its record is not authentic".

### 5.17 The `pred_start` experiment was unreachable — the ~700 ms bias could not be tested

`bias.pred_start_sweep` existed and `align.yaml` documented the hypothesis (`alignment.py`
takes the query that *predicts* token k where OpenAI's reference slices at the query *at*
token k; one decoder step is 220–400 ms here). But the sweep needs synthetic development
data, that data was built by L1b, and L1b requires L1a — so L1a logged
`pred_start sweep is a GPU experiment; run it with the synthetic development set produced by
l1b --only synthetic` and the experiment never ran. Job 38573 then measured Whisper-DTW at
**−840 ms / −830 ms** signed error on the dev set and **−720 ms / −660 ms** on the gate set,
on *both* edges: a uniform shift, exactly the signature the sweep was written to test.

L1a now builds the synthetic development set itself (prerequisite-safe: it needs only the
D-construct role manifest L0 froze), publishes it authenticated at `synthetic/dev_items.parquet`,
runs each configured offset through **`run_whisper_dtw` — the same function production
alignment uses** — scores each against the constructed seam under the canonical unit-edge
convention, writes `metrics/l1a_pred_start_sweep.parquet`, and freezes the winner in
`freeze/l1a_alignment_selection.json`. L1b loads that and passes it to `run_families`.

Three details that matter:

- the offset is **passed**, not injected into `cfg`: mutating `cfg` would change
  `resolved_config_hash` and every artifact identity derived from it;
- the variant label carries the convention (`whisper_dtw/zh_median7` stays the name for the
  historical −1, so nothing recorded earlier changes meaning; offset 0 becomes
  `whisper_dtw/zh_median7_pred0`, appended to `ALIGNER_VARIANT_ORDER` and never inserted);
- the two conventions get **different cache filenames**, because they resolve to the same
  `cfg` and `artifact_matches_identity` cannot tell their tables apart — the sweep would
  otherwise score one convention twice.

Selecting on development data is not judging: Gate A's thresholds still have to be cleared on
a gate set no selection has seen.

### 5.18 The old synthetic gate set is burned, and that is now recorded

Job 38573's gate scores were read during this review and the selection rule (§5.16) was
changed in response to them. Those 100 items can still be scored; they can no longer answer
*"does the configuration we just chose hold on data nobody had looked at"*. Reusing them
would be the oldest way to make a gate pass.

`csasr.lss.align.exposure` records exposure instead of remembering it.
`synthetic/exposure_ledger.json` holds one entry per generation with its source utterance
ids and a fingerprint; **evaluating Gate A exposes the generation it read**, so the next
evaluation draws generation *N+1* from sources no earlier generation used; sets rendered
before the ledger existed are bootstrapped from the item tables they left on disk and
recorded as exposed, with the reason naming job 38573. A fresh generation gets its own
storage label (`synthetic/gate_g2/`) so it cannot read the previous generation's cached
candidate tables, and the score table's manifest records
`synthetic_gate_generation` — a later evaluation refuses a table from an already-exposed
generation however authentic its bytes are. If no unexposed source audio is left the outcome
is `blocked_exposed_gate_set`: missing evidence, not a defect.

The **development** set is deliberately not filtered by exposure. It is meant to be reused —
it is what selects the configuration — and a dev set that changed every run would make the
selection unreproducible. `partition_sources` keeps the two pools disjoint, so no dev
recording can reach a gate generation anyway.

### 5.19 Qwen had no production path, so it could never be a third aligner

`candidates.run_families` consumed whatever the probe persisted: eight *corpus* utterances,
stamped with L1a's `config_hash`, which L1b's identity check refused — hence 0 Qwen rows in
38573's `candidates_all.parquet` and no way to score Qwen on synthetic pairs, whose ids
cannot exist in the corpus. §5.13 had correctly stopped the probe masquerading as a
qualifying family, which left the family contributing nothing at all.

`csasr.lss.align.qwen_prod` is the production path: the same adapter over an arbitrary
manifest, in **its own process group** (`start_new_session=True` + `killpg`, because a
checkpoint stuck in a native call ignores SIGALRM), in bounded chunks, with **its own
deadline** — `production_deadline_minutes: 60` plus `production_seconds_per_utterance: 4.0`,
whichever is larger, because 1,800 utterances is 225× the probe's eight and inheriting a
30-minute probe bound is how a run gets killed for being the size it was asked to be.

Nothing is published unless the run finished:

- chunks are written whole (temp file, then rename) inside an **attempt-private** directory
  the cache cannot see; a timeout kills the group and discards the entire attempt, so the
  next run finds a complete table or nothing — never a prefix that looks complete because it
  is a valid parquet;
- the published table must **cover the expected universe**; a run that aligned 1,400 of
  1,800 utterances is discarded, because a family scores a perfect validity *rate* on
  whatever it did produce. Per-utterance aligner failures are different: the adapter emits
  invalid rows for those, so they are covered and counted against the family, correctly;
- publication is atomic and manifested, with the identity of the run that consumes it;
- a timeout is `blocked`, never `completed_no_go` — a timeout measured nothing.

The 147-row probe stays as a capability check and is never production evidence. One frozen
language variant, for the reason in §4.2.

### 5.20 Natural and synthetic qualifying families did not have to be the same families

Gate A checked two things separately: that two independence classes qualified on natural
speech, and that two independence classes were scored on the synthetic gate set. **Nothing
required them to be the same two.** With Qwen now able to qualify (§5.19) that became a live
false-pass path: CTC+Qwen could supply the spans and the cross-aligner disagreement while
CTC+Whisper supplied the absolute error — every criterion satisfied, and the accuracy claim
about an aligner that produced none of the spans being claimed about.

`autoevidence.family_correspondence` computes the intersection of the two class sets, that
intersection must itself reach `min_valid_families`, a qualifying family with no synthetic
calibration blocks with `blocked_uncalibrated_natural_families`, and a score from a family
*rejected* on natural speech cannot stand in for it. The cross-aligner disagreement is now
measured between the **corresponding** families, so the pair Gate A names is the pair it
measured. Regression test: `test_ctc_plus_qwen_natural_and_ctc_plus_whisper_synthetic_is_not_a_pass`.

### 5.21 `_next_action` returned one blocker and recommended annotation for everything

It returned the *first* matching blocker, and for the commonest one it recommended a manual
audit — which repairs none of a 17% invalid rate, a failed accuracy threshold, a failed
jitter check or a coverage shortfall. On 38573 that produced a `next_action` telling the user
to annotate while eleven other things were wrong.

`_next_actions` now enumerates every blocker **and** every failing criterion, ordered by
dependency (provenance → alignment → configuration → accuracy → robustness → coverage), each
with the one action that resolves it. Annotation appears exactly once, last, marked
`optional`, saying explicitly that it repairs none of the items above it. The list goes into
the status file (`next_actions`) and into a table at the top of the Gate-A report;
`next_action` stays as the highest-priority one for the one-line field.

Also corrected here: `complete` in the status file meant `gate["passed"]`, so a truthful
terminal `blocked` was recorded as an *incomplete* run — the same shape as a job killed
mid-write, which is exactly what `complete` exists to distinguish. It now means "this run
finished the work it set out to do" and is false only for `failed`. Prerequisites are
unaffected: `prereq._status_problem` rejects any status that is not `passed` before it looks
at `complete`.

### 5.22 Diagnosing the two aligner failures — one repairable, one not

Both were investigated on 38573's artifacts, on CPU, read-only. Neither threshold was
touched and no invalid row was relabelled.

**CTC's 2,283 `mapping_failed` units are a romanization gap, and it is fixed.** *[from 38573
artifacts]* All 2,283 are Mandarin, and they are 14 distinct surfaces: 一 (2,010), 十 (58),
三 (45), 二 (44), 四 (37), 五 (25), 零 (15), 六 (10), 九 (10), 百 (8), 八 (8), 千 (6), 七 (6),
陌 (1) — every one a Han numeral. Measured directly: uroman renders `一` as the digit `1`,
`十` as `10`, `百` as `100`, `一百二十三` as `123`. MMS-FA's vocabulary is 31 romanized
letters with no digits, so `align_units` skipped every character of the romanization and the
unit came back with no span. L1a's own diagnostic had already labelled the cause
`empty_after_vocab`; nothing had acted on it.

`ctc_alignment.CJK_NUMERAL_PINYIN` now maps those characters (and the financial variants, and
`陌`, which uroman renders `100` and is simply wrong — it reads *mo*) to toneless pinyin
before uroman sees them. This gives the acoustic model the syllable the speaker said instead
of a digit it cannot represent. It invents no boundary and relabels nothing — CTC still has
to find the span. If it does, CTC's invalid **and** nonmonotonic rates both go to ≈0 (they
are identical in 38573 because the NaN rows fail both), which is what stands between
`existing_ctc` — the family that covers 97.4% of units — and qualifying.

**Whisper-DTW's 15,112 `adjacent_overlap` units are not an alignment error and must not be
"repaired".** *[from 38573 artifacts]* All 15,112 are Mandarin, and in **100.0%** of them the
two spans are byte-identical — same start, same end. The overlap equals the unit's own
duration to three decimal places (mean 391.747 ms on both). That is only possible if both
reference units were assigned the same token span, and that is exactly what happens:
Whisper's BPE emits one token for several Han characters (`他们`, `非常`, `好的` are all in the
first affected utterance), `align_batch` maps a unit to every token whose character range it
intersects, and both characters inherit the same `tok_start`/`tok_end`.

There is no sub-token resolution to recover. Truncating the earlier span would invent a
boundary inside one token from no acoustic evidence and buy exactly the coverage number the
0.95 threshold exists to measure. `repair.attribute_overlaps` records the mechanism
(`shared_source_token`, told apart from a one-frame `frame_grid_artifact` and a genuine
`partial_overlap`) and moves nothing. **The honest reading is that `whisper_dtw` cannot
supply unit-level boundaries for ~17% of Mandarin units**, which is what the invalid flag
already says, and it is why the second aligner has to come from Qwen (§5.19).

Considered and rejected: encoding the reference per-unit so no token spans two units. It
would give per-character queries, but it changes the teacher-forced token sequence, and its
interaction with `max_text_tokens: 220` truncation could silently reduce natural coverage
while looking better on short synthetic items. If anyone pursues it, it belongs in the L1a
sweep as another swept variant, selected on development data and judged on a fresh gate set.

**Whisper's ~700 ms bias** is §5.17: the experiment that can test it is now executable. What
38573 established is that the bias is uniform across both edges and both purposes, which is
consistent with an indexing convention and inconsistent with an acoustic failure.

---

## 6. Current gate state

### 6.1 Job 38573 — measured

**L0: passed**, all 24 gated criteria (27 rows, 3 report-only). `--overwrite` correctly
re-ran it. Qwen is recognised as runnable — `"ok: architecture provided by
qwen_asr.core.transformers_backend"`, so §5.10 is confirmed on the real environment and §4.1
stands. Three families budgeted (`alignment_families: 3`), two of them *timed* in the pilot
(CTC 4.75 utt/s, DTW 9.91 utt/s) with Qwen budgeted as the untimed third;
`projected_l1_alignment_gpu_hours = 0.710`, worst case 0.860, against the 12 h bound.

**L1a: passed**, all 10 gated criteria (36 rows). Qwen probe `ok` in 16.2 s of a 1800 s
deadline: 147 rows, 145 valid, frozen variant `qwen_forced_aligner/Chinese`. §5.9 is fixed
on the real environment.

**L1b: blocked**, exit 0. 27 gated criteria, **15 failing**, on:

| what | measured | required |
|---|---|---|
| `existing_ctc` valid-unit coverage | 0.97418 | ≥0.95 ✓ |
| `existing_ctc` invalid rate | **0.02582** | ≤0.01 ✗ |
| `existing_ctc` nonmonotonic rate | **0.02582** | ≤0.01 ✗ |
| `whisper_dtw` valid-unit coverage | **0.82875** | ≥0.95 ✗ |
| `whisper_dtw` invalid rate | **0.17125** | ≤0.01 ✗ |
| alignment unit coverage | 0.99800 (88,249 / 88,426) | ≥0.95 ✓ |
| synthetic within 100 ms (gate set, worst row) | **0.03** | ≥0.90 ✗ |
| synthetic median abs error | **720 ms** | ≤100 ms ✗ |
| synthetic p90 abs error | **1,146 ms** | ≤200 ms ✗ |
| synthetic max abs bias | **720 ms** | ≤50 ms ✗ |
| jitter ±50 ms mask IoU | 0.665 | ≥0.60 ✓ |
| jitter ±50 ms safe-interior survival | **0.532** | ≥0.90 ✗ |
| jitter ±100 ms mask IoU | **0.444** | ≥0.60 ✗ |
| jitter ±100 ms safe-interior survival | **0.377** | ≥0.90 ✗ |
| automatic usable-item rate | **0.282** (9,048 / 32,086) | ≥0.90 ✗ |
| per-role counts (4 criteria) | **all 0** | — ✗ *and this was a defect, §5.15* |

Scale: 1,800 bilingual utterances across six roles, 88,426 expected reference units,
176,672 candidate rows, 32,086 accepted spans (high 472 / medium 8,576 / low 23,038), 56,340
rejected. Rejection codes: 36,709 `start_disagreement_gt_200ms`, 17,398
`missing_second_aligner`, 2,233 `end_disagreement_gt_200ms`.

Tolerance sweep (EN retention is the number that matters):

| tolerance | accepted | accepted EN | EN retention | ZH retention |
|---|---|---|---|---|
| 50 ms | 472 | 12 | 0.00099 | 0.00602 |
| 100 ms | 6,975 | 234 | 0.01939 | 0.08828 |
| 200 ms | 32,086 | 2,413 | 0.19997 | 0.38860 |
| 300 ms | 43,665 | 4,899 | 0.40598 | 0.50768 |

Synthetic rendering and scoring worked: 100 dev + 100 gate pairs, sources disjoint,
score table authenticated. Blocked on `blocked_unselected_operating_tolerance`,
`blocked_insufficient_independent_aligners`, `blocked_no_paired_cross_aligner_evidence`.

**These measurements are not reinterpreted anywhere below.** A corrected implementation may
legitimately return `completed_no_go`, and on this evidence probably should.

### 6.2 What 38573 did *not* measure

- **Qwen on anything but the probe.** `candidates_all.parquet` contains 0 Qwen rows: the
  probe's 147 rows carry L1a's config identity, so L1b's cache check refused them (§5.19).
- **Any per-role count.** All zero, because `role` never reached the span table (§5.15).
- **Absolute error for the consensus estimator.** Only per-family error was scored; the
  operating tolerance governs the *consensus* seam, which is what §5.16 now measures.

---

## 7. Known unresolved gaps

### 7.1 The remaining blockers are scientific, not structural — superseding the earlier claim

The previous revision of this section said the ceiling was a preregistration: the operating
tolerance needed a human audit, so automatic Gate A could not pass, full stop. **§5.16
supersedes that.** The rule's numbers are unchanged and the human audit remains its
preregistered *validation*; the automatic path measures the same estimand with the instrument
it has. Nothing about the automatic workflow now requires annotation.

What blocks Gate A after that is the alignment itself, and on 38573's numbers it is not
close:

1. **No family qualifies.** CTC misses the 1% invalid limit by 1.6 points (§5.22 should fix
   this); Whisper misses the coverage floor by 12 points for a reason that cannot be fixed
   (§5.22). So the second independent aligner has to be Qwen, which has never run on more
   than eight utterances (§5.19 makes it possible, nothing has measured it).
2. **Absolute error against constructed boundaries is 720 ms** where 100 ms is required, with
   a 720 ms bias where 50 ms is required. §5.17 makes the one licensed repair testable. If
   the convention is the cause, the sweep will show it; if it is not, this is a genuine no-go.
3. **Jitter fails at ±100 ms on two of three criteria and at ±50 ms on one.** Nothing in this
   revision touches it, and nothing should: it is measured on the frozen primary subset and
   it is a statement about whether the mask survives being wrong.
4. **The usable-item rate is 0.282 against 0.90**, and the per-role counts, once real
   (§5.15), still miss their thresholds by an order of magnitude on loc-train.

Items 3 and 4 are `completed_no_go` material — the experiment ran and the answer is no. Items
1 and 2 are what a next run can move. **A corrected implementation reaching
`completed_no_go` would be the honest outcome, and it is the most likely one.**

### 7.2 What each remaining outcome means

| Outcome | Meaning | Exit |
|---|---|---|
| `blocked_no_qualifying_operating_tolerance` | the selection ran on development boundaries and nothing met 100/200 ms. Expected | 0 |
| `blocked_insufficient_independent_aligners` | fewer than two families clear validity **and** universe coverage | 0 |
| `blocked_uncalibrated_natural_families` | a family qualifies naturally but has no synthetic-gate score (§5.20) | 0 |
| `blocked_exposed_gate_set` | no unexposed source audio left for a fresh confirmatory gate (§5.18) | 0 |
| `completed_no_go` (external / jitter / coverage) | the experiment ran and the answer is no | 0 |
| `blocked_missing_manual_verdicts` | manual mode only; the optional audit has not been done | 0 |

### 7.3 Qwen: possible, unmeasured

§5.19 gives Qwen a real production path. **Nothing has run it.** What is known: the probe
aligned 8 utterances in 16.2 s including a checkpoint load, so ~1 s/utterance once loaded,
which is where `production_seconds_per_utterance: 4.0` comes from as a 4× guard. What is not
known: its invalid rate, its coverage of the 88,426-unit universe, its absolute error on
synthetic boundaries, or whether it agrees with CTC on any unit. All four are measured by the
next run, and all four can legitimately come out badly.

Note the dependency this creates: after §5.22, the second aligner is Qwen or nothing. If Qwen
fails its validity floors, Gate A blocks on `blocked_insufficient_independent_aligners`
whatever else improves.

### 7.4 Carried scientific findings — now largely superseded by job 38573

Job 38573 measured on 1,800 utterances what plan v2 had measured on 20, so these are recorded
with their status rather than repeated as current:

| plan v2 finding | 38573 |
|---|---|
| DTW ≈ **−499 ms** constant offset, E1's −490 ms | **confirmed and larger**: −720 to −840 ms signed on both edges, both purposes (§6.1) |
| tightening tolerance destroys English coverage (200 ms → 16 EN spans) | **confirmed at scale**: EN retention 0.200 at 200 ms, 0.019 at 100 ms, 0.001 at 50 ms |
| jitter failing: IoU 0.507, survival 0.738 | **confirmed, worse**: IoU 0.444, survival 0.377 at ±100 ms |
| EN consensus span median 225 ms vs 360 ms start disagreement | not re-measured; the 36,709 `start_disagreement_gt_200ms` rejections are consistent with it |
| `region` unbalanceable by pigeonhole (28 levels / 140 conversations) | unchanged, structural |

One plan-v2 claim is now **refuted**: that the constant offset is *not* a coordinate/convention
artifact. It may well be — §5.17's sweep is the experiment that decides, and it could not run
until now.

---

## 8. How to run the next job

`configs/lss/align.yaml`, `configs/lss/audit.yaml` and `configs/lss/l1a_diag.yaml` all
changed, so every stage's recorded config hash is stale and the chain would re-run L0 and L1a
anyway. `--overwrite` is still the right flag: the candidate tables must be rebuilt, because
§5.17 changes which Whisper variant produces them and §5.22 changes CTC's romanization, and
`--overwrite` is what forces `load_or_run` to rebuild rather than reuse an
identity-matching cache.

```
sbatch cs_asr_lss.sh chain --overwrite
```

| Stage | Estimate | Dominated by |
|---|---|---|
| L0 | ~5 min | role balancing (1:40) + decode pilots + Qwen capability import |
| L1a | ~10–15 min | rendering 100 dev pairs, then 2 × 100 Whisper alignments for the convention sweep, plus the Qwen probe |
| L1b | ~1.5–2.5 h | 1,800-utterance sweep for three families (Qwen is new and is the unknown), a fresh gate generation, synthetic scoring, the tolerance sweep, jitter and coverage |
| Total | **~2–3 h** | against a 20 h allocation |

38573 took 01:13:46 for two families; Qwen adds ~30 min at the probe's measured rate, and its
bound is 60 min plus 4 s/utterance (§5.19), so the allocation has ample headroom.

**Expected outcome.** L0 `passed`; L1a `passed` with
`metrics/l1a_pred_start_sweep.parquet` and `freeze/l1a_alignment_selection.json` written;
L1b **`blocked` on `blocked_no_qualifying_operating_tolerance`**, exit 0, unless the
convention sweep selects an offset that brings the consensus seam inside 100/200 ms on the
development set — in which case the gate proceeds to judge the fresh gate generation and
`completed_no_go` becomes the likely outcome on jitter and coverage.

**What to read first, in this order:**

1. `metrics/l1a_pred_start_sweep.parquet` — is the −700 ms bias a convention artifact? This
   is the single most informative new number.
2. `freeze/l1b_operating_point.json` — `selected_tolerance_ms`, and if null, the per-tolerance
   `rejection_reasons`.
3. `diagnostics/l1b_automatic_evidence.json` → `family_validity` — did CTC's invalid rate go
   to ≈0 (§5.22), and what are Qwen's numbers?
4. `diagnostics/l1b_coverage.json` → `per_role_spans` — non-zero now (§5.15), and how far
   short of the thresholds.
5. `reports/l1b_gate_a.md` → the "Required next actions" table (§5.21).

---

## 9. Files changed (uncommitted, on top of `6adbb49`)

Everything below §5.14 was already in place for job 38573. The **bold** entries are the
changes made *after* it and never run.

```
 M configs/lss/l0_freeze.yaml          include align.yaml; 4 new gate thresholds
 M configs/lss/align.yaml              **Qwen production deadline/chunking; probe_language**
 M configs/lss/audit.yaml              **selection thresholds + 5 blocked-reason responses**
 M configs/lss/l1a_diag.yaml           **include audit.yaml (synthetic block + thresholds)**
 M cs_asr_lss.sh                       --overwrite honoured; config-hash skip rule
 M src/csasr/data/ctc_alignment.py     uroman cache; numpy Viterbi;
                                       **CJK_NUMERAL_PINYIN (5.22)**
 M src/csasr/experiments/_common.py    **NA-safe md_table; complete != gate passed (5.21)**
 M src/csasr/experiments/lss_l0_freeze.py  fp32 site check; representative aligner
                                       sample; runnable-family workload; 3 criteria
 M src/csasr/experiments/lss_l1a_diag.py  **dev set + pred_start sweep + frozen
                                       selection (5.17); overlap attribution (5.22);
                                       dry runs cannot satisfy l1b**
 M src/csasr/experiments/lss_l1b_valid.py  sweep_sample(); synthetic scoring; min_units;
                                       **operating-point selection (5.16); fresh gate
                                       generation (5.18); Qwen runner (5.19); family
                                       correspondence (5.20); next_actions (5.21)**
 M src/csasr/experiments/lss_status.py stage_reusable()
 M src/csasr/lss/align/autoevidence.py BLOCKED_UNSELECTED_TOLERANCE; rewritten
                                       missing-synthetic description;
                                       **4 new codes; family_correspondence (5.20);
                                       gate-generation check (5.18)**
 M src/csasr/lss/align/candidates.py   **qwen_runner; pred_start pass-through;
                                       per-convention cache names (5.17, 5.19)**
 M src/csasr/lss/align/consensus_prod.py  **role join (5.15); instrument field;
                                       with_measured no longer un-selects**
 M src/csasr/lss/align/qwen_probe.py   language argument; PROBE_LANGUAGE; adapter seam
 M src/csasr/lss/align/repair.py       **attribute_overlaps (5.22)**
 M src/csasr/lss/align/synthetic.py    aligner_manifest, boundary_predictions,
                                       score_rendered_set, convention_comparison;
                                       **generation-labelled sets (5.18)**
 M src/csasr/lss/pilot.py              runnable_families(); per-family cost model
 M src/csasr/lss/sites.py              dtype-aware reconstruction report
 M src/csasr/lss/specfreeze.py         pilot -> VOLATILE_KEYS
 M src/csasr/nat5h/aligners.py         **pred_start_offset threaded into
                                       run_whisper_dtw + variant label (5.17)**
 M src/csasr/nat5h/schema.py           **one appended ALIGNER_VARIANT_ORDER entry**
?? src/csasr/lss/align/devselect.py    **new: development-only selection (5.16, 5.17)**
?? src/csasr/lss/align/exposure.py     **new: gate-set exposure ledger (5.18)**
?? src/csasr/lss/align/qwen_prod.py    **new: production Qwen alignment (5.19)**
 M tests/test_ctc_alignment.py         romanizer + Viterbi tests; **numeral tests**
 M tests/test_lss_align_diag.py        signature guard; subprocess end-to-end;
                                       **overlap attribution**
 M tests/test_lss_gate_a_auto.py       **next-action ladder; family correspondence**
 M tests/test_lss_l1b_production_paths.py  **selection, exposure, role, correspondence,
                                       status-contract end-to-end tests**
 M tests/test_lss_sites.py             dtype parametrisation; misplaced-hook test
 M tests/test_lss_specfreeze.py        rerun-idempotence test
?? tests/test_lss_chain_resumption.py  skip/overwrite rules
?? tests/test_lss_dev_selection.py     **new: selection + L1a artifact (5.16, 5.17)**
?? tests/test_lss_gate_set_exposure.py **new: the ledger (5.18)**
?? tests/test_lss_pilot.py             projection/cost-model + capability tests
?? tests/test_lss_qwen_production.py   **new: real subprocess, timeout, partial chunks**
?? tests/test_lss_role_propagation.py  **new: role through consensus (5.15)**
?? tests/test_lss_synthetic_scoring.py absolute boundary error
```

No file under `/mnt/data` was modified, no job was submitted, nothing was committed, and
`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v5.md` is untouched.

---

## 10. Review checklist — where I think problems are most likely

Ordered by how much damage a mistake would do.

0. **§5.16 — is swapping the instrument in a preregistered rule legitimate?** This is now the
   most consequential call in the document, ahead of everything below. My argument: the rule
   is about a quantity (absolute boundary error of accepted spans), the numbers are unchanged,
   the substituted instrument is one the same freeze already registers as external truth, the
   choice is made on development items only, and both instrument and rule are recorded in the
   frozen operating point. The opposing view — that "human" is *part of* the preregistered
   rule and substituting anything for it is a post-hoc change, so the automatic path should
   simply block forever — is coherent, and if you hold it the fix is to delete
   `devselect.select_operating_tolerance` and restore the unconditional block. What is *not*
   defensible either way is what the code did before: block, and tell the user to annotate.
0b. **§5.22 — is repairing the romanizer a repair or a way to clear a threshold?** It changes
   CTC's invalid rate from 0.0258 to (probably) ≈0, which is exactly what stands between the
   family and qualifying. My argument: uroman rendering 一 as `1` is unambiguously a defect,
   the model still has to find the span, and no invalid row was relabelled. But the timing —
   fixing the thing that blocks the gate — deserves scrutiny, and the counterfactual worth
   checking after the next run is CTC's *accuracy* on those units, not just its validity.
1. **§5.14 — should the gate judge both boundary conventions, or only the one steering
   consumes?** I excluded the legacy midpoint convention from the score table Gate A reads,
   on the grounds that it would fail the bias criterion for a convention nothing downstream
   uses. The opposing view — that a gate should see the worst of everything measured — is
   coherent. Note 38573's data cuts against me in one place: under the legacy convention CTC's
   gate-set median is **185 ms** against **670 ms** canonical, so the convention I excluded is
   the one that looks *better*, which is at least not self-serving.
2. **§5.1 — is the replacement gate genuinely stronger, or did I weaken it?** Check the
   arithmetic: 4 ULP of bf16 against a site/block gap of 0.4375 on a site of max-abs 1.64.
3. **§5.13 — is `min_units = ceil(0.95 × |universe|)` the right floor?** It makes per-family
   coverage universe-relative, which is what the proposal's 0.95 appears to mean, but it is
   a *stricter* reading than the code had, and it can now reject a family that the previous
   code accepted.
4. **§5.12 — is the swept sample the right denominator?** I argue the stage should be judged
   on what it set out to align. The alternative reading — that coverage is a claim about the
   whole role — implies the sweep should align the whole role, which is a much larger job.
5. **§4.2 — is one Qwen language variant right?** The evidence is about qwen_asr 0.0.6
   specifically. Check `encode_timestamp` yourself if the package is ever upgraded.
6. **§5.11 — is a config-hash-only skip rule safe enough?** It cannot see a code change.
   The mitigation is `--overwrite`, which is a human remembering. The alternative is
   re-running L0 and L1a on every job (~15 min), which may simply be the better trade.
7. **§5.8 — does the numpy Viterbi really match on the paths that matter?** The equivalence
   test uses random log-probs; real CTC posteriors are extremely peaked. Consider testing
   on real posteriors and on an all-`-inf` column.
8. **The workload constant.** `_workload` assumes 35 CS utterances per conversation and
   projects 4900 alignment utterances, while L1a+L1b align 300. Is the 12 h bound stated
   against the same definition? (Unchanged from the previous revision, still unresolved.)

---

## 11. Tests

**469 collected, all passing** (`pytest tests`, CPU only, no GPU and no model download).
`bash -n cs_asr_lss.sh` and `git diff --check` are clean.

| file | tests | what it pins down |
|---|---|---|
| `test_lss_dev_selection.py` | 22 | the operating-tolerance rule on development items, the gate-set leakage guard, the L1a sweep artifact and the frozen selection (§5.16, §5.17) |
| `test_lss_qwen_production.py` | 15 | full-manifest success, timeout + process-group kill, partial-chunk rejection, incomplete-universe discard, the deadline that is not the probe's (§5.19) |
| `test_lss_gate_set_exposure.py` | 11 | the ledger, exclusion of exposed sources, bootstrapping 38573's sets as exposed (§5.18) |
| `test_lss_role_propagation.py` | 8 | `role` through consensus for all six roles; non-empty input roles cannot become zero counts (§5.15) |
| `test_lss_l1b_production_paths.py` | 36 | every one of the above through `lss_l1b_valid.main()`, plus the status/exit contract |
| `test_lss_gate_a_auto.py` | 66 | the helper level, including the corresponding-pair false-pass path (§5.20) and the action ladder (§5.21) |
| `test_ctc_alignment.py` | 18 | the Han-numeral repair against a letters-only vocabulary (§5.22) |
| `test_lss_align_diag.py` | 37 | the Qwen probe's real subprocess, and overlap attribution (§5.22) |

Two properties the test design depends on, both enforced:

- **the stubs are pinned to the real interfaces.** `test_the_stub_adapters_match_the_real_adapter_signature` compares each stub's `run` call contract against `Qwen3ForcedAlignerAdapter.run`. A permissive `run(*a, **k)` is what let a call missing the required `language` argument reach the cluster in job 38502;
- **the substitutions are the GPU and the source audio, never the logic under test.** In the L1b end-to-end tests, `build_set`, `run_families`, `load_whisper` and `EncoderGeometry.from_bundle` are replaced; the manifest, the seam mapping, the scoring, the selection, the exposure ledger, publishing, and Gate A's read of all of it are real. The Qwen tests replace only the model checkpoint, and drive the actual subprocess.

**Not tested, and why.** Nothing exercises the real Qwen checkpoint, the real Whisper
cross-attention, uroman against MMS-FA's actual vocabulary, or the audio renderer — all four
need a GPU or a model download, which this work was not permitted to use. So the three claims
that need the next run to confirm are: that CTC's invalid rate actually falls once numerals
romanize to syllables; that Qwen covers the 88,426-unit universe within its deadline; and what
the `pred_start` sweep says about the −700 ms bias.
