# DG-01 — Canonical Metric & Result-Schema Specification

**Status:** Stage 2 / DG-01 IN PROGRESS (design). Companion to `METHOD_CONTRACT.md` (MC),
`CODE_MAP.md`, `EXPERIMENT_MATRIX.md`, `DATA_EXPOSURE.md`. This document specifies *what* the
implementation pass must produce; it does not implement metrics, change steering, or rewrite any
historical result. Scientific definitions come from the Stage-1 contract and are not reinterpreted
to match legacy code.

Design rule inherited from the codebase (`steer_sweep/metrics.py` header): **scoring is reused,
never reimplemented.** The canonical layer is a thin, schema-versioned facade over the already-
validated primitives in `csasr.evaluation.mer`, `csasr.evaluation.pier`, and `csasr.lss.outcomes`.

---

## 1. Canonical metric definitions

All metrics are computed on normalized text via the single versioned normalizer
(`csasr.evaluation.normalization`, `NORMALIZATION_VERSION`) and the single Levenshtein alignment
(`csasr.evaluation.mer.align_tokens`, backtrace `sub > del > ins`). A **unit** is one Mandarin
character or one English word (`segment_units`); a **POI** is an English content unit in the
*reference* of a code-switched utterance (`pier.reference_pois`).

| # | Metric | Canonical definition | Source primitive |
|---|---|---|---|
| 1 | **MER** | `(sub+del+ins) / num_ref_units`, mixed units, micro-averaged over the corpus | `mer.corpus_mer["mer"]` |
| 2 | **PIER** | `poi_errors / num_poi` over reference-English POIs | `pier.pier["pier"]` |
| 3 | **EN-WER** | English reference-unit errors / English reference units, **read off the one mixed alignment** | `mer.corpus_mer["en_wer"]` |
| 4 | **ZH-CER** | Mandarin reference-char errors / Mandarin reference chars, off the same alignment | `mer.corpus_mer["zh_cer"]` |
| 5 | **Correction (count/rate)** | baseline-incorrect → method-correct unit flip. **POI-level** `corrections`; **candidate-level** `n_corrected_inside` | `round1_metrics.pier_transition_counts` (POI) / `lss.outcomes.newly_introduced_errors` (candidate) |
| 6 | **Corruption (count/rate)** | baseline-**correct** → method-incorrect unit flip. POI-level `corruptions`; candidate-level `n_corrupted_inside` | same two surfaces |
| 7 | **Outside-edit / outside-harm (count/rate)** | **Canonical outside HARM** = correct→incorrect flip on a **trusted unit outside** the candidate (`n_corrupted_outside`). "Outside edit" (transcript difference outside a region) is reported separately and is **not** harm | `lss.outcomes.newly_introduced_errors["n_corrupted_outside"]` |
| 8 | **Matrix-language (Mandarin) retention** | `kept / baseline_correct` over Mandarin reference units near the intervention, `1 − corruption_rate(ZH)` | to build (§5 gap); today only `v2r3_gate_b.monolingual_retention` |
| 9 | **Embedded-language (English) retention** | `kept / baseline_correct` over `C_EN` = baseline-correct embedded-English units (MC §8); `1 − corruption_rate(C_EN)` | **absent** — to build from `pier.unit_status` restricted to EN units |
| 10 | **Gate coverage** | fraction of the **gate-eligible population** where `g_{ℓ,t} > 0` | **denominator not fixed by Stage 1 → see §4 / Blockers**; gate does not yet exist (MC §6) |
| 11 | **Baseline-vs-method delta** | `ΔM = M_baseline − M_method` (§3) for every error metric | new canonical helper |
| 12 | **Run/result schema** | §7 | new |
| 13 | **Run manifest/provenance** | §8 | `utils.provenance` + `lss.manifest` (reuse) |

**Two correction/corruption levels are kept explicitly distinct** (both are contract-legitimate,
never merged): **POI-transition** accounting (population = reference POIs; PIER-consistent, obeys
the PIER transition identity `round1_metrics.assert_pier_identity`) and **candidate** accounting
(population = reference units assigned to an acoustic candidate by ≥0.5 overlap; feeds utility,
MC §10). Each carries its population in its field name.

---

## 2. Denominator / unit / level / aggregation for each metric

| Metric | Denominator | Unit | Level | Aggregation | Alignment-dependent |
|---|---|---|---|---|---|
| MER | reference units | mixed char/word | corpus | micro | yes |
| PIER | reference POIs | EN word | corpus | micro | yes |
| EN-WER | EN reference units | EN word | corpus | micro | yes |
| ZH-CER | ZH reference chars | ZH char | corpus | micro | yes |
| Correction rate (POI) | baseline-incorrect POIs | POI | corpus | micro | yes |
| Corruption rate (POI) | baseline-correct POIs | POI | corpus | micro | yes |
| Correction/corruption (candidate) | inside-candidate units | unit | candidate→corpus | micro (summed sets) | yes + acoustic spans |
| Outside harm rate | trusted units outside candidate | unit | candidate→corpus | micro | yes + spans |
| Matrix (ZH) retention | baseline-correct ZH units | ZH unit | corpus | micro | yes |
| Embedded (EN) retention | baseline-correct EN units (`C_EN`) | EN word | corpus | micro | yes |
| Gate coverage | gate-eligible positions (**§4**) | decoder step | corpus | micro | no (gate artifact) |

Reported CIs are **dialogue/conversation-block bootstrap** (`evaluation.bootstrap.paired_bootstrap`,
cluster = `dialogue_id`, 10,000 resamples for final intervals), never item bootstrap — matching
`steer_sweep.metrics.delta_pier`'s existing cluster policy.

---

## 3. Delta / sign convention (LOCKED for canonical reporting)

```
ΔM = M_baseline − M_method      for every error metric (MER, PIER, EN-WER, ZH-CER, corruption)
```

**Positive Δ always means improvement.** This matches MC §10 (`PIER_gain = PIER_baseline −
PIER_method`) and `steer_sweep.metrics.delta_pier` (already `baseline − steered`, positive-better).

- Canonical fields are **named** with their direction: `pier_gain`, `mer_gain`, `en_wer_gain`,
  `zh_cer_gain` (all positive = better). The bare ambiguous name `delta_pier` is **not** emitted by
  the canonical layer.
- Correction/corruption are **counts and rates**, not deltas; `net_corrections = corrections −
  corruptions` (positive = better).
- Legacy fields keep their original signs and are only surfaced through the read-only adapter
  (§9), which re-signs them into `*_gain` before any comparison.

---

## 4. Gate coverage denominator — **deferred, do not choose silently**

Stage 1 does **not** fix a single denominator for "gate coverage." MC §6 leaves the factorized
gate's functional form an `OPEN DECISION`; the gate, localizer, and utility selector are unbuilt
(MC §6 / CODE_MAP "Absent"). Existing code offers only unrelated notions —
`G3_dialogue_coverage` (dialogues containing ≥1 correction) and Gate T "practical decoder
coverage" (fraction of target steps reachable). Neither is "fraction of positions the gate fires."

**Decision:** the canonical layer **reserves** the `gate_coverage` field and its
`gate_coverage_denominator` label but does **not** compute a value in DG-01. The denominator
(candidate-eligible decoded positions vs all non-prefix decoded positions vs eligible spans) is
resolved with the gate itself in the gate ticket (DG-03+), because (a) no gate artifact exists to
measure and (b) MC §1's "forced-prefix positions get zero gain" fixes only that prefix positions
are excluded, not the eligible base. Recording a value now would be choosing a scientific
definition without evidence — forbidden. See Blockers.

---

## 5. Legacy inconsistencies found (verified in repo)

1. **Delta sign conflict.** `steer_sweep.metrics.delta_pier` = `baseline − method` (positive
   better); `steer_sweep.round1_metrics.paired_metric_report` emits `delta_MER`/`delta_PIER` =
   `method − baseline` (negative better). Canonical layer re-signs to `*_gain` (§3).
2. **Two WER definitions.** `mer.corpus_mer` gives `en_wer`/`zh_cer` off the single mixed
   alignment (mutually consistent with MER). `v2r3_day5_expansion.corpus_word_error_rate` is a
   whitespace-split blended `wer` with a different alignment and **no** EN/ZH breakdown.
   `steer_sweep.corpus_metrics` currently reports the latter as `WER`. **Canonical = the mer.py
   language-aware pair;** `corpus_word_error_rate` is legacy.
3. **Three "outside edit" fields, none is harm.** `correction_harm.outside_region_edits`
   (projection diff outside ±1 ref radius), `round1_metrics.outside_edit_rate` (whole-transcript
   edit distance per utterance), `job_b_training.outside_edits` (summed target spillover). Canonical
   outside **harm** is the correctness flip `outcomes.n_corrupted_outside` (MC §10). The three edit
   fields are retained as descriptive "transcript-change" magnitude only.
4. **Retention conflation + missing surface.** `v2r3_gate_b.monolingual_retention` (surfaced as
   `zh_retention`) counts **all** non-EN baseline-correct units, mixing matrix-Mandarin near the
   site with whole-utterance monolingual Mandarin — MC §8 keeps these as **two distinct groups**,
   and adds a **third**, **embedded-English (`C_EN`) retention, which has no implementation at all.**
5. **Field-name drift.** `en_wer/zh_cer` (mer.py) vs `EN_WER/ZH_CER` (round1_metrics) vs `WER`
   (steer_sweep). Canonical fixes one casing (`en_wer`, `zh_cer`).
6. **Correction/corruption populations differ silently** between POI-transition and candidate
   accounting; canonical names carry the population (§1).

---

## 6. Proposed canonical module location

New package `src/csasr/metrics_canonical/` **or** (preferred, minimal) new modules inside the
existing `src/csasr/evaluation/` package so the reuse imports stay local:

- `src/csasr/evaluation/canonical.py` — the single authoritative facade. `SCHEMA_VERSION =
  "metrics_v1"`. Functions: `corpus_metrics(refs, base, method, spans=None)` returning MER, PIER,
  EN-WER, ZH-CER, POI corrections/corruptions, `*_gain` deltas, `net_corrections`; and
  `retention(refs, base, method)` returning the three MC §8 retention groups. Internally calls
  `mer.corpus_mer`, `pier.pier`, `round1_metrics.pier_transition_counts`,
  `outcomes.newly_introduced_errors`. No new alignment or normalization is introduced.
- `src/csasr/evaluation/retention.py` — the three retention surfaces: `embedded_en_retention`
  (`C_EN`), `matrix_zh_retention` (near-site), `monolingual_retention` (reuses the v2r3 helper).
- Candidate-level correction/harm/utility stays in `src/csasr/lss/outcomes.py` (already canonical,
  MC §10) and is imported, not duplicated.

---

## 7. Proposed result-schema location

`src/csasr/evaluation/result_schema.py`, `RESULT_SCHEMA_VERSION = "result_v1"`. A canonical result
record is **result-metrics only** (no scientific claims, no GO/NO-GO verdict — those live with the
gate machinery, `csasr.lss.gates`). Fields:

```
schema_version, run_id, system_name, model_id, model_revision,
dataset, split, data_role, seed, decode_regime, beam,
layer, direction_artifact_id, gate_config_id, baseline_ref_run_id,
metrics: {                          # counts, rates, and *_gain deltas
  mer, pier, en_wer, zh_cer,
  mer_gain, pier_gain, en_wer_gain, zh_cer_gain,
  num_poi, corrections, corruptions, net_corrections,
  correction_rate, corruption_rate,
  candidate: {n_corrected_inside, n_corrupted_inside, n_corrupted_outside, ...},
  retention: {embedded_en, matrix_zh, monolingual_zh},
  gate_coverage: null, gate_coverage_denominator: null   # reserved, §4
},
descriptive: {outside_edit_rate, pct_transcript_changed, utterance_wins/ties/losses},
provenance_ref                      # pointer to the manifest of §8
```

`schema_version` gates every reader; `provenance_ref` separates the reproducibility block (§8) from
the metrics block so scientific interpretation never lives in the artifact.

---

## 8. Proposed manifest structure

Reuse existing primitives — do not invent a new provenance format:

- `csasr.utils.provenance.stage_provenance(cfg, stage)` already emits `provenance_version`,
  `model_id`, `model_revision`, `dataset_manifest_hash`, `resolved_config_hash`,
  `source_snapshot_hash`, `normalization_version`, `seed`, `created_at`, `production_artifact`.
- `csasr.lss.manifest.publish` / `run_id` / `identity` for the run id, dataset fingerprint, and
  git/config identity; `csasr.utils.hashing` for content hashes.

The DG-01 manifest = `stage_provenance(cfg, "dg01_metrics")` **plus** `git_commit` (HEAD),
`config_hash` (`resolved_config_hash`), `dataset_fingerprint` (`dataset_manifest_hash`),
`normalization_version`, `metrics_schema_version`, `result_schema_version`, and `legacy_source`
(the origin artifact path + its original field names/signs) when the record is produced by the
adapter of §9. Every canonical result carries exactly one manifest; the manifest is the only place
reproduction identity lives.

---

## 9. Legacy compatibility strategy

- **Read-only.** No file under `results/`, `out/`, `results/round1/` is rewritten (MC / AGENTS
  invariant 3).
- **Adapter, not migration.** `src/csasr/evaluation/legacy_adapter.py` parses
  `results/job_a/summary.json`, `results/job_b/summary.json`, and `results/round1/*/summary.json`
  into the `result_v1` schema **in memory**, re-signing `delta_*` (method−baseline) into `*_gain`
  (baseline−method) and stamping `legacy_source` + `production_artifact=false`. It never claims a
  legacy record as exact-site evidence.
- **Schema-versioned coexistence.** New canonical runs write `result_v1`; legacy artifacts keep
  their bytes and are only ever surfaced through the adapter. A reader must refuse to merge records
  whose `schema_version` differs without passing through the adapter.

---

## 10. Exact files to create / edit in the implementation pass

Create:
- `src/csasr/evaluation/canonical.py` — facade + `SCHEMA_VERSION` (§6).
- `src/csasr/evaluation/retention.py` — three MC §8 retention surfaces (§6).
- `src/csasr/evaluation/result_schema.py` — `result_v1` builder/validator (§7).
- `src/csasr/evaluation/legacy_adapter.py` — read-only legacy→`result_v1` (§9).
- `tests/test_canonical_metrics.py`, `tests/test_result_schema.py`,
  `tests/test_legacy_adapter.py` (§12).

Edit (documentation/wiring only, no scientific logic change):
- `docs/current/CODE_MAP.md` §3 — point "canonical metric surface" at the new modules once built.
- `docs/current/STATUS.md` — DG-01 progress (done in this pass, see below).

**Do not touch** in DG-01: `csasr.lss.sites`, `models/hooks.py`, `directions/*`,
`job_b_training.py` training logic, any config/YAML steering parameters, any hook or decoding path.

---

## 11. Regression fixtures / outputs to use

- **Golden text triples.** A small committed fixture of `(reference, baseline_hyp, method_hyp)`
  covering: pure correction, pure corruption, outside-harm, insertion-near-POI, deletion POI
  (missing-status path), monolingual-ZH utterance, and a no-change identity pair.
- **PIER identity check.** Reuse `round1_metrics.assert_pier_identity` as an invariant the facade
  must satisfy on every fixture.
- **Legacy parse target.** `results/job_a/summary.json` (has the full `correction_harm` block and
  `num_baseline_correct = 0` edge case) and `results/round1/job_b/summary.json` — the adapter must
  round-trip these into `result_v1` without error and with correctly re-signed `*_gain`.
- Do **not** read `/mnt/data/...` artifacts or large result trees; the committed `results/*.json`
  summaries are sufficient.

---

## 12. Acceptance tests

1. **Sign convention:** for every fixture, `pier_gain == baseline_pier − method_pier` and a pure
   corruption fixture yields `pier_gain < 0`, a pure correction `pier_gain > 0`.
2. **Reuse identity:** facade MER/PIER/EN-WER/ZH-CER equal the underlying `mer.corpus_mer` /
   `pier.pier` values bit-for-bit (no reimplementation drift).
3. **PIER transition identity:** `assert_pier_identity` passes on all fixtures.
4. **Outside harm ≠ outside edit:** a fixture that improves a neighbour (transcript change, no
   correctness loss) has `n_corrupted_outside == 0` while an outside-edit field is nonzero.
5. **Three retention groups:** embedded-EN, matrix-ZH, monolingual-ZH are computed independently;
   a monolingual-ZH-only fixture leaves embedded-EN retention `NaN` (empty denominator), not 0.
6. **Edge cases:** empty POI set → PIER `NaN`; deletion POI → missing decoder status handled;
   `num_baseline_correct == 0` → corruption rate `NaN`, not a divide error.
7. **Schema:** `result_v1` validates required keys; `gate_coverage` is `null` and any attempt to
   populate it raises (guards §4).
8. **Legacy adapter:** parses the two committed summaries, re-signs deltas, stamps `legacy_source`,
   and marks `production_artifact=false`; refuses to output them as canonical exact-site evidence.

---

**Not resolved here (correctly deferred):** gate-coverage denominator (§4, Blockers); the exact
candidate-vs-POI weighting for headline correction claims (belongs to Gate B / MC §10 usage, not to
metric plumbing). DG-01 canonicalizes computation and schema; it does not choose these scientific
knobs.

---

## 13. Implementation status (updated during the DG-01 implementation pass)

**Built and unit-tested (CPU-only):**
- `src/csasr/evaluation/canonical.py` — `SCHEMA_VERSION = "metrics_v1"`; `corpus_metrics`, `gain`,
  `error_metric_gains`, `correction_corruption` (POI-level, denominators + rates),
  `candidate_outcomes` (canonical `outside_harm = n_corrupted_outside`), `paired_corpus_report`
  (asserts the PIER transition identity). Reuses `mer.corpus_mer`, `pier.pier`,
  `lss.outcomes.newly_introduced_errors`, `round1_metrics.assert_pier_identity` — no rescoring.
- `src/csasr/evaluation/retention.py` — `matrix_zh_retention`, `embedded_en_retention`,
  `monolingual_retention`, `retention_report`. Code-switched vs monolingual is decided from
  reference language tags; each population returns `{numerator, denominator, rate}`.
- `src/csasr/evaluation/result_schema.py` — `RESULT_SCHEMA_VERSION = "result_v1"`,
  `CanonicalResult`/`MethodConfig`, `validate`, `reserved_gate_coverage`, `populate_gate_coverage`
  (guard that raises), deterministic `to_json`/`from_json`.
- `src/csasr/evaluation/legacy_adapter.py` — `adapt_job_a_summary`, `adapt_job_b_summary`,
  `adapt_round1_paired_report`, `resign_delta`. Read-only; re-signs `method − baseline` deltas;
  stamps `legacy-derived` + `production_artifact=False`; leaves retention/outside-harm `null`.
- Tests: `tests/test_canonical_metrics.py`, `tests/test_retention.py`,
  `tests/test_result_schema.py`, `tests/test_legacy_adapter.py`.

**Deliberately deferred:** gate coverage stays `null` (guarded; §4). Legacy execution scripts
(`experiments/*`, `steer_sweep/*`) are **not** rewired to emit `result_v1` in this pass. Provenance
plumbing reuses `utils.provenance`/`lss.manifest` at call sites; the canonical modules carry a
`provenance` dict that a runner fills — no competing provenance system was introduced.

---

## 14. Regression-validation + emission pass (completion — DG-01)

Completed in the DG-01 completion pass (regression validation + canonical emission + final gate).

**Adapter fix (blocking bug found + fixed):** the committed Job-B `summary.json` stores its flat
per-system metrics under **`results_table`** (its `methods` block only nests `epoch_results` and has
no top-level `MER`). `legacy_adapter.adapt_job_b_summary` previously read a nonexistent `systems`
key and silently emitted **only the baseline**, dropping every method system. It now reads
`results_table` (falling back to `methods`/`systems`), takes the top-level `baseline` block as the
gain reference, and skips the baseline-named row so it is emitted once. Job A was already correct.

**Regression artifacts checked (read-only, bytes untouched):**
- `results/job_a/summary.json` — F0 baseline + F1–F5. MER/PIER are reproduced **exactly**
  (EXACT REPRODUCTION); `pier_gain = PIER_F0 − PIER_system` (positive = improvement, e.g. F2 > 0);
  `correction_harm.{num_corrected,num_corrupted}` map identically to `corrections`/`corruptions`;
  the `num_baseline_correct = 0` edge case yields `corruption_rate = null` (not a divide error).
- `results/job_b/summary.json` — baseline + T1/T2/T3 (greedy + beam5) from `results_table`.
  MER/PIER reproduced exactly; `pier_gain` re-based on the top-level baseline; `zh_retention`
  preserved only as `descriptive.legacy_zh_retention_blended` (**SCHEMA/NAMING MIGRATION** — the
  blended matrix+monolingual population is *not* decomposed into MC §8 groups); `outside_edits`
  preserved only as `descriptive.legacy_outside_edits` (**SEMANTICALLY DIFFERENT** — never mapped to
  canonical `outside_harm`, which stays `null`).
- `round1_metrics.paired_metric_report` shape — `delta_MER`/`delta_PIER` (`method − baseline`) are
  re-signed to `mer_gain`/`pier_gain` (`baseline − method`). No committed file of this exact shape
  exists (`results/round1/job_a/summary.json` is a status stub; `results/round1/job_b/summary.json`
  matches the Job-B `results_table` shape), so this path is covered by a synthetic-record test.

**Canonical emission path (new, current — not legacy):** `src/csasr/evaluation/result_emit.py`
(`build_provenance`, `emit_result`, `emit_baseline_and_method`). It scores an already-produced text
triple through the reuse facade only (no rescoring) and stamps a real manifest built from
`utils.provenance.stage_provenance` + `utils.logging.git_state` (HEAD commit) + `lss.manifest.run_id`
+ `utils.hashing`. It changes **no** decoding/steering/training/data behaviour. Verified emission:
`git_commit` = real 40-char HEAD, `config_hash` = real `resolved_config_hash`,
`direction_artifact_hash`/`calibrator_artifact_hash` = `null` (no artifact ⇒ not fabricated),
`schema_version = result_v1`, correct `data_role`/`decode_regime`, `baseline_ref_run_id` set on the
method record, canonical `pier_gain ≥ 0`, deterministic JSON round-trip, no `NaN`/`Infinity`.

**Duplicate-logic scan (Phase D):** no second active `result_v1`/`metrics_v1` schema and no second
canonical gain-sign convention exist. Remaining `*_gain`/`delta` occurrences are all **LEGACY —
preserve for reproducibility** and non-conflicting: `steer_sweep.metrics.delta_pier`
(`baseline − method`, same direction, superseded), `e4_oracle.pier_relative_gain` /
`e5_boundaries.pier_abs_gain` (legacy feasibility, `baseline − method` direction),
`v2r3_day5_expansion.corpus_word_error_rate` (legacy blended WER). The one opposite-sign legacy
convention (`round1_metrics.delta_PIER = method − baseline`) is reached only through the re-signing
adapter. No `CONFLICT` (blocking) definition was found.

**Regression tests added:** `tests/test_dg01_regression.py` (Job A + Job B baseline/method
round-trip, MER/PIER preservation, both delta directions, null legacy provenance, blended-retention
non-decomposition, outside-edit ≠ outside-harm, `result_v1` validation, canonical emission pair
round-trip with real provenance). The four original focused suites are unchanged.

**Remaining for the DG-01 regression-validation pass (separate ticket):** — none. DG-01 is complete.
Non-blocking debt carried forward: gate-coverage denominator (§4, DG-03+); wiring the *scientific*
free-decoding runner to call `result_emit` is DG-02+ work (the emission helper exists and is tested,
but no exact-site free-decoding runner exists yet to feed it — G1/G5 in STATUS).
