# BASIS-A6 — Execution Plan (Compute / Cache / Slurm)

Status: **FROZEN PLAN — no GPU jobs launched by this stage.** Companion to
`BASIS_A6_EXPANDED_SPEC.md`, `BASIS_A6_FIXED_SPEC.md`, `BASIS_A6_ORACLE_TT_SPEC.md`,
`BASIS_A6_ASCEND_DATA_SPEC.md`. This plan is written so the study can be executed later under
the standard constraints; it is **design + freeze only**.

---

## 0. Standing constraints (from CLAUDE.md / AGENTS.md)

- All GPU work via `sbatch`; **≤ 2 simultaneous GPU jobs**; Round-2/3 code-only.
- Prefer **MIG** if measured VRAM ≤ 40 GB (Whisper and Qwen batch-1 decode both fit MIG per
  A4/A5 preflight: Qwen ~3.86 GiB; Whisper well under 40 GB).
- **Not** one job per layer/cell. Load each model **once per worker/shard**; steer many
  layers/cells/ρ per model load.
- Results resumable by the full scientific key; completed cells never re-decoded.
- Every run writes a provenance manifest (`csasr.utils.provenance` / `csasr.lss.specfreeze`).

---

## 1. Stage order (each resumable; ≤ 2 concurrent GPU jobs)

1. **Data prep (CPU, no inference).** `python scripts/download_ascend.py` → snapshot +
   provenance. Then freeze `ASCEND_CONSTRUCT_MANIFEST.json` (train) and
   `ASCEND_EVAL_MANIFEST.json` (validation) via the canonical adapter + eligibility rule
   (`BASIS_A6_ASCEND_DATA_SPEC §4,6,7`). No test read.
2. **Baselines (GPU).** 16 unsteered free-decodes = 2 models × 4 panels × 2 decode modes. Qwen
   `greedy ≡ official_standard` → decode Qwen once per panel, register both cells to the same
   hypothesis hash. These baseline hypotheses feed A6-TT decoder analysis (§4) and all metrics.
3. **A6-F direction construction (GPU, CPU-heavy).** For each construction source (CS, ASCEND)
   stream uncentered second moments + means in shared model passes (one forward per
   construction utterance records every layer/side/group), producing `v_raw`, `v_unique`,
   `v_shared` (enc+dec) and the paired-condition `v_cond_cs`, `v_cond_all` (dec). Persist
   directions + A5 diagnostics + cross-source geometry (A6-F §8). No per-utterance directions.
4. **A6-F steered decode (GPU).** Enumerate the 46,720 configuration cells (2 sources × 320/264
   direction-sites × 5 ρ × 2 decode × 4 panels). One model load per shard; sweep ρ and layers
   per load; reuse cached baseline **encoder** outputs for decoder-only steering (not for
   encoder steering).
5. **A6-TT per-sample construction + steered decode (GPU).** Per (model, panel, decode mode,
   utterance): analysis once → cache per-sample directions → steer & decode across all ρ
   (§4 below). 23,360 configuration cells.
6. **Aggregation (CPU).** Metrics, eligibility coverage, geometry tables, bundle hashes, figures,
   `FINAL_MANIFEST.json` + `FINAL_REPORT.md`. Duplicate keys = 0; empty provenance = 0.

Order rationale: data → baselines (needed by A6-TT decoder analysis) → A6-F construct →
A6-F decode → A6-TT. A6-F and A6-TT decode stages can interleave within the ≤2-job budget.

---

## 2. VRAM / model-load discipline

- Whisper-large-v3 and Qwen3-ASR-1.7B batch-1 decode → **MIG**. A 1-utterance preflight
  re-confirms peak VRAM per stage and records it in `runtime.json`.
- Construction second-moment accumulators (float64, per layer×2 groups×d²×8 B): ≈1.7 GB
  Whisper / ≈4 GB Qwen — tractable in MIG alongside the model; no giant activation cache. Raw
  activations are not persisted; directions + diagnostics are.

---

## 3. A6-F caching / reuse

- **Baseline encoder reuse:** cache baseline encoder outputs; reuse for decoder-only steering;
  do **not** reuse for encoder steering (needs a fresh forward).
- **Qwen decode-mode collapse:** Qwen `greedy ≡ official_standard`; steered Qwen decodes and
  baselines are computed once and registered to both decode-mode cells by identical hash. This
  is a caching optimization; the 46,720 enumerated cells are unchanged.
- **Comparators read-only:** A4/A5 baseline / Raw-local / Conditioning-local are read from
  `results/basis_a4/`, `results/basis_a5_unique_shared/`, never re-decoded; strict per-cell
  hash/site/mask/panel match before use.

## 4. A6-TT caching / reuse (construction must not repeat per ρ)

For each `(model, dataset, decode mode, utterance)`:
- **Analysis once** → cache per-sample `v_raw`, (`v_unique`,`v_shared` when `r_(x,l) ≥ 2`),
  `v_cond_cs`, `v_cond_all` per eligible (side, layer), plus `n_A`, `n_B`, `r_(x,l)`,
  direction/oracle-alignment hashes.
- **Encoder directions:** decoding-independent where technically exact → reuse across decode
  modes iff input/model states + local alignment hashes match; else cache per decode mode.
- **Decoder directions:** cache **per decode mode** unless the baseline hypothesis hashes are
  identical (Qwen greedy≡official_standard → one shared cache).
- **Paired Conditioning:** cache `c_E`/`c_M` analysis states/directions once per (utterance,
  decode mode, layer).
- Then reuse cached directions across `ρ ∈ {0.5,1,2,4,6}`; only the steered decode repeats per
  ρ. Record construction latency, extra analysis-pass count, and total inference latency/RTF.

---

## 5. Provenance artifacts

- Per-run manifests (resolved config, env, git, model metadata, hashes, status).
- A6-F: per-direction hashes (source, model, side, layer, method) + cross-source geometry rows.
- A6-TT: per-sample direction records → `DIRECTION_BUNDLE_HASH` per (model × dataset × side ×
  method × decode mode); aggregates reference the bundle hash.
- ASCEND: download provenance + both subset manifests/fingerprints (`BASIS_A6_ASCEND_DATA_SPEC
  §8`).
- Final: `results/basis_a6/FINAL_MANIFEST.json` + `FINAL_REPORT.md` with counts, duplicate-key
  = 0, empty-provenance = 0, and the exact realized `N` for `ASCEND-eval`.

---

## 6. Compute-scale note (planning only, not a reduction)

70,080 steered configuration cells + 16 baselines. Cost is dominated by decode, not
construction. The dominant multipliers are ρ (×5) and decode mode (×2); the Qwen decode-mode
collapse and per-sample direction caching (across ρ) remove redundant *compute* without
touching the enumerated configuration matrix. Sharding is by (model, panel) with many
cells per model load, respecting ≤2 concurrent GPU jobs. High-dose (ρ=4,6) destructive results
are retained. **No jobs are launched by this stage.**

---

## 7. Status: **FROZEN PLAN**

Stages, ordering, VRAM discipline, caching (A6-F and A6-TT), and provenance artifacts are
determined. Execution occurs only when explicitly authorized; this document launches nothing.
