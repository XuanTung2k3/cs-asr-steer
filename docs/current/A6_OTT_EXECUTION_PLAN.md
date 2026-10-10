# A6-OTT — Execution Plan (Superseded-Job Policy · Reuse · Compute)

Status: **FROZEN PLAN — no new scientific GPU jobs submitted by the freezing session.**
Companion to `A6_OTT_SPEC.md`, `A6_OTT_SELECTION_PROTOCOL.md`. Migration evidence in
`results/a6_ott_upper_bound/migration/`.

---

## 1. Superseded A6 job policy (FROZEN)

- The large BASIS-A6 execution (70,080-cell A6-F/A6-TT atlas) is **superseded for execution**.
  Its wave coordinator (`53354`, `basis-a6-waves`) and atlas array (`53355`, `basis-a6-atlas`)
  were **cancelled** this session (`migration/SLURM_JOB_AUDIT.md`); `squeue` is empty. At cancel,
  only **1 of 70,080** rows had completed (`a6_f=1, a6_tt=0`).
- **Never delete** `results/basis_a6_expanded/`, old Slurm logs, or any valid partial output.
  They are preserved as reusable evidence (§2). Superseded ≠ erased.
- Do not resubmit the atlas coordinator or array. Any future A6-OTT run uses the compact,
  phased driver only, respecting **≤ 2 concurrent GPU jobs** and the ≤2.5 h target / 3 h hard
  job-time policy (from `preflight/RUNTIME_ESTIMATES.json`).

## 2. Exact-result reuse policy (FROZEN)

A result is reused **only if all** scientifically relevant fields match: model revision,
dataset, exact utterance id, side, layer, steering family, oracle per-sample construction,
direction-construction semantics, intervention site, local-mask semantics, ρ, decode mode,
baseline-hypothesis analysis semantics, NormPreserve, metric version, oracle information budget.
Conditioning-CS additionally requires identical `c_E`/`c_M` semantics, identical
baseline-hypothesis sequence, and identical CS-position definition.

- **Never** reuse corpus-level fixed-direction results as oracle-TT
  (`migration/EXISTING_RESULT_REUSE_MANIFEST.json` → `INVALID_OLD_RESULT`).
- **Never** reuse a dataset-wide Add-Unique vector for per-sample Add-Unique.
- Classifications used: `REUSE_EXACT`, `NEED_RUN`, `INVALID_OLD_RESULT`,
  `PARTIAL_REQUIRES_VALIDATION`.

**What is actually reusable now** (from the reuse manifest):
- **`REUSE_EXACT` — 16 baseline decodes** (`baselines/{model}/{dataset}/{greedy,official_standard}.json`):
  unsteered free-decode, identical between BASIS-A6 and A6-OTT for the same model/dataset/mode;
  needed for baseline-hypothesis analysis and as comparison reference. Load gated on
  model-revision + panel-fingerprint + decode-hash. (Outcome metrics of confirm/SEAME baselines
  remain firewalled during Phase-A per `A6_OTT_SELECTION_PROTOCOL §2`.)
- **`PARTIAL_REQUIRES_VALIDATION` — whisper/cs/greedy/encoder Add-Unique per-sample direction
  caches at L0, L1** (`oracle_tt/cache/.../encoder/{L00,L01}/`): the cached per-sample **direction
  vectors** can be reused to skip re-construction for Search utterances whose IDs fall in the
  frozen `CS_SEARCH_20`, after validating oracle budget (`gold_leakage=false`), mask, rank,
  NormPreserve, and baseline-hypothesis semantics. The **aggregate decoded rows** from that old
  run are **not** reusable (they aggregate over the full ~300 CS panel, not the 40-utt Search
  subset) and are classified `INVALID_OLD_RESULT`.
- Everything else is `NEED_RUN` (all Qwen oracle-TT; all decoder families; Conditioning-CS;
  whisper encoder L2–L31; ASCEND/SEAME oracle-TT; all `official_standard` oracle-TT).

The reuse-planning code may detect compatible artifacts by metadata/IDs/hashes/manifests at any
phase, but must obey the leakage firewall for **outcome** reads.

## 3. Leakage firewall (pointer)

The Search freeze is outcome-blind; confirmation/SEAME outcome metrics are not read during
Phase-A selection; Phase-B outcomes only after Phase-A freeze; SEAME only after final freeze.
Full rules and the code-enforcement obligation are in `A6_OTT_SELECTION_PROTOCOL §2`.

---

## 4. Compute accounting — LOGICAL vs PHYSICAL (measured basis)

Per-steered-evaluation cost on H100 MIG 3g.40GB, from `preflight/RUNTIME_ESTIMATES.json`
(seconds/utt × measured TT-overhead factor): **Whisper ≈ 3.56 s/eval (greedy)**, **Qwen ≈ 2.66
s/eval (greedy)**; Whisper `official_standard` (beam-5) ≈ ~3× greedy; Qwen `official_standard ≡
greedy`. Directions are cached once per (model,dataset,decode,utterance) and reused across ρ.

| Phase | Model | Logical evals | Reused (exact) | Remaining physical | Est. GPU-h |
|---|---|---:|---:|---:|---:|
| **A Search** | Whisper | 11,520 | 0¹ | 11,520 | ~11.4 |
| **A Search** | Qwen | 12,800 | 0 | 12,800 | ~9.5 |
| **B Confirm** | Whisper | ≤5,760 | 0² | ≤5,760 | ~11.4 |
| **B Confirm** | Qwen | ≤5,760 | 0² | ≤5,760 | ~4.3 |
| **C Transfer** | Whisper | ≤600 | 0² | ≤600 | ~1.2 |
| **C Transfer** | Qwen | ≤600 | 0² | ≤600 | ~0.4 |
| **Baselines** | both | (16 decodes) | **16 REUSE_EXACT** | 0 | ~0 (reused) |

¹ Whisper Phase-A: per-sample **direction construction** for Add-Unique encoder L0/L1 on the CS
Search utterances is reusable (`PARTIAL`), but the 40-utt Search **decode/aggregate** must still
run, so exact evaluation-row reuse = 0. ² Phase-B/C reuse is 0 at freeze time (no confirm/SEAME
oracle-TT exists yet) and is gated behind the firewall's phase ordering.

**Totals:** Phase-A ~20.9 GPU-h · Phase-B ~15.7 GPU-h · Phase-C ~1.6 GPU-h →
**~38 GPU-h physical** (+~15% model-load/queue slack ≈ **~44 GPU-h**). Conditional Qwen rescue
(if triggered): **< 1 GPU-h** (20 utts × bounded decoder screen).

**Context:** the superseded atlas was estimated at ~**5,635 GPU-h**
(`RUNTIME_ESTIMATES.json`: A6-F ~2,462 + A6-TT ~3,173). A6-OTT at ~44 GPU-h is a **~130×
reduction**.

## 5. Revised wall-time estimate

At ≤2 concurrent MIG jobs and ≤2.5–3 h/job: ~44 GPU-h / 2 ≈ **~22 h of active dual-GPU
compute**. With array throttling (`%2`), queueing, and the slower Whisper beam-5 Phase-B/C
batches, realistic wall-clock ≈ **~1–2 calendar days** end-to-end (Search → Confirm → Transfer,
run sequentially to honour the firewall's phase ordering). Phase A dominates; Phase C is trivial.

## 6. Stage order (each resumable; ≤2 concurrent GPU jobs)

1. Freeze `SEARCH_FREEZE.json` (outcome-blind, CPU).
2. **Phase A** greedy Search (Whisper ρ{0.5,1,2}; Qwen ρ{0.5,1,2,4}); reuse validated Whisper
   enc AU L0/L1 direction caches; reuse baselines. → `PHASE_A_SELECTION.json` (top-2/family/model
   by the frozen rule).
3. Evaluate the Phase-A **Qwen rescue trigger**; run bounded rescue only if it fires.
4. **Phase B** Confirm (≤6 settings/model, greedy + official_standard) → `PHASE_B_CONFIRM.json`
   → freeze `PHASE_B_FINAL_SETTINGS.json` (before any SEAME read).
5. **Phase C** Transfer on SEAME (greedy + official_standard) → `PHASE_C_TRANSFER.json`.
6. Aggregate → `results/a6_ott_upper_bound/FINAL_MANIFEST.json` + `FINAL_REPORT.md` (bundle
   hashes, eligibility coverage, reuse ledger, firewall audit).

## 7. Status: **FROZEN PLAN**

Superseded-job policy, exact-result reuse policy (with the leakage firewall), measured
logical/physical accounting, and revised GPU-hour/wall-time estimates are determined. Execution
occurs only when explicitly authorized; this session launches no new scientific GPU jobs.
