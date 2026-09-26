# BASIS-A5 — Execution Plan (Compute / Slurm)

Status: **FROZEN PLAN — no GPU jobs launched by this stage.** Companion to
`BASIS_A5_SPEC.md`. Base state BASIS-A4 `d00c54c`.

This plan is written so the local-only atlas can be executed later under the standard
constraints; it is **not** run now (design + freeze only). A5 is **LOCAL-ONLY** at
`rho=0.5`; there are no Global runs, no dose sweep, no new panels, and no re-decode of A4
comparators.

---

## 0. Standing constraints (from CLAUDE.md / AGENTS.md)

- All GPU work via `sbatch`; **≤ 2 simultaneous GPU jobs**; Round-2/3 are code-only.
- Prefer **MIG** if measured VRAM ≤ 40 GB; request a full/main GPU only if required.
- **Not** one job per layer. Load each model **once per worker/shard**; steer many
  layers/cells per model load; all A5 layers/directions share one model load.
- Reuse cached **baseline encoder** outputs for **decoder-only** steering; **do not** reuse
  baseline encoder output for **encoder** steering (encoder steering needs a fresh forward).
- Results resumable by `(utterance, model, side, direction, layer, scope=oracle_local,
  rho=0.5)`; completed cells never re-decoded.
- Free decoding exactly as A4: greedy, temp 0, beam 1, `condition_on_prev_tokens=false`,
  batch 1 per worker, `max_new_tokens=200`; Whisper `language=zh`; Qwen
  `force_language="Chinese"`, `context=""`.

---

## 1. VRAM sizing (determines MIG vs main)

- **Whisper-large-v3** batch-1 decode: well under 40 GB → **MIG** (as A3/A4).
- **Qwen3-ASR-1.7B** batch-1 decode: A4 preflight measured **3.86 GiB** peak → **MIG**.
- **Construction** (streaming second moments, §3 below) holds ≤ (#layers×2×d²×8 B) ≈ 1.7 GB
  (Whisper) / ≈ 4 GB (Qwen) of float64 accumulators plus the model → **MIG** is sufficient;
  a 1-utterance preflight re-confirms peak VRAM and records it in `runtime.json`.

---

## 2. Stages (ordered; each resumable; ≤ 2 concurrent GPU jobs)

**S0 — Protocol freeze (this stage, CPU only).** Seal `BASIS_A5_SPEC.md` +
`BASIS_A5_EXECUTION_PLAN.md`, the A4 invariant fingerprints reused (§1 of the spec), the
A5 protocol hash, and git/commit provenance under `results/basis_a5_atlas/spec/`. No GPU.

**S1 — A4 reuse/comparator audit (CPU only).** Verify the A4 comparators to be referenced
(Raw-local, Conditioning-local, baseline) still match by protocol hash, direction hash,
site, mask hash, panel fingerprint, and model revision. Emit
`results/basis_a5_atlas/reuse/comparator_manifest.json` (read-only pointers; nothing
re-decoded).

**S2 — A5 direction construction (2 GPU jobs, one per model, each model loaded once).** On
CS-Dialogue **D-construct only**, one streaming pass per model records, for **every** layer
and **both** sides, the group-A/group-B `count`, `sum` (→ `mu`), and uncentered second
moment `S = Σ h hᵀ` (float64). Then, on CPU/GPU: eigendecompose `C=S/n` → `U_A,U_B`;
compute principal angles, `s_unique`/`s_shared`, select `i_u`/`i_s`, build
`v_unique`/`v_shared`; freeze signs (§4 of spec); write directions `.npy`, the full §13
diagnostics JSON, `mu_A`/`mu_B`, `n_A`/`n_B`, ranks, and hashes to
`results/basis_a5_atlas/directions_{whisper,qwen}/`. **Enforce `i_s ≠ i_u` and the
orthogonality tolerance; flag failures.**

**S2b — Rank/stability gate (CPU only, construction data).** Recompute `(v_unique,
v_shared)` at `r ∈ {16,32,64}` (capped at `min(n_A,n_B)`) at the A4 anchor/candidate layers;
write `results/basis_a5_atlas/stability/rank_stability.json` with cosine stability. **If
r=32 is clearly unstable (§6 of spec), BLOCK** — do not launch S3/S4.

**S3 — Whisper A5 local atlas (MIG, ≤2 workers).** Decode the **576** Whisper cells
(S1/S2/S3 × E0–E31 + D0–D31 × 3 panels), `scope=oracle_local`, `rho=0.5`. Decoder cells →
**reuse cached baseline encoder outputs**; encoder cells → **fresh encoder forward per
cell**. Whisper loaded once per worker; layer/side/direction-sharded; resumable.

**S4 — Qwen A5 local atlas (MIG, ≤2 workers).** Decode the **468** Qwen cells (S1/S2/S3 ×
E0–E23 + D0–D27 × 3 panels), `scope=oracle_local`, `rho=0.5`, reusing the exact A4 Qwen
Oracle-local masks (CS `cs_ddev_select_alignment_v1`; SEAME frozen target segments) and
cached Qwen baselines for decoder cells; fresh audio-encoder forward for encoder cells.

Concurrency: at any time ≤ 2 GPU jobs (e.g. S3-Whisper ∥ S4-Qwen, or two shards of one).

---

## 3. New decode budget (rho=0.5, oracle_local only)

| Stage | Cells | Encoder-reuse valid? |
|---|---|---|
| S3 Whisper encoder (32×3×3) | 288 | **no** (encoder steering) |
| S3 Whisper decoder (32×3×3) | 288 | yes (decoder-only) |
| S4 Qwen encoder (24×3×3) | 216 | **no** (encoder steering) |
| S4 Qwen decoder (28×3×3) | 252 | yes (decoder-only) |
| **Total new steered cells** | **1044** | |

Reused (no decode): A4 baseline + A4 Raw-local + A4 Conditioning-local comparators. Each
steered cell decodes one panel (CS 300 or SEAME 50). Per-model load amortized across all
its cells; one model instance per worker.

---

## 4. Output layout (`results/basis_a5_atlas/`)

```
spec/               frozen A5 protocol, A4 invariant fingerprints, protocol hash, provenance
reuse/              comparator_manifest.json (A4 baseline/Raw-local/Cond-local pointers)
directions_whisper/ v_unique/v_shared .npy per (side,layer) + diagnostics.json + hashes
directions_qwen/    same, Qwen
stability/          rank_stability.json (r∈{16,32,64} cosine stability at anchor layers)
whisper/{S1,S2,S3}/{enc,dec}/{cs,man,sge}/   S3 cells
qwen/{S1,S2,S3}/{enc,dec}/{cs,man,sge}/       S4 cells
geometry/           within-model A5↔Raw / A5↔Conditioning geometry (CPU)
tables/             basis_a5_atlas.csv + per-model tables
figures/            depth×direction, correction/damage, normalized-depth, cross-corpus
runtime.json        Slurm/GPU/workers/VRAM/throughput/completed/failed
manifests/          per-stage provenance (resolved config, env, git, model, hashes, status)
```

Every cell carries the §10 fairness log and follows AGENTS.md provenance (resolved config,
env, git state, model metadata, hashes, status). Completed cells are keyed and never
re-decoded. Conditioning-Avg is never constructed or decoded.

---

## 5. Provenance + freeze order

The A5 protocol freeze (this doc + spec + A4 invariant fingerprints + A5 protocol hash) is
sealed **before** S2's first GPU result, mirroring A3/A4. SEAME panels and all A4 masks are
reused byte-for-byte with their existing fingerprints. The rank/stability gate (S2b) must be
green before S3/S4. No `D-dev-confirm` / `D-test` is read or produced. Legacy and A4 outputs
are preserved unchanged.
