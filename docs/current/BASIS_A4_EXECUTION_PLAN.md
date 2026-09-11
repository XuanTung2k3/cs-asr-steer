# BASIS-A4 — Execution Plan (Compute / Slurm)

Status: **FROZEN PLAN — no GPU jobs launched by this stage.** Companion to
`BASIS_A4_SPEC.md`. Base state BASIS-A3 `259f3b4`.

This plan is written so the full atlas can be executed later under the standard
constraints; it is **not** run now (reconciliation + freeze only).

---

## 0. Standing constraints (from CLAUDE.md / AGENTS.md)

- All GPU work via `sbatch`; **≤ 2 simultaneous GPU jobs**.
- Prefer **MIG** if measured VRAM ≤ 40 GB; request a full/main GPU only if actually required.
- **Not** one job per layer. Load each model **once per worker/shard**; steer many
  layers/cells per model load.
- Reuse baseline **encoder** outputs for **decoder-only** steering where valid; **do not**
  reuse baseline encoder output for **encoder** steering.
- Results resumable; completed cells never rerun unnecessarily.
- Free decoding, greedy, temperature 0, beam 1, `condition_on_prev_tokens=false`, batch 1
  per worker, `max_new_tokens=200` (both models). Whisper `language=zh`; Qwen
  `force_language="Chinese"`, `context=""` (BASIS_A4_SPEC §7).

---

## 1. VRAM sizing (determines MIG vs main)

- **Whisper-large-v3** (~1.5 B, bf16) + batch-1 decode: well under 40 GB → **MIG** (as in
  A3: one MIG GPU, two batch-1 workers).
- **Qwen3-ASR-1.7B** (~1.7 B thinker, bf16) + batch-1 decode: expected ≤ 40 GB → **MIG**
  by default. **Preflight gate:** a bounded 1-utterance worker preflight measures peak VRAM;
  if measured > 40 GB, escalate that model's decode to a full GPU (still ≤ 2 concurrent
  jobs). Recorded in `runtime.json`.

---

## 2. Stages (ordered; each resumable; ≤ 2 concurrent GPU jobs)

**S0 — Protocol freeze (this stage, CPU only).** Seal `BASIS_A4_SPEC.md` +
`BASIS_A4_EXECUTION_PLAN.md`, panel fingerprints (reused from A3), reuse-acceptance
hashes, git/commit provenance under `results/basis_a4_atlas/spec/`. No GPU.

**S1 — Reuse audit (CPU only).** Verify Whisper Raw R1 rho=0.5 (384 cells) and Whisper
Conditioning D24/D26/D27/D31 rho=0.5 (24 cells) against the A4 freeze (protocol hash,
direction hash, panel fingerprint, model revision, site, dose, NormPreserve). Emit
`results/basis_a4_atlas/reuse/reuse_manifest.json` listing accepted-reuse vs must-decode.
`Conditioning-Avg` never appears.

**S2 — Qwen direction construction (1 GPU job, Qwen loaded once).** On CS-Dialogue
D-construct only: build Raw audio-enc E0–E23, Raw text-dec D0–D27, Conditioning text-dec
D0–D27 (BASIS_A4_SPEC §3/§5/§7), plus per-layer scale `s_l`, measured audio-tower fps, and
vector hashes. Also record Qwen encoder Oracle-local span→frame mapping. One recorder pass
per contrast type. Writes `results/basis_a4_atlas/directions_qwen/`.

**S3 — Whisper Conditioning missing layers (MIG, 2 workers).** Decode the **168** missing
Whisper Conditioning cells (D0–D23, D25, D28, D29, D30 × 2 scopes × 3 datasets) at
rho=0.5. Decoder-only steering → **reuse cached baseline encoder outputs**. Layer-sharded,
resumable; Whisper loaded once per worker. No Raw re-decode.

**S4 — Qwen baselines (same job family as S5, Qwen loaded once).** Unsteered free-decode of
the 3 panels under the frozen baseline policy (§7). Persist decoder baseline + **cache
encoder outputs** for decoder-only reuse in S5.

**S5 — Qwen atlas decode (MIG or main per S1 preflight, 2 workers max).**
- **Encoder Raw** (E0–E23 × 2 scopes × 3 sets = 144 cells): encoder steering → **fresh
  encoder forward per cell** (no baseline-encoder reuse).
- **Text-decoder Raw + Conditioning** ((28+28) × 2 scopes × 3 sets = 336 cells):
  decoder-only → **reuse cached baseline encoder outputs** from S4.
- Sharded by (direction, layer-band); Qwen loaded once per worker; resumable by
  (utterance, layer, direction, scope, rho) key.

Concurrency: at any time ≤ 2 GPU jobs (e.g. S3-Whisper ∥ S5-Qwen, or two Qwen shards).

---

## 3. New decode budget (rho=0.5)

| Stage | Cells | Encoder-reuse valid? |
|---|---|---|
| S3 Whisper Conditioning | 168 | yes (decoder-only) |
| S5 Qwen encoder Raw | 144 | **no** (encoder steering) |
| S5 Qwen decoder Raw+Cond | 336 | yes (decoder-only) |
| **Total new steered cells** | **648** | |

Reused (no decode): Whisper Raw 384 + Whisper Conditioning 24 = 408 cells. Redundancy gate
removes 360 would-be Conditioning-Avg cells.

Each steered cell decodes one panel (CS 300 or SEAME 50). Per-model load amortized across
all its cells; per-worker one model instance.

---

## 4. Output layout (`results/basis_a4_atlas/`)

```
spec/           frozen A4 protocol, panel fingerprints, reuse-acceptance hashes, provenance
reuse/          reuse_manifest.json (accepted A3 cells vs must-decode)
directions_qwen/  Qwen Raw/Conditioning vectors, s_l, fps, span→frame map, hashes
whisper/conditioning/{cs,man,sge}/  S3 new cells (+ pointers to reused A3 cells)
qwen/baseline/{cs,man,sge}/         S4 baselines + cached encoder outputs
qwen/{raw,conditioning}/{cs,man,sge}/  S5 cells
geometry/       within-model Raw↔Conditioning geometry (CPU), per model
runtime.json    Slurm/GPU/workers/VRAM/throughput/completed/failed
```

Every cell carries the §9 fairness log (orig norm, mean pert norm, total energy, relative
pert, edited count, edited fraction). Manifests follow AGENTS.md provenance
(resolved config, env, git, model metadata, hashes, status). Completed cells are keyed and
never re-decoded.

---

## 5. Provenance + freeze order

The A4 protocol freeze (this doc + spec + fingerprints + reuse hashes) is sealed **before**
S2's first GPU result, mirroring A3. SEAME panels are reused byte-for-byte from A3 with
their existing fingerprints. Conditioning-Avg is never constructed or decoded.
