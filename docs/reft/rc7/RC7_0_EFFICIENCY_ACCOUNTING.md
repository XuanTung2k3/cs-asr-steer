# RC7-0 — Efficiency and Pareto reproducibility audit

No benchmark was rerun. Values are archived measurements (core7h-20261020-v2 `reports/development_metrics.csv`,
`status.json`; lra-gate-20261021-v2 `reports/combined.csv`, `accounting.json`). Machine-readable availability:
`RC7_0_EFFICIENCY_AVAILABILITY.json`. Hardware: RTX A5000 24 GB, BF16 backbone, FP32 adapters.

## 1. Measured per-method values (V300, seed 42)

| Method | Trainable params | Checkpoint bytes | Attributed training GPU-h¹ | Peak reserved (B)² | V300 decode s | RTF³ | Loops / nonterm. |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 frozen | 0 | — | — | — | 531.8 | 0.139 | 1 / 1 |
| DiReFT CE (L) | 5,122 | 22,588 | 1.677 | 4,945,084,416 | 535.9 | 0.140 | 0 / 0 |
| LoReFT CE (R) | 5,122 | **6,576,571** | 1.668 | 5,131,730,944 | 543.9 | 0.142 | 1 / 1 |
| LoRA-PM CE (LP) | 5,120 | 22,907 | 2.369 | 5,033,164,800 | 543.3 | 0.142 | 0 / 0 |
| LoRA-Strong CE (LS) | 1,310,720 | 5,281,809 | 2.481 | 5,064,622,080 | 561.5 | 0.147 | 0 / 1 |
| DiReFT WCE / LoReFT WCE / LP WCE / LS WCE | as above | as above | 1.652 / 1.589 / 2.390 / 2.434 | — | 532.7 / 503.5 / 542.9 / 550.5 | 0.131–0.144 | 0 / 0 |
| Gated LoReFT G3/G4/G5 | +289 (5,411 total) | 6,579,701 | 1.95–1.97 (incl. parent) | 4.36 GB | 530–536 | 0.139–0.140 | — |
| Gated DiReFT L2/L3/L4 | +289 (5,411 total) | 36,533 | own 0.264–0.268 allocated; attributed total 1.94–1.95 | 4,299,161,600 | 514–520 | 0.134–0.136 | 0 / 0 |
| Ungated continuation L0 | 5,122 | 33,338 | own 0.249; attributed 1.926 | 4,286,578,688 | 522.6 | 0.137 | 0 / 0 |

¹ "Attributed" includes both LR trials, normalization, validation decodes, failed work and warm start. The parent and
screening costs are shared. **Never sum rows**: gated rows already contain the parent's 1.68 GPU-h.
² The scope of `peak_reserved_bytes` in `development_metrics.csv` is not documented per row (training vs validation
decode). REQUIRES_VERIFICATION before it is quoted as training memory.
³ Model-inference RTF covers frozen encoder + frozen Auto-LID + adaptation + beam-5 search; audio I/O and preprocessing
are excluded. The per-method RTF differences (≤ 0.008) are within decode-length variation and are not adapter cost.

Actual campaign consumption, not attribution: core7h 19.05 recorded GPU-h (sum of per-job elapsed 17.56 h; the rest
is admission/preflight/controller residency); sprint 1.70 GPU-h; RC7-0 N4 continuation 0.022 GPU-h.

## 2. Audit issues

1. **LoReFT checkpoint-size anomaly: explained, not resolved.** `R` checkpoints contain
   `reft.rot.parametrizations.weight.0.base` [1280 × 1280] fp32 = 6,553,600 B, the base buffer of
   `torch.nn.utils.parametrizations.orthogonal`. Trainable tensors are `original` [2 × 1280], `learned.weight`
   [2 × 1280] and `learned.bias` [2] (5,122 total). DiReFT stores only A, B, c. The 291× byte ratio at equal
   parameter count is therefore serialization, not capacity. A minimal deployable LoReFT needs only the
   materialized R (2 × 1280), W and b. That export is **not implemented and not measured**.
2. **Minimal deployable adapter bytes:** MISSING as a measurement. The arithmetic lower bound for 5,122 fp32 values
   is 20,488 B. LoRA-Strong is 5.24 MB of raw fp32.
3. **Cached vs uncached training:** ReFT and gate arms train on a cached frozen encoder output. Weight-LoRA arms
   recompute and backpropagate through the encoder. Their memory and GPU-hours are not like-for-like, and the
   cache build cost belongs to the ReFT pipeline.
4. **Encoder-cache preparation cost:** not separately timed. The B0 bootstrap job (953.8 s) builds 2,300 cache
   entries *and* decodes V300 (531.8 s), so the cache build is ≈ 422 s by subtraction. This is derived, not measured.
5. **Parameter count ≠ memory/latency:** peak memory is dominated by the frozen 1.55 B-parameter backbone. The
   5K-vs-1.3M parameter gap shows ≤ 3 % difference in archived peaks, and decode time is set by beam search.
6. **Failures/recoveries:** core run had 1 FAILED job (N4) and one bounded automatic recovery after a
   disk-monitor `FileNotFoundError` (CE_LP_lr0 attempts = 2). Its cost is inside the attributed figures.

## 3. What a manuscript Pareto needs that does not exist yet

- Standardized peak *training* memory per method, separated from decode.
- Minimal-export adapter bytes, including LoReFT without the base buffer.
- Adapter-only latency (encoder-output edit cost in isolation) beside the full-pipeline RTF.
- Multi-seed cost variance. Cache build timed separately.

These are proposals for RC7-4/RC7-7. None has been executed.
