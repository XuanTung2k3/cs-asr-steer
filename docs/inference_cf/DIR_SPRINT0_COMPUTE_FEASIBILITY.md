# DIR-SPRINT0 resource forecast (CPU-only, pre-run)

No scientific inference was run to make this forecast. Its inputs are:
- the measured SRD2-G0 runtimes (same MIG 3g.40gb type, same mechanics);
- the frozen population durations;
- a CPU engineering smoke on public clips (code paths only; CPU timings are not GPU throughput).

## 1. Inputs

| Input | Value |
|---|---|
| SRD2-G0 Job A | 1,017 s for 400 utterances, 4,027 s heard, 11,720 inventory / 10,267 structural queries |
| SRD2-G0 Job B | 925 s for 10,267 structural queries: D2 autograd, clean step, B1 always, B2/B3 when g > 0, restore |
| Historical D2 p95 / pulse p95 | 0.04346 s / 0.01770 s |
| Prospective audio | 2,851 s (2,695 s heard), 240 utterances |
| Bank audio | 5,269 s (4,908 s heard), 300 utterances |
| Expected prospective queries | about 7,840 inventory / 6,870 structural (SRD2 density per heard second) |
| Expected bank queries | about 14,280 |
| Phone provider (CPU, 4 threads, engineering audit) | 0.16 s per audio second |

## 2. Job A

| Component | Estimate |
|---|---|
| SRD2-G0-equivalent capture (B0, teacher replay, full replay, gate, LID, probes) | about 0.087 s/query × 7,840 ≈ 680 s |
| E / TL / NULL branch replays | 3 × 7,840 cached steps ≈ 700 s |
| R0 window-grid LID (≈ 2 windows/s, shared cache) | ≈ 270 s |
| v_AC masked encoders and cold replays | ≈ 150–300 s |
| Bank B0 greedy + full replay | ≈ 450 s |
| Phone children (parallel to the GPU loop) | 456 s + 843 s at the audited CPU rate; ≤ 2,600 s even at 3× slower |
| CPU construct | ≈ 300 s |

- **Typical:** about 2,700 s.
- **Frozen conservative value:** 6,000 s, against the hard limit of 10,800 s.
- A timeout leaves the attempt preserved as INVALID. It never authorizes a third job or a smaller panel.

## 3. Job B (recost after Job A, from the reference-free planned matrix)

**Frozen formula:**
- `0.18 × structural_queries + 0.05 × planned_nonzero_pulses + 800`.
- Planned nonzero pulses count every arm cell with a valid direction (D3 only for `candidate`) and a nonzero target.

**Observed-component forecast:** sum over structural queries of
- 2 × the measured p95 cached-step time c;
- the autograd calls × 0.0434598979 s;
- each planned pulse × (0.0177027043 + 0.02 s solver allowance + 104,768 bytes ÷ measured durable write throughput v);

plus max(800, model load + 240 × p95 encoder + audio I/O + 180 s).

**Expectation.**
- Ungated: about 6.6 planned pulses per structural query. Gated (g > 0 on about 19%): about 0.9.
- That gives roughly 51,000 pulses: formula ≈ 1,240 + 2,580 + 800 ≈ 4,600 s.
- Measured SRD2-G0 mechanics suggest about 3,000 s.

**Authorization rule.** Both forecasts must be ≤ 9,000 s, leaving 30 min below the 3 h ceiling. Otherwise the result is
`DIR_SPRINT0_COMPUTE_BLOCKED`: STOP for a human resource decision. There is no subset, arm reduction or population
change.

## 4. Storage

| Item | Size |
|---|---|
| Job A lossless arrays (3 × 104 KB BF16 logits + heads + sites per query) | about 3.5 GB |
| Phone posteriors (8,120 s × 50 frames × 392 × 4 B) | about 640 MB |
| Bank sites | about 40 MB |
| Job B executed full-vocabulary logits (≈ 51,000 × 104 KB) | about 5.5 GB |

All arrays are archived content-addressed under `/mnt/data/tungnx/cs-asr-steer/archives/dir_sprint0/run1`. At least
50 GB must be free on both volumes (checked by the pre-run audit).

## 5. Limits

- At most two scientific GPU jobs, each ≤ 03:00:00, on H100 MIG 3g.40gb with 12 CPUs and 96 GB.
- No third job, dose or layer search, or selective rerun.
- The deterministic CPU construct stage can be rerun without the GPU only if it raised inside Job A; the failure is
  preserved.
