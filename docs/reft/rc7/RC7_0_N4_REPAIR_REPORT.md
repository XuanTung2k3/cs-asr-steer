# RC7-0C — N4 BF16 diagnostic repair (continuation record)

**Result: `N4_REPAIRED_DIAGNOSTICS_COMPLETE`** as a new continuation record. The archived failure
(`core7h-20261020-v2/jobs/N4/result.json`, `logs/N4.log`) is preserved unchanged and still reads FAILED.
N5 remains **BLOCKED**: it is the multi-seed + C500 package and is not authorized in RC7-0.

## 1. Failure site (from the archived log, reproduced)

```
reft/a5000_campaign/worker.py:431  n4 → controlled_readout(...)
reft/a5000/diagnostics.py:38       controlled_readout → fit_probe(train_base, train_y)
reft/a5000/diagnostics.py:23       states.detach().cpu().numpy()
TypeError: Got unsupported ScalarType BFloat16
```

Under the frozen N2_NO_GO branch N4 trains nothing. It teacher-forces A5000_T2000[:64] and A5000_V300[:64] through
frozen Whisper and the selected LoReFT CE adapter (`N1_selection.json → R`, sha256 `a4719921…24c8`), collects
decoder last-layer states at predictive positions (BF16, the model dtype), computes D1 and fits D2 probes.
A second identical hazard existed at `diagnostics.py:44` (held-out scoring). Hidden states were never persisted,
so a bounded GPU rerun was required. The original N4 attempt ran 51 s before failing.

## 2. Repair (commit `500eaed`)

`reft/a5000/diagnostics.py` gains `probe_array()`, now used at both NumPy boundaries:

- floating tensors → `detach().to(torch.float32).cpu().numpy()`. This is exact for BF16/FP16, and shape and row order are unchanged;
- integer label tensors are not cast;
- non-finite states raise `ValueError`, which N4 already maps to a recorded `BLOCKED` readout (previously sklearn
  would have raised a less specific `ValueError` later).

Nothing else changed. `decompose`, decoding, scoring, adapters, gates and every training path are untouched;
`reft.a5000.diagnostics` is imported only by `worker.n4`. Historical predictions and scores are therefore unaffected
by construction, and no historical artifact was re-scored.

## 3. Regression tests (`tests/reft/a5000/test_diagnostics_bf16.py`, 12 cases)

| Run (A5000, torch 2.7.1+cu126) | Tests | Failures | Skipped | junit sha256 |
|---|---:|---:|---:|---|
| New tests against **unrepaired** b18dd0c code | 12 | **9** (archived `TypeError` at `diagnostics.py:23`) | 1 (CUDA hidden) | `4e40133a…` |
| Full A5000 CPU suite at `500eaed` (rc7 + a5000_lra_sprint + a5000_campaign + a5000 + a5000_followup) | 133 | 0 | 1 (CUDA hidden) | `790bf292…` |
| Diagnostics tests with GPU 0 visible (CUDA BF16 states) | 12 | 0 | 0 | `e644e4ec…` |
| Baseline suite at b18dd0c, before RC7-0 | 115 | 0 | 0 | `95f93b56…` |

Covered: root cause; exact cast and shape; integer passthrough; BF16 probe ≡ probe on its FP32 cast (identical
coefficients); `controlled_readout` BF16 ≡ FP32 (identical accuracies); masked `-100` labels; zero-energy
states; NaN/Inf → ValueError; empty/single-class → existing ValueError; dialogue-overlap guard; CUDA tensors.
`tests/reft/rc7/test_n4_continuation.py` covers the continuation guards (separate root, immutability,
NO_GO-only, confirmation-package refusal, 2 GPU-h ceiling, recorded overrides).

## 4. Continuation run

| Field | Value |
|---|---|
| Continuation ID / root | `n4-continuation-20261026-v1` under `a5000_experiments/rc7_0/` (sibling of, not inside, the archived run) |
| Archived campaign / source | `core7h-20261020-v2` / `2ca3c2da…98` (plan sha256 `c0f56c6f…`, config `09ca42e5…`, data `fc170e10…`) |
| Executing source | `500eaedcfafabbcf6b1fbcfa88cb84072b748381`, clean tree, delivered to A5000 by git bundle (sha256 `ef89a163…`) |
| Inputs copied and hash-verified | data_manifest, N1/N2 selections, N2_decision, sealed B0 directory (`SEALED.json` `b9757697…`) |
| Reused read-only | archived encoder cache (fingerprint `ba32b258…`, 2,300 T+V entries), LoReFT CE checkpoint |
| Engineering-only plan overrides | `created_epoch` → now; `campaign_root` → continuation root; `resources.gpu_hours` 72 → 1, `reserve_gpu_hours` 3 → 0, `wall_hours` 120 → 2. No scientific field changed. `source_commit`/`data_sha256` keep the archived binding because they key the cache; `environment.json` in the continuation root therefore shows the archived `source_commit`, and `continuation.json` records the executing commit. |
| Executed code | unchanged `reft.a5000_campaign.worker.n4` |
| GPU / time | A5000 GPU 0 (campaign gpu0 lock), 2026-10-26 11:37:39–11:38:58 UTC, **79.7 s resident = 0.0221 allocated GPU-h** |
| Memory | peak allocated 3.26 GB, peak reserved 3.62 GB |
| Backbone hash after run | unchanged |
| Archived root fingerprint (14 files + `N4_analysis.json` absence + B0 count, hash and mtime) | **identical before and after** |
| Outputs | `N4_analysis.json` sha256 `843995bf…2a99`; `continuation.json` `551a4f7f…41e3`; log `c1bae7d9…5223` (copies in `evidence/`) |

## 5. Diagnostic outputs (exploratory, proxy, not recognition causality)

D1: LoReFT CE adapter, teacher-forced, mean over utterances of per-token NLL decomposed with the
Han/Latin/OTHER tokenizer-script proxy (gold = group + within-group, exact):

| Panel | n | gold NLL | group NLL | within-group NLL |
|---|---:|---:|---:|---:|
| T2000[:64] (training data) | 64 | 0.4027 | 0.0956 | 0.3071 |
| V300[:64] (exposed development) | 64 | 0.2623 | 0.0456 | 0.2168 |

Under this proxy the adapted model's remaining loss is mostly within-script (lexical), not script choice.
No frozen-model D1 was recorded by the archived recipe, so no adapted-minus-frozen contrast is available from
N4 as designed.

D2: held-out script-group readout from decoder last-layer predictive states (probes fit on T2000[:64],
evaluated on V300[:64], dialogue-disjoint):

| Probe | Held-out accuracy |
|---|---:|
| frozen states, frozen probe | 0.9819 |
| fixed frozen probe on adapted states | 0.9870 |
| probe refitted on adapted states | 0.9930 |
| prefix-only (zero encoder memory) | 0.8660 |

Script identity of the next token is already linearly readable from frozen decoder states (98.2 %) and from the
gold prefix alone (86.6 %). The adapter's +0.5–1.1 pp readout change is small, on 64 development utterances,
one seed. This is a readout proxy, not evidence that the adapter works through language identity.

D3: `BLOCKED` by the archived design ("No frozen matched-random causal intervention recipe admitted").

## 6. Status changes

| Item | Before | After RC7-0 |
|---|---|---|
| N4 (archived record) | FAILED | FAILED (unchanged, immutable) |
| N4 diagnostics (continuation) | — | COMPLETE, EXPLORATORY_ONLY |
| N5 | BLOCKED | BLOCKED — dependency resolved, but N5 requires explicit authorization (seeds 43/44 training and C500) and must not auto-resume |
| N2_NO_GO | — | unchanged |
