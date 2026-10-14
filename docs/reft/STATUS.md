# CS-ASR-ReFT status

| Stage | State | Evidence (local artifacts under `/mnt/data/tungnx/cs-asr-reft/`) |
|---|---|---|
| S0 | done | branch `cs-asr-reft` from `main` `db365d8f`; scope and protocol frozen (`freeze/p0.json`) |
| P0 | **`P0_PASS_RELEVANT_OPPORTUNITIES`** (unchanged by AMENDMENT L31) | `p0/evaluation.json`. Attempt 1 was BLOCKED by a hash-convention defect and is kept as a superseded record (amendment r1). |
| P1 @ layer 24 (original) | **historical, incomplete**: capture (59135) and basis (59136) built; held-out evaluation never completed (59137 crashed before writing) | `p1/` (untouched; no feasibility outcome exists) |
| P1 @ layer 31 (AMENDMENT L31, active) | **`P1_SUBSPACE_WEAK_OR_NARROW`**: all gates pass except preservation (worst-of-8 flip 0.136 > 0.10) | `p1/l31/evaluation.json`; basis sha256 `f020e93b…`; see `P0P1_RESULT.md` |
| P2 | not started; not authorized. The frozen verdict forbids automatic controller training. Recommended first: CPU selectivity audit (`P0P1_RESULT.md` §5) | — |
| P1-D conditional controllability audit (exploratory, CPU only; does not replace the P1 verdict) | **`P1D_NO_CONVINCING_SIGNAL`**: D1 error-vs-correct AUROC 0.564 (random 0.44–0.62, position-only 0.546); D2 direction cos 0.135 vs constant 0.110 (CI of difference [−0.06, +0.09]); full-state readout AUROC 0.752 / cos 0.212 | `p1d/evaluation.json`; `P1D_PROTOCOL.md`, `P1D_RESULT.md` |
| Next | Projected-only 20-parameter controller **not recommended**. Owner decision pending on a re-scoped, pre-registered decoupled read/write P2 (read from Norm(h), write through frozen V₃₁). | — |
| P1-E decoupled read U / write V (exploratory, CPU only; audit dialogues previously examined) | **`P1E_WHEN_ONLY`**: U (rank 4 from construction-only full-state readers) gives WHEN AUROC 0.757 (V 0.564, random ≤ 0.62, nuisance 0.546); HOW cos 0.168 vs constant 0.110 (CI [−0.02, +0.15]); joint 20-parameter head selects 7/45 errors and 15/199 correct, net first-order benefit CI includes 0, does not beat random-read heads. Construction CV for U-probes leaked (disclosed). | `p1e/evaluation.json`; `P1E_PROTOCOL.md`, `P1E_RESULT.md` |
| Next (after P1-E) | **Frozen-subspace tiny-controller line at layer 31 stopped** (V-read and U-read/V-write). Owner decision pending on a conventionally trained, pre-registered adaptation baseline (editor / LoReFT / LoRA) under Auto beam-5. | — |
