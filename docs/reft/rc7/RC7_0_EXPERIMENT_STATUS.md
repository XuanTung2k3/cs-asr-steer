# RC7-0 — Experiment status board (as of 2026-10-26)

| Run | Source | Status | Jobs | Recorded GPU-h | Decision / blocker |
|---|---|---|---|---:|---|
| a5000-e1-pilot-v1 | 61a380f | COMPLETE | E-ReFT/LRA on T394/V106 | 0.57 wall-h session | pilot only; C150 not evaluated |
| energy-20261020-v1 | 57851ec | ADMITTED, not run | — | 0.03 | superseded |
| campaign-20261020-v1 | 627e179 | STOPPED_BY_USER at N1 | — | 0.10 | superseded |
| admission-20261020-v1 (CE-only) | 3c8c01e | BLOCKED_MISSING_ARTIFACT | — | 0 | historical role manifests missing |
| **core7h-20261020-v2** | **2ca3c2d** | **CORE COMPLETE; controller stopped** | 34 COMPLETE, 1 SKIPPED (G5_WCE), 1 FAILED (N4), 2 BLOCKED (N5, REPORT) | **19.05** | **N2_NO_GO**; N4 BF16 crash |
| lra-gate-20261021-v1 | e02d2bb | PREPARED_NOT_ADMITTED | — | 0 | superseded |
| **lra-gate-20261021-v2** | **b18dd0c** | **COMPLETE** | L0–L4 + AUDIT | **1.70** (sprint) | **NO_MEANINGFUL_L4_BENEFIT** |
| RC7-0 N4 diagnostic continuation | RC7-0 branch | see `RC7_0_N4_REPAIR_REPORT.md` | D1/D2 only | see report | engineering repair, no training |

Stage-level status of core7h-20261020-v2:

| Stage | Content | Status |
|---|---|---|
| B0 | frozen baseline, V300 | COMPLETE |
| N1 | CE R/L/LP/LS × 2 LR × 3 epochs | COMPLETE |
| N2 | scalar G1; gates G3/G4/G5; continuation G8 (LoReFT parent) | COMPLETE → N2_NO_GO |
| N3 | WCE ω=2 for R/L/LP/LS | COMPLETE; G5_WCE correctly SKIPPED |
| N4 | (NO_GO branch) D1 lexical decomposition + D2 controlled readout on T2000[:64] / V300[:64] | FAILED (BF16) |
| N5 | seeds 43/44 + C500 confirmation | BLOCKED; **not authorized**; must not auto-resume |
| REPORT | formal report | BLOCKED; core reports exist |

No job is active on A5000 (verified 2026-10-26: GPUs idle, no tmux server).
