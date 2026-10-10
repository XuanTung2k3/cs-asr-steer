# P2-SEL pre-run audit (before any P2-SEL outcome)

Contract frozen at `b4602b8` (`P2_SEL_SPEC.md`, `P2_SEL_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel.json`,
`P2_SEL_MINI_PANEL.json`; all byte-unchanged). Implementation `6c7644a`. Machine verdict
`results/inference_cf/p2sel/prerun_audit.json` (CPU, no model inference): **`PASS_TO_P2_SEL_S1`**.

## Start state

Local HEAD = `origin/cs-asr-steer-inf` = `b4602b84f1b5ff44bc1f9afd7b9eca0515d57f79`, clean tree, empty queue.
(An earlier session found only untracked P2-SEL drafts and stopped; the contract was then committed by its
author as `b4602b8`. Nothing from that session was committed or run.)

## Reuse map (hash-verified, no recomputation)

| Artifact | Use | Verification |
|---|---|---|
| `results/inference_cf/p2rj/positions.json` | 180 ordered keys/strata, evaluator Y_ref and c* (analysis only) | file sha256 = config `population_hash` |
| `results/inference_cf/p2dir/construction_population.json` | reference-free key list for the runner | self-hash, order = positions |
| `results/inference_cf/p2dir/exp1_run1/rows/*.npz` | extracted h_B/h_E, sealed D0 and D2 vectors | `directions_sealed.json` hashes; D2 sha256 = row record at 180/180; P2-DIR Exp-1 audit PASS |
| `results/inference_cf/p2dir/exp1_run1/rows/*_logits.npz` (local) | C0 NONE logits | sealed hashes; runtime bitwise identity check |
| `results/inference_cf/p2dir/exp1_run1_analysis.json` | ungated-D2 per-position Δm (ratio denominators) | config anchor hash |
| `results/inference_cf/p2r/run3/rows/*.json` | E, R_B, g = E·R_B (B-branch current gate) | exact unique (utterance, absolute query) join at 180/180, t matches, g = E·R_B (1e-12), no fallback |

GPU work avoided: no extraction, no D2 readout recomputation (0 autograd calls planned), no NONE pass,
no ungated-D2 rerun, no LID/localizer recomputation (R_B is recomputed from live logits as an identity
check; E is joined). New GPU work: C1, C2 (pass A) and C3 (pass B) single pulses only.

## Implementation (minimal, reuse-first)

`experiments/inference_cf_p2sel.py`: C1/C2 use the unchanged P2 deployment hook `inference_cf_cached._edit_hook`
(alpha 2, gain g, phi id, NormPreserve; g = 0 or no direction -> exact no-edit, checked bitwise vs NONE). C3 uses
alpha 2, gain 1, direction (λ/2)·d_readout with λ from the unchanged `inference_cf_p2r.solve_scale` (8 evals) at
the pooled target `sqrt(Σ‖edit_C2‖² / 180)`. Both passes start from the identical unedited B cache (crop + bitwise restore).
`inference_cf_p2sel_analyze.py`: P2-DIR evaluator definitions/bootstrap reused unchanged; frozen family-12
(α = 0.05/12), TOST 90% equivalence intervals, standalone/selective rules, label precedence, pre-committed S2 rule.
`inference_cf_p2sel_audit.py`: independent (imports only P2-DIR audit helpers, never an analysis module).
No change to `readout.py`, `directions.py`, `core_r2.py`, `core_p1.py`, `broad.py`, DG-02 hook/repair, cached decode,
P2-R, P2-DIR code or artifacts, or METHOD_CONTRACT.

Tests: `tests/test_inference_cf_p2sel.py` (11) plus P2-DIR/cached/P2-R suites (85 total, all pass).

## Recorded interpretations (not scientific choices)

1. "Actual post-NormPreserve edit" energy = the DG-02 hook's audited `‖steered − site‖` (same quantity P2-DIR used);
   recorder-state norms are co-reported and audited within 5e-3.
2. "Observed corruption" = the dialogue-weighted point estimate (the P2-DIR convention); TOST equivalence
   bounds are closed intervals.
3. Ungated denominators are recomputed on the identical 180 rows from the audited P2-DIR per-position Δm
   and must equal the fixed +4.499 / −2.011 within 5e-4; ratios use the matched estimates, both reported.
4. S2 runner is not implemented: it is conditional on `P2_SEL_GATE_RESCUES_D2` plus audit PASS; its label rule
   is frozen in code and tested now. If S1 rescues, the S2 runner will be written against the frozen contract,
   tested and pushed before its job.

Pre-outcome exposed fact (already in P2-R run3, not new): g > 0 at 42/60 EN-confusion, 0/60 EN-correct,
5/60 ZH-correct positions.

Firewall: exposed D-dev-select only; no router-calib, D-dev-confirm, D-test, P3 or transfer data. P3 HELD.
