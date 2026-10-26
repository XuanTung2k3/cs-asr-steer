# RC7-0 Final Report — Evidence audit, engineering repair, novelty check, canonical source

Date 2026-10-26. Branch `research/rc7-0-20261026`, repository `XuanTung2k3/cs-asr-steer` (public).

## A. Executive scientific decision

**`RC7_0_COMPLETE_WITH_RC7_2_NO_DISTINCT_HYPOTHESIS`**

The proposed RC7 controller classifies as `PARTIAL_OVERLAP_ABLATION_ONLY`:
- Its allocation operator and defaults are identical to the historical gates.
- All but one of its feature families already exist in G5 or L4.
- The remaining element, a causal EMA predictor residual, generalizes an existing lag-1 contrast.

RC7-2 is therefore not admitted as a novel-method pilot. This decision grants no permission to launch anything.
Carry-forward blockers that do not prevent RC7-1 are listed in §C. The main one is that C500 is partially exposed.

## B. Canonical repository

| Item | Value |
|---|---|
| Canonical base | `b18dd0c6840191f218f2f45d79499f9b80923d4e` (A5000 sprint snapshot) |
| Canonical branch | `research/rc7-0-20261026` (base + RC7-0 commits) |
| Local worktree | `~/Code/cs-asr-reft-rc7-0`; main checkout `~/Code/cs-asr-reft` untouched |
| A5000 | all 7 snapshots clean and tree-identical to local commits; RC7-0 clone `~/tung.nx/rc7-0-work/repo-500eaed` |
| GitHub | branch pushed after every milestone; no force-push; `main` and other branches untouched |
| Reconciliation | no file import needed; original ancestry preserved (`RC7_0_SOURCE_RECONCILIATION.md`) |

## C. Historical evidence status

- **Reusable (development scope, seed 42, V300 selection-biased):** B0; N1 CE and N3 WCE for LoReFT, DiReFT,
  LoRA-PM and LoRA-Strong. Revalidated: 187/187 output seals and 12/12 selected checkpoints.
- **Exploratory only:** N2 gates G3/G4/G5/G8/G1 (`N2_NO_GO`, unchanged) and sprint L0–L4 (`NO_MEANINGFUL_L4_BENEFIT`, unchanged).
- **Missing:** H100 P2-B/P2-E/P2-F1/P3 source and artifacts; historical Train394/Inner106/Full500/Dev150 manifests.
- **N4:** the archived record stays FAILED. Repaired continuation `n4-continuation-20261026-v1` is **COMPLETE**
  (D1/D2; D3 blocked by design).
- **N5:** **BLOCKED.** It needs explicit authorization (seeds 43/44 training plus C500) and must not auto-resume.
- **Holdout exposure:**
  - No A5000 adapter was ever fitted, selected, decoded or scored on C500.
  - **41/500 C500 utterances (8.2 %, 10/14 dialogues)** appear in steering-era artifacts: 30 decoded and scored,
    11 in a steering training list → `C500_PARTIAL_HISTORICAL_EXPOSURE_VERIFIED`.
  - H100 role mapping is `UNRESOLVED`.
  - The official test is disjoint from every A5000 role.
  - An owner decision is required before RC7-5.

## D. Scientific novelty decision

- **Genuinely different:**
  - a causal multi-block EMA predictor residual (ρ = 0.8);
  - a causal receptive field longer than one neighbouring block;
  - an evaluation-time dose knob c.
- **Already tested:**
  - the allocation operator `energy_coefficients` (ε 0.25, α 1, 1 s blocks);
  - state, neighbour-contrast, edit-norm, alignment and update-variation features;
  - frozen-parent training and the DiReFT CE parent;
  - the 289-parameter MLP.
  
  Two feature-family swaps (G5−G4 −0.054 pp, L4−L3 −0.011 pp) and both gate-versus-continuation contrasts stayed far
  below 0.5 pp. Learned allocations were near uniform (std 0.008–0.014).
- **Unsupported / unspecified:**
  - the predicted quantity s_b;
  - v_train and δ;
  - the feature budget;
  - predictor initialization;
  - the optimizer budget;
  - whether c is swept.
- **Prior art (verified at abstract level):** ReFT (NeurIPS 2024), Grft (ACL 2025), TARE (ACL 2026),
  GateRA (AAAI 2026). Conditional and token/time-specific representation editing is not novel per se.
- **RC7-2 pilot justified?** Not as a method contribution. At most it is an explicitly labelled L4 feature ablation,
  run only after its blockers are resolved.

## E. Engineering and tests (all on A5000, `/home/ubuntu/tung.nx/reft-a5000`, torch 2.7.1+cu126)

```
# baseline, b18dd0c, CUDA hidden
pytest tests/reft/a5000_lra_sprint tests/reft/a5000_campaign tests/reft/a5000 tests/reft/a5000_followup
  → 115 passed, 0 failed, 0 skipped
# new regression tests against unrepaired code
pytest tests/reft/a5000/test_diagnostics_bf16.py  → 12 tests: 9 FAILED (TypeError: Got unsupported ScalarType BFloat16 at diagnostics.py:23), 2 passed, 1 skipped
# repaired, clean checkout of 500eaed, CUDA hidden
pytest tests/reft/rc7 tests/reft/a5000_lra_sprint tests/reft/a5000_campaign tests/reft/a5000 tests/reft/a5000_followup
  → 133 tests, 0 failed, 1 skipped (CUDA test; GPU hidden)
# repaired, GPU 0 visible
pytest tests/reft/a5000/test_diagnostics_bf16.py  → 12 passed
# continuation
python -m reft.rc7.n4_continuation prepare|run --root a5000_experiments/rc7_0/n4-continuation-20261026-v1 --gpu 0
  → COMPLETE, 79.7 s, backbone unchanged, archived root fingerprint unchanged
```

The skipped test is the CUDA-only case in the CPU run, and it passed in the GPU-visible run. The local Mac has no
PyTorch, so no tests ran on the Mac. JUnit files and their hashes are in `evidence/`.

## F. Resource usage

| Item | Value |
|---|---|
| Additional GPU jobs | 1 (N4 diagnostic continuation, GPU 0) |
| Additional allocated GPU-hours | **0.0221** (79.7 s); CUDA-visible unit test ≈ 1 s |
| Training launched | none |
| C500 / official test accessed | no (metadata/ID-level audit only) |
| Active jobs disturbed | none (all GPUs idle before and after) |

## G. Next recommended action

**Option 3: skip controller expansion and prioritize compact-adaptation and functional analysis.** Begin with RC7-1
baseline verification (ready, ≤ 2 GPU-h, development scope). Then:
- add a C500-free seed-replication job;
- run RC7-3 functional analysis from the sealed predictions;
- in parallel, have the owner decide C500 eligibility.

Do not relaunch N0–N5.

## H. GitHub delivery

| Milestone | Commit | Pushed |
|---|---|---|
| 1 Source reconciliation | `1d78cf0` | yes |
| 2 Evidence + exposure audit | `3c62e06` | yes |
| 3a N4 BF16 repair (code + tests) | `500eaed` | yes |
| 3b N4 continuation record | `406d98b` | yes |
| 4 Gate novelty audit | `3196c0a` | yes |
| 5 Efficiency, readiness, final report | the commit containing this file | yes (SHA in the handoff message / `git log`) |

**Fallback transfers:**
- A5000 → GitHub push is unavailable (no credentials), so all pushes were made from the Mac.
- A GitHub clone on A5000 failed (GnuTLS recv error). Code reached A5000 by git bundle `b18dd0c..500eaed`
  (sha256 `ef89a163…`), with ancestry preserved.
- A tar-over-SSH copy was used only for pre-commit testing in a scratch clone.

No outstanding push failures.

## Document index

`RC7_0_SOURCE_RECONCILIATION.md` · `RC7_0_EVIDENCE_LEDGER.{md,json}` · `RC7_0_EXPERIMENT_STATUS.md` ·
`RC7_0_DATA_EXPOSURE_AUDIT.md` · `RC7_0_N4_REPAIR_REPORT.md` · `RC7_0_N4_REPAIR_VALIDATION.json` ·
`RC7_0_GATE_IMPLEMENTATION_COMPARISON.md` · `RC7_0_NOVELTY_DECISION.json` · `RC7_0_RELATED_WORK_BOUNDARY.md` ·
`RC7_0_EFFICIENCY_ACCOUNTING.md` · `RC7_0_EFFICIENCY_AVAILABILITY.json` · `RC7_0_NEXT_PHASE_READINESS.md` ·
`evidence/` · tools `scripts/rc7/rc7_0_audit.py`, `reft/rc7/n4_continuation.py`
