# P1 pre-run independent audit

**Verdict: `PASS_TO_P1_RUN`.**

**Audited:** P1 spec v1 (`113c48e`), implementation `63b23c3` plus F3 hardening `ae9363d`, the R2
entry contract (`R2_CF_FEASIBLE_NOT_PREFERRED`, `POST_R2_AUDIT: PASS`, gate `g_old`, freeze
`4a2f606`), and the tests.

**State at audit:** no P1 GPU outcome exists. The alpha and the population were frozen in the
spec before implementation.

| Check | Finding |
|---|---|
| **Direction math and sign** | `d = (hE − hB)/(‖hE − hB‖ + 1e-6)` from the L24 site of unsteered cE and cB replays of the identical prefix. Unit tests cover unit norm, sign E−M (edit has positive projection on `d` at the site) and the tiny/nonfinite fallback. |
| **Same-prefix construction** | Inputs are exactly `prompt + emitted prefix` for all three forwards (logged per step). The prompts differ only at the language index. `hB = hM` holds by the forced-zh identity; the manifest asserts `cB == cM`. |
| **Hook semantics** | Canonical `DecoderPostCrossAttnInterventionHook` at decoder layer 24, `r = q + u_source` (post-cross-attention, pre-FFN), returning `u_source + (r' − r)`. A tiny 25-layer Whisper test confirms: layer 23 unchanged, other positions' layer-24 sites unchanged, edited position changed, earlier-query logits unchanged. |
| **NormPreserve** | Via `apply_steering`: `h' = h̃·‖h‖/(‖h̃‖+1e-6)`. Tested (norm equal to rel 1e-5 in fp32) and matched to the float64 reference edit. |
| **Zero-dose identity** | α = 0 and g = 0 are bitwise identities (tests). A real-model CPU smoke shows F3 bitwise equal to F1 at every step, and the α = 0 decode equals the R2 `generate()` tokens. F3 now uses the same forward signature as F1. |
| **Forced prefix** | Positions < 4 are never edited, so `t = 0` is blocked (test and smoke). Documented limitation. |
| **Frozen R2 gate** | `g = E·R_B` computed through unchanged `core_r2` providers (window, LS-B, BC-B, partition hash, fallbacks). The Ecf conflict branch is not used. The test compares against `gate_values(...)["g_old"]`. |
| **Branch/cache isolation** | Every forward is a separate `use_cache=False` replay. `assert_no_site_hooks` runs before F1, after F1/F2 and after F3 at every step. The hook is installed only inside F3's context. |
| **History edits** | Stored edits are re-applied at their own positions each step. The layer-24 site at τ depends only on layers ≤ 24 at positions ≤ τ, which a layer-24 edit cannot change, so this equals KV-cached steering. |
| **Decoder lineage** | Greedy, beam 1, `max_new_tokens = 200`, suppression identical to `generate()`, stops on EOS or cap. |
| **Reference-free** | `decode()` takes audio, encoder output, prompts, partition, null odds and the dose only. The panel holds audio identity only. The R2 `generate()` tokens are read only by the CPU evaluator, for a diagnostic. |
| **Accounting and serialization** | Every step is recorded with gate components, fallback, direction status and norm, hook audit, reference edit, unsteered vs steered token and prefix integrity. The shard is bound to the manifest. The evaluator implements A1–A13; a missing record blocks (tested). |
| **No tuning** | Layer 24 and α ∈ {0, 1.0} are fixed in the config. No sweep code exists. α = 1.0 is justified from P0 site norms, not recognition. |
| **Tests** | 12 P1 + 45 R2/P0/P0-R1 + 36 existing site/hook tests pass. |

**Residual risks (accepted, measured):**

- A4 needs at least one `g > 0` step at `t ≥ 1` in the frozen 10 utterances. Failure would be a
  genuine `P1_BLOCKED`, not a reason to change the panel.
- GPU/bf16 determinism for A1 is expected but empirical.

**PASS_TO_P1_RUN**

## Re-audit for v1.1 (before rerun) — `PASS_TO_P1_RUN`

- **What was checked:** the attempt-1 diagnosis. A6 failed only on 2 of 20 edits with realized
  norms ≤ 0.064, below or near bf16 site resolution. The sign held on all 20, and the other 18
  matched the float64 reference within 2.5%.
- **The fix is instrument-only:**
  - it adds a site-precision replica using the same `apply_steering` on the recorded site;
  - the tolerance is unchanged;
  - no intervention, gate, direction, site, α or population change.
- **Tests:** a new test reproduces the defect in bf16 (the float64 comparison fails, the replica
  is exact). A tiny-model decode confirms the hook-realized edit equals the replica. All 60
  inference-CF tests pass.
- **Provenance:** attempt 1 artifacts are preserved and unmodified. The new output directory is
  versioned, the manifest cites the superseded attempt, and the prepare script hashes the defect
  report.

**PASS_TO_P1_RUN (v1.1)**
