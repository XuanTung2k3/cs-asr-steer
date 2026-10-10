# P1 causal-acceptance specification — v1.1 (v1 frozen before any P1 GPU outcome; §9 instrument amendment)

**Question.** Can the frozen Whisper model perform a correctly localized, same-prefix
counterfactual hidden-state intervention, driven by the R2-selected gate, with correct
intervention, cache and decoding semantics?

**Out of scope for P1 (P2/P3):** which layer, which alpha, best MER/PIER, all-layer editing,
baselines/ablations, generalization.

**P1 acceptance** means the intervention pipeline is computationally and causally valid. It does
not mean ASR improves.

## 1. Frozen inputs from R2 (immutable in P1)

- **R2 outcome:** `R2_CF_FEASIBLE_NOT_PREFERRED`, `POST_R2_AUDIT: PASS` (job 54758, manifest
  `sha256:e8869f0c…`, freeze commit `4a2f606`).
- **Selected gate:** `g_t = g_old = E_t · R_t^B`, computed exactly as in P0-R2:
  - **Localizer:** current-query max 50-frame (1.0 s) alignment-head attention mass, heard
    frames only, earliest tie.
  - **LocalSupport (LS-B):** `E = [tanh(A/2)]_+`, null-template corrected, ε = 1e-12.
  - **BaselineConflict (BC-B):** `R_B = [P_M^B − P_E^B]_+` with partition `whisper_han_ascii_v1`
    (`sha256:7daa5009…`), float32 log-softmax of raw logits.
  - **Structural fallbacks:** `mid_character`, `localizer_fail`, `local_support_fail`,
    `baseline_provider_fail`, `nonfinite_signal` ⇒ `g = 0`.
- **Implementation:** reuses `csasr.inference_cf.core_r2` unchanged.
- **Ecf:** the English counterfactual *conflict* branch plays no role in the gate. The
  English-conditioned *hidden state* is used only for the direction (§2).

## 2. Direction

At decoder layer `ℓ = 24` (zero-indexed) and the query predicting `y_t`, with the current
decoded prefix `y_<t` held fixed:

```text
hB = site_24(x, y_<t; cB)          # cB = cM = [50258,50260,50360,50364]  (forced zh)
hE = site_24(x, y_<t; cE)          # cE =       [50258,50259,50360,50364]  (forced en)
Δ  = hE − hB                       # = hE − hM, since hB = hM for forced zh (P0 G1)
d  = Δ / (‖Δ‖₂ + 1e-6)
```

- **Branch computation:** both states come from **unsteered** full causal replays
  (`use_cache=False`) of the same audio and the same prefix, differing only at the language token.
- **No shortcuts:** no learned or corpus-fixed direction; no neighbour or global fallback.
- **Tiny/nonfinite fallback:** if `‖Δ‖₂` is nonfinite or `< 1e-4`, no edit is made at that
  position (reason `direction_fail`) and the position stays in the record.

## 3. Intervention site and rule

- **Model:** frozen Whisper-large-v3, bf16, eager attention, eval mode.
- **Site:** decoder layer 24, post-cross-attention residual `r = q + u_source`, the input to the
  FFN, before `final_layer_norm`. This is the DG-02-validated site, implemented by the canonical
  `csasr.lss.sites.DecoderPostCrossAttnInterventionHook`. It hooks the `encoder_attn` output and
  returns `u_source + (r' − r)`, so the layer itself computes `r'` and feeds it to its FFN.
  Nothing else is edited.
- **Layer 24:** the historical P0/R2 engineering anchor. It is **not** selected from any
  recognition outcome; layer choice belongs to P2.

Edit, via the contract `models.hooks.apply_steering` (no `sqrt(num_layers)` rescale,
`scale = 1`):

```text
h~  = hB + α · g_t · d
h'  = h~ · ‖hB‖₂ / (‖h~‖₂ + 1e-6)          # NormPreserve
```

- **Zero dose:** if `α · g_t = 0`, addition and rescaling are bypassed and the state is
  bit-identical (`apply_steering` returns the input for `α = 0` and leaves `g = 0` positions
  untouched).
- **Forced prefix:** positions `< num_forced_prefix = 4` are never edited (hook rule). The query
  predicting the first content token (`t = 0`) sits on the final prompt token, so **`t = 0` is
  never edited** (documented limitation).

## 4. Decoding and cache semantics

This is reference-free greedy decoding (beam 1, `max_new_tokens = 200`), with the same logits
processing as the ordinary baseline: `suppress_tokens` always, `begin_suppress_tokens` at
`t = 0`. For each step `t`:

1. **F1, unsteered cB replay** of `cB + y_<t`, with no hook installed (asserted). It yields
   `pB_t` (for R_B), the current-query alignment-head attention (for the window), and `hB`.
2. **F2, unsteered cE replay** of `cE + y_<t`, with no hook installed. It yields `hE`.
3. **Gate and direction** from F1/F2 as in §1–§2. If `α · g_t > 0`, the direction is valid and
   the query position is ≥ 4, then store `edit[t] = (g_t, d_t)`.
4. **F3, steered cB replay** of `cB + y_<t` with the hook installed only for this forward. It
   applies every stored `edit[τ]`, τ ≤ t, at that position's own layer-24 site.
   - **Why this equals KV-cached steering:** the layer-24 site at τ depends only on layers ≤ 24
     at positions ≤ τ, which layer-24 edits cannot change. Re-applying stored edits therefore
     reproduces exactly the states a KV-cached steered decode would hold.
   - **Hook removal:** the hook is removed after F3 (asserted).
5. **Emit:** `y_t = argmax(processed F3 logits)`. Stop on EOS or at 200 tokens.

At `α = 0`, F3 still runs with the hook installed (α = 0). **Its logits must be bit-identical to
F1's.** The gate and direction branches never share state with F3.

## 5. Population and doses (frozen)

- **Utterances:** 10 D-dev-select utterances, positions 0, 30, 60, …, 270 of the frozen R2
  inference panel (sorted by ID). The choice is deterministic, outcome-independent, spans the
  panel, and uses no new data role.
- **Doses:**
  - **P1-0:** `α = 0`, the identity check.
  - **P1-1:** `α = 1.0`, one engineering value. At L24 the P0 panel's site norms were 7.8–10.5
    (median 9.0), so `α·g ≤ 1` bounds the pre-NormPreserve edit to ≤ ~13% of the state norm:
    small, nonzero and measurable. It is chosen for acceptance only, never from recognition
    outcomes, and is not a P2 dose.

## 6. Serialization

**Per utterance and α:**

- manifest hash;
- emitted token IDs and text;
- termination (EOS or cap);
- the R2/B0 `generate()` content for the same ID, as a diagnostic.

**Per step:**

- identity: `t`, prefix length, query index;
- gate: `fallback_reason` and the full gate components (window, `π`, `A`, `E`, `P_M`, `P_E`,
  `Q`, `R_B`, `g`);
- direction: `‖Δ‖`, `direction_status`;
- edit: `edit_applied`, `α`, and the hook audit record for the query position (pre-norm,
  post-norm, realized edit norm, gate, layer);
- expected edit: pure-recomputed edit norm and `cos(h' − hB, d)`;
- tokens: unsteered argmax (F1) and steered argmax (F3), plus the `α = 0` bitwise-identity flag.

**Runtime:** job, GPU, program seconds, peak VRAM, forward/LID counts.

## 7. Acceptance criteria (all required ⇒ `P1_CAUSAL_ACCEPTANCE_PASS`, else `P1_BLOCKED`)

| # | Criterion | Rule |
|---|---|---|
| A1 | α = 0 identity | Every step of every `α = 0` utterance has F3 (hook installed) logits bitwise equal to F1. Every `α = 0` token sequence equals the unsteered reference argmax sequence. |
| A2 | Finite, sample-varying direction | All non-fallback directions finite. At least two directions with cosine < 0.99. |
| A3 | Tiny-direction fallback | Unit-tested. Every runtime `direction_fail` has no edit. |
| A4 | Real edit | ≥ 1 `α = 1` step with `g > 0`, finite `d`, query ≥ 4, and hook-recorded edit norm > 0. |
| A5 | Site | Every hook audit record is layer 24. Unit test: only the layer-24 site is changed. |
| A6 | Sign | Every applied edit has `cos(h' − hB, d) > 0`, with the recomputed edit norm matching the hook audit within 5% relative. |
| A7 | Gate | Logged `g` equals `E·R_B` recomputed from logged components (`core_r2`) within 1e-6. Fallback steps have `g = 0` and no edit. |
| A8 | NormPreserve | Every applied edit has post-norm = pre-norm within 1e-2 relative (bf16). Non-edited steps have zero edit. |
| A9 | Prefix integrity | Each forward's input is exactly `prompt + emitted prefix`. No position < 4 is ever edited. |
| A10 | Isolation | No site hook is present during F1/F2 at any step (asserted), and the model has no hooks after the run. |
| A11 | Continuation | Every `α = 1` decode terminates (EOS or cap) without exception and decodes to text. |
| A12 | Serialization | All 20 (utterance, α) records present and bound to the manifest; per-step records complete. |
| A13 | Cost | Runtime and peak VRAM captured. |

Agreement of the `α = 0` decode with the actual `generate()` B0 tokens is reported as a
diagnostic: cached vs full-replay bf16 numerics may differ rarely. Token changes under `α = 1`
are reported descriptively only, with no recognition claim.

## 8. Execution boundary

- **Commits:** spec, config, code and tests are committed and pushed before the manifest.
- **Pre-run audit:** an independent pre-run audit must give `PASS_TO_P1_RUN`.
- **Run:** one physical `sbatch` job on MIG 3g.40gb, both doses in one job.
- **Reruns:** only for a concrete infrastructure or implementation invalidity, never for an
  outcome.
- **Never in P1:** layer or alpha sweep, P2 controls, test data.

## 9. v1.1 amendment — A6 instrument only (after attempt 1, job 54770)

**Attempt 1.** It ran under v1 and is preserved as `P1_BLOCKED` (`P1_ATTEMPT1_INSTRUMENT_DEFECT.md`).
The failure was in the acceptance instrument, not the intervention. A6's norm clause compared the
**bf16**-realized edit with a **float64** reference under a relative 5% tolerance. That cannot be
met for edits at or below bf16 resolution at the site (≈ ‖h‖·2⁻⁸).

**Changed in v1.1: the A6 reference only.** It is now the frozen edit (`apply_steering`, NormPreserve,
`α`, gate) recomputed at the **site's actual precision and device**. It uses the recorded site
value, which converts back exactly to bf16, and the same bf16-cast gate and direction the hook
receives.

**Kept as it was:**

- the 5% tolerance and the sign clause (`cos(h' − hB, d) > 0`, float64);
- the float64 reference, reported descriptively for edits ≥ 10× bf16 resolution;
- gate, direction, site, α, population, all other criteria and the decode.

**Rerun.** Output goes to `results/inference_cf/p1_r1/` under a new manifest. The rerun is justified
by instrument invalidity only; recognition outcomes played no role.
