# Pre-P2 cached-equivalence acceptance — v1 (frozen before any equivalence outcome)

**Purpose.** Validate the KV-cached steering path (`experiments/inference_cf_cached.py`) as a
faithful execution of the accepted P1 algorithm. It must be validated before it is used for P2
headline recognition comparisons. This stage is implementation and equivalence only; it does no
recognition tuning.

## Frozen settings

| Item | Value |
|---|---|
| Model / precision / hardware | Whisper-large-v3 (weights `sha256:a8e94b85…`), bf16, eager attention, eval; one H100 MIG 3g.40gb |
| Conditions | cB = cM = `[50258,50260,50360,50364]`, cE = `[50258,50259,50360,50364]` |
| Detector (frozen R2) | `g = E·R_B`, via the unchanged R2 providers (max 1.0 s window, LS-B, BC-B, partition `sha256:7daa5009…`, fallbacks) |
| Direction / edit (P1) | `d = norm(hE − hB)` (tiny/nonfinite fallback 1e-4); canonical DG-02 hook at the post-cross-attention / pre-FFN site; `h' = NormPreserve(h + α·g·d)`; zero-dose bypass; positions < 4 never edited |
| Layers | 16 and 24, the P2 candidate set; both validated here |
| Doses (equivalence only) | α ∈ {0, 1.0} with φ(g) = g |
| Population | The 10 P1 utterances, i.e. R2-panel positions 0, 30, …, 270 (D-dev-select, already exposed). Deterministic, chosen before this spec, and never selected on reference correctness. Contains g = 0 and g > 0 steps, `mid_character` fallbacks, and short and long utterances (P1: 374 steps, 20 edits). |
| Decode | Greedy, `max_new_tokens = 200`, `generate()` suppression |

**Comparison reference.** At every step of each cached decode, the harness runs the **accepted P1
full-replay definition on the same prefix**:

- unsteered cB replay for attention, window, E, R_B and hB;
- unsteered cE replay for hE;
- steered cB replay with the replay-side edit history.

Every per-step quantity is compared at matched logical positions.

## Near-tie definition (frozen)

A token disagreement is a near-tie only if the top-1 − top-2 margin of the processed logits is
≤ 0.25 in **either** implementation. That is about two bf16 ulps at typical logit magnitudes.

## Criteria (all required ⇒ `CACHED_EQUIVALENCE: PASS`)

| # | Criterion | Rule |
|---|---|---|
| CE1 | Zero-dose identity | α = 0: at every step the cached S logits are **bitwise equal** to cached B logits. For all 20 (utterance, layer) decodes, the α = 0 token sequence equals the plain cached greedy baseline (`cached_greedy`, cB) exactly. |
| CE2 | Positions and lineage | Every decode has `lineage_ok` (all three branches fed exactly prompt + emitted prefix; positions `0..L−1` without gaps) and consecutive query indices. |
| CE3 | Branch isolation | Every decode reports three distinct cache objects. Unit test: S edits never change B's cache. |
| CE4 | Gate agreement (cached vs replay) | Fallback reason identical on ≥ 99% of steps; window identical on ≥ 95% of steps where both have one; \|ΔR_B\| ≤ 0.02 on ≥ 99% of steps; \|ΔE\| ≤ 0.02 on ≥ 99% of steps with identical windows; \|Δg\| ≤ 0.02 on ≥ 95% of steps. Every exceedance is listed. |
| CE5 | Direction agreement | Direction status identical on 100% of steps; `cos(d_cached, d_replay)` ≥ 0.99 and relative norm difference ≤ 0.02 on ≥ 99% of steps. |
| CE6 | Intervention equivalence (α = 1) | `edit_applied` identical on ≥ 97% of steps. Every cached edit has positive sign (`cos(reference_edit, d) > 0`), realized edit equal to its site-precision replica within 5% (P1 v1.1 rule), and NormPreserve within 1e-2. On commonly edited steps with \|Δg\| ≤ 0.02, cached vs replay realized edit norm within 5% on ≥ 95%. **Every** cached-vs-replay steered argmax disagreement must be a near-tie; any non-near-tie disagreement blocks. |
| CE7 | Hook isolation | No site hook during the unsteered branches at any step (asserted); none leaked after the run. |
| CE8 | Runtime | Wall and program time, processed audio seconds, output tokens, peak VRAM, forward and LID counts captured. |

**Diagnostics (no pass rule).**

- Cached α = 0 tokens vs the R2 `generate()` B0 tokens.
- Maximum \|Δlogit\| between cached and replay.
- Cached vs replay speed.

**Verdict.**

- **PASS:** all of CE1–CE8 hold. The cached path may then be used for P2.
- **Blocked by an implementation bug:** fix it, test, commit, push, re-audit, rerun into a new
  versioned directory.
- **Blocked by a genuine cache-semantic problem:** stop before P2.
- **Never:** fall back to unmatched full replay for headline P2 comparisons.
