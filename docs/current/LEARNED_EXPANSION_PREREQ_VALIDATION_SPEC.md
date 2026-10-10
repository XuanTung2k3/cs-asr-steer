# Learned-Expansion Frozen Prerequisite — Pre-Registration SPEC (frozen before GPU)

**Status:** PRE-REGISTERED. Frozen before any GPU run; do **not** edit after results.
**Runner:** `experiments/learned_expansion_prereq.py`. **Results:** `results/learned_expansion_prereq_frozen/`.

## Purpose
Test whether the 10-utterance BASIS-A2 atlas Conditioning signal at **L26/L27**
survives on the canonical **300-utterance D-dev-select** population, with bounded
damage, before any learned training is run at those layers. This is a **frozen**
(no-training, ungated, fixed-direction) exact-site steering screen.

## Frozen design
- **Layers:** `{24, 26, 27, 31}` — L24 validated anchor/cross-check; L26/L27 the
  atlas Conditioning band; L31 negative control. (Noisy L0/L6 Raw/Local micro-panel
  ranks are deliberately **not** validated — their 10-utt differences are too small.)
- **Directions (all 5):** Raw, Local, Conditioning, Raw+Cond, Local+Cond, taken from
  the frozen BASIS-A2 per-layer `directions.json` (canonical construction on
  D-construct; hash-checked at load).
- **Dose:** ρ = 0.5 **only** (the DG-04 reference low-dose region; this stage asks
  about causal correction headroom, not a new dose search).
- **Site/decode:** decoder post-cross-attention/pre-FFN exact site; NormPreserve;
  greedy, temperature 0, beam 1; Whisper-large-v3; per-layer scale `s_ℓ` from the
  atlas directions record; effective dose `alpha=ρ, scale=s_ℓ`.
- **Population:** `D.build_population(..., "D-dev-select", assert_anchors=True)` — the
  identical 300-utt candidate population used by DG-04/05/06/07.
  - n_utterances = **300**
  - ids_fingerprint = `sha256:1e826e04d41510513f361b95757a3ac6980b811107d07d79772425af3743f67b`
  - dataset composite = `sha256:3981d6a8134e8f73b49e1fc66451980a59d0f460759aab279a142ecb2f7310ab`
- **Baseline:** reuse frozen DG-04 B0 greedy baseline on the identical population.
- **Metrics:** canonical `result_v1` — MER, PIER, embedded WER, matrix CER,
  corrections, corruptions, utility, outside harm, embedded + matrix retention,
  intervention energy.

## Gate rule (frozen — do NOT invent thresholds after seeing outcomes)
The gate is **causal correction headroom with bounded damage**, NOT positive frozen
net utility (a learned controller may selectively rescue a globally-too-broad
direction — cf. L24, where frozen Raw/Local is damaging yet the DG-05/06 learned
controller is net positive). For **Conditioning at L26 and L27**, classify jointly
on the frozen 300-utt screen:

- **VALIDATED CANDIDATE** — Conditioning at the layer shows meaningful correction
  headroom at ρ=0.5: PIER gain > 0 **and** corrections > corruptions (net POI utility
  ≥ 0) **and** matrix retention ≥ 0.90 **and** outside harm not worse than the L24
  Conditioning reference on this same population (bounded damage).
- **WEAK / INCONCLUSIVE** — positive PIER/correction signal but fails one damage
  bound (matrix retention < 0.90, or corruptions ≥ corrections, or outside harm
  above the L24 reference), or the signal is within noise of L31.
- **NOT SUPPORTED** — no correction headroom (PIER gain ≤ 0 or corrections ≈ 0), i.e.
  the 10-utt signal did not survive.

Decisions:
- **L24** remains the anchor regardless of its frozen-screen utility.
- **L31** is the negative control; not a learned candidate unless it unexpectedly
  shows strong, clean correction headroom (which would invalidate the control).
- If **neither** L26 nor L27 is VALIDATED CANDIDATE → do **not** train at L26/L27;
  record "Conditioning cross-layer signal did not survive full-dev validation" and
  continue with L24-only learned conclusions.
- If **one** validates → that single layer is the Conditioning candidate; run one-seed
  learned Cond-only, Raw+Cond, Local+Cond there (Raw-only/Local-only only if needed
  for a fair rank-1-vs-rank-2 read).
- If **both** validate → take the **stronger** by the rule above (PIER gain, then net
  utility, then matrix retention, then −outside harm) for the first learned screen;
  do not train both unless their frozen evidence is genuinely different.

Goal: minimise learned layer search.

## Data safety
Allowed: D-dev-select, loc-train, util-train. **Forbidden: D-dev-confirm, D-test.**
No redesigned method is evaluated on the exposed D-test in this stage.
