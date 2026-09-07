# DG-02 — Exact Post-Cross-Attention / Pre-FFN Intervention Site: Implementation Spec

**Status:** Stage 3 / DG-02 — **COMPLETE / FROZEN**. Implementation + CPU/synthetic validation
green, and the real-model acceptance (§14 rung 1) passed at **both L16 and L24** on
whisper-large-v3 (Slurm job **50369**, `mig` H100 3g.40gb; artifact
`results/dg02_real_acceptance.json`; see §18). Companion to `METHOD_CONTRACT.md` (MC),
`CODE_MAP.md`, `DG01_METRICS_RESULTS_SPEC.md`. Scientific definitions are MC's and are not
reinterpreted to fit code. Layer selection (L16 vs L24) and directions belong to DG-03 and are
**not** decided here.

Authoritative primitive (extended in the implementation pass): `src/csasr/lss/sites.py`
(`DecoderPostCrossAttnRecorder`, `DecoderPostCrossAttnSteeringHook`, `assert_site_reconstruction`,
and the new `DecoderPostCrossAttnInterventionHook`, `AuditRecord`, `num_forced_prefix_from`,
`CONTRACT_DECODER_LAYERS`). See §17 for the delivered implementation and its test/verification
status.

---

## 1. Exact Whisper forward anatomy (installed source, verified)

Model: **Whisper-large-v3** (`src/csasr/models/whisper.py`; 32 decoder layers, `d_model` from
config). `transformers` `WhisperDecoderLayer.forward` (installed source, lines 499–544) runs, in
order:

```
499  residual = hidden_states                 # block input
500  hidden_states = self.self_attn_layer_norm(hidden_states)
503  hidden_states, _ = self.self_attn(..., cache_position=cache_position)
511  hidden_states = dropout(hidden_states)    # identity under eval()
512  hidden_states = residual + hidden_states  # (A) post-self-attn state

# Cross-Attention Block
517  residual = hidden_states                  # (A) again — the decoder-side state
518  hidden_states = self.encoder_attn_layer_norm(hidden_states)
519  hidden_states, cross_attn_weights = self.encoder_attn(hidden_states=..., key_value_states=encoder_hidden_states, past_key_values=..., ...)
527  hidden_states = dropout(hidden_states)    # identity under eval()
528  hidden_states = residual + hidden_states  # (B) === THE SITE (post-cross-attn, pre-FFN)

# Fully Connected (FFN)
531  residual = hidden_states                  # (B) — exactly what the FFN consumes
532  hidden_states = self.final_layer_norm(hidden_states)
533  hidden_states = activation_fn(fc1(hidden_states))
535  hidden_states = fc2(hidden_states)
537  hidden_states = residual + hidden_states  # (C) block output (post-FFN) = REJECTED site
539  return (hidden_states, ...)
```

Module/call order at the site: `self_attn_layer_norm → self_attn → (residual add A) →
encoder_attn_layer_norm → encoder_attn → (residual add B = SITE) → final_layer_norm → fc1 →
activation → fc2 → (residual add C = block output)`.

- **(B) line 528 is the canonical site.** **(C) line 537 (block output) is the rejected site**
  (`DECODER_REJECTED_TENSOR = "decoder_block_output"`, used by legacy post-FFN hooks).
- The dropouts at 511/527 are identity in `eval()`; `assert_dropout_disabled` enforces this — a
  precondition, not an assumption.

---

## 2. Operational definitions of q / u_source / r

Two symbol conventions coexist in the contract; the spec keeps both explicit so nothing is
conflated.

**Forward-pass additive decomposition of the site (DG-02 task / MC §5 disagreement notation):**

| Symbol | Exact tensor | How captured (exact) |
|---|---|---|
| `q_{ℓ,t}` | line 517 `residual` = post-self-attn decoder-side state, *before* adding the cross-attention contribution | `forward_pre_hook` on `encoder_attn_layer_norm`, `args[0]` (already done by the recorder's `_pre_hook`) |
| `u^S_{ℓ,t}` (source) | line 519 `encoder_attn` output = source-conditioned cross-attention contribution (pre-dropout; dropout identity under eval) | `forward_hook` on `encoder_attn`, `output[0]` |
| `r_{ℓ,t}` | line 528 `residual + hidden_states` = **the site** the FFN consumes | computed `q + u_source` |

- **Is `u_source` directly obtainable from a module output?** Yes — `encoder_attn`'s `output[0]`.
- **Is any LayerNorm applied between `q`, cross-attention, and `r`?** `encoder_attn_layer_norm` is
  applied only to the *input* of `encoder_attn` (`u_source = encoder_attn(LN(q))`); the residual
  add `r = q + u_source` uses the **un-normalized** `q`. So the site is a raw residual sum; no LN
  sits on the `q→r` path. (`final_layer_norm` acts only inside the FFN block, on `r`, and does not
  alter the stored `r`.)
- **Exactness of `r = q + u_source`:** exact in real arithmetic; the layer recomputes
  `residual + attn_out` in the model dtype, so a captured `r` matches to rounding
  (`assert_site_reconstruction`: error tracks dtype ε, ≈6.7 ULP fp32 / 1.5 ULP bf16, and is orders
  of magnitude below the site-vs-block gap).

**Steering (MC §2) notation — distinct use of the same letters:** MC §2's pre-intervention
`q_{ℓ,t}` is this site `r` above (the state the FFN would otherwise consume); the steering update
`ũ^S = α·s_ℓ·g·d`; the repaired site `r̃ = NormPreserve(r + ũ^S)`; the *effective* steering update
`= r̃ − r`. The DG-02 infrastructure captures the forward decomposition (q, u_source, r) **and**
applies the MC §2 steering to `r`. The spec always says "site `r`" for line 528 and "steered site
`r̃`" for the repaired tensor to avoid the letter collision.

---

## 3. Chosen hook / interception strategy (least invasive, exact)

Reuse the existing two-hook reconstruction (already in `sites.py`); do **not** monkeypatch
`WhisperDecoderLayer.forward`.

Per target layer `k`:
1. `forward_pre_hook` on `layer.encoder_attn_layer_norm` → capture `q = args[0]` (the site's
   `residual`).
2. `forward_hook` on `layer.encoder_attn` → receive `u_source = output[0]`; reconstruct
   `r = q + u_source`; compute steered site `r̃`; return `_rebuild(output, u_source + (r̃ − r))`
   so the layer's own `residual + attn_out` at line 528 recomputes to `r̃` with **no** assumption
   about downstream FFN behaviour.
3. **New** `forward_pre_hook` on the `WhisperDecoderLayer` itself → read `cache_position` /
   `past_key_values` from kwargs to record absolute decode position (see §5). This replaces the
   `T>1` heuristic with a real position.

**Why this over alternatives:** (a) subclassing/patching the layer forward is invasive and would
diverge from the frozen backbone; (b) a single post-FFN hook is the *rejected* site; (c) the
two-hook reconstruction is already validated (`assert_site_reconstruction`) and yields
`cross_attn_weights` cheaply. Disabled/zero-gain path returns `output` unchanged → **byte-identical
frozen forward** (MC §2 invariants: zero-gain bit-identical, `α=0` no-op).

Capture-only diagnostics keep using `DecoderPostCrossAttnRecorder`, extended to store `q`,
`u_source`, `r` separately (currently it stores only their sum) — an exact, numerics-free change.

---

## 4. Layer-number mapping (verified — no off-by-one)

- `bundle.decoder_layer(k)` returns `model.model.decoder.layers[k]` — **0-indexed, direct**
  (`src/csasr/models/whisper.py:64`).
- Whisper-large-v3 has **32 decoder layers** (indices 0–31); `num_decoder_layers = 32`.
- **Scientific layer 16 → Python index 16; scientific layer 24 → Python index 24.** No translation
  is required; MC §3 already states "decoder, 0-indexed". Both indices exist and are interior.
- The implementation must `assert 0 <= k < bundle.num_decoder_layers` and refuse any layer outside
  `{16, 24}` on contract paths (MC §3). Layer *selection* between the two is DG-03.

---

## 5. Cache-position strategy

- The installed layer forward receives `cache_position: torch.LongTensor` and
  `past_key_values: EncoderDecoderCache`. Absolute position is therefore first-class.
- **Primary:** a `forward_pre_hook` on the decoder layer records `cache_position` (per-token
  absolute indices) each call; the `encoder_attn` post-hook maps its `T` positions to
  `abs_pos = cache_position[0..T-1]`.
- **Fallback / cross-check:** `cache_length(past_key_values)` (already implemented at
  `steer_sweep/hooks.py:46` and `experiments/job_a_frozen.py:187`) gives `_abs_start`; then
  `abs_pos = _abs_start + t_offset`. Job-B already uses exactly this
  (`job_b_training.py:337 _abs_start = cache_length(kwargs.get("past_key_values"))`).
- **Prefill vs step:** prefill is a single call with `T = prefix_len` (positions `0..prefix_len-1`);
  every later call has `T == 1`. Do **not** assume each call starts at position 0. `use_cache=False`
  (teacher-forced screening, `job_b_training.py:738`) runs one call with the full `T`; the same
  `cache_position` mapping holds, so cached and non-cached position semantics agree (test 10).

---

## 6. Beam strategy

- Under `generate(num_beams=B)` the decoder batch dim is `batch*B`; Whisper reorders
  `past_key_values` per step so the hidden-state rows follow the **current** beam order.
- **Gate is computed per row from that row's own `(q, u_source, r)`**, so each beam's gate is
  intrinsically its own — never shared across beams that share source audio. The gate interface is
  `gate_fn(q, u_source, r, abs_pos, row_meta) -> gain (B_eff,) or (B_eff, T)`; a constant gate is
  allowed for tests.
- Legacy pattern to mirror: `job_a_frozen.py` uses `items = B // num_beams` and loops per item —
  evidence the per-beam bookkeeping is understood at the rejected site and must be reproduced here.
- `IMPLEMENTATION GAP (beam-transported gate)`: when a gate value is *precomputed per source item*
  (e.g. an acoustic localizer decision) rather than computed from the row's own state, the caller
  must expand and **reorder** it to the live beam order each step (`_reorder_cache` order). This
  mapping is not yet implemented; DG-02 provides the row-local interface, and the source→beam
  transport is flagged for the gate ticket.

---

## 7. Forced-prefix exclusion strategy

- Prefix = `build_prefix(processor, language, task)` =
  `[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]` → **length 4** for this
  config (`src/csasr/data/alignment.py:134`). Legacy hooks hard-code `PREFIX_WIDTH = 4`
  (`job_a_frozen.py:56`).
- **DG-02 rule:** derive `num_forced_prefix` dynamically from `len(build_prefix(...))` (or the
  length of the forced `decoder_input_ids` / generation config), **not** a hard-coded 4. Exclude
  every position with `abs_pos < num_forced_prefix` from steering (return `q` bit-identical there).
- `IMPLEMENTATION GAP (first-content-token)`: Whisper predicts the first content token from the
  final prefill position (`abs_pos = prefix_len − 1`), a prefix position. The current
  `DecoderPostCrossAttnSteeringHook` skips the whole prefill (`site.shape[1] > 1 and not
  steer_prefill`), so the first content token is never steered. Whether to steer that final prefill
  query is an open implementation detail (testable) — record it, do not silently choose; keep the
  legacy "skip all prefill" default until DG-03 needs otherwise.

---

## 8. Norm-preservation strategy

- Reuse `csasr.models.hooks.apply_steering(..., norm_preserve=True)` (`hooks.py:72`) unchanged. It
  computes `r̃ = (r + ũ^S) · ‖r‖₂ / (‖r + ũ^S‖₂ + EPS)` **per (batch, token)** along the feature
  dim, `EPS = 1e-6`.
- **Precise norm preserved:** the L2 norm of the *site* `r` (line 528), per token, per beam row.
- **No** `sqrt(num_layers)` rescale (MC §1; the exact-site hook already omits it, unlike legacy
  `DecoderSteeringHook` at `hooks.py:170`). Do not switch normalization algorithms; the same
  `apply_steering` must be used identically for the proposed system and every control
  (matched-energy/random/sign-flip) so comparisons change direction/location, not energy.

---

## 9. Detached (analysis) vs gradient-enabled (training) modes

- **Analysis / recording mode:** `DecoderPostCrossAttnRecorder` detaches (`.detach()`) and upcasts
  to fp32 — no autograd graph retained. Extend it to store `q`, `u_source`, `r` (and, opt-in,
  `cross_attn_weights`) detached. Used for MC §5 disagreement scores, DG-04 diagnostics, and audit
  logs.
- **Gradient-enabled mode:** the steering path must **not** detach `q`/`u_source`; `apply_steering`
  is differentiable, so gradients reach `direction`, `alpha`, and `gain`. Frozen backbone stays
  frozen (`requires_grad=False` on Whisper params, loader-enforced); trainable = the intervention/
  gate parameters only.
- `IMPLEMENTATION GAP (trainable direction/gate)`: the current
  `DecoderPostCrossAttnSteeringHook` takes a fixed `direction` tensor + scalar `alpha`. For DG-03+
  gate/direction training these become `nn.Parameter`s / a gate `nn.Module` producing per-row
  `gain`. This is an infrastructure extension (a `mode: {"record","steer","train"}` flag), **not**
  a change to the frozen site semantics.

---

## 10. Audit-record schema (compact, opt-in, last-token by default)

Per steered call, one record (JSON-serialisable, off by default; last position only unless asked):

```
layer                # python index (16 or 24)
abs_pos              # cache_position (absolute decode position)
row / beam_index     # row in the batch*num_beams dim (beam identity if available)
is_forced_prefix     # bool: abs_pos < num_forced_prefix (excluded)
gate                 # g in [0,1] for this row/position
alpha                # steering strength (ρ)
pre_norm             # ||r||_2   (site)
post_norm            # ||r̃||_2  (steered site, ~= pre_norm under norm-preserve)
edit_norm            # ||r̃ - r||_2  (effective edit energy)
steered              # bool: was an edit actually applied (gain>0, alpha!=0, not prefill/prefix)
```

Raw eligibility counts (`n_positions`, `n_prefix_excluded`, `n_gate_positive`) may be logged for
later use, but DG-02 **must not** compute a `gate_coverage` value — that denominator is deferred
(DG01 spec §4 / MC §6). These counts feed the future gate-coverage definition, nothing more.

---

## 11. Legacy separation (classification; none modified)

| Component | Path | Class |
|---|---|---|
| Exact-site recorder + steering hook + reconstruction assert | `src/csasr/lss/sites.py` | **CANONICAL/REUSABLE** — base to extend (record q/u/r; add layer pre-hook, cache pos, dynamic prefix, gate_fn, modes) |
| Norm-preserving update | `csasr.models.hooks.apply_steering` | **CANONICAL/REUSABLE** — reuse verbatim |
| F5 whole-block frozen hook | `experiments/job_a_frozen.py` `NormPreserveDecoderHook` (post-FFN) | **LEGACY** — untouched |
| T1/T2 whole-block training hooks | `experiments/job_b_training.py` `T1Hook`, `T2Hook` (post-FFN) | **LEGACY** — untouched |
| Whole-block steering + sqrt rescale | `csasr.models.hooks.DecoderSteeringHook`; `steer_sweep/hooks.py DecoderSteering` | **LEGACY / rejected site** — untouched |
| Cache-length helper | `steer_sweep/hooks.py:46 cache_length`; `job_a_frozen.py:187 _cache_length` | **REUSABLE** — reuse for abs position |
| Prefix builder | `csasr.data.alignment.build_prefix` | **REUSABLE** — source of `num_forced_prefix` |

The new exact-site path **coexists** with F5/T1; legacy behaviour and outputs are preserved (AGENTS
invariant 3). New code lives in `sites.py` (+ its own test file), off the legacy call paths.

---

## 12. Exact files allowed to change (implementation pass — not this pass)

- `src/csasr/lss/sites.py` — extend the recorder (store q/u_source/r), add the decoder-layer
  pre-hook for `cache_position`, dynamic `num_forced_prefix`, a `gate_fn`/per-row gain interface,
  the `record|steer|train` mode flag, and the audit-record emitter. **Do not** alter the site
  definition or `apply_steering`.
- `tests/test_lss_sites.py` — extend with the §13 matrix (may add `tests/test_dg02_site.py` for the
  new capabilities).
- Docs: this file, `CODE_MAP.md` (mark the extended primitive), `STATUS.md`.

**Forbidden:** `models/hooks.py apply_steering`/`DecoderSteeringHook`, `job_a_frozen.py`,
`job_b_training.py`, `steer_sweep/*`, direction construction, training objectives, configs, any
legacy result. No monkeypatch of `WhisperDecoderLayer`.

---

## 13. Test matrix (CPU/synthetic; real block class, no model download)

Build tests on a **tiny synthetic Whisper** (as `tests/test_lss_sites.py` already does) or a
hand-rolled module exposing `encoder_attn_layer_norm`/`encoder_attn`/`final_layer_norm`, so no
weights are downloaded.

| # | Test |
|---|---|
| 1 | `r == q + u_source` (recorder), to dtype ULP tolerance |
| 2 | `alpha=0` (β=0) is baseline-identical (bit-for-bit) |
| 3 | disabled/removed hook is baseline-identical |
| 4 | norm preserved: `‖r̃‖ ≈ ‖r‖` when repair enabled |
| 5 | forced-prefix position (`abs_pos < num_forced_prefix`) is never edited |
| 6 | eligible position **is** edited |
| 7 | two different per-token gates → different edits |
| 8 | two beam rows with different states → different gate values/edits |
| 9 | cached position increments correctly (`cache_position`/`cache_length`) |
| 10 | cache vs `use_cache=False` position semantics agree |
| 11 | hook acts at the site, **not** block output (`site_vs_block` gap > 0; reuse `assert_site_reconstruction`) |
| 12 | FFN consumes the repaired `r̃` (perturb → block output changes consistently) |
| 13 | detached recording holds no autograd graph (`.grad_fn is None`) |
| 14 | gradient mode: loss on steered output yields non-None grad on `direction`/`gain` params |
| 15 | frozen backbone params keep `requires_grad=False` while intervention params train |
| 16 | hook cleanup restores normal forward (`assert_no_site_hooks` passes after `__exit__`) |
| 17 | layer 16 maps to `decoder.layers[16]` |
| 18 | layer 24 maps to `decoder.layers[24]` |

Plus a small synthetic `WhisperDecoderLayer` instantiation test (real block class, random weights,
CPU) exercising the two-hook reconstruction end-to-end.

---

## 14. Real-model verification ladder (planned; **do not execute** in DG-02)

1. one real utterance — `assert_site_reconstruction` at L16 and L24 on the real model; confirm
   `reconstruction_ok`, `site_differs_from_block_output`.
2. ~20 real utterances — free-decoding smoke: zero-gain ≡ baseline; prefix never steered; per-beam
   gates distinct; positions increment.
3. small predefined development subset — emit a `result_v1` artifact (DG-01) with raw eligibility
   counts logged (no gate-coverage value).

These require GPU/model inference and are out of scope for this pass; the implementation pass stops
before them.

## 15. DG-01 integration

A future exact-site runner emits canonical `result_v1` via `csasr.evaluation.result_schema` +
`canonical`/`retention`; it records raw gate-eligibility counts (§10) but **does not** define MER,
PIER, correction, corruption, outside harm, retention, the gain sign, the result schema, or the
gate-coverage denominator — all frozen/deferred by DG-01.

## 16. Implementation gaps / blockers

Non-blocking `IMPLEMENTATION GAP`s (all are additive infra on a validated primitive, resolvable in
the implementation pass):
- G-a: recorder stores only the summed site; extend to expose q / u_source / r separately.
- G-b: cache position via a new decoder-layer pre-hook (replace the `T>1` heuristic).
- G-c: `num_forced_prefix` derived dynamically, not hard-coded 4.
- G-d: per-row `gate_fn` / trainable `direction`/`gain` params + `record|steer|train` mode flag.
- G-e: beam-transported (source-item) gate reordering — row-local interface now; transport later.
- G-f: first-content-token / final-prefill-position steering choice — record, decide in DG-03.

**No blocker.** The exact site, its reconstruction identity, layer mapping, norm-preservation, and
cache/prefix surfaces are all present and verified in the installed model and repository; the
remaining items are additive and scoped to `sites.py` + its tests.

---

## 17. Delivered implementation (Stage-3 implementation pass)

**Canonical path.** The exact-site intervention is
`csasr.lss.sites.DecoderPostCrossAttnInterventionHook`. It repairs the site `r = q + u_source`
(line 528) and returns `u_source + (r̃ − r)` from the `encoder_attn` forward hook, so the layer's
own `residual + attn_out` recomputes to `r̃`; the FFN then consumes `r̃`. The final block output
(line 537) is never modified. `DecoderPostCrossAttnRecorder` now stores `q`, `u_source`, and `r`
separately (detached, fp32) for the analysis path. The legacy
`DecoderPostCrossAttnSteeringHook` is left unchanged and still backs
`assert_site_reconstruction` and the historical tests.

**q / u / r operational definitions (as implemented).** `q` = `encoder_attn_layer_norm`
forward-pre-hook `args[0]`; `u_source` = `encoder_attn` forward-hook `output[0]`;
`r = q + u_source`. The identity holds to dtype ULP (verified synthetically and, once §14 rung 1
runs, on the real model). `u_source` is taken directly from the module output — never approximated
from an unrelated hidden state.

**Exact site.** post-cross-attention, pre-FFN (line 528); rejected site = block output (line 537).

**L16 / L24 mapping.** `bundle.decoder_layer(k)` → `model.model.decoder.layers[k]`, 0-indexed,
direct: **L16→index 16, L24→index 24**. `CONTRACT_DECODER_LAYERS = (16, 24)`; the hook refuses any
out-of-range index and, with `enforce_contract_layer=True`, any non-`{16,24}` layer on the contract
path. It does **not** pick between them.

**Cache strategy.** A `with_kwargs` forward-pre-hook on the decoder layer captures its own
`cache_position` (primary); the fallback is the KV-cache length (`_cache_length`, mirroring
`steer_sweep.hooks.cache_length`). Absolute positions advance across prefill (T=prefix_len) and
cached steps (T=1) and are never reset to 0.

**Forced-prefix strategy.** `num_forced_prefix` is derived dynamically from
`num_forced_prefix_from(processor, ...)` = `len(build_prefix(...))` (4 for this config) — not a
hard-coded 4. Every position with `abs_pos < num_forced_prefix` gets a zero effective edit. Because
Whisper predicts the first content token from the final prefill position (a prefix position), that
first content token is not steered and the first steered position is `abs_pos = num_forced_prefix`
— the legacy "skip all prefill" default (G-f kept, not silently changed).

**Norm-preservation strategy.** Reuses `csasr.models.hooks.apply_steering(..., norm_preserve=True)`
verbatim — per-(row,token) L2 norm of the site `r`, `EPS=1e-6`, no `sqrt(num_layers)` rescale.

**Gate / edit interface.** External direction/edit tensor + `gate_fn(q, u_source, r, abs_pos)` or a
`gain` tensor (scalar / (B,) / (B,T)); per-row, per-token; state-dependent gates are row-local so
beams with different states get different gains. `mode ∈ {"steer","train"}`: the steer path is
inference; the train path stays differentiable so gradients reach trainable direction/gain/alpha
while the frozen backbone stays frozen. `record=True` emits the §10 `AuditRecord` schema (detached;
raw eligibility counts only — **no** `gate_coverage`). Nothing here hard-codes `v_nat`, `Δ^⊥`, the
disagreement score, the factorized gate, a layer, or `β`.

**CPU/synthetic tests.** `tests/test_dg02_site.py` — 19 tests (all §13 matrix items + the standalone
real-`WhisperDecoderLayer` reconstruction), plus the 11 pre-existing `tests/test_lss_sites.py`:
**30 passed** on CPU. DG-01 regression (`tests/test_canonical_metrics.py`, `test_retention.py`,
`test_result_schema.py`, `test_legacy_adapter.py`, `test_dg01_regression.py`): **31 passed** —
unchanged.

**Real-model acceptance (PASSED — see §18).** `experiments/dg02_real_acceptance.py` (launcher
`sbatch/cs_asr_dg02_real_acceptance.sh`) runs the §14 rung-1 one-utterance check at L16 and L24:
D5 β=0 token+transcript identity, D6 `r=q+u` on real states (full-sequence and a cached step), D7
`assert_site_reconstruction`, D8 forced-prefix zero-edit + one eligible edit, D9 cache-position
advance/alignment, D10 norm preservation under a tiny fixed probe edit. It writes a JSON report and
exits non-zero unless all gates pass. It makes no ASR-quality judgement, no layer/direction/β
selection.

---

## 18. Real-model acceptance result (freeze evidence)

Executed on a GPU node via Slurm — the CPU dev box (2 GB RAM) cannot load whisper-large-v3.

- **Slurm job:** `50369`, partition `mig`, 1× `nvidia_h100_80gb_hbm3_3g.40gb` (40 GB slice),
  node `worker-mig-3g40gb-0`, State `COMPLETED` (exit 0), elapsed 34 s. Logs
  `logs/cs_asr_dg02_real_acceptance_50369.{log,err}`.
- **Model:** `openai/whisper-large-v3`, `torch.bfloat16`, 32 decoder layers, device `cuda`.
- **Utterance:** `ZH-CN_U0091_S0_68`, role `D-dev-select` (integration/debug), duration 2.525 s,
  `num_forced_prefix = 4`.
- **Implementation commit under test:** `4e10399` (`sites.py` = HEAD at submission).
- **Artifact:** `results/dg02_real_acceptance.json`. Verdict **PASS**.

Per-layer gates (all PASS at both layers):

| Check | L16 | L24 |
|---|---|---|
| β=0 token identity | PASS (steered_calls=0) | PASS (steered_calls=0) |
| β=0 transcript identity | PASS | PASS |
| `r=q+u` max abs err (full-seq / cached) vs tol | 0.00305 / 0.00195 ≤ 0.1216 | 0.00781 / 0.00635 ≤ 0.1729 |
| exact site: reconstruction_ok / differs_from_block / err_vs_block_gap | ✓ / ✓ / 0.0078 | ✓ / ✓ / 0.0084 |
| forced-prefix zero-edit | PASS | PASS |
| eligible position edited | PASS | PASS |
| cache positions (advance, no reset, boundary aligned) | `[0..8]` ✓ | `[0..8]` ✓ |
| norm preservation max rel dev (≤ 1e-2) | 4.59e-4 | 1.86e-4 |

Tolerances are dtype-justified (bf16), not weakened to pass: `err_vs_block_gap ≈ 0.008 ≪ 1` shows
the hook acts on the site tensor, not the block output; `r−(q+u)` sits at the bf16 rounding scale.
No layer/direction/β/rank/gate selection was made (that is DG-03).
