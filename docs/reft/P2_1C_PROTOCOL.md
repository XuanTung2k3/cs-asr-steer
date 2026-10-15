# P2-1c — on-policy continuation and switch-back failure audit (protocol)

**Status:** pre-registered before any P2-1c GPU output existed. It is frozen in `p2_1c/freeze.json`.

**Files:**
- Code: `reft/p2_1c_continuation_audit.py`.
- Configuration: `configs/reft/p2_1c_frozen.yaml`.
- Tests: `tests/reft/test_p2_1c.py`.
- Slurm: `slurm/reft_p2_1c.sbatch` (GPU) and `slurm/reft_p2_1c_cpu.sbatch` (CPU).
- Rules: `docs/reft/P2_1C_DECISION_RULES.md`.

**Type.** A bounded, no-training diagnostic on the exposed dev150 panel, using ten cases fixed in P2-1b. It is
exploratory and non-deployable: the anchors are reference-informed. It is not confirmation.

**Out of scope:** training, optimizer steps, adapter edits, λ or LoRA sweeps, M1, encoder adapters, new data or cases,
P2-2 and P3.

## 1. Question

Are the persistent Mandarin switch-back omissions of supervised R2 LoReFT caused by:
- **H1** earlier decoder-prefix history;
- **H2** a local preference at the switch-back itself;
- **H3** sequence- or search-level path preference;
- **H4** none of these clearly (mixed or unresolved)?

These are operational explanations, not mutually exclusive causal mechanisms.

## 2. Population and systems (fixed)

**Cases.** `p2_1b/prefix_audit/cases_private.json`, unchanged. In file order they are named d0–d4 (damaged) and s0–s4
(safe).
- KL-recovered spans: d1 (15 units) and d2 (6 units).
- Persistent spans: d0 (13), d3 (9) and d4 (9). This follows `p2_1b/posthoc/span_recovery_59293_private.json`.
- No case is added or replaced. An unmappable case stays in the inventory.

**Systems.**
- B0: frozen Whisper-large-v3 with frozen Auto-LID.
- CE42: P2-1 R2, selected epoch 2.
- CE43, CE44, KL42, KL43, KL44: the selected epochs from the sealed `selection.json` files.

Every adapter must match the `adapter_sha256` in its run manifest. The backbone weights hash must equal the pinned hash
before and after.

**Common to all systems:** the same audio, the same frozen-LID four-token prompt, and the same canonical decode
configuration.

## 3. T0 — generated-prefix anchors (CPU, reference-aware; `anchors`)

All anchors are defined on **actual beam-5 outputs**, never on reference text.

**Canonical units per output.** One Han character or one Latin word, from the canonical normaliser.

**Clean token boundary k:**
- the first k content tokens and the rest are both valid UTF-8;
- their units concatenate to exactly the output's units.

One Han character is never assumed to be one token.

**Mapping.** The canonical scorer `one(ref, hyp)` maps reference units to B0 hypothesis units.

**B0 anchors.** Each is the largest clean boundary whose prefix holds exactly the target number of units.

| Anchor | Prefix ends | Valid if |
|---|---|---|
| `start` | empty content prefix: the original decode, used for the replay-parity check | always |
| `pre_en` | before B0's unit for the first unit of the English run that precedes the switch-back (the run ends at the last substantive EN unit before the first span unit) | exact boundary; that reference unit is matched or substituted by B0 |
| `post_en` | before B0's unit for the first span unit (the switch-back location). The span units are the B0-correct damaged Mandarin units fixed in P2-1b; the P2-1b `sb_unit` itself can be a unit that B0 already had wrong, so it is recorded but not used as the location | exact boundary; B0 matches that unit |
| `early_zh` | the first clean boundary inside the span after at least the first two span units, with at least one span unit left (a multi-character Han token may straddle units) | B0 matches the first two span units. Only the span units after the boundary are scored |

If no exact boundary exists, the nearest earlier clean boundary is recorded as approximate, with its shortfall, and the
anchor is invalid for precise claims.

**Own-prefix anchor of each adapted system S (T3):**
- **Location.** The boundary after S's unit for the last reference unit before the first span unit that S matches
  or substitutes.
- **Validity.** The boundary must be exact, and that reference unit must lie in the English run, so that both prefixes
  end at the same switch-back location.
- **Also recorded:**
  - whether S's units before the anchor equal B0's units before `post_en` (lexically matched context);
  - whether the token prefixes are identical. If they are, the B0-versus-own comparison is void.

**Divergences of S from B0** (reference-free, kept separate):
- the first token divergence, which may be formatting or tokenization only;
- the first substantive lexical divergence, meaning the first differing canonical unit.

**Firewall outputs.**
- `anchors_private.json` holds all of the above.
- `anchors_inference.json` holds only case key, utterance id, prompt and token counts, plus the recorded P2-1b
  rescoring sums, which are reference-free numbers.
- The GPU steps read only `anchors_inference.json`, model outputs, the audio and the adapters.

## 4. Replay engine (GPU, firewalled)

**Decoder prefix:** `P_k = [SOT, LID, TRANSCRIBE, NOTIMESTAMPS, b_1 … b_k]`, where b are content tokens of an actual
beam-5 output.

**Free continuation.** `transformers` `GenerationMixin.generate`, which is the call Whisper's own `generate` makes
internally:
- the forced prefix is passed as `decoder_input_ids`;
- beam 5, 5 returned beams, length penalty 1.0, `early_stopping` false;
- absolute length budget 4 + 200 tokens (the canonical budget, unchanged for forced prefixes);
- the canonical `suppress_tokens` at every step;
- the canonical `begin_suppress_tokens` (space, EOS) **only at absolute position 4**.

**Why the begin-suppression is pinned.** Whisper's own `generate` would re-arm begin-suppression at the end of any
forced prefix, which would forbid EOS right after it. Pinning it at position 4 keeps the original semantics.

**Teacher-forced scores.**
- One full uncached forward, float32 log-softmax.
- The logit at position i scores token i + 1.
- Prefix and prompt tokens are never scored.
- EOS is scored exactly once, when present.

**Three score types, never conflated:**
- **Unconstrained:** full-vocabulary teacher-forced log-probabilities.
- **Beam-equivalent:** Σ log p / len^1.0 over the generated suffix including EOS. This is HF's finished-beam score. It
  is undefined when a suppressed token occurs.
- **Beam selection:** the actual beam-5 choice.

**LoReFT positions.** The adapter keeps its absolute-position semantics: it uses `cache_position` and applies the
intervention at positions ≥ 3.

## 5. T1 — replay correctness (engineering gate)

Checked per case in the smoke (d0, s0) and for all ten cases in the full run:
1. **Parity.** The empty-prefix replay of every system reproduces its recorded original beam-5 decode token-for-token
   (7 × 10).
2. **Beam versus teacher-forced score.** The top beam's `sequences_score` equals the teacher-forced beam-equivalent
   score of the same tokens within 0.05 (shifting, EOS and normalization).
3. **Recorded scores.** The P2-1b teacher-forced rescoring sums are reproduced within 1e-3 (checkpoint reload).
4. **Cached versus full.** Incremental KV-cached decoding versus one full forward for B0, CE42 and KL42:
   - |Δ log p| of the observed tokens ≤ 0.10;
   - argmax agreement wherever the full-forward top-2 margin is ≥ 0.20;
   - finite logits;
   - layer-31 `cache_position` equals the absolute positions (full forward 0..T−1; cached steps 0..3, then t).
5. **Early positions.** Logits at positions 0–2 are bitwise equal with and without each adapter.
6. **Determinism.** A repeated forced-prefix beam continuation is identical.
7. **Hygiene.**
   - no adapter hooks remain;
   - no cross-case leakage (the first case is re-run at the end and must be identical);
   - backbone weights unchanged;
   - adapter hashes match.

A full forced-prefix beam continuation is **not** required to reproduce the remainder of the original winning sequence,
because the original search had competing beams. Parity is required only at the empty prefix, which is the original
search.

## 6. T2–T5 (one GPU job)

For each case, both of the following, each measured as next-token statistics plus a beam-5 continuation:
- each valid-or-recorded B0 anchor (`start`, `pre_en`, `post_en`, `early_zh`), for all seven models;
- each own-prefix anchor of S, for S, B0 and S's matched counterpart (CE_s ↔ KL_s).

**Next-token statistics:**
- log p, rank and margin of the source output's next token;
- EOS log p;
- top-5;
- at own prefixes, additionally B0's switch-back token.

**T5 candidate pools.** At the B0 `pre_en` and `post_en` prefixes, the pool is:
- B0's own continuation;
- every beam generated from that exact prefix by any of the seven models.

There is no splicing of suffixes onto other prefixes. Every candidate is scored under all seven models.

## 7. Scoring (CPU, reference-aware; `analyze`)

- **Continuation hypothesis:** the source prefix plus the top beam suffix, scored with the canonical scorer.
- **Span retention:** the matched fraction of the case's span units (for `early_zh`, units 3 onward only).
  - recovered ≥ 0.8;
  - dropped ≤ 0.2;
  - partial otherwise.
- **Also scored:**
  - new errors in the generated region (B0-correct reference units from the anchor's reference unit onward that are
    wrong);
  - premature EOS (terminated with < 50% of B0's remaining units);
  - loops (canonical);
  - identity to B0;
  - English-run correctness.
- **Forced-prefix recovery (T4)** is scored on the newly generated suffix only. The supplied prefix never counts as
  recognition.

**Interpretation.**
- The rescue boundary is the earliest tested anchor at which the span is recovered. It is the boundary under these
  interventions, not the original commitment point.
- The flags, cells, verdict and recommendation follow `P2_1C_DECISION_RULES.md`.

## 8. Gates and stopping

- **G1 provenance:**
  - branch;
  - P2-1b → P2-0 chains (CPU steps);
  - input and adapter hashes;
  - weights;
  - no locked roles.
- **G2 anchors:**
  - all ten cases accounted;
  - the GPU manifest is reference-free;
  - at least 3 damaged cases have a valid `post_en` anchor.
- **G3:** the T1 checks.
- **G4:** separation of format and lexical divergence and of score types; this is static and tested.
- **G5 compute:**
  - no training;
  - smoke-based estimate ≤ 170 min (hard), target ≤ 90 min;
  - every job < 3 h.
- **G6:** freeze, tests, determinism, write-once artifacts.
- **G7:** rules frozen before execution.

All gates pass → `READY_FOR_P2_1C_EXECUTION`. Otherwise `P2_1C_BLOCKED_PREFLIGHT_<G>`, and nothing further runs.

**Engineering failure or budget overrun.** Partial artifacts are kept, and the run stops. Any post-freeze repair is a
documented, versioned amendment.

## 9. Claim boundaries

Ten cases (five failures) on 20 dev dialogues, with reference-informed anchors. This is descriptive operational
evidence about fixed models under valid prefix interventions. It does **not** establish:
- generalization;
- an exact causal commitment point;
- acoustic grounding or its absence;
- that any proposed P2-2 method will work.
