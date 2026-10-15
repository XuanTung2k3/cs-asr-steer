# D1 — region-weighted GoldKL pilot (protocol)

**Status:** pre-registered before any D1 training outcome existed. It is frozen in `d1_regionkl/freeze.json`.

**Files:**
- Code: `reft/d1_regionkl_train.py`, `reft/d1_regionkl_evaluate.py`.
- Configuration: `configs/reft/d1_regionkl_frozen.yaml`.
- Tests: `tests/reft/test_d1_regionkl.py`.
- Slurm: `slurm/reft_d1_regionkl*.sbatch` and `slurm/reft_d1_regionkl_submit.sh`.
- Rules: `D1_REGIONKL_DECISION_RULES.md`.

**Type.** A bounded objective experiment on the unchanged R2-LoReFT (5,122 parameters). The exposed dev150 panel both
selects checkpoints and evaluates them, so every result is exploratory screening, not confirmation.

**Out of scope:** D2 (edit locality), region-weight or KL-coefficient sweeps, R3/M1 or any rank, layer or adapter
change, on-policy training, LoRA, E1+D2, P2-3, P3, independent test evaluation, retraining of historical controls, and
any experiment after the D1 verdict.

## 1. Question and hypotheses

Under a fixed low-rank representation-adaptation budget, does redistributing GoldKL preservation pressure toward
teacher-correct Mandarin positions right after English→Mandarin switches improve the English-correction versus
Mandarin-preservation trade-off?

| Hypothesis | Meaning |
|---|---|
| H1 | Region-specific preservation pressure improves the trade-off |
| H2 | Reweighting suppresses useful English corrections |
| H3 | Gold-prefix preservation cannot repair damage caused by generated-prefix divergence (P2-1c) |
| H4 | Most damage-critical positions are poorly covered by the teacher-correct mask |
| H5 | A few dialogues or seeds dominate the observed effects |

Near-switch concentration of the remaining GoldKL damage (P2-1b, P2-1D) motivates D1. It does not show that GoldKL
under-weights those positions; that is what D1 tests.

## 2. Fixed model and recipe (inherited, unchanged)

**Adapter.** R2-LoReFT at decoder layer 31, post-cross-attention residual, pre-FFN, `reft/adapters.py` unchanged:
h̃ = h + Rᵀ(W h + b − R h), rank 2, d = 1280, orthonormal rows of R (torch orthogonal parametrization), W = R and b = 0
at init, `start_pos` 3. Backbone Whisper-large-v3 frozen in BF16, adapter in float32. Frozen Auto-LID; beam-5 primary,
greedy secondary.

**Training (P2-1b):**

| Setting | Value |
|---|---|
| Data | 500 frozen-manifest training utterances (`p2_0/train_manifest.json`) |
| Seeds | 42, 43, 44 (seed semantics as P2-1b: `torch.manual_seed(s)` → backbone → adapter; order `default_rng(s + epoch)`) |
| Epochs, micro-batch, accumulation | 3, 8, 2 (final pending micro-batch of an epoch also steps: 32 updates/epoch, 96 total) |
| Optimizer | AdamW, lr 5e-4, betas (0.9, 0.99), eps 1e-6, weight decay 0, constant schedule, clip 1.0 |
| λ_KL | 1.0 |

## 3. Objective

P2-1b GoldKL is

L_GoldKL = (1 / |M|) Σ_{t∈M} KL(p₀,t ‖ p_θ,t),  M_t = 1[Y_t ≠ −100] · 1[argmax_v z₀,t,v = Y_t].

Teacher and student see the same audio, frozen-LID prompt and gold-prefix decoder inputs. The teacher is the same
forward with the adapter absent, detached.

D1 changes only the KL aggregation:

L_D1 = CE_all + λ_KL · L_RegionKL,  L_RegionKL = Σ_t M_t w_t KL(p₀,t ‖ p_θ,t) / Σ_t M_t w_t,

with w_t = 2 on the switch-back region R_M^SB (§4) and 1 elsewhere, λ_KL = 1.

**Reduction (exactly P2-1b's):**
- CE is the token-mean over content tokens + EOS, unchanged.
- KL is over the full vocabulary in float32, direction KL(p₀ ‖ p_θ), teacher logits detached.
- The weights are a constant float32 tensor (no gradient). They apply only to masked token-level KL values.
- The denominator is the sum of masked weights *within the micro-batch*, divided as a Python scalar (an exact integer).
  There is no dataset-level normalization and no per-language normalization.
- An empty mask contributes a differentiable zero and is counted.
- The total loss is divided by the accumulation factor 2.

With every weight equal to 1 the arithmetic is P2-1b `kl_masked` operation for operation (same products, same
reduction, division by the same scalar); preflight G3 requires bitwise equality.

**Interpretation.** This is a redistribution of the normalized KL mass, not an added Mandarin KL term. With N masked
positions of which K are region positions, per-token weights relative to GoldKL are 2N/(N+K) in the region and
N/(N+K) elsewhere. So D1 *lowers* the per-token preservation pressure outside the region relative to GoldKL. The
normalization does not equalize loss values or gradient norms.

## 4. Training region R_M^SB (training references only)

Built once by `regions` (CPU) from `p1/targets.json` (`target_text`, `target_ids`) of the 500 training utterances,
hashed and frozen. No dev reference, dev error, P2-1D example or locked role is read.

1. **Units:** canonical reference units of `target_text` (`reft.text.units`: one Han character or one Latin word).
2. **Language sequence:** substantive EN = `tag_unit` EN and not a filler; substantive ZH = `tag_unit` ZH and not a
   filler (filler lists of `configs/reft/p0p1.yaml`, as in the canonical evaluator). Every other unit (fillers, OTHER
   such as bare numerals, UNKNOWN such as mixed-script units) is skipped: it is not counted, does not end a run, and is
   never weighted.
3. **Switch-back:** a substantive ZH unit whose previous entry in the language sequence is substantive EN.
4. **Window:** the first 7 substantive ZH units of the Mandarin run starting at that switch-back, ending earlier at the
   next substantive EN unit. Shorter runs give shorter windows. Each switch-back in an utterance has its own window.
5. **Token mapping:** `reft.text.token_unit_map` maps each content token (EOS excluded) to the canonical units whose
   characters it touches; the token bytes must reconstruct `target_text` exactly, else the utterance is unmappable.
6. **Weighting rule (conservative):** token t gets weight 2 iff its unit set is non-empty and lies inside one window.
   A token that touches a window unit and anything outside the windows keeps weight 1 and is recorded as ambiguous:
   - `language_boundary`: it also touches an EN unit;
   - `window_end`: it also touches a ZH unit beyond the window (e.g. across the 7th/8th unit);
   - `nonlexical`: it also touches a filler, OTHER or mixed-script unit;
   - `multi_window`: it spans two windows.
   Never upweighted to increase coverage.
7. **Decoder shift:** target token t is predicted at decoder position 3 + t (`Y[3 + t] = target_ids[t]`), and the
   weight is placed there. `weight_tensor` asserts the label at every weighted position. EOS, prompt and padding
   positions are never weighted.
8. **Teacher mask:** applied at loss time; the region only reweights teacher-correct positions.

**Frozen edge-case policy:**

| Case | Handling |
|---|---|
| Whitespace | `target_text` has single spaces only between Latin words; a leading-space token maps to its word; a pure-space token maps to no unit (weight 1, "unmapped") |
| Punctuation | removed by `target_text` (P2-1 contract), between or inside runs alike |
| Fillers (ZH or EN) | skipped in the language sequence; never weighted; a token touching a window unit and a filler is ambiguous |
| Special tokens, EOS | never weighted (labels at prompt positions are −100; EOS index excluded and asserted) |
| Mixed-script units | tagged UNKNOWN; skipped; tokens touching them and a window unit are ambiguous |
| Token spanning several units | weighted only if every touched unit is in the same window |
| Token spanning the 7th-unit boundary | ambiguous `window_end`, weight 1 |
| Token spanning a language boundary | ambiguous `language_boundary`, weight 1 |
| Partial UTF-8 bytes of one Han character | each piece maps to that unit; weighted if the unit is in a window |
| Unmappable utterance | all weights 1, recorded; > 2% unmappable utterances blocks execution |
| Empty region / no switch-back | all weights 1 |
| Multiple switches | one window per switch-back |
| Runs shorter than 7 units | the window ends at the run end |

**Integrity:** the manifest `target_ids` must equal `p1/targets.json`, each must end in exactly one EOS, and the
preflight re-derives every region from scratch with the real tokenizer and compares.

**Coverage report (training only):**
- supervised positions and teacher-correct positions;
- substantive EN/ZH units and switch-back opportunities;
- tokens touching window units, unambiguous region tokens, and teacher-correct region tokens;
- region coverage (numerator and denominator);
- utterances and dialogues with active weights;
- ambiguous tokens by reason, and window units without an unambiguous token;
- EOS coverage;
- the per-micro-batch distribution of region tokens (all nine seed × epoch orders before the mask; teacher-correct ones
  in manifest order).

First-switch-back-token coverage (P2-1b's measure) is reported separately from first-7-unit-region coverage.

**Engineering floor:** teacher-correct coverage of unambiguous region tokens ≥ 10% and teacher-correct region tokens in
≥ 2 training dialogues. This is a feasibility floor, not a sufficiency claim. Low or concentrated coverage is reported
as a limitation even when the floor passes.

## 5. Preflight gates (GPU; terminal verdict `D1_READY_FOR_TRAINING` or `D1_BLOCKED_<gates>`)

- **G1 — provenance and data firewall:**
  - branch `cs-asr-reft`, clean tree, freeze verified;
  - S0/S1 chain verified, which verifies P2-0, P2-1b, P2-1c and P2-2;
  - canonical CUDA 12.8 environment;
  - backbone weight hash;
  - pinned B0 hashes and train/dev manifest hashes;
  - R2-GoldKL adapters match their recorded hashes;
  - data audit valid with no locked dialogues;
  - no other running reft job;
  - write-once output directories;
  - the GPU firewall is raised before any GPU work.
- **G2 — region alignment:**
  - the frozen region hash is unchanged and the region metadata re-derives exactly;
  - every weighted token is a non-EOS, non-Latin (Han) token;
  - the decoder shift is checked on real batches (label at 3 + t, input = previous token);
  - EOS and prompt positions are unweighted;
  - unmappable utterances ≤ 2%, and targets match the manifest;
  - the coverage floor holds.
  - Synthetic cases (simple, multiple, short, > 7, multi-unit tokens, boundaries, fillers, no switch-back, unmappable,
    decoder shift) are unit tests run at the start of every job.
- **G3 — GoldKL equivalence (mandatory):**
  - With all weights 1, the D1 step equals P2-1b `kl_step` bitwise at zero init and with a perturbed adapter: loss, CE,
    KL, token count, mask tensor, teacher and student logits, and every adapter gradient.
  - Initial adapter states are identical (including the parametrization `base`), and CE equals the P2-1 CE path.
  - The zero-init D1 step with real regions has KL ≤ 1e-6 and student = teacher.
  - **Historical replay:** one full seed-42 epoch with all weights 1 reproduces
    `p2_1b/training/KL_seed42/epoch1/adapter.pt` bitwise. It must also match train KL, train CE, mask positions,
    supervised tokens and the 32 updates. This bounded parity test is not a new arm.
  - If it fails, D1 stops and the report states what matched control would be needed.
- **G4 — weighting correctness** (perturbed adapter, training rows with region tokens):
  - region positions are active;
  - the denominator equals N + K;
  - the weighted KL equals an explicit weighted mean of the same float32 per-token KL (relative tolerance 1e-6), and is
    within 1e-3 relative of a float64 recomputation (float32 precision on KL values of ~1e-4 nats);
  - the region factor 2N/(N+K) is > 1;
  - teacher-incorrect positions are excluded, and weights placed there have no effect;
  - an empty mask gives a differentiable zero;
  - KL ≥ −1e-6, finite, with the teacher and the weights detached;
  - loss and gradients differ from GoldKL, every R, W and b receives gradient, and no backbone gradient appears.
  - Identity init is not used for the gradient check (both objectives are ≈ 0 there).
- **G5 — model and decoding integrity:**
  - 5,122 trainable parameters, W = R, b = 0, orthogonality ≤ 1e-5;
  - zero init and the disabled adapter reproduce frozen teacher-forced logits bitwise; positions 0–2 untouched;
  - checkpoint save/reload gives bitwise logits;
  - zero-init dev 20/20 beam-5 and greedy equal the pinned B0;
  - frozen LID unchanged under a perturbed adapter (20/20);
  - absolute cache positions (greedy and beam), cached greedy vs teacher-forced argmax ≤ 1 mismatch;
  - no X–Y–X state leakage, deterministic repeats, hooks removed;
  - float32 cached-vs-full ≤ 1e-3 nats (P2-1c amendment r1; BF16 recorded descriptively).
- **G6 — resources:**
  - CE / GoldKL / D1 micro-batch timing and the region-weighting overhead;
  - weight-tensor build time and region-mapping time;
  - peak allocated/reserved memory;
  - projected job ≤ 9,000 s (3 h allocation).
- **G7 — protocol freeze:** freeze verified, authorization present, clean tree, git head equal to the frozen head.

If any gate fails, D1 stops without formal training. Post-freeze engineering corrections are versioned amendments
(`freeze-amend`) with their consequences stated.

## 6. Formal runs

Exactly three runs, `R2-RegionKL` seeds 42, 43 and 44: 3 epochs and 96 updates each, region weight 2, λ_KL 1, no
other loss, and the same epoch order as the historical GoldKL run of the same seed. The historical
`p2_1b/training/KL_seed{s}` runs are the matched controls, used only after G3 passes. They are never retrained.

**Per run, the following are recorded:**
- the zero-init identity (150/150 vs B0) and the initial `base` equal to the GoldKL run of the same seed;
- per micro-batch: CE, RegionKL, the *diagnostic* unweighted GoldKL (never in the objective), the total loss, the
  teacher-correct count, weighted-region tokens (all and masked), the weight sum, and the region/non-region KL
  contributions;
- per step: gradient norms, update and parameter norms, learning rate and orthogonality;
- per epoch: pairing with the GoldKL run (mask positions, supervised tokens, updates);
- teacher-forward and training seconds, memory, and checkpoint hashes;
- checkpoints and beam-5 / greedy dev decodes after epochs 1–3.

## 7. Selection and evaluation

**Selection:** the unchanged P2-1 rule per run (lowest beam-5 genuine-CS MER, then overall MER, then Mandarin CER, then
earliest epoch). Results are reported for the selected checkpoints (primary) and for fixed epoch 3 (no selection,
sensitivity). Mandarin retention, switch-back measures, English corrections, RegionKL and representation diagnostics
never select.

**Scorer:** canonical P2-1b/P2-1D/P2-2. GoldKL is recomputed and must reproduce `p2_1b/evaluation.json`.

**Outcomes:**
- **Primary:** genuine-CS MER, overall MER, English WER/PIER, Mandarin CER, new Mandarin errors and repairs, Mandarin
  retention (denominator 4,950 B0-correct units), and English corrections, corruptions and net.
- **Target window (TW):** dev-reference units in the first 7 substantive Mandarin units of each Mandarin run after an
  EN→ZH switch, by the same language-sequence rule. Reported: eligible units, B0-correct units, new errors among them
  (deletions, substitutions), repairs, and per-seed and per-dialogue values.
  - The training teacher-correct mask (teacher-forced, training data) and the evaluation B0-correct TW units (free
    decoding, dev) are different populations with different denominators.
- **Broader damage:**
  - long switch-back dropped units (runs ≥ 5 starting after substantive English) and newly dropped spans;
  - new Mandarin deletions; near-switch (≤ 7) and far damage;
  - damage without an English repair; outside-target damage;
  - loops, non-termination, worst-dialogue mixed-error regression.
  - TW and long SB overlap; they are not independent confirmations.
- **English specificity:** omission → correct, translation-like → correct, lexical → correct, correct → wrong, and
  wrong → different wrong.

**Statistics:**
- Per seed, Δ_s = D1_s − GoldKL_s, with three-seed means and ranges.
- Per-dialogue contributions, leave-one-dialogue-out ranges and worst-dialogue regressions.
- 2,000-replicate dialogue-cluster bootstrap with seed 42. The same dialogue draws are used for D1 and GoldKL and across
  seeds, and metrics are recomputed from resampled dialogue counts.
- The three seeds share the 20 dialogues; intervals are descriptive, and nothing is confirmatory.

## 8. Representation diagnostics (bounded, training utterances only; no training)

At the intervention site, with teacher-forced gold prefixes on all 500 training utterances, for the GoldKL and D1
checkpoints (selected and epoch 3) of every seed:
- ‖Δh‖ and ‖Δh‖/‖h‖, and token-level KL(p₀ ‖ p_θ), by training-token class: switch-back region, other Mandarin, English
  and other. Classes come from the training references, for all positions and for teacher-correct positions.
- Per-position cosine between the GoldKL and D1 edits, and D1/GoldKL edit-norm ratios by class. Selectivity is the
  region ratio divided by the English or other-Mandarin ratio.
- Principal-angle cosines between the R row spaces, ‖b‖ and ‖W − R‖.

These are descriptive. Smaller edits do not imply better preservation; gold-prefix diagnostics do not explain
generated-prefix omissions causally.

## 9. Outputs and data handling

**Private** (`/mnt/data/tungnx/cs-asr-reft/d1_regionkl/`):
- `AUTHORIZATION.json`, `freeze*.json`, `preflight_*.json`, `region_coverage_*.json`;
- `regions_private.jsonl`, `region_static.json`;
- `training/D1_seed{42,43,44}/`;
- `evaluation.json`, `verify.json`, `mechanism_private.json`, `report.json`, `manifest.json`, `figures/`.

**Public:** code, tests, configs and sanitized aggregate documents only. No transcripts, utterance or dialogue IDs,
audio paths, checkpoints or raw decodes.

Compute is via `sbatch` only. Logs go to `/mnt/data/tungnx/slurm_logs`, and `TMPDIR` is under `/mnt/data/tungnx/tmp`.

## 10. Engineering record before the freeze

Pre-freeze dry runs use `D1_DRY_ROOT` (scratch outputs, no freeze; G1/G7 fail by construction). Any change made after a
dry run is listed here before the freeze.

**Dry run 1 (jobs 59551 regions, 59552 preflight; scratch, outputs deleted afterwards).**
- G2, G3, G5 and G6 passed. G3 included the bitwise historical replay of KL_seed42 epoch 1 with all weights 1.
- G1 and G7 failed only because a dry run has no freeze.
- G4 failed only `manual_matches`: the float64 recomputation of the weighted KL differed from the float32 loss by 7.6e-5
  relative, against a planned 1e-5 tolerance.
  - **Cause:** float32 log-softmax precision on KL values of ~2e-4 nats, not the weighting. The same float64
    comparison applies equally to unweighted GoldKL.
  - **Change (before the freeze):** the check is split. (a) The exact aggregation is checked on the same float32
    per-token KL at relative tolerance 1e-6. (b) A float64 cross-check is applied at 1e-3, and the unweighted GoldKL
    comparison is recorded for scale.
  - The region definition, objective, recipe and decision rules were not changed.
- Region counts observed in this dry run (reported in the result) did not lead to any change of the region definition.

**Dry run 2 (job 59553; same scratch root).** All gates passed except G1 and G7 (no freeze, by construction).
- G4: the exact aggregation matches to 1e-9 relative. The float64 relative differences are 7.6e-5 (weighted) and 2.7e-5
  (unweighted GoldKL).
- G3: the historical replay is again bitwise.
- The scratch outputs of both dry runs were deleted.
