# D2 — representation edit-locality pilot (protocol)

**Status:** pre-registered before any D2 calibration or training outcome existed. It is frozen in
`d2_edit_locality/freeze.json`; the coefficient μ is frozen afterwards by the D2.0 rule in `d2_edit_locality/mu_freeze.json`.

**Files:**
- Code: `reft/d2_edit_locality_train.py`, `reft/d2_edit_locality_evaluate.py`.
- Configuration: `configs/reft/d2_edit_locality_frozen.yaml`.
- Tests: `tests/reft/test_d2_edit_locality.py`.
- Slurm: `slurm/reft_d2_edit_locality*.sbatch` and `slurm/reft_d2_edit_locality_submit.sh`.
- Rules: `D2_EDIT_LOCALITY_DECISION_RULES.md`.

**Type.** A bounded objective experiment on the unchanged R2-LoReFT (5,122 parameters). The exposed dev150 panel both
selects checkpoints and evaluates them, so every result is exploratory screening, not confirmation.

**Out of scope:** D1 reweighting or its sweeps, more than one locality coefficient or any coefficient chosen on dev, D1+D2,
R3/M1 or any architecture, layer or rank change, prefix-aware objectives, LoRA, the locked roles (D-test, D-dev-confirm,
router-calib), modification of earlier frozen experiments, independent test evaluation, retraining of historical controls,
and any experiment after the D2 verdict.

## 1. Question and hypotheses

D1 (`D1_NO_INCREMENTAL_GAIN`) lowered teacher-forced switch-back KL by about 17–21% without a reliable free-decoding gain,
and its edits stayed almost collinear with GoldKL's. D2 tests a different, representation-level lever: does directly
penalizing the intervention's edit energy at reliable (teacher-correct) Mandarin positions improve the free-decoding
correction–preservation trade-off beyond R2-LoReFT + GoldKL?

| Hypothesis | Meaning |
|---|---|
| H1 | Selective Mandarin edit suppression improves preservation while retaining English corrections |
| H2 | D2 behaves like generic intervention attenuation and sacrifices useful corrections |
| H3 | Edit norms decrease but free-decoding errors remain |
| H4 | Generated-prefix-history sensitivity dominates; gold-prefix edit regularization is insufficient |
| H5 | Seed/dialogue variation dominates any observed improvement |

**Distinctions from D1.** D1 redistributed the normalized KL mass (output distributions) towards switch-back windows.
D2 keeps GoldKL exactly and adds a separate term on the hidden-state edit itself, over *all* teacher-correct Mandarin
positions (not only switch-back windows). The edit term acts on the intervention's magnitude at those positions; whether
it changes the direction of the edits is measured, not assumed.

## 2. Fixed model and recipe (inherited, unchanged)

**Adapter.** R2-LoReFT at decoder layer 31, post-cross-attention residual, pre-FFN, `reft/adapters.py` unchanged:
h̃ = h + Δh, Δh = Rᵀ(W h + b − R h), rank 2, d = 1280, orthonormal rows of R (torch orthogonal parametrization), W = R and
b = 0 at init, `start_pos` 3. Backbone Whisper-large-v3 frozen in BF16, adapter in float32. Frozen Auto-LID; beam-5 primary,
greedy secondary.

**Training (P2-1b):**

| Setting | Value |
|---|---|
| Data | 500 frozen-manifest training utterances (`p2_0/train_manifest.json`) |
| Seeds | 42, 43, 44 (`torch.manual_seed(s)` → backbone → adapter with init seed s; order `default_rng(s + epoch)`) |
| Epochs, micro-batch, accumulation | 3, 8, 2 (final pending micro-batch of an epoch also steps: 32 updates/epoch, 96 total) |
| Optimizer | AdamW, lr 5e-4, betas (0.9, 0.99), eps 1e-6, weight decay 0, constant schedule, clip 1.0 |
| λ_KL | 1.0 |

## 3. Objective

L_D2 = L_CE + L_GoldKL + μ · L_edit, per micro-batch, divided by the accumulation factor 2.

- **L_CE:** token-mean CE over content tokens + EOS (unchanged).
- **L_GoldKL:** P2-1b `kl_masked`, unchanged: (1/|M|) Σ_{t∈M} KL(p₀,t ‖ p_θ,t), M = teacher-correct mask (Y ≠ −100 and the
  frozen teacher's argmax equals Y, content tokens and EOS), full vocabulary, float32, teacher detached.
- **L_edit** = (1/|R_M|) Σ_{t∈R_M} ‖Δh_t‖² / 1280 over the micro-batch, with R_M = M ∩ (Mandarin-positive token positions,
  §4). An empty R_M contributes a differentiable zero and is counted.
- **μ:** one coefficient for all three seeds, fixed by the D2.0 rule (§5) before formal training.

**The edit tensor.** `LoReFTAdapter._attn` computes `delta = module(h.float()) * mask` (mask = absolute position ≥ 3) and
adds it, cast to BF16, to the cross-attention output. D2 registers a forward hook on the intervention module itself
(`EditObserver`). It reads the module's float32 output in the same forward, applies the identical position mask and records
the per-position squared norm. Consequences, each tested:
- no second adapted forward, no extra adapter call (one module call per adapted forward, asserted every micro-batch);
- the hook returns nothing, so logits are bitwise unchanged;
- gradients of L_edit flow through the same tensor into R (through the parametrization), W and b;
- records are cleared on attach/detach and exactly one record is consumed per micro-batch (no leakage); only a [B, T]
  energy tensor is kept;
- decoder positions are the absolute `cache_position`s (asserted equal to 0..T−1 in teacher forcing); target token t is at
  position 3 + t.

The energy is that of the float32 edit before the BF16 cast; the BF16 cast's effect on the energy is recorded descriptively.

**Rank-2 identity (tested):** with RRᵀ = I and a = (W − R)h + b, Δh = Rᵀa and ‖Δh‖² = ‖a‖². This says nothing about the
edit direction; smaller edit energy does not imply a better edit.

**μ = 0** reproduces P2-1b GoldKL bitwise (loss, logits, CE, KL, mask, gradients), with the observer installed (§6, G4).

**Not changed:** GoldKL is not replaced by D1 RegionKL, CE supervision is not reduced, no English-specific or switch-back
weights, no dev labels in any mask, and the locality term never covers all tokens.

## 4. Edit mask R_M (training references only)

Built once by `mask` (CPU) from `p1/targets.json` (`target_text`, `target_ids`) of the 500 training utterances, hashed and
frozen. No dev reference, dev error or locked role is read.

1. **Units:** canonical reference units of `target_text` (`reft.text.units`: one Han character or one Latin word).
2. **Labels:** substantive ZH = `tag_unit` ZH and not a filler; substantive EN = `tag_unit` EN and not a filler; fillers,
   OTHER (e.g. bare numerals) and UNKNOWN (mixed-script) units are non-lexical (filler lists of `configs/reft/p0p1.yaml`).
3. **Token mapping:** `reft.text.token_unit_map`; token bytes must reconstruct `target_text`, else the utterance is
   unmappable.
4. **Mandarin-positive token:** a content token (EOS excluded) whose unit set is non-empty and consists only of substantive
   ZH units.
5. **Teacher mask:** applied at loss time per micro-batch. R_M = Mandarin-positive ∩ teacher-correct.
6. **Decoder shift:** the token-t mask is placed at decoder position 3 + t; `token_mask_tensor` asserts the label there.

**Frozen edge-case policy:**

| Case | Handling |
|---|---|
| Multi-unit token | in the mask only if every touched unit is substantive ZH |
| Token spanning ZH and EN units | `ambiguous_language_boundary`: excluded, recorded |
| Token touching a ZH unit and a filler / OTHER / mixed-script unit | `ambiguous_nonlexical`: excluded, recorded |
| Filler-only, OTHER-only, EN-with-non-lexical tokens | `other`: excluded |
| Partial UTF-8 bytes of one Han character | each piece maps to that unit: Mandarin-positive |
| Pure-space token | maps to no unit (`unmapped`): excluded |
| Punctuation | removed by `target_text` (P2-1 contract) |
| Mixed-script units | UNKNOWN, non-lexical |
| EOS, prompt, padding | never in the mask |
| Unmappable utterance | no mask positions, recorded; > 2% unmappable blocks execution |
| Empty micro-batch mask | L_edit = differentiable zero, counted |

Excluded positions keep their CE and GoldKL contributions unchanged. The GoldKL teacher-correct mask is not modified.

**Diagnostics only:** Mandarin-positive tokens inside D1's switch-back windows (first 7 substantive ZH units after an EN→ZH
switch) are tagged ZH_SB; the rest ZH_other. EN tokens (all units substantive EN) are tracked for edit-norm diagnostics.

**Coverage report (training only):** supervised and teacher-correct positions; Mandarin-positive tokens and their
teacher-correct share; switch-back vs non-switch-back Mandarin; ambiguous and excluded tokens by reason; EOS coverage;
utterances and dialogues represented; per-micro-batch counts (all nine seed × epoch orders before the teacher mask,
manifest order after it).

**Engineering floor (G3):** teacher-correct coverage of Mandarin-positive tokens ≥ 10%, edit-mask positions in ≥ 2
training dialogues, and, in each of the nine seed × epoch training orders, ≤ 5% of micro-batches without any teacher-correct
edit-mask position (manifest-order counts are reported, not gated; see §12).

## 5. D2.0 — one-coefficient gradient-scale calibration (before formal training)

At R2 identity Δh = 0, so L_edit = 0 and ∇L_edit = 0; μ is never calibrated there.

- **State:** the historical R2-GoldKL seed-42 epoch-1 checkpoint (`p2_1b/training/KL_seed42/epoch1/adapter.pt`, sha256
  checked against its run manifest). It is a training-only, deterministic, non-identity R2 state; no dev outcome chose it.
- **Batches:** training-manifest rows sorted by sha256(utterance_id) hex digest; the first 64 rows form 8 micro-batches of 8
  in that order. Fixed before any calibration value is seen.
- **Gradients:** per batch b, one forward; g_base,b = ∇_θ(L_CE + L_GoldKL) and g_edit,b = ∇_θ L_edit, raw (pre-clipping,
  no accumulation scaling), over exactly the trainable R2 tensors (R's parametrization original, W, b), as the norm of the
  concatenated vector.
- **Rule:** μ = ρ · median_b‖g_base,b‖ / median_b‖g_edit,b‖, ρ = 0.20. This is a gradient-scale heuristic, not an optimality
  claim.
- **Reproducibility:** the whole calibration is computed twice in the same job; |μ₂ − μ₁| ≤ 1e-5 μ₁.
- **Stop checks (all required, else `D2_CALIBRATION_BLOCKED_<checks>` and no training):**
  - all norms finite; every ‖g_edit,b‖ > 1e-12 and ‖g_base,b‖ > 0;
  - every calibration batch has ≥ 1 edit-mask position;
  - ratio spread max_b r_b / min_b r_b ≤ 10, r_b = ‖g_base,b‖ / ‖g_edit,b‖;
  - no pathological dominance: max_b μ‖g_edit,b‖ / ‖g_base,b‖ ≤ 1.0;
  - at identity (seed-42 init, first calibration batch) L_edit = 0 and ∇L_edit = 0 exactly.
- **Reported:** per-batch ‖g_base‖, ‖g_edit‖, cosine(g_base, g_edit), edit-mask counts, L_edit, μ·L_edit / (CE + KL), the
  scaled ratios, and μ.
- **Freeze:** only if every check passes, the job writes write-once `mu_freeze.json` (μ, ρ, calibration hash, freeze record
  hash). Every later job verifies it. An inconvenient μ is never clipped or replaced without an amendment and explicit
  authorization.
- **Formal runs** start from their own seed-specific R2 identity initialization, never from the calibration state (asserted:
  W = R, b = 0, and the parametrization `base` equals the GoldKL run of the same seed).

## 6. Preflight gates (GPU; terminal verdict `D2_READY_FOR_TRAINING` or `D2_BLOCKED_<gates>`)

- **G1 — provenance:** branch `cs-asr-reft`, clean tree, D2 freeze and μ freeze verified; D1 chain verified (which verifies
  S0/S1, P2-0, P2-1b, P2-1c, P2-2); canonical CUDA 12.8 environment; backbone weight hash; pinned B0 and manifest hashes;
  R2-GoldKL adapters match their recorded hashes; data audit valid with no locked dialogues; no other running reft job;
  write-once output directories; GPU firewall raised before GPU work.
- **G2 — site and edit correctness** (perturbed adapter, training rows):
  - exactly one observer record and one adapter call per forward; records cleared; no module hook left;
  - observing leaves logits bitwise unchanged; adapted logits equal the frozen forward with the manually recomputed delta
    injected at the P1 site (single application, bitwise);
  - observer energy equals the manual delta's energy bitwise; rank-2 identity within 1e-4 relative; zero energy at
    positions 0–2; absolute positions;
  - L_edit equals an explicit masked mean (1e-6 relative); R_M ⊆ teacher-correct; EOS and prompt excluded; empty mask gives a
    differentiable zero;
  - the L_edit gradient reaches R, W and b (finite, non-zero), no backbone gradient, orthogonality ≤ 1e-5.
- **G3 — Mandarin mask:** frozen hash; re-derivation with the real tokenizer equal; every mask token is non-EOS Han-only;
  decoder shift on real batches (label at 3 + t, input = previous token); EOS and prompt never in the mask; unmappable ≤ 2%;
  targets equal the manifest; coverage floor (§4).
- **G4 — GoldKL parity (μ = 0, observer installed):** D2 step = P2-1b `kl_step` bitwise at zero init and perturbed (loss,
  CE, KL, token count, mask tensor, teacher and student logits, every adapter gradient); identical initial states (incl.
  `base`); CE equals the P2-1 path; zero init with the frozen μ gives KL ≤ 1e-6, L_edit = 0 and student = teacher.
  **Historical replay:** one full seed-42 epoch with μ = 0 reproduces `p2_1b/training/KL_seed42/epoch1/adapter.pt` bitwise,
  with equal train KL, train CE, mask positions, supervised tokens and 32 updates. If it fails, D2 stops.
- **G5 — locality gradient:** at identity L_edit = 0 and its gradient is exactly zero; at the calibration state L_edit > 0
  and its gradient is finite and non-zero; with the frozen μ the scaled gradient ratio μ‖g_edit‖/‖g_base‖ ≤ 1.0 on a
  separate training batch; μ·L_edit / (CE + KL) recorded; the calibration record is OK and its μ equals the frozen μ.
- **G6 — decoding and resources:** 5,122 trainable parameters; W = R, b = 0; orthogonality; zero init and the disabled
  adapter reproduce frozen logits bitwise; positions 0–2 untouched; save/reload bitwise; zero-init dev 20/20 beam-5 and
  greedy equal the pinned B0; frozen LID unchanged under a perturbed adapter; absolute cache positions (greedy, beam);
  cached greedy vs teacher-forced ≤ 1 mismatch; no X–Y–X leakage; deterministic repeats; hooks removed; float32
  cached-vs-full ≤ 1e-3 nats (BF16 descriptive); CE / GoldKL / D2 micro-batch timing; projected job ≤ 9,000 s; peak memory
  < 38 GB.
- **G7 — protocol freeze:** freeze and μ verified, authorization present, clean tree, git head equal to the frozen head.

Post-freeze engineering corrections are versioned amendments (`freeze-amend`) with their consequences stated.

## 7. Formal runs

Exactly three runs, `R2-GoldKL+Edit` seeds 42, 43, 44: 3 epochs, 96 updates each, the frozen μ, the same epoch order as the
historical GoldKL run of the same seed. The historical `p2_1b/training/KL_seed{s}` runs are the matched controls, used only
after G4 passes; they are never retrained.

**Logged:** per micro-batch CE, GoldKL, L_edit, μ·L_edit, total loss, teacher-correct and edit-mask counts, and mean edit
energy (/1280) at teacher-correct Mandarin, all Mandarin, teacher-correct English, all English and all labelled positions;
per step gradient norms (pre-clip, per tensor), update and parameter norms, learning rate, orthogonality; per epoch pairing
with the GoldKL run (mask positions, supervised tokens, updates), teacher-forward and training seconds, memory, checkpoint
hashes; checkpoints and beam-5 / greedy dev decodes after epochs 1–3.

## 8. Selection and evaluation

**Selection:** the unchanged P2-1 rule per run (lowest beam-5 genuine-CS MER, then overall MER, then Mandarin CER, then
earliest epoch). Preservation, edit energy and locality loss never select. Results are reported for selected checkpoints
(primary), fixed epoch 3 (sensitivity) and every epoch (epoch-matched differences).

**Scorer:** canonical P2-1b / P2-1D / P2-2 / D1 code; GoldKL is recomputed and must reproduce `p2_1b/evaluation.json`.

**Endpoints.** Primary: new Mandarin errors; English corrections, corruptions and net; genuine-CS MER. Secondary: long
switch-back dropped units; target-window (TW, D1 definition) new errors with denominators; Mandarin retention (of 4,950
B0-correct units); overall MER; English WER and PIER; Mandarin CER; Mandarin repairs; near/far-switch damage; deletion vs
substitution; loops and non-termination; worst-dialogue regression; errors in utterances without English repairs; English
fates (omission/translation/lexical → correct, correct → wrong, wrong → other wrong).

**Statistics:** per-seed Δ_s = D2_s − GoldKL_s, means and ranges; per-dialogue contributions, leave-one-dialogue-out ranges;
2,000-replicate dialogue-cluster bootstrap (seed 42) with the same draws for both arms and all seeds, metrics recomputed
from resampled dialogue counts. The seeds share the 20 dialogues; intervals are descriptive.

## 9. Bounded diagnostics (training utterances only; never change μ, selection or the verdict)

**A. Representation selectivity** (`mech`): teacher-forced gold prefixes on all 500 training utterances, GoldKL and D2
checkpoints (selected and epoch 3) of every seed: ‖Δh‖, ‖Δh‖²/1280, ‖Δh‖/‖h‖ and token KL(p₀‖p_θ) by class (ZH_SB,
ZH_other, EN, other; all and teacher-correct positions); D2/GoldKL norm, energy and KL ratios and per-position edit cosines
by class; selectivity = Mandarin norm ratio / English norm ratio (1 = uniform shrinkage); principal-angle cosines of the R
row spaces, ‖b‖, ‖W − R‖.

**B. Same-prefix decoder-history diagnostic** (`prefix`): 40 training utterances with ≥ 1 switch-back Mandarin-positive
token, in sha256('d2-prefix:' + id) order. Systems: the selected GoldKL and D2 checkpoints of every seed. Prefix sources per
case: the gold reference, the B0 (frozen) beam-5 output on the same training audio, and each system's own beam-5 output.

| Prefix | Adapter off | Adapter on |
|---|---|---|
| Gold prefix | frozen next-token distributions | adapted next-token distributions |
| Same generated prefix (B0 or own) | frozen next-token distributions | adapted next-token distributions |

Within each row the audio and the full decoder input are identical, so every position compares the same history. Per
position: KL(p_off ‖ p_on), top-1 change, Mandarin-to-non-Mandarin top-1 change and change to EOS; positions are classed
reference-free by the frozen top-1 token's script (Han / partial multi-byte / Latin / EOS / other). Rows are compared only
through aggregates; no gold label is assigned to a generated prefix. A generated sequence without content or a final EOS
is ineligible for its row. On gold rows, KL at teacher-correct reference classes (ZH, ZH_SB, EN) is also reported. The
"transfer" ratio is (D2/GoldKL KL ratio on generated prefixes) / (D2/GoldKL KL ratio on gold prefixes). Descriptive; not
whole-sequence causality.

## 10. Conditional follow-up (recommendation only)

If D2 shows a positive screening signal, the recommended next control (not executed) is a locality-specificity control for
generic update shrinkage: GoldKL with training-only-calibrated inference attenuation h' = h + αΔh, or GoldKL plus a global
edit penalty with the same gradient-scale methodology. Neither runs without a new protocol and authorization.

## 11. Outputs and data handling

**Private** (`/mnt/data/tungnx/cs-asr-reft/d2_edit_locality/`): `AUTHORIZATION.json`, `freeze*.json`, `mu_freeze.json`,
`calibration_private.json`, `mask_private.jsonl`, `mask_static.json`, `preflight_*.json`, `mask_coverage_*.json`,
`training/D2_seed{42,43,44}/`, `evaluation.json`, `verify.json`, `mechanism_private.json`, `prefix_diagnostic_private.json`,
`report.json`, `manifest.json`, `figures/`.

**Public:** code, tests, configs and sanitized aggregate documents only: no transcripts, utterance or dialogue IDs, audio
paths, checkpoints or raw decodes.

Compute is via `sbatch` only. Logs go to `/mnt/data/tungnx/slurm_logs`, `TMPDIR` under `/mnt/data/tungnx/tmp`.

## 12. Engineering record before the freeze

Pre-freeze dry runs use `D2_DRY_ROOT` (scratch outputs, no freeze; G1/G7 fail by construction). The dry-run calibration
applies the already-written rule; its values do not change ρ, the batch rule or the stop thresholds. Any change made after a
dry run is listed here before the freeze.

**Dry run 1 (jobs 59628 mask, 59629 calibration, 59630/59632 preflight; scratch root, outputs deleted afterwards).**
- Mask: 500/500 mappable, 14,225 Mandarin-positive tokens (0 ambiguous, 1 unmapped token), all 27 dialogues; the
  switch-back subset has 2,825 tokens, equal to D1's region (consistency check).
- Calibration: `D2_CALIBRATION_OK` under the written rule (values reported with the formal calibration, which recomputes
  them). ρ, the batch rule and every stop threshold are unchanged.
- Preflight 59630 crashed when writing its record (a numpy bool in G5 is not JSON-serializable). Fix: cast to `bool`.
- Preflight 59632: G4 (bitwise GoldKL parity incl. the historical KL_seed42 epoch-1 replay with the observer installed), G5
  and G6 passed; G1 and G7 failed by construction (no freeze). Two further engineering findings, fixed before the freeze:
  - G2 `energy_requires_grad` was false because that check's forward ran with gradients disabled (`rt.logits(...,
    grad=False)`); every other G2 check, including the gradient reach of L_edit through `d2_step`, passed. Fix: the G2
    observer forward runs with gradients enabled. The training path is unchanged.
  - G3 coverage: 5 of 63 *manifest-order* micro-batches have no teacher-correct edit position (7.9% > 5%), because the
    manifest groups English-only utterances. Training never uses manifest order: in all nine seed × epoch training orders
    every micro-batch has ≥ 50 Mandarin-positive tokens before the teacher mask. The 5% floor was meant for the trained
    micro-batches, so it is now evaluated on teacher-correct counts in the nine training orders (threshold unchanged);
    manifest-order counts remain reported.
  - The mask definition, objective, calibration rule, recipe and decision rules were not changed.

**Dry run 2 (job 59635; same scratch root).** All gates passed except G1 and G7 (no freeze, by construction). G2
`energy_requires_grad` true; G3 coverage: no training-order micro-batch without a teacher-correct edit position (minimum
45 per micro-batch over the nine orders); G4 historical replay bitwise again.

**Post-training smoke test (jobs 59637 evaluate, 59638 mech, 59639 prefix; same scratch root).** The historical GoldKL run
directories were linked read-only in place of the D2 runs, so the evaluation, mechanism and same-prefix code ran end to end
before any D2 run exists. GoldKL recomputation reproduced `p2_1b/evaluation.json`, and every D2/GoldKL ratio was exactly 1
(identical systems), as expected. No code change resulted. The scratch outputs of both dry runs and the smoke test were
deleted.
