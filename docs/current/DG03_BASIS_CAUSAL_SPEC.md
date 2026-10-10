# DG-03 — Steering Basis Construction & Causal Validation: Pre-Run Specification

**Status:** FROZEN pre-run (2026-09-07), **before** any DG-03 ASR outcome was observed. Authority:
v6 proposal (`docs/proposal_arr/CS_ASR_ARR_October_2026_Method_First_Proposal_v6.md`) and
`METHOD_CONTRACT.md` §1–§7. This file pre-registers every operational rule; **no rule here is
changed after seeing which layer performs better.** DG-03 does no β-tuning, no controller training,
no SALSA/LoRA, no D-dev-confirm/D-test, no beam-5, no new datasets/models.

## 1. Model & site
- **Model:** `openai/whisper-large-v3`, `torch.bfloat16`, frozen backbone (32 decoder layers).
- **Site:** decoder post-cross-attention residual, pre-FFN — FROZEN (DG-02), `r_{ℓ,t}=q+u_source`.
- **Candidate layers:** `{16, 24}` only (Python index = scientific layer; DG-02 §17).
- **Hook:** `csasr.lss.sites.DecoderPostCrossAttnInterventionHook` (frozen); recorder
  `DecoderPostCrossAttnRecorder`.

## 2. Data roles (dialogue-v2r3; no repartition)
- **Basis construction:** `D-construct` only.
- **Causal screen / layer selection:** `D-dev-select` only — the deterministic candidate
  population from `steer_sweep.data.build_population(bundle, cfg, "D-dev-select")` (existing frozen
  candidate subset; no new/favorable sampling). No `D-dev-confirm`, no `D-test`, no SEAME/other.
- Exposure recorded in `DATA_EXPOSURE.md`.

## 3. Direction definitions (v6 §4; canonical builder `csasr.directions.steering_basis`)
Positions are exact-site states `r_{ℓ,t}` captured teacher-forced under `model.eval()`:
- **Embedded (E) positions:** first-token onset of eligible **baseline-correct** embedded-English
  spans (v2r3 `build_spans`, CTC convention `blank_to_preceding`, span family/target floor frozen in
  `v2r3_directions.FROZEN_CONFIG`; baseline-correct via `attach_baseline_status` on POI).
- **Matrix (M) positions:** the matched matrix-language (ZH) control step immediately preceding each
  span (`v2r3_directions.control_step_for`, `control_offset=0`).
- **Raw language contrast:** `v_raw = μ_E^correct − μ_M^correct`, computed as the **dialogue-balanced
  mean** of per-span `(r_E − r_M)` contrasts (`dialogue_balanced_mean`). No per-sample conditioning
  projection, no ridge/clip (those are legacy-only). (Balanced mean of paired E/M ⇒ μ_E − μ_M.)
- **Language-conditioning direction:** for each construction utterance, teacher-force the **same**
  token sequence with only the prefix language token changed — `build_prefix(language="en")` (c_E)
  vs `build_prefix(language="zh")` (c_M); capture site states; per-utterance contrast = mean over
  shared content positions of `r(c_E) − r(c_M)`; `v_cond = normalize(dialogue_balanced_mean(·))`.
- **Conditioning-residualized local direction (aggregate-then-residualize; order matters):**
  `v_local = normalize( v_raw − ⟨v_raw, v_cond⟩ v_cond )`.
- **Initial basis:** `V^0_ℓ = [ v_local, v_cond ]`. Conservative names: `raw_language_contrast`,
  `language_conditioning`, `conditioning_residualized_local`. Not "pure acoustic"/"universal"/"causal".
- **Not** a relabel of legacy `v_nat`: `v_raw`/`v_local` are recomputed at the exact site in the v6
  aggregate-then-residualize order.

## 4. Artifact — `steering_basis_v1`
One JSON per layer under `results/dg03/basis/steering_basis_v1_L{16,24}.json` + tensors
`.../v{raw,cond,local}_L{ℓ}.npy`. Records (§6 of the ticket): model id/revision, git commit, site id,
scientific layer, python index, dataset role, dataset fingerprint, construction config hash,
direction names, tensor files + sha256 hashes, dtype, dim, unit norms, E/M counts, conditioning pair
count, seeds, `cos(v_raw,v_cond)`, `cos(v_local,v_cond)`, removed conditioning fraction, reuse
provenance. Validation gates (hard): finite; dim == `d_model`; `‖v_cond‖≈1`; `‖v_local‖≈1`;
`|⟨v_local,v_cond⟩| ≤ 1e-5` (orthogonality). No fabricated metadata.

## 5. Diagnostic steering dose (pre-registered; NOT tuned to outcomes)
- Unit directions; NormPreserve **on** (frozen). Nominal update magnitude `‖ũ‖ = ρ·s_ℓ` with
  **ρ = 1.0** and `s_ℓ` = mean L2 norm of the site `r` measured on `D-construct` at layer ℓ
  (data-derived, fixed before the screen; stored in the basis artifact). Applied through the frozen
  hook as `alpha=ρ`, `scale=s_ℓ`, unit `direction`, `gain=1` at eligible steps.
- All controls C1–C4 use the **same nominal magnitude** `ρ·s_ℓ`; realized `‖r̃−r‖` is reported per
  condition. No β/ρ grid (that is DG-04).

## 6. Causal screen conditions (both L16 and L24; identical policy)
Oracle embedded decoder steps from `job_a_frozen.oracle_steps_for(pop)` (**oracle diagnostic /
non-deployable** — NOT the final inference mechanism).
- **C0** frozen baseline (no intervention).
- **C1** intended `v_local` at oracle embedded steps.
- **C2** `−v_local` (sign-reversed) at the same steps, same magnitude.
- **C3** matched-magnitude random directions, seeds `{0,1,2}` (`controls.random_direction`), same steps.
- **C4** `v_local` at matched **matrix-continuation** steps: for each oracle embedded step `s`, use
  `s+1` when `s+1` is not itself an oracle step (the matrix continuation immediately after the
  embedded onset). Same per-utterance edit count as C1 ⇒ matched realized-energy budget; a
  location-specificity control.
- Mapping (mirrors the established convention `job_a_frozen`): a generated step index `j` (0-based,
  from `oracle_steps_for`) is steered when `abs_pos − num_forced_prefix == j`; prefix positions
  (`abs_pos < num_forced_prefix = 4`) are never steered (enforced by the frozen hook).
- **Decoding:** free decoding, greedy, `temperature=0`, `num_beams=1`, `max_new_tokens=200`.

## 7. Metrics (DG-01 canonical; no legacy `delta_pier`)
Via `csasr.evaluation.canonical` / `result_emit` → `result_v1`: MER, PIER, embedded-language WER,
matrix-language CER/WER, corrections, corruptions, correction/corruption rates (valid denominators),
matrix retention, embedded retention, canonical outside harm (`outcomes.n_corrupted_outside`),
realized edit energy. Gains = `baseline − method` (positive = better). **Net correction–damage
utility** `U = corrections − corruptions` (η=κ=1, MC §10).

## 8. Layer-selection rule (pre-registered = ticket §15)
All quantities are canonical POI accounting from `paired_corpus_report`: `U = net_corrections =
corrections − corruptions` (η=κ=1); embedded WER = `en_wer`, matrix CER = `zh_cer`.
**Eligibility** (intended `v_local` / C1): (1) `corrections > corruptions`; (2) `U > 0`; (3) `U_C1
> U_C2` (sign-reversed); (4) `U_C1 > mean(U_C3)` (random aggregate over seeds {0,1,2}); (5) `U_C1 >
U_C4` (wrong-location, matched energy); (6) **no catastrophic matrix-retention failure**, defined
pre-run as method matrix CER `zh_cer_C1 ≤ 1.5 × zh_cer_C0` (matrix CER not inflated by more than
50%). **Winner among eligible:** highest `U`; tie-break PIER gain; effective tie (|ΔU| ≤ 1 and
|ΔPIER_gain| ≤ 0.002) → prefer L16
(efficiency). Neither eligible → **DG-03 FAILS**. Genuinely ambiguous → **INCONCLUSIVE**.
Conversation/dialogue-block bootstrap reported where practical (dev selection needs no final-paper
significance claim).

## 9. Seeds
Random-control seeds `{0,1,2}` (fixed). Decoding deterministic (greedy). No seed added post-hoc.

## 10. Slurm plan
- Construction: one GPU job over `D-construct` capturing L16 **and** L24 in a single pass (both
  recorders) → basis artifacts. (Do not process D-construct twice.)
- Screen: after basis frozen, one L16 job + one L24 job, ≤2 concurrent GPU jobs total from this task.
- Every job records: id, commit, model/revision, layer, config, data fingerprint, log paths,
  terminal state, runtime, GPU, peak mem if available. PENDING/RUNNING ≠ done.

## 11. Results (filled after terminal completion)

**Audit resolution (implementation `eac2e37`; repair commits `5465e04`/`12f0cac`; final artifact
freeze `212a06f`; re-run jobs 50470/50471/50472).** The independent freeze audit
raised three artifact/provenance gaps; all are now closed with no change to the pre-registered rule
or the selection: (1) basis artifacts carry a real `dataset_fingerprint`
(`sha256:3981d6a8…`, over 5 D-construct input parquets) and `construction_config_hash`
(`sha256:1c021fe9…`); (2) every screen condition (incl. C0) is a complete, `validate`-passing
`result_v1` — MER/PIER/embedded-WER/matrix-CER + gains + POI transitions + the three retention
populations + reserved gate_coverage — with per-utterance texts persisted; (3) per-condition
edit-count/total-energy are recorded and **C4 is count-matched to C1** (489 intended edits each).
The remaining canonical field, `outside_harm = n_corrupted_outside`, was then reconstructed
offline from the stored hypotheses and the frozen D-dev-select `existing_ctc` candidate spans
under `blank_to_preceding`. It counts only baseline-correct → method-wrong trusted reference
units outside the union of eligible embedded-English target candidates; unknown units and
insertions are excluded exactly as in `csasr.lss.outcomes`. No GPU rerun or transcript change was
needed. Construction is deterministic → basis metrics/hashes reproduced identically on the
re-run.

**Construction** — Slurm jobs **50452** then **50470** (mig H100 3g.40gb; 50470 adds provenance),
COMPLETED exit 0, ~50 s. Commits `749759b` / `eac2e37`. D-construct: **233 baseline-correct
embedded spans / 125 utterances / 20 dialogues**; 125 conditioning pairs (c_E=en / c_M=zh prefix).
Artifacts in `results/dg03/basis/` (`steering_basis_v1`), all validated (finite, dim=1280, unit
`v_cond`/`v_local`, `|⟨v_local,v_cond⟩|≈1e-17`):
- **L16:** `cos(v_raw,v_cond)=−0.0048`, `cos(v_local,v_cond)=−1.7e-18`, conditioning removed 0.0000,
  `s_16=7.479` (dose nominal ‖ũ‖=7.479).
- **L24:** `cos(v_raw,v_cond)=−0.0869`, `cos(v_local,v_cond)=−2.3e-17`, conditioning removed 0.0076,
  `s_24=8.931` (dose nominal ‖ũ‖=8.931).

**Causal screen** (final re-run: 300 D-dev-select utts, 489 oracle steps, greedy free decoding;
complete `result_v1` per condition). Baseline C0 (both layers): MER 0.2625, PIER 0.4700,
embedded-WER 0.4735, matrix-CER 0.2261. `U = net_corrections`. `n` = realized steered edits;
`E` = total realized edit energy `Σ‖r̃−r‖`.

- **L16** — Slurm job **50471**, COMPLETED exit 0. `results/dg03/screen/dg03_screen_L16.json`.

  | Cond | corr | corrupt | U | PIER gain | matrix-CER | n | E |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | C1 v_local @ oracle | 34 | 40 | **−6** | −0.0026 | 0.2178 | 456 | 2508.5 |
  | C2 −v_local | 21 | 52 | −31 | −0.0137 | 0.2318 | 458 | 2607.4 |
  | C3 random (mean of s0/1/2) | 9.3 | 31.7 | −22.3 | −0.0098 | ~0.225 | ~457 | ~2564 |
  | C4 wrong-location (count-matched) | 4 | 7 | −3 | −0.0013 | 0.2247 | 442 | 2455.1 |

- **L24** — Slurm job **50472**, COMPLETED exit 0. `results/dg03/screen/dg03_screen_L24.json`.

  | Cond | corr | corrupt | U | PIER gain | matrix-CER | n | E |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | C1 v_local @ oracle | 61 | 28 | **+33** | +0.0146 | 0.2287 | 456 | 2988.5 |
  | C2 −v_local | 27 | 70 | −43 | −0.0190 | 0.2234 | 456 | 3074.9 |
  | C3 random (mean of s0/1/2) | 15.3 | 22.3 | −7.0 | −0.0031 | ~0.230 | 456 | ~3049 |
  | C4 wrong-location (count-matched) | 35 | 25 | +10 | +0.0044 | 0.2397 | 442 | 2926.6 |

Edit budgets are now auditable: C1 and C4 both intend 489 edits and realize a comparable number
(456 vs 442) at comparable total energy (2988.5 vs 2926.6 at L24), so C1's advantage is not an
edit-budget artifact.

**Eligibility (rule §8):** L16 **fails** criterion 1 (corrections 34 < corruptions 40; U<0) → not
eligible. L24 **passes all six**: 61>28; U=+33>0; +33 > C2(−43); +33 > mean C3(−7.0); +33 > C4(+10);
matrix-CER 0.2287 ≤ 1.5×0.2261 (=0.3392). Direction-specific (sign flip −43), not-any-vector
(random −7), location-specific (count-matched wrong-loc +10 ≪ +33), matrix language preserved.

**Layer decision: SELECT L24** (only eligible layer). **DG-03 PASS / FROZEN** — the three audit
blockers (provenance, complete `result_v1`+retention, C4 edit-count/energy) and the canonical
outside-harm accounting are resolved (see *Audit resolution* above). The added candidate-level
utility is diagnostic only; the pre-registered layer rule remains the POI utility
`U = corrections − corruptions`, so its L24 values remain C1 `+33` and C4 `+10`. No β/ρ tuning;
dose fixed at ρ=1×s_ℓ pre-run.

Offline outside-harm repair values (`outside_harm`; C0, C1, C2, C3 seeds 0/1/2, C4) are:
L16: `0, 140, 152, 33/111/136, 127`; L24: `0, 225, 116, 135/136/150, 265`.
These values are correctness-flip harm, not outside transcript-edit counts, and do not alter the
stored transitions, retention, realized energy, or the pre-registered layer decision.
