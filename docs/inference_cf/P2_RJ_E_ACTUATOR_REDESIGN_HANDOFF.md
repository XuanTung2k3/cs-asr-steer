# Actuator redesign handoff (for one independent Codex/Astra design session)

**Status:** the diagnostic programme is closed (2026-09-26). This document hands the next *design*
decision to an independent reasoning session. It proposes no method.

- **Repository:** `XuanTung2k3/cs-asr-steer`, branch `cs-asr-steer-inf`.
- **Authoritative detail:**
  - `docs/inference_cf/P2_REPORT.md`
  - `docs/inference_cf/P2_R_REPORT.md`
  - `docs/inference_cf/P2_RJ_REPORT.md`
  - `docs/inference_cf/P2_RJ_E_REPORT.md`

## A. Final diagnosis

**`P2_RJ_E_DIRECTION_CONFIRMED_SITE_UNRESOLVED`** (terminal; `P2_RJ_E_AUDIT: PASS`).

- **H1 is established.** The repair direction d = norm(hE − hB) is not a useful lexical-repair
  direction at the tested site.
- **H4 is unresolved.** Whether decoder L16 has enough lexical leverage at the P2 edit budget could
  not be decided after exhausting the eligible diagnostic population.

## B. Established facts (all audited)

**System under test.**

- Whisper-large-v3 with a forced-ZH prompt, greedy decoding.
- A single-layer edit at L16, at the DG-02 decoder post-cross-attention / pre-FFN residual, with
  NormPreserve.
- Detector g = E·R_B, with dose α·g at α = 2 and direction +d = norm(hE − hB).
  - hE and hB are the same site's states under forced-EN vs forced-ZH prompts on the same prefix.
- The gate g = E·R_B was selected in R2 as a repair-need **ranking** score: AUROC 0.820/0.836
  against 0.710/0.726 for the counterfactual gate. Its repairability has never been shown causally.

**P2 efficacy** (D-dev-select, 300 utterances, 20 dialogues):

- The selected configuration removes **2 of 2,268 POI errors** relative to the matched unsteered
  baseline (ΔPIER CI95 [0, 0.0027]).
- Ordinary Whisper, `generate(language=None)` = B0_AUTO, is **8.2 PIER points better** than any
  forced-ZH system.
- Mechanism controls did not support the method:
  - energy-matched constant gate g = 1: +5;
  - g = R_B: +10;
  - −d: +11 with 0 corruptions;
  - selected +d: +2.

**P2-R** (80 utterances; 180 energy-matched single pulses at e\* = 1.126 ≈ 15% of ‖h‖):

- **Every direction is inert at the lexical decision.**
  - Δlog p(ref) ≤ 0.1 nat against a median 10.6-nat gap.
  - 0/60 top-1 corrections.
  - 0/20 companion recoveries.
- **+d is weak and the worst direction tested.**
  - L(+d) = −0.026 [−0.080, 0.025].
  - It is significantly below −d (−0.124 [−0.252, −0.012]) and below random.
- **−d's larger response is descriptive, not established.** The lexical contrast includes 0.
- **Oracle placement gives no rescue.** Energy-matched oracle relocation onto the true confusion
  positions leaves P(oracle − current) at an upper bound of 0.0016 (D2-B). Localization is not the
  explanation.
  - Detector window timing error: median 0.74 s.

**P2-RJ** (evaluator-only Jacobian of m = logsumexp z(Y_ref) − z(c\*) at the 180 frozen states):

- **The first-order model is valid.** Predicted versus realized P2-R arm Δm gives
  **r = 0.918 [0.892, 0.936], slope 1.03**. The fp32 versus bf16 gradient cosine is 0.9998.
- **Leverage.** Median ‖∇m_tan‖ is 4.50 nats per unit state norm; the gradient is almost entirely
  tangent. The first-order maximum A = ‖∇m_tan‖·e\* has a median of 5.07 nats.
- **Alignment of d.**
  - cos(∇m_tan, d) has a median |·| of 0.023, the same as random (0.022).
  - κ(+d), the captured fraction of the optimum, is −0.011 [−0.022, −0.001]; κ(−d) is +0.011.
- **The useful direction is largely a script readout.** At EN-confusion positions,
  cos(∇m_tan, ∇log P_Latin,tan) = **0.80** and cos(∇m_tan, ∇log P_Han,tan) = −0.78. d misses that
  too (−0.006).
  - In correct strata the relation differs: EN-correct 0.46, ZH-correct −0.55. At ZH-correct
    positions, pushing Latin mass lowers the correct margin.
- **Safety scale.** A worst-case e\*-edit could erase the correct-token surplus at 37% of
  EN-correct and 27% of ZH-correct positions, to first order.

**P2-RJ-E** (all 227 eligible EN-confusion positions, 17 dialogues):

- Median gap **10.56 nats**; median A **4.90 nats**; median ρ = A/gap **0.52**.
- ρ ≥ 0.1: 100%; ρ ≥ 0.5: 52%; ρ ≥ 1: 12% (dialogue-weighted 0.147).
- **Q50 = 0.539, Bonferroni CI [0.406, 0.665]**, leverage `UNRESOLVED`; fp32 gives 0.519
  [0.385, 0.647].
- κ(+d) = −0.0098 [−0.019, −0.002]; κ(−d) = +0.0099; κ(random) ≈ 0.

## C. Rejected or unsupported assumptions

- "hE − hB (the forced-EN vs forced-ZH prompt-conditioning displacement) is an English lexical
  repair direction." **Rejected at L16.** It is near-orthogonal to the lexical and script readout
  sensitivities.
- "−d is the fix." **Unsupported.** It captures ≈ +1% of the available leverage.
- "The failure is localization or gating." **Not supported.** The oracle placement gave no rescue.
- "The failure is generic perturbation." **Not established.** The effects were sign-specific.
- "L16 at the P2 energy is sufficient if aligned." **Unresolved.** At first order a perfectly aligned
  edit closes about half the gap at the median, and all of it at only ~12% of positions.
- "The gate g = E·R_B is causally effective." **Never shown.** It is a ranking score only.

## D. Components that may remain frozen

The design session may challenge any of these, with reasons.

- **Inference only.** Inference-time, **reference-free**, **no training** of model weights.
- **Detector.** R2 LocalSupport E (null-corrected native LID on the 1.0 s max-attention window),
  BaselineConflict R_B, and g = E·R_B as the repair-need detector.
  - The designer may argue that the gate should be kept, ablated or replaced once the actuator
    changes. Required ablations must then be specified (at least g = 1 at matched energy, and the
    no-edit baseline).
- **NormPreserve.** The norm-preserving edit at the DG-02 hook
  (`csasr.lss.sites.DecoderPostCrossAttnInterventionHook`), with no `sqrt(num_layers)` rescale.
- **Execution and evaluation.** The KV-cached three-branch execution (`experiments/inference_cf_cached.py`),
  free-decoding evaluation, and damage-aware metrics (ZH-CER, retention, corrections versus
  corruptions).
- **Baselines.** Every claim must also be reported against B0_AUTO.

## E. Components that must be redesigned

- **The direction construction.** It replaces norm(hE − hB); this is established.
- **A predeclared site-failure criterion** for L16 at the chosen budget. Site adequacy is
  unresolved, so the design must say in advance what outcome would trigger a later site or
  representation redesign.

## F. Allowed data roles

- **`D-dev-select`:** the 300-utterance R2/P2 panel, 20 dialogues, already exposed. Use it for
  development and for mechanism checks.
- **`router-calib`:** a *conditional candidate only* (3,031 utterances in 8 dialogues; its 300
  shortest were used for DG-04 calibration). Using it requires **explicit human authorization**.
- **No fresh repair-validation role is currently authorized.** The design must state the minimal
  validation population it needs (size, dialogues, stratum counts, power argument) and must not
  assume one.

## G. Forbidden data roles

- **`D-dev-confirm`:** reserved for a single confirmation freeze of a fully frozen pipeline.
- **`D-test`:** locked, underpowered, with historical DG-08 exposure.
- **P3 populations.**
- **Transfer corpora:** SEAME, CS-FLEURS, ViMedCSS, ASCEND, Qwen.
- None of these may be used for tuning, selection or diagnosis.

## H. Implementation constraints

**Forbidden signals.** The **P2-RJ reference-margin Jacobian was diagnostic only**. It is forbidden
as:

- a steering direction;
- a fitted target;
- a label-derived training signal;
- an inference feature.

The same holds for evaluator or oracle localization (CTC midpoints, R2 alignment, reference target
sets). References, oracle alignment and evaluator gradients may not enter any deployable module.

**Required practice:**

- **Scope.** One actuator, not a sweep or family: no layer search, no α sweep, and no
  random-seed or sign selection.
- **Hook and site.** Reuse the canonical hook. If a non-L16 site is ever proposed, it must come
  from a tiny, theoretically motivated set and be justified mechanistically.
- **Freezing and provenance.** Freeze a versioned spec and config before outcomes. Every run writes
  a provenance manifest with hashes. Invalid attempts are preserved and are never rerun because a
  result is unfavourable.
- **Claims.** Transcript claims come from free decoding only; teacher-forced results are screening
  only.

## I. Compute budget

- One H100 MIG 3g.40gb slice per job, with at most two pending or running jobs.
- For reference, P2-A r1 at L16 (job 54821: 300 utterances, the matched baseline plus the α
  systems, free decoding) took 56 min, and P2-RJ-E (227 Jacobian positions, two precisions) took
  2.3 min.
- Aim for at most a few GPU-hours in total for the first repair experiment.

## J. Exact design question for Codex/Astra

> H1 is established; H4 is unresolved after exhausting the current diagnostic role. Design **ONE**
> direction-first replacement for d = norm(hE − hB) that:
>
> - is reference-free and inference-time;
> - can be tested at the existing L16 DG-02 site and P2 energy budget, with the current detector
>   g = E·R_B kept fixed for the first repair experiment unless you argue otherwise;
> - has a stated mechanistic reason to couple to the downstream language/lexical readout
>   sensitivity that d misses;
> - uses no reference transcript, oracle alignment or evaluator gradient at inference.
>
> Also specify:
>
> 1. **The site-failure criterion.** One predeclared criterion that, if met, would trigger a later
>    site or representation redesign.
> 2. **The evaluation.** A primary endpoint, the required controls (no-edit, matched-energy g = 1,
>    and the old +d at matched energy), and damage constraints on ZH-correct and EN-correct.
> 3. **The validation population.** The minimal population needed, and which role it would come
>    from, subject to human authorization (§F).
> 4. **What would falsify the design.**
>
> Do not propose a sweep of directions, layers or α. Do not reopen the diagnostic programme.
> Choose one construction and justify why it, rather than its alternatives, should come first.

**Guardrail:** no replacement actuator is implemented before this independent design pass.
**P3 stays HELD.**
