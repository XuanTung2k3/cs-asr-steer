# EXPERIMENT MATRIX

Companion to `METHOD_CONTRACT.md` (MC) and `CODE_MAP.md`. Systems, ablations, and controls, with
their data role, layer, basis (direction), gate, objective, seed policy, decoding regime, and
purpose. Terminology is the contract's (MC §1–§8).

**Bases:** `raw` = `Δ^raw` · `resid` = conditioning-residualized `Δ^⊥` (primary) · `legacy` =
rejected-site / pre-narrowing direction · `random`/`sign`/`wrong-loc`/`perm` = controls.
**Gate:** `oracle` = gold CS decoder steps · `factorized` = MC §6 · `global` = all positions ·
`none`. **Decoding:** `TF` = teacher-forced (screening only) · `FD` = free-decoding (claims, MC §9).
**Site/layer:** decoder post-cross-attn residual (MC §1); candidate layers **{16, 24}** only (MC §3).

The matrices below are the **required/proposed design**, not an inventory of implemented runners.
`CODE_MAP.md` is authoritative for implementation status. In particular, current Job-A/Job-B and
Round-1 outputs are post-FFN legacy/current-execution results and do not instantiate these systems.

---

## A. Required — systems (proposed + baselines)

| ID | System | Role | Layer | Basis | Gate | Objective | Seeds | Decode | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| F0 | Frozen baseline, no intervention | D-dev-select / D-dev-confirm | — | — | none | none | 1 (det.) | FD | reference point; freeze `E_EN`/`C_EN` |
| F5 | **Frozen-direction auto soft-local** (proposed) | train gate on `loc∪util`/`router-calib`; select on D-dev-select; confirm on D-dev-confirm | 16 or 24 | resid | factorized | train localizer/selector; direction frozen | ≥3 gate seeds | FD | proposed system; not implemented |
| T1 | **Trained local steering** (proposed trained) | train `loc∪util`; select D-dev-select | 16 or 24 | resid (trained) | factorized | correction CE; retention loss `OPEN`; NormPreserve at intervention | seed 42 (+≥2 more `OPEN`) | FD | trained counterpart of F5 (rank `OPEN`, MC §4) |
| P1–P3 | Prompt baselines: forced-ZH, forced-EN, bilingual-token | D-dev-confirm | — | — | none | none | 1 | FD | training-free reference controls |
| F2 | Global steering, same per-position α | D-dev-confirm | 16 or 24 | resid | global | none | 1 | FD | local-vs-global axis |
| F3 | Global steering, matched total energy | D-dev-confirm | 16 or 24 | resid | global | none | 1 | FD | energy-matched global control |
| T2 | SALSA-D / learned global rank-one steering | train `loc∪util` | 16 or 24 | trained-global | global | learned-steering loss | seed 42 | FD | learned-vs-closed-form × global-vs-local |
| T3 | Matched-data LoRA (PM) | train `loc∪util` **only** | n/a | n/a | n/a | LoRA CE | seed 42 | FD | parameter-efficiency competitor; report hours/params/mem/time |
| U1 | Oracle-local upper bound (= F1) | D-dev-select / D-dev-confirm | 16 or 24 | resid | oracle | none | 1 | FD | headroom ceiling; **upper bound, not deployable** (MC §11) |

## A′. Required — controls (specificity nulls; MC §10 Gate B)

Applied at the **same layer, gate, and realized energy** as the proposed system so the comparison
changes the *direction/location*, not the energy.

| ID | Control | Basis / change | Seeds | Decode | Rules out |
|---|---|---|---|---|---|
| N1 | Within-pair label-permuted direction | `perm` (swap EN/ZH inside matched pairs, re-residualize, normalize) | **≥5** | FD | "any difference-of-means works" |
| N2 | Matched-norm random direction | `random` (`directions/controls.random_direction`) | ≥5 | FD | "any vector at this energy works" |
| N3 | Opposite sign | `sign` (`controls.wrong_sign`) | 1 | FD | direction sign carries the effect |
| N4 | Wrong location | `resid` applied to matched ZH / non-target region | 1 | FD | "steering anywhere works" |
| N5 | Global matched-energy (= F3) | `resid`, energy spread over utterance | 1 | FD | locality matters |
| N6 | Preservation set | correct-EN, neighbouring-ZH, monolingual retention | 1 | FD | intervention is selective, not blunt |

**Seed policy note:** frozen deterministic systems (F*, prompt, oracle) run one seed; the
null directions that *have* a random construction (N1 label-permutation, N2 random) run ≥5 seeds
with mean and spread; trained systems and any localizer run ≥3 seeds (v5 §6-Exp2).

---

## B. Secondary — localizer & selector ablations (post-Gate-B)

Only meaningful once Gate B (oracle headroom, free-decoding) has passed. All secondary ablations
use **free decoding** for their primary outcome metrics (PIER, correction, corruption) unless
noted otherwise; teacher-forced diagnostics may be used for screening but do not determine the
ablation result.

| ID | Experiment | Role | Purpose |
|---|---|---|---|
| L1–L4 | Localizer ladder: transcript heuristic → off-the-shelf LID → linear+smoothing → temporal-conv | loc-train / router-calib | Gate C: ≥70% recall of oracle-correctable spans |
| S1–S5 | Selector ablation: steer-all, uncertainty-only, language-only, full utility, full+abstention | util-train / router-calib | Gate D: outcome supervision beats steer-all & uncertainty |
| G1 | Score decomposition: gate with/without decoder score, with/without disagreement score | router-calib | isolate the value of each §5 score / factor; **determines the disagreement hypothesis** (MC §5): if disagreement adds no held-out incremental value, fall back to retention-aware gate without it (MC §6 fallback) |
| E1 | Layer ablation: 16 vs 24 (single selected layer per site) | D-dev-select | one layer chosen; not both |
| E2 | Basis ablation: raw vs conditioning-residualized (`cos(Δ^raw, Δ^⊥)`, energy removed) | D-construct | justify conditioning residualization (MC §4) |
| E3 | Norm-preserve on vs off | D-dev-select | justify norm-preserving repair (MC §7) |

## C. Optional — drop-first order (v5 §10.1)

| Priority | Item |
|---|---|
| 1 | SEAME same-pair transfer (Job A S0–S2 / Job B pilot) — only if licensed audio ready |
| 2 | Small-MLP selector capacity ablation |
| 3 | Adjacent-layer / fourth-dose appendix checks (out of scope under MC §3) |
| 4 | Optional conventional adapter baseline |
| 5 | Utility-scaled (vs binary) intervention strength |
| — | Round-3 Qwen frozen/training (stub/guard only; experiment pipeline unimplemented) |

Teacher-forced diagnostics (gold-token margin/rank; `steer_sweep.teacher_forced_sanity`) are
**screening only** and never a system claim (MC §9).

---

## D. Decision conditions between stages

| From → To | Condition to proceed |
|---|---|
| **Stage 1 audit → Ticket 2 metric canonicalization** | Repository mappings audited and metric discrepancies recorded. This condition is met; metric canonicalization may begin. |
| **Metric canonicalization → comparable new results** | One versioned PIER/MER denominator, correctness-transition accounting, delta sign, and artifact schema, with focused tests (CODE_MAP §3). |
| **Exact-site integration → new scientific runs** | Exact-site hook verified on the real model at layers 16 and 24 and integrated with cache position, forced-prefix exclusion, and beam expansion; `spec.yaml` candidate layers reconciled to `{16,24}`. |
| Directions frozen → Oracle screen | `Δ^⊥` finite, unit-norm, dialogue-bootstrap-stable; `cos(Δ^raw, Δ^⊥)` and removed-energy reported |
| Oracle screen → Automatic method | **Gate B passes** on **free-decoding**: net utility >0 with conversation-block bootstrap 95% CI excluding 0, corrections > corruptions, and the proposed direction beats N1–N4 in paired bootstrap. If Gate B fails, **stop** — no localizer/selector can rescue a useless action |
| Transport gate T | ≥60% practical decoder coverage with target-step concentration on the **high-confidence alignment subset**; else lock E-only as the practical system |
| Localizer → Selector | Gate C: ≥70% recall of oracle-correctable spans at manageable candidate load |
| Selector → Confirmation | Gate D on development data; then freeze the **entire** pipeline before `D-test` |
| Confirmation → Locked test | one scheduled `D-test` batch, mechanical rerun of the frozen dry-run script; no new code, no threshold change, primary claim already chosen (MC §11) |

**Seed / decoding invariants:** greedy, `temperature=0`, `num_beams=1` for primary decoding;
beam-5 only as a final-pass appendix. All confidence intervals are conversation/dialogue-block
bootstrap (10,000 resamples for final intervals). Development selection estimates are kept
separate from locked test estimates.

**Metric sign invariant for new artifacts:** report named raw method-minus-baseline changes for
MER/PIER (negative is improvement) and/or an explicitly named improvement quantity (positive is
improvement); never use the ambiguous bare name `delta_pier`. Legacy artifacts retain their
original signs and labels and must be converted explicitly before comparison.

**Outcome invariant:** correction = baseline incorrect to method correct; corruption/harm = baseline
correct to method incorrect. An outside-region transcript edit is not outside harm unless it is
also a correctness flip under the frozen candidate-unit accounting.
