# EXPERIMENT MATRIX

Companion to `METHOD_CONTRACT.md` (MC) and `CODE_MAP.md`. **Reconciled 2026-09-07 (DG-03R)** to the
updated proposal: **contrastive steering basis → adaptive controller → damage-aware optimization**.
The older disagreement/localizer/utility-selector/factorized-gate matrix is `LEGACY DESIGN` and is
recorded in §E for provenance only.

**Basis (MC §4):** `V^0 = [v_local, v_cond]` at a candidate layer · `v_local` =
conditioning-residualized local-language direction · `v_cond` = language-conditioning direction ·
`legacy` = rejected-site / pre-narrowing (`v_nat`) · `random`/`sign`/`wrong-loc` = controls.
**Controller (MC §6):** `f_θ(LN(r_t)) → (g_t, π_t)`; `d_{ℓ,t}=normalize(V^0 π_t)`; no oracle CS
location at inference. **Decoding:** `TF` = teacher-forced (optimization + screening only) · `FD` =
free-decoding (checkpoint selection + all transcript claims, MC §9). **Site/layer:** decoder
post-cross-attn residual, pre-FFN, FROZEN (DG-02); candidate layers **{16, 24}** (MC §3).

The matrices below are the **required/proposed design**, not an inventory of implemented runners.
`CODE_MAP.md` is authoritative for implementation status. Legacy Job-A/Job-B/Round-1 outputs are
post-FFN and do **not** instantiate these systems.

Main scientific comparison: the **correction–retention–efficiency trade-off**, visualized as the
**correction–damage frontier** over steering strength / controller operating points.

---

## Post-DG-02 sequence (DG-03 … DG-08)

| Ticket | Purpose | Data role | Decode |
|---|---|---|---|
| **DG-03 — AUDIT BLOCKED** | Basis and causal screen executed; L24 is numerically eligible (U=+33 vs L16 U=−6), but freeze awaits complete provenance/result emission and C4 energy accounting. Spec `DG03_BASIS_CAUSAL_SPEC.md`. | construct `D-construct`; screen `D-dev-select` | FD |
| **DG-04** | Frozen-steering baselines + small strength grid → **RQ1** correction–damage frontier before any learning | `D-dev-select` | FD (TF screen only) |
| **DG-05** | Adaptive controller `f_θ(LN(r_t)) → (g_t, π_t)`, `V^0` fixed; no oracle location | train on `loc-train ∪ util-train`; select on `D-dev-select` | FD |
| **DG-06** | Damage-aware training, staged (MC §8); optional gate penalty; optional basis refinement | train pool; select `D-dev-select` | TF optimize / FD select |
| **DG-07** | Baselines & key ablations (SALSA-global, matched LoRA, local-only, cond-only, fixed vs learned mixture, global vs adaptive, fixed vs refined basis, correction-only vs correction+retention) | train pool; `D-dev-select` | FD |
| **DG-08** | Locked Whisper core evaluation: 3 seeds (trained), greedy primary + beam-5 finalists, frontier, MER/PIER/embedded-WER/matrix-CER, retention, efficiency, bootstrap CIs | `D-dev-confirm` then **`D-test`** (after full freeze) | FD |

Later (only after the Whisper core method succeeds): SEAME / ViMedCSS, selected multilingual sets,
Qwen3-ASR (architecture replication first), other speech/audio-LMs if feasible. Do **not** expand
architecture/dataset scope before the Whisper core study is established.

---

## A. Required — core systems & baselines (DG-04/DG-05/DG-07)

| ID | System | Kind | Layer | Basis | Strength/gate | Objective | Seeds | Decode |
|---|---|---|---|---|---|---|---|---|
| B0 | Frozen backbone, no intervention | baseline | — | — | none | none | 1 (det.) | FD |
| B1 | Global fixed **local** steering | baseline | 16 or 24 | `v_local` | fixed `β`, all positions | none | 1 | FD |
| B2 | Fixed **local+conditioning** mixture | baseline | 16 or 24 | `V^0` fixed mix | fixed `β`, fixed `π` | none | 1 | FD |
| B3 | Exact-site F5-style / projection-gated steering (where scientifically comparable) | baseline | 16 or 24 | `v_local` | projection gate | none | 1 | FD |
| B4 | **SALSA-style learned global vector** | baseline | 16 or 24 | trained-global | learned global | learned-steering loss | seed 42 | FD |
| B5 | **Matched-data LoRA** | baseline | n/a | n/a | n/a | LoRA CE on train pool only | seed 42 | FD |
| **M**  | **Adaptive controller** `f_θ(LN r)→(g_t,π_t)`, `V^0` fixed (proposed) | method | 16 or 24 | `V^0` + controller mix | learned `g_t`, `π_t`, opt. `β` | damage-aware (MC §8) | ≥3 | FD |
| M-ref | Constrained basis refinement `V=V^0+ΔV`, anchor `‖ΔV‖_F²` | ablation | 16 or 24 | refined | controller | + `λ_A` anchor | ≥3 | FD |
| U | Oracle-location upper bound | reference | 16 or 24 | `V^0` | oracle span | none | 1 | FD |

Optional-only baselines (never blockers for the core paper): adapter/ReFT; full fine-tuning.

## A′. Required — causal controls (MC §10 Gate B / DG-03), applied at matched layer & realized energy

| ID | Control | Change | Seeds | Rules out |
|---|---|---|---|---|
| C-sign | Sign-reversed direction (`controls.wrong_sign`) | flip `d` | 1 | direction sign carries the effect |
| C-rand | Matched-norm random direction (`controls.random_direction`) | random `d` at matched energy | ≥5 | "any vector at this energy works" |
| C-loc | Wrong location | steer matched non-target region | 1 | "steering anywhere works" |
| C-pres | Preservation / correct-token analysis | measure baseline-correct embedded + matrix retention | 1 | intervention is blunt, not selective |

Label-permutation may remain if already justified and inexpensive; not required to expand DG-03R
scope. **Do not add new mechanistic analyses** in DG-03R.

## B. Key ablations (DG-07)

local-only vs conditioning-only vs `V^0`; fixed mixture vs learned token mixture; global vs adaptive
strength; fixed vs refined basis; correction-only vs correction+matrix vs correction+matrix+embedded
retention (staged, MC §8). All primary outcomes on **free decoding**.

---

## C. Evaluation contract (DG-08) — preserves DG-01 metrics

Report, per operating point / strength: **MER, PIER, embedded-language WER, matrix-language CER/WER,
correction rate, corruption rate**, outside-harm/edit diagnostics (with their existing DG-01 semantic
distinction), **embedded retention, matrix retention**, gate-strength distribution, gate coverage
*once its denominator is legitimately defined*, trainable parameters, GPU-hours, memory, latency.
Primary visualization: the **correction–damage frontier**. Confidence intervals are
conversation/dialogue-block bootstrap. Metric definitions, gains (`baseline − method`, positive =
better), and correction/corruption/outside-harm semantics are **frozen by DG-01** and unchanged.

## D. Decision conditions between stages

| From → To | Condition |
|---|---|
| DG-03 basis → DG-04 | `V^0` finite, unit-norm, dialogue-bootstrap-stable; intended direction beats sign/random/wrong-location controls at L16 and/or L24; one layer selected |
| DG-04 → DG-05 | a favorable free-decoding correction–damage point exists for some fixed steering (RQ1); corrections > corruptions with CI |
| DG-05 → DG-06 | controller runs at inference from `LN(r_t)` only (no oracle location), beam-safe |
| DG-06 → DG-07 | staged damage-aware training improves the frontier over fixed steering (RQ2) |
| DG-07 → DG-08 | method non-dominated vs SALSA-global / matched LoRA on correction–retention–efficiency (RQ3); then **freeze the entire pipeline before `D-test`** |
| DG-08 confirm → locked test | one scheduled `D-test` batch, mechanical rerun of the frozen script; no new code/threshold change; primary claim already chosen (MC §11) |

**Seed/decoding invariants:** greedy, `temperature=0`, `num_beams=1` primary; beam-5 finalists only.
Trained systems run ≥3 seeds; random-construction controls ≥5. Development selection estimates are
kept separate from locked-test estimates.

**Metric sign invariant:** report named improvement quantities (positive = better) or explicit
`method − baseline` changes; never the bare `delta_pier`. Legacy artifacts keep their original signs
and are converted explicitly before comparison (DG-01).

**Outcome invariant:** correction = baseline-incorrect → method-correct; corruption/harm =
baseline-correct → method-incorrect. An outside-region transcript edit is not outside harm unless it
is also a correctness flip under the frozen candidate-unit accounting (DG-01).

---

## E. LEGACY DESIGN (superseded for the core paper; preserved for provenance)

The former matrix centered on: temporal localizer ladder (L1–L4, Gate C recall); outcome-supervised
utility/abstention selector (S1–S5, Gate D); factorized gate `g_t = 1[t∈S]·m_t·1[p̂≥τ]`;
encoder–decoder **disagreement decision gate** and its repairability study; decoder-transport gate T.
These are **not prerequisites** for the core method. Disagreement may appear only as
`OPTIONAL SUPPORTING ANALYSIS`, never as a gate before controller training. Legacy F5/T1/SALSA-E/LoRA
scaffolding and post-FFN Round-1 results remain on file (CODE_MAP) and are not evidence for the
exact-site method.
