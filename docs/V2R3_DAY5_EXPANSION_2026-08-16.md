# Day 5 — dose extension, eligibility expansion, first free decoding

**Date:** 2026-08-16
**Job:** Slurm 41007, `COMPLETED`, exit 0, **7 m 29 s** (smoke 41005, 2 m 25 s)
**Destination:** `/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/day5/generation_001`

Three answers: **the dose curve does saturate**, within the extended grid;
**effect is flat across eligibility tiers** while coverage barely moves; and the
**named deletion prediction is refuted in its conclusion but partly supported in
its mechanism** — on evidence far too thin to carry either.

Frozen action, recorded and not re-derived: **Site D, decoder layer 16, ρ = 1.**

---

## 1. Task 1 — does the dose curve saturate?

Site D, correct direction versus label-permuted null, substitution-only, G = 19.
ρ ≤ 2 from Day 4; ρ ∈ {3, 4} new.

| Layer | ρ=0.25 | ρ=0.5 | ρ=1 | ρ=2 | **ρ=3** | **ρ=4** |
|---|---|---|---|---|---|---|
| 8 | +0.008\* | +0.017\* | +0.031\* | +0.066\* | **+0.076** | **+0.043** |
| 16 | +0.024\* | +0.054\* | +0.106\* | +0.178\* | **+0.170\*** | **+0.079** |

`*` = interval excludes zero. Full intervals at the new points: layer 8 ρ=3
+0.0755 [−0.0038, +0.1548], ρ=4 +0.0425 [−0.0498, +0.1349]; layer 16 ρ=3 +0.1703
[+0.0290, +0.3115], ρ=4 +0.0786 [−0.0990, +0.2563].

**It saturates, and then declines.** Layer 16 rises to ρ=2, is flat at ρ=3
(+0.178 → +0.170, still excluding zero), and falls to +0.079 at ρ=4 with the
interval crossing zero. Layer 8 peaks at ρ=3 and falls at ρ=4. The Day 4 worry —
an effect that keeps growing is hard to separate from a generic magnitude effect
— is answered: the curve has the saturating-then-declining shape the hypothesis
predicts, with the turn between ρ=2 and ρ=3.

The frozen ρ=1 was **not** revisited on this curve. It sits on the rising limb,
well below the turn.

---

## 2. Task 2 — eligibility expansion

Frozen action at each floor, correct versus label-permuted null:

| Floor | Stratum | Units | G | Effect | CI |
|---|---|---:|---:|---:|---|
| 400 ms | all errors | 489 | 20 | +0.1361\* | [+0.0798, +0.1924] |
| 400 ms | substitution-only | 199 | 19 | +0.1057\* | [+0.0466, +0.1649] |
| 400 ms | wrong-language *(desc.)* | 90 | 12 | +0.0753\* | [+0.0351, +0.1154] |
| 300 ms | all errors | 502 | 20 | +0.1444\* | [+0.0839, +0.2048] |
| 300 ms | substitution-only | 207 | 19 | +0.1103\* | [+0.0497, +0.1709] |
| 300 ms | wrong-language *(desc.)* | 93 | 13 | +0.0809\* | [+0.0369, +0.1249] |
| 200 ms | all errors | 515 | 20 | +0.1415\* | [+0.0854, +0.1977] |
| 200 ms | substitution-only | 215 | 19 | +0.1042\* | [+0.0424, +0.1661] |
| 200 ms | wrong-language *(desc.)* | 94 | 13 | +0.0811\* | [+0.0371, +0.1250] |

**Effect against coverage is flat — because coverage barely moves.** Relaxing
400 → 200 ms adds 26 error units (489 → 515, +5.3%) and 8 substitution units
(199 → 215, +8.0%), while the effect stays within noise of itself (+0.106,
+0.110, +0.104 on the substitution stratum). Every interval excludes zero at
every tier.

The practical reading is that **the expansion tiers buy almost nothing here**.
The headroom analysis anticipated larger gains; on the convention-adjusted spans
the three tiers are nearly the same set. No floor is selected.

---

## 3. Task 3 — first free decoding, oracle-localized

Conservative 400 ms subset, frozen action, gold span and gold decoder step. This
is an **upper bound**, not the practical result; Day 7 runs the non-oracle
version.

| Metric | Baseline | Steered |
|---|---:|---:|
| PIER | 0.5249 | **0.5189** |
| MER | 0.2850 | **0.2780** |
| WER | 0.8684 | **0.8674** |

Corpus WER sums per-utterance edits and reference lengths; aligning one
concatenated string would let errors migrate across utterance boundaries.

**Targeted outcomes: 7 corrections, 0 corruptions**, over 489 baseline-error
targets — a 1.4% correction rate with a corrected-to-corrupted ratio of 7 : 0.

### Spillover dominates the targeted effect

Counted per utterance (per-target counting inflates this, since one utterance can
hold several targets — 1,034 becomes 85):

| Quantity | Per utterance |
|---|---:|
| units changed outside the target | **85** |
| improved | 72 |
| damaged | 13 |
| utterances with any spillover | **5 / 116 (4.3%)** |

**Spillover exceeds the targeted effect by roughly 12×** (85 changed vs 7
corrected), and is concentrated in 5 utterances. Net it is favourable — 72
improved against 13 damaged — but a single-step intervention changing 85 units
elsewhere is not a localized edit. All 13 damaged units are the only corruption
observed anywhere, since every target is baseline-error by construction.

### The named prediction

**Predicted:** Site D's deletion advantage should collapse under free decoding,
because a deleted word has no emitted token and no step to intervene at.

**Outcome: refuted in its conclusion, partly supported in its mechanism.**

| Category | Targets | Hook fired | Corrected (fired only) |
|---|---:|---:|---:|
| deletion | 290 | **207 (71.4%)** | 6 / 207 = **2.90%** |
| substitution | 199 | **198 (99.5%)** | 1 / 198 = **0.51%** |

The mechanism is real and measured: for **28.6% of deletion targets the oracle
step did not exist** — the free decode terminated before reaching it — against
0.5% for substitutions. That is exactly the effect the prediction describes, and
it has no analogue under teacher forcing.

But the advantage did **not** collapse. Conditional on the step existing,
deletions are corrected at 2.90% against 0.51% for substitutions — still roughly
6×, the same direction as the teacher-forced +0.157 vs +0.106.

**This must not be over-read.** The entire comparison rests on **7 corrections**
(6 deletion, 1 substitution). At those counts the ratio is not distinguishable
from chance, and no cluster-bootstrap interval on it would be meaningful. The
honest statement is that the prediction's mechanism is confirmed, its conclusion
is not observed, and the data cannot presently settle which dominates. Day 7's
non-oracle run on a larger effective sample is where this becomes decidable.

---

## 4. Verification

| Check | Result |
|---|---|
| Identity | all prior artifacts byte-verified; directions `813e84a5…` classified **split**, `identity_required: false` with reason recorded |
| `smoke_001` | refused by name; the Day 5 smoke destination was cleared before the real run |
| Code freeze | `4b1fff95eaf1ab056cd3ff92dea00198b54465d20b4c391ab9033b589a9af74e` — identical before publication, inside job 41007, and after completion |
| Published format | split, all three artifacts |
| Immutability | `artifacts_lss`, `artifacts_v2`, `artifacts_dialogue_v2`, `v2r2` at prior mtimes; exposure ledger `1786502490` |
| Selections | none — no layer, strength, or floor selected; no gate evaluated |

Two defects were caught before the real run and are recorded because both would
have produced silently wrong numbers:

1. A walrus expression in `FROZEN_CONFIG` that assigned `None` and always took
   the else branch — removed.
2. `corpus_word_error_rate` used the key names `substitutions`/`deletions`/
   `insertions` while `csasr.evaluation.mer.error_rate` returns `sub`/`del`/`ins`.
   With `.get(..., 0)` this would have reported **WER = 0.0000 for both arms**.
   Found by printing the function's actual return keys, fixed, and unit-checked
   against a 1-edit / 5-token case returning 0.2.

---

## 5. Open items

1. **Spillover is 12× the targeted effect** and concentrated in 5 utterances. A
   single-step intervention is not behaving locally; Day 7 should measure
   spillover as a first-class outcome, not a diagnostic.
2. **The oracle step is absent for 28.6% of deletion targets.** Any non-oracle
   localizer inherits this ceiling on deletions before its own errors are
   counted.
3. **7 corrections is too few to compare categories.** Effect sizes at the token
   level are large and consistent; the free-decoding yield is not.
4. **The expansion tiers are nearly the same set** (+5.3% units from 400→200 ms).
   Whether that reflects the convention-adjusted spans or the corpus is not
   established here.

**No layer, strength, or floor was selected. No gate was evaluated. Production
Gate A is unchanged.**
