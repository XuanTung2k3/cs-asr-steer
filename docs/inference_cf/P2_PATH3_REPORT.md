# P2-PATH3 — frozen G1 fixed-100 breadth confirmation (report, terminal)

**Verdict: `P2_PATH3_SIGNAL_CONCENTRATED`.** G1 (unchanged PATH2 CONSENSUS-1) on the historical fixed100 passes
ABSOLUTE_SAFETY, AGGREGATE_RESCUE, USEFUL_GAIN and BENEFIT_RETENTION. It fails BREADTH_PASS: NOVEL76 has no rescue at
all. Independent audit `P2_PATH3_AUDIT: PASS` (primary phase before references, then full); the audit's own label
agrees.

Scope: **breadth-confirmation DEVELOPMENT** on already-exposed D-dev-select data. NOVEL76 means "not used in the
A3/A4/PATH diagnostic chain". It was already exposed by TTA1, so it is **not fresh validation and not unexposed
references**. No generalization or significance claim.

| Step | Commit / job | Result |
|---|---|---|
| Freeze (spec, design, panel, config) | `5853a8d` | panel sha256 `97de2edb…` |
| Implementation + sealed plan | `ddfa787` | runner/analyzer/auditor/sbatch/tests; prepare all checks true |
| Pre-run audit | `71a51a1` | `PASS_TO_P2_PATH3` |
| Manifest pushed | `cd93f3d` | before the job (manifest git_commit `71a51a1`) |
| Run | Slurm 58015 | 4 min 02 s allocation, 215 s runtime |
| Output seal + primary analysis | `bceb341` | pushed before any reference access (a transient GitHub HTTP 500 outage delayed the push about 10 min; content unchanged) |
| Primary-phase audit | `77f4ee9` | `P2_PATH3_AUDIT: PASS` before references |
| Secondary + full audit + report | this commit | `P2_PATH3_SIGNAL_CONCENTRATED`; `P2_PATH3_AUDIT: PASS` |

## Panel and partitions

The fixed100 is the parent `P2_SEL_MINI_PANEL.json` (100 IDs, 20 dialogues × 5, original order). Its byte sha256 is
`266ea7ea…`; the internal `panel_sha256` `3fd67b93…` is in a different hash domain. Identity, dialogue and role were
verified independently against the role manifest, reading only the ID/dialogue/role columns.

| Partition | Rows | Dialogues | Membership hash |
|---|---:|---:|---|
| FULL100 | 100 | 20 | `3560be21…` |
| DEV24 (exact A3/A4 panel) | 24 | 20 | `5afb4356…` |
| PATH2_12 (exact PATH2 panel) | 12 | 9 | `f469685b…` |
| NOVEL76 = FULL100 − DEV24 | 76 | 20 | `dcda84ca…` |

The nesting PATH2_12 ⊂ DEV24 ⊂ FULL100 is strict. Partitions and execution order were recomputed by prepare, the
pre-run auditor and the post auditor. Execution order is the 12 PATH2 rows first, then the remaining 88 in fixed100
order. Results are stored by canonical index.

## PATH2 replay barrier (first 12, inside the one job)

All 12 rows were **exact**, with no tolerance, against the sealed PATH2 rows on: effective bf16 checkpoint, A4 state
hash, theta0 = B0, sealed native language, G0 = sealed A4 (tokens and termination), detector (trigger, k, prefix,
argmax pairs), b0/b4 (tokens, H_eff, termination), all four per-token log-prob arrays and means, S_cons, margin,
winner, decision hash, G1 tokens and termination, and score-path owner hashes and locks.

There were 5 triggers (U0021_513 k0 A4 −1.236; U0023_666 k5 A4 −0.114; U0023_87 k15 theta0 +0.0005; U0023_664 k0
theta0 +0.985; U0092_101 k10 theta0 +0.042). The 7 no-trigger rows reproduced exactly. The live check on the first row
gave a loss difference of 1.5e−8 and a relative gradient difference of 0.89%, inside the 1e−5 / 2% limits. The remaining
88 rows entered the encoder only after the barrier passed.

## Reference-free controller (all 100 rows)

| | FULL100 | DEV24 | PATH2_12 | NOVEL76 |
|---|---:|---:|---:|---:|
| A4 exact no-op | 89 | 13 | 1 | **76** |
| Triggers / no-trigger | 5 / 95 | 5 / 19 | 5 / 7 | **0 / 76** |
| theta0 / A4 winners | 3 / 2 | 3 / 2 | 3 / 2 | — |
| G1 ≠ A4 transcripts | 3 | 3 | 3 | 0 |
| Trigger / theta0-winner dialogues | 3 / 2 | 3 / 2 | 3 / 2 | 0 / 0 |
| Margin quantiles [0, .25, .5, .75, 1] | −1.236, −0.114, 0.0005, 0.042, 0.985 | same | same | None |

**Why NOVEL76 is inert.** On every NOVEL76 row, and on every one of the 12 other DEV24 rows, the unchanged A4
reconstruction's native `detect_language` returned `<|zh|>`. That makes the AUTO prompt identical to forced-ZH, which
is the frozen exact A4 no-op: two zero-gradient updates, effective state = theta0. So live A4 = B0 on those rows
(verified 88/88). The two instances never disagree, the detector never fires, and G1 = A4 = B0. On NOVEL76 the
historical B0-AUTO transcript is also identical to B0 in every count. There was no AUTO/forced disagreement to
distill and none to guard.

The DEV24_OTHER rows reproduced the sealed A4 exactly (state checkpoint, tokens, text, language).

## FULL100 ASR (all 5 systems)

| System | ZH err | POI err | Mixed err | PIER | MER | EN-WER | ZH-CER | S / D / I | Caps |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| B0-FORCED | 1107 | 345 | 1483 | 0.4957 | 0.2570 | 0.5029 | 0.2193 | 339 / 1026 / 118 | 1 |
| B0-AUTO | 1179 | 284 | 1495 | 0.4080 | 0.2591 | 0.4152 | 0.2336 | 324 / 1053 / 118 | 1 |
| A2 | 1116 | 316 | 1464 | 0.4540 | 0.2537 | 0.4626 | 0.2211 | 351 / 985 / 128 | 1 |
| A4 | 1130 | 327 | 1489 | 0.4698 | 0.2580 | 0.4784 | 0.2239 | 344 / 1026 / 119 | 1 |
| **G1** | **1110** | **327** | **1469** | 0.4698 | **0.2545** | 0.4784 | **0.2199** | 324 / 1026 / 119 | 1 |

G1 versus B0: 18 POI corrections, 0 corruptions. Matrix-ZH retention 0.99926 (A4 0.99433), embedded-EN retention 1.0,
outside-POI harm 0.00074 (A4 0.00567), 0 severe truncations, cap delta 0.

G1 − AUTO (practical comparator, not a baseline): ZH −69, mixed −26, MER −0.0045, ZH-CER −0.0137, but POI +43, PIER
+0.062, EN-WER +0.063. AUTO keeps a much larger English benefit.

Partition context (ZH / POI / mixed):

| | DEV24 | PATH2_12 | NOVEL76 |
|---|---|---|---|
| B0 | 141 / 146 / 291 | 18 / 119 / 137 | 966 / 199 / 1192 |
| AUTO | 213 / 85 / 303 | 90 / 58 / 149 | 966 / 199 / 1192 |
| A2 | 164 / 128 / 297 | 50 / 103 / 154 | 952 / 188 / 1167 |
| A4 | 164 / 128 / 297 | 41 / 101 / 143 | 966 / 199 / 1192 |
| G1 | 144 / 128 / 277 | 21 / 101 / 123 | 966 / 199 / 1192 |

## Frozen gates

| Gate | Values | Result |
|---|---|---|
| ABSOLUTE_SAFETY (G1 vs B0) | ΔMER −0.0024 (≤ 0.01); ΔZH-CER +0.0006 (≤ 0.015); ZH ret 0.9993 (≥ 0.98); EN ret 1.0 (≥ 0.95); outside harm 0.0007 (≤ 0.03); POI corruption 0.0 (≤ 0.05); cap delta 0 (≤ 1); severe 0 | **PASS** (all assessable) |
| AGGREGATE_RESCUE | Z_B0 1107, Z_A4 1130, Z_G1 1110; X_ZH 23, required 12, R_ZH 20 | **PASS** |
| POI benefit | I_A4 18, I_G1 18, retention 1.00; MER_G1 − MER_A4 −0.0035 (≤ 0.005) | USEFUL_GAIN **PASS**; BENEFIT_RETENTION **PASS** |
| NOVEL76 BREADTH | R_plus 0, R_minus 0, R_net_76 0 (≥ 1 ✗); rescue utterances 0 (≥ 3 ✗); rescue dialogues 0 (≥ 3 ✗); C_utt = C_dlg = 1 (R_plus = 0; ≤ .50/.60 ✗) | **FAIL** |

Precedence INVALID → DAMAGE → NO_GAIN → OVERCONSERVATIVE → CONCENTRATED → SUPPORTED gives
**`P2_PATH3_SIGNAL_CONCENTRATED`**.

## Rescues, harms, concentration

- **Top rescues:** a single row, U0023_S0_664 (CSD0012, PATH2_12): ZH 22 → 2, mixed −20, POI unchanged.
- **New harms:** none. No row has more ZH, POI or mixed errors under G1 than under A4.
- **Concentration (positive rescue, R_plus = 20):** 100% from PATH2_12, 0% from the other 12 DEV24 rows, 0% from
  NOVEL76. U0023_S0_664 provides 100% of it (max row share 1.0), and so does dialogue CSD0012 (max dialogue share
  1.0). The same holds on FULL100, DEV24 and PATH2_12. NOVEL76: none.
- **LODO (NOVEL76, all 20 dialogues):** every removal leaves net rescue 0 (min 0, median 0, 0/20 positive; 18
  required). LODO_ROBUST false. This is secondary and descriptive only.
- **Bootstrap (FULL100, 2000 paired dialogue blocks, seed 240924, all 2000 draws valid; descriptive):**
  - G1 − A4: MER −0.0035 [−0.0124, 0], ZH-CER −0.0040 [−0.0145, 0], PIER 0 [0, 0], POI errors 0 [0, 0].
  - G1 − B0: MER −0.0024 [−0.0070, 0], ZH-CER +0.0006 [0, 0.0019], PIER −0.026 [−0.073, 0], POI errors −18 [−51, 0].

## Scientific interpretation

- **Inside the diagnostic cohort,** PATH2's result reproduces bit-exactly. G1 keeps all of A4's POI gain on fixed100
  (retention 1.0) and removes 20 of the 23 excess ZH errors, with no new harm.
- **Outside it, the controller is untested rather than refuted.** A4 itself never activates on the 88 non-PATH2 rows:
  native LID is zh, so the frozen no-op applies, and AUTO = B0 on NOVEL76. G1 therefore has no disagreement to
  adjudicate. The breadth failure is a population/activation fact: the whole A4 → G1 effect lives on the AUTO≠forced
  (D-type) rows. The fixed100's NOVEL76 contains none of them.
- **The entire fixed100 rescue is still one utterance in one dialogue.** The evidence that G1 rescues Mandarin
  generally does not grow.
- **The bottleneck has moved from controller design to evidence population.** Any further test of G1 needs rows
  where A4 actually adapts (native LID ≠ zh / AUTO ≠ forced). Choosing such a population is a new data-exposure and
  design decision. It is not authorized here.

## Next action

STOP. No G1/A4/threshold change, G2, full300, P3 or transfer. A further G1 breadth test would need a separately
frozen contract over a population where A4 activates, plus a human data-role decision. Neither is authorized here.

**Data:** exposed D-dev-select fixed100 only (already TTA1-exposed); DEV24 was used in the A3/A4/PATH chain. No fresh
validation, D-dev-confirm, D-test, router-calib, full300, P3 or transfer data.
