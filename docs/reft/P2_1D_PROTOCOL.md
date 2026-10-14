# P2-1D — correction–damage diagnostic audit (protocol)

**Status.** Written and committed before the data-dependent run.

**Type.** Exploratory, post-hoc, CPU-only, observational analysis of the existing P2-1 dev150 decodes.
- No GPU, Whisper, weight loading, training or new decoding.
- No locked roles are opened.
- P2-0/P2-1 protocols, freezes, configurations, results and the verdict `P2_1_CORRECTION_PRESERVATION_FAILURE` are
  unchanged.
- dev150 is exposed and also served for checkpoint selection. No finding is an independent or causal claim.

Code: `reft/p2_1d_audit.py`. Tests: `tests/reft/test_p2_1d_audit.py`. Private outputs:
`/mnt/data/tungnx/cs-asr-reft/p2_1d/`.

## 1. Inputs

**Systems:**
- B0: the pinned P2 baseline (hash-checked against `configs/reft/p2_1_frozen.yaml`);
- R2: epochs 0–3 (selected epoch 2, read from `selection.json`, never changed);
- L1: selected epoch 3 (diagnostic control only);
- greedy files: secondary.

**References:** the 150 dev references are read with `reft.evaluate_supervised.dev_refs` (exposed D-dev-select IDs
only).

**Scorer.** The canonical scorer is reused unchanged:
- `score_system`, `paired` and `metrics` from `reft.evaluate_supervised`;
- `one`, `alignment`, `compare`, `en_error_kind` and `category` from `reft.text`;
- fillers and loop length from the configs.

## 2. Integrity (G1, G2): every item must hold or P2-1D is blocked

- **Hashes.** Pinned baseline hashes match. Input files are hashed: `report.json`, `evaluation.json`, `selection.json`,
  manifests and every decode file used.
- **Reproduction.** Rescoring R2 epoch 2 against B0 with `paired` and `metrics` reproduces `p2_1/report.json` exactly:
  - 150 utterances, 20 dialogues, 97 CS utterances;
  - 62 English corrections, 25 corruptions;
  - 73 new Mandarin errors, 103 Mandarin repairs;
  - 4,950 B0-correct non-filler Mandarin units;
  - outside damage 76 / 4,971;
  - MER, CS MER and Mandarin CER;
  - selected epoch 2.
- **Event table.** The reference-indexed event table reproduces the same counts independently, from per-unit
  correctness. Each utterance has exactly one event per canonical reference unit, with no duplicates and none missing.
  Insertions (hypothesis units with no reference index) are counted separately, by system and script, and are never
  attributed to a reference unit.

## 3. Definitions

- **Unit status.** For each reference unit k under system S: S is correct at k iff the canonical alignment maps k with
  `match`. The operation is `match`, `sub` or `del`.
- **Transition.** B0 → S per unit: repair (wrong → correct), corruption (correct → wrong), unchanged-correct or
  unchanged-wrong. Substantive English means EN-tagged and not a filler. Mandarin (for damage) means ZH-tagged and not a
  filler, exactly as in `paired`.
- **English category.** `correct`, or `en_error_kind` of the wrong unit under that system: `translation_like`,
  `omission`, `lexical_substitution`, `segmentation_artifact`, `boundary` or `other`.
- **Reference-unit distances.** These count units in the reference sequence and say nothing about acoustic time.
  - d_EN(k) = min |k − j| over substantive English units j.
  - A switch boundary lies between consecutive language-tagged units (EN or ZH; other units are skipped) whose tags
    differ; d_SW(k) = 0 for a unit adjacent to a boundary, and grows by one per unit away.
  - Bins: 0–1, 2–3, 4–7, ≥ 8, plus `no_en` for utterances with no substantive English unit.
- **Utterance flags.**
  - `EN_REPAIR`: at least one substantive English repair.
  - `ZH_DAMAGE`: at least one Mandarin corruption.
  - Other English changes (corruptions; wrong → wrong with a changed hypothesis surface or category) are reported
    separately.
- **Uncertainty.** Dialogue-cluster bootstrap over the same 20 dialogues (2,000 resamples, seed 42) for rates and
  shares, plus Wilson 95% intervals for small binomial denominators. No utterance-level significance claims.

## 4. Analyses

**D1 — Mandarin damage localization.**
- Totals; affected utterances and dialogues.
- Concentration: top-1/3/5 utterance shares, top-1/3 dialogue shares, and the median, 90th percentile and maximum per
  affected utterance.
- CS vs ZH_ONLY counts with B0-correct denominators.
- Damage rate by d_EN and d_SW bin, with denominators and intervals.
- Damage operation (substitution by Han, substitution by Latin, deletion).
- Insertions by system and script, and the decomposition of each dialogue's mixed-error change into reference-indexed
  changes vs insertion changes (the dialogue with +30 errors in particular).

**D2 — co-occurrence.**
- The 2×2 `EN_REPAIR` × `ZH_DAMAGE` table over all utterances and over CS utterances. Each cell reports: utterances,
  dialogues, English corrections and corruptions, new Mandarin errors, Mandarin repairs, and net mixed-error change.
- Severity distributions; damage-only share of total damage, with a dialogue-bootstrap interval.
- English net vs Mandarin net per utterance.

**D3 — epoch dynamics.**
- R2 epochs 0–3: full metrics and paired counts.
- Paired transitions: B0 → epoch 1, 2 and 3; epoch 1 → 2; epoch 2 → 3.
- Training CE and gradient norms from the manifest.
- Epoch-3 loop utterances (loop flag or non-termination): their contribution to ΔMER, and a **descriptive** sensitivity
  excluding them. That sensitivity is never an official score.

**D4 — English transitions.**
- The 7 × 7 B0-category → R2-category matrix over substantive English units (epoch 2; epochs 1 and 3 as well).
- For each B0 category: → correct, same category, different wrong category.
- Corruptions by R2 category.
- The fate of B0 English → English substitutions in particular.
- Private examples go only to the private directory.

**D5 — alignment sensitivity (never replaces the canonical scorer).**
- An alternative minimum-cost alignment with the opposite tie-break order (insertion before deletion before diagonal,
  instead of diagonal before deletion before insertion). Its edit cost must equal the canonical cost.
- Event stability: the fraction of canonical English repair/corruption events and Mandarin corruption events that also
  appear under the alternative.
- Derailment indicators for the 10 most-damaged utterances: hypothesis length ratio R2/B0, insertion counts, loop flag,
  and the normalized B0 ↔ R2 hypothesis edit distance.

**D6 — hypotheses H1–H4.** Each gets supporting evidence, contradicting evidence, unavailable evidence, and a
distinguishing experiment.

**Oracle (secondary, `ORACLE_ONLY — NOT DEPLOYABLE`).**
- Per-utterance reference-based choice between B0 and R2 epoch 2 (fewer mixed errors), with resulting MER, CS MER and
  new Mandarin errors.
- An upper bound only: never used to select, tune or claim anything.

## 5. Figures (private; anonymized aggregates)

| Figure | Content |
|---|---|
| F1 | damage concentration |
| F2 | repair × damage contingency |
| F3 | epoch dynamics |
| F4 | English transition heatmap |
| F5 | damage rate vs d_EN / d_SW, with denominators and intervals |

## 6. Gates and verdict

| Gate | Requirement |
|---|---|
| G1 | integrity and exact reproduction |
| G2 | event-table reconciliation, insertions separated, unit coverage exact |
| G3 | D1 complete |
| G4 | D2 complete |
| G5 | D3 and D4 complete, with the loop isolated |
| G6 | interpretation discloses uncertainty and makes no causal claims |
| G7 | CPU tests pass, input/output hashes recorded, private/public separated, no freeze or verdict modified |

`P2_1D_DIAGNOSTIC_COMPLETE` if every gate holds; otherwise `P2_1D_BLOCKED_<reason>`. No check is downgraded.
