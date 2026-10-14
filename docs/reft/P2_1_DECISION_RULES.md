# P2-1 — terminal decision rules (written before any P2-1 job was submitted)

**Purpose.** The frozen P2-1 design (`configs/reft/p2_1_frozen.yaml`, sha256 `fca2d7d2…`; freeze chain `p2_0.json →
p2_0_r1.json`) fixes the methods, data, optimizer, selection rule and metrics. It does not fix thresholds for the
owner-specified terminal verdicts. This document fixes them before any P2-1 outcome exists. Its sha256 is recorded in
`p2_1/AUTHORIZATION.json`. It is implemented in `reft/report_p2_1.py` (`decide`) and unit-tested in
`tests/reft/test_p2_1_report.py`.

It changes nothing in the frozen design. It is not part of the P2-0 freeze chain; it is a reporting rule.

## Evidence used

- Each arm's **selected checkpoint**, chosen by the frozen rule: lowest dev150 beam-5 genuine-CS MER, then overall MER,
  then Mandarin CER, then the earliest epoch.
- Evaluation under frozen-LID **beam-5**, paired against the **canonical P2 baseline B0** (pinned two-stage decode,
  MER 0.1199 / CS MER 0.1227).
- Greedy decoding, teacher-forced loss and non-selected epochs are reported, but never decide the verdict.
- dev150 is exposed exploratory development data that also served for checkpoint selection. No verdict is an untouched
  test claim.

## Definitions (selected checkpoint, beam-5, against B0)

**Δ.** Δ = system − B0, so a negative ΔMER is better. The CIs are dialogue-cluster bootstrap 95% intervals (2,000
resamples, seed 42) over the 20 dev dialogues.

**Preservation-OK(X)**, all of:
- Mandarin retention ≥ 0.99, i.e. new Mandarin errors ≤ 1% of the B0-correct non-filler Mandarin units;
- Mandarin CER(X) ≤ Mandarin CER(B0) + 0.002;
- loop utterances(X) ≤ loop utterances(B0) + 1.

**Useful(X)**, all of:
1. ΔCS-MER ≤ −0.005, with the upper CI bound < 0;
2. Δ overall MER ≤ 0 (point estimate);
3. net substantive English corrections (corrections − corruptions) > 0, with corrections in ≥ 4 dialogues;
4. the largest single-utterance share of the net ≤ 0.40;
5. Preservation-OK(X).

**Correction-but-damage(X):** there is correction evidence (gross substantive English corrections ≥ 10, or
ΔCS-MER ≤ −0.005), and either Preservation-OK(X) fails or Δ overall MER > 0.

**Uncertain-gain(X):** ΔCS-MER ≤ −0.005 but the upper CI bound ≥ 0, and Preservation-OK(X) holds.

## Verdict (first matching rule)

1. **`P2_1_ENGINEERING_BLOCKED`** if any integrity item fails:
   - a training, selection, evaluation or verification job did not complete;
   - zero-init reproduction of B0 < 150/150;
   - backbone weights changed;
   - the reloaded checkpoint decodes ≠ the recorded decodes;
   - a freeze or pinned-baseline hash check failed.
2. **`P2_1_REPRESENTATION_FEASIBLE`** if Useful(R2).
3. **`P2_1_WEIGHT_PEFT_ONLY`** if Useful(L1) and not Useful(R2).
4. **`P2_1_CORRECTION_PRESERVATION_FAILURE`** if Correction-but-damage(R2) or Correction-but-damage(L1).
5. **`P2_1_INCONCLUSIVE`** if Uncertain-gain(R2) or Uncertain-gain(L1).
6. **`P2_1_NO_FREE_DECODING_GAIN`** otherwise.

**Constraints on any verdict.**
- A positive verdict may recommend a P2-2 design. It authorizes nothing.
- No verdict is upgraded because of greedy or teacher-forced evidence.
- When both arms are Useful, the R2 vs L1 difference is reported descriptively, with its paired CI. It does not change
  the verdict.
