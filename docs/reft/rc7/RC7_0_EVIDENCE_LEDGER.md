# RC7-0A — Historical evidence ledger

Machine-readable companion: `RC7_0_EVIDENCE_LEDGER.json` (37 entries). Raw revalidation output:
`evidence/rc7_0_audit_archives.json`, produced by `scripts/rc7/rc7_0_audit.py archives` (sha256 `d8b13459…c673`)
run on A5000 on 2026-10-26. Path prefix `a5000:` = `/home/ubuntu/tung.nx/cs-asr-reft/a5000_experiments/`.

Naming: new text uses **LoReFT** (historical ID `R`, "E-ReFT") and **DiReFT** (historical ID `L`, "same-site LRA").
Both are the operators of the original ReFT paper (Wu et al., NeurIPS 2024) placed at the Whisper encoder output
after the final LayerNorm; they are not new adapter architectures. Historical IDs are kept in every artifact.

## 1. Revalidation (independent of the archived status report)

| Check | Result |
|---|---|
| core7h-20261020-v2 job outputs re-hashed against `status.json` | **187 / 187 match**, 0 mismatch, 0 missing |
| Selected checkpoints in `reports/development_metrics.csv` | **12 / 12 sha256 match** |
| Source / config / data bindings across all core jobs | single values: `2ca3c2d…`, `09ca42e5…`, `fc170e10…` |
| lra-gate-20261021-v2 new-arm checkpoints (L0, L2, L3, L4) | all present by sha256 under `jobs/<arm>/training/adapter.pt` |
| Sprint parent binding | DiReFT CE seed42 epoch3 `c531a22a…` = core `N1:L:CE` |
| Confirmation package (`confirmation/`, `confirmation_freeze.json`) | absent |
| `N4_analysis.json` | absent (N4 crashed before writing) |
| Cross-check of the eight V300 headline values in the RC7 brief | all eight match the archived CSV to the printed precision |

## 2. Classification

Axes are separated: *validity* (is the measurement correct), *completion* (did it run), *exposure* (was the
population used to choose it), *confirmation* (any independent replication), *reproducibility* (source identity).

| Evidence | Class | Key qualification |
|---|---|---|
| B0 frozen Whisper, V300 | VALID_REUSABLE | 1 retained loop |
| N1 CE: LoReFT, DiReFT, LoRA-PM, LoRA-Strong | VALID_WITH_LIMITATION | seed 42 only; LR/epoch chosen on V300 → V300 numbers are selection-biased; no independent confirmation |
| N3 WCE (ω=2): same four methods | VALID_WITH_LIMITATION | as N1, plus single ω |
| N2 G3/G4/G5/G8, G1 scalar | EXPLORATORY_ONLY | screen on V300 after N1; **N2_NO_GO** stands |
| N2 decision record | VALID_REUSABLE | frozen rule correctly applied; unchanged |
| G5_WCE | MISSING (SKIPPED by rule) | correct skip under NO_GO |
| N4 | ENGINEERING_INVALID | BF16→NumPy crash; no science produced; repaired in RC7-0C as a new record |
| N5 | BLOCKED | never started; includes seeds 43/44 and C500; not authorized here |
| Formal REPORT job | BLOCKED | core CSV/LaTeX/figures exist and are hashed |
| Sprint L0–L4 | EXPLORATORY_ONLY | post-hoc design after seeing V300; seed 42; **NO_MEANINGFUL_L4_BENEFIT** stands |
| Sprint decision record | VALID_REUSABLE | unchanged |
| energy_v1 runs, sprint v1 | EXPLORATORY_ONLY | not run / stopped / not admitted; superseded |
| E1 pilot (T394/V106) | VALID_WITH_LIMITATION | all-1500-frame policy ≠ later valid-frame policy → not a matched RC7 parent |
| CE-only admission | BLOCKED | historical role manifests missing |
| H100 decoder-site ReFT P0…P2-2, S0/S1, D1, D2 | VALID_WITH_LIMITATION | verdict docs only; artifacts on unreachable H100 storage; decoder layer 31, not encoder output |
| H100 P2-B, P2-E, P2-F1, P3-S, P3-M | MISSING | no source/artifacts anywhere inspected |
| Historical Train394/Inner106/Full500/Dev150 manifests | MISSING | never recovered; not reconstructible by resampling |
| A5000 role manifest T2000/V300/C500 | VALID_REUSABLE | all official-train; C500 has partial historical exposure (see exposure audit) |

## 3. Headline V300 development values (seed 42; development-selected, not confirmation)

| Method (historical ID) | Objective | CS-MER | MER | non-CS MER | Params |
|---|---|---:|---:|---:|---:|
| Frozen Whisper (B0) | — | 16.624 | 15.252 | 9.870 | 0 |
| LoReFT (R) | CE | 5.534 | 7.208 | 13.776 | 5,122 |
| DiReFT (L) | CE | 5.545 | 5.613 | 5.880 | 5,122 |
| LoRA-PM (LP) | CE | 6.273 | 6.150 | 5.670 | 5,120 |
| LoRA-Strong (LS) | CE | 5.042 | 5.135 | 5.502 | 1,310,720 |
| LoReFT (R) | WCE | 5.556 | 5.545 | 5.502 | 5,122 |
| DiReFT (L) | WCE | 5.427 | 5.502 | 5.796 | 5,122 |
| LoRA-PM (LP) | WCE | 6.273 | 6.142 | 5.628 | 5,120 |
| LoRA-Strong (LS) | WCE | 4.945 | 5.084 | 5.628 | 1,310,720 |
| Gated LoReFT G5 (vs G4) | CE | 5.416 (−0.054 pp) | 7.114 | 13.776 | 5,411 |
| Gated DiReFT L4 (vs L3) | CE | 5.491 (−0.011 pp) | 5.570 | 5.880 | 5,411 |

All values in %. The CS-MER spread among the three ~5K-parameter methods (5.43–6.27 %) and between objectives is far
below what one seed on 300 utterances (210 CS) can separate; no ranking among LoReFT/DiReFT is established.
LoRA-Strong is best on every recognition axis at 256× the parameters.

## 4. What this ledger does not change

N2_NO_GO, NO_MEANINGFUL_L4_BENEFIT, the 0.5 pp screening threshold, all failure records, and all archived files
are preserved verbatim. RC7-0 adds new records next to them; it does not edit them.
