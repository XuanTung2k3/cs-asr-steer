# P2-SEL-XA XA0 — cross-attention source-compatibility diagnosis (no steering)

Contract `595e174`; implementation `83845ee`; pre-XA0 `PASS_TO_P2_SEL_XA_XA0` (`e6cf957`). XA0 job Slurm 57844,
manifest at `e6cf957`: 50 s compute, 180 scratch λ=0.95 forwards, **0 backward / 0 LID / 0 readout / 0 steering**
(the 180 sealed raw P2-DIR `g_readout` vectors were reused); peak VRAM 3.69 GB allocated / 3.98 GB reserved.

**Frozen label: `P2_SEL_XA_INVALID`** (finite-difference validity failed). **`P2_SEL_XA_AUDIT: PASS (XA0)`**, with the
label independently reproduced. No compatibility gate selected; XA1/XA2 not run.

## Integrity (all pass)

At 180/180 rows:
- model-dtype q + u_source bitwise equals the recorded r, and r bitwise equals the sealed P2-DIR h_B;
- clean logits bitwise equal the sealed NONE logits;
- float64 reconstruction relative error is at most 0.0033 (≤ 0.03125);
- q is unchanged in the scratch forward, and exactly one scaled query per forward;
- the scaled u equals (0.95·u)_bf16;
- J recomputed with the unchanged `readout.objective` matches the sealed readout J within 1e-4;
- S/C sign consistency holds;
- groups are 42/18/55/5/60.

## Fixed finite-difference validity (frozen, failed)

The material subset is |P_nominal| ≥ 0.02 and |P_realized| ≥ 0.02 nat, i.e. |S_src| ≥ 0.4. It contains only
**4 rows over 2 dialogues** (required: ≥ 30 rows, ≥ 10 dialogues). On those 4 rows the observed-vs-realized sign
agreement is 0.75, nominal-vs-realized is 1.00, the median ratio is 0.81, and the relative RMS is 2.56. Under the
frozen rule this is INVALID for the first-order interpretation. No smaller λ, float32 rerun or alternative derivative
is allowed in this stage, and none was run.

## Why the subset is empty (descriptive; scales consistent with sealed lineage)

The source contribution is nearly orthogonal to the raw readout gradient. Medians [IQR]:

| Group | S_src | C_src | F_src | ‖g_J‖ | ‖u_source‖ | observed ΔJ | P_realized |
|---|---|---|---|---|---|---|---|
| EN TP (42) | −0.0025 [−0.076, 0.042] | −0.0016 | 0 [0, 0.042] | 2.43 | 0.93 | +0.0009 | +0.0004 |
| EN FN (18) | +0.047 [−0.144, 0.102] | +0.019 | 0.047 | 3.34 | 1.21 | −0.0027 | −0.0021 |
| ZH TN (55) | −0.013 [−0.069, 0.095] | −0.0028 | 0 | 1.91 | 1.75 | −0.0007 | +0.0007 |
| ZH FP (5) | −0.018 [−0.050, −0.009] | −0.012 | 0 [0, 0] | 1.59 | 1.60 | +0.029 | +0.0007 |
| EN-correct (60) | +0.075 [−0.003, 0.161] | +0.024 | 0.075 | 2.33 | 1.65 | −0.0027 | −0.0037 |

max |S_src| = 0.49 (5 rows ≥ 0.4). At this scale a 5% source change moves J by about 1e-3 nat, below bf16
logit quantization (observed |ΔJ| is typically about 0.01). For completeness only, since validity failed: the
supported-branch predicates would also have failed. All 5 FP have F ≤ 0.10, but 0/42 TP reach F ≥ 0.70 and
F(TP) − F(FP) = +0.030. Recall-only: 0/18 FN.

Firewall: exposed 180 D-dev-select positions only; no evaluator signal used in construction. P3 HELD.
