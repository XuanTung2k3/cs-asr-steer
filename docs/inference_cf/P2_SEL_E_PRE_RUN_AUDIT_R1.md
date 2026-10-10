# P2-SEL-E pre-E0 audit, revision 1 — **PASS_TO_P2_SEL_E_E0**

Revised contract `b693b8d` (Q/H_E3 repair). The original BLOCK audit (`P2_SEL_E_PRE_RUN_AUDIT.md`,
`results/inference_cf/p2sel_e/prerun_audit.json`) is preserved unchanged. Implementation `0703523`.
Machine verdict: `results/inference_cf/p2sel_e/prerun_audit_r1.json` (CPU, no model inference, no E-stage outcome).

Verified: frozen spec/design/config/contract test and original block artifacts byte-equal to `b693b8d`; every
config code and evidence anchor; 100-way provider identity (generation_config sha256, canonical mapping sha256,
100 IDs, EN/ZH = 50259/50260); no Q≈1 rule; population hash and 60/60/60 keys; P2-SEL terminal label with S1 and
final audit PASS; exact P2-R (utterance, query) joins with P2-R E == P2-SEL E at 180/180; frozen-scalar groups
42/18/55/5; D2 seal; mini-panel hash; no e0/e1/e2 output; reference-free E0/E1/E_new code surface; auditor
independence; E1 constants.

Reuse: P2-R run3 traces (E, R_B, windows, lags, existing `acoustic.E_oracle`/timing), P2-SEL S1 C2 rows/logits as
C_old, P2-DIR NONE logits/states/sealed D2. The single E0 GPU job only extracts the missing local/null
probabilities, attention scalars and the 0.5 s crop; no steering, no readout.

Pre-registered before any E0 outcome:
1. **Conditional contract conflict.** E0 precedence can select R3 (H_E3), but config `E1.requires` and spec §3/§6
   authorize E1 only for R1/R2/R4. If E0 selects R3, the pre-E1 audit BLOCKs and the stage stops; no E1 runs.
2. A lag step with `E = None` (a fallback step in the P2-R trace) counts as missing history = 0.
3. A hypothesis whose bootstrap has fewer than 9,900 valid draws fails that criterion.
4. `E_long` is the recomputed 1 s current E, which must reproduce P2-R within 1e-6.

Tests: `tests/test_inference_cf_p2sel_e.py` (16) plus `test_p2_sel_e_q_contract.py` (6) and the P2-SEL suite (12),
all passing. Firewall: exposed D-dev-select only; P3 HELD.
