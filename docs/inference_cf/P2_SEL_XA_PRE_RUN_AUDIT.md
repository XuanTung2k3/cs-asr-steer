# P2-SEL-XA pre-XA0 audit — **PASS_TO_P2_SEL_XA_XA0**

Contract frozen at `595e174` (spec/design/config byte-unchanged). Start: local HEAD = origin/cs-asr-steer-inf =
`595e17436afa9544b487b0b956092ec80ceec734`, clean, empty queue. Implementation `83845ee`. Machine verdict
`results/inference_cf/p2sel_xa/prerun_audit.json` (CPU only, no XA outcome).

Verified: all 24 config anchors; parents `P2_SEL_E_DIAGNOSIS_AMBIGUOUS` and `P2_SEL_T_TOKEN_LID_NOT_DISCRIMINATIVE`
with audit PASS; P2-DIR seal (all row/NPZ/logit hashes) and Exp-1 audit PASS; **180/180 sealed raw `t{t}_g_readout`
vectors present (float32, 1280, finite) with valid D2 status**; 180 ordered keys; historical groups 42/18/55/5/60;
180 P2-SEL-T rows (E_tok); mini-panel hash; constants (λ=0.95, material subset 30 rows / 10 dialogues, 0 new
backward / 0 new LID); no XA output; reference-free construction surface (AST); auditor independence; tests present.

Reuse (no recomputation): sealed raw D2 gradients g_J, h_B and NONE logits (P2-DIR); E_1s, R_B and groups (P2-SEL-E
E0 / P2-SEL); E_tok (P2-SEL-T). XA0 job: per utterance one B replay with a passive DG-02 recorder (q, u_source, r)
and, per key, one isolated scratch forward with this query's encoder_attn output[0] scaled by 0.95. No backward,
LID, D2 steering or evaluator gradient.

Implementation notes (no scientific choice): S/C/F in `src/csasr/inference_cf/source_compatibility.py` (raw g, u
only); J recomputed on CPU with the unchanged `readout.objective` from lossless bf16 logits; the r_0.95 used for
P_realized is the model-dtype q + (0.95·u)_bf16 recorded inside the scaling hook. The XA1 data-loading analysis and
XA2 live decode are written only if XA0 is supported; the XA1 label rule is frozen in code and tested now.

Tests: `tests/test_inference_cf_p2sel_xa.py` (9) plus P2-DIR directions, DG-02 site, P2-SEL, P2-SEL-E and P2-SEL-T
suites (86), all passing. Firewall: exposed D-dev-select only. P3 HELD.
