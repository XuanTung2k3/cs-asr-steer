# P2-SEL-T pre-T0 audit — **PASS_TO_P2_SEL_T_T0**

Contract frozen at `fdace84` (spec, design, config byte-unchanged). Start: local HEAD = origin/cs-asr-steer-inf =
`fdace84f1ef7b241f42b5b029a7fd1201f68c118`, clean, empty queue. Implementation `d36d7f6`. Machine verdict
`results/inference_cf/p2sel_t/prerun_audit.json` (CPU only, no T outcome).

Verified: all 14 config source anchors; 100-way provider identity (generation config/mapping hashes, 100 IDs,
EN/ZH 50259/50260) and frozen alignment-head map present; parent P2-SEL-E `P2_SEL_E_DIAGNOSIS_AMBIGUOUS` with E0 and
final audit PASS; 180 ordered keys / 80 utterances / 20 dialogues; historical groups 42/18/55/5/60 from the frozen
P2-SEL scalars. **The frozen CENTER crop equals the audited P2-SEL-E `short_bounds` at all 180 keys**, and the E0
windows equal the E0 analysis rows. Also verified: null reuse (100-way, sum within 1e-6, 480000 samples), D2 seal,
mini-panel hash, no T-stage output, reference-free and contrast-free E_tok construction surface (AST), auditor
independence, tests present, T0 caps (≤360 LID, 0 autograd).

Reuse: E_1s, W, CENTER probabilities, null and groups from P2-SEL-E E0 (no long/center/null LID rerun); old gated D2
(P2-SEL C2), NONE, sealed D2 and R_B for a conditional T1. The T0 job replays only unedited B prefixes for current-query
attention (needed: E0 stored no attention vectors) and calls `native_lid` for LEFT/RIGHT crops only when their bounds
differ from CENTER (≤360). No steering, D2 or autograd.

Implementation notes (no scientific choice): the selection artifact's gate uses only `E_tok * R_B`; `C_tokctx` is
analysis-only. The T1 data-loading analysis and the T2 live-decode orchestrator are written only if T0 selects R_TOK;
the T1 label rule (`decide_t1`, including the EN no-regression clause) is frozen in code and tested now.

Tests: `tests/test_inference_cf_p2sel_t.py` (13) plus the P2-SEL-E (16), Q-contract (6) and P2-SEL (12) suites, all
passing. Firewall: exposed D-dev-select only. P3 HELD.
