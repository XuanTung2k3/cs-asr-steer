# P2-SEL-LAC pre-LAC0 audit — **PASS_TO_P2_SEL_LAC_LAC0**

Contract frozen at `be460ce` (spec/design/config byte-unchanged). Start: local HEAD = origin/cs-asr-steer-inf =
`be460ce871ea3fda859e8a51a022de3148d29e18`, clean, empty queue. Implementation `4e6062c`; candidate seal committed and
pushed separately before any masked output. Machine verdict `results/inference_cf/p2sel_lac/prerun_audit.json`
(CPU only, no masked outcome).

Verified:
- all config anchors;
- parents (E, T, XA) final audits PASS, XA0 terminal `P2_SEL_XA_INVALID` with all 180 live-baseline identities
  (sealed NONE logits and h_B bitwise) true;
- P2-DIR seal;
- canonical partition hash with |V_E| = 37,858 and |V_M| = 1,667, and suppress/begin lists **disjoint** from V_E ∪ V_M;
- 180 ordered keys;
- `candidates_sealed.json` tracked, equal to HEAD, self-hash valid, and **independently reproduced at 180/180**
  (own sorted lowest-ID argmax on the sealed unmasked logits; W* = P2-SEL-T bounds[j]);
- **in-memory feature adapter bitwise equal to `batch_model_inputs` (features and attention mask) on all 80
  original waveforms**;
- mini-panel hash;
- no LAC output;
- reference-free construction surface, auditor independence, constants (log 17/3, log 11/9, ≤ 180 evaluations).

Reuse: sealed lossless unmasked NONE logits (no unmasked GPU replay), W*/j from P2-SEL-T T0, E_1s/R_B/groups from
P2-SEL-E E0, E_tok from T0. The LAC0 job only builds masked waveforms (hard zero on W*), the whole masked encoder,
and a fresh forced-ZH B prefix replay (exact-prefix LRU-2 per utterance), with no backward, D2, LID or steering.

Implementation note: the runner's prompt is the frozen CB (`[50258, 50260, 50360, 50364]`). It is a parameter only so
that tiny-model tests can pass their own vocabulary, and the runner asserts every input prefix starts with it.
The LAC1 data path and LAC2 decode are written only if LAC0 is supported; the LAC1 label rule is frozen in code
and tested.

Tests: `tests/test_inference_cf_p2sel_lac.py` (8) plus P2-DIR directions, P2-SEL, P2-SEL-E, P2-SEL-T and P2-SEL-XA
suites (78), all passing. Firewall: exposed D-dev-select only. P3 HELD.
