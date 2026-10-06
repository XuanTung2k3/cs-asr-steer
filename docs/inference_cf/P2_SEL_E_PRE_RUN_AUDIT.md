# P2-SEL-E pre-run audit — **BLOCK_BEFORE_P2_SEL_E_E0**

Contract frozen at `11f4e63` (`P2_SEL_E_SPEC.md`, `P2_SEL_E_CODEX_DESIGN.md`, `configs/inference_cf/p2_sel_e.json`).
Start state: local HEAD = `origin/cs-asr-steer-inf` = `11f4e63b21e3ce4a642cb94e8da80cbbd91e3d1a`, clean tree, empty queue.
Machine record: `results/inference_cf/p2sel_e/prerun_audit.json`. No E0/E1/E2 job was submitted; no implementation
was written; no P2-SEL-E outcome exists.

## IMPLEMENTATION GAP (blocking): the frozen two-token-Q premise is false for the frozen provider

The spec (§1, §2 H_E3) and config (`E0.probabilities.provider`, `H_E3.eligibility`) state that `native_lid` softmaxes
exactly the two language-token logits, so `Q_local = pi_local(E) + pi_local(M) = 1`, and freeze the rule
"Q outside [0.999999, 1.000001] -> provider identity changed -> E0 invalid".

The actual frozen provider, byte-identical to the contract's own code anchors, does not do this:

* `experiments/inference_cf_p0_r2.py:native_lid` returns `softmax(logits[language_ids])`;
* every runner, including the P2-R run3 trace this stage reuses, passes
  `language_ids = tuple(sorted(set(lang_to_id.values())))`: all 100 Whisper language tokens
  (`inference_cf_p0_r2.py:292`, `inference_cf_p2.py:150`, `inference_cf_p2r.py:467`);
* already-exposed P0-R2 `local_support.pair_mass` (= pi_E + pi_M) on 5,082 rows from 67 files (no new inference):
  min 0.0008, 10th percentile 0.217, median 0.957, 90th percentile 0.997, max 0.99992. **100% lie outside the frozen
  interval.**

Consequences under the frozen text, if E0 were run:

1. The Q integrity rule would fire by construction, so E0 = `P2_SEL_E_INVALID` and the GPU job would be wasted.
2. H_E3 (low-confidence ratio artifact) is not structurally unavailable. Low Q is common under this
   provider, so its frozen ineligibility and the formal no-op status of R3 rest on a false premise. This
   changes the diagnosis precedence tree itself (H_E3 sits between H_E2 and H_E4). Resolving it requires a
   scientific decision: either keep H_E3 ineligible with Q merely reported, or make it testable with frozen
   criteria and an R3 formula. Either way the integrity rule must change.

Per spec §6 ("any implementation mismatch ... is an implementation gap: record it and stop before outcomes")
and the instruction not to redesign, this audit stops here. The remaining pre-E0 checks were not run, because
they cannot change a BLOCK.

## Required to unblock (a contract revision, not done here)

A revised, re-frozen P2-SEL-E contract must restate the Q identity for the actual 100-way provider. Two examples
of what that could mean:
- the integrity check becomes a recomputation identity (E within 1e-6 of P2-R), not Q ≈ 1;
- H_E3/R3 are either explicitly kept out of scope or given frozen criteria.

All other frozen elements (180 keys, groups 42/18/55/5, D2/R_B/alpha/L16/NormPreserve, E1/E2 rules) appear
implementable with the existing machinery.

Firewall: nothing consumed. P3 HELD.
