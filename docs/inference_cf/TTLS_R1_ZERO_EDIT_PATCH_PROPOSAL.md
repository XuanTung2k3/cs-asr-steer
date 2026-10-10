# TTLS-R1 zero-edit repair proposal (independent audit; NOT applied)

Original experiment sources and outputs are unchanged. This proposal is not Round 2 and does not revise a scientific
objective. The audit finds that `apply_steering` with nonzero gain and a zero direction computes
`h / (norm(h) + EPS) * norm(h)` in BF16. Division then multiplication is not bitwise identity. A zero temporary vector
therefore starts from a perturbed readout. The existing empty-mask test bypasses this computation and misses the defect.

The smallest numerical repair to review is to calculate the **ratio first**, with a denominator floor, then multiply:

```diff
- steered = steered / (new_norm + EPS) * orig_norm
+ steered = steered * (orig_norm / new_norm.clamp_min(EPS))
```

When the actual delta is zero and norm exceeds EPS, the ratio is exactly one. Unlike branching away from the edit on
`z == 0`, this preserves the derivative through the activation variable at initialization. A no-grad `z == 0` bypass
alone would fix inference controls while leaving the adaptation starting point wrong; a training bypass that returns
`h` disconnects the only trainable variable and is unacceptable.

This changes the BF16 numerical repair for nonzero edits as well. **Do not modify the frozen shared kernel or relabel
old outputs.** It requires an explicitly recorded numerical contract revision, zero/empty-mask/alpha-zero/zero-gain
identity tests, nonzero differentiability and norm tests, site-to-FFN consumption tests, and fresh source/seal identities.
The causal outputs of T1/T2/T4/T6 would then need re-execution as corrected results, with the same population,
pseudo-targets, objectives, masks, optimizer, steps and budget. A2, direct substitution, CD and raw reference scoring
are unaffected by this particular defect. Never choose between repairs using lexical outcomes.

The proposal is deliberately unapplied in this audit. The original experiment remains available for exact replay,
and independently reproduced metrics remain valid descriptions of those saved hypotheses.
