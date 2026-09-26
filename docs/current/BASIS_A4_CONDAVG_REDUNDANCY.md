# BASIS-A4 Conditioning-Avg Redundancy Amendment

Status: FROZEN AMENDMENT. Conditioning-Avg is removed from the required A4
execution matrix and is not constructed or decoded.

## Code-traced equivalence

For decoder layer `l`, the traced Conditioning construction pools the same
eligible transcript-content positions in the same two teacher-forced states:

```text
Delta_l(u) = mean_{t in T(u)} r_l,t(x, c_E)
             - mean_{t in T(u)} r_l,t(x, c_M)
v_cond(l) = normalize(dialogue_balanced_mean_u Delta_l(u))
```

The proposed Conditioning-Avg construction is:

```text
mu_all(l,c) = mean_{u,t in T(u)} r_l,t(x, c)
v_cond_avg(l) = normalize(mu_all(l,c_E) - mu_all(l,c_M))
```

`T(u)` is identical in both traces: matrix and embedded transcript content
after the language/control suffix and before EOS, excluding system/user
template tokens, audio-prefix/placeholder states, padding, and special tokens.
The subtraction order and per-layer normalization are also identical. The
only residual distinction is cross-utterance weighting: Conditioning is a
dialogue-balanced mean of per-utterance token means, whereas the literal
Conditioning-Avg formula is token-weighted across utterances.

## Observed redundancy

The independent A4 audit traced both constructions and observed cosine
approximately `1.0` between the resulting directions (the exact audit scalar
was not persisted as a separate artifact). This is a weighting sensitivity,
not a distinct population, contrast, site, or intervention mechanism.

## Decision

No behavioral sweep is scientifically justified: it would duplicate the
Conditioning family while changing only an incidental aggregator weighting.
The final A4 direction set is exactly `{Raw, Conditioning}`. Any future
token-weighting sensitivity study must be labeled an explicit Conditioning
weighting ablation, never a separate Conditioning-Avg direction.
