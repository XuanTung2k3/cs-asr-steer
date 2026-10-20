# A5000 objective and diagnostic readiness

O1 pilot is the first priority. No O2/O3 or GoldKL run is authorized implicitly by its completion.

CPU-safe objectives exist in reft/a5000/objectives.py. O2 uses the downstream decoder-final state at the
position predicting y_t and a 1280x2 linear auxiliary head (2562 extra trainable scalars), not upstream frozen
encoder states. Ambiguous/control/numeric/partial-byte tokens are UNKNOWN and omitted from LID; ASR CE
still supervises content+EOS. Empty LID masks yield zero; lambda0 exactly follows CE.
O3 uses English-targeted script-proxy weights, normalized by total weighted valid tokens in the logical batch.
Weight1 reduces to CE. Do not call this linguistically inferred embedded-language weighting: script is a proxy,
names and romanized Mandarin remain ambiguity limitations. Per-token label coverage must be audited before runs.

The bounded follow-up design is E-ReFT/LRA/practical weight-LoRA x O1/O2/O3 with matched preservation (beta0).
Candidate lambda .1/1 and weight2/5 are planning values, not selected recipes. Existing pilot runners execute
O1 only. End-to-end follow-up CLI/strong practical LoRA integration is not yet claimed ready.

D1 foundations return group NLL and within-group NLL summing exactly to gold-token NLL, using declared script
vocabulary groups, and must be linked to actual decoded corrections/damage. D2 provides fixed/refitted probes,
training-only scaling, prefix-only control and disjoint held-out dialogues. No causal claim from probe accuracy.
Large-scale probe extraction is not launched. Decoder states must predict the next token under a gold prefix.

Future GoldKL bridge is separate: new A5000 beta1 can match the published decoder P2-1b coefficient, but this
is not authentication of the missing encoder recipe. Teacher logits use the same frozen model with adaptation
disabled, no duplicate backbone. Teacher cost and caches must be counted; no guaranteed free-decoding retention.
