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
Candidate lambda .1/1 and weight2/5 are planning values, not selected recipes. The frozen primary pilot executes O1 only. Separate follow-up CLI exists in
`reft/a5000_followup/matrix.py`: default CPU dry-run, explicit --execute/time cap/clean committed source and
completed-primary checks before GPU work. Nine tiny-model method/objective paths are tested. The practical
comparator is broad decoder self/cross-attention q/v weight-LoRA, rank8/alpha16, 2,621,440 learned scalars,
no dropout/bias adaptation. Frozen encoder caching is still exact. It is an implemented practical-capacity
anchor, not an empirically established best-tuned or GPU-memory-validated strong baseline.

The auxiliary head has 2562 learned scalars and is excluded from deployment but counted in training.
Logical-batch normalization uses total ASR tokens, total weighted ASR tokens, and total known LID tokens
for their respective terms. Unknown labels keep CE and weight1. Head and adaptation state are checkpointed.
Actual proxy coverage: T394 11568/14145=81.78%, V106 3317/4081=81.28%; C150 not opened for labels.

CPU dry-run (no GPU workload):

```bash
ssh A5000 'cd ~/tung.nx/cs-asr-reft && PYTHONPATH=.:src ~/tung.nx/reft-a5000/bin/python -m reft.a5000_followup.matrix --method ereft --objective O2'
```

Future execution adds --execute --run-id UNIQUE --max-seconds N, after the primary finishes. It uses the
same aggregate six-hour budget until a separately authorized resource envelope is introduced. No follow-up
GPU cell was automatically started. A head-gradient test is engineering evidence, not a useful-LID result.

D1 foundations return group NLL and within-group NLL summing exactly to gold-token NLL, using declared script
vocabulary groups, and must be linked to actual decoded corrections/damage. D2 provides fixed/refitted probes,
training-only scaling, prefix-only control and disjoint held-out dialogues. No causal claim from probe accuracy.
Large-scale probe extraction is not launched. Decoder states must predict the next token under a gold prefix.

Future GoldKL bridge is separate: new A5000 beta1 can match the published decoder P2-1b coefficient, but this
is not authentication of the missing encoder recipe. Teacher logits use the same frozen model with adaptation
disabled, no duplicate backbone. Teacher cost and caches must be counted; no guaranteed free-decoding retention.

Additional D1/D2 extraction tools are in reft/a5000_followup/analysis.py: exhaustive vocabulary groups,
group probability mass and within-group probability, linked canonical free-decoding corrections/damage,
predictive state masks, and a declared constant-zero-encoder prefix-only shortcut baseline. Fixed/refit probes
use training-only scaling and disjoint held-out dialogue identities. No large-scale probe campaign was run.

A checkpoint-resume unit test exposed that the published Householder map contains a full orthogonal buffer
whose completion depends on the GLOBAL RNG as well as the seeded QR. Tests now match the runner's reset of
global seed42 immediately before adapter construction and serialize all buffers. Both E-ReFT and LRA resumed
optimizer trajectories match uninterrupted CPU trajectories exactly. The frozen primary runner already used
this seed reset and full-buffer checkpoints; it required no change.

Runner limit cases are tested: O2 lambda0 and O3 weight1 reduce to CE, with an unused auxiliary head allowed
to have no gradient at lambda0. The auxiliary head uses an explicitly reset seed42 shared across methods,
independent of adapter-construction RNG. No O2/O3 GPU experiment was run during the O1 campaign.
