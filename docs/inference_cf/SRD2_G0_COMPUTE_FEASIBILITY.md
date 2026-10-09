# SRD2-G0 CPU-only resource forecast

No new throughput inference was run. Inputs are existing timing JSON/shard fields and audio
metadata only; no lexical correctness, gate distribution or reference was used to choose IDs.

| Measured historical input | Value |
|---|---:|
| P0-R2, H100 MIG3g.40gb, 300 utterances | 1,428.4166s |
| Historical heard duration | 4,908.1761s |
| Historical content queries | 13,561 |
| Historical real local-LID encoder calls | 11,855 |
| Historical donor-control LID calls | 8,750 |
| P2-DIR Exp1, 180 D2 calls + 540 pulses, 80 utterances | 113.3278s |
| D2 scratch gradient median / p95 | .03890 / .04346s |
| Individual pulse median / p95 | .01445 / .01770s |
| Historical D2 peak allocated VRAM | 4.01GB |
| Selected400 total / heard duration | 4,286.627 / 4,027.444s |
| Selected >30s utterances | 23 (retained) |

The whole P2-DIR job includes prefix replay, encoder, startup and serialization; it must not be
divided by180 and treated as marginal gradient time. Per-query counters establish the reusable
readout/pulse costs. R2's old job also included Ecf/Russian full replays and donor LID, absent in
this g_old-only capture. Unique identical crop LID calls may be cached within utterance; arbitrary
nearby windows are not interchangeable. Every one-second LID crop is still padded to30s, so crop
duration does not imply a cheap short encoder.

## Projection and hard stopping policy

Historical token density is13,561/4,908.1761=2.763tokens/heard-second. Multiplying by the selected
heard duration gives approximately11,128 content queries; actual structural queries can differ.
Allow twice this estimate:22,256 for conservative planning. This factor is an engineering reserve,
not a guarantee about the new population. Dialogue/token-count variation and long BF16 cache
copies are unresolved until Job A captures the reference-free inventory.

Job A has400 original encodings/greedy baselines and full causal replays, plus at most one native
LID call per unique valid selected window and one null call. Project approximately25–55min;
conservative3,200s includes startup, prefix probes, compatible-path recording, storage and audit
capture. Full no-cache attention replay is per utterance, not a complete prefix replay per query.
Batching different waveforms/window masks cannot change historical numerical semantics.

Job B has400 original encodings, pristine cached replay; one scratch readout-gradient call and
up to three pulse calls per structural query; restored-clean/zero engineering checks and all
output serialization. B0 is the stored clean query, not an extra direction arm. Budget .18s/query
(over four times median D2 cost, plus three p95 pulse costs, clean/restore/cache/solver/storage
reserve) and800s fixed load/encoder/old30-apparatus overhead. Typical projection is2,804s, and
conservative projection is4,806.08s (~80.1min). No benefit is assumed from a larger GPU than the
historical MIG; that resource is compatible. Expected data are ~4.6GB lossless raw-logit arrays
at11,128queries, ~9.2GB at22,256, plus states/gate evidence; verify durable free space before Job A.
Use explicit no-op references/deduplication without losing distributions. Provision>=50GB.

Immediately after Job A, recompute `0.18*actual_structural_queries+800`. Require<=9,000s
(2.5h, retaining30min to the hard3h ceiling) before authorizing Job B. Also require the following
exact observed-component forecast<=9,000s. Let c be CUDA-synchronized p95 cached clean-query
seconds, v durable archive bytes/write-second, a p95 original-encoder seconds, l model-load seconds
and w total audio-IO seconds measured in Job A. Per-query cost is
`max(.18, .0434598979 + 3*.0177027043 + 2*c + .04 + 478416/v)`;
fixed cost is `max(800, l + 400*a + w + 60 + 120)`. Multiply per-query cost by the **entire**
actual structural inventory, then add fixed cost. The478,416byte allowance is four lossless
51,866-token BF16 distributions, native state records and solver/JSON allowance per query.
Use full precision constants from config, not rounded report values. Missing/zero timing or
throughput evidence cannot authorize Job B. A slower observed component cannot be hidden by the
fixed formula. No favorable query
selection, silent truncation, lower cap or arm reduction. If recost fails, label COMPUTE_BLOCKED
and seek an explicit human compute/population revision **before** pulses.

Absolute max baseline inventory is400*200=80,000 queries, whose .18s/query projection is15,200s.
It does **not** fit. This worst-case risk is disclosed; the prospective design is conditional on
the reference-free complete inventory recost. The present metadata-based conservative forecast
fits, so no population revision is required now. A claim of unconditional worst-case feasibility
would be false. Neither job may exceed03:00:00; maximum two scientific jobs. A timeout/partial
matrix is preserved as INVALID, never completed with a third job or reduced population.

No autograd for model weights, no additional sign/layer/dose, AUTO decoder, acoustic donor study,
S1 masking, free continuation or full-ASR evaluation. Later CPU reference/evaluation/audit stages
do not require scientific GPU jobs.
