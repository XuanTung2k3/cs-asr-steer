# INTERNAL-0A — design and scientific preflight

**READY_FOR_INTERNAL0A — full capture awaits explicit human approval.**

Starting branch `research/steering-forensics`, HEAD `4c69723db458542314569b080a4601ddece2a418`, verified descendant of
`d45bf0ec`; initially clean, no active jobs. This separately authorized exploratory discovery study leaves ABTS-0,
its `K1_INSUFFICIENT` verdict, historical kernels and the core method contract unchanged. No local lexical outcomes
were inspected during preflight. New changes are intentionally uncommitted; no full discovery job was submitted.

## Population and exact identity

Use the 26 FULL300 utterances containing all 39 frozen reachable MM error onsets, including all 14 whose word is
absent **anywhere** from the matched global forced-English hypothesis. Those 14 occur across seven dialogues.
Add one hash-selected control from each eligible dialogue: outside the onset utterances, at least one canonical
Mandarin reference unit, and **every Mandarin reference unit MATCHes matched B0**. Controls may have English errors;
they are not claimed completely correct utterances. Order by SHA256(`240924:` + ID), then ID.

There are 19 eligible controls, producing **45 utterances / 19 dialogues**, in FULL300 canonical order. `CSD0538`
has neither an eligible control nor a target onset utterance; there is no replacement or role expansion. Selection
uses already-exposed labels offline, making this enriched diagnostic sampling, not blind confirmation or an
independent dialogue sample. No confirm/test/transfer data or external model is used.

| Identity | SHA256 (prefix `sha256:` omitted) |
|---|---|
| Parent FULL300 panel | `ff28efc645150d122410991d3b2449edc239b26111e97fda09ddf31e9340e2cb` |
| Ordered selected IDs | `3f5d15649c19b3efbdbb8b7ac42ed6cc579314a55ad0f5f8ec1436f166d0a437` |
| Runtime manifest | `e6fcfe7b11176eb1bd4423ebf37e652bedfb64e2e7b9c5d9609a56365f7604cb` |
| Configuration | `9df431fdfaebf8840e4998c8329e3181a08f307b7571001c3551e1e4cd4e6e71` |
| Whisper checkpoint | `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95` |
| Tokenizer JSON | `6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca` |

The runtime manifest pins every model/tokenizer/preprocessor file, source/config file, global capture, audio file,
waveform and generated baseline. `evaluation_panel.json` separately holds labels, references, original onset
registry and control-selection provenance. It is forbidden to the runner. Every runtime row is allowlisted;
neither onset locations nor control labels are transported.

## Decoder and crop protocol

Frozen local Whisper-large-v3 snapshot; CUDA BF16, eager attention, batch one, task transcription, no timestamps,
no gradients or parameter updates. Reuse `abts0_k1.matched_decode`: fresh `inference_cf_cached.Branch`, original
suppression, begin suppression at content step zero, and processed `argmax` with lowest-ID ties. Native AUTO runs
single-SOT language detection on that crop's encoder, checks lowest-ID agreement, then uses the resulting prompt.

For heard samples `N=min(original length,480000)` at 16 kHz, visit every 1-second start before N with a clipped
2-second crop; N<=2 seconds uses one crop. The corresponding 4-second crop is centered on the actual short-crop
midpoint and clipped to `[0,N)`. Retain the final partial window. No waveform padding beyond the original feature
extractor's normal 30-second padding. Reuse an encoder only for identical crop bytes; every decoder and AUTO
detection has a fresh cache. Deduplicate identical branch/bounds pairs while retaining their grid memberships.

Branches: **2_EN, 2_AUTO, 4_AUTO, 4_ZH**. Local cap is 64 content decisions, global cap 200. Capped outputs remain
in discovery denominators and are explicitly flagged. Preserve global B0/EN/AUTO from the sealed matched engine;
rerun global B0 per selected utterance and require identical tokens, termination and every raw-logit hash.

## Discovery, association and frozen selection

Discovery is the **union of every Latin word emission in all four branches**, including unlocated, capped,
boundary-adjacent and singly observed words. Preserve whole generated token sequences, exact BPE pieces, branch,
crop, token indices and flags. ASCII Latin/apostrophe/hyphen extraction inherits the historical ABTS regex.
Multi-canonical-unit renderings are retained but cannot receive a single-word primary literal label.

Reuse the ten published attention heads, unchanged prediction-query DTW, per-frame standardization and width-seven
median filter. Estimated full-audio intervals are local intervals plus crop offset; they are not gold timings.
Associate to any contiguous overlapping B0 Han/Latin block covering >=50% of the interval. No 200-ms onset gate,
R0, language detector or content-t>=1 restriction. Record t=0 separately as inaccessible to the existing steering
hook; this does not exclude discovery. Boundary risk means within 200 ms of either crop edge.

Occurrence groups require same casefold word and complete-link interval overlap >=50% of the shorter interval.
Never merge distant repeated words by surface alone. Cross-context support requires interior, uncapped occurrences
from a 2-second branch and 4_AUTO with **different physical crop bounds**. Identical clipped audio is not context
replication. 4_ZH is a comparator: it cannot veto selection or inflate selection rank.

Select at most one supported candidate per utterance. Abstain on conflicting/unlocated blocks, repeated-occurrence
ambiguity or a word already present in its B0 block. Rank by non-4_ZH branch count, non-4_ZH recurrence, earliest
interval, normalized surface, then candidate ID. Agreement measures stability, never correctness.

Shuffled control reuses donor decodes: within each dialogue, hash-order IDs and choose the nonzero circular
rotation minimizing total heard-duration mismatch. Scale donor occurrence times to recipient duration and rebuild
recipient association/selection. Preserve singleton identity controls as uninformative. This tests wrong-audio
pairing without another GPU pass; it is not a new masked-audio experiment.

## Evaluation and stopping rules

**Preflight correction:** unchanged ABTS A1 co-optimal B0 alignment makes 11/14 absent onsets ambiguous intrinsically
(only three labelable, in two dialogues). Making it primary would impose a discovery ceiling unrelated to local
recognition. Keep A1 as a secondary association diagnostic. Do not relax its historical definitions.

Primary occurrence truth instead requires two independent bilingual reviewers after the output seal, push and
PRIMARY audit. Review original audio/crops and canonical references, blinded to branch, agreement, selection and
control status. Require literal word identity and exact reference occurrence, not mere appearance elsewhere.
Disagreement/unclear language or occurrence is primary-incorrect. Review all target-word appearances, actual and
shuffled selections, one deterministic unfiltered word emission per utterance, and original absent-onset
audibility. The template and filled adjudications are hashed; missing reviews block evaluation, with no timing
proxy fallback. Literal labels must match canonical reference EN units. Distinguish translation, wrong English,
hallucination, wrong occurrence and ambiguity; automatic A1 labels alone cannot establish semantic translation.

Report unique occurrence recall **/39** and new globally absent recovery **/14**, also appearance-anywhere and
support attrition. Never add duplicate crop/branch hits. Primary discovery ignores selection and group-block
disagreements. Report selected precision, sample-weighted unfiltered emission precision, genuine repair counts,
dialogue breadth, ambiguous/unlocated coverage and false selections on all 19 controls. The latter measures
candidate errors, not Mandarin corruption under an intervention. All estimates are conditional on this enriched
panel. Bootstrap whole represented dialogues, seed240924, 2000 draws, descriptive 95% intervals; no significance
or multi-arm winner claim.

Frozen precedence:

1. `INTERNAL0A_INVALID`: incomplete matrix, source/model/audio/baseline/cache/firewall/seal/audit failure.
2. `INTERNAL0A_INSUFFICIENT`: fewer than 12/14 original absent onsets independently adjudicable.
3. `INTERNAL0A_NO_NEW_LEXICAL_DISCOVERY`: fewer than three new correct occurrences or fewer than three dialogues.
4. `INTERNAL0A_DISCOVERY_ONLY_SELECTION_INSUFFICIENT`: discovery passes, but <10 selections or <6 dialogues.
5. `INTERNAL0A_DISCOVERY_ONLY_VERIFICATION_UNSUPPORTED`: selected precision<.80; genuine repairs<6 or <4 dialogues;
   >1 control utterance falsely selected; or paired precision advantage over informative shuffled control <.20
   (undefined comparison also fails this gate).
6. `INTERNAL0A_DISCOVERY_AND_SELECTION_SUPPORTED`: all preceding conditions pass.

No label automatically authorizes activation steering. Low opportunity is not evidence that encoder information
is absent. Mandarin translations, correlated windows, lost coarticulation, attention timing bias and confident
shared errors remain principal risks.

## Implementation, engineering evidence and exact handoff

Additive files: `src/csasr/inference_cf/internal0a.py`, `experiments/internal0a.py`,
`experiments/internal0a_offline.py`, `configs/research_review/internal0a.json`, `slurm/internal0a.sbatch`,
`tests/test_internal0a.py`. Existing implementations and outputs were not modified. Complete source SHA pins are
in `runtime_manifest.json`; the runner checks them before execution.

**40 CPU tests passed**: new synthetic contracts plus existing ABTS cached-engine regressions. Syntax and shell
checks passed. The acl1 library path is required for the existing SciPy C++ runtime; no environment was modified.
Three bounded engineering attempts (58678,58679,58681) were retained while correcting pre-outcome analysis rules;
only the final attempt matches the frozen manifest. No tokens/text/candidate records were saved or lexically
interpreted in those smoke runs. Final job58681: 19-s Slurm wall, 11.83-s model load, 2.69-s measured work;
8 crop-branch decodes, exact cold-encoder/cache repeat, unchanged-weight probe, no gradients/hooks, and exact
15-query global B0 identity. Encoder calls cost48–53 ms; branch decodes53–124 ms; peak reserved VRAM3.75 GiB.

Full plan: **688 grid items, 1374 crop encoders, 2748 branch decodes, 45 global identity replays**, maximum178082
decoder queries. Forecast uses worst observed per-query/encoder cost, factor2, and a120-s load reserve:
**18.9 min expected; 122.1 min cap-based bound; 3-hour hard wall** on H100 MIG3g.40gb, 4 CPUs,48 GB RAM.
The tiny smoke is not a lexical or representative throughput sample. Stop on non-fitting forecast; do not shrink
population or change cap/doses. Raw attention goes to the new content-addressed archive under `/mnt/data`.

Environment for CPU commands:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
"$PY" experiments/internal0a.py forecast
```

**After explicit full-capture approval only:**

```bash
sbatch --output=results/research_review/internal0a/capture-%j.out \
       --error=results/research_review/internal0a/capture-%j.err \
       --export=ALL,PHASE=capture,APPROVED_FULL_CAPTURE=YES slurm/internal0a.sbatch
"$PY" experiments/internal0a.py select
"$PY" experiments/internal0a.py seal
```

Seal/push capture, candidates and provenance before reference scoring. Independent PRIMARY audit must verify
matrix completeness, crop/prefix/cache identity, model/source pins, waveform bounds, archives, selection/shuffle
reconstruction and no reference access. Materialize `{"status":"PASS","independent":true,"seal_hash":"..."}`.
The evaluator requires the exact seal/candidate bytes in a pushed `origin/...` ref:

```bash
"$PY" experiments/internal0a_offline.py review-template \
  --primary-audit results/research_review/internal0a/primary_audit.json \
  --sealed-ref origin/research/steering-forensics
# Two independent reviewers fill the blinded template; preserve the original.
"$PY" experiments/internal0a_offline.py evaluate \
  --primary-audit results/research_review/internal0a/primary_audit.json \
  --sealed-ref origin/research/steering-forensics \
  --adjudications results/research_review/internal0a/adjudications.json
```

Independent FULL audit recomputes /14 and /39, duplicate handling, review agreement, selected precision/control
errors, bootstrap and terminal label; it must not import the primary decision function. Preserve every failed
attempt. No protected roles, optimization, external inference or delivery pilot is authorized.

**READY_FOR_INTERNAL0A**
