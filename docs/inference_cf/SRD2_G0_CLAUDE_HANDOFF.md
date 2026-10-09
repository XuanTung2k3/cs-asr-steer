# SRD2-G0 Claude implementation handoff

**Design ready; implementation and independent PASS_TO_SRD2_G0 are pending. No jobs ran in Codex.**
Implement this isolated single-token screen without changing any pinned historical source. Config
is the numerical authority; the spec defines metrics and the firewall defines phase permissions.
Do not turn retrospective MECH-LANG0 features into a different gate.

## Frozen anchors

- Initial local/remote commit: `b7e7b659b9ba06c305dbb422a0e8ecafa7fd6861`.
- Config: `configs/inference_cf/srd2_g0.json`.
- Config byte SHA: `sha256:218b3be997e8fafc102f7ec866b0a82b9925bb180e671f714c651bddee392eb9`.
- Population: `docs/inference_cf/SRD2_G0_POPULATION.json`; its byte/self/roster/selected hashes are
  pinned in config. Selected-ID SHA:
  `sha256:ff2e3054ed25ecf6ffc944552a98f368878568125cdb2ef446f64763408137ae`.
- Model weights: `sha256:a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`;
  tokenizer: `sha256:6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca`.
  All11 model/tokenizer/preprocess file hashes verified CPU in freeze; exact list is in config.
- Partition: `sha256:7daa50091056677286dd30933300cb9b1245ca778f969df2a95ccc06016dac02`.
- Suppression: `sha256:e93e88ef310ed47a78a9932f3d2d5ffd02d7bc3d59bca0e2e90524e34b2ef8e8`.
- Exact original R2/D2/DG-02/solver/cached/evaluator source bytes and installed Transformers
  forward source SHA are enumerated in config.source_sha256/environment. Freeze all new
  implementation files and environment again in the later run manifest; do not pretend that
  pending implementation hashes already exist. Record the design-freeze Git commit separately.
- `SRD2_G0_FREEZE.json` lists this freeze's new design/test file byte hashes. Verify it before
  implementation, then preserve these design bytes while adding implementation files.

## Required command interface (to implement, not existing launch commands)

One runner CLI, separate evaluator and independent auditor:

```text
python experiments/inference_cf_srd2_g0.py prepare --config configs/inference_cf/srd2_g0.json --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py manifest --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0_audit.py pre --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py capture --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py seal-a --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0_audit.py a --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py permutation --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py authorize-b --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py pulses --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0.py seal-b --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0_audit.py primary --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0_evaluate.py --out results/inference_cf/srd2_g0/run1
python experiments/inference_cf_srd2_g0_audit.py full --out results/inference_cf/srd2_g0/run1
```

Use `/home/tungnx/miniconda3/envs/acl1/bin/python`, project PYTHONPATH, one math thread,
offline local model loading, Torch2.10.0+cu128 / Transformers4.57.6. The future Slurm script may
use partition=mig, one `nvidia_h100_80gb_hbm3_3g.40gb`,4 CPUs,48GB RAM, walltime03:00:00,
PHASE=capture|pulses. **Do not inherit R2's12h walltime or P2-DIR's evaluator-in-GPU-job sequence.**
Each mode must reject an absent/failed audit, changed source/config/model hash or missing remote
seal required by its phase. Prepare serializes the safe four-key runtime projection, not the
full population file. New output directories must not overwrite existing/failed attempts.

## Exact execution order

1. Verify actual branch/HEAD/remote, clean tree and no conflicting job. Inspect pinned source
   interfaces; implement the components/tests in DESIGN only. Commit and push implementation.
   CPU preflight is source/schema/numerical only. Never inspect new references to debug coverage.
2. Prepare immutable run manifest/projection and source/model/data pins. Independent pre-audit
   must return **PASS_TO_SRD2_G0**. Check>=50GB durable storage, Job-A forecast<=9,000s and the
   two-job budget. No new scientific data role.
3. Submit **Job A only**. Capture400 baselines, every actual query, original full-replay R2 gates,
   numeric compatibility/probe evidence. No D2/steering/reference access. If global identity or
   coverage fails, preserve attempt and STOP; no alternate gate/path/panel.
4. Seal A completely, commit/push gate-seal/digest manifests, verify remote and durable raw data.
   Independent A audit PASS. Freeze B3 from the sealed complete structural-query gate multiset;
   push its assignment seal. Compute both fixed-formula and observed-component all-query costs.
   If either>9,000s, STOP for human compute revision; no smaller query panel or third job.
5. Materialize the minimal global authorization hash, with no row IDs/labels/tokens. Commit/push.
   Only then submit **Job B**. Reproduce old30 engineering inputs safely, outside new outcome
   denominators; run the exact four-arm pristine pulse matrix. No continuation or weight update.
6. Seal B fully, commit/push pulse/digest manifests; verify remote/durable raw archives. Independent
   PRIMARY PASS **before** the evaluator opens selected-role reference columns.
7. CPU evaluator computes all frozen outcomes/opportunities/denominators, bootstrap and provisional
   terminal predicates. Independent FULL audit recomputes them without primary decision imports.
   FULL failure=>INVALID; otherwise publish final terminal label with source/seal hashes.
8. Commit/push final report and audits, verify clean local==remote and no active stage job. STOP.

No scientific rerun for a negative result; no arm/horizon/dose/threshold/selection changes. A
failure needs a separate explicit human decision. No automatic submission of a later study.

## Gate and outcome contract summary

R2 g_old uses full causal replay **raw** script logits; D2 uses its original generation-processed
objective. The full/cached paths share audio/prompt/content/query; cross-path bounds are numerical,
while cached/scratch/zero/restore identities remain bitwise. g=0 bypasses the solver. Nonzero
targets are exact native chord requests e*,e*g,e*g_shuffled. Never multiply a solved coefficient
by g, add a minimum dose or redistribute unused energy.

Job A global coverage:400 rows,20 dialogues,>=1,000 structural queries,>=20 nonzero gates across
>=5 dialogues,>=99% finite providers on complete prefixes, all source/compatibility predicates.
Job B all positions remain fixed and include explicit no-ops. B1 matched requests>=95%; executed
energy errors obey original guards. Gated arms each lose<=10% planned squared energy and B2/B3
realized total ratio [.90,1.10], otherwise control comparison INCONCLUSIVE.

Post-seal opportunities:>=30 EN-confusion /30 EN-correct /100 ZH-correct, each>=10 dialogues;
>=40 actually edited B2 ZH-correct across>=10 dialogues;>=80% English unit mapping. Primary B2
power:>=5 corrections across>=3 dialogues. Comparator B1 requires>=5 corrections and>=4 ZH
corruptions across>=3 harmed dialogues; uninformative shuffle queries<=50%. All acceptance,
severe-damage and concentration predicates are numerical JSON in config. Missing denominators
cannot pass. Do not choose an alternative success metric if corrections are absent.

Only `SRD2_G0_SELECTIVE_FEASIBILITY` can recommend a separately frozen next stage. It is
preliminary development evidence, with adjusted dialogue-cluster uncertainty, not a significance,
safe-deployment or full-ASR claim. All other labels STOP. Correct-state no-ops are not active safety.

## Required final Claude report

Report config/design/implementation/run/seal/remote hashes, selected400 identity and exclusions,
all inventory/structural/gate/active denominators, every arm's corrections/corruptions and dialogue
coverage, B2/B1 retention/harm ratio, B2/B3 paired correction/utility/harm and planned/actual energy,
fallback/unreachable counts, per-dialogue/stratum dose distributions, bootstrap uncertainty,
EOS risk proxies, compute/jobs, every audit and the exact first-match terminal label. Report
mapping/unheard-scope exclusions, uninformative permutations and uncertain/small denominators.
Historical results remain unchanged. Stop with the terminal decision; do not run full decoding.
