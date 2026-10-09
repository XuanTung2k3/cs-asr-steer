# SRC-CF0-P — Claude implementation/execution handoff

## Frozen authorization scope

Initial local/remote HEAD:20cd36824822c356f910f42c9b5ca3a5fd5f56b6.
Working branch feature/inference-cf-steering; normal push target origin HEAD:cs-asr-steer-inf.
Configuration `configs/inference_cf/src_cf0_pilot.json` SHA256:
`sha256:f1881050b0718559d05a3ddaa0cb2832953e0fc7553be4c360d952d57f51aa87`.
Original blocked config/spec/panel/firewall/design unchanged. Read PILOT_SPEC, PILOT_DESIGN,
PILOT_FIREWALL and FULL300_SAFETY_PLAN in full. This handoff freezes the design, not a
runnable PASS audit. No scientific outputs were constructed or scored in Codex.

Inherited panel identity `sha256:2f66af05f61a73ff8dd7df825f86c6a7dc69389f3be807f312399a242464d371`;
file SHA256 `sha256:909e683c74acb86505e40f47843af6c3d90bf4776c6b1205f3c7527ceb46edca`.
FULL300 CPU inventory identity
`sha256:27d32c25a6df61c7a779f6c91f96a5fa8f4414d1ab75280bfb57675905355f4b`.
Model/checkpoint/tokenizer/preprocessing hashes are enumerated in config.model.files;
source and historical seal SHA256 pins in config.pins and config.historical_inputs.
No replacement source, model, audio, mask policy or dose is implicitly authorized.

## Exact inherited source pins

| Path | SHA256 |
|---|---|
| `src/csasr/inference_cf/core_p1.py` | `sha256:2e6a95cc00aea023ddc1a76cab5775812252c2748912f753c488b2b860fe18c4` |
| `src/csasr/inference_cf/s1_evidence.py` | `sha256:aee717f72afdfa6758903f87b85d53ff09f661426e1db681ba8ac6f15837ba1e` |
| `src/csasr/inference_cf/r0_regions.py` | `sha256:a30de91bc6d38a6a5b7f90ed80b4869b004aa292e8ec6e3fc12709347c5ee80d` |
| `src/csasr/inference_cf/lexical_compatibility.py` | `sha256:2ece3498492ffd948a6d7f68d1536df5a04ed8bc1668f9885e607abfeb8d5427` |
| `src/csasr/inference_cf/loc0_sites.py` | `sha256:17968cac6f5e7d3e5008db7e0aceaf8629112872d1c9f017318d002b13c1e9fc` |
| `src/csasr/models/hooks.py` | `sha256:4b5182bb3de913e1e79bf52a4a16c349bf29bb2a5242233bb0f321190adf256e` |
| `src/csasr/lss/sites.py` | `sha256:d8cc93c9aea05abd9557e8a70ac66c1d0ba35bbc2af008bcf4e3791406c75e48` |
| `experiments/inference_cf_cached.py` | `sha256:7c6b627eebe17892c4908c0dce11400eae03d0c4b8bd039a119f91ca0e317140` |
| `experiments/inference_cf_p2r.py` | `sha256:ed9d6a94584d7524cf880cd949da8f1cecd73a0273dd0d0f3845c47ec04eef7e` |
| `experiments/acoustic_s1.py` | `sha256:92b58f7eb7023b77b872547f2e4144e5786f2bf87ca908fea29d4fb7ec06fbfb` |
| `experiments/acoustic_s1_audit.py` | `sha256:eec16031fd9e22770361b0055df9ffd04da3ff96ec2918f056dfab9d653b2f8f` |
| `src/csasr/utils/provenance.py` | `sha256:bed06aaf4b192da0a99fb2877e617cd4bf890f4cef73d1f74fee810301ebede0` |
| `src/csasr/lss/specfreeze.py` | `sha256:e92017fe8ed17209d8bddf6fed448d539cfc71b3be8d3b5f27e65377a7fef3c0` |
| `src/csasr/inference_cf/cf_pilot_contract.py` | `sha256:a2a1c34b951662398b6e414a58342c9f9add456ca639d7d8487b5da2848ea799` |
| `experiments/inference_cf_src_cf0_safety_inventory.py` | `sha256:136dbad80f112e6eef2c5ea684195baa2274d7590c1bf973c8173731bac3bc10` |

## Implementation order and required CLI

1. Verify clean current HEAD, fetch remote, preserve newer intentional work; inspect the
   freeze and immutable blocked history. Implement only the isolated proposed interfaces.
   No S1 candidate reranking, vector variants, generalized backbone API or GPU census.
2. CPU preparation `python experiments/inference_cf_src_cf0_pilot.py prepare --config
   configs/inference_cf/src_cf0_pilot.json --run results/inference_cf/src_cf0_pilot/run1`.
   Validate panel/prefix/audio/S1 source seals; write safe runtime projection and exact crop
   byte hashes. Runner is not given offline panel/strata. This CLI does not yet exist.
3. `... manifest` uses provenance/specfreeze; pin ALL new runner/evaluator/auditor source,
   environment/installed Transformers forward source, model and config. Run synthetic tests,
   inspect diff, commit/push implementation/manifest. Any inherited scientific-source change
   requires review and explicit provenance rather than hash relaxation.
4. Independent `inference_cf_src_cf0_pilot_audit.py pre` must issue and push
   `PASS_TO_SRC_CF0_PILOT`. Verify hard historical apparatus/panel/zero/site/cache identity
   requirements and no reference imports. Failure STOP; no scientific allocation.
5. Future job1 PHASE=construct: `... construct`. Two independent cold repeats; both layers
   together; every180 query gets state/abstention records. No lexical references.
6. CPU `... seal-a`; commit/push immutable directions and manifest, verify remote ancestry.
   Independent auditor `construction` -> `SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)`.
7. Separate evaluator `..._evaluate.py gate-a` opens historical strata only AFTER seal/audit,
   computes frozen per-layer coverage/geometry/specificity. Emit detailed evaluator-only
   gate result and MINIMAL GLOBAL hashed PB_authorization.json. Push both. No target tokens
   or per-row correctness in authorization; no vector revision after gate evaluation.
8. If P-A fails: exact CONSTRUCTION_INSUFFICIENT or DIRECTION_NONSPECIFIC label plus
   independent audit; STOP, no job2. If gate passes, future job2 PHASE=pulse: `... pulse
   --authorization .../PB_authorization.json`. Use unchanged sealed directions, pristine
   owner cache, complete8+8+8 matrix and NONE/zero; failed layer cannot be selected but no
   planned arm omitted. No free continuation, optimizer or autograd.
9. CPU `... seal-b`; commit/push, confirm remote immutable pulse seal. Independent auditor
   `primary` -> `SRC_CF0_PILOT_AUDIT: PASS (PRIMARY)` BEFORE lexical reference access.
10. Separate evaluator `..._evaluate.py evaluate-b`: historical acceptable first-token sets,
    paired correction/control comparisons, full/active safety denominators, dialogue bootstrap,
    adverse stops, deterministic terminal and at most one selected configuration. No outcome-
    driven arm unions, thresholds, region policy or direction changes.
11. Independent auditor `full` recomputes numerics/cohorts/labels WITHOUT importing primary
    gate/decision code. Require `SRC_CF0_PILOT_AUDIT: PASS (FULL)`, otherwise INVALID.
    Write a future pilot report only then. Commit/push report and audits, verify clean==remote.

These command verbs/paths are implementation obligations, NOT existing runnable science commands.
Future Slurm launcher PHASE=construct|pulse; project interpreter
/home/tungnx/miniconda3/envs/acl1/bin/python, H100/MIG-compatible, max3h EACH, max2 jobs.
Forecast5–20min per phase; validate non-outcome throughput before production; STOP for human
compute revision if >3h. No silent arm/query reduction, no job parallelism across gates.

Existing CPU-only inventory is runnable now:
`python experiments/inference_cf_src_cf0_safety_inventory.py --output /tmp/src_cf0_inventory.json`.
It does not authorize the proposed FULL300 cached-attention GPU census. Compare output hash
with frozen inventory, inspect no new inference/evaluator path. Active Mandarin count UNKNOWN.

## Schemas and barriers

PILOT_DESIGN specifies runtime/manifest/construction NPZ and JSON/direction seal/audit/gate/
PB authorization/pulse NPZ and JSON/pulse seal/evaluation/audits. Full native BF16 state bytes,
float32 raw logits, masks, all source hashes and no-edit reasons must permit independent
reconstruction. Every query and arm remains represented. Candidate sets are absent.
Log opened paths; deny evaluator targets/strata in both runner phases. Allow R0 PRIMARY array
track_heard_intervals only. No oracle/MMS-FA timings or full-replay attention substitution.
P-A evaluator uses existing strata for GLOBAL coverage decision only, no new lexical scoring.
P-B comparisons all use identical reference-free joint-valid cohorts; target-only diagnostic
outputs cannot inflate a smaller paired-control denominator.

## Frozen stop/go and reporting

Follow exact config/spec precedence: INVALID -> CONSTRUCTION_INSUFFICIENT ->
DIRECTION_NONSPECIFIC -> global OBSERVED_DAMAGE -> SIGNAL_SAFETY_UNRESOLVED ->
power-qualified OBSERVED_DAMAGE -> MARGIN_ONLY -> CAUSAL_INSUFFICIENT.
Strongest label requires actual3 corrections/3 dialogues plus matched control/count/macro/LODO
and observed damage checks, and independent FULL audit. Adjusted CIs are descriptive, not
confirmation evidence. Four paired Mandarin opportunities cannot prove active Mandarin safety.
Report every edited Mandarin change, active denominators, unconditional/60 and uncertainty.

Future terminal report: config/source/panel/seal hashes; P-A coverage/geometry/specificity by
layer/stratum/dialogue;8-arm corrections/control advantage and energy; active correct-state
changes; adjusted intervals/LODO; exact terminal and selected key if any; safety unresolved;
no sequence/CF1/TTO/confirmation permission; jobs/runtime, git local/remote clean equality.

Allowed: same exposed D-dev-select180/80/20, sealed S1/R0/past apparatus artifacts.
CPU planning: existing exposed FULL300 only. Forbidden: new role/data/heldout/confirm/test/P3/
SEAME/other transfer, reference-dependent query selection, gold teacher forcing, changing A2
or adding controller. Failed runs/audit attempts/history remain intact.

No positive pilot label automatically launches any next stage. It permits ONLY proposing a
separately frozen expanded-safety study. Keep the FULL300 opportunity verdict UNKNOWN until
separately authorized exact cached-query census and appropriate independent evaluation.
