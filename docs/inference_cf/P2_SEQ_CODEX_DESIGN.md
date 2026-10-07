# P2-SEQ — implementation handoff

Design only, 2026-10-07; spec/config are frozen pre-outcome. Do not reopen terminal acoustic
diagnostics or modify the active core METHOD_CONTRACT. No runner was implemented in this session.

## Reuse map and actual gaps

| Reuse unchanged | Purpose |
|---|---|
| experiments/inference_cf_cached.py | Branch.step/cache lineage, gate pipeline, _edit_hook and processed selection |
| experiments/inference_cf_p2.py | forced matched zero-dose precedent, exact ordinary AUTO generate call |
| src/csasr/inference_cf/readout.py | p2dir_readout_v1 gradient/tangent provider, clone_scratch, isolation |
| core_r2.py/core_p1.py | E/R_B/localizer/prefix eligibility, selected_gate/argmax |
| csasr.models.whisper | frozen loading/waveform/feature extraction/encoder geometry |
| csasr.lss.sites | exact-site recorder/intervention, NormPreserve/current-query policy |
| evaluation canonical/pier/mer/retention | authoritative recognition and safety counts |
| inference_cf_p2_evaluate.py | reference-panel hash loader and count-sum dialogue bootstrap pattern |
| inference_cf_p2_audit.py / p2sel_audit.py | independent aggregation/manifest and step auditing patterns |
| P2_SEL_MINI_PANEL.json | exact100 saved IDs/order/20 dialogues |
| p2_A_r1_L16 rows/manifest, P0-R2/P2-DIR anchors | candidate AUTO reuse and baseline/model lineage |

Gap: current cached_decode still computes D0=hE-hB. No executed D2 free-decoding loop exists.
Implement one additive D2 sequence driver; do NOT relabel historical D0 output as D2, or change
cached_decode's historical behavior. All100 AUTO row hashes and IDs are sealed in p2_seq.json.
AUTO reuse is expected after semantic audit. S0 must be newly produced through the D2 driver's
zero-dose path; historical B0M is not a substitute. Expected two passes/ID in one allocation.

## Files to add, preserve and output

Add experiments/inference_cf_p2seq.py (CPU prepare/reuse audit inputs, GPU fixed decode),
inference_cf_p2seq_analyze.py (CPU metrics/rules), inference_cf_p2seq_audit.py (independent CPU
prerun/postrun), slurm/inference_cf_p2seq.sbatch, tests/test_inference_cf_p2seq.py.
No new objective/helper scientific formula is needed. Preserve frozen provider modules and all
historical reports/results/panel. Extend canonical documents only with actual execution status.
Output results/inference_cf/p2seq/run1, docs/inference_cf/P2_SEQ_REPORT.md; optional failed-stage
evidence-only P2_SEQ_TTA_HANDOFF.md. No TTA runner/objective design.

## Exact implementation order

1. Implement additive fixed alpha0/2 loop using cached.Branch and existing _edit_hook. Keep B/E/S
   isolated, original encoder immutable, S output feeds all branches. Single-query edits only.
   Gate uses RAW B probabilities, D2 uses processed script objective: do not unify these semantics.
2. At each t>=1 take ordinary-tensor pre-B snapshot using clone_scratch, then clean B/E calls and
   reference-free gate. Positive eligible alpha2 dose calls unchanged readout provider on snapshot;
   require bitwise clean-B logits/site identity and isolated cache/grad flags. Gate0/t0 skips D2.
   S0 alpha0 performs the exact same B/E/S forwards and zero-dose hook, skips snapshot/D2. S0 B/S
   logits must be bitwise equal. S2 zero gate after prior edits is identity to S, not to B.
3. Implement metrics via canonical facade, POI transition identity, corpus retention and lexical
   outside-POI correctness flips. Outside transcript changes are separate descriptive projections;
   never label them acoustic outside-candidate harm or hide unknown alignment behind zero harm.
4. Implement count-sum paired2000-draw dialogue bootstrap and exact four-label rule. Independent
   auditor owns its own aggregation/bootstrap/decision; no primary-analysis import.
5. Implement focused tests below. Resolve model/provider/suppression/environment/input hashes and
   AUTO reuse proof BEFORE outcomes. Keep reused AUTO original tokens/text and cap metadata.
   If full proof fails, all100 AUTO computed in same allocation; scientific rule stays unchanged.
6. Write resolved manifest and reuse_audit.json, independent PASS_TO_P2_SEQ, commit/push. Scientific
   run is forbidden until all these are remote. Submit exactly one sbatch max3h. No experimental
   baseline equivalence GPU job or repeated minipanel run.
7. In one job encode each utterance once for forced branches, run matched alpha0 and STEER fresh
   caches sequentially; free caches between systems, share only exact audio/window LID cache. AUTO
   ordinarily reused, otherwise generate with its own standard path. Log compact steps/edited
   sites and timings/VRAM. No references loaded by runner. Entire driver cannot be inference_mode
   decorated in a way that obstructs the readout provider's explicit autograd boundary.
8. CPU evaluate all100/three systems; independently audit. Failures stay in results and give
   INVALID, never drop IDs. Commit/push outputs/report/status. Always STOP. No full300/P3/TTA.

## Focused test contract

Reuse tests/test_inference_cf_cached.py, test_inference_cf_p2dir_directions.py,
test_inference_cf_p2sel.py, test_dg02_site.py, test_lss_sites.py, test_canonical_metrics.py.
New tests only for new coupling/accounting:
- panel byte hash/order/100unique/20x5; sealed reuse hashes/source semantic identity;
- S0/S2 same driver/Branch/hook/attention/token selection; alpha0 B/S logits bitwise throughout;
- exact AUTO call kwargs/defaults, historical postprocessing/EOS vs token-cap distinction;
- D2 provider objective/version/unit/tangent fallback unchanged, lazy current-prefix pre-B snapshot;
- E/R_B/localizer/L16/site/alpha2/NormPreserve unchanged, no forced-prefix editing;
- persistent cache/encoder/parameter isolation, restored flags, identical scratch-clean B site/logits;
- zero gate identity to S after earlier edit; first divergence follows actual edit;
- exact suppression/greedy/EOS/200cap; no future/reference/evaluator/TTA input/path;
- canonical PIER count/correction-corruption identity, lexical outside-POI edits vs harm distinction;
- deterministic shared dialogue count bootstrap, empty denominator handling;
- rule boundaries/precedence, positive gain with safety failure =>DAMAGE, zero edits=>valid NO_GAIN.

No new GPU acceptance run before this stage allocation. Runtime identity evidence is checked inside
that one job; failure invalidates the run, does not trigger a parallel search or alternate fallback.

## Audit/manifest/output contract

Pre-run independent PASS_TO_P2_SEQ verifies source/model/tokenizer/100 IDs/generation/reuse/test/
firewall/pre-outcome state and absence of TTA/search. Freeze model.files from P2-DIR; record exact
resolved current env/config/git/tree and all100 audio hashes. AUTO proof links P2 manifest and R2
model/env to exact historical rows; timestamps/new manifest do not silently replace old lineage.

Post-run P2_SEQ_AUDIT: PASS independently verifies all outputs, canonical counts/retention/outside
harm, exact deltas, bootstrap/rule, parameter/cache/zero dose/site evidence and first-divergence
attribution. Save hook pre/post bf16 sites only for actual edits (compact lossless NPZ plus hashes),
D2 scratch identity/fallback/call counts, per-system step lineage/query sequence, gate/energy/counts.
Use exact hook audit's norm semantics for energy, no scalar alpha-derived surrogate. No full KV
or vocabulary tensor dumps needed. Model/source/audio/row hashes identify every reused output.

Efficiency: timings from each system including its LID/D2/cache work, exclude/model-load setup
separately; cold/warm shared LID cache hits reported so measured ordering is visible. Report reused
AUTO runtime/VRAM unavailable rather than dividing a300-run total into100. Record allocated AND
reserved peak per new system/whole job, tokens/steps/autograd, utterances/sec. If wall limit hit,
preserve partial files and INVALID; no automatic extra allocation.

Decisions are those in spec/config; no CI-sign requirement for PROMISING. AUTO deltas always
reported independently, even when STEER is promising vs forced. If no gain/damage, handoff simply
summarizes exposed evidence and recommends fresh P2-TTA0 design; it may not freeze adaptation
objective/hyperparameters. Promising requests a separate confirmation design, no automatic run.
