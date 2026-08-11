"""L0 - data roles, the spec freeze, and a measured compute pilot.

    python -m csasr.experiments.lss_l0_freeze --config configs/lss/l0_freeze.yaml --roles-only
    sbatch cs_asr_lss.sh l0

Nothing downstream is allowed to choose these things later: the roles fix who
may be trained on what, and the spec fixes the endpoint, the intervention sites,
the steering scale, the harm definitions and the seed map. The freeze is
*executable* -- the decoder site is verified on the real model and the outcome
definitions are exercised on fixtures inside this stage -- so sealing it means
the decisions were checked, not merely written down.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..lss import manifest as manifest_mod
from ..lss.artifacts import artifact_ref, write_json, write_parquet
from ..lss.balance import BalanceSpec, imbalance
from ..lss.eligibility import conversation_unit_counts, poi_index_consistency
from ..lss.gates import check, evaluate, exit_code, reported
from ..lss.prereq import require_prerequisites
from ..lss.roles import (
    TRAIN_ROLES,
    assert_roles_disjoint,
    build_roles,
    load_role,
    write_roles,
)
from ..lss.seeds import SeedMap, seed_for
from ..lss.sites import assert_site_reconstruction, site_report
from ..lss.specfreeze import build as build_spec
from ..lss.specfreeze import compare as spec_compare
from ..lss.specfreeze import load as load_spec
from ..lss.specfreeze import seal, verify
from ..utils.config import art
from ..utils.hashing import sha256_file
from ..utils.status import StageLock
from ._common import (
    base_parser,
    claim_stage,
    finish,
    last_prepare_context as _last_prepare_context,
    md_table,
    prepare,
    save_report,
    terminal_on_exception,
)

STAGE = "l0_freeze"


def _model_assets(cfg: dict) -> dict:
    """Hash every file that defines the model, not just the weights."""
    model_dir = Path(str((cfg.get("model") or {}).get("id", "")))
    assets: dict[str, str] = {}
    if model_dir.is_dir():
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "tokenizer.json", "tokenizer_config.json", "model.safetensors",
                     "special_tokens_map.json"):
            path = model_dir / name
            if path.is_file():
                assets[name] = sha256_file(path, max_bytes=1 << 20 if name.endswith(
                    ".safetensors") else None)
    return {"model_dir": str(model_dir), "assets": assets}


def _environment(cfg: dict) -> dict:
    import platform

    lock = Path(str((cfg.get("experiment") or {}).get("environment_lock", "")))
    if not lock.is_absolute():
        lock = Path(cfg["experiment"]["output_root"]) / lock
    payload = {"python": platform.python_version(),
               "lock_path": str(lock),
               "lock_sha256": sha256_file(lock) if lock.is_file() else None}
    for name in ("torch", "transformers", "sklearn", "numpy", "pandas"):
        try:
            payload[name] = __import__(name).__version__
        except Exception:                                      # pragma: no cover
            payload[name] = "unavailable"
    return payload


def _outcome_fixtures() -> dict:
    """Run the harm definitions on fixtures so the freeze is verified, not stated."""
    from ..lss.outcomes import candidate_unit_sets, newly_introduced_errors, utility

    sr = 16000
    spans = pd.DataFrame([{"unit_id": i,
                           "consensus_start_sample": int(i * 0.2 * sr),
                           "consensus_end_sample": int((i + 1) * 0.2 * sr)}
                          for i in range(4)])
    sets = candidate_unit_sets(spans, int(0.4 * sr), int(0.6 * sr))
    ref = "我 用 machine 做"
    recovered = newly_introduced_errors(ref, "我 用 做", ref, sets)
    corrupted = newly_introduced_errors(ref, ref, "我 用 做 做", sets)
    checks = {
        "deletion_recovery_is_a_correction": utility(recovered) == 1.0,
        "corruption_is_negative": utility(corrupted) == -1.0,
        "inside_outside_partition": sets.inside == (2,),
        "unknown_units_are_not_outside": candidate_unit_sets(
            spans.head(1), 0, int(0.2 * sr), all_unit_ids=[0, 1]).unknown == (1,),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _feature_guard_check() -> dict:
    from ..lss.features_contract import FEATURE_ALLOWLIST, assert_inference_safe

    frame = pd.DataFrame({f.name: [0.0] for f in FEATURE_ALLOWLIST})
    ok_clean = True
    try:
        assert_inference_safe(frame)
    except Exception:                                          # pragma: no cover
        ok_clean = False
    leaky = frame.copy()
    leaky["consensus_start_disagreement_ms"] = [1.0]
    rejected = False
    try:
        assert_inference_safe(leaky)
    except ValueError:
        rejected = True
    return {"accepts_allowlisted": ok_clean, "rejects_forbidden": rejected,
            "passed": bool(ok_clean and rejected)}


def _run_pilot(cfg: dict, log, limit: int | None) -> tuple[dict, list]:
    """Measure decode / cache / alignment throughput on a small sample."""
    from ..lss import pilot as pilot_mod
    from ..models.whisper import load_whisper
    from ..nat5h.coordinates import EncoderGeometry
    from ..nat5h.schema import RunIdentity

    pcfg = cfg.get("pilot") or {}
    n = int(limit or pcfg.get("utterances", 200))
    manifest = load_role(cfg, str(pcfg.get("role", "D-construct")))
    sample = manifest.sample(min(n, len(manifest)),
                             random_state=seed_for(cfg, "pilot_sample"))
    sample = sample.sort_values("duration_sec").reset_index(drop=True)
    log.info("pilot on %d utterances", len(sample))

    bundle = load_whisper(cfg)
    geometry = EncoderGeometry.from_bundle(bundle)
    identity = RunIdentity.from_cfg(cfg)

    results = list(pilot_mod.pilot_free_decode(
        bundle, sample, cfg, tuple(pcfg.get("decode_batch_sizes", (8, 16)))))
    results.append(pilot_mod.pilot_encoder_only(
        bundle, sample, int(pcfg.get("teacher_forced_batch_size", 8))))
    results.append(pilot_mod.pilot_teacher_forced(
        bundle, sample, list(pcfg.get("encoder_layers", [27])),
        int(pcfg.get("teacher_forced_batch_size", 8))))
    results.append(pilot_mod.pilot_decoder_site_capture(
        bundle, sample, list(pcfg.get("decoder_layers", [16])),
        int(pcfg.get("teacher_forced_batch_size", 8))))
    results.append(pilot_mod.pilot_steered_decode(
        bundle, sample, cfg, int(list(pcfg.get("encoder_layers", [27]))[0])))

    align_sample = sample.head(int(pcfg.get("aligner_utterances", 20)))
    results.extend(pilot_mod.pilot_aligners(
        align_sample, cfg, geometry, identity,
        list(pcfg.get("aligner_families", ["existing_ctc"])), bundle=bundle))

    storage = pilot_mod.pilot_storage(bundle, sample,
                                      list(pcfg.get("encoder_layers", [27])))

    # site verification on the real model, on one real batch
    site_layer = int((cfg.get("sites") or {}).get("verify_layer", 1))
    site_check = _verify_site(bundle, sample.head(2), site_layer)

    workload = _workload(cfg)
    projected = pilot_mod.extrapolate(results, storage, workload)
    payload = {
        "utterances": int(len(sample)),
        "results": [r.to_dict() for r in results],
        "storage": storage,
        "projected": projected,
        "decoder_site_check": site_check,
    }
    return payload, results


def _verify_site(bundle, sample: pd.DataFrame, layer: int) -> dict:
    """Assert the decoder site on the actual model, with real audio."""
    import torch

    from ..data.alignment import build_prefix
    from ..data.normalize import normalize_text
    from ..models.generation import teacher_forced_forward

    tok = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor)
    seqs = [prefix + tok.encode(normalize_text(t), add_special_tokens=False)[:64]
            for t in sample["transcript_raw"]]
    paths = sample["audio_path"].tolist()

    def forward():
        with torch.inference_mode():
            teacher_forced_forward(bundle, paths, seqs)

    return assert_site_reconstruction(bundle, forward, layer)


def _workload(cfg: dict) -> dict:
    """Item counts the projections are made against."""
    roles = cfg.get("roles") or {}
    targets = roles.get("train_conversation_targets") or {}
    # ~35 CS utterances per training conversation, measured on this corpus
    per_conversation = 35
    align_utts = int(sum(targets.values()) * per_conversation)
    return {
        "alignment_utterances": align_utts,
        "alignment_families": len((cfg.get("alignment") or {}).get("families", [])) or 2,
        "utility_candidates": int(targets.get("util-train", 25) * per_conversation * 4),
        "prompt_decodes": int(align_utts * 4),
        "cached_utterances": align_utts,
    }


def _run(argv: list[str] | None = None) -> int:
    parser = base_parser("L0 - roles, spec freeze and compute pilot",
                         "lss/l0_freeze.yaml")
    parser.add_argument("--roles-only", action="store_true",
                        help="build and validate the data roles, then stop")
    parser.add_argument("--pilot-only", action="store_true",
                        help="run the compute pilot against existing roles")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-verify an existing spec freeze and stop")
    parser.add_argument("--pilot-utterances", type=int, default=None)
    args = parser.parse_args(argv)
    if args.output_dir:
        raise SystemExit("--output-dir is not supported for LSS stages: it would "
                         "repoint data.manifest away from the frozen P0 artifacts")

    cfg, rdir, log = prepare(args, STAGE, claim=False)
    root = cfg["experiment"]["output_root"]
    spec_path = Path(root) / str(cfg["experiment"]["spec_freeze_path"])

    with StageLock(root, STAGE):
        claim_stage(cfg, STAGE, rdir)
        prereq = require_prerequisites(root, STAGE, force=args.force_prereq,
                                       overwrite=args.overwrite, log=log)

        if args.verify_only:
            verification = verify(spec_path)
            print(json.dumps(verification, indent=2, default=str))
            # a run that returns without finishing leaves the status at
            # `running` forever, which reads as an in-flight job to every
            # prerequisite check
            finish(cfg, STAGE, rdir, {"verify_only": verification}, None,
                   status_override="completed",
                   full_l0_pass=False,
                   next_action="run l0 without --verify-only to (re)seal",
                   taint=prereq["taint"])
            return exit_code("completed" if verification["ok"] else "failed")

        gcfg = cfg.get("gate") or {}
        criteria: list[dict] = []
        metrics: dict = {"prerequisites": prereq}

        # ---- data roles ---------------------------------------------------
        if not args.pilot_only:
            roles, report = build_roles(cfg)
            disjoint = assert_roles_disjoint(roles)
            spec = BalanceSpec.from_cfg((cfg.get("roles") or {}).get("balance") or {})
            train = pd.concat([roles[r] for r in TRAIN_ROLES], ignore_index=True)
            features = conversation_unit_counts(train)
            assignment = {c: r for r in TRAIN_ROLES
                          for c in roles[r]["conversation_id"].unique()}
            diagnostics = imbalance(features, assignment, spec)
            refs = write_roles(cfg, roles, report, diagnostics, stage=STAGE,
                               run_dir=rdir,
                               taint_reasons=prereq["taint"]["taint_reasons"])
            metrics["roles"] = report
            metrics["role_artifacts"] = refs
            log.info("roles: %s", json.dumps(
                {r: v["conversations"] for r, v in report["roles"].items()}))

            poi = poi_index_consistency(
                pd.concat([roles["D-construct"], roles["D-dev-select"]], ignore_index=True),
                sample=int(gcfg.get("poi_index_sample", 500)),
                seed=seed_for(cfg, "pilot_sample"))
            metrics["poi_index_consistency"] = poi

            criteria += [
                check("role_manifests_written", len(roles), 7, "=="),
                check("train_conversations_assigned", report["train_conversations"],
                      sum((cfg["roles"]["train_conversation_targets"]).values()), "=="),
                check("role_conversation_disjoint", int(disjoint["conversation_disjoint"]),
                      1, "=="),
                check("role_utterance_disjoint", int(disjoint["utterance_disjoint"]), 1, "=="),
                check("test_rows_in_non_test_roles",
                      disjoint["test_rows_in_non_test_roles"], 0, "<="),
                check("max_abs_smd", report["max_abs_smd"],
                      float(gcfg.get("max_abs_smd", 0.25)), "<="),
                check("max_categorical_tv", report["max_categorical_tv"],
                      float(gcfg.get("max_categorical_tv", 0.15)), "<="),
                check("balance_beats_random_median",
                      int(bool(report["balance"]["beats_random_median"])), 1, "=="),
                check("poi_index_equals_unit_id_rate", poi["agreement_rate"],
                      float(gcfg.get("min_poi_index_agreement", 1.0)), ">="),
            ]
        else:
            roles = {r: load_role(cfg, r) for r in TRAIN_ROLES}

        # ---- executable parts of the freeze --------------------------------
        fixtures = _outcome_fixtures()
        guard = _feature_guard_check()
        metrics["outcome_fixtures"] = fixtures
        metrics["feature_guard"] = guard
        criteria += [
            check("outcome_fixture_tests_passed", int(fixtures["passed"]), 1, "=="),
            check("feature_guard_rejects_forbidden", int(guard["passed"]), 1, "=="),
        ]

        if args.roles_only:
            # A roles-only run is a complete, successful, *partial* run: it did
            # what it was asked and nothing is wrong, so it exits 0. It is
            # terminal in its own right rather than `blocked`, and it carries
            # `full_l0_pass: false` so `prereq._status_problem` refuses it as an
            # L1a prerequisite no matter how its own checks came out.
            roles_gate, roles_status = evaluate(
                criteria, name=f"{STAGE}_roles_only",
                note=("Role construction only. The spec freeze was not sealed and "
                      "the pilot did not run, so this cannot satisfy L1a."))
            log.info("roles-only run complete (%s); the spec freeze was not sealed",
                     roles_status)
            write_parquet(criteria_frame(criteria),
                          art(cfg, "metrics", "l0_roles_only_criteria.parquet"))
            finish(cfg, STAGE, rdir, metrics, roles_gate,
                   status_override="completed_roles_only",
                   full_l0_pass=False,
                   next_action="run l0 without --roles-only to seal the spec and "
                               "measure the pilot",
                   taint=prereq["taint"])
            print(json.dumps(metrics.get("roles", {}).get("roles", {}), indent=2))
            return exit_code("completed_roles_only")

        # ---- measured pilot (needs the model) ------------------------------
        if args.dry_run:
            log.info("dry run: skipping the pilot")
            finish(cfg, STAGE, rdir, metrics, None,
                   status_override="completed_roles_only",
                   full_l0_pass=False,
                   next_action="run l0 without --dry-run",
                   taint=prereq["taint"])
            return exit_code("completed_roles_only")
        pilot_payload, _ = _run_pilot(cfg, log, args.pilot_utterances or args.limit)
        write_json(pilot_payload, art(cfg, "metrics", "l0_pilot.json"))
        metrics["pilot"] = pilot_payload
        site = pilot_payload["decoder_site_check"]
        projected = pilot_payload["projected"]

        criteria += [
            check("pilot_utterances_measured", pilot_payload["utterances"],
                  int(gcfg.get("min_pilot_utterances", 200)), ">="),
            check("decoder_site_reconstruction_rel_err", site["rel_err"],
                  float(gcfg.get("max_decoder_site_rel_err", 1e-3)), "<="),
            check("decoder_site_differs_from_block_output",
                  int(bool(site["site_differs_from_block_output"])), 1, "=="),
            check("projected_l1_alignment_gpu_hours",
                  projected["l1_alignment_gpu_hours"],
                  float(gcfg.get("max_projected_l1_alignment_gpu_hours", 12.0)), "<="),
            check("projected_l7a_label_gpu_hours", projected["l7a_label_gpu_hours"],
                  float(gcfg.get("max_projected_l7a_label_gpu_hours", 6.0)), "<="),
            check("projected_storage_gb", projected["storage_gb"],
                  float(gcfg.get("max_projected_storage_gb", 500.0)), "<="),
            reported("measured_decode_utt_per_s", projected["decode_rate_utt_per_s"]),
            reported("measured_alignment_utt_per_s", projected["alignment_rate_utt_per_s"]),
        ]

        # ---- seal the spec --------------------------------------------------
        from ..models.whisper import load_whisper  # noqa: F401  (already loaded above)

        referenced = [ref for ref in (metrics.get("role_artifacts") or {}).values()]
        spec = build_spec(
            cfg,
            roles={"version": (cfg.get("roles") or {}).get("version", "v1"),
                   "targets": dict((cfg["roles"])["train_conversation_targets"]),
                   "router_calib_split": dict((cfg["roles"]).get("router_calib_split") or {}),
                   "assignment_hash": (metrics.get("roles") or {}).get("assignment_hash")},
            sites=site_report_payload(cfg),
            model_assets=_model_assets(cfg),
            environment=_environment(cfg),
            referenced=referenced,
            extra={"pilot": {"projected": projected,
                             "decoder_site_check": site}},
        )
        sealed = spec_path.exists()
        drift = {"identical": True, "differing_sections": []}
        if not sealed:
            seal(spec, spec_path)
            log.info("sealed spec freeze at %s (sha256 %s)", spec_path, spec.sha256)
        else:
            # An existing freeze is left untouched -- that is the point of a
            # freeze -- but leaving it untouched while the live configuration
            # has moved on would let L0 report `passed` against decisions it is
            # no longer making. So the two are compared, not merely tolerated.
            drift = spec_compare(load_spec(spec_path), spec)
            if drift["identical"]:
                log.info("spec freeze at %s still describes this configuration",
                         spec_path)
            else:
                log.error("the sealed freeze no longer describes this run; "
                          "sections that drifted: %s. %s",
                          drift["differing_sections"], drift["resolution"])
        verification = verify(spec_path)
        # downstream prerequisite checks compare the freeze against a recorded
        # expectation; without a manifest there is nothing to compare against
        manifest_mod.publish(spec_path, None, stage=STAGE, cfg=cfg, run_dir=rdir,
                             taint_reasons=prereq["taint"]["taint_reasons"],
                             schema="lss_spec_freeze_v1")
        metrics["spec_freeze"] = {"path": str(spec_path), "drift": drift,
                                  **verification}
        criteria += [
            check("spec_freeze_sealed_and_verified", int(bool(verification["ok"])), 1, "=="),
            check("spec_freeze_referenced_artifacts_ok",
                  int(bool(verification["referenced_artifacts_ok"])), 1, "=="),
            check("spec_freeze_matches_live_config",
                  int(bool(drift["identical"])), 1, "=="),
        ]

        gate_payload, status = evaluate(
            criteria, name=STAGE,
            note=("Roles, the spec freeze and the measured pilot. A failure here "
                  "means a decision was not verifiable, not that the science is "
                  "negative."))

        rows = criteria_frame(criteria)
        save_report(
            art(cfg, "reports", "l0_freeze.md"), "L0 - roles, spec freeze, pilot",
            [
                ("Roles", md_table(pd.DataFrame(
                    (metrics.get("roles") or {}).get("roles", {})).T.reset_index())),
                ("Balance", "```json\n" + json.dumps(
                    (metrics.get("roles") or {}).get("balance", {}), indent=2,
                    default=str) + "\n```"),
                ("Pilot", "```json\n" + json.dumps(projected, indent=2, default=str) + "\n```"),
                ("Decoder site", "```json\n" + json.dumps(site, indent=2, default=str) + "\n```"),
                ("Gate", md_table(rows)),
            ],
        )
        write_parquet(rows, art(cfg, "metrics", "l0_gate_criteria.parquet"))
        # `--pilot-only` skips role construction, so it is a partial run too and
        # must not be able to unlock L1a on a subset of L0's checks
        full_pass = bool(status == "passed" and not args.pilot_only)
        finish(cfg, STAGE, rdir, metrics, gate_payload,
               artifacts=[str(art(cfg, "reports", "l0_freeze.md")), str(spec_path)],
               status_override=status if status != "passed" else None,
               forced_prereq=bool(prereq.get("forced_run")),
               full_l0_pass=full_pass,
               taint=prereq["taint"])
        log.info("L0 gate: %s (full_l0_pass=%s)", status.upper(), full_pass)
        return exit_code(status)


def site_report_payload(cfg: dict) -> dict:
    """Site description for the freeze, without requiring a loaded model."""
    sites = cfg.get("sites") or {}
    steering = cfg.get("steering_spec") or {}
    return {
        "encoder": {
            "module_path": "model.model.encoder.layers[l]",
            "tensor": sites.get("encoder_tensor", "encoder_block_output"),
            "capture": "forward_hook, output[0]",
            "candidate_layers": list(steering.get("encoder_candidate_layers", [])),
        },
        "decoder": {
            "module_path": "model.model.decoder.layers[k]",
            "tensor": sites.get("decoder_tensor", "decoder_post_cross_attn_residual"),
            "definition": "residual_in + encoder_attn(encoder_attn_layer_norm(residual_in))",
            "capture": {
                "residual_in": "forward_pre_hook on encoder_attn_layer_norm, args[0]",
                "attn_out": "forward_hook on encoder_attn, output[0]",
            },
            "intervention": ("encoder_attn hook returns attn_out + "
                             "(steer(site) - site)"),
            "rejected_site": "decoder_block_output",
            "rejected_site_reason": ("models.hooks.DecoderSteeringHook steers after "
                                     "the feed-forward block, which is not the tensor "
                                     "the proposal defines"),
            "depth_rescale": bool(steering.get("decoder_depth_rescale", False)),
            "candidate_layers": list(steering.get("decoder_candidate_layers", [])),
            "implementation": "csasr.lss.sites",
        },
    }


def criteria_frame(criteria: list[dict]) -> pd.DataFrame:
    """Gate criteria as a table whose cells are all strings.

    `_common.md_table` only formats columns with a float dtype; a mixed
    object column of floats and strings reaches `" ".join` and raises.
    """
    rows = []
    for c in criteria:
        rows.append({
            "group": str(c.get("group", "")),
            "name": str(c.get("name", "")),
            "value": ("" if c.get("value") is None else
                      (f"{c['value']:.4f}" if isinstance(c.get("value"), float)
                       else str(c.get("value")))),
            "comparison": str(c.get("comparison", "")),
            "threshold": str(c.get("threshold", "")),
            "passed": str(bool(c.get("passed", False))),
        })
    return pd.DataFrame(rows, columns=["group", "name", "value", "comparison",
                                       "threshold", "passed"])


def main(argv: list[str] | None = None) -> int:
    """Terminal in every path.

    An unhandled exception used to leave the status at `running`, which every
    prerequisite check reads as "a job is still working on it" -- so the stage
    was neither runnable nor reported as broken, and the pipeline stalled.
    """
    try:
        return _run(argv)
    except SystemExit:
        raise
    except BaseException as exc:                       # noqa: BLE001
        cfg, rdir, log = _last_prepare_context()
        if cfg is None:
            raise                                      # nothing to record it in
        return terminal_on_exception(cfg, STAGE, rdir, log, exc)


if __name__ == "__main__":
    raise SystemExit(main())
