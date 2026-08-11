"""The spec freeze: decisions that must be fixed before evidence is seen.

Plan v1 proposed a single `frozen_pipeline.json` that every stage mutated. A
file that is rewritten nine times is not a freeze. Here each stage seals its own
immutable artifact -- this one is `spec_freeze_v1.json` -- and the final lock
manifest at report time simply references them all by hash.

Sealing refuses to overwrite. Changing a frozen decision is possible, but only
through `supersede`, which writes a new numbered file recording what it replaces
and why, so the change is visible in the artifact tree instead of hidden in a
diff.

The self-hash is computed over a canonical serialization with the `sha256`
field blanked, so verification is a pure function of the file's own contents.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.hashing import sha256_bytes, sha256_file
from .artifacts import verify_refs, write_json

SPEC_FREEZE_SCHEMA = "lss_spec_freeze_v1"
VERSION_RE = re.compile(r"_v(\d+)\.json$")


class SpecFreezeError(RuntimeError):
    """A frozen decision was read, written or verified in an invalid way."""


@dataclass(frozen=True)
class SpecFreeze:
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return compute_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.payload)
        out["sha256"] = self.sha256
        return out

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.payload
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node


def canonical_payload(spec: SpecFreeze) -> str:
    """Serialization the hash is taken over: sorted keys, `sha256` blanked."""
    payload = copy.deepcopy(spec.payload)
    payload["sha256"] = ""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def compute_sha256(spec: SpecFreeze) -> str:
    return sha256_bytes(canonical_payload(spec).encode("utf-8"))


def build(cfg: Mapping[str, Any], *,
          roles: Mapping[str, Any],
          sites: Mapping[str, Any],
          model_assets: Mapping[str, Any],
          environment: Mapping[str, Any],
          referenced: Sequence[Mapping[str, Any]] = (),
          extra: Mapping[str, Any] | None = None) -> SpecFreeze:
    """Assemble every decision that must precede the first measurement."""
    from ..data.normalize import NORMALIZATION_VERSION
    from ..evaluation.pier import CATEGORIES
    from .features_contract import allowlist_payload
    from .outcomes import COUNT_INSERTIONS_AS_HARM, DEFAULT_MIN_OVERLAP_RATIO
    from .seeds import SeedMap

    experiment = cfg.get("experiment") or {}
    statistics = cfg.get("statistics") or {}
    steering = cfg.get("steering_spec") or {}
    alignment = cfg.get("alignment_prereg") or {}

    payload: dict[str, Any] = {
        "schema_version": SPEC_FREEZE_SCHEMA,
        "spec_version": str(experiment.get("spec_version", "v1")),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "supersedes": None,

        "eligibility": {
            "unit_source": "csasr.nat5h.units.build_reference_units",
            "eligible_unit": "embedded_english",
            "definition": ("an English content unit in an utterance whose reference "
                           "also contains at least one Mandarin content unit"),
            "rationale": ("P0 counted English units in monolingual <EN> utterances "
                          "too: 17465 of dev_select's 23726 POIs were not "
                          "code-switches"),
            "unit_index_convention": "0-based over all segmented units",
            "poi_index_equals_unit_id": True,
            "corpus_level_reports_unrestricted": True,
            "local_level_requires_consensus": True,
            "normalization_version": NORMALIZATION_VERSION,
        },

        "error_grouping": {
            "taxonomy_source": "csasr.evaluation.pier.CATEGORIES",
            "taxonomy": list(CATEGORIES),
            "deletion": ["deletion"],
            "substitution": ["wrong_language_substitution",
                             "phonetic_transliteration_or_script",
                             "same_language_substitution"],
            "other": ["boundary_error", "insertion_near_poi", "other"],
            "pooling_rule": "`other` is reported separately and never pooled",
        },

        "sites": dict(sites),

        "steering_scale": {
            "formula": "h <- h + alpha * scale * gain * direction, renormalized if norm_preserve",
            "source": "csasr.models.hooks.apply_steering",
            "direction_norm": "unit L2",
            "scale": str(steering.get("scale", "projection_std")),
            "scale_estimator": str(steering.get(
                "scale_estimator",
                "csasr.directions.accumulators.DirectionAccumulator.projection_std")),
            "scale_scope": "per (site, layer, direction_type); controls reuse the primary scale",
            "scale_artifact": {"path": None, "sha256": None},
            "rho_definition": "rho == alpha; rho=1 is one within-role projection SD",
            "decoder_depth_rescale": bool(steering.get("decoder_depth_rescale", False)),
            "decoder_depth_rescale_note": (
                "models.hooks.DecoderSteeringHook divides alpha by sqrt(num_layers); "
                "disabled at the post-cross-attention site so rho is comparable "
                "across sites. E4-era decoder results used it."),
            "norm_preserve": bool(steering.get("norm_preserve", True)),
            "energy_pre": "E_pre = (rho * scale)^2 * ||Delta||^2 * sum_t g_t^2",
            "energy_realized": "E_real = sum_t ||h'_t - h_t||^2 after renormalization",
            "matching_rule": ("controls match E_pre exactly; E_real is reported as a "
                              "distribution with a 10% tolerance band"),
        },

        "selector_features": allowlist_payload(),

        "outcomes": {
            "definition": "correctness flips against the reference, not transcript changes",
            "implementation": "csasr.lss.outcomes",
            "unit_assignment": "acoustic overlap with the candidate",
            "min_overlap_ratio": DEFAULT_MIN_OVERLAP_RATIO,
            "unaligned_units": "assigned to `unknown`, never to `outside`",
            "counts": ["n_corrected_inside", "n_corrupted_inside",
                       "n_corrected_outside", "n_corrupted_outside"],
            "utility": "U = n_corrected_inside - eta*n_corrupted_inside - kappa*n_corrupted_outside",
            "eta_kappa_primary": [1.0, 1.0],
            "eta_kappa_sensitivity": [[0.5, 0.5], [2.0, 2.0]],
            "outside_corrections": "reported as a diagnostic, never credited to utility",
            "insertion_rule": "attributed to the preceding reference position",
            "count_insertions_as_harm": COUNT_INSERTIONS_AS_HARM,
            "candidate_description": "independent booleans, not one categorical class",
            "deprecates": "csasr.evaluation.correction_harm.outside_region_edits",
        },

        "statistics": {
            "cluster_key": str(statistics.get("cluster_key", "conversation_id")),
            "cluster_justification": "conversation_id and speaker_id are 1:1 in CS-Dialogue",
            "bootstrap": {
                "impl": "csasr.evaluation.bootstrap",
                "n": int(statistics.get("bootstrap_resamples", 10000)),
                "seed": int(statistics.get("bootstrap_seed", 342)),
                "ci": float(statistics.get("ci", 0.95)),
                "paired": True,
            },
            "frontier": ("the whole curve is recomputed per resample; confirmatory "
                         "inference only at the frozen operating point"),
            "multiplicity": "one preregistered confirmatory comparison per gate",
        },

        "alignment_prereg": dict(alignment),
        "roles": dict(roles),
        "decoding": dict(cfg.get("decoding") or {}),
        "seeds": SeedMap.from_cfg(cfg).to_dict(),
        "environment": dict(environment),
        "model_assets": dict(model_assets),
        "referenced_artifacts": [dict(r) for r in referenced],
        "sha256": "",
    }
    if extra:
        payload.update(dict(extra))
    return SpecFreeze(payload=payload)


def seal(spec: SpecFreeze, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write the freeze once. Sealing over an existing file is refused."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise SpecFreezeError(
            f"{path} is already sealed. A frozen decision changes through "
            "supersede(), which records what it replaces, not by overwriting.")
    write_json(spec.to_dict(), path)
    return path


def load(path: str | Path) -> SpecFreeze:
    path = Path(path)
    if not path.is_file():
        raise SpecFreezeError(f"spec freeze not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != SPEC_FREEZE_SCHEMA:
        raise SpecFreezeError(
            f"spec freeze schema mismatch: file has {schema!r}, code expects "
            f"{SPEC_FREEZE_SCHEMA!r}")
    return SpecFreeze(payload=payload)


def verify(path: str | Path, *, expected_sha: str | None = None,
           check_artifacts: bool = True) -> dict[str, Any]:
    """Recompute the self-hash and (optionally) re-hash every referenced file."""
    path = Path(path)
    spec = load(path)
    recorded = spec.payload.get("sha256", "")
    actual = compute_sha256(spec)
    result: dict[str, Any] = {
        "path": str(path),
        "recorded_sha256": recorded,
        "computed_sha256": actual,
        "self_hash_ok": bool(recorded) and recorded == actual,
        "expected_sha256": expected_sha,
        "matches_expected": True if expected_sha is None else expected_sha == actual,
        "file_sha256": sha256_file(path),
    }
    if check_artifacts:
        refs = spec.payload.get("referenced_artifacts") or []
        verification = verify_refs(refs)
        result["referenced_artifacts_ok"] = bool(verification["all_ok"])
        result["referenced_artifacts"] = verification["results"]
    else:
        result["referenced_artifacts_ok"] = True
        result["referenced_artifacts"] = []
    result["ok"] = bool(result["self_hash_ok"] and result["matches_expected"]
                        and result["referenced_artifacts_ok"])
    return result


#: keys whose value legitimately differs between two builds of the same
#: decisions: they record *when* and *where* a build ran, not *what* it decided.
VOLATILE_KEYS = frozenset({"sha256", "created_at", "sealed_at", "run_dir",
                           "slurm_job_id", "supersedes", "environment"})


def _comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in VOLATILE_KEYS}


def compare(existing: SpecFreeze, candidate: SpecFreeze) -> dict[str, Any]:
    """Does an existing freeze still describe the decisions this run would make?

    `verify()` only asks whether a freeze is internally consistent, which a
    stale freeze always is. This asks the question that matters: has the live
    configuration drifted away from what was frozen? If it has, the honest move
    is `supersede()` -- recording what changed and why -- not writing a fresh
    `passed` status against yesterday's decisions.
    """
    old, new = _comparable(existing.payload), _comparable(candidate.payload)
    differing = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    return {
        "identical": not differing,
        "differing_sections": differing,
        "existing_sha256": existing.payload.get("sha256", ""),
        "candidate_sha256": compute_sha256(candidate),
        "resolution": ("" if not differing else
                       "the live configuration differs from the sealed freeze; "
                       "call supersede() to record the change, or revert the "
                       "configuration. L0 must not report `passed` against a "
                       "freeze that no longer describes it."),
    }


def assert_matches(spec: SpecFreeze, cfg: Mapping[str, Any]) -> None:
    """Catch a config that has drifted away from what was frozen."""
    problems: list[str] = []
    frozen_cluster = spec.get("statistics", "cluster_key")
    live_cluster = (cfg.get("statistics") or {}).get("cluster_key", "conversation_id")
    if frozen_cluster != live_cluster:
        problems.append(f"cluster_key {live_cluster!r} != frozen {frozen_cluster!r}")

    frozen_roles = spec.get("roles", "assignment_hash")
    live_roles = (cfg.get("_role_assignment_hash") or frozen_roles)
    if frozen_roles and live_roles != frozen_roles:
        problems.append("role assignment hash differs from the frozen one")

    frozen_norm = spec.get("eligibility", "normalization_version")
    live_norm = (cfg.get("experiment") or {}).get("normalization_version")
    if live_norm and frozen_norm and live_norm != frozen_norm:
        problems.append(f"normalization {live_norm!r} != frozen {frozen_norm!r}")

    if problems:
        raise SpecFreezeError("config no longer matches the spec freeze: "
                              + "; ".join(problems))


def supersede(old_path: str | Path, new_spec: SpecFreeze, reason: str) -> Path:
    """Write the next version of a freeze, recording what it replaces and why."""
    old_path = Path(old_path)
    old = load(old_path)
    match = VERSION_RE.search(old_path.name)
    if not match:
        raise SpecFreezeError(f"cannot derive a version from {old_path.name!r}")
    next_version = int(match.group(1)) + 1
    new_path = old_path.with_name(VERSION_RE.sub(f"_v{next_version}.json", old_path.name))
    payload = copy.deepcopy(new_spec.payload)
    payload["supersedes"] = {
        "path": str(old_path),
        "sha256": old.sha256,
        "reason": str(reason),
        "superseded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    payload["spec_version"] = f"v{next_version}"
    return seal(SpecFreeze(payload=payload), new_path)
