"""Package-only BASIS-A3 post-run checks and analysis artifacts.

This script reads existing accepted result records. It never launches inference
and never changes scientific result records. It is intentionally usable after a
protocol blocker has stopped acceptance: blocked acceptance checks are recorded
without fabricating measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("results/basis_a3_raw_cond_scope_depth")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_records():
    for part in ("raw_r1", "raw_r2", "conditioning"):
        for path in sorted((ROOT / part).rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            metrics = data.get("metrics", {})
            if "poi_corrections" not in metrics:
                continue
            yield part, path, data, metrics


def package_metric_audit() -> Path:
    rows = []
    failures = []
    keys = {}
    for part, path, data, metrics in accepted_records():
        key = (
            data.get("dataset"),
            data.get("direction"),
            metrics.get("side"),
            data.get("layer"),
            data.get("scope"),
            data.get("rho"),
        )
        keys.setdefault(key, []).append(str(path))
        lhs = metrics.get("poi_net_utility")
        rhs = metrics.get("poi_corrections", 0) - metrics.get("poi_corruptions", 0)
        ok = lhs == rhs
        row = {"path": str(path), "key": list(key), "valid": ok}
        if not ok:
            row.update({"poi_net_utility": lhs, "expected": rhs})
            failures.append(row)
        rows.append(row)
    duplicate_keys = {str(k): v for k, v in keys.items() if len(v) > 1}
    payload = {
        "schema_version": "basis_a3_final_metric_consistency_v1",
        "status": "PASS" if not failures and not duplicate_keys else "FAIL",
        "accepted_record_count": len(rows),
        "unique_configuration_count": len(keys),
        "duplicate_configuration_keys": duplicate_keys,
        "formula": "poi_net_utility = poi_corrections - poi_corruptions",
        "invalid_rows": failures,
        "candidate_level_utility": "kept separate; not substituted for POI net utility",
        "source_semantics": str(ROOT / "metric_audit/semantics.json"),
    }
    out = ROOT / "metric_audit/final_metric_consistency.json"
    write_json(out, payload)
    return out


def load_conditioning():
    with (ROOT / "tables/conditioning.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def conditioning_figures() -> list[Path]:
    rows = load_conditioning()
    outdir = ROOT / "figures/conditioning"
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = ["cs_dialogue", "seame_dev_man", "seame_dev_sge"]
    labels = {"cs_dialogue": "CS-Dialogue", "seame_dev_man": "SEAME dev-man", "seame_dev_sge": "SEAME dev-sge"}
    layers = [24, 26, 27, 31]
    rhos = [0.25, 0.5, 1.0]
    scopes = ["global", "oracle_local"]

    def make_metric(name: str, ylabel: str, filename: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, dataset in zip(axes, datasets):
            for rho in rhos:
                for scope in scopes:
                    sub = [r for r in rows if r["Dataset"] == dataset and float(r["rho"]) == rho and r["Scope"] == scope]
                    sub.sort(key=lambda r: int(r["Layer"]))
                    ax.plot([int(r["Layer"]) for r in sub], [float(r[name]) for r in sub], marker="o", label=f"{scope}, ρ={rho:g}")
            ax.set_title(labels[dataset])
            ax.set_xticks(layers)
            ax.set_xlabel("Decoder layer")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = outdir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    paths = [
        make_metric("PIER", "PIER", "pier_by_layer_rho.png"),
        make_metric("MER", "MER", "mer_by_layer_rho.png"),
        make_metric("Matrix-Ret", "Matrix retention", "matrix_retention_by_layer_rho.png"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for scope in scopes:
            sub = [r for r in rows if r["Dataset"] == dataset and r["Scope"] == scope]
            sub = [r for r in sub if r["Outside"] not in ("", "NA", "None")]
            if sub:
                ax.scatter([float(r["Corr"]) for r in sub], [float(r["Outside"]) for r in sub], label=scope, alpha=0.8)
        ax.set_title(labels[dataset]); ax.set_xlabel("POI corrections"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Outside harm")
    handles, labels_ = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "correction_damage_frontier.png"
    fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for scope in scopes:
            sub = [r for r in rows if r["Dataset"] == dataset and r["Scope"] == scope and float(r["rho"]) == 0.5]
            sub.sort(key=lambda r: int(r["Layer"]))
            ax.plot([int(r["Layer"]) for r in sub], [float(r["PIER"]) for r in sub], marker="o", label=scope)
        ax.set_title(labels[dataset]); ax.set_xticks(layers); ax.set_xlabel("Decoder layer"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("PIER at rho=0.5")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "cross_dataset_conditioning_comparison.png"
    fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def package_cross_corpus() -> tuple[Path, Path]:
    observations = [
        ("Raw encoder useful region", "none at rho=.5; late decoder signal dominates", "narrow late Oracle-local L27–L28 PIER response", "no positive PIER region; local reduces damage"),
        ("Raw decoder useful region", "late, especially L27–L30 Global/Local tradeoff", "small late response", "small L24/L27 response"),
        ("Raw L27 correction signal", "decoder clear; encoder local remains collateral-limited", "encoder Oracle-local positive PIER (7 corrections, 3 corruptions)", "encoder local has no positive PIER (0 corrections, 7 corruptions)"),
        ("Raw Global→Local rescue", "clear on late decoder", "strong L27 rescue versus Global, narrow benefit", "retention rescue without positive PIER"),
        ("Cond L26 useful", "modest, narrow-dose", "weak", "weak"),
        ("Cond L27 useful", "modest Local at rho=1", "weak Local at rho=1", "absent Local"),
        ("Cond Global→Local rescue", "strong at L31", "strong at L31", "strong at L31"),
        ("Cond L31 destructive", "yes at rho≥.5", "yes at rho≥.5", "yes at rho≥.5"),
    ]
    csv_path = ROOT / "tables/cross_corpus_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Observation", "CS-Dialogue", "SEAME dev-man", "SEAME dev-sge"])
        writer.writerows(observations)
    md_path = ROOT / "tables/cross_corpus_summary.md"
    lines = [
        "# BASIS-A3 Cross-Corpus Summary",
        "",
        "These observations summarize existing accepted exploratory metrics; they do not repair the scientific snapshot mismatch.",
        "",
        "| Observation | CS-Dialogue | SEAME dev-man | SEAME dev-sge |",
        "|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in observations]
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def package_runtime() -> tuple[Path, Path]:
    jobs = [
        ("51625", "preflight", "mig", "00:00:04", "FAILED", False, "logs/basis_a3_basis_a3_preflight_51625.err"),
        ("51626", "preflight", "mig", "00:00:49", "COMPLETED", False, "logs/basis_a3_basis_a3_preflight_51626.out"),
        ("51627", "CS R1 encoder preliminary", "mig", "00:00:13", "FAILED", False, "logs/basis_a3_basis_a3_cs_enc_r1_51627.err"),
        ("51628", "CS R1 decoder preliminary", "mig", "00:10:09", "CANCELLED", False, "logs/basis_a3_basis_a3_cs_dec_r1_51628.err"),
        ("51629", "CS R1 encoder retry", "mig", "00:09:17", "CANCELLED", False, "logs/basis_a3_basis_a3_cs_enc_r1_retry_51629.err"),
        ("51632", "CS R1 encoder retry", "mig", "00:05:11", "CANCELLED", False, "logs/basis_a3_basis_a3_cs_enc_r1_2w_51632.err"),
        ("51633", "CS R1 decoder retry", "mig", "00:05:10", "CANCELLED", False, "logs/basis_a3_basis_a3_cs_dec_r1_2w_51633.err"),
        ("51677", "encoder acceptance harness (bug)", "mig", "00:00:16", "FAILED", False, "logs/basis_a3_basis_a3_encoder_accept_51677.err"),
        ("51679", "encoder acceptance harness (tolerance bug)", "mig", "00:00:15", "FAILED", False, "logs/basis_a3_basis_a3_encoder_accept_51679.err"),
        ("51680", "encoder acceptance suite", "mig", "00:00:16", "COMPLETED", True, "logs/basis_a3_basis_a3_encoder_accept_51680.out"),
        ("51682", "SEAME dev-man encoder R1 + R2 repair", "mig", "00:08:12", "COMPLETED", True, "logs/basis_a3_basis_a3_man_repair_51682.out"),
        ("51683", "SEAME dev-sge encoder R1 + R2 repair", "mig", "00:07:49", "COMPLETED", True, "logs/basis_a3_basis_a3_sge_repair_51683.out"),
        ("51638", "CS-Dialogue R1 encoder", "mig", "02:34:51", "COMPLETED", True, "logs/basis_a3_basis_a3_cs_enc_r1_cache_51638.out"),
        ("51639", "CS-Dialogue R1 decoder", "mig", "02:05:36", "COMPLETED", True, "logs/basis_a3_basis_a3_cs_dec_r1_cache_51639.out"),
        ("51649", "SEAME dev-man R1 encoder", "mig", "00:15:22", "COMPLETED", True, "logs/basis_a3_basis_a3_man_enc_r1_51649.out"),
        ("51650", "SEAME dev-man R1 decoder", "mig", "00:10:03", "COMPLETED", True, "logs/basis_a3_basis_a3_man_dec_r1_51650.out"),
        ("51651", "SEAME dev-sge R1 decoder", "mig", "00:08:05", "COMPLETED", True, "logs/basis_a3_basis_a3_sge_dec_r1_51651.out"),
        ("51652", "SEAME dev-sge R1 encoder", "mig", "00:13:23", "COMPLETED", True, "logs/basis_a3_basis_a3_sge_enc_r1_51652.out"),
        ("51653", "CS-Dialogue R2 + Conditioning", "mig", "02:00:58", "COMPLETED", True, "logs/basis_a3_basis_a3_cs_r2_cond_51653.out"),
        ("51654", "SEAME dev-man R2 + Conditioning", "mig", "00:11:46", "COMPLETED", True, "logs/basis_a3_basis_a3_man_r2_cond_51654.out"),
        ("51655", "SEAME dev-sge R2 + Conditioning", "mig", "00:10:15", "COMPLETED", True, "logs/basis_a3_basis_a3_sge_r2_cond_51655.out"),
    ]
    fields = ["job", "purpose", "dataset_stage", "partition", "runtime", "state", "accepted", "log"]
    data = [dict(zip(fields, (job, purpose, "BASIS-A3", part, runtime, state, accepted, log))) for job, purpose, part, runtime, state, accepted, log in jobs]
    json_path = ROOT / "runtime/runtime_summary.json"
    write_json(json_path, {"schema_version": "basis_a3_runtime_summary_v1", "jobs": data, "note": "Preliminary failures/cancellations are quarantined, not scientific results."})
    md_path = ROOT / "runtime/runtime_summary.md"
    lines = ["# BASIS-A3 Runtime Summary", "", "| Job | Dataset/Stage | Partition | Runtime | State | Accepted? | Log |", "|---|---|---|---|---|---:|---|"]
    lines += [f"| {d['job']} | {d['purpose']} | {d['partition']} | {d['runtime']} | {d['state']} | {'yes' if d['accepted'] else 'no'} | `{d['log']}` |" for d in data]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def blocked_acceptance_artifacts() -> list[Path]:
    outdir = ROOT / "acceptance"
    reason = "Scientific snapshot mismatch: SEAME encoder Oracle-local mask semantics changed between execution waves; acceptance stopped before final acceptance."
    common = {"status": "BLOCKED_NOT_RUN", "blocking_reason": reason, "scientific_rerun_performed": False}
    payloads = {
        "encoder_rho0_identity.json": {**common, "required_evidence": "exact token IDs, transcript, and canonical metrics against baseline"},
        "encoder_normpreserve.json": {**common, "required_evidence": "direct before/after edited-frame norm equality"},
        "encoder_padding_exclusion.json": {**common, "required_evidence": "variable-length batch with zero gain and unchanged padding"},
        "encoder_local_mask.json": {**common, "required_evidence": "expected versus actual frame masks under frozen 80 ms tolerance"},
        "encoder_no_grad.json": {**common, "required_evidence": "frozen inference/no-grad path with no optimizer or controller state"},
    }
    paths = []
    for name, payload in payloads.items():
        path = outdir / name
        if not path.exists():
            write_json(path, payload)
        paths.append(path)
    return paths


def main() -> None:
    paths = []
    paths.extend(conditioning_figures())
    paths.extend(package_cross_corpus())
    paths.extend(package_runtime())
    paths.append(package_metric_audit())
    paths.extend(blocked_acceptance_artifacts())
    print("PACKAGED", len(paths), "BASIS-A3 post-run artifacts")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
