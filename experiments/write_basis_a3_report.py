#!/usr/bin/env python
"""Write the CPU-only BASIS-A3 qualitative index and final Markdown report."""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/basis_a3_raw_cond_scope_depth"


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return None if x in (None, "", "nan", "None") else float(x)
    except (TypeError, ValueError):
        return None


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _files(dataset, direction):
    roots = [OUT / "raw_r1" / dataset, OUT / "raw_r2" / dataset]
    if direction == "conditioning":
        roots = [OUT / "conditioning" / dataset]
    out = []
    for root in roots:
        out.extend(root.glob("L*/*.json"))
    return [p for p in sorted(out) if json.loads(p.read_text()).get("direction", "").lower() == direction.lower()]


def qualitative():
    cases = []
    for dataset in ("cs_dialogue", "seame_dev_man", "seame_dev_sge"):
        panel = json.loads((OUT / "panels" / f"{dataset}_300.json").read_text())
        refs = {str(x["utterance_id"]): x.get("reference", "") for x in panel["rows"]}
        for direction in ("Raw", "conditioning"):
            paths = _files(dataset, direction)
            pairs = {}
            for p in paths:
                x = json.loads(p.read_text()); m = x["metrics"]
                key = (x.get("side", m.get("side")), int(x["layer"]), float(x["rho"]))
                pairs.setdefault(key, {})[x["scope"]] = (p, x)
            ranked = []
            for key, pair in pairs.items():
                if "global" not in pair or "oracle_local" not in pair:
                    continue
                g, l = pair["global"][1], pair["oracle_local"][1]
                score = (num(g["metrics"].get("pier")) or 0) - (num(l["metrics"].get("pier")) or 0)
                score += (num(l["metrics"].get("retention", {}).get("matrix_zh", {}).get("rate")) or 0)
                score -= (num(g["metrics"].get("retention", {}).get("matrix_zh", {}).get("rate")) or 0)
                ranked.append((score, key, pair))
            if not ranked:
                continue
            score, key, pair = sorted(ranked, reverse=True, key=lambda z: z[0])[0]
            gpath, gx = pair["global"]; lpath, lx = pair["oracle_local"]
            base = gx.get("baseline_texts", {})
            changed = []
            for uid in refs:
                b, gt, lt = base.get(uid, ""), gx["texts"].get(uid, ""), lx["texts"].get(uid, "")
                if gt == b and lt == b:
                    continue
                diff = max(1.0 - difflib.SequenceMatcher(None, b, gt).ratio(),
                           1.0 - difflib.SequenceMatcher(None, b, lt).ratio())
                changed.append((diff, uid, b, gt, lt))
            changed.sort(reverse=True)
            cases.append({"dataset": dataset, "direction": direction, "side": key[0],
                          "layer": key[1], "rho": key[2], "selection_score": score,
                          "global_path": str(gpath.relative_to(OUT)),
                          "local_path": str(lpath.relative_to(OUT)),
                          "examples": [{"utterance_id": u, "reference": refs[u],
                                        "baseline": b, "global": gt, "oracle_local": lt}
                                       for _, u, b, gt, lt in changed[:5]]})
    (OUT / "qualitative/cases.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "qualitative/cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    lines = ["# BASIS-A3 qualitative cases", "", "Selection is explanatory only; it did not change any configuration.", ""]
    for c in cases:
        lines += [f"## {c['dataset']} / {c['direction']} / {c['side']} L{c['layer']} rho={c['rho']}", ""]
        for e in c["examples"]:
            lines += [f"- `{e['utterance_id']}`", f"  - REF: {e['reference']}", f"  - BASE: {e['baseline']}",
                      f"  - GLOBAL: {e['global']}", f"  - LOCAL: {e['oracle_local']}"]
        lines.append("")
    (OUT / "qualitative/cases.md").write_text("\n".join(lines), encoding="utf-8")


def report():
    raw = read_csv("tables/raw_full_depth.csv")
    cond = read_csv("tables/conditioning.csv")
    geom = json.loads((OUT / "geometry/raw_cond_decoder_geometry.json").read_text())["layers"]
    summary = json.loads((OUT / "summary.json").read_text())
    jobs = {"R1 CS encoder": "51638", "R1 CS decoder": "51639", "R1 dev-man encoder": "51649",
            "R1 dev-man decoder": "51650", "R1 dev-sge decoder": "51651", "R1 dev-sge encoder": "51652",
            "R2+Cond dev-man": "51654", "R2+Cond dev-sge": "51655", "R2+Cond CS": "51653"}
    def best(ds, side, scope):
        z = [x for x in raw if x["Dataset"] == ds and x["Side"] == side and x["Scope"] == scope]
        if not z: return "n/a"
        z.sort(key=lambda x: (num(x["Corr"]) or 0) - (num(x["Corrupt"]) or 0), reverse=True)
        x = z[0]
        return f"L{x['Layer']} (net POI {int(float(x['Corr'])-float(x['Corrupt']))})"
    def cond_best(ds, layer):
        z = [x for x in cond if x["Dataset"] == ds and int(x["Layer"]) == layer]
        if not z: return "n/a"
        x = min(z, key=lambda y: num(y["MER"]) if num(y["MER"]) is not None else 1e9)
        return f"rho {x['rho']} {x['Scope']} (MER {float(x['MER']):.3f})"
    gcos = [x["cos_raw_cond"] for x in geom]; gang = [x["angle_deg"] for x in geom]; gl2 = [x["unit_l2"] for x in geom]
    lines = ["# BASIS-A3 — Raw/Conditioning Scope × Depth × Cross-Corpus", "",
             "Frozen/no-training explanatory study. Oracle-local results are diagnostic upper bounds, not deployable inference.", "",
             "## Implementation", "", "Exact encoder/decoder hooks, NormPreserve, free decoding, cached decoder encoder outputs, resumable condition blocks, provenance manifests, CPU audit and geometry analysis were implemented in `experiments/basis_a3.py` and `src/csasr/lss/encoder_sites.py`.", "",
             "## Frozen Protocol", "", "Raw only on all 32 encoder and 32 decoder layers at rho=.5 in R1; selected R2 layers use rho=.25 and 1.0. Conditioning is decoder-only at L24/L26/L27/L31 and rho in {.25,.5,1.0}. Global edits every eligible position/frame; Oracle-local edits only reference-aligned embedded-English positions/spans.", "",
             "## Encoder Site Validation", "", "Encoder site: post-self-attention residual `q_enc + u_enc`, immediately before encoder FFN. CPU focused tests passed; MIG preflight job 51626 passed one-vs-two-worker transcript equivalence for encoder L24 and decoder L24.", "",
             "## Evaluation Panels", "", "CS-Dialogue: 300, `sha256:7fb612ee1be7806c624dd0836fbf5e43429d41e3af032e0d4cb7fe4e8a3398ff`; dev-man: 50, `sha256:3f49d08d66e88d4bf9f79ef8a80f6013fb1d1d16e07269fb50bc94e4f26e2a3c`; dev-sge: 50, `sha256:ebdcdd10b6f0d4a5f7666097f96168bf39869a65e39de779dfdc7c8ebfb77238`. The SEAME canonical fixed panel had only 50 eligible examples per split; no outcome-based resampling was done.", "",
             "## Direction Provenance", "", f"Decoder Raw/Conditioning: `{sha(ROOT / 'results/basis_frozen_layer_atlas/directions.json')}` source artifact. Encoder Raw: `{sha(OUT / 'directions/raw_encoder/manifest.json')}` manifest; each layer has its own D-construct artifact, unit normalization and SHA256.", "",
             "## Metric Audit", "", "`poi_corrections` and `poi_corruptions` are correctness-flip POI transitions; `poi_net_utility = corrections - corruptions`. CS candidate-level utility and outside harm were reconstructed separately with the frozen DG-03 existing_ctc candidate population. SEAME outside harm is N/A because no accepted candidate/POI population was available.", "",
             "## GPU Execution", "", "All scientific GPU work used MIG, one H100 3g.40gb allocation per job, two independent batch-1 workers, and sbatch. Valid jobs: 51638 (CS enc R1, 2:34:51), 51639 (CS dec R1, 2:05:36), 51649 (man enc R1, 15:22), 51650 (man dec R1, 10:03), 51651 (sge dec R1, 10:03), 51652 (sge enc R1, 13:23), 51654 (man R2+Cond, 11:46), 51655 (sge R2+Cond, 10:15), 51653 (CS R2+Cond, 2:00:58). Invalid preliminary attempts were quarantined and excluded.", "",
             "## Raw Full-Depth Map", "", f"{summary['raw_r1_rows']} aggregate rows across three datasets. CS decoder Raw becomes strongest in late depth (Global best net: {best('cs_dialogue','decoder','global')}); encoder Raw is weak/negative on CS. Dev-man shows a small decoder response; dev-sge decoder effects are smaller. Required tables and figures are under `tables/` and `figures/`.", "",
             "## Raw Encoder — Global vs Local", "", "CS encoder Local does not rescue the R1 map overall; cross-corpus encoder effects are weak and inconsistent. Local edits are much sparser, and frame-normalized efficiency is reported in the raw tables.", "",
             "## Raw Decoder — Global vs Local", "", "Late decoder Raw is powerful but collateral-sensitive on CS: L27 R1 Global has much worse matrix retention and larger outside harm than Local, while Local preserves matrix output substantially better. Dev-man shows the same direction of Global-to-Local retention improvement; dev-sge is weaker.", "",
             "## Raw Dose Characterization", "", "R2 selected encoder layers were L2/L16/L21; decoder anchors were L0/L24/L27 plus L2 negative control. rho=.5 was reused from R1; only .25 and 1.0 were decoded.", "",
             "## Conditioning — Global vs Local", "", "CS Conditioning L26/L27 remains modest at rho=.25/.5 but becomes damaging at rho=1. L31 is destructive globally at rho=.5/1 and substantially less damaging when local; dev-man and dev-sge reproduce the destructive high-dose L31 pattern, while L26/L27 transfer is weak.", "",
             "## Encoder vs Decoder", "", "The clearest Raw correction power is late decoder depth on CS-Dialogue. Encoder Raw does not show a stable useful region across the three panels; decoder effects are stronger but also more damaging when global.", "",
             "## Localization Effect", "", "Oracle localization generally reduces matrix damage and outside harm, and rescues some late decoder conditions. It is a causal diagnostic only because it uses reference-aligned CS spans.", "",
             "## Cross-Corpus Consistency", "", "| Observation | CS-Dialogue | SEAME dev-man | SEAME dev-sge |", "|---|---|---|---|",
             f"| Raw encoder useful region | weak/inconsistent | weak; small L27 Local response | no stable region |",
             f"| Raw decoder useful region | late, especially L27–L30 Global/Local tradeoff | small late response | small L24/L27 response |",
             f"| Raw L27 correction signal | clear, Local safer | weak positive | weak/absent Local |",
             f"| Raw Global→Local rescue | clear on late decoder | retention rescue | limited rescue |",
             f"| Cond L26 useful | modest, narrow-dose | weak | weak |",
             f"| Cond L27 useful | modest Local at rho=1 | weak Local at rho=1 | absent Local |",
             f"| Cond Global→Local rescue | strong at L31 | strong at L31 | strong at L31 |",
             f"| Cond L31 destructive | yes at rho≥.5 | yes at rho≥.5 | yes at rho≥.5 |", "",
             "## Qualitative Cases", "", "See `qualitative/cases.md` and `qualitative/cases.json`; cases were selected after freezing for explanation only.", "",
             "## Figures", "", "Raw depth: `figures/pier_vs_depth.png`, `mer_vs_depth.png`, `matrix_cer_vs_depth.png`, `matrix_ret_vs_depth.png`, `corr_vs_depth.png`, `outside_vs_depth.png`, `local_minus_global_pier.png`, `local_minus_global_matrix_ret.png`, `raw_correction_damage_frontier.png`, `raw_intervention_efficiency.png`. Geometry: `figures/geometry_cos_raw_cond.png`, `geometry_angle_deg.png`, `geometry_unit_l2.png`, `geometry_raw_norm.png`, `geometry_cond_norm.png`.", "",
             "## Raw–Conditioning Decoder Geometry", "", f"Cosine range {min(gcos):.4f}..{max(gcos):.4f}; angle range {min(gang):.3f}..{max(gang):.3f} degrees; unit-L2 range {min(gl2):.4f}..{max(gl2):.4f}. L24 is among the most separated layers; L31 is the most aligned, while L26/L27 are near-orthogonal and not uniquely unusual. The unit-L2 identity was validated. Four-layer-per-dataset descriptive correlations are stored in `geometry/steering_geometry_correlations.csv`; they are not causal evidence.", "",
             "## Scientific Takeaway", "", "Raw contains real late-decoder causal correction signal on CS-Dialogue, but global application is the dominant collateral-damage problem. Encoder Raw is not yet a stable cross-corpus result. Conditioning has a narrow late-decoder response and a reproducible destructive high-dose L31 control; its positive L26/L27 signal transfers weakly.", "",
             "## Risks / Negative Findings", "", "SEAME panels are 50 rather than 300; SEAME outside harm is unavailable; oracle-localization is not deployable; geometry/effect correlations have n=4 decoder layers per dataset/scope and must not be overinterpreted.", "",
             "## Next Recommendation", "", "Do not run another study in this ticket. If work resumes, first obtain a pre-registered accepted SEAME candidate/POI accounting population before making outside-harm claims.", "",
             "## Data Exposure", "", "BASIS-A3 did not read D-dev-confirm or D-test. CS used only D-construct for directions and D-dev-select for evaluation; SEAME dev-man/dev-sge were exploratory development panels.", "",
             "## Provenance", "", f"R1/R2/Conditioning tables: `tables/`; geometry: `geometry/`; spec freeze self-hash `8def7ba84050103343e8b94332d287258dd3497beece142738cc26e92f2fb5d3`; R2 selection: `selection/r2_layers.json`; summary: `summary.json`.", "",
             "## Blockers", "", "None for the frozen exploratory study; limitations are recorded above.", ""]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    qualitative(); report()
