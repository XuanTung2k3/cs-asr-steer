"""E1 — alignment construction and audit (guide sections 9-16).

Produces reference-unit -> encoder-frame tables for every subset used later,
plus an audit pack and an automatic boundary-reliability check.

Because CS-Dialogue's `long_wav` TextGrids are not present locally, alignment
comes from Whisper cross-attention DTW. Its reliability is therefore *measured*,
not assumed: every switch boundary is aligned twice under two independent
alignment configurations (Mandarin vs English decoding prefix), and the automatic agreement is reported as a reliability diagnostic; the gate is
stated on at least 100 human verdicts. The audit pack is emitted so the
manual review can be completed without rerunning the expensive alignment.
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data.alignment import (
    ALIGNMENT_SOURCE,
    UNIT_COLUMNS,
    align_batch,
    alignment_to_rows,
    resolve_boundary_overlaps,
    validate_unit_table,
)
from ..data.language_tags import EN, ZH, tag_summary
from ..data.splits import load_subset
from ..models.whisper import load_whisper, write_model_metadata
from ..utils.config import art
from ..utils.status import StageLock, criterion, gate, require_passed
from ._common import base_parser, finish, md_table, prepare, save_report

STAGE = "e1"


def ctc_agreement_check(cfg, units: pd.DataFrame, manifest: pd.DataFrame,
                        dtw_boundaries: pd.DataFrame, log) -> dict:
    """Compare DTW switch points against an independent CTC forced aligner.

    Unlike the second DTW configuration, a CTC model infers boundaries from
    frame-local acoustic posteriors, so its failure modes are largely disjoint
    from cross-attention artefacts and agreement is real evidence.

    Returns a skip record rather than raising when the aligner is unavailable.
    """
    from ..data.ctc_alignment import CTCUnavailable, align_units, load_ctc_aligner
    from ..models.whisper import load_audio

    ccfg = (cfg["alignment"].get("ctc", {}) or {})
    if not ccfg.get("enabled", False):
        return {"status": "disabled",
                "note": "alignment.ctc.enabled is false"}
    try:
        aligner = load_ctc_aligner(cfg)
    except CTCUnavailable as exc:
        log.warning("CTC agreement check skipped: %s", exc)
        return {"status": "unavailable", "reason": str(exc)}

    n_utts = int(ccfg.get("num_utterances", 200))
    seed = int(cfg["experiment"]["seed"])
    utts = dtw_boundaries["utterance_id"].drop_duplicates()
    if len(utts) > n_utts:
        utts = utts.sample(n_utts, random_state=seed)
    paths = manifest.set_index("utterance_id")["audio_path"].to_dict()
    sr = int(cfg["data"]["sample_rate"])

    rows: list[dict] = []
    for utt in utts:
        grp = units[units["utterance_id"] == utt].sort_values("unit_id")
        path = paths.get(utt)
        if path is None or not len(grp):
            continue
        try:
            spans = align_units(aligner, load_audio(str(path), sr),
                                grp["surface"].astype(str).tolist(), sr)
        except Exception as exc:
            log.warning("CTC alignment failed for %s: %s", utt, exc)
            continue
        by_unit = {int(s["unit_id"]): s for s in spans}
        ids = grp["unit_id"].astype(int).tolist()
        tags = grp["language_tag"].tolist()
        for i in range(len(ids) - 1):
            if tags[i] == tags[i + 1] or tags[i] not in (EN, ZH) or tags[i + 1] not in (EN, ZH):
                continue
            left, right = by_unit.get(i), by_unit.get(i + 1)
            if left is None or right is None:
                continue
            rows.append({
                "utterance_id": utt,
                "left_unit_id": ids[i],
                "right_unit_id": ids[i + 1],
                "direction": f"{tags[i]}->{tags[i + 1]}",
                # same midpoint convention as switch_boundaries()
                "ctc_boundary_sec": 0.5 * (left["end_sec"] + right["start_sec"]),
            })
    del aligner

    if not rows:
        return {"status": "no_overlap",
                "note": "CTC produced no comparable switch boundary"}
    ctc = pd.DataFrame(rows)
    key = ["utterance_id", "left_unit_id", "right_unit_id", "direction"]
    merged = dtw_boundaries.merge(ctc, on=key, how="inner")
    if not len(merged):
        return {"status": "no_overlap",
                "note": "no switch point was located by both aligners"}
    diff_ms = (merged["boundary_sec"] - merged["ctc_boundary_sec"]) * 1000.0
    merged["abs_diff_ms"] = diff_ms.abs()
    merged["signed_diff_ms"] = -diff_ms          # positive => CTC is later than DTW
    merged.to_parquet(art(cfg, "metrics", "e1_ctc_agreement.parquet"), index=False)
    by_direction = {}
    for direction, grp in merged.groupby("direction"):
        by_direction[str(direction)] = {
            "num_compared": int(len(grp)),
            "pct_within_50ms": float((grp["abs_diff_ms"] <= 50).mean()),
            "pct_within_100ms": float((grp["abs_diff_ms"] <= 100).mean()),
            "median_abs_diff_ms": float(grp["abs_diff_ms"].median()),
            "mean_signed_diff_ms": float(grp["signed_diff_ms"].mean()),
        }
    return {
        "status": "ok",
        "num_compared": int(len(merged)),
        "pct_within_50ms": float((merged["abs_diff_ms"] <= 50).mean()),
        "pct_within_100ms": float((merged["abs_diff_ms"] <= 100).mean()),
        "median_abs_diff_ms": float(merged["abs_diff_ms"].median()),
        "mean_signed_diff_ms": float(merged["signed_diff_ms"].mean()),
        "systematic_offset_ms": float(merged["signed_diff_ms"].mean()),
        "by_direction": by_direction,
        "note": ("agreement between Whisper cross-attention DTW and an independent "
                 "CTC forced aligner; failure modes are largely disjoint, so this is "
                 "evidence rather than a same-method consistency proxy"),
    }


def synthetic_alignment_check(bundle, cfg, manifest: pd.DataFrame, log, *,
                              resume: bool, overwrite: bool) -> dict:
    """Measure aligner coordinate offsets relative to known audio seams.

    Monolingual ZH and EN utterances are concatenated at an exactly known sample,
    the ordinary aligner is run on the result, and the reported switch is
    compared with that sample. RMS/VAD trimming does not supply a known lexical
    word edge, so this diagnostic cannot support lexical absolute error.
    """
    from ..data.alignment_checks import (
        build_synthetic_pairs,
        render_synthetic_pair,
        summarize_boundary_error,
        synthetic_harness_fingerprint,
    )

    scfg = cfg["alignment"].get("synthetic", {}) or {}
    n_pairs = int(scfg.get("num_pairs", 100))
    if n_pairs <= 0:
        return {"num_boundaries": 0, "note": "synthetic check disabled"}
    seed = int(cfg["experiment"]["seed"])
    max_dur = float(scfg.get("max_duration_sec", 28.0))
    pairs = build_synthetic_pairs(manifest, n_pairs=n_pairs, seed=seed,
                                  gap_ms=float(scfg.get("gap_ms", 0.0)),
                                  sample_rate=int(cfg["data"]["sample_rate"]),
                                  max_duration_sec=max_dur)
    if not pairs:
        return {"num_boundaries": 0,
                "note": "no monolingual ZH/EN utterance pair available"}

    audio_dir = art(cfg, "audit", "e1_synthetic", "_").parent
    trim = bool(scfg.get("trim_silence", True))
    pad_ms = float(scfg.get("trim_pad_ms", 20.0))
    rendered = []
    for p in pairs:
        try:
            rendered.append(render_synthetic_pair(
                p, audio_dir / f"{p['pair_id']}.wav",
                trim_silence=trim, pad_ms=pad_ms))
        except Exception as exc:
            log.warning("synthetic pair %s failed to render: %s", p["pair_id"], exc)
    if not rendered:
        return {"num_boundaries": 0, "note": "no synthetic pair could be rendered"}

    truth = pd.DataFrame(rendered).rename(columns={"pair_id": "utterance_id"})
    truth["speaker_id"] = "synthetic"

    # The rendered waveforms are a function of these settings. A cached alignment
    # made under different ones describes audio that no longer exists on disk, so
    # it must be discarded rather than resumed.
    out_path = art(cfg, "alignments", "synthetic_audio_seam.parquet")
    harness_settings = {"n_pairs": n_pairs, "seed": seed, "trim": trim,
                        "pad_ms": pad_ms, "max_duration_sec": max_dur,
                        "gap_ms": float(scfg.get("gap_ms", 0.0)),
                        "sample_rate": int(cfg["data"]["sample_rate"])}
    fingerprint = synthetic_harness_fingerprint(pairs, harness_settings)
    fp_path = out_path.with_suffix(out_path.suffix + ".harness.json")
    cached_fp = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.exists() else None
    stale = cached_fp != fingerprint if fp_path.exists() else out_path.exists()
    if stale:
        log.info("synthetic harness settings/content changed; realigning from scratch")
    units = align_subset(
        bundle, cfg, truth, log, language=cfg["alignment"].get("language_token", "zh"),
        batch_size=int(cfg["alignment"].get("batch_size", 8)),
        median_width=int(cfg["alignment"].get("median_filter_width", 7)),
        desc="align (synthetic audio seam diagnostic)",
        out_path=out_path,
        resume=resume and not stale, overwrite=overwrite or stale)
    fp_path.write_text(json.dumps(fingerprint, indent=2, default=str), encoding="utf-8")

    b = switch_boundaries(units)
    if not len(b):
        return {"num_boundaries": 0,
                "note": "aligner reported no switch boundary on synthetic items"}
    # Each item is ZH-then-EN by construction, so the first ZH->EN switch is the
    # one we built; later boundaries would come from tagging noise inside a side.
    zh_en = b[b["direction"] == "ZH->EN"].sort_values(["utterance_id", "left_unit_id"])
    first = zh_en.drop_duplicates("utterance_id", keep="first")
    merged = first.merge(truth[["utterance_id", "true_boundary_sec", "duration_sec"]],
                         on="utterance_id", how="inner")
    # A pair longer than the encoder window has no observable boundary; scoring it
    # would measure the window rather than the aligner.
    within_window = merged["duration_sec"] <= max_dur
    n_dropped = int((~within_window).sum())
    merged = merged[within_window]
    if not len(merged):
        return {"num_boundaries": 0,
                "note": "every synthetic pair exceeded the encoder window"}
    stats = summarize_boundary_error(
        merged["boundary_sec"].to_numpy(),
        merged["true_boundary_sec"].to_numpy(), reference_kind="audio_splice")
    stats["num_pairs_rendered"] = int(len(rendered))
    stats["num_pairs_scored"] = int(len(merged))
    stats["num_pairs_over_window"] = n_dropped
    stats["silence_trimmed"] = trim
    stats["harness_sha256"] = fingerprint["sha256"]
    merged.to_parquet(art(cfg, "metrics", "e1_synthetic_audio_seam.parquet"),
                      index=False)
    return stats


def _calibration_note(synthetic: dict, exclude_ms: float,
                      coverage: float = 0.90) -> str:
    """State what the measured error implies for the boundary-exclusion window.

    ``exclude_boundary_ms`` exists to keep frames whose language is uncertain out
    of the centroids and steering masks. It is only doing that job if it covers
    most of the aligner's actual error, so the report says plainly whether it
    does and what it would take.
    """
    if not synthetic.get("num_audio_seams"):
        return ("**Not measured.** " + str(synthetic.get("note", "")))
    tolerances = sorted(int(k[len("pct_within_"):-len("ms_of_seam")])
                        for k in synthetic
                        if k.startswith("pct_within_") and k.endswith("ms_of_seam"))
    covered = [t for t in tolerances
               if float(synthetic[f"pct_within_{t}ms_of_seam"]) >= coverage]
    current = synthetic.get(f"pct_within_{int(exclude_ms)}ms_of_seam")
    lines = [
        f"Measured on {synthetic['num_audio_seams']} constructed audio seams: median "
        f"absolute seam offset **{synthetic['median_absolute_seam_offset_ms']:.0f} ms**, "
        f"p95 **{synthetic['p95_absolute_seam_offset_ms']:.0f} ms**, systematic "
        f"seam offset **{synthetic['mean_signed_seam_offset_ms']:+.0f} ms**. "
        "This is not lexical boundary error.",
    ]
    if current is not None:
        lines.append(f"The configured `exclude_boundary_ms` of {exclude_ms:.0f} ms "
                     f"covers {float(current):.1%} of those seam offsets.")
    if covered:
        lines.append(f"To cover {coverage:.0%} of measured seam offsets, set "
                     f"`alignment.exclude_boundary_ms` to **{covered[0]} ms**"
                     + (" — the current value already suffices."
                        if covered[0] <= exclude_ms else
                        f" (currently {exclude_ms:.0f} ms, so boundary-adjacent "
                        f"frames are under-excluded and E2/E3/E4 will absorb "
                        f"mislabelled frames)."))
    else:
        lines.append(f"No tested tolerance up to {tolerances[-1]} ms covers "
                     f"{coverage:.0%} of the error. Widening the exclusion window "
                     "cannot rescue this alone; the alignment itself needs to "
                     "improve (enable `alignment.ctc` for an independent aligner) "
                     "or the boundary-localised claims must be weakened.")
    lines.append("_Concatenated speech lacks cross-boundary coarticulation, so "
                 "this is an optimistic bound on error in natural code-switches._")
    return "\n\n".join(lines)


def align_subset(bundle, cfg, df: pd.DataFrame, log, *, language: str,
                 batch_size: int, median_width: int, desc: str,
                 out_path: Path, resume: bool, overwrite: bool) -> pd.DataFrame:
    """Align with utterance-level checkpoints and explicit processed IDs."""
    processed_path = out_path.with_suffix(out_path.suffix + ".processed.json")
    units = pd.DataFrame(columns=UNIT_COLUMNS)
    processed: set[str] = set()
    if out_path.exists() and resume and not overwrite:
        units = pd.read_parquet(out_path)
        if processed_path.exists():
            processed = set(json.loads(processed_path.read_text(encoding="utf-8")))
        else:
            processed = set(units["utterance_id"].astype(str).unique())
    todo = df[~df["utterance_id"].isin(processed)].sort_values("duration_sec").reset_index(drop=True)
    rows = units.to_dict(orient="records")
    n = math.ceil(len(todo) / batch_size) if len(todo) else 0
    for bi in tqdm(range(n), desc=desc, leave=False):
        batch = todo.iloc[bi * batch_size: (bi + 1) * batch_size]
        try:
            aligns = align_batch(bundle, batch, language=language,
                                 median_filter_width=median_width,
                                 max_text_tokens=int(cfg["alignment"].get("max_text_tokens", 220)))
        except Exception as exc:
            log.warning("alignment failed for batch %d (%s); retrying singly", bi, exc)
            aligns = []
            for _, r in batch.iterrows():
                try:
                    aligns.extend(align_batch(bundle, batch.loc[[r.name]], language=language,
                                              median_filter_width=median_width,
                                              max_text_tokens=int(cfg["alignment"].get("max_text_tokens", 220))))
                except Exception as exc2:
                    log.error("will retry %s on resume: %s", r["utterance_id"], exc2)
        for a in aligns:
            rows.extend(alignment_to_rows(bundle, a))
            processed.add(a.utterance_id)
        checkpoint = pd.DataFrame(rows, columns=UNIT_COLUMNS)
        checkpoint = checkpoint.drop_duplicates(["utterance_id", "unit_id"], keep="last")
        checkpoint.to_parquet(out_path, index=False)
        processed_path.write_text(json.dumps(sorted(processed)), encoding="utf-8")
    result = pd.DataFrame(rows, columns=UNIT_COLUMNS)
    return result.drop_duplicates(["utterance_id", "unit_id"], keep="last").reset_index(drop=True)


def _use_cjk_font(plt) -> bool:
    """Pick a CJK-capable font if one is installed, else drop Han from titles.

    Audit figures carry Chinese reference words; with DejaVu Sans they render as
    empty boxes and emit a warning per glyph. The transcript is always kept in
    records.jsonl regardless, so no information is lost either way.
    """
    import warnings

    from matplotlib import font_manager

    preferred = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = [name]
            return True
    warnings.filterwarnings("ignore", message=".*missing from font.*")
    return False


def unit_duration_diagnostics(units: pd.DataFrame) -> dict:
    """Surface systematically over-long units (a classic DTW failure mode).

    The first token of an utterance tends to absorb leading silence, which
    contaminates the language frames it contributes to E2. This is reported
    rather than silently corrected.
    """
    u = units.copy()
    u["dur"] = u["end_sec"] - u["start_sec"]
    han = u[u["language_tag"] == ZH]
    en = u[u["language_tag"] == EN]
    first = u[u["unit_id"] == u.groupby("utterance_id")["unit_id"].transform("min")]
    return {
        "zh_unit_dur_median_sec": float(han["dur"].median()) if len(han) else None,
        "zh_unit_dur_p95_sec": float(han["dur"].quantile(0.95)) if len(han) else None,
        "en_unit_dur_median_sec": float(en["dur"].median()) if len(en) else None,
        "en_unit_dur_p95_sec": float(en["dur"].quantile(0.95)) if len(en) else None,
        "pct_zh_units_over_0.5s": float((han["dur"] > 0.5).mean()) if len(han) else None,
        "first_unit_dur_median_sec": float(first["dur"].median()) if len(first) else None,
        "note": ("a large first_unit duration relative to the ZH median indicates "
                 "leading-silence absorption by the initial token"),
    }


def switch_boundaries(units: pd.DataFrame) -> pd.DataFrame:
    """Time of every EN<->ZH switch point, one row per boundary."""
    out = []
    content = units[units["language_tag"].isin([EN, ZH])]
    for utt, grp in content.groupby("utterance_id"):
        g = grp.sort_values("unit_id").reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a["language_tag"] != b["language_tag"]:
                out.append({
                    "utterance_id": utt,
                    "left_unit_id": int(a["unit_id"]),
                    "right_unit_id": int(b["unit_id"]),
                    "left_surface": a["surface"],
                    "right_surface": b["surface"],
                    "direction": f"{a['language_tag']}->{b['language_tag']}",
                    "boundary_sec": float(0.5 * (a["end_sec"] + b["start_sec"])),
                    "left_end_sec": float(a["end_sec"]),
                    "right_start_sec": float(b["start_sec"]),
                    "confidence": float(min(a["alignment_confidence"], b["alignment_confidence"])),
                })
    return pd.DataFrame(out)


def boundary_agreement(primary: pd.DataFrame, alternate: pd.DataFrame) -> pd.DataFrame:
    key = ["utterance_id", "left_unit_id", "right_unit_id"]
    merged = primary.merge(alternate[key + ["boundary_sec"]], on=key,
                           how="inner", suffixes=("", "_alt"))
    merged["abs_diff_ms"] = (merged["boundary_sec"] - merged["boundary_sec_alt"]).abs() * 1000
    merged["signed_diff_ms"] = (merged["boundary_sec_alt"] - merged["boundary_sec"]) * 1000
    return merged


def write_audit_pack(cfg, units: pd.DataFrame, manifest: pd.DataFrame,
                     boundaries: pd.DataFrame, n: int, seed: int, log) -> dict:
    """Spectrogram + clip + context for a stratified sample of switch points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    _use_cjk_font(plt)
    out_dir = art(cfg, "audit", "e1").parent / "e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not len(boundaries):
        return {"num_audited": 0, "audit_dir": str(out_dir)}

    b = boundaries.merge(manifest[["utterance_id", "speaker_id", "audio_path",
                                   "duration_sec", "transcript_raw"]],
                         on="utterance_id", how="left")
    # Stratify switch direction/confidence and spread picks across speakers.
    b["conf_bin"] = pd.qcut(
        b["confidence"], q=max(1, min(3, b["confidence"].nunique())),
        labels=False, duplicates="drop")
    picks = []
    for gi, (_, grp) in enumerate(b.groupby(["direction", "conf_bin"], dropna=False)):
        take = max(1, int(round(n * len(grp) / len(b))))
        shuffled = grp.sample(frac=1.0, random_state=seed + gi)
        diverse = shuffled.drop_duplicates("speaker_id")
        ordered = pd.concat([diverse, shuffled.drop(index=diverse.index)])
        picks.append(ordered.head(min(take, len(ordered))))
    sample = pd.concat(picks).drop_duplicates(
        subset=["utterance_id", "left_unit_id"]).head(n)
    target = min(n, len(b))
    if len(sample) < target:
        remaining = b.drop(index=sample.index, errors="ignore").sample(
            frac=1.0, random_state=seed + 1000)
        sample = pd.concat([sample, remaining.head(target - len(sample))])
    sample = sample.head(target).reset_index(drop=True)

    records = []
    clips_dir = out_dir / "clips"
    figs_dir = out_dir / "figures"
    clips_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)
    for i, r in sample.iterrows():
        try:
            audio, sr = sf.read(r["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            t0 = max(0.0, r["boundary_sec"] - 1.0)
            t1 = min(len(audio) / sr, r["boundary_sec"] + 1.0)
            clip = audio[int(t0 * sr): int(t1 * sr)]
            clip_path = clips_dir / f"{i:03d}_{r['utterance_id']}.wav"
            sf.write(clip_path, clip, sr)

            fig, ax = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
            times = np.arange(len(clip)) / sr + t0
            ax[0].plot(times, clip, lw=0.5)
            ax[0].set_ylabel("waveform")
            ax[1].specgram(clip, Fs=sr, NFFT=512, noverlap=384, cmap="magma")
            ax[1].set_xlabel("time (s, clip-relative)")
            ax[1].set_ylabel("Hz")
            for a in ax:
                a.axvline(r["boundary_sec"] if a is ax[0] else r["boundary_sec"] - t0,
                          color="cyan", ls="--", lw=1.2)
            fig.suptitle(f"{r['utterance_id']} {r['direction']} "
                         f"'{r['left_surface']}' | '{r['right_surface']}'", fontsize=9)
            fig.tight_layout()
            fig_path = figs_dir / f"{i:03d}_{r['utterance_id']}.png"
            fig.savefig(fig_path, dpi=110)
            plt.close(fig)
        except Exception as exc:
            log.warning("audit item %d failed: %s", i, exc)
            clip_path = fig_path = None

        records.append({
            "audit_id": int(i),
            "utterance_id": r["utterance_id"],
            "speaker_id": r["speaker_id"],
            "direction": r["direction"],
            "left_surface": r["left_surface"],
            "right_surface": r["right_surface"],
            "predicted_boundary_sec": float(r["boundary_sec"]),
            "left_end_sec": float(r["left_end_sec"]),
            "right_start_sec": float(r["right_start_sec"]),
            "alignment_source": ALIGNMENT_SOURCE,
            "alignment_confidence": float(r["confidence"]),
            "transcript": r["transcript_raw"],
            "clip": str(clip_path) if clip_path else "",
            "figure": str(fig_path) if fig_path else "",
        })

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    template = pd.DataFrame(records)[["audit_id", "utterance_id", "direction",
                                      "left_surface", "right_surface",
                                      "predicted_boundary_sec"]]
    template["verdict"] = ""            # accepted | rejected
    template["true_boundary_sec"] = ""  # optional human estimate
    template.to_csv(out_dir / "verdicts_template.csv", index=False)
    return {"num_audited": len(records), "audit_dir": str(out_dir),
            "verdicts_csv": str(out_dir / "verdicts_template.csv")}


def read_human_verdicts(audit_dir: str | Path) -> dict | None:
    """Use filled-in human verdicts if present (the gate requires them)."""
    path = Path(audit_dir) / "verdicts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["verdict"].astype(str).str.strip() != ""]
    if not len(df):
        return None
    accepted = (df["verdict"].astype(str).str.lower() == "accepted").mean()
    out = {"num_verdicts": int(len(df)), "accepted_fraction": float(accepted)}
    if "true_boundary_sec" in df.columns:
        t = pd.to_numeric(df["true_boundary_sec"], errors="coerce")
        p = pd.to_numeric(df["predicted_boundary_sec"], errors="coerce")
        err = (t - p).abs() * 1000
        err = err[err.notna()]
        if len(err):
            out.update(
                mean_abs_error_ms=float(err.mean()),
                pct_within_50ms=float((err <= 50).mean()),
                pct_within_100ms=float((err <= 100).mean()),
                signed_bias_ms=float(((t - p) * 1000).dropna().mean()),
            )
    return out


def finite_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def human_audit_gate_state(human: dict | None, *, required_items: int,
                           min_usable_fraction: float,
                           allow_unreviewed_alignment: bool = False) -> dict:
    verdicts = int(human.get("num_verdicts", 0)) if human else 0
    complete = verdicts >= int(required_items)
    usable = float(human.get("accepted_fraction", float("nan"))) if complete and human else float("nan")
    passes = bool(complete and usable >= float(min_usable_fraction))
    return {
        "num_verdicts": verdicts,
        "complete": complete,
        "usable_fraction": usable,
        "passes": passes,
        "diagnostic_override_prohibits_pass": bool(allow_unreviewed_alignment and not passes),
    }


def independent_audit_gate_state(stats: dict, *, required_items: int,
                                 min_within_100ms: float,
                                 max_median_abs_ms: float) -> dict:
    ok = stats.get("status") == "ok"
    compared = int(stats.get("num_compared", 0)) if ok else 0
    within = finite_float(stats.get("pct_within_100ms"))
    median = finite_float(stats.get("median_abs_diff_ms"))
    return {
        "status_ok": ok,
        "num_compared": compared,
        "within_100ms": within,
        "median_abs_ms": median,
        "passes": bool(ok and compared >= int(required_items)
                       and within >= float(min_within_100ms)
                       and median <= float(max_median_abs_ms)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("E1 alignment construction and audit",
                         "experiments/e1_alignment.yaml")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="override the subsets to align")
    parser.add_argument("--allow-unreviewed-alignment", action="store_true",
                        help=("diagnostic only: write reports without completed human "
                              "verdicts, but never create a production pass marker"))
    args = parser.parse_args(argv)
    cfg, rdir, log = prepare(args, STAGE)
    root = cfg["experiment"]["output_root"]
    acfg = cfg["alignment"]
    subsets = args.subsets or acfg.get("subsets", ["dev_select", "dev_confirm",
                                                   "train_direction_pilot"])

    with StageLock(root, STAGE):
        if not args.force_prereq:
            require_passed(root, ["p0"])
        bundle = load_whisper(cfg)
        write_model_metadata(bundle, rdir / "model_metadata.json")
        exclude_frames = int(round((acfg["exclude_boundary_ms"] / 1000) / bundle.encoder_step_sec))
        log.info("encoder step %.4f s -> %d frames excluded per language boundary",
                 bundle.encoder_step_sec, exclude_frames)

        if args.dry_run:
            for s in subsets:
                df = load_subset(cfg, s)
                log.info("%s: %d utterances, audio present=%s", s, len(df),
                         all(Path(p).exists() for p in df["audio_path"].head(20)))
            return 0

        metrics: dict = {"encoder_step_sec": bundle.encoder_step_sec,
                         "exclude_boundary_frames": exclude_frames,
                         "alignment_source": ALIGNMENT_SOURCE,
                         "subsets": {}}
        tables: dict[str, pd.DataFrame] = {}
        for subset in subsets:
            out_path = art(cfg, "alignments", f"{subset}.parquet")
            df = load_subset(cfg, subset)
            if args.limit:
                df = df.head(args.limit)
            target_hash = hashlib.sha256("\n".join(sorted(df["utterance_id"].astype(str))).encode()).hexdigest()
            meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
            expected_meta = {
                "model_revision": bundle.revision, "alignment_source": ALIGNMENT_SOURCE,
                "language": acfg.get("language_token", "zh"),
                "median_filter_width": int(acfg.get("median_filter_width", 7)),
                "max_text_tokens": int(acfg.get("max_text_tokens", 220)),
                "target_ids_sha256": target_hash, "num_targets": int(len(df)),
            }
            if out_path.exists() and args.resume and not args.overwrite:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
                if cached_meta != expected_meta:
                    raise RuntimeError(f"alignment cache provenance mismatch: {out_path}; use --overwrite")
            meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
            units = align_subset(
                bundle, cfg, df, log, language=acfg.get("language_token", "zh"),
                batch_size=int(acfg.get("batch_size", 8)),
                median_width=int(acfg.get("median_filter_width", 7)),
                desc=f"align {subset}", out_path=out_path,
                resume=args.resume, overwrite=args.overwrite)
            # floor/ceil rounding makes adjacent spans share the boundary frame;
            # make them disjoint before anything consumes the table
            units, overlap_fix = resolve_boundary_overlaps(units, bundle.encoder_step_sec)
            if overlap_fix["overlaps_resolved"] or overlap_fix["overlaps_unresolved"]:
                log.info("%s boundary overlaps: %s", subset,
                         json.dumps(overlap_fix, default=str))
            units.to_parquet(out_path, index=False)
            tables[subset] = units

            val = validate_unit_table(units, df, bundle.max_encoder_frames,
                                      bundle.encoder_step_sec, strict=False)
            val["boundary_overlap_fix"] = overlap_fix
            tags = tag_summary(list(zip(units["surface"], units["language_tag"])))
            eligible = units[units["language_tag"].isin([EN, ZH])].groupby("utterance_id")[
                "language_tag"].nunique()
            metrics["subsets"][subset] = {
                "num_utterances": int(units["utterance_id"].nunique()),
                "validation": val,
                "tags": {k: v for k, v in tags.items() if k != "top_ambiguous"},
                "top_ambiguous": tags["top_ambiguous"][:25],
                "num_bilingual_utterances": int((eligible == 2).sum()),
                "mean_confidence": float(units["alignment_confidence"].mean()),
                "boundary_adjacent_units": int(units["is_boundary_adjacent"].sum()),
                "durations": unit_duration_diagnostics(units),
            }
            log.info("%s: %d units, %d bilingual utterances", subset, len(units),
                     int((eligible == 2).sum()))

        # Derive the pilot alignment from the already-aligned full training table.
        if "train_direction_full" in tables:
            pilot_manifest = load_subset(cfg, "train_direction_pilot")
            pilot_units = tables["train_direction_full"][
                tables["train_direction_full"]["utterance_id"].isin(pilot_manifest["utterance_id"])
            ].copy()
            pilot_path = art(cfg, "alignments", "train_direction_pilot.parquet")
            pilot_units.to_parquet(pilot_path, index=False)
            tables["train_direction_pilot"] = pilot_units
            val = validate_unit_table(pilot_units, pilot_manifest, bundle.max_encoder_frames,
                                      bundle.encoder_step_sec, strict=False)
            tags = tag_summary(list(zip(pilot_units["surface"], pilot_units["language_tag"])))
            eligible = pilot_units[pilot_units["language_tag"].isin([EN, ZH])].groupby(
                "utterance_id")["language_tag"].nunique()
            metrics["subsets"]["train_direction_pilot"] = {
                "num_utterances": int(pilot_units["utterance_id"].nunique()),
                "validation": val,
                "tags": {k: v for k, v in tags.items() if k != "top_ambiguous"},
                "top_ambiguous": tags["top_ambiguous"][:25],
                "num_bilingual_utterances": int((eligible == 2).sum()),
                "mean_confidence": float(pilot_units["alignment_confidence"].mean()),
                "boundary_adjacent_units": int(pilot_units["is_boundary_adjacent"].sum()),
                "durations": unit_duration_diagnostics(pilot_units),
            }

        # ---- boundary reliability -------------------------------------
        audit_subset = "dev_select" if "dev_select" in tables else subsets[0]
        primary_b = switch_boundaries(tables[audit_subset])
        gcfg = acfg.get("gate", {}) or {}
        require_human = bool(gcfg.get("require_human_audit", True))
        required_human_items = int(gcfg.get("human_audit_items",
                                            acfg.get("audit_boundaries", 100)))
        n_audit = max(int(acfg.get("audit_boundaries", 100)), required_human_items)
        agree_stats: dict = {}
        if len(primary_b):
            sample_utts = (primary_b["utterance_id"].drop_duplicates()
                           .sample(min(200, primary_b["utterance_id"].nunique()),
                                   random_state=int(cfg["experiment"]["seed"])))
            alt_df = load_subset(cfg, audit_subset)
            alt_df = alt_df[alt_df["utterance_id"].isin(sample_utts)]
            alt_path = art(cfg, "alignments", f"{audit_subset}_alternate.parquet")
            alt_units = align_subset(
                bundle, cfg, alt_df, log, language="en",
                batch_size=int(acfg.get("batch_size", 8)), median_width=3,
                desc="align (alternate config)", out_path=alt_path,
                resume=args.resume, overwrite=args.overwrite)
            alt_b = switch_boundaries(alt_units)
            agree = boundary_agreement(primary_b, alt_b)
            if len(agree):
                agree_stats = {
                    "num_compared": int(len(agree)),
                    "pct_within_50ms": float((agree["abs_diff_ms"] <= 50).mean()),
                    "pct_within_100ms": float((agree["abs_diff_ms"] <= 100).mean()),
                    "median_abs_diff_ms": float(agree["abs_diff_ms"].median()),
                    "mean_signed_diff_ms": float(agree["signed_diff_ms"].mean()),
                    "note": ("agreement between two independent alignment configurations "
                             "(zh prefix / median 7 vs en prefix / median 3); this is a "
                             "reliability proxy, not ground truth"),
                }
                agree.to_parquet(art(cfg, "metrics", "e1_boundary_agreement.parquet"),
                                 index=False)
            audit = write_audit_pack(cfg, tables[audit_subset], load_subset(cfg, audit_subset),
                                     primary_b, n_audit, int(cfg["experiment"]["seed"]), log)
        else:
            audit = {"num_audited": 0}
        metrics["boundary_agreement"] = agree_stats
        tagged = tables[audit_subset].sample(
            min(100, len(tables[audit_subset])), random_state=int(cfg["experiment"]["seed"]))
        tagged_path = art(cfg, "audit", "e1", "tagged_examples.jsonl")
        tagged.to_json(tagged_path, orient="records", lines=True, force_ascii=False)
        metrics["tagged_examples"] = {"count": int(len(tagged)), "path": str(tagged_path)}
        metrics["audit_pack"] = audit
        metrics["num_switch_boundaries"] = int(len(primary_b))

        # ---- automatic alignment evidence (no-human protocol) ----------
        # The guide's manual audit is replaced by three automatic measurements.
        # Human verdicts are still honoured when present, but never required.
        from ..data.alignment_checks import silence_absorption

        silence = silence_absorption(
            tables[audit_subset], load_subset(cfg, audit_subset),
            sample_utterances=int(gcfg.get("silence_sample_utterances", 200)),
            seed=int(cfg["experiment"]["seed"]),
            silence_ms=float(gcfg.get("silence_ms", 100.0)))
        metrics["silence_absorption"] = silence
        log.info("silence absorption: %s", json.dumps(silence, default=str))

        synthetic = synthetic_alignment_check(
            bundle, cfg, load_subset(cfg, audit_subset), log,
            resume=args.resume, overwrite=args.overwrite)
        metrics["synthetic_audio_seam"] = synthetic
        log.info("synthetic audio-seam offsets: %s", json.dumps(synthetic, default=str))
        # Advisory but decisive: this is what the exclusion window gets set from,
        # so it goes in the job log rather than only the report.
        for line in _calibration_note(
                synthetic, float(acfg["exclude_boundary_ms"])).splitlines():
            if line.strip():
                log.info("alignment seam diagnostic | %s", line.replace("**", ""))

        ctc_stats = ctc_agreement_check(cfg, tables[audit_subset],
                                        load_subset(cfg, audit_subset), primary_b, log)
        metrics["ctc_agreement"] = ctc_stats
        log.info("CTC agreement: %s", json.dumps(ctc_stats, default=str))

        human = read_human_verdicts(audit.get("audit_dir", "")) if audit.get("audit_dir") else None
        metrics["human_audit"] = human
        human_state = human_audit_gate_state(
            human, required_items=required_human_items,
            min_usable_fraction=float(gcfg.get("min_human_usable_fraction", 0.90)),
            allow_unreviewed_alignment=args.allow_unreviewed_alignment)
        human_verdicts = human_state["num_verdicts"]
        human_complete = human_state["complete"]
        usable = human_state["usable_fraction"]
        usable_source = "human verdicts" if human_complete else "not provided"
        pilot_bilingual = metrics["subsets"].get(
            "train_direction_pilot", {}).get("num_bilingual_utterances", 0)
        units_valid = all(
            s["validation"]["frames_out_of_range"] == 0
            and s["validation"]["empty_spans"] == 0
            and s["validation"]["padding_frames_labelled"] == 0
            and s["validation"]["non_monotonic_units"] == 0
            and s["validation"]["conflicting_language_overlaps"] == 0
            for s in metrics["subsets"].values())
        syn_within_100 = finite_float(synthetic.get("pct_within_100ms_of_seam"))
        syn_bias = abs(finite_float(synthetic.get("mean_signed_seam_offset_ms")))
        agree_within_100 = finite_float(agree_stats.get("pct_within_100ms"))
        absorption = finite_float(silence.get("absorption_rate"))
        min_syn_100 = float(gcfg.get("min_synthetic_within_100ms", 0.90))
        max_syn_bias = float(gcfg.get("max_abs_synthetic_bias_ms",
                                      gcfg.get("max_abs_bias_ms", 50.0)))
        min_syn_boundaries = int(gcfg.get("min_synthetic_boundaries",
                                          gcfg.get("human_audit_items", 100)))
        min_human_usable = float(gcfg.get("min_human_usable_fraction", 0.90))
        require_units = bool(gcfg.get("require_units_validate", True))
        require_independent = bool(gcfg.get("require_independent_alignment_audit", True))
        min_independent = int(gcfg.get("independent_audit_items",
                                       gcfg.get("min_independent_alignment_boundaries", 200)))
        min_independent_100 = float(gcfg.get("min_independent_within_100ms", 0.90))
        max_independent_median = float(gcfg.get("max_independent_median_abs_ms", 100.0))
        criteria = [
            criterion("units_validate", int(units_valid), 1,
                      units_valid if require_units else True, "=="),
            criterion("num_switch_boundaries", int(len(primary_b)), n_audit,
                      len(primary_b) >= n_audit),
            criterion("eligible_bilingual_training_utterances", pilot_bilingual, 500,
                      pilot_bilingual >= 500),
            criterion("config_agreement_within_100ms", agree_within_100,
                      float(gcfg.get("min_agreement_within_100ms", 0.90)),
                      agree_within_100 >= float(gcfg.get("min_agreement_within_100ms", 0.90))),
            criterion("silence_absorption_rate", absorption,
                      float(gcfg.get("max_absorption_rate", 0.20)),
                      absorption <= float(gcfg.get("max_absorption_rate", 0.20)), "<="),
            criterion("synthetic_audio_seams_scored",
                      int(synthetic.get("num_audio_seams", 0)), min_syn_boundaries,
                      int(synthetic.get("num_audio_seams", 0)) >= min_syn_boundaries),
            criterion("synthetic_audio_seams_within_100ms",
                      syn_within_100, min_syn_100, syn_within_100 >= min_syn_100),
            criterion("synthetic_abs_systematic_seam_offset_ms", syn_bias, max_syn_bias,
                      syn_bias <= max_syn_bias, "<="),
        ]
        ctc_state = independent_audit_gate_state(
            ctc_stats, required_items=min_independent,
            min_within_100ms=min_independent_100,
            max_median_abs_ms=max_independent_median)
        ctc_ok = ctc_state["status_ok"]
        ctc_compared = ctc_state["num_compared"]
        ctc_within_100 = ctc_state["within_100ms"]
        ctc_median = ctc_state["median_abs_ms"]
        if require_independent:
            criteria += [
                criterion("independent_alignment_audit_status", ctc_stats.get("status"),
                          "ok", ctc_ok, "=="),
                criterion("independent_alignment_audit_boundaries", ctc_compared,
                          min_independent, ctc_compared >= min_independent),
                criterion("independent_alignment_within_100ms", ctc_within_100,
                          min_independent_100, ctc_within_100 >= min_independent_100),
                criterion("independent_alignment_median_abs_ms", ctc_median,
                          max_independent_median,
                          ctc_median <= max_independent_median, "<="),
            ]
        if require_human:
            criteria += [
                criterion("audit_pack_items", int(audit.get("num_audited", 0)),
                          required_human_items,
                          int(audit.get("num_audited", 0)) >= required_human_items),
                criterion("human_audit_verdicts", human_verdicts,
                          required_human_items, human_complete),
                criterion(f"human_usable_fraction ({usable_source})", usable,
                          min_human_usable, usable >= min_human_usable),
            ]
        if args.allow_unreviewed_alignment and require_human and not human_complete:
            criteria.append(criterion("diagnostic_unreviewed_alignment_override",
                                      "enabled", "production-pass-prohibited",
                                      False, "=="))
        nonhuman_pass = all(c["passed"] for c in criteria
                            if not c["name"].startswith("human_")
                            and c["name"] != "diagnostic_unreviewed_alignment_override")
        awaiting_human = bool(require_human and not human_complete and nonhuman_pass
                              and not args.allow_unreviewed_alignment)
        g = gate("e1", all(c["passed"] for c in criteria), criteria,
                 "Production E1 is blocking: unit-table validation, synthetic "
                 "ground-truth boundary accuracy/bias, independent CTC audit and "
                 "the 100-item human audit must pass. Cross-configuration DTW "
                 "agreement is retained only as a same-method reliability proxy.")

        (art(cfg, "metrics", "e1_alignment_metrics.json")).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        summary_rows = pd.DataFrame([
            {"subset": k, "utterances": v["num_utterances"],
             "bilingual": v["num_bilingual_utterances"],
             "units": v["validation"]["num_units"],
             "mean_confidence": v["mean_confidence"]}
            for k, v in metrics["subsets"].items()
        ])
        save_report(
            art(cfg, "reports", "e1_alignment_audit.md"), "E1 — alignment construction and audit",
            [
                ("Alignment source",
                 f"`{ALIGNMENT_SOURCE}` — CS-Dialogue long_wav TextGrids are not available "
                 f"locally, so dataset-provided boundaries could not be used. Encoder step "
                 f"{bundle.encoder_step_sec:.4f} s ({exclude_frames} frames excluded per "
                 f"language boundary)."),
                ("Subsets", md_table(summary_rows)),
                ("Unit duration diagnostics", "```json\n" + json.dumps(
                    {k: v["durations"] for k, v in metrics["subsets"].items()},
                    indent=2, default=str) + "\n```"),
                ("Boundary reliability", "```json\n" + json.dumps(agree_stats, indent=2) + "\n```"),
                ("Synthetic audio-seam coordinate diagnostic",
                 "```json\n" + json.dumps(synthetic, indent=2, default=str) + "\n```\n\n"
                 + _calibration_note(synthetic, float(acfg["exclude_boundary_ms"]))),
                ("Silence absorption", "```json\n" + json.dumps(silence, indent=2) + "\n```"),
                ("Audit pack", "```json\n" + json.dumps(audit, indent=2) + "\n```\n\n"
                 "Production E1 is blocked until `verdicts_template.csv` is filled, "
                 "saved as `verdicts.csv` in the same directory, and the human-audit "
                 "gate passes. `--allow-unreviewed-alignment` is diagnostic only and "
                 "cannot create a production pass."),
                ("Gate", "```json\n" + json.dumps(g, indent=2, default=str) + "\n```"),
            ],
        )
        finish(cfg, STAGE, rdir, metrics, g,
               artifacts=[str(art(cfg, "alignments", f"{s}.parquet")) for s in subsets],
               status_override="blocked" if awaiting_human else None)
        log.info("E1 gate: %s", "PASSED" if g["passed"] else ("BLOCKED FOR AUDIT" if awaiting_human else "FAILED"))
        return 0 if g["passed"] else (3 if awaiting_human else 2)


if __name__ == "__main__":
    raise SystemExit(main())
