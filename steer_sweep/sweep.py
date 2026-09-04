"""Track A: the staged sweep and its cell definitions.

The full grid is ~160 cells. On twenty dialogue clusters, choosing the best of
160 is a winner's-curse machine and `D-dev-confirm` has ten dialogues to spend,
so the sweep is staged and the between-stage rule is hard-coded in
`metrics.select`. Nothing here reads a result before deciding what to run next
except through that rule.

  Stage A  site x coverage screen, v_nat, mid layer, alpha=1, matched energy
  Stage B  v_prompt at both coverages, both combination modes, + top-2 from A
  Stage C  alpha sweep on the winner of B
  Stage D  layer sweep on the winner of C: singles first, then multi
  Confirm  exactly ONE config, printed and STOPPED for human approval
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from . import config as C
from .hooks import SteeringPlan


@dataclass(frozen=True)
class Cell:
    """One steering configuration. Its hash is the resume key."""

    stage: str
    site: str
    coverage: str
    direction_kind: str
    alpha: float
    encoder_layers: tuple[int, ...] = ()
    decoder_layers: tuple[int, ...] = ()
    scale_mode: str = C.DEFAULT_SCALE_MODE
    matched_energy: bool = True
    num_beams: int = 1
    language: str = "zh"
    combo_mode: str | None = None
    weight_nat: float = 1.0
    weight_prompt: float = 1.0
    note: str = ""

    @property
    def cell_id(self) -> str:
        layers = []
        if self.encoder_layers:
            layers.append("E" + "-".join(str(l) for l in self.encoder_layers))
        if self.decoder_layers:
            layers.append("D" + "-".join(str(l) for l in self.decoder_layers))
        combo = f"_{self.combo_mode}" if self.combo_mode else ""
        if self.combo_mode == C.COMBO_WEIGHTED:
            combo += f"_n{self.weight_nat:g}p{self.weight_prompt:g}"
        beams = f"_b{self.num_beams}" if self.num_beams > 1 else ""
        return (f"{self.stage}:{self.site}:{self.coverage}:"
                f"{self.direction_kind}{combo}:a{self.alpha:g}:"
                f"{'+'.join(layers)}{beams}")

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_plan(self, *, encoder_direction=None, decoder_direction=None) -> SteeringPlan:
        return SteeringPlan(
            site=self.site, coverage=self.coverage, alpha=float(self.alpha),
            scale_mode=self.scale_mode, matched_energy=self.matched_energy,
            encoder_layers=tuple(self.encoder_layers),
            decoder_layers=tuple(self.decoder_layers),
            encoder_direction=encoder_direction,
            decoder_direction=decoder_direction,
            num_beams=int(self.num_beams))


@dataclass(frozen=True)
class BaselineCell:
    """A baseline decode. Runs first; every steered cell is scored against one."""

    name: str
    language: str
    num_beams: int
    note: str = ""

    @property
    def cell_id(self) -> str:
        return f"baseline:{self.name}"

    def config_hash(self) -> str:
        payload = json.dumps({"baseline": self.name, "language": self.language,
                              "num_beams": self.num_beams}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def baseline_cells() -> list[BaselineCell]:
    return [BaselineCell(name=b["name"], language=b["language"],
                         num_beams=b["num_beams"], note=b["note"])
            for b in C.BASELINES]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_a_cells() -> list[Cell]:
    """Site x coverage screen. v_nat only, mid layer, alpha=1, matched energy."""
    cells: list[Cell] = []
    for site in (C.SITE_ENCODER, C.SITE_DECODER, C.SITE_BOTH):
        enc = (C.MID_ENCODER_LAYER,) if site in (C.SITE_ENCODER, C.SITE_BOTH) else ()
        dec = (C.MID_DECODER_LAYER,) if site in (C.SITE_DECODER, C.SITE_BOTH) else ()
        for coverage in (C.COVERAGE_GLOBAL, C.COVERAGE_LOCAL):
            cells.append(Cell(
                stage="A", site=site, coverage=coverage,
                direction_kind=C.KIND_V_NAT, alpha=1.0,
                encoder_layers=enc, decoder_layers=dec,
                note="site x coverage screen at matched energy"))
    return cells


def stage_b_cells(top_a: Sequence[dict[str, Any]], *,
                  full_weighted_grid: bool = False) -> list[Cell]:
    """v_prompt at both coverages plus both combination modes.

    Section 3.2 specifies a coarse 3x3 grid for the weighted mode; section 3.6
    caps Stage B at about eight runs and forbids the full cross product. The
    default therefore runs three mixes along that grid -- nat-heavy, equal,
    prompt-heavy -- and `--full-weighted-grid` restores all nine. Which one ran
    is recorded on every row.
    """
    coverage_from_a = _preferred_decoder_coverage(top_a)
    cells: list[Cell] = []
    for coverage in (C.COVERAGE_GLOBAL, C.COVERAGE_LOCAL):
        cells.append(Cell(
            stage="B", site=C.SITE_DECODER, coverage=coverage,
            direction_kind=C.KIND_V_PROMPT, alpha=1.0,
            decoder_layers=(C.MID_DECODER_LAYER,),
            note="v_prompt is decoder-only: the encoder never sees the "
                 "language token"))

    if full_weighted_grid:
        mixes = [(n, p) for n in C.COMBO_WEIGHT_GRID for p in C.COMBO_WEIGHT_GRID]
    else:
        lo, mid, hi = C.COMBO_WEIGHT_GRID
        mixes = [(hi, lo), (mid, mid), (lo, hi)]
    for weight_nat, weight_prompt in mixes:
        cells.append(Cell(
            stage="B", site=C.SITE_DECODER, coverage=coverage_from_a,
            direction_kind=f"{C.KIND_V_NAT}+{C.KIND_V_PROMPT}", alpha=1.0,
            decoder_layers=(C.MID_DECODER_LAYER,),
            combo_mode=C.COMBO_WEIGHTED, weight_nat=weight_nat,
            weight_prompt=weight_prompt,
            note="weighted combination; cos(v_nat, v_prompt) is 0.010-0.030 so "
                 "the grid is meaningful rather than degenerate"))
    cells.append(Cell(
        stage="B", site=C.SITE_DECODER, coverage=coverage_from_a,
        direction_kind=f"{C.KIND_V_NAT}+{C.KIND_V_PROMPT}", alpha=1.0,
        decoder_layers=(C.MID_DECODER_LAYER,), combo_mode=C.COMBO_RANK2,
        note="rank-2 projection into span{v_nat, v_prompt}, equal weighting"))
    return cells


def _preferred_decoder_coverage(top_a: Sequence[dict[str, Any]]) -> str:
    """The coverage of the best Stage A cell that touched the decoder."""
    for row in top_a:
        if row.get("site") in (C.SITE_DECODER, C.SITE_BOTH):
            return str(row.get("coverage", C.COVERAGE_LOCAL))
    return C.COVERAGE_LOCAL


def stage_c_cells(best_b: dict[str, Any]) -> list[Cell]:
    """Alpha sweep on the winner of Stage B. Five runs."""
    base = _cell_from_row(best_b, stage="C")
    return [Cell(**{**asdict(base), "stage": "C", "alpha": float(a),
                    "note": "alpha sweep on the Stage B winner"})
            for a in C.ALPHA_GRID]


def stage_d_cells(best_c: dict[str, Any]) -> list[Cell]:
    """Layer sweep on the winner of Stage C: singles first, then multi.

    Whichever sites the winner touches are swept; the other site is held at its
    Stage C value so one variable moves at a time.
    """
    base = _cell_from_row(best_c, stage="D")
    cells: list[Cell] = []
    if base.site in (C.SITE_ENCODER, C.SITE_BOTH):
        for layer in C.ENCODER_SINGLE_LAYERS:
            cells.append(Cell(**{**asdict(base), "stage": "D",
                                 "encoder_layers": (int(layer),),
                                 "note": "encoder single-layer sweep"}))
        for layers in C.ENCODER_MULTI_LAYERS:
            cells.append(Cell(**{**asdict(base), "stage": "D",
                                 "encoder_layers": tuple(int(l) for l in layers),
                                 "note": "encoder multi-layer sweep"}))
    if base.site in (C.SITE_DECODER, C.SITE_BOTH):
        for layer in C.DECODER_SINGLE_LAYERS:
            cells.append(Cell(**{**asdict(base), "stage": "D",
                                 "decoder_layers": (int(layer),),
                                 "note": "decoder single-layer sweep"}))
        for layers in C.DECODER_MULTI_LAYERS:
            cells.append(Cell(**{**asdict(base), "stage": "D",
                                 "decoder_layers": tuple(int(l) for l in layers),
                                 "note": "decoder multi-layer sweep"}))
    # de-duplicate while preserving order
    seen, unique = set(), []
    for c in cells:
        if c.config_hash() not in seen:
            seen.add(c.config_hash())
            unique.append(c)
    return unique


def _cell_from_row(row: dict[str, Any], *, stage: str) -> Cell:
    """Rebuild the Cell a result row came from."""
    return Cell(
        stage=stage,
        site=str(row["site"]),
        coverage=str(row["coverage"]),
        direction_kind=str(row["direction_kind"]),
        alpha=float(row.get("alpha", 1.0)),
        encoder_layers=tuple(int(l) for l in (row.get("encoder_layers") or ())),
        decoder_layers=tuple(int(l) for l in (row.get("decoder_layers") or ())),
        scale_mode=str(row.get("scale_mode", C.DEFAULT_SCALE_MODE)),
        matched_energy=bool(row.get("matched_energy", True)),
        num_beams=int(row.get("num_beams", 1)),
        language=str(row.get("language", "zh")),
        combo_mode=row.get("combo_mode"),
        weight_nat=float(row.get("weight_nat", 1.0)),
        weight_prompt=float(row.get("weight_prompt", 1.0)),
    )


def confirm_cell(best_d: dict[str, Any]) -> Cell:
    """The single configuration Confirm would run. NEVER run automatically."""
    return Cell(**{**asdict(_cell_from_row(best_d, stage="Confirm")),
                   "note": "the ONE configuration proposed for D-dev-confirm; "
                           "requires explicit human approval"})
