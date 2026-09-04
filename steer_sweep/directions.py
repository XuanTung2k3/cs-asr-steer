"""The direction store: v_nat, v_nat_relaxed, v_prompt.

Three kinds, kept strictly apart. A relaxed span filter is a DIFFERENT object
and is registered as `v_nat_relaxed`; it is never silently substituted for
`v_nat`, and a request for a kind that does not exist raises rather than
falling back.

What the frozen v2r3 store actually contains (verified 2026-08-18):

    site_e_layer{15,23,27,31}_{hann,central60,uniform}
    site_d_layer{8,16,24,31}

float64, NOT unit norm (0.75 .. 5.54), and every site_d vector already has the
rank-1 prompt subspace PROJECTED OUT. There is therefore no v_prompt in the
store to load, and the layers this study sweeps only partly overlap the stored
set. Both facts are recorded in `config.SPEC_DEVIATIONS`.

  * a stored layer is loaded and unit-normalised          -> source "loaded"
  * a missing layer is rebuilt on `D-construct` through the identical frozen
    Day-3 pipeline                                        -> source "reconstructed"
  * v_prompt is constructed by the specified method       -> source "derived"

Every direction carries its source, so no table can conflate them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as C
from .hooks import DirectionSpec, PromptDirectionAtEncoderError

STORE_RELATIVE = "directions/generation_001/directions.npz"


def store_path(cfg: dict) -> Path:
    return Path(cfg["v2_namespace"]["root"]) / STORE_RELATIVE


def _unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction has zero or non-finite norm")
    return (v / norm).astype(np.float32), norm


class DirectionStore:
    """Loads, reconstructs and caches every direction the sweep needs."""

    def __init__(self, cfg: dict, cache_dir: Path, *, log=None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._raw: dict[str, np.ndarray] | None = None
        self._derived: dict[str, np.ndarray] = {}
        self.provenance: dict[str, dict[str, Any]] = {}
        self._load_derived_cache()

    # -- stored ------------------------------------------------------------
    @property
    def raw(self) -> dict[str, np.ndarray]:
        if self._raw is None:
            path = store_path(self.cfg)
            if not path.exists():
                raise FileNotFoundError(f"direction store missing: {path}")
            from csasr.lss import manifest as manifest_mod
            check = manifest_mod.verify(path, cfg=None, require_identity=False)
            if not check["ok"]:
                raise RuntimeError(f"direction store failed byte verification: {check}")
            self._raw = dict(np.load(path))
        return self._raw

    def stored_keys(self) -> list[str]:
        return sorted(self.raw)

    # -- derived cache -----------------------------------------------------
    def _cache_file(self) -> Path:
        return self.cache_dir / "derived_directions.npz"

    def _prov_file(self) -> Path:
        return self.cache_dir / "derived_directions.json"

    def _load_derived_cache(self) -> None:
        f, p = self._cache_file(), self._prov_file()
        if f.exists():
            self._derived = dict(np.load(f))
        if p.exists():
            self.provenance.update(json.loads(p.read_text(encoding="utf-8")))

    def _save_derived_cache(self) -> None:
        np.savez(self._cache_file(), **self._derived)
        self._prov_file().write_text(
            json.dumps(self.provenance, indent=2, default=str), encoding="utf-8")

    # -- public API --------------------------------------------------------
    def key(self, kind: str, site: str, layer: int, *, pooling: str = "hann") -> str:
        if site == C.SITE_ENCODER:
            return f"site_e_layer{int(layer)}_{pooling}"
        return f"site_d_layer{int(layer)}"

    def get(self, kind: str, site: str, layer: int, *, pooling: str = "hann",
            bundle=None, population=None) -> DirectionSpec:
        """Return one unit-norm float32 direction bound to (kind, site, layer)."""
        if kind in C.DECODER_ONLY_KINDS and site == C.SITE_ENCODER:
            raise PromptDirectionAtEncoderError(
                f"{kind} requested at an encoder site (layer {layer}). Whisper's "
                f"encoder never receives the language token; there is no such "
                f"object. Refusing to produce a number.")

        if kind == C.KIND_V_PROMPT:
            vector, source = self._v_prompt(layer, bundle=bundle, population=population)
        elif kind == C.KIND_V_NAT:
            vector, source = self._v_nat(site, layer, pooling=pooling, bundle=bundle)
        elif kind == C.KIND_V_NAT_RELAXED:
            vector, source = self._v_nat_relaxed(site, layer, pooling=pooling,
                                                 bundle=bundle)
        else:
            raise KeyError(f"unknown direction kind {kind!r}")

        unit, norm = _unit(vector)
        return DirectionSpec(kind=kind, site=site, layer=int(layer),
                             vector=torch.from_numpy(unit), source=source,
                             norm_before_unit=norm)

    # -- v_nat -------------------------------------------------------------
    def _v_nat(self, site: str, layer: int, *, pooling: str, bundle) -> tuple[np.ndarray, str]:
        key = self.key(C.KIND_V_NAT, site, layer, pooling=pooling)
        if key in self.raw:
            self.provenance.setdefault(key, {
                "kind": C.KIND_V_NAT, "site": site, "layer": int(layer),
                "source": "loaded", "store": str(store_path(self.cfg)),
                "stored_norm": float(np.linalg.norm(self.raw[key])),
            })
            return self.raw[key], "loaded"
        if key in self._derived:
            return self._derived[key], "reconstructed"
        if bundle is None:
            raise KeyError(
                f"{key} is not in the frozen store and no model was supplied to "
                f"reconstruct it. The store holds {self.stored_keys()}. Refusing "
                f"to substitute a neighbouring layer.")
        vector = self._reconstruct(site, layer, pooling=pooling, bundle=bundle,
                                   relaxed=False)
        self._derived[key] = vector
        self._save_derived_cache()
        return vector, "reconstructed"

    def _v_nat_relaxed(self, site: str, layer: int, *, pooling: str,
                       bundle) -> tuple[np.ndarray, str]:
        key = self.key(C.KIND_V_NAT, site, layer, pooling=pooling) + "_relaxed"
        if key in self._derived:
            return self._derived[key], "reconstructed"
        if bundle is None:
            raise KeyError(f"{key} not built and no model supplied")
        vector = self._reconstruct(site, layer, pooling=pooling, bundle=bundle,
                                   relaxed=True)
        self._derived[key] = vector
        self._save_derived_cache()
        return vector, "reconstructed"

    def _construction_inputs(self, bundle):
        """The `D-construct` spans and manifest the Day-3 recipe is fitted on."""
        from csasr.experiments import v2r3_day5_expansion as D5

        cfg = self.cfg
        root = Path(cfg["experiment"]["output_root"])
        candidates = pd.read_parquet(root / "alignments" / "candidates_all.parquet")
        role_root = Path(cfg["v2_namespace"]["role_root"])
        poi_root = root.parent.parent / "baselines" / "generation_001"
        manifest = pd.read_parquet(role_root / f"role_{C.CONSTRUCT}.parquet")
        poi = pd.read_parquet(poi_root / f"poi_{C.CONSTRUCT}.parquet").rename(
            columns={"poi_index": "reference_unit_index"})
        dialogue_of = dict(zip(manifest["conversation_id"].astype(str),
                               manifest["dialogue_id"].astype(str)))
        return candidates, manifest, poi, dialogue_of, D5

    def _reconstruct(self, site: str, layer: int, *, pooling: str, bundle,
                     relaxed: bool) -> np.ndarray:
        """Rebuild one v_nat through the frozen Day-3 pipeline.

        Identical recipe: matrix-language control frames, ridge nuisance
        residualisation fitted on `D-construct`, 99th-percentile norm clipping,
        dialogue-balanced aggregation, rank-1 prompt subspace projected out at
        the decoder site. Only the layer changes.

        `relaxed` widens the span duration floor from the conservative 400 ms
        tier to 200 ms. It is a DIFFERENT construction and is only ever
        returned under the `v_nat_relaxed` kind.
        """
        from csasr.experiments import v2r3_directions as D3

        floor = 200.0 if relaxed else C.TARGET_FLOOR_MS
        candidates, manifest, poi, dialogue_of, D5 = self._construction_inputs(bundle)
        spans = D5._spans_at_floor(candidates, dialogue_of, poi, floor)
        construct = spans[(spans["role"] == C.CONSTRUCT) & spans["all_correct"]]
        build = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
        if self.log:
            self.log.info("reconstructing v_nat%s at %s layer %d over %d spans",
                          "_relaxed" if relaxed else "", site, layer, len(construct))

        if site == C.SITE_ENCODER:
            runs = D3.matrix_runs(candidates)
            meta, raw = D3.site_e_contrasts(bundle, build, construct, runs, [int(layer)],
                                            batch_size=4, log=self.log)
            vecs = raw["vectors"][(int(layer), pooling)]
            entry = D3.assemble(np.vstack(vecs), meta)
        else:
            meta, raw = D3.site_d_contrasts(bundle, build, construct, [int(layer)],
                                            batch_size=4, log=self.log)
            prompts = np.vstack(raw["prompts"][int(layer)])
            basis = D3.orthonormal_basis(prompts, D3.PROMPT_SUBSPACE_RANK)
            entry = D3.assemble(np.vstack(raw["vectors"][int(layer)]), meta, basis=basis)

        key = self.key(C.KIND_V_NAT, site, layer, pooling=pooling) + (
            "_relaxed" if relaxed else "")
        self.provenance[key] = {
            "kind": C.KIND_V_NAT_RELAXED if relaxed else C.KIND_V_NAT,
            "site": site, "layer": int(layer), "source": "reconstructed",
            "recipe": "frozen v2r3 Day-3 pipeline, identical except the layer",
            "floor_ms": floor, "pooling": pooling if site == C.SITE_ENCODER else None,
            "pairs": entry.get("pairs"), "dialogues": entry.get("dialogues"),
            "direction_norm": entry.get("direction_norm"),
            "bootstrap_cosine": entry.get("bootstrap_cosine"),
        }
        return np.asarray(entry["direction"], dtype=np.float64)

    # -- v_prompt ----------------------------------------------------------
    def _v_prompt(self, layer: int, *, bundle, population) -> tuple[np.ndarray, str]:
        """Difference-in-means over prompt language-token states.

        For every construction utterance the decoder is run teacher-forced
        twice -- once with the `<|en|>` prefix and once with the `<|zh|>`
        prefix, identical audio and identical continuation -- and the state at
        the LANGUAGE-TOKEN position (prefix index 1) of decoder layer `layer`
        is taken. v_prompt is the dialogue-balanced difference in means.

        This object does not exist in the frozen store: the store's rank-1
        prompt subspace was projected OUT of every v_nat, which is why
        cos(v_nat, v_prompt) is measured at 0.010-0.030 and the two are
        near-orthogonal by construction as well as empirically.
        """
        key = f"site_d_layer{int(layer)}_v_prompt"
        if key in self._derived:
            return self._derived[key], "derived"
        if bundle is None:
            raise KeyError(f"{key} not built and no model supplied")

        from csasr.data.alignment import build_prefix
        from csasr.data.normalize import normalize_and_segment
        from csasr.experiments import v2r3_directions as D3
        from csasr.lss.sites import DecoderPostCrossAttnRecorder
        from csasr.models.generation import teacher_forced_forward

        candidates, manifest, poi, dialogue_of, D5 = self._construction_inputs(bundle)
        spans = D5._spans_at_floor(candidates, dialogue_of, poi, C.TARGET_FLOOR_MS)
        construct = spans[(spans["role"] == C.CONSTRUCT) & spans["all_correct"]]
        build = manifest[manifest["utterance_id"].isin(set(construct["utterance_id"]))]
        build = build.drop_duplicates("utterance_id").reset_index(drop=True)

        tokenizer = bundle.processor.tokenizer
        eot = tokenizer.eos_token_id
        prefix_zh = build_prefix(bundle.processor, language="zh")
        prefix_en = build_prefix(bundle.processor, language="en")
        if len(prefix_zh) != C.PREFIX_WIDTH:
            raise RuntimeError(f"unexpected prefix width {len(prefix_zh)}")
        lang_pos = 1                                   # <|zh|> / <|en|> position

        diffs: list[np.ndarray] = []
        dialogues: list[str] = []
        batch_size = 4
        for start in range(0, len(build), batch_size):
            batch = build.iloc[start:start + batch_size]
            seq_zh, seq_en = [], []
            for _, row in batch.iterrows():
                norm, _units = normalize_and_segment(row["transcript_raw"])
                text_ids = tokenizer.encode(norm, add_special_tokens=False)[:220]
                seq_zh.append(list(prefix_zh) + list(text_ids) + [eot])
                seq_en.append(list(prefix_en) + list(text_ids) + [eot])
            states = {}
            for tag, sequences in (("zh", seq_zh), ("en", seq_en)):
                with DecoderPostCrossAttnRecorder(bundle, [int(layer)]) as rec:
                    out, _padded, _mask = teacher_forced_forward(
                        bundle, batch["audio_path"].tolist(), sequences,
                        output_attentions=False)
                    states[tag] = rec.states[int(layer)].float().cpu().numpy()
                del out
            for b, (_i, row) in enumerate(batch.iterrows()):
                diffs.append(states["en"][b, lang_pos] - states["zh"][b, lang_pos])
                dialogues.append(str(row["dialogue_id"]))
            if self.log and (start // batch_size) % 20 == 0:
                self.log.info("  v_prompt layer %d: %d/%d utterances",
                              layer, start + len(batch), len(build))

        contrasts = np.vstack(diffs)
        direction, weighting = D3.dialogue_balanced_mean(
            contrasts, pd.Series(dialogues, name="dialogue_id"))
        stability = D3.bootstrap_cosine(contrasts, dialogues)
        self.provenance[key] = {
            "kind": C.KIND_V_PROMPT, "site": C.SITE_DECODER, "layer": int(layer),
            "source": "derived",
            "construction": ("difference in means of the decoder state at the "
                             "language-token prefix position (index 1) between "
                             "an <|en|> and a <|zh|> prefix, identical audio and "
                             "continuation; dialogue-balanced aggregation"),
            "pairs": int(len(contrasts)), "dialogues": weighting.get("G"),
            "direction_norm": float(np.linalg.norm(direction)),
            "bootstrap_cosine": stability,
            "split": C.CONSTRUCT,
            "note": ("not present in the frozen store: the store's rank-1 prompt "
                     "subspace was projected OUT of every v_nat"),
        }
        self._derived[key] = np.asarray(direction, dtype=np.float64)
        self._save_derived_cache()
        return self._derived[key], "derived"

    # -- combinations ------------------------------------------------------
    def combined(self, v_nat: DirectionSpec, v_prompt: DirectionSpec, *,
                 mode: str, weight_nat: float = 1.0,
                 weight_prompt: float = 1.0) -> tuple[DirectionSpec, dict[str, Any]]:
        """Cell 5: `weighted` or `rank2` combination at the decoder site.

        `weighted` is alpha*v_nat + beta*v_prompt, renormalised to unit norm so
        the dose stays in alpha and the mix stays in the weights.
        `rank2` projects the edit into span{v_nat, v_prompt} with equal
        weighting, using the orthonormal basis of the two vectors.
        """
        if v_nat.site == C.SITE_ENCODER or v_prompt.site == C.SITE_ENCODER:
            raise PromptDirectionAtEncoderError(
                "a combined v_nat/v_prompt direction is decoder-only")
        a = v_nat.vector.numpy().astype(np.float64)
        b = v_prompt.vector.numpy().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if mode == C.COMBO_WEIGHTED:
            mixed = weight_nat * a + weight_prompt * b
            detail = {"mode": mode, "weight_nat": weight_nat,
                      "weight_prompt": weight_prompt}
        elif mode == C.COMBO_RANK2:
            basis, _ = np.linalg.qr(np.stack([a, b], axis=1))    # (D, 2)
            mixed = basis @ np.array([1.0, 1.0]) / np.sqrt(2.0)
            detail = {"mode": mode, "weight_nat": 1.0, "weight_prompt": 1.0,
                      "basis": "QR orthonormalisation of [v_nat, v_prompt]"}
        else:
            raise ValueError(f"unknown combination mode {mode!r}")
        unit, norm = _unit(mixed)
        detail.update({"cos_v_nat_v_prompt": cos, "norm_before_unit": norm})
        spec = DirectionSpec(
            kind=f"{C.KIND_V_NAT}+{C.KIND_V_PROMPT}:{mode}", site=C.SITE_DECODER,
            layer=v_nat.layer, vector=torch.from_numpy(unit), source="derived",
            norm_before_unit=norm)
        return spec, detail


def cosine(a: DirectionSpec, b: DirectionSpec) -> float:
    x, y = a.vector.numpy(), b.vector.numpy()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
