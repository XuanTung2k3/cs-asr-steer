"""Frame masks g_t(C) for local steering (guide sections 37, 48-50).

All masks return a float array over the full encoder frame axis; padding frames
are always zero. Expansion and jitter answer different questions and are kept
distinct: expansion widens a correctly located span, jitter mislocates it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

MASK_KINDS = (
    "EXACT_HARD", "EXACT_TAPER", "EXPAND_50", "EXPAND_100", "EXPAND_200",
    "JITTER_50", "JITTER_100", "JITTER_200", "BOUNDARY_ONLY",
    "WHOLE_WORD_PLUS_CONTEXT", "GLOBAL",
)


@dataclass
class MaskSpec:
    kind: str
    shoulder_frames: int = 5          # 100 ms at 20 ms/frame
    expand_frames: int = 0
    jitter_frames: int = 0
    boundary_radius_frames: int = 5
    seed: int | None = None
    context_span: tuple[int, int] | None = None   # for WHOLE_WORD_PLUS_CONTEXT
    meta: dict = field(default_factory=dict)
def derive_jitter_seed(seed: int, utterance_id: str) -> int:
    """Stable independent RNG seed for one jitter replicate and utterance."""
    payload = f"{int(seed)}\0{utterance_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")




def ms_to_frames(ms: float, step_sec: float) -> int:
    return int(round((ms / 1000.0) / step_sec))


def _clip(s: int, e: int, num_valid: int) -> tuple[int, int]:
    s = max(0, min(int(s), num_valid - 1))
    e = max(s + 1, min(int(e), num_valid))
    return s, e


def hard_mask(start: int, end: int, num_frames: int, num_valid: int) -> np.ndarray:
    g = np.zeros(num_frames, dtype=np.float32)
    s, e = _clip(start, end, num_valid)
    g[s:e] = 1.0
    return g


def tapered_mask(start: int, end: int, num_frames: int, num_valid: int,
                 shoulder: int) -> np.ndarray:
    """1 inside the core, Hann rise/fall over ``shoulder`` frames, 0 outside."""
    g = hard_mask(start, end, num_frames, num_valid)
    if shoulder <= 0:
        return g
    s, e = _clip(start, end, num_valid)
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(1, shoulder + 1) / (shoulder + 1))))
    for k in range(shoulder):
        i = s - 1 - k
        if 0 <= i < num_valid:
            g[i] = max(g[i], float(ramp[shoulder - 1 - k]))
        j = e + k
        if 0 <= j < num_valid:
            g[j] = max(g[j], float(ramp[shoulder - 1 - k]))
    return g


def jitter_span(start: int, end: int, jitter: int, rng: np.random.Generator,
                num_valid: int) -> tuple[int, int, int, int]:
    """Independently perturb both boundaries; returns (s, e, ds, de)."""
    ds = int(rng.integers(-jitter, jitter + 1))
    de = int(rng.integers(-jitter, jitter + 1))
    s, e = start + ds, end + de
    if e <= s:
        e = s + 1
    s, e = _clip(s, e, num_valid)
    return s, e, ds, de


def build_gain(spec: MaskSpec, start_frame: int, end_frame: int,
               num_frames: int, num_valid: int) -> tuple[np.ndarray, dict]:
    """Build the gain vector for one target span. Returns (gain, applied_meta)."""
    kind = spec.kind
    meta: dict = {"kind": kind, "core_start": int(start_frame), "core_end": int(end_frame)}

    if kind == "GLOBAL":
        g = np.zeros(num_frames, dtype=np.float32)
        g[:num_valid] = 1.0
        meta.update(applied_start=0, applied_end=int(num_valid))
        return g, meta

    if kind == "EXACT_HARD":
        s, e = start_frame, end_frame
        g = hard_mask(s, e, num_frames, num_valid)

    elif kind == "EXACT_TAPER":
        s, e = start_frame, end_frame
        g = tapered_mask(s, e, num_frames, num_valid, spec.shoulder_frames)

    elif kind.startswith("EXPAND"):
        s, e = start_frame - spec.expand_frames, end_frame + spec.expand_frames
        g = tapered_mask(s, e, num_frames, num_valid, spec.shoulder_frames)

    elif kind.startswith("JITTER"):
        rng = np.random.default_rng(spec.seed if spec.seed is not None else 242)
        s, e, ds, de = jitter_span(start_frame, end_frame, spec.jitter_frames,
                                   rng, num_valid)
        meta.update(jitter_start_frames=ds, jitter_end_frames=de)
        g = tapered_mask(s, e, num_frames, num_valid, spec.shoulder_frames)

    elif kind == "BOUNDARY_ONLY":
        r = spec.boundary_radius_frames
        g = np.zeros(num_frames, dtype=np.float32)
        for centre in (start_frame, end_frame):
            a, b = _clip(centre - r, centre + r, num_valid)
            g[a:b] = np.maximum(g[a:b], 1.0)
        s, e = start_frame, end_frame
        meta.update(boundary_radius_frames=r)

    elif kind == "WHOLE_WORD_PLUS_CONTEXT":
        if spec.context_span is None:
            s, e = start_frame, end_frame
        else:
            s, e = spec.context_span
        g = tapered_mask(s, e, num_frames, num_valid, spec.shoulder_frames)

    else:
        raise ValueError(f"unknown mask kind {kind!r}")

    cs, ce = _clip(s, e, num_valid)
    meta.update(applied_start=int(cs), applied_end=int(ce),
                num_active_frames=int((g > 0).sum()),
                mass=float(g.sum()))
    return g, meta


def spec_from_kind(kind: str, step_sec: float, shoulder_ms: float = 100.0,
                   boundary_only_ms: float = 100.0, seed: int | None = None) -> MaskSpec:
    """Instantiate the canonical spec for one of the E5 mask names."""
    shoulder = ms_to_frames(shoulder_ms, step_sec)
    spec = MaskSpec(kind=kind, shoulder_frames=shoulder,
                    boundary_radius_frames=ms_to_frames(boundary_only_ms, step_sec),
                    seed=seed)
    if kind == "EXACT_HARD":
        spec.shoulder_frames = 0
    elif kind.startswith("EXPAND_"):
        spec.expand_frames = ms_to_frames(float(kind.split("_")[1]), step_sec)
    elif kind.startswith("JITTER_"):
        spec.jitter_frames = ms_to_frames(float(kind.split("_")[1]), step_sec)
    return spec
