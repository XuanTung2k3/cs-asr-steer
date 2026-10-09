"""SRD2-G0 reference-free mechanics (frozen config ``configs/inference_cf/srd2_g0.json``).

Allowlisted runtime projection, dense query inventory/identity, the unchanged R2 ``g_old`` wrapper
(original primitives only), the deterministic within-utterance gate permutation, reference-free dose
guards and the native-site observers used by the four-arm pulse screen.

Nothing here reads a reference transcript, target token set, stratum, CTC/MMS-FA timing or evaluator
quantity. The independent auditor must not import this module (it recomputes every equation itself).
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable, Sequence

import numpy as np

from .core import digest

VERSION = "srd2_g0_v1"
CB = (50258, 50260, 50360, 50364)
EOS = 50257
LAYER = 16
RUNTIME_KEYS = ("canonical_index", "utterance_id", "audio_path", "audio_full_sha256")
PANEL_KEYS = ("schema", "rows", "population_selected_ids_hash", "runtime_hash")
PANEL_SCHEMA = "srd2_g0_runtime_panel_v1"
PERMUTATION_TAG = "SRD2-G0-gate-permutation-v1"
SEED = 240924
FALLBACK_PRIORITY = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail",
                     "nonfinite_signal")
ARMS = ("B0", "B1", "B2", "B3")
EDIT_ARMS = ("B1", "B2", "B3")


# ---- runtime projection (firewall) ----------------------------------------------------------------

def runtime_projection(population: dict) -> dict:
    """The only runner input: exactly four identity/audio keys per selected utterance."""
    rows = [{k: r[k] for k in RUNTIME_KEYS} for r in population["selected"]]
    if [r["canonical_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("canonical_index is not dense")
    panel = {"schema": PANEL_SCHEMA, "rows": rows,
             "population_selected_ids_hash": digest([r["utterance_id"] for r in rows])}
    panel["runtime_hash"] = digest(panel)
    validate_runtime_panel(panel)
    return panel


def validate_runtime_panel(panel: dict) -> list[dict]:
    """Reject any extra/missing field at every level instead of ignoring it."""
    if not isinstance(panel, dict) or set(panel) != set(PANEL_KEYS) or panel["schema"] != PANEL_SCHEMA:
        raise ValueError("runtime panel keys differ from the allowlist")
    if digest({k: v for k, v in panel.items() if k != "runtime_hash"}) != panel["runtime_hash"]:
        raise ValueError("runtime panel hash mismatch")
    rows = panel["rows"]
    if not isinstance(rows, list):
        raise ValueError("runtime rows must be a list")
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or set(r) != set(RUNTIME_KEYS):
            raise ValueError(f"runtime row {i} has non-allowlisted fields")
        if r["canonical_index"] != i or not all(isinstance(r[k], str) for k in RUNTIME_KEYS[1:]):
            raise ValueError(f"runtime row {i} malformed")
    if len({r["utterance_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate utterance IDs")
    if digest([r["utterance_id"] for r in rows]) != panel["population_selected_ids_hash"]:
        raise ValueError("runtime IDs differ from the selected population hash")
    return rows


# ---- query identity ---------------------------------------------------------------------------------

def inventory(content: Sequence[int], terminated: str) -> list[int]:
    """t = 0..T-1, plus t = T only if the cached baseline actually stopped on EOS (no cap slot)."""
    if terminated not in ("eos", "cap"):
        raise ValueError("terminated must be eos or cap")
    n = len(content)
    return list(range(n + 1)) if terminated == "eos" else list(range(n))


def query_index(t: int, prompt_len: int = len(CB)) -> int:
    return prompt_len + int(t) - 1


def prefix_hash(prompt: Sequence[int], prefix: Sequence[int]) -> str:
    return digest([int(x) for x in prompt] + [int(x) for x in prefix])


def feed_tokens(content: Sequence[int], t: int, prompt: Sequence[int] = CB) -> list[int]:
    """Tokens fed at step t: the prompt at t=0, else the previous baseline content token."""
    return list(prompt) if t == 0 else [int(content[t - 1])]


def expected_action(content: Sequence[int], t: int, eos_id: int = EOS) -> int:
    return int(content[t]) if t < len(content) else int(eos_id)


def structural(t: int, prefix: Sequence[int], utf8_complete: bool, finite_site: bool, finite_logits: bool,
               eos_id: int = EOS) -> tuple[bool, list[str]]:
    """Frozen pulse eligibility: t>=1, complete UTF-8 prefix, no special content token, finite native site/logits."""
    reasons = []
    if t < 1:
        reasons.append("forced_prompt_query_t0")
    if not utf8_complete:
        reasons.append("mid_character_prefix")
    if any(int(x) >= int(eos_id) for x in prefix):
        reasons.append("unexpected_special_content_token")
    if not finite_site:
        reasons.append("nonfinite_native_site")
    if not finite_logits:
        reasons.append("nonfinite_native_logits")
    return (not reasons), reasons


# ---- unchanged R2 g_old wrapper ---------------------------------------------------------------------

def r2_gate(*, utf8_complete: bool, heads_q, heard_samples: int, raw_logits, partition: dict,
            lid: Callable[[int, int], dict], null_probs: dict, en_id: int, zh_id: int,
            primitives: dict) -> dict:
    """R2 ``inference_utterance`` g_old logic for one query, without the (unneeded) Ecf/donor branches.

    ``primitives`` carries the unchanged historical functions: max_attention_window, conflict_from_logits,
    local_support, gate_values, reason (R2 ``_reason``). Every diagnostic is computed, then the first
    reason in the frozen precedence is recorded; any reason zeroes the gate.
    """
    P = primitives
    out = {"window": None, "localizer_status": None, "baseline": None, "support": None, "lid": None,
           "fallback_reason": None, "eligibility_status": None, "E": None, "R_B": None, "g": 0.0}
    reasons = []
    if not utf8_complete:
        reasons.append("mid_character")
    window = None
    try:
        window = P["max_attention_window"](heads_q, heard_samples)
        out["window"] = window.record()
        out["localizer_status"] = "ok"
    except Exception as exc:
        reasons.append(P["reason"](exc, "localizer_fail"))
        out["localizer_status"] = str(exc)
    try:
        out["baseline"] = P["conflict_from_logits"](raw_logits, partition)
        out["R_B"] = out["baseline"]["R"]
    except Exception as exc:
        reasons.append(P["reason"](exc, "baseline_provider_fail"))
    if not reasons:
        try:
            probs = lid(window.start_sample, window.end_sample)
            out["lid"] = {"start_sample": int(window.start_sample), "end_sample": int(window.end_sample),
                          "pi_en": float(probs[en_id]), "pi_zh": float(probs[zh_id])}
            out["support"] = P["local_support"](probs[en_id], probs[zh_id], null_probs[en_id], null_probs[zh_id])
            out["E"] = out["support"]["E"]
        except Exception as exc:
            reasons.append(P["reason"](exc, "local_support_fail"))
    out["fallback_reason"] = next((r for r in FALLBACK_PRIORITY if r in reasons), None)
    if reasons:
        out["eligibility_status"] = "ineligible"
        return out
    try:
        out["g"] = float(P["gate_values"](out["support"], out["baseline"], {"R": 0.0})["g_old"])
        out["eligibility_status"] = "eligible"
    except Exception as exc:
        out["fallback_reason"] = P["reason"](exc, "nonfinite_signal")
        out["eligibility_status"] = "ineligible"
        out["g"] = 0.0
    return out


def float_bits(x: float) -> str:
    return np.float64(x).view(np.uint64).item().to_bytes(8, "big").hex()


# ---- deterministic within-utterance permutation -----------------------------------------------------

def permutation_key(uid: str, t: int) -> str:
    return hashlib.sha256(f"{PERMUTATION_TAG}|{SEED}|{uid}|{int(t)}".encode()).hexdigest()


def permute(uid: str, gates: dict[int, float]) -> dict:
    """Donors hash-sorted by SHA256(tag|seed|UID|t), then t; assigned to recipients in ascending t.

    ``gates`` maps every structurally eligible query t of ONE utterance to its sealed g (zeros included).
    No reroll; singleton, constant and identity outcomes are retained and flagged.
    """
    recipients = sorted(int(t) for t in gates)
    donors = sorted(recipients, key=lambda t: (permutation_key(uid, t), t))
    assigned = {r: float(gates[d]) for r, d in zip(recipients, donors)}
    pairs = [{"recipient_t": r, "donor_t": d, "g": float(gates[r]), "g_shuffled": assigned[r],
              "g_bits": float_bits(gates[r]), "g_shuffled_bits": float_bits(assigned[r])}
             for r, d in zip(recipients, donors)]
    values = [float(gates[t]) for t in recipients]
    identity_values = all(float_bits(assigned[r]) == float_bits(gates[r]) for r in recipients)
    return {"utterance_id": uid, "n": len(recipients), "pairs": pairs,
            "singleton": len(recipients) == 1, "constant": len({float_bits(v) for v in values}) <= 1,
            "identity_values": identity_values,
            "donor_identity_count": sum(r == d for r, d in zip(recipients, donors)),
            "uninformative": bool(recipients) and identity_values}


def bit_multiset(values: Sequence[float]) -> str:
    return digest(sorted(float_bits(v) for v in values))


def planned_sq_total(targets: Sequence[float]) -> float:
    return math.fsum(sorted(float(x) * float(x) for x in targets))


def target_chord(e_star: float, g: float) -> float:
    return 0.0 if float(g) == 0.0 else float(e_star) * float(g)


# ---- dose guards (CPU float64 chord geometry) ----------------------------------------------------------

def chord_guards(target: float, proposed_chord: float, consumed_chord: float, dose: dict) -> dict:
    """Frozen executed-edit guards: squared energy and chord within 2 %, norm-ratio within 0.5 %."""
    out = {"target": float(target), "proposed_chord": float(proposed_chord), "consumed_chord": float(consumed_chord)}
    if not (target > 0 and math.isfinite(proposed_chord) and math.isfinite(consumed_chord) and proposed_chord > 0):
        out.update(pass_all=False, reason="degenerate_chord")
        return out
    t2 = target * target
    out["proposed_rel_sq_err"] = abs(proposed_chord * proposed_chord / t2 - 1.0)
    out["consumed_rel_sq_err"] = abs(consumed_chord * consumed_chord / t2 - 1.0)
    out["proposed_rel_chord_err"] = abs(proposed_chord / target - 1.0)
    out["consumed_rel_chord_err"] = abs(consumed_chord / target - 1.0)
    out["consumed_vs_proposed_err"] = abs(consumed_chord / proposed_chord - 1.0)
    out["pass_all"] = bool(out["proposed_rel_sq_err"] <= dose["relative_squared_energy_error_max"]
                           and out["consumed_rel_sq_err"] <= dose["relative_squared_energy_error_max"]
                           and out["proposed_rel_chord_err"] <= dose["relative_chord_error_max"]
                           and out["consumed_rel_chord_err"] <= dose["relative_chord_error_max"]
                           and out["consumed_vs_proposed_err"] <= dose["consumed_vs_proposed_relative_error_max"])
    out["reason"] = None if out["pass_all"] else "native_consumption_unattainable"
    return out


# ---- compatibility arithmetic ---------------------------------------------------------------------

def softmax64(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    e = np.exp(z - m)
    return e / e.sum()


def total_variation(z_a: np.ndarray, z_b: np.ndarray) -> float:
    return float(0.5 * np.abs(softmax64(z_a) - softmax64(z_b)).sum())


def site_geometry(full: np.ndarray, cached: np.ndarray) -> dict:
    a, b = np.asarray(full, dtype=np.float64), np.asarray(cached, dtype=np.float64)
    nb = float(np.linalg.norm(b))
    if not nb > 0:
        return {"rel_L2": None, "cos": None}
    return {"rel_L2": float(np.linalg.norm(a - b) / nb),
            "cos": float(a @ b / (np.linalg.norm(a) * nb)) if np.linalg.norm(a) > 0 else None}


def heard_attention(heads_q: np.ndarray, heard_samples: int, frame_samples: int = 320, max_frames: int = 1500) -> np.ndarray:
    a = np.asarray(heads_q, dtype=np.float64)
    valid = min(max_frames, (int(heard_samples) + frame_samples - 1) // frame_samples)
    m = a[:, :valid].mean(axis=0)
    return m / m.sum()


# ---- bf16 lossless storage -------------------------------------------------------------------------

def bf16_bits(x) -> np.ndarray:
    """Float32/bf16 torch tensor holding bf16-representable values -> raw int16 bf16 bits (lossless)."""
    import torch
    b = x.detach().cpu().to(torch.bfloat16)
    if not torch.equal(b.float(), x.detach().cpu().float()):
        raise ValueError("values are not exactly bf16-representable")
    return b.view(torch.int16).numpy()


def from_bf16_bits(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int16)
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b"|" + repr(tuple(a.shape)).encode() + b"|")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


# ---- native observers (no edit) -----------------------------------------------------------------------

class NativeSite:
    """Passive observer of the L16 DG-02 site in NATIVE dtype (q, u_source, r = q + u) and of the actual
    FFN input (``final_layer_norm`` pre-hook argument, i.e. what the FFN consumes). Returns nothing."""

    def __init__(self, bundle, layer: int = LAYER):
        self.bundle, self.layer = bundle, int(layer)
        self.q = self.u = self.r = self.ffn = None
        self._q = None
        self._handles: list = []

    def _ln_pre(self, _mod, args):
        self._q = args[0]
        return None

    def _post(self, _mod, _inp, output):
        u = output[0] if isinstance(output, tuple) else output
        self.q = self._q[:, -1].detach().clone()
        self.u = u[:, -1].detach().clone()
        self.r = (self._q + u)[:, -1].detach().clone()
        return None

    def _ffn_pre(self, _mod, args):
        self.ffn = args[0][:, -1].detach().clone()
        return None

    def __enter__(self):
        lay = self.bundle.decoder_layer(self.layer)
        self._handles.append(lay.encoder_attn_layer_norm.register_forward_pre_hook(self._ln_pre))
        self._handles.append(lay.encoder_attn.register_forward_hook(self._post))
        self._handles.append(lay.final_layer_norm.register_forward_pre_hook(self._ffn_pre))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q = None
        return False


def assert_no_hooks_anywhere(bundle) -> None:
    """DG-02/self-attention/FFN-LN/layer hook leak check over every decoder layer."""
    leaked = []
    for i, lay in enumerate(bundle.model.model.decoder.layers):
        n = (len(lay.encoder_attn._forward_hooks) + len(lay.encoder_attn_layer_norm._forward_pre_hooks)
             + len(lay._forward_pre_hooks) + len(lay._forward_hooks) + len(lay.self_attn._forward_hooks)
             + len(lay.self_attn_layer_norm._forward_pre_hooks) + len(lay.final_layer_norm._forward_pre_hooks))
        if n:
            leaked.append(f"decoder[{i}]:{n}")
    if leaked:
        raise RuntimeError(f"hooks leaked on {leaked}")


class FutureMassProbe:
    """Passive: max decoder self-attention probability mass on strictly-future keys (causality evidence)."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.max_mass = 0.0
        self.calls = 0
        self._handles: list = []

    def _hook(self, _mod, _inp, output):
        w = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if w is None:
            raise RuntimeError("self-attention weights unavailable (output_attentions=False?)")
        L = w.shape[-1]
        if w.shape[-2] == L and L > 1:
            fut = w.float().triu(1).sum(-1).max()
            self.max_mass = max(self.max_mass, float(fut))
        self.calls += 1
        return None

    def __enter__(self):
        for lay in self.bundle.model.model.decoder.layers:
            self._handles.append(lay.self_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def native_preview(q, u, r, dirs):
    """The unchanged DG-02 arithmetic on the native device/dtype: proposed r~ = NormPreserve(r + dirs) via
    ``apply_steering`` (alpha=scale=gain=1), and the FFN input the layer will consume, q + (u + (r~ - r))."""
    import torch
    from csasr.models.hooks import apply_steering
    x = r.reshape(1, 1, -1)
    prop = apply_steering(x, dirs.reshape(1, 1, -1), 1.0, 1.0,
                          torch.ones((1, 1), device=x.device, dtype=x.dtype), True)
    consumed = q.reshape(1, 1, -1) + (u.reshape(1, 1, -1) + (prop - x))
    return prop.reshape(-1), consumed.reshape(-1)


def chord64(a, b) -> float:
    import torch
    return float(torch.linalg.vector_norm(a.detach().double().cpu() - b.detach().double().cpu()))


class CachedSolver:
    """Hands ``pulse_action`` the ONE solve already computed for this (site bits, direction, target); any other
    call is refused, so the hook never solves a second time."""

    def __init__(self, site, v, target: float, result: dict):
        self.key = (tensor_sha(site), tensor_sha(v), float(target))
        self.result = dict(result)
        self.calls = 0

    def __call__(self, site_vec, v, target):
        self.calls += 1
        if (tensor_sha(site_vec), tensor_sha(v), float(target)) != self.key:
            raise RuntimeError("cached solver called with a different site/direction/target")
        return dict(self.result)


def tensor_sha(x) -> str:
    import torch
    t = x.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return array_sha(t.numpy())
