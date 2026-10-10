"""DIR-SPRINT0 reference-free mechanics (frozen config ``configs/inference_cf/dir_sprint0.json``).

Three new direction families at the L16 DG-02 site, plus the frozen construction of every historical comparator:

* D3 acoustic-prior contrast: the frozen plausibility-constrained candidate rule on original (B0) versus null-audio
  processed log-probabilities, and the candidate-conditioned token-margin readout gradient d_AP (zero-probe scratch
  autograd, the historical D2 mechanics with the objective J = z[c_AP] - z[b_t]).
* D4 transcription-fidelity contrast: tangent(h_transcribe - h_translate), historical float64 tangent rule.
* D5 phonetic-concept steering: concept evidence from the frozen phone provider's full-audio posteriors over the R2
  window, calibration-bank concept prototypes, the query-conditioned direction and the deterministic within-utterance
  shuffled-evidence control.
* Comparators: D0 (core_p1.direction), D1 (sealed P2-DIR fold vector of the utterance's dialogue), D2 (unchanged
  ReadoutDirection), v_AC (R0 regions + S1 association + SRC-CF0-P direction), matched isotropic random.

Nothing here reads a reference transcript, acceptable-token set, stratum, CTC/MMS-FA timing, language label or
evaluator quantity. The independent auditor must not import this module (it recomputes every equation itself).
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Sequence

import numpy as np

from .core import digest

VERSION = "dir_sprint0_v1"
CB = (50258, 50260, 50360, 50364)    # <|startoftranscript|><|zh|><|transcribe|><|notimestamps|>  (B0 / transcribe)
CE = (50258, 50259, 50360, 50364)    # forced-EN transcribe (D0 only)
CTL = (50258, 50260, 50359, 50364)   # <|zh|><|translate|> (D4 only)
PROMPT_TOKENS = {"CB": ["<|startoftranscript|>", "<|zh|>", "<|transcribe|>", "<|notimestamps|>"],
                 "CE": ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"],
                 "CTL": ["<|startoftranscript|>", "<|zh|>", "<|translate|>", "<|notimestamps|>"]}
EOS = 50257
LAYER = 16
DIM = 1280
SEED = 240924
RUNTIME_KEYS = ("canonical_index", "utterance_id", "audio_path", "audio_full_sha256")
BANK_KEYS = ("bank_index", "utterance_id", "audio_path", "audio_full_sha256")
PANEL_KEYS = ("schema", "rows", "bank_rows", "population_selected_ids_hash", "bank_ids_hash", "runtime_hash")
PANEL_SCHEMA = "dir_sprint0_runtime_panel_v1"

# ---- arms ---------------------------------------------------------------------------------------------------------
UNGATED = ("D0", "D1", "D2", "VAC", "RND", "D3", "D4", "D5", "D5SH")
GATED = ("D2G", "D3G", "D4G", "D5G", "D5SHG", "RNDG")
PULSE_ARMS = UNGATED + GATED
GATED_BASE = {"D2G": "D2", "D3G": "D3", "D4G": "D4", "D5G": "D5", "D5SHG": "D5SH", "RNDG": "RND"}
NEW_FAMILIES = ("D3", "D4", "D5")

# ---- D3 frozen constants --------------------------------------------------------------------------------------------
D3_LOG_ALPHA = math.log(1e-3)       # plausibility: l_B(v) >= max l_B + log(1e-3)
D3_LOG_NULL_FLOOR = math.log(1e-12)  # floor on the null log-probability, applied AFTER the float32 log-softmax

# ---- D5 frozen constants --------------------------------------------------------------------------------------------
PRIMARY_CONCEPTS = ("NAS", "STOP", "FRIC", "LAB", "COR", "DOR")
DESCRIPTIVE_CONCEPTS = ("VOI_OBS", "ASP")
EMIT_MIN = 0.5            # a provider frame is phone-emitting iff its segmental posterior mass >= 0.5
WINDOW_FRAMES_MIN = 20    # provider frames with positive overlap weight inside the R2 window
EMITTING_WEIGHT_MIN = 2.0  # >= two emitting frame-equivalents in the window
EXCLUDED_SHARE_MAX = 0.1  # excluded-symbol share of (segmental + excluded) mass over emitting frames
Q_LO, Q_HI = 0.2, 0.8     # bank pseudo-feature quantiles (numpy linear)
DIALOGUE_SIDE_MIN = 5     # bank queries per side per dialogue for a dialogue to enter a prototype
DIALOGUES_MIN = 15        # dialogues entering every prototype
SIDE_TOTAL_MIN = 500      # bank queries per side overall
SD_MIN = 1e-6
Z_NORM_MIN = 1e-6
SHUFFLE_TAG = "DIR-SPRINT0-d5-shuffle-v1"
RANDOM_TAG = "DIR-SPRINT0-random-v1"


# ---- runtime projection (firewall) ---------------------------------------------------------------------------------

def runtime_projection(population: dict) -> dict:
    """The only GPU-runner input: four identity/audio keys per prospective row and per calibration-bank row."""
    rows = [{k: r[k] for k in RUNTIME_KEYS} for r in population["selected"]]
    bank = [{k: r[k] for k in BANK_KEYS} for r in population["calibration_bank"]["rows"]]
    if [r["canonical_index"] for r in rows] != list(range(len(rows))) or \
            [r["bank_index"] for r in bank] != list(range(len(bank))):
        raise ValueError("indices are not dense")
    panel = {"schema": PANEL_SCHEMA, "rows": rows, "bank_rows": bank,
             "population_selected_ids_hash": digest([r["utterance_id"] for r in rows]),
             "bank_ids_hash": digest(sorted(r["utterance_id"] for r in bank))}
    panel["runtime_hash"] = digest(panel)
    validate_runtime_panel(panel)
    return panel


def validate_runtime_panel(panel: dict) -> tuple[list[dict], list[dict]]:
    """Reject any extra/missing field at every level instead of ignoring it."""
    if not isinstance(panel, dict) or set(panel) != set(PANEL_KEYS) or panel["schema"] != PANEL_SCHEMA:
        raise ValueError("runtime panel keys differ from the allowlist")
    if digest({k: v for k, v in panel.items() if k != "runtime_hash"}) != panel["runtime_hash"]:
        raise ValueError("runtime panel hash mismatch")
    for name, keys, idx in (("rows", RUNTIME_KEYS, "canonical_index"), ("bank_rows", BANK_KEYS, "bank_index")):
        rows = panel[name]
        if not isinstance(rows, list):
            raise ValueError(f"{name} must be a list")
        for i, r in enumerate(rows):
            if not isinstance(r, dict) or set(r) != set(keys):
                raise ValueError(f"{name} row {i} has non-allowlisted fields")
            if r[idx] != i or not all(isinstance(r[k], str) for k in keys[1:]):
                raise ValueError(f"{name} row {i} malformed")
    ids = [r["utterance_id"] for r in panel["rows"]]
    bank = [r["utterance_id"] for r in panel["bank_rows"]]
    if len(set(ids)) != len(ids) or len(set(bank)) != len(bank) or set(ids) & set(bank):
        raise ValueError("duplicate or overlapping utterance IDs")
    if digest(ids) != panel["population_selected_ids_hash"] or digest(sorted(bank)) != panel["bank_ids_hash"]:
        raise ValueError("runtime IDs differ from the frozen population hashes")
    return panel["rows"], panel["bank_rows"]


# ---- D3: legal tokens, candidate rule -------------------------------------------------------------------------------

def utf8_boundary_legal(b: bytes) -> bool:
    """Token bytes can start right after a complete UTF-8 prefix: the first byte is not a continuation byte and the
    bytes are a valid UTF-8 sequence or a valid prefix of one (strict decode fails only with 'unexpected end')."""
    if not b or 0x80 <= b[0] <= 0xBF:
        return False
    try:
        b.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError as exc:
        return exc.reason == "unexpected end of data" and exc.end == len(b)


def legal_static(tokenizer, suppress: Sequence[int], eos: int = EOS) -> list[int]:
    """Frozen static D3 candidate set: content IDs < EOS, not generation-suppressed, UTF-8-boundary legal."""
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from .core_r2 import token_bytes
    dec = {v: k for k, v in bytes_to_unicode().items()}
    sup = set(int(x) for x in suppress)
    special = set(tokenizer.all_special_ids)
    out = []
    for i in range(int(eos)):
        if i in sup or i in special:
            continue
        if utf8_boundary_legal(token_bytes(tokenizer, i, dec)):
            out.append(i)
    return out


def processed_logsoftmax(raw: np.ndarray, suppress: Sequence[int]) -> np.ndarray:
    """float32 full-vocabulary log-softmax of the generation-processed logits (t >= 1: suppress_tokens only)."""
    import torch
    z = torch.as_tensor(np.asarray(raw, dtype=np.float32)).clone()
    if suppress:
        z[list(suppress)] = -float("inf")
    return torch.log_softmax(z, dim=-1).numpy()


def d3_candidate(raw_b: np.ndarray, raw_null: np.ndarray, legal_mask: np.ndarray, suppress: Sequence[int],
                 baseline_token: int) -> dict:
    """Frozen D3 acoustic-prior candidate at one structural query (t >= 1).

    l_B, l_N: float32 log-softmax of the processed B0 / null-audio logits, upcast to float64.
    A(v) = l_B(v) - max(l_N(v), log 1e-12);  S(v) = l_B(v) + A(v)   (contrast coefficient 1).
    Plausible legal set P = {v legal : l_B(v), l_N(v) finite, l_B(v) >= max_w l_B(w) + log 1e-3}.
    c_AP = argmax_P S, lowest token ID on exact ties. Empty P -> abstain; c_AP == b_t -> exact no-edit.
    """
    lb = processed_logsoftmax(raw_b, suppress).astype(np.float64)
    ln = processed_logsoftmax(raw_null, suppress).astype(np.float64)
    out = {"status": None, "c_AP": None, "baseline_token": int(baseline_token), "plausible_count": 0,
           "lB_max": None, "threshold": None}
    if not (np.isfinite(lb).any() and np.isfinite(ln).any()) or np.isnan(lb).any() or np.isnan(ln).any():
        out["status"] = "nonfinite_branch"
        return out
    lb_max = float(np.max(lb))
    thr = lb_max + D3_LOG_ALPHA
    out["lB_max"], out["threshold"] = lb_max, thr
    legal = np.asarray(legal_mask, dtype=bool) & np.isfinite(lb) & np.isfinite(ln) & (lb >= thr)
    ids = np.flatnonzero(legal)
    out["plausible_count"] = int(ids.size)
    if ids.size == 0:
        out["status"] = "no_plausible_legal_candidate"
        return out
    a = lb[ids] - np.maximum(ln[ids], D3_LOG_NULL_FLOOR)
    s = lb[ids] + a
    best = float(s.max())
    c = int(ids[np.flatnonzero(s == best)].min())
    k = int(np.flatnonzero(ids == c)[0])
    out.update(c_AP=c, S_c=best, A_c=float(a[k]), lB_c=float(lb[c]), lN_c=float(ln[c]),
               null_floor_hit_c=bool(ln[c] < D3_LOG_NULL_FLOOR), ties_at_max=int((s == best).sum()),
               floor_hits_in_P=int((ln[ids] < D3_LOG_NULL_FLOOR).sum()))
    b = int(baseline_token)
    if b < len(lb) and legal[b]:
        kb = int(np.flatnonzero(ids == b)[0])
        out.update(S_b=float(s[kb]), A_b=float(a[kb]), baseline_in_P=True)
    else:
        out["baseline_in_P"] = False
    out["status"] = "baseline_is_candidate" if c == b else "candidate"
    return out


def token_margin_direction(bundle, *, layer: int, cache, encoded, new_tokens: Sequence[int], start: int,
                           c_token: int, b_token: int) -> dict:
    """D3 d_AP: the historical zero-probe scratch-gradient mechanics (``readout.readout_direction``) with the frozen
    objective J = z[c_AP] - z[b_t] on the scratch query's float32 raw logits (c_AP and b_t are never suppressed, so raw
    and processed differences are identical). Pre-step cache deep-copied; one autograd.grad; float64 tangent."""
    import torch
    from .readout import ZeroProbe, cache_fingerprint, clone_scratch, tangent_unit
    from csasr.lss.sites import assert_no_site_hooks
    t0 = time.perf_counter()
    params = list(bundle.model.parameters())
    flags = [p.requires_grad for p in params]
    counters = {"autograd_calls": 0, "scratch_forwards": 0}
    if any(p.grad is not None for p in params):
        raise RuntimeError("model parameter has a gradient before D3 readout")
    fp_before = cache_fingerprint(cache)
    if cache.get_seq_length() != int(start):
        raise RuntimeError("D3 start does not match the pre-step cache length")
    query = int(start) + len(new_tokens) - 1
    out = {"version": VERSION + ":d3_token_margin", "query": query, "direction": None, "status": "invalid",
           "reason": None, "c_token": int(c_token), "b_token": int(b_token)}
    scratch = enc = probe = res = None
    try:
        for p in params:
            p.requires_grad_(False)
        with torch.inference_mode(False), torch.enable_grad():
            scratch, enc = clone_scratch(cache, encoded)
            dev = next(bundle.model.parameters()).device
            ids = torch.tensor([list(new_tokens)], device=dev, dtype=torch.long)
            positions = torch.arange(int(start), int(start) + len(new_tokens), device=dev)
            probe = ZeroProbe(bundle, layer, query)
            with probe:
                res = bundle.model(encoder_outputs=enc, decoder_input_ids=ids, past_key_values=scratch,
                                   use_cache=True, cache_position=positions, output_attentions=False, return_dict=True)
            counters["scratch_forwards"] += 1
            assert_no_site_hooks(bundle)
            z = res.logits[0, -1].float()
            j = z[int(c_token)] - z[int(b_token)]
            out["J"] = float(j.detach())
            out["logits"] = z.detach().cpu()
            if probe.calls != 1 or probe.site is None:
                raise RuntimeError(f"D3 probe fired {probe.calls} times")
            out["site"] = probe.site.float().cpu()
            if not math.isfinite(out["J"]):
                out["reason"] = "nonfinite_objective"
            else:
                (g,) = torch.autograd.grad(j, probe.delta)
                counters["autograd_calls"] += 1
                out["gradient"] = g.detach().float().cpu()
                out.update(tangent_unit(out["gradient"], probe.site))
            probe._zero()
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
        res = scratch = enc = None
        if probe is not None:
            probe.delta = None
        if any(p.grad is not None for p in params):
            raise RuntimeError("model parameter accumulated a gradient during D3 readout")
    if cache_fingerprint(cache) != fp_before:
        raise RuntimeError("persistent cache changed during D3 readout")
    out["counters"] = counters
    out["runtime_sec"] = time.perf_counter() - t0
    return out


# ---- D4 / D0 / random ------------------------------------------------------------------------------------------------

def d4_direction(h_tr, h_tl) -> dict:
    """d_TF = tangent_unit(h_TR - h_TL, h_TR): historical float64 tangent + epsilon normalization + guards."""
    import torch
    from .readout import tangent_unit
    a = torch.as_tensor(np.asarray(h_tr, dtype=np.float32))
    b = torch.as_tensor(np.asarray(h_tl, dtype=np.float32))
    diff = a.double() - b.double()
    r = tangent_unit(diff, a)
    r["raw_norm"] = float(torch.linalg.vector_norm(diff))
    return r


def d0_direction(h_e, h_b) -> dict:
    import torch
    from .core_p1 import direction
    return direction(torch.as_tensor(np.asarray(h_e, dtype=np.float32)), torch.as_tensor(np.asarray(h_b, dtype=np.float32)))


def random_seed(uid: str, t: int, layer: int = LAYER) -> int:
    return int(hashlib.sha256(f"{RANDOM_TAG}|{SEED}|{uid}|{int(t)}|{int(layer)}".encode()).hexdigest()[:16], 16)


def random_direction(uid: str, t: int, layer: int = LAYER) -> np.ndarray:
    """One isotropic PCG64 Gaussian unit vector per UID / t / layer (SRC-CF0-P arithmetic, DIR-SPRINT0 tag)."""
    v = np.random.Generator(np.random.PCG64(random_seed(uid, t, layer))).standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---- D5: concept evidence ------------------------------------------------------------------------------------------

def concept_matrix(concept_table: dict, n_symbols: int) -> tuple[np.ndarray, list[str]]:
    """[V, 8] concept fractions (primary then descriptive); zero rows for non-segmental symbols."""
    classes = list(concept_table["classes"])
    if classes != list(PRIMARY_CONCEPTS) + list(DESCRIPTIVE_CONCEPTS):
        raise ValueError("concept class order differs from the freeze")
    C = np.zeros((n_symbols, len(classes)), dtype=np.float64)
    for r in concept_table["rows"]:
        if r["values"] is not None:
            C[r["id"]] = r["values"]
    return C, classes


def window_weights(start_sample: int, end_sample: int, n_frames: int) -> np.ndarray:
    from .phone_provider import interval_weights
    return interval_weights(int(start_sample), int(end_sample), int(n_frames))


def concept_evidence(log_probs: np.ndarray, weights: np.ndarray, valid: np.ndarray, excluded: np.ndarray,
                     C: np.ndarray) -> dict:
    """Frozen D5 local evidence over one window.

    p = exp(float32 log-posteriors) in float64; s_j = sum_{segmental v} p_jv; emitting e_j = [s_j >= 0.5];
    w'_j = w_j e_j.  E = sum_j w'_j (emitting frame-equivalents).  M = sum_j w'_j s_j.
    q_k = sum_j w'_j sum_{segmental v} p_jv C_vk / M  (fraction of emitting phone mass in concept class k).
    Excluded share = excluded mass / (segmental + excluded mass), both over w'.
    Validity (first failing reason): too_few_window_frames (< 20 positive-weight frames), nonfinite_posteriors,
    low_emitting_weight (E < 2), excluded_mass (share > 0.1), zero_valid_mass. Abstention never imputes q.
    """
    lp = np.asarray(log_probs)
    w = np.asarray(weights, dtype=np.float64)
    out = {"status": None, "q": None, "q_descriptive": None, "frames": int(np.count_nonzero(w)),
           "weight_total": float(w.sum()), "emitting_weight": None, "segmental_mass": None, "excluded_share": None,
           "segmental_mass_all_frames": None}
    if lp.ndim != 2 or w.shape != (lp.shape[0],) or C.shape[0] != lp.shape[1]:
        raise ValueError("shape mismatch in concept evidence")
    if out["frames"] < WINDOW_FRAMES_MIN:
        out["status"] = "too_few_window_frames"
        return out
    if not np.all(np.isfinite(lp)):
        out["status"] = "nonfinite_posteriors"
        return out
    p = np.exp(lp.astype(np.float64))
    s = p[:, valid].sum(axis=1)
    x = p[:, excluded].sum(axis=1)
    e = (s >= EMIT_MIN).astype(np.float64)
    we = w * e
    out["segmental_mass_all_frames"] = float(w @ s)
    out["emitting_weight"] = E = float(we.sum())
    M = float(we @ s)
    X = float(we @ x)
    out["segmental_mass"] = M
    out["excluded_share"] = (X / (M + X)) if (M + X) > 0 else None
    if E < EMITTING_WEIGHT_MIN:
        out["status"] = "low_emitting_weight"
        return out
    if out["excluded_share"] is None or out["excluded_share"] > EXCLUDED_SHARE_MAX:
        out["status"] = "excluded_mass"
        return out
    if not M > 0:
        out["status"] = "zero_valid_mass"
        return out
    pv = p * valid[None, :]
    qall = (we @ pv @ C) / M
    if not np.all(np.isfinite(qall)):
        out["status"] = "nonfinite_evidence"
        return out
    k = len(PRIMARY_CONCEPTS)
    out["q"] = [float(v) for v in qall[:k]]
    out["q_descriptive"] = [float(v) for v in qall[k:]]
    out["status"] = "ok"
    return out


# ---- D5: calibration-bank prototypes -------------------------------------------------------------------------------

def unit_rows(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    n = np.linalg.norm(H, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise ValueError("zero-norm bank state")
    return H / n


def build_prototypes(Q: np.ndarray, H: np.ndarray, dialogues: Sequence[str]) -> dict:
    """Frozen D5 concept prototypes from valid bank queries (rows of Q [N, 6] and states H [N, 1280]).

    For concept k: lo_k = Q_0.2, hi_k = Q_0.8 (numpy linear, pooled); High = {q_k >= hi_k}, Low = {q_k <= lo_k};
    requires hi_k > lo_k. States are unit-normalized (h / ||h||, float64). A dialogue enters concept k iff it has
    >= 5 High and >= 5 Low queries; Delta_d = mean_High,d - mean_Low,d; v_k = normalize(mean_d Delta_d) (dialogue-macro,
    equal dialogue weight). Requires >= 15 entering dialogues and >= 500 High and >= 500 Low queries overall.
    mu_k = dialogue-macro mean of q_k; s_k = pooled population SD (ddof 0) of q_k, required > 1e-6.
    """
    Q = np.asarray(Q, dtype=np.float64)
    U = unit_rows(H)
    dlg = np.asarray(list(dialogues))
    names = sorted(set(dlg.tolist()))
    K = Q.shape[1]
    out = {"concepts": list(PRIMARY_CONCEPTS), "n_valid": int(Q.shape[0]), "per_concept": [], "status": "ok",
           "V": None, "mu": None, "sd": None}
    V, mu, sd = np.zeros((K, U.shape[1])), np.zeros(K), np.zeros(K)
    for k in range(K):
        q = Q[:, k]
        lo, hi = float(np.quantile(q, Q_LO)), float(np.quantile(q, Q_HI))
        rec = {"concept": PRIMARY_CONCEPTS[k], "lo": lo, "hi": hi, "status": None, "n_high": 0, "n_low": 0,
               "dialogues_entering": [], "raw_norm": None}
        mu[k] = float(np.mean([q[dlg == d].mean() for d in names if (dlg == d).any()]))
        sd[k] = float(q.std())
        if not hi > lo:
            rec["status"] = "degenerate_quantiles"
        else:
            high, low = q >= hi, q <= lo
            rec["n_high"], rec["n_low"] = int(high.sum()), int(low.sum())
            deltas = []
            for d in names:
                hm, lm = high & (dlg == d), low & (dlg == d)
                if hm.sum() >= DIALOGUE_SIDE_MIN and lm.sum() >= DIALOGUE_SIDE_MIN:
                    deltas.append(U[hm].mean(axis=0) - U[lm].mean(axis=0))
                    rec["dialogues_entering"].append(d)
            if len(deltas) < DIALOGUES_MIN or rec["n_high"] < SIDE_TOTAL_MIN or rec["n_low"] < SIDE_TOTAL_MIN:
                rec["status"] = "insufficient_construction_coverage"
            elif not sd[k] > SD_MIN:
                rec["status"] = "degenerate_sd"
            else:
                m = np.mean(np.stack(deltas), axis=0)
                n = float(np.linalg.norm(m))
                rec["raw_norm"] = n
                if not n > 0 or not np.isfinite(n):
                    rec["status"] = "zero_prototype"
                else:
                    V[k] = m / n
                    rec["status"] = "ok"
        rec["mu"], rec["sd"] = float(mu[k]), float(sd[k])
        out["per_concept"].append(rec)
    if any(r["status"] != "ok" for r in out["per_concept"]):
        out["status"] = "prototype_construction_failed"
        return out
    out["V"], out["mu"], out["sd"] = V, mu, sd
    out["prototype_cosines"] = (V @ V.T).tolist()
    return out


def d5_direction(q: Sequence[float], mu: np.ndarray, sd: np.ndarray, V: np.ndarray, h) -> dict:
    """d_PHON = tangent_unit(sum_k z_k v_k, h) with z_k = (q_k - mu_k) / s_k (float64); abstain if ||z|| < 1e-6."""
    import torch
    from .readout import tangent_unit
    z = (np.asarray(q, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / np.asarray(sd, dtype=np.float64)
    zn = float(np.linalg.norm(z))
    if not np.all(np.isfinite(z)) or zn < Z_NORM_MIN:
        return {"direction": None, "status": "invalid", "reason": "null_evidence_contrast", "z": z.tolist(), "z_norm": zn}
    u = z @ np.asarray(V, dtype=np.float64)
    r = tangent_unit(torch.from_numpy(u), torch.as_tensor(np.asarray(h, dtype=np.float32)))
    r["z"], r["z_norm"], r["u_norm"] = z.tolist(), zn, float(np.linalg.norm(u))
    return r


def shuffle_key(uid: str, t: int) -> str:
    return hashlib.sha256(f"{SHUFFLE_TAG}|{SEED}|{uid}|{int(t)}".encode()).hexdigest()


def d5_shuffle(uid: str, ts: Sequence[int]) -> dict:
    """Within-utterance derangement of D5-valid query evidence: order by (SHA256(tag|seed|UID|t), t); position i
    receives the evidence of position (i + 1) mod n. n >= 2 has no fixed point; n == 1 is a flagged singleton."""
    order = sorted((int(t) for t in ts), key=lambda t: (shuffle_key(uid, t), t))
    n = len(order)
    donor = {order[i]: order[(i + 1) % n] for i in range(n)} if n else {}
    return {"utterance_id": uid, "n": n, "donor": donor, "singleton": n == 1,
            "fixed_points": sum(1 for r, d in donor.items() if r == d)}


# ---- v_AC (historical R0/S1/SRC-CF0-P construction on new utterances) ---------------------------------------------

def vac_track(waveform: np.ndarray, lid, language_ids: Sequence[int], rc: dict) -> dict:
    """R0 primary region track: frozen window grid, native LID per window, EN/ZH/U rule, sample vote, heard slice."""
    from . import r0_regions as R
    n = int(len(waveform))
    heard = min(n, 480000)
    grid = R.window_grid(n, rc["window_samples"], rc["stride_samples"])
    wins, probs = [], []
    for s, e in grid:
        d = lid(s, e)
        pv = np.array([d[i] for i in language_ids], dtype=np.float32)
        probs.append(pv)
        wins.append({"start": int(s), "end": int(e), **R.classify_window(pv, list(language_ids), waveform[s:e], rc)})
    codes = [w["code"] for w in wins]
    intervals = R.sample_intervals(grid, codes, n, rc["sample_vote_fraction_min"])
    track = R.intervals_to_track(intervals, n)
    track_h = track[:heard]
    th = np.array([(s, e, k) for s, e, k in R.track_to_intervals(track_h)], dtype=np.int64).reshape(-1, 3)
    return {"grid": [[int(a), int(b)] for a, b in grid], "codes": codes, "track_heard_intervals": th,
            "probs": np.stack(probs) if probs else np.zeros((0, len(language_ids)), np.float32), "heard": heard,
            "windows": wins}


def vac_target(heads_q: np.ndarray, track_heard_intervals: np.ndarray, waveform: np.ndarray, heard: int,
               regions_cfg: dict, heard_mass_min: float) -> dict:
    """S1 predicted-English target crop for one query from the cached B0 query's frozen alignment-head attention
    (S1 ``heard_attention`` / ``select_target`` unchanged: association >= 0.10, raw heard mass >= 0.5)."""
    from .s1_evidence import heard_attention, select_target
    att = heard_attention(heads_q, heard, heard_mass_min)
    rec = select_target(att, track_heard_intervals, waveform, heard, {"regions": regions_cfg})
    rec["raw_heard_mass"], rec["mapped"] = att["raw_mass"], att["mapped"]
    return rec


def vac_direction(h_clean, h_mask) -> dict:
    from .src_cf0_pilot import direction
    return direction(np.asarray(h_clean, dtype=np.float32), np.asarray(h_mask, dtype=np.float32))


# ---- targets ---------------------------------------------------------------------------------------------------------

def arm_target(arm: str, e_star: float, g: float) -> float:
    """Desired realized chord: e* for ungated arms; e* * g_old for gated arms (zero gate -> exact zero)."""
    if arm in UNGATED:
        return float(e_star)
    if arm in GATED:
        return 0.0 if float(g) == 0.0 else float(e_star) * float(g)
    raise ValueError(f"unknown arm {arm}")
