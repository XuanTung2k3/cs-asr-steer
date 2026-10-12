"""INTERNAL-0A: reference-free local-view inventory and conservative association.

No references, language-region gates, optimization, or interventions. Historical
ABTS primitives are imported read-only for token bytes and attention timing.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from csasr.inference_cf.abts0_k1 import content_units, unit_interval

SR = 16000
HEARD = 480000
BRANCHES = ("2_EN", "2_AUTO", "4_AUTO", "4_ZH")


def window_grid(n: int) -> list[dict]:
    """One-second hop; clipped tail, and clipped centered 4-s context.

    Never append an outcome-dependent window or duplicate an end-aligned window.
    For <=2 s use one view. Otherwise every 1-s start strictly before heard EOS
    is visited, including a final partial view. No artificial audio padding here.
    """
    n = min(int(n), HEARD)
    if n <= 0:
        raise ValueError("empty heard waveform")
    starts = [0] if n <= 2 * SR else range(0, n, SR)
    out = []
    for k, s in enumerate(starts):
        e = min(s + 2 * SR, n)
        center = (s + e) // 2
        out.append({"grid_index": k, "short": [s, e],
                    "long": [max(0, center - 2 * SR), min(n, center + 2 * SR)]})
    return out


def physical_views(grid: list[dict]) -> list[dict]:
    """Deduplicate identical physical crop/condition pairs without dropping grids."""
    items = {}
    for g in grid:
        for branch in BRANCHES:
            bounds = g["short" if branch.startswith("2_") else "long"]
            # Nominal context family retained even if short utterance clips both.
            key = (branch, *bounds)
            items.setdefault(key, {"view_id": f"{branch}:{bounds[0]}:{bounds[1]}",
                                   "branch": branch, "bounds": bounds, "grid_indices": []})
            items[key]["grid_indices"].append(g["grid_index"])
    return list(items.values())


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def normalized_word(s: str) -> str:
    return s.casefold().replace("’", "'")


def associate(interval, units: list[dict], cover_min: float = .5) -> dict:
    """Contiguous Han/Latin B0 block; no onset tolerance or t>=1 gate."""
    out = {"block": None, "baseline_surface": None, "query_t": None,
           "association": "UNLOCATED", "baseline_cover": 0.0}
    if interval is None or interval[1] <= interval[0]:
        return out
    hits = [k for k, u in enumerate(units) if u.get("interval") is not None
            and overlap(interval, u["interval"]) > 0]
    if not hits:
        return out
    if hits != list(range(hits[0], hits[-1] + 1)):
        return dict(out, association="NONCONTIGUOUS")
    covered = []
    for k in hits:
        a, b = units[k]["interval"]
        covered.append([max(a, interval[0]), min(b, interval[1])])
    merged = []
    for a, b in sorted(covered):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    cover = sum(b - a for a, b in merged) / (interval[1] - interval[0])
    out["baseline_cover"] = cover
    if cover < cover_min:
        return dict(out, association="LOW_COVER")
    return dict(out, block=[hits[0], hits[-1]],
                baseline_surface="".join(u["surface"] for u in units[hits[0]:hits[-1] + 1]),
                query_t=units[hits[0]]["tok_first"], association="ASSOCIATED")


def words_from_view(view: dict, token_bytes: list[bytes], timing: dict,
                    baseline_units: list[dict], boundary_samples: int = 3200) -> list[dict]:
    """Every Latin word retained, including capped/boundary/unlocated words."""
    cu = content_units(token_bytes)
    s, e = view["bounds"]
    words = []
    for u in cu["units"]:
        if u.kind != "LATIN":
            continue
        iv = unit_interval(u, timing)
        full = None if iv is None else [s + iv[0], s + iv[1]]
        boundary = full is None or full[0] < s + boundary_samples or full[1] > e - boundary_samples
        p = {"surface": u.surface, "normalized": normalized_word(u.surface),
             "view_id": view["view_id"], "branch": view["branch"], "crop": [s, e],
             "interval": full, "tok_first": u.tok_first, "tok_last": u.tok_last,
             "pieces": view["tokens"][u.tok_first:u.tok_last + 1],
             "boundary_truncation_possible": boundary, "capped": view["terminated"] == "cap",
             "valid_utf8": all(cu["chars_valid"][u.char_start:u.char_end])}
        p.update(associate(full, baseline_units))
        words.append(p)
    return words


def summarize_candidates(words: list[dict], baseline_units: list[dict]) -> dict:
    """Complete-link occurrence groups; 4-ZH contributes a feature, never a gate.

    Selection requires recurrence in a 2-s branch and 4-AUTO, with genuinely
    different physical crop bounds. This is deliberately separate from union
    discovery, which never uses the selection eligibility field.
    """
    groups = []
    for w in sorted(words, key=lambda p: (p["branch"] == "4_ZH", p["normalized"],
                                         p["interval"] or [-1, -1], p["view_id"], p["tok_first"])):
        def compatible(g):
            if g[0]["normalized"] != w["normalized"] or w["interval"] is None:
                return False
            return all(v["interval"] is not None and overlap(v["interval"], w["interval"]) >=
                       .5 * min(v["interval"][1] - v["interval"][0], w["interval"][1] - w["interval"][0])
                       and min(v["interval"][1] - v["interval"][0], w["interval"][1] - w["interval"][0]) > 0
                       for v in g)
        target = next((g for g in groups if compatible(g)), None)
        if target is None:
            groups.append([w])
        else:
            target.append(w)
    candidates = []
    for k, g in enumerate(groups):
        branches = sorted({w["branch"] for w in g})
        # 4-ZH is only a comparator. Its timing/word agreement cannot veto or
        # promote a primary selection. Per-word discovery keeps every member.
        primary = [w for w in g if w["branch"] != "4_ZH"]
        source = primary if primary else g
        blocks = {tuple(w["block"]) for w in source if w["block"] is not None}
        view_ids = [w["view_id"] for w in source]
        ambiguous = len(blocks) != 1 or len(view_ids) != len(set(view_ids))
        good = [w for w in primary if not w["boundary_truncation_possible"] and not w["capped"]
                and w["valid_utf8"] and w["association"] == "ASSOCIATED"]
        short = [w for w in good if w["branch"].startswith("2_")]
        long = [w for w in good if w["branch"] == "4_AUTO"]
        supported = any(a["crop"] != b["crop"] for a in short for b in long)
        block = list(next(iter(blocks))) if len(blocks) == 1 else None
        present = bool(block is not None and any(normalized_word(baseline_units[j]["surface"]) == g[0]["normalized"]
                                                for j in range(block[0], block[1] + 1)))
        ivs = [w["interval"] for w in g if w["interval"] is not None]
        candidate = {"candidate_id": k, "surface": g[0]["surface"], "normalized": g[0]["normalized"],
                     "members": g, "branches": branches, "recurrence": len(set(view_ids)), "block": block,
                     "primary_branch_count": len({w["branch"] for w in primary}),
                     "ZH_comparator_agreement": "4_ZH" in branches,
                     "interval": [min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)] if ivs else None,
                     "cross_context_supported": supported, "ambiguous": ambiguous, "present_in_baseline_block": present,
                     "selection_eligible": supported and not ambiguous and not present}
        candidates.append(candidate)
    # Same word in separate occurrence groups that claim the same B0 block cannot
    # be distinguished by runtime alignment: abstain, do not choose a lucky one.
    counts = defaultdict(int)
    for p in candidates:
        if p["block"] is not None:
            counts[(p["normalized"], tuple(p["block"]))] += 1
    for p in candidates:
        p["repeated_occurrence_ambiguous"] = p["block"] is not None and counts[(p["normalized"], tuple(p["block"]))] > 1
        if p["repeated_occurrence_ambiguous"]:
            p["selection_eligible"] = False
    eligible = [p for p in candidates if p["selection_eligible"]]
    eligible.sort(key=lambda p: (-p["primary_branch_count"], -p["recurrence"], p["interval"][0],
                                 p["normalized"], p["candidate_id"]))
    return {"words": words, "candidates": candidates, "selected": eligible[0]["candidate_id"] if eligible else None}


def shuffle_map(rows: list[dict], seed: int = 240924) -> dict[str, str]:
    """Within-dialogue derangement; choose rotation minimizing total duration mismatch."""
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["dialogue_id"]].append(r)
    out = {}
    for group in by_d.values():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['utterance_id']}".encode()).hexdigest())
        if len(group) < 2:
            out[group[0]["utterance_id"]] = group[0]["utterance_id"]
            continue
        shift = min(range(1, len(group)), key=lambda k: (
            sum(abs(r["heard_samples"] - group[(i + k) % len(group)]["heard_samples"]) for i, r in enumerate(group)), k))
        for i, r in enumerate(group):
            out[r["utterance_id"]] = group[(i + shift) % len(group)]["utterance_id"]
    return out


def decision(metrics: dict, gates: dict, invalid: list[str] | None = None) -> str:
    if invalid:
        return "INTERNAL0A_INVALID"
    if metrics["evaluable_absent"] < gates["evaluable_absent_min"]:
        return "INTERNAL0A_INSUFFICIENT"
    if metrics["new_occurrence_hits"] < gates["new_hits_min"] or metrics["new_hit_dialogues"] < gates["new_dialogues_min"]:
        return "INTERNAL0A_NO_NEW_LEXICAL_DISCOVERY"
    if (metrics["selected"] < gates["selected_min"] or metrics["selected_dialogues"] < gates["selected_dialogues_min"]):
        return "INTERNAL0A_DISCOVERY_ONLY_SELECTION_INSUFFICIENT"
    if (metrics["precision"] < gates["precision_min"] or metrics["selected_genuine"] < gates["genuine_min"] or
        metrics["selected_genuine_dialogues"] < gates["genuine_dialogues_min"] or
        metrics["false_control_utterances"] > gates["false_control_max"] or
        metrics["precision_minus_shuffled"] is None or
        metrics["precision_minus_shuffled"] < gates["precision_advantage_min"]):
        return "INTERNAL0A_DISCOVERY_ONLY_VERIFICATION_UNSUPPORTED"
    return "INTERNAL0A_DISCOVERY_AND_SELECTION_SUPPORTED"
