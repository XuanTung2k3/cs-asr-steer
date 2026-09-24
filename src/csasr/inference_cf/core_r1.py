"""Pure P0-R1 contracts: K=3 short-continuation counterfactual support.

This module is additive. It imports the frozen K=1 primitives from
``csasr.inference_cf.core`` unchanged (so the original P0 manifest's source hash for
``core.py`` stays valid) and adds only the K=3 sequence machinery.

The single scientific change from P0 is K=1 -> K=3 with token-average sequence scoring;
everything else (site, conditions, panel, permutation, geometry) is reused verbatim.
"""
from __future__ import annotations

import math

# Re-export the frozen K=1 helpers so callers have one import surface. These names are
# imported for reuse (panel/manifest/permutation/geometry alignment identical to P0).
from csasr.inference_cf.core import (  # noqa: F401
    canonical, digest, file_hash, atomic_json, cache_key, permutation,
    generated_content, condition_tokens, conditions_identical, aligned_input,
    validated_row,
)

VERSION_R1 = "p0r1_cross_k3_v1"
K = 3


def truncate_continuation(steps, eos_id, k=K):
    """Frozen EOS/truncation rule for a greedy continuation.

    Given up to ``k`` greedily chosen argmax token ids (in order), the candidate is the
    prefix before the first EOS, capped at ``k`` tokens. EOS is never part of a candidate,
    so scored candidates never include a special/EOS token and their length is well defined.
    """
    out = []
    for t in steps[:k]:
        if t == eos_id:
            break
        out.append(int(t))
    return out


def longest_common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def sequence_logprob(step_logprobs):
    """Token-average log probability sC(y) = (1/|y|) * sum_j log p(y_j | ..., cC).

    ``step_logprobs`` is the list of per-step log-probabilities of the chosen candidate
    tokens under one condition, one entry per candidate token. Requires a non-empty
    candidate; averaging over |y| gives a length-normalized score comparable across
    candidates of different length.
    """
    if not step_logprobs:
        raise ValueError("empty candidate has no token-average log prob")
    vals = [float(x) for x in step_logprobs]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("nonfinite step log prob")
    return sum(vals) / len(vals)


def support_k3(scores, y_e, y_m):
    """Common-candidate cross support from the four token-average sequence scores.

    ``scores`` has keys sE_yE, sE_yM, sM_yE, sM_yM (each a token-average log prob).
    A *candidate collision* is the FULL K=3 sequences being identical; then q_cross is
    uninformative (identically 0) and reported null, mirroring the K=1 collision rule.
    The longest-common-prefix length is logged but never changes the score.
    """
    if not all(math.isfinite(float(scores[x])) for x in ("sE_yE", "sE_yM", "sM_yE", "sM_yM")):
        raise ValueError("nonfinite score")
    collision = list(y_e) == list(y_m)
    return {"collision": collision,
            "lcp": longest_common_prefix(list(y_e), list(y_m)),
            "q_own": scores["sE_yE"] - scores["sM_yM"],
            "q_cross": None if collision else .5 * ((scores["sE_yE"] - scores["sE_yM"])
                                                    + (scores["sM_yM"] - scores["sM_yE"]))}
