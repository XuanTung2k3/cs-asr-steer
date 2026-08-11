"""Mixed Error Rate and the token alignment every other metric builds on.

MER tokenization: Mandarin per character, English per word (see
``csasr.data.normalize.segment_units``). EN-WER and ZH-CER are read off the
same alignment so that all reported numbers are mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .normalization import EN, ZH, normalize_text, segment_units, tag_unit

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


@dataclass(frozen=True)
class Op:
    op: str
    ref_idx: int | None
    hyp_idx: int | None


def align_tokens(ref: Sequence[str], hyp: Sequence[str]) -> list[Op]:
    """Levenshtein alignment with a deterministic backtrace (sub > del > ins)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(Op(MATCH if cost == 0 else SUB, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(Op(DEL, i - 1, None))
            i -= 1
            continue
        ops.append(Op(INS, None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def counts_from_ops(ops: Sequence[Op]) -> dict[str, int]:
    out = {"match": 0, "sub": 0, "del": 0, "ins": 0}
    for o in ops:
        out[o.op] += 1
    return out


def error_rate(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> dict:
    ops = align_tokens(ref_tokens, hyp_tokens)
    c = counts_from_ops(ops)
    n = len(ref_tokens)
    rate = (c["sub"] + c["del"] + c["ins"]) / n if n else float("nan")
    return {**c, "num_ref": n, "num_hyp": len(hyp_tokens), "rate": rate, "ops": ops}


def ref_unit_status(reference_normalized: str, hypothesis: str) -> list[str]:
    """Per reference-unit outcome: "correct" | "sub" | "del".

    Used to restrict direction construction to reference units the frozen
    baseline already recognises correctly (guide section 20).
    """
    ref = [u.surface for u in segment_units(reference_normalized)]
    hyp = [u.surface for u in segment_units(normalize_text(hypothesis))]
    status = ["del"] * len(ref)
    for o in align_tokens(ref, hyp):
        if o.ref_idx is None:
            continue
        status[o.ref_idx] = "correct" if o.op == MATCH else ("sub" if o.op == SUB else "del")
    return status


def mer(reference: str, hypothesis: str) -> dict:
    """Mixed error rate on already-raw text (normalization applied here)."""
    ref = [u.surface for u in segment_units(normalize_text(reference))]
    hyp = [u.surface for u in segment_units(normalize_text(hypothesis))]
    return error_rate(ref, hyp)


def corpus_mer(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """Corpus-level MER plus per-language breakdowns (micro-averaged)."""
    tot = {"sub": 0, "del": 0, "ins": 0, "num_ref": 0}
    lang_tot = {EN: {"err": 0, "n": 0}, ZH: {"err": 0, "n": 0}}
    for r, h in zip(references, hypotheses):
        ref_units = segment_units(normalize_text(r))
        hyp_units = segment_units(normalize_text(h))
        ref = [u.surface for u in ref_units]
        hyp = [u.surface for u in hyp_units]
        ops = align_tokens(ref, hyp)
        c = counts_from_ops(ops)
        tot["sub"] += c["sub"]
        tot["del"] += c["del"]
        tot["ins"] += c["ins"]
        tot["num_ref"] += len(ref)

        ref_tags = [tag_unit(u) for u in ref_units]
        hyp_tags = [tag_unit(u) for u in hyp_units]
        for lang in (EN, ZH):
            lang_tot[lang]["n"] += sum(1 for t in ref_tags if t == lang)
        for o in ops:
            if o.op in (SUB, DEL) and o.ref_idx is not None:
                t = ref_tags[o.ref_idx]
                if t in lang_tot:
                    lang_tot[t]["err"] += 1
            elif o.op == INS and o.hyp_idx is not None:
                t = hyp_tags[o.hyp_idx]
                if t in lang_tot:
                    lang_tot[t]["err"] += 1

    n = tot["num_ref"]
    return {
        "mer": (tot["sub"] + tot["del"] + tot["ins"]) / n if n else float("nan"),
        "substitutions": tot["sub"],
        "deletions": tot["del"],
        "insertions": tot["ins"],
        "num_ref_tokens": n,
        "en_wer": (lang_tot[EN]["err"] / lang_tot[EN]["n"]) if lang_tot[EN]["n"] else float("nan"),
        "zh_cer": (lang_tot[ZH]["err"] / lang_tot[ZH]["n"]) if lang_tot[ZH]["n"] else float("nan"),
        "num_en_ref": lang_tot[EN]["n"],
        "num_zh_ref": lang_tot[ZH]["n"],
    }
