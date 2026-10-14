"""Text, unit alignment and first-divergence primitives.

Reused (copied verbatim, then extended below the marked line) from the historical repository so that P0 metrics are
directly comparable with ACL-0:

- ``target_text``, ``units``, ``first_divergence``, ``_word_char``:
  /home/tungnx/cs-asr-steer-inf/src/csasr/inference_cf/acl0.py @ cs-asr-steer commit 3467db86
- ``alignment``, ``tokenize``, ``one``, ``category``, ``ratios``, ``compare``:
  /home/tungnx/cs-asr-steer-inf/experiments/ttls_r1_independent_analysis.py @ 3467db86 (last changed 54c2d32f)
- ``loops``: /home/tungnx/cs-asr-steer-inf/experiments/acl0_evaluate.py @ 3467db86
- ``tok_bytes_fn``: /home/tungnx/cs-asr-steer-inf/experiments/acl0.py @ 3467db86

Canonical normalisation and language tagging are imported unchanged from ``csasr.data`` (identical on main db365d8f and
3467db86). Units: one Han character or one Latin word; MER = (S+D+I)/reference units.
"""
from __future__ import annotations

import collections
import re
import unicodedata

import numpy as np

from csasr.data.language_tags import tag_unit
from csasr.data.normalize import (ANY_TAG_RE, MARKER_RE, PUNCT_RE, STAR_RE, WS_RE, is_han, normalize_text,
                                  segment_units)

HAN = "[㐀-䶿一-鿿豈-﫿]"
_HAN_SPACE = re.compile(rf"\s+(?={HAN})|(?<={HAN})\s+")


def units(text: str) -> list[str]:
    return [u.surface for u in segment_units(normalize_text(text))]


def target_text(raw: str) -> str:
    """Training target in Whisper's own output style: the canonical normaliser's marker and punctuation removal, but
    case kept and no spaces next to Han characters. Canonical units are identical to the reference's."""
    t = unicodedata.normalize("NFKC", str(raw))
    for rx in (MARKER_RE, ANY_TAG_RE, STAR_RE, PUNCT_RE):
        t = rx.sub(" ", t)
    t = t.replace("　", " ")
    t = "".join(ch if (is_han(ch) or ch.isalnum() or ch in "'- ") else " " for ch in t)
    t = WS_RE.sub(" ", t).strip()
    return _HAN_SPACE.sub("", t)


def _word_char(c: str) -> bool:
    return bool(c) and c.isascii() and (c.isalnum() or c in "'-")


def first_divergence(ref: list[int], base: list[int], tok_bytes, eos: int, is_filler) -> dict:
    """Classify the first token divergence between the reference target ids and the frozen baseline's ids.

    Classes: IDENTICAL, FORMAT_ONLY, SEGMENTATION_FIRST, FORMAT_FIRST, LEXICAL_FIRST (sub: TOKEN, FILLER, EOS_EARLY,
    EOS_LATE, TRUNCATED), ANOMALY. Only LEXICAL_FIRST/TOKEN is a lexical first-divergence target (reference id ref[k]
    under the shared prefix). No earlier mismatch is skipped."""
    if ref == base:
        return {"cls": "IDENTICAL", "k": None}
    n = min(len(ref), len(base))
    k = next((i for i in range(n) if ref[i] != base[i]), n)
    rb = b"".join(tok_bytes(i) for i in ref if i != eos)
    bb = b"".join(tok_bytes(i) for i in base if i != eos)
    rt, bt = rb.decode("utf-8", "replace"), bb.decode("utf-8", "replace")
    UR, UB = units(rt), units(bt)
    if k == n:
        return {"cls": "FORMAT_ONLY", "k": k} if UR == UB else {"cls": "LEXICAL_FIRST", "k": k, "sub": "TRUNCATED"}
    L = len(b"".join(tok_bytes(i) for i in ref[:k] if i != eos))
    d = next((i for i in range(min(len(rb), len(bb))) if rb[i] != bb[i]), min(len(rb), len(bb)))
    lr, lb = len(tok_bytes(ref[k])) if ref[k] != eos else 0, len(tok_bytes(base[k])) if base[k] != eos else 0
    if UR == UB:
        return {"cls": "FORMAT_ONLY", "k": k}
    if ref[k] != eos and base[k] != eos and d >= L + max(lr, lb):
        return {"cls": "SEGMENTATION_FIRST", "k": k}
    d0 = d
    while d0 > 0 and d0 < len(rb) and (rb[d0] & 0xC0) == 0x80:
        d0 -= 1
    shared = rb[:d0].decode("utf-8", "replace")
    cr, cb = rb[d0:].decode("utf-8", "replace"), bb[d0:].decode("utf-8", "replace")
    ns = len(units(shared))
    if shared and _word_char(shared[-1]) and (_word_char(cr[:1]) or _word_char(cb[:1])):
        ns -= 1
    j = next((i for i in range(min(len(UR), len(UB))) if UR[i] != UB[i]), min(len(UR), len(UB)))
    if j < ns:
        return {"cls": "ANOMALY", "k": k, "j": j, "ns": ns}
    if j > ns:
        return {"cls": "FORMAT_FIRST", "k": k, "j": j}
    if base[k] == eos:
        sub = "EOS_EARLY"
    elif ref[k] == eos:
        sub = "EOS_LATE"
    elif (j < len(UR) and is_filler(UR[j])) or (j < len(UB) and is_filler(UB[j])):
        sub = "FILLER"
    else:
        sub = "TOKEN"
    return {"cls": "LEXICAL_FIRST", "k": k, "j": j, "sub": sub, "ref_unit": UR[j] if j < len(UR) else None,
            "base_unit": UB[j] if j < len(UB) else None,
            # extension (not in the ACL-0 copy): the next units, for omission/insertion labelling only
            "ref_next": UR[j + 1] if j + 1 < len(UR) else None, "base_next": UB[j + 1] if j + 1 < len(UB) else None}


def tok_bytes_fn(tok):
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    dec = {v: k for k, v in bytes_to_unicode().items()}
    special = set(tok.all_special_ids)
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    cache = {}

    def f(i):
        if i not in cache:
            cache[i] = b"" if i in special or i >= eot else bytes(dec[c] for c in tok.convert_ids_to_tokens(i))
        return cache[i]
    return f


def alignment(a, b):
    # Wagner-Fischer with canonical diagonal/deletion/insertion ties.
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j - 1] + (a[i - 1] != b[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    ops = []
    i, j = len(a), len(b)
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def tokenize(s):
    return [u.surface for u in segment_units(normalize_text(s))]


def one(ref, hyp):
    """counts = [EN ref units wrong, EN ref units, S+D+I, ref units, ZH errors, ZH units, EN errors (incl. EN insertions),
    EN units, S, D, I]."""
    a, b = tokenize(ref), tokenize(hyp)
    at, bt = list(map(tag_unit, a)), list(map(tag_unit, b))
    ops = alignment(a, b)
    by = {i: (o, j) for o, i, j in ops if i is not None}
    c = collections.Counter(o for o, i, j in ops)
    en = sum(at[i] == "EN" for o, i, j in ops if o in ("sub", "del")) + sum(bt[j] == "EN" for o, i, j in ops if o == "ins")
    zh = sum(at[i] == "ZH" for o, i, j in ops if o in ("sub", "del")) + sum(bt[j] == "ZH" for o, i, j in ops if o == "ins")
    counts = [sum(at[i] == "EN" and by[i][0] != "match" for i in range(len(a))), at.count("EN"),
              c["sub"] + c["del"] + c["ins"], len(a), zh, at.count("ZH"), en, at.count("EN"), c["sub"], c["del"], c["ins"]]
    return {"counts": counts, "ref": a, "hyp": b, "tags": at, "ops": ops, "by": by}


def category(o, i):
    op, j = o["by"][i]
    a, b = o["ref"], o["hyp"]
    if op == "match":
        return "correct", b[j]
    near = [k for p, q, k in o["ops"] if p == "ins" and any(q2 in (i - 1, i, i + 1) and k2 in (k - 1, k + 1)
                                                         for p2, q2, k2 in o["ops"])]
    if op == "del":
        loc = min(i, len(b) - 1)
        if a[i] in b[max(0, loc - 3):min(len(b), loc + 4)]:
            return "boundary_error", ""
        return ("insertion_near_poi", " ".join(b[k] for k in near)) if near else ("deletion", "")
    if any(map(is_han, b[j])):
        extra = [k for p, q, k in o["ops"] if p == "ins" and abs(k - j) == 1 and any(map(is_han, b[k]))]
        return ("phonetic_transliteration_or_script", b[j] + "".join(b[k] for k in extra)) if extra else \
            ("wrong_language_substitution", b[j])
    return ("same_language_substitution" if tag_unit(b[j]) == "EN" else "other"), b[j]


def ratios(c):
    return {n: float(c[i] / c[j]) if c[j] else None for n, i, j in [("pier", 0, 1), ("mer", 2, 3), ("zh_cer", 4, 5),
                                                                    ("en_wer", 6, 7)]}


def compare(b, m, uid, dlg, eos):
    ev = []
    znew = zfix = outdamage = outcorrect = encorrect = zhcorrect = 0
    cs = "EN" in b["tags"] and "ZH" in b["tags"]
    mzden = mzloss = 0
    for i, (tok, tag) in enumerate(zip(b["ref"], b["tags"])):
        ok = b["by"][i][0] == "match"
        mk = m["by"][i][0] == "match"
        if tag == "ZH":
            zhcorrect += ok
            znew += ok and not mk
            zfix += not ok and mk
            if cs:
                mzden += ok
                mzloss += ok and not mk
        if tag != "EN":
            outcorrect += ok
            outdamage += ok and not mk
        else:
            encorrect += ok
            if ok != mk:
                cat, surf = category(b if mk else m, i)
                kind = "lexical"
                if cat in ("deletion", "boundary_error", "insertion_near_poi"):
                    kind = "deletion_or_boundary"
                elif cat == "same_language_substitution" and (tok in surf and tok != surf):
                    kind = "word_boundary_repair"
                ev.append(dict(id=uid, dialogue=dlg, index=i, reference=tok, before=category(b, i)[1],
                               after=category(m, i)[1], type="correction" if mk else "corruption",
                               canonical_category=cat, audit_kind=kind, eos_recovery=eos))
    return dict(events=ev, new_zh=znew, zh_repairs=zfix, correct_zh=zhcorrect, matrix_den=mzden, matrix_loss=mzloss,
                outside_den=outcorrect, outside_damage=outdamage, correct_en=encorrect)


def loops(units_, nmax=12, k=3):
    best = 0
    for n in range(1, nmax + 1):
        for i in range(len(units_) - k * n + 1):
            r = 1
            while i + (r + 1) * n <= len(units_) and units_[i + r * n:i + (r + 1) * n] == units_[i:i + n]:
                r += 1
            if r >= k:
                best = max(best, r * n)
    return best


# ====================================================================== ReFT extensions (new code)

def utterance_stratum(ref_raw: str, fill: set[str]) -> dict:
    """Reference-unit strata. Substantive English = EN-tagged, non-filler; Mandarin = ZH-tagged, non-filler.
    CS (genuine intra-utterance code-switching): both present. EN_ONLY / ZH_ONLY: one present. OTHER: neither."""
    u = tokenize(ref_raw)
    en = sum(1 for x in u if tag_unit(x) == "EN" and x not in fill)
    zh = sum(1 for x in u if tag_unit(x) == "ZH" and x not in fill)
    s = "CS" if en and zh else "EN_ONLY" if en else "ZH_ONLY" if zh else "OTHER"
    return {"stratum": s, "n_sub_en": en, "n_zh": zh, "n_units": len(u)}


def en_error_kind(o: dict, i: int) -> str:
    """Kind of a wrong English reference unit i (alignment ``o`` from ``one``).

    translation_like : substituted by Han (wrong-language substitution or transliteration/script)
    lexical_substitution : substituted by a different English word
    segmentation_artifact : substituted by an English string containing / contained in the reference word (e.g. a i ~ ai)
    omission : deleted (or deleted with nearby insertions)
    boundary : deleted but the word appears within +-3 hypothesis units (alignment boundary)
    other : substituted by a non-English, non-Han unit (numerals, fillers, other scripts)
    A deletion whose neighbouring reference unit is substituted by a string containing this unit (``a i`` -> ``ai``,
    ``dangerous from`` -> ``dangerousfrom``) is a segmentation_artifact, not an omission."""
    cat, surf = category(o, i)
    if o["by"][i][0] == "del":
        for nb in (i - 1, i + 1):
            op, j = o["by"].get(nb, ("", None))
            if op == "sub" and o["ref"][i] in o["hyp"][j] and o["ref"][nb] in o["hyp"][j]:
                return "segmentation_artifact"
    if cat in ("wrong_language_substitution", "phonetic_transliteration_or_script"):
        return "translation_like"
    if cat in ("deletion", "insertion_near_poi"):
        return "omission"
    if cat == "boundary_error":
        return "boundary"
    if cat == "same_language_substitution":
        ref = o["ref"][i]
        return "segmentation_artifact" if (ref in surf or surf in ref) else "lexical_substitution"
    return "other"


TARGET_EN_KINDS = ("translation_like", "lexical_substitution", "omission", "other")


def script_of(s: str) -> str:
    if not s:
        return "none"
    if any(is_han(c) for c in s):
        return "han"
    if any(c.isascii() and c.isalpha() for c in s):
        return "latin"
    return "other"


def fd_category(fd: dict) -> str:
    """Category of a LEXICAL_FIRST/TOKEN first divergence from its reference and baseline units."""
    r, b = fd.get("ref_unit"), fd.get("base_unit")
    rl = tag_unit(r) if r else "NONE"
    if b is not None and b == fd.get("ref_next"):
        return "en_omission" if rl == "EN" else "zh_omission" if rl == "ZH" else "other_omission"
    if r is not None and r == fd.get("base_next"):
        return "insertion"
    bs = script_of(b or "")
    if rl == "EN":
        return "translation_like" if bs == "han" else "en_lexical_substitution" if bs == "latin" else "en_other"
    if rl == "ZH":
        return "zh_lexical_substitution" if bs == "han" else "zh_to_latin" if bs == "latin" else "zh_other"
    return "other"


def token_unit_map(target: str, ids: list[int], tok_bytes) -> list[list[int]] | None:
    """For each target token (content ids, EOS excluded), the canonical unit indices whose characters it touches.

    The target is ``target_text`` output (no punctuation, single spaces). ``normalize_text`` must preserve its length
    (only lower-casing differs); otherwise None is returned and the utterance yields no token-level rows."""
    norm = normalize_text(target)
    if len(norm) != len(target) or norm != target.lower():
        return None
    us = segment_units(norm)
    char_unit = [-1] * len(norm)
    for k, u in enumerate(us):
        for c in range(u.char_start, u.char_end):
            char_unit[c] = k
    byte_char = []
    for ci, ch in enumerate(target):
        byte_char += [ci] * len(ch.encode("utf-8"))
    out, pos = [], 0
    for i in ids:
        b = tok_bytes(i)
        cs = {byte_char[x] for x in range(pos, min(pos + len(b), len(byte_char)))}
        pos += len(b)
        out.append(sorted({char_unit[c] for c in cs if char_unit[c] >= 0}))
    if pos != len(byte_char):
        return None
    return out


def token_is_lexical(b: bytes) -> bool:
    """A token that can carry a lexical decision: contains an alphanumeric ASCII byte or any non-ASCII byte
    (Han characters and partial UTF-8 sequences). Pure space/punctuation tokens and special tokens are not lexical."""
    if not b:
        return False
    return any(x >= 0x80 or chr(x).isalnum() for x in b)


def tokens_format_equivalent(a: bytes, b: bytes) -> bool:
    """Two token strings that differ only by case/leading space, or where one is a prefix of the other
    (segmentation) -- not a lexical decision."""
    x, y = a.decode("utf-8", "ignore").strip().lower(), b.decode("utf-8", "ignore").strip().lower()
    if not x or not y:
        return False
    return x == y or x.startswith(y) or y.startswith(x)
