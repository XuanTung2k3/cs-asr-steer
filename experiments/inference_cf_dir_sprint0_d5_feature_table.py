#!/usr/bin/env python
"""DIR-SPRINT0 D5 provider: frozen phone-symbol -> articulatory-feature table (provider engineering only).

Input: the pinned ``vocab.json`` of facebook/wav2vec2-xlsr-53-espeak-cv-ft (revision 2c733782) and panphon 0.22.2's
feature table. Output: ``results/inference_cf/dir_sprint0/d5_provider/feature_table.json``. Run with the isolated
provider environment (``/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5``), which is the only place panphon is
installed. No audio, transcript, reference, label, alignment or model forward is read here.

Every one of the 392 output symbols receives exactly one status:

* ``blank`` / ``special``: CTC blank ``<pad>`` (id 0) and ``<s>``, ``</s>``, ``<unk>``. Never phonetic evidence.
* ``tone_only``: a bare tone mark with no segment (``1``). Never phonetic evidence.
* ``excluded``: a symbol whose phonetic value is ambiguous or not parseable without guessing. The symbol and its
  explicit reason are listed below; its posterior mass is reported separately and never redistributed.
* ``segmental``: normalized to a canonical IPA segment sequence that panphon parses completely. Its feature vector
  is the arithmetic mean over its segments of panphon's 22 segmental features (+1 / 0 / -1). panphon's two tone
  features (``hitone``, ``hireg``) are dropped. Tone is recorded separately and never enters the vector.

The normalization rules are explicit, ordered and applied to the NFD form of each symbol (see ``RULES``). They encode
the espeak-ng conventions this checkpoint was trained on: Mandarin (``cmn``) mnemonics such as ``ts.`` = retroflex
affricate, ``ph`` = aspirated stop and ``i.`` / ``i̪`` = apical vowels; tone digits 1-5 on Mandarin finals, with tone 3
rendered as a trailing ``ɜ``; Kirshenbaum-style ASCII leaks (``tS``, ``dZ``, ``S``, ``N``, ``t[``, ``d[``); ASCII ``:``
for length; and unmarked affricate digraphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_DIR = Path("/mnt/data/tungnx/cs-asr-steer/providers/wav2vec2-xlsr-53-espeak-cv-ft/2c733782da5604684829819a5eb744c193fe9398")
OUT = ROOT / "results/inference_cf/dir_sprint0/d5_provider/feature_table.json"
VERSION = "dir_sprint0_d5_feature_table_v1"
SPECIAL = {"<pad>": "blank", "<s>": "special", "</s>": "special", "<unk>": "special"}
TONE_ONLY = {"1"}
DROPPED_FEATURES = ("hitone", "hireg")

# Symbols whose phonetic value cannot be fixed without guessing (all outside the EN/ZH espeak conventions).
EXCLUDED = {
    "??": "unknown placeholder symbol",
    "u\"": "nonstandard ASCII mark",
    "X": "ambiguous espeak/Kirshenbaum fricative (velar x vs uvular χ)",
    "oe": "ambiguous ASCII digraph (œ vs o+e)", "oe:": "ambiguous ASCII digraph (œ vs o+e)",
    "ee": "ambiguous ASCII digraph (long e vs e+e)",
    "s^": "ambiguous caret modifier", "t^": "ambiguous caret modifier", "d^": "ambiguous caret modifier",
    "ɪ^": "ambiguous caret modifier", "t^ː": "ambiguous caret modifier",
    "a.": "ambiguous dot modifier on a non-cmn vowel", "a.ː": "ambiguous dot modifier on a non-cmn vowel",
    "u.": "ambiguous dot modifier on a non-cmn vowel", "u.ː": "ambiguous dot modifier on a non-cmn vowel",
    "r.": "ambiguous dot modifier on r",
    "ʲ": "modifier letter without a segment",
    "yɛ5ʲ": "tone digit not in final position",
    "ɯᵝ": "unsupported compression mark ᵝ", "ɯᵝɯᵝ": "unsupported compression mark ᵝ",
    "r̝̊": "unsupported ring-above voicelessness on r̝",
    "ää": "nonstandard doubled segment", "ɐɐ": "nonstandard doubled segment", "ɯɯ": "nonstandard doubled segment",
    "e̞e̞": "nonstandard doubled segment", "o̞o̞": "nonstandard doubled segment", "dˤdˤ": "nonstandard doubled segment",
    "cʰcʰ": "nonstandard doubled segment", "nʲʲ": "nonstandard doubled modifier", "dʲʲ": "nonstandard doubled modifier",
    "iːː": "nonstandard doubled length",
}

# Ordered (pattern, replacement, rule name). Applied to the NFD tone-stripped stem.
RULES = [
    (":", "ː", "ascii_colon_length"),
    ("ts.h", "ʈ͡ʂʰ", "cmn_retroflex_affricate_aspirated"),
    ("ts.", "ʈ͡ʂ", "cmn_retroflex_affricate"),
    ("s.", "ʂ", "cmn_retroflex_fricative"),
    ("i.", "ɻ̩", "cmn_apical_retroflex_vowel"),
    (unicodedata.normalize("NFD", "i̪"), "ɹ̩", "cmn_apical_dental_vowel"),
    ("tsh", "t͡sʰ", "cmn_dental_affricate_aspirated"),
    ("tɕh", "t͡ɕʰ", "cmn_alveolopalatal_affricate_aspirated"),
    ("th", "tʰ", "cmn_aspirated_t"), ("kh", "kʰ", "cmn_aspirated_k"), ("ph", "pʰ", "cmn_aspirated_p"),
    ("nɡ", "ŋ", "cmn_velar_nasal_coda"),
    ("ər", "ə˞", "cmn_rhotic_schwa"),
    ("ɚ", "ə˞", "rhotic_schwa_panphon_form"),
    ("ᵻ", "ɪ̈", "espeak_reduced_vowel"),
    ("tS", "t͡ʃ", "kirshenbaum_tS"), ("dZ", "d͡ʒ", "kirshenbaum_dZ"),
    ("S", "ʃ", "kirshenbaum_S"), ("N", "ŋ", "kirshenbaum_N"),
    ("t[", "t̪", "espeak_dental_t"), ("d[", "d̪", "espeak_dental_d"),
]
AFFRICATES = [("tʃ", "t͡ʃ"), ("dʒ", "d͡ʒ"), ("tɕ", "t͡ɕ"), ("dʑ", "d͡ʑ"), ("ts", "t͡s"), ("dz", "d͡z"), ("pf", "p͡f")]


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def split_tone(sym: str, vocab: set[str]) -> tuple[str, int | None, str | None]:
    """Trailing digit 1-5 is a tone. A trailing ``ɜ`` is tone 3 iff the same stem also exists with a tone digit."""
    if len(sym) > 1 and sym[-1] in "12345":
        return sym[:-1], int(sym[-1]), "trailing_digit"
    if len(sym) > 1 and sym.endswith("ɜ"):
        stem = sym[: -len("ɜ")]
        if any(stem + d in vocab for d in "1245"):
            return stem, 3, "trailing_ɜ_with_digit_tone_siblings"
    return sym, None, None


def normalize(stem: str) -> tuple[str, list[str]]:
    s, applied = nfd(stem), []
    for pat, rep, name in RULES:
        pat_n, rep_n = nfd(pat), nfd(rep)
        if pat_n in s:
            s = s.replace(pat_n, rep_n)
            applied.append(name)
    for pair, tied in AFFRICATES:
        new = re.sub("(?<!͡)" + re.escape(nfd(pair)), lambda _m: nfd(tied), s)
        if new != s:
            s = new
            applied.append("affricate_" + pair)
    return s, applied


def build(vocab_path: Path) -> dict:
    import panphon
    ft = panphon.FeatureTable()
    names = [n for n in ft.names if n not in DROPPED_FEATURES]
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("vocab ids are not contiguous")
    syms = set(vocab)
    rows = []
    for sym, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        row = {"id": idx, "symbol": sym, "symbol_nfd": nfd(sym), "status": None, "reason": None, "tone": None,
               "tone_rule": None, "rules": [], "ipa": None, "segments": None, "features": None}
        if sym in SPECIAL:
            row["status"] = SPECIAL[sym]
        elif sym in TONE_ONLY:
            row.update(status="tone_only", tone=int(sym), reason="bare tone digit")
        elif sym in EXCLUDED:
            row.update(status="excluded", reason=EXCLUDED[sym])
        else:
            stem, tone, how = split_tone(sym, syms)
            ipa, applied = normalize(stem)
            segs = ft.ipa_segs(ipa)
            if not segs or "".join(segs) != ipa or not all(ft.seg_known(x) for x in segs):
                row.update(status="excluded", reason=f"panphon cannot parse {ipa!r} completely (segments {segs})")
            else:
                vecs = [dict(zip(ft.names, ft.fts(x).numeric())) for x in segs]
                row.update(status="segmental", tone=tone, tone_rule=how, rules=applied, ipa=ipa, segments=segs,
                           features=[sum(v[n] for v in vecs) / len(vecs) for n in names])
        rows.append(row)
    valid = [r["status"] == "segmental" for r in rows]
    import importlib.metadata as md
    pp = Path(panphon.__file__).parent
    data_files = sorted(p for p in (pp / "data").glob("*") if p.is_file())
    out = {"schema": VERSION, "provider": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
           "provider_revision": PROVIDER_DIR.name, "vocab_sha256": sha(vocab_path), "vocab_size": len(rows),
           "feature_names": names, "dropped_features": list(DROPPED_FEATURES),
           "feature_aggregation": "arithmetic mean over panphon segments of numeric features (+1/0/-1)",
           "panphon_version": md.version("panphon"),
           "panphon_data_sha256": {p.name: sha(p) for p in data_files},
           "rules": [{"pattern": p, "replacement": r, "name": n} for p, r, n in RULES],
           "affricates": [{"pair": p, "tied": t} for p, t in AFFRICATES],
           "tone_rule": "trailing digit 1-5; trailing ɜ = tone 3 iff stem+{1,2,4,5} exists in the vocabulary",
           "excluded_explicit": EXCLUDED, "special": SPECIAL, "tone_only": sorted(TONE_ONLY),
           "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
           "valid_mask": valid, "blank_id": vocab["<pad>"], "rows": rows,
           "inputs": "vocab.json + panphon only; no audio, transcript, reference, label, alignment or model output"}
    out["table_digest"] = "sha256:" + hashlib.sha256(json.dumps(
        {k: out[k] for k in ("feature_names", "valid_mask", "rows")}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return out


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=PROVIDER_DIR / "vocab.json")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    table = build(a.vocab)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(a.out)
    print(json.dumps({"status_counts": table["status_counts"], "table_digest": table["table_digest"]}, ensure_ascii=False))
    for r in table["rows"]:
        if r["status"] == "excluded" and r["symbol"] not in EXCLUDED:
            print("UNPARSED", r["id"], r["symbol"], r["reason"])


if __name__ == "__main__":
    main()
