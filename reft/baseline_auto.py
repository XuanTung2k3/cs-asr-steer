#!/usr/bin/env python
"""P0 -- operational Whisper Auto baseline (F1) on the ACL-0 dev150 panel.

  decode   -- GPU, behind the reference firewall: F1 = frozen Whisper-large-v3, generate(language=None) (standard
              automatic language identification), transcription task, beam 5. Integrity: model identity, eager attention,
              engine identity (forced-<|zh|> beam-5 reproduces sealed ACL-0 B0), API equivalence (language=None equals
              generate with the detected language), determinism repeats, weights unchanged, audio hashes.
  seal     -- hash the Auto outputs before any reference is opened.
  evaluate -- CPU: references for the 150 IDs are read here for the first time in ReFT. Metrics, error inventory,
              comparison with sealed ACL-0 B0/H1/A1 (not a matched adaptation comparison), P0 verdict.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, cfg, fillers, firewall, jload_gz, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import check_freeze, environment, git_state, now, sha, slurm, weights_hash  # noqa: E402

FORBIDDEN = ("role_", "targets.json", "transcript", "p0/evaluation", "error_inventory", "panel_strata")
SYSTEMS = ("AUTO", "B0", "H1", "A1")


# ====================================================================== model / decode helpers (also used by P1)

def load_model():
    import torch
    import transformers
    from csasr.models.whisper import load_whisper
    C = cfg()
    m = C["model"]
    b = load_whisper({"model": {"id": m["path"], "hub_id": m["hub_id"], "dtype": m["dtype"], "device": "cuda",
                                "attn_implementation": m["attn_implementation"], "local_files_only": True,
                                "freeze_backbone": True}})
    b.model.eval()
    for p in b.model.parameters():
        p.requires_grad_(False)
    impl = collections.Counter()
    for l in b.model.model.encoder.layers:
        impl[l.self_attn.config._attn_implementation] += 1
    for l in b.model.model.decoder.layers:
        impl[l.self_attn.config._attn_implementation] += 1
        impl[l.encoder_attn.config._attn_implementation] += 1
    assert set(impl) == {"eager"} and sum(impl.values()) == 96, impl
    assert b.revision == m["expected_file_revision"], (b.revision, m["expected_file_revision"])
    gen = b.model.generation_config
    env = {**environment(), "attn_implementation": dict(impl), "model_file_revision": b.revision,
           "tokenizer_revision": b.tokenizer_revision, "transformers": transformers.__version__, "torch": torch.__version__,
           "generation_config": gen.to_dict(), "dtype": str(b.dtype), "d_model": b.d_model,
           "n_decoder_layers": b.num_decoder_layers}
    return b, env


def special_ids(b) -> dict:
    gen = b.model.generation_config
    tok = b.processor.tokenizer
    return {"sot": gen.decoder_start_token_id, "transcribe": gen.task_to_id["transcribe"],
            "translate": gen.task_to_id["translate"], "notimestamps": tok.convert_tokens_to_ids("<|notimestamps|>"),
            "eos": tok.eos_token_id, "lang_to_id": dict(gen.lang_to_id)}


def language_scores(b, feats) -> list:
    """Independent re-computation of Whisper's language decision: log-softmax over the language tokens at <|sot|>."""
    import torch
    S = special_ids(b)
    ids = sorted(S["lang_to_id"].values())
    inv = {v: k for k, v in S["lang_to_id"].items()}
    with torch.inference_mode():
        dec = torch.tensor([[S["sot"]]], device=b.device)
        z = b.model(input_features=feats[:, :, :3000], decoder_input_ids=dec, use_cache=False).logits[0, -1].float()
    lp = z[ids].log_softmax(-1)
    order = lp.argsort(descending=True)[:3].tolist()
    return [(inv[ids[i]], float(lp[i])) for i in order]


def generate(b, path, language, beams):
    import torch
    from csasr.models.whisper import batch_model_inputs
    D = cfg()["decode"]
    S = special_ids(b)
    inp = batch_model_inputs(b, [path])
    kw = dict(task=D["task"], language=language, do_sample=D["do_sample"], temperature=D["temperature"],
              max_new_tokens=D["max_new_tokens"], condition_on_prev_tokens=D["condition_on_prev_tokens"], num_beams=beams)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = b.model.generate(**inp, **kw, return_dict_in_generate=True)
    torch.cuda.synchronize()
    sec = time.time() - t0
    seq = out.sequences[0].tolist()
    assert seq[0] == S["sot"] and seq[2] == S["transcribe"] and seq[3] == S["notimestamps"], seq[:4]
    assert seq[1] in S["lang_to_id"].values(), seq[:4]
    prompt, c = seq[:4], seq[4:]
    term = S["eos"] in c
    if term:
        c = c[:c.index(S["eos"]) + 1]
    text = b.processor.batch_decode([seq], skip_special_tokens=True)[0].strip()
    inv = {v: k for k, v in S["lang_to_id"].items()}
    return {"prompt": prompt, "language": inv[seq[1]], "tokens": c, "text": text, "terminated": term,
            "n_content_tokens": len(c), "seconds": sec}, inp


# ====================================================================== decode (GPU)

def run_decode():
    import torch
    C = cfg()
    fz = check_freeze("p0")
    audit = read_json(art("p0/data_audit.json"))
    assert audit["valid"], "P0 data audit invalid"
    rows = read_json(art("p0/panel.json"))["rows"]
    A = Path(C["data"]["acl0_dir"])
    b0 = {x["id"]: x for x in jload_gz(A / "e0/dev_beam5.json.gz")}  # sealed model outputs, not references
    firewall(FORBIDDEN)
    b, env = load_model()
    w0 = weights_hash(b.model)
    D = C["decode"]
    nb = D["num_beams"]
    t_all = time.time()
    torch.cuda.reset_peak_memory_stats()
    # 1. engine identity: forced <|zh|> with this engine must reproduce sealed ACL-0 B0 beam-5
    ident = []
    for r in rows[:D["engine_identity_rows"]]:
        rec, _ = generate(b, r["audio_path"], "zh", nb)
        ident.append({"id": r["utterance_id"], "equal": rec["tokens"] == b0[r["utterance_id"]]["tokens"]})
    # 2. F1 Auto beam-5 on all rows
    out, audio_ok = [], 0
    for r in rows:
        audio_ok += sha(r["audio_path"]).removeprefix("sha256:") == r["audio_sha256"].removeprefix("sha256:")
        rec, inp = generate(b, r["audio_path"], None, nb)
        top = language_scores(b, inp["input_features"])
        rec.update(id=r["utterance_id"], language_top3=top, detected_equals_own_argmax=top[0][0] == rec["language"],
                   num_beams=nb, cap_hit=not rec["terminated"])
        out.append(rec)
    # 3. API equivalence and determinism
    api, rep = [], []
    for r, o in list(zip(rows, out))[:D["api_equivalence_rows"]]:
        e, _ = generate(b, r["audio_path"], o["language"].strip("<|>"), nb)
        api.append({"id": r["utterance_id"], "equal": e["tokens"] == o["tokens"] and e["prompt"] == o["prompt"]})
    for r, o in list(zip(rows, out))[:D["determinism_repeats"]]:
        e, _ = generate(b, r["audio_path"], None, nb)
        rep.append({"id": r["utterance_id"], "equal": e["tokens"] == o["tokens"]})
    w1 = weights_hash(b.model)
    write_jsonl(art("p0/auto_outputs.jsonl"), out)
    zh_id = special_ids(b)["lang_to_id"]["<|zh|>"]
    man = {"schema": "reft_p0_decode_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(),
           "decode": {**D, "language": None, "api": "WhisperForConditionalGeneration.generate(language=None, task='transcribe', num_beams=5)"},
           "weights_before": w0, "weights_after": w1, "weights_unchanged": w0 == w1,
           "weights_equal_acl0": w0 == C["model"]["expected_weights_hash"],
           "requires_grad_params": sum(p.requires_grad for p in b.model.parameters()),
           "engine_identity_forced_zh_vs_acl0_b0": ident, "api_equivalence": api, "determinism": rep,
           "audio_sha_ok": audio_ok, "n_rows": len(out),
           "detected_equals_own_argmax": sum(o["detected_equals_own_argmax"] for o in out),
           "prompt_zh_rows": sum(o["prompt"][1] == zh_id for o in out),
           "language_counts": dict(collections.Counter(o["language"] for o in out)),
           "terminated": sum(o["terminated"] for o in out), "cap_hit": sum(o["cap_hit"] for o in out),
           "mean_seconds": sum(o["seconds"] for o in out) / len(out), "wall_seconds": time.time() - t_all,
           "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())}
    write_json(art("p0/manifest.json"), man)
    print(json.dumps({k: v for k, v in man.items() if k not in ("env", "decode")}, indent=1, default=str))


def run_seal():
    files = ["p0/auto_outputs.jsonl", "p0/manifest.json", "p0/panel.json", "p0/data_audit.json"]
    write_json(art("p0/output_seal.json"), {"schema": "reft_p0_seal_v1", "utc": now(),
                                            "files": {f: sha(art(f)) for f in files}})
    print("sealed", files)


# ====================================================================== evaluate (CPU)

def dev_refs(ids):
    import pyarrow.parquet as pq
    C = cfg()
    t = pq.read_table(Path(C["data"]["role_dir"]) / f"role_{C['data']['dev_role']}.parquet",
                      columns=["utterance_id", "transcript_raw"], filters=[("utterance_id", "in", ids)])
    return {r["utterance_id"]: r["transcript_raw"] for r in t.to_pylist()}


def share_stats(counter: dict) -> dict:
    v = sorted(counter.values(), reverse=True)
    tot = sum(v)
    return {"total": tot, "n_units_with_errors": sum(1 for x in v if x), "top1_share": v[0] / tot if tot else None,
            "top3_share": sum(v[:3]) / tot if tot else None, "sorted_counts": v}


def run_evaluate():
    import numpy as np
    from reft.text import en_error_kind, loops, one, ratios, utterance_stratum
    C = cfg()
    seal = read_json(art("p0/output_seal.json"))
    assert all(sha(art(f)) == h for f, h in seal["files"].items()), "sealed P0 output changed"
    man = read_json(art("p0/manifest.json"))
    audit = read_json(art("p0/data_audit.json"))
    rows = read_json(art("p0/panel.json"))["rows"]
    ids = [r["utterance_id"] for r in rows]
    meta = {r["utterance_id"]: r for r in rows}
    A = Path(C["data"]["acl0_dir"])
    ep = audit["acl0_selected_epochs"]
    dec = {"AUTO": {o["id"]: o for o in read_jsonl(art("p0/auto_outputs.jsonl"))},
           "B0": {x["id"]: x for x in jload_gz(A / "e0/dev_beam5.json.gz")}}
    for a in ("H1", "A1"):
        dec[a] = {x["id"]: x for x in jload_gz(A / f"{a}/epoch{ep[a]}/dev.json.gz")["beam5"]}
    refs = dev_refs(ids)
    assert len(refs) == len(ids)
    FILL = fillers()
    G = C["p0_gate"]
    LOOP = C["evaluation"]["loop_units"]
    sub = lambda o, i: o["tags"][i] == "EN" and o["ref"][i] not in FILL  # noqa: E731
    zhu = lambda o, i: o["tags"][i] == "ZH" and o["ref"][i] not in FILL  # noqa: E731

    per = []
    for i in ids:
        ref = refs[i]
        st = utterance_stratum(ref, FILL)
        O = {s: one(ref, dec[s][i]["text"]) for s in SYSTEMS}
        r = {"id": i, "dialogue": meta[i]["dialogue_id"], **st, "language": dec["AUTO"][i]["language"],
             "language_top3": dec["AUTO"][i]["language_top3"], "counts": {s: O[s]["counts"] for s in SYSTEMS},
             "terminated": {s: bool(dec[s][i]["terminated"]) for s in SYSTEMS},
             "loop_units": {s: loops(O[s]["hyp"]) for s in SYSTEMS},
             "sub_en_ok": {s: [O[s]["by"][k][0] == "match" for k in range(len(O[s]["ref"])) if sub(O[s], k)] for s in SYSTEMS},
             "zh_ok": {s: [O[s]["by"][k][0] == "match" for k in range(len(O[s]["ref"])) if zhu(O[s], k)] for s in SYSTEMS},
             "texts": {s: dec[s][i]["text"] for s in SYSTEMS}}
        r["derailed"] = {s: (not r["terminated"][s]) or r["loop_units"][s] >= LOOP for s in SYSTEMS}
        o = O["AUTO"]
        r["auto_en_errors"] = [{"unit": o["ref"][k], "kind": en_error_kind(o, k)} for k in range(len(o["ref"]))
                               if sub(o, k) and o["by"][k][0] != "match"]
        r["auto_zh_errors"] = sum(1 for k in range(len(o["ref"])) if zhu(o, k) and o["by"][k][0] != "match")
        r["auto_zh_to_latin"] = sum(1 for k in range(len(o["ref"])) if zhu(o, k) and o["by"][k][0] == "sub"
                                    and o["hyp"][o["by"][k][1]].isascii())
        per.append(r)

    def agg(rs):
        out = {"n_utterances": len(rs), "n_dialogues": len({r["dialogue"] for r in rs}),
               "substantive_en_units": sum(r["n_sub_en"] for r in rs), "zh_units_non_filler": sum(r["n_zh"] for r in rs)}
        for s in SYSTEMS:
            c = np.array([r["counts"][s] for r in rs]).sum(0) if rs else np.zeros(11, dtype=int)
            out[s] = {**ratios(c), "mixed_errors": int(c[2]), "ref_units": int(c[3]), "en_units": int(c[1]),
                      "zh_units": int(c[5]), "S": int(c[8]), "D": int(c[9]), "I": int(c[10]),
                      "substantive_en_errors": sum(len(r["sub_en_ok"][s]) - sum(r["sub_en_ok"][s]) for r in rs),
                      "zh_non_filler_errors": sum(len(r["zh_ok"][s]) - sum(r["zh_ok"][s]) for r in rs),
                      "not_terminated": sum(not r["terminated"][s] for r in rs),
                      "loop_utterances": sum(r["loop_units"][s] >= LOOP for r in rs),
                      "derailed_utterances": sum(r["derailed"][s] for r in rs)}
        kinds = collections.Counter(e["kind"] for r in rs for e in r["auto_en_errors"])
        out["AUTO"]["en_error_kinds"] = dict(sorted(kinds.items()))
        out["AUTO"]["zh_to_latin_substitutions"] = sum(r["auto_zh_to_latin"] for r in rs)
        out["AUTO"]["languages"] = dict(collections.Counter(r["language"] for r in rs))
        for s in ("B0", "H1", "A1"):
            out[f"AUTO_vs_{s}"] = {"auto_better": sum(r["counts"]["AUTO"][2] < r["counts"][s][2] for r in rs),
                                   "auto_worse": sum(r["counts"]["AUTO"][2] > r["counts"][s][2] for r in rs),
                                   "tie": sum(r["counts"]["AUTO"][2] == r["counts"][s][2] for r in rs),
                                   "new_zh_errors_in_auto": sum(sum(a and not b for a, b in zip(r["zh_ok"][s], r["zh_ok"]["AUTO"])) for r in rs),
                                   "zh_repairs_in_auto": sum(sum(b and not a for a, b in zip(r["zh_ok"][s], r["zh_ok"]["AUTO"])) for r in rs)}
        cont = collections.Counter()
        for r in rs:
            for k in range(len(r["sub_en_ok"]["AUTO"])):
                cont["".join("1" if r["sub_en_ok"][s][k] else "0" for s in ("B0", "H1", "A1", "AUTO"))] += 1
        out["en_unit_contingency_B0_H1_A1_AUTO"] = dict(sorted(cont.items()))
        out["h1_corrected_vs_b0_auto_already_correct"] = sum(v for k, v in cont.items() if k[0] == "0" and k[1] == "1" and k[3] == "1")
        out["h1_corrected_vs_b0_auto_wrong"] = sum(v for k, v in cont.items() if k[0] == "0" and k[1] == "1" and k[3] == "0")
        out["a1_corrected_vs_b0_auto_already_correct"] = sum(v for k, v in cont.items() if k[0] == "0" and k[2] == "1" and k[3] == "1")
        out["auto_wrong_h1_correct"] = sum(v for k, v in cont.items() if k[1] == "1" and k[3] == "0")
        out["auto_correct_h1_wrong"] = sum(v for k, v in cont.items() if k[1] == "0" and k[3] == "1")
        return out

    strata = ("CS", "EN_ONLY", "ZH_ONLY", "OTHER")
    report = {"overall": agg(per), **{s: agg([r for r in per if r["stratum"] == s]) for s in strata}}

    # ---- P0 target errors: substantive EN errors of target kinds in genuine-CS utterances, non-derailed Auto outputs
    cs = [r for r in per if r["stratum"] == "CS"]
    keep = [r for r in cs if not (G["exclude_derailed_utterances"] and r["derailed"]["AUTO"])]
    t_utt = {r["id"]: sum(e["kind"] in G["target_kinds"] for e in r["auto_en_errors"]) for r in keep}
    t_dlg = collections.Counter()
    for r in keep:
        t_dlg[r["dialogue"]] += t_utt[r["id"]]
    strict_utt = {r["id"]: sum(e["kind"] in ("translation_like", "lexical_substitution") for e in r["auto_en_errors"]) for r in keep}
    strict_dlg = collections.Counter()
    for r in keep:
        strict_dlg[r["dialogue"]] += strict_utt[r["id"]]
    n_target = sum(t_utt.values())
    dl_with = sum(1 for v in t_dlg.values() if v > 0)
    top_share = max(t_utt.values()) / n_target if n_target else None
    en_only_target = sum(sum(e["kind"] in G["target_kinds"] for e in r["auto_en_errors"]) for r in per
                         if r["stratum"] == "EN_ONLY" and not r["derailed"]["AUTO"])
    opp = {"cs_utterances": len(cs), "cs_utterances_derailed_excluded": len(cs) - len(keep),
           "target_errors": n_target, "dialogues_with_target_errors": dl_with,
           "utterances_with_target_errors": sum(1 for v in t_utt.values() if v),
           "by_kind": dict(collections.Counter(e["kind"] for r in keep for e in r["auto_en_errors"] if e["kind"] in G["target_kinds"])),
           "excluded_kinds": dict(collections.Counter(e["kind"] for r in keep for e in r["auto_en_errors"] if e["kind"] in G["excluded_kinds"])),
           "per_dialogue": dict(sorted(t_dlg.items())), "utterance_concentration": share_stats(t_utt),
           "dialogue_concentration": share_stats(t_dlg), "top_utterance_share": top_share,
           "strict_substitutions_only": {"errors": sum(strict_utt.values()), "dialogues": sum(1 for v in strict_dlg.values() if v),
                                         "per_dialogue": dict(sorted(strict_dlg.items()))},
           "cs_mandarin_errors_auto": sum(r["auto_zh_errors"] for r in keep),
           "en_only_target_errors_for_reference": en_only_target}

    # ---- dialogue-block bootstrap for Auto vs B0/H1/A1
    rng = np.random.default_rng(C["evaluation"]["bootstrap"]["seed"])
    dl = sorted({r["dialogue"] for r in per})
    pd = {d: {s: np.array([r["counts"][s] for r in per if r["dialogue"] == d]).sum(0) for s in SYSTEMS} for d in dl}
    acc = collections.defaultdict(list)
    for _ in range(C["evaluation"]["bootstrap"]["reps"]):
        pick = rng.choice(dl, len(dl))
        tot = {s: sum(pd[d][s] for d in pick) for s in SYSTEMS}
        for s in ("B0", "H1", "A1"):
            acc[s].append(tot["AUTO"][2] / tot["AUTO"][3] - tot[s][2] / tot[s][3])
    boot = {f"AUTO-{s}": {"mer_point": report["overall"]["AUTO"]["mer"] - report["overall"][s]["mer"],
                          "ci95": [float(np.quantile(acc[s], q)) for q in (0.025, 0.975)]} for s in ("B0", "H1", "A1")}

    # ---- integrity and verdict
    integ = {"data_audit_valid": audit["valid"], "seal_ok": True, "weights_unchanged": man["weights_unchanged"],
             "weights_equal_acl0": man["weights_equal_acl0"], "requires_grad_params_zero": man["requires_grad_params"] == 0,
             "engine_identity": all(x["equal"] for x in man["engine_identity_forced_zh_vs_acl0_b0"]),
             "api_equivalence": all(x["equal"] for x in man["api_equivalence"]),
             "determinism": all(x["equal"] for x in man["determinism"]),
             "audio_sha_ok": man["audio_sha_ok"] == len(ids), "detected_equals_own_argmax": man["detected_equals_own_argmax"] == len(ids),
             "num_beams_5": man["decode"]["num_beams"] == 5, "auto_language_not_forced": man["decode"]["language"] is None}
    integrity = all(integ.values())
    gate = {"target_errors_min": n_target >= G["min_target_errors"], "dialogues_min": dl_with >= G["min_dialogues"],
            "not_single_utterance_dominated": top_share is not None and top_share <= G["max_single_utterance_share"]}
    verdict = "P0_BLOCKED" if not integrity else "P0_PASS_RELEVANT_OPPORTUNITIES" if all(gate.values()) else "P0_INSUFFICIENT_OPPORTUNITIES"
    ev = {"schema": "reft_p0_evaluation_v1", "utc": now(), "git": git_state(), "verdict": verdict, "integrity": integ,
          "gate": gate, "gate_config": G, "opportunities": opp, "metrics": report, "bootstrap": boot,
          "denominators": {s: report[s]["n_utterances"] for s in strata},
          "note": "H1/A1 are sealed ACL-0 editors trained for the forced-<|zh|> prompt; comparison is descriptive, not a matched adaptation comparison"}
    write_json(art("p0/evaluation.json"), ev)
    write_json(art("p0/error_inventory.json"), {"schema": "reft_p0_error_inventory_v1", "rows": per})
    print(json.dumps({"verdict": verdict, "integrity": integ, "gate": gate,
                      "opportunities": {k: v for k, v in opp.items() if k not in ("per_dialogue",)}}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decode", "seal", "evaluate"])
    a = ap.parse_args(argv)
    {"decode": run_decode, "seal": run_seal, "evaluate": run_evaluate}[a.cmd]()


if __name__ == "__main__":
    main()
