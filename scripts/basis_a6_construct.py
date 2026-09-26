#!/usr/bin/env python3
"""Model-resident BASIS-A6 fixed-direction construction.

This runner is deliberately construction-only: it reads the frozen CS
D-construct population and the frozen ASCEND train manifest, never validation
or test, keeps one model resident, and emits one provenance-bound direction per
source/model/site/layer/method.  The accepted A4/A5 CS vectors are not
overwritten; the CS pass is used only for the missing Conditioning-CS entries.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
OUT = REPO / "results/basis_a6_expanded/fixed"
WHISPER_ENC = tuple(range(32)); WHISPER_DEC = tuple(range(32))
QWEN_ENC = tuple(range(24)); QWEN_DEC = tuple(range(28))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _hash_array(v: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()


class Moments:
    def __init__(self, dim: int):
        self.n = 0; self.sum = np.zeros(dim, dtype=np.float64); self.second = np.zeros((dim, dim), dtype=np.float64)
    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        if len(x): self.n += len(x); self.sum += x.sum(0); self.second += x.T @ x


class DeltaMoments:
    def __init__(self, dim: int):
        self.n = 0; self.sum = np.zeros(dim, dtype=np.float64)
    def add(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        if len(x): self.n += len(x); self.sum += x.sum(0)


def _decode_bytes(value):
    import soundfile as sf
    if isinstance(value, (bytes, bytearray)):
        audio, sr = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
    else:
        audio, sr = sf.read(str(value), dtype="float32", always_2d=False)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _wav_path(value, directory: Path, key: str) -> str:
    if isinstance(value, (str, Path)) and Path(value).is_file(): return str(value)
    audio, sr = _decode_bytes(value)
    path = directory / f"{key}.wav"
    import soundfile as sf
    sf.write(path, audio, sr)
    return str(path)


def _unit_token_groups(tokenizer, norm, units, ref_ids):
    from csasr.data.alignment import token_char_offsets
    from csasr.data.language_tags import tag_units
    tags = tag_units(units)
    cursor = 0; spans = []
    for unit in units:
        start = norm.find(unit.surface, cursor)
        start = cursor if start < 0 else start
        spans.append((start, start + len(unit.surface), tags[len(spans)]))
        cursor = start + len(unit.surface)
    offsets = token_char_offsets(tokenizer, ref_ids)
    groups = {"EN": [], "ZH": [], "ALL": []}
    for i, (lo, hi) in enumerate(offsets[:len(ref_ids)]):
        owners = [kind for a, b, kind in spans if b > lo and a < hi]
        if not owners: continue
        groups["ALL"].append(i)
        if "EN" in owners: groups["EN"].append(i)
        if "ZH" in owners: groups["ZH"].append(i)
    return {k: sorted(set(v)) for k, v in groups.items()}


def _save_source(model, source, dims, moments, deltas, layers, *, cs_only=False, model_revision=None):
    from csasr.basis_a6.directions import normalize
    from csasr.experiments.basis_a5_unique_shared import construct_unique_shared
    methods = {}
    from experiments.basis_a5_unique_shared import _a4_row
    for side, side_layers in layers.items():
        for layer in side_layers:
            key = (side, int(layer)); ma, mb = moments["A"][key], moments["B"][key]
            vectors = {}
            if not cs_only:
                if ma.n == 0 or mb.n == 0: raise RuntimeError(f"empty A/B: {source} {model} {side} L{layer}")
                vectors["raw"] = normalize(ma.sum / ma.n - mb.sum / mb.n, name="raw")
                rank = min(32, ma.n, mb.n, len(ma.sum))
                if rank >= 2:
                    built = construct_unique_shared(ma.second, ma.n, ma.sum, mb.second, mb.n, mb.sum, rank=rank)
                    vectors.update({"add_unique": normalize(built.v_unique), "minus_shared": normalize(-built.v_shared),
                                    "unique_minus_shared": normalize(built.unique_minus_shared)})
            if side == "decoder":
                if not cs_only and deltas["all"][key].n:
                    vectors["conditioning_all"] = normalize(deltas["all"][key].sum / deltas["all"][key].n)
                if deltas["cs"][key].n:
                    vectors["conditioning_cs"] = normalize(deltas["cs"][key].sum / deltas["cs"][key].n)
                elif cs_only:
                    raise RuntimeError(f"empty Conditioning-CS: {model} L{layer}")
            for method, vector in vectors.items():
                path = OUT / source / "directions" / model / side / f"L{layer:02d}" / f"{method}.npy"
                path.parent.mkdir(parents=True, exist_ok=True); np.save(path, vector)
                methods[f"{model}/{side}/L{layer:02d}/{method}"] = {
                    "source": source, "model": model, "side": side, "layer": int(layer), "method": method,
                    "n_A": ma.n, "n_B": mb.n,
                    "n_conditioning": deltas["all"].get((side, int(layer)), DeltaMoments(1)).n,
                    "n_conditioning_cs": deltas["cs"].get((side, int(layer)), DeltaMoments(1)).n,
                    "norm": float(np.linalg.norm(vector)), "direction_hash": _hash_array(vector),
                    "site_hash": _a4_row(model, "cs_dialogue", "Raw", side, int(layer)).get("site_hash"),
                    "model_revision": model_revision, "construction_manifest": (
                        "CS-D-construct" if source == "cs_dialogue" else "ASCEND_CONSTRUCT_MANIFEST.json")}
    _write(OUT / source / "A6_FIXED_DIRECTION_MANIFEST.json", {
        "schema_version": "basis_a6_fixed_direction_manifest_v1", "status": "PASS", "source": source,
        "model": model, "cs_only": cs_only, "rows": methods, "count": len(methods)})
    return methods


def _whisper_pass(bundle, rows, spans_by_uid, source, *, need_ab, cond_cs_only=False, tmpdir=None):
    import torch
    from csasr.data.alignment import build_prefix
    from csasr.data.normalize import normalize_and_segment
    from csasr.lss.encoder_sites import EncoderPostSelfAttnRecorder
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.generation import teacher_forced_forward
    from csasr.models.whisper import batch_model_inputs
    from csasr.experiments.v2r3_directions import _frame_range
    from csasr.basis_a6.directions import normalize
    enc, dec = WHISPER_ENC, WHISPER_DEC
    dim = bundle.d_model
    moments = {"A": {}, "B": {}}
    deltas = {"all": {}, "cs": {}}
    for side, ls in (("encoder", enc), ("decoder", dec)):
        for l in ls:
            moments["A"][(side,l)] = Moments(dim); moments["B"][(side,l)] = Moments(dim)
            deltas["all"][(side,l)] = DeltaMoments(dim); deltas["cs"][(side,l)] = DeltaMoments(dim)
    p_en, p_zh = build_prefix(bundle.processor, language="en"), build_prefix(bundle.processor, language="zh")
    eot = bundle.processor.tokenizer.eos_token_id
    for number, row in enumerate(rows, 1):
        uid = str(row["utterance_id"]); norm, units = normalize_and_segment(row["transcript_raw"])
        text_ids = bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220]
        if not text_ids: continue
        audio_path = _wav_path(row["audio_path"] if "audio_path" in row else row.audio, tmpdir, uid.replace("/", "_"))
        if need_ab:
            features = batch_model_inputs(bundle, [audio_path])["input_features"]
            with EncoderPostSelfAttnRecorder(bundle, list(enc)) as rec, torch.inference_mode(): bundle.model.model.encoder(features)
            spans = spans_by_uid[uid]
            for l in enc:
                state = rec.states[l][0].float().cpu().numpy(); valid = bundle.valid_frames(float(row["duration_sec"]))
                for lang in ("EN", "ZH"):
                    vals=[]
                    for sp in spans:
                        if sp["language_tag"] != lang: continue
                        lo, hi = _frame_range(sp["start_sec"], sp["end_sec"], bundle.encoder_step_sec, valid)
                        if hi > lo: vals.append(state[lo:hi])
                    for x in vals: (moments["A"] if lang == "EN" else moments["B"])[("encoder",l)].add(x)
        seqs = {"en": list(p_en)+list(text_ids)+[eot], "zh": list(p_zh)+list(text_ids)+[eot]}
        groups = _unit_token_groups(bundle.processor.tokenizer, norm, units, text_ids)
        for condition, seq in seqs.items():
            with DecoderPostCrossAttnRecorder(bundle, list(dec)) as rec, torch.inference_mode(): teacher_forced_forward(bundle, [audio_path], [seq])
            states = {l: rec.states[l][0].float().cpu().numpy() for l in dec}
            pos_all = [len(p_zh)+i-1 for i in groups["ALL"] if len(p_zh)+i-1 >= 0]
            pos_cs = [len(p_zh)+i-1 for i in groups["EN"] if len(p_zh)+i-1 >= 0]
            if condition == "en": st_en = states
            else: st_zh = states
            if need_ab:
                positions = groups["EN"], groups["ZH"]
                for l in dec:
                    apos=[len(p_zh)+i-1 for i in groups["EN"] if len(p_zh)+i-1 >= 0]
                    bpos=[len(p_zh)+i-1 for i in groups["ZH"] if len(p_zh)+i-1 >= 0]
                    moments["A"][("decoder",l)].add(states[l][apos]) if condition == "zh" and apos else None
                    moments["B"][("decoder",l)].add(states[l][bpos]) if condition == "zh" and bpos else None
        for l in dec:
            all_pos=[len(p_zh)+i-1 for i in groups["ALL"] if len(p_zh)+i-1 >= 0]
            cs_pos=[len(p_zh)+i-1 for i in groups["EN"] if len(p_zh)+i-1 >= 0]
            if all_pos: deltas["all"][("decoder",l)].add(st_en[l][all_pos]-st_zh[l][all_pos])
            if cs_pos: deltas["cs"][("decoder",l)].add(st_en[l][cs_pos]-st_zh[l][cs_pos])
        if number % 10 == 0: print(f"A6 Whisper {source} {number}/{len(rows)}", flush=True)
    return moments, deltas, {"encoder": enc, "decoder": dec}, bundle.revision, dim


def _load_ascend_rows(manifest_path, dataset_dir):
    from csasr.basis_a6.data import load_ascend_split
    manifest=json.loads(Path(manifest_path).read_text()); wanted=set(manifest["utterance_ids"])
    return [x for x in load_ascend_split(dataset_dir, "train") if x.utterance_id in wanted]


def _ascend_spans(rows):
    # The alignment is computed in the same job with the canonical CTC adapter.
    from csasr.data.ctc_alignment import align_units, load_ctc_aligner
    from csasr.utils.config import load_config
    align_cfg=load_config("lss/align.yaml"); align_cfg["model"]={"device":"cuda"}
    aligner=load_ctc_aligner(align_cfg); out={}
    for i,row in enumerate(rows,1):
        audio,sr=_decode_bytes(row.audio); raw=align_units(aligner,audio,[u["surface"] for u in row.units],sr)
        by={int(x["unit_id"]):x for x in raw}; spans=[]
        for j,u in enumerate(row.units):
            if u["language_tag"] in {"EN","ZH"} and j in by:
                spans.append({"unit_index":j,"language_tag":u["language_tag"],"start_sec":by[j]["start_sec"],"end_sec":by[j]["end_sec"]})
        if not any(x["language_tag"]=="EN" for x in spans) or not any(x["language_tag"]=="ZH" for x in spans): raise RuntimeError(f"ASCEND alignment not mixed: {row.utterance_id}")
        out[row.utterance_id]=spans
        if i%10==0: print(f"A6 CTC alignment {i}/{len(rows)}",flush=True)
    del aligner
    import torch; torch.cuda.empty_cache()
    return out


def _qwen_pass(bundle, rows, spans_by_uid, source, *, need_ab, cs_only=False, tmpdir=None):
    import torch
    from csasr.data.normalize import normalize_and_segment
    from csasr.models.qwen3_asr import inputs_for_audio
    from csasr.lss.qwen_sites import QwenAudioSiteRecorder, QwenTextSiteRecorder
    from experiments.basis_a4_qwen import _content_start
    enc, dec = QWEN_ENC, QWEN_DEC
    dim_e, dim_d = bundle.encoder_dim, bundle.decoder_dim
    moments={"A":{},"B":{}}; deltas={"all":{},"cs":{}}
    for l in enc:
        moments["A"][("encoder",l)]=Moments(dim_e); moments["B"][("encoder",l)]=Moments(dim_e)
    for l in dec:
        moments["A"][("decoder",l)]=Moments(dim_d); moments["B"][("decoder",l)]=Moments(dim_d)
        deltas["all"][("decoder",l)]=DeltaMoments(dim_d); deltas["cs"][("decoder",l)]=DeltaMoments(dim_d)
    for number,row in enumerate(rows,1):
        uid=str(row["utterance_id"]); norm,units=normalize_and_segment(row["transcript_raw"])
        audio_path=_wav_path(row["audio_path"] if "audio_path" in row else row.audio,tmpdir,uid.replace("/","_"))
        inp=inputs_for_audio(bundle,audio_path,language="Chinese",transcript=norm)
        with torch.inference_mode(), QwenAudioSiteRecorder(bundle,enc) as ar, QwenTextSiteRecorder(bundle,dec) as dr:
            bundle.thinker_model(**inp,use_cache=False)
        ast={l:ar.states[l].numpy() for l in enc}; zst={l:dr.states[l].numpy()[0] for l in dec}
        if need_ab:
            fps=ast[0].shape[0]/max(float(row["duration_sec"]),1e-6)
            for l in enc:
                for sp in spans_by_uid[uid]:
                    lo=max(0,int(math.floor(sp["start_sec"]*fps))); hi=min(ast[l].shape[0],int(math.ceil(sp["end_sec"]*fps)))
                    if hi>lo: (moments["A"] if sp["language_tag"]=="EN" else moments["B"])[("encoder",l)].add(ast[l][lo:hi])
        _,_,ref_ids,_,_=__import__("experiments.basis_a4_qwen",fromlist=["_content_plan"])._content_plan(bundle,row,"Chinese")
        groups=_unit_token_groups(bundle.processor.tokenizer,norm,units,ref_ids)
        start=_content_start(inp["input_ids"],ref_ids)
        apos=[start+i-1 for i in groups["EN"] if start+i-1>=0]; bpos=[start+i-1 for i in groups["ZH"] if start+i-1>=0]
        if apos and bpos and need_ab:
            for l in dec: moments["A"][("decoder",l)].add(zst[l][apos]); moments["B"][("decoder",l)].add(zst[l][bpos])
        paired={}
        for lang in ("English","Chinese"):
            p=inputs_for_audio(bundle,audio_path,language=lang,transcript=norm)
            with torch.inference_mode(), QwenTextSiteRecorder(bundle,dec) as pr: bundle.thinker_model(**p,use_cache=False)
            if _content_start(p["input_ids"],ref_ids) != start: raise RuntimeError("Qwen paired prompt positions do not align")
            paired[lang]={l:pr.states[l].numpy()[0] for l in dec}
        allpos=[start+i-1 for i in groups["ALL"] if start+i-1>=0]; cspos=apos
        for l in dec:
            if allpos: deltas["all"][("decoder",l)].add(paired["English"][l][allpos]-paired["Chinese"][l][allpos])
            if cspos: deltas["cs"][("decoder",l)].add(paired["English"][l][cspos]-paired["Chinese"][l][cspos])
        if number%10==0: print(f"A6 Qwen {source} {number}/{len(rows)}",flush=True)
    return moments,deltas,{"encoder":enc,"decoder":dec},bundle.revision,dim_d


def _row_dict(item, tmpdir):
    return {"utterance_id":item.utterance_id,"audio_path":_wav_path(item.audio,tmpdir,item.utterance_id.replace("/","_")),
            "transcript_raw":item.transcript_raw,"duration_sec":item.duration_sec}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=("whisper","qwen3_asr_1p7b"),required=True); args=ap.parse_args()
    import torch
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    model=args.model
    with tempfile.TemporaryDirectory(prefix="basis_a6_construct_") as temp:
        tmp=Path(temp)
        ascend_rows=_load_ascend_rows(REPO/"results/basis_a6_expanded/ascend/ASCEND_CONSTRUCT_MANIFEST.json",REPO/"data/external/ASCEND/dataset")
        asc_spans=_ascend_spans(ascend_rows)
        if model=="whisper":
            from csasr.models.whisper import load_whisper
            from csasr.utils.config import load_config
            bundle=load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
            # CS Conditioning-CS is the only missing CS inventory; exact source population is loaded from the locked A4/A5 role files.
            from experiments.dg03_build_basis import _load_construct
            _,manifest,construct=_load_construct(); uids=set(str(x) for x in construct["utterance_id"])
            cs_rows=manifest[manifest["utterance_id"].astype(str).isin(uids)].to_dict("records")
            cs_spans={str(uid):g.to_dict("records") for uid,g in construct.groupby("utterance_id")}
            moments,deltas,layers,rev,dim=_whisper_pass(bundle,cs_rows,cs_spans,"cs_dialogue",need_ab=False,cond_cs_only=True,tmpdir=tmp)
            _save_source(model,"cs_dialogue",dim,moments,deltas,layers,cs_only=True,model_revision=rev)
            ac_rows=[_row_dict(x,tmp) for x in ascend_rows]
            moments,deltas,layers,rev,dim=_whisper_pass(bundle,ac_rows,asc_spans,"ascend",need_ab=True,tmpdir=tmp)
            _save_source(model,"ascend",dim,moments,deltas,layers,model_revision=rev)
        else:
            from csasr.models.qwen3_asr import load_qwen
            bundle=load_qwen(device="cuda:0"); bundle.model.eval()
            from experiments.dg03_build_basis import _load_construct
            _,manifest,construct=_load_construct(); uids=set(str(x) for x in construct["utterance_id"])
            cs_rows=manifest[manifest["utterance_id"].astype(str).isin(uids)].to_dict("records")
            moments,deltas,layers,rev,dim=_qwen_pass(bundle,cs_rows,{},"cs_dialogue",need_ab=False,cs_only=True,tmpdir=tmp)
            _save_source(model,"cs_dialogue",dim,moments,deltas,layers,cs_only=True,model_revision=rev)
            ac_rows=[_row_dict(x,tmp) for x in ascend_rows]
            moments,deltas,layers,rev,dim=_qwen_pass(bundle,ac_rows,asc_spans,"ascend",need_ab=True,tmpdir=tmp)
            _save_source(model,"ascend",dim,moments,deltas,layers,model_revision=rev)
    return 0


if __name__=="__main__": raise SystemExit(main())
