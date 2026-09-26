"""Qwen3-ASR-1.7B loading and frozen input/generation helpers for A4."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def model_revision(path: str | Path) -> str:
    """Content-address the complete pinned local official model directory."""
    root = Path(path)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".locks" not in p.parts:
            files[str(p.relative_to(root))] = file_sha256(p)
    return "sha256:" + hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class QwenResolvedModules:
    thinker: Any
    audio_tower: Any
    text_model: Any
    audio_layers: tuple[Any, ...]
    text_layers: tuple[Any, ...]
    encoder_dim: int
    decoder_dim: int


def resolve_qwen_modules(model: Any, *, encoder_layers: int = 24,
                         decoder_layers: int = 28) -> QwenResolvedModules:
    """Resolve the pinned model's two Transformer stacks by module structure.

    The resolver intentionally does not assume a wrapper alias.  It searches
    ``named_modules()`` for unique ModuleLists whose layer members expose the
    frozen pre-FFN/pre-MLP normalization modules, then verifies dimensions and
    identities before returning them.
    """
    candidates = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            continue
        members = tuple(layers)
        if not members:
            continue
        if all(hasattr(x, "final_layer_norm") for x in members):
            dim = int(members[0].final_layer_norm.weight.numel())
            candidates.append(("audio", name, module, members, dim))
        if all(hasattr(x, "post_attention_layernorm") for x in members):
            dim = int(members[0].post_attention_layernorm.weight.numel())
            candidates.append(("text", name, module, members, dim))
    aud = [x for x in candidates if x[0] == "audio" and len(x[3]) == encoder_layers]
    txt = [x for x in candidates if x[0] == "text" and len(x[3]) == decoder_layers]
    if len(aud) != 1 or len(txt) != 1:
        raise RuntimeError(f"Qwen stack resolution expected one {encoder_layers}/{decoder_layers} pair; "
                           f"found audio={[(x[1], len(x[3])) for x in aud]} "
                           f"text={[(x[1], len(x[3])) for x in txt]}")
    audio, text = aud[0], txt[0]
    if set(map(id, audio[3])) & set(map(id, text[3])):
        raise RuntimeError("Qwen audio/text layer identities overlap")
    thinkers = [(name, module) for name, module in model.named_modules()
                if hasattr(module, "audio_tower") and hasattr(module, "model")]
    if len(thinkers) != 1:
        raise RuntimeError(f"Qwen thinker container resolution was not unique: {[x[0] for x in thinkers]}")
    return QwenResolvedModules(
        thinker=thinkers[0][1], audio_tower=audio[2], text_model=text[2],
        audio_layers=audio[3], text_layers=text[3],
        encoder_dim=audio[4], decoder_dim=text[4])


@dataclass
class QwenBundle:
    wrapper: Any
    model: Any
    processor: Any
    device: torch.device
    dtype: torch.dtype
    model_id: str
    model_path: str
    revision: str
    thinker_model: Any
    audio_tower: Any
    text_model: Any
    audio_layers: tuple[Any, ...]
    text_layers: tuple[Any, ...]
    num_encoder_layers: int
    num_decoder_layers: int
    encoder_dim: int
    decoder_dim: int

    @property
    def thinker(self):
        # Compatibility accessor for already-written callers; new resolver
        # paths use thinker_model/text_model/audio_layers directly.
        return self.thinker_model

    def audio_layer(self, index: int):
        return self.audio_layers[int(index)]

    def text_layer(self, index: int):
        return self.text_layers[int(index)]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "model_path": self.model_path,
            "model_revision": self.revision, "dtype": str(self.dtype),
            "device": str(self.device), "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "encoder_dim": self.encoder_dim, "decoder_dim": self.decoder_dim,
            "backend": "official qwen-asr transformers backend",
        }


def load_qwen(model_path: str = "/mnt/data/tungnx/Qwen3-ASR-1.7B", *,
              device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
              max_new_tokens: int = 200) -> QwenBundle:
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("Qwen A4 requires CUDA; refusing CPU scientific execution")
    from qwen_asr import Qwen3ASRModel

    wrapper = Qwen3ASRModel.from_pretrained(
        model_path, dtype=dtype, device_map=device, local_files_only=True,
        attn_implementation="eager", max_new_tokens=max_new_tokens,
        max_inference_batch_size=1,
    )
    model = wrapper.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resolved = resolve_qwen_modules(model)
    return QwenBundle(
        wrapper=wrapper, model=model, processor=wrapper.processor,
        device=torch.device(device), dtype=dtype,
        model_id="Qwen/Qwen3-ASR-1.7B", model_path=str(model_path),
        revision=model_revision(model_path),
        thinker_model=resolved.thinker, audio_tower=resolved.audio_tower,
        text_model=resolved.text_model, audio_layers=resolved.audio_layers,
        text_layers=resolved.text_layers,
        num_encoder_layers=len(resolved.audio_layers),
        num_decoder_layers=len(resolved.text_layers),
        encoder_dim=resolved.encoder_dim, decoder_dim=resolved.decoder_dim,
    )


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sr) != sample_rate:
        import scipy.signal as sps
        audio = sps.resample(audio, int(round(len(audio) * sample_rate / int(sr))))
    return np.asarray(audio, dtype=np.float32)


def text_prompt(bundle: QwenBundle, language: str, transcript: str = "") -> str:
    """Return the official chat-template prompt plus the frozen language tag."""
    messages = bundle.wrapper._build_messages(context="", audio_payload="")
    base = bundle.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    return base + f"language {language}<asr_text>" + transcript


def inputs_for_audio(bundle: QwenBundle, audio_path: str, *, language: str,
                     transcript: str = "") -> dict[str, torch.Tensor]:
    inputs = bundle.processor(
        text=[text_prompt(bundle, language, transcript)],
        audio=[load_audio(audio_path)], return_tensors="pt", padding=True)
    result = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif value.is_floating_point():
            result[key] = value.to(device=bundle.device, dtype=bundle.dtype)
        else:
            result[key] = value.to(device=bundle.device)
    return result


@torch.inference_mode()
def generate_one(bundle: QwenBundle, audio_path: str, *, language: str = "Chinese",
                 hook=None, cached_encoder=None) -> tuple[str, dict[str, Any]]:
    inputs = inputs_for_audio(bundle, audio_path, language=language)
    context = hook if hook is not None else _null_context()
    with context:
        if cached_encoder is None:
            out = bundle.model.generate(
                **inputs, max_new_tokens=200, do_sample=False, num_beams=1)
        else:
            # Qwen has no cross-attention encoder output.  Its reusable
            # decoder-only cache is the frozen audio feature sequence injected
            # into the text embeddings.  Replacing the placeholders here keeps
            # the audio tower outside every decoder-only intervention run.
            embeds = bundle.thinker_model.get_input_embeddings()(inputs["input_ids"])
            mask = bundle.thinker_model.get_placeholder_mask(inputs["input_ids"], embeds)
            embeds = embeds.masked_scatter(
                mask, cached_encoder.to(device=embeds.device, dtype=embeds.dtype))
            decoder_inputs = {"inputs_embeds": embeds,
                              "attention_mask": inputs.get("attention_mask")}
            out = bundle.thinker_model.generate(
                **decoder_inputs, max_new_tokens=200, do_sample=False, num_beams=1,
                return_dict_in_generate=True)
    seq = out.sequences if hasattr(out, "sequences") else out
    generated = seq[:, inputs["input_ids"].shape[1]:] if seq.shape[1] >= inputs["input_ids"].shape[1] else seq
    raw = bundle.processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # The frozen policy forces the language, so the official package treats
    # the decoded continuation as text-only.  The processor does not expose
    # the inference wrapper's extract_transcription method.
    text = raw
    return str(text).strip(), {"input_length": int(inputs["input_ids"].shape[1]),
                               "output_length": int(seq.shape[1])}


class _null_context:
    def __enter__(self): return self
    def __exit__(self, *_exc): return False
