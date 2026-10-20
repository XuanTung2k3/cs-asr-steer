"""A5000 engineering checks using the published implementation, never historical experiment entrypoints.

Private manifests, transcripts, predictions and checkpoints stay outside the repository.
Only official training text/audio is opened. Other splits are inventoried by ID only.
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def kv(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            k, v = line.split(maxsplit=1)
            if k in rows:
                raise ValueError(f'duplicate ID {k} in {path}')
            rows[k] = v
    return rows


def write(path, obj):
    """Atomic repeatable output; immutable run directories are selected by the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def provenance():
    import platform
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'python': platform.python_version(),
            'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
            'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                        [Path(__file__), ROOT/'configs/a5000/runtime.yaml', ROOT/'reft/adapters.py',
                         ROOT/'reft/auto_policy.py', ROOT/'reft/train_supervised.py',
                         ROOT/'src/csasr/models/whisper.py']}}


def inventory(C):
    d = Path(C['data']['root']).expanduser()
    ix = d/'data/index/short_wav'
    split = {s: kv(ix/s/'wav.scp') for s in ('train', 'dev', 'test')}
    from csasr.data.manifest import parse_utterance_id
    speakers = {s: {parse_utterance_id(k)[1] for k in v} for s, v in split.items()}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not (split[a].keys() & split[b].keys()), (a, b, 'ID overlap')
        assert not (speakers[a] & speakers[b]), (a, b, 'speaker overlap')
    rec = {**provenance(), 'dataset_revision': C['data']['revision'],
           'counts': {s: {'utterances': len(v), 'speakers': len(speakers[s])} for s, v in split.items()},
           'index_sha256': {s: sha(ix/s/'wav.scp') for s in split},
           'test_transcripts_read': False, 'test_audio_opened': False,
           'historical_roles_available': False, 'historical_train394_inner106_full500_available': False}
    return d, split, rec


def extract(C):
    d, split, rec = inventory(C)
    parts = sorted((d/'data/short_wav').glob('short_wav.tar.gz[0-9][0-9]'))
    assert len(parts) == 19, f'Expected 19 release parts, got {len(parts)}'
    wanted = {Path(v).name: k for k, v in split['train'].items()}
    dest = d/'data/short_wav/extracted'
    count = 0
    # Stream concatenated gzip parts without making a second 9.6 GB archive.
    cat = subprocess.Popen(['cat', *map(str, parts)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=cat.stdout, mode='r|gz') as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in wanted:
                    continue
                target = dest/split['train'][wanted[name]]
                assert target.resolve().is_relative_to(dest.resolve())
                if target.exists():
                    assert target.stat().st_size == member.size, f'existing audio size differs: {target}'
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    f = tar.extractfile(member)
                    with target.open('xb') as out:
                        import shutil
                        shutil.copyfileobj(f, out)
                count += 1
    finally:
        cat.stdout.close()
        rc = cat.wait()
    assert rc == 0 and count == len(wanted), (rc, count, len(wanted))
    rec.update(extracted_train_audio=count, extraction_root=str(dest),
               archive_parts={p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in parts})
    write(Path(C['output_root']).expanduser()/'data-release.json', rec)
    print(json.dumps({'extracted_train_audio': count, 'root': str(dest)}), flush=True)


def e0(C):
    d, split, rec = inventory(C)
    from csasr.data.manifest import build_manifest, dialogue_pairing_evidence
    cfg = {'data': {'index_root': str(d/'data/index/short_wav'),
                    'audio_root': str(d/'data/short_wav/extracted'),
                    'information_index': str(d/'data/index/total_information/Information_Index.txt'),
                    'sample_rate': 16000, 'splits': ['train'], 'audio_hash_bytes': 65536}}
    train = build_manifest(cfg, splits=['train'])
    assert len(train) == len(split['train'])
    assert set(train.official_split) == {'train'}
    assert train.duration_sec.notna().all() and (train.sample_rate == 16000).all()
    out = Path(C['output_root']).expanduser()/'e0'
    out.mkdir(parents=True, exist_ok=True)
    mp = out/'official_train.parquet'
    temp_manifest = out/'official_train.parquet.tmp'
    train.to_parquet(temp_manifest, index=False)
    temp_manifest.replace(mp)
    rec.update(training_audio_checked=len(train), hours=float(train.duration_sec.sum()/3600),
               train_dialogue_pairing=dialogue_pairing_evidence(train), manifest=str(mp),
               manifest_sha256=sha(mp), audio_hash_convention='sha256(str(size)+first 65536 bytes)',
               scope='Engineering inventory only; official train is not a reconstruction of historical research roles',
               objective_specification='Existing decoder CE uses content+EOS token mean; positions 0..2/padding masked. '
                                       'Encoder-output E1 and O2/O3 remain unimplemented.')
    write(out/'audit.json', rec)
    print(json.dumps({k: rec[k] for k in ['counts', 'training_audio_checked', 'hours', 'manifest']}), flush=True)


def gpu(C, model_only=False):
    import torch
    import transformers
    from csasr.models.whisper import load_whisper, batch_model_inputs
    from reft.provenance import weights_hash
    torch.manual_seed(42)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
    assert torch.cuda.is_available()
    p = torch.cuda.get_device_properties(0)
    assert 'A5000' in p.name
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    model_root = Path(C['model']['root']).expanduser()
    assert sha(model_root/'model.safetensors') == C['model']['sha256']
    out = Path(C['output_root']).expanduser()/'smoke'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    rec = {**provenance(), 'gpu': p.name, 'gpu_total_bytes': p.total_memory,
           'compute_capability': list(torch.cuda.get_device_capability()),
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
           'model_root': str(model_root), 'model_sha256': C['model']['sha256'],
           'model_revision': C['model']['revision'], 'scope': 'Bounded engineering smoke; not a historical experiment'}
    def memory():
        torch.cuda.synchronize()
        return {'allocated_peak_bytes': torch.cuda.max_memory_allocated(),
                'reserved_peak_bytes': torch.cuda.max_memory_reserved()}
    for dtype in [torch.float16, torch.bfloat16]:
        a = torch.ones(128, 128, dtype=dtype, device='cuda')
        assert float((a@a)[0, 0]) == 128
    b = load_whisper({'model': {'id': str(model_root), 'dtype': C['model']['dtype'], 'device': 'cuda',
                               'attn_implementation': 'eager', 'local_files_only': True, 'freeze_backbone': True}})
    assert b.d_model == 1280 and b.num_encoder_layers == 32 and b.num_decoder_layers == 32
    assert b.processor.feature_extractor.feature_size == 128
    assert b.dtype == torch.bfloat16 and next(b.model.parameters()).dtype == torch.bfloat16
    w0 = weights_hash(b.model)
    rec['g0'] = {'passed': True, 'weights_bf16_sha256': w0, **memory()}
    write(out/'report.json', rec)
    if model_only:
        print(json.dumps(rec), flush=True)
        return
    from reft.auto_policy import build_tf_batch, prompt_ids, tf_loss
    from reft.baseline_auto import generate
    from reft.adapters import LoReFTAdapter, site_hook_count
    from reft.train_supervised import batch_loss
    from reft.text import target_text
    from csasr.evaluation.mer import mer
    import pandas as pd
    manifest = Path(C['output_root']).expanduser()/'e0/official_train.parquet'
    rows = pd.read_parquet(manifest)
    permitted = rows[(rows.official_split == 'train') & rows.contains_code_switch &
                     (rows.duration_sec >= 2) & (rows.duration_sec <= 7)]
    assert len(permitted)
    row = permitted.sort_values('utterance_id').iloc[0]
    rec['manifest_sha256'] = sha(manifest)
    path = str(row.audio_path)
    # Published Auto beam-5 policy (200-token cap), no gold language choice.
    torch.cuda.reset_peak_memory_stats()
    pred, inp = generate(b, path, None, 5)
    assert pred['text'] and pred['terminated']
    replay, _ = generate(b, path, None, 5)
    assert pred['tokens'] == replay['tokens']
    rec['g1'] = {'passed': True, 'deterministic_repeat': True, 'decoding': 'Auto beam5 max_new_tokens200', **memory()}
    write(out/'predictions.json', {'utterance_id': row.utterance_id, 'official_split': 'train', **pred})
    eos = b.processor.tokenizer.eos_token_id
    targets = b.processor.tokenizer.encode(target_text(row.transcript_raw), add_special_tokens=False) + [eos]
    train_rows = [{'audio_path': path, 'prompt': pred['prompt'], 'target_ids': targets}]
    ad = LoReFTAdapter(1280, 2, 31, 42, 3, 'cuda')
    assert ad.n_trainable() == 5122 and not any(q.requires_grad for q in b.model.parameters())
    with torch.no_grad():
        enc = b.model.model.encoder(inp['input_features']).last_hidden_state
        X, Y = build_tf_batch([pred['prompt']], [targets], eos, eos)
        X, Y = X.to('cuda'), Y.to('cuda')
        plain = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
        with ad.active(b.model):
            ident = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
        assert torch.equal(plain, ident)
    del plain, ident, enc, inp
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(ad.parameters(), lr=5e-4, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0)
    before = ad.state_dict()
    loss, n, logits, _, _ = batch_loss(b, ad, train_rows, grad=True)
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    loss.backward()
    gradients = {name: float(q.grad.norm()) if q.grad is not None else None for name, q in ad.module.named_parameters()}
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in ad.parameters())
    assert gradients['learned.weight'] > 0 and gradients['learned.bias'] > 0
    assert all(q.grad is None for q in b.model.parameters())
    opt.step()
    assert any(not torch.equal(before[k], ad.state_dict()[k]) for k in before)
    # R's first-step gradient can be zero at exact no-op init. Verify the path after the legitimate optimizer step.
    opt.zero_grad(set_to_none=True)
    del logits
    loss2, _, logits2, _, _ = batch_loss(b, ad, train_rows, grad=True)
    assert torch.isfinite(loss2)
    loss2.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm() > 0 for q in ad.parameters())
    assert all(q.grad is None for q in b.model.parameters())
    train_memory = memory()
    opt.zero_grad(set_to_none=True)
    del logits2
    checkpoint = out/'adapter.pt'
    torch.save(ad.state_dict(), checkpoint)
    restored = LoReFTAdapter(1280, 2, 31, 42, 3, 'cuda')
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in ad.state_dict().items())
    with torch.no_grad():
        _, _, z1, _, _ = batch_loss(b, ad, train_rows, grad=False)
        _, _, z2, _, _ = batch_loss(b, restored, train_rows, grad=False)
        assert torch.equal(z1, z2)
        with ad.disabled():
            _, _, disabled, _, _ = batch_loss(b, ad, train_rows, grad=False)
        _, _, base, _, _ = batch_loss(b, None, train_rows, grad=False)
        assert torch.equal(disabled, base)
    assert site_hook_count(b.model, 31) == 0
    assert weights_hash(b.model) == w0
    rec['g2_available_decoder_reft'] = {'passed': True, 'site': 'decoder31 post-cross-attention pre-FFN',
        'rank': 2, 'trainable_parameters': 5122, 'microbatch': 1, 'optimizer_steps': 1,
        'second_backward_only': True, 'loss': float(loss.detach()), 'post_step_loss': float(loss2.detach()),
        'target_tokens': n, 'initial_gradients': gradients, 'checkpoint_sha256': sha(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'reload_logits_exact': True, 'disabled_identity_exact': True, 'backbone_weights_unchanged': True,
        'activation_checkpointing': False, 'gradient_accumulation': False, **train_memory}
    rec['g2_encoder_reft'] = {'passed': False, 'status': 'MISSING_IMPLEMENTATION'}
    write(out/'report.json', rec)
    # Training-side pipeline metric only; never a dev/test scientific result.
    score = mer(row.transcript_raw, pred['text'])
    score['ops'] = [asdict(op) for op in score['ops']]
    write(out/'training-side-mer.json', score)
    rec['g3'] = {'passed': True, 'scope': 'official training-side engineering pipeline', 'mer': score,
                 'rerun': 'E0 is atomic and repeatable; each GPU rerun creates a unique directory'}
    rec['seconds'] = time.monotonic()-t0
    write(out/'report.json', rec)
    print(json.dumps({'output': str(out), **rec}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract-train', 'e0', 'g0', 'smoke'])
    ap.add_argument('--config', default=str(ROOT/'configs/a5000/runtime.yaml'))
    args = ap.parse_args()
    C = yaml.safe_load(Path(args.config).read_text())
    if args.stage == 'extract-train':
        extract(C)
    elif args.stage == 'e0':
        e0(C)
    else:
        gpu(C, model_only=args.stage == 'g0')


if __name__ == '__main__':
    main()
