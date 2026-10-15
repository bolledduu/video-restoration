"""Run pinned OmniVR on prepared arrays; defaults to a zero-inference dry run.

Use --execute only on the target GPU machine. Generated sound is deliberately
not exported to the downstream evaluator. Every output carries its input hashes.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from prepare_causal import ROOT, array_sha, canonical_sha, sha


def jobs(manifest):
    p = manifest['protocol']
    for row in manifest['records']:
        for seed in p['seeds']:
            for condition in p['conditions']:
                yield row, seed, condition
        if p['repeat_matching_seed_zero']:
            yield row, 0, 'matching_repeat'


def verify_bundle(directory, manifest):
    if canonical_sha(manifest['protocol']) != manifest['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    for row in manifest['records']:
        path = directory / row['path']
        if sha(path) != row['bundle_sha256']:
            raise ValueError(f'Input bundle changed: {path}')
        with np.load(path, allow_pickle=False) as data:
            for field, expected in [('degraded', 'video_array_sha256'), ('original_audio', 'original_audio_sha256'),
                                    ('conditioning_matching', 'matching_audio_sha256'), ('conditioning_swapped', 'swapped_audio_sha256')]:
                if array_sha(data[field]) != row[expected]:
                    raise ValueError(f'Array hash mismatch: {field}')
            if not np.array_equal(data['original_audio'], data['conditioning_matching']):
                raise ValueError('Matching audio differs from source')
            if data['conditioning_matching'].shape != data['conditioning_swapped'].shape:
                raise ValueError('Unequal audio lengths confound the intervention')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'causal')
    parser.add_argument('--upstream', type=Path, default=ROOT / 'external' / 'OminiVR')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--text-encoder-path', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='Optional smoke-run job cap; partial runs are never a complete pilot')
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    verify_bundle(directory, manifest)
    p = manifest['protocol']
    planned = list(jobs(manifest))
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.limit:
        planned = planned[:args.limit]
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN_ONLY', inputs_verified=True, jobs=len(planned),
            restoration_executed=False, annotations='PROVISIONAL unless independently reviewed',
            next='Provide pinned weights and an approved Gemma directory on a suitable GPU host; then use --execute.'), indent=2))
        return
    for label, path in [('base', args.model_path), ('adapter', args.checkpoint)]:
        if path is None or not path.is_file():
            parser.error(f'Missing {label} checkpoint')
    if not (args.text_encoder_path and args.text_encoder_path.is_dir()):
        parser.error('Provide the authorized Gemma model directory')
    revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != p['omnivr']['code_revision']:
        raise ValueError('Upstream revision differs from protocol')
    if subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Upstream checkout contains local changes; record a new protocol before using them')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available to this PyTorch runtime; no inference attempted')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Published inference requires BF16 support')
    source = args.upstream / 'packages' / 'omini-restore-trainer' / 'scripts' / 'infer.py'
    spec = importlib.util.spec_from_file_location('pinned_omnivr', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    config = SimpleNamespace(gpu=0, model_path=str(args.model_path.resolve()), checkpoint=str(args.checkpoint.resolve()),
        text_encoder_path=str(args.text_encoder_path.resolve()) if args.text_encoder_path else None,
        prompt_cache=None,
        lora_structure=None, with_audio=True, tiny_decoder=None,
        decode_tile_size=512, decode_tile_overlap=64, decode_temporal_tile_frames=64, decode_temporal_overlap_frames=24)
    weights = dict(base_sha256=sha(args.model_path), adapter_sha256=sha(args.checkpoint))
    assets = json.loads((ROOT / 'causal_assets.json').read_text())
    for role in ('base', 'adapter'):
        expected = next(r['sha256'] for r in assets['files'] if r['role'] == role)
        if weights[role + '_sha256'] != expected:
            raise ValueError(f'{role} checkpoint differs from pinned release')
    for record in assets['files']:
        if record['role'] != 'text_encoder':
            continue
        path = args.text_encoder_path / record['file']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing/incorrect Gemma file: {record["file"]}')
        if record['sha256']:
            actual, expected = sha(path), record['sha256']
        else:
            content = path.read_bytes()
            actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            expected = record['git_blob_sha1']
        if actual != expected:
            raise ValueError(f'Gemma file hash mismatch: {record["file"]}')
    output = directory / 'outputs'
    output.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    model = upstream.load_model(config)
    for row, seed, condition in planned:
        name = f"{row['id']}__seed{seed}__{condition}"
        target, receipt = output / (name + '.npz'), output / (name + '.json')
        identity = dict(input_id=row['id'], seed=seed, condition=condition,
            protocol_sha256=manifest['protocol_sha256'], bundle_sha256=row['bundle_sha256'], **weights)
        if target.exists() or receipt.exists():
            if not (target.exists() and receipt.exists()):
                raise ValueError(f'Incomplete previous output: {name}; inspect before removing/retrying')
            prior = json.loads(receipt.read_text())
            if any(prior.get(k) != v for k, v in identity.items()) or prior['output_sha256'] != sha(target):
                raise ValueError(f'Stale or changed result: {name}')
            continue
        with np.load(directory / row['path'], allow_pickle=False) as data:
            video = (torch.from_numpy(data['degraded'].transpose(3, 0, 1, 2).copy()).unsqueeze(0).float() / 127.5 - 1).to(torch.bfloat16)
            field = 'conditioning_swapped' if condition == 'swapped' else 'conditioning_matching'
            audio = None if condition == 'audio_disabled' else torch.from_numpy(data[field].copy()).view(1, 1, -1).expand(-1, 2, -1)
        # Upstream seed controls initial noise, but condition_noise uses the
        # global RNG. Reset BOTH before every paired run, including the repeat.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with torch.inference_mode():
            restored, generated_audio = upstream.restore_video(model, video, audio, p['audio_rate'], p['fps'],
                strategy=p['omnivr']['strategy'], guidance_scale=p['omnivr']['guidance_scale'],
                num_steps=p['omnivr']['num_steps'], condition_noise=p['omnivr']['condition_noise'], seed=seed)
        values = restored.detach().cpu().permute(1, 2, 3, 0).float().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Restoration contains nonfinite values')
        frames = (values * 255).round().clip(0, 255).astype(np.uint8)
        if frames.shape != (p['frames'], p['height'], p['width'], 3):
            raise ValueError(f'Output shape/timing mismatch: {frames.shape}')
        np.savez_compressed(target, restored=frames)
        record = dict(identity, status='RESTORATION_EXECUTED', output_sha256=sha(target),
            conditioning_audio_sha256=None if audio is None else row['swapped_audio_sha256' if condition == 'swapped' else 'matching_audio_sha256'],
            downstream_audio_sha256=row['original_audio_sha256'], generated_audio_used_for_evaluation=False,
            device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, hip=torch.version.hip,
            code_revision=revision, frame_count=len(frames), wall_seconds=time.perf_counter() - started)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        del restored, generated_audio
        print(name, 'completed', flush=True)


if __name__ == '__main__':
    main()
