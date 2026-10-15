"""Framewise inference callback using the unchanged official Ear networks."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def degrade(frame, resolution):
    source = frame.astype(np.float64) / 255
    span = np.ptp(source)
    scaled = ((source - source.min()) * (255 / span if span else 1) + .5).clip(0, 255).astype(np.uint8)
    small = Image.fromarray(scaled).resize((resolution, resolution), Image.Resampling.BICUBIC)
    if resolution != 8:
        small = small.resize((8, 8), Image.Resampling.BICUBIC)
    return np.asarray(small), np.asarray(small.resize((128, 128), Image.Resampling.BICUBIC))


def run(session, im, au, visual, audio, fused, latent, rendered, wav_path, prepared_audio):
    directory = ROOT / 'data' / 'unitalk'
    manifest = json.loads((directory / 'manifest.json').read_text())
    p = manifest['protocol']
    if p != json.loads((ROOT / 'unitalk_cpu_protocol.json').read_text()):
        raise ValueError('Protocol changed after preparing inputs')
    out = directory / 'outputs'
    out.mkdir(exist_ok=True)
    weights = sha(ROOT / 'external' / 'ear_for_face_super_resolution' / 'checkpoint' / 'model-0.data-00000-of-00001')
    if weights != 'fc99bed110e02f0507fa84c6043f759def33d681f22aef4dcdc14a3e1cc5bda5':
        raise ValueError('Unexpected official weights')
    embeddings, spectrograms = {}, {}
    for row in manifest['records']:
        path = directory / row['path']
        if sha(path) != row['bundle_sha256']:
            raise ValueError('Input bundle changed')
        with np.load(path, allow_pickle=False) as data:
            wavfile.write(directory / (row['id'] + '.wav'), 16000,
                          np.clip(data['conditioning'] * 32768, -32768, 32767).astype(np.int16))
        spectrogram = session.run(prepared_audio, {wav_path: str(directory / (row['id'] + '.wav'))})
        if not np.isfinite(spectrogram).all():
            raise ValueError('Invalid source spectrogram')
        embeddings[row['id']] = session.run(audio, {au: spectrogram})
        spectrograms[row['id']] = spectrogram
    for row in manifest['records']:
        for resolution in p['effective_resolutions']:
            name = f"{row['id']}__r{resolution}"
            target, receipt = out / (name + '.npz'), out / (name + '.json')
            donor = next(r for r in manifest['records'] if r['id'] == row['donor_id'])
            identity = dict(id=name, protocol_sha256=manifest['protocol_sha256'], bundle_sha256=row['bundle_sha256'],
                donor_bundle_sha256=donor['bundle_sha256'], weights_sha256=weights, resolution=resolution)
            if target.exists() or receipt.exists():
                if not (target.exists() and receipt.exists()):
                    raise ValueError('Incomplete previous output; inspect before retrying')
                prior = json.loads(receipt.read_text())
                if any(prior.get(k) != v for k, v in identity.items()) or sha(target) != prior['output_sha256']:
                    raise ValueError('Stale output')
                print(name, 'verified cached output', flush=True)
                continue
            with np.load(directory / row['path'], allow_pickle=False) as data:
                reference = data['reference']
            images = {key: [] for key in ('degraded', 'small', 'visual_only', 'matching', 'swapped')}
            started = time.perf_counter()
            repeat = None
            latent_codes = {key: [] for key in ('visual_only', 'matching', 'swapped')}
            # Group encoder/fuser work before synthesis to avoid repeatedly
            # paging large, disjoint weight blocks on an 8 GB Windows laptop.
            # The graph operations and values are identical; no batching/math
            # or condition/decision changes are introduced by this scheduling.
            for index, frame in enumerate(reference):
                small, degraded = degrade(frame, resolution)
                images['small'].append(small)
                images['degraded'].append(degraded)
                v = session.run(visual, {im: small.astype(np.float32)[None] / 127.5 - 1})
                for condition in ('visual_only', 'matching', 'swapped'):
                    embedding = embeddings[row['donor_id'] if condition == 'swapped' else row['id']]
                    z = v if condition == 'visual_only' else session.run(fused, {visual: v, audio: embedding})
                    latent_codes[condition].append(z)
            print(name, 'latents ready', round(time.perf_counter() - started, 1), 's', flush=True)
            for index in range(len(reference)):
                for condition in ('visual_only', 'matching', 'swapped'):
                    z = latent_codes[condition][index]
                    values = session.run(rendered, {latent: z})[0]
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite restoration')
                    result = (values * 255).round().astype(np.uint8)
                    images[condition].append(result)
                    if condition == 'matching' and index == 0:
                        z2 = session.run(fused, {im: images['small'][0].astype(np.float32)[None] / 127.5 - 1,
                                                 au: spectrograms[row['id']]})
                        again = (session.run(rendered, {latent: z2})[0] * 255).round().astype(np.uint8)
                        repeat = bool(np.array_equal(again, result))
                if (index + 1) % 5 == 0:
                    print(name, index + 1, '/', len(reference), 'frames', round(time.perf_counter() - started, 1), 's', flush=True)
            if not repeat:
                raise ValueError('Identical-input repeat failed')
            np.savez_compressed(target, **{key: np.stack(value) for key, value in images.items()})
            record = dict(identity, output_sha256=sha(target), repeat_equal=repeat, frame_count=len(reference),
                execution_schedule='encode_then_synthesize_per_clip', runner_sha256=sha(Path(__file__)),
                device='CPU', forward_passes=3 * len(reference) + 1, wall_seconds=time.perf_counter() - started,
                original_downstream_audio_sha256=row['original_audio_sha256'],
                source_audio_sha256=row['conditioning_sha256'], donor_audio_sha256=donor['conditioning_sha256'])
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            print(name, 'DONE', round(record['wall_seconds'], 1), 's', flush=True)
