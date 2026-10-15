"""Prepare model-independent, paired audio-intervention inputs on CPU.

No speaker label is inferred from a model or a multiple-choice dataset answer.
All arrays and media stay in ignored data/. Existing annotations are preserved.
"""
import hashlib
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def extract(path, p):
    grid = p['source_start_seconds'] + np.arange(p['frames']) / p['fps']
    times, frames = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError('Missing source frame timestamp')
            times.append(float(frame.time))
            frames.append(cv2.resize(frame.to_ndarray(format='rgb24'), (p['width'], p['height']), interpolation=cv2.INTER_AREA))
            if frame.time > grid[-1] + .1:
                break
    if not times or times[0] > grid[0] or times[-1] < grid[-1]:
        raise ValueError('Insufficient source duration; padding is forbidden')
    indices = np.abs(np.asarray(times)[:, None] - grid[None, :]).argmin(axis=0)
    selected = np.stack([frames[i] for i in indices])
    sample_count = round(p['frames'] / p['fps'] * p['audio_rate'])
    chunks, expected_sample, rate = [], 0, None
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            if frame.time is None:
                raise ValueError('Missing audio timestamp')
            if rate is None:
                rate = frame.sample_rate
            if rate != frame.sample_rate or abs(round(float(frame.time) * rate) - expected_sample) > 1:
                raise ValueError('Audio gap/rate change; refuse silent repair')
            raw = frame.to_ndarray()
            if not frame.format.is_planar:
                raw = raw.reshape(-1, len(frame.layout.channels)).T
            if np.issubdtype(raw.dtype, np.integer):
                raw = raw.astype(np.float64) / max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
            raw = raw.mean(axis=0)
            chunks.append(raw)
            expected_sample += frame.samples
    if not chunks:
        raise ValueError('Missing source audio')
    gcd = math.gcd(rate, p['audio_rate'])
    audio = resample_poly(np.concatenate(chunks), p['audio_rate'] // gcd, rate // gcd).astype(np.float32)
    start = round(grid[0] * p['audio_rate'])
    if len(audio[start:start + sample_count]) != sample_count:
        raise ValueError('Insufficient audio; padding is forbidden')
    return selected, audio[start:start + sample_count], np.asarray(times)[indices], grid


def main():
    p = json.loads((ROOT / 'causal_protocol.json').read_text())
    sources = json.loads((ROOT / p['source_manifest']).read_text())['samples']
    dest = ROOT / 'data' / 'causal'
    dest.mkdir(parents=True, exist_ok=True)
    extracted = {}
    for source in sources:
        key = source['sample_id']
        path = ROOT / 'data' / 'raw' / (key + '.mp4')
        extracted[key] = extract(path, p)
    records = []
    for i, source in enumerate(sources):
        key = source['sample_id']
        reference, audio, source_times, grid = extracted[key]
        donor_id = sources[(i + 1) % len(sources)]['sample_id']
        donor = extracted[donor_id][1]
        for level, settings in p['degradations'].items():
            degraded = []
            for frame in reference:
                image = cv2.GaussianBlur(frame, (0, 0), settings['blur_sigma'])
                image = cv2.resize(image, (p['width'] // settings['downsample'], p['height'] // settings['downsample']), interpolation=cv2.INTER_AREA)
                degraded.append(cv2.resize(image, (p['width'], p['height']), interpolation=cv2.INTER_CUBIC))
            degraded = np.stack(degraded)
            name = key + '__' + level
            path = dest / (name + '.npz')
            np.savez_compressed(path, reference=reference, degraded=degraded, original_audio=audio,
                conditioning_matching=audio, conditioning_swapped=donor,
                source_timestamps=source_times, target_timestamps=grid)
            records.append(dict(id=name, sample_id=key, level=level, path=path.name,
                bundle_sha256=sha(path), source_sha256=sha(ROOT / 'data' / 'raw' / (key + '.mp4')),
                video_array_sha256=array_sha(degraded), original_audio_sha256=array_sha(audio),
                matching_audio_sha256=array_sha(audio), swapped_audio_sha256=array_sha(donor), donor_id=donor_id))
        annotations = dest / (key + '__annotations.json')
        if not annotations.exists():
            annotations.write_text(json.dumps(dict(sample_id=key, status='PROVISIONAL', reviewer=None,
                source_sha256=sha(ROOT / 'data' / 'raw' / (key + '.mp4')), protocol_sha256=canonical_sha(p),
                instructions='Annotate tracked face boxes and speaking state from original AV, independently of restoration/detector outputs. Unknown is null. Do not infer labels from the dataset multiple-choice answer.',
                frames=[dict(index=j, source_time_seconds=float(t), faces=[]) for j, t in enumerate(source_times)],
                face_schema=dict(track_id='stable identity within clip', box_xyxy='four pixel coordinates at 512x288', speaking='0, 1, or null')), indent=2) + '\n')
        else:
            annotation = json.loads(annotations.read_text())
            if annotation.get('status') == 'PROVISIONAL' and not any(f['faces'] for f in annotation['frames']):
                annotation['source_sha256'] = sha(ROOT / 'data' / 'raw' / (key + '.mp4'))
                annotation['protocol_sha256'] = canonical_sha(p)
                annotations.write_text(json.dumps(annotation, indent=2) + '\n')
    report = dict(status='INPUTS_PREPARED_NOT_RESTORED', protocol=p, protocol_sha256=canonical_sha(p),
        records=records, gpu_runs=0, speaking_labels_verified=False,
        paired_jobs=len(records) * (len(p['conditions']) * len(p['seeds']) + 1))
    (dest / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    # Commit only provenance/aggregate preparation metadata, never media.
    (ROOT / 'results' / 'causal_preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], inputs=len(records), scheduled_jobs=report['paired_jobs'], labels='PROVISIONAL'), indent=2))


if __name__ == '__main__':
    main()
