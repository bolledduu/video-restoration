"""Select published-label face tracks without inspecting restoration outcomes."""
import csv
import io
import json
import math
import zipfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly
from prepare_causal import ROOT, sha, canonical_sha, array_sha


def get(url, **kwargs):
    for attempt in range(5):
        response = requests.get(url, timeout=120, **kwargs)
        if response.status_code not in (429, 502, 503, 504):
            response.raise_for_status()
            return response
        response.close()
        time.sleep(min(15 * (attempt + 1), 60))
    response.raise_for_status()


def fetch(url, path):
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.partial')
        with get(url, stream=True) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        tmp.replace(path)
    return path


def main():
    p = json.loads((ROOT / 'unitalk_cpu_protocol.json').read_text())
    out = ROOT / 'data' / 'unitalk'
    out.mkdir(exist_ok=True)
    base = f"https://huggingface.co/datasets/{p['dataset']}/resolve/{p['revision']}/"
    listing = out / 'archive_listing.json'
    if listing.exists() and listing.stat().st_size > 10:
        entries = json.loads(listing.read_text(encoding='utf-8-sig'))
    else:
        entries = get(f"https://huggingface.co/api/datasets/{p['dataset']}/tree/{p['revision']}/clips_videos/val?limit=1000").json()
        listing.write_text(json.dumps(entries))
    archives = sorted((x for x in entries if x['path'].endswith('.zip')), key=lambda x: (x['size'], x['path']))
    records, audit = [], []
    for archive in archives:
        vid = Path(archive['path']).stem
        print('Checking input eligibility', vid, flush=True)
        meta = fetch(base + f'csv/val/{vid}.csv', out / (vid + '.csv'))
        groups = defaultdict(list)
        for row in csv.reader(meta.read_text().splitlines()):
            groups[row[7]].append(row)
        candidates = {0: [], 1: []}
        for entity, rows in sorted(groups.items()):
            rows.sort(key=lambda row: float(row[1]))
            for offset in range(len(rows) - p['context_frames'] + 1):
                context = rows[offset:offset + p['context_frames']]
                times = np.array([float(row[1]) for row in context])
                if not np.allclose(np.diff(times), 1 / p['fps'], atol=1e-4):
                    continue
                target = context[p['target_offset']:p['target_offset'] + p['frames']]
                labels = [int(row[8]) for row in target]
                if len(set(labels)) == 1 and labels[0] in (0, 1):
                    candidates[labels[0]].append((float(target[0][1]), entity, context, target, float(rows[0][1])))
        if not all(candidates.values()):
            audit.append(dict(video=vid, status='NO_BALANCED_ELIGIBLE_LABEL_WINDOWS'))
            continue
        video_path = fetch(base + archive['path'], out / (vid + '_video.zip'))
        if video_path.stat().st_size != archive['size']:
            raise ValueError('Archive size mismatch')
        chosen = []
        with zipfile.ZipFile(video_path) as z:
            names = { (name.split('/')[-2], round(float(Path(name).stem), 4)): name
                      for name in z.namelist() if name.endswith('.jpg') }
            for label in (0, 1):
                for _, entity, context, target, first_time in sorted(candidates[label], key=lambda x: (x[0], x[1])):
                    images = []
                    for row in target:
                        key = (entity, round(float(row[1]), 4))
                        if key not in names:
                            break
                        image = Image.open(io.BytesIO(z.read(names[key]))).convert('RGB')
                        if min(image.size) < p['minimum_crop_side']:
                            break
                        images.append(np.asarray(image.resize((128, 128), Image.Resampling.BICUBIC)))
                    if len(images) == p['frames']:
                        chosen.append(dict(entity=entity, context=context, target=target,
                            first_time=first_time, label=label, reference=np.stack(images)))
                        break
        if len(chosen) != 2:
            audit.append(dict(video=vid, status='INSUFFICIENT_NATIVE_CROP_RESOLUTION_OR_MISSING_IMAGES'))
            continue
        audio_path = fetch(base + f'clips_audios/val/{vid}.zip', out / (vid + '_audio.zip'))
        pending = []
        with zipfile.ZipFile(audio_path) as z:
            for item in chosen:
                name = next(n for n in z.namelist() if Path(n).stem == item['entity'] and n.endswith('.wav'))
                rate, pcm = wavfile.read(io.BytesIO(z.read(name)))
                if pcm.dtype != np.int16 or pcm.ndim != 1:
                    raise ValueError(f'Unexpected source audio format: {rate}, {pcm.shape}, {pcm.dtype}')
                audio = pcm.astype(np.float32) / 32768
                if rate != 16000:
                    divisor = math.gcd(rate, 16000)
                    audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
                start = round((float(item['context'][0][1]) - item['first_time']) * 16000)
                context_audio = audio[start:start + 64000]
                if len(context_audio) != 64000:
                    raise ValueError('Selected audio is incomplete; no padding allowed')
                index = p['target_offset'] * 640
                target_audio = context_audio[index:index + 16000].copy()
                sid = f"{vid}__label{item['label']}"
                target = out / (sid + '.npz')
                np.savez_compressed(target, reference=item['reference'], conditioning=context_audio,
                    original_audio=target_audio, labels=np.full(p['frames'], item['label'], dtype=np.int64))
                pending.append(dict(id=sid, source=vid, entity=item['entity'], label=item['label'],
                    target_start=float(item['target'][0][1]), context_start=float(item['context'][0][1]),
                    track_audio_origin=item['first_time'], path=target.name, bundle_sha256=sha(target),
                    csv_sha256=sha(meta), video_zip_sha256=sha(video_path), audio_zip_sha256=sha(audio_path),
                    original_audio_sha256=array_sha(target_audio), conditioning_sha256=array_sha(context_audio),
                    annotation_provenance='Published UniTalk validation labels; not model-generated or independently re-annotated here'))
        records.extend(pending)
        audit.append(dict(video=vid, status='SELECTED'))
        print('Selected', vid, [(x['entity'], x['label'], x['target_start']) for x in pending], flush=True)
        if len(records) == 2 * p['sources']:
            break
    if len(records) != 2 * p['sources']:
        raise ValueError('Not enough eligible sources')
    sources = list(dict.fromkeys(r['source'] for r in records))
    for row in records:
        donor_source = sources[(sources.index(row['source']) + 1) % len(sources)]
        row['donor_id'] = next(r['id'] for r in records if r['source'] == donor_source and r['label'] == 1)
    manifest = dict(protocol=p, protocol_sha256=canonical_sha(p), records=records, selection_audit=audit)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'results' / 'unitalk_cpu_inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared', len(records), 'labeled clips', flush=True)


if __name__ == '__main__':
    main()
