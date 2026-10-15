"""CPU evaluation of published OmniVR outputs; NOT local OmniVR inference."""
import hashlib
import json
import time
import subprocess
from pathlib import Path
import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from cpu_media import ROOT, audit
from video_baseline import FastDVDnet, denoise_seq_fastdvdnet, measure, save_video
from talknet_cpu import talkNetModel, audio_segment, python_speech_features


def read_video(path):
    images, times = [], []
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 2
        for frame in container.decode(video=0):
            images.append(cv2.resize(frame.to_ndarray(format='rgb24'), (256, 144), interpolation=cv2.INTER_AREA))
            times.append(float(frame.time))
    return np.stack(images), np.asarray(times)


def load_asd():
    checkpoint = ROOT / 'external/TalkNet-ASD/pretrain_TalkSet.model'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = talkNetModel().cpu().eval()
    model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items() if k.startswith('model.')}, strict=True)
    heads = {}
    for name, size in [('V', 128), ('A', 128), ('AV', 256)]:
        head = torch.nn.Linear(size, 2).eval()
        prefix = 'loss' + name + '.FC.'
        head.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        heads[name] = head
    return model, heads, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cv2.setNumThreads(2)
    metadata = json.loads((ROOT / 'results/omnivr_sources.json').read_text())
    output_dir = ROOT / 'data/omnivr/evaluation'
    output_dir.mkdir(exist_ok=True)
    denoiser = FastDVDnet().cpu().eval()
    weight = ROOT / 'external/fastdvdnet/model.pth'
    state = torch.load(weight, map_location='cpu', weights_only=True)
    denoiser.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model, heads, checkpoint_hash = load_asd()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    results = []
    for name in sorted({r['sample']['name'] for r in metadata['records']}):
        print('Evaluating ' + name, flush=True)
        began = time.perf_counter()
        files = {r['condition']: ROOT / r['local_path'] for r in metadata['records'] if r['sample']['name'] == name}
        media = {condition: audit(path) for condition, path in files.items()}
        videos = {condition: read_video(path) for condition, path in files.items()}
        ref, times = videos['reference']
        for condition, (frames, timestamps) in videos.items():
            if len(timestamps) != len(times) or not np.allclose(timestamps, times, atol=.001, rtol=0):
                raise ValueError('Unaligned published output: ' + condition)
        indices = np.flatnonzero((times >= 1.0) & (times < 3.0))
        assert len(indices) >= 40
        arrays = {condition: frames[indices] for condition, (frames, _) in videos.items()}
        del videos, ref
        excerpt_times = times[indices]
        tensor = torch.from_numpy(arrays['degraded'].transpose(0, 3, 1, 2).copy()).float() / 255
        inference_start = time.perf_counter()
        with torch.inference_mode():
            restored = denoise_seq_fastdvdnet(tensor, torch.tensor([10 / 255]), 5, denoiser)
        fast_seconds = time.perf_counter() - inference_start
        print('CPU denoising complete: ' + str(round(fast_seconds, 2)) + ' seconds', flush=True)
        arrays['fastdvdnet'] = (restored.permute(0, 2, 3, 1).numpy().clip(0, 1) * 255).round().astype(np.uint8)
        quality = {condition: measure(arrays['reference'], value) for condition, value in arrays.items() if condition != 'reference'}
        n = int(np.floor((excerpt_times[-1] - excerpt_times[0] + np.median(np.diff(excerpt_times))) * 25))
        grid = excerpt_times[0] + np.arange(n) / 25
        chosen = np.abs(excerpt_times[:, None] - grid[None, :]).argmin(axis=0)
        boxes, counts = [], []
        for image in arrays['reference'][chosen]:
            large = cv2.resize(image, (512, 288))
            found = detector.detectMultiScale(cv2.cvtColor(large, cv2.COLOR_RGB2GRAY), 1.1, 3, minSize=(24, 24))
            counts.append(len(found))
            if len(found):
                x, y, w, h = max(found, key=lambda b: b[2] * b[3])
                cx, cy, side = (x + w / 2) / 2, (y + h / 2) / 2, max(w, h) * .8
                box = [max(0, int(cx - side / 2)), max(0, int(cy - side / 2)), min(256, int(cx + side / 2)), min(144, int(cy + side / 2))]
            elif boxes:
                box = boxes[-1]
            else:
                raise ValueError('No initial face: do not invent a target')
            boxes.append(box)
        audio = audio_segment(files['reference'], float(excerpt_times[0]), n / 25)
        pcm = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        features = python_speech_features.mfcc(pcm, 16000, numcep=13, winlen=.025, winstep=.010)
        n = min(n, len(features) // 4)
        ta = torch.from_numpy(features[:4 * n].copy()).float().unsqueeze(0)
        predictions = {}
        for condition, frames in arrays.items():
            crops = []
            for image, (x1, y1, x2, y2) in zip(frames[chosen[:n]], boxes[:n]):
                crops.append(cv2.resize(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY), (112, 112)))
            tv = torch.from_numpy(np.stack(crops)).float().unsqueeze(0)
            with torch.inference_mode():
                ve, ae = model.forward_visual_frontend(tv), model.forward_audio_frontend(ta)
                ca, cv = model.forward_cross_attention(ae, ve)
                logits = {'visual_only': heads['V'](model.forward_visual_backend(ve)).numpy(),
                          'audio_only': heads['A'](model.forward_audio_backend(ae)).numpy(),
                          'fixed_audio_visual': heads['AV'](model.forward_audio_visual_backend(ca, cv)).numpy()}
            assert all(np.isfinite(x).all() for x in logits.values())
            predictions[condition] = {k: v.tolist() for k, v in logits.items()}
        assert all(predictions[c]['audio_only'] == predictions['reference']['audio_only'] for c in predictions)
        changes = {}
        for head in ('visual_only', 'audio_only', 'fixed_audio_visual'):
            labels = {c: np.argmax(p[head], axis=1) for c, p in predictions.items()}
            changes[head] = {'class1_frames': {c: int(x.sum()) for c, x in labels.items()},
                             'disagreements_vs_degraded': {c: int((x != labels['degraded']).sum()) for c, x in labels.items()},
                             'omnivr_vs_fastdvdnet': int((labels['omnivr_author_output'] != labels['fastdvdnet']).sum())}
        stem = Path(name).stem
        canvas = Image.new('RGB', (4 * 256, 3 * 164), 'white')
        draw = ImageDraw.Draw(canvas)
        for col, (condition, frames) in enumerate(arrays.items()):
            for row, index in enumerate([0, len(frames) // 2, len(frames) - 1]):
                draw.text((col * 256 + 3, row * 164 + 3), condition + f' {excerpt_times[index]:.2f}s', fill='black')
                canvas.paste(Image.fromarray(frames[index]), (col * 256, row * 164 + 20))
            save_video(output_dir / (stem + '__' + condition + '.mp4'), frames, 24)
        canvas.save(output_dir / (stem + '.png'))
        row = dict(sample=name, restoration_provenance='OmniVR step02400 neg_cfg published author output; no local OmniVR inference',
                   excerpt_seconds=[1, 3], quality_resolution=[256, 144], media_audit=media,
                   quality_frames=len(indices), quality=quality, asd_frames=n,
                   asd_timestamps=grid[:n].tolist(), reference_crop_boxes=boxes[:n], reference_face_detection_counts=counts[:n],
                   crop_policy='Largest frontal reference face per frame; previous crop on misses; not verified identity tracking',
                   downstream_audio='Clean reference audio fixed for every condition; restored audio never used',
                   downstream_audio_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                   talknet_checkpoint_sha256=checkpoint_hash, fastdvdnet_checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                   code_commits={x: subprocess.check_output(['git', '-C', str(ROOT / 'external' / x), 'rev-parse', 'HEAD'], text=True).strip() for x in ('fastdvdnet', 'TalkNet-ASD')},
                   logits=predictions, prediction_changes=changes,
                   speaker_ground_truth_verified=False, accuracy=None, delta_false_attribution=None,
                   limitation='Resized short excerpts; uncalibrated logits; no audio-swap intervention; FastDVDnet is task-mismatched and not a matched ablation',
                   local_fastdvdnet_seconds=fast_seconds, total_seconds=time.perf_counter() - began)
        results.append(row)
        (ROOT / 'results/omnivr_cpu_evaluation.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(sample=name, quality=quality, changes=changes, seconds=row['total_seconds'])), flush=True)


if __name__ == '__main__':
    main()
